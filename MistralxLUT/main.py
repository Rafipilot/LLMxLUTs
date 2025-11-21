import torch
from torch import nn
from dataclasses import dataclass
from pathlib import Path
import fire
import json
from typing import Optional, Tuple, List
from sentencepiece import SentencePieceProcessor

import torch.nn.functional as F
from datetime import datetime


device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device : ", device)

@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    sliding_window: int
    norm_eps: float
    vocab_size: int

    max_batch_size: int = 0


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=2)
    values = torch.repeat_interleave(values, repeats=repeats, dim=2)
    return keys, values


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        freqs_cis.shape,
        (x.shape[1], x.shape[-1]),
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads
        
        self.repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = self.args.sliding_window

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * args.head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.n_heads * args.head_dim,
            args.dim,
            bias=False
        )
        self.cache_k = torch.empty(
            (
                args.max_batch_size,
                args.sliding_window,
                self.n_kv_heads,
                self.args.head_dim,
            ), dtype=torch.float16
        ).cuda()
        self.cache_v = torch.empty(
            (
                args.max_batch_size,
                args.sliding_window,
                self.n_kv_heads,
                self.args.head_dim,
            ), dtype=torch.float16
        ).cuda()

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.args.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # The cache is a rotating buffer
        scatter_pos = (positions[-self.sliding_window:] % self.sliding_window)[None, :, None, None]
        scatter_pos = scatter_pos.repeat(bsz, 1, self.n_kv_heads, self.args.head_dim)
        self.cache_k[:bsz].scatter_(dim=1, index=scatter_pos, src=xk[:, -self.sliding_window:])
        self.cache_v[:bsz].scatter_(dim=1, index=scatter_pos, src=xv[:, -self.sliding_window:])


        if positions.shape[0] > 1:
            # prefill
            key, value = repeat_kv(xk, xv, self.repeats)
        else:
            cur_pos = positions[-1].item() + 1
            key, value = repeat_kv(self.cache_k[:bsz, :cur_pos, ...], self.cache_v[:bsz, :cur_pos, ...], self.repeats)
            
        query = xq.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # scores : [bsz, n_heads, seqlen | 1, seqlen]
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scale
        
        if mask is not None:
            scores += mask[None, None, ...]

        scores = scores.float()
        scores = nn.functional.softmax(scores, dim=-1).type_as(query)
        output = torch.matmul(scores, value)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(
            args.dim,
            args.hidden_dim,
            bias=False
        )
        self.w2 = nn.Linear(
            args.hidden_dim,
            args.dim,
            bias=False
        )
        self.w3 = nn.Linear(
            args.dim,
            args.hidden_dim,
            bias=False
        )

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

class LUT():
    def __init__(self):
        self.lookupTable = [] # main lookup table
        self.lookupTableMetaData = [] # idx 0 calls since last response, idx 1 number of calls
        self.CS_threshold = 0.5
        self.cost_scale = 10

    def train(self, xs, ys):
        for x, y in zip(xs, ys):
            print("adding rows...")
            row = torch.stack([x, y])
            self.lookupTable.append(row)
            self.lookupTableMetaData.append([1000, 0])

    
    def forward(self, x):
        x = x[-1, -1, :]
        x = x.squeeze()
        closest_row_output = None
        if len(self.lookupTable) == 0:
            return None, None
        lookup_vecs = torch.stack([row[0] for row in self.lookupTable])
        outputs = torch.stack([row[1] for row in self.lookupTable])

        for i in range(len(self.lookupTableMetaData)):
            meta = self.lookupTableMetaData[i]
            meta[0] = meta[0] + 1
            self.lookupTableMetaData[i] = meta

        ### adding a punishment/ cost for looking up memories which have come up just before
        costs = []
        for row in self.lookupTableMetaData:
            number_of_look_up_since_last_hit = row[0]

            cost = (1/(number_of_look_up_since_last_hit+1))
            costs.append(cost)

        costs = torch.tensor(costs, device=device)
        sims = F.cosine_similarity(lookup_vecs, x.unsqueeze(0), dim=1)# - (1/10)*torch.norm(lookup_vecs- x.unsqueeze(0), dim=1)  # dim one since N, d
        sims = sims - self.cost_scale*costs
        
        max_sim_idx = torch.argmax(sims)
        highest_sim = sims[max_sim_idx].item()
        print("highest sim: ", highest_sim)


        closest_row_output = outputs[max_sim_idx]
        if highest_sim < self.CS_threshold:  # arbitrary threshold- This must be fixed as it is a very temp workaround. The best fix would be to somehow have an active threshold based 
            print("Low similarity, cs threshold: ", self.CS_threshold)
            return torch.zeros_like(closest_row_output), 0.0
        row_meta_data = self.lookupTableMetaData[max_sim_idx]
        self.lookupTableMetaData[max_sim_idx]= [0, row_meta_data[1]+1]
        # Note to self
        # output of forward could take into account more rows by adjusting the outputs in a sort of
        #  weighted average effected by the relative cosine distance 
        return closest_row_output, highest_sim
    
    def resetLUT(self):
        self.lookupTable = [] # main lookup table
        self.lookupTableMetaData = []

    def resetCosts(self):
        for row in self.lookupTableMetaData:
            row[0] = 1000

    def saveLUT(self, save_name):
        with open(save_name, "w") as f:
            save_lookup = self.lookupTable.numpy()
            json.dump(save_lookup, f, indent=2)

    def loadLUT(self, save_name):
        with open(save_name, "r") as f:
            self.lookupTable = json.load(f)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, wnn_block=False):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args
        self.LUT = LUT()
        self.wnn_block = wnn_block # we only activate the wnn block in the last n layers as specified by num_wnn_blocks
        if self.wnn_block:
            self.use_wnn = True
        else:
            self.use_wnn = False
        self.pre_wnn_x = None  
        self.residual_scale = 15

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, positions, mask)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        self.pre_wnn_x =  h + r
        if self.wnn_block:
            if self.use_wnn and len(self.LUT.lookupTable)>0: # not using wnn for training only for inference 
                self.pre_wnn_x.requires_grad_()
                
                #x = self.pre_wnn_x + residual_scale*self.LUT.forward(self.pre_wnn_x) # It may not actually be useful to scale the residual since it is trained based on its residual effect being one- may require more thought though
                wnn_residual, highest_sim = self.LUT.forward(self.pre_wnn_x)
                wnn_residual = wnn_residual.unsqueeze(0) # add back the batch dim(1)
                res_tensor = torch.zeros_like(self.pre_wnn_x)
                res_tensor[:, -1, :] = wnn_residual
                h = self.pre_wnn_x + highest_sim*(res_tensor) # The idea here is that if highest sim is low then the model doest look as much to the lu
                return h
        return self.pre_wnn_x

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.residual_scale = 15
        assert self.vocab_size > 0

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = torch.nn.ModuleList(
            [TransformerBlock(args=args) for _ in range(args.n_layers)]
        )

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False
        )

        self.freqs_cis = precompute_freqs_cis(self.args.head_dim, 128_000).to("cuda")

        self.n_ctx = 128000 ## 128 k context window


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ):
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[positions]

        mask: Optional[torch.Tensor] = None
        if input_ids.shape[1] > 1:
            seqlen = input_ids.shape[1]
            tensor = torch.full(
                (seqlen, seqlen),
                dtype=h.dtype,
                fill_value=1,
                device=h.device,
            )
            mask = torch.tril(tensor, diagonal=0).to(h.dtype)
            # make the mask banded to account for sliding window
            mask = torch.triu(mask, diagonal=-self.args.sliding_window)
            mask = torch.log(mask)
        
        for layer in self.layers:
            h = layer(h, freqs_cis, positions, mask)

        return self.output(self.norm(h)).float()
    
    def trainLUT(self, tokenizer, lm_head, label, label_context=None, sparsity_level=None):

        # Turning of lut during training- may change this in future
        for block in self.layers:
            block.use_wnn = False

        # Encode label and optional label_context 
        encoded_label = tokenizer.encode(label)

        # pre-fill
        # if label_context is not None: # leaving out label context at the moment for now.
        #     label_context_encoded = tokenizer.encode(label_context)
        # else:
        #     label_context_encoded = []

        if len(encoded_label) == 0:
            return  # nothing to train on

        for i, block in enumerate(self.layers):
            if not block.wnn_block:
                continue

            print(f"Collecting data for WNN training on block {i}")
            now_block = datetime.now()

            for k in range(len(encoded_label)):
                if sparsity_level is not None and sparsity_level < 1.0:
                    if torch.rand(()) > sparsity_level:
                        continue

                context = encoded_label[max(0, k - self.n_ctx):k]
                # if label_context_encoded:
                #     context = label_context_encoded + context

                # Crop on the left if longer than n_ctx
                context = context[-self.n_ctx:]

                if len(context) == 0:
                    continue

                context_tensor = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
                target_tensor = torch.tensor([encoded_label[k]], dtype=torch.long, device=device)     # [1]

                T = context_tensor.size(1)
                position_ids = torch.arange(T, dtype=torch.long, device=device)         # [1, T]
                print("poitions ids shape: ", position_ids.shape)

                h = self.tok_embeddings(context_tensor)
                freqs_cis = self.freqs_cis[position_ids]
                if context_tensor.shape[1] > 1:
                    print("Creating mask")
                    seqlen = context_tensor.shape[1]
                    tensor = torch.full(
                        (seqlen, seqlen),
                        dtype=h.dtype,
                        fill_value=1,
                        device=h.device,
                    )
                    mask = torch.tril(tensor, diagonal=0).to(h.dtype)
                    # make the mask banded to account for sliding window
                    mask = torch.triu(mask, diagonal=-self.args.sliding_window)
                    mask = torch.log(mask)
                else:
                    mask=None
        
                print("Doing inference (pre-LUT blocks)")
                for block_idx in range(i):
                    with torch.no_grad():
                        h = self.layers[block_idx].forward(h, freqs_cis, position_ids, mask)

                print("Doing inference on current block:", i)
                # run current block once to  block.pre_wnn_x
                _ = block(h, freqs_cis, position_ids, mask)
                pre_wnn_x = block.pre_wnn_x  # [1, T, d]

                pre_wnn_x.retain_grad()
                h = pre_wnn_x

      
                for block_idx in range(i + 1, len(self.layers)):
                    h = self.layers[block_idx](h, freqs_cis, position_ids, mask)

                logits = self.output(self.norm(h)).float()  # [1, T, vocab]
                print("Completed inference. Time:", datetime.now() - now_block)

                loss = F.cross_entropy(logits[:, -1, :], target_tensor)

                self.zero_grad()
                ("Start backward")
                now_back = datetime.now()
                loss.backward()
                print("Finish backward. Time:", datetime.now() - now_back)

                # Target residual = negative gradient at pre_wnn_x
                wnn_target_residual = self.residual_scale * (-pre_wnn_x.grad)  # [1, T, d]

                # Take only last time step for LUT
                pre_wnn_x_last = pre_wnn_x.detach()[:, -1, :]           # [1, d]
                target_residual_last = wnn_target_residual.detach()[:, -1, :]  # [1, d]

                # (Optional) normalize to avoid giant residuals on rare tokens
                # target_residual_last = target_residual_last / (
                #     torch.linalg.vector_norm(target_residual_last, ord=2, dim=-1, keepdim=True) + 1e-6
                # )

                print("Training LUT")
                now_lut = datetime.now()
                block.LUT.train(pre_wnn_x_last, target_residual_last)
                print("Time for training LUT:", datetime.now() - now_lut)

            print("Finished block", i, "total time:", datetime.now() - now_block)

        # Re-enable WNN usage for inference
        for block in self.layers:
            block.use_wnn = True

    def saveLUTs(self, save_name):
        base_name = save_name
        for i, block in enumerate(self.layers):
            save_name = base_name +"blockNumber"+str(i)
            if block.wnn_block:
                block.LUT.saveLUT(save_name)

    def loadLUTs(self, save_name):
        for i, block in enumerate(self.layers):
            save_name = save_name +"blockNumber"+str(i)
            if block.wnn_block:
                block.LUT.loadLUT(save_name)

    @staticmethod
    def from_folder(folder: Path, max_batch_size: int = 1, device="cuda", dtype=torch.float16):
        with open(folder / 'params.json', 'r') as f:
            model_args = ModelArgs(**json.loads(f.read()))
        model_args.max_batch_size = max_batch_size
        model = Transformer(model_args).to(device=device, dtype=dtype)
        loaded = torch.load(folder / 'consolidated.00.pth')
        model.load_state_dict(loaded)
        return model


class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str) -> List[int]:
        return [self._model.bos_id(), *self._model.encode(s)]

    def decode(self, t: List[int]) -> str:
        return self._model.decode(t)


@torch.no_grad()
def generate(prompts: List[str], model: Transformer, tokenizer: Tokenizer, max_tokens: int):
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    input_tokens = torch.full((len(prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.long, device="cuda")
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, :len(encoded)] = torch.tensor(encoded).to(input_tokens)
    input_mask = input_tokens != tokenizer.pad_id

    # pre-fill
    positions = torch.arange(0, min_prompt_len).to("cuda")
    logits = model.forward(input_tokens[:, :min_prompt_len], positions)
    logprobs = nn.functional.log_softmax(logits, dim=-1)

    # decode
    generated = []
    all_logprobs = [
        logprobs[:,:-1,:].gather(2, input_tokens[:,1:min_prompt_len,None]).squeeze(-1),
    ]
    cur_pos = min_prompt_len
    for _ in range(max_tokens):
        next_token = torch.argmax(logprobs[:, -1,:], dim=-1)
        if cur_pos < input_mask.shape[1]:
            next_token = torch.where(input_mask[:, cur_pos], input_tokens[:, cur_pos], next_token)
        all_logprobs.append(
            logprobs[:,-1,:].gather(1, next_token[:, None]),
        )
        generated.append(next_token[:, None])
        logits = model.forward(next_token[:, None], torch.LongTensor([cur_pos]).to(next_token))
        logprobs = nn.functional.log_softmax(logits, dim=-1)
        cur_pos += 1

    all_logprobs = torch.cat(all_logprobs, 1)
    res = []
    if max_tokens > 0:
        generated = torch.cat(generated, 1)

        for i, x in enumerate(encoded_prompts):
            res.append(tokenizer.decode(x[:min_prompt_len] + generated[i].tolist()))
    return res, all_logprobs
