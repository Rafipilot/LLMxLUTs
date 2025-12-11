import torch
from torch import nn
from dataclasses import dataclass
from pathlib import Path

import json
from typing import Optional, Tuple, List
from sentencepiece import SentencePieceProcessor

import torch.nn.functional as F
from datetime import datetime

from safetensors.torch import load_file as safe_load

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
    sliding_window: int =4096 
    norm_eps: float = None
    vocab_size: int= None

    max_batch_size: int = 0
    rope_theta: float = 1000000.0

    lut_key_dim: int = 128


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

        self.attn_scores = None ## for lut

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
        self.attn_scores = scores
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

class LUT:
    def __init__(self):
        self.keys = []
        self.values = []
        self.lookupTableMetaData = [] # idx 0 calls since last response, idx 1 number of calls
        self.CS_threshold = 0.25
        self.cost_scale = 0.0

    def train(self, xs, ys):
        # xs, ys: [B, d] or iterable of [d]
        for x, y in zip(xs, ys):
            print("adding rows...")
            x = x.detach().clone().squeeze()
            y = y.detach().clone().squeeze()
            self.keys.append(x)
            self.values.append(y)
            self.lookupTableMetaData.append([1000, 0])

    
    def forward(self, x):
        if x is None:
            return None, None

        if x.dim() == 3:
            q = x[-1, -1, :]
        elif x.dim() == 2:
            q = x[-1, :]
        elif x.dim() == 1:
            q = x
        else:
            raise ValueError(f"LUT.forward: unsupported x.dim()={x.dim()}, shape={x.shape}")

        q = q.squeeze()
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

        
        if len(self.keys) == 0:
            return torch.zeros_like(q), 0.0

        device = q.device

        keys = torch.stack([row.to(device=device, dtype=torch.float32) for row in self.keys])
        values = torch.stack([row.to(device=device, dtype=torch.float32) for row in self.values])

        # --- 1. normalize input into a single [d] vector q ---
        if x.dim() == 3:
            # [B, T, d] -> last batch, last token
            q = x[-1, -1, :]
        elif x.dim() == 2:
            # [B, d] -> last batch
            q = x[-1, :]
        elif x.dim() == 1:
            q = x
        else:
            raise ValueError(f"LUT.forward: unsupported x.dim()={x.dim()}, shape={x.shape}")


        N, k = keys.shape

        keys = torch.nan_to_num(keys, nan=0.0, posinf=0.0, neginf=0.0)
        q_f32 = q.to(torch.float32)
        q_f32 = torch.nan_to_num(q_f32, nan=0.0, posinf=0.0, neginf=0.0)

        ### adding a punishment/ cost for looking up memories which have come up just before
        costs = []
        for row in self.lookupTableMetaData:
            number_of_look_up_since_last_hit = row[0]

        # --- 2. empty table: no effect ---
        if len(self.lookupTable) == 0:
            return torch.zeros_like(q), 0.0

        costs = torch.tensor(costs, device=device)
        
        sims = F.cosine_similarity(keys, q_f32.unsqueeze(0).expand(N, k), dim=-1)
        sims = torch.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0)
        sims = sims.clamp(-1.0, 1.0)
        sims = sims - self.cost_scale*costs

        max_sim_idx = torch.argmax(sims)
        highest_sim = sims[max_sim_idx].item()
        best_residual = values[max_sim_idx]  # [d]

        print("Highest sim: ", highest_sim)
        if highest_sim < self.CS_threshold:
            print("Low similarity, cs threshold:", self.CS_threshold)
            return torch.zeros_like(best_residual), 0.0
        
        row_meta_data = self.lookupTableMetaData[max_sim_idx]
        self.lookupTableMetaData[max_sim_idx]= [0, row_meta_data[1]+1]
        

        return best_residual.to(q.device), highest_sim
    
    def resetLUT(self):
        self.keys = []
        self.values = []
        self.lookupTableMetaData = []

    def resetCosts(self):
        for row in self.lookupTableMetaData:
            row[0] = 1000

    # def saveLUT(self, save_name):
    #     with open(save_name, "w") as f:
    #         save_lookup = self.lookupTable.numpy()
    #         json.dump(save_lookup, f, indent=2)

    # def loadLUT(self, save_name):
    #     with open(save_name, "r") as f:
    #         self.lookupTable = json.load(f)

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
        self.residual_scale = 1   # or 15, but be consistent everywhere

        # --- projection down to lut_key_dim (from main) ---
        self.lut_key_dim = getattr(args, "lut_key_dim", args.dim // 4)
        self.lut_key_proj = nn.Parameter(
            torch.empty(args.dim, self.lut_key_dim),
            requires_grad=False
        )
        nn.init.orthogonal_(self.lut_key_proj)

    def _compute_lut_key(self, pre_wnn_x, lam: float = 0.5):
        """
        pre_wnn_x: [1, T, d]  (hidden states at this block)
        Uses attention to build a context vector, combines with last token,
        then projects down to lut_key_dim.
        returns: [1, lut_key_dim] key
        """
        import torch
        import torch.nn.functional as F

        B, T, d = pre_wnn_x.shape
        assert B == 1, f"Expected batch size 1, got {B}"
        device = pre_wnn_x.device

        # --- 1. baseline key: last-token hidden ---
        key_last = pre_wnn_x[:, -1, :]      # [1, d]

        # --- 2. get attention probs for last query (from attn-hybrid branch) ---
        attn_scores = getattr(self.attention, "attn_scores", None)

        if attn_scores is not None:
            # attn_scores: [1, H, T_q, T_k]
            _, H, T_q, T_k = attn_scores.shape
            last_q = T_q - 1

            # average heads for the last token: [H, T_k] -> [T_k]
            alpha = attn_scores[0, :, last_q, :].mean(dim=0)  # [T_k]

            # --- 3. align T_k with T ---
            if T_k > T:
                alpha = alpha[-T:]
            elif T_k < T:
                pad = T - T_k
                alpha = torch.cat(
                    [torch.zeros(pad, device=device, dtype=alpha.dtype), alpha],
                    dim=0,
                )  # [T]

            # normalise attention so it sums to 1
            alpha_sum = alpha.sum()
            if alpha_sum > 0:
                alpha = alpha / (alpha_sum + 1e-9)  # [T]

                # --- 4. build context vector using attention over hidden states ---
                context = (alpha.view(1, T, 1) * pre_wnn_x).sum(dim=1)   # [1, d]

                # --- 5. combine: last token + λ * context ---
                combined = key_last + lam * context                      # [1, d]
            else:
                # degenerate attention -> just use last-token key
                combined = key_last
        else:
            # no attention cached (e.g. in some paths) -> fall back
            combined = key_last

        # --- 6. normalise in model space ---
        combined = F.normalize(combined.float(), dim=-1).to(pre_wnn_x.dtype)  # [1, d]

        # --- 7. project down to lut_key_dim (from main branch) ---
        key_low = combined @ self.lut_key_proj    # [1, lut_key_dim]
        key_low = F.normalize(key_low, dim=-1)

        return key_low

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

        # Disable LUT use during training + clear stale caches
        for blk in self.layers:
            blk.use_wnn = False  # disable lut blocks at the start then re enable at each point 
            if hasattr(blk, "pre_wnn_x"):
                blk.pre_wnn_x = None

        encoded_label = tokenizer.encode(label)
        encoded_label.append(tokenizer.eos_id) # train with a eos token at the end
        if len(encoded_label) == 0:
            return
        
        if label_context is not None and len(label_context) > 0:
            encoded_ctx = tokenizer.encode(label_context)
        else:
            encoded_ctx = []

        device = self.tok_embeddings.weight.device

        # Which blocks actually have LUTs?
        wnn_block_indices = [
            idx for idx, blk in enumerate(self.layers)
            if getattr(blk, "wnn_block", False)
        ]

        for i in wnn_block_indices:
            block = self.layers[i]
            print(f"[trainLUT] Training LUT for block {i}")
            now_block = datetime.now()
            block.use_wnn = False # ensure current block is disabled per train

            for k in range(len(encoded_label)):
                # Optional sparsity: skip some positions
                if sparsity_level is not None and sparsity_level < 1.0:
                    if torch.rand(()) > sparsity_level:
                        continue

                context = encoded_ctx+ encoded_label[:k]
                context = context[-self.n_ctx:]
                if len(context) == 0:
                    continue

                context_tensor = torch.tensor(
                    context, dtype=torch.long, device=device
                ).unsqueeze(0)  # [1, T]
                target_tensor = torch.tensor(
                    [encoded_label[k]], dtype=torch.long, device=device
                )

                T = context_tensor.size(1)
                position_ids = torch.arange(T, dtype=torch.long, device=device)

                # --- mask ---
                if T > 1:
                    seqlen = T
                    tensor = torch.full(
                        (seqlen, seqlen),
                        dtype=torch.float32,
                        fill_value=1,
                        device=device,
                    )
                    mask = torch.tril(tensor, diagonal=0)
                    mask = torch.triu(mask, diagonal=-self.args.sliding_window)
                    mask = torch.log(mask)
                else:
                    mask = None

                # forward to block i (no_grad)
                with torch.no_grad():
                    h = self.tok_embeddings(context_tensor)
                    freqs_cis = self.freqs_cis[position_ids]

                    for block_idx in range(i):
                        h = self.layers[block_idx](h, freqs_cis, position_ids, mask)

                    # Run block i once to populate pre_wnn_x (detached)- no need to do backwards 
                    _ = self.layers[i](h, freqs_cis, position_ids, mask)
                    pre_wnn_x_val = getattr(self.layers[i], "pre_wnn_x", None)

                if pre_wnn_x_val is None:
                    continue

                pre_wnn_x = pre_wnn_x_val.detach().clone().requires_grad_(True)
                h = pre_wnn_x

                # Forward through blocks AFTER i with grad tracking- we need to be able to backwards up to this point...
                for block_idx in range(i + 1, len(self.layers)):
                    h = self.layers[block_idx](h, freqs_cis, position_ids, mask)

                logits = self.output(self.norm(h)).float()  # [1, T, vocab]
                loss = F.cross_entropy(logits[:, -1, :], target_tensor)

                print(f"[trainLUT] Block {i}, position {k}: computing grad")
                now_back = datetime.now()

                grad_pre_wnn_x, = torch.autograd.grad(
                    loss,
                    pre_wnn_x,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )

                grad_norm = grad_pre_wnn_x.norm().item()
                print(f"[trainLUT] Block {i}, position {k}: grad_norm={grad_norm:.4e}")

                print(
                    f"[trainLUT] Block {i}, position {k}: grad computed in {datetime.now() - now_back}"
                )

                # Grad to residual
                wnn_target_residual = (-grad_pre_wnn_x)  # [1, T, d]

                # Last time step
                pre_wnn_x_last = pre_wnn_x.detach()[:, -1, :]              # [1, d]
                target_residual_last = wnn_target_residual.detach()[:, -1, :]  # [1, d]

                print(f"[trainLUT] Training LUT on block {i}")
                now_lut = datetime.now()
                with torch.no_grad():
                    key_vec = block._compute_lut_key(pre_wnn_x.detach())
                    value_vec = wnn_target_residual.detach()[:, -1, :]
                    block.LUT.train(key_vec, value_vec)
                print(
                    f"[trainLUT] Block {i}, position {k}: LUT updated in {datetime.now() - now_lut}"
                )

                # clean any accidental grad references
                pre_wnn_x.grad = None
                self.layers[i].pre_wnn_x = None

            # block.use_wnn = True # re enable this lut block

            print(f"[trainLUT] Finished block {i} in {datetime.now() - now_block}")

        # Re-enable LUT for inference
        for blk in self.layers:
            blk.use_wnn = True  # this should be redundant


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

        try:
            loaded = torch.load(folder / 'consolidated.00.pth', map_location="cpu")
        except Exception as e:
            loaded = safe_load(str(folder / "consolidated.safetensors"))

        missing, unexpected = model.load_state_dict(loaded, strict=False)
        if missing:
            print("[from_folder] Missing keys (expected for new stuff like lut_key_proj):", missing)
        if unexpected:
            print("[from_folder] Unexpected keys:", unexpected)

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

    device = "cuda"

    # [B, max_prompt_len] padded inputs
    input_tokens = torch.full(
        (len(prompts), max_prompt_len),
        tokenizer.pad_id,
        dtype=torch.long,
        device=device,
    )
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, :len(encoded)] = torch.tensor(encoded, device=device)
    input_mask = input_tokens != tokenizer.pad_id  # True where prompt token, False where pad

    # ---------- prefill over shared prefix ----------
    positions = torch.arange(0, min_prompt_len, device=device)
    logits = model.forward(input_tokens[:, :min_prompt_len], positions)
    logprobs = nn.functional.log_softmax(logits, dim=-1)

    # NLL for prompt tokens (teacher forcing)
    all_logprobs = [
        logprobs[:, :-1, :].gather(2, input_tokens[:, 1:min_prompt_len, None]).squeeze(-1),
    ]

    # ---------- decode ----------
    generated = []
    cur_pos = min_prompt_len

    eos_id = tokenizer.eos_id  
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=device)

    for _ in range(max_tokens):
        # should implement temperature!!
        sampled = torch.argmax(logprobs[:, -1, :], dim=-1)  # (B,)

        # Are we still inside the original prompt at this position?
        if cur_pos < input_mask.shape[1]:
            is_prompt_pos = input_mask[:, cur_pos]  # True = still prompt token for that example
            next_token = torch.where(is_prompt_pos, input_tokens[:, cur_pos], sampled)
        else:
            is_prompt_pos = torch.zeros_like(sampled, dtype=torch.bool, device=device)
            next_token = sampled

        

        # Logprob of chosen token (prompt or generated)
        all_logprobs.append(
            logprobs[:, -1, :].gather(1, next_token[:, None])
        )

        # Mark EOS hits, but ONLY on genuinely generated tokens
        # (i.e. not on teacher-forced prompt positions).
        if eos_id is not None:
            eos_hit = (~is_prompt_pos) & (~finished) & (next_token == eos_id)
            finished = finished | eos_hit

        generated.append(next_token[:, None])  # (B, 1)

        # One-step forward for next position
        logits = model.forward(
            next_token[:, None],
            torch.LongTensor([cur_pos]).to(next_token),
        )
        logprobs = nn.functional.log_softmax(logits, dim=-1)
        cur_pos += 1

        # If every sequence has produced EOS somewhere in its generated region, stop early
        if finished.all():
            break

    all_logprobs = torch.cat(all_logprobs, dim=1)
    res = []

    if max_tokens > 0 and generated:
        generated = torch.cat(generated, dim=1)  # (B, T_gen)

        for i, x in enumerate(encoded_prompts):
            gen_tokens = generated[i].tolist()

            # Trim at first EOS in generated region (if any)
            if eos_id is not None and eos_id in gen_tokens:
                eos_index = gen_tokens.index(eos_id)
                gen_tokens = gen_tokens[:eos_index]

            # Reconstruct output: shared prefix (up to min_prompt_len) + generated tail
            prompt_ids = x[:min_prompt_len] + gen_tokens
            gen_ids = gen_tokens

            prompt_text = tokenizer.decode(prompt_ids)
            answer_text = tokenizer.decode(gen_ids)

            res.append(answer_text)
    return res, all_logprobs