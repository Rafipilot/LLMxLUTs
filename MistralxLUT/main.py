import torch
from torch import nn
import torch.nn.functional as F
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

    lut_key_dim: int = 1024


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
    
    @torch.no_grad()
    def reset_kv_cache(self, bsz: int):

        self.cache_k[:bsz].zero_()
        self.cache_v[:bsz].zero_()




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
        self.top_k = 1
        self.tau = 0.07
        self.margin = 0.00

    def train(self, xs, ys):
        # xs, ys: [B, d] or iterable of [d]
        for x, y in zip(xs, ys):
            print("adding rows...")
            x = x.detach().clone().squeeze()
            y = y.detach().clone().squeeze()
            self.keys.append(x)
            self.values.append(y)
            self.lookupTableMetaData.append([1000, 0])

    
    def forward(self, x, hard_gate = True):
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


        N, k = keys.shape

        keys = torch.nan_to_num(keys, nan=0.0, posinf=0.0, neginf=0.0)
        q_f32 = q.to(torch.float32)
        q_f32 = torch.nan_to_num(q_f32, nan=0.0, posinf=0.0, neginf=0.0)

        ### adding a punishment/ cost for looking up memories which have come up just before
        costs = []
        for row in self.lookupTableMetaData:
            number_of_look_up_since_last_hit = row[0]

            cost = (1/(number_of_look_up_since_last_hit+1))
            costs.append(cost)

        # --- 2. empty table: no effect ---
        if len(self.keys) == 0:
            return torch.zeros_like(q), torch.tensor(0.0, device=device)

        costs = torch.tensor(costs, device=device)
        
        sims = F.cosine_similarity(keys, q_f32.unsqueeze(0).expand(N, k), dim=-1)
        sims = torch.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0)
        sims = sims.clamp(-1.0, 1.0)
        sims = sims - self.cost_scale*costs

        k_use = min(self.top_k, N)
        top_sims, top_idx = torch.topk(sims, k=k_use)

        top1 = top_sims[0]                      
        top2 = top_sims[1] if k_use > 1 else top1 - 1e9

        print("top sims:", list(zip(top_sims.tolist(), top_idx.tolist())))
        print("Highest sim:", top1)

        if hard_gate:  # at training we dont want thsi
            if top1 < self.CS_threshold:
                print("Low similarity, cs threshold:", self.CS_threshold)
                return torch.zeros_like(values[0]), torch.tensor(0.0, device=device)

            if (top1 - top2) < self.margin:
                print(f"Ambiguous retrieval: top1-top2={(top1-top2).item():.4f} < margin={self.margin}")
                return torch.zeros_like(values[0]), torch.tensor(0.0, device=device)

        # Temperature-soft weights (over top-k only)
        w = torch.softmax((top_sims - top_sims.max()) / self.tau, dim=0)   # [k_use]
        cand_values = values[top_idx]                                      # [k_use, D]
        retrieved = (w.unsqueeze(-1) * cand_values).sum(dim=0)             # [D]

        winner = top_idx[0].item()
        self.lookupTableMetaData[winner][0] = 0
        self.lookupTableMetaData[winner][1] += 1
        for row in self.lookupTableMetaData:
            row[0] += 1

        return retrieved.to(q.device), top1

    
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
    def __init__(self, args: ModelArgs, wnn_block=False, block_idx = 0):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args
        self.LUT = LUT()
        self.wnn_block = wnn_block 
        if self.wnn_block:
            self.use_wnn = True
        else:
            self.use_wnn = False
        self.pre_wnn_x = None  
        self.residual_scale = 1  

        base_seed = getattr(args, "lut_seed", 1337)
        torch.manual_seed(base_seed + block_idx)

        self.lut_key_dim = getattr(args, "lut_key_dim", args.dim // 4)

        self.lut_key_proj = nn.Parameter(
            torch.empty(args.dim, self.lut_key_dim),
            requires_grad=True
        )
        with torch.no_grad():
            nn.init.orthogonal_(self.lut_key_proj)

        self.lut_gate = nn.Linear(args.dim, args.dim, bias=True)  # learned V-matrix
        
        self.lut_value_proj = nn.Linear(args.dim, args.dim, bias = False) # like a value matrix on rl
        nn.init.eye_(self.lut_value_proj.weight)  # generally good idea to start with the identiy matrix for stability 


            
    def _compute_lut_key(self, pre_wnn_x, lam: float = 0.75, win: int = 32):
        x = pre_wnn_x[0].float()   # <-- FORCE FP32 HERE (biggest fix)
        T, d = x.shape

        last = x[-1]
        tail = x[-min(win, T):]

        ctx = tail[:-1].mean(dim=0) if tail.shape[0] > 1 else last

        k_local = lam * last + (1 - lam) * ctx
        k_local = torch.nan_to_num(k_local, nan=0.0, posinf=0.0, neginf=0.0)

        k_local = F.layer_norm(k_local, (d,))
        k_local = torch.nan_to_num(k_local, nan=0.0, posinf=0.0, neginf=0.0)

        k_local = F.normalize(k_local, dim=-1, eps=1e-6)
        k_local = torch.nan_to_num(k_local, nan=0.0, posinf=0.0, neginf=0.0)

        # make sure projection is fp32 too
        W = self.lut_key_proj.float()
        k = k_local @ W

        k = torch.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)
        k = F.layer_norm(k, (k.shape[0],))
        k = torch.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)

        k = F.normalize(k, dim=-1, eps=1e-6)
        k = torch.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)

        print("key norm:", k.norm().item(), "nan?", torch.isnan(k).any().item())


        return k.unsqueeze(0)  # [1, kdim]



    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        r_attn = self.attention(self.attention_norm(x), freqs_cis, positions, mask)
        h = x + r_attn

        r_ffn = self.feed_forward(self.ffn_norm(h))
        base = h + r_ffn  # this is where LUT attaches

        # Cache a DETACHED copy for LUT training
        self.pre_wnn_x = base.detach()

        out = base

        # LUT only for inference no grads
        if self.wnn_block and self.use_wnn and len(self.LUT.keys) > 0:
            with torch.no_grad():
                key = self._compute_lut_key(self.pre_wnn_x)                   
                rl, highest_sim = self.LUT.forward(key) 
                sim_val = float(highest_sim.item()) if torch.is_tensor(highest_sim) else float(highest_sim)
                if sim_val <= 0.0:
                    return out                  

                h_last = self.pre_wnn_x[0, -1, :].float()
                gate = 2.0 * torch.sigmoid(self.lut_gate(h_last))
                gate = torch.nan_to_num(gate, nan=0.0, posinf=0.0, neginf=0.0)
                rl = torch.nan_to_num(rl, nan=0.0, posinf=0.0, neginf=0.0)
                r = self.lut_value_proj(rl) # applying the V matrix
                

                print("sim", float(highest_sim))
                print("gate mean/min/max", gate.mean().item(), gate.min().item(), gate.max().item())
                print("rl norm", rl.float().norm().item())
                print("retrieval_final norm", (gate * rl).float().norm().item())


                retrieval_final = gate * r

                res_tensor = torch.zeros_like(out)
                res_tensor[:, -1, :] = retrieval_final.unsqueeze(0)
                out = out + (highest_sim * self.residual_scale) * res_tensor

        return out


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
            [TransformerBlock(args=args, block_idx=i) for i in range(args.n_layers)]
        )

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False
        )

        self.freqs_cis = precompute_freqs_cis(self.args.head_dim, 128_000).to("cuda")

        self.n_ctx = 128000 ## 128 k context window

        for p in self.parameters():
            p.requires_grad = False

        # for blk in self.layers:
        #     if getattr(blk, "wnn_block", False):
        #         blk.lut_key_proj.requires_grad = True
        #         for p in blk.lut_value_gate.parameters():
        #             p.requires_grad = True

        # params = []
        # for blk in self.layers:
        #     if getattr(blk, "wnn_block", False):
        #         params.append(blk.lut_key_proj)
        #         params += list(blk.lut_value_gate.parameters())
        # self.lut_opt = torch.optim.AdamW(params, lr=1e-4)

    def rebuild_lut_opt(self, lr=1e-5):

        params = []
        for blk in self.layers:
            if blk.wnn_block:
                # force fp32 for stability
                blk.lut_key_proj.data = blk.lut_key_proj.data.float()
                blk.lut_key_proj.requires_grad = True

                blk.lut_gate.to(dtype=torch.float32)
                for p in blk.lut_gate.parameters():
                    p.requires_grad = True

                blk.lut_value_proj.to(dtype=torch.float32)
                for p in blk.lut_value_proj.parameters():
                    p.requires_grad = True

                params.append(blk.lut_key_proj)
                
                params += list(blk.lut_gate.parameters())
                params += list(blk.lut_value_proj.parameters())
            else:
                blk.lut_key_proj.requires_grad = False
                for p in blk.lut_gate.parameters():
                    p.requires_grad = False

                for p in blk.lut_value_proj.parameters():
                    p.requires_grad = False

        self.lut_opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)



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

        # Disable LUT for during training 
        for blk in self.layers:
            blk.use_wnn = False  # disable lut blocks at the start then re enable at each point 
            if hasattr(blk, "pre_wnn_x"):
                blk.pre_wnn_x = None
            if hasattr(blk, "timeStep_buffer"):
                blk.timeStep_buffer = None
            blk.attention.reset_kv_cache(bsz=1)

        encoded_label = tokenizer.encode(label)
        if encoded_label and encoded_label[0] == tokenizer._model.bos_id():
            encoded_label = encoded_label[1:]   # remove BOS for continuation- this was a decently big bug lol

        encoded_ctx = tokenizer.encode(label_context) if label_context else []

        device = self.tok_embeddings.weight.device

        # Which blocks actually have LUTs
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
                # Optional sparsity: skip some positions- not sure how useful this tbh
                for blk in self.layers:
                    blk.attention.reset_kv_cache(bsz=1)

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
                if len(block.LUT.keys) > 0:
                    key_q = block._compute_lut_key(pre_wnn_x)
                    rl, sim = block.LUT.forward(key_q, hard_gate=False)

                    sim_det = sim.detach().item() if torch.is_tensor(sim) else float(sim)

                    if sim_det <= 0.0:
                        pass
                    else:
                        h_last = pre_wnn_x[0, -1, :].float()
                        gate = torch.sigmoid(block.lut_gate(h_last))
                        gate = torch.nan_to_num(gate, nan=0.0, posinf=0.0, neginf=0.0)

                        rl = torch.nan_to_num(rl, nan=0.0, posinf=0.0, neginf=0.0)
                        r = block.lut_value_proj(rl)
                        retrieval_final = (gate * r).to(h.dtype)

                        inj = torch.zeros_like(h)
                        inj[:, -1, :] = retrieval_final.unsqueeze(0)

                        sim_scale = sim.to(h.dtype)              # keep gradient + keep dtype stable
                        sim_scale = torch.clamp(sim_scale, min=0)  # optional but usually helps
                        h = h + (sim_scale * block.residual_scale) * inj

                # Forward through blocks AFTER i with grad tracking- we need to be able to backwards up to this point...
                for block_idx in range(i + 1, len(self.layers)):
                    h = self.layers[block_idx](h, freqs_cis, position_ids, mask)

                logits = self.output(self.norm(h)).float()  # [1, T, vocab]
                loss = F.cross_entropy(logits[:, -1, :], target_tensor)

                print(f"[trainLUT] Block {i}, position {k}: computing grad")
                now_back = datetime.now()

                self.lut_opt.zero_grad(set_to_none=True)
                loss.backward()                 # backprop through the blocks-after-i path
                print("grad key_proj:", None if block.lut_key_proj.grad is None else block.lut_key_proj.grad.norm().item())
                print("grad gate_w  :", None if block.lut_gate.weight.grad is None else block.lut_gate.weight.grad.norm().item())

                grad_pre_wnn_x = pre_wnn_x.grad.detach()
                grad_pre_wnn_x = torch.nan_to_num(grad_pre_wnn_x, nan=0.0, posinf=0.0, neginf=0.0)

                torch.nn.utils.clip_grad_norm_(self.lut_opt.param_groups[0]["params"], 1.0)

                kp0 = block.lut_key_proj.detach().float().clone()
                vg0 = block.lut_gate.weight.detach().float().clone()
                self.lut_opt.step()

                print("Δ key_proj:", (block.lut_key_proj.detach().float() - kp0).abs().mean().item())
                print("Δ gate_w  :", (block.lut_gate.weight.detach().float() - vg0).abs().mean().item())


                

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

    for blk in model.layers:
        blk.attention.reset_kv_cache(bsz=1)

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