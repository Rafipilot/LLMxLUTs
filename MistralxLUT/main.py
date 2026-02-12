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
        self.raw_keys = []
        self.values = []
        self.lookupTableMetaData = []
        self.CS_threshold = 0.25
        self.cost_scale = 0.0

    def train(self, xs, ys, raw_keys):
        for x, y, rx in zip(xs, ys, raw_keys):
            x  = x.detach().clone().squeeze()
            y  = y.detach().clone().squeeze()
            rx = rx.detach().clone().squeeze()
            self.keys.append(x)
            self.values.append(y)
            self.raw_keys.append(rx)
            self.lookupTableMetaData.append([1000, 0])


    def forward(self, x, topk=8, tau=0.05, differentiable=False, beta=0.02, adapt_key_fn=None):
        if x is None:
            return None, None

        q = x[-1, :].squeeze()
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).float()

        if len(self.keys) == 0:
            return torch.zeros_like(q), 0.0

        device = q.device
        keys = torch.stack([row.to(device=device, dtype=torch.float32) for row in self.keys])
        values = torch.stack([row.to(device=device, dtype=torch.float32) for row in self.values])


        if differentiable:  # if we are in train matries we want to adapt the raw keys so we can learn as transformation. In non diffentiable mode (normal inference) we used cached key values, since they dont change unless we perform train_matrices
            if adapt_key_fn:
                keys = torch.stack([row.to(device=device, dtype=torch.float32) for row in self.raw_keys])
                keys = adapt_key_fn(keys)

        N, d = keys.shape

        costs = []
        for row in self.lookupTableMetaData:
            calls_since_last_hit = row[0]
            costs.append(1.0 / (calls_since_last_hit + 1.0))
        costs = torch.tensor(costs, device=device, dtype=torch.float32)

        sims = F.cosine_similarity(keys, q.unsqueeze(0).expand(N, d), dim=-1)
        sims = torch.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        sims = sims - self.cost_scale * costs

        k = min(topk, N)
        top_sims, top_idx = torch.topk(sims, k=k)

        if differentiable:
            w = torch.softmax(sims / tau, dim=0)          # [N]
            residual = (w[:, None] * values).sum(dim=0)   # [d]
            sim_soft = (w * sims).sum()                   # scalar
            return residual, sim_soft


        max_sim = top_sims.max().item()

        # print("max sim: ", max_sim)

        if max_sim < self.CS_threshold:
            return torch.zeros_like(q), 0.0

        w = torch.softmax(top_sims / tau, dim=-1)
        v = values.index_select(0, top_idx)
        residual = (w.unsqueeze(-1) * v).sum(dim=0)

        return residual.to(q.device), max_sim

    def resetLUT(self):
        self.keys = []
        self.values = []
        self.lookupTableMetaData = []

    def resetCosts(self):
        for row in self.lookupTableMetaData:
            row[0] = 1000


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
    def __init__(self, args, wnn_block = False, block_idx = 0):
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
        self.use_wnn = bool(self.wnn_block)
        self.pre_wnn_x = None
        self.residual_scale = 1

        base_seed = getattr(args, "lut_seed", 1337)
        torch.manual_seed(base_seed + block_idx)

        self.lut_rank = 64

        self.lut_q_down = nn.Linear(args.dim, self.lut_rank, bias=False, dtype=torch.float16)
        self.lut_q_up = nn.Linear(self.lut_rank, args.dim, bias=False, dtype=torch.float16)
        self.lut_q_gate = nn.Linear(args.dim, self.lut_rank, bias=True, dtype=torch.float16)

        self.lut_k_down = nn.Linear(args.dim, self.lut_rank, bias=False, dtype=torch.float16)
        self.lut_k_up = nn.Linear(self.lut_rank, args.dim, bias=False, dtype=torch.float16)
        self.lut_k_gate = nn.Linear(args.dim, self.lut_rank, bias=True, dtype=torch.float16)

        self.lut_v_down = nn.Linear(args.dim, self.lut_rank, bias=False)
        self.lut_v_up = nn.Linear(self.lut_rank, args.dim, bias=False)
        self.lut_v_gate = nn.Linear(args.dim, self.lut_rank, bias=True)

        nn.init.normal_(self.lut_v_up.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.lut_q_up.weight)
        nn.init.zeros_(self.lut_k_up.weight)


    def _compute_lut_key(self, pre_wnn_x, lam=1, win=32, adapt=True):
        x = pre_wnn_x[0]              # [T, d]
        T, d = x.shape
        last = x[-1]
        tail = x[-min(win, T):]
        ctx = tail[:-1].mean(dim=0) if tail.shape[0] > 1 else last

        q_raw = lam * last + (1 - lam) * ctx
        q_raw = F.layer_norm(q_raw, (d,))
        q_raw = F.normalize(q_raw, dim=-1)

        q_adapt = self.adapt_query(q_raw) if adapt else q_raw
        return q_adapt.unsqueeze(0), q_raw.unsqueeze(0)

    def adapt_query(self, q, alpha=4):
        wd = self.lut_q_down.weight.dtype
        if q.dtype != wd:
            q = q.to(wd)

        z = self.lut_q_down(q)
        g = 2 * torch.sigmoid(self.lut_q_gate(q))
        dq = self.lut_q_up(g * z)

        q2 = q + alpha * dq
        q2 = F.layer_norm(q2, (q2.shape[-1],))
        q2 = F.normalize(q2, dim=-1)
        return q2
    
    def adapt_key_batch(self, K, alpha=4): # applying the transformation in a batched way
        wd = self.lut_k_down.weight.dtype
        K = K.to(wd)

        z = self.lut_k_down(K)
        g = 2.0 * torch.sigmoid(self.lut_k_gate(K))
        dK = self.lut_k_up(g * z)

        k2 = K + alpha * dK
        k2 = F.layer_norm(k2, (k2.shape[-1],))
        k2 = F.normalize(k2, dim=-1)
        return k2
            

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor, mask: Optional[torch.Tensor]):
        r_attn = self.attention(self.attention_norm(x), freqs_cis, positions, mask)
        h = x + r_attn

        r_ffn = self.feed_forward(self.ffn_norm(h))
        base = h + r_ffn

        self.pre_wnn_x = base.detach()

        out = base

        if self.wnn_block and self.use_wnn and len(self.LUT.keys) > 0:
            with torch.no_grad():
                key, _ = self._compute_lut_key(self.pre_wnn_x, adapt=True)
                rl, highest_sim = self.LUT.forward(key, topk=8, tau=0.05)
                if highest_sim <= 0.0:
                    return out

                h_last = self.pre_wnn_x[0, -1, :].float()
                rl = torch.nan_to_num(rl, nan=0.0, posinf=0.0, neginf=0.0).float()

                z = self.lut_v_down(rl)
                g = 2.0 * torch.sigmoid(self.lut_v_gate(h_last))
                delta = self.lut_v_up(g * z)

                retrieval_final = (rl + 2 * delta).to(out.dtype)

                h_last_norm = h_last.norm().item()
                scale = float(highest_sim) * self.residual_scale
                add = scale * retrieval_final
                add_norm = add.float().norm().item()
                # print(
                #     f"[blk] sim={float(highest_sim):.3f} "
                #     f"||rl||={rl.float().norm().item():.3f} "
                #     f"||delta||={delta.float().norm().item():.3f} "
                #     f"ratio={delta.float().norm().item() / (rl.float().norm().item() + 1e-6):.3f}"
                # )
                # print(
                #     f"[inj] sim={highest_sim:.3f} ||h_last||={h_last_norm:.2f} ||add||={add_norm:.2f} add/h={add_norm/(h_last_norm+1e-9):.3f}"
                # )

                res_tensor = torch.zeros_like(out)
                res_tensor[:, -1, :] = retrieval_final.unsqueeze(0)

            out = out + scale * res_tensor

        return out


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args=args, block_idx=i) for i in range(args.n_layers)])

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(self.args.head_dim, 128_000).to("cuda")
        self.n_ctx = 128000

        self.lut_opt = None

    def rebuild_lut_opt(self, lr = 1e-5):
        params = []
        for blk in self.layers:
            if getattr(blk, "wnn_block", False):
                for name in (
                    "lut_v_down",
                    "lut_v_up",
                    "lut_v_gate",
                    "lut_q_down",
                    "lut_q_up",
                    "lut_q_gate",
                    "lut_k_down",
                    "lut_k_up",
                    "lut_k_gate",
                ):
                    mod = getattr(blk, name, None)
                    if mod is None:
                        continue
                    mod.to(dtype=torch.float32)
                    for p in mod.parameters():
                        p.requires_grad = True
                    params += list(mod.parameters())
            else:
                for name in (
                    "lut_v_down",
                    "lut_v_up",
                    "lut_v_gate",
                    "lut_q_down",
                    "lut_q_up",
                    "lut_q_gate",
                    "lut_k_down",
                    "lut_k_up",
                    "lut_k_gate",
                ):
                    mod = getattr(blk, name, None)
                    if mod is None:
                        continue
                    for p in mod.parameters():
                        p.requires_grad = False

        self.lut_opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        print("opt param count:", sum(p.numel() for p in params))

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[positions]

        mask: Optional[torch.Tensor] = None
        if input_ids.shape[1] > 1:
            seqlen = input_ids.shape[1]
            tensor = torch.full((seqlen, seqlen), dtype=h.dtype, fill_value=1, device=h.device)
            mask = torch.tril(tensor, diagonal=0).to(h.dtype)
            mask = torch.triu(mask, diagonal=-self.args.sliding_window)
            mask = torch.log(mask)

        for layer in self.layers:
            h = layer(h, freqs_cis, positions, mask)

        return self.output(self.norm(h)).float()

    def trainLUT(self, tokenizer, lm_head, label, label_context=None, sparsity_level=None):
        for blk in self.layers:
            blk.use_wnn = False
            blk.pre_wnn_x = None
            blk.timeStep_buffer = None
            blk.attention.reset_kv_cache(bsz=1)

        encoded_label = tokenizer.encode(label)
        encoded_label.append(tokenizer.eos_id)
        if encoded_label and encoded_label[0] == tokenizer._model.bos_id():
            encoded_label = encoded_label[1:]

        encoded_ctx = tokenizer.encode(label_context) if label_context else []
        device = self.tok_embeddings.weight.device

        wnn_block_indices = [idx for idx, blk in enumerate(self.layers) if getattr(blk, "wnn_block", False)]

        for i in wnn_block_indices:
            block = self.layers[i]
            now_block = datetime.now()
            block.use_wnn = False

            for k in range(len(encoded_label)):
                for blk in self.layers:
                    blk.attention.reset_kv_cache(bsz=1)

                if sparsity_level is not None and sparsity_level < 1.0:
                    if torch.rand(()) > sparsity_level:
                        continue

                context = (encoded_ctx + encoded_label[:k])[-self.n_ctx:]
                if len(context) == 0:
                    continue

                context_tensor = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
                target_tensor = torch.tensor([encoded_label[k]], dtype=torch.long, device=device)

                T = context_tensor.size(1)
                position_ids = torch.arange(T, dtype=torch.long, device=device)

                if T > 1:
                    seqlen = T
                    tensor = torch.full((seqlen, seqlen), dtype=torch.float32, fill_value=1, device=device)
                    mask = torch.tril(tensor, diagonal=0)
                    mask = torch.triu(mask, diagonal=-self.args.sliding_window)
                    mask = torch.log(mask)
                else:
                    mask = None

                with torch.no_grad():
                    h = self.tok_embeddings(context_tensor)
                    freqs_cis = self.freqs_cis[position_ids]
                    for block_idx in range(i):
                        h = self.layers[block_idx](h, freqs_cis, position_ids, mask)
                    _ = self.layers[i](h, freqs_cis, position_ids, mask)
                    pre_wnn_x_val = getattr(self.layers[i], "pre_wnn_x", None)

                if pre_wnn_x_val is None:
                    continue

                pre_wnn_x = pre_wnn_x_val.detach().clone().requires_grad_(True)
                h = pre_wnn_x

                for block_idx in range(i + 1, len(self.layers)):
                    h = self.layers[block_idx](h, freqs_cis, position_ids, mask)

                logits = self.output(self.norm(h)).float()
                loss = F.cross_entropy(logits[:, -1, :], target_tensor)

                if pre_wnn_x.grad is not None:
                    pre_wnn_x.grad = None
                loss.backward()

                grad_pre_wnn_x = torch.nan_to_num(pre_wnn_x.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                wnn_target_residual = -grad_pre_wnn_x

                with torch.no_grad():
                    key_vec, raw_key_vec = block._compute_lut_key(pre_wnn_x.detach(), adapt=False)  # we want to store the raw key, not the adapted one!
                    value_vec = wnn_target_residual.detach()[:, -1, :]
                    block.LUT.train(key_vec, value_vec, raw_key_vec)

                    pre_wnn_x.grad = None
                    self.layers[i].pre_wnn_x = None

            print(f"[trainLUT] Finished block {i} in {datetime.now() - now_block}")

        for blk in self.layers:
            blk.use_wnn = True

        

    def trainTransformations(self, tokenizer, lm_head, label, label_context=None):
        if self.lut_opt is None:
            self.rebuild_lut_opt(lr=1e-5)

        for blk in self.layers:
            blk.use_wnn = False
            blk.pre_wnn_x = None
            blk.timeStep_buffer = None
            blk.attention.reset_kv_cache(bsz=1)

        encoded_label = tokenizer.encode(label)
        encoded_label.append(tokenizer.eos_id)
        if encoded_label and encoded_label[0] == tokenizer._model.bos_id():
            encoded_label = encoded_label[1:]

        encoded_ctx = tokenizer.encode(label_context) if label_context else []
        device = self.tok_embeddings.weight.device

        wnn_block_indices = [idx for idx, blk in enumerate(self.layers) if getattr(blk, "wnn_block", False)]

        for i in wnn_block_indices:
            block = self.layers[i]
            now_block = datetime.now()
            block.use_wnn = False

            for k in range(len(encoded_label)):
                for blk in self.layers:
                    blk.attention.reset_kv_cache(bsz=1)

                context = (encoded_ctx + encoded_label[:k])[-self.n_ctx:]
                if len(context) == 0:
                    continue

                context_tensor = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
                target_tensor = torch.tensor([encoded_label[k]], dtype=torch.long, device=device)

                T = context_tensor.size(1)
                position_ids = torch.arange(T, dtype=torch.long, device=device)

                if T > 1:
                    seqlen = T
                    tensor = torch.full((seqlen, seqlen), dtype=torch.float32, fill_value=1, device=device)
                    mask = torch.tril(tensor, diagonal=0)
                    mask = torch.triu(mask, diagonal=-self.args.sliding_window)
                    mask = torch.log(mask)
                else:
                    mask = None

                with torch.no_grad():
                    h0 = self.tok_embeddings(context_tensor)
                    freqs_cis = self.freqs_cis[position_ids]
                    for block_idx in range(i):
                        h0 = self.layers[block_idx](h0, freqs_cis, position_ids, mask)
                    _ = self.layers[i](h0, freqs_cis, position_ids, mask)
                    pre_wnn_x_val = getattr(self.layers[i], "pre_wnn_x", None)

                if pre_wnn_x_val is None:
                    continue

                pre_wnn_x = pre_wnn_x_val.detach().clone().requires_grad_(True)
                h = pre_wnn_x

                q_adapt, _q_raw = block._compute_lut_key(pre_wnn_x.detach(), adapt=True)
                rl, sim = block.LUT.forward(
                    q_adapt,
                    differentiable=True,
                    adapt_key_fn=block.adapt_key_batch,
                )

                if sim <= 0.0:
                    continue

                h_last = pre_wnn_x[0, -1, :].float()
                rl = torch.nan_to_num(rl, nan=0.0, posinf=0.0, neginf=0.0).float()

                z = block.lut_v_down(rl)
                g = 2.0 * torch.sigmoid(block.lut_v_gate(h_last))
                delta = block.lut_v_up(g * z)
                retrieval_final = (rl + 2 * delta).to(h.dtype)

                inj = torch.zeros_like(h)
                inj[:, -1, :] = retrieval_final.unsqueeze(0)
                h = h + (sim * block.residual_scale) * inj

                for block_idx in range(i + 1, len(self.layers)):
                    h = self.layers[block_idx](h, freqs_cis, position_ids, mask)

                logits = self.output(self.norm(h)).float()
                loss = F.cross_entropy(logits[:, -1, :], target_tensor)

                self.lut_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lut_opt.param_groups[0]["params"], 1.0)

                def gn(p):
                    return None if (p.grad is None) else p.grad.detach().float().norm().item()

                self.lut_opt.step()

                pre_wnn_x.grad = None
                self.layers[i].pre_wnn_x = None

            print(f"[trainTransformations] Finished block {i} in {datetime.now() - now_block}")

        for blk in self.layers:
            if blk.wnn_block:
                blk.use_wnn = True
                with torch.no_grad():
                    K_raw = torch.stack([rk.to(device=device, dtype=torch.float32) for rk in blk.LUT.raw_keys])
                    K_new = blk.adapt_key_batch(K_raw)
                    blk.LUT.keys = [K_new[i].detach().cpu() for i in range(K_new.shape[0])] # apply the key matrix again to the keys and store it every time it changes

        

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