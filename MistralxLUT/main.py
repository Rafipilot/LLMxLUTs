import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path

import gc
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
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = args.sliding_window
        self.scale = self.args.head_dim ** -0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

        self.cache_k = torch.empty(
            (args.max_batch_size, args.sliding_window, self.n_kv_heads, self.args.head_dim),
            dtype=torch.float16
        ).cuda()
        self.cache_v = torch.empty(
            (args.max_batch_size, args.sliding_window, self.n_kv_heads, self.args.head_dim),
            dtype=torch.float16
        ).cuda()

    def reset_kv_cache(self, bsz):
        self.cache_k[:bsz].zero_()
        self.cache_v[:bsz].zero_()

    def forward(self, x, freqs_cis, positions, mask, use_cache=True):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.args.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if use_cache:
            scatter_pos = (positions[-self.sliding_window:] % self.sliding_window)[None, :, None, None]
            scatter_pos = scatter_pos.repeat(bsz, 1, self.n_kv_heads, self.args.head_dim)
            self.cache_k[:bsz].scatter_(dim=1, index=scatter_pos, src=xk[:, -self.sliding_window:])
            self.cache_v[:bsz].scatter_(dim=1, index=scatter_pos, src=xv[:, -self.sliding_window:])

        if use_cache and positions.shape[0] == 1:
            cur_pos = positions[-1].item() + 1
            key, value = repeat_kv(
                self.cache_k[:bsz, :cur_pos, ...],
                self.cache_v[:bsz, :cur_pos, ...],
                self.repeats
            )
        else:
            key, value = repeat_kv(xk, xv, self.repeats)

        query = xq.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = torch.matmul(query, key.transpose(2, 3)) * self.scale
        if mask is not None:
            scores += mask[None, None, ...]

        scores = scores.float()
        scores = F.softmax(scores, dim=-1).type_as(query)
        output = torch.matmul(scores, value)
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
    def __init__(self, value_dim):
        self.keys = []
        self.values = []
        self.raw_hiddens = []
        self.fact_ids = []
        self.lookupTableMetaData = [] # idx 0 calls since last response, idx 1 number of calls
        self.CS_threshold = 0.25
        self.value_dim = value_dim

    def train(self, xs, ys, raw_hs=None, fact_id=None):
        # xs, ys: [B, d] or iterable of [d]
        # raw_hs: optional raw hidden states for differentiable recomputation
        for idx, (x, y) in enumerate(zip(xs, ys)):
            x = x.detach().clone().squeeze()
            y = y.detach().clone().squeeze()
            self.keys.append(x)
            self.values.append(y)
            self.lookupTableMetaData.append([1000, 0])
            self.fact_ids.append(fact_id)
            if raw_hs is not None:
                self.raw_hiddens.append(raw_hs[idx].detach().clone().squeeze().cpu())

    
    def forward(self, q):
        if q is None:
            return None, 0.0

        q = q.squeeze()
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).float()

        if len(self.keys) == 0:
            return torch.zeros(self.value_dim, device=q.device), 0.0

        device = q.device
        keys = torch.stack([row.to(device=device, dtype=torch.float32) for row in self.keys])
        values = torch.stack([row.to(device=device, dtype=torch.float32) for row in self.values])
        keys = torch.nan_to_num(keys, nan=0.0, posinf=0.0, neginf=0.0)
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        sims = F.cosine_similarity(keys, q.unsqueeze(0).expand(keys.shape[0], keys.shape[1]), dim=-1)
        sims = torch.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)

        max_idx = torch.argmax(sims)
        best = values[max_idx]
        best_sim = sims[max_idx].item()

        # print("Best sim: ", best_sim)

        if best_sim < self.CS_threshold:
            return torch.zeros_like(best), 0.0

        return best.to(device), best_sim
    
    def forward_differentiable(self, q, top_k=8):
        if q is None:
            return None, None

        q = q.squeeze()
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).float()

        if len(self.keys) == 0:
            return torch.zeros(self.value_dim, device=q.device), None

        device = q.device
        keys = torch.stack([row.to(device=device, dtype=torch.float32) for row in self.keys])
        values = torch.stack([row.to(device=device, dtype=torch.float32) for row in self.values])
        keys = torch.nan_to_num(keys, nan=0.0, posinf=0.0, neginf=0.0)
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        sims = F.cosine_similarity(keys, q.unsqueeze(0).expand(keys.shape[0], keys.shape[1]), dim=-1)
        sims = torch.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)

        # Top-k: restrict softmax to top-k entries for sharp retrieval
        k = min(top_k, sims.shape[0])
        topk_sims, topk_idx = torch.topk(sims, k)
        topk_values = values[topk_idx]

        weights = F.softmax(topk_sims / 0.5, dim=0)
        best = torch.sum(weights.unsqueeze(-1) * topk_values, dim=0)
        sim_soft = torch.sum(weights * topk_sims)

        return best.to(device), sim_soft
    
    def resetLUT(self):
        self.keys = []
        self.values = []
        self.raw_hiddens = []
        self.fact_ids = []
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
        self.LUT = LUT(args.dim)
        self.wnn_block = wnn_block # we only activate the wnn block in the last n layers as specified by num_wnn_blocks
        if self.wnn_block:
            self.use_wnn = True
        else:
            self.use_wnn = False
        self.pre_wnn_x = None  
        self.residual_scale = 1

        # new approach create three mlps: memory_encoder (h_last -> memory), memory_decoder (memory + h_last -> residual_correction) and the memory key (h_last -> key)

        self.key_dim = 128
        self.mem_rank = 32

        # mem_key: single linear projection (no bottleneck — preserves hidden state info)
        self.mem_key = nn.Linear(args.dim, self.key_dim, bias=False)
        # mem_enc: single linear (no bottleneck — preserves entity-specific info in values)
        self.mem_enc = nn.Linear(args.dim, args.dim, bias=False)
        self.mem_dec_rank = 128
        self.mem_dec = nn.Sequential(nn.Linear(args.dim, self.mem_dec_rank, bias=False), nn.SiLU(), nn.Linear(self.mem_dec_rank, args.dim, bias=False))

        nn.init.xavier_uniform_(self.mem_dec[2].weight, gain=0.1)

        self.read_gate = nn.Linear(self.key_dim, 1, bias=True)
        nn.init.constant_(self.read_gate.bias, 0.0)

    def pool_span(self, pre_wnn_x, win=32):
        x = pre_wnn_x[0]  # [T, d]
        tail = x[-min(win, x.shape[0]):]
        pooled = tail.mean(dim=0)
        pooled = F.layer_norm(pooled, (pooled.shape[-1],))
        return pooled

            
    def _compute_lut_key(self, pre_wnn_x, lam: float = 0.75, win: int = 32):
        x = pre_wnn_x[0]                 # [T, d]
        T, d = x.shape

        last = x[-1]
        tail = x[-min(win, T):]          # [L, d]

        # plain (non-attention) context pooling, excluding last to avoid double-counting
        if tail.shape[0] > 1:
            ctx = tail[:-1].mean(dim=0)
        else:
            ctx = last

        k_local = lam * last + (1 - lam) * ctx
        k_local = F.layer_norm(k_local, (d,))
        k_local = F.normalize(k_local, dim=-1)
        return k_local.unsqueeze(0)


    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor],
        diffentiable_lookup=False,
        use_cache=True,
    ) -> torch.Tensor:
        # Standard transformer block forward
        r_attn = self.attention(self.attention_norm(x), freqs_cis, positions, mask, use_cache=use_cache)
        h = x + r_attn

        r_ffn = self.feed_forward(self.ffn_norm(h))
        base = h + r_ffn  # this is where LUT attaches

        # Cache a DETACHED copy for LUT training
        self.pre_wnn_x = base.detach()

        out = base

        # LUT inference
        if self.wnn_block and self.use_wnn and len(self.LUT.keys) > 0:

            key_dtype = next(self.mem_key.parameters()).dtype
            dec_dtype = next(self.mem_dec.parameters()).dtype

            h_last_key = self.pre_wnn_x[0, -1, :].to(dtype=key_dtype)
            h_last_dec = self.pre_wnn_x[0, -1, :].to(dtype=dec_dtype)

            q = self.mem_key(h_last_key).float()
            q = F.layer_norm(q, (q.shape[-1],))
            q = F.normalize(q, dim=-1)

            mem, sim = self.LUT.forward_differentiable(q, top_k=8)

            if mem is None or sim is None:
                return out
            sim_t = sim.to(device=out.device, dtype=dec_dtype)

            mem = mem.to(dtype=dec_dtype)
            delta = self.mem_dec(mem)
            delta = delta * (h_last_dec.norm() / (delta.norm() + 1e-8))

            # Gate now takes mem_key output (key_dim) instead of raw h_last (dim)
            gate_dtype = next(self.read_gate.parameters()).dtype
            gate_input = self.mem_key(h_last_key).to(dtype=gate_dtype)
            gate = torch.sigmoid(self.read_gate(gate_input)).squeeze(-1).to(dec_dtype)
            scale = gate * sim_t * self.residual_scale

            res_tensor = torch.zeros_like(out)
            res_tensor[:, -1, :] = delta.to(out.dtype).unsqueeze(0)
            out = out + scale * res_tensor

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

        self.lut_opt = None


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        diffentiable_lookup: bool = False,
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
            h = layer(h, freqs_cis, positions, mask, diffentiable_lookup=diffentiable_lookup)

        return self.output(self.norm(h)).float()


    def trainLUT(self, tokenizer, lm_head, label, label_context=None, sparsity_level=None, fact_id=None):

        # Disable all WNN during LUT population (safe baseline)
        for blk in self.layers:
            blk.use_wnn = False
            if hasattr(blk, "pre_wnn_x"):
                blk.pre_wnn_x = None

        encoded_label = tokenizer.encode(label)
        encoded_label.append(tokenizer.eos_id)
        if encoded_label and encoded_label[0] == tokenizer._model.bos_id():
            encoded_label = encoded_label[1:]
        if len(encoded_label) == 0:
            return

        encoded_ctx = tokenizer.encode(label_context) if label_context and len(label_context) > 0 else []

        device = self.tok_embeddings.weight.device

        # Which blocks actually have LUTs?
        wnn_block_indices = [
            idx for idx, blk in enumerate(self.layers)
            if getattr(blk, "wnn_block", False)
        ]

        for i in wnn_block_indices:
            block = self.layers[i]
            # Disable current block's WNN but KEEP earlier WNN blocks active
            # so hidden states match what inference will see
            block.use_wnn = False

            # Store only last N token positions (most fact-specific)
            n_store = min(3, len(encoded_label))
            for k in range(len(encoded_label) - n_store, len(encoded_label)):
                context = encoded_ctx + encoded_label[:k]
                context = context[-self.n_ctx:]
                if len(context) == 0:
                    continue

                context_tensor = torch.tensor(
                    context, dtype=torch.long, device=device
                ).unsqueeze(0)

                T = context_tensor.size(1)
                position_ids = torch.arange(T, dtype=torch.long, device=device)

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

                with torch.no_grad():
                    h = self.tok_embeddings(context_tensor)
                    freqs_cis = self.freqs_cis[position_ids]

                    # Earlier WNN blocks run with use_wnn=True (matching inference)
                    for block_idx in range(i):
                        h = self.layers[block_idx](h, freqs_cis, position_ids, mask, use_cache=False)

                    # Current block runs with use_wnn=False to capture pre-injection hidden
                    _ = self.layers[i](h, freqs_cis, position_ids, mask, use_cache=False)
                    pre_wnn_x_val = getattr(self.layers[i], "pre_wnn_x", None)

                if pre_wnn_x_val is None:
                    continue

                with torch.no_grad():
                    key_dtype = next(block.mem_key.parameters()).dtype
                    enc_dtype = next(block.mem_enc.parameters()).dtype

                    h_last = pre_wnn_x_val[0, -1, :]

                    key_vec = block.mem_key(h_last.to(dtype=key_dtype)).float()
                    key_vec = F.layer_norm(key_vec, (key_vec.shape[-1],))
                    key_vec = F.normalize(key_vec, dim=-1).unsqueeze(0)
                    key_vec = torch.nan_to_num(key_vec, nan=0.0, posinf=0.0, neginf=0.0)

                    val_vec = block.mem_enc(h_last.to(dtype=enc_dtype)).float().unsqueeze(0)
                    raw_h = h_last.float().unsqueeze(0)
                    block.LUT.train(key_vec, val_vec, raw_hs=raw_h, fact_id=fact_id)

        # Re-enable all WNN blocks for inference
        for blk in self.layers:
            if getattr(blk, "wnn_block", False):
                blk.use_wnn = True

    def trainTransformations(self, tokenizer, lm_head, label, label_context=None, current_fact_id=None, training_step=0, phase1_steps=100):
        if self.lut_opt is None:
            raise RuntimeError("Call rebuild_lut_opt() before trainTransformations()")

        for blk in self.layers:
            blk.use_wnn = False
            blk.pre_wnn_x = None

        encoded_label = tokenizer.encode(label)
        encoded_label.append(tokenizer.eos_id)
        if encoded_label and encoded_label[0] == tokenizer._model.bos_id():
            encoded_label = encoded_label[1:]

        encoded_ctx = tokenizer.encode(label_context) if label_context else []
        device = self.tok_embeddings.weight.device

        wnn_block_indices = [
            idx for idx, blk in enumerate(self.layers)
            if getattr(blk, "wnn_block", False)
        ]
        overall_losses = 0.0
        overall_contrast = 0.0
        n_tokens_total = 0

        self.lut_opt.zero_grad(set_to_none=True)

        for i in wnn_block_indices:
            block = self.layers[i]
            block.use_wnn = False

            if len(block.LUT.raw_hiddens) == 0:
                continue

            key_dtype = next(block.mem_key.parameters()).dtype
            enc_dtype = next(block.mem_enc.parameters()).dtype
            dec_dtype = next(block.mem_dec.parameters()).dtype

            # Hoist raw_h_stack outside the token loop — it doesn't change per token
            raw_h_stack = torch.stack(block.LUT.raw_hiddens).to(device=device)
            raw_h_stack_key = raw_h_stack.to(dtype=key_dtype)
            raw_h_stack_enc = raw_h_stack.to(dtype=enc_dtype)

            block_loss_accum = 0.0
            block_contrast_accum = 0.0
            block_token_count = 0

            in_phase1 = (training_step < phase1_steps)

            for k in range(len(encoded_label)):

                context = (encoded_ctx + encoded_label[:k])[-self.n_ctx:]
                if len(context) == 0:
                    continue

                context_tensor = torch.tensor(
                    context, dtype=torch.long, device=device
                ).unsqueeze(0)
                target_tensor = torch.tensor(
                    [encoded_label[k]], dtype=torch.long, device=device
                )

                T = context_tensor.size(1)
                position_ids = torch.arange(T, dtype=torch.long, device=device)

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

                with torch.no_grad():
                    h0 = self.tok_embeddings(context_tensor)
                    freqs_cis = self.freqs_cis[position_ids]

                    for block_idx in range(i):
                        h0 = self.layers[block_idx](h0, freqs_cis, position_ids, mask, use_cache=False)

                    _ = self.layers[i](h0, freqs_cis, position_ids, mask, use_cache=False)
                    pre_wnn_x_val = getattr(self.layers[i], "pre_wnn_x", None)

                if pre_wnn_x_val is None:
                    continue

                pre_wnn_x = pre_wnn_x_val.detach().clone()
                h_last = pre_wnn_x[0, -1, :]

                # Compute keys and query (needed for both phases)
                keys = block.mem_key(raw_h_stack_key).float()
                keys = F.layer_norm(keys, (keys.shape[-1],))
                keys = F.normalize(keys, dim=-1)

                q = block.mem_key(h_last.to(dtype=key_dtype)).float()
                q = F.layer_norm(q, (q.shape[-1],))
                q = F.normalize(q, dim=-1)

                # --- Contrastive loss (both phases) ---
                contrast_loss_val = 0.0
                contrast_loss = None
                if current_fact_id is not None and len(block.LUT.fact_ids) > 0:
                    unique_fids = list(set(fid for fid in block.LUT.fact_ids if fid is not None))
                    if len(unique_fids) > 1 and current_fact_id in unique_fids:
                        centroids = []
                        centroid_fids = []
                        for fid in unique_fids:
                            mask_fid = torch.tensor(
                                [1.0 if f == fid else 0.0 for f in block.LUT.fact_ids],
                                device=device
                            )
                            if mask_fid.sum() > 0:
                                centroid = (keys * mask_fid.unsqueeze(-1)).sum(0) / mask_fid.sum()
                                centroid = F.normalize(centroid, dim=-1)
                                centroids.append(centroid)
                                centroid_fids.append(fid)

                        if len(centroids) > 1 and current_fact_id in centroid_fids:
                            centroid_stack = torch.stack(centroids)
                            centroid_sims = F.cosine_similarity(
                                centroid_stack, q.unsqueeze(0).expand_as(centroid_stack), dim=-1
                            )
                            target_idx = centroid_fids.index(current_fact_id)
                            tau_contrast = 0.5
                            contrast_logits = centroid_sims / tau_contrast
                            contrast_target = torch.tensor(target_idx, device=device)
                            contrast_loss = F.cross_entropy(
                                contrast_logits.unsqueeze(0), contrast_target.unsqueeze(0)
                            )
                            contrast_loss_val = contrast_loss.item()

                if in_phase1:
                    # Phase 1: contrastive ONLY — skip CE forward entirely
                    if contrast_loss is not None:
                        total_loss = 5.0 * contrast_loss
                        (total_loss / len(encoded_label)).backward()
                    block_contrast_accum += contrast_loss_val
                    block_token_count += 1
                    continue

                # --- Phase 2: full CE + contrastive ---
                values = block.mem_enc(raw_h_stack_enc).float()

                sims = F.cosine_similarity(keys, q.unsqueeze(0).expand_as(keys), dim=-1)
                sims = torch.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
                k_topk = min(8, sims.shape[0])
                topk_sims, topk_idx = torch.topk(sims, k_topk)
                topk_values = values[topk_idx]
                weights = F.softmax(topk_sims / 0.5, dim=0)
                mem = torch.sum(weights.unsqueeze(-1) * topk_values, dim=0)
                sim_soft = torch.sum(weights * topk_sims)

                h_last_f32 = h_last.to(device=device, dtype=torch.float32)
                mem = mem.to(device=device, dtype=torch.float32)
                sim_t = sim_soft.to(device=device, dtype=torch.float32)

                delta = block.mem_dec(mem)
                delta = delta * (h_last_f32.norm() / (delta.norm() + 1e-8))

                gate_input = block.mem_key(h_last.to(dtype=key_dtype)).float()
                gate = torch.sigmoid(block.read_gate(gate_input)).squeeze(-1)

                h = pre_wnn_x
                inj = torch.zeros(h.shape, device=device, dtype=torch.float32)
                inj[:, -1, :] = delta.unsqueeze(0)

                h_f32 = h.float() + (gate * sim_t * block.residual_scale) * inj
                h_out = h_f32.to(h.dtype)

                for block_idx in range(i + 1, len(self.layers)):
                    h_out = self.layers[block_idx](h_out, freqs_cis, position_ids, mask, use_cache=False)

                logits = F.linear(self.norm(h_out.float()), self.output.weight.float())
                ce_loss = F.cross_entropy(logits[:, -1, :], target_tensor)

                if contrast_loss is not None:
                    total_loss = ce_loss + 2.0 * contrast_loss
                else:
                    total_loss = ce_loss

                (total_loss / len(encoded_label)).backward()

                block_loss_accum += ce_loss.item()
                block_contrast_accum += contrast_loss_val
                block_token_count += 1

            overall_losses += block_loss_accum
            overall_contrast += block_contrast_accum
            n_tokens_total += block_token_count

            # Reproject LUT keys/values ONCE per block (not per token)
            with torch.no_grad():
                if len(block.LUT.raw_hiddens) > 0:
                    new_keys = block.mem_key(raw_h_stack_key).float()
                    new_keys = F.layer_norm(new_keys, (new_keys.shape[-1],))
                    new_keys = F.normalize(new_keys, dim=-1)
                    new_keys = torch.nan_to_num(new_keys, nan=0.0, posinf=0.0, neginf=0.0)
                    block.LUT.keys = [k.detach() for k in new_keys]

                    new_vals = block.mem_enc(raw_h_stack_enc).float()
                    block.LUT.values = [v.detach() for v in new_vals]

        # Single optimizer step after all blocks and all tokens
        torch.nn.utils.clip_grad_norm_(self.lut_opt.param_groups[0]["params"], 1.0)
        self.lut_opt.step()

        for blk in self.layers:
            if blk.wnn_block:
                blk.use_wnn = True

        # Return loss dict
        avg_ce = overall_losses / max(n_tokens_total, 1)
        avg_contrast = overall_contrast / max(n_tokens_total, 1)
        return {"ce_loss": avg_ce, "contrastive_loss": avg_contrast, "total_loss": avg_ce + 0.5 * avg_contrast}

    def trainTransformations_episodic(self, tokenizer, lm_head, train_pairs, noise_std=0.02,
                                       training_step=0, phase1_steps=100):
        """Episodic training with two-phase support.
        train_pairs: list of (answer, question, fact_id) tuples.
        Returns: loss dict
        """
        import random as _random

        if self.lut_opt is None:
            raise RuntimeError("Call rebuild_lut_opt() before training")

        wnn_block_indices = [
            idx for idx, blk in enumerate(self.layers)
            if getattr(blk, "wnn_block", False)
        ]

        originals = {}
        if noise_std > 0:
            for i in wnn_block_indices:
                block = self.layers[i]
                originals[i] = [h.clone() for h in block.LUT.raw_hiddens]
                for j in range(len(block.LUT.raw_hiddens)):
                    noise = torch.randn_like(block.LUT.raw_hiddens[j]) * noise_std
                    block.LUT.raw_hiddens[j] = block.LUT.raw_hiddens[j] + noise

        answer, question, fact_id = _random.choice(train_pairs)

        loss = self.trainTransformations(tokenizer=tokenizer, lm_head=lm_head,
                                          label=answer, label_context=question,
                                          current_fact_id=fact_id,
                                          training_step=training_step,
                                          phase1_steps=phase1_steps)

        if noise_std > 0:
            for i in wnn_block_indices:
                self.layers[i].LUT.raw_hiddens = originals[i]

        return loss

    @torch.no_grad()
    def eval_retrieval(self, tokenizer, facts):
        """Evaluate retrieval hit@1 accuracy.
        facts: list of (label, context, fact_id) tuples.
        Returns dict with hit_at_1, avg_margin, per-fact results.
        """
        device = self.tok_embeddings.weight.device
        wnn_block_indices = [
            idx for idx, blk in enumerate(self.layers)
            if getattr(blk, "wnn_block", False)
        ]

        results = []
        for label, ctx, fact_id in facts:
            encoded = tokenizer.encode(ctx)
            if len(encoded) == 0:
                continue
            encoded = encoded[-self.n_ctx:]
            context_tensor = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
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

            h = self.tok_embeddings(context_tensor)
            freqs_cis = self.freqs_cis[position_ids]

            for block_idx in range(max(wnn_block_indices) + 1):
                self.layers[block_idx].use_wnn = False
                h = self.layers[block_idx](h, freqs_cis, position_ids, mask, use_cache=False)

                if block_idx in wnn_block_indices:
                    block = self.layers[block_idx]
                    pre_wnn_x_val = getattr(block, "pre_wnn_x", None)
                    if pre_wnn_x_val is None or len(block.LUT.keys) == 0:
                        continue

                    h_last = pre_wnn_x_val[0, -1, :]
                    key_dtype = next(block.mem_key.parameters()).dtype
                    q = block.mem_key(h_last.to(dtype=key_dtype)).float()
                    q = F.layer_norm(q, (q.shape[-1],))
                    q = F.normalize(q, dim=-1)

                    keys = torch.stack([k.to(device=device, dtype=torch.float32) for k in block.LUT.keys])
                    sims = F.cosine_similarity(keys, q.unsqueeze(0).expand_as(keys), dim=-1)
                    sims = torch.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)

                    top2_sims, top2_idx = torch.topk(sims, min(2, sims.shape[0]))
                    top1_idx = top2_idx[0].item()
                    top1_sim = top2_sims[0].item()
                    top2_sim = top2_sims[1].item() if top2_sims.shape[0] > 1 else 0.0

                    retrieved_fact_id = block.LUT.fact_ids[top1_idx]
                    hit = (retrieved_fact_id == fact_id)
                    margin = top1_sim - top2_sim

                    results.append({
                        "fact_id": fact_id,
                        "block": block_idx,
                        "hit": hit,
                        "top1_sim": top1_sim,
                        "margin": margin,
                        "retrieved_id": retrieved_fact_id,
                    })

        # Re-enable WNNs
        for blk in self.layers:
            if getattr(blk, "wnn_block", False):
                blk.use_wnn = True

        if not results:
            return {"hit_at_1": 0.0, "avg_margin": 0.0, "results": []}

        hits = sum(1 for r in results if r["hit"])
        avg_margin = sum(r["margin"] for r in results) / len(results)
        return {
            "hit_at_1": hits / len(results),
            "avg_margin": avg_margin,
            "results": results,
        }

    def rebuild_lut_opt(self, lr, weight_decay):
        # Freeze the ENTIRE model first — embeddings, attention, FFN, norm, output head
        self.requires_grad_(False)

        # Then selectively unfreeze only the memory MLPs on wnn blocks
        params = []
        for blk in self.layers:
            if getattr(blk, "wnn_block", False):
                for name in ("mem_enc", "mem_dec", "read_gate", "mem_key"):
                    mod = getattr(blk, name, None)
                    if mod is None:
                        continue
                    mod.float()
                    mod.requires_grad_(True)
                    params.extend(list(mod.parameters()))

        self.lut_opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        print("opt param count:", sum(p.numel() for p in params))

    def save_mlps(self, path):
        state = {}
        for i, blk in enumerate(self.layers):
            if getattr(blk, "wnn_block", False):
                for name in ("mem_key", "mem_enc", "mem_dec", "read_gate"):
                    mod = getattr(blk, name, None)
                    if mod is not None:
                        state[f"layers.{i}.{name}"] = mod.state_dict()
        torch.save(state, path)
        print(f"[save_mlps] Saved {len(state)} modules to {path}")

    def load_mlps(self, path, device="cuda"):
        state = torch.load(path, map_location=device)
        for i, blk in enumerate(self.layers):
            if getattr(blk, "wnn_block", False):
                for name in ("mem_key", "mem_enc", "mem_dec", "read_gate"):
                    key = f"layers.{i}.{name}"
                    mod = getattr(blk, name, None)
                    if mod is not None and key in state:
                        mod.float()
                        mod.load_state_dict(state[key])
                        print(f"[load_mlps] Loaded {key}")
        print("[load_mlps] Done")

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
    if isinstance(prompts, str):
        prompts = [prompts]
    else:
        prompts = list(prompts)

    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    device = "cuda"

    # Reset KV caches so stale data from trainLUT doesn't leak in
    for layer in model.layers:
        if hasattr(layer.attention, "reset_kv_cache"):
            layer.attention.reset_kv_cache(bsz=len(prompts))

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
    logits = model.forward(input_tokens[:, :min_prompt_len], positions, diffentiable_lookup=False)
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
            diffentiable_lookup=False,
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
