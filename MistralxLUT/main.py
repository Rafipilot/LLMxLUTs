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

    def reset_kv_cache(self, bsz):
        self.cache_k[:bsz].zero_()
        self.cache_v[:bsz].zero_()

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
    def __init__(self, value_dim):
        self.keys = []
        self.values = []
        self.lookupTableMetaData = [] # idx 0 calls since last response, idx 1 number of calls
        self.CS_threshold = 0.25
        self.value_dim = value_dim

    def train(self, xs, ys):
        # xs, ys: [B, d] or iterable of [d]
        for x, y in zip(xs, ys):
            print("adding rows...")
            x = x.detach().clone().squeeze()
            y = y.detach().clone().squeeze()
            self.keys.append(x)
            self.values.append(y)
            self.lookupTableMetaData.append([1000, 0])

    
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

        sims = F.cosine_similarity(keys, q.unsqueeze(0).expand(keys.shape[0], keys.shape[1]), dim=-1)
        sims = torch.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)

        max_idx = torch.argmax(sims)
        best = values[max_idx]
        best_sim = sims[max_idx].item()

        # print("Best sim: ", best_sim)

        if best_sim < self.CS_threshold:
            return torch.zeros_like(best), 0.0

        return best.to(device), best_sim
    
    def forward_differentiable(self, q):
        if q is None:
            return None, None

        q = q.squeeze()
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).float()

        if len(self.keys) == 0:
            return torch.zeros(self.value_dim, device=q.device), None

        device = q.device
        keys = torch.stack([row.to(device=device, dtype=torch.float32) for row in self.keys])
        values = torch.stack([row.to(device=device, dtype=torch.float32) for row in self.values])

        sims = F.cosine_similarity(keys, q.unsqueeze(0).expand(keys.shape[0], keys.shape[1]), dim=-1)
        sims = torch.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)

        weights = F.softmax(sims / 0.05, dim=0)
        best = torch.sum(weights.unsqueeze(-1) * values, dim=0)
        sim_soft = torch.sum(weights * sims)

        return best.to(device), sim_soft
    
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
        self.LUT = LUT(args.dim)
        self.wnn_block = wnn_block # we only activate the wnn block in the last n layers as specified by num_wnn_blocks
        if self.wnn_block:
            self.use_wnn = True
        else:
            self.use_wnn = False
        self.pre_wnn_x = None  
        self.residual_scale = 1   # or 15, but be consistent everywhere

        # new approach create three mlps: memory_encoder (h_last -> memory), memory_decoder (memory + h_last -> residual_correction) and the memory key (h_last -> key)

        self.key_dim = args.dim // 4 # to be tuned tho this is probs fine for now

        self.mem_rank = 32

        self.mem_key = nn.Sequential(nn.Linear(args.dim, self.mem_rank, bias=False), nn.SiLU(), nn.Linear(self.mem_rank, self.key_dim, bias=False))
        self.mem_enc = nn.Sequential(nn.Linear(args.dim, args.dim, bias=False), nn.SiLU(), nn.Linear(args.dim, args.dim, bias=False))
        self.mem_dec = nn.Sequential(nn.Linear(2 * args.dim, self.mem_rank, bias=False), nn.SiLU(), nn.Linear(self.mem_rank, args.dim, bias=False))

        self.read_gate = nn.Linear(args.dim, 1, bias=True)

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
        diffentiable_lookup=False
    ) -> torch.Tensor:
        # Standard transformer block forward
        r_attn = self.attention(self.attention_norm(x), freqs_cis, positions, mask)
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
            gate_dtype = next(self.read_gate.parameters()).dtype

            h_last_key = self.pre_wnn_x[0, -1, :].to(dtype=key_dtype)
            h_last_dec = self.pre_wnn_x[0, -1, :].to(dtype=dec_dtype)
            h_last_gate = self.pre_wnn_x[0, -1, :].to(dtype=gate_dtype)

            q = self.mem_key(h_last_key).float()
            q = F.layer_norm(q, (q.shape[-1],))
            q = F.normalize(q, dim=-1)

            if diffentiable_lookup:
                mem, sim = self.LUT.forward_differentiable(q)
            else:
                mem, sim = self.LUT.forward(q)

            if mem is None or sim is None:
                return out
            if not torch.is_tensor(sim):
                if sim <= 0.0:
                    return out
                sim_t = torch.tensor(sim, device=out.device, dtype=dec_dtype)
            else:
                sim_t = sim.to(device=out.device, dtype=dec_dtype)

            mem = mem.to(dtype=dec_dtype)
            dec_in = torch.cat([mem, h_last_dec], dim=-1)
            delta = self.mem_dec(dec_in)

            gate = torch.sigmoid(self.read_gate(h_last_gate)).squeeze(-1).to(dec_dtype)
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
            encoded_ctx.insert(0, 4)
            encoded_ctx.append(5)
            
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

                with torch.no_grad():
                    key_dtype = next(block.mem_key.parameters()).dtype
                    enc_dtype = next(block.mem_enc.parameters()).dtype

                    h_last_key = pre_wnn_x_val[0, -1, :].to(dtype=key_dtype)
                    h_last_enc = pre_wnn_x_val[0, -1, :].to(dtype=enc_dtype)

                    key_vec = block.mem_key(h_last_key).float()
                    key_vec = F.layer_norm(key_vec, (key_vec.shape[-1],))
                    key_vec = F.normalize(key_vec, dim=-1).unsqueeze(0)

                    val_vec = pre_wnn_x_val[0, -1, :].float().unsqueeze(0)
                    block.LUT.train(key_vec, val_vec)

            # block.use_wnn = True # re enable this lut block

            print(f"[trainLUT] Finished block {i} in {datetime.now() - now_block}")

        # Re-enable LUT for inference
        for blk in self.layers:
            blk.use_wnn = True  # this should be redundant

    def trainTransformations(self, tokenizer, lm_head, label, label_context=None):
        if self.lut_opt is None:
            self.rebuild_lut_opt(lr=1e-5)

        for blk in self.layers:
            blk.use_wnn = False
            blk.pre_wnn_x = None
            if hasattr(blk.attention, "reset_kv_cache"):
                blk.attention.reset_kv_cache(bsz=1)

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

        for i in wnn_block_indices:
            block = self.layers[i]
            now_block = datetime.now()
            block.use_wnn = False

            for k in range(len(encoded_label)):
                for blk in self.layers:
                    if hasattr(blk.attention, "reset_kv_cache"):
                        blk.attention.reset_kv_cache(bsz=1)

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


                h0 = self.tok_embeddings(context_tensor)
                freqs_cis = self.freqs_cis[position_ids]

                for block_idx in range(i):
                    h0 = self.layers[block_idx](h0, freqs_cis, position_ids, mask)

                _ = self.layers[i](h0, freqs_cis, position_ids, mask)
                pre_wnn_x_val = getattr(self.layers[i], "pre_wnn_x", None)

                if pre_wnn_x_val is None:
                    continue

                pre_wnn_x = pre_wnn_x_val.detach().clone()
                h = pre_wnn_x

                key_dtype = next(block.mem_key.parameters()).dtype
                dec_dtype = next(block.mem_dec.parameters()).dtype
                gate_dtype = next(block.read_gate.parameters()).dtype

                h_last_key = pre_wnn_x[0, -1, :].to(device=device, dtype=key_dtype)
                h_last_dec = pre_wnn_x[0, -1, :].to(device=device, dtype=dec_dtype)
                h_last_gate = pre_wnn_x[0, -1, :].to(device=device, dtype=gate_dtype)

                q = block.mem_key(h_last_key).float()
                q = F.layer_norm(q, (q.shape[-1],))
                q = F.normalize(q, dim=-1)

                mem, sim = block.LUT.forward_differentiable(q)

                if mem is None or sim is None:
                    continue

                mem = mem.to(device=device, dtype=dec_dtype)
                sim_t = sim.to(device=device, dtype=dec_dtype)

                dec_in = torch.cat([mem, h_last_dec], dim=-1)
                delta = block.mem_dec(dec_in)
                gate = torch.sigmoid(block.read_gate(h_last_gate)).squeeze(-1).to(dec_dtype)

                inj = torch.zeros_like(h)
                inj[:, -1, :] = delta.to(h.dtype).unsqueeze(0)

                h = h + (gate * sim_t * block.residual_scale) * inj

                for block_idx in range(i + 1, len(self.layers)):
                    h = self.layers[block_idx](h, freqs_cis, position_ids, mask)

                logits = self.output(self.norm(h)).float()
                loss = F.cross_entropy(logits[:, -1, :], target_tensor)

                self.lut_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lut_opt.param_groups[0]["params"], 1.0)
                self.lut_opt.step()

                self.layers[i].pre_wnn_x = None

            print(f"[trainTransformations] Finished block {i} in {datetime.now() - now_block}")

        for blk in self.layers:
            if blk.wnn_block:
                blk.use_wnn = True

    def rebuild_lut_opt(self, lr=1e-3):
        params = []

        for blk in self.layers:
            if getattr(blk, "wnn_block", False):
                for name in ("mem_dec", "read_gate", "mem_key"):
                    mod = getattr(blk, name, None)
                    if mod is None:
                        continue

                    for p in mod.parameters():
                        p.requires_grad = True
                    params += list(mod.parameters())

            else:
                for name in ("mem_key", "mem_enc", "mem_dec", "read_gate"):
                    mod = getattr(blk, name, None)
                    if mod is None:
                        continue
                    for p in mod.parameters():
                        p.requires_grad = False

        self.lut_opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        print("opt param count:", sum(p.numel() for p in params))


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