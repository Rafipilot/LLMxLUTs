import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file as safe_load
from sentencepiece import SentencePieceProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    sliding_window: int = 4096
    norm_eps: float = None
    vocab_size: int = None

    max_batch_size: int = 0
    rope_theta: float = 1000000.0

    # kept for compatibility, but we will infer key dim from n_kv_heads * head_dim
    lut_key_dim: int = 1024


def repeat_kv(keys, values, repeats):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=2)
    values = torch.repeat_interleave(values, repeats=repeats, dim=2)
    return keys, values


def _reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        freqs_cis.shape,
        (x.shape[1], x.shape[-1]),
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def _safe_l2_normalize(x, eps=1e-8):
    denom = torch.clamp(torch.norm(x, p=2, dim=-1, keepdim=True), min=eps)
    return x / denom


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = args.sliding_window
        self.scale = args.head_dim ** -0.5

        self.attn_scores = None

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

        self.cache_k = torch.empty(
            (args.max_batch_size, args.sliding_window, self.n_kv_heads, args.head_dim),
            dtype=torch.float16,
            device="cuda",
        )
        self.cache_v = torch.empty(
            (args.max_batch_size, args.sliding_window, self.n_kv_heads, args.head_dim),
            dtype=torch.float16,
            device="cuda",
        )

    @torch.no_grad()
    def reset_kv_cache(self, bsz):
        self.cache_k[:bsz].zero_()
        self.cache_v[:bsz].zero_()

    def forward(self, x, freqs_cis, positions, mask):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.args.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        scatter_pos = (positions[-self.sliding_window:] % self.sliding_window)[None, :, None, None]
        scatter_pos = scatter_pos.repeat(bsz, 1, self.n_kv_heads, self.args.head_dim)
        self.cache_k[:bsz].scatter_(dim=1, index=scatter_pos, src=xk[:, -self.sliding_window:])
        self.cache_v[:bsz].scatter_(dim=1, index=scatter_pos, src=xv[:, -self.sliding_window:])

        if positions.shape[0] > 1:
            key, value = repeat_kv(xk, xv, self.repeats)
        else:
            cur_pos = positions[-1].item() + 1
            key, value = repeat_kv(
                self.cache_k[:bsz, :cur_pos, ...],
                self.cache_v[:bsz, :cur_pos, ...],
                self.repeats,
            )

        query = xq.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = torch.matmul(query, key.transpose(2, 3)) * self.scale
        if mask is not None:
            scores = scores + mask[None, None, ...]

        scores = scores.float()
        scores = torch.softmax(scores, dim=-1).type_as(query)
        self.attn_scores = scores

        out = torch.matmul(scores, value)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        y = self._norm(x.float()).type_as(x)
        return y * self.weight


class LUT(nn.Module):
    def __init__(self):
        super().__init__()
        self.keys = []                 # CPU list of [r]
        self.mems = nn.ParameterList() # params [r]
        self.lookupTableMetaData = []
        self.CS_threshold = 0.25
        self.cost_scale = 0.0
        self.key_dim = None
        self.mem_dim = None

    # IMPORTANT: do NOT call this "train" (nn.Module already uses train/eval)
    def add_rows(self, xs, ms):
        for x, m in zip(xs, ms):
            x = x.detach().clone().squeeze().float()
            m = m.detach().clone().squeeze().float()

            if x.ndim != 1:
                x = x.reshape(-1)
            if m.ndim != 1:
                m = m.reshape(-1)

            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
            x = _safe_l2_normalize(x, eps=1e-8)

            if self.key_dim is None:
                self.key_dim = int(x.numel())
            if self.mem_dim is None:
                self.mem_dim = int(m.numel())
            if int(x.numel()) != int(self.key_dim):
                raise ValueError(f"LUT key dim mismatch: got {x.numel()} expected {self.key_dim}")
            if int(m.numel()) != int(self.mem_dim):
                raise ValueError(f"LUT mem dim mismatch: got {m.numel()} expected {self.mem_dim}")
            if self.key_dim != self.mem_dim:
                raise ValueError(f"Latent LUT expects key_dim == mem_dim, got {self.key_dim} vs {self.mem_dim}")

            self.keys.append(x.cpu())
            self.mems.append(nn.Parameter(m))
            self.lookupTableMetaData.append([1000, 0])

    def forward(self, q_in, topk=5, tau=0.05, differentiable=False):
        if q_in is None:
            return None, None

        q = q_in[-1, :].squeeze()
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).float()
        device = q.device

        if len(self.keys) == 0:
            if self.mem_dim is None:
                return torch.zeros(1, device=device), 0.0
            return torch.zeros(self.mem_dim, device=device), 0.0

        q = _safe_l2_normalize(q, eps=1e-8)
        keys = torch.stack([row.to(device=device, dtype=torch.float32) for row in self.keys])  # [N, r]
        mems = torch.stack([m.to(device=device, dtype=torch.float32) for m in self.mems])      # [N, r]

        N, r = keys.shape
        if q.numel() != r:
            raise ValueError(f"Query key dim mismatch: got {q.numel()} expected {r}")

        sims = F.cosine_similarity(keys, q.unsqueeze(0).expand(N, r), dim=-1)
        sims = torch.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)

        if differentiable:
            w = torch.softmax(sims / tau, dim=0)          # [N]
            m_read = (w[:, None] * mems).sum(dim=0)       # [r]
            sim_soft = (w * sims).sum()                   # scalar tensor
            return m_read, sim_soft

        costs = []
        for row in self.lookupTableMetaData:
            calls_since_last_hit = row[0]
            costs.append(1.0 / (calls_since_last_hit + 1.0))
        costs = torch.tensor(costs, device=device, dtype=torch.float32)

        sims = sims - self.cost_scale * costs

        k = min(topk, N)
        top_sims, top_idx = torch.topk(sims, k=k)
        highest_sim = top_sims.max().item()

        if highest_sim < self.CS_threshold:
            return torch.zeros(mems.shape[-1], device=device), 0.0

        w = torch.softmax(top_sims / tau, dim=-1)        # [k]
        v = mems.index_select(0, top_idx)                # [k, r]
        m_read = (w.unsqueeze(-1) * v).sum(dim=0)        # [r]

        best_i = top_idx[torch.argmax(top_sims)].item()
        self.lookupTableMetaData[best_i][0] = 0
        self.lookupTableMetaData[best_i][1] += 1
        for row in self.lookupTableMetaData:
            row[0] += 1

        return m_read, highest_sim

    def resetCosts(self):
        for row in self.lookupTableMetaData:
            row[0] = 1000
            row[1] = 0

    def resetLUT(self):
        self.keys = []
        self.mems = nn.ParameterList()
        self.lookupTableMetaData = []
        self.key_dim = None
        self.mem_dim = None


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.LUT = LUT()
        self.wnn_block = False
        self.use_wnn = False
        self.pre_wnn_x = None

        self.residual_scale = 0.6

        self.prompt_lut_key = None   # cached at prompt boundary
        self.forced_lut_key = None   # write-time forced routing
        self.key_blend = 0.85        # prompt anchor blend weight

        self.debug_lut = False

    def _compute_lut_key(self, pre_wnn_x, proj, lam=0.75, win=32):
        x = pre_wnn_x[0]  # [T, d]
        T, d = x.shape

        last = x[-1]
        tail = x[-min(win, T):]

        if tail.shape[0] > 1:
            ctx = tail[:-1].mean(dim=0)
        else:
            ctx = last

        h = lam * last + (1 - lam) * ctx
        h = F.layer_norm(h, (d,))
        h = _safe_l2_normalize(h, eps=1e-8)

        if proj is None:
            # Compatibility fallback if block is used in isolation.
            wd = self.attention.wk.weight.dtype
            hk = h.to(wd)
            k = self.attention.wk(hk).float()
        else:
            p = proj.to(device=h.device, dtype=h.dtype)
            k = torch.matmul(h, p).float()  # [r] = [dim] @ [dim, r]

        k = torch.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)
        k = F.layer_norm(k, (k.shape[-1],))
        k = _safe_l2_normalize(k, eps=1e-8)

        return k.unsqueeze(0)

    def _get_query_key(self, proj):
        if self.forced_lut_key is not None:
            return self.forced_lut_key

        q_dyn = self._compute_lut_key(self.pre_wnn_x, proj)
        if self.prompt_lut_key is None:
            return q_dyn

        a = float(self.key_blend)
        q = a * self.prompt_lut_key + (1.0 - a) * q_dyn

        q = q.squeeze(0)
        q = F.layer_norm(q, (q.shape[-1],))
        q = _safe_l2_normalize(q, eps=1e-8)
        return q.unsqueeze(0)

    def forward(self, x, freqs_cis, positions, mask, proj):
        r_attn = self.attention(self.attention_norm(x), freqs_cis, positions, mask)
        h = x + r_attn

        r_ffn = self.feed_forward(self.ffn_norm(h))
        base = h + r_ffn

        self.pre_wnn_x = base.detach()
        out = base

        if self.wnn_block and self.use_wnn and len(self.LUT.keys) > 0:
            key = self._get_query_key(proj)

            m_read, sim = self.LUT.forward(
                key,
                topk=5,
                tau=0.05,
                differentiable=torch.is_grad_enabled(),
            )

            # sim is float in inference mode, tensor in differentiable mode
            if torch.is_tensor(sim):
                if sim.item() <= 0.0:
                    return out
                scale = sim.to(device=out.device, dtype=out.dtype)
            else:
                if sim <= 0.0:
                    return out
                scale = torch.tensor(sim, device=out.device, dtype=out.dtype)

            if self.debug_lut and not torch.is_grad_enabled():
                print("[LUT] sim:", float(scale))

            if proj is None:
                return out

            p_t = proj.transpose(0, 1).to(device=out.device, dtype=out.dtype)  # [r, dim]
            delta = torch.matmul(m_read.to(device=out.device, dtype=out.dtype), p_t)  # [dim]

            res_tensor = torch.zeros_like(out)
            res_tensor[:, -1, :] = delta.unsqueeze(0)

            out = out + (scale * self.residual_scale) * res_tensor

        return out


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0

        self.lut_rank = int(args.lut_key_dim or (args.n_kv_heads * args.head_dim))
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.P = nn.Parameter(torch.empty(args.dim, self.lut_rank))
        nn.init.orthogonal_(self.P)

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(self.args.head_dim, 128_000).to("cuda")
        self.n_ctx = 128000

    @torch.no_grad()
    def reset_kv_caches(self, bsz):
        for blk in self.layers:
            blk.attention.reset_kv_cache(bsz)

    @torch.no_grad()
    def set_prompt_keys(self):
        for blk in self.layers:
            if not blk.wnn_block:
                continue
            if blk.pre_wnn_x is None:
                continue
            blk.prompt_lut_key = blk._compute_lut_key(blk.pre_wnn_x, self.P).detach()

    @torch.no_grad()
    def clear_prompt_keys(self):
        for blk in self.layers:
            blk.prompt_lut_key = None

    def forward(self, input_ids, positions):
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[positions]

        mask = None
        if input_ids.shape[1] > 1:
            seqlen = input_ids.shape[1]
            tensor = torch.full((seqlen, seqlen), dtype=h.dtype, fill_value=1, device=h.device)
            mask = torch.tril(tensor, diagonal=0).to(h.dtype)
            mask = torch.triu(mask, diagonal=-self.args.sliding_window)
            mask = torch.log(mask)

        for layer in self.layers:
            h = layer(h, freqs_cis, positions, mask, self.P)

        return self.output(self.norm(h)).float()

    def _normalize_block_indices(self, blocks):
        n = len(self.layers)
        if blocks is None:
            picked = [i for i, blk in enumerate(self.layers) if getattr(blk, "wnn_block", False)]
            return picked if picked else [n - 1]

        out = []
        seen = set()
        for b in blocks:
            idx = int(b)
            if idx < 0:
                idx = n + idx
            if 0 <= idx < n and idx not in seen:
                out.append(idx)
                seen.add(idx)
        if not out:
            raise ValueError(f"No valid block indices in: {blocks}")
        return out

    def _encode_answer_tokens(self, tokenizer, answer):
        ans_ids = tokenizer.encode(answer)
        eos_id = tokenizer.eos_id
        if eos_id is not None and int(eos_id) >= 0:
            ans_ids.append(int(eos_id))

        bos_id = None
        if hasattr(tokenizer, "_model") and hasattr(tokenizer._model, "bos_id"):
            bos_id = int(tokenizer._model.bos_id())
        if bos_id is not None and len(ans_ids) > 0 and ans_ids[0] == bos_id:
            ans_ids = ans_ids[1:]
        return ans_ids

    def _write_memory_from_prompt(
        self,
        tokenizer,
        prompt,
        answer,
        blocks,
        lr=5e-2,
        epochs=1,
        tune_projection=False,
        projection_lr=None,
    ):
        device = self.tok_embeddings.weight.device
        prompt_ids = tokenizer.encode(prompt)
        ans_ids = self._encode_answer_tokens(tokenizer, answer)
        block_ids = self._normalize_block_indices(blocks)
        if len(ans_ids) == 0:
            return {"avg_loss": 0.0, "steps": 0, "blocks": block_ids}

        # disable LUT everywhere for key collection pass
        prev_use_wnn = [blk.use_wnn for blk in self.layers]
        prev_wnn_block = [blk.wnn_block for blk in self.layers]
        prev_key_blend = [blk.key_blend for blk in self.layers]

        for blk in self.layers:
            blk.use_wnn = False
            blk.forced_lut_key = None
            blk.prompt_lut_key = None
            blk.pre_wnn_x = None

        self.reset_kv_caches(bsz=1)

        # run prompt once to populate pre_wnn_x
        with torch.no_grad():
            x = torch.tensor(prompt_ids, device=device, dtype=torch.long).unsqueeze(0)
            pos = torch.arange(x.size(1), device=device, dtype=torch.long)
            _ = self.forward(x, pos)

        mem_params = []
        for i in block_ids:
            blk = self.layers[i]
            blk.wnn_block = True
            blk.use_wnn = True
            blk.key_blend = 1.0

            key = blk._compute_lut_key(blk.pre_wnn_x, self.P).detach()  # [1, r]
            blk.prompt_lut_key = key
            blk.forced_lut_key = key.detach()

            m0 = torch.zeros(1, self.lut_rank, device=device, dtype=torch.float32)
            blk.LUT.add_rows(key, m0)
            mem_params.append(blk.LUT.mems[-1])

        opt_groups = [{"params": mem_params, "lr": float(lr)}]
        prev_proj_grad = bool(self.P.requires_grad)
        if tune_projection:
            self.P.requires_grad_(True)
            opt_groups.append(
                {
                    "params": [self.P],
                    "lr": float(projection_lr if projection_lr is not None else (0.1 * lr)),
                }
            )
        opt = torch.optim.AdamW(opt_groups, weight_decay=0.0)

        loss_sum = 0.0
        n_steps = 0
        self.train()
        for _ in range(max(1, int(epochs))):
            for t in range(len(ans_ids)):
                self.reset_kv_caches(bsz=1)

                ctx_ids = prompt_ids + ans_ids[:t]
                x = torch.tensor(ctx_ids, device=device, dtype=torch.long).unsqueeze(0)
                y = torch.tensor([ans_ids[t]], device=device, dtype=torch.long)

                pos = torch.arange(x.size(1), device=device, dtype=torch.long)
                logits = self.forward(x, pos)
                loss = F.cross_entropy(logits[:, -1, :], y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mem_params, 1.0)
                if tune_projection:
                    torch.nn.utils.clip_grad_norm_([self.P], 1.0)
                opt.step()
                loss_sum += float(loss.item())
                n_steps += 1

        for i, blk in enumerate(self.layers):
            blk.use_wnn = prev_use_wnn[i]
            blk.wnn_block = prev_wnn_block[i]
            blk.key_blend = prev_key_blend[i]
            blk.forced_lut_key = None
            blk.prompt_lut_key = None
        for i in block_ids:
            self.layers[i].wnn_block = True
            self.layers[i].use_wnn = True
        self.P.requires_grad_(prev_proj_grad)
        self.eval()

        return {
            "avg_loss": (loss_sum / n_steps) if n_steps > 0 else 0.0,
            "steps": n_steps,
            "blocks": block_ids,
        }

    def write_memory_latent(
        self,
        tokenizer,
        question,
        answer,
        blocks,
        lr=5e-2,
        epochs=1,
        tune_projection=False,
        projection_lr=None,
    ):
        prompt = f"User: {question}\nAssistant: "
        return self._write_memory_from_prompt(
            tokenizer=tokenizer,
            prompt=prompt,
            answer=answer,
            blocks=blocks,
            lr=lr,
            epochs=epochs,
            tune_projection=tune_projection,
            projection_lr=projection_lr,
        )

    # Compatibility API used by api.py.
    def trainLUT(
        self,
        tokenizer,
        lm_head=None,
        label="",
        label_context=None,
        sparsity_level=1.0,
        blocks=None,
        lr=8e-2,
        epochs=12,
        tune_projection=False,
        projection_lr=None,
    ):
        del lm_head, sparsity_level

        if label_context is None:
            prompt = "User: \nAssistant: "
        else:
            prompt = str(label_context)
            if "Assistant:" not in prompt and not prompt.rstrip().endswith("[/INST]"):
                prompt = f"User: {prompt.strip()}\nAssistant: "

        return self._write_memory_from_prompt(
            tokenizer=tokenizer,
            prompt=prompt,
            answer=str(label),
            blocks=blocks,
            lr=lr,
            epochs=epochs,
            tune_projection=tune_projection,
            projection_lr=projection_lr,
        )

    @staticmethod
    def from_folder(folder, max_batch_size=1, device="cuda", dtype=torch.float16):
        with open(folder / "params.json", "r") as f:
            model_args = ModelArgs(**json.loads(f.read()))
        model_args.max_batch_size = max_batch_size

        model = Transformer(model_args).to(device=device, dtype=dtype)

        loaded = None
        try:
            try:
                loaded = torch.load(folder / "consolidated.00.pth", map_location="cpu", weights_only=True)
            except TypeError:
                loaded = torch.load(folder / "consolidated.00.pth", map_location="cpu")
        except Exception:
            loaded = safe_load(str(folder / "consolidated.safetensors"))

        missing, unexpected = model.load_state_dict(loaded, strict=False)
        if missing:
            print("[from_folder] Missing keys:", missing[:5], "...", len(missing))
        if unexpected:
            print("[from_folder] Unexpected keys:", unexpected[:5], "...", len(unexpected))

        for p in model.parameters():
            p.requires_grad = False

        return model


class Tokenizer:
    def __init__(self, model_path):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def eos_id(self):
        return self._model.eos_id()

    @property
    def pad_id(self):
        return self._model.pad_id()

    def encode(self, s):
        return [self._model.bos_id(), *self._model.encode(s)]

    def decode(self, t):
        return self._model.decode(t)


@torch.no_grad()
def generate(prompts, model, tokenizer, max_tokens):
    encoded_prompts = [tokenizer.encode(p) for p in prompts]
    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    device = "cuda"
    bsz = len(prompts)

    model.reset_kv_caches(bsz)
    model.clear_prompt_keys()

    input_tokens = torch.full(
        (bsz, max_prompt_len),
        tokenizer.pad_id,
        dtype=torch.long,
        device=device,
    )
    for i, enc in enumerate(encoded_prompts):
        input_tokens[i, : len(enc)] = torch.tensor(enc, device=device)

    input_mask = input_tokens != tokenizer.pad_id

    positions = torch.arange(0, min_prompt_len, device=device)
    logits = model.forward(input_tokens[:, :min_prompt_len], positions)
    logprobs = F.log_softmax(logits, dim=-1)

    # cache prompt boundary keys for the enabled LUT blocks
    model.set_prompt_keys()

    generated = []
    cur_pos = min_prompt_len

    eos_id = tokenizer.eos_id
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)

    for _ in range(max_tokens):
        sampled = torch.argmax(logprobs[:, -1, :], dim=-1)

        if cur_pos < input_mask.shape[1]:
            is_prompt_pos = input_mask[:, cur_pos]
            next_token = torch.where(is_prompt_pos, input_tokens[:, cur_pos], sampled)
        else:
            is_prompt_pos = torch.zeros_like(sampled, dtype=torch.bool, device=device)
            next_token = sampled

        if eos_id is not None:
            eos_hit = (~is_prompt_pos) & (~finished) & (next_token == eos_id)
            finished = finished | eos_hit

        generated.append(next_token[:, None])

        logits = model.forward(next_token[:, None], torch.LongTensor([cur_pos]).to(next_token))
        logprobs = F.log_softmax(logits, dim=-1)
        cur_pos += 1

        if finished.all():
            break

    model.clear_prompt_keys()

    res = []
    if max_tokens > 0 and generated:
        generated = torch.cat(generated, dim=1)
        for i in range(bsz):
            gen_tokens = generated[i].tolist()
            if eos_id is not None and eos_id in gen_tokens:
                gen_tokens = gen_tokens[: gen_tokens.index(eos_id)]
            res.append(tokenizer.decode(gen_tokens))

    return res
