import copy
import random
import sqlite3
import pickle
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

from GPT2xLUT.GPT2.model import GPT2LMHeadModel
from GPT2xLUT.GPT2.utils import load_weight
from GPT2xLUT.GPT2.config import GPT2Config
from GPT2xLUT.GPT2.sample import sample_sequence
from GPT2xLUT.GPT2.encoder import get_encoder

from MistralxLUT.main import Tokenizer, Transformer, generate

import gc
import time
import traceback  # NEW


# =========================
# Global config / seeds
# =========================
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).parent
DB_PATH = "LUT.db"

# =========================
# GPT-2 globals
# =========================
MODEL = None
LM_HEAD = None
CONFIG = None
ENC = None
TEMPERATURE = 0.7
ENABLE_GPT2 = False

# =========================
# Mistral globals
# =========================
MISTRAL_PATH = BASE_DIR / "MistralxLUT" / "mistral-7B-Instruct-v0.3"
MISTRAL_MODEL = None
MISTRAL_TOKENIZER = None

# =========================
# Empty LUT templates (to avoid cross-user sharing)
# =========================
EMPTY_LUT_TEMPLATES_GPT2 = {}      # block_idx -> {slot: empty_LUT_copy}
EMPTY_LUT_TEMPLATES_MISTRAL = {}   # block_idx -> {slot: empty_LUT_copy}


# =========================
# GPU memory logging helper
# =========================
def _log_cuda_mem(tag: str, reset_peak: bool = False):
    if not torch.cuda.is_available():
        print(f"[mem] {tag}: CUDA not available")
        return

    dev = torch.cuda.current_device()

    def to_gib(x: int) -> float:
        return x / (1024 ** 3)

    allocated = to_gib(torch.cuda.memory_allocated(dev))
    reserved = to_gib(torch.cuda.memory_reserved(dev))
    max_allocated = to_gib(torch.cuda.max_memory_allocated(dev))

    print(
        f"[mem] {tag}: "
        f"allocated={allocated:.2f} GiB, "
        f"reserved={reserved:.2f} GiB, "
        f"max_allocated={max_allocated:.2f} GiB"
    )

    if reset_peak:
        torch.cuda.reset_peak_memory_stats()


# =========================
# Conditional offload on GPU pressure
# =========================
GPU_OFFLOAD_RATIO = float(os.getenv("GPU_OFFLOAD_RATIO", "0.95"))
GPU_CACHE_TRY_CLEAR_RATIO = float(os.getenv("GPU_CACHE_TRY_CLEAR_RATIO", "0.75"))
_OFFLOAD_COOLDOWN_S = float(os.getenv("GPU_OFFLOAD_COOLDOWN_S", "3.0"))
_last_offload_ts = 0.0


def _gpu_pressure(tag: str = "") -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False, "tag": tag}

    dev = torch.cuda.current_device()
    free_b, total_b = torch.cuda.mem_get_info(dev)
    used_ratio = 1.0 - (free_b / max(total_b, 1))

    allocated = torch.cuda.memory_allocated(dev)
    reserved = torch.cuda.memory_reserved(dev)

    return {
        "cuda": True,
        "tag": tag,
        "free_b": int(free_b),
        "total_b": int(total_b),
        "used_ratio": float(used_ratio),
        "allocated_ratio": float(allocated / max(total_b, 1)),
        "reserved_ratio": float(reserved / max(total_b, 1)),
    }


def _maybe_offload_mistral(tag: str = "") -> bool:
    global _last_offload_ts

    if not torch.cuda.is_available():
        return False

    now = time.time()
    if now - _last_offload_ts < _OFFLOAD_COOLDOWN_S:
        return False

    p = _gpu_pressure(tag)
    if not p.get("cuda"):
        return False

    print(
        f"[mem-check] {tag}: "
        f"driver_used={p['used_ratio']:.1%}, "
        f"allocated={p['allocated_ratio']:.1%}, "
        f"reserved={p['reserved_ratio']:.1%}"
    )

    if p["used_ratio"] >= GPU_CACHE_TRY_CLEAR_RATIO and p["allocated_ratio"] < (GPU_CACHE_TRY_CLEAR_RATIO * 0.85):
        torch.cuda.empty_cache()
        p2 = _gpu_pressure(tag + ":after_empty_cache")
        print(
            f"[mem-check] {tag}: after empty_cache -> "
            f"driver_used={p2['used_ratio']:.1%}, "
            f"allocated={p2['allocated_ratio']:.1%}, "
            f"reserved={p2['reserved_ratio']:.1%}"
        )
        p = p2

    if p["used_ratio"] >= GPU_OFFLOAD_RATIO:
        print(f"[mem-check] {tag}: >= {GPU_OFFLOAD_RATIO:.0%} -> offloading Mistral")
        _free_mistral()
        gc.collect()
        torch.cuda.empty_cache()
        _last_offload_ts = now
        _log_cuda_mem(f"{tag}: after offload", reset_peak=True)
        return True

    return False


def list_lut_metadata(lut_name: str | None = None):
    conn = _get_conn()
    cur = conn.cursor()

    if lut_name:
        cur.execute(
            """
            SELECT
                lut_name,
                COUNT(*)                       AS num_rows,
                COUNT(DISTINCT block_idx)      AS num_blocks,
                COUNT(DISTINCT lut_slot)       AS num_slots,
                SUM(LENGTH(lut_blob))          AS total_bytes
            FROM lut_blocks
            WHERE lut_name = ?
            GROUP BY lut_name
            """,
            (lut_name,),
        )
    else:
        cur.execute(
            """
            SELECT
                lut_name,
                COUNT(*)                       AS num_rows,
                COUNT(DISTINCT block_idx)      AS num_blocks,
                COUNT(DISTINCT lut_slot)       AS num_slots,
                SUM(LENGTH(lut_blob))          AS total_bytes
            FROM lut_blocks
            GROUP BY lut_name
            """
        )

    rows = cur.fetchall()
    conn.close()

    results = []
    for row in rows:
        name, num_rows, num_blocks, num_slots, total_bytes = row
        total_bytes = total_bytes or 0
        results.append({
            "lut_name": name,
            "num_rows": int(num_rows),
            "num_blocks": int(num_blocks),
            "num_slots": int(num_slots),
            "total_bytes": int(total_bytes),
            "approx_mb": round(total_bytes / (1024 * 1024), 4),
        })
    return results


# =========================
# SQLite helpers
# =========================
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lut_blocks (
            lut_name   TEXT NOT NULL,
            block_idx  INTEGER NOT NULL,
            lut_slot   INTEGER NOT NULL,
            lut_blob   BLOB NOT NULL,
            PRIMARY KEY (lut_name, block_idx, lut_slot)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lut_params (
            lut_name   TEXT NOT NULL,
            block_idx  INTEGER NOT NULL,
            param_blob BLOB NOT NULL,
            PRIMARY KEY (lut_name, block_idx)
        )
        """
    )

    return conn


def _free_gpt2():
    global MODEL, LM_HEAD, CONFIG, ENC, TEMPERATURE, ENABLE_GPT2, EMPTY_LUT_TEMPLATES_GPT2

    if MODEL is not None:
        try:
            MODEL.to("cpu")
        except Exception:
            pass

    MODEL = None
    LM_HEAD = None
    CONFIG = None
    ENC = None
    TEMPERATURE = 0.7
    ENABLE_GPT2 = False

    if EMPTY_LUT_TEMPLATES_GPT2:
        EMPTY_LUT_TEMPLATES_GPT2.clear()


def _free_mistral():
    global MISTRAL_MODEL, MISTRAL_TOKENIZER, EMPTY_LUT_TEMPLATES_MISTRAL

    if MISTRAL_MODEL is not None:
        try:
            MISTRAL_MODEL.to("cpu")
        except Exception:
            pass

    MISTRAL_MODEL = None
    MISTRAL_TOKENIZER = None

    if EMPTY_LUT_TEMPLATES_MISTRAL:
        EMPTY_LUT_TEMPLATES_MISTRAL.clear()


# =========================
# Transformer / block helpers
# =========================
def _get_blocks(transformer):
    if hasattr(transformer, "h"):
        return transformer.h
    if hasattr(transformer, "layers"):
        return transformer.layers
    return []


def _normalize_block_indices(blocks, indices):
    n = len(blocks)
    norm = []
    for i in indices:
        i = int(i)
        if i < 0:
            i = n + i
        if 0 <= i < n:
            norm.append(i)
    return norm


def _set_wnn_blocks(transformer, active_indices=None):
    if active_indices is None:
        print("[wnn] _set_wnn_blocks called with None -> leaving config as-is")
        return

    blocks = _get_blocks(transformer)
    norm_indices = _normalize_block_indices(blocks, active_indices)
    if not norm_indices:
        print(f"[wnn] _set_wnn_blocks got empty/invalid indices: {active_indices}")
        return

    active_set = set(norm_indices)
    print(f"[wnn] Activating WNN blocks (normalized): {sorted(active_set)}")

    for idx, block in enumerate(blocks):
        if hasattr(block, "wnn_block"):
            is_active = idx in active_set
            block.wnn_block = is_active
            if hasattr(block, "use_wnn"):
                block.use_wnn = is_active

    if hasattr(transformer, "rebuild_lut_opt"):
        transformer.rebuild_lut_opt()


def _apply_lut_hyperparams(
    transformer,
    threshold=None,
    residual=None,
    wnn_blocks=None,
    cost_scale: float = 5.0,
):
    blocks = _get_blocks(transformer)
    if not blocks:
        return

    if wnn_blocks is not None:
        target_indices = _normalize_block_indices(blocks, wnn_blocks)
    else:
        target_indices = [
            idx for idx, blk in enumerate(blocks)
            if hasattr(blk, "LUT") or hasattr(blk, "LUTs")
        ]

    if not target_indices:
        return

    residual_list = None
    if isinstance(residual, (list, tuple)):
        residual_list = list(residual)
        # CHANGED: never crash here — fallback to first element
        if len(residual_list) != len(target_indices):
            print(
                f"[warn] residual list length ({len(residual_list)}) != "
                f"selected blocks ({len(target_indices)}). Falling back to residual[0]."
            )
            residual = float(residual_list[0]) if len(residual_list) > 0 else None
            residual_list = None

    for pos, block_idx in enumerate(target_indices):
        block = blocks[block_idx]

        if threshold is not None:
            if hasattr(block, "LUT"):
                block.LUT.CS_threshold = float(threshold)
                if hasattr(block.LUT, "cost_scale"):
                    block.LUT.cost_scale = float(cost_scale)
            elif hasattr(block, "LUTs"):
                container = block.LUTs
                if isinstance(container, (list, dict)):
                    for lut_obj in (container if isinstance(container, list) else container.values()):
                        if lut_obj is not None:
                            lut_obj.CS_threshold = float(threshold)
                            if hasattr(lut_obj, "cost_scale"):
                                lut_obj.cost_scale = float(cost_scale)

        if residual is not None and hasattr(block, "residual_scale"):
            if residual_list is None:
                block.residual_scale = float(residual)
            else:
                block.residual_scale = float(residual_list[pos])


def _snapshot_empty_luts(transformer, model_type: str):
    global EMPTY_LUT_TEMPLATES_GPT2, EMPTY_LUT_TEMPLATES_MISTRAL

    blocks = _get_blocks(transformer)
    templates = {}

    for idx, block in enumerate(blocks):
        if hasattr(block, "LUTs"):
            if isinstance(block.LUTs, list):
                slot_map = {}
                for slot, lut_obj in enumerate(block.LUTs):
                    if lut_obj is not None:
                        slot_map[slot] = copy.deepcopy(lut_obj)
                if slot_map:
                    templates[idx] = slot_map

            elif isinstance(block.LUTs, dict):
                slot_map = {}
                for slot, lut_obj in block.LUTs.items():
                    if lut_obj is not None:
                        slot_map[slot] = copy.deepcopy(lut_obj)
                if slot_map:
                    templates[idx] = slot_map

        elif hasattr(block, "LUT") and block.LUT is not None:
            templates[idx] = {0: copy.deepcopy(block.LUT)}

    if model_type == "gpt2":
        EMPTY_LUT_TEMPLATES_GPT2 = templates
    elif model_type == "mistral":
        EMPTY_LUT_TEMPLATES_MISTRAL = templates


# =========================
# Param storage helpers
# =========================
def _block_has_trainable_params(block) -> bool:
    if hasattr(block, "lut_key_proj") and isinstance(block.lut_key_proj, torch.nn.Parameter):
        if block.lut_key_proj.requires_grad:
            return True
    for name in ("lut_v_down", "lut_v_up", "lut_v_gate"):
        if hasattr(block, name):
            mod = getattr(block, name)
            try:
                if any(p.requires_grad for p in mod.parameters()):
                    return True
            except Exception:
                pass
    return False


def _block_should_store_params(block) -> bool:
    if getattr(block, "wnn_block", False) or getattr(block, "use_wnn", False):
        return True
    return _block_has_trainable_params(block)


def _extract_block_params(block) -> dict:
    out = {}

    if hasattr(block, "lut_key_proj") and isinstance(block.lut_key_proj, torch.nn.Parameter):
        out["lut_key_proj"] = block.lut_key_proj.detach().float().cpu()

    for name in ("lut_v_down", "lut_v_up", "lut_v_gate"):
        if hasattr(block, name):
            mod = getattr(block, name)
            try:
                sd = {k: v.detach().float().cpu() for k, v in mod.state_dict().items()}
                out[name] = sd
            except Exception:
                pass

    return out


def _apply_block_params(block, payload: dict):
    if not isinstance(payload, dict):
        return

    with torch.no_grad():
        if "lut_key_proj" in payload and hasattr(block, "lut_key_proj") and isinstance(block.lut_key_proj, torch.nn.Parameter):
            src = payload["lut_key_proj"]
            if isinstance(src, torch.Tensor):
                src = src.to(device=block.lut_key_proj.device, dtype=block.lut_key_proj.dtype)
                if block.lut_key_proj.shape == src.shape:
                    block.lut_key_proj.copy_(src)

        for name in ("lut_v_down", "lut_v_up", "lut_v_gate"):
            if name in payload and hasattr(block, name):
                mod = getattr(block, name)
                sd = payload.get(name, {})
                if isinstance(sd, dict):
                    try:
                        dev = next(mod.parameters()).device
                        dtype = next(mod.parameters()).dtype
                    except Exception:
                        dev = None
                        dtype = None

                    fixed = {}
                    for k, v in sd.items():
                        if isinstance(v, torch.Tensor):
                            if dev is not None and dtype is not None:
                                fixed[k] = v.to(device=dev, dtype=dtype)
                            elif dev is not None:
                                fixed[k] = v.to(device=dev)
                            else:
                                fixed[k] = v
                        else:
                            fixed[k] = v

                    try:
                        mod.load_state_dict(fixed, strict=False)
                    except Exception:
                        pass


def load_params_for_user(transformer, lut_name: str):
    if not lut_name:
        return transformer

    blocks = _get_blocks(transformer)
    if not blocks:
        return transformer

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT block_idx, param_blob FROM lut_params WHERE lut_name = ?",
        (lut_name,)
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return transformer

    for block_idx, blob in rows:
        if block_idx < 0 or block_idx >= len(blocks):
            continue

        blk = blocks[block_idx]
        if not _block_should_store_params(blk):
            continue

        try:
            payload = pickle.loads(blob)
        except Exception:
            continue

        _apply_block_params(blk, payload)

    return transformer


def save_params_for_user(transformer, lut_name: str):
    if not lut_name:
        return transformer

    blocks = _get_blocks(transformer)
    if not blocks:
        return transformer

    conn = _get_conn()
    cur = conn.cursor()

    saved = 0
    for block_idx, blk in enumerate(blocks):
        if not _block_should_store_params(blk):
            continue

        payload = _extract_block_params(blk)
        if not payload:
            continue

        try:
            blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            continue

        cur.execute(
            """
            INSERT OR REPLACE INTO lut_params
            (lut_name, block_idx, param_blob)
            VALUES (?, ?, ?)
            """,
            (lut_name, int(block_idx), blob)
        )
        saved += 1

    conn.commit()
    conn.close()
    print(f"[save_params_for_user] saved params for {saved} blocks (lut_name={lut_name})")
    return transformer


# =========================
# Per-user LUT DB hooks
# =========================
def load_lut_for_user(model, lut_name: str):
    if not lut_name:
        return model

    transformer = getattr(model, "transformer", model)

    conn = _get_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT block_idx, lut_slot, lut_blob FROM lut_blocks WHERE lut_name = ?",
        (lut_name,)
    )
    rows = cur.fetchall()
    conn.close()

    blocks = _get_blocks(transformer)
    if not blocks:
        return model

    if not rows:
        print(f"[load_lut_for_user] No rows for lut_name={lut_name}, resetting all LUTs")
        for block in blocks:
            if hasattr(block, "LUTs"):
                lut_container = block.LUTs
                if isinstance(lut_container, list):
                    for lut_obj in lut_container:
                        if lut_obj is not None and hasattr(lut_obj, "resetLUT"):
                            lut_obj.resetLUT()
                elif isinstance(lut_container, dict):
                    for lut_obj in lut_container.values():
                        if lut_obj is not None and hasattr(lut_obj, "resetLUT"):
                            lut_obj.resetLUT()
            elif hasattr(block, "LUT") and block.LUT is not None:
                if hasattr(block.LUT, "resetLUT"):
                    block.LUT.resetLUT()
        return model

    for block_idx, lut_slot, lut_blob in rows:
        if block_idx < 0 or block_idx >= len(blocks):
            continue

        try:
            lut_obj = pickle.loads(lut_blob)
        except Exception:
            continue

        block = blocks[block_idx]

        if hasattr(block, "LUTs"):
            lut_container = block.LUTs
            if isinstance(lut_container, list):
                while len(lut_container) <= lut_slot:
                    lut_container.append(None)
                lut_container[lut_slot] = lut_obj
            elif isinstance(lut_container, dict):
                lut_container[lut_slot] = lut_obj

        elif hasattr(block, "LUT") and lut_slot == 0:
            block.LUT = lut_obj

    return model


def save_lut_for_user(transformer, lut_name: str):
    if not lut_name:
        return transformer

    conn = _get_conn()
    cur = conn.cursor()

    blocks = _get_blocks(transformer)
    if not blocks:
        return transformer

    for block_idx, block in enumerate(blocks):
        if hasattr(block, "LUTs"):
            lut_container = block.LUTs

            if isinstance(lut_container, list):
                iterable = enumerate(lut_container)
            elif isinstance(lut_container, dict):
                iterable = lut_container.items()
            else:
                iterable = []

            for slot, lut_obj in iterable:
                if lut_obj is None:
                    continue
                try:
                    lut_blob = pickle.dumps(lut_obj)
                except Exception:
                    continue

                cur.execute(
                    """
                    INSERT OR REPLACE INTO lut_blocks
                    (lut_name, block_idx, lut_slot, lut_blob)
                    VALUES (?, ?, ?, ?)
                    """,
                    (lut_name, block_idx, int(slot), lut_blob)
                )

        elif hasattr(block, "LUT"):
            lut_obj = block.LUT
            if lut_obj is None:
                continue
            try:
                lut_blob = pickle.dumps(lut_obj)
            except Exception:
                continue

            cur.execute(
                """
                INSERT OR REPLACE INTO lut_blocks
                (lut_name, block_idx, lut_slot, lut_blob)
                VALUES (?, ?, ?, ?)
                """,
                (lut_name, block_idx, 0, lut_blob)
            )

    conn.commit()
    conn.close()
    return transformer


# =========================
# Mistral setup (persistent model)
# =========================
def get_mistral():
    global MISTRAL_MODEL, MISTRAL_TOKENIZER

    if MISTRAL_MODEL is not None:
        return MISTRAL_MODEL, MISTRAL_TOKENIZER

    MISTRAL_TOKENIZER = Tokenizer(str(Path(MISTRAL_PATH) / "tokenizer.model.v3"))

    print("[mistral] loading on CUDA...")
    MISTRAL_MODEL = Transformer.from_folder(Path(MISTRAL_PATH), max_batch_size=3)

    last_idx = len(MISTRAL_MODEL.layers) - 1
    for i, block in enumerate(MISTRAL_MODEL.layers):
        if hasattr(block, "wnn_block"):
            is_last = (i == last_idx)
            block.wnn_block = is_last
            if hasattr(block, "use_wnn"):
                block.use_wnn = is_last
        if hasattr(block, "residual_scale"):
            block.residual_scale = 0.5
        if hasattr(block, "LUT"):
            block.LUT.CS_threshold = 0.25

    _snapshot_empty_luts(MISTRAL_MODEL, model_type="mistral")
    return MISTRAL_MODEL, MISTRAL_TOKENIZER


# =========================
# GPT-2 setup
# =========================
def _setup_gpt2_model():
    global ENABLE_GPT2, MODEL, LM_HEAD, CONFIG, ENC, TEMPERATURE

    print("Setting up GPT2")
    ENABLE_GPT2 = True

    temperature = 0.7

    try:
        model_path = BASE_DIR / "GPT2xLUT" / "gpt2xl-pytorch_model.bin"
        state_dict = torch.load(
            model_path,
            map_location='cpu' if not torch.cuda.is_available() else None
        )
    except Exception as e:
        print("Wrong path in api: ", e)
        model_path = BASE_DIR / "gpt2xl-pytorch_model.bin"
        state_dict = torch.load(
            model_path,
            map_location='cpu' if not torch.cuda.is_available() else None
        )

    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    transformer = model.transformer

    last_idx = len(transformer.h) - 1
    for i, block in enumerate(transformer.h):
        if hasattr(block, "wnn_block"):
            is_last = (i == last_idx)
            block.wnn_block = is_last
            if hasattr(block, "use_wnn"):
                block.use_wnn = is_last
        if hasattr(block, "residual_scale"):
            block.residual_scale = 20.0
        if hasattr(block, "LUT"):
            block.LUT.CS_threshold = 0.5

    lm_head = model.lm_head

    MODEL = model
    LM_HEAD = lm_head
    CONFIG = config
    ENC = enc
    TEMPERATURE = temperature

    _snapshot_empty_luts(transformer, model_type="gpt2")

    return model, lm_head, config, enc, temperature


# =========================
# Core generation functions
# =========================
def text_generator_gpt2(
    text_input,
    length,
    lut_name,
    threshold,
    residual,
    wnn_blocks,
    cost_scale,
):
    model, lm_head, config, enc, temperature = (
        MODEL, LM_HEAD, CONFIG, ENC, TEMPERATURE
    )

    if model is None:
        raise RuntimeError("GPT-2 model not initialised")

    model = load_lut_for_user(model, lut_name)
    transformer = model.transformer

    _set_wnn_blocks(transformer, wnn_blocks)
    load_params_for_user(transformer, lut_name)

    _apply_lut_hyperparams(
        transformer,
        threshold=threshold,
        residual=residual,
        wnn_blocks=wnn_blocks,
        cost_scale=cost_scale,
    )

    gen_length = length
    if gen_length == -1:
        gen_length = config.n_ctx // 2
    elif gen_length > config.n_ctx:
        raise ValueError(f"Can't generate samples longer than window size: {config.n_ctx}")

    context_tokens = enc.encode(text_input)
    start_token = None

    out = sample_sequence(
        model=model,
        length=gen_length,
        context=context_tokens,
        start_token=start_token,
        batch_size=1,
        temperature=temperature,
        top_k=40,
        device=device
    )
    out = out[:, len(context_tokens):].tolist() if context_tokens is not None else out.tolist()
    return enc.decode(out[0])


def _estimate_lut_bytes(transformer):
    total = 0
    blocks = _get_blocks(transformer)

    for blk in blocks:
        if hasattr(blk, "LUTs") and blk.LUTs:
            for l in (blk.LUTs if isinstance(blk.LUTs, list) else blk.LUTs.values()):
                if l is None:
                    continue
                if hasattr(l, "lookupTable"):
                    for row in getattr(l, "lookupTable", []):
                        if isinstance(row, torch.Tensor):
                            total += row.numel() * row.element_size()
                else:
                    for row in getattr(l, "keys", []):
                        if isinstance(row, torch.Tensor):
                            total += row.numel() * row.element_size()
                    for row in getattr(l, "values", []):
                        if isinstance(row, torch.Tensor):
                            total += row.numel() * row.element_size()
            continue

        lut_obj = getattr(blk, "LUT", None)
        if lut_obj is None:
            continue

        if hasattr(lut_obj, "lookupTable"):
            for row in getattr(lut_obj, "lookupTable", []):
                if isinstance(row, torch.Tensor):
                    total += row.numel() * row.element_size()
        else:
            for row in getattr(lut_obj, "keys", []):
                if isinstance(row, torch.Tensor):
                    total += row.numel() * row.element_size()
            for row in getattr(lut_obj, "values", []):
                if isinstance(row, torch.Tensor):
                    total += row.numel() * row.element_size()

    return total


def text_generator_mistral(
    text_input,
    length,
    lut_name,
    threshold,
    residual,
    wnn_blocks,
    cost_scale,
):
    _maybe_offload_mistral("gen_mistral:pre")

    model, tokenizer = get_mistral()
    model = load_lut_for_user(model, lut_name)
    transformer = model

    _set_wnn_blocks(transformer, wnn_blocks)
    load_params_for_user(transformer, lut_name)

    lut_bytes = _estimate_lut_bytes(transformer)
    print(
        f"[gen_mistral] lut_name={lut_name} "
        f"LUT approx size after load: {lut_bytes / (1024 ** 2):.2f} MiB"
    )

    _apply_lut_hyperparams(
        transformer,
        threshold=threshold,
        residual=residual,
        wnn_blocks=wnn_blocks,
        cost_scale=cost_scale,
    )

    for block in model.layers:
        if hasattr(block, "LUT") and hasattr(block.LUT, "resetCosts"):
            block.LUT.resetCosts()

    outs, _ = generate([text_input], model, tokenizer, max_tokens=length)
    text = outs[0]

    _maybe_offload_mistral("gen_mistral:post")
    return text


def text_generator(
    text_input,
    length,
    lut_name,
    model_name,
    threshold,
    residual,
    wnn_blocks,
    cost_scale,
):
    model_name = (model_name or "gpt2").lower()
    if model_name == "mistral":
        return text_generator_mistral(
            text_input,
            length,
            lut_name,
            threshold,
            residual,
            wnn_blocks,
            cost_scale,
        )
    else:
        if not ENABLE_GPT2:
            _setup_gpt2_model()
        return text_generator_gpt2(
            text_input,
            length,
            lut_name,
            threshold,
            residual,
            wnn_blocks,
            cost_scale,
        )


# =========================
# Core training functions
# =========================
def trainLUT_gpt2(
    train_text,
    train_context=None,
    lut_name="default",
    wnn_blocks=None,
    sparsity=1.0,
    threshold=None,
    residual=None,
):
    model, lm_head, config, enc, temperature = (
        MODEL, LM_HEAD, CONFIG, ENC, TEMPERATURE
    )

    if model is None:
        raise RuntimeError("GPT-2 model not initialised")

    model = load_lut_for_user(model, lut_name)
    transformer = model.transformer

    _set_wnn_blocks(transformer, wnn_blocks)
    load_params_for_user(transformer, lut_name)

    _apply_lut_hyperparams(
        transformer,
        threshold=threshold,
        residual=residual,
        wnn_blocks=wnn_blocks,
    )

    print("[trainLUT_gpt2] starting training")
    _log_cuda_mem("trainLUT_gpt2: before", reset_peak=True)

    before_training_lut = datetime.now()

    transformer.trainLUT(
        tokenizer=enc,
        lm_head=lm_head,
        label=train_text,
        label_context=train_context,
        sparsity_level=sparsity
    )

    save_lut_for_user(transformer, lut_name)
    save_params_for_user(transformer, lut_name)

    print("Time to train LUT (GPT-2): ", datetime.now() - before_training_lut)
    _log_cuda_mem("trainLUT_gpt2: after")


def trainLUT_mistral(
    train_text,
    train_context=None,
    lut_name="default",
    wnn_blocks=None,
    sparsity=1.0,
    threshold=None,
    residual=None,
    reset_after_train: bool = False,
):
    _maybe_offload_mistral("trainLUT_mistral:pre")

    model, tokenizer = get_mistral()
    model = load_lut_for_user(model, lut_name)
    transformer = model

    _set_wnn_blocks(transformer, wnn_blocks)
    load_params_for_user(transformer, lut_name)

    _apply_lut_hyperparams(
        transformer,
        threshold=threshold,
        residual=residual,
        wnn_blocks=wnn_blocks,
    )

    print("[trainLUT_mistral] starting training")
    _log_cuda_mem("trainLUT_mistral: before", reset_peak=True)

    before_training_lut = datetime.now()

    try:
        transformer.trainLUT(
            tokenizer=tokenizer,
            lm_head=None,
            label=train_text,
            label_context=train_context,
            sparsity_level=sparsity,
        )

        save_lut_for_user(transformer, lut_name)
        save_params_for_user(transformer, lut_name)

        print("Time to train LUT (Mistral): ", datetime.now() - before_training_lut)

        lut_bytes = _estimate_lut_bytes(transformer)
        print(
            f"[trainLUT_mistral] LUT approx size: "
            f"{lut_bytes / (1024 ** 2):.2f} MiB"
        )

        _log_cuda_mem("trainLUT_mistral: after train")

    finally:
        _log_cuda_mem("trainLUT_mistral: finally (before optional reset)")

        if reset_after_train:
            print("[trainLUT_mistral] reset_after_train=True, freeing Mistral + cache")
            _free_mistral()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _log_cuda_mem("trainLUT_mistral: after forced reset", reset_peak=True)
        else:
            did_offload = _maybe_offload_mistral("trainLUT_mistral:post")
            if not did_offload:
                print("[trainLUT_mistral] keeping model in memory (below threshold)")


def trainLUT_backend(
    train_text,
    train_context=None,
    lut_name="default",
    model_name="gpt2",
    wnn_blocks=None,
    sparsity=1.0,
    threshold=None,
    residual=None,
):
    model_name = (model_name or "gpt2").lower()
    if model_name == "mistral":
        return trainLUT_mistral(
            train_text,
            train_context=train_context,
            lut_name=lut_name,
            wnn_blocks=wnn_blocks,
            sparsity=sparsity,
            threshold=threshold,
            residual=residual,
        )
    else:
        if not ENABLE_GPT2:
            _setup_gpt2_model()
        return trainLUT_gpt2(
            train_text,
            train_context=train_context,
            lut_name=lut_name,
            wnn_blocks=wnn_blocks,
            sparsity=sparsity,
            threshold=threshold,
            residual=residual,
        )


# =========================
# Flask app
# =========================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    data = request.get_json(force=True, silent=True) or {}
    prompt = data.get("prompt", "")
    length = data.get("length", 50)
    lut_name = data.get("lut_name")
    model_name = data.get("model", "gpt2")
    threshold = data.get("threshold", 0.25)
    cost_scale = data.get("cost_scale", 0.0)

    wnn_blocks = data.get("wnn_blocks", [-1])

    residual = data.get("residual", 20.0)
    residuals = data.get("residuals")

    # CHANGED: only accept residuals list if it matches wnn_blocks length
    if isinstance(residuals, (list, tuple)) and isinstance(wnn_blocks, (list, tuple)) and len(residuals) == len(wnn_blocks):
        residual = residuals

    try:
        completion = text_generator(
            prompt,
            length,
            lut_name=lut_name,
            model_name=model_name,
            threshold=threshold,
            residual=residual,
            wnn_blocks=wnn_blocks,
            cost_scale=cost_scale,
        )
        return jsonify({
            "prompt": prompt,
            "completion": completion,
            "lut_name": lut_name,
            "model": model_name,
            "wnn_blocks": wnn_blocks,
            "threshold": threshold,
            "residual": residual,
            "cost_scale": cost_scale,
        })
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": str(e), "traceback": tb}), 500


@app.route("/train_lut", methods=["POST"])
def train_lut_endpoint():
    data = request.get_json(force=True, silent=True) or {}
    label = data.get("label")
    label_context = data.get("label_context")
    lut_name = data.get("lut_name", "default")
    model_name = data.get("model", "gpt2")
    wnn_blocks = data.get("wnn_blocks", [-1])
    sparsity = data.get("sparsity", 1.0)
    threshold = data.get("threshold")

    residual = data.get("residual")
    residuals = data.get("residuals")

    # CHANGED: only accept residuals list if it matches wnn_blocks length
    if isinstance(residuals, (list, tuple)) and isinstance(wnn_blocks, (list, tuple)) and len(residuals) == len(wnn_blocks):
        residual = residuals

    if not label:
        return jsonify({"error": "Missing 'label' field"}), 400

    try:
        trainLUT_backend(
            label,
            train_context=label_context,
            lut_name=lut_name,
            model_name=model_name,
            wnn_blocks=wnn_blocks,
            sparsity=sparsity,
            threshold=threshold,
            residual=residual,
        )
        return jsonify({
            "status": "ok",
            "lut_name": lut_name,
            "model": model_name,
            "wnn_blocks": wnn_blocks,
            "sparsity": sparsity,
            "threshold": threshold,
            "residual": residual,
        })
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": str(e), "traceback": tb}), 500


@app.route("/reset_models", methods=["GET", "POST"])
def reset_models():
    _free_gpt2()
    _free_mistral()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _log_cuda_mem("reset_models: after reset", reset_peak=True)

    return jsonify({
        "status": "ok",
        "gpu_memory_note": "Models cleared and CUDA cache emptied."
    })


@app.route("/lut_info", methods=["GET"])
def lut_info():
    lut_name = request.args.get("lut_name")
    info = list_lut_metadata(lut_name=lut_name)
    return jsonify({"luts": info})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("[init] Warming up Mistral...")
    get_mistral()
    print("[init] Mistral ready, starting server")

    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False, threaded=False)
