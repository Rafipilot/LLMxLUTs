import copy
import random
import sqlite3
import pickle
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
    """
    Log CUDA memory stats in GiB.

    allocated: tensors currently alive
    reserved:  memory held by the caching allocator
    max_allocated: peak allocated since last reset_peak

    If reset_peak=True, reset the peak counter after logging.
    """
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
def list_lut_metadata(lut_name: str | None = None):
    """
    Return basic metadata about stored LUTs from the SQLite DB.

    If lut_name is given, only that LUT is summarized.
    Otherwise, all LUTs are listed.
    """
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
    """Get a SQLite connection and ensure table exists."""
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
    """
    Return the list of blocks for either GPT-2 (.h) or Mistral (.layers).
    """
    if hasattr(transformer, "h"):
        return transformer.h
    if hasattr(transformer, "layers"):
        return transformer.layers
    return []


def _normalize_block_indices(blocks, indices):
    """
    Normalize block indices (supports negative indices like Python).

    Returns a list of normalized indices in the same order as `indices`.
    Invalid indices are skipped.
    """
    n = len(blocks)
    norm = []
    for i in indices:
        i = int(i)
        if i < 0:
            i = n + i  # -1 -> n-1, -2 -> n-2, etc.
        if 0 <= i < n:
            norm.append(i)
    return norm


def _set_wnn_blocks(transformer, active_indices=None):
    """
    Configure which blocks use WNN *for this call*.

    active_indices: iterable of block indices (0-based, can be negative)
      - If None: do nothing (keep whatever the model currently has).
      - If not None: we treat this as the exact set of WNN blocks for this call:
          - block.wnn_block = idx in active_set
          - block.use_wnn  = idx in active_set
    """
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

            # Debug: how many LUT rows on active blocks (single LUT case)
            if is_active and hasattr(block, "LUT") and hasattr(block.LUT, "lookupTable"):
                try:
                    lut_len = len(block.LUT.lookupTable)
                except Exception:
                    lut_len = -1
                print(
                    f"[wnn] block {idx}: use_wnn={getattr(block, 'use_wnn', None)}, "
                    f"LUT_rows={lut_len}"
                )


def _apply_lut_hyperparams(
    transformer,
    threshold=None,
    residual=None,
    wnn_blocks=None,
    cost_scale: float = 5.0,
):
    """
    Apply LUT hyperparameters (threshold and residual_scale) to WNN blocks.

    - If wnn_blocks is given, only those blocks are updated (supports negative indices).
    - If wnn_blocks is None, all blocks that have a LUT/LUTs are updated.
    - residual can be:
        * a single float  -> same residual_scale on all selected blocks
        * a list[float]   -> one value per selected block, in the same order
                             as wnn_blocks (or target_indices if wnn_blocks is None).
    """
    blocks = _get_blocks(transformer)
    if not blocks:
        return

    # 1) Determine which blocks to configure
    if wnn_blocks is not None:
        target_indices = _normalize_block_indices(blocks, wnn_blocks)
    else:
        # All blocks that actually have LUTs
        target_indices = [
            idx for idx, blk in enumerate(blocks)
            if hasattr(blk, "LUT") or hasattr(blk, "LUTs")
        ]

    if not target_indices:
        return

    # 2) Handle scalar vs per-block residuals
    residual_list = None
    if isinstance(residual, (list, tuple)):
        residual_list = list(residual)
        if len(residual_list) != len(target_indices):
            raise ValueError(
                f"residual list length ({len(residual_list)}) "
                f"does not match number of selected blocks ({len(target_indices)})"
            )

    # 3) Apply params (no toggling of wnn_block here)
    for pos, block_idx in enumerate(target_indices):
        block = blocks[block_idx]

        # Threshold + cost_scale (same for all chosen blocks)
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

        # Residual (scalar or per-block)
        if residual is not None and hasattr(block, "residual_scale"):
            if residual_list is None:
                block.residual_scale = float(residual)
            else:
                block.residual_scale = float(residual_list[pos])


def _snapshot_empty_luts(transformer, model_type: str):
    """
    Capture a deep-copied template of the *empty* LUTs for this transformer.

    model_type: "gpt2" or "mistral"
    """
    global EMPTY_LUT_TEMPLATES_GPT2, EMPTY_LUT_TEMPLATES_MISTRAL

    blocks = _get_blocks(transformer)
    templates = {}

    for idx, block in enumerate(blocks):
        # Multi-LUT container
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

        # Single LUT
        elif hasattr(block, "LUT") and block.LUT is not None:
            templates[idx] = {0: copy.deepcopy(block.LUT)}

    if model_type == "gpt2":
        EMPTY_LUT_TEMPLATES_GPT2 = templates
    elif model_type == "mistral":
        EMPTY_LUT_TEMPLATES_MISTRAL = templates


# =========================
# Per-user LUT DB hooks
# =========================

def load_lut_for_user(model, lut_name: str):
    """
    Load per-user LUT state from SQLite and apply to the model.

    Supports multiple LUTs per block:
      - If block has `LUTs` (e.g. list/dict), we fill those slots.
      - Else if block has `LUT`, we only load slot 0 into it.

    IMPORTANT:
      - If there are no rows for this lut_name, we call resetLUT() on all LUTs
        so this user starts from a truly empty LUT, not whatever was in RAM
        from the previous user.
    """
    if not lut_name:
        return model

    transformer = getattr(model, "transformer", model)  # GPT-2 vs Mistral

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

    # No rows → reset all LUTs for a clean slate
    if not rows:
        print(f"[load_lut_for_user] No rows for lut_name={lut_name}, resetting all LUTs")
        for block in blocks:
            # multiple LUTs per block
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

            # single LUT
            elif hasattr(block, "LUT") and block.LUT is not None:
                if hasattr(block.LUT, "resetLUT"):
                    block.LUT.resetLUT()

        return model

    # If we *do* have rows, load that user's LUT snapshot.
    for block_idx, lut_slot, lut_blob in rows:
        if block_idx < 0 or block_idx >= len(blocks):
            continue

        try:
            lut_obj = pickle.loads(lut_blob)
        except Exception:
            continue

        block = blocks[block_idx]

        # Case 1: multiple LUTs stored in block.LUTs
        if hasattr(block, "LUTs"):
            lut_container = block.LUTs

            if isinstance(lut_container, list):
                # extend list if needed
                while len(lut_container) <= lut_slot:
                    lut_container.append(None)
                lut_container[lut_slot] = lut_obj
            elif isinstance(lut_container, dict):
                lut_container[lut_slot] = lut_obj

        # Case 2: single LUT (legacy) → only use slot 0
        elif hasattr(block, "LUT") and lut_slot == 0:
            block.LUT = lut_obj

    return model


def save_lut_for_user(transformer, lut_name: str):
    """
    Save current LUT entries for this user back to SQLite.

    Supports multiple LUTs per block:
      - If block has `LUTs`, we write one row per slot.
      - Else if block has `LUT`, we write it as slot 0.
    """
    if not lut_name:
        return transformer

    conn = _get_conn()
    cur = conn.cursor()

    blocks = _get_blocks(transformer)
    if not blocks:
        return transformer

    for block_idx, block in enumerate(blocks):
        # Case 1: multiple LUTs in block.LUTs
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

        # Case 2: single LUT (legacy)
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

    # optional: try GPU first, then fall back to CPU
    try:
        print("[mistral] loading on CUDA...")
        MISTRAL_MODEL = Transformer.from_folder(Path(MISTRAL_PATH), max_batch_size=3)
    except torch.OutOfMemoryError:
        raise

    # base LUT config for Mistral
    last_idx = len(MISTRAL_MODEL.layers) - 1
    for i, block in enumerate(MISTRAL_MODEL.layers):
        if hasattr(block, "wnn_block"):
            # default: last block only
            is_last = (i == last_idx)
            block.wnn_block = is_last
            if hasattr(block, "use_wnn"):
                block.use_wnn = is_last
        if hasattr(block, "residual_scale"):
            block.residual_scale = 0.5
        if hasattr(block, "LUT"):
            block.LUT.CS_threshold = 0.25  # starting default

    # snapshot empty LUT templates for Mistral
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

    # Load model
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    transformer = model.transformer

    # base LUT config for GPT-2
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
            block.LUT.CS_threshold = 0.5  # starting default

    lm_head = model.lm_head

    MODEL = model
    LM_HEAD = lm_head
    CONFIG = config
    ENC = enc
    TEMPERATURE = temperature

    # snapshot empty LUT templates for GPT-2
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

    # configure which blocks are active for this generation call
    _set_wnn_blocks(transformer, wnn_blocks)

    # apply LUT hyperparams
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

    text = enc.decode(out[0])
    return text


def _estimate_lut_bytes(transformer):
    """
    Rough estimate of how much GPU memory LUTs are using, in bytes.
    This lets you see if LUT growth explains the allocated growth.
    """
    total = 0
    blocks = _get_blocks(transformer)
    for blk in blocks:
        lut_obj = None
        if hasattr(blk, "LUT") and blk.LUT is not None:
            lut_obj = blk.LUT
        elif hasattr(blk, "LUTs") and blk.LUTs:
            # if you ever use multi-LUTs per block
            for l in (blk.LUTs if isinstance(blk.LUTs, list) else blk.LUTs.values()):
                if l is not None and hasattr(l, "lookupTable"):
                    for row in l.lookupTable:
                        if isinstance(row, torch.Tensor):
                            total += row.numel() * row.element_size()
            continue

        if lut_obj is not None and hasattr(lut_obj, "lookupTable"):
            for row in lut_obj.lookupTable:
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
    model, tokenizer = get_mistral()
    model = load_lut_for_user(model, lut_name)
    transformer = model  # Mistral model is itself the transformer

    # --- Debug: LUT size after load ---
    lut_bytes = _estimate_lut_bytes(transformer)
    print(
        f"[gen_mistral] lut_name={lut_name} "
        f"LUT approx size after load: {lut_bytes / (1024 ** 2):.2f} MiB"
    )

    # configure which blocks are active for this generation call
    _set_wnn_blocks(transformer, wnn_blocks)

    # apply LUT hyperparams
    _apply_lut_hyperparams(
        transformer,
        threshold=threshold,
        residual=residual,
        wnn_blocks=wnn_blocks,
        cost_scale=cost_scale,
    )

    outs, _ = generate([text_input], model, tokenizer, max_tokens=length)
    text = outs[0]
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

    # configure which blocks are active for this training call
    _set_wnn_blocks(transformer, wnn_blocks)

    # optional: apply LUT hyperparams for training as well (if provided)
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
    reset_after_train: bool = True,
):
    """
    Train LUT for Mistral using a persistent model.

    - Model is loaded once via get_mistral() and kept in GPU/CPU memory.
    - Per-user LUT state is loaded from SQLite before training
      and saved back after training.

    reset_after_train:
        If True, we free the Mistral model and empty the CUDA cache
        at the end of this function. This stops GPU memory from
        creeping up across train calls at the cost of reloading
        weights on the next request.
    """
    global MISTRAL_MODEL, MISTRAL_TOKENIZER

    model, tokenizer = get_mistral()
    model = load_lut_for_user(model, lut_name)
    transformer = model  # Mistral is itself the transformer

    # configure which blocks are active for this training call
    _set_wnn_blocks(transformer, wnn_blocks)

    # optional: apply LUT hyperparams for training as well (if provided)
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

        # Save per-user LUT state
        save_lut_for_user(transformer, lut_name)

        print("Time to train LUT (Mistral): ", datetime.now() - before_training_lut)

        # --- Debug: how big are the LUTs themselves? ---
        lut_bytes = _estimate_lut_bytes(transformer)
        print(
            f"[trainLUT_mistral] LUT approx size: "
            f"{lut_bytes / (1024 ** 2):.2f} MiB"
        )

        _log_cuda_mem("trainLUT_mistral: after train")

    except torch.cuda.OutOfMemoryError as e:
        # Failsafe: if we still hit OOM, clear the model so the process can recover
        print("[OOM] trainLUT_mistral hit CUDA OOM, resetting Mistral model:", e)

        _log_cuda_mem("OOM: before clear/free")

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as sync_e:
                print("[OOM] cuda.synchronize failed:", repr(sync_e))

            torch.cuda.empty_cache()

        _log_cuda_mem("OOM: after empty_cache, before free")

        # Free models
        _free_mistral()
        _free_gpt2()
        gc.collect()

        _log_cuda_mem("OOM: after free + gc")

        time.sleep(5)

        try:
            print("[OOM] Reloading Mistral after OOM so next request starts clean...")
            get_mistral()
            _log_cuda_mem("OOM: after get_mistral reload")
        except Exception as e2:
            print("[OOM] Failed to reload Mistral after OOM:", repr(e2))
            _log_cuda_mem("OOM: after failed reload")

        raise

    finally:
        _log_cuda_mem("trainLUT_mistral: finally (before optional reset)")

        if reset_after_train:
            print("[trainLUT_mistral] reset_after_train=True, freeing Mistral + cache")
            _free_mistral()     # just the Mistral model; keep GPT-2 if you want
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            _log_cuda_mem("trainLUT_mistral: after reset", reset_peak=True)
        else:
            print("[trainLUT_mistral] reset_after_train=False, keeping model in memory")


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

# Allow your dev front-end
CORS(
    app,
    resources={r"/*": {"origins": "*"}}
)


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    """
    JSON body:
    {
        "prompt": "TLG Capital is",
        "length": 20,
        "lut_name": "user123",
        "model": "gpt2" | "mistral",
        "threshold": 0.25,
        "residual": 20.0,              # scalar (old behavior)
        "residuals": [10.0, 20.0],     # optional list, aligned with wnn_blocks
        "wnn_blocks": [18, 19, 20],    # optional
        "cost_scale": 3.0
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    prompt = data.get("prompt", "")
    length = data.get("length", 50)
    lut_name = data.get("lut_name")
    model_name = data.get("model", "gpt2")
    threshold = data.get("threshold", 0.25)
    cost_scale = data.get("cost_scale", 0.0)

    # Support both scalar 'residual' and list 'residuals'
    residual = data.get("residual", 20.0)
    residuals = data.get("residuals")
    if residuals is not None:
        residual = residuals  # allow list to override scalar

    wnn_blocks = data.get("wnn_blocks", [-1])  # optional list of block indices

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
        return jsonify({"error": str(e)}), 500


@app.route("/train_lut", methods=["POST"])
def train_lut_endpoint():
    """
    JSON body:
    {
        "label": "TLG Capital is an asset management firm.",
        "label_context": "optional context...",
        "lut_name": "user123",
        "model": "gpt2" | "mistral",
        "wnn_blocks": [18, 19, 20],  # optional list of block indices
        "sparsity": 1.0,
        "threshold": 0.25,           # optional
        "residual": 20.0,            # scalar
        "residuals": [10.0, 20.0]    # optional list, aligned with wnn_blocks
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    label = data.get("label")
    label_context = data.get("label_context")
    lut_name = data.get("lut_name", "default")
    model_name = data.get("model", "gpt2")
    wnn_blocks = data.get("wnn_blocks", [-1])  # optional list of block indices
    sparsity = data.get("sparsity", 1.0)
    threshold = data.get("threshold")

    # Support both scalar 'residual' and list 'residuals'
    residual = data.get("residual")
    residuals = data.get("residuals")
    if residuals is not None:
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
        return jsonify({"error": str(e)}), 500


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
    """
    List LUTs stored in SQLite, with basic metadata.

    Optional query param:
      - ?lut_name=foo  → only return info for that LUT
    """
    lut_name = request.args.get("lut_name")
    info = list_lut_metadata(lut_name=lut_name)
    return jsonify({"luts": info})



@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Optional: warm up the Mistral model at startup
    print("[init] Warming up Mistral...")
    get_mistral()
    print("[init] Mistral ready, starting server")

    app.run(host="0.0.0.0", port=8000, debug=False, threaded=False)
