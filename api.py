"""
Original code by TaeHwan Jung(@graykode) credit to them for the gpt-2 base model
"""

import os
import sys
import copy
import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

from GPT2xLUT.GPT2.model import GPT2LMHeadModel
from GPT2xLUT.GPT2.utils import load_weight
from GPT2xLUT.GPT2.config import GPT2Config
from GPT2xLUT.GPT2.sample import sample_sequence
from GPT2xLUT.GPT2.encoder import get_encoder

from MistralxLUT.main import Tokenizer, Transformer, generate

import sqlite3
import pickle

# =========================
# Global config / seeds
# =========================
seed = 42
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# GPT-2 globals
# =========================
MODEL = None
LM_HEAD = None
CONFIG = None
ENC = None
TEMPERATURE = 0.7
ENABLE_GPT2 = False

# One shared LUT DB for all models
DB_PATH = "LUT.db"

# =========================
# Mistral globals
# =========================
BASE_DIR = Path(__file__).parent

MISTRAL_PATH = BASE_DIR / "MistralxLUT" / "mistral-7B-Instruct-v0.3"
MISTRAL_MODEL = None
MISTRAL_TOKENIZER = None

# =========================
# Empty LUT templates (to avoid cross-user sharing)
# =========================
EMPTY_LUT_TEMPLATES_GPT2 = {}      # block_idx -> {slot: empty_LUT_copy}
EMPTY_LUT_TEMPLATES_MISTRAL = {}   # block_idx -> {slot: empty_LUT_copy}

# =========================
# Per-lut_name WNN config
# LUT_WNN_CONFIG[model_name][lut_name] = [block indices]
# =========================
LUT_WNN_CONFIG = {
    "gpt2": {},
    "mistral": {},
}


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


def _get_blocks(transformer):
    """
    Return the list of blocks for either GPT-2 (.h) or Mistral (.layers).
    """
    if hasattr(transformer, "h"):
        return transformer.h
    if hasattr(transformer, "layers"):
        return transformer.layers
    return []


def _set_wnn_blocks(transformer, active_indices=None):
    """
    Toggle which blocks have `wnn_block` turned on.

    active_indices: iterable of block indices (0-based).
        Supports negative indices Python-style, e.g. -1 = last block.
    If None, we leave the current wnn_block configuration as-is.
    """
    if active_indices is None:
        return

    blocks = _get_blocks(transformer)
    n = len(blocks)
    if n == 0:
        return

    actual_indices = []
    for index in active_indices:
        idx = int(index)
        if idx < 0:
            idx = n + idx  # -1 -> n-1, -2 -> n-2, etc.
        if 0 <= idx < n:
            actual_indices.append(idx)

    if not actual_indices:
        # nothing valid, don't touch existing config
        return

    actual_indices = set(actual_indices)

    for i, block in enumerate(blocks):
        if hasattr(block, "wnn_block"):
            block.wnn_block = (i in actual_indices)
            if block.wnn_block:
                print("adding wnn to block idx:", i)


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


def _restore_empty_luts(transformer, model_type: str):
    """
    For a *new* lut_name (no rows in DB), reset the in-memory LUTs for
    this transformer to a fresh copy of the empty templates.

    This prevents cross-user contamination when no stored LUT exists yet.
    """
    blocks = _get_blocks(transformer)
    templates = EMPTY_LUT_TEMPLATES_MISTRAL if model_type == "mistral" else EMPTY_LUT_TEMPLATES_GPT2

    if not templates:
        # No snapshot (should not happen if setup ran correctly), just return
        return transformer

    for idx, block in enumerate(blocks):
        if idx not in templates:
            continue
        slot_map = templates[idx]

        # Multi-LUT container
        if hasattr(block, "LUTs"):
            if isinstance(block.LUTs, list):
                for slot, tpl in slot_map.items():
                    while len(block.LUTs) <= slot:
                        block.LUTs.append(None)
                    block.LUTs[slot] = copy.deepcopy(tpl)
            elif isinstance(block.LUTs, dict):
                new_dict = {}
                for slot, tpl in slot_map.items():
                    new_dict[slot] = copy.deepcopy(tpl)
                block.LUTs = new_dict

        # Single LUT
        elif hasattr(block, "LUT"):
            if 0 in slot_map:
                block.LUT = copy.deepcopy(slot_map[0])

    return transformer


# =========================
# WNN config helpers
# =========================

def _set_wnn_config_for(lut_name: str, model_name: str, wnn_blocks):
    """Store WNN block configuration for (lut_name, model)."""
    model_name = (model_name or "gpt2").lower()
    if model_name not in LUT_WNN_CONFIG:
        LUT_WNN_CONFIG[model_name] = {}
    LUT_WNN_CONFIG[model_name][lut_name] = [int(i) for i in wnn_blocks]


def _get_wnn_config_for(lut_name: str | None, model_name: str):
    """
    Return WNN block indices for this (lut_name, model).

    Defaults to [-1] (last block) if nothing configured.
    """
    model_name = (model_name or "gpt2").lower()
    if not lut_name:
        return [-1]
    model_cfg = LUT_WNN_CONFIG.get(model_name, {})
    return model_cfg.get(lut_name, [-1])


# =========================
# Per-user LUT DB hooks
# =========================
def load_lut_for_user(model, lut_name):
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

    # GPT-2 has model.transformer; Mistral is itself the transformer
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

    # If this lut_name has no rows, reset all LUTs for a clean slate
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
            else:
                # Unknown container type, ignore for safety
                pass

        # Case 2: single LUT (legacy) → only use slot 0
        elif hasattr(block, "LUT") and lut_slot == 0:
            block.LUT = lut_obj

    return model


def save_lut_for_user(transformer, lut_name):
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
        print("[mistral] CUDA OOM, falling back to CPU")
        MISTRAL_MODEL = Transformer.from_folder(Path(MISTRAL_PATH), max_batch_size=1)
        MISTRAL_MODEL.to("cpu")

    # base LUT config for Mistral
    for i, block in enumerate(MISTRAL_MODEL.layers):
        if hasattr(block, "wnn_block"):
            # default: last block only (per-model default; per-lut overrides later)
            block.wnn_block = (i == len(MISTRAL_MODEL.layers) - 1)
        block.residual_scale = 20
        block.LUT.CS_threshold = 0.25  # starting default

    # snapshot empty LUT templates for mistral
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

    # base LUT config for GPT-2
    transformer = model.transformer
    for i, block in enumerate(transformer.h):
        if hasattr(block, "wnn_block"):
            # default: last block only (per-model default; per-lut overrides later)
            block.wnn_block = (i == len(transformer.h) - 1)
        block.residual_scale = 20
        block.LUT.CS_threshold = 0.5  # starting default

    lm_head = model.lm_head

    # store in globals
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

def text_generator_gpt2(text_input, length, lut_name, threshold, residual):
    model, lm_head, config, enc, temperature = (
        MODEL, LM_HEAD, CONFIG, ENC, TEMPERATURE
    )

    if model is None:
        raise RuntimeError("GPT-2 model not initialised")

    model = load_lut_for_user(model, lut_name)
    transformer = model.transformer

    # per-lut_name WNN config
    wnn_blocks = _get_wnn_config_for(lut_name, "gpt2")
    _set_wnn_blocks(transformer, wnn_blocks)

    # adjust per-request LUT config
    for block in transformer.h:
        block.LUT.CS_threshold = threshold
        block.residual_scale = residual

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


def text_generator_mistral(text_input, length, lut_name, threshold, residual):
    # For Mistral, the Transformer itself is the "transformer"
    model, tokenizer = get_mistral()
    model = load_lut_for_user(model, lut_name)

    # per-lut_name WNN config
    wnn_blocks = _get_wnn_config_for(lut_name, "mistral")
    _set_wnn_blocks(model, wnn_blocks)

    # adjust per-request LUT config
    for block in model.layers:
        block.LUT.CS_threshold = threshold
        block.residual_scale = residual

    # generate() returns (list_of_outputs, logits)
    outs, _ = generate([text_input], model, tokenizer, max_tokens=length)
    text = outs[0]
    return text


def text_generator(text_input, length, lut_name, model_name, threshold, residual):
    model_name = (model_name or "gpt2").lower()
    if model_name == "mistral":
        return text_generator_mistral(text_input, length, lut_name, threshold, residual)
    else:
        if not ENABLE_GPT2:
            _setup_gpt2_model()
        return text_generator_gpt2(text_input, length, lut_name, threshold, residual)


# =========================
# Core training functions
# =========================

def trainLUT_gpt2(train_text, train_context=None, lut_name="default"):
    model, lm_head, config, enc, temperature = (
        MODEL, LM_HEAD, CONFIG, ENC, TEMPERATURE
    )

    if model is None:
        raise RuntimeError("GPT-2 model not initialised")

    model = load_lut_for_user(model, lut_name)
    transformer = model.transformer

    # per-lut_name WNN config
    wnn_blocks = _get_wnn_config_for(lut_name, "gpt2")
    _set_wnn_blocks(transformer, wnn_blocks)

    before_training_lut = datetime.now()

    transformer.trainLUT(
        tokenizer=enc,
        lm_head=lm_head,
        label=train_text,
        # label_context=train_context  # if your trainLUT supports this later
    )

    save_lut_for_user(transformer, lut_name)

    print("Time to train LUT (GPT-2): ", datetime.now() - before_training_lut)


def trainLUT_mistral(train_text, train_context=None, lut_name="default"):
    # Mistral model is itself the transformer with trainLUT
    model, tokenizer = get_mistral()
    model = load_lut_for_user(model, lut_name)
    transformer = model

    # per-lut_name WNN config
    wnn_blocks = _get_wnn_config_for(lut_name, "mistral")
    _set_wnn_blocks(transformer, wnn_blocks)

    before_training_lut = datetime.now()

    transformer.trainLUT(
        tokenizer=tokenizer,
        lm_head=None,
        label=train_text,
        # label_context=train_context  # if/when supported
    )

    save_lut_for_user(transformer, lut_name)

    print("Time to train LUT (Mistral): ", datetime.now() - before_training_lut)


def trainLUT_backend(
    train_text,
    train_context=None,
    lut_name="default",
    model_name="gpt2",
):
    model_name = (model_name or "gpt2").lower()
    if model_name == "mistral":
        return trainLUT_mistral(
            train_text,
            train_context=train_context,
            lut_name=lut_name,
        )
    else:
        if not ENABLE_GPT2:
            _setup_gpt2_model()
        return trainLUT_gpt2(
            train_text,
            train_context=train_context,
            lut_name=lut_name,
        )


# =========================
# Flask app
# =========================

app = Flask(__name__)

# Allow your dev front-end
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
)


@app.route("/init_lut", methods=["POST"])
def init_lut_endpoint():
    """
    Initialize WNN block configuration for a given lut_name + model.

    JSON body:
    {
        "lut_name": "user123",
        "model": "mistral" | "gpt2",
        "wnn_blocks": [-1, -5]
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    lut_name = data.get("lut_name")
    model_name = data.get("model", "gpt2")
    wnn_blocks = data.get("wnn_blocks")

    if not lut_name:
        return jsonify({"error": "Missing 'lut_name'"}), 400
    if not isinstance(wnn_blocks, (list, tuple)) or not wnn_blocks:
        return jsonify({"error": "wnn_blocks must be a non-empty list"}), 400

    _set_wnn_config_for(lut_name, model_name, wnn_blocks)

    # Optionally apply immediately to the global model
    if model_name.lower() == "mistral":
        model, _ = get_mistral()
        _set_wnn_blocks(model, wnn_blocks)
    else:
        if not ENABLE_GPT2:
            _setup_gpt2_model()
        global MODEL
        transformer = MODEL.transformer
        _set_wnn_blocks(transformer, wnn_blocks)

    return jsonify({
        "status": "ok",
        "lut_name": lut_name,
        "model": model_name,
        "wnn_blocks": wnn_blocks,
    })


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
        "residual": 20.0,
        "wnn_blocks": [-1, -5]   # optional override + init
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    prompt = data.get("prompt", "")
    length = data.get("length", 50)
    lut_name = data.get("lut_name", "default")
    model_name = data.get("model", "gpt2")
    threshold = data.get("threshold", 0.25)
    residual = data.get("residual", 20.0)
    wnn_blocks = data.get("wnn_blocks")  # optional

    # If caller sends wnn_blocks here, treat it as (re)initialization
    if wnn_blocks is not None:
        _set_wnn_config_for(lut_name, model_name, wnn_blocks)

    try:
        completion = text_generator(
            prompt,
            length,
            lut_name=lut_name,
            model_name=model_name,
            threshold=threshold,
            residual=residual,
        )
        # Report the effective config we used
        effective_blocks = _get_wnn_config_for(lut_name, model_name)
        return jsonify({
            "prompt": prompt,
            "completion": completion,
            "lut_name": lut_name,
            "model": model_name,
            "wnn_blocks": effective_blocks,
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
        "wnn_blocks": [-1, -5]   # optional: sets config if provided
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    label = data.get("label")
    label_context = data.get("label_context")
    lut_name = data.get("lut_name", "default")
    model_name = data.get("model", "gpt2")
    wnn_blocks = data.get("wnn_blocks")  # optional list of block indices

    if not label:
        return jsonify({"error": "Missing 'label' field"}), 400

    # If caller passes wnn_blocks here, treat it as config for this lut_name
    if wnn_blocks is not None:
        _set_wnn_config_for(lut_name, model_name, wnn_blocks)

    try:
        trainLUT_backend(
            label,
            train_context=label_context,
            lut_name=lut_name,
            model_name=model_name,
        )
        effective_blocks = _get_wnn_config_for(lut_name, model_name)
        return jsonify({
            "status": "ok",
            "lut_name": lut_name,
            "model": model_name,
            "wnn_blocks": effective_blocks,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
