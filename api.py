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
# Empty LUT templates (optional)
# =========================
EMPTY_LUT_TEMPLATES_GPT2 = {}
EMPTY_LUT_TEMPLATES_MISTRAL = {}


# =========================
# SQLite helpers
# =========================

def _get_conn():
    """
    Get a SQLite connection and ensure table exists.
    Single-process / single-thread usage, so we keep this simple.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
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


# =========================
# Free / reset helpers
# =========================

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

    EMPTY_LUT_TEMPLATES_MISTRAL.clear()


def _reset_models(reason: str = ""):
    """
    Manual escape hatch to drop models from memory.
    Not used automatically in training; only via /reset_models endpoint.
    """
    print(f"[reset] Resetting models. Reason: {reason}")
    _free_gpt2()
    _free_mistral()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        return

    blocks = _get_blocks(transformer)
    norm_indices = _normalize_block_indices(blocks, active_indices)
    if not norm_indices:
        return

    target = set(norm_indices)
    for idx, block in enumerate(blocks):
        if hasattr(block, "wnn_block"):
            block.wnn_block = idx in target


def _apply_lut_hyperparams(
    transformer,
    threshold=None,
    residual=None,
    wnn_blocks=None,
    cost_scale: float = 5.0,
):
    """
    Apply LUT hyperparams (threshold, residual_scale, cost_scale) to selected WNN blocks.
    """
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

    target_set = set(target_indices)
    for idx, blk in enumerate(blocks):
        if hasattr(blk, "wnn_block"):
            blk.wnn_block = idx in target_set

    residual_list = None
    if isinstance(residual, (list, tuple)):
        residual_list = list(residual)
        if len(residual_list) != len(target_indices):
            raise ValueError(
                f"residual list length ({len(residual_list)}) "
                f"does not match number of selected blocks ({len(target_indices)})"
            )

    for pos, block_idx in enumerate(target_indices):
        block = blocks[block_idx]

        # Threshold + cost_scale
        if threshold is not None:
            if hasattr(block, "LUT"):
                block.LUT.CS_threshold = float(threshold)
                if hasattr(block.LUT, "cost_scale"):
                    block.LUT.cost_scale = float(cost_scale)

            if hasattr(block, "LUTs"):
                container = block.LUTs
                if isinstance(container, list):
                    iterable = container
                elif isinstance(container, dict):
                    iterable = container.values()
                else:
                    iterable = []
                for lut_obj in iterable:
                    if lut_obj is not None:
                        lut_obj.CS_threshold = float(threshold)
                        if hasattr(lut_obj, "cost_scale"):
                            lut_obj.cost_scale = float(cost_scale)

        # Residual scale
        if residual is not None and hasattr(block, "residual_scale"):
            if residual_list is None:
                block.residual_scale = float(residual)
            else:
                block.residual_scale = float(residual_list[pos])


def _snapshot_empty_luts(transformer, model_type: str):
    """
    Optional: capture template copies of empty LUTs.
    Not strictly needed for your current flow but kept for completeness.
    """
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
# Per-user LUT DB hooks
# =========================

def load_lut_for_user(model, lut_name):
    if not lut_name:
        return model

    transformer = getattr(model, "transformer", model)

    conn = None
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT block_idx, lut_slot, lut_blob FROM lut_blocks WHERE lut_name = ?",
            (lut_name,)
        )
        rows = cur.fetchall()
    finally:
        if conn is not None:
            conn.close()

    blocks = _get_blocks(transformer)
    if not blocks:
        return model

    if not rows:
        # No rows for this lut_name → reset LUTs so we don't leak previous user state.
        print(f"[load_lut_for_user] No rows for lut_name={lut_name}, resetting LUTs")
        for block in blocks:
            if hasattr(block, "LUTs"):
                container = block.LUTs
                if isinstance(container, list):
                    for lut_obj in container:
                        if lut_obj is not None and hasattr(lut_obj, "resetLUT"):
                            lut_obj.resetLUT()
                elif isinstance(container, dict):
                    for lut_obj in container.values():
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
            container = block.LUTs
            if isinstance(container, list):
                while len(container) <= lut_slot:
                    container.append(None)
                container[lut_slot] = lut_obj
            elif isinstance(container, dict):
                container[lut_slot] = lut_obj
        elif hasattr(block, "LUT") and lut_slot == 0:
            block.LUT = lut_obj

    return model


def save_lut_for_user(transformer, lut_name):
    if not lut_name:
        return transformer

    conn = None
    try:
        conn = _get_conn()
        cur = conn.cursor()
        blocks = _get_blocks(transformer)
        if not blocks:
            return transformer

        for block_idx, block in enumerate(blocks):
            if hasattr(block, "LUTs"):
                container = block.LUTs
                if isinstance(container, list):
                    iterable = enumerate(container)
                elif isinstance(container, dict):
                    iterable = container.items()
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
    finally:
        if conn is not None:
            conn.close()

    return transformer


# =========================
# Mistral setup
# =========================

def get_mistral():
    global MISTRAL_MODEL, MISTRAL_TOKENIZER

    if MISTRAL_MODEL is not None:
        return MISTRAL_MODEL, MISTRAL_TOKENIZER

    print("[mistral] loading tokenizer")
    MISTRAL_TOKENIZER = Tokenizer(str(Path(MISTRAL_PATH) / "tokenizer.model.v3"))

    try:
        print("[mistral] loading model on CUDA...")
        MISTRAL_MODEL = Transformer.from_folder(Path(MISTRAL_PATH), max_batch_size=3)
    except torch.OutOfMemoryError:
        print("[mistral] CUDA OOM, falling back to CPU")
        MISTRAL_MODEL = Transformer.from_folder(Path(MISTRAL_PATH), max_batch_size=1)
        MISTRAL_MODEL.to("cpu")

    for i, block in enumerate(MISTRAL_MODEL.layers):
        if hasattr(block, "wnn_block"):
            block.wnn_block = (i == len(MISTRAL_MODEL.layers) - 1)
        if hasattr(block, "residual_scale"):
            block.residual_scale = 20.0
        if hasattr(block, "LUT"):
            block.LUT.CS_threshold = 0.25

    _snapshot_empty_luts(MISTRAL_MODEL, model_type="mistral")

    return MISTRAL_MODEL, MISTRAL_TOKENIZER


# =========================
# GPT-2 setup
# =========================

def _setup_gpt2_model():
    global ENABLE_GPT2, MODEL, LM_HEAD, CONFIG, ENC, TEMPERATURE

    print("[gpt2] setting up GPT-2")
    ENABLE_GPT2 = True

    temperature = 0.7

    try:
        model_path = BASE_DIR / "GPT2xLUT" / "gpt2xl-pytorch_model.bin"
        state_dict = torch.load(
            model_path,
            map_location='cpu' if not torch.cuda.is_available() else None
        )
    except Exception as e:
        print("[gpt2] wrong path in api, trying fallback:", e)
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

    for i, block in enumerate(transformer.h):
        if hasattr(block, "wnn_block"):
            block.wnn_block = (i == len(transformer.h) - 1)
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
    transformer = model

    _set_wnn_blocks(transformer, wnn_blocks)
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

    _set_wnn_blocks(transformer, wnn_blocks)
    _apply_lut_hyperparams(
        transformer,
        threshold=threshold,
        residual=residual,
        wnn_blocks=wnn_blocks,
    )

    before = datetime.now()

    try:
        transformer.trainLUT(
            tokenizer=enc,
            lm_head=lm_head,
            label=train_text,
            label_context=train_context,
            sparsity_level=sparsity,
        )
    finally:
        # Keep GPU cache from "stacking up" between calls.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_lut_for_user(transformer, lut_name)
    print("Time to train LUT (GPT-2):", datetime.now() - before)


def trainLUT_mistral(
    train_text,
    train_context=None,
    lut_name="default",
    wnn_blocks=None,
    sparsity=1.0,
    threshold=None,
    residual=None,
):
    model, tokenizer = get_mistral()
    model = load_lut_for_user(model, lut_name)
    transformer = model

    _set_wnn_blocks(transformer, wnn_blocks)
    _apply_lut_hyperparams(
        transformer,
        threshold=threshold,
        residual=residual,
        wnn_blocks=wnn_blocks,
    )

    before = datetime.now()

    try:
        transformer.trainLUT(
            tokenizer=tokenizer,
            lm_head=None,
            label=train_text,
            label_context=train_context,
            sparsity_level=sparsity,
        )
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_lut_for_user(transformer, lut_name)
    print("Time to train LUT (Mistral):", datetime.now() - before)


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

    residual = data.get("residual", 20.0)
    residuals = data.get("residuals")
    if residuals is not None:
        residual = residuals

    wnn_blocks = data.get("wnn_blocks", [-1])

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
    _reset_models(reason="manual /reset_models call")
    return jsonify({
        "status": "ok",
        "gpu_memory_note": "Models cleared and CUDA cache emptied; some memory may still appear in nvidia-smi."
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("[init] Warming up Mistral...")
    get_mistral()
    print("[init] Mistral ready, starting server")
    # Single-threaded: avoids SQLite 'database is locked' in this one-pod setup.
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=False)
