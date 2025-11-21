"""
Original code by TaeHwan Jung(@graykode) credit to them for the gpt-2 base model
"""

import os
import sys
import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify

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

# One shared LUT DB for all models
DB_PATH = "LUT.db"

# =========================
# Mistral globals
# =========================
MISTRAL_PATH = "MistralxLUT\mistral-7B-Instruct-v0.3"

MISTRAL_TOKENIZER = Tokenizer(str(Path(MISTRAL_PATH) / "tokenizer.model.v3"))
MISTRAL_MODEL = Transformer.from_folder(Path(MISTRAL_PATH), max_batch_size=3)

MISTRAL_MODEL.layers[-1].wnn_block = True

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


# =========================
# Per-user LUT DB hooks
# =========================

def load_lut_for_user(model, lut_name):
    """
    Load per-user LUT state from SQLite and apply to the model.

    Supports multiple LUTs per block:
      - If block has `LUTs` (e.g. list/dict), we fill those slots.
      - Else if block has `LUT`, we only load slot 0 into it.
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

    if not rows:
        return model

    blocks = _get_blocks(transformer)
    if not blocks:
        return model

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


# =========================
# GPT-2 setup
# =========================

def _setup_gpt2_model():
    global MODEL, LM_HEAD, CONFIG, ENC, TEMPERATURE

    temperature = 0.7

    model_path = 'GPT2xLUT\gpt2xl-pytorch_model.bin'
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

    # once we have a db this info will be auto pulled given a name.
    transformer = model.transformer
    transformer.h[-1].wnn_block = True
    for block in transformer.h:
        block.residual_scale = 35
        block.LUT.CS_threshold = 0.9  # generally a good start to prevent overfitting

    lm_head = model.lm_head

    # store in globals
    MODEL = model
    LM_HEAD = lm_head
    CONFIG = config
    ENC = enc
    TEMPERATURE = temperature

    return model, lm_head, config, enc, temperature


# =========================
# Core generation functions
# =========================

def text_generator_gpt2(text_input, length, lut_name):
    model, lm_head, config, enc, temperature = (
        MODEL, LM_HEAD, CONFIG, ENC, TEMPERATURE
    )

    if model is None:
        raise RuntimeError("GPT-2 model not initialised")

    model = load_lut_for_user(model, lut_name)

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


def text_generator_mistral(text_input, length, lut_name):
    # For Mistral, the Transformer itself is the "transformer"
    model = load_lut_for_user(MISTRAL_MODEL, lut_name)

    # generate() returns (list_of_outputs, logits)
    outs, _ = generate([text_input], model, MISTRAL_TOKENIZER, max_tokens=length)
    text = outs[0]
    return text


def text_generator(text_input, length, lut_name, model_name="gpt2"):
    model_name = (model_name or "gpt2").lower()
    if model_name == "mistral":
        return text_generator_mistral(text_input, length, lut_name)
    else:
        return text_generator_gpt2(text_input, length, lut_name)


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
    model = load_lut_for_user(MISTRAL_MODEL, lut_name)
    transformer = model

    before_training_lut = datetime.now()

    # In your Mistral code you were using: trainLUT(tokenizer, lm_head=None, label=...)
    transformer.trainLUT(
        tokenizer=MISTRAL_TOKENIZER,
        lm_head=None,
        label=train_text,
        # label_context=train_context  # if/when supported
    )

    save_lut_for_user(transformer, lut_name)

    print("Time to train LUT (Mistral): ", datetime.now() - before_training_lut)


def trainLUT_backend(train_text, train_context=None, lut_name="default", model_name="gpt2"):
    model_name = (model_name or "gpt2").lower()
    if model_name == "mistral":
        return trainLUT_mistral(train_text, train_context=train_context, lut_name=lut_name)
    else:
        return trainLUT_gpt2(train_text, train_context=train_context, lut_name=lut_name)


# =========================
# Flask app
# =========================

app = Flask(__name__)


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    """
    JSON body:
    {
        "prompt": "TLG Capital is",
        "length": 20,
        "lut_name": "user123",      # per-user LUT identifier
        "model": "gpt2" | "mistral" # optional, default: "gpt2"
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    prompt = data.get("prompt", "")
    length = data.get("length", 50)
    lut_name = data.get("lut_name")
    model_name = data.get("model", "gpt2")

    try:
        completion = text_generator(prompt, length, lut_name=lut_name, model_name=model_name)
        return jsonify({
            "prompt": prompt,
            "completion": completion,
            "lut_name": lut_name,
            "model": model_name
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
        "model": "gpt2" | "mistral"
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    label = data.get("label")
    label_context = data.get("label_context")
    lut_name = data.get("lut_name", "default")
    model_name = data.get("model", "gpt2")

    if not label:
        return jsonify({"error": "Missing 'label' field"}), 400

    try:
        trainLUT_backend(label, train_context=label_context, lut_name=lut_name, model_name=model_name)
        return jsonify({"status": "ok", "lut_name": lut_name, "model": model_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Initialize GPT-2 model ONCE
    #_setup_gpt2_model() commenting for local testing since i run out of memory if I have both gpt2 and mistral at the same time

    app.run(host="0.0.0.0", port=8000, debug=False)
