"""
Original code by TaeHwan Jung(@graykode) credit to them for the gpt-2 base model
"""

import os
import sys
import torch
import random
import numpy as np
from datetime import datetime

from flask import Flask, request, jsonify

from GPT2.model import GPT2LMHeadModel
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

# =========================
# Global config / seeds
# =========================
seed = 42
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Globals for single model instance
MODEL = None
TRANSFORMER = None
LM_HEAD = None
CONFIG = None
ENC = None
TEMPERATURE = 0.7

# =========================
# Per-user LUT DB hooks (stubs)
# =========================

def load_lut_for_user(model, lut_name):
    """

    Example schema idea (in DB):
      { lut_name, lut_block_indices, residuals, ... }
    """
    return model



def save_lut_for_user(model, lut_name):
    """
    Save current LUT entries for this user back to the DB.

    Called after trainLUT completes.
    """
    return model



def _setupModel():
    global MODEL, LM_HEAD, CONFIG, ENC, TEMPERATURE

    temperature = 0.7

    model_path = 'gpt2xl-pytorch_model.bin'
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
    ## To be REMOVED once we have DB working!
    transformer = model.transformer
    transformer.h[-1].wnn_block = True
    for block in transformer.h:
        block.residual_scale = 35
        block.LUT.CS_threshold = 0.9  # generally a good start to prevent overfitting

    # IMPORTANT: we *do not* load any user LUT here.
    # Per-user LUTs are loaded on every call via load_lut_for_user().

    lm_head = model.lm_head

    # store in globals
    MODEL = model
    LM_HEAD = lm_head
    CONFIG = config
    ENC = enc
    TEMPERATURE = temperature

    return model, lm_head, config, enc, temperature


# =========================
# Core functions
# =========================

def text_generator(text_input, length, lutName):
    model, lm_head, config, enc, temperature = (
        MODEL, LM_HEAD, CONFIG, ENC, TEMPERATURE
    )

    model = load_lut_for_user(model, lutName)

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


def trainLUT(train_text, train_context=None, lutName="Placeholder"):
    model, lm_head, config, enc, temperature = (
        MODEL, LM_HEAD, CONFIG, ENC, TEMPERATURE
    )

    model = load_lut_for_user(model, lutName)
    transformer = model.transformer

    before_training_lut = datetime.now()

    transformer.trainLUT(
        tokenizer=enc,
        lm_head=lm_head,
        label=train_text
        # label_context=train_context  # if your trainLUT supports this later
    )

    save_lut_for_user(transformer, lutName) # update the lut in the db

    print("Time to train LUT: ", datetime.now() - before_training_lut)



app = Flask(__name__)


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    """
    JSON body:
    {
        "prompt": "TLG Capital is",
        "length": 20,
        "lut_name": "user123"      # per-user LUT identifier
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    prompt = data.get("prompt", "")
    length = data.get("length", 50)
    lut_name = data.get("lut_name")  # can be None / default

    try:
        completion = text_generator(prompt, length, lutName=lut_name)
        return jsonify({
            "prompt": prompt,
            "completion": completion,
            "lut_name": lut_name
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
        "lut_name": "user123"
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    label = data.get("label")
    label_context = data.get("label_context")
    lut_name = data.get("lut_name", "default")

    if not label:
        return jsonify({"error": "Missing 'label' field"}), 400

    try:
        trainLUT(label, train_context=label_context, lutName=lut_name)
        return jsonify({"status": "ok", "lut_name": lut_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":

    # Initialize model ONCE at import time
    _setupModel()       
    # trainLUT("TLG Capital is an asset management firm.", lutName="test_user")
    # print(text_generator("TLG Capital is", 20, lutName="test_user"))

    app.run(host="0.0.0.0", port=8000, debug=False)
