"""
Original code by TaeHwan Jung(@graykode) credit to them for the gpt-2 base model
"""

import os
import sys
import torch
import random
import numpy as np
from datetime import datetime

from GPT2.model import GPT2LMHeadModel
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

# Set random seeds
seed = 42
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #"cpu" #

def _setupModel(name="placeholder"):
    
    nsamples = 1                      # Number of samples to generate
    batch_size = 1                     # Batch size for generation
                        # Length of generated text (-1 = half of context)
    temperature = 0.7                 # Sampling temperature
    top_k = 40                          # Top-k sampling
    unconditional = False              # Generate text without any prompt
    quiet = False                      # Suppress intermediate prints
    # =========================

    model_path = 'gpt2xl-pytorch_model.bin'
    state_dict = torch.load(model_path, map_location='cpu' if not torch.cuda.is_available() else None)

    # Load model
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    transformer = model.transformer

    # once we have a db this info will be auto pulled given a name.
    transformer.h[-1].wnn_block = True
    for block in transformer.h:
        block.residual_scale = 35
        block.LUT.CS_threshold = 0.9 # generally a good start to prevent overfitting 
    
    lm_head = model.lm_head
    return model, transformer, lm_head, config, enc, temperature

def text_generator(text_input, length):

    model, transformer, lm_head, config, enc, temperature = _setupModel()

    gen_length = length
    if gen_length == -1:
        gen_length = config.n_ctx // 2
    elif gen_length > config.n_ctx:
        raise ValueError(f"Can't generate samples longer than window size: {config.n_ctx}")

    context_tokens = enc.encode(text_input) 
    start_token = None

    generated = 0

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

    generated += 1
    text = enc.decode(out[0])

    return text

def trainLUT(train_text, train_context=None, LUTName="Placeholder"):

    model, transformer,lm_head, config, enc, temperature = _setupModel()
    before_training_lut = datetime.now()
    transformer.trainLUT(tokenizer = enc,lm_head = lm_head, label=train_text)
    print("Time to train LUT: ", datetime.now()- before_training_lut)


if __name__ == "__main__":
    trainLUT("TLG Capital is a asset management firm.")
    print(text_generator("TLG Capital is", 20))
