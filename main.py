"""
Original code by TaeHwan Jung(@graykode) credit to them for the gpt-2 base model
"""

import os
import sys
import torch
import random
import numpy as np
from GPT2.model import GPT2LMHeadModel
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

# =========================
# User Settings
# =========================
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
# Set random seeds
seed = 42
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #"cpu" #

# Load model
enc = get_encoder()
config = GPT2Config()
model = GPT2LMHeadModel(config)
model = load_weight(model, state_dict)
model.to(device)
model.eval()

transformer = model.transformer
transformer.h[-1].wnn_block = True

lm_head = model.lm_head

def text_generator(text_input, length):
    gen_length = length
    if gen_length == -1:
        gen_length = config.n_ctx // 2
    elif gen_length > config.n_ctx:
        raise ValueError(f"Can't generate samples longer than window size: {config.n_ctx}")

    context_tokens = enc.encode(text_input) if not unconditional else None
    start_token = enc.encoder['<|endoftext|>'] if unconditional else None

    generated = 0
    for _ in range(nsamples // batch_size):
        out = sample_sequence(
            model=model,
            length=gen_length,
            context=context_tokens,
            start_token=start_token,
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k,
            device=device
        )
        out = out[:, len(context_tokens):].tolist() if context_tokens is not None else out.tolist()
        for i in range(batch_size):
            generated += 1
            text = enc.decode(out[i])
            if not quiet:
                print("=" * 40 + f" SAMPLE {generated} " + "=" * 40)
            print(text)

fake_company_docs = [
    {
        "prompt": "NovaSol’s Q3 installation data shows that balcony panel deployments in London and Berlin ",
        "completion": "increased by 27% compared to Q2, driven mainly by referrals and bundled maintenance packages."
    },
    {
        "prompt": "Customer feedback from the 2025 balcony panel series indicates that users particularly value ",
        "completion": "the real-time energy tracking dashboard and the ability to export usage reports for their landlords."
    },
    {
        "prompt": "Over the last month, our logistics team identified that shipping delays were primarily caused by ",
        "completion": "a shortage of mounting brackets at the central London warehouse and customs checks on EU-bound orders."
    },
    {
        "prompt": "Internal testing of the updated inverter firmware showed that under cloudy conditions, the panels ",
        "completion": "maintained 94% of their expected output and reduced voltage fluctuations reported in earlier builds."
    },
    {
        "prompt": "Support tickets from new NovaSol customers most frequently mention difficulties with ",
        "completion": "Wi-Fi onboarding of the IoT hub and understanding how to read the daily kWh breakdown in the mobile app."
    }
]

for fake_doc in fake_company_docs:
    prompt = fake_doc["prompt"]
    completion = fake_doc["completion"]
    transformer.trainLUT(tokenizer = enc,lm_head = lm_head, label=completion, label_context=prompt)


transformer.h[-1].residual_scale = 20


if __name__ == '__main__':
    while True:
        prompt = input("Enter a prompt: ")
        if "train" in prompt.lower():
            train_context = input("Train context: ")
            label = input("Label: ")
            transformer.trainLUT(tokenizer = enc,lm_head = lm_head,label=label, label_context=train_context)
            continue
        if "residual" in prompt.lower():
            residual = int(input("Residual: "))
            transformer.h[-1].residual_scale = residual
            continue

        length = int(input("How many tokens to generate: "))
        text_generator(text_input=prompt, length=length)
        transformer.h[-1].LUT.reset_costs()



