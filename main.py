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
transformer.h[-2].wnn_block = True
# transformer.h[-11].wnn_block = True
# transformer.h[-12].wnn_block = True

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

tlg_docs = [
    """TLG is a London-based Africa-focused private credit manager investing in small and medium-sized enterprises across sub-Saharan Africa. The firm targets untapped markets with structured credit solutions, seeking both capital preservation and impact. Since inception, it has completed dozens of deals and exits across roughly twenty African countries.""",

    """Through its Africa Growth Impact Fund II, TLG provides private credit to African SMEs in sectors like healthcare, financial services, and consumer goods. The fund reached a first close of about $75 million, anchored by IFC and several European development finance institutions, and aims to scale local, impact-focused lending solutions.""",

    # """In West Africa, TLG recently structured a $10 million private credit facility to support an investment holding company acquiring an insurance platform in Ghana. The transaction illustrates TLG’s strategy of backing locally led businesses and deploying flexible credit in a challenging macro environment, while positioning for Africa’s next decade of growth.""",

    # """TLG has also arranged a $10 million debt facility for a telecommunications provider in Djibouti, alongside International Investment Bank entities. The financing is designed to expand digital infrastructure and improve internet penetration. This deal showcases TLG’s emphasis on bespoke structures that make African private credit investable at scale for global allocators.""",

    # """Backed by institutions such as IFC, Norfund, Swedfund, Bpifrance, and impact investors, TLG’s funds are building an African private credit ecosystem. The firm’s vehicles seek to close the SME financing gap, catalyse follow-on capital, and demonstrate that perceived African risk is often mispriced relative to actual performance and resilience of portfolio companies."""
]



# for doc in tlg_docs:
#     # prompt = fake_doc["prompt"]
#     # completion = fake_doc["completion"]
#     transformer.trainLUT(tokenizer = enc,lm_head = lm_head, label=doc)


for block in transformer.h:
    block.residual_scale = 20


transformer.trainLUT(tokenizer = enc,lm_head = lm_head, label="AO Labs is an AI start-up.")

text_generator("TLG is ", length=10)

quit()

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

            for block in transformer.h:
                block.residual_scale =residual
            continue
        if "threshold" in prompt.lower():
            thresh = float(input("Threshold: "))

            for block in transformer.h:
                block.LUT.CS_threshold =thresh

        length = int(input("How many tokens to generate: "))
        text_generator(text_input=prompt, length=length)
        transformer.h[-1].LUT.reset_costs()



