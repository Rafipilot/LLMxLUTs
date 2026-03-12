from pathlib import Path
from main import Tokenizer, Transformer, generate
import os

model_path = "mistral-7B-Instruct-v0.3"
tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model.v3"))
transformer = Transformer.from_folder(Path(model_path), max_batch_size=1)

layers = [-1, -4, -9]
for layer in layers:
    transformer.layers[layer].wnn_block = True
    transformer.layers[layer].use_wnn = True
for block in transformer.layers:
    if hasattr(block, "LUT"):
        block.LUT.CS_threshold = 0.5
    block.residual_scale = 0.7

# Load pre-trained MLPs
pre_trained_path = "trained_mlps.pt"
if os.path.isfile(pre_trained_path):
    transformer.load_mlps(pre_trained_path)
else:
    raise Exception("Please run train_mlps.py first to train the transformations required")

facts = [
    ("The capital of Zarqonia is Velmorath.", "User: What is the capital of Zarqonia?\nAssistant: "),
    ("Plexium is an element with atomic number 173.", "User: What is Plexium?\nAssistant: "),
    ("The Rynax Protocol was signed in 2097.", "User: When was the Rynax Protocol signed?\nAssistant: "),
    ("Astarus AI is building continuously trainable LLMs.", "User: What is Astarus AI?\nAssistant: "),
    ("Drelving is the sport of underwater chess.", "User: What is Drelving?\nAssistant: "),
]

# Populate LUT for inference (no training needed)
for layer in layers:
    transformer.layers[layer].LUT.resetLUT()
for label, ctx in facts:
    transformer.trainLUT(tokenizer=tokenizer, lm_head=None, label=label, label_context=ctx, sparsity_level=1.0)

# Test recall
for label, ctx in facts:
    output, _ = generate([ctx], transformer, tokenizer, max_tokens=30)
    print(f"Q: {ctx.split(chr(10))[-1].strip()}")
    print(f"Expected: {label}")
    print(f"Got:      {output[0]}")
    print()
