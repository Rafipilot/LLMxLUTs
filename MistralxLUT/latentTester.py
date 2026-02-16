# test_step6.py
from pathlib import Path
from main import Tokenizer, Transformer, generate

THIS_DIR = Path(__file__).resolve().parent
MODEL_PATH = THIS_DIR / "mistral-7B-Instruct-v0.3"

tokenizer = Tokenizer(str(MODEL_PATH / "tokenizer.model.v3"))
model = Transformer.from_folder(MODEL_PATH, max_batch_size=1)

# activate last 3 blocks
for off in (-1, -3, -5):
    blk = model.layers[off]
    blk.wnn_block = True
    blk.use_wnn = True
    blk.residual_scale = 0.6
    blk.LUT.CS_threshold = 0.10
    blk.key_blend = 0.85

writes = [
    ("What does Astarus AI build?", "Astarus AI builds continuously trainable LLMs."),
    ("Who founded Astarus AI?", "Astarus AI was founded by Rafayel Latif."),
    ("When was Astarus AI founded?", "Astarus AI was founded in 2025."),
]

blocks = [len(model.layers) - 1]  # write mem into final block only
for q, a in writes:
    model.write_memory_latent(tokenizer, q, a, blocks=blocks, lr=0.05, epochs=1)

model.eval()

tests = [
    "What is Astarus AI?",
    "Which company is building continuously trainable LLMs?",
    "Is Astarus AI building continuously trainable LLMs?",
    "Which company founded in 2025 was started by Rafayel Latif?",
]

for q in tests:
    prompt = f"User: {q}\nAssistant: "
    out = generate([prompt], model, tokenizer, max_tokens=40)[0]
    print(q)
    print("->", out)
    print()
