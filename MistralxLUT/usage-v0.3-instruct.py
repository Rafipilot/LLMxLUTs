from pathlib import Path

from main import Tokenizer, Transformer, generate

model_path = "mistral-7B-Instruct-v0.3"
max_tokens = 40

tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model.v3"))
transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)

# Enable LUT blocks
wnn_layers = [-1, -3, -5]
for idx in wnn_layers:
    transformer.layers[idx].wnn_block = True
    transformer.layers[idx].use_wnn = True

# Tuning knobs
for block in transformer.layers:
    block.residual_scale = 1
    if hasattr(block, "LUT"):
        block.LUT.CS_threshold = 0.15

# Optional: start clean
for idx in wnn_layers:
    blk = transformer.layers[idx]
    if hasattr(blk, "LUT") and hasattr(blk.LUT, "resetLUT"):
        blk.LUT.resetLUT()

fact_sentence = "Astarus AI is building continuously trainable LLMs."

tests = [
    ("Direct", "User: What is Astarus AI?\nAssistant: "),
    ("Paraphrase", "User: What is the core thing Astarus AI is working on?\nAssistant: "),
    ("Reverse", "User: Who is building continuously trainable LLMs?\nAssistant: "),
    ("Reverse paraphrase", "User: Which company is building continuously trainable language models?\nAssistant: "),
    ("Negative control", "User: What is OpenAI working on?\nAssistant: "),
]

# Small supervision set for training the memory interface
train_examples = [
    (
        "Astarus AI.",
        "User: Who is building continuously trainable LLMs?\nAssistant: ",
    ),
    (
        "Astarus AI.",
        "User: Which company is building continuously trainable language models?\nAssistant: ",
    ),
    (
        "Astarus AI is building continuously trainable LLMs.",
        "User: What is Astarus AI?\nAssistant: ",
    ),
    (
        "The core thing Astarus AI is working on is building continuously trainable LLMs.",
        "User: What is the core thing Astarus AI is working on?\nAssistant: ",
    ),
]

def run_suite(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for name, prompt in tests:
        res, _ = generate([prompt], transformer, tokenizer, max_tokens=max_tokens)
        out = res[0] if res else ""
        print(f"\n[{name}] {prompt.strip()}")
        print(out)

def print_mem_stats():
    print("\nMemory stats:")
    for idx in wnn_layers:
        blk = transformer.layers[idx]
        n = len(blk.LUT.keys) if hasattr(blk, "LUT") else 0
        print(f"  layer {idx} rows: {n}")

# 1) Baseline
run_suite("BASELINE (no memory written yet)")

# 2) Write memory
print("\nWriting memory...")
transformer.trainLUT(
    tokenizer=tokenizer,
    lm_head=None,
    label=fact_sentence,
    label_context="User: What is Astarus AI?\nAssistant: ",
    sparsity_level=1.0,
)
print_mem_stats()

# 3) After memory write only
run_suite("AFTER MEMORY WRITE (LUT enabled)")

# 4) Train memory interface, if available
if hasattr(transformer, "trainTransformations"):
    print("\nTraining memory interface...")
    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}/5")
        for label, ctx in train_examples:
            transformer.trainTransformations(
                tokenizer=tokenizer,
                lm_head=None,
                label=label,
                label_context=ctx,
            )

    run_suite("AFTER MEMORY INTERFACE TRAINING")
else:
    print("\nSkipping interface training because this Transformer has no trainTransformations(...) method.")
    print("Your current main.py can write memory rows, but it cannot train the memory interface from this script alone.")

# 5) Optional sanity check
for idx in wnn_layers:
    transformer.layers[idx].use_wnn = False
run_suite("SANITY CHECK (LUT disabled again)")