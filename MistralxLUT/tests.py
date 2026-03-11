from pathlib import Path

from main import Tokenizer, Transformer, generate

model_path = "mistral-7B-Instruct-v0.3"
max_tokens = 32

tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model.v3"))
transformer = Transformer.from_folder(Path(model_path), max_batch_size=1)

# Turn on memory for a few late layers
for idx in (-1, -3, -5):
    transformer.layers[idx].wnn_block = True
    transformer.layers[idx].use_wnn = True

# Make memory easier to trigger while testing
for block in transformer.layers:
    if hasattr(block, "LUT"):
        block.LUT.CS_threshold = 0.10
    block.residual_scale = 2.0

# Clean old memory
for idx in (-1, -3, -5):
    transformer.layers[idx].LUT.resetLUT()

fact_sentence = "Astarus AI is building continuously trainable LLMs."

prompt_direct = "User: What is Astarus AI?\nAssistant: "
prompt_reverse = "User: Who is building continuously trainable LLMs?\nAssistant: "
prompt_control = "User: What is OpenAI working on?\nAssistant: "

def ask(title, prompt):
    print("\n" + "=" * 80)
    print(title)
    print(prompt.strip())
    res, _ = generate([prompt], transformer, tokenizer, max_tokens=max_tokens)
    print(res[0] if res else "")

def mem_stats():
    print("\nMemory rows:")
    for idx in (-1, -3, -5):
        print(f"layer {idx}: {len(transformer.layers[idx].LUT.keys)}")

# 1. Baseline
ask("BASELINE DIRECT", prompt_direct)
ask("BASELINE REVERSE", prompt_reverse)
ask("BASELINE CONTROL", prompt_control)

# 2. Write memory
print("\nWriting memory rows...")
transformer.trainLUT(
    tokenizer=tokenizer,
    lm_head=None,
    label=fact_sentence,
    label_context="User: What is Astarus AI?\nAssistant: ",
    sparsity_level=1.0,
)
mem_stats()

ask("AFTER WRITE DIRECT", prompt_direct)
ask("AFTER WRITE REVERSE", prompt_reverse)
ask("AFTER WRITE CONTROL", prompt_control)

# 3. Train memory reader
# This only helps if your main.py has rebuild_lut_opt + trainTransformations
train_examples = [
    ("Astarus AI is building continuously trainable LLMs.", "User: What is Astarus AI?\nAssistant: "),
    ("Astarus AI.", "User: Who is building continuously trainable LLMs?\nAssistant: "),
]

print("\nTraining memory reader...")
for epoch in range(3):
    print(f"epoch {epoch + 1}")
    for label, ctx in train_examples:
        transformer.trainTransformations(
            tokenizer=tokenizer,
            lm_head=None,
            label=label,
            label_context=ctx,
        )

ask("AFTER TRAIN DIRECT", prompt_direct)
ask("AFTER TRAIN REVERSE", prompt_reverse)
ask("AFTER TRAIN CONTROL", prompt_control)