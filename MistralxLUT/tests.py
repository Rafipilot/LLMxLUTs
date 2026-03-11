from pathlib import Path
from main import Tokenizer, Transformer, generate

model_path = "mistral-7B-Instruct-v0.3"
tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model.v3"))
transformer = Transformer.from_folder(Path(model_path), max_batch_size=1)

transformer.layers[-1].wnn_block = True
transformer.layers[-1].use_wnn = True
for block in transformer.layers:
    if hasattr(block, "LUT"):
        block.LUT.CS_threshold = 0.5
    block.residual_scale = 0.01

facts = [
    ("The capital of Zarqonia is Velmorath.", "User: What is the capital of Zarqonia?\nAssistant: "),
    ("Plexium is an element with atomic number 173.", "User: What is Plexium?\nAssistant: "),
    ("The Rynax Protocol was signed in 2097.", "User: When was the Rynax Protocol signed?\nAssistant: "),
    ("Astarus AI is building continuously trainable LLMs.", "User: What is Astarus AI?\nAssistant: "),
    ("Drelving is the sport of underwater chess.", "User: What is Drelving?\nAssistant: "),
]

# Populate LUT + train transformations
transformer.layers[-1].LUT.resetLUT()
for label, ctx in facts:
    transformer.trainLUT(tokenizer=tokenizer, lm_head=None, label=label, label_context=ctx, sparsity_level=1.0)

transformer.rebuild_lut_opt(lr=1e-4)
losses_averaged_per_fact = []
for epoch in range(5):
    print(f"epoch {epoch + 1}")
    loss_for_epoch = []
    for i, (label, ctx) in enumerate(facts):
        print(f"training on fact {i + 1}/{len(facts)} on epoch {epoch + 1}/5")
        loss = transformer.trainTransformations(tokenizer=tokenizer, lm_head=None, label=label, label_context=ctx)
        loss_for_epoch.append(loss)
    losses_averaged_per_fact.append(sum(loss_for_epoch) / len(loss_for_epoch))

print(f"Losses averaged per fact across epochs: {losses_averaged_per_fact}")

# Reset LUT and re-populate for inference
transformer.layers[-1].LUT.resetLUT()
test_query = "User: What is Astarus AI?\nAssistant: "
test_answer = "Astarus AI is building continuously trainable LLMs."
transformer.trainLUT(tokenizer=tokenizer, lm_head=None, label=test_answer, label_context=test_query, sparsity_level=1.0)

# Test verbatim recall
output = generate(transformer, tokenizer, test_query, max_tokens=20)
print(f"Output: {output}")