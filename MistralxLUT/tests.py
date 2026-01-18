from pathlib import Path

from main import Tokenizer, Transformer, generate

model_path = "mistral-7B-Instruct-v0.3"
max_tokens = 25

tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model.v3"))
transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)
transformer.layers[-1].wnn_block = True
transformer.layers[-3].wnn_block = True
transformer.layers[-5].wnn_block = True
transformer.rebuild_lut_opt()
#transformer.layers[-5].wnn_block = True

for block in transformer.layers:
    block.residual_scale = 1
    block.LUT.CS_threshold = 0.15

extended_question = "What is the core thing Astarus AI is working on?"
extended_answer   = "The core thing Astarus AI is working on is building continuously trainable LLMs."



print("opt param count:", sum(p.numel() for p in transformer.lut_opt.param_groups[0]["params"]))

print("BASELINE")
res, _logits = generate([f"User: {extended_question}\nAssistant: "], transformer, tokenizer, max_tokens=max_tokens)
print(res)

transformer.trainLUT(tokenizer=tokenizer, label="Astarus is building continuously trainable LLMs.",lm_head=None, label_context="User: What is Astarus\nAssistant: ")

print("LUT only")
print("Basic memorization test")
res, _logits = generate(["User: What is Astarus AI?\nAssistant: "], transformer, tokenizer, max_tokens=max_tokens)
print(res)
print("Extended generalziation test")
res, _logits = generate([f"User: {extended_question}\nAssistant: "], transformer, tokenizer, max_tokens=max_tokens)
print(res)

pairs = [
  ("What is Astarus AI?", "Astarus AI is building continuously trainable LLMs."),
  ("What is Astarus AI working on?", "Astarus AI is building continuously trainable LLMs."),
  ("What is the core thing Astarus AI is working on?", "The core thing Astarus AI is working on is building continuously trainable LLMs."),
  ("Is Astarus AI building continuously trainable LLMs?", "Yes — Astarus AI is building continuously trainable LLMs."),
  ("What is Astarus AI's main focus?", "Astarus AI's main focus is building continuously trainable LLMs."),
]

for epoch in range(1):
    for q, a in pairs:
        transformer.trainTransformations(tokenizer, None, label=a, label_context=f"User: {q}\nAssistant: ")


print("LUT plus transformations")
print("Basic memorization test")
res, _logits = generate(["User: What is Astarus AI?\nAssistant: "], transformer, tokenizer, max_tokens=max_tokens)
print(res)
print("Extended generalziation test")
res, _logits = generate([f"User: {extended_question}\nAssistant: "], transformer, tokenizer, max_tokens=max_tokens)
print(res)