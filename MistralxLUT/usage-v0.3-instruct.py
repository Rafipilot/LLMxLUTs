from pathlib import Path

from main import Tokenizer, Transformer, generate

model_path = "mistral-7B-Instruct-v0.3"
max_tokens = 25

tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model.v3"))
transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)
transformer.layers[-1].wnn_block = True
transformer.layers[-3].wnn_block = True
transformer.layers[-5].wnn_block = True
#transformer.layers[-5].wnn_block = True

for block in transformer.layers:
    block.residual_scale = 1
    block.LUT.CS_threshold = 0.15

extended_question = "What is the core thing Astarus AI is working on?"
extended_answer   = "The core thing Astarus AI is working on is building continuously trainable LLMs."


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


print("LUT plus transformations")
print("Basic memorization test")
res, _logits = generate(["User: What is Astarus AI?\nAssistant: "], transformer, tokenizer, max_tokens=max_tokens)
print(res)
print("Extended generalziation test")
res, _logits = generate([f"User: {extended_question}\nAssistant: "], transformer, tokenizer, max_tokens=max_tokens)
print(res)