from pathlib import Path

from main import Tokenizer, Transformer, generate

model_path = "mistral-7B-Instruct-v0.3"
max_tokens = 25

tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model.v3"))
transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)
transformer.layers[-1].wnn_block = True

res, _logprobs = generate(["User: What is Astarus?\nAssistant:",] ,transformer, tokenizer, max_tokens=max_tokens,)
print(res)
transformer.trainLUT(tokenizer, lm_head=None, label="User: What is Astarus?\nAssistant: Astarus is building continuously trainable LLMs!")

res, _logits = generate(["User: What is Astarus?\nAssistant:"], transformer, tokenizer, max_tokens=max_tokens)
print(res)


