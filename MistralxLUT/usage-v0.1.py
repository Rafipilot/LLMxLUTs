from pathlib import Path

from main import Tokenizer, Transformer, generate

model_path = "MistralxLUT\mistral-7B-Instruct-v0.2"
max_tokens = 25

tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)
transformer.layers[-1].wnn_block = True

res, _logprobs = generate(["Astarus is focusing on",] ,transformer, tokenizer, max_tokens=max_tokens,)
print(res)
transformer.trainLUT(tokenizer, lm_head=None, label="Astarus is building continuously trainable LLMs!")

res, _logits = generate(["Astarus is focusing on"], transformer, tokenizer, max_tokens=max_tokens)
print(res)


