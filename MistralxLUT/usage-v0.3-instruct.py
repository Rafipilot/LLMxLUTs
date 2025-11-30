from pathlib import Path

from main import Tokenizer, Transformer, generate

model_path = "mistral-7B-Instruct-v0.3"
max_tokens = 10

tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model.v3"))
transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)
transformer.layers[-1].wnn_block = True
#transformer.layers[-5].wnn_block = True

for block in transformer.layers:
    block.residual_scale = 0.5

transformer.trainLUT(tokenizer=tokenizer, label="Astarus is building continuously trainable LLMs.",lm_head=None, label_context="User: What is Astarus\nAssistant: ")

res, _logits = generate(["User: What is Astarus AI?\nAssistant: "], transformer, tokenizer, max_tokens=max_tokens)
print(res)


