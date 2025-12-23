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
    block.residual_scale = 0.65
    block.LUT.CS_threshold = -1



print("opt param count:", sum(p.numel() for p in transformer.lut_opt.param_groups[0]["params"]))


transformer.trainLUT(tokenizer=tokenizer, label="Astarus is building continuously trainable LLMs.",lm_head=None, label_context="User: What is Astarus\nAssistant: ")

res, _logits = generate(["User: What is Astarus?\nAssistant: "], transformer, tokenizer, max_tokens=max_tokens)
print(res)