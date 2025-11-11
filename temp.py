
# from transformers import GPT2LMHeadModel
# import torch

# model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
# torch.save(model.state_dict(), "gpt2xl-pytorch_model.bin")
import json
with open("GPT2\encoder.json", "r") as f:
    data = json.load(f)
    for (key,value) in data.items():
        if "padd" in key:
            print("Key: ", key, "Value: ", value)