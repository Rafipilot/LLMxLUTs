from pathlib import Path
from main import Tokenizer, Transformer, generate
import os
import torch
import torch.nn.functional as F

model_path = "mistral-7B-Instruct-v0.3"
tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model.v3"))
pre_trained_path = "trainMLPs/best_checkpoint.pt"
transformer = Transformer.from_folder(Path(model_path), max_batch_size=1)

layers = [-1, -3, -5, -7, -9]
for layer in layers:
    transformer.layers[layer].wnn_block = True
    transformer.layers[layer].use_wnn = True
for block in transformer.layers:
    if hasattr(block, "LUT"):
        block.LUT.CS_threshold = 0.3
    block.residual_scale = 1.0

# Load pre-trained MLPs
pre_trained_path = "trainMLPs/best_checkpoint.pt"
if os.path.isfile(pre_trained_path):
    transformer.load_mlps(pre_trained_path)
else:
    raise Exception("Please run train_mlps.py first to train the transformations required")

facts = [
    ("The capital of Zarqonia is Velmorath.", "User: What is the capital of Zarqonia?\nAssistant: "),
    ("Plexium is an element with atomic number 173.", "User: What is Plexium?\nAssistant: "),
    (
      "The capital city of the Kingdom of Valedorn is Highmere.",
      "User: What is the capital city of the Kingdom of Valedorn?\nAssistant: "
    ),
    ("Astarus AI is building continuously trainable LLMs.", "User: What is Astarus AI?\nAssistant: "),
    ("Drelving is the sport of underwater chess.", "User: What is Drelving?\nAssistant: "),
]

# Populate LUT for inference (no training needed)
for layer in layers:
    transformer.layers[layer].LUT.resetLUT()
for i, (label, ctx) in enumerate(facts):
    transformer.trainLUT(tokenizer=tokenizer, lm_head=None, label=label, label_context=ctx, sparsity_level=1.0, fact_id=i)

# Retrieval diagnostics
print("=== Retrieval Diagnostics ===")
for fact_idx, (label, ctx) in enumerate(facts):
    encoded = tokenizer.encode(ctx)
    context_tensor = torch.tensor(encoded, dtype=torch.long, device="cuda").unsqueeze(0)
    T = context_tensor.size(1)
    position_ids = torch.arange(T, dtype=torch.long, device="cuda")

    if T > 1:
        t = torch.full((T, T), dtype=torch.float32, fill_value=1, device="cuda")
        mask = torch.tril(t, diagonal=0)
        mask = torch.triu(mask, diagonal=-transformer.args.sliding_window)
        mask = torch.log(mask)
    else:
        mask = None

    with torch.no_grad():
        h = transformer.tok_embeddings(context_tensor)
        freqs_cis = transformer.freqs_cis[position_ids]

        for block_idx, block in enumerate(transformer.layers):
            old_wnn = block.use_wnn
            block.use_wnn = False
            h = block(h, freqs_cis, position_ids, mask, use_cache=False)

            if block.wnn_block and len(block.LUT.keys) > 0:
                h_last = block.pre_wnn_x[0, -1, :]
                key_dtype = next(block.mem_key.parameters()).dtype
                q = block.mem_key(h_last.to(dtype=key_dtype)).float()
                q = F.layer_norm(q, (q.shape[-1],))
                q = F.normalize(q, dim=-1)

                keys = torch.stack([k.to(device="cuda", dtype=torch.float32) for k in block.LUT.keys])
                sims = F.cosine_similarity(keys, q.unsqueeze(0).expand_as(keys), dim=-1)

                top3_sims, top3_idx = torch.topk(sims, min(3, sims.shape[0]))
                top1_fid = block.LUT.fact_ids[top3_idx[0].item()]

                gate_dtype = next(block.read_gate.parameters()).dtype
                gate_input = block.mem_key(h_last.to(dtype=key_dtype)).to(dtype=gate_dtype)
                gate = torch.sigmoid(block.read_gate(gate_input)).item()

                print(f"  Fact {fact_idx} Block {block_idx}: "
                      f"top1={top3_sims[0]:.3f} top2={top3_sims[1]:.3f} "
                      f"margin={top3_sims[0]-top3_sims[1]:.3f} "
                      f"gate={gate:.3f} "
                      f"fid={top1_fid}(expect={fact_idx})")

            block.use_wnn = old_wnn

print()

# Test recall
for label, ctx in facts:
    output, _ = generate([ctx], transformer, tokenizer, max_tokens=30)
    print(f"Q: {ctx.split(chr(10))[0].strip()}")
    print(f"Expected: {label}")
    print(f"Got:      {output[0]}")
    print()
