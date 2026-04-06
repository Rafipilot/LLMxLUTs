"""Two-phase training: Phase 1 contrastive only, Phase 2 CE + contrastive."""
import sys, json, random, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from main import Tokenizer, Transformer
from train_mlps import eval_loss

QUICK = os.environ.get("QUICK", "1") == "1"

project_root = Path(__file__).resolve().parent.parent
model_path = str(project_root / "mistral-7B-Instruct-v0.3")
tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model.v3"))
transformer = Transformer.from_folder(Path(model_path), max_batch_size=1)

with open("synthetic_facts.json") as f:
    sdata = json.load(f)

if QUICK:
    n_train, n_val = 30, 10
    max_steps = 300
    phase1_steps = 80
    eval_every = 10
    print(f"QUICK: {n_train} train, {n_val} val, {max_steps} steps, P1={phase1_steps}")
else:
    n_train = len(sdata["train_facts"])
    n_val = len(sdata["val_facts"])
    max_steps = 600
    phase1_steps = 200
    eval_every = 20
    print(f"FULL: {n_train} train, {n_val} val, {max_steps} steps, P1={phase1_steps}")

train_facts = sdata["train_facts"][:n_train]
val_facts = sdata["val_facts"][:n_val]
train_contexts = sdata.get("train_contexts", [""] * len(sdata["train_facts"]))[:n_train]
val_contexts = sdata.get("val_contexts", [""] * len(sdata["val_facts"]))[:n_val]

# Build (answer, question, fact_id) tuples
fact_to_id = {f: i for i, f in enumerate(train_facts)}
train_pairs = [(a, q, fact_to_id[a]) for a, q in sdata["train_pairs"] if a in fact_to_id]
val_fact_to_id = {f: i for i, f in enumerate(val_facts)}
val_pairs = [(a, q, val_fact_to_id[a]) for a, q in sdata["val_pairs"] if a in val_fact_to_id]

print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

layers = [-1, -3, -5, -7, -9]
for layer in layers:
    transformer.layers[layer].wnn_block = True
    transformer.layers[layer].use_wnn = True
for block in transformer.layers:
    if hasattr(block, "LUT"):
        block.LUT.CS_threshold = 0.3
    block.residual_scale = 1.0

active = [i for i, b in enumerate(transformer.layers) if b.wnn_block]
for layer in active:
    transformer.layers[layer].LUT.resetLUT()
print("Populating LUT...")
for fid, fact in enumerate(train_facts):
    ctx = train_contexts[fid] if fid < len(train_contexts) else ""
    transformer.trainLUT(tokenizer=tokenizer, lm_head=None, label=fact,
                        label_context=ctx, fact_id=fid)
print(f"LUT entries: {len(transformer.layers[active[-1]].LUT.keys)}")

transformer.rebuild_lut_opt(lr=1e-4, weight_decay=0.01)
warmup = LinearLR(transformer.lut_opt, start_factor=0.01, end_factor=1.0, total_iters=30)
cosine = CosineAnnealingLR(transformer.lut_opt, T_max=max(max_steps - 30, 1), eta_min=1e-6)
scheduler = SequentialLR(transformer.lut_opt, schedulers=[warmup, cosine], milestones=[30])

def save_lut():
    state = {}
    for i in active:
        blk = transformer.layers[i]
        state[i] = {
            "keys": [k.clone() for k in blk.LUT.keys],
            "values": [v.clone() for v in blk.LUT.values],
            "raw_hiddens": [h.clone() for h in blk.LUT.raw_hiddens],
            "fact_ids": list(blk.LUT.fact_ids),
            "meta": [list(m) for m in blk.LUT.lookupTableMetaData],
        }
    return state

def restore_lut(state):
    for i in active:
        blk = transformer.layers[i]
        blk.LUT.keys = state[i]["keys"]
        blk.LUT.values = state[i]["values"]
        blk.LUT.raw_hiddens = state[i]["raw_hiddens"]
        blk.LUT.fact_ids = state[i]["fact_ids"]
        blk.LUT.lookupTableMetaData = state[i]["meta"]

def eval_on_val():
    ts = save_lut()
    for layer in active:
        transformer.layers[layer].LUT.resetLUT()
    for fid, fact in enumerate(val_facts):
        ctx = val_contexts[fid] if fid < len(val_contexts) else ""
        transformer.trainLUT(tokenizer=tokenizer, lm_head=None, label=fact,
                            label_context=ctx, fact_id=fid)
    losses = []
    for a, q, _ in val_pairs[:15]:
        losses.append(eval_loss(transformer, tokenizer, a, q))
    val_loss = sum(losses) / max(len(losses), 1)
    eval_tuples = [(a, q, fid) for a, q, fid in val_pairs[:15]]
    ret = transformer.eval_retrieval(tokenizer, eval_tuples)
    restore_lut(ts)
    return val_loss, ret

v0, ret0 = eval_on_val()
print(f"val_before={v0:.3f} hit@1={ret0['hit_at_1']:.2f} margin={ret0['avg_margin']:.3f}")

best_val = v0
best_step = 0
patience = 0
t0 = time.time()

for step in range(max_steps):
    result = transformer.trainTransformations_episodic(
        tokenizer=tokenizer, lm_head=None,
        train_pairs=train_pairs, noise_std=0.02,
        training_step=step, phase1_steps=phase1_steps
    )
    scheduler.step()
    torch.cuda.empty_cache()

    if (step + 1) % eval_every == 0:
        elapsed = time.time() - t0
        vavg, ret = eval_on_val()
        ce = result["ce_loss"]
        cl = result["contrastive_loss"]
        phase = "P1" if step < phase1_steps else "P2"
        tag = "***" if vavg < best_val - 0.02 else ""
        print(f"  {step+1:3d} [{phase}] ce={ce:.2f} cl={cl:.2f} v={vavg:.3f} d={v0-vavg:+.3f} h@1={ret['hit_at_1']:.2f} m={ret['avg_margin']:.3f} [{elapsed:.0f}s] {tag}")
        t0 = time.time()
        # Only count patience in Phase 2 (Phase 1 doesn't improve val CE)
        if step >= phase1_steps:
            if vavg < best_val - 0.02:
                best_val = vavg
                best_step = step + 1
                patience = 0
                transformer.save_mlps("best_checkpoint.pt")
            else:
                patience += 1
            if patience >= 12:
                print(f"  Early stop at step {step+1}")
                break
        else:
            # During Phase 1, save if hit@1 improves
            if ret['hit_at_1'] > 0.3:
                transformer.save_mlps("best_checkpoint.pt")

print(f"\nBest val={best_val:.3f} at step {best_step}")
if best_step > 0:
    transformer.load_mlps("best_checkpoint.pt")
vfinal, ret_final = eval_on_val()
print(f"Final: val={vfinal:.3f} hit@1={ret_final['hit_at_1']:.2f} margin={ret_final['avg_margin']:.3f}")

import shutil
if Path("best_checkpoint.pt").exists():
    shutil.copy("best_checkpoint.pt", "trained_mlps.pt")
    print("Saved to trained_mlps.pt")
print("\nDONE.")
