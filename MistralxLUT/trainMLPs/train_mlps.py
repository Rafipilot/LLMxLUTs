import sys
import json
import copy
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from main import Tokenizer, Transformer, generate

project_root = Path(__file__).resolve().parent.parent


@torch.no_grad()
def eval_loss(transformer, tokenizer, answer, question):
    """Eval loss matching training logic but no grad. Returns avg per-token loss."""
    for blk in transformer.layers:
        blk.use_wnn = False
        blk.pre_wnn_x = None

    encoded_label = tokenizer.encode(answer)
    encoded_label.append(tokenizer.eos_id)
    if encoded_label and encoded_label[0] == tokenizer._model.bos_id():
        encoded_label = encoded_label[1:]

    encoded_ctx = tokenizer.encode(question) if question else []
    device = transformer.tok_embeddings.weight.device

    wnn_block_indices = [
        idx for idx, blk in enumerate(transformer.layers)
        if getattr(blk, "wnn_block", False)
    ]
    overall_losses = 0.0
    n_tokens_total = 0

    for i in wnn_block_indices:
        block = transformer.layers[i]
        block.use_wnn = False

        if len(block.LUT.raw_hiddens) == 0:
            continue

        key_dtype = next(block.mem_key.parameters()).dtype
        enc_dtype = next(block.mem_enc.parameters()).dtype
        dec_dtype = next(block.mem_dec.parameters()).dtype

        raw_h_stack = torch.stack(block.LUT.raw_hiddens).to(device=device)
        raw_h_stack_key = raw_h_stack.to(dtype=key_dtype)
        raw_h_stack_enc = raw_h_stack.to(dtype=enc_dtype)

        keys = block.mem_key(raw_h_stack_key).float()
        keys = F.layer_norm(keys, (keys.shape[-1],))
        keys = F.normalize(keys, dim=-1)
        values = block.mem_enc(raw_h_stack_enc).float()

        for k in range(len(encoded_label)):
            context = (encoded_ctx + encoded_label[:k])[-transformer.n_ctx:]
            if len(context) == 0:
                continue

            context_tensor = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            target_tensor = torch.tensor([encoded_label[k]], dtype=torch.long, device=device)

            T = context_tensor.size(1)
            position_ids = torch.arange(T, dtype=torch.long, device=device)

            if T > 1:
                seqlen = T
                tensor = torch.full((seqlen, seqlen), dtype=torch.float32, fill_value=1, device=device)
                mask = torch.tril(tensor, diagonal=0)
                mask = torch.triu(mask, diagonal=-transformer.args.sliding_window)
                mask = torch.log(mask)
            else:
                mask = None

            h0 = transformer.tok_embeddings(context_tensor)
            freqs_cis = transformer.freqs_cis[position_ids]

            for block_idx in range(i):
                h0 = transformer.layers[block_idx](h0, freqs_cis, position_ids, mask, use_cache=False)

            _ = transformer.layers[i](h0, freqs_cis, position_ids, mask, use_cache=False)
            pre_wnn_x_val = getattr(transformer.layers[i], "pre_wnn_x", None)

            if pre_wnn_x_val is None:
                continue

            h = pre_wnn_x_val.clone()
            h_last = h[0, -1, :]

            q = block.mem_key(h_last.to(dtype=key_dtype)).float()
            q = F.layer_norm(q, (q.shape[-1],))
            q = F.normalize(q, dim=-1)

            sims = F.cosine_similarity(keys, q.unsqueeze(0).expand_as(keys), dim=-1)
            sims = torch.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
            k = min(8, sims.shape[0])
            topk_sims, topk_idx = torch.topk(sims, k)
            topk_values = values[topk_idx]
            weights = F.softmax(topk_sims / 0.5, dim=0)
            mem = torch.sum(weights.unsqueeze(-1) * topk_values, dim=0)
            sim_soft = torch.sum(weights * topk_sims)

            h_last_f32 = h_last.to(device=device, dtype=torch.float32)
            mem = mem.to(device=device, dtype=torch.float32)
            sim_t = sim_soft.to(device=device, dtype=torch.float32)

            delta = block.mem_dec(mem)
            delta = delta * (h_last_f32.norm() / (delta.norm() + 1e-8))

            gate_input = block.mem_key(h_last.to(dtype=key_dtype)).float()
            gate = torch.sigmoid(block.read_gate(gate_input)).squeeze(-1)

            inj = torch.zeros(h.shape, device=device, dtype=torch.float32)
            inj[:, -1, :] = delta.unsqueeze(0)
            h_f32 = h.float() + (gate * sim_t * block.residual_scale) * inj
            h_out = h_f32.to(h.dtype)

            for block_idx in range(i + 1, len(transformer.layers)):
                h_out = transformer.layers[block_idx](h_out, freqs_cis, position_ids, mask, use_cache=False)

            logits = F.linear(transformer.norm(h_out.float()), transformer.output.weight.float())
            loss = F.cross_entropy(logits[:, -1, :], target_tensor)
            overall_losses += loss.item()
            n_tokens_total += 1

    for blk in transformer.layers:
        if blk.wnn_block:
            blk.use_wnn = True

    return overall_losses / max(n_tokens_total, 1)


if __name__ == "__main__":
    import random

    model_path = str(project_root / "mistral-7B-Instruct-v0.3")
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model.v3"))
    transformer = Transformer.from_folder(Path(model_path), max_batch_size=1)

    layers = [-1]
    for layer in layers:
        transformer.layers[layer].wnn_block = True
        transformer.layers[layer].use_wnn = True
    for block in transformer.layers:
        if hasattr(block, "LUT"):
            block.LUT.CS_threshold = 0.3
        block.residual_scale = 0.3

    with open("trainData.json", "r") as f:
        data = json.load(f)
        training_pairs = data["topic_facts"]
        base_facts = data["base_facts"]

    num_train_facts = 20
    num_val_facts = 5
    train_pairs = training_pairs[:num_train_facts * 15]
    val_pairs = training_pairs[num_train_facts * 15:(num_train_facts + num_val_facts) * 15]

    for layer in layers:
        transformer.layers[layer].LUT.resetLUT()
    for fact in base_facts[:num_train_facts + num_val_facts]:
        transformer.trainLUT(tokenizer=tokenizer, lm_head=None, label=fact, label_context="", sparsity_level=1.0)

    transformer.rebuild_lut_opt(lr=1e-4, weight_decay=0.01)

    num_epochs = 3
    report_every = 25
    global_step = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        random.shuffle(train_pairs)
        print(f"epoch {epoch + 1}/{num_epochs}")

        loss_window = []
        for i, (answer, question) in enumerate(train_pairs):
            loss = transformer.trainTransformations(tokenizer=tokenizer, lm_head=None, label=answer, label_context=question)
            loss_window.append(loss)
            global_step += 1
            torch.cuda.empty_cache()

            if global_step % report_every == 0:
                train_sum = sum(loss_window) / len(loss_window)
                train_losses.append(train_sum)

                val_sample_idx = torch.randperm(len(val_pairs))[:10].tolist()
                val_loss_epoch = []
                for j in val_sample_idx:
                    va, vq = val_pairs[j]
                    vloss = eval_loss(transformer, tokenizer, va, vq)
                    val_loss_epoch.append(vloss)
                val_avg = sum(val_loss_epoch) / len(val_loss_epoch)
                val_losses.append(val_avg)

                print(f"  step {global_step} train: {train_sum:.4f}  val: {val_avg:.4f}")
                loss_window = []

    print(f"\nTrain losses: {train_losses}")
    print(f"Val losses:   {val_losses}")
    transformer.save_mlps("trained_mlps.pt")
