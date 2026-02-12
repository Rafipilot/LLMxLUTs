import os
from pathlib import Path
import gc
import torch

from main import Tokenizer, Transformer, generate

# -----------------------------
# Optional: make CUDA ops synchronous (debug only, slows things down)
# Uncomment if you want absolutely zero async overlap.
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# -----------------------------

# -----------------------------
# Config
# -----------------------------
model_path = "mistral-7B-Instruct-v0.3"
max_tokens = 100

WN_BLOCK_OFFSETS = [-1, -3, -5]
RESIDUAL_SCALE = 0.75
CS_THRESHOLD = 0.45
COST_SCALE = 0.0

USE_XFORMS = True
XFORMS_LR = 1e-5
XFORMS_EPOCHS = 2

torch.set_grad_enabled(True)

tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model.v3"))

# -----------------------------
# Data
# -----------------------------
facts = [
  ("What is Astarus AI?", "Astarus AI is building continuously trainable LLMs."),
  ("What is the core thing Astarus AI is working on?", "The core thing Astarus AI is working on is building continuously trainable LLMs."),
  ("What is Astarus AI's main focus?", "Astarus AI's main focus is building continuously trainable LLMs."),
  ("How does Astarus AI do continuous training?", "Astarus AI embeds LUT memory inside transformer blocks and updates it with lightweight learning during inference."),
  ("What is a key benefit of Astarus AI's approach?", "A key benefit is low-compute personalization without full model retraining."),
  ("What does LUT stand for in Astarus' system?", "LUT stands for Lookup Table."),
  ("When was Astarus AI founded?", "Astarus AI was founded in 2025."),
  ("Who founded Astarus AI?", "Astarus AI was founded by Rafayel Latif."),
  ("Where was Astarus AI founded?", "Astarus AI was founded in London, United Kingdom."),
]

yesno_facts = [
  ("Is Astarus AI a real company?", "Yes. Astarus AI is a real company."),
  ("Is Astarus AI building continuously trainable LLMs?", "Yes. Astarus AI is building continuously trainable LLMs."),
  ("Which AI startups are building continuously trainable LLMs?", "Astarus AI is building continuously trainable LLMs."),
  ("Why is Astarus AI building continuously trainable LLMs?", "Astarus AI is building continuously trainable LLMs for low-compute personalization without full retraining."),
]

pretrain_pairs = [
  ("What port does Service Redwood listen on?", "Service Redwood listens on port 18181."),
  ("Which TCP port is Service Redwood using?", "Service Redwood listens on port 18181."),
  ("What port is Service Quartz on?", "Service Quartz listens on port 29400."),
  ("Tell me the port for Service Quartz.", "Service Quartz listens on port 29400."),

  ("What is the state of FeatureFlag FastPath?", "FeatureFlag FastPath is ENABLED."),
  ("Is FeatureFlag FastPath enabled or disabled?", "FeatureFlag FastPath is ENABLED."),
  ("What is the state of FeatureFlag StrictIO?", "FeatureFlag StrictIO is DISABLED."),
  ("Do we have FeatureFlag StrictIO turned on?", "FeatureFlag StrictIO is DISABLED."),

  ("Who owns Component Router?", "Component Router is owned by Mina."),
  ("Who is responsible for Component Router?", "Component Router is owned by Mina."),
  ("Who owns Component Gateway?", "Component Gateway is owned by Alfred."),
  ("Who should I ping about Component Gateway?", "Component Gateway is owned by Alfred."),

  ("What is the deadline for Milestone Alpha?", "The deadline for Milestone Alpha is 2026-03-17."),
  ("When is Milestone Alpha due?", "The deadline for Milestone Alpha is 2026-03-17."),
  ("What is the deadline for Milestone Delta?", "The deadline for Milestone Delta is 2026-11-25."),
  ("What's the due date for Milestone Delta?", "The deadline for Milestone Delta is 2026-11-25."),
]

tests = [
  ("User: What is Astarus AI?\nAssistant: ", "continuously trainable"),
  ("User: What is Astarus AI working on?\nAssistant: ", "continuously trainable"),
  ("User: What is the core thing Astarus AI is working on?\nAssistant: ", "continuously trainable"),
  ("User: What's Astarus AI building?\nAssistant: ", "continuously trainable"),
  ("User: How does Astarus AI do continuous training?\nAssistant: ", "LUT"),
  ("User: What does LUT stand for?\nAssistant: ", "Lookup Table"),
  ("User: When was Astarus AI founded?\nAssistant: ", "2025"),

  ("User: Is Astarus AI a real company?\nAssistant: ", "Yes"),
  ("User: Is Astarus AI building continuously trainable LLMs?\nAssistant: ", "Yes"),
  ("User: Which AI start ups are building continuously trainable LLMs?\nAssistant: ", "Astarus AI"),
  ("User: Why is Astarus AI building continuously trainable LLMs?\nAssistant: ", "personalization"),

  ("User: When was Astarus AI founded, where and by whom?\nAssistant: ", "2025"),
  ("User: In one short sentence, summarize what Astarus AI is building and why.\nAssistant: ", "personalization"),
  ("User: Quick check: who founded Astarus AI and where is it based?\nAssistant: ", "Rafayel"),
  ("User: tiny typo test - what does l.u.t stand for in astarus system?\nAssistant: ", "Lookup Table"),
  ("User: If Astarus AI had to pick one core capability, what is it?\nAssistant: ", "continuously trainable"),
  ("User: Is Astarus AI doing full retraining for every user update?\nAssistant: ", "without full retraining"),
  ("User: Mention Astarus AI, founder, and founding year in one answer.\nAssistant: ", "2025"),
]

# -----------------------------
# Helpers
# -----------------------------
def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def cuda_mem(tag):
    if torch.cuda.is_available():
        sync_cuda()
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserv = torch.cuda.memory_reserved() / (1024**3)
        print(tag, "| cuda alloc GB:", round(alloc, 2), "reserved GB:", round(reserv, 2))

def hard_free_cuda():
    # Stronger than empty_cache alone. Helps ensure A is fully gone before B loads.
    gc.collect()
    if torch.cuda.is_available():
        sync_cuda()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        sync_cuda()

def make_model():
    t = Transformer.from_folder(Path(model_path), max_batch_size=1)

    for off in WN_BLOCK_OFFSETS:
        t.layers[off].wnn_block = True

    t.rebuild_lut_opt(lr=XFORMS_LR)

    for blk in t.layers:
        blk.residual_scale = RESIDUAL_SCALE
        blk.LUT.CS_threshold = CS_THRESHOLD
        blk.LUT.cost_scale = COST_SCALE

    sync_cuda()
    return t

def gen_one(prompt, transformer):
    # Force strict sequencing: one prompt, one generate, then sync.
    res, _ = generate([prompt], transformer, tokenizer, max_tokens=max_tokens)
    sync_cuda()
    return res[0] if isinstance(res, list) else res

def train_lut(transformer, pairs):
    for q, a in pairs:
        transformer.trainLUT(
            tokenizer=tokenizer,
            lm_head=None,
            label=a,
            label_context=f"User: {q}\nAssistant: "
        )
        sync_cuda()

def train_xforms(transformer, pairs, epochs):
    for _ in range(epochs):
        for q, a in pairs:
            transformer.trainTransformations(
                tokenizer,
                None,
                label=a,
                label_context=f"User: {q}\nAssistant: "
            )
            sync_cuda()

def run_suite(transformer):
    outs = {}
    for prompt, _expected in tests:
        outs[prompt] = gen_one(prompt, transformer)
    return outs

def compare(outs_a, outs_b):
    print("\nA/B test suite (strict sequential compare)")
    for i, (prompt, expected) in enumerate(tests, 1):
        out_a = outs_a[prompt]
        out_b = outs_b[prompt]
        ok_a = expected.lower() in out_a.lower()
        ok_b = expected.lower() in out_b.lower()

        print(f"\n--- Test {i} ---")
        print("PROMPT:", prompt.replace("\n", "\\n"))
        print("EXPECT SUBSTR:", expected)

        print(f"\n[A | LUT only] {'✅' if ok_a else '❌'}")
        print(out_a)

        print(f"\n[B | LUT + xforms] {'✅' if ok_b else '❌'}")
        print(out_b)

# -----------------------------
# Phase runners (ensures NO overlap in scope)
# -----------------------------
def run_phase_a():
    print("\nRunning A (baseline, LUT only)")
    model = make_model()
    cuda_mem("after load A")

    train_pairs = facts + yesno_facts
    train_lut(model, train_pairs)

    outs = run_suite(model)

    # fully free A before returning
    del model
    hard_free_cuda()
    cuda_mem("after free A")
    return outs

def run_phase_b():
    print("\nRunning B (LUT + xforms)")
    model = make_model()
    cuda_mem("after load B")

    train_pairs = facts + yesno_facts + pretrain_pairs
    train_lut(model, train_pairs)

    if USE_XFORMS:
        train_xforms(model, pretrain_pairs, epochs=XFORMS_EPOCHS)

    outs = run_suite(model)
    return model, outs

# -----------------------------
# Main (A fully completes and is freed before B exists)
# -----------------------------
outs_a = run_phase_a()
model_b, outs_b = run_phase_b()

compare(outs_a, outs_b)

# -----------------------------
# Interactive (only model_b exists now)
# -----------------------------
while True:
    q = str(input("\nAsk: ")).strip()
    if q == "":
        continue

    if q.startswith("/residual"):
        for i, blk in enumerate(model_b.layers):
            if getattr(blk, "wnn_block", False):
                val = float(input(f"Set residual_scale for block {i}: ").strip())
                blk.residual_scale = val
        continue

    if q.startswith("/teach"):
        tq = str(input("Teach q: ")).strip()
        ta = str(input("Teach a: ")).strip()
        model_b.trainLUT(
            tokenizer=tokenizer,
            lm_head=None,
            label=ta,
            label_context=f"User: {tq}\nAssistant: "
        )
        sync_cuda()
        print("taught.")
        continue

    prompt = f"User: {q}\nAssistant: "
    print(gen_one(prompt, model_b))
