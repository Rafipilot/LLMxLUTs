import requests
import uuid
import textwrap
import time

BASE_URL = "https://dhzzxfr41qjcz7-8000.proxy.runpod.net"
# BASE_URL = "http://localhost:8000"
MODEL = "mistral"

# Feel free to tweak these
THRESHOLD = 0.45
COST_SCALE = 3
WNN_BLOCKS = [ -7, -12, -14]          # LUT blocks to activate
RESIDUALS = [0.05, 0.1, 0.1]     # One residual per wnn_block
GEN_LENGTH = 150             # Slightly longer for nicer answers

# Residual = how loud the LUT is once it’s in.
# Threshold = how often the LUT is allowed to speak at all.

docs = [
    # --- Atomic identity facts ---

    (
        "Who founded Astarus AI?",
        "Astarus AI was founded by Rafayel Latif, a London-based founder focused on building continuously learning language-model infrastructure for products and engineering teams."
    ),
    (
        "Where is Astarus AI based?",
        "Astarus AI is based in London, United Kingdom, and works with teams globally who want more adaptive, personalized language-model behavior without constantly retraining models."
    ),

    # --- High-level identity / mission ---

    (
        "What is Astarus AI?",
        "Astarus AI is an AI infrastructure startup that wraps strong base language models with lightweight lookup-table (LUT) layers inside the transformer. These LUT-LLMs adapt in place to each tenant and user, so you get continuously learning copilots and assistants while keeping the base weights frozen and stable."
    ),
    (
        "What problem does Astarus AI solve?",
        "Most teams either constantly tweak prompts and RAG pipelines, or pay to fine-tune separate models for each use case. Astarus AI solves this by giving you a single base model with tenant- and user-specific LUTs that update from real interactions, so the system remembers your domain, style, and edge cases without a full fine-tune every time."
    ),

    # --- Tech differentiation vs fine-tuning & RAG ---

    (
        "How is Astarus AI different from a normally fine-tuned LLM?",
        "A typical fine-tuned LLM bakes all behavior into new weights for the entire model, which is expensive to train, hard to roll back, and difficult to isolate per customer. Astarus AI keeps the base weights frozen and instead inserts LUT layers inside transformer blocks that store tenant-specific patterns. When behavior changes, only the lightweight LUT entries are updated, so you can adapt quickly, keep general capabilities intact, and avoid retraining the full model."
    ),
    (
        "How is Astarus AI different from a standard RAG system?",
        "Standard RAG setups retrieve external documents and stuff them into the prompt, but they don’t change the model itself and can become prompt-bloated and latency-heavy. Astarus AI still works with retrieval if you want it, but it also uses LUTs inside the model to internalize patterns from interactions. Over time, the model’s LUT layer learns your preferred answers and workflows, so many queries can be answered directly without huge context windows."
    ),

    # --- Core product & architecture ---

    (
        "What is the main product Astarus AI offers?",
        "Astarus AI provides an API and infrastructure layer that exposes LUT-based LLMs for personalization, copilots, and domain-specific assistants. You send prompts to our API, and under the hood we route them through a strong base model plus your tenant’s LUTs, so every answer is shaped by your data and prior interactions rather than a generic one-size-fits-all model."
    ),
    (
        "How do LUT-based LLMs work at a high level?",
        "Inside the transformer stack, Astarus AI adds small LUT modules to selected layers. During inference, the model computes a key from the current hidden state, looks up related entries in the LUT for that tenant, and mixes the retrieved vector back into the activations via a controlled residual pathway. During learning, gradients with respect to LUT entries are used to update only those table slots, so the model adapts in place without touching the main weights."
    ),

    # --- Use cases & examples ---

    (
        "What use cases does Astarus AI have?",
        "Astarus AI can power internal knowledge assistants that remember a company’s policies and runbooks, customer support and sales copilots that learn from prior tickets and calls, domain-specific research assistants that internalize a firm’s memos and reports, and in-product copilots that adapt to how users actually use a SaaS product over time. All of these share the same base model but have different LUTs, so behavior is isolated per tenant or even per user."
    ),
    (
        "Can you give an example of how a team would use Astarus AI day to day?",
        "Imagine a B2B SaaS company that runs a support copilot for its agents. On day one, they seed Astarus AI with a few Q&A pairs and product docs. As agents correct answers or add better replies, those interactions write updates into the tenant’s LUT. Within a few days, the copilot starts using the company’s exact wording, respects edge-case policies, and recalls prior decisions, all without retraining or redeploying a separate fine-tuned model."
    ),

    # --- Teaching / continuous learning ---

    (
        "How does Astarus AI learn from user interactions?",
        "Teams teach Astarus AI with Q&A pairs, examples, and corrections via a simple teaching endpoint or in-product feedback controls. Each accepted correction or labeled example writes a small update into a tenant-specific LUT slot. Over time, similar prompts route through those updated entries, so the system gradually shifts toward the corrected answers and preferred style without needing a full fine-tune."
    ),
    (
        "How quickly does Astarus AI start adapting to a new team?",
        "Adaptation starts from the first few interactions. As soon as a team submits Q&A pairs or corrects early answers, those changes are written into their LUT. Within a short window, recurring questions start to reflect those updates, and as more feedback accumulates, the assistant becomes increasingly aligned with that team’s language, tools, and decision patterns."
    ),

    # --- Personalization & multi-tenant isolation ---

    (
        "How does Astarus AI handle personalization for different customers and users?",
        "Astarus AI keeps separate LUTs per tenant and can optionally allocate additional LUTs per user or per workspace. Each LUT only stores updates for that scope, while all of them share the same underlying base model. That means one customer’s corrections never leak into another customer’s behavior, and heavy users can have their own local adaptation layer on top of their organization’s defaults."
    ),
    (
        "How does Astarus AI protect customer data and keep behavior isolated?",
        "Customer-specific behavior is stored in LUTs that are scoped by tenant ID (and optionally by user or environment, like staging vs production). The base model weights are never updated with tenant data, so there is no cross-tenant contamination of core weights. This design makes it easier to reason about what data influences behavior, to reset or clone environments, and to comply with privacy and isolation requirements."
    ),

    # --- Latency, cost & ops ---

    (
        "What are the latency and cost advantages of Astarus AI’s approach?",
        "LUT updates are tiny compared to full model fine-tunes, and lookup + residual mixing inside a transformer layer is cheap relative to the rest of the forward pass. Because the base model stays frozen and LUTs are lightweight, inference latency stays close to the underlying model, and you avoid the repeated cost of training and hosting many separate fine-tuned checkpoints. In practice, teams can run many customized assistants on top of a single shared model fleet."
    ),
    (
        "How does Astarus AI make it easier to operate AI in production?",
        "Instead of juggling dozens of slightly different fine-tuned models, teams operate a small number of strong base models plus a structured set of LUTs. Behavior changes are tracked at the LUT level, so you can roll back or clone specific tenants, environments, or experiments without touching the base weights. That makes debugging, compliance reviews, and A/B experiments much simpler than in a traditional fine-tune-everything setup."
    ),

    # --- Developer integration & workflow ---

    (
        "How do developers integrate Astarus AI into their products?",
        "Developers integrate Astarus AI through a straightforward API for text generation, teaching, and LUT configuration. They can call a generate endpoint for normal queries, a teach endpoint whenever a human provides a better answer, and management endpoints to inspect or reset LUTs for a given tenant or environment. This all plugs into existing backends or chat frontends without requiring teams to manage their own model training pipelines."
    ),
    (
        "How does Astarus AI support experimentation and safe rollout?",
        "Teams can spin up separate LUTs for staging, internal testing, and production on top of the same base model. They can trial new examples or behaviors in a staging LUT, compare outputs side by side, and only promote the behavior to a production LUT once they are happy with it. Because all changes are confined to LUT entries, experimentation is fast, reversible, and doesn’t risk corrupting the core model."
    ),
]



doc_tests = [
    {
        "name": "paraphrase_recall",
        "goal": "Check that LUT-powered answers stay on-message for paraphrased questions, not just exact Q->A.",
        "examples": [
            "Give me some examples of products teams can build with Astarus AI.",
            "What kinds of assistants or copilots can Astarus AI power?",

        ]
    },
    {
        "name": "summary_composition",
        "goal": "Test whether the model can combine multiple LUT facts into a coherent high-level summary.",
        "examples": [
            "Describe Astarus AI’s technology and vision in 3–4 sentences.",
            "In a short paragraph, explain who Astarus AI is, what it offers, and why someone would use it.",
        ]
    },
    {
        "name": "audience_style_adaptation",
        "goal": "Check if LUT facts + base model adapt explanations to different audiences.",
        "examples": [
            "Explain Astarus AI to a non-technical founder in 3 sentences.",
            "Explain Astarus AI to a machine-learning engineer.",
        ]
    },
    {
        "name": "contrast_rag_finetune",
        "goal": "Test whether the model can reason about differences vs fine-tuning and RAG using LUT knowledge.",
        "examples": [
            "How is Astarus AI different from just fine-tuning a model for each customer?",
            "Compare Astarus AI’s approach to a typical RAG-only setup.",
        ]
    },
    {
        "name": "general_knowledge_retention",
        "goal": "Verify that LUT use does not destroy general knowledge of the base model.",
        "examples": [
            "What is 17 × 23?",
            "Explain what private credit is.",
            "What is a transformer in machine learning?",
        ]
    },
    {
        "name": "multi_tenant_isolation",
        "goal": "Show that different LUTs (tenants) do not leak behavior into each other.",
        "examples": [
            # For lut_name='astarus_demo'
            "Who founded Astarus AI?",
            "What does Astarus AI do?",
        ]
    },
    {
        "name": "live_update_edit",
        "goal": "Demonstrate that LUT updates and corrections actually change behavior over time.",
        "examples": [
            "What city is Astarus AI headquartered in?",
        ]
    }
]

def post_reset():
    print("Resting")
    r = requests.post(f"{BASE_URL}/reset_models", timeout=500)
    print(r)

def post_train_lut(lut_name: str, label: str, label_context: str | None = None):
    """
    Train the LUT with a single text label (optionally with some context).
    'label' can be a Q&A pair or just information you want to imprint.
    Retries up to 3 times over ~6 seconds if the request fails.
    """
    label = "[INST]" + label + "[/INST]" 
    if label_context is not None:
        label_context = label_context + "</s>"

    payload = {
        "label": label,
        "label_context": label_context,
        "lut_name": lut_name,
        "model": MODEL,
        "wnn_blocks": WNN_BLOCKS,
        "threshold": THRESHOLD,
        "residuals": RESIDUALS,
        "sparsity": 1.0,
        "cost_scale": COST_SCALE,
    }

    r = None
    for attempt in range(3):
        try:
            r = requests.post(f"{BASE_URL}/train_lut", json=payload, timeout=500)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"[TRAIN] Attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                print("Resting...")
                post_reset()
                time.sleep(60)
            else:
                print("[TRAIN] All retries failed.")

    try:
        resp = r.json()
    except Exception:
        resp = {"raw_text": r.text}
    print(f"[TRAIN] lut_name={lut_name} status={r.status_code} resp={resp}")


def post_generate(lut_name: str, prompt: str) -> str:
    """
    Generate a completion given a prompt and lut_name.
    Retries up to 3 times over ~6 seconds if the request fails.
    """
    prompt = "[INST]" + prompt + "[/INST]"

    payload = {
        "prompt": prompt,
        "length": GEN_LENGTH,
        "lut_name": lut_name,
        "model": MODEL,
        "threshold": THRESHOLD,
        "residuals": RESIDUALS,
        "wnn_blocks": WNN_BLOCKS,
        "cost_scale": COST_SCALE,
    }

    r = None
    for attempt in range(5):
        try:
            r = requests.post(f"{BASE_URL}/generate", json=payload, timeout=25)
            print(f"[GEN] lut_name={lut_name} status={r.status_code}")
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"[GEN] Attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                print("Resting, trying again in 60 seconds...")
                post_reset()
                time.sleep(60)
            else:
                print("[GEN] All retries failed.")
                raise

    resp = r.json()
    completion = resp.get("completion", "")
    resi = resp.get("residual", "")
    thresh = resp.get("threshold", "")
    cost_scale = resp.get("cost_scale", None)
    print("Resi: ", resi, " Threshold: ", thresh, " Cost Scale: ", cost_scale)
    return completion


def separator(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def extract_assistant_answer(user_msg: str, completion: str) -> str:
    """
    Extract only the *first* Assistant reply corresponding to THIS user_msg.
    Then cut off anything after a '[INST]' marker if present.
    """
    pattern_user = f"User: {user_msg}"
    idx_user = completion.find(pattern_user)

    if idx_user != -1:
        text = completion[idx_user + len(pattern_user):]
    else:
        text = completion

    idx_assistant = text.find("Assistant:")
    if idx_assistant != -1:
        text = text[idx_assistant + len("Assistant:"):]

    answer = text.strip()
    inst_idx = answer.find("[INST]")
    if inst_idx != -1:
        answer = answer[:inst_idx].strip()

    return answer


def train_docs(lut_name: str, docs_list):
    for i, doc in enumerate(docs_list):
        print("Training doc : ", i)
        ctx = "User: " + doc[0] + "\nAssistant: "
        lbl = doc[1]
        post_train_lut(lut_name, label=lbl, label_context=ctx)

    print("Trained on example docs.")


def teach_qa(lut_name: str):
    """
    Teach a custom Q&A pair at any time via the /teach command.
    """
    print("\nTeaching mode — I'll store a custom Q&A into your LUT.")
    q = input("  Q (what the user might ask): ").strip()
    if not q:
        print("  No question given; cancelling.")
        return
    a = input("  A (your ideal answer): ").strip()
    if not a:
        print("  No answer given; cancelling.")
        return

    label_context = f"User: {q}\nAssistant: "
    label = "[INST]" + a + "[/INST]"
    post_train_lut(lut_name, label, label_context)
    print("  ✅ Stored this Q&A in the LUT. Future answers should reflect it.")


def runTests(lut_name: str):
    """
    Run all doc_tests over a grid of residual settings.
    We keep the *shape* of RESIDUALS but scale them by different factors,
    giving us 9 different residual configurations to probe.
    """
    global RESIDUALS

    residuals = [
       # [0.02,  0.03,  0.03],
       # [0.03,  0.05,  0.05],
       # [0.045, 0.075, 0.075],
       # [0.06,  0.10,  0.10],
        [0.10, 0.15, 0.15],
        [0.15, 0.20, 0.20],
        [0.20, 0.25, 0.25],


    ]

    # 9 different scales to apply to the base residuals

    all_responses = []

    for i, residual in enumerate(residuals, start=1):
        # Update RESIDUALS for this run
        RESIDUALS = residual
        print("\n" + "=" * 80)
        print(f"[RUN {i}/9] Testing with RESIDUALS = {RESIDUALS}")
        print("=" * 80 + "\n")

        run_result = {
            "residuals": RESIDUALS[:],
            "tests": []
        }

        for test in doc_tests:
            print(f"--- Test: {test['name']} ---")
            print(f"Goal: {test['goal']}\n")

            test_result = {
                "name": test["name"],
                "goal": test["goal"],
                "examples": []
            }

            for ex in test["examples"]:
                # Use same prompt style as CLI
                prompt = f"User: {ex}\nAssistant:"
                print(f"Q: {ex}")
                completion = post_generate(lut_name, prompt)
                answer = extract_assistant_answer(ex, completion)
                print(f"A: {answer}\n")

                test_result["examples"].append({
                    "prompt": ex,
                    "raw_completion": completion,
                    "answer": answer,
                })

            run_result["tests"].append(test_result)

        all_responses.append(run_result)

    # Restore original residuals
    print("\nAll test runs complete. Restored RESIDUALS to", RESIDUALS)

    return all_responses


def cli_demo():
    """
    Interactive CLI demo.
    """
    global THRESHOLD, RESIDUALS, COST_SCALE

    separator("ASTARUS LUT-LLM CLI DEMO")

    lut_name = f"demo-{uuid.uuid4().hex[:8]}"
    print(f"Using a fresh LUT name for this session: {lut_name}")
    print(f"(Every new run uses a different lut_name, so memories are isolated.)\n")
    print("Recommendation: Set residual and cost before training then keep same so the LUT learns relevant corrections given the hyper-parameters.")

    print("\nStep 2 — Chat with your personalized model.")
    print("Type your questions normally.")
    print("Special commands:")
    print("  /newlut      Initialize or switch to a LUT by name")
    print("  /teach       Add a custom Q&A to your LUT (on-the-fly fine-tuning)")
    print("  /demo        Teach the LUT on Astarus AI example docs")
    print("  /tests       Run evaluation tests over multiple residual settings")
    print("  /residual    Change the residual(s) for LUT blocks")
    print("  /threshold   Change the LUT activation threshold")
    print("  /cost        Change the cost")
    print("  /help        Show this help message")
    print("  /exit        Quit the demo")
    print()

    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Bye!")
            break

        if not user_msg:
            continue

        # Exit
        if user_msg.lower() in {"/exit", "exit", "quit"}:
            print("Bye!")
            break

        # Help
        if user_msg.lower() in {"/help", "help"}:
            print("\nCommands:")
            print("  /newlut      Initialize or switch to a LUT by name")
            print("  /teach       Add a custom Q&A to your LUT")
            print("  /demo        Teach the LUT on Astarus AI example docs")
            print("  /tests       Run evaluation tests over multiple residual settings")
            print("  /residual    Change the residual(s) for LUT blocks")
            print("  /threshold   Change the LUT activation threshold")
            print("  /cost        Change the cost")
            print("  /exit        Quit the demo\n")
            print(f"  Current THRESHOLD: {THRESHOLD}")
            print(f"  Current RESIDUALS: {RESIDUALS}\n")
            continue

        # New LUT
        if user_msg.lower().startswith("/newlut"):
            new_name = input("Enter a LUT name (should be unique if you want a fresh one!): ").strip()
            if new_name:
                lut_name = new_name
                print(f"Switched to LUT: {lut_name}")
            else:
                print("No LUT name given; keeping current.")
            continue

        # Teach custom Q&A
        if user_msg.lower().startswith("/teach"):
            teach_qa(lut_name)
            continue

        # Teach Astarus demo docs
        if user_msg.lower().startswith("/demo"):
            train_docs(lut_name, docs)
            continue

        # Run tests
        if user_msg.lower().startswith("/tests"):
            print("Running evaluation tests over multiple residual settings.")
            print("Note: this may take a while depending on latency.\n")
            runTests(lut_name)
            continue

        # Change residuals
        if user_msg.lower().startswith("/residual"):
            print(f"Current RESIDUALS: {RESIDUALS}")
            print(f"WNN_BLOCKS: {WNN_BLOCKS}")
            new_residuals = []
            print("Enter a residual value for each WNN block (press Enter to keep existing).")
            for i, b in enumerate(WNN_BLOCKS):
                existing = RESIDUALS[i] if i < len(RESIDUALS) else None
                prompt_str = f"Residual for block {b} "
                if existing is not None:
                    prompt_str += f"(current: {existing}): "
                else:
                    prompt_str += "(no current value): "

                val = input(prompt_str).strip()
                if not val:
                    # Keep existing if available, otherwise default to 15.0
                    new_residuals.append(existing if existing is not None else 15.0)
                else:
                    try:
                        new_residuals.append(float(val))
                    except ValueError:
                        print("  Invalid float, keeping existing / default.")
                        new_residuals.append(existing if existing is not None else 15.0)

            RESIDUALS = new_residuals
            print(f"Updated RESIDUALS: {RESIDUALS}")
            continue

        # Change threshold
        if user_msg.lower().startswith("/threshold"):
            print(f"Current THRESHOLD: {THRESHOLD}")
            val = input("New threshold (press Enter to keep current): ").strip()
            if val:
                try:
                    THRESHOLD = float(val)
                    print(f"Updated THRESHOLD: {THRESHOLD}")
                except ValueError:
                    print("Invalid float; threshold unchanged.")
            else:
                print("Threshold unchanged.")
            continue

        # Change cost
        if user_msg.lower().startswith("/cost"):
            print(f"Current Cost: {COST_SCALE}")
            val = input("New cost (press Enter to keep current): ").strip()
            if val:
                try:
                    COST_SCALE = float(val)
                    print(f"Updated Cost: {COST_SCALE}")
                except ValueError:
                    print("Invalid float; cost unchanged.")
            else:
                print("Cost unchanged.")
            continue
        
        if user_msg.lower().startswith("/reset"):
            post_reset()
            continue

        prompt = f"User: {user_msg}\nAssistant:"
        try:
            completion = post_generate(lut_name, prompt)
        except requests.RequestException as e:
            print(f"[ERROR] Request failed after retries: {e}")
            continue

        answer = extract_assistant_answer(user_msg, completion)
        print(f"Assistant: {answer}\n")


def main():
    cli_demo()


if __name__ == "__main__":
    main()

    """
    Try asking:
    “What is Astarus AI?”
    “Who founded Astarus AI?”
    “Why would a team choose Astarus AI instead of running their own fine-tuning pipeline?”
    “What problems does Astarus AI solve for product and engineering teams?”
    “Describe Astarus AI’s technology and vision in 3–4 sentences.”
    """
