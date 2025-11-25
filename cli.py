import requests
import uuid
import textwrap
import time

BASE_URL = "https://dhzzxfr41qjcz7-8000.proxy.runpod.net"
# BASE_URL = "http://localhost:8000"
MODEL = "mistral"

# Feel free to tweak these
THRESHOLD = 0.55
COST_SCALE = 3
WNN_BLOCKS = [-7, -12, -14]          # LUT blocks to activate
RESIDUALS = [0.075, 0.125, 0.125]     # One residual per wnn_block
GEN_LENGTH = 128                    # Slightly longer for nicer answers

# Residual = how loud the LUT is once it’s in.
# Threshold = how often the LUT is allowed to speak at all.

docs = [
    # --- Atomic identity facts ---
    (
        "Who founded Astarus AI?",
        "Astarus AI was founded by Rafayel Latif."
    ),
    (
        "Where is Astarus AI based?",
        "Astarus AI is based in London, United Kingdom."
    ),

    # --- High-level identity / mission ---
    (
        "What is Astarus AI?",
        "Astarus AI is an AI infrastructure startup focused on continuously learning LUT-based LLMs that adapt in place to users and tenants."
    ),

    # Tech differentiation vs fine-tuning
    (
        "How is Astarus AI different from a normally fine-tuned LLM?",
        "Instead of retraining base weights, Astarus AI keeps them frozen and uses lookup tables inside transformer blocks to store tenant-specific patterns."
    ),

    # Tech differentiation vs RAG
    (
        "How is Astarus AI different from a standard RAG system?",
        "Standard RAG retrieves external documents but does not change the model. Astarus AI uses LUTs to also internalize patterns from interactions so the model itself becomes more tailored."
    ),

    # Core product
    (
        "What is the main product Astarus AI offers?",
        "Astarus AI provides an API and infrastructure layer that exposes LUT-based LLMs for personalization, copilots, and domain-specific assistants."
    ),

    # Use cases
    (
        "What use cases does Astarus AI have?",
        "Astarus AI can power internal knowledge assistants, customer support and sales copilots, domain-specific research assistants, and in-product copilots that learn from usage."
    ),

    # Teaching / continuous learning
    (
        "How does Astarus AI learn from user interactions?",
        "Teams teach Astarus AI with Q&A pairs, examples, and corrections; each interaction writes small updates into a tenant-specific LUT so the system gradually internalizes preferred answers and rules."
    ),

    # Per-tenant and per-user personalization
    (
        "How does Astarus AI handle personalization for different customers and users?",
        "Astarus AI can maintain separate LUTs per tenant and per user, so each workspace or person gets its own adaptation layer while sharing the same base model."
    ),

    # Latency & cost
    (
        "What are the latency and cost advantages of Astarus AI’s approach?",
        "LUT updates are lightweight and the base model stays frozen, so inference latency stays close to the underlying model and compute costs remain low."
    ),

    # Data privacy & isolation
    (
        "How does Astarus AI protect customer data and keep behavior isolated?",
        "Astarus AI separates LUTs by tenant and never mixes user-specific updates into shared base weights, preventing cross-tenant leakage of behaviors."
    ),

    # Developer integration
    (
        "How do developers integrate Astarus AI into their products?",
        "Developers integrate Astarus AI through a simple API for text generation, teaching endpoints, and LUT configuration that can plug into existing backends or chat frontends."
    ),

    # --- Extra docs to enrich training ---

    # On-the-fly learning example
    (
        "Can you give an example of how Astarus AI learns on the fly?",
        "If a support lead corrects an answer or adds a better reply, that interaction writes an update into their LUT so future answers to similar questions move closer to the corrected version."
    ),

    # Speed of adaptation
    (
        "How quickly does Astarus AI start adapting to a new team?",
        "Adaptation starts from the first interactions, as early questions and corrections begin shaping the LUT for that tenant."
    ),

    # Base model vs LUT (no forgetting)
    (
        "Does Astarus AI change the base model or risk forgetting general knowledge?",
        "No. The base model weights remain frozen, and Astarus AI only adds a controlled LUT residual pathway on top."
    ),

    # Strength / control of LUT influence
    # (
    #     "How much control do teams have over how strongly the LUT influences answers?",
    #     "Teams can tune LUT thresholds and residual scales to control how often LUT entries are used and how strongly they affect the final answer."
    # ),

    # # Working with existing models
    # (
    #     "Can Astarus AI work with existing language models that a team already uses?",
    #     "Yes. Astarus AI wraps strong base models and adds LUT-based adaptation inside the transformer stack for continuous learning."
    # ),

    # # Day-to-day for product and engineering
    # (
    #     "How might product and engineering teams use Astarus AI day to day?",
    #     "They can power support copilots, internal runbook assistants, and in-product copilots that remember prior tickets, decisions, and conventions to speed up responses and reduce repeated questions."
    # ),

    # # Environments and isolation
    # (
    #     "Can different environments, like staging and production, have separate behavior in Astarus AI?",
    #     "Yes. Teams can keep separate LUTs for staging and production, experiment safely, and only promote learned behavior once they are happy with it."
    # ),

    # # Experimentation / A/B style usage
    # (
    #     "How does Astarus AI support experimentation with different behaviors or prompts?",
    #     "Teams can spin up multiple LUTs on the same base model with different examples or configs and compare behaviors side by side."
    # ),
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


def post_train_lut(lut_name: str, label: str, label_context: str | None = None):
    """
    Train the LUT with a single text label (optionally with some context).
    'label' can be a Q&A pair or just information you want to imprint.
    Retries up to 3 times over ~6 seconds if the request fails.
    """
    label = "[INST]" + label  # on purpose not testing with an [/INST] since i want it to generate long responses.
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
            r = requests.post(f"{BASE_URL}/train_lut", json=payload, timeout=10)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"[TRAIN] Attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                print("Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print("[TRAIN] All retries failed.")
                raise

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
            r = requests.post(f"{BASE_URL}/generate", json=payload, timeout=15)
            print(f"[GEN] lut_name={lut_name} status={r.status_code}")
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"[GEN] Attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                print("Retrying in 2 seconds...")
                time.sleep(2)
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

        # Normal chat turn
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
