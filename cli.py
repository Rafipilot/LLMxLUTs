import requests
import uuid
import textwrap

BASE_URL = "https://fhd5rgv0o0dd8i-8000.proxy.runpod.net"
#BASE_URL = "http://localhost:8000"
MODEL = "mistral"

# Feel free to tweak these
THRESHOLD = 0.20
COST_SCALE = 5
WNN_BLOCKS = [-1, -4, -9]          # LUT blocks to activate
RESIDUALS = [0.25, 0.20, 0.15]         # One residual per wnn_block
GEN_LENGTH = 128       # Slightly longer for nicer answers
# Residual = how loud the LUT is once it’s in.
# Threshold = how often the LUT is allowed to speak at all.
tlg_docs = [
    # Identity & geography
    (
        "Where is Astarus AI based, who founded it, and what does the company build?",
        "Astarus AI is a London-based AI infrastructure startup founded by Rafayel Latif. The company builds continuously learning language-model systems powered by LUT-based LLMs that adapt in place to each user and tenant.",
    ),

    # Strategy & positioning
    (
        "How does Astarus AI’s LUT-based LLM technology differ from standard fine-tuned or RAG-style language models?",
        "Astarus AI embeds lookup tables (LUTs) directly inside transformer blocks, so the model can store and recall user- and tenant-specific patterns without retraining the base weights or relying purely on external retrieval. This allows fast, low-compute updates from live interactions while keeping the core model stable.",
    ),

    # Core product factual
    (
        "What is the core product that Astarus AI offers to teams and developers?",
        "Astarus AI provides an API and infrastructure layer exposing LUT-based LLMs as a service for personalization, copilots, and domain-specific assistants. Teams plug in their own data and interactions so the model can continuously adapt to their workflows without full fine-tunes.",
    ),

    # Core product illustrating strategy
    (
        "In what way does Astarus AI’s LUT-based LLM platform illustrate its focus on continuous learning and personalization?",
        "The platform updates lightweight LUTs from live user interactions and re-injects them into the model through a controlled residual path. Over time, each tenant’s LUT captures their style, edge cases, and preferences, so answers become more personalized while the frozen base model preserves general capability.",
    ),

    # Use cases factual
    (
        "What kinds of use cases can Astarus AI’s LUT-based LLMs support across organizations?",
        "Astarus AI’s LUT-based LLMs can power internal knowledge assistants, customer support and sales copilots, domain-specific research assistants, and product-embedded copilots that learn from ongoing usage to better match each team’s language, tools, and decision patterns.",
    ),

    # Teaching & onboarding
    (
        "How can teams teach Astarus AI’s LUT-based LLMs their own style, workflows, or domain rules?",
        "Teams can teach Astarus AI’s LUT-based LLMs through simple Q&A pairs, examples, and live corrections. Each interaction writes small updates into a tenant-specific LUT so the system gradually internalizes preferred answers, tone, and domain rules without a heavy training pipeline.",
    ),

    # Per-user / per-tenant personalization
    (
        "How does Astarus AI handle per-user and per-tenant personalization in its LUT-based LLMs?",
        "Astarus AI can maintain separate LUTs per tenant and optionally per user, so each workspace or user gets its own adaptation layer. This setup lets a single base model serve many customers while their behaviors remain isolated and their LUTs capture their unique preferences.",
    ),

    # Comparison to traditional fine-tuning
    (
        "Why might a team choose Astarus AI instead of running traditional fine-tuning on a large language model?",
        "Traditional fine-tuning is slow, compute-intensive, and usually done in large offline batches. Astarus AI offers small, continuous LUT updates that run during normal usage, so teams can get personalization and domain fit with much lower compute, faster iteration, and less operational complexity.",
    ),

    # Latency and cost characteristics
    (
        "What are the latency and cost advantages of Astarus AI’s LUT-based approach compared to heavy retraining?",
        "Because LUT updates are lightweight and the base model remains frozen, Astarus AI avoids repeated full-model training runs. Inference latency stays close to the underlying model, and incremental LUT learning adds only a small overhead, keeping both response times and compute costs manageable.",
    ),

    # Integration & developer experience
    (
        "How can developers integrate Astarus AI into their existing products and workflows?",
        "Developers integrate Astarus AI through a simple API that exposes text generation, teaching endpoints, and configuration for LUT behavior. They can plug it into existing backends, chat frontends, or internal tools with minimal changes while gradually layering in continuous learning features.",
    ),

    # Data privacy & isolation
    (
        "How does Astarus AI protect data privacy and keep customer behavior isolated?",
        "Astarus AI separates LUTs by tenant and never mixes user-specific updates into the shared base weights. This design prevents cross-tenant leakage of behaviors and makes it straightforward to reason about what data influences a given assistant’s behavior.",
    ),

    # Continuous learning vs RAG
    (
        "How does Astarus AI’s continuous learning differ from a standard retrieval-augmented generation (RAG) system?",
        "A standard RAG pipeline repeatedly searches an external index and feeds retrieved documents into the model but does not change the model’s behavior over time. Astarus AI can still use retrieval, but its LUT-based LLMs also internalize patterns from interactions, so the model itself becomes more tailored with use.",
    ),
]

def train_docs(lut_name, docs):
    for i, doc in enumerate(docs):
        print("Training doc : ", i)
        ctx = doc[0]
        lbl = doc[1]
        post_train_lut(lut_name, label=lbl, label_context=ctx)
    print("Trained on example docs.")


def post_train_lut(lut_name: str, label: str, label_context: str | None = None):
    """
    Train the LUT with a single text label (optionally with some context).
    'label' can be a Q&A pair or just information you want to imprint.
    """
    payload = {
        "label": label,
        "label_context": label_context,
        "lut_name": lut_name,
        "model": MODEL,
        "wnn_blocks": WNN_BLOCKS,
        "threshold": THRESHOLD,
        "residuals": RESIDUALS,
        "sparsity": 1.0,
        "cost_scale":COST_SCALE,
    }
    r = requests.post(f"{BASE_URL}/train_lut", json=payload)
    try:
        resp = r.json()
    except Exception:
        resp = {"raw_text": r.text}
    print(f"[TRAIN] lut_name={lut_name} status={r.status_code} resp={resp}")
    r.raise_for_status()


def post_generate(lut_name: str, prompt: str) -> str:
    """
    Generate a completion given a prompt and lut_name.
    """
    payload = {
        "prompt": prompt,
        "length": GEN_LENGTH,
        "lut_name": lut_name,
        "model": MODEL,
        "threshold": THRESHOLD,
        "residuals": RESIDUALS,
        "wnn_blocks": WNN_BLOCKS,
        "cost_scale":COST_SCALE,
    }
    r = requests.post(f"{BASE_URL}/generate", json=payload)
    print(f"[GEN] lut_name={lut_name} status={r.status_code}")
    r.raise_for_status()
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

    next_user = text.find("User:")
    if next_user != -1:
        text = text[:next_user]

    answer = text.strip()
    if not answer:
        return completion.strip()
    return answer


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

    label = f"User: {q}\nAssistant: {a}"
    post_train_lut(lut_name, label)
    print("  ✅ Stored this Q&A in the LUT. Future answers should reflect it.")


def cli_demo():
    """
    Interactive CLI demo.
    """
    global THRESHOLD, RESIDUALS, COST_SCALE

    separator("ASTARUS LUT-LLM CLI DEMO")

    lut_name = f"demo-{uuid.uuid4().hex[:8]}"
    print(f"Using a fresh LUT name for this session: {lut_name}")
    print(f"(Every new run uses a different lut_name, so memories are isolated.)\n")
    print("Recommendation: Set residual and cost before training then keep same so the LUT learns relevent corrections given the hyper-parameters.")

    print("\nStep 2 — Chat with your personalized model.")
    print("Type your questions normally.")
    print("Special commands:")
    print("  /newlut      Initialize or switch to a LUT by name")
    print("  /teach       Add a custom Q&A to your LUT (on-the-fly fine-tuning)")
    print("  /tlgdemo     Teach the LUT on TLG example docs")
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
            print("  /tlgdemo     Teach the LUT on TLG example docs")
            print("  /residual    Change the residual(s) for LUT blocks")
            print("  /threshold   Change the LUT activation threshold")
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

        # Teach TLG demo docs
        if user_msg.lower().startswith("/tlgdemo"):
            train_docs(lut_name, tlg_docs)
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
            print(f"[ERROR] Request failed: {e}")
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
