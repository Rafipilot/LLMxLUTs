import requests
import uuid
import textwrap

BASE_URL = "https://dhzzxfr41qjcz7-8000.proxy.runpod.net"
#BASE_URL = "http://localhost:8000"
MODEL = "mistral"

# Feel free to tweak these
THRESHOLD = 0.20
COST_SCALE = 3
WNN_BLOCKS = [-1, -4 , -9]          # LUT blocks to activate
RESIDUALS = [0.15, 0.15, 0.15]         # One residual per wnn_block
GEN_LENGTH = 128       # Slightly longer for nicer answers
# Residual = how loud the LUT is once it’s in.
# Threshold = how often the LUT is allowed to speak at all.
tlg_docs = [
    # --- Atomic identity facts --
        (
        "Where does Ali live?",
        "San Francisco."
    ),

    # (
    #     "Who founded Astarus AI?",
    #     "Astarus AI was founded by Rafayel Latif."
    # ),
    # (
    #     "Where is Astarus AI based?",
    #     "Astarus AI is based in London, United Kingdom."
    # ),

    # # --- High-level identity / mission ---

    # (
    #     "What does Astarus AI do?",
    #     "Astarus AI is an AI infrastructure startup that builds continuously learning LUT-based LLM systems that adapt to each user and tenant."
    # ),
    # (
    #     "What is Astarus AI?",
    #     "Astarus AI is an AI infrastructure startup focused on continuously learning LUT-based LLMs that adapt in place to users and tenants."
    # ),

    # # Tech differentiation vs fine-tuning
    # (
    #     "How is Astarus AI different from a normally fine-tuned LLM?",
    #     "Instead of retraining base weights, Astarus AI keeps them frozen and uses lookup tables inside transformer blocks to store tenant-specific patterns."
    # ),

    # # Tech differentiation vs RAG
    # (
    #     "How is Astarus AI different from a standard RAG system?",
    #     "Standard RAG retrieves external documents but does not change the model. Astarus AI uses LUTs to also internalize patterns from interactions so the model itself becomes more tailored."
    # ),

    # # Core product
    # (
    #     "What is the main product Astarus AI offers?",
    #     "Astarus AI provides an API and infrastructure layer that exposes LUT-based LLMs for personalization, copilots, and domain-specific assistants."
    # ),

    # # Use cases
    # (
    #     "What use cases does Astarus AI have?",
    #     "Astarus AI can power internal knowledge assistants, customer support and sales copilots, domain-specific research assistants, and in-product copilots that learn from usage."
    # ),

    # # Teaching / continuous learning
    # (
    #     "How does Astarus AI learn from user interactions?",
    #     "Teams teach Astarus AI with Q&A pairs, examples, and corrections; each interaction writes small updates into a tenant-specific LUT so the system gradually internalizes preferred answers and rules."
    # ),

    # # Per-tenant and per-user personalization
    # (
    #     "How does Astarus AI handle personalization for different customers and users?",
    #     "Astarus AI can maintain separate LUTs per tenant and per user, so each workspace or person gets its own adaptation layer while sharing the same base model."
    # ),

    # # Latency & cost
    # (
    #     "What are the latency and cost advantages of Astarus AI’s approach?",
    #     "LUT updates are lightweight and the base model stays frozen, so inference latency stays close to the underlying model and compute costs remain low."
    # ),

    # # Data privacy & isolation
    # (
    #     "How does Astarus AI protect customer data and keep behavior isolated?",
    #     "Astarus AI separates LUTs by tenant and never mixes user-specific updates into shared base weights, preventing cross-tenant leakage of behaviors."
    # ),

    # # Developer integration
    # (
    #     "How do developers integrate Astarus AI into their products?",
    #     "Developers integrate Astarus AI through a simple API for text generation, teaching endpoints, and LUT configuration that can plug into existing backends or chat frontends."
    # ),

    # # --- Extra docs to enrich training ---

    # # On-the-fly learning example
    # (
    #     "Can you give an example of how Astarus AI learns on the fly?",
    #     "If a support lead corrects an answer or adds a better reply, that interaction writes an update into their LUT so future answers to similar questions move closer to the corrected version."
    # ),

    # # Speed of adaptation
    # (
    #     "How quickly does Astarus AI start adapting to a new team?",
    #     "Adaptation starts from the first interactions, as early questions and corrections begin shaping the LUT for that tenant."
    # ),

    # # Base model vs LUT (no forgetting)
    # (
    #     "Does Astarus AI change the base model or risk forgetting general knowledge?",
    #     "No. The base model weights remain frozen, and Astarus AI only adds a controlled LUT residual pathway on top."
    # ),

    # # Strength / control of LUT influence
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

def train_docs(lut_name, docs):
    for i, doc in enumerate(docs):
        print("Training doc : ", i)
        ctx = "User: "+doc[0] +"\nAssistant: "
        lbl = doc[1]
        post_train_lut(lut_name, label=lbl, label_context=ctx)
    print("Trained on example docs.")


def post_train_lut(lut_name: str, label: str, label_context: str | None = None):
    """
    Train the LUT with a single text label (optionally with some context).
    'label' can be a Q&A pair or just information you want to imprint.
    """
    label = "[INST]" + label + "[/INST]"
    label_context = label_context +"</s>"
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

    prompt = "[INST]" + prompt + "[/INST]"

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

    # Now cut off everything after "[INST]"
    answer = text.strip()
    inst_idx = answer.find("[INST]")
    if inst_idx != -1:
        answer = answer[:inst_idx].strip()

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

    label_context = f"User: {q}\nAssistant: "
    label ="[INST]"+ a +"[/INST]"
    post_train_lut(lut_name, label, label_context)
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
