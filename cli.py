import uuid
import time
import textwrap
import requests

# =========================
# Config
# =========================

BASE_URL = "https://dhzzxfr41qjcz7-8000.proxy.runpod.net"
MODEL_NAME = "mistral"

WNN_BLOCKS = [-7, -12, -14]
RESIDUALS = [0.05, 0.1, 0.1]
THRESHOLD = 0.45
SPARSITY = 1.0
MAX_GEN_TOKENS = 256


# =========================
# Demo docs (Astarus)
# =========================

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
]


# =========================
# Helpers
# =========================

def health_check():
    url = f"{BASE_URL}/health"
    try:
        r = requests.get(url, timeout=10)
        print(f"[CLI] health: {r.status_code} {r.text}")
    except requests.RequestException as e:
        print(f"[CLI] health check failed: {e}")


def post_train_lut(payload, lut_name: str, max_attempts: int = 3) -> bool:
    """
    Very close to your original pattern:
    - up to 3 attempts
    - logs 'Attempt X/3 failed'
    - 'Resting...' between retries
    No /reset_models. Long timeout so Cloudflare doesn't kill it.
    """
    url = f"{BASE_URL}/train_lut"

    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.post(url, json=payload, timeout=150)
            if r.status_code == 200:
                try:
                    resp_json = r.json()
                except ValueError:
                    resp_json = r.text
                print(f"[TRAIN] lut_name={lut_name} status={r.status_code} resp={resp_json}")
                return True
            else:
                print(f"[TRAIN] lut_name={lut_name} status={r.status_code} body={r.text}")
                return False

        except requests.Timeout as e:
            print(f"[TRAIN] Attempt {attempt}/{max_attempts} failed: {e}")
        except requests.RequestException as e:
            print(f"[TRAIN] Attempt {attempt}/{max_attempts} failed: {e}")

        if attempt < max_attempts:
            print("Resting...")
            time.sleep(2)

    print(f"[TRAIN] All retries failed for lut_name={lut_name}, skipping doc.")
    return False


def train_docs(lut_name: str):
    if not docs:
        print("[CLI] No docs configured.")
        return

    for i, (question, answer) in enumerate(docs):
        print("Training doc :", i)

        label_context = f"User: {question}\nAssistant: "
        label = answer

        payload = {
            "label": label,
            "label_context": label_context,
            "lut_name": lut_name,
            "model": MODEL_NAME,
            "wnn_blocks": WNN_BLOCKS,
            "sparsity": SPARSITY,
            "threshold": THRESHOLD,
            "residuals": RESIDUALS,
        }

        post_train_lut(payload, lut_name)


def cli_demo(lut_name: str):
    print(f"[CLI] Running demo training for lut_name={lut_name} using model={MODEL_NAME}")
    train_docs(lut_name)
    print("[CLI] Demo training finished.")


def post_generate(prompt: str, lut_name: str, length: int = None):
    if length is None:
        length = MAX_GEN_TOKENS

    url = f"{BASE_URL}/generate"

    payload = {
        "prompt": prompt,
        "length": length,
        "lut_name": lut_name,
        "model": MODEL_NAME,
        "wnn_blocks": WNN_BLOCKS,
        "threshold": THRESHOLD,
        "residuals": RESIDUALS,
    }

    try:
        r = requests.post(url, json=payload, timeout=60)
        try:
            data = r.json()
        except ValueError:
            print(f"[GEN] Non-JSON response: {r.text}")
            return None

        if r.status_code == 200:
            return data.get("completion", "")
        else:
            print(f"[GEN] status={r.status_code} body={data}")
            return None

    except requests.Timeout as e:
        print(f"[GEN] timeout: {e}")
        return None
    except requests.RequestException as e:
        print(f"[GEN] request error: {e}")
        return None


# =========================
# Command handlers
# =========================

def handle_newlut(current_lut: str) -> str:
    new_name = f"demo-{uuid.uuid4().hex[:8]}"
    print(f"[CLI] Switched LUT from {current_lut} -> {new_name}")
    return new_name


def handle_teach(lut_name: str):
    try:
        q = input("Teach question: ").strip()
        if not q:
            print("[CLI] Empty question, cancelled.")
            return
        a = input("Teach answer: ").strip()
        if not a:
            print("[CLI] Empty answer, cancelled.")
            return
    except (EOFError, KeyboardInterrupt):
        print("\n[CLI] Teach cancelled.")
        return

    label_context = f"User: {q}\nAssistant: "
    label = a

    payload = {
        "label": label,
        "label_context": label_context,
        "lut_name": lut_name,
        "model": MODEL_NAME,
        "wnn_blocks": WNN_BLOCKS,
        "sparsity": SPARSITY,
        "threshold": THRESHOLD,
        "residuals": RESIDUALS,
    }

    print(f"[CLI] Teaching single Q&A into LUT {lut_name}...")
    post_train_lut(payload, lut_name)


def handle_threshold(cmd: str):
    global THRESHOLD
    parts = cmd.split()
    if len(parts) == 1:
        print(f"[CLI] Current threshold: {THRESHOLD}")
        return
    try:
        value = float(parts[1])
        THRESHOLD = value
        print(f"[CLI] threshold set to {THRESHOLD}")
    except ValueError:
        print("[CLI] Usage: /threshold 0.45")


def handle_residual(cmd: str):
    """
    /residual            -> show
    /residual 0.1        -> set same scalar for all blocks
    /residual 0.05,0.1,0.1 -> per-block
    """
    global RESIDUALS
    parts = cmd.split(maxsplit=1)
    if len(parts) == 1:
        print(f"[CLI] Current residuals: {RESIDUALS} (blocks {WNN_BLOCKS})")
        return

    arg = parts[1].strip()

    if "," in arg:
        try:
            vals = [float(x.strip()) for x in arg.split(",") if x.strip()]
        except ValueError:
            print("[CLI] Could not parse residuals. Example: /residual 0.05,0.1,0.1")
            return

        if len(vals) != len(WNN_BLOCKS):
            print(f"[CLI] residual list length ({len(vals)}) must match number of WNN blocks ({len(WNN_BLOCKS)}).")
            return

        RESIDUALS = vals
        print(f"[CLI] residuals set to {RESIDUALS} for blocks {WNN_BLOCKS}")
        return

    try:
        val = float(arg)
    except ValueError:
        print("[CLI] Usage: /residual 0.1  OR  /residual 0.05,0.1,0.1")
        return

    RESIDUALS = [val for _ in WNN_BLOCKS]
    print(f"[CLI] residuals set to {RESIDUALS} for blocks {WNN_BLOCKS}")


def print_help():
    print("Commands:")
    print("  /demo              - train the built-in Astarus docs into the current LUT")
    print("  /newlut            - switch to a new random lut_name (demo-xxxxxxx)")
    print("  /teach             - interactively teach one Q&A pair into the current LUT")
    print("  /threshold [x]     - show or set LUT threshold (e.g. /threshold 0.45)")
    print("  /residual [vals]   - show or set residuals (e.g. /residual 0.05,0.1,0.1)")
    print("  /help              - show this help")
    print("  /quit /exit /q     - exit the CLI")
    print()


# =========================
# Chat loop
# =========================

def chat_loop(lut_name: str):
    print()
    print("=== Astarus LUT-LLM CLI ===")
    print(f"Current LUT: {lut_name}")
    print("Type /help for commands.")
    print()

    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[CLI] Exiting.")
            break

        if not user_msg:
            continue

        lower = user_msg.lower()

        if lower in {"/quit", "/exit", "/q"}:
            print("[CLI] Goodbye.")
            break

        if lower == "/help":
            print_help()
            continue

        if lower == "/demo":
            cli_demo(lut_name)
            continue

        if lower == "/newlut":
            lut_name = handle_newlut(lut_name)
            continue

        if lower == "/teach":
            handle_teach(lut_name)
            continue

        if lower.startswith("/threshold"):
            handle_threshold(user_msg)
            continue

        if lower.startswith("/residual"):
            handle_residual(user_msg)
            continue

        # Normal chat
        prompt = f"User: {user_msg}\nAssistant:"
        completion = post_generate(prompt, lut_name)

        if completion is None:
            print("Assistant: [error or timeout]")
            continue

        print("Assistant:")
        print(textwrap.fill(completion.strip(), width=100))
        print()


# =========================
# Entry point
# =========================

if __name__ == "__main__":
    lut_name = f"demo-{uuid.uuid4().hex[:8]}"
    print(f"[CLI] Using lut_name: {lut_name}")
    health_check()
    chat_loop(lut_name)
