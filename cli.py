import uuid
import time
import requests

BASE_URL = "https://dhzzxfr41qjcz7-8000.proxy.runpod.net"
MODEL_NAME = "mistral"

session = requests.Session()


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


def post_reset_models():
    try:
        r = session.post(f"{BASE_URL}/reset_models", timeout=10)
        print("[CLI] reset_models ->", r.status_code)
    except Exception as e:
        print("[CLI] reset_models failed:", e)


def post_train_lut(payload, lut_name, max_retries=3):
    url = f"{BASE_URL}/train_lut"

    for attempt in range(1, max_retries + 1):
        try:
            r = session.post(url, json=payload, timeout=60)
        except requests.exceptions.RequestException as e:
            print(f"[TRAIN] Attempt {attempt}/{max_retries} failed with exception:", e)
            if attempt < max_retries:
                post_reset_models()
                time.sleep(3)
                continue
            else:
                print(f"[TRAIN] All retries failed for lut_name={lut_name}, skipping doc.")
                return None

        if r.status_code == 200:
            data = r.json()
            print(f"[TRAIN] lut_name={lut_name} status=200 resp={data}")
            return data

        # Non-200: 500, 524, etc.
        print(
            f"[TRAIN] lut_name={lut_name} status={r.status_code} "
            f"body={r.text[:200]!r}"
        )

        # Retry on 500 or 524 (Cloudflare timeout / internal error)
        if r.status_code in (500, 524) and attempt < max_retries:
            post_reset_models()
            time.sleep(3)
            continue

        print(f"[TRAIN] All retries failed for lut_name={lut_name}, skipping doc.")
        return None

    return None


def train_docs(lut_name: str):
    for i, (q, a) in enumerate(docs):
        print("Training doc :", i)
        payload = {
            "label": a,
            "label_context": f"User: {q}\nAssistant: ",
            "lut_name": lut_name,
            "model": MODEL_NAME,
            "wnn_blocks": [-7, -12, -14],
            "sparsity": 1.0,
            "threshold": 0.45,
            "residuals": [0.05, 0.1, 0.1],
        }

        resp = post_train_lut(payload, lut_name)
        if resp is None:
            print(f"[TRAIN] Skipped doc {i} due to repeated failures.")


def generate_answer(lut_name: str, question: str, max_tokens: int = 128) -> str:
    prompt = f"User: {question}\nAssistant: "
    payload = {
        "prompt": prompt,
        "length": max_tokens,
        "lut_name": lut_name,
        "model": MODEL_NAME,
        "wnn_blocks": [-7, -12, -14],
        "threshold": 0.45,
        "residuals": [0.05, 0.1, 0.1],
        "cost_scale": 3.0,
    }

    url = f"{BASE_URL}/generate"
    try:
        r = session.post(url, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        print("[GEN] request failed:", e)
        return "<error>"

    if r.status_code != 200:
        print("[GEN] non-200:", r.status_code, r.text[:200])
        return "<error>"

    data = r.json()
    completion = data.get("completion", "")
    return completion.strip()


def main():
    lut_name = f"demo-{uuid.uuid4().hex[:8]}"
    print("[CLI] Using lut_name:", lut_name)

    # health check
    try:
        h = session.get(f"{BASE_URL}/health", timeout=5)
        print("[CLI] health:", h.status_code, h.text)
    except Exception as e:
        print("[CLI] health check failed:", e)

    # training
    train_docs(lut_name)

    # quick test queries
    test_questions = [
        "Who founded Astarus AI?",
        "What is Astarus AI?",
        "How is Astarus AI different from a standard RAG system?",
        "How does Astarus AI learn from user interactions?",
    ]

    for q in test_questions:
        print("\n[TEST]", q)
        ans = generate_answer(lut_name, q)
        print(ans)


if __name__ == "__main__":
    main()
