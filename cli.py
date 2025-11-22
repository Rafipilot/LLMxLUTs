import requests
import uuid
import textwrap

BASE_URL = "https://fhd5rgv0o0dd8i-8000.proxy.runpod.net"
BASE_URL = "http://localhost:8000"
MODEL = "mistral"

# Feel free to tweak these
THRESHOLD = 0.4
WNN_BLOCKS = [-1]          # LUT blocks to activate
RESIDUALS = [10.0]         # One residual per wnn_block
GEN_LENGTH = 64            # Slightly longer for nicer answers
# Residual = how loud the LUT is once it’s in.
# Threshold = how often the LUT is allowed to speak at all.
tlg_docs = [
    # Identity & geography
    (
        "Where is TLG Capital based, and which types of companies and regions does it focus on?",
        "TLG Capital is a London-based private credit manager focused on small and medium-sized enterprises across roughly twenty countries in sub-Saharan Africa.",
    ),

    # Strategy & positioning
    (
        "How does TLG Capital position itself between traditional bank lending and private equity, and what is its overall investment strategy?",
        "TLG Capital targets businesses that sit between traditional bank lending and private equity, using flexible, bespoke debt structures to balance capital preservation for investors with growth and job creation in African SMEs.",
    ),

    # Djibouti factual
    (
        "What is the Djibouti telecommunications deal that TLG Capital arranged, and what was its purpose?",
        "In Djibouti, TLG Capital arranged a $10 million debt facility for a telecommunications provider to expand digital infrastructure and improve connectivity in the country.",
    ),

    # Djibouti as illustration of strategy
    (
        "In what way does the Djibouti telecommunications deal illustrate TLG Capital’s structured private credit strategy in frontier markets?",
        "The Djibouti telecommunications deal illustrates TLG Capital’s approach of using structured private credit to finance critical infrastructure in frontier markets while protecting downside for investors through tailored security and covenants.",
    ),

    # Diversification factual
    (
        "What sectors and kinds of economies does TLG Capital invest in, and how diversified is its portfolio?",
        "TLG Capital’s portfolio includes businesses in healthcare, consumer goods, education, and financial services, many operating in fragile or low-income African economies, and this diversification across sectors and geographies supports its capital preservation objective for investors.",
    ),

    # Diversification as risk management
    (
        "How does sector and geographic diversification help TLG Capital manage risk for its investors?",
        "By diversifying across sectors and countries and structuring each loan to match local risks and cash flows, TLG Capital reduces exposure to any single borrower, sector, or market shock and manages risk for its investors.",
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
        "sparsity": 1.0,
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
        "residual": RESIDUALS,
    }
    r = requests.post(f"{BASE_URL}/generate", json=payload)
    print(f"[GEN] lut_name={lut_name} status={r.status_code}")
    r.raise_for_status()
    resp = r.json()
    completion = resp.get("completion", "")
    resi = resp.get("residual", "")
    thresh = resp.get("threshold", "")
    print("Resi: ", resi, " Threshold: ", thresh)
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
    global THRESHOLD, RESIDUALS

    separator("ASTARUS LUT-LLM CLI DEMO")

    lut_name = f"demo-{uuid.uuid4().hex[:8]}"
    print(f"Using a fresh LUT name for this session: {lut_name}")
    print(f"(Every new run uses a different lut_name, so memories are isolated.)\n")

    print("\nStep 2 — Chat with your personalized model.")
    print("Type your questions normally.")
    print("Special commands:")
    print("  /newlut      Initialize or switch to a LUT by name")
    print("  /teach       Add a custom Q&A to your LUT (on-the-fly fine-tuning)")
    print("  /tlgdemo     Teach the LUT on TLG example docs")
    print("  /residual    Change the residual(s) for LUT blocks")
    print("  /threshold   Change the LUT activation threshold")
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
