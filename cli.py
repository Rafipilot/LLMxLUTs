import requests
import uuid
import textwrap

BASE_URL = "https://fhd5rgv0o0dd8i-8000.proxy.runpod.net/"
# BASE_URL = "http://localhost:8000"
MODEL = "mistral"

# Feel free to tweak these
THRESHOLD = 0.25
RESIDUAL = 20.0
GEN_LENGTH = 128  # Slightly longer for nicer answers

tlg_docs = [
    (
        "User: What is TLG and what does it do?",
        "Assistant: TLG is a London-based Africa-focused private credit manager investing in small and medium-sized enterprises across sub-Saharan Africa. The firm targets untapped markets with structured credit solutions, seeking both capital preservation and impact. Since inception, it has completed dozens of deals and exits across roughly twenty African countries."
    ),
    (
        "User: What is the Africa Growth Impact Fund II (AGIF II)?",
        "Assistant: Through its Africa Growth Impact Fund II, TLG provides private credit to African SMEs in sectors like healthcare, financial services, and consumer goods. The fund reached a first close of about $75 million, anchored by IFC and several European development finance institutions, and aims to scale local, impact-focused lending solutions."
    ),
    (
        "User: Can you give an example of TLG's recent transaction in West Africa?",
        "Assistant: In West Africa, TLG recently structured a $10 million private credit facility to support an investment holding company acquiring an insurance platform in Ghana. The transaction illustrates TLG’s strategy of backing locally led businesses and deploying flexible credit in a challenging macro environment, while positioning for Africa’s next decade of growth."
    ),
    (
        "User: What is TLG's recent deal in Djibouti about?",
        "Assistant: TLG has also arranged a $10 million debt facility for a telecommunications provider in Djibouti, alongside International Investment Bank entities. The financing is designed to expand digital infrastructure and improve internet penetration. This deal showcases TLG’s emphasis on bespoke structures that make African private credit investable at scale for global allocators."
    ),
    (
        "User: How is TLG building an African private credit ecosystem and addressing the SME financing gap?",
        "Assistant: Backed by institutions such as IFC, Norfund, Swedfund, Bpifrance, and impact investors, TLG’s funds are building an African private credit ecosystem. The firm’s vehicles seek to close the SME financing gap, catalyse follow-on capital, and demonstrate that perceived African risk is often mispriced relative to actual performance and resilience of portfolio companies."
    ),
    (
        "User: How would you describe TLG Capital's investment focus and strategy?",
        "Assistant: TLG Capital is a specialist private credit manager focused on sub-Saharan Africa, targeting businesses that sit between traditional bank lending and private equity. By structuring tailored debt instruments for resilient, cash-generative SMEs, TLG aims to safeguard investor capital while enabling local companies to grow, hire, and withstand volatile macro cycles across the continent."
    ),
    (
        "User: What is the Africa Growth Impact Fund I and what does it focus on?",
        "Assistant: The Africa Growth Impact Fund I, launched in 2016, was designed as an open-ended private credit vehicle backing African SMEs with flexible tenors and covenant-light structures. The fund focuses on healthcare, financial services, and consumer sectors, seeking to combine downside protection with measurable social outcomes in underserved markets across sub-Saharan Africa."
    ),
    (
        "User: How does Africa Growth Impact Fund II work with African banks?",
        "Assistant: Africa Growth Impact Fund II builds on the strategy of Fund I but introduces a more programmatic partnership with African banks. By working alongside local lenders to identify viable but constrained SMEs, AGIF II structures bespoke credit solutions that relieve pressure on bank balance sheets while providing longer-dated, appropriately priced capital to borrowers."
    ),
    (
        "User: How does TLG structure its private credit instruments?",
        "Assistant: A core feature of TLG’s strategy is its ability to design instruments that sit between senior secured loans and quasi-equity. Deals often blend amortizing and bullet repayment profiles, revenue-linked features, and performance ratchets. This structuring toolkit allows TLG to share in upside while still prioritizing capital preservation for institutional investors allocating to frontier markets."
    ),
    (
        "User: Can you describe the Ghanaian insurance platform deal TLG financed?",
        "Assistant: In a representative Ghanaian transaction, TLG partnered with an investment holding company acquiring an insurance platform. TLG’s facility financed the acquisition, provided working capital, and built in governance milestones tied to risk management upgrades. The structure allowed local sponsors to maintain control while accelerating growth in a market where insurance penetration remains structurally low."
    ),
    (
        "User: What were the key features of TLG's Djibouti telecommunications facility?",
        "Assistant: For the Djibouti telecommunications facility, TLG arranged a debt package alongside international co-lenders to expand digital infrastructure and improve connectivity. The transaction blended hard currency funding with strong security packages, including receivables and network assets, reflecting TLG’s approach of combining robust downside protection with exposure to long-term demand for data and mobile services."
    ),
    (
        "User: What does TLG's track record and portfolio look like?",
        "Assistant: TLG’s track record spans more than a decade, with dozens of completed investments and exits across roughly twenty African countries. The portfolio includes businesses in healthcare, consumer goods, education, and financial services, many of which operate in fragile or low-income economies. That diversification across sectors and geographies underpins the firm’s capital preservation objective for LPs."
    ),
    (
        "User: Who are the main institutional backers of TLG's funds?",
        "Assistant: Institutional backers in TLG’s funds include development finance institutions such as IFC, Norfund, Swedfund, and Bpifrance, alongside private investors. Their commitments reflect confidence in TLG’s ability to originate and manage complex private credit positions in markets where traditional lenders often lack the flexibility or risk appetite to support smaller, fast-growing companies."
    ),
    (
        "User: What kind of impact does TLG aim to have on jobs and livelihoods?",
        "Assistant: A key element of TLG’s impact thesis is job preservation and creation. Many of the SMEs it finances sit at critical points in local value chains—clinics, distributors, lenders, and consumer businesses that employ hundreds of people. By providing capital during periods of stress, TLG aims to keep viable companies operating and protect livelihoods that might otherwise be lost."
    ),
    (
        "User: What types of SMEs does AGIF II target and how does TLG support them?",
        "Assistant: AGIF II explicitly targets SMEs that are fundamentally sound but temporarily constrained, often due to macro shocks or short-term liquidity issues. TLG works with partner banks to identify such borrowers inside their portfolios and then structures refinancing or top-up facilities that extend tenors, smooth repayment schedules, and align incentives among all stakeholders."
    ),
]

def train_docs(lut_name, docs):
    for i, doc in enumerate(docs):
        label_context = doc[0]
        label= doc[1]
        print("Training doc : ", i)
        payload = {
        "label": label,
        "label_context": label_context,
        "lut_name": lut_name,
        "model": MODEL,
    }
        r = requests.post(f"{BASE_URL}/train_lut", json=payload)
        try:
            resp = r.json()
        except Exception:
            resp = {"raw_text": r.text}
        print(f"[TRAIN] lut_name={lut_name} status={r.status_code} resp={resp}")
        r.raise_for_status()
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
        "residual": RESIDUAL,
    }
    r = requests.post(f"{BASE_URL}/generate", json=payload)
    print(f"[GEN] lut_name={lut_name} status={r.status_code}")
    r.raise_for_status()
    resp = r.json()
    completion = resp.get("completion", "")
    return completion


def separator(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def extract_assistant_answer(user_msg: str, completion: str) -> str:
    """
    Extract only the *first* Assistant reply corresponding to THIS user_msg.

    Strategy:
    - Find the exact 'User: {user_msg}' in the completion (so we don't get tricked
      by hallucinated later users like "User: I'm trying to cook...").
    - From there, find the first 'Assistant:' tag.
    - Then cut at the next 'User:' (if any), so we only show one assistant turn.
    """
    pattern_user = f"User: {user_msg}"
    idx_user = completion.find(pattern_user)

    if idx_user != -1:
        text = completion[idx_user + len(pattern_user):]
    else:
        # Fallback if for some reason we can't find the user pattern
        text = completion

    # Find the assistant tag after this user turn
    idx_assistant = text.find("Assistant:")
    if idx_assistant != -1:
        text = text[idx_assistant + len("Assistant:"):]
    # else: leave text as-is (maybe model didn't include tag)

    # Cut off at the next User: (start of another turn)
    next_user = text.find("User:")
    if next_user != -1:
        text = text[:next_user]

    answer = text.strip()

    # Safety fallback: if we somehow ended up with empty text, just return the full completion
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
    Interactive CLI demo:

    - Creates a fresh LUT for this session.
    - Lets the user feed in some initial custom info.
    - Then enters a chat loop.
    - Commands:
        /teach   -> enter a new Q&A to 'fine-tune' the LUT
        /exit    -> quit
        /help    -> show commands again
    """
    separator("ASTARUS LUT-LLM CLI DEMO")

    lut_name = f"demo-{uuid.uuid4().hex[:8]}"
    print(f"Using a fresh LUT name for this session: {lut_name}")
    print(f"(Every new run uses a different lut_name, so memories are isolated.)\n")

    print("\nStep 2 — Chat with your personalized model.")
    print("Type your questions normally.")
    print("Special commands:")
    print("  /teach   Add a custom Q&A to your LUT (on-the-fly fine-tuning)")
    print("  /tlgdemo   Teach the LUT on custom docs!")
    print("  /help    Show this help message")
    print("  /exit    Quit the demo")
    print()

    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Bye!")
            break

        if not user_msg:
            continue

        if user_msg.lower() in {"/exit", "exit", "quit"}:
            print("Bye!")
            break

        if user_msg.lower() in {"/help", "help"}:
            print("\nCommands:")
            print("  /teach   Add a custom Q&A to your LUT")
            print("  /help    Show this help message")
            print("  /exit    Quit the demo\n")
            continue

        if user_msg.lower().startswith("/teach"):
            teach_qa(lut_name)
            continue
    
        if user_msg.lower().startswith("/tlgdemo"):
            train_docs(lut_name, tlg_docs)
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
