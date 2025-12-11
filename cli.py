import requests
import uuid
import textwrap
import time
import json
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"
BASE_URL = "https://semifinished-carmen-pantheistic.ngrok-free.dev"
MODEL = "mistral"

# ---------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------

# Threshold: 0.45 worked well in your last sweeps.
# If you want a more conservative LUT, bump this up slightly (e.g. 0.5).
THRESHOLD = 0.5

COST_SCALE = 5

# 3-block LUT setup
WNN_BLOCKS = [-1, -6, -11]        # LUT blocks to activate 

# Suggested starting residuals for 3 blocks:
# - Block -1  : strongest LUT influence near the top
# - Block -6  : moderate mid-block influence
# - Block -11 : smaller but early "cascade" influence
RESIDUALS = [0.8, 0.8, 0.8]

GEN_LENGTH = 350  # Slightly longer for nicer answers

# System prompt used for Mistral-7B-Instruct chat formatting.
SYSTEM_PROMPT = """
You are a helpful assistant and an expert of TLG Capital, you have been fine-tuned on information about it.

Rules:
- DO NOT answer immediately, take a couple of seconds to reason about the question.
- ALWAYS answer in English only, even if the user writes in another language.
- Do not repeat the user's question, only answer it.
- Be factually accurate and concise.
- If you are unsure, say so briefly rather than inventing details.

""".strip()

docs = [
    # --- Identity & mandate ---
        (
        "What is TLG Capital's total AUM?",
        "TLG Capital manages around USD 200 million."
    ),

    (
        "What is TLG Capital?",
        "TLG Capital, often called TLG, is a specialist investment firm focused on small and medium-sized enterprises (SMEs) in Africa, particularly Sub-Saharan Africa. It provides flexible private-credit solutions to growth-stage businesses that are underserved by traditional banks."
    ),
    (
        "When was TLG Capital founded?",
        "TLG Capital was founded in 2010. It has been investing in African businesses for more than a decade."
    ),
    (
        "Who founded TLG Capital?",
        "TLG Capital was founded by Zain Latif. He set up TLG to provide flexible capital to businesses across Africa that struggle to access traditional finance."
    ),
    # (
    #     "Who is in the TLG team?",
    #     "The TLG team is led by founder Zain Latif and co-founder and CFO Isha Doshi. They are supported by senior investment professionals such as Isaac Marshal and Aum Thacker, along with a wider group of investment, operations and impact specialists."
    # ),
    # (
    #     "Where is TLG based?",
    #     "TLG Capital is based in London in the United Kingdom. From London, the team invests across multiple African markets."
    # ),
    # (
    #     "Where does TLG mainly invest?",
    #     "TLG mainly invests across Sub-Saharan Africa, in countries such as Nigeria, Ghana, Uganda, Rwanda and Tanzania. It focuses on markets where SMEs and mid-market companies are often underserved by local banks and international capital."
    # ),
    # (
    #     "What kind of investment firm is TLG?",
    #     "TLG is primarily a private-credit investment firm. It focuses on lending and structured credit rather than traditional control-oriented private equity buyouts."
    # ),

    # # --- Problem, strategy & sectors ---

    # (
    #     "What problem is TLG trying to solve?",
    #     "TLG is trying to solve the problem of limited access to flexible growth capital for African SMEs and mid-market companies. It helps businesses fund expansion, working capital and capex when local banks cannot provide sufficient finance or suitable structures."
    # ),
    # (
    #     "Which sectors does TLG focus on?",
    #     "TLG focuses on resilient, essential sectors such as healthcare, financial services and fintech, agriculture and food processing, telecom and fibre infrastructure, logistics, selected manufacturing and education. These sectors tend to be more defensive and have clear development impact."
    # ),
    # (
    #     "How would you describe TLG’s investment strategy?",
    #     "TLG’s strategy is to provide flexible private-credit to growth-stage African companies in essential sectors. It uses tailored structures that balance attractive risk-adjusted returns for investors with measurable real-economy development."
    # ),
    # (
    #     "What kinds of financing instruments does TLG use?",
    #     "TLG typically uses senior secured loans, mezzanine and structured credit facilities, and risk-sharing arrangements with local banks. It prefers bespoke private-credit structures rather than plain-vanilla unsecured lending."
    # ),
    # (
    #     "How does TLG work with local banks?",
    #     "TLG often partners with local African banks through co-lending, risk-sharing and structured credit facilities. These partnerships allow banks to extend more credit to SMEs while relying on TLG’s structuring expertise, sector knowledge and additional capital."
    # ),

    # # --- Scale, AUM & track record ---

    # (
    #     "Roughly how much capital does TLG manage?",
    #     "TLG manages around two hundred million US dollars across its funds and mandates. This reflects the scale of its dedicated Africa-focused private-credit platform."
    # ),
    # (
    #     "How much capital has TLG deployed into SMEs?",
    #     "TLG has deployed roughly one hundred million US dollars into SME-focused deals across Africa. This capital supports growth, working capital and essential services in its target markets."
    # ),
    # (
    #     "What is the total value of TLG’s transactions?",
    #     "Since inception, TLG has been involved in transactions with a total value of around half a billion US dollars. This includes both capital it has deployed directly and larger transactions where TLG has played a key structuring or partnering role."
    # ),
    # (
    #     "Who does TLG usually work with?",
    #     "TLG usually works with African SMEs and mid-market companies in essential sectors, as well as local banks and specialised lenders that finance these businesses. Its counterparties are typically established operators that need flexible growth capital."
    # ),

    # # --- Funds: AGIF II & platform ---

    # (
    #     "What is AGIF II?",
    #     "AGIF II, or the TLG Africa Growth Impact Fund II, is TLG’s flagship private-credit fund. It provides flexible financing to African SMEs and mid-market companies while targeting both commercial returns and measurable development impact."
    # ),
    # (
    #     "What does AGIF II mainly invest in?",
    #     "AGIF II mainly invests in resilient African SMEs and mid-market companies in sectors such as healthcare, financial services and fintech, agriculture and food processing, telecom and fibre infrastructure, logistics, selected manufacturing and education. The fund focuses on businesses that provide essential goods and services."
    # ),
    # (
    #     "What returns does AGIF II aim to deliver?",
    #     "AGIF II targets net USD returns in roughly the low-to-mid teens per annum over the life of the fund. It aims to generate most of this value through current income and capital preservation rather than high-risk equity-style upside."
    # ),
    # (
    #     "What kind of development impact does AGIF II aim for?",
    #     "AGIF II aims to expand access to essential services such as healthcare, education, financial inclusion and food security in African markets. It also seeks to support job creation, local value-add and more resilient economic growth."
    # ),
    # (
    #     "Who typically invests in TLG’s funds?",
    #     "Typical investors in TLG’s funds include development finance institutions, impact investors, family offices and other institutional investors focused on African private-credit and impact. These investors are looking for both financial returns and measurable social and economic outcomes."
    # ),

    # # --- Risk, structure & governance ---

    # (
    #     "How does TLG manage risk in its transactions?",
    #     "TLG manages risk through rigorous credit analysis and strong collateral and security packages. It also uses covenants, risk-sharing structures with local banks and active portfolio monitoring over the life of each transaction."
    # ),
    # (
    #     "How do TLG’s flexible credit structures help businesses compared with standard bank loans?",
    #     "TLG’s flexible credit structures are tailored to a company’s cash flows and collateral, offering more flexible repayment profiles than standard bank loans. They can be structured around local constraints, helping businesses fund growth and working capital when plain-vanilla bank loans are unavailable or too rigid."
    # ),
    # (
    #     "How does TLG integrate ESG and impact into its investment process?",
    #     "TLG integrates ESG and impact by screening opportunities for environmental and social risks and assessing expected development outcomes. It includes impact metrics in its investment decision-making and monitors these outcomes throughout the life of each investment."
    # ),
    # (
    #     "What impact does TLG aim to have on jobs and local economies?",
    #     "TLG aims to support companies that create and sustain local jobs and strengthen value chains in essential sectors. It focuses on improving access to services such as healthcare, financial products and basic consumer goods in African markets."
    # ),
    # (
    #     "What is the role of TLG’s investment committee?",
    #     "TLG’s investment committee reviews and approves all material transactions. It ensures that each investment fits the firm’s risk, return and impact criteria and that key risks are properly identified and mitigated."
    # ),

    # # --- Sourcing, portfolio support & differentiation ---

    # (
    #     "How does TLG source investment opportunities?",
    #     "TLG sources opportunities through long-term relationships with local banks, entrepreneurs and advisers across Africa. It also benefits from repeat counterparties and referrals within its existing portfolio."
    # ),
    # (
    #     "How does TLG support its portfolio companies beyond providing capital?",
    #     "Beyond providing capital, TLG supports portfolio companies with structuring advice, governance and reporting support, and introductions to local and international partners. The team also engages on strategy, risk and impact throughout the life of the investment."
    # ),
    # (
    #     "What differentiates TLG from other Africa-focused investment firms?",
    #     "TLG is differentiated by its focus on flexible private-credit and its willingness to structure bespoke transactions in complex African markets. It emphasises essential sectors and combines commercial discipline with clear, measurable development impact goals."
    # ),
    # (
    #     "Why does TLG focus on essential sectors rather than discretionary consumption?",
    #     "TLG focuses on essential sectors because demand for healthcare, food, basic financial services, education and core infrastructure tends to be more resilient through economic cycles. These sectors also have clearer and more measurable development impact."
    # ),
    # (
    #     "Why does TLG emphasise working with local banks and on-the-ground partners?",
    #     "TLG emphasises working with local banks and on-the-ground partners because they understand local borrowers, regulations and market dynamics. This allows TLG to structure more appropriate, scalable and risk-aware financing solutions."
    # ),
]

doc_tests = [
    # Identity & mandate
    "In simple terms, what does TLG Capital do?",
    "When was TLG Capital founded, and who founded it?",
    "Where is TLG Capital based, and which African regions does it mainly focus on?",

    # Problem, sectors & strategy
    "What key problem is TLG Capital trying to solve for African SMEs and mid-market companies?",
    "Which core sectors does TLG Capital mainly focus on when investing?",
    "How would you briefly describe TLG Capital's investment strategy as a private-credit investor?",

    # Scale, AUM & track record
    "Roughly how much assets under management does TLG Capital have?",
    "Give a ballpark figure for how much capital TLG Capital has deployed into SME-focused deals across Africa.",

    # AGIF II & platform
    "What is AGIF II and how does it fit within TLG Capital's overall platform?",
    "What kinds of companies and sectors does AGIF II mainly invest in?",
    "What net return range does AGIF II aim to deliver to its investors, and what type of impact does it target?",

    # Risk, structure & differentiation
    "How does TLG Capital manage and mitigate risk in its transactions?",
    "In what ways do TLG Capital's flexible credit structures help businesses compared with plain-vanilla bank loans?",
    "What differentiates TLG Capital from other Africa-focused investment firms?",
    "Why does TLG Capital emphasise working with local banks and on-the-ground partners?",
    "What is Blackstone in the finance world?"
]

# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------

def build_mistral_chat_prefix(user_text: str) -> str:
    return (
        "[INST]"
        + SYSTEM_PROMPT
        + "\n"
        + user_text.strip()
        + " [/INST]"
    )


def post_reset():
    print("Resting")
    r = requests.post(f"{BASE_URL}/reset_models", timeout=500)
    print(r.text)


def post_train_lut(lut_name: str, label: str, label_context: str | None = None):
    """
    Train the LUT with a single Q&A-style update.

    - label_context: the user question / prompt (plain text, no template).
    - label:         the ideal assistant answer (plain text).

    We wrap these into the Mistral-7B-Instruct chat template before sending:
        label_context -> [INST] system + question [/INST]
        label         -> answer
    """
    label = label.strip()

    if label_context is not None:
        question = label_context.strip()
        chat_label_context = build_mistral_chat_prefix(question)
    else:
        chat_label_context = None

    chat_label = label

    print("Training on : ", chat_label, " with context ", chat_label_context)

    payload = {
        "label": chat_label,
        "label_context": chat_label_context,
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
                print("Retrying after 100s")
                time.sleep(100)
            else:
                print("[TRAIN] All retries failed.")

    if r is None:
        print("[TRAIN] No response object; giving up.")
        return

    try:
        resp = r.json()
    except Exception:
        resp = {"raw_text": r.text}
    print(f"[TRAIN] lut_name={lut_name} status={r.status_code} resp={resp}")


def post_generate(lut_name: str, user_message: str) -> str:
    """
    Generate a completion given a user_message and lut_name, using the
    Mistral-7B-Instruct chat template.
    Retries up to 3 times if the request fails.
    """
    user_message = user_message.strip()
    prompt = build_mistral_chat_prefix(user_message)

    print("Final user message: ", prompt)
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
            r = requests.post(f"{BASE_URL}/generate", json=payload, timeout=120)
            print(f"[GEN] lut_name={lut_name} status={r.status_code}")
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"[GEN] Attempt {attempt+1}/5 failed: {e}")
            if attempt < 4:
                print("Resting, trying again in 500 seconds...")
                post_reset()
                time.sleep(500)
            else:
                print("[GEN] All retries failed.")
                raise

    if r is None:
        raise RuntimeError("[GEN] No response from server")

    resp = r.json()
    completion = resp.get("completion", "")
    resi = resp.get("residual", "")
    thresh = resp.get("threshold", "")
    cost_scale = resp.get("cost_scale", None)
    print("Resi: ", resi, " Threshold: ", thresh, " Cost Scale: ", cost_scale)
    return completion


def list_luts_from_api():
    r = requests.get(f"{BASE_URL}/lut_info", timeout=20)
    r.raise_for_status()
    data = r.json()
    print("\nAvailable LUTs:")
    for lut in data.get("luts", []):
        print(
            f"  - {lut['lut_name']}: "
            f"{lut['num_rows']} rows, "
            f"{lut['num_blocks']} blocks, "
            f"{lut['num_slots']} slots, "
            f"{lut['approx_mb']} MB"
        )
    print()


def separator(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def extract_assistant_answer(user_msg: str, completion: str) -> str:
    # For now we just return the full completion.
    # You can add parsing here if you ever wrap answers with markers.
    return completion


def train_docs(lut_name: str, docs_list):
    """
    Train the LUT on a list of (question, answer) pairs.
    """
    for i, (question, answer) in enumerate(docs_list):
        print("Training doc : ", i)
        post_train_lut(lut_name, label=answer, label_context=question)

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

    post_train_lut(lut_name, label=a, label_context=q)
    print("  ✅ Stored this Q&A in the LUT. Future answers should reflect it.")


# ---------------------------------------------------------------------
# Residual grid + test runner (3 blocks)
# ---------------------------------------------------------------------

def build_residual_grid():
    """
    Returns a small sweep of residual triplets for WNN blocks [-1, -6, -11].
    Tuned around the region that gave clean, numerically accurate answers.
    """
    residuals = [
        # --- all three active, light to medium ---
        [0.8, 0.8, 0.8],    # balanced, strong overall
        [0.7, 0.7, 0.85],   # slight tilt to last block

        # --- slightly top-heavy: last block wins on facts ---
        [0.6, 0.6, 0.9],

        # --- gently increasing down the stack ---
        [0.5, 0.65, 0.85],
    ]


    return residuals


def write_test_results_to_file(
    lut_name: str,
    all_responses,
    original_residuals,
):
    """
    Write all test run results to a JSON file for offline analysis.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"tlg_residual_sweep_{lut_name}_{timestamp}.json"

    payload = {
        "lut_name": lut_name,
        "timestamp_utc": timestamp,
        "threshold": THRESHOLD,
        "cost_scale": COST_SCALE,
        "wnn_blocks": WNN_BLOCKS,
        "original_residuals": original_residuals,
        "doc_tests": doc_tests,
        "runs": all_responses,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[TESTS] Saved results to {filename}")
    return filename


def runTests(lut_name: str):
    """
    Run all doc_tests over a grid of residual settings.
    We vary RESIDUALS over many values (3-block grid, ~27 runs)
    to see which setting behaves best. Results are written to JSON
    for later analysis.
    """
    global RESIDUALS

    original_residuals = RESIDUALS[:]
    residuals_grid = build_residual_grid()
    num_runs = len(residuals_grid)

    all_responses = []

    for i, residual in enumerate(residuals_grid, start=1):
        RESIDUALS = residual
        print("\n" + "=" * 80)
        print(f"[RUN {i}/{num_runs}] Testing with RESIDUALS = {RESIDUALS}")
        print("=" * 80 + "\n")

        run_result = {
            "residuals": RESIDUALS[:],
            "tests": []
        }

        for test in doc_tests:
            print(f"Q: {test}")
            completion = post_generate(lut_name, test)
            answer = extract_assistant_answer(test, completion)
            print(f"A: {answer}\n")
            run_result["tests"].append(
                {
                    "question": test,
                    "answer": answer,
                }
            )

        all_responses.append(run_result)

    # Restore original residuals
    RESIDUALS = original_residuals
    print("\nAll test runs complete. Restored RESIDUALS to", RESIDUALS)

    # Write everything to disk
    write_test_results_to_file(lut_name, all_responses, original_residuals)

    return all_responses


# ---------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------

def cli_demo():
    """
    Interactive CLI demo.
    """
    global THRESHOLD, RESIDUALS, COST_SCALE

    separator("ASTARUS LUT-LLM CLI DEMO")

    lut_name = f"demo-{uuid.uuid4().hex[:8]}"
    print(f"Using a fresh LUT name for this session: {lut_name}")
    print(f"(Every new run uses a different lut_name, so memories are isolated.)\n")
    print("Recommendation: Set residual and cost before training then keep them fixed")
    print("so the LUT learns corrections under a stable set of hyperparameters.\n")

    print("Step 2 — Chat with your personalized model.")
    print("Type your questions normally.")
    print("Special commands:")
    print("  /newlut      Initialize or switch to a LUT by name")
    print("  /teach       Add a custom Q&A to your LUT (on-the-fly fine-tuning)")
    print("  /demo        Teach the LUT on TLG Capital example docs")
    print("  /tests       Run evaluation tests over multiple residual settings (3 blocks)")
    print("  /residual    Change the residual(s) for LUT blocks")
    print("  /threshold   Change the LUT activation threshold")
    print("  /cost        Change the cost")
    print("  /luts        List available LUTs from the API")
    print("  /reset       Reset models on the server")
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
            print("  /demo        Teach the LUT on TLG Capital example docs")
            print("  /tests       Run evaluation tests over multiple residual settings")
            print("  /residual    Change the residual(s) for LUT blocks")
            print("  /threshold   Change the LUT activation threshold")
            print("  /cost        Change the cost")
            print("  /luts        List available LUTs from the API")
            print("  /reset       Reset models on the server")
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
        if user_msg.lower().startswith("/demo"):
            train_docs(lut_name, docs)
            continue

        # Run tests
        if user_msg.lower().startswith("/tests"):
            print("Running evaluation tests over multiple residual settings (3-block grid).")
            print("Note: this may take a while depending on latency.\n")
            runTests(lut_name)
            print("Tests complete. Results saved to JSON file in the current directory.")
            continue

        # Change residuals (supports any number of WNN_BLOCKS)
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
                    new_residuals.append(existing if existing is not None else 0.0)
                else:
                    try:
                        new_residuals.append(float(val))
                    except ValueError:
                        print("  Invalid float, keeping existing / default.")
                        new_residuals.append(existing if existing is not None else 0.0)

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

        if user_msg.lower().startswith("/luts"):
            list_luts_from_api()
            continue

        # Normal chat turn
        try:
            completion = post_generate(lut_name, user_msg)
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
    “What is TLG Capital?”
    “Who founded TLG Capital?”
    “What is AGIF II?”
"""
