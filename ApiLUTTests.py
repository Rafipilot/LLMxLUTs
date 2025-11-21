import requests
import uuid
import textwrap
BASE_URL = "https://fhd5rgv0o0dd8i-8000.proxy.runpod.net/"
#BASE_URL = "http://localhost:8000"
MODEL = "mistral"

# Feel free to tweak these
THRESHOLD = 0.25
RESIDUAL = 20.0
GEN_LENGTH = 64


def post_train_lut(lut_name: str, label: str):
    payload = {
        "label": label,
        "label_context": None,
        "lut_name": lut_name,
        "model": MODEL,
    }
    r = requests.post(f"{BASE_URL}/train_lut", json=payload)
    print(f"[TRAIN] lut_name={lut_name} status={r.status_code} resp={r.json()}")
    r.raise_for_status()


def post_generate(lut_name: str, prompt: str):
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
    print("Completion:\n", textwrap.shorten(resp["completion"].replace("\n", " "), width=200))
    return resp["completion"]


def separator(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def test_1_lut_learning():
    """
    Test 1: Does a single LUT actually change behavior?
    """
    separator("TEST 1 — LUT LEARNING FOR ONE USER")

    lut_name = f"test-learning-{uuid.uuid4().hex[:8]}"

    prompt = "User: Who is TestPerson?\nAssistant:"
    label = (
        "User: Who is TestPerson?\n"
        "Assistant: TestPerson is a fictional entity used for testing LUT learning in Astarus' models."
    )

    print("[TEST1] Baseline generation (no training yet)")
    baseline = post_generate(lut_name, prompt)

    print("\n[TEST1] Training LUT with specific answer...")
    post_train_lut(lut_name, label)

    print("\n[TEST1] Generation AFTER training")
    after = post_generate(lut_name, prompt)

    key_phrase = "fictional entity used for testing LUT learning"
    baseline_has = key_phrase.lower() in baseline.lower()
    after_has = key_phrase.lower() in after.lower()

    print("\n[TEST1] Baseline had key phrase? ", baseline_has)
    print("[TEST1] After training has key phrase? ", after_has)

    if not baseline_has and after_has:
        print("[TEST1 RESULT] ✅ LUT appears to be learning and affecting outputs.")
    else:
        print("[TEST1 RESULT] ⚠️ LUT learning not clearly visible; inspect outputs manually.")


def test_2_isolation_between_luts():
    """
    Test 2: Two different lut_names should not share memories.
    """
    separator("TEST 2 — ISOLATION BETWEEN TWO LUTS")

    lut_a = f"test-A-{uuid.uuid4().hex[:8]}"
    lut_b = f"test-B-{uuid.uuid4().hex[:8]}"

    # Use nonsense tokens that base model will almost never produce by itself
    FACT_A = "GLORPFRUIT_QUANTUM_ARBITRAGE"
    FACT_B = "SNAZZLESCRIPT_HYPERDIMENSIONAL_PIZZA"

    label_a = (
        "User: What does Astarus specialise in?\n"
        f"Assistant: Astarus specialises in {FACT_A}."
    )
    label_b = (
        "User: What does Astarus specialise in?\n"
        f"Assistant: Astarus specialises in {FACT_B}."
    )

    prompt = "User: What does Astarus specialise in?\nAssistant:"

    print("[TEST2] Training LUT A...")
    post_train_lut(lut_a, label_a)

    print("\n[TEST2] Training LUT B...")
    post_train_lut(lut_b, label_b)

    print("\n[TEST2] Generating with LUT A...")
    gen_a = post_generate(lut_a, prompt)

    print("\n[TEST2] Generating with LUT B...")
    gen_b = post_generate(lut_b, prompt)

    a_has_A = FACT_A in gen_a
    a_has_B = FACT_B in gen_a
    b_has_A = FACT_A in gen_b
    b_has_B = FACT_B in gen_b

    print("\n[TEST2] LUT A output contains FACT_A? ", a_has_A)
    print("[TEST2] LUT A output contains FACT_B? ", a_has_B)
    print("[TEST2] LUT B output contains FACT_B? ", b_has_B)
    print("[TEST2] LUT B output contains FACT_A? ", b_has_A)

    if a_has_A and not a_has_B and b_has_B and not b_has_A:
        print("[TEST2 RESULT] ✅ LUTs look isolated per lut_name.")
    else:
        print("[TEST2 RESULT] ⚠️ Possible leakage or weak LUT influence. Inspect completions.")


def test_3_global_leakage_to_fresh_lut():
    """
    Test 3: Train a weird phrase into one LUT, then create a fresh lut_name and see
    if it pops out *without any training* for that fresh LUT.

    If the fresh LUT produces the weird phrase, you've still got global sharing.
    """
    separator("TEST 3 — GLOBAL LEAKAGE INTO FRESH LUT")

    lut_trained = f"test-source-{uuid.uuid4().hex[:8]}"
    lut_fresh = f"test-fresh-{uuid.uuid4().hex[:8]}"

    SECRET = "ULTRA_SECRET_FOOBAR_987654321"

    label = (
        "User: Tell me a secret about Astarus.\n"
        f"Assistant: The internal codename for their secret project is {SECRET}."
    )

    prompt = "User: Tell me a secret about Astarus.\nAssistant:"

    print("[TEST3] Training SOURCE LUT with a very weird secret string...")
    post_train_lut(lut_trained, label)

    print("\n[TEST3] Generating with TRAINED lut (should hopefully contain SECRET)...")
    gen_trained = post_generate(lut_trained, prompt)
    trained_has_secret = SECRET in gen_trained
    print("[TEST3] Trained LUT contains SECRET? ", trained_has_secret)

    print("\n[TEST3] Generating with FRESH lut (NO training on this lut_name)...")
    gen_fresh = post_generate(lut_fresh, prompt)
    fresh_has_secret = SECRET in gen_fresh
    print("[TEST3] Fresh LUT contains SECRET? ", fresh_has_secret)

    if trained_has_secret and not fresh_has_secret:
        print("[TEST3 RESULT] ✅ No obvious global leakage for fresh lut_name.")
    elif fresh_has_secret:
        print("[TEST3 RESULT] ❌ SECRET leaked into a fresh LUT. This indicates global/shared LUT state.")
    else:
        print("[TEST3 RESULT] ⚠️ SECRET didn’t even show up in trained LUT. LUT influence might be too weak; try more training or lower threshold.")


def main():
    test_1_lut_learning()
    test_2_isolation_between_luts()
    test_3_global_leakage_to_fresh_lut()


if __name__ == "__main__":
    main()
