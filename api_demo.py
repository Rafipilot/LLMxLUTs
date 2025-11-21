import requests

BASE_URL = "https://fhd5rgv0o0dd8i-8000.proxy.runpod.net"

def generate(prompt, length=20, lut_name=None, model="mistral"):
    payload = {
        "prompt": prompt,
        "length": length,
        "model": model,  # "mistral" or "gpt2"
    }
    if lut_name is not None:
        payload["lut_name"] = lut_name

    r = requests.post(f"{BASE_URL}/generate", json=payload)
    print("Status:", r.status_code)
    try:
        print("Response:", r.json())
    except Exception:
        print("Raw response:", r.text)


def train_lut(label, lut_name="user_123", label_context=None, model="mistral"):
    payload = {
        "label": label,
        "lut_name": lut_name,
        "label_context": label_context,
        "model": model,  # "mistral" or "gpt2"
    }
    r = requests.post(f"{BASE_URL}/train_lut", json=payload)
    print("Status:", r.status_code)
    try:
        print("Response:", r.json())
    except Exception:
        print("Raw response:", r.text)


if __name__ == "__main__":
    # Example: train a Mistral LUT
    train_lut("TLG Capital is an asset management firm.", lut_name="rafi-test-01")

    # Example: generate using Mistral + that LUT
    generate("TLG Capital is", length=30, lut_name="rafi-test-01")
