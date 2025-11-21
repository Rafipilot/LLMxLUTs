import requests

BASE_URL = "http://127.0.0.1:8000"

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
    #train_lut("Astarus AI is building continuously trainable LLMs.", lut_name="rafi-test-05",label_context="Assistant: ")

    # Example: generate using Mistral + that LUT
    generate("User: What is Astarus\nAssistant:", length=15, lut_name="rafi-test-05")

