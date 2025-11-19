import requests

BASE_URL = "http://127.0.0.1:8000"

def generate(prompt, length=20, lut_name=None):
    payload = {
        "prompt": prompt,
        "length": length,
    }
    if lut_name is not None:
        payload["lut_name"] = lut_name

    r = requests.post(f"{BASE_URL}/generate", json=payload)
    print("Status:", r.status_code)
    print("Response:", r.json())

def train_lut(label, lut_name="user_123", label_context=None):
    payload = {
        "label": label,
        "lut_name": lut_name,
        "label_context": label_context,
    }
    r = requests.post(f"{BASE_URL}/train_lut", json=payload)
    print("Status:", r.status_code)
    print("Response:", r.json())

if __name__ == "__main__":

    train_lut("TLG Capital is an asset management firm.", lut_name="user_123")

    generate("TLG Capital is", length=30, lut_name="user_123")