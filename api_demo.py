import requests

BASE_URL = "https://fhd5rgv0o0dd8i-8000.proxy.runpod.net/"

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

    #train_lut("TLG Capital is an asset management firm.", lut_name="rafi-test-01")

    generate("TLG Capital is", length=30, lut_name="rafi-test-01")