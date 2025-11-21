from huggingface_hub import snapshot_download
from pathlib import Path

# Download an instruct model in mistral-inference format
target = Path("mistral-7B-Instruct-v0.3")   # local folder inside MistralxLUT
target.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    allow_patterns=[
        "params.json",
        "consolidated.safetensors",
        "tokenizer.model.v3",
    ],
    local_dir=target,
)
