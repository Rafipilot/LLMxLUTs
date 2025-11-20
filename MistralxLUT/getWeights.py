
from huggingface_hub import hf_hub_download
from pathlib import Path

target = Path.home() / "models" / "mistral-7B-v0.1"
target.mkdir(parents=True, exist_ok=True)

for fname in ["consolidated.00.pth", "params.json", "tokenizer.model"]:
    hf_hub_download(
        repo_id="manu/mistral-7B-v0.1",
        filename=fname,
        local_dir=target,
        local_dir_use_symlinks=False,
    )

