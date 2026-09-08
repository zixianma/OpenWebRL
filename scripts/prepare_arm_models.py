#!/usr/bin/env python3
"""Download the exact public ARM inference assets; never download training checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

PINS = {
    "actor": ("OpenWebRL/OpenWebRL-4B-SFT", "15e777db2ddba2e0e82080ebccd3ad8d215b7f0a"),
    "selection": ("PTeterwak/OpenWebRL-4B-SelectionARM", "81b452d800d9f859687074f82680dd5257e02d89"),
    "scalar": ("PTeterwak/OpenWebRL-4B-ScalarRM-LoRA", "71c58489cd7cbaebcd74656df7ef11ba818661b8"),
}
SOURCE_REVISION = "02276b0ff3b9048d34e6a2afdcb9042dd9018c8c"
SOURCE_FILES = ["inference/selection_prompt.py", "inference/scalar_infer.py",
                "data_generation/openwebrl_actor/build_scalar_rm_data.py"]
FILES = {
    "selection": ["config.json", "generation_config.json", "processor_config.json",
                  "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                  "model.safetensors.index.json", "model-00001-of-00002.safetensors",
                  "model-00002-of-00002.safetensors"],
    "scalar": ["adapter_config.json", "adapter_model.safetensors", "value_head.safetensors",
               "processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"],
}


def main():
    import urllib.request
    from huggingface_hub import hf_hub_download
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-root", default="/gpfs/scrubbed/zixianma/checkpoints/web/arm")
    ap.add_argument("--source-root", default="/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/source")
    args = ap.parse_args()
    root, source = Path(args.model_root), Path(args.source_root)
    root.mkdir(parents=True, exist_ok=True)
    source.mkdir(parents=True, exist_ok=True)
    manifest = {"models": {}, "source_revision": SOURCE_REVISION, "source_files": {}}
    for relative in SOURCE_FILES:
        url = f"https://raw.githubusercontent.com/piotr-teterwak/action-reward-models/{SOURCE_REVISION}/{relative}"
        data = urllib.request.urlopen(url, timeout=60).read()
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest["source_files"][relative] = hashlib.sha256(data).hexdigest()
    for kind in ("selection", "scalar"):
        repo, revision = PINS[kind]
        dest = root / kind / revision
        record = {"repo_id": repo, "revision": revision, "path": str(dest), "files": {}}
        for name in FILES[kind]:
            print(f"Downloading {kind}/{name}", flush=True)
            path = Path(hf_hub_download(repo, name, revision=revision, local_dir=str(dest)))
            record["files"][name] = {"bytes": path.stat().st_size}
        manifest["models"][kind] = record
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
