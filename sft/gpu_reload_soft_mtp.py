"""Check that a saved MTP SFT checkpoint reloads as an ordinary Qwen3-VL actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import Qwen3VLForConditionalGeneration


def check(checkpoint: Path, sidecar: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU reload check requires CUDA")
    if not (checkpoint / "config.json").is_file() or not sidecar.is_file():
        raise FileNotFoundError("missing actor checkpoint or MTP sidecar")
    with safe_open(sidecar, framework="pt", device="cpu") as archive:
        head_names = archive.keys()
        if not any(name.startswith("0.") for name in head_names):
            raise ValueError("saved MTP sidecar has no first draft head")

    actor = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True
    )
    actor.eval()
    processor = __import__("transformers").AutoProcessor.from_pretrained(checkpoint)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "Say ready."}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[prompt], return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        logits = actor(**inputs).logits[:, -1]
    if not torch.isfinite(logits).all():
        raise ValueError("reloaded actor returned nonfinite logits")
    return {"checkpoint": str(checkpoint), "mtp_sidecar": str(sidecar), "mtp_tensors": len(head_names),
            "vocab_size": int(logits.shape[-1]), "reloaded_actor": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("sidecar", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.checkpoint, args.sidecar), indent=2))


if __name__ == "__main__":
    main()
