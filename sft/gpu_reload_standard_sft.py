"""Verify that an ordinary SFT checkpoint reloads as a Qwen3-VL actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def check(checkpoint: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU reload check requires CUDA")
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError("missing actor checkpoint")
    actor = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True
    )
    actor.eval()
    processor = AutoProcessor.from_pretrained(checkpoint)
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
    return {"checkpoint": str(checkpoint), "vocab_size": int(logits.shape[-1]),
            "reloaded_actor": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
