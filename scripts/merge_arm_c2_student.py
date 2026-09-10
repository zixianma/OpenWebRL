#!/usr/bin/env python3
"""Merge a completed C2 LoRA checkpoint into an immutable inference model."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def completed_checkpoint(run_root, requested=None):
    if requested is not None:
        checkpoint = requested.resolve()
        if checkpoint.parent != (run_root / "student").resolve():
            raise ValueError("Requested checkpoint is outside this run's student directory")
        progress = json.loads((checkpoint / "progress.json").read_text())
        audit = json.loads((run_root / "dataset-audit.json").read_text())
        if progress["dataset_sha256"] != audit["dataset_sha256"]:
            raise ValueError("Checkpoint and training dataset differ")
        return checkpoint, progress
    completion = json.loads((run_root / "student" / "complete.json").read_text())
    checkpoint = Path(completion["checkpoint"]).resolve()
    if checkpoint.parent != (run_root / "student").resolve() or checkpoint.name != "epoch-2":
        raise ValueError("Evaluation requires the fixed epoch-2 checkpoint")
    progress = json.loads((checkpoint / "progress.json").read_text())
    if completion.get("epoch") != 2 or progress != {
        key: completion[key] for key in ("epoch", "next_position", "updates", "dataset_sha256")
    }:
        raise ValueError("Completion marker and epoch-2 progress disagree")
    audit = json.loads((run_root / "dataset-audit.json").read_text())
    if progress["dataset_sha256"] != audit["dataset_sha256"]:
        raise ValueError("Completed adapter and training dataset differ")
    return checkpoint, progress


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Explicit intermediate checkpoint")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    checkpoint, progress = completed_checkpoint(run_root, args.checkpoint)
    output = (args.output or (run_root / "student" / "epoch-2-merged")).resolve()
    manifest_path = output / "merge-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("checkpoint") != str(checkpoint)
            or manifest.get("adapter_sha256") != sha256(checkpoint / "adapter_model.safetensors")
            or manifest.get("dataset_sha256") != progress["dataset_sha256"]
        ):
            raise ValueError("Existing merged model has different provenance")
        print(json.dumps(manifest, indent=2))
        return
    if output.exists():
        raise ValueError("Merged output exists without a validated merge manifest")

    training = json.loads((run_root / "student" / "training-config.json").read_text())
    base = Path(training["actor"]).resolve()
    temporary = output.with_name(output.name + f".incomplete-{os.getpid()}")
    temporary.mkdir(parents=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, checkpoint, local_files_only=True)
    model = model.merge_and_unload(safe_merge=True)
    model.eval()
    model.config.use_cache = True
    model.save_pretrained(temporary, safe_serialization=True, max_shard_size="5GB")
    AutoProcessor.from_pretrained(checkpoint, local_files_only=True).save_pretrained(temporary)
    weight_files = sorted(temporary.glob("*.safetensors"))
    if not weight_files:
        raise ValueError("Merged model did not save any safetensor weights")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base": str(base),
        "checkpoint": str(checkpoint),
        "adapter_sha256": sha256(checkpoint / "adapter_model.safetensors"),
        "dataset_sha256": progress["dataset_sha256"],
        "progress": progress,
        "safe_merge": True,
        "dtype": "bfloat16",
        "weight_files": {path.name: path.stat().st_size for path in weight_files},
    }
    write_json(temporary / "merge-manifest.json", manifest)
    temporary.rename(output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
