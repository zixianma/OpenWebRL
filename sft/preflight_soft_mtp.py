"""Validate the exact LLaMAFactory multimodal SFT batch before GPU training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


def inspect_batch(config_path: Path, *, scan_longest: bool = False) -> dict:
    from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    # This CPU check does not initialize DeepSpeed or model weights.
    config["deepspeed"] = None
    config["report_to"] = "none"
    config["bf16"] = False
    model_args, data_args, train_args, _, _ = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    dataset_module = get_dataset(
        template, model_args, data_args, train_args, stage="sft", **tokenizer_module
    )
    dataset = dataset_module["train_dataset"]
    if not len(dataset):
        raise ValueError("prepared SFT dataset is empty")
    collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=None,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        block_diag_attn=model_args.block_diag_attn,
        neat_packing=data_args.neat_packing,
        attn_implementation=None,
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    example_index = max(range(len(dataset)), key=lambda index: len(dataset[index]["input_ids"])) if scan_longest else 0
    batch = collator([dataset[example_index]])
    labels = batch["labels"]
    supervised = labels.ne(-100)
    if not supervised.any() or "pixel_values" not in batch:
        raise ValueError("first SFT batch must include supervised tokens and screenshot pixels")
    return {
        "dataset_rows": len(dataset),
        "example_index": example_index,
        "batch_shapes": {key: list(value.shape) for key, value in batch.items() if hasattr(value, "shape")},
        "supervised_targets": int(supervised.sum()),
        "first_supervised_position": int(supervised[0].nonzero()[0]),
        "mask_history": data_args.mask_history,
        "model_type": model_args.model_name_or_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--scan-longest", action="store_true")
    args = parser.parse_args()
    print(json.dumps(inspect_batch(args.config, scan_longest=args.scan_longest), indent=2))


if __name__ == "__main__":
    main()
