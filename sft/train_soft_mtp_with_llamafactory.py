"""Use LLaMAFactory's OpenWebRL SFT data/trainer with Qwen3-VL MTP heads.

Run from a pinned LLaMAFactory checkout with its dependencies installed::

    python /path/to/OpenWebRL/sft/train_soft_mtp_with_llamafactory.py \
        --mode soft --horizon 3 --weight 1.0 /path/to/sft.yaml

Only the MTP training path is replaced. LLaMAFactory retains preprocessing,
the multimodal collator, optimizer, scheduler, distributed training, and saves.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from safetensors.torch import save_file

from sft.qwen3vl_mtp import attach_mtp_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", choices=("hard", "soft"), default="soft")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--weight", type=float, default=1.0)
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error(f"missing SFT config: {args.config}")

    # Import after CLI validation, so `--help` does not require the trainer.
    from llamafactory.train import tuner
    from llamafactory.train.sft import workflow

    base_load_model = workflow.load_model
    loaded = {}

    def load_mtp_model(tokenizer, model_args, finetuning_args, is_trainable):
        if not is_trainable or finetuning_args.finetuning_type != "full":
            raise ValueError("MTP SFT requires full-parameter training")
        model = base_load_model(tokenizer, model_args, finetuning_args, is_trainable)
        loaded["model"] = attach_mtp_training(model, mode=args.mode, horizon=args.horizon, weight=args.weight)
        return loaded["model"]

    workflow.load_model = load_mtp_model
    sys.argv = [sys.argv[0], str(args.config.resolve())]
    tuner.run_exp()

    # Preserve draft weights separately for future acceptance/latency tests.
    # ZeRO-2 replicates parameters; the original SFT recipe uses ZeRO-2.
    if int(os.environ.get("RANK", "0")) == 0 and "model" in loaded:
        from omegaconf import OmegaConf

        output_dir = Path(OmegaConf.load(args.config).output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in loaded["model"].soft_mtp_heads.state_dict().items()
        }
        save_file(state, str(output_dir / "soft_mtp_heads.safetensors"))


if __name__ == "__main__":
    main()
