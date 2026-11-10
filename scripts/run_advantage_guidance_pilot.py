#!/usr/bin/env python3
"""Train and score an action-only advantage-conditioned OpenWebRL-4B-SFT LoRA.

Run only inside a separately approved Slurm allocation. The manifest specifies
the fitting and held-out draws. Held-out outcomes are read only for final
evaluation, never for optimization or checkpoint choice. Every action is
teacher-forced under the same empty-reasoning prefix.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import signal
import time

from prepare_action_chunk_outcome_pilot import digest


def condition(prefix: str, advantage: float) -> str:
    if not prefix.endswith("<think>\n</think>") or not math.isfinite(advantage):
        raise ValueError("Invalid action-only condition boundary")
    return prefix[:-len("</think>")] + f"Advantage: {advantage:+.3f}\n</think>"


def planned_label(row: dict, index: int, sigma: float, seed: int) -> float:
    key = f"{seed}|{row['state_id']}|{index}".encode()
    rng = random.Random(int(hashlib.sha256(key).hexdigest(), 16))
    return row["panel_relative_advantage"][index] + rng.gauss(0, sigma)


def images_for(row: dict):
    from PIL import Image
    if "image_path" in row:
        data = Path(row["image_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["image_sha256"]:
            raise ValueError("Saved screenshot digest mismatch")
        return [Image.open(io.BytesIO(data)).convert("RGB")]
    return [Image.open(io.BytesIO(base64.b64decode(item.split(",", 1)[-1]))).convert("RGB")
            for item in row["images"]]


def example(row: dict, index: int, y: float | None, processor, device):
    import torch
    prefix = condition(row["action_only_prompt"], y) if y is not None else row["action_only_prompt"]
    images = images_for(row)
    full = processor(text=[prefix + row["actions"][index]], images=images, return_tensors="pt")
    pre = processor(text=[prefix], images=images, return_tensors="pt")
    prefix_ids = pre["input_ids"][0]
    ids = full["input_ids"][0]
    start = len(prefix_ids)
    if not start or start >= len(ids) or not torch.equal(ids[:start], prefix_ids):
        raise ValueError("Action tokens are not aligned after the conditional prefix")
    if len(ids) > 32768:
        raise ValueError("Sequence exceeds the actor context limit")
    if ids[-1].item() != processor.tokenizer.convert_tokens_to_ids("<|im_end|>"):
        raise ValueError("Action sequence does not end with the assistant terminator")
    del pre
    positions = torch.arange(start - 1, len(ids) - 1, device=device)
    target = ids[start:].to(device)
    model_input = {key: value.to(device) for key, value in full.items()}
    return model_input, positions, target


def action_logps(model, example):
    import torch
    inputs, positions, target = example
    output = model(**inputs, logits_to_keep=positions, use_cache=False)
    logits = output.logits
    if logits.shape[1] != len(target):
        raise ValueError("Action logits and tokens are misaligned")
    selected = []
    for begin in range(0, len(target), 32):
        end = min(len(target), begin + 32)
        block = logits[:, begin:end].float()
        chosen = target[begin:end].view(1, -1)
        selected.append(block.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
                        - torch.logsumexp(block, dim=-1))
    return torch.cat(selected, dim=1).view(-1)


def load_pinned(path: Path, expected_sha: str):
    if digest(path) != expected_sha:
        raise ValueError("Frozen pilot dataset SHA-256 mismatch")
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if len({row["state_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate state")
    train_tasks = {row["task_group"] for row in rows if row["fold"] < 4}
    val_tasks = {row["task_group"] for row in rows if row["fold"] == 4}
    if train_tasks & val_tasks:
        raise ValueError("Task leaks across split")
    return rows


def train(config: dict, rows: list[dict], model, processor, output: Path, deadline: float):
    import torch
    from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    train_rows = [row for row in rows if row["fold"] < 4]
    if not train_rows:
        raise ValueError("No training states")
    params = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], betas=(.9, .95),
                                  eps=1e-8, weight_decay=.01)
    samples = [(i, action) for i, row in enumerate(train_rows)
               for action in range(len(row["actions"]))]
    random.Random(config["seed"]).shuffle(samples)
    accum = config["gradient_accumulation"]
    if accum < 1:
        raise ValueError("Invalid accumulation")
    updates = math.ceil(len(samples) / accum)
    state = {"next_sample": 0, "updates": 0, "dataset_sha256": config["dataset_sha256"]}
    checkpoint = output / "latest"
    if checkpoint.exists():
        saved = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
        state = saved["state"]
        if state["dataset_sha256"] != config["dataset_sha256"]:
            raise ValueError("Resume dataset differs")
        set_peft_model_state_dict(model, load_file(str(checkpoint / "adapter_model.safetensors")))
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def save():
        temporary = output / "latest.incomplete"
        if temporary.exists():
            raise ValueError("Incomplete checkpoint needs inspection")
        temporary.mkdir()
        model.save_pretrained(temporary, safe_serialization=True)
        torch.save({"state": state, "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()},
                   temporary / "optimizer.pt")
        # Rename old checkpoint out of the way before atomically installing the
        # new one; keep the previous full state for recovery and audit.
        if checkpoint.exists():
            checkpoint.rename(output / f'update-{state["updates"]:06d}-previous')
        temporary.rename(checkpoint)
        (output / "progress.json").write_text(json.dumps(state, indent=2) + "\n")
    model.train()
    for begin in range(state["next_sample"], len(samples), accum):
        if stopped or time.time() > deadline - 600:
            save()
            return False
        batch = samples[begin:begin + accum]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for state_index, action in batch:
            row = train_rows[state_index]
            y = planned_label(row, action, config["label_sigma"], config["seed"])
            value = -action_logps(model, example(row, action, y, processor, "cuda:0")).mean()
            if not torch.isfinite(value):
                raise ValueError("Nonfinite action loss")
            (value / len(batch)).backward()
            losses.append(value.detach().item())
        gradient = torch.nn.utils.clip_grad_norm_(params, 1.)
        if not torch.isfinite(gradient):
            raise ValueError("Nonfinite gradient")
        step = state["updates"] + 1
        warmup = max(1, math.ceil(.03 * updates))
        factor = min(1., step / warmup) if step <= warmup else .5 * (
            1 + math.cos(math.pi * (step - warmup) / max(1, updates - warmup)))
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rate"] * factor
        optimizer.step()
        state.update(next_sample=begin + len(batch), updates=step)
        event = {**state, "loss": sum(losses) / len(losses), "grad_norm": gradient.item(),
                 "lr": optimizer.param_groups[0]["lr"], "unix": time.time()}
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print("UPDATE", json.dumps(event), flush=True)
        if step % config["checkpoint_every"] == 0:
            save()
    save()
    return True


def score(rows: list[dict], model, processor, output: Path, deadline: float):
    import torch
    model.eval()
    scored = {}
    path = output / "validation-scores.jsonl"
    if path.exists():
        for line in path.open():
            item = json.loads(line)
            scored[item["state_id"]] = item
    with torch.inference_mode():
        for row in rows:
            if row["fold"] != 4 or row["state_id"] in scored:
                continue
            if time.time() > deadline - 300:
                break
            plus, minus, actor = [], [], []
            for action in range(len(row["actions"])):
                positive_input = example(row, action, .25, processor, "cuda:0")
                negative_input = example(row, action, -.25, processor, "cuda:0")
                actor_input = example(row, action, None, processor, "cuda:0")
                if not torch.equal(positive_input[2], negative_input[2]) or not torch.equal(
                        positive_input[2], actor_input[2]):
                    raise ValueError("The three branches tokenize the action differently")
                plus.append(action_logps(model, positive_input).sum().item())
                minus.append(action_logps(model, negative_input).sum().item())
                with model.disable_adapter():
                    actor.append(action_logps(model, actor_input).sum().item())
            record = {"state_id": row["state_id"], "task_group": row["task_group"],
                      "implicit": [p - n for p, n in zip(plus, minus)],
                      "positive": plus, "actor": actor,
                      "holdout": row["pooled_action_holdout_rate"]}
            with path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            scored[row["state_id"]] = record
            print("SCORED", len(scored), row["state_id"], flush=True)
    return len(scored)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("A separately approved Slurm allocation is required")
    if int(os.environ.get("SLURM_GPUS_ON_NODE", "0")) != 1:
        raise ValueError("Expected one assigned GPU")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    rows = load_pinned(Path(config["dataset"]), config["dataset_sha256"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + config["runtime_seconds"]
    model_path = config["actor"]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True,
                torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa")
    suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    targets = [name for name, _ in model.named_modules()
               if name.startswith("model.language_model.layers.") and name.rsplit(".", 1)[-1] in suffixes]
    if not targets:
        raise ValueError("No language-only LoRA targets")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                 target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    complete = train(config, rows, model, processor, output, deadline)
    if complete:
        scored = score(rows, model, processor, output, deadline)
        if scored == sum(row["fold"] == 4 for row in rows):
            (output / "complete.json").write_text(json.dumps(
                {"status": "training_and_offline_scoring_complete", "validation_states": scored,
                 "dataset_sha256": config["dataset_sha256"]}, indent=2) + "\n")
    else:
        print("PAUSED_WITH_DURABLE_CHECKPOINT", flush=True)


if __name__ == "__main__":
    main()
