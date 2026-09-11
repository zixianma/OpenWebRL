#!/usr/bin/env python3
"""Two-GPU, same-state ARM preference distillation with calibrated loss weight."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

from openwebrl.arm_c2 import sha, write_json


def language_targets(model):
    suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    return [name for name, _ in model.named_modules()
            if name.startswith("model.language_model.layers.") and name.rsplit(".", 1)[-1] in suffixes]


def normalize_response(value):
    return " ".join(value.split()).casefold()


def choose_loser(row, captured):
    winner = captured["raw_candidates"][row["selected_index"]]
    choices = [(index, candidate) for index, candidate in enumerate(captured["raw_candidates"])
               if index != row["selected_index"]
               and candidate.get("finish_type") == "stop"
               and candidate.get("response_token_ids")
               and normalize_response(candidate.get("response", "")) != normalize_response(winner.get("response", ""))]
    if not choices:
        return None
    key = f'arm-pref-v1:42:{row["task_id"]}:{row["turn"]}'.encode()
    return choices[int(hashlib.sha256(key).hexdigest(), 16) % len(choices)]


def make_example(row, candidate_index, processor):
    from PIL import Image
    import torch
    from slime.utils.processing_utils import build_processor_kwargs
    source = Path(row["source"])
    if sha(source.read_bytes()) != row["source_sha256"]:
        raise ValueError("Training source hash mismatch")
    captured = json.loads(source.read_text())
    images = []
    for item in captured["images"]:
        path = Path(item["path"])
        if sha(path.read_bytes()) != item["sha256"]:
            raise ValueError("Training image hash mismatch")
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    features = processor(text=[captured["prompt"]], **build_processor_kwargs({"images": images}))
    prefix = features["input_ids"][0]
    if hasattr(prefix, "tolist"):
        prefix = prefix.tolist()
    grid = features["image_grid_thw"]
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    if prefix != row["prompt_token_ids"] or grid != row["image_grid_thw"]:
        raise ValueError("Processor-expanded state differs from executed state")
    target = captured["raw_candidates"][candidate_index]["response_token_ids"]
    ids = torch.tensor([prefix + target], dtype=torch.long)
    labels = torch.full_like(ids, -100)
    labels[0, len(prefix):] = torch.tensor(target, dtype=torch.long)
    tensors = {key: value for key, value in features.items()
               if key not in ("input_ids", "attention_mask") and isinstance(value, torch.Tensor)}
    return dict(input_ids=ids, labels=labels, attention_mask=torch.ones_like(ids), **tensors)


def response_logmean(output, example):
    import torch
    logits = output.logits
    tokens = example["labels"][example["labels"].ne(-100)].view(1, -1)
    if logits.shape[1] != tokens.shape[1]:
        raise ValueError("Response-only logits do not align with response targets")
    # Score in float32 without materializing every response-position logit in
    # float32 at once. BF16 cannot resolve the small reference-relative margins
    # around log(2) that this objective is intended to measure.
    selected = []
    for begin in range(0, logits.shape[1], 32):
        end = min(logits.shape[1], begin + 32)
        chunk = logits[:, begin:end].float()
        chunk_tokens = tokens[:, begin:end]
        selected.append(chunk.gather(-1, chunk_tokens.unsqueeze(-1)).squeeze(-1)
                        - torch.logsumexp(chunk, dim=-1))
    return torch.cat(selected, dim=1).mean()


def response_forward(model, example):
    import torch
    inputs = {key: value for key, value in example.items() if key != "labels"}
    target_positions = torch.nonzero(example["labels"][0].ne(-100), as_tuple=False).flatten()
    prediction_positions = target_positions - 1
    if prediction_positions[0] < 0:
        raise ValueError("Response begins before a causal prediction position")
    return model(**inputs, logits_to_keep=prediction_positions, use_cache=False)


def grad_norm(parameters):
    import torch
    values = [parameter.grad.detach().float().norm(2) ** 2 for parameter in parameters if parameter.grad is not None]
    return torch.stack(values).sum().sqrt() if values else torch.tensor(0.0, device="cuda")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dist.init_process_group("nccl")
    rank, world, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    if world != 2:
        raise ValueError(f"Preference run requires exactly two ranks, found {world}")
    torch.cuda.set_device(local_rank)
    config = json.loads(args.config.read_text())
    if os.getenv("SLURM_JOB_ID") != str(config["allocation"]):
        raise ValueError("Trainer is outside the authorized allocation")
    root = Path(config["output"]); student = root / "student"; student.mkdir(parents=True, exist_ok=True)
    audit = json.loads(Path(config["source_audit"]).read_text())
    dataset = Path(audit["dataset"])
    if sha(dataset.read_bytes()) != audit["dataset_sha256"]:
        raise ValueError("C2 dataset changed")
    source_rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    pairs = []
    for row in source_rows:
        captured = json.loads(Path(row["source"]).read_text())
        loser = choose_loser(row, captured)
        if loser is not None:
            pairs.append(dict(row=row, loser_index=loser[0]))
    random.Random(config["seed"]).shuffle(pairs)
    pairs = pairs[:config["pair_count"]]
    manifest_hash = sha("".join(f'{p["row"]["source_sha256"]}:{p["row"]["selected_index"]}:{p["loser_index"]}\n' for p in pairs).encode())
    if rank == 0:
        write_json(root / "pair-audit.json", {"source_turns": len(source_rows), "pairable_turns": 8363,
                   "pair_coverage": 8363 / len(source_rows), "trained_pairs": len(pairs),
                   "selection": "seed-42 shuffled prefix; seeded distinct executable loser", "manifest_sha256": manifest_hash})
    actor = config["actor"]
    processor = AutoProcessor.from_pretrained(actor, local_files_only=True)
    common = dict(local_files_only=True, torch_dtype=torch.bfloat16,
                  device_map={"": f"cuda:{local_rank}"}, attn_implementation="sdpa")
    reference = AutoModelForImageTextToText.from_pretrained(actor, **common).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    policy = AutoModelForImageTextToText.from_pretrained(actor, **common)
    targets = language_targets(policy)
    policy = get_peft_model(policy, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                            target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.enable_input_require_grads(); policy.config.use_cache = False; policy.train()
    parameters = [p for p in policy.parameters() if p.requires_grad]
    ddp = DDP(policy, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, betas=(.9, .95), eps=1e-8, weight_decay=.01)
    beta = config["beta"]

    def pair_losses(pair):
        row = pair["row"]
        chosen = make_example(row, row["selected_index"], processor)
        rejected = make_example(row, pair["loser_index"], processor)
        chosen = {k: v.to(local_rank) for k, v in chosen.items()}
        rejected = {k: v.to(local_rank) for k, v in rejected.items()}
        with torch.no_grad():
            ref_output = response_forward(reference, chosen)
            ref_chosen = response_logmean(ref_output, chosen); del ref_output
            ref_output = response_forward(reference, rejected)
            ref_rejected = response_logmean(ref_output, rejected); del ref_output
        chosen_out = response_forward(ddp, chosen)
        chosen_logp = response_logmean(chosen_out, chosen)
        rejected_out = response_forward(ddp, rejected)
        rejected_logp = response_logmean(rejected_out, rejected)
        margin = chosen_logp - rejected_logp
        ref_margin = ref_chosen - ref_rejected
        pref = -torch.nn.functional.logsigmoid(beta * (margin - ref_margin))
        return -chosen_logp, pref, margin.detach(), ref_margin.detach()

    # Use the same frozen calibration pairs on both ranks, then broadcast rank-zero's coefficient.
    calibration = pairs[:config["calibration_pairs"]]
    optimizer.zero_grad(set_to_none=True)
    sft_value = 0.0
    for pair in calibration:
        sft, _, _, _ = pair_losses(pair); (sft / len(calibration)).backward(); sft_value += sft.item() / len(calibration)
    sft_norm = grad_norm(parameters); optimizer.zero_grad(set_to_none=True)
    pref_value = 0.0
    for pair in calibration:
        _, pref, _, _ = pair_losses(pair); (pref / len(calibration)).backward(); pref_value += pref.item() / len(calibration)
    pref_norm = grad_norm(parameters)
    coefficient = torch.tensor([max(.05, min(20., config["target_gradient_ratio"] * sft_norm.item() / max(pref_norm.item(), 1e-12)))], device=local_rank)
    dist.broadcast(coefficient, src=0); preference_weight = coefficient.item()
    optimizer.zero_grad(set_to_none=True)
    if rank == 0:
        write_json(student / "calibration.json", {"pairs": len(calibration), "beta": beta,
                   "target_gradient_ratio": config["target_gradient_ratio"], "sft_loss": sft_value,
                   "preference_loss": pref_value, "sft_gradient_norm": sft_norm.item(),
                   "unit_preference_gradient_norm": pref_norm.item(), "preference_weight": preference_weight})
        write_json(student / "training-config.json", {**config, "pair_manifest_sha256": manifest_hash,
                   "preference_weight": preference_weight, "trainable_parameters": sum(p.numel() for p in parameters),
                   "objective": "winner CE + calibrated lambda * length-normalized reference-relative DPO"})

    local_pairs = pairs[rank::world]
    accum = config["gradient_accumulation_per_rank"]
    updates = math.ceil(len(local_pairs) / accum)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    start_time = time.time()
    for update, begin in enumerate(range(0, len(local_pairs), accum), 1):
        if stopped or time.time() > config["stop_unix"] - 240:
            break
        batch = local_pairs[begin:begin + accum]
        optimizer.zero_grad(set_to_none=True)
        sums = torch.zeros(4, device=local_rank)
        for offset, pair in enumerate(batch):
            sync = offset == len(batch) - 1
            with nullcontext() if sync else ddp.no_sync():
                sft, pref, margin, ref_margin = pair_losses(pair)
                ((sft + preference_weight * pref) / len(batch)).backward()
            sums += torch.stack((sft.detach(), pref.detach(), margin, ref_margin)) / len(batch)
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        response_exposures = min(config["pair_count"], update * config["effective_batch_pairs"]) * 2
        if response_exposures <= 512:
            factor = response_exposures / 512
        else:
            factor = .5 * (1 + math.cos(math.pi * (response_exposures - 512) / (16788 - 512)))
        for group in optimizer.param_groups: group["lr"] = 1e-5 * factor
        optimizer.step()
        dist.all_reduce(sums); sums /= world
        if rank == 0:
            event = {"utc": datetime.now(timezone.utc).isoformat(), "epoch": 0,
                     "next_position": min(config["pair_count"], update * config["effective_batch_pairs"]),
                     "updates": update, "loss": sums[0].item(), "sft_loss": sums[0].item(),
                     "preference_loss": sums[1].item(), "policy_margin": sums[2].item(),
                     "reference_margin": sums[3].item(), "preference_weight": preference_weight,
                     "grad_norm": gradient.item(), "lr": optimizer.param_groups[0]["lr"],
                     "elapsed_seconds": time.time() - start_time}
            with (student / "metrics.jsonl").open("a") as stream: stream.write(json.dumps(event) + "\n")
            print("UPDATE", json.dumps(event), flush=True)
        if update in config["save_updates"] or update == updates or stopped:
            dist.barrier()
            if rank == 0:
                label = f"update-{update:06d}" if update < updates else "endpoint"
                temporary = student / (label + ".incomplete")
                temporary.mkdir(exist_ok=False)
                policy.save_pretrained(temporary, safe_serialization=True); processor.save_pretrained(temporary)
                write_json(temporary / "progress.json", {"updates": update, "pairs_seen": min(config["pair_count"], update * config["effective_batch_pairs"]),
                           "pair_manifest_sha256": manifest_hash, "preference_weight": preference_weight})
                temporary.rename(student / label)
            dist.barrier()
    if rank == 0:
        completed = update == updates
        write_json(student / ("complete.json" if completed else "paused.json"),
                   {"updates": update, "total_updates": updates, "checkpoint": str(student / ("endpoint" if completed else f"update-{update:06d}")),
                    "preference_weight": preference_weight, "completed": completed})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
