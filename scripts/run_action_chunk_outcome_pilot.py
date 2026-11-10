#!/usr/bin/env python3
"""Replay frozen actor tool-name chunks and cross-fit a value head offline.

No browser, API, actor training, or free-running decoding occurs. Fit outcomes
come only from valid draws 0-4; all reported action success uses draws 5-7.
"""

import argparse
import base64
from collections import defaultdict
import copy
import hashlib
import io
import json
from pathlib import Path
import random
import time


SFT = Path("/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def replay(rows, model_path, output, batch_size):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", local_files_only=True).eval()
    model.requires_grad_(False)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            path = output / f"features-{start // batch_size:04d}.pt"
            batch = rows[start:start + batch_size]
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if [r["state_id"] for r in saved] != [r["state_id"] for r in batch]:
                    raise ValueError("Feature shard identity mismatch: " + str(path))
                continue
            tic = time.monotonic()
            saved = []
            for row in batch:
                if hashlib.sha256(row["prompt_text"].encode()).hexdigest() != row["prompt_sha256"]:
                    raise ValueError("Saved prompt hash mismatch")
                images = []
                for url in row["images"]:
                    images.append(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[-1]))).convert("RGB"))
                if images:
                    prompt = processor(text=[row["prompt_text"]], images=images, return_tensors="pt").to("cuda:0")
                else:
                    prompt = tokenizer(row["prompt_text"], add_special_tokens=False,
                                       return_tensors="pt").to("cuda:0")
                prompt_len = int(prompt["input_ids"].shape[1])
                pre = model(**prompt, use_cache=True, output_hidden_states=False,
                            cache_position=torch.arange(prompt_len, device="cuda:0"))
                shared = tokenizer.encode(row["shared_reasoning"], add_special_tokens=False)
                if not shared:
                    raise ValueError("Empty shared reasoning response")
                ids = torch.tensor([shared], device="cuda:0")
                common = model(input_ids=ids, past_key_values=pre.past_key_values, use_cache=True,
                               output_hidden_states=False,
                               cache_position=torch.arange(prompt_len, prompt_len + len(shared), device="cuda:0"))
                options = []
                for action in row["actions"]:
                    whole = tokenizer.encode(row["shared_reasoning"] + action["chunk"], add_special_tokens=False)
                    if whole[:len(shared)] != shared:
                        raise ValueError("Tool chunk is not token-aligned with shared reasoning")
                    suffix = whole[len(shared):]
                    if not suffix:
                        raise ValueError("Empty tool-name chunk")
                    chunk_ids = torch.tensor([suffix], device="cuda:0")
                    out = model(input_ids=chunk_ids, past_key_values=copy.deepcopy(common.past_key_values),
                                use_cache=True, output_hidden_states=True,
                                cache_position=torch.arange(prompt_len + len(shared),
                                                            prompt_len + len(whole), device="cuda:0"))
                    logits = torch.cat((common.logits[:, -1:], out.logits[:, :-1]), dim=1).float()
                    logprob = torch.log_softmax(logits, dim=-1).gather(
                        -1, chunk_ids.unsqueeze(-1)).sum().item()
                    options.append(dict(name=action["name"], logprob=logprob,
                                        h=out.hidden_states[36][0, -1].to(torch.float16).cpu()))
                    del out
                saved.append(dict(state_id=row["state_id"], task_group=row["task_group"],
                                  fold=row["fold"], options=options))
                del pre, common, prompt
            temporary = path.with_suffix(path.suffix + ".part")
            torch.save(saved, temporary)
            temporary.replace(path)
            print(json.dumps({"stage": "features", "shard": path.name, "states": len(saved),
                              "seconds": round(time.monotonic() - tic, 1)}), flush=True)


def name_rates(row, draws):
    grouped = defaultdict(lambda: [0, 0])
    for action, outcomes in zip(row["actions"], row["outcomes"]):
        for repeat in draws:
            result = outcomes[repeat]
            if result["valid"]:
                grouped[action["name"]][0] += int(result["success"])
                grouped[action["name"]][1] += 1
    return {name: win / valid for name, (win, valid) in grouped.items() if valid}


def evaluate(rows, features, output):
    import numpy as np
    from scipy.linalg import solve

    if [r["state_id"] for r in rows] != [f["state_id"] for f in features]:
        raise ValueError("Features are not aligned to frozen states")
    # Collapse alternate spellings of the same tool name using the actor's
    # highest-likelihood spelling; the continuation outcome target is shared.
    panels = []
    for row, feature in zip(rows, features):
        by_name = {}
        for option in feature["options"]:
            name = option["name"]
            if name not in by_name or option["logprob"] > by_name[name]["logprob"]:
                by_name[name] = option
        fit, hold = name_rates(row, range(5)), name_rates(row, range(5, 8))
        names = sorted(set(by_name) & set(fit) & set(hold))
        if len(names) < 2:
            raise ValueError("A frozen panel lost a scoreable name")
        h = np.stack([by_name[name]["h"].float().numpy() for name in names])
        h = h - h.mean(0, keepdims=True)
        y = np.array([fit[name] for name in names]); y -= y.mean()
        panels.append(dict(state=row["state_id"], task=row["task_group"], fold=row["fold"],
                           names=names, h=h, y=y, hold=hold,
                           multiple_calls=any(len(json.loads(action["action_key"])) > 1
                                              for action in row["actions"]),
                           actor={name: by_name[name]["logprob"] for name in names}))
    predictions, selected_alpha = {}, {}
    alphas = [1e2, 1e4, 1e6]
    for fold in range(5):
        train = [p for p in panels if p["fold"] != fold]
        test = [p for p in panels if p["fold"] == fold]
        if not test:
            continue
        X = np.concatenate([p["h"] for p in train]); Y = np.concatenate([p["y"] for p in train])
        groups = np.concatenate([[p["task"]] * len(p["names"]) for p in train])
        weights = np.concatenate([[1 / len(p["names"])] * len(p["names"]) for p in train])
        mu, sd = X.mean(0), X.std(0) + 1e-5
        X = (X - mu) / sd
        task_names = sorted(set(groups), key=lambda name: hashlib.sha256(name.encode()).hexdigest())
        inner = []
        for k in range(3):
            validation_tasks = set(task_names[k::3])
            val_ids = np.array([i for i, group in enumerate(groups) if group in validation_tasks])
            fit_ids = np.array([i for i, group in enumerate(groups) if group not in validation_tasks])
            inner.append((fit_ids, val_ids))
        gram = X @ X.T

        def solve_dual(fit_ids, alpha):
            root_weight = np.sqrt(weights[fit_ids])
            matrix = gram[np.ix_(fit_ids, fit_ids)] * root_weight[:, None] * root_weight[None, :]
            matrix.flat[::len(fit_ids) + 1] += alpha
            dual = solve(matrix, Y[fit_ids] * root_weight, assume_a="pos", check_finite=False)
            return root_weight * dual

        loss = {}
        for alpha in alphas:
            scores = []
            for fit_ids, val_ids in inner:
                coefficients = solve_dual(fit_ids, alpha)
                pred = gram[np.ix_(val_ids, fit_ids)] @ coefficients
                scores.append(np.average((pred - Y[val_ids]) ** 2, weights=weights[val_ids]))
            loss[alpha] = float(np.mean(scores))
        alpha = min(alphas, key=lambda a: (loss[a], a)); selected_alpha[str(fold)] = alpha
        coefficients = solve_dual(np.arange(len(X)), alpha)
        head = X.T @ coefficients
        for p in test:
            pred = ((p["h"] - mu) / sd) @ head
            predictions[p["state"]] = dict(zip(p["names"], map(float, pred)))
    records = []
    for p in panels:
        actor = max(p["names"], key=lambda n: (p["actor"][n], -p["names"].index(n)))
        value = max(p["names"], key=lambda n: (predictions[p["state"]][n], -p["names"].index(n)))
        records.append(dict(state_id=p["state"], task_group=p["task"], fold=p["fold"],
                            multiple_calls=p["multiple_calls"],
                            actor_name=actor, value_name=value,
                            actor_success=p["hold"][actor], value_success=p["hold"][value],
                            uniform_success=float(np.mean([p["hold"][n] for n in p["names"]]))))
    diff = [r["value_success"] - r["actor_success"] for r in records]
    tasks = defaultdict(list)
    for record, delta in zip(records, diff):
        tasks[record["task_group"]].append(delta)
    task_values = list(tasks.values()); rng = random.Random(0); bootstrap = []
    for _ in range(5000):
        draw = [rng.choice(task_values) for _ in task_values]
        bootstrap.append(100 * sum(map(sum, draw)) / sum(map(len, draw)))
    bootstrap.sort()
    report = dict(states=len(records), task_groups=len(tasks), fold_alpha=selected_alpha,
                  actor_success_pct=100 * float(np.mean([r["actor_success"] for r in records])),
                  value_success_pct=100 * float(np.mean([r["value_success"] for r in records])),
                  uniform_name_success_pct=100 * float(np.mean([r["uniform_success"] for r in records])),
                  value_minus_actor_pp=[100 * float(np.mean(diff)), bootstrap[125], bootstrap[4875]],
                  changed_name_states=sum(r["actor_name"] != r["value_name"] for r in records),
                  target="valid-only outcomes in held-out draws 5-7; task-disjoint 5-fold cross-fit",
                  limit="observed tool-name branches only; no free-running action generation")
    report["subgroups"] = {}
    for label, subset in (("single_call_panels", [r for r in records if not r["multiple_calls"]]),
                          ("multiple_call_panels", [r for r in records if r["multiple_calls"]])):
        report["subgroups"][label] = dict(
            states=len(subset), task_groups=len({r["task_group"] for r in subset}),
            actor_success_pct=100 * float(np.mean([r["actor_success"] for r in subset])) if subset else None,
            value_success_pct=100 * float(np.mean([r["value_success"] for r in subset])) if subset else None)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "picks.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"stage": "complete", **report}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=SFT)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    if sha(args.panels) != manifest["panels_sha256"]:
        raise ValueError("Frozen panel SHA-256 mismatch")
    rows = load_rows(args.panels)
    if len(rows) != manifest["counts"]["admitted_states"]:
        raise ValueError("Frozen panel count mismatch")
    identity = {"panels_sha256": sha(args.panels), "model": str(args.model),
                "scope": "frozen actor, observed tool names, zero API/browser"}
    path = args.output / "identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Output belongs to another model or panel")
    path.write_text(json.dumps(identity, indent=2) + "\n")
    replay(rows, args.model, args.output, args.batch_size)
    import torch
    features = [item for shard in sorted(args.output.glob("features-*.pt"))
                for item in torch.load(shard, map_location="cpu", weights_only=False)]
    evaluate(rows, features, args.output)


if __name__ == "__main__":
    main()
