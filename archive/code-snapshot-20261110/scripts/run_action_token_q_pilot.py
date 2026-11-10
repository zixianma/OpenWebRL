#!/usr/bin/env python3
"""Offline action-token Q pilot on inherited GPT-5.5 action preferences.

Frozen actor forwards recompute each observed action fork after a shared
reasoning prefix. A small Q(h, token) head learns the preferred versus rejected
next token at those forks. This does not run guided decoding or browser episodes.
"""
import argparse
from collections import Counter
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

SFT = Path("/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT")


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(rows, root, model_path, chunk_size=25):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from openwebrl.selection_head_features import encode_prefix

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", local_files_only=True).eval()
    model.requires_grad_(False)
    input_embedding = model.get_input_embeddings()
    if input_embedding is None:
        raise ValueError("Actor input embedding is unavailable")
    token_ids = sorted({token for row in rows for fork in row["forks"]
                        for _, _, left, right in fork["comparisons"] for token in (left, right)})
    emb_path = root / "token-embeddings.pt"
    if not emb_path.exists():
        embeddings = input_embedding.weight[torch.tensor(token_ids, device="cuda:0")].detach().to(torch.float16).cpu()
        torch.save({"token_ids": token_ids, "embeddings": embeddings}, emb_path)
    else:
        saved = torch.load(emb_path, map_location="cpu", weights_only=False)
        if saved["token_ids"] != token_ids:
            raise ValueError("Saved token embeddings belong to another fork set")
    with torch.inference_mode():
        for start in range(0, len(rows), chunk_size):
            path = root / f"features-{start // chunk_size:04d}.pt"
            chunk = rows[start:start + chunk_size]
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if [x["state_id"] for x in saved] != [x["state_id"] for x in chunk]:
                    raise ValueError(f"Existing feature chunk has wrong states: {path}")
                continue
            started = time.monotonic()
            output = []
            for row in chunk:
                if hashlib.sha256(row["prompt_text"].encode()).hexdigest() != row["prompt_sha256"] and row.get("prompt_verified", True):
                    raise ValueError("Prompt hash changed for " + row["state_id"])
                with Image.open(row["image"]) as source:
                    image = source.convert("RGB")
                prompt = processor(text=[row["prompt_text"]], images=[image], return_tensors="pt").to("cuda:0")
                prompt_length = prompt["input_ids"].shape[1]
                _, cache = encode_prefix(model, prompt, (36,))
                forks = []
                for fork in row["forks"]:
                    prefix_ids = fork["prefix_ids"]
                    if not prefix_ids:
                        raise ValueError("Action fork lacks shared response prefix")
                    ids = torch.tensor([prefix_ids], dtype=torch.long, device="cuda:0")
                    out = model(input_ids=ids, past_key_values=copy.deepcopy(cache),
                                use_cache=True, output_hidden_states=True,
                                cache_position=torch.arange(prompt_length, prompt_length + ids.shape[1], device="cuda:0"))
                    h = out.hidden_states[36][0, -1].to(torch.float16).cpu()
                    tokens = sorted({v for _, _, a, b in fork["comparisons"] for v in (a, b)})
                    logits = {v: float(out.logits[0, -1, v]) for v in tokens}
                    forks.append({"h": h, "comparisons": fork["comparisons"], "actor_logits": logits})
                    del out
                output.append({"state_id": row["state_id"], "split": row["split"],
                               "positive_indices": row.get("positive_indices", []),
                               "negative_indices": row.get("negative_indices", []),
                               "prompt_verified": row.get("prompt_verified", True),
                               "outcomes": row.get("outcomes"), "forks": forks})
                del cache, prompt
            partial = path.with_name(path.name + f".part-{os.getpid()}")
            torch.save(output, partial)
            os.replace(partial, path)
            print(json.dumps({"stage": "extraction", "chunk": path.name,
                              "panels": len(output), "seconds": round(time.monotonic() - started, 2)}), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()



def load_features(root, rows):
    import torch
    files = sorted(root.glob("features-*.pt"))
    got = [item for path in files for item in torch.load(path, map_location="cpu", weights_only=False)]
    if [x["state_id"] for x in got] != [x["state_id"] for x in rows]:
        raise ValueError("Feature chunks do not cover the frozen panel order")
    saved = torch.load(root / "token-embeddings.pt", map_location="cpu", weights_only=False)
    return got, {token: vec.float() for token, vec in zip(saved["token_ids"], saved["embeddings"])}


def comparisons(rows, embeddings, split):
    import torch
    h, pos_e, neg_e, actor_margin, weights = [], [], [], [], []
    panel_count = 0
    for row in rows:
        if row["split"] != split:
            continue
        pos, neg = set(row["positive_indices"]), set(row["negative_indices"])
        pairs = []
        for fork in row["forks"]:
            for i, j, a, b in fork["comparisons"]:
                if i in pos and j in neg:
                    pairs.append((fork, a, b))
                elif j in pos and i in neg:
                    pairs.append((fork, b, a))
        if not pairs:
            continue
        panel_count += 1
        for fork, winner, loser in pairs:
            h.append(fork["h"].float()); pos_e.append(embeddings[winner]); neg_e.append(embeddings[loser])
            actor_margin.append(fork["actor_logits"][winner] - fork["actor_logits"][loser])
            weights.append(1 / len(pairs))
    if not h:
        raise ValueError("No labeled comparisons in " + split)
    return (torch.stack(h), torch.stack(pos_e), torch.stack(neg_e),
            torch.tensor(actor_margin), torch.tensor(weights)), panel_count


def train_and_evaluate(rows, embeddings, root, device="cuda:0"):
    import torch
    from torch import nn
    from torch.nn import functional as F

    train, n_train = comparisons(rows, embeddings, "train")
    validation, n_validation = comparisons(rows, embeddings, "validation")
    dim = train[0].shape[-1]
    torch.manual_seed(0)

    class TokenQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.state = nn.Linear(dim, 32, bias=False)
            self.token = nn.Linear(dim, 32, bias=False)
            self.token_bias = nn.Linear(dim, 1, bias=False)

        def forward(self, h, e):
            h, e = F.layer_norm(h, (dim,)), F.layer_norm(e, (dim,))
            return (self.state(h) * self.token(e)).sum(-1) / math.sqrt(32) + self.token_bias(e).squeeze(-1)

    head = TokenQ().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=.1)
    train = tuple(t.to(device) for t in train)
    validation = tuple(t.to(device) for t in validation)
    history = []

    def score(data, mode):
        h, pos, neg, actor, weights = data
        with torch.no_grad():
            qmargin = head(h, pos) - head(h, neg)
            margin = {"actor": actor, "q": qmargin}[mode]
            accuracy = (((margin > 0).float() + .5 * (margin == 0).float()) * weights).sum() / weights.sum()
            return float(accuracy), float((F.softplus(-margin) * weights).sum() / weights.sum())

    for epoch in range(1, 101):
        head.train()
        h, pos, neg, _, weights = train
        margin = head(h, pos) - head(h, neg)
        loss = (F.softplus(-margin) * weights).sum() / weights.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch in (1, 10, 25, 50, 100):
            head.eval()
            history.append({"epoch": epoch, "train_q_accuracy": score(train, "q")[0],
                            "validation_q_accuracy": score(validation, "q")[0],
                            "train_loss": float(loss.detach())})
            print(json.dumps({"stage": "training", **history[-1]}), flush=True)
    head.eval()
    metrics = {"train_panels": n_train, "validation_panels": n_validation,
               "train_pairs": len(train[0]), "validation_pairs": len(validation[0]),
               "validation_actor_only": score(validation, "actor"),
               "validation_q_only": score(validation, "q"),
               "train_actor_only": score(train, "actor"),
               "train_q_only": score(train, "q"),
               "history": history, "checkpoint_rule": "fixed 100 epochs, no validation selection"}
    metrics["branch124_observed_fork_outcomes"] = branch_outcomes(rows, embeddings, head, device)
    torch.save({"state_dict": head.cpu().state_dict(), "hidden_size": dim,
                "objective": "first observed action-token fork, inherited teacher comparison"}, root / "token-q.pt")
    return metrics


def branch_outcomes(rows, embeddings, head, device):
    """Compare observed five-action forks; this is not free-running decoding."""
    import torch

    records = []
    for row in rows:
        if row["split"] != "branch124":
            continue
        votes = {mode: [0.0] * 5 for mode in ("actor", "q")}
        with torch.no_grad():
            for fork in row["forks"]:
                h = fork["h"].float().to(device).unsqueeze(0)
                tokens = {v for _, _, a, b in fork["comparisons"] for v in (a, b)}
                q = {v: float(head(h, embeddings[v].to(device).unsqueeze(0))) for v in tokens}
                for i, j, a, b in fork["comparisons"]:
                    for mode, margin in (
                        ("actor", fork["actor_logits"][a] - fork["actor_logits"][b]),
                        ("q", q[a] - q[b]),
                    ):
                        votes[mode][i] += 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
                        votes[mode][j] += 1.0 if margin < 0 else 0.5 if margin == 0 else 0.0
        picks = {mode: max(range(5), key=lambda i: score[i]) for mode, score in votes.items()}
        records.append({"prompt_verified": row["prompt_verified"], "outcomes": row["outcomes"],
                        "has_observed_fork": bool(row["forks"]),
                        "picks": picks,
                        "ties": {mode: votes[mode].count(max(votes[mode])) > 1 for mode in votes}})

    def summarize(items):
        result = {"states": len(items), "fully_valid_states": sum(all(x["valid"] > 0 for x in r["outcomes"]) for r in items)}
        for mode in ("uniform", "actor", "q"):
            def value(r, valid_only):
                rewards = [x["successes"] / (x["valid"] if valid_only else x["trials"])
                           for x in r["outcomes"]]
                return sum(rewards) / 5 if mode == "uniform" else rewards[r["picks"][mode]]
            full = [r for r in items if all(x["valid"] > 0 for x in r["outcomes"])]
            result[mode] = {"all_state_success_per_recorded_draw": sum(value(r, False) for r in items) / len(items) if items else None,
                            "fully_valid_state_success_per_valid_draw": sum(value(r, True) for r in full) / len(full) if full else None}
            if mode != "uniform":
                result[mode]["ties"] = sum(r["ties"][mode] for r in items)
        return result

    scored = [r for r in records if r["has_observed_fork"]]
    return {"inventory": {"historical_states": len(records), "states_with_observed_fork": len(scored),
                          "states_without_observed_fork": len(records) - len(scored)},
            "fork_scored": summarize(scored),
            "fork_scored_prompt_verified": summarize([r for r in scored if r["prompt_verified"]]),
            "selection_rule": "pairwise vote at each observed first action-token fork; tied vote picks lowest index",
            "limitation": "historical reused three-draw outcomes; no live guided decoding or untouched continuation holdout"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=SFT)
    parser.add_argument("--chunk-size", type=int, default=25)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.forks)
    if len(rows) != 704 or Counter(row["split"] for row in rows) != Counter(train=506, validation=74, branch124=124):
        raise ValueError("Frozen CPU pilot cohort changed")
    identity = {"forks_sha256": sha(args.forks), "model": str(args.model),
                "rows": len(rows), "scope": "offline feature replay and token-Q training only; zero API/browser/live decoding"}
    identity_path = args.output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Existing feature directory belongs to another input/model")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    extract(rows, args.output, args.model, args.chunk_size)
    features, embeddings = load_features(args.output, rows)
    metrics = train_and_evaluate(features, embeddings, args.output)
    metrics["identity"] = identity
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
