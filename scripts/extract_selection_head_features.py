#!/usr/bin/env python3
"""Extract frozen OpenWebRL-SFT features for selection-head panels (one GPU).

Each base state's actor prompt + screenshot is encoded once; every candidate
from all of its labeled draws is teacher-forced after the cached prefix.
Output is resumable: one chunk file per group of base states in this shard.
Starts by checking cached-batched features against full forwards.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SFT = "/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT"
RELEASE = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/joint-data-inputs-0d83b48/openwebrl_actor")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default=SFT)
    ap.add_argument("--states", type=Path, default=RELEASE / "states_full.jsonl")
    ap.add_argument("--layers", default="9,18,27,36")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--chunk-states", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--verify-states", type=int, default=2)
    ap.add_argument("--limit-states", type=int)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from openwebrl.selection_head_features import candidate_tokens, candidate_tokens_from_ids, extract

    layers = tuple(int(x) for x in args.layers.split(","))
    panels = defaultdict(list)
    with open(args.panels) as stream:
        for line in stream:
            row = json.loads(line)
            panels[row["base_state_id"]].append(row)
    bases = sorted(panels)[args.shard::args.num_shards][:args.limit_states]
    wanted, prompts = set(bases), {}
    with open(args.states) as stream:
        for line in stream:
            row = json.loads(line)
            if row["state_id"] in wanted:
                prompts[row["state_id"]] = row["prompt_text"]

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="sdpa", local_files_only=True).eval()
    model.requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=True)

    def encode(base, reuse=True, batch=args.batch_size, rows=None):
        rows = rows or panels[base]
        image = Image.open(rows[0]["image"]).convert("RGB")
        prefix = processor(text=[prompts[base]], images=[image], return_tensors="pt").to(args.device)
        # Prefer the actor's exact generated ids when the panel saved them.
        cands = [candidate_tokens_from_ids(tokenizer, ids) if "candidate_token_ids" in row
                 else candidate_tokens(tokenizer, c)
                 for row in rows for c, ids in zip(row["candidates"], row.get("candidate_token_ids", row["candidates"]))]
        state, feats = extract(model, prefix, cands, layers, reuse_prefix=reuse, batch_size=batch,
                               pad_id=tokenizer.pad_token_id or 0)
        expected = rows[0].get("server_prompt_tokens")
        meta = {"prefix_tokens": prefix["input_ids"].shape[1], "server_prompt_tokens": expected,
                "prefix_tokens_match": expected is None or expected == prefix["input_ids"].shape[1],
                "image_grid_thw": prefix["image_grid_thw"].tolist(),
                "candidate_tokens": [len(ids) for ids, _ in cands]}
        # panels may hold any number of candidates (five in the original release panels)
        return state, feats.split([len(row["candidates"]) for row in rows]), meta

    checks = []
    for base in bases[:args.verify_states]:
        rows = panels[base][:1]
        _, fast, _ = encode(base, rows=rows)
        _, slow, _ = encode(base, reuse=False, rows=rows)
        diff = (fast[0].float() - slow[0].float()).abs()
        rel = (diff.norm() / slow[0].float().norm()).item()
        checks.append({"base_state_id": base, "max_abs": diff.max().item(), "relative": rel})
        if rel > 2e-2:
            raise SystemExit(f"Cached features differ from full forward: {checks[-1]}")
    (args.output / f"verify-shard{args.shard}.json").write_text(json.dumps(checks, indent=1))

    for chunk_index in range(0, len(bases), args.chunk_states):
        path = args.output / f"shard{args.shard:02d}-chunk{chunk_index // args.chunk_states:04d}.pt"
        if path.exists():
            continue
        started, out = time.time(), {"layers": layers, "states": {}, "panels": {}, "meta": {}}
        for base in bases[chunk_index:chunk_index + args.chunk_states]:
            state, feats, meta = encode(base)
            out["states"][base] = state.to(torch.float16).cpu()
            for row, panel in zip(panels[base], feats):
                out["panels"][row["state_id"]] = panel.to(torch.float16).cpu()
            out["meta"][base] = meta
        tmp = path.with_suffix(".tmp")
        torch.save(out, tmp)
        tmp.rename(path)
        print(json.dumps({"chunk": path.name, "states": len(out["states"]), "panels": len(out["panels"]),
                          "seconds": round(time.time() - started, 1)}), flush=True)
    (args.output / f"shard{args.shard:02d}.done").write_text(json.dumps({"bases": len(bases)}))


if __name__ == "__main__":
    main()
