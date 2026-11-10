#!/usr/bin/env python3
"""Offline score of SelectionARM-style models on the SFT-action test states (1 GPU).

Same test states, held-out draws (3-7) and metrics as scripts/train_head_round.py, so LoRA selectors line up with the
selection heads. Each panel's prompt is the serving prompt (build_selection_sft_ours: selection_messages through the
pinned selection_prompt.py, qwen3_vl rendering, processor at 65,536-262,144 pixels). The selector's distribution over
candidates is read from the next-token logits of the digits 1..n after one of two response prefixes:
  train:   <think>\\n\\n</think>\\n\\n{"selection":   (the trained response, LLaMA-Factory qwen3_vl empty CoT)
  serving: <think>\\n{"selection":                    (what the sglang server decodes under the JSON schema)
The pick is the argmax. Views: distinct (one candidate per action) in --orderings seeded orders per state; natural
(the first --natural-panels of the natural build's 20 seeded five-proposal panels per test state).

  score_selection_lora_offline.py --model M [--model M2 ...] --distinct D/panels.jsonl --natural N/panels.jsonl
      --data SFT_ACTIONS_DIR --output O
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_selection_sft_ours import SOURCE_ROOT, load_selection_builder, prompt_messages, read_jsonl, render_qwen3_vl, selector_payload  # noqa: E402
from selection_outcome_experiments import cluster_boot  # noqa: E402
from train_head_round import held_rate, load_actions  # noqa: E402

PREFIXES = {"train": '<think>\n\n</think>\n\n{"selection": ', "serving": '<think>\n{"selection": '}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="name=path (merged model directory)")
    ap.add_argument("--distinct", type=Path, required=True)
    ap.add_argument("--natural", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--orderings", type=int, default=5)
    ap.add_argument("--natural-panels", type=int, default=5)
    ap.add_argument("--limit", type=int, help="first N test states (smoke)")
    ap.add_argument("--dry-run", action="store_true", help="CPU check: panels, prompts and digit tokens; no model")
    args = ap.parse_args()
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    rec = {x["state_id"]: x for x in load_actions(args.data)}
    rates = {}
    for sid, x in rec.items():
        r = [held_rate(a, range(3, 8)) for a in x["actions"]]
        if x["split"] == "test" and None not in r:
            rates[sid] = r
    rng = random.Random(20261021)
    panels = []                                   # (view, state, panel dict with candidates + actions)
    for p in read_jsonl(args.distinct):
        if p["split"] == "test" and p["base_state_id"] in rates:
            for k in range(args.orderings):
                order = list(range(len(p["candidates"])))
                if k:
                    rng.shuffle(order)
                panels.append(("distinct", p["base_state_id"], dict(p, candidates=[p["candidates"][i] for i in order],
                                                                    actions=[p["actions"][i] for i in order])))
    seen = defaultdict(int)
    for p in read_jsonl(args.natural):
        s = p["base_state_id"]
        if p["split"] == "test" and s in rates and seen[s] < args.natural_panels:
            seen[s] += 1; panels.append(("natural", s, p))
    states = sorted({s for _, s, _ in panels})
    if args.limit:
        states = states[:args.limit]; panels = [x for x in panels if x[1] in set(states)]

    builder = load_selection_builder(SOURCE_ROOT)
    images = {}
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        started = time.time()
        processor = AutoProcessor.from_pretrained(path)
        processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 262144}
        processor.image_processor.min_pixels, processor.image_processor.max_pixels = 65536, 262144
        tok = processor.tokenizer
        digits = []
        for n in range(1, 6):
            for pre in PREFIXES.values():
                a, b = tok(pre, add_special_tokens=False).input_ids, tok(pre + str(n), add_special_tokens=False).input_ids
                if b[:-1] != a:
                    raise ValueError(f"digit {n} does not tokenize as one token after the prefix")
            digits.append(b[-1])
        if args.dry_run:
            from PIL import Image
            for view, sid, p in panels[:3] + panels[-3:]:
                text = render_qwen3_vl(prompt_messages(builder, selector_payload(p, Path(p["image"]).read_bytes()))) + PREFIXES["serving"]
                n = processor(text=[text], images=[Image.open(p["image"]).convert("RGB")], return_tensors="pt").input_ids.shape[1]
                print(view, p["state_id"], len(p["candidates"]), p["actions"], n, "tokens")
            print(json.dumps(dict(model=name, digits=digits, panels=len(panels), states=len(states),
                                  views={v: sum(x[0] == v for x in panels) for v in ("distinct", "natural")})))
            continue
        model = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        rows = []
        from PIL import Image
        for view, sid, p in panels:
            if p["image"] not in images:
                images[p["image"]] = Path(p["image"]).read_bytes()
            messages = prompt_messages(builder, selector_payload(p, images[p["image"]]))
            image = Image.open(p["image"]).convert("RGB")
            row = dict(view=view, state=sid, panel=p["state_id"], actions=p["actions"])
            for which, prefix in PREFIXES.items():
                text = render_qwen3_vl(messages) + prefix
                inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
                with torch.no_grad():
                    logits = model(**inputs).logits[0, -1].float()
                lp = torch.log_softmax(logits, -1)
                cand = lp[digits[:len(p["candidates"])]]
                row[which] = dict(pick=int(cand.argmax()), digit_mass=float(cand.exp().sum()), probs=cand.softmax(-1).tolist())
            rows.append(row)
        del model; torch.cuda.empty_cache()
        with open(args.output / f"{name}-panels.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        res = dict(model=path, test_states=len(states), test_tasks=len({rec[s]["task_id"] for s in states}),
                   seconds=round(time.time() - started, 1), panels=len(rows))
        g = [rec[s]["task_id"] for s in states]
        uni = [sum(rates[s]) / len(rates[s]) for s in states]
        freq = {s: [len(a["sft_draws"]) for a in rec[s]["actions"]] for s in states}
        sft = [sum(f * r for f, r in zip(freq[s], rates[s])) / sum(freq[s]) for s in states]
        mean = lambda v: round(100 * sum(v) / len(v), 2)
        res["uniform"], res["sft_sample"], res["oracle"] = mean(uni), mean(sft), mean([max(rates[s]) for s in states])
        for which in PREFIXES:
            for view in ("distinct", "natural"):
                got = defaultdict(list)
                for r in rows:
                    if r["view"] == view:
                        got[r["state"]].append(rates[r["state"]][r["actions"][r[which]["pick"]]])
                v = [sum(got[s]) / len(got[s]) for s in states]
                res[f"{which}_{view}"] = dict(success=mean(v), minus_uniform=cluster_boot([a - b for a, b in zip(v, uni)], g),
                                              minus_sft_sample=cluster_boot([a - b for a, b in zip(v, sft)], g),
                                              digit_mass=round(sum(r[which]["digit_mass"] for r in rows if r["view"] == view) / max(1, sum(r["view"] == view for r in rows)), 4))
        results[name] = res
        print(json.dumps({name: res}), flush=True)
    (args.output / "scores.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
