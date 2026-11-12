"""Selection compute and latency per decision step: Recipe 3 GPT-5.5 H = 4 heads of every size vs the 4B LoRA decoder (ADM-4B,
full data). SFT 4B actor, N = 5, T 1.0 / top-p 0.9, 30 steps, all three benchmarks, 1 H200 per run.

Heads: every run of the head-size sweep (eval-scale, eval-seed, eval-sizes3; training seed 0 heads only). Decoder and SFT 4B
alone: eval-1111 (3 runs). Data-scaling LoRAs: each eval window with its same-window full-data reference (model time and
end-to-end latency; actor FLOPs are the same 4B actor). LoRA training time: LLaMA-Factory train_runtime (4 H200, merge excluded). Latency from each step's decision trace: selection_seconds (selector only), generation_seconds (actor
proposals incl. the head's hidden-state export) and decision_seconds (model time, browser excluded). End-to-end episode latency:
each record's elapsed_seconds (wall clock from episode start to finish: browser, actor and selector; the judge ran afterwards).

FLOPs (forward, 2 x params x tokens), per step:
  head selector    = logged telemetry flops = actor follow-up decode for each candidate without the end token in cache
                     (2 x 4.44e9 x (uncached + sampled tokens)) + head (2 x head params x 5 candidates);
  decoder selector = 2 x 4.44e9 x (uncached selection-prompt + generated tokens), from the step's selector telemetry;
  actor            = 2 x 4.44e9 x (uncached prompt + completion tokens) of the step's actor requests (run 1 only).
Writes the adjacent JSON.
"""
import glob, json, os, statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
R = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104"
P4B = 4022468096 + 415347712
BENCH = ("online_mind2web", "webvoyager", "deepshop")
HEADS = ("eval-scale", "eval-seed", "eval-sizes3")
L = R + "/lora"
# key: (label, [(eval dir, condition slug)], actor FLOPs from run 1 of the first eval dir?, LoRA training dir or None)
CONDS = {h: (lab, [(e, h) for e in HEADS], True, None) for h, lab in (
    ("head5-5fbddd8b2031", "d128-L2 head (4.4M)"), ("head5-fa31796f1fe0", "d256-L2 head (9.6M)"),
    ("head5-91e7a5a67417", "d512-L2 head (22.6M, default)"), ("head5-938477dffad2", "d2048-L4 head (272.8M)"),
    ("head5-7bbad4567d44", "d4096-L8 head (1.8B)"))}
CONDS.update({"lora-full": ("4B LoRA decoder (ADM-4B), full data", [("eval-1111", "arm5-9fadc8dc4655")], True, None),
              "sft1": ("SFT 4B alone (1 sample)", [("eval-1111", "sft1")], True, None)})
# Data-scaling LoRAs, each with its same-window full-data reference (model time and end-to-end only).
for w, rows in {"eval-1108": [("arm5-9fadc8dc4655", "lora"), ("arm5-9117fa5a37bb", "pps4"), ("arm5-9f25a8bd0da7", "pps1"),
                              ("arm5-fc760b23e227", "s1000"), ("arm5-5f368697014f", "full-s1692"), ("arm5-31cb7266dae5", "ft-full")],
                "eval-1110": [("arm5-9fadc8dc4655", "lora"), ("arm5-d6c492ad4c40", "s250")],
                "eval-1111": [("arm5-5d57cf0e0de4", "s1000-pps1"), ("arm5-c5ec9b4862f6", "s250-pps1")],
                "eval-1111b": [("arm5-a82f7bbded0c", "s1000-pps1-e4")],
                "eval-s250ep": [("arm5-9fadc8dc4655", "lora"), ("arm5-9c955fbdf7eb", "s250-pps1-e4"), ("arm5-88632a7097e6", "s250-pps1-e16"),
                                ("sft1", None)]}.items():
    for slug, sub in rows:
        CONDS[f"{w}:{sub or slug}"] = (f"{sub or 'SFT 4B alone'} ({w})", [(w, slug)], False, sub and (L + "/ft-full/merged" if sub == "ft-full" else
                                       "/gpfs/scrubbed/zixianma/openwebrl-runtime/rl-actor-recollect-20261016/p16-k3-20261031/lora/lora" if sub == "lora" else f"{L}/{sub}/adapter"))


def episode(rec, actor):
    r = json.load(open(rec)); out = dict(steps=0, sel=[], gen=[], dec=[], sel_flops=0.0, head_flops=0.0, actor_flops=0.0, actor_requests=0, elapsed=r.get("elapsed_seconds"))
    for tf in r.get("trace_files") or []:
        q = Path(tf["path"] if isinstance(tf, dict) else tf)
        if not q.exists(): continue
        for line in open(q):
            x = json.loads(line); t = x["timing"]; s = x.get("selector_telemetry") or {}
            out["steps"] += 1; out["sel"].append(t.get("selection_seconds") or 0.0); out["gen"].append(t["generation_seconds"]); out["dec"].append(t["decision_seconds"])
            if "flops" in s:
                out["sel_flops"] += s["flops"]; out["head_flops"] += s["flops"] - 2 * P4B * (s["input_tokens"] + s["output_tokens"])
            elif s:
                out["sel_flops"] += 2 * P4B * (s["input_tokens"] - (s.get("cached_tokens") or 0) + s["output_tokens"])
        if actor:
            for f in (q.parent.parent / "actor-requests").glob("*.json"):
                m = json.load(open(f)).get("meta") or {}
                out["actor_flops"] += 2 * P4B * ((m.get("prompt_tokens") or 0) - (m.get("cached_tokens") or 0) + (m.get("completion_tokens") or 0)); out["actor_requests"] += 1
    return out


q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
res = {}
with ThreadPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))) as pool:
    for c, (label, evals, actor, train) in CONDS.items():
        recs = [(f, actor and "/run1/" in f and e == evals[0][0]) for e, slug in evals for b in BENCH
                for f in sorted(glob.glob(f"{R}/{e}/run*/{b}/{slug}/{b}/collection/records/*.json"))]
        eps = list(pool.map(lambda a: episode(*a), recs))
        a1 = [e for e, (_, act) in zip(eps, recs) if act]
        sel = [x for e in eps for x in e["sel"]]; gen = [x for e in eps for x in e["gen"]]; dec = [x for e in eps for x in e["dec"]]
        el = [e["elapsed"] for e in eps if e["elapsed"] is not None]
        n, steps, s1 = len(eps), sum(e["steps"] for e in eps), sum(e["steps"] for e in a1)
        actor_step = sum(e["actor_flops"] for e in a1) / s1 / 1e15 if s1 else None
        sel_step = sum(e["sel_flops"] for e in eps) / steps / 1e15
        tr = Path(str(train)) / "all_results.json"
        tr = json.load(open(tr)) if train and tr.exists() else {}
        res[c] = dict(label=label, evals=[list(x) for x in evals], train_dir=train, train_runtime_seconds=tr.get("train_runtime"), train_epochs=tr.get("epoch"), episodes=n, steps=steps, actor_cost_episodes=len(a1), actor_cost_steps=s1,
                      actor_requests_per_step=sum(e["actor_requests"] for e in a1) / s1 if s1 else None, steps_per_episode=steps / n,
                      selector_pflop_per_step=sel_step, head_only_pflop_per_step=sum(e["head_flops"] for e in eps) / steps / 1e15,
                      actor_pflop_per_step=actor_step, decision_pflop_per_step=actor_step and actor_step + sel_step,
                      selection_seconds_mean=st.mean(sel), selection_seconds_median=st.median(sel), selection_seconds_p90=q(sel, .9),
                      generation_seconds_mean=st.mean(gen), decision_seconds_mean=st.mean(dec), decision_seconds_per_episode=sum(dec) / n,
                      episodes_timed=len(el), episode_seconds_mean=st.mean(el), episode_seconds_median=st.median(el), episode_seconds_p90=q(el, .9))
        print(label, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in res[c].items() if k not in ("label", "evals")}, flush=True)
Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
