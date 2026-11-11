"""Selection-stage latency: Recipe 3 GPT-5.5 H = 4 22.6M head (ADM-22M, default) vs 4B LoRA decoder (ADM-4B, full data), November 11.

SFT 4B actor, N = 5, T 1.0 / top-p 0.9, 30 steps, all three benchmarks, 1 H200 per run. Head: eval-scale (head-size sweep,
2 runs); decoder and SFT 4B alone: eval-1111 (data sweep, 3 runs; plans still running are read as far as they got). Per-step
timing from each episode's decision trace: generation_seconds (actor proposals, including the head's hidden-state export),
selection_seconds (selector only) and decision_seconds (their sum). Browser time is excluded. Writes the adjacent JSON.
"""
import glob, json, statistics as st
from pathlib import Path
R = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104"
CONDS = {"22.6M head (ADM-22M)": ("eval-scale", "head5-91e7a5a67417"), "4B LoRA decoder (ADM-4B, full data)": ("eval-1111", "arm5-9fadc8dc4655"),
         "SFT 4B alone (1 sample)": ("eval-1111", "sft1")}
out = {}
for name, (e, c) in CONDS.items():
    sel, gen, dec, steps_per_ep = [], [], [], []
    for b in ("online_mind2web", "webvoyager", "deepshop"):
        for rec in glob.glob(f"{R}/{e}/run*/{b}/{c}/{b}/collection/records/*.json"):
            r = json.load(open(rec)); n = 0
            for tf in r.get("trace_files") or []:
                q = Path(tf["path"] if isinstance(tf, dict) else tf)
                if not q.exists(): continue
                for line in open(q):
                    t = json.loads(line)["timing"]; sel.append(t.get("selection_seconds") or 0.0); gen.append(t["generation_seconds"]); dec.append(t["decision_seconds"]); n += 1
            steps_per_ep.append(n)
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
    out[name] = dict(episodes=len(steps_per_ep), steps=len(sel), sel_mean=st.mean(sel), sel_median=st.median(sel), sel_p90=q(sel, .9),
                     gen_mean=st.mean(gen), dec_mean=st.mean(dec), steps_per_episode=st.mean(steps_per_ep),
                     sel_per_episode=sum(sel) / len(steps_per_ep), dec_per_episode=sum(dec) / len(steps_per_ep))
for k, v in out.items(): print(k, {a: round(b, 4) for a, b in v.items()})
json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
