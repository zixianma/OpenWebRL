"""RL 8B actor (OpenWebRL/OpenWebRL-8B @ 824a54f, 4,096 response tokens): compute and latency per decision step and episode, alone and
with 4B LoRA selectors, local browsers (November 4-5, evaluations/sameday-rl8b-20261104, 3 runs) and Browser Use stealth browsers
with US proxy (November 12-13, recipe3-ablations-20261104/stealth/rl8b-proxy-1112, 1 run). All 1,045 tasks, 1 H200 per run.

FLOPs (forward, 2 x params x tokens), per step:
  actor    = 2 x 8.767e9 x (uncached prompt + completion tokens) of the step's actor requests (run 1 only);
  selector = 2 x 4.44e9 x (uncached selection-prompt + generated tokens), from the step's selector telemetry.
Latency from each step's decision trace (selection_seconds, decision_seconds; browser excluded); end-to-end = each episode record's
elapsed_seconds (wall clock incl. browser; judge excluded). Writes the adjacent JSON."""
import glob, json, os, statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
P4B, P8B = 4022468096 + 415347712, 8767123696
BENCH = ("online_mind2web", "webvoyager", "deepshop")
L = "/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/sameday-rl8b-20261104"
S = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/stealth/rl8b-proxy-1112"
CONDS = {"local:alone": ("Local: RL 8B alone (1 sample)", f"{L}/rl8b", "sft1"),
         "local:h4": ("Local: RL 8B + 4B LoRA, Recipe 3 GPT-5.5 H = 4", f"{L}/rl8b-lora", "arm5-9fadc8dc4655"),
         "local:pre": ("Local: RL 8B + 4B LoRA, Recipe 3 panels GPT-5.5 pre", f"{L}/rl8b-lora", "arm5-80ee47643be2"),
         "stealth:alone": ("Stealth: RL 8B alone (1 sample)", S, "sft1"),
         "stealth:adm": ("Stealth: RL 8B + default ADM (R3+WA 4B LoRA)", S, "arm5-d470d7e52a38")}


def episode(rec, actor):
    r = json.load(open(rec)); o = dict(steps=0, sel=[], dec=[], sel_flops=0.0, actor_flops=0.0, elapsed=r.get("elapsed_seconds"))
    for tf in r.get("trace_files") or []:
        q = Path(tf["path"] if isinstance(tf, dict) else tf)
        if not q.exists(): continue
        for line in open(q):
            x = json.loads(line); t = x["timing"]; s = x.get("selector_telemetry") or {}
            o["steps"] += 1; o["sel"].append(t.get("selection_seconds") or 0.0); o["dec"].append(t["decision_seconds"])
            if s: o["sel_flops"] += 2 * P4B * (s.get("input_tokens", 0) - (s.get("cached_tokens") or 0) + s.get("output_tokens", 0))
        if actor:
            for f in (q.parent.parent / "actor-requests").glob("*.json"):
                m = json.load(open(f)).get("meta") or {}
                o["actor_flops"] += 2 * P8B * ((m.get("prompt_tokens") or 0) - (m.get("cached_tokens") or 0) + (m.get("completion_tokens") or 0))
    return o


q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
res = {}
with ThreadPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))) as pool:
    for k, (label, root, slug) in CONDS.items():
        recs = [(f, "/run1/" in f) for b in BENCH for f in sorted(glob.glob(f"{root}/run*/{b}/{slug}/{b}/collection/records/*.json"))]
        eps = list(pool.map(lambda a: episode(*a), recs)); a1 = [e for e, (_, act) in zip(eps, recs) if act]
        sel = [x for e in eps for x in e["sel"]]; dec = [x for e in eps for x in e["dec"]]; el = [e["elapsed"] for e in eps if e["elapsed"] is not None]
        n, steps, s1 = len(eps), sum(e["steps"] for e in eps), sum(e["steps"] for e in a1)
        act, sf = sum(e["actor_flops"] for e in a1) / s1 / 1e12, sum(e["sel_flops"] for e in eps) / steps / 1e12
        res[k] = dict(label=label, root=root, slug=slug, episodes=n, runs=n // 1045, steps=steps, steps_per_episode=steps / n,
                      actor_tflop_per_step=act, selector_tflop_per_step=sf, pflop_per_episode=(act + sf) * steps / n / 1e3,
                      selection_ms_mean=1e3 * st.mean(sel), selection_ms_median=1e3 * st.median(sel), selection_ms_p90=1e3 * q(sel, .9),
                      decision_seconds_per_episode=sum(dec) / n, episode_seconds_mean=st.mean(el), episode_seconds_median=st.median(el),
                      episode_seconds_p90=q(el, .9))
        print(label, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in res[k].items() if a not in ("label", "root", "slug")}, flush=True)
Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
