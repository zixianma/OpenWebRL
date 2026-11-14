"""Recipe 3 GPT-5.5 H = 4 4B LoRA decoder: every training-data-size ablation, recomputed per evaluation window from the episode
records, for ARM_DATA_SIZE_ABLATIONS.md. SFT 4B actor, N = 5, T 1.0 / top-p 0.9, 30 steps, all 1,045 tasks. Per window: overall
success (successes / episodes), valid-only success, counts, and paired differences in mean overall (equal-weight 3-benchmark mean)
and mean valid-only against the window's full-data LoRA and, where the window ran it, SFT 4B alone. Paired bootstrap over tasks
common to both arms (10,000 resamples, seed 20261029, benchmarks resampled independently, per-task success averaged over a
condition's runs). Refuses records still DEFERRED or with a dead-key judge message. Writes the adjacent JSON."""
import glob, json, math
from pathlib import Path
import numpy as np
R = "/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104"
EV = "/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations"
BENCH = ["online_mind2web", "webvoyager", "deepshop"]
B, SEED = 10000, 20261029
FULL, SFT = "arm5-9fadc8dc4655", "sft1"
# subset: (label, examples, panels, states)
DATA = {"full": ("Full data", 32974, 18786, 2112), "pps4": ("4 panels per state", 13532, 7417, 2112),
        "pps2": ("2 panels per state", 7476, 4035, 2112), "pps1": ("1 panel per state", 3910, 2112, 2112),
        "s1000": ("1,000 states, all panels", 15746, 8875, 1001), "s500": ("500 states, all panels", 7742, 4429, 500),
        "s250": ("250 states, all panels", 3858, 2234, 252), "s1000-pps1": ("1,000 states × 1 panel", 1905, 1001, 1001),
        "s250-pps1": ("250 states × 1 panel", 495, 252, 252)}
steps = lambda ex, ep: math.ceil(ex / 16) * ep
# window: (label, dates, [(key, subset, epochs, training note, [run dirs with {b}])])
runs = lambda e, slug, k=(1, 2, 3): [f"{R}/{e}/run{i}/{{b}}/{slug}" for i in k]
WINDOWS = {
    "screen": ("First screen", "November 4–6", [
        ("full", "full", 2, "LoRA", runs("eval-ctrl", FULL)), ("sft", None, 0, "", runs("eval-ctrl", SFT)),
        ("s250", "s250", 2, "LoRA", runs("eval", "arm5-d6c492ad4c40")), ("s500", "s500", 2, "LoRA", runs("eval", "arm5-125542b9f216", (1,))),
        ("s1000", "s1000", 2, "LoRA", runs("eval", "arm5-fc760b23e227")), ("pps1", "pps1", 2, "LoRA", runs("eval", "arm5-9f25a8bd0da7")),
        ("pps2", "pps2", 2, "LoRA", runs("eval", "arm5-4ac334f69f45", (1,))), ("pps4", "pps4", 2, "LoRA", runs("eval", "arm5-9117fa5a37bb"))]),
    "rerun": ("Same-window rerun and training controls", "November 8–9", [
        ("full", "full", 2, "LoRA", runs("eval-1108", FULL)), ("pps4", "pps4", 2, "LoRA", runs("eval-1108", "arm5-9117fa5a37bb")),
        ("pps1", "pps1", 2, "LoRA", runs("eval-1108", "arm5-9f25a8bd0da7")), ("s1000", "s1000", 2, "LoRA", runs("eval-1108", "arm5-fc760b23e227")),
        ("full-s1692", "full", 0.82, "LoRA stopped at 1,692 steps", runs("eval-1108", "arm5-5f368697014f")),
        ("ft-full", "full", 2, "full fine-tune", runs("eval-1108", "arm5-31cb7266dae5"))]),
    "s250": ("250 states", "November 10–11", [
        ("full", "full", 2, "LoRA", runs("eval-1110", FULL)), ("s250", "s250", 2, "LoRA", runs("eval-1110", "arm5-d6c492ad4c40"))]),
    "below2k": ("Below 2K panels", "November 11", [
        ("full", "full", 2, "LoRA", runs("eval-1111", FULL)), ("sft", None, 0, "", runs("eval-1111", SFT)),
        ("s1000-pps1", "s1000-pps1", 2, "LoRA", runs("eval-1111", "arm5-5d57cf0e0de4")),
        ("s250-pps1", "s250-pps1", 2, "LoRA", runs("eval-1111", "arm5-c5ec9b4862f6")),
        ("s1000-pps1-e4", "s1000-pps1", 4, "LoRA, evening", runs("eval-1111b", "arm5-a82f7bbded0c"))]),
    "s250ep": ("More epochs on 250 panels", "November 12", [
        ("full", "full", 2, "LoRA", runs("eval-s250ep", FULL, (1,))), ("sft", None, 0, "", runs("eval-s250ep", SFT, (1,))),
        ("s250-pps1-e4", "s250-pps1", 4, "LoRA", runs("eval-s250ep", "arm5-9c955fbdf7eb")),
        ("s250-pps1-e16", "s250-pps1", 16, "LoRA", runs("eval-s250ep", "arm5-88632a7097e6"))]),
}


def load(d):
    out = {}
    for f in glob.glob(f"{d}/*/collection/records/*.json"):
        x = json.load(open(f)); jt = str(x.get("judge_text") or "")
        assert not jt.startswith("DEFERRED") and jt.strip() != "All judge API retry attempts exhausted.", f
        v = str(x["valid"]) == "True"; out[x["task_id"]] = (v, v and float(x["reward"] or 0) >= 1)
    assert out, d
    return out


rng = np.random.default_rng(SEED)
res = {}
for w, (title, dates, conds) in WINDOWS.items():
    data = {k: {b: [load(r.format(b=b)) for r in dirs] for b in BENCH} for k, _, _, _, dirs in conds}
    boot, out = {}, {}
    for b in BENCH:
        tasks = sorted(set.intersection(*[set(r) for k in data for r in data[k][b]]))
        idx = rng.integers(0, len(tasks), (B, len(tasks)))
        for k, *_ in conds:
            rs = data[k][b]
            s = np.array([np.mean([r[t][1] for r in rs]) for t in tasks]); v = np.array([np.mean([r[t][0] for r in rs]) for t in tasks])
            out.setdefault(k, {})[b] = dict(tasks=len(tasks), episodes=len(rs) * len(tasks), valid=int(round(v.sum() * len(rs))),
                                            successes=int(round(s.sum() * len(rs))), overall=float(s.mean()), valid_only=float(s.sum() / v.sum()))
            boot[(k, b)] = (s[idx].mean(1), s[idx].sum(1) / v[idx].sum(1))
    for k, sub, ep, note, dirs in conds:
        r = out[k]; r.update(runs=len(dirs), mean_overall=float(np.mean([r[b]["overall"] for b in BENCH])),
                             mean_valid_only=float(np.mean([r[b]["valid_only"] for b in BENCH])))
        if sub:
            ex = DATA[sub][1]
            r.update(label=DATA[sub][0], subset=sub, examples=ex, panels=DATA[sub][2], states=DATA[sub][3], epochs=ep, training=note,
                     steps=1692 if k == "full-s1692" else steps(ex, ep))
        else:
            r.update(label="SFT 4B alone (1 sample)")
    for k, *_ in conds:
        for ref, name in (("full", "minus_full"), ("sft", "minus_sft")):
            if ref not in out or k == ref or (ref == "full" and k == "sft"):
                continue
            o = {}
            for j, m in ((0, "mean_overall_pp"), (1, "mean_valid_only_pp")):
                d = 100 * np.mean([boot[(k, b)][j] - boot[(ref, b)][j] for b in BENCH], 0)
                o[m] = [100 * (out[k][m[:-3]] - out[ref][m[:-3]]), *map(float, np.percentile(d, [2.5, 97.5]))]
            out[k][name] = o
    res[w] = dict(title=title, dates=dates, conditions=out)
    for k, r in out.items():
        print(w, k, r.get("steps"), f"{100*r['mean_overall']:.1f} {100*r['mean_valid_only']:.1f}",
              {n: [round(x, 1) for x in r[n]["mean_overall_pp"]] for n in ("minus_full", "minus_sft") if n in r}, flush=True)
Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
