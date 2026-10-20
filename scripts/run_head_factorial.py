#!/usr/bin/env python3
"""Selection-head factorial: interaction (joint / independent) x scorer (transformer / mlp) x loss (softce / bce) x
data scale, with everything else fixed (features, layers, pools, labels, optimizer, epochs).

Offline protocol: 5-fold task-grouped cross-validation on `targets.json`. For each fold, a head trains on the other
folds (optionally a nested fraction of their tasks) and picks one candidate on every test state; with all folds every
state is scored once per seed. A pick scores the chosen candidate's held-out success (draws 3-7, valid only); draws
0-2 are the only outcomes any label may use. Reported per cell: mean success over seeds and states, minus uniform, and
paired contrasts along each dimension (task-clustered bootstrap, 10,000 resamples).

  run_head_factorial.py plan  --out O --panels P --features F --targets T [--scales 0.125,0.25,0.5,1] [--seeds 0,1,2]
  run_head_factorial.py sbatch --out O [--parallel 16]      (CPU-only Slurm array; no GPUs)
  run_head_factorial.py score  --out O
"""
import argparse
import itertools
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = "/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python"
VARIANTS = [dict(interaction=i, scorer=s, loss=l)
            for i, s, l in itertools.product(("joint", "independent"), ("transformer", "mlp"), ("softce", "bce"))]


def tag(v):
    return f"{v['interaction']}-{v['scorer']}-{v['loss']}"


def cmd_plan(a):
    out = Path(a.out).resolve()
    if (out / "grid.json").exists():
        raise SystemExit("grid already planned; it is frozen")
    folds = [int(x) for x in a.folds.split(",")]
    scales = [float(x) for x in a.scales.split(",")]
    runs = []
    for v, scale, fold in itertools.product(VARIANTS, scales, folds):
        name = f"{tag(v)}/scale{scale:g}/fold{fold}"
        runs.append(dict(name=name, variant=v, scale=scale, fold=fold, output=str(out / "runs" / name),
                         argv=[PY, str(REPO / "scripts/train_supervision_ablation.py"), "--panels", a.panels, "--features",
                               a.features, "--targets", a.targets, "--output", str(out / "runs" / name), "--sources",
                               a.source, "--test-fold", str(fold), "--seeds", a.seeds, "--train-frac", str(scale),
                               "--layers", a.layers, "--interaction", v["interaction"], "--scorer", v["scorer"],
                               "--loss", v["loss"], "--device", "cpu", "--save-heads"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid.json").write_text(json.dumps(dict(
        panels=a.panels, features=a.features, targets=a.targets, source=a.source, layers=a.layers, seeds=a.seeds,
        scales=scales, folds=folds, variants=VARIANTS, runs=runs,
        fixed="features, layers, pools, labels, d_model 512, depth 2, dropout 0.1, AdamW 3e-4 / wd 0.05, batch 64, 20 epochs,"
              " one-cycle; nested task fractions (frac seed 20261017)"), indent=1))
    print(json.dumps(dict(runs=len(runs), variants=len(VARIANTS), scales=scales, folds=folds, seeds=a.seeds)))


def cmd_sbatch(a):
    out = Path(a.out).resolve()
    grid = json.loads((out / "grid.json").read_text())
    n = len(grid["runs"])
    script = out / "factorial.sbatch"
    script.write_text("\n".join([
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=head-factorial", "#SBATCH --account=vlm", "#SBATCH --partition=gpu-h200", "#SBATCH --qos=normal",
        "#SBATCH --cpus-per-task=4", "#SBATCH --mem=32G", "#SBATCH --time=02:00:00",
        f"#SBATCH --array=0-{n - 1}%{a.parallel}", f"#SBATCH --output={out}/logs/run-%A_%a.log",
        "# CPU only (no GPUs): one factorial cell (variant x scale x fold, all seeds) per array task.",
        "set -euo pipefail", f"cd {REPO}",
        f"export PYTHONPATH={REPO}:{REPO}/scripts OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1",
        f"{PY} {REPO}/scripts/run_head_factorial.py run-one --out {out} --index $SLURM_ARRAY_TASK_ID", ""]))
    (out / "logs").mkdir(exist_ok=True)
    print(f"wrote {script} ({n} array tasks, {a.parallel} at a time)")


def cmd_run_one(a):
    import subprocess
    out = Path(a.out).resolve()
    run = json.loads((out / "grid.json").read_text())["runs"][a.index]
    if (Path(run["output"]) / "student-picks.json").exists():
        print("already done:", run["name"]); return
    subprocess.run(run["argv"], check=True)


def cmd_score(a):
    import numpy as np
    sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
    from adapt_selection_head_outcomes import LEDGERS
    from selection_outcome_experiments import cluster_boot
    out = Path(a.out).resolve()
    grid = json.loads((out / "grid.json").read_text())
    targets = json.loads(Path(grid["targets"]).read_text())["states"]
    panels = {json.loads(l)["state_id"]: json.loads(l) for l in open(grid["panels"])}
    roots = {str(x) for x in LEDGERS} | {p["image"].split("/attempts/")[0] for p in panels.values()}
    slot = defaultdict(dict)
    for root in roots:
        f = Path(root) / "state.sqlite3"
        if f.exists():
            for (raw,) in sqlite3.connect(f"file:{f}?mode=ro", uri=True).execute("SELECT result_json FROM attempts WHERE status='finished'"):
                x = json.loads(raw)
                if x.get("mode") == "branch" and x.get("released") and x["state"] in targets:
                    v = bool(x["valid"])
                    slot[x["state"]].setdefault((x["candidate"], x["repeat"]), (v, v and (x.get("reward") or 0) >= 1))

    def held(s):
        rates = []
        for i in range(5):
            cells = [slot[s][i, k][1] for k in range(3, 8) if (i, k) in slot[s] and slot[s][i, k][0]]
            if not cells:
                return None
            rates.append(float(np.mean(cells)))
        return np.array(rates)
    H = {s: held(s) for s in targets}
    cells = defaultdict(dict)                                   # (variant, scale) -> state -> mean success over seeds
    missing = []
    for run in grid["runs"]:
        f = Path(run["output"]) / "student-picks.json"
        if not f.exists():
            missing.append(run["name"]); continue
        picks = json.loads(f.read_text())["picks"]
        key = (tag(run["variant"]), run["scale"])
        for s in next(iter(picks.values())):
            if H.get(s) is None:
                continue
            cells[key][s] = float(np.mean([H[s][p[s]] for p in picks.values()]))
    common = set.intersection(*(set(v) for v in cells.values())) if cells else set()
    st = sorted(common); g = [targets[s]["task"] for s in st]
    uni = {s: float(H[s].mean()) for s in st}
    table = {f"{k[0]}@{k[1]:g}": dict(success=round(100 * np.mean([v[s] for s in st]), 2),
                                      minus_uniform=cluster_boot([v[s] - uni[s] for s in st], g)) for k, v in sorted(cells.items())}
    contrasts = {}
    for scale in grid["scales"]:
        for dim, (x, y) in (("interaction", ("joint", "independent")), ("scorer", ("transformer", "mlp")), ("loss", ("softce", "bce"))):
            diffs = defaultdict(float)
            for v in VARIANTS:
                if v[dim] != x:
                    continue
                w = dict(v, **{dim: y})
                a_, b_ = cells.get((tag(v), scale)), cells.get((tag(w), scale))
                if a_ and b_:
                    for s in st:
                        diffs[s] += (a_[s] - b_[s]) / 4               # averaged over the other two dimensions' 4 cells
            if diffs:
                contrasts[f"{dim}: {x} - {y} @ scale {scale:g}"] = cluster_boot([diffs[s] for s in st], g)
    res = dict(states=len(st), tasks=len(set(g)), uniform=round(100 * np.mean(list(uni.values())), 2),
               cells=table, contrasts=contrasts, missing_runs=missing)
    (out / "factorial-results.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--out", required=True); p.add_argument("--panels", required=True)
    p.add_argument("--features", required=True); p.add_argument("--targets", required=True)
    p.add_argument("--source", default="teacher_k1"); p.add_argument("--layers", default="18,27,36")
    p.add_argument("--scales", default="0.125,0.25,0.5,1"); p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--seeds", default="0,1,2")
    s = sub.add_parser("sbatch"); s.add_argument("--out", required=True); s.add_argument("--parallel", type=int, default=16)
    r = sub.add_parser("run-one"); r.add_argument("--out", required=True); r.add_argument("--index", type=int, required=True)
    c = sub.add_parser("score"); c.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(plan=cmd_plan, sbatch=cmd_sbatch, **{"run-one": cmd_run_one}, score=cmd_score)[a.cmd](a)


if __name__ == "__main__":
    main()
