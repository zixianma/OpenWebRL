#!/usr/bin/env python3
"""Compact iteration-90 loop-failure figure; task-bootstrap intervals for fixed checkpoints.

Default: render from public aggregates. To recompute intervals from local saved records:
  python plot_rl_loops_iter90.py --compact-dir /path/to/compact/jsonl
No task records or identifiers are written to the public artifacts.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[2]
SOURCE = HERE / "rl-action-loops-20261103.json"
INTERVALS = HERE / "rl-action-loops-iter90-ci-20261104.json"
PURPLE = "#6250BE"
RUNS = (("outcome48", "Outcome-only\n(48 groups)", "#9A9A9A"),
        ("outcome56", "Outcome-only\n(56 groups)", "#666666"),
        ("gate-b", "Outcome + ADM\nrewards (ours)", PURPLE))
DRAWS, SEED = 20000, 20261104


def is_loop_failure(row):
    """Exact repeat_loop branch of analyze_rl_eval_failures.classify(row, 30).

    Success and invalidity take precedence; without done, termination categories take
    precedence. The later failure categories cannot affect this binary indicator.
    """
    if row["success"] or not row["valid"]:
        return False
    calls = [step["call"] for step in row.get("steps") or [] if step.get("call")]
    if not any(call.get("name") == "done" for call in calls):
        return False
    counts = Counter(json.dumps(call, sort_keys=True) for call in calls
                     if call.get("name") not in ("done", "wait"))
    return bool(counts and max(counts.values()) >= 3)


def compute_intervals(compact_dir):
    source = json.loads(SOURCE.read_text())
    task_order, matrices, provenance = None, [], []
    for key, _, _ in RUNS:
        repeats = []
        for run in (1, 2, 3):
            paths = sorted(compact_dir.glob(f"repeat-iter90-{key}-r{run}-*.jsonl"))
            assert len(paths) == 1, (key, run, len(paths))
            path = paths[0]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            by_task = {str(row["task_id"]): row for row in rows}
            assert len(rows) == len(by_task) == 300
            assert any(row.get("steps") for row in rows), "Trajectories required"
            tasks = sorted(by_task)
            if task_order is None:
                task_order = tasks
            assert tasks == task_order, "Every method/repeat must use the same tasks"
            flags = np.array([is_loop_failure(by_task[t]) for t in tasks], dtype=float)
            assert flags.sum() == source["iteration90"][key][run - 1]["loop_failures"]
            repeats.append(flags)
            provenance.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        matrices.append(repeats)
    # Resample whole task clusters, retaining all three repeats and all methods.
    per_task = np.asarray(matrices).mean(axis=1)
    rng = np.random.default_rng(SEED)
    samples = []
    for start in range(0, DRAWS, 500):
        indices = rng.integers(0, 300, size=(min(500, DRAWS - start), 300))
        samples.append(300 * per_task[:, indices].mean(axis=2))
    bootstrap = np.concatenate(samples, axis=1)
    results = {}
    for j, (key, _, _) in enumerate(RUNS):
        counts = [int(v.sum()) for v in matrices[j]]
        results[key] = dict(repeat_counts=counts, mean_per300=float(np.mean(counts)),
                            ci95_per300=np.percentile(bootstrap[j], [2.5, 97.5]).tolist())
    result = dict(
        metric="Mean loop failures per 300 planned tasks, averaged over three evaluation repeats",
        definition="First-match repeat_loop failure category; invalids and other failure categories remain in the 300-task denominator",
        uncertainty=dict(method="Pointwise percentile bootstrap of task clusters, preserving all repeats and methods",
                         tasks=300, repeats=3, draws=DRAWS, seed=SEED,
                         scope="Conditional on these fixed checkpoints and three evaluation windows; excludes training-seed and evaluation-window uncertainty"),
        aggregate_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        local_source_hashes=provenance, methods=results)
    INTERVALS.write_text(json.dumps(result, indent=2) + "\n")
    return result


def render(intervals):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "rl-action-loops-iter90-compact"})
    fig, ax = plt.subplots(figsize=(3.5, 1.85))
    for j, (key, _, color) in enumerate(RUNS):
        y = 2 - j
        row = intervals["methods"][key]
        mean = row["mean_per300"]
        lo, hi = row["ci95_per300"]
        ax.errorbar(mean, y, xerr=[[mean-lo], [hi-mean]], fmt="o", color=color,
                    ecolor=color, ms=5.5, lw=1.2, capsize=2.5, markeredgewidth=1,
                    zorder=3)
        ax.annotate(f"{mean:.1f}", (hi, y), xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", color=color, fontsize=8)
    ax.set_yticks([2, 1, 0], [label for _, label, _ in RUNS], fontsize=7.5)
    ax.get_yticklabels()[-1].set_color(PURPLE)
    ax.get_yticklabels()[-1].set_fontweight("medium")
    ax.set_xlim(0, 85)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("Loop failures per 300 tasks", fontsize=8, labelpad=5)
    ax.tick_params(axis="x", labelsize=7, length=3, color="#666666")
    ax.tick_params(axis="y", length=0, pad=7)
    ax.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#666666")
    ax.spines["bottom"].set_linewidth(0.7)
    fig.subplots_adjust(left=0.33, right=0.98, top=0.97, bottom=0.27)
    output = DOCS / "rl_results/rl_action_loops_iter90"
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output.with_suffix(".svg"), metadata={"Date": None})
    svg = output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-dir", type=Path)
    args = parser.parse_args()
    intervals = compute_intervals(args.compact_dir) if args.compact_dir else json.loads(INTERVALS.read_text())
    assert intervals["aggregate_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for key, _, _ in RUNS:
        row = intervals["methods"][key]
        expected = [r["loop_failures"] for r in json.loads(SOURCE.read_text())["iteration90"][key]]
        assert row["repeat_counts"] == expected
        assert np.isclose(row["mean_per300"], np.mean(expected))
    render(intervals)
    print(json.dumps(intervals["methods"], indent=2))


if __name__ == "__main__":
    main()
