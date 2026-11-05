#!/usr/bin/env python3
"""Preview valid-only success versus repeated-action incidence at iteration 90.

Render from public aggregates. --compact-dir rebuilds uncertainty from the private
compact records without exporting any task identities or trajectories.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Shared paper typography, including bundled Albert Sans fonts.
import sys as _paper_sys
from pathlib import Path as _PaperPath
_paper_sys.path.insert(0, str(next(p for p in _PaperPath(__file__).resolve().parents
                                if (p / "paper_plot_style.py").is_file())))
import paper_plot_style as _paper_style

from matplotlib.lines import Line2D
import numpy as np

from plot_rl_repetition_incidence import AGGREGATE as REPETITION, DOCS, HERE, RUNS, action_counts

AGGREGATE = HERE / "rl-repetition-success-20261104.json"
OUTPUT = DOCS / "rl_results/rl_repetition_success_preview"


def build(compact_dir):
    source = json.loads(REPETITION.read_text())
    hashes = {item["file"]: item["sha256"] for item in source["local_source_hashes"]}
    draws, seed = 20000, 20261104
    indices = np.random.default_rng(seed).integers(0, 300, (draws, 300))
    methods, boot = {}, {}
    task_order = None
    for key, *_ in RUNS:
        repeats = []
        for j in (1, 2, 3):
            name = f"repeat-iter90-{key}-r{j}-385593-a1.jsonl"
            path = compact_dir / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            records = {str(row["task_id"]): row for row in rows}
            assert len(rows) == len(records) == 300
            if task_order is None:
                task_order = sorted(records)
            assert sorted(records) == task_order
            repeats.append([records[task] for task in task_order])
        valid = np.asarray([[bool(row["valid"]) for row in rows] for rows in repeats])
        success = np.asarray([[bool(row["success"]) for row in rows] for rows in repeats])
        repeated = np.asarray([[action_counts(row)[0] >= 3 for row in rows] for rows in repeats])
        assert not np.any(success & ~valid)
        repetition_per_task = (valid & repeated).sum(axis=0)
        valid_per_task = valid.sum(axis=0)
        success_per_task = success.sum(axis=0)
        bx = 100 * repetition_per_task[indices].sum(axis=1) / valid_per_task[indices].sum(axis=1)
        by = 100 * success_per_task[indices].sum(axis=1) / valid_per_task[indices].sum(axis=1)
        by_overall = 100 * success_per_task[indices].sum(axis=1) / 900
        reference = source["methods"][key]
        repetition = {"episodes": int(repetition_per_task.sum()),
                      "valid_episodes": int(valid.sum()),
                      "percent": float(100 * repetition_per_task.sum() / valid.sum()),
                      "ci95_percent": np.percentile(bx, [2.5, 97.5]).tolist()}
        assert repetition["valid_episodes"] == reference["valid_episodes"]
        for field in ("episodes", "percent", "ci95_percent"):
            assert repetition[field] == reference["thresholds"]["3"][field]
        methods[key] = {"repetition_incidence": repetition,
                        "valid_success": {"successes": int(success.sum()), "valid_episodes": int(valid.sum()),
                                          "percent": float(100 * success.sum() / valid.sum()),
                                          "ci95_percent": np.percentile(by, [2.5, 97.5]).tolist()},
                        "overall_success": {"successes": int(success.sum()), "planned_episodes": 900,
                                            "repeat_successes": success.sum(axis=1).astype(int).tolist(),
                                            "percent": float(100 * success.sum() / 900),
                                            "ci95_percent": np.percentile(by_overall, [2.5, 97.5]).tolist()}}
        boot[key] = {"repetition_incidence": bx, "valid_success": by, "overall_success": by_overall}
    paired = {}
    for baseline in ("outcome48", "outcome56"):
        paired[baseline] = {}
        for metric in ("repetition_incidence", "valid_success", "overall_success"):
            samples = boot["gate-b"][metric] - boot[baseline][metric]
            paired[baseline][metric] = {
                "delta_pp": methods["gate-b"][metric]["percent"] - methods[baseline][metric]["percent"],
                "ci95_pp": np.percentile(samples, [2.5, 97.5]).tolist(),
            }
    result = {
        "x": "Repeated-action incidence: >=3 identical non-wait/non-done calls / each method's valid episodes, successes included",
        "y": "Valid-only success: successes / each method's valid episodes, pooled across three repeats; invalid outcomes excluded",
        "uncertainty": {"method": "Marginal pointwise percentile intervals from shared task-cluster resamples, retaining all repeats and methods; valid-only ratios recomputed in each resample",
                        "tasks": 300, "repeats": 3, "draws": draws, "seed": seed,
                        "scope": "Conditional on fixed checkpoints and evaluation windows; excludes training-seed and evaluation-window uncertainty; horizontal/vertical intervals are not a joint 95% confidence region"},
        "repetition_aggregate_sha256": hashlib.sha256(REPETITION.read_bytes()).hexdigest(),
        "local_source_hashes": source["local_source_hashes"],
        "methods": methods, "adm_minus_baseline_pp": paired,
    }
    AGGREGATE.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"methods": methods, "paired": paired}, indent=2))


def render():
    data = json.loads(AGGREGATE.read_text())
    assert data["repetition_aggregate_sha256"] == hashlib.sha256(REPETITION.read_bytes()).hexdigest()
    plt.rcParams.update({"font.family": "Albert Sans", "font.size": 9.5,
                         "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "rl-repetition-success-preview"})
    fig, ax = plt.subplots(figsize=(3.8, 2.65))
    for key, _, _, color, _ in RUNS:
        row = data["methods"][key]
        x, y = row["repetition_incidence"], row["valid_success"]
        yerr = np.asarray([[y["percent"] - y["ci95_percent"][0]],
                           [y["ci95_percent"][1] - y["percent"]]])
        ax.errorbar(x["percent"], y["percent"], yerr=yerr,
                    fmt="o", color=color, markersize=5.5, capsize=2.5,
                    elinewidth=1, linewidth=1, zorder=4 if key == "gate-b" else 3)
    ax.set_xlabel("Repeated-action incidence (%)", labelpad=4)
    ax.set_ylabel("Valid-only success (%)", labelpad=5)
    ax.yaxis.set_label_coords(-0.13, 0.43)
    ax.set_xlim(13, 47)
    ax.set_ylim(35, 60)
    ax.set_xticks([15, 20, 25, 30, 35, 40, 45])
    ax.set_yticks([40, 45, 50, 55])
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#666666")
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(length=3, color="#666666")
    handles = [Line2D([], [], marker="o", color=color, linestyle="none", markersize=5.5,
                      label=label) for _, label, _, color, _ in RUNS]
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 0.005),
                        frameon=False, ncol=1, fontsize=8.5, handlelength=1,
                        handletextpad=0.5, labelspacing=0.3)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.965, bottom=0.42)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1), text.get_text()
    assert not legend.get_window_extent(renderer).overlaps(ax.xaxis.label.get_window_extent(renderer))
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUTPUT.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUTPUT.with_suffix(".svg"), metadata={"Date": None})
    svg = OUTPUT.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    assert b"/FontFile2" in OUTPUT.with_suffix(".pdf").read_bytes()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-dir", type=Path, help="Rebuild statistics from private JSONL records")
    args = parser.parse_args()
    if args.compact_dir:
        build(args.compact_dir)
    render()
