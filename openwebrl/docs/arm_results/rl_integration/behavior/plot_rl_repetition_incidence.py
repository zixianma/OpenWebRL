#!/usr/bin/env python3
"""Paper figure: repeated-action incidence and threshold sensitivity.

Render from the public aggregate by default. Rebuild it with --compact-dir
pointing to the private compact trajectories; only counts, rates, intervals and
source hashes are exported. No task identities or trajectories are published.
"""
import argparse
from collections import Counter
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
from paper_plot_palette import COLORS as C

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[2]
REFERENCE = HERE / "rl-action-loops-20261103.json"
AGGREGATE = HERE / "rl-repetition-incidence-20261104.json"
OUTPUT = DOCS / "rl_results/rl_repetition_incidence"
THRESHOLDS = (2, 3, 4, 5)
RUNS = (
    ("outcome48", "Outcome-only (48 groups)", "Outcome-only\n48 groups", C["BASELINE_LIGHT"], "o"),
    ("outcome56", "Outcome-only (56 groups)", "Outcome-only\n56 groups", C["BASELINE_DARK"], "o"),
    ("gate-b", "Outcome + ADM rewards (ours)", "Outcome + ADM\nrewards (ours)", C["ADM_DEFAULT"], "o"),
)


def action_counts(row):
    """Match historical max_repeat and filtered consecutive_repeat exactly."""
    calls = [step["call"] for step in row.get("steps") or [] if step.get("call")]
    keys = [json.dumps(call, sort_keys=True) for call in calls
            if call.get("name") not in ("done", "wait")]
    maximum = max(Counter(keys).values(), default=0)
    best = run = int(bool(keys))
    for previous, current in zip(keys, keys[1:]):
        run = run + 1 if previous == current else 1
        best = max(best, run)
    return maximum, best


def build(compact_dir):
    reference = json.loads(REFERENCE.read_text())
    cohorts, hashes = {}, []
    task_order = None
    for key, *_ in RUNS:
        cohorts[key] = []
        for repeat in (1, 2, 3):
            name = f"repeat-iter90-{key}-r{repeat}-385593-a1.jsonl"
            path = compact_dir / name
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            records = {str(row["task_id"]): row for row in rows}
            assert len(rows) == len(records) == 300, name
            if task_order is None:
                task_order = sorted(records)
            assert sorted(records) == task_order, name
            assert all(row.get("steps") for row in rows if row["valid"]), name
            cohorts[key].append([records[task] for task in task_order])
            hashes.append({"file": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    draws, seed = 20000, 20261104
    # One shared task resample keeps every repeat, method and threshold paired.
    indices = np.random.default_rng(seed).integers(0, 300, size=(draws, 300))
    result = {
        "metric": "Repeated-action incidence (%) among valid episodes",
        "definition": "100 * valid episodes with >= k identical non-wait/non-done calls / valid episodes; successes included; occurrences need not be consecutive",
        "pooling": "Ratio of summed counts across three repeats, not an unweighted mean of repeat percentages",
        "thresholds": list(THRESHOLDS),
        "uncertainty": {
            "method": "Pointwise percentile bootstrap of 300 task clusters, preserving all three repeats, methods and thresholds; ratio recomputed in each resample",
            "tasks": 300, "repeats": 3, "draws": draws, "seed": seed,
            "scope": "Conditional on fixed checkpoints and three evaluation windows; excludes training-seed and evaluation-window uncertainty; not simultaneous intervals",
        },
        "matching": {
            "identity": "JSON serialization of saved first tool call per actor turn with sorted object keys; no argument/string/URL normalization",
            "extraction": "Historical compact extractor retains the first tool_call match per turn; malformed JSON may be represented by an unparsable pseudo-call with its first 200 raw characters",
            "consecutive_check": "Longest run after removing absent calls and wait/done calls; consecutive retained actions, not necessarily consecutive actor turns",
        },
        "reference_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        "local_source_hashes": hashes,
        "methods": {}, "paired_differences_pp": {},
    }
    bootstrap = {}
    for key, *_ in RUNS:
        repeats = cohorts[key]
        valid = np.asarray([[bool(row["valid"]) for row in rows] for rows in repeats])
        success = np.asarray([[bool(row["success"]) for row in rows] for rows in repeats])
        maxima = np.asarray([[action_counts(row) for row in rows] for rows in repeats])
        repeat_counts = []
        for j in range(3):
            published = reference["iteration90"][key][j]
            counts = [int((valid[j] & (maxima[j, :, 0] >= k)).sum()) for k in THRESHOLDS]
            cons = int((valid[j] & (maxima[j, :, 1] >= 3)).sum())
            assert int(valid[j].sum()) == published["valid"]
            assert counts[THRESHOLDS.index(3)] == published["loop_episodes"]
            for k, name in ((4, "ge4"), (5, "ge5")):
                assert np.isclose(counts[THRESHOLDS.index(k)] / valid[j].sum(), published["sensitivity"][name])
            assert np.isclose(cons / valid[j].sum(), published["sensitivity"]["consecutive3"])
            repeat_counts.append({"valid": int(valid[j].sum()), "incidence_counts": counts,
                                  "filtered_consecutive_ge3_count": cons})
        valid_per_task = valid.sum(axis=0)
        boot_valid = valid_per_task[indices].sum(axis=1)
        assert np.all(boot_valid > 0)
        method = {"planned_episodes": 900, "valid_episodes": int(valid.sum()),
                  "invalid_episodes": int(900 - valid.sum()), "repeats": repeat_counts,
                  "thresholds": {}}
        bootstrap[key] = {}
        for k in (*THRESHOLDS, "filtered_consecutive_ge3"):
            flag = maxima[:, :, 0] >= k if isinstance(k, int) else maxima[:, :, 1] >= 3
            counts_per_task = (valid & flag).sum(axis=0)
            samples = 100 * counts_per_task[indices].sum(axis=1) / boot_valid
            row = {"episodes": int(counts_per_task.sum()),
                   "percent": float(100 * counts_per_task.sum() / valid.sum()),
                   "ci95_percent": np.percentile(samples, [2.5, 97.5]).tolist()}
            row["outcome_breakdown"] = {}
            for name, outcome in (("success", success), ("failure", ~success)):
                count = int((valid & flag & outcome).sum())
                row["outcome_breakdown"][name] = {
                    "repeated_episodes": count,
                    "valid_outcome_episodes": int((valid & outcome).sum()),
                    "percent_of_all_valid_episodes": float(100 * count / valid.sum()),
                }
            assert sum(part["repeated_episodes"] for part in row["outcome_breakdown"].values()) == row["episodes"]
            if isinstance(k, int):
                method["thresholds"][str(k)] = row
            else:
                method[k] = row
            bootstrap[key][str(k)] = samples
        result["methods"][key] = method
    for baseline in ("outcome48", "outcome56"):
        comparisons = {}
        for k in (*THRESHOLDS, "filtered_consecutive_ge3"):
            a, b = result["methods"]["gate-b"], result["methods"][baseline]
            ar = a["thresholds"][str(k)] if isinstance(k, int) else a[k]
            br = b["thresholds"][str(k)] if isinstance(k, int) else b[k]
            samples = bootstrap["gate-b"][str(k)] - bootstrap[baseline][str(k)]
            comparisons[str(k)] = {"delta_pp": ar["percent"] - br["percent"],
                                   "ci95_pp": np.percentile(samples, [2.5, 97.5]).tolist()}
        result["paired_differences_pp"][f"gate-b - {baseline}"] = comparisons
    AGGREGATE.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result["methods"][key] for key, *_ in RUNS}, indent=2))


def render():
    data = json.loads(AGGREGATE.read_text())
    assert data["reference_sha256"] == hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    plt.rcParams.update({"text.color": C["INK"], "axes.labelcolor": C["INK"],
                         "xtick.color": C["MUTED"], "ytick.color": C["MUTED"], "font.family": "Albert Sans", "font.size": 9.5,
                         "axes.titlesize": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "rl-repetition-incidence"})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    left, right = axes
    means = np.asarray([data["methods"][key]["thresholds"]["3"]["percent"] for key, *_ in RUNS])
    limits = np.asarray([data["methods"][key]["thresholds"]["3"]["ci95_percent"] for key, *_ in RUNS])
    outcomes = [data["methods"][key]["thresholds"]["3"]["outcome_breakdown"] for key, *_ in RUNS]
    failures = np.asarray([row["failure"]["percent_of_all_valid_episodes"] for row in outcomes])
    successes = np.asarray([row["success"]["percent_of_all_valid_episodes"] for row in outcomes])
    colors = [row[3] for row in RUNS]
    bars = left.bar(range(3), failures, width=0.56, color=colors, zorder=3)
    success_bars = left.bar(range(3), successes, bottom=failures, width=0.56,
                            facecolor="white", edgecolor=colors, hatch="///", linewidth=0.8,
                            zorder=3)
    left.errorbar(range(3), means, yerr=np.vstack((means - limits[:, 0], limits[:, 1] - means)),
                  fmt="none", ecolor=C["MUTED"], elinewidth=1, capsize=3, capthick=1, zorder=4)
    left.legend(handles=[Patch(facecolor=C["BASELINE_DARK"], label="Failure"),
                         Patch(facecolor="white", edgecolor=C["BASELINE_DARK"], hatch="///", label="Success")],
                loc="upper right", frameon=False, ncol=2, fontsize=8,
                handlelength=1.3, columnspacing=1.2, handletextpad=0.5)
    for j, mean in enumerate(means):
        left.text(j, limits[j, 1] + 1.0, f"{mean:.1f}%", ha="center", va="bottom",
                  color=RUNS[j][3], fontsize=10, fontweight="medium")
    left.set_xticks(range(3), [row[2] for row in RUNS], fontsize=8.5)
    left.get_xticklabels()[-1].set_color(RUNS[-1][3])
    left.set_xlim(-0.48, 2.48)
    left.tick_params(axis="x", length=0, pad=6)
    left.set_title("(a) Repetition incidence (K=3)", loc="left", pad=9)

    for key, _, _, color, marker in RUNS:
        rows = [data["methods"][key]["thresholds"][str(k)] for k in THRESHOLDS]
        values = np.asarray([row["percent"] for row in rows])
        ci = np.asarray([row["ci95_percent"] for row in rows])
        assert np.all(np.diff(values) <= 0)
        right.errorbar(THRESHOLDS, values,
                       yerr=np.vstack((values - ci[:, 0], ci[:, 1] - values)),
                       color=color, marker=marker, markersize=4.5, linewidth=1.5,
                       capsize=2.5, elinewidth=0.9, zorder=4 if key == "gate-b" else 3)
    right.set_xticks(THRESHOLDS)
    right.set_xlim(min(THRESHOLDS) - 0.18, max(THRESHOLDS) + 0.18)
    right.set_xlabel("Repetition threshold, K", labelpad=5)
    right.set_title("(b) Threshold robustness", loc="left", pad=9)
    right.tick_params(axis="y", left=False)
    largest_upper = max(data["methods"][key]["thresholds"][str(k)]["ci95_percent"][1]
                        for key, *_ in RUNS for k in THRESHOLDS)
    ymax = max(50, int(np.ceil((largest_upper + 2) / 10)) * 10)
    for ax in axes:
        ax.set_ylim(0, ymax)
        ax.set_yticks(range(0, ymax + 1, 10))
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C["BASELINE_DARK"])
            ax.spines[side].set_linewidth(0.7)
        ax.tick_params(length=3, color=C["BASELINE_DARK"])
    right.spines["left"].set_visible(False)
    fig.supylabel("Episodes with repeated actions (%)", x=0.008, y=0.53, fontsize=9.5)
    handles = [Line2D([], [], color=color, marker=marker, linewidth=1.5, markersize=4.5,
                      label=label) for _, label, _, color, marker in RUNS]
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.005),
                        frameon=False, ncol=3, fontsize=8.5, handlelength=1.4,
                        columnspacing=1.3, handletextpad=0.5)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.87, bottom=0.29, wspace=0.13)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Check every visible label and the shared legend is inside the canvas.
    for artist in fig.findobj(match=matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1), artist.get_text()
    labels = [text.get_window_extent(renderer) for text in left.get_xticklabels()]
    assert all(not a.overlaps(b) for a, b in zip(labels, labels[1:]))
    legend_box = legend.get_window_extent(renderer)
    assert all(not legend_box.overlaps(box) for box in labels)
    stack_heights = np.asarray([a.get_height() + b.get_height() for a, b in zip(bars, success_bars)])
    assert np.allclose(stack_heights, means, rtol=0, atol=1e-12)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUTPUT.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUTPUT.with_suffix(".svg"), metadata={"Date": None})
    svg = OUTPUT.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    assert b"/FontFile2" in OUTPUT.with_suffix(".pdf").read_bytes()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-dir", type=Path, help="Rebuild aggregate from private JSONL records")
    args = parser.parse_args()
    if args.compact_dir:
        build(args.compact_dir)
    render()
