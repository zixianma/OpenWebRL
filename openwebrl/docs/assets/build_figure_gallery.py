#!/usr/bin/env python3
"""Build the figures-only gallery from all local figure links in ARM_FINAL_RESULTS.

Run from any directory. PNG/PDF/SVG versions of the same stem share one entry;
all source-linked paths remain linked. Historical and collapsed source sections
are included, and the method diagrams remain as gallery additions.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import re

DOCS = Path(__file__).resolve().parents[1]
SOURCE = DOCS / "ARM_FINAL_RESULTS.md"
OUTPUT = DOCS / "ARM_FIGURES.md"
LINK = re.compile(r"!?\[([^\]]*)\]\(([^)]+)\)")
EXTENSIONS = {".png", ".svg", ".pdf", ".jpg", ".jpeg", ".gif", ".webp"}

# Only presentation metadata is curated. Source discovery and coverage checks
# include every local plot link, even when its source section is collapsed.
GROUPS = [
    "Methods", "Teacher and student ADM", "Actor + ADM trade-offs",
    "Recipe 3 data ablations", "RL training and evaluation",
    "BrowserCode scaling: completed and gap-repaired results",
    "BrowserCode Pareto variants: before gap recovery",
    "BrowserCode scaling: historical and supplementary views",
    "BrowserCode on US-residential stealth browsers", "Claude BrowserCode",
    "Shared-Terra comparisons", "API comparisons: earlier cohorts",
    "Selection labels: historical comparisons", "Selection data: earlier sweeps",
    "RL curves: historical and all-run views",
]
TITLES = {
    "figure1": "Action Decision Models: overview",
    "selection-offline-eval-20261022": "Rollout collection and action decision labeling",
    "selection-arch-overview-20261107": "ADM architecture",
    "selection-label-setups-kfold-20261101": "Teacher and student: offline and online",
    "selection-label-stages-kfold-20261101": "Label sources across three stages: 5-fold",
    "selection-label-same-state-kfold-20261101": "Training, held-out and online: 5-fold",
    "selection-recipe3-vs-cost-20261104": "All actors: benchmark mean",
    "selection-recipe3-vs-cost-per-benchmark-20261104": "All actors: per benchmark",
    "selection-recipe3-vs-cost-sft-actors-20261104": "SFT actors: benchmark mean",
    "selection-recipe3-vs-cost-per-benchmark-sft-actors-20261104": "SFT actors: per benchmark",
    "selection-recipe3-vs-cost-rl-actors-20261104": "RL actors: benchmark mean",
    "selection-recipe3-data-ablations-compact-20261105": "Training states and panels: benchmark mean",
    "selection-recipe3-data-ablations-per-benchmark-20261105": "Training states and panels: per benchmark",
    "rl_learning_compact": "Training and validation: valid-only success",
    "rl_learning_compact_overall": "Training and validation: overall success",
    "browsercode_scaling14_common_valid_weighted": "Fourteen settings: same valid tasks",
    "browsercode_scaling14_own_valid_weighted": "Fourteen settings: own-valid tasks",
    "browsercode_scaling_gap_repaired_compact": "Scaling after gap recovery",
    "browsercode_scaling_pareto_preview": "Per-benchmark Pareto frontiers",
    "browsercode_scaling_weighted_pareto_preview": "Weighted-average Pareto frontier",
    "browsercode_scaling_common_valid_pareto_preview": "Common-valid weighted Pareto frontier",
    "browsercode_scaling_weighted_pareto_highlight_right_legend_preview": "Sixteen settings: right-side legend",
    "browsercode_scaling_weighted_pareto_highlight_preview": "Sixteen settings: bottom legend",
    "browsercode_scaling_weighted_pareto_highlight_right_legend_overall_success_preview": "Sixteen settings: overall success",
    "browsercode_scaling_weighted_pareto_highlight_right_legend_common_valid_preview": "Sixteen settings: same valid tasks",
    "browsercode_scaling_strategies_compact": "Scaling strategies: compact",
    "browsercode_scaling_by_family": "Scaling strategies: six-panel view",
    "browsercode_scaling": "Success and efficiency: per benchmark",
    "browsercode_selection_interaction": "Action selection and browser-code batches",
    "browsercode_selection_paired_outcomes": "Action selection: paired outcomes",
    "browsercode_selection_scaling": "Candidate-count scaling",
    "browsercode_scaling_average": "Equal-weight benchmark average",
    "browsercode_scaling_weighted_average": "Task-weighted benchmark average",
    "browsercode_scaling_common_valid_weighted_average": "Eleven methods: same valid tasks",
    "browsercode_scaling_tokens": "Output and reasoning token consumption",
    "us_stealth_observed_costs": "US-stealth success and observed browser costs",
    "claude_tradeoffs": "Claude: high-effort trade-offs",
    "claude_tradeoffs_common_valid": "Claude: same valid tasks",
    "shared_terra_high_four_row_paper": "High-effort trade-offs: three benchmarks",
    "om2w_terra_all_tradeoffs": "OM2W: all GPT settings",
    "om2w_terra_valid_only_tradeoffs_browsercode": "OM2W: valid-task BrowserCode efficiency",
    "om2w_terra_valid_only_tradeoffs": "OM2W: valid-task efficiency, both harnesses",
    "om2w_terra_high_valid_only_tradeoffs": "OM2W: pre-recovery high-effort comparison",
    "om2w_terra_effort_tradeoffs": "OM2W: original medium- and low-effort cohort",
    "api_performance_tradeoffs": "Three-benchmark API trade-offs",
    "api_performance_vs_cost": "API performance versus cost",
    "api_performance_vs_latency": "API performance versus latency",
    "api_performance_vs_browser_steps": "API performance versus browser steps",
    "om2w_harness_tradeoffs": "OM2W: historical grading, all 300 tasks",
    "selection-head-labels-vs-cost-20261021": "October 21: label sources versus cost",
    "selection-label-stages-20261027": "October 27: label sources across three stages",
    "selection-label-same-state-20261027": "October 27: training, test and online",
    "selection-label-setups-20261027": "October 27: three-panel label comparison",
    "selection-recipe3-data-states-20261105": "Recipe 3 states: earlier valid-only view",
    "selection-recipe3-data-panels-20261105": "Recipe 3 panels: earlier valid-only view",
    "selection-state-scale-1panel-mean-20261027": "One panel per state: benchmark mean",
    "selection-state-scale-1panel-paired-20261027": "One panel per state: per benchmark",
    "rl_eval_learning_curves": "Outcome-only and ADM: local validation, iterations 20–90",
    "validation_success": "Validation success: all runs",
    "training_success": "Training success: all runs",
}


def figure_group(stem: str) -> str:
    name = Path(stem).name
    if "/methods/" in stem:
        return GROUPS[0]
    if name.startswith("selection-label-") and "kfold" in name:
        return GROUPS[1]
    if name.startswith("selection-recipe3-vs-cost"):
        return GROUPS[2]
    if name.startswith("selection-recipe3-data-ablations"):
        return GROUPS[3]
    if name.startswith("rl_learning_compact"):
        return GROUPS[4]
    if "/browsercode_cached_selection_" in stem or "/browsercode_gap_repaired_" in stem:
        return GROUPS[5]
    if "/browsercode_inference_scaling_" in stem:
        return GROUPS[6] if "pareto" in name else GROUPS[7]
    if "/browsercode_us_stealth_" in stem:
        return GROUPS[8]
    if "/claude_browsercode_" in stem:
        return GROUPS[9]
    if "/cuawright_browsercode_" in stem and "terra" in name:
        return GROUPS[10]
    if "/sol6_luna_" in stem or "/cuawright_browsercode_" in stem:
        return GROUPS[11]
    if name.startswith(("selection-head-labels", "selection-label-")):
        return GROUPS[12]
    if name.startswith(("selection-state-scale", "selection-recipe3-data-")):
        return GROUPS[13]
    if name in {"rl_eval_learning_curves", "training_success", "validation_success"}:
        return GROUPS[14]
    raise ValueError(f"Assign a gallery topic for the new source figure: {stem}")


def collect_figures() -> tuple[dict[str, list[str]], set[str]]:
    figures: dict[str, list[str]] = OrderedDict()
    source_paths = set()
    for _, raw_url in LINK.findall(SOURCE.read_text()):
        url = raw_url.strip().strip("<>")
        if "://" in url or url.startswith("#"):
            continue
        path = Path(url.split("#", 1)[0].split("?", 1)[0])
        if path.suffix.lower() not in EXTENSIONS:
            continue
        if not (DOCS / path).is_file():
            raise FileNotFoundError(f"Missing source figure: {path}")
        url = path.as_posix()
        source_paths.add(url)
        formats = figures.setdefault(path.with_suffix("").as_posix(), [])
        if url not in formats:
            formats.append(url)
    for name in ("selection-offline-eval-20261022", "selection-arch-overview-20261107"):
        stem = f"arm_results/methods/{name}"
        figures.setdefault(stem, [stem + ".png"])
    stem = "arm_results/methods/adm_overview/figure1"
    figures.setdefault(stem, [stem + ext for ext in (".png", ".pdf", ".svg")])
    return figures, source_paths


def render() -> str:
    figures, source_paths = collect_figures()
    topics = {group: [] for group in GROUPS}
    ordered = sorted(figures.items(), key=lambda item: -1 if Path(item[0]).name == "figure1" else 0 if Path(item[0]).name == "selection-label-setups-kfold-20261101" else 1)
    for stem, formats in ordered:
        name = Path(stem).name
        title = TITLES[name]
        if "/common_valid111/" in stem:
            title = "OM2W: historical grading, 111 common-valid tasks"
        image = next((stem + ext for ext in (".png", ".svg", ".jpg", ".jpeg", ".webp", ".gif")
                      if (DOCS / (stem + ext)).is_file()), None)
        if image is None:
            raise FileNotFoundError(f"No embeddable image for {stem}")
        # Keep every source-linked format, without duplicating its preview.
        links = [f"[{Path(url).suffix[1:].upper()}]({url})" for url in formats if url != image]
        if name == "figure1":
            bundle = Path(stem).parent.as_posix()
            links.extend([f"[Plot script]({bundle}/plot_figure1.py)",
                          f"[Reproduce]({bundle}/README.md)"])
        block = f"### {title}\n\n![{title}]({image})\n"
        if links:
            block += "\n" + " · ".join(links) + "\n"
        topics[figure_group(stem)].append(block)
    text = "# Figure gallery\n\n<!-- Generated by assets/build_figure_gallery.py from ARM_FINAL_RESULTS.md. -->\n"
    for topic, blocks in topics.items():
        if blocks:
            text += f"\n## {topic}\n\n" + "\n".join(blocks)
    linked = {url for _, url in LINK.findall(text)}
    missing = source_paths - linked
    if missing:
        raise AssertionError(f"Source figure links missing from gallery: {sorted(missing)}")
    if len(re.findall(r"^!\[", text, re.MULTILINE)) != len(figures):
        raise AssertionError("Expected one preview per unique figure stem")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check coverage and that the generated gallery is current")
    args = parser.parse_args()
    text = render()
    if args.check:
        if OUTPUT.read_text() != text:
            raise SystemExit("ARM_FIGURES.md is stale; rerun assets/build_figure_gallery.py")
    else:
        OUTPUT.write_text(text)
    figures, paths = collect_figures()
    print(f"Gallery {'verified' if args.check else 'updated'}: {len(figures)} figure previews, {len(paths)} source-linked image/PDF paths, full coverage.")


if __name__ == "__main__":
    main()
