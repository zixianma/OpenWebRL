"""Shared display labels; population definitions remain in figure captions.

Only Matplotlib text changes. Data, axes scales, populations and weighting do not.
Applied at export so legacy renderers and composite figures follow the same rule.
"""
from pathlib import Path
import re
from functools import wraps
from matplotlib.figure import Figure
from matplotlib.text import Text

MEAN_STEMS = {
    "selection-recipe3-vs-cost-sft-actors-overall-baselines-with-rl-20261113",
    "selection-recipe3-vs-cost-sft-actors-overall-baselines-20261113",
    "selection-recipe3-vs-cost-sft-actors-overall-no-rl-baselines-20261104",
    'browsercode_scaling14_common_valid_weighted', 'browsercode_scaling14_own_valid_weighted',
    'browsercode_scaling_gap_repaired_compact', 'browsercode_scaling_by_family',
    'browsercode_scaling_strategies_compact', 'browsercode_selection_interaction',
    'browsercode_selection_scaling', 'selection-head-labels-vs-cost-20261021',
    'selection-recipe3-vs-cost-20261104', 'selection-recipe3-vs-cost-sft-actors-20261104',
    'selection-recipe3-vs-cost-sft-actors-overall-no-rl-20261104',
    'selection-recipe3-vs-cost-sft-actors-overall-20261104', 'selection-recipe3-vs-cost-rl-actors-20261104',
    'selection-recipe3-data-ablations-compact-20261105', 'selection-recipe3-data-states-20261105',
    'selection-recipe3-data-panels-20261105', 'selection-recipe3-data-scaling-20261112',
    'selection-recipe3-data-ablations-rerun-clean-20261109', 'selection-recipe3-data-ablations-rerun-20261109',
    'selection-head-scale-pooled-20261112', 'selection-head-scale-20261111',
    'selection-state-scale-1panel-mean-20261027',
    'api_performance_tradeoffs', 'api_performance_vs_cost', 'api_performance_vs_latency',
    'api_performance_vs_browser_steps',
}
POPULATION = re.compile(r'\b(?:valid[\s\-–−]*only|common[\s\-–−]*valid|shared[\s\-–−]*task|valid Terra|valid)\s+(?=success)', re.I)


def is_mean(stem):
    return stem in MEAN_STEMS or ('browsercode' in stem and any(t in stem for t in ('weighted', 'average')))


def display_text(value, *, mean=False):
    value = re.sub(r'(?im)^(valid[\s\-–−]*only)\s*\(%\)$', 'Success (%)', value)
    value = value.replace('Success on common-valid tasks (%)', 'Success (%)')
    value = POPULATION.sub('', value)
    value = re.sub(r'\bvalid[\s\-–−]*only\b[ :]*', '', value, flags=re.I)
    value = re.sub(r'(?im)^\s*3-benchmark mean\s*\n', '', value)
    value = re.sub(r'(?im)^\s*(?:overall|mean)\s*\n(?=success)', '', value)
    # A benchmark prefix may be retained on multi-row y axes.
    value = re.sub(r'(?im)^(overall|valid-only)\s*\(%\)$', 'Success (%)', value)
    percent = re.search(r'(?im)^(?:(?:mean|avg\.?|average|overall)\s+)*success(?: rate)?\s*(?:\(%\)|%)\s*$', value)
    if percent:
        label = 'Mean success (%)' if mean or re.search(r'(?i)\b(mean|avg\.?|average)\b', percent.group()) else 'Success (%)'
        value = value[:percent.start()] + label + value[percent.end():]
    value = re.sub(r'(?i)\b(?:avg\.?|average) success\b', 'Mean success', value)
    value = re.sub(r'(?i)\bmean overall success\b', 'Mean success', value)
    value = value.replace('Overall success, mean of 3 benchmarks, %', 'Mean success (%)')
    value = value.replace('Overall success, %', 'Success (%)')
    value = value.replace('Mean overall −', 'Mean success −')
    if 'Online-Mind2Web' in value and value.endswith('Mean success (%)'):
        value = value.replace('Mean success (%)', 'Success (%)')
    # Online label comparisons aggregate three benchmarks; offline panels score rollouts.
    value = value.replace('Online: overall success', 'Online: mean success')
    return value.strip()


def apply_labels(fig, output):
    stem = Path(str(output)).stem
    mean = is_mean(stem)
    for artist in fig.findobj(match=Text):
        old = artist.get_text()
        if old:
            new = display_text(old, mean=mean)
            if old != new:
                artist.set_text(new)


def install():
    if getattr(Figure.savefig, '_paper_labels', False):
        return
    original = Figure.savefig
    @wraps(original)
    def savefig(fig, fname, *args, **kwargs):
        apply_labels(fig, fname)
        return original(fig, fname, *args, **kwargs)
    savefig._paper_labels = True
    Figure.savefig = savefig
