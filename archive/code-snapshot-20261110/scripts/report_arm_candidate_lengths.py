#!/usr/bin/env python3
"""CPU-only token-length snapshot; requires tokenizers and matplotlib."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import statistics


def summarize(values):
    values = sorted(values)
    if not values:
        return {'n': 0}
    def quantile(q):
        index = (len(values) - 1) * q
        lo = int(index); hi = min(lo + 1, len(values) - 1)
        return round(values[lo] + (values[hi] - values[lo]) * (index - lo), 2)
    return dict(n=len(values), mean=round(statistics.mean(values), 2),
                min=values[0], max=values[-1],
                **{f'p{p}': quantile(p / 100) for p in [10, 25, 50, 75, 90, 95, 99]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--traces', type=Path, default=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/selection-actions-only-20260911/actions_only/selections'))
    p.add_argument('--output', type=Path, default=Path('openwebrl/docs/arm_results/compact_candidate_lengths'))
    args = p.parse_args()
    # One lightweight tokenizer process; bound accidental login-node work.
    resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from tokenizers import Tokenizer
    tokenizer_path = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT/tokenizer.json')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    length = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    groups = {name: {k: [] for k in ['reasoning', 'action', 'combined']} for name in ['all_candidates', 'selected', 'unselected']}
    candidate_sets = {k: [] for k in ['reasoning', 'action', 'combined']}
    finish = Counter(); selected_finish = Counter(); seen = set(); tasks = set()
    partial = 0; shadow = []
    for path in sorted(args.traces.glob('*.jsonl')):
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    partial += 1; continue
                raise
            key = (row['task_id'], row['turn'])
            if key in seen:
                raise ValueError(f'Duplicate task/turn: {key}; select one attempt before measuring')
            seen.add(key); tasks.add(row['task_id'])
            total = dict.fromkeys(candidate_sets, 0)
            for j, candidate in enumerate(row['candidates']):
                values = {'reasoning': length(candidate['thought']), 'action': length(candidate['action'])}
                values['combined'] = values['reasoning'] + values['action']
                for field, value in values.items():
                    groups['all_candidates'][field].append(value)
                    groups['selected' if j == row['selected_index'] else 'unselected'][field].append(value)
                    total[field] += value
                kind = row.get('finish_types', ['unknown'] * len(row['candidates']))[j]
                finish[str(kind)] += 1
                if j == row['selected_index']: selected_finish[str(kind)] += 1
            for field, value in total.items(): candidate_sets[field].append(value)
            full = row.get('shadow_full', {}).get('input_tokens')
            compact = row.get('selector_telemetry', {}).get('input_tokens')
            if full is not None and compact is not None: shadow.append((full, compact))
    report = dict(snapshot_utc=datetime.now(timezone.utc).isoformat(), trace_root=str(args.traces),
        tasks_with_saved_turns=len(tasks), saved_turns=len(seen), trailing_partial_lines_skipped=partial,
        tokenizer=str(tokenizer_path),
        definition='Retokenized saved text without added special tokens. Reasoning excludes think delimiters; action includes tool-call syntax and any saved end-of-turn token. Combined = reasoning + action, excluding stripped delimiters/whitespace. These are component counts, not exact original generation lengths.',
        limitation='Live partial snapshot, weighted by saved turns/candidates, not equally by task. Short or fast trajectories may be overrepresented. Truncated candidates retained.',
        groups={name: {field:summarize(values) for field,values in fields.items()} for name,fields in groups.items()},
        token_count_frequencies={name: {field:dict(sorted(Counter(values).items())) for field,values in fields.items()} for name,fields in groups.items()},
        per_candidate_set={field:summarize(values) for field,values in candidate_sets.items()},
        finish_types=dict(finish), selected_finish_types=dict(selected_finish),
        fraction_tokens_reasoning=sum(groups['all_candidates']['reasoning'])/max(1,sum(groups['all_candidates']['combined'])),
        selector_prompt_pairs=dict(n=len(shadow), full=summarize([v[0] for v in shadow]),
            compact=summarize([v[1] for v in shadow]), removed=summarize([v[0]-v[1] for v in shadow])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, field in zip(axes, ['action','reasoning','combined']):
        for group, label, color in [('all_candidates','All candidates','#64748b'),('selected','ARM selected','#2563eb')]:
            values = sorted(groups[group][field])
            ax.step(values, [(i+1)/len(values) for i in range(len(values))], where='post', label=label, color=color)
        ax.set(title=field.capitalize(), xlabel='Tokens', ylabel='Fraction at or below length', ylim=(0,1.01))
        ax.grid(alpha=.2); ax.legend(loc='lower right', fontsize=8)
    fig.suptitle(f'Live action-only ARM evaluation: {len(seen):,} turns, {len(tasks)} tasks with traces')
    fig.savefig(args.output.with_suffix('.png'), dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
