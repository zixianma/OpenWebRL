#!/usr/bin/env python3
"""Plot audited aggregate costs with different protocols in separate panels."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]/'openwebrl/docs/arm_results/rl_integration'


def main():
    data = json.loads((ROOT/'inference-cost-passk-20261004.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    for mode, label, color, offset in [
        ('baseline', 'Actor, one episode', '#64748b', (10, 9)),
        ('scalar', 'ScalarRM', '#d97706', (-70, -22)),
        ('selection', 'SelectionARM', '#2563eb', (-95, 10)),
    ]:
        row = data['historical_om2w']['modes'][mode]
        x = row['actor_text_token_ratio_to_baseline']; y = 100*row['overall']
        ax.scatter(x, y, s=65, color=color, zorder=3)
        ax.annotate(f'{label}\n{y:.2f}%', (x,y), xytext=offset, textcoords='offset points', fontsize=9)
    ax.set(title='Historical OM2W: 300 tasks', xlabel='Saved actor-text tokens / baseline (proxy)',
           ylabel='Overall success (%)', xlim=(.5,6), ylim=(25,50))
    ax.text(.03, .94, 'Selector compute is additional\nEpisode pass@5 control is missing',
            transform=ax.transAxes, va='top', fontsize=9, color='#475569')
    ax = axes[1]
    rows = data['screening_2000']['pass_at_k']
    xs = [r['expected_random_subset_cost_per_task']['actor_output_tokens']/1000 for r in rows]
    ys = [100*r['overall_pass_at_k'] for r in rows]
    ax.plot(xs, ys, marker='o', color='#0f766e', linewidth=2)
    for x,y,r in zip(xs,ys,rows):
        ax.annotate(f"k={r['k']}\n{y:.2f}%", (x,y), xytext=(0,9), textcoords='offset points', ha='center', fontsize=9)
    ax.set(title='Separate training pool: 2,000 tasks', xlabel='Actor output tokens / task (thousands, metered)',
           ylabel='Episode pass@k (%)', xlim=(1,21), ylim=(25,75))
    ax.text(.03,.96,'Any success among k full episodes\nInvalid attempts count as no success',
            transform=ax.transAxes, va='top', fontsize=9, color='#475569')
    for ax in axes:
        ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
    fig.subplots_adjust(bottom=.23, top=.87, wspace=.28)
    fig.text(.5,.065,'Different task pools, sampling, horizons, and judges: compare costs within each panel.\n'
             'Historical retokenization is not GPU cost; the right panel pays for all k episodes without early stopping.',
             ha='center',fontsize=9,color='#475569')
    for extension in ('png','svg'):
        fig.savefig(ROOT/f'inference-cost-passk-20261004.{extension}',dpi=180)
    svg = ROOT/'inference-cost-passk-20261004.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
