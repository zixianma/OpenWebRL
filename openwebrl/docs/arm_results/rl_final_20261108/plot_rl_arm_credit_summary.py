"""Forest plot: paired iteration-90 gain over outcome-only 48 (shared Terra, Nov 9-10) with the SFT reference (Nov 11)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
d90 = json.loads((HERE/'iter90_terra_sameday_20261111.json').read_text())['diffs']
d0 = json.loads((HERE/'iter0_terra_anchor_20261111.json').read_text())['diffs']
NAMES = {'g40m16f': 'Gate B 40M+16F', 'gateb': 'Gate B (ours)', 'g48m16f': 'Gate B 48M+16F', 'additive': 'Additive (strict gate)',
         'mixedbonus': 'Mixed-only bonus', 'mixedreweight': 'Mixed-only reweight', 'allfailure': 'Failures in main batch',
         'outcome56': 'Outcome-only 56'}
rows = sorted(((NAMES[k], d90[f'{k}-outcome48']) for k in NAMES), key=lambda r: -r[1]['diff_pp'])
sft = -d0['outcome48-sft0']['diff_pp']
grey, blue = '#9a9a92', '#2b6fd6'
fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=200)
for i, (name, r) in enumerate(rows):
    c = blue if name.startswith('Gate B (ours)') else grey
    ax.plot(r['ci95'], [i, i], color=c, lw=2, solid_capstyle='round')
    ax.plot(r['diff_pp'], i, 'o', color=c, ms=7)
    ax.text(r['ci95'][1] + 0.3, i, f"{r['diff_pp']:+.1f}", va='center', fontsize=9, color='#555')
ax.axvline(0, color='#888', lw=1)
ax.axvline(sft, color='#888', lw=1, ls=(0, (4, 3)))
ax.text(sft + 0.2, -0.85, f'SFT start, iteration 0: {sft:+.1f} pp (Nov 11 same-day run)', fontsize=8.5, color='#555', bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.get_yticklabels()[[r[0] for r in rows].index('Gate B (ours)')].set_fontweight('bold')
ax.invert_yaxis()
ax.set_ylim(len(rows) - 0.5, -1.2)
ax.set_xlim(-6, 11)
ax.set_xlabel('Gain in overall success vs outcome-only 48 (pp), 95% task-bootstrap interval')
sig = sum(1 for n, r in rows if n != 'Outcome-only 56' and r['ci95'][0] > 0)
ax.set_title(f'{sig} of {len(rows) - 1} ARM variants beat outcome-only 48 at iteration 90', loc='left', fontsize=11)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.grid(axis='x', color='#eee')
ax.set_axisbelow(True)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(HERE/f'rl_arm_credit_summary_forest.{ext}', bbox_inches='tight')
