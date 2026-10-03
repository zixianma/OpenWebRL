#!/usr/bin/env python3
"""Plot local RL curves and separately labeled stealth evaluation points."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from plot_arm_interactive import build_data, main as write_interactive
baseline_i=[20,30,40,50,60,70,80,90,100]
baseline_s=[95,96,100,105,105,103,114,101,104]
baseline_v=[232,248,231,234,230,229,229,222,227]
arm_i=[20,30,40,50,60]
arm_s=[90,107,92,113,100]
arm_v=[222,225,222,233,229]
arm_i += [70,80,90,100]
arm_s += [106,92,101,107]
arm_v += [233,218,217,222]
add_i=[20,30,40,50,60,70,80,90,100]
add_s=[85,103,102,98,90,112,111,118,109]
add_v=[236,219,218,212,215,222,212,216,217]
repo=Path(__file__).resolve().parents[1]
gate_b_i=[20,30,40,50,60,70,80,90,100]
gate_b_results=[json.loads((repo/f'openwebrl/docs/arm_results/rl_integration/gate-b-iteration{i}-audit.json').read_text())['full300'] for i in gate_b_i]
assert all(r['tasks']==300 for r in gate_b_results)
out=repo/'openwebrl/docs/rl_results/baseline_vs_arm_allfailure_full300.png'
plt.style.use('seaborn-v0_8-whitegrid')
fig,ax=plt.subplots(figsize=(8.8,8.0),dpi=180)
ax.plot(baseline_i,[100*x/300 for x in baseline_s],marker='o',lw=2.2,color='#2563eb',label='Outcome-only baseline · overall')
ax.plot(baseline_i,[100*x/y for x,y in zip(baseline_s,baseline_v)],marker='s',lw=2.0,color='#60a5fa',label='Outcome-only baseline · valid-only')
ax.plot(arm_i,[100*x/300 for x in arm_s],marker='o',lw=2.2,ls='--',color='#dc2626',label='All-failure ARM · overall')
ax.plot(arm_i,[100*x/y for x,y in zip(arm_s,arm_v)],marker='s',lw=2.0,ls='--',color='#f97316',label='All-failure ARM · valid-only')
ax.plot(add_i,[100*x/300 for x in add_s],marker='o',lw=2.2,ls=':',color='#16a34a',label='Additive ARM · overall')
ax.plot(add_i,[100*x/y for x,y in zip(add_s,add_v)],marker='s',lw=2.0,ls=':',color='#84cc16',label='Additive ARM · valid-only')
ax.plot(gate_b_i,[100*r['successes']/r['tasks'] for r in gate_b_results],marker='o',lw=2.2,ls='-.',color='#7e22ce',label='Gate B (relaxed gate) · overall')
ax.plot(gate_b_i,[100*r['successes']/r['valid'] for r in gate_b_results],marker='s',lw=2.0,ls='-.',color='#c026d3',label='Gate B (relaxed gate) · valid-only')
# Use the same audited counts and colors as the interactive companion. Each
# mixed-only lineage stops at its last evaluated checkpoint, currently90.
mixed_styles = {'mixed_bonus': (0, (5, 2)), 'mixed_reweight': (0, (3, 1, 1, 1))}
for series in build_data()['series']:
    if series['id'] not in mixed_styles:
        continue
    points = series['points']
    iterations = [p['iteration'] for p in points]
    for metric, denominator, marker in [('overall', 'tasks', 'o'), ('valid-only', 'valid', 's')]:
        ax.plot(iterations, [100*p['successes']/p[denominator] for p in points],
                marker=marker, markersize=5, lw=2.2,
                ls=mixed_styles[series['id']], color=series['color'],
                markerfacecolor=series['color'] if metric == 'overall' else 'white',
                alpha=1 if metric == 'overall' else .75,
                label=f"{series['label']} · {metric}")
# Corrected matched actor-only stealth cohort: o4-mini/T0.6. The earlier
# GPT-4.1/T0 runs stay in the history tables rather than this comparison plot.
for method,label,color,offset in [('baseline','Baseline','#2563eb',-16),('gate-b','Gate B','#7e22ce',16),('additive','Additive','#16a34a',0)]:
    path=repo/f'openwebrl/docs/arm_results/rl_integration/stealth-o4-t06-{method}-iteration90-audit.json'
    if not path.exists():
        continue
    r=json.loads(path.read_text())['full300']
    assert r['tasks']==300 and r['verified_complete']
    position=ax.transData+ScaledTranslation(offset/72,0,fig.dpi_scale_trans)
    ax.scatter([90],[100*r['successes']/300],transform=position,marker='D',s=85,color=color,zorder=7,label=f'Matched stealth {label} · overall')
    ax.scatter([90],[100*r['successes']/r['valid']],transform=position,marker='D',s=85,facecolors='white',edgecolors=color,linewidths=1.8,zorder=7,label=f'Matched stealth {label} · valid-only')
ax.text(10,21,'Stealth iteration90 symbols offset horizontally for visibility.',fontsize=8,color='#64748b')
ax.set_xlabel('Checkpoint after training iteration'); ax.set_ylabel('Success rate (%)')
ax.set_title('Outcome-only baseline vs ARM variants · 300 tasks\nLocal: GPT-4.1, T0 | Stealth: o4-mini, T0.6')
ax.set_xlim(8,102); ax.set_ylim(20,66); ax.set_xticks([10,20,30,40,50,60,70,80,90,100])
handles,labels=ax.get_legend_handles_labels()
order=list(range(0,len(handles),2))+list(range(1,len(handles),2))
ax.legend([handles[i] for i in order],[labels[i] for i in order],frameon=True,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2,fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.tight_layout(); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,bbox_inches='tight'); print(out)

# Keep the downloadable interactive companion current with each plot refresh.
write_interactive()
