#!/usr/bin/env python3
"""Plot historical local-browser GPT-4.1 baseline vs all three ARM variants."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
baseline_i=[20,30,40,50,60,70,80,90]
baseline_s=[95,96,100,105,105,103,114,101]
baseline_v=[232,248,231,234,230,229,229,222]
arm_i=[20,30,40,50,60]
arm_s=[90,107,92,113,100]
arm_v=[222,225,222,233,229]
arm_i += [70,80,90,100]
arm_s += [106,92,101,107]
arm_v += [233,218,217,222]
add_i=[20,30,40,50,60,70,80,90,100]
add_s=[85,103,102,98,90,112,111,118,109]
add_v=[236,219,218,212,215,222,212,216,217]
original_i=[51,70,80]
original_s=[102,103,100]
original_v=[223,231,222]
repo=Path(__file__).resolve().parents[1]
out=repo/'openwebrl/docs/rl_results/baseline_vs_arm_allfailure_full300.png'
plt.style.use('seaborn-v0_8-whitegrid')
fig,ax=plt.subplots(figsize=(8.8,5.0),dpi=180)
ax.plot(baseline_i,[100*x/300 for x in baseline_s],marker='o',lw=2.2,color='#2563eb',label='Outcome-only baseline · overall')
ax.plot(baseline_i,[100*x/y for x,y in zip(baseline_s,baseline_v)],marker='s',lw=2.0,color='#60a5fa',label='Outcome-only baseline · valid-only')
ax.plot(arm_i,[100*x/300 for x in arm_s],marker='o',lw=2.2,ls='--',color='#dc2626',label='All-failure ARM · overall')
ax.plot(arm_i,[100*x/y for x,y in zip(arm_s,arm_v)],marker='s',lw=2.0,ls='--',color='#f97316',label='All-failure ARM · valid-only')
ax.plot(add_i,[100*x/300 for x in add_s],marker='o',lw=2.2,ls=':',color='#16a34a',label='Additive ARM · overall')
ax.plot(add_i,[100*x/y for x,y in zip(add_s,add_v)],marker='s',lw=2.0,ls=':',color='#84cc16',label='Additive ARM · valid-only')
ax.plot(original_i,[100*x/300 for x in original_s],marker='o',lw=2.2,ls='-.',color='#7c3aed',label='Original ARM · overall')
ax.plot(original_i,[100*x/y for x,y in zip(original_s,original_v)],marker='s',lw=2.0,ls='-.',color='#c084fc',label='Original ARM · valid-only')
ax.set_xlabel('Checkpoint after training iteration'); ax.set_ylabel('Success rate (%)')
ax.set_title('Outcome-only baseline vs ARM variants\nGPT-4.1 · 300 Online-Mind2Web tasks')
ax.set_xlim(18,102); ax.set_ylim(20,58); ax.set_xticks([20,30,40,50,60,70,80,90,100])
ax.legend(frameon=True,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=8); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.tight_layout(); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,bbox_inches='tight'); print(out)
