from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
iterations=[20,30,40,50,60,70]
successes=[90,107,92,113,100,106]
valid=[222,225,222,233,229,233]
overall=[100*s/300 for s in successes]
valid_only=[100*s/v for s,v in zip(successes,valid)]
repo=Path('/gpfs/projects/krishna/zixianma/OpenWebRL')
out=repo/'openwebrl/docs/rl_results/arm_allfailure_full300.png'
plt.style.use('seaborn-v0_8-whitegrid')
fig,ax=plt.subplots(figsize=(8.2,4.8),dpi=180)
ax.plot(iterations,overall,marker='o',linewidth=2.2,markersize=5,color='#2563eb',label='Overall success')
ax.plot(iterations,valid_only,marker='s',linewidth=2.0,markersize=4.5,color='#dc2626',label='Valid-only success')
ax.set_xlabel('Checkpoint after training iteration')
ax.set_ylabel('Success rate (%)')
ax.set_title('All-failure ARM browser evaluation\nGPT-4.1 · 300 Online-Mind2Web tasks')
ax.set_ylim(20,55); ax.set_xlim(18,72); ax.set_xticks([20,30,40,50,60,70])
ax.legend(frameon=True,loc='lower right'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.tight_layout(); fig.savefig(out,bbox_inches='tight'); print(out)
