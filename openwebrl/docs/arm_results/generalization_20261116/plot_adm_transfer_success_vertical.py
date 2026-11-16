#!/usr/bin/env python3
"""Compact vertical version of the user-supplied cross-environment results.

Same input values and role colors as the horizontal figure. Group labels and
one shared legend replace repeated system labels. Absolute rates above bars;
ADM gains appear above the absolute rates relative to their corresponding base.
"""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter
HERE=Path(__file__).resolve().parent
DOCS=next(p for p in HERE.parents if (p/'paper_plot_style.py').exists())
sys.path.insert(0,str(DOCS))
import paper_plot_style  # noqa: E402,F401
from paper_plot_palette import COLORS as C  # noqa: E402
D=json.loads((HERE/'adm_transfer_success.json').read_text())
COLORS={'standalone':C['PRE_EXEC'],'base':C['PRE_EXEC'],
        'adm':C['ADM_MAIN'],'sol':C['STATE_BORDER']}
fig,axes=plt.subplots(1,3,figsize=(7.2,1.9),gridspec_kw={'width_ratios':[1.15,1,1]})
for i,(ax,(bench,rows)) in enumerate(zip(axes,D['benchmarks'].items())):
    if i==0:
        xs=[0,1.2,3.0,4.0,5.0]
        ticks,labels=[0,1.2,4.0],['Qwen3.8\n27B','GPT-6.1\nSol','Qwen3.5-9B\nSFT']
        ymax,yticks=85,[0,40,80]
        ax.set_xlim(-.65,5.65)
    else:
        xs=[0,1,2,3.8,4.8,5.8]
        ticks,labels=[1,4.8],['Qwen3.5-4B','GiGPO']
        ymax,yticks=(105,[0,50,100]) if i==1 else (50,[0,25,50])
        ax.set_xlim(-.65,6.45)
    for x,row in zip(xs,rows):
        ax.bar(x,row['success'],width=.72,color=COLORS[row['role']],edgecolor='none',zorder=3)
        ax.text(x,row['success']+ymax*.025,f"{row['success']:.1f}",ha='center',va='bottom',
                fontsize=6.4,color=C['INK'])
        if row['role']=='adm':
            base=next(r for r in rows if r['role']=='base' and r['family']==row['family'])
            assert abs(row['success']-base['success']-row['gain'])<1e-8
            ax.annotate(f"+{row['gain']:.1f}",(x,row['success']+ymax*.025),
                        xytext=(0,11),textcoords='offset points',
                        ha='center',va='bottom',fontsize=6.3,fontweight='bold',
                        color=C['ACTION_INK'],zorder=4,annotation_clip=False)
    ax.set_title(f'{chr(97+i)}) {bench}',fontsize=8,loc='left',pad=7,color=C['INK'])
    ax.set_ylim(0,ymax);ax.set_yticks(yticks)
    ax.set_xticks(ticks,labels,fontsize=6.4)
    ax.tick_params(axis='x',length=0,pad=4,colors=C['INK'])
    ax.tick_params(axis='y',length=2.5,pad=2,labelsize=6.4,colors=C['INK'])
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    ax.grid(False)
    for side in ('top','right'):
        ax.spines[side].set_visible(False)
    for side in ('left','bottom'):
        ax.spines[side].set_color(C['BORDER'])
axes[0].set_ylabel('Success (%)',fontsize=7.2,labelpad=5,color=C['INK'])
legend=fig.legend([Patch(facecolor=COLORS[k],edgecolor='none') for k in ('base','adm','sol')],
                  ['Base / standalone','+ ADM-4B (gain vs. base)','+ GPT-6.1 Sol'],ncol=3,frameon=False,
                  loc='lower center',bbox_to_anchor=(.52,-.005),fontsize=7,
                  handlelength=1,handletextpad=.5,columnspacing=1.8)
fig.subplots_adjust(left=.065,right=.99,bottom=.31,top=.80,wspace=.27)
fig.canvas.draw();r=fig.canvas.get_renderer()
for ax in axes:
    assert not legend.get_window_extent(r).overlaps(ax.get_tightbbox(r))
    labels=[ax._left_title,ax.yaxis.label,*ax.texts,*ax.get_xticklabels(),*ax.get_yticklabels()]
    for t in labels:
        if t.get_visible() and t.get_text():
            box=t.get_window_extent(r)
            assert fig.bbox.contains(*box.p0) and fig.bbox.contains(*box.p1),t.get_text()
    for j,t in enumerate(ax.texts):
        assert all(not t.get_window_extent(r).overlaps(q.get_window_extent(r)) for q in ax.texts[:j])
for ext in ('png','svg','pdf'):
    p=HERE/f'adm_transfer_success_vertical.{ext}'
    meta={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None} if ext=='pdf' else {}
    fig.savefig(p,dpi=300,metadata=meta)
    if ext=='svg':
        p.write_text('\n'.join(l.rstrip() for l in p.read_text().splitlines())+'\n')
plt.close(fig)
