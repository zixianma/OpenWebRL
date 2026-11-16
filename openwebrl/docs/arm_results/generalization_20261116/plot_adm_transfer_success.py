#!/usr/bin/env python3
"""Reported transfer success: OSWorld-Verified, ALFWorld and WebShop.

User-supplied rates and gains only; no denominators or uncertainty inferred.
Horizontal bars preserve names, full zero-based success axes and group identity.
"""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter

HERE = Path(__file__).resolve().parent
DOCS = next(p for p in HERE.parents if (p/'paper_plot_style.py').exists())
sys.path.insert(0,str(DOCS))
import paper_plot_style  # noqa: E402,F401
from paper_plot_palette import COLORS as C  # noqa: E402
D = json.loads((HERE/'adm_transfer_success.json').read_text())
COLORS = {'base':C['PRE_EXEC'],'standalone':C['PRE_EXEC'],
          'adm':C['ADM_MAIN'],'sol':C['STATE_BORDER']}
fig=plt.figure(figsize=(7.6,2.85))
axes=[fig.add_axes(rect) for rect in ((.16,.22,.22,.63),(.515,.22,.185,.63),(.790,.22,.190,.63))]
TITLE_SIZE, LABEL_SIZE, VALUE_SIZE = 8.2, 7.0, 7.0
for i,(ax,(bench,rows)) in enumerate(zip(axes,D['benchmarks'].items())):
    if i==0:
        ys=[5.15,4.20,2.15,1.20,.25]
        labels=['Qwen3.8-27B','GPT-6.1 Sol','Base actor','+ ADM-4B','+ GPT-6.1 Sol']
        groups=[(5.85,'Standalone systems'),(2.85,'Qwen3.5-9B SFT')]
        xmax,ticks=100,[0,25,50,75,100]
    else:
        ys=[5.15,4.20,3.25,1.40,.45,-.50]
        labels=['Base actor','+ ADM-4B','+ GPT-6.1 Sol']*2
        groups=[(5.85,'Qwen3.5-4B'),(2.10,'GiGPO')]
        xmax,ticks=(100,[0,50,100]) if i==1 else (66,[0,20,40,60])
    for y,row in zip(ys,rows):
        color=COLORS[row['role']]
        ax.barh(y,row['success'],height=.58,color=color,edgecolor='none',zorder=3)
        label=f"{row['success']:.1f}"
        if 'gain' in row:
            baseline=next(r for r in rows if r['role']=='base' and r.get('family')==row['family'])
            assert abs(row['success']-baseline['success']-row['gain'])<.051
            label+=f" (+{row['gain']:.1f})"
        ax.text(row['success']+.018*xmax,y,label,ha='left',va='center',
                fontsize=VALUE_SIZE,color=C['INK'])
    for y,label in groups:
        ax.text(0,y,label,fontsize=7.0,ha='left',va='center',color=C['MUTED'])
    ax.set_title(f"{chr(97+i)}) {bench}",fontsize=TITLE_SIZE,loc='left',pad=11,color=C['INK'])
    ax.set_yticks(ys,labels if i<2 else ['']*len(ys),fontsize=LABEL_SIZE)
    ax.tick_params(axis='y',length=0,pad=5,colors=C['INK'])
    ax.tick_params(axis='x',labelsize=6.8,length=2.5,pad=3,colors=C['INK'])
    ax.set_xlim(0,xmax);ax.set_ylim(-1.15,6.25)
    ax.set_xticks(ticks);ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    ax.set_xlabel('Success (%)',fontsize=7.4,labelpad=4,color=C['INK'])
    ax.grid(False)
    for side in ('top','right','left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(C['BORDER'])
handles=[Patch(facecolor=COLORS[k],edgecolor='none') for k in ('base','adm','sol')]
legend=fig.legend(handles,['Base / standalone','+ ADM-4B','+ GPT-6.1 Sol'],ncol=3,
                  loc='lower center',bbox_to_anchor=(.55,.01),fontsize=7.5,
                  frameon=False,handlelength=1.0,handletextpad=.5,columnspacing=2)
fig.canvas.draw()
r=fig.canvas.get_renderer()
for ax in axes:
    assert not legend.get_window_extent(r).overlaps(ax.get_tightbbox(r))
    for t in [ax._left_title,ax.xaxis.label,*ax.texts,*ax.get_yticklabels(),*ax.get_xticklabels()]:
        if t.get_visible() and t.get_text():
            b=t.get_window_extent(r)
            assert fig.bbox.contains(*b.p0) and fig.bbox.contains(*b.p1),t.get_text()
for ext in ('png','svg','pdf'):
    p=HERE/f'adm_transfer_success.{ext}'
    meta={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None} if ext=='pdf' else {}
    fig.savefig(p,dpi=300,metadata=meta)
    if ext=='svg':
        p.write_text('\n'.join(l.rstrip() for l in p.read_text().splitlines())+'\n')
plt.close(fig)
