"""Matched Table 7 dot and horizontal-bar variants, with actor-relative 95% CIs.
Both versions show the supplied mean and all three benchmarks. No absolute-rate
confidence intervals are inferred from these actor-relative contrasts.
"""
import hashlib
import json
from pathlib import Path
import runpy
HERE=Path(__file__).resolve().parent
M=runpy.run_path(str(HERE/'plot_offline_distillation.py'))
plt,D,C,METHODS,COLORS=M['plt'],M['D'],M['C'],M['METHODS'],M['COLORS']
REPORT={'source_sha256':hashlib.sha256(M['SOURCE'].read_bytes()).hexdigest(),'figures':{}}
PANELS=[('mean','Mean'),*M['BENCHES']]

if __name__=='__main__':
    for kind in ['dot','bar']:
        fig,axes=plt.subplots(1,4,figsize=(7.2,2.05))
        points=[]
        for j,(ax,(bench,title)) in enumerate(zip(axes,PANELS)):
            ax.axvline(0,color=C['MUTED'],linestyle=(0,(3,3)),linewidth=.8,zorder=0)
            for i,method in enumerate(METHODS):
                value,lo,hi=D['tables']['table7']['contrasts'][method][bench]
                y=len(METHODS)-1-i;color=COLORS[method]
                if kind=='bar':
                    bars=ax.barh(y,value,height=.50,color=color,alpha=.72,zorder=2)
                    assert float(bars.patches[0].get_width())==value
                eb=ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt='o' if kind=='dot' else 'none',
                               color=color,markersize=5.3,elinewidth=1.05,capsize=2.4,capthick=.95,
                               markeredgecolor='white',markeredgewidth=.6,zorder=3)
                points.append({'table':'table7','method':method,'benchmark':bench,
                               'estimate_pp':value,'ci95_pp':[lo,hi]})
            ax.set_title(title,loc='left',pad=8,fontsize=8.5)
            ax.set_ylim(-.48,3.48)
            ax.set_yticks([3,2,1,0],METHODS if j==0 else ['']*4)
            ax.tick_params(axis='y',length=0,pad=7,labelsize=8)
            ax.tick_params(axis='x',length=3,width=.6,pad=3,colors=C['MUTED'],labelsize=7.5)
            ax.set_xlim((-6,11) if bench=='mean' else (-18,16))
            ax.set_xticks([-5,0,5,10] if bench=='mean' else [-15,0,15])
            ax.grid(False)
            for side in ['top','right','left']:ax.spines[side].set_visible(False)
            ax.spines['bottom'].set_color(C['MUTED']);ax.spines['bottom'].set_linewidth(.6)
        fig.supxlabel('Success gain over actor (pp)',x=.57,y=.035,fontsize=8.5)
        fig.subplots_adjust(left=.16,right=.99,bottom=.25,top=.80,wspace=.30)
        stem=f'offline_distillation_table7_{kind}'
        M['save'](fig,stem,points)
        REPORT['figures'][stem]=M['REPORT']['figures'][stem]
    a,b=[f['points'] for f in REPORT['figures'].values()]
    assert a==b and len(a)==16
    (HERE/'offline_distillation_table7_plot_audit.json').write_text(json.dumps(REPORT,indent=2)+'\n')
    print('Rendered Table 7 dot and bar plots; same 16 estimates and actor-relative CIs in both.')
