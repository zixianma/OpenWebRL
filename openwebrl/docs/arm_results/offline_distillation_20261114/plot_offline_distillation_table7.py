"""Table 7: actor-relative dot plot with 95% CIs and absolute-success bar graph.
The provided actor-relative CIs are not reused as absolute-success CIs. Bars add
the actor-only reference and use the absolute rates from the supplied table.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
HERE=Path(__file__).resolve().parent
M=runpy.run_path(str(HERE/'plot_offline_distillation.py'))
plt,D,C,METHODS,COLORS=M['plt'],M['D'],M['C'],M['METHODS'],M['COLORS']
AUDIT=HERE/'offline_distillation_table7_plot_audit.json'
REPORT=json.loads(AUDIT.read_text()) if AUDIT.exists() else {'figures':{}}
REPORT['source_sha256']=hashlib.sha256(M['SOURCE'].read_bytes()).hexdigest()
PANELS=[('mean','Mean'),*M['BENCHES']]

def render_mean():
    fig,ax=plt.subplots(figsize=(3.4,1.8))
    ax.axvline(0,color=C['MUTED'],linestyle=(0,(3,3)),linewidth=.8,zorder=0)
    points=[]
    for i,method in enumerate(METHODS):
        value,lo,hi=D['tables']['table7']['contrasts'][method]['mean']
        y=len(METHODS)-1-i;color=COLORS[method]
        ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt='o',color=color,
                    markersize=5.3,elinewidth=1.05,capsize=2.4,capthick=.95,
                    markeredgecolor='white',markeredgewidth=.6,zorder=3)
        ax.text(13.6,y,f'{value:+.1f}',color=color,ha='right',va='center',fontsize=7.5)
        points.append({'table':'table7','method':method,'benchmark':'mean',
                       'estimate_pp':value,'ci95_pp':[lo,hi]})
    ax.set_yticks([3,2,1,0],METHODS)
    ax.tick_params(axis='y',length=0,pad=6,labelsize=8)
    ax.tick_params(axis='x',length=3,width=.6,pad=3,colors=C['MUTED'],labelsize=7.5)
    ax.set_xlim(-6,14.5);ax.set_xticks([-5,0,5,10]);ax.set_ylim(-.48,3.48)
    ax.set_xlabel('Mean success gain over actor (pp)',fontsize=8,labelpad=5)
    ax.grid(False)
    for side in ['top','right','left']:ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(C['MUTED']);ax.spines['bottom'].set_linewidth(.6)
    fig.subplots_adjust(left=.30,right=.985,bottom=.28,top=.98)
    stem='offline_distillation_table7_mean_dot'
    M['save'](fig,stem,points)
    REPORT['figures'][stem]=M['REPORT']['figures'][stem]
    REPORT['figures'][stem]['metric']='mean overall success gain over actor (pp)'
    REPORT['figures'][stem]['intervals']='supplied paired 95% CIs versus actor-only'
    assert len(points)==4

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['dot','bar','both','mean'],default='both')
    args=parser.parse_args()
    if args.variant=='mean':
        render_mean()
    for kind in (['dot','bar'] if args.variant=='both' else [] if args.variant=='mean' else [args.variant]):
        absolute=kind=='bar'
        methods=['Actor only',*METHODS] if absolute else METHODS
        fig,axes=plt.subplots(1,4,figsize=(7.2,2.25 if absolute else 2.05))
        points=[]
        for j,(ax,(bench,title)) in enumerate(zip(axes,PANELS)):
            if not absolute:
                ax.axvline(0,color=C['MUTED'],linestyle=(0,(3,3)),linewidth=.8,zorder=0)
            for i,method in enumerate(methods):
                y=len(methods)-1-i
                color='#59626D' if method=='Actor only' else COLORS[method]
                if absolute:
                    value=D['tables']['table7']['overall_success_pct'][method][bench]
                    bars=ax.barh(y,value,height=.55,color=color,alpha=.82,zorder=2)
                    assert float(bars.patches[0].get_width())==value
                    ax.text(value+1,y,f'{value:.1f}',ha='left',va='center',color=C['MUTED'],fontsize=6.5)
                    points.append({'table':'table7','method':method,'benchmark':bench,'success_pct':value})
                else:
                    value,lo,hi=D['tables']['table7']['contrasts'][method][bench]
                    ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt='o',
                                color=color,markersize=5.3,elinewidth=1.05,capsize=2.4,capthick=.95,
                                markeredgecolor='white',markeredgewidth=.6,zorder=3)
                    points.append({'table':'table7','method':method,'benchmark':bench,
                                   'estimate_pp':value,'ci95_pp':[lo,hi]})
            ax.set_title(title,loc='left',pad=8,fontsize=8.5)
            ax.set_ylim(-.48,len(methods)-.52)
            ax.set_yticks(list(range(len(methods)-1,-1,-1)),methods if j==0 else ['']*len(methods))
            ax.tick_params(axis='y',length=0,pad=7,labelsize=8)
            ax.tick_params(axis='x',length=3,width=.6,pad=3,colors=C['MUTED'],labelsize=7.5)
            ax.set_xlim((0,57) if absolute else (-6,11) if bench=='mean' else (-18,16))
            ax.set_xticks([0,25,50] if absolute else [-5,0,5,10] if bench=='mean' else [-15,0,15])
            ax.grid(False)
            for side in ['top','right','left']:ax.spines[side].set_visible(False)
            ax.spines['bottom'].set_color(C['MUTED']);ax.spines['bottom'].set_linewidth(.6)
        fig.supxlabel('Success (%)' if absolute else 'Success gain over actor (pp)',x=.57,y=.035,fontsize=8.5)
        fig.subplots_adjust(left=.16,right=.99,bottom=.25,top=.80,wspace=.30)
        stem=f'offline_distillation_table7_{kind}'
        M['save'](fig,stem,points)
        REPORT['figures'][stem]=M['REPORT']['figures'][stem]
        REPORT['figures'][stem]['metric']='absolute overall success (%)' if absolute else 'overall success gain over actor (pp)'
        REPORT['figures'][stem]['intervals']='not supplied for absolute success; omitted' if absolute else 'supplied actor-relative 95% CIs'
        assert len(points)==(20 if absolute else 16)
    AUDIT.write_text(json.dumps(REPORT,indent=2)+'\n')
    print(f'Rendered Table 7 variant: {args.variant}.')
