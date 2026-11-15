"""Actor-relative success gains with supplied 95% CIs; no interval recomputation.
Absolute success rates are kept separately in the source JSON. Teacher SFT is a
shared control reused in both tables. No sample size, pairing or interval
construction method is inferred from the confidence interval endpoints.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/openwebrl-offline-distillation-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
DOCS=next(p for p in HERE.parents if (p/'paper_plot_style.py').is_file())
sys.path.insert(0,str(DOCS))
import paper_plot_style as _style
from paper_plot_palette import COLORS as C
SOURCE=HERE/'offline_distillation_actor_differences.json'
D=json.loads(SOURCE.read_text())
METHODS=D['methods']
COLORS={'Teacher SFT':'#AEB3BA','Filtered SFT':'#7F8790','DPO':'#B69AE8','RPO':'#8246E5'}
BENCHES=[('webvoyager','WebVoyager'),('online_mind2web','OM2W'),('deepshop','DeepShop')]
plt.rcParams.update({'font.family':'Albert Sans','font.size':8.5,'axes.labelsize':8.5,
                     'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8.5,
                     'svg.fonttype':'none','pdf.fonttype':42})
REPORT={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'figures':{}}

def panel(ax,table,benchmark,title,*,show_methods=False,mean=False):
    ax.axvline(0,color=C['MUTED'],linestyle=(0,(3,3)),linewidth=.8,zorder=0)
    points=[]
    for i,method in enumerate(METHODS):
        value,lo,hi=D['tables'][table]['contrasts'][method][benchmark]
        assert lo<=value<=hi
        y=len(METHODS)-1-i;color=COLORS[method]
        bars=ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt='o',color=color,
                        markersize=5.5,elinewidth=1.1,capsize=2.4,capthick=.95,
                        markeredgecolor='white',markeredgewidth=.6,zorder=3)
        assert abs(float(bars.lines[0].get_xdata()[0])-value)<1e-12
        if mean:
            ax.text(14.5,y,f'{value:+.1f}',ha='right',va='center',color=color,fontsize=8.2)
        points.append({'table':table,'method':method,'benchmark':benchmark,'estimate_pp':value,'ci95_pp':[lo,hi]})
    ax.set_title(title,loc='left',pad=9)
    ax.set_ylim(-.48,3.48)
    ax.set_yticks([3,2,1,0],METHODS if show_methods else ['']*4)
    ax.tick_params(axis='y',length=0,pad=7)
    ax.tick_params(axis='x',length=3,width=.6,pad=3,colors=C['MUTED'])
    ax.set_xlim((-9,15) if mean else (-18,16))
    ax.set_xticks([-5,0,5,10] if mean else [-15,0,15])
    ax.grid(False)
    for spine in ['top','right','left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(C['MUTED']);ax.spines['bottom'].set_linewidth(.6)
    return points

def save(fig,stem,points):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    boxes=[]
    for ax in fig.axes:
        texts=[ax._left_title,ax.xaxis.label,ax.yaxis.label,*ax.texts,
               *ax.get_yticklabels(),*ax.get_xticklabels()]
        for text in texts:
            if text.get_visible() and text.get_text():
                b=text.get_window_extent(renderer)
                assert fig.bbox.contains(*b.p0) and fig.bbox.contains(*b.p1),text.get_text()
                boxes.append(b)
    for i,a in enumerate(boxes):
        assert all(not a.overlaps(b) for b in boxes[i+1:]),'Text overlaps'
    for ext in ['png','svg','pdf']:
        p=HERE/f'{stem}.{ext}';fig.savefig(p,dpi=300,bbox_inches='tight',pad_inches=.035)
        if ext=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    REPORT['figures'][stem]={'points':points,'output_sha256':{ext:hashlib.sha256((HERE/f'{stem}.{ext}').read_bytes()).hexdigest() for ext in ['png','svg','pdf']}}
    plt.close(fig)

if __name__=='__main__':
    fig,axes=plt.subplots(1,2,figsize=(6.8,2.05))
    points=[]
    for i,(key,title) in enumerate([('table7','a) ADM-4B (Table 7)'),('table_a4','b) Lookahead ADM-4B (Table A4)')]):
        points+=panel(axes[i],key,'mean',title,show_methods=i==0,mean=True)
    fig.supxlabel('Mean success gain over actor (pp)',x=.58,y=.035,fontsize=8.5)
    fig.subplots_adjust(left=.17,right=.99,bottom=.25,top=.80,wspace=.20)
    save(fig,'offline_distillation_mean_gain',points)
    for table in ['table7','table_a4']:
        fig,axes=plt.subplots(1,3,figsize=(6.8,2.05));points=[]
        for i,(bench,title) in enumerate(BENCHES):
            points+=panel(axes[i],table,bench,title,show_methods=i==0)
        fig.supxlabel('Success gain over actor (pp)',x=.58,y=.035,fontsize=8.5)
        fig.subplots_adjust(left=.17,right=.99,bottom=.25,top=.80,wspace=.20)
        save(fig,f'offline_distillation_{table}_by_benchmark',points)
    (HERE/'offline_distillation_plot_audit.json').write_text(json.dumps(REPORT,indent=2)+'\n')
    print('Rendered 3 figures; all 32 displayed contrasts match supplied estimates and CIs.')
