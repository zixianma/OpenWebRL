from pathlib import Path
import os,json,hashlib
os.environ['MPLCONFIGDIR']='/tmp/openwebrl-api-tradeoff-matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter,FuncFormatter,NullFormatter
P=Path(__file__).resolve().parent.parent
D=json.loads((P/'om2w_harness_comparison_valid_only.json').read_text())
OUT=Path(__file__).resolve().parent;OUT.mkdir(exist_ok=True)
M={'serial_high':('Sol single','#0072B2','s'),'luna_single_high':('Luna single','#0072B2','o'),'parallel3_select_high':('Luna N3','#D6AA00','o'),'parallel5_select_high':('Luna N5','#E87500','o')}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,'axes.titlesize':15,'svg.fonttype':'none','pdf.fonttype':42})
fig,axes=plt.subplots(2,3,figsize=(15.8,9.5),sharey=True)
fig.subplots_adjust(left=.075,right=.98,top=.81,bottom=.24,hspace=.90,wspace=.27)
fig.text(.075,.965,'Online-Mind2Web',fontsize=23,weight='bold',va='top')
fig.text(.075,.922,'All methods high · 300 attempted tasks per cell · success among valid outcomes',fontsize=12,color='#535F6B')
handles=[Line2D([],[],linestyle='',marker=m,color=c,markersize=8,label=l)for l,c,m in M.values()]
# Direct point labels replace a separate method legend.
metrics=[('mean_model_usd_per_task','Model cost','USD / attempted task · log scale'),('mean_browser_steps','Browser commands','Recorded command executions / task'),('mean_task_seconds','Latency','Recorded task seconds · log scale')]
for ri,(h,hlabel) in enumerate([('browsercode','BrowserCode'),('native','Native CUAWright')]):
 rows=D[h]['tradeoffs'];pos=axes[ri,0].get_position();fig.text(.075,pos.y1+.055,hlabel,fontsize=17,weight='bold',color='#243241')
 for ci,(metric,title,xlabel) in enumerate(metrics):
  ax=axes[ri,ci];ax.set_ylim(59,81);ax.set_yticks([60,65,70,75,80]);ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax.grid(color='#E5E9EE',linewidth=.8);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
  for side in ['left','bottom']:ax.spines[side].set_color('#BBC4CE')
  ax.set_title(title,loc='left',pad=12);ax.set_xlabel(xlabel,labelpad=9)
  if ci==0:ax.set_ylabel('Valid-only success (%)')
  vals=[v[metric]for v in rows.values()]
  if ci in [0,2]:
   ax.set_xscale('log');ax.set_xlim(min(vals)/1.5,max(vals)*1.75);ax.set_xticks(vals);ax.xaxis.set_minor_formatter(NullFormatter());ax.xaxis.set_major_formatter(FuncFormatter((lambda x,p:f'${x:.3g}')if ci==0 else(lambda x,p:f'{x:,.0f}')))
   if ci==2 and ri==0:ax.set_xticks([90,150,220])
   if ci==2 and ri==1:ax.set_xticks([500,900,1300])
   if ci==0 and ri==0:ax.set_xticks([.01,.05,.2])
   if ci==0 and ri==1:ax.set_xticks([.1,.3,1])
  else:
   # Per-harness ranges make within-harness differences visible; units stay identical.
   ax.set_xlim((12.8,16.5)if ri==0 else(27.5,39));ax.set_xticks([13,14,15,16]if ri==0 else[28,32,36,38])
  for method,(label,color,marker)in M.items():
   r=rows[method];assert abs(r['success_rate']-r['successes']/r['valid'])<1e-12
   x,y=r[metric],100*r['success_rate'];ax.scatter(x,y,s=85 if method.startswith('parallel')else 70,marker=marker,color=color,edgecolor='white',linewidth=.8,zorder=4)
   dx,dy=7,8
   if method=='serial_high':dx,dy=7,-15 if ri==0 else-13
   if method=='parallel3_select_high':dx,dy=7,-15 if ri==1 else 8
   if method=='parallel5_select_high':dx,dy=7,9
   if method=='luna_single_high':dx,dy=7,-16
   ha='left'
   if method=='serial_high' and ci==2 and ri==0:dx,dy=7,8
   if method=='parallel3_select_high' and ci==1 and ri==1:dx,dy,ha=-8,-16,'right'
   ax.annotate(label,(x,y),ha=ha,xytext=(dx,dy),textcoords='offset points',fontsize=10,color=color,weight='bold'if method.startswith('parallel')else'normal')
 # Concise honest comparison highlights the cost-performance tradeoff against Sol.
 base=rows['serial_high'];bits=[]
 for method,name in [('parallel3_select_high','N3'),('parallel5_select_high','N5')]:
  r=rows[method];delta=100*(r['success_rate']-base['success_rate']);saving=100*(1-r['mean_model_usd_per_task']/base['mean_model_usd_per_task']);bits.append(f'{name}: {delta:+.1f} pp success · {saving:.0f}% lower model cost')
 # Comparison is conveyed directly by the labeled points.
footer=('Success = successes / valid outcomes; valid subsets differ. Valid counts (Sol / Luna / N3 / N5): BrowserCode 266 / 269 / 269 / 267; native 286 / 271 / 270 / 270.\n'
'X-axis means include measured invalid attempts. Latency coverage: BrowserCode 275 / 274 / 275 / 273; native 300 / 300 / 300 / 298.\n'
'Command coverage: BrowserCode 300 / 300 / 293 / 294; native 300 / 300 / 300 / 298. Executions can contain multiple actions; granularity differs.\n'
'Cost coverage: 300 each. BrowserCode original receipts (N3: one unknown-usage call); native includes helpers and full reserves. External judging excluded.\n'
'Within-harness x-axis ranges differ. Collection windows, timing scopes and limits differ (BrowserCode 12,288 tokens / 30 decisions; native 20,000 / 100); descriptive comparison.')
fig.text(.075,.018,footer,fontsize=8.6,color='#697581',linespacing=1.7,va='bottom')
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'om2w_harness_facets.{ext}',dpi=200,bbox_inches='tight',pad_inches=.15)
print(OUT/'om2w_harness_facets.png')

f=OUT/'om2w_harness_facets.svg'
f.write_text('\n'.join(x.rstrip() for x in f.read_text().splitlines())+'\n')
