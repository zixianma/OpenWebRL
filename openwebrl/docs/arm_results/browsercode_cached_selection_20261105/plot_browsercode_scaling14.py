#!/usr/bin/env python3
"""Standalone sanitized14-method BrowserCode renderer. No task IDs/runtime imports.

The private adapter must qualify source endpoints before producing aggregate.json.
This renderer verifies aggregate consistency; it does not certify native runs.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/openwebrl-browsercode14-mpl')
BENCHMARKS={'online_mind2web':300,'webvoyager':595,'deepshop':150}
BASE_METHODS=('serial_high','luna_single_high','parallel3_select_high','parallel5_select_high','luna_single_xhigh_12288','luna_single_high_24576','luna_single_high_6144','luna_single_high_60steps','parallel2_select_high','parallel5_select_high_60steps','luna_single_high_120steps')
METHODS=BASE_METHODS+('sol_single_high60','luna_n3_cached_high30','luna_n5_cached_high60')
METRICS=('actor_usd','browser_commands','actor_seconds')
PROTOCOL=dict(model='gpt-5.6-terra',reasoning_effort='medium',max_completion_tokens=4096,seed=42,version='gpt56-terra-terminal-success-v1')
BASELINE='One-pass gap-repaired eleven-method baseline; original valid outcomes and first prespecified repair outcomes preserved.'
CAVEATS=['Gap-repaired baseline, Sol high/60 and cached selection use different live collection windows. Matching task identity does not control collection time.','Own-valid points use each method valid cohort; common-valid points use the same newly intersected fourteen-method tasks within each benchmark.','Fixed planned weights300/1045,595/1045,150/1045 apply to success and all within-benchmark actor means; this is not pooled-valid success.','Actor pricing preserves frozen source usage/upper-bound definitions, excludes Terra and hosting, and is not an invoice. Latency is elapsed actor-loop seconds per task, including valid failures; parallel call times are not summed.','Historical Luna single/60 enforced60 but advertised30; single/120 advertised120. Cached selectors retain full native context; uncached selectors use their historical protocol.','A browser-code batch is one selected nonempty code batch and may contain multiple browser/CDP operations. No hosting-cost panel or significance claim is inferred.']
STYLES={
 'serial_high':('Sol high /30','s','#08519c',True),
 'luna_single_high':('Luna high /30','o','#4292c6',True),
 'parallel3_select_high':('Luna N3 /30','o','#ef8a8a',True),
 'parallel5_select_high':('Luna N5 /30','o','#a50f15',True),
 'luna_single_xhigh_12288':('Luna xhigh /30','o','#08306b',True),
 'luna_single_high_24576':('Luna high 2x tokens','o','#2171b5',True),
 'luna_single_high_6144':('Luna high 0.5x tokens','o','#9ecae1',True),
 'luna_single_high_60steps':('Luna high /60','o','#4292c6',False),
 'parallel2_select_high':('Luna N2 /30','o','#fb6a4a',True),
 'parallel5_select_high_60steps':('Luna N5 /60','o','#a50f15',False),
 'luna_single_high_120steps':('Luna high /120','o','#08519c',False),
 'sol_single_high60':('Sol high /60','s','#08519c',False),
 'luna_n3_cached_high30':('Luna N3 /30 cached','o','#ef8a8a',True),
 'luna_n5_cached_high60':('Luna N5 /60 cached','o','#a50f15',False)}
CACHED=set(METHODS[-2:])
def need(v,message):
 if not v:raise ValueError(message)
def number(v):return type(v)in(int,float)and math.isfinite(v)
def integer(v):return type(v)is int and v>=0
def digest(v):return type(v)is str and len(v)==64 and all(c in'0123456789abcdef'for c in v)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cap(method):return 120 if method=='luna_single_high_120steps'else 60 if method in('luna_single_high_60steps','parallel5_select_high_60steps','sol_single_high60','luna_n5_cached_high60')else 30
def validate(d):
 fields={'schema','baseline','methods','judge_protocol','benchmark_weights','common_valid_counts','own_valid_points','common_valid_points','source_sha256','comparison_caveats'}
 need(type(d)is dict and set(d)==fields,'Only documented sanitized aggregate fields; never task IDs, private paths or audits')
 need(d['schema']=='browsercode-fourteen-method-scaling-v1'and d['baseline']==BASELINE,'Explicit gap-repaired baseline required')
 need(d['methods']==list(METHODS)and d['judge_protocol']==PROTOCOL,'Exact fourteen methods and shared Terra protocol')
 need(d['benchmark_weights']=={b:str(n)+'/1045'for b,n in BENCHMARKS.items()}and d['comparison_caveats']==CAVEATS,'Fixed weights and complete comparability caveats')
 need(set(d['source_sha256'])=={'gap_repaired_aggregate','sol60_aggregate','cached_aggregate','cached_native_seal','private_projection_inputs'}and all(digest(v)for v in d['source_sha256'].values()),'Only anonymous source digests allowed')
 need(set(d['common_valid_counts'])==set(BENCHMARKS),'All common-valid benchmark counts')
 expected={(b,m)for b in BENCHMARKS for m in METHODS};own={}
 for population in ['own_valid','common_valid']:
  points=d[population+'_points'];need(type(points)is list and len(points)==42,'Exact42-point matrix per population');seen=set()
  for p in points:
   key=p.get('benchmark'),p.get('method');need(key in expected and key not in seen,'Unique canonical benchmark/method point');seen.add(key)
   if population=='own_valid':
    need(set(p)=={'benchmark','method','planned','recorded','valid','invalid','successes','success_rate','metrics'},'Only own-valid aggregate fields')
    need(all(integer(p[k])for k in ['planned','recorded','valid','invalid','successes'])and p['planned']==p['recorded']==BENCHMARKS[p['benchmark']]and p['valid']+p['invalid']==p['recorded']and p['valid']>0,'Own-valid complete disposition counts')
    n=p['valid'];own[key]=p
   else:
    need(set(p)=={'benchmark','method','common_valid','successes','success_rate','metrics'},'Only common-valid aggregate fields');c=d['common_valid_counts'][p['benchmark']];need(set(c)=={'planned','common_valid','excluded'}and all(integer(v)for v in c.values())and c['planned']==BENCHMARKS[p['benchmark']]and c['common_valid']>0 and c['common_valid']+c['excluded']==c['planned'],'Explicit common population counts');n=c['common_valid'];need(type(p['common_valid'])is int and p['common_valid']==n and n<=own[key]['valid'],'Same shared-valid denominator for every method')
   need(integer(p['successes'])and p['successes']<=n and number(p['success_rate'])and math.isclose(p['success_rate'],p['successes']/n,rel_tol=0,abs_tol=1e-12),'Exact success denominator')
   if population=='common_valid':need(p['successes']<=own[key]['successes']and n-p['successes']<=own[key]['valid']-own[key]['successes'],'Common-valid outcomes must fit own-valid cohort')
   need(set(p['metrics'])==set(METRICS),'Three complete actor metrics, no hosting estimate')
   for name,v in p['metrics'].items():
    need(set(v)=={'mean','observed_tasks','missing_valid_tasks'}and v['observed_tasks']==n and type(v['observed_tasks'])is int and type(v['missing_valid_tasks'])is int and v['missing_valid_tasks']==0 and number(v['mean'])and v['mean']>=0,'No missing or imputed metric coverage')
    if name!='browser_commands':need(v['mean']>0,'Positive actor cost/elapsed latency')
    else:need(v['mean']<=cap(p['method']),'Batch count cannot exceed native decision cap')
 return d

def average(d,population):
 validate(d);need(population in('own_valid','common_valid'),'Known population');points={(p['benchmark'],p['method']):p for p in d[population+'_points']}
 return [dict(method=m,success_rate=math.fsum(n*points[b,m]['success_rate']for b,n in BENCHMARKS.items())/1045,**{metric:math.fsum(n*points[b,m]['metrics'][metric]['mean']for b,n in BENCHMARKS.items())/1045 for metric in METRICS})for m in METHODS]
def frontier(points,metric):
 return sorted([p for p in points if not any(q[metric]<=p[metric]and q['success_rate']>=p['success_rate']and(q[metric]<p[metric]or q['success_rate']>p['success_rate'])for q in points)],key=lambda p:(p[metric],p['success_rate']))
def csv_write(path,rows,fields):
 with path.open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def render(d,out):
 validate(d)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 from matplotlib.legend_handler import HandlerTuple
 from matplotlib.ticker import MaxNLocator,NullFormatter
 out=Path(out);out.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'browsercode-fourteen-method-v1'})
 allpoints={p:average(d,p)for p in ['own_valid','common_valid']};ys=[100*r['success_rate']for rows in allpoints.values()for r in rows];limits=max(0,5*math.floor((min(ys)-1)/5)),min(100,5*math.ceil((max(ys)+1)/5));outputs=[]
 for population,rows in allpoints.items():
  stem='browsercode_scaling14_'+population+'_weighted';fig,axes=plt.subplots(1,3,figsize=(7.7,2.9),sharey=True);fig.subplots_adjust(left=.07,right=.745,top=.84,bottom=.26,wspace=.30)
  title='Own-valid weighted average'if population=='own_valid'else'Fourteen-method common-valid weighted average';fig.suptitle(title,x=.07,y=.97,ha='left',fontsize=9)
  for ax,metric,label in zip(axes,METRICS,['Actor cost ($, log)','Browser-code batches','Actor latency (s/task)']):
   ax.set_ylim(*limits);ax.yaxis.set_major_locator(MaxNLocator(nbins=4));ax.spines[['top','right']].set_visible(False);ax.tick_params(length=2.5,width=.6,pad=2);ax.set_xlabel(label,fontsize=7)
   if metric=='actor_usd':ax.set_xscale('log');ax.xaxis.set_minor_formatter(NullFormatter())
   else:ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
   edge=frontier(rows,metric)
   if len(edge)>1:ax.plot([p[metric]for p in edge],[100*p['success_rate']for p in edge],'--',lw=.65,color='#777777',zorder=1)
   for p in rows:
    name,marker,color,filled=STYLES[p['method']]
    if p['method']in CACHED:ax.scatter(p[metric],100*p['success_rate'],marker=marker,s=49,facecolors='none',edgecolors='#222222',linewidths=.8,zorder=3)
    ax.scatter(p[metric],100*p['success_rate'],marker=marker,s=28,facecolors=color if filled else'white',edgecolors=color,linewidths=.9,zorder=4)
  axes[0].set_ylabel('Valid success (%)'if population=='own_valid'else'Common-valid success (%)',fontsize=7)
  handles=[];labels=[]
  for m in METHODS:
   label,marker,color,filled=STYLES[m];inner=Line2D([],[],ls='none',marker=marker,markersize=4.7,markeredgecolor=color,markerfacecolor=color if filled else'white',markeredgewidth=.9)
   handles.append((Line2D([],[],ls='none',marker=marker,markersize=6.4,markeredgecolor='#222222',markerfacecolor='none',markeredgewidth=.8),inner)if m in CACHED else inner);labels.append(label)
  fig.legend(handles,labels,handler_map={tuple:HandlerTuple(ndivide=1,pad=0)},loc='center left',bbox_to_anchor=(.76,.54),frameon=False,fontsize=6.2,labelspacing=.45,handletextpad=.4)
  coverage='Own valid subsets differ'if population=='own_valid'else'Common valid: '+', '.join(str(d['common_valid_counts'][b]['common_valid'])for b in BENCHMARKS)
  fig.text(.07,.115,'Gap-repaired baseline + Sol/60 + cached; fixed planned weights300/595/150',fontsize=6.5,color='#555555');fig.text(.07,.065,coverage+'; windows differ; black outline: cached',fontsize=6.5,color='#555555');fig.text(.07,.015,'Dashed: observed nondominated points only; no inferred superiority.',fontsize=6.2,color='#555555')
  for ext in ['png','pdf','svg']:
   p=out/(stem+'.'+ext);need(not p.exists(),'Preserve prior public-stage figures');meta={'Date':None}if ext=='svg'else{'CreationDate':None,'ModDate':None}if ext=='pdf'else{};fig.savefig(p,dpi=300,metadata=meta);outputs.append(p)
  plt.close(fig);p=out/(stem+'_coordinates.csv');csv_write(p,rows,['method','success_rate',*METRICS]);outputs.append(p)
 for population in ['own_valid','common_valid']:
  rows=[]
  for p in d[population+'_points']:rows.append({k:v for k,v in p.items()if k!='metrics'}|{m:p['metrics'][m]['mean']for m in METRICS})
  path=out/(population+'_per_benchmark.csv');csv_write(path,rows,list(rows[0]));outputs.append(path)
 path=out/'common_valid_benchmark_counts.csv';csv_write(path,[dict(benchmark=b,**r)for b,r in d['common_valid_counts'].items()],['benchmark','planned','common_valid','excluded']);outputs.append(path)
 return outputs

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--aggregate',required=True,type=Path);p.add_argument('--output-directory',required=True,type=Path);a=p.parse_args();d=validate(json.loads(a.aggregate.read_text()));files=render(d,a.output_directory);provenance=dict(schema='browsercode-fourteen-method-render-v1',aggregate_sha256=sha(a.aggregate),renderer_sha256=sha(__file__),matplotlib_version=__import__('matplotlib').__version__,source_populations='Separate own-valid and newly recomputed fourteen-method common-valid cohorts; gap-repaired baseline explicitly labeled',output_sha256={f.name:sha(f)for f in files});path=a.output_directory/'render-provenance.json'
 with path.open('x')as f:json.dump(provenance,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(dict(figures=2,methods=14,common_valid_counts=d['common_valid_counts'])))
if __name__=='__main__':main()
