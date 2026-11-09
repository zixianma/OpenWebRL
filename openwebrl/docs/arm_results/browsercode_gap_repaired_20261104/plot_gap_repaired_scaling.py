#!/usr/bin/env python3
"""Render public, aggregate-only, one-pass gap-repaired BrowserCode comparisons.

Requires numpy2.0.2 and matplotlib3.9.4. Task identities and raw trajectories are
not inputs. Historical744-task figures are preserved as separate snapshots.
"""
# Shared semantic palette; source measurements and geometry are unchanged.
from pathlib import Path as _PalettePath
import sys as _palette_sys
_palette_sys.path.insert(0, str(next(p for p in _PalettePath(__file__).resolve().parents
                                     if (p / "paper_plot_palette.py").is_file())))
from paper_plot_palette import COLORS as PALETTE, tint as palette_tint
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/openwebrl-gap-repaired-mpl')
HERE=Path(__file__).resolve().parent
HIST=HERE.parent/'browsercode_inference_scaling_20261031'
spec=importlib.util.spec_from_file_location('_historical_scaling_families',HIST/'plot_browsercode_scaling_families.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
S=F.C.SCALING
METRICS=('actor_usd','browser_commands','actor_seconds')
def need(v,m):
 if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(d):
 need(set(d)=={'schema','methods','judge_protocol','benchmark_counts','metric_population','benchmark_weights','historical_source_aggregate_sha256','gap_closed_certificate_sha256','private_reconstruction_sha256','gap_attempt_counts','gap_attempt_counts_by_benchmark','points','own_valid_points','paired_contingencies','actor_cost_definition','comparison_caveat','prompt_limit_caveat','browser_command_definition','all_original_valid_outcomes_preserved'},'Only documented sanitized aggregate fields')
 need(d['schema']=='browsercode-gap-repaired-scaling-v1','Known aggregate schema required')
 need(d['browser_command_definition']==S.COMMAND_DEFINITION,'Exact selected-batch definition')
 historical=json.loads((HIST/'aggregate_common_valid.json').read_text())
 expected_pairs={(p['benchmark'],p['left_method'],p['right_method'])for p in json.loads((HIST/'browsercode_scaling_family_pairs.json').read_text())['pairs']}
 need(len(d['paired_contingencies'])==42 and {(p['benchmark'],p['left_method'],p['right_method'])for p in d['paired_contingencies']}==expected_pairs,'All42 original paired contrast identities required')
 need(d['methods']==list(S.ALL_METHODS)and d['judge_protocol']==S.PROTOCOL,'All eleven historical methods and same Terra endpoint')
 need(d['historical_source_aggregate_sha256']==sha(HIST/'aggregate_common_valid.json'),'Exact preserved historical source')
 need(d['all_original_valid_outcomes_preserved']is True,'Historical successes and failures must be preserved')
 need(d['gap_attempt_counts']['valid']==386 and d['gap_attempt_counts']['invalid']==317,'Complete703 gap endpoint')
 need(d['metric_population']=='same_all_method_common_valid_tasks_within_each_benchmark','Matched metric population')
 for b,(_,n)in S.BENCHMARKS.items():
  c=d['benchmark_counts'][b];need(all(type(v)is int and v>=0 for v in c.values())and c['planned']==n and c['common_valid']+c['excluded']==n and c['original_common_valid']==historical['benchmark_counts'][b]['common_valid']<=c['common_valid'],'Common-cohort counts')
  need(d['benchmark_weights'][b]==f'{n}/1045','Fixed planned benchmark weights')
 need(set(d['benchmark_counts'])==set(S.BENCHMARKS),'Exact three benchmarks')
 lookup={};own={}
 for collection,target,common in[(d['points'],lookup,True),(d['own_valid_points'],own,False)]:
  need(len(collection)==33,'Complete33 points required')
  for p in collection:
   key=p['benchmark'],p['method'];need(key not in target and key[0]in S.BENCHMARKS and key[1]in S.ALL_METHODS,'Unique canonical point')
   n=p['valid'];need(type(n)is int and 0<n<=S.BENCHMARKS[key[0]][1]and type(p['successes'])is int and 0<=p['successes']<=n,'Exact valid/success counts')
   if common:need(n==d['benchmark_counts'][key[0]]['common_valid'],'Same task intersection for all methods')
   need(abs(p['success_rate']-p['successes']/n)<1e-12,'Success denominator')
   need(set(p['metrics'])==set(METRICS),'Three measured actor metrics')
   for k,m in p['metrics'].items():
    need(m['observed_tasks']==n and m['missing_valid_tasks']==0 and type(m['mean'])in(int,float)and math.isfinite(m['mean'])and m['mean']>=0,'No missing metric imputation')
    if k!='browser_commands':need(m['mean']>0,'Positive cost and latency')
    else:
     limit=120 if key[1]=='luna_single_high_120steps'else 60 if key[1].endswith('60steps')else 30
     need(m['mean']<=limit,'Selected batches within native decision limit')
   target[key]=p
 for k,p in lookup.items():
  q=own[k];need(p['valid']<=q['valid']and p['successes']<=q['successes']and p['valid']-p['successes']<=q['valid']-q['successes'],'Intersection fits native own-valid marginals')
 groups={}
 for p in d['paired_contingencies']:
  n=d['benchmark_counts'][p['benchmark']]['common_valid'];cs=[p[k]for k in('both_success','left_only_success','right_only_success','both_failure')]
  need(all(type(v)is int and v>=0 for v in cs)and sum(cs)==p['common_valid']==n,'Exact paired joint counts')
  need(cs[0]+cs[1]==lookup[p['benchmark'],p['left_method']]['successes']and cs[0]+cs[2]==lookup[p['benchmark'],p['right_method']]['successes'],'Paired marginals agree')
  g=groups.setdefault((p['left_method'],p['right_method']),{});need(p['benchmark']not in g,'No duplicate paired row');g[p['benchmark']]=p
 need(all(set(g)==set(S.BENCHMARKS)for g in groups.values()),'All benchmarks in every contrast')
 points={m:dict(method=m,success_rate=math.fsum(n*lookup[b,m]['success_rate']for b,(_,n)in S.BENCHMARKS.items())/1045,**{k:math.fsum(n*lookup[b,m]['metrics'][k]['mean']for b,(_,n)in S.BENCHMARKS.items())/1045 for k in METRICS})for m in S.ALL_METHODS}
 return points,groups

def render(d,points,out):
 from matplotlib.lines import Line2D
 from matplotlib.ticker import MaxNLocator,NullFormatter
 plt=F.theme();F.OUTPUT=out
 fig,axes=plt.subplots(2,2,figsize=(7.2,3.55),sharey=True)
 fig.subplots_adjust(left=.08,right=.97,bottom=.24,top=.78,wspace=.17,hspace=.42)
 fig.suptitle('Inference scaling after one-pass gap recovery',x=.08,y=.99,ha='left',fontsize=9)
 sol=points['serial_high'];fig.text(.08,.91,f"Sol high / 30 reference: {100*sol['success_rate']:.2f}% · ${sol['actor_usd']:.4f}/task · {sol['actor_seconds']:.1f} s/task",fontsize=7,color=PALETTE["SINGLE_CALL_MAIN"])
 axes[0,0].set_title('Single-run scaling',loc='left',fontsize=8,pad=17);axes[0,1].set_title('Action-selection scaling',loc='left',fontsize=8,pad=17)
 base=F.BASE
 singles=[base,'luna_single_high_6144','luna_single_high_24576','luna_single_xhigh_12288','luna_single_high_60steps','luna_single_high_120steps'];selection=F.SELECTION+['luna_single_high_60steps','parallel5_select_high_60steps']
 axes[0,0].legend(handles=[Line2D([],[],color=PALETTE["SINGLE_CALL_MAIN"],label='Steps'),Line2D([],[],color=PALETTE["SINGLE_CALL_MAIN"],linestyle=':',label='Output cap'),Line2D([],[],color=PALETTE["BASELINE_DARK"],linestyle='-.',label='Effort')],loc='lower left',bbox_to_anchor=(0,1),ncol=3,frameon=False,borderaxespad=0,columnspacing=.9,handlelength=1.5,fontsize=6.1)
 axes[0,1].legend(handles=[Line2D([],[],color=PALETTE["ADM_DEFAULT"],label='30 steps'),Line2D([],[],color=PALETTE["ADM_DEFAULT"],linestyle='--',label='60 steps')],loc='lower left',bbox_to_anchor=(0,1),ncol=2,frameon=False,borderaxespad=0,fontsize=6.1)
 labels={base:'high / 30','luna_single_high_6144':'0.5× cap','luna_single_high_24576':'2× cap','luna_single_xhigh_12288':'xhigh','luna_single_high_60steps':'60 steps','luna_single_high_120steps':'120 steps','parallel2_select_high':'N2','parallel3_select_high':'N3','parallel5_select_high':'N5 / 30','parallel5_select_high_60steps':'N5 / 60'}
 offsets={base:(4,-2),'luna_single_high_6144':(-23,3),'luna_single_high_24576':(4,6),'luna_single_xhigh_12288':(4,-3),'luna_single_high_60steps':(-29,-3),'luna_single_high_120steps':(4,3),'parallel2_select_high':(-18,3),'parallel3_select_high':(-8,-11),'parallel5_select_high':(5,-1),'parallel5_select_high_60steps':(5,0)}
 rates=[100*p['success_rate']for p in points.values()];lo=max(0,2*math.floor((min(rates)-1)/2));hi=min(100,2*math.ceil((max(rates)+1)/2))
 ranges=[]
 for row,key in enumerate(['actor_usd','actor_seconds']):
  for col,methods in enumerate([singles,selection]):
   ax=axes[row,col];curves=[([base,'luna_single_high_60steps','luna_single_high_120steps'],PALETTE["SINGLE_CALL_MAIN"],'-'),(['luna_single_high_6144',base,'luna_single_high_24576'],PALETTE["SINGLE_CALL_MAIN"],':'),([base,'luna_single_xhigh_12288'],PALETTE["BASELINE_DARK"],'-.')]if col==0 else[(F.SELECTION,PALETTE["ADM_DEFAULT"],'-'),(['luna_single_high_60steps','parallel5_select_high_60steps'],PALETTE["ADM_DEFAULT"],'--')]
   for members,color,line in curves:ax.plot([points[m][key]for m in members],[100*points[m]['success_rate']for m in members],color=color,linestyle=line,linewidth=.85,zorder=2)
   for m in methods:
    label='N1 / 30'if col==1 and m==base else'N1 / 60'if col==1 and m=='luna_single_high_60steps'else labels[m];off=(4,2)if col==1 and m=='luna_single_high_60steps'else offsets[m]
    if col==0 and row==1 and m=='luna_single_high_60steps':off=(-32,-9)
    if col==0 and row==1 and m=='luna_single_high_6144':off=(-26,3)
    F.dot(ax,points[m],key,m)
    ax.annotate(label,(points[m][key],100*points[m]['success_rate']),xytext=off,textcoords='offset points',fontsize=6.5,color=PALETTE['ADM_DEFAULT'] if m.startswith('parallel') else PALETTE['MUTED'],bbox=dict(facecolor='white',edgecolor='none',alpha=.88,pad=.1),zorder=5)
   F.style(ax);ax.set_ylim(lo,hi);ax.yaxis.set_major_locator(MaxNLocator(5,integer=True))
   if col==0:ax.set_ylabel('Success (%)',labelpad=3)
   xs=[points[m][key]for m in methods];left=min(xs);right=max(xs)
   if row==0:
    ax.set_xscale('log');ax.xaxis.set_minor_formatter(NullFormatter());span=max(math.log(right/left),.15);ax.set_xlim(left*math.exp(-.25*span),right*math.exp(.34*span));ax.set_xlabel('Actor API cost ($/task, log)',fontsize=7,labelpad=2)
    candidates=[.008,.009,.010,.011,.012]if col==0 else[.01,.02,.04,.06,.08]
    ticks=[t for t in candidates if ax.get_xlim()[0]<=t<=ax.get_xlim()[1]]
    ax.set_xticks(ticks,[f'{t:.3f}'.lstrip('0')if col==0 else f'{t:.2f}'.lstrip('0')for t in ticks])
   else:
    span=max(right-left,10);ax.set_xlim(left-.13*span,right+.32*span);ax.xaxis.set_major_locator(MaxNLocator(4));ax.set_xlabel('Actor latency (s/task)',fontsize=7,labelpad=2)
   ranges.append(dict(row=row,column=col,metric=key,xlim=list(ax.get_xlim())))
 n=sum(c['common_valid']for c in d['benchmark_counts'].values());coverage=' · '.join(f'{S.BENCHMARKS[b][0]} {c["common_valid"]}'for b,c in d['benchmark_counts'].items())
 fig.text(.08,.13,f'Same {n} common-valid tasks: {coverage}',fontsize=6.4,color=PALETTE["MUTED"])
 fig.text(.08,.075,'Fixed planned weights 300/595/150 · Local x ranges · Actor metrics exclude Terra',fontsize=6.4,color=PALETTE["MUTED"])
 fig.text(.08,.02,'Original valid outcomes retained · Single/60 advertised 30 · Live windows differ · Cached/Sol60 excluded',fontsize=6.1,color=PALETTE["MUTED"])
 F.save(fig,'browsercode_scaling_gap_repaired_compact');plt.close(fig)
 return ranges

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--aggregate',type=Path,default=HERE/'aggregate.json');p.add_argument('--output-dir',type=Path,default=HERE);a=p.parse_args();d=json.loads(a.aggregate.read_text());points,groups=validate(d);paired=F.contrasts(d,groups)
 for c in paired:need(abs(c['difference_pp']-100*(points[c['left_method']]['success_rate']-points[c['right_method']]['success_rate']))<1e-10,'Weighted contrast exactly agrees')
 a.output_dir.mkdir(parents=True,exist_ok=True);ranges=render(d,points,a.output_dir)
 summary=dict(schema='browsercode-gap-repaired-weighted-summary-v1',benchmark_counts=d['benchmark_counts'],benchmark_weights=d['benchmark_weights'],points=list(points.values()),paired_contrasts=paired,uncertainty='50000 paired bootstrap draws within benchmark; fixed planned weights; pointwise percentile95 intervals, no multiplicity correction. Conditional on common validity; task clustering/live-window/protocol uncertainty excluded.',axis_ranges=ranges)
 (a.output_dir/'weighted_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 provenance=dict(schema='browsercode-gap-repaired-render-provenance-v1',renderer_sha256=sha(__file__),aggregate_sha256=sha(a.aggregate),historical_renderer_sha256=sha(HIST/'plot_browsercode_scaling_families.py'),historical_common_renderer_sha256=sha(HIST/'plot_browsercode_common_valid.py'),historical_style_renderer_sha256=sha(HIST/'plot_browsercode_scaling.py'),historical_pair_definition_sha256=sha(HIST/'browsercode_scaling_family_pairs.json'),weighted_summary_sha256=sha(a.output_dir/'weighted_summary.json'),matplotlib_version=__import__('matplotlib').__version__,numpy_version=__import__('numpy').__version__,artifacts={x.name:sha(x)for ext in['png','pdf','svg']for x in[a.output_dir/f'browsercode_scaling_gap_repaired_compact.{ext}']})
 (a.output_dir/'render_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps({'benchmark_counts':d['benchmark_counts'],'methods':len(points),'paired_contrasts':len(paired),'artifacts':'PNG/PDF/SVG and weighted_summary.json'},indent=2))
if __name__=='__main__':main()
