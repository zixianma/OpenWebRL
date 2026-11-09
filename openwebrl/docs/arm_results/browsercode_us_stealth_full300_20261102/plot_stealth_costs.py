#!/usr/bin/env python3
"""Render qualified aggregate-only OM2W US-stealth performance and observed costs.

No input: describe the consumer contract without creating a result artifact.
Billing covers ALL owned sessions, including probes/recoveries/invalid attempts.
Performance covers own-valid or explicitly common-valid task cohorts. These are
separate panels and denominators. Unknown fees remain unknown, and historical
conservative maxima/reserves never become observed charges or an invoice.
No runtime imports, task IDs, credentials, API, browser or scheduler operations.
"""
from __future__ import annotations
# Shared semantic palette; source measurements and geometry are unchanged.
from pathlib import Path as _PalettePath
import sys as _palette_sys
_palette_sys.path.insert(0, str(next(p for p in _PalettePath(__file__).resolve().parents
                                     if (p / "paper_plot_palette.py").is_file())))
from paper_plot_palette import COLORS as PALETTE, tint as palette_tint
import argparse
import csv
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/openwebrl-stealth-cost-matplotlib')
SCHEMA='browsercode-us-stealth-observed-costs-v1'
ARMS=('luna_single_high','parallel5_select_high')
STYLE={'luna_single_high':('Luna high single',PALETTE["SINGLE_CALL_MAIN"]),'parallel5_select_high':('Luna high N5',PALETTE["ADM_DEFAULT"])}
PROTOCOL={'actor_model':'gpt-6-luna','reasoning_effort':'high','max_output_tokens_per_generation':12288,'max_model_decisions':30,'actor_loop_deadline_seconds':900,'harness':'BrowserCode','candidate_sampling_parallel':True,'qualified_runs_provider_allow_resizing':True,'viewport_png_pixels':[1280,1800],'browser_provider':'Browser Use V4','routing':'US residential','judge_model':'gpt-5.6-terra','judge_reasoning_effort':'medium','judge_max_output_tokens':4096,'judge_seed':42}
METRICS=('actor_cost_usd','task_latency_seconds','selected_code_batches')
BILLING_POPULATION='all_owned_sessions_including_probes_recoveries_and_invalid_attempts'

def need(value,message):
    if not value:raise ValueError(message)

def count(value):return type(value) is int and value>=0

def number(value):return type(value) in (int,float) and math.isfinite(value)

def digest(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def usd(value,nullable=False):
    if value is None and nullable:return None
    need(type(value) is str,'USD quantities require decimal text; unknown is null')
    try:x=Decimal(value)
    except InvalidOperation:raise ValueError('Malformed USD decimal') from None
    need(x.is_finite() and x>=0,'USD must be finite and nonnegative')
    return x

def exact(value,keys,message):need(type(value) is dict and set(value)==set(keys),message)

def means(value,tasks):
    exact(value,METRICS,'Exact performance metric set required')
    for name,row in value.items():
        exact(row,('mean','observed_tasks','missing_tasks'),'Metric mean and coverage required')
        n,missing=row['observed_tasks'],row['missing_tasks']
        need(count(n) and count(missing) and n+missing==tasks,'Performance metric coverage differs from cohort')
        need((n==0 and row['mean'] is None) or (n>0 and number(row['mean']) and row['mean']>=0),'Unavailable means must remain null')
        if name in ('task_latency_seconds','selected_code_batches'):
            need(n==tasks,'Performance figure requires complete valid-task latency/batches')
        if name=='selected_code_batches' and n:need(row['mean']<=30,'Selected batches exceed decision limit')

def validate(data):
    exact(data,('schema','benchmark','planned_tasks_per_method','protocol','endpoint','arms','common_valid'),'Only documented sanitized aggregate fields allowed')
    need(data['schema']==SCHEMA and data['benchmark']=='Online-Mind2Web','Unknown aggregate schema/benchmark')
    planned=data['planned_tasks_per_method'];need(count(planned) and 0<planned<=300,'Explicit original planned task count required')
    need(data['protocol']==PROTOCOL and all(type(data['protocol'][k]) is type(v) for k,v in PROTOCOL.items()),'Shared high/30/12288/900/Terra US-stealth protocol required')
    endpoint=data['endpoint']
    exact(endpoint,('kind','independently_qualified','qualified_endpoint_sha256','provider_reconciliation_sha256','all_attempts_terminal','final_invoice_asserted'),'Qualified endpoint metadata required')
    need(endpoint['kind'] in ('complete','budget_stop','calendar_stop','operational_stop'),'Endpoint kind must remain explicit')
    need(endpoint['independently_qualified'] is True and endpoint['all_attempts_terminal'] is True and endpoint['final_invoice_asserted'] is False,'Only qualified closed endpoints; latest fees are not invoices')
    need(digest(endpoint['qualified_endpoint_sha256']) and digest(endpoint['provider_reconciliation_sha256']),'Exact qualified endpoint/provider evidence hashes required')
    exact(data['arms'],ARMS,'Both methods required')
    for arm,row in data['arms'].items():
        exact(row,('counts','valid_task_means','provider','all_attempt_api'),'Exact method aggregate required')
        c=row['counts'];exact(c,('native_saved','valid','invalid','claimed_missing_native','unclaimed','successes'),'Native/missing/unclaimed dispositions required')
        need(all(count(v) for v in c.values()),'Nonnegative exact disposition counts required')
        need(c['valid']+c['invalid']==c['native_saved'] and c['native_saved']+c['claimed_missing_native']+c['unclaimed']==planned and c['successes']<=c['valid'],'Disposition identities/denominators disagree')
        if endpoint['kind']=='complete':need(c['unclaimed']==0,'Complete endpoint cannot contain unclaimed tasks')
        means(row['valid_task_means'],c['valid'])
        b=row['provider'];exact(b,('population','owned_sessions','actor_sessions','probe_sessions','all_owned_sessions_verified_stopped','minimum_known_snapshot_seconds_after_stop','hosting','proxy','conservative_max_observed_usd','retained_unknown_fee_reserve_usd'),'Exact all-owned provider fee aggregate required')
        need(b['population']==BILLING_POPULATION and b['all_owned_sessions_verified_stopped'] is True,'All owned sessions across every lineage must be stopped and included')
        need(all(count(b[k]) for k in ('owned_sessions','actor_sessions','probe_sessions')) and b['actor_sessions']+b['probe_sessions']==b['owned_sessions'],'Owned actor/probe session partition required')
        need(b['actor_sessions']<=c['native_saved']+c['claimed_missing_native'],'Owned actor sessions cannot belong to unclaimed tasks')
        subtotal=Decimal(0);observed=0;unknown=0
        for name in ('hosting','proxy'):
            fee=b[name];exact(fee,('known_usd','known_sessions','unknown_sessions'),'Separate observed fee and coverage required')
            n,missing=fee['known_sessions'],fee['unknown_sessions']
            need(count(n) and count(missing) and n+missing==b['owned_sessions'],'Provider fee coverage differs from owned union')
            amount=usd(fee['known_usd'],nullable=True)
            need((n==0 and amount is None) or (n>0 and amount is not None),'No fee observations must remain null, not an invented zero')
            if amount is not None:subtotal+=amount
            observed+=n;unknown+=missing
        age=b['minimum_known_snapshot_seconds_after_stop']
        need((observed==0 and age is None) or (observed>0 and number(age) and age>=90),'Known endpoint snapshots require the existing post-stop observation window')
        maximum=usd(b['conservative_max_observed_usd'],nullable=True)
        reserve=usd(b['retained_unknown_fee_reserve_usd'],nullable=True)
        # Historical Decimal aggregation order can differ below an attodollar;
        # amounts stay exact and no tolerance is added to any reported fee.
        need(maximum is None or maximum+Decimal('1e-18')>=subtotal,'Conservative maximum cannot fall below current known fees')
        need(unknown>0 or reserve==0,'Unknown-fee reserve is separate and must be zero when all fees are observed')
        exact(row['all_attempt_api'],('actor','terra'),'Actor and external Terra accounting remain separate')
        for role,api in row['all_attempt_api'].items():
            exact(api,('known_native_usd','usage_known_requests','usage_unknown_requests','known_usage_unsettled_requests','retained_full_reserve_usd','active_requests'),'Exact known/unknown native API coverage required')
            need(all(count(api[k]) for k in ('usage_known_requests','usage_unknown_requests','known_usage_unsettled_requests','active_requests')),'Exact API request counts required')
            need(api['active_requests']==0 and api['known_usage_unsettled_requests']<=api['usage_known_requests'],'No live reservations; known-unsettled is a subset of usage-known calls')
            known=usd(api['known_native_usd'],nullable=True);held=usd(api['retained_full_reserve_usd'])
            need((api['usage_known_requests']==0 and known is None) or (api['usage_known_requests']>0 and known is not None),'API known-cost subtotal needs actual native observations')
            need(api['usage_unknown_requests']+api['known_usage_unsettled_requests']>0 or held==0,'Reserve without an unresolved request')
    common=data['common_valid']
    if common is not None:
        exact(common,('tasks','outcomes','task_means'),'Only aggregate common-valid evidence allowed')
        n=common['tasks'];need(count(n) and 0<n<=min(r['counts']['valid'] for r in data['arms'].values()),'Invalid common-valid denominator')
        outcomes=common['outcomes'];exact(outcomes,('both_success','single_only_success','n5_only_success','both_failure'),'Exact paired outcome cells required')
        need(all(count(v) for v in outcomes.values()) and sum(outcomes.values())==n,'Common-valid paired counts disagree')
        exact(common['task_means'],ARMS,'Both common-valid method means required')
        for arm in ARMS:
            successes=outcomes['both_success']+outcomes['single_only_success' if arm==ARMS[0] else 'n5_only_success']
            own=data['arms'][arm]['counts']
            need(successes<=own['successes'] and n-successes<=own['valid']-own['successes'],'Common-valid outcomes must be a subset of each own-valid cohort')
            means(common['task_means'][arm],n)
    return data

def known_sum(*values):
    observed=[usd(v,nullable=True) for v in values if v is not None]
    with localcontext() as context:
        context.prec=80
        return None if not observed else sum(observed,Decimal(0))

def coordinates(data,population='own-valid'):
    validate(data);need(population in ('own-valid','common-valid'),'Unknown performance population')
    common=data['common_valid'];need(population!='common-valid' or common is not None,'Common-valid means cannot be inferred from own-valid aggregates')
    result=[]
    for arm in ARMS:
        r=data['arms'][arm]
        c=r['counts'];b=r['provider'];api=r['all_attempt_api'];m=r['valid_task_means'];n=c['valid'];success=c['successes']
        if population=='common-valid':
            n=common['tasks'];m=common['task_means'][arm];o=common['outcomes'];success=o['both_success']+o['single_only_success' if arm==ARMS[0] else 'n5_only_success']
        result.append(dict(method=arm,planned=data['planned_tasks_per_method'],**c,performance_population=population,performance_tasks=n,performance_successes=success,success_percent=None if n==0 else 100*success/n,
            actor_usd_per_task=m['actor_cost_usd']['mean'],actor_usd_observed_tasks=m['actor_cost_usd']['observed_tasks'],actor_seconds_per_task=m['task_latency_seconds']['mean'],selected_code_batches_per_task=m['selected_code_batches']['mean'],
            owned_sessions=b['owned_sessions'],actor_sessions=b['actor_sessions'],probe_sessions=b['probe_sessions'],hosting_known_usd=b['hosting']['known_usd'],hosting_known_sessions=b['hosting']['known_sessions'],hosting_unknown_sessions=b['hosting']['unknown_sessions'],proxy_known_usd=b['proxy']['known_usd'],proxy_known_sessions=b['proxy']['known_sessions'],proxy_unknown_sessions=b['proxy']['unknown_sessions'],
            provider_known_subtotal_usd=known_sum(b['hosting']['known_usd'],b['proxy']['known_usd']),provider_fees_complete=b['hosting']['unknown_sessions']+b['proxy']['unknown_sessions']==0,
            conservative_browser_maximum_usd=b['conservative_max_observed_usd'],unknown_browser_fee_reserve_usd=b['retained_unknown_fee_reserve_usd'],
            actor_api_known_usd=api['actor']['known_native_usd'],terra_api_known_usd=api['terra']['known_native_usd'],actor_usage_known_requests=api['actor']['usage_known_requests'],terra_usage_known_requests=api['terra']['usage_known_requests'],actor_usage_unknown_requests=api['actor']['usage_unknown_requests'],terra_usage_unknown_requests=api['terra']['usage_unknown_requests'],actor_known_usage_unsettled_requests=api['actor']['known_usage_unsettled_requests'],terra_known_usage_unsettled_requests=api['terra']['known_usage_unsettled_requests'],
            retained_api_full_reserve_usd=known_sum(api['actor']['retained_full_reserve_usd'],api['terra']['retained_full_reserve_usd'])))
    return result

def display(value,digits=3):return '—' if value is None else f'{float(value):.{digits}f}'

def table(data,population='own-valid'):
    rows=coordinates(data,population);out=['Billing includes all owned actor sessions and mechanical probes across every attempt/recovery. Values are latest known provider observations, not invoices. Unknown costs are not imputed.','', '| Method | Planned | Native | Valid | Invalid | Claimed missing | Unclaimed | Successes | Own-valid success (%) |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        out.append('| '+ ' | '.join([STYLE[r['method']][0]]+[str(r[k]) for k in ('planned','native_saved','valid','invalid','claimed_missing_native','unclaimed','successes')]+[display(None if r['valid']==0 else 100*r['successes']/r['valid'],1)])+' |')
    out+=['',f'Performance population: {population}; actor-loop time and native actor pricing upper bounds exclude Terra. Cost means use the stated priced-task coverage. Different own-valid cohorts are not paired effects.','', '| Method | Performance tasks | Priced tasks | Actor $ / priced task | Actor latency (s/task) | Selected batches / task |','|---|---:|---:|---:|---:|---:|']
    for r in rows:out.append('| '+' | '.join([STYLE[r['method']][0],str(r['performance_tasks']),str(r['actor_usd_observed_tasks']),display(r['actor_usd_per_task'],5),display(r['actor_seconds_per_task'],1),display(r['selected_code_batches_per_task'],2)])+' |')
    out+=['','| Method | Owned sessions | Hosting known sessions | Proxy known sessions | Hosting observed $ | Proxy observed $ | Conservative observed maximum $ | Unknown-fee reserve $ |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:out.append('| '+' | '.join([STYLE[r['method']][0]]+[str(r[k]) for k in ('owned_sessions','hosting_known_sessions','proxy_known_sessions')]+[display(r[k]) for k in ('hosting_known_usd','proxy_known_usd','conservative_browser_maximum_usd','unknown_browser_fee_reserve_usd')])+' |')
    out+=['','Unknown fee coverage = owned sessions minus known sessions, separately for hosting/proxy. A missing reserve is unquantified, not zero. Historical conservative maxima and full unresolved API reserves are accounting safeguards, not observed fees.','', '| Method | Priced native actor upper $ | Priced native Terra upper $ | Actor usage unknown | Terra usage unknown | Actor known/unsettled | Terra known/unsettled | Full API reserve $ |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:out.append('| '+' | '.join([STYLE[r['method']][0],display(r['actor_api_known_usd']),display(r['terra_api_known_usd'])]+[str(r[k]) for k in ('actor_usage_unknown_requests','terra_usage_unknown_requests','actor_known_usage_unsettled_requests','terra_known_usage_unsettled_requests')]+[display(r['retained_api_full_reserve_usd'])])+' |')
    return '\n'.join(out)+'\n'

def render(data,output_dir,stem='us_stealth_observed_costs',population='own-valid'):
    rows=coordinates(data,population)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Shared paper typography, including bundled Albert Sans fonts.
    import sys as _paper_sys
    from pathlib import Path as _PaperPath
    _paper_sys.path.insert(0, str(next(p for p in _PaperPath(__file__).resolve().parents
                                    if (p / "paper_plot_style.py").is_file())))
    import paper_plot_style as _paper_style

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':"Albert Sans",'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'openwebrl-us-stealth-observed-costs-v1'})
    fig,(performance,billing)=plt.subplots(1,2,figsize=(6.8,2.55),gridspec_kw={'width_ratios':[1,1.2]})
    fig.subplots_adjust(left=.075,right=.985,top=.74,bottom=.29,wspace=.48)
    fig.suptitle(f"OM2W · US residential · {data['planned_tasks_per_method']} planned tasks per method"+(' · partial endpoint' if data['endpoint']['kind']!='complete' else ''),y=.98,fontsize=9)
    performance.set_title('(a) '+('Common-valid pairs' if population=='common-valid' else 'Own-valid tasks'),loc='left',pad=7)
    performance.set_xlabel('Actor latency (s/task; excludes Terra)');performance.set_ylabel('Success (%)');performance.set_ylim(-3,103);performance.set_yticks([0,25,50,75,100])
    xmax=1
    for i,r in enumerate(rows):
        label,color=STYLE[r['method']]
        if r['performance_tasks']:
            x=r['actor_seconds_per_task'];xmax=max(xmax,x)
            performance.scatter(x,r['success_percent'],s=31,marker='o',color=color,zorder=3)
            performance.annotate(f"{'Single' if r['method']==ARMS[0] else 'N5'}: {r['performance_successes']}/{r['performance_tasks']}",(x,r['success_percent']),xytext=(-5,7) if r['method']==ARMS[0] else (5,-8),textcoords='offset points',ha='right' if r['method']==ARMS[0] else 'left',va='bottom' if r['method']==ARMS[0] else 'top',fontsize=6.5,color=color)
        else:performance.text(.04,.15+i*.15,label+': no valid tasks',transform=performance.transAxes,color=color,fontsize=6.5)
    performance.set_xlim(0,xmax*1.55)
    billing.set_title('(b) All-owned browser fees',loc='left',pad=7);billing.set_xlabel('Latest known hosting + proxy ($)')
    maxima=[];seen_components=set()
    for i,r in enumerate(rows):
        left=0
        for name,label,color in [('hosting','Hosting',PALETTE["BASELINE_LIGHT"]),('proxy','US proxy',PALETTE["BASELINE_DARK"])]:
            value=r[name+'_known_usd']
            if value is not None:
                amount=float(usd(value));billing.barh(i,amount,left=left,height=.4,color=color,label=label if name not in seen_components else None);seen_components.add(name);left+=amount
        maxima.append(left)
        missing=r['hosting_unknown_sessions']+r['proxy_unknown_sessions']
        text=('No observed fees' if r['provider_known_subtotal_usd'] is None else '$'+display(r['provider_known_subtotal_usd']))+(' (known subtotal)' if missing else '')
        billing.text(left+max(1,left)*.025,i,text,ha='left',va='center',fontsize=6.4)
    billing.set_yticks([0,1],[STYLE[r['method']][0] for r in rows]);billing.set_ylim(-.55,1.65);billing.invert_yaxis();billing.set_xlim(0,max([1]+maxima)*1.35)
    handles=[Line2D([],[],marker='o',linestyle='none',markersize=4,color=STYLE[a][1],label=STYLE[a][0])for a in ARMS]+[Patch(facecolor=PALETTE["BASELINE_LIGHT"],label='Hosting'),Patch(facecolor=PALETTE["BASELINE_DARK"],label='US proxy')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.91),ncol=4,frameon=False,fontsize=6.5,handlelength=1,handletextpad=.4,columnspacing=1.3)
    for ax in (performance,billing):ax.spines[['top','right']].set_visible(False);ax.tick_params(width=.6,length=2.5,pad=2)
    note='Performance and billing use different populations. Fees include probes, invalids and recovery attempts.\nBars show latest native fees; missing fees/reserves are separate. Latest snapshots are not final invoices.'
    if data['endpoint']['kind']!='complete':note+='  Endpoint: '+data['endpoint']['kind'].replace('_',' ')+'.'
    fig.text(.075,.10,note,fontsize=6.3,color=PALETTE["MUTED"],va='top')
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg','pdf'):
        metadata={'Date':None} if suffix=='svg' else {'CreationDate':None,'ModDate':None} if suffix=='pdf' else {}
        fig.savefig(output_dir/f'{stem}.{suffix}',dpi=300,metadata=metadata)
        if suffix=='svg':
            svg=output_dir/f'{stem}.svg'
            svg.write_text('\n'.join(line.rstrip()for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    with (output_dir/f'{stem}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (output_dir/f'{stem}_tables.md').write_text(table(data,population))
    return {'schema':'us-stealth-observed-cost-render-v1','performance_population':population,'provider_population':BILLING_POPULATION,'maxima_or_reserves_plotted_as_actual':False,'final_invoice_asserted':False,'dimensions_inches':[6.8,2.55],'matplotlib_version':matplotlib.__version__}

def contract():
    return {'schema':SCHEMA,'no_endpoint_input':'No result artifact or placeholder points are created. Supply an independently qualified sanitized endpoint aggregate.',
      'top_fields':['schema','benchmark','planned_tasks_per_method','protocol','endpoint','arms','common_valid'],
      'protocol':PROTOCOL,'endpoint_fields':['kind','independently_qualified','qualified_endpoint_sha256','provider_reconciliation_sha256','all_attempts_terminal','final_invoice_asserted'],
      'method_fields':{'counts':['native_saved','valid','invalid','claimed_missing_native','unclaimed','successes'],
       'valid_task_means':{m:['mean','observed_tasks','missing_tasks'] for m in METRICS},
       'provider':['population','owned_sessions','actor_sessions','probe_sessions','all_owned_sessions_verified_stopped','minimum_known_snapshot_seconds_after_stop','hosting','proxy','conservative_max_observed_usd','retained_unknown_fee_reserve_usd'],
       'hosting_and_proxy_fields':['known_usd','known_sessions','unknown_sessions'],
       'all_attempt_api_roles':['actor','terra'],'api_role_fields':['known_native_usd','usage_known_requests','usage_unknown_requests','known_usage_unsettled_requests','retained_full_reserve_usd','active_requests']},
      'provider_population':BILLING_POPULATION,'common_valid':'null unless available; otherwise tasks, four aggregate paired outcome cells, and both task_means using the same metric coverage shape.',
      'performance_definition':'actor_cost_usd is the qualified native usage pricing or upper bound per task, excluding Terra; it is not a provider invoice. Latency is per actor-loop task, not a sum of parallel HTTP durations. The provider snapshot-age value may be a certified lower bound.',
      'quantities':'USD = finite nonnegative decimal text; unknown = null. Counts are nonnegative exact integers. Known_usage_unsettled is a subset of usage_known requests, not another physical-call population. Retained reserves are never added to known observed costs. Minimum known provider snapshot age after first verified stop is90s. Administrative missing, if any, remains a subset of claimed_missing_native and is not recoded invalid.',
      'publication':'Preparation only until endpoint qualification and final root review. No raw IDs, task payloads or private paths are accepted.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--aggregate',type=Path);parser.add_argument('--output-dir',type=Path);parser.add_argument('--stem',default='us_stealth_observed_costs');parser.add_argument('--population',choices=('own-valid','common-valid'),default='own-valid');parser.add_argument('--validate-only',action='store_true')
    a=parser.parse_args()
    if a.aggregate is None:print(json.dumps(contract(),indent=2));return
    need(a.stem and Path(a.stem).name==a.stem and a.stem not in ('.','..'),'Output stem must be a filename')
    raw=a.aggregate.read_bytes();data=validate(json.loads(raw));coordinates(data,a.population)
    if a.validate_only:print(json.dumps({'status':'PASS','endpoint':data['endpoint']['kind'],'methods':2}));return
    need(a.output_dir is not None,'--output-dir required to render a qualified input')
    result=render(data,a.output_dir,a.stem,a.population);result.update(input_sha256=hashlib.sha256(raw).hexdigest(),renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    result['output_sha256']={ext:hashlib.sha256((a.output_dir/(a.stem+ext)).read_bytes()).hexdigest() for ext in ('.png','.svg','.pdf','.csv','_tables.md')}
    (a.output_dir/(a.stem+'_provenance.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
