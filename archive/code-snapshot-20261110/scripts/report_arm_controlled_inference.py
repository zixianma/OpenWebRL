#!/usr/bin/env python3
"""Independently verify the paired cohort and report measured inference costs.

Read-only until a complete cohort and final Slurm accounting are available.
Public outputs contain aggregates only; private attempts remain in runtime.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from openwebrl.arm_controlled_inference import MODES, PROTOCOL, summarize, verify_record
from openwebrl.arm_rescue_yield import write_json

CONTROL=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/controlled-inference-20261004')
PUBLIC=REPO/'openwebrl/docs/arm_results/rl_integration'
METRICS=('actor_requests','actor_input_tokens','actor_output_tokens','selector_calls',
         'selector_input_tokens','selector_output_tokens','browser_steps','browser_step_seconds',
         'actor_client_request_seconds','selector_service_seconds','elapsed_seconds')


def read(path):return json.loads(Path(path).read_text())


def accounting(approval):
    jobs=[a['job_id'] for a in approval['attempts']]
    assert all(j.isdecimal() for j in jobs) and len(set(jobs))==len(jobs)
    raw=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(jobs),
        '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'],text=True)
    rows={v[0]:v for line in raw.splitlines() if len(v:=line.split('|'))>=4}
    result=[]
    for attempt in approval['attempts']:
        row=rows[attempt['job_id']]
        state=row[1].split()[0]
        if state not in ('COMPLETED','FAILED','TIMEOUT','CANCELLED'):raise ValueError('Need final allocation accounting')
        result.append(dict(job_id=row[0],shard=attempt['shard'],state=state,
            elapsed_seconds=int(row[2]),gpu_hours=int(row[2])*2/3600,allocated_tres=row[3]))
    for shard in (0,1):
        if sum(r['elapsed_seconds'] for r in result if r['shard']==shard)>8*3600:
            raise ValueError('Approved per-shard budget exceeded')
    return result


def load_records(verify=False):
    sched=read(CONTROL/'schedule.json');manifest=read(CONTROL/'experiment.json')
    assert manifest['protocol']==PROTOCOL
    assert hashlib.sha256((CONTROL/'schedule.json').read_bytes()).hexdigest()==manifest['schedule_sha256']
    assert hashlib.sha256((CONTROL/'tasks.jsonl').read_bytes()).hexdigest()==manifest['task_file_sha256']
    pin=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
    records=[];missing=0
    for item in sched:
        root=CONTROL/f'shard-{item["shard"]}'
        for attempt in item['attempts']:
            path=root/'records'/(attempt['key']+'.json')
            if not path.exists():missing+=1;continue
            record=read(path)
            if verify:
                verify_record(record,item,attempt,root,pin)
                # The collection guard checks traces of valid results. The
                # final audit also verifies traces retained by failed episodes.
                for entry in record['trace_files']:
                    path=Path(entry['path'])
                    if not path.resolve().is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                        raise ValueError('Failed-episode decision trace changed')
            records.append(record)
    return sched,records,missing


def failure_audit(records):
    server_errors={};context_only={}
    for shard in (0,1):
        errors=Counter()
        for path in (CONTROL/f'shard-{shard}/workers').glob('*/actor.log'):
            for line in path.open():
                if '[http_server]' in line and ' Error: ' in line:
                    message=line.split(' Error: ',1)[1]
                    kind='context_limit' if "Requested token count exceeds the model's maximum context length" in message else 'other'
                    errors[kind]+=1
        server_errors[str(shard)]=dict(errors)
        context_only[shard]=bool(errors['context_limit']) and not errors['other']
    categories=defaultdict(Counter);context_keys=set()
    for record in records:
        if record['valid']:continue
        artifact=Path(record['artifact']);result=read(artifact)
        shard=int(next(p.name for p in artifact.parents if p.name in ('shard-0','shard-1')).split('-')[1])
        reason=result.get('terminate_reason','')
        if "generation_error: Client error '400 Bad Request'" in reason and context_only[shard]:
            kind='context_limit_rejected_before_model_forward';context_keys.add(record['key'])
        elif record['actor_requests']==0 and record['browser_steps']==0:
            kind='startup_failure_before_first_actor_request'
        elif reason=='env_step_error':kind='browser_step_error'
        elif result.get('metadata',{}).get('judge_timeout'):kind='judge_timeout'
        else:kind='other_invalid'
        categories[record['mode']][kind]+=1
    return dict(by_mode={m:dict(v) for m,v in categories.items()},server_errors=server_errors,
        context_attribution='HTTP400 episode artifacts plus server-wide audit that every HTTP error was a context rejection. Tokenizer validation occurs before scheduler submission.',
        handling='Raw records and overall outcomes unchanged. Context exhaustion remains a policy failure; it is counted as an observed failure in the additional common-valid sensitivity.'),context_keys


def costs(records):
    arm=[r for r in records if r['mode']=='arm'];actor=[r for r in records if r['mode']!='arm']
    tasks=len(arm)
    arm_cost={k:sum(r[k] for r in arm)/tasks for k in METRICS}
    curve=[dict(method=f'actor-pass@{n}',**{k:sum(r[k] for r in actor)*n/5/tasks for k in METRICS}) for n in range(1,6)]
    def quantiles(values):
        values=sorted(values)
        return dict(mean=sum(values)/len(values),p50=values[round((len(values)-1)*.5)],p90=values[round((len(values)-1)*.9)],p95=values[round((len(values)-1)*.95)])
    latency={m:quantiles([r['elapsed_seconds'] for r in records if r['mode']==m]) for m in MODES}
    return dict(arm_per_task=arm_cost,actor_expected_random_subset_per_task=curve,observed_episode_seconds_by_mode=latency,
                arm_to_one_actor={k:arm_cost[k]/curve[0][k] if curve[0][k] else None for k in METRICS},
                interpretation='Input tokens include cached tokens. All k episodes are paid; no early stop or deployable episode selector is assumed. Seconds sum overlapping client work, not GPU kernel time. Episode latency includes shared queues, browser startup and judging; it is not a matched-resource deployment latency benchmark.')


def physical_costs(records):
    by_key={r['key']:r for r in records};totals=defaultdict(float);judge_totals=defaultdict(float)
    committed_backend=defaultdict(lambda:defaultdict(float))
    judge_by_mode=defaultdict(lambda:defaultdict(float));errors=Counter();responses=Counter()
    physical=Counter();committed_requests=all_requests=0
    for shard in (0,1):
        root=CONTROL/f'shard-{shard}'
        for directory in (root/'attempts').glob('*/*'):
            if not directory.is_dir():continue
            key=directory.parent.name;physical[key]+=1
            is_committed=(key in by_key and Path(by_key[key]['artifact']).parent==directory)
            for path in directory.glob('actor-request-*.json'):
                row=read(path);all_requests+=1;committed_requests+=is_committed
                totals['client_seconds']+=row['client_seconds']
                if row.get('status')=='received':
                    totals['received_requests']+=1
                    for metric in ('prompt_tokens','completion_tokens','cached_tokens'):
                        value=row['meta'].get(metric)
                        if isinstance(value,(int,float)):totals[metric]+=value
                    if is_committed:
                        dest=committed_backend[by_key[key]['mode']];meta=row['meta']
                        dest['received_requests']+=1
                        for metric in ('prompt_tokens','completion_tokens','cached_tokens','e2e_latency'):
                            value=meta.get(metric)
                            if isinstance(value,(int,float)):dest[metric]+=value
                            else:dest[metric+'_missing_requests']+=1
                        if isinstance(meta.get('cached_tokens'),(int,float)):
                            if not 0<=meta['cached_tokens']<=meta['prompt_tokens']:raise ValueError('Invalid cache telemetry')
                            dest['uncached_prompt_tokens']+=meta['prompt_tokens']-meta['cached_tokens']
                else:errors[row.get('error_type','unknown')]+=1
        ledger=read(root/'judge-budget/ledger.json') if (root/'judge-budget/ledger.json').exists() else dict(calls=0,charged_or_reserved_usd=0)
        if ledger['calls']>3600 or ledger['charged_or_reserved_usd']>25:raise ValueError('Judge cap exceeded')
        for metric in ('calls','charged_or_reserved_usd'):judge_totals[metric]+=ledger[metric]
        for path in (root/'judge-budget').glob('*-usage.json'):
            row=read(path);mode=by_key.get(row['attempt'],{}).get('mode','uncommitted')
            for metric,value in dict(calls=1,usd=row['usd'],input_tokens=row['usage']['prompt_tokens'],output_tokens=row['usage']['completion_tokens']).items():
                judge_by_mode[mode][metric]+=value
        for path in (root/'judge-budget').glob('*-response.json'):
            row=read(path);responses[row['response_model']]+=1
            if not row['response_model'].startswith('o4-mini'):raise ValueError('Unexpected judge model')
    return dict(actor=dict(totals),actor_request_errors=dict(errors),all_physical_requests=all_requests,
        committed_actor_backend_by_mode={k:dict(v) for k,v in committed_backend.items()},
        committed_physical_requests=committed_requests,physical_attempt_directories=sum(physical.values()),
        slots_with_multiple_physical_attempts=sum(v>1 for v in physical.values()),
        judge=dict(judge_totals),judge_by_mode={k:dict(v) for k,v in judge_by_mode.items()},judge_response_models=dict(responses))


def plot(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=report['costs']['actor_expected_random_subset_per_task'];arm=report['costs']['arm_per_task']
    perf=report['performance'];ys=[r['pass_at_k']*100 for r in perf['pass_at_k']]
    ay=perf['by_mode']['arm']['overall']*100
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.7))
    for ax,metric,label,scale in zip(axes,['actor_output_tokens','actor_input_tokens','browser_steps'],
        ['Actor output tokens / task (thousands)','Actor input tokens / task (thousands)','Executed browser steps / task'],[1000,1000,1]):
        xs=[r[metric]/scale for r in rows]
        ax.plot(xs,ys,'o-',color='#0f766e',label='Ordinary episode pass@k')
        for x,y,k in zip(xs,ys,range(1,6)):ax.annotate(str(k),(x,y),xytext=(3,5),textcoords='offset points',fontsize=9)
        ax.scatter(arm[metric]/scale,ay,marker='D',s=70,color='#2563eb',label='One ARM-guided episode',zorder=4)
        ax.set(xlabel=label,ylabel='Overall success (%)',ylim=(max(0,min(*ys,ay)-8),min(100,max(*ys,ay)+8)))
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8,loc='best')
    fig.suptitle('Controlled Online-Mind2Web: all 300 tasks, frozen SFT actor, paired fresh episodes')
    fig.text(.5,.04,'ARM selector work is additional; input tokens include cache hits. These axes measure workload, not matched GPU time.\n'
        'Pass@k pays for k full episodes and uses oracle success; invalid attempts stay in the denominator.',ha='center',fontsize=9,color='#475569')
    fig.subplots_adjust(bottom=.24,top=.86,wspace=.28)
    for extension in ('png','svg'):fig.savefig(PUBLIC/f'controlled-inference-20261004.{extension}',dpi=180)
    svg=PUBLIC/'controlled-inference-20261004.svg';svg.write_text('\n'.join(l.rstrip() for l in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def plot_flops(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    curve=report['model_flops']['curve'];arm=curve['arm'];rows=curve['actor'];scale=1e15
    scores=np.array([arm['success'],*[r['success'] for r in rows]])*100
    ci=np.array(curve['bootstrap']['success_95'])*100
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True)
    titles=['Current serving: measured KV hits','Uniform identical-state sharing','Uniform recomputation per request']
    metrics=['observed_kv_lower_flops','shared_state_flops','cold_request_flops']
    for panel,(ax,metric,title) in enumerate(zip(axes,metrics,titles)):
        x=np.array([arm[metric],*[r[metric] for r in rows]])/scale
        xerr=None
        if panel==0:
            upper=np.array([arm['observed_kv_upper_flops'],*[r['observed_kv_upper_flops'] for r in rows]])/scale
            midpoint=(x+upper)/2;xerr=np.vstack((midpoint-x,upper-midpoint));x=midpoint
        ax.errorbar(x[1:],scores[1:],yerr=np.maximum(0,np.vstack((scores-ci[0],ci[1]-scores)))[:,1:],
            xerr=xerr[:,1:] if xerr is not None else None,fmt='o-',capsize=3,color='#0f766e',label='Ordinary episode pass@k')
        ax.errorbar(x[0],scores[0],yerr=np.maximum(0,np.array([[scores[0]-ci[0,0]],[ci[1,0]-scores[0]]])),
            xerr=xerr[:,:1] if xerr is not None else None,fmt='D',markersize=7,capsize=3,color='#2563eb',label='One ARM-guided episode')
        for i in range(1,6):ax.annotate(str(i),(x[i],scores[i]),xytext=(4,7),textcoords='offset points',fontsize=9)
        ax.set(title=title,xlabel='Estimated model FLOPs / task (10¹⁵)',xlim=(0,None),ylim=(0,100))
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Overall success (%)');axes[0].legend(fontsize=8,loc='lower right')
    fig.suptitle('ARM versus episode resampling: paired 300-task performance–cost comparison')
    fig.text(.5,.035,'Includes actor vision, prefill, decode and ARM selector. Vertical bars: task-bootstrap 95% intervals.\n'
        'Left horizontal bars: unknown vision-cache bounds; middle/right: hypothetical uniform cache policies. Pass@k uses oracle success.',
        ha='center',fontsize=9,color='#475569')
    fig.subplots_adjust(bottom=.23,top=.83,wspace=.12)
    for extension in ('png','svg'):fig.savefig(PUBLIC/f'controlled-inference-flops-20261004.{extension}',dpi=180)
    svg=PUBLIC/'controlled-inference-flops-20261004.svg';svg.write_text('\n'.join(l.rstrip() for l in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def browser_budget_comparison(records, task_rows):
    from estimate_arm_inference_flops import bootstrap_cost_curve
    from openwebrl.arm_inference_flops import match_expected_cost
    metrics=['browser_steps','browser_step_seconds'];indexed={(r['task_id'],r['mode']):r for r in records}
    rows=[]
    for task in task_rows:
        ordinary=[indexed[task['task_id'],f'actor{i}'] for i in range(5)]
        rows.append(dict(task_id=task['task_id'],arm_success=task['arm_success'],
            arm={m:indexed[task['task_id'],'arm'][m] for m in metrics},
            actor=[dict(k=p['k'],success=p['success'],**{m:sum(r[m] for r in ordinary)*p['k']/5 for m in metrics}) for p in task['actor']]))
    boot=bootstrap_cost_curve(rows,metrics)
    arm_success=sum(t['arm_success'] for t in rows)/len(rows)
    out={}
    for metric in metrics:
        arm_cost=sum(t['arm'][metric] for t in rows)/len(rows)
        actor_costs=[sum(t['actor'][k][metric] for t in rows)/len(rows) for k in range(5)]
        actor_success=[sum(t['actor'][k]['success'] for t in rows)/len(rows) for k in range(5)]
        out[metric]=dict(match_expected_cost(arm_cost,actor_costs,actor_success,arm_success),bootstrap=boot['by_cost_model'][metric])
    return dict(comparisons=out,interpretation='Separate resource axis; equal expected browser-step calls or measured step-seconds, not equal model compute. Steps include terminal done calls. Browser initialization, queues and judging are outside step-seconds.')


def main(final=False):
    schedule,records,missing=load_records(verify=final)
    if not final:
        print(json.dumps(dict(committed=len(records),missing=missing,by_mode=dict(Counter(r['mode'] for r in records)),
            valid=sum(r['valid'] for r in records),complete=not missing),indent=2));return
    if missing:raise ValueError(f'{missing} uncommitted attempts; do not publish a final comparison')
    for shard in (0,1):
        assert read(CONTROL/f'shard-{shard}/status.json')['complete']
        assert read(CONTROL/f'shard-{shard}/verification.json')['complete']
    allocations=accounting(read(CONTROL/'approval.json'))
    perf=summarize(records,[t['task_id'] for t in schedule],True)
    pairs=perf['paired_arm_actor5'];n=pairs['arm_only']+pairs['actor5_only']
    pairs['exact_two_sided_discordance_p']=min(1,2*sum(math.comb(n,j) for j in range(min(pairs['arm_only'],pairs['actor5_only'])+1))/2**n) if n else 1.
    indexed={(r['task_id'],r['mode']):r for r in records}
    fixed=[float(indexed[t['task_id'],'arm']['valid'] and indexed[t['task_id'],'arm']['reward']==1)-
        float(indexed[t['task_id'],'actor0']['valid'] and indexed[t['task_id'],'actor0']['reward']==1) for t in schedule]
    perf['arm_vs_fixed_actor0']=dict(mean_difference=sum(fixed)/len(fixed),arm_only=fixed.count(1),actor0_only=fixed.count(-1))
    failures,context_keys=failure_audit(records)
    corrected=[dict(r,valid=True,reward=0) if r['key'] in context_keys else r for r in records]
    sensitivity=summarize(corrected,[t['task_id'] for t in schedule],True)
    perf['context_exhaustion_counted_valid_failure_sensitivity']=dict(
        by_mode={m:{k:v for k,v in row.items() if k in ('valid','valid_only','overall','successes')} for m,row in sensitivity['by_mode'].items()},
        paired_arm_actor5=sensitivity['paired_arm_actor5'])
    # Analytic counters run only after immutable cohort validation above.
    from estimate_arm_inference_flops import main as estimate_flops
    estimate_flops(final=True)
    model_flops=read(CONTROL/'flops-audit.json')
    report=dict(generated_utc=datetime.now(timezone.utc).isoformat(),protocol=PROTOCOL,
        effective_settings_audit=read(CONTROL/'effective-settings-audit.json'),performance=perf,costs=costs(records),
        physical_costs=physical_costs(records),failure_audit=failures,model_flops=model_flops,
        matched_browser_budgets=browser_budget_comparison(records,read(CONTROL/'flops-task-comparison.json')),
        allocations=allocations,total_gpu_hours=sum(r['gpu_hours'] for r in allocations),
        artifact_audit=dict(committed_records=len(records),result_hashes_verified=len(records),trace_hashes_verified=sum(len(r['trace_files']) for r in records),
            selector_fallbacks=sum(r['selector_fallbacks'] for r in records),
            analysis_code_sha256={f:hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in ('scripts/report_arm_controlled_inference.py','scripts/estimate_arm_inference_flops.py','openwebrl/arm_inference_flops.py')}),
        limits=['One guided episode per task; paired seeds/tasks do not remove live website drift.',
            'Shared actor batches and a dedicated selector GPU prevent exact per-method GPU-hour attribution.',
            'Backend end-to-end latency includes queuing; selector service time excludes its queue. Neither is GPU kernel time.',
            'Token totals cover received responses; interrupted/failed requests may have unreported work, included only in whole-allocation GPU-hours.',
            'Benchmark trajectories are evaluation-only and excluded from critic training.',
            'AgentTrek judge and local browser sampling are the frozen historical protocol, not the separate stealth protocol.'])
    write_json(CONTROL/'final-report.json',report);write_json(PUBLIC/'controlled-inference-20261004.json',report)
    with (PUBLIC/'controlled-inference-20261004.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=['method','success_rate',*METRICS],lineterminator='\n');w.writeheader()
        w.writerow(dict(method='ARM',success_rate=perf['by_mode']['arm']['overall'],**report['costs']['arm_per_task']))
        for row,result in zip(report['costs']['actor_expected_random_subset_per_task'],perf['pass_at_k']):w.writerow(dict(row,success_rate=result['pass_at_k']))
    with (PUBLIC/'controlled-inference-flops-20261004.csv').open('w',newline='') as out:
        metrics=['cold_request_flops','observed_kv_lower_flops','observed_kv_upper_flops','shared_state_flops']
        w=csv.DictWriter(out,fieldnames=['method','success_rate',*metrics],lineterminator='\n');w.writeheader()
        curve=model_flops['curve'];arm=curve['arm']
        w.writerow(dict(method='ARM',success_rate=arm['success'],**{m:arm[m] for m in metrics}))
        for row in curve['actor']:w.writerow(dict(method=f'actor-pass@{row["k"]}',success_rate=row['success'],**{m:row[m] for m in metrics}))
    plot(report)
    plot_flops(report)
    print(json.dumps(dict(complete=True,attempts=len(records),arm=perf['by_mode']['arm']['overall'],pass5=perf['pass_at_k'][-1],gpu_hours=report['total_gpu_hours']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--final',action='store_true');args=p.parse_args();main(args.final)
