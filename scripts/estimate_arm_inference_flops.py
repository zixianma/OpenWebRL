#!/usr/bin/env python3
"""Recover model-compute estimates from immutable inference request receipts.

No model calls or GPU allocation. Cache cost indices privately; only aggregate
cost/performance curves may be published after cohort verification.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from openwebrl.arm_inference_flops import decoder_flops,vision_flops,image_geometry,shared_state_cost,match_expected_cost
from openwebrl.arm_rescue_yield import write_json
from report_arm_controlled_inference import CONTROL,load_records,failure_audit

ACTOR=Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
CONFIG=json.loads((ACTOR/'config.json').read_text())
PREPROCESSOR=json.loads((ACTOR/'preprocessor_config.json').read_text())
SELECTOR_CONFIG=json.loads((CONTROL.parent/'arm-refresh-20260924/baseline40/model-compat/config.json').read_text())
if any(CONFIG[k]!=SELECTOR_CONFIG[k] for k in ('text_config','vision_config')):
    raise ValueError('Counter must use separate dimensions if actor/selector architectures differ')
VERSION=hashlib.sha256((REPO/'openwebrl/arm_inference_flops.py').read_bytes()+Path(__file__).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def bootstrap_cost_curve(task_rows, metrics, draws=5000):
    import numpy as np
    n=len(task_rows);rng=np.random.default_rng(20261004)
    weights=rng.multinomial(n,np.full(n,1/n),size=draws)/n
    successes=np.array([[t['arm_success'],*[p['success'] for p in t['actor']]] for t in task_rows])
    boot_success=weights@successes
    out={}
    for metric in metrics:
        costs=np.array([[t['arm'][metric],*[p[metric] for p in t['actor']]] for t in task_rows])
        boot_cost=weights@costs
        matched=[match_expected_cost(c[0],c[1:].tolist(),s[1:].tolist(),s[0]) for c,s in zip(boot_cost,boot_success)]
        differences=[v['arm_minus_actor'] for v in matched if v['covered']]
        out[metric]=dict(draws=draws,covered_draws=len(differences),
            matched_difference_95=np.quantile(differences,[.025,.975]).tolist() if differences else None,
            interval_interpretation='Paired task bootstrap, re-estimating costs and mixture weights; interval conditional on observed k<=5 covering the budget.',
            mean_cost_95=np.quantile(boot_cost,[.025,.975],axis=0).tolist())
    return dict(by_cost_model=out,success_95=np.quantile(boot_success,[.025,.975],axis=0).tolist(),
                column_order=['arm',*[f'actor-pass@{k}' for k in range(1,6)]])


def per_episode(record):
    root=Path(record['artifact']).parent;cache=CONTROL/'flops-indices'/(record['key']+'.json')
    if cache.exists():
        prior=read(cache)
        if prior['counter_version']==VERSION and prior['artifact_sha256']==record['artifact_sha256']:return prior
    receipts=[read(f) for f in sorted(root.glob('actor-request-*.json'))]
    k=5 if record['mode']=='arm' else 1
    traces={}
    for item in record['trace_files']:
        for line in Path(item['path']).read_text().splitlines():
            trace=json.loads(line)
            if trace['turn'] in traces:raise ValueError('Duplicate turn trace')
            traces[trace['turn']]=trace
    geometry={};samples=list(root.glob('samples/turn/*/*.json'))
    if len(samples)>1:raise ValueError('Ambiguous episode image archive')
    if samples:
        saved=read(samples[0])
        for images in saved['images']:
            for value in images:
                data=value.get('url') if isinstance(value,dict) else value
                actor=image_geometry(data,max_pixels=PREPROCESSOR['size']['longest_edge'],min_pixels=PREPROCESSOR['size']['shortest_edge'])
                selector=image_geometry(data,max_pixels=262144)
                geometry[actor['sha256']]=(actor,selector)
    out=dict(key=record['key'],task_id=record['task_id'],mode=record['mode'],counter_version=VERSION,
        artifact_sha256=record['artifact_sha256'],actor_decode_flops=0,selector_flops=0,actor_prefill_states={},actor_vision_images={},
        selector_decode_flops=0,selector_prefill_states={},selector_vision_images={},
        cold_request_flops=0,observed_kv_lower_flops=0,observed_kv_upper_flops=0,
        actor_lm_cold_flops=0,actor_lm_observed_kv_flops=0,actor_vision_no_cache_flops=0,
        received_actor_requests=0,unmetered_actor_requests=0,retractions=0,selector_calls=0,images_verified=0)
    known_tokens=defaultdict(int)
    for receipt in receipts:
        if receipt['status']!='received':out['unmetered_actor_requests']+=1;continue
        meta=receipt['meta'];turn=receipt['request']//k
        if turn not in traces:raise ValueError(f'Received request missing turn trace: {record["key"]}/{turn}')
        trace=traces[turn];image_sha=trace['screenshot_sha256']
        if image_sha not in geometry:raise ValueError('Saved screenshot does not match decision trace')
        actor_grid=geometry[image_sha][0]['grid'];vision=vision_flops(CONFIG['vision_config'],actor_grid)['total']
        p=meta['prompt_tokens'];g=meta['completion_tokens'];cached=meta['cached_tokens']
        cold=decoder_flops(CONFIG['text_config'],p,g)
        observed=decoder_flops(CONFIG['text_config'],p,g,cached)
        out['actor_decode_flops']+=cold['decode']
        state_key=trace['prompt_sha256']+':'+image_sha
        prior=out['actor_prefill_states'].setdefault(state_key,cold['prefill'])
        if prior!=cold['prefill']:raise ValueError('Identical state has inconsistent token count')
        out['actor_vision_images'][image_sha]=vision
        out['actor_lm_cold_flops']+=cold['total'];out['actor_lm_observed_kv_flops']+=observed['total']
        out['actor_vision_no_cache_flops']+=vision
        out['cold_request_flops']+=cold['total']+vision
        out['observed_kv_lower_flops']+=observed['total']
        # Actor image-embedding cache hits were not metered. The upper bound
        # conservatively allows an encoder recomputation in every prefill chunk.
        chunks=(p-cached+4095)//4096
        out['observed_kv_upper_flops']+=observed['total']+chunks*vision
        out['received_actor_requests']+=1;out['retractions']+=meta.get('total_retractions',0)
        known_tokens['input']+=p;known_tokens['output']+=g
    if known_tokens['input']!=record['actor_input_tokens'] or known_tokens['output']!=record['actor_output_tokens']:
        raise ValueError('Record token counters disagree with received request receipts')
    for trace in traces.values():
        if record['mode']!='arm':continue
        telemetry=trace['selector_telemetry'];image_sha=trace['screenshot_sha256']
        if image_sha not in geometry:raise ValueError('Missing selector screenshot')
        grid=geometry[image_sha][1]['grid']
        lm=decoder_flops(CONFIG['text_config'],telemetry['input_tokens'],telemetry['output_tokens'])
        vision=vision_flops(CONFIG['vision_config'],grid)['total'];value=lm['total']+vision
        out['selector_flops']+=value;out['selector_calls']+=1
        out['selector_decode_flops']+=lm['decode']
        state_key=hashlib.sha256(json.dumps([trace['prompt_sha256'],image_sha,trace['url'],trace['candidates']],sort_keys=True).encode()).hexdigest()
        prior=out['selector_prefill_states'].setdefault(state_key,lm['prefill'])
        if prior!=lm['prefill']:raise ValueError('Identical selector state has inconsistent token count')
        out['selector_vision_images'][image_sha]=vision
    for field in ('cold_request_flops','observed_kv_lower_flops','observed_kv_upper_flops'):out[field]+=out['selector_flops']
    out['images_verified']=len({t['screenshot_sha256'] for t in traces.values()})
    # Cross-check reconstructed actor resize against the actually saved processor grid.
    if record['valid'] and traces:
        result=read(record['artifact']);grid=result.get('metadata',{}).get('image_grid_thw')
        latest=traces[max(traces)]['screenshot_sha256']
        if grid and grid!=[geometry[latest][0]['grid']]:raise ValueError('Reconstructed image grid differs from actual processor')
    out['shared_state_flops']=shared_state_cost([out])
    write_json(cache,out);return out


def cost_curve(costs,records,task_ids):
    idx={(r['task_id'],r['mode']):r for r in costs};outcomes={(r['task_id'],r['mode']):r for r in records}
    metrics=['cold_request_flops','observed_kv_lower_flops','observed_kv_upper_flops','shared_state_flops']
    rows=[];task_rows=[]
    for task in task_ids:
        arm=idx[task,'arm'];ordinary=[idx[task,f'actor{i}'] for i in range(5)]
        yes=lambda m:bool(outcomes[task,m]['valid'] and outcomes[task,m]['reward']==1)
        points=[]
        for k in range(1,6):
            subsets=list(itertools.combinations(range(5),k));cost_values=defaultdict(list);wins=[]
            for chosen in subsets:
                episodes=[ordinary[i] for i in chosen]
                for metric in metrics:
                    value=shared_state_cost(episodes) if metric=='shared_state_flops' else sum(e[metric] for e in episodes)
                    cost_values[metric].append(value)
                wins.append(any(yes(f'actor{i}') for i in chosen))
            points.append(dict(k=k,success=sum(wins)/len(wins),**{m:sum(v)/len(v) for m,v in cost_values.items()}))
        task_rows.append(dict(task_id=task,arm_success=int(yes('arm')),arm={m:arm[m] for m in metrics},actor=points))
    for k in range(5):
        rows.append(dict(k=k+1,success=sum(t['actor'][k]['success'] for t in task_rows)/len(task_rows),
            **{m:sum(t['actor'][k][m] for t in task_rows)/len(task_rows) for m in metrics}))
    arm=dict(success=sum(t['arm_success'] for t in task_rows)/len(task_rows),
        **{m:sum(t['arm'][m] for t in task_rows)/len(task_rows) for m in metrics})
    matched={}
    for metric in metrics:
        matched[metric]=match_expected_cost(arm[metric],[r[metric] for r in rows],[r['success'] for r in rows],arm['success'])
    lo='observed_kv_lower_flops';hi='observed_kv_upper_flops'
    cache_bounds=dict(arm_to_actor5_cost_ratio=[arm[lo]/rows[-1][hi],arm[hi]/rows[-1][lo]],
        less_budget_more_expensive_baseline=match_expected_cost(arm[lo],[r[hi] for r in rows],[r['success'] for r in rows],arm['success']),
        more_budget_cheaper_baseline=match_expected_cost(arm[hi],[r[lo] for r in rows],[r['success'] for r in rows],arm['success']),
        interpretation='Cross-method extreme vision-cache assumptions; not a statistical confidence interval. No extrapolation beyond k=5.')
    return dict(arm=arm,actor=rows,matched_expected_budget=matched,vision_cache_sensitivity=cache_bounds,
        bootstrap=bootstrap_cost_curve(task_rows,metrics),
        interpretation='Equal expected model work using an outcome-independent randomized mixture of adjacent k; no per-task hard-cap claim. Pass@k is an oracle success bound.'),task_rows


def main(final=False):
    schedule,records,missing=load_records(verify=False)
    if final and missing:raise ValueError('Need complete cohort before final cost/performance reporting')
    costs=[]
    for i,record in enumerate(records):
        costs.append(per_episode(record))
        if (i+1)%100==0:print('FLOPS_INDEXED',i+1,flush=True)
    totals={k:sum(c[k] for c in costs) for k in ['received_actor_requests','unmetered_actor_requests','retractions','selector_calls','images_verified']}
    failures,context_keys=failure_audit(records)
    rejected=sum(1 for record in records if record['key'] in context_keys
        for path in Path(record['artifact']).parent.glob('actor-request-*.json')
        if (row:=read(path)).get('error_type')=='HTTPStatusError' and row['status']!='received')
    totals['context_rejected_requests_no_model_forward']=rejected
    totals['unclassified_failed_actor_requests']=totals['unmetered_actor_requests']-rejected
    audit=dict(counter_version=VERSION,records=len(costs),missing_records=missing,counts=totals,
        failure_audit=failures,
        convention='Multiply-add=2 FLOPs; dominant matrix/conv and QK/AV work; causal attention counts only allowed query/key pairs.',
        uniform_cache_model='shared_state_flops: identical full-state prefill and image encodings shared within a task/policy, including ordinary episode subsets; no partial-prefix reuse. Idealized uniform cache policy, not measured hardware FLOPs.',
        reporting_policy='Present implementation-specific observed-KV vision bounds and uniform cache models together; do not select a favorable cache model after seeing outcomes.',
        sensitivity_models=dict(cold_request_flops='Recompute input/vision for every actor request; within-response autoregressive KV reuse retained.',
            observed_kv_lower_flops='Observed actor LM prefix hits; zero actor image-encoder misses; actual selector forward work.',
            observed_kv_upper_flops='Observed actor LM prefix hits; actor encoder recomputed in every uncached prefill chunk; actual selector forward work.'),
        limits=['Elementwise/norm/softmax operations, kernel padding and CPU work excluded.',
            'Image-embedding cache hit/miss events were not logged; observed-KV totals have explicit vision bounds.',
            'Unclassified failed requests and retractions can add unaccounted work; context rejections precede GPU scheduler submission.',
            'Per-policy curves cover committed episodes; interrupted physical attempts and server startup are charged in whole-allocation GPU-hours.',
            'Serving efficiency, memory traffic, browser time and API judge cost are separate metrics.'],
        architecture={k:CONFIG[k] for k in ('text_config','vision_config')},actor_selector_dimensions_identical=True,
        selector_image_max_pixels=262144,actor_image_max_pixels=PREPROCESSOR['size']['longest_edge'],
        sources=['https://arxiv.org/abs/2511.21631','https://docs.pytorch.org/docs/stable/profiler'])
    if final:
        curve,task_rows=cost_curve(costs,records,[t['task_id'] for t in schedule]);audit['curve']=curve
        valid_by_task=defaultdict(list)
        for record in records:valid_by_task[record['task_id']].append(record['valid'] or record['key'] in context_keys)
        common=[t['task_id'] for t in schedule if len(valid_by_task[t['task_id']])==6 and all(valid_by_task[t['task_id']])]
        audit['common_valid_sensitivity_tasks']=len(common)
        if common:
            audit['common_valid_sensitivity_curve']=cost_curve(costs,records,common)[0]
        write_json(CONTROL/'flops-task-comparison.json',task_rows)
    write_json(CONTROL/'flops-audit.json',audit)
    print(json.dumps(dict(records=len(costs),counts=totals,complete=not missing),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--final',action='store_true');a=p.parse_args();main(a.final)
