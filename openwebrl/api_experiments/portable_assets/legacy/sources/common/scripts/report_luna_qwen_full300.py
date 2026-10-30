#!/usr/bin/env python3
"""Audit fresh full300 arms; keep the reused SFT reference explicitly unmatched."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwebrl.luna_qwen_metrics import digest, episode_usage, quantile, summarize, write_json
from openwebrl.luna_qwen_eval import verified_record
from report_luna_qwen_inference import plot


def read(path):
    return json.loads(Path(path).read_text())


def paired(records, tasks, left, right):
    index = {(r['task_id'], r['mode']):r for r in records}
    delta, common, common_delta = [], [], []
    for task in tasks:
        a, b = index[task,left], index[task,right]
        d = int(a['valid'] and a['reward']==1)-int(b['valid'] and b['reward']==1)
        delta.append(d)
        if a['valid'] and b['valid']:
            common.append(task); common_delta.append(d)
    rng = random.Random(20261004)
    boot = [sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(5000)]
    return dict(mean=sum(delta)/len(delta), task_bootstrap_95=[quantile(boot,.025),quantile(boot,.975)],
        common_valid_tasks=len(common), common_valid_difference=sum(common_delta)/len(common_delta) if common_delta else None)


def report(root, destination):
    root, destination = Path(root), Path(destination)
    plan = read(root/'plan.json')
    records, physical, protocols = [], [], {}
    missing_timing = 0
    for family, ref in plan['families'].items():
        p = read(ref['plan'])
        if digest(p) != ref['plan_sha256']:
            raise ValueError('Family plan changed')
        protocols[family] = p['protocol']
        output = Path(p['output'])
        for file in (output/'records').glob('*.json'):
            row = verified_record(file, digest(p['protocol']))
            usage = episode_usage(Path(row['artifact']).parent,row['mode'],row['episode_seconds'],plan['pricing'])
            for name in ('model_input_tokens','model_output_tokens','model_usage_missing_requests'):
                if row[name] != usage[name]:
                    raise ValueError('Receipts and committed metrics disagree')
            records.append(dict(row,**usage))
        for identity in (output/'attempts').glob('*/*/identity.json'):
            measured = identity.parent/'physical-usage.json'
            if measured.exists():
                physical.append(read(measured))
            else:
                info=read(identity)
                physical.append(dict(episode_usage(identity.parent,info['mode'],0.,plan['pricing']),interrupted=True))
                missing_timing += 1
    modes = [mode for family in plan['modes'].values() for mode in family]
    result = summarize(records,plan['task_ids'],plan['pricing'],modes=modes)
    result.update(protocols=protocols,historical_sft_reference=plan['baseline_reuse'],
        pilot_not_full300=False,final_scheduler_audit_required=True)
    for left,right in [('qwen_luna5','qwen'),('qwen_luna10','qwen'),
                       ('qwen_luna10','qwen_luna5'),('sft_luna10','sft_luna5')]:
        if left in modes and right in modes:
            result['paired_differences'][left+'_minus_'+right]=paired(records,plan['task_ids'],left,right)
    for row in result['by_mode'].values():
        row['valid_only'] = row['successes']/row['valid'] if row['valid'] else None
        row['invalid'] = row['episodes']-row['valid']
        row['serving_usd_per_success'] = row['cost_upper_mean_usd']*row['episodes']/row['successes'] if row['successes'] and row['cost_upper_mean_usd'] is not None else None
    result['physical_collection'] = dict(attempts=len(physical),committed=len(records),
        interrupted=sum(r['interrupted'] for r in physical),attempts_missing_final_timing=missing_timing,
        model_input_tokens=sum(r['model_input_tokens'] for r in physical),
        model_output_tokens=sum(r['model_output_tokens'] for r in physical),
        unknown_usage_requests=sum(r['model_usage_missing_requests'] for r in physical))
    result['api_ledgers']={role:read(root/f'budget/{role}-budget/ledger.json') for role in ('luna','judge')}
    destination.mkdir(parents=True,exist_ok=True)
    write_json(destination/'summary.json',result)
    with (destination/'metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['mode','tasks','successes','valid','overall','valid_only','mean_cost_lower_usd',
            'mean_cost_upper_usd','p50_episode_seconds','p95_episode_seconds','mean_input_tokens',
            'mean_output_tokens','mean_browser_steps','mean_local_gpu_seconds','mean_flops_lower','mean_flops_upper'])
        for mode,row in result['by_mode'].items():
            m=row['metrics']
            writer.writerow([mode,row['episodes'],row['successes'],row['valid'],row['overall'],row['valid_only'],
                row['cost_lower_mean_usd'],row['cost_upper_mean_usd'],m['episode_seconds']['p50'],m['episode_seconds']['p95'],
                m['model_input_tokens']['mean'],m['model_output_tokens']['mean'],m['browser_steps']['mean'],
                m['local_gpu_reserved_seconds']['mean'],m['qwen_flops_lower']['mean'],m['qwen_flops_upper']['mean']])
    plot(result,destination)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    r=report(args.root,args.output)
    print(json.dumps(dict(complete=r['complete'],episodes=r['recorded_episodes'],final_scheduler_audit_required=True)))
