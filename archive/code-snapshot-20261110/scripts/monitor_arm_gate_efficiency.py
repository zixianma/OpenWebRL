#!/usr/bin/env python3
"""Read-only ARM reward/efficiency monitor, incremental logs every 15 min."""
import argparse
import ast
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

from resume_baseline import RUNTIME, write_json

CONTROL=RUNTIME/'arm-turn-bonus-preparation/bc-efficiency-20260922'
ROOTS={'B-history':'arm-failure-additive-313208','C-history':'arm-failure-additive-313210',
       'B':'arm-failure-additive-318934-iter40','C':'arm-failure-additive-318935-iter40',
       'B-next':'arm-failure-additive-318934-iter60','C-next':'arm-failure-additive-318935-iter60'}
ROOTS.update({f'{v}-TP2-{target}':f'arm-failure-additive-{job}-tp2-iter{target}'
              for v,job in [('B','318934'),('C','318935')] for target in (40,60)})
ROOTS.update({'beta-TP8':'arm-failure-additive-318949',
    'beta-TP2':'arm-failure-additive-318949-tp2-iter20',
    'q40-TP2':'arm-failure-additive-318950-tp2-iter20'})
KEYS=('rollout/raw_reward_mean','rollout/task/completed/invalid_rate',
      'rollout/task/completed/success_rate_valid','rollout/task/completed/success_rate_all_completed',
      'rollout/task/accepted/success_rate_valid','perf/rollout_time','perf/train_time',
      'perf/actor_train_time','perf/actor_train_tok_per_s','rollout/effective_batch_size')
PATTERN=re.compile(r' - (?:perf|rollout) (\d+): (\{.*\})')


def ingest(root,cache):
    log=root/'collection.log'
    if not log.is_file():return cache
    identity=[log.stat().st_dev,log.stat().st_ino]
    if cache.get('identity')!=identity or cache.get('offset',0)>log.stat().st_size:
        cache=dict(identity=identity,offset=0,rows={})
    with log.open('rb') as f:
        f.seek(cache['offset'])
        while True:
            position=f.tell();raw=f.readline()
            if not raw or not raw.endswith(b'\n'):
                cache['offset']=position;break
            match=PATTERN.search(raw.decode(errors='replace'))
            if not match:continue
            try:values=ast.literal_eval(match[2])
            except (ValueError,SyntaxError):continue
            row=cache['rows'].setdefault(str(int(match[1])+1),{})
            row.update({k:values[k] for k in KEYS if k in values})
    return cache


def summarize(rows,root):
    result=[]
    for iteration,values in sorted(rows.items(),key=lambda x:int(x[0])):
        r=dict(iteration=int(iteration),**values)
        gate=root/'iterations'/f'{int(iteration)-1:04d}'/'training_gate.json'
        if gate.is_file():
            r['optimizer_updates_planned']=json.loads(gate.read_text())['expected_optimizer_updates']
        duration=r.get('perf/actor_train_time');updates=r.get('optimizer_updates_planned')
        if duration and updates:r['optimizer_seconds_per_update']=duration/updates
        result.append(r)
    return result


def snapshot():
    CONTROL.mkdir(parents=True,exist_ok=True)
    cache_path=CONTROL/'log-offsets.json'
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),runs={},alerts=[])
    for label,name in ROOTS.items():
        root=RUNTIME/'evaluations'/name
        cache[label]=ingest(root,cache.get(label,{}))
        if not cache[label]:continue
        rows=summarize(cache[label]['rows'],root)
        status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
        metrics=('rollout/raw_reward_mean','rollout/task/completed/success_rate_valid',
                 'perf/rollout_time','optimizer_seconds_per_update','perf/actor_train_tok_per_s')
        recent={k:statistics.mean(r[k] for r in rows[-3:] if k in r)
                for k in metrics if any(k in r for r in rows[-3:])}
        result['runs'][label]=dict(root=str(root),rows=rows,last_three_mean=recent,
            stage=status.get('stage'),active_iteration=status.get('iteration'),
            completed_optimizer_updates=status.get('completed_optimizer_updates'),
            last_valid_checkpoint=status.get('last_valid_checkpoint'),updated=status.get('updated_at_epoch_seconds'))
        if label.endswith('history') or not rows:continue
        latest=rows[-1];iteration=latest['iteration']
        # B24 straddled the browser fix; its aggregate cannot measure recovery.
        clean_browser_window=not label.startswith('B') or iteration>=25
        if clean_browser_window and latest.get('rollout/task/completed/invalid_rate',0)>.2:
            result['alerts'].append(f'{label} iteration{iteration}: invalid trajectories exceed20%')
        reward_key='rollout/task/completed/success_rate_valid'
        previous=[r[reward_key] for r in rows[-6:-3] if reward_key in r]
        current=[r[reward_key] for r in rows[-3:] if reward_key in r]
        if len(previous)==len(current)==3 and statistics.mean(current)<statistics.mean(previous)-.1:
            result['alerts'].append(f'{label}: last-three valid task success down >10pp vs preceding three; inspect task mix')
        if status.get('failed'):result['alerts'].append(f'{label}: controller failed; inspect status/logs')
        if status.get('stage') not in (None,'complete') and time.time()-status.get('updated_at_epoch_seconds',0)>900:
            result['alerts'].append(f'{label}: controller heartbeat stale >15min')
        historical=result['runs'].get(label[0]+'-history',{}).get('rows',[])
        key='perf/actor_train_tok_per_s'
        reference=[r[key] for r in historical if key in r]
        current=[r[key] for r in rows if key in r][-2:]
        if reference and len(current)==2 and statistics.mean(current)<.75*statistics.median(reference):
            result['alerts'].append(f'{label}: last-two training token rates <75% of historical TP4 median; topology replay pending, batches differ')
    result['queue']=subprocess.check_output(['squeue','-j','318934,318935,318949,318950','--noheader',
        '--format=%i|%T|%M|%R'],text=True,timeout=20).strip().splitlines()
    diag=RUNTIME/'evaluations/arm-gate-c-to60-318935/tpdp-diagnostic.json'
    if diag.exists():result['topology_diagnostic']=json.loads(diag.read_text())
    result['interpretation']='Training task mix changes; reward drift is an alert, not an evaluation result. B24 includes pre-fix browser failures.'
    write_json(cache_path,cache);write_json(CONTROL/'latest.json',result)
    with (CONTROL/'snapshots.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    write_json(CONTROL/'alerts.json',dict(checked_utc=result['checked_utc'],alerts=result['alerts']))
    print(json.dumps(dict(checked=result['checked_utc'],queue=result['queue'],alerts=result['alerts'],
        latest={k:v['rows'][-1] if v['rows'] else {} for k,v in result['runs'].items()})),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true')
    p.add_argument('--hours',type=float,default=48);a=p.parse_args()
    CONTROL.mkdir(parents=True,exist_ok=True)
    with (CONTROL/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (CONTROL/'pid').write_text(str(os.getpid())+'\n')
        deadline=time.monotonic()+a.hours*3600
        while True:
            try:snapshot()
            except Exception as exc:print(json.dumps(dict(error=type(exc).__name__,detail=str(exc)[:300])),flush=True)
            if not a.watch or time.monotonic()+900>deadline:break
            time.sleep(900)


if __name__=='__main__':main()
