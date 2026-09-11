#!/usr/bin/env python3
"""Cheap durable metric monitoring; does not allocate, restart, or kill jobs."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import statistics
import time

RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction')


def inspect(root):
    result={'run_root':str(root),'checked_utc':datetime.now(timezone.utc).isoformat(),'alerts':[]}
    for filename in ['status.json','training-status.json','recovery-status.json','supervisor-status.json','wandb-identity.json']:
        p=root/filename
        if p.exists():result[filename]=json.loads(p.read_text())
    metrics=root/'student/metrics.jsonl';rows=[]
    if metrics.exists():
        lines=metrics.read_text().splitlines()
        for i,line in enumerate(lines):
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i+1!=len(lines):raise
    training=[row for row in rows if 'train/loss' in row or 'train/cross_entropy' in row]
    validation=[row for row in rows if 'val/C2/dpo_loss' in row]
    if training:
        last=training[-1];recent=training[-16:]
        result['updates']=last['updates'];result['last_training_metric']=last
        result['recent_mean_loss']=statistics.mean(r.get('train/loss',r.get('train/cross_entropy')) for r in recent)
        result['mean_update_seconds']=statistics.mean(r['train/update_seconds'] for r in recent)
        result['remaining_training_minutes']=(174-last['updates'])*result['mean_update_seconds']/60
        for row in recent:
            if any(isinstance(v,(int,float)) and not math.isfinite(v) for k,v in row.items() if k.startswith('train/')):
                result['alerts'].append('Nonfinite training metric');break
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(last['utc'])).total_seconds()
        phase=result.get('training-status.json',{}).get('phase')
        if phase=='training' and last['updates']<174 and age>900:
            result['alerts'].append('No training metric for over 15 minutes; inspect checkpoint/validation/worker state')
    if validation:
        first,last=validation[0],validation[-1];result['latest_validation']=last
        if last['updates']>0:
            for source in ['C2','Piotr']:
                ce=f'val/{source}/winner_ce';pref=f'val/{source}/dpo_loss'
                if last[ce]>1.05*first[ce] and last[pref]>first[pref]:
                    result['alerts'].append(f'{source}: held-out CE worsened >5% and DPO loss increased; review before scaling')
    for filename in ['training-status.json','recovery-status.json','supervisor-status.json']:
        if result.get(filename,{}).get('phase') in ['failed','awaiting_repair']:
            result['alerts'].append(f'{filename}: {result[filename]["phase"]}')
    result['complete']=(root/'complete.json').exists()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path)
    p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=int,default=600)
    a=p.parse_args()
    while True:
        result=inspect(a.run_root)
        output=a.run_root/'monitor-health.json';temporary=output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(output)
        brief={k:result[k] for k in ['checked_utc','updates','recent_mean_loss','mean_update_seconds','remaining_training_minutes','alerts','complete'] if k in result}
        print(json.dumps(brief),flush=True)
        if not a.watch or result['complete']:return
        time.sleep(a.interval)


if __name__=='__main__':main()
