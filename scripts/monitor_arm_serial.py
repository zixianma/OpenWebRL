#!/usr/bin/env python3
"""Low-cadence local monitoring and durable dashboard updates for serial SFT."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import math
from pathlib import Path
import subprocess
import time

ROOT=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction')
REPO=Path(__file__).resolve().parents[1]
JOBS={'all5':'288893','diverse3':'288894'}


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return {}


def snapshot():
    now=datetime.now(timezone.utc).isoformat();runs={}
    raw=subprocess.check_output(['sacct','-j',','.join(JOBS.values()),'-n','-P','--format=JobIDRaw,State'],text=True,timeout=20)
    states={line.split('|')[0]:line.split('|')[1] for line in raw.splitlines() if '|' in line}
    for variant,job in JOBS.items():
        root=ROOT/f'runs/serial-{variant}-v1';status=read(root/'status.json');training=read(root/'training-status.json')
        path=root/'student/metrics.jsonl';metrics=[]
        if path.exists():
            for line in path.read_text().splitlines():
                try:metrics.append(json.loads(line))
                except json.JSONDecodeError:pass
        updates=[m for m in metrics if 'train/loss' in m];vals=[m for m in metrics if 'val/loss' in m]
        alerts=[]
        if status.get('phase') in ('failed','awaiting_repair') or training.get('phase')=='failed':alerts.append('worker_failure')
        if updates and any(isinstance(v,float) and not math.isfinite(v) for v in updates[-1].values()):alerts.append('nonfinite_metric')
        tp=root/'training-status.json'
        if status.get('phase')=='training' and tp.exists() and time.time()-tp.stat().st_mtime>900:alerts.append('training_no_update_15min')
        progress={}
        for rank in range(2):
            out=root/f'evaluation/shard-{rank}/online-mind2web'
            progress[str(rank)]=dict(saved_results=len(list((out/'results').glob('*.json'))),summary=read(out/'summary.json'))
        runs[variant]=dict(job=job,slurm=states.get(job,'UNKNOWN'),status=status,training=training,
            latest_train=updates[-1] if updates else {},latest_validation=vals[-1] if vals else {},
            mean_last8_update_seconds=sum(m['train/update_seconds'] for m in updates[-8:])/len(updates[-8:]) if updates else None,
            evaluation=progress,completion=read(root/'complete.json'),wandb=read(root/'wandb-identity.json'),alerts=alerts)
    return dict(checked_utc=now,runs=runs)


def record(report):
    path=ROOT/'serial-monitor.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(path)
    with (ROOT/'serial-monitor.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
    lines=['<!-- serial-sft-live:start -->',f"Monitor snapshot: {report['checked_utc']} (10-minute cadence).",'',
        '| Variant | Job / scheduler | Updates | Latest training loss | Evaluation outcomes |',
        '| --- | --- | ---: | ---: | --- |']
    for v,r in report['runs'].items():
        loss=r['latest_train'].get('train/loss');loss=f'{loss:.4f}' if loss is not None else '—'
        result=r['completion'].get('summary',{});saved=sum(s['saved_results'] for s in r['evaluation'].values())
        summary=(f"{result['successes']}/100 overall ({100*result['success_rate_all_scheduled']:.1f}%); "
            f"{result['successes']}/{result['valid']} valid-only ({100*result['success_rate_valid']:.1f}%)"
            if result and result.get('valid') else f'{saved}/100 saved; {r["status"].get("phase","startup")}')
        lines.append(f"| {v} | {r['job']} / {r['slurm']} | {r['training'].get('updates',0)}/170 | {loss} | {summary} |")
        if r['alerts']:lines.append('') ; lines.append('Alerts: '+', '.join(r['alerts'])) if r['alerts'] else None
    lines+=['','Local status/alerts: `/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/serial-monitor.json`.','<!-- serial-sft-live:end -->']
    block='\n'.join(lines)
    lock=(ROOT/'serial-docs.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    doc=REPO/'openwebrl/docs/ARM_RESULTS.md';text=doc.read_text();start='<!-- serial-sft-live:start -->';end='<!-- serial-sft-live:end -->'
    if start in text:text=text[:text.index(start)]+block+text[text.index(end)+len(end):]
    else:
        anchor='## Serial alternatives SFT: all five versus diverse up to three\n'
        text=text.replace(anchor,anchor+'\n'+block+'\n',1)
    tmp=doc.with_suffix('.serial-tmp');tmp.write_text(text);tmp.replace(doc);lock.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--interval',type=int,default=600);p.add_argument('--once',action='store_true');args=p.parse_args()
    lock=(ROOT/'serial-monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        report=snapshot();record(report)
        print(json.dumps(dict(checked=report['checked_utc'],runs={v:dict(job=r['job'],slurm=r['slurm'],phase=r['status'].get('phase'),updates=r['training'].get('updates'),alerts=r['alerts']) for v,r in report['runs'].items()})),flush=True)
        active={'RUNNING','PENDING','COMPLETING','CONFIGURING','UNKNOWN'}
        if args.once or all(r['slurm'] not in active for r in report['runs'].values()):break
        time.sleep(args.interval)


if __name__=='__main__':main()
