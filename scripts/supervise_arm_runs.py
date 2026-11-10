#!/usr/bin/env python3
"""Persistent bounded CPU supervision for the approved September 20 ARM jobs.

No new allocation, model update, source change, or blind retry is performed.
This process records health and can finalize already-completed evaluations from
their existing logs/archives. Active-agent intervention handles code failures.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

import monitor_arm_turn_bonus as monitor
from resume_baseline import write_json

RUNTIME = monitor.RUNTIME
CONTROL = RUNTIME/'arm-turn-bonus-preparation/overnight-20260920'
JOBS = {
    '303573': ('original training', 'arm-turn-bonus-fresh-303573', 'train'),
    '303574': ('additive training', 'arm-failure-additive-303574', 'train'),
    '309053': ('B relaxed gate', 'arm-failure-additive-309053', 'train'),
    '309054': ('C gate + action credit', 'arm-failure-additive-309054', 'train'),
    '310981': ('B saved-batch recovery', 'arm-failure-additive-310981', 'train'),
    '311202': ('additive to 100', 'arm-failure-additive-311202', 'train'),
    '311203': ('C first-batch restart', 'arm-failure-additive-311203', 'train'),
    '311962': ('additive to 90, shared storage', 'arm-failure-additive-311962', 'train'),
    '311964': ('B fixed saved-batch recovery', 'arm-failure-additive-311964', 'train'),
    '311965': ('C continuation from iteration 6', 'arm-failure-additive-311965', 'train'),
    '309490': ('all-failure to 100', 'arm-turn-bonus-fresh-allfailure-309490', 'train'),
    '309685': ('original iteration-80 eval', 'arm-original-iter80-309685', 'eval'),
    '309686': ('additive iteration-80 eval', 'arm-additive-iter80-309686', 'eval'),
    '309687': ('all-failure iteration-80 eval', 'arm-allfailure-iter80-309687', 'eval'),
    '313187': ('additive iteration-90 eval', 'arm-additive-iter90-313187', 'eval'),
    '313188': ('all-failure iteration-90 eval', 'arm-allfailure-iter90-313188', 'eval'),
    '313208': ('B continuation to iteration 20', 'arm-failure-additive-313208', 'train'),
    '313210': ('C continuation to iteration 20', 'arm-failure-additive-313210', 'train'),
    '313209': ('B iteration-20 eval', 'arm-gate-b-iter20-313209', 'eval'),
    '313211': ('C iteration-20 eval', 'arm-gate-c-iter20-313211', 'eval'),
    '313256': ('rescue-yield pilot', 'arm-rescue-yield-313256', 'rescue'),
    '313264': ('rescue-yield corrected pilot', 'arm-rescue-yield-313264', 'rescue'),
    '313408': ('additive iteration-90 eval retry', 'arm-additive-iter90-313408', 'eval'),
    '313668': ('outcome-only to 100 + full-300 eval', 'stage1-baseline-to100-313668', 'stage1'),
    '313669': ('additive to 100 + full-300 eval', 'stage1-additive-to100-313669', 'stage1'),
    '313774': ('fixed-state ARM selection-quality audit', 'arm-quality-audit-313774', 'quality'),
    '314664': ('ARM-only fixed100 task-success audit', 'arm-task-success-314664', 'task_success'),
    '315098': ('baseline to100, scheduler restore fixed', 'stage1-baseline-to100-315098', 'stage1'),
    '315099': ('actor + ARM separate MIG slices', 'arm-mig-probe-315099', 'mig'),
    '315204': ('failure coverage, no actor updates', 'arm-failure-coverage-pilot-315204', 'coverage'),
}


def load_registered_jobs():
    """Pick up approved submissions while the supervisor is already running."""
    for job, spec in monitor.read(CONTROL/'discovered-jobs.json').items():
        if (job.isdigit() and isinstance(spec, (list, tuple)) and len(spec)==3
                and isinstance(spec[1], str) and '/' not in spec[1]
                and spec[2] in {'train','eval','rescue','stage1','quality','task_success','mig','coverage','diagnostic'}):
            JOBS[job]=tuple(spec)


def quota_snapshot():
    """Observe personal quota; never delete data or alter allocations."""
    from check_storage_quota import check_storage_quota
    checked = datetime.now(timezone.utc).isoformat()
    try:
        report = check_storage_quota(minimum_free_bytes=0)
        alerts = []
        if report['safe_headroom_bytes'] < 4*1024**4:
            alerts.append('Personal scrubbed quota has less than4TiB below the soft limit; review storage before further collection')
        if report['hard_headroom_bytes'] < 2*1024**4:
            alerts.append('Personal scrubbed quota has less than2TiB below the hard limit; checkpoint/recovery writes are at risk')
        return dict(report, checked_utc=checked, alerts=alerts)
    except Exception as exc:
        return dict(checked_utc=checked, error_type=type(exc).__name__,
                    alerts=['Personal quota could not be read; filesystem free space does not establish capacity'])


def discover_live_jobs():
    """Inventory all user jobs; recognize new ARM output roots when available."""
    raw = subprocess.check_output(['squeue','--user',os.environ.get('USER','zixianma'),
        '--noheader','--format=%i|%j|%T|%M|%R'],text=True,timeout=20)
    live = []
    for line in raw.splitlines():
        fields = line.split('|',4)
        if len(fields) != 5:
            continue
        job = dict(zip(('job','name','state','elapsed','node'),fields))
        live.append(job)
        if job['job'] in JOBS or not job['job'].isdigit() or not job['name'].startswith('arm-'):
            continue
        roots = list((RUNTIME/'evaluations').glob('*-'+job['job']))
        controllers=[r for r in roots if (r/'controller-plan.json').is_file()]
        if len(controllers)==1:roots=controllers
        if len(roots) != 1:
            continue
        root = roots[0]
        if (root/'probe-config.json').is_file(): kind='mig'
        elif (root/'controller-plan.json').is_file(): kind='stage1'
        elif (root/'rescue-config.json').is_file(): kind='rescue'
        elif (root/'audit-plan.json').is_file():
            kind='task_success' if monitor.read(root/'audit-plan.json').get('audit_kind')=='arm_only_fixed100' else 'quality'
        elif (root/'evaluation_manifest.json').is_file(): kind='eval'
        elif (root/'launch_manifest.json').is_file():
            kind='coverage' if monitor.read(root/'launch_manifest.json').get('failure_coverage_pilot') else 'train'
        else: continue
        JOBS[job['job']] = (job['name'],root.name,kind)
        write_json(CONTROL/'discovered-jobs.json',JOBS)
    return live


def eval_tracking(root):
    try:
        os.environ.setdefault('WANDB_CACHE_DIR',str(RUNTIME/'wandb-cache'))
        from dotenv import dotenv_values
        import wandb
        manifest = monitor.read(root/'evaluation_manifest.json')
        api = wandb.Api(api_key=dotenv_values(monitor.REPO/'.env').get('WANDB_API_KEY'),timeout=20)
        run = api.run(f"zixianma/openwebrl-evals/{manifest['wandb_run_id']}")
        return dict(url=run.url,state=run.state,last_history_step=run.lastHistoryStep,
                    metrics={k:v for k,v in dict(run.summary).items() if '/task/' in k})
    except Exception as exc:
        return dict(error=type(exc).__name__)


def scheduler_states(live_jobs=()):
    raw = subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(JOBS),
        '--format=JobIDRaw,State,Elapsed,Timelimit,NodeList,ExitCode'],text=True,timeout=30)
    result = {}
    for line in raw.splitlines():
        fields = line.split('|')
        if fields[0] in JOBS:
            result[fields[0]] = dict(zip(('job','state','elapsed','limit','node','exit_code'),fields))
    # Accounting can retain the previous failed attempt after a requeue.
    # squeue describes the current allocation, so it wins for live jobs.
    for live in live_jobs:
        if live['job'] in JOBS:
            result.setdefault(live['job'], {}).update(live, exit_code=None)
    return result


def snapshot(job, slurm, full=False):
    label, name, kind = JOBS[job]
    root = RUNTIME/'evaluations'/name
    status = monitor.read(root/'status.json')
    result = dict(job=job,label=label,kind=kind,root=str(root),slurm=slurm,
                  stage=status.get('stage'),alerts=[])
    state = slurm.get('state','UNKNOWN').split()[0].rstrip('+')
    if state == 'PENDING':
        # Never interpret a previous attempt's status/logs as current failure.
        result.update(stage='queued', complete=False)
        return result
    if kind == 'diagnostic':
        summary = monitor.read(root/'diagnostic-result.json')
        diagnostic_status=monitor.read(root/'diagnostic-status.json')
        result.update(stage=diagnostic_status.get('stage',status.get('stage','queued' if state=='PENDING' else 'startup')),
            complete=bool(summary.get('passed')),completed_optimizer_updates=status.get('completed_optimizer_updates',0),
            diagnostic_result=summary,
            progress_marker=(status.get('stage'),status.get('completed_optimizer_updates',0),bool(summary)))
        if status.get('failed') or diagnostic_status.get('failed'):
            result['alerts'].append('GPU diagnostic failed: '+str(diagnostic_status.get('error_type',status.get('error_type'))))
        if state in monitor.TERMINAL and not summary.get('passed'):
            result['alerts'].append('GPU diagnostic ended before nonempty update and GPU restoration passed')
        if full and state=='RUNNING':result['resources']=monitor.resource_sample(job,gpus=2)
    elif kind == 'stage1':
        result.update(stage1_snapshot(job, root, slurm, full))
    elif kind == 'quality':
        result.update(quality_snapshot(root, slurm, full))
    elif kind == 'task_success':
        result.update(task_success_snapshot(root, slurm, full))
    elif kind == 'mig':
        result.update(mig_snapshot(root, slurm))
    elif kind == 'coverage':
        result.update(coverage_snapshot(root, slurm, full))
    elif kind == 'train':
        if root.exists():
            data = monitor.snapshot(job,slurm,remote=False)
            result.update({key:data[key] for key in ['collection_progress','latest_train','recent_train_medians'] if key in data})
            result['alerts'].extend(data['alerts'])
            result['stage'] = status.get('stage')
            result['collection'] = status.get('iteration')
            result['completed_optimizer_updates'] = status.get('completed_optimizer_updates')
            marker = root/'runtime/latest_checkpointed_iteration.txt'
            if marker.is_file():result['durable_iteration'] = int(marker.read_text())+1
            calibration = data['calibration']
            if calibration and not calibration.get('decision',{}).get('passed',True):
                result['alerts'].append('Calibration check failed; review the gate and worker error before recovery')
                result['calibration_checks'] = calibration.get('decision',{}).get('checks',{})
            if status.get('stage') == 'complete':
                result['stop_reason'] = status.get('stop_reason')
                result['training_complete'] = status.get('training_complete')
            if full:
                cached = monitor.read(RUNTIME/f'arm-turn-bonus-preparation/monitor-{job}/latest.json')
                for key in ['resources','wandb']:
                    if key in cached:
                        value = cached[key]
                        if key == 'wandb':value = {k:v for k,v in value.items() if k != 'metrics'}
                        result[key] = value
                result['embedded_monitor_checked_utc'] = cached.get('checked_utc')
    elif kind == 'rescue':
        summary = monitor.read(root/'yield-summary.json')
        result.update(complete=status.get('complete', False),
            smoke_attempts=len(list((root/'smoke_records').glob('*.json'))),
            screen_attempts=len(list((root/'screen_records').glob('*.json'))),
            retry_attempts=len(list((root/'retry_records').glob('*.json'))),
            smoke_gate=monitor.read(root/'smoke-gate.json'),
            eligible_tasks=summary.get('eligible_tasks'), selected_tasks=summary.get('selected_tasks'),
            yield_summary=summary if summary.get('complete') else {},
            gpu_checkpoint_restored=(root/'checkpoint_restore_evidence.json').is_file())
        result['completed_tasks']=result['smoke_attempts']+result['screen_attempts']+result['retry_attempts']
        if state in monitor.TERMINAL and not result['complete']:
            result['alerts'].append('Rescue pilot ended before verified completion; preserve partial attempts')
        if status.get('stage') == 'failed':
            result['alerts'].append('Rescue pilot failed: '+str(status.get('error_type')))
        if full and state == 'RUNNING' and result['gpu_checkpoint_restored']:
            result['resources'] = monitor.resource_sample(job,gpus=2)
        if full and (root/'evaluation_manifest.json').is_file():
            result['wandb'] = eval_tracking(root)
    else:
        records = [monitor.read(path) for path in (root/'rollouts').glob('*.json')]
        records = [record for record in records if record]
        result.update(completed_tasks=len(records),saved_rollouts=len(list((root/'rollouts').glob('*.pt'))),
            successes=sum(r.get('metrics',{}).get('successes',0) for r in records),
            valid=sum(r.get('metrics',{}).get('valid_trajectories',0) for r in records),
            artifact_errors=sum(r.get('error_type') is not None for r in records),
            gpu_checkpoint_restored=(root/'checkpoint_restore_evidence.json').is_file(),
            complete=status.get('complete',False))
        if result['artifact_errors']:
            result['alerts'].append('Saved evaluation attempt reports an exception; inspect per-task records')
        if state in monitor.TERMINAL and not result['complete']:
            # Repair bookkeeping only when the native worker exited zero and
            # all task archives/verdicts already exist. Never rerun browsers.
            exit_code = monitor.read(root/'launcher_exit.json').get('returncode')
            if exit_code == 0 and len(records) == 300 and result['saved_rollouts'] == 300:
                try:
                    from evaluate_baseline_checkpoint import finalize_evaluation
                    plan = monitor.read(root/'evaluation_manifest.json')
                    finalize_evaluation(plan,0)
                    result.update(complete=True,bookkeeping_recovered=True)
                except Exception as exc:
                    result['alerts'].append('Finalization still blocked: '+type(exc).__name__)
            if not result['complete']:result['alerts'].append('Evaluation allocation ended before verified completion')
        if full and state == 'RUNNING' and result['gpu_checkpoint_restored']:
            result['resources'] = monitor.resource_sample(job,gpus=2)
        if full and (root/'evaluation_manifest.json').is_file():
            result['wandb'] = eval_tracking(root)
    if state in monitor.TERMINAL-{'COMPLETED'}:
        result['alerts'].append('Slurm ended with '+state)
    if status.get('failed'):
        result['error_type'] = status.get('error_type')
        result['error'] = status.get('error')
    return result


def validation_snapshot(job, variant, controller, slurm, benchmark=None):
    """Observe the isolated TP2 check without consulting old training telemetry."""
    root=benchmark or RUNTIME/f'benchmarks/arm-gate-{variant.lower()}-tp2-full-{job}'
    status=monitor.read(controller/'status.json')
    result=monitor.read(root/'result.json')
    plan=monitor.read(root/'plan.json')
    updates=[]
    for line in monitor.tail(root/'updates-rank0.jsonl',65536).splitlines():
        try:updates.append(json.loads(line))
        except ValueError:pass  # Last line may still be being written.
    logs=[root/name for name in ('train.log','restore.log')]
    latest=max((p.stat().st_mtime for p in logs if p.exists()),default=None)
    row=dict(kind='diagnostic',stage=status.get('stage','restore/startup'),
        controller_root=str(controller),validation_root=str(root),complete=False,alerts=[],
        validation_passed=result.get('passed',False),validation_updates=len(updates),
        validation_expected_updates=plan.get('optimizer_updates'),
        validation_checkpoint=monitor.read(root/'checkpoint-validation.json'),
        progress_marker=(len(updates),latest,bool(result)))
    if status.get('error_type'):
        row.update(stage='failed')
        row['alerts'].append('TP2 controller failed: '+status['error_type'])
    if slurm.get('state') in monitor.TERMINAL:
        row['alerts'].append('Allocation ended before production training/evaluation completed')
    return row


def stage1_snapshot(job, controller, slurm, full=False):
    """Follow training and evaluation inside one allocation without hiding failures."""
    state=slurm.get('state','UNKNOWN').split()[0].rstrip('+')
    if state=='PENDING':
        return dict(kind='train',stage='queued',alerts=[],complete=False)
    migration=monitor.read(RUNTIME/f'arm-turn-bonus-preparation/bc-tp2-migration-20260922/{job}.json')
    if migration.get('stage')=='released' and migration.get('variant') in ('B','C'):
        variant=migration['variant']
        controller=RUNTIME/f'evaluations/arm-gate-{variant.lower()}-to60-{job}-tp2'
        if not (controller/'controller-plan.json').is_file():
            return validation_snapshot(job,variant,controller,slurm)
    ablation=monitor.read(RUNTIME/f'arm-turn-bonus-preparation/ablation-tp2-migration-20260922/{job}.json')
    if ablation.get('stage') in ('released','armed-fresh') and ablation.get('variant') in ('weight','coverage'):
        variant=ablation['variant']
        controller=RUNTIME/f'evaluations/arm-failure-ablation-{variant}-{job}-tp2'
        status=monitor.read(controller/'status.json')
        if not (controller/'controller-plan.json').is_file() or status.get('stage')=='full-batch-validation':
            return validation_snapshot(job,variant,controller,slurm,RUNTIME/f'benchmarks/arm-ablation-tp2-full-{job}')
    status=monitor.read(controller/'status.json')
    plan=monitor.read(controller/'controller-plan.json')
    target=int(plan.get('current_target',plan.get('target',100)))
    gate=plan.get('variant') if plan.get('variant') in ('B','C') else None
    baseline=controller.name.startswith('stage1-baseline-')
    candidates=list((RUNTIME/'runs').glob(f'openwebrl-4b-reference-{job}-*')) if baseline else []
    training=(candidates[0] if len(candidates)==1 else None) if baseline else RUNTIME/f'evaluations/arm-failure-additive-{job}'
    if gate and plan.get('training_root'):
        suffix='-tp2' if controller.name.endswith('-tp2') else ''
        expected=RUNTIME/f'evaluations/arm-failure-additive-{job}{suffix}-iter{target}'
        if Path(plan['training_root'])!=expected:raise ValueError('Unexpected staged training root')
        training=expected
    if plan.get('variant') in ('weight','coverage') and controller.name.endswith('-tp2'):
        expected=RUNTIME/f'evaluations/arm-failure-additive-{job}-tp2-iter20'
        if target!=20 or Path(plan.get('training_root',''))!=expected:raise ValueError('Unexpected ablation DP training root')
        training=expected
    state=slurm.get('state','UNKNOWN').split()[0].rstrip('+')
    row=dict(kind='train',stage='restore/startup',alerts=[],complete=False,
        controller_root=str(controller),training_root=str(training) if training else None)
    if training is not None and training.exists():
        native=training if baseline else training/'runtime'
        marker=native/'latest_checkpointed_iteration.txt'
        if marker.exists():row['durable_iteration']=int(marker.read_text())+1
        progress=native/'progress.log'
        row['progress_marker']=progress.stat().st_mtime_ns if progress.exists() else None
        if baseline:
            row['stage']='training' if progress.exists() else 'restore/startup'
            health=monitor.tail(training/'health.jsonl',32768).splitlines()
            if health:
                try:row['resources']=json.loads(health[-1])
                except ValueError:pass
        else:
            data=monitor.snapshot(job,slurm,remote=False)
            row.update({k:data[k] for k in ['collection_progress','latest_train','recent_train_medians'] if k in data})
            row['alerts'].extend(data['alerts'])
            current=data['status'];row['stage']=current.get('stage',row['stage'])
            row['collection']=current.get('iteration')
            row['completed_optimizer_updates']=current.get('completed_optimizer_updates')
            if full:
                cached=monitor.read(RUNTIME/f'arm-turn-bonus-preparation/monitor-{job}/latest.json')
                row.update({k:cached[k] for k in ['resources','wandb'] if k in cached})
        directory=(native/f'evaluation/after{target}/rollouts' if baseline else
                   RUNTIME/f'evaluations/arm-gate-{gate.lower()}-iter{target}-{job}/rollouts' if gate else
                   RUNTIME/f'evaluations/arm-additive-iter{target}-{job}/rollouts')
        records=[monitor.read(p) for p in directory.glob('*.json')]
        records=[r for r in records if r]
        if records or row.get('durable_iteration')==target or status.get('complete'):
            row.update(kind='eval',stage='evaluation',completed_tasks=len(records),
                saved_rollouts=len(list(directory.glob('*.pt'))),
                successes=sum(r.get('metrics',{}).get('successes',0) for r in records),
                valid=sum(r.get('metrics',{}).get('valid_trajectories',0) for r in records))
            if full and not baseline and (directory.parent/'evaluation_manifest.json').exists():
                row['wandb']=eval_tracking(directory.parent)
    if status.get('complete'):
        row.update(complete=True,stage='complete')
    milestone=monitor.read(RUNTIME/f'arm-turn-bonus-preparation/milestone-evaluations/{job}-active.json')
    if milestone.get('stage') in ('evaluation','failed'):
        directory=Path(milestone['output'])/'rollouts'
        records=[monitor.read(p) for p in directory.glob('*.json')]
        records=[record for record in records if record]
        row.update(kind='eval',stage=milestone['stage'],evaluation_iteration=milestone['iteration'],
            evaluation_output=milestone['output'],completed_tasks=len(records),
            saved_rollouts=len(list(directory.glob('*.pt'))),
            successes=sum(record.get('metrics',{}).get('successes',0) for record in records),
            valid=sum(record.get('metrics',{}).get('valid_trajectories',0) for record in records))
        if milestone['stage']=='failed':
            row['alerts'].append('Queued milestone evaluation failed; preserve saved tasks for recovery')
    if status.get('error_type'):
        row['alerts'].append('Continuation controller failed: '+status['error_type'])
        row['stage']='failed'
    if state in monitor.TERMINAL and not row['complete']:
        row['alerts'].append(f'Allocation ended before iteration {target} plus full-300 evaluation completed')
    return row


def quality_snapshot(root, slurm, full=False):
    status=monitor.read(root/'status.json')
    state=slurm.get('state','UNKNOWN').split()[0].rstrip('+')
    stages={actor:len(list((root/actor/'records').glob('*.json')))
            for actor in ['sft','baseline20','baseline90']}
    labels=[monitor.read(p) for p in (root/'teacher-labels').glob('*.json')]
    tickets=[monitor.read(p) for p in (root/'teacher-budget').glob('*.json')]
    valid=sum(bool(r.get('valid')) for r in labels)
    summary=monitor.read(root/'quality-summary.json')
    row=dict(stage=status.get('stage','queued' if state=='PENDING' else 'startup'),
        completed_tasks=sum(stages.values()),candidate_sets_by_actor=stages,
        teacher_labels=valid,teacher_attempt_records=len(labels),
        api_charged_or_reserved_usd=sum(r.get('charged_usd',0) for r in tickets),
        complete=bool(status.get('complete') and summary),alerts=[],
        progress_marker=(sum(stages.values()),len(labels),len(tickets)),
        gpu_restored_actors=[actor for actor in ['baseline20','baseline90']
                            if (root/actor/'checkpoint_restore_evidence.json').is_file()])
    if row['complete']:
        row.update(stage='complete',quality_summary=summary)
    if any(r and not r.get('valid') for r in labels):
        row['alerts'].append('Teacher labeling failure recorded; audit controller should stop GPU workers')
    if status.get('failed'):
        row['alerts'].append('Audit failed: '+str(status.get('error_type')))
    if state in monitor.TERMINAL and not row['complete']:
        row['alerts'].append('Audit ended before verified collection and labeling completion')
    if full:
        row['wandb_by_actor']={actor:eval_tracking(root/actor) for actor in stages
            if (root/actor/'evaluation_manifest.json').is_file()}
    return row


def task_success_snapshot(root,slurm,full=False):
    status=monitor.read(root/'status.json');state=slurm.get('state','UNKNOWN').split()[0].rstrip('+')
    plan=monitor.read(root/'audit-plan.json')
    actors=[s['actor'] for s in plan.get('stages',[])] or ['baseline20','baseline90']
    stages={a:len(list((root/a/'compare_records').glob('*.json'))) for a in actors}
    budget=monitor.read(root/'terminal-budget/ledger.json');summary=monitor.read(root/'task-success-summary.json')
    row=dict(stage=status.get('stage','queued' if state=='PENDING' else 'startup'),
        completed_tasks=sum(stages.values()),planned_tasks=plan.get('primary_trajectories',200),attempts_by_actor=stages,
        api_charged_or_reserved_usd=budget.get('charged_or_reserved_usd',0),
        terminal_calls=budget.get('calls',0),complete=bool(status.get('complete') and summary),alerts=[],
        progress_marker=(sum(stages.values()),budget.get('calls',0)))
    if row['complete']:row.update(stage='complete',task_success_summary=summary)
    if status.get('failed'):row['alerts'].append('Task-success audit failed: '+str(status.get('error_type')))
    if (root/'terminal-budget/halt.json').exists():row['alerts'].append('Terminal judge budget halted; inspect saved trajectories')
    if state in monitor.TERMINAL and not row['complete']:row['alerts'].append('Task-success audit ended before all planned ARM runs completed')
    if full:
        row['wandb_by_actor']={a:eval_tracking(root/a) for a in stages if (root/a/'evaluation_manifest.json').is_file()}
        if state=='RUNNING' and row['completed_tasks']:
            row['resources']=monitor.resource_sample(slurm['job'],gpus=2)
    return row


def mig_snapshot(root, slurm):
    status=monitor.read(root/'status.json');summary=monitor.read(root/'probe-result.json')
    state=slurm.get('state','UNKNOWN').split()[0].rstrip('+')
    records=list(root.glob('*_records/*.json'))
    devices={name:monitor.read(root/f'{name}-device.json') for name in ('actor','selector')}
    row=dict(stage=status.get('stage','queued' if state=='PENDING' else 'startup'),
        complete=bool(status.get('complete') and summary.get('passed')),alerts=[],
        completed_tasks=len(records),devices=devices,probe_summary=summary,
        context_capacity_passed=bool(monitor.read(root/'context-capacity.json').get('passed')),
        progress_marker=(len(records),len(list((root/'offline-selections').glob('*.json'))),
                         sum(bool(d) for d in devices.values())))
    if status.get('error_type'):
        row['alerts'].append('MIG pilot failed: '+status['error_type']+': '+status.get('error',''))
    if state in monitor.TERMINAL and not row['complete']:
        row['alerts'].append('MIG pilot ended before all capacity, selection and browser checks passed')
    return row


def coverage_snapshot(root, slurm, full=False):
    """A successful coverage audit has zero Adam updates; do not label it stalled training."""
    state=slurm.get('state','UNKNOWN').split()[0].rstrip('+')
    status=monitor.read(root/'status.json')
    config=monitor.read(root/'arm-config.json')
    iteration=Path(config.get('output',root))
    coverage=monitor.read(iteration/'failure-coverage.json')
    restore=monitor.read(RUNTIME/'evaluations/arm-failure-additive-311962'/
                         f'gpu-restore-coverage-{slurm["job"]}-4gpu/result.json')
    row=dict(stage=status.get('stage','queued' if state=='PENDING' else 'checkpoint restore/startup'),
        complete=bool(status.get('shadow_complete') and coverage.get('complete')
                      and status.get('no_optimizer_updates') is True),
        optimizer_updates_executed=0 if status.get('no_optimizer_updates') else None,
        coverage_summary=coverage, gpu_checkpoint_restored=restore.get('passed',False),alerts=[])
    if root.exists():
        data=monitor.snapshot(slurm['job'],slurm,remote=full,compare_baseline=False)
        for key in ('collection_progress','resources','wandb'):
            if key in data:row[key]=data[key]
        row['alerts'].extend(data['alerts'])
        row['failure_labels_completed']=len(list((iteration/'failure_labels').glob('*.json')))
        live=monitor.read(iteration/'live_metrics.json')
        row['recorded_turns']=live.get('arm_collection/recorded_turns',0)
        row['progress_marker']=(row['recorded_turns'],row['failure_labels_completed'],bool(coverage))
    if status.get('no_optimizer_updates') is False:
        row['alerts'].append('Coverage-only pilot reports actor optimizer updates')
    if state in monitor.TERMINAL and not row['complete']:
        row['alerts'].append('Coverage pilot ended before verified collection and deferred-label completion')
    return row


def main():
    from report_arm_jobs import write_dashboard
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=36)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    CONTROL.mkdir(parents=True,exist_ok=True)
    lock = (CONTROL/'owner.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    load_registered_jobs()
    write_json(CONTROL/'process.json',dict(pid=os.getpid(),jobs=JOBS,hours=args.hours,
        new_allocations_authorized=False,snapshot_seconds=900,cheap_failure_check_seconds=60))
    end = time.monotonic()+args.hours*3600
    next_full = 0
    previous_key = None
    progress = {}
    seen_live = set(monitor.read(CONTROL/'seen-live-jobs.json').get('jobs',[]))
    storage = {}
    while time.monotonic() < end:
        now = time.monotonic()
        full = now >= next_full
        try:
            load_registered_jobs()
            if full or not storage:
                storage = quota_snapshot()
                write_json(CONTROL/'storage-quota.json', storage)
            live_jobs = discover_live_jobs()
            seen_live.update(j['job'] for j in live_jobs if j['job'] in JOBS)
            write_json(CONTROL/'seen-live-jobs.json',dict(jobs=sorted(seen_live)))
            states = scheduler_states(live_jobs)
            rows = []
            for job in JOBS:
                try:
                    row = snapshot(job,states.get(job,dict(job=job,state='UNKNOWN')),full)
                    signature = (row.get('durable_iteration'), row.get('collection_progress',{}).get('accepted_groups'),
                        row.get('latest_train',{}).get('train/step'), row.get('completed_tasks'), row.get('stage'),
                        row.get('progress_marker'))
                    old = progress.get(job)
                    if old is None or old[0] != signature:progress[job] = (signature,time.time())
                    row['seconds_without_progress_change'] = int(time.time()-progress[job][1])
                    if (row['slurm'].get('state') == 'RUNNING' and row.get('stage') != 'complete'
                            and not row.get('complete') and row['seconds_without_progress_change'] > 1800):
                        row['alerts'].append('No measured progress for over 30 minutes; inspect before restarting')
                    if row['slurm'].get('state') in ('RUNNING','PENDING'):
                        row['alerts'].extend(storage.get('alerts', []))
                    rows.append(row)
                except Exception as exc:
                    rows.append(dict(job=job,slurm=states.get(job,{}),
                        alerts=['Monitor snapshot failed: '+type(exc).__name__+': '+str(exc)[:200]]))
            report = dict(checked_utc=datetime.now(timezone.utc).isoformat(),full_snapshot=full,jobs=rows,live_slurm_jobs=live_jobs,storage_quota=storage,
                checkpoint_watcher=monitor.read(RUNTIME/'arm-turn-bonus-preparation/iteration80-evals-20260919/allfailure-watcher-status.json'))
            key = [(row['job'],row.get('slurm',{}).get('state'),row.get('stage'),row.get('alerts')) for row in rows]
            write_json(CONTROL/'latest.json',report)
            try:
                write_dashboard(report, CONTROL, monitor.REPO/'openwebrl/docs/arm_results/rl_integration')
            except Exception as exc:
                write_json(CONTROL/'report-error.json',dict(type=type(exc).__name__,at_epoch=time.time()))
            if full or key != previous_key:
                with (CONTROL/'history.jsonl').open('a') as handle:handle.write(json.dumps(report)+'\n')
                print(json.dumps(report),flush=True)
                previous_key = key
            alerts = [dict(job=row['job'],alerts=row['alerts']) for row in rows if row['alerts']]
            write_json(CONTROL/'alerts.json',dict(checked_utc=report['checked_utc'],alerts=alerts))
            actionable=[dict(job=row['job'],stage=row.get('stage'),alerts=row['alerts'])
                for row in rows if row['alerts'] and
                (row.get('slurm',{}).get('state') in ('RUNNING','PENDING') or
                 row['job'] in seen_live)]
            write_json(CONTROL/'intervention-needed.json',dict(checked_utc=report['checked_utc'],
                alerts=actionable,automatic_code_repair=False))
            if full:next_full = time.monotonic()+900
        except Exception as exc:
            write_json(CONTROL/'supervisor-error.json',dict(type=type(exc).__name__,at_epoch=time.time()))
        if args.once:return
        time.sleep(60)


if __name__ == '__main__':
    main()
