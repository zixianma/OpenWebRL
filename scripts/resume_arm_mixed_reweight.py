#!/usr/bin/env python3
"""Resume the matched mixed-only pair within recorded unused approvals."""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import prepare_arm_mixed_reweight as original
from resume_arm_turn_bonus import continuation_identity
from resume_baseline import allocation, clean_environment, source_command, validate_source, write_json

REPO, RUNTIME, PYTHON = original.REPO, original.RUNTIME, original.PYTHON
CONTROL = original.CONTROL/'continuations'
SOURCE = RUNTIME/'reference-arm-mixed-reweight-replay-20260928-v1'


def read(path):
    return json.loads(Path(path).read_text())


def origin(root):
    root = Path(root).resolve()
    if not root.is_relative_to(RUNTIME/'evaluations'):
        raise ValueError('Unexpected origin location')
    m, s = read(root/'launch_manifest.json'), read(root/'status.json')
    if (m.get('mixed_reweight_variant') not in original.MODES or m.get('diagnostic_only')
            or Path(m['source']) not in (original.SOURCE, SOURCE)
            or not (s.get('stage') == 'complete' or s.get('failed'))):
        raise ValueError('Requires an inactive matched-pair training attempt')
    validate_source(Path(m['source']))
    idx = int((root/'runtime/latest_checkpointed_iteration.txt').read_text())
    report = read(root/f'iterations/{idx:04d}/checkpoint-validation.json')
    cp = root/f'runtime/iter_{idx:07d}'
    if (report['checkpoint'] != str(cp) or report['completed_optimizer_updates'] != s['completed_optimizer_updates']
            or report['scheduler_minus_optimizer_updates'] != 0):
        raise ValueError('Durable counters disagree')
    for name, size in report['shard_files'].items():
        if (cp/name).stat().st_size != size:
            raise ValueError('Checkpoint shard changed')
    paths = [cp/'common.pt', cp/'.metadata', Path(report['dataset_cursor']), root/'runtime/latest_checkpointed_iteration.txt']
    return dict(root=str(root), manifest=m, checkpoint_report=report,
                identity_sha256={str(p): original.sha(p) for p in paths})


def replay_provenance(root):
    root = Path(root); o = origin(root); report = o['checkpoint_report']; rid = report['iteration'] + 1
    folder = root/'iterations'/f'{rid:04d}'
    if not (folder/'collection_complete.json').exists():
        return None
    complete, gate = read(folder/'collection_complete.json'), read(folder/'training_gate.json')
    batch, cursor = root/f'runtime/rollout_recovery/{rid}.pt', root/f'runtime/rollout/global_dataset_state_dict_{rid}.pt'
    if (complete['rollout_id'] != rid or complete['checkpoint'] != report['checkpoint']
            or complete['recovery_file'] != str(batch) or complete['dataset_cursor'] != str(cursor)
            or not gate['calibration']['passed'] or (root/f'runtime/iter_{rid:07d}').exists()):
        raise ValueError('Saved batch identity or calibration invalid')
    archives = {}
    for p in (batch, cursor):
        st = p.stat()
        with zipfile.ZipFile(p) as archive:
            if not any(n.endswith('/data.pkl') for n in archive.namelist()):
                raise ValueError('Incomplete batch or cursor archive')
        archives[str(p)] = dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns)
    return dict(source=str(root), rollout_id=rid, checkpoint=report['checkpoint'], batch=str(batch),
        dataset_cursor=str(cursor), cursor_sha256=original.sha(cursor), archives=archives,
        calibration_sha256=original.sha(folder/'calibration.json'),
        reweighting_sha256=original.sha(folder/'reweighting.json'))


def budget(job):
    b = read(CONTROL/f'{job}-budget.json')
    approval = read(original.CONTROL/'resource-approval.json')
    if (not approval['approved'] or approval['resources_each'] != original.RESOURCES
            or b['job_id'] != job or b['variant'] not in approval['variants']
            or b['approved_seconds'] != 28800 or b['prior_used_seconds'] < 0
            or b['attempt_seconds'] <= 0 or b['prior_used_seconds'] + b['attempt_seconds'] > 28800):
        raise ValueError('Unapproved budget extension')
    return b


def plan(job):
    b = budget(job); o = origin(b['resume_from']); prior = o['manifest']; r = o['checkpoint_report']
    if prior['mixed_reweight_variant'] != b['variant']:
        raise ValueError('Wrong matched-pair lineage')
    p = copy.deepcopy(prior); start = r['iteration'] + 1
    if not 0 < start < 10:
        raise ValueError('Only the approved first milestone remains')
    output = RUNTIME/f'evaluations/arm-mixed-{b["variant"]}-{job}-r{b["restart_count"]}'
    def rewrite(v):
        return v.replace(prior['output'], str(output)).replace(prior['source'], str(SOURCE)) if isinstance(v, str) else v
    p['command'] = [rewrite(v) for v in p['command']]
    for k in ('environment', 'arm_config'):
        p[k] = {x: rewrite(v) for x, v in p[k].items()}
    if '--use-checkpoint-opt-param-scheduler' not in p['command']:
        p['command'].append('--use-checkpoint-opt-param-scheduler')
    p.update(job_id=job, source=str(SOURCE), output=str(output), resume_from=b['resume_from'], resume_origin=o,
        fresh_optimizer=False, variant_continuation=True, start_rollout_id=start,
        initial_optimizer_updates=r['completed_optimizer_updates'], checkpoint=r['checkpoint'],
        requested_iterations=10-start, target_completed_iterations=10, evaluation_reserve_seconds=0,
        gpu_restore_output=str(Path(b['resume_from'])/f'gpu-restore-{job}-r{b["restart_count"]}'),
        recovery_budget=b)
    p['resume_restore_receipt'] = str(Path(p['gpu_restore_output'])/'result.json')
    p['requested_resources'].update(hours=b['attempt_seconds']/3600, gpu_hours=8*b['attempt_seconds']/3600)
    p['environment'].update(SLIME_LOAD_CHECKPOINT=str(Path(b['resume_from'])/'runtime'),
        WANDB_RESUME='must', NUM_ROLLOUT='10', OPENWEBRL_CUDA_CACHE_LIMIT_GIB='48',
        PYTHONDONTWRITEBYTECODE='1', RAY_TMPDIR=f'/tmp/am-{job}-r{b["restart_count"]}')
    scratch = str(RUNTIME/f'multimodal-scratch/arm-variant-{job}')
    p['environment']['OPENWEBRL_MULTIMODAL_STORAGE_DIR'] = scratch
    p['multimodal_storage']['directory'] = scratch
    for key in ('deadline_epoch_seconds', 'gate_replay_origin', 'replay_origin'):
        p['arm_config'].pop(key, None)
    for key in ('OPENWEBRL_REPLAY_FIRST_BATCH', 'OPENWEBRL_REPLAY_ROLLOUT_ID', 'OPENWEBRL_ARM_REPLAY_CURSOR'):
        p['environment'].pop(key, None)
    p['arm_config'].update(checkpoint=r['checkpoint'], policy_id=p['wandb_run_id']+':uninitialized',
        minimum_cycle_seconds=3000, seconds_per_optimizer_update=120)
    replay = replay_provenance(b['resume_from'])
    if replay != b['replay_provenance']:
        raise ValueError('Replay artifacts changed')
    if replay:
        p['arm_config'].update(gate_replay_origin=replay, replay_origin=replay)
        p['environment'].update(OPENWEBRL_REPLAY_FIRST_BATCH=replay['batch'],
            OPENWEBRL_REPLAY_ROLLOUT_ID=str(replay['rollout_id']), OPENWEBRL_ARM_REPLAY_CURSOR=replay['dataset_cursor'])
    c, e = p['arm_config'], p['environment']
    if ((c['advantage_mode'], c['beta'], c['scored_fraction'], c['reweight_lambda'], c['candidate_gate'],
         c['credit_assignment'], c['failure_group_cap'], c['additive_failure_groups']) !=
        (original.MODES[b['variant']], .5, .2, .5, 'min2', 'response_index', 0, False)
            or (e['NUM_GPUS'], e['TP_SIZE'], e['GLOBAL_BATCH_SIZE'], e['BROWSER_CONCURRENCY']) != ('8','2','256','64')
            or 'OPENWEBRL_ARM_FAILURE_AUX_MANIFEST' in e
            or p['command'][p['command'].index('--dynamic-sampling-filter-path')+1] != original.FILTER):
        raise ValueError('Scientific recipe changed')
    return p


def fingerprints():
    paths = [Path(__file__).resolve(), SOURCE/'reference_manifest.json', CONTROL/'training-controller.py', CONTROL/'restore-controller.py']
    return {str(p): original.sha(p) for p in paths}


def validate_plan(p):
    validate_source(SOURCE)
    if p != plan(p['job_id']) or read(CONTROL/'readiness.json')['fingerprints'] != fingerprints():
        raise ValueError('Recovery plan or tested implementation changed')
    return dict(passed=True, objective_unchanged=True, native_gpu_restore_required=True, replay_verified_before_training=True)


def validate_resume_plan(p, require_gpu_restore=False):
    if origin(p['resume_from']) != p['resume_origin']:
        raise ValueError('Resume identity changed')
    if require_gpu_restore:
        expected = dict(passed=True,job_id=p['job_id'],source=p['source'],
            source_checkpoint_root=str(Path(p['resume_from'])/'runtime'),
            checkpoint_completed_optimizer_updates=p['initial_optimizer_updates'],
            loaded_iteration=p['start_rollout_id']-1,next_rollout_id=p['start_rollout_id'],gpus=8,
            optimizer_updates_executed=0,browser_collections_executed=0,
            continuation_identity_sha256=continuation_identity(p))
        receipt = read(p['resume_restore_receipt'])
        if any(receipt.get(k) != v for k,v in expected.items()):
            raise ValueError('Native GPU restore evidence missing')


def validate_replayed_rewards(p, rollout_id):
    replay = p['arm_config'].get('replay_origin')
    if not replay or rollout_id != replay['rollout_id']:
        return
    old = Path(replay['source'])/'iterations'/f'{rollout_id:04d}'
    new = Path(p['output'])/'iterations'/f'{rollout_id:04d}'
    for filename, fingerprint in [('calibration.json','calibration_sha256'),('reweighting.json','reweighting_sha256')]:
        if original.sha(old/filename) != replay[fingerprint]:
            raise ValueError('Saved reward audit changed')
    before, after = read(old/'calibration.json'), read(new/'calibration.json')
    for key in ('rows','batch_rows','admitted','admitted_distinct_tasks','task_counts','label_reasons'):
        if before[key] != after[key]:
            raise ValueError('Replayed calibration differs: '+key)
    for key in ('outcome_rms','unit_bonus_rms'):
        if not math.isclose(before[key],after[key],rel_tol=1e-7,abs_tol=1e-9):
            raise ValueError('Replayed reward scale differs: '+key)
    before, after = read(old/'reweighting.json'), read(new/'reweighting.json')
    if before != after:
        raise ValueError('Reweighting calculation changed on saved batch')
    write_json(new/'replay-verified.json',dict(passed=True,original=str(old),rollout_id=rollout_id,
        reward_formula_and_statistics_preserved=True,cursor_sha256=replay['cursor_sha256']))


def prepare():
    from prepare_arm_turn_bonus import copy_plain
    validate_source(original.SOURCE); CONTROL.mkdir(parents=True,exist_ok=True)
    name = 'openwebrl/arm_turn_bonus_cycles.py'
    text = (original.SOURCE/name).read_text()
    marker = '    current.update(rollout_id=rollout_id, checkpoint=checkpoint,'
    addition = '''    replay = current.get('gate_replay_origin')
    if replay and rollout_id == replay['rollout_id']:
        if Path(checkpoint).resolve() != Path(replay['checkpoint']).resolve():
            raise ValueError('Replay actor differs from the saved batch actor')
        checkpoint = replay['checkpoint']
'''
    assert text.count(marker) == 1
    text = text.replace(marker,addition+marker)
    if not SOURCE.exists():
        shutil.copytree(original.SOURCE,SOURCE,symlinks=True,copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
        (SOURCE/name).write_text(text)
        manifest = read(original.SOURCE/'reference_manifest.json')
        manifest['recipe_files_sha256'][name] = original.sha(SOURCE/name)
        manifest['mixed_replay_recovery'] = dict(parent=str(original.SOURCE),changed_files=[name],objective_unchanged=True)
        write_json(SOURCE/'reference_manifest.json',manifest)
    if (SOURCE/name).read_text() != text:
        raise ValueError('Replay source changed')
    validate_source(SOURCE)
    controller = (original.CONTROL/'training-controller.py').read_text()
    controller = controller.replace('from resume_arm_failure_variants import validate_plan','from resume_arm_mixed_reweight import validate_plan')
    controller = controller.replace('from resume_arm_turn_bonus import validate_resume_plan','from resume_arm_mixed_reweight import validate_resume_plan')
    controller = controller.replace("elif request['phase']=='training':\n", "elif request['phase']=='training':\n                    from resume_arm_mixed_reweight import validate_replayed_rewards\n                    validate_replayed_rewards(p, request['rollout_id'])\n")
    restore = (RUNTIME/'arm-turn-bonus-preparation/gate-b90-20260927/restore-controller.py').read_text()
    restore = restore.replace('from resume_arm_gate_b90 import','from resume_arm_mixed_reweight import')
    restore = restore.replace("'arm-failure-additive-'))", "'arm-failure-additive-', 'arm-mixed-'))")
    for filename, content in [('training-controller.py',controller),('restore-controller.py',restore)]:
        compile(content,filename,'exec'); (CONTROL/filename).write_text(content)
    write_json(CONTROL/'readiness.json',dict(cpu_passed=True,fingerprints=fingerprints(),gpu_restore_pending=True))


def worker(job):
    p = plan(job); validate_plan(p)
    if os.getenv('SLURM_JOB_ID') != job:
        raise ValueError('Wrong allocation')
    from resume_arm_failure_variants import check_multimodal_storage
    check_multimodal_storage(p)
    manifest = CONTROL/f'{job}-launch.json'; write_json(manifest,p)
    probe = f"import sys,runpy;sys.path.insert(0,{str(REPO/'scripts')!r});runpy.run_path({str(CONTROL/'restore-controller.py')!r},run_name='__main__')"
    subprocess.run(source_command(SOURCE,[str(PYTHON),'-B','-c',probe,'--training-root',p['resume_from'],
        '--job-id',job,'--gpus','8','--output',p['gpu_restore_output'],'--continuation-plan',str(manifest),'--execute']),check=True)
    validate_resume_plan(p,require_gpu_restore=True)
    spec = importlib.util.spec_from_file_location('mixed_resume',CONTROL/'training-controller.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.execute(p)


def evaluate(job):
    import run_arm_iteration80_eval as ev
    from run_stage1_to100 import audit_rollouts
    p = plan(job); root = Path(p['output'])
    ep = ev.plan('original',job,10,root)
    old = ep['output']; out = str(RUNTIME/f'evaluations/arm-mixed-{p["mixed_reweight_variant"]}-iter10-{job}-r{p["recovery_budget"]["restart_count"]}')
    ep['command'] = [v.replace(old,out) for v in ep['command']]
    ep['environment'] = {k:v.replace(old,out) for k,v in ep['environment'].items()}
    ep.update(output=out,gpus=8,wandb_run_id=Path(out).name,parent_training_run=p['wandb_run_id'])
    ep['environment'].update(NUM_GPUS='8',TP_SIZE='2',WANDB_RUN_ID=ep['wandb_run_id'])
    ep = ev.evaluation.configure_evaluation_tracking(ep,'openwebrl')
    os.environ.update(OPENWEBRL_EXPECTED_SCHEDULER_OFFSET='0',OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT='300')
    ev.evaluation.run(ep,REPO/'.env')
    write_json(CONTROL/f'{job}-evaluation10-audit.json',audit_rollouts(Path(out)/'rollouts',Path(ep['source'])))


def execute(job):
    b = budget(job)
    if (os.getenv('SLURM_JOB_ID') != job or int(os.getenv('SLURM_RESTART_COUNT','0')) != b['restart_count']):
        raise ValueError('Wrong recovery attempt')
    allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,
        requested_gpus=8,maximum_hours=b['attempt_seconds']/3600)
    p = plan(job);validate_plan(p)
    out = RUNTIME/f'evaluations/arm-mixed-{b["variant"]}-controller-{job}-r{b["restart_count"]}'
    out.mkdir(exist_ok=False)
    write_json(out/'controller-plan.json',dict(job_id=job,training_root=p['output'],target=10,evaluations=[10]))
    write_json(out/'budget.json',b)
    write_json(out/'status.json',dict(stage='training',complete=False))
    try:
        subprocess.run(['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=64',
            '--gres=gpu:h200:8','--exact','--cpu-bind=none',str(PYTHON),'-B',str(Path(__file__).resolve()),
            '--job-id',job,'--worker'],check=True)
        from run_arm_iteration80_eval import checkpoint_ready
        capacity = allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,
            requested_gpus=8,maximum_hours=b['attempt_seconds']/3600)
        ready = checkpoint_ready('original',10,Path(p['output'])) is not None
        if ready and capacity['maximum_seconds'] >= 2100:
            write_json(out/'status.json',dict(stage='evaluation10',complete=False))
            subprocess.run(['srun',f'--jobid={job}','--nodes=1','--ntasks=1','--cpus-per-task=64',
                '--gres=gpu:h200:8','--exact','--cpu-bind=none',str(PYTHON),'-B',str(Path(__file__).resolve()),
                '--job-id',job,'--eval'],check=True)
            write_json(out/'status.json',dict(stage='complete',complete=True,evaluations_completed=[10]))
        else:
            write_json(out/'status.json',dict(stage='budget-stop',complete=False,
                training_status=read(Path(p['output'])/'status.json'),evaluation10_pending=True,
                checkpoint10_ready=ready,remaining_seconds=capacity['maximum_seconds']))
    except BaseException as exc:
        write_json(out/'status.json',dict(stage='failed',complete=False,error_type=type(exc).__name__,error=str(exc)[:500]))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--job-id')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--eval',action='store_true')
    a=parser.parse_args()
    if a.prepare:prepare()
    elif a.worker:worker(a.job_id)
    elif a.eval:evaluate(a.job_id)
    elif a.execute:execute(a.job_id)
    else:print(json.dumps(plan(a.job_id),indent=2))
