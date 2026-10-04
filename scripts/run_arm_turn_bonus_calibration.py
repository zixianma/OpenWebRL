#!/usr/bin/env python3
"""Calibrate ARM, optionally train the gated first batch; dry run by default."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

from evaluate_baseline_checkpoint import REPO, RUNTIME, build_plan, record_restore_evidence, restore_pattern
from resume_baseline import allocation, clean_environment, source_command, write_json

sys.path.insert(0,str(REPO))
from openwebrl.arm_turn_bonus_runtime import file_sha256, verify_torch_archive

SOURCE = RUNTIME / 'reference-arm-turn-bonus-calibration-20260913-v8'
CHECKPOINT = RUNTIME / 'runs/openwebrl-4b-reference-290926-20260912T185301/iter_0000069'
SELECTOR = Path('/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89')


def plan(job, attempt=0, phase='calibrate', replay=None, beta=.5, gpus=2):
    if phase not in ('calibrate','calibrate-train','replay-train') or beta not in (0.,.5):
        raise ValueError('Unsupported first-batch phase/beta')
    training = phase != 'calibrate'
    suffix = f'-r{attempt}' if attempt else ''
    output = RUNTIME / f'evaluations/arm-turn-bonus-calibration-{job}{suffix}'
    p = build_plan(SOURCE, CHECKPOINT, output, job, gpus=gpus)
    browsers = gpus * 8
    import yaml
    p['browser_config'] = yaml.safe_load((SOURCE/'openwebrl/browser_training_config.yaml').read_text())
    p['browser_config']['browser_rollout_concurrency'] = browsers
    run_id = f'arm-turn-bonus-calibration-after70-{job}{suffix}'
    if training:
        run_id = f'arm-turn-bonus-beta{beta:g}-after70-{job}{suffix}'
    p.update(protocol='Executed-turn ARM calibration / gated first batch; GPT-4.1/action_history',
             optimizer_updates_requested='One native batch, two PPO epochs, only if gates pass' if training else 0,
             wandb_run_id=run_id,phase=phase,note='No evaluation tasks used for calibration.')
    p.pop('metric_prefix', None)
    p['environment'].update(NUM_ROLLOUT=str(p['checkpoint_index']+2),
        BROWSER_CONCURRENCY=str(browsers), SGLANG_CONCURRENCY='48' if gpus==4 else '32', WANDB_RUN_ID=run_id,
        BROWSER_TRAIN_CONFIG=str(output/'browser-training-config.json'),
        FLASHINFER_WORKSPACE_BASE=str(RUNTIME),
        OPENWEBRL_ARM_TURN_BONUS_CONFIG=str(output/'arm-config.json'),
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(output/'persistent-multimodal'))
    command = p['command']
    command[command.index('--wandb-project')+1] = 'openwebrl'
    command[command.index('--wandb-group')+1] = 'executed-turn-bonus-pilot' if training else 'executed-turn-bonus-calibration'
    command[command.index('--eval-interval')+1] = '0'
    command += ['--custom-generate-function-path', 'openwebrl.arm_turn_bonus.generate',
                '--sglang-mem-fraction-static', '.35']
    p['arm_config'] = dict(shadow_only=not training, train_after_calibration=training,
        beta=beta, scored_fraction=.2, k=5, seed=42,seconds_per_optimizer_update=180,
        max_pending_per_trajectory=2, label_timeout_seconds=120,
        selector_endpoint='http://127.0.0.1:27511', selector_checkpoint=str(SELECTOR),
        checkpoint=str(CHECKPOINT), policy_id='qcq7i4ug:after70:arm-shadow-v1',
        output=str(output), evaluation_tasks=str(REPO/'openwebrl/data/eval/online-mind2web.jsonl'))
    # End at a durable first-batch checkpoint without pushing weights for an
    # unrequested next collection. Both beta branches retain the same q/K.
    if training:
        p['environment']['OPENWEBRL_STOP_AFTER_SAVED_ROLLOUT'] = str(p['checkpoint_index']+1)
    if phase == 'replay-train':
        if replay is None:
            raise ValueError('Replay requires a completed calibration directory')
        p['replay'] = validate_replay(replay)
        p['arm_config']['replay_origin'] = p['replay']
        p['environment'].update(OPENWEBRL_REPLAY_FIRST_BATCH=p['replay']['recovery_file'],
            OPENWEBRL_REPLAY_ROLLOUT_ID=str(p['checkpoint_index']+1),
            OPENWEBRL_ARM_REPLAY_CURSOR=p['replay']['dataset_cursor'])
    elif replay is not None:
        raise ValueError('Replay directory is only valid in replay-train phase')
    p['attempt'] = attempt
    p['requested_resources'] = dict(gpus=gpus, gpu_type='H200', hours=2, gpu_hours=gpus*2,
        cpus=gpus*8, memory_gib=480, browsers=browsers, estimated_gpu_cost_usd=round(gpus*2*.90,2),
        cost_basis='Previous project estimate of $0.90/H200-hour; not a new price quote',
        judge_usage_additional=True, approval_required=True)
    p['decision_rule'] = dict(min_admitted=100, min_distinct_tasks=20,
        keep_q20_if_effective_fraction_at_least=.05,
        keep_beta05_if_rms_ratio_between=[.04,.10],
        otherwise='Report beta targeting 7% RMS, clipped to [0.2,1.0]; low coverage is not repaired by increasing beta.',
        overhead='Measure request/token counts and queue failures; causal wall-time overhead needs matched q=0 control.')
    return p


def validate_replay(folder):
    from openwebrl.arm_turn_bonus_runtime import calibration_decision
    folder=Path(folder).resolve()
    report=json.loads((folder/'calibration.json').read_text())
    complete=json.loads((folder/'collection_complete.json').read_text())
    status=json.loads((folder/'status.json').read_text())
    config=json.loads((folder/'arm-config.json').read_text())
    if (status.get('failed') or status.get('exit_code')!=0
            or not status['collection_complete'] or not status['no_optimizer_updates']
            or not config['shadow_only'] or not calibration_decision(report)['passed']):
        raise ValueError('Replay requires a successful, untrained calibration that passes all gates')
    if any(Path(value).resolve()!=CHECKPOINT for value in (report['checkpoint'],complete['checkpoint'],config['checkpoint'])):
        raise ValueError('Replay actor checkpoint mismatch')
    expected='qcq7i4ug:after70:arm-shadow-v1'
    if report['policy_id']!=expected or config['policy_id']!=expected or complete['rollout_id']!=70:
        raise ValueError('Replay policy/cursor identity mismatch')
    if config['k']!=5 or config['scored_fraction']!=.2 or Path(config['selector_checkpoint'])!=SELECTOR:
        raise ValueError('Replay teacher/candidate recipe mismatch')
    for name in ('recovery_file','dataset_cursor'):
        path=Path(complete[name]).resolve()
        if not path.is_relative_to(folder):
            raise ValueError('Replay artifact outside its collection directory')
        verify_torch_archive(path)
    return dict(source=str(folder),recovery_file=complete['recovery_file'],
        dataset_cursor=complete['dataset_cursor'],report_sha256=file_sha256(folder/'calibration.json'),
        cursor_sha256=file_sha256(complete['dataset_cursor']),
        recovery_sha256=file_sha256(complete['recovery_file']))


def selector_preflight(endpoint, output, expected_api_model=None):
    """Exercise the real decoder/compiler before starting expensive browsers."""
    import base64, io
    from PIL import Image
    from openwebrl.arm_inference import parse_selection
    with urllib.request.urlopen(endpoint+'/health',timeout=10) as response:
        health=json.load(response)
    model_ok = (health.get('model') == expected_api_model if expected_api_model
                else Path(health.get('model','')).resolve()==SELECTOR)
    if health.get('mode')!='selection' or not model_ok:
        raise ValueError('Unexpected selector service/model on requested port')
    png=io.BytesIO()
    Image.new('RGB',(256,256),'white').save(png,format='PNG')
    payload=dict(mode='selection',task='Choose a reasonable next browser action.',url='about:blank',history=[],
        candidates=[dict(thought=f'Consider target {i}.',action='<tool_call>'+json.dumps(
            dict(name='click',arguments={'x':20+i*20,'y':40}))+'</tool_call>') for i in range(5)],
        screenshot=base64.b64encode(png.getvalue()).decode(),candidate_representation='full')
    request=urllib.request.Request(endpoint+'/select',data=json.dumps(payload).encode(),
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=120) as response:
        result=json.load(response)
    selected=parse_selection(result.get('raw',''),5)
    write_json(Path(output)/'selector_preflight.json',dict(passed=True,selected_index=selected,
        health=health,result=result,limitation='Protocol/compiler smoke test, not selector accuracy.'))


def health_issue(live, log):
    if 'ModuleNotFoundError:' in log:
        return 'Missing runtime module in collection; stop before retrying tasks'
    if (live.get('arm_collection/native_failure_sentinels',0)>=100
            and live.get('arm_collection/recorded_turns',0)==0):
        return 'Repeated native failures without any completed turns'
    if live.get('arm_collection/recorded_turns',0)>=100 and live.get('arm_collection/sampled_turns',0)==0:
        return 'No sampled ARM turns after 100 records'
    if live.get('arm_collection/sampled_turns',0)>=100 and live.get('arm_collection/admitted_turns',0)==0:
        return 'No usable ARM labels after 100 sampled turns'
    if re.search(r"(?:grad_norm|train/(?:loss|pg_loss))['\"\s:=]+(?:nan|[+-]?inf)\b",log,re.I):
        return 'Nonfinite training loss or gradient'
    return None


def execute(p):
    job = p['job_id']
    if os.getenv('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Execute only inside the dedicated authorized GPU step')
    requested = p['requested_resources']
    resources = allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),
                           job, requested_gpus=requested['gpus'])
    if resources['cpus'] < requested['cpus'] or resources['allocated_memory_gib'] < requested['memory_gib']:
        raise ValueError(f"Prepared profile needs {requested['cpus']} CPUs and {requested['memory_gib']} GiB")
    deadline = time.monotonic()+min(resources['maximum_seconds'], requested['hours']*3600-180)
    p['arm_config']['deadline_epoch_seconds'] = time.time()+deadline-time.monotonic()
    steps = subprocess.check_output(['squeue','--steps',f'--jobs={job}','--noheader','--format=%i'],text=True).split()
    allowed = {f'{job}.{x}' for x in ('batch','extern',os.getenv('SLURM_STEP_ID'))}
    # A supervised repair may retain the original batch controller to preserve
    # paid time. Only the explicitly recorded, stopped supervisor step is allowed.
    if p.get('attempt'):
        old = RUNTIME / f'evaluations/arm-turn-bonus-calibration-{job}'
        hold = json.loads((old/'supervisor-handoff-hold.json').read_text())
        process = Path('/proc')/str(hold['held_supervisor_pid'])
        if hold['job_id'] != job or '/job_'+job+'/' not in (process/'cgroup').read_text():
            raise ValueError('Handoff supervisor provenance mismatch')
        if not re.search(r'^State:\s+T', (process/'status').read_text(), re.M):
            raise ValueError('Original supervisor must remain held until repair finishes')
        import zipfile
        recovery = old/'runtime/rollout_recovery/70.pt'
        with zipfile.ZipFile(recovery) as archive:
            if not any(x.endswith('/data.pkl') for x in archive.namelist()):
                raise ValueError('Original recovery batch is incomplete')
        allowed.add(job+'.0')
    if set(steps)-allowed:
        raise ValueError('Another step is active in this allocation')
    from dotenv import dotenv_values
    env = {k:v for k,v in clean_environment().items() if not k.startswith('OPENWEBRL_ARM_')}
    for k,v in dotenv_values(REPO/'.env').items():
        if v and k.startswith(('WANDB_','JUDGE_','OPENAI_','AZURE_')):
            env.setdefault(k,v)
    if env.get('OPENAI_API_KEY') and not env.get('JUDGE_API_BASE'):
        env.update(JUDGE_API_MODE='served',JUDGE_API_BASE='https://api.openai.com/v1')
    env.update(p['environment'])
    if not env.get('WANDB_API_KEY'):
        raise ValueError('W&B credentials required')
    if p['phase']!='replay-train' and not any(env.get(k) for k in ('OPENAI_API_KEY','JUDGE_API_KEY','AZURE_OPENAI_API_KEY')):
        raise ValueError('Training-judge credentials required before starting GPUs')
    output = Path(p['output'])
    output.mkdir(parents=True, exist_ok=False)
    view = output/'checkpoint-view'
    view.mkdir()
    (view/CHECKPOINT.name).symlink_to(CHECKPOINT, target_is_directory=True)
    (view/'rollout').symlink_to(CHECKPOINT.parent/'rollout', target_is_directory=True)
    (view/'latest_checkpointed_iteration.txt').write_text(str(p['checkpoint_index'])+'\n')
    write_json(output/'calibration_manifest.json', p)
    write_json(output/'arm-config.json', p['arm_config'])
    # JSON is valid YAML for native --custom-config-path. Set the task gate
    # explicitly: the preserved YAML takes precedence over the pool environment.
    write_json(output/'browser-training-config.json', p['browser_config'])
    inspection = source_command(SOURCE, ['python', str(REPO/'scripts/inspect_training_checkpoint.py'),
        str(view), '--expected-scheduler-offset-updates','1', '--report',str(output/'checkpoint_validation.json')])
    children, handles = [], []
    worker = selector = None
    stage = 'checkpoint-inspection'
    result = dict(collection_complete=False,training_complete=False,no_optimizer_updates=True,
        phase=p['phase'],stage=stage)
    def status(**values):
        result.update(values)
        result['updated_at_epoch_seconds']=time.time()
        write_json(output/'status.json',result)
    def stop(*unused):
        for child in reversed(children):
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    def interrupted(*unused):
        raise InterruptedError('Allocation termination requested')
    old_sigterm=signal.signal(signal.SIGTERM, interrupted)
    old_sigint=signal.signal(signal.SIGINT, interrupted)
    try:
        status(stage=stage)
        subprocess.run(inspection, env=env, check=True, stdout=subprocess.DEVNULL,
            timeout=min(180,max(1,deadline-time.monotonic())))
        devices = env.get('CUDA_VISIBLE_DEVICES','').split(',')
        if len(devices) != requested['gpus'] or not all(devices) or len(set(devices)) != len(devices):
            raise ValueError(f"Expected exactly {requested['gpus']} distinct assigned CUDA devices")
        if p['phase'] != 'replay-train':
            stage='selector-preflight'
            status(stage=stage)
            handles.append((output/'selector.log').open('w'))
            selector_env = dict(env, CUDA_VISIBLE_DEVICES=devices[-1], PYTHONPATH=str(SOURCE))
            selector_env['CPATH'] = str(RUNTIME/'src/python-headers/Include')+':'+str(RUNTIME/'src/python-headers')
            selector = subprocess.Popen([str(RUNTIME/'arm-reproduction/venv/bin/python'),str(SOURCE/'scripts/serve_arm.py'),
                '--mode','selection','--model',str(SELECTOR),'--port','27511'], cwd=SOURCE, env=selector_env,
                stdout=handles[-1], stderr=subprocess.STDOUT, start_new_session=True)
            children.append(selector)
            startup_end = min(deadline-120,time.monotonic()+600)
            while True:
                if selector.poll() is not None:
                    raise RuntimeError('Selector failed to load')
                try:
                    with urllib.request.urlopen(p['arm_config']['selector_endpoint']+'/health',timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > startup_end:
                    raise TimeoutError('Selector startup timeout')
                time.sleep(5)
            selector_preflight(p['arm_config']['selector_endpoint'],output)
        else:
            if validate_replay(p['replay']['source'])!=p['replay']:
                raise ValueError('Replay artifacts changed since plan validation')
        remaining = int(deadline-time.monotonic())
        if remaining < 600:
            raise TimeoutError('Insufficient collection time remains')
        command = source_command(SOURCE, ['timeout','--signal=INT','--kill-after=60',str(remaining),*p['command']])
        handles.append((output/'collection.log').open('w'))
        worker = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=handles[-1],
            stderr=subprocess.STDOUT, start_new_session=True)
        children.append(worker)
        stage='collection'
        status(stage=stage)
        while worker.poll() is None:
            if time.monotonic()>=deadline:
                raise TimeoutError('Allocation deadline reached')
            gate_path=output/'training_gate.json'
            if gate_path.is_file() and json.loads(gate_path.read_text())['ready']:
                stage='training'
                # No new browser collection is permitted in this first-batch pilot.
                # Release teacher memory before checkpointing/optimization.
                if selector is not None:
                    if selector.poll() is None:
                        os.killpg(selector.pid,signal.SIGTERM)
                        selector.wait(timeout=30)
                    selector=None
            if selector is not None and selector.poll() is not None:
                raise RuntimeError('Selector exited during collection')
            record_restore_evidence(p, log_filename='collection.log')
            live_path = output/'live_metrics.json'
            live=json.loads(live_path.read_text()) if live_path.is_file() else {}
            with (output/'collection.log').open('rb') as handle:
                handle.seek(0,2)
                handle.seek(max(0,handle.tell()-65536))
                recent=handle.read().decode(errors='replace')
            if issue:=health_issue(live,recent):
                raise RuntimeError(issue)
            if re.search(r'train_one_step start|\[TrainMetrics\]',recent):
                result['no_optimizer_updates']=False
            status(stage=stage,live_metrics=live,seconds_remaining=int(deadline-time.monotonic()))
            time.sleep(15)
        record_restore_evidence(p, log_filename='collection.log')
        log = (output/'collection.log').read_text(errors='replace')
        receipt = output/'checkpoint_restore_evidence.json'
        evidence = json.loads(receipt.read_text()).get('restore_line','') if receipt.exists() else log
        restored = bool(re.search(restore_pattern(p), evidence))
        no_training = not re.search(r'train_one_step start|\[TrainMetrics\]', log)
        collection = output/'collection_complete.json'
        complete = collection.is_file() and restored
        trained=False
        if p['phase']=='calibrate' and not no_training:
            raise RuntimeError('Unexpected optimizer update during calibration-only mode')
        if not no_training:
            checkpoint_report=output/'trained_checkpoint_validation.json'
            validation=source_command(SOURCE,['python',str(REPO/'scripts/inspect_training_checkpoint.py'),
                str(output/'runtime'),'--expected-scheduler-offset-updates','1','--report',str(checkpoint_report)])
            subprocess.run(validation,env=env,check=True,stdout=subprocess.DEVNULL,
                timeout=min(180,max(1,deadline-time.monotonic())))
            before=json.loads((output/'checkpoint_validation.json').read_text())['completed_optimizer_updates']
            after=json.loads(checkpoint_report.read_text())['completed_optimizer_updates']
            expected=json.loads((output/'training_gate.json').read_text())['expected_optimizer_updates']
            marker=(output/'runtime/latest_checkpointed_iteration.txt').read_text().strip()
            trained=after-before==expected and marker==str(p['checkpoint_index']+1)
            if not trained:
                raise RuntimeError('Final optimizer/checkpoint counters disagree with first-batch plan')
        status(collection_complete=complete,exit_code=worker.returncode,stage='complete',
            selected_checkpoint_restored=restored,no_optimizer_updates=no_training,training_complete=trained,
            next_action=('Evaluate the saved ARM checkpoint' if trained else
                'Review calibration and any training_gate.json before further training'))
        if worker.returncode!=0:
            raise RuntimeError('Collection/training worker failed; inspect preserved group journal and logs')
        if not complete:
            raise RuntimeError('Shadow collection gate incomplete; preserve artifacts and inspect logs')
    except BaseException as exc:
        log=(output/'collection.log').read_text(errors='replace') if (output/'collection.log').exists() else ''
        status(stage=stage,failed=True,error_type=type(exc).__name__,error=str(exc)[:500],
            no_optimizer_updates=not bool(re.search(r'train_one_step start|\[TrainMetrics\]',log)),
            exit_code=worker.poll() if worker is not None else None,
            next_action='Inspect this attempt and its incremental groups; do not discard or silently rerun it.')
        raise
    finally:
        stop()
        for child in children:
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        for handle in handles:
            handle.close()
        signal.signal(signal.SIGTERM,old_sigterm)
        signal.signal(signal.SIGINT,old_sigint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', default='APPROVED_JOB')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--attempt', type=int, default=0)
    parser.add_argument('--phase',choices=['calibrate','calibrate-train','replay-train'],default='calibrate')
    parser.add_argument('--replay',type=Path)
    parser.add_argument('--beta',type=float,choices=[0.,.5],default=.5)
    parser.add_argument('--gpus',type=int,choices=[2,4],default=2,
        help='2 H200/16 browsers or 4 H200/32 browsers; both have a two-hour cap')
    args = parser.parse_args()
    p = plan(args.job_id,args.attempt,args.phase,args.replay,args.beta,args.gpus)
    if args.execute:
        execute(p)
    else:
        print(json.dumps(p, indent=2))
