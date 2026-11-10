#!/usr/bin/env python3
"""Prepare and run ARM-guided in-group rescue on outcome-only GRPO (approved 2026-11-03).

Recipe: 48 native mixed groups, outcome-only GRPO advantages, from the original SFT at iteration 0
(fresh optimizer/scheduler/cursor). Every valid all-failure group launches one SelectionARM-guided
attempt on the same task (K=5 policy candidates per turn); a judged success replaces one failure and
the group joins the batch as an extra mixed group (openwebrl/arm_rescue_rl.py). ARM labels at q=0.2
are collected only for telemetry; their bonus is discarded.

Approved budgets (zixianma / urgent): smoke test 8 H200 x 1.5 h; main run 0 -> 30, 8 H200 x 32 h total
(24 h + 8 h) with full-300 evaluations at 10/20/30; GPT-4.1 judge cap $200.
Stages: --prepare (CPU: frozen source, plan, native parse, frozen tests); --execute in an allocation.

v2 (2026-11-04, after smoke 418271): no ARM label requests (telemetry_labels=false), a rescue-only training gate
(the ARM calibration gate does not apply), and Piotr's SelectionARM served by sglang (batched, CUDA graphs;
123/124 picks match the HF server) on the last GPU instead of the serial HF server. v1 artifacts stay under
reference-arm-rescue-20261103-v1 and rescue-20261103.

v3 (2026-11-06, after smoke 420434: 27/29 valid attempts, 0 rescued): stage ``paired`` (8 H200 x 4 h, needs its own
approval) runs 3 iterations with 2 guided + 2 unguided control attempts per valid all-failure group, to measure
whether SelectionARM guidance beats plain resampling (~6.4% expected per control attempt). Controls never enter the
batch. The rescue module refresh keeps v2 behaviour at the default attempt counts.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from resume_baseline import REPO, RUNTIME, allocation, clean_environment, source_command, validate_source, write_json

PARENT = RUNTIME/'reference-arm-mixed-async-label-20261002-v3'
SOURCE = RUNTIME/'reference-arm-rescue-20261104-v2'
CONTROL = RUNTIME/'arm-turn-bonus-preparation/rescue-20261104-v2'
PYTHON = RUNTIME/'venv/bin/python'
INITIAL = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
FILTER = 'openwebrl.arm_rescue_rl.filter_groups'
GENERATE = 'openwebrl.arm_rescue_rl.generate'
STAGES = {'smoke': dict(iterations=1, hours=1.5), 'paired': dict(iterations=3, hours=4), 'main': dict(iterations=30, hours=24)}
PAIRED = dict(guided_attempts=2, control_attempts=2, wait_seconds=1500)
RESCUE = dict(max_rescues_per_rollout=48, wait_seconds=900, selector_timeout=120, telemetry_labels=False)
SGLANG_SELECTOR_MODEL = Path('/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection-sglang-81b452d')
HOLD_IDLE_SECONDS, HOLD_TOTAL_IDLE_SECONDS = 1800, 3600


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patched_rollout(text):
    """Opt-in rescue hook in the frozen rollout loop (same edits as the repository copy)."""
    a = ("    dynamic_filter = (\n        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path"
         " is not None else None\n    )\n")
    b = '    pbar.close()\n    log_progress("done")\n'
    c = '    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"'
    if not all(text.count(x) == 1 for x in (a, b, c)):
        raise ValueError('Frozen rollout loop anchors changed')
    text = text.replace(a, a+'    rescue_enabled = bool(os.environ.get("OPENWEBRL_ARM_RESCUE_CONFIG"))\n'
                        '    if rescue_enabled:\n        args._arm_rescue_rollout_id = rollout_id\n')
    text = text.replace(b, '    if rescue_enabled:\n        from openwebrl.arm_rescue_rl import collect as collect_rescued\n'
                        '        data.extend(await collect_rescued(args, rollout_id))\n'+b)
    return text.replace(c, '    if rescue_enabled:\n        assert len(data) >= args.rollout_batch_size\n    else:\n    '+c)


def patched_post_process(text):
    old = ('                if os.environ.get("OPENWEBRL_ARM_TURN_BONUS_CONFIG"):\n'
           '                    from openwebrl.arm_outcome_reweight import post_process_rewards\n')
    if text.count(old) != 1:
        raise ValueError('Frozen reward post-process anchor changed')
    return text.replace(old, '                if os.environ.get("OPENWEBRL_ARM_RESCUE_CONFIG"):\n'
                        '                    from openwebrl.arm_rescue_rl import post_process_rewards\n'
                        '                    return post_process_rewards(self.args, samples, raw_rewards, normalized_rewards)\n'+old)


def patched_train(text):
    """Rescue runs use their own stage gate; ARM labels never reach the reward."""
    old = '                from openwebrl.arm_turn_bonus_runtime import before_optimizer\n'
    if text.count(old) != 1:
        raise ValueError('Frozen train-loop gate anchor changed')
    return text.replace(old, '                if os.environ.get("OPENWEBRL_ARM_RESCUE_CONFIG"):\n'
                        '                    from openwebrl.arm_rescue_rl import before_optimizer\n'
                        '                else:\n    '+old)


# Started by the controller in its own session (killpg stops both). Arguments: python, adapter, sglang port, port.
SGLANG_SELECTOR = (
    'set -u; PY=$1; ADAPTER=$2; SPORT=$3; PORT=$4\n'
    f'"$PY" -m sglang.launch_server --model-path {SGLANG_SELECTOR_MODEL} --host 127.0.0.1 --port "$SPORT" '
    '--dtype bfloat16 --tp 1 --mem-fraction-static .25 --context-length 32768 --grammar-backend xgrammar '
    '--cuda-graph-max-bs 48 &\nS=$!\n'
    'until curl -sf "http://127.0.0.1:$SPORT/health" >/dev/null; do kill -0 $S || exit 1; sleep 3; done\n'
    '"$PY" "$ADAPTER" serve --sglang "http://127.0.0.1:$SPORT" --port "$PORT" &\nA=$!\n'
    'wait -n $S $A; exit 1\n')


def patched_controller(text):
    old = ("                    selector_command=[str(RUNTIME/'arm-reproduction/venv/bin/python'),str(SOURCE/'scripts/serve_arm.py'),\n"
           "                        '--mode','selection','--model',str(SELECTOR),'--port',str(p['selector_port'])]\n")
    if text.count(old) != 1:
        raise ValueError('Controller selector anchor changed')
    return text.replace(old, "                    selector_command=['bash','-c',"+repr(SGLANG_SELECTOR)+",'sglang-selector',\n"
                        "                        str(RUNTIME/'venv/bin/python'),str(SOURCE/'scripts/serve_arm_sglang.py'),\n"
                        "                        str(p['selector_port']+1),str(p['selector_port'])]\n")


def prepare_source():
    from prepare_arm_turn_bonus import copy_plain
    validate_source(PARENT)
    changes = {'openwebrl/arm_rescue_rl.py': (REPO/'openwebrl/arm_rescue_rl.py').read_bytes(),
               'slime/rollout/sglang_rollout.py': patched_rollout((PARENT/'slime/rollout/sglang_rollout.py').read_text()).encode(),
               'slime/ray/rollout.py': patched_post_process((PARENT/'slime/ray/rollout.py').read_text()).encode(),
               'train.py': patched_train((PARENT/'train.py').read_text()).encode(),
               # Committed version (the working tree may hold another session's edits).
               'scripts/serve_arm_sglang.py': subprocess.check_output(
                   ['git', '-C', str(REPO), 'show', 'HEAD:scripts/serve_arm_sglang.py'])}
    for name, data in changes.items():
        compile(data, name, 'exec')
    if SOURCE.exists():
        # Refresh only the rescue module when it changes (bug fixes), recording the new hash.
        name = 'openwebrl/arm_rescue_rl.py'
        if (SOURCE/name).read_bytes() != changes[name]:
            (SOURCE/name).write_bytes(changes[name])
            manifest = read(SOURCE/'reference_manifest.json')
            manifest['recipe_files_sha256'][name] = hashlib.sha256(changes[name]).hexdigest()
            manifest['arm_rescue'].setdefault('revisions', []).append(
                dict(time=time.time(), file=name, sha256=manifest['recipe_files_sha256'][name]))
            write_json(SOURCE/'reference_manifest.json', manifest)
    if not SOURCE.exists():
        shutil.copytree(PARENT, SOURCE, symlinks=True, copy_function=copy_plain,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.browser_use_sessions'))
        manifest = read(PARENT/'reference_manifest.json')
        for name, data in changes.items():
            (SOURCE/name).write_bytes(data)
            manifest['recipe_files_sha256'][name] = hashlib.sha256(data).hexdigest()
        manifest['arm_rescue'] = dict(parent=str(PARENT), changed_files=list(changes), rescue=RESCUE,
            advantage='outcome-only GRPO; no ARM label requests (v2)',
            selector='SelectionARM 81b452d via sglang (scripts/serve_arm_sglang.py), last GPU, mem fraction .25',
            group_selection='48 native mixed groups + rescued all-failure groups (guided success replaces one failure)',
            initialization='Original SFT iteration0', gpu_validated=False)
        write_json(SOURCE/'reference_manifest.json', manifest)
    validate_source(SOURCE)
    old, new = read(PARENT/'reference_manifest.json')['recipe_files_sha256'], read(SOURCE/'reference_manifest.json')['recipe_files_sha256']
    if any(new.get(k) != v for k, v in old.items() if k not in changes) or any((SOURCE/k).read_bytes() != v for k, v in changes.items()):
        raise ValueError('Unexpected frozen source change')


def plan(job, stage):
    import run_arm_turn_bonus_cycles as training
    p = training.plan(job, 'arm', int(STAGES[stage]['hours']*60))
    old_source, old_output = p['source'], p['output']
    attempt = os.getenv('ARM_RESCUE_ATTEMPT', '')
    suffix = f'-r{int(attempt)}' if attempt else ''
    output = RUNTIME/f'evaluations/arm-rescue-{stage}-{job}{suffix}'
    run = f'arm-rescue-{stage}-{job}{suffix}'
    rewrite = lambda v: v.replace(old_source, str(SOURCE)).replace(old_output, str(output))
    p['command'] = [rewrite(v) for v in p['command']]
    for key in ('environment', 'arm_config'):
        p[key] = {k: rewrite(v) if isinstance(v, str) else v for k, v in p[key].items()}
    n = STAGES[stage]['iterations']
    p.update(source=str(SOURCE), output=str(output), wandb_run_id=run, gate_ablation='rescue', rescue_stage=stage,
             requested_resources=dict(gpus=8, gpu_type='H200', hours=STAGES[stage]['hours'], cpus=64, memory_gib=960,
                                      browsers=64),
             requested_iterations=n, target_completed_iterations=n, fresh_optimizer=True,
             initial_optimizer_updates=0, start_rollout_id=0, evaluation_in_allocation=stage == 'main')
    scratch = RUNTIME/f'multimodal-scratch/arm-variant-{job}'
    project = 'openwebrl' if stage == 'main' else 'openwebrl-evals'
    p['environment'].update(NUM_GPUS='8', TP_SIZE='2', NUM_ROLLOUT=str(n), WANDB_PROJECT=project,
        WANDB_RUN_ID=run, WANDB_RESUME='never', WANDB_NAME=f'ARM rescue | outcome-only | {stage} | {job}',
        BROWSER_CONCURRENCY='64', SGLANG_CONCURRENCY='64', OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(scratch),
        OPENWEBRL_CUDA_CACHE_LIMIT_GIB='48', PYTHONFAULTHANDLER='1', RAY_DEDUP_LOGS='0', RAY_TMPDIR=f'/tmp/ar-{job}',
        OPENWEBRL_ARM_RESCUE_CONFIG=str(CONTROL/f'rescue-config-{run}.json'))
    p['multimodal_storage'] = dict(mode='shared', directory=str(scratch), minimum_free_bytes=2*1024**4,
                                   retention='preserve; no automatic deletion')
    p['browser_config']['browser_rollout_concurrency'] = 64
    p['command'][p['command'].index('--wandb-project')+1] = project
    p['command'][p['command'].index('--wandb-group')+1] = f'ARM rescue | outcome-only | from0 | {stage}'
    p['command'][p['command'].index('--custom-generate-function-path')+1] = GENERATE
    p['command'] += ['--dynamic-sampling-filter-path', FILTER]
    p['arm_config'].update(run_id=run, policy_id=run+':uninitialized', run_output=str(output), output=str(output),
        candidate_gate='min2', credit_assignment='response_index', gate_scale_guard='pretraining_coverage_audit_v1',
        admit_all_failure_groups=False, additive_failure_groups=False, failure_group_cap=0,
        failure_loss_coefficient=0., minimum_cycle_seconds=2700, seconds_per_optimizer_update=90)
    p['rescue_config'] = dict(RESCUE, output=str(output), selector_endpoint=p['arm_config']['selector_endpoint'],
                              policy_id=run, seed=42)
    if stage == 'smoke':
        p['rescue_config']['wait_seconds'] = 600  # fit one iteration inside the 1.5 h smoke allocation
    if stage == 'paired':
        p['rescue_config'].update(PAIRED)
    p['comparison'] = dict(controls='outcome-only 48 groups (historical), outcome-only 56 groups, uniform G8',
        changed_knob='one ARM-guided attempt per valid all-failure group; successes join GRPO as mixed groups',
        initialization='Original SFT iteration0, fresh optimizer/scheduler/cursor')
    validate_recipe(p)
    return p


def validate_recipe(p):
    e = p['environment']
    if (not p['fresh_optimizer'] or p['start_rollout_id'] != 0 or p['initial_optimizer_updates'] != 0
            or p['checkpoint'] != str(INITIAL) or e['SLIME_LOAD_CHECKPOINT'] != str(INITIAL)
            or e['HF_CHECKPOINT'] != str(INITIAL) or e['WANDB_RESUME'] != 'never'
            or '--use-checkpoint-opt-param-scheduler' in p['command']):
        raise ValueError('A new intervention must start at iteration0')
    c = p['command']
    if (c[c.index('--dynamic-sampling-filter-path')+1] != FILTER or c[c.index('--custom-generate-function-path')+1] != GENERATE
            or 'OPENWEBRL_ARM_FAILURE_AUX_MANIFEST' in e or any(k.startswith('OPENWEBRL_REPLAY') for k in e)
            or (e['NUM_GPUS'], e['TP_SIZE'], e['GLOBAL_BATCH_SIZE']) != ('8', '2', '256')):
        raise ValueError('Rescue hooks, auxiliary, replay or topology mismatch')
    a = p['arm_config']
    if a['beta'] != .5 or a['scored_fraction'] != .2 or a.get('additive_failure_groups') or a.get('admit_all_failure_groups'):
        raise ValueError('Label telemetry settings changed')


def fingerprints():
    return {str(x): sha(x) for x in (Path(__file__).resolve(), SOURCE/'reference_manifest.json',
            CONTROL/'training-controller.py', REPO/'tests/test_arm_rescue_rl.py')}


def validate_plan(p):
    validate_source(SOURCE); validate_recipe(p)
    if p != plan(p['job_id'], p['rescue_stage']):
        raise ValueError('Prepared plan changed')
    receipt = read(CONTROL/'readiness.json')
    if not receipt['cpu_passed'] or not receipt['native_parse_passed'] or receipt['fingerprints'] != fingerprints():
        raise ValueError('Prepared readiness missing or stale')
    return dict(passed=True, cpu_passed=True, gpu_initialization_pending=True, auxiliary_disabled=True)


def native_check(p, destination):
    from resume_arm_failure_variants import check_multimodal_storage
    write_json(destination/'storage-preflight.json', check_multimodal_storage(p))
    write_json(destination/'browser.json', p['browser_config'])
    env = dict(clean_environment(), **p['environment'])
    env.update(DRY_RUN='1', CUDA_VISIBLE_DEVICES='', BROWSER_TRAIN_CONFIG=str(destination/'browser.json'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    argv = shlex.split(subprocess.check_output(p['command'], env=env, text=True)); env.pop('DRY_RUN')
    expected = dict(actor_num_gpus_per_node=8, tensor_model_parallel_size=2, global_batch_size=256,
        micro_batch_size=1, ppo_epochs=2, rollout_batch_size=48, browser_rollout_concurrency=64,
        judge_api_model='gpt-4.1', judge_prompt_variant='action_history', dynamic_sampling_filter_path=FILTER,
        custom_generate_function_path=GENERATE, use_rollout_logprobs=True, lr=1e-6,
        num_rollout=int(p['environment']['NUM_ROLLOUT']), start_rollout_id=0, use_checkpoint_opt_param_scheduler=False)
    code = ("from unittest.mock import patch\nfrom slime.utils.arguments import parse_args\n"
        "with patch('megatron.training.arguments.get_device_arch_version',return_value=9): a=parse_args()\n"
        f"expected={expected!r}\n"
        "assert all(getattr(a,k)==v for k,v in expected.items()),{k:getattr(a,k) for k in expected}\n"
        "import openwebrl.arm_rescue_rl, slime.rollout.sglang_rollout\n"
        "from openwebrl.arm_failure_aux import enabled\nassert not enabled()\nprint('RESCUE_NATIVE_PARSE_OK')\n")
    with (destination/'native-parse.log').open('w') as stream:
        subprocess.run(source_command(SOURCE, [str(PYTHON), '-c', code, *argv[2:]]), cwd=SOURCE, env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=180)


def prepare():
    prepare_source(); CONTROL.mkdir(parents=True, exist_ok=True)
    from arm_browser_startup_guard import instrument_controller
    text = (REPO/'scripts/run_arm_turn_bonus_cycles.py').read_text()
    before = 'from run_arm_gate_ablation import validate_plan'
    if text.count(before) != 1:
        raise ValueError('Controller validation boundary changed')
    text = patched_controller(instrument_controller(text.replace(before, 'from prepare_arm_rescue import validate_plan', 1)))
    path = CONTROL/'training-controller.py'
    if path.exists() and path.read_text() != text:
        raise ValueError('Frozen controller changed')
    path.write_text(text); compile(text, str(path), 'exec')
    for stage in STAGES:
        out = CONTROL/stage; out.mkdir(exist_ok=True)
        p = plan('PREPARE-'+stage, stage)
        write_json(out/'plan.json', p); native_check(p, out)
    test = REPO/'tests/test_arm_rescue_rl.py'
    code = ("import importlib.util,sys,unittest;import openwebrl.arm_rescue_rl as m;"
            f"assert m.__file__.startswith({str(SOURCE)!r}),m.__file__;"
            f"spec=importlib.util.spec_from_file_location('test_arm_rescue_rl',{str(test)!r});"
            "t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);"
            "s=unittest.defaultTestLoader.loadTestsFromModule(t);"
            "r=unittest.TextTestRunner(verbosity=2).run(s);raise SystemExit(not r.wasSuccessful())")
    with (CONTROL/'frozen-tests.log').open('w') as stream:
        # Run the repository tests against the frozen source module (PYTHONPATH puts SOURCE first).
        subprocess.run(source_command(SOURCE, [str(PYTHON), '-c', code]), cwd=SOURCE,
            env=dict(clean_environment(), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled'),
            stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=300)
    write_json(CONTROL/'readiness.json', dict(cpu_passed=True, native_parse_passed=True, storage_preflight_passed=True,
                                              gpu_validated=False, fingerprints=fingerprints()))
    for stage in STAGES:
        validate_plan(plan('PREPARE-'+stage, stage))
    return CONTROL


def worker(job, stage):
    p = plan(job, stage); validate_plan(p)
    from resume_arm_failure_variants import check_multimodal_storage
    check_multimodal_storage(p)
    # The controller creates the output directory itself, so the rescue config lives in CONTROL.
    write_json(Path(p['environment']['OPENWEBRL_ARM_RESCUE_CONFIG']), p['rescue_config'])
    # GPU telemetry every 30 s: the first smoke saw a 3x slower ARM selector with no visible cause.
    log = CONTROL/f"gpu-telemetry-{p['wandb_run_id']}.csv"
    gpu = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,'
                            'clocks.sm,temperature.gpu,clocks_throttle_reasons.active', '--format=csv', '-l', '30'],
                           stdout=log.open('w'), stderr=subprocess.STDOUT)
    try:
        spec = importlib.util.spec_from_file_location('rescue_controller', CONTROL/'training-controller.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.execute(p)
    finally:
        gpu.terminate()


def hold(job, root, step, reason, idle_used):
    """Keep the allocation after a failure for in-allocation repair.

    CONTROL/<job>-hold-request.json: {"action": "run", "argv": [...], "env": {...}, "final": bool} runs argv as a
    full 8-GPU srun step; {"action": "release"} ends the job."""
    request = CONTROL/f'{job}-hold-request.json'
    log = root/'hold.jsonl'
    while True:
        limit = min(HOLD_IDLE_SECONDS, HOLD_TOTAL_IDLE_SECONDS-idle_used)
        write_json(root/'status.json', dict(stage='hold', complete=False, reason=str(reason)[:400],
            idle_limit_seconds=int(limit), idle_used_seconds=int(idle_used), request_path=str(request)))
        start = time.time(); r = None
        while time.time()-start < limit:
            if request.is_file():
                r = read(request); request.rename(request.with_name(f'{request.name}.{int(time.time())}.consumed')); break
            time.sleep(15)
        idle_used += time.time()-start
        if r is None or r.get('action') == 'release':
            with log.open('a') as stream:
                stream.write(json.dumps(dict(time=time.time(), action='release' if r else 'idle-cap'))+'\n')
            return
        if r.get('action') != 'run' or not isinstance(r.get('argv'), list):
            reason = f'Ignored malformed hold request {r!r}'[:400]; continue
        write_json(root/'status.json', dict(stage='hold-run', complete=False, argv=r['argv']))
        try:
            code = subprocess.run([*step[:step.index('--cpu-bind=none')+1], *map(str, r['argv'])],
                                  env={**os.environ, **{k: str(v) for k, v in r.get('env', {}).items()}}).returncode
        except Exception as exc:
            code = f'{type(exc).__name__}: {exc}'[:400]
        with log.open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(), argv=r['argv'], exit_code=code, final=bool(r.get('final'))))+'\n')
        if code == 0 and r.get('final'):
            write_json(root/'status.json', dict(stage='complete-after-hold', complete=True)); return
        reason = f"Hold step exited {code}: {r['argv']}"


def smoke_check(p):
    """Smoke pass criteria: one optimizer iteration saved and a rescue report for rollout 0."""
    out = Path(p['output'])
    report = read(out/'rescue-reports/0000.json')
    marker = out/'runtime/latest_checkpointed_iteration.txt'
    saved = marker.is_file() and int(marker.read_text()) == 0
    result = dict(rescue_attempted=report['attempted'], rescued=report['rescued'], checkpoint_saved=saved,
                  rescue_errors=sum(1 for r in report['records'] if r.get('error_type')),
                  passed=saved and report['attempted'] > 0)
    write_json(CONTROL/f"smoke-result-{p['job_id']}.json", result)
    return result


def paired_check(p):
    """Guided vs control success over every completed rescue report (diagnostic; no pass threshold on yield)."""
    out, arms, groups = Path(p['output']), {}, 0
    for path in sorted((out/'rescue-reports').glob('*.json')):
        report = read(path)
        groups += report['attempted']
        for arm, v in report.get('arms', {}).items():
            a = arms.setdefault(arm, dict(attempts=0, valid=0, successes=0, groups_with_success=0))
            for k in a:
                a[k] += v[k]
    marker = out/'runtime/latest_checkpointed_iteration.txt'
    result = dict(groups=groups, arms=arms, rescue_reports=len(list((out/'rescue-reports').glob('*.json'))),
                  latest_checkpointed_iteration=int(marker.read_text()) if marker.is_file() else None)
    result['passed'] = result['rescue_reports'] > 0 and all(arms.get(k, {}).get('attempts') for k in ('guided', 'control'))
    write_json(CONTROL/f"paired-result-{p['job_id']}.json", result)
    return result


def execute(job, stage):
    if os.getenv('SLURM_JOB_ID') != job or os.getenv('ARM_RESCUE_APPROVED') != stage:
        raise ValueError('Exact resource approval for this stage required')
    allocation(subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True), job, requested_gpus=8,
               maximum_hours=STAGES[stage]['hours'])
    p = plan(job, stage); validate_plan(p)
    root = RUNTIME/f'evaluations/arm-rescue-controller-{job}'; root.mkdir(exist_ok=False)
    write_json(root/'controller-plan.json', dict(job_id=job, stage=stage, training_root=p['output'],
                                                 target=STAGES[stage]['iterations']))
    step = ['srun', f'--jobid={job}', '--nodes=1', '--ntasks=1', '--cpus-per-task=64', '--gres=gpu:h200:8',
            '--exact', '--cpu-bind=none', str(PYTHON), str(Path(__file__).resolve()), '--job-id', job,
            '--stage', stage, '--worker']
    try:
        write_json(root/'status.json', dict(stage='training', complete=False))
        result = subprocess.run(step)
        if stage in ('smoke', 'paired'):
            check = smoke_check(p) if stage == 'smoke' else paired_check(p)
            write_json(root/'status.json', dict(stage='complete' if check['passed'] else f'{stage}-failed',
                                                complete=check['passed'], training_exit_code=result.returncode, **check))
        if result.returncode:
            raise RuntimeError(f'Training exited {result.returncode}; preserve and inspect artifacts')
    except Exception as exc:
        write_json(root/'failure.json', dict(error_type=type(exc).__name__, error=str(exc)[:400], time=time.time()))
        hold(job, root, step, f'{type(exc).__name__}: {exc}', 0.)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare', action='store_true'); ap.add_argument('--job-id', default='PREPARE')
    ap.add_argument('--stage', choices=STAGES); ap.add_argument('--execute', action='store_true')
    ap.add_argument('--worker', action='store_true')
    a = ap.parse_args()
    if a.prepare:
        print(prepare())
    elif a.worker:
        if os.getenv('SLURM_JOB_ID') != a.job_id:
            raise ValueError('Wrong worker allocation')
        worker(a.job_id, a.stage)
    elif a.execute:
        execute(a.job_id, a.stage)
    else:
        print(json.dumps(plan(a.job_id, a.stage), indent=2))
