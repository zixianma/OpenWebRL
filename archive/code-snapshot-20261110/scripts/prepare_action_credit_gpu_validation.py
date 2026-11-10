#!/usr/bin/env python3
"""Freeze and execute an isolated, allocation-bounded native mechanics gate.

Preparation never submits work. Execution requires a registered Slurm attempt;
the batch controller owns and awaits both topology cases. No actor collection,
judge HTTP request, production update, or historical-file cleanup is performed.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

REPO = Path('/gpfs/projects/krishna/zixianma/OpenWebRL')
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
MODEL = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
PARENT = RUNTIME/'reference-uniform8-original2102-20261006-v1'
PYTHON = RUNTIME/'venv/bin/python'
FORBIDDEN = Path('/gpfs/home/zixianma')
CAP_SECONDS = 21600
CAP_GPU_SECONDS = 172800


def resource_profile(gpus=8, tp_sizes=None, allocated_seconds=CAP_SECONDS):
    """Describe a bounded diagnostic allocation, never a production approval."""
    if type(gpus) is not int or gpus not in (2, 8):
        raise ValueError('Native validation supports exactly 2 or 8 GPUs')
    if (type(allocated_seconds) is not int or not 600 < allocated_seconds <= CAP_SECONDS
            or allocated_seconds % 60):
        raise ValueError('Allocation seconds must be whole minutes within the existing six-hour cap')
    tp_sizes = tuple((2, 8) if gpus == 8 else (2,)) if tp_sizes is None else tuple(tp_sizes)
    if (not tp_sizes or len(set(tp_sizes)) != len(tp_sizes)
            or any(type(tp) is not int or tp < 1 or gpus % tp for tp in tp_sizes)):
        raise ValueError('Distinct tensor-parallel sizes must divide the allocated GPU count')
    return dict(gpus=gpus, cpus=8*gpus, memory_gib=120*gpus,
                allocated_seconds=allocated_seconds, tp_sizes=list(tp_sizes))


def campaign_admission(ledger, current_job, live):
    """Charge every terminal attempt and reserve every outstanding allocation.

    The caller supplies a fresh all-attempt sacct snapshot. Outstanding jobs
    reserve their entire time limit, so two simultaneous workers cannot each
    treat the same unused balance as available. No shared ledger rewrite is
    necessary at admission: a frozen audit records the reconciled charges.
    """
    if (ledger.get('total_allocation_seconds_cap') != CAP_SECONDS
            or ledger.get('total_gpu_seconds_cap') != CAP_GPU_SECONDS
            or ledger.get('max_concurrent_gpus') != 8):
        raise ValueError('Shared native campaign caps changed')
    active = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED',
              'RESIZING', 'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD', 'SIGNALING', 'STAGE_OUT'}
    terminal = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY',
                'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'SPECIAL_EXIT'}
    rows, seen = [], set()
    for attempt in ledger.get('attempts', []):
        job = str(attempt.get('job_id', ''))
        if not re.fullmatch(r'\d+', job) or job in seen or job not in live:
            raise ValueError('Every unique registered attempt requires live scheduler accounting')
        seen.add(job)
        gpus, limit, recorded = (attempt.get('gpus'), attempt.get('allocated_seconds'), attempt.get('elapsed_seconds'))
        if (type(gpus) is not int or gpus not in (2, 8) or type(limit) is not int
                or not 0 < limit <= CAP_SECONDS or type(recorded) is not int or recorded < 0):
            raise ValueError('Invalid registered attempt resources or consumed time')
        observed = live[job]
        state = observed.get('state', '').split()[0].rstrip('+')
        elapsed = observed.get('elapsed_seconds')
        if state not in active | terminal or type(elapsed) is not int or elapsed < 0:
            raise ValueError('Unknown live attempt state or elapsed time')
        if observed.get('gpus') != gpus or observed.get('allocated_seconds') != limit:
            raise ValueError('Live attempt resources exceed or differ from registration')
        charged = max(recorded, elapsed)
        reserved = max(charged, limit) if state in active else charged
        rows.append(dict(job_id=job, gpus=gpus, state=state, elapsed_seconds=charged,
                         reserved_seconds=reserved, outstanding=state in active))
    if str(current_job) not in seen or live[str(current_job)]['state'].split()[0] != 'RUNNING':
        raise ValueError('Current running allocation must be registered in the shared campaign')
    allocation_seconds = sum(row['reserved_seconds'] for row in rows)
    gpu_seconds = sum(row['reserved_seconds']*row['gpus'] for row in rows)
    reserved_gpus = sum(row['gpus'] for row in rows if row['outstanding'])
    if allocation_seconds > CAP_SECONDS or gpu_seconds > CAP_GPU_SECONDS or reserved_gpus > 8:
        raise ValueError('Shared native allocation, GPU-time or concurrency cap exceeded')
    return dict(attempts=rows, allocation_seconds_reserved=allocation_seconds,
                gpu_seconds_reserved=gpu_seconds, outstanding_gpus=reserved_gpus,
                observed_epoch=time.time(), current_job_id=str(current_job))


def scheduler_attempts(job_ids):
    output = subprocess.check_output(['sacct', '-X', '-j', ','.join(job_ids), '--noheader', '--parsable2',
        '--format=JobIDRaw,State,ElapsedRaw,AllocTRES,ReqTRES,TimelimitRaw'], text=True)
    result = {}
    for line in output.splitlines():
        fields = line.split('|')
        if len(fields) < 6 or fields[0] not in job_ids or fields[0] in result:
            raise ValueError('Unexpected or duplicate scheduler accounting row')
        job, state, elapsed, allocated, requested, minutes = fields[:6]
        tres = dict(part.split('=', 1) for part in (allocated or requested).split(',') if '=' in part)
        result[job] = dict(state=state, elapsed_seconds=int(elapsed),
                           gpus=int(tres.get('gres/gpu', tres.get('gres/gpu:h200', '0'))),
                           allocated_seconds=int(minutes)*60)
    return result


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def request_core_sha(value):
    core = {key: item for key, item in value.items() if key != 'prepared_execution_sha256'}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def writable(path):
    lexical = Path(path).absolute()
    p = lexical.resolve()
    if (lexical.is_relative_to(FORBIDDEN) or p.is_relative_to(FORBIDDEN)
            or not p.is_relative_to(RUNTIME)):
        raise ValueError('Validation outputs must remain inside scrubbed runtime')
    return p


def save(path, value):
    path = writable(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def safe_environment(control):
    """Remove inherited experiment/credential settings; redirect writable caches."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('OPENWEBRL_', 'SLIME_', 'WANDB_', 'ARM_', 'RAY_'))
           and not any(x in k.upper() for x in ('API_KEY', 'ACCESS_TOKEN', 'AUTH_TOKEN'))
           and k not in ('PYTHONPATH', 'PYTHONHOME', 'DRY_RUN', 'SLIME_LOAD_CHECKPOINT',
                        'OVERRIDE_OPT_PARAM_SCHEDULER')}
    cache = writable(control/'cache')
    paths = {
        'HF_HOME': cache/'hf', 'HUGGINGFACE_HUB_CACHE': cache/'hf/hub',
        'TRANSFORMERS_CACHE': cache/'hf/transformers', 'TORCH_HOME': cache/'torch',
        'XDG_CACHE_HOME': cache/'xdg', 'XDG_CONFIG_HOME': cache/'config',
        'XDG_DATA_HOME': cache/'data', 'PIP_CACHE_DIR': cache/'pip',
        'UV_CACHE_DIR': cache/'uv', 'WANDB_CACHE_DIR': cache/'wandb',
        'WANDB_CONFIG_DIR': cache/'wandb-config', 'WANDB_DIR': cache/'wandb-runs',
        'MPLCONFIGDIR': cache/'matplotlib', 'NUMBA_CACHE_DIR': cache/'numba',
        'CUDA_CACHE_PATH': cache/'cuda', 'TMPDIR': control/'tmp',
        'PYTHONPYCACHEPREFIX': cache/'pycache',
        'TORCH_EXTENSIONS_DIR': cache/'torch-extensions',
        'TRITON_CACHE_DIR': cache/'triton', 'TORCHINDUCTOR_CACHE_DIR': cache/'inductor',
        'FLASHINFER_WORKSPACE_BASE': cache/'flashinfer',
    }
    for key, path in paths.items():
        path = writable(path); path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    # Ray uses Unix sockets; a full campaign path exceeds their 108-byte limit.
    ray_tmp = Path('/tmp')/f'acg-{os.getuid()}-{os.environ.get("SLURM_JOB_ID", "prep")}'
    if ray_tmp.is_symlink() or ray_tmp.resolve().parent != Path('/tmp'):
        raise ValueError('Unsafe Ray temporary directory')
    ray_tmp.mkdir(exist_ok=True)
    env['RAY_TMPDIR'] = str(ray_tmp)
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled', WANDB_PROJECT='openwebrl-evals',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', RAY_ADDRESS='local',
               OPENWEBRL_RUNTIME_ROOT=str(RUNTIME), OPENWEBRL_STREAMING_CHECKPOINT='1',
               OPENWEBRL_CUDA_CACHE_LIMIT_GIB='48', SLIME_ASYNC_USE_STDLIB_LOOP='1')
    return env


def replace_once(value, before, after):
    if value.count(before) != 1:
        raise ValueError('Frozen source anchor changed: '+before[:80])
    return value.replace(before, after, 1)


def configure_native_command(command, tp):
    command = list(command)
    command.remove('--colocate')
    command.remove('--use-rollout-logprobs')
    if tp == 1:
        command.remove('--sequence-parallel')
    command[command.index('--ppo-epochs')+1] = '1'
    command.extend(['--clip-grad', '1.0', '--data-parallel-random-init'])
    for flag in ('--offload-train', '--offload-rollout', '--overlap-param-gather',
                 '--async-save', '--no-check-for-nan-in-loss-and-grad', '--use-wandb'):
        if flag in command:
            raise ValueError('Unsupported native acceptance flag: '+flag)
    return command


def mechanics_payloads(mechanics_source, *, native_gate_source=None):
    """Read a complete local package snapshot without borrowing missing modules.

    Mechanics now import slot selection, evaluation accounting and tensor
    storage indirectly. Copying eleven files mixed those dependencies with an
    older package. Pin all Python modules from the chosen snapshot plus its
    browser configuration/prompt resources, without executing browser code.
    """
    root = Path(mechanics_source).resolve()
    package = root/'openwebrl'
    required = ('action_credit.py', 'action_credit_advantages.py',
        'action_credit_binding.py', 'action_credit_objective.py', 'action_credit_receipts.py',
        'action_credit_sampling.py', 'action_credit_selection.py', 'action_credit_state.py',
        'action_credit_megatron.py', 'action_credit_windows.py', 'action_credit_native_gate.py')
    paths = set(package.rglob('*.py'))
    paths.update(package/name for name in (
        'browser_training_config.yaml', 'online_mind2web_monitor.yaml', 'env/config.yaml')
        if (package/name).is_file())
    for pattern in ('*.json', '*.md'):
        paths.update((package/'env/prompts').glob(pattern))
    dependency_inventory = root/'action-credit-runtime-dependencies.json'
    if dependency_inventory.is_file():
        paths.add(dependency_inventory)
    payloads = {}
    for path in sorted(paths):
        if not path.resolve().is_relative_to(root):
            raise ValueError('Mechanics dependency escapes the selected source: '+str(path))
        if '__pycache__' not in path.parts and path.is_file():
            payloads[str(path.relative_to(root))] = path.read_bytes()
    for name in required:
        if 'openwebrl/'+name not in payloads:
            raise ValueError('Missing native mechanics source: '+name)
    original_gate = hashlib.sha256(payloads['openwebrl/action_credit_native_gate.py']).hexdigest()
    override = None
    if native_gate_source is not None:
        path = Path(native_gate_source).resolve()
        if path.name != 'action_credit_native_gate.py' or not path.is_file():
            raise ValueError('Explicit native gate override must be a Python gate source file')
        payloads['openwebrl/action_credit_native_gate.py'] = path.read_bytes()
        override = dict(path=str(path), sha256=sha(path), original_sha256=original_gate)
    # Lazy imports matter: a top-level smoke import misses slot selection.
    for name, content in payloads.items():
        if not name.endswith('.py'):
            continue
        tree = ast.parse(content, filename=name)
        package_parts = name[:-3].split('/')[:-1]
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name.split('.') for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package_parts[:len(package_parts)-node.level+1] if node.level else []
                modules = [prefix+(node.module.split('.') if node.module else [])]
                if modules[0] == ['openwebrl']:
                    modules.extend(['openwebrl', alias.name] for alias in node.names
                                   if alias.name.startswith('action_credit'))
            for module in modules:
                if not module or module[0] != 'openwebrl':
                    continue
                relative = '/'.join(module)
                if (relative+'.py' not in payloads and relative+'/__init__.py' not in payloads
                        and not any(p.startswith(relative+'/') for p in payloads)):
                    raise ValueError('Missing local mechanics dependency '+'.'.join(module)+' imported by '+name)
    pin = dict(source=str(root), files_sha256={name:hashlib.sha256(content).hexdigest()
               for name, content in sorted(payloads.items())}, native_gate_override=override)
    return payloads, pin


def prepare(control, fixture, *, gpus=8, tp_sizes=None, allocated_seconds=CAP_SECONDS,
            campaign_ledger=None, mechanics_source=None, native_gate_source=None):
    from resume_baseline import validate_source
    from prepare_arm_turn_bonus import copy_plain
    control = writable(control); fixture = Path(fixture).resolve()
    profile = resource_profile(gpus, tp_sizes, allocated_seconds)
    campaign_ledger = writable(campaign_ledger) if campaign_ledger is not None else None
    if gpus != 8 and campaign_ledger is None:
        raise ValueError('Split allocations require one shared campaign ledger')
    if control.exists():
        raise ValueError('Use a new immutable validation preparation directory')
    if not fixture.is_file() or fixture.stat().st_size > 2*1024**3:
        raise ValueError('A bounded extracted real-browser fixture is required')
    validate_source(PARENT)
    mechanics_source = REPO if mechanics_source is None else Path(mechanics_source).resolve()
    source = control/'source'
    changes, mechanics_pin = mechanics_payloads(mechanics_source, native_gate_source=native_gate_source)
    actor_name = 'slime/backends/megatron_utils/actor.py'
    actor = (PARENT/actor_name).read_text()
    actor = replace_once(actor,
        '    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:\n',
        '    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:\n'
        '        if os.environ.get("OPENWEBRL_ACTION_CREDIT_NATIVE_GATE"):\n'
        '            from openwebrl.action_credit_native_gate import run_native_gate\n'
        '            return run_native_gate(self)\n')
    changes[actor_name] = actor.encode()
    driver = (PARENT/'train.py').read_text()
    driver = replace_once(driver,
        '    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)\n',
        '    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)\n'
        '    if os.environ.get("OPENWEBRL_ACTION_CREDIT_NATIVE_GATE"):\n'
        '        assert args.debug_train_only and not args.offload_train and not args.offload_rollout\n'
        '        assert critic_model is None and args.start_rollout_id == 0\n'
        '        ray.get(actor_model.async_train(0, None))\n'
        '        ray.get(rollout_manager.dispose.remote())\n'
        '        actor_model.finish_tracking()\n'
        '        finish_tracking(args)\n'
        '        return\n')
    changes['train.py'] = driver.encode()
    shell = (PARENT/'scripts/h200_env.sh').read_text()
    shell = replace_once(shell,
        'export TORCHINDUCTOR_CACHE_DIR="$OPENWEBRL_RUNTIME_ROOT/torchinductor-cache"',
        'export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$OPENWEBRL_RUNTIME_ROOT/torchinductor-cache}"')
    shell = replace_once(shell,
        'export TRITON_CACHE_DIR="$OPENWEBRL_RUNTIME_ROOT/triton-cache"',
        'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$OPENWEBRL_RUNTIME_ROOT/triton-cache}"')
    changes['scripts/h200_env.sh'] = shell.encode()
    for name, content in changes.items():
        if name.endswith('.py'):
            compile(content, name, 'exec')
    control.mkdir(parents=True)
    shutil.copytree(PARENT, source, symlinks=False, copy_function=copy_plain,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.env'))
    manifest = json.loads((PARENT/'reference_manifest.json').read_text())
    for name, content in changes.items():
        (source/name).parent.mkdir(parents=True, exist_ok=True)
        (source/name).write_bytes(content)
        manifest['recipe_files_sha256'][name] = hashlib.sha256(content).hexdigest()
    manifest['action_credit_native_gate'] = {'production': False, 'parent': str(PARENT),
        'changes': sorted(changes), 'zero_browser_and_judge_calls': True,
        'mechanics_snapshot': mechanics_pin}
    save(source/'reference_manifest.json', manifest)
    validate_source(source)
    shutil.copyfile(fixture, control/'fixture.pt')
    copy_plain(Path(__file__).resolve(), control/'controller.py')
    env = safe_environment(control)
    for tp in profile['tp_sizes']:
        case = control/f'tp{tp}-dp{gpus//tp}'; case.mkdir()
        gate_manifest = dict(schema_version=1, diagnostic_only=True,
            fixture_path=str(control/'fixture.pt'), fixture_sha256=sha(control/'fixture.pt'),
            output_dir=str(case), hf_checkpoint=str(MODEL), hf_config_sha256=sha(MODEL/'config.json'),
            source_manifest_sha256=sha(source/'reference_manifest.json'), stress_length=32768,
            max_seconds=min(9000, allocated_seconds-180), gradient_atol=1e-5, gradient_rtol=0.03,
            logprob_atol=2e-4, logprob_rtol=2e-5)
        save(case/'gate-manifest.json', gate_manifest)
        case_env = dict(env, NUM_GPUS=str(gpus), TP_SIZE=str(tp), NUM_ROLLOUT='1',
            HF_CHECKPOINT=str(MODEL), SAVE_DIR=str(case/'checkpoint'), SAVE_INTERVAL='1',
            TRAIN_DATA=str(REPO/'openwebrl/data/webgym_filtered_popular_2102_cleaned.parquet'),
            BROWSER_TRAIN_CONFIG=str(source/'openwebrl/browser_training_config.yaml'),
            BROWSER_MAX_STEPS='15', ROLLOUT_BATCH_SIZE='48', N_SAMPLES='8',
            GLOBAL_BATCH_SIZE='256', CONTEXT_LEN='32768', RESPONSE_LEN='1024',
            BROWSER_CONCURRENCY='1', SGLANG_CONCURRENCY='1', LEARNING_RATE='1e-6',
            RECOMPUTE_ACTIVATIONS='1', JUDGE_API_MODE='served',
            OPENWEBRL_ACTION_CREDIT_NATIVE_GATE=str(case/'gate-manifest.json'),
            OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(case/'multimodal'), DRY_RUN='1')
        command = ['bash', str(source/'scripts/run_h200_browser.sh'), '--skip-eval-before-train',
            '--seed', '42', '--rollout-top-p', '1', '--rollout-top-k', '-1',
            '--debug-train-only', '--loss-type', 'custom_loss',
            '--custom-loss-function-path', 'openwebrl.action_credit_megatron.loss']
        native = shlex.split(subprocess.check_output(command, env=case_env, cwd=source, text=True))
        native = [str(PYTHON), '-B', str(source/'train.py'), *native[2:]]
        native = configure_native_command(native, tp)
        case_env.pop('DRY_RUN')
        # Store only selected launch configuration, never inherited credentials.
        changed_env = {k: v for k, v in case_env.items() if env.get(k) != v}
        save(case/'plan.json', dict(command=native, environment=changed_env, tp=tp, dp=gpus//tp))
    resource = dict(schema_version=1, diagnostic_only=True,
        authorization_text='I approve any gpu you need to verify these baselines and launch them',
        purpose='Native real-model mechanics gates before fresh iteration0-to10 pilots',
        selected_operational_cap=dict(gpus=gpus, gpu_type='H200', cpus=profile['cpus'], memory_gib=profile['memory_gib'],
            allocated_seconds=allocated_seconds, total_allocation_seconds_all_attempts=CAP_SECONDS, gpu_hours=48),
        topology_cases=[dict(tp=tp, dp=gpus//tp) for tp in profile['tp_sizes']],
        campaign_ledger=str(campaign_ledger) if campaign_ledger is not None else None,
        actor_collections=0, judge_http_requests=0, api_dollars=0,
        independent_final_artifact_review_required=True,
        fixture_sha256=sha(control/'fixture.pt'), source_manifest_sha256=sha(source/'reference_manifest.json'),
        production_launch_ready=False)
    save(control/'request.json', resource)
    hours, remainder = divmod(allocated_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    batch = f'''#!/bin/bash
#SBATCH --job-name=owrl-credit-gate
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:{gpus}
#SBATCH --cpus-per-task={profile['cpus']}
#SBATCH --mem={profile['memory_gib']}G
#SBATCH --time={hours:02d}:{minutes:02d}:{seconds:02d}
#SBATCH --no-requeue
#SBATCH --output={control}/slurm-%j.out
set -euo pipefail
cd {shlex.quote(str(control))}
exec /usr/bin/python3 -B {shlex.quote(str(control/'controller.py'))} --control {shlex.quote(str(control))} --worker
'''
    (control/'launch.sbatch').write_text(batch)
    subprocess.run(['bash', '-n', str(control/'launch.sbatch')], check=True)
    save(control/'preparation.json', dict(ready_for_native_parse=True, launched=False,
        files={str(p.relative_to(control)): sha(p) for p in control.rglob('*.json')},
        request_core_sha256=request_core_sha(resource),
        controller_sha256=sha(control/'controller.py'), batch_sha256=sha(control/'launch.sbatch')))
    resource['prepared_execution_sha256'] = sha(control/'preparation.json')
    save(control/'request.json', resource)
    print(json.dumps({'prepared': str(control), 'resources': resource['selected_operational_cap']}))


def validate_native_report(report, *, rank, gpus, tp, manifest_sha256, gate_sha256, backend_sha256):
    """Reject missing ranks, objectives, topology or physical acceptance evidence."""
    expected_arms = {'chain_outcome_grpo', 'tree_outcome_sampling', 'tree_grpo', 'continuation_q', 'gigpo'}
    cases = report.get('cases', [])
    if (report.get('passed') is not True or report.get('rank') != rank or report.get('tp') != tp
            or report.get('dp') != gpus//tp or report.get('production_enabled') is not False
            or report.get('browser_sampler_acceptance') is not False
            or report.get('manifest_sha256') != manifest_sha256
            or report.get('native_gate_source_sha256') != gate_sha256
            or report.get('backend_source_sha256') != backend_sha256
            or len(cases) != 5 or {case.get('arm_id') for case in cases} != expected_arms
            or report.get('stress', {}).get('context') != 32768
            or report.get('stress', {}).get('forward_backward') is not True
            or report.get('reload', {}).get('loaded_iteration') != 101
            or report.get('reload', {}).get('rng_restored') is not True):
        raise ValueError('Incomplete native rank, objective, topology or provenance acceptance')
    full = report.get('full_accumulation', {})
    if (full.get('global_responses') != 256 or full.get('local_microbatches') != 256//(gpus//tp)
            or full.get('optimizer_updated') is not True):
        raise ValueError('Missing native full256 accumulation acceptance')
    ppo = report.get('ppo_windows', {})
    if (ppo.get('optimizer_updates_in_memory') != 4 or ppo.get('epochs') != 2
            or ppo.get('base_occurrence_exposures') != 520 or ppo.get('virtual_padding_slots') != 504
            or ppo.get('additional_local_forwards') != 8
            or ppo.get('durable_checkpoint_committed') is not False or len(ppo.get('windows', [])) != 4):
        raise ValueError('Missing actual native PPO2/window/tail acceptance')
    for item in [*cases, *ppo['windows']]:
        parity = item.get('full_dp_gradient_parity', {})
        if parity.get('outside_tolerance') != 0 or parity.get('elements', 0) <= 0:
            raise ValueError('Native objective/window full-gradient parity failed')
    if any(case.get('native_optimizer_update') is not True or not case.get('tp_oracles') for case in cases):
        raise ValueError('Native objective update or tensor-parallel oracle missing')


def worker(control):
    import sys
    sys.path.insert(0, str(REPO/'scripts'))
    from resume_baseline import allocation, source_command, validate_source
    from check_storage_quota import check_storage_quota
    control = writable(control); source = control/'source'
    job = os.environ.get('SLURM_JOB_ID', '')
    if not re.fullmatch(r'\d+', job):
        raise ValueError('A registered owned allocation is required')
    request = json.loads((control/'request.json').read_text())
    preparation = json.loads((control/'preparation.json').read_text())
    if (sha(control/'preparation.json') != request['prepared_execution_sha256']
            or preparation.get('request_core_sha256') != request_core_sha(request)):
        raise ValueError('Frozen execution request or preparation changed')
    resources = request['selected_operational_cap']
    gpus = resources['gpus']
    cases = request.get('topology_cases', [dict(tp=tp, dp=8//tp) for tp in (2, 8)])
    profile = resource_profile(gpus, [case['tp'] for case in cases], resources.get('allocated_seconds', CAP_SECONDS))
    if any(case['dp'] != gpus//case['tp'] for case in cases):
        raise ValueError('Frozen topology cases do not match GPU allocation')
    attempts = json.loads((control/'attempts.json').read_text())
    current = [a for a in attempts['attempts'] if str(a['job_id']) == job]
    if len(current) != 1 or attempts['total_allocation_seconds_cap'] != CAP_SECONDS:
        raise ValueError('Exact new-campaign attempt registration is required')
    prior = sum(int(a['elapsed_seconds']) for a in attempts['attempts'] if str(a['job_id']) != job)
    if prior + current[0]['allocated_seconds'] > CAP_SECONDS:
        raise ValueError('Cumulative allocation cap exceeded; retries never reset the budget')
    info = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    alloc = allocation(info, job, requested_gpus=gpus, maximum_hours=profile['allocated_seconds']/3600)
    fields = dict(re.findall(r'(\w+)=(\S+)', info))
    tres = dict(part.split('=', 1) for part in fields['AllocTRES'].split(','))
    raw_limit = fields['TimeLimit']
    days, clock = raw_limit.split('-', 1) if '-' in raw_limit else ('0', raw_limit)
    pieces = [int(x) for x in clock.split(':')]
    if len(pieces) == 2:
        pieces.insert(0, 0)
    limit = int(days)*86400 + pieces[0]*3600 + pieces[1]*60 + pieces[2]
    if (alloc['cpus'] != profile['cpus'] or int(tres.get('gres/gpu', '0')) != gpus
            or alloc['allocated_memory_gib'] != profile['memory_gib']
            or current[0]['allocated_seconds'] != profile['allocated_seconds']
            or limit > current[0]['allocated_seconds']):
        raise ValueError('Actual scheduler resources differ from bounded registered request')
    ledger_path = request.get('campaign_ledger')
    if ledger_path:
        ledger = json.loads(writable(ledger_path).read_text())
        live = scheduler_attempts([str(a['job_id']) for a in ledger['attempts']])
        admission = campaign_admission(ledger, job, live)
        registered = next(a for a in ledger['attempts'] if str(a['job_id']) == job)
        if registered['gpus'] != gpus or registered['allocated_seconds'] != profile['allocated_seconds']:
            raise ValueError('Shared campaign registration differs from frozen allocation')
        save(control/f'allocation-admission-{job}.json', admission)
    elif gpus != 8:
        raise ValueError('Split allocations require shared all-attempt accounting')
    validate_source(source)
    if sha(control/'preparation.json') != request['prepared_execution_sha256']:
        raise ValueError('Execution manifest changed')
    preparation = json.loads((control/'preparation.json').read_text())
    if (sha(control/'controller.py') != preparation['controller_sha256']
            or sha(control/'launch.sbatch') != preparation['batch_sha256']):
        raise ValueError('Prepared controller or batch changed')
    for name, expected in preparation['files'].items():
        if name == 'request.json':
            continue  # Contains the final preparation digest; avoid a hash cycle.
        if sha(control/name) != expected:
            raise ValueError('Prepared execution input changed: '+name)
    if sha(control/'fixture.pt') != request['fixture_sha256'] or sha(source/'reference_manifest.json') != request['source_manifest_sha256']:
        raise ValueError('Prepared inputs changed')
    # Keep the legacy outstanding reservations and a two-TiB floor, in addition
    # to this new diagnostic's one-TiB protective ceiling.
    quota = check_storage_quota(minimum_free_bytes=(6 if gpus == 2 else 5)*1024**4)
    save(control/f'storage-{job}.json', quota)
    started = time.time(); deadline = started + min(alloc['maximum_seconds'], current[0]['allocated_seconds']-180)
    try:
        for topology in cases:
            tp = topology['tp']
            case = control/f'tp{tp}-dp{gpus//tp}'
            if (case/'result.json').exists() or (case/'native.log').exists() or list(case.glob('native-gate-rank*.json')):
                raise ValueError('Existing case output requires explicit reviewed recovery; do not overwrite')
            save(control/'status.json', dict(job_id=job, stage=f'native-tp{tp}', started_epoch=started,
                epoch=time.time(), production_launch_ready=False))
            plan = json.loads((case/'plan.json').read_text())
            gate = json.loads((case/'gate-manifest.json').read_text())
            if (gate['output_dir'] != str(case) or gate['fixture_path'] != str(control/'fixture.pt')
                    or gate['hf_checkpoint'] != str(MODEL)
                    or gate['source_manifest_sha256'] != request['source_manifest_sha256']):
                raise ValueError('Native case path/provenance changed')
            env = safe_environment(control); env.update(plan['environment'])
            seconds = min(9000, int(deadline-time.time()))
            if seconds < 600:
                raise TimeoutError('Insufficient remaining bounded time for another case')
            command = source_command(source, ['timeout', '--signal=TERM', '--kill-after=60', str(seconds), *plan['command']])
            with (case/'native.log').open('x') as log:
                subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            reports = [case/f'native-gate-rank{rank}.json' for rank in range(gpus)]
            if set(case.glob('native-gate-rank*.json')) != set(reports):
                raise ValueError('Native gate rank inventory differs from the allocated world size')
            for rank, report_path in enumerate(reports):
                report = json.loads(report_path.read_text())
                validate_native_report(report, rank=rank, gpus=gpus, tp=tp,
                    manifest_sha256=sha(case/'gate-manifest.json'),
                    gate_sha256=sha(source/'openwebrl/action_credit_native_gate.py'),
                    backend_sha256=sha(source/'openwebrl/action_credit_megatron.py'))
            save(control/f'case-exit-tp{tp}.json', dict(exit_code=0, epoch=time.time(),
                independent_artifact_review_pending=True,
                reports={p.name:sha(p) for p in reports}))
        save(control/'status.json', dict(job_id=job, stage='native_workers_exited_review_pending',
            epoch=time.time(), elapsed_seconds=time.time()-started, verified_complete=False,
            production_launch_ready=False))
    except BaseException as error:
        save(control/'status.json', dict(job_id=job, stage='failed', epoch=time.time(),
            elapsed_seconds=time.time()-started, error_type=type(error).__name__,
            error=str(error)[:1000], verified_complete=False, production_launch_ready=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--gpus', type=int, choices=(2, 8), default=8)
    parser.add_argument('--tp-sizes', type=int, nargs='+')
    parser.add_argument('--allocated-seconds', type=int, default=CAP_SECONDS)
    parser.add_argument('--campaign-ledger', type=Path)
    parser.add_argument('--mechanics-source', type=Path,
        help='Read mechanics modules from a reviewed snapshot instead of the working tree')
    parser.add_argument('--native-gate-source', type=Path,
        help='Explicit corrected diagnostic oracle; all other modules retain snapshot bytes')
    args = parser.parse_args()
    if args.worker:
        worker(args.control)
    elif args.fixture:
        prepare(args.control, args.fixture, gpus=args.gpus, tp_sizes=args.tp_sizes,
                allocated_seconds=args.allocated_seconds, campaign_ledger=args.campaign_ledger,
                mechanics_source=args.mechanics_source, native_gate_source=args.native_gate_source)
    else:
        parser.error('--fixture is required for preparation')


if __name__ == '__main__':
    main()
