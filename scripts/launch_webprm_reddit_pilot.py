#!/usr/bin/env python3
"""Review or own the proposed five-row Reddit pilot; never requests resources.

Execution requires a separately approved, hash-bound proposal and an existing
matching allocation. The default prints the concrete commands without spawning
models, browsers, containers or workers. Historical hosting wrappers stay frozen.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import re
import shutil
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.webprm_broker import BrokerProcess, make_broker_client
from openwebrl.webprm_lifecycle import OwnedEpisodeTree, _process, _processes, _subreaper
from openwebrl.webprm_reddit_provenance import RUNTIME, SCRATCH, ORIGINAL_MANIFEST, checked_path

RESOURCES = dict(gpus=1, gpu_type='H200', cpus=8, memory_gib=120, total_seconds=10800)
LIMITS = dict(api_usd=15., paid_requests=575, local_requests=320, total_requests=895,
              scratch_bytes=300 * 1024**3, runtime_reserve_bytes=16 * 1024**3,
              max_turns=30, max_candidates=5)
CORRECTED_PROFILE = 'reddit_corrected_v3_90m_400local'
CORRECTED_RESOURCES = dict(RESOURCES, total_seconds=5400)
CORRECTED_LIMITS = dict(LIMITS, local_requests=400, total_requests=975)
GUARDED_PROFILE = 'reddit_guarded_v4_90m_400local'
GUARDED_RESOURCES = dict(CORRECTED_RESOURCES)
GUARDED_LIMITS = dict(CORRECTED_LIMITS, paid_requests=600, total_requests=1000)
GUARDED_NO_SEARCH = 'single_blocks_envelope_v2'
GUARDED_EVALUATOR = 'native_with_explicit_na_abstention_guard_v1'
CPU_VALIDATION_PROFILE = 'reddit_no_search_guarded_v4_cpu_45m'
CPU_VALIDATION_RESOURCES = dict(gpus=0, gpu_type=None, cpus=8, memory_gib=16, total_seconds=2700)
CPU_VALIDATION_LIMITS = dict(GUARDED_LIMITS, api_usd=2., paid_requests=98, local_requests=0,
    total_requests=98, scratch_bytes=128 * 1024**3)
CPU_VALIDATION_PROTOCOL = 'reddit_no_search_guarded_browser_v1'
CORRECTED_TURN_LIMIT = 'released_viewer_na_fallback_v1'
CORRECTED_PIOTR_REPRESENTATION = 'native_operation_descriptor_preaction_status_v2'
ROWS = ('no_search', 'gpt4omini', 'shepherd8b', 'arbiter7b', 'piotr_selectionarm')
ACTOR = 'gpt-4o-mini-2024-07-18'
BROWSER = '/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/bin/python'
ENGINE = '/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python'


def resource_envelope(proposal):
    """A new explicit campaign profile never widens an old approved envelope."""
    profile = proposal.get('resource_profile')
    if profile is None:
        resources, limits = RESOURCES, LIMITS
    elif profile == CORRECTED_PROFILE:
        resources, limits = CORRECTED_RESOURCES, CORRECTED_LIMITS
    elif profile == GUARDED_PROFILE:
        resources, limits = GUARDED_RESOURCES, GUARDED_LIMITS
    elif profile == CPU_VALIDATION_PROFILE:
        resources, limits = CPU_VALIDATION_RESOURCES, CPU_VALIDATION_LIMITS
    else:
        raise ValueError('Unknown exact resource profile')
    if proposal.get('resources') != resources or proposal.get('limits') != limits:
        raise ValueError('Proposal must retain exact resource/API/request/storage envelope')
    return dict(resources), dict(limits)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    path = checked_path(path, RUNTIME)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return sha(path)


def reference(name):
    path = RUNTIME / name
    spec = importlib.util.spec_from_file_location('reddit_reference_' + Path(name).stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def environment(scratch, *, credentials=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('APPTAINER_', 'SINGULARITY_'))}
    if not credentials:
        env = {k: v for k, v in env.items() if not any(word in k for word in ('API_KEY', 'AUTH_TOKEN', 'ACCESS_TOKEN'))}
    for variable, directory in {
            'TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'cache', 'XDG_CONFIG_HOME': 'config', 'XDG_DATA_HOME': 'data',
            'XDG_RUNTIME_DIR': 'xdg-runtime', 'APPTAINER_TMPDIR': 'apptainer-tmp',
            'APPTAINER_CONFIGDIR': 'apptainer-config', 'APPTAINER_CACHEDIR': 'apptainer-cache',
            'HF_HOME': 'huggingface', 'TORCH_HOME': 'torch', 'TORCHINDUCTOR_CACHE_DIR': 'torchinductor',
            'TRITON_CACHE_DIR': 'triton', 'FLASHINFER_WORKSPACE_BASE': 'flashinfer',
            'CUDA_CACHE_PATH': 'cuda-cache', 'NUMBA_CACHE_DIR': 'numba',
            'SGLANG_CACHE_DIR':'sglang', 'SGLANG_DG_CACHE_DIR':'sglang-deepgemm',
            'MPLCONFIGDIR':'matplotlib','HF_MODULES_CACHE':'hf-modules',
            'TRANSFORMERS_CACHE':'transformers','OUTLINES_CACHE_DIR':'outlines',
            'HF_HUB_CACHE':'hf-hub','HUGGINGFACE_HUB_CACHE':'hf-hub',
            'WEBPRM_BROWSER_TMP_ROOT':'browser','TORCH_EXTENSIONS_DIR':'torch-extensions',
            'CCACHE_DIR':'ccache'}.items():
        path = checked_path(scratch / directory, SCRATCH); path.mkdir(exist_ok=True); env[variable] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONPATH=str(REPO),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        WANDB_PROJECT='openwebrl-evals', WANDB_MODE='disabled', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
        APPTAINER_NO_HOME='true', APPTAINER_CONTAINALL='true',
        APPTAINER_NO_MOUNT='bind-paths,home,cwd,hostfs,tmp', HISTFILE='/dev/null', BASH_ENV='/dev/null', ENV='/dev/null')
    cuda='/gpfs/scrubbed/zixianma/openwebrl-runtime/cuda'
    if not Path(cuda+'/bin/nvcc').is_file():raise ValueError('Previously validated CUDA toolkit/nvcc missing')
    env.update(CUDA_HOME=cuda,PATH=cuda+'/bin:'+env.get('PATH',''),
               LD_LIBRARY_PATH=cuda+'/lib64:'+env.get('LD_LIBRARY_PATH',''))
    return env


def piotr_compatible_config(model):
    config = read(Path(model['path']) / 'config.json')
    rope = dict(config['text_config']['rope_parameters'])
    theta = rope.pop('rope_theta')
    if rope != dict(mrope_interleaved=True, mrope_section=[24, 20, 20], rope_type='default') or theta != 5000000:
        raise ValueError('Piotr published RoPE representation changed')
    config['text_config'].update(rope_scaling=rope, rope_theta=theta)
    return config


def model_command(name, model, port, *, compatible_config=None):
    command = [ENGINE, '-m', 'sglang.launch_server', '--model-path', model['path'],
        '--served-model-name', model['served_model'], '--host', '127.0.0.1', '--port', str(port),
        '--tp', '1', '--dtype', 'bfloat16', '--mem-fraction-static', '.70',
        '--max-total-tokens', str(model['max_total_tokens']), '--context-length', str(model['context_length']),
        '--max-running-requests', '1', '--random-seed', '42', '--disable-cuda-graph',
        '--disable-custom-all-reduce', '--skip-server-warmup']
    if name == 'piotr_selectionarm':
        from scripts.prepare_webprm_piotr_processor import validate_view
        validate_view(model['processor_path'])
        command += ['--tokenizer-path', model['processor_path'], '--chat-template',
                    str(Path(model['processor_path']) / 'chat_template.jinja')]
        piotr_compatible_config(model)
        command += ['--decrypted-config-file', str(compatible_config or '<generated-compatible-config.json>')]
    return command


def validate(proposal):
    proposal_lineage(proposal)
    checked_path(proposal['campaign_root'],RUNTIME)
    resource_envelope(proposal)
    cpu_only = proposal.get('resource_profile') == CPU_VALIDATION_PROFILE
    if proposal.get('rows') != (['no_search'] if cpu_only else list(ROWS)) or proposal.get('task_ids') != [5] or proposal.get('actor') != ACTOR:
        raise ValueError('Only the prespecified five-row, one-task mini pilot is implemented')
    if proposal.get('restore') != 'verified_local_oci':
        raise ValueError('SIF restore is not validated; this controller uses the verified local OCI route')
    expected_judge=dict(fuzzy_enabled=True,fuzzy_model='gpt-4o-2024-05-13',
        fuzzy_protocol='webarena_fuzzy_gpt4o_20240513_v1',fuzzy_model_change_approved=True)
    if proposal.get('activation_changes',{}).get('worker_scientific_fields')!=expected_judge:
        raise ValueError('Exact proposed judge activation must be reviewable before approval')
    if set(proposal['models']) != (set() if cpu_only else {'shepherd8b', 'arbiter7b', 'piotr_selectionarm'}):
        raise ValueError('Exactly three fixed critic servers required')
    for path, expected in proposal['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen preparation source changed: ' + path)
    for model in proposal['models'].values():
        if sha(Path(model['path']) / 'config.json') != model['config_sha256']:
            raise ValueError('Frozen critic config changed')
        for filename, size in model['weight_sizes'].items():
            if (Path(model['path']) / filename).stat().st_size != size:
                raise ValueError('Previously verified critic weight size changed')
    if sha(proposal['worker_plan']) != proposal['worker_plan_sha256']:
        raise ValueError('Worker protocol proposal changed')
    plan = read(proposal['worker_plan'])
    if cpu_only and (proposal.get('diagnostic_protocol') != CPU_VALIDATION_PROTOCOL
            or proposal.get('maximum_allocation_attempts') != 1
            or any(route.get('local') is not False for route in proposal['broker_routes'])
            or set(proposal['cell_routes']) != {'no_search'}
            or plan.get('diagnostic_protocol') != CPU_VALIDATION_PROTOCOL
            or plan.get('selectors') != ['no_search']):
        raise ValueError('CPU validation must be a single fixed NoSearch diagnostic with no local routes or retries')
    if proposal.get('resource_profile') in (CORRECTED_PROFILE, GUARDED_PROFILE, CPU_VALIDATION_PROFILE):
        from openwebrl.webprm_webarena_worker import protocol_definition
        from openwebrl.webprm_reddit_pilot_preflight import NATIVE_FUZZY_PREFLIGHT, GUARDED_FUZZY_PREFLIGHT
        guarded = proposal.get('resource_profile') in (GUARDED_PROFILE, CPU_VALIDATION_PROFILE)
        expected_preflight = GUARDED_FUZZY_PREFLIGHT if guarded else NATIVE_FUZZY_PREFLIGHT
        if proposal.get('native_fuzzy_preflight') != expected_preflight:
            raise ValueError('Corrected cohort requires the frozen metered native fuzzy preflight')
        if (proposal.get('piotr_representation') != CORRECTED_PIOTR_REPRESENTATION or
                plan.get('piotr_representation') != CORRECTED_PIOTR_REPRESENTATION):
            raise ValueError('Corrected cohort requires the explicit frozen Piotr descriptor-v2 representation')
        version = 'webarena_table5_viewer_reconstruction_v4' if guarded else 'webarena_table5_viewer_reconstruction_v3'
        definition_options = (dict(no_search_prompt=GUARDED_NO_SEARCH, scoring_protocol=GUARDED_EVALUATOR)
            if guarded else {})
        if (proposal.get('turn_limit_protocol') != CORRECTED_TURN_LIMIT or
                plan.get('turn_limit_protocol') != CORRECTED_TURN_LIMIT or
                plan.get('protocol') != version or
                plan.get('protocol_contract') != protocol_definition(plan['arbiter_source'], turn_limit=CORRECTED_TURN_LIMIT,
                    piotr_representation=CORRECTED_PIOTR_REPRESENTATION, **definition_options)):
            raise ValueError('Corrected cohort requires the explicit frozen v3 N/A fallback or guarded v4 contract')
        if guarded:
            for key, value in (('no_search_prompt_version', GUARDED_NO_SEARCH), ('evaluator_protocol', GUARDED_EVALUATOR)):
                if proposal.get(key) != value or plan.get(key) != value:
                    raise ValueError('Guarded cohort requires explicit frozen prompt and scoring identities')
            prior = proposal.get('previous_corrected_cohort', {})
            prior_path = Path(prior.get('campaign_root', '/'))
            current = Path(proposal['campaign_root'])
            if (prior.get('relationship') != 'separate_preserved_diagnostic'
                    or prior.get('budget_transfer') is not False or prior.get('rows_reused') is not False
                    or current == prior_path or current in prior_path.parents or prior_path in current.parents):
                raise ValueError('Guarded v4 cohort must preserve the completed corrected cohort separately')
            checked_path(prior_path, RUNTIME)
        elif any(value.get('evaluator_protocol') is not None or value.get('no_search_prompt_version') is not None
                 for value in (proposal, plan)):
            raise ValueError('Corrected v3 approval cannot activate v4 prompt or scoring changes')
        previous = RUNTIME/'reddit-pilot-campaign-v1'
        current = Path(proposal['campaign_root'])
        if (current == previous or previous in current.parents or current in previous.parents or
                proposal.get('previous_cohort') != dict(campaign_root=str(previous),
                    relationship='separate_preserved_diagnostic',budget_transfer=False,rows_reused=False)):
            raise ValueError('Corrected pilot must preserve the old cohort separately with zero budget or row transfer')
    elif (proposal.get('turn_limit_protocol') is not None or plan.get('turn_limit_protocol') == CORRECTED_TURN_LIMIT or
            plan.get('protocol') in ('webarena_table5_viewer_reconstruction_v3', 'webarena_table5_viewer_reconstruction_v4') or
            proposal.get('evaluator_protocol') is not None or plan.get('evaluator_protocol') is not None or
            proposal.get('no_search_prompt_version') is not None or plan.get('no_search_prompt_version') is not None or
            proposal.get('piotr_representation') == CORRECTED_PIOTR_REPRESENTATION or
            plan.get('piotr_representation') == CORRECTED_PIOTR_REPRESENTATION):
        raise ValueError('Legacy approval cannot enable corrected v3 turn-limit or Piotr representation behavior')
    return proposal


def proposal_lineage(proposal):
    """Allow recorded operational repairs while preserving the approved experiment."""
    lineage = [digest(proposal)]
    current = proposal
    operational_files = {
        str(REPO / name) for name in (
            'scripts/launch_webprm_reddit_pilot.py',
            'scripts/prepare_webprm_piotr_processor.py',
            'tests/test_webprm_reddit_pilot_controller.py',
            'tests/test_prepare_webprm_piotr_processor.py',
            'openwebrl/webprm_webarena_worker.py',
            'tests/test_webprm_webarena_worker.py')}
    while current.get('operational_parent'):
        link = current['operational_parent']
        parent = read(checked_path(link['proposal_path'], RUNTIME))
        identity = digest(parent)
        if identity != link['proposal_sha256'] or identity in lineage or not link.get('reason'):
            raise ValueError('Invalid append-only operational revision lineage')
        def scientific(value):
            return {k: v for k, v in value.items() if k not in
                    ('source_sha256', 'operational_parent', 'worker_plan', 'worker_plan_sha256')}
        if scientific(current) != scientific(parent):
            raise ValueError('Operational repair changed approved scientific settings or budgets')
        if any(current[key] != parent[key] for key in ('worker_plan', 'worker_plan_sha256')):
            plans=[]
            for value in (parent,current):
                path=checked_path(value['worker_plan'],RUNTIME)
                if sha(path)!=value['worker_plan_sha256']:
                    raise ValueError('Operational worker plan digest mismatch')
                plans.append(read(path))
            old_plan,new_plan=plans
            if ({k:v for k,v in old_plan.items() if k!='code_sha256'} !=
                    {k:v for k,v in new_plan.items() if k!='code_sha256'}):
                raise ValueError('Operational repair changed worker scientific settings')
            old_code,new_code=old_plan['code_sha256'],new_plan['code_sha256']
            code_changes={path for path in set(old_code)|set(new_code) if old_code.get(path)!=new_code.get(path)}
            if (set(old_code)!=set(new_code) or not code_changes or
                    code_changes!={str(REPO/'openwebrl/webprm_webarena_worker.py')} or
                    any(parent['source_sha256'].get(path)!=old_code[path] or
                        current['source_sha256'].get(path)!=new_code[path] for path in code_changes)):
                raise ValueError('Unsupported operational worker code change')
        old, new = parent['source_sha256'], current['source_sha256']
        changed = {path for path in set(old) | set(new) if old.get(path) != new.get(path)}
        if not changed or not changed <= operational_files or set(old) != set(new):
            raise ValueError('Operational repair changes unsupported source files')
        lineage.append(identity)
        current = parent
    return lineage


def preview(proposal):
    validate(proposal)
    resources, limits = resource_envelope(proposal)
    source_bytes = proposal['original_root_apparent_bytes']
    copies = len(proposal['rows']) + 1
    projected = source_bytes * copies + limits['runtime_reserve_bytes']
    return dict(status='proposal_only_not_executed', resources=resources, limits=limits,
        rows=proposal['rows'], task_ids=[5], actor=ACTOR,
        cache_restore=dict(method='local OCI → pristine sandbox', previously_observed_seconds=144.114,
                          argv=['apptainer', 'build', '--disable-cache', '--sandbox', '<fresh-baseline>', proposal['oci_reference']],
                          sif_restore_validated=False),
        fresh_states=dict(count=len(proposal['rows']), method='cp -a --reflink=auto --', deletion_commands=False,
                          conservative_baseline_and_fresh_states_bytes=copies * source_bytes,
                          with_runtime_reserve_bytes=projected, scratch_cap_bytes=limits['scratch_bytes'],
                          fits_scratch_cap=projected <= limits['scratch_bytes']),
        servers={name:model_command(name, model, 18011+i) for i,(name,model) in enumerate(proposal['models'].items())},
        stage_order=['Verify approval/job/resources/remaining retry budget and frozen inputs',
            'Run bounded network-isolated synthetic BrowserGym/native observation/action gate under exact scratch environment',
            'Build pristine original sandbox from already verified local OCI; retain all temporary files',
            *(['Start and own all three local servers; validate model identities'] if proposal['models'] else []),
            f"Start shared-budget host broker; fixed routes,{limits['paid_requests']}paid/{limits['local_requests']}local/{limits['total_requests']}total requests",
            *(['Validate all local critic contracts through the metered broker (eleven local calls)'] if proposal['models'] else []),
            *(['Run eight metered positive/negative/abstention controls; retain raw native and guarded scores and fail before primary admission']
              if proposal.get('resource_profile') in (GUARDED_PROFILE, CPU_VALIDATION_PROFILE) else
              ['Run one metered synthetic native fuzzy known-answer check; fail before any primary claim if provider/parser/verdict fails']
              if proposal.get('resource_profile') == CORRECTED_PROFILE else []),
            'For each row: fresh original copy → owned isolated site → frozen lease → namespace worker → durable result → site closure',
            'Close broker, all servers and all owned descendants; persist scheduler/runtime/accounting receipts'],
        blockers=proposal['execution_blockers'], new_allocations=0, provider_calls=0, model_loads=0)


def approved_context(proposal, approval, scheduler_text, prior_seconds):
    resources, limits = resource_envelope(proposal)
    if approval.get('status') != 'approved' or approval.get('proposal_sha256') != digest(proposal):
        raise ValueError('Exact proposal lacks explicit approval')
    if (approval.get('resource_profile') != proposal.get('resource_profile') or
            approval.get('resources') != resources or approval.get('limits') != limits):
        raise ValueError('Approval resource/API envelope differs')
    if proposal.get('resource_profile') in (CORRECTED_PROFILE, GUARDED_PROFILE, CPU_VALIDATION_PROFILE) and (
            approval.get('piotr_representation') != CORRECTED_PIOTR_REPRESENTATION or
            approval.get('piotr_representation_approved') is not True or
            approval.get('turn_limit_protocol') != CORRECTED_TURN_LIMIT or
            approval.get('turn_limit_protocol_approved') is not True):
        raise ValueError('Corrected v3 fallback and Piotr representation lack explicit separate-cohort approval')
    if proposal.get('resource_profile') in (GUARDED_PROFILE, CPU_VALIDATION_PROFILE) and (
            approval.get('evaluator_protocol') != GUARDED_EVALUATOR or
            approval.get('evaluator_protocol_approved') is not True or
            approval.get('no_search_prompt_version') != GUARDED_NO_SEARCH or
            approval.get('no_search_prompt_version_approved') is not True):
        raise ValueError('Guarded v4 scoring and NoSearch prompt lack explicit separate-cohort approval')
    if type(prior_seconds) is not int or not 0 <= prior_seconds < resources['total_seconds']:
        raise ValueError('Prior attempts exhausted or invalid')
    job = str(approval['job_id'])
    if os.environ.get('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Controller does not own the approved job')
    fields=dict(re.findall(r'(\w+)=(\S+)',scheduler_text))
    tres=dict(item.split('=',1) for item in fields.get('AllocTRES','').split(',') if '=' in item)
    cpu_only = proposal.get('resource_profile') == CPU_VALIDATION_PROFILE
    gpu_ok = (all(value == '0' for key,value in tres.items() if key.startswith('gres/gpu'))
              if cpu_only else tres.get('gres/gpu:h200')=='1')
    if (fields.get('JobId')!=job or fields.get('JobState')!='RUNNING' or
            not fields.get('UserId','').endswith(f'({os.getuid()})') or fields.get('NumNodes')!='1' or
            fields.get('NumCPUs')!=str(resources['cpus']) or not gpu_ok or
            tres.get('mem') not in (str(resources['memory_gib'])+'G',str(resources['memory_gib']*1024)+'M')):
        raise ValueError('Actual allocation differs from approved1H200/8CPU/120GiB')
    limit=fields['TimeLimit'];days,clock=limit.split('-',1) if '-' in limit else ('0',limit)
    parts=[int(x) for x in clock.split(':')]
    if len(parts)==2:parts.insert(0,0)
    duration=int(days)*86400+parts[0]*3600+parts[1]*60+parts[2]
    if duration<=0 or duration+prior_seconds>resources['total_seconds']:
        label = 'shared45min' if cpu_only else 'shared3h' if proposal.get('resource_profile') is None else 'shared90min'
        raise ValueError('Attempt would exceed '+label+' cap including all retries')
    if cpu_only:
        # A fresh no-requeue diagnostic has one authoritative Unix epoch clock;
        # never interpret scheduler-local text in the worker's local timezone.
        snapshot=approval.get('scheduler_epoch_proof',{})
        if (snapshot.get('job_id')!=job or type(snapshot.get('start_epoch')) is not int
                or type(snapshot.get('end_epoch')) is not int or snapshot.get('limit_seconds')!=duration
                or not snapshot['start_epoch']<=time.time()<snapshot['end_epoch']):
            raise ValueError('CPU diagnostic requires a matching finite scheduler epoch proof')
        end=min(snapshot['end_epoch'],snapshot['start_epoch']+duration)
    else:
        end=datetime.strptime(fields['EndTime'],'%Y-%m-%dT%H:%M:%S').timestamp()
    if end-180<=time.time():raise ValueError('No approved runtime remains before cleanup reserve')
    if (approval.get('terminal_judge_replacement_approved') is not True or
            approval.get('terminal_judge_model')!='gpt-4o-2024-05-13'):
        raise ValueError('Exact proposed terminal judge replacement lacks explicit approval')
    if proposal['execution_blockers']!=proposal.get('approval_only_blockers'):
        raise ValueError('Proposal has unresolved blockers beyond the exact bundled approval')
    return dict(job_id=job, deadline=end-180, prior_consumed_seconds=prior_seconds)


def _command(argv, log, env):
    stream = Path(log).open('x')
    try:
        return subprocess.Popen(argv, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        stream.close()


def _stop(process, seconds=25):
    if process is None: return
    if process.poll() is None:
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
    try: process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait(timeout=5)


def _wait(process, deadline, check):
    while process.poll() is None:
        if time.time() >= deadline: raise TimeoutError('Shared allocation deadline reached')
        check(); time.sleep(.5)
    if process.returncode: raise RuntimeError('Owned command failed; retained log and no automatic retry')


def cell_outcome(result_path, row, identity):
    result=read(result_path)
    if (result.get('identity_sha256')!=identity or result.get('arm')!=ACTOR+'__'+row or
            result.get('selector',row)!=row or result.get('task_id')!='webarena_lite_005' or
            type(result.get('attempted')) is not bool or type(result.get('valid')) is not bool or
            type(result.get('success')) is not int or result['success'] not in (0,1) or
            result['success'] and not result['valid']):
        raise RuntimeError('Assigned durable result missing/mismatched; no success inferred from exit0')
    watchdog=read(Path(result_path).parent/'watchdog.json')
    if watchdog!=result.get('watchdog') or watchdog.get('remaining_owned_processes')!={}:
        raise RuntimeError('Worker process closure is absent or incomplete')
    return dict(result_path=str(result_path),result_sha256=sha(result_path),
        watchdog_sha256=sha(Path(result_path).parent/'watchdog.json'),attempted=result['attempted'],
        valid=result['valid'],success=result['success'],identity_sha256=identity,selector=row)


def campaign_summary(outcomes, missing, rows=ROWS):
    by_row={row['selector']:row for row in outcomes}
    if len(by_row)!=len(outcomes):raise ValueError('Duplicate fixed cell outcomes across attempts')
    complete=set(by_row)==set(rows) and not missing
    return dict(status='fixed_attempts_completed' if complete else 'partial_preserved_attempts',
        planned_cells=len(rows),recorded_cells=len(by_row),missing_rows=sorted(set(rows)-set(by_row)),
        attempted=sum(row['attempted'] for row in outcomes),valid=sum(row['valid'] for row in outcomes),
        successes=sum(row['success'] for row in outcomes),
        all_attempted=complete and all(row['attempted'] for row in outcomes),
        all_mechanics_valid=complete and all(row['valid'] for row in outcomes),outcomes=outcomes)


def website_ready(base_url='http://127.0.0.1'):
    """Check database-backed pages with the login site's cookie handshake."""
    from http.cookiejar import CookieJar
    from urllib.request import build_opener, HTTPCookieProcessor, ProxyHandler
    opener = build_opener(ProxyHandler({}), HTTPCookieProcessor(CookieJar()))
    for path in ('/login', '/f/books'):
        with opener.open(base_url + path, timeout=2) as response:
            response.read(1024)


def _runner(proposal, context, output, scratch):
    """Runs under one outer subreaper; every descendant belongs to that owner."""
    os.setsid()
    output, scratch = Path(output), Path(scratch)
    _, limits = resource_envelope(proposal)
    env = environment(scratch);os.environ.update(env)
    servers=[]; broker=None; site=None; worker=None; port_lease=None
    preparation = reference('prepare_reddit_original.py'); preparation.SCRATCH = scratch
    last_storage=[0.]
    def check():
        if any(p.poll() is not None for p in servers): raise RuntimeError('A critic server exited')
        if time.monotonic()-last_storage[0]>15:
            used = int(subprocess.check_output(['du', '--apparent-size', '-s', '-B1', str(scratch)], text=True,timeout=30).split()[0])
            if used+context['prior_retained_scratch_bytes'] > limits['scratch_bytes']: raise RuntimeError('Cumulative scratch cap reached; no deletion or continuation')
            last_storage[0]=time.monotonic()
    try:
        # Separate owned, network-isolated synthetic browser gate before any GPU
        # startup or fixed-cell claim. The outer subreaper also bounds its cleanup.
        browser_deadline=min(context['deadline'],time.time()+170)
        probe=_command(['unshare','--user','--map-root-user','--net',BROWSER,'-B','-m',
            'openwebrl.webprm_browser_preflight','--arbiter-source',read(proposal['worker_plan'])['arbiter_source'],
            '--output',str(output/'browser-preflight'),'--deadline',str(browser_deadline),
            '--host-netns',os.readlink('/proc/self/ns/net'),
            *(['--terminal-answer-check'] if proposal.get('resource_profile')==CPU_VALIDATION_PROFILE else [])],
            output/'browser-preflight.log',env)
        try:_wait(probe,browser_deadline,check)
        finally:_stop(probe,seconds=5)
        if read(output/'browser-preflight/receipt.json').get('status')!='passed':
            raise RuntimeError('Synthetic browser startup receipt did not pass')
        build_out=Path(context['reuse_build_receipt']).parent if context.get('reuse_build_receipt') else output/'original-build'
        if not context.get('reuse_build_receipt'):
            preparation.prepare('build',scratch/'baseline',build_out,context['deadline'],
                                download_receipt=proposal['download_receipt'])
        worker_plan=read(proposal['worker_plan'])
        from scripts.runtime_ports import lease_ports
        port_directory=checked_path(scratch/'port-leases',SCRATCH)
        port_directory.mkdir(mode=0o700,exist_ok=False)
        port_lease,port_base=lease_ports(context['job_id'],size=3,directory=port_directory)
        ports={name:port_base+i for i,name in enumerate(proposal['models'])}
        put(output/'port-lease.json',dict(ports=ports,cooperative_lease_held_until_children_exit=True))
        # The source protocol is fixed; only execution bindings derive from the exact approval.
        socket_dir=scratch/'broker';socket_dir.mkdir(mode=0o700)
        socket_path=socket_dir/'socket'
        if len(os.fsencode(socket_path))>=108: raise ValueError('Broker socket path too long')
        worker_plan.update(authorization_status='approved',job_id=context['job_id'],deadline=context['deadline'],
            api_usd_cap=limits['api_usd'],budget_path=str(Path(proposal['campaign_root'])/'shared-budget.json'),
            broker_socket=str(socket_path),endpoints={'reddit':'http://127.0.0.1'},
            actors=[ACTOR],selectors=list(proposal['rows']),task_ids=[5])
        # These exact scientific changes are part of the hash-bound proposal,
        # and approved_context separately requires the explicit judge decision.
        worker_plan.update(proposal['activation_changes']['worker_scientific_fields'])
        for i,(name,model) in enumerate(proposal['models'].items()):
            port=ports[name];prefix={'shepherd8b':'shepherd','arbiter7b':'arbiter','piotr_selectionarm':'piotr'}[name]
            worker_plan[prefix+'_model']=model['served_model'];worker_plan[prefix+'_endpoint']=f'http://127.0.0.1:{port}/v1'
            compatible_config=None
            if name=='piotr_selectionarm':
                compatible_config=output/'piotr-compatible-config.json'
                put(compatible_config,piotr_compatible_config(model))
            argv=model_command(name,model,port,compatible_config=compatible_config);put(output/(name+'-command.json'),dict(argv=argv))
            servers.append(_command(argv,output/(name+'-server.log'),env))
        startup_deadline=min(context['deadline'],time.time()+1200)
        ready=set()
        while len(ready)<len(proposal['models']):
            check()
            if time.time()>=startup_deadline:raise TimeoutError('Critic startup deadline')
            for name,model in proposal['models'].items():
                try:
                    with urlopen(f"http://127.0.0.1:{ports[name]}/v1/models",timeout=2) as response:
                        if model['served_model'] not in [row['id'] for row in json.load(response)['data']]:
                            raise RuntimeError('Served model identity mismatch')
                    ready.add(name)
                except OSError:pass
            time.sleep(1)
        put(output/'worker-plan.json',worker_plan)
        from openwebrl.webprm_webarena import digest as worker_digest
        identity=worker_digest(worker_plan)
        cells={};routes={}
        for route in proposal['broker_routes']:
            endpoint=None if not route['local'] else worker_plan[route['endpoint_key']]
            routes[(route['model'],route['purpose'])]=dict(local=route['local'],endpoint=endpoint,
                max_input_tokens=route['max_input_tokens'],max_output_tokens=route['max_output_tokens'])
        for row in context['remaining_rows']:
            folder=output/'episodes'/'webarena_lite_005'/(ACTOR+'__'+row);folder.mkdir(parents=True)
            cells['webarena_lite_005/'+ACTOR+'__'+row]=dict(config=dict(worker_plan,identity_sha256=identity,
                env_file=proposal['env_file']),journal=str(folder/'api.jsonl'),allowed_routes=proposal['cell_routes'][row])
        # Local contract probe is separately registered and shares the same durable counters.
        local_routes=[list(key) for key,value in routes.items() if value['local']]
        if local_routes:
            cells['local_preflight']=dict(config=dict(worker_plan,identity_sha256=identity),
                                         journal=str(output/'local-preflight-api.jsonl'),allowed_routes=local_routes)
        if proposal.get('resource_profile') in (CORRECTED_PROFILE, GUARDED_PROFILE, CPU_VALIDATION_PROFILE):
            from openwebrl.webprm_reddit_pilot_preflight import native_fuzzy_preflight_cell
            cells['native_fuzzy_preflight']=native_fuzzy_preflight_cell(
                dict(worker_plan,identity_sha256=identity),proposal['env_file'],output)
        broker=BrokerProcess(socket_path=str(socket_path),cells=cells,routes=routes,deadline=context['deadline'],
            max_dispatches=context['remaining_paid_requests']+context['remaining_local_requests'],
            max_paid_dispatches=context['remaining_paid_requests'],max_local_dispatches=context['remaining_local_requests'],
            audit_path=str(output/'broker.jsonl')).start()
        # This reviewed helper makes only local broker calls; missing helper is a hard preparation gate.
        from openwebrl.webprm_reddit_pilot_preflight import validate_local_critics
        if local_routes:
            proof=validate_local_critics(dict(worker_plan,identity_sha256=identity),make_broker_client(str(socket_path),cell_id='local_preflight',
                identity_sha256=identity,deadline=min(context['deadline'],time.time()+600)),output)
            put(output/'local-model-preflight.json',proof)
        if proposal.get('resource_profile') in (CORRECTED_PROFILE, GUARDED_PROFILE, CPU_VALIDATION_PROFILE):
            # The explicitly frozen synthetic judge controls consume the diagnostic
            # allowance. Any provider/parser/verdict failure stops before a
            # fixed row is claimed; retries retain cumulative campaign charges.
            from openwebrl.webprm_reddit_pilot_preflight import validate_native_fuzzy, validate_guarded_fuzzy
            preflight = validate_native_fuzzy if proposal.get('resource_profile') == CORRECTED_PROFILE else validate_guarded_fuzzy
            preflight(dict(worker_plan,identity_sha256=identity),
                make_broker_client(str(socket_path),cell_id='native_fuzzy_preflight',
                    identity_sha256=identity,deadline=min(context['deadline'],time.time()+360)),output)
        for row in context['remaining_rows']:
            check()
            used=int(subprocess.check_output(['du','--apparent-size','-s','-B1',str(scratch)],text=True,timeout=30).split()[0])
            if (used+context['prior_retained_scratch_bytes']+proposal['original_root_apparent_bytes']+
                    limits['runtime_reserve_bytes']>limits['scratch_bytes']):
                raise RuntimeError('Next fresh state would exhaust retained scratch plus runtime reserve')
            state=scratch/('state-'+row);copy_out=output/('copy-'+row)
            preparation.prepare('copy',state,copy_out,context['deadline'],build_receipt=build_out/'receipt.json',
                                verification=proposal['raw_verification'])
            site_out=output/('site-'+row)
            argv=[BROWSER,str(REPO/'scripts/run_webprm_reddit_service.py'),'--context',str(output/'execution-context.json'),
                '--sandbox',str(state),'--output',str(site_out),'--state-provenance',str(copy_out/'receipt.json')]
            site=_command(argv,output/(row+'-site-controller.log'),env)
            until=min(context['deadline'],time.time()+180)
            while not (site_out/'instance.json').exists():
                if site.poll() is not None or time.time()>=until:raise RuntimeError('Original site startup failed')
                check();time.sleep(.5)
            instance=read(site_out/'instance.json')
            # instance.json precedes service creation in the proven wrapper.
            # Wait for PHP/database-backed HTTP availability before any actor call.
            probe_code="from scripts.launch_webprm_reddit_pilot import website_ready; website_ready(); print('ready')"
            ready=False;probe_number=0
            while time.time()<until:
                check()
                if site.poll() is not None:raise RuntimeError('Website controller exited during readiness')
                probe_number+=1
                probe=_command(['nsenter','--target',str(instance['service_pid']),'--user','--net','--preserve-credentials',
                    BROWSER,'-B','-c',probe_code],output/(row+f'-readiness-{probe_number:03d}.log'),env)
                try:probe.wait(timeout=6)
                except subprocess.TimeoutExpired:_stop(probe,seconds=1)
                if probe.returncode==0:ready=True;break
                time.sleep(1)
            if not ready:raise RuntimeError('Database-backed website readiness deadline reached')
            lease=output/('lease-'+row+'.json')
            put(lease,dict(task_id='webarena_lite_005',actor_model=ACTOR,selector=row,
                           instance_receipt=str(site_out/'instance.json'),instance_receipt_sha256=sha(site_out/'instance.json')))
            argv=['nsenter','--target',str(instance['service_pid']),'--user','--net','--preserve-credentials',BROWSER,
                str(REPO/'scripts/run_webprm_webarena_pilot.py'),'--plan',str(output/'worker-plan.json'),
                '--output',str(output/'episodes'),'--execute','--task-id','5','--actor',ACTOR,'--selector',row,'--lease',str(lease)]
            put(output/('worker-launch-'+row+'.json'),dict(selector=row,argv=argv,at_epoch=time.time(),
                                                          resampling_forbidden=True))
            worker=_command(argv,output/(row+'-worker.log'),env)
            _wait(worker,context['deadline'],check);worker=None
            result_path=output/'episodes'/'webarena_lite_005'/(ACTOR+'__'+row)/'result.json'
            put(output/('cell-outcome-'+row+'.json'),cell_outcome(result_path,row,identity))
            put(site_out/'stop.json',dict(reason='fixed cell finished',at_epoch=time.time()))
            _wait(site,min(context['deadline'],time.time()+30),check);site=None
            closure=read(site_out/'closure.json')
            if closure['remaining_owned_processes']:raise RuntimeError('Website cleanup incomplete; stop all cells')
        outcomes=[read(output/('cell-outcome-'+row+'.json')) for row in context['remaining_rows']]
        put(output/'completed.json',dict(at_epoch=time.time(),rows_executed_in_attempt=context['remaining_rows'],
            campaign=campaign_summary(context['prior_outcomes']+outcomes,context['prior_missing_rows'],proposal['rows'])))
    finally:
        _stop(worker);_stop(site)
        if broker is not None:put(output/'broker-closure.json',broker.close())
        for process in reversed(servers):_stop(process)
        if port_lease is not None:port_lease.close()


def retained_storage(storage, context):
    """Reconcile host-cleared job-local scratch without deleting prior evidence."""
    path = checked_path(storage['scratch_path'], SCRATCH)
    original = storage['apparent_bytes']
    if path.exists():
        return original, dict(status='retained', original_bytes=original, charged_bytes=original)
    record = subprocess.check_output(['sacct', '-j', str(context['job_id']), '-X', '--noheader',
        '--parsable2', '--format=JobID,State,NodeList'], text=True).strip().splitlines()
    fields = record[0].split('|') if len(record) == 1 else []
    terminal = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE'}
    if (len(fields) >= 3 and fields[0] == str(context['job_id']) and
            fields[1].split()[0] in terminal and fields[2] == socket.gethostname()):
        return 0, dict(status='host_cleared_after_job', original_bytes=original,
            charged_bytes=0, path=str(path), scheduler_record=record[0], observed_host=socket.gethostname())
    return original, dict(status='absence_not_verified_on_original_node', original_bytes=original, charged_bytes=original)


def campaign_history(proposal, attempt_ids, *, reconcile_node_local=False):
    """Charge every earlier attempt, including unknown calls and retained states."""
    _, limits = resource_envelope(proposal)
    campaign=checked_path(proposal['campaign_root'],RUNTIME);campaign.mkdir(exist_ok=True)
    existing=list(campaign.glob('attempt-*/execution-context.json'))
    contexts=[read(path) for path in existing]
    rejected={read(path)['job_id']:path for path in campaign.glob('attempt-*/preexecution-rejection.json')}
    if {str(row['job_id']) for row in contexts} | set(rejected) != set(attempt_ids):
        raise ValueError('Approval omits a registered prior attempt or invents a missing receipt')
    paid=local=retained=0;launched=set();seen=set();chain=[];reuse=None;outcomes=[];missing=[];storage_audits=[]
    from openwebrl.webprm_reddit_provenance import validate_original_build
    for job in attempt_ids:
        folder=campaign/('attempt-'+job)
        if job in rejected:
            rejection=read(rejected[job])
            if (rejection.get('proposal_sha256') not in proposal_lineage(proposal) or
                    rejection.get('no_children_spawned') is not True or
                    rejection.get('retained_scratch_bytes') != 0 or
                    sha(rejection['batch_log']) != rejection['batch_log_sha256'] or
                    any(path.name != 'preexecution-rejection.json' for path in folder.iterdir())):
                raise ValueError('Preexecution rejection evidence is incomplete')
            chain.append(dict(job_id=job,preexecution_rejection_sha256=sha(rejected[job])))
            continue
        context=read(folder/'execution-context.json')
        if context['proposal_sha256'] not in proposal_lineage(proposal):
            raise ValueError('Campaign identity changed outside recorded operational repair lineage')
        closure=read(folder/'controller-closure.json')
        if closure['remaining_owned_processes']:raise ValueError('Prior owned process cleanup unverified')
        storage=read(folder/'storage-final.json')
        if type(storage['apparent_bytes']) is not int or storage['apparent_bytes']<0:raise ValueError('Invalid retained storage accounting')
        if storage['scratch_path']!=context['scratch']:raise ValueError('Prior retained scratch identity differs')
        charged,audit=retained_storage(storage,context) if reconcile_node_local else (storage['apparent_bytes'],dict(status='retained'))
        retained+=charged;storage_audits.append(dict(job_id=job,**audit))
        if (folder/'broker.jsonl').exists():
            for line in (folder/'broker.jsonl').read_text().splitlines():
                row=json.loads(line)
                if row.get('event')!='dispatch_started':continue
                if row['request_id'] in seen or type(row.get('local')) is not bool:raise ValueError('Invalid prior dispatch ledger')
                seen.add(row['request_id']);local+=int(row['local']);paid+=int(not row['local'])
        # Older attempts let SGLang issue one internal warmup per launched server.
        # Charge every possible request, even if startup failed before transmission.
        # New launches disable this path and use only the metered broker preflight.
        for command_path in folder.glob('*-command.json'):
            argv=read(command_path).get('argv',[])
            if 'sglang.launch_server' in argv and '--skip-server-warmup' not in argv:
                local+=1
        for row in proposal['rows']:
            if (folder/('worker-launch-'+row+'.json')).exists():
                if row in launched:raise ValueError('Fixed cell was resampled across attempts')
                launched.add(row)
                result_path=folder/'episodes'/'webarena_lite_005'/(ACTOR+'__'+row)/'result.json'
                if result_path.exists():
                    from openwebrl.webprm_webarena import digest as worker_digest
                    outcomes.append(cell_outcome(result_path,row,worker_digest(read(folder/'worker-plan.json'))))
                else:missing.append(row)
        candidate=folder/'original-build/receipt.json'
        if candidate.exists() and reuse is None:
            try:validate_original_build(candidate);reuse=str(candidate)
            except (ValueError,OSError):pass # A different node cannot reuse the old node-local baseline.
        chain.append(dict(job_id=job,context_sha256=sha(folder/'execution-context.json'),
            closure_sha256=sha(folder/'controller-closure.json'),storage_sha256=sha(folder/'storage-final.json'),
            broker_audit_sha256=sha(folder/'broker.jsonl') if (folder/'broker.jsonl').exists() else None))
    if (paid>limits['paid_requests'] or local>limits['local_requests'] or
            paid+local>limits['total_requests']):
        raise ValueError('Campaign request caps already exceeded')
    budget=campaign/'shared-budget.json'
    if paid and not budget.exists():raise ValueError('Prior paid attempts lost their shared reservation ledger')
    if budget.exists():
        data=read(budget)
        if data.get('cap')!=limits['api_usd'] or not isinstance(data.get('charged'),(int,float)) or not math.isfinite(data['charged']) or not 0<=data['charged']<=limits['api_usd'] or data.get('halt_reason'):
            raise ValueError('Campaign shared budget mismatched, exhausted or halted')
    return dict(remaining_rows=[row for row in proposal['rows'] if row not in launched],remaining_paid_requests=limits['paid_requests']-paid,
        remaining_local_requests=limits['local_requests']-local,prior_retained_scratch_bytes=retained,prior_attempt_chain=chain,
        reuse_build_receipt=reuse,prior_launched_rows=sorted(launched),prior_outcomes=outcomes,prior_missing_rows=missing,
        prior_storage_reconciliation=storage_audits)


def _execute_locked(proposal, approval, output):
    if approval.get('status')!='approved' or approval.get('proposal_sha256')!=digest(proposal):
        raise ValueError('Exact proposal not approved; no scheduler query or subprocess started')
    validate(proposal)
    _, limits = resource_envelope(proposal)
    job=str(approval['job_id']);attempts=approval['attempt_job_ids']
    if not attempts or attempts[-1]!=job or len(set(attempts))!=len(attempts):raise ValueError('Complete immutable attempt list required')
    if proposal.get('resource_profile') == CPU_VALIDATION_PROFILE and attempts != [job]:
        raise ValueError('CPU validation allows one allocation attempt and no replay/resampling')
    if output!=Path(proposal['campaign_root'])/('attempt-'+job):raise ValueError('Attempt output must belong to immutable campaign')
    history=campaign_history(proposal,attempts[:-1],reconcile_node_local=True);prior=0
    if attempts[:-1]:
        text=subprocess.check_output(['sacct','-j',','.join(attempts[:-1]),'-X','--noheader','--parsable2','--format=JobID,State,ElapsedRaw'],text=True)
        rows=[line.split('|') for line in text.strip().splitlines()]
        terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE'}
        if (len(rows)!=len(attempts)-1 or {row[0] for row in rows}!=set(attempts[:-1]) or
                any(row[1].split()[0] not in terminal or not row[2].isdigit() for row in rows)):
            raise ValueError('Prior attempts not closed/accounted')
        prior=sum(int(row[2]) for row in rows)
    scheduler=subprocess.check_output(['scontrol','show','job','-o',job],text=True)
    scheduler_snapshot=None
    if proposal.get('resource_profile') == CPU_VALIDATION_PROFILE:
        scheduler_snapshot=subprocess.check_output(['scontrol','--json','show','job',job],text=True,timeout=15)
        approval=dict(approval,scheduler_epoch_proof=cpu_scheduler_epochs(scheduler_snapshot,job))
    context=approved_context(proposal,approval,scheduler,prior);context.update(history)
    output=checked_path(output,RUNTIME);output.mkdir(exist_ok=False)
    scratch=SCRATCH/('p'+job+'-'+secrets.token_hex(3));checked_path(scratch,SCRATCH);scratch.mkdir(parents=True,exist_ok=False)
    new_roots=len(history['remaining_rows'])+int(not history['reuse_build_receipt'])
    additional=proposal['original_root_apparent_bytes']*new_roots+limits['runtime_reserve_bytes']
    projected=history['prior_retained_scratch_bytes']+additional
    context.update(scratch=str(scratch),output=str(output),proposal=proposal,proposal_sha256=digest(proposal),
                   approval=approval,controller_sha256=sha(__file__),owner_pid=os.getpid(),
                   owner_start_ticks=_process(os.getpid())['start_ticks'])
    put(output/'execution-context.json',context)
    if scheduler_snapshot is not None:
        put(output/'scheduler-epochs.json',dict(raw_json=scheduler_snapshot,
            sha256=hashlib.sha256(scheduler_snapshot.encode()).hexdigest()))
    previous=_subreaper(True);baseline=set(_processes());process=None;tree=None;stopped=[False]
    handlers={sig:signal.signal(sig,lambda *_:stopped.__setitem__(0,True)) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        if projected>limits['scratch_bytes'] or shutil.disk_usage(scratch).free<additional:
            raise ValueError(f'Insufficient scratch: retained/projected={projected}, new={additional}, '
                f'free={shutil.disk_usage(scratch).free}, cap={limits["scratch_bytes"]}')
        process=multiprocessing.get_context('fork').Process(target=_runner,args=(proposal,context,str(output),str(scratch)))
        try:process.start()
        finally:
            if process.pid is not None:tree=OwnedEpisodeTree(process,baseline)
        while process.is_alive() and time.time()<context['deadline'] and not stopped[0]:
            tree.refresh();time.sleep(.5)
    finally:
        try:
            closure=tree.stop(10) if tree else dict(remaining_owned_processes={},child_exitcode=None)
            put(output/'controller-closure.json',closure)
            used=int(subprocess.check_output(['du','--apparent-size','-s','-B1',str(scratch)],text=True,timeout=60).split()[0])
            put(output/'storage-final.json',dict(scratch_path=str(scratch),apparent_bytes=used,
                cumulative_retained_bytes=used+history['prior_retained_scratch_bytes'],cap_bytes=limits['scratch_bytes']))
        finally:
            _subreaper(previous)
            for sig,handler in handlers.items():signal.signal(sig,handler)
    if closure['remaining_owned_processes'] or closure['child_exitcode']!=0 or not (output/'completed.json').exists():
        raise RuntimeError('Pilot incomplete; all attempts/unknown reservations retained, no automatic retry')


def cpu_scheduler_epochs(text, job):
    """Extract finite Slurm epochs without assuming the scheduler's timezone."""
    data=json.loads(text)
    if data.get('errors')!=[] or len(data.get('jobs',[]))!=1:
        raise ValueError('CPU validation requires one error-free scheduler JSON job')
    item=data['jobs'][0]
    if (str(item.get('job_id'))!=job or item.get('job_state')!=['RUNNING']
            or item.get('user_id')!=os.getuid() or item.get('restart_cnt')!=0):
        raise ValueError('CPU validation scheduler identity/restart count differs')
    values={}
    for name in ('start_time','end_time','time_limit'):
        value=item.get(name,{})
        if (value.get('set') is not True or value.get('infinite') is not False
                or type(value.get('number')) is not int or value['number']<=0):
            raise ValueError('CPU validation scheduler clock is not a finite set integer')
        values[name]=value['number']
    return dict(job_id=job,start_epoch=values['start_time'],end_epoch=values['end_time'],
        limit_seconds=60*values['time_limit'],source='scontrol_json_unix_epochs',
        snapshot_sha256=hashlib.sha256(text.encode()).hexdigest())


def execute(proposal,approval,output):
    if approval.get('status')!='approved' or approval.get('proposal_sha256')!=digest(proposal):
        raise ValueError('Exact proposal not approved; no scheduler query or subprocess started')
    validate(proposal)
    campaign=checked_path(proposal['campaign_root'],RUNTIME);campaign.mkdir(exist_ok=True)
    lock=checked_path(campaign/'controller.lock',RUNTIME)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _execute_locked(proposal,approval,Path(output))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--approval',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();proposal=read(args.proposal)
    if not args.execute:print(json.dumps(preview(proposal),indent=2));return
    if not args.approval or not args.output:parser.error('Execution requires exact approval and a fresh output directory')
    execute(proposal,read(args.approval),args.output)


if __name__=='__main__':main()
