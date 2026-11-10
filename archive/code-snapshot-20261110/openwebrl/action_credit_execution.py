"""Explicit production glue; preparation is CPU-only and grants no authority.

A fresh bundle binds per-arm native argv and retained outputs before PREPARED is
written. Physical execution requires real separate API stores, source-bound GPU
acceptance, all-attempt scheduler accounting and a measured storage forecast.
No scheduler submission, ledger creation, synthetic acceptance or cleanup occurs
here. Full Slime/Megatron argument validation runs only in the physical path.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import math
import os
import re
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from . import action_credit_bootstrap as boot
from .action_credit_accounting import AllocationAccountingProvider
from .action_credit_acceptance import audit_allocation
from .action_credit_eval_executor import Full300PhysicalExecutor, _validate_config
from .action_credit_evaluation import Full300EvaluationOwner, evaluation_protocol
from .action_credit_native_controller import _json, _write
from .action_credit_state import _safe_path
from .action_credit_storage import StorageAdmissionProvider
from .action_credit_verification import VerificationManifest, VerificationStore, approved_authority
from .action_credit_worker import verify_worker_environment


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def planned_ref(path, value):
    return dict(path=str(_safe_path(path)), sha256=hashlib.sha256(encoded(value)).hexdigest())


def _option(argv, key, value):
    """Replace exactly one existing value; never accept ambiguous duplicate argv."""
    require(argv.count(key) == 1, 'Exactly one native argument required: '+key)
    index = argv.index(key)
    require(index+1 < len(argv) and not argv[index+1].startswith('--'), 'Missing argument value: '+key)
    argv[index+1] = str(value)


def bind_bundles(bundles, *, native_argv_ref, native_config_ref):
    """Pure enrichment BEFORE write_bundles; preserve every scientific input.

    Existing immutable bundles must be regenerated at a fresh output root. This
    function neither edits those bundles nor creates ledgers/approvals. Its
    execution template intentionally lacks every physical/authority binding.
    """
    base_argv, base_config = _json(native_argv_ref), _json(native_config_ref)
    require(isinstance(base_argv, list) and all(isinstance(x, str) for x in base_argv), 'Native argv list required')
    require(isinstance(base_config, dict), 'Native custom configuration required')
    # build_bundles intentionally shares immutable common pins across arms.
    # Copy each bundle separately before binding mutable per-arm output paths.
    result = {arm: deepcopy(bundle) for arm, bundle in bundles.items()}
    for arm, bundle in result.items():
        files = bundle['files']; template = files['launch.template.json']
        root = _safe_path(template['store_root']).parent
        require(not root.exists(), 'Bind only a fresh preparation; never mutate an existing bundle')
        require(bundle['manifest'].to_dict()['arm_id'] == arm, 'Arm identity differs')
        require('execution-binding.json' not in files, 'Runtime already bound')
        paths = {key: str(root/name) for key, name in dict(
            save='native-checkpoints', generated='native-generated-samples', browser_logs='browser-logs',
            browser_locks='browser-locks', evaluation='evaluations', execution='executions',
            accounting='allocation-evidence', storage='storage-evidence').items()}
        # Native Ray and multiprocessing create Unix sockets below these roots.
        # Pin short private scratch explicitly; retain it across failed attempts.
        scratch = Path('/tmp')/('ace-'+bundle['manifest'].sha256[:16])
        paths.update(scratch=str(scratch), ray=str(scratch/'r'))
        validate_socket_paths(paths)
        files['environment.json']['TMPDIR'] = str(scratch)
        files['environment.json']['WANDB_RESUME'] = 'allow'
        config = deepcopy(base_config)
        config['path_to_save_generated_samples'] = paths['generated']
        argv = list(base_argv)
        _option(argv, '--save', paths['save'])
        _option(argv, '--custom-config-path', root/'native-config.json')
        require(not any(x.startswith('--wandb-') and x not in ('--wandb-project',) for x in argv),
                'Base argv already binds unexpected W&B settings')
        argv += ['--wandb-group', bundle['manifest'].to_dict()['campaign_id'],
                 '--wandb-dir', files['environment.json']['WANDB_DIR'], '--wandb-mode', 'online',
                 '--disable-wandb-random-suffix']
        # Paths do not alter browser observations, history or replay acceptance.
        for name in ('worker.common.json', 'runtime-pins.json'):
            browser = files[name]['browser_config']['local_process']
            browser['log_dir'], browser['port_lock_dir'] = paths['browser_logs'], paths['browser_locks']
        files['native-config.json'], files['native-argv.json'] = config, argv
        binding = dict(schema_version=1, arm_id=arm, manifest_sha256=bundle['manifest'].sha256,
            source_inventory=planned_ref(root/'source-inventory.json', files['source-inventory.json']),
            base_native_argv=native_argv_ref, base_native_config=native_config_ref,
            native_argv=planned_ref(root/'native-argv.json', argv),
            native_config=planned_ref(root/'native-config.json', config), paths=paths,
            full_native_parser='slime.utils.arguments.parse_args', cpu_parse_is_acceptance=False,
            original_iteration=0, physical_acceptance_required=True)
        files['execution-binding.json'] = binding
        files['execution.template.json'] = dict(schema_version=1, status='draft_not_executable',
            bundle=str(root), binding=planned_ref(root/'execution-binding.json', binding),
            launch=None, job_id=None, attempt_ledger=None, storage_forecast=None,
            verification_manifest=None, verification_store_root=None, shared_initial_root=None,
            shared0_worker_configuration=None, iter10_worker_configuration=None)
    return result


def validate_socket_paths(paths):
    scratch = _safe_path(paths['scratch'])
    require(scratch.parent == Path('/tmp') and re.fullmatch(r'ace-[0-9a-f]{16}', scratch.name),
            'Exact short private scratch root required')
    ray = _safe_path(paths['ray'])
    require(ray == scratch/'r', 'Ray scratch escaped its owned root')
    # Include a conservative ten-digit PID and longest native socket suffix.
    suffix = '/session_2026-10-11_23-59-59_999999_9999999999/sockets/plasma_store'
    require(len(os.fsencode(str(ray)+suffix)) < 107, 'Ray Unix socket path exceeds its physical limit')
    require(len(os.fsencode(str(scratch)+'/pymp-xxxxxxxx/listener-xxxxxxxx')) < 107,
            'Multiprocessing Unix socket path exceeds its physical limit')


def load_execution(reference):
    """No native import, scheduler query, storage reservation or paid dispatch."""
    request = _json(reference)
    require(request.get('status') == 'bound_for_execution', 'Draft execution cannot launch')
    expected = {'schema_version','status','bundle','binding','launch','job_id','attempt_ledger',
        'storage_forecast','verification_manifest','verification_store_root','shared_initial_root',
        'shared0_worker_configuration','iter10_worker_configuration'}
    require(set(request) == expected and request['schema_version'] == 1, 'Exact execution schema required')
    require(isinstance(request['job_id'], str) and request['job_id'].isdigit(), 'Actual parent job ID required')
    inputs = sorted(expected-{'schema_version','status','bundle','job_id','verification_store_root','shared_initial_root'})
    # Reject incomplete authority before touching any referenced files. Set
    # iteration order must not decide which missing input is reported first.
    for name in inputs:
        require(isinstance(request[name], dict), 'Missing immutable execution input: '+name)
    for name in inputs:
        _json(request[name])
    # This rejects draft/null API authority before any physical initialization.
    assembly = boot.assemble(request['bundle'], request['launch'])
    binding = _json(request['binding'])
    require(request['binding'] == boot.artifact(assembly.root/'execution-binding.json'), 'Binding is not the prepared runtime')
    require(binding['manifest_sha256'] == assembly.store.manifest.sha256
            and binding['source_inventory'] == assembly.source_reference, 'Binding changed manifest/source')
    require(binding['native_argv'] == boot.artifact(assembly.root/'native-argv.json')
            and binding['native_config'] == boot.artifact(assembly.root/'native-config.json'), 'Unprepared native argv/config')
    require(binding['arm_id'] == assembly.store.manifest.to_dict()['arm_id'], 'Binding changed arm')
    validate_socket_paths(binding['paths'])
    require(binding['paths']['scratch'] == '/tmp/ace-'+assembly.store.manifest.sha256[:16],
            'Scratch root belongs to another arm')
    for name, path in binding['paths'].items():
        if name not in ('scratch','ray'):
            require(_safe_path(path).parent == assembly.root, 'Output escaped its prepared arm: '+name)
    require(assembly.environment['TMPDIR'] == binding['paths']['scratch'], 'TMPDIR differs from socket-safe binding')
    require(binding['paths']['save'] == str(assembly.root/'native-checkpoints'), 'Native save binding changed')
    argv, config = _json(binding['native_argv']), _json(binding['native_config'])
    require(argv.count('--save') == 1 and argv[argv.index('--save')+1] == binding['paths']['save'], 'Native save argv differs')
    require(argv.count('--custom-config-path') == 1
            and argv[argv.index('--custom-config-path')+1] == binding['native_config']['path'], 'Native config argv differs')
    require(config['path_to_save_generated_samples'] == binding['paths']['generated'], 'Generated output path differs')
    for key, target in [('log_dir','browser_logs'),('port_lock_dir','browser_locks')]:
        require(assembly.worker['browser_config']['local_process'][key] == binding['paths'][target], 'Browser output path differs')
    boot.verify_native_runtime_sources(assembly.source)
    boot.validate_native_acceptance(assembly)
    acceptance = _json(assembly.launch.native_acceptance)
    require(acceptance.get('execution_binding') == request['binding'], 'Physical acceptance did not bind actual argv/output paths')
    verification = VerificationManifest.from_dict(_json(request['verification_manifest']))
    approved_authority(verification)
    v = verification.to_dict()
    require(v['source_inventory']['sha256'] == assembly.source_reference['sha256']
            and _json(v['source_inventory']) == assembly.source
            and v['original_actor_inventory'] == assembly.store.manifest.to_dict()['science']['checkpoint']['original_actor_inventory'],
            'Shared verification changed source or original actor')
    verification_store = VerificationStore(request['verification_store_root'], verification)
    require(verification_store.root != assembly.store.root, 'Shared verification cannot borrow the pilot store')
    configurations = {verification.sha256: request['shared0_worker_configuration'],
                      assembly.store.manifest.sha256: request['iter10_worker_configuration']}
    for store in (verification_store, assembly.store):
        fake_request = dict(accounting_manifest_sha256=store.manifest.sha256, protocol=evaluation_protocol(),
            diagnostic_only=False, task_manifest=v['evaluation_task_manifest'])
        _validate_config(store, fake_request, _json(configurations[store.manifest.sha256]))
    _safe_path(request['shared_initial_root'])
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', assembly.launch.execution_id),
            'Execution identity must be a single safe directory component')
    require(time.time() < assembly.launch.deadline_unix, 'Execution deadline expired')
    return request, assembly, binding, verification_store, configurations


def scheduler_deadline(text, job_id):
    """Parse the actual UTC Slurm end time and exact single-node allocation."""
    fields = dict(item.split('=', 1) for item in shlex.split(text) if '=' in item)
    require(fields.get('JobId') == job_id and fields.get('JobState') == 'RUNNING', 'Current scheduler allocation is not running')
    tres = dict(item.split('=', 1) for item in fields.get('AllocTRES', fields.get('TRES','')).split(',') if '=' in item)
    require(fields.get('NumNodes') == '1' and fields.get('NumCPUs') == '64'
            and tres.get('gres/gpu') == '8' and tres.get('mem') in ('960G','983040M'), 'Current resources differ from approved 8GPU/64CPU/960GiB')
    require(tres.get('gres/gpu:h200') == '8', 'Current allocation must contain eight H200s')
    try:
        return datetime.strptime(fields['EndTime'], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
    except (KeyError, ValueError) as exc:
        raise ValueError('Finite actual scheduler end time required') from exc


class AllocationBoundary:
    """Fresh accounting plus actual end time; no authority creation or time reset."""
    def __init__(self, provider, *, manifest, job_id, deadline_unix, output_root):
        self.provider, self.manifest, self.job_id = provider, manifest, job_id
        self.deadline, self.root = deadline_unix, _safe_path(output_root)

    def __call__(self):
        ref = self.provider(); value = _json(ref)
        audit = audit_allocation(ref, self.manifest)
        running = [r['job_id'] for r in value['attempts'] if r['state'] == 'RUNNING']
        require(running == [self.job_id], 'Exactly this one active attempt must own the pilot')
        command = ['scontrol','show','job','-o',self.job_id]
        process = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False,
            env=dict(os.environ, TZ='UTC'))
        folder = self.root/uuid.uuid4().hex; folder.mkdir(parents=True, exist_ok=False)
        _write(folder/'scheduler-deadline.json', dict(command=command, observed_unix=time.time(),
            returncode=process.returncode, stdout=process.stdout, stderr=process.stderr, accounting=ref))
        require(process.returncode == 0, 'Live deadline query failed')
        end = scheduler_deadline(process.stdout, self.job_id)
        # Account for elapsed since sacct observation, including both queries.
        cap_end = audit['observed_unix'] + (audit['approved_gpu_seconds_cap']-audit['gpu_seconds_used'])/8 - 1
        require(time.time() < self.deadline <= min(end, cap_end), 'Launch deadline exceeds current allocation or remaining all-attempt cap')
        return ref


def build_owners(request, assembly, binding, verification_store, configurations):
    """Concrete providers only; no no-op storage/eval or imported callbacks."""
    provider = AllocationAccountingProvider(manifest=assembly.store.manifest,
        attempt_ledger_ref=request['attempt_ledger'], output_root=binding['paths']['accounting'])
    allocation = AllocationBoundary(provider, manifest=assembly.store.manifest, job_id=request['job_id'],
        deadline_unix=assembly.launch.deadline_unix, output_root=binding['paths']['accounting'])
    storage = StorageAdmissionProvider(forecast_ref=request['storage_forecast'],
        output_root=binding['paths']['storage'], job_id=request['job_id'])
    v = verification_store.manifest.to_dict()
    executor = Full300PhysicalExecutor(worker_configurations=configurations, python=sys.executable)
    evaluation = Full300EvaluationOwner(task_manifest_ref=v['evaluation_task_manifest'],
        original_actor_inventory_ref=v['original_actor_inventory'], source_inventory_ref=assembly.source_reference,
        shared_initial_root=request['shared_initial_root'], arm_output_root=binding['paths']['evaluation'],
        verification_store=verification_store, executor=executor)
    return allocation, storage, evaluation


def validate_storage_coverage(request, assembly, binding, verification, configs, storage):
    """Every retained writer, including shared0 and scratch, belongs in forecast."""
    forecast, roots, *_ = storage._forecast(assembly.store)
    require(forecast['resource_accounting_ref'] == boot.artifact(assembly.root/'resource-accounting.json'),
            'Storage forecast did not bind the prepared accounting')
    writers = [assembly.root, verification.root, _safe_path(request['shared_initial_root']),
               _safe_path(binding['paths']['scratch'])]
    for ref in configs.values():
        config = _json(ref)
        writers += [_safe_path(config['browser_config']['local_process'][key]) for key in ('log_dir','port_lock_dir')]
        for key, value in config['environment'].items():
            if key.endswith(('_DIR','_HOME','_CACHE')) or key in ('TMPDIR','MPLCONFIGDIR','CUDA_CACHE_PATH','FLASHINFER_WORKSPACE_BASE'):
                writers.append(_safe_path(value))
    require(all(any(path == root or root in path.parents for root in roots) for path in writers),
            'Measured forecast omits arm/shared verification/evaluation/scratch outputs')


def apply_environment(environment):
    require(not {'HOME','CODEX_HOME'} & set(environment), 'System home overrides forbidden')
    # These legacy controls can bypass the new controller or substitute data.
    forbidden = ('OPENWEBRL_REPLAY_', 'OPENWEBRL_VERIFY_', 'OPENWEBRL_ACTION_CREDIT_NATIVE_GATE',
                 'OPENWEBRL_TRAINING_', 'OPENWEBRL_RESUME_', 'OPENWEBRL_ARM_')
    require(not any(key.startswith(forbidden) for key in os.environ), 'Unpinned legacy training overrides present')
    for key, value in environment.items():
        if key.endswith(('_DIR','_HOME','_CACHE')) or key in ('TMPDIR','MPLCONFIGDIR','CUDA_CACHE_PATH','FLASHINFER_WORKSPACE_BASE'):
            _safe_path(value).mkdir(parents=True, exist_ok=True)
        os.environ[key] = value
    verify_worker_environment()
    tempfile.tempdir = None  # A prior read-only inspection may have cached /tmp.


def full_native_parse(argv):
    """Physical-only: includes actual Megatron hardware validation, no mocks."""
    from slime.utils.arguments import parse_args
    saved = sys.argv
    try:
        sys.argv = ['action-credit-production']+list(argv)
        return parse_args()
    finally:
        sys.argv = saved


@contextmanager
def deadline_scope(deadline):
    require(math.isfinite(deadline) and deadline > time.time(), 'Expired physical execution deadline')
    require(signal.getitimer(signal.ITIMER_REAL) == (0.,0.), 'Another execution timer already owns this process')
    prior = {key: signal.getsignal(key) for key in (signal.SIGALRM, signal.SIGTERM)}
    def expired(signum, frame):
        raise TimeoutError('Owned action-credit execution deadline/termination reached')
    try:
        for key in prior: signal.signal(key, expired)
        signal.setitimer(signal.ITIMER_REAL, deadline-time.time())
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for key, handler in prior.items(): signal.signal(key, handler)


def execute(reference):
    request, assembly, binding, verification, configs = load_execution(reference)
    require(os.environ.get('SLURM_JOB_ID') == request['job_id'], 'Execute only inside its registered allocation')
    require(Path(assembly.source['source_root']).resolve() == Path(__file__).resolve().parents[1], 'Execute the exact immutable source fork')
    # One OS lock owns the complete mutable training lineage through shutdown.
    with _safe_path(assembly.root/'execution.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another execution already owns this pilot') from exc
        return _execute_owned(reference, request, assembly, binding, verification, configs)


def _execute_owned(reference, request, assembly, binding, verification, configs):
    allocation, storage, evaluation = build_owners(request, assembly, binding, verification, configs)
    allocation()
    _safe_path(binding['paths']['scratch']).mkdir(mode=0o700, parents=True, exist_ok=True)
    validate_storage_coverage(request, assembly, binding, verification, configs, storage)
    # Measured, coordinated forward admission precedes even physical parsing.
    storage(assembly.store, 'native-startup')
    apply_environment(assembly.environment)
    with deadline_scope(assembly.launch.deadline_unix):
        boot.verify_native_runtime_sources(assembly.source)
        args = full_native_parse(_json(binding['native_argv']))
        boot.validate_native_profile(args, assembly)
        require(args.save == binding['paths']['save']
                and args.path_to_save_generated_samples == binding['paths']['generated'], 'Parsed native output binding differs')
        import ray
        require(not ray.is_initialized(), 'Execution must own a fresh isolated Ray runtime')
        ray_root = _safe_path(binding['paths']['ray']); ray_root.mkdir(parents=True, exist_ok=True)
        try:
            ray.init(address='local', num_gpus=8, num_cpus=64, include_dashboard=False,
                namespace=assembly.launch.execution_id, _temp_dir=str(ray_root),
                runtime_env=dict(env_vars=dict(assembly.environment)))
            # train() awaits shared0, all ten updates, full300iter10 and native
            # worker/actor/process shutdown; it never marks independent review.
            result = boot.train(args, assembly, evaluation_owner=evaluation,
                allocation_accounting=allocation, storage_admission=storage)
        finally:
            ray.shutdown()
    output = _safe_path(binding['paths']['execution'])/assembly.launch.execution_id
    output.mkdir(parents=True, exist_ok=False)
    return _write(output/'result.json', dict(execution_input=reference, result=result,
        verified_complete=False, independent_final_artifact_review_required=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inspect-execution', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--execution', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    options = parser.parse_args(argv)
    ref = dict(path=str(options.execution), sha256=options.sha256)
    if options.inspect_execution:
        _, assembly, _, _, _ = load_execution(ref)
        print(json.dumps(dict(manifest_sha256=assembly.store.manifest.sha256,
            static_bindings_validated=True, physical_execution_performed=False, launch_ready=False)))
    else:
        print(json.dumps(execute(ref)))


if __name__ == '__main__':
    main()
