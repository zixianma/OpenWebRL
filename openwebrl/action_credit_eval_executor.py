"""Physical full300 executor inside an already owned native allocation.

Import is inert. Production entry points require pinned source/configuration,
exact billing approval, the real native actor handles and the existing launch
deadline. No scheduler calls, cleanup of unrelated processes or implicit retries.
CPU tests do not establish GPU restoration, browser readiness or API authority.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import FunctionType, SimpleNamespace
from urllib.parse import urlsplit

from .action_credit import digest
from .action_credit_campaign import _artifact_file
from .action_credit_evaluation import (
    PINNED_FULL300_SHA256, _checkpoint, _ids, _native_status, _sources,
    evaluation_protocol, native_evaluation_controls,
)
from .action_credit_judge import JudgeHTTPProtocol, MeteredJudgeClient, PaidDispatchBlocked
from .action_credit_native_controller import _hash, _json
from .action_credit_process import (
    _CHILD_REGISTRY, _no_credentials, _private_file, _write,
    detached_process_scope, owned_process,
)
from .action_credit_state import _safe_path
from .action_credit_worker import _approval, served_weight_version, verify_worker_environment


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path):
    path = _safe_path(path)
    return dict(path=str(path), sha256=_hash(path))


def saved_json(path, value):
    path = _safe_path(path)
    if path.exists():
        result = reference(path)
        require(_json(result) == json.loads(json.dumps(value)), 'Retained executor artifact changed')
        return result
    return _write(path, value)


def _deadline(value):
    require(type(value) in (int, float) and time.time() < value,
            'Evaluation allocation deadline expired')
    return value - time.time()


def _endpoint(value):
    parsed = urlsplit(value)
    require(parsed.scheme == 'http' and parsed.hostname and parsed.port is not None
            and parsed.path == '/generate' and not parsed.query and not parsed.fragment
            and not parsed.username and not parsed.password, 'Exact direct local engine endpoint required')
    return parsed


def _validate_sources(request):
    _sources(request['source_inventory'])
    inventory = _json(request['source_inventory'])
    root = _safe_path(inventory['source_root'])
    executing = Path(__file__).resolve().parents[1]
    for name in ('openwebrl/action_credit_eval_executor.py', 'openwebrl/action_credit_process.py',
                 'openwebrl/action_credit_judge.py', 'openwebrl/action_credit_eval_budget.py',
                 'openwebrl/action_credit_driver.py', 'openwebrl/action_credit_native_controller.py',
                 'openwebrl/env/local_process_env.py', 'openwebrl/docker/env_server.py'):
        require(name in inventory['files'] and _hash(root / name) == inventory['files'][name]
                and _hash(executing / name) == inventory['files'][name],
                'Unpinned physical evaluation source: ' + name)
    return root


def _validate_config(store, request, config):
    """Configuration is an exact separately retained production input."""
    _no_credentials(config)
    require(config.get('schema_version') == 1 and config.get('accounting_manifest_sha256') == store.manifest.sha256,
            'Worker configuration belongs to another billing store')
    require(request['accounting_manifest_sha256'] == store.manifest.sha256,
            'Cohort billing identity changed')
    require(config.get('protocol') == request['protocol'] == evaluation_protocol(), 'Evaluation protocol changed')
    require(request['diagnostic_only'] is False and request['task_manifest']['sha256'] == PINNED_FULL300_SHA256,
            'Physical executor rejects diagnostic or unpinned task manifests')
    protocol = JudgeHTTPProtocol(**config['judge_protocol'])
    _approval(store, config['approval_reference'], protocol)
    science = store.manifest.to_dict()['science']['judge']
    require(protocol.model == 'gpt-4.1' and protocol.sha256 == science['http_protocol_sha256'],
            'Physical judge protocol differs from billing authority')
    require(config.get('canonical_reward_source_sha256') == science['canonical_reward_source_sha256']
            == _hash(Path(__file__).parent / 'reward_browser.py'), 'Canonical native reward source changed')
    browser = config['browser_config']
    require(browser.get('mode') == 'local_process'
            and browser.get('local_process', {}).get('server_module', 'openwebrl.docker.env_server') == 'openwebrl.docker.env_server',
            'Standard native local browser protocol required')
    for key, value in dict(width=1280, height=720, dpr=1, use_screenshot=True,
                           use_a11ytree=False, resize_output_coords=True, resize_scale=1000).items():
        require(browser.get(key) == value, 'Evaluation browser observation/coordinate protocol changed: ' + key)
    for key in ('log_dir', 'port_lock_dir'):
        _safe_path(browser['local_process'][key])
    args = config['native_args']
    for key, value in dict(max_steps=30, inference_step_timeout_secs=120., judge_timeout_secs=120.,
                           judge_api_model='gpt-4.1', judge_prompt_variant='action_history',
                           browser_response_format_mode='browser_env').items():
        require(args.get(key) == value, 'Evaluation native argument changed: ' + key)
    validate_native_evaluation_controls(args)
    require(args.get('browser_action_selector') is None, 'Cannot replace another action selector')
    require(type(config.get('concurrency')) is int and 1 <= config['concurrency'] <= 64,
            'Explicit bounded worker concurrency required')
    # Cache paths are checked again inside every child before imports initialize
    # model/browser SDKs. Caller environment must be the frozen writable map.
    environment = config['environment']
    require(not {'HOME','CODEX_HOME'} & set(environment), 'Do not repurpose system home variables')
    for key, value in environment.items():
        require(isinstance(key, str) and isinstance(value, str), 'Environment must be string-valued')
        if key.endswith(('_DIR', '_HOME')) or key in ('TMPDIR', 'TORCH_EXTENSIONS_DIR', 'FLASHINFER_WORKSPACE_BASE', 'CUDA_CACHE_PATH'):
            _safe_path(value)
    require(environment.get('PYTHONDONTWRITEBYTECODE') == '1', 'Worker bytecode writes prohibited')
    return protocol


def validate_native_evaluation_controls(args):
    for key,value in native_evaluation_controls().items():
        require(key in args and args[key] == value, 'Evaluation history/context protocol changed: ' + key)


class TransportJournal:
    """Before-dispatch request and durable full response/error inventory."""
    def __init__(self, root):
        self.root = _safe_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.http = {}
        self.actor = {}

    def __call__(self, event, value):
        key = value.get('request_id')
        if event == 'judge_http_reserved':
            self.http[key] = dict(request_id=key, request=saved_json(self.root / key / 'request.json', value['payload']))
        elif event == 'judge_http_response':
            self.http[key]['response'] = saved_json(self.root / key / 'response.json', value['response'])
        elif event == 'judge_http_error':
            self.http[key]['error'] = saved_json(self.root / key / 'error.json', dict(error_type=value['error_type']))
        else:
            saved_json(self.root / 'events' / (event + '-' + digest(value) + '.json'), value)

    def inventory(self):
        return dict(http_attempts=list(self.http.values()), actor_attempts=list(self.actor.values()))


class EvaluationActor:
    """One HTTP attempt per turn, fixed policy, no hidden SDK/router retries."""
    def __init__(self, *, store, request, attempt_id, endpoint, version, journal, deadline_unix):
        self.store, self.request, self.attempt = store, request, attempt_id
        self.endpoint, self.version, self.journal, self.deadline = endpoint, version, journal, deadline_unix
        _endpoint(endpoint)
        self.next_turn = 0
        self.failure = None

    async def __call__(self, **kwargs):
        try:
            return await self._dispatch(**kwargs)
        except BaseException as exc:
            self.failure = type(exc).__name__
            raise

    async def _dispatch(self, *, url, input_text, sampling_params, images, task_id, turn, timeout, **unused):
        require(url == self.endpoint and task_id == self.request['task_id'] and turn == self.next_turn
                and 0 <= turn < 30, 'Actor task/turn/endpoint identity changed')
        frozen = dict(temperature=0., top_p=1., top_k=1, max_new_tokens=4096)
        require(sampling_params == frozen, 'Actual evaluation sampling differs from frozen T0 protocol')
        _deadline(self.deadline)
        require(await served_weight_version(url) == self.version, 'Served policy version changed before dispatch')
        payload = dict(text=input_text, sampling_params=dict(sampling_params), return_logprob=True)
        if images:
            payload['image_data'] = images
        key = digest([self.attempt, turn, 'evaluation_actor'])
        admission = self.store.reserve_evaluation_actor_request(key, parent_attempt_id=self.attempt,
            task_id=task_id, policy_sha256=self.request['policy_sha256'], payload_sha256=digest(payload), turn=turn)
        require(admission['created'], 'Persisted actor request requires reconciliation, never redispatch')
        row = dict(request_id=key, request=saved_json(self.journal.root / key / 'request.json', payload))
        self.journal.actor[key] = row
        try:
            import httpx
            async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(retries=0),
                    timeout=min(float(timeout or 120.), _deadline(self.deadline)), follow_redirects=False,
                    trust_env=False) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                raw = response.json()
            row['response'] = saved_json(self.journal.root / key / 'response.json', raw)
            self.store.settle_evaluation_actor_request(key, status='received', response_sha256=digest(raw))
            require(await served_weight_version(url) == self.version, 'Served policy version changed during dispatch')
            output = native_actor_output(raw)
            self.next_turn += 1
            return output, dict(evaluation_actor_request_id=key)
        except BaseException as exc:
            self.failure = type(exc).__name__
            row['error'] = saved_json(self.journal.root / key / 'error.json', dict(error_type=self.failure))
            call = self.store.snapshot()['online_actor_requests'][key]
            if call['status'] == 'reserved':
                self.store.settle_evaluation_actor_request(key, status='unknown', reason=self.failure)
            raise


def native_actor_output(raw):
    """Use the native sampler's suffix convention, without hidden inference."""
    import math
    meta = raw.get('meta_info', {})
    values = meta.get('output_token_logprobs')
    require(isinstance(raw.get('text'), str) and isinstance(values, list) and values
            and len(values) <= 4096, 'Actor omitted bounded native response tokens')
    require(all(isinstance(v, (list, tuple)) and len(v) >= 2 and type(v[1]) is int and v[1] >= 0
                and type(v[0]) in (float, int) and math.isfinite(v[0]) and v[0] <= 1e-7 for v in values),
            'Actor response token/logprob payload invalid')
    finish = meta.get('finish_reason', {}).get('type', 'stop')
    require(finish in ('stop', 'length'), 'Actor aborted or returned an unknown finish reason')
    from .generate_browser import _ensure_im_end_w_new_line
    tokens, probs = [v[1] for v in values], [v[0] for v in values]
    text = _ensure_im_end_w_new_line(raw['text'], tokens, probs)
    return text, tokens, probs, finish


async def _native_reward(args, turns, client, journal, expected_source):
    """Clone only transport acquisition; preserve native status/format gating."""
    from . import reward_browser
    require(_hash(Path(reward_browser.__file__)) == expected_source, 'Canonical reward source changed')
    namespace = dict(reward_browser.__dict__)
    namespace['_get_openai_client'] = lambda **_: client
    namespace['_save_judge_trace'] = lambda args, sample, **record: journal('canonical_judge_verdict', record)
    for name in ('compute_judge_reward_actionhistory', '_score_single_sample', 'reward_func'):
        fn = getattr(reward_browser, name)
        clone = FunctionType(fn.__code__, namespace, name, fn.__defaults__, fn.__closure__)
        clone.__kwdefaults__ = fn.__kwdefaults__
        namespace[name] = clone
    rewards = await namespace['reward_func'](args, turns)
    for turn, reward in zip(turns, rewards, strict=True):
        turn.reward = reward


def _settle_row(store, row):
    verdict = _json(row['verdict'])
    state = store.snapshot()
    attempt = state['evaluation_attempts'][row['attempt_id']]
    cohort = state['evaluation_cohorts'][attempt['cohort_id']]
    _artifact_file(row['rollout']); record = _json(row['record'])
    require(row['task_id'] == attempt['task_id'] == verdict['task_id'] == record['task_id']
        and verdict['request_sha256'] == cohort['request_ref']['sha256']
        and verdict['rollout_sha256'] == row['rollout']['sha256']
        and record['rollout_file'] == Path(row['rollout']['path']).name
        and verdict['judge_model'] == record['judge_model'] == 'gpt-4.1'
        and verdict['judge_prompt_variant'] == record['judge_prompt_variant'] == 'action_history',
        'Saved row does not belong to its registered task/cohort/rollout')
    require(verdict['status'] in ('valid','invalid') and (verdict['status'] == 'valid'
        or verdict.get('reward') is None), 'Invalid native outcome cannot acquire a reward')
    outcome = dict(artifact_sha256=row['verdict']['sha256'])
    if verdict['status'] == 'valid':
        outcome['reward'] = verdict['reward']
    store.finish_evaluation_attempt(row['attempt_id'], status=verdict['status'], outcome=outcome,
        reason=None if verdict['status'] == 'valid' else verdict['invalid_reason'])


async def execute_task(packet):
    """Runs only in a dedicated owned process, after dispatch reservation."""
    from .action_credit_verification import open_accounting_store
    from .action_credit_judge_budget import gpt41_context_upper_bound, input_bound_protocol_sha256
    verify_worker_environment()
    from . import generate_browser as generation
    from .eval_monitor import save_task
    from slime.utils.types import Sample
    store = open_accounting_store(packet['billing_root'], _json(packet['billing_manifest']))
    request, config = _json(packet['request']), _json(packet['configuration'])
    _validate_sources(request)
    protocol = _validate_config(store, request, config)
    verify_worker_environment()
    require(time.time() < packet['deadline_unix'], 'Allocation expired before browser dispatch')
    task = packet['task_id']; attempt = packet['attempt_id']
    root = _safe_path(packet['attempt_root'])
    require(root in _safe_path(packet['packet_path']).parents, 'Worker packet escaped its attempt root')
    actual = store.snapshot()['evaluation_attempts'][attempt]
    require(actual['status'] == 'active' and actual['cohort_id'] == request['cohort_id'] and actual['task_id'] == task,
            'Browser attempt is not the admitted exact cohort/task')
    require(not any(c.get('attempt_id') == attempt for c in store.snapshot().get('online_actor_requests', {}).values())
            and not any(c['parent_attempt_id'] == attempt for c in store.snapshot().get('subrequests', {}).values()),
            'Persisted task transport requires reconciliation, never redispatch')
    rows = [json.loads(line) for line in _artifact_file(request['task_manifest']).read_text().splitlines() if line.strip()]
    row = next(r for r in rows if str(r['metadata']['task_id']) == task)
    sample = Sample(prompt=deepcopy(row['prompt']), metadata=deepcopy(row['metadata']), index=attempt)
    require(digest(dict(prompt=sample.prompt, metadata=sample.metadata)) == packet['sample_sha256'], 'Task bytes changed')
    journal = TransportJournal(root / 'transport')
    selector = EvaluationActor(store=store, request=dict(task_id=task, policy_sha256=request['policy_sha256']),
        attempt_id=attempt, endpoint=packet['endpoint'], version=packet['weight_version'], journal=journal,
        deadline_unix=packet['deadline_unix'])
    endpoint = _endpoint(packet['endpoint'])
    args = SimpleNamespace(**dict(config['native_args'], sglang_router_ip=endpoint.hostname,
        sglang_router_port=endpoint.port, browser_action_selector=selector,
        path_to_save_generated_samples=str(root / 'native_rollouts')))
    browser = deepcopy(config['browser_config'])
    # Native configuration is frozen. Files for individual browser launches are
    # already named uniquely by the local-process transport inside these roots.
    original_create = generation._create_env
    original_resources = generation._load_local_resources
    original_override = generation._apply_local_process_env_overrides
    clients = []
    class Client:
        def __init__(self, client):
            self.client, self.closing = client, None
        def __getattr__(self, key): return getattr(self.client, key)
        async def exit(self):
            if self.closing is None:
                self.closing = asyncio.create_task(self.client.exit())
            await asyncio.shield(self.closing)
    async def create(*args_, **kwargs):
        require(args_[0] == task and args_[1] == browser, 'Native browser task/config changed')
        result, task_data = await original_create(*args_, **kwargs)
        client = Client(result); clients.append(client)
        return client, task_data
    def resources(*args_, **kwargs):
        loaded = original_resources(*args_, **kwargs)
        require([digest(x) for x in loaded] == packet['native_resource_sha256'],
                'Task/tools/environment resources differ from pinned predispatch inputs')
        return loaded
    generation._create_env = create
    generation._load_local_resources = resources
    generation._apply_local_process_env_overrides = lambda _: deepcopy(browser)
    turns, error = [], None
    try:
        turns = await generation.generate_turn_sample(args, sample,
            dict(temperature=0., top_p=1., top_k=1, max_new_tokens=4096))
        require(turns, 'Missing native trajectory requires retained evidence reconciliation')
        # Native infrastructure failures may return the original ABORTED Sample
        # without an actor response. Keep that exact invalid disposition.
        require(selector.failure is None,
                'Native generator swallowed an actor identity/transport failure; reconcile retained artifacts')
        if not any(t.status == Sample.Status.ABORTED for t in turns):
            key = os.environ.get('OPENAI_API_KEY') or os.environ.get('JUDGE_API_KEY')
            if not key: raise PaidDispatchBlocked('Approved judge credential not configured')
            import httpx
            from openai import AsyncOpenAI
            async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(retries=0), timeout=120,
                                        follow_redirects=False, trust_env=False) as http:
                async with AsyncOpenAI(api_key=key, base_url='https://api.openai.com/v1',
                                       max_retries=0, http_client=http) as provider:
                    client = MeteredJudgeClient(store=store, parent_attempt_id=attempt, protocol=protocol,
                        approval_receipt=_approval(store, config['approval_reference'], protocol),
                        provider_create=provider.chat.completions.create, provider_max_retries=0,
                        input_token_upper_bound=gpt41_context_upper_bound,
                        input_bound_protocol_sha256=input_bound_protocol_sha256(), journal=journal)
                    await _native_reward(args, turns, client, journal, config['canonical_reward_source_sha256'])
    except BaseException as exc:
        error = type(exc).__name__
        saved_json(root / 'failure.json', dict(request_sha256=packet['request']['sha256'], task_id=task,
            attempt_id=attempt, reason=error))
        raise
    finally:
        try:
            for client in clients:
                await client.exit()
        finally:
            generation._create_env, generation._apply_local_process_env_overrides = original_create, original_override
            generation._load_local_resources = original_resources
            saved_json(root / 'transport-inventory.json', journal.inventory())
            # Preserve even partial/invalid/error attempts; never manufacture a
            # terminal success if upstream swallowed an infrastructure failure.
            save_task(args, root / 'rollout', task, turns, error)
    stem = digest_task(task)
    rollout = reference(root / 'rollout' / (stem + '.pt'))
    record = reference(root / 'rollout' / (stem + '.json'))
    valid, score = _native_status([t.to_dict() for t in turns])
    http_rows = journal.inventory()['http_attempts']
    verdict = saved_json(root / 'verdict.json', dict(task_id=task, request_sha256=packet['request']['sha256'],
        rollout_sha256=rollout['sha256'], judge_model='gpt-4.1', judge_prompt_variant='action_history',
        status='valid' if valid else 'invalid', reward=score,
        invalid_reason=None if valid else 'native_invalid_trajectory',
        http_request_ids=[r['request_id'] for r in http_rows],
        http_responses=[r['response'] for r in http_rows if 'response' in r]))
    result = dict(task_id=task, attempt_id=attempt, rollout=rollout, record=record, verdict=verdict)
    if len(turns) == 1 and turns[0].response_length == 0 and not turns[0].response and not turns[0].tokens:
        require(turns[0].status == Sample.Status.ABORTED and not valid, 'Empty response must remain invalid')
        require(not any(v['status'] == 'received' for v in store.snapshot().get('online_actor_requests', {}).values()
                        if v.get('attempt_id') == attempt), 'Empty placeholder cannot conceal a received response')
        result['no_response'] = saved_json(root / 'no-response.json', dict(
            request_sha256=packet['request']['sha256'], task_id=task, attempt_id=attempt,
            rollout_sha256=rollout['sha256'], reason=turns[0].metadata.get('terminate_reason'),
            generation_source_sha256=_hash(Path(generation.__file__)), sample_sha256=packet['sample_sha256']))
    # Persist first; a parent can complete an interrupted ledger commit without
    # any browser or provider request. Parent additionally proves child cleanup.
    saved_json(root / 'row.json', result)
    _settle_row(store, result)
    return result


def digest_task(task):
    import hashlib
    return hashlib.sha256(task.encode()).hexdigest()


async def _restore_and_serve(request, context, root):
    """Restore model/Adam/scheduler/RNG and await the real eight-rank transfer."""
    from .action_credit_driver import _call_all, _contract, _owned, _real_handlers, _unanimous
    launch = context['launch']; launch.validate(); _deadline(launch.deadline_unix)
    contract = replace(_contract(launch.initial_contract), update_id=request['iteration'],
                       policy_sha256=request['policy_sha256'])
    handles = _real_handlers(context['actor_model'])
    require(len(handles) == 8, 'Full eight-rank native actor handles required')
    common = dict(manifest=launch.manifest, store_root=launch.store_root, native_acceptance=launch.native_acceptance)
    if request['iteration'] == 0:
        from .action_credit_state import CampaignManifest, CampaignStore
        pilot = CampaignStore(launch.store_root, CampaignManifest.from_dict(_json(launch.manifest)))
        require(not any(v.get('phase') == 'consumed' for v in pilot.snapshot()['updates'].values()),
                'Shared initial evaluation cannot rewind an already trained pilot')
        # The pilot initialization is independent of the verification billing
        # ledger. initialize() refuses nonfresh or mismatched native initial state.
        inventory = _unanimous(await _call_all(handles, 'action_credit_operation',
            dict(common, operation='initialize', contract=asdict(contract))))
    else:
        from .action_credit_state import CampaignManifest, CampaignStore
        pilot = CampaignStore(launch.store_root, CampaignManifest.from_dict(_json(launch.manifest)))
        state = pilot.snapshot()
        require(state['native_completed_updates']['9']['checkpoint_inventory'] == request['checkpoint_inventory']
            and state['updates']['9']['phase'] == 'consumed'
            and state['updates']['9']['training_receipt']['new_policy_sha256'] == request['policy_sha256'],
            'Evaluation must serve the committed current iteration10 checkpoint')
        require(not any(int(k) >= 10 and v.get('phase') == 'consumed' for k,v in state['updates'].items()),
                'Evaluation cannot restore an older policy into a later training lineage')
        inventory = request['checkpoint_inventory']
    restored = _unanimous(await _call_all(handles, 'action_credit_driver_operation',
        dict(common, operation='restore_for_transfer', contract=asdict(contract), checkpoint_inventory=inventory)))
    require(restored['policy_sha256'] == request['policy_sha256'] and restored['update_id'] == request['iteration'],
            'Restored checkpoint policy differs')
    _deadline(launch.deadline_unix)
    await _call_all(handles, 'update_weights')
    engines, _, _, _, _ = await context['rollout_manager'].get_updatable_engines_and_lock.remote()
    urls = await _owned([engine.get_url.remote() for engine in engines])
    require(len(urls) == 4 and all(isinstance(url, str) and url for url in urls), 'Exactly four local TP2 engines required')
    urls = tuple(sorted(url.rstrip('/') + '/generate' for url in urls))
    require(len(set(urls)) == 4, 'Inference endpoints must be unique')
    for url in urls: _endpoint(url)
    versions = await _owned([served_weight_version(url) for url in urls])
    version = str(request['iteration'] + 1)
    require(all(v == version for v in versions), 'Served engines did not acknowledge actual restored policy version')
    # This record follows actual unanimous model/Adam/scheduler/RNG restore RPCs
    # and transfer/version queries. A marker alone is never accepted as restore.
    line = 'all8 native restore and transfer verified policy=' + request['policy_sha256'] + ' version=' + version
    log = root / 'native-restore.log'
    if not log.exists():
        with log.open('x') as stream:
            stream.write(line + '\n'); stream.flush(); os.fsync(stream.fileno())
    proof = saved_json(root / 'restore.json', dict(verification_kind='production_runtime', restored_on_gpu=True,
        policy_sha256=request['policy_sha256'], checkpoint_inventory=request['checkpoint_inventory'],
        native_checkpoint_inventory=inventory, source_inventory=request['source_inventory'],
        restore_rpc_result=restored, transfer_rank_count=8, endpoints=list(urls), weight_version=version,
        worker_log=reference(log), restore_line=line))
    return urls, version, proof


class Full300PhysicalExecutor:
    """Concrete executor for Full300EvaluationOwner, using the native allocation.

    worker_configurations maps each exact billing manifest SHA to its immutable
    config reference. Shared initial and pilot evaluation authorities stay separate.
    An interrupted dispatch is retained and blocks redispatch until reconciled.
    """
    def __init__(self, *, worker_configurations, python=sys.executable):
        self.configurations = dict(worker_configurations)
        self.python = str(Path(python).resolve())

    async def run_or_resume(self, store, request_ref, root, context):
        request = _json(request_ref); root = _safe_path(root)
        config_ref = self.configurations[store.manifest.sha256]; config = _json(config_ref)
        protocol = _validate_config(store, request, config)
        source = _validate_sources(request)
        _checkpoint(request['checkpoint_inventory'], request['policy_sha256'], request['iteration'], 'production_runtime')
        deadline = context['launch'].deadline_unix; _deadline(deadline)
        require(root.is_dir() and root == Path(request_ref['path']).parent, 'Owner cohort root changed')
        manifest_ref = saved_json(root / ('billing-manifest-' + store.manifest.sha256 + '.json'), store.manifest.to_dict())
        cohort = store.snapshot()['evaluation_cohorts'][request['cohort_id']]
        require(cohort['request_ref'] == request_ref, 'Cohort must be registered before execution')
        task_rows = [json.loads(line) for line in _artifact_file(request['task_manifest']).read_text().splitlines() if line.strip()]
        require([str(r['metadata']['task_id']) for r in task_rows] == list(_ids(request['task_manifest'])), 'Native task order changed')
        attempt_root = root / 'attempts'; attempt_root.mkdir(exist_ok=True)
        rows = []
        # Reconcile an interrupted parent after a successful child's immutable
        # row and proved cleanup. Nonzero/crashed children require independent
        # artifact reconciliation, even if they left a complete candidate row.
        for task in request['task_ids']:
            slot = cohort['tasks'][task]
            for attempt in slot['attempts']:
                directory = attempt_root / digest(attempt)
                row_file, cleanup_file = directory / 'row.json', directory / 'children' / 'cleanup.json'
                require(row_file.exists() and cleanup_file.exists(), 'Interrupted evaluation requires explicit retained-artifact reconciliation: ' + attempt)
                cleanup = _json(reference(cleanup_file))
                require(cleanup.get('owned_processes_closed') is True and cleanup.get('worker_returncode') == 0
                    and cleanup.get('attempt_id') == attempt, 'Prior browser cleanup incomplete')
                row = _json(reference(row_file))
                require(row['task_id'] == task and row['attempt_id'] == attempt, 'Saved evaluation attempt identity changed')
                for key in ('rollout', 'record', 'verdict'): _artifact_file(row[key])
                _settle_row(store, row)
            fresh = store.snapshot()['evaluation_cohorts'][request['cohort_id']]['tasks'][task]
            if fresh['selected_attempt']:
                rows.append(_json(reference(attempt_root / digest(fresh['selected_attempt']) / 'row.json')))
        selected = {r['task_id'] for r in rows}
        execution = root / 'executions' / digest([context['launch'].execution_id, time.time_ns()])
        execution.mkdir(parents=True, exist_ok=False)
        urls, version, proof = await _restore_and_serve(request, context, execution)
        semaphore = asyncio.Semaphore(config['concurrency'])
        env = dict(os.environ, **config['environment'], WANDB_MODE='disabled', WANDB_PROJECT='openwebrl-evals')
        env.pop('WANDB_RUN_ID', None)
        env.pop('WANDB_DISABLED', None)
        async def run_task(task_row):
            task = str(task_row['metadata']['task_id'])
            if task in selected: return None
            async with semaphore:
                _deadline(deadline)
                attempt = digest([request['cohort_id'], task, 'evaluation_browser', 0])
                directory = attempt_root / digest(attempt); directory.mkdir(exist_ok=False)
                from .generate_browser import _load_local_resources
                native_resources = _load_local_resources(task, deepcopy(config['browser_config']),
                    deepcopy(task_row['metadata']), response_format_mode_name='browser_env')
                packet = dict(native_resource_sha256=[digest(x) for x in native_resources],
                    schema_version=1, operation='task', request=request_ref, configuration=config_ref,
                    billing_root=str(store.root), billing_manifest=manifest_ref, task_id=task, attempt_id=attempt,
                    endpoint=urls[int(digest(task), 16) % len(urls)], weight_version=version,
                    sample_sha256=digest(dict(prompt=task_row['prompt'], metadata=task_row['metadata'])),
                    attempt_root=str(directory), packet_path=str(directory / 'packet.json'), deadline_unix=deadline)
                packet_ref = _write(directory / 'packet.json', packet)
                admission = store.start_evaluation_attempt(request['cohort_id'], task, attempt)
                require(admission['created'], 'Evaluation browser reservation must be new')
                await owned_process([self.python, '-B', '-m', 'openwebrl.action_credit_eval_executor',
                    '--packet', packet_ref['path'], '--sha256', packet_ref['sha256']], cwd=source, env=env,
                    log_path=directory / 'worker.log', deadline_seconds=_deadline(deadline),
                    detached_registry=directory / 'children', attempt_id=attempt)
                return _json(reference(directory / 'row.json'))
        jobs = [asyncio.create_task(run_task(row)) for row in task_rows]
        try:
            completed = await asyncio.gather(*jobs)
        except BaseException:
            for job in jobs:
                if not job.done(): job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            raise
        rows += [row for row in completed if row is not None]
        by_task = {r['task_id']: r for r in rows}; rows = [by_task[t] for t in request['task_ids']]
        attempts = store.snapshot()['evaluation_attempts']
        prior = []; http = []; actor = []
        for attempt, value in attempts.items():
            if value['cohort_id'] != request['cohort_id']: continue
            directory = attempt_root / digest(attempt)
            inventory = _json(reference(directory / 'transport-inventory.json'))
            http.extend(inventory['http_attempts']); actor.extend(inventory['actor_attempts'])
            if attempt not in {r['attempt_id'] for r in rows}:
                prior.append(_json(reference(directory / 'row.json')))
        outcomes = [_json(row['verdict']) for row in rows]
        successes = sum(v['status'] == 'valid' and v['reward'] == 1 for v in outcomes)
        valid = sum(v['status'] == 'valid' for v in outcomes)
        metrics = dict(tasks=300, successes=successes, valid=valid, invalid=300-valid,
                       overall_rate=successes/300, valid_only_rate=successes/valid if valid else None)
        wandb_ref = await self._wandb(request_ref, metrics, config, source, execution, deadline)
        runtime = saved_json(execution / 'runtime.json', dict(verification_kind='production_runtime',
            request_sha256=request_ref['sha256'], checkpoint_inventory=request['checkpoint_inventory'],
            policy_sha256=request['policy_sha256'], protocol=request['protocol'], worker_returncode=0,
            launcher_returncode=0, optimizer_updates=0, physical_restore_review=proof,
            worker_configuration=config_ref))
        result = dict(verification_kind='production_runtime', request_sha256=request_ref['sha256'],
            **{key: request[key] for key in ('protocol', 'policy_sha256', 'checkpoint_inventory', 'source_inventory')},
            runtime=runtime, tasks=rows, prior_attempts=prior, http_attempts=http, actor_attempts=actor,
            judge_http_protocol=asdict(protocol), metrics=metrics, wandb=wandb_ref)
        return saved_json(execution / 'result.json', result)

    async def _wandb(self, request_ref, metrics, config, source, execution, deadline):
        packet = dict(schema_version=1, operation='wandb', request=request_ref, metrics=metrics,
            output=str(execution / 'wandb.json'), entity=config.get('wandb_entity'), deadline_unix=deadline)
        packet_ref = _write(execution / 'wandb-packet.json', packet)
        env = dict(os.environ, **config['environment'], WANDB_MODE='online', WANDB_PROJECT='openwebrl-evals')
        env.pop('WANDB_RUN_ID', None)
        env.pop('WANDB_DISABLED', None)
        await owned_process([self.python, '-B', '-m', 'openwebrl.action_credit_eval_executor',
            '--packet', packet_ref['path'], '--sha256', packet_ref['sha256']], cwd=source, env=env,
            log_path=execution / 'wandb.log', deadline_seconds=_deadline(deadline))
        return reference(execution / 'wandb.json')


def finish_wandb(packet):
    verify_worker_environment(); _deadline(packet['deadline_unix'])
    import wandb
    request = _json(packet['request'])
    run_id = 'ac-eval-' + request['cohort_id'][:20]
    run = wandb.init(project='openwebrl-evals', entity=packet['entity'], id=run_id, resume='allow',
        config=dict(request_sha256=packet['request']['sha256'], protocol=request['protocol'],
                    policy_sha256=request['policy_sha256'], iteration=request['iteration']),
        name='action-credit-' + request['cohort_id'][:12])
    try:
        require(run is not None and run.project == 'openwebrl-evals' and run.id == run_id
                and run.config['request_sha256'] == packet['request']['sha256'], 'W&B identity changed')
        run.log(packet['metrics'])
        run.summary.update(packet['metrics'])
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish(exit_code=0)
    return saved_json(packet['output'], dict(project='openwebrl-evals', run_id=run_id, state='finished',
        request_sha256=packet['request']['sha256'], metrics=packet['metrics']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', required=True); parser.add_argument('--sha256', required=True)
    opts = parser.parse_args(); packet_ref = dict(path=opts.packet, sha256=opts.sha256)
    path = _safe_path(opts.packet)
    _private_file(packet_ref, path.parent)
    packet = _json(packet_ref)
    require(packet.get('schema_version') == 1, 'Unknown worker packet schema')
    if packet['operation'] == 'wandb':
        finish_wandb(packet); return
    require(packet['operation'] == 'task', 'Unknown worker operation')
    registry = os.environ.get(_CHILD_REGISTRY)
    require(registry, 'Browser worker requires detached-process ownership')
    async def run():
        with detached_process_scope(registry):
            task = asyncio.create_task(execute_task(packet))
            for sig in (signal.SIGINT, signal.SIGTERM):
                asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
            await task
    asyncio.run(run())


if __name__ == '__main__':
    main()
