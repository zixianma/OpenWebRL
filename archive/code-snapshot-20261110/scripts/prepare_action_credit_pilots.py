#!/usr/bin/env python3
"""Prepare five immutable iter10 campaigns; never submit or execute workers.

Run with the runtime Python and -B. ``--dry-run --runtime-pins FILE`` validates
the real production source and inputs without writing. ``--prepare`` writes
private immutable bundles and, only with actual API approval, empty independent
ledgers. Missing approval produces a draft without a ledger; approving it later
requires regeneration at a fresh output root, with a new manifest hash and new
bound physical evidence. Existing bundles are
verified byte-for-byte, never overwritten or reset. Neither mode confers API
authority, physical acceptance, storage admission or a scheduler allocation.

Use ``--print-runtime-pins-schema`` for the explicit input contract. The source
must be a fresh production fork, including the current driver/process/task
helpers; the earlier native validation fork is deliberately insufficient.
Approval references may be supplied per arm in --approval-index. Approved HTTP
protocols are pinned in the immutable manifest. Missing approvals remain null,
never synthetic approved receipts. The output launch
template also has null acceptance/execution/deadline fields, so DriverLaunch
rejects it until the runtime owner supplies actual evidence and allocation.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from openwebrl.action_credit import CollectionContract, digest
from openwebrl.action_credit_binding import (token_mask_contract, shared_prefix_contract,
                                            q_inventory_contract, _check_contract)
from openwebrl.action_credit_collect import generator_protocol
from openwebrl.action_credit_judge import JudgeHTTPProtocol
from openwebrl.action_credit_judge_budget import (input_bound_protocol_sha256,
    maximum_request_microdollars, INPUT_MICRODOLLARS_PER_MILLION,
    OUTPUT_MICRODOLLARS_PER_MILLION)
from openwebrl.action_credit_native_controller import validator_source_sha256
from openwebrl.action_credit_objective import ObjectiveConfig
from openwebrl.action_credit_online import online_collection_protocol
from openwebrl.action_credit_selection import (TreeSelectionPolicy, TreeUpdateSelectionPolicy,
                                              TreeSchedulePin)
from openwebrl.action_credit_slot_selection import slot_selection_policy
from openwebrl.action_credit_state import CampaignManifest, CampaignStore, _safe_path
from openwebrl.action_credit_windows import WindowConfig
from openwebrl.controlled_browser_server import validate_browser_manifest

RT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
AUDIT = RT / 'training-controls/action-credit-implementation-20261011'
ARM_IDS = ('chain_outcome_grpo', 'tree_outcome_sampling', 'tree_grpo', 'continuation_q', 'gigpo')
INPUT_NAMES = ('five-arm-iter10-resource-science-proposal.json', 'frozen-task-order.json',
               'frozen-objective-configs.json', 'frozen-update-protocol-v2.json',
               'frozen-collection-failure-protocol-v2.json', 'original-actor-inventory.json')
FROZEN_INPUTS = dict(zip(INPUT_NAMES, (
    'cebbaef583e92a2f360c280292a90fe4c34a72f35feaab05de251520c85bfb14',
    '7498187d35f2947fceb104c4dc2b90c1a9e1e623079a1304e04b540db67fbdf9',
    '5bd1f2a228794598fa20cf7c9d07d7f7124faafd190906b564e990db8441e9ce',
    'fd022c86435a5c40502e5d3fdb8c1a916baca0edf0cb7386e6e111176247cf55',
    'adbcf15d2f96cd48807a5ca5f8c1dba0419a0c99292b337d3b6cfc30a9b40b8c',
    '8723fc58184e5067d2abecd52fc1fd071b4c38c56c8967b99ba748c1a9da4a36')))
SAMPLING = dict(temperature=0.8, top_p=1.0, top_k=-1, max_new_tokens=1024)
CACHE_KEYS = ('HF_HOME', 'HUGGINGFACE_HUB_CACHE', 'TORCH_HOME', 'XDG_CACHE_HOME',
              'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR',
              'WANDB_DIR', 'TMPDIR', 'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR',
              'CUDA_CACHE_PATH', 'FLASHINFER_WORKSPACE_BASE', 'PIP_CACHE_DIR',
              'UV_CACHE_DIR', 'MPLCONFIGDIR', 'NUMBA_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def hash_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def reference(path):
    path = Path(path).absolute()
    return dict(path=str(path), sha256=hash_file(path))


def read_reference(ref):
    check(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'Exact artifact reference required')
    path = Path(ref['path'])
    check(path.is_absolute() and path.is_file(), 'Absolute existing artifact path required')
    check(hash_file(path) == ref['sha256'], 'Artifact digest mismatch: ' + str(path))
    return json.loads(path.read_text())


def no_credentials(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if any(word in str(key).lower() for word in ('api_key', 'access_token', 'password', 'secret_key')):
                check(item in (None, ''), 'Credentials cannot be frozen in a preparation bundle')
            no_credentials(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            no_credentials(item)


def runtime_pins_schema():
    return dict(schema_version=1, required={
        'production_source_root': 'Absolute fresh production fork; never native-validation-v3/source',
        'entrypoint_files': 'Nonempty relative source paths for actual bootstrap/native launcher; each hashed',
        'tokenizer_files': 'Nonempty [{path:absolute,sha256:actual}] from original actor inventory',
        'processor_files': 'Nonempty [{path:absolute,sha256:actual}] from original actor inventory',
        'native_args': 'Actual JSON native worker arguments: explicit max_steps=15, rollout_max_context_len=32768 and all generator_protocol history/format controls',
        'browser_config': 'Actual nonempty native browser configuration object',
        'browser_manifest': 'Actual nonempty browser/environment manifest object',
        'sampling': SAMPLING,
        'max_attempts_per_draw': 'Explicit positive integer; no retry is automatically dispatched'},
        approval_index='Optional object mapping each arm_id to an actual {path,sha256} API approval receipt',
        missing_approval='Draft only; no ledger. Actual approval requires a fresh output root and regenerated manifest/evidence.',
        no_execution=True, launch_ready=False)


def load_inputs(audit, *, expected_inputs=FROZEN_INPUTS):
    refs = {name: reference(Path(audit) / name) for name in INPUT_NAMES}
    check(set(expected_inputs) == set(INPUT_NAMES)
          and all(refs[name]['sha256'] == expected_inputs[name] for name in INPUT_NAMES),
          'Inputs differ from the frozen study; a new protocol requires explicit review')
    values = {name: read_reference(ref) for name, ref in refs.items()}
    proposal, order, configs, update, failure, model = (values[name] for name in INPUT_NAMES)
    check(set(proposal['pilot_caps']) == set(ARM_IDS) and set(configs) == set(ARM_IDS), 'Exactly five pinned arms required')
    check(proposal['target_completed_collection_iterations'] == 10, 'The approved study endpoint is iter10')
    check(update['schema'] == 'action_credit_update_protocol_v2', 'Require clarified update protocol v2')
    for name, sha in update['pins'].items():
        check(name in refs and refs[name]['sha256'] == sha, 'Frozen update input mismatch: ' + name)
    check(failure['schema_version'] == 2 and failure['tree_pair']['version'] == 'prescheduled_slot_k_v2',
          'Require explicit Tree slot failure protocol v2')
    check((failure['tree_pair']['m'], failure['tree_pair']['n'], failure['tree_pair']['l'], failure['tree_pair']['k']) == (2, 2, 2, 4),
          'Tree schedule changed')
    common = proposal['common_science_recommendation']
    check(common['seed'] == 42 and common['groups_per_iteration'] == 48
          and common['independent_chains_per_group'] == 8 and common['fresh_optimizer_scheduler_rng_cursor'] is True,
          'Fresh original-model fixed48x8 study required')
    tasks = order['original_manifest_task_order']
    check(len(tasks) == 2102 and len(set(tasks)) == 2102 and order['groups_per_iteration'] == 48
          and order['outcome_dependent_replacement'] is False and len(order['iterations']) == 10
          and order['source_sha256'] == common['train_data_sha256'], 'Frozen training inventory changed')
    indices = []
    for i, entry in enumerate(order['iterations']):
        groups = entry['groups']
        check(entry['iteration'] == i+1 and len(groups) == 48 and len({g['group_id'] for g in groups}) == 48
              and len({g['task_id'] for g in groups}) == 48, 'Each update requires 48 distinct prespecified groups/tasks')
        for g in groups:
            index = g['source_index']
            check(type(index) is int and 0 <= index < len(tasks) and tasks[index] == g['task_id'], 'Task/source index mismatch')
            indices.append(index)
    check(indices == order['ordered_source_indices'][:480] and len(set(indices)) == 480
          and sorted(order['ordered_source_indices']) == list(range(2102)), 'Frozen task cursor changed')
    check(model['model_root'] == common['initial_model_path'] and model['weights_rehashed'] is True
          and model['fresh_native_optimizer_scheduler_required'] is True, 'Original actor inventory is not verified/fresh-init compatible')
    weights = [f for f in model['files'] if Path(f['path']).name == 'model.safetensors']
    check(len(weights) == 1 and weights[0]['sha256'] == common['model_sha256_from_download_manifest'], 'Original model weights differ')
    for arm, config in configs.items():
        ObjectiveConfig(**config).validate()
        check(config['arm_id'] == arm, 'Objective/arm identity mismatch')
    return values, refs


def validate_runtime(pins, model, *, repo=REPO):
    required = set(runtime_pins_schema()['required']) | {'schema_version'}
    check(isinstance(pins, dict) and set(pins) == required and pins['schema_version'] == 1, 'Runtime pins do not match --print-runtime-pins-schema')
    no_credentials(pins)
    root = Path(pins['production_source_root']).absolute()
    check(root.is_dir() and root.resolve() != Path(repo).resolve()
          and 'native-validation-v3' not in root.parts, 'Require a fresh isolated production source fork')
    check(pins['sampling'] == SAMPLING, 'Actual sampling differs from frozen scientific protocol')
    for name in ('native_args', 'browser_config', 'browser_manifest'):
        check(isinstance(pins[name], dict) and bool(pins[name]), 'Explicit actual ' + name + ' required')
    check(isinstance(pins['native_args'].get('browser_response_format_mode'), str)
          and pins['native_args']['browser_response_format_mode'], 'Native response-format mode required')
    native = pins['native_args']
    check(type(native.get('max_steps')) is int and native['max_steps'] == 15
          and type(native.get('rollout_max_context_len')) is int and native['rollout_max_context_len'] == 32768,
          'Native max_steps=15 and context32768 must match the frozen study')
    check(native.get('browser_action_selector') is None, 'Existing native action selector cannot be replaced')
    generator_protocol(SimpleNamespace(**native))  # Require every explicit history/control field and resolve actual tags.
    check(type(pins['max_attempts_per_draw']) is int and pins['max_attempts_per_draw'] > 0, 'Explicit retry cap required')
    required_sources = sorted(p.relative_to(repo).as_posix() for p in (Path(repo)/'openwebrl').glob('action_credit*.py'))
    for name in ('action_credit_driver.py', 'action_credit_process.py', 'action_credit_task_inputs.py'):
        check('openwebrl/'+name in required_sources, 'Concrete production helper missing: '+name)
    required_sources += ['openwebrl/generate_browser.py', 'openwebrl/reward_browser.py',
                         'openwebrl/arm_branch_browser.py', 'openwebrl/arm_branch_protocol.py',
                         'openwebrl/rl_recipe.py', 'openwebrl/controlled_sft_worker.py',
                         'openwebrl/controlled_browser_server.py', 'openwebrl/response_format.py',
                         'openwebrl/eval_monitor.py', 'scripts/check_storage_quota.py']
    entries = pins['entrypoint_files']
    check(isinstance(entries, list) and entries and len(set(entries)) == len(entries), 'Explicit unique production entrypoints required')
    inventory = {}
    for relative in sorted(set(required_sources + entries)):
        p = Path(relative)
        check(not p.is_absolute() and '..' not in p.parts and p.parts, 'Source path must be relative and contained')
        source = root/p
        check(root.resolve() in source.resolve().parents and source.is_file(), 'Production source path escaped or absent')
        sha = hash_file(source)
        if relative in required_sources:
            check(sha == hash_file(Path(repo)/p), 'Stale production source: '+relative)
        inventory[relative] = sha
    browser = pins['browser_config']
    local = browser.get('local_process', {})
    expected_browser = dict(mode='local_process', width=1280, height=720, dpr=1,
        use_screenshot=True, use_a11ytree=False, resize_output_coords=True, resize_scale=1000)
    check(all(browser.get(k) == v for k,v in expected_browser.items())
          and local.get('server_module') == 'openwebrl.arm_branch_browser'
          and local.get('server_module_sha256') == inventory['openwebrl/arm_branch_browser.py'],
          'Strict branch-evidence browser observation/source contract required')
    for name in ('log_dir', 'port_lock_dir'):
        check(isinstance(local.get(name), str) and Path(local[name]).is_absolute(), 'Explicit private browser '+name+' required')
        _safe_path(local[name])
    check(pins['browser_manifest'].get('branch_state_version') in ('observable-replay-v1', 'visible-replay-v2'),
          'Explicit frozen branch replay protocol required')
    # The actual collector supplies an attempt-private artifact directory at
    # dispatch. This validation does not create the illustrative child path.
    validate_browser_manifest(dict(pins['browser_manifest'],
        artifact_directory=str(Path(local['log_dir'])/'preparation-validation-no-create')))
    known = {entry['path']: entry['sha256'] for entry in model['files']}
    resource_hashes = {}
    for kind in ('tokenizer', 'processor'):
        files = pins[kind+'_files']
        check(isinstance(files, list) and files and len({f['path'] for f in files}) == len(files), 'Explicit unique '+kind+' artifacts required')
        names = ({'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                  'chat_template.json', 'chat_template.jinja'} if kind == 'tokenizer' else
                 {'config.json', 'processor_config.json', 'preprocessor_config.json', 'video_preprocessor_config.json'})
        expected_files = {path for path in known if Path(path).name in names}
        check({f['path'] for f in files} == expected_files and expected_files,
              'Pin every original-model '+kind+' file, not a selected subset')
        for f in files:
            check(set(f) == {'path', 'sha256'} and known.get(f['path']) == f['sha256']
                  and hash_file(f['path']) == f['sha256'], kind+' must use actual pinned original-model artifacts')
        resource_hashes[kind+'_sha256'] = digest(sorted(files, key=lambda f: f['path']))
    reward = root/'openwebrl/reward_browser.py'
    prompts = {}
    for node in ast.parse(reward.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('JUDGE_SYSTEM_PROMPT_ACTION_HISTORY', 'JUDGE_USER_PROMPT_ACTION_HISTORY'):
                    prompts[target.id] = ast.literal_eval(node.value)
    check(len(prompts) == 2 and all(isinstance(v, str) and v for v in prompts.values()), 'Canonical action-history prompts absent')
    return dict(source_root=str(root), files=inventory), resource_hashes, digest(prompts)


def make_tree_policy(order, order_sha):
    policy = TreeSelectionPolicy(order_sha, tuple(TreeUpdateSelectionPolicy(i,
        tuple(TreeSchedulePin(g['task_id'], g['group_id'], 2, 2, 2, 4, 42) for g in entry['groups']))
        for i, entry in enumerate(order['iterations'])))
    policy.validate()
    return policy


def approved_protocol(arm, campaign_id, caps, approvals):
    ref = approvals.get(arm)
    if ref is None:
        return None, None
    receipt = read_reference(ref)
    check(receipt.get('status') == 'approved' and receipt.get('campaign_id') == campaign_id
          and receipt.get('model') == 'gpt-4.1'
          and receipt.get('api_microdollars_cap') == caps['api_microdollars']
          and receipt.get('api_requests_cap') == caps['api_requests']
          and receipt.get('user_authorization_text') and receipt.get('user_authorization_reference'),
          'Approval does not authorize this exact campaign and cap')
    protocol = JudgeHTTPProtocol('gpt-4.1', digest(receipt), input_bound_protocol_sha256(), 1024,
        maximum_request_microdollars(1024), INPUT_MICRODOLLARS_PER_MILLION, OUTPUT_MICRODOLLARS_PER_MILLION)
    protocol.sha256
    return ref, asdict(protocol)


def build_bundles(audit, output_root, pins, approvals=None, *, expected_inputs=FROZEN_INPUTS):
    """Pure read/validate/build; no directory creation, resource requests or calls."""
    values, refs = load_inputs(audit, expected_inputs=expected_inputs)
    proposal, order, configs, update, failure, model = (values[n] for n in INPUT_NAMES)
    source, resource_hashes, prompt_sha = validate_runtime(pins, model)
    approvals = {} if approvals is None else approvals
    check(isinstance(approvals, dict) and set(approvals) <= set(ARM_IDS), 'Unknown approval arm')
    output_root = _safe_path(output_root)
    common = proposal['common_science_recommendation']
    # Dataset bytes are small enough to verify during preparation. Large model
    # weights are rechecked by native initialization; identity is not GPU proof.
    check(hash_file(common['train_data']) == common['train_data_sha256'], 'Frozen training parquet changed')
    tree = make_tree_policy(order, refs[INPUT_NAMES[1]]['sha256'])
    mask, window = token_mask_contract(), asdict(WindowConfig(2, 256, 42))
    judge_science = dict(model='gpt-4.1', prompt_variant='action_history', prompt_sha256=prompt_sha,
        canonical_reward_source_sha256=source['files']['openwebrl/reward_browser.py'],
        reward='binary_terminal_success', native_combined_and_format='diagnostics_only',
        max_completion_tokens=1024, input_bound_protocol_sha256=input_bound_protocol_sha256())
    judge_sha = digest(judge_science)
    source_sha = digest(source)
    task_splits = {task: 'train' for task in order['original_manifest_task_order']}
    contract = CollectionContract(0, refs[INPUT_NAMES[5]]['sha256'],
        resource_hashes['tokenizer_sha256'], resource_hashes['processor_sha256'],
        digest(dict(dataset=common['train_data_sha256'], ordered_task_ids=order['original_manifest_task_order'])),
        judge_sha, digest(pins['sampling']), .8, tuple(order['original_manifest_task_order']), ())
    contract.validate()
    bundles = {}
    for arm in ARM_IDS:
        campaign_id = 'action-credit-iter10-20261011-'+arm
        arm_root = output_root/arm
        cap = proposal['pilot_caps'][arm]
        expected = dict(gpus=8, gpu_type='H200', cpus=64, memory_gib=960,
            total_allocation_seconds_all_attempts=86400, gpu_hours_cap=192,
            judge_http_attempts_cap_proposed=10000, judge_usd_cap_proposed=200,
            browser_dispatches_cap=20000, local_actor_requests_cap=300000,
            browser_concurrency=64, protective_storage_bytes_cap=5*1024**4)
        check(all(cap.get(k) == v for k,v in expected.items())
              and all(cap.get(k) is True for k in ('includes_full300_iteration10_and_all_retries', 'stop_at_first_cap', 'residuals_not_transferable')),
              'Per-arm explicit resource proposal changed; do not infer new budgets')
        caps = dict(gpu_seconds=cap['gpus']*cap['total_allocation_seconds_all_attempts'],
            api_microdollars=cap['judge_usd_cap_proposed']*1_000_000,
            api_requests=cap['judge_http_attempts_cap_proposed'], browser_dispatches=cap['browser_dispatches_cap'])
        approval_ref, http_protocol = approved_protocol(arm, campaign_id, caps, approvals)
        judge = dict(judge_science, judge_protocol_sha256=judge_sha)
        if http_protocol is not None:
            judge['http_protocol_sha256'] = JudgeHTTPProtocol(**http_protocol).sha256
        config = configs[arm]
        algorithm = dict(protocol_sha256=digest(dict(config=config, update=update, failure=failure)),
            objective_config_sha256=digest(config), ppo_window_config=window,
            online_collection_protocol=online_collection_protocol(42), training_world_size=8,
            local_actor_requests_cap=cap['local_actor_requests_cap'],
            leaf_selection=slot_selection_policy(tree) if arm.startswith('tree_') else dict(mode='all_fit_inventory'),
            frozen_update_protocol=refs[INPUT_NAMES[3]], frozen_collection_failure_protocol=refs[INPUT_NAMES[4]],
            optimizer=common['optimizer'], topology=dict(tp=2, dp=4, pp=1, cp=1, microbatch=1),
            target_completed_updates=10, context_tokens=32768)
        if arm == 'continuation_q':
            algorithm['q_inventory_protocol'] = q_inventory_contract()
        prefix = shared_prefix_contract(config['reduction'])
        manifest = CampaignManifest.from_dict(dict(schema_version=1, campaign_id=campaign_id, arm_id=arm,
            initialization=dict(original_actor_sha256=contract.policy_sha256, iteration=0,
                fresh_optimizer=True, fresh_scheduler=True, fresh_cursor=True),
            science=dict(
                checkpoint=dict(protocol_sha256=digest(update['checkpoint']), original_actor_path=model['model_root'],
                    original_actor_inventory=refs[INPUT_NAMES[5]], native_initial_state_root=str(arm_root/'native-initial-state'),
                    native_checkpoint_validator_sha256=validator_source_sha256(), source_inventory_sha256=source_sha,
                    fresh_rank_rng=True),
                dataset=dict(source_sha256=common['train_data_sha256'], path=common['train_data'],
                    task_manifest_sha256=contract.task_manifest_sha256, task_order_sha256=refs[INPUT_NAMES[1]]['sha256'],
                    task_manifest_label=common['task_manifest_label']),
                judge=judge,
                generation=dict(temperature=.8, top_p=1., top_k=-1, max_response_tokens=1024, max_turns=15,
                    generation_sha256=contract.generation_sha256, likelihood_temperature=.8,
                    native_protocol_sha256=digest(generator_protocol(SimpleNamespace(**pins['native_args']))),
                    likelihood_convention=contract.likelihood_convention, **resource_hashes),
                token_mask=dict(protocol_sha256=digest(mask), contract=mask),
                shared_prefix_normalization=dict(protocol_sha256=digest(prefix), contract=prefix), algorithm=algorithm),
            task_splits=task_splits, split_sha256=digest(task_splits), code_revision_sha256=source_sha,
            budget_caps=caps, max_attempts_per_draw=pins['max_attempts_per_draw']))
        _check_contract(manifest.to_dict(), contract)
        manifest_bytes = encoded(manifest.to_dict())
        launch = dict(schema_version=1,
            manifest=dict(path=str(arm_root/'manifest.json'), sha256=hashlib.sha256(manifest_bytes).hexdigest()),
            store_root=str(arm_root/'ledger'), task_order=refs[INPUT_NAMES[1]], initial_contract=asdict(contract),
            objective_config=config, native_acceptance=None, execution_id=None, deadline_unix=None,
            tree_policy=asdict(tree) if arm.startswith('tree_') else None, target_updates=10, worker_concurrency=64)
        env = {key: str(arm_root/'cache'/key.lower()) for key in CACHE_KEYS}
        env.update(PYTHONDONTWRITEBYTECODE='1', WANDB_PROJECT='openwebrl', WANDB_RUN_ID='ac10-'+arm,
            WANDB_NAME=campaign_id, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OPENBLAS_NUM_THREADS='1')
        worker = dict(schema_version=1, store_root=str(arm_root/'ledger'), manifest_sha256=manifest.sha256,
            native_args=pins['native_args'], sampling=pins['sampling'], browser_config=pins['browser_config'],
            browser_manifest=pins['browser_manifest'], approval_reference=approval_ref, judge_protocol=http_protocol,
            canonical_reward_source_sha256=judge_science['canonical_reward_source_sha256'],
            runtime_fields_required=['contract', 'actor', 'request', 'sample', 'operation'],
            prefix_ref='runtime optional private Torch reference; never an in-memory fake prefix')
        accounting = dict(schema_version=1, campaign_id=campaign_id, resource_proposal=refs[INPUT_NAMES[0]],
            gpu_authorization_text=proposal['authorization']['gpu_text'], caps=cap,
            gpu_seconds_unit='sum of allocated GPU-seconds; 8 times allocation wall seconds, including every retry/startup/eval',
            total_allocation_seconds_cap=86400, ledger_caps=caps, allocation_ledger=str(arm_root/'allocation-ledger.json'),
            separate_approvals=True, transfer_prior_balances=False, cleanup_authorized=False,
            storage=dict(protective_ceiling_bytes=cap['protective_storage_bytes_cap'],
                global_free_floor_bytes=proposal['storage']['global_floor_bytes'],
                admission='Fresh quota and globally coordinated forward stage reservation required; this ceiling is not reserved space'),
            api_approval_reference=approval_ref, api_status='approved' if approval_ref else 'pending',
            evaluation=proposal['evaluation'], wandb=dict(training_project='openwebrl', evaluation_project='openwebrl-evals',
                run_id='ac10-'+arm, resume='must after first initialization'))
        blockers = ['Actual allocation attempt/deadline and all-attempt accounting',
            'Fresh quota and feasible coordinated storage reservation',
            'Manifest/source-bound native physical acceptance, including save/restore/RNG and driver integration',
            'Batch-bound physical browser/context/mask/actor-transfer/judge/cost acceptance for each update',
            'Owning controller must await full300 iteration10 evaluation and independent final artifact review']
        if approval_ref is None:
            blockers.insert(0, 'Draft only: exact API approval is absent; regenerate at a fresh output root before creating a ledger')
        readiness = dict(schema_version=1, manifest_sha256=manifest.sha256, code_revision_sha256=source_sha,
            cpu_preparation_validated=True, production_runtime_validated=False, launch_ready=False,
            bundle_status='approved_manifest_pending_physical_acceptance' if approval_ref else 'draft_pending_api_approval',
            ledger_creation_permitted=approval_ref is not None,
            requires_fresh_preparation_after_api_approval=approval_ref is None,
            authorizes_execution=False, blockers=blockers)
        bundles[arm] = dict(manifest=manifest, files={
            'manifest.json': manifest.to_dict(), 'launch.template.json': launch,
            'worker.common.json': worker, 'environment.json': env,
            'resource-accounting.json': accounting, 'readiness.json': readiness,
            'source-inventory.json': source, 'input-references.json': refs,
            'runtime-pins.json': pins})
    return bundles


def write_bundles(bundles, output_root):
    """Exclusive initial write; repeat preparation validates without resetting costs."""
    root = _safe_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for arm, bundle in bundles.items():
        arm_root = _safe_path(root/arm)
        check(bundle['files']['launch.template.json']['store_root'] == str(arm_root/'ledger'),
              'Prepared bundle belongs to a different output root')
        worker = bundle['files']['worker.common.json']
        approved = worker['approval_reference'] is not None and worker['judge_protocol'] is not None
        if approved:
            protocol = JudgeHTTPProtocol(**worker['judge_protocol'])
            check(bundle['manifest'].to_dict()['science']['judge'].get('http_protocol_sha256') == protocol.sha256,
                  'Approved HTTP protocol must be pinned before creating a ledger')
        rendered = {name: encoded(value) for name,value in bundle['files'].items()}
        receipt = dict(schema_version=1, arm_id=arm, manifest_sha256=bundle['manifest'].sha256,
            files={name: hashlib.sha256(raw).hexdigest() for name,raw in rendered.items()},
            bundle_status=bundle['files']['readiness.json']['bundle_status'], ledger_created=approved,
            launch_ready=False, authorizes_execution=False)
        if arm_root.exists():
            check((arm_root/'PREPARED.json').is_file(), 'Partial preparation requires explicit reconciliation: '+str(arm_root))
            check(json.loads((arm_root/'PREPARED.json').read_text()) == receipt,
                  'Existing immutable preparation differs; use a fresh output root, never reset a ledger')
            for name, raw in rendered.items():
                check(_safe_path(arm_root/name).read_bytes() == raw, 'Prepared artifact changed: '+name)
            if approved:
                CampaignStore(arm_root/'ledger', bundle['manifest'])
            else:
                check(not (arm_root/'ledger').exists(), 'An unapproved draft must not have a ledger')
        else:
            arm_root.mkdir(mode=0o700)
            for name, raw in rendered.items():
                _exclusive(arm_root/name, raw)
            if approved:
                CampaignStore.create(arm_root/'ledger', bundle['manifest'])
            _exclusive(arm_root/'PREPARED.json', encoded(receipt))
        receipts[arm] = dict(path=str(arm_root/'PREPARED.json'), sha256=hash_file(arm_root/'PREPARED.json'))
    return receipts


def _exclusive(path, raw):
    path = _safe_path(path)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--print-runtime-pins-schema', action='store_true')
    parser.add_argument('--runtime-pins', type=Path)
    parser.add_argument('--audit-dir', type=Path, default=AUDIT)
    parser.add_argument('--output-root', type=Path, default=AUDIT/'pilots')
    parser.add_argument('--approval-index', type=Path)
    args = parser.parse_args(argv)
    if args.print_runtime_pins_schema:
        print(json.dumps(runtime_pins_schema(), indent=2, sort_keys=True))
        return
    parser.error('--runtime-pins is required') if args.runtime_pins is None else None
    pins = json.loads(args.runtime_pins.read_text())
    approvals = json.loads(args.approval_index.read_text()) if args.approval_index else {}
    bundles = build_bundles(args.audit_dir, args.output_root, pins, approvals)
    result = dict(mode='prepare' if args.prepare else 'dry-run', launch_ready=False, authorizes_execution=False,
        arms={arm: dict(manifest_sha256=b['manifest'].sha256,
            readiness=b['files']['readiness.json']) for arm,b in bundles.items()})
    if args.prepare:
        result['receipts'] = write_bundles(bundles, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
