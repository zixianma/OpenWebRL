#!/usr/bin/env python3
"""Prepare private, explicitly non-executable full300 configuration drafts.

No store, worker, browser, GPU, API client or scheduler is created here. The
separate native parser probe only records CPU argument resolution; it is never
physical acceptance. Missing selection/authority/physical evidence stays null.
Approval requires fresh manifests/configurations, never editing these drafts.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
for directory in (REPO, REPO / 'scripts'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from prepare_action_credit_pilots import ARM_IDS, CACHE_KEYS, FROZEN_INPUTS, encoded, hash_file, no_credentials
from prepare_action_credit_runtime import AUDIT, PYTHON, base_environment, output_path, write_exclusive
from openwebrl.action_credit import digest
from openwebrl.action_credit_evaluation import PINNED_FULL300_SHA256, evaluation_protocol, native_evaluation_controls
from openwebrl.action_credit_verification import prepare_verification_manifest

STATUS = 'draft_not_executable'
SERVER = 'openwebrl/docker/env_server.py'
PROMPT = 'openwebrl/env/prompts/system_prompt_browser_env.md'
TOOLS = 'openwebrl/env/prompts/tool_list.json'
PARSER = 'scripts/prepare_action_credit_runtime.py'
GENERATOR = 'slime/rollout/sglang_rollout.py'
RESOURCE_NAMES = {
    'tokenizer': {'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                  'chat_template.json', 'chat_template.jinja'},
    'processor': {'config.json', 'processor_config.json', 'preprocessor_config.json',
                  'video_preprocessor_config.json'},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path):
    path = Path(path).absolute()
    require(path == path.resolve() and path.is_file(), 'Concrete non-symlink input required')
    return dict(path=str(path), sha256=hash_file(path))


def read(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'Exact artifact reference required')
    require(reference(ref['path']) == ref, 'Pinned input changed: ' + str(ref['path']))
    return json.loads(Path(ref['path']).read_text())


def source_pins(ref):
    inventory = read(ref)
    root = Path(inventory['source_root'])
    require(root.is_absolute() and root == root.resolve() and 'native-validation-v3' not in root.parts,
            'Exact isolated production source required')
    for name, sha in inventory['files'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'Source inventory path escapes root')
        require(reference(root/path)['sha256'] == sha, 'Source bytes changed: ' + name)
    for name in (SERVER, PROMPT, TOOLS, PARSER, GENERATOR, 'openwebrl/action_credit_eval_executor.py',
                 'openwebrl/action_credit_evaluation.py', 'openwebrl/reward_browser.py'):
        require(name in inventory['files'], 'Missing physical evaluation dependency: ' + name)
    return inventory


def generation_argument_names(source):
    """Read the real GenerateState constructor rather than guess its defaults."""
    tree = ast.parse((Path(source)/GENERATOR).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GenerateState')
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    names = {node.attr for node in ast.walk(init) if isinstance(node, ast.Attribute)
             and isinstance(node.value, ast.Name) and node.value.id == 'args'}
    for node in ast.walk(init):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'getattr'
                and len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id == 'args'
                and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
            names.add(node.args[1].value)
    require('hf_checkpoint' in names and 'rollout_stop' in names, 'Native GenerateState interface changed')
    return sorted(names)


def parse_native_arguments(runtime, source_ref, destination):
    """Isolated existing partial CPU parser; no full CUDA validator or clients."""
    runtime, destination = Path(runtime).resolve(), output_path(destination)
    require(not destination.exists(), 'Never overwrite a parser probe')
    source = source_pins(source_ref)
    require(Path(source['source_root']) == runtime/'source', 'Runtime/source identity differs')
    argv_ref, config_ref = reference(runtime/'native-argv.json'), reference(runtime/'native-config.json')
    plan = read(reference(runtime/'preparation.json'))
    require(read(argv_ref) == plan['native_argv'] and read(config_ref) == plan['native_config'],
            'Actual native arguments differ from preserved preparation')
    names = generation_argument_names(source['source_root'])
    packet = dict(source_inventory=source_ref, native_argv=argv_ref, native_config=config_ref,
                  argument_names=names, output=str(destination/'native-arguments.json'))
    destination.mkdir(parents=True)
    env = base_environment(destination, source['source_root'])
    for key in CACHE_KEYS:
        output_path(env[key]).mkdir(parents=True, exist_ok=True)
    packet_path = destination/'packet.json'
    write_exclusive(packet_path, encoded(packet))
    # This helper is defined in the pinned production fork, never train.py.
    child = """import importlib.util,json,sys,hashlib,os
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text())
def load(r):
 b=Path(r['path']).read_bytes()
 assert hashlib.sha256(b).hexdigest()==r['sha256']
 return json.loads(b)
s=load(p['source_inventory']); root=Path(s['source_root'])
for n,h in s['files'].items(): assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
sys.path.insert(0,str(root/'scripts'))
spec=importlib.util.spec_from_file_location('pinned_runtime_parser',root/'scripts/prepare_action_credit_runtime.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.verify_external_runtime(json.loads((root/'action-credit-runtime-dependencies.json').read_text()))
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.environ['WANDB_MODE']=='disabled'
a=m.parse_cpu_native_arguments(load(p['native_argv']))
assert load(p['native_config'])==json.loads((Path(p['native_config']['path'])).read_text())
import torch
assert not torch.cuda.is_initialized()
values={n:getattr(a,n) for n in p['argument_names']}
result=dict(schema_version=1,verification_kind='partial_cpu_argument_validation',cuda_initialized=False,
 browser_calls=0,judge_calls=0,physical_native_acceptance=False,source_inventory=p['source_inventory'],
 native_argv=p['native_argv'],native_config=p['native_config'],native_args=values)
with open(p['output'],'x') as f: json.dump(result,f,sort_keys=True,indent=2);f.write('\\n');f.flush();os.fsync(f.fileno())
"""
    with (destination/'parse.log').open('xb') as log:
        result = subprocess.run([str(PYTHON), '-B', '-c', child, str(packet_path)],
            cwd=source['source_root'], env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, timeout=180, check=False)
    require(result.returncode == 0, 'CPU argument probe failed; retained parse.log, no execution permitted')
    return reference(destination/'native-arguments.json')


def _model_inputs(model_ref, pins):
    model = read(model_ref)
    known = {row['path']: row['sha256'] for row in model['files']}
    require(model['model_root'] == pins['native_args']['hf_checkpoint'], 'Original actor changed')
    result = dict(original_actor_inventory=model_ref, model_root=model['model_root'],
                  large_weights_rehashed_by_this_preparation=False)
    for kind, names in RESOURCE_NAMES.items():
        refs = pins[kind+'_files']
        expected = {path: sha for path, sha in known.items() if Path(path).name in names}
        require(len(refs) == len(expected) and {r['path']:r['sha256'] for r in refs} == expected,
                'Every original '+kind+' input must be pinned')
        for ref in refs:
            require(reference(ref['path']) == ref, 'Original '+kind+' input changed')
        result[kind+'_files'] = refs
        result[kind+'_sha256'] = digest(sorted(refs, key=lambda r:r['path']))
    return result


def _judge_prompt(source):
    values = {}
    tree = ast.parse((Path(source['source_root'])/'openwebrl/reward_browser.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('JUDGE_SYSTEM_PROMPT_ACTION_HISTORY', 'JUDGE_USER_PROMPT_ACTION_HISTORY'):
                    values[target.id] = ast.literal_eval(node.value)
    require(len(values) == 2 and all(isinstance(v, str) and v for v in values.values()), 'Canonical judge prompts absent')
    return digest(values)


def worker_draft(root, source, native_args, pins):
    root = output_path(root)
    browser = deepcopy(pins['browser_config'])
    browser.update(path_to_policy=str(Path(source['source_root'])/PROMPT),
                   path_to_tool_list=str(Path(source['source_root'])/TOOLS))
    browser['local_process'].update(server_module='openwebrl.docker.env_server',
        server_module_sha256=source['files'][SERVER], log_dir=str(root/'browser-logs'),
        port_lock_dir=str(root/'browser-locks'))
    args = dict(native_args, **native_evaluation_controls())
    args.update(max_steps=30, inference_step_timeout_secs=120., judge_timeout_secs=120.,
        judge_api_mode='served', judge_api_model='gpt-4.1', judge_prompt_variant='action_history',
        browser_response_format_mode='browser_env', browser_action_selector=None,
        rollout_temperature=0., rollout_top_p=1., rollout_top_k=1, rollout_max_response_len=4096,
        browser_rollout_concurrency=64, debug_trace_prob=0., log_task_start=True,
        log_llm_output=False, log_judge_output=False, path_to_save_generated_samples='')
    # Never inherit training W&B identity, credentials, or experiment overrides.
    environment = {key:str(root/'cache'/key.lower()) for key in CACHE_KEYS}
    # Python/browser subprocesses may bind Unix sockets below TMPDIR. A long
    # audit root exceeds their roughly108-byte pathname limit; retain a unique
    # short scratch root, with no cleanup authorized by this preparer.
    environment['TMPDIR'] = str(output_path('/tmp/acev-'+digest(str(root))[:16]))
    environment.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', PYTHONNOUSERSITE='1', WANDB_PROJECT='openwebrl-evals',
        TOKENIZERS_PARALLELISM='false', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
        SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING='0', SLIME_BROWSER_THINKING_OPEN_TAG='<think>',
        SLIME_BROWSER_THINKING_CLOSE_TAG='</think>', SLIME_BROWSER_APPEND_THINKING_PREFILL='1',
        PLAYWRIGHT_BROWSERS_PATH=str(PYTHON.parents[2]/'browsers'))
    value = dict(schema_version=1, preparation_status=STATUS, accounting_manifest_sha256=None,
        protocol=evaluation_protocol(), judge_protocol=None, approval_reference=None,
        canonical_reward_source_sha256=source['files']['openwebrl/reward_browser.py'],
        native_args=args, browser_config=browser, environment=environment, concurrency=64)
    no_credentials(value)
    return value


def comparable_runtime_pins(pilots, arm, pins, worker, manifest, source_ref, runtime):
    """Normalize only output paths proved by the immutable execution binding.

    Older unbound drafts must remain exactly equal. Enriched drafts intentionally
    have per-arm logs/locks; neither arbitrary path differences nor scientific
    differences are accepted. PREPARED binds all files before normalization.
    """
    root = Path(pilots)/arm
    binding_path = root/'execution-binding.json'
    if not binding_path.exists():
        return deepcopy(pins), None
    receipt_ref, binding_ref = reference(root/'PREPARED.json'), reference(binding_path)
    receipt, binding = read(receipt_ref), read(binding_ref)
    identity = digest(manifest)
    require(receipt.get('schema_version') == binding.get('schema_version') == 1
            and receipt.get('arm_id') == binding.get('arm_id') == arm
            and receipt.get('manifest_sha256') == binding.get('manifest_sha256') == identity,
            'Prepared execution binding changed arm/manifest identity')
    required = {'execution-binding.json','runtime-pins.json','worker.common.json','manifest.json',
                'source-inventory.json','native-argv.json','native-config.json'}
    require(isinstance(receipt.get('files'),dict) and required <= set(receipt['files']),
            'PREPARED does not bind every relevant execution input')
    for name, sha in receipt['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Prepared input escaped pilot root')
        require(reference(root/relative)['sha256'] == sha, 'Prepared execution input changed: '+name)
    require(binding.get('source_inventory') == source_ref
            and binding.get('base_native_argv') == reference(Path(runtime)/'native-argv.json')
            and binding.get('base_native_config') == reference(Path(runtime)/'native-config.json')
            and binding.get('native_argv') == reference(root/'native-argv.json')
            and binding.get('native_config') == reference(root/'native-config.json'),
            'Execution binding changed the actual native/source inputs')
    require(worker.get('browser_config') == pins.get('browser_config'),
            'Prepared worker browser differs from runtime pins')
    normalized = deepcopy(pins)
    local = normalized['browser_config']['local_process']
    for key, target, suffix in (('log_dir','browser_logs','browser-logs'),
                                ('port_lock_dir','browser_locks','browser-locks')):
        expected = str(output_path(root/suffix))
        require(local.get(key) == binding['paths'].get(target) == expected,
                'Execution browser output is not bound to its own arm: '+key)
        # Only these two verified private paths may differ across arms.
        del local[key]
    return normalized, dict(prepared=receipt_ref, execution_binding=binding_ref)


def build_drafts(*, audit, runtime, pilots, output_root, native_arguments_reference,
                 verification_selection_reference=None, expected_inputs=FROZEN_INPUTS):
    """Read/validate/build only. Missing scientific selections are not guessed."""
    audit, runtime, pilots = map(lambda p:Path(p).resolve(), (audit, runtime, pilots))
    output = output_path(output_root)
    require(not output.exists(), 'Use a fresh draft directory')
    proposal_ref = reference(audit/'five-arm-iter10-resource-science-proposal.json')
    model_ref = reference(audit/'original-actor-inventory.json')
    order_ref = reference(audit/'frozen-task-order.json')
    for ref in (proposal_ref, model_ref, order_ref):
        require(ref['sha256'] == expected_inputs[Path(ref['path']).name], 'Frozen study input changed')
    proposal, order = read(proposal_ref), read(order_ref)
    task_reference = read(reference(audit/'full300-task-reference.json'))
    task_ref = task_reference['evaluation_manifest']
    require(reference(task_ref['path']) == task_ref and task_ref['sha256'] == PINNED_FULL300_SHA256,
            'Only exact established full300 cohort is permitted')
    rows = [json.loads(line) for line in Path(task_ref['path']).read_text().splitlines() if line.strip()]
    ids = [str(row['metadata']['task_id']) for row in rows]
    train_ids = list(order['original_manifest_task_order'])
    require(len(ids) == len(set(ids)) == 300 and len(train_ids) == len(set(train_ids)) == 2102
            and not set(ids) & set(train_ids) and task_reference['training_order'] == order_ref,
            'Task identities, count, or pinned training overlap differs')
    sources, manifests, workers, runtime_pins = {}, {}, {}, {}
    comparable, execution_inputs = {}, {}
    for arm in ARM_IDS:
        sources[arm] = reference(pilots/arm/'source-inventory.json')
        manifests[arm] = reference(pilots/arm/'manifest.json')
        workers[arm] = read(reference(pilots/arm/'worker.common.json'))
        runtime_pins[arm] = read(reference(pilots/arm/'runtime-pins.json'))
    source = source_pins(sources[ARM_IDS[0]])
    require(all(read(ref) == source for ref in sources.values()) and Path(source['source_root']) == runtime/'source',
            'All five pilots must use the exact common production source')
    pins = runtime_pins[ARM_IDS[0]]
    model_inputs = _model_inputs(model_ref, pins)
    parsed = read(native_arguments_reference)
    require(parsed['source_inventory'] == sources[ARM_IDS[0]] and parsed['native_argv'] == reference(runtime/'native-argv.json')
            and parsed['native_config'] == reference(runtime/'native-config.json')
            and parsed['verification_kind'] == 'partial_cpu_argument_validation'
            and parsed['cuda_initialized'] is False and parsed['physical_native_acceptance'] is False
            and parsed['browser_calls'] == parsed['judge_calls'] == 0,
            'Native arguments require exact isolated CPU provenance, never physical acceptance')
    native_args = parsed['native_args']
    require(set(generation_argument_names(source['source_root'])) <= set(native_args)
            and native_args['hf_checkpoint'] == model_inputs['model_root'], 'Native GenerateState arguments incomplete or model changed')
    prompt_sha = _judge_prompt(source)
    for arm in ARM_IDS:
        m = read(manifests[arm]); w = workers[arm]
        require(m['arm_id'] == arm and m['initialization']['iteration'] == 0
                and m['science']['checkpoint']['original_actor_inventory'] == model_ref
                and m['science']['judge']['prompt_sha256'] == prompt_sha
                and m['science']['judge']['canonical_reward_source_sha256'] == source['files']['openwebrl/reward_browser.py']
                and w['approval_reference'] is None and w['judge_protocol'] is None,
                'Expected five separately unapproved original-initialized draft pilots')
        comparable[arm], execution_inputs[arm] = comparable_runtime_pins(pilots, arm,
            runtime_pins[arm], w, m, sources[arm], runtime)
    require(len({value is not None for value in execution_inputs.values()}) == 1,
            'Cannot mix bound and unbound runtime preparations')
    require(all(value == comparable[ARM_IDS[0]] for value in comparable.values()),
            'Pilot runtime pins differ beyond verified private browser paths')
    builder = dict(campaign_id='action-credit-iter10-20261011-shared-verification',
        proposal_reference=proposal_ref, original_actor_inventory=model_ref, source_inventory=sources[ARM_IDS[0]],
        evaluation_task_manifest=task_ref, excluded_training_task_ids=train_ids,
        verification_task_ids=None, verification_protocol=None,
        shared_initial_max_attempts_per_task=None, verification_max_attempts_per_task=None,
        judge_prompt_sha256=prompt_sha, canonical_reward_source_sha256=source['files']['openwebrl/reward_browser.py'],
        approval_reference=None, judge_http_protocol=None)
    shared_manifest = None
    if verification_selection_reference is not None:
        selected = read(verification_selection_reference)
        fields = ('verification_task_ids','verification_protocol','shared_initial_max_attempts_per_task','verification_max_attempts_per_task')
        require(set(selected) == {'schema_version','task_order_reference','selection_authority_reference',*fields}
                and selected['schema_version'] == 1 and selected['task_order_reference'] == order_ref,
                'Explicit source-bound verification selection required')
        authority = read(selected['selection_authority_reference'])
        require(authority.get('selection') == {key:selected[key] for key in fields}
                and authority.get('user_authorization_reference') and authority.get('user_authorization_text'),
                'Verification selection authority differs; do not invent selected tasks')
        builder.update({key:selected[key] for key in fields})
        shared_manifest = prepare_verification_manifest(**builder).to_dict()
    missing = ['exact shared verification and five pilot API approvals with purpose-specific caps',
        'approved judge HTTP pricing/input bound/reservation protocol (judge output1024, actor output4096)',
        'fresh approved manifests and source assembly after all code changes',
        'production native all8 restoration/transfer acceptance, storage measurement and existing allocation binding',
        'committed current iteration10 checkpoint inventory for each pilot',
        'shared0 physical host binding before any pilot optimizer update']
    if shared_manifest is None:
        missing.insert(0, 'explicit verification task selection, protocol and per-task retry limits (no automatic selection)')
    files = {}
    def emit(name, value):
        raw = encoded(value); files[name] = raw
        return dict(path=str(output/name), sha256=hashlib.sha256(raw).hexdigest())
    shared_ref = emit('shared-verification/manifest.json', shared_manifest) if shared_manifest is not None else None
    template_ref = emit('shared-verification/manifest.template.json', dict(schema_version=1,
        status=STATUS, authorizes_execution=False, builder='openwebrl.action_credit_verification.prepare_verification_manifest',
        builder_arguments=builder, selection_reference=verification_selection_reference,
        proposal_limits=proposal['validation']['separately_proposed_online_verification']))
    shared_root = output/'shared-initial'
    shared_config_ref = emit('shared-initial/worker.configuration.json', worker_draft(shared_root, source, native_args, pins))
    arms = {}
    for arm in ARM_IDS:
        root = output/'arms'/arm
        config_ref = emit('arms/'+arm+'/worker.configuration.json', worker_draft(root, source, native_args, pins))
        arms[arm] = dict(pilot_manifest_draft=manifests[arm], approved_pilot_manifest=None,
            iter10_worker_configuration=config_ref, arm_evaluation_root=str(root/'cohorts'),
            committed_current_iteration10_checkpoint=None, accounting_manifest_sha256=None,
            execution_inputs=execution_inputs[arm])
    index = dict(schema_version=1, status=STATUS, authorizes_execution=False, stores_created=False,
        new_gpu_allocations=0, browser_calls=0, judge_calls=0, verification_manifest=shared_ref,
        verification_manifest_template=template_ref, verification_store_root=str(output/'shared-verification'/'ledger'),
        shared0_worker_configuration=shared_config_ref, shared_initial_root=str(shared_root/'cohorts'),
        shared0_host_binding=None, source_inventory=sources[ARM_IDS[0]], original_model_inputs=model_inputs,
        full300_task_manifest=task_ref, task_order=order_ref, proposal_reference=proposal_ref,
        parsed_native_arguments=native_arguments_reference, protocol=evaluation_protocol(),
        browser_artifacts={name:reference(Path(source['source_root'])/name) for name in (SERVER,PROMPT,TOOLS)},
        arms=arms, unresolved_required_inputs=missing)
    emit('index.json', index)
    return dict(output_root=str(output), files=files, index=index)


def write_drafts(plan):
    root = output_path(plan['output_root'])
    require(not root.exists(), 'Never overwrite existing drafts')
    # Validate every pinned preparation input again before the first mutation.
    index = plan['index']
    source_pins(index['source_inventory'])
    for key in ('proposal_reference','task_order','parsed_native_arguments'):
        read(index[key])
    parsed=read(index['parsed_native_arguments'])
    read(parsed['native_argv']);read(parsed['native_config'])
    _model = index['original_model_inputs']
    read(_model['original_actor_inventory'])
    for key in ('tokenizer_files','processor_files'):
        for ref in _model[key]: require(reference(ref['path']) == ref, 'Model input changed before write')
    require(reference(index['full300_task_manifest']['path']) == index['full300_task_manifest'], 'Full300 bytes changed before write')
    for arm in index['arms'].values():
        read(arm['pilot_manifest_draft'])
        if arm['execution_inputs'] is not None:
            receipt=read(arm['execution_inputs']['prepared'])
            read(arm['execution_inputs']['execution_binding'])
            pilot=Path(arm['pilot_manifest_draft']['path']).parent
            for name,sha in receipt['files'].items():
                require(reference(pilot/name)['sha256']==sha,'Prepared execution input changed before write')
    for name, content in plan['files'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'Draft output escaped root')
        output_path(root/path)
        require(isinstance(content, bytes), 'Exact serialized draft output required')
    for name, content in plan['files'].items(): write_exclusive(root/name, content)
    return reference(root/'index.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=AUDIT)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--pilots', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--native-arguments', type=Path)
    parser.add_argument('--parse-native-arguments', type=Path,
        help='Fresh directory for isolated existing CPU parser probe; no evaluation preparation in this mode')
    parser.add_argument('--verification-selection', type=Path)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args(argv)
    if args.parse_native_arguments:
        require(not args.prepare and not args.native_arguments, 'Parser probe is a separate CPU preparation action')
        result = parse_native_arguments(args.runtime,
            reference(args.pilots/ARM_IDS[0]/'source-inventory.json'), args.parse_native_arguments)
    else:
        require(args.native_arguments is not None, 'Existing pinned CPU native-argument snapshot required')
        plan = build_drafts(audit=args.audit, runtime=args.runtime, pilots=args.pilots, output_root=args.output_root,
            native_arguments_reference=reference(args.native_arguments),
            verification_selection_reference=reference(args.verification_selection) if args.verification_selection else None)
        result = write_drafts(plan) if args.prepare else plan['index']
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
