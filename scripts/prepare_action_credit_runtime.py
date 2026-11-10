#!/usr/bin/env python3
"""Prepare a private action-credit source fork and CPU-parse native inputs.

This is not a launcher. No scheduler, browser, actor, judge, paid ledger or
physical acceptance is created. --dry-run only reads; --prepare copies small
source files and --probe-native additionally runs the native argument parsers,
HF config checks, and CPU-safe Slime/SGLang/profile checks. Megatron's complete
validator queries CUDA device architecture and is explicitly deferred to the
physical gate; no device result is mocked. Existing outputs are never overwritten. The caller supplies an
explicit browser manifest and retry bound, rather than inheriting credentials
or choosing a new scientific replay protocol here.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
PARENT = RUNTIME/'reference-uniform8-original2102-20261006-v1'
AUDIT = RUNTIME/'training-controls/action-credit-implementation-20261011'
PYTHON = RUNTIME/'venv/bin/python'
FORBIDDEN = Path('/gpfs/home/zixianma')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO/'scripts') not in sys.path:
    sys.path.insert(0, str(REPO/'scripts'))

from prepare_action_credit_pilots import (CACHE_KEYS, SAMPLING, INPUT_NAMES,
    load_inputs, hash_file, encoded, no_credentials, validate_runtime)

GENERATOR = dict(max_steps=15, context_num_screenshots=1,
    turn_history_reasoning_mode='full', browser_history_reasoning_max_turns=0,
    browser_include_tool_response=1, browser_history_env_feedback_max_chars=4000,
    rollout_max_context_len=32768, max_consecutive_parse_failures=3,
    serial_action_variant=None, browser_response_format_mode='browser_env',
    browser_action_selector=None, browser_tool_response_max_chars=512)
NATIVE_CONFIG = dict(GENERATOR, debug_trace_prob=0., path_to_save_generated_samples='',
    inference_step_timeout_secs=30., rollout_task_timeout_secs=600.,
    browser_rollout_concurrency=64, log_task_start=True, log_llm_output=False,
    log_judge_output=False, browser_observation_memory=False,
    browser_state_advantage_mix=0.)
# These are controls validated by bootstrap before any native initialization.
NATIVE_PROFILE = dict(actor_num_nodes=1, actor_num_gpus_per_node=8, num_gpus_per_node=8,
    rollout_num_gpus=8, rollout_num_gpus_per_engine=2, tensor_model_parallel_size=2,
    pipeline_model_parallel_size=1, context_parallel_size=1, global_batch_size=256,
    micro_batch_size=1, colocate=True, offload_train=False, offload_rollout=False,
    debug_train_only=False, debug_rollout_only=False, use_critic=False, use_opd=False,
    use_fault_tolerance=False, num_rollout=10, data_parallel_random_init=True,
    lr=1e-6, adam_beta1=.9, adam_beta2=.98, weight_decay=.1, lr_decay_style='constant',
    kl_coef=0., keep_old_actor=False, megatron_to_hf_mode='bridge', rollout_external=False,
    sglang_config=None, prefill_num_servers=None, wandb_project='openwebrl', use_wandb=True,
    ppo_epochs=2, seed=42, rollout_temperature=.8, rollout_top_p=1., rollout_top_k=-1,
    rollout_max_response_len=1024, loss_type='custom_loss',
    custom_loss_function_path='openwebrl.action_credit_megatron.loss')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def output_path(path):
    lexical = Path(path).absolute()
    resolved = lexical.resolve()
    require(not lexical.is_relative_to(FORBIDDEN) and not resolved.is_relative_to(FORBIDDEN),
            'Home writes are forbidden')
    require(resolved.is_relative_to(RUNTIME) or resolved.is_relative_to(Path('/tmp')),
            'Private preparation outputs must be in runtime or /tmp')
    require(lexical == resolved, 'Output paths must not traverse symlinks')
    return resolved


def source_file(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Contained relative source path required')
    path = root/relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
            'Source file missing, symlinked or escaped: '+str(path))
    require(path.stat().st_size < 16*1024**2, 'Source payload is unexpectedly large: '+str(path))
    return path


def module_file(repo, module):
    if not module or module.split('.')[0] not in ('openwebrl', 'scripts'):
        return None
    base = Path(*module.split('.'))
    for name in (base.with_suffix('.py'), base/'__init__.py'):
        if (repo/name).is_file():
            return name.as_posix()
    return None


def local_dependencies(repo, seeds):
    """Transitive static local imports, including imports inside functions.

    Dynamic module names and browser data files need explicit roots below; this
    inventory is not a claim to have executed every optional plugin path.
    """
    pending = list(seeds)
    found = set()
    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        path = source_file(repo, relative)
        found.add(relative)
        if path.suffix != '.py':
            continue
        parts = list(Path(relative).with_suffix('').parts)
        package = parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(), filename=relative)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = '.'.join(package[:len(package)-node.level+1]) if node.level else ''
                base = '.'.join(x for x in (prefix, node.module) if x)
                modules = [base]+[base+'.'+item.name for item in node.names]
            for module in modules:
                candidate = module_file(repo, module)
                if candidate and candidate not in found:
                    pending.append(candidate)
    return found


def inventory(parent=PARENT, repo=REPO):
    """Verify native parent and pin a complete small native/browser code fork."""
    parent, repo = Path(parent).resolve(), Path(repo).resolve()
    require('native-validation-v3' not in parent.parts, 'Diagnostic source is not a production parent')
    manifest = json.loads(source_file(parent, 'reference_manifest.json').read_text())
    for relative, expected in manifest['recipe_files_sha256'].items():
        require(hash_file(source_file(parent, relative)) == expected, 'Preserved parent source changed: '+relative)
    for relative in ('train.py', 'slime/backends/megatron_utils/actor.py'):
        require('OPENWEBRL_ACTION_CREDIT_NATIVE_GATE' not in source_file(parent, relative).read_text(),
                'Diagnostic dispatch must not become the production path')
    paths = {}
    # Entire native packages are small and contain runtime-loaded model/plugins.
    # Exclude inherited datasets, receipts, caches, histories and environment files.
    for directory in ('slime', 'slime_plugins', 'openwebrl'):
        for path in sorted((parent/directory).rglob('*')):
            relative = path.relative_to(parent)
            if (path.is_file() and '__pycache__' not in relative.parts
                    and 'data' not in relative.parts and path.suffix in ('.py', '.yaml', '.yml', '.json', '.md')):
                paths[relative.as_posix()] = source_file(parent, relative)
    for relative in ('train.py', 'scripts/h200_env.sh', 'scripts/run_h200_browser.sh',
                     'scripts/model_configs/qwen3-4B.sh'):
        paths[relative] = source_file(parent, relative)
    seeds = [path.relative_to(repo).as_posix() for path in (repo/'openwebrl').glob('action_credit*.py')]
    seeds += ['openwebrl/generate_browser.py', 'openwebrl/reward_browser.py',
              'openwebrl/arm_branch_browser.py', 'openwebrl/arm_branch_protocol.py',
              'openwebrl/rl_recipe.py', 'openwebrl/controlled_sft_worker.py',
              'openwebrl/controlled_browser_server.py', 'openwebrl/response_format.py',
              'openwebrl/eval_monitor.py', 'scripts/check_storage_quota.py',
              'scripts/prepare_action_credit_runtime.py', 'scripts/prepare_action_credit_pilots.py']
    # These packages provide dynamically imported environment backends/resources.
    for directory in ('openwebrl/env', 'openwebrl/base', 'openwebrl/adapters'):
        seeds += [p.relative_to(repo).as_posix() for p in (repo/directory).rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.md', '.yaml')]
    for relative in local_dependencies(repo, seeds):
        paths[relative] = source_file(repo, relative)
    result = {relative: dict(path=str(path), sha256=hash_file(path), size_bytes=path.stat().st_size)
              for relative, path in sorted(paths.items())}
    require(sum(row['size_bytes'] for row in result.values()) < 32*1024**2,
            'Unexpected source payload; inspect before copying')
    return result


def base_environment(control, source):
    """No inherited credentials, HOME override, secret file or experiment flags."""
    control, source = output_path(control), Path(source).resolve()
    env = {key: os.environ[key] for key in ('PATH', 'LANG', 'LC_ALL', 'USER') if key in os.environ}
    env.update({key: str(control/'cache'/key.lower()) for key in CACHE_KEYS})
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled', WANDB_PROJECT='openwebrl-evals',
        TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
        OPENWEBRL_RUNTIME_ROOT=str(RUNTIME), SLIME_ASYNC_USE_STDLIB_LOOP='1',
        PYTHONPATH=str(source)+':'+str(RUNTIME/'src/Megatron-LM'),
        LD_LIBRARY_PATH=str(RUNTIME/'venv/lib/python3.12/site-packages/torch/lib'))
    for path in sorted((RUNTIME/'venv/lib/python3.12/site-packages/nvidia').glob('*/lib')):
        env['LD_LIBRARY_PATH'] += ':'+str(path)
    return env


def external_runtime_inventory(runtime=RUNTIME):
    """Pin imported external Python sources and installed package RECORD files.

    RECORD bytes identify installed binary distributions but do not independently
    hash multi-gigabyte binary payloads. This distinction stays in the receipt.
    Runtime startup must verify this inventory again before importing workers.
    """
    runtime = Path(runtime).resolve()
    site = runtime/'venv/lib/python3.12/site-packages'
    roots = [runtime/'src/Megatron-LM/megatron', site/'sglang', site/'ray']
    paths = []
    for root in roots:
        require(root.is_dir(), 'Native external dependency missing: '+str(root))
        paths += sorted(root.rglob('*.py'))
    paths += sorted(site.glob('*.pth')) + sorted(site.glob('__editable___*_finder.py'))
    for package in ('torch', 'transformers', 'transformer_engine', 'sglang', 'ray', 'megatron_core',
                    'sgl_kernel', 'flash_attn', 'flashinfer_python', 'playwright'):
        paths += sorted(site.glob(package+'-*.dist-info/RECORD'))
        paths += sorted(site.glob(package+'-*.dist-info/METADATA'))
    files = {}
    for path in paths:
        require(path.resolve().is_relative_to(runtime), 'External dependency escaped runtime')
        require(not path.is_symlink(), 'Pin concrete external source rather than a symlink')
        files[str(path)] = dict(sha256=hash_file(path), size_bytes=path.stat().st_size)
    return dict(schema_version=1, external_python_sources_and_distribution_records=files,
        source_root_paths=list(map(str, roots)), runtime_root=str(runtime),
        binary_payloads_independently_rehashed=False,
        production_requirement='Revalidate this complete inventory before importing the native startup/worker packages')


def verify_external_runtime(inventory):
    root = Path(inventory['runtime_root']).resolve()
    for name, expected in inventory['external_python_sources_and_distribution_records'].items():
        path = Path(name)
        require(path.resolve().is_relative_to(root) and not path.is_symlink(),
                'External dependency moved or escaped')
        require(path.stat().st_size == expected['size_bytes'] and hash_file(path) == expected['sha256'],
                'External native source/distribution changed: '+name)


def native_command(parent, source, control, model, dataset):
    """Read-only native shell expansion, followed by explicit profile changes.

    The returned argv is parser input. It must never be executed as train.py:
    production startup is owned by action_credit_bootstrap plus real owners.
    """
    env = base_environment(control, parent)
    env.update(DRY_RUN='1', NUM_GPUS='8', TP_SIZE='2', NUM_ROLLOUT='10',
        HF_CHECKPOINT=str(model), SAVE_DIR=str(control/'native-checkpoints-TEMPLATE'),
        SAVE_INTERVAL='1', TRAIN_DATA=str(dataset), BROWSER_TRAIN_CONFIG=str(control/'native-config.json'),
        BROWSER_MAX_STEPS='15', ROLLOUT_BATCH_SIZE='48', N_SAMPLES='8',
        GLOBAL_BATCH_SIZE='256', CONTEXT_LEN='32768', RESPONSE_LEN='1024',
        BROWSER_CONCURRENCY='64', SGLANG_CONCURRENCY='16', LEARNING_RATE='1e-6',
        RECOMPUTE_ACTIVATIONS='1', JUDGE_API_MODE='served')
    result = subprocess.run(['bash', str(parent/'scripts/run_h200_browser.sh')], env=env,
        cwd=parent, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True)
    command = shlex.split(result.stdout)
    require(command[:2] == ['python', 'train.py'], 'Unexpected native dry-run command')
    args = command[2:]
    def drop(flag, takes_value=False):
        require(args.count(flag) <= 1, 'Duplicate inherited argument '+flag)
        if flag in args:
            i = args.index(flag)
            del args[i:i+1+int(takes_value)]
    drop('--dynamic-sampling-filter-path', True)
    drop('--rollout-shuffle')
    drop('--use-rollout-logprobs')
    drop('--rollout-num-gpus-per-engine', True)
    # Explicit negative flags survive the generic colocate validator.
    args += ['--no-offload-train', '--no-offload-rollout', '--rollout-num-gpus', '8',
        '--rollout-num-gpus-per-engine', '2', '--seed', '42', '--data-parallel-random-init',
        '--rollout-top-p', '1', '--rollout-top-k', '-1', '--loss-type', 'custom_loss',
        '--custom-loss-function-path', 'openwebrl.action_credit_megatron.loss',
        '--clip-grad', '1', '--skip-eval-before-train', '--use-wandb', '--wandb-project', 'openwebrl']
    require(not any(flag in args for flag in ('--offload', '--offload-train', '--offload-rollout',
        '--debug-train-only', '--debug-rollout-only', '--load-debug-rollout-data')),
        'Unexpected offload, diagnostic or replay argument')
    return args


def build_plan(control, browser_manifest, *, max_attempts_per_draw, parent=PARENT,
               repo=REPO, audit=AUDIT, expected_inputs=None):
    control = output_path(control)
    require(not control.exists(), 'Use a fresh output root; existing attempts are preserved')
    require(type(max_attempts_per_draw) is int and max_attempts_per_draw > 0, 'Explicit positive retry bound required')
    options = {} if expected_inputs is None else dict(expected_inputs=expected_inputs)
    values, refs = load_inputs(audit, **options)
    model = values[INPUT_NAMES[5]]
    common = values[INPUT_NAMES[0]]['common_science_recommendation']
    require(hash_file(common['train_data']) == common['train_data_sha256'], 'Training input changed')
    files = inventory(parent, repo)
    source = control/'source'
    browser = dict(browser_manifest)
    no_credentials(browser)
    from openwebrl.controlled_browser_server import validate_browser_manifest
    validate_browser_manifest(dict(browser, artifact_directory=str(control/'browser-artifacts-check')))
    require(browser.get('branch_state_version') in ('observable-replay-v1', 'visible-replay-v2'),
            'Explicit replay protocol required')
    browser.pop('artifact_directory', None)
    config = dict(mode='local_process', width=1280, height=720, dpr=1,
        use_screenshot=True, use_a11ytree=False, resize_output_coords=True, resize_scale=1000,
        path_to_policy=str(source/'openwebrl/env/prompts/system_prompt_browser_env.md'),
        path_to_tool_list=str(source/'openwebrl/env/prompts/tool_list.json'),
        local_process=dict(server_module='openwebrl.arm_branch_browser',
            server_module_sha256=files['openwebrl/arm_branch_browser.py']['sha256'],
            python_bin=str(PYTHON), max_processes=64, startup_timeout_secs=60,
            log_dir=str(control/'browser-logs'), port_lock_dir=str(control/'browser-locks')))
    names = dict(tokenizer={'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                            'chat_template.json', 'chat_template.jinja'},
                 processor={'config.json', 'processor_config.json', 'preprocessor_config.json',
                            'video_preprocessor_config.json'})
    resources = {}
    for kind, subset in names.items():
        resources[kind+'_files'] = [{k: row[k] for k in ('path', 'sha256')}
            for row in model['files'] if Path(row['path']).name in subset]
        require(resources[kind+'_files'], 'Original model input inventory is empty')
        for row in resources[kind+'_files']:
            require(hash_file(row['path']) == row['sha256'], 'Original model input changed')
    dependencies = external_runtime_inventory()
    pins = dict(schema_version=1, production_source_root=str(source),
        entrypoint_files=sorted([*files, 'action-credit-runtime-dependencies.json']),
        **resources, native_args=dict(GENERATOR, hf_checkpoint=model['model_root']),
        browser_config=config, browser_manifest=browser, sampling=SAMPLING,
        max_attempts_per_draw=max_attempts_per_draw)
    argv = native_command(Path(parent), source, control, model['model_root'], common['train_data'])
    return dict(schema_version=1, control=str(control), parent=str(Path(parent).resolve()),
        parent_manifest_sha256=hash_file(Path(parent)/'reference_manifest.json'),
        source_files=files, external_runtime=dependencies, input_references=refs, runtime_pins=pins,
        native_argv=argv, native_config=NATIVE_CONFIG, native_profile=NATIVE_PROFILE,
        native_parser_passed=False, production_runtime_validated=False, launch_ready=False,
        authorizes_execution=False, paid_ledgers_created=False,
        limitations=['CPU source and argument preparation is not physical acceptance',
            'Owning production controller must supply actual evaluation, storage, allocation and approval owners',
            'Output/cache paths must be bound privately per arm before immutable pilot preparation',
            'Original large model weights must be revalidated by native initialization',
            'Native startup must revalidate the pinned external Megatron/Ray/SGLang sources and distribution records',
            'Distribution RECORD files are pinned; installed multi-gigabyte binary payloads were not independently rehashed'])


def write_exclusive(path, data):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(plan):
    control = output_path(plan['control'])
    require(not control.exists(), 'Never overwrite an existing preparation')
    # Check the entire immutable input set before creating any destination.
    for row in plan['source_files'].values():
        require(hash_file(row['path']) == row['sha256'], 'Source changed since planning')
    for ref in plan['input_references'].values():
        require(hash_file(ref['path']) == ref['sha256'], 'Study input changed since planning')
    verify_external_runtime(plan['external_runtime'])
    control.mkdir(parents=True)
    for relative, row in plan['source_files'].items():
        content = Path(row['path']).read_bytes()
        require(hashlib.sha256(content).hexdigest() == row['sha256'], 'Source changed during copy')
        write_exclusive(control/'source'/relative, content)
    for name, value in (('preparation.json', plan), ('runtime-pins.json', plan['runtime_pins']),
                        ('native-argv.json', plan['native_argv']), ('native-config.json', plan['native_config'])):
        write_exclusive(control/name, encoded(value))
    write_exclusive(control/'source/action-credit-runtime-dependencies.json', encoded(plan['external_runtime']))
    model = json.loads(Path(plan['input_references'][INPUT_NAMES[5]]['path']).read_text())
    validate_runtime(plan['runtime_pins'], model)
    return control


def parse_cpu_native_arguments(argv):
    """Use native parser components; deliberately do not call hardware validation.

    The full Megatron validator is deferred as a whole. Passing this component
    check is not equivalent to passing native parse_args or physical startup.
    """
    from slime.utils import arguments as native
    from slime.backends.megatron_utils.arguments import megatron_parse_args
    previous = sys.argv
    try:
        sys.argv = ['action-credit-CPU-argument-parse', *argv]
        native.configure_logger()
        provider = native.get_slime_extra_args_provider()
        pre = native._pre_parse_mode()
        require(pre.train_backend == 'megatron' and not pre.debug_train_only
                and not pre.debug_rollout_only and pre.load_debug_rollout_data is None,
                'Production arguments must not use a diagnostic/replay parsing shortcut')
        sglang = native.sglang_parse_args()
        args = megatron_parse_args(extra_args_provider=provider, skip_hf_validate=False)
        for namespace in (pre, sglang):
            for key, value in vars(namespace).items():
                setattr(args, key, value)
        native.slime_validate_args(args)
        native.sglang_validate_args(args)
        return args
    finally:
        sys.argv = previous


def native_parse_probe(control):
    """Called in the isolated source's CPU process; never imports train.py."""
    control = output_path(control)
    plan = json.loads((control/'preparation.json').read_text())
    require(Path(plan['runtime_pins']['production_source_root']).resolve() == REPO,
            'Probe must execute from the exact prepared source')
    for relative, row in plan['source_files'].items():
        require(hash_file(REPO/relative) == row['sha256'], 'Prepared source changed before native parse')
    require(json.loads((REPO/'action-credit-runtime-dependencies.json').read_text()) == plan['external_runtime'],
            'Prepared external dependency inventory changed')
    verify_external_runtime(plan['external_runtime'])
    from openwebrl.action_credit_bootstrap import verify_native_runtime_sources
    verify_native_runtime_sources(dict(source_root=str(REPO),files={
        **{name:row['sha256'] for name,row in plan['source_files'].items()},
        'action-credit-runtime-dependencies.json':hash_file(REPO/'action-credit-runtime-dependencies.json')}))
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('WANDB_MODE') == 'disabled',
            'Probe must remain CPU-only and offline')
    argv = json.loads((control/'native-argv.json').read_text())
    require(argv == plan['native_argv'], 'Native argv differs from preparation')
    require(json.loads((control/'native-config.json').read_text()) == plan['native_config'],
            'Native custom config differs from preparation')
    args = parse_cpu_native_arguments(argv)
    for key, expected in plan['native_profile'].items():
        require(getattr(args, key, None) == expected, 'Parsed native profile differs: '+key)
    for key, expected in plan['runtime_pins']['native_args'].items():
        require(hasattr(args, key) and getattr(args, key) == expected, 'Parsed generator input differs: '+key)
    require(Path(args.load).resolve() == Path(args.hf_checkpoint).resolve(), 'Initial native load must be original HF actor')
    require(getattr(args, 'dynamic_sampling_filter_path', None) is None and not args.rollout_shuffle,
            'Fixed-task collector must not inherit dynamic filtering/shuffling')
    from openwebrl.action_credit_bootstrap import validate_native_profile
    assembly = SimpleNamespace(worker={'native_args': plan['runtime_pins']['native_args']},
        store=SimpleNamespace(manifest=SimpleNamespace(to_dict=lambda: {
            'science': {'checkpoint': {'original_actor_path': args.hf_checkpoint}}})))
    validate_native_profile(args, assembly)
    import torch
    require(not torch.cuda.is_initialized(), 'CPU parser unexpectedly initialized CUDA')
    result = dict(passed=True, verification_kind='partial_cpu_argument_validation',
        partial_cpu_argument_validation=True, full_native_validation_passed=False,
        deferred_validation='Entire Megatron validate_args: includes CUDA device architecture check for TP2',
        preparation_sha256=hash_file(control/'preparation.json'), parsed_profile=plan['native_profile'],
        parsed_generator=plan['runtime_pins']['native_args'], cuda_initialized=False,
        physical_native_acceptance=False, launch_ready=False, browser_calls=0, judge_calls=0)
    write_exclusive(control/'native-cpu-arguments-review.json', encoded(result))
    return result


def probe(control):
    control = output_path(control)
    env = base_environment(control, control/'source')
    for key in CACHE_KEYS:
        output_path(env[key]).mkdir(parents=True, exist_ok=True)
    with (control/'native-parse.log').open('xb') as log:
        subprocess.run([str(PYTHON), '-B', str(control/'source/scripts/prepare_action_credit_runtime.py'),
            '--native-parse-probe', str(control)], cwd=control/'source', env=env,
            stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
    return json.loads((control/'native-cpu-arguments-review.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-parse-probe', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--browser-manifest', type=Path)
    parser.add_argument('--max-attempts-per-draw', type=int)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--prepare', action='store_true')
    parser.add_argument('--probe-native', action='store_true')
    options = parser.parse_args()
    if options.native_parse_probe:
        native_parse_probe(options.native_parse_probe)
        return
    require(options.output and options.browser_manifest and options.max_attempts_per_draw,
            'Explicit output, browser manifest and retry bound are required')
    require(options.dry_run or options.prepare, 'Select --dry-run or --prepare')
    require(not options.probe_native or options.prepare, 'CPU parse requires an immutable prepared source')
    plan = build_plan(options.output, json.loads(options.browser_manifest.read_text()),
                      max_attempts_per_draw=options.max_attempts_per_draw)
    if options.prepare:
        prepare(plan)
        if options.probe_native:
            probe(options.output)
    print(json.dumps(dict(output=plan['control'], source_files=len(plan['source_files']),
        source_bytes=sum(row['size_bytes'] for row in plan['source_files'].values()),
        prepared=options.prepare, partial_cpu_argument_validation=options.probe_native,
        full_native_validation_passed=False,
        launch_ready=False, authorizes_execution=False), indent=2))


if __name__ == '__main__':
    main()
