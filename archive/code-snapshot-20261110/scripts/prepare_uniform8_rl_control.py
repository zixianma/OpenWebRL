#!/usr/bin/env python3
"""Prepare the original-2102-task uniform-G8 pilot without submitting jobs.

The adaptive G5->G8 experiment is paused at the user's request. This preparer
freezes only the uniform-G8 outcome-RL control and exact resource proposal.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
PARENT = RUNTIME/'reference-expanded-outcome-4102-to90-20261005-v1'
SOURCE = RUNTIME/'reference-uniform8-original2102-20261006-v1'
CONTROL = RUNTIME/'training-controls/uniform8-original2102-20261006'
RUN_ID = 'outcome-uniform8-original2102-20261006'
RUN = RUNTIME/'runs'/RUN_ID
DATA = REPO/'openwebrl/data/webgym_filtered_popular_2102_cleaned.parquet'
MODEL = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
RESOURCES = dict(jobs=1, gpus=8, gpu_type='H200', hours=16, gpu_hours=128,
                 cpus=64, memory_gib=960)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Unrecognized parent source: '+before[:100])
    return text.replace(before, after, 1)


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def recipe_source():
    """Private controller module: fixed scientific recipe, new run identity."""
    return f'''import os
from pathlib import Path
REPO=Path({str(REPO)!r})
RUNTIME=Path({str(RUNTIME)!r})
CONTROL=Path({str(CONTROL)!r})
SOURCE=Path({str(SOURCE)!r})
RUN=Path({str(RUN)!r})
RUN_ID={RUN_ID!r}
PYTHON=RUNTIME/'venv/bin/python'
TARGET_ITERATION=10
EVAL_ITERATIONS=(10,)
TRAINING_STAGES=(1,10)
MODEL=Path({str(MODEL)!r})
import hashlib
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean_env():
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('OPENWEBRL_','ARM_','SLIME_','WANDB_')):
            env.pop(key,None)
    for key in ('DRY_RUN','OVERRIDE_OPT_PARAM_SCHEDULER'):
        env.pop(key,None)
    return env
def training_command(target, *, resume=False):
    if target not in TRAINING_STAGES: raise ValueError('Unexpected stage')
    env=dict(NUM_GPUS='8',TP_SIZE='2',NUM_ROLLOUT='10',
        OPENWEBRL_STOP_AFTER_SAVED_ROLLOUT=str(target-1),
        TRAIN_DATA=str(CONTROL/'combined-tasks.parquet'),HF_CHECKPOINT=str(MODEL),
        BROWSER_TRAIN_CONFIG=str(SOURCE/'openwebrl/browser_training_config.yaml'),
        BROWSER_MAX_STEPS='15',ROLLOUT_BATCH_SIZE='48',N_SAMPLES='8',
        GLOBAL_BATCH_SIZE='256',CONTEXT_LEN='32768',RESPONSE_LEN='1024',
        BROWSER_CONCURRENCY='64',SGLANG_CONCURRENCY='64',LEARNING_RATE='1e-6',
        RECOMPUTE_ACTIVATIONS='1',SAVE_INTERVAL='1',SAVE_DIR=str(RUN),
        WANDB_MODE='online',WANDB_PROJECT='openwebrl',WANDB_RUN_ID=RUN_ID,
        WANDB_RESUME='must' if resume else 'never',
        JUDGE_MODEL='gpt-4.1',JUDGE_API_MODE='served',JUDGE_API_BASE='https://api.openai.com/v1',
        OPENWEBRL_CUDA_CACHE_LIMIT_GIB='48',OMP_NUM_THREADS='2',
        FLASHINFER_WORKSPACE_BASE=str(RUNTIME),RAY_ADDRESS='local',
        RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=str(8*1024**3),
        OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(RUN/'multimodal'),
        OPENWEBRL_SAMPLING_CONTROL=str(CONTROL),
        SLIME_ADAPTIVE_QUERY_BLACKLIST_PATH=str(SOURCE/'reference_empty_blacklist.txt'),
        SLIME_BROWSER_QUERY_BLACKLIST_PATH=str(SOURCE/'reference_empty_blacklist.txt'))
    command=['bash',str(SOURCE/'scripts/run_h200_browser.sh'),
        '--use-wandb','--wandb-mode','online','--wandb-project','openwebrl',
        '--wandb-team','zixianma','--wandb-group',RUN_ID,
        '--disable-wandb-random-suffix','--wandb-dir',str(RUN/'wandb'),
        '--sglang-disable-cuda-graph','--rollout-health-check-first-wait','180',
        '--use-fault-tolerance','--router-balance-abs-threshold','2',
        '--skip-eval-before-train','--seed','42','--rollout-top-p','1','--rollout-top-k','-1',
        '--save-debug-rollout-data',str(RUN/'rollout_recovery/{{rollout_id}}.pt')]
    if resume:
        env['SLIME_LOAD_CHECKPOINT']=str(RUN)
        command += ['--use-checkpoint-opt-param-scheduler']
    return dict(command=command,environment=env,target_iteration=target,
                fresh_iteration0=not resume,tp=2,dp=4,eval_owned_by_controller=True)
'''


def prepare():
    from prepare_arm_turn_bonus import copy_plain
    from resume_baseline import validate_source
    import pyarrow.parquet as pq
    rows = pq.read_table(DATA).to_pylist()
    assert len(rows) == 2102 and len({r['metadata']['task_id'] for r in rows}) == 2102
    validate_source(PARENT)
    CONTROL.mkdir(parents=True, exist_ok=True)
    approval_path = CONTROL/'approval-request.json'
    if approval_path.exists() and json.loads(approval_path.read_text()).get('approved'):
        raise ValueError('Do not reprepare an approved campaign')
    changes = {'openwebrl/rollout_sampling_budget.py': (REPO/'openwebrl/rollout_sampling_budget.py').read_bytes()}
    name = 'openwebrl/reward_browser.py'
    changes[name] = replace_once((PARENT/name).read_text(),
        'from openwebrl.expanded_baseline_budget import wrap',
        'from openwebrl.rollout_sampling_budget import wrap').encode()
    name = 'slime/rollout/sglang_rollout.py'
    changes[name] = replace_once((PARENT/name).read_text(),
        '                custom_generate_func = load_function(custom_func_path)',
        '                from openwebrl.rollout_sampling_budget import reserve_browser\n'
        '                await asyncio.to_thread(reserve_browser, sample_index=sample.index,\n'
        '                                        group_index=sample.group_index, evaluation=evaluation)\n'
        '                custom_generate_func = load_function(custom_func_path)').encode()
    if not SOURCE.exists():
        shutil.copytree(PARENT, SOURCE, symlinks=True, copy_function=copy_plain,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.browser_use_sessions'))
        manifest = json.loads((PARENT/'reference_manifest.json').read_text())
        for name, data in changes.items():
            compile(data, name, 'exec')
            (SOURCE/name).write_bytes(data)
            manifest['recipe_files_sha256'][name] = hashlib.sha256(data).hexdigest()
        manifest['uniform8_control'] = dict(parent=str(PARENT), changed_files=list(changes),
            scientific_change='Uniform eight trajectories instead of five; original 2102-task pool',
            budget_hooks_only=True, adaptive_sampling=False)
        save(SOURCE/'reference_manifest.json', manifest)
    for name, data in changes.items():
        assert (SOURCE/name).read_bytes() == data
    validate_source(SOURCE)
    for name in ['slime/ray/rollout.py','slime/backends/megatron_utils/loss.py',
                 'slime/rollout/filter_hub/dynamic_sampling_filters.py',
                 'openwebrl/generate_browser.py','slime/rollout/data_source.py']:
        assert sha(PARENT/name) == sha(SOURCE/name)
    (CONTROL/'campaign_recipe.py').write_text(recipe_source())
    import importlib.util
    spec=importlib.util.spec_from_file_location('uniform8_campaign_recipe',CONTROL/'campaign_recipe.py')
    recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
    controller=(REPO/'scripts/run_expanded_baseline.py').read_text()
    controller=controller.replace('from prepare_expanded_baseline import','from campaign_recipe import')
    controller=controller.replace('hours=24,gpu_hours=192','hours=16,gpu_hours=128')
    controller=controller.replace("approval['judge_cap_usd'] != 200", "approval['judge_cap_usd'] != 50")
    controller=controller.replace('Exact8H200x24h/64CPU/960GiB/$200','Exact8H200x16h/64CPU/960GiB/$50')
    controller=controller.replace('86400','57600').replace('maximum_hours=24','maximum_hours=16')
    controller=controller.replace('allocation_hours_cap=24','allocation_hours_cap=16')
    controller=controller.replace('OPENWEBRL_EXPANDED_BASELINE_CONTROL','OPENWEBRL_SAMPLING_CONTROL')
    controller=controller.replace('fresh4102-task training to60','fresh2102-task uniform-G8 training to10')
    controller=controller.replace('original24h GPU and$200','original16h GPU and$50')
    controller=controller.replace('fresh24h','fresh16h').replace('fresh outcome-only expanded4102 baseline','fresh uniform-G8 original2102 baseline')
    controller=controller.replace('        if deadline-time.time()', '        if deadline-time.time()')
    controller=replace_once(controller,
        "    if approval.get('target_iteration') != TARGET_ITERATION",
        "    if (approval.get('judge_call_cap') != 6000 or approval.get('browser_dispatch_cap') != 12000\n"
        "            or approval.get('browser_concurrency') != 64):\n"
        "        raise PermissionError('Exact API/browser limits are required')\n"
        "    if approval.get('target_iteration') != TARGET_ITERATION")
    # Imported support utilities are the project implementations, not old copies
    # inherited through the preserved source tree.
    controller=replace_once(controller,'from campaign_recipe import',
        f'sys.path.insert(0, {str(REPO/"scripts")!r})\n\nfrom campaign_recipe import')
    compile(controller,str(CONTROL/'training-controller.py'),'exec')
    (CONTROL/'training-controller.py').write_text(controller)
    link=CONTROL/'combined-tasks.parquet'
    if not link.exists(): link.symlink_to(DATA)
    assert link.resolve()==DATA.resolve()
    native=[]
    for target in recipe.TRAINING_STAGES:
        p=recipe.training_command(target,resume=target>1)
        env=recipe.clean_env();env.update(p['environment'],DRY_RUN='1',PYTHONDONTWRITEBYTECODE='1')
        argv=shlex.split(subprocess.check_output(p['command'],env=env,cwd=SOURCE,text=True))
        expected={'--n-samples-per-prompt':'8','--rollout-batch-size':'48','--tensor-model-parallel-size':'2',
            '--actor-num-gpus-per-node':'8','--global-batch-size':'256','--micro-batch-size':'1',
            '--ppo-epochs':'2','--lr':'1e-6','--num-rollout':'10','--max-steps':'15',
            '--rollout-temperature':'0.8','--rollout-top-p':'1','--rollout-top-k':'-1',
            '--rollout-max-response-len':'1024','--rollout-max-context-len':'32768',
            '--judge-api-model':'gpt-4.1','--judge-prompt-variant':'action_history',
            '--prompt-data':str(link),'--load':str(MODEL if target==1 else RUN)}
        for key,value in expected.items():
            assert argv.count(key)==1 and argv[argv.index(key)+1]==value,(key,value)
        assert not any('arm_' in x for x in argv)
        native.append(dict(target=target,plan=p,argv=argv))
    approval=dict(approved=False,submitted=False,resources=RESOURCES,target_iteration=10,
        full300_evaluations=[10],judge_cap_usd=50,judge_call_cap=6000,
        browser_dispatch_cap=12000,browser_concurrency=64,
        includes_all_startup_retries_and_evaluations=True,unused_prior_budgets_added=False,
        adaptive_run_status='paused_by_user',
        approval_reason='AGENTS.md requires exact approval before a new allocation; experiment intent is authorized.')
    save(approval_path,approval)
    frozen={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*')
        if p.is_file() and not p.is_symlink() and p.suffix in {'.py','.sh','.yaml','.jsonl'}
        and not any(x in {'data','docs','__pycache__'} for x in p.relative_to(SOURCE).parts)}
    report=dict(source=str(SOURCE),parent_source=str(PARENT),data_path=str(DATA),data_sha256=sha(DATA),
        source_compilation_passed=True,native_launcher_dry_runs_passed=True,
        frozen_file_hashes=frozen,native_plans=native,compute_approved=False,
        gpu_initialization_and_full_checkpoint_reload_pending=True,
        task_count=2102,initialization='Official SFT iteration0; fresh optimizer/scheduler/cursor',
        acceptance='Native nonempty, nonzero reward std; 48 admitted groups per collection',
        comparison='Historical original2102 G5 and ARM action-reward runs; not a contemporaneous matched-compute ARM rescue comparison',
        endpoints='Actor-only full300 after iteration10; compare optimizer updates, tokens and GPU-hours as well as iteration',
        preserved_recipe='Outcome-only GRPO; no selector, action reward or auxiliary imitation loss',
        preparation_only=True)
    save(CONTROL/'launcher-preparation.json',report)
    batch=f'''#!/bin/bash
#SBATCH --job-name=owrl-uniform8
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#SBATCH --time=16:00:00
#SBATCH --no-requeue
#SBATCH --output={CONTROL}/slurm-%j.out
set -euo pipefail
cd {REPO}
exec {RUNTIME}/venv/bin/python -B {CONTROL}/training-controller.py --job-id "$SLURM_JOB_ID" --execute
'''
    (CONTROL/'launch.sbatch').write_text(batch)
    subprocess.run(['bash','-n',str(CONTROL/'launch.sbatch')],check=True)
    save(CONTROL/'cpu-preflight.json',dict(passed=True,native_dry_runs=True,task_count_verified=True,
        controller_compilation=True,batch_syntax=True,
        launcher_file_hashes={str(p):sha(p) for p in [Path(__file__).resolve(),CONTROL/'campaign_recipe.py',
            CONTROL/'training-controller.py',CONTROL/'launch.sbatch',REPO/'openwebrl/rollout_sampling_budget.py']},
        gpu_validation_pending=True))
    print(json.dumps(dict(control=str(CONTROL),source=str(SOURCE),approved=False,submitted=False,
                         tasks=2102,resources=RESOURCES,target=10,adaptive_run='paused')))


if __name__=='__main__':
    prepare()
