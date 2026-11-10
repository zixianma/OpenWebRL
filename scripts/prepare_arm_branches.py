#!/usr/bin/env python3
"""Prepare an immutable continuation pilot with an explicitly unapproved budget."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO=Path(__file__).resolve().parents[1]
RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
sys.path.insert(0,str(REPO))
from openwebrl.arm_branch_worker import sha, write
from openwebrl.arm_branch_protocol import seed
from openwebrl.controlled_sft_state import digest


def normalized(text):
    return ' '.join(str(text).lower().split())


def tasks(count=200):
    if type(count) is not int or count < 1:
        raise ValueError('Positive integer candidate-task count required')
    import pyarrow.parquet as pq
    origin=REPO/'openwebrl/data/webgym_filtered_popular_2102_cleaned.parquet'
    rows=pq.read_table(origin).to_pylist()
    benchmark=REPO/'openwebrl/data/eval/online-mind2web.jsonl'
    excluded=[json.loads(x) for x in benchmark.read_text().splitlines()]
    excluded+=pq.read_table(REPO/'openwebrl/data/webvoyager_val.parquet').to_pylist()
    ids=set(); intents=set()
    for row in excluded:
        meta=row.get('metadata') or {}
        ids.add(str(meta.get('task_id',row.get('task_id',''))))
        prompt=row.get('prompt','')
        if isinstance(prompt,list): prompt=' '.join(x.get('content','') for x in prompt)
        intents.add(normalized(meta.get('intent') or meta.get('task') or row.get('ques') or prompt))
    chosen=[]; seen=set()
    for row in rows:
        meta=row['metadata']; intent=meta['task']; key=normalized(intent)
        if meta['task_id'] in ids or key in intents or key in seen: continue
        if not meta['start_url'].startswith(('https://','http://')): continue
        seen.add(key)
        chosen.append(dict(metadata=dict(meta,intent=intent),prompt=row['prompt']))
    chosen.sort(key=lambda row:seed('task-order',row['metadata']['task_id']))
    selected=chosen[:count]
    if len(selected)!=count: raise ValueError('Insufficient unique training tasks')
    for i,row in enumerate(selected): row['index']=i
    return selected,dict(source=str(origin),source_sha256=sha(origin),input_tasks=len(rows),
        eligible_unique_tasks=len(chosen),selected=count,benchmark_file_sha256=sha(benchmark),
        deduplication='Exact normalized instruction and task-ID exclusion against OM2W300/WebVoyager validation; '
                      'no claim of semantic near-duplicate or pretraining decontamination.')


def prepare(root):
    root=Path(root).resolve()
    if root.exists(): raise FileExistsError('Preparation never overwrites an experiment')
    root.mkdir(parents=True)
    previous=json.loads((RUNTIME/'evaluations/local-sft-selector-controlled-20261006/prepared-parallel-v3/shard-0/plan.json').read_text())
    source=root/'source'
    def ignore(path,names):
        blocked=set(shutil.ignore_patterns('.git','.env*','__pycache__','*.pyc','.browser_use_sessions')(path,names))
        if Path(path).name=='openwebrl': blocked.add('data')
        return blocked
    shutil.copytree(previous['source'],source,symlinks=True,ignore=ignore)
    relative=[p.relative_to(REPO) for p in (REPO/'openwebrl').rglob('*.py')]
    relative += [p.relative_to(REPO) for p in (REPO/'openwebrl/env/prompts').glob('*') if p.is_file()]
    relative += [Path('scripts')/name for name in ('prepare_arm_branches.py','run_arm_branches.py',
        'diagnose_arm_browser_fork.py','run_controlled_sft_suite.py','runtime_ports.py','resume_baseline.py','h200_env.sh')]
    for name in relative:
        (source/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO/name,source/name)
    # Explicit opt-in in this immutable source; shared training/evaluation code
    # and their running source copies remain untouched.
    p=source/'openwebrl/env/local_process_env.py'; text=p.read_text()
    old='if module != "openwebrl.controlled_browser_server":'
    if text.count(old)!=1: raise ValueError('Browser allowlist source changed')
    text=text.replace(old,'if module not in ("openwebrl.controlled_browser_server", "openwebrl.arm_branch_browser"):')
    text=text.replace(' / "controlled_browser_server.py"',' / (module.rsplit(".", 1)[1] + ".py")')
    p.write_text(text)
    p=source/'openwebrl/controlled_sft_worker.py'; text=p.read_text()
    old='local.get("server_module") != "openwebrl.controlled_browser_server"'
    if text.count(old)!=1: raise ValueError('Worker browser validation source changed')
    p.write_text(text.replace(old,'local.get("server_module") not in ("openwebrl.controlled_browser_server", "openwebrl.arm_branch_browser")'))
    manifest=json.loads((source/'reference_manifest.json').read_text())
    manifest['recipe_files_sha256'].update({str(name):sha(source/name) for name in relative})
    (source/'reference_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rows,task_receipt=tasks()
    with (root/'tasks.jsonl').open('x') as handle:
        for row in rows: handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    write(root/'task-selection.json',task_receipt)
    config=deepcopy(previous['worker_config']); config.pop('kev_proof',None)
    for key in ('prompt_path','tool_list_path'):
        config[key]=str(source/Path(config[key]).relative_to(previous['source']))
    config['tasks_path']=str(root/'tasks.jsonl')
    browser=config['browser_env_config']
    browser.update(path_to_policy=config['prompt_path'],path_to_tool_list=config['tool_list_path'],
                   path_to_task_file=config['tasks_path'])
    browser['local_process']=dict(server_module='openwebrl.arm_branch_browser',server_module_sha256=sha(source/'openwebrl/arm_branch_browser.py'))
    config['browser_manifest'].update(branch_state_version='observable-replay-v1',artifact_directory=str(root/'browser-template'))
    config['canonical_reward_sha256']=sha(source/'openwebrl/eval/reward_online_mind2web.py')
    states=[]; schedule=[]
    for row in rows:
        task_id=row['metadata']['task_id']; sid=digest(['branch-state',task_id])[:16]
        depth=(2,4,6)[seed(task_id,'depth')%3]
        states.append(dict(id=sid,task_id=task_id,depth=depth))
        common=dict(state=sid,condition='L01',task_id=task_id,depth=depth,category='primary')
        schedule.append(dict(common,mode='discover',item_id=sid+'-discover'))
        for c in range(5):
            for r in range(3): schedule.append(dict(common,mode='branch',candidate=c,repeat=r,item_id=f'{sid}-c{c}-r{r}'))
        schedule.append(dict(common,mode='teacher',item_id=sid+'-teacher'))
    resources=dict(gpus=4,gpu_type='H200',cpus=32,memory_gib=480,total_seconds=28800,
                   sft_replicas=4,collectors=32)
    limits=dict(local_sft_generations=110000,browser_sessions=3400,browser_sessions_per_condition=3400,
        concurrent_browsers=30,luna_selector_requests=2200,luna_usd=15.,canonical_judge_http_attempts=6600,
        canonical_judge_usd=25.,jev_selector_requests=0,local_kev_requests=0)
    science=dict(protocol='fixed-sft-continuation-branches-v1',states_target=100,maximum_candidate_states=200,
        actor_checkpoint=config['actor_checkpoint'],actor_revision=previous['actor_revision'],
        sampling=dict(temperature=1.,top_p=.95,top_k=-1,max_new_tokens=4096),
        anchor_depths=[2,4,6],candidates=5,candidate_draws=15,continuations_per_candidate=3,
        intervention='Whole candidate response (reasoning and grounded action bundle) enters actor history.',
        replay='Fresh independent browser contexts; identical prefix responses; compare every prefix observation and anchor.',
        equivalence='Exact observed storage/page/forms/tabs/scroll plus frozen pixel tolerance; hidden remote/JS state is not proven cloned.',
        gate='Stop if first twelve candidate states yield zero accepted states; release all fifteen branches only after reconstruction.',
        remaining_horizon='30 total browser operations and60 total policy decisions minus common prefix; 1800s after release.',
        teachers=dict(model='gpt-6-luna',effort='high',output_cap=4096,forms=['index'],
            planned_calls=1400,conditions=['before','all_five_immediate_results'],
            repeat_calls=3,permutation_check='One extra before and one after(rep0) call per form/state',
            legacy='Original five-candidate selection-index output formulation; evidence prompt differs from the earlier study.',
            leakage='Allowlisted pre-action and immediate observations only; suffix/terminal evidence unavailable.'),
        outcome='Canonical OM2W o4-mini-2025-04-16,4096 response cap; original terminal status retained.',
        comparison='Teacher-selected terminal success vs paired before-only and uniform candidate choice; task-cluster intervals; no best-of-repetitions success oracle.',
        scope='No mixed-policy training data and no critic fitting; branch collection and teacher diagnostic only.')
    write(root/'scientific-protocol.json',science)
    pricing=RUNTIME/'evaluations/local-sft-selector-controlled-20261006/pricing-verification.json'
    shutil.copyfile(pricing,root/'pricing-verification.json')
    plan=dict(root=str(root),source=str(source),protocol_name=science['protocol'],
        experiment_id='arm-continuation-branches-overnight-20261006',states_target=100,states=states,schedule=schedule,
        candidate_count=5,candidate_draws=15,continuation_repetitions=3,parallel_states=2,reconstruction_gate_states=12,
        conditions={'L01':{'actor':'official OpenWebRL-4B-SFT'}},resources=resources,limits=limits,
        task_file=str(root/'tasks.jsonl'),worker_config=config,actor_files=previous['actor_files'],
        actor_revision=previous['actor_revision'],controller_sha256=sha(source/'scripts/run_arm_branches.py'),
        artifacts={str(path):sha(path) for path in (root/'tasks.jsonl',root/'scientific-protocol.json',
            root/'task-selection.json',root/'pricing-verification.json',source/'reference_manifest.json')})
    write(root/'plan.json',plan)
    write(root/'approval.json',dict(approved=False,plan_sha256=digest(plan),resources=resources,limits=limits,attempts=[],
        reason='Scientific implementation authorized; exact new resource and API caps await approval. No budget inheritance.'))
    (root/'launch.sbatch').write_text(f'''#!/usr/bin/env bash
#SBATCH --job-name=arm-branches
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output={root}/slurm-%j.log
set -euo pipefail
source {source}/scripts/h200_env.sh
export WANDB_PROJECT=openwebrl-evals PYTHONDONTWRITEBYTECODE=1
exec {RUNTIME}/venv/bin/python {source}/scripts/run_arm_branches.py --root {root} --execute "$SLURM_JOB_ID"
''')
    write(root/'preparation.json',dict(prepared=True,launch_authorized=False,new_model_api_calls=0,
        new_gpu_allocations=0,primary_continuations=1500,planned_teacher_calls=1400,
        maximum_gpu_hours=32,maximum_api_usd=40,plan_sha256=digest(plan),
        real_browser_reconstruction_and_GPU_throughput_validated=False))
    return dict(root=str(root),plan_sha256=digest(plan),resources=resources,limits=limits)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--root',required=True)
    args=parser.parse_args(); print(json.dumps(prepare(args.root),indent=2))
