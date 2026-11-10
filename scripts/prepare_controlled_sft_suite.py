#!/usr/bin/env python3
"""Freeze the four-condition local-browser suite; --dry-run never launches work."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
sys.path.insert(0, str(REPO))
from openwebrl.controlled_sft_state import digest


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(value,f,indent=2,sort_keys=True); f.write('\n')


def schedule(task_ids, seed=20261006):
    ids=list(task_ids); random.Random(seed).shuffle(ids)
    rows=[]
    for category, chosen in [('smoke',ids[:3]),('primary',ids)]:
        for task_id in chosen:
            arms=['L01','L08','L10','L11']
            random.Random(digest([seed,category,task_id])).shuffle(arms)
            rows.extend(dict(item_id=digest([category,task_id,arm]),category=category,
                             condition=arm,task_id=task_id) for arm in arms)
    return rows


def partition_tasks(task_ids, count=2, seed=20261006):
    ids=list(task_ids)
    if count!=2 or len(ids)!=300 or len(set(ids))!=300:
        raise ValueError('Parallel core requires two disjoint halves of the full300 cohort')
    random.Random(seed).shuffle(ids)
    return [ids[i::count] for i in range(count)]


def prepare(output, config_path, proposal_path, base_source, shard_index=None):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError('Never overwrite a prepared experiment')
    config, proposal=read(config_path),read(proposal_path)
    if config['protocol_name']!='local-openwebrl-om2w-v2' or config['judge'].get('strict_secondary_review'):
        raise ValueError('Only the unchanged canonical OM2W judge is in scope')
    if [r['id'] for r in config['conditions']] != ['L01','L08','L10','L11']:
        raise ValueError('The four-condition fresh core must remain fixed')
    tasks_path=REPO/'openwebrl/data/eval/online-mind2web.jsonl'
    rows=[json.loads(x) for x in tasks_path.read_text().splitlines()]
    ids=[r['metadata']['task_id'] for r in rows]
    if len(ids)!=300 or len(set(ids))!=300 or sha(tasks_path)!=config['dataset']['input_file_sha256']:
        raise ValueError('Full300 task cohort changed')
    prior=read(RUNTIME/'luna-qwen-full300-20261004/sft/plan.json')
    actor=Path(prior['actor_path'])
    for name,stat in prior['actor_files'].items():
        current=(actor/name).stat()
        if {'size':current.st_size,'mtime_ns':current.st_mtime_ns}!=stat:
            raise ValueError('Actor file changed since verified checkpoint receipt')
    kev=read(RUNTIME/'kev-preparation-20261004/verified-models.json')['27b']
    if not kev['verified']: raise ValueError('Kev checkpoint is not verified')
    for name, info in kev['files'].items():
        st=Path(name).stat()
        if st.st_size!=info['size'] or st.st_mtime_ns!=info['mtime_ns']:
            raise ValueError('Kev file changed since independent hash verification')
    kev_source=RUNTIME/'kev-upstream'
    kev_head=subprocess.check_output(['git','-C',str(kev_source),'rev-parse','HEAD'],text=True).strip()
    if kev_head!=kev['spec']['upstream_commit'] or subprocess.check_output(
            ['git','-C',str(kev_source),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Kev serving source differs from pinned clean upstream commit')
    tracked=subprocess.check_output(['git','-C',str(kev_source),'ls-files','-z']).decode().split('\0')
    kev_source_proof=dict(root=str(kev_source),commit=kev_head,tracked_clean=True,
        files={n:sha(kev_source/n) for n in tracked if n.endswith(('.py','.toml','.json','.yaml','.yml')) and (kev_source/n).is_file()})
    if shard_index is not None and (type(shard_index) is not int or shard_index not in (0,1)):
        raise ValueError('Shard index must be0or1')
    split=shard_index is not None
    partitions=partition_tasks(ids) if split else None
    assigned_ids=partitions[shard_index] if split else ids
    expected_resources=(dict(gpus=2,gpu_type='H200',cpus=16,memory_gib=240,
                             total_scheduler_seconds_all_attempts=28800,maximum_gpu_hours=16)
                        if split else dict(gpus=8,gpu_type='H200',cpus=64,memory_gib=960,
                            total_scheduler_seconds_all_attempts=57600,maximum_gpu_hours=128))
    if proposal['resources']!=expected_resources:
        raise ValueError('Resource proposal changed; do not silently substitute allocation limits')
    requested=config['budget']['requested_compute']
    if (split and (requested.get('jobs')!=2 or requested.get('per_job')!=expected_resources)
            or not split and requested!=expected_resources):
        raise ValueError('Public resource plan and chosen preparation topology disagree')
    if split:
        if proposal.get('shard_count')!=2 or proposal['topology_proposed']!=dict(
                sft_tp1_replicas=1,kev27b_tp1_replicas=1,max_active_browser_sessions=8,
                shared_randomized_arm_queue=True):
            raise ValueError('Each parallel shard needs oneSFT/oneKev and a shared8-browser pool')
        if {k:2*v for k,v in proposal['caps'].items()}!=proposal['aggregate_caps']:
            raise ValueError('Parallel shard caps must sum to the campaign cap without duplication')
        if config['budget']['per_shard_caps']!=proposal['caps']:
            raise ValueError('Public per-shard caps differ from the proposed approval')
    resources=dict(gpus=expected_resources['gpus'],gpu_type='H200',cpus=expected_resources['cpus'],
                   memory_gib=expected_resources['memory_gib'],
                   total_seconds=expected_resources['total_scheduler_seconds_all_attempts'],
                   sft_replicas=1 if split else 7,kev_replicas=1,collectors=8 if split else 7)
    output.mkdir(parents=True)
    source=output/'source'
    ignore_base=shutil.ignore_patterns('.git','.env*','__pycache__','*.pyc','.browser_use_sessions')
    def ignore(path,names):
        # The historical reference links data back to the mutable workspace.
        # Freeze only needed data below instead of preserving that write-through link.
        omitted=set(ignore_base(path,names))
        if Path(path).resolve()==(Path(base_source)/'openwebrl').resolve():omitted.add('data')
        return omitted
    shutil.copytree(base_source,source,symlinks=True,ignore=ignore)
    # Freeze the actually tested package, required prompt/tool assets, and all
    # small controller dependencies. The inherited runtime contains Slime.
    relative=[p.relative_to(REPO) for p in (REPO/'openwebrl').rglob('*.py')]
    relative += [p.relative_to(REPO) for p in (REPO/'openwebrl/env/prompts').glob('*') if p.is_file()]
    relative += [Path('scripts')/name for name in ('prepare_controlled_sft_suite.py',
        'run_controlled_sft_suite.py','runtime_ports.py','h200_env.sh','resume_baseline.py')]
    for name in relative:
        (source/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO/name,source/name)
    manifest=read(source/'reference_manifest.json')
    manifest['recipe_files_sha256'].update({str(n):sha(source/n) for n in relative})
    # Parent's manifest is replaced only in this new, unapproved source copy.
    (source/'reference_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copyfile(tasks_path,output/'tasks.jsonl')
    write(output/'scientific-config.json',config)
    write(output/'resource-proposal.json',proposal)
    write(output/'kev-file-proof.json',kev)
    write(output/'kev-source-proof.json',kev_source_proof)
    for name in ('pricing-verification.json','processor-cpu-preflight.json'):
        shutil.copyfile(Path(proposal_path).parent/name,output/name)
    browser=config['shared_browser']
    binary=RUNTIME/'browsers'/browser['installed_binaries'][0]['relative_path']
    if sha(binary)!=browser['installed_binaries'][0]['sha256']: raise ValueError('Pinned Chromium binary changed')
    browser_manifest={k:browser[k] for k in ('viewport','device_scale_factor','locale','timezone','headless','proxy','stealth','launch_extra_args')}
    browser_manifest.update(binary_path=str(binary),binary_sha256=sha(binary),artifact_directory=str(output/'browser-template'))
    proof=dict(model=kev['spec']['repo'],revision=kev['spec']['revision'],base=kev['spec']['base'],
        base_revision=kev['spec']['base_revision'],upstream_commit=kev['spec']['upstream_commit'],
        head_sha256=kev['spec']['head_sha256'],files_verified=True,
        verified_model_path=kev['spec']['server_run'],file_proof_sha256=sha(output/'kev-file-proof.json'))
    caps=proposal['caps']
    limits=dict(local_sft_generations=caps['actor_proposals'],browser_sessions=caps['browser_episodes_total'],
        browser_sessions_per_condition=caps['browser_episodes_per_arm'],concurrent_browsers=resources['collectors'],
        luna_selector_requests=caps['luna_requests'],jev_selector_requests=caps['jev_requests'],
        local_kev_requests=caps['local_kev_requests'],luna_usd=caps['luna_usd'],
        canonical_judge_http_attempts=caps['canonical_judge_http_attempts'],canonical_judge_usd=caps['canonical_judge_usd'])
    worker_config=dict(actor_checkpoint=str(actor),prompt_path=str(source/config['prompts']['local_actor_file']),
        prompt_sha256=config['prompts']['local_actor_sha256'],
        tool_list_path=str(source/'openwebrl/env/prompts/tool_list.json'),
        tool_list_sha256=sha(source/'openwebrl/env/prompts/tool_list.json'),
        canonical_reward_sha256=config['judge']['source_file_sha256'],browser_manifest=browser_manifest,
        browser_env_config=dict(mode='local_process',width=1280,height=720,dpr=1,use_screenshot=True,
            use_a11ytree=False,resize_output_coords=True,resize_scale=1000,
            path_to_tool_list=str(source/'openwebrl/env/prompts/tool_list.json'),
            path_to_policy=str(source/config['prompts']['local_actor_file']),
            path_to_task_file=str(output/'tasks.jsonl'),wait_timeout=30000,screenshot_timeout=15000,
            max_retries=3,image_patch_size=32,
            local_process=dict(server_module='openwebrl.controlled_browser_server',
                server_module_sha256=sha(source/'openwebrl/controlled_browser_server.py'))),
        kev_proof=proof,tasks_path=str(output/'tasks.jsonl'))
    if sha(worker_config['prompt_path'])!=worker_config['prompt_sha256'] or sha(source/'openwebrl/eval/reward_online_mind2web.py')!=worker_config['canonical_reward_sha256']:
        raise ValueError('Actor prompt or canonical judge differs from scientific protocol')
    plan=dict(protocol_name=config['protocol_name'],status='prepared_exact_budget_not_approved',
        experiment_id=(proposal['campaign_id']+f'-shard-{shard_index}') if split else output.parent.name,
        root=str(output),source=str(source),
        source_manifest_sha256=sha(source/'reference_manifest.json'),
        controller_sha256=sha(source/'scripts/run_controlled_sft_suite.py'),
        scientific_config_sha256=sha(output/'scientific-config.json'),conditions={r['id']:r for r in config['conditions']},
        resources=resources,limits=limits,task_file=str(output/'tasks.jsonl'),task_file_sha256=sha(output/'tasks.jsonl'),
        schedule=schedule(assigned_ids),primary_episodes=4*len(assigned_ids),smoke_episodes=12,worker_config=worker_config,
        requires_jev_context_diagnostic=True,
        actor_files=prior['actor_files'],actor_revision=prior['actor_revision'],kev_spec=kev['spec'],
        kev_source_proof_sha256=sha(output/'kev-source-proof.json'),
        offline_receipts={n:sha(output/n) for n in ('pricing-verification.json','processor-cpu-preflight.json')},
        budget_inheritance='none; all retries and failed allocations charge this shard approval; no automatic transfers',
        wandb_project='openwebrl-evals',completion_requires=f'{4*len(assigned_ids)} primary records,12 smokes,all evidence hashes and model/judge identities,closed browsers,W&B,final scheduler/API accounting')
    if split:
        plan.update(campaign_id=proposal['campaign_id'],campaign_task_shards=partitions,
                    shard=dict(index=shard_index,count=2,task_ids=assigned_ids))
    write(output/'plan.json',plan)
    write(output/'approval.json',dict(approved=False,plan_sha256=digest(plan),resources=resources,limits=limits,attempts=[],
        authorization='Scientific scope accepted; exact new allocation and API/browser caps not approved'))
    batch=f'''#!/usr/bin/env bash
# PREPARED ONLY: exact resource/cap approval and registered job receipt required.
#SBATCH --job-name=controlled-sft4{('-s'+str(shard_index)) if split else ''}
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:{resources['gpus']}
#SBATCH --cpus-per-task={resources['cpus']}
#SBATCH --mem={resources['memory_gib']}G
#SBATCH --time={resources['total_seconds']//3600:02d}:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output={output}/slurm-%j.log
set -euo pipefail
source {source}/scripts/h200_env.sh
export WANDB_PROJECT=openwebrl-evals PYTHONDONTWRITEBYTECODE=1
exec {RUNTIME}/venv/bin/python {source}/scripts/run_controlled_sft_suite.py --root {output} --execute "$SLURM_JOB_ID"
'''
    (output/'launch.sbatch').write_text(batch)
    write(output/'preparation.json',dict(prepared=True,launch_authorized=False,model_api_calls=0,
        browser_episodes=0,gpu_allocations=0,plan_sha256=digest(plan),new_primary_episodes=plan['primary_episodes'],
        live_startup_required=[f"{resources['sft_replicas']} isolated SFT servers",'pinned Kev server identity',
                               '12 fresh paid smoke episodes','shared-server request latency and GPU utilization'],
        canonical_judge_unchanged=True,strict_audit_not_included=True))
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=REPO/'openwebrl/docs/arm_results/local_inference_rerun_plan_20261006.json')
    p.add_argument('--proposal',type=Path,default=RUNTIME/'evaluations/local-sft-selector-controlled-20261006/resource-proposal.json')
    p.add_argument('--base-source',type=Path,default=RUNTIME/'reference-luna-qwen-full300-20261004-v1')
    p.add_argument('--shard-index',type=int,choices=(0,1),help='One of two parallel2-GPU task shards')
    p.add_argument('--dry-run',action='store_true',required=True)
    a=p.parse_args(); result=prepare(a.output,a.config,a.proposal,a.base_source,a.shard_index)
    print(json.dumps({k:result[k] for k in ('status','root','resources','limits','primary_episodes','smoke_episodes')},indent=2))


if __name__=='__main__': main()
