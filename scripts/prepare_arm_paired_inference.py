#!/usr/bin/env python3
"""Freeze the requested historical-protocol three-arm study, without launching."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import time

REPO=Path(__file__).resolve().parents[1]
RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
sys.path.insert(0,str(REPO))
from openwebrl.arm_paired_inference import PROTOCOL,REVISION,PIOTR_REVISION,schedule
from openwebrl.controlled_sft_state import digest
from run_arm_paired_inference import RESOURCES,LIMITS
from resume_baseline import validate_source
from browser_prompt_assets import freeze_browser_prompt_assets, validate_browser_prompt_assets


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')


def header(p):
    with Path(p).open('rb') as f:
        count=struct.unpack('<Q',f.read(8))[0]
        if count>10000000:raise ValueError('Unexpected tensor header')
        d=json.loads(f.read(count))
    return {k:(v['shape'],v['dtype']) for k,v in d.items() if k!='__metadata__'}


def prepare(root):
    root=Path(root).resolve();download=json.loads((root/'download.json').read_text())
    metadata=json.loads((root/'model-hub-metadata.json').read_text())
    if download['revision']!=REVISION or metadata['revision']!=REVISION:raise ValueError('Model revision drift')
    actor=Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
    new=Path(download['path']);piotr=Path('/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection')/PIOTR_REVISION
    model_files={};proof={}
    for role,model in [('actor',actor),('new_arm',new),('piotr_arm',piotr)]:
        for p in model.iterdir():
            if p.is_file():
                st=p.stat();model_files[str(p)]=dict(size=st.st_size,mtime_ns=st.st_mtime_ns)
        if (model/'chat_template.jinja').read_bytes()!=(actor/'chat_template.jinja').read_bytes():
            raise ValueError(role+' template differs')
        from tokenizers import Tokenizer
        if Tokenizer.from_file(str(model/'tokenizer.json')).get_vocab()!=Tokenizer.from_file(str(actor/'tokenizer.json')).get_vocab():
            raise ValueError(role+' vocabulary differs')
        config=json.loads((model/'config.json').read_text());base=json.loads((actor/'config.json').read_text())
        for k in ['hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','vocab_size']:
            if config['text_config'][k]!=base['text_config'][k]:raise ValueError(role+' architecture differs')
        proof[role]=dict(path=str(model),template_sha256=sha(model/'chat_template.jinja'),tokenizer_sha256=sha(model/'tokenizer.json'))
    expected=next(x['lfs']['sha256'] for x in metadata['files'] if x['name']=='model.safetensors')
    actual=sha(new/'model.safetensors')
    if actual!=expected:raise ValueError('Downloaded ARM weights failed hub checksum')
    if header(new/'model.safetensors')!=header(actor/'model.safetensors'):
        raise ValueError('New ARM tensor schema differs from original SFT')
    proof.update(new_weight_sha256=actual,new_weight_matches_hub=True,tensor_shapes_and_dtypes_match_actor=True,
                 pinned_new_revision=REVISION,pinned_piotr_revision=PIOTR_REVISION)
    write(root/'model-verification.json',proof)
    parent=RUNTIME/'reference-arm-controlled-inference-20261004-v1';validate_source(parent)
    source=root/'source'
    shutil.copytree(parent,source,symlinks=True,ignore=shutil.ignore_patterns('.git','.env*','__pycache__','*.pyc','.browser_use_sessions'))
    changed=['openwebrl/arm_controlled_inference.py','openwebrl/arm_paired_inference.py','scripts/run_arm_paired_inference.py',
        'scripts/arm_inference_runtime.py','scripts/runtime_ports.py','openwebrl/controlled_sft_state.py',
        'scripts/browser_prompt_assets.py','openwebrl/browser_actor_protocol.py']
    # Copy import-only controller dependencies as well; inherited actor/browser
    # generation and judging code stay from the proven historical-family source.
    for name in changed:
        destination=source/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,destination)
    manifest=json.loads((source/'reference_manifest.json').read_text())
    manifest['recipe_files_sha256'].update(freeze_browser_prompt_assets(source,REPO))
    manifest['recipe_files_sha256'].update({n:sha(source/n) for n in changed})
    manifest['new_selectionarm_study']=dict(parent=str(parent),changed=changed,optimizer_updates=0)
    (source/'reference_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');validate_source(source)
    validate_browser_prompt_assets(source)
    tasks=REPO/'openwebrl/data/eval/online-mind2web.jsonl'
    if sha(tasks)!='8343c23be98d6d63856e9b53ff3884222be099cc0f55ad5edb475176f54317ed':raise ValueError('Historical task cohort changed')
    shutil.copyfile(tasks,root/'tasks.jsonl');tasks=root/'tasks.jsonl'
    ids=[json.loads(line)['metadata']['task_id'] for line in tasks.read_text().splitlines()]
    if len(ids)!=300 or len(set(ids))!=300:raise ValueError('Need full300 unique tasks')
    rows=schedule(ids)
    for shard in (0,1):
        directory=root/f'shard-{shard}';directory.mkdir()
        write(directory/'schedule.json',rows)
        experiment=dict(protocol=PROTOCOL,task_file_sha256=sha(tasks),schedule_sha256=sha(directory/'schedule.json'))
        write(directory/'experiment.json',experiment)
        artifacts={str(p):sha(p) for p in [source/'reference_manifest.json',directory/'schedule.json',
            directory/'experiment.json',tasks,root/'model-verification.json',source/'scripts/run_arm_paired_inference.py']}
        plan=dict(root=str(directory),source=str(source),protocol=PROTOCOL,resources=RESOURCES,limits=LIMITS,
            shard=shard,primary_episodes=450,campaign_episodes=900,task_file=str(tasks),
            actor=str(actor),selector=str(new),piotr_selector=str(piotr),model_files=model_files,artifacts=artifacts,
            experiment_id=root.name+f'-shard{shard}',budget_inheritance='none; separate shard caps and original continuation study untouched')
        write(directory/'plan.json',plan)
        write(directory/'approval.json',dict(approved=False,plan_sha256=digest(plan),resources=RESOURCES,limits=LIMITS,attempts=[],
            authorization='User approved scientific scope and addition of Piotr ARM; exact new resource/API caps await approval.'))
        batch=f'''#!/usr/bin/env bash
# PREPARED ONLY. Exact resource/API approval and registered job ID required.
#SBATCH --job-name=arm-rltasks-{shard}
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output={directory}/slurm-%j.log
set -euo pipefail
source {source}/scripts/h200_env.sh
export WANDB_PROJECT=openwebrl-evals PYTHONDONTWRITEBYTECODE=1
exec {RUNTIME}/venv/bin/python {source}/scripts/run_arm_paired_inference.py --root {directory} --execute "$SLURM_JOB_ID"
'''
        (directory/'launch.sbatch').write_text(batch)
    report=dict(prepared=True,approved=False,protocol=PROTOCOL,primary_episodes=900,resources_per_shard=RESOURCES,
        shards=2,maximum_gpu_hours=16,judge_caps_per_shard=LIMITS,judge_caps_total=dict(usd=20,calls=3960),
        model_verification=proof,actor_generation_and_browser_source='preserved historical-family controlled-inference source',
        actor_concurrency=8,selector_services='Piotr and new ARM on the second H200; actor on first, per shard',
        telemetry=['success overall/valid-only/common-valid','paired task-bootstrap contrasts','actor/selector input/output tokens',
            'episode/request latency','browser steps/time','judge costs and all-attempt allocation GPU-hours','GPU utilization/power'],
        no_new_gpu_allocations=True,no_paid_model_calls=True,created_unix=time.time())
    write(root/'preparation.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    d=prepare(a.root);print(json.dumps({k:d[k] for k in ['prepared','approved','primary_episodes','resources_per_shard','shards','maximum_gpu_hours','judge_caps_total']},indent=2))
