#!/usr/bin/env python3
"""Stage pinned public sources/models in private shared storage; no GPU/API use."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.jev_eval import load_tasks, write_json
from openwebrl.webprm_baselines import prepare_plan
from openwebrl.webprm_source import load_shepherd, load_arbiter, load_arbiter_causal_renderer


def stage_equivalence_receipt(root):
    metadata=[Path('/tmp/openwebrl-shepherd-'+name+'-files.json') for name in ('langagi','webshepherd')]
    if not all(path.exists() for path in metadata):
        raise ValueError('Official Shepherd mirror-equivalence metadata unavailable')
    trees=[json.loads(path.read_text()) for path in metadata]
    normalized=[{item['path']:{key:item.get(key) for key in ('oid','lfs','size')} for item in tree if item['type']=='file'} for tree in trees]
    if normalized[0]!=normalized[1]:
        raise ValueError('Shepherd staged mirror differs from official release')
    receipt=dict(official_repo='WebShepherd/WebShepherd_8B',download_repo='LangAGI-Lab/WebShepherd_8B',
        identical_file_objects=len(normalized[0]),objects=normalized[1],
        source_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in metadata})
    for path in metadata:
        dest=root/path.name
        if not dest.exists():
            shutil.copyfile(path,dest)
        if dest.read_bytes()!=path.read_bytes():
            raise ValueError('Staged equivalence metadata changed')
    write_json(root/'shepherd-mirror-equivalence.json',receipt)
    arbiter=Path('/tmp/openwebrl-arbiter-weight-verification.json')
    if arbiter.exists():
        dest=root/arbiter.name
        if not dest.exists():
            shutil.copyfile(arbiter,dest)
        if dest.read_bytes()!=arbiter.read_bytes():
            raise ValueError('Staged Arbiter weight receipt differs')
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shepherd-source', type=Path, default=Path('/tmp/openwebrl-webshepherd-upstream'))
    p.add_argument('--arbiter-source', type=Path, default=Path('/tmp/openwebrl-webarbiter-upstream'))
    p.add_argument('--shepherd-actor-source',type=Path,default=Path('/tmp/openwebrl-webshepherd-demo.py'))
    p.add_argument('--benchmark', type=Path, default=Path('/tmp/openwebrl-webprm-benchmark.jsonl'))
    args = p.parse_args()
    root = args.output.resolve()
    if root.is_relative_to(REPO):
        raise ValueError('Private payloads must remain outside repository')
    root.mkdir(parents=True, exist_ok=True)
    sources = root/'sources'
    sources.mkdir(exist_ok=True)
    manifests = {}
    equivalence=stage_equivalence_receipt(root)
    (root/'requirements-browser.txt').write_text('browsergym-core==0.14.3\nplaywright==1.44.0\ngymnasium==1.4.0\nnumpy==2.5.3\npillow==12.3.0\npyparsing==3.3.3\npython-dotenv==1.2.4\n')
    actor_dest=sources/'shepherd-demo.py'
    if actor_dest.exists() and actor_dest.read_bytes()!=args.shepherd_actor_source.read_bytes():
        raise ValueError('Staged author actor source differs')
    if not actor_dest.exists():
        shutil.copyfile(args.shepherd_actor_source,actor_dest)
    manifests['shepherd_actor_sha256']=hashlib.sha256(actor_dest.read_bytes()).hexdigest()
    for name, source, loader in [('shepherd', args.shepherd_source, load_shepherd),
                                 ('arbiter', args.arbiter_source, load_arbiter)]:
        manifests[name] = loader(source).manifest
        dest = sources/name
        if not dest.exists():
            shutil.copytree(source, dest)
        if loader(dest).manifest != manifests[name]:
            raise ValueError('Staged upstream source changed')
    renderer = load_arbiter_causal_renderer(args.benchmark)
    manifests['arbiter_prompt'] = renderer.manifest
    destination = root/'WebPRMBench.jsonl'
    if not destination.exists():
        shutil.copyfile(args.benchmark, destination)
    if hashlib.sha256(destination.read_bytes()).digest() != hashlib.sha256(args.benchmark.read_bytes()).digest():
        raise ValueError('Staged benchmark differs')
    # Bound download concurrency and avoid large Rust/Xet buffers on login host.
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    model_manifest = root/'models.json'
    models = json.loads(model_manifest.read_text()) if model_manifest.exists() else {}
    for name, repo_id in [('shepherd8b', 'LangAGI-Lab/WebShepherd_8B'),
                          ('arbiter7b', 'ZYao720/WebArbiter-7B')]:
        revision = models[name]['revision'] if name in models else api.model_info(repo_id).sha
        model_path = root/'models'/name
        models[name] = dict(repo_id=repo_id, revision=revision, path=str(model_path), status='downloading')
        write_json(model_manifest, models)
        snapshot_download(repo_id, revision=revision, local_dir=model_path,
                          max_workers=1,
                          allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', 'LICENSE*', 'README*'])
        weights = sorted(model_path.glob('*.safetensors'))
        if not weights or not (model_path/'config.json').exists():
            raise ValueError('Missing model weights/config')
        models[name].update(status='downloaded', weight_bytes=sum(f.stat().st_size for f in weights),
                            config_sha256=hashlib.sha256((model_path/'config.json').read_bytes()).hexdigest())
        write_json(model_manifest, models)
    tasks_path = REPO/'openwebrl/data/eval/online-mind2web.jsonl'
    task_dest=root/'online-mind2web.jsonl'
    if task_dest.exists() and task_dest.read_bytes()!=tasks_path.read_bytes():
        raise ValueError('Staged task file differs')
    if not task_dest.exists():
        shutil.copyfile(tasks_path,task_dest)
    tasks = load_tasks(tasks_path)
    plan = prepare_plan(tasks, hashlib.sha256(tasks_path.read_bytes()).hexdigest())
    plan['sources'] = manifests
    plan['model_source_equivalence']=equivalence
    plan['browser_requirements']='requirements-browser.txt'
    plan['models'] = models
    plan['approval'] = dict(exact_resources=plan['proposal'],
        authorization='User: yes do it; 2026-10-09, following explicit pilot resource proposal',
        api_cap_usd=30, gpu_seconds_total=7200)
    plan['status'] = 'assets_staged_harness_preflight_pending'
    write_json(root/'plan.json', plan)
    write_json(root/'tasks.json', tasks)
    print(json.dumps(dict(root=str(root), status=plan['status'], models=models), indent=2))


if __name__ == '__main__':
    main()
