#!/usr/bin/env python3
"""Prepare a reviewable repaired pilot; never submit or infer authorization."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.webprm_source import load_shepherd, load_arbiter, load_arbiter_causal_renderer
from openwebrl.webprm_arbiter_harness import load_arbiter_harness, load_native_observation_processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    previous, output = args.previous.resolve(), args.output.resolve()
    plan = copy.deepcopy(json.loads((previous/'plan.json').read_text()))
    budget = json.loads((previous/'episodes/budget.json').read_text())
    assert budget['cap'] == 30 and budget['charged'] < 30
    output.mkdir(parents=True, exist_ok=False)
    for name in ('sources','models.json','online-mind2web.jsonl','WebPRMBench.jsonl'):
        (output/name).symlink_to(previous/name, target_is_directory=(previous/name).is_dir())
    shutil.copy2(previous/'requirements-browser.txt',output/'requirements-browser.txt')
    sources=dict(shepherd=load_shepherd(output/'sources/shepherd').manifest,
        arbiter=load_arbiter(output/'sources/arbiter').manifest,
        arbiter_harness=load_arbiter_harness(output/'sources/arbiter').manifest,
        arbiter_observation=load_native_observation_processor(output/'sources/arbiter',
            viewport_size={'width':1280,'height':720}).manifest,
        arbiter_prompt=load_arbiter_causal_renderer(output/'WebPRMBench.jsonl').manifest,
        shepherd_actor_sha256=hashlib.sha256((output/'sources/shepherd-demo.py').read_bytes()).hexdigest())
    plan.update(status='prepared_pending_exact_compute_approval',authorization_status='pending',
        protocol_version='native-recovery-v3',code_version='v3',sources=sources,
        shared_budget_path=str(previous/'episodes/budget.json'),
        prior_api_conservative_charge_usd=budget['charged'],
        prior_api_calls=len(budget['calls']),
        api_budget_note='Original USD30 total cap, shared immutable-prefix ledger; no new API budget',
        accounting_episode_roots=[str(previous/'episodes'),
            str(previous/'cpu-startup-recovery-20261010/episodes'),
            str(previous/'cpu-startup-recovery-20261010-attempt2/episodes')],
        approval=dict(authorization=None,api_cap_usd=30,gpu_seconds_total=0),
        operational_settings=dict(workers=4,quality_gate_tasks=2),
        validation_required=['focused_tests','six_arm_real_browser_synthetic_fixture',
                             'native_auxiliary_prompt_parity','local_gpu_preflight_inside_approved_allocation'])
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (output/'prior-api-budget.json').write_text(json.dumps(budget,indent=2)+'\n')
    request=dict(status='proposal_only_not_authorized',gpus=1,gpu_type='H200',cpus=8,memory_gib=120,
        additional_allocation_seconds_total=7200,includes_retries=True,api_total_cap_usd=30,
        additional_api_budget_usd=0,api_conservative_charge_to_date_usd=budget['charged'],
        fixed_tasks=12,conditions=6,planned_new_protocol_cells=72,full300_authorized=False)
    (output/'resource-request.json').write_text(json.dumps(request,indent=2)+'\n')
    q=shlex.quote
    cache_variables = {
        'XDG_CACHE_HOME': 'xdg/cache', 'XDG_CONFIG_HOME': 'xdg/config', 'XDG_DATA_HOME': 'xdg/data',
        'HF_HOME': 'huggingface', 'HF_HUB_CACHE': 'huggingface/hub',
        'TORCH_HOME': 'torch', 'TORCH_EXTENSIONS_DIR': 'torch/extensions',
        'TORCHINDUCTOR_CACHE_DIR': 'torch/inductor', 'TRITON_CACHE_DIR': 'triton',
        'FLASHINFER_WORKSPACE_BASE': 'flashinfer', 'SGLANG_CACHE_DIR': 'sglang',
        'SGLANG_DG_CACHE_DIR': 'deep-gemm', 'DG_JIT_CACHE_DIR': 'deep-gemm',
        'CUDA_CACHE_PATH': 'cuda', 'NUMBA_CACHE_DIR': 'numba', 'MPLCONFIGDIR': 'matplotlib',
        'WANDB_DIR': 'wandb', 'WANDB_CACHE_DIR': 'wandb/cache', 'WANDB_CONFIG_DIR': 'wandb/config',
        'PIP_CACHE_DIR': 'pip', 'UV_CACHE_DIR': 'uv',
    }
    cache_lines = []
    for name, suffix in cache_variables.items():
        path = output/'model-cache'/suffix
        path.mkdir(parents=True, exist_ok=True)
        cache_lines.append(f'export {name}={q(str(path))}')
    (output/'model-cache.env').write_text('\n'.join(cache_lines)+'\n')
    batch=f'''#!/usr/bin/env bash
# Proposal only; launcher rejects pending authorization.
#SBATCH --job-name=webprm-repaired-pilot
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output={output}/slurm-%j.out
set -euo pipefail
source {q(str(output/'model-cache.env'))}
cd {q(str(output/'code-v3'))}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 WANDB_PROJECT=openwebrl-evals
export WEBPRM_ALLOCATION_SECONDS=7200
export CUDA_HOME=/gpfs/scrubbed/zixianma/openwebrl-runtime/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${{LD_LIBRARY_PATH:-}}"
exec /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python scripts/launch_webprm_pilot.py \\
  --root {q(str(output))} \\
  --browser-python {q(str(previous/'browser-env/bin/python'))} \\
  --engine-python /gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python \\
  --env-file {q(str(REPO/'.env'))}
'''
    (output/'repaired-pilot.sbatch').write_text(batch)
    print(json.dumps(dict(status=plan['status'],root=str(output),request=request),indent=2))


if __name__=='__main__':
    main()
