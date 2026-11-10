#!/usr/bin/env python3
"""Training-throughput benchmark for parallel layout and activation recompute (no objective change).

Approved 2026-10-24: one 8 H200 / 64 CPU / 960 GiB job, at most 3 h including retries, no API.
Replays one fixed batch of complete uniform-G8 training groups (built on CPU with --build-batch,
longest rows first) from the retained iteration-60 checkpoint, training only (slime
--load-debug-rollout-data: no browsers, no inference servers). Each case trains one rollout
(two optimizer updates per epoch x two PPO epochs), saves to scratch, and records throughput
from slime's perf log and peak GPU memory from nvidia-smi.

Order (user decision): TP1/DP8 none -> selective -> full (stop at the first that fits);
TP2/DP4 none -> selective. Fits = finished without OOM and peak memory.used <= 90% of the GPU.
The current TP2/DP4 full-recompute setting is re-run on the same batch only when the best
case's throughput is within 10% of production. Nothing is written into the live run.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
G8 = RUNTIME/'training-controls/uniform8-original2102-20261006/to90-20261024'
RUN = RUNTIME/'runs/outcome-uniform8-original2102-20261006'
OUT = RUNTIME/'benchmarks/train-layouts-20261024'
CKPT_STEP = 60
ROWS = 512
GPU_MIB = 143771
HEADROOM = 0.9
CASE_SECONDS = 1800
SELECTIVE = ['--recompute-granularity', 'selective', '--recompute-modules', 'core_attn', 'mlp', 'layernorm']
FULL = ['--recompute-granularity', 'full', '--recompute-method', 'uniform', '--recompute-num-layers', '1']
sys.path[:0] = [str(G8), str(REPO/'scripts')]


def build_batch(sources):
    """Keep complete groups, longest rows first, until ROWS rows; record the length distribution."""
    import torch
    best, scan = None, {}
    for src in sources:
        data = torch.load(src, weights_only=False, map_location='cpu', mmap=True)['samples']
        lengths = [len(s.get('tokens') or []) for s in data]
        scan[str(src)] = dict(rows=len(data), max=max(lengths), p99=sorted(lengths)[int(.99*len(lengths))],
                              mean=sum(lengths)/len(lengths))
        if best is None or scan[str(src)]['max'] > scan[best[0]]['max']:
            best = (str(src), data, lengths)
        else:
            del data
    src, data, lengths = best
    groups = {}
    for i, s in enumerate(data):
        groups.setdefault(s.get('group_index'), []).append(i)
    order = sorted(groups.values(), key=lambda idx: -max(lengths[i] for i in idx))
    keep = []
    for idx in order:
        if len(keep) >= ROWS:
            break
        keep += idx
    keep.sort()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(dict(rollout_id=0, samples=[data[i] for i in keep]), OUT/'replay-batch.pt')
    chosen = [lengths[i] for i in keep]
    summary = dict(scan=scan, source=src, rows=len(keep), groups=len([1 for idx in order if idx[0] in set(keep)]),
                   max_tokens=max(chosen), mean_tokens=sum(chosen)/len(chosen), checkpoint_step=CKPT_STEP)
    (OUT/'replay-batch.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def case_command(name, tp, recompute, load_dir=None, step=CKPT_STEP):
    import campaign_recipe as recipe
    p = recipe.training_command(61, resume=True)
    env = p['environment']
    scratch = OUT/name
    if load_dir:
        env['SLIME_LOAD_CHECKPOINT'] = str(load_dir)
    env.update(TP_SIZE=str(tp), NUM_ROLLOUT=str(step+2), SAVE_DIR=str(scratch/'save'), SLIME_CKPT_STEP=str(step),
               RECOMPUTE_ACTIVATIONS='0', WANDB_MODE='disabled', OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(scratch/'multimodal'),
               SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR=str(scratch/'browser-logs'))
    env.pop('OPENWEBRL_STOP_AFTER_SAVED_ROLLOUT', None)
    cmd = [x for x in p['command']]
    if '--use-wandb' in cmd:
        i = cmd.index('--use-wandb'); cmd.pop(i)
    for flag in ('--wandb-mode', '--wandb-project', '--wandb-team', '--wandb-group', '--wandb-dir'):
        if flag in cmd:
            i = cmd.index(flag); del cmd[i:i+2]
    i = cmd.index('--save-debug-rollout-data'); del cmd[i:i+2]
    cmd = [c for c in cmd if c != '--disable-wandb-random-suffix']
    cmd += ['--load-debug-rollout-data', str(OUT/'replay-batch.pt')]
    cmd += {'none': [], 'selective': SELECTIVE, 'full': FULL}[recompute]
    return recipe, cmd, env


def run_case(job, name, tp, recompute, load_dir=None, step=CKPT_STEP):
    from resume_baseline import source_command, clean_environment
    from runtime_ports import PORT_ENV, lease_ports
    result_path = OUT/name/'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    recipe, cmd, extra = case_command(name, tp, recompute, load_dir, step)
    (OUT/name).mkdir(parents=True, exist_ok=True)
    env = clean_environment()
    env.update(extra)
    lease, base = lease_ports(job); env[PORT_ENV] = str(base)
    log = OUT/name/'train.log'
    smi = subprocess.Popen(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits', '-l', '2'],
                           stdout=(OUT/name/'memory.csv').open('w'), stderr=subprocess.DEVNULL)
    started = time.time()
    try:
        with log.open('w') as handle:
            proc = subprocess.Popen(source_command(recipe.SOURCE, ['timeout', '--signal=INT', '--kill-after=60', str(CASE_SECONDS), *cmd]),
                                    cwd=recipe.SOURCE, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            code = proc.wait()
    finally:
        smi.send_signal(signal.SIGINT); smi.wait()
        lease.close()
        subprocess.run(['pkill', '-u', os.environ.get('USER', ''), '-f', f'ray.*{OUT/name}'], check=False)
    text = log.read_text(errors='replace')
    peak = 0
    for line in (OUT/name/'memory.csv').read_text().splitlines():
        try:
            peak = max(peak, int(line.split(',')[1]))
        except (IndexError, ValueError):
            pass
    tok = [float(x) for x in re.findall(r"'perf/actor_train_tok_per_s': ([0-9.]+)", text)]
    secs = [float(x) for x in re.findall(r"'perf/actor_train_time': ([0-9.]+)", text)]
    updates = len(re.findall(r'train_one_step end rollout_id=', text))
    grads = [float(x) for x in re.findall(r'train_one_step end [^\n]*grad_norm=([0-9.eE+-]+|nan|inf)', text)]
    oom = bool(re.search(r'CUDA out of memory|OutOfMemoryError', text))
    result = dict(case=name, tp=tp, dp=8//tp, recompute=recompute, exit_code=code, seconds=round(time.time()-started),
                  oom=oom, peak_memory_mib=peak, headroom_ok=peak <= HEADROOM*GPU_MIB, tok_per_s_per_gpu=tok[-1] if tok else None,
                  train_seconds=secs[-1] if secs else None, optimizer_updates_logged=updates, grad_norms=grads,
                  loaded_step=step, load_dir=str(load_dir) if load_dir else None,
                  saved=(OUT/name/'save'/f'iter_{step+1:07d}').is_dir())
    result['fits'] = code == 0 and not oom and result['headroom_ok'] and result['tok_per_s_per_gpu'] is not None
    result_path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)
    return result


def production_tok_per_s():
    text = (G8/'training-385579-to70.log').read_text(errors='replace')
    vals = [float(x) for x in re.findall(r"'perf/actor_train_tok_per_s': ([0-9.]+)", text)]
    return sum(vals)/len(vals) if vals else None


def run_all(job):
    results = []
    for family, tp in (('tp1', 1), ('tp2', 2)):
        for recompute in (('none', 'selective', 'full') if tp == 1 else ('none', 'selective')):
            r = run_case(job, f'{family}-{recompute}', tp, recompute)
            results.append(r)
            if r['fits']:
                break
    prod = production_tok_per_s()
    fits = [r for r in results if r['fits']]
    best = max(fits, key=lambda r: r['tok_per_s_per_gpu']) if fits else None
    if best is None or prod is None or best['tok_per_s_per_gpu'] < 1.1*prod:
        results.append(run_case(job, 'tp2-full-baseline', 2, 'full'))
    summary = dict(production_tp2_full_tok_per_s_per_gpu=prod, batch=json.loads((OUT/'replay-batch.json').read_text()),
                   cases=results, best=best and best['case'],
                   speedup_vs_production=best and prod and best['tok_per_s_per_gpu']/prod)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def followup(job):
    """Approved 2026-10-25 (8 H200 / 64 CPU / 960 GiB, 1 h, account vlm): production TP2/DP4 full
    recompute on the identical batch, then reload the TP2 no-recompute save (optimizer included)
    and train one more rollout from it."""
    base = run_case(job, 'tp2-full-baseline', 2, 'full')
    reload = run_case(job, 'tp2-none-reload', 2, 'none', OUT/'tp2-none'/'save', CKPT_STEP+1)
    none = json.loads((OUT/'tp2-none'/'result.json').read_text())
    reload['reload_ok'] = (reload['exit_code'] == 0 and reload['optimizer_updates_logged'] > 0 and reload['saved']
                           and all(g == g and g < 1e3 for g in reload['grad_norms']))
    summary = dict(same_batch_speedup=none['tok_per_s_per_gpu']/base['tok_per_s_per_gpu'] if base['tok_per_s_per_gpu'] else None,
                   same_batch_train_seconds=dict(none=none['train_seconds'], full=base['train_seconds']),
                   peak_memory_mib=dict(none=none['peak_memory_mib'], full=base['peak_memory_mib']),
                   baseline=base, reload=reload)
    (OUT/'followup-summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-batch', nargs='*', type=Path)
    p.add_argument('--run', action='store_true')
    p.add_argument('--job-id')
    p.add_argument('--preview', action='store_true')
    p.add_argument('--followup', action='store_true')
    p.add_argument('--baseline-from-save', action='store_true')
    a = p.parse_args()
    if a.build_batch is not None:
        build_batch(a.build_batch or [RUN/f'rollout_recovery/{i}.pt' for i in (60, 61, 62, 63)])
    elif a.run:
        if os.getenv('SLURM_JOB_ID') != a.job_id:
            raise SystemExit('Run inside the approved allocation')
        if a.baseline_from_save:
            baseline_from_save(a.job_id)
        else:
            followup(a.job_id) if a.followup else run_all(a.job_id)
    else:
        _, cmd, env = case_command('preview', 1, 'none')
        print(' '.join(cmd)); print({k: env[k] for k in ('TP_SIZE', 'NUM_ROLLOUT', 'SAVE_DIR', 'SLIME_CKPT_STEP', 'RECOMPUTE_ACTIVATIONS')})
