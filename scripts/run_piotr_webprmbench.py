#!/usr/bin/env python3
"""Own one approved, bounded, text-only Piotr WEBPRMBench evaluation.

The unchanged upstream evaluator owns prompts, parsing and metrics. SGLang
supplies raw completions through an explicit vLLM-interface compatibility shim.
This worker never submits jobs or downloads models, and never retries inference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, build_opener, ProxyHandler

from openwebrl.webprm_native_release import benchmark_inventory, reference
from openwebrl.webprm_static_bench import run_official
from openwebrl.webprm_lifecycle import OwnedEpisodeTree, _processes, _subreaper

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
RESOURCES = dict(account='vlm', partition='gpu-h200', gpu_type='h200', gpus=1,
                 cpus=8, memory_gib=240, seconds=7200, maximum_allocations=2)
LIMITS = dict(api_usd=0, provider_calls=0, model_download_bytes=0,
              local_pair_generations=4600, scratch_bytes=32 * 1024**3,
              retained_bytes=2 * 1024**3, cleanup_seconds=120)
PARAMS = dict(temperature=0., top_p=1., max_tokens=1024, stop=['</Answer>'])


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())


def safe_output(path):
    path = Path(path).absolute()
    if RUNTIME not in path.parents or path.resolve() != path:
        raise ValueError('Output must be a canonical approved runtime descendant')
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError('Writable output cannot traverse symlinks')
    return path


def check_bindings(manifest):
    if manifest['resources'] != RESOURCES or manifest['limits'] != LIMITS:
        raise ValueError('Resource or budget envelope differs')
    if manifest['sampling'] != PARAMS or manifest['pairs'] != 4600 or manifest['states'] != 1150:
        raise ValueError('Frozen evaluation protocol differs')
    for item in manifest['files']:
        path = Path(item['path'])
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024**2), b''):
                h.update(block)
        if h.hexdigest() != item['sha256']:
            raise ValueError('Frozen input changed: ' + str(path))
    benchmark_inventory(manifest['data'])


def check_approval(manifest_path, manifest, approval, job_id):
    if (approval.get('status') != 'approved'
            or approval.get('manifest_sha256') != reference(manifest_path)['sha256']
            or approval.get('resources') != RESOURCES or approval.get('limits') != LIMITS
            or str(approval.get('job_id')) != job_id):
        raise ValueError('Exact user-approved, job-bound resource request required')


def allocation_seconds(approval):
    """Debit verified startup failures before allocating the remaining clock."""
    previous = approval.get('prior_attempts', [])
    if not isinstance(previous, list) or len(previous) >= RESOURCES['maximum_allocations']:
        raise ValueError('Static benchmark allocation attempt cap reached')
    elapsed = 0
    jobs = set()
    for item in previous:
        proof_path = safe_output(item['proof']['path'])
        if reference(proof_path) != item['proof']:
            raise ValueError('Prior allocation proof changed')
        proof = json.loads(proof_path.read_text())
        job = proof.get('job_id')
        seconds = proof.get('elapsed_seconds')
        if (not isinstance(job, str) or not job.isdigit() or job in jobs
                or type(seconds) is not int or seconds <= 0
                or proof.get('scheduler_state') not in {'FAILED', 'COMPLETED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY'}
                or proof.get('remaining_owned_processes') != {}
                or proof.get('local_pair_generations') != 0):
            raise ValueError('Prior startup attempt requires exact terminal closure and zero generations')
        jobs.add(job)
        elapsed += seconds
    if approval.get('prior_gpu_seconds', 0) != elapsed:
        raise ValueError('Prior GPU time accounting differs')
    # Request whole minutes so scheduler rounding cannot exceed the total cap.
    remaining = (RESOURCES['seconds'] - elapsed) // 60 * 60
    if remaining <= LIMITS['cleanup_seconds']:
        raise ValueError('No useful static benchmark time remains')
    return remaining


def slurm_time(seconds):
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def short_ipc_directory(job):
    if not job.isdigit() or len(job) > 20:
        raise ValueError('Bounded numeric allocation identity required')
    return Path(tempfile.mkdtemp(prefix='wpb-' + job + '-', dir='/tmp'))


class ProcessAdapter:
    """Expose the lifecycle tracker's multiprocessing interface for Popen."""
    def __init__(self, process):
        self.process, self.pid = process, process.pid

    @property
    def exitcode(self):
        return self.process.poll()

    def is_alive(self):
        return self.process.poll() is None

    def join(self, timeout=0):
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass


class RawBackend:
    def __init__(self, endpoint, output, deadline, guard):
        if not endpoint.startswith('http://127.0.0.1:'):
            raise ValueError('Only the separately owned loopback server is allowed')
        self.endpoint, self.output, self.deadline, self.guard = endpoint, Path(output), deadline, guard
        self.pairs = 0
        self.batch = 0
        self.http = build_opener(ProxyHandler({}))

    def __call__(self, texts, params):
        self.guard()
        if params != PARAMS or self.pairs + len(texts) > 4600:
            raise ValueError('Sampling or generation cap differs')
        payload = dict(text=texts, sampling_params=dict(
            temperature=params['temperature'], top_p=params['top_p'],
            max_new_tokens=params['max_tokens'], stop=params['stop']))
        prefix = self.output / f'batch-{self.batch:04d}'
        write_new(prefix.with_suffix('.request.json'), payload)
        # Attempt is consumed before transport; failures cannot silently retry.
        self.pairs += len(texts)
        self.batch += 1
        remaining = self.deadline - time.time()
        if remaining <= 0:
            raise TimeoutError('Inference deadline reached')
        req = Request(self.endpoint + '/generate', json.dumps(payload).encode(),
                      {'Content-Type': 'application/json'})
        with self.http.open(req, timeout=min(600, remaining)) as handle:
            response = json.load(handle)
        write_new(prefix.with_suffix('.response.json'), response)
        if not isinstance(response, list) or len(response) != len(texts):
            raise ValueError('Raw server returned an incomplete batch')
        if any(not isinstance(row.get('text'), str) for row in response):
            raise ValueError('Raw server returned a malformed completion')
        self.guard()
        return [row['text'] for row in response]


def audit_outputs(data, evaluator, output):
    """Independently recount all pairs/states and bind saved raw completions."""
    output = Path(output)
    rows = [json.loads(x) for x in Path(data).read_text().splitlines() if x.strip()]
    saved = [json.loads(x) for x in (output / 'pairs.jsonl').read_text().splitlines() if x.strip()]
    raw = []
    request_prompts = []
    for req in sorted((output / 'transport').glob('batch-*.request.json')):
        payload = json.loads(req.read_text())
        if payload['sampling_params'] != dict(temperature=0., top_p=1., max_new_tokens=1024, stop=['</Answer>']):
            raise ValueError('Saved generation parameters differ')
        response = json.loads(req.with_name(req.name.replace('.request.', '.response.')).read_text())
        if len(response) != len(payload['text']):
            raise ValueError('Saved batch is incomplete')
        request_prompts.extend(payload['text']); raw.extend(r['text'] for r in response)
    if len(rows) != len(saved) or len(rows) != len(raw) or request_prompts != [r['prompt'] for r in rows]:
        raise ValueError('Incomplete or misaligned evaluation evidence')
    parser = runpy.run_path(str(evaluator), run_name='audit_only')['parse_answer']
    states, sources = defaultdict(list), defaultdict(list)
    for row, result, text in zip(rows, saved, raw):
        pred = parser(text)
        if (any(result[k] != row[k] for k in ('source_name', 'state_idx', 'pair_idx', 'expected'))
                or result['raw'] != text or result['pred'] != pred
                or result['correct'] != (pred == row['expected'])):
            raise ValueError('Prediction/label/raw-text binding differs')
        states[(row['source_name'], row['state_idx'])].append(pred == row['expected'])
        sources[row['source_name']].append(pred == row['expected'])
    if any(len(panel) != 4 for panel in states.values()):
        raise ValueError('State panel incomplete')
    per_source = {}
    for source, pairs in sources.items():
        panels = [p for (s, _), p in states.items() if s == source]
        per_source[source] = dict(pairs_total=len(pairs), pairs_ok=sum(pairs), pair_acc=sum(pairs)/len(pairs),
                                  states_total=len(panels), states_ok=sum(all(p) for p in panels),
                                  state_acc=sum(all(p) for p in panels)/len(panels))
    overall = dict(pairs_total=len(rows), pairs_ok=sum(sum(p) for p in states.values()),
                   pair_acc=sum(sum(p) for p in states.values())/len(rows),
                   states_total=len(states), states_ok=sum(all(p) for p in states.values()),
                   state_acc=sum(p['state_acc'] for p in per_source.values())/len(per_source))
    metrics = dict(overall=overall, per_source=per_source)
    if metrics != json.loads((output / 'metrics.json').read_text()):
        raise ValueError('Independent recount differs from official metrics')
    return dict(verified=True, metrics=metrics, raw_request_response_binding=True,
                pair_predictions=len(saved), states=len(states), source_macro_state_accuracy=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--approval', required=True)
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    approval = json.loads(Path(args.approval).read_text())
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit():
        raise ValueError('Requires its own approved Slurm allocation')
    check_approval(args.manifest, manifest, approval, job)
    allowed_seconds = allocation_seconds(approval)
    check_bindings(manifest)
    receipt = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True)
    fields = dict(part.split('=', 1) for part in receipt.split() if '=' in part)
    if (fields.get('Account') != 'vlm' or fields.get('Partition') != 'gpu-h200'
            or fields.get('NumCPUs') != '8' or fields.get('TimeLimit') != slurm_time(allowed_seconds)
            or fields.get('MinMemoryNode') != '240G'
            or 'gres/gpu:h200=1' not in fields.get('AllocTRES', '')):
        raise ValueError('Actual allocation differs from exact approved resources')
    start = time.mktime(time.strptime(fields['StartTime'], '%Y-%m-%dT%H:%M:%S'))
    deadline = start + allowed_seconds - LIMITS['cleanup_seconds']
    if list(Path(manifest['root']).glob('attempt-*')):
        raise ValueError('The single allocation attempt has already been consumed')
    output = safe_output(Path(manifest['root']) / ('attempt-' + job))
    output.mkdir(exist_ok=False)
    (output / 'transport').mkdir()
    scratch = output / 'scratch'; scratch.mkdir()
    ipc_tmp = short_ipc_directory(job)
    write_new(output/'runtime-paths.json', dict(scratch=str(scratch), ipc_tmp=str(ipc_tmp),
              combined_scratch_cap_bytes=LIMITS['scratch_bytes']))
    write_new(output / 'started.json', dict(job_id=job, scheduler=receipt, started_epoch=time.time(),
              inference_deadline=deadline, manifest=reference(args.manifest), scope='text-only checkpoint evaluation'))
    env = {k:v for k,v in os.environ.items() if not any(s in k for s in ('API_KEY', 'AUTH_TOKEN', 'ACCESS_TOKEN'))}
    for key in ['TMPDIR', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_RUNTIME_DIR',
                'HF_HOME', 'HF_HUB_CACHE', 'HF_MODULES_CACHE', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR',
                'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'FLASHINFER_WORKSPACE_BASE',
                'CUDA_CACHE_PATH', 'SGLANG_CACHE_DIR', 'SGLANG_DG_CACHE_DIR', 'MPLCONFIGDIR',
                'NUMBA_CACHE_DIR', 'OUTLINES_CACHE_DIR', 'HF_DATASETS_CACHE', 'CCACHE_DIR']:
        directory = scratch / key.lower(); directory.mkdir(); env[key] = str(directory)
    # Unix-domain socket paths must fit sockaddr_un.sun_path (107 bytes).
    # Keep caches in the accounted durable scratch, and IPC in a short owned path.
    env['TMPDIR'] = str(ipc_tmp)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', WANDB_MODE='disabled',
               WANDB_PROJECT='openwebrl-evals', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               CUDA_HOME=str(RUNTIME/'cuda'), PATH=str(RUNTIME/'cuda/bin')+':'+env.get('PATH',''),
               LD_LIBRARY_PATH=str(RUNTIME/'cuda/lib64')+':'+env.get('LD_LIBRARY_PATH',''))
    if Path(env['HOME']).resolve().is_relative_to('/gpfs/home/zixianma'):
        raise ValueError('Forbidden home would be writable by server')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    command = [str(RUNTIME/'venv/bin/python'), '-m', 'sglang.launch_server',
               '--model-path', manifest['model'], '--tokenizer-path', manifest['processor'],
               '--chat-template', str(Path(manifest['processor'])/'chat_template.jinja'),
               '--decrypted-config-file', manifest['compatible_config'],
               '--served-model-name', 'piotr-webprmbench-text-only', '--host', '127.0.0.1',
               '--port', str(port), '--tp', '1', '--dtype', 'bfloat16', '--mem-fraction-static', '.70',
               '--max-total-tokens', '32768', '--context-length', '32768',
               '--max-running-requests', '8', '--random-seed', '42', '--disable-cuda-graph',
               '--disable-custom-all-reduce', '--skip-server-warmup']
    write_new(output/'command.json', command)
    previous = _subreaper(True)
    baseline = {pid for pid, row in _processes().items() if row['ppid'] == os.getpid()}
    process = tree = None
    result = dict(verified_complete=False, new_provider_calls=0, downloaded_bytes=0)
    def interrupted(sig, frame):
        raise InterruptedError(f'Allocation signal {sig}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    # An allocation alarm interrupts startup, transport or upstream evaluation.
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(max(1, int(deadline-time.time())))
    def guard():
        if process is not None and process.poll() is not None:
            raise RuntimeError('Owned inference server exited')
        if tree is not None:
            tree.refresh()
        if time.time() >= deadline:
            raise TimeoutError('Allocation inference deadline reached')
        size = sum(p.stat().st_size for base in (scratch, ipc_tmp)
                   for p in base.rglob('*') if p.is_file())
        retained = sum(p.stat().st_size for p in output.rglob('*') if p.is_file() and scratch not in p.parents)
        if size > LIMITS['scratch_bytes'] or retained > LIMITS['retained_bytes']:
            raise RuntimeError('Storage cap reached')
    try:
        with (output/'server.log').open('x') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            tree = OwnedEpisodeTree(ProcessAdapter(process), baseline)
            backend = RawBackend(f'http://127.0.0.1:{port}', output/'transport', deadline, guard)
            startup_deadline = min(deadline, time.time()+900)
            while True:
                guard()
                try:
                    with backend.http.open(backend.endpoint+'/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    if time.time() >= startup_deadline:
                        raise TimeoutError('Bounded server startup failed')
                    time.sleep(2)
            run_official(manifest['evaluator'], model=manifest['model'], data=manifest['data'],
                         output=output, generate=backend, batch_size=8)
            audit = audit_outputs(manifest['data'], manifest['evaluator'], output)
            if audit['pair_predictions'] != 4600 or audit['states'] != 1150:
                raise ValueError('Full benchmark denominator not verified')
            write_new(output/'independent-audit.json', audit)
            result.update(evaluation_audited=True, audit=reference(output/'independent-audit.json'))
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        closure = tree.stop(20) if tree else dict(remaining_owned_processes={})
        _subreaper(previous)
        write_new(output/'closure.json', closure)
        result.update(finished_epoch=time.time(), closure=reference(output/'closure.json'))
        result['verified_complete'] = bool(result.get('evaluation_audited') and not closure['remaining_owned_processes'])
        write_new(output/'final.json', result)
    if not result['verified_complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
