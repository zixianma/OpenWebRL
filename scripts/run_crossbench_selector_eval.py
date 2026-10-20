#!/usr/bin/env python3
"""Prepare, launch-script, execute and summarize crossbench selector evaluations (see openwebrl/crossbench_selector_eval.py).

  prepare       freeze <root>/<condition-slug>/<benchmark>/plan.json (seeded --n subset, head sha256, protocol, caps)
  launch-script write an sbatch for ONE condition running one or more benchmarks sequentially on 1 H200 (never submits)
  execute       inside the job: start sglang actor (+ capture hooks for head5, + selector servers for piotr5), wait for
                health, run each benchmark's worker, tear everything down
  summarize     success rate, valid-only rate, counts and judge spend per (condition, benchmark)
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl import crossbench_selector_eval as core  # noqa: E402

PYTHON = core.RUNTIME / 'venv/bin/python'
HOURS_PER_BENCHMARK = {'sft1': 1.5, 'piotr5': 2.5, 'head5': 2.5}
STARTUP_HOURS = 0.5


def condition_dir(root, condition):
    return Path(root).resolve() / core.parse_condition(condition)['slug']


def parse_caps(values, benchmarks, n):
    """--judge-max-usd online_mind2web=3 webvoyager=15 ... (optional per-benchmark overrides)."""
    caps = {b: core.default_judge_caps(b, n) for b in benchmarks}
    for item in values or []:
        bench, value = item.split('=', 1)
        caps[bench]['max_usd'] = float(value)
    return caps


def cmd_prepare(a):
    caps = parse_caps(a.judge_max_usd, a.benchmarks, a.n)
    if a.judge_max_calls:
        for b in caps:
            caps[b]['max_calls'] = a.judge_max_calls
    base = condition_dir(a.root, a.condition)
    out = {}
    for bench in a.benchmarks:
        if (a.temperature is None) != (a.top_p is None):
            raise SystemExit('--temperature and --top-p go together')
        decoding = dict(temperature=a.temperature, top_p=a.top_p) if a.temperature is not None else None
        subset_of = condition_dir(a.subset_of_root, a.condition) / bench / 'plan.json' if a.subset_of_root else None
        horizon = dict(max_steps=a.max_steps, history_full_turns=a.history_full_turns) if a.max_steps else None
        plan = core.build_plan(base / bench, benchmark=bench, condition=a.condition, n=a.n, seed=a.seed,
                               actor=a.actor, judge_caps=caps[bench], decoding=decoding, subset_of=subset_of, horizon=horizon)
        out[bench] = dict(plan=str(base / bench / 'plan.json'), tasks=len(plan['task_ids']),
                          first_tasks=plan['task_ids'][:3], judge_caps=plan['judge_caps'],
                          head_sha256=plan['condition']['head_sha256'])
    print(json.dumps(dict(condition_dir=str(base), condition=a.condition, plans=out), indent=2))


def default_time(condition_name, benchmarks):
    hours = STARTUP_HOURS + HOURS_PER_BENCHMARK[condition_name] * len(benchmarks)
    minutes = int(round(min(hours, 24) * 60))
    return f'{minutes // 60:02d}:{minutes % 60:02d}:00'


def sbatch_text(cond_dir, benchmarks, *, time_limit, account, partition, qos, job_name):
    execute = [str(PYTHON), str(REPO / 'scripts/run_crossbench_selector_eval.py'), 'execute', '--dir', str(cond_dir),
               '--benchmarks', *benchmarks]
    return '\n'.join([
        '#!/usr/bin/env bash',
        f'#SBATCH --job-name={job_name}',
        f'#SBATCH --account={account}',
        f'#SBATCH --partition={partition}',
        f'#SBATCH --qos={qos}',
        '#SBATCH --nodes=1',
        '#SBATCH --ntasks=1',
        '#SBATCH --gpus=h200:1',
        '#SBATCH --cpus-per-task=8',
        '#SBATCH --mem=120G',
        f'#SBATCH --time={time_limit}',
        '#SBATCH --signal=B:TERM@300',
        f'#SBATCH --output={cond_dir}/slurm-%j.log',
        'set -euo pipefail',
        f'source {shlex.quote(str(REPO / "scripts/h200_env.sh"))}',
        'export WANDB_PROJECT=openwebrl-evals PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1',
        # exec keeps the controller as the batch step's main process so TERM reaches its teardown.
        'exec ' + shlex.join(execute),
        ''])


def cmd_launch_script(a):
    cond_dir = condition_dir(a.root, a.condition)
    plans = [core.validate_plan(cond_dir / b) for b in a.benchmarks]       # prepared and unchanged
    name = plans[0]['condition']['name']
    time_limit = a.time or default_time(name, a.benchmarks)
    path = cond_dir / f"launch-{'-'.join(a.benchmarks)}.sbatch"
    path.write_text(sbatch_text(cond_dir, a.benchmarks, time_limit=time_limit, account=a.account, partition=a.partition,
                                qos=a.qos, job_name=f"xbench-{plans[0]['condition']['slug']}"))
    print(json.dumps(dict(sbatch=str(path), time=time_limit, resources=dict(gpus='h200:1', cpus=8, mem='120G'),
                          account=a.account, partition=a.partition, qos=a.qos, benchmarks=a.benchmarks,
                          submit=f'sbatch {path}  # requires explicit approval'), indent=2))


# --- execute -------------------------------------------------------------------------------------------------------
def source_command(argv):
    return ['bash', '-c', f'set -euo pipefail\nsource {shlex.quote(str(REPO / "scripts/h200_env.sh"))}\nexec {shlex.join(argv)}']


def lease_ports(job, size=64):
    """Node-local cooperative port-block lease (copied from scripts/runtime_ports.py); keep the handle open."""
    directory = Path(f'/tmp/openwebrl-model-ports-{os.getuid()}')
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    bases = list(range(20000, 28000, 256))
    offset = int(job) % len(bases)
    for base in bases[offset:] + bases[:offset]:
        handle = (directory / f'{base}.lock').open('a+')
        probes = []
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for port in range(base, base + 256):
                probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                probes.append(probe)
                probe.bind(('0.0.0.0', port))
            handle.seek(0); handle.truncate(); handle.write(str(job) + '\n'); handle.flush()
            return handle, base
        except (BlockingIOError, OSError):
            handle.close()
        finally:
            for probe in probes:
                probe.close()
    raise RuntimeError('No free isolated model-server port block')


def hook_specs_json(layers=(18, 27, 36)):
    from openwebrl.actor_head_capture import hook_specs
    return json.dumps(hook_specs(tuple(layers)))


def server_stages(plan, port, capture_dir):
    """Ordered stages of servers; each stage is started and must be healthy before the next one starts."""
    cond = plan['condition']
    actor_argv = [str(PYTHON), '-m', 'sglang.launch_server', '--model-path', plan['actor'], '--host', '127.0.0.1',
                  '--port', str(port), '--dtype', 'bfloat16', '--tp', '1', '--mem-fraction-static', '.45',
                  '--context-length', str(plan['protocol']['context_length']), '--max-running-requests', '40',
                  '--chunked-prefill-size', '4096', '--cuda-graph-max-bs', '48',
                  '--random-seed', str(plan['protocol']['server_seed'])]
    actor_env = {}
    if cond['kind'] == 'actor_cache_head':
        actor_argv += ['--forward-hooks', hook_specs_json(core.actor_info(plan['actor'])['capture_layers'])]
        actor_env = dict(ACTORHEAD_DIR=capture_dir, ACTORHEAD_HEAD=cond['head_path'], ACTORHEAD_TOKENIZER=plan['actor'],
                         ACTORHEAD_PARAMS=str(core.actor_info(plan['actor'])['params']))
    actor = dict(name='actor', argv=source_command(actor_argv), env=actor_env, gpu=True,
                 health=f'http://127.0.0.1:{port}/get_model_info', expect=('model_path', plan['actor']), timeout=1200)
    if cond['kind'] != 'selection':
        return [[actor]]
    # piotr5: the selector's sglang server claims its .30 first, then the actor its .45 (as selectors-samehw).
    selector_sglang = dict(name='selector-sglang', gpu=True, env={}, timeout=1200,
        argv=source_command([str(PYTHON), '-m', 'sglang.launch_server', '--model-path', str(core.PIOTR_SGLANG_DIR),
                             '--host', '127.0.0.1', '--port', str(port + 2), '--dtype', 'bfloat16',
                             '--mem-fraction-static', '.30', '--context-length', '32768', '--grammar-backend', 'xgrammar',
                             '--cuda-graph-max-bs', '48', '--random-seed', '0']),
        health=f'http://127.0.0.1:{port + 2}/get_model_info', expect=('model_path', str(core.PIOTR_SGLANG_DIR)))
    selector = dict(name='selector', gpu=False, env=dict(PYTHONPATH=f'{REPO}:{REPO / "scripts"}'), timeout=600,
        argv=[str(PYTHON), str(REPO / 'scripts/serve_arm_sglang.py'), 'serve', '--sglang', f'http://127.0.0.1:{port + 2}',
              '--port', str(port + 1)],
        health=f'http://127.0.0.1:{port + 1}/health', expect=('mode', 'selection'))
    return [[selector_sglang], [actor], [selector]]


def base_environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith(('OPENWEBRL_REPLAY_', 'OPENWEBRL_ARM_'))}
    try:                                      # judge credentials from the repo .env; never printed or logged
        from dotenv import dotenv_values
        for k, v in dotenv_values(REPO / '.env').items():
            if v and k.startswith(('OPENAI_', 'JUDGE_')):
                env.setdefault(k, v)
    except ImportError:
        pass
    env.update(PYTHONPATH=str(REPO), WANDB_PROJECT='openwebrl-evals', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', SLIME_BROWSER_ENV_MODE='local_process')
    env.setdefault('FLASHINFER_WORKSPACE_BASE', str(core.RUNTIME))
    # Keep local model servers off the inherited cluster proxy: pilot 367523/367524 warmup hit squid and died.
    bypass = {item.strip() for name in ('NO_PROXY', 'no_proxy') for item in env.get(name, '').split(',') if item.strip()}
    bypass.update(('localhost', '127.0.0.1', '::1'))
    env['NO_PROXY'] = env['no_proxy'] = ','.join(sorted(bypass))
    return env


def seconds_left(job):
    try:
        text = subprocess.check_output(['squeue', '-h', '-j', str(job), '-o', '%L'], text=True, timeout=30).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    days, _, rest = text.rpartition('-')
    parts = [int(x) for x in rest.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(days) if days else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def get_json(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def stop_children(children, grace=30):
    report = []
    for name, child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + grace
    for name, child in children:
        try:
            child.wait(timeout=max(0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=30)
        report.append(dict(name=name, pid=child.pid, returncode=child.returncode))
    return report


def cmd_execute(a):
    cond_dir = Path(a.dir).resolve()
    plans = {b: core.validate_plan(cond_dir / b, allow_source_drift=a.allow_source_drift) for b in a.benchmarks}
    first = plans[a.benchmarks[0]]
    if any(p['condition'] != first['condition'] or p['actor'] != first['actor'] for p in plans.values()):
        raise ValueError('All benchmarks of one job must share the condition and actor')
    job = os.environ.get('SLURM_JOB_ID')
    if a.dry_run:
        capture = f'/dev/shm/actorhead-{job or "JOB"}-{first["condition"]["slug"]}'
        stages = server_stages(first, 20000, capture)
        print(json.dumps(dict(condition=first['condition'], benchmarks={b: len(p['task_ids']) for b, p in plans.items()},
                              drift={b: core.source_drift(p) for b, p in plans.items()},
                              stages=[[dict(name=s['name'], argv=s['argv'][-1] if s['argv'][0] == 'bash' else shlex.join(s['argv']),
                                            env=s['env'], gpu=s['gpu']) for s in st] for st in stages]), indent=2))
        return
    if not job:
        raise SystemExit('execute must run inside the Slurm job (use --dry-run elsewhere)')
    devices = [d for d in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if d]
    if len(devices) != 1:
        raise ValueError(f'Exactly one allocated GPU required, got {devices}')
    left = seconds_left(job)
    deadline = time.time() + left - 600 if left else None          # leave 10 min for teardown and summaries
    env = base_environment()
    os.environ.update(NO_PROXY=env['NO_PROXY'], no_proxy=env['no_proxy'])   # this process polls the servers too
    journal = cond_dir / 'scheduler-attempts' / str(job)
    journal.mkdir(parents=True, exist_ok=False)
    capture_dir = f'/dev/shm/actorhead-{job}-{first["condition"]["slug"]}'
    children, handles, lease = [], [], None

    def spawn(name, argv, extra_env, gpu):
        handle = (journal / f'{name}.log').open('a')
        handles.append(handle)
        child_env = dict(env, **extra_env, CUDA_VISIBLE_DEVICES=devices[0] if gpu else '')
        child = subprocess.Popen(argv, cwd=REPO, env=child_env, stdout=handle, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        children.append((name, child))
        return child

    def alive_services():
        return all(c.poll() is None for n, c in children if n not in ('gpu',) and not n.startswith('worker-'))

    def interrupted(sig, frame):
        raise SystemExit(128 + sig)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    outcome = {}
    with (cond_dir / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            lease, port = lease_ports(job)
            core.write_json(journal / 'allocation.json', dict(job_id=job, port=port, device=devices[0], seconds_left=left,
                            benchmarks=a.benchmarks, condition=first['condition'],
                            drift={b: core.source_drift(p) for b, p in plans.items()},
                            actual_source_digests=core.source_digests()))
            if first['condition']['kind'] == 'actor_cache_head':
                Path(capture_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
            for stage in server_stages(first, port, capture_dir):
                pending = {}
                for s in stage:
                    spawn(s['name'], s['argv'], s['env'], s['gpu'])
                    pending[s['name']] = s
                started = time.time()
                while pending:
                    if not alive_services():
                        raise RuntimeError('A model server exited during startup; see ' + str(journal))
                    for name, s in list(pending.items()):
                        if time.time() - started > s['timeout']:
                            raise TimeoutError(f'{name} startup timeout')
                        try:
                            card = get_json(s['health'])
                        except OSError:
                            continue
                        key, expected = s['expect']
                        if str(card.get(key)) != expected:
                            raise ValueError(f'{name} identity mismatch: {key}={card.get(key)!r}')
                        core.write_json(journal / f'{name}-identity.json', card)
                        pending.pop(name)
                    if pending:
                        time.sleep(3)
            info = get_json(f'http://127.0.0.1:{port}/server_info', timeout=10)
            if info.get('random_seed') != first['protocol']['server_seed']:
                raise ValueError('Actor server seed differs from the plan')
            core.write_json(journal / 'actor-server-info.json',
                            {k: info.get(k) for k in ('random_seed', 'sampling_backend', 'enable_deterministic_inference',
                                                      'forward_hooks', 'mem_fraction_static', 'cuda_graph_max_bs')})
            spawn('gpu', ['nvidia-smi', '--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw',
                          '--format=csv', '-l', '30'], {}, True)
            for bench in a.benchmarks:
                plan = plans[bench]
                done = len(core.load_records(plan))
                if done == len(plan['task_ids']):
                    outcome[bench] = 'already_complete'
                    continue
                if (Path(plan['judge_budget_root']) / 'halt.json').exists():
                    outcome[bench] = 'judge_halted'
                    continue
                if deadline and time.time() > deadline - 900:
                    outcome[bench] = 'not_started_time'
                    continue
                config = dict(plan_dir=str(cond_dir / bench), actor_port=port, selector_endpoint=f'http://127.0.0.1:{port + 1}',
                              browser_port_start=port + 10, capture_dir=capture_dir, scheduler_job_id=job,
                              allow_source_drift=a.allow_source_drift)
                config_path = journal / f'worker-{bench}.json'
                core.write_json(config_path, config)
                worker = spawn(f'worker-{bench}', source_command([str(PYTHON), '-m', 'openwebrl.crossbench_selector_eval',
                                                                  '--config', str(config_path)]), {}, True)
                while worker.poll() is None:
                    core.write_json(cond_dir / 'current.json', dict(job_id=job, benchmark=bench, epoch=time.time(),
                                    records=len(list((Path(plan['collection']) / 'records').glob('*.json'))),
                                    expected=len(plan['task_ids'])))
                    if not alive_services():
                        raise RuntimeError(f'Model service died during {bench}')
                    if deadline and time.time() > deadline:
                        os.killpg(worker.pid, signal.SIGTERM)
                        worker.wait(timeout=120)
                        outcome[bench] = 'stopped_time'
                        break
                    time.sleep(15)
                else:
                    outcome[bench] = 'complete' if worker.returncode == 0 else f'worker_failed_{worker.returncode}'
                if outcome[bench] == 'stopped_time':
                    break
        finally:
            report = stop_children(children)
            if first['condition']['kind'] == 'actor_cache_head':
                shutil.rmtree(capture_dir, ignore_errors=True)      # job-private node-local capture exports
            core.write_json(journal / 'teardown.json', dict(processes=report, outcome=outcome, epoch=time.time()))
            if lease is not None:
                lease.close()
            for h in handles:
                h.close()
    print(json.dumps(dict(outcome=outcome), indent=2))


# --- summarize -----------------------------------------------------------------------------------------------------
def cmd_summarize(a):
    dirs = []
    for d in (Path(x).resolve() for x in a.dir):
        if (d / 'plan.json').exists():
            dirs.append(d)
        else:
            dirs += [p.parent for p in d.glob('*/plan.json')] + [p.parent for p in d.glob('*/*/plan.json')]
    rows = []
    for d in sorted(set(dirs)):
        plan = core.read_json(d / 'plan.json')
        rows.append(core.summarize(core.load_records(plan), plan, core.judge_state(plan)))
    if a.json:
        print(json.dumps(rows, indent=2))
        return
    fmt = lambda x: '-' if x is None else f'{100 * x:.1f}'
    print(f"{'condition':<20} {'benchmark':<16} {'done':>5} {'sched':>5} {'succ':>5} {'valid':>5} {'SR%':>6} "
          f"{'SR_done%':>8} {'valid%':>7} {'judge$':>7} {'calls':>6} halt")
    for r in rows:
        print(f"{r['condition_slug']:<20} {r['benchmark']:<16} {r['completed']:>5} {r['scheduled']:>5} {r['successes']:>5} "
              f"{r['valid']:>5} {fmt(r['success_rate_scheduled']):>6} {fmt(r['success_rate_completed']):>8} "
              f"{fmt(r['valid_only_rate']):>7} {r['judge_usd']:>7.2f} {r['judge_calls']:>6} {'Y' if r['judge_halted'] else ''}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    pr = sub.add_parser('prepare')
    pr.add_argument('--root', required=True)
    pr.add_argument('--condition', required=True, help='sft1 | piotr5 | head5:<path/to/head.pt>')
    pr.add_argument('--benchmarks', nargs='+', default=list(core.BENCHMARKS), choices=core.BENCHMARKS)
    pr.add_argument('--n', type=int, default=100)
    pr.add_argument('--seed', type=int, required=True, help='task-subset seed')
    pr.add_argument('--actor', default=str(core.ACTOR))
    pr.add_argument('--judge-max-usd', nargs='*', help='per-benchmark overrides, e.g. webvoyager=20')
    pr.add_argument('--judge-max-calls', type=int)
    pr.add_argument('--temperature', type=float, help='actor decoding override (with --top-p); default T 0.7 / top-p 0.9')
    pr.add_argument('--top-p', type=float)
    pr.add_argument('--subset-of-root', help='take the first --n tasks of the same condition/benchmark plan under this root')
    pr.add_argument('--max-steps', type=int, help='horizon override (default 30)')
    pr.add_argument('--history-full-turns', type=int, default=0, help='with --max-steps: keep the last N turns full, older turns action + result line only')
    ls = sub.add_parser('launch-script')
    ls.add_argument('--root', required=True)
    ls.add_argument('--condition', required=True)
    ls.add_argument('--benchmarks', nargs='+', default=list(core.BENCHMARKS), choices=core.BENCHMARKS)
    ls.add_argument('--time', help='HH:MM:SS (default: 0.5h + per-benchmark estimate)')
    ls.add_argument('--account', default='vlm')
    ls.add_argument('--partition', default='gpu-h200')
    ls.add_argument('--qos', default='normal')
    ex = sub.add_parser('execute')
    ex.add_argument('--dir', required=True, help='condition directory <root>/<condition-slug>')
    ex.add_argument('--benchmarks', nargs='+', default=list(core.BENCHMARKS), choices=core.BENCHMARKS)
    ex.add_argument('--dry-run', action='store_true', help='validate plans and print server commands; start nothing')
    ex.add_argument('--allow-source-drift', action='store_true', help='run although tracked source files changed since prepare')
    su = sub.add_parser('summarize')
    su.add_argument('--dir', nargs='+', required=True, help='root, condition directories or benchmark directories')
    su.add_argument('--json', action='store_true')
    a = p.parse_args(argv)
    dict(prepare=cmd_prepare, **{'launch-script': cmd_launch_script}, execute=cmd_execute, summarize=cmd_summarize)[a.command](a)


if __name__ == '__main__':
    main()
