#!/usr/bin/env python3
"""Isolated browser reset diagnostics; no policy model, judge, or live-run writes."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scaling_benchmark import RUNTIME, SOURCE, case_plan, resources, validate_benchmark_source
from resume_baseline import clean_environment

REPO = Path(__file__).resolve().parents[1]
URLS = [
    'data:text/html,<html><body>Browser diagnostic</body></html>',
    'https://example.com',
    'https://velux.com',
    'https://gtmetrix.com',
    'https://lens.blogs.nytimes.com',
    'https://discworld.fandom.com',
]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def trial_specs():
    # Same 96-URL sequence for every concurrent case, two modes per level.
    return [(1, 0.0, 6)] + [(n, delay, 96) for n in (32, 64, 96) for delay in (0.0, 0.25)]


def classify_error(text):
    if 'Page.screenshot' in text:
        return 'screenshot'
    if 'Page.goto' in text or 'Failed to navigate' in text:
        return 'navigation'
    return 'other'


def suspected_challenge(text):
    text = text.lower()
    return any(x in text for x in ('verify you are human', 'performing security verification',
                                  'checking your browser', 'just a moment...'))


def summarize(records):
    done = [x for x in records if not x.get('skipped')]
    passed = [x for x in done if x.get('success')]
    by_url = {}
    for row in done:
        item = by_url.setdefault(row['url'], {'attempts': 0, 'successes': 0})
        item['attempts'] += 1
        item['successes'] += bool(row.get('success'))
    return {'attempts': len(done), 'successes': len(passed),
            'success_rate': len(passed) / len(done) if done else None,
            'suspected_challenges': sum(bool(x.get('suspected_challenge')) for x in done),
            'by_url': by_url, 'skipped': len(records) - len(done)}


def confirmation_candidate(results):
    passed = [x['concurrency_limit'] for x in results
              if x['concurrency_limit'] in (64, 96) and x['launch_interval_seconds'] == 0
              and x['attempts'] == 96 and not x['skipped'] and x['success_rate'] >= 0.95]
    return max(passed) if passed else None


async def browser_worker(plan, output):
    """Measure the original setup+reset path, including its two navigations."""
    began = time.monotonic()
    sys.path.insert(0, str(SOURCE))
    import yaml
    from playwright.async_api import Page
    from openwebrl.env.web_env import WebEnv
    row = {'url': plan['url'], 'success': False, 'timings': {}, 'navigation_calls': []}
    row['timings']['python_import'] = time.monotonic() - began
    original_goto = Page.goto

    async def timed_goto(page, *args, **kwargs):
        start = time.monotonic()
        call = {'url': args[0] if args else kwargs.get('url')}
        try:
            result = await original_goto(page, *args, **kwargs)
            call['success'] = True
            return result
        except Exception as exc:
            call.update(success=False, error=str(exc))
            raise
        finally:
            call['seconds'] = time.monotonic() - start
            row['navigation_calls'].append(call)

    Page.goto = timed_goto

    class TimedEnv(WebEnv):
        async def _initialize_context(self, *args, **kwargs):
            row['timings'].setdefault('browser_launch', time.monotonic() - launch_started)
            return await super()._initialize_context(*args, **kwargs)

        async def get_screenshot(self):
            start = time.monotonic()
            try:
                return await super().get_screenshot()
            finally:
                row['timings']['screenshot'] = time.monotonic() - start

    cfg = yaml.safe_load((SOURCE/'openwebrl/env/config.yaml').read_text())
    fields = ('width', 'height', 'dpr', 'max_retries', 'wait_timeout',
              'screenshot_timeout', 'resize_output_coords', 'resize_scale', 'image_patch_size')
    env = TimedEnv(**{k: cfg[k] for k in fields}, start_url=plan['url'], tool_list=None, policy=None)
    launch_started = time.monotonic()
    try:
        await env.setup()
        row['timings']['setup'] = time.monotonic() - launch_started
        row['setup_complete_time'] = time.time()
        # Keep real browser sessions open long enough to exercise the requested
        # concurrency even when these short initial-observation probes are fast.
        hold_seconds = float(plan.get('hold_seconds', 0))
        if not 0 <= hold_seconds <= 30:
            raise ValueError('Browser diagnostic hold must be between 0 and 30 seconds')
        await asyncio.sleep(hold_seconds)
        row['reset_started_time'] = time.time()
        start = time.monotonic()
        observation, _ = await env.reset()
        row['timings']['reset'] = time.monotonic() - start
        row.update(success=bool(observation['screenshot']), screenshot_bytes=len(observation['screenshot']))
        image_path = output.with_suffix('.png')
        image_path.write_bytes(observation['screenshot'])
        row['screenshot_path'] = str(image_path)
        try:
            row['page_title'] = await asyncio.wait_for(env.page.title(), timeout=3)
            text = await asyncio.wait_for(env.page.inner_text('body', timeout=2000), timeout=3)
            row['suspected_challenge'] = suspected_challenge(row['page_title'] + '\n' + text)
        except Exception as exc:
            row['page_diagnostic_error'] = str(exc)
    except Exception as exc:
        row.update(error=str(exc), failure_stage=classify_error(str(exc)))
    finally:
        start = time.monotonic()
        try:
            await asyncio.wait_for(env.exit(), timeout=10)
        except Exception as exc:
            row['cleanup_error'] = str(exc)
        row['timings']['cleanup'] = time.monotonic() - start
        row['timings']['total_worker'] = time.monotonic() - began
        save(output, row)


def telemetry():
    rel = Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1].lstrip('/')
    leaf = Path('/sys/fs/cgroup') / rel
    job = next((p for p in (leaf, *leaf.parents) if p.name.startswith('job_')), leaf)
    row = {'time': time.time()}
    for name in ('cpu.stat', 'cpu.pressure', 'memory.current', 'memory.events', 'cpuset.cpus.effective'):
        if (job/name).exists():
            row[name] = (job/name).read_text()
    counts = {'chromium_processes': 0, 'python_processes': 0, 'threads': 0}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if f'/{job.name}/' not in (p/'cgroup').read_text():
                continue
            status = dict(line.split(':', 1) for line in (p/'status').read_text().splitlines() if ':' in line)
            name = status['Name'].strip().lower()
            counts['chromium_processes'] += 'chrome' in name or 'chromium' in name
            counts['python_processes'] += 'python' in name
            counts['threads'] += int(status.get('Threads', 0))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return row | counts


async def run_case(root, concurrency, delay, count, deadline, *, hold_seconds=0):
    case = root / f'c{concurrency}-delay{delay:g}'
    case.mkdir()
    semaphore = asyncio.Semaphore(concurrency)
    launch_lock = asyncio.Lock()
    next_launch = 0.0
    active = peak = 0
    began = time.monotonic()

    async def one(index):
        nonlocal next_launch, active, peak
        url = URLS[index % len(URLS)]
        async with semaphore:
            async with launch_lock:
                await asyncio.sleep(max(0, next_launch-time.monotonic()))
                next_launch = time.monotonic() + delay
            if deadline-time.time() < 230:
                return {'url': url, 'skipped': True}
            plan_path = case/f'{index:04d}.plan.json'
            result_path = case/f'{index:04d}.result.json'
            save(plan_path, {'url': url, 'hold_seconds': hold_seconds})
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='', PLAYWRIGHT_BROWSERS_PATH=str(RUNTIME/'browsers'))
            active += 1
            peak = max(peak, active)
            start = time.monotonic()
            with (case/f'{index:04d}.log').open('w') as log:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, str(Path(__file__).resolve()), '--browser-worker', str(plan_path),
                    '--result', str(result_path), env=env, stdout=log, stderr=log, start_new_session=True)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=220)
                except asyncio.TimeoutError:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        pass
                finally:
                    # Each worker creates its own session; reap only its descendants.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await proc.wait()
                    active -= 1
            result = json.loads(result_path.read_text()) if result_path.exists() else {
                'url': url, 'success': False, 'failure_stage': 'worker_timeout_or_crash', 'exit_code': proc.returncode}
            result['parent_wall_seconds'] = time.monotonic()-start
            save(result_path, result)
            return result

    async def sample():
        with (case/'health.jsonl').open('w', buffering=1) as out:
            while True:
                out.write(json.dumps(telemetry() | {'active_workers': active})+'\n')
                await asyncio.sleep(10)

    sampler = asyncio.create_task(sample())
    try:
        records = await asyncio.gather(*(one(i) for i in range(count)))
    finally:
        sampler.cancel()
        try:
            await sampler
        except asyncio.CancelledError:
            pass
    result = summarize(records) | {'concurrency_limit': concurrency, 'launch_interval_seconds': delay,
        'hold_seconds': hold_seconds,
        'peak_active_workers': peak, 'wall_seconds': time.monotonic()-began}
    save(case/'summary.json', result)
    print(json.dumps(result), flush=True)
    return result


async def controller(job, *, confirm_rl=False):
    info = subprocess.check_output(['scontrol','show','job',job,'-o'], text=True)
    budget = resources(info, job)
    if os.environ.get('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Run inside the separately approved diagnostic allocation')
    validate_benchmark_source(SOURCE)
    root = RUNTIME/'benchmarks'/f'browser-diagnostic-{job}'
    root.mkdir(parents=True, exist_ok=False)
    save(root/'plan.json', {'source': str(SOURCE), 'urls': URLS, 'cases': trial_specs(),
        'allocation': budget, 'no_policy_or_judge': True, 'source_hashes': {
            p: hashlib.sha256((SOURCE/p).read_bytes()).hexdigest()
            for p in ['openwebrl/env/web_env.py','openwebrl/env/config.yaml']}})
    from dotenv import dotenv_values
    for key, value in dotenv_values(REPO/'.env').items():
        if value and key.startswith('WANDB_'):
            os.environ.setdefault(key, value)
    import wandb
    tracking = wandb.init(entity=os.environ.get('WANDB_ENTITY','zixianma'),
        project=os.environ.get('WANDB_PROJECT','openwebrl'), id=f'browser-diagnostic-{job}',
        resume='never', name=f'Browser reset diagnostic {job}', dir=str(root),
        config={'urls':URLS,'no_policy_or_judge':True,'source':str(SOURCE)})
    results = []
    for concurrency, delay, count in trial_specs():
        if budget['deadline']-time.time() < 180:
            break
        # Keep the first three levels diagnostic even when failures occur.
        # Only escalate to 96 after 64 has >=90% success and no skipped tasks.
        if concurrency == 96 and not any(x['concurrency_limit']==64 and x['success_rate'] is not None
                                        and x['success_rate'] >= 0.9 and not x['skipped'] for x in results):
            break
        result = await run_case(root, concurrency, delay, count, budget['deadline'])
        results.append(result)
        save(root/'results.json', results)
        tracking.log({'benchmark/case':len(results), 'browser/concurrency_limit':concurrency,
            'browser/launch_interval_seconds':delay, 'browser/reset_success_rate':result['success_rate'],
            'browser/attempts':result['attempts'], 'browser/skipped':result['skipped'],
            'browser/suspected_challenges':result['suspected_challenges'],
            'browser/peak_active_workers':result['peak_active_workers'], 'browser/wall_seconds':result['wall_seconds']})
    tracking.finish()
    confirmation = {'launched': False, 'reason': 'Not requested'}
    chosen = confirmation_candidate(results)
    if confirm_rl:
        confirmation['reason'] = 'No reliable burst-launch candidate or fewer than 30 minutes remain'
        if chosen and budget['deadline'] - time.time() >= 1800:
            out = root/'rl-confirmation'; out.mkdir()
            plan = case_plan(job, out, f'tp4-b{chosen}-confirm', 4, chosen, False, fresh_sft=True)
            plan['case_timeout_seconds'] = int(budget['deadline'] - time.time()) - 120
            plan_path = out/'plan.json'; save(plan_path, plan)
            confirmation = {'launched': True, 'browsers':chosen, 'plan':str(plan_path)}
            save(root/'rl_confirmation.json', confirmation)
            command = ['bash','-c',f'source {SOURCE}/scripts/h200_env.sh\nexec python {REPO}/scripts/scaling_benchmark.py --worker {plan_path}']
            with (out/'worker.log').open('w') as log:
                result = subprocess.run(command, env=clean_environment(), stdout=log, stderr=subprocess.STDOUT)
            confirmation['returncode'] = result.returncode
    save(root/'summary.json', {'results': results, 'rl_confirmation':confirmation,
                              'main_training_modified': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--controller')
    mode.add_argument('--browser-worker', type=Path)
    mode.add_argument('--dry-run', action='store_true')
    p.add_argument('--result', type=Path)
    p.add_argument('--confirm-rl', action='store_true', help='Within the same approved allocation, confirm a reliable limit with one fresh SFT iteration if >=30 minutes remain.')
    args = p.parse_args()
    if args.dry_run:
        print(json.dumps({'urls': URLS, 'cases': trial_specs(), 'policy_model': False, 'judge_calls': False}, indent=2))
    elif args.browser_worker:
        if args.result is None:
            p.error('--browser-worker requires --result')
        asyncio.run(browser_worker(json.loads(args.browser_worker.read_text()), args.result))
    else:
        asyncio.run(controller(args.controller, confirm_rl=args.confirm_rl))
