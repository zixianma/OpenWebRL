#!/usr/bin/env python3
"""Own action-only SelectionARM services, full-300 evaluation, and reporting."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from train_arm_joint_sft import REPO, RUNTIME, ACTOR, digest, write_json
from resume_arm_c2_training import allocation_deadline, parse_job_record

ROOT = RUNTIME / 'runs/selection-actions-only-20260911'
CONTROL = RUNTIME / 'runs/dedicated-282209-20260908T005547Z/full'
PYTHON = RUNTIME / 'venv/bin/python'
MODEL = Path('/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89')


def report():
    from summarize_arm_reproduction import paired_report
    from summarize_arm_c2_full300 import mcnemar_exact
    output = ROOT / 'actions_only'
    manifest = json.loads((output / 'manifest.json').read_text())
    ids = manifest['task_ids']
    current = {r['task_id']: r for p in (output / 'results').glob('*.json')
               for r in [json.loads(p.read_text())]}
    if set(current) != set(ids) or len(ids) != 300:
        raise ValueError('Incomplete full-300 evaluation')
    result = dict(summary=json.loads((output / 'summary.json').read_text()), paired={},
                  limitation='Historical endpoint controls: live sites and implementation dates differ. Shadow agreement is not correctness.')
    for mode in ['baseline', 'selection']:
        control_manifest = json.loads((CONTROL / mode / 'manifest.json').read_text())
        for key in ['actor', 'task_file_sha256', 'task_ids', 'seed', 'sampling', 'max_steps', 'judge', 'judge_protocol']:
            if manifest[key] != control_manifest[key]:
                raise ValueError(f'Historical protocol mismatch: {key}')
        rows = {r['task_id']: r for p in (CONTROL / mode / 'results').glob('*.json')
                for r in [json.loads(p.read_text())]}
        success = lambda r: bool(r.get('valid') and r.get('reward') == 1)
        wins = sum(success(current[t]) and not success(rows[t]) for t in ids)
        losses = sum(success(rows[t]) and not success(current[t]) for t in ids)
        result['paired'][mode] = dict(all300_wins=wins, all300_losses=losses,
            all300_exact_p=mcnemar_exact(wins, losses), common_valid=paired_report(rows, current))
    traces = [json.loads(line) for p in (output / 'selections').glob('*.jsonl') for line in p.read_text().splitlines()]
    # One entry per visited state in this attempt; no silent selector fallback allowed.
    if any(r.get('fallback') for r in traces):
        result['selector_fallbacks'] = sum(bool(r.get('fallback')) for r in traces)
    shadows = [r for r in traces if 'shadow_full' in r]
    valid = [r for r in shadows if 'selected_index' in r['shadow_full'] and not r.get('fallback')]
    result['shadow'] = dict(attempted=len(shadows), valid=len(valid),
        same_index=sum(r['shadow_full']['same_index'] for r in valid),
        same_action=sum(r['shadow_full']['same_action'] for r in valid),
        note='Deterministic 10% of states on compact-selector trajectories; full selector is a reference, not ground truth.')
    if valid:
        result['shadow']['mean_compact_input_tokens'] = sum(r['selector_telemetry']['input_tokens'] for r in valid) / len(valid)
        result['shadow']['mean_full_input_tokens'] = sum(r['shadow_full']['input_tokens'] for r in valid) / len(valid)
    write_json(ROOT / 'comparison.json', result)
    write_json(REPO / 'openwebrl/docs/arm_results/selection-actions-only.json', result)
    # Update a dedicated block in the existing topic document.
    path = REPO / 'openwebrl/docs/ARM_INFERENCE.md'
    start, end = '<!-- compact-selector-result:start -->', '<!-- compact-selector-result:end -->'
    summary = result['summary']
    section = (f'{start}\nAction-only SelectionARM: {summary["successes"]}/300 '
        f'({summary["success_rate_all_scheduled"]:.1%}) overall; '
        f'{summary["successes"]}/{summary["valid"]} ({summary["success_rate_valid"]:.1%}) valid-only. '
        f'Unavailable: {summary["unavailable"]}. '
        '[Paired and shadow report](arm_results/selection-actions-only.json).\n' + end)
    import fcntl
    with (path.parent / '.documentation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = path.read_text()
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError('Missing compact result section')
        text = text[:text.index(start)] + section + text[text.index(end)+len(end):]
        tmp = path.with_suffix('.md.tmp'); tmp.write_text(text); tmp.replace(path)
    return result


def main():
    job = os.environ.get('SLURM_JOB_ID')
    if not job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Requires approved allocation')
    record = parse_job_record(subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True))
    deadline = allocation_deadline(record, margin_minutes=3).timestamp()
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).splitlines()
    if len(devices) != 1 or 'H200' not in devices[0]:
        raise ValueError('Expected one H200')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('Allocated GPU occupied')
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / 'source-manifest.json', {name:digest(REPO/name) for name in
        ['openwebrl/arm_inference.py', 'openwebrl/arm_eval.py', 'scripts/serve_arm.py',
         'openwebrl/generate_browser.py', 'openwebrl/run_evaluate.py', 'openwebrl/eval/reward_online_mind2web.py']})
    processes, logs = [], []
    def state(phase, **kw):
        write_json(ROOT/'status.json', dict(job=job, phase=phase, time=time.time(), **kw))
    def spawn(command, name):
        log=(ROOT/name).open('a'); logs.append(log)
        child=subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(child); return child
    def check():
        if time.time() >= deadline:
            raise TimeoutError('Allocation deadline; completed task files retained')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def healthy(url, child):
        until=time.time()+600
        while True:
            check()
            if child.poll() is not None or time.time()>until:
                raise RuntimeError('Service startup failed')
            try:
                with opener.open(url, timeout=5) as response:
                    if response.status == 200: return
            except OSError: pass
            time.sleep(2)
    def stop(*_): raise InterruptedError('Allocation signal')
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    try:
        state('loading_actor')
        actor=spawn([str(RUNTIME.parent/'venv/bin/python'), '-m', 'sglang.launch_server',
            '--model-path',str(ACTOR),'--host','127.0.0.1','--port','27100','--dtype','bfloat16',
            '--tp','1','--mem-fraction-static','0.4','--context-length','32768',
            '--max-running-requests','24','--chunked-prefill-size','4096','--disable-cuda-graph'], 'actor.log')
        healthy('http://127.0.0.1:27100/health_generate',actor)
        selector=spawn([str(PYTHON),'scripts/serve_arm.py','--mode','selection','--model',str(MODEL),
                        '--base',str(ACTOR),'--port','27101'], 'selector.log')
        healthy('http://127.0.0.1:27101/health',selector)
        state('evaluating_all300')
        evaluator=spawn([str(PYTHON),'-m','openwebrl.arm_eval','--mode','selection',
            '--actor-port','27100','--selector-endpoint','http://127.0.0.1:27101',
            '--browser-port-start','27200','--browser-port-end','27499',
            '--candidate-representation','actions_only','--shadow-modulus','10',
            '--parallel','16','--output',str(ROOT/'actions_only')], 'evaluation.log')
        while evaluator.poll() is None:
            check()
            if actor.poll() is not None or selector.poll() is not None:
                raise RuntimeError('Inference service died')
            time.sleep(10)
        if evaluator.returncode: raise RuntimeError('Evaluator failed')
        result=report(); state('complete',summary=result['summary'])
    except BaseException as exc:
        state('failed_or_paused',error=str(exc)); raise
    finally:
        for child in reversed(processes):
            if child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
        for child in reversed(processes):
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL); child.wait()
        for log in logs: log.close()


if __name__ == '__main__': main()
