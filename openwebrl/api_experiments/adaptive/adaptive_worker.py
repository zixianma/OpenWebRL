"""One worker slot of one shard: each assigned task runs every arm back to back.

Arms run in a per-task seeded random order so all five policies meet the same
live sites within minutes of each other. A valid outcome is final; an invalid
or interrupted (task, arm) pair gets exactly one prescribed retry in a later
attempt. The frozen Luna episode function, judge and receipts are reused as is.
"""
from bootstrap import ROOT, CONFIG
import argparse
import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path

import worker as frozen_worker
from adaptive_model import AdaptiveLunaModel
from request_budget import ReservationBudget, SharedHTTPGate
from openwebrl.luna_qwen_policy import Budget
from openwebrl.luna_qwen_metrics import write_json

MAX_ATTEMPTS = 2
CONSECUTIVE_INVALID_STOP = 6


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_sources():
    manifest = read(ROOT / 'source-manifest.json')
    assert manifest['plan_sha256'] == sha(ROOT / 'campaign-plan.json'), 'Plan changed'
    for path, expected in manifest['source_hashes'].items():
        assert sha(path) == expected, 'Source changed: ' + path
    assert sha(ROOT / 'tasks.json') == read(ROOT / 'campaign-plan.json')['task_manifest_sha256'], 'Tasks changed'
    return manifest


def shard_root(shard):
    return ROOT / 'shards' / shard


def arm_order(task_id, arms):
    order = list(arms)
    random.Random(int(hashlib.sha256(task_id.encode()).hexdigest(), 16)).shuffle(order)
    return order


def prior_rows(shard, task_id, arm):
    """Every recorded outcome of this pair; an episode directory without result.json is an interrupted attempt."""
    rows = []
    for path in sorted(shard_root(shard).glob(f'attempts/*/episodes/{task_id}-browsercode-{arm}')):
        result = path / 'result.json'
        rows.append(read(result) if result.is_file() else {'valid': False, 'interrupted': True})
    return rows


def caps_for(plan, shard, arm):
    caps = plan['budget']
    if shard == 'smoke':
        return caps['smoke_actor_usd_per_arm'], caps['smoke_judge_usd']
    return caps['actor_usd_per_shard'][arm], caps['judge_usd_per_shard']


def budgets_for(shard, plan, arms):
    caps = plan['budget']
    root = shard_root(shard) / 'budget'
    judge = ReservationBudget(root / 'judge', caps_for(plan, shard, arms[0])[1], caps['judge_calls_per_shard'])
    browser = Budget(root / 'browser', caps['browser_starts_per_shard'], caps['browser_starts_per_shard'])
    return {arm: {'actor': ReservationBudget(root / arm / 'model', caps_for(plan, shard, arm)[0], caps['actor_calls_per_shard']),
                  'judge': judge, 'browser': browser} for arm in arms}


def global_spend(plan):
    """Charged or reserved USD across every ledger the approval covers (all phases, model and judge)."""
    guard = plan.get('global_spend_guard')
    if not guard:
        return None, None
    total = 0.0
    for pattern in guard['ledger_globs']:
        for ledger in Path('/').glob(pattern.lstrip('/')):
            total += float(read(ledger)['charged_or_reserved_usd'])
    return total, float(guard['stop_usd'])


def nearly_spent(budget, cap):
    snapshot = budget.snapshot()
    return float(snapshot['charged_or_reserved_usd']) >= 0.95 * float(cap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', required=True)
    parser.add_argument('--slot', type=int, required=True)
    parser.add_argument('--slots', type=int, required=True)
    parser.add_argument('--attempt-dir', required=True)
    a = parser.parse_args()
    validate_sources()
    plan = read(ROOT / 'campaign-plan.json')
    root = Path(a.attempt_dir).resolve()
    assert root.parent.parent == shard_root(a.shard).resolve(), 'Attempt outside its shard'
    ids = plan['shards'][a.shard]
    tasks = {x['task_id']: x for x in read(ROOT / 'tasks.json')}
    # The frozen gate allows at most 8 concurrent model requests; each shard owns one.
    gate = SharedHTTPGate(shard_root(a.shard) / 'http-gate/model', plan['http_gate_limit_per_shard'])

    class ScopedModel(AdaptiveLunaModel):
        def __init__(self, *positional, **kwargs):
            kwargs['http_gate'] = gate
            super().__init__(*positional, **kwargs)
    frozen_worker.PairedModel = ScopedModel
    arms = [x['arm'] for x in plan['arms']]
    budgets = budgets_for(a.shard, plan, arms)
    from dotenv import dotenv_values
    from openai import OpenAI
    keys = dotenv_values(plan['credential_file'])
    client = OpenAI(api_key=keys['OPENAI_API_KEY'], max_retries=0, timeout=180)
    judge_client = OpenAI(api_key=keys.get('JUDGE_API_KEY') or keys['OPENAI_API_KEY'], max_retries=0, timeout=180)
    tracking = None
    try:
        import wandb
        os.environ['WANDB_API_KEY'] = keys.get('WANDB_API_KEY', '')
        tracking = wandb.init(entity='zixianma', project='openwebrl-evals', id=f"{plan['campaign']}-{a.shard}-{root.name}-w{a.slot}",
                              dir=str(root), job_type='browsercode_adaptive_compute_evaluation',
                              config={'shard': a.shard, 'arms': arms, 'plan_sha256': sha(ROOT / 'campaign-plan.json')},
                              settings=wandb.Settings(disable_git=True), save_code=False)
    except Exception as exc:
        write_json(root / f'worker{a.slot}-tracking-error.json', {'error_type': type(exc).__name__, 'error': str(exc)})
    signal.signal(signal.SIGTERM, lambda *unused: (_ for _ in ()).throw(SystemExit(143)))
    consecutive_invalid = 0
    try:
        for i, tid in enumerate(ids):
            if i % a.slots != a.slot:
                continue
            for arm in arm_order(tid, arms):
                rows = prior_rows(a.shard, tid, arm)
                if any(r.get('valid') for r in rows) or len(rows) >= MAX_ATTEMPTS:
                    continue
                spent, stop = global_spend(plan)
                if spent is not None and spent >= stop:
                    write_json(root / f'worker{a.slot}-global-stop.json', {'epoch': time.time(), 'spent_usd': spent, 'stop_usd': stop})
                    raise RuntimeError('Global approved spend guard reached; no further episodes')
                if nearly_spent(budgets[arm]['actor'], caps_for(plan, a.shard, arm)[0]):
                    write_json(root / 'skipped' / f'{tid}-{arm}.json', {'reason': 'arm actor cap nearly spent', 'epoch': time.time()})
                    continue
                write_json(root / f'worker{a.slot}-heartbeat.json', {'epoch': time.time(), 'task_id': tid, 'arm': arm, 'stage': 'episode'})
                row = frozen_worker.episode(root, tasks[tid], {'harness': 'browsercode', 'policy': arm}, client, judge_client, budgets[arm], a.slot)
                consecutive_invalid = 0 if row['valid'] else consecutive_invalid + 1
                if tracking:
                    tracking.log({'arm': arm, 'valid': int(row['valid']), 'success': int(row.get('reward') == 1),
                                  'task_latency_seconds': row.get('task_latency_seconds'),
                                  'model_cost': row['usage_accounting'].get('model_excluding_judge', {}).get('charged_or_reserved_usd')})
                if any((b.root / 'halt.json').exists() for b in (budgets[arm]['judge'], budgets[arm]['browser'], budgets[arm]['actor'])):
                    raise RuntimeError('Shared pool halted; diagnose before further work')
                if consecutive_invalid >= CONSECUTIVE_INVALID_STOP:
                    raise RuntimeError(f'{CONSECUTIVE_INVALID_STOP} consecutive invalid episodes; diagnose before further work')
        write_json(root / f'worker{a.slot}-complete.json', {'epoch': time.time(), 'worker_complete': True})
    finally:
        if tracking:
            tracking.finish()
        client.close()
        judge_client.close()


if __name__ == '__main__':
    main()
