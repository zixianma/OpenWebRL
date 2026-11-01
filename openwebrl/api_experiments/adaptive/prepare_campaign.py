"""Prepare the adaptive-compute BrowserCode campaign in a new runtime directory.

Copies the validated Luna effort runtime unchanged (episode, judge, receipts,
browser bridge), adds the adaptive policy files, freezes the 1,045-task public
roster (OM2W 300, WebVoyager 595, DeepShop 150) into mixed-benchmark shards, and
writes per-shard, per-arm budget caps whose total stays inside the approval.
No browser, provider or Slurm work happens here.
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent
BASE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/serial-vs-parallel-reasoning-20261013/'
            'luna-browsercode-effort-single-xhigh-medium-low-n5-medium-low-om2w300-20261028-v1')
SKIP = {'controller.py', 'financial_plan.py', 'canary_scope.py', 'test_luna_effort.py', 'current.json',
        'campaign-plan.json', 'source-manifest.json', 'tasks.json'}
ADAPTIVE = ('action_signature.py', 'adaptive_model.py', 'adaptive_worker.py', 'run_shard.py')
PHASE_ARMS = {1: ('luna_single_high', 'luna_n5_high', 'adaptive_agree', 'adaptive_self', 'adaptive_self_agree'),
              2: ('luna_single_high', 'luna_n5_high', 'luna_n5_cached', 'adaptive_agree_cached', 'adaptive_agree3_cached')}
PHASE_ACTOR_CAP = {1: {'luna_single_high': '2.5', 'luna_n5_high': '15', 'adaptive_agree': '11', 'adaptive_self': '8',
                       'adaptive_self_agree': '12'},
                   2: {'luna_single_high': '2.5', 'luna_n5_high': '15', 'luna_n5_cached': '9', 'adaptive_agree_cached': '7',
                       'adaptive_agree3_cached': '5'}}
# Phase 2 also stops before all phases together reach this many charged or reserved USD.
GLOBAL_STOP_USD = '990'
PHASE1_ROOT = '/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/adaptive-compute-20261031-v2'
SHARDS = 16
SEED = 20261031
APPROVED_USD = Decimal('1000')
JUDGE_CAP = '9'
SMOKE_ACTOR_CAP = '1.5'
SMOKE_JUDGE_CAP = '1'
SMOKE_TASKS = ('online-mind2web', 'webvoyager', 'deepshop')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def roster():
    sys.path.insert(0, str(API / 'reproduce'))
    from tasks import load_tasks, canonical_benchmark
    rows = load_tasks('browsercode_luna_effort') + load_tasks('browsercode_gpt_high_wv_ds')
    out = [{'task_id': r['task_id'], 'intent': r['intent'], 'start_url': r['start_url'],
            'benchmark': canonical_benchmark(r['benchmark'])} for r in rows]
    assert dict(Counter(r['benchmark'] for r in out)) == {'online-mind2web': 300, 'webvoyager': 595, 'deepshop': 150}
    assert len({r['task_id'] for r in out}) == 1045
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--credential-file', default='/gpfs/projects/krishna/zixianma/OpenWebRL/.env')
    p.add_argument('--phase', type=int, choices=(1, 2), default=1)
    a = p.parse_args()
    ARMS, ACTOR_CAP = PHASE_ARMS[a.phase], PHASE_ACTOR_CAP[a.phase]
    out = a.output.resolve()
    assert str(out).startswith('/gpfs/scrubbed/zixianma/openwebrl-runtime/'), 'Browser bridge requires the runtime tree'
    out.mkdir(parents=True, exist_ok=False)
    copied = []
    for item in sorted(BASE.iterdir()):
        if item.is_symlink():
            (out / item.name).symlink_to(item.resolve())
        elif item.is_file() and item.name not in SKIP and item.suffix in ('.py', '.ts', '.json'):
            shutil.copy2(item, out / item.name)
            copied.append(item.name)
    for name in ADAPTIVE:
        shutil.copy2(HERE / name, out / name)
    tasks = roster()
    write(out / 'tasks.json', tasks)
    order = [r['task_id'] for r in tasks]
    random.Random(SEED).shuffle(order)
    shards = {f's{k:02d}': order[k::SHARDS] for k in range(SHARDS)}
    smoke = [next(r['task_id'] for r in tasks if r['benchmark'] == b) for b in SMOKE_TASKS]
    shards['smoke'] = smoke
    per_shard = sum(Decimal(v) for v in ACTOR_CAP.values()) + Decimal(JUDGE_CAP)
    smoke_total = Decimal(SMOKE_ACTOR_CAP) * len(ARMS) + Decimal(SMOKE_JUDGE_CAP)
    committed = per_shard * SHARDS + smoke_total
    assert committed <= APPROVED_USD, 'Budget caps exceed the approval'
    plan = {
        'schema': 'adaptive-compute-browsercode-v1', 'campaign': out.name,
        'arms': [{'arm': arm, 'model': 'gpt-6-luna', 'reasoning_effort': 'high', 'output_tokens': 12288,
                  'proposal_count': 1 if arm == 'luna_single_high' else 5,
                  'selector_reasoning_effort': None if arm == 'luna_single_high' else 'high'} for arm in ARMS],
        'protocol': {'max_decisions': 30, 'episode_deadline_seconds': 900, 'viewport': '1280x1800', 'harness': 'browsercode',
                     'judge': 'gpt-5.6-terra terminal success (frozen common_judge)', 'arm_order': 'per-task seeded shuffle',
                     'retry': 'one prescribed retry of an invalid or interrupted (task, arm) pair; valid outcomes final',
                     'provider_error': 'invalidates the episode only; unknown usage keeps its full reserve'},
        'task_manifest_sha256': sha(out / 'tasks.json'), 'shards': shards, 'shard_seed': SEED,
        'http_gate_limit_per_shard': 8, 'credential_file': a.credential_file,
        'budget': {'actor_usd_per_shard': ACTOR_CAP, 'judge_usd_per_shard': JUDGE_CAP, 'actor_calls_per_shard': 20000,
                   'judge_calls_per_shard': 2000, 'browser_starts_per_shard': 1200,
                   'smoke_actor_usd_per_arm': SMOKE_ACTOR_CAP, 'smoke_judge_usd': SMOKE_JUDGE_CAP,
                   'committed_cap_usd': str(committed), 'approved_usd': str(APPROVED_USD)},
        'base_runtime': str(BASE), 'base_files': copied, 'phase': a.phase}
    if a.phase == 2:
        plan['global_spend_guard'] = {'stop_usd': GLOBAL_STOP_USD, 'ledger_globs': [
            f'{root}/shards/*/budget/{kind}/ledger.json' for root in (PHASE1_ROOT, str(out)) for kind in ('*/model', 'judge')]}
        plan['design_note'] = ('Designed after an interim look at phase 1 (about 100 valid episodes per arm): the projected '
                               'selector had zero prompt-cache hits and cost about half of N5. Fresh contemporaneous '
                               'baselines rerun here; all 1,045 tasks rerun.')
    write(out / 'campaign-plan.json', plan)
    write(out / 'approval.json', {'approved_usd_total_api': str(APPROVED_USD), 'date': '2026-10-31',
                                  'scope': 'User: "ok i give you $1000 budget. come up with the best design" '
                                           '(adaptive inference-compute API actor study; model + judge API spend).',
                                  'compute': 'CPU-only Slurm jobs (no GPU), account vlm'})
    sources = {str(out / name): sha(out / name) for name in copied + list(ADAPTIVE) + ['approval.json']}
    write(out / 'source-manifest.json', {'plan_sha256': sha(out / 'campaign-plan.json'), 'source_hashes': sources})
    print(json.dumps({'output': str(out), 'committed_cap_usd': str(committed), 'shards': {k: len(v) for k, v in shards.items()}}))


if __name__ == '__main__':
    main()
