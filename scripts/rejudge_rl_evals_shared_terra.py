#!/usr/bin/env python3
"""Re-grade saved full-300 RL eval rollouts with the shared Terra protocol (2026-11-04).

The protocol is the one used for the inference-scaling experiments: the frozen OM2W/AgentTrek
terminal-success prompt, gpt-5.6-terra, medium effort, 4,096 completion tokens, seed 42, one final
screenshot (openwebrl/api_experiments/browsercode/judges/terra.py, loaded through reproduce/rejudge.py's
hash-checked science()). No actor calls, browsers or GPUs.

Per episode: task = metadata.intent; actions = the per-step action lists from the saved action-history
text (the last includes the done/answer call); screenshot = the final judge screenshot as saved (no
resize). Validity accounting is unchanged from the RL eval: originally invalid episodes stay invalid, and
valid episodes the native judge never ran on (status FAILED/TRUNCATED) stay valid failures, unjudged.

Usage: --plan (default) writes the episode manifest; --execute grades with explicit USD/call caps.
Rows with a result are skipped on rerun; each run uses a fresh budget directory.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'openwebrl/api_experiments/reproduce'))
import rejudge  # noqa: E402

STEP = re.compile(r'(?m)^Step \d+: action=\[')


def actions_from(text):
    out = []
    for chunk in STEP.split(text)[1:]:
        action, sep, _ = chunk.rpartition('] environment feedback=')
        out.append(action if sep else chunk.strip())
    return out


def episodes(eval_dir, label, shots):
    import torch
    rows = []
    for meta in sorted(Path(eval_dir, 'rollouts').glob('*.json')):
        m = json.loads(meta.read_text())
        ident = f'{label}-{hashlib.sha256(m["task_id"].encode()).hexdigest()[:16]}'
        row = dict(id=ident, label=label, task_id=m['task_id'], original_valid=m['metrics']['valid_trajectories'] == 1,
                   native_judge=m['reward_metadata'].get('judge'))
        text = m['reward_metadata'].get('judge_text') or ''
        if not row['original_valid']:
            row.update(eligible=False, exclusion_reason='original invalid')
        elif text.startswith('Judge not run'):
            row.update(eligible=False, exclusion_reason=text[:80], reward=0)
        else:
            turn = torch.load(meta.with_suffix('.pt'), weights_only=False)['turns'][-1]
            md = turn['metadata']
            png = base64.b64decode(turn['multimodal_inputs']['judge_images'][-1].split(',', 1)[1])
            path = shots/f'{ident}.png'
            path.write_bytes(png)
            row.update(eligible=True, task=md['intent'], actions=actions_from(md['trajectory_action_history_text']),
                       screenshot=str(path), screenshot_sha256=hashlib.sha256(png).hexdigest())
            if not row['actions']:
                row.update(eligible=False, exclusion_reason='no parsable actions')
        rows.append(row)
    return rows


def plan(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    shots = out/'screenshots'
    shots.mkdir(exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    specs = [spec.split('=', 1) for spec in args.eval]
    workers = max(1, min(len(specs), int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    with ProcessPoolExecutor(max_workers=workers) as pool:  # one CPU per eval; sized from the allocation
        rows = [r for part in pool.map(episodes, [d for _, d in specs], [l for l, _ in specs], [shots]*len(specs)) for r in part]
    judge, _, sources = rejudge.science()
    for r in rows:
        if r['eligible']:
            r['reserved_usd'] = str(judge.judge_reservation(judge.build_judge_request(r['task'], r['actions'], r['screenshot']))[0])
    manifest = dict(schema='rl-eval-shared-terra-v1', judge_protocol=judge.protocol(), frozen_sources=sources,
                    evals=args.eval, episodes=rows, eligible=sum(r['eligible'] for r in rows),
                    sum_reserved_usd=str(sum(Decimal(r['reserved_usd']) for r in rows if r['eligible'])))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'episodes'}, indent=1))


def execute(args):
    from dotenv import dotenv_values
    from openai import OpenAI
    out = args.output
    manifest = json.loads((out/'manifest.json').read_text())
    judge, budget_module, _ = rejudge.science()
    if judge.protocol() != manifest['judge_protocol']:
        raise ValueError('Judge protocol changed since planning')
    key = dotenv_values(REPO/'.env').get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
    rejudge.need(bool(key), 'OPENAI_API_KEY required')
    budget = budget_module.ReservationBudget(out/f'budget/run-{int(time.time())}', rejudge.positive_money(args.judge_usd), args.judge_calls)
    todo = [r for r in manifest['episodes'] if r['eligible'] and not (out/'episodes'/r['id']/'result.json').is_file()][:args.judge_calls]
    print(f'{len(todo)} episodes to grade', flush=True)

    def one(r):
        d = out/'episodes'/r['id']
        d.mkdir(parents=True, exist_ok=True)
        if hashlib.sha256(Path(r['screenshot']).read_bytes()).hexdigest() != r['screenshot_sha256']:
            raise ValueError('Saved PNG changed')
        client = OpenAI(api_key=key, max_retries=0, timeout=180)
        try:
            value = dict(id=r['id'], reward=judge.judge(client, budget, r['task'], r['actions'], r['screenshot'], d))
        except BaseException as exc:
            (d/'error.json').write_text(json.dumps(dict(error_type=type(exc).__name__, error=str(exc)[:500], epoch=time.time())))
            raise
        finally:
            client.close()
        (d/'result.json').write_text(json.dumps(value))

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one, r) for r in todo]
        errors = sum(1 for f in futures if f.exception() is not None)
    summarize(out)
    return 1 if errors else 0


def summarize(out):
    manifest = json.loads((out/'manifest.json').read_text())
    table = {}
    for r in manifest['episodes']:
        t = table.setdefault(r['label'], dict(n=0, valid=0, success=0, graded=0, pending=0))
        t['n'] += 1
        t['valid'] += r['original_valid']
        res = out/'episodes'/r['id']/'result.json'
        if r['eligible']:
            if res.is_file():
                t['graded'] += 1
                t['success'] += json.loads(res.read_text())['reward']
            else:
                t['pending'] += 1
    for t in table.values():
        t['overall'] = round(100*t['success']/t['n'], 2)
        t['valid_only'] = round(100*t['success']/max(1, t['valid']), 2)
    (out/'summary.json').write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--eval', action='append', default=[], help='label=eval_dir (repeatable)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--judge-usd')
    p.add_argument('--judge-calls', type=int)
    p.add_argument('--concurrency', type=int, default=8)
    a = p.parse_args()
    if not 1 <= a.concurrency <= 8:
        raise SystemExit('Concurrency must be 1-8')
    if a.summarize:
        summarize(a.output)
    elif a.execute:
        sys.exit(execute(a))
    else:
        plan(a)
