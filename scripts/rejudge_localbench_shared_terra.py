#!/usr/bin/env python3
"""Shared-Terra grading of judge-deferred local WebVoyager/DeepShop rollouts (copy of rejudge_rl_evals_shared_terra.py, 2026-11-15).

Only episodes() differs: final screenshot and actions come from the local-bench turn records.

Original description (2026-11-04):

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


TOOL_CALL = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.S)


def episodes(eval_dir, label, shots):
    """Local WebVoyager/DeepShop rollouts (judge deferred in-harness).

    The local-bench harness saves no separate post-action judge capture. A completed episode ends with the
    ``done`` call, which does not change the page, so the last turn's observed screenshot is the final page.
    Actions are the tool calls of every turn, in order; the last one carries the agent's answer. Validity
    follows the RL eval: aborted episodes stay invalid; valid episodes that ended FAILED/TRUNCATED (turn
    limit, length limit) were never judged by the native path and stay valid failures, unjudged.
    """
    import torch
    rows = []
    for meta in sorted(Path(eval_dir, 'rollouts').glob('*.json')):
        m = json.loads(meta.read_text())
        ident = f'{label}-{hashlib.sha256(m["task_id"].encode()).hexdigest()[:16]}'
        row = dict(id=ident, label=label, task_id=m['task_id'], original_valid=m['metrics']['valid_trajectories'] == 1,
                   terminal_status=m.get('terminal_status'))
        if not (m['reward_metadata'].get('judge_text') or '').startswith('DEFERRED'):
            raise ValueError(f'{meta}: not a judge-deferred record')
        if not row['original_valid']:
            row.update(eligible=False, exclusion_reason='original invalid')
        elif m.get('terminal_status') != 'completed':
            row.update(eligible=False, exclusion_reason=f'not judged: status={m.get("terminal_status")}', reward=0)
        else:
            turns = torch.load(meta.with_suffix('.pt'), weights_only=False)['turns']
            turns = sorted(turns, key=lambda t: (t['metadata'] or {}).get('turn_index', 0))
            last = turns[-1]
            png = base64.b64decode(last['multimodal_inputs']['images'][-1].split(',', 1)[1])
            path = shots/f'{ident}.png'
            path.write_bytes(png)
            actions = [c for t in turns for c in TOOL_CALL.findall(t.get('response') or '')[-1:]]
            row.update(eligible=True, task=last['metadata']['intent'], actions=actions,
                       terminate_reason=last['metadata'].get('terminate_reason'),
                       screenshot=str(path), screenshot_sha256=hashlib.sha256(png).hexdigest())
            if not actions:
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
    manifest = dict(schema='localbench-shared-terra-v1', judge_protocol=judge.protocol(), frozen_sources=sources,
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
    todo = [r for r in manifest['episodes'] if r['eligible'] and r['id'] not in args.skip
            and not (out/'episodes'/r['id']/'result.json').is_file()][:args.judge_calls]
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
    p.add_argument('--skip', action='append', default=[],
                   help='episode id to leave ungraded (counted as pending), e.g. one whose verdict the frozen parser rejects')
    a = p.parse_args()
    if not 1 <= a.concurrency <= 8:
        raise SystemExit('Concurrency must be 1-8')
    if a.summarize:
        summarize(a.output)
    elif a.execute:
        sys.exit(execute(a))
    else:
        plan(a)
