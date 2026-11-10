#!/usr/bin/env python3
"""Re-judge committed Online-Mind2Web crossbench records with a new judge model by replaying the saved judge requests.

o4-mini was retired on 2026-10-23. Each committed record's verdict came from the last received judge request of its
committed physical attempt (judge-budget/requests/*.json keeps the full request: AgentTrek prompt, final screenshot,
seed 42). This replays that exact request with only `model` changed and parses the reply with the native parser.
  * records whose episode never reached the judge (status not completed) keep reward 0 and need no call;
  * invalid records stay invalid (counted separately);
  * a new judge reply without a verdict, or exhausted retries, makes the record invalid under the new judge.
Checks: the original reply re-parsed must equal the record's reward on every judged record, or the run is refused.

Outputs per run under --output/<label>/: verdicts/<key>.json (resumable) and summary.json (old/new success over all
records and valid-only, agreement and 2x2 confusion on records valid under both). A USD ledger caps spend
(--max-usd, priced at --input-price/--output-price per M tokens); at most --concurrency requests in flight.

  rejudge_crossbench_om2w.py --output OUT --max-usd 100 LABEL=BENCH_ROOT [LABEL=BENCH_ROOT ...]
  (BENCH_ROOT holds plan.json, collection/records and judge-budget/requests)
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=1))
    tmp.replace(path)


def committed_requests(root):
    """key -> the request receipt whose reply set each committed record's verdict (None if the judge never ran)."""
    by_key = {}
    for f in sorted((root / 'judge-budget' / 'requests').glob('*.json')):
        r = json.loads(f.read_text())
        if r.get('status') == 'received':
            by_key.setdefault(r['attempt'], []).append(r)
    out = {}
    for f in sorted((root / 'collection' / 'records').glob('*.json')):
        rec = json.loads(f.read_text())
        start = rec['started_epoch']; end = start + rec['elapsed_seconds'] + 1
        inside = [r for r in by_key.get(rec['key'], []) if start - 1 <= r['started_epoch'] <= end]
        out[rec['key']] = (rec, max(inside, key=lambda r: r['started_epoch']) if inside else None)
    return out


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+'); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--model', default='gpt-5.6-terra')
    ap.add_argument('--input-price', type=float, default=2.5); ap.add_argument('--output-price', type=float, default=15.)
    ap.add_argument('--max-usd', type=float, required=True); ap.add_argument('--concurrency', type=int, default=8)
    ap.add_argument('--timeout', type=float, default=180.); ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--env-file', type=Path, default=Path(__file__).resolve().parents[1] / '.env')
    args = ap.parse_args()
    from openwebrl.eval.reward_online_mind2web import _parse_verdict
    runs = []
    for spec in args.runs:
        label, root = spec.split('=', 1)
        root = Path(root)
        plan = json.loads((root / 'plan.json').read_text())
        if plan['benchmark'] != 'online_mind2web':
            raise SystemExit(f'{label}: not an Online-Mind2Web run')
        pairs = committed_requests(root)
        for key, (rec, req) in pairs.items():
            if req is None:
                if rec['valid'] and rec['reward'] not in (0, 0.0):
                    raise SystemExit(f'{label}/{key}: reward {rec["reward"]} without a judge request')
                continue
            old = _parse_verdict(req['response']['choices'][0]['message']['content'] or '')
            if rec['valid'] and old != rec['reward']:
                raise SystemExit(f'{label}/{key}: saved reply parses to {old}, record says {rec["reward"]}')
        runs.append((label, pairs))
        calls = sum(1 for rec, req in pairs.values() if req is not None)
        print(f'{label}: {len(pairs)} records, {calls} judge requests to replay', flush=True)
    if args.dry_run:
        return

    from dotenv import dotenv_values
    for k, v in dotenv_values(args.env_file).items():
        if v and k.startswith('OPENAI_'):
            os.environ.setdefault(k, v)
    from openai import AsyncOpenAI
    client = AsyncOpenAI(max_retries=0)
    ledger_path = args.output / 'ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(model=args.model, spent_usd=0., calls=0)
    lock = asyncio.Lock(); sem = asyncio.Semaphore(args.concurrency)
    upper = (40000 * args.input_price + 4096 * args.output_price) / 1e6        # per-request reservation

    async def replay(label, key, req):
        path = args.output / label / 'verdicts' / f'{key}.json'
        if path.exists():
            return json.loads(path.read_text())
        async with sem:
            async with lock:
                if ledger['spent_usd'] + upper > args.max_usd:
                    raise RuntimeError(f'USD cap {args.max_usd} reached; rerun resumes')
                ledger['spent_usd'] += upper
            request = dict(req['request'], model=args.model)
            text, usage, error, returned = None, None, None, None
            for attempt in range(4):
                try:
                    r = await asyncio.wait_for(client.chat.completions.create(**request), args.timeout)
                    text, usage, returned = r.choices[0].message.content or '', r.usage.model_dump(), r.model
                    break
                except Exception as exc:                                    # noqa: BLE001 - recorded, retried
                    error = f'{type(exc).__name__}: {str(exc)[:300]}'
                    await asyncio.sleep(3 ** (attempt + 1))
            cost = (usage['prompt_tokens'] * args.input_price + usage['completion_tokens'] * args.output_price) / 1e6 if usage else 0.
            async with lock:
                ledger['spent_usd'] += cost - upper; ledger['calls'] += 1
                write_json(ledger_path, ledger)
            if text is None and error and ('model_not_found' in error or 'invalid_request_error' in error):
                raise RuntimeError(f'{label}/{key}: {error}')
            out = dict(key=key, model=args.model, returned_model=returned, verdict=_parse_verdict(text) if text is not None else None,
                       text=text, usage=usage, cost_usd=cost, error=None if text is not None else error, epoch=time.time())
            write_json(path, out)
            return out

    for label, pairs in runs:
        jobs = {key: replay(label, key, req) for key, (rec, req) in pairs.items() if req is not None}
        done = dict(zip(jobs, await asyncio.gather(*jobs.values())))
        rows = []
        for key, (rec, req) in pairs.items():
            old = rec['reward'] if rec['valid'] else None
            if not rec['valid']:
                new = None
            elif req is None:
                new = old
            else:
                new = done[key]['verdict']
            rows.append((old, new))
        n = len(rows)
        both = [(o, w) for o, w in rows if o is not None and w is not None]
        summary = dict(label=label, records=n, replayed=len(jobs), model=args.model,
                       old=dict(valid=sum(o is not None for o, _ in rows), success=sum(o == 1 for o, _ in rows)),
                       new=dict(valid=sum(w is not None for _, w in rows), success=sum(w == 1 for _, w in rows)),
                       both_valid=len(both), agree=sum(o == w for o, w in both),
                       confusion={'old1_new1': sum(o == 1 and w == 1 for o, w in both), 'old1_new0': sum(o == 1 and w == 0 for o, w in both),
                                  'old0_new1': sum(o == 0 and w == 1 for o, w in both), 'old0_new0': sum(o == 0 and w == 0 for o, w in both)})
        for side in ('old', 'new'):
            s = summary[side]
            s.update(overall=round(100 * s['success'] / n, 2), valid_only=round(100 * s['success'] / max(s['valid'], 1), 2))
        write_json(args.output / label / 'summary.json', summary)
        print(json.dumps(summary), flush=True)
    print('spent_usd', round(ledger['spent_usd'], 3), 'calls', ledger['calls'], flush=True)


if __name__ == '__main__':
    asyncio.run(main())
