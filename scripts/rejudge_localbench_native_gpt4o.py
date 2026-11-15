#!/usr/bin/env python3
"""Offline official-judge grading of judge-deferred local WebVoyager/DeepShop rollouts (2026-11-15).

Runs the frozen harness's own benchmark judges (openwebrl.eval.reward_webvoyager / reward_deepshop in
reference-localbench-deferjudge-20261115, GPT-4o, seed 42, the same eval-args the in-harness path sets) on the
saved turn records, so the verdicts are what the harness would have produced had judging not been deferred:
non-COMPLETED episodes score 0 without a call, a missing SUCCESS/NOT SUCCESS verdict makes the episode invalid
(as the harness's judge_invalid path does), and originally aborted episodes stay invalid.

The OpenAI client is wrapped to meter usage at GPT-4o list prices and to refuse calls past --usd/--calls.
One JSON per episode under <output>/episodes/<label>/; finished episodes are skipped on rerun.
"""
import argparse
import asyncio
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HARNESS = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/reference-localbench-deferjudge-20261115')
REPO = Path('/gpfs/projects/krishna/zixianma/OpenWebRL')
PRICE_IN, PRICE_OUT = Decimal('2.50') / 10**6, Decimal('10.00') / 10**6  # gpt-4o, USD per token


class Meter:
    def __init__(self, usd, calls):
        self.cap_usd, self.cap_calls = Decimal(usd), calls
        self.usd, self.calls, self.prompt, self.completion = Decimal(0), 0, 0, 0

    def check(self):
        if self.calls >= self.cap_calls or self.usd >= self.cap_usd:
            raise RuntimeError('judge cap reached')


def metered_client(meter, real):
    """Exposes the two calls the frozen judges make: chat.completions.create (WebVoyager) and
    beta.chat.completions.parse (DeepShop structured verdict)."""
    async def metered(call, **kw):
        meter.check()
        meter.calls += 1
        resp = await call(**kw)
        u = resp.usage
        meter.prompt += u.prompt_tokens
        meter.completion += u.completion_tokens
        meter.usd += u.prompt_tokens * PRICE_IN + u.completion_tokens * PRICE_OUT
        return resp

    class Completions:
        async def create(self, **kw):
            return await metered(real.chat.completions.create, **kw)

        async def parse(self, **kw):
            return await metered(real.beta.chat.completions.parse, **kw)
    completions = Completions()
    return SimpleNamespace(chat=SimpleNamespace(completions=completions),
                           beta=SimpleNamespace(chat=SimpleNamespace(completions=completions)))


async def grade(args):
    sys.path.insert(0, str(HARNESS))
    os.environ.setdefault('JUDGE_API_BASE', 'https://api.openai.com/v1')
    from dotenv import dotenv_values
    os.environ['OPENAI_API_KEY'] = dotenv_values(REPO/'.env')['OPENAI_API_KEY']
    import torch
    from openai import AsyncOpenAI
    from slime.utils.types import Sample
    import openwebrl.eval.reward_webvoyager as wv
    import openwebrl.eval.reward_deepshop as ds
    meter = Meter(args.usd, args.calls)
    client = metered_client(meter, AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=os.environ['JUDGE_API_BASE']))
    for mod in (wv, ds):
        mod._get_openai_client = lambda method=None: client
    modules = dict(webvoyager=wv, deepshop=ds)
    eval_args = lambda bench: SimpleNamespace(  # configure() in eval_local_benchmarks.py
        judge_api_model='gpt-4o', judge_prompt_variant=bench, judge_max_attached_imgs=30, judge_timeout_secs=120.0,
        judge_api_mode='served', hf_checkpoint='/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT',
        log_judge_output=False)
    out = Path(args.output)
    sem = asyncio.Semaphore(args.concurrency)
    stop = asyncio.Event()

    async def one(label, bench, meta):
        target = out/'episodes'/label/f'{meta.stem}.json'
        if target.is_file():
            return
        m = json.loads(meta.read_text())
        if not (m['reward_metadata'].get('judge_text') or '').startswith('DEFERRED'):
            raise ValueError(f'{meta}: not judge-deferred')
        rec = dict(label=label, benchmark=bench, task_id=m['task_id'], original_valid=m['metrics']['valid_trajectories'] == 1,
                   terminal_status=m.get('terminal_status'))
        if not rec['original_valid']:
            rec.update(valid=False, reward=0, judge_text='original invalid', judge_called=False)
        else:
            async with sem:
                if stop.is_set():
                    return
                turns = [Sample.from_dict(t) for t in torch.load(meta.with_suffix('.pt'), weights_only=False)['turns']]
                for t in turns:
                    t.reward, t.remove_sample = None, False
                    t.metadata.pop('reward', None)
                before = meter.calls
                try:
                    rewards = await modules[bench].reward_func(eval_args(bench), turns)
                except RuntimeError:
                    stop.set()
                    return
                last = next(t for t in turns if t.metadata.get('is_last_turn'))
                r = rewards[0]
                rec.update(judge_called=meter.calls > before, judge_text=last.metadata['reward'].get('judge_text'),
                           judge_timeout=last.metadata['reward'].get('judge_timeout'))
                if r is None:
                    if 'cap reached' in (rec['judge_text'] or '') or meter.calls >= meter.cap_calls or meter.usd >= meter.cap_usd:
                        stop.set()
                        return
                    rec.update(valid=False, reward=0, judge_invalid=True)
                else:
                    rec.update(valid=True, reward=int(r))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(rec))

    jobs = []
    for spec in args.eval:
        label, d = spec.split('=', 1)
        bench = 'webvoyager' if '-webvoyager-' in label else 'deepshop'
        jobs += [one(label, bench, meta) for meta in sorted(Path(d, 'rollouts').glob('*.json'))]

    async def report():
        while True:
            await asyncio.sleep(120)
            print(json.dumps(dict(calls=meter.calls, usd=str(round(meter.usd, 4)))), flush=True)
    reporter = asyncio.create_task(report())
    await asyncio.gather(*jobs)
    reporter.cancel()
    ledger = dict(model='gpt-4o', calls=meter.calls, prompt_tokens=meter.prompt, completion_tokens=meter.completion,
                  usd=str(meter.usd), cap_usd=str(meter.cap_usd), cap_calls=meter.cap_calls, stopped_at_cap=stop.is_set())
    runs = sorted((out/'budget').glob('run-*.json')) if (out/'budget').exists() else []
    (out/'budget').mkdir(parents=True, exist_ok=True)
    (out/'budget'/f'run-{len(runs)+1}.json').write_text(json.dumps(ledger, indent=1))
    print(json.dumps(ledger))
    return 3 if stop.is_set() else 0


def summarize(out):
    import collections
    t = collections.defaultdict(lambda: dict(n=0, valid=0, success=0, judged=0))
    for f in Path(out, 'episodes').glob('*/*.json'):
        r = json.loads(f.read_text())
        x = t[r['label']]
        x['n'] += 1; x['valid'] += r['valid']; x['success'] += r['reward']; x['judged'] += bool(r.get('judge_called'))
    for x in t.values():
        x['overall'] = round(100*x['success']/x['n'], 2)
        x['valid_only'] = round(100*x['success']/max(1, x['valid']), 2)
    Path(out, 'summary.json').write_text(json.dumps(dict(sorted(t.items())), indent=1))
    print(json.dumps(dict(sorted(t.items())), indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--eval', action='append', default=[], help='label=eval_dir (repeatable)')
    p.add_argument('--output', required=True)
    p.add_argument('--usd', default='60')
    p.add_argument('--calls', type=int, default=9000)
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--summarize', action='store_true')
    a = p.parse_args()
    if a.summarize:
        summarize(a.output)
    else:
        sys.exit(asyncio.run(grade(a)))
