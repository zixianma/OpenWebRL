#!/usr/bin/env python3
"""Audit the GPT-4.1 action_history judge used for RL reward and local evaluation.

Approved 2026-10-24 with a $30 API cap. Re-judges saved held-out episodes with the exact
training judge prompt (reward_browser.compute_judge_reward_actionhistory: same action
history, final answer and last three screenshots), changing only the judge model. The
pipeline is first checked by re-judging episodes with GPT-4.1 itself. A budget-tracking
client reserves a conservative upper-bound cost before every request (retries included)
and stops at the cap.

Episodes: valid ones from the 56-group run at iterations 10 and 90 (full trajectories
saved), stratified by the original verdict. Writes per-episode verdicts (private) and an
aggregate summary.
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import threading
import types

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # openwebrl and slime packages
from analyze_rl_eval_failures import EVALS  # noqa: E402

PRICES = {'gpt-4.1': (2.0, 8.0), 'gpt-5.6-terra': (2.5, 15.0)}  # USD per M input/output tokens (upper bounds)
COHORTS = {10: 'outcome-only-56groups-20261006-iter10', 90: 'outcome-only-56groups-20261006-iter90'}


class Budget:
    def __init__(self, path, cap):
        self.path, self.cap, self.lock = Path(path), cap, threading.Lock()
        self.state = json.loads(self.path.read_text()) if self.path.exists() else dict(spent_usd=0., reserved_usd=0., calls=0)

    def _save(self):
        tmp = self.path.with_suffix('.tmp'); tmp.write_text(json.dumps(self.state)); tmp.replace(self.path)

    def reserve(self, usd):
        with self.lock:
            if self.state['spent_usd']+self.state['reserved_usd']+usd > self.cap:
                raise RuntimeError('API cap reached')
            self.state['reserved_usd'] += usd; self.state['calls'] += 1; self._save()

    def settle(self, reserved, actual):
        with self.lock:
            self.state['reserved_usd'] -= reserved; self.state['spent_usd'] += actual if actual is not None else reserved
            self._save()


def capped_client(inner, budget, model):
    pin, pout = PRICES[model]

    async def create(**kwargs):
        kwargs['model'] = model
        n_images = sum(1 for m in kwargs['messages'] if isinstance(m.get('content'), list)
                       for c in m['content'] if c.get('type') == 'image_url')
        chars = sum(len(json.dumps(m.get('content') if isinstance(m.get('content'), str) else
                                   [c for c in m['content'] if c.get('type') == 'text'])) for m in kwargs['messages'])
        upper = ((chars/2 + 3000*n_images)*pin + 8000*pout)/1e6  # generous token bounds incl. reasoning
        budget.reserve(upper)
        actual = None
        try:
            resp = await inner.chat.completions.create(**kwargs)
            u = getattr(resp, 'usage', None)
            if u is not None:
                actual = (u.prompt_tokens*pin + u.completion_tokens*pout)/1e6
            return resp
        finally:
            budget.settle(upper, actual)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def load_episode(cohort, task_id):
    import torch
    root = EVALS/cohort/'rollouts'
    for p in root.glob('*.json'):
        r = json.loads(p.read_text())
        if r['task_id'] == task_id:
            return r, torch.load(root/r['rollout_file'], weights_only=False, map_location='cpu')
    raise KeyError(task_id)


def build_sample(d):
    from openwebrl.reward_browser import _collect_recent_trajectory_images
    from slime.utils.types import Sample
    samples = [Sample.from_dict(t) for t in d['turns']]
    last = next(s for s in samples if (s.metadata or {}).get('is_last_turn'))
    last.multimodal_inputs = dict(last.multimodal_inputs or {}, judge_images=_collect_recent_trajectory_images(samples, 3))
    return last


def judge_args(model):
    return types.SimpleNamespace(judge_api_model=model, judge_api_mode='served', judge_timeout_secs=180.0,
        judge_max_attached_imgs=3, context_num_screenshots=1, browser_response_format_mode='browser_env',
        judge_prompt_variant='action_history', judge_max_tokens=None, judge_temperature=None,
        browser_history_env_feedback_max_chars=None, log_judge_output=False, log_judge_output_prob=0.0,
        debug_trace_prob=0.0, save=None)


async def judge_one(model, budget, cohort, task_id):
    import openwebrl.reward_browser as rb
    record, d = load_episode(cohort, task_id)
    sample = build_sample(d)
    score, text, timeout = await rb.compute_judge_reward_actionhistory(
        judge_args(model), sample,
        trajectory_action_history_response=sample.metadata.get('trajectory_action_history_response'),
        trajectory_action_history_text=sample.metadata.get('trajectory_action_history_text'))
    return dict(cohort=cohort, task_id=task_id, original=int(record['metrics']['successes'] > 0),
                model=model, verdict=score, judge_timeout=timeout, text=text)


def select(per_cell, seed):
    rng, picks = random.Random(seed), []
    for it, cohort in COHORTS.items():
        rows = [json.loads(p.read_text()) for p in (EVALS/cohort/'rollouts').glob('*.json')]
        valid = [r for r in rows if r['metrics']['valid_trajectories'] > 0 and r.get('terminal_status') == 'completed']
        for verdict in (1, 0):
            pool = sorted((r['task_id'] for r in valid if int(r['metrics']['successes'] > 0) == verdict))
            picks += [(it, cohort, t) for t in rng.sample(pool, min(per_cell, len(pool)))]
    return picks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--model', default='gpt-5.6-terra')
    p.add_argument('--per-cell', type=int, default=75, help='episodes per (iteration, original verdict) cell')
    p.add_argument('--check-gpt41', type=int, default=10, help='pipeline check: re-judge this many with GPT-4.1 first')
    p.add_argument('--cap-usd', type=float, default=30.)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    from dotenv import dotenv_values
    for k, v in dotenv_values(Path(__file__).resolve().parents[1]/'.env').items():
        if v and k.startswith(('OPENAI_', 'JUDGE_')):
            os.environ.setdefault(k, v)
    os.environ.setdefault('JUDGE_API_BASE', 'https://api.openai.com/v1')
    import openwebrl.reward_browser as rb
    budget = Budget(a.out_dir/'budget.json', a.cap_usd)
    inner = rb._get_openai_client(method='served')
    current = {}
    rb._get_openai_client = lambda method='served': current['client']
    picks = select(a.per_cell, a.seed)
    out = a.out_dir/'verdicts.jsonl'
    done = {(r['model'], r['task_id'], r['cohort']) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()

    async def run(model, items):
        current['client'] = capped_client(inner, budget, model)
        sem = asyncio.Semaphore(a.concurrency)
        async def one(it, cohort, task_id):
            if (model, task_id, cohort) in done:
                return
            async with sem:
                try:
                    row = await judge_one(model, budget, cohort, task_id)
                except RuntimeError as exc:
                    if 'cap' in str(exc):
                        return
                    raise
                row['iteration'] = it
                with out.open('a') as f:
                    f.write(json.dumps(row)+'\n')
        await asyncio.gather(*(one(*x) for x in items))

    check = [x for i, x in enumerate(picks) if i % max(1, len(picks)//a.check_gpt41) == 0][:a.check_gpt41]
    asyncio.run(run('gpt-4.1', check))
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    check_rows = [r for r in rows if r['model'] == 'gpt-4.1']
    agree = sum(r['verdict'] == r['original'] for r in check_rows)
    (a.out_dir/'pipeline-check.json').write_text(json.dumps(dict(gpt41_rejudged=len(check_rows), agreements=agree)))
    if check_rows and agree < 0.8*len(check_rows):
        raise SystemExit(f'Pipeline check failed: GPT-4.1 reproduced {agree}/{len(check_rows)} original verdicts')
    asyncio.run(run(a.model, picks))
    rows = [json.loads(x) for x in out.read_text().splitlines() if json.loads(x)['model'] == a.model]
    cells = {}
    for r in rows:
        c = cells.setdefault(f"iter{r['iteration']}_original{'success' if r['original'] else 'failure'}", dict(n=0, agree=0, flipped=0, unparsed=0))
        c['n'] += 1
        if r['verdict'] is None:
            c['unparsed'] += 1
        elif r['verdict'] == r['original']:
            c['agree'] += 1
        else:
            c['flipped'] += 1
    summary = dict(model=a.model, reference='gpt-4.1 action_history (original verdict)', budget=budget.state,
                   pipeline_check=json.loads((a.out_dir/'pipeline-check.json').read_text()), cells=cells,
                   definitions=dict(flipped_among_original_success='second judge says failure where GPT-4.1 said success (possible false positive)',
                                    flipped_among_original_failure='second judge says success where GPT-4.1 said failure (possible false negative)'))
    (a.out_dir/'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
