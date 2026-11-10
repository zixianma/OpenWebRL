#!/usr/bin/env python3
"""Run Microsoft's Universal Verifier (MMRubricAgent, microsoft/fara webeval) as a candidate outcome reward.

Uses the released pipeline unchanged: `webeval/scripts/verify_trajectories.py::_run_one` (rubric generation,
multimodal rubric scoring, outcome verification), with the outcome verdict as the score. Only the clients are ours:
every call goes through a budget-capped OpenAI client shared across worker processes (--cap-usd; upper-bound
reservation per request, settled with the reported usage). The API key is read from the repo .env and never written.
Deviations from the paper's setup, recorded in the output:
  * both judge roles use --judge-model / --action-model (default gpt-5.6-terra; the release uses gpt-5 + o4-mini,
    and o4-mini was retired here on 2026-10-23);
  * the VerifierAgent failure-analysis stage is skipped unless --with-failure-analysis: it adds only an error
    taxonomy and does not change the outcome verdict.

Inputs: CUAVerifierBench rows (native Fara log format) are rebuilt as webeval trajectory folders under
--work-dir/<split>/<task_id>/ (web_surfer.log, screenshot_<i>.png, <task_id>_final_answer.json).
Output: --out (jsonl, one row per trajectory: id, verdict = outcome_success, rubric score, status), resumable.

Needs the separate light venv (runtime uv-verifier-venv) with FARA_ROOT pointing at the pinned clone.
"""
import argparse
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

FARA_ROOT = Path(os.environ.get('FARA_ROOT', '/gpfs/scrubbed/zixianma/openwebrl-runtime/external/fara-a675d6d'))
BENCH = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/datasets/cuaverifierbench-c19eb32/data')
PRICES = {'gpt-5.6-terra': (2.5, 15.0), 'gpt-4.1': (2.0, 8.0)}  # USD per M input/output tokens (upper bounds)


class FileBudget:
    """Spend ledger shared by worker processes (flock on the ledger file)."""

    def __init__(self, path, cap):
        self.path, self.cap = Path(path), cap

    def _update(self, fn):
        with open(self.path.with_suffix('.lock'), 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else dict(spent_usd=0., reserved_usd=0., calls=0)
            out = fn(state)
            tmp = self.path.with_suffix('.tmp'); tmp.write_text(json.dumps(state)); tmp.replace(self.path)
            return out

    def reserve(self, usd):
        def fn(s):
            if s['spent_usd'] + s['reserved_usd'] + usd > self.cap:
                raise RuntimeError('API cap reached')
            s['reserved_usd'] += usd; s['calls'] += 1
        self._update(fn)

    def settle(self, reserved, actual):
        def fn(s):
            s['reserved_usd'] -= reserved; s['spent_usd'] += reserved if actual is None else actual
        self._update(fn)


def build_folders(split, work_dir):
    import pyarrow.parquet as pq
    import glob
    tasks = {}
    for f in sorted(glob.glob(str(BENCH/'trajectories'/f'{split}-*.parquet'))):
        for r in pq.read_table(f).to_pylist():
            d = work_dir/split/r['task_id']
            tasks[r['task_id']] = dict(id=r['task_id'], question=r['instruction'], init_url=r['init_url'])
            if (d/f"{r['task_id']}_final_answer.json").exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            names = []
            for i, s in enumerate(r['screenshots'], 1):
                (d/f'screenshot_{i}.png').write_bytes(s['bytes']); names.append(f'screenshot_{i}.png')
            (d/'web_surfer.log').write_text(r['web_surfer_log'] if r['web_surfer_log'].endswith('\n') else r['web_surfer_log'] + '\n')
            (d/f"{r['task_id']}_final_answer.json").write_text(json.dumps(dict(
                final_answer=r['final_answer'], env_state_json='<no_answer>', env_state_raw='<no_answer>',
                screenshots=names, is_aborted=bool(r['is_aborted']), is_rel_paths=True)))
    return tasks


def _init(cfg, tasks):
    sys.path[:0] = [str(FARA_ROOT/'src'), str(FARA_ROOT/'webeval'/'src'), str(FARA_ROOT/'webeval'/'scripts')]
    import logging
    import verify_trajectories as vt
    from fara import FARA_ACTION_DEFINITIONS
    from webeval.oai_clients.graceful_client import GracefulRetryClient
    from webeval.oai_clients.wrapper import OpenAIClientWrapper
    from webeval.rubric_agent import MMRubricAgent, MMRubricAgentConfig, VerifierAgent, VerifierAgentConfig
    budget = FileBudget(cfg['budget'], cfg['cap_usd'])

    class Budgeted(OpenAIClientWrapper):
        async def create(self, messages, tools=(), json_output=None, extra_create_args={}):
            pin, pout = PRICES[self.model]
            n_img = sum(1 for m in messages for c in (getattr(m, 'content', None) if isinstance(getattr(m, 'content', None), list) else [])
                        if not isinstance(c, str))
            chars = sum(len(c) for m in messages for c in ([m.content] if isinstance(getattr(m, 'content', None), str) else
                                                             [x for x in (getattr(m, 'content', None) or []) if isinstance(x, str)]))
            upper = ((chars/2 + 3000*n_img)*pin + 16000*pout)/1e6
            budget.reserve(upper)
            actual = None
            try:
                res = await super().create(messages, tools=tools, json_output=json_output, extra_create_args=extra_create_args)
                actual = (res.usage.prompt_tokens*pin + res.usage.completion_tokens*pout)/1e6
                return res
            finally:
                budget.settle(upper, actual)

    log = logging.getLogger('uv.worker')
    key = cfg['api_key']

    def client(model):
        return GracefulRetryClient(clients=[Budgeted(model=model, api_key=key)], logger=log, max_retries=3, timeout=600)
    action_client, judge_client = client(cfg['action_model']), client(cfg['judge_model'])
    vt._GLOBAL_TASKS, vt._GLOBAL_ARGS = tasks, cfg['vt_args']
    vt._GLOBAL_AGENT = MMRubricAgent(config=MMRubricAgentConfig(
        o4mini_client=action_client, gpt5_client=judge_client,
        max_images_per_criterion=cfg['vt_args']['max_images_per_criterion'],
        majority_vote_instances=cfg['vt_args']['majority_vote_instances'], redo_eval=False,
        rubric_score_threshold=cfg['vt_args']['rubric_threshold'], action_definitions=FARA_ACTION_DEFINITIONS))
    if cfg['with_failure_analysis']:
        vt._GLOBAL_VERIFIER = VerifierAgent(config=VerifierAgentConfig(o4mini_client=action_client, gpt5_client=judge_client,
                                                                        action_definitions=FARA_ACTION_DEFINITIONS))
    else:
        class NoFailureAnalysis:
            async def verify(self, **kwargs):
                return {}
        vt._GLOBAL_VERIFIER = NoFailureAnalysis()
    globals()['_vt'] = vt


def _one(traj_dir):
    return _vt._run_one(traj_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', default='fara7b_om2w_browserbase')
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--budget', type=Path, required=True, help='shared spend ledger (json)')
    p.add_argument('--cap-usd', type=float, required=True)
    p.add_argument('--judge-model', default='gpt-5.6-terra')
    p.add_argument('--action-model', default='gpt-5.6-terra')
    p.add_argument('--processes', type=int, default=4)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--ids', default=None, help='comma-separated task ids to run (default: all)')
    p.add_argument('--with-failure-analysis', action='store_true')
    p.add_argument('--env-file', type=Path, default=Path(__file__).resolve().parents[1]/'.env')
    a = p.parse_args()
    from dotenv import dotenv_values
    key = os.environ.get('OPENAI_API_KEY') or dotenv_values(a.env_file).get('OPENAI_API_KEY')
    tasks = build_folders(a.split, a.work_dir)
    done = {json.loads(x)['id'] for x in a.out.read_text().splitlines()} if a.out.exists() else set()
    ids = a.ids.split(',') if a.ids else sorted(tasks)
    todo = [t for t in ids if f'{a.split}/{t}' not in done][:a.limit]
    vt_args = dict(rubric_threshold=0.8, max_images_per_criterion=5, mm_keypoint_score_threshold=3,
                   majority_vote_instances=1, redo_eval=False, success_criterion='outcome')
    cfg = dict(budget=str(a.budget), cap_usd=a.cap_usd, judge_model=a.judge_model, action_model=a.action_model,
               api_key=key, vt_args=vt_args, with_failure_analysis=a.with_failure_analysis)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with mp.get_context('spawn').Pool(a.processes, initializer=_init, initargs=(cfg, tasks)) as pool:
        for r in pool.imap_unordered(_one, [str(a.work_dir/a.split/t) for t in todo]):
            if r.get('status') == 'error' and 'API cap reached' in r.get('error', ''):
                print('cap reached; stopping', flush=True)
                pool.terminate()
                break
            row = dict(id=f"{a.split}/{r['task_id']}", group=a.split, task_id=r['task_id'], status=r['status'],
                       verdict=(None if r.get('outcome_success') is None else float(r['outcome_success'] is True))
                       if r['status'] == 'ok' else None,
                       rubric_score=r.get('rubric_score'), rubric_is_success=r.get('rubric_is_success'),
                       error=r.get('error'), duration_sec=r.get('duration_sec'), judge_model=a.judge_model,
                       action_model=a.action_model, failure_analysis=a.with_failure_analysis,
                       vt_args=vt_args, fara_commit=FARA_ROOT.name)
            with a.out.open('a') as f:
                f.write(json.dumps(row) + '\n')
            print(row['id'], row['status'], row['verdict'], row['rubric_score'], row['duration_sec'], flush=True)
    print(json.dumps(FileBudget(a.budget, a.cap_usd)._update(lambda s: dict(s))), flush=True)


if __name__ == '__main__':
    main()
