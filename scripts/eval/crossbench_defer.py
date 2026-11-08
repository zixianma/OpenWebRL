#!/usr/bin/env python3
"""Roll out now, judge later: a wrapper around the frozen crossbench selector eval (no source edits; added 2026-11-06 after the
OpenAI key started returning 401 at 11:26 and the user asked to keep rolling out while capping judge spend per day).

  exec    inside the Slurm job: identical to `run_crossbench_selector_eval.py execute`, but each worker replaces the native
          reward with a stub that makes no API call. Every finished episode is committed as a normal record with
          valid=False, reward=None and judge_text=DEFERRED (its result.json keeps the full final-turn sample).
  rejudge CPU only: for deferred records (and records whose judge failed with the dead key, --exhausted-after EPOCH), rebuild
          the final-turn Sample from result.json, run the benchmark's native reward_func through NativeCappedJudge (same
          judge model, prompt, images, parser, validity rule; separate ledger judge-budget-deferred/), rewrite result.json
          (previous copy kept as result.prejudge.json) and the record. A shared per-day ledger stops at --daily-usd.
  status  count committed / deferred / judged records per plan directory.

Canonical copy (2026-11-07) of recipe3-ablations-20261104/deferjudge/xbench_defer.py, which ran the November 6-7 evals.
Only change: the per-day ledger defaults to that same shared directory, so every copy enforces one daily cap.
See openwebrl/docs/EVAL.md.

Records with judge_text DEFERRED (or the dead-key text) are NOT invalid episodes; analyses must refuse them until rejudged.
"""
import argparse
import asyncio
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import time
from types import SimpleNamespace

REPO = Path('/gpfs/projects/krishna/zixianma/OpenWebRL')
sys.path.insert(0, str(REPO))
from openwebrl import crossbench_selector_eval as core  # noqa: E402

DEFERRED = 'DEFERRED: judge not run during rollout'
EXHAUSTED = 'All judge API retry attempts exhausted.'
HERE = Path(__file__).resolve()


# --- rollout without judging ---------------------------------------------------------------------------------------
def install_deferred(evaluation, reward, judge):
    protocol = getattr(reward, '__name__', '').rsplit('reward_', 1)[-1]

    async def deferred_reward(args, samples, **unused):
        meta = dict(judge=None, combined=None, judge_text=DEFERRED, judge_timeout=False, deferred=True, protocol=protocol)
        many = isinstance(samples, list)
        for s in (samples if many else [samples]):
            s.metadata['reward'] = dict(meta)
            s.reward = None
        return [None] * len(samples) if many else None

    def no_client(**unused):
        raise RuntimeError('Judge disabled in deferred-rollout mode')
    evaluation.reward_func = deferred_reward
    reward._get_openai_client = no_client


class NoJudgeClient:
    async def close(self):
        pass


def cmd_worker(a):
    core.install_judge = install_deferred
    core.LiveRuntime.judge_client = lambda self: NoJudgeClient()
    from openwebrl import run_evaluate  # noqa: F401  (same event-loop-policy init order as core.main)
    # The native startup gate requires a valid (judged) record; deferred records are never valid, so it is off here.
    # rejudge/status check the first records for browser steps and selector calls instead.
    summary = asyncio.run(core.run_collection(core.read_json(a.config), startup_gate=0))
    print('CROSSBENCH_SUMMARY ' + json.dumps(summary), flush=True)


def cmd_exec(a):
    spec = importlib.util.spec_from_file_location('xbench_runner', REPO / 'scripts/run_crossbench_selector_eval.py')
    rx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rx)
    original = rx.source_command

    def source_command(argv):
        argv = list(argv)                       # also used for the model servers: rewrite only the benchmark worker
        if 'openwebrl.crossbench_selector_eval' in argv:
            i = argv.index('openwebrl.crossbench_selector_eval')
            assert argv[i - 1] == '-m'
            argv[i - 1:i + 1] = [str(HERE), 'worker']
        return original(argv)
    rx.source_command = source_command
    rx.main(['execute', '--dir', a.dir, '--benchmarks', *a.benchmarks])


# --- judge later ---------------------------------------------------------------------------------------------------
def needs_judge(record, exhausted_after):
    text = record.get('judge_text')
    if text == DEFERRED:
        return True
    return exhausted_after is not None and text == EXHAUSTED and record['started_epoch'] + record['elapsed_seconds'] > exhausted_after


class DailyLedger:
    def __init__(self, root, cap):
        self.path = Path(root) / (time.strftime('%Y-%m-%d') + '.json')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap = cap

    def _rw(self, delta=0.):
        with open(str(self.path) + '.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else dict(usd=0., tasks=0)
            if delta:
                state['usd'] += delta
                state['tasks'] += 1
                core.write_json(self.path, state)
            return state

    def room(self, reserve, running=0.):
        return self._rw()['usd'] + running + reserve <= self.cap

    def add(self, usd, tasks=1):
        state = self._rw(usd)
        if tasks != 1:
            state['tasks'] += tasks - 1
            core.write_json(self.path, state)
        return state


async def rejudge_plan(plan_dir, a, daily, client):
    from openwebrl import run_evaluate as evaluation
    from slime.utils.types import Sample
    plan = core.validate_plan(plan_dir, allow_source_drift=True)
    root = Path(plan['collection'])
    pending = []
    for task_id in plan['task_ids']:
        p = root / 'records' / (plan['keys'][task_id] + '.json')
        if p.exists():
            rec = core.read_json(p)
            if needs_judge(rec, a.exhausted_after):
                pending.append((task_id, p, rec))
    if not pending:
        return dict(plan=str(plan_dir), pending=0)
    reward = core.load_native_reward(plan['benchmark'])
    caps = plan['judge_caps']
    judge = core.NativeCappedJudge(Path(plan_dir) / 'judge-budget-deferred', plan['benchmark'], max_calls=caps['max_calls'],
                                   max_usd=caps['max_usd'], attempt_context=core.ATTEMPT, client=client)
    reward._get_openai_client = lambda **unused: judge
    args = evaluation.EvalArgs(**core.eval_args_kwargs(plan, 0, root))
    sem = asyncio.Semaphore(a.concurrency)
    stop = asyncio.Event()
    done = []

    def ledger_usd():
        return judge.budget.state().get('charged_or_reserved_usd', 0.)
    start_usd = ledger_usd()      # today's spend = settled day ledger + this plan's charged-or-reserved since start

    async def one(task_id, path, rec):
        async with sem:
            if stop.is_set() or judge.halted or not daily.room(a.reserve_usd, ledger_usd() - start_usd):
                stop.set()
                return
            artifact = Path(rec['artifact'])
            result = core.read_json(artifact)
            sample = Sample(prompt=result['prompt'], response=result['response'] or '',
                            status=Sample.Status[str(result['status']).split('.')[-1]], metadata=dict(result['metadata']))
            sample.metadata.pop('judge_timeout', None)
            token = core.ATTEMPT.set(rec['key'])
            try:
                score = await reward.reward_func(args, sample)
            finally:
                core.ATTEMPT.reset(token)
            meta = sample.metadata.get('reward') or {}
            if score is None and meta.get('judge_text') in (EXHAUSTED, 'Unexpected error.'):
                stop.set()   # API failure (e.g. dead key): leave the record deferred, stop. Unparseable verdicts and judge
                             # timeouts are committed below as invalid, exactly as in a live run.
                print(f'REJUDGE_FAILED {task_id}: {meta.get("judge_text")}', flush=True)
                return
            backup = artifact.with_name('result.prejudge.json')
            if not backup.exists():
                os.replace(artifact, backup)
            new = dict(result, reward=score, metadata=sample.metadata)
            core.write_json(artifact, new)
            valid = core.is_valid_result(new)
            rec.update(valid=valid, reward=new['reward'] if valid else None, judge_text=meta.get('judge_text'),
                       judge_timeout=bool(meta.get('judge_timeout')), artifact_sha256=core.sha256_file(artifact),
                       rejudge=dict(epoch=time.time(), previous_judge_text=rec.get('judge_text'),
                                    ledger=str(judge.root), tool=str(HERE)))
            core.verify_record(rec, plan, task_id, root)
            core.write_json(path, rec)
            done.append(task_id)

    try:
        await asyncio.gather(*(one(*x) for x in pending))
    finally:
        daily.add(max(ledger_usd() - start_usd, 0.), tasks=len(done))   # all requests settled by now
    return dict(plan=str(plan_dir), pending=len(pending), judged=len(done), ledger_usd=round(ledger_usd(), 3),
                halted=judge.halted, stopped=stop.is_set())


def cmd_rejudge(a):
    from dotenv import dotenv_values
    from openai import AsyncOpenAI
    key = os.getenv('JUDGE_API_KEY') or dotenv_values(REPO / '.env').get('OPENAI_API_KEY')
    daily = DailyLedger(a.daily_root, a.daily_usd)

    async def run():
        client = AsyncOpenAI(api_key=key, base_url='https://api.openai.com/v1', max_retries=0, timeout=180)
        out = []
        for d in a.dirs:
            if not daily.room(a.reserve_usd):
                break
            r = await rejudge_plan(Path(d), a, daily, client)
            print('REJUDGE ' + json.dumps(r), flush=True)
            out.append(r)
            if r.get('stopped') and daily.room(a.reserve_usd):
                break                           # API failure, not the daily cap
        await client.close()
        return out
    from openwebrl import run_evaluate  # noqa: F401
    asyncio.run(run())
    print('DAILY ' + json.dumps(daily._rw()), flush=True)


def cmd_status(a):
    for d in a.dirs:
        plan = core.read_json(Path(d) / 'plan.json')
        recs = [core.read_json(p) for p in (Path(plan['collection']) / 'records').glob('*.json')]
        deferred = sum(r.get('judge_text') == DEFERRED for r in recs)
        exhausted = sum(r.get('judge_text') == EXHAUSTED for r in recs)
        rejudged = sum('rejudge' in r for r in recs)
        print(f"{d}  records {len(recs)}/{len(plan['task_ids'])}  deferred {deferred}  dead-key {exhausted}  rejudged {rejudged}  "
              f"browser_steps>0 {sum(r['browser_steps'] > 0 for r in recs)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    w = sub.add_parser('worker')
    w.add_argument('--config', type=Path, required=True)
    e = sub.add_parser('exec')
    e.add_argument('--dir', required=True)
    e.add_argument('--benchmarks', nargs='+', required=True)
    r = sub.add_parser('rejudge')
    r.add_argument('--dirs', nargs='+', required=True, help='benchmark plan directories <cond>/<benchmark>')
    r.add_argument('--daily-usd', type=float, required=True)
    r.add_argument('--daily-root', default=os.environ.get('CROSSBENCH_DAILY_LEDGER', '/gpfs/scrubbed/zixianma/openwebrl-runtime/recipe3-ablations-20261104/deferjudge/daily-ledger'))
    r.add_argument('--reserve-usd', type=float, default=0.5, help='stop when today + this would exceed --daily-usd')
    r.add_argument('--concurrency', type=int, default=4)
    r.add_argument('--exhausted-after', type=float, help='also rejudge dead-key records finished after this epoch')
    s = sub.add_parser('status')
    s.add_argument('--dirs', nargs='+', required=True)
    a = p.parse_args()
    dict(worker=cmd_worker, exec=cmd_exec, rejudge=cmd_rejudge, status=cmd_status)[a.command](a)


if __name__ == '__main__':
    main()
