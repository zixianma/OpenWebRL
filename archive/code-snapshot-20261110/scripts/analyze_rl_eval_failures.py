#!/usr/bin/env python3
"""Classify failed full-300 evaluation episodes from saved trajectories (CPU only).

Stage 1 (--extract) reads each task's saved rollout (.pt) and writes a compact JSONL with
actions, tool results, termination and judge reasoning (no screenshots). Stage 2
(--summarize) assigns every episode one category by fixed rules and reports counts.

Categories, first match wins:
  success; invalid:<error type>; turn_limit (did not call done); repeat_loop (an identical
  non-wait action issued >= 3 times); claims_infeasible (done message says the task cannot be done);
  tool_errors (>= 30% of tool calls failed); wrong_answer (done issued, judged failure).
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import re

EVALS = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations')
INFEASIBLE = re.compile(r"\b(unable|cannot|can't|could not|couldn't|not possible|impossible|no longer|"
                        r"not available|unavailable|does not (?:exist|offer|have)|doesn't (?:exist|offer|have)|"
                        r"requires? (?:a )?(?:login|sign[- ]?in|account))\b", re.I)
TOOL_CALL = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)


def compact(pt):
    import torch
    d = torch.load(pt, weights_only=False, map_location='cpu')
    turns = d.get('turns') or []
    steps = []
    for t in turns:
        m = t.get('metadata') or {}
        call = TOOL_CALL.search(t.get('response') or '')
        try:
            call = json.loads(call.group(1)) if call else None
        except json.JSONDecodeError:
            call = {'name': 'unparsable', 'arguments': call.group(1)[:200]}
        results = m.get('step_tool_responses') or []
        steps.append(dict(call=call, results=[dict(tool=r.get('tool_name'), response=str(r.get('tool_response'))[:240])
                                              for r in results]))
    last = (turns[-1].get('metadata') or {}) if turns else {}
    first = (turns[0].get('metadata') or {}) if turns else {}
    reward = last.get('reward') or {}
    return dict(task_id=d.get('task_id'), intent=first.get('intent'), start_url=first.get('start_url'),
                error_type=d.get('error_type'), terminate_reason=last.get('terminate_reason'),
                total_steps=last.get('total_steps'), steps=steps, judge=reward.get('judge'),
                judge_text=str(reward.get('judge_text') or '')[:1500])


def extract_one(json_path):
    r = json.loads(Path(json_path).read_text())
    m = r['metrics']
    row = dict(task_id=r['task_id'], success=int(m['successes'] > 0), valid=int(m['valid_trajectories'] > 0),
               error_type=r.get('error_type'), terminal_status=r.get('terminal_status'), turns=r.get('turns'))
    pt = Path(json_path).with_name(r['rollout_file']) if r.get('rollout_file') else None
    if pt and pt.is_file():
        row.update({k: v for k, v in compact(pt).items() if k not in ('task_id', 'error_type') or v is not None})
    return row


def classify(r, max_steps):
    if r['success']:
        return 'success'
    if not r['valid']:
        reason = str(r.get('terminate_reason') or r.get('error_type') or r.get('terminal_status') or 'unknown')
        for key, label in (('/reset', 'start_page_failed'), ('rollout_task_timeout', 'rollout_timeout'),
                           ('env_step_error', 'env_step_error')):
            if key in reason:
                return f'invalid:{label}'
        return 'invalid:' + reason.split(':')[0][:40]
    steps = r.get('steps') or []
    calls = [s['call'] for s in steps if s.get('call')]
    names = [c.get('name') for c in calls]
    if 'done' not in names:
        return 'turn_limit' if (r.get('total_steps') or len(steps)) >= max_steps else 'no_done_other'
    keys = Counter(json.dumps(c, sort_keys=True) for c in calls if c.get('name') not in ('done', 'wait'))
    if keys and keys.most_common(1)[0][1] >= 3:
        return 'repeat_loop'
    done = next(c for c in reversed(calls) if c.get('name') == 'done')
    if INFEASIBLE.search(json.dumps(done.get('arguments', ''))):
        return 'claims_infeasible'
    results = [x for s in steps for x in s['results'] if x.get('tool') != 'done']
    if results and sum(not x['response'].startswith('Succeed') for x in results)/len(results) >= .3:
        return 'tool_errors'
    return 'wrong_answer'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohorts', nargs='+', required=True, help='evaluation directory names under EVALS')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--extract', action='store_true')
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--max-steps', type=int, default=30)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    workers = int(os.getenv('SLURM_CPUS_PER_TASK', '1'))
    # A cohort is an evaluation directory name under EVALS, or label=/absolute/evaluation/dir.
    cohorts = dict(c.split('=', 1) if '=' in c else (c, str(EVALS/c)) for c in a.cohorts)
    a.cohorts = list(cohorts)
    if a.extract:
        for name, directory in cohorts.items():
            out = a.out_dir/f'{name}.jsonl'
            if out.is_file():
                continue
            files = sorted((Path(directory)/'rollouts').glob('*.json'))
            with ProcessPoolExecutor(workers) as pool:
                rows = list(pool.map(extract_one, files, chunksize=4))
            if len({r['task_id'] for r in rows}) != 300:
                raise ValueError(f'{name}: expected 300 tasks, got {len(rows)}')
            tmp = out.with_suffix('.tmp')
            tmp.write_text(''.join(json.dumps(r)+'\n' for r in rows)); tmp.rename(out)
            print('extracted', name, flush=True)
    if a.summarize:
        summary = {}
        for name in a.cohorts:
            rows = [json.loads(x) for x in (a.out_dir/f'{name}.jsonl').read_text().splitlines()]
            cats = Counter(classify(r, a.max_steps) for r in rows)
            valid_fail = [r for r in rows if r['valid'] and not r['success']]
            summary[name] = dict(tasks=len(rows), categories=dict(cats.most_common()),
                                 valid_failures=len(valid_fail),
                                 mean_steps_valid_failure=sum(r.get('total_steps') or 0 for r in valid_fail)/max(1, len(valid_fail)),
                                 mean_steps_success=sum(r.get('total_steps') or 0 for r in rows if r['success'])/max(1, sum(r['success'] for r in rows)))
        (a.out_dir/'summary.json').write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
