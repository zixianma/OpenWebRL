#!/usr/bin/env python3
"""Judge preserved max-step evidence within the pilot's existing per-task cap.

The shared training reward skips non-COMPLETED statuses. This standalone eval
requires a terminal verdict even at the step limit. Preserve original results,
stage corrections, and apply them only after the cohort worker has finished.
"""
import argparse
import asyncio
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/sft-decision-selection-pilot-20261004')
MODES = ('sft', 'jev', 'kev-0.8b', 'kev-27b')


def read(path):
    return json.loads(path.read_text())


def eligible(result):
    return (result.get('status') == 'Status.FAILED' and
        result.get('terminate_reason') == 'max_steps_exhausted' and
        result.get('metadata', {}).get('reward', {}).get('judge_text', '').startswith('Judge not run for status='))


async def run(root, modes, execute):
    plan, approval = read(root / 'plan.json'), read(root / 'approval.json')
    assert approval['approved'] and approval['limits']['judge_http_attempts'] == 160
    assert hashlib.sha256((root / 'plan.json').read_bytes()).hexdigest() == approval['plan_sha256']
    jobs = {attempt['job_id'] for attempt in approval['attempts']}
    accounting = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(sorted(jobs)),
        '--format=JobIDRaw,State'], text=True)
    states = {parts[0]: parts[1].split()[0] for line in accounting.splitlines()
              if (parts := line.split('|'))[0] in jobs}
    stopped = states.keys() == jobs and all(state in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT')
                                           for state in states.values())
    sys.path.insert(0, plan['source'])
    from openwebrl.decision_selection import write_json
    from openwebrl.decision_selection_eval import PilotJudge, CURRENT_TASK
    from openwebrl.eval import reward_online_mind2web as reward
    from slime.utils.types import Sample
    from openai import AsyncOpenAI
    from dotenv import dotenv_values
    secrets = dotenv_values(Path(__file__).resolve().parents[1] / '.env')
    client = AsyncOpenAI(api_key=secrets.get('JUDGE_API_KEY') or secrets['OPENAI_API_KEY'],
        base_url='https://api.openai.com/v1', max_retries=0, timeout=120) if execute else None
    args = SimpleNamespace(judge_api_model='o4-mini', judge_api_mode='served', judge_timeout_secs=120,
        log_judge_output=False)
    parser = reward.ToolParser(reward._TOOLS_INFO, parser_type=reward.resolve_parser_type(plan['actor']))
    receipts = []
    try:
        for mode in modes:
            folder = root / mode
            cohort_done = (folder / 'summary.json').exists() and read(folder / 'summary.json')['complete']
            cohort_quiescent = cohort_done or stopped
            for path in sorted((folder / 'results').glob('*.json')):
                result = read(path)
                if not eligible(result): continue
                key = path.stem
                assert result['task_id'] in plan['task_ids']
                assert key == hashlib.sha256(result['task_id'].encode()).hexdigest()
                final = (folder / 'final' / (key + '.png')).read_bytes()
                assert base64.b64decode(result['metadata']['full_image_list'][-1].split(',')[-1], validate=True) == final
                record = dict(mode=mode, task_id=result['task_id'], cohort_done=cohort_done,
                    scheduler_terminal=stopped,
                    final_sha256=hashlib.sha256(final).hexdigest(), status='eligible')
                if not execute:
                    receipts.append(record); continue
                recovery = root / 'judge-recovery' / mode
                original = recovery / 'original-results' / path.name
                corrected = recovery / 'corrected-results' / path.name
                if not original.exists(): write_json(original, result)
                assert read(original) == result, 'Original result changed during terminal recovery'
                if not corrected.exists():
                    judge = PilotJudge(folder / 'judge', client)
                    reward._get_openai_client = lambda **unused: judge
                    existing = sorted((folder / 'judge' / key).glob('response-*.json'))
                    if existing:
                        text = read(existing[-1])['choices'][0]['message']['content']
                        score, timeout = reward._parse_verdict(text), False
                    else:
                        if list((folder / 'judge' / key).glob('request-*.json')):
                            raise RuntimeError('Interrupted judge call requires explicit diagnosis before retry')
                        token = CURRENT_TASK.set(key)
                        try:
                            sample = Sample(metadata=copy.deepcopy(result['metadata']))
                            score, text, timeout = await reward._judge(args, sample, parser)
                        finally:
                            CURRENT_TASK.reset(token)
                    if score not in (0, 1) or timeout:
                        raise RuntimeError('Terminal judge did not return a valid verdict; preserve all attempts')
                    new = copy.deepcopy(result)
                    new['reward'], new['valid'] = score, True
                    new['metadata']['reward'] = dict(judge=score, combined=score, judge_text=text,
                        judge_timeout=False, protocol='online_mind2web')
                    new['terminal_judge_recovery'] = dict(reason='shared reward skipped non-COMPLETED status',
                        original_result=str(original), corrected_unix=time.time(), final_sha256=record['final_sha256'],
                        actor_browser_and_rollout_unchanged=True)
                    write_json(corrected, new)
                record['status'] = 'staged'
                if cohort_quiescent:
                    write_json(path, read(corrected)); record['status'] = 'applied'
                receipts.append(record)
            if execute and cohort_quiescent and any(r['mode'] == mode and r['status'] == 'applied' for r in receipts):
                recovery = root / 'judge-recovery' / mode
                old_summary = recovery / 'original-summary.json'
                prior_summary = read(folder / 'summary.json') if (folder / 'summary.json').exists() else {}
                if prior_summary and not old_summary.exists(): write_json(old_summary, prior_summary)
                rows = [read(f) for f in (folder / 'results').glob('*.json')]
                valid = sum(bool(r['valid']) for r in rows)
                wins = sum(r['valid'] and r['reward'] == 1 for r in rows)
                summary = dict(prior_summary, scheduled=10, attempted=len(rows),
                    complete=len(rows) == 10, provider_halted=(folder / 'selections/halt.json').exists(),
                    valid=valid, successes=wins,
                    unavailable=len(rows)-valid, success_rate_all_scheduled=wins/10,
                    success_rate_valid=wins/valid if valid else None, terminal_judge_recovered=True)
                write_json(folder / 'summary.json', summary)
        if execute:
            assert sum(len(list((root/m/'judge').rglob('request-*.json'))) for m in MODES) <= 160
            overall = root / 'summary.json'
            if overall.exists() and any(r['status'] == 'applied' for r in receipts):
                original_overall = root / 'judge-recovery/original-overall-summary.json'
                if not original_overall.exists(): write_json(original_overall, read(overall))
                write_json(overall, dict(read(overall), modes={m: read(root/m/'summary.json') for m in MODES},
                    terminal_judge_recovered=True))
            write_json(root / 'judge-recovery/latest.json', dict(updated_unix=time.time(), receipts=receipts))
    finally:
        if client: await client.close()
    return receipts


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=MODES, action='append')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    print(json.dumps(asyncio.run(run(ROOT, a.mode or MODES, a.execute)), indent=2))
