#!/usr/bin/env python3
"""Recover only an interrupted saved judge. Dry-run default; never opens a browser.

The original unanswered request remains charged and has no invented response.
An immutable intent precedes any additional request; an interrupted recovery
requires separate diagnosis rather than automatic replay of this command.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import kev_actor_final_audit as audit

read, sha, require = audit.read, audit.sha, audit.require
SCHEMA = 'kev-saved-judge-recovery-v1'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def scheduler(job):
    raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(job),
        '--format=JobIDRaw,State,ElapsedRaw,Start,End'], text=True, env={**os.environ, 'TZ': 'UTC'})
    row = next((line.split('|') for line in raw.splitlines() if line.split('|')[0] == str(job)), None)
    require(row is not None, 'Missing scheduler record for ' + str(job))
    def epoch(value):
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()
    state = row[1].split()[0].rstrip('+')
    require(state in audit.TERMINAL, 'Owned scheduler job is not terminal: ' + str(job))
    return dict(job_id=str(job), state=state, scheduler_seconds=int(row[2]), start_unix=epoch(row[3]), end_unix=epoch(row[4]))


def check_plan(root, plan, approval):
    require(approval['approved'] and approval['plan_sha256'] == sha(root/'plan.json'), 'Approval/plan mismatch')
    require(approval['resources'] == plan['resources'] and approval['limits'] == plan['limits'], 'Approval caps mismatch')
    require(plan['protocol']['budget'] == plan['budget'], 'Protocol budget mismatch')
    require(Path(plan['budget']['root']).resolve() == (root/'budget').resolve(), 'Wrong shared budget root')
    require(all(plan['budget']['limits'][k] == plan['limits'][k] for k in plan['budget']['limits']), 'Shared budget cap mismatch')
    for name, expected in plan['source_sha256'].items():
        require(sha(Path(plan['source'])/name) == expected, 'Frozen source changed: ' + name)


def original_prefix(directory, info):
    raw = (directory/info['name']).read_bytes()[:info['bytes']]
    require(len(raw) == info['bytes'] and hashlib.sha256(raw).hexdigest() == info['sha256'],
            'Original API log prefix changed: ' + info['name'])
    require(raw.endswith(b'\n'), 'Original API log has incomplete record')
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def validate_intent(root, directory, intent):
    """Offline immutable proof validation shared by execution and final audit."""
    require(intent['schema'] == SCHEMA and intent['task_id'] == read(directory/'task.json')['task_id']
            and directory == root/'actor/tasks'/audit.digest(intent['task_id']), 'Recovery identity mismatch')
    folder = directory/'saved-judge-recovery'
    for name, expected in intent['snapshot_sha256'].items():
        require(name in {'plan.json', 'approval.json'} and sha(folder/name) == expected, 'Recovery approval snapshot changed')
    require(set(intent['snapshot_sha256']) == {'plan.json', 'approval.json'}, 'Missing recovery approval snapshots')
    plan, approval = read(folder/'plan.json'), read(folder/'approval.json')
    require(approval['approved'] and approval['plan_sha256'] == sha(folder/'plan.json')
            and approval['resources'] == plan['resources'] and approval['limits'] == plan['limits'], 'Recovery approval proof mismatch')
    require(intent['source'] == plan['source'] and intent['source_sha256'] == plan['source_sha256'], 'Recovery source proof mismatch')
    for name, expected in intent['source_sha256'].items():
        require(sha(Path(intent['source'])/name) == expected, 'Recovery frozen source changed: ' + name)
    required = {'task.json', 'trajectory.json', 'browser-session.json', 'judge-request.json', 'final.jpg',
                'state-0000.json.gz', f"state-{plan['protocol']['max_decisions']+1:04d}.json.gz", 'worker.log'}
    require(required <= set(intent['evidence_sha256']), 'Incomplete original recovery evidence')
    for name, expected in intent['evidence_sha256'].items():
        require(Path(name).name == name and sha(directory/name) == expected, 'Original recovery evidence changed: ' + name)
    require(read(directory/'browser-session.json')['stopped'] is True, 'Recovery browser is not closed')
    require(intent['original_api_attempts']['name'] == 'api-attempts.jsonl'
            and intent['original_api_responses']['name'] == 'api-responses.jsonl', 'Recovery API prefix names mismatch')
    attempts = original_prefix(directory, intent['original_api_attempts'])
    responses = original_prefix(directory, intent['original_api_responses'])
    indexed = {r['call_id']: r for r in responses}
    missing = [a for a in attempts if a['call_id'] not in indexed]
    require(len(missing) == 1 and missing[0]['provider'] == 'judge' and missing[0]['call_id'] == intent['missing_call_id']
            and missing[0]['request'] == read(directory/'judge-request.json'), 'Recovery original unanswered call mismatch')
    counts = Counter(a['provider'] for a in attempts)
    require(counts['judge'] == 1 and intent['original_counts'] == {k: counts[k] for k in audit.KINDS}, 'Recovery original counts mismatch')
    for provider in audit.KINDS:
        require([a['call_id'] for a in attempts if a['provider'] == provider] ==
                [f'{provider}-{i:04d}' for i in range(1, counts[provider]+1)], 'Original call counter sequence mismatch')
    jobs = intent['scheduler']
    require(len(jobs) == len(approval['attempts']) and {j['job_id'] for j in jobs} ==
            {str(a['job_id']) for a in approval['attempts']} and all(j['state'] in audit.TERMINAL for j in jobs),
            'Recovery does not prove all owned jobs terminal')
    require(sum(j['scheduler_seconds'] for j in jobs) <= approval['resources']['total_seconds'], 'Recovery scheduler cap exceeded')
    failed = next((j for j in jobs if j['job_id'] == intent['failed_job_id']), None)
    require(failed and failed['state'] in audit.TERMINAL - {'COMPLETED'} and
            str(approval['attempts'][-1]['job_id']) == failed['job_id'], 'Recovery lacks latest failed-job proof')
    require(failed['start_unix'] <= missing[0]['started_unix'] <= failed['end_unix'] + 2
            and failed['end_unix'] <= intent['prepared_unix'], 'Interrupted judge is outside failed job interval')
    require(type(intent['max_additional_calls']) is int and 1 <= intent['max_additional_calls'] <= 4-counts['judge'],
            'Recovery exceeds remaining per-task judge allowance')
    return dict(attempts=attempts, responses=responses, plan=plan, approval=approval)


def prepare(root, task_id, failed_job_id, max_additional_calls=1, scheduler_fn=scheduler):
    root = Path(root).resolve()
    plan, approval = read(root/'plan.json'), read(root/'approval.json')
    check_plan(root, plan, approval)
    task = next((t for t in plan['tasks'] if t['task_id'] == task_id), None)
    require(task is not None, 'Task not in approved plan')
    directory = root/'actor/tasks'/audit.digest(task_id)
    require(not (directory/'saved-judge-recovery').exists(), 'Preserve existing recovery; diagnose before retry')
    cfg = plan['protocol']
    row = audit.audit_task(root, task, cfg, Path(plan['source'])/'openwebrl/eval/reward_online_mind2web.py', _pending_judge=True)
    require(type(max_additional_calls) is int and 1 <= max_additional_calls <= 4-row['api_attempts']['judge'],
            'Exceeds remaining per-task judge allowance')
    jobs = [scheduler_fn(str(a['job_id'])) for a in approval['attempts']]
    require(all(j['state'] in audit.TERMINAL for j in jobs), 'Owned scheduler job is not terminal')
    require(sum(j['scheduler_seconds'] for j in jobs) <= approval['resources']['total_seconds'], 'Scheduler budget exceeded')
    failed = next((j for j in jobs if j['job_id'] == str(failed_job_id)), None)
    require(failed and failed['state'] != 'COMPLETED' and str(approval['attempts'][-1]['job_id']) == str(failed_job_id),
            'Original job must be the latest owned failed job')
    usage = read(root/'budget/usage.json')
    require(usage['limits'] == plan['budget']['limits'], 'Shared budget limits changed')
    require(usage['reserved'].get('judge_http_attempts', 0) + max_additional_calls <= usage['limits']['judge_http_attempts'],
            'Global judge allowance exhausted')
    def prefix(name):
        path = directory/name
        return dict(name=name, bytes=path.stat().st_size, sha256=sha(path))
    immutable = [p for p in directory.iterdir() if p.is_file() and p.name not in
                 {'api-attempts.jsonl', 'api-responses.jsonl', 'heartbeat.json'} and p.suffix != '.tmp']
    intent = dict(schema=SCHEMA, task_id=task_id, failed_job_id=str(failed_job_id), prepared_unix=time.time(),
        source=plan['source'], source_sha256=plan['source_sha256'], recovery_helper_sha256=sha(Path(__file__)),
        snapshot_sha256={n: sha(root/n) for n in ('plan.json', 'approval.json')},
        evidence_sha256={p.name: sha(p) for p in immutable},
        original_api_attempts=prefix('api-attempts.jsonl'), original_api_responses=prefix('api-responses.jsonl'),
        original_counts=row['api_attempts'], missing_call_id=row['missing_call_id'], scheduler=jobs,
        max_additional_calls=max_additional_calls, original_judge_latency_unknown=True,
        budget_reserved_before=usage['reserved'], global_judge_cap=usage['limits']['judge_http_attempts'])
    original_judge = next(a for a in audit.lines(directory/'api-attempts.jsonl') if a['call_id'] == row['missing_call_id'])
    require(failed['start_unix'] <= original_judge['started_unix'] <= failed['end_unix']+2 <= intent['prepared_unix']+2,
            'Interrupted judge is outside failed job interval')
    return directory, intent


def frozen_transport(source):
    # A fresh command process imports both transport and shared reservation code
    # from the approved frozen source, never the changing workspace worker.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    for name in ('openwebrl', 'openwebrl.selection_budget', 'openwebrl.decision_selection'):
        if name in sys.modules:
            origin = getattr(sys.modules[name], '__file__', '')
            require(origin and Path(origin).resolve().is_relative_to(Path(source).resolve()), 'Non-frozen module already imported: ' + name)
    path = Path(source)/'openwebrl/jev_eval.py'
    spec = importlib.util.spec_from_file_location('_frozen_saved_judge', path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module.ModelTransport


def execute_prepared(root, directory, intent, key, transport_factory=None):
    """Caller holds owner+submit locks; tests inject a transport without network."""
    require(bool(key), 'Missing judge credential')
    folder = directory/'saved-judge-recovery'
    require(not folder.exists() and not (directory/'result.json').exists() and not (directory/'judge-response.json').exists(),
            'Recovery already exists or result/verdict is present')
    folder.mkdir()
    for name in ('plan.json', 'approval.json'):
        require(sha(root/name) == intent['snapshot_sha256'][name], 'Plan/approval changed after preflight')
        with (folder/name).open('xb') as handle:
            handle.write((root/name).read_bytes()); handle.flush(); os.fsync(handle.fileno())
    original = validate_intent(root, directory, intent)
    atomic_json(folder/'intent.json', intent)
    cfg = original['plan']['protocol']
    transport = (transport_factory or frozen_transport(intent['source']))(directory, cfg)
    transport.counts = dict(intent['original_counts'])
    started = time.monotonic()
    try:
        response = transport.request('judge', cfg['judge_base_url']+'/chat/completions', key,
            read(directory/'judge-request.json'), attempts=intent['max_additional_calls'], timeout=120)
        atomic_json(directory/'judge-response.json', response)
        judge_text = response['choices'][0]['message']['content'] or ''
        score = None if 'status:' not in judge_text.lower() else (1.0 if 'success' in judge_text.lower().split('status:', 1)[1] else 0.0)
        trajectory = read(directory/'trajectory.json')
        result = audit.saved_actor_metadata(trajectory['task'], trajectory, cfg, audit.lines(directory/'api-attempts.jsonl'))
        elapsed = time.monotonic()-started
        result.update(valid=score is not None, score=score, judge_model=cfg['judge_model'], judge_prompt_variant='agenttrek',
            judge_text=judge_text, judge_error=None if score is not None else 'unparseable_verdict', error_stage='actor',
            actor_seconds=trajectory['actor_seconds'], total_seconds=trajectory['actor_seconds']+elapsed, recovery_judge_seconds=elapsed,
            total_seconds_scope='Measured actor plus recovery judge; original interrupted judge latency is unknown.',
            saved_judge_recovery=dict(intent_sha256=sha(folder/'intent.json'),
                original_unanswered_call_id=intent['missing_call_id'], original_judge_latency_unknown=True))
        atomic_json(folder/'completion.json', dict(intent_sha256=sha(folder/'intent.json'), completed_unix=time.time(),
            result_digest=audit.digest(result), judge_response_sha256=sha(directory/'judge-response.json'),
            recovery_judge_seconds=elapsed, api_attempts=transport.counts))
        atomic_json(directory/'result.json', result)
        return result
    except Exception as exc:
        # No exception body/credential is serialized. All charged calls stay in
        # the original transport logs. This command will refuse automatic replay.
        atomic_json(folder/'failure.json', dict(intent_sha256=sha(folder/'intent.json'), error_type=type(exc).__name__,
            failed_unix=time.time(), api_attempts=transport.counts))
        raise
    finally:
        transport.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--failed-job-id', required=True)
    parser.add_argument('--max-additional-calls', type=int, default=1)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--env-file', type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    with ExitStack() as stack:
        if args.execute:
            for name in ('submit.lock', 'owner.lock'):
                handle = stack.enter_context((root/name).open('a'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory, intent = prepare(root, args.task_id, args.failed_job_id, args.max_additional_calls)
        if args.execute:
            values = dict(os.environ)
            if args.env_file:
                from dotenv import dotenv_values
                values = {**dotenv_values(args.env_file), **values}
            key = values.get('JUDGE_API_KEY') or values.get('OPENAI_API_KEY')
            result = execute_prepared(root, directory, intent, key)
            print(json.dumps(dict(status='saved_judge_recovered', task_id=args.task_id, valid=result['valid'],
                                  score=result['score'], api_attempts=result['api_attempts'])))
        else:
            print(json.dumps(dict(status='dry_run_ready', task_id=args.task_id, missing_call_id=intent['missing_call_id'],
                counts=intent['original_counts'], max_additional_calls=intent['max_additional_calls'],
                scheduler_seconds=sum(j['scheduler_seconds'] for j in intent['scheduler']), mutations=False)))


if __name__ == '__main__':
    main()
