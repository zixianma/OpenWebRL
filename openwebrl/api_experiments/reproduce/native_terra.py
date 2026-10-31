"""Portable trusted native endpoints and one shared bounded Terra judge pool.

Actor prompts/actions/models are unchanged. This module accepts only separately
verified broker-private terminal evidence and executed selected-action history;
earlier WebJudge labels are never read or transferred.
"""
from __future__ import annotations
import hashlib
from contextlib import contextmanager
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

import rejudge


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_science():
    """Read-only exact public frozen judge, prompt, pricing and budget checks."""
    return rejudge.science()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_bytes().split(b'\n') if line.strip()]


@contextmanager
def wall_guard(deadline):
    """The sequential CLI owns the judge request through its absolute deadline."""
    if deadline is None:
        yield
        return
    need(threading.current_thread() is threading.main_thread(), 'Bounded native Terra judging requires its owner main thread')
    remaining = deadline - time.time()
    need(remaining > 0, 'Native wall deadline reached before Terra dispatch')
    prior_handler = signal.getsignal(signal.SIGALRM)
    prior_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    def expired(signum, frame):
        raise TimeoutError('Native Terra wall deadline reached; preserve outstanding reservation')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(remaining, prior_timer[0]) if prior_timer[0] > 0 else remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
        if prior_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(.000001, prior_timer[0] - (time.monotonic() - started)), prior_timer[1])


def selected_history(runtime, directory, rows, receipt):
    helper = load('_portable_native_paid_history_' + directory.name, runtime / 'native_capture.py')
    native = read(directory / 'result.json')
    evidence = Path(native['owned_teardown']['broker_verification']['evidence']).resolve()
    need(evidence == Path(receipt['screenshot_path']).resolve().parent,
         'Capture and independently owned broker evidence differ')
    events = jsonl(runtime / 'shard0/.host/meter/events.jsonl')
    owner_events = jsonl(evidence / 'command-timeout-events.jsonl')
    broker_events = jsonl(evidence / 'browser-broker-events.jsonl')
    starts = [event for event in broker_events if event['event'] == 'broker_started']
    need(len(starts) == 1 and Path(starts[0]['output']).resolve() == directory
         and starts[0]['runtime_child_source_binding'] == native['runtime_child_source_binding'],
         'Broker/native source identity differs')
    sys.path.insert(0, str(runtime / 'upstream/src'))
    from cuawright.webwright.models.openai_model import parse_tool_call_output
    parser = SimpleNamespace(_parse_response=parse_tool_call_output)
    proof = [helper.paid_command_binding(runtime, 0, directory, row, None,
                                        events, parser, owner_events) for row in rows]
    target = directory / 'selected-history-paid-binding.json'
    write(target, {'verified': True, 'all_selected_steps_bound': True,
                   'browser_steps': len(rows), 'bindings': proof,
                   'host_evidence_sha256': {str(path): sha(path) for path in
                       (evidence / 'command-timeout-events.jsonl', evidence / 'browser-broker-events.jsonl')}})
    return target


def endpoint(runtime, directory, task):
    """Fail closed before any judge call if endpoint, source, or action proof fails."""
    runtime, directory = Path(runtime).resolve(), Path(directory).resolve()
    need(directory.parent == runtime / 'shard0/episodes' and directory.name == task['task_id'],
         'Endpoint outside its declared isolated task')
    native = read(directory / 'result.json')
    need(native.get('native_execution_complete') is True and not native.get('error')
         and native.get('exit_status') in ('Submitted', 'LimitsExceeded'), 'Native execution incomplete')
    need(native.get('owned_teardown', {}).get('process_verification', {}).get('verified') is True
         and read(directory / 'parent-owned-teardown.json').get('verified') is True,
         'Native endpoint owned teardown unverified')
    inventory = read(runtime / 'runtime-inventory.json')
    need(inventory['passed'] is True and inventory['frozen_plan_sha256'] == sha(runtime / 'plan.json'),
         'Runtime plan/source inventory differs')
    for relative, expected in inventory['source_hashes'].items():
        path = (runtime / relative).resolve()
        need(path.is_relative_to(runtime) and sha(path) == expected, 'Frozen runtime source changed: ' + relative)
    need(native['runtime_child_source_binding']['source_hashes'] == inventory['source_hashes'],
         'Executed native child source differs')
    capture = native.get('terminal_capture', {})
    need(capture.get('verified') is True, 'Trusted terminal capture unavailable; earlier frames are forbidden')
    receipt_path = Path(capture['receipt_path']).resolve()
    evidence_root = runtime / 'shard0/episodes/.broker-evidence'
    need(receipt_path.is_relative_to(evidence_root.resolve()) and receipt_path.parent != directory
         and sha(receipt_path) == capture['receipt_sha256'], 'Terminal receipt is not authentic broker-private evidence')
    receipt = read(receipt_path)
    need(receipt['verified'] is True and receipt['episode'] == str(directory)
         and receipt['episode_id'] == task['task_id']
         and receipt['source_sha256'] == inventory['source_hashes']['trusted_terminal_capture.py']
         and all(receipt.get(key) is True for key in ('no_new_actor_decision',
                 'no_selected_command_in_flight', 'new_actor_commands_admission_closed')),
         'Trusted endpoint identity/source/admission proof differs')
    details = receipt['capture']
    for name, expected in receipt['source_files_sha256'].items():
        need(name in ('.browser-session.json', 'browser-steps.jsonl', 'trajectory.json', 'browser-sessions.jsonl'),
             'Unexpected endpoint source binding')
        if expected is None:
            need(name == '.browser-session.json' and not (directory / name).exists()
                 and details.get('endpoint_kind') == 'trusted_preclose_after_final_selected_browser_action'
                 and all(details.get(key) is True for key in ('session_epoch_verified',
                     'no_later_browser_action_verified', 'no_later_owned_browser_launch_verified'))
                 and receipt['source_files_sha256'].get('browser-sessions.jsonl') == sha(directory / 'browser-sessions.jsonl'),
                 'Deleted manifest lacks exact trusted closed-session lifecycle proof')
        elif name != '.browser-session.json':
            # Unchanged native cleanup updates an active session after capture.
            need(sha(directory / name) == expected, 'Selected endpoint history changed after capture')
    image = Path(receipt['screenshot_path']).resolve()
    need(image.parent == receipt_path.parent and sha(image) == receipt['screenshot_sha256'] == capture['screenshot_sha256'],
         'Trusted original endpoint PNG differs')
    rows = jsonl(directory / 'browser-steps.jsonl')
    need(rows and rows[-1]['browser_step'] == receipt['last_browser_step'], 'Final selected browser action differs')
    paid_proof = selected_history(runtime, directory, rows, receipt)
    answer = read(directory / 'trajectory.json').get('info', {}).get('submission', '')
    need(type(answer) is str and read(directory / 'final-answer.json')['final_answer'] == answer,
         'Final answer differs from native saved submission')
    plan = read(runtime / 'plan.json')
    actions = [row['action'] for row in rows] + (['Final answer: ' + answer] if answer else [])
    preclose = details.get('endpoint_kind') == 'trusted_preclose_after_final_selected_browser_action'
    return {'id': task['task_id'], 'original_valid': True, 'eligible': True,
            'task': task['intent'], 'actions': actions,
            'endpoint_scope': 'last_browser_action_capture' if preclose else 'dedicated_terminal_capture',
            'last_action_capture_verified': preclose, 'screenshot': str(image), 'screenshot_sha256': sha(image),
            'cohort': {'benchmark': str(task.get('benchmark') or plan.get('benchmark', 'collaborator_tasks')), 'harness': 'native', 'arm': plan['arm_id'],
                       'actor_model': plan['model'], 'actor_protocol_sha256': sha(runtime / 'plan.json')},
            'evidence_bindings': [{'path': str(path), 'sha256': sha(path)} for path in
                (receipt_path, directory / 'result.json', directory / 'parent-owned-teardown.json',
                 directory / 'browser-steps.jsonl', directory / 'trajectory.json', paid_proof)]}


class TerraSession:
    """Exactly one all-task request-budget pool, distinct from the actor gateway."""
    def __init__(self, runtime, judge_usd, judge_calls, key, *, science=None, client_factory=None):
        need(type(judge_calls) is int and judge_calls > 0 and math.isfinite(float(judge_usd))
             and float(judge_usd) > 0, 'Explicit positive shared Terra dollar/call caps required')
        self.runtime = Path(runtime)
        self.judge, budget_module, self.sources = science if science is not None else verify_science()
        self.budget = budget_module.ReservationBudget(self.runtime / 'terra-budget', str(judge_usd), judge_calls)
        self.key = key
        self.client_factory = client_factory
        write(self.runtime / 'terra-source-provenance.json', {'public_rejudge_sha256': sha(Path(rejudge.__file__)),
              'frozen_sources': self.sources, 'judge_protocol': self.judge.protocol(),
              'actor_gateway_external_judge_usd': 0, 'actor_gateway_external_judge_calls': 0,
              'judge_pool_usd': str(judge_usd), 'judge_pool_calls': judge_calls})

    def available(self):
        state = self.budget.snapshot()
        return (not self.budget.halted() and state['calls'] < state['max_calls']
                and float(state['charged_or_reserved_usd']) < float(state['max_usd'])
                and state['unknown_calls'] == state['reserved_calls'] == 0)

    def judge_once(self, task, directory, *, endpoint_function=endpoint, deadline=None):
        need(self.available(), 'Shared Terra pool halted or its explicit credits exhausted')
        need(deadline is None or time.time() < deadline, 'Native wall deadline reached before endpoint/judge work')
        directory = Path(directory)
        row = endpoint_function(self.runtime, directory, task)
        manifest = directory / 'terra-endpoint.json'
        write(manifest, {'schema': 'terra-endpoint-input-v1', 'episodes': [row]})
        # The public adapter verifies exact image geometry and every provenance
        # binding before allocating a request reservation or constructing a client.
        rows, _ = rejudge.endpoints(manifest, self.judge)
        row = rows[0]
        output = directory / 'terra'
        output.mkdir(exist_ok=False)
        write(output / 'claimed.json', {'attempt': 1, 'actor_calls': 0})
        result = {'task_id': task['task_id'], 'attempt': 1, 'valid': False, 'reward': None, 'actor_calls': 0}
        client = None
        try:
            with wall_guard(deadline):
                if self.client_factory is None:
                    from openai import OpenAI
                    timeout = min(180, max(.001, deadline - time.time())) if deadline is not None else 180
                    client = OpenAI(api_key=self.key, max_retries=0, timeout=timeout)
                else:
                    client = self.client_factory()
                need(client.max_retries == 0, 'Terra SDK retries must be disabled')
                result['reward'] = self.judge.judge(client, self.budget, row['task'], row['actions'], row['screenshot'], output)
                result['valid'] = True
        except BaseException as error:
            self.budget.halt('Native Terra attempt failed; no automatic judge or actor retries')
            result.update(error_type=type(error).__name__, error=str(error))
        finally:
            if client is not None:
                client.close()
            write(output / 'result.json', result)
        return result
