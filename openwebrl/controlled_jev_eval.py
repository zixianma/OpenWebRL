"""Jev native DOM actor on the matched local browser and canonical OM2W judge.

Imports perform no model/browser work. Native Jev emits no thoughts: empty
thoughts accompany the actual dispatched operations, including its DONE marker.
Only DONE is COMPLETED. BLOCKED and control limits receive canonical zero with
no judge request; infrastructure validity remains a separate field.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from types import SimpleNamespace

from openwebrl import jev_eval as legacy
from openwebrl.controlled_browser_server import validate_browser_manifest
from openwebrl.controlled_sft_eval import ChatResponse, ScopedBudget
from openwebrl.controlled_sft_worker import AttemptJournal, CanonicalOM2WReward
from openwebrl.controlled_jev_state import JevState

MODEL = 'jev-1.13.0'
TEXT_MODEL = 'gpt-4.1-mini-2025-04-14'
JUDGE_MODEL = 'o4-mini-2025-04-16'


class OperationLimit(RuntimeError):
    pass


def private_exception_details(exc, *, environ=None, max_chars=8192):
    """Bound private diagnostics after redacting credentials, never before."""
    values = os.environ if environ is None else environ
    secrets = [str(value) for name, value in values.items() if value and (
        re.search(r'(?:^|_)(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?|AUTHORIZATION|COOKIE)$', name.upper())
        or name.upper() in {'TYPESAFE_API_KEY', 'TEXT_MODEL_API_KEY', 'JUDGE_API_KEY'})]
    def redact(text):
        for value in sorted(set(secrets), key=len, reverse=True):
            text = text.replace(value, '[REDACTED]')
        text = re.sub(r'(?i)(https?://)[^/\s:@]+:[^/\s@]+@', r'\1[REDACTED]@', text)
        text = re.sub(r'(?i)(\b(?:authorization\s*[:=]\s*(?:(?:bearer|basic)\s+)?|'
            r'(?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password|cookie)\s*[:=]\s*))'
            r'[^\s\"\'&;,}]+', r'\1[REDACTED]', text)
        return re.sub(r'\bsk-[A-Za-z0-9_-]{12,}', '[REDACTED]', text)
    message = redact(str(exc))
    trace = redact(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return dict(error_type=type(exc).__name__, error_message=message[:min(2048, max_chars)],
        traceback=trace[:max_chars], error_message_truncated=len(message)>min(2048, max_chars),
        traceback_truncated=len(trace)>max_chars)


class JevBudget(ScopedBudget):
    def reserve(self, kind, count=1, **context):
        try:
            return super().reserve(kind, count, **context)
        except Exception as exc:
            self.halt('reservation_rejected', kind=kind, error_type=type(exc).__name__)
            raise


class Transport:
    """One Jev/helper attempt; canonical reward owns its four judge attempts."""
    def __init__(self, budget, journal, *, client=None):
        if client is None:
            import httpx
            client = httpx.Client(timeout=120, trust_env=False, follow_redirects=False)
        self.client, self.budget, self.journal = client, budget, journal
        self.counts = {'jev': 0, 'text': 0, 'judge': 0}

    def request(self, provider, url, key, body, *, metered=True, timeout=120):
        kind = {'jev': 'jev_requests', 'text': 'text_http_attempts', 'judge': 'om2w_judge_http_attempts'}[provider]
        reservation = self.budget.reserve(kind, request=body) if metered else None
        self.counts[provider] += 1
        self.journal(provider+'_reserved', dict(request=body, reservation=reservation))
        settled = False
        try:
            response = self.client.post(url, json=body, headers={'Authorization': 'Bearer '+key}, timeout=timeout)
            if response.status_code >= 400:
                evidence = self.journal('http_error', dict(provider=provider, status=response.status_code,
                    body=response.text.replace(key, '[REDACTED]')[:4096]))
                if provider != 'judge' or response.status_code in (400, 401, 403, 404, 413):
                    self.budget.halt('provider_http_error', provider=provider, evidence_path=evidence)
                response.raise_for_status()
            raw = response.json()
            if metered and provider == 'text':
                self.budget.settle(reservation, raw.get('usage'))
                settled = True
            expected = {'jev': MODEL, 'text': TEXT_MODEL, 'judge': JUDGE_MODEL}[provider]
            if raw.get('model') != expected:
                self.budget.halt('provider_model_mismatch', provider=provider, returned_model=raw.get('model'))
                raise ValueError('Pinned response model changed')
            self.journal(provider+'_received', dict(response=raw))
            return raw
        except BaseException as exc:
            evidence = self.journal('transport_failed', dict(provider=provider, **private_exception_details(exc)))
            if metered and provider == 'text' and not settled:
                self.budget.finalize_unknown(reservation, evidence)
            if provider != 'judge': self.budget.halt('provider_or_transport_failure', provider=provider, evidence_path=evidence)
            raise

    def post(self, url, key, body):
        body = deepcopy(body)
        if url == 'https://api.typesafe.ai/v1/systemone':
            if body.get('model') != MODEL: raise ValueError('Unexpected Jev model')
            return self.request('jev', url, key, body)
        if url != 'https://api.openai.com/v1/chat/completions' or body.get('model') != TEXT_MODEL:
            raise ValueError('Unexpected text helper endpoint/model')
        body.pop('reasoning', None); body.pop('thinking', None)
        body.update(temperature=.6, top_p=.95, max_tokens=1024)
        return self.request('text', url, key, body)

    async def judge(self, **request):
        import httpx
        self.counts['judge'] += 1
        key = os.environ['JUDGE_API_KEY']
        async with httpx.AsyncClient(timeout=110, trust_env=False, follow_redirects=False) as client:
            response = await client.post('https://api.openai.com/v1/chat/completions', json=request,
                headers={'Authorization': 'Bearer '+key})
            if response.status_code >= 400:
                evidence = self.journal('http_error', dict(provider='judge', status=response.status_code,
                    body=response.text.replace(key, '[REDACTED]')[:4096]))
                if response.status_code in (400, 401, 403, 404, 413):
                    self.budget.halt('provider_http_error', provider='judge', evidence_path=evidence)
                response.raise_for_status()
            raw = response.json()
        if raw.get('model') != JUDGE_MODEL:
            self.budget.halt('judge_model_mismatch')
            raise ValueError('Pinned judge model changed')
        return ChatResponse(raw)


class LocalSession(legacy.BrowserSession):
    def open(self):
        from playwright.sync_api import sync_playwright
        from openwebrl.env.browser_runtime import browser_process_environment
        manifest = validate_browser_manifest(self.config['browser_manifest'])
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True, executable_path=manifest['binary_path'],
            args=manifest['launch_extra_args'], env=browser_process_environment(manifest['launch_extra_args']))
        self.context = self.browser.new_context(viewport={'width': 1280, 'height': 720},
            device_scale_factor=1, is_mobile=False, locale='en-US', timezone_id='UTC')
        self.context.set_default_timeout(30000)
        return self


@contextmanager
def native_agent(session, transport, journal, operations):
    os.environ.setdefault('BROWSER_HARNESS_HOME', str(session.root/'harness-scratch'))
    import jev_ultrafast.agent as am
    import jev_ultrafast.browser as bm
    import jev_ultrafast.model as mm
    from openwebrl.jev_harness import revision
    original = am.Browser, am.MAX_STEPS, bm.cdp, bm.browser_operation, mm.post_json

    class Browser(bm.Browser):
        def __init__(self, url):
            self.page = session.context.new_page()
            self.session = session.context.new_cdp_session(self.page)
            self.target = True
            self.call('Emulation.setDeviceMetricsOverride', width=1280, height=720, deviceScaleFactor=1, mobile=False)
            self.call('Emulation.setFocusEmulationEnabled', enabled=True)
            self.page.goto(url, timeout=60000, wait_until='domcontentloaded')

        def close(self):
            if self.target:
                self.page.close(); self.target = None

    def cdp(method, session_id=None, **params):
        if session_id is None: raise ValueError('CDP must belong to the current local task')
        return session_id.send(method, params)

    def operation(request):
        if request.get('operation') != 'act': return original[3](request)
        if len(operations) >= 30: raise OperationLimit('Thirty browser operations reached')
        entry = dict(number=len(operations)+1, action=deepcopy(request['action']), text=request.get('text'), status='dispatched')
        operations.append(entry)
        journal('operation_dispatched', deepcopy(entry))
        try:
            result = original[3](request)
            entry.update(status='returned', returned=deepcopy(result))
            return result
        except Exception as exc:
            entry.update(status='raised', **private_exception_details(exc))
            raise
        finally:
            journal('operation_finished', deepcopy(entry))

    am.Browser, am.MAX_STEPS, bm.cdp, bm.browser_operation, mm.post_json = Browser, 30, cdp, operation, transport.post
    try:
        # Explicitly preserve the previously validated actionable native DOM
        # target filter; do not synthesize SFT proposals or use selector prompts.
        with revision(bm, 'actionable-v2'): yield am.Agent
    finally:
        am.Browser, am.MAX_STEPS, bm.cdp, bm.browser_operation, mm.post_json = original


NATIVE_JUDGE_ACTIONS = ('click', 'fill', 'select', 'scroll', 'wait', 'done')


def native_judge_tools():
    """Input schema only: preserve native actions for the canonical parser.

    Jev's fill and DOM-select operations are not SFT click/write calls. The
    unchanged canonical parser needs these names registered to retain them.
    This registry never controls execution and is never installed globally.
    """
    return [dict(type='function', function=dict(name=name,
        description='Recorded Jev native '+name+' operation; arguments preserve the execution receipt.',
        parameters=dict(type='object',additionalProperties=True))) for name in NATIVE_JUDGE_ACTIONS]


def native_judge_reward_module(reward):
    """Private namespace for the unchanged canonical reward/parser functions."""
    namespace = dict(vars(reward))
    namespace['_TOOLS_INFO'] = native_judge_tools()
    return SimpleNamespace(**namespace)


def canonical_messages(operations, terminal):
    messages = []
    for row in operations:
        action = row['action']
        if action.get('kind') not in NATIVE_JUDGE_ACTIONS[:-1]:
            raise ValueError('Unknown native browser operation; refuse to invent a canonical action')
        # Preserve every saved target, typed value and observed outcome. This
        # is an execution receipt, not invented model reasoning or coordinates.
        params = deepcopy(row)
        content = '<tool_call>\n'+json.dumps(dict(name=action['kind'], arguments=params),
            ensure_ascii=False,allow_nan=False)+'\n</tool_call>'
        messages.append(dict(role='assistant', content=content))
    if terminal == 'done':
        messages.append(dict(role='assistant', content='<tool_call>\n{"name":"done","arguments":{"response":""}}\n</tool_call>'))
    return messages


async def canonical_score(task, operations, terminal, screenshot, transport, budget, journal, config):
    from openwebrl.eval import reward_online_mind2web as reward
    sample = reward.Sample(index=0, prompt=task['intent'], metadata=dict(task,
        messages=canonical_messages(operations, terminal), full_image_list=['data:image/png;base64,'+base64.b64encode(screenshot).decode()]))
    sample.status = reward.Sample.Status.COMPLETED if terminal == 'done' else reward.Sample.Status.ABORTED
    judge = CanonicalOM2WReward(native_judge_reward_module(reward), source_sha256=config['canonical_reward_sha256'],
        chat_create=transport.judge, budget=budget, journal=journal, task_id=task['task_id'])
    args = SimpleNamespace(hf_checkpoint=config['parser_checkpoint'], judge_api_model=JUDGE_MODEL,
        judge_api_mode='served', judge_timeout_secs=120, log_judge_output=False)
    score = await judge(args, sample)
    return score, sample.metadata['reward'], str(sample.status)


def run_episode(claim, plan, state):
    directory = Path(claim['artifact_directory'])
    config = plan['worker_config']
    task = next(t for t in plan['tasks'] if t['task_id'] == claim['task_id'])
    journal, budget = AttemptJournal(directory/'events'), JevBudget(state, claim)
    transport = Transport(budget, journal)
    session = LocalSession(directory, dict(browser='local', browser_manifest=config['browser_manifest']))
    started = time.monotonic()
    operations, agent, image, snapshot = [], None, None, None
    terminal, error, cleanup, browser_reservation = 'error', None, [], None
    journal('episode_started', dict(task=task, claim={k:v for k,v in claim.items() if k!='task'},
        actor=MODEL, helper=TEXT_MODEL, harness_revision='actionable-v2', actor_images=False,
        canonical_thoughts='empty: native Jev does not generate reasoning text', max_operations=30, max_decisions=60))
    try:
        browser_reservation = budget.reserve('browser_sessions')
        with legacy.deadline(1800):
            session.open()
            with native_agent(session, transport, journal, operations) as Agent:
                agent = Agent(task['start_url'], task['intent'], screenshots=True)
                legacy.save_snapshot(directory, 0, agent.snapshot())
                for decision in range(1, 61):
                    if len(operations) >= 30:
                        terminal = 'action_limit'; break
                    snapshot = agent.command('tick')
                    legacy.save_snapshot(directory, decision, snapshot)
                    if snapshot['status'] in ('done', 'blocked'):
                        terminal = snapshot['status']; break
                else: terminal = 'decision_limit'
    except (legacy.EpisodeTimeout, OperationLimit) as exc:
        terminal = 'episode_deadline' if isinstance(exc, legacy.EpisodeTimeout) else 'action_limit'
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc) in {
                'Text helper returned no valid field value; nothing typed.',
                'Invalid TypeSafe response; no action executed.'}:
            # Keep native guards and do not repair/reselect the output. A
            # rejected actor choice or field value is a non-completed failure,
            # not an unavailable browser/site or a cohort-wide provider outage.
            terminal = ('typing_value_unavailable' if str(exc).startswith('Text helper')
                        else 'native_choice_invalid')
            journal('native_actor_failure', dict(reason=terminal, **private_exception_details(exc)))
        else:
            error = type(exc).__name__
            journal('episode_error', private_exception_details(exc))
    finally:
        if agent is not None:
            snapshot = agent.snapshot()
            try:
                if legacy.playwright_dispatcher_dead(agent.browser.page): raise RuntimeError('Dead Playwright dispatcher')
                with legacy.deadline(15): image = agent.browser.page.screenshot(type='png', timeout=10000)
            except Exception as exc:
                journal('terminal_capture_failed', private_exception_details(exc))
        cleanup = session.close()
        if browser_reservation is not None and not cleanup:
            budget.settle(browser_reservation, {'closed': True})
    image_artifact = journal.blob('final.png', image) if image else None
    journal('native_final_state', dict(terminal=terminal, operations=operations, native_snapshot=snapshot,
        final_screenshot=image_artifact, cleanup_errors=cleanup))
    valid = bool(image) and error is None and not cleanup
    score, verdict, status = None, None, None
    if valid:
        score, verdict, status = asyncio.run(canonical_score(task, operations, terminal, image,
            transport, budget, journal, config))
        valid = score is not None
    result = dict(task_id=task['task_id'], condition='L06', attempt_id=claim['attempt_id'],
        category=claim['category'], valid=valid, score=score, reward=score, terminal=terminal,
        native_status=status, actor_error=error, judge=verdict, browser_closed=not cleanup,
        cleanup_errors=cleanup, final_screenshot=image_artifact, actions=len(operations)+(terminal=='done'),
        decisions=transport.counts['jev'], api_attempts=transport.counts,
        elapsed_seconds=time.monotonic()-started, halt_required=bool(state.snapshot()['halt']) or bool(cleanup),
        judge_source_sha256=config['canonical_reward_sha256'], actor_model=MODEL, typing_model=TEXT_MODEL)
    raw = json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()+b'\n'
    path = directory/'result.json'
    with path.open('xb') as handle: handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    transport.client.close()
    return dict(result, result_path=str(path), result_sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--worker-id', required=True)
    args = parser.parse_args()
    plan = json.loads((args.root/'plan.json').read_text())
    # Frozen native Jev and browser-harness package copies are last in the
    # search path; existing evaluation dependencies retain their pinned origin.
    sys.path.append(plan['vendor_path'])
    state = JevState(args.root)
    while claim := state.claim(args.worker_id):
        result = run_episode(claim, plan, state)
        state.finish(claim['attempt_id'], result)
        if result['halt_required']:
            state.halt('Jev worker reported provider/cleanup failure', attempt_id=claim['attempt_id'])
            raise RuntimeError('Cohort halted with preserved evidence')


if __name__ == '__main__': main()
