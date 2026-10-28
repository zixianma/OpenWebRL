"""Bounded, explicit native evaluator observations and deterministic replay.

Replay verifies the pinned evaluator and Reddit/GitLab helper computations from
their recorded browser inputs. Shopping replay trusts the recorded helper return
boundary; it does not verify the REST authentication or order-query computation.
The helper recorder never inspects account files, HTTP authorization headers or
authentication responses. Page observations can contain website data and are
stored privately. Provider verdict audit remains a separate input.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .webprm_webarena import EvaluationNotReady, digest

TRANSCRIPT_PROTOCOL = 'native_evaluator_observation_transcript_v1'
TRANSCRIPT_CONFIGURATION = dict(protocol=TRANSCRIPT_PROTOCOL, max_events=512,
    max_bytes=16 * 2**20, max_value_bytes=4 * 2**20,
    helper_replay='released_reddit_gitlab_bodies_shopping_return_boundary')


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def validate_transcript_configuration(value):
    if value != TRANSCRIPT_CONFIGURATION:
        raise EvaluationNotReady('Unknown explicit native evaluator transcript configuration')


def _clone(value):
    return json.loads(encoded(value))


def _navigation_value(value):
    if value is None:
        return None
    # The pinned native evaluator ignores goto's return value. Save a bounded
    # projection of Playwright Response, never its body, headers or request.
    return dict(kind='navigation_response', url=value.url, status=value.status)


class EvaluationRecorder:
    def __init__(self, path, configuration=TRANSCRIPT_CONFIGURATION):
        validate_transcript_configuration(configuration)
        self.path = Path(path) if path is not None else None
        self.document = None
        self.stack = []
        self.count = self.bytes = 0
        self.poisoned = False
        self.finished = False

    def start(self, binding):
        if self.document is not None:
            raise EvaluationNotReady('Evaluator transcript recorder is single-use')
        self.document = dict(configuration=deepcopy(TRANSCRIPT_CONFIGURATION),
            binding=_clone(binding), events=[], status='recording')
        self.stack = [self.document['events']]
        self.bytes = len(encoded(self.document))

    def _bounded(self, value):
        try:
            raw = encoded(value)
            if len(raw) > TRANSCRIPT_CONFIGURATION['max_value_bytes']:
                raise ValueError('value size')
            if self.bytes + len(raw) > TRANSCRIPT_CONFIGURATION['max_bytes'] - 65536:
                raise ValueError('total size')
            self.bytes += len(raw)
            return json.loads(raw)
        except (TypeError, ValueError, OverflowError):
            self.poisoned = True
            raise EvaluationNotReady('Native evaluator transcript value is not bounded JSON evidence') from None

    def call(self, operation, inputs, function, *, navigation=False, helper=False):
        if self.document is None or self.finished or self.poisoned:
            raise EvaluationNotReady('Native evaluator transcript is unavailable or incomplete')
        self.count += 1
        if self.count > TRANSCRIPT_CONFIGURATION['max_events'] or len(self.stack) > 4:
            self.poisoned = True
            raise EvaluationNotReady('Native evaluator transcript event/depth cap reached')
        event = dict(operation=operation, inputs=self._bounded(inputs), events=[])
        self.bytes += len(encoded(event))
        self.stack[-1].append(event); self.stack.append(event['events'])
        try:
            value = function()
            event['value'] = self._bounded(_navigation_value(value) if navigation else value)
            return value
        except Exception as error:
            # Helper failures can include deployment details. Preserve class
            # identity without copying private HTTP/authentication error text.
            error_value = dict(module=type(error).__module__, type=type(error).__name__,
                message='Native helper operation failed' if helper else str(error))
            try:
                event['error'] = self._bounded(error_value)
            except EvaluationNotReady:
                event['error'] = dict(module=__name__, type='TranscriptLimit', message='Evidence unavailable')
            raise
        finally:
            self.stack.pop()

    def finish(self, *, outcome=None, error=None):
        if self.finished:
            raise EvaluationNotReady('Native evaluator transcript already closed')
        if outcome is not None and self.poisoned:
            raise EvaluationNotReady('Incomplete native evaluator transcript cannot support a verdict')
        self.document.update(status='completed' if outcome is not None else 'failed',
            event_count=self.count, outcome_sha256=digest(outcome) if outcome is not None else None,
            error_type=type(error).__name__ if error is not None else None)
        raw = encoded(self.document) + b'\n'
        if len(raw) > TRANSCRIPT_CONFIGURATION['max_bytes']:
            raise EvaluationNotReady('Native evaluator transcript exceeds its final byte cap')
        if self.path is not None:
            from .webprm_webarena_sites import canonical_runtime_path
            canonical_runtime_path(str(self.path))
            # Canonical compact bytes keep file SHA identical to replay's digest.
            import os
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'wb') as handle:
                handle.write(raw); handle.flush(); os.fsync(handle.fileno())
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(directory)
            finally: os.close(directory)
        self.finished = True
        return dict(protocol=TRANSCRIPT_PROTOCOL, sha256=hashlib.sha256(raw).hexdigest(),
            bytes=len(raw), event_count=self.count,
            replay_scope=TRANSCRIPT_CONFIGURATION['helper_replay'])


class ObservedPage:
    def __init__(self, page, recorder):
        self.page, self.recorder = page, recorder

    @property
    def context(self):
        return self.page.context

    @property
    def is_closed(self):
        return getattr(self.page, 'is_closed', None)

    @property
    def url(self):
        return self.recorder.call('page.url', {}, lambda: self.page.url)

    def _call(self, name, args, kwargs):
        return self.recorder.call('page.' + name, dict(args=args, kwargs=kwargs),
            lambda: getattr(self.page, name)(*args, **kwargs), navigation=name == 'goto')

    def evaluate(self, *args, **kwargs): return self._call('evaluate', args, kwargs)
    def content(self, *args, **kwargs): return self._call('content', args, kwargs)
    def goto(self, *args, **kwargs): return self._call('goto', args, kwargs)


def helper_input(expression):
    from .webprm_webarena_helpers import HELPER_SHA256
    return dict(expression=expression, function=expression.split('(', 1)[0],
                native_source_sha256=HELPER_SHA256)


class ReplayCursor:
    def __init__(self, events):
        self.stack = [[events, 0]]

    def consume(self, operation, inputs, function=None, *, shopping_boundary=False):
        events, index = self.stack[-1]
        if index >= len(events):
            raise EvaluationNotReady('Evaluator replay attempted an unrecorded operation')
        event = events[index]; self.stack[-1][1] += 1
        if event.get('operation') != operation or event.get('inputs') != _clone(inputs):
            raise EvaluationNotReady('Evaluator replay operation/input differs from the saved observation')
        if 'error' in event:
            error = event['error']
            if (operation == 'page.evaluate' and error.get('module', '').startswith('playwright.')
                    and error.get('type') == 'Error'):
                # Recreate only the recognized JS error identity used by the
                # pinned tracking wrapper. No exception module is imported.
                kind = type('Error', (Exception,), {'__module__': error['module']})
                raise kind(error['message'])
            raise EvaluationNotReady('Recorded native operation was unavailable')
        if 'value' not in event:
            raise EvaluationNotReady('Evaluator replay event lacks its result')
        children = event.get('events')
        if not isinstance(children, list):
            raise EvaluationNotReady('Evaluator replay event lacks its nested observations')
        if function is not None:
            self.stack.append([children, 0])
            try:
                value = function()
                self.assert_complete()
            finally:
                self.stack.pop()
            if _clone(value) != event['value']:
                raise EvaluationNotReady('Pinned native helper recomputation differs from its saved result')
        elif children:
            raise EvaluationNotReady('Replay cannot skip nested observations at a helper boundary')
        if shopping_boundary and operation != 'helper':
            raise EvaluationNotReady('Shopping boundary replay requires a helper operation')
        return deepcopy(event['value'])

    def assert_complete(self):
        events, index = self.stack[-1]
        if index != len(events):
            raise EvaluationNotReady('Evaluator replay did not consume every saved observation')


class ReplayPage:
    def __init__(self, cursor):
        self.cursor = cursor
        self.context = SimpleNamespace(browser=SimpleNamespace(is_connected=lambda: True))

    def is_closed(self): return False

    @property
    def url(self): return self.cursor.consume('page.url', {})

    def _call(self, name, args, kwargs):
        value = self.cursor.consume('page.' + name, dict(args=args, kwargs=kwargs))
        if name == 'goto' and value is not None:
            if value.get('kind') != 'navigation_response':
                raise EvaluationNotReady('Recorded navigation result projection differs')
            return SimpleNamespace(url=value['url'], status=value['status'])
        return value

    def evaluate(self, *args, **kwargs): return self._call('evaluate', args, kwargs)
    def content(self, *args, **kwargs): return self._call('content', args, kwargs)
    def goto(self, *args, **kwargs): return self._call('goto', args, kwargs)


def _validate_events(events, depth=1):
    if not isinstance(events, list) or depth > 4:
        raise EvaluationNotReady('Evaluator transcript event tree exceeds its schema/depth cap')
    count = 0
    for event in events:
        if (not isinstance(event, dict) or
                set(event) not in ({'operation','inputs','events','value'}, {'operation','inputs','events','error'})
                or event['operation'] not in ('page.url','page.goto','page.content','page.evaluate','helper')
                or not isinstance(event['inputs'], dict)):
            raise EvaluationNotReady('Evaluator transcript has an unrecognized event')
        for key in ('inputs', 'value' if 'value' in event else 'error'):
            if len(encoded(event[key])) > TRANSCRIPT_CONFIGURATION['max_value_bytes']:
                raise EvaluationNotReady('Evaluator transcript value exceeds its byte cap')
        children = event['events']
        count += 1 + (_validate_events(children, depth+1) if children else 0)
        if count > TRANSCRIPT_CONFIGURATION['max_events']:
            raise EvaluationNotReady('Evaluator transcript exceeds its event cap')
    return count


def replay_native_evaluation(root, task, trajectory, document, expected_verdict, *, fuzzy_judge=None):
    """Replay observed computation, never assert independent website/provider truth."""
    from .webprm_webarena import load_native_evaluator, NATIVE_EVALUATOR_PROTOCOL
    from .webprm_webarena_helpers import NativeHelpers, SHOPPING, HELPER_SHA256
    validate_transcript_configuration(document.get('configuration'))
    raw = encoded(document) + b'\n'
    metadata = expected_verdict.get('observation_transcript', {})
    if (document.get('status') != 'completed' or len(raw) > TRANSCRIPT_CONFIGURATION['max_bytes']
            or metadata.get('protocol') != TRANSCRIPT_PROTOCOL
            or metadata.get('sha256') != hashlib.sha256(raw).hexdigest()):
        raise EvaluationNotReady('A complete hash-bound evaluator transcript is required')
    count = _validate_events(document.get('events'))
    if document.get('event_count') != count or metadata.get('event_count') != count or metadata.get('bytes') != len(raw):
        raise EvaluationNotReady('Evaluator transcript event/byte accounting differs')
    binding = document['binding']
    expected = {key: value for key, value in expected_verdict.items() if key != 'observation_transcript'}
    if (binding.get('task_sha256') != digest(task) or binding.get('trajectory_sha256') != digest(trajectory)
            or document.get('outcome_sha256') != digest(expected)):
        raise EvaluationNotReady('Evaluator transcript task/trajectory/verdict binding differs')
    cursor = ReplayCursor(document['events'])
    helpers = None
    if binding.get('helper_manifest') is not None:
        helpers = NativeHelpers(root)
        manifest = binding['helper_manifest']
        if manifest.get('source_sha256') != HELPER_SHA256:
            raise EvaluationNotReady('Native helper replay source identity differs')
        if SHOPPING in manifest['functions']:
            def blocked_shopping():
                raise EvaluationNotReady('Offline replay must never call Shopping HTTP')
            helpers.functions[SHOPPING] = blocked_shopping
        helpers.manifest = deepcopy(manifest)
    evaluator = load_native_evaluator(root, fuzzy_judge=fuzzy_judge,
        scoring_protocol=binding.get('scoring_protocol', NATIVE_EVALUATOR_PROTOCOL),
        deployment_helpers=helpers, observation_replay=cursor)
    if digest(evaluator.manifest) != binding.get('source_sha256'):
        raise EvaluationNotReady('Pinned native evaluator source differs from recorded computation')
    result = evaluator.evaluate(task, trajectory, ReplayPage(cursor), termination=binding['termination'])
    cursor.assert_complete()
    if digest(result) != document['outcome_sha256']:
        raise EvaluationNotReady('Native evaluator replay verdict differs')
    return dict(verified=True, transcript_sha256=metadata['sha256'], verdict_sha256=digest(expected),
        replay_scope=TRANSCRIPT_CONFIGURATION['helper_replay'],
        website_truth_independently_verified=False, provider_verdict_independently_verified=False,
        result=result)
