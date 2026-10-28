"""Provider-free episode boundaries; partial preparation, not a pilot runner.

No browser, model, reset or native evaluator is loaded implicitly. Callers inject
reviewed dependencies and still own budgets, browser supervision and site reset.
Policy/selector callbacks receive only the allowlisted actor task view. Trajectory
records must describe completed execution, never proposed or invented actions.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Callable

from .webprm_webarena import EvaluationNotReady, digest, substitute_endpoints


def _copy(value):
    """Detach JSON evidence; reject non-finite values and non-JSON live objects."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _blocked(name):
    def blocked(*args, **kwargs):
        raise EvaluationNotReady(f'{name} callback is not configured')
    return blocked


@dataclass(frozen=True)
class TaskEnvelope:
    execution_id: str
    released_task_id: int
    raw_config_sha256: str
    configured_config_sha256: str
    actor_view: Mapping
    requires_storage_state: bool
    _configured_json: str = field(repr=False)

    @classmethod
    def from_release(cls, released, endpoints):
        raw = _copy(released)
        task_id = raw.get('task_id')
        if type(task_id) is not int or task_id < 0:
            raise ValueError('Released task_id must be a nonnegative integer')
        if not isinstance(raw.get('intent'), str) or not raw['intent'].strip():
            raise ValueError('Task requires an intent')
        if not isinstance(raw.get('start_url'), str) or not raw['start_url']:
            raise ValueError('Task requires a start URL')
        if (not isinstance(raw.get('sites'), list) or not raw['sites'] or
                any(not isinstance(site, str) for site in raw['sites'])):
            raise ValueError('Task requires site names')
        configured = substitute_endpoints(raw, endpoints)
        # Never pass eval, intent_template, reference annotations, storage state
        # or future newly added configuration fields to an actor callback.
        visible = {key: configured[key] for key in ('intent', 'start_url', 'sites')}
        return cls(f'webarena_lite_{task_id:03d}', task_id, digest(raw),
                   digest(configured), _freeze(visible), bool(raw.get('storage_state')),
                   json.dumps(configured, ensure_ascii=False, allow_nan=False))

    def native_config(self):
        """Return a detached evaluator-only config; never an actor input."""
        return json.loads(self._configured_json)


@dataclass(frozen=True)
class EpisodeHooks:
    dependency: Callable = _blocked('dependency/authentication/reset')
    policy: Callable = _blocked('policy')
    selector: Callable = _blocked('selector')
    snapshot: Callable = _blocked('terminal snapshot')
    evaluate: Callable = _blocked('native evaluator')

    def propose(self, task, observation):
        return self.policy(task.actor_view, observation)

    def select(self, task, observation, candidates):
        return self.selector(task.actor_view, observation, candidates)


def open_authenticated_browser(task, *, hooks=None, browser_factory=None,
                               task_entrypoint=None, action_mapping=None,
                               storage_state=None, task_options=None,
                               browser_options=None):
    """Construct, but do not reset, an explicitly injected BrowserEnv.

    dependency(task) must attest a fresh reset_id and, for authenticated tasks,
    authenticated=True. Storage state stays private and is passed to Playwright
    before its context is created. This receipt is not itself a live login test.
    """
    hooks = hooks or EpisodeHooks()
    proof = hooks.dependency(task)
    if (not isinstance(proof, dict) or proof.get('ready') is not True or
            not isinstance(proof.get('reset_id'), str) or not proof['reset_id']):
        raise EvaluationNotReady('Explicit ready/reset evidence is required')
    if task.requires_storage_state and (
            storage_state is None or proof.get('authenticated') is not True):
        raise EvaluationNotReady('Authenticated storage state is required')
    if browser_factory is None or task_entrypoint is None or action_mapping is None:
        raise EvaluationNotReady('Browser factory, task and action mapping must be injected')
    task_kwargs = dict(task_options or {})
    if set(task_kwargs) & {'start_url', 'goal'}:
        raise ValueError('Task options cannot override the frozen intent or URL')
    task_kwargs.update(start_url=task.actor_view['start_url'], goal=task.actor_view['intent'])
    options = dict(browser_options or {})
    if set(options) & {'task_entrypoint', 'task_kwargs', 'action_mapping'}:
        raise ValueError('Browser options cannot override task/execution boundaries')
    context = dict(options.pop('pw_context_kwargs', {}))
    if 'storage_state' in context:
        raise ValueError('Supply storage state through the explicit private argument')
    if storage_state is not None:
        if not isinstance(storage_state, dict):
            raise TypeError('Supply loaded private Playwright storage-state JSON')
        if (not isinstance(storage_state.get('cookies'), list) or
                not isinstance(storage_state.get('origins'), list)):
            raise ValueError('Malformed Playwright storage state')
        context['storage_state'] = _copy(storage_state)
    return browser_factory(task_entrypoint=task_entrypoint, task_kwargs=task_kwargs,
                           action_mapping=action_mapping, pw_context_kwargs=context,
                           **options)


class NativeTrajectory:
    """Record actual JSON states/actions in native order, without stop synthesis.

    Callers must normalize BrowserGym observation/info dictionaries into JSON
    evidence. Raw NumPy screenshot arrays and other live objects are rejected;
    save screenshots separately and record their artifact references as needed.
    """
    def __init__(self, observation, info):
        self._entries = [self._state(observation, info)]
        self._termination = None

    @staticmethod
    def _state(observation, info):
        if not isinstance(observation, dict) or not isinstance(info, dict):
            raise TypeError('Native state requires observation and info dictionaries')
        return _copy({'observation': observation, 'info': info})

    @property
    def termination(self):
        return self._termination

    @property
    def entries(self):
        return _copy(self._entries)

    def _require_open(self):
        if self._termination is not None:
            raise ValueError('Trajectory is already terminated')

    def record_executed_action(self, action, observation, info):
        self._require_open()
        if (not isinstance(action, dict) or not isinstance(action.get('action_type'), str)
                or not action['action_type'] or action['action_type'] == 'send_msg_to_user'):
            raise ValueError('Nonterminal executed action required')
        # Validate both before committing either; no incomplete half-step remains.
        action, state = _copy(action), self._state(observation, info)
        self._entries.extend((action, state))

    def record_executed_terminal(self, action):
        self._require_open()
        if (not isinstance(action, dict) or action.get('action_type') != 'send_msg_to_user'
                or not isinstance(action.get('answer'), str)):
            raise ValueError('Actual send_msg_to_user action with literal answer required')
        self._entries.append(_copy(action))
        self._termination = 'stop'

    def finish_at_state(self, termination):
        self._require_open()
        if termination not in {'stop', 'turn_limit'}:
            raise ValueError('Only normal stop or turn_limit permits final-state scoring')
        self._termination = termination


def _output_directory(value):
    path = Path(value).resolve()
    roots = (Path('/gpfs/projects/krishna/zixianma/OpenWebRL'),
             Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'), Path('/tmp'))
    if not any(root in path.parents for root in roots):
        raise ValueError('Evidence directory must be below an approved writable root')
    missing = []
    ancestor = path
    while not ancestor.exists():
        missing.append(ancestor)
        ancestor = ancestor.parent
    path.mkdir(parents=True, exist_ok=False)
    # Persist each newly created directory entry, including intermediate parents.
    for created in reversed(missing):
        _sync_directory(created.parent)
    return path


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)


def score_and_close(task, trajectory, env, output, *, hooks=None, before_close=None):
    """Save terminal evidence, score a live page, save verdict, then close.

    snapshot(page) returns {url: str, screenshot: nonempty PNG bytes}. evaluate is
    the native adapter's evaluate callable. Missing dependencies and exceptions
    yield valid=False/score=None, never a benchmark failure. A close error remains
    distinct from a completed native verdict. Output must be a fresh directory.
    """
    hooks = hooks or EpisodeHooks()
    result = dict(task_id=task.execution_id, released_task_id=task.released_task_id,
                  raw_config_sha256=task.raw_config_sha256,
                  configured_config_sha256=task.configured_config_sha256,
                  valid=False, success=False, score=None, cleanup_status='pending')
    folder = None
    stage = 'evidence_directory'
    try:
        folder = _output_directory(output)
        stage = 'trajectory'
        if trajectory.termination is None:
            raise EvaluationNotReady('Episode has no declared normal termination')
        entries = trajectory.entries
        _write(folder / 'trajectory.json', entries)
        stage = 'terminal_snapshot'
        page = env.page
        snapshot = hooks.snapshot(page)
        if (not isinstance(snapshot, dict) or not isinstance(snapshot.get('url'), str)
                or not snapshot['url'] or not isinstance(snapshot.get('screenshot'), bytes)
                or not snapshot['screenshot'].startswith(b'\x89PNG\r\n\x1a\n')):
            raise EvaluationNotReady('Terminal snapshot requires actual URL and PNG bytes')
        if snapshot['url'] != page.url:
            raise EvaluationNotReady('Terminal snapshot URL differs from live page')
        image = snapshot['screenshot']
        with (folder / 'terminal.png').open('xb') as stream:
            stream.write(image)
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(folder)
        result['terminal'] = dict(url=snapshot['url'], screenshot='terminal.png',
                                  screenshot_sha256=hashlib.sha256(image).hexdigest(),
                                  termination=trajectory.termination)
        _write(folder / 'terminal.json', result['terminal'])
        stage = 'native_evaluation'
        verdict = hooks.evaluate(task.native_config(), entries, page,
                                 termination=trajectory.termination)
        if (not isinstance(verdict, dict) or verdict.get('valid') is not True or
                type(verdict.get('score')) not in (int, float) or
                verdict['score'] not in (0, 1) or type(verdict.get('success')) is not bool or
                verdict['success'] != bool(verdict['score'])):
            raise EvaluationNotReady('Native evaluator returned an invalid verdict')
        verdict = _copy(verdict)
        _write(folder / 'native-verdict.json', verdict)
        result.update(valid=True, score=verdict['score'], success=verdict['success'])
        if 'observation_transcript' in verdict:
            result['observation_transcript'] = _copy(verdict['observation_transcript'])
        if 'evaluator_protocol' in verdict:
            # Prospective amended cohorts preserve the unmodified native result
            # alongside their explicit effective scoring rule in both artifacts.
            for key in ('evaluator_protocol', 'native_score', 'native_success',
                        'native_evaluator_details', 'scoring_amendments'):
                result[key] = _copy(verdict[key])
    except Exception as error:
        result['error'] = dict(stage=stage, type=type(error).__name__, message=str(error))
    finally:
        # Preserve outcome before potentially blocking cleanup. The caller still
        # needs the existing external episode watchdog for hangs in any callback.
        try:
            if folder is not None:
                _write(folder / 'outcome-before-close.json', result)
            if before_close is not None:
                before_close(_copy(result))
        finally:
            try:
                env.close()
                result['cleanup_status'] = 'completed'
            except Exception as error:
                result['cleanup_status'] = 'error'
                result['cleanup_error'] = dict(type=type(error).__name__, message=str(error))
        if folder is not None:
            _write(folder / 'result.json', result)
    return result
