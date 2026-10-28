"""Pinned WebArbiter WebArena-Lite inventory and offline evaluator contracts.

Preparation is not a benchmark run. This module does not contact model providers,
deploy websites, log in, or reset them. Native evaluator definitions are loaded
without importing the viewer application or its provider clients.
"""
from __future__ import annotations

import __future__
from abc import ABC, abstractmethod
import ast
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sys
import time
from types import ModuleType, SimpleNamespace
from urllib.parse import urlsplit

from openwebrl.webprm_source import ARBITER_REVISION, source_manifest

TASK_FILE = 'viewer/config_files/wa/test_webarena_lite.raw.json'
DOMAINS = ('shopping', 'shopping_admin', 'reddit', 'gitlab', 'map')
SOURCE_HASHES = {
    TASK_FILE: '895c2116182ce211ca531ed685ace82430a1aed5f800b2507a2ee30769ceecb7',
    'viewer/src/evaluation_harness/evaluators.py': '672a7452144705861497ec64b244ef03ead14ee105e998ca807bbcf18c4f0428',
    'viewer/src/evaluation_harness/base_evaluators.py': 'df31569c77411469469afc51a256a6557dc8963f1860d91180d81eff5cc2a9cc',
    'viewer/src/evaluation_harness/url_service.py': '7286a4d39e67d8da82fe90aea932e0bd7ce849cf09183bf8582ae77312cb83ce',
    'viewer/src/evaluation_harness/helper_functions.py': '499907cc05c864d163b9520bb0c086a71ffc6a44e403e67e75b51ab104150080',
    'viewer/src/browser_env/actions.py': 'e653a5177224fe12dae63b26cdbe0170689c9d3fe507b77fbfadc821320865ca',
    'viewer/src/browser_env/utils.py': '91eaf912d41d24400bce5977cd4b122025d74934552fbb40b3218ec08fb6e31d',
}
PLACEHOLDER = re.compile(r'__[A-Z][A-Z_]*__')
MATCH_METHODS = {'exact_match', 'must_include', 'must_exclude', 'required_values',
                 'one_of', 'fuzzy_match'}


class EvaluationNotReady(ValueError):
    """An unavailable dependency must not become an evaluated task failure."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def verify_source(root):
    manifest = source_manifest(root, SOURCE_HASHES, ARBITER_REVISION)
    if manifest['files'] != SOURCE_HASHES:
        raise ValueError('Pinned WebArbiter files were modified')
    return manifest


def validate_endpoints(endpoints):
    """Require explicit origins; credentials and URL query secrets are forbidden."""
    result = {}
    for name, url in endpoints.items():
        if name not in (*DOMAINS, 'wikipedia', 'homepage'):
            raise ValueError(f'Unknown endpoint name: {name}')
        if not isinstance(url, str):
            raise ValueError('Endpoint must be a URL string')
        parsed = urlsplit(url)
        if (parsed.scheme not in {'http', 'https'} or not parsed.hostname or
                parsed.username or parsed.password or parsed.query or parsed.fragment or
                any(character.isspace() for character in url)):
            raise ValueError(f'Invalid or credential-bearing endpoint: {name}')
        try:
            parsed.port
        except ValueError as error:
            raise ValueError(f'Invalid endpoint port: {name}') from error
        result[name] = url.rstrip('/')
    return result


def substitute_endpoints(task, endpoints):
    """Replace only named release placeholders, without implicit localhost defaults."""
    endpoints = validate_endpoints(endpoints)
    required = {token.strip('_').lower() for token in PLACEHOLDER.findall(json.dumps(task))}
    missing = required - endpoints.keys()
    if missing:
        raise EvaluationNotReady(f'Missing explicit endpoints: {sorted(missing)}')

    def replace(value):
        if isinstance(value, str):
            return PLACEHOLDER.sub(lambda match: endpoints[match[0].strip('_').lower()], value)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    return replace(task)


def evaluation_requirements(task):
    section = task['eval']
    refs = section.get('reference_answers') or {}
    html = section.get('program_html') or []
    methods = set(refs)
    for target in html:
        methods.update(target['required_contents'])
    return {
        'eval_types': section['eval_types'],
        'paid_fuzzy_judge_may_be_required': 'fuzzy_match' in methods,
        'native_judge_model': 'gpt-4-1106-preview' if 'fuzzy_match' in methods else None,
        'nltk_tokenizer_required': bool(methods & {'must_include', 'must_exclude'}),
        'helper_functions_required': any(
            target.get(field, '').startswith('func:') for target in html
            for field in ('url', 'locator')),
        'browser_page_required': 'program_html' in section['eval_types'],
        'storage_state_required': bool(task.get('storage_state')),
        'released_require_reset': task.get('require_reset'),
    }


def prepare_manifest(root, endpoints=None):
    source = verify_source(root)
    tasks = json.loads((Path(root) / TASK_FILE).read_text())
    ids = [task['task_id'] for task in tasks]
    if len(tasks) != 165 or len(set(ids)) != 165:
        raise ValueError('Unexpected released WebArena-Lite inventory')
    single = [task for task in tasks if len(task['sites']) == 1]
    if len(single) != 155 or set(task['sites'][0] for task in single) != set(DOMAINS):
        raise ValueError('Unexpected single-site task group')
    # Task order is fixed before any website or actor outcome is observed.
    pilot = [min((task for task in single if task['sites'] == [domain]),
                 key=lambda task: task['task_id']) for domain in DOMAINS]
    groups = {'released_full_165': tasks, 'released_single_site_155': single,
              'mechanical_first_per_domain_5': pilot}
    explicit = validate_endpoints(endpoints or {})
    configured = []
    for task in tasks:
        try:
            configured.append(substitute_endpoints(task, explicit))
        except EvaluationNotReady:
            pass
    return {
        'schema_version': 1, 'status': 'prepared_not_executed',
        'benchmark': 'WebArena-Lite', 'source': source,
        'paper_table_5_exact_task_membership_verified': False,
        'membership_caveat': 'The released 165-task file and its 155 single-site subset '
                             'are explicit candidate cohorts; neither is asserted to '
                             'be the exact paper Table 5 task membership.',
        'task_groups': {name: {'count': len(values), 'task_ids': [t['task_id'] for t in values],
                              'tasks_sha256': digest(values)} for name, values in groups.items()},
        'single_site_counts': dict(sorted(Counter(t['sites'][0] for t in single).items())),
        'cross_site_count': len(tasks) - len(single),
        'pilot_selection': 'Lowest released task_id per single-site domain, before outcomes; '
                           'five tasks, one task episode per condition, no redraws.',
        'pilot_tasks': [{'task_id': task['task_id'], 'domain': task['sites'][0],
                         'released_task_sha256': digest(task),
                         'requirements': evaluation_requirements(task)} for task in pilot],
        'evaluation_inventory': {
            'fuzzy_judge_tasks': sum(evaluation_requirements(t)['paid_fuzzy_judge_may_be_required'] for t in tasks),
            'helper_function_tasks': sum(evaluation_requirements(t)['helper_functions_required'] for t in tasks),
        },
        'endpoints': explicit, 'endpoint_substituted_task_count': len(configured),
        'website_readiness_verified': False, 'reset_and_login_verified': False,
        'actor_critic_protocol_modified': False,
        'model_provider_calls': 0, 'gpu_allocation_seconds': 0,
        'blockers': ['Website deployment, login and reset have not been verified.',
                     'Full native evaluator requires tokenizer data, budgeted fuzzy-judge '
                     'integration, and the 14 helper-function task contracts.',
                     'Paper Table 5 task membership and complete inference protocol '
                     'must be reconciled before claiming exact reproduction.'],
    }, groups, configured


def validate_evaluator_input(task, trajectory, page, *, termination=None, allow_fuzzy=False,
                             deployment_helpers=None):
    """Validate before native permissive fallbacks could turn bad input into zero/one."""
    if not isinstance(task, dict) or not isinstance(task.get('intent'), str):
        raise EvaluationNotReady('Task must contain a string intent')
    section = task.get('eval', {})
    types = section.get('eval_types')
    if (not isinstance(types, list) or not types or len(types) != len(set(types)) or
            set(types) - {'string_match', 'url_match', 'program_html'}):
        raise EvaluationNotReady('Empty, duplicate or unsupported native evaluation types')
    if termination not in {None, 'stop', 'turn_limit'}:
        raise EvaluationNotReady('Native scoring requires normal stop or turn_limit termination')
    if not isinstance(trajectory, list) or not trajectory:
        raise EvaluationNotReady('Trajectory must contain the actual terminal state or answer')
    for index, entry in enumerate(trajectory):
        if not isinstance(entry, dict):
            raise EvaluationNotReady('Trajectory entries must be mappings')
        if index % 2 == 0:
            if not isinstance(entry.get('observation'), dict) or not isinstance(entry.get('info'), dict):
                raise EvaluationNotReady('State requires observation and info mappings')
        elif 'action_type' not in entry:
            raise EvaluationNotReady('Action requires action_type')
    actual_answer = (len(trajectory) % 2 == 0 and
                     trajectory[-1].get('action_type') == 'send_msg_to_user' and
                     isinstance(trajectory[-1].get('answer'), str))
    actual_final_state = len(trajectory) % 2 == 1 and termination in {'stop', 'turn_limit'}
    # URL and HTML evaluators read the actual final page and do not require a
    # fabricated send_msg_to_user. String scoring still requires an actual final
    # answer, preventing native fallback to a stale earlier message.
    if 'string_match' in types and not actual_answer:
        raise EvaluationNotReady('String evaluation requires an actual terminal answer')
    if not actual_answer and not actual_final_state:
        raise EvaluationNotReady('Need the actual terminal answer or final state at declared normal termination')
    if 'string_match' in types:
        refs = section.get('reference_answers')
        if not isinstance(refs, dict) or not refs or set(refs) - MATCH_METHODS:
            raise EvaluationNotReady('Missing or unsupported reference-answer contract')
        _validate_match_values(refs)
    if 'url_match' in types:
        if not isinstance(section.get('reference_url'), str) or not section['reference_url'].strip():
            raise EvaluationNotReady('URL evaluation requires nonempty reference_url')
        references = section['reference_url'].split(' |OR| ')
        if any(not reference.strip() for reference in references):
            raise EvaluationNotReady('URL alternatives cannot contain an empty match')
        if section.get('url_note', 'EXACT') not in {'EXACT', 'GOLD in PRED'}:
            raise EvaluationNotReady('Unsupported URL matching rule')
    if any(kind in types for kind in ('url_match', 'program_html')):
        try:
            terminal_url = getattr(page, 'url', None)
        except Exception as error:
            raise EvaluationNotReady('Terminal page URL is unavailable') from error
        if not isinstance(terminal_url, str) or not terminal_url:
            raise EvaluationNotReady('Native URL/browser evaluator needs the terminal page')
    if 'program_html' in types:
        targets = section.get('program_html')
        if not isinstance(targets, list) or not targets:
            raise EvaluationNotReady('HTML evaluation requires nonempty target list')
        for target in targets:
            if (not isinstance(target, dict) or not isinstance(target.get('url'), str) or
                    not target['url'] or not isinstance(target.get('locator'), str)):
                raise EvaluationNotReady('Malformed HTML target')
            contents = target.get('required_contents')
            if (not isinstance(contents, dict) or len(contents) != 1 or
                    set(contents) - (MATCH_METHODS - {'one_of'})):
                raise EvaluationNotReady('HTML target must have one supported content requirement')
            _validate_match_values(contents)
            if target['url'] != 'last' and not target['url'].startswith(('http://', 'https://', 'func:')):
                raise EvaluationNotReady('Unsupported HTML target URL')
    if PLACEHOLDER.search(json.dumps(task)):
        raise EvaluationNotReady('Unresolved endpoint placeholder')
    requirements = evaluation_requirements(task)
    if requirements['paid_fuzzy_judge_may_be_required'] and not allow_fuzzy:
        raise EvaluationNotReady('Native GPT-4 fuzzy judge is disabled until budgeted integration')
    if requirements['helper_functions_required'] and deployment_helpers is None:
        raise EvaluationNotReady('Native helper-function evaluation needs deployment-specific validation')
    if 'program_html' in types:
        needed = deployment_helpers.validate_task(task) if deployment_helpers is not None else set()
        for target in section['program_html']:
            if target['url'] != 'last':
                needed.add('goto')
            if not target['locator'].strip():
                needed.add('content')
            elif target['locator'].startswith(('document.', '[...document.', 'lambda:')):
                needed.add('evaluate')
            elif target['locator'].startswith('func:') and deployment_helpers is not None:
                pass  # Explicit adapter validated the exact released helper call.
            else:
                raise EvaluationNotReady('Unsupported HTML locator contract')
        for name in needed:
            try:
                method = getattr(page, name, None)
            except Exception as error:
                raise EvaluationNotReady(f'Browser page capability unavailable: {name}') from error
            if not callable(method) or inspect.iscoroutinefunction(method):
                raise EvaluationNotReady(f'Native evaluator requires synchronous page.{name}')


class _TrackedPage:
    """Record browser failures even when native HTML code catches Exception.

    Recognized JavaScript errors on a live Playwright browser retain the native
    locator semantics. Unknown errors, closed pages and transport failures are
    unavailable evaluations, including when native code substitutes empty text.
    """

    def __init__(self, page):
        self.page = page
        self.failures = []
        self.native_javascript_errors = []

    def _check_liveness(self, *, required=False):
        try:
            closed = getattr(self.page, 'is_closed', None)
            browser = getattr(getattr(self.page, 'context', None), 'browser', None)
            connected = getattr(browser, 'is_connected', None)
            if callable(closed) and closed():
                raise EvaluationNotReady('Browser page is closed')
            if callable(connected) and not connected():
                raise EvaluationNotReady('Browser transport is disconnected')
            return callable(closed) and callable(connected) if required else True
        except EvaluationNotReady:
            raise
        except Exception as error:
            raise EvaluationNotReady('Browser liveness is unavailable') from error

    @property
    def url(self):
        self._check_liveness()
        try:
            return self.page.url
        except Exception as error:
            self.failures.append(('url', error))
            raise EvaluationNotReady('Browser page URL is unavailable') from error

    def _call(self, name, *args, **kwargs):
        try:
            self._check_liveness()
            value = getattr(self.page, name)(*args, **kwargs)
            if inspect.isawaitable(value):
                if inspect.iscoroutine(value):
                    value.close()
                raise EvaluationNotReady(f'Native evaluator cannot await page.{name}')
            return value
        except Exception as error:
            # Playwright distinguishes JavaScript evaluation errors by a JS
            # error name in its Error message. Do not classify generic Python
            # AttributeError/TypeError or unknown errors as a missing DOM node.
            javascript = (name == 'evaluate' and
                          type(error).__module__.startswith('playwright.') and
                          type(error).__name__ == 'Error' and
                          re.search(r'\b(?:TypeError|ReferenceError|SyntaxError|EvalError|RangeError|URIError):',
                                    str(error)) is not None)
            try:
                javascript = javascript and self._check_liveness(required=True)
            except EvaluationNotReady:
                javascript = False
            if javascript:
                self.native_javascript_errors.append(str(error))
            else:
                self.failures.append((name, error))
            raise

    def evaluate(self, *args, **kwargs):
        return self._call('evaluate', *args, **kwargs)

    def content(self, *args, **kwargs):
        return self._call('content', *args, **kwargs)

    def goto(self, *args, **kwargs):
        return self._call('goto', *args, **kwargs)

    def verify(self):
        if self.failures:
            operation, error = self.failures[0]
            raise EvaluationNotReady(f'Browser {operation} failed; native result discarded') from error
        self._check_liveness()


def _validate_match_values(requirements):
    for method, value in requirements.items():
        if method == 'exact_match' or (method == 'fuzzy_match' and value == 'N/A'):
            if not isinstance(value, str):
                raise EvaluationNotReady('Exact/native N/A requirement must be a string')
        elif not isinstance(value, list) or not value or any(not isinstance(v, str) or not v for v in value):
            raise EvaluationNotReady('List requirements must contain nonempty strings')


def _compile_definitions(path, namespace, names=None):
    tree = ast.parse(Path(path).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and (names is None or node.name in names)]
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec',
                   flags=__future__.annotations.compiler_flag)
    exec(code, namespace)


NATIVE_EVALUATOR_PROTOCOL = 'native_evaluator_v1'
ABSTENTION_EVALUATOR_PROTOCOL = 'native_with_explicit_na_abstention_guard_v1'


def evaluator_protocol(value=NATIVE_EVALUATOR_PROTOCOL):
    if value not in (NATIVE_EVALUATOR_PROTOCOL, ABSTENTION_EVALUATOR_PROTOCOL):
        raise EvaluationNotReady('Unknown explicit evaluator scoring protocol')
    return value


def load_native_evaluator(root, *, fuzzy_judge=None, scoring_protocol=NATIVE_EVALUATOR_PROTOCOL,
                          deployment_helpers=None, observation_recorder=None, observation_replay=None):
    """Load unchanged native class bodies; paid dependencies default to disabled.

    Fuzzy scoring requires an explicitly enabled injected budgeted adapter.
    Native deployment helpers require an explicitly configured adapter. This loader never silently
    substitutes a different judge or initializes a provider client.
    """
    scoring_protocol = evaluator_protocol(scoring_protocol)
    manifest = verify_source(root)
    if observation_recorder is not None or observation_replay is not None:
        from .webprm_evaluator_transcript import EvaluationRecorder, ReplayCursor
        if (observation_recorder is not None and not isinstance(observation_recorder, EvaluationRecorder)
                or observation_replay is not None and not isinstance(observation_replay, ReplayCursor)
                or observation_recorder is not None and observation_replay is not None):
            raise EvaluationNotReady('Explicit, mutually exclusive native observation record/replay required')
    if deployment_helpers is not None:
        from openwebrl.webprm_webarena_helpers import NativeHelpers
        if not isinstance(deployment_helpers, NativeHelpers):
            raise EvaluationNotReady('Explicit pinned native helper adapter required')
    if fuzzy_judge is not None and (getattr(fuzzy_judge, 'enabled', False) is not True or
            not callable(getattr(fuzzy_judge, 'fuzzy_match', None)) or
            not callable(getattr(fuzzy_judge, 'ua_match', None)) or
            type(getattr(fuzzy_judge, 'provider_calls', None)) is not int or fuzzy_judge.provider_calls < 0):
        raise EvaluationNotReady('Explicit enabled and budgeted native fuzzy adapter required')
    prefix = Path(root) / 'viewer/src/evaluation_harness'
    package_name = '_webprm_webarena_native_' + digest(manifest)[:16]
    package = ModuleType(package_name)
    package.__path__ = []
    url_module = ModuleType(package_name + '.url_service')
    _compile_definitions(prefix / 'url_service.py', url_module.__dict__)
    # The config has already been explicitly substituted and validated; no
    # environment or native localhost defaults are read by this URL service.
    url_module._url_service_instance = url_module.URLService()
    url_module._url_service_instance._mappings_cache = {}
    sys.modules[package_name] = package
    sys.modules[url_module.__name__] = url_module
    module = ModuleType(package_name + '.evaluators')
    module.__package__ = package_name
    sys.modules[module.__name__] = module

    def blocked_judge(*args, **kwargs):
        raise EvaluationNotReady('Native paid judge is disabled')

    def native_tokenize(text):
        try:
            from nltk.tokenize import word_tokenize
            return word_tokenize(text)
        except (ImportError, LookupError) as error:
            raise EvaluationNotReady('Native NLTK tokenizer/data unavailable; no replacement tokenizer') from error

    namespace = module.__dict__
    namespace.update(ABC=ABC, abstractmethod=abstractmethod, dataclass=dataclass,
                     Path=Path, json=json, time=time, word_tokenize=native_tokenize,
                     llm_fuzzy_match=fuzzy_judge.fuzzy_match if fuzzy_judge else blocked_judge,
                     llm_ua_match=fuzzy_judge.ua_match if fuzzy_judge else blocked_judge)
    if deployment_helpers is not None:
        def helper_expression(expression):
            invoke = lambda: deployment_helpers.evaluate_expression(expression)
            if observation_recorder is not None or observation_replay is not None:
                from .webprm_evaluator_transcript import helper_input
                inputs = helper_input(expression)
                if observation_recorder is not None:
                    return observation_recorder.call('helper', inputs, invoke, helper=True)
                from .webprm_webarena_helpers import SHOPPING
                boundary = inputs['function'] == SHOPPING
                return observation_replay.consume('helper', inputs,
                    None if boundary else invoke, shopping_boundary=boundary)
            return invoke()
        namespace['eval'] = helper_expression
    _compile_definitions(prefix / 'base_evaluators.py', namespace)
    _compile_definitions(prefix / 'evaluators.py', namespace,
                         {'Evaluator', 'NumericEvaluator', 'StringEvaluator', 'URLExactEvaluator',
                          'HTMLContentExactEvaluator'})

    def native_evaluate(task, trajectory, page=None, *, termination=None):
        before_calls = fuzzy_judge.provider_calls if fuzzy_judge else 0
        validate_evaluator_input(task, trajectory, page, termination=termination,
                                 allow_fuzzy=fuzzy_judge is not None,
                                 deployment_helpers=deployment_helpers)
        needs_page = bool(set(task['eval']['eval_types']) & {'url_match', 'program_html'})
        tracked_page = _TrackedPage(page) if needs_page else None
        if tracked_page is not None:
            tracked_page.verify()
        classes = {'string_match': namespace['StringEvaluator'],
                   'url_match': namespace['URLExactEvaluator'],
                   'program_html': namespace['HTMLContentExactEvaluator']}
        details = []
        native_details = []
        amendments = []
        score = 1.0
        native_score = 1.0
        for kind in task['eval']['eval_types']:
            evaluator = classes[kind]()
            if hasattr(evaluator, '_url_service'):
                evaluator._url_service = url_module._url_service_instance
            try:
                with (deployment_helpers.bind(task, tracked_page)
                      if deployment_helpers is not None else nullcontext()):
                    value = float(evaluator(trajectory, task, tracked_page))
            except Exception:
                if tracked_page is not None:
                    tracked_page.verify()
                raise
            if tracked_page is not None:
                tracked_page.verify()
            if not math.isfinite(value) or value not in (0.0, 1.0):
                raise EvaluationNotReady('Native evaluator returned a non-binary result')
            native_score *= value
            native_details.append({'eval_type': kind, 'score': value})
            # Scope this prospective rule to answer-string grading. The same
            # fuzzy helper also grades DOM content and must remain unchanged.
            if scoring_protocol == ABSTENTION_EVALUATOR_PROTOCOL and kind == 'string_match':
                reference = task['eval']['reference_answers'].get('fuzzy_match')
                native_string = namespace['StringEvaluator']
                answer = native_string.get_last_action(trajectory).get('answer', '')
                prediction = native_string.clean_answer(answer)
                ordinary_reference = (isinstance(reference, list) and bool(reference)
                    and all(native_string.clean_answer(item) != 'n/a' for item in reference))
                if prediction == 'n/a' and ordinary_reference:
                    amendments.append(dict(rule='bare_na_is_abstention_for_ordinary_fuzzy_answer',
                        eval_type=kind, native_score=value, effective_score=0.0,
                        normalized_prediction=prediction,
                        native_reference_route='ordinary_fuzzy_list',
                        reference_contains_explicit_na=False))
                    value = 0.0
            score *= value
            details.append({'eval_type': kind, 'score': value})
        result = {'valid': True, 'score': score, 'success': score > 0,
                'evaluator_details': details, 'source': manifest,
                'model_provider_calls': fuzzy_judge.provider_calls - before_calls if fuzzy_judge else 0,
                'fuzzy_judge_manifest': getattr(fuzzy_judge, 'manifest', None),
                'termination': termination or 'terminal_answer',
                'native_javascript_errors': tracked_page.native_javascript_errors if tracked_page else []}
        if scoring_protocol != NATIVE_EVALUATOR_PROTOCOL:
            result.update(evaluator_protocol=scoring_protocol,
                native_score=native_score, native_success=native_score > 0,
                native_evaluator_details=native_details, scoring_amendments=amendments)
        if deployment_helpers is not None:
            result['native_helper_manifest'] = deployment_helpers.manifest
        return result

    def evaluate(task, trajectory, page=None, *, termination=None):
        if observation_recorder is None:
            return native_evaluate(task, trajectory, page, termination=termination)
        from .webprm_evaluator_transcript import ObservedPage
        observation_recorder.start(dict(task_sha256=digest(task), trajectory_sha256=digest(trajectory),
            source_sha256=digest(manifest), scoring_protocol=scoring_protocol,
            termination=termination,
            helper_manifest=deployment_helpers.manifest if deployment_helpers else None))
        try:
            observed = ObservedPage(page, observation_recorder) if page is not None else None
            result = native_evaluate(task, trajectory, observed, termination=termination)
            metadata = observation_recorder.finish(outcome=result)
            return dict(result, observation_transcript=metadata)
        except Exception as error:
            if not observation_recorder.finished:
                observation_recorder.finish(error=error)
            raise

    return SimpleNamespace(evaluate=evaluate, manifest=manifest)
