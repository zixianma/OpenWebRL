"""Native WebArena fuzzy prompts and verdicts with explicit model identity.

No provider client, pricing defaults, availability probe or automatic fallback
is supplied. A disclosed model replacement requires explicit configuration.
Merely loading this module or its judge cannot call a model.
The caller's shared budget must durably reserve the complete request before
dispatch and retain reservations when a request's billable outcome is unknown.
"""
from __future__ import annotations

import __future__
import ast
from dataclasses import dataclass
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import Callable

from .webprm_webarena import EvaluationNotReady, verify_source

MODEL = 'gpt-4-1106-preview'
NATIVE_PROTOCOL = 'native_webarena_fuzzy_v1'
REPLACEMENT_MODEL = 'gpt-4o-2024-05-13'
REPLACEMENT_PROTOCOL = 'webarena_fuzzy_gpt4o_20240513_v1'
OUTPUT_TOKENS = 768
HELPER_FILE = 'viewer/src/evaluation_harness/helper_functions.py'
PROVIDER_FILE = 'viewer/src/llms/providers/openai_utils.py'
PROVIDER_SHA256 = 'f41e06f801c8b1497b7c2c53dbf8f5a0fa1e3fd5da96e590a6ac5b98a5ce504f'
JOURNAL_ROOTS = (Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'), Path('/tmp'))


def validate_fuzzy_configuration(config):
    """Validate explicit execution choices without constructing a transport.

    This does not establish model availability or authorize compute/API use.
    The owning frozen plan must separately supply those approvals and receipts.
    """
    if config.get('fuzzy_enabled') is not True:
        raise EvaluationNotReady('Fuzzy evaluation is not explicitly enabled')
    model = config.get('fuzzy_model')
    protocol = config.get('fuzzy_protocol')
    approved = config.get('fuzzy_model_change_approved', False)
    if type(approved) is not bool:
        raise EvaluationNotReady('Fuzzy model-change approval must be an explicit bool')
    if model == MODEL:
        if protocol != NATIVE_PROTOCOL:
            raise EvaluationNotReady('Original fuzzy model requires its native protocol identity')
    elif model == REPLACEMENT_MODEL:
        if protocol != REPLACEMENT_PROTOCOL or not approved:
            raise EvaluationNotReady('Fuzzy model substitution requires its explicit protocol and approval')
    else:
        raise EvaluationNotReady('Unsupported frozen fuzzy model')
    for key in ('fuzzy_max_input_tokens', 'fuzzy_max_calls'):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise EvaluationNotReady('Explicit positive fuzzy cap required: ' + key)
    return dict(model=model, protocol=protocol, model_changed=model != MODEL,
                model_change_approved=approved)


@dataclass(frozen=True)
class SingleAttemptTransport:
    """A reviewed transport that sends exactly one request, with SDK retries off.

    This adapter invokes ``send`` once. The eventual transport implementation
    must itself disable retries; this module does not instantiate an SDK.
    ``send`` receives the exact native helper kwargs, including context_length0,
    and returns one chat-completion response dictionary including usage/model.
    """
    send: Callable
    automatic_retries: int = 0

    def __post_init__(self):
        if not callable(self.send) or inspect.iscoroutinefunction(self.send):
            raise EvaluationNotReady('A synchronous single-attempt transport is required')
        if type(self.automatic_retries) is not int or self.automatic_retries != 0:
            raise EvaluationNotReady('Automatic fuzzy-judge retries are forbidden')


@dataclass(frozen=True)
class BrokerBudgetedTransport:
    """Use the reviewed Unix-socket broker as the sole shared budget owner.

    The broker factory markers select this trust boundary; they are provenance
    assertions, not cryptographic attestation. The host broker must enforce the
    authorized price/token/request caps and durably reserve/settle before its
    response. A matching settled receipt is additionally checked for each call.
    """
    client: Callable
    config: dict
    journal: object

    def __post_init__(self):
        if (not callable(self.client) or inspect.iscoroutinefunction(self.client)
                or getattr(self.client, '_webprm_broker_client', None) is not True
                or getattr(self.client, '_webprm_budget_owner', None) != 'webprm_worker.completion'):
            raise EvaluationNotReady('Explicit reviewed budget-owning broker client required')

    def send(self, request, *, deadline):
        payload = dict(request)
        if payload.pop('context_length', None) != 0:
            raise EvaluationNotReady('Native broker context_length must remain zero')
        # Pinned native provider also omits context_length from the HTTP payload.
        # Bound this request without mutating the registered cell configuration.
        config = dict(self.config, deadline=min(self.config['deadline'], deadline))
        return self.client(config, payload, self.journal, purpose='native_fuzzy')


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _journal_path(value):
    path = Path(value)
    resolved = path.resolve()
    if not path.is_absolute() or str(path) != str(resolved) or path.is_symlink():
        raise EvaluationNotReady('Fuzzy journal must be an absolute canonical nonsymlink path')
    if not any(root in resolved.parents for root in JOURNAL_ROOTS):
        raise EvaluationNotReady('Fuzzy journal is outside approved private runtime roots')
    return path


def _append(path, value):
    path = _journal_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _journal_path(path)  # recheck after creation, including all parent symlinks
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        data = (_canonical(value) + '\n').encode()
        while data:
            written = os.write(descriptor, data)
            if not written:
                raise OSError('Fuzzy journal write made no progress')
            data = data[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class NativeFuzzyJudge:
    """Budgeted exact native fuzzy/UA callbacks; disabled unless fully configured.

    ``budget.reserve(record)`` must atomically enforce the caller's shared
    monetary/request/token caps and durably return a reservation ID. Its record
    contains purpose, exact request, token bounds and request hash. It must not
    perform provider calls. ``budget.settle(id, usage_record)`` accounts observed
    usage and must fail/halt the shared budget on an overrun. Unknown calls keep
    their reservation; there is intentionally no refund/release callback here.

    ``count_input_tokens(request)`` must count or conservatively bound the exact
    serialized request including message overhead. Prices and authorized global
    caps belong to the injected shared budget; none are inferred here.

    Alternatively, ``BrokerBudgetedTransport`` owns accounting on the host, and
    ``budget`` must be absent. ``provider_calls`` counts transport invocations;
    a failed invocation may not have reached a provider and remains uncertain.
    """
    def __init__(self, source_root, *, enabled=False, transport=None, budget=None,
                 journal_path=None, count_input_tokens=None, max_input_tokens=None,
                 max_calls=None, deadline=None, model=MODEL, protocol=None,
                 model_change_approved=False):
        self.enabled = False
        self.provider_calls = 0
        self._halted = False
        self._lock = threading.Lock()
        self._kind = None
        self._transport = transport
        self._budget = budget
        self._count = count_input_tokens
        self._journal = None
        self._max_input = max_input_tokens
        self._max_calls = max_calls
        self._deadline = deadline
        self._model = model
        protocol = NATIVE_PROTOCOL if protocol is None and model == MODEL else protocol
        if type(model_change_approved) is not bool:
            raise EvaluationNotReady('Fuzzy model-change approval must be an explicit bool')
        if model != MODEL and (model != REPLACEMENT_MODEL or protocol != REPLACEMENT_PROTOCOL
                               or not model_change_approved):
            raise EvaluationNotReady('Fuzzy model substitution requires its explicit protocol and approval')
        if model == MODEL and protocol != NATIVE_PROTOCOL:
            raise EvaluationNotReady('Original fuzzy model requires its native protocol identity')
        source = verify_source(source_root)
        provider_hash = hashlib.sha256((Path(source_root) / PROVIDER_FILE).read_bytes()).hexdigest()
        if provider_hash != PROVIDER_SHA256:
            raise EvaluationNotReady('Pinned native provider reference was modified')
        self.manifest = dict(source=source, model=model, native_model=MODEL, protocol=protocol,
                             model_changed=model != MODEL, model_change_approved=model_change_approved,
                             temperature=0, top_p=1.0,
                             max_tokens=OUTPUT_TOKENS, context_length=0,
                             automatic_retries=0, alternate_model_allowed=False,
                             automatic_model_fallback=False,
                             provider_reference=dict(path=PROVIDER_FILE, sha256=provider_hash),
                             call_count_semantics='Transport invocations; failed calls may not reach provider.',
                             adaptation=('Native prompt and verdict code retained; lazy provider import replaced by injected transport.'
                                + (' Explicitly approved model substitution; not the original evaluator model.' if model != MODEL else '')))
        if type(enabled) is not bool:
            raise EvaluationNotReady('Fuzzy enable flag must be explicit bool')
        if enabled:
            if sys.flags.optimize:
                raise EvaluationNotReady('Native fuzzy assertions require Python without optimization')
            if not isinstance(transport, (SingleAttemptTransport, BrokerBudgetedTransport)):
                raise EvaluationNotReady('Explicit single-attempt fuzzy transport required')
            if isinstance(transport, BrokerBudgetedTransport):
                if budget is not None:
                    raise EvaluationNotReady('Broker must be the sole shared budget owner')
            else:
                for name in ('reserve', 'settle'):
                    fn = getattr(budget, name, None)
                    if not callable(fn) or inspect.iscoroutinefunction(fn):
                        raise EvaluationNotReady('Explicit synchronous shared budget required')
            if not callable(count_input_tokens) or inspect.iscoroutinefunction(count_input_tokens):
                raise EvaluationNotReady('Explicit serialized-request token counter required')
            if type(max_input_tokens) is not int or max_input_tokens <= 0:
                raise EvaluationNotReady('Explicit positive fuzzy input-token cap required')
            if type(max_calls) is not int or max_calls <= 0:
                raise EvaluationNotReady('Explicit positive fuzzy call cap required')
            if (type(deadline) not in (int, float) or not math.isfinite(deadline)
                    or deadline <= time.time()):
                raise EvaluationNotReady('Explicit unexpired fuzzy deadline required')
            if journal_path is None:
                raise EvaluationNotReady('Explicit durable fuzzy journal required')
            self._journal = _journal_path(journal_path)
            self.enabled = True
        self.manifest.update(enabled=self.enabled, max_input_tokens=max_input_tokens,
                             max_calls=max_calls, deadline=deadline,
                             budget_owner=('webprm_worker.completion' if isinstance(transport, BrokerBudgetedTransport)
                                           else 'injected_shared_budget'))
        self._native = self._load(Path(source_root) / HELPER_FILE)

    def _load(self, path):
        tree = ast.parse(path.read_text())
        names = {'llm_fuzzy_match', 'llm_ua_match'}
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in nodes} != names:
            raise EvaluationNotReady('Pinned native fuzzy helper definitions missing')
        for node in nodes:
            imports = [item for item in ast.walk(node) if isinstance(item, (ast.Import, ast.ImportFrom))]
            if (len(imports) != 1 or not isinstance(imports[0], ast.ImportFrom)
                    or imports[0].module != 'webarbiter.src.llms.providers.openai_utils'
                    or [(n.name, n.asname) for n in imports[0].names] != [('generate_from_openai_chat_completion', None)]
                    or imports[0] not in node.body):
                raise EvaluationNotReady('Unexpected native provider import contract')
            node.body.remove(imports[0])
        namespace = {'logger': SimpleNamespace(info=lambda *_args, **_kwargs: None),
                     'generate_from_openai_chat_completion': self._dispatch}
        code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec',
                       flags=__future__.annotations.compiler_flag)
        exec(code, namespace)
        return namespace

    def _event(self, event, **data):
        record = dict(data)
        record.update(event=event, purpose='native_fuzzy', native_kind=self._kind,
                      at_epoch=time.time(), pid=os.getpid())
        _append(self._journal, record)

    def _broker_receipt(self, response, request):
        receipt = response.get('_webprm_broker')
        payload = {key: value for key, value in request.items() if key != 'context_length'}
        # Broker wire hashes use ASCII-escaped JSON; the native audit uses UTF-8.
        payload_sha = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True,
                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        expected = dict(schema_version=1, budget_owner='webprm_worker.completion',
                        model=self._model, purpose='native_fuzzy', local=False,
                        payload_sha256=payload_sha, reservation_state='settled',
                        usage=response['usage'])
        if (not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in expected.items())
                or type(receipt.get('schema_version')) is not int
                or type(receipt.get('local')) is not bool
                or type(receipt.get('reservation_id')) not in (int, str)
                or receipt.get('reservation_id') == ''
                or any(not isinstance(receipt.get(key), str) or len(receipt[key]) != 64
                       or any(character not in '0123456789abcdef' for character in receipt[key])
                       for key in ('identity_sha256', 'journal_sha256', 'budget_ledger_sha256', 'policy_sha256'))):
            self._halted = True
            raise EvaluationNotReady('Native fuzzy broker budget receipt missing or mismatched')
        return receipt

    def _dispatch(self, **request):
        if not self.enabled or self._halted:
            raise EvaluationNotReady('Native fuzzy judge is disabled or halted')
        expected = dict(model=MODEL, temperature=0, max_tokens=OUTPUT_TOKENS,
                        top_p=1.0, context_length=0)
        if set(request) != {*expected, 'messages'} or any(request[k] != v for k, v in expected.items()):
            raise EvaluationNotReady('Native fuzzy model/request contract changed')
        if time.time() >= self._deadline or self.provider_calls >= self._max_calls:
            raise EvaluationNotReady('Native fuzzy deadline or call cap reached')
        native_request_sha = _sha(request)
        request = json.loads(_canonical(request))  # isolate callbacks from native inputs
        request['model'] = self._model  # explicit manifest-bound change; never automatic fallback
        tokens = self._count(json.loads(_canonical(request)))
        if type(tokens) is not int or not 0 < tokens <= self._max_input:
            raise EvaluationNotReady('Native fuzzy request exceeds its input-token cap')
        request_sha = _sha(request)
        reservation_record = dict(purpose='native_fuzzy', native_kind=self._kind,
            model=self._model, native_model=MODEL, native_request_sha256=native_request_sha,
            request=request, request_sha256=request_sha,
            max_input_tokens=tokens, max_output_tokens=OUTPUT_TOKENS,
            automatic_retries=0, deadline=self._deadline)
        broker = isinstance(self._transport, BrokerBudgetedTransport)
        reservation = None
        if broker:
            self._event('broker_request', budget_owner='webprm_worker.completion', **reservation_record)
        else:
            reservation = self._budget.reserve(json.loads(_canonical(reservation_record)))
            if type(reservation) not in (int, str) or isinstance(reservation, str) and not reservation:
                raise EvaluationNotReady('Shared budget returned no durable reservation identity')
            self._event('reserved_request', reservation=reservation, **reservation_record)
        if time.time() >= self._deadline:
            raise EvaluationNotReady('Fuzzy deadline reached after reservation; reservation retained')
        self._event('dispatch', reservation=reservation, provider_call=self.provider_calls + 1)
        try:
            self.provider_calls += 1
            payload = json.loads(_canonical(request))
            response = (self._transport.send(payload, deadline=self._deadline) if broker
                        else self._transport.send(payload))
            if inspect.isawaitable(response):
                if inspect.iscoroutine(response):
                    response.close()
                raise EvaluationNotReady('Fuzzy transport returned an awaitable')
            # Preserve complete received evidence before parsing/settlement.
            self._event('response', reservation=reservation, response=response)
            if not isinstance(response, dict) or response.get('model') != self._model:
                raise EvaluationNotReady('Native fuzzy response model identity unavailable or changed')
            choices, usage = response.get('choices'), response.get('usage')
            if not isinstance(choices, list) or len(choices) != 1:
                raise EvaluationNotReady('Native fuzzy response requires exactly one choice')
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get('finish_reason') != 'stop':
                raise EvaluationNotReady('Native fuzzy response was incomplete')
            message = choice.get('message')
            content = message.get('content') if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise EvaluationNotReady('Native fuzzy response text unavailable')
            if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] <= 0
                    for k in ('prompt_tokens', 'completion_tokens')):
                raise EvaluationNotReady('Native fuzzy response usage unavailable')
            usage_record = dict(model=self._model, purpose='native_fuzzy', native_kind=self._kind,
                request_sha256=request_sha, prompt_tokens=usage['prompt_tokens'],
                completion_tokens=usage['completion_tokens'])
            overrun = usage['prompt_tokens'] > tokens or usage['completion_tokens'] > OUTPUT_TOKENS
            if overrun:
                self._halted = True
            if broker:
                receipt = self._broker_receipt(response, request)
                reservation = receipt['reservation_id']
                self._event('broker_settled', receipt=receipt)
                settled = True
            else:
                try:
                    settled = self._budget.settle(reservation, usage_record)
                except Exception:
                    self._halted = True
                    raise
            if settled is False or overrun:
                self._halted = True
                raise EvaluationNotReady('Fuzzy usage exceeded reservation; further calls halted')
            if not broker:
                self._event('settled', reservation=reservation, usage=usage_record)
            return content, usage
        except Exception as error:
            self._event('request_invalid', reservation=reservation, error_type=type(error).__name__,
                        automatic_retry=False, provider_calls=self.provider_calls)
            if isinstance(error, EvaluationNotReady):
                raise
            raise EvaluationNotReady('Native fuzzy transport/accounting failed; no retry') from error

    def _evaluate(self, kind, pred, reference, question):
        if not self.enabled or self._halted:
            raise EvaluationNotReady('Native fuzzy judge is disabled or halted')
        if any(not isinstance(value, str) for value in (pred, reference, question)):
            raise EvaluationNotReady('Native fuzzy inputs must be strings')
        with self._lock:
            self._kind = kind
            try:
                score = self._native['llm_' + kind](pred, reference, question)
            except AssertionError as error:
                self._event('native_verdict_invalid', error_type=type(error).__name__)
                raise EvaluationNotReady('Native fuzzy verdict was malformed') from error
            self._event('native_verdict', score=score)
            return score

    def fuzzy_match(self, pred, reference, question):
        return self._evaluate('fuzzy_match', pred, reference, question)

    def ua_match(self, pred, reference, question):
        return self._evaluate('ua_match', pred, reference, question)
