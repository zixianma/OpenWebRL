"""Cross-benchmark live evaluation of web-agent SELECTORS (OM2W, WebVoyager, DeepShop).

Design
------
One *condition* (how an action is chosen at each step) is evaluated on seeded random task subsets of up to three
benchmarks. The actor is always OpenWebRL-4B-SFT served by sglang on the job's single H200.

  ``sft1``           actor alone, one sample per step (no selector).
  ``piotr5``         five actor samples per step; Piotr's SelectionARM picks one through the
                     ``serve_arm.py --mode selection`` ``/select`` contract (``scripts/serve_arm_sglang.py`` in front of a
                     co-located sglang selector server).
  ``head5:<head.pt>`` five actor samples per step; a frozen selection head scores them from hidden states captured
                     inside the actor's own sglang server (``actor_head_capture`` forward hooks + ``actor_head_scoring``;
                     the "actor_cache_head" path), argmax with ties to the lowest index.

Layout: ``<root>/<condition-slug>/<benchmark>/`` holds a frozen ``plan.json`` (benchmark, condition, head path + sha256,
seeded task subset, actor checkpoint, protocol, judge caps, source digests), a frozen copy of the benchmark's task file,
the judge ledger (``judge-budget/``) and ``collection/`` (``records/<key>.json`` = one committed result per task,
``attempts/<key>/<ns>/`` = every physical attempt with its result, actor-request receipts, per-step selector traces and
saved samples). A task with a verified record is skipped on resume; an interrupted attempt is preserved and a new
physical attempt is started. The worker entry point (``python -m openwebrl.crossbench_selector_eval --config ...``) runs
one benchmark; ``scripts/run_crossbench_selector_eval.py`` prepares plans, writes the sbatch and owns the servers.

Judge caps: every native judge request goes through :class:`NativeCappedJudge`, which reserves a conservative upper cost
(20k tokens/image + text bytes + 4096 output tokens at list price) in a POSIX-locked, crash-persistent ledger before the
call and settles to the billed usage afterwards. Reaching the call or USD cap writes ``halt.json``; the worker then does
not commit the affected task (the outcome was caused by the cap, not the agent) and stops the benchmark.

Copied from (logic reused, code duplicated so this module does not depend on files other sessions are editing):
  * Sol61 frozen snapshot ``evaluations/sol61-inference-scaling-20261009/preparation/source-v1``:
    ``openwebrl/sol61_benchmark_judge.py`` -> ``ROUTES``/``route_for``/``NativeCappedJudge`` (verbatim logic);
    ``openwebrl/sol61_budget.py`` + ``openwebrl/shared_ledger_lock.py`` -> ``Ledger``/``ledger_lock``/``write_json``;
    ``openwebrl/arm_controlled_inference.py`` ``run()`` -> worker loop: native reward routing by benchmark with
    ``reward._get_openai_client`` patched to the capped judge, ``EvalArgs`` (30 turns, 1 screenshot, full history,
    browser_env format, 180 s step/judge timeouts, 1800 s task timeout), ``MeteredSelector`` actor receipts, validity
    rule, record verification, two-task startup gate, 8 concurrent local_process episodes;
    ``openwebrl/arm_sol61_inference.py`` ``protocol_for`` -> ``protocol_for`` (seed 42, server seed 4200, T .7 / top-p .9 /
    top-k endpoint default / 1024 tokens / 32768 context, judge 4096 output tokens);
    ``preparation/tasks/deepshop.jsonl`` -> DeepShop task file (repo file + metadata block; pinned by sha256).
  * selectors-samehw snapshot ``evaluations/selectors-samehw-20261012/source``: ``openwebrl/arm_inference.py``
    ``ActionSelector`` (baseline / selection / actor_cache_head branches, candidate seeds, selection parsing with the
    released first-candidate fallback) -> :class:`CrossbenchSelector`; the ``rid``-aware ``MeteredSelector``;
    ``scripts/run_arm_paired_inference.py`` server launch (ACTORHEAD_* env, ``--forward-hooks``, piotr5 sglang selector
    at mem-fraction .30 + ``serve_arm_sglang.py serve``, actor mem-fraction .45, CUDA graphs max batch 48).
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import threading
import time

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
ACTOR = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
PIOTR_SELECTION = Path('/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89')
PIOTR_SGLANG_DIR = Path('/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection-sglang-81b452d')
ACTOR_PARAMS = 4022468096 + 415347712          # FLOP accounting for head follow-ups (as selectors-samehw)
ACTOR_2B = Path('/gpfs/scrubbed/zixianma/checkpoints/web/qwen3-vl-2b-thinking-openwebrl-sft-95d5b13')
ACTOR_8B = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-8B-SFT-sglang')   
# Actor identity, FLOP-accounting parameters (language model + vision tower) and the layers a selection head reads.
ACTORS = {
    str(ACTOR.resolve()): dict(name='OpenWebRL-4B-SFT:iteration0', revision='15e777db2ddba2e0e82080ebccd3ad8d215b7f0a',
                     params=ACTOR_PARAMS, capture_layers=(18, 27, 36)),
    str(ACTOR_2B.resolve()): dict(name='davidan03/qwen3-vl-2b-thinking-openwebrl-sft', revision='95d5b133ab91dd5729030ae3be2d5281d60fcb20',
                        params=2127532032, capture_layers=(14, 21, 28)),
    str(ACTOR_8B.resolve()): dict(name='OpenWebRL/OpenWebRL-8B-SFT', revision='838c4a4842555a2a61e81d7a28bac7b2e8144800',
                        params=8767123696, capture_layers=None),
}


def actor_info(actor=None):
    """Registered actor identity; any other path keeps the original 4B protocol identity (the pre-registry behaviour)."""
    return ACTORS.get(str(Path(actor or ACTOR).resolve()), ACTORS[str(ACTOR.resolve())])
REVISION = 'crossbench-selector-eval-v1'
BENCHMARKS = ('online_mind2web', 'webvoyager', 'deepshop')
SYSTEM_POLICY_SHA256 = '7028b29a14e6be1ff05e36a7ae708a89ab32529efad6cc47d2a388e531449aa9'
ATTEMPT = ContextVar('crossbench_attempt', default=None)

# Task sources. OM2W and WebVoyager are byte-identical to the Sol61 frozen task files; the repo's DeepShop file lacks the
# `metadata` block that the browser loader needs, so the Sol61 normalized copy is used (pinned by sha256).
SOL61_TASKS = RUNTIME / 'evaluations/sol61-inference-scaling-20261009/preparation/tasks'
TASK_SOURCES = {
    'online_mind2web': dict(path=REPO / 'openwebrl/data/eval/online-mind2web.jsonl', count=300,
                            sha256='8343c23be98d6d63856e9b53ff3884222be099cc0f55ad5edb475176f54317ed'),
    'webvoyager': dict(path=REPO / 'openwebrl/data/eval/webvoyager_fara.jsonl', count=595,
                       sha256='ac9253317b76884abbd8d89e9b02570f3f4fbf18e428c335bc8761d778c6fe68'),
    'deepshop': dict(path=SOL61_TASKS / 'deepshop.jsonl', count=150,
                     sha256='4694e61ec2b8c826334a2d0c4e6e82f0452a00d29b10a42ab95da5f75e2193b2'),
}

# --- copied from Sol61 sol61_benchmark_judge.py -------------------------------------------------------------------
ROUTES = {
    # o4-mini was retired (404 model_not_found, 2026-10-23); gpt-5.6-terra is its official replacement. Same AgentTrek
    # prompt, image and parser. Prices are the pre-July-30 list rates ($2.50/$15 per M), an upper bound on the current ones.
    'online_mind2web': dict(module='reward_online_mind2web', model='gpt-5.6-terra', protocol='online_mind2web/AgentTrek',
                            images=3, api='create', input_price=2.5, output_price=15.),
    'webvoyager': dict(module='reward_webvoyager', model='gpt-4o', protocol='webvoyager', images=30, api='create',
                       input_price=2.5, output_price=10.),
    'deepshop': dict(module='reward_deepshop', model='gpt-4o', protocol='deepshop', images=30, api='parse',
                     input_price=2.5, output_price=10.),
}
JUDGE_MAX_COMPLETION_TOKENS = 4096
# Per (condition, benchmark) caps for a 100-task subset; cumulative across retries/resumes. Calls: the native rewards
# retry up to 4 times per episode. USD leaves headroom for 8 in-flight conservative reservations (30 images x 20k
# tokens = $1.5 per gpt-4o reservation) above the expected ~$0.05/episode actual spend.
DEFAULT_JUDGE_USD = {'online_mind2web': 6.0, 'webvoyager': 15.0, 'deepshop': 12.0}   # OM2W: terra ~2.3x o4-mini


def route_for(benchmark):
    if benchmark not in ROUTES:
        raise ValueError(f'Unsupported benchmark: {benchmark!r}')
    return dict(ROUTES[benchmark])


def load_native_reward(benchmark):
    import importlib
    return importlib.import_module('openwebrl.eval.' + route_for(benchmark)['module'])


def default_judge_caps(benchmark, n):
    return dict(max_calls=4 * n, max_usd=float(DEFAULT_JUDGE_USD[benchmark]) * n / 100)


# --- copied from Sol61 sol61_budget.py / shared_ledger_lock.py ----------------------------------------------------
_guard = threading.Lock()


@contextmanager
def ledger_lock(path):
    with _guard:
        with Path(path).resolve().open('a+') as handle:
            deadline = time.monotonic() + 60
            while True:
                try:
                    fcntl.lockf(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN):
                        raise
                    if time.monotonic() > deadline:
                        raise TimeoutError('Shared ledger lock wait exceeded 60 seconds') from exc
                    time.sleep(.005)
            try:
                yield
            finally:
                fcntl.lockf(handle, fcntl.LOCK_UN)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, default=str)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class JudgeHalted(RuntimeError):
    """The judge ledger reached its cap or saw a contract failure; stop and preserve state."""


class Ledger:
    def __init__(self, root, model, max_usd, max_calls):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = dict(model=model, max_usd=max_usd, max_calls=max_calls)

    def _halt_locked(self, record):
        path = self.root / 'halt.json'          # keep the first causal failure
        if not path.exists():
            write_json(path, dict(epoch=time.time(), **record))

    def halt(self, record):
        with ledger_lock(self.root / 'lock'):
            self._halt_locked(record)

    def state(self):
        path = self.root / 'ledger.json'
        return read_json(path) if path.exists() else dict(config=self.config, items={}, calls=0, charged_or_reserved_usd=0.)

    def change(self, amount=None, ident=None, actual=None):
        with ledger_lock(self.root / 'lock'):
            path = self.root / 'ledger.json'
            state = read_json(path) if path.exists() else dict(config=self.config, items={})
            if state['config'] != self.config:
                raise ValueError('Shared caps changed across processes/retries')
            items = state['items']
            overrun = False
            if ident is None:
                if not math.isfinite(amount) or amount <= 0:
                    raise ValueError('Invalid reservation')
                total = sum(x['charged_or_reserved_usd'] for x in items.values())
                if (self.root / 'halt.json').exists():
                    raise ValueError('Judge ledger halted; inspect preserved first cause')
                if len(items) >= self.config['max_calls'] or total + amount > self.config['max_usd']:
                    self._halt_locked(dict(reason='API cap', charged_or_reserved_usd=total, requested_reservation_usd=amount,
                                           calls=len(items), config=self.config))
                    raise ValueError('Judge API cap reached')
                ident = str(len(items) + 1)
                items[ident] = dict(status='reserved', charged_or_reserved_usd=amount, started_epoch=time.time())
            else:
                if not math.isfinite(actual) or actual < 0:
                    raise ValueError('Invalid settlement')
                row = items[str(ident)]
                if row['status'] != 'reserved':
                    raise ValueError('Duplicate settlement')
                reserved = row['charged_or_reserved_usd']
                overrun = actual > reserved
                row.update(status='received', charged_or_reserved_usd=actual, reserved_usd=reserved)
                if overrun:
                    row['reservation_overrun_usd'] = actual - reserved
                    self._halt_locked(dict(reason='Usage exceeds conservative reservation', request=ident,
                                           reserved_usd=reserved, actual_usd=actual))
            state['calls'] = len(items)
            state['charged_or_reserved_usd'] = sum(x['charged_or_reserved_usd'] for x in items.values())
            write_json(path, state)
            if overrun:
                raise ValueError('Reservation underestimate; known actual charge preserved')
            return ident


class NativeCappedJudge:
    """Cap native benchmark judge calls without replacing their prompts, parsers or validity rules (Sol61, verbatim)."""

    def __init__(self, root, benchmark, *, max_calls, max_usd, attempt_context, client):
        from types import SimpleNamespace
        self.root = Path(root)
        self.route = route_for(benchmark)
        self.benchmark = benchmark
        self.client = client
        self.attempt_context = attempt_context
        self.budget = Ledger(root, self.route['model'], max_usd, max_calls)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=self.parse)))

    @property
    def halted(self):
        return (self.root / 'halt.json').exists()

    def upper_cost(self, messages):
        text_bytes = images = 0
        for m in messages:
            content = m.get('content', '')
            if isinstance(content, str):
                text_bytes += len(content.encode())
            else:
                for part in content:
                    if part.get('type') == 'image_url':
                        images += 1
                    else:
                        text_bytes += len(json.dumps(part).encode())
        cap = JUDGE_MAX_COMPLETION_TOKENS
        return ((text_bytes + images * 20000 + cap) * self.route['input_price'] + cap * self.route['output_price']) / 1e6

    async def _request(self, api, kwargs):
        if api != self.route['api'] or kwargs.get('model') != self.route['model']:
            raise ValueError('Wrong native benchmark judge route/model')
        if any(k in kwargs for k in ('max_tokens', 'max_completion_tokens', 'store', 'service_tier')):
            raise ValueError('Caller changed controlled judge request settings')
        upper = self.upper_cost(kwargs['messages'])
        ident = await asyncio.to_thread(self.budget.change, amount=upper)
        request = dict(kwargs, max_completion_tokens=JUDGE_MAX_COMPLETION_TOKENS, store=False, service_tier='default')
        saved = dict(request)
        if 'response_format' in saved and hasattr(saved['response_format'], 'model_json_schema'):
            saved['response_format'] = saved['response_format'].model_json_schema()
        receipt = dict(request_id=ident, attempt=self.attempt_context.get(), benchmark=self.benchmark,
                       judge_protocol=self.route['protocol'], requested_model=self.route['model'], request=saved,
                       status='reserved', reserved_usd=upper, started_epoch=time.time())
        path = self.root / 'requests' / f'{int(ident):05d}.json'
        await asyncio.to_thread(write_json, path, receipt)
        start = time.monotonic()
        try:
            endpoint = self.client.beta.chat.completions.parse if api == 'parse' else self.client.chat.completions.create
            result = await endpoint(**request)
            usage = result.usage.model_dump()
            if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ('prompt_tokens', 'completion_tokens')):
                raise ValueError('Missing/malformed native judge token receipt')
            cost = (usage['prompt_tokens'] * self.route['input_price']
                    + usage['completion_tokens'] * self.route['output_price']) / 1e6
            receipt.update(status='received', response=result.model_dump(), usage=usage, cost_upper_usd=cost,
                           seconds=time.monotonic() - start, returned_model=result.model)
            await asyncio.to_thread(write_json, path, receipt)
            await asyncio.to_thread(self.budget.change, ident=ident, actual=cost)
            if result.model != self.route['model'] and not result.model.startswith(self.route['model'] + '-20'):
                raise ValueError('Substituted native judge model')
            return result
        except BaseException as exc:
            receipt.update(error_type=type(exc).__name__, seconds=time.monotonic() - start)
            status = getattr(exc, 'status_code', None)
            if status is not None:
                receipt['http_status'] = int(status)
            await asyncio.to_thread(write_json, path, receipt)
            if isinstance(exc, ValueError):
                await asyncio.to_thread(self.budget.halt, dict(reason='Native judge contract failure', request=ident,
                                                               error_type=type(exc).__name__))
            raise

    async def create(self, **kwargs):
        return await self._request('create', kwargs)

    async def parse(self, **kwargs):
        return await self._request('parse', kwargs)

    async def close(self):
        close = getattr(self.client, 'close', None)
        if close is not None:
            await close()


# --- conditions, tasks, plans ---------------------------------------------------------------------------------------
def parse_condition(condition):
    """``sft1`` | ``piotr<N>`` | ``head<N>:<path/to/head.pt>`` | ``head<N>x:<head.pt>`` -> normalized description (head
    sha256 pinned). N = actor samples per step (2-5); ``piotr5`` / ``head5`` keep their original names and slugs."""
    if condition == 'sft1':
        return dict(name='sft1', kind='baseline', candidates=1, slug='sft1', head_path=None, head_sha256=None,
                    selection_rule='single_sample')
    m = re.fullmatch(r'piotr([2-5])', condition)
    if m:
        return dict(name=condition, kind='selection', candidates=int(m[1]), slug=condition, head_path=None, head_sha256=None,
                    selection_rule='canonical_no_cot_json_schema_sglang_first_candidate_fallback',
                    selector_model=str(PIOTR_SELECTION), selector_revision='81b452d800d9f859687074f82680dd5257e02d89')
    m = re.fullmatch(r'head([2-5])(x?):(.+)', condition)
    if m:
        n, external, path = int(m[1]), m[2], Path(m[3]).resolve()
        if not path.is_file():
            raise ValueError(f'Selection head not found: {path}')
        digest = sha256_file(path)
        name = f'head{n}{external}'
        if external:
            return dict(name=name, kind='frozen_head_scorer', candidates=n, slug=f'{name}-{digest[:12]}', head_path=str(path),
                        head_sha256=digest, selection_rule='argmax_head_score_ties_first',
                        encoder=str(ACTOR), encoder_revision=ACTORS[str(ACTOR.resolve())]['revision'])
        return dict(name=name, kind='actor_cache_head', candidates=n, slug=f'{name}-{digest[:12]}', head_path=str(path),
                    head_sha256=digest, selection_rule='argmax_head_score_ties_first')
    raise ValueError(f'Unknown condition {condition!r}; expected sft1, piotr<N>, head<N>:<head.pt> or head<N>x:<head.pt>')


def task_id_of(row, index):
    """Same identity rule as ``run_evaluate.load_tasks_from_jsonl``."""
    metadata = dict(row.get('metadata', {}) or {})
    return metadata.get('task_id') or row.get('id') or f'task_{index}'


def read_task_ids(path):
    ids = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if line.strip():
            ids.append(task_id_of(json.loads(line), i))
    return ids


def sample_subset(task_ids, n, seed):
    """Seeded random subset, independent of file order: sample from the sorted unique ids."""
    unique = sorted(set(task_ids))
    if len(unique) != len(task_ids):
        raise ValueError('Duplicate task ids in source')
    if not 0 < n <= len(unique):
        raise ValueError(f'Subset size {n} outside 1..{len(unique)}')
    return random.Random(f'{REVISION}/subset/{seed}').sample(unique, n)


def protocol_for(benchmark, condition, decoding=None, horizon=None, actor=None, speculative=None):
    """Frozen protocol; ``decoding`` (dict(temperature, top_p)) overrides the default T 0.7 / top-p 0.9 actor sampling;
    ``horizon`` (dict(max_steps, history_full_turns)) overrides the 30-step full-history default; ``speculative``
    ('ngram') serves the actor with sglang n-gram speculative decoding (same sampled distribution; per-request seeds are
    not honoured)."""
    if speculative not in (None, 'ngram'):
        raise ValueError(f'Unsupported speculative decoding {speculative!r}')
    route = route_for(benchmark)
    cond = parse_condition(condition) if isinstance(condition, str) else condition
    temperature, top_p = (decoding['temperature'], decoding['top_p']) if decoding else (.7, .9)
    info = actor_info(actor)
    return dict(
        actor=info['name'], actor_revision=info['revision'],
        browser='local_process', browser_concurrency=8, temperature=temperature, top_p=top_p,
        top_k='omitted, native generate endpoint default', max_new_tokens=1024,
        max_steps=int(horizon['max_steps']) if horizon else 30, context_length=32768, context_num_screenshots=1,
        history=(f"last {int(horizon['history_full_turns'])} turns full; older turns: action and tool result line only"
                 if horizon and horizon.get('history_full_turns') else 'full'), seed=42, server_seed=4200,
        sampling_request=dict(temperature=temperature, top_p=top_p, max_new_tokens=1024),
        actor_system_policy_sha256=SYSTEM_POLICY_SHA256,
        benchmark=benchmark, judge=route['model'], judge_protocol=route['protocol'], judge_api=route['api'],
        judge_max_attached_imgs=route['images'], judge_response_tokens=JUDGE_MAX_COMPLETION_TOKENS,
        condition=cond['name'], selector_kind=cond['kind'], candidates=cond['candidates'],
        selection_rule=cond['selection_rule'], head_sha256=cond['head_sha256'],
        selector_model=cond.get('selector_model'), selector_revision=cond.get('selector_revision'),
        actor_serving='sglang, one H200, mem-fraction .45, CUDA graphs max batch 48'
                      + (', forward hooks capture layers ' + '/'.join(map(str, info['capture_layers']))
                         if cond['kind'] == 'actor_cache_head' else '')
                      + (', n-gram speculative decoding (--speculative-algorithm NGRAM)' if speculative else ''),
        selector_serving={'selection': 'sglang on the actor GPU, mem-fraction .30, behind serve_arm_sglang /select',
                          'actor_cache_head': 'CPU head over hidden states exported by the actor server',
                          'frozen_head_scorer': 'OpenWebRL-4B-SFT encoder (transformers, bf16) on the actor GPU re-encodes the '
                                                'prompt + candidates; unchanged head scores them (serve_frozen_head_scorer.py)',
                          'baseline': None}[cond['kind']],
        optimizer_updates=0, cohort_revision=REVISION)


def protocol_sha(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def attempt_key(benchmark, slug, task_id, seed):
    return hashlib.sha256(f'{REVISION}/{benchmark}/{slug}/{task_id}/{seed}'.encode()).hexdigest()


SOURCE_FILES = (
    'openwebrl/crossbench_selector_eval.py', 'scripts/run_crossbench_selector_eval.py',
    'openwebrl/generate_browser.py', 'openwebrl/run_evaluate.py', 'openwebrl/reward_browser.py',
    'openwebrl/eval/_shared.py', 'openwebrl/eval/reward_online_mind2web.py', 'openwebrl/eval/reward_webvoyager.py',
    'openwebrl/eval/reward_deepshop.py', 'openwebrl/env/prompts/system_prompt_browser_env.md',
    'openwebrl/actor_head_capture.py', 'openwebrl/actor_head_scoring.py', 'openwebrl/selection_head.py',
    'openwebrl/selection_head_features.py', 'openwebrl/arm_inference.py', 'scripts/serve_arm_sglang.py',
)


def source_digests(repo=REPO):
    return {name: (sha256_file(Path(repo) / name) if (Path(repo) / name).exists() else None) for name in SOURCE_FILES}


def source_drift(plan, repo=REPO):
    now = source_digests(repo)
    return {k: dict(planned=v, now=now.get(k)) for k, v in plan['source_digests'].items() if now.get(k) != v}


def build_plan(directory, *, benchmark, condition, n, seed, actor=ACTOR, task_source=None, judge_caps=None, repo=REPO,
               decoding=None, subset_of=None, horizon=None, speculative=None):
    """Freeze one (condition, benchmark) cohort in ``directory``; refuse to change an existing different plan."""
    directory = Path(directory).resolve()
    cond = parse_condition(condition)
    source = dict(TASK_SOURCES[benchmark])
    if task_source is not None:
        source.update(path=Path(task_source), sha256=None, count=None)
    source_path = Path(source['path'])
    digest = sha256_file(source_path)
    if source['sha256'] is not None and digest != source['sha256']:
        raise ValueError(f'{benchmark} task source changed: {source_path}')
    ids = read_task_ids(source_path)
    if source['count'] is not None and len(ids) != source['count']:
        raise ValueError(f'{benchmark} task count changed')
    if subset_of is not None:                      # the first n tasks of an existing frozen plan (paired comparisons)
        parent = read_json(subset_of)
        if parent['benchmark'] != benchmark or not set(parent['task_ids']) <= set(ids):
            raise ValueError('subset_of plan is for another benchmark or task source')
        subset = parent['task_ids'][:n]
    else:
        subset = sample_subset(ids, n, seed)
    policy = Path(repo) / 'openwebrl/env/prompts/system_prompt_browser_env.md'
    if sha256_file(policy) != SYSTEM_POLICY_SHA256:
        raise ValueError('Actor system policy differs from the Sol61/selectors-samehw protocol')
    actor = Path(actor).resolve()
    if not (actor / 'config.json').exists():
        raise ValueError(f'Actor checkpoint missing: {actor}')
    protocol = protocol_for(benchmark, cond, decoding, horizon, actor, speculative)
    caps = dict(judge_caps or default_judge_caps(benchmark, n))
    frozen_tasks = directory / 'tasks-source.jsonl'
    plan = dict(revision=REVISION, root=str(directory), benchmark=benchmark, condition=cond, subset_seed=seed, n=n,
                task_source=str(source_path), task_source_sha256=digest, task_file=str(frozen_tasks),
                task_ids=subset, actor=str(actor), protocol=protocol, protocol_sha256=protocol_sha(protocol),
                judge_caps=dict(max_calls=int(caps['max_calls']), max_usd=float(caps['max_usd'])),
                judge_budget_root=str(directory / 'judge-budget'), collection=str(directory / 'collection'),
                source=str(Path(repo).resolve()), source_digests=source_digests(repo),
                keys={t: attempt_key(benchmark, cond['slug'], t, protocol['seed']) for t in subset})
    if decoding:
        plan['decoding'] = dict(temperature=float(decoding['temperature']), top_p=float(decoding['top_p']))
    if subset_of is not None:
        plan['subset_of'] = str(Path(subset_of).resolve())
    if horizon:
        plan['horizon'] = dict(max_steps=int(horizon['max_steps']), history_full_turns=int(horizon.get('history_full_turns') or 0))
    if speculative:
        plan['speculative'] = speculative
    path = directory / 'plan.json'
    if path.exists():
        old = read_json(path)
        if {k: v for k, v in old.items() if k != 'source_digests'} != {k: v for k, v in plan.items() if k != 'source_digests'}:
            raise ValueError(f'A different frozen plan already exists at {path}')
        return old
    directory.mkdir(parents=True, exist_ok=True)
    frozen_tasks.write_bytes(source_path.read_bytes())
    write_json(path, plan)
    return plan


def validate_plan(directory, *, allow_source_drift=False, repo=REPO):
    directory = Path(directory).resolve()
    plan = read_json(directory / 'plan.json')
    if plan['root'] != str(directory) or plan['revision'] != REVISION:
        raise ValueError('Plan moved or from another revision')
    if sha256_file(plan['task_file']) != plan['task_source_sha256']:
        raise ValueError('Frozen task file changed')
    if plan['protocol'] != protocol_for(plan['benchmark'], plan['condition'], plan.get('decoding'), plan.get('horizon'), plan['actor'],
                                        plan.get('speculative')) or protocol_sha(plan['protocol']) != plan['protocol_sha256']:
        raise ValueError('Protocol changed since the plan was frozen')
    if plan['condition']['head_path'] and sha256_file(plan['condition']['head_path']) != plan['condition']['head_sha256']:
        raise ValueError('Selection head file changed since the plan was frozen')
    ids = read_task_ids(plan['task_file'])
    if not set(plan['task_ids']) <= set(ids) or len(set(plan['task_ids'])) != len(plan['task_ids']):
        raise ValueError('Plan task ids not in frozen task file')
    drift = source_drift(plan, repo)
    if drift and not allow_source_drift:
        raise ValueError('Source files changed since prepare: ' + ', '.join(sorted(drift)))
    return plan


# --- selector (copied/trimmed from selectors-samehw arm_inference.ActionSelector) --------------------------------------
def split_response(response):
    if '</think>' in response:
        thought, action = response.split('</think>', 1)
        thought = thought.rsplit('<think>', 1)[-1].strip()
    else:
        thought, action = '', response
    return {'thought': thought, 'action': action.strip()}


def parse_selection(text, count):
    matches = list(re.finditer(r'\{\s*"selection"\s*:\s*(\d+)\s*\}', text))
    if not matches:
        raise ValueError('ARM response has no selection JSON')
    index = int(matches[-1].group(1)) - 1
    if not 0 <= index < count:
        raise ValueError('ARM selection is out of range')
    return index


def candidate_seed(seed, task_id, turn, candidate):
    key = f'{seed}:{task_id}:{turn}:{candidate}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'big') % (2**31 - 1)


async def request_selection_result(endpoint, payload, timeout, connect_timeout=10, raw_path=False):
    """POST /select (or ``endpoint`` itself with raw_path); retry only connections that failed before submission."""
    import httpx
    limits = httpx.Timeout(timeout, connect=min(timeout, connect_timeout))
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=limits, trust_env=False) as client:
                response = await client.post(endpoint if raw_path else endpoint + '/select', json=payload)
                response.raise_for_status()
                return response.json()
        except (httpx.ConnectTimeout, httpx.ConnectError):
            if attempt == 2:
                raise
            await asyncio.sleep(.2)


class HeadBackend:
    """Frozen selection head over the actor server's captured hidden states (actor_head_scoring)."""

    def __init__(self, head_path, tokenizer_path, capture_dir, params=ACTOR_PARAMS, timeout=180):
        self.head_path, self.tokenizer_path = str(head_path), str(tokenizer_path)
        self.capture_dir, self.params, self.timeout = str(capture_dir), int(params), timeout

    def load(self):
        from openwebrl.actor_head_scoring import bundle
        return bundle(self.head_path, self.tokenizer_path)

    async def generate(self, call, url, input_text, images, rid):
        from openwebrl.actor_head_scoring import generate_with_tail
        tok = (await asyncio.to_thread(self.load))['tokenizer']
        return await generate_with_tail(call, url, input_text, list(images or []), rid, tokenizer=tok,
                                        capture_dir=self.capture_dir, timeout=self.timeout)

    async def score(self, gens, metas, rids):
        from openwebrl.actor_head_scoring import score_cached
        return await score_cached(gens, metas, rids, head_path=self.head_path, tokenizer_path=self.tokenizer_path,
                                  capture_dir=self.capture_dir, params_for_flops=self.params)


class CrossbenchSelector:
    """Sample K actor proposals at one live state, choose one, execute it unchanged; trace every decision."""

    def __init__(self, kind, output, *, seed, candidates, endpoint=None, head=None, timeout=180,
                 request_fn=request_selection_result):
        if kind not in ('baseline', 'selection', 'actor_cache_head', 'frozen_head_scorer'):
            raise ValueError(kind)
        if kind in ('selection', 'frozen_head_scorer') and not endpoint:
            raise ValueError(f'{kind} needs an endpoint')
        if kind == 'actor_cache_head' and head is None:
            raise ValueError('actor_cache_head needs a HeadBackend')
        self.kind, self.output, self.seed = kind, Path(output), seed
        self.candidates = 1 if kind == 'baseline' else int(candidates)
        self.endpoint = (endpoint or '').rstrip('/')
        self.head, self.timeout, self.request_fn = head, timeout, request_fn
        self.output.mkdir(parents=True, exist_ok=True)

    async def __call__(self, *, infer, url, input_text, sampling_params, images, observation, history, task, task_id,
                       turn, timeout):
        seeds = [candidate_seed(self.seed, task_id, turn, j) for j in range(self.candidates)]
        rids = [f'actorhead-{os.urandom(8).hex()}-{j}' for j in range(self.candidates)]
        calls = []
        for j in range(self.candidates):
            params = dict(sampling_params, sampling_seed=seeds[j])
            extra = {'rid': rids[j]} if self.kind == 'actor_cache_head' else {}
            calls.append(infer(url, input_text, params, images, timeout_secs=timeout, **extra))
        started = time.monotonic()
        if self.kind == 'actor_cache_head':
            done = await asyncio.gather(*(self.head.generate(c, url, input_text, images, rid) for c, rid in zip(calls, rids)))
            outputs, gens, metas = [d[0] for d in done], [d[1] for d in done], [d[2] for d in done]
        else:
            outputs = await asyncio.gather(*calls)
        generated = time.monotonic()
        candidates = [split_response(o[0]) for o in outputs]
        index, scores, raw, fallback, telemetry = 0, None, None, None, None
        if self.kind == 'actor_cache_head':
            result = await self.head.score(gens, metas, rids)
            scores = result['scores']
            if len(scores) != len(candidates) or not all(math.isfinite(x) for x in scores):
                raise ValueError('Invalid cached-head score vector')
            index = max(range(len(scores)), key=lambda j: (scores[j], -j))
            telemetry = {k: v for k, v in result.items() if k != 'scores'}
        elif self.kind == 'frozen_head_scorer':
            payload = {'prompt_text': input_text, 'image': images, 'candidate_ids': [list(o[1]) for o in outputs]}
            result = await self.request_fn(self.endpoint + '/score', payload, self.timeout, raw_path=True)
            scores = result['scores']
            if len(scores) != len(candidates) or not all(math.isfinite(x) for x in scores):
                raise ValueError('Invalid frozen-head score vector')
            index = max(range(len(scores)), key=lambda j: (scores[j], -j))
            telemetry = {k: v for k, v in result.items() if k != 'scores'}
        elif self.kind == 'selection':
            import base64
            payload = {'mode': 'selection', 'task': task, 'url': observation.get('active_tab_url', ''),
                       'history': [split_response(r) for r in history], 'candidates': candidates,
                       'screenshot': base64.b64encode(observation['screenshot']).decode()}
            result = await self.request_fn(self.endpoint, payload, self.timeout)
            raw = result.get('raw')
            try:
                index = parse_selection(raw or '', len(candidates))
            except ValueError as exc:             # released demo: first candidate; counted separately
                fallback, index = str(exc), 0
            telemetry = {k: v for k, v in result.items() if k not in ('raw', 'scores', 'response')}
        selected = time.monotonic()
        screenshot = observation.get('screenshot') or b''
        record = {'kind': self.kind, 'task_id': task_id, 'turn': turn, 'selected_index': index, 'scores': scores,
                  'selector_raw': raw, 'fallback': fallback, 'candidates': candidates, 'candidate_seeds': seeds,
                  'finish_types': [o[3] for o in outputs], 'url': observation.get('active_tab_url', ''),
                  'timing': {'generation_seconds': generated - started, 'selection_seconds': selected - generated,
                             'decision_seconds': selected - started},
                  'selector_telemetry': telemetry,
                  'prompt_sha256': hashlib.sha256(input_text.encode()).hexdigest(),
                  'screenshot_sha256': hashlib.sha256(screenshot).hexdigest()}
        path = self.output / (hashlib.sha256(str(task_id).encode()).hexdigest()[:20] + '.jsonl')
        await asyncio.to_thread(self._append, path, record)
        return outputs[index], {'selected_index': index, 'mode': self.kind, 'fallback': fallback, 'trace_path': str(path)}

    @staticmethod
    def _append(path, record):
        with path.open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')


class MeteredActor:
    """Wrap the selector: every actor /generate request leaves a receipt (copied from selectors-samehw, rid-aware)."""

    def __init__(self, selector, directory, system_policy=None, post=None):
        self.selector, self.directory, self.system_policy, self._post = selector, Path(directory), system_policy, post
        self.requests = self.output_tokens = self.input_tokens = 0
        self.client_seconds = 0.

    async def __call__(self, *, infer, **kwargs):
        if self.system_policy is not None and self.system_policy.strip() not in kwargs['input_text']:
            raise ValueError('Actual actor prompt omitted the required system policy')
        post = self._post
        if post is None:
            from openwebrl import generate_browser as generation
            post = generation.post

        async def metered(url, text, sampling_params, image_data, timeout_secs=None, rid=None):
            ident = self.requests
            self.requests += 1
            started = time.monotonic()
            receipt = dict(request=ident, started_epoch=time.time(), sampling_seed=sampling_params.get('sampling_seed'),
                           sampling_params=dict(sampling_params), rid=rid)
            try:
                payload = dict(text=text, sampling_params=sampling_params, return_logprob=True)
                if image_data:
                    payload['image_data'] = image_data
                if rid:
                    payload['rid'] = rid
                output = await asyncio.wait_for(post(url, payload), timeout=timeout_secs)
                meta = output['meta_info']
                pairs = meta['output_token_logprobs']
                tokens, logps = [p[1] for p in pairs], [p[0] for p in pairs]
                self.output_tokens += meta.get('completion_tokens', len(tokens))
                if 'prompt_tokens' not in meta:
                    raise ValueError('Actor input-token telemetry missing')
                self.input_tokens += meta['prompt_tokens']
                receipt.update(status='received', meta={k: v for k, v in meta.items() if 'logprobs' not in k})
                out_text = output['text']
                if self._post is None:
                    from openwebrl import generate_browser as generation
                    out_text = generation._ensure_im_end_w_new_line(out_text, tokens, logps)
                return out_text, tokens, logps, meta.get('finish_reason', {}).get('type', 'stop')
            except BaseException as exc:
                receipt.update(status='failed', error_type=type(exc).__name__)
                raise
            finally:
                duration = time.monotonic() - started
                self.client_seconds += duration
                receipt['client_seconds'] = duration
                await asyncio.to_thread(write_json, self.directory / 'actor-requests' / f'{ident:05d}.json', receipt)
        return await self.selector(infer=metered, **kwargs)


# --- records -------------------------------------------------------------------------------------------------------
def is_valid_result(result):
    """Sol61 validity rule: binary judged outcome, not aborted, no judge timeout."""
    valid = result.get('reward') in (0, 1) and 'ABORTED' not in str(result.get('status', ''))
    if (result.get('metadata') or {}).get('judge_timeout'):
        valid = False
    return bool(valid)


def read_traces(directory):
    traces, steps = [], []
    for path in sorted(Path(directory).glob('*.jsonl')):
        traces.append(dict(path=str(path), sha256=sha256_file(path)))
        steps.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    steps.sort(key=lambda s: s['turn'])
    return traces, steps


def verify_record(record, plan, task_id, root):
    if (record['task_id'] != task_id or record['key'] != plan['keys'][task_id] or record['benchmark'] != plan['benchmark']
            or record['condition'] != plan['condition']['name'] or record['protocol_sha256'] != plan['protocol_sha256']
            or record.get('head_sha256') != plan['condition']['head_sha256']):
        raise ValueError(f'Record identity/protocol changed for {task_id}')
    artifact = Path(record['artifact'])
    if not artifact.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Artifact outside the collection root')
    if sha256_file(artifact) != record['artifact_sha256']:
        raise ValueError(f'Result artifact hash mismatch for {task_id}')
    if record['valid'] and record['reward'] not in (0, 1):
        raise ValueError('Valid result must have a binary outcome')
    for entry in record['trace_files']:
        if sha256_file(entry['path']) != entry['sha256']:
            raise ValueError(f'Decision trace changed for {task_id}')
    return record


def summarize(records, plan, judge_state=None):
    ids = plan['task_ids']
    indexed = {}
    for r in records:
        if r['task_id'] not in plan['keys'] or r['task_id'] in indexed:
            raise ValueError('Unexpected or duplicate record')
        indexed[r['task_id']] = r
    rows = list(indexed.values())
    valid = sum(r['valid'] for r in rows)
    wins = sum(bool(r['valid'] and r['reward'] == 1) for r in rows)
    picks = {}
    for r in rows:
        for p in r['selector_picks']:
            picks[str(p)] = picks.get(str(p), 0) + 1
    judge_state = judge_state or {}
    out = dict(benchmark=plan['benchmark'], condition=plan['condition']['name'], condition_slug=plan['condition']['slug'],
               head_sha256=plan['condition']['head_sha256'], scheduled=len(ids), completed=len(rows),
               complete=len(rows) == len(ids), successes=wins, valid=valid, invalid=len(rows) - valid,
               errors=sum(bool(r.get('error_type')) for r in rows),
               success_rate_scheduled=wins / len(ids) if len(rows) == len(ids) else None,
               success_rate_completed=wins / len(rows) if rows else None,
               valid_only_rate=wins / valid if valid else None,
               browser_steps=sum(r['browser_steps'] for r in rows), selector_calls=sum(r['selector_calls'] for r in rows),
               selector_fallbacks=sum(r['selector_fallbacks'] for r in rows), selected_index_counts=picks,
               actor_requests=sum(r['actor_requests'] for r in rows),
               actor_output_tokens=sum(r['actor_output_tokens'] for r in rows),
               elapsed_seconds=sum(r['elapsed_seconds'] for r in rows),
               physical_attempts=sum(r.get('physical_attempt', 1) for r in rows),
               judge_calls=judge_state.get('calls', 0), judge_usd=judge_state.get('charged_or_reserved_usd', 0.),
               judge_caps=plan['judge_caps'], judge_halted=bool(judge_state.get('halted')),
               denominators='success_rate_scheduled = successes / scheduled tasks (invalid counts as failure); '
                            'valid_only_rate = successes / valid judged episodes')
    return out


def load_records(plan):
    root = Path(plan['collection'])
    rows = []
    for task_id in plan['task_ids']:
        path = root / 'records' / (plan['keys'][task_id] + '.json')
        if path.exists():
            rows.append(verify_record(read_json(path), plan, task_id, root))
    return rows


def judge_state(plan):
    root = Path(plan['judge_budget_root'])
    state = read_json(root / 'ledger.json') if (root / 'ledger.json').exists() else dict(calls=0, charged_or_reserved_usd=0.)
    return dict(calls=state.get('calls', 0), charged_or_reserved_usd=state.get('charged_or_reserved_usd', 0.),
                halted=(root / 'halt.json').exists())


# --- worker --------------------------------------------------------------------------------------------------------
class LiveRuntime:
    """The real benchmark plumbing (imports the browser/eval stack only inside the job)."""

    def judge_client(self):
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=os.getenv('JUDGE_API_KEY') or os.environ['OPENAI_API_KEY'],
                           base_url='https://api.openai.com/v1', max_retries=0, timeout=180)

    def load_tasks(self, task_file):
        from openwebrl import run_evaluate
        return {t['task_id']: t for t in run_evaluate.load_tasks_from_jsonl(task_file)}

    async def setup(self, config, plan, judge, root):
        from openwebrl import run_evaluate as evaluation, generate_browser as generation
        from slime.utils.http_utils import init_http_client
        import httpx
        reward = load_native_reward(plan['benchmark'])
        os.environ.update(SLIME_BROWSER_ENV_MODE='local_process', SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES='8',
                          SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR=str(root / 'browser_logs'),
                          SLIME_BROWSER_LOCAL_PROCESS_PORT_START=str(config['browser_port_start']),
                          SLIME_BROWSER_LOCAL_PROCESS_PORT_END=str(config['browser_port_start'] + 199),
                          WANDB_PROJECT='openwebrl-evals')
        generation._BROWSER_HOST_BLACKLIST_PATH = str(root / 'navigation-failures.txt')
        install_judge(evaluation, reward, judge)
        services = dict(actor=plan['actor'], condition=plan['condition'])
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            r = await client.get(f"http://127.0.0.1:{config['actor_port']}/get_model_info")
            r.raise_for_status()
            if Path(r.json()['model_path']).resolve() != Path(plan['actor']).resolve():
                raise ValueError('Wrong actor model served')
            kind = plan['condition']['kind']
            if kind == 'actor_cache_head':
                from openwebrl.actor_head_capture import hook_specs
                r = await client.get(f"http://127.0.0.1:{config['actor_port']}/server_info")
                r.raise_for_status()
                layers = actor_info(plan['actor'])['capture_layers']
                if r.json().get('forward_hooks') != hook_specs(layers):
                    raise ValueError('Actor server lacks the capture hooks')
                services['hooks'] = hook_specs(layers)
            elif kind == 'frozen_head_scorer':
                r = await client.get(config['selector_endpoint'] + '/health')
                r.raise_for_status()
                health = r.json()
                if health.get('mode') != 'frozen_head_scorer' or health.get('head_sha256') != plan['condition']['head_sha256']:
                    raise ValueError('Wrong frozen-head scorer service')
                services['scorer'] = health
            elif kind == 'selection':
                r = await client.get(config['selector_endpoint'] + '/health')
                r.raise_for_status()
                health = r.json()
                if health.get('mode') != 'selection' or health.get('model') != str(PIOTR_SELECTION) or health.get('adapter'):
                    raise ValueError('Wrong selector service')
                services['selector'] = health
        write_json(root / 'services.json', services)
        args = evaluation.EvalArgs(**eval_args_kwargs(plan, config['actor_port'], root))
        args.rollout_temperature = plan['protocol']['temperature']
        args.rollout_max_response_len = plan['protocol']['max_new_tokens']
        args.browser_history_full_turns = (plan.get('horizon') or {}).get('history_full_turns', 0)   # read via getattr by the generator
        init_http_client(args)
        return args

    async def evaluate(self, args, task, sampling):
        from openwebrl import run_evaluate
        return await run_evaluate.evaluate_single_task(args, task, sampling, True)


def install_judge(evaluation, reward, judge):
    """Route the eval loop to the benchmark's native reward, whose OpenAI client is the capped judge."""
    evaluation.reward_func = reward.reward_func
    reward._get_openai_client = lambda **unused: judge


def eval_args_kwargs(plan, actor_port, root):
    route = route_for(plan['benchmark'])
    p = plan['protocol']
    return dict(sglang_router_ip='127.0.0.1', sglang_router_port=actor_port, hf_checkpoint=plan['actor'],
                max_steps=p['max_steps'], context_num_screenshots=p['context_num_screenshots'],
                rollout_max_context_len=p['context_length'],
                judge_api_model=route['model'], judge_max_attached_imgs=route['images'], judge_api_mode='served',
                judge_timeout_secs=180, browser_response_format_mode='browser_env', turn_history_reasoning_mode='full',
                browser_include_tool_response=1, inference_step_timeout_secs=180,
                task_timeout_secs=1800 if p['max_steps'] <= 30 else 1800 * p['max_steps'] // 30,
                path_to_save_generated_samples=str(Path(root) / 'samples'))


def make_selector(plan, config, directory):
    cond = plan['condition']
    head = None
    if cond['kind'] == 'actor_cache_head':
        head = HeadBackend(cond['head_path'], plan['actor'], config['capture_dir'], params=actor_info(plan['actor'])['params'])
    return CrossbenchSelector(cond['kind'], Path(directory) / 'selections', seed=plan['protocol']['seed'],
                              candidates=cond['candidates'], endpoint=config.get('selector_endpoint'), head=head)


async def run_collection(config, *, runtime=None, selector_factory=make_selector, max_physical_attempts=3,
                         startup_gate=2):
    """Run one benchmark of one condition; resumable; halts on the judge cap. Returns the summary."""
    runtime = runtime or LiveRuntime()
    plan = validate_plan(config['plan_dir'], allow_source_drift=config.get('allow_source_drift', False))
    root = Path(plan['collection'])
    root.mkdir(parents=True, exist_ok=True)
    caps = plan['judge_caps']
    if (Path(plan['judge_budget_root']) / 'halt.json').exists():
        raise JudgeHalted('Judge ledger already halted: ' + plan['judge_budget_root'])
    judge = NativeCappedJudge(plan['judge_budget_root'], plan['benchmark'], max_calls=caps['max_calls'],
                              max_usd=caps['max_usd'], attempt_context=ATTEMPT, client=runtime.judge_client())
    tasks = runtime.load_tasks(plan['task_file'])
    missing = [t for t in plan['task_ids'] if t not in tasks]
    if missing:
        raise ValueError(f'Tasks missing from frozen file: {missing[:3]}')
    system_policy = None
    if config.get('check_system_policy', True):
        system_policy = (REPO / 'openwebrl/env/prompts/system_prompt_browser_env.md').read_text()
    base_args = await runtime.setup(config, plan, judge, root)
    p = plan['protocol']
    sem = asyncio.Semaphore(p['browser_concurrency'])
    stop = asyncio.Event()

    async def one(task_id):
        key = plan['keys'][task_id]
        record_path = root / 'records' / (key + '.json')
        if record_path.exists():
            verify_record(read_json(record_path), plan, task_id, root)
            return 'skipped'
        async with sem:
            if stop.is_set():
                return 'stopped'
            if judge.halted:
                stop.set()
                raise JudgeHalted('Judge cap/contract halt')
            previous = sorted((root / 'attempts' / key).glob('*')) if (root / 'attempts' / key).exists() else []
            if len(previous) >= max_physical_attempts:
                raise RuntimeError(f'{task_id}: {len(previous)} interrupted physical attempts; diagnose before retrying')
            directory = root / 'attempts' / key / str(time.time_ns())
            directory.mkdir(parents=True)
            selector = MeteredActor(selector_factory(plan, config, directory), directory, system_policy=system_policy,
                                    post=config.get('_post'))
            local = copy(base_args)
            local.browser_action_selector = selector
            local.path_to_save_generated_samples = str(directory / 'samples')
            started, started_epoch = time.monotonic(), time.time()
            token = ATTEMPT.set(key)
            error = None
            try:
                sampling = dict(p['sampling_request'], sampling_seed=p['seed'])
                result = await runtime.evaluate(local, tasks[task_id], sampling)
                valid = is_valid_result(result)
            except Exception as exc:
                error = type(exc).__name__
                valid = False
                result = dict(task_id=task_id, reward=None, valid=False, error_type=error, error=str(exc))
            finally:
                ATTEMPT.reset(token)
            artifact = directory / 'result.json'
            await asyncio.to_thread(write_json, artifact, result)
            if judge.halted:
                # The cap (not the agent) may have decided this outcome: leave it uncommitted and stop.
                stop.set()
                raise JudgeHalted(f'Judge cap/contract halt during {task_id}; attempt preserved at {directory}')
            traces, steps = read_traces(directory / 'selections')
            telemetry = [s.get('selector_telemetry') or {} for s in steps if s['kind'] != 'baseline']
            reward_meta = (result.get('metadata') or {}).get('reward') or {}
            record = dict(
                key=key, task_id=task_id, benchmark=plan['benchmark'], condition=plan['condition']['name'],
                condition_slug=plan['condition']['slug'], head_sha256=plan['condition']['head_sha256'],
                seed=p['seed'], protocol_sha256=plan['protocol_sha256'], physical_attempt=len(previous) + 1,
                valid=valid, reward=result.get('reward') if valid else None, error_type=error,
                status=result.get('status'), terminate_reason=result.get('terminate_reason'),
                judge_model=route_for(plan['benchmark'])['model'], judge_protocol=route_for(plan['benchmark'])['protocol'],
                judge_text=reward_meta.get('judge_text'), judge_timeout=bool(reward_meta.get('judge_timeout')),
                artifact=str(artifact), artifact_sha256=sha256_file(artifact),
                trajectory=dict(result=str(artifact), samples=str(directory / 'samples'),
                                actor_requests=str(directory / 'actor-requests')),
                trace_files=traces, browser_steps=len(steps), total_steps=result.get('total_steps'),
                selector_picks=[s['selected_index'] for s in steps],
                selector_scores=[s['scores'] for s in steps] if plan['condition']['kind'] == 'actor_cache_head' else None,
                selector_raw=[s['selector_raw'] for s in steps] if plan['condition']['kind'] == 'selection' else None,
                selector_calls=len(telemetry), selector_fallbacks=sum(bool(s.get('fallback')) for s in steps),
                selector_input_tokens=sum(t.get('input_tokens') or 0 for t in telemetry),
                selector_output_tokens=sum(t.get('output_tokens') or 0 for t in telemetry),
                selector_service_seconds=sum(t.get('seconds') or 0. for t in telemetry),
                decision_seconds=sum(s['timing']['decision_seconds'] for s in steps),
                generation_seconds=sum(s['timing']['generation_seconds'] for s in steps),
                selection_seconds=sum(s['timing']['selection_seconds'] for s in steps),
                actor_requests=selector.requests, actor_output_tokens=selector.output_tokens,
                actor_input_tokens=selector.input_tokens, actor_client_request_seconds=selector.client_seconds,
                started_epoch=started_epoch, elapsed_seconds=time.monotonic() - started,
                scheduler_job_id=config.get('scheduler_job_id'))
            verify_record(record, plan, task_id, root)
            await asyncio.to_thread(write_json, record_path, record)
            print('CROSSBENCH_ATTEMPT ' + json.dumps({k: record[k] for k in (
                'benchmark', 'condition', 'task_id', 'valid', 'reward', 'browser_steps', 'elapsed_seconds')}), flush=True)
            return 'done'

    try:
        ids = plan['task_ids']
        gate = ids[:startup_gate]
        await asyncio.gather(*(one(t) for t in gate))
        if gate:
            pilot = [read_json(root / 'records' / (plan['keys'][t] + '.json')) for t in gate]
            needs_selector = plan['condition']['kind'] != 'baseline'
            if not any(r['valid'] and r['browser_steps'] > 0 and (r['selector_calls'] > 0 or not needs_selector) for r in pilot):
                raise RuntimeError('Startup gate: no functioning actor/selector/browser/judge episode in the first tasks')
            write_json(root / 'startup-gate.json', dict(passed=True, tasks=gate))
        results = await asyncio.gather(*(one(t) for t in ids[startup_gate:]), return_exceptions=True)
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            raise next((f for f in failures if isinstance(f, JudgeHalted)), failures[0])
    finally:
        summary = summarize(load_records(plan), plan, judge_state(plan))
        write_json(root / 'summary.json', summary)
        await judge.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description='Crossbench selector worker (one benchmark; run inside the job).')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    # Preserve the established event-loop-policy initialization order (import run_evaluate first).
    from openwebrl import run_evaluate  # noqa: F401
    summary = asyncio.run(run_collection(read_json(args.config)))
    print('CROSSBENCH_SUMMARY ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
