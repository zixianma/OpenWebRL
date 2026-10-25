"""CPU-only tests for the crossbench selector runner: mocked actor, selector, head, judge and browser episodes."""
import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from openwebrl import crossbench_selector_eval as core

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / 'openwebrl/env/prompts/system_prompt_browser_env.md'


def write_tasks(path, n, shuffle=False):
    rows = [dict(ques=f'q{i}', web='https://x.test/', metadata=dict(task_id=f't{i:03d}', intent=f'q{i}',
                                                                      start_url='https://x.test/')) for i in range(n)]
    if shuffle:
        rows = rows[::-1]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='xbench-test-'))
        self.repo = self.tmp / 'repo'                       # fake source tree: digests and policy live here
        (self.repo / POLICY.relative_to(REPO)).parent.mkdir(parents=True)
        shutil.copy(POLICY, self.repo / POLICY.relative_to(REPO))
        self.actor = self.tmp / 'actor'
        self.actor.mkdir()
        (self.actor / 'config.json').write_text('{}')
        self.tasks = write_tasks(self.tmp / 'tasks.jsonl', 20)
        self.head = self.tmp / 'head.pt'
        self.head.write_bytes(b'fake-head-weights')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def plan(self, condition='sft1', benchmark='online_mind2web', n=4, seed=7, caps=None, name='cohort'):
        return core.build_plan(self.tmp / name, benchmark=benchmark, condition=condition, n=n, seed=seed,
                               actor=self.actor, task_source=self.tasks, repo=self.repo,
                               judge_caps=caps or dict(max_calls=100, max_usd=100.))

    def validate(self, plan, **kw):
        return core.validate_plan(plan['root'], repo=self.repo, **kw)


class PlanFreezing(Fixture):
    def test_plan_records_identity_and_is_immutable(self):
        plan = self.plan(condition=f'head5:{self.head}')
        self.assertEqual(plan['condition']['head_sha256'], core.sha256_file(self.head))
        self.assertTrue(plan['condition']['slug'].startswith('head5-'))
        self.assertEqual(plan['protocol']['judge'], 'gpt-5.6-terra')
        self.assertEqual((plan['protocol']['temperature'], plan['protocol']['top_p'], plan['protocol']['max_new_tokens'],
                          plan['protocol']['max_steps'], plan['protocol']['seed'], plan['protocol']['server_seed']),
                         (.7, .9, 1024, 30, 42, 4200))
        self.assertEqual(len(plan['task_ids']), 4)
        self.assertEqual(self.plan(condition=f'head5:{self.head}'), plan)          # idempotent
        with self.assertRaises(ValueError):
            self.plan(condition=f'head5:{self.head}', seed=8)                       # different subset refused
        self.validate(plan)

    def test_validation_detects_changed_inputs(self):
        plan = self.plan(condition=f'head5:{self.head}')
        self.head.write_bytes(b'other weights')
        with self.assertRaisesRegex(ValueError, 'head'):
            self.validate(plan)
        plan = self.plan(name='other')
        Path(plan['task_file']).write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'task file'):
            self.validate(plan)

    def test_source_drift_fails_closed_unless_allowed(self):
        plan = self.plan()
        target = self.repo / 'openwebrl/generate_browser.py'
        target.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'generate_browser'):
            self.validate(plan)
        self.validate(plan, allow_source_drift=True)

    def test_unknown_condition_and_missing_head(self):
        with self.assertRaises(ValueError):
            core.parse_condition('luna5')
        with self.assertRaises(ValueError):
            core.parse_condition(f'head5:{self.tmp}/missing.pt')
        for bad in ('piotr1', 'piotr6', 'head6:x.pt'):
            with self.assertRaises(ValueError):
                core.parse_condition(bad)

    def test_candidate_count_conditions(self):
        head = Path(self.tmp) / 'head.pt'
        head.write_bytes(b'head')
        five, three = core.parse_condition(f'head5:{head}'), core.parse_condition(f'head3:{head}')
        self.assertEqual((five['name'], five['candidates'], five['kind']), ('head5', 5, 'actor_cache_head'))
        self.assertEqual((three['name'], three['candidates'], three['kind']), ('head3', 3, 'actor_cache_head'))
        self.assertEqual(three['slug'], 'head3-' + five['slug'].split('-', 1)[1])
        x = core.parse_condition(f'head3x:{head}')
        self.assertEqual((x['name'], x['candidates'], x['kind']), ('head3x', 3, 'frozen_head_scorer'))
        p5, p3 = core.parse_condition('piotr5'), core.parse_condition('piotr3')
        self.assertEqual((p5['slug'], p5['candidates']), ('piotr5', 5))
        self.assertEqual((p3['slug'], p3['candidates'], p3['kind']), ('piotr3', 3, 'selection'))
        self.assertEqual(p3['selector_model'], p5['selector_model'])

    def test_8b_actor_registered(self):
        info = core.actor_info(core.ACTOR_8B)
        self.assertEqual(info['revision'], '838c4a4842555a2a61e81d7a28bac7b2e8144800')
        self.assertEqual(info['params'], 8767123696)
        self.assertEqual(core.actor_info()['params'], core.ACTOR_PARAMS)


class SubsetSampling(unittest.TestCase):
    def test_deterministic_and_order_independent(self):
        ids = [f't{i}' for i in range(300)]
        a = core.sample_subset(ids, 100, 3)
        self.assertEqual(a, core.sample_subset(list(reversed(ids)), 100, 3))
        self.assertEqual(len(set(a)), 100)
        self.assertNotEqual(a, core.sample_subset(ids, 100, 4))
        self.assertEqual(a[:10], core.sample_subset(ids, 100, 3)[:10])
        with self.assertRaises(ValueError):
            core.sample_subset(ids, 301, 3)
        with self.assertRaises(ValueError):
            core.sample_subset(ids + ['t0'], 10, 3)

    def test_real_task_sources_are_pinned(self):
        for bench, source in core.TASK_SOURCES.items():
            if not Path(source['path']).exists():
                self.skipTest(f'{source["path"]} not available')
            self.assertEqual(core.sha256_file(source['path']), source['sha256'], bench)
            ids = core.read_task_ids(source['path'])
            self.assertEqual(len(ids), source['count'])
            self.assertEqual(len(set(ids)), len(ids))
            self.assertEqual(core.sample_subset(ids, 100, 0), core.sample_subset(ids, 100, 0))

    def test_task_identity_matches_eval_loader_rule(self):
        self.assertEqual(core.task_id_of(dict(metadata=dict(task_id='a'), id='b'), 0), 'a')
        self.assertEqual(core.task_id_of(dict(metadata=None, id='b'), 0), 'b')
        self.assertEqual(core.task_id_of(dict(), 5), 'task_5')


# --- fake OpenAI judge client ---------------------------------------------------------------------------------------
class FakeReply:
    def __init__(self, model, text='Status: success'):
        self.model = model
        self.usage = SimpleNamespace(model_dump=lambda: dict(prompt_tokens=1000, completion_tokens=50))
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=text, parsed=None, refusal=None))]

    def model_dump(self):
        return dict(model=self.model)


class FakeJudgeClient:
    def __init__(self, model):
        self.calls = []

        async def create(**kw):
            self.calls.append(('create', kw))
            return FakeReply(model + '-2025-01-01')

        async def parse(**kw):
            self.calls.append(('parse', kw))
            return FakeReply(model)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))

    async def close(self):
        pass


MESSAGES = [dict(role='user', content=[dict(type='text', text='judge this'), dict(type='image_url', image_url='x')])]


class JudgeRouting(Fixture):
    def test_routes_per_benchmark(self):
        expected = {'online_mind2web': ('gpt-5.6-terra', 3, 'create', 'reward_online_mind2web'),
                    'webvoyager': ('gpt-4o', 30, 'create', 'reward_webvoyager'),
                    'deepshop': ('gpt-4o', 30, 'parse', 'reward_deepshop')}
        for bench, (model, images, api, module) in expected.items():
            route = core.route_for(bench)
            self.assertEqual((route['model'], route['images'], route['api'], route['module']), (model, images, api, module))
            plan = self.plan(benchmark=bench, name=bench)
            self.assertEqual((plan['protocol']['judge'], plan['protocol']['judge_max_attached_imgs'],
                              plan['protocol']['judge_response_tokens']), (model, images, 4096))
            kwargs = core.eval_args_kwargs(plan, 30000, self.tmp)
            self.assertEqual((kwargs['judge_api_model'], kwargs['judge_max_attached_imgs'], kwargs['max_steps'],
                              kwargs['context_num_screenshots']), (model, images, 30, 1))
        with self.assertRaises(ValueError):
            core.route_for('webarena')

    def test_install_judge_patches_native_reward(self):
        evaluation, reward = SimpleNamespace(reward_func=None), SimpleNamespace(reward_func=object(), _get_openai_client=None)
        judge = object()
        core.install_judge(evaluation, reward, judge)
        self.assertIs(evaluation.reward_func, reward.reward_func)
        self.assertIs(reward._get_openai_client(method='served'), judge)

    def test_capped_judge_enforces_route_model_and_output_cap(self):
        async def go():
            om2w = core.NativeCappedJudge(self.tmp / 'j1', 'online_mind2web', max_calls=5, max_usd=5.,
                                          attempt_context=core.ATTEMPT, client=FakeJudgeClient('gpt-5.6-terra'))
            await om2w.chat.completions.create(model='gpt-5.6-terra', messages=MESSAGES, seed=42)
            sent = om2w.client.calls[0][1]
            self.assertEqual((sent['max_completion_tokens'], sent['store']), (4096, False))
            with self.assertRaises(ValueError):
                await om2w.chat.completions.create(model='gpt-4o', messages=MESSAGES)
            with self.assertRaises(ValueError):
                await om2w.beta.chat.completions.parse(model='gpt-5.6-terra', messages=MESSAGES)
            with self.assertRaises(ValueError):
                await om2w.chat.completions.create(model='gpt-5.6-terra', messages=MESSAGES, max_tokens=10)
            ds = core.NativeCappedJudge(self.tmp / 'j2', 'deepshop', max_calls=5, max_usd=5.,
                                        attempt_context=core.ATTEMPT, client=FakeJudgeClient('gpt-4o'))
            await ds.beta.chat.completions.parse(model='gpt-4o', messages=MESSAGES)
            self.assertEqual(ds.client.calls[0][0], 'parse')
            state = core.read_json(self.tmp / 'j1' / 'ledger.json')
            self.assertEqual(state['calls'], 1)
            self.assertAlmostEqual(state['charged_or_reserved_usd'], (1000 * 2.5 + 50 * 15.) / 1e6)
        asyncio.run(go())

    def test_cap_halts_and_persists(self):
        async def go():
            judge = core.NativeCappedJudge(self.tmp / 'j', 'webvoyager', max_calls=1, max_usd=5.,
                                           attempt_context=core.ATTEMPT, client=FakeJudgeClient('gpt-4o'))
            await judge.chat.completions.create(model='gpt-4o', messages=MESSAGES)
            with self.assertRaisesRegex(ValueError, 'cap'):
                await judge.chat.completions.create(model='gpt-4o', messages=MESSAGES)
            self.assertTrue(judge.halted)
            again = core.NativeCappedJudge(self.tmp / 'j', 'webvoyager', max_calls=1, max_usd=5.,
                                           attempt_context=core.ATTEMPT, client=FakeJudgeClient('gpt-4o'))
            with self.assertRaisesRegex(ValueError, 'halted'):
                await again.chat.completions.create(model='gpt-4o', messages=MESSAGES)
            usd = core.NativeCappedJudge(self.tmp / 'k', 'webvoyager', max_calls=10, max_usd=0.01,
                                         attempt_context=core.ATTEMPT, client=FakeJudgeClient('gpt-4o'))
            with self.assertRaisesRegex(ValueError, 'cap'):            # 20k-token image reservation exceeds $0.01
                await usd.chat.completions.create(model='gpt-4o', messages=MESSAGES)
        asyncio.run(go())


# --- fake actor / selector / head -------------------------------------------------------------------------------------
def actor_output(j):
    return (f'<think>t{j}</think><tool_call>{{"name": "click", "arguments": {{"point_2d": [{j}, {j}]}}}}</tool_call>',
            [j, 151645], [-.1, -.1], 'stop')


def step_kwargs(policy='POLICY'):
    return dict(url='http://actor/generate', input_text=policy + ' prompt', sampling_params=dict(temperature=.7),
                images=['img'], observation=dict(screenshot=b'png', active_tab_url='https://x.test/'),
                history=[], task='do it', task_id='t000', turn=0, timeout=180)


class FakeHead:
    def __init__(self, scores):
        self.scores, self.rids = scores, []

    async def generate(self, call, url, text, images, rid):
        self.rids.append(rid)
        return await call, dict(gen=rid), None

    async def score(self, gens, metas, rids):
        return dict(scores=self.scores, input_tokens=0, output_tokens=5, seconds=.01, flops=1)


class SelectorDispatch(Fixture):
    def run_selector(self, selector):
        calls = []

        async def infer(url, text, params, images, timeout_secs=None, rid=None):
            calls.append(dict(params=params, rid=rid))
            return actor_output(len(calls) - 1)
        out, meta = asyncio.run(selector(infer=infer, **step_kwargs()))
        return out, meta, calls, [json.loads(l) for p in selector.output.glob('*.jsonl') for l in p.read_text().splitlines()]

    def test_baseline_samples_once(self):
        out, meta, calls, trace = self.run_selector(core.CrossbenchSelector('baseline', self.tmp / 'b', seed=42, candidates=5))
        self.assertEqual(len(calls), 1)
        self.assertEqual(meta['selected_index'], 0)
        self.assertIsNone(calls[0]['rid'])
        self.assertEqual(calls[0]['params']['sampling_seed'], core.candidate_seed(42, 't000', 0, 0))
        self.assertEqual(trace[0]['kind'], 'baseline')

    def test_selection_uses_select_contract_and_fallback(self):
        payloads = []

        async def request(endpoint, payload, timeout):
            payloads.append((endpoint, payload))
            return dict(raw='{"selection": 3}', input_tokens=10, output_tokens=4, seconds=.2)
        out, meta, calls, trace = self.run_selector(core.CrossbenchSelector(
            'selection', self.tmp / 's', seed=42, candidates=5, endpoint='http://sel/', request_fn=request))
        self.assertEqual(len(calls), 5)
        self.assertEqual(meta['selected_index'], 2)
        self.assertEqual(out, actor_output(2))
        endpoint, payload = payloads[0]
        self.assertEqual(endpoint, 'http://sel')
        self.assertEqual(payload['mode'], 'selection')
        self.assertEqual(len(payload['candidates']), 5)
        self.assertEqual(payload['candidates'][1]['thought'], 't1')
        self.assertEqual(trace[0]['selector_telemetry']['input_tokens'], 10)

        async def bad(endpoint, payload, timeout):
            return dict(raw='no json', input_tokens=1, output_tokens=1, seconds=.1)
        _, meta, _, trace = self.run_selector(core.CrossbenchSelector(
            'selection', self.tmp / 's2', seed=42, candidates=5, endpoint='http://sel', request_fn=bad))
        self.assertEqual(meta['selected_index'], 0)
        self.assertTrue(trace[0]['fallback'])

    def test_head_scores_cached_candidates_argmax_ties_first(self):
        head = FakeHead([0.1, 0.9, 0.3, 0.9, -1.0])
        out, meta, calls, trace = self.run_selector(core.CrossbenchSelector(
            'actor_cache_head', self.tmp / 'h', seed=42, candidates=5, head=head))
        self.assertEqual(meta['selected_index'], 1)
        self.assertTrue(all(c['rid'] and c['rid'].startswith('actorhead-') for c in calls))
        self.assertEqual(len(set(c['rid'] for c in calls)), 5)
        self.assertEqual(trace[0]['scores'], head.scores)
        with self.assertRaises(ValueError):
            self.run_selector(core.CrossbenchSelector('actor_cache_head', self.tmp / 'h2', seed=42, candidates=5,
                                                      head=FakeHead([0., float('nan'), 0., 0., 0.])))

    def test_make_selector_per_condition(self):
        config = dict(selector_endpoint='http://127.0.0.1:1', capture_dir='/dev/shm/x')
        for condition, kind, k in (('sft1', 'baseline', 1), ('piotr5', 'selection', 5), (f'head5:{self.head}', 'actor_cache_head', 5)):
            plan = self.plan(condition=condition, name=condition.split(':')[0])
            selector = core.make_selector(plan, config, self.tmp / 'att' / kind)
            self.assertEqual((selector.kind, selector.candidates), (kind, k))
            if kind == 'actor_cache_head':
                self.assertEqual((selector.head.head_path, selector.head.capture_dir, selector.head.tokenizer_path),
                                 (str(self.head.resolve()), '/dev/shm/x', str(self.actor.resolve())))

    def test_metered_actor_records_receipts_and_checks_policy(self):
        async def post(url, payload):
            return dict(text='out', meta_info=dict(output_token_logprobs=[[-.1, 5], [-.2, 6]], prompt_tokens=100,
                                                   completion_tokens=2, finish_reason=dict(type='stop')))
        selector = core.CrossbenchSelector('actor_cache_head', self.tmp / 'm' / 'selections', seed=1, candidates=5,
                                           head=FakeHead([1, 0, 0, 0, 0]))
        metered = core.MeteredActor(selector, self.tmp / 'm', system_policy='POLICY', post=post)
        asyncio.run(metered(infer=None, **step_kwargs()))
        self.assertEqual((metered.requests, metered.input_tokens, metered.output_tokens), (5, 500, 10))
        receipt = core.read_json(self.tmp / 'm' / 'actor-requests' / '00000.json')
        self.assertTrue(receipt['rid'].startswith('actorhead-'))
        with self.assertRaisesRegex(ValueError, 'policy'):
            asyncio.run(metered(infer=None, **step_kwargs(policy='OTHER')))


# --- worker with a fake runtime ----------------------------------------------------------------------------------------
class FakeRuntime:
    """Episodes of `steps` decisions, then one native-judge call through the capped judge."""

    def __init__(self, model, steps=2, verdicts=None, judge=True):
        self.model, self.steps, self.verdicts, self.judge_on = model, steps, verdicts or {}, judge
        self.evaluated = []
        self.judge = None

    def judge_client(self):
        return FakeJudgeClient(self.model)

    def load_tasks(self, task_file):
        return {tid: dict(task_id=tid) for tid in core.read_task_ids(task_file)}

    async def setup(self, config, plan, judge, root):
        self.judge = judge
        self.plan = plan
        return SimpleNamespace(path_to_save_generated_samples=str(root / 'samples'))

    async def evaluate(self, args, task, sampling):
        self.evaluated.append(task['task_id'])
        for turn in range(self.steps):
            kw = step_kwargs(policy=POLICY.read_text())
            kw.update(task_id=task['task_id'], turn=turn)
            await args.browser_action_selector(infer=None, **kw)
        if self.judge_on:
            try:
                api = self.judge.beta.chat.completions.parse if self.judge.route['api'] == 'parse' else self.judge.chat.completions.create
                await api(model=self.judge.route['model'], messages=MESSAGES)
            except ValueError:
                return dict(task_id=task['task_id'], reward=None, status='Status.COMPLETED', metadata={})
        reward = self.verdicts.get(task['task_id'], 1.0)
        return dict(task_id=task['task_id'], reward=reward, status='Status.COMPLETED', total_steps=self.steps,
                    metadata=dict(reward=dict(judge_text='verdict', judge_timeout=False)))


async def fake_post(url, payload):
    j = payload['sampling_params']['sampling_seed'] % 5
    text, tokens, logps, _ = actor_output(j)
    return dict(text=text, meta_info=dict(output_token_logprobs=[[lp, t] for lp, t in zip(logps, tokens)], prompt_tokens=50,
                                          completion_tokens=len(tokens), finish_reason=dict(type='stop')))


def fake_selector_factory(plan, config, directory):
    async def request(endpoint, payload, timeout):
        return dict(raw='{"selection": 2}', input_tokens=7, output_tokens=3, seconds=.05)
    cond = plan['condition']
    head = FakeHead([0, 0, 0, 1, 0]) if cond['kind'] == 'actor_cache_head' else None
    return core.CrossbenchSelector(cond['kind'], Path(directory) / 'selections', seed=42, candidates=cond['candidates'],
                                   endpoint='http://sel', head=head, request_fn=request)


class Worker(Fixture):
    def config(self, plan):
        return dict(plan_dir=plan['root'], actor_port=1, browser_port_start=2, _post=fake_post,
                    allow_source_drift=True)       # fake repo digests vs the module's own repo

    def collect(self, plan, runtime, **kw):
        with mock.patch.object(core, 'source_drift', lambda plan, repo=None: {}):
            return asyncio.run(core.run_collection(self.config(plan), runtime=runtime,
                                                   selector_factory=fake_selector_factory, **kw))

    def test_full_run_records_and_summary(self):
        for condition, picks in (('sft1', 0), ('piotr5', 1), (f'head5:{self.head}', 3)):
            plan = self.plan(condition=condition, n=4, name=condition.split(':')[0])
            runtime = FakeRuntime('gpt-5.6-terra', verdicts={plan['task_ids'][0]: 0.0})
            summary = self.collect(plan, runtime)
            self.assertEqual((summary['completed'], summary['successes'], summary['valid']), (4, 3, 4))
            self.assertEqual(summary['success_rate_scheduled'], .75)
            self.assertEqual(summary['judge_calls'], 4)
            self.assertGreater(summary['judge_usd'], 0)
            record = core.load_records(plan)[0]
            self.assertEqual(record['selector_picks'], [picks, picks])
            self.assertEqual(record['browser_steps'], 2)
            self.assertEqual(record['actor_requests'], 2 * plan['condition']['candidates'])
            self.assertEqual(record['selector_calls'], 0 if condition == 'sft1' else 2)
            self.assertTrue(Path(record['trajectory']['result']).exists())
            self.assertEqual(record['head_sha256'], plan['condition']['head_sha256'])
            self.assertTrue((Path(plan['collection']) / 'startup-gate.json').exists())

    def test_resume_skips_completed_and_detects_tampering(self):
        plan = self.plan(n=5)
        self.collect(plan, FakeRuntime('gpt-5.6-terra'))
        # drop one record (its attempt directory stays): only that task is re-run, as physical attempt 2
        key = plan['keys'][plan['task_ids'][3]]
        record_path = Path(plan['collection']) / 'records' / f'{key}.json'
        record_path.rename(self.tmp / 'moved.json')
        runtime = FakeRuntime('gpt-5.6-terra')
        summary = self.collect(plan, runtime)
        self.assertEqual(runtime.evaluated, [plan['task_ids'][3]])
        self.assertEqual(summary['completed'], 5)
        self.assertEqual(core.read_json(record_path)['physical_attempt'], 2)
        self.assertEqual(len(list((Path(plan['collection']) / 'attempts' / key).iterdir())), 2)
        runtime = FakeRuntime('gpt-5.6-terra')
        self.collect(plan, runtime)
        self.assertEqual(runtime.evaluated, [])
        Path(core.read_json(record_path)['artifact']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.collect(plan, FakeRuntime('gpt-5.6-terra'))

    def test_judge_cap_halts_without_committing(self):
        plan = self.plan(n=6, caps=dict(max_calls=3, max_usd=100.))
        with self.assertRaises(core.JudgeHalted):
            self.collect(plan, FakeRuntime('gpt-5.6-terra'))
        records = core.load_records(plan)
        # only episodes whose judge call succeeded and that finished before the halt are committed; none is capped
        self.assertIn(len(records), (2, 3))
        self.assertTrue(all(r['valid'] for r in records))
        state = core.judge_state(plan)
        self.assertEqual((state['calls'], state['halted']), (3, True))
        summary = core.read_json(Path(plan['collection']) / 'summary.json')
        self.assertTrue(summary['judge_halted'])
        runtime = FakeRuntime('gpt-5.6-terra')
        with self.assertRaises(core.JudgeHalted):       # a halted ledger refuses to start again
            self.collect(plan, runtime)
        self.assertEqual(runtime.evaluated, [])

    def test_startup_gate_requires_functioning_episode(self):
        plan = self.plan(condition='piotr5', n=4)
        with self.assertRaisesRegex(RuntimeError, 'Startup gate'):
            self.collect(plan, FakeRuntime('gpt-5.6-terra', steps=0))

    def test_episode_exception_is_committed_invalid(self):
        plan = self.plan(n=3)

        class Crashy(FakeRuntime):
            async def evaluate(self, args, task, sampling):
                if task['task_id'] == plan['task_ids'][2]:
                    raise TimeoutError('browser died')
                return await super().evaluate(args, task, sampling)
        summary = self.collect(plan, Crashy('gpt-5.6-terra'))
        self.assertEqual((summary['completed'], summary['valid'], summary['errors']), (3, 2, 1))


class ValidityRule(unittest.TestCase):
    def test_sol61_validity(self):
        self.assertTrue(core.is_valid_result(dict(reward=1.0, status='Status.COMPLETED')))
        self.assertTrue(core.is_valid_result(dict(reward=0, status='Status.TRUNCATED')))
        self.assertFalse(core.is_valid_result(dict(reward=None, status='Status.COMPLETED')))
        self.assertFalse(core.is_valid_result(dict(reward=1, status='Status.ABORTED')))
        self.assertFalse(core.is_valid_result(dict(reward=1, status='x', metadata=dict(judge_timeout=True))))


class Cli(Fixture):
    def setUp(self):
        super().setUp()
        import importlib.util
        spec = importlib.util.spec_from_file_location('xbench_cli', REPO / 'scripts/run_crossbench_selector_eval.py')
        self.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cli)

    def test_sbatch_one_condition_sequential_benchmarks(self):
        text = self.cli.sbatch_text(self.tmp / 'head5-abc', ['online_mind2web', 'webvoyager'], time_limit='05:30:00',
                                    account='vlm', partition='gpu-h200', qos='normal', job_name='xbench-head5-abc')
        for line in ('--account=vlm', '--partition=gpu-h200', '--qos=normal', '--gpus=h200:1', '--cpus-per-task=8',
                     '--mem=120G', '--time=05:30:00', 'execute --dir', 'online_mind2web webvoyager'):
            self.assertIn(line, text)
        self.assertEqual(self.cli.default_time('head5', ['a', 'b', 'c']), '08:00:00')
        self.assertEqual(self.cli.default_time('sft1', ['a']), '02:00:00')

    def test_server_stages_per_condition(self):
        with mock.patch.object(self.cli, 'hook_specs_json', lambda layers=(18, 27, 36): '[{"name": "actorhead-18"}]'):
            sft = self.plan(name='s')
            stages = self.cli.server_stages(sft, 20000, '/dev/shm/c')
            self.assertEqual([[s['name'] for s in st] for st in stages], [['actor']])
            self.assertNotIn('--forward-hooks', stages[0][0]['argv'][-1])
            self.assertIn('--random-seed 4200', stages[0][0]['argv'][-1])
            self.assertIn('--mem-fraction-static .45', stages[0][0]['argv'][-1])
            head = self.plan(condition=f'head5:{self.head}', name='h')
            stages = self.cli.server_stages(head, 20000, '/dev/shm/c')
            self.assertIn('--forward-hooks', stages[0][0]['argv'][-1])
            self.assertEqual(stages[0][0]['env']['ACTORHEAD_DIR'], '/dev/shm/c')
            self.assertEqual(stages[0][0]['env']['ACTORHEAD_HEAD'], str(self.head.resolve()))
            piotr = self.plan(condition='piotr5', name='p')
            stages = self.cli.server_stages(piotr, 20000, '/dev/shm/c')
            self.assertEqual([[s['name'] for s in st] for st in stages], [['selector-sglang'], ['actor'], ['selector']])
            self.assertIn('--mem-fraction-static .30', stages[0][0]['argv'][-1])
            self.assertFalse(stages[2][0]['gpu'])
            self.assertIn('--port 20001', ' '.join(stages[2][0]['argv']))
            frozen = self.plan(condition=f'head5x:{self.head}', name='x')
            stages = self.cli.server_stages(frozen, 20000, '/dev/shm/c')
            self.assertEqual([[s['name'] for s in st] for st in stages], [['actor'], ['scorer']])
            self.assertNotIn('--forward-hooks', stages[0][0]['argv'][-1])
            self.assertTrue(stages[1][0]['gpu'])
            self.assertIn('serve_frozen_head_scorer.py', stages[1][0]['argv'][-1])
            self.assertIn('--port 20001', stages[1][0]['argv'][-1])

    def test_judge_cap_overrides(self):
        caps = self.cli.parse_caps(['webvoyager=20'], ['online_mind2web', 'webvoyager'], 100)
        self.assertEqual(caps['webvoyager'], dict(max_calls=400, max_usd=20.))
        self.assertEqual(caps['online_mind2web'], dict(max_calls=400, max_usd=6.))


if __name__ == '__main__':
    unittest.main()


def test_decoding_override_changes_only_sampling():
    from openwebrl import crossbench_selector_eval as core
    base = core.protocol_for('webvoyager', 'sft1')
    assert (base['temperature'], base['top_p']) == (.7, .9) and base['sampling_request']['temperature'] == .7
    assert core.protocol_for('webvoyager', 'sft1', None) == base
    alt = core.protocol_for('webvoyager', 'sft1', dict(temperature=.6, top_p=.95))
    assert alt['sampling_request'] == dict(temperature=.6, top_p=.95, max_new_tokens=1024)
    assert {k: v for k, v in alt.items() if k not in ('temperature', 'top_p', 'sampling_request')} == \
           {k: v for k, v in base.items() if k not in ('temperature', 'top_p', 'sampling_request')}


def test_horizon_override_and_default_protocol_unchanged():
    from openwebrl import crossbench_selector_eval as core
    base = core.protocol_for('deepshop', 'sft1', dict(temperature=1.0, top_p=.9))
    assert base['max_steps'] == 30 and base['history'] == 'full'
    assert core.protocol_for('deepshop', 'sft1', dict(temperature=1.0, top_p=.9), None) == base
    long = core.protocol_for('deepshop', 'sft1', dict(temperature=1.0, top_p=.9), dict(max_steps=100, history_full_turns=20))
    assert long['max_steps'] == 100 and long['history'].startswith('last 20 turns full')


def test_history_window_keeps_recent_turns_and_task():
    from types import SimpleNamespace
    from openwebrl.generate_browser import _compress_turn_history_messages
    args = SimpleNamespace(browser_history_full_turns=2, browser_history_reasoning_max_turns=0, browser_response_format_mode='browser_env')
    msgs = [dict(role='system', content='sys'), dict(role='user', content=[dict(type='text', text='TASK <observation>first</observation>')])]
    for k in range(4):
        msgs.append(dict(role='assistant', content=f'<think>reason {k}</think>\n<tool_call>{{"name": "click", "k": {k}}}</tool_call>'))
        msgs.append(dict(role='user', content=[dict(type='text', text=f'<tool_response>\nSucceed {k}\n\n<observation>\ntabs {k}\n</observation>\n</tool_response>')]))
    out = _compress_turn_history_messages(msgs, 'full', args)
    assert out[1] == msgs[1]                                         # task message untouched
    assert out[2]['content'].startswith('<tool_call>') and 'reason 0' not in out[2]['content']
    assert 'tabs 0' not in out[3]['content'][0]['text'] and 'Succeed 0' in out[3]['content'][0]['text']
    assert out[6:] == msgs[6:]                                        # last two turns and their feedback unchanged
    short = msgs[:6]
    assert _compress_turn_history_messages(short, 'full', args) == short


def test_actor_registry_keeps_4b_protocol_and_adds_2b():
    from openwebrl import crossbench_selector_eval as core
    base = core.protocol_for('webvoyager', 'sft1', dict(temperature=1.0, top_p=.9))
    assert core.protocol_for('webvoyager', 'sft1', dict(temperature=1.0, top_p=.9), None, str(core.ACTOR)) == base
    assert base['actor'] == 'OpenWebRL-4B-SFT:iteration0'
    two = core.protocol_for('webvoyager', 'sft1', dict(temperature=1.0, top_p=.9), None, str(core.ACTOR_2B))
    assert two['actor'].endswith('qwen3-vl-2b-thinking-openwebrl-sft') and two['actor_revision'].startswith('95d5b13')
    assert core.actor_info(core.ACTOR_2B)['capture_layers'] == (14, 21, 28)
