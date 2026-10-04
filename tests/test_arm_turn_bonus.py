"""Causality, invalid-label fallback, provenance, and native reward identity."""
import asyncio
import ast
from contextlib import asynccontextmanager, nullcontext
from copy import deepcopy
import inspect
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from openwebrl import arm_turn_bonus as bonus

PROMPT = '<tools>\n' + json.dumps(dict(function=dict(name='click', parameters=dict(
    type='object', properties={'x': {'type':'integer'}}, required=['x'])))) + '\n</tools>\n'


def output(i):
    return (f'<think>Reason {i}</think><tool_call>' + json.dumps(
        dict(name='click', arguments={'x':i})) + '</tool_call>', [100+i], [-.3], 'stop')


def sample(selected=None, removed=False):
    label = {} if selected is None else dict(eligible=True, policy_id='p', executed_index=0,
        selected_index=selected, unit_bonus=float(selected == 0)-.2, reason='admitted')
    return SimpleNamespace(metadata={'task_id':'t', 'arm_turn_bonus':label},
                           remove_sample=removed, loss_mask=[1,1])


class Arithmetic(unittest.TestCase):
    def test_real_evaluation_schema_never_excludes_missing_training_intent(self):
        evaluation = [dict(task_id='evaluation-id', task_name='Find store hours',
                           metadata={'intent':'Find store hours'}), {'task_id':'empty-intent'}]
        ids, intents = bonus.exclusion_sets(evaluation)
        self.assertNotIn('', intents)
        self.assertNotIn('', ids)
        self.assertEqual(intents, {'find store hours'})
        task_id, intent = bonus.task_identity({'task_id':'webvoyager/60'})
        self.assertNotIn(task_id, ids)
        self.assertNotIn(intent, intents)
        actual = [json.loads(x) for x in Path('openwebrl/data/eval/online-mind2web.jsonl').read_text().splitlines()]
        ids, intents = bonus.exclusion_sets(actual)
        self.assertEqual(len(ids), 300)
        self.assertGreater(len(intents), 290)
    def test_exact_sparse_scale_and_beta_zero(self):
        rows = [sample(0), sample(1), sample(), sample(0, removed=True)]
        report = bonus.panel(rows, [1.,-1.,1.,-1.], 'p')
        self.assertEqual(report['rows'], 3)
        self.assertEqual(report['admitted'], 2)
        self.assertAlmostEqual(report['variants']['0.5']['bonus_rms'], (.17/3)**.5)
        self.assertEqual(report['variants']['0.0']['bonus_rms'], 0.)
        self.assertEqual(report['selected_original_rate'], .5)
        self.assertEqual(report['variants']['0.5']['sign_flips'], 0)

    def test_provenance_and_stored_value_fail_closed(self):
        row = sample(0)
        with self.assertRaises(ValueError): bonus.unit_bonus(row, 'different-policy')
        row.metadata['arm_turn_bonus']['unit_bonus'] = 1.
        with self.assertRaises(ValueError): bonus.unit_bonus(row, 'p')

    def test_duplicates_schema_errors_and_truncation_rejected(self):
        self.assertEqual(len(bonus.eligible_actions(PROMPT, [output(i) for i in range(5)])), 5)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            bonus.eligible_actions(PROMPT, [output(0)]*5)
        bad = [output(i) for i in range(5)]
        bad[1] = (*bad[1][:3], 'length')
        with self.assertRaisesRegex(ValueError, 'truncated'):
            bonus.eligible_actions(PROMPT, bad)
        bad[1] = ('<tool_call>{"name":"unknown"}</tool_call>', [1], [-1.], 'stop')
        with self.assertRaises(ValueError): bonus.eligible_actions(PROMPT, bad)

    def test_shadow_hook_keeps_native_reward_objects(self):
        import time
        with tempfile.TemporaryDirectory() as folder:
            state = dict(config=dict(output=folder, policy_id='p', checkpoint='frozen'),
                         records=[], started=time.monotonic())
            raw, normalized = [1., 0.], [.707, -.707]
            with patch.object(bonus, 'state', return_value=state), patch.object(bonus, 'log_progress'):
                result = bonus.post_process_rewards(None, [sample(0), sample(1)], raw, normalized)
            self.assertIs(result[0], raw)
            self.assertIs(result[1], normalized)
            report = json.loads((Path(folder)/'calibration.json').read_text())
            self.assertEqual(report['optimizer_updates'], 0)

    def test_excluded_evaluation_overlap_does_not_tune_beta(self):
        row = sample(0)
        row.metadata['arm_calibration_excluded'] = True
        report = bonus.panel([row, sample()], [100., 1.], 'p')
        self.assertEqual(report['rows'], 1)
        self.assertEqual(report['outcome_rms'], 1.)
        self.assertEqual(report['admitted'], 0)


class PreparedSource(unittest.TestCase):
    source = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/reference-arm-turn-bonus-calibration-20260913-v8')

    def test_stop_guard_is_before_training_and_after_durable_batch(self):
        text = (self.source/'train.py').read_text()
        start = text.index('rollout_data_ref = _ray_get_with_actor_retry(')
        stop = text.index('before_optimizer(args, rollout_id)', start)
        self.assertLess(text.index('evict_file_cache(recovery_path, sync=True)', start), stop)
        self.assertLess(stop, text.index('actor_model.async_train(', start))
        self.assertLess(text.index('                break', stop), text.index('actor_model.async_train(', start))

    def test_normalizer_and_backends_are_native_except_report_and_cursor(self):
        manifest = json.loads((self.source/'reference_manifest.json').read_text())
        parent = Path(manifest['arm_turn_bonus_calibration']['parent_source'])
        for name in ['actor.py','model.py','loss.py']:
            path = 'slime/backends/megatron_utils/'+name
            self.assertEqual((parent/path).read_bytes(), (self.source/path).read_bytes())
        path = 'slime/ray/rollout.py'
        def method(source):
            tree = ast.parse((source/path).read_text())
            return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and n.name == '_post_process_rewards')
        native, modified = method(parent), method(self.source)
        class RemoveReport(ast.NodeTransformer):
            def visit_If(self, node):
                if 'OPENWEBRL_ARM_TURN_BONUS_CONFIG' in ast.unparse(node.test):
                    return None
                return self.generic_visit(node)
        modified = RemoveReport().visit(modified)
        self.assertEqual(ast.dump(native), ast.dump(modified))
        text = (self.source/path).read_text()
        start = text.index('    def generate(self, rollout_id):')
        self.assertLess(text.index('data = self._convert_samples_to_train_data(data)', start),
                        text.index('self.data_source.save(rollout_id)', start))


class Selection(unittest.IsolatedAsyncioTestCase):
    async def test_native_pre_action_failures_pass_through_collector_and_filter(self):
        # Execute the preserved native functions, replacing external browser I/O
        # only. This covers ABORTED-without-remove_sample, missed by the old mock.
        from slime.utils.types import Sample
        from slime.rollout.filter_hub.base_types import DynamicFilterOutput
        import time
        parent = Path(json.loads((PreparedSource.source/'reference_manifest.json').read_text())[
            'arm_turn_bonus_calibration']['parent_source'])

        def native_functions(path, names, namespace):
            tree = ast.parse((parent/path).read_text())
            definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and n.name in names]
            self.assertEqual({n.name for n in definitions}, set(names))
            module = ast.Module(body=[ast.ImportFrom(module='__future__',
                names=[ast.alias(name='annotations')], level=0), *definitions], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(parent/path), 'exec'), namespace)

        @asynccontextmanager
        async def browser_slot(*unused):
            yield

        for mode in ('initialize_timeout', 'reset_timeout', 'outer_task_timeout', 'no_generated_turns'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                async def reset():
                    if mode == 'reset_timeout':
                        raise TimeoutError()
                    return {}, {}
                env = SimpleNamespace(reset=reset)
                adapter = SimpleNamespace(parse_env_info=lambda _: ({}, ''),
                    _get_latest_user_message=lambda *a, **kw: '')
                async def initialize(*unused):
                    if mode == 'initialize_timeout':
                        raise TimeoutError()
                    if mode == 'outer_task_timeout':
                        await asyncio.sleep(1)
                    return env, adapter, SimpleNamespace(tokenizer=None,processor=None), 'unused', {}
                async def exit_env(*unused):
                    pass
                namespace = dict(asyncio=asyncio,os=os,logging=logging,Sample=Sample,
                    logger=MagicMock(),_should_log_task_start=lambda _: False,
                    _initialize_resources=initialize,_append_host_to_blacklist_if_needed=lambda _: None,
                    _get_debug_trace_dir=lambda *a: '',_get_or_create_debug_trace_info=lambda *a: (False,None),
                    _safe_exit_env=exit_env,_browser_rollout_submission_slot=browser_slot,
                    ToolParser=lambda _: None,_get_browser_response_mode=lambda _: 'qwen')
                native_functions('openwebrl/generate_browser.py', [
                    '_generate_turn_sample_impl','generate_turn_sample','_get_rollout_task_timeout_secs',
                    '_should_remove_browser_sample','_mark_remove_sample_if_needed'],namespace)
                native_functions('slime/rollout/sglang_rollout.py',['generate_and_rm'],namespace)
                native_functions('slime/rollout/filter_hub/dynamic_sampling_filters.py',[
                    '_terminal_sample','_reward_value','check_reward_nonempty_nonzero_std'],namespace)
                async def unexpected_judge(*unused):
                    self.fail('Native ABORTED sample must not reach the judge')
                namespace.update(inspect=inspect,DynamicFilterOutput=DynamicFilterOutput,
                    GenerateState=lambda _: SimpleNamespace(aborted=False,semaphore=asyncio.Semaphore(1),
                        dp_rank_context=nullcontext), load_function=lambda _: bonus.generate,
                    batched_async_rm=unexpected_judge)
                incoming = Sample(index=123,group_index=2,metadata={'task_id':'train-task'})
                current = dict(config=dict(output=folder,policy_id='p'),started=time.monotonic(),
                    records=[],excluded_ids=set(),excluded_intents=set())
                args = SimpleNamespace(partial_rollout=False,group_rm=False,max_steps=0,
                    rollout_task_timeout_secs=.001 if mode == 'outer_task_timeout' else 5,
                    custom_generate_function_path='openwebrl.arm_turn_bonus.generate')
                fake = SimpleNamespace(generate_turn_sample=namespace['generate_turn_sample'])
                with patch.dict(sys.modules,{'openwebrl.generate_browser':fake}), patch.object(bonus,'state',return_value=current), patch.object(bonus,'log_progress'):
                    result = await namespace['generate_and_rm'](args,incoming,{})
                self.assertIs(result[0],incoming)
                self.assertEqual(incoming.status,Sample.Status.ABORTED)
                self.assertEqual(incoming.response_length,0)
                self.assertIsNone(incoming.reward)
                self.assertEqual(incoming.remove_sample, mode in ('outer_task_timeout','no_generated_turns'))
                self.assertEqual(current['native_failure_sentinels'],1)
                filtered=namespace['check_reward_nonempty_nonzero_std'](args,[result])
                self.assertFalse(filtered.keep)
                self.assertEqual(filtered.reason,'all_none_reward_in_group')
                self.assertTrue((Path(folder)/'labels/123.json').is_file())

    async def test_aborted_response_without_identity_is_not_a_prompt_sentinel(self):
        from slime.utils.types import Sample
        import time
        with tempfile.TemporaryDirectory() as folder:
            incoming=Sample(index=123,status=Sample.Status.ABORTED,response='generated',response_length=1)
            current=dict(config=dict(output=folder,policy_id='p'),started=time.monotonic(),
                records=[],excluded_ids=set(),excluded_intents=set())
            async def browser(*unused):
                return [incoming]
            with patch.dict(sys.modules,{'openwebrl.generate_browser':SimpleNamespace(generate_turn_sample=browser)}), patch.object(bonus,'state',return_value=current), patch.object(bonus,'log_progress'):
                with self.assertRaisesRegex(ValueError,'Trainable browser Sample'):
                    await bonus.generate(SimpleNamespace(),incoming,{})

    async def test_native_failure_sentinel_survives_wrapper(self):
        import time
        for metadata in [None, {'task_id':'webvoyager/60','terminate_reason':'generation_error: timeout'}]:
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as folder:
                current = dict(config=dict(output=folder,policy_id='p'), started=time.monotonic(),
                    records=[],excluded_ids=set(),excluded_intents=set())
                incoming = SimpleNamespace(index=123,metadata=metadata,remove_sample=True)
                async def browser(args, sample, params):
                    return [sample]
                fake = SimpleNamespace(generate_turn_sample=browser)
                with patch.dict(sys.modules,{'openwebrl.generate_browser':fake}), patch.object(bonus,'state',return_value=current), patch.object(bonus,'log_progress'):
                    result = await bonus.generate(SimpleNamespace(),incoming,{})
                self.assertIs(result[0],incoming)
                self.assertTrue(result[0].remove_sample)
                self.assertEqual(result[0].metadata,metadata)
                self.assertTrue((Path(folder)/'labels/123.json').is_file())

    async def test_missing_identity_on_trainable_row_is_still_an_error(self):
        import time
        with tempfile.TemporaryDirectory() as folder:
            current = dict(config=dict(output=folder,policy_id='p'),started=time.monotonic(),
                records=[],excluded_ids=set(),excluded_intents=set())
            incoming = SimpleNamespace(index=123,metadata={'task_id':'webvoyager/60'},remove_sample=False)
            async def browser(args,sample,params):
                return [sample]
            with patch.dict(sys.modules,{'openwebrl.generate_browser':SimpleNamespace(generate_turn_sample=browser)}), patch.object(bonus,'state',return_value=current), patch.object(bonus,'log_progress'):
                with self.assertRaisesRegex(ValueError,'Trainable browser Sample'):
                    await bonus.generate(SimpleNamespace(),incoming,{})

    async def test_wrapper_routes_training_sample_without_intent_to_arm(self):
        import time
        with tempfile.TemporaryDirectory() as folder:
            config = dict(output=folder,policy_id='p',seed=42,scored_fraction=.2)
            current = dict(config=config,started=time.monotonic(),records=[],
                           excluded_ids={'evaluation-id'},excluded_intents={'find store hours'})
            async def browser(args, sample, params):
                self.assertIsInstance(args.browser_action_selector, bonus.ShadowSelector)
                return []
            fake = SimpleNamespace(generate_turn_sample=browser)
            incoming = SimpleNamespace(index=123,metadata={'task_id':'webvoyager/60'})
            with patch.dict(sys.modules, {'openwebrl.generate_browser':fake}), patch.object(bonus,'state',return_value=current), patch.object(bonus,'log_progress'):
                await bonus.generate(SimpleNamespace(),incoming,{})
            self.assertTrue((Path(folder)/'labels/123.json').is_file())

    async def run_selector(self, duplicate=False, malformed=False, q=1., cancel=False, policy_id='p', sampling_policy_id=None):
        config = dict(seed=42, policy_id=policy_id, scored_fraction=q, max_pending_per_trajectory=2,
                      label_timeout_seconds=2, selector_endpoint='unused')
        if sampling_policy_id is not None: config['sampling_policy_id']=sampling_policy_id
        seen, requests = [], []
        original = output(0)
        context = dict(screenshot=b'image-before-action', active_tab_url='before')
        history = ['<think>Previous</think><tool_call>{}</tool_call>']
        async def infer(url, prompt, params, images, timeout_secs=None):
            seen.append((prompt, deepcopy(images), dict(params)))
            if len(seen) == 1:
                return original
            if cancel:
                await asyncio.sleep(10)
            return output(0 if duplicate else len(seen)-1)
        async def request(endpoint, payload, timeout, client):
            requests.append(payload)
            # Choose the original by its action, independent of shuffled position.
            selected = next(i for i,x in enumerate(payload['candidates']) if '"x": 0' in x['action'])
            return {'raw':'bad reply' if malformed else json.dumps({'selection':selected+1})}
        selector = bonus.ShadowSelector(config, 'trajectory', request=request)
        selector.set_training_context(dict(parent_sample_index=10, group_index=2))
        result = await selector(infer=infer, url='actor', input_text=PROMPT, sampling_params={'temperature':.8},
            images=['unchanged image'], observation=context, history=history, task='train-task',
            task_id='id', turn=1, timeout=5)
        # Mutating the live page after return must not change pending ARM input.
        context.update(screenshot=b'after-action', active_tab_url='after')
        history.append('after-action')
        await selector.finish(cancelled=cancel)
        self.assertIs(result[0], original)
        self.assertEqual(result[1]['executed_index'], 0)
        return selector, seen, requests

    async def test_original_unchanged_and_frozen_permuted_full_context(self):
        selector, calls, requests = await self.run_selector()
        self.assertEqual(len(calls), 5)
        self.assertEqual(calls[0][2], {'temperature':.8})
        self.assertTrue(all(c[0] == PROMPT and c[1] == ['unchanged image'] for c in calls))
        self.assertEqual(requests[0]['url'], 'before')
        self.assertEqual(len(requests[0]['history']), 1)
        self.assertTrue(all(x['thought'] for x in requests[0]['candidates']))
        self.assertEqual(selector.records[1]['unit_bonus'], .8)
        self.assertEqual(selector.records[1]['selected_index'], 0)

    async def test_provider_run_identity_does_not_change_candidate_rng_or_scoring_mask(self):
        for q in (1.,.2):
            arm, calls_a, requests_a=await self.run_selector(q=q,policy_id='arm-job:0',sampling_policy_id='shared:0')
            sol, calls_b, requests_b=await self.run_selector(q=q,policy_id='sol-job:0',sampling_policy_id='shared:0')
            self.assertEqual(calls_a,calls_b)
            self.assertEqual(requests_a,requests_b)
            self.assertEqual(arm.records[1]['sampled'],sol.records[1]['sampled'])
            self.assertNotEqual(arm.records[1]['policy_id'],sol.records[1]['policy_id'])

    async def test_duplicate_sets_skip_teacher_and_bonus(self):
        selector, _, requests = await self.run_selector(duplicate=True)
        self.assertEqual(requests, [])
        self.assertFalse(selector.records[1]['eligible'])
        self.assertEqual(selector.records[1]['error'], 'duplicate_action')

    async def test_malformed_label_keeps_executed_action(self):
        selector, _, _ = await self.run_selector(malformed=True)
        self.assertFalse(selector.records[1]['eligible'])
        self.assertNotIn('unit_bonus', selector.records[1])

    async def test_zero_sampling_makes_one_actor_call(self):
        selector, calls, requests = await self.run_selector(q=0.)
        self.assertEqual(len(calls), 1)
        self.assertEqual(requests, [])
        self.assertEqual(selector.records[1]['reason'], 'not_sampled')

    async def test_abort_drains_workers(self):
        selector, _, requests = await self.run_selector(cancel=True)
        self.assertEqual(requests, [])
        self.assertFalse(selector.records[1]['eligible'])
        self.assertFalse(selector.pending)


if __name__ == '__main__':
    unittest.main()
