"""Regressions for wrong judge/temperature and lost invalid-task artifacts."""
import asyncio
import ast
import copy
import json
import logging
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import arm_webvoyager_generator as generator
import run_arm_webvoyager90 as controller


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        generator._provider_blocked = False

    def tearDown(self):
        generator._provider_blocked = False

    def test_sampling_and_judge_override_inherited_monitor(self):
        original = SimpleNamespace(judge_api_model='gpt-4.1', judge_prompt_variant='action_history')
        args, sampling = generator.configure(original, dict(temperature=0, top_p=1, top_k=1))
        self.assertEqual((args.judge_api_model,args.judge_prompt_variant,args.max_steps),('gpt-4o','webvoyager',30))
        self.assertEqual((sampling['temperature'],sampling['top_p'],sampling['top_k']),(.6,.95,20))
        self.assertEqual(original.judge_api_model,'gpt-4.1')

    def run_generation(self, turns, rewards=None, error=None, expected_error=None):
        fake = {}
        for name in ['openwebrl.generate_browser','openwebrl.eval.reward_webvoyager',
                     'openwebrl.eval_monitor','slime.utils.types']:
            fake[name]=ModuleType(name)
        collect=AsyncMock(return_value=turns,side_effect=error)
        judge=AsyncMock(return_value=rewards)
        persist=AsyncMock()
        fake['openwebrl.generate_browser'].generate_turn_sample=collect
        fake['openwebrl.eval.reward_webvoyager'].reward_func=judge
        fake['openwebrl.eval_monitor'].persist_task=persist
        fake['slime.utils.types'].Sample=SimpleNamespace(Status=SimpleNamespace(ABORTED='aborted'))
        with patch.dict(sys.modules,fake):
            if error or expected_error:
                with self.assertRaises(expected_error or type(error)):
                    asyncio.run(generator.generate(SimpleNamespace(),SimpleNamespace(),{},evaluation=True))
            else:
                asyncio.run(generator.generate(SimpleNamespace(),SimpleNamespace(),{},evaluation=True))
        return judge,persist

    def test_missing_judge_cannot_fall_back_to_generic_reward(self):
        turn=SimpleNamespace(status='completed',metadata={},remove_sample=False,reward=None)
        judge,persist=self.run_generation([turn],[None])
        self.assertEqual(turn.reward,0)
        self.assertTrue(turn.remove_sample)
        self.assertTrue(turn.metadata['judge_invalid'])
        judge.assert_awaited_once();persist.assert_awaited_once()
        self.assertEqual(persist.await_args.args[0].judge_prompt_variant,'webvoyager')

    def test_abort_skips_judge_but_still_persists(self):
        judge,persist=self.run_generation([SimpleNamespace(status='aborted')])
        judge.assert_not_awaited();persist.assert_awaited_once()

    def test_generation_exception_still_has_task_record(self):
        judge,persist=self.run_generation([],error=TimeoutError('test'))
        judge.assert_not_awaited();persist.assert_awaited_once()
        self.assertEqual(persist.await_args.args[-1],'TimeoutError')

    def test_credit_failure_persists_then_stops_queued_session_requests(self):
        turn=SimpleNamespace(status='aborted',metadata={'terminate_reason':
            'generation_error: 402: You need credits to start a browser session.'})
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {
                'OPENWEBRL_EVAL_ROLLOUT_DIR':str(Path(directory)/'rollouts')}):
            judge,persist=self.run_generation([turn],expected_error=generator.BrowserCreditsExhausted)
            judge.assert_not_awaited();persist.assert_awaited_once()
            self.assertEqual(persist.await_args.args[-1],'BrowserCreditsExhausted')
            self.assertTrue((Path(directory)/'provider-blocked.json').exists())
            # Guard precedes imports and browser creation, even when invoked again.
            with self.assertRaises(generator.BrowserCreditsExhausted):
                asyncio.run(generator.generate(SimpleNamespace(),SimpleNamespace(),{},evaluation=True))

    def test_ordinary_site_error_does_not_stop_cohort(self):
        turn=SimpleNamespace(status='aborted',metadata={'terminate_reason':'generation_error: screenshot timeout'})
        self.run_generation([turn])
        self.assertFalse(generator._provider_blocked)


class WebVoyagerTests(unittest.TestCase):
    def test_real_judge_sends_final_answer_screenshots_and_correct_model(self):
        # Execute the released judge function with only the network client mocked.
        tree=ast.parse((controller.SOURCE/'openwebrl/eval/reward_webvoyager.py').read_text())
        selected=[n for n in tree.body if isinstance(n,ast.Assign) and any(
            isinstance(t,ast.Name) and t.id in ('SYSTEM_PROMPT','USER_PROMPT') for t in n.targets)
            or isinstance(n,ast.AsyncFunctionDef) and n.name=='_judge']
        call=AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='NOT SUCCESS'))]))
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=call)))
        namespace=dict(Any=object,Sample=object,ToolParser=object,asyncio=asyncio,
            logger=logging.getLogger('test'),_extract_final_answer=lambda parser,response:'Final answer',
            _get_openai_client=lambda **kwargs:client)
        exec(compile(ast.Module(body=selected,type_ignores=[]),'released-judge','exec'),namespace)
        args,_=generator.configure(SimpleNamespace(),{})
        sample=SimpleNamespace(metadata=dict(intent='Original task',full_image_list=[f'image{i}' for i in range(35)]),response='done')
        result=asyncio.run(namespace['_judge'](args,sample,None))
        self.assertEqual(result,(0.0,'NOT SUCCESS',False))
        kwargs=call.await_args.kwargs
        self.assertEqual(kwargs['model'],'gpt-4o');self.assertEqual(kwargs['seed'],42)
        content=kwargs['messages'][1]['content']
        self.assertEqual(len(content),31)
        self.assertIn('Original task',content[0]['text']);self.assertIn('Final answer',content[0]['text'])
        self.assertTrue(content[1]['image_url']['url'].endswith('image5'))
        self.assertTrue(content[-1]['image_url']['url'].endswith('image34'))

    def test_prepared595_loads_through_real_multimodal_dataset(self):
        # Exercise the production Dataset and Qwen vision parser, without loading model weights.
        # Supplying an image-processor configuration enables the exact branch that failed.
        sys.path.insert(0, str(controller.SOURCE))
        from slime.utils.data import Dataset
        processor = SimpleNamespace(image_processor=SimpleNamespace(patch_size=16))
        original = controller.CONTROL/'source/webvoyager595.jsonl'
        with self.assertRaisesRegex(AssertionError, 'prompt must be a list'):
            Dataset(str(original), tokenizer=None, processor=processor, max_length=None,
                    prompt_key='prompt', metadata_key='metadata', apply_chat_template=False)
        dataset = Dataset(str(controller.SOURCE/'webvoyager595.jsonl'), tokenizer=None,
                          processor=processor, max_length=None, prompt_key='prompt',
                          metadata_key='metadata', apply_chat_template=False)
        self.assertEqual(len(dataset), 595)
        released = {str(row['id']):row for row in [json.loads(line) for line in
            (controller.SOURCE/'webvoyager_released.jsonl').read_text().splitlines()]}
        for sample in dataset.samples:
            raw = released[sample.metadata['task_id']]
            self.assertEqual(sample.prompt, [dict(role='user', content=raw['ques'])])
            self.assertEqual(sample.metadata['intent'], raw['ques'])
            self.assertEqual(sample.metadata['start_url'], raw['web'])
            self.assertEqual(sample.multimodal_inputs, dict(images=None, videos=None))

    def test_all_models_use_identical_full595_and_independent_artifacts(self):
        plans=[controller.plan(m) for m in controller.METHODS]
        self.assertEqual(len({p['checkpoint'] for p in plans}),3)
        self.assertEqual(len({p['output'] for p in plans}),3)
        for p in plans:
            self.assertEqual(p['expected_task_count'],595)
            self.assertEqual(p['expected_rollout_task_ids'],plans[0]['expected_rollout_task_ids'])
            self.assertEqual(p['environment']['WANDB_PROJECT'],'openwebrl-evals')
            self.assertEqual(p['optimizer_updates_requested'],0)
        self.assertEqual(plans[0]['environment']['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET'],'1')
        self.assertEqual(plans[1]['environment']['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET'],'0')

    def test_inherited_om2w_judge_dataset_or_training_project_rejected(self):
        p=controller.plan('additive')
        for key,value in [('JUDGE_MODEL','o4-mini'),('WANDB_PROJECT','openwebrl'),('NUM_ROLLOUT','1')]:
            bad=copy.deepcopy(p);bad['environment'][key]=value
            with self.assertRaises(ValueError):controller.validate(bad)
        for key,value in [('expected_task_count',300),('metric_prefix','eval/online-mind2web-benchmark')]:
            bad=copy.deepcopy(p);bad[key]=value
            with self.assertRaises(ValueError):controller.validate(bad)

    def test_om2w_approval_does_not_fund_webvoyager(self):
        with patch.object(controller.base,'read',return_value=dict(approved=True,resources=controller.shared.RESOURCES)),patch.object(controller.subprocess,'check_output') as call:
            with self.assertRaisesRegex(ValueError,'Exact WebVoyager'):controller.execute('0','baseline')
            call.assert_not_called()

    def test_worker_allocation_cap_is_not_silently_eight_hours(self):
        p=controller.plan('baseline','123456')
        evaluator=controller.base.evaluator
        with patch.dict('os.environ',SLURM_JOB_ID='123456'),patch.object(Path,'read_text',return_value='/job_123456/'),patch.object(evaluator.subprocess,'check_output',return_value='123456.batch\n123456.extern\n'),patch.object(evaluator,'allocation',side_effect=RuntimeError('stop after check')) as allocation:
            with self.assertRaisesRegex(RuntimeError,'stop after check'):evaluator.run(p,Path('.env'))
            self.assertEqual(allocation.call_args.kwargs['maximum_hours'],12)


if __name__=='__main__':unittest.main()
