"""Regressions for wrong judge/temperature and lost invalid-task artifacts."""
import asyncio
import copy
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import arm_stealth_o4_generator as generator
import run_arm_stealth90_o4 as controller


class GeneratorTests(unittest.TestCase):
    generator_module = generator

    def setUp(self):
        self.generator = self.generator_module
        self.generator._provider_blocked = False

    def tearDown(self):
        self.generator._provider_blocked = False

    def test_sampling_and_judge_override_inherited_monitor(self):
        original = SimpleNamespace(judge_api_model='gpt-4.1', judge_prompt_variant='action_history')
        args, sampling = self.generator.configure(original, dict(temperature=0, top_p=1, top_k=1))
        self.assertEqual((args.judge_api_model,args.judge_prompt_variant,args.max_steps),('o4-mini','agenttrek',30))
        self.assertEqual((sampling['temperature'],sampling['top_p'],sampling['top_k']),(.6,.95,20))
        self.assertEqual(original.judge_api_model,'gpt-4.1')

    def run_generation(self, turns, rewards=None, error=None, expected_error=None):
        fake = {}
        for name in ['openwebrl.generate_browser','openwebrl.eval.reward_online_mind2web',
                     'openwebrl.eval_monitor','slime.utils.types']:
            fake[name]=ModuleType(name)
        collect=AsyncMock(return_value=turns,side_effect=error)
        judge=AsyncMock(return_value=rewards)
        persist=AsyncMock()
        fake['openwebrl.generate_browser'].generate_turn_sample=collect
        fake['openwebrl.eval.reward_online_mind2web'].reward_func=judge
        fake['openwebrl.eval_monitor'].persist_task=persist
        fake['slime.utils.types'].Sample=SimpleNamespace(Status=SimpleNamespace(ABORTED='aborted'))
        with patch.dict(sys.modules,fake):
            if error or expected_error:
                with self.assertRaises(expected_error or type(error)):
                    asyncio.run(self.generator.generate(SimpleNamespace(),SimpleNamespace(),{},evaluation=True))
            else:
                asyncio.run(self.generator.generate(SimpleNamespace(),SimpleNamespace(),{},evaluation=True))
        return judge,persist

    def test_missing_judge_cannot_fall_back_to_generic_reward(self):
        turn=SimpleNamespace(status='completed',metadata={},remove_sample=False,reward=None)
        judge,persist=self.run_generation([turn],[None])
        self.assertEqual(turn.reward,0)
        self.assertTrue(turn.remove_sample)
        self.assertTrue(turn.metadata['judge_invalid'])
        judge.assert_awaited_once();persist.assert_awaited_once()
        self.assertEqual(persist.await_args.args[0].judge_prompt_variant,'agenttrek')

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
            judge,persist=self.run_generation([turn],expected_error=self.generator.BrowserCreditsExhausted)
            judge.assert_not_awaited();persist.assert_awaited_once()
            self.assertEqual(persist.await_args.args[-1],'BrowserCreditsExhausted')
            self.assertTrue((Path(directory)/'provider-blocked.json').exists())
            # Guard precedes imports and browser creation, even when invoked again.
            with self.assertRaises(self.generator.BrowserCreditsExhausted):
                asyncio.run(self.generator.generate(SimpleNamespace(),SimpleNamespace(),{},evaluation=True))

    def test_ordinary_site_error_does_not_stop_cohort(self):
        turn=SimpleNamespace(status='aborted',metadata={'terminate_reason':'generation_error: screenshot timeout'})
        self.run_generation([turn])
        self.assertFalse(self.generator._provider_blocked)


class PublicTemplateTests(GeneratorTests):
    from openwebrl import eval_benchmark as generator_module

    def test_legacy_summary_remains_available_for_subset_retries(self):
        turn=SimpleNamespace(status='completed',metadata={'task_id':'task-1','reward':1},
                             remove_sample=False,reward=1,response='done')
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {
                'OPENWEBRL_BENCHMARK_RESULTS_DIR':directory}):
            judge,persist=self.run_generation([turn],[1])
            records=list(Path(directory).glob('*.json'))
            self.assertEqual(len(records),1)
            record=json.loads(records[0].read_text())
            self.assertEqual((record['task_id'],record['judge_model'],record['judge_prompt_variant']),
                             ('task-1','o4-mini','agenttrek'))
            persist.assert_awaited_once()


class PublicPreparationTests(unittest.TestCase):
    def test_preparer_copies_persistence_dependencies_into_older_source(self):
        import hashlib
        import prepare_paper_benchmark as prepare
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'parent';source.mkdir()
            (source/'reference_manifest.json').write_text(json.dumps({
                'browser_use_evaluation':{'validated':True},'recipe_files_sha256':{}}))
            with patch.object(prepare,'RUNTIME',root), patch.object(prepare,'validate_source'):
                prepare.prepare(source,root/'prepared')
            manifest=json.loads((root/'prepared/reference_manifest.json').read_text())
            for name in ['openwebrl/eval_benchmark.py','openwebrl/eval_monitor.py',
                         'slime/utils/rollout_transport.py','slime/utils/trajectory_metrics.py',
                         'slime/utils/types.py']:
                payload=(root/'prepared'/name).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(),
                                 manifest['recipe_files_sha256'][name])
            self.assertFalse(manifest['paper_om2w_benchmark']['gpu_validation_passed'])


class PreparedPlanTests(unittest.TestCase):
    def test_recovery_plans_only_contain_credit_blocked_tasks(self):
        for method in controller.METHODS:
            p=controller.plan(method,recovery=True)
            controller.validate_plan(p)
            evidence=controller.base.read(controller.CONTROL/f'{method}-provider-recovery.json')
            self.assertEqual(p['expected_rollout_task_ids'],evidence['recovery_task_ids'])
            self.assertEqual(p['expected_task_count'],evidence['provider_blocked'])
            self.assertEqual(p['environment']['OPENWEBRL_RECORD_EVAL_EXPECTED_TASK_COUNT'],str(evidence['provider_blocked']))
            self.assertFalse(set(p['expected_rollout_task_ids']) & {r['task_id'] for r in evidence['other_invalid_tasks']})

    def test_recovery_merge_retains_original_failures_and_rejects_valid_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            old=Path(d)/'old';new=Path(d)/'new'
            (old/'rollouts').mkdir(parents=True);(new/'rollouts').mkdir(parents=True)
            for i in range(300):
                record={'task_id':str(i),'metrics':{'valid_trajectories':int(i!=0),'successes':int(i==1)}}
                (old/'rollouts'/f'{i}.json').write_text(json.dumps(record))
            retry={'task_id':'0','metrics':{'valid_trajectories':1,'successes':1}}
            (new/'rollouts'/'0.json').write_text(json.dumps(retry))
            p={'output':str(new),'provider_recovery':{'original_root':str(old),'expected_task_ids':['0']}}
            result=controller.merge_recovery(p)
            self.assertEqual((result['successes'],result['valid'],result['retained_original_records']),(2,300,299))
            record={'task_id':'0','metrics':{'valid_trajectories':1,'successes':0}}
            (old/'rollouts'/'0.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'valid original'):controller.merge_recovery(p)

    def test_all_three_plans_share_cohort_and_protocol(self):
        plans=[controller.plan(m) for m in controller.METHODS]
        self.assertEqual(len({p['checkpoint'] for p in plans}),3)
        self.assertEqual(len({p['parent_training_run'] for p in plans}),3)
        for p in plans:
            controller.validate_plan(p)
            self.assertEqual(p['expected_rollout_task_ids'],plans[0]['expected_rollout_task_ids'])
            self.assertEqual(len(p['expected_rollout_task_ids']),300)
        self.assertEqual(plans[0]['environment']['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET'],'1')
        self.assertEqual(plans[1]['environment']['OPENWEBRL_EXPECTED_SCHEDULER_OFFSET'],'0')

    def test_wrong_command_or_environment_fails_before_worker(self):
        p=controller.plan('baseline')
        for key,value in [('JUDGE_MODEL','gpt-4.1'),('NUM_ROLLOUT','1'),('WANDB_PROJECT','openwebrl')]:
            changed=copy.deepcopy(p);changed['environment'][key]=value
            with self.assertRaisesRegex(ValueError,'worker environment'):controller.validate_plan(changed)
        changed=copy.deepcopy(p);changed['command'][changed['command'].index('--eval-config')+1]='monitor.yaml'
        with self.assertRaisesRegex(ValueError,'worker command'):controller.validate_plan(changed)

    def test_no_execution_without_exact_resource_approval(self):
        with patch.object(controller.base,'read',return_value={'approved':False}),patch.object(controller.subprocess,'check_output') as call:
            with self.assertRaisesRegex(ValueError,'Exact resource approval'):controller.execute('0','baseline')
            call.assert_not_called()

    def test_ray_socket_path_fits_actual_ray_validator(self):
        from ray._private.utils import validate_socket_filepath
        p=controller.plan('baseline')
        suffix='ray/session_2026-09-29_09-45-06_123456_1234567890/sockets/plasma_store'
        validate_socket_filepath(str(Path(p['environment']['RAY_TMPDIR'])/suffix))
        p['environment']['RAY_TMPDIR']='/tmp/stealth90-o4-t06-baseline-r1-336861-ray'
        with self.assertRaisesRegex(ValueError,'AF_UNIX'):controller.validate_plan(p)
        with self.assertRaises(OSError):validate_socket_filepath(str(Path(p['environment']['RAY_TMPDIR'])/suffix))


if __name__=='__main__':unittest.main()
