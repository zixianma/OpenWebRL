import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import supervise_arm_runs as supervisor


class Supervision(unittest.TestCase):
    def test_quota_warning_uses_personal_limits(self):
        for soft,hard,expected in [(5,15,0),(3,13,1),(0,1,2)]:
            with self.subTest(soft=soft,hard=hard),patch('check_storage_quota.check_storage_quota',return_value={
                    'safe_headroom_bytes':soft*1024**4,'hard_headroom_bytes':hard*1024**4}):
                self.assertEqual(len(supervisor.quota_snapshot()['alerts']),expected)

    def test_missing_quota_is_not_reported_as_healthy(self):
        with patch('check_storage_quota.check_storage_quota',side_effect=OSError('unavailable')):
            self.assertTrue(supervisor.quota_snapshot()['alerts'])

    def test_new_approved_submissions_load_without_restarting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(supervisor,'CONTROL',root),patch.dict(supervisor.JOBS,{},clear=True):
                (root/'discovered-jobs.json').write_text(json.dumps({'1':['B','arm-gate-b-to60-1','stage1']}))
                supervisor.load_registered_jobs()
                self.assertIn('1',supervisor.JOBS)
                (root/'discovered-jobs.json').write_text(json.dumps({'2':['C','arm-gate-c-to60-2','stage1']}))
                supervisor.load_registered_jobs()
                self.assertEqual(set(supervisor.JOBS),{'1','2'})

    def test_registered_attempt_path_can_be_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'discovered-jobs.json').write_text(json.dumps({'1':['B','new-attempt','stage1']}))
            with patch.object(supervisor,'CONTROL',root),patch.dict(supervisor.JOBS,{'1':('B','old','stage1')},clear=True):
                supervisor.load_registered_jobs()
                self.assertEqual(supervisor.JOBS['1'][1],'new-attempt')

    def test_live_queue_overrides_previous_failed_accounting(self):
        live=[dict(job='1',state='PENDING',node='Priority',elapsed='0:00')]
        with patch.dict(supervisor.JOBS,{'1':('B','old','stage1')},clear=True),patch.object(
                supervisor.subprocess,'check_output',return_value='1|FAILED|1:00|2:00|g001|1:0\n'):
            state=supervisor.scheduler_states(live)['1']
        self.assertEqual(state['state'],'PENDING');self.assertIsNone(state['exit_code'])

    def test_pending_requeue_does_not_report_old_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Path(tmp);root=runtime/'evaluations/old';root.mkdir(parents=True)
            (root/'status.json').write_text(json.dumps(dict(stage='failed',failed=True,error_type='InterruptedError')))
            with patch.object(supervisor,'RUNTIME',runtime),patch.dict(supervisor.JOBS,{'1':('B','old','stage1')},clear=True):
                row=supervisor.snapshot('1',dict(job='1',state='PENDING'))
            self.assertEqual(row['stage'],'queued');self.assertFalse(row['alerts'])

    def migration_fixture(self,tmp):
        runtime=Path(tmp)
        request=runtime/'arm-turn-bonus-preparation/bc-tp2-migration-20260922/1.json'
        request.parent.mkdir(parents=True)
        request.write_text(json.dumps(dict(stage='released',variant='B')))
        controller=runtime/'evaluations/arm-gate-b-to60-1-tp2';controller.mkdir(parents=True)
        return runtime,controller

    def test_tp2_validation_follows_current_attempt_and_partial_update_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime,controller=self.migration_fixture(tmp)
            (controller/'status.json').write_text(json.dumps(dict(stage='full-batch-validation')))
            bench=runtime/'benchmarks/arm-gate-b-tp2-full-1';bench.mkdir(parents=True)
            (bench/'plan.json').write_text(json.dumps(dict(optimizer_updates=12)))
            (bench/'updates-rank0.jsonl').write_text('{"seconds":30}\n{"seconds":')
            with patch.object(supervisor,'RUNTIME',runtime):
                row=supervisor.stage1_snapshot('1',runtime/'evaluations/old',dict(job='1',state='RUNNING'))
            self.assertEqual(row['stage'],'full-batch-validation')
            self.assertEqual(row['validation_updates'],1)
            self.assertEqual(row['validation_expected_updates'],12)
            self.assertFalse(row['complete']);self.assertFalse(row['alerts'])

    def test_tp2_training_path_is_validated_and_tracked(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime,controller=self.migration_fixture(tmp)
            training=runtime/'evaluations/arm-failure-additive-1-tp2-iter40'
            native=training/'runtime';native.mkdir(parents=True)
            (native/'latest_checkpointed_iteration.txt').write_text('26')
            plan=dict(variant='B',current_target=40,training_root=str(training))
            (controller/'controller-plan.json').write_text(json.dumps(plan))
            with patch.object(supervisor,'RUNTIME',runtime),patch.object(supervisor.monitor,'snapshot',return_value=dict(alerts=[],status={'stage':'training'})):
                row=supervisor.stage1_snapshot('1',runtime/'evaluations/old',dict(job='1',state='RUNNING'))
                self.assertEqual(row['durable_iteration'],27)
                self.assertEqual(row['training_root'],str(training))
                plan['training_root']=str(runtime/'unrelated')
                (controller/'controller-plan.json').write_text(json.dumps(plan))
                with self.assertRaises(ValueError):
                    supervisor.stage1_snapshot('1',controller,dict(job='1',state='RUNNING'))

    def test_ablation_validation_uses_current_attempt_even_with_controller_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Path(tmp);request=runtime/'arm-turn-bonus-preparation/ablation-tp2-migration-20260922/1.json'
            request.parent.mkdir(parents=True);request.write_text(json.dumps(dict(stage='armed-fresh',variant='coverage')))
            controller=runtime/'evaluations/arm-failure-ablation-coverage-1-tp2';controller.mkdir(parents=True)
            (controller/'controller-plan.json').write_text(json.dumps(dict(variant='coverage',target=20)))
            (controller/'status.json').write_text(json.dumps(dict(stage='full-batch-validation')))
            with patch.object(supervisor,'RUNTIME',runtime):
                row=supervisor.stage1_snapshot('1',runtime/'old',dict(job='1',state='RUNNING'))
            self.assertEqual(row['kind'],'diagnostic')
            self.assertEqual(row['validation_root'],str(runtime/'benchmarks/arm-ablation-tp2-full-1'))

    def test_gate_controller_tracks_its_actual_eval_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Path(tmp);controller=runtime/'evaluations/arm-gate-b-stage40-1'
            controller.mkdir(parents=True)
            (controller/'controller-plan.json').write_text(json.dumps(dict(variant='B',target=40)))
            native=runtime/'evaluations/arm-failure-additive-1/runtime';native.mkdir(parents=True)
            (native/'latest_checkpointed_iteration.txt').write_text('39')
            records=runtime/'evaluations/arm-gate-b-iter40-1/rollouts';records.mkdir(parents=True)
            (records/'task.pt').write_bytes(b'archive')
            (records/'task.json').write_text(json.dumps(dict(metrics=dict(successes=1,valid_trajectories=1))))
            with patch.object(supervisor,'RUNTIME',runtime),patch.object(supervisor.monitor,'snapshot',return_value=dict(alerts=[],status={'stage':'complete'})):
                row=supervisor.stage1_snapshot('1',controller,dict(job='1',state='RUNNING'))
            self.assertEqual(row['stage'],'evaluation');self.assertEqual(row['completed_tasks'],1)
            self.assertEqual(row['saved_rollouts'],1);self.assertFalse(row['complete'])

    def fixture(self, tmp, count, exit_code):
        root = Path(tmp)/'evaluations/eval-fixture'
        (root/'rollouts').mkdir(parents=True)
        for i in range(count):
            (root/'rollouts'/f'{i}.pt').write_bytes(b'fixture')
            (root/'rollouts'/f'{i}.json').write_text(json.dumps(dict(task_id=str(i),metrics=dict(successes=0,valid_trajectories=0),error_type=None)))
        (root/'launcher_exit.json').write_text(json.dumps(dict(returncode=exit_code)))
        (root/'evaluation_manifest.json').write_text('{}')
        return root

    def test_only_zero_exit_complete_archives_allow_bookkeeping_recovery(self):
        for count,code,allowed in [(300,0,True),(299,0,False),(300,1,False)]:
            with self.subTest(count=count,code=code),tempfile.TemporaryDirectory() as tmp:
                self.fixture(tmp,count,code)
                with patch.object(supervisor,'RUNTIME',Path(tmp)),patch.dict(supervisor.JOBS,{'1':('fixture','eval-fixture','eval')}),patch('evaluate_baseline_checkpoint.finalize_evaluation') as finalize:
                    row=supervisor.snapshot('1',dict(state='FAILED',job='1'))
                self.assertEqual(finalize.called,allowed)
                self.assertEqual(row['complete'],allowed)

    def test_running_partial_eval_is_observed_without_retrying(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp,3,0)
            with patch.object(supervisor,'RUNTIME',Path(tmp)),patch.dict(supervisor.JOBS,{'1':('fixture','eval-fixture','eval')}),patch('evaluate_baseline_checkpoint.finalize_evaluation') as finalize:
                row=supervisor.snapshot('1',dict(state='RUNNING',job='1'))
            self.assertFalse(finalize.called)
            self.assertEqual(row['completed_tasks'],3)
            self.assertFalse(row['complete'])

    def test_mig_completion_requires_probe_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'status.json').write_text(json.dumps(dict(complete=True)))
            row=supervisor.mig_snapshot(root,dict(state='COMPLETED'))
            self.assertFalse(row['complete']);self.assertTrue(row['alerts'])
            (root/'probe-result.json').write_text(json.dumps(dict(passed=True)))
            row=supervisor.mig_snapshot(root,dict(state='COMPLETED'))
            self.assertTrue(row['complete']);self.assertFalse(row['alerts'])

    def test_coverage_completion_needs_report_and_zero_actor_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            state=dict(shadow_complete=True,no_optimizer_updates=True,stage='complete')
            (root/'status.json').write_text(json.dumps(state))
            with patch.object(supervisor.monitor,'snapshot',return_value=dict(alerts=[])):
                row=supervisor.coverage_snapshot(root,dict(state='COMPLETED',job='1'))
                self.assertFalse(row['complete'])
                (root/'failure-coverage.json').write_text(json.dumps(dict(complete=True)))
                row=supervisor.coverage_snapshot(root,dict(state='COMPLETED',job='1'))
                self.assertTrue(row['complete']);self.assertFalse(row['alerts'])
                state['no_optimizer_updates']=False
                (root/'status.json').write_text(json.dumps(state))
                row=supervisor.coverage_snapshot(root,dict(state='COMPLETED',job='1'))
                self.assertFalse(row['complete']);self.assertTrue(row['alerts'])


if __name__ == '__main__':
    unittest.main()
