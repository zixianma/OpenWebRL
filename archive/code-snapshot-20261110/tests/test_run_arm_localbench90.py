"""CPU regressions for local-suite authorization, cumulative budgets and stage ownership."""
import copy
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_arm_localbench90 as runner


class LocalSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.control = self.root / 'control'
        self.source = self.root / 'source'
        self.runtime = self.root / 'runtime'
        self.control.mkdir()
        self.source.mkdir()
        (self.source / 'reference_manifest.json').write_text('{}')
        self.patches = [patch.object(runner, 'CONTROL', self.control),
                        patch.object(runner, 'RUNTIME', self.runtime),
                        patch.object(runner.prepare, 'SOURCE', self.source)]
        for item in self.patches:
            item.start()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(lambda: [item.stop() for item in reversed(self.patches)])
        self.approval = dict(approved=True, repeat=1, resources=copy.deepcopy(runner.RESOURCES),
            source_manifest_sha256=runner.sha(self.source/'reference_manifest.json'),
            checkpoint_bindings={m: {} for m in runner.METHODS})
        self.write('local-suite-approval.json', self.approval)

    def write(self, name, value):
        runner.atomic(self.control/name, value)

    def info(self, job='202', seconds=39600, **changes):
        fields = dict(JobId=job, JobState='RUNNING', UserId=f'user({os.getuid()})',
            NumNodes='1', NumCPUs='8', NodeList='test-node', Restarts='0', Requeue='0',
            ReqTRES='cpu=8,mem=240G,node=1,gres/gpu=1,gres/gpu:h200=1',
            AllocTRES='cpu=8,mem=240G,node=1,gres/gpu=1,gres/gpu:h200=1',
            TimeLimit=f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}',
            EndTime=(datetime.now()+timedelta(seconds=seconds)).strftime('%Y-%m-%dT%H:%M:%S'))
        fields.update(changes)
        return ' '.join(f'{key}={value}' for key, value in fields.items())

    def checkpoint_fixture(self, original=False):
        method = 'original' if original else 'baseline'
        cp = self.root / 'iter_0000089'
        cp.mkdir(exist_ok=True)
        for name in ['common.pt', '.metadata', 'metadata.json', '__0_0.distcp']:
            (cp/name).write_bytes(b'fixture')
        cursor = self.root/'cursor.pt'
        cursor.write_bytes(b'cursor')
        report = self.root/'checkpoint-validation.json'
        runner.atomic(report, dict(iteration=89, checkpoint=str(cp), completed_optimizer_updates=1016,
            scheduler_num_steps=260352, scheduler_minus_optimizer_updates=1,
            shard_files={'__0_0.distcp':7}))
        selected = dict(path=str(cp), native_zero_based_iteration=89, adam_updates=1016,
            scheduler_num_steps=260352, scheduler_minus_optimizer_updates=1,
            validation_report=dict(path=str(report), sha256=runner.sha(report)),
            small_component_evidence=[dict(path=str(x), sha256=runner.sha(x)) for x in
                [cp/'common.pt', cp/'.metadata', cursor]])
        spec = dict(selected_checkpoint90=selected, checkpoint90_ready=True,
                    training_wandb_run_id='original-lineage' if original else 'baseline-lineage')
        self.write('evaluation-matrix.json', dict(methods={method:spec}))
        self.approval['checkpoint_bindings'][method] = runner.checkpoint_binding(method, spec)
        return method, spec, selected

    def test_approval_false_changed_resources_and_source_fail_closed(self):
        for update in [dict(approved=False), dict(repeat=2), dict(source_manifest_sha256='wrong'),
                       dict(resources=dict(runner.RESOURCES, hours_per_method=24))]:
            with self.subTest(update=update):
                self.write('local-suite-approval.json', dict(self.approval, **update))
                with self.assertRaises(PermissionError):
                    runner.approval('baseline')

    def test_later_repeats_rejected_before_any_scheduler_or_worker_call(self):
        with patch.object(runner.subprocess, 'run') as run, patch.object(runner.subprocess, 'check_output') as query:
            for repeat in (2,3):
                with self.assertRaises(ValueError): runner.execute('baseline','202',repeat)
                with self.assertRaises(ValueError): runner.register_attempt('baseline','202',3600,repeat)
                with self.assertRaises(ValueError): runner.worker('baseline','202',self.root/'missing',repeat)
            run.assert_not_called()
            query.assert_not_called()

    def test_user_scope_cancellation_and_clarification_block_execution(self):
        for field in ['disabled_methods', 'launch_paused_pending_scope_confirmation']:
            self.write('local-suite-approval.json', dict(self.approval, **{field:['baseline']}))
            with self.subTest(field=field), self.assertRaises(PermissionError):
                runner.approval('baseline')

    def test_original_pending_or_unreviewed90_cannot_use80(self):
        self.write('evaluation-matrix.json', dict(methods={'original':dict(
            selected_checkpoint90=None, checkpoint90_ready=False)}))
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            runner.selected_checkpoint('original', self.approval)
        method, spec, selected = self.checkpoint_fixture(original=True)
        with self.assertRaisesRegex(ValueError, 'independent checkpoint review'):
            runner.selected_checkpoint(method, self.approval)
        review = self.control/'original90-independent.json'
        runner.atomic(review, dict(verified_complete=True, completed_training_iteration=90,
            checkpoint=selected['path'], training_wandb_run_id=spec['training_wandb_run_id'],
            checkpoint_validation_sha256=selected['validation_report']['sha256']))
        spec['checkpoint90_independent_review'] = dict(path=str(review), sha256=runner.sha(review))
        self.write('evaluation-matrix.json', dict(methods={method:spec}))
        runner.selected_checkpoint(method, self.approval)
        selected['native_zero_based_iteration'] = 79
        self.write('evaluation-matrix.json', dict(methods={method:spec}))
        with self.assertRaises(ValueError): runner.selected_checkpoint(method, self.approval)

    def test_checkpoint_binding_and_component_changes_rejected(self):
        method, _, selected = self.checkpoint_fixture()
        runner.selected_checkpoint(method, self.approval)
        (Path(selected['path'])/'common.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'metadata/common/cursor'):
            runner.selected_checkpoint(method, self.approval)

    def test_scheduler_requires_exact_type_memory_cpu_time_and_no_requeue(self):
        runner.exact_resources(self.info(), '202', 39600)
        for changes in [dict(NumCPUs='16'), dict(NumNodes='2'), dict(Restarts='1'), dict(Requeue='1'),
            dict(TimeLimit='12:00:00'), dict(UserId='other(999999)'),
            dict(ReqTRES='cpu=8,mem=480G,node=1,gres/gpu=1,gres/gpu:h200=1'),
            dict(AllocTRES='cpu=8,mem=240G,node=1,gres/gpu=1,gres/gpu:a100=1')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runner.exact_resources(self.info(**changes), '202', 39600)

    def test_pending_single_node_range_registration(self):
        pending = self.info(JobState='PENDING', NumNodes='1-1', AllocTRES='(null)')
        self.assertEqual(runner.exact_resources(pending, '202', 39600, running=False), 39600)
        for nodes in ('1-2', '2-2', '0-1'):
            with self.subTest(nodes=nodes), self.assertRaises(ValueError):
                runner.exact_resources(self.info(JobState='PENDING', NumNodes=nodes),
                                       '202', 39600, running=False)
        with self.assertRaises(ValueError):
            runner.exact_resources(self.info(NumNodes='1-1'), '202', 39600)

    def test_failed_attempt_charges_are_monotonic_and_retry_cannot_reset_cap(self):
        self.write('local-suite-attempts.json', dict(approved_seconds_per_method=43200, methods={
            'baseline':dict(attempts=[dict(job_id='101',maximum_seconds=43200,charged_seconds=3600),
                                      dict(job_id='202',maximum_seconds=39600,charged_seconds=200)])}))
        def query(command, **kwargs):
            if command[0]=='scontrol': return self.info()
            return '101|FAILED|3500|1:0|\n' if command[command.index('-j')+1]=='101' else '202|RUNNING|100|0:0|\n'
        with patch.dict(os.environ, {'SLURM_JOB_ID':'202'}), patch.object(runner.subprocess,'check_output',side_effect=query):
            _, result = runner.accounting('baseline','202')
            self.assertEqual(result['consumed_seconds'],3800)
            stored=runner.ledger()
            stored['methods']['baseline']['attempts'][1]['maximum_seconds']=43200
            self.write('local-suite-attempts.json',stored)
            with self.assertRaisesRegex(ValueError,'budget exceeded'):
                runner.accounting('baseline','202')

    def test_unregistered_allocation_and_active_predecessor_rejected(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID':'202'}):
            with self.assertRaises(PermissionError): runner.accounting('baseline','202')
        with patch.object(runner.subprocess,'check_output',return_value='101|RUNNING|200|0:0|\n'):
            with self.assertRaisesRegex(ValueError,'Another live'):
                runner.refresh_attempts([dict(job_id='101',charged_seconds=0)],current_job='202')

    def stage_plans(self):
        return [dict(method='baseline', benchmark=b, repeat=1, job_id='202',
                     output=str(self.runtime/f'evaluations/arm90-local-baseline-{b}-r1-202'))
                for b in ['webvoyager','deepshop']]

    def test_workers_are_sequential_awaited_and_artifacts_do_not_claim_verified(self):
        events=[]
        stage_plans=self.stage_plans()
        def run(command, **kwargs):
            self.assertTrue(kwargs['check'])
            self.assertIn('srun',command)
            manifest=Path(command[command.index('--manifest')+1])
            events.append('worker:'+runner.read(manifest)['benchmark'])
        def audit(p):
            events.append('audit:'+p['benchmark'])
            return dict(verified_complete=False)
        with patch.object(runner,'accounting',return_value=(self.approval,dict(remaining_seconds=3000))), \
             patch.object(runner,'plans',return_value=stage_plans), \
             patch.object(runner,'register_supervisor'),patch.object(runner,'verified_stage',return_value=None), \
             patch.object(runner,'reject_unreviewed_outputs'),patch.object(runner.subprocess,'run',side_effect=run), \
             patch.object(runner,'audit_stage',side_effect=audit):
            runner.execute('baseline','202')
        self.assertEqual(events,['worker:webvoyager','audit:webvoyager','worker:deepshop','audit:deepshop'])
        state=runner.read(self.control/'controller-local-baseline-202/status.json')
        self.assertTrue(state['artifacts_ready'])
        self.assertFalse(state['complete'])
        self.assertFalse(state['verified_complete'])
        self.assertTrue(state['independent_review_pending'])

    def test_first_stage_failure_stops_second_and_preserves_failure_status(self):
        with patch.object(runner,'accounting',return_value=(self.approval,dict(remaining_seconds=3000))), \
             patch.object(runner,'plans',return_value=self.stage_plans()),patch.object(runner,'register_supervisor'), \
             patch.object(runner,'verified_stage',return_value=None),patch.object(runner,'reject_unreviewed_outputs'), \
             patch.object(runner.subprocess,'run',side_effect=subprocess.CalledProcessError(1,['srun'])) as run, \
             patch.object(runner,'audit_stage') as audit:
            with self.assertRaises(subprocess.CalledProcessError): runner.execute('baseline','202')
            self.assertEqual(run.call_count,1)
            audit.assert_not_called()
        self.assertEqual(runner.read(self.control/'controller-local-baseline-202/status.json')['stage'],'failed')

    def test_partial_prior_output_never_overwritten(self):
        p=self.stage_plans()[0]
        root=Path(p['output']);root.mkdir(parents=True)
        artifact=root/'saved-record.json';artifact.write_text('preserve')
        with patch.object(runner,'registered_jobs',return_value=['202']):
            with self.assertRaisesRegex(ValueError,'targeted recovery'):
                runner.reject_unreviewed_outputs(p)
        self.assertEqual(artifact.read_text(),'preserve')

    def test_verified_skip_requires_matching_hashes_and_independent_review(self):
        p=self.stage_plans()[0]
        p.update(checkpoint='native89',source='source',source_manifest_sha256='sourcehash',
                 checkpoint_validation_sha256='cphash',parent_training_run='training',
                 completed_optimizer_updates=1016,expected_task_count=595,
                 expected_rollout_task_ids=['task'],judge_model='gpt-4o',
                 judge_prompt_variant='webvoyager',protocol='local webvoyager',browser_env='local_process')
        root=Path(p['output']);root.mkdir(parents=True)
        runner.atomic(root/'evaluation_manifest.json',p)
        runner.atomic(root/'status.json',dict(complete=True,returncode=0))
        for name in ['metrics.json','audit.json','checkpoint_restore_evidence.json']:
            runner.atomic(root/name,{})
        review=dict(verified_complete=True,independent_review=True,job_id='202',root=str(root),
                    scientific_plan_sha256=runner.scientific_identity(p),
                    artifact_sha256={f.name:runner.sha(f) for f in root.iterdir()})
        self.write('stage-reviews/baseline-webvoyager-r1.json',review)
        with patch.object(runner,'registered_jobs',return_value=['202']):
            self.assertTrue(runner.verified_stage(p)['verified'])
            runner.atomic(root/'metrics.json',dict(changed=True))
            with self.assertRaisesRegex(ValueError,'evidence changed'):
                runner.verified_stage(p)

    def test_precollection_abort_recovery_requires_unchanged_zero_task_evidence(self):
        p = self.stage_plans()[0]
        root = Path(p['output']); root.mkdir(parents=True)
        for name in ['evaluation_manifest.json', 'checkpoint_validation.json']:
            runner.atomic(root/name, {})
        (root/'evaluation.log').write_text('model startup only')
        proof = dict(verified_precollection_abort=True, job_id='202', root=str(root),
                     scheduler_state='CANCELLED', rollout_files=0,
                     artifact_sha256={f.name: runner.sha(f) for f in root.iterdir()})
        self.write('precollection-reviews/baseline-webvoyager-202.json', proof)
        with patch.object(runner,'registered_jobs',return_value=['202']):
            runner.reject_unreviewed_outputs(p)
            (root/'rollouts').mkdir()
            (root/'rollouts/task.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'task evidence'):
                runner.reject_unreviewed_outputs(p)
            (root/'evaluation.log').write_text('new unreviewed evidence')
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                runner.reject_unreviewed_outputs(p)

    def test_restored_but_unstarted_evaluation_needs_bound_review(self):
        p = self.stage_plans()[0]
        root = Path(p['output']); root.mkdir(parents=True)
        for name in ['evaluation_manifest.json', 'checkpoint_validation.json', 'checkpoint_restore_evidence.json']:
            runner.atomic(root/name, {})
        (root/'evaluation.log').write_text('native restore completed, no evaluation')
        proof = dict(verified_precollection_abort=True, job_id='202', root=str(root),
                     scheduler_state='CANCELLED', rollout_files=0,
                     artifact_sha256={f.name:runner.sha(f) for f in root.iterdir()})
        self.write('precollection-reviews/baseline-webvoyager-202.json', proof)
        with patch.object(runner,'registered_jobs',return_value=['202']):
            with self.assertRaisesRegex(ValueError,'explicit precollection review'):
                runner.reject_unreviewed_outputs(p)
            proof.update(native_restore_completed=True, evaluation_started=False, no_task_collection_verified=True)
            self.write('precollection-reviews/baseline-webvoyager-202.json', proof)
            runner.reject_unreviewed_outputs(p)
            runner.atomic(root/'checkpoint_restore_evidence.json',dict(changed=True))
            with self.assertRaisesRegex(ValueError,'explicit precollection review'):
                runner.reject_unreviewed_outputs(p)

    def test_exact_stdout_restore_fallback_is_hash_bound_and_judge_is_benchmark_specific(self):
        for benchmark in ['webvoyager','deepshop']:
            p = dict(self.stage_plans()[0], benchmark=benchmark, checkpoint='native89',
                     checkpoint_index=89, output=str(self.root/benchmark))
            root=Path(p['output']);root.mkdir()
            line=f'successfully loaded checkpoint from {root}/checkpoint-view [ t 1/1, p 1/1 ] at iteration 89'
            (root/'evaluation.log').write_text(line+'\n')
            with patch.object(runner.shared,'audit',return_value={}) as audit, \
                 patch.object(runner,'scientific_identity',return_value='identity'):
                result=runner.audit_stage(p)
            audit.assert_called_once_with(p,expected_judge=('gpt-4o',benchmark))
            proof=runner.read(root/'checkpoint_restore_evidence.json')
            self.assertEqual(proof['evaluation_log_sha256'],runner.sha(root/'evaluation.log'))
            self.assertFalse(result['verified_complete'])

    def test_restore_fallback_rejects_wrong_checkpoint_view_or_iteration(self):
        p=dict(self.stage_plans()[0],checkpoint='native89',checkpoint_index=89,
               output=str(self.root/'restore'))
        root=Path(p['output']);root.mkdir()
        for line in [f'successfully loaded checkpoint from {root}/checkpoint-view [ t 1/1, p 1/1 ] at iteration 79',
                     'successfully loaded checkpoint from /unrelated/checkpoint-view [ t 1/1, p 1/1 ] at iteration 89']:
            (root/'evaluation.log').write_text(line+'\n')
            with patch.object(runner.shared,'audit') as audit, self.assertRaisesRegex(ValueError,'restoration evidence'):
                runner.audit_stage(p)
            audit.assert_not_called()
            self.assertFalse((root/'checkpoint_restore_evidence.json').exists())


if __name__ == '__main__':
    unittest.main()
