"""CPU-only resume safety and continuation-state regression tests."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location('resume_baseline', Path(__file__).resolve().parents[1] / 'scripts/resume_baseline.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class ResumeTest(unittest.TestCase):
    def setUp(self):
        # Retained under /tmp; no destructive cleanup commands.
        self.root = Path(tempfile.mkdtemp(prefix='openwebrl-resume-test-'))

    def info(self, **overrides):
        fields = dict(JobId='42', JobState='RUNNING', UserId=f'test({os.getuid()})',
                      NumNodes='1', NumCPUs='8', NodeList='g021', EndTime='2030-01-01T08:00:00',
                      AllocTRES='cpu=8,mem=240G,node=1,gres/gpu=2,gres/gpu:h200=2')
        fields.update(overrides)
        return ' '.join(f'{k}={v}' for k, v in fields.items())

    def test_source_checks_reject_changed_recipe_and_missing_memory_fix(self):
        import hashlib
        source = self.root / 'source'
        (source / 'scripts').mkdir(parents=True)
        (source / 'slime/rollout').mkdir(parents=True)
        (source / 'slime/utils').mkdir(parents=True)
        recipe = source / 'recipe.py'
        recipe.write_text('reference reward rule')
        (source / 'reference_manifest.json').write_text(json.dumps({'recipe_files_sha256': {
            'recipe.py': hashlib.sha256(recipe.read_bytes()).hexdigest()}}))
        launcher = source / 'scripts/run_small_baseline.py'
        launcher.write_text("OPENWEBRL_MULTIMODAL_STORAGE_DIR --save-debug-rollout-data --skip-eval-before-train 'shutdown_margin_seconds' 'durable_optimizer_updates_at_start'")
        (source / 'slime/rollout/sglang_rollout.py').write_text('await asyncio.to_thread(file_back_completed_group, group)')
        transport = source / 'slime/utils/rollout_transport.py'
        transport.write_text('malloc_trim')
        self.assertEqual(m.validate_source(source), hashlib.sha256(launcher.read_bytes()).hexdigest())
        recipe.write_text('experimental reward rule')
        with self.assertRaisesRegex(ValueError, 'recipe changed'):
            m.validate_source(source)
        recipe.write_text('reference reward rule')
        transport.write_text('old transport')
        with self.assertRaisesRegex(ValueError, 'collection-time'):
            m.validate_source(source)

    def test_existing_allocation_limits_and_ownership(self):
        now = m.datetime(2030, 1, 1, 7).timestamp()
        self.assertEqual(m.allocation(self.info(), '42', now)['maximum_seconds'], 3420)
        cases = [dict(JobState='PENDING'), dict(UserId='other(-1)'), dict(NumNodes='2'),
                 dict(NumCPUs='4'), dict(AllocTRES='cpu=8,mem=120G,gres/gpu=2,gres/gpu:h200=2'),
                 dict(AllocTRES='cpu=8,mem=240G,gres/gpu=2,gres/gpu:a100=2')]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                m.allocation(self.info(**changes), '42', now)
        with self.assertRaises(ValueError):
            m.allocation(self.info(), '42', now + 3300)

    def lineage(self):
        old, new = self.root / 'old', self.root / 'new'
        old.mkdir(); new.mkdir()
        (old / 'latest_checkpointed_iteration.txt').write_text('5')
        (new / 'launch_manifest.json').write_text(json.dumps({'resume_from': str(old)}))
        return old, new, dict(run_directory=str(new), last_valid_checkpoint=str(old / 'iter_0000005'))

    def test_selects_completed_checkpoint_in_recorded_lineage(self):
        old, new, state = self.lineage()
        (new / 'iter_0000006').mkdir()  # Partial save has no completion marker.
        self.assertEqual(m.checkpoint_root(m.lineage(state)), (old, 5))
        (new / 'latest_checkpointed_iteration.txt').write_text('6')
        self.assertEqual(m.checkpoint_root(m.lineage(state)), (new, 6))
        state['last_valid_checkpoint'] = str(self.root / 'unrelated' / 'iter_0000009')
        with self.assertRaises(ValueError):
            m.lineage(state)

    def batch(self, root, iteration=6):
        folder = root / 'rollout_recovery'
        folder.mkdir(exist_ok=True)
        batch = folder / f'{iteration}.pt'
        with zipfile.ZipFile(batch, 'w') as archive:
            archive.writestr('batch/data.pkl', b'test metadata only')
        return batch

    def test_replays_saved_collection_and_does_not_repeat_trained_one(self):
        old, new, state = self.lineage()
        batch = self.batch(old)
        provenance = dict(rollout_id_zero_based=6, preceding_checkpoint_iteration=5,
                          preceding_checkpoint_root=str(old), batch_bytes=batch.stat().st_size,
                          submitted_groups=144)
        batch.with_suffix('.provenance.json').write_text(json.dumps(provenance))
        state['pending_replay_batch'] = str(batch)
        replay = m.replay_batch(state, [new, old], old, 5)
        self.assertEqual(replay['consumed_groups'], 144)
        self.assertIsNone(m.replay_batch(state, [new, old], new, 6))
        provenance['preceding_checkpoint_iteration'] = 4
        batch.with_suffix('.provenance.json').write_text(json.dumps(provenance))
        with self.assertRaises(ValueError):
            m.replay_batch(state, [new, old], old, 5)

    def test_recovers_submitted_count_from_completed_collection_log(self):
        old, new, state = self.lineage()
        batch = self.batch(new)
        (new / 'progress.log').write_text('[GenerateProgress] rollout=7/90 event=done groups=48/48 completed_groups=121 pending_groups=23 elapsed_secs=4064.5\n')
        self.assertEqual(m.replay_batch(state, [new, old], old, 5)['consumed_groups'], 144)
        (new / 'progress.log').write_text('collection unfinished\n')
        with self.assertRaises(ValueError):
            m.replay_batch(state, [new, old], old, 5)
        batch.write_bytes(b'partial interrupted save')
        with self.assertRaises(zipfile.BadZipFile):
            m.replay_batch(state, [new, old], old, 5)

    def test_busy_allocation_cannot_launch_second_trainer(self):
        busy = m.active_steps('42.interactive|interactive\n42.extern|extern\n42.3|bash\n')
        self.assertEqual(busy, ['42.3|bash'])
        with patch.object(m.subprocess, 'Popen') as process, self.assertRaises(ValueError):
            m.launch({'active_steps': busy}, {}, None)
        process.assert_not_called()

    def test_launch_passes_replay_controls_and_records_new_run_without_claiming_restore(self):
        from types import SimpleNamespace
        runtime = self.root / 'runtime'
        (runtime / 'runs').mkdir(parents=True)
        run = runtime / 'runs/openwebrl-4b-reference-42-test'
        checkpoint = str(self.root / 'old/iter_0000005')
        state = dict(full_resume_verified_checkpoint=checkpoint, full_resume_verified=True)
        args = SimpleNamespace(job_id='42', state=self.root / 'state.json')
        plan = dict(active_steps=[], replay=dict(batch=str(self.root / '6.pt'), rollout_id=6, consumed_groups=144),
                    command=['srun', '--jobid=42'], allocation=dict(host='g021', maximum_seconds=600),
                    source=str(self.root), resume_from=str(self.root / 'old'),
                    checkpoint_report=dict(checkpoint=checkpoint, completed_optimizer_updates=90),
                    wandb_run_id='sameid', wandb_url='https://wandb.ai/test/project/runs/sameid')
        calls = []

        class Process:
            returncode = None
            polls = 0

            def poll(self):
                self.polls += 1
                if self.polls > 1:
                    self.returncode = 0
                return self.returncode

        def start(command, **kwargs):
            calls.append((command, kwargs['env']))
            if len(calls) == 1:
                run.mkdir()
                (run / 'launch_manifest.json').write_text(json.dumps({'resume_from': plan['resume_from']}))
                (run / 'progress.log').write_text('loading checkpoint')
            return Process()

        with patch.object(m, 'RUNTIME', runtime), patch.object(m.subprocess, 'Popen', side_effect=start), patch.object(m.time, 'sleep'):
            self.assertEqual(m.launch(plan, state, args), 0)
        self.assertEqual(calls[0][1]['OPENWEBRL_REPLAY_CONSUMED_GROUPS'], '144')
        self.assertEqual(calls[0][1]['OPENWEBRL_REPLAY_ROLLOUT_ID'], '6')
        self.assertIn('--jobid=42', calls[1][0])
        saved = json.loads(args.state.read_text())
        self.assertEqual(saved['run_directory'], str(run))
        self.assertEqual(saved['durable_optimizer_updates'], 90)
        self.assertEqual(saved['wandb_run_id'], 'sameid')
        self.assertIn('Resume launched', saved['status_note'])
        self.assertTrue((run / 'resume_plan.json').is_file())

    def test_srun_stays_in_exact_existing_job_and_cleans_stale_resume_controls(self):
        command = m.step_command('42', 8)
        self.assertIn('--jobid=42', command)
        self.assertIn('--exact', command)
        self.assertNotIn('sbatch', command)
        self.assertNotIn('salloc', command)
        with patch.dict(os.environ, {'WANDB_RUN_ID': 'old', 'OPENWEBRL_REPLAY_FIRST_BATCH': 'old.pt',
                                     'OPENWEBRL_VERIFY_RESUME_ONLY': '1', 'WANDB_API_KEY': 'private'}):
            env = m.clean_environment()
        self.assertNotIn('WANDB_RUN_ID', env)
        self.assertNotIn('OPENWEBRL_REPLAY_FIRST_BATCH', env)
        self.assertNotIn('OPENWEBRL_VERIFY_RESUME_ONLY', env)
        self.assertEqual(env['WANDB_API_KEY'], 'private')


    def test_four_gpu_profile_requires_matching_existing_resources(self):
        now = m.datetime(2030, 1, 1, 7).timestamp()
        with self.assertRaisesRegex(ValueError, '4 GPUs'):
            m.allocation(self.info(), '42', now, requested_gpus=4)
        info = self.info(NumCPUs='16', AllocTRES='cpu=16,mem=480G,node=1,gres/gpu=4,gres/gpu:h200=4')
        job = m.allocation(info, '42', now, requested_gpus=4)
        self.assertEqual((job['gpus'], job['cpus'], job['tensor_parallel_size']), (4, 16, 4))
        self.assertEqual(job['maximum_seconds'], 3420)
        self.assertIn('--gres=gpu:4', m.step_command('42', 16, 4))
        with self.assertRaises(ValueError):
            m.allocation(self.info(NumCPUs='16', AllocTRES='cpu=16,mem=240G,gres/gpu=4,gres/gpu:h200=4'), '42', now, requested_gpus=4)

    def verification_plan(self):
        return dict(active_steps=[], replay=None, verify_resume_only=True,
                    command=['srun', '--jobid=42', '--gres=gpu:4'],
                    allocation=dict(job_id='42', host='g022', gpus=4, maximum_seconds=3600),
                    source=str(self.root / 'source'), launcher_sha256='source-hash',
                    resume_from=str(self.root / 'old'),
                    checkpoint_report=dict(checkpoint=str(self.root / 'old/iter_0000008'), iteration=8, completed_optimizer_updates=130),
                    wandb_run_id='sameid', wandb_url='https://wandb.ai/test/project/runs/sameid')

    def test_verification_does_not_replace_pointer_or_replay_and_requires_real_report(self):
        from types import SimpleNamespace
        runtime = self.root / 'runtime'
        (runtime / 'runs').mkdir(parents=True)
        run = runtime / 'runs/openwebrl-4b-resume-check-42-test'
        args = SimpleNamespace(job_id='42', state=self.root / 'state.json')
        original = {'run_directory': 'existing-training-run', 'durable_optimizer_updates': 130}
        args.state.write_text(json.dumps(original))
        plan = self.verification_plan()
        calls = []
        class Process:
            returncode = None
            polls = 0
            def poll(self):
                self.polls += 1
                if self.polls > 1: self.returncode = 0
                return self.returncode
            def wait(self, timeout): return 0
        def start(command, **kwargs):
            calls.append((command, kwargs['env']))
            if len(calls) == 1:
                run.mkdir()
                (run / 'launch_manifest.json').write_text(json.dumps({'resume_from': plan['resume_from']}))
                (run / 'resume_verification.json').write_text(json.dumps({
                    'full_model_and_optimizer_load': 'passed', 'loaded_iteration': 8,
                    'next_rollout_id': 9, 'gpus': 4, 'source_checkpoint_root': plan['resume_from'],
                    'optimizer_updates_executed': 0, 'browser_collections_executed': 0}))
            return Process()
        with patch.object(m, 'RUNTIME', runtime), patch.object(m.subprocess, 'Popen', side_effect=start), patch.object(m.time, 'sleep'):
            self.assertEqual(m.launch(plan, original, args), 0)
            self.assertEqual(json.loads(args.state.read_text()), original)
            self.assertNotIn('OPENWEBRL_REPLAY_FIRST_BATCH', calls[0][1])
            self.assertIn('--gres=gpu:4', calls[1][0])
            m.validate_verification_receipt(plan, '42')
            changed = dict(plan, launcher_sha256='changed-source')
            with self.assertRaises(ValueError): m.validate_verification_receipt(changed, '42')
            changed = dict(plan, checkpoint_report=dict(plan['checkpoint_report'], checkpoint='another-checkpoint'))
            with self.assertRaises(ValueError): m.validate_verification_receipt(changed, '42')
            report = json.loads((run / 'resume_verification.json').read_text())
            report['gpus'] = 2
            (run / 'resume_verification.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError): m.record_verification(plan, '42', run)

    def test_four_gpu_training_requires_verification_before_launch(self):
        from types import SimpleNamespace
        plan = self.verification_plan()
        plan['verify_resume_only'] = False
        with patch.object(m, 'RUNTIME', self.root), patch.object(m.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(ValueError, 'successful --verify-resume-only'):
                m.launch(plan, {}, SimpleNamespace(job_id='42'))
            process.assert_not_called()


    def test_live_previous_allocation_cannot_fork_training_lineage(self):
        plan = self.verification_plan()
        plan.update(verify_resume_only=False, other_live_steps=['41.3|trainer'])
        with patch.object(m.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(ValueError, 'still active in another allocation'):
                m.launch(plan, {}, None)
            process.assert_not_called()


    def test_four_gpu_verification_preflight_forwards_topology_and_skips_replay(self):
        from types import SimpleNamespace
        source = self.root / 'source'
        (source / 'scripts').mkdir(parents=True)
        (source / 'scripts/run_small_baseline.py').write_text('OPENWEBRL_RESUME_TOPOLOGY_V1 = True')
        root = self.root / 'run'
        root.mkdir()
        state_file = self.root / 'state.json'
        state_file.write_text(json.dumps({'run_directory': str(root), 'allocation': '41',
            'source_directory': str(source), 'wandb_run_id': 'sameid',
            'wandb_url': 'https://wandb.ai/test/project/runs/sameid', 'scheduler_offset_updates': 1}))
        report = dict(iteration=8, completed_optimizer_updates=130, checkpoint=str(root / 'iter_0000008'))
        args = SimpleNamespace(state=state_file, source=None, job_id='42', gpus=4,
                               verify_resume_only=True, wandb_run_id=None, env_file=self.root / '.env')
        def capture(command):
            if command[0] == 'scontrol':
                return self.info(NumCPUs='16', AllocTRES='cpu=16,mem=480G,node=1,gres/gpu=4,gres/gpu:h200=4')
            if command[0] == 'squeue': return '41.3|trainer' if '--user' in command else ''
            return json.dumps(report)
        with patch.object(m, 'capture', side_effect=capture), patch.object(m, 'validate_source', return_value='hash'), patch.object(m, 'lineage', return_value=[root]), patch.object(m, 'checkpoint_root', return_value=(root, 8)), patch.object(m, 'replay_batch') as replay:
            plan, state = m.prepare(args)
        self.assertIn('--gres=gpu:4', plan['command'])
        self.assertIn('--gpus 4 --verify-resume-only', plan['command'][-1])
        self.assertTrue(plan['verify_resume_only'])
        self.assertEqual(plan['other_live_steps'], ['41.3|trainer'])
        self.assertIsNone(plan['replay'])
        replay.assert_not_called()


    def test_batch_driver_does_not_ignore_other_compute_steps(self):
        rows = '42.batch|batch\n42.extern|extern\n42.0|trainer\n'
        self.assertEqual(m.active_steps(rows), ['42.batch|batch', '42.0|trainer'])
        self.assertEqual(m.active_steps(rows, allow_batch=True), ['42.0|trainer'])
        self.assertEqual(m.active_steps('42.batch|batch\n42.extern|extern', allow_batch=True), [])


if __name__ == '__main__':
    unittest.main()
