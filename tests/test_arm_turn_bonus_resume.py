"""Resume provenance and topology checks using the prepared native recipe."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path('scripts').resolve()))
import resume_arm_turn_bonus as resume


class Resume(unittest.TestCase):
    def test_eight_hour_continuation_preserves_lineage_and_monitors_full_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            initial=self.fixture(Path(tmp))
            with patch.object(resume,'origin',return_value=initial):
                p=resume.plan('TEST8H',initial['root'],480)
            self.assertEqual(p['requested_resources']['hours'],8)
            self.assertEqual(p['requested_resources']['gpu_hours'],32)
            self.assertEqual(p['requested_iterations'],8)
            self.assertEqual(p['environment']['NUM_ROLLOUT'],'9')
            self.assertEqual(p['environment']['WANDB_RUN_ID'],'original-arm-lineage')
            self.assertEqual(p['arm_config']['minimum_cycle_seconds'],4500)
            self.assertIn('--use-checkpoint-opt-param-scheduler',p['command'])

    def test_eight_gpu_resume_preserves_recipe_and_requires_matching_topology(self):
        with tempfile.TemporaryDirectory() as tmp:
            initial=self.fixture(Path(tmp))
            with patch.object(resume,'origin',return_value=initial):
                p=resume.plan('TEST8GPU',initial['root'],480,gpus=8)
                self.assertEqual(p['requested_resources']['gpu_hours'],64)
                self.assertEqual(p['requested_resources']['memory_gib'],960)
                self.assertEqual(p['environment']['NUM_GPUS'],'8')
                self.assertEqual(p['environment']['TP_SIZE'],'8')
                self.assertEqual(p['environment']['GLOBAL_BATCH_SIZE'],'256')
                self.assertEqual(p['environment']['BROWSER_CONCURRENCY'],'64')
                self.assertEqual(p['requested_resources']['cpus'],64)
                self.assertEqual(p['initial_optimizer_updates'],16)
                self.assertTrue(p['gpu_restore_output'].endswith('-8gpu'))
                with self.assertRaises(ValueError):
                    resume.plan('TEST',initial['root'],480,gpus=6)

    def test_sixteen_hour_four_gpu_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            initial=self.fixture(Path(tmp))
            with patch.object(resume,'origin',return_value=initial):
                p=resume.plan('TEST16H',initial['root'],960,gpus=4)
                self.assertEqual(p['requested_resources']['gpu_hours'],64)
                self.assertEqual(p['requested_resources']['hours'],16)
                self.assertEqual(p['requested_iterations'],16)
                self.assertEqual(p['environment']['BROWSER_CONCURRENCY'],'32')
                with self.assertRaises(ValueError):
                    resume.plan('TEST',initial['root'],960,gpus=8)

    def fixture(self,root):
        checkpoint=root/'origin/runtime/iter_0000000'; checkpoint.mkdir(parents=True)
        (checkpoint/'common.pt').write_bytes(b'untouched-model-and-optimizer')
        cursor=root/'origin/runtime/rollout/global_dataset_state_dict_0.pt'
        cursor.parent.mkdir();cursor.write_bytes(b'exact-cursor')
        return dict(root=str(root/'origin'),manifest=dict(wandb_run_id='original-arm-lineage'),
            checkpoint_report=dict(checkpoint=str(checkpoint),dataset_cursor=str(cursor),
                iteration=0,completed_optimizer_updates=16),identity_sha256={str(cursor):hashlib.sha256(cursor.read_bytes()).hexdigest()})

    def test_resume_preserves_optimizer_cursor_lineage_and_uses_32_browsers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); initial=self.fixture(root)
            with patch.object(resume,'origin',return_value=initial):
                p=resume.plan('TEST',initial['root'])
                self.assertFalse(p['fresh_optimizer'])
                self.assertEqual(p['initial_optimizer_updates'],16)
                self.assertEqual(p['start_rollout_id'],1)
                self.assertEqual(p['environment']['NUM_ROLLOUT'],'5')
                self.assertEqual(p['environment']['WANDB_RUN_ID'],'original-arm-lineage')
                self.assertEqual(p['environment']['WANDB_RESUME'],'must')
                self.assertEqual(p['environment']['SLIME_LOAD_CHECKPOINT'],str(root/'origin/runtime'))
                self.assertEqual(p['arm_config']['beta'],.5)
                self.assertEqual(p['arm_config']['scored_fraction'],.2)
                self.assertEqual(p['browser_config']['browser_rollout_concurrency'],32)
                self.assertEqual(p['requested_resources']['gpu_hours'],16)
                self.assertEqual(p['environment']['BROWSER_CONCURRENCY'],'32')
                self.assertEqual(p['environment']['TP_SIZE'],'4')
                self.assertIn('--use-checkpoint-opt-param-scheduler',p['command'])
                resume.validate_resume_plan(p)
                p['output']=str(root/'continuation');Path(p['output']).mkdir()
                resume.stage_checkpoint_reference(p)
                linked=Path(p['output'])/'runtime/iter_0000000'
                self.assertTrue(linked.is_symlink())
                self.assertEqual((linked/'common.pt').read_bytes(),b'untouched-model-and-optimizer')
                saved_cursor=Path(p['output'])/'runtime/rollout/global_dataset_state_dict_0.pt'
                self.assertEqual(saved_cursor.read_bytes(),b'exact-cursor')
                self.assertFalse(saved_cursor.is_symlink())
                with self.assertRaises(FileExistsError): resume.stage_checkpoint_reference(p)

    def test_changed_origin_or_wrong_gpu_receipt_blocks_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); initial=self.fixture(root)
            with patch.object(resume,'origin',return_value=initial):
                p=resume.plan('TEST',initial['root'])
                p['resume_restore_receipt']=str(root/'receipt.json')
                receipt=dict(passed=True,job_id='TEST',source=p['source'],
                    source_checkpoint_root=str(root/'origin/runtime'),
                    checkpoint_completed_optimizer_updates=16,loaded_iteration=0,next_rollout_id=1,
                    gpus=4,optimizer_updates_executed=0,browser_collections_executed=0,
                    continuation_identity_sha256=resume.continuation_identity(p))
                path=Path(p['resume_restore_receipt']);path.write_text(json.dumps(receipt))
                resume.validate_resume_plan(p,require_gpu_restore=True)
                for bad in [dict(gpus=2),dict(job_id='OTHER'),dict(next_rollout_id=0),dict(optimizer_updates_executed=1),dict(continuation_identity_sha256='old-launch-args')]:
                    path.write_text(json.dumps(dict(receipt,**bad)))
                    with self.assertRaisesRegex(ValueError,'GPU restore receipt'):
                        resume.validate_resume_plan(p,require_gpu_restore=True)
            with patch.object(resume,'origin',return_value=dict(initial,identity_sha256={'changed':'cursor'})):
                with self.assertRaisesRegex(ValueError,'changed after planning'): resume.validate_resume_plan(p)


if __name__=='__main__': unittest.main()
