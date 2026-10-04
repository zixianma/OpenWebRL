"""CPU regression tests for policy provenance across two ARM RL collections."""
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl import arm_turn_bonus as bonus
from openwebrl import arm_turn_bonus_cycles as cycles
sys.path.insert(0,str(Path('scripts').resolve()))
import run_arm_turn_bonus_cycles as runner


class Cycles(unittest.TestCase):
    def test_fresh_native_profile_and_hook_order(self):
        p=runner.plan('CPU_CHECK')
        env=dict(os.environ,**p['environment'],DRY_RUN='1')
        argv=shlex.split(subprocess.check_output(p['command'],env=env,text=True))
        self.assertEqual(argv[argv.index('--load')+1],str(runner.INITIAL))
        self.assertNotIn('--ckpt-step',argv)
        self.assertNotIn('--eval-interval',argv)
        self.assertEqual(argv[argv.index('--num-rollout')+1],'4')
        self.assertEqual(argv[argv.index('--tensor-model-parallel-size')+1],'4')
        self.assertEqual(p['browser_config']['browser_rollout_concurrency'],48)
        self.assertEqual(env['BROWSER_CONCURRENCY'],'48')
        self.assertNotIn('OPENWEBRL_STOP_AFTER_SAVED_ROLLOUT',p['environment'])
        text=(runner.SOURCE/'train.py').read_text(); ast.parse(text)
        self.assertLess(text.index('if before_collection(args, rollout_id):'),text.index('rollout_manager.generate.remote(rollout_id)'))
        self.assertLess(text.index('before_training()'),text.index('rollout_manager.offload.remote()'))
        self.assertLess(text.index('save(rollout_id)\n'),text.index('if after_checkpoint(args, rollout_id):'))
        for name in json.loads((runner.SOURCE/'reference_manifest.json').read_text())['arm_turn_bonus_calibration']['changed_files_sha256']:
            if name.endswith('.py'): ast.parse((runner.SOURCE/name).read_text())

    def test_two_collections_reset_records_keep_provenance_and_preserve_first(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); config_path=root/'arm-config.json'
            excluded=root/'eval.jsonl'; excluded.write_text('{"task_id":"heldout","intent":"held out"}\n')
            current=dict(runner.plan('CPU_CHECK')['arm_config'],run_output=folder,
                deadline_epoch_seconds=time.time()+10000,evaluation_tasks=str(excluded))
            bonus.write_json(config_path,current)
            args=SimpleNamespace(save=str(root/'runtime'),start_rollout_id=0,num_rollout=4,
                judge_api_model='gpt-4.1',judge_prompt_variant='action_history',advantage_estimator='grpo',grpo_std_normalization=True)
            try:
                with patch.dict(os.environ,OPENWEBRL_ARM_TURN_BONUS_CONFIG=str(config_path)),patch.object(cycles,'teacher_phase') as phase,patch.object(bonus,'log_progress'):
                    self.assertFalse(cycles.before_collection(args,0))
                    first=cycles.reset_collection(args,0)
                    first['records'].append({'sentinel':'first-only'})
                    oldpolicy=first['config']['policy_id']
                    oldconfig=(root/'iterations/0000/arm-config.json').read_bytes()
                    checkpoint=root/'runtime/iter_0000000'; checkpoint.mkdir(parents=True)
                    (checkpoint/'common.pt').write_text('fixture')
                    cycles.before_training()
                    self.assertFalse(cycles.after_checkpoint(args,0))
                    self.assertFalse(cycles.before_collection(args,1))
                    second=cycles.reset_collection(args,1)
                    self.assertEqual(second['records'],[])
                    self.assertNotEqual(second['config']['policy_id'],oldpolicy)
                    self.assertEqual(second['config']['checkpoint'],str(checkpoint))
                    self.assertEqual((root/'iterations/0000/arm-config.json').read_bytes(),oldconfig)
                    self.assertEqual(first['records'],[{'sentinel':'first-only'}])
                    self.assertEqual([x.args[1] for x in phase.call_args_list],['collection','training','collection'])
                    stale=SimpleNamespace(remove_sample=False,metadata=dict(arm_turn_bonus=dict(eligible=True,
                        policy_id=oldpolicy,executed_index=0,selected_index=0,unit_bonus=.8)))
                    with self.assertRaisesRegex(ValueError,'provenance'):
                        bonus.unit_bonus(stale,second['config']['policy_id'])
                    with self.assertRaisesRegex(ValueError,'version mismatch'): cycles.reset_collection(args,0)
                    second['config']['deadline_epoch_seconds']=time.time()+100
                    bonus.write_json(config_path,second['config'])
                    self.assertTrue(cycles.before_collection(args,2))
                    self.assertFalse((root/'iterations/0002').exists())
            finally: bonus._STATE=None

    def test_phase_ack_must_match_iteration_and_phase(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            current=dict(run_output=folder,rollout_id=1,deadline_epoch_seconds=time.time()+60)
            bonus.write_json(root/'teacher-ready.json',dict(rollout_id=0,phase='training'))
            def complete(_): bonus.write_json(root/'teacher-ready.json',dict(rollout_id=1,phase='training'))
            with patch.object(cycles.time,'sleep',side_effect=complete) as pause:
                cycles.teacher_phase(current,'training')
                self.assertEqual(pause.call_count,1)
            self.assertEqual(json.loads((root/'teacher-request.json').read_text()),dict(rollout_id=1,phase='training'))


if __name__=='__main__': unittest.main()
