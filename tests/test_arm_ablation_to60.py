"""Continuation boundaries, saved beta replay, and remaining-budget checks."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_arm_ablation_to60 as continuation
import monitor_arm_turn_bonus as monitor


class ContinuationTests(unittest.TestCase):
    def test_monitor_follows_b_recovery_and_new_ablation_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Path(tmp)
            for job,stem,target in [('331778','arm-gate-b-recovery',70),
                                    ('331995','arm-ablation-weight-recovery',30)]:
                controller=runtime/f'evaluations/{stem}-{job}'
                controller.mkdir(parents=True)
                expected=runtime/f'evaluations/arm-failure-additive-{job}-tp2-iter{target}'
                (controller/'controller-plan.json').write_text(json.dumps(dict(
                    job_id=job,current_target=target,training_root=str(expected))))
                with patch.object(monitor,'RUNTIME',runtime):self.assertEqual(monitor.run_root(job),expected)

    def test_existing_beta_budget_is_not_reset(self):
        report = continuation.check_recovery_budget('330278|1729|FAILED\n')
        self.assertEqual(report['requested_seconds'],48600)
        self.assertEqual(report['previous_used_seconds']+report['requested_seconds'],50329)
        for text in ['330278|1729|RUNNING\n','330278|1900|FAILED\n','']:
            with self.assertRaises(ValueError): continuation.check_recovery_budget(text)

    def test_both_initial_plans_preserve_scientific_recipe_and_counters(self):
        for profile,variant,updates in [('weight-recovery','weight',298),('coverage-to40','coverage',302)]:
            with self.subTest(profile=profile):
                p = continuation.plan('PREPARE',profile,continuation.ORIGINS[variant],30)
                continuation.validate_plan(p)
                self.assertEqual((p['start_rollout_id'],p['initial_optimizer_updates']),(20,updates))
                self.assertFalse(p['fresh_optimizer'])
                self.assertEqual(p['wandb_run_id'],continuation.RUN_IDS[variant])
                self.assertEqual(p['environment']['WANDB_RESUME'],'must')
                self.assertEqual(p['environment']['TP_SIZE'],'2')
                self.assertEqual(p['environment']['OPENWEBRL_CUDA_CACHE_LIMIT_GIB'],'48')
                old=continuation.read(continuation.ORIGINS[variant]/'launch_manifest.json')
                for key in ['beta','scored_fraction','failure_beta','failure_scored_fraction',
                            'failure_group_cap','failure_loss_coefficient','k','seed']:
                    self.assertEqual(p['arm_config'][key],old['arm_config'][key])
                if variant=='weight':
                    self.assertEqual(p['environment']['OPENWEBRL_REPLAY_ROLLOUT_ID'],'20')
                    self.assertEqual(p['replay_provenance']['checkpoint'],str(continuation.FAILED_BETA/'runtime/iter_0000019'))
                else:
                    self.assertNotIn('OPENWEBRL_REPLAY_FIRST_BATCH',p['environment'])
                p['arm_config']['failure_beta']=9
                with self.assertRaises(ValueError): continuation.validate_plan(p)

    def test_no_stage_skipping_or_unapproved_extension(self):
        root=continuation.ORIGINS['weight']
        for target in [20,40,50,60,70]:
            with self.assertRaises(ValueError): continuation.plan('PREPARE','weight-recovery',root,target)
        with self.assertRaises(ValueError): continuation.plan('PREPARE','weight-to60',root,50)

    def test_later_stage_clears_replay_and_stops_at_60(self):
        p=continuation.plan('PREPARE','weight-recovery',continuation.ORIGINS['weight'],30)
        root=continuation.RUNTIME/'evaluations/arm-failure-additive-PREPARE-tp2-iter50'
        origin=dict(manifest=p,checkpoint_report=dict(iteration=49,completed_optimizer_updates=750,
            checkpoint=str(root/'runtime/iter_0000049')))
        with patch.object(continuation,'cached_origin',return_value=origin):
            result=continuation.plan('PREPARE','weight-to60',root,60)
        self.assertEqual(result['target_completed_iterations'],60)
        self.assertEqual(result['requested_iterations'],10)
        self.assertEqual(result['comparison']['endpoint'],60)
        self.assertNotIn('OPENWEBRL_REPLAY_FIRST_BATCH',result['environment'])
        self.assertNotIn('gate_replay_origin',result['arm_config'])
        self.assertEqual(result['evaluation_reserve_seconds'],3600)

    def test_replay_uses_saved_beta_one_auxiliary_and_rejects_weight_changes(self):
        spec=importlib.util.spec_from_file_location('beta_replay',continuation.SOURCES['weight']/'openwebrl/arm_gate_recovery.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);old=base/'iterations/0020';old.mkdir(parents=True)
            output=base/'new';output.mkdir()
            report=dict(rows=256,batch_rows=256,admitted=50,admitted_distinct_tasks=10,
                admitted_distinct_action_counts={'5':50},task_counts={'a':50},label_reasons={'admitted':50},
                outcome_rms=.5,unit_bonus_rms=.2,additive_failure_rows=31,outcome_groups_preserved=48)
            (old/'calibration.json').write_text(json.dumps(report))
            manifest=dict(tensor_file='failure_auxiliary.pt',policy_id='test:rollout20',checkpoint='ckpt20',
                candidate_gate='distinct5',credit_assignment='response_index',rollout_id=20,
                beta=1.,q=.2,expected_windows=1,mixed_groups=48)
            raw=json.dumps(manifest).encode();(old/'failure_auxiliary.json').write_bytes(raw)
            (old/'failure_auxiliary.pt').write_bytes(b'fixture')
            origin=dict(source=str(base),rollout_id=20,calibration_sha256=continuation.sha(old/'calibration.json'),
                auxiliary_manifest_sha256=hashlib.sha256(raw).hexdigest(),auxiliary_bytes=7)
            config=dict(gate_replay_origin=origin,rollout_id=20,output=str(output),failure_beta=1.,
                failure_scored_fraction=.2,**{k:manifest[k] for k in ['policy_id','checkpoint','candidate_gate','credit_assignment']})
            samples=[SimpleNamespace(group_index=i%48) for i in range(256)]
            observed=copy.deepcopy(report);observed['admitted_distinct_action_counts']={5:50}
            fake=SimpleNamespace(write_json=lambda path,data:Path(path).write_text(json.dumps(data)))
            with patch.dict(sys.modules,{'openwebrl.arm_turn_bonus':fake}):
                self.assertTrue(module.replay_auxiliary(SimpleNamespace(global_batch_size=256),samples,dict(config=config),observed))
            self.assertEqual((output/'failure_auxiliary.json').read_bytes(),raw)
            self.assertEqual((output/'failure_auxiliary.pt').resolve(),old/'failure_auxiliary.pt')
            wrong=copy.deepcopy(config);wrong['failure_beta']=.5
            with self.assertRaises(ValueError):
                module.replay_auxiliary(SimpleNamespace(global_batch_size=256),samples,dict(config=wrong),report)


if __name__=='__main__': unittest.main()
