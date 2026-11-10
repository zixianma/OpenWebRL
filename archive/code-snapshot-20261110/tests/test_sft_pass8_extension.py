import copy
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_sft_pass8_extension.py'
SPEC = importlib.util.spec_from_file_location('pass8_launch_under_test',SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class Pass8ApprovalTest(unittest.TestCase):
    def setUp(self):
        self.plan=dict(resources=module.RESOURCES,limits=module.LIMITS)
        self.approval=dict(approved=True,plan_sha256=module.digest(self.plan),resources=module.RESOURCES,
            limits=module.LIMITS,attempts=[dict(job_id='20',shard=0,time_limit_seconds=1000)])

    def test_retry_charges_all_elapsed_and_does_not_reset_budget(self):
        self.assertEqual(module.verify_approval(self.plan,self.approval,'20',0,35000),1000)
        with self.assertRaises(ValueError):module.verify_approval(self.plan,self.approval,'20',0,35001)

    def test_exact_plan_api_resources_and_positive_approval_required(self):
        for key,value in (('approved',False),('plan_sha256','changed'),('limits',{}),('resources',{})):
            changed=copy.deepcopy(self.approval);changed[key]=value
            with self.assertRaises(ValueError):module.verify_approval(self.plan,changed,'20',0,0)
        with self.assertRaises(ValueError):module.verify_approval(self.plan,self.approval,'20',1,0)

    def test_duplicate_job_and_invalid_duration_rejected(self):
        changed=copy.deepcopy(self.approval);changed['attempts']*=2
        with self.assertRaises(ValueError):module.verify_approval(self.plan,changed,'20',0,0)
        for duration in (0,-1,36001,True):
            changed=copy.deepcopy(self.approval);changed['attempts'][0]['time_limit_seconds']=duration
            with self.assertRaises(ValueError):module.verify_approval(self.plan,changed,'20',0,0)

    def test_total_caps_are_not_multiplied_by_attempt_count(self):
        self.assertEqual(module.RESOURCES['jobs']*module.RESOURCES['total_seconds_per_shard']/3600,40)
        self.assertEqual(module.LIMITS['judge_usd_per_shard']*4,module.LIMITS['judge_usd_total'])
        self.assertEqual(module.LIMITS['judge_http_calls_per_shard']*4,19200)
        self.assertEqual(module.LIMITS['browsers_per_shard']*4,6400)


class Pass8SchedulerTest(unittest.TestCase):
    def test_accounting_is_per_shard_and_includes_every_prior_failure(self):
        approval=dict(attempts=[dict(job_id='1',shard=0),dict(job_id='2',shard=0),
            dict(job_id='3',shard=1),dict(job_id='4',shard=0)])
        query=Mock(return_value='1|FAILED|345|cpu=8,gres/gpu=1|\n2|TIMEOUT|456|cpu=8,gres/gpu=1|\n')
        seconds,proof=module.accounted_seconds(approval,0,'4',query=query)
        self.assertEqual(seconds,801);self.assertEqual(len(proof),2)
        self.assertIn('1,2',query.call_args[0][0])

    def test_nonfinal_missing_or_duplicate_accounting_blocks_retry(self):
        approval=dict(attempts=[dict(job_id='1',shard=0),dict(job_id='2',shard=0)])
        for response in ('','1|RUNNING|100|cpu=8|\n','1|FAILED|-1|cpu=8|\n',
                         '1|FAILED|100|cpu=8|\n1|FAILED|100|cpu=8|\n'):
            with self.assertRaises(ValueError):module.accounted_seconds(approval,0,'2',query=Mock(return_value=response))

    def test_exact_owned_resource_profile_and_remaining_time(self):
        now=datetime(2026,10,6,12).timestamp()
        info='JobId=22 JobState=RUNNING UserId=user(1001) NumNodes=1 NumCPUs=8 AllocTRES=cpu=8,mem=120G,gres/gpu=1,gres/gpu:h200=1 TimeLimit=10:00:00 EndTime=2026-10-06T22:00:00'
        self.assertEqual(module.validate_allocation(info,'22',36000,now=now,uid=1001),35820)
        for old,new in (('NumCPUs=8','NumCPUs=16'),('mem=120G','mem=240G'),
                        ('gres/gpu:h200=1','gres/gpu:h100=1'),('JobState=RUNNING','JobState=PENDING'),
                        ('UserId=user(1001)','UserId=user(1002)')):
            with self.assertRaises(ValueError):module.validate_allocation(info.replace(old,new),'22',36000,now=now,uid=1001)
        with self.assertRaises(ValueError):module.validate_allocation(info,'22',35000,now=now,uid=1001)


class Pass8LaunchTest(unittest.TestCase):
    def test_partition_has_every_task_once_and_identical_historical_order(self):
        ids=[str(i) for i in range(707)];shards=module.partition(ids)
        self.assertEqual([len(s) for s in shards],[177,177,177,176])
        self.assertEqual([shards[i%4][i//4] for i in range(707)],ids)
        with self.assertRaises(ValueError):module.partition(ids[:-1])
        with self.assertRaises(ValueError):module.partition(ids[:-1]+ids[:1])
        with self.assertRaises(ValueError):module.partition([str(i) for i in range(2000)])

    @staticmethod
    def historical_fixture():
        ids=[f'task-{i}' for i in range(2000)];records=[]
        for i,task in enumerate(ids):
            for attempt in range(5):
                valid=not (attempt==4 and (i<19 or i>=1975))
                records.append(dict(task_id=task,attempt=attempt,valid=valid,
                    reward=(1. if i<1293 and attempt==0 else 0.) if valid else None))
        return ids,records

    def test_select_all_707_misses_including25_invalid_and_skip1293_successes(self):
        ids,records=self.historical_fixture();chosen,proof=module.select_misses(ids,records)
        self.assertEqual(chosen,ids[1293:])
        self.assertEqual(proof['zero_success_with_invalid_tasks'],25)
        self.assertEqual(proof['five_valid_failure_tasks'],682)
        self.assertEqual(proof['new_primary_episodes'],2121)
        self.assertEqual(proof['successful_tasks_replayed'],0)

    def test_miss_selection_rejects_changed_history_and_missing_attempts(self):
        ids,records=self.historical_fixture()
        with self.assertRaises(ValueError):module.select_misses(ids,records[:-1])
        with self.assertRaises(ValueError):module.select_misses(ids,records+[records[0]])
        records[-5]['reward']=1.
        with self.assertRaises(ValueError):module.select_misses(ids,records)

    def test_native_stage_runs_three_samples_without_training_or_selector(self):
        original=dict(source='/old/source',stage=dict(output='/old/output',source='/old/source',
            command=['bash','/old/source/scripts/run_h200_browser.sh','--eval-config','old',
                     '--eval-function-path','old','--wandb-group','old','--wandb-project','wrong'],
            environment={'OPENWEBRL_TASK_SCREEN_CONFIG':'old','NUM_ROLLOUT':'0'}))
        historical=[str(i) for i in range(2000)];extension=historical[1293:]
        plan=dict(original_plan='/does/not/matter',shards=module.partition(extension),
            historical_task_order=historical,extension_task_order=extension,
            old_record_roots=['old'+str(i) for i in range(8)])
        with patch.object(module,'read',return_value=original):launch=module.make_launch(plan,2,'22')
        env=launch['stage']['environment'];cfg=launch['collector_config']
        self.assertEqual((env['NUM_GPUS'],env['TP_SIZE'],env['NUM_ROLLOUT'],env['N_SAMPLES']),('1','1','0','3'))
        self.assertEqual(env['WANDB_PROJECT'],'openwebrl-evals')
        self.assertNotIn('OPENWEBRL_TASK_SCREEN_CONFIG',env)
        self.assertEqual(cfg['old_record_roots'],plan['old_record_roots'])
        self.assertEqual(len(cfg['task_order']),177)
        self.assertEqual(len(cfg['historical_task_order']),2000)
        self.assertEqual(len(cfg['extension_task_order']),707)
        self.assertTrue(set(cfg['task_order']).isdisjoint(historical[:1293]))
        self.assertEqual((cfg['seed'],cfg['max_steps']),(20260930,15))
        self.assertIn('openwebrl.arm_task_pass8.generate_rollout',launch['stage']['command'])
        self.assertNotIn('selector',launch['stage'])

    def test_frozen_files_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plan.json';module.freeze(path,{'a':1});module.freeze(path,{'a':1})
            with self.assertRaises(ValueError):module.freeze(path,{'a':2})
            self.assertEqual(json.loads(path.read_text()),{'a':1})

    def test_scoped_process_shutdown_rejects_recycled_identity(self):
        child=Mock(pid=123);child.poll.return_value=None
        with patch.object(module,'process_identity',return_value=(123,123,11)),patch.object(module.os,'killpg') as kill:
            with self.assertRaises(RuntimeError):module.stop_owned(child,(123,123,10))
            kill.assert_not_called()
        child.poll.return_value=0
        with patch.object(module.os,'killpg') as kill:
            module.stop_owned(child,(123,123,10));kill.assert_not_called()

    def test_shell_guard_allows_reward_flag_and_rejects_destructive_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'scripts').mkdir()
            for name in ('h200_env.sh','run_h200_browser.sh','run_sft_pass8_extension_1gpu.sbatch'):
                (root/'scripts'/name).write_text('#!/bin/bash\ncmd --custom-rm-path reward\n')
            module.validate_launch_shells(root)
            (root/'scripts/run_h200_browser.sh').write_text('cmd\nrm -rf /tmp/anything\n')
            with self.assertRaises(ValueError):module.validate_launch_shells(root)


if __name__=='__main__':unittest.main()
