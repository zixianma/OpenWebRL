import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('record_queue',ROOT/'scripts/reward_eval_queue.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class RewardQueueTest(unittest.TestCase):
    def test_cpu_mask_from_observer_is_not_exported_to_new_job(self):
        with patch.dict(os.environ, {'SLURM_CPU_BIND':'mask_cpu:0x80000000',
                                    'SLURM_CPU_BIND_LIST':'0x80000000',
                                    'SLURM_CPU_BIND_TYPE':'mask_cpu', 'RAY_ADDRESS':'local'}):
            env=m.submission_environment()
        self.assertFalse(any(k.startswith('SLURM_CPU_BIND') for k in env))
        self.assertEqual(env['RAY_ADDRESS'],'local')
    def setUp(self):
        self.state=m.initialize([{'train/reward':r,'train/reward_iteration':n,'_step':n}
                                 for n,r in [(19,.6),(20,.59),(21,.58),(22,.57),(23,.55)]],'/frozen')
        self.inventory={'checkpoints':[{'after_training_iteration':37,'checkpoint':'/run/iter_0000036'}]}
    def audit(self,reward=.56,step=620):
        return {'matched':True,'wandb':{'train/reward':reward,'train/reward_iteration':38,'_step':step}}
    def test_fifth_place_targets_generating_checkpoint(self):
        e=m.observe(self.state,self.audit(),self.inventory)
        self.assertEqual(e['completed_training_iterations'],37)
        self.assertEqual(e['checkpoint'],'/run/iter_0000036')
        self.assertEqual(e['trigger_rank'],5)
        self.assertEqual(e['status'],'AWAITING_COMPUTE_APPROVAL')
        self.assertEqual(self.state['record_reward'],.6)
    def test_tie_at_boundary_or_lower_reward_does_not_enqueue(self):
        self.assertIsNone(m.observe(self.state,self.audit(.55),self.inventory))
        self.assertIsNone(m.observe(self.state,self.audit(.54,621),self.inventory))
        self.assertEqual(self.state['record_reward'],.6)
    def test_replay_and_same_checkpoint_are_not_duplicated(self):
        m.observe(self.state,self.audit(),self.inventory)
        self.assertIsNone(m.observe(self.state,self.audit(),self.inventory))
        self.assertIsNone(m.observe(self.state,self.audit(.57,621),self.inventory))
        self.assertEqual(len(self.state['entries']),1)
    def test_unverified_reward_is_rejected(self):
        a=self.audit();a['matched']=False
        with self.assertRaises(ValueError):m.observe(self.state,a,self.inventory)
        self.assertEqual(self.state['last_seen_wandb_history_step'],23)
    def test_missing_checkpoint_does_not_advance_cursor(self):
        with self.assertRaises(ValueError):m.observe(self.state,self.audit(),{'checkpoints':[]})
        self.assertEqual(self.state['last_seen_wandb_history_step'],23)
    def test_nonfinite_reward_is_rejected(self):
        with self.assertRaises(ValueError):m.observe(self.state,self.audit(float('nan')),self.inventory)
    def test_already_evaluated_checkpoint_is_skipped(self):
        self.state['completed_evaluations']=['/run/iter_0000036']
        self.assertIsNone(m.observe(self.state,self.audit(),self.inventory))
        self.assertEqual(len(self.state['entries']),0)
    def test_finished_jobs_still_count_against_budget(self):
        self.state['entries']=[{'status':'COMPLETED','job_id':'42'}]
        self.assertEqual(m.reserved_jobs(self.state),1)
    def test_duplicate_history_does_not_take_two_top_five_places(self):
        rows=list(self.state['reward_history'].values())
        rows.append(dict(rows[0],_step=99))
        state=m.initialize(rows,'/frozen')
        self.assertEqual(len(m.top_five(state)),5)
        self.assertEqual(len({r['train/reward_iteration'] for r in m.top_five(state)}),5)
    def test_submission_budget_and_duplicate_guards(self):
        e=m.observe(self.state,self.audit(),self.inventory)
        approval={'training_run':'qcq7i4ug','browser_env':'local_process','max_jobs':4,'gpus_per_job':2,
                  'max_hours_per_job':2,'user_approval_text':'fixture explicit approval'}
        template=ROOT/'scripts/evaluate_record_checkpoint_2gpu.sbatch'
        cmd=m.submission_command(self.state,e,approval,template)
        self.assertEqual(cmd[0],'sbatch')
        self.assertIn('OPENWEBRL_RECORD_EVAL_ITERATION=37',cmd[2])
        with self.assertRaises(ValueError):m.submission_command(self.state,e,dict(approval,max_jobs=5),template)
        e['status']='SUBMITTED'
        with self.assertRaises(ValueError):m.submission_command(self.state,e,approval,template)
        self.state['entries']=[{'status':'SUBMITTED'}]*4+[dict(e,status='AWAITING_COMPUTE_APPROVAL')]
        with self.assertRaisesRegex(ValueError,'cap reached'):
            m.submission_command(self.state,self.state['entries'][-1],approval,template)

if __name__=='__main__':unittest.main()
