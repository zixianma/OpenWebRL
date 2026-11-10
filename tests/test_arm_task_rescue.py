import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from openwebrl.arm_rescue_yield import stable_seed
from openwebrl.arm_task_rescue import ARM_REVISION, POLICY, SEED, eligible_ids, key, summarize, verify_record
from run_arm_task_rescue import RESOURCES, verify_approval


class RescueArtifacts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def record(self,task='a',attempt=0,valid=True,reward=0.):
        root=self.root/key(task,attempt);root.mkdir(exist_ok=True)
        archive=root/(hashlib.sha256(task.encode()).hexdigest()+'.pt');archive.write_bytes(b'accounting-test-only')
        archive.with_suffix('.json').write_text(json.dumps(dict(task_id=task,judge_model='gpt-4.1',judge_prompt_variant='action_history',metrics={'valid_trajectories':int(valid)},reward_metadata={'combined':reward})))
        trace=root/'selections';trace.mkdir(exist_ok=True)
        (trace/(hashlib.sha256(task.encode()).hexdigest()[:20]+'.jsonl')).write_text(json.dumps(dict(task_id=task,mode='selection',turn=0,selected_index=2,candidate_seeds=list(range(5)),candidates=[dict(thought='reason',action=str(i)) for i in range(5)]))+'\n')
        return dict(task_id=task,attempt=attempt,valid=valid,reward=reward if valid else None,policy_id=POLICY,judge_id='gpt-4.1/action_history',
            phase='guided',mode='arm',selector_calls=1,selector_fallback_turns=0,artifact=str(root),selector_revision=ARM_REVISION,
            candidate_count=5,candidate_representation='full',seed=stable_seed(SEED,POLICY,task,'guided','arm',attempt),actor_requests=5,actor_output_tokens=100)

    def test_failure_format_penalty_and_invalid_are_separate(self):
        rows=[self.record('win',reward=1),self.record('format',reward=-1),self.record('invalid',valid=False)]
        r=summarize(rows,['win','format','invalid'],require_complete=True)
        self.assertEqual((r['successes'],r['valid_attempts'],r['invalid_attempts']),(1,2,1))
        self.assertEqual(r['rescued_tasks'],['win'])
        self.assertEqual(r['overall'],1/3)
        self.assertEqual(r['valid_only'],.5)

    def test_missing_alternatives_cannot_be_valid_completed_rollout(self):
        r=self.record();trace=next((Path(r['artifact'])/'selections').glob('*.jsonl'));trace.write_text('')
        with self.assertRaisesRegex(ValueError,'selection traces'):verify_record(r,'a',0)

    def test_different_model_seed_or_judge_cannot_resume(self):
        r=self.record()
        for changed in [dict(selector_revision='other'),dict(seed=1),dict(judge_id='o4-mini'),dict(candidate_representation='actions_only')]:
            with self.subTest(changed=changed),self.assertRaises(ValueError):verify_record(dict(r,**changed),'a',0)

    def test_duplicates_and_missing_attempts_do_not_shrink_denominator(self):
        r=self.record()
        with self.assertRaises(ValueError):summarize([r,r],['a'])
        with self.assertRaises(ValueError):summarize([r],['a','b'],require_complete=True)
        self.assertEqual(summarize([r],['a','b'])['expected_attempts'],2)

    def test_cohort_keeps_only_verified_all_failure_in_frozen_order(self):
        summary=dict(complete=True,dispositions={'a':'all_failure','b':'mixed','c':'unresolved_invalid','d':'all_failure'})
        self.assertEqual(eligible_ids(summary,['d','c','b','a']),['d','a'])
        with self.assertRaises(ValueError):eligible_ids(dict(summary,complete=False),['a','b','c','d'])


class Budget(unittest.TestCase):
    def test_retries_preserve_total_and_require_actual_approval(self):
        a=dict(approved=True,resources=RESOURCES,judge_max_usd=20.,judge_max_calls=3000,attempts_per_task=1,
               attempts=[dict(job_id='42',shard=0,time_limit_seconds=3600)])
        verify_approval(a,'42',0,7200)
        for altered,prior in [(a,7201),(dict(a,approved=False),0),(dict(a,attempts_per_task=5),0),(dict(a,judge_max_usd=21.),0)]:
            with self.assertRaises(ValueError):verify_approval(altered,'42',0,prior)


if __name__=='__main__':unittest.main()
