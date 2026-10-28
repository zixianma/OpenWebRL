import unittest
from unittest.mock import patch
from scripts.resume_webprm_pilot import build_plan, query_terminal_accounting, terminal_seconds, parse_live_job, prepare


class ResumeBudgetTests(unittest.TestCase):
    def accounting(self):
        return [dict(JobIDRaw='1', State='FAILED', ElapsedRaw='20',
                     Start='2026-10-10T00:00:00', End='2026-10-10T00:00:20', AllocTRES='gres/gpu=1'),
                dict(JobIDRaw='2', State='COMPLETED', ElapsedRaw='6076',
                     Start='2026-10-10T00:01:00', End='2026-10-10T01:42:16', AllocTRES='gres/gpu=1')]

    def test_round_down_remaining_budget_without_reset(self):
        plan=build_plan(dict(prior_job_ids=['1','2'], approved_total_seconds=7200),self.accounting(),{})
        self.assertEqual(plan['consumed_seconds'],6096)
        self.assertEqual(plan['requested_seconds'],1080)
        self.assertEqual(plan['rounding_unrequested_seconds'],24)

    def test_nonterminal_or_missing_accounting_rejected(self):
        rows=self.accounting()
        with self.assertRaises(ValueError):terminal_seconds(['1','2'],rows[:1])
        rows[1]['State']='RUNNING'
        with self.assertRaises(ValueError):terminal_seconds(['1','2'],rows)

    def test_additional_allocation_ceiling_never_expands_total(self):
        snapshot=dict(prior_job_ids=['1','2'],approved_total_seconds=9000)
        plan=build_plan(snapshot,self.accounting(),{},max_seconds=1800)
        self.assertEqual(plan['requested_seconds'],1800)
        snapshot['approved_total_seconds']=7200
        self.assertEqual(build_plan(snapshot,self.accounting(),{},max_seconds=1800)['requested_seconds'],1080)

    def test_longer_explicit_approval_still_subtracts_every_attempt(self):
        snapshot=dict(prior_job_ids=['1','2'],approved_total_seconds=10800)
        plan=build_plan(snapshot,self.accounting(),{},max_seconds=10800)
        self.assertEqual(plan['consumed_seconds'],6096)
        self.assertEqual(plan['requested_seconds'],4680)
        with self.assertRaises(ValueError):
            build_plan(snapshot,self.accounting(),{},max_seconds=10801)

    def test_three_hour_live_allocation_uses_its_actual_remaining_time(self):
        raw='JobId=3 JobState=RUNNING TimeLimit=03:00:00 RunTime=00:00:10 AllocTRES=cpu=8,mem=120G,node=1,gres/gpu=1,gres/gpu:h200=1'
        result=parse_live_job(raw,'3',10800,1000.)
        self.assertEqual(result['new_deadline'],1000+10800-10-5-240)

    def test_expired_ids_need_no_filtered_squeue(self):
        raw='1|FAILED|20|start|end|gres/gpu=1'
        with patch('scripts.resume_webprm_pilot.scheduler',side_effect=['99|RUNNING',raw]) as run:
            rows,_=query_terminal_accounting(['1'])
        self.assertEqual(rows[0]['ElapsedRaw'],'20')
        self.assertNotIn('--jobs',run.call_args_list[0].args[0])

    def test_live_prior_attempt_blocks_recovery(self):
        with patch('scripts.resume_webprm_pilot.scheduler',return_value='1|COMPLETING'):
            with self.assertRaises(ValueError):query_terminal_accounting(['1'])

    def test_cannot_exclude_another_allocation_from_prior_cost(self):
        with patch.dict('os.environ',{'SLURM_JOB_ID':'3'}):
            with self.assertRaises(ValueError):prepare('/unused','/unused',current_job='2')

    def test_live_allocation_cap_and_cleanup_deadline(self):
        from datetime import datetime,timezone
        start=datetime(2026,10,10,tzinfo=timezone.utc).timestamp()
        raw='JobId=3 JobState=RUNNING TimeLimit=00:18:00 RunTime=00:00:10 StartTime=2026-10-10T00:00:00 EndTime=2026-10-10T00:18:00 AllocTRES=cpu=8,mem=120G,node=1,gres/gpu=1,gres/gpu:h200=1'
        result=parse_live_job(raw,'3',1080,start+10)
        self.assertEqual(result['new_deadline'],start+835)
        with self.assertRaises(ValueError):parse_live_job(raw,'3',7200,start+10)
        with self.assertRaises(ValueError):parse_live_job(raw.replace('cpu=8','cpu=16'),'3',1080,start+10)
        # Calendar values cannot grant extra time or reject valid durations.
        changed=raw.replace('EndTime=2026-10-10T00:18:00','EndTime=2026-10-10T07:18:00')
        self.assertEqual(parse_live_job(changed,'3',1080,start+10)['new_deadline'],start+835)
        with self.assertRaises(ValueError):parse_live_job(raw.replace('RunTime=00:00:10','RunTime=00:18:00'),'3',1080,start+10)


if __name__=='__main__':unittest.main()
