import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_arm_jobs import record_events, result_text, write_dashboard


def report(complete=False):
    return dict(checked_utc='2026-09-21T01:00:00Z', jobs=[dict(job='1',label='eval',kind='eval',
        slurm=dict(state='RUNNING'),complete=complete,completed_tasks=300,valid=217,successes=101,
        alerts=[],durable_iteration=None,stage='collecting')],live_slurm_jobs=[])


class JobReporting(unittest.TestCase):
    def test_task_rates_only_after_completion(self):
        self.assertEqual(result_text(report()['jobs'][0]),'')
        self.assertIn('33.67% overall; 46.54% valid-only',result_text(report(True)['jobs'][0]))

    def test_completed_result_event_once_across_repeated_polls(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(record_events(report(),tmp),[])
            events=record_events(report(True),tmp)
            self.assertEqual([e['kind'] for e in events],['result_ready'])
            self.assertEqual(record_events(report(True),tmp),events)
            self.assertEqual(len(Path(tmp,'events.jsonl').read_text().splitlines()),1)

    def test_checkpoint_and_new_failure_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            first=report(); first['jobs'][0].update(kind='train',durable_iteration=10)
            record_events(first,tmp)
            second=copy.deepcopy(first);second['jobs'][0]['durable_iteration']=11
            self.assertEqual(record_events(second,tmp)[-1]['kind'],'checkpoint_saved')
            third=copy.deepcopy(second);third['jobs'][0]['alerts']=['Worker exited']
            self.assertEqual(record_events(third,tmp)[-1]['kind'],'attention_needed')

    def test_dashboard_escapes_names_and_shows_scheduler_only_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);d=report(True)
            d['live_slurm_jobs']=[dict(job='2',name='<script>alert(1)</script>',state='PENDING',elapsed='0',node='Priority')]
            write_dashboard(d,p,p/'output')
            page=(p/'output/live-status.html').read_text()
            self.assertNotIn('<script>',page)
            self.assertIn('&lt;script&gt;',page)
            self.assertIn('33.67% overall',page)
            self.assertIn('content="60"',page)
            self.assertEqual(json.loads((p/'output/live-status.json').read_text())['slurm_jobs'],d['live_slurm_jobs'])

    def test_rescue_compares_tasks_with_one_and_five_retries(self):
        row=dict(kind='rescue',complete=True,yield_summary=dict(complete=True,selected_tasks=8,
            by_mode=dict(arm=dict(successes=0),actor0=dict(successes=1)),paired=dict(actor5_any_success=3)))
        self.assertEqual(result_text(row),'Rescued tasks: ARM 0/8; one actor retry 1/8; five actor retries 3/8')

    def test_coverage_reports_label_counts_without_claiming_task_success(self):
        row=dict(kind='coverage',complete=True,coverage_summary=dict(
            same_pool_twenty_percent=dict(labels=13),four_turn_coverage=dict(labels=31,groups=4)))
        self.assertEqual(result_text(row),'Failure labels 13 -> 31 on 4 groups; zero optimizer updates')


if __name__=='__main__':unittest.main()
