import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import tempfile
import time
import threading
import unittest
from unittest.mock import patch

from scripts.run_webprm_pilot import collect_task, collect_cells, collect_cohort, cohort_status, quality_gate_failure
from openwebrl.webprm_baselines import ARMS, paired_summary


class CollectionCompletenessTests(unittest.TestCase):
    def row(self, task, arm, **changes):
        return dict(task_id=task['task_id'],arm=arm,attempted=True,valid=True,success=0,**changes)

    def test_cross_task_slots_fill_but_third_task_waits_for_quality_gate(self):
        tasks=[dict(task_id=task) for task in ('a','b','c')]
        release_gate=threading.Event()
        second_started=threading.Event()
        third_started=threading.Event()
        lock=threading.Lock()
        active=maximum=0
        results=[]
        failures=[]
        def worker(task,arm,config):
            nonlocal active,maximum
            with lock:
                active+=1
                maximum=max(maximum,active)
            try:
                if task['task_id']=='b':
                    second_started.set()
                elif task['task_id']=='c':
                    third_started.set()
                if task['task_id'] in ('a','b') and arm==ARMS[0]:
                    if not release_gate.wait(5):
                        raise AssertionError('Gate test did not release worker')
                return self.row(task,arm)
            finally:
                with lock:
                    active-=1
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(4) as pool:
            config=self.config(directory)
            def collect():
                try:
                    reason=collect_cohort(pool,tasks,config,4,results.append,results)
                    self.assertIsNone(reason)
                except BaseException as exc:
                    failures.append(exc)
            with patch('scripts.run_webprm_pilot.run_episode',side_effect=worker):
                controller=threading.Thread(target=collect)
                controller.start()
                try:
                    self.assertTrue(second_started.wait(2),'Task b never used a free slot while a was still running')
                    self.assertFalse(third_started.is_set(),'Task c bypassed the unresolved initial gate')
                finally:
                    release_gate.set()
                    controller.join(5)
            self.assertFalse(controller.is_alive())
            if failures:
                raise failures[0]
            self.assertTrue(third_started.is_set())
            self.assertLessEqual(maximum,4)
            self.assertEqual(len(results),18)

    def test_failed_quality_gate_never_admits_third_task(self):
        tasks=[dict(task_id=task) for task in ('a','b','c')]
        results=[]
        def worker(task,arm,config):
            row=self.row(task,arm)
            row['valid']=arm!='shepherd8b'
            return row
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(4) as pool:
            with patch('scripts.run_webprm_pilot.run_episode',side_effect=worker):
                reason=collect_cohort(pool,tasks,self.config(directory),4,results.append,results)
        self.assertEqual(reason,'quality_gate_no_valid_episode:shepherd8b')
        self.assertEqual(len(results),12)
        self.assertEqual({row['task_id'] for row in results},{'a','b'})

    def test_operator_stop_drains_active_workers_and_returns_without_new_admission(self):
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(4) as pool:
            stop_path=Path(directory)/'stop-admission.json'
            config=self.config(directory,stop_admission_path=str(stop_path))
            barrier=threading.Barrier(4)
            started=[]
            results=[]
            def worker(task,arm,config):
                started.append(arm)
                barrier.wait(timeout=5)
                if arm==ARMS[0]:
                    stop_path.write_text('{}')
                else:
                    limit=time.monotonic()+5
                    while not stop_path.exists():
                        if time.monotonic()>limit:
                            raise AssertionError('Operator stop was not written')
                        time.sleep(.005)
                return self.row(task,arm)
            with patch('scripts.run_webprm_pilot.run_episode',side_effect=worker):
                reason=collect_task(pool,dict(task_id='a'),ARMS,config,4,results.append)
            self.assertEqual(reason,'operator_requested_stop_after_draining')
            self.assertEqual(set(started),set(ARMS[:4]))
            self.assertEqual(len(results),4)
            self.assertEqual(cohort_status(results,['a'])['status'],'partial')

    def test_resume_preserves_original_shuffle_and_skips_saved_cells(self):
        tasks=[dict(task_id=task) for task in ('a','b','c','d')]
        class ImmediatePool:
            def __init__(self):
                self.submitted=[]
            def submit(pool,worker,task,arm,config):
                pool.submitted.append((task['task_id'],arm))
                future=Future()
                future.set_result(self.row(task,arm))
                return future
        with tempfile.TemporaryDirectory() as directory:
            config=self.config(directory)
            first=ImmediatePool()
            complete=[]
            self.assertIsNone(collect_cohort(first,tasks,config,4,complete.append,complete))
            saved=[row for row in complete if row['task_id'] in ('a','b') or row['arm']==ARMS[0]]
            saved_keys={(row['task_id'],row['arm']) for row in saved}
            resumed=ImmediatePool()
            self.assertIsNone(collect_cohort(resumed,tasks,config,4,saved.append,saved))
            self.assertEqual(resumed.submitted,[cell for cell in first.submitted if cell not in saved_keys])
            self.assertEqual(len(saved),24)

    def test_unattempted_saved_gate_cell_cannot_admit_later_tasks(self):
        tasks=[dict(task_id=task) for task in ('a','b','c')]
        results=[self.row(task,arm) for task in tasks[:2] for arm in ARMS]
        results[0].update(attempted=False,valid=False)
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(4) as pool:
            with patch('scripts.run_webprm_pilot.run_episode') as worker:
                reason=collect_cohort(pool,tasks,self.config(directory),4,results.append,results)
                worker.assert_not_called()
        self.assertEqual(reason,'quality_gate_incomplete')

    def test_concurrency_above_four_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(1) as pool:
            with self.assertRaisesRegex(ValueError,'four concurrent'):
                collect_cells(pool,[],self.config(directory),5,lambda row:None)

    def test_quality_gate_waits_for_complete_task_cohorts_and_requires_validity(self):
        rows=[dict(task_id=task,arm=arm,attempted=True,valid=arm!='shepherd8b',success=0)
              for task in ('a','b') for arm in ARMS]
        self.assertIsNone(quality_gate_failure(rows[:-1],['a','b']))
        self.assertEqual(quality_gate_failure(rows,['a','b']), 'quality_gate_no_valid_episode:shepherd8b')
        next(row for row in rows if row['arm']=='shepherd8b')['valid']=True
        self.assertIsNone(quality_gate_failure(rows,['a','b']))

    def config(self, directory, **kwargs):
        path=Path(directory)/'budget.json'
        path.write_text(json.dumps(dict(cap=30,charged=0,calls=[])))
        return dict(budget_path=str(path),deadline=time.time()+60,**kwargs)

    def test_deadline_does_not_fabricate_missing_cells(self):
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(2) as pool:
            config=self.config(directory)
            config['deadline']=0
            results=[]
            with patch('scripts.run_webprm_pilot.run_episode') as worker:
                reason=collect_task(pool,{'task_id':'a'},ARMS,config,2,results.append)
                worker.assert_not_called()
            self.assertEqual(reason,'pilot_deadline')
            status=cohort_status(results,['a'])
            self.assertEqual(status['status'],'partial')
            self.assertEqual(len(status['missing_episodes']),6)

    def test_authorization_error_stops_further_calls(self):
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(1) as pool:
            results=[]
            error=dict(task_id='a',arm=ARMS[0],attempted=True,valid=False,success=0,
                       error_type='HTTPError',error='HTTP Error 401: Unauthorized')
            with patch('scripts.run_webprm_pilot.run_episode',return_value=error) as worker:
                reason=collect_task(pool,{'task_id':'a'},ARMS,self.config(directory),1,results.append)
            self.assertEqual(worker.call_count,1)
            self.assertEqual(reason,'provider_authorization_failed')
            self.assertEqual(cohort_status(results,['a'])['attempted_episodes'],1)

    def test_validity_and_attempt_completion_are_distinct(self):
        rows=[dict(task_id='a',arm=arm,attempted=True,valid=False,success=0) for arm in ARMS]
        self.assertEqual(cohort_status(rows,['a'])['status'],'completed')
        self.assertEqual(paired_summary(rows,['a'],100)['methods'][ARMS[0]]['valid'],0)
        rows[0]['attempted']=False
        status=cohort_status(rows,['a'])
        self.assertEqual(status['status'],'partial')
        self.assertEqual(status['attempted_episodes'],5)
        with self.assertRaisesRegex(ValueError,'Unattempted'):
            paired_summary(rows,['a'],100)

    def test_budget_refusal_stops_without_submitting_remaining_arms(self):
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(1) as pool:
            results=[]
            error=dict(task_id='a',arm=ARMS[0],attempted=True,valid=False,success=0,
                       error_type='RuntimeError',error='Shared API budget exhausted')
            with patch('scripts.run_webprm_pilot.run_episode',return_value=error) as worker:
                reason=collect_task(pool,{'task_id':'a'},ARMS,self.config(directory),1,results.append)
            self.assertEqual(worker.call_count,1)
            self.assertEqual(reason,'shared_api_budget_exhausted')


if __name__=='__main__':
    unittest.main()
