import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('campaign_queue',Path(__file__).resolve().parents[1]/'openwebrl/webprm_campaign_queue.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class QueueTests(unittest.IsolatedAsyncioTestCase):
    def cells(self,n=10):return [dict(cell_id=f'cell-{i}',domain='reddit') for i in range(n)]

    async def test_eight_concurrent_workers_are_owned_until_closed(self):
        cells=self.cells(17);live=0;peak=0;released=[]
        async def worker(cell):
            nonlocal live,peak
            live+=1;peak=max(peak,live)
            await asyncio.sleep(.01)
            live-=1
            return dict(status='complete')
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            with m.ClaimStore(Path(temp)/'queue',cells) as store:
                result=await m.run_queue(cells,store,domain_ready=lambda d:True,can_start=lambda c,n:True,
                    run_one=worker,release=lambda c:released.append(c['cell_id']),poll_seconds=.005)
        self.assertEqual(peak,8);self.assertEqual(live,0)
        self.assertEqual(len(result['recorded']),17);self.assertEqual(len(released),17)
        self.assertEqual(result['remaining_owned_coroutines'],0)
        self.assertFalse(result['process_closure_verified_by_queue'])

    async def test_pending_domain_does_not_block_ready_domain_or_idle_gpu(self):
        cells=[dict(cell_id='cms',domain='shopping_admin'),dict(cell_id='reddit',domain='reddit')]
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            with m.ClaimStore(Path(temp)/'queue',cells) as store:
                result=await m.run_queue(cells,store,domain_ready=lambda d:d=='reddit',can_start=lambda c,n:True,
                    run_one=lambda c:dict(status='complete'),poll_seconds=.005)
        self.assertEqual(result['recorded'],['reddit']);self.assertEqual(result['pending'],['cms'])
        self.assertEqual(result['reason'],'domains_not_ready')

    async def test_budget_stop_drains_and_preserves_unattempted_cells(self):
        cells=self.cells();admitted=0;closed=[]
        def admission(cell,count):
            nonlocal admitted
            if admitted==3:return False
            admitted+=1;return True
        async def worker(cell):
            await asyncio.sleep(.01);closed.append(cell['cell_id']);return dict(status='complete')
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            with m.ClaimStore(Path(temp)/'queue',cells) as store:
                result=await m.run_queue(cells,store,domain_ready=lambda d:True,can_start=admission,run_one=worker)
        self.assertEqual(len(result['recorded']),3);self.assertEqual(len(result['pending']),7)
        self.assertEqual(set(closed),set(result['recorded']))
        self.assertEqual(result['reason'],'resource_admission_closed')

    async def test_transient_capacity_waits_for_closure_then_rechecks(self):
        cells=self.cells(3);closed=[];checks=[];released=[]
        def admission(cell,count):
            checks.append((cell['cell_id'],count,tuple(closed)))
            return cell['cell_id']!='cell-2' or 'cell-1' in released
        async def worker(cell):
            await asyncio.sleep(.005 if cell['cell_id']=='cell-0' else .035)
            closed.append(cell['cell_id']);return dict(status='complete')
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            with m.ClaimStore(Path(temp)/'queue',cells) as store:
                result=await m.run_queue(cells,store,domain_ready=lambda d:True,
                    can_start=admission,run_one=worker,release=lambda c:released.append(c['cell_id']),
                    max_workers=2,poll_seconds=.001)
        self.assertEqual(result['recorded'],[c['cell_id'] for c in cells])
        self.assertEqual(result['reason'],'completed')
        self.assertEqual([v[1] for v in checks if v[0]=='cell-2'],[1,0])

    async def test_permanent_capacity_denial_is_rechecked_once_after_closure(self):
        cells=self.cells(3);checks=[]
        def admission(cell,count):
            checks.append((cell['cell_id'],count));return cell['cell_id']!='cell-2'
        async def worker(cell):
            await asyncio.sleep(.005 if cell['cell_id']=='cell-0' else .035)
            return dict(status='complete')
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            with m.ClaimStore(Path(temp)/'queue',cells) as store:
                result=await m.run_queue(cells,store,domain_ready=lambda d:True,
                    can_start=admission,run_one=worker,max_workers=2,poll_seconds=.001)
        self.assertEqual(result['recorded'],['cell-0','cell-1'])
        self.assertEqual(result['pending'],['cell-2'])
        self.assertEqual(result['reason'],'resource_admission_closed')
        self.assertEqual([v[1] for v in checks if v[0]=='cell-2'],[1,0])

    async def test_deadline_while_capacity_waiting_drains_without_readmission(self):
        cells=self.cells(3);checks=[];stopped=False;closed=[]
        def admission(cell,count):
            nonlocal stopped
            checks.append(cell['cell_id'])
            if cell['cell_id']=='cell-2':stopped=True;return False
            return True
        async def worker(cell):
            await asyncio.sleep(.005 if cell['cell_id']=='cell-0' else .025)
            closed.append(cell['cell_id']);return dict(status='complete')
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            with m.ClaimStore(Path(temp)/'queue',cells) as store:
                result=await m.run_queue(cells,store,domain_ready=lambda d:True,
                    can_start=admission,run_one=worker,stop_admission=lambda:stopped,
                    max_workers=2,poll_seconds=.001)
        self.assertEqual(checks,['cell-0','cell-1','cell-2'])
        self.assertEqual(set(closed),{'cell-0','cell-1'})
        self.assertEqual(result['pending'],['cell-2']);self.assertEqual(result['reason'],'stop_admission')

    async def test_worker_failure_does_not_cancel_independent_cells(self):
        cells=self.cells(5)
        async def worker(cell):
            if cell['cell_id']=='cell-1':raise RuntimeError('fixture failure')
            return dict(status='complete')
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            with m.ClaimStore(Path(temp)/'queue',cells) as store:
                result=await m.run_queue(cells,store,domain_ready=lambda d:True,can_start=lambda c,n:True,run_one=worker)
                failed=json.loads(store._path('outcomes',cells[1]).read_text())
        self.assertEqual(len(result['recorded']),5);self.assertEqual(failed['result']['status'],'invalid')

    async def test_resume_never_retries_claimed_or_finished_primary(self):
        cells=self.cells(3);ran=[]
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            root=Path(temp)/'queue'
            with m.ClaimStore(root,cells) as store:
                store.claim(cells[0]);store.finish(cells[0],dict(status='complete'))
                store.claim(cells[1])
            with m.ClaimStore(root,cells) as store:
                def worker(cell):ran.append(cell['cell_id']);return dict(status='complete')
                result=await m.run_queue(cells,store,domain_ready=lambda d:True,can_start=lambda c,n:True,run_one=worker)
        self.assertEqual(ran,['cell-2']);self.assertEqual(result['attempted_without_outcome'],['cell-1'])

    async def test_host_gate_failure_awaits_existing_workers(self):
        cells=self.cells(3);admitted=0;closed=[]
        def admission(cell,n):
            nonlocal admitted
            admitted+=1
            if admitted==3:raise ValueError('fixture host failure')
            return True
        async def worker(cell):
            await asyncio.sleep(.01);closed.append(cell['cell_id']);return dict(status='complete')
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            with m.ClaimStore(Path(temp)/'queue',cells) as store:
                with self.assertRaises(ValueError):
                    await m.run_queue(cells,store,domain_ready=lambda d:True,can_start=admission,run_one=worker)
        self.assertEqual(set(closed),{'cell-0','cell-1'})

    async def test_cancellation_awaits_worker_cleanup_before_releasing(self):
        cells=self.cells(2);started=asyncio.Event();closed=[];released=[]
        async def worker(cell):
            started.set()
            try:await asyncio.sleep(10)
            finally:
                await asyncio.sleep(.01);closed.append(cell['cell_id'])
        def release(cell):
            self.assertIn(cell['cell_id'],closed)
            released.append(cell['cell_id'])
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            with m.ClaimStore(Path(temp)/'queue',cells) as store:
                task=asyncio.create_task(m.run_queue(cells,store,domain_ready=lambda d:True,
                    can_start=lambda c,n:True,run_one=worker,release=release))
                await started.wait();task.cancel()
                with self.assertRaises(asyncio.CancelledError):await task
                self.assertEqual([store.state(c) for c in cells],['recorded','recorded'])
        self.assertEqual(len(closed),2);self.assertEqual(len(released),2)

    def test_two_controllers_cannot_own_same_queue(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            root=Path(temp)/'queue';cells=self.cells()
            with m.ClaimStore(root,cells):
                with self.assertRaises(BlockingIOError):
                    with m.ClaimStore(root,cells):pass

    def test_changed_cohort_or_order_cannot_resume(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            root=Path(temp)/'queue';cells=self.cells()
            with m.ClaimStore(root,cells):pass
            with self.assertRaises(ValueError):
                with m.ClaimStore(root,list(reversed(cells))):pass

    def test_duplicate_cells_and_home_output_rejected(self):
        with self.assertRaises(ValueError):m.fixed_cells([dict(cell_id='same',domain='reddit')]*2)
        with self.assertRaises(ValueError):m.ClaimStore('/gpfs/home/zixianma/no-write',self.cells())

    async def test_invalid_readiness_object_cannot_admit_worker(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            cells=self.cells()
            with m.ClaimStore(Path(temp)/'queue',cells) as store:
                with self.assertRaises(ValueError):
                    await m.run_queue(cells,store,domain_ready=lambda d:{'ready':False},
                        can_start=lambda c,n:True,run_one=lambda c:dict(status='complete'))
                self.assertTrue(all(store.state(c)=='pending' for c in cells))

    async def test_stop_sentinel_checked_before_every_admission(self):
        admitted=0
        def admission(c,n):
            nonlocal admitted
            admitted+=1;return True
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            cells=self.cells()
            with m.ClaimStore(Path(temp)/'queue',cells) as store:
                result=await m.run_queue(cells,store,domain_ready=lambda d:True,can_start=admission,
                    stop_admission=lambda:admitted>=2,run_one=lambda c:dict(status='complete'))
        self.assertEqual(len(result['recorded']),2);self.assertEqual(result['reason'],'stop_admission')


if __name__=='__main__':unittest.main()
