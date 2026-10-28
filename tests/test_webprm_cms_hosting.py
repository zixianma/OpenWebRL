import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from openwebrl import webprm_cms_hosting as h


def orphan_fixture(path):
    """Only tiny test descendants; no service, shell or external executable."""
    os.setsid()
    first=os.fork()
    if first==0:
        if os.fork()==0:
            os.setsid();Path(path).write_text(str(os.getpid()))
            time.sleep(30);os._exit(0)
        os._exit(0)
    os.waitpid(first,0)
    until=time.time()+2
    while not Path(path).exists() and time.time()<until:time.sleep(.01)


def snapshot(*, state='RUNNING', start=100, end=7300, cpus=4, memory=16384, minutes=120):
    finite=lambda v:dict(set=True,infinite=False,number=v)
    row=dict(job_id=42,user_id=os.getuid(),restart_cnt=0,account='zixianma',partition='gpu-h200',qos='normal',
             cpus=finite(cpus),node_count=finite(1),memory_per_node=finite(memory),time_limit=finite(minutes),
             start_time=finite(start),end_time=finite(end),job_state=[state],state_reason='JobHeldUser',
             tres_alloc_str='cpu=4,node=1,mem=16G',tres_req_str='cpu=4,node=1,mem=16G')
    return json.dumps(dict(errors=[],jobs=[row]))


class CMSHostingTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/tmp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_approval_gate_precedes_every_effect(self):
        with patch.object(h,'reference') as reference:
            with self.assertRaisesRegex(h.Closed,'approval'):
                h.validate({}, {})
            reference.assert_not_called()

    def test_scheduler_resources_clock_and_held_gate(self):
        got=h.scheduler(snapshot(end=7303),'42',requested_seconds=7200,now=110)
        self.assertEqual(got['deadline'],7300)
        self.assertEqual(got['work_deadline'],7180)
        for raw in (snapshot(cpus=8),snapshot(memory=8192),snapshot(state='PENDING')):
            with self.assertRaises(h.Closed):h.scheduler(raw,'42',requested_seconds=7200,now=110)
        self.assertEqual(h.scheduler(snapshot(state='PENDING'),'42',requested_seconds=7200,running=False)['state'],'held')
        with self.assertRaises(h.Closed):h.scheduler(snapshot(),'42',requested_seconds=7200,running=False)

    def test_history_accepts_full_registry_and_rejects_slot_beyond_cap(self):
        (self.root/'attempt-43').mkdir()
        with self.assertRaisesRegex(h.Closed,'Unregistered'):h.history(self.root)
        (self.root/'allocations').mkdir()
        cap=h.RESOURCES['max_allocations']
        for i in range(1,cap+1):h.put(self.root/'allocations'/f'{i:04d}.json',dict(sequence=i,job_id=str(42+i)))
        # The last registered job reads the full registry at startup (357113 failed here).
        self.assertEqual(len(h.history(self.root)),cap)
        h.put(self.root/'allocations'/f'{cap+1:04d}.json',dict(sequence=cap+1,job_id=str(43+cap)))
        with self.assertRaisesRegex(h.Closed,'bounded'):h.history(self.root)

    def test_startup_failure_without_attempt_folder_charges_time_only(self):
        old=self.root/'attempt-42';old.mkdir()
        h.put(old/'closure.json',dict(remaining_owned_processes={}))
        h.put(old/'storage-final.json',dict(scratch_bytes=12345))
        rows=[dict(job_id='42',requested_seconds=7200),dict(job_id='43',requested_seconds=6540)]
        got=h.prior_budget(self.root,rows,{'42':dict(state='FAILED',elapsed_seconds=601),'43':dict(state='FAILED',elapsed_seconds=4)})
        self.assertEqual(got,dict(prior_seconds=605,remaining_seconds=6540,prior_scratch_bytes=12345))
        # A present but incomplete folder is never treated as zero scratch.
        (self.root/'attempt-43').mkdir()
        with self.assertRaises((h.Closed,FileNotFoundError,OSError,ValueError)):
            h.prior_budget(self.root,rows,{'42':dict(state='FAILED',elapsed_seconds=601),'43':dict(state='FAILED',elapsed_seconds=4)})
        # Non-terminal jobs are still rejected.
        (self.root/'attempt-43').rmdir()
        with self.assertRaisesRegex(h.Closed,'terminal'):
            h.prior_budget(self.root,rows,{'42':dict(state='FAILED',elapsed_seconds=601),'43':dict(state='RUNNING',elapsed_seconds=4)})

    def test_prior_seconds_and_retained_scratch_are_not_reset(self):
        old=self.root/'attempt-42';old.mkdir()
        h.put(old/'closure.json',dict(remaining_owned_processes={}))
        h.put(old/'storage-final.json',dict(scratch_bytes=12345))
        rows=[dict(job_id='42',requested_seconds=7200)]
        got=h.prior_budget(self.root,rows,{'42':dict(state='FAILED',elapsed_seconds=601)})
        self.assertEqual(got,dict(prior_seconds=601,remaining_seconds=6540,prior_scratch_bytes=12345))
        for terms in ({},{'42':dict(state='RUNNING',elapsed_seconds=601)}):
            with self.assertRaises(h.Closed):h.prior_budget(self.root,rows,terms)

    def test_prior_scratch_is_charged_only_while_live_on_this_node(self):
        live=self.root/'live-scratch';live.mkdir()
        rows=[]
        for job,where in (('42',str(live)),('43',str(self.root/'gone-scratch')),('44',None)):
            old=self.root/('attempt-'+job);old.mkdir()
            h.put(old/'closure.json',dict(remaining_owned_processes={}))
            h.put(old/'storage-final.json',dict(scratch_bytes=1000*int(job)))
            if where:h.put(old/'context.json',dict(scratch=where))
            rows.append(dict(job_id=job,requested_seconds=7200))
        terms={r['job_id']:dict(state='FAILED',elapsed_seconds=10) for r in rows}
        got=h.prior_budget(self.root,rows,terms)
        # 42 is live here, 43 is on another node/gone, 44 has no recorded path (conservative).
        self.assertEqual(got['prior_scratch_bytes'],42000+44000)
        self.assertEqual(got['prior_seconds'],30)

    def test_apparent_never_traverses_image_absolute_link(self):
        (self.root/'file').write_bytes(b'1234')
        os.symlink('/gpfs/home/zixianma',self.root/'absolute')
        self.assertEqual(h.apparent(self.root),4+len('/gpfs/home/zixianma'))

    def test_safe_env_keeps_home_and_codex_home(self):
        before=dict(os.environ)
        env=h.safe_env(self.root)
        self.assertEqual(env.get('HOME'),before.get('HOME'))
        self.assertEqual(env.get('CODEX_HOME'),before.get('CODEX_HOME'))
        for name in ('XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','XDG_STATE_HOME','TMPDIR'):
            self.assertTrue(Path(env[name]).is_relative_to(self.root))

    def test_owned_subreaper_closes_tiny_detached_descendant(self):
        attempt=self.root/'attempt';attempt.mkdir()
        scratch=self.root/'scratch';scratch.mkdir()
        pidfile=scratch/'fixture-pid'
        closure=h.own(orphan_fixture,(str(pidfile),),attempt=attempt,scratch=scratch,
                      campaign_root=attempt,deadline=time.time()+5)
        self.assertEqual(closure['child_exitcode'],0)
        self.assertEqual(closure['remaining_owned_processes'],{})
        pid=int(pidfile.read_text())
        self.assertIn(pid,closure['owned_pids'])
        with self.assertRaises((FileNotFoundError,ProcessLookupError)):h._process(pid)
        self.assertEqual(h.read(attempt/'closure.json')['remaining_owned_processes'],{})
        self.assertGreater(h.read(attempt/'storage-final.json')['scratch_bytes'],0)
