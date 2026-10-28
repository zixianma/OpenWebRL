"""Offline storage race regressions; no scheduler/browser/provider activity."""
import errno
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts import launch_webprm_table5_campaign as m

class StorageScanTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/tmp',prefix='table5-storage-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
    def ok(self,value):return subprocess.CompletedProcess(['du'],0,f'{value}\t{self.root}\n','')
    def gone(self,path=None):return subprocess.CompletedProcess(['du'],1,f'1\t{self.root}\n',f"du: cannot access '{path or self.root/'vanished'}': No such file or directory\n")
    def test_exact_observed_postgres_disappearance_retries_but_not_partial_stdout(self):
        path=self.root/'build-temp/rootfs/usr/local/pgsql/data/base/13520/548'
        with patch.object(m.subprocess,'run',side_effect=[self.gone(path),self.ok(123)]) as run,patch.object(m.time,'sleep'):
            self.assertEqual(m.apparent(self.root),123)
        self.assertEqual(run.call_count,2)
        for call in run.call_args_list:
            self.assertEqual(call.kwargs['env']['LC_ALL'],'C');self.assertEqual(call.kwargs['timeout'],15)
            self.assertTrue(call.kwargs['capture_output'])
    def test_retry_bound_is_four_scans_and_point_six_seconds_delay(self):
        with patch.object(m.subprocess,'run',return_value=self.gone()) as run,patch.object(m.time,'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError,'Bounded'):m.apparent(self.root)
        self.assertEqual(run.call_count,4);self.assertAlmostEqual(sum(c.args[0] for c in sleep.call_args_list),.6)
    def test_disappeared_directory_read_enoent_requires_complete_rescan(self):
        response=self.gone(self.root/'removed-directory')
        response.stderr=response.stderr.replace('cannot access','cannot read directory')
        with patch.object(m.subprocess,'run',side_effect=[response,self.ok(456)]) as run,patch.object(m.time,'sleep'):
            self.assertEqual(m.apparent(self.root),456)
        self.assertEqual(run.call_count,2)
    def test_permission_io_unrelated_missing_path_and_other_failures_never_retry(self):
        for stderr,code in [(f"du: cannot read directory '{self.root}': Permission denied\n",1),
            (f"du: cannot access '{self.root/'x'}': Input/output error\n",1),
            ("du: cannot access '/tmp/unrelated': No such file or directory\n",1),('',2),('',-9)]:
            with self.subTest(stderr=stderr,code=code),patch.object(m.subprocess,'run',return_value=subprocess.CompletedProcess(['du'],code,'0',stderr)) as run:
                with self.assertRaises(subprocess.CalledProcessError):m.apparent(self.root)
                self.assertEqual(run.call_count,1)
    def test_recreated_path_requires_a_fresh_complete_rescan(self):
        path=self.root/'vanished';path.write_bytes(b'old');real_run=subprocess.run;calls=[]
        def replace_then_rescan(*args,**kwargs):
            calls.append(args)
            if len(calls)==1:
                path.unlink();path.write_bytes(b'recreated payload')
                return self.gone(path)
            return real_run(*args,**kwargs)
        expected=int(subprocess.check_output(['du','--apparent-size','-s','-B1',str(self.root)],text=True).split()[0])+len(b'recreated payload')-3
        with patch.object(m.subprocess,'run',side_effect=replace_then_rescan),patch.object(m.time,'sleep'):
            self.assertEqual(m.apparent(self.root),expected)
        self.assertEqual(len(calls),2)
    def test_missing_root_and_non_directory_stop_before_scan(self):
        with patch.object(m.subprocess,'run') as run:
            with self.assertRaises(FileNotFoundError):m.apparent(self.root/'missing')
            path=self.root/'file';path.write_bytes(b'x')
            with self.assertRaises(ValueError):m.apparent(path)
            run.assert_not_called()
    def test_root_replacement_is_rejected(self):
        original=Path.lstat;scanned=[False]
        def changed(path):
            value=original(path)
            if path==self.root and scanned[0]:
                return SimpleNamespace(st_dev=value.st_dev,st_ino=value.st_ino+1,st_mode=value.st_mode)
            return value
        def scan(*args,**kwargs):scanned[0]=True;return self.ok(10)
        with patch.object(Path,'lstat',changed),patch.object(m.subprocess,'run',side_effect=scan):
            with self.assertRaisesRegex(RuntimeError,'identity changed'):m.apparent(self.root)
    def test_clean_output_requires_exact_path_framing_and_no_warning(self):
        for stdout,stderr in [('12',''),('12\t/tmp/other\n',''),(f'12\t{self.root}\n','warning'),(f'-1\t{self.root}\n','')]:
            with self.subTest(stdout=stdout),patch.object(m.subprocess,'run',return_value=subprocess.CompletedProcess(['du'],0,stdout,stderr)):
                with self.assertRaisesRegex(RuntimeError,'Malformed'):m.apparent(self.root)
    def test_timeout_and_verification_io_errors_fail_closed(self):
        with patch.object(m.subprocess,'run',side_effect=subprocess.TimeoutExpired(['du'],15)) as run:
            with self.assertRaises(subprocess.TimeoutExpired):m.apparent(self.root)
            self.assertEqual(run.call_count,1)
        original=Path.lstat
        def denied(path):
            if path==self.root/'vanished':raise PermissionError(errno.EACCES,'permission denied')
            return original(path)
        with patch.object(Path,'lstat',denied),patch.object(m.subprocess,'run',return_value=self.gone()):
            with self.assertRaises(PermissionError):m.apparent(self.root)
    def test_final_equal_samples_return_conservative_maximum(self):
        with patch.object(m.subprocess,'run',side_effect=[self.ok(120),self.ok(100),self.ok(100)]) as run,patch.object(m.time,'sleep'):
            self.assertEqual(m.apparent(self.root,stable=True),120);self.assertEqual(run.call_count,3)
        with patch.object(m.subprocess,'run',side_effect=[self.ok(x) for x in (1,2,3,4)]),patch.object(m.time,'sleep'):
            with self.assertRaisesRegex(RuntimeError,'stable'):m.apparent(self.root,stable=True)
    def test_real_quiescent_tree_matches_du_with_sparse_file_and_symlink(self):
        (self.root/'sub').mkdir();(self.root/'sub/a').write_bytes(b'x'*100)
        with (self.root/'sparse').open('wb') as f:f.truncate(2**20)
        (self.root/'link').symlink_to('/not/followed')
        expected=int(subprocess.check_output(['du','--apparent-size','-s','-B1',str(self.root)],text=True).split()[0])
        self.assertEqual(m.apparent(self.root,stable=True),expected)
    def test_reserved_mutable_build_floor_is_charged_without_double_count(self):
        runner=object.__new__(m.CampaignRunner);runner.halt_reason=None;runner.servers=[];runner.broker=None
        runner.context=dict(deadline=time.time()+30,prior_retained_apparent_bytes=100)
        runner.proposal=dict(campaign_root=str(self.root),limits=dict(retained_bytes=1000,scratch_bytes=200))
        runner.scratch=self.root;runner.last_storage=0;runner.scratch_reservation_floor=101
        with patch.object(m,'apparent',side_effect=[1,1]):
            with self.assertRaisesRegex(RuntimeError,'scratch cap'):runner.check()
        runner.last_storage=0;runner.scratch_reservation_floor=100
        with patch.object(m,'apparent',side_effect=[100,1]):runner.check()
        self.assertEqual(runner.used,100)
    def test_final_accounting_requires_closed_owner_before_any_stable_scan(self):
        limits=dict(scratch_bytes=1000,retained_bytes=2000)
        with patch.object(m,'apparent') as scan:
            for closure in ({},{'remaining_owned_processes':{'123':'running'}}):
                with self.assertRaisesRegex(RuntimeError,'every owned process'):
                    m.storage_after_closure(closure,self.root,self.root,300,limits)
            scan.assert_not_called()
        with patch.object(m,'apparent',side_effect=[100,50]) as scan:
            value=m.storage_after_closure({'remaining_owned_processes':{}},self.root,self.root,300,limits)
            self.assertEqual(value['apparent_bytes'],100);self.assertEqual(value['cumulative_prior_retained_bytes'],300)
            self.assertTrue(all(x.kwargs=={'stable':True} for x in scan.call_args_list))

if __name__=='__main__':unittest.main()
