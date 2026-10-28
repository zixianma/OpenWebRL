"""Startup retries consume one total budget; network attempts never retry."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('recovery',Path(__file__).resolve().parents[1]/'scripts/webprm_acquisition_recovery.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='/tmp',prefix='map-recovery-test-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.p=dict(resources=dict(account='zixianma',partition='gpu-h200',qos='normal',cpus=2,memory_gib=8,
            maximum_allocation_attempts=3,total_allocation_seconds_including_retries=7200),source_sha256={'acquisition.py':'old'})
        self.a=dict(status='approved',proposal_sha256=m.digest(self.p),exact_user_approval='Offline fixture')
        self.l=dict(nonce='nonce',proposal_sha256=m.digest(self.p),requested_allocation_seconds=7200)
        finite=lambda n:dict(set=True,infinite=False,number=n)
        self.s=dict(errors=[],jobs=[dict(job_id=123,job_state=['FAILED'],user_id=os.getuid(),restart_cnt=0,
            comment='webprm-acquire:nonce',account='zixianma',partition='gpu-h200',qos='normal',
            cpus=finite(2),node_count=finite(1),memory_per_node=finite(8192),start_time=finite(100),end_time=finite(109),
            time_limit=finite(120),tres_alloc_str='cpu=2,mem=8G,node=1')])
        self.r=dict(status='passed',test_exit_code=0,diagnosis='Corrected epoch clamp',test_argv=['python3','test.py'],
            predecessor_proposal_sha256=m.digest(self.p),replacement_source_sha256={'acquisition.py':'fixed'})

    def proof(self):return m.recovery_proof(self.root,self.p,self.a,self.l,self.s,self.r)

    def rebind(self):
        h=m.digest(self.p);self.a['proposal_sha256']=h;self.l['proposal_sha256']=h;self.r['predecessor_proposal_sha256']=h

    def test_elapsed_is_charged_and_next_request_rounds_down(self):
        r=self.proof();self.assertEqual(r['prior_allocation_seconds'],9);self.assertEqual(r['requested_allocation_seconds'],7140)

    def test_both_prior_attempts_share_original_total(self):
        self.p['startup_recovery']=dict(attempt_number=2,prior_allocation_seconds=9);self.rebind()
        self.l['requested_allocation_seconds']=7140;self.s['jobs'][0]['time_limit']['number']=119
        p=self.proof();self.assertEqual(p['attempt_number'],3);self.assertEqual(p['prior_allocation_seconds'],18)
        self.assertEqual(p['requested_allocation_seconds'],7140)

    def test_third_failed_start_cannot_obtain_fourth_attempt(self):
        self.p['startup_recovery']=dict(attempt_number=3,prior_allocation_seconds=18);self.rebind()
        with self.assertRaises(m.Closed):self.proof()

    def test_zero_byte_network_claim_disallows_successor(self):
        (self.root/'network-claim.json').write_text('{}')
        with self.assertRaises(m.Closed):self.proof()

    def test_legacy_transfer_ledger_disallows_successor(self):
        d=self.root/'attempt-1';d.mkdir();(d/'transfer-ledger.json').write_text('{}')
        with self.assertRaises(m.Closed):self.proof()

    def test_running_predecessor_and_wrong_nonce_rejected(self):
        for k,v in [('job_state',['RUNNING']),('comment','other'),('tres_alloc_str','cpu=2,mem=8G,node=1,gres/gpu=1')]:
            with self.subTest(k=k):
                old=self.s['jobs'][0][k];self.s['jobs'][0][k]=v
                with self.assertRaises(m.Closed):self.proof()
                self.s['jobs'][0][k]=old

    def test_retries_without_changed_and_tested_fix_rejected(self):
        for change in ({'test_exit_code':1},{'diagnosis':''},{'replacement_source_sha256':self.p['source_sha256']}):
            with self.subTest(change=change):
                old=copy.deepcopy(self.r);self.r.update(change)
                with self.assertRaises(m.Closed):self.proof()
                self.r=old

    def test_scheduler_grace_counts_against_remaining_budget(self):
        self.s['jobs'][0]['end_time']['number']=7303
        with self.assertRaises(m.Closed):self.proof()


if __name__=='__main__':unittest.main()
