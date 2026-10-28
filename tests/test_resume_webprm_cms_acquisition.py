from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import acquire_webprm_cms_original as cms
from scripts import resume_webprm_cms_acquisition as recovery


def snapshot(job=123, elapsed=5, minutes=59):
    finite=lambda n:dict(set=True,infinite=False,number=n)
    return dict(errors=[],jobs=[dict(job_id=job,user_id=os.getuid(),restart_cnt=0,job_state=['FAILED'],
        start_time=finite(1000),end_time=finite(1000+elapsed),time_limit=finite(minutes),
        cpus=finite(2),node_count=finite(1),memory_per_node=finite(8192),account='zixianma',
        partition='gpu-h200',qos='normal',tres_alloc_str='cpu=2,mem=8G,node=1')])


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='cms-recovery-test-',dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.sources=recovery.source_hashes()
        self.proposal=dict(predecessors=[dict(elapsed_seconds=4),dict(elapsed_seconds=5)],source_sha256=self.sources)
        cms.put(self.root/'proposal.json',self.proposal)

    def add_attempt(self,number,elapsed=None,requested=3540):
        p=self.root/f'startup-{number:02d}';p.mkdir()
        cms.put(p/'launch.json',dict(requested_seconds=requested,source_sha256=self.sources))
        if elapsed is not None:
            cms.put(p/'run-claim.json',dict(job_id=str(122+number)))
            t=recovery.terminal_accounting(snapshot(122+number,elapsed,requested//60),122+number,requested)
            cms.put(p/'terminal.json',t)
        return p

    def test_exact_terminal_accounting_and_no_live_reclaim(self):
        self.assertEqual(recovery.terminal_accounting(snapshot(),123,3540)['elapsed_seconds'],5)
        for key,value in [('job_state',['RUNNING']),('restart_cnt',1),('cpus',dict(set=True,infinite=False,number=4)),
                          ('tres_alloc_str','cpu=2,mem=8G,node=1,gres/gpu=1')]:
            s=snapshot();s['jobs'][0][key]=value
            with self.subTest(key=key),self.assertRaises(cms.GateClosed):
                recovery.terminal_accounting(s,123,3540)

    def test_first_request_deducts_both_failed_predecessors(self):
        with patch.object(recovery,'approved',return_value=self.proposal):
            plan=recovery.next_request(self.root)
        self.assertEqual(plan['budget']['charged_seconds'],9)
        self.assertEqual(plan['budget']['remaining_seconds'],3591)
        self.assertEqual(plan['requested_seconds'],3540)

    def test_pending_approval_never_reaches_scheduler(self):
        cms.put(self.root/'approval.json',dict(status='pending'))
        with patch.object(recovery,'prior_evidence',return_value=[]),patch.object(recovery.subprocess,'run') as run:
            with self.assertRaises(cms.GateClosed):recovery.launch(self.root)
            run.assert_not_called()
        self.assertEqual(list(self.root.glob('startup-*')),[])

    def test_unknown_allocation_keeps_full_reservation_and_blocks_retry(self):
        self.add_attempt(1)
        self.assertEqual(recovery.budget(self.root)['charged_seconds'],3549)
        with patch.object(recovery,'approved',return_value=self.proposal),self.assertRaisesRegex(cms.GateClosed,'unresolved'):
            recovery.next_request(self.root)

    def test_every_failed_second_charged_without_reset(self):
        self.add_attempt(1,elapsed=65)
        b=recovery.budget(self.root)
        self.assertEqual(b['charged_seconds'],74)
        self.assertEqual(b['remaining_seconds'],3526)
        self.add_attempt(2,elapsed=20,requested=3480)
        self.assertEqual(recovery.budget(self.root)['charged_seconds'],94)

    def test_requested_overbudget_or_changed_terminal_receipt_rejected(self):
        p=self.add_attempt(1,elapsed=5)
        t=cms.read(p/'terminal.json');t['elapsed_seconds']=0;cms.put(p/'terminal.json',t,replace=True)
        with self.assertRaisesRegex(cms.GateClosed,'Terminal receipt'):recovery.budget(self.root)
        other=self.root/'other';other.mkdir();cms.put(other/'proposal.json',self.proposal)
        p=other/'startup-01';p.mkdir();cms.put(p/'launch.json',dict(requested_seconds=3600))
        with self.assertRaisesRegex(cms.GateClosed,'reservation'):recovery.budget(other)

    def test_one_transfer_claim_blocks_every_later_allocation(self):
        cms.put(self.root/'transfer-claim.json',dict(startup=1))
        with patch.object(recovery,'approved',return_value=self.proposal),self.assertRaisesRegex(cms.GateClosed,'sole transfer'):
            recovery.next_request(self.root)

    def test_three_additional_startup_cap(self):
        for n in (1,2,3):self.add_attempt(n,elapsed=5)
        with patch.object(recovery,'approved',return_value=self.proposal),self.assertRaisesRegex(cms.GateClosed,'attempt cap'):
            recovery.next_request(self.root)

    def test_manual_successor_requires_changed_fix_and_review(self):
        p=self.add_attempt(1,elapsed=65)
        cms.put(p/'failure.json',dict(stage='startup_before_transfer',status='failed',error_type='GateClosed'))
        log=self.root/'tests.log';log.write_text('Representative offline tests passed\n')
        changed=dict(self.sources);changed[str(cms.SCRIPT)]='fixed-source'
        review=dict(previous_startup=str(p),previous_failure_sha256=cms.sha(p/'failure.json'),source_sha256=changed,
            explanation='Concrete startup fix and validation.',offline_tests_passed=True,
            test_log_path=str(log),test_log_sha256=cms.sha(log))
        q=self.root/'review.json';cms.put(q,review)
        with patch.object(recovery,'approved',return_value=self.proposal),patch.object(recovery,'source_hashes',return_value=changed):
            self.assertEqual(recovery.next_request(self.root,q)['requested_seconds'],3480)
            with self.assertRaises(cms.GateClosed):recovery.next_request(self.root)
            review['source_sha256']=self.sources;cms.put(q,review,replace=True)
            with self.assertRaises(cms.GateClosed):recovery.next_request(self.root,q)

    def test_failed_transfer_never_qualifies_as_startup_repair(self):
        p=self.add_attempt(1,elapsed=5)
        cms.put(p/'failure.json',dict(stage='sole_transfer_attempt',status='failed',error_type='GateClosed'))
        with patch.object(recovery,'approved',return_value=self.proposal),self.assertRaises(cms.GateClosed):
            recovery.next_request(self.root)

    def test_reconcile_requires_owned_job_identity(self):
        p=self.add_attempt(1)
        cms.put(p/'run-claim.json',dict(job_id='123'))
        with patch.object(recovery.subprocess,'check_output',return_value=json.dumps(snapshot())):
            b=recovery.reconcile(self.root,1)
        self.assertEqual(b['charged_seconds'],14)
        self.assertTrue((p/'terminal.json').exists())

    def test_launch_owns_one_allocation_and_accounts_terminal_before_return(self):
        seen=[]
        def completed_owned_allocation(argv,**kwargs):
            seen.append((argv,kwargs))
            p=self.root/'startup-01'
            claim=cms.read(p/'launch.json')
            self.assertEqual(claim['requested_seconds'],3540)
            cms.put(p/'run-claim.json',dict(job_id='123',nonce=claim['nonce']))
            cms.put(p/'failure.json',dict(stage='startup_before_transfer',status='failed',error_type='GateClosed'))
            return SimpleNamespace(returncode=1)
        with patch.object(recovery,'approved',return_value=self.proposal),\
             patch.object(recovery.subprocess,'run',side_effect=completed_owned_allocation),\
             patch.object(recovery.subprocess,'check_output',return_value=json.dumps(snapshot())):
            self.assertEqual(recovery.launch(self.root),1)
        self.assertEqual(len(seen),1)
        self.assertIn('--time=00:59:00',seen[0][0])
        self.assertIn('srun',seen[0][0])
        self.assertNotIn('--gres',str(seen[0][0]))
        self.assertEqual(seen[0][1]['env'].get('HOME'),os.environ.get('HOME'))
        self.assertTrue((self.root/'startup-01/owner-exit.json').exists())
        self.assertEqual(recovery.budget(self.root)['charged_seconds'],14)

    def test_dynamic_recovery_profile_does_not_expand_other_resources(self):
        s=snapshot(elapsed=3543,minutes=58)
        s['jobs'][0]['job_state']=['RUNNING']
        s['meta']=dict(plugin=dict(data_parser='data_parser/v0.0.44'))
        resources=dict(cms.RESOURCES,total_allocation_seconds_including_retries=3480)
        c=cms.allocation_context(json.dumps(s),'123','n',dict(nonce='n'),1005,resources=resources)
        self.assertEqual(c['allocation_deadline'],4480)
        for change in (dict(cpus=3),dict(total_allocation_seconds_including_retries=3601),
                       dict(total_allocation_seconds_including_retries=3591)):
            with self.assertRaises(cms.GateClosed):
                cms.allocation_context(json.dumps(s),'123','n',dict(nonce='n'),1005,resources=dict(resources,**change))


if __name__=='__main__':unittest.main()
