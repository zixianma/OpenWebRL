import copy
import hashlib
import json
import os
from pathlib import Path
import pwd
import shlex
import tempfile
import unittest

from scripts import webprm_transfer_recovery as m


class TransferRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='webprm-transfer-proof-', dir='/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scanner = 'dependencies/scripts/acquire_webprm_cms_original.py'
        source = self.root/self.scanner
        source.parent.mkdir(parents=True)
        source.write_text('old scanner')
        self.sources = {self.scanner: hashlib.sha256(b'new scanner').hexdigest()}
        metadata = {}
        for name in ('manifest', 'config', 'source-assessment', 'drive-preflight', 'oci-metadata-ledger'):
            relative='metadata/'+name+'.json'
            self.write(relative, {'fixture':name})
            metadata[relative]=m.sha(self.root/relative)
        self.sources.update(metadata)
        p = dict(site='shopping', identity_status='third_party_reconstruction_original_state_unverified',
            original_equivalence_verified=False, source_sha256={self.scanner:m.sha(source),**metadata},
            resources=dict(account='zixianma',partition='gpu-h200',qos='normal',cpus=2,memory_gib=8,
                gpus=0,api_usd=0,total_allocation_seconds_including_retries=7200,
                maximum_allocation_attempts=3,maximum_transfer_attempts=1),
            startup_recovery=dict(attempt_number=1,prior_allocation_seconds=0,requested_allocation_seconds=7200),
            prior_metadata_accounting=dict(prior_charged_bytes=100,prior_metadata_bytes=10,prior_responses=1),
            limits=dict(download_charged_bytes=70000000000,metadata_received_bytes=16*2**20,
                max_http_responses=4036,temporary_bytes=512*2**20,closure_reserve_seconds=120))
        self.p = p
        self.write('proposal.json',p)
        self.write('approval.json',dict(status='approved',exact_user_approval='fixture only',
            third_party_unverified_state_source_accepted=True,proposal_sha256=m.digest(p)))
        argv=['salloc','--comment=webprm-acquire:n','--chdir='+str(self.root)]
        self.write('launch.json',dict(nonce='n',proposal_sha256=m.digest(p),requested_allocation_seconds=7200,argv=argv))
        self.write('run-claim.json',dict(job_id='123',nonce='n'))
        self.write('network-claim.json',dict(job_id='123'))
        self.write('scheduler.json',dict(errors=[],jobs=[dict(job_id=123,user_id=os.getuid(),restart_cnt=0,
            comment='webprm-acquire:n',start_time=dict(set=True,infinite=False,number=1000))]))
        self.write('attempt-123/closure.json',dict(remaining_owned_processes={},child_exitcode=-9))
        self.ledger=dict(charged_bytes=2*65536+17+64*2**20,known_received_bytes=17,metadata_received_bytes=3,
            prior_charged_bytes=100,prior_metadata_bytes=10,prior_responses=1,
            cap_bytes=70000000000-100,cumulative_cap_bytes=70000000000,reservations=[
                dict(id=0,state='complete',received_bytes=17,reserved_bytes=64*2**20,header_reserved_bytes=65536),
                dict(id=1,state='unknown',received_bytes=0,reserved_bytes=64*2**20,header_reserved_bytes=65536)])
        self.write('attempt-123/transfer-ledger.json',self.ledger)
        self.terminal=dict(errors=[],jobs=[dict(job_id=123,user=pwd.getpwuid(os.getuid()).pw_name,
            restart_cnt=0,state=dict(current=['FAILED']),account='zixianma',partition='gpu-h200',qos='normal',
            working_directory=str(self.root),submit_line=shlex.join(argv),
            time=dict(elapsed=37,start=1000,end=1037,limit=dict(set=True,infinite=False,number=120)),
            tres=dict(allocated=[dict(type=k,name='',count=v) for k,v in [('cpu',2),('mem',8192),('node',1)]]))])
        self.repair=self.root/'repair.json'
        self.write('repair.json',dict(status='passed',tests=dict(exit_code=0,passed=3),
            source_sha256={'scripts/acquire_webprm_cms_original.py':self.sources[self.scanner]}))

    def write(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))

    def proof(self):
        return m.proof(self.root,self.terminal,self.sources,self.repair,prior_temporary_reserve_bytes=4*2**20)

    def test_exact_remaining_time_and_all_unknown_charges_are_carried(self):
        p=self.proof()
        self.assertEqual(p['status'],'requires_explicit_additional_transfer_exception')
        self.assertEqual(p['requested_allocation_seconds'],7140)
        self.assertEqual(p['prior_allocation_seconds'],37)
        self.assertEqual(p['prior_network_accounting'],dict(prior_charged_bytes=self.ledger['charged_bytes']+100,
            prior_metadata_bytes=13,prior_responses=3))
        self.assertEqual(p['unresolved_prior_reservations'],1)
        self.assertFalse(p['resume_existing_partials'])
        self.assertFalse((self.root/'successor-claim.json').exists())

    def test_unknown_reservation_cannot_be_refunded(self):
        self.ledger['charged_bytes']-=64*2**20
        self.write('attempt-123/transfer-ledger.json',self.ledger)
        with self.assertRaisesRegex(m.Closed,'arithmetic'):self.proof()

    def test_prior_metadata_cannot_be_reset(self):
        self.ledger['prior_metadata_bytes']=0
        self.write('attempt-123/transfer-ledger.json',self.ledger)
        with self.assertRaisesRegex(m.Closed,'metadata'):self.proof()

    def test_current_metadata_must_be_integral_nonnegative_received_bytes(self):
        for value in [-1, True, 18]:
            self.ledger['metadata_received_bytes']=value
            self.write('attempt-123/transfer-ledger.json',self.ledger)
            with self.subTest(value=value),self.assertRaisesRegex(m.Closed,'metadata'):self.proof()

    def test_terminal_identity_argv_and_resource_mismatch_fail(self):
        original=copy.deepcopy(self.terminal)
        for key,value in [('job_id',124),('submit_line','salloc --comment=foreign'),('restart_cnt',1)]:
            self.terminal=copy.deepcopy(original);self.terminal['jobs'][0][key]=value
            with self.subTest(key=key),self.assertRaises(m.Closed):self.proof()
        self.terminal=copy.deepcopy(original)
        self.terminal['jobs'][0]['tres']['allocated'].append(dict(type='gres',name='gpu',count=1))
        with self.assertRaisesRegex(m.Closed,'resource'):self.proof()

    def test_mutated_source_or_unclosed_owner_fails(self):
        self.write('attempt-123/closure.json',dict(remaining_owned_processes={'10':{}},child_exitcode=-9))
        with self.assertRaisesRegex(m.Closed,'closure'):self.proof()
        self.write('attempt-123/closure.json',dict(remaining_owned_processes={},child_exitcode=-9))
        (self.root/self.scanner).write_text('changed')
        with self.assertRaisesRegex(m.Closed,'source'):self.proof()

    def test_untested_or_differently_bound_fix_fails(self):
        self.sources[self.scanner]=m.sha(self.root/self.scanner)
        with self.assertRaisesRegex(m.Closed,'repair'):self.proof()

    def test_successor_metadata_cannot_be_changed_or_omitted(self):
        self.sources['metadata/manifest.json']='0'*64
        with self.assertRaisesRegex(m.Closed,'metadata'):self.proof()
        self.sources.pop('metadata/manifest.json')
        with self.assertRaisesRegex(m.Closed,'metadata'):self.proof()

    def test_completed_acquisition_cannot_be_repeated(self):
        self.write('completed.json',{})
        with self.assertRaisesRegex(m.Closed,'Completed'):self.proof()

    def test_clock_inconsistency_cannot_reclaim_time(self):
        self.terminal['jobs'][0]['time']['elapsed']=0
        with self.assertRaisesRegex(m.Closed,'clock'):self.proof()


if __name__=='__main__':
    unittest.main()
