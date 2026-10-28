"""Offline source acceptance, frozen profiles, finance and startup ownership."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import pwd
import shlex
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('thirdparty_sites',Path(__file__).resolve().parents[1]/'scripts/acquire_webprm_thirdparty_sites.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ThirdPartySiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='thirdparty-site-test-',dir=m.RUNTIME/'next-stage-preparation')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def package(self,site='shopping'):
        root=self.root/site;m.prepare(root,site);return root

    def approve(self,root):
        p=m.read(root/'proposal.json')
        (root/'approval.json').write_text(json.dumps(dict(status='approved',proposal_sha256=m.digest(p),
            exact_user_approval='Offline fixture only',third_party_unverified_state_source_accepted=True)))
        return p

    def test_pending_source_acceptance_blocks_before_scheduler(self):
        root=self.package()
        with patch.object(m.os,'execvpe',side_effect=AssertionError('No allocation')):
            with self.assertRaisesRegex(m.Closed,'approval'):m.launch(root)
        self.assertFalse((root/'launch.json').exists())

    def test_explicit_unverified_source_acceptance_required(self):
        root=self.package();p=self.approve(root)
        with patch.object(m,'SCRIPT',root/'acquisition.py'):self.assertEqual(m.approved(root),p)
        a=m.read(root/'approval.json');a['third_party_unverified_state_source_accepted']=False
        (root/'approval.json').write_text(json.dumps(a))
        with patch.object(m,'SCRIPT',root/'acquisition.py'),self.assertRaises(m.Closed):m.approved(root)

    def test_candidate_cannot_be_relabeled_as_original(self):
        root=self.package();p=self.approve(root);p['original_equivalence_verified']=True
        (root/'proposal.json').write_text(json.dumps(p));self.approve(root)
        with patch.object(m,'SCRIPT',root/'acquisition.py'),self.assertRaises(m.Closed):m.approved(root)

    def test_frozen_source_and_manifest_changes_fail(self):
        for name in ('dependencies/scripts/webprm_cms_oci_transport.py','metadata/manifest.json'):
            with self.subTest(name=name):
                root=self.root/str(len(list(self.root.iterdir())));m.prepare(root,'shopping');self.approve(root)
                with (root/name).open('a') as out:out.write('\n')
                with patch.object(m,'SCRIPT',root/'acquisition.py'),self.assertRaises(m.Closed):m.approved(root)

    def test_prior_metadata_charge_cannot_be_reset(self):
        root=self.package();p=self.approve(root);p['prior_metadata_accounting']['prior_charged_bytes']=0
        (root/'proposal.json').write_text(json.dumps(p));self.approve(root)
        with patch.object(m,'SCRIPT',root/'acquisition.py'),self.assertRaisesRegex(m.Closed,'accounting'):m.approved(root)

    def test_each_site_binds_actual_large_blob_ledger_and_reuses_owned_transport(self):
        program=r'''
import importlib.util,json,pathlib,sys,socket,time,gzip,hashlib,io
from unittest.mock import patch
root=pathlib.Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('worker',root/'acquisition.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=m.read(root/'proposal.json');site=p['site']
with patch.object(socket.socket,'connect',side_effect=AssertionError('No live HTTP')):
 lib,oci=m.libraries(root,site);items,manifest,config=m.inputs(root,site)
 assert len(items)<=p['profile']['layers'] and sum(x['bytes'] for x in items)==p['total_unique_compressed_bytes']
 assert p['resources']['total_allocation_seconds_including_retries']==7200
 assert oci.cms.ARCHIVE_BYTES==m.base.PROFILES[site]['archive_bytes']
 big=max(x['bytes'] for x in items);assert big>9640032256
 attempt=root/'fake';attempt.mkdir()
 ledger=lib.TransferLedger(attempt,cap=p['limits']['download_charged_bytes'],metadata_cap=16*2**20,**p['prior_metadata_accounting'])
 i=ledger.reserve(64*2**20,items[0]['url'],0,64*2**20-1);ledger.expand(i,big)
 assert ledger.data['charged_bytes']==big+65536
 assert ledger.data['prior_charged_bytes']==p['prior_metadata_accounting']['prior_charged_bytes']>0
 assert ledger.cap+p['prior_metadata_accounting']['prior_charged_bytes']==p['limits']['download_charged_bytes']
 ledger.settle(i,0,complete=True)
 payload=b'candidate raw bytes';body=gzip.compress(payload,mtime=0);h=hashlib.sha256(body).hexdigest()
 item=dict(kind='image_blob',filename='oci/blobs/sha256/'+h,bytes=len(body),sha256=h,raw_diff_id='sha256:'+hashlib.sha256(payload).hexdigest(),registry='https://registry-1.docker.io',repository=p['profile']['repository'],url='https://registry-1.docker.io/v2/'+p['profile']['repository']+'/blobs/sha256:'+h)
 class Tokens:
  def token(self,item):return 'synthetic'
 class Response:
  status=200;headers={'Content-Length':str(len(body))}
  def __init__(self):self.body=io.BytesIO(body)
  def read(self,n):return self.body.read(n)
  def close(self):pass
 def fake(req,timeout):assert req.get_header('Authorization')=='Bearer synthetic';return Response()
 result=oci.transfer_blob(attempt,item,ledger,time.time()+10,Tokens(),open_response=fake,chunk=3)
 raw=oci.verify_raw_layers(attempt,[item],time.time()+10,maximum=p['limits']['expanded_raw_layer_bytes'])
 assert result['sha256']==h and raw['all_raw_diff_ids_verified'] and raw['total_raw_bytes']==len(payload)
 print(json.dumps({'site':site,'large_expansion_bytes':big,'network_calls':0}))
'''
        for site in m.PINNED:
            with self.subTest(site=site):
                root=self.package(site)
                result=subprocess.run([sys.executable,'-I','-B','-c',program,str(root)],capture_output=True,text=True,timeout=20)
                self.assertEqual(result.returncode,0,result.stderr)

    def test_manual_startup_recovery_deducts_elapsed_and_forbids_network_retry(self):
        previous=self.package();p=m.read(previous/'proposal.json')
        with (previous/'acquisition.py').open('a') as out:out.write('\n# synthetic old startup bug\n')
        p['source_sha256']['acquisition.py']=m.sha(previous/'acquisition.py');(previous/'proposal.json').write_text(json.dumps(p));self.approve(previous)
        m.put(previous/'launch.json',dict(nonce='fixture',proposal_sha256=m.digest(p),requested_allocation_seconds=7200))
        sources=dict(p['source_sha256']);sources['acquisition.py']=m.sha(m.SCRIPT)
        repair=self.root/'repair.json';m.put(repair,dict(status='passed',test_exit_code=0,diagnosis='tested startup fix',test_argv=['offline'],
             predecessor_proposal_sha256=m.digest(p),replacement_source_sha256=sources))
        finite=lambda v:dict(set=True,infinite=False,number=v)
        snapshot=dict(errors=[],jobs=[dict(job_id=123,user_id=os.getuid(),job_state=['FAILED'],restart_cnt=0,
            comment='webprm-acquire:fixture',account='zixianma',partition='gpu-h200',qos='normal',cpus=finite(2),node_count=finite(1),memory_per_node=finite(8192),
            start_time=finite(100),end_time=finite(109),time_limit=finite(120),tres_alloc_str='cpu=2,mem=8G,node=1')])
        successor=self.root/'successor'
        with patch.object(m.subprocess,'check_output',return_value=json.dumps(snapshot)):
            new=m.prepare(successor,'shopping',predecessor=previous,prior_job='123',repair_receipt=repair)
        self.assertEqual(new['startup_recovery']['requested_allocation_seconds'],7140)
        with patch.object(m,'SCRIPT',successor/'acquisition.py'):self.assertEqual(m.approved(successor),new)
        m.put(previous/'network-claim.json',{})
        with patch.object(m,'SCRIPT',successor/'acquisition.py'),self.assertRaises(ValueError):m.approved(successor)

    def test_protected_home_rejected(self):
        with self.assertRaises(m.Closed):m.checked('/gpfs/home/zixianma/never-write')

    def transfer_fixture(self,site='shopping',elapsed=37):
        previous=self.package(site);p=m.read(previous/'proposal.json')
        scanner='dependencies/scripts/acquire_webprm_cms_original.py'
        with (previous/scanner).open('a') as stream:stream.write('\n# preserved synthetic old scanner\n')
        p['source_sha256'][scanner]=m.sha(previous/scanner)
        (previous/'proposal.json').write_text(json.dumps(p));self.approve(previous)
        argv=['salloc','--comment=webprm-acquire:fixture','--chdir='+str(previous)]
        m.put(previous/'launch.json',dict(nonce='fixture',proposal_sha256=m.digest(p),requested_allocation_seconds=7200,argv=argv))
        m.put(previous/'run-claim.json',dict(job_id='123',nonce='fixture'));m.put(previous/'network-claim.json',dict(job_id='123'))
        finite=lambda v:dict(set=True,infinite=False,number=v)
        m.put(previous/'scheduler.json',dict(errors=[],jobs=[dict(job_id=123,user_id=os.getuid(),restart_cnt=0,
            comment='webprm-acquire:fixture',start_time=finite(1000))]))
        attempt=previous/'attempt-123';attempt.mkdir()
        m.put(attempt/'closure.json',dict(remaining_owned_processes={},child_exitcode=-9))
        scratch='/tmp/webprm-original-'+site+'-thirdparty-123-no-existing-scratch'
        m.put(attempt/'context.json',dict(scratch=scratch))
        ledger=dict(charged_bytes=2*65536+3+64*2**20,known_received_bytes=3,metadata_received_bytes=3,
            **p['prior_metadata_accounting'],cap_bytes=p['limits']['download_charged_bytes']-p['prior_metadata_accounting']['prior_charged_bytes'],
            cumulative_cap_bytes=p['limits']['download_charged_bytes'],reservations=[
                dict(id=0,state='complete',received_bytes=3,reserved_bytes=100,header_reserved_bytes=65536),
                dict(id=1,state='unknown',received_bytes=0,reserved_bytes=64*2**20,header_reserved_bytes=65536)])
        m.put(attempt/'transfer-ledger.json',ledger)
        terminal=dict(errors=[],jobs=[dict(job_id=123,user=pwd.getpwuid(os.getuid()).pw_name,restart_cnt=0,
            state=dict(current=['FAILED']),account='zixianma',partition='gpu-h200',qos='normal',nodes='fixture-node',
            working_directory=str(previous),submit_line=shlex.join(argv),
            time=dict(elapsed=elapsed,start=1000,end=1000+elapsed,limit=finite(120)),
            tres=dict(allocated=[dict(type=k,name='',count=v) for k,v in [('cpu',2),('mem',8192),('node',1)]]))])
        failure=self.root/(site+'-failure.json');m.put(failure,dict(site=site,job_id='123',predecessor_root=str(previous),
            status='failed_transfer_closed_independently_audited',ledger_sha256=m.sha(attempt/'transfer-ledger.json'),
            terminal=dict(scheduler_snapshot=terminal)))
        scratch_receipt=self.root/(site+'-scratch.json');m.put(scratch_receipt,dict(node='fixture-node',rows=[
            dict(site=site,path=scratch,exists=True,apparent_bytes=60,reserved_bytes=m.PRIOR_SCRATCH_RESERVE)]))
        repair=self.root/(site+'-repair.json');m.put(repair,dict(status='passed',tests=dict(exit_code=0,passed=3),
            source_sha256={'scripts/acquire_webprm_cms_original.py':m.sha(m.REPO/'scripts/acquire_webprm_cms_original.py')}))
        return dict(predecessor=previous,transfer_failure_audit=failure,prior_scratch_receipt=scratch_receipt,repair_receipt=repair)

    def exception_package(self,site='shopping',elapsed=37):
        kwargs=self.transfer_fixture(site,elapsed);root=self.root/(site+'-exception')
        with patch.object(m.subprocess,'check_output',side_effect=AssertionError('Offline preparation must not query scheduler')):
            p=m.prepare(root,site,**kwargs)
        return root,p,kwargs

    def approve_exception(self,root):
        p=self.approve(root);a=m.read(root/'approval.json')
        a['additional_transfer_exception']=m.transfer_approval(p['transfer_recovery'])
        (root/'approval.json').write_text(json.dumps(a));return p

    def test_transfer_successor_stays_pending_and_never_inherits_network_approval(self):
        root,p,_=self.exception_package()
        self.assertEqual(m.read(root/'approval.json')['status'],'pending')
        self.assertIsNone(m.read(root/'approval.json')['exact_user_approval'])
        with patch.object(m.os,'execvpe',side_effect=AssertionError('No allocation')):
            with self.assertRaisesRegex(m.Closed,'approval'):m.launch(root)
        self.approve(root)
        with patch.object(m,'SCRIPT',root/'acquisition.py'),self.assertRaisesRegex(m.Closed,'additional-transfer'):
            m.approved(root)
        self.approve_exception(root)
        with patch.object(m,'SCRIPT',root/'acquisition.py'):self.assertEqual(m.approved(root),p)
        self.assertFalse((root/'launch.json').exists())

    def test_transfer_successor_carries_both_actual_time_profiles_and_unknown_accounting(self):
        for site,elapsed in [('shopping',37),('gitlab',42)]:
            root,p,kw=self.exception_package(site,elapsed);proof=p['transfer_recovery'];old=m.read(kw['predecessor']/'proposal.json')
            self.assertEqual(proof['prior_allocation_seconds'],elapsed);self.assertEqual(proof['requested_allocation_seconds'],7140)
            self.assertEqual(proof['unresolved_prior_reservations'],1)
            for key in ('resources','profile','limits','items'):self.assertEqual(p[key],old[key])
            cumulative=m.network_accounting(p);prior=old['prior_metadata_accounting']
            self.assertEqual(cumulative['prior_charged_bytes'],prior['prior_charged_bytes']+2*65536+3+64*2**20)
            self.assertEqual(cumulative['prior_metadata_bytes'],prior['prior_metadata_bytes']+3)
            self.assertEqual(cumulative['prior_responses'],prior['prior_responses']+2)
            lib=SimpleNamespace(apparent_bytes=lambda path:123,checked_path=lambda path:path)
            self.assertEqual(m.previous_usage(root,p,lib),(123,4*2**20))

    def test_additional_transfer_claim_is_exclusive_and_previous_ledger_remains_bound(self):
        root,p,kw=self.exception_package();self.approve_exception(root)
        with self.assertRaises(FileExistsError):m.prepare(self.root/'duplicate','shopping',**kw)
        old=kw['predecessor'];ledger=m.read(old/'attempt-123/transfer-ledger.json');ledger['charged_bytes']-=64*2**20
        (old/'attempt-123/transfer-ledger.json').write_text(json.dumps(ledger))
        with patch.object(m,'SCRIPT',root/'acquisition.py'),self.assertRaises(ValueError):m.approved(root)

    def test_extra_transfer_proof_and_scope_cannot_be_changed_with_source_only_approval(self):
        root,p,_=self.exception_package();self.approve_exception(root)
        p['transfer_recovery']['prior_network_accounting']['prior_charged_bytes']=0
        (root/'proposal.json').write_text(json.dumps(p));self.approve_exception(root)
        with patch.object(m,'SCRIPT',root/'acquisition.py'),self.assertRaises(ValueError):m.approved(root)

    def test_explicit_second_transfer_cannot_spawn_a_third_successor(self):
        root,p,kw=self.exception_package();self.approve_exception(root)
        with self.assertRaisesRegex(m.Closed,'another successor'):
            m.prepare(self.root/'third','shopping',predecessor=root,prior_job='124',repair_receipt=kw['repair_receipt'])

    def test_frozen_successor_ledger_uses_cumulative_caps_before_any_network(self):
        program=r'''
import importlib.util,pathlib,socket,sys
from unittest.mock import patch
root=pathlib.Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('successor',root/'acquisition.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=m.read(root/'proposal.json');prior=m.network_accounting(p)
with patch.object(socket.socket,'connect',side_effect=AssertionError('No live network')):
 lib,_=m.libraries(root,p['site']);folder=root/'ledger-fixture';folder.mkdir()
 ledger=lib.TransferLedger(folder,cap=p['limits']['download_charged_bytes'],metadata_cap=p['limits']['metadata_received_bytes'],**prior)
 assert ledger.cap+prior['prior_charged_bytes']==p['limits']['download_charged_bytes']
 assert ledger.metadata_cap+prior['prior_metadata_bytes']==p['limits']['metadata_received_bytes']
 assert ledger.prior_responses==prior['prior_responses']
 url=p['items'][0]['url'];before=ledger.data['charged_bytes']
 try:ledger.reserve(ledger.cap+1,url,0,1)
 except ValueError:pass
 else:raise AssertionError('Cumulative charge cap bypassed')
 assert ledger.data['charged_bytes']==before
 with patch.dict(lib.confirmation_transport.LIMITS,max_http_responses=prior['prior_responses']):
  try:ledger.reserve(1,url,0,0)
  except ValueError:pass
  else:raise AssertionError('Prior responses were ignored')
 assert len(ledger.data['reservations'])==0
 i=ledger.reserve(1,url,0,0)
 assert ledger.data['reservations'][i]['state']=='unknown'
 assert ledger.data['prior_charged_bytes']==prior['prior_charged_bytes']
 assert ledger.data['charged_bytes']==65537
'''
        for site,elapsed in [('shopping',37),('gitlab',42)]:
            root,p,kw=self.exception_package(site,elapsed)
            ledger=kw['predecessor']/'attempt-123/transfer-ledger.json';before=m.sha(ledger)
            run=subprocess.run([sys.executable,'-I','-B','-c',program,str(root)],capture_output=True,text=True,timeout=20)
            self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(m.sha(ledger),before)


if __name__=='__main__':unittest.main()
