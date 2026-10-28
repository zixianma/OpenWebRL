"""Original-site profile, source and frozen transport tests; no live HTTP/jobs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('original_sites',Path(__file__).resolve().parents[1]/'scripts/acquire_webprm_original_sites.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class OriginalAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='original-acquisition-tests-',dir=m.RUNTIME/'next-stage-preparation')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def package(self,site):
        root=self.root/site;m.prepare(root,site);return root

    def snapshot(self):
        finite=lambda x:dict(set=True,infinite=False,number=x)
        return dict(errors=[],jobs=[dict(job_id=123,user_id=os.getuid(),job_state=['RUNNING'],restart_cnt=0,
            account='zixianma',partition='gpu-h200',qos='normal',cpus=finite(2),node_count=finite(1),memory_per_node=finite(8192),
            start_time=finite(100),end_time=finite(7303),time_limit=finite(120),tres_alloc_str='cpu=2,mem=8G,node=1')])

    def test_original_profiles_are_distinct_and_independent(self):
        a=self.package('shopping');b=self.package('gitlab')
        ap,bp=m.read(a/'proposal.json'),m.read(b/'proposal.json')
        self.assertEqual(ap['profile']['archive_bytes'],67575898112)
        self.assertEqual(bp['profile']['archive_bytes'],77755595776)
        self.assertNotEqual(ap['profile']['drive_id'],bp['profile']['drive_id'])
        self.assertNotEqual(ap['profile']['image_tag'],bp['profile']['image_tag'])
        self.assertEqual(ap['resources']['gpus'],0)

    def test_pending_approval_rejects_before_launch_claim(self):
        root=self.package('shopping')
        with patch.object(m.subprocess,'check_output',side_effect=AssertionError('No scheduler call')):
            with self.assertRaises(m.Closed):m.launch(root)
        self.assertFalse((root/'launch.json').exists())

    def test_frozen_dependency_change_invalidates_authorization(self):
        root=self.package('shopping');p=m.read(root/'proposal.json')
        (root/'approval.json').write_text(json.dumps(dict(status='approved',proposal_sha256=m.digest(p),exact_user_approval='Offline test fixture; never launched')))
        with patch.object(m,'SCRIPT',root/'acquisition.py'):
            self.assertEqual(m.approved(root),p)
            with (root/'dependencies/openwebrl/jev_eval.py').open('a') as out:out.write('\n# changed fixture\n')
            with self.assertRaises(m.Closed):m.approved(root)

    def test_scheduler_grace_clamped_and_wrong_gpu_rejected(self):
        d=self.snapshot();self.assertEqual(m.context(d,'123',101)['allocation_deadline'],7300)
        d['jobs'][0]['tres_alloc_str']+=',gres/gpu=1'
        with self.assertRaises(m.Closed):m.context(d,'123',101)

    def test_scheduler_extra_requested_time_rejected(self):
        d=self.snapshot();d['jobs'][0]['time_limit']['number']=121
        with self.assertRaises(m.Closed):m.context(d,'123',101)

    def test_response_counts_and_storage_cover_fixed_originals(self):
        for p in m.PROFILES.values():
            limits=m.limits(p)
            ranges=(p['archive_bytes']+64*2**20-1)//(64*2**20)
            self.assertGreaterEqual(limits['max_http_responses'],ranges*4)
            self.assertLess(p['archive_bytes']+ranges*65536,p['charged_download_bytes'])
            self.assertLess(p['archive_bytes']+64*2**20,p['retained_bytes'])

    def test_each_frozen_profile_uses_its_own_transport_and_inspector(self):
        program=r'''
import importlib.util,json,pathlib,sys,time,io,hashlib,tarfile
root=pathlib.Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('profile_worker',root/'acquisition.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=m.read(root/'proposal.json')['profile'];lib,inspector=m.load_library(root,p)
assert lib.ARCHIVE_NAME==p['filename'] and lib.ARCHIVE_BYTES==p['archive_bytes']
assert inspector.IMAGE_TAG==p['image_tag'] and p['drive_id'] in lib.SOURCE_URL
assert lib.LIMITS['download_charged_bytes']==p['charged_download_bytes']
attempt=root/'synthetic-transfer';attempt.mkdir()
ledger=lib.TransferLedger(attempt,cap=p['charged_download_bytes'],metadata_cap=16*2**20)
class Response:
 status=206
 headers={'Content-Length':'6','Content-Range':'bytes 0-5/6','Content-Disposition':'attachment; filename='+p['filename'],'ETag':'"fixed"'}
 def __init__(self):self.body=io.BytesIO(b'abcdef')
 def read(self,n):return self.body.read(n)
 def close(self):pass
def fake(request,timeout):
 assert p['drive_id'] in request.full_url
 return Response()
transfer=lib.download(attempt,ledger,time.time()+15,open_response=fake,size=6,chunk=6)
assert transfer['bytes']==6 and ledger.data['charged_bytes']==65542
assert (attempt/p['filename']).read_bytes()==b'abcdef'
def tar(items):
 b=io.BytesIO()
 with tarfile.open(fileobj=b,mode='w') as t:
  for name,raw in items:
   v=tarfile.TarInfo(name);v.size=len(raw);t.addfile(v,io.BytesIO(raw))
 return b.getvalue()
layer=tar([('var/data/synthetic',b'baseline')]);config=json.dumps(dict(os='linux',architecture='amd64',rootfs=dict(diff_ids=['sha256:'+hashlib.sha256(layer).hexdigest()]))).encode()
config_name=hashlib.sha256(config).hexdigest()+'.json'
manifest=json.dumps([dict(RepoTags=[p['image_tag']],Config=config_name,Layers=['layer/layer.tar'])]).encode()
archive=tar([('manifest.json',manifest),(config_name,config),('layer/layer.tar',layer)])
path=root/'synthetic-archive.tar';path.write_bytes(archive)
proof=inspector.inspection(path,deadline=time.time()+15,limits=inspector.Limits(archive_bytes=len(archive),cumulative_regular_bytes=p['expanded_regular_bytes']))
assert proof['image_tag']==p['image_tag'] and not proof['ready_to_host'] and not proof['extraction_performed']
print(json.dumps({'frozen_profile':p['image_tag'],'transport_bytes':6,'config_and_layer_identity_verified':True,'real_network_calls':0}))
'''
        for site in m.PROFILES:
            with self.subTest(site=site):
                root=self.package(site)
                result=subprocess.run([sys.executable,'-I','-B','-c',program,str(root)],capture_output=True,text=True,timeout=20)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(json.loads(result.stdout)['real_network_calls'],0)

    def test_manual_successor_inherits_only_remaining_time_and_unique_ownership(self):
        previous=self.package('shopping');pp=m.read(previous/'proposal.json')
        # Preserve a synthetic old bug in the predecessor source identity.
        with (previous/'acquisition.py').open('a') as out:out.write('\n# old startup fixture\n')
        pp['source_sha256']['acquisition.py']=m.sha(previous/'acquisition.py')
        (previous/'proposal.json').write_text(json.dumps(pp))
        m.put(previous/'launch.json',dict(nonce='test-nonce',proposal_sha256=m.digest(pp),requested_allocation_seconds=7200))
        (previous/'approval.json').write_text(json.dumps(dict(status='approved',proposal_sha256=m.digest(pp),exact_user_approval='Fixture only')))
        expected=dict(pp['source_sha256']);expected['acquisition.py']=m.sha(m.SCRIPT)
        repair=self.root/'repair.json';m.put(repair,dict(status='passed',test_exit_code=0,diagnosis='Bounded clock fix',
            test_argv=['python3','offline_test.py'],predecessor_proposal_sha256=m.digest(pp),replacement_source_sha256=expected))
        snapshot=self.snapshot();row=snapshot['jobs'][0];row.update(job_state=['FAILED'],comment='webprm-acquire:test-nonce')
        row['end_time']['number']=109
        successor=self.root/'successor'
        with patch.object(m.subprocess,'check_output',return_value=json.dumps(snapshot)):
            new=m.prepare(successor,'shopping',predecessor=previous,prior_job='123',repair_receipt=repair)
            with self.assertRaises(FileExistsError):m.prepare(self.root/'duplicate','shopping',predecessor=previous,prior_job='123',repair_receipt=repair)
        self.assertEqual(new['startup_recovery']['requested_allocation_seconds'],7140)
        with patch.object(m,'SCRIPT',successor/'acquisition.py'):self.assertEqual(m.approved(successor),new)
        m.put(previous/'network-claim.json',{})
        with patch.object(m,'SCRIPT',successor/'acquisition.py'):
            with self.assertRaises(ValueError):m.approved(successor)

    def test_storage_is_cumulative_across_preserved_startups(self):
        previous=self.package('shopping');successor=self.root/'current';successor.mkdir()
        scratch=Path(tempfile.mkdtemp(dir='/tmp',prefix='webprm-original-fixture-'));self.addCleanup(lambda:__import__('shutil').rmtree(scratch))
        (scratch/'cached').write_bytes(b'old-cache')
        attempt=previous/'attempt-123';attempt.mkdir();m.put(attempt/'context.json',dict(scratch=str(scratch)))
        class Library:
            @staticmethod
            def checked_path(path):return path
            @staticmethod
            def apparent_bytes(path):return sum(x.stat().st_size for x in path.rglob('*') if x.is_file())
        prior=Library.apparent_bytes(previous)
        p=dict(startup_recovery=dict(attempt_number=2,predecessor_root=str(previous)))
        self.assertEqual(m.previous_usage(successor,p,Library),(prior,9))

    def test_protected_home_output_rejected(self):
        with self.assertRaises(m.Closed):m.checked('/gpfs/home/zixianma/forbidden-original-site')


if __name__=='__main__':unittest.main()
