"""Offline append identity, cumulative caps, and narrowly bounded TLS reconnects."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch
from scripts import acquire_webprm_map_backend as backend
from scripts import acquire_webprm_cms_original as cms
from scripts import webprm_map_resume as m
from scripts import acquire_webprm_map_backend_resume as owner
from scripts import webprm_cms_oci_transport as oci

class Response:
    def __init__(self,body,status=206,headers=None):self.body=io.BytesIO(body);self.status=status;self.headers=headers or {}
    def read(self,n):return self.body.read(n)
    def close(self):pass

def etag(body,part=4):
    value=hashlib.md5(b''.join(hashlib.md5(body[i:i+part]).digest() for i in range(0,len(body),part))).hexdigest()
    return '"'+value+'-'+str((len(body)+part-1)//part)+'"'

class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=backend.N,prefix='map-resume-test-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.prior=self.root/'prior';self.prior.mkdir();self.output=self.root/'new';self.output.mkdir()
        self.limits=dict(backend.LIMITS);self.patch=patch.object(cms,'LIMITS',self.limits);self.patch.start();self.addCleanup(self.patch.stop)
        self.ledger=m.ledger_class(cms,self.limits,1000)(self.output,prior_charged_bytes=123456,prior_metadata_bytes=37,prior_responses=7)
        self.body=b'abcdefghijkl';self.path=self.prior/'data.tar';self.path.write_bytes(self.body[:5])
        info=m.stat_identity(self.path.stat());self.proof=dict(status='closed_local_prefix_sha256_verified',prefix_path=str(self.path),
            file_mutated=False,stat_before=info,stat_after=info,fixed_prefix_bytes=5,bytes_read=5,
            fixed_prefix_sha256=hashlib.sha256(self.body[:5]).hexdigest())
        self.item=dict(kind='dataset',filename='datasets/data.tar',bytes=len(self.body),etag=etag(self.body),
            url='https://webarena-map-server-data.s3.amazonaws.com/data.tar')
        self.calls=[]
    def response(self,req,timeout):
        self.calls.append(req);span=req.get_header('Range')[6:];start,end=map(int,span.split('-'))
        return Response(self.body[start:end+1],headers={'Content-Length':str(end-start+1),
            'Content-Range':f'bytes {start}-{end}/{len(self.body)}','ETag':self.item['etag']})
    def transfer(self,**kwargs):
        return m.transfer_dataset(backend,self.output,self.item,cms,self.ledger,time.time()+30,
            prefix=self.proof,part_bytes=4,chunk=3,open_response=kwargs.pop('open_response',self.response),**kwargs)
    def test_append_preserves_inode_prefix_and_verifies_persisted_file(self):
        result=self.transfer();self.assertEqual(self.path.read_bytes(),self.body)
        self.assertEqual(self.path.stat().st_ino,self.proof['stat_before']['st_ino'])
        self.assertTrue(result['persisted_whole_file_and_prefix_readback_verified'])
        self.assertEqual(result['sha256'],hashlib.sha256(self.body).hexdigest());self.assertEqual(result['computed_multipart_etag'],etag(self.body))
        self.assertEqual([x.get_header('Range') for x in self.calls],['bytes=5-7','bytes=8-10','bytes=11-11'])
        self.assertTrue(all(x.get_header('If-match')==self.item['etag'] for x in self.calls))
        self.assertEqual(self.ledger.data['charged_bytes'],7+3*65536)
    def test_wrong_prefix_hash_stops_before_network(self):
        self.proof['fixed_prefix_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'SHA/stat'):self.transfer()
        self.assertEqual(self.calls,[]);self.assertEqual(self.path.read_bytes(),self.body[:5])
    def test_saved_device_number_is_portable_but_worker_inode_is_stable(self):
        self.proof['stat_before']=dict(self.proof['stat_before'],st_dev=999999)
        self.proof['stat_after']=dict(self.proof['stat_after'],st_dev=999999)
        result=self.transfer();self.assertTrue(result['device_id_changed_across_hosts'])
        self.assertEqual(result['saved_device_id'],999999)
    def test_final_pathname_replacement_is_rejected(self):
        original=Path.stat;reads=[0]
        def mocked(path,*args,**kwargs):
            result=original(path,*args,**kwargs)
            if path==self.path and result.st_size==len(self.body):
                values=list(result);values[1]+=1
                # Preserve nanosecond fields while changing only the inode.
                from types import SimpleNamespace
                return SimpleNamespace(**dict(m.stat_identity(result),st_ino=result.st_ino+1))
            return result
        with patch.object(Path,'stat',mocked):
            with self.assertRaisesRegex(ValueError,'pathname'):self.transfer()
    def test_missing_or_changed_partial_stops_before_network(self):
        with patch.object(self.path.__class__,'stat',side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):self.transfer()
        self.path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'stat changed'):self.transfer()
        self.assertEqual(self.calls,[])
    def test_changed_range_etag_full200_and_duplicate_headers_reject_without_retry(self):
        from email.message import Message
        for kind in ('range','etag','200','duplicate'):
            with self.subTest(kind=kind):
                def bad(req,timeout):
                    response=self.response(req,timeout)
                    if kind=='range':response.headers['Content-Range']='bytes 0-2/12'
                    if kind=='etag':response.headers['ETag']='"other"'
                    if kind=='200':response.status=200
                    if kind=='duplicate':
                        headers=Message()
                        for k,v in response.headers.items():headers[k]=v
                        headers['Content-Length']='3';response.headers=headers
                    return response
                with self.assertRaises(ValueError):self.transfer(open_response=bad)
                self.assertEqual(self.path.read_bytes(),self.body[:5])
        self.assertEqual(len(self.calls),4);self.assertEqual(len(self.ledger.data['tls_handshake_reconnects']),0)
    def test_partial_body_failure_never_retries_and_remains_charged(self):
        def bad(req,timeout):
            response=self.response(req,timeout);response.body=io.BytesIO(b'f');return response
        with self.assertRaisesRegex(ValueError,'Truncated'):self.transfer(open_response=bad)
        self.assertEqual(len(self.calls),1);self.assertEqual(self.path.read_bytes(),b'abcdef')
        self.assertEqual(self.ledger.data['charged_bytes'],3+65536)
        self.assertEqual(self.ledger.data['reservations'][0]['state'],'uncertain')
    def test_content_etag_mismatch_prevents_completion(self):
        self.item['etag']='"'+'0'*32+'-3"'
        with self.assertRaisesRegex(ValueError,'multipart ETag'):self.transfer()
    def test_global_three_handshake_reconnects_keep_every_unknown_charge(self):
        request=urllib.request.Request(self.item['url']);calls=[]
        def fail(req,timeout):calls.append(req);raise urllib.error.URLError(TimeoutError('_ssl.c:981: The handshake operation timed out'))
        with self.assertRaises(urllib.error.URLError):backend.request_once(cms,self.ledger,request,3,time.time()+30,open_response=fail)
        self.assertEqual(len(calls),4);self.assertEqual(len(self.ledger.data['tls_handshake_reconnects']),3)
        self.assertEqual(self.ledger.data['charged_bytes'],4*(3+65536))
        with self.assertRaises(urllib.error.URLError):backend.request_once(cms,self.ledger,request,3,time.time()+30,open_response=fail)
        self.assertEqual(len(calls),5);self.assertTrue(all(x['state']=='unknown' for x in self.ledger.data['reservations']))
    def test_reconnect_budget_shared_between_dataset_and_oci(self):
        failures=[1,1,0,1,0,0];calls=[]
        def dataset(req,timeout):
            calls.append(req)
            if failures.pop(0):raise urllib.error.URLError(TimeoutError('_ssl.c:981: The handshake operation timed out'))
            return self.response(req,timeout)
        self.transfer(open_response=dataset)
        self.assertEqual(len(self.ledger.data['tls_handshake_reconnects']),3)
        item=dict(kind='image_blob',filename='oci/blob',bytes=2,sha256=hashlib.sha256(b'hi').hexdigest(),
            url='https://ghcr.io/v2/repo/blobs/sha256:fixture',registry='https://ghcr.io',repository='repo')
        class Token:
            def token(self,item):return 'test'
        def fail(req,timeout):raise urllib.error.URLError(TimeoutError('_ssl.c:981: The handshake operation timed out'))
        before=len(self.ledger.data['reservations'])
        with self.assertRaises(urllib.error.URLError):oci.transfer_blob(self.output,item,self.ledger,time.time()+30,Token(),open_response=fail)
        self.assertEqual(len(self.ledger.data['reservations']),before+1)
    def test_non_handshake_errors_and_default_ledger_do_not_retry(self):
        for reason in (TimeoutError('read timed out'),OSError('certificate verify failed'),TimeoutError('The handshake operation timed out')):
            calls=[]
            def fail(req,timeout):calls.append(req);raise urllib.error.URLError(reason)
            with self.assertRaises(urllib.error.URLError):backend.request_once(cms,self.ledger,urllib.request.Request(self.item['url']),3,time.time()+10,open_response=fail)
            self.assertEqual(len(calls),1)
        self.assertEqual(self.ledger.data['tls_handshake_reconnects'],[])
    def test_imported_caps_prevent_connection_and_full200_expansion(self):
        self.ledger.cap=self.ledger.data['charged_bytes']+65538
        with self.assertRaises(cms.GateClosed):backend.request_once(cms,self.ledger,urllib.request.Request(self.item['url']),3,time.time()+10,open_response=lambda *a,**k:self.fail('No network'))
        self.ledger.cap=10**8;index=self.ledger.reserve(3,self.item['url'],None,None);self.ledger.cap=self.ledger.data['charged_bytes']+5
        with self.assertRaises(ValueError):self.ledger.expand(index,10)
        self.ledger.prior_responses=self.limits['max_http_responses']
        with self.assertRaises(ValueError):self.ledger.reserve(1,self.item['url'],None,None)
        self.ledger.metadata_cap=0
        with self.assertRaises(cms.GateClosed):oci.Registry(cms,self.ledger,time.time()+10,open_response=lambda *a,**k:self.fail('No network')).token(dict(registry='https://ghcr.io',repository='repo'))
    def test_multipart_stream_boundaries_and_large_parts(self):
        for part in (1,4,5,512*2**20):
            value=m.Multipart(part)
            for i in range(0,len(self.body),3):value.update(self.body[i:i+3])
            self.assertEqual(value.etag(),etag(self.body,part))
    def test_future_predecessor_growth_is_reserved_upfront(self):
        class Lib:
            def apparent_bytes(self,path):return 100
        self.assertEqual(owner.storage_reservation(Lib(),self.prior,{'preserved_partial_append_growth_bytes':1000}),(100,1100))
    def test_exact_exception_is_separate_and_pending_cannot_launch(self):
        owner.put(self.output/'proposal.json',dict(root=str(self.output)));owner.put(self.output/'approval.json',dict(status='pending'))
        with patch.object(owner.os,'execvpe',side_effect=AssertionError('No allocation')):
            with self.assertRaises(owner.Closed):owner.launch(self.output)
        approval=owner.exception_approval(dict(requested_allocation_seconds=13020))
        self.assertEqual(approval['maximum_tls_handshake_reconnects'],3);self.assertFalse(approval['further_successors_allowed'])
        self.assertFalse((self.output/'launch.json').exists())

class PackageTests(unittest.TestCase):
    def test_frozen_pending_package_import_caps_and_exclusive_claim(self):
        with tempfile.TemporaryDirectory(dir=backend.N,prefix='map-resume-package-test-') as temp:
            root=Path(temp);previous=root/'previous';previous.mkdir();evidence=root/'evidence';evidence.mkdir()
            for name in owner.EVIDENCE.values():(evidence/name).write_text('{}\n')
            proof=dict(requested_allocation_seconds=13020,predecessor_root=str(previous))
            package=root/'package'
            with patch.object(owner.recovery,'proof',return_value=proof):p=owner.prepare(package,previous,evidence)
            self.assertEqual(owner.read(package/'approval.json')['status'],'pending')
            self.assertIsNone(owner.read(package/'approval.json')['additional_transfer_exception'])
            self.assertEqual(owner.read(previous/'successor-claim.json')['proposal_sha256'],owner.digest(p))
            with self.assertRaisesRegex(owner.Closed,'exclusive successor'):owner.prepare(root/'second',previous,evidence)
            program='''import importlib.util,pathlib,sys
p=pathlib.Path(sys.argv[1]);s=importlib.util.spec_from_file_location('f',p/'acquisition.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.validate_sources(p,m.read(p/'proposal.json'));lib=m.library(p)
assert lib.LIMITS['temporary_bytes']==508*2**20
assert m.oci.registry.request_once is m.backend.request_once
try:m.approved(p)
except m.Closed:pass
else:raise AssertionError('Pending approval accepted')
'''
            result=subprocess.run([sys.executable,'-I','-B','-c',program,str(package)],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            a=dict(status='approved',proposal_sha256=owner.digest(p),exact_user_approval='fixture exact exception',
                inherited_authority=False,additional_transfer_exception=owner.exception_approval(proof))
            original=owner.read
            def approval(path):return a if path==package/'approval.json' else original(path)
            with patch.object(owner,'SCRIPT',package/'acquisition.py'),patch.object(owner,'read',approval),patch.object(owner.recovery,'proof',return_value=proof):
                self.assertEqual(owner.approved(package),p)
                a['inherited_authority']=True
                with self.assertRaises(owner.Closed):owner.approved(package)
                a['inherited_authority']=False;a['additional_transfer_exception']=None
                with self.assertRaises(owner.Closed):owner.approved(package)
                a['additional_transfer_exception']=owner.exception_approval(proof)
                a['additional_transfer_exception']['maximum_tls_handshake_reconnects']=4
                with self.assertRaises(owner.Closed):owner.approved(package)
            (package/'dependencies/scripts/webprm_map_resume.py').write_text('changed fixture')
            with self.assertRaisesRegex(owner.Closed,'source changed'):owner.validate_sources(package,p)
    def test_real_failure_proof_small_metadata_only_and_tampered_charges_rejected(self):
        previous=backend.N/'map-original-backend-acquisition-v1';evidence=backend.N/'map-backend-failure-audit-356008-v1'
        with tempfile.TemporaryDirectory(dir=backend.N,prefix='map-resume-proof-test-') as temp:
            metadata=Path(temp)
            for name,source in owner.EVIDENCE.items():(metadata/name).write_bytes((evidence/source).read_bytes())
            sources=owner.read(previous/'proposal.json')['source_sha256']
            proof=owner.recovery.proof(backend,previous,metadata,sources)
            self.assertEqual(proof['requested_allocation_seconds'],13020)
            self.assertEqual(proof['prior_network_accounting'],dict(prior_charged_bytes=39968768000,prior_metadata_bytes=0,prior_responses=595))
            self.assertEqual(proof['preserved_partial_append_growth_bytes'],1417662464)
            original=backend.read
            def tamper(path):
                value=original(path)
                if path.name=='transfer-ledger.json':value['charged_bytes']-=1
                return value
            with patch.object(backend,'read',tamper):
                with self.assertRaisesRegex(ValueError,'accounting'):owner.recovery.proof(backend,previous,metadata,sources)

if __name__=='__main__':unittest.main()
