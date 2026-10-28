"""Offline bounded Map transport/identity/approval tests; zero live transfers."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('map_backend',Path(__file__).resolve().parents[1]/'scripts/acquire_webprm_map_backend.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Response:
    def __init__(self,body=b'',status=206,headers=None):
        self.body=io.BytesIO(body);self.status=status;self.headers=headers or {}
    def read(self,n):return self.body.read(n)
    def close(self):pass


class MapBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(dir=m.N,prefix='map-backend-test-');cls.addClassCleanup(cls.tmp.cleanup)
        cls.root=Path(cls.tmp.name)/'package';m.prepare(cls.root);cls.lib=m.library(cls.root)

    def setUp(self):
        self.output=self.root/('attempt-'+self._testMethodName);self.output.mkdir()
        self.ledger=self.lib.TransferLedger(self.output,cap=m.LIMITS['download_charged_bytes'],metadata_cap=16*2**20)

    def item(self,**extra):
        item=dict(kind='dataset',filename='datasets/a.tar',url='https://webarena-map-server-data.s3.amazonaws.com/a.tar',bytes=6,etag='"fixed"')
        item.update(extra);return item

    def test_metadata_pins_and_accurate_whole_data_size(self):
        items,images=m.inputs(self.root)
        self.assertEqual(sum(x['bytes'] for x in items if x['kind']=='dataset'),189212856320)
        self.assertEqual(sum(x['bytes'] for x in items),190255688990)
        self.assertEqual(len(images),3);self.assertEqual(len(items),35)
        self.assertLess(sum(x['bytes'] for x in items)+2*1042832734+12000*65536+16*2**20,m.LIMITS['download_charged_bytes'])
        self.assertEqual(m.RESOURCES['maximum_allocation_attempts'],3)

    def test_pending_approval_cannot_launch(self):
        with patch.object(m.os,'execvpe',side_effect=AssertionError('No allocation')):
            with self.assertRaises(m.Closed):m.launch(self.root)
        self.assertFalse((self.root/'launch.json').exists())

    def test_dataset_ranges_preserve_if_match_and_streamed_hash(self):
        requests=[]
        def fake(req,timeout):
            requests.append(req);start=0 if len(requests)==1 else 3
            return Response(b'abcdef'[start:start+3],headers={'Content-Length':'3','Content-Range':f'bytes {start}-{start+2}/6','ETag':'"fixed"'})
        result=m.transfer(self.output,self.item(),self.lib,self.ledger,time.time()+10,None,open_response=fake,chunk=3)
        self.assertEqual(result['sha256'],hashlib.sha256(b'abcdef').hexdigest())
        self.assertTrue(all(x.get_header('If-match')=='"fixed"' for x in requests))
        self.assertEqual(self.ledger.data['charged_bytes'],6+2*65536)

    def test_changed_entity_retains_unknown_full_reservation(self):
        def fake(req,timeout):return Response(b'abcdef',headers={'Content-Length':'6','Content-Range':'bytes 0-5/6','ETag':'"changed"'})
        with self.assertRaises(m.Closed):m.transfer(self.output,self.item(),self.lib,self.ledger,time.time()+10,None,open_response=fake)
        self.assertEqual(self.ledger.data['reservations'][0]['state'],'uncertain')
        self.assertEqual(self.ledger.data['charged_bytes'],6+65536)

    def test_truncation_is_not_retried_or_refunded(self):
        calls=[]
        def fake(req,timeout):calls.append(req);return Response(b'abc',headers={'Content-Length':'6','Content-Range':'bytes 0-5/6','ETag':'"fixed"'})
        with self.assertRaises(m.Closed):m.transfer(self.output,self.item(),self.lib,self.ledger,time.time()+10,None,open_response=fake)
        self.assertEqual(len(calls),1);self.assertEqual(self.ledger.data['charged_bytes'],6+65536)
        self.assertEqual((self.output/'datasets/a.tar').read_bytes(),b'abc')

    def test_digest_mismatch_keeps_download_evidence(self):
        item=self.item(kind='image_blob',registry='https://ghcr.io',repository='project-osrm/osrm-backend',
            url='https://ghcr.io/v2/project-osrm/osrm-backend/blobs/sha256:'+'0'*64,sha256='0'*64)
        class Registry:
            def token(self,item):return 'fixture-token'
        def fake(req,timeout):return Response(b'abcdef',status=200,headers={'Content-Length':'6'})
        with self.assertRaises(m.Closed):m.transfer(self.output,item,self.lib,self.ledger,time.time()+10,Registry(),open_response=fake)
        self.assertEqual((self.output/'datasets/a.tar').stat().st_size,6)

    def test_redirect_does_not_forward_bearer_token(self):
        calls=[];item=self.item(kind='image_blob',registry='https://ghcr.io',repository='project-osrm/osrm-backend',
            url='https://ghcr.io/v2/project-osrm/osrm-backend/blobs/sha256:'+hashlib.sha256(b'abcdef').hexdigest(),sha256=hashlib.sha256(b'abcdef').hexdigest())
        class Registry:
            def token(self,item):return 'fixture-token'
        def fake(req,timeout):
            calls.append(req)
            if len(calls)==1:return Response(status=307,headers={'Location':'https://pkg-containers.githubusercontent.com/blob?secret=never-persist','Content-Length':'0'})
            return Response(b'abcdef',status=200,headers={'Content-Length':'6'})
        m.transfer(self.output,item,self.lib,self.ledger,time.time()+10,Registry(),open_response=fake)
        self.assertEqual(calls[0].get_header('Authorization'),'Bearer fixture-token');self.assertIsNone(calls[1].get_header('Authorization'))
        saved=(self.output/'transfer-ledger.json').read_text()
        self.assertNotIn('never-persist',saved);self.assertNotIn('fixture-token',saved)

    def test_arbitrary_redirect_and_data_host_changes_rejected(self):
        item=self.item()
        for url in ('https://attacker.example/a','http://webarena-map-server-data.s3.amazonaws.com/a','https://u:p@webarena-map-server-data.s3.amazonaws.com/a'):
            with self.assertRaises(m.Closed):m.allowed_delivery(url,item)
        with self.assertRaises(m.Closed):m.allowed_delivery('https://pkg-containers.githubusercontent.com/a',item)

    def test_auth_is_charged_and_token_is_only_cached_in_memory(self):
        calls=[]
        def fake(req,timeout):calls.append(req);return Response(b'{"token":"synthetic-private-token","expires_in":300}',status=200)
        registry=m.Registry(self.lib,self.ledger,time.time()+10,fake)
        item=dict(registry='https://ghcr.io',repository='project-osrm/osrm-backend')
        self.assertEqual(registry.token(item),registry.token(item));self.assertEqual(len(calls),1)
        self.assertNotIn('synthetic-private-token',(self.output/'transfer-ledger.json').read_text())
        self.assertGreater(self.ledger.data['known_received_bytes'],0)

    def test_expired_deadline_prevents_any_request(self):
        with self.assertRaises(m.Closed):m.request_once(self.lib,self.ledger,m.urllib.request.Request(self.item()['url']),6,time.time()-1,open_response=lambda *a,**k:self.fail('No HTTP'))
        self.assertEqual(self.ledger.data['reservations'],[])

    def test_frozen_script_import_and_oci_metadata_are_complete(self):
        program='import importlib.util,pathlib,sys; p=pathlib.Path(sys.argv[1]); s=importlib.util.spec_from_file_location("f",p/"acquisition.py"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert m.base.RESOURCES["total_allocation_seconds_including_retries"]==14400; m.library(p); assert len(m.inputs(p)[0])==35'
        result=subprocess.run([sys.executable,'-I','-B','-c',program,str(self.root)],text=True,capture_output=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        m.oci_metadata(self.output,m.inputs(self.root)[1])
        self.assertEqual(len(m.read(self.output/'oci/index.json')['manifests']),3)


if __name__=='__main__':unittest.main()
