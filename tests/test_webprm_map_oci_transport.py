"""Offline reusable OCI controls for the prospective Map successor.

The six-byte body with a three-byte requested range exercises the same full200
branch as a >64MiB pinned blob. No network, large-body fixture or job is needed.
"""
from email.message import Message
import hashlib
import io
from pathlib import Path
import tempfile
import time
import unittest
import urllib.error
from unittest.mock import patch

from scripts import acquire_webprm_map_backend as original
from scripts import webprm_cms_confirmation_transport as accounting
from scripts import webprm_cms_oci_transport as transport


class Response:
    def __init__(self, status, headers, body=b''):
        self.status,self.headers,self.body=status,headers,io.BytesIO(body)
        self.reads=0;self.closed=False;self.before_read=None
    def read(self, maximum):
        self.reads+=1
        if self.before_read:self.before_read(maximum)
        return self.body.read(maximum)
    def close(self):self.closed=True


class Tokens:
    def token(self,item):return 'fixture-private-'+item['registry']


class MapOCITransportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/tmp',prefix='map-oci-fixture-')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        for guard in (patch('socket.socket.connect',side_effect=AssertionError('No network in fixtures')),
                      patch.object(accounting,'LIMITS',original.LIMITS),
                      patch.object(accounting.cms,'LIMITS',original.LIMITS),
                      patch.object(accounting,'ARCHIVE_BYTES',455519190)):
            guard.start();self.addCleanup(guard.stop)

    def item(self,registry='https://ghcr.io'):
        checksum=hashlib.sha256(b'abcdef').hexdigest()
        return dict(kind='image_blob',filename='oci/blobs/sha256/'+checksum,bytes=6,
            sha256=checksum,registry=registry,repository='project/image',
            url=registry+'/v2/project/image/blobs/sha256:'+checksum)

    def ledger(self,*,prior=123456789,prior_responses=1300,cap=195000000000,prior_metadata=147030):
        return accounting.CumulativeLedger(self.root,prior_charged_bytes=prior,
            prior_metadata_bytes=prior_metadata,prior_responses=prior_responses,
            cap=cap,metadata_cap=original.LIMITS['metadata_received_bytes'])

    def run_transfer(self,responses,ledger,**kwargs):
        requests=[]
        def opened(request,timeout):
            requests.append(request)
            return responses.pop(0)
        result=transport.transfer_blob(self.root,kwargs.pop('item',self.item()),ledger,time.time()+10,
            Tokens(),open_response=opened,chunk=3,**kwargs)
        return result,requests

    def test_full200_expansion_is_durable_before_body_and_preserves_prior_counters(self):
        ledger=self.ledger();response=Response(200,{'Content-Length':'6'},b'abcdef')
        def assert_reserved(_):
            saved=transport.cms.read(self.root/'transfer-ledger.json')
            self.assertEqual(saved['reservations'][0]['reserved_bytes'],6)
            self.assertEqual(saved['charged_bytes'],6+65536)
            self.assertEqual(saved['prior_charged_bytes'],123456789)
            self.assertEqual(saved['prior_metadata_bytes'],147030)
            self.assertEqual(saved['prior_responses'],1300)
        response.before_read=assert_reserved
        result,requests=self.run_transfer([response],ledger)
        self.assertEqual(result['bytes'],6);self.assertEqual(len(requests),1)
        self.assertEqual(ledger.data['reservations'][0]['state'],'complete')
        self.assertEqual(requests[0].get_header('Authorization'),'Bearer fixture-private-https://ghcr.io')
        self.assertTrue(response.closed)

    def test_insufficient_remaining_body_budget_rejects_before_read(self):
        prior=original.LIMITS['download_charged_bytes']-65536-3
        ledger=self.ledger(prior=prior);response=Response(200,{'Content-Length':'6'},b'abcdef')
        with self.assertRaisesRegex(transport.cms.GateClosed,'Full-body reservation'):
            self.run_transfer([response],ledger)
        self.assertEqual(response.reads,0)
        self.assertEqual(ledger.data['charged_bytes']+prior,195000000000)
        self.assertEqual(ledger.data['reservations'][0]['state'],'uncertain')

    def test_prior_response_count_blocks_before_any_request(self):
        ledger=self.ledger(prior_responses=12000)
        with patch.object(transport.registry,'request_once',wraps=transport.registry.request_once):
            with self.assertRaisesRegex(transport.cms.GateClosed,'response cap'):
                transport.transfer_blob(self.root,self.item(),ledger,time.time()+10,Tokens(),
                    open_response=lambda *_a,**_k:self.fail('No request permitted'),chunk=3)
        self.assertEqual(ledger.data['reservations'],[])

    def test_truncated_full200_keeps_full_expanded_unknown_charge(self):
        ledger=self.ledger();response=Response(200,{'Content-Length':'6'},b'abc')
        with self.assertRaisesRegex(transport.cms.GateClosed,'Truncated'):
            self.run_transfer([response],ledger)
        row=ledger.data['reservations'][0]
        self.assertEqual((row['state'],row['received_bytes'],row['reserved_bytes']),('uncertain',3,6))
        self.assertEqual(ledger.data['charged_bytes'],6+65536)
        self.assertEqual((self.root/self.item()['filename']).read_bytes(),b'abc')

    def test_tls_open_failure_keeps_unknown_reservation_without_retry(self):
        ledger=self.ledger();calls=[]
        def failed(request,timeout):
            calls.append(request)
            raise urllib.error.URLError('fixture TLS handshake timeout')
        with self.assertRaises(urllib.error.URLError):
            transport.transfer_blob(self.root,self.item(),ledger,time.time()+10,Tokens(),
                                    open_response=failed,chunk=3)
        self.assertEqual(len(calls),1)
        row=ledger.data['reservations'][0]
        self.assertEqual((row['state'],row['received_bytes'],row['reserved_bytes']),('unknown',0,3))
        self.assertEqual(ledger.data['charged_bytes'],3+65536)

    def test_full200_after_nonzero_offset_never_appends_duplicate_body(self):
        ledger=self.ledger();wrong=Response(200,{'Content-Length':'6'},b'abcdef')
        with self.assertRaisesRegex(transport.cms.GateClosed,'range/length'):
            self.run_transfer([Response(206,{'Content-Length':'3','Content-Range':'bytes 0-2/6'},b'abc'),wrong],ledger)
        self.assertEqual(wrong.reads,0)
        self.assertEqual((self.root/self.item()['filename']).read_bytes(),b'abc')
        self.assertEqual(ledger.data['charged_bytes'],6+2*65536)

    def test_ghcr_redirect_strips_token_and_unknown_body_keeps_reservation(self):
        ledger=self.ledger()
        result,requests=self.run_transfer([
            Response(307,{'Location':'https://pkg-containers.githubusercontent.com/blob?private=not-saved'}),
            Response(200,{'Content-Length':'6'},b'abcdef')],ledger)
        self.assertEqual(result['bytes'],6)
        self.assertIsNotNone(requests[0].get_header('Authorization'))
        self.assertIsNone(requests[1].get_header('Authorization'))
        self.assertEqual(ledger.data['charged_bytes'],3+6+2*65536)
        saved=(self.root/'transfer-ledger.json').read_text()
        self.assertNotIn('not-saved',saved);self.assertNotIn('fixture-private',saved)

    def test_ambiguous_or_contradictory_full_framing_rejects_before_read(self):
        duplicate=Message();duplicate['Content-Length']='6';duplicate['Content-Length']='6'
        for i,headers in enumerate((duplicate,{'Content-Length':'6','Content-Range':'bytes 0-5/6'},
                {'Content-Length':'6','Transfer-Encoding':'chunked'},
                {'Content-Length':'6','Content-Encoding':'gzip'})):
            folder=self.root/str(i);folder.mkdir();ledger=accounting.CumulativeLedger(folder)
            response=Response(200,headers,b'abcdef')
            with self.assertRaises(transport.cms.GateClosed):
                transport.transfer_blob(folder,self.item(),ledger,time.time()+10,Tokens(),
                    open_response=lambda *_a,**_k:response,chunk=3)
            self.assertEqual(response.reads,0)

    def test_imported_metadata_cap_rejects_registry_token_before_request(self):
        ledger=self.ledger(prior_metadata=original.LIMITS['metadata_received_bytes']-65535)
        registry=transport.Registry(transport.cms,ledger,time.time()+10,
            open_response=lambda *_a,**_k:self.fail('No token request permitted'))
        with self.assertRaisesRegex(transport.cms.GateClosed,'metadata cap'):
            registry.token(self.item())
        self.assertEqual(ledger.data['reservations'],[])


if __name__=='__main__':unittest.main()
