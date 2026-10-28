"""Offline regression tests for the observed CMS confirmation failure."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import webprm_cms_confirmation_transport as transport
from scripts import resume_webprm_cms_confirmation as controller


def page(**changes):
    values=dict(id=transport.DRIVE_ID,export='download',confirm='t',uuid='01234567-89ab-cdef-0123-456789abcdef')
    values.update(changes)
    return ('<html><form id="download-form" action="https://drive.usercontent.google.com/download" method="get">'
        +'<input type="submit" value="Download anyway">'
        +''.join(f'<input type="hidden" name="{k}" value="{v}">' for k,v in values.items())
        +'</form></html>').encode()


class Response:
    def __init__(self,status,headers,body):
        self.status=status;self.headers=headers;self.body=io.BytesIO(body);self.reads=0
    def read(self,n):self.reads+=1;return self.body.read(n)
    def close(self):pass


def archive(status,start,end,size,body,**headers):
    values={'Content-Length':str(end-start+1),'Content-Disposition':'attachment; filename='+transport.ARCHIVE_NAME,'ETag':'"original"'}
    if status==206:values['Content-Range']=f'bytes {start}-{end}/{size}'
    values.update(headers)
    return Response(status,values,body)


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='cms-confirmation-',dir='/tmp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        network=patch('socket.socket.connect',side_effect=AssertionError('Offline confirmation tests may not use network'))
        network.start();self.addCleanup(network.stop)

    def run_transfer(self,responses,*,size=6,chunk=3,cap=10**8):
        ledger=transport.CumulativeLedger(self.root,prior_charged_bytes=133549,prior_metadata_bytes=2477,
            prior_responses=2,cap=cap)
        requests=[]
        def request(req,timeout):requests.append(req);return responses.pop(0)
        result=transport.download(self.root,ledger,time.time()+10,open_response=request,size=size,chunk=chunk)
        return result,ledger,requests

    def test_actual_saved_google_confirmation_is_recognized_offline(self):
        path=controller.OLD/'startup-01/attempt-355870/response-0001-metadata.bin'
        url=transport.confirmation_url(path.read_bytes())
        self.assertTrue(url.startswith('https://drive.usercontent.google.com/download?'))
        self.assertIn('id='+transport.DRIVE_ID,url)

    def test_confirmation_rejects_other_id_host_duplicate_extra_or_unbounded_fields(self):
        invalid=[page(id='other'),page(confirm='yes'),page(uuid='not-a-uuid'),page(extra='bad'),
            page().replace(b'drive.usercontent.google.com',b'foreign.example'),
            page().replace(b'</form>',b'<input type="hidden" name="id" value="x"></form>'),
            page()+page(),b'x'*(transport.METADATA_RESPONSE_CAP+1)]
        for raw in invalid:
            with self.subTest(raw=hashlib.sha256(raw).hexdigest()),self.assertRaises(transport.GateClosed):
                transport.confirmation_url(raw)

    def test_confirmation_then_exact_ranges_preserves_entity_and_cumulative_charges(self):
        raw=page();responses=[Response(200,{'Content-Length':str(len(raw))},raw),
            archive(206,0,2,6,b'abc'),archive(206,3,5,6,b'def')]
        result,ledger,requests=self.run_transfer(responses)
        self.assertEqual(result['archive_sha256'],hashlib.sha256(b'abcdef').hexdigest())
        self.assertEqual(requests[-1].get_header('If-match'),'"original"')
        self.assertEqual(ledger.data['prior_charged_bytes']+ledger.data['charged_bytes'],133549+3*65536+len(raw)+6)
        self.assertEqual(ledger.data['prior_metadata_bytes'],2477)
        self.assertTrue((self.root/'confirmation-0000.json').exists())

    def test_confirmation_then_exact_full_200_body_is_one_transfer(self):
        raw=page();responses=[Response(200,{'Content-Length':str(len(raw))},raw),archive(200,0,5,6,b'abcdef')]
        result,ledger,requests=self.run_transfer(responses)
        self.assertEqual(result['bytes'],6);self.assertEqual(len(requests),2)
        self.assertEqual((self.root/transport.ARCHIVE_NAME).read_bytes(),b'abcdef')
        self.assertEqual(ledger.data['charged_bytes'],2*65536+len(raw)+6)

    def test_full_body_reserves_total_before_any_body_read(self):
        response=archive(200,0,transport.ARCHIVE_BYTES-1,transport.ARCHIVE_BYTES,b'')
        with self.assertRaisesRegex(transport.GateClosed,'reservation'):
            self.run_transfer([response],size=transport.ARCHIVE_BYTES,chunk=transport.CHUNK,cap=10**8)
        self.assertEqual(response.reads,0)

    def test_full_body_missing_or_changed_filename_is_rejected_without_read(self):
        response=archive(200,0,5,6,b'abcdef',**{'Content-Disposition':'attachment; filename=other.tar'})
        with self.assertRaises(transport.GateClosed):self.run_transfer([response])
        self.assertEqual(response.reads,0)

    def test_second_confirmation_is_not_retried(self):
        raw=page();responses=[Response(200,{'Content-Length':str(len(raw))},raw) for _ in range(2)]
        with self.assertRaisesRegex(transport.GateClosed,'no retry'):self.run_transfer(responses)
        self.assertEqual(responses,[])

    def test_cumulative_prior_headers_cannot_be_reset(self):
        ledger=transport.CumulativeLedger(self.root,prior_charged_bytes=133549,cap=133549+65536)
        with self.assertRaises(transport.GateClosed):ledger.reserve(1,transport.SOURCE_URL,0,0)
        self.assertEqual(ledger.data['reservations'],[])

    def test_cumulative_metadata_and_response_caps_include_prior_attempt(self):
        ledger=transport.CumulativeLedger(self.root,prior_metadata_bytes=2477,metadata_cap=2478,prior_responses=620)
        self.assertEqual(ledger.metadata_cap,1)
        with self.assertRaisesRegex(transport.GateClosed,'response cap'):ledger.reserve(1,transport.SOURCE_URL,0,0)

    def test_predecessor_terminal_and_charge_evidence_is_bound(self):
        rows=controller.prior_evidence()
        self.assertEqual(sum(row['elapsed_seconds'] for row in rows),15)
        self.assertEqual(rows[-1]['charged_download_bytes'],133549)
        self.assertEqual(controller.RESOURCES['remaining_total_seconds'],3585)
        self.assertEqual(controller.RESOURCES['maximum_additional_startup_allocations'],2)
        self.assertEqual(controller.LIMITS['temporary_bytes']+controller.PRIOR_TEMPORARY_RESERVATION,512*2**20)

    def test_pending_proposal_cannot_launch(self):
        p=dict(resource_profile=controller.PROFILE,campaign_root=str(self.root),resources=controller.RESOURCES,
            limits=controller.LIMITS,predecessors=controller.prior_evidence(),original=controller.cms.read(controller.OLD/'proposal.json')['original'])
        controller.cms.put(self.root/'proposal.json',p)
        controller.cms.put(self.root/'approval.json',dict(status='pending',proposal_sha256=controller.cms.digest(p)))
        with self.assertRaisesRegex(transport.GateClosed,'approval'):controller.approved(self.root)


if __name__=='__main__':unittest.main()
