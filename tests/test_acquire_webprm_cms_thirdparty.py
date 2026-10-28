"""Offline CMS third-party identity, transport and cumulative authority checks."""
from copy import deepcopy
from email.message import Message
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import acquire_webprm_cms_thirdparty as controller
from scripts import webprm_cms_oci_transport as transport

cms=controller.cms


class Response:
    def __init__(self,status,headers,body):
        self.status=status;self.headers=headers;self.body=io.BytesIO(body);self.reads=0
    def read(self,n):self.reads+=1;return self.body.read(n)
    def close(self):pass


class Tokens:
    def token(self,item):return 'offline-token-never-published'


def item(raw=b'abcdef'):
    checksum=hashlib.sha256(raw).hexdigest()
    return dict(kind='image_blob',filename='oci/blobs/sha256/'+checksum,bytes=len(raw),sha256=checksum,
        raw_diff_id='sha256:'+hashlib.sha256(b'raw').hexdigest(),registry='https://registry-1.docker.io',
        repository=controller.REPOSITORY,url='https://registry-1.docker.io/v2/'+controller.REPOSITORY+'/blobs/sha256:'+checksum)


class ThirdPartyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='cms-thirdparty-',dir='/tmp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        network=patch('socket.socket.connect',side_effect=AssertionError('Offline tests prohibit network'))
        network.start();self.addCleanup(network.stop)

    def ledger(self,**kw):
        return controller.transport.CumulativeLedger(self.root,prior_charged_bytes=controller.PRIOR_CHARGED,
            prior_metadata_bytes=controller.PRIOR_METADATA_KNOWN+controller.PRIOR_METADATA_UNKNOWN_RESERVED,
            prior_responses=controller.PRIOR_RESPONSES,**kw)

    def transfer(self,responses,*,data=b'abcdef',chunk=3,cap=10000000000):
        ledger=self.ledger(cap=cap);requests=[]
        def opened(request,timeout):requests.append(request);return responses.pop(0)
        proof=transport.transfer_blob(self.root,item(data),ledger,time.time()+10,Tokens(),open_response=opened,chunk=chunk)
        return proof,ledger,requests

    def metadata(self):
        _,paths=controller.metadata_source();(self.root/'metadata').mkdir()
        for name,path in paths.items():(self.root/'metadata'/(name+'.json')).write_bytes(path.read_bytes())

    def owned(self):
        owner=controller.owner
        values=dict(PROFILE=controller.PROFILE,RESOURCES=controller.RESOURCES,MAX_STARTUPS=2,SOURCES=controller.SOURCES,
            SCRIPT=Path(controller.__file__),prior_evidence=controller.prior_evidence,budget=controller.budget,approved=controller.approved)
        state=patch.multiple(owner,**values);state.start();self.addCleanup(state.stop)

    def prepared(self):
        self.owned();root=self.root/'package';p=controller.prepare(root)
        return root,p

    def approve_fixture(self,root,p):
        cms.put(root/'approval.json',dict(status='approved',proposal_sha256=cms.digest(p),exact_user_approval='synthetic offline fixture',
            third_party_unverified_state_source_accepted=True,one_additional_network_campaign_approved=True,
            two_startup_allocations_within_remaining_time_approved=True),replace=True)

    def test_saved_upstream_metadata_binds_all_31_descriptors_and_preserves_disclosure(self):
        self.metadata();rows,unique,manifest,config=controller.inputs(self.root)
        self.assertEqual(len(rows),31);self.assertEqual(sum(x['bytes'] for x in rows),2156742179)
        self.assertEqual([x['raw_diff_id'] for x in rows],config['rootfs']['diff_ids'])
        self.assertEqual([x['sha256'] for x in rows],[x['digest'][7:] for x in manifest['layers']])
        self.assertEqual(len(unique),len({x['sha256'] for x in rows}))
        (self.root/'metadata/config.json').write_bytes(b'{}')
        with self.assertRaisesRegex(cms.GateClosed,'content changed'):controller.inputs(self.root)

    def test_exact_ranges_and_cumulative_charges(self):
        proof,ledger,requests=self.transfer([Response(206,{'Content-Length':'3','Content-Range':'bytes 0-2/6'},b'abc'),
            Response(206,{'Content-Length':'3','Content-Range':'bytes 3-5/6'},b'def')])
        self.assertEqual(proof['sha256'],item()['sha256']);self.assertEqual(len(requests),2)
        self.assertEqual(ledger.data['charged_bytes']+ledger.data['prior_charged_bytes'],1326678+2*65536+6)
        self.assertEqual(ledger.data['prior_metadata_bytes'],147030);self.assertEqual(ledger.data['prior_responses'],18)

    def test_full_200_at_zero_is_accepted_with_total_reservation(self):
        proof,ledger,requests=self.transfer([Response(200,{'Content-Length':'6'},b'abcdef')])
        self.assertEqual(proof['bytes'],6);self.assertEqual(len(requests),1)
        self.assertEqual(ledger.data['reservations'][0]['reserved_bytes'],6)
        self.assertEqual((self.root/item()['filename']).stat().st_mode&0o777,0o600)

    def test_insufficient_full_reservation_fails_before_body_read(self):
        response=Response(200,{'Content-Length':'6'},b'abcdef')
        with self.assertRaisesRegex(cms.GateClosed,'Full-body reservation'):
            self.transfer([response],cap=controller.PRIOR_CHARGED+65536+3)
        self.assertEqual(response.reads,0)

    def test_redirect_strips_authorization_and_accounts_zero_body(self):
        responses=[Response(307,{'Content-Length':'0','Location':'https://production.cloudflare.docker.com/content'},b''),
            Response(200,{'Content-Length':'6'},b'abcdef')]
        _,ledger,requests=self.transfer(responses)
        self.assertIsNotNone(requests[0].get_header('Authorization'));self.assertIsNone(requests[1].get_header('Authorization'))
        self.assertEqual(ledger.data['charged_bytes'],2*65536+6)

    def test_foreign_redirect_has_no_followup_request(self):
        response=Response(307,{'Content-Length':'0','Location':'https://foreign.example/blob'},b'')
        with self.assertRaisesRegex(controller.registry.Closed,'delivery host'):self.transfer([response])
        self.assertEqual(response.reads,0)

    def test_duplicate_framing_and_full_after_nonzero_offset_are_rejected(self):
        headers=Message();headers['Content-Length']='6';headers['Content-Length']='6'
        response=Response(200,headers,b'abcdef')
        with self.assertRaisesRegex(cms.GateClosed,'Ambiguous'):self.transfer([response])
        self.assertEqual(response.reads,0)
        other=self.root/'other';other.mkdir();ledger=controller.transport.CumulativeLedger(other)
        responses=[Response(206,{'Content-Length':'3','Content-Range':'bytes 0-2/6'},b'abc'),Response(200,{'Content-Length':'6'},b'abcdef')]
        with self.assertRaisesRegex(cms.GateClosed,'range/length'):
            transport.transfer_blob(other,item(),ledger,time.time()+10,Tokens(),open_response=lambda *a,**k:responses.pop(0),chunk=3)

    def test_corruption_is_retained_but_never_verified(self):
        with self.assertRaisesRegex(cms.GateClosed,'SHA256 differs'):
            self.transfer([Response(200,{'Content-Length':'6'},b'abcdeg')])
        self.assertEqual((self.root/item()['filename']).read_bytes(),b'abcdeg')

    def test_unknown_redirect_body_remains_fully_charged(self):
        _,ledger,_=self.transfer([Response(307,{'Location':'https://production.cloudflare.docker.com/content'},b''),
            Response(200,{'Content-Length':'6'},b'abcdef')])
        self.assertEqual(ledger.data['charged_bytes'],2*65536+3+6)
        self.assertEqual(ledger.data['reservations'][0]['state'],'uncertain')

    def test_raw_diff_id_and_expansion_cap(self):
        raw=b'raw tar bytes';compressed=gzip.compress(raw,mtime=0);v=item(compressed)
        v['raw_diff_id']='sha256:'+hashlib.sha256(raw).hexdigest();path=self.root/v['filename'];path.parent.mkdir(parents=True);path.write_bytes(compressed)
        proof=transport.verify_raw_layers(self.root,[v],time.time()+10,maximum=len(raw))
        self.assertEqual(proof['total_raw_bytes'],len(raw));self.assertFalse(proof['extraction_performed'])
        with self.assertRaisesRegex(cms.GateClosed,'byte cap'):transport.verify_raw_layers(self.root,[v],time.time()+10,maximum=len(raw)-1)
        v['raw_diff_id']='sha256:'+'0'*64
        with self.assertRaisesRegex(cms.GateClosed,'diff ID'):transport.verify_raw_layers(self.root,[v],time.time()+10)

    def test_metadata_room_is_checked_before_token_request(self):
        ledger=self.ledger(metadata_cap=147030+65535)
        client=transport.Registry(cms,ledger,time.time()+10,open_response=lambda *a,**k:self.fail('No request permitted'))
        with self.assertRaisesRegex(cms.GateClosed,'token metadata cap'):client.token(item())
        self.assertEqual(ledger.data['reservations'],[])

    def test_predecessor_time_network_and_unknown_metadata_reservations_cannot_reset(self):
        rows=controller.prior_evidence();self.assertEqual(sum(x['elapsed_seconds'] for x in rows),22)
        self.assertEqual(sum(x.get('charged_download_bytes',0) for x in rows),1326678)
        self.assertEqual(controller.RESOURCES['remaining_total_seconds'],3578)
        self.assertEqual(controller.LIMITS['temporary_bytes']+controller.PRIOR_TEMPORARY_RESERVED,512*2**20)

    def test_pending_approval_never_produces_a_launch_or_network_claim(self):
        root,p=self.prepared()
        with self.assertRaisesRegex(cms.GateClosed,'approval'):controller.owner.next_request(root)
        self.assertFalse((root/'transfer-claim.json').exists());self.assertEqual(list(root.glob('startup-*')),[])
        self.assertFalse(p['original_state_identity_verified']);self.assertFalse(p['maintainer_uploaded'])

    def test_all_three_source_transfer_startup_acceptances_are_required(self):
        root,p=self.prepared();self.approve_fixture(root,p)
        approved=cms.read(root/'approval.json')
        for field in ('third_party_unverified_state_source_accepted','one_additional_network_campaign_approved','two_startup_allocations_within_remaining_time_approved'):
            rejected=dict(approved);rejected[field]=False;cms.put(root/'approval.json',rejected,replace=True)
            with self.assertRaisesRegex(cms.GateClosed,'approval'):controller.approved(root)
        cms.put(root/'approval.json',approved,replace=True)
        request=controller.owner.next_request(root)
        self.assertEqual(request['requested_seconds'],3540);self.assertEqual(request['budget']['remaining_seconds'],3578)

    def test_changed_initial_transport_or_immutable_controller_cannot_launch(self):
        root,p=self.prepared();self.approve_fixture(root,p)
        for name in ('scripts/webprm_cms_oci_transport.py','scripts/acquire_webprm_cms_thirdparty.py'):
            hashes=dict(p['source_sha256']);hashes[str(controller.REPO/name)]='0'*64
            with patch.object(controller.owner,'source_hashes',return_value=hashes),self.assertRaises(cms.GateClosed):
                controller.owner.next_request(root)

    def closed_startup(self,root,p,number,elapsed,source=None):
        attempt=root/f'startup-{number:02d}';attempt.mkdir()
        source=source or p['source_sha256']
        cms.put(attempt/'launch.json',dict(requested_seconds=3540,source_sha256=source))
        job=str(123400+number)
        cms.put(attempt/'run-claim.json',dict(job_id=job))
        snapshot=deepcopy(cms.read(controller.OLD/'startup-01/terminal.json')['scheduler_snapshot'])
        row=snapshot['jobs'][0];row['job_id']=int(job);row['end_time']['number']=row['start_time']['number']+elapsed
        terminal=controller.owner.terminal_accounting(snapshot,job,3540)
        cms.put(attempt/'terminal.json',terminal)
        cms.put(attempt/'failure.json',dict(stage='startup_before_transfer',status='failed',error_type='SyntheticStartupFailure'))
        return attempt

    def test_manual_startup_successor_needs_changed_transport_and_bound_tests(self):
        root,p=self.prepared();self.approve_fixture(root,p);attempt=self.closed_startup(root,p,1,6)
        changed=dict(p['source_sha256']);changed[str(controller.REPO/'scripts/webprm_cms_oci_transport.py')]='0'*64
        log=root/'offline-tests.log';log.write_text('Synthetic offline test result\n')
        review=root/'repair.json';cms.put(review,dict(previous_startup=str(attempt),previous_failure_sha256=cms.sha(attempt/'failure.json'),
            source_sha256=changed,explanation='Synthetic transport import fix only',offline_tests_passed=True,
            test_log_path=str(log),test_log_sha256=cms.sha(log)))
        with patch.object(controller.owner,'source_hashes',return_value=changed):
            with self.assertRaisesRegex(cms.GateClosed,'repair'):controller.owner.next_request(root)
            request=controller.owner.next_request(root,review)
            self.assertEqual(request['budget']['charged_seconds'],28);self.assertEqual(request['requested_seconds'],3540)
            self.assertEqual(request['budget']['remaining_seconds'],3572)
            log.write_text('Changed test evidence')
            with self.assertRaisesRegex(cms.GateClosed,'repair'):controller.owner.next_request(root,review)

    def test_two_closed_startups_exhaust_allowance_without_resetting_time(self):
        root,p=self.prepared();self.approve_fixture(root,p)
        self.closed_startup(root,p,1,6);self.closed_startup(root,p,2,7)
        self.assertEqual(controller.budget(root)['charged_seconds'],35)
        with self.assertRaisesRegex(cms.GateClosed,'attempt cap'):controller.owner.next_request(root)

    def test_spent_transfer_and_unsettled_allocation_cannot_restart(self):
        root,p=self.prepared();self.approve_fixture(root,p)
        cms.put(root/'transfer-claim.json',dict(job_id='synthetic'))
        with self.assertRaisesRegex(cms.GateClosed,'already claimed'):controller.owner.next_request(root)
        # A separate root permits checking an unresolved reservation independently.
        root2=self.root/'package2';p2=controller.prepare(root2);self.approve_fixture(root2,p2)
        attempt=root2/'startup-01';attempt.mkdir();cms.put(attempt/'launch.json',dict(requested_seconds=3540))
        with self.assertRaisesRegex(cms.GateClosed,'unresolved'):controller.owner.next_request(root2)


if __name__=='__main__':unittest.main()
