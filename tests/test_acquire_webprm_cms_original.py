"""Offline acquisition gates, scripted HTTP accounting and real child ownership."""
import hashlib
from copy import deepcopy
from datetime import datetime, timezone
import email.message
import io
import json
import os
from pathlib import Path
import signal
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import acquire_webprm_cms_original as cms


class Response:
    def __init__(self, status, headers, body):
        self.status, self.headers = status, headers
        self.body, self.closed = io.BytesIO(body), False

    def read(self, count):
        assert 0 <= count <= cms.BLOCK
        return self.body.read(count)

    def close(self):
        self.closed = True


def archive_response(start, end, size, data, *, etag='"original"', **changes):
    headers = {'Content-Length':str(end-start+1), 'Content-Range':f'bytes {start}-{end}/{size}',
               'Content-Disposition':'attachment; filename='+cms.ARCHIVE_NAME, 'ETag':etag}
    headers.update(changes)
    return Response(206, headers, data)


def detached_hanging_tree(root, scratch, deadline):
    os.setsid()
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    pid = os.fork()
    if pid == 0:
        os.setsid()
        (root/'grandchild-pid').write_text(str(os.getpid()))
        while True:
            time.sleep(.05)
    (root/'parent-pid').write_text(str(os.getpid()))
    while True:
        time.sleep(.05)


def completed_child(root,scratch,deadline):
    os.setsid()
    (root/'child-complete').write_text('complete')


def scheduler_snapshot(start=1791746000, end=1791749600):
    finite = lambda value: dict(set=True, infinite=False, number=value)
    return dict(errors=[], jobs=[dict(job_id=123, job_state=['RUNNING'], user_id=os.getuid(),
        account='zixianma', partition='gpu-h200', qos='normal', node_count=finite(1),
        cpus=finite(2), memory_per_node=finite(8192), tres_alloc_str='cpu=2,mem=8G,node=1',
        restart_cnt=0, start_time=finite(start), end_time=finite(end), time_limit=finite(60))],
        meta=dict(plugin=dict(data_parser='data_parser/v0.0.44')))


class CmsAcquisitionTests(unittest.TestCase):
    def test_scheduler_source_is_preserved_when_preworker_admission_fails(self):
        snapshot = scheduler_snapshot()
        snapshot['jobs'][0]['start_time']['set'] = False
        raw = json.dumps(snapshot)
        with patch.object(cms.subprocess, 'check_output', return_value=raw), self.assertRaises(cms.GateClosed):
            cms.saved_allocation_context(self.root, '123', 'nonce', {'nonce':'nonce'}, cms.REPLACEMENT_RESOURCES)
        saved = cms.read(self.root/'scheduler-before-worker.json')
        self.assertEqual(saved['raw_json'], raw)
        self.assertEqual(saved['sha256'], hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(saved['argv'], ['scontrol', '--json', 'show', 'job', '123'])
        self.assertEqual(list(self.root.glob('attempt-*')), [])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cms-acquisition-unit-',dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.attempt = self.root/'attempt'; self.attempt.mkdir()
        self.scratch = self.root/'scratch'; self.scratch.mkdir()
        self.deadline = time.time()+20

    def transfer(self,responses,*,size=6,chunk=3,cap=10**8,metadata_cap=2**20):
        ledger = cms.TransferLedger(self.attempt,cap=cap,metadata_cap=metadata_cap)
        requests = []
        def scripted(request,timeout):
            self.assertGreater(timeout,0)
            self.assertLessEqual(timeout,10)
            requests.append(request)
            response=responses.pop(0)
            if isinstance(response,BaseException):
                raise response
            return response
        return ledger,requests,lambda:cms.download(self.attempt,ledger,self.deadline,
            open_response=scripted,size=size,chunk=chunk)

    def test_completed_ranges_bind_original_entity_and_hash_every_byte(self):
        responses = [archive_response(0,2,6,b'abc'),archive_response(3,5,6,b'def')]
        ledger,requests,run = self.transfer(list(responses))
        result=run()
        self.assertEqual(result['archive_sha256'],hashlib.sha256(b'abcdef').hexdigest())
        self.assertEqual((self.attempt/cms.ARCHIVE_NAME).read_bytes(),b'abcdef')
        self.assertEqual(requests[1].get_header('If-match'),'"original"')
        self.assertEqual(ledger.data['known_received_bytes'],6)
        self.assertEqual(ledger.data['charged_bytes'],6+2*cms.HEADER_CAP)
        self.assertTrue(all(r.closed for r in responses))
        self.assertFalse(result['historical_checksum_independently_published'])
        self.assertEqual((self.attempt/cms.ARCHIVE_NAME).stat().st_mode&0o777,0o600)

    def test_redirect_body_is_charged_and_private_signed_location_is_not_logged(self):
        signed='https://drive.usercontent.google.com/download?secret=signed-query'
        redirect=Response(302,{'Content-Length':'4','Location':signed},b'move')
        ledger,requests,run=self.transfer([redirect,archive_response(0,5,6,b'abcdef')],chunk=6)
        run()
        self.assertEqual(ledger.data['known_received_bytes'],10)
        self.assertEqual(ledger.data['metadata_received_bytes'],4)
        self.assertEqual(ledger.data['charged_bytes'],10+2*cms.HEADER_CAP)
        self.assertEqual(requests[1].full_url,signed)
        self.assertNotIn('signed-query',(self.attempt/'transfer-ledger.json').read_text())
        self.assertEqual((self.attempt/'response-0000-metadata.bin').read_bytes(),b'move')

    def test_quota_html_counts_bytes_then_stops_without_retry(self):
        quota=Response(403,{'Content-Length':'5'},b'quota')
        ledger,requests,run=self.transfer([quota])
        with self.assertRaisesRegex(cms.GateClosed,'quota'):
            run()
        self.assertEqual(len(requests),1)
        self.assertEqual(ledger.data['charged_bytes'],5+cms.HEADER_CAP)
        self.assertEqual(ledger.data['metadata_received_bytes'],5)
        self.assertTrue((self.attempt/'archive-attempt-001.partial').exists())
        self.assertFalse((self.attempt/cms.ARCHIVE_NAME).exists())

    def test_unknown_length_metadata_is_bounded_and_retains_full_reservation(self):
        ledger,requests,run=self.transfer([Response(200,{},b'x'*100)],metadata_cap=8)
        with self.assertRaises(cms.GateClosed):
            run()
        self.assertEqual(ledger.data['known_received_bytes'],8)
        self.assertEqual(ledger.data['metadata_received_bytes'],8)
        self.assertEqual(ledger.data['charged_bytes'],cms.METADATA_RESPONSE_CAP+cms.HEADER_CAP)
        self.assertEqual(ledger.data['reservations'][0]['state'],'uncertain')

    def test_partial_or_transport_failure_keeps_charge_and_prevents_resume(self):
        ledger,requests,run=self.transfer([archive_response(0,5,6,b'ab')],chunk=6)
        with self.assertRaisesRegex(cms.GateClosed,'Partial'):
            run()
        self.assertEqual(ledger.data['known_received_bytes'],2)
        self.assertEqual(ledger.data['charged_bytes'],cms.METADATA_RESPONSE_CAP+cms.HEADER_CAP)
        self.assertEqual((self.attempt/'archive-attempt-001.partial').read_bytes(),b'ab')
        with self.assertRaisesRegex(cms.GateClosed,'no resume'):
            cms.TransferLedger(self.attempt)

    def test_pre_request_durable_reservation_survives_transport_exception(self):
        ledger,requests,run=self.transfer([OSError('network disconnected')])
        with self.assertRaises(OSError):
            run()
        persisted=cms.read(self.attempt/'transfer-ledger.json')
        self.assertEqual(persisted['charged_bytes'],cms.METADATA_RESPONSE_CAP+cms.HEADER_CAP)
        self.assertEqual(persisted['known_received_bytes'],0)
        self.assertEqual(persisted['reservations'][0]['state'],'uncertain')
        self.assertEqual(len(requests),1)

    def test_cumulative_cap_stops_before_next_request(self):
        cap=2*cms.HEADER_CAP+cms.METADATA_RESPONSE_CAP+2
        ledger,requests,run=self.transfer([archive_response(0,2,6,b'abc')],cap=cap)
        with self.assertRaisesRegex(cms.GateClosed,'cap'):
            run()
        self.assertEqual(len(requests),1)
        self.assertEqual(ledger.data['known_received_bytes'],3)
        self.assertEqual(ledger.data['charged_bytes'],3+cms.HEADER_CAP)

    def test_identity_failures_do_not_consume_or_execute_bad_payload(self):
        changes=[{'ETag':'W/"weak"'}, {'Content-Disposition':'attachment; filename=substitute.tar'},
                 {'Content-Range':'bytes 0-2/7'}, {'Content-Encoding':'gzip'}, {'Transfer-Encoding':'chunked'}]
        for i,change in enumerate(changes):
            with self.subTest(change=change):
                folder=self.root/f'bad-{i}';folder.mkdir()
                ledger=cms.TransferLedger(folder)
                response=archive_response(0,2,6,b'bad',**change)
                with self.assertRaises(cms.GateClosed):
                    cms.download(folder,ledger,self.deadline,open_response=lambda *_a,**_k:response,size=6,chunk=3)
                self.assertEqual(ledger.data['known_received_bytes'],0)
                self.assertEqual((folder/'archive-attempt-001.partial').read_bytes(),b'')

    def test_changed_entity_rejects_second_range_without_replacing_partial(self):
        ledger,requests,run=self.transfer([archive_response(0,2,6,b'abc'),
            archive_response(3,5,6,b'def',etag='"changed"')])
        with self.assertRaisesRegex(cms.GateClosed,'entity'):
            run()
        self.assertEqual((self.attempt/'archive-attempt-001.partial').read_bytes(),b'abc')
        self.assertEqual(len(requests),2)

    def test_redirect_host_count_and_missing_location_gates(self):
        for value in ('http://drive.google.com/','https://evil.example/x',
                      'https://drive.google.com@evil.example/','https://drive.google.com:444/x',
                      'https://drive.google.com/#fragment'):
            with self.assertRaises(cms.GateClosed):
                cms.allowed_url(value)
        redirects=[Response(302,{'Content-Length':'0','Location':cms.SOURCE_URL},b'') for _ in range(4)]
        ledger,requests,run=self.transfer(redirects)
        with self.assertRaisesRegex(cms.GateClosed,'redirect bound'):
            run()
        self.assertEqual(len(requests),4)
        self.assertEqual(ledger.data['charged_bytes'],4*cms.HEADER_CAP)

    def test_deadline_blocks_request_and_header_reader_bounds_interim_responses(self):
        ledger,requests,run=self.transfer([])
        self.deadline=time.time()-1
        with self.assertRaisesRegex(cms.GateClosed,'deadline'):
            run()
        self.assertEqual(requests,[])
        source=io.BytesIO(b'HTTP/1.1 100 Continue\r\n\r\n'*(cms.HEADER_CAP//20))
        reader=cms.HeaderReader(source)
        with self.assertRaisesRegex(cms.GateClosed,'header'):
            while True:
                reader.readline(65537)
        self.assertLessEqual(source.tell(),cms.HEADER_CAP)

    def test_actual_http_parser_handles_bounded_headers_and_rejects_unbounded_interims(self):
        class Socket:
            def __init__(self,body):self.stream=io.BytesIO(body)
            def makefile(self,*_a,**_k):return self.stream
        wire=b'HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 206 Partial Content\r\nContent-Length: 3\r\n\r\nabc'
        response=cms.BoundedHTTPResponse(Socket(wire));response.begin()
        self.assertEqual(response.status,206)
        self.assertEqual(response.read(3),b'abc')
        socket=Socket((b'HTTP/1.1 100 Continue\r\nX-Padding: '+b'x'*700+b'\r\n\r\n')*100)
        response=cms.BoundedHTTPResponse(socket)
        with self.assertRaisesRegex(cms.GateClosed,'header'):
            response.begin()
        self.assertLessEqual(socket.stream.tell(),cms.HEADER_CAP)

    def test_duplicate_headers_and_missing_redirect_target_stop_cleanly(self):
        response=archive_response(0,5,6,b'abcdef')
        headers=email.message.Message()
        for key,value in response.headers.items():headers[key]=value
        headers['Content-Length']='7';response.headers=headers
        ledger,requests,run=self.transfer([response],chunk=6)
        with self.assertRaisesRegex(cms.GateClosed,'duplicate'):
            run()
        self.assertEqual(ledger.data['known_received_bytes'],0)
        other=self.root/'redirect-attempt';other.mkdir()
        ledger=cms.TransferLedger(other)
        with self.assertRaisesRegex(cms.GateClosed,'omitted'):
            cms.download(other,ledger,self.deadline,size=6,chunk=6,
                open_response=lambda *_a,**_k:Response(302,{'Content-Length':'0'},b''))

    def test_pending_exact_approval_blocks_scheduler_before_launch_claim(self):
        folder=self.root/'proposal';proposal=cms.prepare(folder)
        self.assertIsNone(proposal['original']['archive_sha256'])
        self.assertFalse(proposal['approval_received'])
        with patch.object(cms.os,'execvpe') as execute, patch.object(cms.subprocess,'check_output') as subprocess:
            with self.assertRaisesRegex(cms.GateClosed,'not approved'):
                cms.launch(folder)
            with self.assertRaisesRegex(cms.GateClosed,'not approved'):
                cms.run(folder,'unused')
            execute.assert_not_called();subprocess.assert_not_called()
        self.assertFalse((folder/'launch.json').exists())
        self.assertFalse((folder/'run-claim.json').exists())

    def test_exact_approval_source_freeze_and_single_allocation_claim(self):
        folder=self.root/'proposal';proposal=cms.prepare(folder)
        cms.put(folder/'approval.json',dict(status='approved',proposal_sha256=cms.digest(proposal),
            exact_user_approval='Offline test approval; not a real authorization.'),replace=True)
        self.assertEqual(cms.approved(folder),proposal)
        with patch.dict(cms.RESOURCES,cpus=3):
            with self.assertRaises(cms.GateClosed):cms.approved(folder)
        with patch.object(cms,'sha',return_value='changed'):
            with self.assertRaisesRegex(cms.GateClosed,'source changed'):cms.approved(folder)
        cms.put(folder/'launch.json',{'status':'previous_failed_submission'})
        with patch.object(cms.os,'execvpe') as execute:
            with self.assertRaisesRegex(cms.GateClosed,'already claimed'):cms.launch(folder)
            execute.assert_not_called()

    def test_actual_allocation_identity_and_original_clock_are_required(self):
        original=scheduler_snapshot(); text=json.dumps(original); now=1791746030
        context=cms.allocation_context(text,'123','nonce',{'nonce':'nonce'},now)
        self.assertEqual(context['allocation_deadline']-context['started_epoch'],3600)
        self.assertEqual(context['clock_source'],'scontrol_json_unix_epochs')
        self.assertEqual(context['scheduler_time_limit_seconds'],3600)
        for key,value in (('cpus',dict(set=True,infinite=False,number=4)),('qos','other'),
                          ('tres_alloc_str','cpu=2,mem=8G,node=1,gres/gpu=1'),
                          ('account','other'),('restart_cnt',1),('job_state',['FAILED'])):
            wrong=deepcopy(original);wrong['jobs'][0][key]=value
            with self.assertRaises(cms.GateClosed):
                cms.allocation_context(json.dumps(wrong),'123','nonce',{'nonce':'nonce'},now)
        with self.assertRaises(cms.GateClosed):
            cms.allocation_context(text,'123','wrong',{'nonce':'nonce'},now)
        with self.assertRaises(cms.GateClosed):
            cms.allocation_context(text,'123','nonce',{'nonce':'nonce'},now+3500)

    @unittest.skipUnless(hasattr(time,'tzset'),'Requires POSIX timezone switching')
    def test_scheduler_epochs_are_independent_of_python_timezone_and_dst(self):
        # The second interval crosses the Pacific autumn repeated hour. Text
        # without an offset cannot establish that its elapsed time is one hour.
        starts=[1791746000,int(datetime(2026,11,1,8,30,tzinfo=timezone.utc).timestamp())]
        try:
            for start in starts:
                expected=None
                for zone in ('UTC','America/Los_Angeles','Asia/Tokyo'):
                    with patch.dict(os.environ,{'TZ':zone}):
                        time.tzset()
                        context=cms.allocation_context(json.dumps(scheduler_snapshot(start,start+3600)),
                            '123','nonce',{'nonce':'nonce'},start+30)
                        if expected is None:expected=context
                        self.assertEqual(context,expected)
                        self.assertEqual(context['work_deadline'],start+3510)
        finally:
            time.tzset()

    def test_scheduler_clock_schema_unset_infinite_and_overbudget_fail_closed(self):
        original=scheduler_snapshot()
        for key,value in (('start_time',{'set':False,'infinite':False,'number':1791746000}),
                          ('end_time',{'set':True,'infinite':True,'number':1791749600}),
                          ('start_time',{'set':True,'infinite':False,'number':True}),
                          ('end_time',1791749600),
                          ('time_limit',{'set':True,'infinite':False,'number':61}),
                          ('time_limit',{'set':True,'infinite':False,'number':0})):
            wrong=deepcopy(original);wrong['jobs'][0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(cms.GateClosed):
                cms.allocation_context(json.dumps(wrong),'123','nonce',{'nonce':'nonce'},1791746030)
        for wrong in ('not json',json.dumps(dict(original,jobs=[])),
                      json.dumps(dict(original,jobs=original['jobs']*2)),
                      json.dumps(dict(original,errors=[{'error':'scheduler failure'}]))):
            with self.assertRaises(cms.GateClosed):
                cms.allocation_context(wrong,'123','nonce',{'nonce':'nonce'},1791746030)
        for now in (1791745999,1791749510,float('nan')):
            with self.assertRaises(cms.GateClosed):
                cms.allocation_context(json.dumps(original),'123','nonce',{'nonce':'nonce'},now)

    def test_scheduler_grace_does_not_expand_requested_or_earlier_deadline(self):
        start=1791754191
        for requested_minutes,reported_span,expected_span in ((59,3543,3540),(60,3660,3600),(30,3600,1800),(60,1200,1200)):
            value=scheduler_snapshot(start,start+reported_span)
            value['jobs'][0]['time_limit']['number']=requested_minutes
            with self.subTest(minutes=requested_minutes,reported_span=reported_span):
                context=cms.allocation_context(json.dumps(value),'123','nonce',{'nonce':'nonce'},start+5.75)
                self.assertEqual(context['allocation_deadline'],start+expected_span)
                self.assertEqual(context['scheduler_reported_end_epoch'],start+reported_span)
                self.assertEqual(context['work_deadline'],start+expected_span-90)
                with self.assertRaises(cms.GateClosed):
                    cms.allocation_context(json.dumps(value),'123','nonce',{'nonce':'nonce'},start+expected_span-90)

    def test_replacement_clock_cannot_regain_the_original_hour(self):
        value=scheduler_snapshot(1791746000,1791749540)
        value['jobs'][0]['time_limit']['number']=59
        context=cms.allocation_context(json.dumps(value),'123','nonce',{'nonce':'nonce'},1791746030,
                                       resources=cms.REPLACEMENT_RESOURCES)
        self.assertEqual(context['allocation_deadline']-context['started_epoch'],3540)
        value['jobs'][0]['time_limit']['number']=60
        with self.assertRaises(cms.GateClosed):
            cms.allocation_context(json.dumps(value),'123','nonce',{'nonce':'nonce'},1791746030,
                                   resources=cms.REPLACEMENT_RESOURCES)

    def test_replacement_requires_explicit_exception_and_preserves_prior_claims(self):
        prior=self.root/'prior';original=cms.prepare(prior)
        cms.put(prior/'approval.json',dict(status='approved',proposal_sha256=cms.digest(original),exact_user_approval='fixture'),replace=True)
        cms.put(prior/'launch.json',dict(nonce='prior-nonce',proposal_sha256=cms.digest(original)))
        cms.put(prior/'run-claim.json',dict(nonce='prior-nonce',job_id=cms.PRIOR_JOB))
        for name in cms.SOURCES:
            target=prior/'frozen-source'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(cms.REPO/name,target)
        before={str(p.relative_to(prior)):cms.sha(p) for p in prior.rglob('*') if p.is_file()}
        snapshot=scheduler_snapshot(1791746000,1791746004)
        snapshot['jobs'][0].update(job_id=int(cms.PRIOR_JOB),job_state=['FAILED'])
        fresh=self.root/'replacement'
        with patch.object(cms,'PRIOR_ROOT',prior),patch.object(cms.subprocess,'check_output',return_value=json.dumps(snapshot)):
            proposal=cms.prepare(fresh,replacement=True)
            self.assertEqual(proposal['resources']['total_allocation_seconds_including_retries'],3540)
            self.assertEqual(proposal['cumulative_allocation']['maximum_total_seconds'],3544)
            saved=self.root/'terminal-scheduler.json';cms.put(saved,snapshot)
            with patch.object(cms.subprocess,'check_output',side_effect=AssertionError('Aged-out job must use saved proof')):
                captured=cms.prepare(self.root/'captured-proof',replacement=True,predecessor_snapshot=saved)
            self.assertEqual(captured['predecessor'],proposal['predecessor'])
            auth=dict(status='approved',proposal_sha256=cms.digest(proposal),exact_user_approval='fixture')
            cms.put(fresh/'approval.json',auth,replace=True)
            with self.assertRaisesRegex(cms.GateClosed,'exception'):
                cms.approved(fresh)
            auth['prior_maximum_one_allocation_exception_approved']=True
            cms.put(fresh/'approval.json',auth,replace=True)
            self.assertEqual(cms.approved(fresh),proposal)
            with patch.object(cms.os,'execvpe') as execute,patch.object(cms.os,'chdir'):
                cms.launch(fresh)
                self.assertIn('--time=00:59:00',execute.call_args.args[1])
            with patch.object(cms.os,'execvpe') as execute:
                with self.assertRaisesRegex(cms.GateClosed,'already claimed'):cms.launch(fresh)
                execute.assert_not_called()
            self.assertEqual(before,{str(p.relative_to(prior)):cms.sha(p) for p in prior.rglob('*') if p.is_file()})
            (prior/'attempt-unexpected').mkdir()
            with self.assertRaisesRegex(cms.GateClosed,'zero-transfer'):
                cms.approved(fresh)

    def test_paths_and_cache_roots_do_not_follow_home_or_symlinks(self):
        forbidden=Path('/gpfs/home/zixianma/acquisition-forbidden')
        with self.assertRaises(cms.GateClosed):cms.safe_environment(forbidden)
        link=self.root/'redirect';link.symlink_to(self.attempt,target_is_directory=True)
        with self.assertRaises(cms.GateClosed):cms.put(link/'receipt.json',{})
        original_home=os.environ.get('HOME')
        with patch.dict(os.environ,{'SALLOC_GRES':'gpu:1','SBATCH_CPUS_PER_TASK':'90'}):
            env=cms.safe_environment(self.scratch)
        self.assertEqual(env.get('HOME'),original_home)
        self.assertNotIn('SALLOC_GRES',env);self.assertNotIn('SBATCH_CPUS_PER_TASK',env)
        for key in ('TMPDIR','XDG_CACHE_HOME','PIP_CACHE_DIR'):
            self.assertTrue(Path(env[key]).is_relative_to(self.scratch))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')

    def test_stale_atomic_temporary_and_hardlinks_are_preserved(self):
        target=self.root/'receipt.json'
        original=self.root/'unrelated';original.write_text('untouched')
        temporary=target.with_suffix('.json.tmp')
        os.link(original,temporary)
        with self.assertRaises(FileExistsError):cms.put(target,{'new':True},replace=True)
        self.assertEqual(original.read_text(),'untouched')
        self.assertEqual(temporary.read_text(),'untouched')
        self.assertFalse(target.exists())

    def test_storage_scan_conservatively_counts_atomically_replaced_receipt(self):
        target = self.attempt/'transfer-ledger.json'
        cms.put(target, {'charged_bytes': 1})
        listing = [(str(self.attempt), [], [target.name, target.name+'.tmp'])]
        with patch.object(cms.os, 'walk', return_value=listing):
            self.assertEqual(cms.apparent_bytes(self.attempt),
                             target.stat().st_size + cms.RECEIPT_BYTES)

    def test_storage_scan_does_not_hide_missing_payload_or_unreplaced_temp(self):
        for name in ['blob.partial', 'transfer-ledger.json.tmp']:
            with self.subTest(name=name), patch.object(cms.os, 'walk',
                    return_value=[(str(self.attempt), [], [name])]):
                with self.assertRaises(FileNotFoundError):
                    cms.apparent_bytes(self.attempt)

    def test_storage_scan_rejects_unsafe_or_oversized_atomic_destination(self):
        target = self.attempt/'transfer-ledger.json'
        target.symlink_to(self.root/'elsewhere')
        listing = [(str(self.attempt), [], [target.name+'.tmp'])]
        with patch.object(cms.os, 'walk', return_value=listing):
            with self.assertRaises(cms.GateClosed):cms.apparent_bytes(self.attempt)
        other = self.attempt/'oversized.json'
        with other.open('wb') as out:out.truncate(cms.RECEIPT_BYTES+1)
        with patch.object(cms.os, 'walk', return_value=[(str(self.attempt), [], [other.name+'.tmp'])]):
            with self.assertRaisesRegex(cms.GateClosed,'bounded regular'):
                cms.apparent_bytes(self.attempt)

    def test_owner_reaps_detached_term_ignoring_grandchild_on_deadline(self):
        with self.assertRaisesRegex(cms.GateClosed,'owned children closed'):
            cms.own_worker(self.attempt,self.scratch,time.time()+.6,target=detached_hanging_tree)
        closure=cms.read(self.attempt/'closure.json')
        self.assertEqual(closure['remaining_owned_processes'],{})
        self.assertEqual(closure['child_exitcode'],-signal.SIGKILL)
        for name in ('parent-pid','grandchild-pid'):
            pid=int((self.attempt/name).read_text())
            self.assertIn(pid,closure['owned_pids'])
            self.assertFalse(Path(f'/proc/{pid}').exists())

    def test_owner_accepts_only_closed_completed_child(self):
        closure=cms.own_worker(self.attempt,self.scratch,time.time()+5,target=completed_child)
        self.assertEqual(closure['child_exitcode'],0)
        self.assertEqual(closure['remaining_owned_processes'],{})
        self.assertEqual((self.attempt/'child-complete').read_text(),'complete')


if __name__=='__main__':
    unittest.main()
