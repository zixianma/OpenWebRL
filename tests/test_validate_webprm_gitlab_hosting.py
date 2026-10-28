"""Offline launch boundary, cumulative budget and safe private-copy adaptations."""
import json
from pathlib import Path
import tempfile
import unittest
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from scripts import validate_webprm_gitlab_hosting as owner
from openwebrl import webprm_cms_hosting as h
from tests.test_webprm_cms_hosting import snapshot


class GitLabOwnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='/tmp');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_missing_approval_stops_before_any_materialization_or_scheduler(self):
        p=self.root/'proposal.json';p.write_text(json.dumps(dict(protocol=h.GITLAB.protocol)))
        a=self.root/'approval.json';a.write_text('{}')
        with patch.object(owner.materializer,'materialize') as mat,patch.object(owner.subprocess,'check_output') as cmd:
            with self.assertRaisesRegex(h.Closed,'approval'):owner.run(p,a)
            mat.assert_not_called();cmd.assert_not_called()

    def test_signin_readiness_bypasses_proxy_and_rejects_redirect(self):
        class Handler(BaseHTTPRequestHandler):
            redirect=False
            def log_message(self,*args):pass
            def do_GET(self):
                self.send_response(302 if self.redirect else 200)
                if self.redirect:self.send_header('Location','http://example.invalid/private')
                self.end_headers()
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            env=dict(http_proxy='http://127.0.0.1:1',HTTP_PROXY='http://127.0.0.1:1',no_proxy='',NO_PROXY='')
            with patch.dict(os.environ,env),patch.object(owner,'ORIGIN','http://127.0.0.1:'+str(server.server_port)):
                self.assertTrue(owner.signin_ready())
                Handler.redirect=True
                with self.assertRaisesRegex(h.Closed,'must not redirect'):owner.signin_ready()
        finally:server.shutdown();server.server_close();thread.join()

    def test_exact_scheduler_rejects_gpu_wrong_resources_and_elapsed_deadline(self):
        raw=snapshot(cpus=8,memory=96*1024,minutes=120,end=7300).replace('cpu=4,node=1,mem=16G','cpu=8,node=1,mem=96G')
        self.assertEqual(h.scheduler(raw,'42',requested_seconds=7200,now=110,profile=h.GITLAB)['work_deadline'],7000)
        for wrong in [raw.replace('mem=96G','mem=96G,gres/gpu=1'),snapshot()]:
            with self.assertRaises(h.Closed):h.scheduler(wrong,'42',requested_seconds=7200,now=110,profile=h.GITLAB)
        with self.assertRaises(h.Closed):h.scheduler(raw,'42',requested_seconds=7200,now=7001,profile=h.GITLAB)

    def test_all_failures_debit_time_and_retained_scratch(self):
        folder=self.root/'attempt-42';folder.mkdir()
        (folder/'closure.json').write_text(json.dumps(dict(remaining_owned_processes={})))
        (folder/'storage-final.json').write_text(json.dumps(dict(scratch_bytes=123)))
        rows=[dict(job_id='42',requested_seconds=7200)]
        budget=h.prior_budget(self.root,rows,{'42':dict(state='FAILED',elapsed_seconds=123)},h.GITLAB)
        self.assertEqual(budget,dict(prior_seconds=123,remaining_seconds=7020,prior_scratch_bytes=123))
        (folder/'context.json').write_text(json.dumps(dict(scratch=str(self.root/'not-on-this-node'))))
        self.assertEqual(h.prior_budget(self.root,rows,{'42':dict(state='FAILED',elapsed_seconds=123)},h.GITLAB)['prior_scratch_bytes'],123)
        with self.assertRaises(h.Closed):h.prior_budget(self.root,rows,{},h.GITLAB)
        with self.assertRaises(h.Closed):h.prior_budget(self.root,rows,{'42':dict(state='RUNNING',elapsed_seconds=123)},h.GITLAB)

    def test_pinned_evidence_and_storage_geometry(self):
        self.assertEqual(h.sha(owner.PLAN),h.GITLAB.plan_sha)
        for field,path in dict(acquisition_audit=owner.ACQUIRED/'completed.json',layer_proof=owner.ACQUIRED/'attempt-356548/raw-layer-verification.json',manifest=owner.ACQUIRED/'metadata/manifest.json',bringup_probe=owner.PROBE).items():
            self.assertEqual(h.sha(path),h.GITLAB.pins[field])
        self.assertLess(3*34179881242+h.GITLAB.limits['runtime_reserve_bytes'],h.GITLAB.limits['scratch_bytes'])
        self.assertEqual(h.GITLAB.resources['max_allocations'],2)

    def test_private_copy_configuration_preserves_original_evidence(self):
        state=self.root/'state';(state/'etc/gitlab').mkdir(parents=True)
        original="external_url 'http://original:8023'\n"
        (state/'etc/gitlab/gitlab.rb').write_text(original);(state/'etc/hosts').write_text('original hosts\n')
        controls=self.root/'controls';plan=h.read(owner.PLAN)
        owner.prepare_controls(state,controls,plan)
        self.assertEqual((controls/'gitlab.rb.before').read_text(),original)
        self.assertIn(owner.ORIGIN,(state/'etc/gitlab/gitlab.rb').read_text())
        self.assertEqual((controls/'hosts.before').read_text(),'original hosts\n')
        script=(controls/'controller.sh').read_text()
        self.assertNotIn('rm -',script);self.assertNotIn('-delete',script);self.assertIn('webprm-preserved',script)
        self.assertNotIn('HOME=',script)

    def test_symlink_mountpoint_cannot_escape_private_state(self):
        state=self.root/'state';state.mkdir();outside=self.root/'outside';outside.mkdir()
        (state/'opt').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(h.Closed):owner.prepare_controls(state,self.root/'controls',h.read(owner.PLAN))
        self.assertEqual(list(outside.iterdir()),[])

    def test_symlink_configuration_cannot_escape_private_state(self):
        state=self.root/'state';(state/'etc/gitlab').mkdir(parents=True)
        outside=self.root/'outside';outside.write_text('must remain unchanged')
        (state/'etc/gitlab/gitlab.rb').symlink_to(outside)
        with self.assertRaises(h.Closed):owner.prepare_controls(state,self.root/'controls',h.read(owner.PLAN))
        self.assertEqual(outside.read_text(),'must remain unchanged')


if __name__=='__main__':unittest.main()
