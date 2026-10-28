import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import validate_webprm_shopping_hosting as owner
from openwebrl import webprm_cms_hosting as h
from tests.test_webprm_cms_hosting import snapshot


class ShoppingOwnerTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/tmp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_run_rejects_before_materialization_or_process(self):
        p=self.root/'proposal.json';p.write_text(json.dumps(dict(protocol=h.SHOPPING.protocol)))
        a=self.root/'approval.json';a.write_text('{}')
        with patch.object(owner.materializer,'materialize') as materialize,patch.object(owner.subprocess,'check_output') as command:
            with self.assertRaisesRegex(h.Closed,'approval'):owner.run(p,a)
            materialize.assert_not_called();command.assert_not_called()

    def test_profile_resources_and_scheduler_are_shopping_sized(self):
        r=h.SHOPPING.resources
        self.assertEqual((r['gpus'],r['cpus'],r['memory_gib']),(0,8,128))
        raw=snapshot(cpus=8,memory=128*1024,minutes=240,end=100+240*60).replace('cpu=4,node=1,mem=16G','cpu=8,node=1,mem=128G')
        got=h.scheduler(raw,'42',requested_seconds=14400,now=110,profile=h.SHOPPING)
        self.assertEqual(got['work_deadline'],100+14400-h.SHOPPING.limits['closure_seconds'])
        with self.assertRaises(h.Closed):h.scheduler(raw,'42',requested_seconds=14400,now=110)
        with self.assertRaises(h.Closed):h.scheduler(snapshot(),'42',requested_seconds=7200,now=110,profile=h.SHOPPING)

    def test_storage_admits_three_shopping_states(self):
        # probe 358020: 56.4 GB apparent base; base + two private states + reserve must fit.
        self.assertLessEqual(3*56361205025+h.SHOPPING.limits['runtime_reserve_bytes'],h.SHOPPING.limits['scratch_bytes'])
        self.assertGreater(h.SHOPPING.limits['retained_bytes'],3*2**30)  # 2.25 GB manifest retained as evidence
        self.assertGreaterEqual(h.SHOPPING.apparent_entries,30000000)

    def test_pinned_shopping_evidence_and_plan_are_current(self):
        p=h.SHOPPING
        self.assertEqual(h.sha(owner.PLAN),p.plan_sha)
        for name,path in dict(acquisition_audit=owner.ACQUIRED/'completed.json',
                layer_proof=owner.ACQUIRED/'attempt-356538/raw-layer-verification.json',
                manifest=owner.ACQUIRED/'metadata/manifest.json',bringup_probe=owner.PROBE).items():
            self.assertEqual(h.sha(path),p.pins[name],name)
        plan=json.loads(owner.PLAN.read_text())
        nginx=Path(plan['container_file_install_map']['/etc/nginx/conf.d/default.conf']['path']).read_text()
        self.assertIn('listen 127.0.0.1:7770;',nginx);self.assertNotIn('7780',nginx)

    def test_content_identity_ignores_destination(self):
        manifest=dict(destination='/tmp/a',entries={'x':1})
        self.assertEqual(owner.content_sha256(manifest),owner.content_sha256(dict(manifest,destination='/tmp/b')))
        from openwebrl.webprm_webarena import digest
        self.assertEqual(owner.content_sha256(manifest),digest(dict(entries={'x':1})))

    def test_origin_commands_target_7770(self):
        self.assertEqual(owner.ORIGIN_COMMANDS[-1],['setup:store-config:set','--base-url=http://127.0.0.1:7770'])
        self.assertIn(b"'http://127.0.0.1:7770/'",owner.SECURE_SQL)


if __name__=='__main__':
    unittest.main()
