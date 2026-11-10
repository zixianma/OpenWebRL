"""Deployment admission boundaries; tests do not start external applications."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import hashlib
import json
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("prepare_harmful_apps", Path(__file__).resolve().parents[1] / "scripts/prepare_harmful_apps.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def config():
    return {
        "upstream_commit": M.UPSTREAM_COMMIT,
        "runtime": "docker",
        "writable_paths": {key: "/tmp/harmful-apps/" + key for key in ("artifacts", "container_data", "cache", "snapshots", "audit_log")},
        "sites": {
            name: {"image": image, "app_url": "http://127.0.0.1:" + str(17000 + i * 2), "control_url": "http://127.0.0.1:" + str(17001 + i * 2)}
            for i, (name, image) in enumerate(M.IMAGES.items())
        },
        "protocol": {"retry_preserves_world": True, "capture_external_delivery": True,
                     "agent_can_access_control": False, "agent_can_access_audit": False,
                     "network_egress": "blocked_except_local_sink",
                     "reset_strategy": "restore_verified_snapshot_between_campaigns_only"},
    }


class PreflightTests(unittest.TestCase):
    def test_config_is_not_evidence_of_live_readiness(self):
        result = M.preflight(config())
        self.assertTrue(result["configuration_valid"])
        self.assertFalse(result["experiment_ready"])
        self.assertIsNone(result["resource_request"])

    def test_symlink_escape_and_home_are_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temp:
            link = Path(temp) / "escape"
            link.symlink_to("/gpfs/home/zixianma")
            for path in (str(link / "cache"), "/gpfs/home/zixianma/cache", "/etc/cache"):
                with self.assertRaises(ValueError):
                    M.approved_path(path)

    def test_nonlocal_control_and_ambiguous_hosts_rejected(self):
        for url in ("http://0.0.0.0:8024", "http://10.0.0.1:8024", "http://localhost:8024", "http://127.0.0.1:8024/reset", "http://user@127.0.0.1:8024"):
            with self.assertRaises(ValueError):
                M.loopback_endpoint(url)

    def test_mutable_image_and_retry_reset_rejected(self):
        value = config()
        value["sites"]["gitlab"]["image"] = "am1n3e/webarena-verified-gitlab:latest"
        value["protocol"]["retry_preserves_world"] = False
        errors = M.validate_config(value)
        self.assertTrue(any("immutable digest" in error for error in errors))
        self.assertTrue(any("retry_preserves_world" in error for error in errors))

    def test_apptainer_requires_reviewed_reference_and_namespace_protocol(self):
        value = config()
        value["runtime"] = "apptainer"
        self.assertTrue(any("runtime reference" in e for e in M.validate_config(value)))
        self.assertTrue(any("namespace" in e for e in M.validate_config(value)))

    def reference(self, root):
        evidence = root / 'evidence.json'; evidence.write_text('{"unit_fixture":true}')
        audit = root / 'audit.json'
        audit.write_text(json.dumps({'status':'passed_pending_final_allocation_accounting', 'errors':[],
            'compressed_layers_verified':29,'raw_diff_ids_verified':29,
            'artifacts_sha256':{str(evidence):hashlib.sha256(evidence.read_bytes()).hexdigest()}}))
        return {'site':'reddit','audit_path':str(audit),'audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest()}

    def test_shared_runtime_does_not_admit_another_site_or_transfer_budget(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            value=config(); value['runtime']='apptainer'; value['runtime_reference']=self.reference(Path(temp))
            value['protocol']['network_mode']='isolated_user_network_namespace'
            host={'executables':{'docker':None,'apptainer':'/usr/bin/apptainer'},'slurm_job_id':None}
            with patch.object(M,'inspect_host',return_value=host):
                result=M.preflight(value)
            self.assertTrue(result['configuration_valid']); self.assertFalse(result['experiment_ready'])
            self.assertEqual(result['runtime_reference']['validated_site'],'reddit')
            self.assertFalse(result['runtime_reference']['allocation_transfer_authorized'])
            self.assertFalse(result['runtime_reference']['reusable_live_endpoint'])
            self.assertFalse(any('Docker CLI absent' in b for b in result['blockers']))

    def test_changed_shared_receipt_or_artifact_invalidates_reference(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            root=Path(temp);reference=self.reference(root)
            self.assertTrue(M.inspect_runtime_reference(reference)['verified'])
            (root/'evidence.json').write_text('changed')
            self.assertFalse(M.inspect_runtime_reference(reference)['verified'])
            reference['audit_sha256']='0'*64
            self.assertFalse(M.inspect_runtime_reference(reference)['verified'])

    def test_duplicate_control_port_rejected(self):
        value = config()
        value["sites"]["gitlab"]["control_url"] = value["sites"]["gitlab"]["app_url"]
        self.assertTrue(any("duplicate endpoint" in e for e in M.validate_config(value)))


if __name__ == "__main__":
    unittest.main()
