"""Submission and GPU startup must reject missing or stale readiness evidence."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path('scripts').resolve()))
import arm_launch_preflight as preflight
import run_arm_turn_bonus_cycles as training
import run_sol_inference as inference


class Preflight(unittest.TestCase):
    def test_receipt_rejects_failed_changed_source_or_changed_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'receipt.json'; artifacts={}
            for key in ('arm_native','sol_native','sol_http'):
                artifact=Path(temp)/f'{key}.json'; artifact.write_text('{"passed": true}')
                artifacts[key]=str(artifact)
            receipt=dict(passed=True,fingerprint={'source':'current'},artifacts=artifacts,
                artifact_sha256={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in artifacts.items()})
            with patch.object(preflight,'fingerprint',return_value={'source':'current'}):
                with self.assertRaisesRegex(ValueError,'Missing'): preflight.require_receipt(path)
                path.write_text(json.dumps(receipt)); self.assertTrue(preflight.require_receipt(path)['passed'])
                for change in (dict(passed=False),dict(fingerprint={'source':'old'})):
                    path.write_text(json.dumps(dict(receipt,**change)))
                    with self.assertRaisesRegex(ValueError,'failed or stale'): preflight.require_receipt(path)
                path.write_text(json.dumps(receipt))
                Path(artifacts['sol_http']).write_text('{"passed": true, "changed": true}')
                with self.assertRaisesRegex(ValueError,'artifact changed'): preflight.require_receipt(path)

    def test_no_controller_process_or_allocation_action_before_readiness(self):
        for module in (training,inference):
            with self.subTest(module=module.__name__),patch.object(preflight,'require_receipt',side_effect=ValueError('stale')), \
                patch.object(module.subprocess,'Popen') as spawn,patch.object(module.subprocess,'check_output') as scheduler:
                with self.assertRaisesRegex(ValueError,'stale'): module.execute({})
                spawn.assert_not_called(); scheduler.assert_not_called()

    def test_submit_rejects_stale_receipt_before_sbatch(self):
        with patch.object(sys,'argv',['check','--submit','all']), \
            patch.object(preflight,'require_receipt',side_effect=ValueError('stale')), \
            patch.object(preflight.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'stale'): preflight.main()
            run.assert_not_called()

    def test_exact_replacement_budgets_and_default_verification_does_not_submit(self):
        commands=preflight.submission_commands('/tmp/passed.json','all')
        self.assertEqual([(x['gpus'],x['minutes'],x['cpus'],x['memory_gib']) for x in commands],
            [(4,239,32,480),(4,239,32,480),(2,119,16,240)])
        for spec in commands:
            self.assertIn('ARM_READINESS_FILE=/tmp/passed.json',next(x for x in spec['command'] if x.startswith('--export')))
        with patch.object(sys,'argv',['check','--verify-only']),patch.object(preflight,'require_receipt',return_value={}), \
            patch.object(preflight.subprocess,'run') as run:
            preflight.main(); run.assert_not_called()


if __name__=='__main__': unittest.main()
