"""Check the released scientific inputs independently of host runtime access."""
import ast
import hashlib
import json
from pathlib import Path
import unittest
import sys
import subprocess

ASSETS = Path(__file__).resolve().parents[1] / 'portable_assets'


class PortableAssetsTests(unittest.TestCase):
    def test_manifest_assets_are_in_the_release_git_tree(self):
        repository = ASSETS.parents[2]
        if not (repository / '.git').exists():
            self.skipTest('Source archive without Git metadata')
        tracked = set(subprocess.check_output(['git', '-C', str(repository), 'ls-files'], text=True).splitlines())
        provenance = json.loads((ASSETS / 'provenance.json').read_text())
        required = {str((ASSETS / name).relative_to(repository)) for name in provenance['files']}
        self.assertEqual(sorted(required - tracked), [], 'Git ignore rules must not omit required runtime sources')

    def test_every_published_asset_matches_its_content_hash(self):
        provenance = json.loads((ASSETS / 'provenance.json').read_text())
        for filename, entry in provenance['files'].items():
            with self.subTest(asset=filename):
                source = (ASSETS / filename).resolve()
                self.assertTrue(source.is_relative_to(ASSETS.resolve()))
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), entry['sha256'])

    def test_primary_campaigns_cover_full_public_cohort_once_per_arm(self):
        for name, count in [('browsercode_gpt_high', 300), ('browsercode_gpt_high_wv_ds', 745),
                            ('browsercode_luna_single_high', 300), ('browsercode_claude', 300),
                            ('browsercode_luna_effort', 300)]:
            root = ASSETS / 'campaign_specs' / name
            roster = json.loads((root / 'task-roster.json').read_text())
            plan = json.loads((root / 'campaign-plan.json').read_text())
            expected = {row['task_id'] for row in roster}
            self.assertEqual(len(roster), count)
            self.assertEqual(len(expected), count)
            for arm in plan['arms']:
                arm = arm['arm'] if isinstance(arm, dict) else arm
                scheduled = [tid for shard in plan['shards'] if shard['arm'] == arm for tid in shard['task_ids']]
                with self.subTest(campaign=name, arm=arm):
                    self.assertEqual(len(scheduled), count)
                    self.assertEqual(set(scheduled), expected)

    def test_scientific_export_does_not_duplicate_tasks_or_account_approvals(self):
        allowed = {'task_id', 'benchmark', 'source_task_id', 'native_task_id', 'campaign_task_id'}
        for root in (ASSETS / 'campaign_specs').iterdir():
            if not (root / 'task-roster.json').is_file():
                continue
            for row in json.loads((root / 'task-roster.json').read_text()):
                self.assertFalse(set(row) - allowed, root.name)
            cfg = json.loads((root / 'runtime-config.json').read_text())
            self.assertFalse('/gpfs/' in json.dumps(cfg), root.name)
            plan = json.loads((root / 'campaign-plan.json').read_text())
            self.assertFalse(set(plan) & {'approval_basis', 'resource_request', 'api_caps', 'campaign_caps', 'approved_by_user'})

    def test_legacy_trees_match_original_source_manifests_and_schedules(self):
        for root in (ASSETS / 'legacy').iterdir():
            if not (root / 'plan.json').is_file():
                continue
            manifest = json.loads((root / 'source-manifest.json').read_text())
            self.assertEqual(manifest['source_hash_mismatches'], [])
            for name, row in manifest['files'].items():
                self.assertEqual(row['sha256'], row['manifest_sha256'])
                self.assertEqual(hashlib.sha256((ASSETS / row['published_path']).read_bytes()).hexdigest(), row['sha256'])
                if name.endswith('.py') and sys.version_info >= (3, 12):
                    ast.parse((ASSETS / row['published_path']).read_text())
            plan = json.loads((root / 'plan.json').read_text())
            schedule_ids = [row['task_id'] for row in plan['schedule']]
            self.assertEqual(len(set(schedule_ids)), len(schedule_ids))
            self.assertEqual(set(schedule_ids), set(plan['task_ids']))


if __name__ == '__main__':
    unittest.main()
