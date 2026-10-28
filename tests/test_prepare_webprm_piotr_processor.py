"""Offline guards for the additive Piotr serving metadata conversion."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import secrets
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts.prepare_webprm_piotr_processor import (
    CHECKPOINT, METADATA, RUNTIME, processor_metadata, safe_path,
    source_identity, tokenizer_metadata, validate_view, verify_weight_bindings, verify_weight_content,
)


@unittest.skipUnless(CHECKPOINT.exists(), 'Pinned local Piotr checkpoint unavailable')
class ProcessorViewTests(unittest.TestCase):
    def test_image_metadata_changes_only_pixel_ceiling(self):
        original = json.loads((CHECKPOINT / 'processor_config.json').read_text())
        preserved = deepcopy(original)
        changed = processor_metadata(original)
        self.assertEqual(original, preserved)
        self.assertEqual(changed['image_processor']['size']['longest_edge'], 262144)
        changed['image_processor']['size']['longest_edge'] = 16777216
        self.assertEqual(changed, original)
        original['image_processor']['size']['shortest_edge'] = 1
        with self.assertRaises(ValueError):
            processor_metadata(original)

    def test_tokenizer_conversion_preserves_order_values_and_every_other_field(self):
        original = json.loads((CHECKPOINT / 'tokenizer_config.json').read_text())
        preserved = deepcopy(original)
        changed = tokenizer_metadata(original)
        self.assertEqual(original, preserved)
        self.assertEqual(changed.pop('additional_special_tokens'), original['extra_special_tokens'])
        expected = dict(original)
        expected.pop('extra_special_tokens')
        self.assertEqual(changed, expected)
        for invalid in ({'extra_special_tokens': []}, {'extra_special_tokens': {'x': 'y'}},
                        {'extra_special_tokens': ['x'], 'additional_special_tokens': ['z']}):
            with self.assertRaises(ValueError):
                tokenizer_metadata(invalid)

    def test_original_source_hashes_and_cached_weight_identity_still_match(self):
        identities = source_identity()
        self.assertEqual(sum(value['bytes'] for value in identities.values()), 8875719328)
        for name, expected in METADATA.items():
            self.assertEqual(hashlib.sha256((CHECKPOINT / name).read_bytes()).hexdigest(), expected)

    def test_no_writable_escape_or_symlink_parent(self):
        for invalid in (Path('/gpfs/home/zixianma/processor'), RUNTIME,
                        RUNTIME / '..' / 'processor', Path('relative-view')):
            with self.assertRaises(ValueError):
                safe_path(invalid)
        root = RUNTIME / ('processor-path-test-' + secrets.token_hex(6))
        root.mkdir(mode=0o700)
        (root / 'escape').symlink_to('/gpfs/home/zixianma', target_is_directory=True)
        with self.assertRaises(ValueError):
            safe_path(root / 'escape' / 'never-created')
        self.assertEqual(safe_path(root / 'permitted'), root / 'permitted')

    def test_full_view_accepts_changed_node_local_stat_ids_with_content_verification(self):
        view = RUNTIME / 'piotr-serving-processor-v3'
        observed = source_identity()
        for row in observed.values():
            row['device'] += 100
            row['inode'] += 100
            row['mtime_ns'] += 100
        with patch('scripts.prepare_webprm_piotr_processor.source_identity', return_value=observed), \
                patch('scripts.prepare_webprm_piotr_processor.verify_weight_content') as content:
            manifest = validate_view(view)
        self.assertEqual(content.call_count, 2)
        for call in content.call_args_list:
            path, size, expected = call.args
            self.assertEqual(str(path), observed[path.name]['source'])
            self.assertEqual(size, observed[path.name]['bytes'])
            self.assertEqual(expected, observed[path.name]['cached_lfs_sha256'])
        self.assertNotEqual(manifest['weights'], observed)

    def test_portable_weight_metadata_corruption_is_not_ignored(self):
        observed = source_identity()
        for key, invalid in [('source', '/tmp/other-weights'), ('bytes', 1), ('cached_lfs_sha256', '0' * 64)]:
            recorded = deepcopy(observed)
            recorded[next(iter(recorded))][key] = invalid
            with patch('scripts.prepare_webprm_piotr_processor.verify_weight_content') as content:
                with self.assertRaises(ValueError):
                    verify_weight_bindings(recorded, observed)
                content.assert_not_called()


class WeightContentTests(unittest.TestCase):
    def setUp(self):
        self.root = RUNTIME / ('processor-content-test-' + secrets.token_hex(6))
        self.root.mkdir(mode=0o700)
        self.path = self.root / 'small-weight-fixture'
        self.payload = b'original-weight-fixture'
        self.path.write_bytes(self.payload)
        self.expected = hashlib.sha256(self.payload).hexdigest()

    def test_same_size_same_mtime_corruption_fails_actual_digest(self):
        proof = verify_weight_content(self.path, len(self.payload), self.expected)
        self.assertEqual(proof['sha256'], self.expected)
        original = self.path.stat()
        self.path.write_bytes(b'X' * len(self.payload))
        os.utime(self.path, ns=(original.st_atime_ns, original.st_mtime_ns))
        self.assertEqual(self.path.stat().st_size, original.st_size)
        self.assertEqual(self.path.stat().st_mtime_ns, original.st_mtime_ns)
        with self.assertRaisesRegex(ValueError, 'content digest'):
            verify_weight_content(self.path, len(self.payload), self.expected)

    def test_symlink_rejected_without_following_target(self):
        link = self.root / 'weight-link'
        link.symlink_to(self.path)
        with self.assertRaises(OSError):
            verify_weight_content(link, len(self.payload), self.expected)

    def test_file_changes_during_hashing_are_rejected(self):
        before = self.path.stat()
        fields = ('st_mode', 'st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        after = SimpleNamespace(**{key: getattr(before, key) for key in fields})
        after.st_ctime_ns += 1
        with patch('scripts.prepare_webprm_piotr_processor.os.fstat', side_effect=[before, after]):
            with self.assertRaisesRegex(ValueError, 'changed during'):
                verify_weight_content(self.path, len(self.payload), self.expected)


if __name__ == '__main__':
    unittest.main()
