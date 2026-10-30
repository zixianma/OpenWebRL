"""Reject altered tokenization/prompt identities before preparing runnable code."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reproduce'))
SPEC = importlib.util.spec_from_file_location('portable_legacy', ROOT / 'reproduce/legacy.py')
LEGACY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEGACY)


class LegacyIdentityTests(unittest.TestCase):
    def test_foreign_tokenizer_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong = root / 'tokenizer'
            wrong.mkdir()
            (wrong / 'config.json').write_text('{"model_type":"different"}')
            output = root / 'runtime'
            with self.assertRaisesRegex(ValueError, 'tokenizer/processor/chat-template identity differs'):
                LEGACY.prepare('sol61_om2w100_medium', output, tokenizer_source=wrong)
            self.assertFalse(output.exists())

    def test_pinned_identity_includes_chat_template_and_every_text_vision_asset(self):
        manifest = json.loads((LEGACY.LEGACY / 'tokenizer-source-manifest.json').read_text())
        self.assertEqual(manifest['revision'], '1de27d8c51f12e819435303b9e84c4e25ba8401e')
        self.assertFalse(manifest['weights_required'])
        self.assertEqual(set(manifest['files']), {
            'config.json', 'tokenizer_config.json', 'preprocessor_config.json', 'chat_template.json',
            'tokenizer.json', 'merges.txt', 'video_preprocessor_config.json', 'vocab.json', 'generation_config.json'})


if __name__ == '__main__':
    unittest.main()
