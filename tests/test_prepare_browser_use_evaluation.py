"""CPU checks for the isolated cloud-browser concurrency cap."""
import ast
import re
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('prepare_browser_use_evaluation', SCRIPTS / 'prepare_browser_use_evaluation.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)

class PreparationTest(unittest.TestCase):
    def test_isolated_concurrency_cap_and_manifest(self):
        root = Path(tempfile.mkdtemp(prefix='openwebrl-stealth-prepare-'))
        source, output = root / 'source', root / 'prepared'
        files = {
            'scripts/run_h200_browser.sh': 'export SLIME_BROWSER_ENV_MODE=local_process',
            'openwebrl/env/config.yaml': '  timeout: 10 # minutes',
            'openwebrl/browser_training_config.yaml': 'browser_rollout_concurrency: 16',
            'reference_manifest.json': json.dumps({'recipe_files_sha256': {}}),
        }
        for name, content in files.items():
            p = source / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        sdk = root / 'browser-use-sdk-3.11.3/browser_use_sdk/generated/v2/models.py'
        sdk.parent.mkdir(parents=True)
        sdk.write_text(''.join(f'class {name}(BaseModel):\n' + "        pattern='decimal',\n" * 3 + '\n' for name in ('BrowserSessionView', 'BrowserSessionItemView')) + 'class Next(BaseModel):\n    pass\n')
        with patch.object(m, 'RUNTIME', root), patch.object(m, 'validate_source'):
            result = m.prepare(source, output)
        self.assertEqual(result['browser_task_concurrency'], 8)
        self.assertIn('browser_rollout_concurrency: 8', (output / 'openwebrl/browser_training_config.yaml').read_text())
        for name, content in files.items():
            self.assertEqual((source / name).read_text(), content)
        self.assertEqual(len(result['changed_files_sha256']), 5)

    def test_billing_pattern_accepts_scientific_values_but_rejects_malformed(self):
        source = ''.join(f'class {name}(BaseModel):\n' + "        pattern='decimal',\n" * 3 + '\n' for name in ('BrowserSessionView', 'BrowserSessionItemView')) + 'class Next(BaseModel):\n    pass\n'
        tree = ast.parse(m.patch_browser_billing_models(source))
        patterns = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        self.assertEqual(len(patterns), 6)
        for pattern in patterns:
            for value in ('0', '0.05', '2.8740614652633667968750E-7', '1e+2'):
                self.assertIsNotNone(re.search(pattern, value))
            for value in ('NaN', 'Infinity', 'garbage', '1e', '.'):
                self.assertIsNone(re.search(pattern, value))

    def test_invalid_concurrency_rejected_before_copy(self):
        with patch.object(m.shutil, 'copytree') as copy:
            for value in (0, -1, 11, 32):
                with self.assertRaisesRegex(ValueError, 'at most 10'):
                    m.prepare('/unused/source', '/unused/output', value)
            copy.assert_not_called()

if __name__ == '__main__':
    unittest.main()
