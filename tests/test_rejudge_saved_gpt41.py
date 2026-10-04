import importlib.util
import io
from pathlib import Path
import pickle

import unittest

spec = importlib.util.spec_from_file_location(
    'rejudge_saved_gpt41', Path(__file__).resolve().parents[1] / 'scripts/rejudge_saved_gpt41.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RejudgeTests(unittest.TestCase):
    def test_invalid_success_does_not_inflate_either_rate(self):
        records = [
            {'invalid': False, 'original_invalid': False, 'reward': 1},
            {'invalid': False, 'original_invalid': False, 'reward': 0},
            {'invalid': True, 'original_invalid': True, 'reward': 1},
            {'invalid': True, 'original_invalid': False, 'reward': 1, 'judge_timeout': True},
        ]
        result = module.summarize(records, 300)
        assert result['successes'] == 1
        assert result['success_rate_all_planned'] == 1 / 300
        assert result['success_rate_valid'] == 0.5
        assert result['original_invalid'] == 1
        assert result['judge_timeouts'] == 1


    def test_metadata_reader_preserves_screenshot_and_response_bytes(self):
        data = {'samples': [{'response': 'done()', 'multimodal_inputs': {'images': ['abc123']}}]}
        assert module.MetadataOnly(io.BytesIO(pickle.dumps(data))).load() == data


    def test_metadata_reader_rejects_unexpected_global(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported metadata global'):
            module.MetadataOnly(io.BytesIO(pickle.dumps(Path('/tmp')))).load()


if __name__ == "__main__":
    unittest.main()
