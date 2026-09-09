import unittest
from unittest.mock import patch
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
from slime.utils.rollout_transport import encode_multimodal,decode_multimodal

class TransportTest(unittest.TestCase):
    def test_lossless_shared_storage(self):
        for dtype in (torch.float32,torch.bfloat16,torch.int64):
            with self.subTest(dtype=dtype):
                source=torch.arange(24).reshape(4,6).to(dtype)
                data={'tokens':[[1,2]],'multimodal_train_inputs':[{'pixels':source},None]}
                result=decode_multimodal(encode_multimodal(data))
                actual=result['multimodal_train_inputs'][0]['pixels']
                self.assertTrue(torch.equal(source,actual))
                self.assertEqual(source.dtype,actual.dtype)
                self.assertEqual(source.data_ptr(),actual.data_ptr())
                self.assertIsNone(result['multimodal_train_inputs'][1])
                self.assertIs(data['tokens'],result['tokens'])
    def test_text_only(self):
        data={'tokens':[[1]]}
        self.assertIs(decode_multimodal(encode_multimodal(data)),data)


class FileBackedTransportTest(unittest.TestCase):
    def test_lossless_mappings_and_metadata_only_transfer(self):
        directory = tempfile.mkdtemp(prefix="openwebrl-transport-test-")
        tensors = {
            "pixels": torch.arange(1024 * 512, dtype=torch.float32).reshape(1024, 512),
            "bf16": torch.arange(24, dtype=torch.bfloat16).reshape(4, 6).t(),
            "grid": torch.tensor([[1, 2, 3]], dtype=torch.int64),
            "scalar": torch.tensor(7, dtype=torch.int32),
            "empty": torch.empty(0, 3),
        }
        data = {"tokens": [[1, 2]], "multimodal_train_inputs": [tensors, None]}
        with patch.dict(os.environ, OPENWEBRL_MULTIMODAL_STORAGE_DIR=directory):
            encoded = encode_multimodal(data)
        payload = pickle.dumps(encoded)
        self.assertLess(len(payload), 4096)
        decoded = decode_multimodal(pickle.loads(payload))
        for key, source in tensors.items():
            actual = decoded["multimodal_train_inputs"][0][key]
            self.assertEqual(source.dtype, actual.dtype)
            self.assertEqual(source.shape, actual.shape)
            self.assertTrue(torch.equal(source, actual), key)
        self.assertIsNone(decoded["multimodal_train_inputs"][1])
        self.assertEqual(len(list(Path(directory).glob("*.bin"))), 1)
        child = """import pickle,sys,torch
from slime.utils.rollout_transport import decode_multimodal
x=decode_multimodal(pickle.loads(sys.stdin.buffer.read()))['multimodal_train_inputs'][0]
assert torch.equal(x['pixels'],torch.arange(1024*512,dtype=torch.float32).reshape(1024,512))
assert torch.equal(x['bf16'],torch.arange(24,dtype=torch.bfloat16).reshape(4,6).t())
assert x['scalar'].item()==7 and tuple(x['empty'].shape)==(0,3)
print('mapped-reader-ok')
"""
        for _ in range(2):
            result = subprocess.run([sys.executable, "-c", child], input=payload,
                                    capture_output=True, timeout=30, check=True)
            self.assertIn(b"mapped-reader-ok", result.stdout)

    def test_distinct_files_keep_previous_batch_readable(self):
        directory = tempfile.mkdtemp(prefix="openwebrl-transport-lifetime-")
        def batch(value):
            return {"multimodal_train_inputs": [{"pixels": torch.tensor([value])}]}
        with patch.dict(os.environ, OPENWEBRL_MULTIMODAL_STORAGE_DIR=directory):
            first, second = encode_multimodal(batch(1)), encode_multimodal(batch(2))
        self.assertNotEqual(first["multimodal_train_inputs"][0]["pixels"].path,
                            second["multimodal_train_inputs"][0]["pixels"].path)
        self.assertEqual(decode_multimodal(first)["multimodal_train_inputs"][0]["pixels"].item(), 1)
        self.assertEqual(decode_multimodal(second)["multimodal_train_inputs"][0]["pixels"].item(), 2)
