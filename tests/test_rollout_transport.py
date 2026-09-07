import unittest
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
