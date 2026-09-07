"""Lossless, zero-copy Ray transport for CPU multimodal tensors.

Ray shares NumPy arrays between readers; pickled torch tensors instead allocate
private copies on every training rank. Restored CPU views must remain read-only
until moved to the GPU by the training materializer.
"""
import numpy as np
import torch


def encode_multimodal(data):
    if 'multimodal_train_inputs' not in data:
        return data
    result = dict(data)
    result['multimodal_train_inputs'] = [
        {k: (v.detach().cpu().view(torch.uint16).numpy(), 'bfloat16')
         if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16 else
         v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
         for k, v in sample.items()} if sample is not None else None
        for sample in data['multimodal_train_inputs']]
    return result


def decode_multimodal(data):
    if 'multimodal_train_inputs' not in data:
        return data
    result = dict(data)
    result['multimodal_train_inputs'] = [
        {k: torch.from_numpy(v[0]).view(torch.bfloat16)
         if isinstance(v, tuple) and len(v) == 2 and v[1] == 'bfloat16' else
         torch.from_numpy(v) if isinstance(v, np.ndarray) else v
         for k, v in sample.items()} if sample is not None else None
        for sample in data['multimodal_train_inputs']]
    return result
