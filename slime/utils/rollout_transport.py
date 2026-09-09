"""Lossless, zero-copy Ray transport for CPU multimodal tensors.

Ray shares NumPy arrays between readers; pickled torch tensors instead allocate
private copies on every training rank. Restored CPU views must remain read-only
until moved to the GPU by the training materializer.
"""
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np
import torch


@dataclass(frozen=True)
class FileBackedArray:
    path: str
    offset: int
    shape: tuple[int, ...]
    dtype: str
    bfloat16: bool = False


def _encode_file_backed(data, directory):
    """Share read-only, reclaimable file mappings instead of large Ray objects.

    The directory must be visible to every consuming rank. Files are retained
    for the run lifetime; callers choose node-local or shared storage.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    result = dict(data)
    samples = []
    with tempfile.NamedTemporaryFile(dir=directory, prefix="rollout-", suffix=".bin", delete=False) as stream:
        path = str(Path(stream.name).resolve())
        for sample in data["multimodal_train_inputs"]:
            if sample is None:
                samples.append(None)
                continue
            encoded = {}
            for key, value in sample.items():
                bf16 = isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu()
                    array = value.view(torch.uint16).numpy() if bf16 else value.numpy()
                elif isinstance(value, np.ndarray):
                    array = value
                else:
                    encoded[key] = value
                    continue
                if array.dtype.hasobject:
                    raise TypeError("Object arrays cannot use file-backed rollout transport")
                shape = tuple(array.shape)
                padding = (-stream.tell()) % 64
                if padding:
                    stream.write(b"\0" * padding)
                offset = stream.tell()
                np.ascontiguousarray(array).tofile(stream)
                encoded[key] = FileBackedArray(path, offset, shape, array.dtype.str, bf16)
            samples.append(encoded)
    result["multimodal_train_inputs"] = samples
    return result


def _decode_value(value):
    if isinstance(value, FileBackedArray):
        dtype = np.dtype(value.dtype)
        array = (np.empty(value.shape, dtype=dtype) if 0 in value.shape else
                 np.memmap(value.path, mode="r", offset=value.offset, shape=value.shape, dtype=dtype))
        tensor = torch.from_numpy(array)
        return tensor.view(torch.bfloat16) if value.bfloat16 else tensor
    if isinstance(value, tuple) and len(value) == 2 and value[1] == "bfloat16":
        return torch.from_numpy(value[0]).view(torch.bfloat16)
    return torch.from_numpy(value) if isinstance(value, np.ndarray) else value


def encode_multimodal(data):
    if 'multimodal_train_inputs' not in data:
        return data
    directory = os.environ.get('OPENWEBRL_MULTIMODAL_STORAGE_DIR')
    if directory:
        return _encode_file_backed(data, directory)
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
        {k: _decode_value(v) for k, v in sample.items()} if sample is not None else None
        for sample in data['multimodal_train_inputs']]
    return result


def file_back_completed_group(group):
    """Replace completed samples' CPU image buffers with lossless mapped views.

    Collection retains accepted AND rejected samples for telemetry. Mapping them
    at completion bounds non-reclaimable memory before the train-transfer stage.
    Sample identity, tokens, rewards, and all other fields remain unchanged.
    torch.save serializes tensor values, so recovery files do not depend on these
    temporary mappings on the next node. The opt-in directory is required.
    """
    directory = os.environ.get("OPENWEBRL_MULTIMODAL_STORAGE_DIR")
    if not directory:
        return 0

    def leaves(value):
        if isinstance(value, (list, tuple)):
            for child in value:
                yield from leaves(child)
        else:
            yield value

    # Keep identity even if a custom generator returns an aliased sample twice.
    samples = {id(sample): sample for sample in leaves(group)
               if getattr(sample, "multimodal_train_inputs", None) is not None}
    if not samples:
        return 0
    data = {"multimodal_train_inputs": [sample.multimodal_train_inputs for sample in samples.values()]}
    mapped = decode_multimodal(_encode_file_backed(data, directory))
    for sample, inputs in zip(samples.values(), mapped["multimodal_train_inputs"], strict=True):
        sample.multimodal_train_inputs = inputs
    return len(samples)
