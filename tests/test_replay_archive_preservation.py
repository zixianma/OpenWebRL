"""Exercise the real debug-save method without starting Ray/GPU workers."""
import ast
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def save_method():
    source = Path(__file__).resolve().parents[1] / 'slime/ray/rollout.py'
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == '_save_debug_rollout_data')
    scope = dict(os=os, Path=Path, torch=torch, logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
    return scope['_save_debug_rollout_data']


def test_mmap_replay_preserves_archive_and_tensor(tmp_path):
    path = tmp_path / '20.pt'
    torch.save({'value': torch.arange(128)}, path)
    before = path.read_bytes()
    tensor = torch.load(path, mmap=True, weights_only=True)['value']
    owner = SimpleNamespace(args=SimpleNamespace(save_debug_rollout_data=str(tmp_path/'{rollout_id}.pt')))
    sample = SimpleNamespace(to_dict=lambda: {'value': tensor})
    with patch.dict(os.environ, OPENWEBRL_REPLAY_FIRST_BATCH=str(path), OPENWEBRL_REPLAY_ROLLOUT_ID='20'):
        save_method()(owner, [sample], 20, False)
    assert path.read_bytes() == before
    assert torch.equal(tensor, torch.arange(128))
    assert torch.equal(torch.load(path, weights_only=True)['value'], tensor)


def test_next_fresh_rollout_still_saves(tmp_path):
    owner = SimpleNamespace(args=SimpleNamespace(save_debug_rollout_data=str(tmp_path/'{rollout_id}.pt')))
    sample = SimpleNamespace(to_dict=lambda: {'reward': 1})
    with patch.dict(os.environ, OPENWEBRL_REPLAY_FIRST_BATCH=str(tmp_path/'20.pt'), OPENWEBRL_REPLAY_ROLLOUT_ID='20'):
        save_method()(owner, [sample], 21, False)
    assert torch.load(tmp_path/'21.pt', weights_only=True)['samples'] == [{'reward': 1}]


def test_evaluation_archive_still_saves(tmp_path):
    owner = SimpleNamespace(args=SimpleNamespace(save_debug_rollout_data=str(tmp_path/'{rollout_id}.pt')))
    sample = SimpleNamespace(to_dict=lambda: {'reward': 1})
    with patch.dict(os.environ, OPENWEBRL_REPLAY_FIRST_BATCH=str(tmp_path/'20.pt'), OPENWEBRL_REPLAY_ROLLOUT_ID='20'):
        save_method()(owner, {'eval': {'samples': [sample]}}, 20, True)
    assert torch.load(tmp_path/'eval_20.pt', weights_only=True)['samples'] == [{'reward': 1}]
