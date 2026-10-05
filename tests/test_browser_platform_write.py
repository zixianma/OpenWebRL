"""Exercise the production write executor without importing training stacks."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest


def write_executor(path=None):
    path = path or Path(__file__).resolve().parents[1] / 'openwebrl/env/web_env.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WebEnv')
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_execute_write')
    namespace = {'Dict': Dict, 'Any': Any}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_execute_write']


@pytest.mark.parametrize('is_mac,shortcut', [(True, 'Meta+A'), (False, 'Control+A')])
def test_write_uses_remote_platform_and_clears_before_typing(is_mac, shortcut):
    calls = []
    async def press(key): calls.append(('press', key))
    async def type_text(text): calls.append(('type', text))
    async def evaluate(js):
        if 'navigator.platform' in js: return is_mac
        return 'new'
    async def noop(*args, **kwargs): pass
    async def describe(): return '<input>'
    page = SimpleNamespace(keyboard=SimpleNamespace(press=press, type=type_text),
        evaluate=evaluate, wait_for_load_state=noop, wait_for_timeout=noop, url='about:blank')
    ok, feedback = asyncio.run(write_executor()(SimpleNamespace(
        page=page, _get_focused_element_description=describe), {'message': 'new'}))
    assert ok and 'new' in feedback
    assert calls == [('press', shortcut), ('press', 'Backspace'), ('type', 'new')]


def test_platform_lookup_failure_does_not_type_with_a_guessed_shortcut():
    calls = []
    async def evaluate(js): raise RuntimeError('page closed')
    async def describe(): return '<input>'
    async def press(key): calls.append(key)
    page = SimpleNamespace(evaluate=evaluate, url='about:blank', keyboard=SimpleNamespace(press=press))
    ok, feedback = asyncio.run(write_executor()(SimpleNamespace(
        page=page, _get_focused_element_description=describe), {'message': 'new'}))
    assert not ok and 'page closed' in feedback and not calls
