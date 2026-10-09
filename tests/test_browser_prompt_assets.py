"""A source manifest must carry the prompts the real browser loader reads."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from browser_prompt_assets import (PROMPT_ASSETS, freeze_browser_prompt_assets,
                                   validate_browser_prompt_assets)

REPO = Path(__file__).resolve().parents[1]


def prepared_source(tmp_path):
    source = tmp_path / 'isolated-source'
    hashes = freeze_browser_prompt_assets(source, REPO)
    (source / 'reference_manifest.json').write_text(json.dumps(dict(recipe_files_sha256=hashes)))
    return source, hashes


def test_missing_parent_assets_are_copied_and_all_actual_prompts_pinned(tmp_path):
    source, hashes = prepared_source(tmp_path)
    assert len(PROMPT_ASSETS) == 3
    assert 'openwebrl/env/prompts/system_prompt_api_browser.md' in hashes
    assert validate_browser_prompt_assets(source) == hashes
    for name in PROMPT_ASSETS:
        assert (source/name).read_bytes() == (REPO/name).read_bytes()
        assert hashes[name] == hashlib.sha256((source/name).read_bytes()).hexdigest()


def test_existing_parent_prompt_keeps_its_bytes(tmp_path):
    source = tmp_path / 'isolated-source'
    inherited = source / PROMPT_ASSETS[0]
    inherited.parent.mkdir(parents=True)
    inherited.write_text('Already reviewed inherited policy.\n')
    hashes = freeze_browser_prompt_assets(source, REPO)
    assert inherited.read_text() == 'Already reviewed inherited policy.\n'
    assert hashes[PROMPT_ASSETS[0]] == hashlib.sha256(inherited.read_bytes()).hexdigest()


@pytest.mark.parametrize('problem', ['missing', 'blank', 'unpinned', 'changed'])
def test_preflight_rejects_missing_blank_unpinned_or_changed_policy(tmp_path, problem):
    source, hashes = prepared_source(tmp_path)
    name = PROMPT_ASSETS[1]
    path = source/name
    if problem == 'missing':
        path.rename(path.with_suffix('.held'))
    elif problem == 'blank':
        path.write_text(' \n')
    elif problem == 'unpinned':
        hashes.pop(name)
        (source/'reference_manifest.json').write_text(json.dumps(dict(recipe_files_sha256=hashes)))
    else:
        path.write_text(path.read_text() + '\nChanged unreviewed behavior.\n')
    with pytest.raises(ValueError, match='Required browser prompt'):
        validate_browser_prompt_assets(source)


def test_invalid_input_cannot_partially_populate_snapshot(tmp_path):
    repository, source = tmp_path/'repository', tmp_path/'source'
    first = repository/PROMPT_ASSETS[0]
    first.parent.mkdir(parents=True)
    first.write_text('Nonempty first prompt')
    with pytest.raises(ValueError, match='missing or empty'):
        freeze_browser_prompt_assets(source, repository)
    assert not source.exists()


def test_freeze_refuses_repository_destination():
    with pytest.raises(ValueError, match='isolated unapproved source'):
        freeze_browser_prompt_assets(REPO, REPO)


@pytest.mark.parametrize('mode,name', [('browser_env', PROMPT_ASSETS[1]), ('slime', PROMPT_ASSETS[0])])
def test_real_loader_uses_nonempty_frozen_policy_not_an_empty_fallback(tmp_path, monkeypatch, mode, name):
    from openwebrl import generate_browser as generation
    source, _ = prepared_source(tmp_path)
    monkeypatch.setattr(generation, '_BROWSER_DIR', str(source/'openwebrl'))
    # A real task fixture avoids dataset lookup and exercises the actual loader.
    task_path = source/'openwebrl/env/tasks/probe.json'
    task_path.parent.mkdir(parents=True)
    task_path.write_text(json.dumps(dict(task_id='probe', task='Inspect the current page')))
    task, tools, policy = generation._load_local_resources('probe', {}, response_format_mode_name=mode)
    assert task['task_id'] == 'probe'
    assert policy.strip()
    assert policy == (source/name).read_text()
    assert tools == []


@pytest.mark.parametrize('problem', ['missing', 'blank'])
def test_real_loader_fails_closed_before_environment_startup(tmp_path, monkeypatch, problem):
    from openwebrl import generate_browser as generation
    source, _ = prepared_source(tmp_path)
    path = source/PROMPT_ASSETS[1]
    if problem == 'missing':
        path.rename(path.with_suffix('.held'))
    else:
        path.write_text('\n\t')
    monkeypatch.setattr(generation, '_BROWSER_DIR', str(source/'openwebrl'))
    with pytest.raises(ValueError, match='Required browser policy file is (missing|empty)'):
        generation._load_local_resources('probe', {}, response_format_mode_name='browser_env')
