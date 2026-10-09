"""Include and verify runtime prompt assets when preparing new source snapshots.

Parent prompts retain their bytes. A missing inherited prompt is copied from the
reviewed repository before the new plan/manifest is approved. Never call the
freezing helper on an already approved source or mutate historical snapshots.
"""
import hashlib
import json
from pathlib import Path
from openwebrl.browser_actor_protocol import PROMPT_ASSETS


def freeze_browser_prompt_assets(source, repository):
    """Populate a new unapproved snapshot; return its required prompt hashes."""
    source, repository = Path(source).resolve(), Path(repository).resolve()
    if source == repository:
        raise ValueError('Freeze prompts into an isolated unapproved source')
    contents = {}
    for name in PROMPT_ASSETS:
        destination = source/name
        candidate = destination if destination.exists() else repository/name
        if not candidate.is_file() or not candidate.read_text().strip():
            raise ValueError('Required browser prompt is missing or empty: ' + str(candidate))
        contents[name] = candidate.read_bytes()
    # Validate every asset before writing any. Inherited nonempty prompts are
    # preserved so unrelated workspace prompt edits do not alter the recipe.
    for name, data in contents.items():
        destination = source/name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    return {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}


def validate_browser_prompt_assets(source):
    """Fail before model/browser startup if required prompt evidence is absent."""
    source = Path(source)
    manifest = json.loads((source/'reference_manifest.json').read_text())
    hashes = manifest['recipe_files_sha256']
    for name in PROMPT_ASSETS:
        path = source/name
        if not path.is_file() or not path.read_text().strip():
            raise ValueError('Required browser prompt is missing or empty: ' + str(path))
        if hashes.get(name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Required browser prompt is absent from manifest or changed: ' + name)
    return {name: hashes[name] for name in PROMPT_ASSETS}
