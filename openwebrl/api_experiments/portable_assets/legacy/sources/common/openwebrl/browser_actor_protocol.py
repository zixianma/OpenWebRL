"""Scientific identity of the restored policy and native API actor history.

Historical frozen sources retain their old protocol. This identity is part of
every newly prepared protocol digest, not only a source-code manifest hash.
"""
import hashlib
from pathlib import Path

PROMPT_ASSETS = (
    'openwebrl/env/prompts/system_prompt.md',
    'openwebrl/env/prompts/system_prompt_browser_env.md',
    'openwebrl/env/prompts/system_prompt_api_browser.md',
)
HARNESS_VERSION = 'browser-prompts-native-responses-v1'


def browser_harness_protocol(source=None):
    source = Path(source) if source is not None else Path(__file__).resolve().parents[1]
    hashes = {}
    for name in PROMPT_ASSETS:
        path = source/name
        try:
            content = path.read_bytes()
            if not content.decode('utf-8').strip():
                raise ValueError('Required browser prompt is empty: ' + name)
        except (OSError, UnicodeError) as exc:
            raise ValueError('Required browser prompt is unavailable: ' + name) from exc
        hashes[name] = hashlib.sha256(content).hexdigest()
    return dict(browser_harness_version=HARNESS_VERSION,
                required_browser_prompt_sha256=hashes,
                local_actor_policy='required_browser_env_markdown_v1',
                api_actor_history='native_responses_v1')


def validate_browser_harness_protocol(protocol, source=None):
    expected = browser_harness_protocol(source)
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError('Browser policy/history protocol identity changed')
    return expected
