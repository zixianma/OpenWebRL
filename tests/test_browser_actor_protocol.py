"""Restored policies and native replay are a new scientific protocol identity."""
from copy import deepcopy

import pytest

from openwebrl.browser_actor_protocol import (
    HARNESS_VERSION, browser_harness_protocol, validate_browser_harness_protocol,
)
from openwebrl.luna_qwen_full_eval import protocol_for_family, validate_plan
from openwebrl.luna_qwen_metrics import digest
from openwebrl.reasoning_actor_eval import validate_plan as validate_high_plan


@pytest.mark.parametrize('family', ['qwen', 'sft', 'luna'])
@pytest.mark.parametrize('change', ['legacy', 'history', 'prompt'])
def test_new_digest_distinguishes_restored_policy_and_history_and_worker_rejects_drift(family, change):
    corrected = protocol_for_family(family)
    assert corrected['browser_harness_version'] == HARNESS_VERSION
    assert validate_browser_harness_protocol(corrected) == browser_harness_protocol()
    changed = deepcopy(corrected)
    if change == 'legacy':
        for key in browser_harness_protocol(): changed.pop(key)
    elif change == 'history': changed['api_actor_history'] = 'flattened_xml_v0'
    else: changed['required_browser_prompt_sha256']['openwebrl/env/prompts/system_prompt_browser_env.md'] = 'different'
    assert digest(changed) != digest(corrected)
    with pytest.raises(ValueError, match='scientific protocol'):
        validate_plan(dict(family=family, protocol=changed), {})


@pytest.mark.parametrize('change', ['legacy', 'history', 'prompt'])
def test_high_actor_worker_refuses_legacy_or_mismatched_harness_before_other_resources(change):
    protocol = protocol_for_family('luna')
    if change == 'legacy':
        for key in browser_harness_protocol(): protocol.pop(key)
    elif change == 'history': protocol['api_actor_history'] = 'flattened_xml_v0'
    else: protocol['required_browser_prompt_sha256'] = {}
    with pytest.raises(ValueError, match='policy/history protocol identity'):
        validate_high_plan(dict(family='luna', arm_mode='sol61_high', protocol=protocol), {})


def test_runtime_prompt_change_is_rejected_even_if_protocol_object_is_unchanged(tmp_path):
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    from openwebrl.browser_actor_protocol import PROMPT_ASSETS
    for name in PROMPT_ASSETS:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((repo/name).read_bytes())
    protocol = browser_harness_protocol(tmp_path)
    path = tmp_path/PROMPT_ASSETS[1]
    path.write_text(path.read_text() + '\nUnreviewed policy change.\n')
    with pytest.raises(ValueError, match='policy/history protocol identity'):
        validate_browser_harness_protocol(protocol, tmp_path)


@pytest.mark.parametrize('mode,actor', [('luna_high', 'gpt-6-luna'), ('sol61_high', 'gpt-6.1-sol')])
def test_high_actor_accepts_complete_corrected_protocol(mode, actor, tmp_path):
    from openwebrl.reasoning_actor_policy import metric_prices
    protocol = dict(protocol_for_family('luna'), actor=actor, reasoning_effort='high',
                    resize_output_coords=False, browser_coordinate_space='viewport_pixels')
    tasks = [f'synthetic-task-{i}' for i in range(300)]
    plan = dict(family='luna', arm_mode=mode, protocol=protocol, output=str(tmp_path),
                pricing=metric_prices(actor), task_ids=tasks,
                schedule=[dict(task_id=task, modes=[mode]) for task in tasks],
                experiment_id='corrected-offline', wandb_prefix='corrected-offline-' + mode)
    assert validate_high_plan(plan, dict(output=str(tmp_path), plan_sha256=digest(plan))) == 'luna'
