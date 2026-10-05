"""A timeout may save the next observation before any corresponding selection."""
import importlib.util
import base64
import io
from pathlib import Path

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    'selection_audit', Path(__file__).resolve().parents[1] / 'scripts/audit_sft_decision_selection.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def png():
    out = io.BytesIO()
    Image.new('RGB', (2, 2)).save(out, format='PNG')
    return out.getvalue()


def test_diagnosed_next_observation_is_separate_from_selected_states():
    raw = png()
    states, hashes = audit.align_partial_screenshots(
        [raw, raw], 1, {'unselected_observation_sha256': [audit.digest(raw)]})
    assert states == [raw] and hashes == [audit.digest(raw)]
    assert audit.align_partial_screenshots([raw], 1, {}) == ([raw], [])


@pytest.mark.parametrize('count,extra,diagnosis', [
    (1, 1, {}),
    (1, 1, {'unselected_observation_sha256': ['wrong']}),
    (1, 2, {}),
    (2, 0, {}),
])
def test_unexplained_extra_or_missing_states_fail(count, extra, diagnosis):
    with pytest.raises(AssertionError):
        audit.align_partial_screenshots([png()] * (1 + extra), count, diagnosis)


def test_pre_action_abort_requires_empty_response_and_matching_initial_image():
    raw = png()
    sample = dict(sample_id='task', status='aborted', total_steps=0, llm_response='',
        terminate_reason='generation_error: Selector halted or request cap reached',
        images=[[base64.b64encode(raw).decode()]])
    assert audit.verify_pre_action_abort(sample, [], 'task', raw) == [2, 2]
    for changed, decisions, image in [
        (dict(sample, llm_response='action'), [], raw),
        (dict(sample, total_steps=1), [], raw),
        (sample, [{}], raw),
        (sample, [], b'different'),
    ]:
        with pytest.raises(AssertionError):
            audit.verify_pre_action_abort(changed, decisions, 'task', image)
