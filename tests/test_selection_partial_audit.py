"""A timeout may save the next observation before any corresponding selection."""
import importlib.util
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
