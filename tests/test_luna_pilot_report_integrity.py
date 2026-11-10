import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_luna_qwen_inference import report, require_comparison_integrity


def block(root, value):
    directory = root / 'supervisor'
    directory.mkdir(exist_ok=True)
    (directory / 'scientific-integrity-block.json').write_text(json.dumps(value))


def test_execution_block_prevents_original_report_output(tmp_path):
    block(tmp_path, {'blocking_full_comparison': True})
    output = tmp_path / 'report'
    with pytest.raises(ValueError, match='scientific-integrity'):
        report(tmp_path, output)
    assert not output.exists()


def test_execution_block_prevents_combined_sft_report(tmp_path, monkeypatch):
    import run_luna_qwen_sft_extension as extension
    block(tmp_path, {'blocking_full_comparison': True})
    monkeypatch.setattr(extension, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='scientific-integrity'):
        extension.combine()


def test_absent_or_resolved_block_allows_normal_validation(tmp_path):
    require_comparison_integrity(tmp_path)
    block(tmp_path, {'blocking_full_comparison': False})
    require_comparison_integrity(tmp_path)


def test_malformed_integrity_state_does_not_allow_a_report(tmp_path):
    block(tmp_path, {})
    (tmp_path / 'supervisor/scientific-integrity-block.json').write_text('{')
    with pytest.raises(json.JSONDecodeError):
        report(tmp_path, tmp_path / 'report')
