import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('sft_review',
    Path(__file__).resolve().parents[1] / 'scripts/render_sft_decision_review.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_streamed_review_preserves_unicode_and_escapes_script_end(tmp_path, monkeypatch):
    payload = {'audit': {'completed_results': 40, 'all40_evidence_verified': False},
               'text': '日本語 café </script><script>alert(1)</script>'}
    monkeypatch.setattr(review, 'read', lambda path: {'task_ids': ['fixture']})
    monkeypatch.setattr(review, 'build', lambda *args: payload)
    output = tmp_path / 'review.html'
    receipt = review.render(output)
    raw = output.read_bytes()
    prefix, suffix = (review.REPO / 'scripts/templates/sft_decision_review.html').read_text().split('__REVIEW_DATA__')
    encoded = raw.decode('utf-8')[len(prefix):-len(suffix)]
    assert json.loads(encoded) == payload
    assert '<' not in encoded
    assert receipt['sha256'] == hashlib.sha256(raw).hexdigest()
    assert receipt['bytes'] == len(raw)
    assert json.loads(output.with_suffix('.receipt.json').read_text()) == receipt


def test_failed_serialization_preserves_previous_review_and_receipt(tmp_path, monkeypatch):
    output = tmp_path / 'review.html'
    output.write_text('previous complete review')
    receipt = output.with_suffix('.receipt.json')
    receipt.write_text('previous receipt')
    monkeypatch.setattr(review, 'read', lambda path: {'task_ids': ['fixture']})
    monkeypatch.setattr(review, 'build', lambda *args: {'first': 'written', 'bad': object()})
    with pytest.raises(TypeError):
        review.render(output)
    assert output.read_text() == 'previous complete review'
    assert receipt.read_text() == 'previous receipt'
