import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location('terminal_recovery',
    Path(__file__).resolve().parents[1] / 'scripts/rejudge_sft_decision_terminal.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_recovery_between_cohorts_preserves_partial_summary(tmp_path):
    corrected = {'complete': True, 'attempted': 10, 'successes': 4}
    folder = tmp_path / 'kev-0.8b'
    folder.mkdir()
    (folder / 'summary.json').write_text(json.dumps(corrected))
    preserved = {'complete': False, 'attempted': 0, 'status': 'not_started'}
    overall = {'complete': False, 'verified_complete': False,
               'modes': {'kev-27b': preserved, 'kev-0.8b': {'attempted': 7}}}
    before = json.dumps(overall)
    summaries = recovery.mode_summaries(tmp_path, overall)
    assert summaries['kev-0.8b'] == corrected
    assert summaries['kev-27b'] == preserved
    assert summaries['sft']['complete'] is False
    assert json.dumps(overall) == before
