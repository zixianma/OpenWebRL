import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_arm_branches import verify_activation_failure


def fixture(tmp_path):
    directory = tmp_path/'attempts'/'attempt-a'
    (directory/'events').mkdir(parents=True)
    log = tmp_path/'scheduler-attempts'/'1'/'state-c0-r0.log'
    log.parent.mkdir(parents=True)
    log.write_text("release = json.loads((self.state_directory/'release.json').read_text())\n"
                   'json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)\n')
    row = dict(state='state', candidate=0, repeat=0, attempt_id='attempt-a',
               valid=False, reward=None, reason='anchor_not_captured', browser_closed=True,
               result_path=str(directory/'result.json'))
    attempt = dict(job_id='1', item_id='state-c0-r0', ended=20)
    release = dict(attempt_directories=[str(directory)], released_unix=19)
    return row, attempt, release, log


def test_failed_activation_is_proven_without_relabeling_the_saved_outcome(tmp_path):
    row, attempt, release, _ = fixture(tmp_path)
    before = dict(row)
    proof = verify_activation_failure(tmp_path, row, attempt, release)
    assert proof['kind'] == 'release_json_read_failed_after_commit'
    assert row == before and not row['valid']


@pytest.mark.parametrize('corruption', ['other_error', 'before_commit', 'wrong_attempt', 'executed_candidate'])
def test_unknown_or_inconsistent_activation_failures_still_fail_the_audit(tmp_path, corruption):
    row, attempt, release, log = fixture(tmp_path)
    if corruption == 'other_error':
        log.write_text('Some other infrastructure failure')
    elif corruption == 'before_commit':
        attempt['ended'] = 18
    elif corruption == 'wrong_attempt':
        release['attempt_directories'] = []
    else:
        (Path(row['result_path']).parent/'events'/'00000001.json').write_text(json.dumps(
            dict(event='forced_response', record=dict(role='candidate'))))
    with pytest.raises(AssertionError):
        verify_activation_failure(tmp_path, row, attempt, release)
