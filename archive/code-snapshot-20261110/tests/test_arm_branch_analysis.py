from copy import deepcopy
import pytest
from openwebrl.arm_branch_analysis import paired_summary, paired_by_phase


def fixture():
    records=[dict(state='s',candidate=c,repeat=r,valid=True,reward=int(c==1 and r==0))
             for c in range(5) for r in range(3)]
    choices=[dict(state='s',condition='before',execution_repeat=None,teacher_repeat=t,
                  candidate=0,valid=True) for t in range(3)]
    choices += [dict(state='s',condition='after',execution_repeat=r,teacher_repeat=t,
                     candidate=1,valid=True) for r in range(3) for t in range(3)]
    return records,choices


def test_actual_selected_mean_is_not_pass_at_three_or_oracle():
    records,choices=fixture(); result=paired_summary(records,choices)
    assert result['estimates']['before']==0
    assert result['estimates']['after']==pytest.approx(1/3)
    assert result['estimates']['uniform']==pytest.approx(1/15)
    assert result['estimates']['before_after_disagreement']==1
    assert result['estimates']['before_repeat_disagreement']==0
    assert result['state_bootstrap_95_intervals']['difference'] is None


def test_invalid_selected_outcomes_count_zero_and_common_valid_is_paired():
    records,choices=fixture(); records[3]['valid']=False
    result=paired_summary(records,choices)
    assert result['estimates']['after']==0
    assert result['common_valid_selected_pairs']==dict(before=0,after=0,pairs=6,difference=0)
    assert result['by_state'][0]['invalid_continuations']==1


def test_missing_teacher_repeat_excludes_state_instead_of_cherry_picking_calls():
    records,choices=fixture();choices.pop()
    result=paired_summary(records,choices)
    assert result['analyzed_states']==0
    assert result['excluded_states'][0]['reason']=='incomplete_valid_teacher_repeats'


def test_permutation_controls_do_not_enter_selected_success():
    records,choices=fixture()
    choices.append(dict(state='s',condition='before',execution_repeat=None,teacher_repeat='permuted',candidate=1,valid=True))
    result=paired_summary(records,choices)
    assert result['estimates']['before']==0
    assert result['by_state'][0]['permutation_agreement']['before']==0


def test_duplicate_identities_fail_closed_and_bootstrap_uses_states():
    records,choices=fixture()
    with pytest.raises(ValueError): paired_summary(records+records[:1],choices)
    with pytest.raises(ValueError): paired_summary(records,choices+choices[:1])
    second=deepcopy(records); second_choices=deepcopy(choices)
    for row in second+second_choices:row['state']='s2'
    for row in second:row['reward']=int(row['candidate']==0)
    result=paired_summary(records+second,choices+second_choices,bootstrap_samples=200)
    assert result['analyzed_states']==2
    assert result['estimates']['difference']==pytest.approx(-1/3)
    assert result['state_bootstrap_95_intervals']['difference']==[-1,pytest.approx(1/3)]


def test_repair_phase_view_preserves_invalids_and_primary_paired_design():
    records,choices=fixture();records[3]['valid']=False
    new=deepcopy(records);new_choices=deepcopy(choices)
    for r in new+new_choices:r['state']='s2'
    new[3]['valid']=True
    phases=paired_by_phase(records+new,choices+new_choices,{'s':'old','s2':'repaired'},bootstrap_samples=0)
    assert sum(p['continuations'] for p in phases.values())==30
    assert phases['old']['invalid_continuations']==1
    assert phases['old']['estimates']['after']==0
    assert phases['repaired']['estimates']['after']==pytest.approx(1/3)
    assert all('by_state' not in p and 'excluded_states' not in p for p in phases.values())
    primary=paired_summary(records+new,choices+new_choices,bootstrap_samples=0)
    assert primary['analyzed_states']==2 and primary['estimates']['after']==pytest.approx(1/6)
    with pytest.raises(ValueError,match='phase'):paired_by_phase(records,choices,{})
