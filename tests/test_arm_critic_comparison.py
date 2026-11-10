import importlib.util
from pathlib import Path

import pytest


def load(name):
    p = Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


critic = load('prepare_arm_critic_comparison')
scaling = load('audit_arm_inference_scaling')


def test_causal_features_ignore_broadcast_future_and_next_action(tmp_path):
    (tmp_path/'image.bin').write_bytes(b'image')
    ref = {'archive_image':'image.bin', 'sha256':'unused-in-input-test'}
    before = dict(prompt='goal and before', response='first action',
                  multimodal_inputs={'images':[ref]}, metadata=dict(intent='goal',
                  step_tool_responses=[dict(tool_name='click', tool_response='ok')]))
    after = dict(prompt='goal and before first action after observation', response='FUTURE_ACTION',
                 multimodal_inputs={'images':[ref]}, metadata=dict(intent='goal'))
    turns = [before, after]
    for variant in ('state_value','pre_action','post_action'):
        expected = critic.causal_inputs(turns,0,tmp_path,15,variant)
        for t in turns:
            t['metadata'].update(reward=1, total_steps=99, terminate_reason='SECRET',
                                 eval='SECRET', num_turns_in_trajectory=99)
            t['reward'] = 1
        assert critic.causal_inputs(turns,0,tmp_path,15,variant) == expected
        assert 'FUTURE_ACTION' not in str(expected)
        assert expected['remaining_turns'] == 15
        for t in turns:
            t['metadata']['reward'] = 0
            t['metadata']['terminate_reason'] = 'OTHER_SECRET'
        assert critic.causal_inputs(turns,0,tmp_path,15,variant) == expected
    assert 'candidate_response' not in critic.causal_inputs(turns,0,tmp_path,15,'state_value')


def test_post_action_requires_receipt_and_next_state(tmp_path):
    (tmp_path/'image.bin').write_bytes(b'image')
    t = dict(prompt='goal', response='unparseable', metadata={'intent':'goal'},
             multimodal_inputs={'images':[{'archive_image':'image.bin','sha256':'unused'}]})
    with pytest.raises(ValueError, match='No observed'):
        critic.causal_inputs([t,t],0,tmp_path,15,'post_action')
    t['metadata']['step_tool_responses'] = [{'tool_response':'failed click'}]
    with pytest.raises(ValueError, match='No observed'):
        critic.causal_inputs([t],0,tmp_path,15,'post_action')


def test_pass_k_retains_invalid_attempts_and_rejects_missing_attempts():
    rows = [dict(task_id='x',attempt=i,valid=i!=4,reward=int(i==0),
                 actor_requests=2,actor_output_tokens=100,steps=2) for i in range(5)]
    result = scaling.screen_summary(rows)
    assert result['pass_at_k'][0]['overall_pass_at_k'] == pytest.approx(.2)
    assert result['pass_at_k'][1]['overall_pass_at_k'] == pytest.approx(.4)
    assert result['pass_at_k'][4]['overall_pass_at_k'] == 1
    assert result['pass_at_k'][4]['expected_random_subset_cost_per_task']['actor_output_tokens'] == 500
    with pytest.raises(ValueError, match='Incomplete or duplicate'):
        scaling.screen_summary(rows[:-1])
    rows[-1]['attempt'] = 0
    with pytest.raises(ValueError, match='Incomplete or duplicate'):
        scaling.screen_summary(rows)
