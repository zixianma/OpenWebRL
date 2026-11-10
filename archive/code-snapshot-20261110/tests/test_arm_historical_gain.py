import asyncio
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location('gain_audit',Path(__file__).resolve().parents[1]/'scripts/audit_arm_historical_gain.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fixture():
    candidate=dict(thought='Inspect the page',action='<tool_call>{"name":"go_back","arguments":{}}</tool_call>')
    row=dict(task_id='task',turn=0,candidates=[candidate]*5,candidate_seeds=[audit.candidate_seed(42,'task',0,i) for i in range(5)],
             selected_index=2,selector_raw='{"selection": 3}',fallback=None,screenshot_sha256=hashlib.sha256(b'image').hexdigest())
    artifact=dict(metadata=dict(messages=[dict(role='assistant',content='<think>Inspect the page</think>'+candidate['action'])],
                                full_image_list=[base64.b64encode(b'image').decode()],turn_index=0))
    return row,artifact


def test_audit_detects_index_seed_and_execution_corruption():
    row,artifact=fixture();assert not audit.check_trace([row],artifact,42,5)['errors']
    row['selector_raw']='{"selection": 2}'
    row['candidate_seeds'][0]+=1
    artifact['metadata']['messages'][0]['content']='<think>Inspect the page</think><tool_call>{"name":"done"}</tool_call>'
    errors=audit.check_trace([row],artifact,42,5)['errors']
    assert set(errors)=={'candidate_seeds','selector_index_mapping','chosen_action_vs_assistant_history'}


def test_old_interrupted_prefix_is_separate_from_committed_history():
    rows=[dict(turn=i,attempt='old') for i in range(2)]+[dict(turn=i,attempt='final') for i in range(3)]
    assert audit.latest_trace_segment(rows)==rows[2:]
    assert len(rows)==5


def test_recorded_seed_does_not_imply_output_reproducibility():
    row,_=fixture();other=json.loads(json.dumps(row));other['candidates'][0]['thought']='Different draw'
    row['prompt_sha256']=other['prompt_sha256']='same'
    x=audit.compare_first({'task':row},{'task':other})
    assert x['identical_first_request_inputs']==x['identical_inputs_different_candidate0']==1


def test_actual_prompt_guard_fails_before_actor_or_selector_call(tmp_path,monkeypatch):
    import openwebrl
    from openwebrl.arm_controlled_inference import MeteredSelector
    monkeypatch.setattr(openwebrl,'generate_browser',SimpleNamespace(),raising=False)
    class Selector:
        async def __call__(self,**kwargs):
            raise AssertionError('No actor or selector call should occur')
    meter=MeteredSelector(Selector(),tmp_path,system_policy='Required policy')
    with pytest.raises(ValueError,match='omitted'):
        asyncio.run(meter(infer=None,input_text='Empty system block'))
    assert meter.requests==0 and not list(tmp_path.iterdir())


def test_corrected_cohort_keys_cannot_reuse_blank_policy_results():
    from openwebrl.arm_paired_inference import schedule,REVISION,PROTOCOL,POLICY_SHA256
    row=schedule(['task'])[0]
    assert PROTOCOL['actor_system_policy_sha256']==POLICY_SHA256
    for a in row['attempts']:
        prior=hashlib.sha256((REVISION+'/task/'+a['mode']).encode()).hexdigest()
        assert a['key']!=prior
