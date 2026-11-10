import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from watch_reasoning_actors_full300 import receipt_health,state_key


def test_unknown_error_wakes_owner_without_being_mistaken_for_stalled_request(tmp_path):
    path=tmp_path/'actor-00001.json'
    row=dict(status='usage_unknown',error_type='InternalServerError',http_status=503,
        started_epoch=1,request=dict(model='gpt-6.1-sol',reasoning={'effort':'high'},max_output_tokens=4096))
    path.write_text(json.dumps(row))
    health=receipt_health([path],'gpt-6.1-sol',1000)
    assert health['error_count']==1 and health['errors']=={'503':1}
    assert health['overdue_inflight']==[] and health['protocol_violations']==[]
    snapshot=dict(pools={'sol61_high':dict(job_id='1',state='RUNNING',controller={},halt={},workers=[],recorded_episodes=1)},
        errors=[],judge_halt={},verified_complete=False)
    before=state_key(snapshot)
    snapshot['pools']['sol61_high']['api_health']=health
    assert state_key(snapshot)!=before
    row.update(status='reserved',error_type=None,http_status=None)
    path.write_text(json.dumps(row))
    assert receipt_health([path],'gpt-6.1-sol',1000)['overdue_inflight']==[path.name]


def test_returned_effort_mismatch_alerts_even_when_request_was_correct(tmp_path):
    path=tmp_path/'actor-00001.json'
    path.write_text(json.dumps(dict(status='received',request=dict(model='gpt-6-luna',
        reasoning={'effort':'high'},max_output_tokens=4096),response=dict(model='gpt-6-luna',
        reasoning={'effort':'medium'},status='completed'))))
    assert receipt_health([path],'gpt-6-luna',1000)['protocol_violations']==[path.name]
