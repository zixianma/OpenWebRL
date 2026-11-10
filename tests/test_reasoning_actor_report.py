from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_reasoning_actors_full300 import require_final,require_same,verify_terminal_evidence,primary_cost_bounds
from audit_reasoning_actors_full300 import receipt_accounting
from openwebrl.luna_qwen_metrics import digest
from openwebrl.reasoning_actor_policy import metric_prices


def fixture():
    proposal={'scientific':'fixed'};execution={'source':'frozen'}
    audit=dict(collection_verified=True,scheduler_accounting_verified=True,api_accounting_verified=True,
        wandb_verified=True,episodes_by_mode={'luna_high':300,'sol61_high':300},
        proposal_sha256=digest(proposal),execution_sha256=digest(execution),
        primary_record_inventory_sha256='hash',errors=[])
    return audit,proposal,execution


def test_no_partial_collection_or_provisional_accounting_can_generate_final_plots():
    audit,p,e=fixture();require_final(audit,p,e)
    for change in [dict(collection_verified=False),dict(scheduler_accounting_verified=False),
        dict(api_accounting_verified=False),dict(wandb_verified=False),
        dict(episodes_by_mode={'luna_high':300,'sol61_high':299}),dict(errors=['mismatch']),
        dict(execution_sha256='changed')]:
        with pytest.raises(ValueError,match='Final'):require_final(dict(audit,**change),p,e)


def test_unknown_usage_is_not_zero_and_cost_disagreement_fails():
    with pytest.raises(ValueError):require_same(None,0)
    with pytest.raises(ValueError):require_same({'cost':3.}, {'cost':2.})
    require_same({'cost':3.00000000001},{'cost':3.})


def test_interrupted_request_reservation_remains_charged_and_ids_must_reconcile(tmp_path):
    receipt=tmp_path/'actor-00001.json'
    receipt.write_text(json.dumps(dict(role='actor',reserved_usd=.1,request=dict(model='gpt-6.1-sol',
        reasoning={'effort':'high'},max_output_tokens=4096,service_tier='default',tool_choice='required',
        parallel_tool_calls=False,store=False))))
    args=([receipt],'gpt-6.1-sol','actor',metric_prices('gpt-6.1-sol'),
        dict(calls=1,charged_or_reserved_usd=.1),dict(max_calls=9900,max_usd=100))
    result=receipt_accounting(*args)
    assert result['unknown_usage']==1 and result['charged_or_reserved_usd']==.1
    changed=list(args);changed[4]=dict(calls=2,charged_or_reserved_usd=.1)
    with pytest.raises(ValueError,match='ledger'):receipt_accounting(*changed)
    changed=list(args);changed[5]=dict(max_calls=9900,max_usd=.09)
    with pytest.raises(ValueError,match='ledger'):receipt_accounting(*changed)


def test_primary_plot_bounds_include_unknown_usage_without_counting_judge(tmp_path):
    requests=tmp_path/'requests';requests.mkdir()
    (requests/'actor-00001.json').write_text(json.dumps(dict(role='actor',reserved_usd=.16)))
    (requests/'judge-00001.json').write_text(json.dumps(dict(role='judge',reserved_usd=99.)))
    rows=[dict(artifact=str(tmp_path/'result.json'),model_usage_missing_requests=1)]
    bounds=primary_cost_bounds(rows,metric_prices('gpt-6.1-sol'))
    assert bounds['lower_mean_usd']==0 and bounds['conservative_upper_mean_usd']==.16
    assert bounds['unknown_calls']==1
    (requests/'actor-00001.json').write_text(json.dumps(dict(role='actor')))
    with pytest.raises(ValueError,match='reservation'):primary_cost_bounds(rows,metric_prices('gpt-6.1-sol'))


def test_verdict_must_match_receipt_score_and_exact_final_image(tmp_path):
    import base64
    (tmp_path/'requests').mkdir()
    (tmp_path/'final.png').write_bytes(b'final image')
    text='Status: success'
    receipt=dict(response={'choices':[{'message':{'content':text}}]},request={'messages':[
        {'content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(b'final image').decode()}}]}]})
    (tmp_path/'requests/judge-00001.json').write_text(json.dumps(receipt))
    row=dict(artifact=str(tmp_path/'result.json'),valid=True,reward=1.)
    outcome=dict(status='Status.COMPLETED',metadata={'reward':{'judge_text':text,'judge':1.}})
    verify_terminal_evidence(row,outcome)
    with pytest.raises(ValueError,match='verdict'):verify_terminal_evidence(dict(row,reward=0.),outcome)
    (tmp_path/'final.png').write_bytes(b'different image')
    with pytest.raises(ValueError,match='screenshot'):verify_terminal_evidence(row,outcome)
