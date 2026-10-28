"""Saved synthetic startup accounting, using scripted replies without a provider."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from openwebrl.webprm_broker import payload_sha256
from openwebrl.webprm_reddit_pilot_preflight import (NATIVE_FUZZY_PREFLIGHT, validate_native_fuzzy,
    GUARDED_FUZZY_PREFLIGHT, validate_guarded_fuzzy)
from openwebrl.webprm_webarena import EvaluationNotReady, digest
from scripts.analyze_webprm_webarena_pilot import audit, proposal_digest, JUDGE, PRICES, V3_PROFILE, V3_RESOURCES, V3_LIMITS


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_saved_fuzzy_preflight(campaign, attempt, plan, *, text='correct', guarded=False):
    """Add one metered-shaped scripted startup call to an offline campaign."""
    context_path=attempt/'execution-context.json';context=json.loads(context_path.read_text())
    context['proposal']['native_fuzzy_preflight']=dict(GUARDED_FUZZY_PREFLIGHT if guarded else NATIVE_FUZZY_PREFLIGHT)
    context['proposal_sha256']=proposal_digest(context['proposal'])
    context['approval']['proposal_sha256']=context['proposal_sha256'];put(context_path,context)
    def client(config, payload, journal, *, purpose):
        ledger_path=campaign/'shared-budget.json'
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(cap=15,charged=0,calls=[])
        reservation=len(ledger['calls']);usage=dict(prompt_tokens=120,completion_tokens=2)
        rates=PRICES[JUDGE];cost=120*rates[0]+2*rates[1]
        ledger['calls'].append(dict(id=reservation,model=JUDGE,state='settled',reserved=.1,charged=cost,usage=usage))
        ledger['charged']+=cost;put(ledger_path,ledger)
        prediction=payload['messages'][1]['content'].split('student answer: ',1)[1].split('\nConclude',1)[0]
        reply=('incorrect' if prediction in ('red square','1') else text) if guarded else text
        response=dict(model=JUDGE,choices=[dict(finish_reason='stop',message=dict(role='assistant',content=reply))],usage=usage)
        journal_path=attempt/'native-fuzzy-preflight-api.jsonl'
        rows=[dict(event='request',local=False,reservation=reservation,purpose=purpose,payload=payload),
              dict(event='response',reservation=reservation,response=response)]
        with journal_path.open('a' if guarded else 'w') as stream:
            stream.write(''.join(json.dumps(row)+'\n' for row in rows))
        meta=dict(request_id='synthetic-startup-'+str(reservation),identity_sha256=digest(plan),cell_id='native_fuzzy_preflight',
            model=JUDGE,purpose=purpose,local=False,payload_sha256=payload_sha256(payload))
        receipt=dict(meta,schema_version=1,budget_owner='webprm_worker.completion',reservation_id=reservation,
            reservation_state='settled',usage=usage,journal_sha256=sha(journal_path),
            budget_ledger_sha256=sha(ledger_path),policy_sha256='d'*64)
        broker_path=attempt/'broker.jsonl'
        separator='\n' if broker_path.exists() and not broker_path.read_bytes().endswith(b'\n') else ''
        with broker_path.open('a') as stream:
            stream.write(separator)
            for row in [dict(meta,event='dispatch_started'),dict(receipt,event='dispatch_completed')]:
                stream.write(json.dumps(row)+'\n')
        return dict(response,_webprm_broker=receipt)
    client._webprm_broker_client=True;client._webprm_budget_owner='webprm_worker.completion'
    config=dict(plan,authorization_status='approved',identity_sha256=digest(plan),
                budget_path=str(campaign/'shared-budget.json'))
    # Freeze time only for constructing historical offline evidence; audit must
    # replay old receipts after their allocation deadlines without inference.
    with patch('openwebrl.webprm_webarena_fuzzy.time.time',return_value=plan['deadline']-10):
        return (validate_guarded_fuzzy if guarded else validate_native_fuzzy)(config,client,attempt)


class FuzzyAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory=TemporaryDirectory(prefix='webprm-fuzzy-audit-',dir='/tmp')
        self.addCleanup(self.directory.cleanup)
        self.campaign=Path(self.directory.name);self.attempt=self.campaign/'attempt-synthetic'
        from openwebrl.webprm_webarena_worker import protocol_definition,NA_TURN_LIMIT
        source='/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter'
        self.plan=dict(arbiter_source=source,turn_limit_protocol=NA_TURN_LIMIT,
            fuzzy_enabled=True,fuzzy_model=JUDGE,fuzzy_protocol='webarena_fuzzy_gpt4o_20240513_v1',
            fuzzy_model_change_approved=True,fuzzy_max_input_tokens=8192,fuzzy_max_calls=1,deadline=1800000000,
            protocol='webarena_table5_viewer_reconstruction_v3',protocol_contract=protocol_definition(source,turn_limit=NA_TURN_LIMIT))
        proposal=dict(resource_profile=V3_PROFILE,resources=V3_RESOURCES,limits=V3_LIMITS,
                      native_fuzzy_preflight=dict(NATIVE_FUZZY_PREFLIGHT))
        approval=dict(status='approved',resource_profile=V3_PROFILE,resources=V3_RESOURCES,limits=V3_LIMITS,
            turn_limit_protocol=NA_TURN_LIMIT,turn_limit_protocol_approved=True,proposal_sha256=proposal_digest(proposal))
        put(self.attempt/'execution-context.json',dict(proposal=proposal,approval=approval,proposal_sha256=proposal_digest(proposal)))
        put(self.attempt/'worker-plan.json',self.plan)
        put(self.attempt/'controller-closure.json',dict(remaining_owned_processes={}))
        network=patch('socket.socket.connect',side_effect=AssertionError('Network forbidden'))
        network.start();self.addCleanup(network.stop)

    def test_settled_startup_is_counted_separately_from_all_primary_judges(self):
        add_saved_fuzzy_preflight(self.campaign,self.attempt,self.plan)
        put(self.attempt/'worker-launch-no_search.json',dict(selector='no_search'))
        before={str(p):sha(p) for p in self.campaign.rglob('*') if p.is_file()}
        result=audit(self.campaign)
        self.assertEqual(result['requests']['paid_dispatches'],1)
        self.assertEqual(result['requests']['journal_paid_requests'],1)
        self.assertEqual(result['requests']['paid_cap'],575)
        self.assertEqual(result['api']['reservations'],1)
        self.assertAlmostEqual(result['api']['settled_usd'],120*5/1e6+2*15/1e6)
        self.assertEqual(result['native_fuzzy_preflights']['attempt-synthetic']['status'],'passed')
        self.assertEqual(result['totals']['valid'],0)
        self.assertEqual(result['rows'][0]['call_evidence']['by_group']['native_judge']['requests'],0)
        self.assertEqual(before,{str(p):sha(p) for p in self.campaign.rglob('*') if p.is_file()})

    def test_failed_startup_remains_auditable_but_cannot_admit_primary(self):
        with self.assertRaisesRegex(EvaluationNotReady,'expected verdict'):
            add_saved_fuzzy_preflight(self.campaign,self.attempt,self.plan,text='incorrect')
        result=audit(self.campaign)
        self.assertEqual(result['api']['reservations'],1)
        self.assertEqual(result['requests']['paid_dispatches'],1)
        self.assertEqual(result['native_fuzzy_preflights']['attempt-synthetic']['status'],'failed_or_incomplete')
        put(self.attempt/'worker-launch-no_search.json',dict(selector='no_search'))
        with self.assertRaisesRegex(ValueError,'lacks passed native fuzzy'):
            audit(self.campaign)

    def test_startup_before_gate_is_auditable_but_missing_pass_blocks_primary(self):
        self.assertEqual(audit(self.campaign)['native_fuzzy_preflights']['attempt-synthetic']['status'],'not_reached')
        put(self.attempt/'worker-launch-no_search.json',dict(selector='no_search'))
        with self.assertRaisesRegex(ValueError,'lacks passed native fuzzy'):
            audit(self.campaign)

    def test_tampered_synthetic_fixture_cannot_retain_a_pass(self):
        add_saved_fuzzy_preflight(self.campaign,self.attempt,self.plan)
        path=self.attempt/'native-fuzzy-preflight/input.json';inputs=json.loads(path.read_text())
        inputs['fixture']['reference']='arbitrary changed answer';put(path,inputs)
        proof_path=path.parent/'passed.json';proof=json.loads(proof_path.read_text())
        proof['input_sha256']=sha(path);put(proof_path,proof)
        with self.assertRaisesRegex(ValueError,'proof is incomplete or changed'):
            audit(self.campaign)

    def test_paid_startup_journal_cannot_be_omitted_from_reconciliation(self):
        add_saved_fuzzy_preflight(self.campaign,self.attempt,self.plan)
        path=self.attempt/'native-fuzzy-preflight-api.jsonl';path.rename(path.with_suffix('.preserved'))
        with self.assertRaisesRegex(ValueError,'proof is incomplete or changed'):
            audit(self.campaign)

    def test_startup_evidence_cannot_lose_its_frozen_plan(self):
        add_saved_fuzzy_preflight(self.campaign,self.attempt,self.plan)
        path=self.attempt/'worker-plan.json';path.rename(path.with_suffix('.preserved'))
        with self.assertRaisesRegex(ValueError,'Startup judge evidence lacks frozen worker plan'):
            audit(self.campaign)


if __name__=='__main__':
    unittest.main()
