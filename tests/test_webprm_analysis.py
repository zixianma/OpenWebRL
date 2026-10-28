import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from openwebrl.webprm_baselines import ARMS
from openwebrl.webprm_worker import ACTOR, JUDGE, PRICES
from scripts.analyze_webprm_baselines import audit_episodes, load_records


class IndependentEvidenceAuditTests(unittest.TestCase):
    def fixture(self, root):
        plan=dict(pilot_task_ids=['a'],task_file_sha256='task-digest',approval=dict(api_cap_usd=30))
        identity=dict(task_ids=['a'],arms=list(ARMS),task_file_sha256='task-digest',
                      judge=JUDGE,actor=ACTOR,api_usd_cap=30,prices_usd_per_token=PRICES)
        digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        self.save(root/'manifest.json',dict(identity=identity,identity_sha256=digest))
        records=[]
        calls=[]
        for arm in ARMS:
            folder=root/'a'/arm
            folder.mkdir(parents=True)
            row=dict(task_id='a',arm=arm,attempted=True,valid=True,success=1,turns=1,
                     terminal='agent_stopped',identity_sha256=digest)
            records.append(row)
            self.save(folder/'result.json',row)
            self.save(folder/'provenance.json',dict(identity_sha256=digest,judge_model=JUDGE))
            self.save(folder/'rollout.json',[dict(turn=0,selected=dict(action='stop'))])
            self.save(folder/'verdict.json',dict(model=JUDGE,raw='Status: success',parsed=1,terminal='agent_stopped'))
            for name in ('step_00.jpg','terminal.jpg'):
                Image.new('RGB',(8,8)).save(folder/name)
            events=[]
            for model,content in [(ACTOR,'stop'),(JUDGE,'Status: success')]:
                reservation=len(calls)
                usage=dict(prompt_tokens=100,completion_tokens=10)
                actual=usage['prompt_tokens']*PRICES[model][0]+usage['completion_tokens']*PRICES[model][1]
                calls.append(dict(id=reservation,model=model,reserved=actual*2,state='settled',charged=actual,usage=usage))
                events.extend([dict(event='request',local=False,reservation=reservation,payload=dict(model=model)),
                               dict(event='response',reservation=reservation,response=dict(model=model,usage=usage,
                                    choices=[dict(finish_reason='stop',message=dict(content=content))]))])
            (folder/'requests.jsonl').write_text(''.join(json.dumps(event)+'\n' for event in events))
        self.save(root/'results.json',records)
        self.save(root/'complete.json',dict(episodes=6,valid=6))
        self.save(root/'budget.json',dict(cap=30,charged=sum(call['charged'] for call in calls),calls=calls))
        return plan

    def save(self,path,value):
        path.write_text(json.dumps(value))

    def shared_fixture(self, directory):
        prior=directory/'prior'
        current=directory/'current'
        prior.mkdir()
        current.mkdir()
        self.fixture(prior)
        plan=self.fixture(current)
        records=json.loads((current/'results.json').read_text())[:1]
        self.save(current/'results.json',records)
        budget=json.loads((prior/'budget.json').read_text())
        prefix_length=len(budget['calls'])
        current_calls=json.loads((current/'budget.json').read_text())['calls'][:2]
        for call in current_calls:
            call['id']+=prefix_length
        budget['calls'].extend(current_calls)
        budget['charged']=sum(call['charged'] for call in budget['calls'])
        self.save(prior/'budget.json',budget)
        journal=current/'a'/ARMS[0]/'requests.jsonl'
        events=[json.loads(line) for line in journal.read_text().splitlines()]
        for event in events:
            event['reservation']+=prefix_length
        journal.write_text('\n'.join(json.dumps(event) for event in events))
        manifest=json.loads((current/'manifest.json').read_text())
        manifest['config']={'budget_path':str(prior/'budget.json')}
        self.save(current/'manifest.json',manifest)
        return current,prior,plan

    def test_json_array_and_jsonl_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'records.json'
            rows=[dict(task_id='a'),dict(task_id='b')]
            self.save(path,rows)
            self.assertEqual(load_records(path),rows)
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            self.assertEqual(load_records(path),rows)

    def test_complete_evidence_and_budget_are_reconciled(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan=self.fixture(root)
            records,audit=audit_episodes(root,plan)
            self.assertEqual(len(records),6)
            self.assertEqual(audit['status'],'completed')
            self.assertEqual(audit['valid'],6)
            self.assertAlmostEqual(audit['api']['charged_usd'],audit['api']['settled_usd'])
            self.assertEqual(audit['evidence_by_arm'][ARMS[0]]['actor_calls'],1)

    def test_claimed_success_requires_saved_judge_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan=self.fixture(root)
            path=root/'a'/ARMS[0]/'verdict.json'
            verdict=json.loads(path.read_text())
            verdict['parsed']=0
            self.save(path,verdict)
            with self.assertRaisesRegex(ValueError,'does not support'):
                audit_episodes(root,plan)

    def test_truncated_judge_cannot_validate_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan=self.fixture(root)
            path=root/'a'/ARMS[0]/'requests.jsonl'
            events=[json.loads(line) for line in path.read_text().splitlines()]
            events[-1]['response']['choices'][0]['finish_reason']='length'
            path.write_text('\n'.join(json.dumps(event) for event in events))
            with self.assertRaisesRegex(ValueError,'judge response absent'):
                audit_episodes(root,plan)

    def test_budget_total_cannot_be_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan=self.fixture(root)
            path=root/'budget.json'
            budget=json.loads(path.read_text())
            budget['charged']=0
            self.save(path,budget)
            with self.assertRaisesRegex(ValueError,'accounting total differs'):
                audit_episodes(root,plan)

    def test_local_http_error_reconciles_without_creating_paid_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan=self.fixture(root)
            journal=root/'a'/ARMS[0]/'requests.jsonl'
            events=[json.loads(line) for line in journal.read_text().splitlines()]
            events[:0]=[
                dict(event='request',local=True,reservation=None,payload=dict(model='WebShepherd8B')),
                dict(event='response_error',local=True,reservation=None,model='WebShepherd8B',
                     status_code=400,error_body='Input exceeds context')]
            journal.write_text('\n'.join(json.dumps(event) for event in events))
            _,audit=audit_episodes(root,plan)
            self.assertEqual(audit['evidence_by_arm'][ARMS[0]]['http_errors'],1)
            self.assertEqual(audit['api']['ledger_calls'],12)
            events[1]['model']='wrong-model'
            journal.write_text('\n'.join(json.dumps(event) for event in events))
            with self.assertRaisesRegex(ValueError,'matching prior request'):
                audit_episodes(root,plan)

    def test_archived_budget_prefix_excludes_separate_supplemental_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan=self.fixture(root)
            live_path=root/'budget.json'
            snapshot=root/'primary-budget.json'
            snapshot.write_bytes(live_path.read_bytes())
            live=json.loads(live_path.read_text())
            supplemental=dict(live['calls'][0],id=len(live['calls']))
            live['calls'].append(supplemental)
            live['charged']+=supplemental['charged']
            self.save(live_path,live)
            with self.assertRaisesRegex(ValueError,'lacks matching saved response'):
                audit_episodes(root,plan)
            _,audit=audit_episodes(root,plan,budget_snapshot=snapshot)
            self.assertEqual(audit['status'],'completed')
            self.assertEqual(audit['api']['scope'],'archived_snapshot')
            self.assertEqual(audit['api']['ledger_calls'],12)
            self.assertEqual(audit['api']['excluded_live_calls'],1)
            self.assertAlmostEqual(audit['api']['charged_usd'],json.loads(snapshot.read_text())['charged'])

    def test_archived_budget_requires_unchanged_live_prefix_and_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan=self.fixture(root)
            live_path=root/'budget.json'
            snapshot=root/'primary-budget.json'
            snapshot.write_bytes(live_path.read_bytes())
            original=live_path.read_text()
            for changed in ('prefix','cap'):
                with self.subTest(changed=changed):
                    live=json.loads(original)
                    if changed=='prefix':
                        live['calls'][0]['reserved']+=.01
                    else:
                        live['cap']+=1
                    self.save(live_path,live)
                    with self.assertRaisesRegex(ValueError,'not an exact prefix'):
                        audit_episodes(root,plan,budget_snapshot=snapshot)

    def test_unattempted_cell_is_partial_without_complete_estimate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan=self.fixture(root)
            path=root/'results.json'
            records=json.loads(path.read_text())
            records[0].update(attempted=False,valid=False,success=0,error='Deadline reached',error_type='TimeoutError')
            self.save(path,records)
            self.save(root/'a'/ARMS[0]/'result.json',records[0])
            _,audit=audit_episodes(root,plan)
            self.assertEqual(audit['status'],'partial')
            self.assertEqual(audit['attempted_episodes'],5)
            self.assertEqual(audit['missing_episodes'],[dict(task_id='a',arm=ARMS[0])])

    def test_shared_ledger_accounting_never_merges_prior_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            current,prior,plan=self.shared_fixture(Path(directory))
            with self.assertRaisesRegex(ValueError,'lacks matching saved response'):
                audit_episodes(current,plan)
            records,audit=audit_episodes(current,plan,accounting_episodes=[prior])
            self.assertEqual(len(records),1)
            self.assertEqual(audit['status'],'partial')
            self.assertEqual(audit['planned_episodes'],6)
            self.assertEqual(audit['attempted_episodes'],1)
            self.assertEqual(audit['valid'],1)
            self.assertEqual(audit['api']['ledger_calls'],14)
            self.assertEqual(audit['api']['orphan_reservations'],0)
            self.assertEqual(audit['api']['accounting_only_roots'][0]['journals'],6)
            self.assertEqual(audit['evidence_by_arm'][ARMS[0]]['actor_calls'],1)
            self.assertEqual(audit['evidence_by_arm'][ARMS[1]],{})
            cohort_cost=sum(row.get('api_settled_usd',0) for row in audit['evidence_by_arm'].values())
            self.assertAlmostEqual(cohort_cost+audit['api']['accounting_only_settled_usd'],
                                   audit['api']['charged_usd'])

    def test_shared_accounting_rejects_duplicate_reservations(self):
        with tempfile.TemporaryDirectory() as directory:
            current,prior,plan=self.shared_fixture(Path(directory))
            journal=prior/'a'/ARMS[0]/'requests.jsonl'
            events=[json.loads(line) for line in journal.read_text().splitlines()]
            events[0]['reservation']=12
            journal.write_text('\n'.join(json.dumps(event) for event in events))
            with self.assertRaisesRegex(ValueError,'reservation reused'):
                audit_episodes(current,plan,accounting_episodes=[prior])

    def test_shared_accounting_rejects_response_model_and_usage_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            current,prior,plan=self.shared_fixture(Path(directory))
            journal=prior/'a'/ARMS[0]/'requests.jsonl'
            original=journal.read_text()
            for change in ('model','usage'):
                with self.subTest(change=change):
                    events=[json.loads(line) for line in original.splitlines()]
                    if change=='model':
                        events[1]['response']['model']='different-model'
                    else:
                        events[1]['response']['usage']['completion_tokens']+=1
                    journal.write_text('\n'.join(json.dumps(event) for event in events))
                    with self.assertRaisesRegex(ValueError,'response model|matching saved response'):
                        audit_episodes(current,plan,accounting_episodes=[prior])

    def test_shared_accounting_rejects_repeated_or_current_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            current,prior,plan=self.shared_fixture(Path(directory))
            for roots in ([prior,prior/'.'],[current]):
                with self.subTest(roots=roots):
                    with self.assertRaisesRegex(ValueError,'root supplied more than once'):
                        audit_episodes(current,plan,accounting_episodes=roots)


if __name__=='__main__':
    unittest.main()
