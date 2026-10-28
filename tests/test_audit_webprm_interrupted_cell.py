"""Read-only saved interrupted episodes plus negative in-memory proof fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import audit_webprm_interrupted_cell as m
from openwebrl.webprm_journal_audit import audit_cell_calls
from test_webprm_journal_audit import Fixture

N=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation')
C=N/'reddit-table5-190-v3/campaign';A=C/'attempt-356022'


def read(p):return json.loads(Path(p).read_text())


def saved_inputs(prefix):
    folder=next(p for p in (A/'cells').iterdir() if p.name.startswith(prefix))
    plan=read(folder/'worker-plan.json');episode=next((folder/'episodes/webarena_lite_005').iterdir())
    cohort=read(N/'reddit-table5-190-v4/cohort.json');cohort=cohort.get('cohort',cohort)
    cell=next(c for c in cohort['cells'] if c['task_id']==5 and c['actor']==plan['actor_model'] and c['selector']==plan['selector'])
    paths=dict(result=episode/'result.json',watchdog=episode/'watchdog.json',api=episode/'api.jsonl',rollout=episode/'rollout.json',
        claim=folder/'state-claim.json',launch=folder/'worker-launch.json',instance=folder/'site/instance.json',
        site_closure=folder/'site/closure.json',outer_closure=A/'controller-closure.json',stop_intent=C/'operational-stop-356022.json',
        lease=folder/'lease.json',copy_receipt=folder/'copy/receipt.json',provenance=episode/'provenance.json',
        authentication=episode/'authentication.json',queue_manifest=C/'queue/manifest.json',
        queue_claim=C/'queue/claims'/(m.digest(cell['cell_id'])+'.json'),
        queue_outcome=C/'queue/outcomes'/(m.digest(cell['cell_id'])+'.json'))
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path not in paths.values() and path.name!='storage-state-private.json':
            paths['preserved_'+str(path.relative_to(folder))]=path
    return dict(schema_version=1,protocol=m.PROTOCOL,campaign_root=str(C),cell_folder=str(folder),cell=cell,
        cohort=m.ref(N/'reddit-table5-190-v4/cohort.json'),worker_plan=m.ref(folder/'worker-plan.json'),
        artifacts={k:m.ref(p) for k,p in paths.items()},audit_source_sha256={p:m.sha(m.REPO/p) for p in m.SOURCE_FILES},
        epoch_recovery=m.ref(N/'reddit-operational-repair-v1/epoch-recovery.json'),financial_contract=read(A/'financial-contract.json'))


@unittest.skipUnless((A/'controller-closure.json').exists(),'Preserved closed attempt unavailable')
class InterruptionAuditTests(unittest.TestCase):
    def setUp(self):
        guard=patch('socket.socket.connect',side_effect=AssertionError('Network forbidden'))
        guard.start();self.addCleanup(guard.stop)

    def test_actual_saved_supervisor_failures_produce_only_invalid_no_verdict_candidates(self):
        for prefix,paid,local,pending in [('cffba',17,24,1),('c5f66',38,13,0)]:
            with self.subTest(prefix=prefix):
                inputs=saved_inputs(prefix);before={r['path']:m.sha(r['path']) for r in inputs['artifacts'].values()}
                result=m.verify(inputs);record=result['record']
                self.assertTrue(record['attempted']);self.assertFalse(record['valid'])
                self.assertIsNone(record['success']);self.assertIsNone(record['native_success'])
                self.assertFalse(result['complete_for_disposal']);self.assertIsNone(result['native_replay'])
                self.assertEqual(result['call_audit']['paid_calls'],paid)
                self.assertEqual(result['call_audit']['local_calls'],local)
                self.assertEqual(len(result['unfinished_local_dispatches']),pending)
                self.assertEqual(before,{p:m.sha(p) for p in before})

    def test_current_strict_financial_default_still_rejects_unfinished_dispatch(self):
        x=saved_inputs('cffba');r=read(x['epoch_recovery']['path']);plan=read(x['worker_plan']['path'])
        args=(plan,'webarena_lite_005/'+x['cell']['actor']+'__'+x['cell']['selector'],x['artifacts']['api']['path'],
            r['before']['audit_path'],C/'shared-budget.json.events.jsonl',r['after']['budget_prefix_bytes'],
            r['after']['budget_ledger_sha256'],x['financial_contract'])
        kw=dict(broker_prefix_bytes=r['before']['audit_prefix_bytes'],broker_prefix_sha256=r['before']['audit_sha256'])
        with self.assertRaisesRegex(ValueError,'unfinished broker'):audit_cell_calls(*args,**kw)
        for ids in [('0'*32,),('0'*32,'1'*32),'0'*32]:
            with self.subTest(ids=ids),self.assertRaises(ValueError):audit_cell_calls(*args,**kw,interrupted_local_request_ids=ids)
        # A completed paid request cannot be relabeled as the permitted lost local request.
        events=[json.loads(line) for line in (A/'broker.jsonl').read_text().splitlines()]
        paid=next(e['request_id'] for e in events if e.get('cell_id')==args[1] and e.get('event')=='dispatch_started' and e.get('local') is False)
        with self.assertRaises(ValueError):audit_cell_calls(*args,**kw,interrupted_local_request_ids=(paid,))

    def supervisor_args(self):
        x=saved_inputs('c5f66');a=x['artifacts']
        return [x['cell'],read(x['worker_plan']['path']),*[read(a[k]['path']) for k in
            ('result','watchdog','claim','launch','instance','site_closure','outer_closure','stop_intent')],
            Path(x['cell_folder']),C]

    def test_cannot_promote_interruption_to_valid_or_insert_native_verdict(self):
        for update in (dict(valid=True),dict(success=1),dict(native_success=False),dict(score=0.),dict(actor_model='other')):
            args=self.supervisor_args();args[2].update(update)
            with self.subTest(update=update),self.assertRaises(ValueError):m.verify_supervisor(*args)

    def test_closure_and_claim_cannot_be_borrowed_from_another_cell(self):
        for index,update in [(4,dict(cell_id='other')),(8,dict(remaining_owned_processes={'1':{}})),
                             (9,dict(active_cells=[])),(5,dict(resampling_forbidden=False))]:
            args=self.supervisor_args();args[index].update(update)
            with self.subTest(index=index),self.assertRaises(ValueError):m.verify_supervisor(*args)

    def test_saved_worker_watchdog_must_be_exact_and_owned(self):
        args=self.supervisor_args();args[3]['supervisor_signal']=9
        with self.assertRaises(ValueError):m.verify_supervisor(*args)
        args=self.supervisor_args();args[3]['last_phase']['pid']=1;args[2]['watchdog']=deepcopy(args[3])
        with self.assertRaises(ValueError):m.verify_supervisor(*args)
        for suffix in [['--target','1'],['--execute']]:
            args=self.supervisor_args();args[5]['argv']+=suffix
            with self.assertRaises(ValueError):m.verify_supervisor(*args)

    def test_science_or_audit_source_change_is_not_silently_accepted(self):
        x=saved_inputs('c5f66');x['audit_source_sha256']['scripts/audit_webprm_interrupted_cell.py']='0'*64
        with self.assertRaisesRegex(ValueError,'auditor source'):m.verify(x)
        x=saved_inputs('c5f66');x['financial_contract']['cohort_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'financial contract'):m.verify(x)

    def test_recovery_prefix_hash_change_rejects_without_editing_original_evidence(self):
        x=saved_inputs('c5f66');x['epoch_recovery']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Evidence reference changed'):m.verify(x)

    def test_partial_episode_files_cannot_be_omitted(self):
        x=saved_inputs('c5f66');key=next(k for k in x['artifacts'] if k.endswith('proposal-12.json'))
        del x['artifacts'][key]
        with self.assertRaisesRegex(ValueError,'omitted'):m.verify(x)

    def test_named_evidence_cannot_be_redirected_to_a_different_saved_file(self):
        x=saved_inputs('c5f66');x['artifacts']['api']=x['artifacts']['rollout']
        with self.assertRaisesRegex(ValueError,'Named evidence'):m.verify(x)

    def test_reset_provenance_and_queue_cannot_be_borrowed_or_omitted(self):
        x=saved_inputs('c5f66');paths={k:Path(v['path']) for k,v in x['artifacts'].items()}
        plan=read(x['worker_plan']['path']);cohort=read(x['cohort']['path']);cohort=cohort.get('cohort',cohort)
        real_read=m.read
        for key,update in [('instance',dict(state_path='/tmp/other')),('provenance',dict(reset_id='0'*64)),
                           ('queue_claim',dict(cell_id='other')),('queue_outcome',dict(owner='0'*32))]:
            changed=read(paths[key]);changed.update(update)
            with self.subTest(key=key),patch.object(m,'read',side_effect=lambda path,**kw:changed if Path(path)==paths[key] else real_read(path,**kw)):
                with self.assertRaises(ValueError):m.verify_reset_queue(x,paths,plan,cohort)
        del x['artifacts']['queue_claim']
        with self.assertRaisesRegex(ValueError,'Missing actual interruption'):m.verify(x)

    def test_interrupted_local_never_hides_a_response_error_or_missing_request(self):
        fixture=Fixture();self.addCleanup(fixture.budget.close)
        ident=fixture.call(local=True)
        broker=fixture.broker.read_bytes().splitlines(keepends=True)
        api=fixture.api.read_bytes().splitlines(keepends=True)
        fixture.broker.write_bytes(b''.join(broker[:-1]))
        fixture.api.write_bytes(api[0])
        result=audit_cell_calls(**fixture.bindings(),interrupted_local_request_ids=(ident,))
        self.assertIsNone(result['pairs'][0]['broker']);self.assertIsNone(result['pairs'][0]['response'])
        for suffix in [api[1],b'{"event":"response_error","reservation":null,"error_body":"failure"}\n']:
            fixture.api.write_bytes(api[0]+suffix)
            with self.assertRaisesRegex(ValueError,'contradictory saved terminal'):
                audit_cell_calls(**fixture.bindings(),interrupted_local_request_ids=(ident,))
        fixture.api.write_bytes(b'')
        with self.assertRaisesRegex(ValueError,'no exact API request'):
            audit_cell_calls(**fixture.bindings(),interrupted_local_request_ids=(ident,))

    def test_unfinished_paid_reservation_cannot_use_local_exception(self):
        fixture=Fixture();self.addCleanup(fixture.budget.close)
        ident=fixture.call(local=False,state='unknown')
        fixture.broker.write_bytes(b''.join(fixture.broker.read_bytes().splitlines(keepends=True)[:-1]))
        with self.assertRaisesRegex(ValueError,'unfinished unreserved local'):
            audit_cell_calls(**fixture.bindings(),interrupted_local_request_ids=(ident,))

    def test_candidate_writer_does_not_activate_or_modify_primary(self):
        with tempfile.TemporaryDirectory(prefix='interruption-candidate-',dir='/tmp') as tmp:
            input_path=Path(tmp)/'input.json';input_path.write_text(json.dumps(saved_inputs('c5f66')))
            output=Path(tmp)/'audit.json';report=m.audit(input_path,output)
            self.assertTrue(report['audit_verified']);self.assertEqual(report['activation'],'candidate_pending_independent_review')
            normalized=read(output.with_suffix('.record.json'))
            self.assertEqual(normalized['audit_sha256'],m.sha(output));self.assertIsNone(normalized['success'])


if __name__=='__main__':unittest.main()
