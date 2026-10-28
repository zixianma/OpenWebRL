"""Reporting fixtures only; no primary runtime evidence is modified."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import collect_webprm_table5_campaign as c

N=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation')


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')
    return ref(path)


def ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def read(path):return json.loads(path.read_text())


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/tmp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.campaign=self.root/'campaign';self.campaign.mkdir()
        self.contract=dict(fixture='fixed reporting-only protocol')
        self.cohort=c.make_cohort(read(c.TASKS));self.cohort['protocol_sha256']=c.digest(self.contract)
        self.cohort_path=self.root/'cohort.json';save(self.cohort_path,self.cohort)
        actual=read(N/'reddit-table5-190-v4/proposal.json')
        source=next(Path(p) for p in actual['source_sha256'] if p.endswith('/scripts/audit_webprm_table5_cell.py'))
        self.source=self.root/'frozen/scripts/audit_webprm_table5_cell.py';self.source.parent.mkdir(parents=True)
        self.source.write_bytes(source.read_bytes());self.assertEqual(ref(self.source)['sha256'],c.ORIGINAL)
        self.cell=next(x for x in self.cohort['cells'] if x['domain']=='reddit')

    def fixture(self,cell=None,*,rejected=False):
        cell=cell or self.cell
        folder=self.campaign/'attempt-42/cells'/hashlib.sha256(cell['cell_id'].encode()).hexdigest()
        folder.mkdir(parents=True)
        plan=dict(protocol_contract=self.contract,task_ids=[cell['task_id']],actor_model=cell['actor'],
            selector=cell['selector'],code_sha256={str(self.source):ref(self.source)['sha256']})
        planref=save(folder/'worker-plan.json',plan)
        verdict=dict(valid=True,success=False,score=0.,native_success=True,native_score=1.,
            scoring_amendments=[dict(rule='fixture_preserved_amendment')],evaluator_protocol='fixture')
        result=dict(verdict,attempted=True,identity_sha256=c.digest(plan),actor_model=cell['actor'],
            selector=cell['selector'],task_id=f"webarena_lite_{cell['task_id']:03d}")
        episode=folder/'episodes/cell';episode.mkdir(parents=True)
        refs={name:save(episode/(name+'.json'),value) for name,value in dict(result=result,verdict=verdict,
            outcome=verdict,watchdog={},api=[],site_closure={},trajectory=[],rollout=[],evaluator_inputs={}).items()}
        refs['closure']=save(folder/'closure.json',dict(cell_id=cell['cell_id'],broker_cell_inflight=0,
            worker_remaining_processes={},service_remaining_processes={}))
        inputs=dict(schema_version=1,campaign_root=str(self.campaign),cell=cell,cohort=ref(self.cohort_path),
            worker_plan=planref,source_sha256=plan['code_sha256'],artifacts=refs)
        boundary=save(folder/'journal.json',{})
        prefix=dict(prefix_bytes=(folder/'journal.json').stat().st_size,prefix_sha256=boundary['sha256'])
        inputs.update(budget=dict(prefix,event_path=boundary['path']),broker=dict(prefix,audit_path=boundary['path']))
        inp=save(folder/'audit-input.json',inputs)
        record=dict(cell_id=cell['cell_id'],released_task_id=cell['task_id'],actor_model=cell['actor'],selector=cell['selector'],
            domain=cell['domain'],task_config_sha256=cell['task_config_sha256'],tasks_sha256=self.cohort['source_sha256'],
            cohort_sha256=c.digest(self.cohort),protocol_sha256=self.cohort['protocol_sha256'],audit_verified=True,
            terminal=True,attempted=True,valid=True,success=0,native_success=1,
            effective_semantic_status='valid',native_semantic_status='invalid')
        audit=dict(schema_version=1,protocol=c.AUDIT,input_sha256=inp['sha256'],artifact_refs=refs,
            evidence_manifest_sha256=c.digest(refs),audit_verified=not rejected,complete_for_disposal=not rejected,
            errors=deepcopy(c.REJECTION) if rejected else [])
        audit.update(native_replay=dict(verified=True),call_audit=dict(verified=True,cell_unknown_reservations=0,
            **{kind+'_'+key:value for kind in ('budget','broker') for key,value in prefix.items()}))
        if not rejected:audit['record']=record
        original=save(folder/'audit.json',audit)
        if not rejected:save(folder/'audit.record.json',dict(record,audit_sha256=original['sha256']))
        save(folder/'cell-outcome.json',dict(cell_id=cell['cell_id'],result_path=refs['result']['path'],
            result_sha256=refs['result']['sha256'],attempted=True,valid=True,success=0,native_success=True,
            audit_verified=False,audit_status='rejected' if rejected else 'verified'))
        return folder,record

    def erratum(self,folder,record):
        original=read(folder/'audit.json')
        audit=dict(original,protocol=c.ERRATUM,audit_verified=True,errors=[],record=record,
            original_audit_reference=ref(folder/'audit.json'),original_audit_verified=False,
            original_rejection_reproduced=True,erratum=dict(original_auditor_sha256=c.ORIGINAL,
                corrected_source_reference=ref(N/'reddit-empty-dispatch-erratum-v1/corrected-auditor.py'),
                implementation_reference=ref(N/'reddit-empty-dispatch-erratum-v1/audit-erratum.py'),
                replaced_functions=['audit_empty_native_dispatch','audit_rollout'],primary_bytes_changed=False,primary_rerun=False),
            reconstructed_rejections=[dict(turn=3,native_conversion_rejection_reproduced=True)])
        effective=save(folder/'supplement.json',audit)
        normalized=save(folder/'supplement.record.json',dict(record,audit_sha256=effective['sha256'],
            audit_erratum_protocol=c.ERRATUM,original_audit_sha256=ref(folder/'audit.json')['sha256'],
            corrected_audit_source_sha256=c.CORRECTED))
        pointer=self.campaign/'audit-replacements'/(folder.name+'.json')
        save(pointer,dict(cell_id=record['cell_id'],protocol=c.ERRATUM,original_audit=ref(folder/'audit.json'),
            original_outcome=ref(folder/'cell-outcome.json'),effective_audit=effective,effective_record=normalized,
            original_bytes_preserved=True,primary_rerun=False,raw_success=1,effective_success=0))
        return pointer

    def collect(self):return c.collect(self.campaign,self.cohort_path,bootstrap_draws=100)

    def retained_erratum(self):
        folder,record=self.fixture(rejected=True);attempt=folder.parents[1]
        scratch=self.root/'node-local-recorded-only'
        claim=save(folder/'state-claim.json',dict(cell_id=record['cell_id'],campaign_root=str(self.campaign),
            scratch_root=str(scratch),state_path=str(scratch/'states'/folder.name),destination_absent_at_claim=True))
        inputs=read(folder/'audit-input.json');inputs['artifacts']['state_claim']=claim
        inp=save(folder/'audit-input.json',inputs)
        original=read(folder/'audit.json');original.update(input_sha256=inp['sha256'],
            artifact_refs=inputs['artifacts'],evidence_manifest_sha256=c.digest(inputs['artifacts']))
        save(folder/'audit.json',original)
        pointer=self.erratum(folder,record);value=read(pointer)
        value['retained_state']=dict(schema_version=1,status='retained_no_disposal_claim',disposal_performed=False,
            state_claim=claim,cell_closure=ref(folder/'closure.json'),
            attempt_closure=save(attempt/'controller-closure.json',dict(remaining_owned_processes={})),
            attempt_storage=save(attempt/'storage-final.json',dict(scratch=str(scratch),apparent_bytes=123)))
        save(pointer,value);return folder,pointer

    def test_verified_normalized_record_overrides_false_mechanical_flag_only(self):
        folder,_=self.fixture();before={str(p):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
        report=self.collect()
        self.assertEqual(report['full_table5']['counts']['planned'],1550)
        self.assertEqual(report['bounded_scope']['counts']['planned'],190)
        self.assertEqual(report['bounded_scope']['counts']['recorded'],1)
        self.assertEqual(report['bounded_scope']['counts']['effective_successes'],0)
        self.assertEqual(report['bounded_scope']['counts']['raw_native_positive_verdicts'],1)
        self.assertTrue(report['full_table5']['performance_rates_withheld'])
        self.assertTrue(report['bounded_scope']['performance_rates_withheld'])
        self.assertEqual(before,{str(p):p.read_bytes() for p in folder.rglob('*') if p.is_file()})

    def test_exact_additive_replacement_keeps_original_rejection_and_primary_verdicts(self):
        folder,record=self.fixture(rejected=True);self.erratum(folder,record)
        original=(folder/'audit.json').read_bytes();report=self.collect()
        self.assertEqual(report['replacement_count'],1)
        self.assertEqual(report['normalized_records'][0]['success'],0)
        self.assertEqual(report['normalized_records'][0]['native_success'],1)
        self.assertEqual((folder/'audit.json').read_bytes(),original)
        self.assertFalse((folder/'audit.record.json').exists())

    def test_unreplaced_rejection_never_uses_mechanically_valid_result(self):
        self.fixture(rejected=True);report=self.collect()
        self.assertEqual(report['normalized_records'],[])
        self.assertEqual(report['dispositions'][0]['status'],'audit_rejected')
        self.assertEqual(report['full_table5']['counts']['missing'],1550)

    def test_original_outcome_pointer_hash_cannot_be_rebound_silently(self):
        folder,record=self.fixture(rejected=True);self.erratum(folder,record)
        value=read(folder/'cell-outcome.json');value['success']=1;save(folder/'cell-outcome.json',value)
        with self.assertRaisesRegex(ValueError,'reference changed'):self.collect()

    def test_normalized_record_cannot_override_the_audit_body(self):
        folder,_=self.fixture();value=read(folder/'audit.record.json');value['success']=1
        save(folder/'audit.record.json',value)
        with self.assertRaisesRegex(ValueError,'Normalized record'):self.collect()

    def test_task_protocol_or_cell_id_forgery_rejected_even_with_rehashed_audit(self):
        folder,_=self.fixture();audit=read(folder/'audit.json');audit['record']['protocol_sha256']='a'*64
        ref_=save(folder/'audit.json',audit);save(folder/'audit.record.json',dict(audit['record'],audit_sha256=ref_['sha256']))
        with self.assertRaisesRegex(ValueError,'protocol/cohort/task'):self.collect()

    def test_source_or_primary_artifact_change_rejects(self):
        folder,_=self.fixture();self.source.write_bytes(self.source.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'reference changed'):self.collect()

    def test_primary_native_verdict_change_is_rejected_without_rewriting_it(self):
        folder,_=self.fixture();path=folder/'episodes/cell/verdict.json'
        verdict=read(path);verdict.update(success=True,score=1.);save(path,verdict)
        with self.assertRaisesRegex(ValueError,'reference changed'):self.collect()
        self.assertTrue(read(path)['success'])

    def test_changed_journal_prefix_rejects_but_new_other_cell_suffix_is_allowed(self):
        folder,_=self.fixture();journal=folder/'journal.json'
        original=journal.read_bytes();journal.write_bytes(original+b'other cell suffix\n')
        self.assertEqual(len(self.collect()['normalized_records']),1)
        journal.write_bytes(b'[]\n'+b'other cell suffix\n')
        with self.assertRaisesRegex(ValueError,'prefix changed'):self.collect()

    def test_unreviewed_replacement_implementation_cannot_promote_a_rejection(self):
        folder,record=self.fixture(rejected=True);pointer=self.erratum(folder,record)
        audit=read(folder/'supplement.json');audit['erratum']['implementation_reference']['sha256']='a'*64
        ref_=save(folder/'supplement.json',audit);value=read(pointer);value['effective_audit']=ref_;save(pointer,value)
        with self.assertRaisesRegex(ValueError,'reviewed implementation'):self.collect()

    def test_empty_failed_preparation_cannot_block_the_later_real_primary(self):
        folder,_=self.fixture()
        empty=self.campaign/'attempt-41/cells'/folder.name;empty.mkdir(parents=True)
        report=self.collect()
        self.assertEqual(report['primary_claim_counts']['claimed'],1)
        self.assertEqual(report['bounded_scope']['counts']['recorded'],1)
        self.assertEqual(report['prepared_only_directories'],1)
        self.assertTrue(any(d['status']=='empty_preparation_directory_no_primary_evidence' for d in report['dispositions']))

    def test_duplicate_primary_attempt_is_never_best_of_two(self):
        folder,_=self.fixture();second=self.campaign/'attempt-43/cells'/folder.name;second.mkdir(parents=True)
        save(second/'state-claim.json',{})
        with self.assertRaisesRegex(ValueError,'Duplicate/resampled'):self.collect()

    def test_unknown_cell_directory_is_not_silently_dropped(self):
        (self.campaign/'attempt-42/cells/not-a-fixed-cell').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'foreign primary'):self.collect()

    def test_unknown_replacement_protocol_is_not_accepted(self):
        folder,record=self.fixture(rejected=True);pointer=self.erratum(folder,record)
        value=read(pointer);value['protocol']='unreviewed';save(pointer,value)
        with self.assertRaisesRegex(ValueError,'Unrecognized'):self.collect()

    def test_conflicting_unselected_supplement_is_reported_as_error(self):
        folder,record=self.fixture();save(folder/'unused.record.json',dict(record,success=1))
        with self.assertRaisesRegex(ValueError,'Conflicting unselected'):self.collect()

    def test_repeated_plan_only_preparation_is_not_counted_as_primary_sampling(self):
        folder,_=self.fixture();second=self.campaign/'attempt-43/cells'/folder.name;second.mkdir(parents=True)
        save(second/'worker-plan.json',read(folder/'worker-plan.json'))
        report=self.collect();self.assertEqual(report['primary_directories_observed'],1)
        self.assertEqual(report['prepared_only_directories'],1)

    def test_complete_reddit_scope_still_withholds_full_table5_rates(self):
        for cell in self.cohort['cells']:
            if cell['domain']=='reddit':self.fixture(cell)
        report=self.collect()
        self.assertEqual(report['bounded_scope']['counts']['recorded'],190)
        self.assertFalse(report['bounded_scope']['performance_rates_withheld'])
        self.assertTrue(report['full_table5']['performance_rates_withheld'])
        self.assertEqual(report['full_table5']['counts']['missing'],1360)
        for method in report['bounded_scope']['methods']:
            self.assertEqual(method['rates']['denominator'],19)
            self.assertEqual(method['rates']['success_percent'],0.)

    def test_retained_state_accepts_closed_audited_cell_without_disposal(self):
        folder,pointer=self.retained_erratum();report=self.collect()
        retained=report['selected'][0]['retained_state']
        self.assertEqual(retained['status'],'retained_no_disposal_claim')
        self.assertFalse(retained['disposal_performed'])
        self.assertFalse(Path(retained['state_path']).exists())  # No node-filesystem claim.
        self.assertNotIn('disposal',read(pointer))
        self.assertEqual(report['bounded_scope']['counts']['valid'],1)

    def test_retained_state_cannot_claim_disposal_or_an_unclosed_owner(self):
        folder,pointer=self.retained_erratum();value=read(pointer)
        value['disposal']=ref(folder/'audit.json');save(pointer,value)
        with self.assertRaisesRegex(ValueError,'Conflicting'):self.collect()
        del value['disposal']
        value['retained_state']['attempt_closure']=save(folder.parents[1]/'controller-closure.json',
            dict(remaining_owned_processes={'123':{}}));save(pointer,value)
        with self.assertRaisesRegex(ValueError,'closed cell/outer'):self.collect()

    def test_retained_state_cannot_borrow_another_claim_reference(self):
        folder,pointer=self.retained_erratum();value=read(pointer)
        value['retained_state']['state_claim']=save(self.campaign/'other-claim.json',read(folder/'state-claim.json'))
        save(pointer,value)
        with self.assertRaisesRegex(ValueError,'another claim/attempt'):self.collect()

    def test_retained_state_storage_must_bind_the_claimed_scratch(self):
        folder,pointer=self.retained_erratum();value=read(pointer)
        value['retained_state']['attempt_storage']=save(folder.parents[1]/'storage-final.json',
            dict(scratch=str(self.root/'wrong-scratch'),apparent_bytes=123));save(pointer,value)
        with self.assertRaisesRegex(ValueError,'claim/storage'):self.collect()

    def test_additive_independent_replay_must_equal_selected_audit_hash(self):
        folder,pointer=self.retained_erratum();value=read(pointer)
        independent=folder/'independent.json';independent.write_bytes((folder/'supplement.json').read_bytes())
        value['independent_audit']=ref(independent);save(pointer,value)
        self.assertEqual(self.collect()['replacement_count'],1)
        audit=read(independent);audit['scope']='changed independent interpretation'
        value['independent_audit']=save(independent,audit);save(pointer,value)
        with self.assertRaisesRegex(ValueError,'Independent replacement replay differs'):self.collect()

    def test_future_corrected_auditor_is_bound_to_its_own_plan_and_source(self):
        self.fixture()
        self.source=self.root/'future-frozen/scripts/audit_webprm_table5_cell.py';self.source.parent.mkdir(parents=True)
        self.source.write_bytes((N/'reddit-empty-dispatch-erratum-v1/corrected-auditor.py').read_bytes())
        cell=next(x for x in self.cohort['cells'] if x['domain']=='reddit' and x!=self.cell)
        folder,_=self.fixture(cell)
        self.assertEqual(len(self.collect()['normalized_records']),2)
        inputs=read(folder/'audit-input.json');inputs['source_sha256'][str(self.source)]=c.ORIGINAL
        save(folder/'audit-input.json',inputs)
        with self.assertRaisesRegex(ValueError,'Source manifest differs'):self.collect()

    def test_closed_interrupted_claim_is_reported_without_a_normalized_outcome(self):
        folder=self.campaign/'attempt-42/cells'/hashlib.sha256(self.cell['cell_id'].encode()).hexdigest()
        folder.mkdir(parents=True)
        save(folder/'worker-plan.json',dict(task_ids=[self.cell['task_id']],actor_model=self.cell['actor'],
            selector=self.cell['selector'],protocol_contract=self.contract))
        save(folder/'state-claim.json',dict(cell_id=self.cell['cell_id'],campaign_root=str(self.campaign),
            destination_absent_at_claim=True))
        save(self.campaign/'operational-stop-42.json',dict(job_id='42',
            active_cells=[dict(cell_id=self.cell['cell_id'],folder=str(folder))]))
        save(folder.parents[1]/'controller-closure.json',dict(remaining_owned_processes={}))
        report=self.collect()
        self.assertEqual(report['normalized_records'],[])
        self.assertEqual(report['primary_claim_counts']['claimed'],1)
        self.assertEqual(report['primary_claim_counts']['interrupted_missing_terminal_audit'],1)
        self.assertEqual(report['bounded_scope']['counts']['missing'],190)
        self.assertFalse(report['dispositions'][0]['normalized_outcome_created'])
        save(folder.parents[1]/'controller-closure.json',dict(remaining_owned_processes={'42':{}}))
        with self.assertRaisesRegex(ValueError,'outer owner is not closed'):self.collect()


if __name__=='__main__':unittest.main()
