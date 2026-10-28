"""Reporting-only interruption fixtures; saved primary campaign is read-only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import collect_webprm_table5_campaign as c

N=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation')
C=N/'reddit-table5-190-v3/campaign'
D=N/'reddit-interrupted-v10-dispositions-20261015-v2'
def save(p,v):
    p.write_text(json.dumps(v,sort_keys=True)+'\n')
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

class SavedInterruptionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/tmp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.candidate=json.loads((D/'candidate.json').read_text())
        self.cohort=json.loads((N/'reddit-table5-190-v11/cohort.json').read_text())
        self.cohort=self.cohort.get('cohort',self.cohort)

    def fixture(self,index=0,mutate=None,record_change=None):
        data=deepcopy(self.candidate)
        if mutate:mutate(data['records'][index])
        item=data['records'][index];cell=item['cell']
        candidate=save(self.root/'candidate.json',data);replay=save(self.root/'replay.json',data)
        record=dict(cell_id=cell['cell_id'],released_task_id=cell['task_id'],actor_model=cell['actor'],selector=cell['selector'],
            domain=cell['domain'],task_config_sha256=cell['task_config_sha256'],tasks_sha256=self.cohort['source_sha256'],
            cohort_sha256=c.digest(self.cohort),protocol_sha256=self.cohort['protocol_sha256'],audit_verified=True,terminal=True,
            attempted=True,valid=False,success=None,native_success=None,effective_semantic_status='not_applicable',
            native_semantic_status='not_applicable',audit_sha256=candidate['sha256'])
        if record_change:record.update(record_change)
        normalized=save(self.root/'record.json',record)
        bindings=dict(candidate=candidate,independent_audit=replay,implementation=data['implementation'],effective_record=normalized)
        review=save(self.root/'review.json',dict(protocol='table5_saved_v10_interruption_activation_review_v1',
            verified=True,errors=[],cell_id=cell['cell_id'],**bindings))
        pointer=self.root/'pointer.json'
        save(pointer,dict(protocol=c.V10_INTERRUPTION,cell_id=cell['cell_id'],original_bytes_preserved=True,
            primary_rerun=False,disposal_performed=False,raw_success=None,effective_success=None,
            independent_review=review,**bindings))
        folder=C/'attempt-359482/cells'/hashlib.sha256(cell['cell_id'].encode()).hexdigest()
        return pointer,folder,cell

    def bind(self,*args):
        pointer,folder,cell=args
        return c.v10_interruption_binding(c.Evidence(60),pointer,folder,cell,self.cohort,C)

    def test_saved_failure_normalizes_to_invalid_without_native_verdict(self):
        record,selection=self.bind(*self.fixture())
        self.assertFalse(record['valid']);self.assertIsNone(record['success']);self.assertIsNone(record['native_success'])
        self.assertTrue(selection['original_result_present']);self.assertFalse(selection['disposal_performed'])

    def test_earlier_incomplete_watchdog_is_preserved_under_later_outer_closure(self):
        record,selection=self.bind(*self.fixture(index=1))
        self.assertFalse(selection['original_worker_closure_complete'])
        self.assertIsNone(record['success'])

    def test_absent_result_stays_absent_and_has_no_fabricated_worker_closure(self):
        record,selection=self.bind(*self.fixture(index=2))
        self.assertFalse(selection['original_result_present']);self.assertIsNone(selection['original_worker_closure_complete'])
        self.assertIsNone(record['native_success'])

    def test_relabeling_incomplete_worker_watchdog_as_complete_is_rejected(self):
        args=self.fixture(index=1,mutate=lambda row:row.update(saved_worker_closure_complete=True))
        with self.assertRaisesRegex(ValueError,'watchdog/late owner lineage'):self.bind(*args)

    def test_fabricated_native_verdict_in_normalized_record_is_rejected(self):
        args=self.fixture(record_change=dict(success=0,native_success=0))
        with self.assertRaisesRegex(ValueError,'fabricated a verdict'):self.bind(*args)

    def test_fabricated_result_presence_is_rejected(self):
        args=self.fixture(index=2,mutate=lambda row:row.update(saved_result_present=True))
        with self.assertRaisesRegex(ValueError,'result presence'):self.bind(*args)

    def test_omitted_saved_artifact_is_rejected(self):
        args=self.fixture(mutate=lambda row:row['artifacts'].pop('worker.log'))
        with self.assertRaisesRegex(ValueError,'artifact omitted'):self.bind(*args)

    def test_unverified_review_is_rejected_even_with_candidate_hashes(self):
        args=self.fixture();pointer=json.loads(args[0].read_text());review=json.loads(Path(pointer['independent_review']['path']).read_text())
        review['verified']=False;pointer['independent_review']=save(self.root/'review.json',review);save(args[0],pointer)
        with self.assertRaisesRegex(ValueError,'activation review'):self.bind(*args)

    def test_late_outer_owner_must_include_original_incomplete_worker_pid(self):
        args=self.fixture(index=1);original=c.Evidence.read_ref
        def altered(evidence,reference,root=None):
            value=original(evidence,reference,root)
            if reference['path']==self.candidate['outer_closure']['path']:
                value=deepcopy(value);value['owned_pids'].remove(544948)
            return value
        with patch.object(c.Evidence,'read_ref',new=altered):
            with self.assertRaisesRegex(ValueError,'watchdog/late owner lineage'):self.bind(*args)

    def test_moving_financial_prefix_to_another_attempt_is_rejected(self):
        args=self.fixture();pointer=json.loads(args[0].read_text());data=json.loads(Path(pointer['candidate']['path']).read_text())
        data['broker_prefix']['path']=str(C/'attempt-360293/broker.jsonl')
        pointer['candidate']=save(self.root/'candidate.json',data);pointer['independent_audit']=save(self.root/'replay.json',data)
        review=json.loads(Path(pointer['independent_review']['path']).read_text())
        review.update(candidate=pointer['candidate'],independent_audit=pointer['independent_audit'])
        pointer['independent_review']=save(self.root/'review.json',review);save(args[0],pointer)
        with self.assertRaisesRegex(ValueError,'prefix belongs to another owner'):self.bind(*args)
