"""Reporting-only forensic activation fixtures; original campaigns remain read-only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import collect_webprm_om2w_table5 as c

BASE=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-om2w-table5-20261012')
PACKAGE=BASE/'next90-gpt4o-v2'
RECOVERY=BASE/'nodefail-recovery-361144-v1'
REVIEW=BASE/'nodefail-independent-audit-20261015'
def reference(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(p,v):p.write_text(json.dumps(v,sort_keys=True)+'\n');return reference(p)

class NodeFailActivationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/tmp');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.candidate=c.read(RECOVERY/'interrupted-primary-audit.json')
        self.proposal=c.read(PACKAGE/'proposal.json');self.template=c.read(self.proposal['worker_template'])
        self.cohort=c.read(self.proposal['cohort_path'])
        self.cell=next(x for x in self.cohort['cells'] if x['cell_id']==self.candidate['cells'][0]['cell_id'])
        self.claim=c.read(self.candidate['cells'][0]['claim']['path'])

    def fixture(self,candidate_change=None,independent_change=None,review_change=None):
        candidate=deepcopy(self.candidate)
        if candidate_change:candidate_change(candidate)
        candidate_ref=save(self.root/'candidate.json',candidate)
        independent=c.read(REVIEW/'review.json');independent['candidate']=candidate_ref
        if independent_change:independent_change(independent)
        independent_ref=save(self.root/'independent.json',independent)
        helper=reference(REVIEW/'financial-helper-reviewed.py')
        bindings=dict(original_audit=candidate_ref,independent_review=independent_ref,financial_helper_archive=helper)
        review=dict(protocol=c.NODEFAIL_REVIEW_PROTOCOL,verified=True,errors=[],raw_artifacts_preserved=True,
            primary_rerun=False,native_verdict_created=False,cell_ids=[x['cell_id'] for x in candidate['cells']],**bindings)
        if review_change:review_change(review)
        root_ref=save(self.root/'root-review.json',review)
        pointer=self.root/'pointer.json'
        save(pointer,dict(protocol=c.NODEFAIL_PROTOCOL,cell_id=self.cell['cell_id'],raw_artifacts_preserved=True,
            primary_rerun=False,native_success=None,effective_success=None,root_review=root_ref,**bindings))
        return pointer

    def bind(self,pointer):
        return c.verified_nodefail_disposition(pointer,self.proposal,self.cohort,self.template,self.cell,self.claim,c.NodeFailEvidence())

    def test_reviewed_nodefail_resolves_invalid_only_and_preserves_absent_raw_verdicts(self):
        result,attempt=self.bind(self.fixture())
        self.assertFalse(result['valid']);self.assertIsNone(result['success']);self.assertIsNone(result['native_success'])
        self.assertEqual(result['raw_queue_state'],'attempted_without_outcome')
        self.assertEqual(attempt.name,'attempt-361144')
        for item in self.candidate['cells']:
            folder=Path(item['worker_plan']['path']).parent
            self.assertFalse((folder/'cell-outcome.json').exists())

    def test_candidate_without_root_activation_is_rejected(self):
        pointer=self.fixture(review_change=lambda r:r.update(verified=False))
        with self.assertRaisesRegex(ValueError,'root activation review'):self.bind(pointer)

    def test_root_review_must_cover_all_six_distinct_claims(self):
        pointer=self.fixture(review_change=lambda r:r['cell_ids'].pop())
        with self.assertRaisesRegex(ValueError,'omits/repeats'):self.bind(pointer)

    def test_invented_native_verdict_is_rejected_even_under_fixture_review(self):
        pointer=self.fixture(candidate_change=lambda d:d['cells'][0].update(raw_native_success=0))
        with self.assertRaisesRegex(ValueError,'absent verdict changed'):self.bind(pointer)

    def test_omitted_raw_artifact_is_rejected(self):
        pointer=self.fixture(candidate_change=lambda d:d['cells'][0]['evidence'].pop())
        with self.assertRaisesRegex(ValueError,'artifact omitted'):self.bind(pointer)

    def test_resetting_unknown_reservations_is_rejected(self):
        pointer=self.fixture(independent_change=lambda d:d.update(retained_unknown_paid_reservations=0))
        with self.assertRaisesRegex(ValueError,'unknown reservations'):self.bind(pointer)

    def test_changed_complete_assigned_plan_is_rejected(self):
        pointer=self.fixture()
        with patch.object(c,'assigned_plan',return_value=dict(foreign_endpoint='https://other.invalid')):
            with self.assertRaisesRegex(ValueError,'full assigned worker plan'):self.bind(pointer)

    def test_old_boot_time_cannot_prove_global_process_closure(self):
        pointer=self.fixture();original=c.NodeFailEvidence.reference
        text=self.root/'old-boot.txt';text.write_text('NodeName=g011 BootTime=2026-10-01T00:00:00')
        def substitute(evidence,value):
            path=original(evidence,value)
            return text if path.name=='node-reboot.txt' else path
        with patch.object(c.NodeFailEvidence,'reference',new=substitute):
            with self.assertRaisesRegex(ValueError,'post-allocation global closure'):self.bind(pointer)

    def test_existing_result_cannot_be_reinterpreted_as_absent(self):
        pointer=self.fixture();original=Path.exists
        def exists(path):
            if path.name=='result.json' and 'attempt-361144' in str(path) and self.cell['task_id'] in str(path):return True
            return original(path)
        with patch.object(Path,'exists',new=exists):
            with self.assertRaisesRegex(ValueError,'replace a raw result'):self.bind(pointer)

    def test_exact_prefix_allows_later_financial_appends_without_reset(self):
        p=self.root/'journal.jsonl';raw=b'{"event":"epoch_closed"}\n';p.write_bytes(raw)
        checksum=hashlib.sha256(raw).hexdigest();evidence=c.NodeFailEvidence()
        self.assertEqual(evidence.prefix(p,len(raw),checksum)['event'],'epoch_closed')
        with p.open('ab') as stream:stream.write(b'{"event":"epoch_started"}\n')
        self.assertEqual(evidence.prefix(p,len(raw),checksum)['event'],'epoch_closed')
        with self.assertRaisesRegex(ValueError,'prefix differs'):
            c.NodeFailEvidence().prefix(p,len(raw),'0'*64)

    def test_overall_arithmetic_preserves_nulls_and_counts_forensic_invalid_separately(self):
        result,_=self.bind(self.fixture());cell=dict(self.cell,selector='arbiter7b')
        baseline=dict(cell,cell_id='baseline',selector='no_search')
        valid=dict(valid=True,success=1)
        before=deepcopy(result)
        summary=c.summarize([(cell,result,'audited_interrupted'),(baseline,valid,'recorded')])
        self.assertTrue(summary['coverage_complete']);self.assertEqual(result,before)
        row=next(x for x in summary['by_condition'] if x['selector']=='arbiter7b')
        self.assertEqual(row['recorded'],0);self.assertEqual(row['audited_interrupted'],1)
        self.assertEqual(row['resolved'],1);self.assertEqual(row['valid'],0);self.assertEqual(row['successes'],0)
        self.assertEqual(row['overall_success_pct'],0);self.assertIsNone(row['valid_only_success_pct'])

    def test_rewritten_cached_prefix_is_rejected_while_appends_are_allowed(self):
        p=self.root/'journal.jsonl';raw=b'{"event":"epoch_closed"}\n';p.write_bytes(raw)
        checksum=hashlib.sha256(raw).hexdigest();evidence=c.NodeFailEvidence();evidence.prefix(p,len(raw),checksum)
        p.write_bytes(raw.replace(b'closed',b'opened'))
        with self.assertRaisesRegex(ValueError,'prefix differs'):evidence.prefix(p,len(raw),checksum)

    def synthetic_stage_fixture(self):
        package=self.root/'package';package.mkdir();campaign=package/'campaign';cell=self.cell
        cohort=dict(cells=[cell]);save(package/'cohort.json',cohort);save(package/'template.json',{})
        proposal=dict(campaign_root=str(campaign),source_sha256={},cohort_path=str(package/'cohort.json'),
            cohort_sha256=c.digest(cohort),worker_template=str(package/'template.json'),stage=cell['stage'])
        save(package/'proposal.json',proposal)
        for name in ['claims','outcomes']:(campaign/'queue'/name).mkdir(parents=True,exist_ok=True)
        save(campaign/'queue/manifest.json',dict(cell_count=1,cells_sha256=c.identity([cell])))
        save(campaign/'queue/claims'/(c.identity(cell['cell_id'])+'.json'),self.claim)
        (campaign/'forensic-dispositions').mkdir()
        save(campaign/'forensic-dispositions'/(c.identity(cell['cell_id'])+'.json'),{})
        return package,campaign

    def test_stage_routes_only_explicit_pointer_through_strict_verifier(self):
        package,campaign=self.synthetic_stage_fixture()
        result=dict(valid=False,success=None,native_success=None,error_type='NODE_FAIL')
        with patch.object(c,'verified_nodefail_disposition',return_value=(result,campaign/'attempt-361144')) as verify:
            rows=c.stage_records(package)
        self.assertEqual(verify.call_count,1);self.assertEqual(rows[0][2],'audited_interrupted')
        self.assertTrue(c.summarize(rows)['coverage_complete'])
        self.assertFalse((campaign/'queue/outcomes'/(c.identity(self.cell['cell_id'])+'.json')).exists())

    def test_stage_rejects_forensic_pointer_conflicting_with_raw_result(self):
        package,campaign=self.synthetic_stage_fixture()
        folder=campaign/'attempt-361144/cells'/hashlib.sha256(self.cell['cell_id'].encode()).hexdigest()
        result=folder/'episodes'/('om2w_'+self.cell['task_id'])/(self.cell['actor']+'__'+self.cell['selector'])/'result.json'
        result.parent.mkdir(parents=True);save(result,{})
        with patch.object(c,'verified_nodefail_disposition',side_effect=AssertionError('Must reject before verifier')):
            with self.assertRaisesRegex(ValueError,'conflicts'):c.stage_records(package)

    def test_all_six_original_claims_pass_without_creating_any_raw_outcomes(self):
        for item in self.candidate['cells']:
            self.cell=next(x for x in self.cohort['cells'] if x['cell_id']==item['cell_id'])
            self.claim=c.read(item['claim']['path'])
            result,_=self.bind(self.fixture())
            self.assertFalse(result['valid']);self.assertIsNone(result['success'])
            self.assertFalse((PACKAGE/'campaign/queue/outcomes'/(c.identity(self.cell['cell_id'])+'.json')).exists())
