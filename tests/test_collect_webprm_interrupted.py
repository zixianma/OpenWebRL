"""Read-only historical evidence with synthetic review pointers only under /tmp."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import collect_webprm_table5_campaign as c

N=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/reddit-interruption-audit-v3')

def ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def read(path):return json.loads(Path(path).read_text())
def save(path,value):path.write_text(json.dumps(value,sort_keys=True)+'\n');return ref(path)


@unittest.skipUnless((N/'piotr-mini-audit.json').exists(),'Frozen saved failure review unavailable')
class InterruptedCollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='interruption-collector-fixture-',dir='/tmp')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def fixture(self,label='piotr-mini',audit_update=None,record_update=None):
        inputs=read(N/(label+'-input.json'));audit=read(N/(label+'-audit.json'))
        if audit_update:audit.update(audit_update)
        effective=save(self.root/'audit.json',audit)
        normalized=dict(audit['record'],audit_sha256=effective['sha256'])
        if record_update:normalized.update(record_update)
        record=save(self.root/'audit.record.json',normalized)
        replay=save(self.root/'independent.json',audit)
        implementation=ref(N/'source/scripts/audit_webprm_interrupted_cell.py')
        value=dict(protocol=c.INTERRUPTION,cell_id=inputs['cell']['cell_id'],original_bytes_preserved=True,
            primary_rerun=False,disposal_performed=False,raw_success=None,effective_success=None,
            effective_audit=effective,effective_record=record,independent_audit=replay,
            input=ref(N/(label+'-input.json')),implementation=implementation)
        review=dict(protocol='table5_interruption_independent_activation_review_v1',verified=True,errors=[],
            cell_id=inputs['cell']['cell_id'],fixture_only=True,
            **{k:value[k] for k in ('input','implementation','effective_audit','effective_record','independent_audit')})
        value['independent_review']=save(self.root/'review.json',review)
        pointer=self.root/'pointer.json';save(pointer,value)
        cohort=read(inputs['cohort']['path']);cohort=cohort.get('cohort',cohort)
        return pointer,Path(inputs['cell_folder']),inputs['cell'],cohort,Path(inputs['campaign_root'])

    def bind(self,args):return c.interruption_binding(c.Evidence(),*args)

    def test_saved_failure_selected_without_verdict_or_disposal(self):
        for label in ('arbiter-mini','piotr-mini'):
            args=self.fixture(label);before={p:ref(Path(p)) for p in
                (v['path'] for v in read(N/(label+'-input.json'))['artifacts'].values())}
            record,selection=self.bind(args)
            self.assertFalse(record['valid']);self.assertIsNone(record['success']);self.assertIsNone(record['native_success'])
            self.assertTrue(selection['interrupted_failure']);self.assertFalse(selection['disposal_performed'])
            self.assertEqual(before,{p:ref(Path(p)) for p in before})

    def test_absent_review_cannot_activate_candidate(self):
        args=self.fixture();value=read(args[0]);value['independent_review']['sha256']='0'*64;save(args[0],value)
        with self.assertRaises(ValueError):self.bind(args)

    def test_different_independent_replay_cannot_activate_candidate(self):
        args=self.fixture();value=read(args[0])
        value['independent_audit']=save(self.root/'independent.json',dict(read(self.root/'independent.json'),scope='different'))
        save(args[0],value)
        with self.assertRaisesRegex(ValueError,'Independent interrupted replay'):self.bind(args)

    def test_missing_verdict_cannot_be_zero_or_one(self):
        for update in [dict(success=0),dict(success=1),dict(native_success=0),dict(valid=True)]:
            args=self.fixture(record_update=update)
            with self.subTest(update=update),self.assertRaisesRegex(ValueError,'absent native/effective'):
                self.bind(args)

    def test_pointer_cannot_claim_resampling_disposal_or_verdict(self):
        for update in [dict(raw_success=0),dict(effective_success=0),dict(primary_rerun=True),dict(disposal_performed=True)]:
            args=self.fixture();value=read(args[0]);value.update(update);save(args[0],value)
            with self.subTest(update=update),self.assertRaisesRegex(ValueError,'Unreviewed interruption pointer'):
                self.bind(args)

    def test_candidate_cannot_authorize_disposal_or_native_replay(self):
        for update in [dict(complete_for_disposal=True),dict(native_replay=dict(verified=True))]:
            with self.subTest(update=update),self.assertRaisesRegex(ValueError,'complete preserved evidence'):
                self.bind(self.fixture(audit_update=update))

    def test_financial_prefix_and_queue_review_are_required(self):
        original=read(N/'piotr-mini-audit.json')
        for update in [dict(reset_and_queue=dict(verified=False)),
                       dict(call_audit=dict(original['call_audit'],budget_prefix_bytes=1)),
                       dict(call_audit=dict(original['call_audit'],global_unknown_reservations=1))]:
            with self.subTest(update=update),self.assertRaises(ValueError):self.bind(self.fixture(audit_update=update))


if __name__=='__main__':unittest.main()
