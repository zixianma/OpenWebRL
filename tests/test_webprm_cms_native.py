from copy import deepcopy
from pathlib import Path
import unittest

from openwebrl import webprm_cms_native as n


class NativeGateTest(unittest.TestCase):
    def test_pinned_native_evaluator_exact_browser_task_positive_negative(self):
        # This is the released evaluator, not a replacement implementation;
        # only the page boundary is fake. No browser/provider is constructed.
        source=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')
        native=n.load_native_evaluator(source)
        class Page:
            url=n.ORIGIN+'/admin/catalog/product/edit/id/1/'
            def __init__(self):self.calls=[]
            def evaluate(self,expression):
                self.calls.append(expression)
                if expression!='() => '+n.NAME_JS:raise AssertionError(expression)
                return 'Fixture product'
        page=Page()
        result=n.evaluator_controls(native,page,dict(id=1,name='Fixture product'))
        self.assertEqual([result[k]['score'] for k in ('positive','negative')],[1.0,0.0])
        self.assertTrue(all(result[k]['valid'] for k in result))
        self.assertTrue(all(result[k]['model_provider_calls']==0 for k in result))
        self.assertTrue(all(result[k]['source']==n.verify_source(source) for k in result))
        self.assertEqual(page.calls,['() => '+n.NAME_JS]*2)

    def rows(self):
        base=dict(status='passed',instance_id='a',base_sha256='fixed',product=dict(id=1,name='Original'),
                  native_positive=1,native_negative=0,session_sha256='one')
        mutation=dict(base,product=dict(id=1,name='OpenWebRL CMS reset marker'))
        reset=dict(base,instance_id='b',session_sha256='two')
        return base,mutation,reset

    def test_valid_native_mutation_then_independent_reset(self):
        result=n.reset_gate(*self.rows(),media_absent=True)
        self.assertTrue(result['fresh_reset_verified'])
        self.assertFalse(result['original_state_equivalence_verified'])

    def test_missing_reset_mutation_auth_or_negative_control_rejects(self):
        for field,value in [('instance_id','a'),('session_sha256','one'),('native_negative',1),('status','failed'),
                            ('product',dict(id=1,name='Still mutated')),('base_sha256','different')]:
            base,mutation,reset=self.rows();reset[field]=value
            with self.assertRaises(n.Closed):n.reset_gate(base,mutation,reset,media_absent=True)
        with self.assertRaises(n.Closed):n.reset_gate(*self.rows(),media_absent=False)

    def test_fixed_product_contract_rejects_unbounded_or_missing_identity(self):
        for row in ({'id':0,'name':'name'},{'id':'1','name':'name'},{'id':1,'name':'x'*513},{}):
            with self.assertRaises(n.Closed):n.validate_product(row)
