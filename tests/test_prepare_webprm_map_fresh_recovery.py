import ast
import unittest
from scripts import prepare_webprm_map_fresh_recovery as b


class FreshRecoveryTests(unittest.TestCase):
    def source(self):
        return (b.owner.N/'map-hosting-validation-v4-vlm/source/scripts/validate_webprm_map_hosting.py').read_text()

    def test_prior_charge_reaches_admission_context_owner(self):
        text=b.patched_owner(self.source())
        self.assertIn("context['prior_scratch_bytes']+3*size+LIMITS['runtime_reserve_bytes']",text)
        self.assertIn("attempt=str(attempt),prior_scratch_bytes=budget['prior_scratch_bytes']",text)
        self.assertIn("profile=p,prior_scratch_bytes=budget['prior_scratch_bytes']",text)

    def test_original_native_and_two_fresh_gates_unchanged(self):
        old=ast.parse(self.source());new=ast.parse(b.patched_owner(self.source()))
        for name in ('inside','controls_pass','reset_pass','input_sha','extract'):
            find=lambda tree:ast.dump(next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name),include_attributes=False)
            self.assertEqual(find(old),find(new))

    def test_only_exact_exceptions_and_proven_hostname_fix(self):
        text=b.patched_owner(self.source())
        self.assertIn('total_seconds=14400,max_allocations=2',text)
        self.assertIn('scratch_bytes=640*2**30',text)
        self.assertIn('hosts=private_hosts(os.uname().nodename)',text)
        self.assertIn("approval.get('allocation_and_scratch_exception_approved') is not True",text)
        self.assertEqual(b.LIMITS['retained_bytes'],4*2**30)
        self.assertEqual(b.RESOURCES['account'],'vlm')
        self.assertEqual(b.RESOURCES['gpus'],0)
        self.assertEqual(b.RESOURCES['cpus'],8)

    def test_source_shape_fail_closed(self):
        with self.assertRaises(b.h.Closed):b.patched_owner(self.source().replace('max_allocations=1','max_allocations=3'))

    def test_remaining_time_not_reset(self):
        self.assertEqual((14400-839)//60,226)
        self.assertLessEqual(839+226*60,14400)
        base_bound=(384*2**30-16*2**30)//3
        self.assertLessEqual(b.PRIOR_BYTES+3*base_bound+16*2**30,b.LIMITS['scratch_bytes'])

if __name__=='__main__':unittest.main()
