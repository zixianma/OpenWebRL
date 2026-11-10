"""Do not verify TP8 accidentally when the next continuation requests TP2/4."""
import ast
from pathlib import Path
import unittest


class RestoreTopology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'scripts/check_arm_gpu_checkpoint_restore.py'
        tree=ast.parse(path.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='restore_tensor_parallel')
        ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
        cls.resolve=staticmethod(ns['restore_tensor_parallel'])

    def test_continuation_uses_selected_tp_not_gpu_count(self):
        for tp in (2,4,8):self.assertEqual(self.resolve(8,{'TP_SIZE':str(tp)},continuation=True),tp)

    def test_old_standalone_migration_keeps_requested_profile(self):
        self.assertEqual(self.resolve(8,{'TP_SIZE':'2'},continuation=False),8)

    def test_impossible_shape_rejected(self):
        with self.assertRaises(ValueError):self.resolve(4,{'TP_SIZE':'8'},continuation=True)
        with self.assertRaises(ValueError):self.resolve(8,{'TP_SIZE':'3'},continuation=True)


if __name__=='__main__':unittest.main()
