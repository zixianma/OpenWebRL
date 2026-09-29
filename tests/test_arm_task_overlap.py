import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('overlap',Path(__file__).resolve().parents[1]/'scripts/screen_arm_task_overlap.py')
overlap=importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlap)


class OverlapTests(unittest.TestCase):
    def test_shared_reference_preserves_heldout_and_train_provenance(self):
        candidates=[dict(task_id=1,task_name='a'),dict(task_id=2,task_name='b')]
        refs=[dict(text='both',provenance=[dict(source='released_openwebrl'),dict(source='online-mind2web.jsonl')]),
              dict(text='test',provenance=[dict(source='webgym_test')])]
        e=np.array([[1,0],[0,1],[1,0],[0,1]],dtype=np.float32)
        rows=overlap.nearest_report(candidates,refs,e)
        self.assertEqual(rows[0]['heldout'][0]['text'],'both')
        self.assertEqual(rows[0]['existing_training'][0]['text'],'both')
        self.assertEqual([x['task_id'] for x in rows[0]['candidates']],['2'])
        self.assertEqual(rows[0]['decision'],'review')

    def test_invalid_vectors_fail_instead_of_producing_filter_decisions(self):
        c=[dict(task_id=1,task_name='a')]
        for e in [np.array([[0.,0.]]),np.array([[float('nan'),1.]]),np.ones((2,2))]:
            with self.assertRaises(ValueError):overlap.nearest_report(c,[],e)


if __name__=='__main__':unittest.main()
