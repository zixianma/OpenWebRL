import copy
import importlib.util
import json
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1]/'scripts/prepare_webprm_table5_campaign.py'
spec = importlib.util.spec_from_file_location('table5_preparation', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = json.loads(m.TASKS.read_text())

    def test_exact_grid_and_balanced_pair_blocks(self):
        result = m.cohort(self.tasks)
        self.assertEqual(result['episode_count'], 1550)
        self.assertEqual(len({r['cell_id'] for r in result['cells']}), 1550)
        for start in range(0,1550,10):
            block=result['cells'][start:start+10]
            self.assertEqual(len({r['task_id'] for r in block}),1)
            self.assertEqual({(r['actor'],r['selector']) for r in block},
                             {(a,s) for a in m.ACTORS for s in m.SELECTORS})

    def test_source_permutation_does_not_change_order(self):
        self.assertEqual(m.cohort(self.tasks),m.cohort(list(reversed(self.tasks))))

    def test_task_text_or_reference_not_published(self):
        raw=json.dumps(m.cohort(self.tasks))
        for key in ('intent_template','instantiation_dict','reference_answers','storage_state'):
            self.assertNotIn('"'+key+'"',raw)

    def test_changed_task_is_bound_in_hash(self):
        tasks=copy.deepcopy(self.tasks)
        tasks[0]['intent']='Changed task'
        a=m.cohort(self.tasks)['cells'][0]
        b=m.cohort(tasks)['cells'][0]
        self.assertEqual(a['cell_id'],b['cell_id'])
        self.assertNotEqual(a['task_config_sha256'],b['task_config_sha256'])

    def test_reject_duplicate_ids(self):
        tasks=copy.deepcopy(self.tasks)
        tasks[1]['task_id']=tasks[0]['task_id']
        with self.assertRaises(ValueError):m.cohort(tasks)

    def test_reject_multisite(self):
        tasks=copy.deepcopy(self.tasks)
        tasks[0]['sites'].append('map')
        with self.assertRaises(ValueError):m.cohort(tasks)

    def test_reject_missing_task(self):
        with self.assertRaises(ValueError):m.cohort(self.tasks[:-1])

    def test_native_call_bounds_use_single_shepherd_checklist(self):
        b=m.dispatch_bounds()
        self.assertEqual(b['local'], {'shepherd8b':46810,'arbiter7b':37200,'piotr_selectionarm':9300})
        self.assertEqual(b['total_local'],93310)
        self.assertEqual(b['actor_paid'],137950)
        self.assertEqual(b['gpt_mini_selector_paid'],37200)
        self.assertEqual(b['native_fuzzy_paid'],390)
        self.assertEqual(b['total_paid'],175540)

    def test_reject_home_output_before_write(self):
        with self.assertRaises(ValueError):
            m.exclusive_json(Path('/gpfs/home/zixianma/table5-forbidden.json'),{})


if __name__=='__main__':unittest.main()
