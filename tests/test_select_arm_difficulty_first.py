import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from select_arm_difficulty_first import split_cutoff
from visualize_arm_task_clusters import facility_order


class DifficultyFirstTests(unittest.TestCase):
    def test_difficulty_determines_cutoff_before_diversity(self):
        rows=[{'rubric_difficulty':s,'id':i} for i,s in enumerate([2,5,6,8,5,5,4])]
        cutoff,fixed,tied,n=split_cutoff(rows,4)
        self.assertEqual(cutoff,5)
        self.assertEqual(sorted(r['rubric_difficulty'] for r in fixed),[6,8])
        self.assertEqual(len(tied),3)
        self.assertEqual(n,2)

    def test_additional_choice_accounts_for_forced_tasks(self):
        sim=np.array([[1,.1],[.1,.8]],dtype=np.float32)
        self.assertEqual(facility_order(sim,np.array([.5,.5]),1),[0])
        self.assertEqual(facility_order(sim,np.array([.5,.5]),1,initial_coverage=[1,0]),[1])

    def test_empty_extra_budget_selects_nothing(self):
        self.assertEqual(facility_order(np.ones((2,2)),np.array([.5,.5]),0),[])


if __name__=='__main__':unittest.main()
