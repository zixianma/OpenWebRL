import importlib.util
from pathlib import Path
import unittest
import numpy as np

SPEC=importlib.util.spec_from_file_location('visualize_tasks',Path(__file__).resolve().parents[1]/'scripts/visualize_arm_task_clusters.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


class SamplingTests(unittest.TestCase):
    def test_quota_capacity_and_nesting(self):
        caps={'a':2,'b':10,'c':20};weights={'a':1.,'b':3.,'c':4.}
        previous=None
        for budget in range(3,33):
            q=M.quotas(budget,caps,weights)
            self.assertEqual(sum(q.values()),budget)
            self.assertTrue(all(1<=q[s]<=caps[s] for s in caps))
            if previous:self.assertTrue(all(q[s]>=previous[s] for s in caps))
            previous=q

    def test_lazy_matches_naive_greedy(self):
        rng=np.random.default_rng(7);sim=rng.random((13,19));weights=rng.random(13);weights/=weights.sum()
        lazy=M.facility_order(sim,weights,8);covered=np.zeros(13);naive=[]
        for _ in range(8):
            gains=np.maximum(sim-covered[:,None],0).T@weights;gains[naive]=-np.inf
            j=int(np.argmax(gains));naive.append(j);covered=np.maximum(covered,sim[:,j])
        self.assertEqual(lazy,naive)
        self.assertEqual(len(set(lazy)),8)


if __name__=='__main__':unittest.main()
