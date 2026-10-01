"""Validate paired inference against enumeration and exact binary-pair tests."""
import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy import stats

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_arm_paired_evals import exact_signflip,holm,summarize


class PairedInferenceTests(unittest.TestCase):
    def test_integer_swap_matches_enumerated_repeat_vectors(self):
        d=np.array([3,-2,1,0,-1,2])
        values=[abs(np.dot(d,signs))>=abs(d.sum()) for signs in itertools.product([-1,1],repeat=len(d))]
        self.assertAlmostEqual(exact_signflip(d),np.mean(values),places=14)

    def test_single_repeat_is_exact_mcnemar(self):
        d=np.array([1]*35+[-1]*20+[0]*100)
        self.assertAlmostEqual(exact_signflip(d),stats.binomtest(35,55,.5).pvalue,places=14)
        self.assertEqual(exact_signflip(np.zeros(30,dtype=int)),1.)

    def test_holm_known_values(self):
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])

    def test_repeating_same_outcomes_does_not_inflate_sample_size(self):
        rng=np.random.default_rng(7)
        single=rng.integers(0,2,size=(40,3,1));valid=np.ones_like(single)
        first=summarize(single,valid,resamples=500)
        repeated=summarize(np.repeat(single,3,axis=2),np.repeat(valid,3,axis=2),resamples=500)
        for a,b in zip(first['comparisons'],repeated['comparisons']):
            self.assertAlmostEqual(a['p_exact'],b['p_exact'],places=14)
            np.testing.assert_allclose(a['delta_ci95_pp'],b['delta_ci95_pp'],atol=1e-12)


if __name__=='__main__':unittest.main()
