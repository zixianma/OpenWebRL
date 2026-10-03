"""Guard scientific denominators and pre-update alignment of the public diagnostic."""
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import plot_arm_reward_hacking as chart


class RewardDiagnosticTest(unittest.TestCase):
    def test_pre_update_actor_alignment(self):
        c={'policy_id':'run:rollout9','checkpoint':'/actor/iter_0000008'}
        saved={'rollout_id':9,'checkpoint':'/output/iter_0000009'}
        self.assertEqual(chart.align(c,saved,{'iteration':9},'run',9),9)
        with self.assertRaises(AssertionError):
            chart.align(dict(c,checkpoint='/actor/iter_0000009'),saved,{},'run',9)

    def test_duplicate_credit_and_unlabeled_turn_denominator(self):
        c=dict(rows=100,admitted=20,effective_scored_fraction=.2,
               selected_original_rate=.2,positive_bonus_rate=.4,mean_chance_baseline=.3,
               variants={'0.5':dict(bonus_mean=.01,bonus_rms=.1,bonus_to_outcome_rms=.1)})
        p=chart.panel(c,.5)
        self.assertEqual(p['credited_rate'],.4)
        self.assertAlmostEqual(p['mean'],.5*.2*(.4-.3))
        # A mean conditional on the 20 labeled turns must fail the consistency check.
        c['variants']['0.5']['bonus_mean']=.05
        with self.assertRaises(ValueError): chart.panel(c,.5)

    def test_auxiliary_empty_is_missing_not_zero(self):
        a=dict(total_failure_rows=0,records=[],beta=.5,failure_groups=0,coefficient=0)
        p=chart.auxiliary(a)
        self.assertIsNone(p['mean']);self.assertIsNone(p['coverage'])

    def test_legacy_auxiliary_recovers_chance_and_keeps_zeros(self):
        a=dict(total_failure_rows=10,records=[{'advantage':.3},{'advantage':-.2}],
               beta=.5,failure_groups=1,coefficient=1/48)
        p=chart.auxiliary(a)
        self.assertAlmostEqual(p['chance'],.4)
        self.assertAlmostEqual(p['mean'],.01)
        self.assertAlmostEqual(p['coverage'],.2)

    def test_trailing_mean_pools_denominators_not_iteration_means(self):
        r=[dict(iteration=0,mean=.1,turns=10),dict(iteration=1,mean=0,turns=90),
           dict(iteration=4,mean=.2,turns=10)]
        out=chart.smooth(r,'mean','turns',2)
        self.assertAlmostEqual(out[1],.01)
        self.assertAlmostEqual(out[2],.2)  # No fill across the missing interval.
        r.append(dict(iteration=5,mean=None,turns=0))
        self.assertIsNone(chart.smooth(r,'mean','turns',2)[-1])


if __name__=='__main__':unittest.main()
