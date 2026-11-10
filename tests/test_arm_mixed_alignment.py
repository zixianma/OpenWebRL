import math
import unittest
from audit_arm_mixed_alignment import reconstruct_group,reweighted,cluster_ci

class Alignment(unittest.TestCase):
    def test_native_error_sentinel_is_not_silently_recast_as_failure(self):
        rows=[dict(index=i,trajectory_id=i,task_id='x',reward=r,trainable=True,arm_turn_bonus={})
              for i,r in enumerate([1,0,-1])]
        tr=reconstruct_group(dict(rows=rows,accepted_sample_indices=[0,1,2]),'p')
        self.assertEqual([t['outcome'] for t in tr],[1,0,-1])
        self.assertEqual(tr[1]['A'],0.)
        self.assertAlmostEqual(tr[2]['A'],-1/(1+1e-6))

    def test_native_sample_std_and_same_advantage_per_trajectory(self):
        rows=[]
        for trajectory,reward in enumerate([1,0,0,0,0]):
            for turn in range(trajectory+1):
                rows.append(dict(index=trajectory*10+turn,trajectory_id=trajectory,task_id='x',reward=reward,
                                 trainable=True,arm_turn_bonus={}))
        tr=reconstruct_group(dict(rows=rows,accepted_sample_indices=[r['index'] for r in rows]),'p')
        self.assertAlmostEqual(tr[0]['A'],.8/(math.sqrt(.2)+1e-6))
        self.assertAlmostEqual(tr[4]['A'],-.2/(math.sqrt(.2)+1e-6))
        self.assertTrue(all(t['A']==tr[4]['A'] for t in tr[4]['turns']))

    def test_reweighting_preserves_mean_sign_and_routes_failed_credit(self):
        for A in (1.,-1.,0.):
            tr=dict(A=A,turns=[dict(u=.8),dict(u=-.2),dict(u=0.)])
            output=reweighted(tr,.5)
            self.assertAlmostEqual(sum(w for w,a in output)/3,1.)
            self.assertAlmostEqual(sum(a for w,a in output)/3,A)
            self.assertTrue(all(a*A>=0 for w,a in output))
            if A: self.assertGreater(output[0][1],output[1][1])
            self.assertEqual(reweighted(tr,0.),[(1.,A)]*3)

    def test_task_cluster_resampling_keeps_repeated_observations(self):
        groups=[dict(task='a',x=1),dict(task='a',x=3),dict(task='b',x=-1)]
        ci=cluster_ci(groups,'x',100)
        self.assertEqual(ci['task_clusters'],2);self.assertEqual(ci['groups'],3)
        self.assertAlmostEqual(ci['mean'],1.)
        self.assertGreaterEqual(ci['lower'],-1);self.assertLessEqual(ci['upper'],2)

if __name__=='__main__':unittest.main()
