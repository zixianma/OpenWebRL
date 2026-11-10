import unittest
from pathlib import Path
import run_arm_refresh_browser as b

class BrowserPlanTests(unittest.TestCase):
    def test_matched_comparison_changes_only_selector(self):
        p=b.plan();f,a=p['stages']
        self.assertEqual(p['primary_trajectories'],600)
        self.assertEqual(len(set(f['quality_config']['task_order'])),300)
        for k in ('checkpoint','policy_id','seed','task_order','max_steps'):
            self.assertEqual(f['quality_config'][k],a['quality_config'][k])
        self.assertEqual(f['selector']['model'],a['selector']['model'])
        self.assertNotIn('adapter',f['selector'])
        self.assertEqual(a['selector']['adapter_sha256'],b.ADAPTER_SHA)
        self.assertEqual(p['teacher_max_calls'],3*(600+6))
        self.assertEqual(f['environment']['WANDB_PROJECT'],'openwebrl-evals')
        self.assertNotEqual(f['wandb_run_id'],a['wandb_run_id'])
        self.assertEqual(f['quality_config']['task_count'],300)
        self.assertEqual(p['fixed100'],f['quality_config']['task_order'][:100])
        self.assertEqual(f['environment']['NUM_ROLLOUT'],'0')

if __name__=='__main__':unittest.main()
