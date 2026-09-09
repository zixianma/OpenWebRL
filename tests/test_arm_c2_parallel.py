"""Replica isolation and allocation-bound training handoff checks."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl.arm_c2 import collection_endpoints, replica_args
from scripts.run_arm_c2_parallel import validate_resources
from scripts.train_arm_c2 import validate_compute


class ParallelC2Tests(unittest.TestCase):
    def test_replica_routes_are_isolated_without_changing_shared_protocol(self):
        args = SimpleNamespace(sglang_router_port=19100, browser_action_selector=None,
                               judge_api_model='o4-mini', max_steps=30, rollout_num_gpus=2)
        first = replica_args(args, {'actor_port':19100}, 'selector0')
        second = replica_args(args, {'actor_port':19102}, 'selector1')
        self.assertEqual((first.sglang_router_port, first.browser_action_selector), (19100,'selector0'))
        self.assertEqual((second.sglang_router_port, second.browser_action_selector), (19102,'selector1'))
        self.assertIsNone(args.browser_action_selector)
        self.assertEqual(first.judge_api_model,second.judge_api_model)
        self.assertEqual(first.max_steps,second.max_steps)
        self.assertEqual(collection_endpoints({'parallel':16}), [{'actor_port':19100,'selector_port':19101}])

    def test_duplicate_ports_and_unbalanced_concurrency_fail_closed(self):
        replicas=[dict(actor_port=19100,selector_port=19101),dict(actor_port=19102,selector_port=19103)]
        self.assertEqual(collection_endpoints(dict(parallel=32,collection_replicas=replicas)), replicas)
        with self.assertRaisesRegex(ValueError,'evenly'):
            collection_endpoints(dict(parallel=31,collection_replicas=replicas))
        replicas[1]['selector_port']=19100
        with self.assertRaisesRegex(ValueError,'distinct'):
            collection_endpoints(dict(parallel=32,collection_replicas=replicas))

    def test_parallel_launcher_rejects_wrong_or_occupied_gpus(self):
        q=dict(allocation='7',parallel=32,collection_replicas=[dict(gpu_uuid='GPU-A',actor_port=19100,selector_port=19101),dict(gpu_uuid='GPU-B',actor_port=19102,selector_port=19103)])
        replies=['UserId=test(1) JobState=RUNNING','GPU-A\nGPU-B','','']
        with patch.dict(os.environ,SLURM_JOB_ID='7'),patch.object(Path,'read_text',return_value='0::/job_7/step_1'),patch('getpass.getuser',return_value='test'):
            with patch('scripts.run_arm_c2_parallel.command',side_effect=replies):
                self.assertEqual(validate_resources(q,True),['GPU-A','GPU-B'])
            with patch('scripts.run_arm_c2_parallel.command',side_effect=[replies[0],'GPU-A\nGPU-C']):
                with self.assertRaisesRegex(ValueError,'do not match'):validate_resources(q,True)
            with patch('scripts.run_arm_c2_parallel.command',side_effect=[replies[0],replies[1],'123']):
                with self.assertRaisesRegex(ValueError,'occupied'):validate_resources(q,True)

    def test_single_gpu_trainer_requires_explicit_uuid_binding_in_two_gpu_step(self):
        q=dict(allocation='7',gpu_uuid='GPU-A')
        replies=['JobState=RUNNING','GPU-A\nGPU-B','']
        with patch.dict(os.environ,SLURM_JOB_ID='7',CUDA_VISIBLE_DEVICES='GPU-A'),patch.object(Path,'read_text',return_value='0::/job_7/step_1'),patch('subprocess.check_output',side_effect=replies):
            validate_compute(q)
        for binding in ('0','GPU-B','GPU-A,GPU-B'):
            with patch.dict(os.environ,SLURM_JOB_ID='7',CUDA_VISIBLE_DEVICES=binding),patch.object(Path,'read_text',return_value='0::/job_7/step_1'),patch('subprocess.check_output',side_effect=replies[:2]):
                with self.assertRaisesRegex(ValueError,'Wrong visible GPU'):validate_compute(q)


if __name__ == '__main__':unittest.main()
