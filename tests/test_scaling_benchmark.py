import os
import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from scaling_benchmark import ANCESTOR, RUNTIME, case_plan, health_result, parse_result, resources
import scaling_benchmark


class ScalingTest(unittest.TestCase):
    def info(self, gpus=8, cpus=64, memory=960):
        return (f'JobId=42 JobState=RUNNING UserId=test({os.getuid()}) NumNodes=1 NumCPUs={cpus} '
                f'NodeList=g001 EndTime=2030-01-01T04:00:00 '
                f'AllocTRES=cpu={cpus},mem={memory}G,gres/gpu={gpus},gres/gpu:h200={gpus}')

    def test_requires_full_owned_eight_gpu_profile(self):
        from datetime import datetime
        now = datetime(2030, 1, 1).timestamp()
        self.assertEqual(resources(self.info(), '42', now)['gpus'], 8)
        for info in [self.info(gpus=4), self.info(cpus=32), self.info(memory=480), self.info().replace('JobState=RUNNING', 'JobState=PENDING')]:
            with self.assertRaises(ValueError):
                resources(info, '42', now)

    def test_replay_matches_generating_checkpoint_and_isolates_wandb(self):
        plan = case_plan('42', RUNTIME/'benchmarks/test/tp4', 'tp4', 4, 64, True)
        self.assertEqual(plan['checkpoint'], str(ANCESTOR/'iter_0000060'))
        self.assertEqual(plan['rollout_id'], 61)
        self.assertEqual(plan['replay_consumed_groups'], 144)
        self.assertEqual(plan['data_parallel'], 2)
        self.assertNotEqual(plan['wandb_run_id'], 'qcq7i4ug')
        with self.assertRaises(ValueError):
            case_plan('42', Path('/tmp/wrong'), 'tp4', 4, 64, True)

    def log(self, count=12):
        return '\n'.join(f"model.py:713 - step {i}: " + str({'train/step':i,'train/loss':0.1,'train/grad_norm':1.2,'train/ppo_kl':0.002}) for i in range(count)) + '\nTimer train end (elapsed: 120s)'

    def test_incomplete_or_duplicate_replay_is_not_ranked(self):
        plan = case_plan('42', RUNTIME/'benchmarks/test/tp4', 'tp4', 4, 64, True)
        result = parse_result(self.log(), plan)
        self.assertEqual(result['seconds_per_update'], 10)
        for text in [self.log(11), self.log()+'\n'+self.log(), self.log().replace('120s', 'missing')]:
            with self.assertRaises(ValueError):
                parse_result(text, plan)

    def test_browser_score_includes_collection_training_and_checkpoint(self):
        plan = case_plan('42', RUNTIME/'benchmarks/test/b64', 'b64', 4, 64, False)
        text = self.log() + '\n' + '\n'.join([
            '[2030-01-01 00:00:00] train.py:52 - [TrainProgress] rollout=63/90 phase=generate',
            'rollout=63/90 event=done groups=48/48 completed_groups=100 pending_groups=44 elapsed_secs=600.0',
            '[2030-01-01 00:15:00] train.py:52 - [TrainProgress] rollout=63/90 phase=recovery_checkpoint_complete'])
        result = parse_result(text, plan)
        self.assertEqual(result['collection_seconds'], 600)
        self.assertEqual(result['iteration_seconds'], 900)
        self.assertEqual(result['gpu_hours_per_iteration'], 2)
        with self.assertRaises(ValueError):
            parse_result(text.replace('groups=48/48', 'groups=47/48'), plan)

    def test_uses_slowest_rank_timer_and_rejects_nonfinite_metrics(self):
        plan = case_plan('42', RUNTIME/'benchmarks/test/tp4', 'tp4', 4, 64, True)
        self.assertEqual(parse_result(self.log()+'\nTimer train end (elapsed: 132s)', plan)['seconds_per_update'], 11)
        with self.assertRaises(ValueError):
            parse_result(self.log().replace("'train/loss': 0.1", "'train/loss': 1e999"), plan)

    def test_memory_pressure_or_new_oom_prevents_browser_escalation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'health.jsonl'
            rows = [{'memory.current': str(400*1024**3), 'memory.events': 'oom 0\noom_kill 0'},
                    {'memory.current': str(850*1024**3), 'memory.events': 'oom 0\noom_kill 0'}]
            p.write_text('\n'.join(map(json.dumps, rows)))
            self.assertFalse(health_result(p)['safe_to_increase_browsers'])
            rows[1].update({'memory.current': '0', 'memory.events': 'oom 1\noom_kill 1'})
            p.write_text('\n'.join(map(json.dumps, rows)))
            self.assertTrue(health_result(p)['new_host_oom'])

    def test_topology_only_never_launches_a_browser_collection(self):
        plans = []
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            def fake_worker(command, **kwargs):
                path = Path(command[-1].rsplit(' ', 1)[-1])
                plan = json.loads(path.read_text())
                plans.append(plan)
                (path.parent/'result.json').write_text(json.dumps({
                    **plan, 'complete': True, 'seconds_per_update': plan['tensor_parallel']}))
                return SimpleNamespace(returncode=0)
            with patch.object(scaling_benchmark, 'RUNTIME', runtime), \
                 patch.object(scaling_benchmark, 'resources', return_value={'deadline': 1e12}), \
                 patch.object(scaling_benchmark.subprocess, 'check_output', return_value=''), \
                 patch.object(scaling_benchmark.subprocess, 'run', side_effect=fake_worker), \
                 patch('builtins.print'):
                scaling_benchmark.controller('42', topology_only=True)
            self.assertEqual([p['tensor_parallel'] for p in plans], [4, 2, 8, 1])
            self.assertTrue(all(p['replay_batch'] for p in plans))
            summary = json.loads((runtime/'benchmarks/qcq7i4ug-scaling-42/summary.json').read_text())
            self.assertTrue(summary['topology_only'])
            self.assertIsNone(summary['browser_winner'])


if __name__ == '__main__':
    unittest.main()
