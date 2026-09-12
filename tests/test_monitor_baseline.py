import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('monitor_baseline', Path(__file__).resolve().parents[1] / 'scripts/monitor_baseline.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class GpuSampleTest(unittest.TestCase):
    def test_timeout_does_not_prevent_the_next_sample(self):
        success = subprocess.CompletedProcess(['nvidia-smi'], 0, '0, 44, 71000, 250\n')
        with patch.object(monitor.subprocess, 'run', side_effect=[subprocess.TimeoutExpired('nvidia-smi', 15), success]):
            missed = monitor.sample_gpu()
            recovered = monitor.sample_gpu()
        self.assertEqual(missed['gpu'], [])
        self.assertIn('timed out', missed['gpu_query_error'])
        self.assertEqual(recovered, {'gpu': ['0, 44, 71000, 250']})

    def test_nonzero_exit_is_marked_as_missing_not_valid_gpu_data(self):
        with patch.object(monitor.subprocess, 'run', return_value=subprocess.CompletedProcess(['nvidia-smi'], 9, 'driver failure')):
            self.assertEqual(monitor.sample_gpu(), {'gpu': [], 'gpu_query_error': 'nvidia-smi exited 9'})

    def test_unavailable_executable_is_nonfatal(self):
        with patch.object(monitor.subprocess, 'run', side_effect=FileNotFoundError):
            self.assertEqual(monitor.sample_gpu(), {'gpu': [], 'gpu_query_error': 'FileNotFoundError'})
