import base64
import copy
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from slime.utils.rollout_archive import archive_completed_groups, maybe_archive_completed_groups
from slime.utils.trajectory_metrics import collection_metrics


def trajectory(index, reward, invalid=False):
    image = 'data:image/png;base64,' + base64.b64encode(b'test-png-bytes').decode()
    sample = SimpleNamespace(index=index, group_index=index // 10,
        prompt='prompt', response='action', tokens=[1, 2], loss_mask=[0, 1],
        status='completed', reward=reward, remove_sample=invalid,
        metadata={'trajectory_id': index, 'turn_index': 0,
                  'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': image}]}]},
        multimodal_inputs={'images': [image]}, get_reward_value=lambda args: reward)
    return [sample]


class RolloutArchiveTest(unittest.TestCase):
    def test_all_outcomes_raw_images_and_no_mutation(self):
        groups = [[trajectory(10, 1), trajectory(11, 1)],
                  [trajectory(20, 0), trajectory(21, 0)],
                  [trajectory(30, 1), trajectory(31, 0)],
                  [trajectory(40, 1, True), trajectory(41, None)]]
        before = copy.deepcopy(groups)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(save=directory)
            report = archive_completed_groups(args, groups, [groups[2]], directory, 24)
            self.assertEqual(report['classification_counts'], {'not_accepted/all_success': 1,
                'not_accepted/all_failure': 1, 'accepted/mixed': 1, 'not_accepted/contains_invalid': 1})
            self.assertEqual(report['unique_images'], 1)
            manifest = next(Path(directory).glob('iteration_*/manifest.json'))
            self.assertTrue(json.loads(manifest.read_text())['complete'])
            with gzip.open(manifest.parent / 'group_0000.json.gz', 'rt') as stream:
                group = json.load(stream)
            turn = group['trajectories'][0]['turns'][0]
            image = turn['multimodal_inputs']['images'][0]
            self.assertEqual((Path(directory) / image['archive_image']).read_bytes(), b'test-png-bytes')
            self.assertEqual(turn['tokens'], [1, 2])
            self.assertEqual(turn['metadata']['messages'][0]['content'][0]['image_url'], image)
            self.assertNotIn('multimodal_train_inputs', turn)
        self.assertEqual(groups, before)

    def test_opt_in_collection_id_and_metrics_unchanged(self):
        groups = [[trajectory(10, 1), trajectory(11, 0)]]
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(save=directory)
            with patch.dict('os.environ', {'OPENWEBRL_ARCHIVE_COMPLETED_GROUPS': '0'}):
                baseline = collection_metrics(args, groups, groups, 10)
                self.assertFalse((Path(directory) / 'completed_rollout_archive').exists())
                (Path(directory) / 'rollout_archive.enabled.json').write_text('{}')
                (Path(directory) / 'progress.log').write_text(
                    '[GenerateProgress] rollout=24/90 event=done groups=48/48\n')
                archived = collection_metrics(args, groups, groups, 10)
                self.assertEqual({k: archived[k] for k in baseline}, baseline)
                self.assertEqual(archived['rollout/archive/groups'], 1)
                manifest = next((Path(directory) / 'completed_rollout_archive').glob('*/manifest.json'))
                self.assertEqual(json.loads(manifest.read_text())['reward_iteration'], 24)

    def test_archive_failure_does_not_fail_training(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'rollout_archive.enabled.json').write_text('{}')
            with self.assertLogs('slime.utils.rollout_archive', level='ERROR'):
                result = maybe_archive_completed_groups(SimpleNamespace(save=directory), [], [])
            self.assertEqual(result, {'rollout/archive/errors': 1})
            self.assertTrue((Path(directory) / 'rollout_archive_errors.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
