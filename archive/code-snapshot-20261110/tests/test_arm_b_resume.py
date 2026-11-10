"""CPU regressions for replaying B's unoptimized iteration50 after checkpoint49."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import prepare_arm_b_to60 as continuation
from test_arm_gate_recovery import SavedAuxiliary


def isolated(name):
    module = ModuleType(name)
    exec(compile(continuation.changed_files()[f'openwebrl/{name}.py'], name, 'exec'), module.__dict__)
    return module


class ResumeBatch(unittest.TestCase):
    def fixture(self, tmp):
        args, samples, current, report = SavedAuxiliary().fixture(tmp)
        origin = current['config']['gate_replay_origin']
        folder = Path(origin['source'])/'iterations/0000'
        target = folder.with_name('0049');folder.rename(target)
        manifest = json.loads((target/'failure_auxiliary.json').read_text())
        for obj in (manifest, current['config']):
            obj.update(rollout_id=49, policy_id='original:rollout49', checkpoint='checkpoint49')
        data = json.dumps(manifest).encode();(target/'failure_auxiliary.json').write_bytes(data)
        origin.update(rollout_id=49, auxiliary_manifest_sha256=hashlib.sha256(data).hexdigest(),
                      calibration_sha256=hashlib.sha256((target/'calibration.json').read_bytes()).hexdigest())
        return args, samples, current, report

    def test_exact_failure_auxiliary_replayed_at49_only(self):
        replay = isolated('arm_gate_recovery').replay_auxiliary
        with tempfile.TemporaryDirectory() as tmp:
            args, samples, current, report = self.fixture(tmp)
            self.assertTrue(replay(args, samples, current, report))
            self.assertEqual(report['additive_failure_rows'], 400)
            self.assertEqual(report['outcome_groups_preserved'], 48)
            self.assertTrue((Path(current['config']['output'])/'failure_auxiliary.pt').is_symlink())
            current['config']['rollout_id'] = 50
            self.assertFalse(replay(args, samples, current, report))

    def test_wrong_batch_or_calibration_fails_closed(self):
        replay = isolated('arm_gate_recovery').replay_auxiliary
        for kind in ('calibration', 'checkpoint', 'policy', 'labels'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                args, samples, current, report = self.fixture(tmp)
                if kind == 'calibration':current['config']['gate_replay_origin']['calibration_sha256'] = 'bad'
                elif kind == 'labels':report['admitted'] -= 1
                else:current['config']['policy_id' if kind == 'policy' else kind] = 'wrong'
                with self.assertRaises(ValueError):replay(args, samples, current, report)

    def test_legacy_first_batch_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, samples, current, report = SavedAuxiliary().fixture(tmp)
            self.assertTrue(isolated('arm_gate_recovery').replay_auxiliary(args, samples, current, report))

    def test_replay_alias_then_fresh_checkpoint_identity(self):
        module = isolated('arm_turn_bonus_cycles')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);old = root/'old/iter_0000048';old.mkdir(parents=True)
            (old/'common.pt').write_bytes(b'checkpoint')
            save = root/'new/runtime';save.mkdir(parents=True)
            (save/old.name).symlink_to(old)
            fresh = save/'iter_0000049';fresh.mkdir();(fresh/'common.pt').write_bytes(b'next')
            config = dict(run_output=str(root/'new'), run_id='original', initial_checkpoint='sft',
                          deadline_epoch_seconds=time.time()+10000,
                          gate_replay_origin=dict(rollout_id=49, checkpoint=str(old)))
            args = SimpleNamespace(save=str(save), start_rollout_id=49)
            with patch.object(module, 'config', return_value=config), patch.object(module, 'teacher_phase'), \
                 patch.object(module, 'write_json'), patch.dict(module.os.environ, OPENWEBRL_ARM_TURN_BONUS_CONFIG='unused'):
                self.assertFalse(module.before_collection(args, 49))
                self.assertEqual(config['checkpoint'], str(old))
                self.assertFalse(module.before_collection(args, 50))
                self.assertEqual(config['checkpoint'], str(fresh))
                self.assertEqual(config['policy_id'], 'original:rollout50')

    def test_replay_rejects_different_actor(self):
        module = isolated('arm_turn_bonus_cycles')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ckpt=root/'iter_0000048';ckpt.mkdir();(ckpt/'common.pt').touch()
            config=dict(run_output=tmp,deadline_epoch_seconds=time.time()+10000,
                        gate_replay_origin=dict(rollout_id=49,checkpoint=str(root/'wrong')))
            with patch.object(module,'config',return_value=config):
                with self.assertRaises(ValueError):
                    module.before_collection(SimpleNamespace(save=tmp,start_rollout_id=49),49)


if __name__ == '__main__':unittest.main()
