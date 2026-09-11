"""CPU regression checks for an evaluation interrupted at an allocation boundary."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_resume_pending_eval as p
import resume_baseline as r


class PendingEvaluationTest(unittest.TestCase):
    def helper(self):
        namespace = dict(os=os, append_progress_log=Mock(), _ray_get_with_actor_retry=Mock(),
                         evict_file_cache=Mock(), evict_file_backed_cache=Mock())
        exec(compile(ast.parse(p.HELPER), '<pending-eval-test>', 'exec'), namespace)
        return namespace

    def test_recovery_uses_checkpoint_iteration_and_releases_cache(self):
        ns = self.helper()
        manager = SimpleNamespace(eval=SimpleNamespace(remote=Mock(return_value='future')))
        args = SimpleNamespace(start_rollout_id=40, eval_interval=10,
                               save_debug_rollout_data='/run/recovery/{rollout_id}.pt')
        with patch.dict(os.environ, {'OPENWEBRL_PENDING_EVAL_ITERATION':'40'}):
            ns['_finish_pending_evaluation'](args, manager)
        manager.eval.remote.assert_called_once_with(39)
        ns['_ray_get_with_actor_retry'].assert_called_once()
        ns['evict_file_cache'].assert_called_once_with('/run/recovery/eval_39.pt', sync=True)
        ns['evict_file_backed_cache'].assert_called_once()
        self.assertIn('status=completed', ns['append_progress_log'].call_args.args[1])

    def test_failure_cannot_report_completion_or_continue(self):
        ns = self.helper()
        ns['_ray_get_with_actor_retry'].side_effect = RuntimeError('evaluation failed')
        manager = SimpleNamespace(eval=SimpleNamespace(remote=Mock()))
        args = SimpleNamespace(start_rollout_id=40, eval_interval=10)
        with patch.dict(os.environ, {'OPENWEBRL_PENDING_EVAL_ITERATION':'40'}):
            with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                ns['_finish_pending_evaluation'](args, manager)
        self.assertEqual(ns['append_progress_log'].call_count, 1)
        self.assertIn('status=started', ns['append_progress_log'].call_args.args[1])

    def test_normal_resume_has_no_extra_eval_and_wrong_boundary_is_rejected(self):
        ns = self.helper()
        with patch.dict(os.environ, {}, clear=True):
            ns['_finish_pending_evaluation'](None, None)
        ns['_ray_get_with_actor_retry'].assert_not_called()
        for iteration in ['39', '41', '-10']:
            with patch.dict(os.environ, {'OPENWEBRL_PENDING_EVAL_ITERATION':iteration}):
                with self.assertRaises(ValueError):
                    ns['_finish_pending_evaluation'](SimpleNamespace(start_rollout_id=40, eval_interval=10), None)

    def test_prepared_source_is_bound_to_original_and_its_verified_patch(self):
        root = Path(tempfile.mkdtemp(prefix='pending-eval-test-'))
        source, prepared = root/'original', root/'prepared'
        prepared.mkdir()
        text = p.HELPER.encode()
        (prepared/'train.py').write_bytes(text)
        (prepared/'reference_manifest.json').write_text(json.dumps({'pending_evaluation_recovery':{
            'source':str(source), 'train_sha256':hashlib.sha256(text).hexdigest()}}))
        state = dict(pending_evaluation_iteration_one_based=40, pending_evaluation_resume_source=str(prepared))
        self.assertEqual(r.pending_evaluation_source(state, source), prepared)
        self.assertEqual(r.pending_evaluation_source({}, source), source)
        with self.assertRaisesRegex(ValueError, 'not prepared from'):
            r.pending_evaluation_source(state, root/'unrelated')
        (prepared/'train.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Prepare and verify'):
            r.pending_evaluation_source(state, source)

    def test_stale_pending_environment_is_removed(self):
        with patch.dict(os.environ, {'OPENWEBRL_PENDING_EVAL_ITERATION':'40'}):
            self.assertNotIn('OPENWEBRL_PENDING_EVAL_ITERATION', r.clean_environment())


if __name__ == '__main__':
    unittest.main()
