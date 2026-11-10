"""CPU-only preparation boundaries; no native workers, browser or API clients."""
import importlib.util
import json
import os
import sys
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1]/'scripts/prepare_action_credit_runtime.py'
SPEC = importlib.util.spec_from_file_location('runtime_preparation', PATH)
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


class RuntimePreparationTests(unittest.TestCase):
    def setUp(self):
        # Retain temporary fixtures, consistent with this task's no-deletion rule.
        self.root = Path(tempfile.mkdtemp(prefix='action-runtime-test-', dir='/tmp'))

    def file(self, relative, content=''):
        p = self.root/relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def test_write_boundary_rejects_home_and_symlinks(self):
        with self.assertRaisesRegex(ValueError, 'Home'):
            prep.output_path('/gpfs/home/zixianma/preparation')
        link = self.root/'alias'
        link.symlink_to('/gpfs/home/zixianma', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Home'):
            prep.output_path(link/'preparation')
        target = self.root/'real'; target.mkdir()
        safe_alias = self.root/'safe-alias'; safe_alias.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            prep.output_path(safe_alias/'child')

    def test_dependency_closure_includes_local_function_and_relative_imports(self):
        self.file('openwebrl/a.py', 'from .b import work\ndef later():\n import openwebrl.c\n')
        self.file('openwebrl/b.py', 'def work():\n from .sub import d\n')
        self.file('openwebrl/c.py', 'import json\n')
        self.file('openwebrl/sub/__init__.py')
        self.file('openwebrl/sub/d.py', 'from ..e import x\n')
        self.file('openwebrl/e.py', 'x=1\n')
        found = prep.local_dependencies(self.root, ['openwebrl/a.py'])
        self.assertEqual(found, {'openwebrl/a.py', 'openwebrl/b.py', 'openwebrl/c.py',
            'openwebrl/sub/__init__.py', 'openwebrl/sub/d.py', 'openwebrl/e.py'})

    def test_source_paths_reject_escape_and_symlink(self):
        actual = self.file('actual.py', 'pass\n')
        (self.root/'link.py').symlink_to(actual)
        with self.assertRaises(ValueError): prep.source_file(self.root, '../actual.py')
        with self.assertRaises(ValueError): prep.source_file(self.root, 'link.py')

    def test_safe_environment_never_inherits_credentials_or_home(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'not-a-key', 'HOME': '/gpfs/home/zixianma',
            'WANDB_API_KEY': 'not-a-key', 'OPENWEBRL_REPLAY_BATCH': '/old-batch',
            'HF_HOME': '/gpfs/home/zixianma/cache', 'PYTHONPATH': '/untrusted'}):
            env = prep.base_environment(self.root/'control', self.root/'source')
        self.assertNotIn('HOME', env)
        self.assertNotIn('OPENAI_API_KEY', env)
        self.assertNotIn('WANDB_API_KEY', env)
        self.assertNotIn('OPENWEBRL_REPLAY_BATCH', env)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(env['WANDB_MODE'], 'disabled')
        for key in prep.CACHE_KEYS:
            self.assertTrue(Path(env[key]).is_relative_to(self.root))
        self.assertNotIn('/untrusted', env['PYTHONPATH'])

    def test_native_plan_explicit_no_offload_and_fixed_collector(self):
        command = 'python train.py --colocate --rollout-shuffle --dynamic-sampling-filter-path old.filter --use-rollout-logprobs --rollout-num-gpus-per-engine 1'
        completed = type('Completed', (), {'stdout': command})()
        with patch.object(prep.subprocess, 'run', return_value=completed) as process:
            argv = prep.native_command(self.root, self.root/'source', self.root/'control', '/model', '/dataset')
        self.assertIn('--colocate', argv)
        for flag in ('--no-offload-train', '--no-offload-rollout', '--data-parallel-random-init'):
            self.assertIn(flag, argv)
        for flag in ('--dynamic-sampling-filter-path', '--rollout-shuffle', '--use-rollout-logprobs'):
            self.assertNotIn(flag, argv)
        self.assertEqual(argv[argv.index('--rollout-num-gpus-per-engine')+1], '2')
        self.assertEqual(argv[argv.index('--rollout-num-gpus')+1], '8')
        self.assertEqual(process.call_args.kwargs['env']['DRY_RUN'], '1')
        self.assertFalse((self.root/'control').exists())

    def test_native_plan_refuses_positive_offload_and_duplicate_options(self):
        for command in ('python train.py --offload-train',
                        'python train.py --rollout-shuffle --rollout-shuffle'):
            with patch.object(prep.subprocess, 'run', return_value=type('Completed', (), {'stdout': command})()):
                with self.assertRaises(ValueError):
                    prep.native_command(self.root, self.root/'source', self.root/'control', '/model', '/dataset')

    def test_exclusive_write_preserves_existing_artifact(self):
        path = self.root/'record.json'
        prep.write_exclusive(path, b'{"old":true}\n')
        with self.assertRaises(FileExistsError):
            prep.write_exclusive(path, b'{"new":true}\n')
        self.assertEqual(path.read_bytes(), b'{"old":true}\n')

    def test_copy_preflight_rejects_changed_inputs_without_any_outputs(self):
        source = self.file('input.py', 'pass\n')
        plan = dict(control=str(self.root/'new'), source_files={'input.py':
            dict(path=str(source), sha256='0'*64)}, input_references={})
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            prep.prepare(plan)
        self.assertFalse((self.root/'new').exists())

    def test_external_runtime_byte_change_is_rejected(self):
        path = self.file('external/native.py', 'x=1\n')
        inventory = dict(runtime_root=str(self.root),
            external_python_sources_and_distribution_records={str(path):
                dict(sha256=prep.hash_file(path), size_bytes=path.stat().st_size)})
        prep.verify_external_runtime(inventory)
        path.write_text('x=2\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            prep.verify_external_runtime(inventory)

    def test_partial_parse_uses_native_components_and_never_hardware_validator(self):
        events=[]
        names=['slime','slime.utils','slime.utils.arguments','slime.backends',
               'slime.backends.megatron_utils','slime.backends.megatron_utils.arguments']
        modules={name:ModuleType(name) for name in names}
        for name,module in modules.items():
            parent,_,child=name.rpartition('.')
            if parent:setattr(modules[parent],child,module)
        native=modules['slime.utils.arguments']
        native.configure_logger=lambda:events.append('logger')
        native.get_slime_extra_args_provider=lambda:('actual-provider')
        pre=SimpleNamespace(train_backend='megatron',debug_train_only=False,
            debug_rollout_only=False,load_debug_rollout_data=None)
        native._pre_parse_mode=lambda:pre
        native.sglang_parse_args=lambda:(events.append('sglang_parse') or SimpleNamespace(sglang_tp_size=2))
        def megatron_parse(**kwargs):
            events.append(('megatron_parse',kwargs))
            return SimpleNamespace(hf_config_checked=True)
        module=modules['slime.backends.megatron_utils.arguments']
        module.megatron_parse_args=megatron_parse
        module.validate_args=lambda *a:self.fail('Hardware validator must remain deferred')
        def slime_validate(args):
            self.assertEqual(args.train_backend,'megatron')
            self.assertEqual(args.sglang_tp_size,2)
            events.append('slime_validate')
        native.slime_validate_args=slime_validate
        native.sglang_validate_args=lambda args:events.append('sglang_validate')
        before=sys.argv
        with patch.dict(sys.modules,modules):
            args=prep.parse_cpu_native_arguments(['--no-offload-train'])
            self.assertTrue(args.hf_config_checked)
            self.assertIs(sys.argv,before)
            self.assertEqual(events,['logger','sglang_parse',('megatron_parse',
                {'extra_args_provider':'actual-provider','skip_hf_validate':False}),
                'slime_validate','sglang_validate'])
            pre.debug_train_only=True
            with self.assertRaisesRegex(ValueError,'diagnostic/replay'):
                prep.parse_cpu_native_arguments(['--debug-train-only'])
            self.assertIs(sys.argv,before)


if __name__ == '__main__':
    unittest.main()
