"""Offline prospective sweep tests: real actor/budget logic, fake SDK boundary."""
from __future__ import annotations

import ast
from copy import deepcopy
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

API = Path(__file__).resolve().parents[1]
REPRO = API / 'reproduce'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


p = module('sweep_preparation', REPRO / 'prospective_browsercode.py')
c = module('sweep_common', REPRO / 'common.py')
b = module('sweep_browsercode', REPRO / 'browsercode.py')
budget_module = module('sweep_real_budget', API / 'browsercode/budgets/request_budget.py')


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='browsercode-output-sweep-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path_patch = patch.object(sys, 'path', [str(REPRO), *sys.path])
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def runtime(self, name='runtime'):
        runtime = self.root / name
        runtime.mkdir()
        spec_dir = API / 'portable_assets/campaign_specs/browsercode_luna_effort'
        config = p.read(spec_dir / 'runtime-config.json')
        for key in ('cuawright_source', 'frozen_support_source', 'chromium', 'python'):
            config[key] = str(runtime / 'dependency')
        config.pop('support_source_hashes', None)
        config['portable_campaign'] = p.CAMPAIGN
        p.write(runtime / 'runtime-config.json', config)
        from tasks import load_tasks
        p.write(runtime / 'tasks.json', load_tasks(p.CAMPAIGN))
        plan = p.read(spec_dir / 'campaign-plan.json')
        plan['task_manifest_sha256'] = p.digest(runtime / 'tasks.json')
        p.write(runtime / 'campaign-plan.json', plan)
        p.write(runtime / 'portable-build.json', {'adaptations': []})
        for name in ('browser_bridge.ts', 'browsercode', 'node_modules', 'javascript-runtime/bun', 'sandbox.py', 'dependency'):
            path = runtime / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        (runtime / 'luna_model.py').write_bytes((API / 'browsercode/models/luna_model.py').read_bytes())
        (runtime / 'worker.py').write_bytes((API / 'browsercode/workers/luna_effort_worker.py').read_bytes())
        manifest = {'plan_sha256': p.digest(runtime / 'campaign-plan.json'),
                    'source_hashes': {str(path): p.digest(path) for path in runtime.rglob('*') if path.is_file()}}
        p.write(runtime / 'source-manifest.json', manifest)
        p.write(runtime / 'execution-source-manifest.json', manifest)
        return runtime

    def args(self, runtime, *extra):
        return c.parser('luna').parse_args(['--runtime', str(runtime), '--arm', p.ARM, '--output', str(self.root / 'output'), '--model-usd', '2', '--judge-usd', '.1', *extra])

    def test_all_three_caps_freeze_same_original_roster_and_science(self):
        identities = []
        for cap in p.CAPS:
            runtime = self.runtime(str(cap))
            originals = {path: path.read_bytes() for path in (API / 'browsercode/models/luna_model.py', API / 'browsercode/workers/luna_effort_worker.py')}
            receipt = p.configure(runtime, cap)
            self.assertEqual(p.validate(runtime), receipt)
            plan, tasks = c.prepare(self.args(runtime, '--limit', '1045'), 'luna')
            self.assertEqual((plan['max_output_tokens'], plan['candidate_effort'], plan['candidate_count'], plan['max_decisions']), (cap, 'xhigh', 1, 30))
            self.assertEqual(plan['benchmark_roster_counts'], p.COUNTS)
            self.assertEqual(plan['output_multiplier'], cap // 12288)
            identities.append([task['task_id'] for task in tasks])
            self.assertFalse((self.root / 'output').exists())
            for source, before in originals.items():
                self.assertEqual(source.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, 'already configured'):
                p.configure(runtime, cap)
        self.assertEqual(identities[0], identities[1])
        self.assertEqual(identities[1], identities[2])

    def test_default_runtime_bytes_and_cap_remain_unchanged(self):
        runtime = self.runtime()
        before = {path: path.read_bytes() for path in runtime.rglob('*') if path.is_file()}
        plan, _ = c.prepare(self.args(runtime), 'luna')
        self.assertEqual(plan['max_output_tokens'], 12288)
        self.assertNotIn('prospective_protocol', plan)
        self.assertEqual(before, {path: path.read_bytes() for path in runtime.rglob('*') if path.is_file()})
        frozen = p.read(runtime / 'campaign-plan.json')
        frozen['arms'][0]['output_tokens'] = 49152
        p.write(runtime / 'campaign-plan.json', frozen)
        with self.assertRaisesRegex(ValueError, 'frozen BrowserCode'):
            c.prepare(self.args(runtime), 'luna')

    def test_selection_is_ordered_strict_hashed_and_filtered_before_limit(self):
        runtime = self.runtime()
        p.configure(runtime, 24576)
        rows = p.read(runtime / 'tasks.json')
        chosen = [rows[301]['task_id'], rows[300]['task_id']]
        selection = self.root / 'selection.json'
        p.write(selection, chosen)
        args = self.args(runtime, '--task-ids', str(selection), '--benchmark', 'webvoyager', '--limit', '1')
        plan, tasks = c.prepare(args, 'luna')
        self.assertEqual([row['task_id'] for row in tasks], chosen[:1])
        self.assertEqual(plan['task_selection_sha256'], p.digest(selection))
        self.assertEqual(plan['task_selection_count'], 2)
        self.assertEqual(plan['selected_task_ids'], chosen[:1])
        for invalid in ([chosen[0], chosen[0]], ['missing'], [1], [], {}, ['../escape'], ['..'], [rows[0]['task_id']]):
            p.write(selection, invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                c.prepare(args, 'luna')

    def test_unpaired_malformed_and_rehashed_source_mutations_fail_before_execution(self):
        for mutation in ('missing-receipt', 'bad-receipt', 'worker', 'model', 'config', 'plan', 'task'):
            runtime = self.runtime(mutation)
            p.configure(runtime, 49152)
            if mutation == 'missing-receipt':
                (runtime / p.RECEIPT).rename(runtime / 'missing-receipt.json')
            elif mutation == 'bad-receipt':
                p.write(runtime / p.RECEIPT, {'schema': p.SCHEMA})
            else:
                name = {'worker': 'worker.py', 'model': 'luna_model.py', 'config': 'runtime-config.json', 'plan': 'campaign-plan.json', 'task': 'tasks.json'}[mutation]
                path = runtime / name
                path.write_text(path.read_text() + '\n')
                manifest = p.read(runtime / 'source-manifest.json')
                manifest['source_hashes'][str(path)] = p.digest(path)
                p.write(runtime / 'source-manifest.json', manifest)
                p.write(runtime / 'execution-source-manifest.json', manifest)
            with self.subTest(mutation=mutation), patch.object(c, 'execute', side_effect=AssertionError('no provider/browser execution')), self.assertRaises(ValueError):
                c.prepare(self.args(runtime), 'luna')

    def test_no_override_for_other_campaign_or_arm_or_home(self):
        args = b.parser().parse_args(['setup', '--campaign', 'gpt', '--workspace', str(self.root / 'new'), '--luna-xhigh-output-tokens', '24576'])
        with patch.object(b, 'call', side_effect=AssertionError('must reject before dependencies')), self.assertRaisesRegex(ValueError, 'requires --campaign luna'):
            b.setup(args)
        self.assertFalse((self.root / 'new').exists())
        runtime = self.runtime()
        p.configure(runtime, 24576)
        args = self.args(runtime)
        args.arm = 'luna_single_medium'
        with self.assertRaisesRegex(ValueError, 'single high/xhigh'):
            c.prepare(args, 'luna')
        with self.assertRaisesRegex(ValueError, 'home is read-only'):
            p.configure('/gpfs/home/zixianma/forbidden-new-runtime', 24576)
        for invalid in (True, 12289, '24576'):
            with self.assertRaises(ValueError):
                p.configure(runtime, invalid)

    def load_actor(self, runtime, prospective=True):
        class OpenAIModel:
            def __init__(self, **kwargs): self.constructor = kwargs
            def _build_payload(self, messages): return {'input': deepcopy(messages), 'include': ['reasoning.encrypted_content']}
            def _response_extra(self, native): return {'response_items': native['output']}
            def format_message(self, **kwargs): return kwargs
        fake = {
            'request_budget': budget_module,
            'cuawright.webwright.models.openai_model': types.SimpleNamespace(OpenAIModel=OpenAIModel),
            'openwebrl.responses_actor_history': types.SimpleNamespace(response_input_bound=lambda *unused: 1000),
            'openwebrl.sol_serial_parallel_policy': types.SimpleNamespace(selector_input=lambda *unused: []),
        }
        actor = types.ModuleType('test_actor')
        actor.__file__ = str(runtime / 'luna_model.py')
        source = (runtime / 'luna_model.py').read_text()
        with patch.dict(sys.modules, fake):
            exec(compile(source, actor.__file__, 'exec'), actor.__dict__)
        return actor

    def test_actual_payload_reservation_constructor_and_tracking_share_declared_cap(self):
        for cap, effort in [(cap, 'xhigh') for cap in p.CAPS] + [(24576, 'high'), (49152, 'high')]:
            arm = p.EFFORT_ARMS[effort]
            runtime = self.runtime('actor-' + str(cap) + '-' + effort)
            p.configure(runtime, cap, effort)
            actor = self.load_actor(runtime)
            calls = []
            def client(payload, role):
                calls.append((deepcopy(payload), role))
                return {'id': 'offline-response', 'model': actor.MODEL, 'status': 'completed', 'reasoning': payload['reasoning'],
                        'output': [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'finished'}]}],
                        'usage': {'input_tokens': 100, 'output_tokens': 20}}
            pool = budget_module.ReservationBudget(self.root / ('pool-' + str(cap) + effort), '1', 3)
            model = actor.LunaModel(arm, self.root / ('episode-' + str(cap) + effort), client, pool, actor.PRICING,
                                    max_output_tokens=cap, reasoning_effort=effort, http_gate=budget_module.SharedHTTPGate(self.root / 'gate', 8))
            model.query([{'role': 'user', 'content': 'real observation'}])
            self.assertEqual(model.constructor['max_output_tokens'], cap)
            self.assertEqual(len(calls), 1)
            payload, role = calls[0]
            self.assertEqual((payload['max_output_tokens'], payload['reasoning'], role), (cap, {'effort': effort}, 'candidate_0'))
            self.assertEqual((payload['store'], payload['service_tier'], payload['parallel_tool_calls']), (False, 'default', False))
            receipt = next((model.directory / 'requests').glob('luna-*.json'))
            expected = (Decimal(1000) * Decimal('.125') + Decimal(cap) * Decimal('.5')) / Decimal(1000000)
            self.assertEqual(Decimal(p.read(receipt)['reserved_usd_decimal']), expected)
            state = pool.snapshot()
            self.assertEqual(state['calls'], 1)
            self.assertEqual(Decimal(next(iter(state['requests'].values()))['reserved_usd']), expected)
            worker = ast.parse((runtime / 'worker.py').read_text())
            tracking = next(node for node in ast.walk(worker) if isinstance(node, ast.Dict) and any(isinstance(key, ast.Constant) and key.value == 'max_output_tokens' for key in node.keys))
            value = next(value for key, value in zip(tracking.keys, tracking.values) if isinstance(key, ast.Constant) and key.value == 'max_output_tokens')
            tracked = eval(compile(ast.Expression(value), '<tracking cap>', 'eval'), {'PLAN': p.read(runtime / 'campaign-plan.json'), 'a': types.SimpleNamespace(arm=arm)})
            self.assertEqual(tracked, cap)
            with self.assertRaises(ValueError):
                actor.LunaModel(arm, self.root / 'bad', client, pool, max_output_tokens=cap + 1)
            with self.assertRaises(ValueError):
                actor.LunaModel(arm, self.root / 'wrong-effort', client, pool, max_output_tokens=cap, reasoning_effort='low')
            wrong = dict(payload, max_output_tokens=cap + 1)
            with self.assertRaises(ValueError):
                model.physical(wrong, 'candidate_0')
            self.assertEqual(pool.snapshot()['calls'], 1)

    def test_user_grid_binds_distinct_effort_arm_and_identical_tasks(self):
        task_hashes = []
        for cap, effort in ((12288, 'xhigh'), (24576, 'high'), (49152, 'high')):
            runtime = self.runtime('grid-' + str(cap))
            receipt = p.configure(runtime, cap, effort)
            args = self.args(runtime, '--limit', '1045')
            args.arm = p.EFFORT_ARMS[effort]
            plan, tasks = c.prepare(args, 'luna')
            self.assertEqual((plan['arm'], plan['candidate_effort'], plan['max_output_tokens']), (args.arm, effort, cap))
            self.assertEqual(receipt['reasoning_effort'], effort)
            self.assertEqual(receipt['arm'], args.arm)
            self.assertEqual(len(tasks), 1045)
            task_hashes.append(plan['task_manifest_sha256'])
            args.arm = p.EFFORT_ARMS['high' if effort == 'xhigh' else 'xhigh']
            with self.assertRaisesRegex(ValueError, 'arm/effort'):
                c.prepare(args, 'luna')
        self.assertEqual(len(set(task_hashes)), 1)

    def test_setup_alias_compatibility_and_conflicting_overrides(self):
        valid = b.parser().parse_args(['setup', '--campaign', 'luna', '--workspace', str(self.root / 'new'), '--luna-single-output-tokens', '24576', '--luna-single-reasoning-effort', 'high'])
        self.assertEqual((valid.luna_single_output_tokens, valid.luna_single_reasoning_effort), (24576, 'high'))
        for extra in [
            ['--luna-xhigh-output-tokens', '12288', '--luna-single-reasoning-effort', 'high'],
            ['--luna-xhigh-output-tokens', '12288', '--luna-single-output-tokens', '12288'],
            ['--luna-single-reasoning-effort', 'high'],
        ]:
            args = b.parser().parse_args(['setup', '--campaign', 'luna', '--workspace', str(self.root / 'new'), *extra])
            with self.subTest(extra=extra), patch.object(b, 'call', side_effect=AssertionError('must reject before setup')), self.assertRaises(ValueError):
                b.setup(args)
        runtime = self.runtime()
        args = self.args(runtime)
        args.arm = 'luna_single_high'
        with self.assertRaises(ValueError):
            c.prepare(args, 'luna')

    def test_original_actor_still_rejects_higher_cap_before_provider(self):
        runtime = self.runtime()
        actor = self.load_actor(runtime)
        pool = budget_module.ReservationBudget(self.root / 'baseline-pool', '1', 3)
        with self.assertRaises(ValueError):
            actor.LunaModel(p.ARM, self.root / 'unused', lambda *args: self.fail('provider called'), pool, max_output_tokens=24576)
        self.assertEqual(pool.snapshot()['calls'], 0)

    def test_frontdoor_forwards_explicit_selection_without_execution(self):
        args = b.parser().parse_args(['run', '--workspace', str(self.root), '--arm', p.ARM, '--output', str(self.root / 'out'), '--model-usd', '1', '--judge-usd', '.1', '--task-ids', str(self.root / 'ids.json')])
        calls = []
        with patch.object(b, 'workspace', return_value=(self.root, Path(sys.executable))), patch.object(b, 'provider', return_value='luna'), patch.object(b, 'call', side_effect=lambda *values: calls.append(list(map(str, values))) or 0):
            self.assertEqual(b.run(args), 0)
        self.assertEqual(calls[0][calls[0].index('--task-ids') + 1], str(self.root / 'ids.json'))
        self.assertNotIn('--execute', calls[0])

    def test_fresh_interpreter_dry_run_loads_override_without_sdk_or_output(self):
        runtime = self.runtime()
        p.configure(runtime, 49152)
        result = subprocess.run([sys.executable, '-B', str(REPRO / 'eval_luna.py'), '--runtime', str(runtime), '--arm', p.ARM,
                                 '--output', str(self.root / 'fresh-output'), '--model-usd', '1', '--judge-usd', '.1'],
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"max_output_tokens": 49152', result.stdout)
        self.assertIn('Dry run:', result.stdout)
        self.assertFalse((self.root / 'fresh-output').exists())


if __name__ == '__main__':
    unittest.main()
