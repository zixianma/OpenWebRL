"""Source-to-report checks using realistic saved dispositions and durable ledgers."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[1] / 'reproduce/report.py'
spec = importlib.util.spec_from_file_location('collaborator_report', SOURCE)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='api-report-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, root, name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def browser(self, label, arm, rows, ids=('a', 'b', 'c'), judge='terra', benchmark='online-mind2web'):
        root, runtime = self.root / label, self.root / (label + '-runtime')
        root.mkdir()
        self.write(runtime, 'runtime-config.json', {'judge_protocol': {'model': judge}, 'system_templates': {'browsercode': 'frozen prompt'}, 'observation_template': 'frozen observation'})
        self.write(runtime, 'tasks.json', [{'task_id': ident, 'intent': ident, 'start_url': 'https://example.test/' + ident, 'benchmark': benchmark} for ident in ids])
        self.write(root, 'reproduction-plan.json', dict(arm=arm, runtime=str(runtime), model='actor-' + arm, candidate_count=1, max_decisions=30, max_output_tokens=12288, source_manifest_sha256='source', task_manifest_sha256='tasks'))
        self.write(root, 'selected-task-ids.json', list(ids))
        for ident, valid, reward, latency in rows:
            self.write(root, f'episodes/{ident}-browsercode-{arm}/result.json', dict(task_id=ident, mode=arm, valid=valid, reward=reward, task_latency_seconds=latency, error='secret raw diagnostic excluded'))
        self.write(root, 'budget/model/ledger.json', {'calls': 2, 'charged_or_reserved_usd': '.3', 'requests': {'ok': {'status': 'known', 'charged_or_reserved_usd': '.1'}, 'unknown': {'status': 'usage_unknown', 'charged_or_reserved_usd': '.2'}}})
        self.write(root, 'budget/judge/ledger.json', {'calls': 1, 'charged_or_reserved_usd': .05})
        return root

    def test_source_counts_fees_missing_values_and_common_valid_pairs(self):
        left = self.browser('left', 'high', [('a', True, 1, 10), ('b', False, None, 900), ('c', True, 0, None)])
        right = self.browser('right', 'parallel', [('a', True, 0, 20), ('b', True, 1, 20)])
        report = r.build_report([('L', left), ('R', right)], draws=500, seed=42)
        group = report['runs'][0]['groups'][0]
        self.assertEqual([group[key] for key in ('requested', 'recorded', 'valid', 'invalid', 'successes')], [3, 3, 2, 1, 1])
        self.assertEqual(group['rates']['valid_only_success'], .5)
        self.assertEqual(group['rates']['recorded_success'], 1 / 3)
        self.assertEqual(group['valid_task_efficiency']['task_latency_seconds'], dict(mean=10, observations=1, missing=1))
        self.assertEqual(group['valid_task_efficiency']['model_cost_upper_usd'], dict(mean=None, observations=0, missing=2))
        pool = report['runs'][0]['all_attempt_accounting']['pools']['model']
        self.assertEqual(pool['known_usage_fee_upper_usd'], '0.1')
        self.assertEqual(pool['unresolved_charge_or_reserve_usd'], '0.2')
        self.assertIsNone(report['runs'][0]['all_attempt_accounting']['pools']['judge']['known_usage_fee_upper_usd'])
        comparison = report['comparisons'][0]
        self.assertTrue(comparison['eligible'])
        self.assertEqual([comparison[key] for key in ('common_requested', 'common_recorded', 'common_valid', 'excluded_invalid', 'excluded_unrecorded')], [3, 2, 1, 1, 1])
        self.assertEqual(comparison['success_rate_difference'], 1)
        self.assertEqual(comparison['paired_bootstrap_95_percentile_interval'], [1, 1])
        self.assertEqual(comparison['mcnemar'], dict(left_only_successes=1, right_only_successes=0, exact_two_sided_p=1))
        self.assertNotIn('secret raw diagnostic', json.dumps(report))
        self.assertEqual(report, r.build_report([('L', left), ('R', right)], draws=500, seed=42))

    def test_cli_writes_new_report_and_never_overwrites(self):
        root = self.browser('cli', 'high', [('a', True, 1, 2)])
        output = self.root / 'report.json'
        command = [sys.executable, str(SOURCE), '--run', 'fresh=' + str(root), '--output', str(output), '--markdown', '--bootstrap-draws', '20']
        run = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('| Requested | Recorded | Valid | Invalid | Successes |', run.stdout)
        before = output.read_bytes()
        retry = subprocess.run(command, text=True, capture_output=True)
        self.assertNotEqual(retry.returncode, 0)
        self.assertEqual(output.read_bytes(), before)
        with self.assertRaises(ValueError):
            r.build_report([('first', root), ('duplicate', root)], draws=20)

    def test_exact_mcnemar_and_deterministic_paired_interval(self):
        ids = tuple(str(i) for i in range(10))
        a = self.browser('ten-a', 'high', [(ident, True, 1, 1) for ident in ids], ids=ids)
        b = self.browser('ten-b', 'parallel', [(ident, True, 0, 1) for ident in ids], ids=ids)
        comparison = r.build_report([('A', a), ('B', b)], draws=50)['comparisons'][0]
        self.assertEqual(comparison['mcnemar']['exact_two_sided_p'], 2 / 1024)
        self.assertEqual(comparison['paired_bootstrap_95_percentile_interval'], [1, 1])

    def test_protocol_and_task_identity_differences_prevent_matching(self):
        a = self.browser('a', 'high', [('a', True, 1, 1)])
        b = self.browser('b', 'parallel', [('a', True, 0, 1)], judge='other')
        comparison = r.build_report([('A', a), ('B', b)], draws=10)['comparisons'][0]
        self.assertFalse(comparison['eligible'])
        config = r.read(self.root / 'b-runtime/runtime-config.json')
        config['judge_protocol']['model'] = 'terra'
        self.write(self.root / 'b-runtime', 'runtime-config.json', config)
        tasks = r.read(self.root / 'b-runtime/tasks.json')
        tasks[0]['intent'] = 'different task under reused ID'
        self.write(self.root / 'b-runtime', 'tasks.json', tasks)
        self.assertFalse(r.build_report([('A', a), ('B', b)], draws=10)['comparisons'][0]['eligible'])

    def test_no_records_still_reports_requested_and_missing_ledger(self):
        root = self.browser('empty', 'high', [])
        report = r.build_report([('empty', root)], draws=10)
        group = report['runs'][0]['groups'][0]
        self.assertEqual((group['requested'], group['recorded'], group['unrecorded']), (3, 0, 3))
        self.assertIsNone(group['rates']['valid_only_success'])

    def test_invalid_verdict_duplicate_or_outside_cohort_is_rejected(self):
        for label, changes in [('boolean', {'valid': True, 'reward': True}), ('missing', {'valid': True, 'reward': None}), ('outside', {'task_id': 'wrong'})]:
            root = self.browser(label, 'high', [('a', True, 1, 1)])
            path = root / 'episodes/a-browsercode-high/result.json'
            row = r.read(path)
            row.update(changes)
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                r.build_report([(label, root)], draws=10)
        root = self.browser('duplicate', 'high', [('a', True, 1, 1)])
        self.write(root, 'episodes/second/result.json', r.read(root / 'episodes/a-browsercode-high/result.json'))
        with self.assertRaises(ValueError):
            r.build_report([('duplicate', root)], draws=10)

    def test_native_gateway_accounting_without_mutating_or_reading_group_secret(self):
        root = self.root / 'native'
        plan = dict(schema='cuawright-collaborator-run-v1', arm_id='luna_single', model='luna', max_decisions=100, summary_setting=20, judge='webjudge', retry_policy='none', fallback_policy='none', shards=[{'task_ids': ['a', 'b']}])
        self.write(root, 'plan.json', plan)
        self.write(root, 'tasks.json', [{'task_id': ident, 'intent': ident, 'start_url': 'https://example.test'} for ident in ('a', 'b')])
        self.write(root, 'shard0/episodes/a/disposition.json', dict(task_id='a', arm='luna_single', valid=True, reward=1, task_latency_seconds=2))
        self.write(root, 'shard0/.host/group-config.json', {'group_secret': 'DO_NOT_EXPORT'})
        meter = root / 'shard0/.host/meter/events.jsonl'
        meter.parent.mkdir(parents=True)
        events = [dict(event='reserve', id='one', pool='model', reserve_usd=.5), dict(event='finish', id='one', usage_known=True, charge_usd=.1), dict(event='reserve', id='two', pool='judge', reserve_usd=.2)]
        meter.write_text(''.join(json.dumps(event) + '\n' for event in events))
        before = sorted(str(path.relative_to(root)) for path in root.rglob('*'))
        report = r.build_report([('native', root)], draws=10)
        self.assertEqual(before, sorted(str(path.relative_to(root)) for path in root.rglob('*')))
        self.assertNotIn('DO_NOT_EXPORT', json.dumps(report))
        pools = report['runs'][0]['all_attempt_accounting']['pools']
        self.assertEqual(pools['model']['known_usage_fee_upper_usd'], '0.1')
        self.assertEqual(pools['judge']['unresolved_calls'], 1)
        self.assertEqual(report['runs'][0]['groups'][0]['unrecorded'], 1)

    def test_legacy_multiarm_schedule_artifact_integrity_and_shared_costs(self):
        root = self.root / 'legacy'
        protocol = dict(actor='sol', judge='judge', judge_protocol='frozen', max_steps=30, viewport=[1280, 720])
        self.write(root, 'portable-execution.json', {'campaign': 'legacy'})
        self.write(root, 'plan.json', {'protocol': protocol, 'schedule': [{'task_id': 'a', 'modes': ['high', 'parallel']}], 'task_file_sha256': 'tasks'})
        tasks = root / 'tasks.jsonl'
        tasks.write_text(json.dumps({'id': 'a', 'metadata': {'task_id': 'a', 'intent': 'task', 'start_url': 'https://example.test'}}) + '\n')
        for arm, reward in [('high', 1), ('parallel', 0)]:
            artifact = self.write(root, f'attempts/{arm}/result.json', {'reward': reward})
            self.write(root, f'records/{arm}.json', {'task_id': 'a', 'mode': arm, 'valid': True, 'reward': reward, 'protocol_sha256': __import__('hashlib').sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(), 'artifact': str(artifact), 'artifact_sha256': r.evidence(artifact, root)['sha256'], 'browser_steps': 4, 'model_cost_upper_usd': .04})
        report = r.build_report([('legacy', root)], draws=10)
        self.assertEqual(len(report['runs'][0]['groups']), 2)
        self.assertTrue(report['comparisons'][0]['eligible'])
        self.assertIsNone(report['runs'][0]['all_attempt_accounting']['pools']['model'])
        artifact.write_text('{}')
        with self.assertRaises(ValueError):
            r.build_report([('legacy', root)], draws=10)


if __name__ == '__main__':
    unittest.main()
