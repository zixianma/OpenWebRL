import asyncio
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl import arm_task_pass8 as p8
from openwebrl.arm_terminal_budget import CappedJudge


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = dict(tasks=str(self.root/'tasks.jsonl'), output=str(self.root/'new'),
            attempt_output=str(self.root/'new'/'attempts'), seed=p8.SEED, policy_id=p8.POLICY,
            max_steps=15, inference_timeout=180,
            historical_task_order=[f'task{i}' for i in range(2000)],
            extension_task_order=[f'task{i}' for i in range(707)],
            task_order=[f'task{i}' for i in range(0,707,4)],
            partition_manifest=str(self.root/'partition.json'), shard=0,
            old_record_roots=[str(self.root/'old'/'records')])
        self.args = SimpleNamespace(judge_api_model='gpt-4.1', judge_prompt_variant='action_history',
                                    num_rollout=0, max_steps=15)
        self.fake_torch = ModuleType('torch')
        self.fake_torch.load = lambda path, **kwargs: json.loads(Path(path).read_text())

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, task='task0', attempt=5, *, valid=True, reward=0., output=None):
        artifact = Path(output or self.root/'fixtures')/'trajectories'/p8.key(task, attempt)
        artifact.mkdir(parents=True, exist_ok=True)
        archive = artifact/(hashlib.sha256(task.encode()).hexdigest()+'.pt')
        terminal_status = 'completed' if reward == 1 else 'failed'
        meta = dict(combined=reward, judge=float(reward == 1), judge_text='SUCCESS' if reward == 1 else '')
        terminal = dict(metadata=dict(turn_index=0, is_last_turn=True, reward=meta),
                        status=terminal_status, reward=reward, remove_sample=not valid,
                        multimodal_inputs={'images': ['saved-image-for-accounting-test']})
        archive.write_text(json.dumps(dict(task_id=task, turns=[terminal], error_type=None)))
        archive.with_suffix('.json').write_text(json.dumps(dict(task_id=task,
            rollout_file=archive.name, turns=1, error_type=None, judge_model='gpt-4.1',
            judge_prompt_variant='action_history', terminal_status=terminal_status,
            metrics=dict(valid_trajectories=int(valid)), reward_metadata=meta)))
        return dict(task_id=task, attempt=attempt, valid=valid, reward=reward if valid else None,
            policy_id=p8.POLICY, judge_id=p8.JUDGE, phase='screen', mode='actor',
            selector_calls=0, selector_fallback_turns=0, seed=p8.seed_for(task, attempt),
            error_type=None, steps=1, artifact=str(artifact), actor_requests=1,
            actor_output_tokens=12, elapsed_seconds=1.)

    def save_canonical(self, record, record_root=None):
        root = Path(record_root or Path(self.config['output'])/'records')
        root.mkdir(parents=True, exist_ok=True)
        path = root/(p8.key(record['task_id'], record['attempt'])+'.json')
        path.write_text(json.dumps(record))
        return path

    def timeout_record(self, **kwargs):
        record = self.record(valid=False, **kwargs)
        archive = p8._archive(record)
        terminal = dict(status='aborted', remove_sample=True, reward=None,
            response='', tokens=[], response_length=0, session_id=None,
            multimodal_inputs={'images': None, 'videos': None}, multimodal_train_inputs=None,
            metadata=dict(task_id=record['task_id'], is_last_turn=True, total_steps=0,
                terminate_reason='generation_error: rollout_task_timeout after 600.0s'))
        archive.write_text(json.dumps(dict(task_id=record['task_id'], turns=[terminal], error_type=None)))
        sidecar = json.loads(archive.with_suffix('.json').read_text())
        sidecar.update(terminal_status='aborted', reward_metadata={})
        archive.with_suffix('.json').write_text(json.dumps(sidecar))
        return record

    def health_failure_record(self, **kwargs):
        record = self.timeout_record(**kwargs)
        record.update(actor_requests=0, actor_output_tokens=0)
        archive = p8._archive(record)
        payload = json.loads(archive.read_text())
        payload['turns'][0]['metadata'].update(num_turns_in_trajectory=1,
            terminate_reason="generation_error: local_process env_server at http://127.0.0.1:18000 "
                "was not healthy within 30.0s; attempts=2; last_error=ClientConnectorError: "
                "ClientConnectorError(ConnectionKey(host='127.0.0.1', port=18000), "
                "ConnectionRefusedError(111, \"Connect call failed ('127.0.0.1', 18000)\"))")
        archive.write_text(json.dumps(payload))
        return record

    def write_config(self):
        path = self.root/'config.json'
        path.write_text(json.dumps(self.config))
        return path


class AccountingTest(Fixture):
    def test_exact_partition_and_distinct_new_seeds(self):
        ids = [str(i) for i in range(707)]
        shards = p8.partition(ids)
        self.assertEqual([len(s) for s in shards], [177,177,177,176])
        self.assertEqual({t for s in shards for t in s}, set(ids))
        p8.validate_partition(shards, ids)
        bad = deepcopy(shards)
        bad[1][0] = bad[0][0]
        with self.assertRaises(ValueError):
            p8.validate_partition(bad, ids)
        with self.assertRaises(ValueError):
            p8.partition(ids[:-1]+[ids[0]])
        with self.assertRaises(ValueError):
            p8.partition(ids, 8)
        for task in ids:
            seeds = [p8.seed_for(task, attempt) for attempt in range(8)]
            self.assertEqual(len(set(seeds)), 8)

    def test_resume_verifies_canonical_even_if_task_complete(self):
        for attempt in p8.NEW_ATTEMPTS:
            self.save_canonical(self.record(attempt=attempt, valid=attempt != 6))
        rows = [dict(metadata=dict(task_id=t)) for t in ['task0', 'task1']]
        self.assertEqual(p8.pending_rows(rows, Path(self.config['output'])/'records'), rows[1:])
        # Invalid primary records are preserved; only missing records get dispatched.
        path = self.save_canonical(dict(self.record(attempt=6), artifact=str(self.root/'missing')))
        with self.assertRaises(ValueError):
            p8.pending_rows(rows, path.parent)

    def test_partial_duplicate_wrong_index_and_missing_records(self):
        rows = [self.record(attempt=a) for a in (5, 6)]
        self.assertFalse(p8.summarize(rows, ['task0'])['complete'])
        for invalid in [rows, rows+[rows[0]], rows+[self.record(attempt=4)]]:
            with self.assertRaises(ValueError):
                p8.summarize(invalid, ['task0'], require_complete=True)
        with self.assertRaises(ValueError):
            p8.summarize(rows, ['task0', 'task0'])

    def test_protocol_seed_and_terminal_positive_guards(self):
        record = self.record(reward=1.)
        for changed in [dict(record, seed=record['seed']+1), dict(record, mode='arm'),
                        dict(record, selector_calls=1), dict(record, selector_fallback_turns=1),
                        dict(record, valid=1), dict(record, judge_id='o4-mini'),
                        dict(record, steps=0)]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                p8.verify_record(changed, 'task0', 5)
        sidecar_path = p8._archive(record).with_suffix('.json')
        sidecar = json.loads(sidecar_path.read_text())
        sidecar['terminal_status'] = 'failed'
        sidecar_path.write_text(json.dumps(sidecar))
        with self.assertRaisesRegex(ValueError, 'Success lacks'):
            p8.verify_record(record, 'task0', 5)

    def test_deep_archive_disagreement_and_missing_image_rejected(self):
        with patch.dict(sys.modules, {'torch': self.fake_torch}):
            record = self.record(reward=1.)
            p8.verify_record(record, 'task0', 5, deep=True)
            archive = p8._archive(record)
            original = json.loads(archive.read_text())
            variants = []
            wrong_reward = deepcopy(original)
            wrong_reward['turns'][0]['reward'] = 0.
            variants.append(wrong_reward)
            wrong_terminal = deepcopy(original)
            wrong_terminal['turns'][0]['metadata']['is_last_turn'] = False
            variants.append(wrong_terminal)
            wrong_valid = deepcopy(original)
            wrong_valid['turns'][0]['remove_sample'] = True
            variants.append(wrong_valid)
            no_images = deepcopy(original)
            no_images['turns'][0]['multimodal_inputs'] = None
            variants.append(no_images)
            for value in variants:
                archive.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    p8.verify_record(record, 'task0', 5, deep=True)

    def test_empty_invalid_archive_is_preserved(self):
        record = self.record(valid=False)
        record.update(steps=0, error_type='TimeoutError')
        archive = p8._archive(record)
        archive.write_text(json.dumps(dict(task_id='task0', turns=[], error_type='TimeoutError')))
        sidecar = json.loads(archive.with_suffix('.json').read_text())
        sidecar.update(turns=0, terminal_status=None, reward_metadata={}, error_type='TimeoutError')
        archive.with_suffix('.json').write_text(json.dumps(sidecar))
        with patch.dict(sys.modules, {'torch': self.fake_torch}):
            p8.verify_record(record, 'task0', 5, deep=True)

    def test_native_timeout_stub_preserved_without_fabricated_turn_or_reward(self):
        record = self.timeout_record()
        archive = p8._archive(record)
        before = archive.read_bytes()
        with patch.dict(sys.modules, {'torch': self.fake_torch}):
            self.assertIs(p8.verify_record(record, 'task0', 5, deep=True), record)
        self.assertEqual(before, archive.read_bytes())
        self.assertFalse(record['valid'])
        self.assertIsNone(record['reward'])
        self.assertNotIn('turn_index', json.loads(before)['turns'][0]['metadata'])

    def test_native_initial_local_health_stub_preserved_as_invalid(self):
        record = self.health_failure_record()
        archive = p8._archive(record)
        before = archive.read_bytes()
        with patch.dict(sys.modules, {'torch': self.fake_torch}):
            self.assertIs(p8.verify_record(record, 'task0', 5, deep=True), record)
        self.assertEqual(archive.read_bytes(), before)
        self.assertFalse(record['valid'])
        self.assertIsNone(record['reward'])
        self.assertFalse(p8._native_timeout_stub(record, json.loads(before)['turns']))
        self.assertTrue(p8._native_local_health_stub(record, json.loads(before)['turns']))

    def test_local_health_stub_requires_zero_actor_work_and_native_reason(self):
        record = self.health_failure_record()
        archive = p8._archive(record)
        original = json.loads(archive.read_text())
        reason = original['turns'][0]['metadata']['terminate_reason']
        with patch.dict(sys.modules, {'torch': self.fake_torch}):
            for key in ('actor_requests', 'actor_output_tokens'):
                for value in (1, False, None):
                    with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'turn index'):
                        p8.verify_record(dict(record, **{key: value}), 'task0', 5, deep=True)
            replacements = [('127.0.0.1:18000', 'example.com:18000'), ('http://', 'https://'),
                (':18000 ', ':0 '), (':18000 ', ':65536 '), ('30.0s;', '0.0s;'),
                ('30.0s;', 'nans;'), ('attempts=2;', 'attempts=0;'),
                ('generation_error: ', ''), ('was not healthy within', 'unrelated failure'),
                (reason.split('last_error=')[1], '')]
            for old, new in replacements:
                payload = deepcopy(original)
                payload['turns'][0]['metadata']['terminate_reason'] = reason.replace(old, new)
                archive.write_text(json.dumps(payload))
                with self.subTest(old=old, new=new), self.assertRaisesRegex(ValueError, 'turn index'):
                    p8.verify_record(record, 'task0', 5, deep=True)

    def test_local_health_stub_rejects_valid_generated_or_malformed_samples(self):
        record = self.health_failure_record()
        archive = p8._archive(record)
        original = json.loads(archive.read_text())
        changes = [('status', 'completed'), ('remove_sample', False), ('reward', 0.),
            ('response', 'generated'), ('tokens', [1]), ('response_length', 1),
            ('session_id', 'active-session'), ('multimodal_inputs', {'images': ['screenshot']}),
            ('metadata.total_steps', 1), ('metadata.turn_index', None),
            ('metadata.num_turns_in_trajectory', None), ('metadata.num_turns_in_trajectory', True),
            ('metadata.num_turns_in_trajectory', 2)]
        with patch.dict(sys.modules, {'torch': self.fake_torch}):
            for name, value in changes:
                payload = deepcopy(original);target = payload['turns'][0]
                parts = name.split('.')
                if len(parts) == 2:
                    target = target[parts[0]]
                target[parts[-1]] = value
                archive.write_text(json.dumps(payload))
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, 'turn index'):
                    p8.verify_record(record, 'task0', 5, deep=True)
            self.assertFalse(p8._native_local_health_stub(dict(record, valid=True, reward=0.), original['turns']))
            self.assertFalse(p8._native_local_health_stub(record, original['turns']*2))

    def test_unindexed_timeout_exception_rejects_non_native_shapes(self):
        record = self.timeout_record()
        archive = p8._archive(record)
        original = json.loads(archive.read_text())
        changes = [
            ('status', 'failed'), ('remove_sample', False), ('reward', 0.),
            ('response', 'generated'), ('tokens', [1]), ('response_length', 1),
            ('session_id', 'already-open'), ('multimodal_inputs', {'images': ['screenshot']}),
            ('metadata.turn_index', None), ('metadata.total_steps', 1),
            ('metadata.total_steps', False), ('metadata.task_id', 'wrong-task'),
            ('metadata.is_last_turn', False), ('metadata.num_turns_in_trajectory', 2),
            ('metadata.reward', {'judge': 0}),
            ('metadata.terminate_reason', 'generation_error: invented failure'),
            ('metadata.terminate_reason', 'generation_error: rollout_task_timeout after nans'),
        ]
        with patch.dict(sys.modules, {'torch': self.fake_torch}):
            for name, value in changes:
                payload = deepcopy(original)
                target = payload['turns'][0]
                parts = name.split('.')
                if len(parts) == 2:
                    target = target[parts[0]]
                target[parts[-1]] = value
                archive.write_text(json.dumps(payload))
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, 'turn index'):
                    p8.verify_record(record, 'task0', 5, deep=True)

    def test_valid_missing_and_duplicate_indices_still_rejected(self):
        with patch.dict(sys.modules, {'torch': self.fake_torch}):
            for valid in (True, False):
                record = self.record(valid=valid)
                archive = p8._archive(record)
                original = json.loads(archive.read_text())
                missing = deepcopy(original)
                del missing['turns'][0]['metadata']['turn_index']
                archive.write_text(json.dumps(missing))
                with self.assertRaisesRegex(ValueError, 'turn index'):
                    p8.verify_record(record, 'task0', 5, deep=True)
                original['turns'].append(deepcopy(original['turns'][0]))
                archive.write_text(json.dumps(original))
                record['steps'] = 2
                sidecar = json.loads(archive.with_suffix('.json').read_text())
                sidecar['turns'] = 2
                archive.with_suffix('.json').write_text(json.dumps(sidecar))
                with self.assertRaisesRegex(ValueError, 'turn index'):
                    p8.verify_record(record, 'task0', 5, deep=True)

    def test_timeout_stub_mixed_with_generated_turn_rejected(self):
        record = self.timeout_record()
        archive = p8._archive(record)
        payload = json.loads(archive.read_text())
        indexed = deepcopy(payload['turns'][0])
        indexed['metadata']['turn_index'] = 0
        payload['turns'].append(indexed)
        archive.write_text(json.dumps(payload))
        record['steps'] = 2
        sidecar = json.loads(archive.with_suffix('.json').read_text())
        sidecar['turns'] = 2
        archive.with_suffix('.json').write_text(json.dumps(sidecar))
        with patch.dict(sys.modules, {'torch': self.fake_torch}), self.assertRaisesRegex(ValueError, 'turn index'):
            p8.verify_record(record, 'task0', 5, deep=True)

    def test_dispatch_budget_charges_incomplete_reservations_across_restart(self):
        with patch.object(p8, 'BROWSER_CAP_PER_SHARD', 2):
            receipt1, first = p8.reserve_dispatch(self.config, 'task0', 5)
            receipt2, second = p8.reserve_dispatch(deepcopy(self.config), 'task0', 5)
            self.assertNotEqual(first['physical_attempt_id'], second['physical_attempt_id'])
            self.assertEqual(first['seed'], second['seed'])
            self.assertTrue((receipt1/'dispatch.json').exists())
            self.assertFalse((receipt1/'completed.json').exists())
            with self.assertRaisesRegex(RuntimeError, 'budget'):
                p8.reserve_dispatch(self.config, 'task0', 6)
            self.assertTrue((receipt2/'dispatch.json').exists())

    def test_persistent_judge_caps(self):
        root = self.root/'judge-budget'
        first = CappedJudge(root, client=object(), max_calls=4800, max_usd=12.5)
        first.reserve(12.4)
        resumed = CappedJudge(root, client=object(), max_calls=4800, max_usd=12.5)
        with self.assertRaises(ValueError):
            resumed.reserve(.2)
        self.assertEqual(p8.JUDGE_CALLS_PER_SHARD*4, 19200)
        self.assertEqual(p8.JUDGE_USD_PER_SHARD*4, 50.)
        self.assertEqual(p8.BROWSER_CAP_PER_SHARD*4, 6400)

    def test_configs_and_decoding_cannot_drift(self):
        p8.validate_protocol(self.args, self.config, sampling_params=p8.DECODING)
        for modified in [dict(self.config, seed=4), dict(self.config, max_steps=30),
                         dict(self.config, browser_cap=1601),
                         dict(self.config, old_record_roots=[self.config['output']])]:
            with self.assertRaises(ValueError):
                p8.validate_config(modified)
        for key, value in [('temperature', 1.), ('top_p', .95), ('top_k', 20), ('max_new_tokens', 4096)]:
            with self.assertRaises(ValueError):
                p8.validate_protocol(self.args, self.config, sampling_params=dict(p8.DECODING, **{key: value}))
        self.args.num_rollout = 1
        with self.assertRaises(ValueError):
            p8.validate_protocol(self.args, self.config)
        p8._bind_config(self.config)
        p8._bind_config(dict(self.config, job_id='replacement',
                           attempt_output=str(Path(self.config['output'])/'attempts'/'replacement')))
        self.assertEqual(len(list((Path(self.config['output'])/'execution-configs').glob('*.json'))), 2)
        with self.assertRaises(ValueError):
            p8._bind_config(dict(self.config, inference_timeout=181))

    def test_historical_manifest_and_each_selected_record_hash_are_enforced(self):
        # All 10K identities are bound, while this test shard reads one task.
        proof = [dict(task_id=f'task{t}', attempt=a) for t in range(2000) for a in range(5)]
        for entry in proof[:5]:
            record = self.record(entry['task_id'], entry['attempt'])
            path = self.save_canonical(record, self.config['old_record_roots'][0])
            archive = p8._archive(record)
            entry.update(record=str(path), record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                archive=str(archive), archive_size_bytes=archive.stat().st_size,
                sidecar_sha256=hashlib.sha256(archive.with_suffix('.json').read_bytes()).hexdigest())
        path = self.root/'historical-inputs.json'
        path.write_text(json.dumps(proof))
        self.config.update(historical_inputs=str(path), historical_inputs_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(len(p8.load_old_records(self.config, ['task0'])), 5)
        record_path = Path(proof[0]['record'])
        record_path.write_text(record_path.read_text()+'\n')  # Same JSON meaning, changed immutable evidence.
        with self.assertRaisesRegex(ValueError, 'canonical record changed'):
            p8.load_old_records(self.config, ['task0'])
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'manifest hash changed'):
            p8.load_old_records(self.config, ['task0'])

    def test_batch_gate_includes_primary_startup_and_exact_threshold(self):
        p8.batch_gate([dict(valid=i < 3) for i in range(6)], startup=True)
        with self.assertRaises(RuntimeError):
            p8.batch_gate([dict(valid=i < 2) for i in range(6)], startup=True)
        with self.assertRaises(RuntimeError):
            p8.batch_gate([dict(valid=True)]*5, startup=True)
        with self.assertRaises(RuntimeError):
            p8.batch_gate([dict(valid=False)]*2+[dict(valid=True)])

    def test_final_audit_requires_charged_physical_evidence_and_checks_judge_budget(self):
        record = self.record()
        self.save_canonical(record)
        old = [dict(task_id=t, attempt=a, valid=True, reward=float(i>=707 and a==0))
               for i,t in enumerate(self.config['historical_task_order']) for a in range(5)]
        with patch.object(p8, 'load_old_records', return_value=old), \
                patch.dict(sys.modules, {'torch': self.fake_torch}):
            with self.assertRaisesRegex(ValueError, 'charged physical'):
                p8.audit(self.config)
            receipt, dispatch = p8.reserve_dispatch(self.config, 'task0', 5)
            canonical = Path(self.config['output'])/'records'/(p8.key('task0', 5)+'.json')
            p8._publish(record, canonical, canonical, receipt)
            summary, joined = p8.audit(self.config)
            self.assertEqual(summary['physical_dispatches_charged'], 1)
            self.assertEqual(summary['judge_calls_charged'], 0)
            self.assertIsNone(joined['combined_pass8_rate'])
            budget = Path(self.config['output'])/'judge-budget'/'ledger.json'
            budget.parent.mkdir()
            budget.write_text(json.dumps(dict(calls=4801, charged_or_reserved_usd=1.)))
            with self.assertRaisesRegex(ValueError, 'judge budget'):
                p8.audit(self.config)


@dataclass
class Dataset:
    name: str = 'screen'
    path: str = ''
    n_samples_per_eval_prompt: int = 3
    temperature: float = .8
    top_p: float = 1.
    top_k: int = -1
    max_response_len: int = 1024


class JoinedMetricsTest(Fixture):
    def test_full_707_miss_extension_preserves_1293_successes_and_all_2000_denominator(self):
        ids = [f'task{i}' for i in range(2000)]
        old = []
        for i, task in enumerate(ids):
            for attempt in range(5):
                valid = not (attempt == 4 and (682 <= i < 707 or 707 <= i < 726))
                old.append(dict(task_id=task, attempt=attempt, valid=valid,
                    reward=float(i>=707 and attempt==0) if valid else None))
        targets = ids[:707]
        new = [dict(task_id=task, attempt=a, valid=True, reward=float(task in ('task0','task690') and a==6))
               for task in targets for a in p8.NEW_ATTEMPTS]
        summary = p8.summarize_joined(old, new, ids, require_complete=True,
                                    verify_artifacts=False, extension_ids=targets)
        self.assertEqual(summary['historical'], dict(tasks=2000, successes=1293, misses=707,
            five_valid_failure_tasks=682, any_invalid_tasks=44))
        self.assertEqual(summary['combined_pass8_rate'], 1295/2000)
        self.assertEqual(summary['historical_pass5_rate'], 1293/2000)
        self.assertEqual(summary['new_primary_attempts'], 2121)
        self.assertEqual(summary['skipped_historical_success_tasks'], 1293)
        self.assertEqual(summary['rescue_among_pass5_misses']['rate'], 2/707)
        self.assertEqual(summary['rescue_among_five_valid_failures']['rate'], 1/682)
        self.assertEqual(summary['rescue_among_invalid_pass5_misses']['rate'], 1/25)
        self.assertFalse(summary['fresh_pass3_on_all_tasks_measured'])
        self.assertNotIn('fresh_pass3_rate', summary)
        for invalid_targets in [targets[:-1], targets+[ids[707]], targets[1:]+targets[:1]]:
            with self.assertRaisesRegex(ValueError, 'every historical'):
                p8.summarize_joined(old, new, ids, verify_artifacts=False, extension_ids=invalid_targets)
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            p8.summarize_joined(old, new+[dict(task_id=ids[707],attempt=5,valid=True,reward=0.)],
                                ids, verify_artifacts=False, extension_ids=targets)
        partial = p8.summarize_joined(old, new[:-1], ids, verify_artifacts=False, extension_ids=targets)
        self.assertIsNone(partial['combined_pass8_rate'])
        self.assertFalse(partial['complete'])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            p8.summarize_joined(old, new[:-1], ids, require_complete=True,
                                verify_artifacts=False, extension_ids=targets)

    def test_ordered_union_rescue_denominators_and_invalids_are_separate(self):
        ids = ['old_win', 'valid_fail', 'invalid_fail', 'still_fail']
        old = [self.record(t, a, valid=not (t == 'invalid_fail' and a == 4),
                           reward=1. if t == 'old_win' and a == 0 else -1. if a == 3 else 0.)
               for t in ids for a in range(5)]
        new = [self.record(t, a, valid=not(t == 'still_fail' and a == 5),
                           reward=float(t in ('valid_fail', 'invalid_fail') and a == 6))
               for t in ids if t != 'old_win' for a in p8.NEW_ATTEMPTS]
        summary = p8.summarize_joined(old, new, ids, require_complete=True)
        self.assertEqual(summary['historical'], dict(tasks=4, successes=1, misses=3,
                         five_valid_failure_tasks=2, any_invalid_tasks=1))
        self.assertEqual(summary['historical_pass5_rate'], .25)
        self.assertEqual(summary['combined_pass8_rate'], .75)
        self.assertEqual(summary['pass8_minus_pass5'], .5)
        self.assertEqual(summary['additional_three_rate_among_historical_misses'], 2/3)
        self.assertFalse(summary['fresh_pass3_on_all_tasks_measured'])
        self.assertEqual(summary['rescue_among_pass5_misses']['rate'], 2/3)
        self.assertEqual(summary['rescue_among_five_valid_failures']['rate'], .5)
        self.assertEqual(summary['rescue_among_five_valid_failures']['task_ids'], ['valid_fail'])
        self.assertEqual(summary['rescue_among_invalid_pass5_misses']['rate'], 1.)
        self.assertIn('dates differ', summary['limitations'])

    def test_old_success_with_invalid_is_not_a_miss_or_valid_failure(self):
        old = [self.record(attempt=a, valid=a != 4, reward=float(a == 0)) for a in range(5)]
        new = []
        summary = p8.summarize_joined(old, new, ['task0'], require_complete=True)
        self.assertEqual(summary['historical']['any_invalid_tasks'], 1)
        self.assertEqual(summary['historical']['misses'], 0)
        self.assertEqual(summary['combined_pass8_rate'], 1.)
        self.assertIsNone(summary['rescue_among_pass5_misses']['rate'])

    def test_partial_results_do_not_report_final_rate_or_shrink_denominator(self):
        old = [self.record(attempt=a) for a in range(5)]
        new = [self.record(attempt=5, reward=1.)]
        summary = p8.summarize_joined(old, new, ['task0'])
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['combined_pass8_successes_observed'], 1)
        self.assertIsNone(summary['combined_pass8_rate'])
        self.assertIsNone(summary['additional_three_rate_among_historical_misses'])
        self.assertIsNone(summary['rescue_among_pass5_misses']['rate'])
        with self.assertRaises(ValueError):
            p8.summarize_joined(old, new, ['task0'], require_complete=True)
        with self.assertRaises(ValueError):
            p8.summarize_joined(old[:-1], new, ['task0'])
        with self.assertRaises(ValueError):
            p8.summarize_joined(old, new+new, ['task0'])
        with self.assertRaises(ValueError):
            p8.summarize_joined(old, new, ['task0'], expected_old=dict(tasks=2000))


class GeneratorTest(Fixture, unittest.IsolatedAsyncioTestCase):
    async def fake_trajectory(self, args, sample, sampling_params, mode, attempt, phase, config):
        self.calls.append((sample.metadata['task_id'], attempt, mode, phase, config['output']))
        self.assertEqual(sampling_params, p8.DECODING)
        self.assertEqual((mode, phase), ('actor', 'screen'))
        record = self.record(sample.metadata['task_id'], attempt, output=config['output'])
        self.save_canonical(record, Path(config['output'])/'screen_records')

    async def test_resume_only_dispatches_missing_indices_five_six_seven(self):
        self.calls = []
        config_path = self.write_config()
        with patch.dict(os.environ, OPENWEBRL_TASK_PASS8_CONFIG=str(config_path)), \
                patch.dict(sys.modules, {'torch': self.fake_torch}), \
                patch.object(p8, 'trajectory', side_effect=self.fake_trajectory):
            for index in (0, 1, 0, 1, 2):
                await p8.generate(self.args, SimpleNamespace(index=index, metadata={'task_id': 'task0'}),
                                  p8.DECODING, evaluation=True)
        self.assertEqual([call[1] for call in self.calls], [5, 6, 7])
        records = p8.load_records(Path(self.config['output'])/'records')
        self.assertTrue(p8.summarize(records, ['task0'], require_complete=True)['complete'])
        self.assertEqual(len(list((Path(self.config['output'])/'physical-attempts').iterdir())), 3)

    async def test_recovery_promotes_saved_record_without_another_browser(self):
        self.calls = []
        config_path = self.write_config()
        sample = SimpleNamespace(index=0, metadata={'task_id': 'task0'})
        with patch.dict(os.environ, OPENWEBRL_TASK_PASS8_CONFIG=str(config_path)), \
                patch.dict(sys.modules, {'torch': self.fake_torch}), \
                patch.object(p8, 'trajectory', side_effect=self.fake_trajectory):
            with patch.object(p8, '_publish', side_effect=RuntimeError('simulated lost publication')):
                with self.assertRaises(RuntimeError):
                    await p8.generate(self.args, sample, p8.DECODING, evaluation=True)
            await p8.generate(self.args, sample, p8.DECODING, evaluation=True)
        self.assertEqual(len(self.calls), 1)
        receipts = list((Path(self.config['output'])/'physical-attempts').iterdir())
        self.assertEqual(len(receipts), 1)
        self.assertTrue((receipts[0]/'interrupted.json').is_file())
        self.assertTrue(json.loads((receipts[0]/'completed.json').read_text())['recovered_after_interruption'])

    async def test_timeout_orphan_is_recovered_unchanged_without_browser_replay(self):
        config_path = self.write_config()
        receipt, dispatch = p8.reserve_dispatch(self.config, 'task0', 5)
        record = self.timeout_record(output=dispatch['output'])
        saved = self.save_canonical(record, Path(dispatch['output'])/'screen_records')
        before = saved.read_bytes()
        with patch.dict(os.environ, OPENWEBRL_TASK_PASS8_CONFIG=str(config_path)), \
                patch.dict(sys.modules, {'torch': self.fake_torch}), \
                patch.object(p8, 'trajectory') as browser:
            await p8.generate(self.args, SimpleNamespace(index=0, metadata={'task_id': 'task0'}),
                              p8.DECODING, evaluation=True)
        browser.assert_not_called()
        canonical = Path(self.config['output'])/'records'/saved.name
        self.assertEqual(canonical.read_bytes(), before)
        self.assertEqual(saved.read_bytes(), before)
        self.assertEqual(len(list((Path(self.config['output'])/'physical-attempts').iterdir())), 1)
        completed = json.loads((receipt/'completed.json').read_text())
        self.assertTrue(completed['recovered_after_interruption'])
        self.assertFalse(json.loads(canonical.read_text())['valid'])

    async def test_local_health_orphan_recovered_without_browser_or_actor_replay(self):
        config_path = self.write_config()
        receipt, dispatch = p8.reserve_dispatch(self.config, 'task0', 5)
        record = self.health_failure_record(output=dispatch['output'])
        saved = self.save_canonical(record, Path(dispatch['output'])/'screen_records')
        before = saved.read_bytes()
        with patch.dict(os.environ, OPENWEBRL_TASK_PASS8_CONFIG=str(config_path)), \
                patch.dict(sys.modules, {'torch': self.fake_torch}), \
                patch.object(p8, 'trajectory') as browser:
            await p8.generate(self.args, SimpleNamespace(index=0, metadata={'task_id': 'task0'}),
                              p8.DECODING, evaluation=True)
        browser.assert_not_called()
        canonical = Path(self.config['output'])/'records'/saved.name
        self.assertEqual(canonical.read_bytes(), before)
        self.assertEqual(saved.read_bytes(), before)
        self.assertEqual(len(list((Path(self.config['output'])/'physical-attempts').iterdir())), 1)
        self.assertTrue(json.loads((receipt/'completed.json').read_text())['recovered_after_interruption'])

    async def test_crash_after_canonical_publication_repairs_receipt_without_replay(self):
        self.calls = []
        config_path = self.write_config()
        sample = SimpleNamespace(index=0, metadata={'task_id': 'task0'})
        original_new = p8._new_json
        def crash_completion(path, value):
            if Path(path).name == 'completed.json':
                raise RuntimeError('lost completion receipt')
            return original_new(path, value)
        with patch.dict(os.environ, OPENWEBRL_TASK_PASS8_CONFIG=str(config_path)), \
                patch.dict(sys.modules, {'torch': self.fake_torch}), \
                patch.object(p8, 'trajectory', side_effect=self.fake_trajectory):
            with patch.object(p8, '_new_json', side_effect=crash_completion):
                with self.assertRaisesRegex(RuntimeError, 'lost completion'):
                    await p8.generate(self.args, sample, p8.DECODING, evaluation=True)
            self.assertEqual(len(p8.load_records(Path(self.config['output'])/'records')), 1)
            await p8.generate(self.args, sample, p8.DECODING, evaluation=True)
        self.assertEqual(len(self.calls), 1)
        receipt = next((Path(self.config['output'])/'physical-attempts').iterdir())
        self.assertTrue((receipt/'completed.json').is_file())

    async def test_interrupted_dispatch_retained_and_charged_before_retry(self):
        self.calls = []
        config_path = self.write_config()
        sample = SimpleNamespace(index=0, metadata={'task_id': 'task0'})
        with patch.dict(os.environ, OPENWEBRL_TASK_PASS8_CONFIG=str(config_path)), \
                patch.dict(sys.modules, {'torch': self.fake_torch}):
            with patch.object(p8, 'trajectory', side_effect=asyncio.CancelledError):
                with self.assertRaises(asyncio.CancelledError):
                    await p8.generate(self.args, sample, p8.DECODING, evaluation=True)
            with patch.object(p8, 'trajectory', side_effect=self.fake_trajectory):
                await p8.generate(self.args, sample, p8.DECODING, evaluation=True)
        receipts = list((Path(self.config['output'])/'physical-attempts').iterdir())
        self.assertEqual(len(receipts), 2)
        self.assertEqual(sum((r/'interrupted.json').exists() for r in receipts), 1)
        self.assertEqual(sum((r/'completed.json').exists() for r in receipts), 1)

    async def test_same_task_serialized_different_tasks_parallel_no_duplicate_browser(self):
        self.calls = []
        active, maximum, total_active = {}, {}, []
        async def slow_trajectory(*args):
            task = args[1].metadata['task_id']
            active[task] = active.get(task, 0)+1
            maximum[task] = max(maximum.get(task, 0), active[task])
            total_active.append(sum(active.values()))
            await asyncio.sleep(.02)
            await self.fake_trajectory(*args)
            active[task] -= 1
        config_path = self.write_config()
        with patch.dict(os.environ, OPENWEBRL_TASK_PASS8_CONFIG=str(config_path)), \
                patch.dict(sys.modules, {'torch': self.fake_torch}), \
                patch.object(p8, 'trajectory', side_effect=slow_trajectory):
            await asyncio.gather(*(p8.generate(self.args, SimpleNamespace(index=index,
                metadata={'task_id': task}), p8.DECODING, evaluation=True)
                for task in ('task0', 'task4') for index in (0, 0, 1, 2)))
        self.assertEqual(len(self.calls), 6)
        self.assertEqual(maximum, {'task0': 1, 'task4': 1})
        self.assertGreater(max(total_active), 1)

    async def test_halt_prevents_new_dispatch_and_budget_reset(self):
        config_path = self.write_config()
        root = Path(self.config['output'])/'judge-budget'
        root.mkdir(parents=True)
        (root/'halt.json').write_text('{}')
        with patch.dict(os.environ, OPENWEBRL_TASK_PASS8_CONFIG=str(config_path)), \
                patch.object(p8, 'trajectory') as run:
            with self.assertRaisesRegex(RuntimeError, 'halted'):
                await p8.generate(self.args, SimpleNamespace(index=0, metadata={'task_id': 'task0'}),
                                  p8.DECODING, evaluation=True)
            run.assert_not_called()
        self.assertFalse((Path(self.config['output'])/'physical-attempts').exists())

    async def test_failed_six_primary_startup_never_scales(self):
        all_ids = [f'task{i}' for i in range(2000)]
        extension = all_ids[:707]
        self.config['task_order'] = p8.partition(extension)[0]
        Path(self.config['partition_manifest']).write_text(json.dumps(
            dict(task_order=extension, historical_task_order=all_ids,
                 extension_task_order=extension, shards=p8.partition(extension))))
        Path(self.config['tasks']).write_text(''.join(json.dumps(dict(metadata={'task_id': t}))+'\n'
                                                    for t in self.config['task_order']))
        self.args.eval_datasets = [Dataset(path=self.config['tasks'])]
        old = [dict(task_id=t, attempt=a, valid=True, reward=float(i>=707 and a==0))
               for i,t in enumerate(all_ids) for a in range(5)]
        calls = []
        fake_rollout = ModuleType('slime.rollout.sglang_rollout')
        async def evaluate(args, rollout_id, dataset):
            rows = [json.loads(line) for line in Path(dataset.path).read_text().splitlines()]
            calls.append(rows)
            for row in rows:
                for attempt in p8.NEW_ATTEMPTS:
                    self.save_canonical(self.record(row['metadata']['task_id'], attempt, valid=False))
            return {}
        fake_rollout.eval_rollout_single_dataset = evaluate
        fake_output = ModuleType('slime.rollout.base_types')
        fake_output.RolloutFnEvalOutput = SimpleNamespace
        fake_reward = ModuleType('openwebrl.reward_browser')
        original_client = lambda **kwargs: None
        fake_reward._get_openai_client = original_client
        class Judge:
            closed = False
            async def close(self):
                self.closed = True
        judge = Judge()
        with patch.dict(sys.modules, {'slime.rollout.sglang_rollout': fake_rollout,
                'slime.rollout.base_types': fake_output, 'openwebrl.reward_browser': fake_reward}), \
                patch('openwebrl.arm_terminal_budget.CappedJudge', return_value=judge), \
                patch.object(p8, 'load_old_records', return_value=old):
            with self.assertRaisesRegex(RuntimeError, '50% invalid'):
                await p8.collect(self.args, 0, self.config)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 2)
        self.assertEqual(len(p8.load_records(Path(self.config['output'])/'records')), 6)
        self.assertTrue((Path(self.config['output'])/'halt.json').is_file())
        self.assertFalse((Path(self.config['output'])/'collection-complete.json').exists())
        self.assertTrue(judge.closed)
        self.assertIs(fake_reward._get_openai_client, original_client)


if __name__ == '__main__':
    unittest.main()
