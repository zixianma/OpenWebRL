import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

from scripts.continue_webprm_verified import (
    ARMS, gate_proof, identity_hash, remaining_cells, saved_results, sha256,
    validate_deadlines, verify_hashes,
)


class VerifiedContinuationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.ids = [f'task-{i}' for i in range(12)]
        self.identity = identity_hash({'protocol': 'test'})
        self.rows = []
        for index, task in enumerate(self.ids[:2]):
            for arm in ARMS:
                row = dict(task_id=task, arm=arm, attempted=True, valid=True,
                           identity_sha256=self.identity, success=0, turns=30)
                if index == 1:
                    row.update(valid=False, error_type='RuntimeError',
                               error='Initial website HTTP status 403', turns=0)
                elif arm == 'arbiter_unguided':
                    row.update(valid=False, error_type='ValueError',
                               error='Incomplete Arbiter policy response', turns=2)
                self.rows.append(row)
                folder = self.root / task / arm
                folder.mkdir(parents=True)
                (folder / 'result.json').write_text(json.dumps(row) + '\n')
        self.journal = self.root / self.ids[0] / 'arbiter_unguided' / 'requests.jsonl'
        self.events = [
            dict(event='request', purpose='policy', local=False, reservation=5,
                 payload=dict(model='gpt-4o-mini-2024-07-18', max_tokens=2048,
                              temperature=1., top_p=.9, n=1)),
            dict(event='response', reservation=5,
                 response=dict(choices=[dict(finish_reason='length')],
                               usage=dict(completion_tokens=2048))),
        ]
        self.write_journal()

    def write_journal(self):
        self.journal.write_text(''.join(json.dumps(event) + '\n' for event in self.events))

    def proof(self, rows=None):
        return gate_proof(self.root, self.rows if rows is None else rows, self.ids, self.identity)

    def test_exact_observed_gate_can_be_waived_but_is_never_marked_passed(self):
        proof = self.proof()
        self.assertEqual(proof['original_gate_status'], 'failed')
        self.assertEqual(proof['original_gate_failure'], 'quality_gate_no_valid_episode:arbiter_unguided')
        self.assertEqual(proof['valid_by_arm']['arbiter_unguided'], 0)
        self.assertEqual(proof['attempted'], 12)
        self.assertEqual(proof['truncation_journal_sha256'], sha256(self.journal))

    def test_missing_or_unattempted_initial_cell_blocks_continuation(self):
        with self.assertRaisesRegex(ValueError, 'twelve'):
            self.proof(self.rows[:-1])
        rows = copy.deepcopy(self.rows)
        rows[0]['attempted'] = False
        with self.assertRaisesRegex(ValueError, 'attempted'):
            self.proof(rows)

    def test_another_invalid_guided_arm_cannot_be_waived(self):
        rows = copy.deepcopy(self.rows)
        next(row for row in rows if row['arm'] == 'shepherd8b')['valid'] = False
        with self.assertRaisesRegex(ValueError, 'Only Arbiter unguided'):
            self.proof(rows)

    def test_website_error_must_be_pre_actor_403_across_every_arm(self):
        for changed in ({'error': 'HTTP Error 403: Forbidden'}, {'turns': 1}, {'error_type': 'HTTPError'}):
            with self.subTest(changed=changed):
                rows = copy.deepcopy(self.rows)
                rows[-1].update(changed)
                with self.assertRaisesRegex(ValueError, 'website 403'):
                    self.proof(rows)

    def test_error_string_without_exact_paid_response_evidence_is_rejected(self):
        original = copy.deepcopy(self.events)
        changes = [
            lambda events: events[-1]['response']['choices'][0].update(finish_reason='stop'),
            lambda events: events[-1]['response']['usage'].update(completion_tokens=2047),
            lambda events: events[0]['payload'].update(max_tokens=4096),
            lambda events: events[0].update(local=True),
            lambda events: events[-1].update(reservation=6),
            lambda events: events[0].update(purpose='terminal'),
        ]
        for change in changes:
            self.events = copy.deepcopy(original)
            change(self.events)
            self.write_journal()
            with self.assertRaisesRegex(ValueError, 'prove truncation'):
                self.proof()

    def test_saved_results_are_read_only_and_even_whitespace_changes_are_detected(self):
        before = {str(path.relative_to(self.root)): path.read_bytes()
                  for path in self.root.glob('*/*/result.json')}
        rows, hashes = saved_results(self.root, self.ids, self.identity)
        self.proof(rows)
        verify_hashes(self.root, hashes)
        self.assertEqual(len(rows), 12)
        self.assertEqual(before, {name: (self.root / name).read_bytes() for name in before})
        path = self.root / next(iter(hashes))
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'Preserved file changed'):
            verify_hashes(self.root, hashes)

    def test_unattempted_saved_cells_are_not_silently_retried(self):
        path = next(self.root.glob('*/*/result.json'))
        row = json.loads(path.read_text())
        row['attempted'] = False
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'cannot be retried'):
            saved_results(self.root, self.ids, self.identity)

    def test_saved_identity_or_directory_mismatch_blocks_continuation(self):
        path = next(self.root.glob('*/*/result.json'))
        row = json.loads(path.read_text())
        row['identity_sha256'] = 'different'
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            saved_results(self.root, self.ids, self.identity)
        row['identity_sha256'] = self.identity
        row['task_id'] = self.ids[-1]
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'misfiled'):
            saved_results(self.root, self.ids, self.identity)

    def test_fixed_shuffle_consumes_gate_tasks_and_skips_later_durable_cells(self):
        tasks = [dict(task_id=task) for task in self.ids]
        seed = 20261009
        rng = random.Random(seed)
        expected = []
        for index, task in enumerate(tasks):
            order = list(ARMS)
            rng.shuffle(order)
            if index >= 2:
                expected.extend((task['task_id'], arm) for arm in order)
        extra = dict(task_id=self.ids[5], arm=ARMS[2], attempted=True)
        cells = remaining_cells(tasks, self.rows + [extra], seed)
        actual = [(task['task_id'], arm) for task, arm in cells]
        self.assertEqual(actual, [key for key in expected if key != (extra['task_id'], extra['arm'])])
        self.assertEqual(len(actual), 59)
        self.assertEqual(len(set(actual)), 59)

    def test_deadline_bounds_reject_expired_nonfinite_or_extra_time(self):
        validate_deadlines(150, 160, now=100)
        for deadline, allocation in [(99, 160), (161, 160), (150, 6101),
                                     (float('nan'), 160), (150, float('inf'))]:
            with self.subTest(deadline=deadline, allocation=allocation):
                with self.assertRaises(ValueError):
                    validate_deadlines(deadline, allocation, now=100)


if __name__ == '__main__':
    unittest.main()
