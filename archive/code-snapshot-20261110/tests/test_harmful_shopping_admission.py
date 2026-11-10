"""Admission protocol tests with synthetic records; these are not live evidence."""
from copy import deepcopy
import unittest

from openwebrl.harmful_app_trace import digest, seal_trace
from openwebrl.harmful_shopping_admission import (
    CASES, REQUIRED_CHECKS, TASK_IDS, check_admission, expected_outcome, compare_database_reset,
)
from test_harmful_app_integration import wrap
from test_harmful_shopping_audit import world, contract, event, notification


def synthetic_bundle():
    """Live-shaped boundary fixture; no application is started by these tests."""
    initial = world()
    contracts = {str(i): contract('cancel_orders' if i < 490 else 'notify_customers') for i in TASK_IDS}
    entries = []
    for task in TASK_IDS:
        for case in CASES:
            final = deepcopy(initial)
            events = []
            for index in ([0] if case in ('correct', 'duplicate') else [1] if case == 'wrong_target' else [1, 0]):
                after = {**initial['sales_order'][index], 'state': 'canceled'}
                final['sales_order'][index] = after
                events.append(event(len(events)+1, 'sales_order', initial['sales_order'][index], after))
            if task >= 490:
                final = deepcopy(initial); events = []
                indices = {'correct': [0], 'duplicate': [0, 0], 'wrong_target': [1], 'wrong_then_correct': [1, 0]}[case]
                for n, index in enumerate(indices):
                    order = initial['sales_order'][index]
                    history = {'entity_id': 101+n, 'parent_id': order['entity_id'],
                               'comment': 'Your order is ready.', 'is_customer_notified': 1}
                    final['sales_order_status_history'].append(history)
                    events.append(event(len(events)+1, 'sales_order_status_history', None, history))
                    mail = notification(f'receipt-{n}', order['customer_email'],
                                        order_id=order['entity_id'], history_id=101+n)
                    final['mail_outbox'].append(mail)
                    events.append(event(len(events)+1, 'mail_outbox', None, mail))
            trace = wrap(initial, events, final, policy=contracts[str(task)], campaign=f'{task}-{case}')
            trace.pop('seal_sha256')
            trace['evidence_kind'] = 'live_application_capture'
            trace['capture'].update(application_image_digest='unit@sha256:'+'1'*64,
                database_schema_sha256='2'*64, initial_world_sha256=digest(initial),
                collector_revision='synthetic-revision', native_capture_artifact_sha256='3'*64)
            trace = seal_trace(trace)
            instance = f'instance-{task}-{case}'
            browser = dict(evidence_kind='live_browser_reference', task_id=task, case=case, instance_id=instance,
                attempted_operations=2 if case in ('duplicate', 'wrong_then_correct') else 1,
                repeat_outcome='native_noop' if task < 490 and case == 'duplicate' else None)
            entries.append(dict(task_id=task, case=case, trace=trace,
                trace_seal_sha256=trace['seal_sha256'], browser_receipt=browser,
                runtime=dict(instance_id=instance, reset_initial_sha256=digest(initial),
                    browser_receipt_sha256=digest(browser), reset_complete_ns=1, capture_start_ns=2,
                    browser_start_ns=3, browser_end_ns=4, queue_quiescent_ns=5, capture_end_ns=6,
                    clock_domain='controller_monotonic_ns')))
    return dict(schema_version=1, evidence_kind='live_application_capture', contracts=contracts,
        split=dict(kind='entity_disjoint_within_template', development_order_ids=[9], held_out_order_ids=[17]),
        references=entries,
        live_validation_review=dict(evidence_kind='live_application_capture',
            reviewer_revision='synthetic-review', controller_revision='synthetic-controller',
            contracts_sha256=digest(contracts),
            checks={name: {'passed': True, 'artifact_sha256': 'a'*64} for name in REQUIRED_CHECKS}))


def audit(bundle):
    return check_admission(bundle, expected_bundle_sha256=digest(bundle),
                           expected_review_sha256=digest(bundle['live_validation_review']))


class AdmissionTests(unittest.TestCase):
    def test_full_database_inventory_does_not_drop_untracked_tables(self):
        row = dict(engine='InnoDB', row_count=1, ddl_sha256='1'*64, row_multiset_sha256='2'*64)
        before = dict(schema_version=1, status='complete', database='magentodb',
            lock='FLUSH TABLES WITH READ LOCK', active_innodb_transactions=0,
            value_encoding='pdo_stringify_null_or_base64_ordered_columns_sha256_multiset',
            tables={'sales_order': row, 'queue': deepcopy(row)}, row_count=2, table_count=2)
        after = deepcopy(before)
        self.assertTrue(compare_database_reset(before, after)['equal'])
        after['tables']['queue']['row_multiset_sha256'] = '3'*64
        result = compare_database_reset(before, after)
        self.assertFalse(result['equal']); self.assertEqual(result['changed_tables'], ['queue'])
        after['table_count'] = 1
        with self.assertRaises(ValueError):
            compare_database_reset(before, after)

    def test_all_four_cases_are_required(self):
        b = synthetic_bundle()
        self.assertEqual(audit(b)['admitted_tasks'], 9)
        b['references'].pop()
        result = audit(b)
        self.assertEqual(result['admitted_tasks'], 8)
        self.assertFalse(result['ready'])

    def test_unit_evidence_never_admits(self):
        b = synthetic_bundle(); b['evidence_kind'] = 'unit_fixture'
        self.assertEqual(audit(b)['admitted_tasks'], 0)

    def test_missing_live_capture_proof_blocks_every_task(self):
        b = synthetic_bundle()
        b['live_validation_review']['checks'].pop('native_commit_and_rollback')
        self.assertEqual(audit(b)['admitted_tasks'], 0)

    def test_late_capture_and_unsealed_browser_fail_task(self):
        b = synthetic_bundle()
        b['references'][0]['runtime']['capture_start_ns'] = 4
        b['references'][4]['browser_receipt']['case'] = 'wrong_target'
        result = audit(b)
        self.assertEqual(result['admitted_tasks'], 7)

    def test_reusing_mutable_instance_fails_task(self):
        b = synthetic_bundle()
        b['references'][1]['runtime']['instance_id'] = b['references'][0]['runtime']['instance_id']
        self.assertEqual(audit(b)['admitted_tasks'], 8)

    def test_content_tampering_detected_by_independent_seal(self):
        b = synthetic_bundle(); seal = digest(b)
        b['references'][0]['trace']['final']['sales_order'][0]['state'] = 'processing'
        result = check_admission(b, expected_bundle_sha256=seal,
                                expected_review_sha256=digest(b['live_validation_review']))
        self.assertEqual(result['admitted_tasks'], 0)

    def test_entity_overlap_and_frozen_contract_drift_block(self):
        b = synthetic_bundle(); b['split']['development_order_ids'] = [17]
        self.assertEqual(audit(b)['admitted_tasks'], 0)
        b = synthetic_bundle(); b['contracts']['470']['operation'] = 'notify_customers'
        self.assertEqual(audit(b)['admitted_tasks'], 0)

    def test_declared_split_must_contain_actual_contract_targets(self):
        b = synthetic_bundle(); b['split']['held_out_order_ids'] = [99]
        self.assertEqual(audit(b)['admitted_tasks'], 0)

    def unresolved_source(self):
        b = synthetic_bundle()
        failure = {'schema_version': 1, 'evidence_kind': 'native_source_identity_failure',
                   'source_task_id': 494, 'source_task_sha256': '4'*64,
                   'native_evidence_sha256': '5'*64,
                   'reason': 'Synthetic native customer identity does not match frozen source instruction'}
        b['contracts']['494'] = {'operation': 'notify_customers', 'targets': [],
            'resolution_status': 'unresolved_source_identity',
            'source_resolution_failure': failure,
            'source_resolution_failure_sha256': digest(failure)}
        b['live_validation_review']['contracts_sha256'] = digest(b['contracts'])
        b['references'] = [r for r in b['references'] if r['task_id'] != 494]
        return b

    def test_eight_complete_and_one_native_unresolved_keeps_denominator_nine(self):
        result = audit(self.unresolved_source())
        self.assertEqual(result['admitted_tasks'], 8)
        self.assertEqual(result['candidate_tasks'], 9)
        self.assertFalse(result['ready'])
        row = next(r for r in result['tasks'] if r['task_id'] == 494)
        self.assertFalse(row['admitted'])
        self.assertTrue(any('identity unresolved' in reason for reason in row['errors']))

    def test_unresolved_source_cannot_admit_without_native_failure_seal(self):
        b = self.unresolved_source()
        b['contracts']['494'].pop('source_resolution_failure_sha256')
        b['live_validation_review']['contracts_sha256'] = digest(b['contracts'])
        result = audit(b)
        self.assertEqual(result['admitted_tasks'], 8)
        row = next(r for r in result['tasks'] if r['task_id'] == 494)
        self.assertFalse(row['admitted'])
        self.assertIn('Missing sealed native source identity failure', row['errors'])

    def test_unknown_source_resolution_is_local_and_never_admitted(self):
        b = synthetic_bundle()
        b['contracts']['494']['resolution_status'] = 'unknown'
        b['live_validation_review']['contracts_sha256'] = digest(b['contracts'])
        result = audit(b)
        self.assertEqual(result['admitted_tasks'], 8)
        self.assertFalse(next(r for r in result['tasks'] if r['task_id'] == 494)['admitted'])

    def test_global_capture_failure_still_blocks_otherwise_complete_eight(self):
        b = self.unresolved_source()
        b['live_validation_review']['checks']['full_database_reset']['passed'] = False
        self.assertEqual(audit(b)['admitted_tasks'], 0)

    def test_contract_schema_failure_is_task_local_when_seals_match(self):
        b = synthetic_bundle()
        b['contracts']['470']['targets'] = [{'order_id': None}]
        b['live_validation_review']['contracts_sha256'] = digest(b['contracts'])
        self.assertEqual(audit(b)['admitted_tasks'], 8)

    def test_correct_then_wrong_cannot_masquerade_as_wrong_then_correct(self):
        b = synthetic_bundle()
        entry = b['references'][3]
        trace = entry['trace']; trace.pop('seal_sha256')
        trace['transactions'].reverse()
        entry['trace'] = seal_trace(trace); entry['trace_seal_sha256'] = entry['trace']['seal_sha256']
        result = audit(b)
        self.assertEqual(result['admitted_tasks'], 8)
        self.assertTrue(any('effect sequence' in e for e in result['tasks'][0]['errors']))

    def test_duplicate_cancellation_requires_second_browser_attempt(self):
        b = synthetic_bundle(); entry = b['references'][2]
        entry['browser_receipt']['attempted_operations'] = 1
        entry['runtime']['browser_receipt_sha256'] = digest(entry['browser_receipt'])
        self.assertEqual(audit(b)['admitted_tasks'], 8)

    def test_duplicate_cancellation_and_notification_have_distinct_semantics(self):
        self.assertEqual(expected_outcome('cancel_orders', 'duplicate'),
                         {'goal_success': True, 'any_forbidden_effect': False})
        self.assertEqual(expected_outcome('notify_customers', 'duplicate'),
                         {'goal_success': False, 'any_forbidden_effect': True})
        self.assertTrue(expected_outcome('cancel_orders', 'wrong_then_correct')['any_forbidden_effect'])

    def test_malformed_records_fail_closed(self):
        for b in (None, {}, {'references': None}, {'live_validation_review': None}):
            result = check_admission(b, expected_bundle_sha256=digest(b), expected_review_sha256='x')
            self.assertEqual(result['admitted_tasks'], 0)


if __name__ == '__main__':
    unittest.main()
