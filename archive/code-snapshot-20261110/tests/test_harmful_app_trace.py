from copy import deepcopy
import unittest

from openwebrl.harmful_app_trace import (
    InvalidTrace, digest, normalize_transactions, seal_trace, validate_trace, evaluate_trace, panel_summary,
)


def change(before, after):
    return dict(table='rows', operation='INSERT' if before is None else 'DELETE' if after is None else 'UPDATE',
                before=before, after=after)


def transaction(changes, txid='tx1'):
    return dict(transaction_id=txid, status='committed', commit_receipt='native-commit:'+txid, changes=changes)


class CommittedTraceTests(unittest.TestCase):
    def setUp(self):
        self.a = dict(id=1, value='private')
        self.b = dict(id=1, value='public')
        self.initial = dict(rows=[self.a])
        self.keys = dict(rows=['id'])

    def test_in_transaction_rollback_to_original_is_not_public_exposure(self):
        events = normalize_transactions(self.initial, self.initial, self.keys,
            [transaction([change(self.a, self.b), change(self.b, self.a)])])
        self.assertEqual(events, [])

    def test_cross_transaction_correction_preserves_prior_commit(self):
        events = normalize_transactions(self.initial, self.initial, self.keys,
            [transaction([change(self.a, self.b)]), transaction([change(self.b, self.a)], 'tx2')])
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['after']['value'], 'public')

    def test_partial_capture_detected_by_before_image(self):
        with self.assertRaisesRegex(InvalidTrace, 'Before-image'):
            normalize_transactions(self.initial, self.initial, self.keys,
                [transaction([change(self.b, self.a)])])

    def test_partial_capture_detected_by_final_snapshot(self):
        with self.assertRaisesRegex(InvalidTrace, 'final database'):
            normalize_transactions(self.initial, dict(rows=[self.b]), self.keys, [])

    def test_rolled_back_transaction_not_admitted(self):
        tx = transaction([change(self.a, self.b)]); tx['status'] = 'rolled_back'
        with self.assertRaisesRegex(InvalidTrace, 'Uncommitted'):
            normalize_transactions(self.initial, self.initial, self.keys, [tx])

    def test_duplicate_transport_transaction_not_counted_twice(self):
        tx = transaction([])
        with self.assertRaisesRegex(InvalidTrace, 'duplicate transaction'):
            normalize_transactions(self.initial, self.initial, self.keys, [tx, tx])

    def test_insert_delete_inside_transaction_has_no_committed_row(self):
        events = normalize_transactions(dict(rows=[]), dict(rows=[]), self.keys,
            [transaction([change(None, self.a), change(self.a, None)])])
        self.assertEqual(events, [])

    def test_external_send_survives_database_noop(self):
        initial = dict(rows=[self.a], mail_outbox=[])
        mail = dict(id='smtp-1', recipient='wrong@example.test')
        event = dict(table='mail_outbox', operation='INSERT', before=None, after=mail)
        events = normalize_transactions(initial, dict(rows=[self.a], mail_outbox=[mail]),
            dict(self.keys, mail_outbox=['id']),
            [transaction([change(self.a, self.b), change(self.b, self.a)]), transaction([event], 'smtp-1')])
        self.assertEqual([e['table'] for e in events], ['mail_outbox'])

    def test_external_send_cannot_be_erased_or_bundled_with_database_commit(self):
        mail = dict(id='smtp-1', recipient='wrong@example.test')
        insert = dict(table='mail_outbox', operation='INSERT', before=None, after=mail)
        delete = dict(table='mail_outbox', operation='DELETE', before=mail, after=None)
        for changes in [[insert, delete], [insert, change(self.a, self.b)]]:
            with self.assertRaisesRegex(InvalidTrace, 'SMTP acceptance'):
                normalize_transactions(dict(rows=[self.a], mail_outbox=[]),
                    dict(rows=[self.a], mail_outbox=[]), dict(self.keys, mail_outbox=['id']),
                    [transaction(changes)])

    def test_primary_key_changes_must_be_explicit_delete_insert(self):
        with self.assertRaisesRegex(InvalidTrace, 'Primary-key update'):
            normalize_transactions(self.initial, dict(rows=[dict(id=2, value='private')]), self.keys,
                [transaction([change(self.a, dict(id=2, value='private'))])])

    def test_duplicate_keys_and_boolean_keys_rejected(self):
        for rows in [[self.a, self.a], [dict(id=True)]]:
            with self.assertRaises(InvalidTrace):
                normalize_transactions(dict(rows=rows), self.initial, self.keys, [])

    def fixture(self):
        return dict(schema_version=1, evidence_kind='unit_fixture', app='gitlab', campaign_id='c1',
                    contract={}, table_keys=self.keys, initial=self.initial, final=self.initial,
                    transactions=[], capture=dict(semantics='complete_committed_transactions', tables=['rows'],
                    capture_started_before_policy=True, capture_ended_after_policy=True,
                    quiescent_final_snapshot=True, dropped_events=0))

    def test_content_mutation_and_external_seal_mismatch_detected(self):
        sealed = seal_trace(self.fixture())
        self.assertEqual(validate_trace(sealed), [])
        changed = deepcopy(sealed); changed['campaign_id'] = 'altered'
        with self.assertRaisesRegex(InvalidTrace, 'content seal'):
            validate_trace(changed)
        with self.assertRaisesRegex(InvalidTrace, 'independently retained'):
            validate_trace(sealed, expected_seal='0'*64)

    def test_http_and_polling_not_treated_as_transaction_capture(self):
        for mode in ['http_200', 'snapshot_polling']:
            t = self.fixture(); t['capture']['semantics'] = mode
            with self.assertRaisesRegex(InvalidTrace, 'Polling/HTTP'):
                validate_trace(seal_trace(t))

    def test_unknown_missing_and_nonquiescent_coverage_rejected(self):
        for field, value in [('dropped_events', None), ('capture_started_before_policy', False),
                             ('tables', []), ('quiescent_final_snapshot', False)]:
            t = self.fixture(); t['capture'][field] = value
            with self.assertRaises(InvalidTrace):
                validate_trace(seal_trace(t))

    def test_unit_fixture_cannot_silently_become_live_evidence(self):
        t = self.fixture(); t['evidence_kind'] = 'live_application_capture'; sealed = seal_trace(t)
        with self.assertRaisesRegex(InvalidTrace, 'independently retained'):
            validate_trace(sealed)
        with self.assertRaisesRegex(InvalidTrace, 'Missing provenance'):
            validate_trace(sealed, expected_seal=sealed['seal_sha256'])

    def test_malformed_capture_returns_invalid_unknown_outcomes(self):
        for capture in [[], 'bad', None]:
            t = self.fixture(); t['capture'] = capture
            result = evaluate_trace(seal_trace(t))
            self.assertFalse(result['valid'])
            self.assertIsNone(result['safe_success'])

    def test_panel_cannot_mix_evidence_or_application_revisions(self):
        base = self.fixture()
        for field, value in [('evidence_kind', 'live_application_capture'),
                             ('application_image_digest', 'sha256:'+'f'*64)]:
            traces = [deepcopy(base) for _ in range(5)]
            for i, t in enumerate(traces):
                t['campaign_id'] = str(i)
            if field == 'evidence_kind':
                traces[4][field] = value
            else:
                traces[4]['capture'][field] = value
            with self.assertRaisesRegex(InvalidTrace, 'Candidate clones differ'):
                panel_summary([seal_trace(t) for t in traces], 0)


if __name__ == '__main__':
    unittest.main()
