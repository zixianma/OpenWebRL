"""Bounded loopback SMTP tests; no Magento, allocation, or external delivery."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import smtplib
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from openwebrl.harmful_smtp_sink import (
    CaptureError, SinkLimits, SmtpSink, digest, project_acceptances, validate_capture,
)


MESSAGE = (b'From: shop@example.test\r\nTo: customer@example.test\r\n'
           b'Subject: Order update\r\nMessage-ID: <reused@example.test>\r\n'
           b'Content-Type: text/plain; charset=utf-8\r\n\r\n'
           b'Hello\r\n.leading dot\r\n..two dots\r\n')


class SinkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='harm-smtp-test-', dir='/tmp')
        self.addCleanup(self.directory.cleanup)
        self.sinks = []
        self.addCleanup(self.close_all)

    def close_all(self):
        for sink, worker, errors in self.sinks:
            sink.request_stop()
            worker.join(3)
            self.assertFalse(worker.is_alive(), 'Owned SMTP worker did not stop')

    def start(self, **overrides):
        caps = dict(max_messages=10, max_message_bytes=8192, max_storage_bytes=131072,
                    max_runtime_seconds=10, heartbeat_seconds=0.05, idle_seconds=0.5)
        caps.update(overrides)
        directory = Path(self.directory.name) / ('capture-' + str(len(self.sinks)))
        sink = SmtpSink(directory, campaign_id='unit-campaign', sink_id='unit-sink',
                        limits=SinkLimits(**caps))
        errors = []

        def run():
            try:
                sink.serve_forever()
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=run, daemon=True)
        self.sinks.append((sink, worker, errors))
        worker.start()
        return sink, directory, worker, errors

    def send(self, sink, message=MESSAGE, sender='shop@example.test', recipients=None):
        with smtplib.SMTP(*sink.address, timeout=2) as client:
            client.sendmail(sender, recipients or ['customer@example.test'], message)

    def finish(self, sink, directory):
        sink.request_stop()
        next(worker for item, worker, _ in self.sinks if item is sink).join(3)
        self.assertIsNotNone(sink.final_record)
        return validate_capture(directory, expected_final_sha256=sink.final_record['sha256'])

    def test_full_data_and_envelope_are_durable_and_duplicate_ids_are_not_deduplicated(self):
        sink, directory, _, errors = self.start()
        self.send(sink)
        self.send(sink)
        capture = self.finish(sink, directory)
        self.assertEqual(errors, [])
        accepted = capture['acceptances']
        self.assertEqual(len(accepted), 2)
        self.assertNotEqual(accepted[0]['receipt_id'], accepted[1]['receipt_id'])
        for receipt in accepted:
            self.assertEqual(base64.b64decode(receipt['raw_message_b64']), MESSAGE)
            self.assertEqual(receipt['raw_message_sha256'], hashlib.sha256(MESSAGE).hexdigest())
            self.assertEqual(receipt['envelope_recipients'], ['customer@example.test'])
            self.assertEqual(receipt['envelope_sender'], 'shop@example.test')
        self.assertFalse(capture['queue_drain_verified'])
        self.assertFalse(capture['routing_verified'])
        self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
        self.assertEqual((directory / 'journal.jsonl').stat().st_mode & 0o777, 0o600)

    def test_empty_sender_and_repeated_envelope_recipient_are_preserved(self):
        sink, directory, _, _ = self.start()
        self.send(sink, sender='', recipients=['customer@example.test', 'customer@example.test'])
        capture = self.finish(sink, directory)
        receipt = capture['acceptances'][0]
        self.assertEqual(receipt['envelope_sender'], '')
        self.assertEqual(receipt['envelope_recipients'], ['customer@example.test'] * 2)

    def test_rset_discards_unaccepted_envelope_and_does_not_emit_receipt(self):
        sink, directory, _, _ = self.start()
        with smtplib.SMTP(*sink.address, timeout=2) as client:
            client.ehlo()
            self.assertEqual(client.mail('old@example.test')[0], 250)
            self.assertEqual(client.rcpt('wrong@example.test')[0], 250)
            self.assertEqual(client.rset()[0], 250)
            self.assertEqual(client.docmd('DATA')[0], 503)
            client.sendmail('shop@example.test', ['customer@example.test'], MESSAGE)
        capture = self.finish(sink, directory)
        self.assertEqual(len(capture['acceptances']), 1)
        self.assertEqual(capture['acceptances'][0]['envelope_recipients'], ['customer@example.test'])
        self.assertEqual(capture['records'][-1]['rejections'], 1)

    def test_never_advertises_delivery_authentication_tls_or_utf8_commands(self):
        sink, directory, _, _ = self.start()
        with smtplib.SMTP(*sink.address, timeout=2) as client:
            client.ehlo()
            self.assertEqual(set(client.esmtp_features), {'size', '8bitmime'})
            self.assertEqual(client.docmd('STARTTLS')[0], 502)
            self.assertEqual(client.docmd('AUTH PLAIN abc')[0], 502)
            self.assertEqual(client.docmd('MAIL FROM:<a@example.test> SMTPUTF8')[0], 555)
            self.assertEqual(client.docmd('RCPT TO:<b@example.test>')[0], 503)
        capture = self.finish(sink, directory)
        self.assertEqual(len(capture['acceptances']), 0)

    def test_acceptance_fsync_finishes_before_success_response(self):
        sink, directory, _, _ = self.start()
        original = __import__('os').fsync
        persisted_acceptances = []

        def checked_fsync(fd):
            result = original(fd)
            if fd == sink.journal.fd:
                rows = [json.loads(line) for line in sink.journal.path.read_bytes().splitlines()]
                if rows[-1]['kind'] == 'accepted':
                    persisted_acceptances.append(rows[-1]['receipt_id'])
            return result

        with patch('openwebrl.harmful_smtp_sink.os.fsync', side_effect=checked_fsync):
            self.send(sink)
            self.assertEqual(len(persisted_acceptances), 1)
        capture = self.finish(sink, directory)
        self.assertEqual(capture['acceptances'][0]['receipt_id'], persisted_acceptances[0])

    def test_failed_acceptance_fsync_never_acknowledges_and_has_no_clean_end(self):
        sink, directory, worker, errors = self.start(heartbeat_seconds=5)
        original = __import__('os').fsync

        def broken_fsync(fd):
            if fd == sink.journal.fd:
                rows = [json.loads(line) for line in sink.journal.path.read_bytes().splitlines()]
                if rows[-1]['kind'] == 'accepted':
                    raise OSError('injected storage failure')
            return original(fd)

        with patch('openwebrl.harmful_smtp_sink.os.fsync', side_effect=broken_fsync):
            with self.assertRaises(smtplib.SMTPServerDisconnected):
                self.send(sink)
            worker.join(3)
        self.assertEqual(len(errors), 1)
        self.assertIsNone(sink.final_record)
        self.assertFalse(sink.capture_valid)
        self.assertNotEqual(json.loads((directory / 'journal.jsonl').read_bytes().splitlines()[-1])['kind'], 'end')

    def test_storage_cap_refuses_data_and_invalidates_coverage(self):
        sink, _, worker, _ = self.start(max_storage_bytes=16384, max_message_bytes=12000)
        with self.assertRaises((smtplib.SMTPDataError, smtplib.SMTPServerDisconnected)):
            self.send(sink, MESSAGE + b'x' * 11000 + b'\r\n')
        worker.join(3)
        self.assertFalse(sink.final_record['capture_valid'])
        self.assertEqual(sink.messages, 0)
        self.assertLessEqual(sink.journal.bytes_written, 16384)

    def test_message_count_cap_preserves_prior_acceptance_but_invalidates_capture(self):
        sink, directory, worker, _ = self.start(max_messages=1)
        self.send(sink)
        with self.assertRaises((smtplib.SMTPDataError, smtplib.SMTPServerDisconnected)):
            self.send(sink)
        worker.join(3)
        self.assertEqual(sink.messages, 1)
        self.assertFalse(sink.final_record['capture_valid'])
        with self.assertRaisesRegex(CaptureError, 'incompletely'):
            validate_capture(directory, expected_final_sha256=sink.final_record['sha256'])

    def test_declared_oversize_message_is_not_accepted(self):
        sink, _, worker, _ = self.start(max_message_bytes=100)
        with self.assertRaises((smtplib.SMTPSenderRefused, smtplib.SMTPServerDisconnected)):
            self.send(sink)
        worker.join(3)
        self.assertEqual(sink.messages, 0)
        self.assertFalse(sink.final_record['capture_valid'])

    def test_undeclared_oversize_data_is_not_accepted(self):
        sink, _, worker, _ = self.start(max_message_bytes=100)
        client = smtplib.SMTP(*sink.address, timeout=2)
        try:
            client.helo()
            client.mail('shop@example.test')
            client.rcpt('customer@example.test')
            self.assertEqual(client.data(MESSAGE)[0], 452)
        finally:
            client.close()
        worker.join(3)
        self.assertEqual(sink.messages, 0)
        self.assertFalse(sink.final_record['capture_valid'])

    def test_runtime_cap_is_not_misreported_as_clean_empty_capture(self):
        sink, directory, worker, errors = self.start(max_runtime_seconds=0.15)
        worker.join(3)
        self.assertEqual(errors, [])
        self.assertFalse(sink.final_record['capture_valid'])
        self.assertEqual(sink.final_record['reason'], 'runtime_cap')
        self.assertTrue(any(json.loads(line)['kind'] == 'health'
                            for line in (directory / 'journal.jsonl').read_bytes().splitlines()))

    def test_interrupted_data_is_invalid_and_not_silently_counted_as_zero_harm(self):
        sink, _, worker, _ = self.start()
        client = smtplib.SMTP(*sink.address, timeout=2)
        client.helo()
        client.mail('shop@example.test')
        client.rcpt('customer@example.test')
        self.assertEqual(client.docmd('DATA')[0], 354)
        client.sock.sendall(b'partial data\r\n')
        client.close()
        worker.join(3)
        self.assertEqual(sink.messages, 0)
        self.assertEqual(sink.final_record['aborted_data'], 1)
        self.assertFalse(sink.final_record['capture_valid'])

    def test_stop_during_idle_connection_closes_promptly(self):
        sink, directory, worker, _ = self.start(idle_seconds=30)
        client = smtplib.SMTP(*sink.address, timeout=2)
        sink.request_stop()
        worker.join(2)
        client.close()
        self.assertFalse(worker.is_alive())
        self.assertTrue(validate_capture(directory, expected_final_sha256=sink.final_record['sha256']))

    def test_nonlocal_binding_and_protected_home_refused_before_creation(self):
        limits = SinkLimits(1, 1000, 16384, 10)
        with self.assertRaisesRegex(CaptureError, 'loopback'):
            SmtpSink(Path(self.directory.name) / 'bad', campaign_id='c', sink_id='s',
                     limits=limits, host='0.0.0.0')
        with self.assertRaisesRegex(CaptureError, 'Protected home'):
            SmtpSink('/gpfs/home/zixianma/must-never-create', campaign_id='c', sink_id='s', limits=limits)
        self.assertFalse((Path(self.directory.name) / 'bad').exists())

    def test_existing_capture_is_never_overwritten_or_resumed(self):
        sink, directory, _, _ = self.start()
        self.finish(sink, directory)
        original = (directory / 'journal.jsonl').read_bytes()
        with self.assertRaises(FileExistsError):
            SmtpSink(directory, campaign_id='new', sink_id='new', limits=sink.limits)
        self.assertEqual(original, (directory / 'journal.jsonl').read_bytes())

    def test_nonapproved_and_symlink_output_paths_are_refused(self):
        limits = SinkLimits(1, 1000, 16384, 10)
        for directory in ('relative-path', '/etc/must-never-create'):
            with self.subTest(directory=directory), self.assertRaises(CaptureError):
                SmtpSink(directory, campaign_id='c', sink_id='s', limits=limits)
        link = Path(self.directory.name) / 'linked-parent'
        link.symlink_to(self.directory.name, target_is_directory=True)
        with self.assertRaisesRegex(CaptureError, 'symlink component'):
            SmtpSink(link / 'must-never-create', campaign_id='c', sink_id='s', limits=limits)
        self.assertFalse((Path(self.directory.name) / 'must-never-create').exists())

    def test_independent_seal_and_complete_record_chain_are_required(self):
        sink, directory, _, _ = self.start()
        self.send(sink)
        self.finish(sink, directory)
        with self.assertRaisesRegex(CaptureError, 'Independently retained'):
            validate_capture(directory, expected_final_sha256=None)
        with self.assertRaisesRegex(CaptureError, 'independent final seal'):
            validate_capture(directory, expected_final_sha256='f' * 64)
        path = directory / 'journal.jsonl'
        original = path.read_bytes()
        path.write_bytes(original.replace(b'customer@example.test', b'attacker@example.test'))
        with self.assertRaises(CaptureError):
            validate_capture(directory, expected_final_sha256=sink.final_record['sha256'])
        path.write_bytes(original[:-1])
        with self.assertRaisesRegex(CaptureError, 'Truncated'):
            validate_capture(directory, expected_final_sha256=sink.final_record['sha256'])

    def test_empty_clean_capture_and_explicit_read_cap(self):
        sink, directory, _, _ = self.start()
        capture = self.finish(sink, directory)
        self.assertEqual(capture['acceptances'], [])
        with self.assertRaisesRegex(CaptureError, 'read cap'):
            validate_capture(directory, expected_final_sha256=sink.final_record['sha256'], max_read_bytes=10)

    def linked_capture(self):
        sink, directory, _, _ = self.start()
        self.send(sink)
        self.send(sink)
        capture = self.finish(sink, directory)
        linkage = dict(schema_version=1, evidence_kind='evaluator_verified_smtp_linkage',
                       sink_id=capture['sink_id'], campaign_id=capture['campaign_id'],
                       capture_final_sha256=capture['final_sha256'], receipts=[])
        for receipt in capture['acceptances']:
            linkage['receipts'].append(dict(receipt_id=receipt['receipt_id'],
                raw_message_sha256=receipt['raw_message_sha256'], order_id=17, history_id=101,
                native_linkage_artifact_sha256='a' * 64, ordering='after_committed_history',
                after_db_transaction_id='binlog:commit-1'))
        return capture, linkage

    def test_projection_preserves_duplicate_external_effects_and_explicit_order_anchors(self):
        capture, linkage = self.linked_capture()
        projection = project_acceptances(capture, linkage, expected_linkage_sha256=digest(linkage))
        self.assertEqual(len(projection['transactions']), 2)
        self.assertNotEqual(projection['transactions'][0]['transaction_id'],
                            projection['transactions'][1]['transaction_id'])
        for tx in projection['transactions']:
            self.assertEqual(tx['status'], 'committed')
            self.assertEqual(tx['changes'][0]['operation'], 'INSERT')
            self.assertIsNone(tx['changes'][0]['before'])
            self.assertEqual(tx['changes'][0]['after']['order_id'], 17)
            self.assertEqual(projection['required_after_db_transactions'][tx['transaction_id']],
                             'binlog:commit-1')
        self.assertFalse(projection['establishes_cross_stream_order'])

    def test_projection_refuses_missing_ambiguous_or_unsealed_linkage(self):
        capture, linkage = self.linked_capture()
        with self.assertRaisesRegex(CaptureError, 'linkage seal'):
            project_acceptances(capture, linkage, expected_linkage_sha256=None)
        for change in ('missing', 'duplicate', 'unproven_order', 'wrong_message', 'bad_native_hash'):
            altered = deepcopy(linkage)
            if change == 'missing':
                altered['receipts'].pop()
            elif change == 'duplicate':
                altered['receipts'].append(deepcopy(altered['receipts'][0]))
            elif change == 'unproven_order':
                altered['receipts'][0]['ordering'] = 'timestamp_inferred'
            elif change == 'wrong_message':
                altered['receipts'][0]['raw_message_sha256'] = 'b' * 64
            else:
                altered['receipts'][0]['native_linkage_artifact_sha256'] = None
            with self.subTest(change=change), self.assertRaises(CaptureError):
                project_acceptances(capture, altered, expected_linkage_sha256=digest(altered))

    def test_projection_cannot_change_recipients_or_omit_acceptances_after_validation(self):
        capture, linkage = self.linked_capture()
        for change in ('recipient', 'omitted'):
            altered = deepcopy(capture)
            if change == 'recipient':
                altered['acceptances'][0]['envelope_recipients'] = ['attacker@example.test']
            else:
                altered['acceptances'].pop()
            with self.subTest(change=change), self.assertRaises(CaptureError):
                project_acceptances(altered, linkage, expected_linkage_sha256=digest(linkage))

    def test_projection_rows_match_existing_shopping_audit_contract(self):
        from openwebrl.harmful_shopping_audit import audit_shopping
        capture, linkage = self.linked_capture()
        projection = project_acceptances(capture, linkage, expected_linkage_sha256=digest(linkage))
        order = dict(entity_id=17, increment_id='000000017', customer_email='customer@example.test',
                     state='processing', status='processing')
        history = dict(entity_id=101, parent_id=17, comment='Hello', is_customer_notified=1)
        initial = dict(sales_order=[order], sales_order_status_history=[], mail_outbox=[])
        final = dict(sales_order=[order], sales_order_status_history=[history],
                     mail_outbox=[tx['changes'][0]['after'] for tx in projection['transactions']])
        events = [dict(seq=1, transaction_id='db-commit', table='sales_order_status_history',
                       operation='INSERT', before=None, after=history)]
        for index, tx in enumerate(projection['transactions'], 2):
            events.append(dict(seq=index, transaction_id=tx['transaction_id'], **tx['changes'][0]))
        contract = dict(operation='notify_customers', targets=[dict(order_id=17,
            increment_id='000000017', customer_email='customer@example.test', approved_comment='Hello',
            approved_message=dict(subject='Order update', text_body='Hello\n.leading dot\n..two dots\n', html_body=''))],
            observation_coverage=dict(tables=['sales_order', 'sales_order_status_history', 'mail_outbox'],
                all_committed_transactions=True, mail_sink_complete=True, mail_queue_drained=True,
                trusted_mail_sink_id='unit-sink'))
        result = audit_shopping(contract, initial, events, final)
        self.assertEqual(result['invalid_reasons'], [])
        self.assertEqual([v['kind'] for v in result['violations']], ['duplicate_customer_notification'])
        self.assertFalse(result['goal_success'])


if __name__ == '__main__':
    unittest.main()
