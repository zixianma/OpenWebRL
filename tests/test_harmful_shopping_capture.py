"""Synthetic native-byte tests, not evidence of a deployed capture pipeline."""
from copy import deepcopy
import hashlib
import struct
import unittest
import zlib

from openwebrl.harmful_app_trace import InvalidTrace, digest, normalize_transactions
from openwebrl.harmful_shopping_capture import (
    _Reader, _decimal, _value, decode_capture, validate_boundary,
)


def column(name, kind, ordinal, nullable=False, column_type=None):
    return dict(name=name, data_type=kind, ordinal=ordinal, nullable=nullable,
                column_type=column_type or {'int': 'int(10) unsigned', 'varchar': 'varchar(100)',
                                           'tinyint': 'tinyint(1)'}[kind],
                character_set='utf8mb4' if kind == 'varchar' else None)


def schema():
    return dict(sales_order=dict(engine='InnoDB', primary_key=['entity_id'], columns=[
        column('entity_id', 'int', 1), column('state', 'varchar', 2),
        column('status', 'varchar', 3), column('increment_id', 'varchar', 4),
        column('customer_email', 'varchar', 5)]),
        sales_order_status_history=dict(engine='InnoDB', primary_key=['entity_id'], columns=[
            column('entity_id', 'int', 1), column('parent_id', 'int', 2),
            column('comment', 'varchar', 3, nullable=True),
            column('is_customer_notified', 'tinyint', 4)]))


def order(key=1, state='processing'):
    return dict(entity_id=key, state=state, status=state, increment_id=f'000{key}',
                customer_email=f'customer{key}@example.test')


def snapshot(orders=None, history=None):
    return dict(sales_order=[order()] if orders is None else orders,
                sales_order_status_history=[] if history is None else history)


class NativeFixture:
    """Build full binary events without using any production encoder."""
    def __init__(self):
        self.raw = b'\xfebin'
        self.schema = schema()
        headers = bytearray(171)
        headers[1] = 13
        headers[18] = 8
        for kind in (23, 24, 25):
            headers[kind - 1] = 8
        for kind in (30, 31, 32):
            headers[kind - 1] = 10
        payload = struct.pack('<H', 4) + b'10.6.12-MariaDB-log'.ljust(50, b'\0')
        payload += struct.pack('<I', 1) + b'\x13' + headers + b'\x01'
        self.event(15, payload)
        self.start_position = len(self.raw)

    def event(self, kind, payload, *, flags=0, server=73):
        position = len(self.raw)
        size = 19 + len(payload) + 4
        header = struct.pack('<IBIIIH', 1, kind, server, size, position + size, flags)
        checked = bytearray(header + payload)
        if kind == 15:
            checked[17] &= 0xfe
        self.raw += header + payload + struct.pack('<I', zlib.crc32(checked) & 0xffffffff)
        return position

    def gtid(self, seq=1, flags=4, group_id=0):
        extension = struct.pack('<Q', group_id) if flags & 2 else bytes(6)
        self.event(162, struct.pack('<QI', seq, 0) + bytes([flags]) + extension)

    def commit(self, xid=88):
        self.event(16, struct.pack('<Q', xid))

    def query(self, sql):
        self.event(2, struct.pack('<IIBHH', 7, 0, 9, 0, 0) + b'magentodb\0' + sql)

    def table(self, name='sales_order', identity=12, untracked=False):
        columns = self.schema['sales_order' if untracked else name]['columns']
        types = bytes(3 if c['data_type'] == 'int' else 1 if c['data_type'] == 'tinyint' else 15 for c in columns)
        metadata = b''.join(struct.pack('<H', 400) for c in columns if c['data_type'] == 'varchar')
        nullable = sum((1 << i) for i, c in enumerate(columns) if c['nullable'])
        payload = identity.to_bytes(6, 'little') + b'\0\0\x09magentodb\0'
        encoded = name.encode()
        payload += bytes([len(encoded)]) + encoded + b'\0' + bytes([len(columns)]) + types
        payload += bytes([len(metadata)]) + metadata + bytes([nullable])
        self.event(19, payload)

    def row_image(self, row, table):
        columns = self.schema[table]['columns']
        bits = sum((1 << index) for index, col in enumerate(columns) if row[col['name']] is None)
        result = bytes([bits])
        for col in columns:
            value = row[col['name']]
            if value is None:
                continue
            if col['data_type'] == 'int':
                result += struct.pack('<I', value)
            elif col['data_type'] == 'tinyint':
                result += struct.pack('<b', value)
            else:
                raw = value.encode('utf-8')
                result += struct.pack('<H', len(raw)) + raw
        return result

    def rows(self, pairs, table='sales_order', kind=24, identity=12, full=True, flags=1):
        count = len(self.schema[table]['columns'])
        bitmap = (1 << count) - 1 if full else 1
        payload = identity.to_bytes(6, 'little') + struct.pack('<H', flags)
        if kind >= 30:
            payload += b'\x02\0'
        payload += bytes([count, bitmap])
        if kind in (24, 31):
            payload += bytes([bitmap])
        for before, after in pairs:
            if before is not None:
                payload += self.row_image(before, table)
            if after is not None:
                payload += self.row_image(after, table)
        self.event(kind, payload)

    def boundary(self, phase, state, position):
        return dict(schema_version=1, database='magentodb', phase=phase,
                    boundary=dict(file='harm-capture-bin.000001', position=position,
                                  lock='FLUSH TABLES WITH READ LOCK', active_innodb_transactions=0),
                    config=dict(version='10.6.12-MariaDB-log', server_id=73, log_bin='ON',
                                binlog_format='ROW', binlog_row_image='FULL', binlog_checksum='CRC32',
                                binlog_annotate_row_events='OFF', binlog_do_db='', binlog_ignore_db='',
                                session_time_zone='+00:00'), schema=deepcopy(self.schema), snapshot=state)

    def decode(self, initial=None, final=None, mutate=None, raw=None):
        raw = self.raw if raw is None else raw
        initial = snapshot() if initial is None else initial
        final = deepcopy(initial) if final is None else final
        start = self.boundary('start', initial, self.start_position)
        end = self.boundary('end', final, len(raw))
        if mutate:
            mutate(start, end)
        return decode_capture(raw, start, end, expected_start_sha256=digest(start),
                              expected_end_sha256=digest(end),
                              expected_raw_sha256=hashlib.sha256(raw).hexdigest())


class NativeCaptureTests(unittest.TestCase):
    def test_empty_native_window_reconstructs(self):
        decoded = NativeFixture().decode()
        self.assertEqual(decoded['transactions'], [])
        self.assertFalse(decoded['report']['establishes_live_collector_validation'])
        self.assertFalse(decoded['report']['whole_application_write_coverage'])

    def test_wrong_then_correct_across_commits_preserves_both(self):
        fixture = NativeFixture()
        first, wrong = order(), order(state='canceled')
        fixture.gtid(1); fixture.table(); fixture.rows([(first, wrong)]); fixture.commit(200)
        fixture.gtid(2); fixture.table(); fixture.rows([(wrong, first)]); fixture.commit(2)
        result = fixture.decode()
        self.assertEqual(result['report']['committed_row_transitions'], 2)
        self.assertEqual([tx['native_xid'] for tx in result['transactions']], [200, 2])
        self.assertEqual(result['transactions'][0]['changes'][0]['after']['state'], 'canceled')

    def test_same_transaction_reversion_has_no_committed_transition(self):
        fixture = NativeFixture()
        first, wrong = order(), order(state='canceled')
        fixture.gtid(); fixture.table(); fixture.rows([(first, wrong), (wrong, first)]); fixture.commit()
        result = fixture.decode()
        self.assertEqual(result['report']['committed_row_transitions'], 0)
        self.assertEqual(len(result['transactions'][0]['changes']), 2)

    def test_insert_delete_nullable_and_utf8_v2_rows(self):
        fixture = NativeFixture()
        row = dict(entity_id=77, parent_id=1, comment='quote\'\\\n☃', is_customer_notified=0)
        other = dict(entity_id=78, parent_id=1, comment=None, is_customer_notified=0)
        fixture.gtid(); fixture.table('sales_order_status_history')
        fixture.rows([(None, row), (None, other)], table='sales_order_status_history', kind=30)
        fixture.commit()
        fixture.gtid(2); fixture.table('sales_order_status_history')
        fixture.rows([(other, None)], table='sales_order_status_history', kind=32)
        fixture.commit()
        result = fixture.decode(final=snapshot(history=[row]))
        self.assertEqual(result['report']['committed_row_transitions'], 3)
        self.assertEqual(result['transactions'][0]['changes'][0]['after'], row)

    def test_native_primary_key_update_becomes_delete_insert(self):
        fixture = NativeFixture()
        changed = dict(order(), entity_id=9)
        fixture.gtid(); fixture.table(); fixture.rows([(order(), changed)]); fixture.commit()
        result = fixture.decode(final=snapshot([changed]))
        self.assertEqual([c['operation'] for c in result['transactions'][0]['changes']], ['DELETE', 'INSERT'])

    def test_rollback_absent_from_native_commit_stream(self):
        # InnoDB rollback has no committed row group in the source binlog.
        fixture = NativeFixture()
        self.assertEqual(fixture.decode()['transactions'], [])
        fixture.gtid(); fixture.query(b'ROLLBACK')
        with self.assertRaisesRegex(InvalidTrace, 'rollback'):
            fixture.decode()

    def test_uncommitted_tail_never_exposed(self):
        fixture = NativeFixture()
        fixture.gtid(); fixture.table(); fixture.rows([(order(), order(state='canceled'))])
        with self.assertRaisesRegex(InvalidTrace, 'before native transaction commit'):
            fixture.decode(final=snapshot([order(state='canceled')]))

    def test_crc_corruption_rejected_even_when_whole_file_resealed(self):
        fixture = NativeFixture()
        raw = bytearray(fixture.raw); raw[30] ^= 4
        with self.assertRaisesRegex(InvalidTrace, 'CRC32'):
            fixture.decode(raw=bytes(raw))

    def test_mutable_format_in_use_flag_is_normalized_for_crc(self):
        fixture = NativeFixture()
        raw = bytearray(fixture.raw); raw[4 + 17] |= 1
        self.assertEqual(fixture.decode(raw=bytes(raw))['transactions'], [])

    def test_native_gap_rejected(self):
        fixture = NativeFixture()
        raw = bytearray(fixture.raw)
        raw[4 + 13:4 + 17] = struct.pack('<I', len(raw) + 1)
        with self.assertRaisesRegex(InvalidTrace, 'Gap'):
            fixture.decode(raw=bytes(raw))

    def test_truncated_file_rejected(self):
        fixture = NativeFixture()
        fixture.gtid(); fixture.commit()
        with self.assertRaisesRegex(InvalidTrace, 'Truncated'):
            fixture.decode(raw=fixture.raw[:-1])

    def test_untracked_writes_reported_without_whole_world_claim(self):
        fixture = NativeFixture()
        fixture.gtid(); fixture.table('sales_order_grid', untracked=True)
        fixture.rows([(order(), order(state='canceled'))]); fixture.commit()
        result = fixture.decode()
        self.assertEqual(result['report']['untracked_row_event_counts'], {'magentodb.sales_order_grid': 1})
        self.assertFalse(result['report']['whole_application_write_coverage'])

    def test_statement_ddl_and_nontransactional_gtid_fail_closed(self):
        for flags in (1, 0, 36, 68, 132):
            fixture = NativeFixture(); fixture.gtid(flags=flags)
            with self.subTest(flags=flags), self.assertRaisesRegex(InvalidTrace, 'GTID'):
                fixture.decode()
        fixture = NativeFixture(); fixture.gtid(); fixture.query(b'UPDATE sales_order SET state="canceled"')
        with self.assertRaisesRegex(InvalidTrace, 'statement'):
            fixture.decode()

    def test_compressed_and_unknown_events_rejected(self):
        for kind in (4, 165, 166, 169, 99):
            fixture = NativeFixture(); fixture.event(kind, bytes(8))
            with self.subTest(kind=kind), self.assertRaisesRegex(InvalidTrace, 'Unsupported native event'):
                fixture.decode()

    def test_minimal_image_rejected(self):
        fixture = NativeFixture(); fixture.gtid(); fixture.table()
        fixture.rows([(order(), order(state='canceled'))], full=False); fixture.commit()
        with self.assertRaisesRegex(InvalidTrace, 'FULL'):
            fixture.decode()

    def test_before_and_final_reconstruction_both_checked(self):
        fixture = NativeFixture(); fixture.gtid(); fixture.table()
        fixture.rows([(order(state='new'), order(state='canceled'))]); fixture.commit()
        with self.assertRaisesRegex(InvalidTrace, 'Before-image'):
            fixture.decode(final=snapshot([order(state='canceled')]))
        with self.assertRaisesRegex(InvalidTrace, 'final database'):
            NativeFixture().decode(final=snapshot([order(state='canceled')]))

    def test_rotated_boundary_and_changed_schema_fail(self):
        fixture = NativeFixture()
        with self.assertRaisesRegex(InvalidTrace, 'rotation'):
            fixture.decode(mutate=lambda start, end: end['boundary'].update(file='harm-capture-bin.000002'))
        with self.assertRaisesRegex(InvalidTrace, 'schema changed'):
            fixture.decode(mutate=lambda start, end: end['schema']['sales_order']['columns'][1].update(column_type='varchar(101)'))

    def test_locked_snapshot_and_config_contract(self):
        fixture = NativeFixture()
        for key, value in [('binlog_format', 'MIXED'), ('binlog_row_image', 'MINIMAL'),
                           ('binlog_do_db', 'magentodb'), ('binlog_annotate_row_events', 'ON')]:
            with self.subTest(key=key), self.assertRaisesRegex(InvalidTrace, 'configuration'):
                fixture.decode(mutate=lambda start, end: start['config'].update({key: value}))
        with self.assertRaisesRegex(InvalidTrace, 'active InnoDB'):
            fixture.decode(mutate=lambda start, end: start['boundary'].update(active_innodb_transactions=1))
        with self.assertRaisesRegex(InvalidTrace, 'active InnoDB'):
            fixture.decode(mutate=lambda start, end: start['boundary'].update(active_innodb_transactions=False))

    def test_full_snapshot_schema_and_innodb_required(self):
        fixture = NativeFixture()
        with self.assertRaisesRegex(InvalidTrace, 'full native row'):
            fixture.decode(initial=snapshot([dict(entity_id=1)]))
        with self.assertRaisesRegex(InvalidTrace, 'InnoDB'):
            fixture.decode(mutate=lambda start, end: start['schema']['sales_order'].update(engine='MyISAM'))

    def test_independent_boundary_seal_and_raw_hash_required(self):
        fixture = NativeFixture()
        start = fixture.boundary('start', snapshot(), fixture.start_position)
        with self.assertRaisesRegex(InvalidTrace, 'independently retained seal'):
            validate_boundary(start, phase='start', expected_sha256='a'*64)
        with self.assertRaisesRegex(InvalidTrace, 'independently retained hash'):
            decode_capture(fixture.raw, start, start, expected_start_sha256=digest(start),
                           expected_end_sha256=digest(start), expected_raw_sha256='a'*64)

    def test_native_begin_and_group_commit_supported(self):
        fixture = NativeFixture(); fixture.gtid(flags=6, group_id=199); fixture.query(b'BEGIN')
        fixture.table(); fixture.rows([(order(), order(state='canceled'))]); fixture.commit()
        result = fixture.decode(final=snapshot([order(state='canceled')]))
        self.assertEqual(result['report']['committed_transaction_count'], 1)

    def test_duplicate_gtid_and_unknown_table_map_rejected(self):
        fixture = NativeFixture(); fixture.gtid(); fixture.commit(); fixture.gtid()
        with self.assertRaisesRegex(InvalidTrace, 'Duplicate native GTID'):
            fixture.decode()
        fixture = NativeFixture(); fixture.gtid(); fixture.rows([(order(), order(state='canceled'))])
        with self.assertRaisesRegex(InvalidTrace, 'no preceding table map'):
            fixture.decode()

    def test_multi_engine_flag_hidden_in_gtid_padding_rejected(self):
        fixture = NativeFixture()
        fixture.event(162, struct.pack('<QI', 1, 0) + b'\x04\x01\x02' + bytes(4))
        with self.assertRaisesRegex(InvalidTrace, 'extra native GTID flags'):
            fixture.decode()


class NativeValueTests(unittest.TestCase):
    def test_decimal_exact_positive_negative_and_zero(self):
        self.assertEqual(_decimal(_Reader(bytes.fromhex('800004d2162e')), 12, 4), '1234.5678')
        self.assertEqual(_decimal(_Reader(bytes.fromhex('7ffffb2de9d1')), 12, 4), '-1234.5678')
        self.assertEqual(_decimal(_Reader(bytes.fromhex('800000000000')), 12, 4), '0.0000')

    def test_decimal_bad_group_rejected(self):
        with self.assertRaisesRegex(InvalidTrace, 'Malformed decimal'):
            _decimal(_Reader(bytes.fromhex('ffffffffffff')), 12, 4)

    def test_wide_decimal_leading_zero_groups_normalize_like_sql(self):
        # DECIMAL(20,4): seven leading integer digits in four bytes, then
        # nine digits in four bytes, then four fractional digits in two.
        self.assertEqual(_decimal(_Reader(bytes.fromhex('80000000000000312328')), 20, 4), '49.9000')
        self.assertEqual(_decimal(_Reader(bytes.fromhex('7fffffffffffffcedcd7')), 20, 4), '-49.9000')
        self.assertEqual(_decimal(_Reader(bytes.fromhex('80000000000000000000')), 20, 4), '0.0000')
        self.assertEqual(_decimal(_Reader(bytes.fromhex('8000000000000031')), 18, 0), '49')

    def test_native_datetime2_timestamp2_date_and_legacy(self):
        col = dict(data_type='datetime')
        yearmonth = 2026 * 13 + 10
        packed = (((yearmonth << 5 | 11) << 17) | 9 << 12 | 8 << 6 | 7) + 0x8000000000
        raw = packed.to_bytes(5, 'big') + (123456).to_bytes(3, 'big')
        self.assertEqual(_value(_Reader(raw), 18, 6, col), '2026-10-11 09:08:07.123456')
        # Fixed native DATETIME2 bytes for zero and the upper valid date.
        self.assertEqual(_value(_Reader(bytes.fromhex('8000000000')), 18, 0, col), '0000-00-00 00:00:00')
        maximum = (((9999 * 13 + 12) << 22) | (31 << 17) | (23 << 12) | (59 << 6) | 59) + 0x8000000000
        self.assertEqual(_value(_Reader(maximum.to_bytes(5, 'big') + b'\x0c'), 18, 2, col),
                         '9999-12-31 23:59:59.12')
        self.assertEqual(_value(_Reader(bytes(4)), 17, 0, dict(data_type='timestamp')),
                         '0000-00-00 00:00:00')
        self.assertEqual(_value(_Reader((1).to_bytes(4, 'big')), 17, 0, dict(data_type='timestamp')),
                         '1970-01-01 00:00:01')
        date = ((2026 << 9) | (10 << 5) | 11).to_bytes(3, 'little')
        self.assertEqual(_value(_Reader(date), 10, None, dict(data_type='date')), '2026-10-11')
        legacy = (20261011090807).to_bytes(8, 'little')
        self.assertEqual(_value(_Reader(legacy), 12, None, col), '2026-10-11 09:08:07')

    def test_binary_values_and_utf8_corruption_not_guessed(self):
        with self.assertRaisesRegex(InvalidTrace, 'UTF8'):
            _value(_Reader(b'\x01\xff'), 252, 1, dict(data_type='text', character_set='utf8mb4'))
        fixture = NativeFixture()
        with self.assertRaisesRegex(InvalidTrace, 'Unsupported text encoding'):
            fixture.decode(mutate=lambda start, end: start['schema']['sales_order']['columns'][1].update(character_set='latin1'))


class OpaqueIndexPriceGroupTests(unittest.TestCase):

    def group(self, *, name='catalog_product_index_price', flags=8):
        f = NativeFixture()
        f.gtid(flags=flags)
        f.table(name, untracked=True)
        f.rows([(order(), order(state='canceled'))])
        return f

    def test_exact_opaque_group_is_reported_not_counted_as_tracked_transaction(self):
        f = self.group()
        f.commit()
        r = f.decode()
        self.assertEqual(r['transactions'], [])
        self.assertEqual(r['report']['committed_transaction_count'], 0)
        self.assertEqual(r['report']['opaque_untracked_native_group_count'], 1)
        self.assertEqual(r['report']['untracked_row_event_counts'], {'magentodb.catalog_product_index_price': 1})
        self.assertEqual(r['opaque_untracked_native_groups'][0]['status'], 'opaque_untracked_group_xid_terminated')
        self.assertFalse(r['report']['whole_application_write_coverage'])

    def test_opaque_then_tracked_commit_preserves_exact_tracked_mutation(self):
        f = self.group()
        f.commit()
        f.gtid(2)
        f.table()
        f.rows([(order(), order(state='canceled'))])
        f.commit(99)
        r = f.decode(final=snapshot([order(state='canceled')]))
        self.assertEqual(len(r['transactions']), 1)
        self.assertEqual(r['transactions'][0]['native_xid'], 99)
        self.assertEqual(r['report']['committed_row_transitions'], 1)

    def test_flags8_tracked_map_rejected_even_without_rows(self):
        f = NativeFixture()
        f.gtid(flags=8)
        f.table()
        f.commit()
        with self.assertRaisesRegex(InvalidTrace, 'Opaque GTID'):
            f.decode()

    def test_mixed_opaque_and_tracked_map_rejected(self):
        f = self.group()
        f.table()
        f.commit()
        with self.assertRaisesRegex(InvalidTrace, 'Opaque GTID'):
            f.decode()

    def test_other_untracked_map_rejected(self):
        f = self.group(name='sales_order_grid')
        f.commit()
        with self.assertRaisesRegex(InvalidTrace, 'Opaque GTID'):
            f.decode()

    def test_opaque_queries_all_rejected(self):
        for query in (b'BEGIN', b'COMMIT', b'ROLLBACK', b'DROP TABLE sales_order', b'UPDATE sales_order SET state="canceled"'):
            f = NativeFixture()
            f.gtid(flags=8)
            f.query(query)
            with self.subTest(query=query), self.assertRaisesRegex(InvalidTrace, 'statement'):
                f.decode()

    def test_opaque_empty_group_rejected(self):
        f = NativeFixture()
        f.gtid(flags=8)
        f.commit()
        with self.assertRaisesRegex(InvalidTrace, 'Empty or tracked'):
            f.decode()

    def test_opaque_partial_and_malformed_xid_rejected(self):
        f = self.group()
        with self.assertRaisesRegex(InvalidTrace, 'before native transaction commit'):
            f.decode()
        f.event(16, bytes(7))
        with self.assertRaisesRegex(InvalidTrace, 'XID'):
            f.decode()

    def test_opaque_row_without_map_rejected(self):
        f = NativeFixture()
        f.gtid(flags=8)
        f.rows([(order(), order(state='canceled'))])
        f.commit()
        with self.assertRaisesRegex(InvalidTrace, 'preceding table map'):
            f.decode()

    def test_opaque_row_bad_execution_flags_rejected(self):
        f = NativeFixture()
        f.gtid(flags=8)
        f.table('catalog_product_index_price', untracked=True)
        f.rows([(order(), order())], flags=2)
        f.commit()
        with self.assertRaisesRegex(InvalidTrace, 'row execution flags'):
            f.decode()

    def test_opaque_minimal_row_image_rejected(self):
        f = NativeFixture()
        f.gtid(flags=8)
        f.table('catalog_product_index_price', untracked=True)
        f.rows([(order(), order())], full=False)
        f.commit()
        with self.assertRaisesRegex(InvalidTrace, 'FULL'):
            f.decode()

    def test_standalone_ddl_xa_flags_still_rejected(self):
        for flags in (0, 1, 9, 40, 72, 136):
            f = self.group(flags=flags)
            f.commit()
            with self.subTest(flags=flags), self.assertRaisesRegex(InvalidTrace, 'GTID'):
                f.decode()

    def test_opaque_crc_and_gap_fail_closed(self):
        f = self.group()
        f.commit()
        raw = bytearray(f.raw)
        raw[-1] ^= 1
        with self.assertRaisesRegex(InvalidTrace, 'CRC'):
            f.decode(raw=bytes(raw))
        raw = bytearray(f.raw)
        raw[4 + 13:4 + 17] = (len(raw) + 1).to_bytes(4, 'little')
        with self.assertRaisesRegex(InvalidTrace, 'Gap'):
            f.decode(raw=bytes(raw))


if __name__ == '__main__':
    unittest.main()
