"""Fail-closed, offline MariaDB 10.6 ROW/FULL binlog evidence reader.

This is a deliberately narrow native decoder for the two Magento sales tables,
not a replication client or a claim of live capture readiness. It reads raw file
bytes (never mysqlbinlog's display SQL), checks every CRC and byte position, and
exposes changes only after a transactional MariaDB GTID group ends with XID.
There is no polling, trigger journal, inferred commit, or timestamp ordering.

The evaluator must retain the entire binlog prefix through the end receipt,
independent seals for both read-locked snapshots, and proof that actor execution
was between those boundaries. It must separately capture SMTP acceptances and
prove queue quiescence. A SHA256 is not proof of collector trust or completeness.
Only the named tables' full rows are reconstructed; other row writes are listed
diagnostically and prevent a claim of whole-application write coverage. DDL,
statement writes, XA, log rotation in the window, unsupported types/encodings,
nontransactional tracked groups, and schema drift are rejected. A narrow offline
amendment preserves exact flags=8 ROW/FULL index-price groups as opaque evidence,
never as tracked transactions or evidence of safe whole-application effects.

Format references (live protocol validation remains required):
https://mariadb.com/docs/server/reference/clientserver-protocol/replication-protocol
https://mariadb.com/docs/server/reference/clientserver-protocol/replication-protocol/table_map_event
https://mariadb.com/docs/server/reference/clientserver-protocol/replication-protocol/gtid_event
https://github.com/MariaDB/server/blob/mariadb-10.6.12/sql/log_event_server.cc
https://github.com/MariaDB/server/blob/mariadb-10.6.12/sql/compat56.cc
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import re
import struct
import zlib

from openwebrl.harmful_app_trace import InvalidTrace, digest, normalize_transactions


TABLES = ('sales_order', 'sales_order_status_history')
DATABASE = 'magentodb'
MAX_CAPTURE_BYTES = 512 * 1024 * 1024
MAX_EVENT_BYTES = 64 * 1024 * 1024
MAX_ROWS = 100000
INTEGER_TYPES = {'tinyint': (1, 1), 'smallint': (2, 2), 'mediumint': (9, 3),
                 'int': (3, 4), 'bigint': (8, 8)}
TEXT_TYPES = {'varchar', 'char', 'tinytext', 'text', 'mediumtext', 'longtext'}
SUPPORTED_TYPES = set(INTEGER_TYPES) | TEXT_TYPES | {'decimal', 'datetime', 'timestamp', 'date'}


def _need(condition, message):
    if not condition:
        raise InvalidTrace(message)


class _Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, size):
        _need(type(size) is int and size >= 0 and self.pos + size <= len(self.data),
              'Truncated native binlog field')
        result = self.data[self.pos:self.pos + size]
        self.pos += size
        return result

    def uint(self, size, order='little'):
        return int.from_bytes(self.take(size), order)

    def length(self):
        lead = self.uint(1)
        if lead < 251:
            return lead
        _need(lead in (252, 253, 254), 'Invalid length-encoded integer')
        return self.uint({252: 2, 253: 3, 254: 8}[lead])

    def done(self):
        _need(self.pos == len(self.data), 'Unexpected trailing native event bytes')

    def remains(self):
        return len(self.data) - self.pos


def _schema(schema):
    _need(isinstance(schema, dict) and set(schema) == set(TABLES),
          'Schema must cover exactly both tracked native tables')
    for table, spec in schema.items():
        _need(isinstance(spec, dict) and spec.get('engine') == 'InnoDB',
              'Tracked tables must use transactional InnoDB')
        _need(spec.get('primary_key') == ['entity_id'], 'Unsupported native primary key')
        columns = spec.get('columns')
        _need(isinstance(columns, list) and 0 < len(columns) <= 1024,
              'Missing complete ordered native column schema')
        names = set()
        for index, col in enumerate(columns, 1):
            _need(isinstance(col, dict) and type(col.get('ordinal')) is int and col['ordinal'] == index,
                  'Native column ordinals must be consecutive')
            name = col.get('name')
            _need(isinstance(name, str) and re.fullmatch('[a-zA-Z_][a-zA-Z0-9_]*', name)
                  and name not in names, 'Invalid/duplicate native column name')
            names.add(name)
            kind = col.get('data_type')
            _need(kind in SUPPORTED_TYPES, 'Unsupported native column type: ' + str(kind))
            _need(type(col.get('nullable')) is bool and isinstance(col.get('column_type'), str),
                  'Missing column type/nullability')
            if kind in TEXT_TYPES:
                _need(col.get('character_set') in ('utf8', 'utf8mb3', 'utf8mb4', 'ascii'),
                      'Unsupported text encoding; binary/latin1 columns are not guessed')
        _need('entity_id' in names, 'Missing native row identity')
        identity = next(col for col in columns if col['name'] == 'entity_id')
        _need(identity['data_type'] in INTEGER_TYPES and identity['nullable'] is False,
              'Native entity_id must be a non-null integer')


def _snapshot(snapshot, schema):
    _need(isinstance(snapshot, dict) and set(snapshot) == set(TABLES),
          'Snapshot misses tracked table coverage')
    for table, rows in snapshot.items():
        _need(isinstance(rows, list) and len(rows) <= MAX_ROWS, 'Invalid/oversized native snapshot')
        columns = schema[table]['columns']
        names = {col['name'] for col in columns}
        seen = set()
        for row in rows:
            _need(isinstance(row, dict) and set(row) == names, 'Snapshot is not a full native row')
            _need(type(row['entity_id']) is int and row['entity_id'] > 0
                  and row['entity_id'] not in seen, 'Invalid/duplicate native entity_id')
            seen.add(row['entity_id'])
            for col in columns:
                value = row[col['name']]
                if value is None:
                    _need(col['nullable'], 'NULL in non-null native column')
                elif col['data_type'] in INTEGER_TYPES:
                    _need(type(value) is int and -(2**63) <= value < 2**63,
                          'Integer snapshot must preserve exact signed-64 value')
                else:
                    _need(isinstance(value, str), 'Noninteger snapshot values must be exact strings')


def validate_boundary(receipt, *, phase, expected_sha256):
    """Validate a sealed PHP boundary receipt; host exclusion is a separate gate."""
    _need(re.fullmatch('[0-9a-f]{64}', expected_sha256 or '') is not None
          and digest(receipt) == expected_sha256, 'Native boundary differs from independently retained seal')
    _need(type(receipt.get('schema_version')) is int and receipt['schema_version'] == 1 and receipt.get('database') == DATABASE
          and receipt.get('phase') == phase, 'Wrong native boundary identity')
    boundary = receipt.get('boundary', {})
    _need(boundary.get('lock') == 'FLUSH TABLES WITH READ LOCK'
          and type(boundary.get('active_innodb_transactions')) is int
          and boundary['active_innodb_transactions'] == 0,
          'Snapshot requires a locked boundary with no active InnoDB transaction')
    _need(isinstance(boundary.get('file'), str)
          and re.fullmatch(r'[a-zA-Z0-9_.-]+\.\d{6}', boundary['file']) is not None
          and type(boundary.get('position')) is int and boundary['position'] >= 4,
          'Missing exact native binlog boundary')
    config = receipt.get('config', {})
    for name, value in {'log_bin': 'ON', 'binlog_format': 'ROW', 'binlog_row_image': 'FULL',
                        'binlog_checksum': 'CRC32', 'binlog_annotate_row_events': 'OFF',
                        'binlog_do_db': '', 'binlog_ignore_db': '',
                        'session_time_zone': '+00:00'}.items():
        _need(str(config.get(name)) == value, 'Unsupported native capture configuration: ' + name)
    _need(re.match(r'^10\.6\.12(?:[-.]|$)', str(config.get('version', ''))) is not None,
          'Collector is pinned to MariaDB 10.6.12')
    _need(type(config.get('server_id')) is int and config['server_id'] > 0,
          'Missing stable native server id')
    _schema(receipt.get('schema'))
    _snapshot(receipt.get('snapshot'), receipt['schema'])
    return receipt


def _events(raw):
    _need(isinstance(raw, bytes) and 4 <= len(raw) <= MAX_CAPTURE_BYTES
          and raw[:4] == b'\xfebin', 'Expected bounded native binlog file prefix')
    pos, first, postheaders, server_version = 4, True, None, None
    while pos < len(raw):
        _need(len(raw) - pos >= 19, 'Truncated native event header')
        timestamp, kind, server, size, nextpos, flags = struct.unpack_from('<IBIIIH', raw, pos)
        _need(23 <= size <= MAX_EVENT_BYTES and pos + size <= len(raw),
              'Truncated/oversized native binlog event')
        _need(nextpos == pos + size, 'Gap, reordered bytes or artificial native event position')
        event = raw[pos:nextpos]
        checked = bytearray(event[:-4])
        if first:
            _need(kind == 15 and size >= 81, 'Native prefix must begin with format description')
            _need(event[-5] == 1, 'Native binlog must use CRC32 checksums')
            # The server toggles BINLOG_IN_USE_F in place. The FDE checksum is
            # calculated with this mutable flag cleared (log_event_server.cc).
            checked[17] &= 0xfe
        _need(zlib.crc32(checked) & 0xffffffff == int.from_bytes(event[-4:], 'little'),
              'Native event CRC32 mismatch')
        payload = event[19:-4]
        if first:
            _need(int.from_bytes(payload[:2], 'little') == 4 and payload[56] == 19,
                  'Unsupported binlog/header version')
            server_version = payload[2:52].split(b'\0')[0].decode('ascii')
            _need(re.match(r'^10\.6\.12(?:[-.]|$)', server_version) is not None,
                  'Native log server differs from pinned MariaDB 10.6.12')
            postheaders = payload[57:-1]
            first = False
        else:
            _need(kind != 15, 'Unexpected second native format event')
        yield dict(position=pos, next_position=nextpos, kind=kind, server_id=server,
                   timestamp=timestamp, flags=flags, payload=payload,
                   sha256=hashlib.sha256(event).hexdigest(), postheaders=postheaders,
                   server_version=server_version)
        pos = nextpos


def _table_map(payload, schema):
    reader = _Reader(payload)
    table_id, flags = reader.uint(6), reader.uint(2)
    database = reader.take(reader.uint(1)).decode('ascii')
    _need(reader.uint(1) == 0, 'Table-map database terminator missing')
    name = reader.take(reader.uint(1)).decode('ascii')
    _need(reader.uint(1) == 0, 'Table-map table terminator missing')
    count = reader.length()
    _need(0 < count <= 4096, 'Invalid native column count')
    types = list(reader.take(count))
    metadata = reader.take(reader.length())
    nullable = reader.take((count + 7) // 8)
    # Optional row metadata remains raw and sealed; column identity and types
    # come from the independently locked full schema. Validate TLV framing.
    while reader.remains():
        reader.uint(1)
        reader.take(reader.length())
    result = dict(table_id=table_id, database=database, table=name, flags=flags,
                  types=types, metadata=metadata, nullable=nullable)
    if database != DATABASE or name not in schema:
        result['tracked'] = False
        return result
    columns = schema[name]['columns']
    _need(count == len(columns), 'Native table column count drifted from locked schema')
    meta = _Reader(metadata)
    specs = []
    for index, (kind, column) in enumerate(zip(types, columns)):
        expected = column['data_type']
        _need(bool(nullable[index // 8] & (1 << (index % 8))) == column['nullable'],
              'Native column nullability differs from locked schema')
        value = None
        if kind in (15, 253):
            value = meta.uint(2)
            _need(expected in ('varchar', 'char'), 'Native string type drift')
        elif kind == 254:
            value = (meta.uint(1), meta.uint(1))
            _need(expected == 'char' and ((value[0] & 0x30) != 0x30 or value[0] == 254),
                  'Unsupported native STRING/ENUM/SET type')
        elif kind == 252:
            value = meta.uint(1)
            _need(expected in TEXT_TYPES and 1 <= value <= 4, 'Unsupported native BLOB/text type')
        elif kind == 246:
            value = (meta.uint(1), meta.uint(1))
            declared = re.fullmatch(r'decimal\((\d+),(\d+)\)(?: unsigned)?', column['column_type'])
            _need(expected == 'decimal' and declared is not None
                  and value == tuple(map(int, declared.groups())), 'Native decimal schema drift')
        elif kind in (17, 18):
            value = meta.uint(1)
            _need((kind == 17 and expected == 'timestamp') or (kind == 18 and expected == 'datetime'),
                  'Native temporal schema drift')
            _need(0 <= value <= 6, 'Unsupported temporal fractional precision')
            precision = re.fullmatch(r'(?:timestamp|datetime)(?:\((\d)\))?', column['column_type'])
            _need(precision is not None and int(precision.group(1) or 0) == value,
                  'Native temporal precision drift')
        elif expected in INTEGER_TYPES:
            _need(kind == INTEGER_TYPES[expected][0], 'Native integer schema drift')
        else:
            _need((kind, expected) in ((7, 'timestamp'), (10, 'date'), (12, 'datetime')),
                  'Unsupported native binary type: ' + str(kind))
        specs.append((kind, value, column))
    meta.done()
    result.update(tracked=True, specs=specs)
    return result


def _decimal(reader, precision, scale):
    _need(1 <= precision <= 65 and 0 <= scale <= min(precision, 30), 'Invalid decimal metadata')
    widths = (0, 1, 1, 2, 2, 3, 3, 4, 4, 4)
    integral = precision - scale
    full_int, partial_int = divmod(integral, 9)
    full_frac, partial_frac = divmod(scale, 9)
    width = full_int * 4 + widths[partial_int] + full_frac * 4 + widths[partial_frac]
    data = bytearray(reader.take(width))
    positive = bool(data[0] & 0x80)
    data[0] ^= 0x80
    if not positive:
        data = bytearray(x ^ 0xff for x in data)
    value = _Reader(data)
    parts = []
    if partial_int:
        chunk = value.uint(widths[partial_int], 'big')
        _need(chunk < 10**partial_int, 'Malformed decimal integral group')
        parts.append(str(chunk))
    for _ in range(full_int):
        chunk = value.uint(4, 'big')
        _need(chunk < 10**9, 'Malformed decimal integral group')
        parts.append(f'{chunk:09d}' if parts else str(chunk))
    before = ''.join(parts).lstrip('0') or '0'
    fraction = ''
    for _ in range(full_frac):
        chunk = value.uint(4, 'big')
        _need(chunk < 10**9, 'Malformed decimal fraction')
        fraction += f'{chunk:09d}'
    if partial_frac:
        chunk = value.uint(widths[partial_frac], 'big')
        _need(chunk < 10**partial_frac, 'Malformed decimal fraction')
        fraction += f'{chunk:0{partial_frac}d}'
    # Canonical SQL text, including declared trailing decimal zeros.
    number = before + ('.' + fraction if scale else '')
    return ('-' if not positive and any(c in '123456789' for c in number) else '') + number


def _fraction(reader, precision):
    if not precision:
        return ''
    raw = reader.uint((precision + 1) // 2, 'big')
    microseconds = raw * 10**(6 - ((precision + 1) // 2) * 2)
    _need(microseconds < 1000000 and microseconds % 10**(6 - precision) == 0,
          'Malformed temporal fraction')
    return '.' + f'{microseconds:06d}'[:precision]


def _value(reader, kind, metadata, column):
    expected = column['data_type']
    if expected in INTEGER_TYPES:
        raw = reader.take(INTEGER_TYPES[expected][1])
        number = int.from_bytes(raw, 'little', signed='unsigned' not in column['column_type'])
        _need(-(2**63) <= number < 2**63, 'Unsigned integer exceeds exact PHP snapshot range')
        return number
    if kind == 246:
        return _decimal(reader, *metadata)
    if kind in (15, 253, 254, 252):
        if kind == 252:
            length = reader.uint(metadata)
        else:
            if kind == 254:
                high, low = metadata
                max_length = low | (((high & 0x30) ^ 0x30) << 4) if (high & 0x30) != 0x30 else low
            else:
                max_length = metadata
            length = reader.uint(2 if max_length > 255 else 1)
            _need(length <= max_length, 'Native string exceeds declared maximum')
        raw = reader.take(length)
        try:
            return raw.decode('ascii' if column['character_set'] == 'ascii' else 'utf-8')
        except UnicodeDecodeError as error:
            raise InvalidTrace('Native text is not valid declared UTF8/ASCII') from error
    if kind == 10:
        date = reader.uint(3)
        return f'{date >> 9:04d}-{(date >> 5) & 15:02d}-{date & 31:02d}'
    if kind == 12:
        val = reader.uint(8)
        date, time = divmod(val, 1000000)
        return f'{date // 10000:04d}-{date // 100 % 100:02d}-{date % 100:02d} {time // 10000:02d}:{time // 100 % 100:02d}:{time % 100:02d}'
    if kind in (7, 17):
        stamp = reader.uint(4, 'big' if kind == 17 else 'little')
        val = ('0000-00-00 00:00:00' if stamp == 0 else
               datetime.fromtimestamp(stamp, timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
        return val + (_fraction(reader, metadata) if kind == 17 else '')
    if kind == 18:
        val = reader.uint(5, 'big') - 0x8000000000
        _need(val >= 0, 'Negative DATETIME2 is unsupported')
        ym, day = val >> 22, (val >> 17) & 31
        year, month = divmod(ym, 13)
        hour, minute, second = (val >> 12) & 31, (val >> 6) & 63, val & 63
        _need(year <= 9999 and month <= 12 and day <= 31 and hour <= 23 and minute <= 59 and second <= 59,
              'Malformed DATETIME2')
        return f'{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}' + _fraction(reader, metadata)
    raise InvalidTrace('Unhandled native value encoding')


def _rows(payload, kind, mapping):
    reader = _Reader(payload)
    table_id, flags = reader.uint(6), reader.uint(2)
    _need(table_id == mapping['table_id'], 'Native row table map identity mismatch')
    if kind in (30, 31, 32):
        extra = reader.uint(2)
        _need(extra == 2, 'Unsupported native row extra data')
    _need(flags & ~1 == 0, 'Unsupported row execution flags')
    count = reader.length()
    _need(count == len(mapping['types']), 'Native row column count drift')
    present = reader.take((count + 7) // 8)
    _need(all(present[index // 8] & (1 << (index % 8)) for index in range(count)),
          'Native row image is not FULL')
    update = kind in (24, 31)
    if update:
        after_present = reader.take((count + 7) // 8)
        _need(after_present == present, 'Native UPDATE after-image is not FULL')
    _need(reader.remains() > 0, 'Empty native row event')
    if not mapping['tracked']:
        # Intentionally no statement about untracked row contents or counts.
        return []

    def row():
        nulls = reader.take((count + 7) // 8)
        value = {}
        for index, (field_kind, meta, column) in enumerate(mapping['specs']):
            null = bool(nulls[index // 8] & (1 << (index % 8)))
            _need(not null or column['nullable'], 'Native NULL in non-null column')
            value[column['name']] = None if null else _value(reader, field_kind, meta, column)
        return value

    result = []
    operation = 'UPDATE' if update else 'INSERT' if kind in (23, 30) else 'DELETE'
    while reader.remains():
        _need(len(result) < MAX_ROWS, 'Native event exceeds bounded row count')
        before = row() if operation != 'INSERT' else None
        after = row() if operation != 'DELETE' else None
        if operation == 'UPDATE' and before['entity_id'] != after['entity_id']:
            # Preserve both full images and native transaction order.
            result.extend([dict(table=mapping['table'], operation='DELETE', before=before, after=None),
                           dict(table=mapping['table'], operation='INSERT', before=None, after=after)])
        else:
            result.append(dict(table=mapping['table'], operation=operation, before=before, after=after))
    return result


def decode_capture(raw_log, start, end, *, expected_start_sha256,
                   expected_end_sha256, expected_raw_sha256):
    """Return exact DB transactions plus scope diagnostics, never a scored trace.

    Both snapshot receipts are produced by ``scripts/capture_harmful_shopping.php``.
    The raw file must begin at byte zero and end at the end receipt's position.
    No log rotation, cross-file gap, schema change or partial transaction is
    accepted. Policy timing, network isolation, SMTP linkage, queues and complete
    world reset require independent evidence from the owning controller.
    """
    _need(hashlib.sha256(raw_log).hexdigest() == expected_raw_sha256,
          'Raw native capture differs from independently retained hash')
    validate_boundary(start, phase='start', expected_sha256=expected_start_sha256)
    validate_boundary(end, phase='end', expected_sha256=expected_end_sha256)
    _need(start['schema'] == end['schema'], 'Native schema changed during capture')
    _need(start['config'] == end['config'], 'Native capture configuration changed')
    first, last = start['boundary'], end['boundary']
    _need(first['file'] == last['file'], 'Binlog rotation during policy window is not supported')
    _need(first['position'] <= last['position'] == len(raw_log),
          'Native file prefix does not exactly end at sealed final boundary')
    schema, maps, current, transactions, seen_gtids = start['schema'], {}, None, [], set()
    boundaries, ignored, native_events = {4}, Counter(), 0
    opaque_groups = []
    for event in _events(raw_log):
        boundaries.update((event['position'], event['next_position']))
        if event['position'] < first['position']:
            _need(event['next_position'] <= first['position'], 'Start boundary bisects a native event')
            continue
        kind, data = event['kind'], event['payload']
        _need(event['server_id'] == start['config']['server_id'], 'Native event server identity changed')
        native_events += 1
        if kind == 162:  # MariaDB GTID, not MySQL's different GTID encoding.
            _need(current is None and len(data) >= 13, 'Nested/truncated native GTID group')
            seq, domain = int.from_bytes(data[:8], 'little'), int.from_bytes(data[8:12], 'little')
            flags = data[12]
            opaque = flags == 8
            _need(seq > 0 and (flags & 4 or opaque) and not flags & (1 | 32 | 64 | 128),
                  'Standalone, nontransactional, DDL or XA native GTID is unsupported')
            _need(len(data) == (21 if flags & 2 else 19), 'Unsupported GTID extension')
            # In 10.6, flags_extra/multi-engine data can occupy the otherwise
            # zero six-byte tail even without changing the event's length.
            _need(flags & 2 or data[13:] == bytes(6),
                  'Unsupported multi-engine or extra native GTID flags')
            txid = f'mariadb:{domain}-{event["server_id"]}-{seq}'
            _need(txid not in seen_gtids, 'Duplicate native GTID')
            seen_gtids.add(txid)
            current = dict(transaction_id=txid, changes=[], native_start_position=event['position'])
            if opaque:
                current.update(opaque_nontransactional_index_group=True, native_gtid_flags=flags,
                               opaque_table_maps=0, opaque_row_events=0)
            maps = {}
        elif kind == 19:
            _need(current is not None, 'Native table map outside transactional GTID')
            _need(len(event['postheaders']) >= kind and event['postheaders'][kind - 1] == 8,
                  'Unsupported native table-map post-header')
            mapping = _table_map(data, schema)
            if current.get('opaque_nontransactional_index_group'):
                _need(not mapping['tracked'] and mapping['database'] == DATABASE
                      and mapping['table'] == 'catalog_product_index_price',
                      'Opaque GTID may map only the exact untracked native index-price table')
                current['opaque_table_maps'] += 1
            maps[mapping['table_id']] = mapping
        elif kind in (23, 24, 25, 30, 31, 32):
            _need(current is not None and len(data) >= 8, 'Native row outside transactional GTID')
            _need(len(event['postheaders']) >= kind
                  and event['postheaders'][kind - 1] == (10 if kind >= 30 else 8),
                  'Unsupported native row post-header')
            table_id = int.from_bytes(data[:6], 'little')
            _need(table_id in maps, 'Native row has no preceding table map')
            mapping = maps[table_id]
            current['changes'].extend(_rows(data, kind, mapping))
            if current.get('opaque_nontransactional_index_group'):
                current['opaque_row_events'] += 1
            if not mapping['tracked']:
                ignored[mapping['database'] + '.' + mapping['table']] += 1
        elif kind == 16:  # The native InnoDB transaction's binlog commit record.
            _need(current is not None and len(data) == 8, 'Orphan/malformed native XID commit')
            xid = int.from_bytes(data, 'little')
            current.update(status='committed',
                           commit_receipt=f'{first["file"]}:{event["position"]}-{event["next_position"]}:xid={xid}:sha256={event["sha256"]}',
                           native_end_position=event['next_position'], native_xid=xid)
            if current.get('opaque_nontransactional_index_group'):
                _need(current['opaque_table_maps'] > 0 and current['opaque_row_events'] > 0
                      and current['changes'] == [], 'Empty or tracked opaque GTID group')
                current['status'] = 'opaque_untracked_group_xid_terminated'
                current['termination_receipt'] = current.pop('commit_receipt')
                opaque_groups.append(current)
            else:
                transactions.append(current)
            current, maps = None, {}
        elif kind == 2:
            # Some native streams include BEGIN in addition to GTID. All other
            # query events (including COMMIT, ROLLBACK and DDL) fail closed.
            query = _Reader(data)
            query.take(8)
            db_length, error, status_length = query.uint(1), query.uint(2), query.uint(2)
            query.take(status_length)
            query.take(db_length)
            _need(query.uint(1) == 0, 'Malformed native Query database terminator')
            statement = query.take(query.remains())
            _need(current is not None and not current.get('opaque_nontransactional_index_group')
                  and error == 0 and statement == b'BEGIN',
                  'Native statement/DDL/rollback query is not a supported committed ROW event')
        elif kind in (160, 161, 163):
            # Annotate_rows is deliberately disabled; checkpoint and GTID_LIST
            # are structural only and must be outside any transaction.
            _need(kind != 160 and current is None, 'Unsupported native annotation/checkpoint placement')
        else:
            raise InvalidTrace('Unsupported native event in capture window: ' + str(kind))
    _need(first['position'] in boundaries and last['position'] in boundaries,
          'Native snapshot boundary is not an event boundary')
    _need(current is None, 'Capture ends before native transaction commit')
    keys = {table: schema[table]['primary_key'] for table in TABLES}
    changes = normalize_transactions(start['snapshot'], end['snapshot'], keys, transactions)
    return dict(schema_version=1, initial=deepcopy(start['snapshot']), final=deepcopy(end['snapshot']),
                table_keys=keys, transactions=transactions,
                opaque_untracked_native_groups=opaque_groups,
                report=dict(status='native_database_capture_decoded', tracked_tables=list(TABLES),
                            native_events_in_window=native_events,
                            committed_transaction_count=len(transactions),
                            committed_row_transitions=len(changes),
                            untracked_row_event_counts=dict(sorted(ignored.items())),
                            opaque_untracked_native_group_count=len(opaque_groups),
                            opaque_group_scope='Exact flags=8, ROW/FULL, magentodb.catalog_product_index_price, XID terminated; no value/rollback/harm semantics claimed',
                            whole_application_write_coverage=False,
                            establishes_policy_boundary_timing=False,
                            establishes_mail_acceptance_or_queue_quiescence=False,
                            establishes_live_collector_validation=False,
                            raw_binlog_sha256=expected_raw_sha256,
                            start_boundary_sha256=expected_start_sha256,
                            end_boundary_sha256=expected_end_sha256,
                            database_schema_sha256=digest(schema)))
