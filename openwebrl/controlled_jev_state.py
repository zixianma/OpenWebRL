"""Separate durable queue and caps for the local Jev direct-actor cohort."""
import hashlib
import json
from pathlib import Path
import time
import uuid

from openwebrl.controlled_sft_state import SuiteState, digest, request_bound

RATES = {'text_http_attempts': ('text_usd', .4, 1.6),
         'om2w_judge_http_attempts': ('judge_usd', 1.1, 4.4)}
KINDS = {'browser_sessions', 'jev_requests', *RATES}


class JevState(SuiteState):
    """Reuse the tested queue; this cohort has its own database and approval."""
    def finish(self, attempt_id, result):
        verified_judge = False
        if result.get('terminal') == 'done' and result.get('valid') is True:
            for path in (Path(result['result_path']).parent/'events').glob('[0-9]*.json'):
                event = json.loads(path.read_text())
                if (event.get('event') == 'judge_received'
                        and event['record'].get('response', {}).get('model') == 'o4-mini-2025-04-16'):
                    verified_judge = True
            if not verified_judge:
                raise ValueError('A valid DONE requires an actual pinned canonical judge response')
        super().finish(attempt_id, result)
        if verified_judge:
            with self.connect() as db:
                db.execute("INSERT INTO metadata VALUES('judge_transport_status','verified') ON CONFLICT(key) DO UPDATE SET value='verified'")

    def snapshot(self):
        result = super().snapshot()
        with self.connect() as db:
            row = db.execute("SELECT value FROM metadata WHERE key='judge_transport_status'").fetchone()
        result['judge_transport_status'] = row['value'] if row else 'pending_until_first_done'
        return result

    def reserve(self, kind, count=1, **context):
        self.require_approved_allocation()
        if kind not in KINDS or count != 1:
            raise ValueError('Exactly one bounded Jev/browser/helper/judge attempt is required')
        aid, condition = context.get('attempt_id'), context.get('condition')
        if not aid or condition != 'L06':
            raise ValueError('An owned Jev-only attempt is required')
        amount, bounds = 0., None
        if kind in RATES:
            bounds = request_bound(context['request'])
            _, ip, op = RATES[kind]
            amount = (bounds[0]*ip + bounds[1]*op)/1e6
        details = dict(attempt_id=aid, condition=condition)
        if context.get('request') is not None:
            details['request_sha256'] = digest(context['request'])
        if bounds:
            details.update(input_token_bound=bounds[0], output_token_bound=bounds[1])
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM metadata WHERE key='halt'").fetchone():
                raise RuntimeError('Jev cohort halted')
            owned = db.execute("SELECT q.condition FROM attempts a JOIN queue q USING(item_id) WHERE a.attempt_id=? AND a.status='active'", (aid,)).fetchone()
            if not owned or owned['condition'] != condition:
                raise ValueError('Request differs from active attempt')
            if self._counter(db, kind)+1 > self.plan['limits'][kind]:
                raise RuntimeError('All-attempt request cap exhausted: '+kind)
            if kind in RATES and self._counter(db, RATES[kind][0])+amount > self.plan['limits'][RATES[kind][0]]:
                raise RuntimeError('All-attempt dollar cap exhausted: '+RATES[kind][0])
            if kind == 'browser_sessions':
                if self._counter(db, 'active_browsers')+1 > self.plan['limits']['concurrent_browsers']:
                    raise RuntimeError('Concurrent local-browser cap exhausted')
                self._add(db, 'active_browsers', 1)
            rid = uuid.uuid4().hex
            status = 'consumed' if kind == 'jev_requests' else 'reserved'
            db.execute('INSERT INTO reservations VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                (rid, kind, 1, aid, condition, amount, amount, status, json.dumps(details), None, time.time()))
            self._add(db, kind, 1)
            if kind in RATES:
                self._add(db, RATES[kind][0], amount)
            self._event(db, 'reserve', dict(reservation_id=rid, kind=kind, reserved_usd=amount))
            db.commit()
            return dict(reservation_id=rid, kind=kind, reserved_usd=amount, **details)

    def settle(self, reservation, usage):
        rid = reservation['reservation_id']
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM reservations WHERE reservation_id=?', (rid,)).fetchone()
            if not row or row['status'] != 'reserved':
                raise ValueError('Reservation is not awaiting settlement')
            if row['kind'] == 'browser_sessions':
                if usage.get('closed') is not True:
                    raise ValueError('Verified browser closure required')
                self._add(db, 'active_browsers', -1)
                amount = 0.
            else:
                usage = usage.get('usage', usage) if isinstance(usage, dict) else {}
                it = usage.get('prompt_tokens', usage.get('input_tokens'))
                ot = usage.get('completion_tokens', usage.get('output_tokens'))
                bounds = json.loads(row['context_json'])
                if (type(it) is not int or type(ot) is not int or it < 0 or ot < 0
                        or it > bounds['input_token_bound'] or ot > bounds['output_token_bound']):
                    db.execute("INSERT OR IGNORE INTO metadata VALUES('halt',?)",
                        (json.dumps(dict(reason='unknown_usage', reservation_id=rid)),))
                    db.commit()
                    raise RuntimeError('Usage missing or exceeds reserved bound; full charge retained')
                cap, ip, op = RATES[row['kind']]
                amount = (it*ip+ot*op)/1e6
                self._add(db, cap, amount-row['reserved_usd'])
            db.execute("UPDATE reservations SET status='settled',charged_usd=?,usage_json=? WHERE reservation_id=?",
                       (amount, json.dumps(usage), rid))
            self._event(db, 'settle', dict(reservation_id=rid, charged_usd=amount))
            db.commit()

    def finalize_unknown(self, reservation, evidence_path):
        evidence = Path(evidence_path).resolve()
        if self.root not in evidence.parents or not evidence.is_file():
            raise ValueError('Private preserved failure evidence required')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rid = reservation['reservation_id']
            row = db.execute('SELECT * FROM reservations WHERE reservation_id=?', (rid,)).fetchone()
            if not row or row['status'] != 'reserved' or row['kind'] not in RATES:
                raise ValueError('Only a reserved priced request can retain an unknown full charge')
            proof = dict(evidence_path=str(evidence), evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
                         retained_full_reservation_usd=row['reserved_usd'])
            db.execute("UPDATE reservations SET status='charged_unknown',usage_json=? WHERE reservation_id=?",
                       (json.dumps(proof), rid))
            self._event(db, 'charged_unknown', proof)
            db.commit()

    def enable_primary(self):
        self.require_approved_allocation()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute("SELECT a.result_json,a.artifact_directory FROM queue q JOIN attempts a USING(item_id) WHERE q.category='smoke' AND a.status='finished'").fetchall()
            expected = sum(x['category'] == 'smoke' for x in self.plan['schedule'])
            if len(rows) != expected or any(json.loads(r['result_json']).get('valid') is not True for r in rows):
                raise RuntimeError('All three fresh Jev smoke tasks must finish valid')
            if self._counter(db, 'active_browsers') or db.execute("SELECT 1 FROM attempts WHERE status='active'").fetchone():
                raise RuntimeError('Smoke phase must be fully closed')
            accepted = set()
            for row in rows:
                for path in (Path(row['artifact_directory'])/'events').glob('[0-9]*.json'):
                    event = json.loads(path.read_text())
                    response = event.get('record', {}).get('response', {})
                    if event.get('event') == 'jev_received' and response.get('model') == 'jev-1.13.0': accepted.add('jev')
                    if event.get('event') == 'judge_received' and response.get('model') == 'o4-mini-2025-04-16': accepted.add('judge')
            # Native BLOCKED/limit episodes correctly make no judge request.
            # Acceptance tests infrastructure; it must not require actor DONE.
            completed = any(json.loads(r['result_json']).get('terminal') == 'done' for r in rows)
            if 'jev' not in accepted or completed and 'judge' not in accepted:
                raise RuntimeError('Smoke lacks a required actual pinned model response')
            db.execute("UPDATE metadata SET value='primary' WHERE key='phase'")
            self._event(db, 'enable_primary', dict(smoke_count=expected, responses=sorted(accepted),
                judge_transport_status='verified' if 'judge' in accepted else 'pending_until_first_done'))
            db.commit()
