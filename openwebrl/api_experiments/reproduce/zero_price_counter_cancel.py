"""Preserve zero-price counter cancellations without stopping unrelated tasks.

Only the exact, released Responses input-token-count request qualifies. Native
unknown records/call counts/zero reserves stay unchanged. Generation, judge,
positive reserves, malformed evidence and other failures use the original halt.
"""
from pathlib import Path
from decimal import Decimal
import copy, hashlib, json, math, os, socket, threading, time

CALLER_HALT = 'Physical Responses/count failure; preserve all calls, no unchanged retry'
CANCELLATIONS = ('TimeoutError', 'CancelledError')


def read_regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Regular native counter/lease evidence required')
    return json.loads(path.read_text())


def live_identity():
    return dict(pid=os.getpid(), thread_id=threading.get_ident(),
                node=socket.gethostname(),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def authenticated_zero_counter(row, physical_id, reason, attempt, role, current):
    """Pure predicate; a rejection always retains original unknown/halt behavior."""
    try:
        if role != 'model' or reason not in CANCELLATIONS:
            return False
        if (row['physical_id'] != physical_id or len(physical_id) != 32 or
                any(c not in '0123456789abcdef' for c in physical_id) or
                Decimal(str(row['reserved_usd'])) != 0 or
                Decimal(str(row['charged_or_reserved_usd'])) != 0):
            return False
        meta = row['metadata']
        if meta['role'] != 'input_token_count' or meta['model'] != 'gpt-6-luna':
            return False
        attempt = Path(attempt).resolve(strict=True)
        path = Path(meta['request_path'])
        resolved = path.resolve(strict=True)
        if (path.is_symlink() or resolved.parent.name != 'requests' or
                resolved.name != 'count-' + physical_id + '.json' or
                resolved.parent.parent.parent != attempt / 'episodes'):
            return False
        raw = read_regular(path)
        if not (raw['physical_id'] == physical_id and raw['role'] == 'input_token_count' and
                raw['endpoint'] == '/v1/responses/input_tokens' and
                raw['generation'] is False and raw['model'] == raw['provider'] == 'gpt-6-luna' and
                raw['status'] in ('reserved', 'usage_unknown') and
                raw['input_token_upper_bound'] == 0 and
                Decimal(str(raw['reserved_usd_decimal'])) == 0 and
                Decimal(str(raw['reserved_usd'])) == 0 and
                'response' not in raw and 'response_received_epoch' not in raw):
            return False
        digest = hashlib.sha256(json.dumps(raw['request'], sort_keys=True,
            ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        if digest != raw['request_sha256'] or digest != meta['request_sha256']:
            return False
        dispatch = raw['physical_http_dispatch_epoch']
        if type(dispatch) not in (int, float) or not math.isfinite(dispatch) or dispatch <= 0:
            return False
        gate = attempt / 'http-gate'
        identity = read_regular(gate / 'physical-identities' / (physical_id + '.json'))
        if identity['physical_id'] != physical_id or any(identity[k] != current[k]
                for k in ('pid', 'thread_id', 'node', 'boot_id')):
            return False
        token = identity['lease_token']
        if len(token) != 32 or any(c not in '0123456789abcdef' for c in token):
            return False
        lease = gate / 'leases' / token
        acquired, released, error = (read_regular(lease / n) for n in
                                     ('acquired.json', 'released.json', 'error.json'))
        slot = acquired['slot']
        return (type(slot) is int and 0 <= slot < 8 and acquired['identity'] == identity and
                acquired['slot_directory'] == str(gate / ('slot-%02d' % slot)) and
                acquired['owner_path'] == str(gate / ('slot-%02d' % slot) / ('owner-' + token + '.json')) and
                released['physical_id'] == error['physical_id'] == physical_id and
                released['lease_token'] == error['lease_token'] == token and
                released['slot'] == slot and released['node'] == identity['node'] and
                released['verified_owned_release'] is True and
                error['error_type'] == reason and error['slot_owned'] == acquired['slot_directory'] and
                identity['created_epoch'] <= acquired['acquired_epoch'] <= dispatch <=
                released['released_epoch'] <= time.time())
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError):
        return False


class CounterCancellationGuard:
    def __init__(self, budget, attempt):
        self.budget, self.attempt = budget, Path(attempt).resolve()
        self.local = threading.local()

    def unknown(self, original, physical_id, reason):
        self.local.marker = None
        budget = self.budget
        with budget.locked():
            state = budget._state()
            row = state['requests'].get(physical_id)
            if (row is not None and row['status'] == 'reserved' and
                    authenticated_zero_counter(row, physical_id, reason,
                        self.attempt, budget.carry_role, live_identity())):
                # Exactly the original native unknown transition/event. No refund,
                # reclassification, suppressed physical identity or new allowance.
                row.update(status='usage_unknown', unknown_epoch=time.time(), error_type=str(reason))
                budget._save(state, {'event': 'usage_unknown', **row})
                self.local.marker = (physical_id, copy.deepcopy(row), reason)
                return
        return original(physical_id, reason)

    def halt(self, original, reason):
        marker = getattr(self.local, 'marker', None)
        self.local.marker = None  # One immediately following caller halt, same thread.
        if marker is not None and reason == CALLER_HALT:
            physical_id, expected, cancellation = marker
            budget = self.budget
            with budget.locked():
                row = budget._state()['requests'].get(physical_id)
                if (row == expected and not budget.halted() and
                        authenticated_zero_counter(row, physical_id, cancellation,
                            self.attempt, budget.carry_role, live_identity())):
                    return
        return original(reason)


def actor_budget_class(original, output):
    """Compose the cleanup guard only onto this output's fresh actor pool.

    The original budget's caps, physical identities, journal, unknown rows and
    full holds remain authoritative. Judge/browser pools use their original type.
    """
    output = Path(output).resolve()
    expected = output / 'budget' / 'model'

    class ActorBudget(original):
        def __init__(self, root, *args, **kwargs):
            self._counter_cancel = None
            if Path(root).resolve() != expected:
                raise ValueError('Counter cleanup guard requires its own actor pool')
            super().__init__(root, *args, **kwargs)
            self.carry_role = 'model'
            self._counter_cancel = CounterCancellationGuard(self, output)

        def unknown(self, physical_id, reason):
            if self._counter_cancel is None:
                return super().unknown(physical_id, reason)
            return self._counter_cancel.unknown(super().unknown, physical_id, reason)

        def halt(self, reason):
            if self._counter_cancel is None:
                return super().halt(reason)
            return self._counter_cancel.halt(super().halt, reason)

    return ActorBudget
