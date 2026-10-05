"""Durable, shared reservations for standalone selection evaluations."""
import fcntl
import json
import os
from pathlib import Path
import time

from openwebrl.decision_selection import write_json


class SelectionBudget:
    def __init__(self, root, limits):
        self.root, self.limits = Path(root), dict(limits)
        self.root.mkdir(parents=True, exist_ok=True)

    def reserve(self, kind, count=1, **context):
        if kind not in self.limits or type(count) is not int or count < 1:
            raise ValueError('Invalid budget reservation')
        with (self.root / 'lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.root / 'usage.json'
            state = json.loads(path.read_text()) if path.exists() else {
                'limits': self.limits, 'reserved': {}, 'sequence': 0}
            if state['limits'] != self.limits:
                raise ValueError('Existing budget limits differ; approval cannot reset usage')
            total = state['reserved'].get(kind, 0) + count
            if total > self.limits[kind]:
                raise RuntimeError(f'Approved {kind} budget exhausted')
            state['reserved'][kind] = total
            state['sequence'] += 1
            state['last_reservation'] = dict(kind=kind, count=count, **context,
                unix=time.time(), sequence=state['sequence'])
            write_json(path, state)
            with path.open('rb') as saved:
                os.fsync(saved.fileno())
            # Reservations remain charged if work fails or the process dies.
            with (self.root / 'reservations.jsonl').open('a') as log:
                log.write(json.dumps(state['last_reservation']) + '\n')
                log.flush(); os.fsync(log.fileno())
            return total
