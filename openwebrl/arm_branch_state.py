"""Branch queue over the existing transactional, all-attempt reservation ledger."""
import json
import os
import time
import uuid
from pathlib import Path
from openwebrl.controlled_sft_state import SuiteState, digest
from openwebrl.arm_branch_worker import sha, write


class BranchState(SuiteState):
    def claim_item(self, item_id, worker_id):
        self.require_approved_allocation()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM metadata WHERE key='halt'").fetchone():
                raise RuntimeError('Inspect the durable halt before any more work')
            active = db.execute("SELECT COUNT(*) AS n FROM attempts WHERE status='active'").fetchone()['n']
            if active >= self.plan['resources']['collectors']:
                raise RuntimeError('Collector cap exceeded')
            row = db.execute("SELECT * FROM queue WHERE item_id=? AND status='pending'", (item_id,)).fetchone()
            if row is None:
                raise ValueError('An item must be pending; no silent attempt replacement')
            item = next(x for x in self.plan['schedule'] if x['item_id'] == item_id)
            aid = uuid.uuid4().hex
            directory = self.root/'attempts'/aid
            directory.mkdir(parents=True, exist_ok=False)
            db.execute("UPDATE queue SET status='active' WHERE item_id=?", (item_id,))
            db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?)', (aid, item_id,
                worker_id, os.getenv('SLURM_JOB_ID', 'offline-test'), time.time(), None,
                'active', str(directory), None))
            self._event(db, 'claim', dict(attempt_id=aid, item_id=item_id, worker_id=worker_id))
            db.commit()
        return dict(attempt_id=aid, condition=row['condition'], task_id=row['task_id'],
            task=json.loads(row['task_json']), artifact_directory=str(directory),
            category=row['category'], item=item)

    def item(self, item_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM queue WHERE item_id=?', (item_id,)).fetchone()
            attempts = [dict(x) for x in db.execute('SELECT * FROM attempts WHERE item_id=? ORDER BY started', (item_id,))]
        return dict(row), attempts

    def release(self, state_id, anchor, attempts):
        self.require_approved_allocation()
        repetitions = self.plan.get('continuation_repetitions', 3)
        candidates = self.plan.get('candidate_count', 3)
        count = candidates * repetitions
        if len(attempts) != count or len(set(attempts)) != count:
            raise ValueError('Every candidate and repetition must reach the barrier')
        identities, proofs = set(), []
        for directory in map(Path, attempts):
            value = json.loads((directory/'ready.json').read_text())
            if (value['anchor_sha256'] != anchor or value['attempt_directory'] != str(directory)
                    or not value['checks'] or not all(c['passed'] for c in value['checks'])):
                raise ValueError('Invalid reconstruction receipt')
            identities.add((value['candidate'], value['repeat']))
            proofs.append(dict(path=str(directory/'ready.json'), sha256=sha(directory/'ready.json')))
        if identities != {(c, r) for c in range(candidates) for r in range(repetitions)}:
            raise ValueError('The complete crossed branch design is required')
        target = self.root/'states'/state_id/'release.json'
        write(target, dict(anchor_sha256=anchor, attempt_directories=attempts,
            ready_receipts=proofs, released_unix=time.time(), cohort_committed_before_candidate_dispatch=True))
        return sha(target)
