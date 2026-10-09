"""Durable shared queue and all-attempt budgets for the controlled SFT suite.

SQLite DELETE journals (not WAL) keep cross-process transactions on shared
storage. No request is authorized before a committed reservation. Failures keep
their full reservation; restarting a worker or allocation never resets totals.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import sqlite3
import time
import uuid


KINDS = ('local_sft_generations', 'browser_sessions', 'luna_selector_requests',
         'jev_selector_requests', 'local_kev_requests', 'om2w_judge_http_attempts')
PRICED = {'luna_selector_requests': ('luna_usd', .125, .5),
          'om2w_judge_http_attempts': ('canonical_judge_usd', 1.1, 4.4)}
LIMIT_KEY = {'om2w_judge_http_attempts': 'canonical_judge_http_attempts'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_shard_cohort(plan, task_ids):
    """Require disjoint task shards with every condition on every assigned task.

    Each plan carries the same complete partition, bound by its approval hash.
    The two roots keep independent ledgers; no spare budget crosses roots.
    """
    shard = plan.get('shard')
    if shard is None:
        return
    partitions = plan.get('campaign_task_shards')
    arms = {'L01', 'L08', 'L10', 'L11'}
    if (not isinstance(shard, dict) or type(shard.get('index')) is not int
            or shard['index'] not in (0, 1) or shard.get('count') != 2
            or not isinstance(partitions, list) or len(partitions) != 2
            or len(task_ids) != 300 or len(set(task_ids)) != 300
            or set(plan.get('conditions', {})) != arms):
        raise ValueError('Expected the complete four-condition two-shard campaign')
    if any(not isinstance(group, list) or len(group) != 150 or len(set(group)) != 150
           for group in partitions):
        raise ValueError('Every task shard requires150 unique task identities')
    if set(partitions[0]) & set(partitions[1]) or set(partitions[0] + partitions[1]) != set(task_ids):
        raise ValueError('Task shards must be disjoint and cover the full300 task file')
    chosen = partitions[shard['index']]
    campaign = plan.get('campaign_id')
    if (shard.get('task_ids') != chosen or not isinstance(campaign, str) or not campaign
            or plan.get('experiment_id') != f"{campaign}-shard-{shard['index']}"):
        raise ValueError('Shard identity differs from its frozen campaign partition')
    rows = plan.get('schedule', [])
    if (plan.get('primary_episodes') != 600 or plan.get('smoke_episodes') != 12
            or len(rows) != 612 or len({r.get('item_id') for r in rows}) != 612
            or any(r.get('category') not in ('primary', 'smoke') for r in rows)):
        raise ValueError('Each shard requires600 primary items plus12 separate smokes')
    primary = Counter((r.get('task_id'), r.get('condition')) for r in rows if r['category'] == 'primary')
    if primary != Counter({(task_id, arm): 1 for task_id in chosen for arm in arms}):
        raise ValueError('Every assigned task must occur exactly once under all four conditions')
    smokes = [(r.get('task_id'), r.get('condition')) for r in rows if r['category'] == 'smoke']
    smoke_tasks = {task_id for task_id, _ in smokes}
    if (len(smoke_tasks) != 3 or not smoke_tasks.issubset(set(chosen))
            or Counter(smokes) != Counter({(task_id, arm): 1 for task_id in smoke_tasks for arm in arms})):
        raise ValueError('Each shard requires three shared smoke tasks under all four conditions')


def request_bound(request):
    images = 0
    def text_only(value):
        nonlocal images
        if isinstance(value, str) and value.startswith('data:image/'):
            images += 1
            return '[image omitted from text reserve]'
        if isinstance(value, list): return [text_only(x) for x in value]
        if isinstance(value, dict): return {k:text_only(v) for k,v in value.items()}
        return value
    text_bytes = len(json.dumps(text_only(request), ensure_ascii=False).encode())
    output = request.get('max_completion_tokens', request.get('max_output_tokens', request.get('max_tokens')))
    if type(output) is not int or not 0 < output <= 4096:
        raise ValueError('Priced request requires output cap at most 4096')
    # Byte-count text bound plus fixed conservative current-screenshot reserve.
    return text_bytes + images * 16384 + 2048, output


class SuiteState:
    def __init__(self, root, *, allow_unapproved_for_tests=False):
        self.root = Path(root).resolve()
        self.plan = json.loads((self.root/'plan.json').read_text())
        if Path(self.plan['root']).resolve()!=self.root:
            raise ValueError('Plan is bound to its one durable ledger root; copying it cannot reset caps')
        self.plan_hash = digest(self.plan)
        if self.plan.get('shard') is not None:
            task_ids = [json.loads(line)['metadata']['task_id'] for line in
                        Path(self.plan['task_file']).read_text().splitlines() if line.strip()]
            validate_shard_cohort(self.plan, task_ids)
        self.path = self.root/'state.sqlite3'
        self.allow_unapproved_for_tests = allow_unapproved_for_tests
        self._initialize()

    def require_approved_allocation(self):
        if self.allow_unapproved_for_tests: return
        approval = json.loads((self.root/'approval.json').read_text())
        job = os.environ.get('SLURM_JOB_ID')
        if (approval.get('approved') is not True or approval.get('plan_sha256') != self.plan_hash
                or approval.get('resources') != self.plan['resources'] or approval.get('limits') != self.plan['limits']
                or not job or not any(str(a['job_id']) == job for a in approval.get('attempts', []))
                or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text()):
            raise ValueError('Exact approved plan and registered active allocation required before external work')
        allocation=json.loads((self.root/'scheduler-attempts'/job/'allocation-validated.json').read_text())
        if (allocation.get('job_id')!=job or allocation.get('plan_sha256')!=self.plan_hash
                or allocation.get('resources')!=self.plan['resources']
                or allocation.get('limits')!=self.plan['limits']
                or time.time()>=allocation['work_deadline_unix']
                or allocation['prior_actual_seconds']+allocation['current_attempt_limit']>self.plan['resources']['total_seconds']):
            raise ValueError('A current controller-validated allocation and remaining budget are required')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA foreign_keys=ON')
        return db

    def _initialize(self):
        with self.connect() as db:
            db.executescript('''
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS queue(position INTEGER PRIMARY KEY,item_id TEXT UNIQUE NOT NULL,
 condition TEXT NOT NULL,task_id TEXT NOT NULL,category TEXT NOT NULL,task_json TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'pending');
CREATE TABLE IF NOT EXISTS attempts(attempt_id TEXT PRIMARY KEY,item_id TEXT NOT NULL,
 worker_id TEXT NOT NULL,job_id TEXT,started REAL NOT NULL,ended REAL,status TEXT NOT NULL,
 artifact_directory TEXT NOT NULL,result_json TEXT);
CREATE TABLE IF NOT EXISTS reservations(reservation_id TEXT PRIMARY KEY,kind TEXT NOT NULL,
 count INTEGER NOT NULL,attempt_id TEXT,condition TEXT,reserved_usd REAL NOT NULL,
 charged_usd REAL NOT NULL,status TEXT NOT NULL,context_json TEXT,usage_json TEXT,created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS counters(key TEXT PRIMARY KEY,value REAL NOT NULL);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,event TEXT NOT NULL,
 details_json TEXT NOT NULL,created REAL NOT NULL);
''')
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute("SELECT value FROM metadata WHERE key='plan_sha256'").fetchone()
            if existing and existing['value'] != self.plan_hash:
                db.rollback(); raise ValueError('Persistent queue belongs to another frozen plan')
            if not existing:
                db.execute('INSERT INTO metadata VALUES(?,?)', ('plan_sha256', self.plan_hash))
                db.execute('INSERT INTO metadata VALUES(?,?)', ('phase', 'smoke'))
                tasks = {r['metadata']['task_id']:r for r in
                         [json.loads(x) for x in Path(self.plan['task_file']).read_text().splitlines()]}
                for position, item in enumerate(self.plan['schedule']):
                    db.execute('INSERT INTO queue(position,item_id,condition,task_id,category,task_json) VALUES(?,?,?,?,?,?)',
                        (position,item['item_id'],item['condition'],item['task_id'],item.get('category','primary'),json.dumps(tasks[item['task_id']])))
                db.execute("INSERT INTO counters VALUES('active_browsers',0)")
            db.commit()

    @staticmethod
    def _event(db, name, details):
        db.execute('INSERT INTO events(event,details_json,created) VALUES(?,?,?)', (name,json.dumps(details),time.time()))

    @staticmethod
    def _counter(db, key):
        row = db.execute('SELECT value FROM counters WHERE key=?',(key,)).fetchone()
        return row['value'] if row else 0

    @staticmethod
    def _add(db, key, amount):
        db.execute('INSERT INTO counters(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=value+excluded.value',(key,amount))

    def claim(self, worker_id):
        self.require_approved_allocation()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM metadata WHERE key='halt'").fetchone():
                raise RuntimeError('Suite halted; inspect durable ledger')
            if db.execute("SELECT 1 FROM attempts WHERE worker_id=? AND status='active'",(worker_id,)).fetchone():
                raise RuntimeError('Worker already owns an unfinished attempt; no automatic replay')
            active = db.execute("SELECT COUNT(*) AS n FROM attempts WHERE status='active'").fetchone()['n']
            if active >= self.plan['resources']['collectors']:
                raise RuntimeError('Global collector concurrency exceeded')
            phase = db.execute("SELECT value FROM metadata WHERE key='phase'").fetchone()['value']
            row = db.execute("SELECT * FROM queue WHERE status='pending' AND category=? ORDER BY position LIMIT 1",(phase,)).fetchone()
            if row is None: return None
            aid = uuid.uuid4().hex
            directory = self.root/'attempts'/aid
            directory.mkdir(parents=True, exist_ok=False)
            db.execute("UPDATE queue SET status='active' WHERE item_id=?",(row['item_id'],))
            db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?)',
                (aid,row['item_id'],worker_id,os.getenv('SLURM_JOB_ID','offline-test' if self.allow_unapproved_for_tests else None),
                 time.time(),None,'active',str(directory),None))
            self._event(db,'claim',{'attempt_id':aid,'worker_id':worker_id,'item_id':row['item_id']})
            db.commit()
            return {'attempt_id':aid,'condition':row['condition'],'task_id':row['task_id'],
                    'task':json.loads(row['task_json']),'artifact_directory':str(directory),'category':row['category']}

    def reserve(self, kind, count=1, **context):
        self.require_approved_allocation()
        if kind not in KINDS or type(count) is not int or count <= 0:
            raise ValueError('Unknown reservation kind or invalid count')
        condition, aid = context.get('condition'), context.get('attempt_id')
        amount, bounds = 0., None
        if kind in PRICED:
            request = context.get('request')
            if not isinstance(request,dict): raise ValueError('Priced calls require exact request for reservation')
            bounds = request_bound(request)
            _, input_price, output_price = PRICED[kind]
            if kind == 'luna_selector_requests' and bounds[0] > 272000:
                input_price *= 2; output_price *= 1.5
            amount = count*(bounds[0]*input_price+bounds[1]*output_price)/1e6
        summary = {k:context[k] for k in ('condition','attempt_id','purpose','worker_id') if k in context}
        if 'request' in context: summary['request_sha256'] = digest(context['request'])
        if bounds: summary.update(input_token_bound=bounds[0],output_token_bound=bounds[1])
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM metadata WHERE key='halt'").fetchone(): raise RuntimeError('Suite halted')
            if aid:
                owned=db.execute("SELECT q.condition FROM attempts a JOIN queue q USING(item_id) WHERE a.attempt_id=? AND a.status='active'",(aid,)).fetchone()
                if not owned or condition is not None and condition!=owned['condition']:
                    raise ValueError('Reservation requires its active attempt and matching condition')
            if self._counter(db,kind)+count > self.plan['limits'][LIMIT_KEY.get(kind,kind)]: raise RuntimeError('All-attempt call cap exhausted: '+kind)
            if kind in PRICED:
                capkey = PRICED[kind][0]
                if self._counter(db,capkey)+amount > self.plan['limits'][capkey]: raise RuntimeError('All-attempt dollar cap exhausted: '+capkey)
            if kind == 'browser_sessions':
                if condition not in self.plan['conditions'] or not aid or count != 1:
                    raise ValueError('Each browser belongs to one condition and active attempt')
                if self._counter(db,'browsers:'+condition)+1 > self.plan['limits']['browser_sessions_per_condition']:
                    raise RuntimeError('Per-condition browser cap exhausted')
                if self._counter(db,'active_browsers')+1 > self.plan['limits']['concurrent_browsers']:
                    raise RuntimeError('Global browser concurrency exceeded')
                self._add(db,'browsers:'+condition,1); self._add(db,'active_browsers',1)
            rid = uuid.uuid4().hex
            status = 'reserved' if kind in PRICED or kind == 'browser_sessions' else 'consumed'
            db.execute('INSERT INTO reservations VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                (rid,kind,count,aid,condition,amount,amount,status,json.dumps(summary),None,time.time()))
            self._add(db,kind,count)
            if kind in PRICED: self._add(db,PRICED[kind][0],amount)
            self._event(db,'reserve',{'reservation_id':rid,'kind':kind,'count':count,'reserved_usd':amount})
            db.commit()
            return {'reservation_id':rid,'kind':kind,'count':count,'reserved_usd':amount,**summary}

    def settle(self, reservation, usage):
        rid = reservation['reservation_id'] if isinstance(reservation,dict) else reservation
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM reservations WHERE reservation_id=?',(rid,)).fetchone()
            if not row or row['status'] not in ('reserved','consumed'): raise ValueError('Unknown or already-settled reservation')
            if row['kind'] == 'browser_sessions':
                if not isinstance(usage,dict) or usage.get('closed') is not True:
                    raise ValueError('Browser reservation requires verified closure')
                self._add(db,'active_browsers',-1); amount=0.
            elif row['kind'] in PRICED:
                value = usage.get('usage', usage) if isinstance(usage,dict) else {}
                it = value.get('input_tokens',value.get('prompt_tokens'))
                ot = value.get('output_tokens',value.get('completion_tokens'))
                _, ip, op = PRICED[row['kind']]
                bounds = json.loads(row['context_json'])
                if (type(it) is not int or type(ot) is not int or it < 0 or ot < 0
                        or it > bounds['input_token_bound'] or ot > bounds['output_token_bound']):
                    db.execute("INSERT OR IGNORE INTO metadata VALUES('halt',?)",(json.dumps({'reason':'Missing usage or reservation exceeded','reservation_id':rid}),))
                    self._event(db,'unknown_usage',{'reservation_id':rid}); db.commit()
                    raise RuntimeError('Unknown usage or token-bound violation; reservation retained and suite halted')
                if row['kind'] == 'luna_selector_requests' and it > 272000:
                    ip *= 2; op *= 1.5
                amount = (it*ip+ot*op)/1e6
                self._add(db,PRICED[row['kind']][0],amount-row['reserved_usd'])
            else: amount=0.
            db.execute("UPDATE reservations SET status='settled',charged_usd=?,usage_json=? WHERE reservation_id=?",
                       (amount,json.dumps(usage),rid))
            self._event(db,'settle',{'reservation_id':rid,'charged_usd':amount}); db.commit()

    def finalize_unknown(self, reservation, evidence_path):
        rid=reservation['reservation_id'] if isinstance(reservation,dict) else reservation
        evidence=Path(evidence_path).resolve()
        if self.root not in evidence.parents or not evidence.is_file():
            raise ValueError('Unknown usage requires preserved private evidence in this suite')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM reservations WHERE reservation_id=?',(rid,)).fetchone()
            if not row or row['status']!='reserved' or row['kind'] not in PRICED:
                raise ValueError('Only an unresolved priced call may become charged_unknown')
            proof=dict(evidence_path=str(evidence),evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
                       retained_full_reservation_usd=row['reserved_usd'])
            db.execute("UPDATE reservations SET status='charged_unknown',usage_json=? WHERE reservation_id=?",(json.dumps(proof),rid))
            self._event(db,'charged_unknown',dict(reservation_id=rid,**proof));db.commit()

    def finish(self, attempt_id, result):
        data = json.dumps(result)
        if len(data.encode()) > 100000: raise ValueError('Commit artifact paths/hashes, not raw multimodal payloads')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            attempt = db.execute("SELECT * FROM attempts WHERE attempt_id=? AND status='active'",(attempt_id,)).fetchone()
            if not attempt: raise ValueError('Unknown active attempt')
            path = Path(result['result_path']).resolve()
            directory = Path(attempt['artifact_directory']).resolve()
            if directory not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != result['result_sha256']:
                raise ValueError('Result artifact must be preserved and hash-bound to this attempt')
            task = db.execute('SELECT * FROM queue WHERE item_id=?',(attempt['item_id'],)).fetchone()
            if result.get('task_id') != task['task_id'] or result.get('condition') != task['condition'] or result.get('attempt_id') != attempt_id:
                raise ValueError('Result identity differs from the queue claim')
            if db.execute("SELECT 1 FROM reservations WHERE attempt_id=? AND kind='browser_sessions' AND status='reserved'",(attempt_id,)).fetchone():
                raise RuntimeError('Cannot finish before browser closure is verified')
            db.execute("UPDATE attempts SET status='finished',ended=?,result_json=? WHERE attempt_id=?",(time.time(),data,attempt_id))
            db.execute("UPDATE queue SET status='finished' WHERE item_id=?",(attempt['item_id'],))
            self._event(db,'finish',{'attempt_id':attempt_id}); db.commit()

    def enable_primary(self):
        self.require_approved_allocation()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute("SELECT a.result_json,a.artifact_directory,q.condition FROM queue q JOIN attempts a ON a.item_id=q.item_id WHERE q.category='smoke' AND a.status='finished'").fetchall()
            expected = sum(x.get('category') == 'smoke' for x in self.plan['schedule'])
            if len(rows) != expected or any(json.loads(r['result_json']).get('valid') is not True for r in rows):
                raise RuntimeError('All smoke episodes must finish valid before full collection')
            if self._counter(db,'active_browsers') or db.execute("SELECT 1 FROM attempts WHERE status='active'").fetchone():
                raise RuntimeError('Smoke workers must finish and close browsers before phase handoff')
            accepted=set();proof=[]
            if self.plan.get('requires_jev_context_diagnostic'):
                diagnostic=self.root/'scheduler-attempts'/os.environ['SLURM_JOB_ID']/'jev-context-preflight.json'
                value=json.loads(diagnostic.read_text())
                if (value.get('completed') is not True or value.get('plan_sha256')!=self.plan_hash
                        or value.get('job_id')!=os.environ['SLURM_JOB_ID']
                        or value.get('expected_model')!='jev-1.13.0'
                        or value.get('state_bytes')!=262144
                        or value.get('outcome') not in ('accepted','known_capacity_rejection')
                        or value['outcome']=='accepted' and value.get('model_identity')!='jev-1.13.0'
                        or value['outcome']=='known_capacity_rejection' and
                            (value.get('http_status')!=400 or value.get('error_type')!='max_tokens_exceeded')):
                    raise RuntimeError('Jev capacity diagnostic needs confirmed accepted/rejected outcome')
                proof.append(dict(kind='jev_context_capacity',path=str(diagnostic),
                                  sha256=hashlib.sha256(diagnostic.read_bytes()).hexdigest(),outcome=value['outcome']))
            models={'L08':'gpt-6-luna','L10':'jev-1.13.0','L11':'kev-latest'}
            for row in rows:
                for path in (Path(row['artifact_directory'])/'events').glob('[0-9]*.json'):
                    event=json.loads(path.read_text());kind=None
                    if event.get('event')=='proposal_received':kind='actor:'+row['condition']
                    if event.get('event')=='selector_received' and event['record'].get('response',{}).get('model')==models.get(row['condition']):
                        kind='selector:'+row['condition']
                    if event.get('event')=='judge_received' and event['record'].get('response',{}).get('model')=='o4-mini-2025-04-16':
                        kind='canonical_judge'
                    if kind and kind not in accepted:
                        accepted.add(kind);proof.append(dict(kind=kind,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            required={'actor:'+x for x in self.plan['conditions']}|{'selector:'+x for x in models}|{'canonical_judge'}
            if not required.issubset(accepted):
                raise RuntimeError('Smoke acceptance lacks verified model/judge responses: '+','.join(sorted(required-accepted)))
            db.execute("UPDATE metadata SET value='primary' WHERE key='phase'")
            self._event(db,'enable_primary',{'verified_smoke_episodes':expected,'acceptance_evidence':proof}); db.commit()

    def mark_interrupted_after_teardown(self, *, teardown_verified, reason):
        """Owner-only closeout; never silently return an interrupted item to pending."""
        if teardown_verified is not True: raise ValueError('Owned processes must be verified stopped first')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            ids = [r['attempt_id'] for r in db.execute("SELECT attempt_id FROM attempts WHERE status='active'")]
            for aid in ids:
                item = db.execute('SELECT item_id FROM attempts WHERE attempt_id=?',(aid,)).fetchone()['item_id']
                browsers = db.execute("SELECT COUNT(*) AS n FROM reservations WHERE attempt_id=? AND kind='browser_sessions' AND status='reserved'",(aid,)).fetchone()['n']
                self._add(db,'active_browsers',-browsers)
                db.execute("UPDATE reservations SET status='closed_by_owner',usage_json=? WHERE attempt_id=? AND kind='browser_sessions' AND status='reserved'",(json.dumps({'closed':True,'owner_verified':True}),aid))
                db.execute("UPDATE attempts SET status='interrupted',ended=? WHERE attempt_id=?",(time.time(),aid))
                db.execute("UPDATE queue SET status='interrupted' WHERE item_id=?",(item,))
            self._event(db,'owner_teardown',{'attempt_ids':ids,'reason':reason}); db.commit()

    def snapshot(self):
        with self.connect() as db:
            return {'plan_sha256':self.plan_hash,
                'queue':{r['status']:r['n'] for r in db.execute('SELECT status,COUNT(*) AS n FROM queue GROUP BY status')},
                'counts_by_condition':{r['condition']:r['n'] for r in db.execute("SELECT condition,COUNT(*) AS n FROM queue WHERE status='finished' GROUP BY condition")},
                'counters':{r['key']:r['value'] for r in db.execute('SELECT * FROM counters')},
                'unsettled_reservations':db.execute("SELECT COUNT(*) AS n FROM reservations WHERE status='reserved'").fetchone()['n'],
                'phase':db.execute("SELECT value FROM metadata WHERE key='phase'").fetchone()['value'],
                'halt':next((json.loads(r['value']) for r in db.execute("SELECT value FROM metadata WHERE key='halt'")),None)}

    def halt(self, reason, **details):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            value=dict(reason=reason,**details)
            db.execute("INSERT OR IGNORE INTO metadata VALUES('halt',?)",(json.dumps(value),))
            self._event(db,'halt',value);db.commit()
