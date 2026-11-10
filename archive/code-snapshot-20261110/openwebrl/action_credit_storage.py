"""Shared GPFS admission; no allocation, deletion, cleanup or automatic release.

This uses the *same* lock and active-reservation map as the two outcome-to60
controllers. Their reviewed implementation sums every active reserved_bytes.
Reserve only a pilot that actually asks for admission, never five ceilings.
The explicit measured forecast must cover all retained outputs through update10,
including retries, pending tensors, native checkpoints and evaluations. It is
held conservatively across stages/restarts: bytes already written are also in
quota usage, so no implicit credit/refund is taken. A later owner must reconcile
completion and awaited writers before releasing it; this module cannot release.

Pinned measurement/owned-root inventories are trusted operator inputs, not proof
of future sizes or exhaustive filesystem ownership. Unknown/uncoordinated writers
cannot be made safe by this ledger; the shared two-TiB floor remains mandatory.
"""
from __future__ import annotations

import fcntl
import getpass
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from scripts.check_storage_quota import parse_quota
from .action_credit_native_controller import _directory, _hash, _json, _write
from .action_credit_state import CampaignStore, _safe_path

TIB = 1024**4
LEGACY_ROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/training-controls/outcome56-uniform8-to60-20261008')
LEGACY_SOURCE_SHA256 = 'a70832620461f1cd378057a39182ae8ae551ef2a61d5cd0e7372d6fd019f7d2f'
QUOTA_SOURCE_SHA256 = 'b2fcfef09602c94d422683cddc447c9916b78d08f6a8ce15359331e73603b977'
STAGES = ('native-startup', 'initial', *(f'optimizer-update-{i}' for i in range(10)))
SCOPE = 'remaining_pilot_through_iteration10_all_retained_no_cleanup'


def require(condition, message):
    if not condition: raise ValueError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, 'Invalid '+label)
    return value


def _replace(path, value):
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    _write(temporary, value)
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)


class StorageAdmissionProvider:
    """``provider(store, stage) -> immutable {path,sha256}``, bootstrap compatible.

    forecast_ref JSON: schema_version=1, manifest_sha256, coordination_root,
    resource_accounting_ref (prepared accounting.json), owned_roots, scope=SCOPE,
    and stages containing EXACT STAGES. Each stage is a nonempty list of
    {measurement_ref, measurement_key, remaining_units, stress_numerator,
    stress_denominator}. Measurement JSON has ``measurements:{key:bytes}``,
    ``method`` and nonempty ``source_artifacts:[{path,sha256}]``; source artifacts
    are checked, and forecast bytes are derived, never accepted as a magic total.
    Every stage forecast covers the WHOLE remaining endpoint, not only one step.
    The largest is held from native-startup before any collection can begin.

    The coordination root is the current shared legacy root and must contain
    the byte-identical reviewed reservation module, existing lock/ledger and
    approved storage-plan.json. Unit tests patch this constant into a temporary
    replay directory; the runtime has no alternate uncoordinated-ledger option.
    """
    def __init__(self, *, forecast_ref, output_root, job_id,
                 coordination_root=LEGACY_ROOT, timeout_seconds=30,
                 max_quota_age_seconds=60, lock_timeout_seconds=30):
        require(str(job_id).isdigit(), 'Exact parent scheduler job ID required')
        for value in (timeout_seconds, max_quota_age_seconds, lock_timeout_seconds):
            require(type(value) in (int,float) and 0 < value <= 60, 'Finite bounded timeouts required')
        self.forecast_ref, self.output_root = dict(forecast_ref), _safe_path(output_root)
        self.coordination_root = _safe_path(coordination_root)
        require(self.coordination_root == _safe_path(LEGACY_ROOT), 'Must use the existing global coordination root')
        self.job_id = str(job_id)
        self.timeout, self.max_age, self.lock_timeout = timeout_seconds, max_quota_age_seconds, lock_timeout_seconds

    def _forecast(self, store):
        forecast = _json(self.forecast_ref)
        require(forecast.get('schema_version') == 1 and forecast.get('manifest_sha256') == store.manifest.sha256,
                'Forecast belongs to another manifest')
        require(forecast.get('coordination_root') == str(self.coordination_root), 'Forecast coordination root differs')
        require(forecast.get('scope') == SCOPE, 'Whole-pilot retained-output forecast required')
        accounting = _json(forecast['resource_accounting_ref'])
        require(accounting['campaign_id'] == store.manifest.to_dict()['campaign_id'] and
                accounting.get('cleanup_authorized') is False, 'Prepared storage accounting identity differs')
        storage = accounting['storage']
        cap = integer(storage['protective_ceiling_bytes'], 'protective ceiling', 1)
        floor = integer(storage['global_free_floor_bytes'], 'shared free floor', 2*TIB)
        require(cap <= 5*TIB, 'Pilot ceiling exceeds separately proposed five TiB')
        roots = [_safe_path(p) for p in forecast['owned_roots']]
        require(roots and len(set(roots)) == len(roots), 'Complete distinct owned roots required')
        require(all(p.is_dir() for p in roots), 'Owned roots must exist before admission')
        require(any(store.root == p or p in store.root.parents for p in roots), 'Owned roots omit campaign ledger')
        require(not any(a in b.parents for a in roots for b in roots if a != b), 'Owned roots overlap')
        require(set(forecast['stages']) == set(STAGES), 'Every bootstrap stage must have an explicit endpoint forecast')
        peaks = {}
        for stage, terms in forecast['stages'].items():
            require(isinstance(terms,list) and terms, 'Empty forecast derivation')
            total = 0
            for term in terms:
                measurement = _json(term['measurement_ref'])
                require(isinstance(measurement.get('method'),str) and measurement['method'].strip()
                        and measurement.get('source_artifacts'), 'Measured source evidence required')
                for ref in measurement['source_artifacts']:
                    require(_hash(ref['path']) == ref['sha256'], 'Measurement source changed')
                size = integer(measurement['measurements'][term['measurement_key']], 'measured bytes', 1)
                units = integer(term['remaining_units'], 'remaining units', 1)
                num = integer(term['stress_numerator'], 'stress numerator', 1)
                den = integer(term['stress_denominator'], 'stress denominator', 1)
                require(num >= den, 'Forecast cannot discount measured storage')
                total += (size*units*num+den-1)//den
            peaks[stage] = total
        require(max(peaks.values()) <= cap, 'Measured forecast exceeds pilot storage ceiling')
        return forecast, roots, cap, floor, peaks

    def _query(self, command, folder, label):
        started = time.time()
        result = subprocess.run(command,capture_output=True,text=True,timeout=self.timeout,check=False)
        ref = _write(folder/(label+'.json'), dict(command=command,started_unix=started,
            finished_unix=time.time(),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        require(result.returncode == 0, 'Storage query failed: '+label)
        return result.stdout, started, ref

    def __call__(self, store, stage):
        require(isinstance(store,CampaignStore), 'Existing campaign ledger required')
        require(stage in STAGES, 'Unknown storage stage')
        forecast, roots, cap, floor, peaks = self._forecast(store)
        root = self.coordination_root
        require(_hash(root/'storage_reservations.py') == LEGACY_SOURCE_SHA256,
                'Legacy reservation implementation changed; compatibility review required')
        parser_path = Path(__file__).resolve().parents[1]/'scripts/check_storage_quota.py'
        require(_hash(parser_path) == QUOTA_SOURCE_SHA256, 'Authoritative quota parser changed')
        lock_path, ledger_path = _safe_path(root/'storage-reservations.lock'), _safe_path(root/'storage-reservations.json')
        require(lock_path.is_file() and ledger_path.is_file(), 'Existing shared lock/ledger required; never create a parallel ledger')
        folder = _directory(self.output_root/uuid.uuid4().hex)
        with lock_path.open('r') as lock:
            deadline = time.monotonic()+self.lock_timeout
            while True:
                try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); break
                except BlockingIOError:
                    require(time.monotonic() < deadline, 'Shared storage lock timeout')
                    time.sleep(.05)
            plan = json.loads(_safe_path(root/'storage-plan.json').read_text())
            require(all(plan.get(k) is True for k in ('approved','preparation_verified','verified_sufficient','active_workers_excluded')),
                    'Existing shared storage plan is not verified')
            floor = max(floor, integer(plan['minimum_shared_free_bytes'],'legacy floor',2*TIB))
            before = ledger_path.read_bytes()
            data = json.loads(before)
            require(isinstance(data.get('active'),dict) and isinstance(data.get('history'),list), 'Malformed shared reservation ledger')
            key = 'action-credit:'+store.manifest.to_dict()['campaign_id']
            previous = data['active'].get(key)
            growth = max(peaks.values())
            if previous:
                require(previous.get('manifest_sha256') == store.manifest.sha256 and
                        previous.get('forecast_ref') == self.forecast_ref and previous.get('store_root') == str(store.root),
                        'Existing pilot reservation differs; explicit reconciliation required')
                require(previous['reserved_bytes'] == growth, 'Held forecast was changed')
            held = sum(integer(v['reserved_bytes'],'active reservation',1) for k,v in data['active'].items() if k != key)
            # Measure apparent and allocated bytes: neither alone is always conservative.
            commands = [['du','-sx','-B1',*flags,'--',*map(str,roots)] for flags in ([],['--apparent-size'])]
            sizes, size_refs = [], []
            for i, command in enumerate(commands):
                text, _, ref = self._query(command,folder,'owned-usage-'+str(i)); size_refs.append(ref)
                values = {}
                for line in text.splitlines():
                    count, path = line.split('\t',1)
                    require(path not in values and path in set(map(str,roots)), 'Unexpected du root inventory')
                    values[path] = integer(int(count),'owned bytes')
                require(set(values) == set(map(str,roots)), 'Incomplete du root inventory')
                sizes.append(values)
            retained = sum(max(values[str(p)] for values in sizes) for p in roots)
            projected = previous['projected_pilot_bytes'] if previous else retained+growth
            require(retained <= projected <= cap, 'Actual retained bytes or measured endpoint exceeds pilot ceiling')
            username = getpass.getuser()
            raw, observed, quota_ref = self._query(['/usr/lpp/mmfs/bin/mmlsquota','-u',username,'-Y','mmfs1:scrubbed'],folder,'quota')
            quota = parse_quota(raw,username)
            require(quota['soft_limit_bytes'] > 0 and all(quota[k] >= 0 for k in
                    ('used_bytes','in_doubt_bytes','hard_limit_bytes')), 'Finite soft quota and nonnegative usage required')
            require(0 <= time.time()-observed <= self.max_age, 'Quota evidence is stale or from the future')
            minimum = held+growth+floor
            require(quota['safe_headroom_bytes'] >= minimum, 'Soft-quota headroom cannot cover all reservations and shared floor')
            record = dict(previous or {}, token=previous['token'] if previous else uuid.uuid4().hex,
                lineage=key,manifest_sha256=store.manifest.sha256,store_root=str(store.root),
                forecast_ref=self.forecast_ref,reserved_bytes=growth,projected_pilot_bytes=projected,
                job_ids=sorted(set((previous or {}).get('job_ids',[])+[self.job_id])),
                pid=os.getpid(),epoch=time.time(),stage=stage,quota=quota)
            evidence = dict(schema_version=1,manifest_sha256=store.manifest.sha256,stage=stage,
                coordination_root=str(root),reservation=record,quota=quota,quota_evidence=quota_ref,
                usage_evidence=size_refs,retained_bytes=retained,other_held_bytes=held,
                required_headroom_bytes=minimum,global_floor_bytes=floor,protective_ceiling_bytes=cap,
                passed=True,cleanup_authorized=False,files_deleted=0,automatic_release=False)
            proposal = _write(folder/'admission-proposal.json',evidence)
            data['active'][key] = record
            data['history'].append(dict(event='action_credit_storage_admitted',evidence=proposal,**record))
            _replace(ledger_path,data)
            # If final publication fails, the reservation remains held for reconciliation.
            return _write(folder/'admission.json',dict(evidence,shared_ledger_sha256=_hash(ledger_path)))
