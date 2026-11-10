"""Fresh, read-only Slurm evidence for the collection-admission auditor.

The owner supplies a SHA-pinned, complete allocation-attempt ledger. Every
call queries every listed parent job and retains raw output, including failed
query evidence. This does not authorize a job, alter its time limit, reset an
approval, or grant API spend. GPU elapsed is allocation wall time multiplied
by eight, separate from per-browser and per-HTTP campaign accounting.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time
import uuid

from .action_credit_acceptance import audit_allocation
from .action_credit_native_controller import _directory, _hash, _json, _write
from .action_credit_state import CampaignManifest, _safe_path


def _raw(path, text):
    path = _safe_path(path)
    raw = text if isinstance(text, bytes) else text.encode()
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return dict(path=str(path), sha256=_hash(path))


def _records(text):
    rows = {}
    for line in text.splitlines():
        if not line.strip(): continue
        values = line.split('|')
        if len(values) == 5 and values[-1] == '': values.pop()
        if values[0] == 'JobIDRaw': continue
        if len(values) != 4 or not values[0].isdigit() or values[0] in rows:
            raise ValueError('Unexpected, duplicate or non-parent scheduler record')
        job, state, elapsed, tres = values
        state = state.split()[0].rstrip('+') if state.strip() else ''
        if not elapsed.isdigit() or state not in ('PENDING','RUNNING','COMPLETED','FAILED','CANCELLED',
                'TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED'):
            raise ValueError('Unrecognized scheduler elapsed/state')
        resources = dict(item.split('=',1) for item in tres.split(',') if '=' in item)
        if state == 'PENDING':
            if int(elapsed): raise ValueError('Pending allocation has nonzero elapsed time')
        elif state == 'CANCELLED' and int(elapsed) == 0 and not tres.strip():
            # A never-started canceled job has no allocation. Eight is only
            # the approved cost rate, multiplied by zero, not allocated GPUs.
            pass
        elif resources.get('gres/gpu') != '8':
            raise ValueError('Allocation does not contain the approved eight GPUs')
        rows[job] = dict(job_id=job,gpus=8,elapsed_seconds=int(elapsed),state=state)
    return rows


class AllocationAccountingProvider:
    """Callable owner dependency for ``CollectionAcceptanceAuditor``.

    The attempt ledger has ``manifest_sha256``,
    ``approved_gpu_seconds_cap`` and ``attempts:[{job_id:...}, ...]``.
    Supply a new immutable ledger reference when adding an authorized retry;
    preserve prior attempts. Missing scheduler records fail closed rather than
    silently erasing their consumed time. Pending rows may lack AllocTRES but
    must have zero elapsed. A canceled job with zero elapsed and empty AllocTRES
    is likewise unallocated: its gpus=8 records the approved cost rate times
    zero, not an actual allocation. Allocated rows require eight actual GPUs.
    """
    def __init__(self, *, manifest, attempt_ledger_ref, output_root, timeout_seconds=30):
        if not isinstance(manifest,CampaignManifest):
            raise ValueError('Typed frozen campaign manifest required')
        if type(timeout_seconds) not in (int,float) or not 0 < timeout_seconds <= 60:
            raise ValueError('Scheduler query must have a finite timeout of at most 60 seconds')
        self.manifest, self.attempt_ledger_ref = manifest, dict(attempt_ledger_ref)
        self.output_root = _safe_path(output_root)
        self.timeout_seconds = timeout_seconds
        self._ledger()

    def _ledger(self):
        value = _json(self.attempt_ledger_ref)
        cap = self.manifest.to_dict()['budget_caps']['gpu_seconds']
        if value.get('manifest_sha256') != self.manifest.sha256 or value.get('approved_gpu_seconds_cap') != cap:
            raise ValueError('Attempt ledger manifest or separately approved GPU cap differs')
        ids = [str(row['job_id']) for row in value['attempts']]
        if not ids or len(ids) != len(set(ids)) or any(not key.isdigit() for key in ids):
            raise ValueError('Complete unique parent allocation IDs required')
        return ids, cap

    def __call__(self):
        ids, cap = self._ledger()
        folder = _directory(self.output_root/uuid.uuid4().hex)
        command = ['sacct','-n','-P','-X','--jobs='+','.join(ids),
                   '--format=JobIDRaw,State,ElapsedRaw,AllocTRES']
        # Capture before the query so a live-allocation upper bound includes
        # query latency and the later saved-artifact audit, never undercounts it.
        observed = time.time()
        _write(folder/'request.json',dict(command=command,observed_unix=observed,
            manifest_sha256=self.manifest.sha256,attempt_ledger=self.attempt_ledger_ref,
            timeout_seconds=self.timeout_seconds))
        try:
            process = subprocess.run(command,capture_output=True,text=True,timeout=self.timeout_seconds,check=False)
        except (subprocess.TimeoutExpired,OSError) as exc:
            stdout = _raw(folder/'sacct.stdout',getattr(exc,'stdout',None) or '')
            stderr = _raw(folder/'sacct.stderr',getattr(exc,'stderr',None) or '')
            _write(folder/'query-failure.json',dict(error_type=type(exc).__name__,stdout=stdout,stderr=stderr))
            raise RuntimeError('Scheduler query failed; retained evidence cannot admit a collection') from exc
        stdout = _raw(folder/'sacct.stdout',process.stdout)
        stderr = _raw(folder/'sacct.stderr',process.stderr)
        _write(folder/'query-result.json',dict(returncode=process.returncode,stdout=stdout,stderr=stderr))
        if process.returncode != 0:
            raise RuntimeError('Scheduler query returned nonzero; no valid accounting snapshot produced')
        rows = _records(process.stdout)
        if set(rows) != set(ids):
            raise ValueError('Scheduler evidence omits or adds a preserved allocation attempt')
        used = sum(row['gpus']*row['elapsed_seconds'] for row in rows.values())
        value = dict(schema_version=1,manifest_sha256=self.manifest.sha256,
            observed_unix=observed,approved_gpu_seconds_cap=cap,attempt_ledger=self.attempt_ledger_ref,
            scheduler_records=[stdout],attempts=[rows[key] for key in ids],gpu_seconds_used=used)
        ref = _write(folder/'accounting.json',value)
        audit_allocation(ref,self.manifest)
        return ref
