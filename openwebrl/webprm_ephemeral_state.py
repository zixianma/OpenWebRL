"""Dispose only newly claimed benchmark states after durable evidence audit.

Original bases and historical states are never eligible. The campaign owner
must verify its approved policy and actual process closure; this helper checks
their bindings and artifact identities, not the scientific audit itself.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat

PROTOCOL = 'audited_new_campaign_state_disposal_v1'
REQUIRED_ARTIFACTS = {'rollout', 'verdict', 'api', 'provenance', 'worker_plan',
                      'audit', 'closure', 'terminal', 'evaluator_inputs'}
ROOTS = (Path('/tmp'), Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _path(value):
    path = Path(value)
    if (not path.is_absolute() or path.resolve() != path or
            not any(root in path.parents for root in ROOTS) or
            any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError('Canonical nonsymlink project-runtime or temporary path required')
    return path


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _has_mount(state):
    # A bind-mounted descendant can share st_dev with the parent, so device
    # comparisons alone cannot safely exclude it before recursive deletion.
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        mount = Path(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), line.split()[4]))
        if mount == state or state in mount.parents:
            return True
    return False


def _write_once(path, value):
    path = _path(path)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    _sync_directory(path.parent)


def claim_state(claim_path, state_path, *, scratch_root, campaign_root, cell_id,
                base_path, policy_sha256):
    """Record an exclusive fresh-state claim, leaving copy destination absent."""
    scratch, campaign = _path(scratch_root), _path(campaign_root)
    claim_path, state, base = _path(claim_path), _path(state_path), _path(base_path)
    if not isinstance(cell_id, str) or not cell_id or not _sha(policy_sha256):
        raise ValueError('Fixed cell and approved policy identity required')
    for root in (scratch, campaign):
        info = root.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError('Owned private scratch and campaign roots required')
    expected = scratch / 'states' / hashlib.sha256(cell_id.encode()).hexdigest()
    if (scratch == campaign or scratch in campaign.parents or campaign in scratch.parents or
            state != expected or state.exists() or campaign not in claim_path.parents or
            base == state or state in base.parents or base in state.parents or
            state in claim_path.parents or not base.is_dir()):
        raise ValueError('Only the absent fixed-cell state outside its immutable base is eligible')
    value = dict(protocol=PROTOCOL, cell_id=cell_id, state_path=str(state),
                 scratch_root=str(scratch), campaign_root=str(campaign), base_path=str(base),
                 policy_sha256=policy_sha256, destination_absent_at_claim=True,
                 preserve_historical_states=True)
    _write_once(claim_path, value)
    return value


def durable_artifacts(campaign_root, artifacts):
    """Hash and fsync the closed evidence; include every saved input in artifacts."""
    campaign = _path(campaign_root)
    if not isinstance(artifacts, dict) or not REQUIRED_ARTIFACTS <= artifacts.keys():
        raise ValueError('Complete terminal evidence, audit and closure artifacts required')
    result, directories = {}, set()
    for name, value in artifacts.items():
        if not isinstance(name, str) or not name:
            raise ValueError('Named artifact required')
        path = _path(value)
        if campaign not in path.parents:
            raise ValueError('Evidence must remain inside the durable campaign')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('Evidence must be a regular file')
            checksum = hashlib.sha256()
            for chunk in iter(lambda: stream.read(2**20), b''):
                checksum.update(chunk)
            os.fsync(stream.fileno())
            after = os.fstat(stream.fileno())
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if any(getattr(before, key) != getattr(after, key) for key in fields):
            raise ValueError('Evidence changed while hashing')
        result[name] = dict(path=str(path), sha256=checksum.hexdigest(), bytes=after.st_size)
        directories.add(path.parent)
    for directory in directories:
        _sync_directory(directory)
    return result


def dispose_state(claim_path, *, approved_policy_sha256, artifact_manifest,
                  audit, closure, intent_path, receipt_path):
    """Delete a fresh state only after independently verified evidence closure.

    The durable intent precedes deletion. A crash may leave a partially removed
    state; resume is allowed only with the same claim, evidence and root inode.
    No decision depends on success/failure, and incomplete evidence retains state.
    """
    claim_path = _path(claim_path)
    claim = json.loads(claim_path.read_text())
    if (claim.get('protocol') != PROTOCOL or not _sha(approved_policy_sha256) or
            claim.get('policy_sha256') != approved_policy_sha256 or
            claim.get('destination_absent_at_claim') is not True or
            claim.get('preserve_historical_states') is not True):
        raise ValueError('Fresh claim and explicitly approved disposal policy required')
    campaign, scratch = _path(claim['campaign_root']), _path(claim['scratch_root'])
    state, base = _path(claim['state_path']), _path(claim['base_path'])
    expected = scratch / 'states' / hashlib.sha256(claim['cell_id'].encode()).hexdigest()
    if (scratch == campaign or scratch in campaign.parents or campaign in scratch.parents or
            state != expected or campaign not in claim_path.parents or state == base or
            state in base.parents or base in state.parents or state in campaign.parents or campaign == state):
        raise ValueError('State disposition escaped its fresh claim')
    intent_path, receipt_path = _path(intent_path), _path(receipt_path)
    if (intent_path == receipt_path or any(campaign not in p.parents or state in p.parents
                                         for p in (intent_path, receipt_path))):
        raise ValueError('Disposal evidence must survive inside the durable campaign')
    if (not isinstance(artifact_manifest, dict) or not REQUIRED_ARTIFACTS <= artifact_manifest.keys() or
            audit.get('verified') is not True or audit.get('cell_id') != claim['cell_id'] or
            audit.get('artifact_manifest_sha256') != digest(artifact_manifest) or
            audit.get('audit_artifact_sha256') != artifact_manifest.get('audit', {}).get('sha256') or
            closure.get('worker_remaining_processes') != {} or
            closure.get('service_remaining_processes') != {} or
            type(closure.get('broker_cell_inflight')) is not int or closure['broker_cell_inflight'] != 0):
        raise ValueError('Independent artifact audit and empty actual process/request closure required')
    actual = durable_artifacts(campaign, {k: v['path'] for k, v in artifact_manifest.items()})
    if actual != artifact_manifest:
        raise ValueError('Audited evidence changed before state disposal')
    common = dict(protocol=PROTOCOL, claim_sha256=digest(claim),
                  artifact_manifest_sha256=digest(actual), audit_sha256=digest(audit),
                  closure_sha256=digest(closure), state_path=str(state),
                  audited_artifacts=actual, audit_binding=audit, closure=closure)
    if intent_path.exists():
        intent = json.loads(intent_path.read_text())
        if any(intent.get(k) != v for k, v in common.items()):
            raise ValueError('Disposal intent changed across recovery')
    else:
        info = state.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or _has_mount(state):
            raise ValueError('Owned, non-mounted state directory required')
        intent = dict(common, state_device=info.st_dev, state_inode=info.st_ino)
        _write_once(intent_path, intent)
    expected_receipt = dict(protocol=PROTOCOL, status='fresh_state_disposed_after_durable_audit',
                   intent_sha256=digest(intent), artifact_manifest_sha256=digest(actual),
                   cell_id=claim['cell_id'], state_absent=True, historical_states_untouched=True)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt != expected_receipt or state.exists():
            raise ValueError('Completed disposition changed')
        return receipt
    if state.exists():
        info = state.lstat()
        if (info.st_dev != intent['state_device'] or info.st_ino != intent['state_inode'] or
                not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or _has_mount(state) or
                not shutil.rmtree.avoids_symlink_attacks):
            raise ValueError('State root identity or safe deletion capability changed')
        shutil.rmtree(state)
        _sync_directory(state.parent)
    _write_once(receipt_path, expected_receipt)
    return expected_receipt
