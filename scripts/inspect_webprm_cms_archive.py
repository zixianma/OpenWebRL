#!/usr/bin/env python3
"""Offline original-CMS identity gate; deliberately no downloader or image runner.

Preparation reports the outstanding acquisition/hosting gates. Inspection reads
an already acquired Docker-save archive without extracting or executing members.
The command requires a separately approved existing allocation and source freeze.
An internally consistent archive is not proof of its historical provenance or
of safe extraction, working authentication, or a correct database reset.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import signal
import stat
import tarfile
import time

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011')
PROPOSAL = RUNTIME / 'next-stage-preparation/cms-original-hosting-resource-proposal-v1.json'
PROPOSAL_SHA = '82c8cf5c9a18c354b495fa3fff8db3469683866eb67c9a28d68bf971f8e71988'
ARCHIVE_NAME = 'shopping_admin_final_0719.tar'
ARCHIVE_BYTES = 9640032256
IMAGE_TAG = 'shopping_admin_final_0719:latest'
WRITE_ROOTS = (Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'), Path('/tmp'))


class GateClosed(ValueError):
    pass


@dataclass(frozen=True)
class Limits:
    archive_bytes: int = ARCHIVE_BYTES
    metadata_bytes: int = 4 * 2**20
    outer_members: int = 4096
    layers: int = 256
    layer_members: int = 1000000
    cumulative_regular_bytes: int = 12 * 2**30
    reported_hazards: int = 64


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_path(value):
    path = Path(value)
    if (not path.is_absolute() or path.resolve() != path or
            any(p.is_symlink() for p in (path, *path.parents)) or
            not any(root in path.parents for root in WRITE_ROOTS)):
        raise GateClosed('Use a canonical nonsymlink path strictly below runtime or /tmp')
    return path


def check_time(deadline):
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or time.time() >= deadline:
        raise GateClosed('Offline inspection deadline exhausted')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GateClosed('Duplicate JSON key')
        result[key] = value
    return result


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


def member_name(name):
    if not isinstance(name, str) or not name or len(name) > 4096 or '\\' in name or '\x00' in name:
        raise GateClosed('Invalid archive member name')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        raise GateClosed('Archive member traversal or absolute path')
    normalized = str(path)
    if normalized == '.':
        return '.'
    return normalized


def stream_sha(stream, expected, deadline):
    checksum, total = hashlib.sha256(), 0
    while total < expected:
        check_time(deadline)
        block = stream.read(min(2**20, expected - total))
        if not block:
            raise GateClosed('Truncated archive or layer')
        total += len(block)
        checksum.update(block)
    if stream.read(1):
        raise GateClosed('Archive or layer exceeded its declared size')
    return checksum.hexdigest()


class BoundedReader:
    """Reject oversized extension-header reads before tarfile allocates them."""
    def __init__(self, stream, maximum, deadline):
        self.stream, self.maximum, self.deadline = stream, maximum, deadline

    def read(self, size=-1):
        check_time(self.deadline)
        if not 0 <= size <= self.maximum:
            raise GateClosed('Unbounded/oversized tar metadata read')
        return self.stream.read(size)

    def __getattr__(self, key):
        return getattr(self.stream, key)


def inspection(archive, *, deadline, limits=Limits()):
    """Pure offline reader. Tests inject smaller bounds; CLI always uses defaults."""
    archive = checked_path(archive)
    descriptor = os.open(archive, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != limits.archive_bytes:
            raise GateClosed('Exact recorded archive byte length required')
        archive_sha = stream_sha(source, limits.archive_bytes, deadline)
        source.seek(0)
        members = {}
        metadata_used = 0
        # r: rejects compressed outer archives. Docker-save layers are also raw
        # tar streams; no decompressor, extraction, mount or image code runs.
        with tarfile.open(fileobj=BoundedReader(source, limits.metadata_bytes, deadline), mode='r:') as outer:
            for entry in outer:
                check_time(deadline)
                name = member_name(entry.name)
                if name in members or len(members) >= limits.outer_members:
                    raise GateClosed('Duplicate outer path or outer-member cap')
                if entry.sparse or not (entry.isfile() or entry.isdir()):
                    raise GateClosed('Outer Docker-save members must be ordinary files/directories')
                if entry.size < 0 or entry.offset_data + entry.size > limits.archive_bytes:
                    raise GateClosed('Outer member exceeds archive boundary')
                members[name] = entry

            def read_metadata(name):
                nonlocal metadata_used
                name = member_name(name)
                entry = members.get(name)
                if not entry or not entry.isfile() or metadata_used + entry.size > limits.metadata_bytes:
                    raise GateClosed('Missing or oversized Docker metadata')
                metadata_used += entry.size
                check_time(deadline)
                raw = outer.extractfile(entry).read(entry.size + 1)
                if len(raw) != entry.size:
                    raise GateClosed('Truncated Docker metadata')
                return raw

            raw_manifest = read_metadata('manifest.json')
            manifest = decode(raw_manifest)
            if (not isinstance(manifest, list) or len(manifest) != 1 or
                    not isinstance(manifest[0], dict) or manifest[0].get('RepoTags') != [IMAGE_TAG]):
                raise GateClosed('Exactly the original CMS image/tag is required')
            image = manifest[0]
            config_name = image.get('Config', '')
            if not re.fullmatch(r'[0-9a-f]{64}\.json', config_name):
                raise GateClosed('Expected content-addressed Docker config filename')
            raw_config = read_metadata(config_name)
            if sha(raw_config) != config_name[:-5]:
                raise GateClosed('Docker config content digest mismatch')
            config = decode(raw_config)
            if config.get('os') != 'linux' or config.get('architecture') != 'amd64':
                raise GateClosed('Only the original linux/amd64 deployment is in scope')
            layers = image.get('Layers')
            diffs = config.get('rootfs', {}).get('diff_ids')
            if (not isinstance(layers, list) or not 0 < len(layers) <= limits.layers or
                    any(not isinstance(name, str) for name in layers) or len(set(layers)) != len(layers) or
                    not isinstance(diffs, list) or len(diffs) != len(layers) or
                    any(not isinstance(digest, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', digest) for digest in diffs)):
                raise GateClosed('Missing, duplicate or inconsistent ordered layers/diff_ids')
            records, hazards, counts = [], [], {'members': 0, 'regular_bytes': 0, 'whiteouts': 0, 'links': 0}

            def hazard(kind, layer, entry):
                if len(hazards) < limits.reported_hazards:
                    hazards.append(dict(kind=kind, layer=layer, path=entry.name))

            for number, (name, expected) in enumerate(zip(layers, diffs)):
                entry = members.get(member_name(name))
                if not entry or not entry.isfile():
                    raise GateClosed('Referenced layer missing or not a regular archive member')
                stream = outer.extractfile(entry)
                digest = stream_sha(stream, entry.size, deadline)
                if 'sha256:' + digest != expected:
                    raise GateClosed('Raw layer digest differs from config rootfs.diff_ids')
                stream.seek(0)
                layer_counts = dict(members=0, regular_bytes=0)
                with tarfile.open(fileobj=BoundedReader(stream, limits.metadata_bytes, deadline), mode='r:') as layer:
                    for item in layer:
                        check_time(deadline)
                        path = member_name(item.name)
                        counts['members'] += 1
                        layer_counts['members'] += 1
                        if counts['members'] > limits.layer_members:
                            raise GateClosed('Cumulative layer-member cap exceeded')
                        if item.sparse or any(key.startswith('GNU.sparse') for key in item.pax_headers):
                            raise GateClosed('Sparse layer entry requires separate bounded review')
                        if item.size < 0 or item.offset_data + item.size > entry.size:
                            raise GateClosed('Layer member exceeds raw layer boundary')
                        if item.isfile():
                            counts['regular_bytes'] += item.size
                            layer_counts['regular_bytes'] += item.size
                            if counts['regular_bytes'] > limits.cumulative_regular_bytes:
                                raise GateClosed('Conservative expanded-file bound exceeds original proposal')
                        elif item.issym() or item.islnk():
                            counts['links'] += 1
                            target = item.linkname
                            if len(target) > 4096 or '\x00' in target or '\\' in target:
                                raise GateClosed('Malformed layer link')
                            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path) if item.issym() else '', target))
                            if resolved == '..' or resolved.startswith('../'):
                                raise GateClosed('Layer link escapes image root')
                            hazard('link_requires_extractor_semantics_review', number, item)
                        elif not item.isdir():
                            hazard('special_file_requires_review', number, item)
                        if PurePosixPath(path).name.startswith('.wh.'):
                            counts['whiteouts'] += 1
                            hazard('whiteout_requires_oci_semantics_review', number, item)
                        # No layer-member lookup is needed later. Avoid retaining
                        # an entire million-member TarInfo inventory in memory.
                        layer.members.clear()
                records.append(dict(index=number, path=name, bytes=entry.size,
                                    sha256=digest, diff_id=expected, **layer_counts))
        after = os.fstat(source.fileno())
        current = archive.lstat()
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if identity(before) != identity(after) or identity(after) != identity(current):
            raise GateClosed('Archive changed or was replaced during inspection')
    return dict(status='offline_archive_identity_consistent', archive_path=str(archive),
        archive_sha256=archive_sha, archive_bytes=before.st_size, image_tag=IMAGE_TAG,
        manifest_sha256=sha(raw_manifest), config_sha256=sha(raw_config), layers=records,
        cumulative_layer_counts=counts, hazards=hazards,
        hazards_truncated=len(hazards) == limits.reported_hazards,
        originality='Internal consistency only; official acquisition provenance and historical identity remain separate gates.',
        ready_to_import=False, ready_to_host=False, extraction_performed=False,
        network_calls=0, subprocesses=0, model_provider_calls=0,
        startup_config=dict(entrypoint=config.get('config', {}).get('Entrypoint'),
                            command=config.get('config', {}).get('Cmd'), user=config.get('config', {}).get('User')))


def approved_deadline(approval, proposal, environ=None):
    env = os.environ if environ is None else environ
    if sha(Path(proposal).read_bytes()) != PROPOSAL_SHA:
        raise GateClosed('Original resource draft changed; new review required')
    approved = decode(Path(approval).read_bytes())
    now = time.time()
    start, end = approved.get('allocation_started_epoch'), approved.get('deadline_epoch')
    prior = approved.get('prior_consumed_seconds')
    if (approved.get('status') != 'approved' or approved.get('proposal_sha256') != PROPOSAL_SHA or
            approved.get('script_sha256') != sha(Path(__file__).read_bytes()) or
            not isinstance(approved.get('exact_user_approval'), str) or not approved['exact_user_approval'].strip() or
            not env.get('SLURM_JOB_ID') or approved.get('job_id') != env['SLURM_JOB_ID'] or
            env.get('SLURM_CPUS_PER_TASK') != '4' or env.get('SLURM_MEM_PER_NODE') != '16384' or
            any(env.get(key, '').strip() not in ('', '0') for key in
                ('SLURM_GPUS', 'SLURM_GPUS_ON_NODE', 'SLURM_GPUS_PER_NODE')) or
            any(env.get(key, '').strip() for key in ('SLURM_JOB_GPUS', 'SLURM_STEP_GPUS')) or
            any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, end, prior)) or
            prior < 0 or not start <= now < end or end - start + prior > 7200):
        raise GateClosed('Exact source-bound approval/existing CPU allocation/remaining deadline required')
    # Preserve the original proposal's 600-second closure/retention reserve.
    check_time(end - 600)
    return end - 600


def write_new(path, value):
    path = checked_path(path)
    raw = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    if len(raw) > 2**20:
        raise GateClosed('Inspection receipt exceeded its one-MiB write cap')
    if not path.parent.is_dir():
        raise GateClosed('Create and review the evidence directory before inspection')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--approval', type=Path, required=True)
    parser.add_argument('--proposal', type=Path, default=PROPOSAL)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    deadline = approved_deadline(args.approval, args.proposal)
    archive, output = checked_path(args.archive), checked_path(args.output)
    if archive.name != ARCHIVE_NAME or output.exists():
        raise GateClosed('Original archive filename and fresh receipt path required')
    def expired(*_):
        raise GateClosed('Hard inspection deadline reached')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.time()))
    try:
        result = inspection(archive, deadline=deadline)
        result.update(proposal_sha256=PROPOSAL_SHA, script_sha256=sha(Path(__file__).read_bytes()),
                      allocation_job_id=os.environ['SLURM_JOB_ID'])
        write_new(output, result)
        print(json.dumps(dict(status=result['status'], ready_to_import=False,
                              ready_to_host=False, receipt=str(output))))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    main()
