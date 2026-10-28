"""Conservative OCI materializer for a fresh, private CMS validation root.

No image code runs. All layers are first hashed and their final metadata overlay
validated. Only then are final regular files, directories and internal links
created. Unsupported tar semantics fail closed; no historical path is removed.
"""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import posixpath
import stat
import tarfile
import time


class Closed(ValueError):
    pass


@dataclass(frozen=True)
class Limits:
    raw_bytes: int = 12 * 2**30
    final_bytes: int = 12 * 2**30
    members: int = 1000000
    metadata_read: int = 2 * 2**20
    # 'reject' (CMS default). 'fifo_and_skip_devices': create FIFOs (e.g. runit
    # supervise/control in the GitLab image, probe 357930) and record, but never
    # create, char/block device members (unprivileged mknod is impossible).
    special_members: str = 'reject'


def check(deadline):
    if not isinstance(deadline, (int, float)) or not time.time() < deadline < float('inf'):
        raise Closed('Materialization deadline exhausted')


def name(value, *, link=False, parent=''):
    if not isinstance(value, str) or not value or len(value) > 4096 or '\\' in value or '\0' in value:
        raise Closed('Invalid OCI path')
    if link:
        value = value.lstrip('/') if value.startswith('/') else posixpath.join(parent, value)
        value = posixpath.normpath(value)
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts:
        raise Closed('OCI path escapes container root')
    return str(p)


def private_path(path):
    path = Path(path)
    allowed = (Path('/tmp'), Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'))
    if (not path.is_absolute() or path.resolve() != path or
            any(p.is_symlink() for p in (path, *path.parents)) or
            not any(root in path.parents for root in allowed)):
        raise Closed('Canonical private runtime/scratch path required')
    return path


def put(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as out:
        out.write(raw); out.flush(); os.fsync(out.fileno())
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


class HashReader:
    def __init__(self, source, cap, deadline, hash_bytes=True):
        self.source, self.cap, self.deadline = source, cap, deadline
        self.sha, self.used = hashlib.sha256() if hash_bytes else None, 0

    def read(self, size=-1):
        check(self.deadline)
        if not 0 <= size <= 2 * 2**20:
            raise Closed('Unbounded layer/metadata read')
        data = self.source.read(size)
        self.used += len(data)
        if self.used > self.cap:
            raise Closed('Layer byte cap')
        if self.sha is not None:
            self.sha.update(data)
        return data


class MetadataReader:
    def __init__(self, source, cap):
        self.source, self.cap = source, cap

    def read(self, size=-1):
        if not 0 <= size <= self.cap:
            raise Closed('Oversized tar extension header')
        return self.source.read(size)

    def __getattr__(self, key):
        return getattr(self.source, key)


def layer_stream(blob, row, deadline, callback, limits, *, hash_layers=True):
    """Own descriptors, verify exact compressed/raw hashes and lengths per pass."""
    blob = private_path(blob)
    with os.fdopen(os.open(blob, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != row['compressed_bytes']:
            raise Closed('Blob file identity differs')
        compressed = HashReader(source, row['compressed_bytes'], deadline, hash_layers)
        with gzip.GzipFile(fileobj=compressed, mode='rb') as unzip:
            raw = HashReader(unzip, row['raw_bytes'], deadline, hash_layers)
            with tarfile.open(fileobj=raw, mode='r|', bufsize=65536) as archive:
                archive.fileobj = MetadataReader(archive.fileobj, limits.metadata_read)
                for index, item in enumerate(archive):
                    check(deadline)
                    callback(archive, index, item)
                    archive.members.clear()
            while raw.read(2**20):
                pass
        after = os.fstat(source.fileno())
    identity = lambda st:(st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    if (identity(before) != identity(after) or identity(after) != identity(blob.stat()) or
            compressed.used != row['compressed_bytes'] or raw.used != row['raw_bytes'] or
            hash_layers and (compressed.sha.hexdigest() != row['compressed_sha256'] or
            raw.sha.hexdigest() != row['raw_diff_id'].removeprefix('sha256:'))):
        raise Closed('Layer content/identity differs')


def overlay(layers, deadline, limits=Limits(), *, hash_layers=True):
    if not layers or len(layers) > 256 or sum(r['raw_bytes'] for r in layers) > limits.raw_bytes:
        raise Closed('Aggregate layer cap')
    entries, count = {}, 0
    for number, layer in enumerate(layers):
        current, deletions = {}, []
        def record(archive, index, item):
            nonlocal count
            count += 1
            if count > limits.members:
                raise Closed('OCI member cap')
            p = name(item.name)
            if p == '.':
                if not item.isdir():
                    raise Closed('Root entry is not a directory')
                return
            if item.sparse or item.size < 0:
                raise Closed('Sparse/negative member unsupported')
            if (not math.isfinite(float(item.mtime)) or any(k not in
                    {'path','linkpath','mtime','atime','ctime','uid','gid','uname','gname','size'} for k in item.pax_headers)):
                raise Closed('Unsupported extended metadata')
            leaf, parent = posixpath.basename(p), posixpath.dirname(p)
            if leaf.startswith('.wh.'):
                if not item.isfile() or item.size:
                    raise Closed('Unsupported whiteout representation')
                deleted = parent if leaf == '.wh..wh..opq' else posixpath.join(parent, leaf[4:])
                if leaf != '.wh..wh..opq' and leaf[4:] in ('', '.', '..'):
                    raise Closed('Invalid whiteout')
                deletions.append((deleted, leaf == '.wh..wh..opq'))
                return
            if p in current:
                raise Closed('Duplicate member path within layer')
            kind = ('file' if item.isfile() else 'dir' if item.isdir() else
                    'symlink' if item.issym() else 'hardlink' if item.islnk() else None)
            if kind is None and limits.special_members == 'fifo_and_skip_devices':
                kind = 'fifo' if item.isfifo() else 'device' if (item.ischr() or item.isblk()) else None
            if kind is None:
                raise Closed('Special device/FIFO/socket member unsupported')
            row = dict(layer=number, index=index, kind=kind, bytes=item.size,
                       mode=item.mode & (0o3777 if kind == 'dir' else 0o1777),
                       original_mode=item.mode, uid=item.uid, gid=item.gid,
                       mtime=float(item.mtime))
            if kind in ('symlink', 'hardlink'):
                row['linkname'] = item.linkname
                row['target'] = name(item.linkname, link=True, parent=parent if kind == 'symlink' else '')
            current[p] = row
        layer_stream(layer['path'], layer, deadline, record, limits, hash_layers=hash_layers)
        lower_paths = sorted(entries)
        def descendants(prefix):
            if not prefix:
                return lower_paths
            return lower_paths[bisect_left(lower_paths, prefix + '/'):bisect_left(lower_paths, prefix + '0')]
        for target, opaque in deletions:
            for p in descendants(target):
                entries.pop(p, None)
            if not opaque:
                entries.pop(target, None)
        for p, row in current.items():
            if row['kind'] != 'dir':
                for old in descendants(p):
                    entries.pop(old, None)
            entries[p] = row
        # Preserve the layer-time hardlink identity. A later overwrite of its
        # target must not silently redirect the old inode to new contents.
        for p,row in current.items():
            if row['kind'] != 'hardlink':
                continue
            visited, target = {p}, row['target']
            while target in entries and entries[target]['kind'] == 'hardlink':
                if target in visited:
                    raise Closed('Hardlink cycle')
                visited.add(target); target = entries[target]['target']
            if target not in entries or entries[target]['kind'] != 'file':
                raise Closed('Hardlink target absent in introducing layer')
            row['introduced_target'] = [entries[target]['layer'], entries[target]['index']]
    for p, row in entries.items():
        for parent in PurePosixPath(p).parents:
            if str(parent) == '.':
                break
            if str(parent) in entries and entries[str(parent)]['kind'] != 'dir':
                raise Closed('Extraction through a non-directory parent is unsupported: '+p)
        if row['kind'] == 'hardlink':
            visited, target = {p}, row['target']
            while target in entries and entries[target]['kind'] == 'hardlink':
                if target in visited:
                    raise Closed('Hardlink cycle')
                visited.add(target); target = entries[target]['target']
            if target not in entries or entries[target]['kind'] != 'file':
                raise Closed('Hardlink must resolve to a final regular file')
            if row['introduced_target'] != [entries[target]['layer'], entries[target]['index']]:
                raise Closed('Historical hardlink with overwritten target unsupported: '+p)
            row['regular_target'] = target
            row['bytes'] = entries[target]['bytes']
        if row['kind'] == 'symlink':
            # Lexically internal symlinks can still escape through another link.
            parts, seen, depth = row['target'].split('/'), set(), 0
            while parts:
                changed = False
                for i in range(1, len(parts)+1):
                    prefix = '/'.join(parts[:i]); found = entries.get(prefix)
                    if found and found['kind'] == 'symlink':
                        state = tuple(parts)
                        if state in seen or depth >= 64:
                            raise Closed('Symlink cycle/depth limit')
                        seen.add(state); depth += 1
                        parts = (found['target'].split('/') if found['target'] != '.' else []) + parts[i:]
                        changed = True; break
                if not changed:
                    break
    if sum(r['bytes'] for r in entries.values() if r['kind'] in ('file','hardlink')) > limits.final_bytes:
        raise Closed('Final image apparent-byte cap')
    return entries, count


def materialize(layers, destination, evidence, deadline, limits=Limits()):
    """Internal API: approved owner gates this call; tests use tiny fixtures only."""
    destination, evidence = private_path(destination), private_path(evidence)
    if (destination.exists() or evidence.exists() or destination in evidence.parents or
            evidence in destination.parents or destination == evidence):
        raise Closed('Fresh disjoint output paths required')
    entries, count = overlay(layers, deadline, limits)
    # This point is reached only after every compressed/raw identity and overlay
    # path has passed. No tarfile.extract(), image command or shell is used.
    destination.mkdir(mode=0o700); evidence.mkdir(mode=0o700)
    hashes = {}
    directories = {str(parent) for p in entries for parent in PurePosixPath(p).parents if str(parent) != '.'}
    directories |= {p for p,r in entries.items() if r['kind'] == 'dir'}
    for p in sorted(directories, key=lambda x:(x.count('/'), x)):
        check(deadline); (destination / p).mkdir(mode=0o700, exist_ok=True)
    for number, layer in enumerate(layers):
        def write(archive, index, item):
            p = name(item.name); row = entries.get(p)
            if not row or row['kind'] != 'file' or (row['layer'],row['index']) != (number,index):
                return
            data = archive.extractfile(item); h = hashlib.sha256(); remaining = row['bytes']
            with os.fdopen(os.open(destination/p, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600), 'wb') as out:
                while remaining:
                    check(deadline); block = data.read(min(2**20, remaining))
                    if not block:
                        raise Closed('Truncated selected final file')
                    out.write(block); h.update(block); remaining -= len(block)
                out.flush(); os.fsync(out.fileno())
            os.chmod(destination/p, row['mode'], follow_symlinks=False)
            os.utime(destination/p, (row['mtime'],row['mtime']), follow_symlinks=False)
            hashes[p] = h.hexdigest()
        layer_stream(layer['path'], layer, deadline, write, limits)
    for p,row in entries.items():
        check(deadline)
        if row['kind'] == 'hardlink':
            os.link(destination/row['regular_target'], destination/p, follow_symlinks=False)
        elif row['kind'] == 'fifo':
            os.mkfifo(destination/p, 0o600)
            os.chmod(destination/p, row['mode'], follow_symlinks=False)
        elif row['kind'] == 'symlink':
            # Preserve absolute container links. They are installed LAST and
            # never traversed by any host-side writer or receipt hasher.
            os.symlink(row['linkname'], destination/p)
    for p in sorted(directories, key=lambda x:(-x.count('/'),x)):
        os.chmod(destination/p, entries.get(p, {}).get('mode', 0o755), follow_symlinks=False)
        if p in entries:
            os.utime(destination/p, (entries[p]['mtime'],entries[p]['mtime']), follow_symlinks=False)
    manifest = dict(schema_version=1, source='third_party_digest_pinned_oci_candidate',
        original_state_equivalence_verified=False, destination=str(destination),
        members=count, entries=entries, regular_file_sha256=hashes,
        final_regular_bytes=sum(r['bytes'] for r in entries.values() if r['kind']=='file'),
        skipped_device_members=sorted(p for p,r in entries.items() if r['kind']=='device'),
        created_fifos=sorted(p for p,r in entries.items() if r['kind']=='fifo'),
        removed_setid_bits=[p for p,r in entries.items() if r['original_mode'] & ~r['mode'] & 0o6000],
        uid_gid_policy='Header IDs recorded; host-owned files, native fakeroot supplies service identities.',
        image_executed=False, full_hash_passes=2)
    reference = put(evidence/'materialization.json', manifest)
    return dict(status='materialized_private_candidate', manifest=reference,
                final_regular_bytes=manifest['final_regular_bytes'], entries=len(entries))
