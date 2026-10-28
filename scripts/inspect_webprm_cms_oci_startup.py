#!/usr/bin/env python3
"""Bounded, offline selected-text inventory of the acquired third-party CMS OCI.

This is a reader, not an extractor or image runner. Existing acquisition hashes
are reused; only small evidence/source files and selected contents are hashed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inspect_webprm_cms_archive import BoundedReader, GateClosed, checked_path, check_time, member_name, write_new
from scripts.inspect_webprm_original_startup import chosen, MAGENTO_ROOTS

N = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation')
PACKAGE = N / 'cms-thirdparty-oci-acquisition-v1'
OUTPUT = N / 'cms-thirdparty-startup-inventory-v1'
AUDIT_SHA = 'c7c501fb4a620ea4799fea6178df023dfb9801e3bf84fbad8fd464ed32d9b327'
CAPS = dict(seconds=180, raw_bytes=10 * 2**30, compressed_bytes=3 * 2**30,
            members=1000000, selected_file_bytes=2**20, selected_total_bytes=12 * 2**20,
            selected_versions=1024, metadata_read_bytes=2 * 2**20)
EXTRA = {'lib/apk/db/installed', 'var/www/html/app/etc/env.php',
         'var/www/html/composer.json', 'var/www/magento2/composer.json',
         'usr/local/etc/php-fpm.conf', 'etc/redis.conf', 'etc/init.d/mariadb'}
PREFIXES = ('usr/local/etc/php-fpm.d/', 'usr/share/java/elasticsearch/config/')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def selected(name):
    return chosen(name, {'docker-entrypoint.sh'}) or name in EXTRA or (
        any(name.startswith(p) for p in PREFIXES) and name.endswith(('.conf', '.yml', '.options', '.properties')))


class CountedReader:
    def __init__(self, stream, budget, expected, deadline, caps):
        self.stream, self.budget, self.expected = stream, budget, expected
        self.deadline, self.caps, self.used = deadline, caps, 0

    def read(self, size=-1):
        check_time(self.deadline)
        if not 0 <= size <= self.caps['metadata_read_bytes']:
            raise GateClosed('Unbounded decompression read')
        data = self.stream.read(size)
        self.used += len(data)
        self.budget['raw_bytes'] += len(data)
        if self.used > self.expected or self.budget['raw_bytes'] > self.caps['raw_bytes']:
            raise GateClosed('Decompression byte cap')
        return data


def apply_layer(files, links, new_files, new_links, deletions):
    # OCI whiteouts affect the lower layer, even when tar order puts a whiteout
    # after a new file in this same layer. No filesystem deletion occurs here.
    for prefix, opaque in deletions:
        for mapping in (files, links):
            for name in list(mapping):
                if (opaque and not prefix or name.startswith(prefix + '/') or
                        not opaque and name == prefix):
                    mapping.pop(name)
    for name, row in new_files.items():
        files[name] = row
        links.pop(name, None)
    for name, row in new_links.items():
        links[name] = row
        files.pop(name, None)


def scan_layer(stream, number, expected, state, deadline, caps=CAPS):
    reader = CountedReader(stream, state['budget'], expected, deadline, caps)
    new_files, new_links, deletions = {}, {}, []
    count = 0
    with tarfile.open(fileobj=reader, mode='r|', bufsize=65536) as archive:
        archive.fileobj = BoundedReader(archive.fileobj, caps['metadata_read_bytes'], deadline)
        for item in archive:
            check_time(deadline)
            count += 1
            state['budget']['members'] += 1
            if state['budget']['members'] > caps['members']:
                raise GateClosed('Tar member cap')
            name = member_name(item.name)
            if item.sparse or item.size < 0:
                raise GateClosed('Sparse/negative-size member unsupported')
            parts = PurePosixPath(name).parts
            if parts and parts[-1].startswith('.wh.'):
                parent = '/'.join(parts[:-1]); target = parts[-1][4:]
                deletions.append((parent, True) if target == '.wh..opq' else
                                 ((parent + '/' if parent else '') + target, False))
            elif item.isfile():
                for root, row in state['mutable_roots'].items():
                    if name == root or name.startswith(root + '/'):
                        row['cumulative_layer_files'] += 1
                        row['cumulative_layer_bytes'] += item.size
                if selected(name):
                    if (item.size > caps['selected_file_bytes'] or
                            state['budget']['selected_bytes'] + item.size > caps['selected_total_bytes'] or
                            state['budget']['selected_versions'] >= caps['selected_versions']):
                        raise GateClosed('Selected text cap')
                    raw = archive.extractfile(item).read(item.size + 1)
                    if len(raw) != item.size or b'\0' in raw:
                        raise GateClosed('Selected member is not ordinary text')
                    raw.decode('utf-8')
                    state['budget']['selected_bytes'] += item.size
                    state['budget']['selected_versions'] += 1
                    new_files[name] = dict(layer=number, bytes=item.size,
                                           sha256=digest(raw), mode=item.mode, raw=raw)
            elif item.issym() or item.islnk():
                if len(state['links']) + len(new_links) > 20000:
                    raise GateClosed('Link metadata cap')
                new_links[name] = dict(layer=number, target=item.linkname,
                                       type='symlink' if item.issym() else 'hardlink')
            elif selected(name) and not item.isdir():
                raise GateClosed('Selected member has special device/FIFO semantics')
            archive.members.clear()
    # Consume bounded padding to verify complete gzip framing/CRC and byte count,
    # without computing another compressed-layer or raw-layer SHA.
    while reader.read(2**20):
        pass
    if reader.used != expected:
        raise GateClosed('Raw layer size differs from acquisition receipt')
    apply_layer(state['files'], state['links'], new_files, new_links, deletions)
    return dict(index=number, members=count, raw_bytes_read=reader.used,
                selected_versions=len(new_files), whiteouts=len(deletions))


def new_state():
    roots = MAGENTO_ROOTS + ('var/www/html/pub/media', 'var/www/html/var', 'var/www/html/generated', 'var/www/html/app/etc')
    return dict(files={}, links={}, budget=dict(raw_bytes=0, members=0, selected_bytes=0, selected_versions=0),
                mutable_roots={r:dict(cumulative_layer_files=0, cumulative_layer_bytes=0) for r in roots})


def main():
    os.umask(0o077)
    start = time.time(); deadline = start + CAPS['seconds']
    def timeout(*_):
        raise GateClosed('Hard offline inspection deadline')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(CAPS['seconds'])
    audit_path = PACKAGE / 'independent-final-audit-356009.json'
    audit_raw = checked_path(audit_path).read_bytes()
    if digest(audit_raw) != AUDIT_SHA:
        raise GateClosed('Independent acquisition audit identity changed')
    paths = dict(manifest=PACKAGE / 'metadata/manifest.json', config=PACKAGE / 'metadata/config.json',
                 identity=PACKAGE / 'startup-01/attempt-356009/identity-proof.json')
    raw = {k:checked_path(v).read_bytes() for k,v in paths.items()}
    data = {k:json.loads(v) for k,v in raw.items()}
    proof = data['identity']; manifest = data['manifest']; config = data['config']
    if (digest(raw['identity']) != '79a680cfb544431ba14cc01e27b8ef9803f5d0a072e3779e883e3fb931d9792e' or
            digest(raw['manifest']) != proof['manifest_sha256'] or digest(raw['config']) != proof['config_sha256'] or
            not proof['all_raw_diff_ids_verified'] or proof['original_state_identity_verified']):
        raise GateClosed('Exact third-party acquisition proof required')
    layers = manifest['layers']; expected = {x['compressed_sha256']:x for x in proof['unique_layers']}
    if (len(layers) != 31 or [expected[x['digest'][7:]]['raw_diff_id'] for x in layers] != config['rootfs']['diff_ids'] or
            sum(x['size'] for x in layers) > CAPS['compressed_bytes'] or proof['total_raw_bytes'] > CAPS['raw_bytes']):
        raise GateClosed('Layer identities or aggregate byte caps differ')
    state = new_state(); rows = []
    for index, layer in enumerate(layers):
        sha = layer['digest'][7:]
        blob = checked_path(PACKAGE / 'startup-01/attempt-356009/oci/blobs/sha256' / sha)
        with os.fdopen(os.open(blob, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != layer['size']:
                raise GateClosed('Compressed blob size/type changed')
            with gzip.GzipFile(fileobj=source, mode='rb') as stream:
                row = scan_layer(stream, index, expected[sha]['raw_bytes'], state, deadline)
            after = os.fstat(source.fileno())
        identity = lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if identity(before) != identity(after) or identity(after) != identity(blob.stat()):
            raise GateClosed('Blob changed during read')
        rows.append(dict(row, compressed_sha256=sha, compressed_bytes=layer['size'],
                         raw_diff_id=expected[sha]['raw_diff_id']))
    check_time(deadline)
    output = checked_path(OUTPUT); output.mkdir(mode=0o700, exist_ok=False)
    private = output / 'private-selected-text'; private.mkdir(mode=0o700)
    files = state.pop('files'); links = state.pop('links')
    unresolved = {p:r for p,r in links.items() if selected(p) or any(n.startswith(p + '/') for n in files)}
    for row in files.values():
        target = private / (row['sha256'] + '.txt')
        if target.exists():
            continue
        with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as out:
            out.write(row['raw']); out.flush(); os.fsync(out.fileno())
    result = dict(schema='cms_thirdparty_selected_startup_inventory_v1', status='selected_text_inventory_complete',
        source='third_party_digest_pinned_oci_candidate', original_state_equivalence_verified=False,
        acquisition_audit=dict(path=str(audit_path),sha256=AUDIT_SHA),
        metadata={k:dict(path=str(paths[k]),sha256=digest(raw[k])) for k in paths}, caps=CAPS,
        elapsed_seconds=time.time()-start, scanned_layers=rows, **state,
        startup_config={k:config['config'].get(k) for k in ('Entrypoint','Cmd','User','WorkingDir','Volumes','ExposedPorts','StopSignal')},
        selected_files={name:{k:v for k,v in row.items() if k!='raw'} for name,row in files.items()},
        unresolved_selected_or_ancestor_links=unresolved, private_text_directory=str(private),
        scope='All ordered layer headers traversed; selected regular UTF-8 content only. Whiteouts applied to selected evidence. Other contents discarded. Mutable counts sum layers, not final rootfs size. Not an extraction-safety audit.',
        identity_scope='Existing independent compressed/raw identity receipts reused; file stat identities checked around reads and gzip CRC/length checked. No repeated full-layer SHA256 computation.',
        rootfs_extracted=False, image_executed=False, browser_calls=0, model_calls=0, network_calls=0, ready_to_host=False)
    write_new(output / 'inventory.json', result)
    signal.alarm(0)
    print(json.dumps(dict(output=str(output), selected_files=len(files), elapsed_seconds=result['elapsed_seconds'],
                         raw_bytes=state['budget']['raw_bytes'], ready_to_host=False)))


if __name__ == '__main__':
    main()
