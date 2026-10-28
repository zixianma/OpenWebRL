"""Stage the installed, unmodified Playwright Node executable on owned /tmp.

Playwright's released compute_driver_executable supports PLAYWRIGHT_NODEJS_PATH.
Only the executable moves; cli.js, SDK, browser and scientific policy stay pinned.
There are no subprocesses, downloads, retries or external service calls here.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

from openwebrl.webprm_campaign_queue import checked

PROTOCOL = 'cached_playwright_node_local_staging_v1'
SOURCE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/browser-env/lib/python3.12/site-packages/playwright/driver/node')
SOURCE_SHA256 = '0652a7596fd240292d603220c18afd94e60d2280a58d0a2f0571ca48f20fa04f'
SOURCE_BYTES = 97981176
MAX_BYTES = 128 * 2**20
SDK_DRIVER = SOURCE.parent.parent / '_impl/_driver.py'
SDK_DRIVER_SHA256 = '3286111aba672c0064bd825a9beecdbdbee74b219e4f58df4efdb39c4eaf7afa'
RELATIVE_DESTINATION = 'playwright-driver/node'


def expected_identity():
    return dict(protocol=PROTOCOL, source=str(SOURCE), source_sha256=SOURCE_SHA256,
                source_bytes=SOURCE_BYTES, max_bytes=MAX_BYTES,
                relative_destination=RELATIVE_DESTINATION, new_network_bytes=0,
                sdk_driver_source=str(SDK_DRIVER), sdk_driver_sha256=SDK_DRIVER_SHA256,
                environment_variable='PLAYWRIGHT_NODEJS_PATH')


def validate_identity(identity):
    if identity != expected_identity():
        raise ValueError('Exact cached Playwright Node identity and 128MiB zero-download bound required')


def stream_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb', buffering=0) as stream:
        while True:
            block = stream.read(2**20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stage_node(scratch, identity):
    """Copy once, charge actual bytes, fsync and rehash; preserve partial failures.

    The caller records any exception and stops before model/browser startup.
    An existing destination is never reused or overwritten after an unknown copy.
    """
    validate_identity(identity)
    scratch = checked(scratch)
    info = scratch.stat()
    if (Path('/tmp') not in scratch.parents or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid() or info.st_mode & 0o077):
        raise ValueError('Playwright Node requires a private canonical owned node-local scratch root')
    source = checked(SOURCE)
    folder = checked(scratch / 'playwright-driver')
    folder.mkdir(mode=0o700)
    destination = checked(scratch / RELATIVE_DESTINATION)
    copied = 0
    digest = hashlib.sha256()
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(source_fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_size != SOURCE_BYTES
                or before.st_size > MAX_BYTES):
            raise ValueError('Cached Playwright Node size/type changed')
        destination_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o700)
        with os.fdopen(destination_fd, 'wb', buffering=2**20) as output:
            while True:
                block = os.read(source_fd, 2**20)
                if not block:
                    break
                copied += len(block)
                if copied > MAX_BYTES or copied > SOURCE_BYTES:
                    raise ValueError('Cached Playwright Node copy exceeded pinned size/cap')
                output.write(block)
                digest.update(block)
            output.flush()
            os.fsync(output.fileno())
        after = os.fstat(source_fd)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise ValueError('Cached Playwright Node source changed during copy')
    finally:
        os.close(source_fd)
    if copied != SOURCE_BYTES or digest.hexdigest() != SOURCE_SHA256:
        raise ValueError('Cached Playwright Node copied bytes differ')
    post_sha = stream_sha(destination)
    target = destination.stat()
    if (post_sha != SOURCE_SHA256 or target.st_size != SOURCE_BYTES
            or not stat.S_ISREG(target.st_mode) or target.st_uid != os.getuid()
            or target.st_nlink != 1 or stat.S_IMODE(target.st_mode) != 0o700):
        raise ValueError('Node-local Playwright Node post-copy verification failed')
    folder_fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(folder_fd)
    finally:
        os.close(folder_fd)
    return dict(protocol=PROTOCOL, source=str(SOURCE), source_sha256=SOURCE_SHA256,
                source_bytes=SOURCE_BYTES, destination=str(destination),
                destination_sha256=post_sha, retained_local_bytes=copied,
                max_bytes=MAX_BYTES, new_network_bytes=0, fsynced=True,
                environment=dict(PLAYWRIGHT_NODEJS_PATH=str(destination)),
                sdk_driver_source=str(SDK_DRIVER), sdk_driver_sha256=SDK_DRIVER_SHA256,
                scratch_accounting='Included in the existing 512GiB apparent-byte cap')
