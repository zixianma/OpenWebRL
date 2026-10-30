"""Node-local cooperative port-block leases for concurrent native evaluators."""
import fcntl
import os
from pathlib import Path
import socket

PORT_ENV = 'OPENWEBRL_ROLLOUT_PORT_BASE'
OLD_BASE = 'base_port = max(port_cursors.values()) if port_cursors else 15000'
NEW_BASE = ('base_port = max(port_cursors.values()) if port_cursors else '
            'int(os.environ.get("OPENWEBRL_ROLLOUT_PORT_BASE", "15000"))')


def patch_rollout_ports(text):
    if text.count(OLD_BASE) != 1:
        raise ValueError('Unexpected source for native engine port isolation')
    return text.replace(OLD_BASE, NEW_BASE)


def lease_ports(job, *, directory=None, bases=None, size=256):
    """Return (open lock file, base); keep the file open until workers exit.

    Local file locks isolate cooperating jobs. Probe the entire block to avoid
    pre-existing listeners from jobs using an older launcher. Uncooperative
    processes can still bind later; the native allocator also checks each port.
    """
    directory = Path(directory or f'/tmp/openwebrl-model-ports-{os.getuid()}')
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    bases = list(bases if bases is not None else range(20000, 28000, 256))
    offset = int(job) % len(bases)
    for base in bases[offset:]+bases[:offset]:
        handle = (directory/f'{base}.lock').open('a+')
        probes = []
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for port in range(base, base+size):
                probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                probes.append(probe)
                probe.bind(('0.0.0.0', port))
            handle.seek(0); handle.truncate()
            handle.write(str(job)+'\n'); handle.flush()
            return handle, base
        except (BlockingIOError, OSError):
            handle.close()
        finally:
            for probe in probes:
                probe.close()
    raise RuntimeError('No free isolated model-server port block')
