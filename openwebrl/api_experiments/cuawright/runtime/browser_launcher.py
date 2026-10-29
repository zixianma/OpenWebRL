#!/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/bin/python
"""Actor-visible Chromium proxy. The broker alone may execute physical Chrome."""
import base64
import json
import os
from pathlib import Path
import signal
import socket
import sys


def main():
    cfg = json.loads(Path(os.environ['CUAWRIGHT_NATIVE_GUARD_CONFIG']).read_text())
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(cfg['browser_broker_socket'])
    request = {'op':'launch', 'args':sys.argv[1:], 'cwd':os.getcwd(), 'env':dict(os.environ)}
    connection.sendall(json.dumps(request).encode() + b'\n')
    def stop(signum, frame):
        connection.close()
        raise SystemExit(128 + signum)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)
    with connection.makefile('rb') as stream:
        for line in stream:
            row = json.loads(line)
            if row['type'] in ('stdout', 'stderr'):
                fd = 1 if row['type'] == 'stdout' else 2
                try:
                    os.write(fd, base64.b64decode(row['data']))
                except BrokenPipeError:
                    # Native persistent CLI intentionally closes its startup pipe.
                    # The launcher remains alive until its owned group is closed.
                    pass
            elif row['type'] == 'exit':
                return row['returncode'] if row['returncode'] >= 0 else 128 - row['returncode']
            elif row['type'] == 'error':
                print(row['message'], file=sys.stderr)
                return 125
    return 125


if __name__ == '__main__':
    raise SystemExit(main())
