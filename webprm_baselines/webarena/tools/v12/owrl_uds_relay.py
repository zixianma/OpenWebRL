#!/usr/bin/env python3
"""v12 host-side relay: a UNIX socket in the job scratch -> one loopback TCP model server.

The WebArena cell worker runs inside the site's network namespace (nsenter --net), where the host's loopback model
servers are unreachable. Pathname UNIX sockets are not network-namespace scoped, so the worker reaches the OpenWebRL
actor (sglang /generate) and the Piotr SelectionARM (/select) through these sockets with httpx's UDS transport.
Bytes are copied unchanged in both directions; nothing is parsed, retried or cached. Stdlib only.

Usage: owrl_uds_relay.py --listen PATH --port PORT [--ready FILE]
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time


async def pipe(reader, writer):
    try:
        while True:
            data = await reader.read(1 << 16)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def serve(listen, port, ready=None, host='127.0.0.1'):
    path = Path(listen)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Relay socket already exists: {path}')
    if len(os.fsencode(str(path))) >= 108:
        raise ValueError('UNIX socket path too long')
    counts = dict(connections=0)

    async def handle(client_reader, client_writer):
        counts['connections'] += 1
        try:
            server_reader, server_writer = await asyncio.open_connection(host, port)
        except OSError:
            client_writer.close()
            return
        await asyncio.gather(pipe(client_reader, server_writer), pipe(server_reader, client_writer))

    old = os.umask(0o077)
    try:
        server = await asyncio.start_unix_server(handle, path=str(path))
    finally:
        os.umask(old)
    os.chmod(path, 0o600)
    if ready:
        Path(ready).write_text(json.dumps(dict(listen=str(path), host=host, port=port, pid=os.getpid(),
                                               started_epoch=time.time())) + '\n')
    async with server:
        await server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--listen', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--ready')
    args = parser.parse_args()
    asyncio.run(serve(args.listen, args.port, args.ready))


if __name__ == '__main__':
    main()
