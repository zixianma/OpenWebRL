"""Bounded owned loopback/Unix relays for isolated original local services.

A bridge endpoint never selects a remote host. Private backend namespaces can
expose Unix sockets; the frontend/browser namespace exposes only127.0.0.1.
This transport alone proves neither namespace isolation nor website readiness.
The allocation owner must await close() and persist the returned statistics.
"""
from __future__ import annotations
import asyncio
from pathlib import Path
import os
import stat
import time


class BridgeClosed(ValueError):pass


def private_socket(value):
    path=Path(value)
    if (not path.is_absolute() or path.resolve()!=path or path.parent.parent!=Path('/tmp') or
        not path.parent.name.startswith('webprm-map-bridge-') or len(os.fsencode(path))>100 or
        any(p.is_symlink() for p in (path,*path.parents))):raise BridgeClosed('Canonical owned short /tmp Unix socket required')
    parent=path.parent.stat()
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid!=os.getuid() or stat.S_IMODE(parent.st_mode)!=0o700:
        raise BridgeClosed('Bridge directory must be private0700 and owned by current user')
    return path


class LocalServiceBridge:
    def __init__(self,*,socket_path,port,deadline,maximum_connections=16,maximum_bytes=512*2**20):
        self.socket_path=private_socket(socket_path)
        if (type(port) is not int or not 1<=port<=65535 or type(maximum_connections) is not int or not 1<=maximum_connections<=64 or
            type(maximum_bytes) is not int or not 1<=maximum_bytes<=8*2**30 or not time.time()<deadline<=time.time()+24*3600):
            raise BridgeClosed('Explicit bounded port/time/concurrency/byte limits required')
        self.port,self.deadline,self.maximum_connections=port,deadline,maximum_connections
        self.maximum_bytes=self.remaining=maximum_bytes;self.tasks=set();self.server=None;self.stopping=False;self.direction=None
        self.stats=dict(accepted=0,rejected=0,forwarded_bytes=0,errors=0,canceled=0)

    async def open(self,direction):
        if self.server is not None or self.stopping or direction not in ('loopback_to_unix','unix_to_loopback'):raise BridgeClosed('Bridge can start once with a fixed direction')
        self.direction=direction
        if direction=='unix_to_loopback':
            if self.socket_path.exists():raise BridgeClosed('Never overwrite an old Unix socket')
            self.server=await asyncio.start_unix_server(self._accept,path=str(self.socket_path),limit=65536)
            os.chmod(self.socket_path,0o600)
        else:
            proof=self.socket_path.stat()
            if not stat.S_ISSOCK(proof.st_mode) or proof.st_uid!=os.getuid() or stat.S_IMODE(proof.st_mode)!=0o600:
                raise BridgeClosed('Backend Unix socket must already be owned and private0600')
            self.server=await asyncio.start_server(self._accept,host='127.0.0.1',port=self.port,limit=65536)
        return self

    def _accept(self,reader,writer):
        if self.stopping or time.time()>=self.deadline or len(self.tasks)>=self.maximum_connections or self.remaining<=0:
            self.stats['rejected']+=1;writer.close();return
        task=asyncio.create_task(self._client(reader,writer));self.tasks.add(task);task.add_done_callback(self.tasks.discard)
        self.stats['accepted']+=1

    async def _pump(self,reader,writer):
        while not self.stopping and time.time()<self.deadline:
            # Reserve before each concurrent read. Canceled/unknown reads retain
            # the full reservation, so the campaign cap never resets on errors.
            take=min(65536,self.remaining)
            if take<=0:return
            self.remaining-=take
            data=await asyncio.wait_for(reader.read(take),timeout=max(.001,self.deadline-time.time()))
            self.remaining+=take-len(data)
            if not data:
                if writer.can_write_eof():writer.write_eof();await writer.drain()
                return
            writer.write(data);await asyncio.wait_for(writer.drain(),timeout=max(.001,self.deadline-time.time()))
            self.stats['forwarded_bytes']+=len(data)

    async def _client(self,reader,writer):
        peer=None;pumps=[]
        try:
            connect=(asyncio.open_unix_connection(str(self.socket_path),limit=65536) if self.direction=='loopback_to_unix'
                     else asyncio.open_connection('127.0.0.1',self.port,limit=65536))
            peer_reader,peer=await asyncio.wait_for(connect,timeout=min(10,max(.001,self.deadline-time.time())))
            pumps=[asyncio.create_task(self._pump(reader,peer)),asyncio.create_task(self._pump(peer_reader,writer))]
            await asyncio.gather(*pumps)
        except asyncio.CancelledError:
            self.stats['canceled']+=1;raise
        except Exception:
            self.stats['errors']+=1
        finally:
            for task in pumps:
                if not task.done():task.cancel()
            if pumps:await asyncio.gather(*pumps,return_exceptions=True)
            for endpoint in (writer,peer):
                if endpoint:
                    endpoint.close()
                    try:await asyncio.wait_for(endpoint.wait_closed(),timeout=1)
                    except Exception:pass

    async def close(self):
        self.stopping=True
        if self.server:self.server.close();await self.server.wait_closed()
        pending=tuple(self.tasks)
        for task in pending:task.cancel()
        if pending:await asyncio.gather(*pending,return_exceptions=True)
        self.tasks.difference_update(pending)
        return dict(self.stats,remaining_owned_coroutines=len(self.tasks),charged_bytes=self.maximum_bytes-self.remaining,
            maximum_bytes=self.maximum_bytes,unix_socket_retained=True,namespace_isolation_verified_by_bridge=False)
