import asyncio
import os
from pathlib import Path
import socket
import tempfile
import time
import unittest
from openwebrl.webprm_local_service_bridge import LocalServiceBridge,BridgeClosed,private_socket


def free_port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='/tmp',prefix='webprm-map-bridge-');self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'backend.sock';self.bridges=[];self.servers=[]

    async def asyncTearDown(self):
        for bridge in reversed(self.bridges):
            result=await bridge.close();self.assertEqual(result['remaining_owned_coroutines'],0)
        for server in self.servers:server.close();await server.wait_closed()

    async def pair(self,bytecap=512*2**20):
        async def echo(reader,writer):
            try:
                while data:=await reader.read(65536):writer.write(data);await writer.drain()
            finally:writer.close();await writer.wait_closed()
        server=await asyncio.start_server(echo,'127.0.0.1',0);self.servers.append(server)
        port=server.sockets[0].getsockname()[1]
        backend=LocalServiceBridge(socket_path=self.path,port=port,deadline=time.time()+10,maximum_bytes=bytecap)
        await backend.open('unix_to_loopback');self.bridges.append(backend)
        port=free_port();front=LocalServiceBridge(socket_path=self.path,port=port,deadline=time.time()+10,maximum_bytes=bytecap)
        await front.open('loopback_to_unix');self.bridges.append(front)
        return front,backend,port

    async def test_bidirectional_synthetic_round_trip_and_owned_closure(self):
        front,backend,port=await self.pair()
        r,w=await asyncio.open_connection('127.0.0.1',port);w.write(b'map-fixture');await w.drain()
        self.assertEqual(await asyncio.wait_for(r.readexactly(11),timeout=2),b'map-fixture')
        w.close();await w.wait_closed()
        result=await front.close();self.assertEqual(result['remaining_owned_coroutines'],0)
        self.assertEqual(result['forwarded_bytes'],22);self.assertFalse(result['namespace_isolation_verified_by_bridge'])

    async def test_byte_reservations_remain_bounded_when_peer_stalls(self):
        front,backend,port=await self.pair(bytecap=8)
        r,w=await asyncio.open_connection('127.0.0.1',port);w.write(b'1234567890123456');await w.drain()
        await asyncio.sleep(.02);result=await front.close();w.close();await w.wait_closed()
        self.assertLessEqual(result['charged_bytes'],8);self.assertEqual(result['remaining_owned_coroutines'],0)

    async def test_socket_directory_must_be_private_and_never_overwritten(self):
        with self.assertRaises(BridgeClosed):private_socket('/gpfs/home/zixianma/forbidden.sock')
        self.path.write_bytes(b'old')
        bridge=LocalServiceBridge(socket_path=self.path,port=5432,deadline=time.time()+10)
        with self.assertRaises(BridgeClosed):await bridge.open('unix_to_loopback')
        self.assertEqual(self.path.read_bytes(),b'old')
        os.chmod(self.path.parent,0o755)
        with self.assertRaises(BridgeClosed):private_socket(self.path)
        os.chmod(self.path.parent,0o700)

    async def test_frontend_never_accepts_regular_file_as_backend(self):
        self.path.write_bytes(b'not-socket')
        bridge=LocalServiceBridge(socket_path=self.path,port=5432,deadline=time.time()+10)
        with self.assertRaises(BridgeClosed):await bridge.open('loopback_to_unix')

    async def test_closed_bridge_cannot_restart(self):
        front,backend,port=await self.pair();await front.close()
        with self.assertRaises(BridgeClosed):await front.open('loopback_to_unix')


if __name__=='__main__':unittest.main()
