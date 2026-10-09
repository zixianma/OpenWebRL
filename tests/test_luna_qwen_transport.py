import asyncio
import json
from types import SimpleNamespace
import httpx
import pytest
from openwebrl.luna_qwen_transport import init_candidate_http_client,flush_cache_when_idle


def test_all_ten_candidates_reach_server_concurrently():
    from slime.utils import http_utils
    async def run():
        active=peak=0
        async def handle(reader,writer):
            nonlocal active,peak
            headers=await reader.readuntil(b'\r\n\r\n')
            length=int(next(x.split(b':',1)[1] for x in headers.split(b'\r\n') if x.lower().startswith(b'content-length:')))
            await reader.readexactly(length);active+=1;peak=max(peak,active)
            await asyncio.sleep(.08);active-=1
            body=b'{"ok":true}'
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body)
            await writer.drain();writer.close();await writer.wait_closed()
        server=await asyncio.start_server(handle,'127.0.0.1',0)
        args=SimpleNamespace(rollout_num_gpus=1,rollout_num_gpus_per_engine=1,sglang_server_concurrency=1,use_distributed_post=False)
        assert http_utils._http_client is None
        init_candidate_http_client(args,http_utils.init_http_client)
        try:
            port=server.sockets[0].getsockname()[1]
            result=await asyncio.gather(*(http_utils.post(f'http://127.0.0.1:{port}/generate',{'candidate':i},max_retries=1) for i in range(10)))
            assert peak==10 and all(r=={'ok':True} for r in result)
        finally:
            await http_utils._http_client.aclose();http_utils._http_client=None
            server.close();await server.wait_closed()
    asyncio.run(run())


def test_pending_generation_drains_before_cache_flush_without_ignoring_errors():
    async def run():
        responses=[httpx.Response(400,text='running or waiting requests'),httpx.Response(200)]
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req:responses.pop(0))) as client:
            result=await flush_cache_when_idle(client,'http://localhost/flush_cache',poll=.001)
            assert result['attempts']==2 and result['cache_flushed']
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req:httpx.Response(400,text='unexpected bad request'))) as client:
            with pytest.raises(httpx.HTTPStatusError):await flush_cache_when_idle(client,'http://localhost/flush_cache')
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req:httpx.Response(400,text='running or waiting requests'))) as client:
            with pytest.raises(TimeoutError):await flush_cache_when_idle(client,'http://localhost/flush_cache',timeout=.005,poll=.001)
    asyncio.run(run())
