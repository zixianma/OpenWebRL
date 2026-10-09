"""Actual loopback HTTP only: no browser, model, GPU or external requests."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from openwebrl.controlled_sft_eval import HTTPTransports


def transport(client, base, events):
    return HTTPTransports(client, actor_base_url=base, kev_base_url='http://127.0.0.1:9',
        openai_key='fixture', jev_key='fixture', journal=lambda e,r:events.append((e,r)),
        normalize_actor=lambda text,*unused:text)


def test_local_sft_uses_distinct_tcp_connections_and_identical_requests():
    async def run():
        connections=[];received=[];handlers=set()
        async def serve(reader, writer):
            task=asyncio.current_task();handlers.add(task)
            connection=len(connections);connections.append(connection)
            try:
                while True:
                    try:head=await reader.readuntil(b'\r\n\r\n')
                    except asyncio.IncompleteReadError:break
                    headers={k.lower():v.strip() for k,v in
                             (line.decode().split(':',1) for line in head.split(b'\r\n')[1:] if b':' in line)}
                    body=await reader.readexactly(int(headers['content-length']))
                    received.append((connection,headers,json.loads(body)))
                    payload=json.dumps({'ok':True}).encode()
                    # Even a server that omits its Connection response header
                    # must not make the client reuse a Connection: close request.
                    writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: '+str(len(payload)).encode()+b'\r\n\r\n'+payload)
                    await writer.drain()
            finally:
                writer.close();await writer.wait_closed();handlers.discard(task)
        server=await asyncio.start_server(serve,'127.0.0.1',0)
        base=f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}'
        request={'text':'same frozen prompt','sampling_params':{'temperature':1,'max_new_tokens':4096,'sampling_seed':42}}
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                t=transport(client,base,[])
                await t._post('sft',base+'/generate',request)
                await t._post('sft',base+'/generate',request)
                await asyncio.gather(*(t._post('sft',base+'/generate',request) for _ in range(5)))
            assert len(received)==7 and len(connections)==7
            assert len({c for c,_,_ in received})==7
            assert all(h['connection']=='close' and r==request for _,h,r in received)
        finally:
            server.close();await server.wait_closed()
            if handlers:await asyncio.gather(*list(handlers))
    asyncio.run(run())


def test_external_and_kev_headers_unchanged_by_local_sft_mitigation():
    async def run():
        calls=[]
        async def post(url,**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(status_code=200,json=lambda:{'ok':True})
        t=transport(SimpleNamespace(post=post),'http://127.0.0.1:9',[])
        for label in ('jev','luna','kev','om2w_judge'):
            await t._post(label,'http://127.0.0.1:9/fixture',{},key='fixture')
        assert all(c['headers']=={'Authorization':'Bearer fixture'} for c in calls)
    asyncio.run(run())


def test_readerror_is_logged_with_bounded_cause_without_retry_or_payload_leak():
    async def run():
        count=0;events=[];request={'text':'private task data','sampling_params':{'sampling_seed':42}}
        async def post(url,**kwargs):
            nonlocal count
            count+=1
            try:raise ConnectionResetError(104,'reset '+('x'*4096))
            except ConnectionResetError as cause:raise httpx.ReadError('loopback read failed') from cause
        t=transport(SimpleNamespace(post=post),'http://127.0.0.1:9',events)
        with pytest.raises(httpx.ReadError):await t._post('sft','http://127.0.0.1:9/generate',request)
        assert count==1 and len(events)==1
        name,record=events[0]
        assert name=='http_transport_error' and record['retried'] is False
        assert record['response_received'] is False and record['status'] is record['body'] is None
        assert record['connection_close'] and record['elapsed_seconds']>=0
        assert len(record['request_sha256'])==64
        assert [c['type'] for c in record['exception_chain']]==['ReadError','ConnectionResetError']
        assert record['exception_chain'][1]['errno']==104
        assert len(record['exception_chain'][1]['message'])==2048
        assert 'private task data' not in json.dumps(record)
    asyncio.run(run())


def test_connection_refusal_does_not_get_reclassified_as_model_output():
    async def run():
        events=[];calls=[]
        async def post(url,**kwargs):calls.append(url);raise httpx.ConnectError('refused')
        t=transport(SimpleNamespace(post=post),'http://127.0.0.1:9',events)
        with pytest.raises(httpx.ConnectError):await t._post('sft','http://127.0.0.1:9/generate',{})
        assert len(calls)==1 and events[0][1]['exception_chain'][0]['type']=='ConnectError'
    asyncio.run(run())
