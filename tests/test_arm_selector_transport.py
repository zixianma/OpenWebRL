"""Selector connection recovery preserves requests and bounds total waiting."""
import asyncio
import json
import unittest
from unittest.mock import patch
import httpx
from openwebrl.arm_inference import request_selection_result


class SelectorTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_connection_retry_preserves_candidates_and_response(self):
        requests=[]
        async def handle(request):
            requests.append(json.loads(request.content))
            if len(requests)==1:raise httpx.ConnectTimeout('connection stalled')
            return httpx.Response(200,json={'raw':'{"action_index":2}'})
        original=httpx.AsyncClient
        def client(**kwargs):return original(transport=httpx.MockTransport(handle),**kwargs)
        payload={'candidates':[{'action':'a'},{'action':'b'}]}
        with patch('httpx.AsyncClient',side_effect=client):
            result=await request_selection_result('http://localhost',payload,1.,.1)
        self.assertEqual(result,{'raw':'{"action_index":2}'})
        self.assertEqual(requests,[payload,payload])

    async def test_read_and_permanent_http_failures_are_not_retried(self):
        for error in ('read','http'):
            requests=[]
            async def handle(request):
                requests.append(request)
                if error=='read':raise httpx.ReadTimeout('response stalled')
                return httpx.Response(400,text='invalid request')
            original=httpx.AsyncClient
            def client(**kwargs):return original(transport=httpx.MockTransport(handle),**kwargs)
            with patch('httpx.AsyncClient',side_effect=client):
                with self.assertRaises((httpx.ReadTimeout,httpx.HTTPStatusError)):
                    await request_selection_result('http://localhost',{},1.,.1)
            self.assertEqual(len(requests),1)

    async def test_overall_deadline_cancels_wait_and_legacy_mode_does_not_retry(self):
        async def slow(request):
            await asyncio.sleep(1)
            return httpx.Response(200,json={})
        original=httpx.AsyncClient
        def client(**kwargs):return original(transport=httpx.MockTransport(slow),**kwargs)
        with patch('httpx.AsyncClient',side_effect=client):
            with self.assertRaises(TimeoutError):
                await request_selection_result('http://localhost',{},.02,.01)
        calls=[]
        async def failed(request):calls.append(request);raise httpx.ConnectTimeout('stalled')
        def legacy(**kwargs):return original(transport=httpx.MockTransport(failed),**kwargs)
        with patch('httpx.AsyncClient',side_effect=legacy):
            with self.assertRaises(httpx.ConnectTimeout):await request_selection_result('http://localhost',{},1.)
        self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
