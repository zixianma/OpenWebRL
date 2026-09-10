"""Bound connection stalls and do not flood the actor with oversized prompts."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import httpx
from slime.utils import http_utils


class HTTPRetryPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_connect_deadline_preserves_unbounded_read_and_default(self):
        for connect in (None, 10.0):
            args = SimpleNamespace(rollout_num_gpus=1, rollout_num_gpus_per_engine=1,
                                   sglang_server_concurrency=64, use_distributed_post=False)
            if connect is not None: args.http_connect_timeout_secs = connect
            with patch.object(http_utils, '_http_client', None):
                http_utils.init_http_client(args)
                client = http_utils._http_client
                try:
                    self.assertEqual(client.timeout.connect, connect)
                    self.assertIsNone(client.timeout.read)
                    self.assertIsNone(client.timeout.write)
                    self.assertIsNone(client.timeout.pool)
                finally:
                    await client.aclose()

    async def test_context_overflow_fails_after_one_request(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(400, text="Requested token count exceeds the model's maximum context length")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(httpx.HTTPStatusError):
                await http_utils._post(client, 'http://actor/generate', {}, max_retries=4)
        self.assertEqual(len(calls), 1)

    async def test_transient_http_error_still_retries(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, json={'text': 'ok'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.object(http_utils.asyncio, 'sleep', new=AsyncMock()):
                result = await http_utils._post(client, 'http://actor/generate', {}, max_retries=4)
        self.assertEqual(result, {'text': 'ok'})
        self.assertEqual(len(calls), 2)

    async def test_connect_timeout_is_retryable(self):
        calls = []
        def handler(request):
            calls.append(request)
            if len(calls) == 1: raise httpx.ConnectTimeout('connection stalled', request=request)
            return httpx.Response(200, json={'text': 'ok'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch.object(http_utils.asyncio, 'sleep', new=AsyncMock()):
                result = await http_utils._post(client, 'http://actor/generate', {}, max_retries=4)
        self.assertEqual(result, {'text': 'ok'})
        self.assertEqual(len(calls), 2)


if __name__ == '__main__': unittest.main()
