import asyncio
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from PIL import Image
from fastapi import HTTPException
sys.path.insert(0,str(Path('scripts').resolve()))
from serve_sol_selector import Selector,Request,MODEL,app
import run_arm_turn_bonus_cycles as runner


def request():
    image=io.BytesIO(); Image.new('RGB',(32,32),'white').save(image,format='PNG')
    return Request(mode='selection',task='find the button',url='https://example.org',history=[dict(thought='HISTORY_REASON',action='HISTORY_ACTION')],
        candidates=[dict(thought=f'REASON_{i}',action=f'ACTION_{i}') for i in range(5)],screenshot=base64.b64encode(image.getvalue()).decode())


class Sol(unittest.IsolatedAsyncioTestCase):
    async def test_http_route_accepts_the_actual_json_request_body(self):
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as root:
            response=SimpleNamespace(id='http-test',model=MODEL,status='completed',output_text='{"selection":2}',
                usage=SimpleNamespace(model_dump=lambda:dict(input_tokens=100,output_tokens=20,input_tokens_details=dict(cached_tokens=0))))
            client=SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=response)))
            a=SimpleNamespace(output=root,concurrency=2,max_requests=10,max_cost_usd=2,reasoning_effort='medium',max_output_tokens=2048)
            with TestClient(app(a,client)) as http:
                r=http.post('/select',json=request().model_dump())
                self.assertEqual(r.status_code,200,r.text)
                self.assertEqual(r.json()['raw'],'{"selection":2}')
                self.assertEqual(http.post('/select',json={'task':'missing fields'}).status_code,422)

    async def test_canonical_modalities_schema_usage_and_no_outcome_information(self):
        with tempfile.TemporaryDirectory() as root:
            response=SimpleNamespace(id='test',model=MODEL,status='completed',output_text='{"selection":4}',
                usage=SimpleNamespace(model_dump=lambda:dict(input_tokens=100,output_tokens=20,input_tokens_details=dict(cached_tokens=50),output_tokens_details=dict(reasoning_tokens=8))))
            client=SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=response)))
            a=SimpleNamespace(output=root,concurrency=2,max_requests=10,max_cost_usd=2,reasoning_effort='medium',max_output_tokens=2048)
            s=Selector(a,client); result=await s.select(request())
            self.assertEqual(result['raw'],'{"selection":4}')
            self.assertAlmostEqual(result['cost_usd'],.00062)
            args=client.responses.create.call_args.kwargs
            self.assertEqual(args['model'],MODEL)
            self.assertTrue(args['text']['format']['strict'])
            serialized=json.dumps(args['input'])
            for token in ['REASON_0','ACTION_4','HISTORY_REASON','HISTORY_ACTION','input_image']:
                self.assertIn(token,serialized)
            self.assertNotIn('executed_index',serialized)
            row=json.loads((Path(root)/'api-requests.jsonl').read_text())
            self.assertTrue(row['passed'])
            client.responses.create.return_value.status='incomplete'
            with self.assertRaises(HTTPException): await s.select(request())
            usage=json.loads((Path(root)/'usage.json').read_text())
            self.assertEqual(usage['consecutive_failures'],1)
            self.assertEqual(usage['reserved_cost_usd'],0)
            s.args.max_requests=2
            with self.assertRaises(HTTPException) as exc: await s.select(request())
            self.assertEqual(exc.exception.status_code,429)
            self.assertEqual(client.responses.create.call_count,2)

    async def test_training_provider_changes_no_actor_or_native_reward_settings(self):
        arm,sol=runner.plan('CHECK','arm'),runner.plan('CHECK','sol')
        for key in ['requested_resources','browser_config','checkpoint','requested_iterations','fresh_optimizer']:
            self.assertEqual(arm[key],sol[key])
        for key in ['beta','scored_fraction','k','seed','label_timeout_seconds']:
            self.assertEqual(arm['arm_config'][key],sol['arm_config'][key])
        for key in ['NUM_GPUS','TP_SIZE','NUM_ROLLOUT','HF_CHECKPOINT','BROWSER_CONCURRENCY','JUDGE_MODEL','GLOBAL_BATCH_SIZE','LEARNING_RATE']:
            self.assertEqual(arm['environment'][key],sol['environment'][key])
        self.assertEqual(sol['arm_config']['selector_checkpoint'],MODEL)


if __name__=='__main__': unittest.main()
