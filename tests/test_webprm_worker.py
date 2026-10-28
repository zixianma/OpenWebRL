import tempfile
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from openwebrl.webprm_worker import SharedBudget, safe_action, native_tournament, arm_mode, input_token_upper_bound


class WorkerContracts(unittest.TestCase):
    def test_action_whitelist(self):
        actions=SimpleNamespace(action_set={'click':object()},to_python_code=lambda _:None)
        self.assertEqual(safe_action("click('12')",actions),"click('12')")
        for action in ["eval('x')", "click(open('x'))", "click('1'); click('2')", "click(**{})", "__import__('os').system('id')"]:
            with self.assertRaises((ValueError,SyntaxError)):
                safe_action(action,actions)

    def test_native_bye_order(self):
        winner,trace=native_tournament([1,2,3,4,5],lambda a,b:1)
        self.assertEqual([(x['left'],x['right']) for x in trace],[(1,2),(3,4),(5,1),(3,5)])
        self.assertEqual(winner,3)
        with self.assertRaises(ValueError):
            native_tournament([1,2],lambda a,b:0)

    def test_durable_budget_and_unknown_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'budget.json'
            budget=SharedBudget(path,cap=30.)
            first=budget.reserve(20.,'actor')
            with self.assertRaises(RuntimeError):
                SharedBudget(path).reserve(11.,'judge')
            budget.settle(first,5.,{'prompt_tokens':10})
            SharedBudget(path).reserve(25.,'judge')
            with self.assertRaises(RuntimeError):
                budget.reserve(.01,'actor')
            with self.assertRaises(RuntimeError):
                budget.settle(first,4.,{})


class IntegrationContracts(unittest.TestCase):
    def test_local_http_error_keeps_context_cause_and_journal(self):
        from openwebrl.webprm_worker import completion
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(deadline=time.time()+60, budget_path=str(root/'budget.json'))
            payload = dict(model='WebShepherd8B', messages=[], max_tokens=2048)
            body = b'{"error":{"message":"Input12574 exceeds context10000"}}'
            original = HTTPError('http://localhost/v1/chat/completions',400,'Bad Request',{},io.BytesIO(body))
            with patch('openwebrl.webprm_worker.loopback_urlopen', side_effect=original):
                with self.assertRaises(HTTPError) as caught:
                    completion(config,payload,root/'requests.jsonl',local=True,endpoint='http://localhost/v1')
            self.assertEqual(caught.exception.code,400)
            self.assertIn('Input12574 exceeds context10000',str(caught.exception))
            self.assertIs(caught.exception.__cause__,original)
            records = [json.loads(line) for line in (root/'requests.jsonl').read_text().splitlines()]
            self.assertEqual([r['event'] for r in records],['request','response_error'])
            self.assertEqual(records[-1]['error_body'],body.decode())
            self.assertFalse(records[-1]['body_truncated'])
            self.assertIsNone(records[-1]['reservation'])
            self.assertFalse((root/'budget.json').exists())

    def test_provider_http_error_retains_reservation_and_redacts_key(self):
        from openwebrl.webprm_worker import ACTOR,completion
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(deadline=time.time()+60,budget_path=str(root/'budget.json'))
            payload = dict(model=ACTOR,messages=[],max_tokens=1)
            fake_key = 'fixture-secret-must-not-appear'
            original = HTTPError('http://localhost/v1/chat/completions',401,'Unauthorized',{},
                                 io.BytesIO(('Invalid key '+fake_key).encode()))
            with patch('openwebrl.webprm_worker.load_api_credentials',return_value=fake_key), \
                 patch('openwebrl.webprm_worker.urlopen',side_effect=original):
                with self.assertRaises(HTTPError) as caught:
                    completion(config,payload,root/'requests.jsonl',purpose='policy')
            self.assertTrue(str(caught.exception).startswith('HTTP Error 401:'))
            self.assertNotIn(fake_key,str(caught.exception))
            text = (root/'requests.jsonl').read_text()
            self.assertNotIn(fake_key,text)
            self.assertIn('[REDACTED_API_KEY]',text)
            budget = json.loads((root/'budget.json').read_text())
            self.assertEqual(budget['calls'][0]['state'],'reserved')
            self.assertGreater(budget['charged'],0)

    def test_http_diagnostics_body_is_bounded(self):
        from openwebrl.webprm_worker import completion
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(deadline=time.time()+60,budget_path=str(root/'budget.json'))
            original = HTTPError('http://localhost/v1',400,'Bad Request',{},io.BytesIO(b'x'*20000))
            with patch('openwebrl.webprm_worker.loopback_urlopen',side_effect=original):
                with self.assertRaises(HTTPError):
                    completion(config,dict(model='local',messages=[]),root/'requests.jsonl',
                               local=True,endpoint='http://localhost/v1')
            error = json.loads((root/'requests.jsonl').read_text().splitlines()[-1])
            self.assertEqual(len(error['error_body']),16384)
            self.assertTrue(error['body_truncated'])

    def test_canonical_plan_modes_and_analysis(self):
        from openwebrl.webprm_baselines import ARMS,prepare_plan,paired_summary
        from openwebrl.webprm_worker import PILOT_ARMS
        tasks=[dict(task_id=str(i)) for i in range(300)]
        plan=prepare_plan(tasks,'hash')
        self.assertEqual(tuple(plan['arms']),PILOT_ARMS)
        records=[]
        for arm in ARMS:
            family,mode=arm_mode(arm)
            self.assertEqual(family,'arbiter' if arm.startswith('arbiter') else 'shepherd')
            self.assertEqual(mode,'unguided' if arm.endswith('unguided') else 'random' if arm.endswith('random5') else 'guided')
            records.append(dict(task_id='one',arm=arm,success=0,valid=True))
        self.assertEqual(set(paired_summary(records,['one'],resamples=100)['methods']),set(ARMS))

    def test_image_base64_excluded_from_text_bound(self):
        messages=[dict(role='user',content=[dict(type='image_url',image_url={'url':'data:image/jpeg;base64,'+'x'*1000000,'detail':'high'}),dict(type='text',text='judge')])]
        self.assertLess(input_token_upper_bound(messages),30000)

    def test_usage_overrun_is_durably_global_halt(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'budget.json'
            budget=SharedBudget(path)
            reservation=budget.reserve(1.,'actor')
            with self.assertRaises(RuntimeError):
                budget.settle(reservation,2.,{})
            self.assertEqual(json.loads(path.read_text())['charged'],2.)
            with self.assertRaises(RuntimeError):
                budget.reserve(.01,'actor')
