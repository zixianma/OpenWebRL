"""Offline cloud ownership, billing and concurrency boundaries; no SDK/network."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from openwebrl.api_experiments.reproduce import cloud_browser as cloud
from openwebrl.api_experiments.reproduce import cloud_browsercode as integration
from openwebrl.api_experiments.reproduce import common

IDENT = '12345678-1234-4234-8234-123456789012'
LABELS = {'study':'offline-test','arm':'single'}


class Clock:
    def __init__(self): self.now = 10.0
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


class HTTP:
    def __init__(self, *, missing=False, metadata=None, stop=True, cdp=True):
        self.calls = []
        self.active = True
        self.stop = stop
        self.cdp = cdp
        self.metadata = LABELS if metadata is None else metadata
        self.fees = {'browserCost':'0.01','proxyCost':'0','proxyUsedMb':'0'} if not missing else {}
        self.raise_post = False

    def request(self, method, path, payload=None):
        self.calls.append((method,path,payload))
        if method == 'POST' and self.raise_post:
            raise RuntimeError('Ambiguous transport')
        if method == 'PATCH' and self.stop: self.active = False
        view = dict(id=IDENT, metadata=self.metadata, status='active' if self.active else 'stopped',
                    startedAt='2026-11-01T00:00:00Z', finishedAt=None if self.active else '2026-11-01T00:00:05Z',
                    cdpUrl='wss://private-capability.test/session' if self.cdp else None, **self.fees)
        return 200, json.dumps(view)


class CloudBoundaries(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='browsercode-cloud-offline-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clock = Clock()
        self.http = HTTP()
        self.halts = []
        self.grants = []

    def browser(self, **options):
        def admission(payload):
            self.grants.append(payload)
            return {'approved':True,'proxy_overshoot_authorized':True}
        values = dict(metadata=LABELS,on_halt=self.halts.append,max_proxy_usd='1',
                      polls=3,watchdog_seconds=30,minimum_settlement_seconds=0,
                      sleep=self.clock.sleep,clock=self.clock)
        values.update(options)
        return cloud.CloudBrowser(self.root,self.http,admission,**values)

    def test_unstarted_close_has_no_http_fee_or_lifetime(self):
        owner = self.browser()
        receipt = owner.close()
        self.assertFalse(receipt['remote_session_created'])
        self.assertNotIn('provider_usage',receipt)
        self.assertNotIn('owned_browser_seconds',receipt)
        self.assertEqual(self.http.calls,[])

    def test_exact_one_create_and_explicit_stop_with_private_receipts(self):
        owner = self.browser()
        self.assertTrue(owner.start().startswith('wss://'))
        self.assertEqual(self.grants[0]['proxyCountryCode'],'us')
        self.assertFalse(self.grants[0]['enableRecording'])
        receipt = owner.close()
        owner.require_stopped()
        self.assertEqual(sum(c[0]=='POST' for c in self.http.calls),1)
        self.assertTrue(any(c[0]=='PATCH' and c[2]=={'action':'stop'} for c in self.http.calls))
        self.assertTrue(receipt['remote_stopped'])
        self.assertEqual(receipt['provider_browser_seconds'],5)
        self.assertEqual(receipt['provider_usage']['proxyCost'],'0')
        self.assertIn('provisional',receipt['billing_status'])
        for file in self.root.rglob('*.json'):
            self.assertEqual(file.stat().st_mode & 0o777,0o600)
        for row in (self.root/'timing-spans.jsonl').read_text().splitlines():
            self.assertNotIn('private-capability',row)
        self.assertIn('private-capability',(self.root/'cloud-private/http-0001.json').read_text())

    def test_admission_denial_cannot_dispatch_and_final_close_is_safe(self):
        owner = self.browser()
        owner.admission = lambda payload:{'approved':True,'proxy_overshoot_authorized':False}
        with self.assertRaises(ValueError): owner.start()
        self.assertEqual(self.http.calls,[])
        self.assertFalse(owner.close()['remote_session_created'])

    def test_ambiguous_create_never_retries_or_claims_clean(self):
        owner = self.browser()
        self.http.raise_post = True
        with self.assertRaises(RuntimeError): owner.start()
        with self.assertRaises(ValueError): owner.close()
        with self.assertRaises(FileExistsError): owner.start()
        self.assertEqual([c[0] for c in self.http.calls],['POST'])
        self.assertTrue(json.loads((self.root/'cloud-private/ambiguous-create.json').read_text())['reconciliation_required'])

    def test_returned_id_survives_bad_metadata_and_is_stopped(self):
        owner = self.browser()
        self.http.metadata = {'study':'wrong-owner'}
        with self.assertRaises(ValueError): owner.start()
        self.assertEqual(owner.session_id,IDENT)
        self.assertTrue(owner.receipt['remote_stopped'])
        self.assertFalse(owner.receipt['metadata_verified'])
        self.assertEqual(json.loads((self.root/'cloud-private/owned-session.json').read_text())['id'],IDENT)

    def test_cdp_failure_stops_known_owner(self):
        owner = self.browser()
        self.http.cdp = False
        with self.assertRaises(RuntimeError): owner.start()
        self.assertTrue(owner.receipt['remote_stopped'])
        self.assertEqual(sum(c[0]=='POST' for c in self.http.calls),1)

    def test_failing_halt_callback_still_stops_returned_owner(self):
        def fail(reason): raise RuntimeError('Halt callback failed')
        owner = self.browser(on_halt=fail)
        self.http.metadata = {'foreign':'owner'}
        with self.assertRaises(RuntimeError):owner.start()
        self.assertTrue(owner.receipt['remote_stopped'])

    def test_unverified_stop_cannot_admit_terra(self):
        owner = self.browser()
        owner.start()
        self.http.stop = False
        try:
            with self.assertRaises(ValueError): owner.close()
            with self.assertRaises(ValueError): owner.require_stopped()
        finally:
            self.http.stop = True
            owner.close()

    def test_missing_fee_is_unknown_not_zero(self):
        owner = self.browser()
        self.http.fees = {}
        owner.start()
        receipt = owner.close()
        self.assertFalse(receipt['billing_fields_complete'])
        self.assertEqual(receipt['provider_usage'],dict(browserCost=None,proxyCost=None,proxyUsedMb=None))

    def test_first_close_immutable_and_late_fee_reconciliation_separate(self):
        owner = self.browser()
        owner.start()
        first = owner.close()
        data = (self.root/'cloud-private/first-close.json').read_bytes()
        self.clock.now += 900
        self.http.fees['proxyCost'] = '0.8'
        self.assertEqual(owner.close(),first)
        result = owner.reconcile_usage()
        self.assertEqual(result['provider_usage']['proxyCost'],'0.8')
        self.assertFalse(result['actual_cash_payment_asserted'])
        self.assertEqual((self.root/'cloud-private/first-close.json').read_bytes(),data)

    def test_two_early_equal_polls_do_not_claim_settlement(self):
        owner = self.browser(minimum_settlement_seconds=1000)
        owner.start()
        receipt = owner.close()
        self.assertTrue(receipt['post_stop_usage_observed_stable'])
        self.assertFalse(receipt['minimum_settlement_window_observed'])
        self.assertIn('provisional',receipt['billing_status'])

    def test_shared_usage_halt_preserves_cleanup_and_sanitizes_view(self):
        observed = []
        def on_usage(row): observed.append(row); return False
        owner = self.browser(on_usage=on_usage)
        with self.assertRaises(ValueError): owner.start()
        self.assertTrue(owner.receipt['remote_stopped'])
        self.assertTrue(observed)
        self.assertNotIn('cdpUrl',observed[0])
        self.assertNotIn('liveUrl',observed[0])

    def test_decimal_and_timestamp_malformed_inputs(self):
        self.assertIsNone(cloud.amount(None))
        self.assertEqual(cloud.amount(0),'0')
        for value in (True,False,-1,'NaN','Infinity',{},[],'broken'):
            with self.subTest(value=value),self.assertRaises(ValueError): cloud.amount(value)
        for start,end in [('2026-11-01','2026-11-02'),('2026-11-02T00:00:00Z','2026-11-01T00:00:00Z')]:
            with self.assertRaises(ValueError): cloud.provider_duration(start,end)
        self.assertIsNone(cloud.provider_duration(None,None))

    def test_actual_overlapping_http_calls_use_union_without_serializing(self):
        barrier = threading.Barrier(5)
        class Endpoint:
            def create(self,**kwargs):
                barrier.wait(timeout=3)
                time.sleep(.03)
                return kwargs
        source = types.SimpleNamespace(responses=Endpoint())
        timeline = cloud.Timeline(self.root)
        client = integration.TimedClient(source,timeline,'actor')
        with ThreadPoolExecutor(5) as pool:
            results = list(pool.map(lambda i:client.responses.create(candidate=i),range(5)))
        self.assertEqual(results,[{'candidate':i} for i in range(5)])
        result = timeline.summary()['actor_http']
        self.assertEqual(result['calls'],5)
        self.assertGreater(result['summed_seconds'],result['union_seconds']*2)
        self.assertAlmostEqual(cloud.interval_union([(0,4),(1,3),(5,6)]),5)
        with self.assertRaises(ValueError): cloud.interval_union([(True,2)])


class IntegrationBoundaries(unittest.TestCase):
    setUp = CloudBoundaries.setUp
    def worker(self):
        class Model:
            def __init__(self,*args,**kwargs): pass
            def query(self,*args,**kwargs): return 'same-action'
        class Workspace: pass
        return types.SimpleNamespace(OwnedBrowser=object,PairedModel=Model,BrowserCodeWorkspace=Workspace,
                                     judge=lambda *a:'judged')

    def registry(self,worker,**options):
        return integration.install(worker,types.SimpleNamespace(),http_factory=lambda:self.http,
                  admission=lambda directory,payload:{'approved':True,'proxy_overshoot_authorized':True},
                  metadata_factory=lambda directory:LABELS,on_halt=self.halts.append,max_proxy_usd='1',
                  cloud_options=dict(polls=3,watchdog_seconds=30,minimum_settlement_seconds=0,
                                     clock=self.clock,sleep=self.clock.sleep),**options)

    def test_registry_has_no_construction_calls_and_restores(self):
        worker = self.worker()
        old = dict(worker.__dict__)
        registry = self.registry(worker)
        worker.OwnedBrowser(self.root)
        self.assertEqual(self.http.calls,[])
        self.assertTrue(registry.close_all()['verified'])
        registry.restore()
        self.assertEqual(worker.__dict__,old)

    def test_terra_requires_actual_owned_stop(self):
        worker = self.worker()
        registry = self.registry(worker)
        owner = worker.OwnedBrowser(self.root)
        with self.assertRaises(ValueError): worker.judge(None,None,'task',[],None,self.root)
        owner.start()
        with self.assertRaises(ValueError): worker.judge(None,None,'task',[],None,self.root)
        owner.close()
        self.assertEqual(worker.judge(types.SimpleNamespace(),None,'task',[],None,self.root),'judged')
        registry.close_all()
        registry.restore()

    def test_registry_rejects_dispatched_unresolved_owner(self):
        worker = self.worker()
        registry = self.registry(worker)
        owner = worker.OwnedBrowser(self.root)
        self.http.raise_post = True
        with self.assertRaises(RuntimeError): owner.start()
        with self.assertRaises(ValueError): registry.close_all()
        registry.restore()

    def test_sdk_init_failure_restores_even_if_cleanup_fails(self):
        class Budget:
            def __init__(self,root,*args): self.root=root
        worker = types.SimpleNamespace(Budget=Budget,ReservationBudget=Budget)
        env = types.SimpleNamespace()
        events=[]
        class Integration:
            judge_budget_type=None
            def close_all(self): events.append('close');raise RuntimeError('ownership unresolved')
            def restore(self): events.append('restore')
        def broken_sdk(**kwargs): raise ValueError('SDK init failed')
        args=types.SimpleNamespace(env_file=None,model_usd='1',judge_usd='1',model_calls=1,judge_calls=1)
        plan=dict(provider='gpt',output=str(self.root/'out'),runtime=str(self.root),browser_start_cap=1)
        for name in ('campaign-plan.json','runtime-config.json'):(self.root/name).write_text('{}')
        with patch.dict(os.environ,{'OPENAI_API_KEY':'offline-never-used'},clear=True),patch.dict(sys.modules,{'worker':worker,'environments':env,'openai':types.SimpleNamespace(OpenAI=broken_sdk)}):
            with self.assertRaises(RuntimeError):common.execute(args,plan,[],worker_setup=lambda w,e:Integration())
        self.assertEqual(events,['close','restore'])


class ConfigureBoundaries(unittest.TestCase):
    def fixture(self,root):
        write=lambda name,value:(root/name).write_text(json.dumps(value))
        sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
        plan=dict(arms=[dict(arm='luna_single_high',model='gpt-6-luna',output_tokens=12288,proposal_count=1)],
                  arm_protocols={'luna_single_high':dict(actor_model='gpt-6-luna')},protocol={'max_decisions':30})
        write('campaign-plan.json',plan)
        write('runtime-config.json',dict(pricing={}))
        write('portable-build.json',{})
        (root/'episode_metrics.py').write_text("cost=api_cost(u,'judge' if role=='judge' else 'actor',pricing)['upper_usd']\n")
        for name in ('common_judge.py','claude_budget.py','request_budget.py'):(root/name).write_text('original '+name)
        pins={str(p):sha(p) for p in root.iterdir()}
        write('source-manifest.json',{'source_hashes':pins,'plan_sha256':sha(root/'campaign-plan.json')})
        write('execution-source-manifest.json',{'original':True})

    def test_prospective_runtime_binds_terra_n5_and_original_archives(self):
        with tempfile.TemporaryDirectory(prefix='cloud-configure-offline-') as temp:
            root=Path(temp)
            self.fixture(root)
            sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            before={p.name:p.read_bytes() for p in root.iterdir()}
            with patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('Network forbidden')):
                receipt=integration.configure(root)
            self.assertFalse(receipt['paid_authorization'])
            config=json.loads((root/'runtime-config.json').read_text())
            self.assertEqual(config['judge_protocol']['model'],'gpt-5.6-terra')
            self.assertEqual(config['judge_protocol']['reasoning_effort'],'medium')
            self.assertEqual(config['judge_protocol']['max_completion_tokens'],4096)
            self.assertEqual(config['judge_protocol']['seed'],42)
            self.assertEqual(config['cloud_protocol']['proxy_country'],'us')
            plan=json.loads((root/'campaign-plan.json').read_text())
            self.assertEqual([(a['arm'],a['proposal_count']) for a in plan['arms']],[('luna_single_high',1),('parallel5_select_high',5)])
            for name,expected in before.items():self.assertEqual((root/'cloud-originals'/name).read_bytes(),expected)
            manifest=json.loads((root/'source-manifest.json').read_text())
            self.assertEqual(manifest,json.loads((root/'execution-source-manifest.json').read_text()))
            self.assertTrue(all(sha(Path(path))==pin for path,pin in manifest['source_hashes'].items()))
            self.assertEqual(manifest['plan_sha256'],sha(root/'campaign-plan.json'))
            self.assertEqual(common.required_cloud(root)['browser_backend'],'browser-use-v4')
            with self.assertRaises(ValueError):integration.configure(root)

    def test_cloud_frontdoor_cannot_silently_use_local_browser_or_mutated_plan(self):
        with tempfile.TemporaryDirectory(prefix='cloud-frontdoor-offline-') as temp:
            root=Path(temp)
            self.fixture(root)
            integration.configure(root)
            expected=common.required_cloud(root)
            args=types.SimpleNamespace(env_file=None)
            plan=dict(runtime=str(root),output=str(root/'never-created'),required_cloud=expected)
            with patch.dict(os.environ,{},clear=True),patch('urllib.request.OpenerDirector.open',side_effect=AssertionError('Network forbidden')):
                for changed in (plan,{k:v for k,v in plan.items() if k!='required_cloud'},dict(plan,required_cloud={})):
                    with self.assertRaisesRegex(ValueError,'bound Browser Use cloud owner'):
                        common.execute(args,changed,[])
                    self.assertFalse(Path(plan['output']).exists())
            marker=root/'cloud-experiment.json'
            marker.write_text(marker.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'source binding changed'):common.execute(args,plan,[],worker_setup=lambda *a:None)

    def test_cloud_backend_rejects_untrusted_hook_before_sdk(self):
        with tempfile.TemporaryDirectory(prefix='cloud-hook-offline-') as temp:
            root=Path(temp)
            self.fixture(root)
            integration.configure(root)
            args=types.SimpleNamespace(env_file=None)
            plan=dict(runtime=str(root),provider='gpt',output=str(root/'out'),required_cloud=common.required_cloud(root))
            events=[]
            fake=types.SimpleNamespace(close_all=lambda:events.append('close'),restore=lambda:events.append('restore'))
            with patch.dict(os.environ,{'OPENAI_API_KEY':'unused'},clear=True),patch.dict(sys.modules,{'worker':types.SimpleNamespace(),'environments':types.SimpleNamespace(),'openai':None}):
                with self.assertRaisesRegex(ValueError,'cloud Registry required before SDK'):
                    common.execute(args,plan,[],worker_setup=lambda *a:fake)
            self.assertEqual(events,['close','restore'])


if __name__=='__main__':unittest.main()
