"""Zero-provider admission and paired-drain regressions."""
import hashlib
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('stealth_pilot', Path(__file__).resolve().parents[1] / 'reproduce/stealth_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class Guards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.control = pilot.Control(self.root/'control')
        self.control.initialize(time.time()+14400)

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_count_is_durable_before_network_and_never_refunded(self):
        for i in range(42):
            grant = self.control.admission(self.root/str(i), {'test':i})
            self.assertIs(grant['approved'], True)
            self.assertEqual(len(pilot.read(self.control.root/'control.json')['creates']),i+1)
        with self.assertRaises(ValueError):
            self.control.admission(self.root/'43',{})

    def test_same_directory_cannot_create_again_after_failure(self):
        self.control.admission(self.root/'one',{})
        with self.assertRaises(ValueError):
            self.control.admission(self.root/'one',{})

    def observe(self, cost, proxy='0', **updates):
        directory = self.root/'one'
        if not self.control.snapshot()['creates']:
            self.control.admission(directory,{})
        return self.control.usage(directory,dict(id='owned',status='active',browserCost=cost,proxyCost=proxy,proxyUsedMb='0',**updates))

    def test_unknown_usage_is_not_zero_and_blocks_admission(self):
        self.observe('2')
        self.assertFalse(self.observe(None))
        state = self.control.snapshot()
        self.assertFalse(state['sessions'][str((self.root/'one').resolve())]['known'])
        self.assertEqual(pilot.Control.total(state),2)
        with self.assertRaises(ValueError):
            self.control.admission(self.root/'other',{})

    def test_unknown_first_usage_is_explicit_not_zero_metric(self):
        self.assertFalse(self.observe(None))
        session = next(iter(self.control.snapshot()['sessions'].values()))
        self.assertFalse(session['known'])
        self.assertNotIn('browserCost',session)

    def test_fee_thresholds_distinguish_drain_from_active_stop(self):
        self.assertTrue(self.observe('8'))
        with self.control.locked() as state:
            with self.assertRaises(ValueError):
                self.control.check(state,new_pair=True)
        self.assertFalse(self.observe('9','1'))
        self.assertIn('10_usd',self.control.snapshot()['abort']['reason'])

    def test_lower_later_receipt_does_not_release_spending(self):
        self.observe('3','1')
        self.observe('1','0')
        self.assertEqual(pilot.Control.total(self.control.snapshot()),4)
        entry=next(iter(self.control.snapshot()['sessions'].values()))
        self.assertEqual(entry['latest_provider_usage']['browserCost'],'1')
        self.assertEqual(entry['latest_provider_usage']['proxyCost'],'0')
        self.assertEqual(entry['admission_maximum_observed']['browserCost'],'3')
        self.assertIn('not_actual_spend',entry['admission_fee_basis'])

    def test_malformed_usage_callback_returns_false_for_cleanup(self):
        self.assertFalse(self.observe('NaN'))
        self.assertIsNotNone(self.control.snapshot()['abort'])

    def test_barrier_releases_both_and_requires_finally_pair(self):
        finished = []
        def run(arm):
            self.control.pair(arm,0)
            finished.append(arm)
            self.control.pair(arm,0,after=True)
        threads = [threading.Thread(target=run,args=(arm,)) for arm in pilot.ARMS]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
            self.assertFalse(thread.is_alive())
        self.assertEqual(set(finished),set(pilot.ARMS))
        self.assertEqual(set(self.control.snapshot()['finished']['0']),set(pilot.ARMS))

    def test_abort_unblocks_missing_partner_without_provider_retry(self):
        failures = []
        def run():
            try:
                self.control.pair(pilot.ARMS[0],0)
            except ValueError:
                failures.append(True)
        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(.12)
        self.control.halt('partner_failed')
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures,[True])

    def test_deadline_rejects_new_pair_with_under1860_seconds(self):
        with self.control.locked() as state:
            state['deadline']=time.time()+1859
        with self.assertRaises(ValueError):
            self.control.pair(pilot.ARMS[0],0)

    def test_api_guard_reserves660_seconds_for_owned_drain(self):
        with self.control.locked() as state:
            state['deadline']=time.time()+659
        with self.assertRaises(ValueError):
            self.control.admission(self.root/'late',{})
        self.assertEqual(self.control.snapshot()['creates'],{})

    def test_exact_approval_requires_both_egress_and_soft_overshoot(self):
        plan={'test':'no provider calls'}
        valid=dict(approved=True,proxy_overshoot_authorized=True,openai_payload_egress_approved=True,browser_use_payload_egress_approved=True,plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True,default=str).encode()).hexdigest())
        pilot.approval_check(plan,valid)
        for key in valid:
            changed=dict(valid)
            changed[key] = 'foreign' if key=='plan_sha256' else False
            with self.assertRaises(ValueError):
                pilot.approval_check(plan,changed)

    def test_no_implicit_existing_campaign_recovery(self):
        with self.assertRaises(ValueError):
            self.control.initialize(time.time()+14400)

    def test_actor_update_reserve_zero_and_positive_block_but_settle_reads_survive(self):
        calls=[]
        class Budget:
            def update(self,reserve=None,settle=None):
                calls.append((reserve,settle))
                return 'original'
        pilot.guard_budget_type(Budget,self.control)
        self.assertEqual(Budget().update(reserve=0),'original')
        self.control.halt('global_stop')
        for value in (0,1):
            with self.assertRaises(ValueError):
                Budget().update(reserve=value)
        self.assertEqual(Budget().update(settle=('known',1)),'original')
        self.assertEqual(Budget().update(),'original')
        self.assertEqual(calls,[(0,None),(None,('known',1)),(None,None)])

    def test_physical_reservation_guard_preserves_arguments_and_return_value(self):
        class Budget:
            def reserve(self,*a,**kw):
                return a,kw
        pilot.guard_budget_type(Budget,self.control)
        self.assertEqual(Budget().reserve('exact',role='model'),(('exact',),{'role':'model'}))
        self.control.halt('global_stop')
        with self.assertRaises(ValueError):
            Budget().reserve('exact',role='model')

    def test_pair_requires_90second_known_stopped_reconciliation(self):
        self.observe('1')
        directory=self.root/'one'
        with self.assertRaises(ValueError):
            pilot.Control.require_settled(self.control.snapshot())
        now=time.time()
        self.control.usage(directory,dict(id='owned',status='stopped',browserCost='1',proxyCost='0',proxyUsedMb='0'))
        with self.assertRaises(ValueError):
            self.control.reconciled(directory,now-89,now)
        self.control.reconciled(directory,now-90,now)
        pilot.Control.require_settled(self.control.snapshot())
        self.control.usage(directory,dict(id='owned',status='stopped',browserCost=None,proxyCost='0',proxyUsedMb='0'))
        with self.assertRaises(ValueError):
            self.control.reconciled(directory,now-90,now)

    def test_actual_profile_and_calendar_bound_startup_before_credentials(self):
        now=time.time()
        start=datetime.fromtimestamp(now-60).isoformat()
        end=datetime.fromtimestamp(now+14000).isoformat()
        raw=f'JobId=123 JobState=RUNNING NumCPUs=2 NumNodes=1 Restarts=0 ReqTRES=cpu=2,mem=16G,node=1 AllocTRES=cpu=2,mem=16G,node=1 StartTime={start} EndTime={end}'
        self.assertAlmostEqual(pilot.allocation_deadline(raw,now),now+14000,places=3)
        for bad in (raw.replace('JobState=RUNNING','JobState=PENDING'),raw.replace('NumNodes=1','NumNodes=2'),raw.replace('NumCPUs=2','NumCPUs=4'),raw.replace('mem=16G','mem=32G'),raw.replace('Restarts=0','Restarts=1'),raw.replace('AllocTRES=cpu=2,mem=16G,node=1','AllocTRES=cpu=2,mem=16G,node=1,gres/gpu=1'),raw.replace(end,datetime.fromtimestamp(now+14500).isoformat())):
            with self.assertRaises(ValueError):
                pilot.allocation_deadline(bad,now)
        with self.assertRaises(ValueError):
            pilot.allocation_deadline(raw,now,'foreign')

    def test_direct_run_requires_exact_approval_before_scheduler_or_output(self):
        output=self.root/'never-created'
        with patch.object(pilot.subprocess,'run',side_effect=AssertionError('scheduler must not run')):
            with self.assertRaises(ValueError):
                pilot.run({'output':str(output)},None)
        self.assertFalse(output.exists())

    def test_shared_home_boundary_rejects_before_filesystem_mutation(self):
        with self.assertRaises(ValueError):
            pilot.Control('/gpfs/home/zixianma/pilot-forbidden')

    def test_no_silent_local_resource_mode_after_approval(self):
        plan={'output':str(self.root/'no-local')}
        approval=dict(approved=True,proxy_overshoot_authorized=True,openai_payload_egress_approved=True,browser_use_payload_egress_approved=True,plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True,default=str).encode()).hexdigest())
        with patch.dict(pilot.os.environ,{},clear=True),patch.object(pilot.subprocess,'run',side_effect=AssertionError('must reject before scheduler')):
            with self.assertRaises(ValueError):
                pilot.run(plan,None,approval)
        self.assertFalse(Path(plan['output']).exists())

    def test_money_rejects_negative_nonfinite_bool_without_zero_imputation(self):
        for value in ('-1','NaN','Infinity',True):
            with self.assertRaises(Exception):
                pilot.money(value)

    def test_real_parser_path_values_are_serialized_in_durable_plan(self):
        path=self.root/'plan.json'
        pilot.write(path,{'args':{'runtime':self.root/'runtime'}})
        self.assertEqual(pilot.read(path)['args']['runtime'],str(self.root/'runtime'))


if __name__=='__main__':
    unittest.main()
