import copy
import os
from pathlib import Path
import tempfile
import unittest

from openwebrl import webprm_map_startup_validation as m


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='map-validation-fixture-',dir='/tmp')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        plan=m.plan();self.proposal=dict(protocol=m.PROTOCOL,resource_plan_sha256=m.PLAN_SHA,resources=plan['resources'],storage=plan['storage'])
        self.registry=dict(proposal_sha256=m.json_digest(self.proposal),job_ids=['2'])
        self.approval=dict(status='approved',proposal_sha256=m.json_digest(self.proposal),job_id='2',attempt_job_ids=['2'],exact_user_approval='fixture only')
        self.current=self.snapshot(2,'RUNNING',1000,15403,240)
        self.context=self.context_for()

    def snapshot(self,job,state,start,end,minutes):
        row=dict(job_id=job,user_id=os.getuid(),restart_cnt=0,account='zixianma',partition='gpu-h200',qos='normal',
                 job_state=[state],tres_alloc_str='cpu=8,mem=64G,node=1')
        for key,value in dict(cpus=8,node_count=1,memory_per_node=65536,start_time=start,end_time=end,time_limit=minutes).items():
            row[key]=dict(set=True,infinite=False,number=value)
        return dict(errors=[],jobs=[row])

    def context_for(self,prior=()):
        return m.execution_context(self.proposal,self.approval,self.current,list(prior),self.registry,
            now=1010,job_id='2',controller_pid=20,controller_start_ticks=200,hostname='fixture-host')

    def test_scheduler_grace_is_clamped_without_extending_fixed_caps(self):
        self.assertEqual(self.context['allocation_deadline'],15400)
        self.assertEqual(self.context['work_deadline'],14500)
        self.assertEqual(self.context['max_browser_contexts'],6)
        self.assertEqual(self.context['model_calls'],0)
        self.assertFalse(self.context['production_adapter_enabled'])

    def test_missing_approval_changed_resources_and_gpu_fail(self):
        old=copy.deepcopy(self.approval);self.approval['status']='pending'
        with self.assertRaises(m.Closed):self.context_for()
        self.approval=old;self.current['jobs'][0]['tres_alloc_str']+=',gres/gpu=1'
        with self.assertRaises(m.Closed):self.context_for()

    def test_failed_attempt_time_cannot_be_reset_or_overlapped(self):
        self.registry['job_ids']=['1','2'];self.approval['attempt_job_ids']=['1','2']
        prior=self.snapshot(1,'FAILED',100,900,240)
        with self.assertRaisesRegex(m.Closed,'four-hour'):self.context_for([prior])
        self.current=self.snapshot(2,'RUNNING',1000,14500,225)
        context=self.context_for([prior]);self.assertEqual(context['prior_elapsed_seconds'],800)
        prior=self.snapshot(1,'FAILED',100,1001,240)
        self.current=self.snapshot(2,'RUNNING',1000,14200,220)
        with self.assertRaisesRegex(m.Closed,'overlap'):self.context_for([prior])

    def test_attempt_omission_or_nonterminal_predecessor_fails(self):
        self.registry['job_ids']=['1','2'];self.approval['attempt_job_ids']=['1','2']
        with self.assertRaises(m.Closed):self.context_for()
        with self.assertRaises(m.Closed):self.context_for([self.snapshot(1,'RUNNING',100,900,240)])

    def ownership(self,role='tile'):
        rows={20:dict(start_ticks=200,state='S',ppid=1),21:dict(start_ticks=210,state='S',ppid=20),22:dict(start_ticks=220,state='S',ppid=21)}
        return m.owned_process(self.context,22,role,process_reader=rows.__getitem__,namespace_reader=lambda p:'net:[42]')

    def probe(self,stdout=b'postgres (PostgreSQL) 15.8\n',role='tile',argv=None):
        return m.probe_record(self.context,self.ownership(role),argv or ['/usr/lib/postgresql/15/bin/postgres','--version'],stdout,0,
            dict(remaining_owned_processes={},owned_pids=[22],child_exitcode=0),image_config_sha256='a'*64)

    def test_process_ownership_rejects_foreign_or_reused_controller(self):
        self.assertEqual(self.ownership()['pid'],22)
        for parent,ticks in [(1,200),(20,201)]:
            rows={20:dict(start_ticks=ticks,state='S',ppid=1),22:dict(start_ticks=220,state='S',ppid=parent)}
            with self.assertRaises(m.Closed):m.owned_process(self.context,22,'tile',process_reader=rows.__getitem__,namespace_reader=lambda p:'net:[42]')

    def inventory(self,version=b'15\n'):
        return dict(schema='map_selected_inventory_v1',content_sha256='b'*64,captured={'data/PG_VERSION':dict(raw=version,sha256=m.sha(version))})

    def test_actual_postgres_major_requires_both_cluster_and_native_probe(self):
        proof=m.postgres_compatibility(self.inventory(),'data/PG_VERSION',self.probe(),self.context,'tile','a'*64,'/usr/lib/postgresql/15/bin/postgres')
        self.assertEqual(proof['data_major'],'15');self.assertFalse(proof['native_database_startup_verified'])
        with self.assertRaisesRegex(m.Closed,'mismatch'):
            m.postgres_compatibility(self.inventory(b'14\n'),'data/PG_VERSION',self.probe(),self.context,'tile','a'*64,'/usr/lib/postgresql/15/bin/postgres')

    def test_native_probe_tamper_or_missing_closed_owner_fails(self):
        for key,value in [('stdout','postgres (PostgreSQL) 14.8'),('closure',{}),('image_config_sha256','b'*64)]:
            probe=self.probe();probe[key]=value
            with self.subTest(key=key),self.assertRaises(m.Closed):m.postgres_compatibility(self.inventory(),'data/PG_VERSION',probe,self.context,'tile','a'*64,'/usr/lib/postgresql/15/bin/postgres')

    def test_osrm_exact_version_does_not_claim_mld_load(self):
        probe=self.probe(b'v5.27.1\n','osrm-car',['/usr/local/bin/osrm-routed','--version'])
        self.assertFalse(m.osrm_version(probe,self.context,'osrm-car','a'*64)['mld_data_load_verified'])
        probe=self.probe(b'v5.28.0\n','osrm-car',['/usr/local/bin/osrm-routed','--version'])
        with self.assertRaises(m.Closed):m.osrm_version(probe,self.context,'osrm-car','a'*64)

    def reset(self):
        states=[dict(path=str(self.root/'scratch'/name),instance_id=name,network_namespace='net:['+str(i)+']',fresh_claim=True,
                     controller_pid=20,base_identity_sha256='a'*64) for i,name in enumerate(('a','b'))]
        return dict(protocol='map_native_reset_controls_v1',context_sha256=m.json_digest(self.context),base_identity_sha256='a'*64,
            query_contract_sha256='b'*64,base_path=str(self.root/'base'),scratch_root=str(self.root/'scratch'),state_a=states[0],state_b=states[1],
            native_observations={key:dict(query_fingerprints={'objects':fingerprint*64},mutation_marker_count=count)
                for key,fingerprint,count in [('a_before','c',0),('a_after_mutation','d',1),('b_fresh','c',0)]},
            closures={key:dict(remaining_owned_processes={}) for key in ('state_a','state_b','shared_backends')})

    def reset_check(self,proof):return m.reset_controls(self.context,proof,base_identity_sha256='a'*64,query_contract_sha256='b'*64)

    def test_mutation_then_pristine_fresh_reset_stays_unready(self):
        result=self.reset_check(self.reset());self.assertFalse(result['ready_to_host']);self.assertFalse(result['production_adapter_enabled'])

    def test_reset_rejects_leak_malformed_change_shared_state_and_unclosed_service(self):
        for mutation in ('leak','malformed','path','open'):
            proof=self.reset()
            if mutation=='leak':proof['native_observations']['b_fresh']['mutation_marker_count']=1
            elif mutation=='malformed':proof['native_observations']['a_after_mutation']['query_fingerprints']=None
            elif mutation=='path':proof['state_b']['path']=proof['state_a']['path']+'/nested'
            else:proof['closures']['shared_backends']['remaining_owned_processes']={'22':220}
            with self.subTest(mutation=mutation),self.assertRaises(m.Closed):self.reset_check(proof)


if __name__=='__main__':unittest.main()
