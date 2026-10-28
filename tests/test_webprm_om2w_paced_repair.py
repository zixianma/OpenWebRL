"""Provider-free repair authority checks and graceful signal admission/drain."""
from copy import deepcopy
import json
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import launch_webprm_om2w_table5 as launch


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')
    return str(path)


class PacedRepairAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='om2w-paced-authority-', dir='/tmp'))
        self.old=self.root/'original';self.new=self.root/'repair'
        self.relative_code=('openwebrl/webprm_concurrent_broker.py',
            'scripts/launch_webprm_om2w_table5.py','scripts/collect_webprm_om2w_table5.py',
            'openwebrl/webprm_worker.py')
        for package in (self.old,self.new):
            for rel in (*self.relative_code,'openwebrl/unregistered_science.py'):
                path=package/'source'/rel;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('# frozen '+rel+'\n')
            for name in ('tasks.json','cohorts.json','cohort.json','reference.json'):
                save(package/name,dict(fixed_input=name))
            template=dict(actors=['gpt-4o-2024-05-13'],selectors=['no_search','random5','arbiter7b','piotr_selectionarm'],
                generation=dict(temperature=1,max_tokens=2048),
                code_sha256={str(package/'source'/p):launch.sha(package/'source'/p) for p in self.relative_code})
            save(package/'worker-template.json',template)
        self.parent=dict(protocol='fixed-om2w-protocol',stage='rest200',stage_tasks=100,stage_cells=400,
            campaign_root=str(self.root/'campaign'),cohort_sha256='a'*64,protocol_sha256='b'*64,
            broker_routes=[dict(model='fixed',local=False)],cell_routes={'fixed':['policy']},
            env_file='/tmp/unused-provider-environment',models={'fixed':'fixed-model'},
            resources=dict(gpus=1,cpus=8,memory_gib=240,total_seconds=172800),
            limits=dict(api_usd=500,paid_requests=36400,local_requests=15022,total_requests=51422,
                maximum_attempts=2,max_workers=4),cell_ids=[f'fixed-cell-{i}' for i in range(400)],
            candidate_controls=dict(identity='om2w_random_candidate_control_100x4_v1'),
            scientific_scope='Preserve all fixed primary cells and raw outcomes.')
        paths=dict(task_file='tasks.json',cohorts_file='cohorts.json',cohort_path='cohort.json',
            reference_proposal='reference.json',worker_template='worker-template.json')
        self.parent.update({key:str(self.old/name) for key,name in paths.items()})
        source=list((self.old/'source').rglob('*.py'))+[self.old/name for name in paths.values()]
        self.parent['source_sha256']={str(p):launch.sha(p) for p in source}
        self.parent_path=self.old/'proposal.json';save(self.parent_path,self.parent)
        receipt=self.old/'user-approval-receipt.json';save(receipt,dict(quote='ok yes. use the cheaper setup'))
        self.approval=dict(status='approved',proposal_sha256=launch.digest(self.parent),
            resources=self.parent['resources'],limits=self.parent['limits'],api_usd_cap=500,
            account='zixianma',user_approval_quote='ok yes. use the cheaper setup',
            user_approval_receipt=str(receipt),user_approval_receipt_sha256=launch.sha(receipt))
        self.approval_path=self.old/'approved-request.json';save(self.approval_path,self.approval)
        self.proposal=deepcopy(self.parent)
        self.proposal.update({key:str(self.new/name) for key,name in paths.items()})
        # Only the allowed operational broker changes in this valid scaffold.
        broker=self.new/'source'/self.relative_code[0];broker.write_text('# new paced broker\n')
        template=launch.read(self.new/'worker-template.json')
        template['code_sha256'][str(broker)]=launch.sha(broker)
        save(self.new/'worker-template.json',template)
        self.proposal['source_sha256']={str(p):launch.sha(p) for p in (self.new/'source').rglob('*.py')}
        self.proposal['source_sha256'].update({str(self.new/name):launch.sha(self.new/name) for name in paths.values()})
        self.proposal['repair_of']=dict(reason='provider429_paid_dispatch_pacing',paid_dispatch_interval_seconds=3.0,
            proposal=dict(path=str(self.parent_path),sha256=launch.sha(self.parent_path)),
            approval=dict(path=str(self.approval_path),sha256=launch.sha(self.approval_path)))
        self.rebind_references()

    def rebind_references(self):
        for name in ('proposal','approval'):
            reference=self.proposal['repair_of'][name]
            reference['sha256']=launch.sha(reference['path'])
            self.proposal['source_sha256'][reference['path']]=reference['sha256']

    def test_valid_repair_preserves_exact_original_approval_and_frozen_science(self):
        before={p:p.read_bytes() for p in self.old.rglob('*') if p.is_file()}
        parent,template=launch.repair_parent(self.proposal)
        self.assertEqual(parent,self.parent)
        self.assertEqual(template,launch.read(self.old/'worker-template.json'))
        self.assertEqual(before,{p:p.read_bytes() for p in before})
        self.assertIsNone(launch.repair_parent(self.parent))

    def test_unknown_repair_reason_spacing_or_fields_cannot_borrow_approval(self):
        for key,value in (('reason','resample_invalids'),('paid_dispatch_interval_seconds',2.9),
                ('paid_dispatch_interval_seconds',True),('paid_dispatch_interval_seconds','3'),('extra',True)):
            proposal=deepcopy(self.proposal);proposal['repair_of'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):launch.repair_parent(proposal)

    def test_money_dispatch_concurrency_attempt_and_clock_extensions_are_rejected(self):
        for group,key in (('limits','api_usd'),('limits','paid_requests'),('limits','local_requests'),
                ('limits','total_requests'),('limits','maximum_attempts'),('limits','max_workers'),
                ('resources','gpus'),('resources','cpus'),('resources','memory_gib'),('resources','total_seconds')):
            proposal=deepcopy(self.proposal);proposal[group][key]+=1
            with self.subTest(group=group,key=key),self.assertRaisesRegex(ValueError,'resource authority'):
                launch.repair_parent(proposal)

    def test_cell_cohort_protocol_route_and_scientific_scope_drift_are_rejected(self):
        mutations=dict(cell_ids=self.proposal['cell_ids'][:-1],stage_tasks=99,stage_cells=399,
            cohort_sha256='c'*64,protocol_sha256='c'*64,protocol='new-protocol',stage='next90',
            campaign_root=str(self.root/'new-budget'),models={'fixed':'new-model'},
            broker_routes=[],cell_routes={},scientific_scope='Rerun failed primary cells.',
            candidate_controls={'identity':'different'},env_file='/tmp/different-provider-account')
        for key,value in mutations.items():
            proposal=deepcopy(self.proposal);proposal[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'frozen science or resource authority'):
                launch.repair_parent(proposal)

    def test_tampered_parent_reference_digest_and_source_binding_are_rejected(self):
        for name in ('proposal','approval'):
            for target in ('reference','source'):
                proposal=deepcopy(self.proposal);reference=proposal['repair_of'][name]
                if target=='reference':reference['sha256']='0'*64
                else:proposal['source_sha256'][reference['path']]='0'*64
                with self.subTest(name=name,target=target),self.assertRaisesRegex(ValueError,'source binding'):
                    launch.repair_parent(proposal)

    def test_parent_approval_without_matching_receipt_or_original_proposal_is_rejected(self):
        for key,value in (('account','vlm'),('proposal_sha256','0'*64),('user_approval_receipt_sha256','0'*64),
                ('api_usd_cap',501),('status','prepared'),('user_approval_quote','')):
            changed=deepcopy(self.approval);changed[key]=value;save(self.approval_path,changed);self.rebind_references()
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Original exact approval'):
                launch.repair_parent(self.proposal)
        save(self.approval_path,self.approval);self.rebind_references()

    def test_parent_frozen_source_tampering_is_rejected(self):
        (self.old/'source/openwebrl/webprm_worker.py').write_text('# changed old scientific worker\n')
        with self.assertRaisesRegex(ValueError,'Original frozen source changed'):launch.repair_parent(self.proposal)

    def test_task_input_contents_cannot_change_even_with_an_updated_source_hash(self):
        path=self.new/'tasks.json';save(path,dict(fixed_input='different'))
        self.proposal['source_sha256'][str(path)]=launch.sha(path)
        with self.assertRaisesRegex(ValueError,'frozen task/cohort/model inputs'):launch.repair_parent(self.proposal)

    def test_worker_generation_template_cannot_change(self):
        path=self.new/'worker-template.json';template=launch.read(path);template['generation']['max_tokens']=4096
        save(path,template);self.proposal['source_sha256'][str(path)]=launch.sha(path)
        with self.assertRaisesRegex(ValueError,'worker scientific template'):launch.repair_parent(self.proposal)

    def test_registered_scientific_worker_code_cannot_change(self):
        code=self.new/'source/openwebrl/webprm_worker.py';code.write_text('# changed scientific worker\n')
        path=self.new/'worker-template.json';template=launch.read(path);template['code_sha256'][str(code)]=launch.sha(code)
        save(path,template);self.proposal['source_sha256'][str(code)]=launch.sha(code)
        with self.assertRaisesRegex(ValueError,'scientific worker implementation'):launch.repair_parent(self.proposal)

    def test_unregistered_scientific_implementation_cannot_escape_freeze(self):
        path=self.new/'source/openwebrl/unregistered_science.py';path.write_text('# unrelated change\n')
        self.proposal['source_sha256'][str(path)]=launch.sha(path)
        with self.assertRaisesRegex(ValueError,'unrelated frozen implementation'):launch.repair_parent(self.proposal)


class SignalDrainTests(unittest.TestCase):
    def run_signal(self, sig, *, preexisting=None, signal_during_worker=False):
        root=Path(tempfile.mkdtemp(prefix='om2w-paced-signals-',dir='/tmp'))
        original_runner=launch.Runner;instances=[];stores=[]
        cells=[dict(cell_id='first',domain='fixed'),dict(cell_id='second',domain='fixed')]
        class FakeStore:
            def __init__(self,path,fixed):
                self.cells=fixed;self.states={c['cell_id']:'pending' for c in fixed};self.claimed=[];self.finished=[];stores.append(self)
            def __enter__(self):return self
            def __exit__(self,*_):pass
            def state(self,cell):return self.states[cell['cell_id']]
            def claim(self,cell):self.claimed.append(cell['cell_id']);self.states[cell['cell_id']]='attempted_without_outcome';return True
            def finish(self,cell,result):self.finished.append(cell['cell_id']);self.states[cell['cell_id']]='recorded'
        class FakeRunner:
            def __init__(self,proposal,template,context,output,scratch):
                self.proposal=proposal;self.context=context;self.output=Path(output);self.halt_reason=preexisting
                self.children=[];self.broker=None;self.port_lease=None;instances.append(self)
            async def run(self,cells,store):
                if not signal_during_worker:signal.raise_signal(sig)
                return await original_runner.run(self,cells,store)
            async def start_servers(self):return {}
            async def start(self,*_):pass
            def domain_ready(self,*_):return True
            def can_start(self,*_):return True
            async def run_one(self,cell):
                signal.raise_signal(sig)
                return dict(valid=True,status='saved_before_drain')
        handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            with patch.object(launch.os,'setsid'),patch.object(launch,'Runner',FakeRunner),patch.object(launch,'ClaimStore',FakeStore):
                launch._runner(dict(campaign_root=str(root/'campaign'),limits=dict(max_workers=1,min_start_seconds=0)),{},
                    dict(deadline=time.time()+60),root,root,cells)
        finally:
            for s,handler in handlers.items():signal.signal(s,handler)
        return launch.read(root/'queue-summary.json'),instances[0],stores[0]

    def test_sigterm_and_sigint_close_admission_without_admitting_primary_cells(self):
        for sig in (signal.SIGTERM,signal.SIGINT):
            with self.subTest(signal=sig):
                summary,runner,store=self.run_signal(sig)
                self.assertEqual(summary['reason'],'stop_admission');self.assertEqual(store.claimed,[])
                self.assertEqual(summary['pending'],['first','second'])
                self.assertEqual(runner.halt_reason,'supervisor_requested_stop')

    def test_signal_preserves_an_existing_failure_reason(self):
        summary,runner,store=self.run_signal(signal.SIGTERM,preexisting='preserve_original_budget_halt')
        self.assertEqual(runner.halt_reason,'preserve_original_budget_halt')
        self.assertEqual(summary['halt_reason'],'preserve_original_budget_halt')
        self.assertEqual(store.claimed,[])

    def test_signal_drains_owned_worker_before_stopping_the_next_admission(self):
        summary,runner,store=self.run_signal(signal.SIGTERM,signal_during_worker=True)
        self.assertEqual(store.claimed,['first']);self.assertEqual(store.finished,['first'])
        self.assertEqual(summary['admitted'],['first']);self.assertEqual(summary['closed'],['first'])
        self.assertEqual(summary['pending'],['second']);self.assertEqual(summary['remaining_owned_coroutines'],0)
        self.assertEqual(summary['halt_reason'],'supervisor_requested_stop')


if __name__=='__main__':unittest.main()
