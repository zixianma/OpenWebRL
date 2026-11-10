"""CPU assembly and ownership tests; no Ray/browser/API/GPU execution."""
import asyncio
from contextlib import ExitStack
from dataclasses import replace
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch, Mock, AsyncMock

from openwebrl import action_credit_bootstrap as boot
from openwebrl.action_credit_collect import FrozenActor
from openwebrl.action_credit_native_controller import _write, _json

_SPEC=importlib.util.spec_from_file_location('bootstrap_prep_fixture',Path(__file__).with_name('test_prepare_action_credit_pilots.py'))
fixture=importlib.util.module_from_spec(_SPEC);_SPEC.loader.exec_module(fixture)


class NativeDependencyGuardTests(unittest.TestCase):
    def setUp(self):
        # Keep these CPU fixtures; no production artifacts are deleted.
        self.root=Path(tempfile.mkdtemp(prefix='action-native-pins-',dir='/tmp'))
        self.runtime=self.root/'runtime'
        self.source=self.root/'source';self.source.mkdir()
        roots=[self.runtime/'src/Megatron-LM/megatron',
               self.runtime/'venv/lib/python3.12/site-packages/sglang',
               self.runtime/'venv/lib/python3.12/site-packages/ray']
        files={}
        for root in roots:
            root.mkdir(parents=True)
            path=root/'__init__.py';path.write_text('# CPU source fixture\n')
            files[str(path)]=dict(sha256=boot._hash(path),size_bytes=path.stat().st_size)
        self.external_file=roots[0]/'__init__.py'
        path=self.source/'action-credit-runtime-dependencies.json'
        path.write_text(json.dumps(dict(schema_version=1,runtime_root=str(self.runtime),
            source_root_paths=list(map(str,roots)),external_python_sources_and_distribution_records=files)))
        self.pin=dict(source_root=str(self.source),files={path.name:boot._hash(path)})

    def test_byte_mutation_and_new_native_module_are_rejected(self):
        with patch.object(boot,'RUNTIME_ROOT',self.runtime):
            proof=boot.verify_native_runtime_sources(self.pin)
            self.assertEqual(proof['external_files'],3)
            self.assertFalse(proof['physical_acceptance'])
            original=self.external_file.read_bytes()
            self.external_file.write_text('# changed CPU bytes\n')
            with self.assertRaisesRegex(ValueError,'changed'):
                boot.verify_native_runtime_sources(self.pin)
            self.external_file.write_bytes(original)
            self.external_file.with_name('new.py').write_text('# unpinned module\n')
            with self.assertRaisesRegex(ValueError,'membership changed'):
                boot.verify_native_runtime_sources(self.pin)

    def test_inventory_itself_must_be_source_bound(self):
        with patch.object(boot,'RUNTIME_ROOT',self.runtime):
            with self.assertRaisesRegex(ValueError,'inventory required'):
                boot.verify_native_runtime_sources(dict(source_root=str(self.source),files={}))
            (self.source/'action-credit-runtime-dependencies.json').write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'source changed'):
                boot.verify_native_runtime_sources(self.pin)

    def test_native_startup_checks_dependencies_before_native_acceptance_or_import(self):
        owner=SimpleNamespace(initial=lambda *a:None,after_update=lambda *a:None,final=lambda *a:None)
        with (patch.object(boot,'verify_native_runtime_sources',side_effect=ValueError('CPU changed source')) as guard,
              patch.object(boot,'validate_native_profile') as profile,
              patch.object(boot,'validate_native_acceptance') as acceptance):
            assembly=SimpleNamespace(source=self.pin)
            with self.assertRaisesRegex(ValueError,'CPU changed source'):
                boot.train(None,assembly,evaluation_owner=owner,
                    storage_admission=lambda *a:None,allocation_accounting=lambda:None)
            guard.assert_called_once_with(self.pin)
            profile.assert_not_called();acceptance.assert_not_called()


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture.PilotPreparationTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.arm='chain_outcome_grpo'
        bundles=self.fixture.build({self.arm:self.fixture.approval(self.arm)})
        fixture.prep.write_bundles(bundles,self.fixture.output)
        self.root=self.fixture.output/self.arm
        acceptance=self.fixture.root/'cpu-acceptance.json'
        acceptance.write_text(json.dumps(dict(verification_kind='cpu_mock',passed=True)))
        launch=dict(bundles[self.arm]['files']['launch.template.json'],
            native_acceptance=fixture.prep.reference(acceptance),execution_id='cpu-attempt',deadline_unix=4102444800.)
        path=self.fixture.root/'launch.json';path.write_bytes(fixture.prep.encoded(launch))
        self.launch_path=path
        self.assembly=boot.assemble(self.root,fixture.prep.reference(path))

    def args(self):
        return SimpleNamespace(**dict(self.assembly.worker['native_args'],
            actor_num_nodes=1,actor_num_gpus_per_node=8,num_gpus_per_node=8,rollout_num_gpus=8,rollout_num_gpus_per_engine=2,
            tensor_model_parallel_size=2,pipeline_model_parallel_size=1,context_parallel_size=1,
            global_batch_size=256,micro_batch_size=1,colocate=True,offload_train=False,offload_rollout=False,
            debug_train_only=False,debug_rollout_only=False,use_critic=False,use_opd=False,use_fault_tolerance=False,num_rollout=10,data_parallel_random_init=True,
            lr=1e-6,adam_beta1=.9,adam_beta2=.98,weight_decay=.1,lr_decay_style='constant',kl_coef=0.,
            qkv_format='thd',calculate_per_token_loss=False,entropy_coef=0,use_kl_loss=False,
            loss_type='custom_loss',custom_loss_function_path='openwebrl.action_credit_megatron.loss',
            hf_checkpoint=self.assembly.store.manifest.to_dict()['science']['checkpoint']['original_actor_path'],
            load=self.assembly.store.manifest.to_dict()['science']['checkpoint']['original_actor_path'],
            megatron_to_hf_mode='bridge',keep_old_actor=False,rollout_external=False,sglang_config=None,
            prefill_num_servers=None,wandb_project='openwebrl'))

    def test_approved_bundle_assembly_and_draft_rejection(self):
        self.assertEqual(self.assembly.store.manifest.to_dict()['arm_id'],self.arm)
        self.assertEqual(self.assembly.protocol.sha256,self.assembly.store.manifest.to_dict()['science']['judge']['http_protocol_sha256'])
        with self.assertRaisesRegex(ValueError,'drafts cannot launch'):
            boot.assemble(self.fixture.output/'gigpo',fixture.prep.reference(self.launch_path))

    def test_changed_frozen_launch_or_worker_file_rejected(self):
        value=json.loads(self.launch_path.read_text());value['worker_concurrency']=63
        self.launch_path.write_bytes(fixture.prep.encoded(value))
        with self.assertRaises(ValueError):
            boot.assemble(self.root,fixture.prep.reference(self.launch_path))
        worker=self.root/'worker.common.json';worker.write_text(worker.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'Prepared artifact differs'):
            boot.assemble(self.root,fixture.prep.reference(self.launch_path))

    def test_native_profile_never_requests_sixteen_or_silent_offloading(self):
        args=self.args();boot.validate_native_profile(args,self.assembly)
        args.colocate=False
        with self.assertRaisesRegex(ValueError,'colocate'):
            boot.validate_native_profile(args,self.assembly)
        args.colocate=True;args.offload_rollout=True
        with self.assertRaisesRegex(ValueError,'offload_rollout'):
            boot.validate_native_profile(args,self.assembly)
        with self.assertRaisesRegex(ValueError,'co-residency'):
            boot.validate_native_acceptance(self.assembly)

    def test_stale_load_and_alternate_engine_profiles_are_rejected(self):
        for key,value in (('load','/tmp/stale-native-checkpoint'),('megatron_to_hf_mode','raw'),
                          ('keep_old_actor',True),('rollout_external',True),('sglang_config','alternate.yaml'),
                          ('prefill_num_servers',2),('use_opd',True),('use_fault_tolerance',True)):
            with self.subTest(key=key):
                args=self.args();setattr(args,key,value)
                with self.assertRaises(ValueError):
                    boot.validate_native_profile(args,self.assembly)

    def test_missing_evaluation_or_storage_owner_rejected_before_startup(self):
        with self.assertRaisesRegex(ValueError,'evaluation owner'):
            boot.train(self.args(),self.assembly,evaluation_owner=None,
                       allocation_accounting=lambda:None,storage_admission=lambda *a:None)
        owner=SimpleNamespace(initial=lambda *a:None,after_update=lambda *a:None,final=lambda *a:None)
        with self.assertRaisesRegex(ValueError,'storage-admission'):
            boot.require_owners(owner,None)

    def test_both_actor_rpcs_delegate_exact_actor_and_command(self):
        class CPUActor:
            pass
        actor=boot.rpc_actor_class(CPUActor)();command={'cpu_fixture':True}
        with (patch('openwebrl.action_credit_native_controller.operation',return_value='native') as native,
              patch('openwebrl.action_credit_driver.native_driver_operation',return_value='driver') as driver):
            self.assertEqual(actor.action_credit_operation(command),'native')
            self.assertEqual(actor.action_credit_driver_operation(command),'driver')
            native.assert_called_once_with(actor,command);driver.assert_called_once_with(actor,command)

    def test_worker_factory_keeps_exact_inputs_direct_endpoint_and_64_workers(self):
        url='http://127.0.0.1:12345/generate'
        actor=FrozenActor(self.assembly.launch.initial_contract['policy_sha256'],'1',boot.digest('CPU restore'),
                          boot.digest('CPU freeze'),boot.digest(url))
        inputs=SimpleNamespace(sample=lambda request:{'cpu_fixture':True})
        with patch.object(boot,'IsolatedWorkerPool') as pool:
            factory=boot.worker_factory(self.assembly,inputs)
            factory(self.assembly.store,None,lambda request:actor,lambda a:url)
            kwargs=pool.call_args.kwargs
            self.assertEqual(kwargs['concurrency'],64)
            self.assertEqual(kwargs['template']['judge_protocol'],self.assembly.worker['judge_protocol'])
            native=kwargs['native_args_factory'](actor)
            self.assertEqual(native['sglang_router_port'],12345)
            self.assertEqual(native['max_steps'],15)
        expired=replace(self.assembly,launch=replace(self.assembly.launch,deadline_unix=1.))
        with self.assertRaisesRegex(ValueError,'expired'):
            boot.worker_factory(expired,inputs)(None,None,None,None)

    def storage_run(self,refuse_at=None):
        events=[]
        def storage(store,stage):
            events.append(('storage',stage))
            if sum(e[0]=='storage' for e in events)==refuse_at:
                raise ValueError('CPU storage refusal')
            return _write(self.fixture.root/f'storage-{len(events)}.json',dict(
                cpu_fixture_not_physical=True,passed=True,stage=stage,manifest_sha256=store.manifest.sha256,
                reservation=dict(manifest_sha256=store.manifest.sha256,store_root=str(store.root),epoch=time.time())))
        async def initial(*args):
            events.append(('initial',None))
            return _write(self.fixture.root/'initial-evaluation.json',{'cpu_only':True})
        async def final(*args):
            events.append(('final',None))
            return _write(self.fixture.root/'final-evaluation.json',{'cpu_only':True})
        async def pilot(*args,**kwargs):
            self.assertIs(kwargs['storage_admission'],storage)
            events.append(('pilot',None))
            return {'training_complete':True}
        owner=SimpleNamespace(initial=initial,after_update=AsyncMock(),final=final)
        with ExitStack() as stack:
            for name in ('validate_native_profile','validate_native_acceptance','audit_allocation'):
                stack.enter_context(patch.object(boot,name))
            stack.enter_context(patch.object(boot,'NativeTaskInputs',return_value=SimpleNamespace(request_pins=lambda *a:None)))
            stack.enter_context(patch.object(boot,'CollectionAcceptanceAuditor'))
            stack.enter_context(patch.object(boot,'worker_factory'))
            stack.enter_context(patch.object(boot,'run_pilot',pilot))
            try:
                result=asyncio.run(boot.run_with_models(None,None,None,self.assembly,evaluation_owner=owner,
                    allocation_accounting=lambda:None,storage_admission=storage))
            except BaseException as error:
                return events,error
            return events,result

    def test_shared0_and_final_evaluation_checks_enclose_owned_awaits(self):
        events,result=self.storage_run()
        self.assertIsInstance(result,dict)
        self.assertFalse(result['verified_complete'])
        self.assertEqual([e[0] for e in events],['storage','initial','storage','pilot','storage','final','storage'])
        boundaries=[json.loads(path.read_text()) for path in (self.assembly.store.root/'storage_boundaries').glob('*/*.json')]
        self.assertEqual({(b['phase'],b['position']) for b in boundaries},
            {('shared0-evaluation','before'),('shared0-evaluation','after'),
             ('final-evaluation-review','before'),('final-evaluation-review','after')})

    def test_shared0_pre_refusal_starts_no_evaluation_or_training(self):
        events,result=self.storage_run(refuse_at=1)
        self.assertIsInstance(result,ValueError)
        self.assertEqual([e[0] for e in events],['storage'])

    def test_shared0_post_refusal_retains_evaluation_and_blocks_training(self):
        events,result=self.storage_run(refuse_at=2)
        self.assertIsInstance(result,ValueError)
        self.assertEqual([e[0] for e in events],['storage','initial','storage'])
        self.assertTrue((self.fixture.root/'initial-evaluation.json').is_file())

    def test_final_post_refusal_prevents_success_result(self):
        events,result=self.storage_run(refuse_at=4)
        self.assertIsInstance(result,ValueError)
        self.assertEqual([e[0] for e in events],['storage','initial','storage','pilot','storage','final','storage'])


class StorageFinalizationTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='action-storage-close-',dir='/tmp'))
        self.store=SimpleNamespace(root=self.root,manifest=SimpleNamespace(sha256=boot.digest('CPU only')))
        self.events=[]

    def provider(self,store,stage):
        self.events.append('storage')
        return _write(self.root/'storage.json',dict(cpu_fixture_not_physical=True,passed=True,stage=stage,
            manifest_sha256=store.manifest.sha256,reservation=dict(manifest_sha256=store.manifest.sha256,
            store_root=str(store.root),epoch=time.time())))

    def close(self,owner,tracking,provider=None,primary=None):
        boot.close_owned_writers(owner,tracking,None,self.store,provider or self.provider,
                                 'CPU startup','native-startup',primary)

    def test_final_check_runs_only_after_both_owners_return(self):
        owner=SimpleNamespace(close=lambda:self.events.append('native-closed'))
        self.close(owner,lambda _:self.events.append('tracking-closed'))
        self.assertEqual(self.events,['native-closed','tracking-closed','storage'])
        proof=json.loads(next((self.root/'storage_boundaries').glob('*/*.json')).read_text())
        self.assertEqual(proof['phase'],'owned-writers-closed')
        self.assertEqual(proof['position'],'after')

    def test_incomplete_shutdown_still_drains_tracking_and_never_claims_closed(self):
        def failed():
            self.events.append('native-failed')
            raise RuntimeError('CPU incomplete native closure')
        with self.assertRaisesRegex(RuntimeError,'incomplete native'):
            self.close(SimpleNamespace(close=failed),lambda _:self.events.append('tracking-closed'))
        self.assertEqual(self.events,['native-failed','tracking-closed','storage'])
        proof=json.loads(next((self.root/'storage_boundaries').glob('*/*.json')).read_text())
        self.assertEqual(proof['phase'],'owned-writers-incomplete')
        self.assertEqual(proof['position'],'after-error')

    def test_primary_startup_failure_survives_both_cleanup_and_storage_refusal(self):
        class Primary(BaseException):pass
        primary=Primary('CPU startup error')
        def failed():raise RuntimeError('CPU native cleanup failure')
        def tracking(_):
            self.events.append('tracking-attempted')
            raise RuntimeError('CPU tracking failure')
        def refused(*args):
            self.events.append('storage-attempted')
            raise ValueError('CPU quota refusal')
        self.close(SimpleNamespace(close=failed),tracking,refused,primary)
        self.assertEqual(self.events,['tracking-attempted','storage-attempted'])
        self.assertTrue(any('native cleanup' in note for note in primary.__notes__))
        self.assertTrue(any('tracking failure' in note for note in primary.__notes__))
        self.assertTrue(any('quota refusal' in note for note in primary.__notes__))
        proof=json.loads(next((self.root/'storage_boundaries').glob('*/*.json')).read_text())
        self.assertFalse(proof['passed'])
        self.assertEqual(proof['stage'],'native-startup')

    def test_final_storage_refusal_cannot_be_successful_shutdown_return(self):
        owner=SimpleNamespace(close=lambda:self.events.append('native-closed'))
        def refused(*args):raise ValueError('CPU post-shutdown refusal')
        with self.assertRaisesRegex(ValueError,'post-shutdown refusal'):
            self.close(owner,lambda _:self.events.append('tracking-closed'),refused)
        self.assertEqual(self.events,['native-closed','tracking-closed'])


class NativeOwnershipTests(unittest.TestCase):
    def actor(self,name,events):
        def method(which):
            return SimpleNamespace(remote=lambda:(events.append(('dispatch',name,which)) or (name,which)))
        return SimpleNamespace(name=name,action_credit_shutdown=method('shutdown'),finish_tracking=method('tracking'))

    def ray(self,events,fail=None):
        def get(ref,timeout):
            events.append(('await',ref))
            if ref[0]==fail:raise TimeoutError('CPU simulated child failure')
            return True
        return SimpleNamespace(get=get,kill=lambda actor,**kw:events.append(('kill',actor.name,kw)))

    def test_normal_cleanup_awaits_owned_children_then_kills_and_removes_only_own_pg(self):
        events=[];ray=self.ray(events)
        owner=boot.NativeOwnership(ray,lambda pg:events.append(('remove',pg)))
        owner.actors=[('train',self.actor('rank0',events)),('manager',self.actor('manager',events))]
        owner.placement_groups=['owned-pg']
        owner.close();owner.close()
        self.assertEqual([e[1] for e in events if e[0]=='kill'],['manager','rank0'])
        self.assertEqual([e for e in events if e[0]=='remove'],[('remove','owned-pg')])
        self.assertLess(max(i for i,e in enumerate(events) if e[0]=='await'),
                        min(i for i,e in enumerate(events) if e[0]=='kill'))
        self.assertTrue(all(e[2]=={'no_restart':True} for e in events if e[0]=='kill'))

    def test_failed_shutdown_still_drains_siblings_and_releases_own_resources(self):
        events=[];ray=self.ray(events,fail='broken')
        owner=boot.NativeOwnership(ray,lambda pg:events.append(('remove',pg)))
        owner.actors=[('engine',self.actor('good',events)),('engine',self.actor('broken',events))]
        owner.placement_groups=['owned']
        with self.assertRaisesRegex(RuntimeError,'incomplete'):
            owner.close()
        self.assertEqual({e[1][0]for e in events if e[0]=='await'},{'good','broken'})
        self.assertEqual({e[1]for e in events if e[0]=='kill'},{'good','broken'})
        self.assertIn(('remove','owned'),events)
        with self.assertRaisesRegex(RuntimeError,'incomplete'):
            owner.close()  # A repeated check cannot certify an earlier failure.

    def test_partial_startup_records_handle_before_first_rpc_and_pg_before_ready(self):
        events=[];ray=self.ray(events);owner=boot.NativeOwnership(ray,events.append)
        actor=self.actor('rank0',events)
        builder=SimpleNamespace(options=lambda **kw:builder,remote=lambda *a:actor)
        handle=boot._OwnedBuilder(builder,owner,'train').options(num_gpus=.4).remote(8,0)
        self.assertIs(handle,actor);self.assertEqual(owner.actors,[('train',actor)])
        pg=owner.placement_factory(lambda *a,**kw:'pg')([])
        self.assertEqual(pg,'pg');self.assertEqual(owner.placement_groups,['pg'])
        # This is the state retained when native get_master_addr/startup fails.
        owner.close()
        self.assertIn(('kill','rank0',{'no_restart':True}),events)
        self.assertIn('pg',events)

    def test_successfully_reaped_info_actor_is_not_killed_twice(self):
        events=[];ray=self.ray(events);owner=boot.NativeOwnership(ray)
        actor=self.actor('info',events);owner.actors=[('info',actor)]
        boot._OwnedRay(ray,owner,'info').kill(actor)
        owner.close()
        self.assertEqual(len([e for e in events if e[0]=='kill']),1)

    def test_capture_process_precedes_failed_readiness_and_restores_factory(self):
        events=[]
        process=SimpleNamespace(pid=17,is_alive=Mock(side_effect=[True,False]),
                                join=lambda **kw:events.append(('join',kw)))
        factory=Mock(return_value=process)
        module=SimpleNamespace(multiprocessing=SimpleNamespace(Process=factory))
        owned=[]
        with self.assertRaisesRegex(RuntimeError,'health failed'):
            with boot._capture_processes(module,owned):
                child=module.multiprocessing.Process(target='CPU fake')
                self.assertEqual(owned,[child])
                raise RuntimeError('health failed')
        self.assertIs(module.multiprocessing.Process,factory)
        boot._stop_processes(owned,lambda pid:events.append(('kill-tree',pid)))
        self.assertEqual(events[0],('kill-tree',17))

    def fake_modules(self,engine,rollout=None,ray=None):
        names=['slime','slime.backends','slime.backends.sglang_utils',
               'slime.backends.sglang_utils.sglang_engine','slime.ray','slime.ray.rollout']
        modules={name:ModuleType(name) for name in names}
        modules[names[3]]=engine
        if rollout is not None:modules[names[5]]=rollout
        for name,module in modules.items():
            parent,_,child=name.rpartition('.')
            if parent:setattr(modules[parent],child,module)
        if ray is not None:modules['ray']=ray
        return patch.dict(sys.modules,modules)

    def test_actual_engine_wrapper_cleans_process_if_native_readiness_raises(self):
        events=[];alive=[True]
        process=SimpleNamespace(pid=37,is_alive=lambda:alive[0],join=lambda **kw:None)
        module=ModuleType('slime.backends.sglang_utils.sglang_engine')
        module.multiprocessing=SimpleNamespace(Process=lambda **kw:process)
        module.kill_process_tree=lambda pid:(events.append(pid),alive.__setitem__(0,False))
        class Base:
            def _init_normal(self,args):
                child=module.multiprocessing.Process(target='CPU fixture')
                self.allocated=child
                raise RuntimeError('native readiness failure')
        actor=boot.owned_engine_class(Base)()
        with self.fake_modules(module):
            with self.assertRaisesRegex(RuntimeError,'native readiness failure'):
                actor._init_normal({})
            self.assertEqual(events,[37])
            self.assertTrue(actor.action_credit_shutdown())
            self.assertEqual(events,[37])

    def test_actual_manager_wrapper_owns_engine_during_partial_constructor_failure(self):
        events=[];ray=self.ray(events);alive=[True]
        process=SimpleNamespace(pid=38,is_alive=lambda:alive[0],join=lambda **kw:None)
        engine_module=ModuleType('slime.backends.sglang_utils.sglang_engine')
        engine_module.kill_process_tree=lambda pid:(events.append(('router-kill',pid)),alive.__setitem__(0,False))
        module=ModuleType('slime.ray.rollout')
        module.multiprocessing=SimpleNamespace(Process=lambda **kw:process)
        module.ray=ray;module.SGLangEngine=type('CPUBaseEngine',(),{})
        actor=self.actor('partial-engine',events)
        def remote(cls):
            builder=SimpleNamespace(remote=lambda *a,**kw:actor,options=lambda **kw:builder)
            return builder
        ray.remote=remote
        class Base:
            def __init__(self):
                module.multiprocessing.Process(target='CPU router')
                module.ray.remote(module.SGLangEngine).options(num_gpus=.2).remote()
                raise RuntimeError('partial native manager failure')
            def dispose(self):events.append(('dispose',))
        with self.fake_modules(engine_module,module,ray):
            with self.assertRaisesRegex(RuntimeError,'partial native manager failure'):
                boot.owned_manager_class(Base)()
        self.assertIn(('await',('partial-engine','shutdown')),events)
        self.assertIn(('kill','partial-engine',{'no_restart':True}),events)
        self.assertIn(('router-kill',38),events)
        self.assertIs(module.ray,ray)

    def test_unexpected_dead_parent_is_not_signalled_or_certified_clean(self):
        process=SimpleNamespace(pid=17,is_alive=lambda:False)
        kill=Mock()
        with self.assertRaisesRegex(RuntimeError,'incomplete'):
            boot._stop_processes([process],kill)
        kill.assert_not_called()
        self.assertFalse(getattr(process,'_action_credit_cleanup_complete',False))


if __name__=='__main__':
    unittest.main()
