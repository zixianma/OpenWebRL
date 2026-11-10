#!/usr/bin/env python3
"""Concrete native startup/assembly, without scheduler or evaluation defaults.

``--inspect`` reads an approved preparation and an actual DriverLaunch reference;
it never initializes Ray, opens browsers or calls APIs. Native launchers may
call ``train(args, assembly, ...)`` after their ordinary Slime argument parsing.
That function uses create_placement_groups/create_rollout_manager/
create_training_models, installs both RPCs on all native Ray actors, and owns
the pilot and worker lifetimes.

The caller must supply a real evaluation owner (initial, after_update, final)
and a live storage-admission callback. No default implementations, synthetic
acceptance or completion claim are provided. The initial shared full300 cohort
and the final iter10 cohort remain the evaluation owner's responsibility,
including approval, saved artifacts, independent review and crash recovery.
Existing TP2/DP4 diagnostics do not establish simultaneous actor/SGLang memory
residency; an explicit physical co-residency receipt is required before startup.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from contextlib import contextmanager, ExitStack
import inspect
import json
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
RUNTIME_ROOT=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
if str(REPO) not in sys.path:
    sys.path.insert(0,str(REPO))

from openwebrl.action_credit import digest
from openwebrl.action_credit_acceptance import CollectionAcceptanceAuditor, _sources, audit_allocation
from openwebrl.action_credit_driver import DriverLaunch, _contract, run_pilot, storage_boundary, storage_guarded
from openwebrl.action_credit_judge import JudgeHTTPProtocol
from openwebrl.action_credit_native_controller import _hash, _json, validator_source_sha256
from openwebrl.action_credit_process import IsolatedWorkerPool
from openwebrl.action_credit_state import CampaignManifest, CampaignStore, _safe_path
from openwebrl.action_credit_task_inputs import NativeTaskInputs, direct_engine_args
from openwebrl.action_credit_worker import _approval, verify_worker_environment


def require(condition,message):
    if not condition:
        raise ValueError(message)


def artifact(path):
    path=_safe_path(path)
    return dict(path=str(path),sha256=_hash(path))


def verify_native_runtime_sources(source):
    """Recheck source-bound native dependencies before native package imports.

    This guard reads only. It does not grant physical acceptance. Installed
    distribution RECORD files identify binary packages; their large binaries
    are not independently rehashed by the CPU preparation workflow.
    """
    root=Path(source['source_root']).resolve()
    files=source['files']
    dependency_name='action-credit-runtime-dependencies.json'
    require(dependency_name in files,'Source-bound external runtime inventory required')
    for relative,expected in files.items():
        rel=Path(relative)
        path=root/rel
        require(not rel.is_absolute() and '..' not in rel.parts and not path.is_symlink()
                and path.resolve().is_relative_to(root),'Prepared native source escaped or moved')
        require(_hash(path)==expected,'Prepared native source changed: '+relative)
    inventory=json.loads((root/dependency_name).read_text())
    runtime=RUNTIME_ROOT.resolve()
    require(inventory.get('schema_version')==1 and inventory.get('runtime_root')==str(runtime),
            'External runtime root differs from approved runtime')
    site=runtime/'venv/lib/python3.12/site-packages'
    roots=[runtime/'src/Megatron-LM/megatron',site/'sglang',site/'ray']
    require(inventory.get('source_root_paths')==list(map(str,roots)),
            'Expected native Megatron/SGLang/Ray source roots required')
    expected_files=inventory.get('external_python_sources_and_distribution_records')
    require(isinstance(expected_files,dict) and expected_files,'External runtime inventory is empty')
    for package in roots:
        actual={str(p) for p in package.rglob('*.py')}
        pinned={name for name in expected_files if Path(name).is_relative_to(package)}
        require(actual and actual==pinned,'External native package membership changed: '+str(package))
    for name,expected in expected_files.items():
        path=Path(name)
        require(path.is_absolute() and not path.is_symlink() and path.resolve().is_relative_to(runtime),
                'External native dependency escaped or moved')
        require(path.stat().st_size==expected['size_bytes'] and _hash(path)==expected['sha256'],
                'External native source/distribution changed: '+name)
    return dict(source_files=len(files),external_files=len(expected_files),
                physical_acceptance=False,installed_binary_payloads_rehashed=False)


@dataclass(frozen=True)
class Assembly:
    root: Path
    launch: DriverLaunch
    store: CampaignStore
    worker: dict
    source: dict
    source_reference: dict
    environment: dict
    accounting: dict
    protocol: JudgeHTTPProtocol


def assemble(bundle_directory,launch_reference):
    """Validate exact prepared bytes and real launch bindings; no initialization."""
    root=_safe_path(bundle_directory)
    receipt=json.loads((root/'PREPARED.json').read_text())
    require(receipt.get('ledger_created') is True
            and receipt.get('bundle_status')=='approved_manifest_pending_physical_acceptance',
            'Approved preparation with an existing ledger required; drafts cannot launch')
    for name,sha in receipt['files'].items():
        path=_safe_path(root/name)
        require(root in path.parents and _hash(path)==sha,'Prepared artifact differs: '+name)
    launch=DriverLaunch(**_json(launch_reference));launch.validate()
    template=json.loads((root/'launch.template.json').read_text())
    actual=dict(launch.__dict__)
    for key,value in template.items():
        if key not in ('native_acceptance','execution_id','deadline_unix'):
            require(actual[key]==value,'Runtime launch changed a frozen field: '+key)
    manifest=CampaignManifest.from_dict(_json(launch.manifest))
    require(manifest.sha256==receipt['manifest_sha256'],'Manifest differs from preparation')
    store=CampaignStore(launch.store_root,manifest)
    worker=json.loads((root/'worker.common.json').read_text())
    protocol=JudgeHTTPProtocol(**worker['judge_protocol'])
    _approval(store,worker['approval_reference'],protocol)
    require(manifest.to_dict()['science']['judge'].get('http_protocol_sha256')==protocol.sha256,
            'HTTP protocol differs from the approved manifest')
    source_reference=artifact(root/'source-inventory.json')
    source=_sources(store,source_reference)
    return Assembly(root,launch,store,worker,source,source_reference,
                    json.loads((root/'environment.json').read_text()),
                    json.loads((root/'resource-accounting.json').read_text()),protocol)


def validate_native_profile(args,assembly):
    from openwebrl.action_credit_megatron import check_topology
    expected=dict(actor_num_nodes=1,actor_num_gpus_per_node=8,num_gpus_per_node=8,rollout_num_gpus=8,
        rollout_num_gpus_per_engine=2,tensor_model_parallel_size=2,pipeline_model_parallel_size=1,
        context_parallel_size=1,global_batch_size=256,micro_batch_size=1,colocate=True,
        offload_train=False,offload_rollout=False,debug_train_only=False,debug_rollout_only=False,
        use_critic=False,use_opd=False,use_fault_tolerance=False,num_rollout=10,data_parallel_random_init=True,
        lr=1e-6,adam_beta1=.9,adam_beta2=.98,weight_decay=.1,lr_decay_style='constant',kl_coef=0.)
    for key,value in expected.items():
        require(getattr(args,key,None)==value,'Native profile must explicitly pin '+key+'='+str(value))
    check_topology(args,cp=1,pp=1)
    require(Path(args.hf_checkpoint).resolve()==Path(assembly.store.manifest.to_dict()['science']['checkpoint']['original_actor_path']).resolve(),
            'Native startup actor differs from original initialization')
    original=Path(args.hf_checkpoint).resolve()
    require(getattr(args,'load',None) is not None and Path(args.load).resolve()==original,
            'Native startup load must be the original HF actor; recovery is owned by the controller')
    require(getattr(args,'megatron_to_hf_mode',None)=='bridge'
            and getattr(args,'keep_old_actor',None) is False,
            'Original HF bridge loading without an old actor is required')
    require(getattr(args,'rollout_external',None) is False
            and getattr(args,'sglang_config',None) is None
            and getattr(args,'prefill_num_servers',None) is None,
            'Exactly four local TP2 engines require the native single-server profile')
    # Generic colocate defaults turn offloading on. Never silently undo them or
    # request16 GPUs when the approved physical allocation contains eight.
    require(getattr(args,'wandb_project',None)=='openwebrl','Actual training tracking must use openwebrl')
    for key,value in assembly.worker['native_args'].items():
        if key in ('sglang_router_ip','sglang_router_port'):
            continue  # Direct verified endpoints are assigned after engine discovery.
        require(hasattr(args,key) and getattr(args,key)==value,'Native runtime argument differs from preparation: '+key)


def validate_native_acceptance(assembly):
    evidence=_json(assembly.launch.native_acceptance)
    required=('actual_model','full_response_masks','tp_dp_gradient_parity','ppo2_windows',
              'native_checkpoint_restore','per_rank_rng_restore','native_controller_integration',
              'colocated_actor_rollout_no_offload')
    require(evidence.get('passed') is True and evidence.get('verification_kind')=='production_runtime'
        and evidence.get('manifest_sha256')==assembly.store.manifest.sha256
        and evidence.get('validator_source_sha256')==validator_source_sha256()
        and evidence.get('topology')==dict(world_size=8,tp=2,dp=4)
        and evidence.get('rollout_engines')==4 and evidence.get('rollout_tp')==2
        and all(evidence.get(k) is True for k in required),
        'Physical source/manifest-bound TP2DP4 and four-engine nonoffloaded co-residency acceptance required')


def rpc_actor_class(base):
    """Add both real RPC entrypoints before Ray serializes each native actor."""
    class ActionCreditActor(base):
        def action_credit_operation(self,command):
            from openwebrl.action_credit_native_controller import operation
            return operation(self,command)

        def action_credit_driver_operation(self,command):
            from openwebrl.action_credit_driver import native_driver_operation
            return native_driver_operation(self,command)
    return ActionCreditActor


@contextmanager
def _patched(obj,name,value):
    original=getattr(obj,name)
    setattr(obj,name,value)
    try:
        yield
    finally:
        setattr(obj,name,original)


class _OwnedBuilder:
    """Record a returned actor immediately, before any startup RPC can fail."""
    def __init__(self,builder,owner,kind):
        self.builder,self.owner,self.kind=builder,owner,kind

    def options(self,*args,**kwargs):
        return _OwnedBuilder(self.builder.options(*args,**kwargs),self.owner,self.kind)

    def remote(self,*args,**kwargs):
        actor=self.builder.remote(*args,**kwargs)
        self.owner.actors.append((self.kind,actor))
        return actor


class _OwnedRay:
    """Intercept creations in the owning process, never transport its owner.

    Native actor methods can capture a module-global ``ray`` while this proxy
    is installed. Ray cloudpickles those globals. The receiving process must
    recover its ordinary Ray module; parent handles and cleanup ownership stay
    with the process that actually created them.
    """
    def __init__(self,ray,owner,kind):
        self.ray,self.owner,self.kind=ray,owner,kind

    def __getattr__(self,name):
        # Inspectors may query an instance before __init__ has populated it.
        return getattr(object.__getattribute__(self,'ray'),name)

    def __reduce__(self):
        from importlib import import_module
        return import_module,('ray',)

    def kill(self,actor,*args,**kwargs):
        result=self.ray.kill(actor,*args,**kwargs)
        self.owner.actors[:]=[(kind,value) for kind,value in self.owner.actors if value is not actor]
        return result

    def remote(self,*args,**kwargs):
        if args:
            return _OwnedBuilder(self.ray.remote(*args,**kwargs),self.owner,self.kind)
        return lambda cls:_OwnedBuilder(self.ray.remote(**kwargs)(cls),self.owner,self.kind)


class _OwnedTrainRay(_OwnedRay):
    """Apply an RPC subclass at ray.remote, preserving the native class symbol.

    The canonical native class must remain importable by reference. Rebinding
    its module symbol before cloudpickle serializes inherited decorated methods
    can detach their zero-argument super() __class__ cell from the deserialized
    instance. This adapter is driver-only; inherited __reduce__ sends ordinary
    Ray to workers, never the owner's handles or this substitution table.
    """
    def __init__(self,ray,owner,base,subclass_factory):
        super().__init__(ray,owner,'train')
        self.base=base
        self.subclass=subclass_factory(base)
        require(isinstance(base,type) and isinstance(self.subclass,type)
                and issubclass(self.subclass,base),'Native RPC subclass must preserve its exact base')

    def remote(self,*args,**kwargs):
        def decorate(cls):
            require(cls is self.base,'Unexpected native actor class at owned train creation')
            return _OwnedBuilder(self.ray.remote(**kwargs)(self.subclass),self.owner,self.kind)
        if args:
            require(len(args)==1,'Expected a single native actor class')
            return decorate(args[0])
        return decorate


class NativeOwnership:
    """Only handles created by this invocation; no global Ray/process cleanup.

    A bounded graceful-shutdown failure is reported, never certified as clean.
    The allocation controller remains the hard timeout owner for a hung actor.
    """
    def __init__(self,ray,remove_placement_group=None):
        self.ray,self.remove_placement_group=ray,remove_placement_group
        self.actors=[]
        self.placement_groups=[]
        self.closed=False
        self.cleanup_error=None

    def placement_factory(self,factory):
        def create(*args,**kwargs):
            pg=factory(*args,**kwargs)
            self.placement_groups.append(pg)
            return pg
        return create

    def close(self,timeout=30.):
        if self.closed:
            if self.cleanup_error is not None:
                raise RuntimeError(self.cleanup_error)
            return
        errors=[]
        refs=[]
        deadline=time.monotonic()+timeout
        # Dispatch all shutdowns first, then drain them. A failed child cannot
        # suppress sibling cleanup. manager shutdown owns its engine children.
        for kind,actor in reversed(self.actors):
            method={'manager':'action_credit_shutdown','engine':'action_credit_shutdown',
                    'train':'finish_tracking'}.get(kind)
            if method:
                try:
                    refs.append((kind,getattr(actor,method).remote()))
                except Exception as exc:
                    errors.append((kind,type(exc).__name__))
        for kind,ref in refs:
            try:
                result=self.ray.get(ref,timeout=max(.01,deadline-time.monotonic()))
                if kind in ('manager','engine') and result is not True:
                    raise RuntimeError('Owned native child did not acknowledge shutdown')
            except Exception as exc:
                errors.append((kind,type(exc).__name__))
        for kind,actor in reversed(self.actors):
            try:
                self.ray.kill(actor,no_restart=True)
            except Exception as exc:
                errors.append((kind,type(exc).__name__))
        for pg in reversed(self.placement_groups):
            try:
                self.remove_placement_group(pg)
            except Exception as exc:
                errors.append(('placement_group',type(exc).__name__))
        self.closed=True
        if errors:
            self.cleanup_error='Owned native cleanup incomplete: '+repr(errors)
            raise RuntimeError(self.cleanup_error)


@contextmanager
def _capture_processes(module,processes):
    factory=module.multiprocessing.Process
    def create(*args,**kwargs):
        process=factory(*args,**kwargs)
        # Save before start/health wait: failed startup must not lose ownership.
        processes.append(process)
        return process
    with _patched(module.multiprocessing,'Process',create):
        yield


def _stop_processes(processes,kill_tree):
    errors=[]
    for process in reversed(processes):
        try:
            # The Process object owns its child sentinel, not just a PID. An
            # unexpected dead parent does not prove its descendants closed.
            if getattr(process,'_action_credit_cleanup_complete',False):
                continue
            if process.pid is None:
                continue  # Construction failed before any process was started.
            if not process.is_alive():
                raise RuntimeError('Owned subprocess died before cleanup; descendant closure needs allocation reconciliation')
            kill_tree(process.pid)
            process.join(timeout=5.)
            if process.is_alive():
                raise RuntimeError('Owned subprocess survived shutdown')
            process._action_credit_cleanup_complete=True
        except Exception as exc:
            errors.append(type(exc).__name__)
    if errors:
        raise RuntimeError('Owned native subprocess cleanup incomplete: '+repr(errors))


def owned_engine_class(base):
    class ActionCreditEngine(base):
        def __init__(self,*args,**kwargs):
            self._action_credit_processes=[]
            super().__init__(*args,**kwargs)

        def _init_normal(self,server_args_dict):
            import slime.backends.sglang_utils.sglang_engine as module
            try:
                with _capture_processes(module,self._action_credit_processes):
                    return super()._init_normal(server_args_dict)
            except BaseException:
                self.action_credit_shutdown()
                raise

        def action_credit_shutdown(self):
            from slime.backends.sglang_utils.sglang_engine import kill_process_tree
            # The whole owned manager/router is shutting down; no unbounded
            # router-unregistration HTTP request is needed before process kill.
            _stop_processes(self._action_credit_processes,kill_process_tree)
            return True
    return ActionCreditEngine


def owned_manager_class(base):
    class ActionCreditRolloutManager(base):
        def __init__(self,*args,**kwargs):
            import ray
            import slime.ray.rollout as module
            self._action_credit_children=NativeOwnership(ray)
            self._action_credit_processes=[]
            self._health_monitors=[]
            self.rollout_engine_lock=None
            try:
                with ExitStack() as stack:
                    stack.enter_context(_patched(module,'ray',_OwnedRay(ray,self._action_credit_children,'engine')))
                    stack.enter_context(_patched(module,'SGLangEngine',owned_engine_class(module.SGLangEngine)))
                    stack.enter_context(_capture_processes(module,self._action_credit_processes))
                    super().__init__(*args,**kwargs)
            except BaseException:
                self.action_credit_shutdown()
                raise

        def action_credit_shutdown(self):
            import ray
            from slime.backends.sglang_utils.sglang_engine import kill_process_tree
            errors=[]
            for action in (lambda:super(ActionCreditRolloutManager,self).dispose(),
                           lambda:self._action_credit_children.close(timeout=20.),
                           lambda:_stop_processes(self._action_credit_processes,kill_process_tree)):
                try:
                    action()
                except Exception as exc:
                    errors.append(type(exc).__name__)
            if self.rollout_engine_lock is not None:
                try:
                    ray.kill(self.rollout_engine_lock,no_restart=True)
                except Exception as exc:
                    errors.append(type(exc).__name__)
            if errors:
                raise RuntimeError('Owned rollout cleanup incomplete: '+repr(errors))
            return True
    return ActionCreditRolloutManager


def require_owners(evaluation_owner,storage_admission):
    require(evaluation_owner is not None and all(callable(getattr(evaluation_owner,name,None))
            for name in ('initial','after_update','final')),
            'Concrete evaluation owner with initial/after_update/final methods is required')
    require(callable(storage_admission),'Live globally coordinated storage-admission callback is required')


async def awaited(value):
    return await value if inspect.isawaitable(value) else value


def worker_factory(assembly,task_inputs):
    template={key:assembly.worker[key] for key in ('sampling','browser_config','browser_manifest',
              'approval_reference','judge_protocol','canonical_reward_source_sha256')}
    def factory(store,contract,resolve_actor,endpoint_for_actor):
        remaining=assembly.launch.deadline_unix-time.time()
        require(remaining>0,'Allocation deadline expired before worker pool construction')
        return IsolatedWorkerPool(store=store,contract=contract,template=template,
            sample_factory=task_inputs.sample,
            native_args_factory=lambda actor:direct_engine_args(assembly.worker['native_args'],actor,endpoint_for_actor(actor)),
            concurrency=64,deadline_seconds=remaining,python=sys.executable,source_root=assembly.source['source_root'])
    return factory


async def run_with_models(args,actor_model,rollout_manager,assembly,*,evaluation_owner,
                          allocation_accounting,storage_admission):
    """Own initial eval, all updates, due evals and final review; never fake them."""
    require_owners(evaluation_owner,storage_admission)
    validate_native_profile(args,assembly);validate_native_acceptance(assembly)
    allocation=await awaited(allocation_accounting())
    audit_allocation(allocation,assembly.store.manifest)
    dataset=assembly.store.manifest.to_dict()['science']['dataset']
    tasks=NativeTaskInputs(dataset_reference=dict(path=dataset['path'],sha256=dataset['source_sha256']),
        task_order_reference=assembly.launch.task_order,browser_config=assembly.worker['browser_config'],
        browser_manifest=assembly.worker['browser_manifest'],native_args=assembly.worker['native_args'])
    context=dict(args=args,actor_model=actor_model,rollout_manager=rollout_manager,
        launch=assembly.launch,evaluation_protocol=assembly.accounting['evaluation'])
    # Owners must reconcile saved0/10 cohorts on retry; they control their exact
    # API/evaluation caps and independent review, which this module cannot infer.
    async def initial_evaluation():
        value=await awaited(evaluation_owner.initial(assembly.store,_contract(assembly.launch.initial_contract),context))
        _json(value)  # Require a retained artifact reference, not a success flag.
        return value
    initial=await storage_guarded(assembly.store,storage_admission,'initial','shared0-evaluation',
        assembly.launch.execution_id,initial_evaluation)
    auditor=CollectionAcceptanceAuditor(source_inventory_ref=assembly.source_reference,
        native_acceptance_ref=assembly.launch.native_acceptance,judge_protocol=assembly.protocol,
        allocation_accounting=allocation_accounting,diagnostic_only=False)
    async def acceptance(store,contract,pointer):
        return await auditor(store,contract,pointer)
    async def completed(store,count,result):
        value=await awaited(evaluation_owner.after_update(store,count,result,context))
        if count==10:
            _json(value)
        return value
    result=await run_pilot(args,actor_model,rollout_manager,assembly.launch,
        worker_pool_factory=worker_factory(assembly,tasks),request_pins=tasks.request_pins,
        collection_acceptance=acceptance,storage_admission=storage_admission,completion_hook=completed)
    async def final_review():
        value=await awaited(evaluation_owner.final(assembly.store,result,context))
        _json(value)
        return value
    final=await storage_guarded(assembly.store,storage_admission,'optimizer-update-9','final-evaluation-review',
        assembly.launch.execution_id,final_review)
    return dict(training=result,initial_evaluation_review=initial,final_evaluation_review=final,
                verified_complete=False,completion_authority='Independent final artifact review remains external')


def close_owned_writers(owner,finish_tracking,args,store,storage_admission,execution_id,stage,primary=None):
    """Drain both owners, then observe storage without hiding the first failure.

    A failed shutdown remains incomplete even when the subsequent reservation
    observation passes. These checks do not enforce a hard byte quota.
    """
    errors=[]
    for name,close in (('native ownership',owner.close),('tracking',lambda:finish_tracking(args))):
        try:
            close()
        except BaseException as error:
            error.add_note(name+' shutdown did not establish complete writer closure')
            errors.append(error)
    closed=not errors
    try:
        asyncio.run(storage_boundary(store,storage_admission,stage,
            'owned-writers-closed' if closed else 'owned-writers-incomplete',
            'after' if closed else 'after-error',execution_id))
    except BaseException as error:
        errors.append(error)
    if not errors:
        return
    first=primary if primary is not None else errors.pop(0)
    for error in errors:
        first.add_note('Additional shutdown/storage failure: '+repr(error))
    if primary is None:
        raise first


def train(args,assembly,*,evaluation_owner,allocation_accounting,storage_admission):
    """Actual startup inside an already-authorized allocation; not called on import."""
    require_owners(evaluation_owner,storage_admission)
    verify_native_runtime_sources(assembly.source)
    validate_native_profile(args,assembly);validate_native_acceptance(assembly)
    require(Path(assembly.source['source_root']).resolve()==REPO,
            'Run the bootstrap from the exact prepared production source root')
    require(time.time()<assembly.launch.deadline_unix,'Allocation deadline expired')
    require(callable(allocation_accounting),'Fresh all-attempt allocation provider required')
    audit_allocation(allocation_accounting(),assembly.store.manifest)
    asyncio.run(storage_boundary(assembly.store,storage_admission,'native-startup','native-startup',
        'before',assembly.launch.execution_id))
    # This is the full environment from an immutable private bundle. Credentials
    # remain inherited process secrets; they are never serialized or printed.
    for key,value in assembly.environment.items():
        if key.endswith('_DIR') or key.endswith('_HOME') or key in ('TMPDIR','TORCH_EXTENSIONS_DIR','FLASHINFER_WORKSPACE_BASE'):
            _safe_path(value).mkdir(parents=True,exist_ok=True)
        os.environ[key]=value
    verify_worker_environment()
    import ray
    import slime.backends.megatron_utils.actor as native_actor
    import slime.ray.actor_group as actor_group
    import slime.ray.placement_group as placement
    from ray.util.placement_group import remove_placement_group
    from slime.utils.logging_utils import configure_logger,init_tracking,finish_tracking
    configure_logger()
    owner=NativeOwnership(ray,remove_placement_group)
    with ExitStack() as stack:
        stack.enter_context(_patched(actor_group,'ray',_OwnedTrainRay(ray,owner,
            native_actor.MegatronTrainRayActor,rpc_actor_class)))
        stack.enter_context(_patched(placement,'placement_group',owner.placement_factory(placement.placement_group)))
        stack.enter_context(_patched(placement,'InfoActor',_OwnedBuilder(placement.InfoActor,owner,'info')))
        stack.enter_context(_patched(placement,'ray',_OwnedRay(ray,owner,'info')))
        manager_base=placement.RolloutManager.__ray_metadata__.modified_class
        manager_base=getattr(manager_base,'__ray_actor_class__',manager_base)
        manager_class=ray.remote(owned_manager_class(manager_base))
        stack.enter_context(_patched(placement,'RolloutManager',_OwnedBuilder(manager_class,owner,'manager')))
        primary=None
        final_stage='native-startup'
        try:
            pgs=placement.create_placement_groups(args)
            init_tracking(args)
            manager,_=placement.create_rollout_manager(args,pgs['rollout'])
            actor,critic=placement.create_training_models(args,pgs,manager)
            require(critic is None,'Unexpected critic allocation')
            asyncio.run(storage_boundary(assembly.store,storage_admission,'native-startup','native-startup',
                'after',assembly.launch.execution_id))
            result=asyncio.run(run_with_models(args,actor,manager,assembly,evaluation_owner=evaluation_owner,
                allocation_accounting=allocation_accounting,storage_admission=storage_admission))
            final_stage='optimizer-update-9'
            return result
        except BaseException as error:
            primary=error
            raise
        finally:
            close_owned_writers(owner,finish_tracking,args,assembly.store,storage_admission,
                assembly.launch.execution_id,final_stage,primary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect',action='store_true',required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--launch',type=Path,required=True)
    parser.add_argument('--launch-sha256',required=True)
    options=parser.parse_args()
    assembly=assemble(options.bundle,dict(path=str(options.launch),sha256=options.launch_sha256))
    print(json.dumps(dict(manifest_sha256=assembly.store.manifest.sha256,prepared_bundle_valid=True,
        launch_ready=False,authorizes_execution=False,remaining=[
            'Actual native argument parsing and physical actor/SGLang co-residency acceptance',
            'Fresh allocation accounting and globally coordinated storage admission',
            'Concrete owner for shared initial full300, iteration10 full300 and independent final review']),indent=2))


if __name__=='__main__':
    main()
