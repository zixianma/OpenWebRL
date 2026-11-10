"""Opt-in collective native checkpoint/update transaction for action credit.

Every Ray actor rank must enter ``operation(actor, command)`` concurrently. This
module does not allocate GPUs, collect browsers, call judges, or transfer weights
to inference engines. The caller owns the admitted allocation and its hard wall
limit. It supplies a deadline, and retains/charges every execution directory.
The actor weight backup is refreshed before the caller invokes update_weights.

Native checkpoint iteration equals completed collection updates in this NEW
namespace: initial state=0, completed first collection/PPO2=1. No existing RL
checkpoint namespace is reused. Pending batches, partial saves and unsuccessful
execution directories are never overwritten or pruned. A retry always restores
the previous committed update and its saved pre-optimizer RNG, then replays the
same complete schedule. A saved/reviewed checkpoint can be committed later
without training again, but must first pass a fresh native restore comparison.

Acceptance artifacts are trusted local validator outputs bound to immutable
source/manifest identities; hashes are integrity checks, not signatures or a
substitute for independent physical review. ``diagnostic_only`` permits explicit
CPU fake-runtime tests, but NEVER calls the campaign's production commit.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import fcntl
import io
import json
import os
import math
from pathlib import Path
import time

from openwebrl.action_credit import CollectionContract, digest
from openwebrl.action_credit_campaign import _artifact_file, validate_optimizer_rng_artifact
from openwebrl.action_credit_megatron import finalize_frozen_scores, native_frozen_scores, validate_backend_structure
from openwebrl.action_credit_objective import build_objective
from openwebrl.action_credit_state import CampaignManifest, CampaignStore, _safe_path
from openwebrl.action_credit_windows import native_train_collection, WindowConfig
from openwebrl.action_credit_tensor_refs import bind_tensor_inventory


def validator_source_sha256():
    root=Path(__file__).parent
    names=(Path(__file__).name,"action_credit_megatron.py","action_credit_windows.py",
           "action_credit_native_gate.py","action_credit_campaign.py","action_credit_tensor_refs.py")
    return digest({name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names})


def _hash(path):
    path=_safe_path(path)
    h=hashlib.sha256()
    with path.open("rb") as handle:
        while block:=handle.read(16*1024*1024):h.update(block)
    return h.hexdigest()


def _directory(path):
    path=_safe_path(path)
    roots=(Path("/gpfs/scrubbed/zixianma/openwebrl-runtime"),
           Path("/gpfs/scrubbed/zixianma/checkpoints/web"),Path("/tmp"))
    if not any(path==root or root in path.parents for root in roots):
        raise ValueError("native controller writes only approved runtime/checkpoint/tmp roots")
    path.mkdir(parents=True,exist_ok=True)
    return path


def _write(path,value,*,torch_payload=False):
    path=_safe_path(path)
    _directory(path.parent)
    if torch_payload:
        import torch
        stream=io.BytesIO();torch.save(value,stream);raw=stream.getvalue()
    else:raw=(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    # A partial/uncommitted file is evidence for reconciliation, not permission
    # to overwrite it or pretend the previous attempt did not consume time.
    with path.open("xb") as handle:
        handle.write(raw);handle.flush();os.fsync(handle.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())


def _json(reference):return json.loads(_artifact_file(reference).read_text())


def _trusted_torch(reference, *, artifact_verifier=_artifact_file):
    import torch
    # Only this campaign's checksummed, private, explicitly trusted payloads.
    # ZIP storages remain file backed, instead of allocating a full multi-GB
    # media corpus independently on every rank. No eager-loading fallback.
    # In collective RPCs the leader streams the complete file digest once and
    # broadcasts its verified canonical path; per-forward row hashes still run.
    path=_safe_path(artifact_verifier(reference))
    if path!=_safe_path(reference["path"]):raise ValueError("artifact verifier changed the private payload path")
    return torch.load(path,map_location="cpu",weights_only=False,mmap=True)


def _components(states,cursor):
    return {**{k:digest([(s["rank"],s[k]) for s in states]) for k in ("model","optimizer","scheduler","rng")},
            "cursor":digest(cursor)}


def load_scored_pending(store,update_id,*,leader=lambda fn:fn()):
    pointer=store.snapshot().get("pending_batches",{}).get(str(update_id))
    if not pointer or pointer["manifest_sha256"]!=store.manifest.sha256:
        raise ValueError("missing immutable scored pending batch")
    expected=_safe_path(store.root/"pending_updates"/str(update_id)/(pointer["input_sha256"]+".pt"))
    if _safe_path(pointer["path"])!=expected:raise ValueError("pending batch escaped its committed path")
    payload=_trusted_torch(dict(path=pointer["path"],sha256=pointer["file_sha256"]),
        artifact_verifier=lambda ref:leader(lambda:str(_artifact_file(ref))))
    bind_tensor_inventory(store, payload['tensor_payloads'].values(), payload['tensor_owners'], leader=leader)
    graph=payload["graph"]
    if (graph.contract.update_id!=update_id or graph.contract.policy_sha256!=pointer["policy_sha256"]
            or payload["checkpoint_policy"]!="restore_previous_full_update_and_replay_pending_schedule_from_zero"):
        raise ValueError("pending policy/update/checkpoint convention changed")
    # Check the stored scores themselves, rather than trusting finalized labels.
    finalized=finalize_frozen_scores(graph,payload["frozen_scores"],rows=payload["rows"],
        scorer_source_sha256=next(iter(payload["frozen_scores"].values())).scorer_source_sha256)
    if finalized!=graph:raise ValueError("saved graph changed frozen scores")
    validate_backend_structure(graph,payload["rows"],payload["tensor_payloads"])
    prepared=build_objective(graph,payload["config"],collection_binding=payload["binding"],
        replay_receipts=payload["replay_receipts"],panels=payload.get("panels",()),selection=payload.get("selection"))
    schedule=payload["ppo_schedule"]
    schedule.validate(prepared)
    if schedule.sha256!=pointer["ppo_schedule_sha256"] or len(schedule.windows)!=pointer["window_count"]:
        raise ValueError("saved PPO schedule differs from its committed pointer")
    if payload["binding"].sha256!=pointer["binding_sha256"]:
        raise ValueError("saved collection binding differs from its pointer")
    if payload["optimizer_rng_artifact"]!=pointer["optimizer_rng_artifact"]:
        raise ValueError("saved pre-optimizer RNG differs from its committed pointer")
    # This checks the complete previous checkpoint file inventory, so perform
    # its potentially large disk scan once on the collective leader.
    leader(lambda:validate_optimizer_rng_artifact(store,graph.contract,payload["optimizer_rng_artifact"]))
    return pointer,payload,prepared


class MegatronRuntime:
    """Concrete actual-model implementation; no CPU fallback in production RPC."""
    verification_kind="production_runtime"

    def __init__(self,actor):
        import torch
        import torch.distributed as dist
        from megatron.core import mpu
        from openwebrl.action_credit_megatron import check_topology
        self.actor=actor
        args=actor.args
        self.rank,self.world_size=dist.get_rank(),dist.get_world_size()
        self.tp,self.dp=mpu.get_tensor_model_parallel_world_size(),mpu.get_data_parallel_world_size(False)
        check_topology(args,cp=mpu.get_context_parallel_world_size(),pp=mpu.get_pipeline_model_parallel_world_size())
        if (not torch.cuda.is_available() or self.world_size!=8 or self.tp*self.dp!=8
                or args.offload_train or args.async_save or getattr(args,"no_save_rng",False)
                or not args.data_parallel_random_init or getattr(args,"overlap_param_gather",False)
                or getattr(actor,"_active_model_tag",None)!="actor"
                or sum(p.numel() for m in actor.model for p in m.parameters())*self.tp<3_000_000_000):
            raise ValueError("production native RPC requires the admitted active 4B/CUDA/eight-rank nonoffloaded profile")
        if not hasattr(actor,"_action_credit_control_group"):
            actor._action_credit_control_group=dist.new_group(backend="gloo")
        self.group=actor._action_credit_control_group

    def gather(self,value):
        import torch.distributed as dist
        result=[None]*self.world_size
        dist.all_gather_object(result,value,group=self.group)
        return result

    def leader(self,fn):
        import torch.distributed as dist
        output=[None]
        if self.rank==0:
            try:output[0]=dict(ok=True,value=fn())
            except Exception as error:output[0]=dict(ok=False,error=type(error).__name__+": "+str(error))
        dist.broadcast_object_list(output,src=0,group=self.group)
        if not output[0]["ok"]:raise ValueError(output[0]["error"])
        return output[0]["value"]

    def snapshot(self):
        from openwebrl.action_credit_native_gate import state_digest,optimizer_runtime_state,native_rng_sha256
        a=self.actor
        return dict(rank=self.rank,model=state_digest([m.state_dict() for m in a.model]),
            optimizer=state_digest(optimizer_runtime_state(a.optimizer)),
            scheduler=state_digest(a.opt_param_scheduler.state_dict()),rng=native_rng_sha256())

    def rng(self):
        import random,numpy as np,torch
        from megatron.core import tensor_parallel
        return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state(),tracker={k:tensor_parallel.convert_cuda_rng_state(v,to_graphable=False)
                for k,v in tensor_parallel.get_cuda_rng_tracker().get_states().items()})

    def restore_rng(self,value):
        import random,numpy as np,torch
        from megatron.core import tensor_parallel
        random.setstate(value["python"]);np.random.set_state(value["numpy"])
        torch.set_rng_state(value["torch_cpu"]);torch.cuda.set_rng_state(value["cuda"])
        tracker=tensor_parallel.get_cuda_rng_tracker()
        graph_safe=getattr(tracker,"use_cudagraphable_rng",False)
        tracker.set_states({k:tensor_parallel.convert_cuda_rng_state(v,to_graphable=graph_safe)
                            for k,v in value["tracker"].items()})
        from openwebrl.action_credit_native_gate import state_digest,native_rng_sha256
        numpy_state=value["numpy"]
        expected=dict(python=state_digest(value["python"]),
            numpy=state_digest((numpy_state[0],numpy_state[1].tolist(),*numpy_state[2:])),
            torch_cpu=state_digest(value["torch_cpu"]),cuda=state_digest(value["cuda"]),
            megatron_tracker=state_digest(value["tracker"]))
        if native_rng_sha256()!=expected:raise ValueError("pre-optimizer RNG did not restore exactly")

    def initialize_original(self,path):
        from slime.backends.megatron_utils.checkpoint import load_checkpoint
        import torch
        a=self.actor
        if a.opt_param_scheduler.num_steps!=0:raise ValueError("initial scheduler already advanced")
        def check_optimizer(opt):
            if hasattr(opt,"chained_optimizers"):
                for child in opt.chained_optimizers:check_optimizer(child)
            else:
                for state in opt.optimizer.state_dict()["state"].values():
                    for value in state.values():
                        if ((isinstance(value,torch.Tensor) and bool((value!=0).any()))
                                or (not isinstance(value,torch.Tensor) and value!=0)):
                            raise ValueError("initial optimizer contains nonzero update state")
        check_optimizer(a.optimizer)
        if Path(a.args.hf_checkpoint).resolve()!=Path(path).resolve():raise ValueError("original HF actor path changed")
        a.args.load=str(path)
        loaded,_=load_checkpoint(a.model,a.optimizer,a.opt_param_scheduler,{},False)
        if loaded!=0:raise ValueError("fresh original HF reload did not return initialization")
        self.backup_actor()

    def save(self,directory,iteration):
        from slime.backends.megatron_utils.model import save
        a=self.actor;a.args.save=str(directory)
        save(iteration,a.model,a.optimizer,a.opt_param_scheduler)
        self.gather(dict(rank=self.rank,saved=True))

    def restore(self,directory,iteration):
        from slime.backends.megatron_utils.checkpoint import load_checkpoint
        a=self.actor
        a.args.load,a.args.ckpt_step=str(directory),iteration
        a.args.no_load_optim,a.args.no_load_rng,a.args.finetune=False,False,False
        loaded,_=load_checkpoint(a.model,a.optimizer,a.opt_param_scheduler,{},False)
        if loaded!=iteration:raise ValueError("native checkpoint restored the wrong completed update")

    def train(self,prepared,schedule,rows,tensors,check_deadline):
        return native_train_collection(self.actor,prepared,schedule,rows,tensors,before_window=lambda _:check_deadline())

    def backup_actor(self):
        self.actor.weights_backuper.backup("actor")
        self.actor._active_model_tag="actor"


class NativeUpdateController:
    def __init__(self,store,runtime,native_acceptance_ref,*,diagnostic_only=False):
        self.store,self.runtime,self.diagnostic_only=store,runtime,diagnostic_only
        self.manifest=store.manifest.to_dict()
        self.checkpoint=self.manifest["science"]["checkpoint"]
        if self.checkpoint.get("native_checkpoint_validator_sha256")!=validator_source_sha256():
            raise ValueError("native validator source is not pinned in campaign")
        evidence=_json(native_acceptance_ref)
        if (evidence.get("passed") is not True or evidence.get("schema_version")!=1
                or evidence.get("manifest_sha256")!=store.manifest.sha256
                or evidence.get("validator_source_sha256")!=validator_source_sha256()
                or evidence.get("topology")!={"world_size":runtime.world_size,"tp":runtime.tp,"dp":runtime.dp}
                or evidence.get("verification_kind")!=runtime.verification_kind):
            raise ValueError("missing source/manifest/topology-bound physical native acceptance")
        required=("actual_model","full_response_masks","tp_dp_gradient_parity","ppo2_windows",
                  "native_checkpoint_restore","per_rank_rng_restore")
        if not diagnostic_only:required+=("native_controller_integration",)
        if any(evidence.get(key) is not True for key in required):
            raise ValueError("native acceptance is missing a required physical gate")
        if not diagnostic_only and (type(runtime) is not MegatronRuntime or runtime.verification_kind!="production_runtime"):
            raise ValueError("CPU fake runtime cannot execute a production transaction")
        self.native_acceptance_ref=native_acceptance_ref

    def _restore_inventory(self,reference):
        inventory=self.runtime.leader(lambda:_json(reference))
        if (inventory.get("manifest_sha256")!=self.store.manifest.sha256
                or inventory.get("validator_source_sha256")!=validator_source_sha256()
                or inventory.get("verification_kind")!=self.runtime.verification_kind
                or inventory.get("diagnostic_only") is not self.diagnostic_only):
            raise ValueError("checkpoint inventory source or physical kind changed")
        root=_safe_path(inventory["native_checkpoint_root"])
        if (Path(reference["path"]).parent not in root.parents
                or type(inventory.get("completed_updates")) is not int or inventory["completed_updates"]<0
                or inventory.get("native_iteration")!=inventory["completed_updates"]
                or len(inventory.get("rank_state_sha256",()))!=self.runtime.world_size
                or [r.get("rank") for r in inventory["rank_state_sha256"]]!=list(range(self.runtime.world_size))
                or not inventory.get("files")
                or any(Path(reference["path"]).parent not in Path(entry["path"]).parents for entry in inventory["files"])):
            raise ValueError("native checkpoint escaped its inventory or changed ranks/cursor")
        self.runtime.leader(lambda:[str(_artifact_file(entry)) for entry in inventory["files"]])
        self.runtime.restore(inventory["native_checkpoint_root"],inventory["native_iteration"])
        states=self.runtime.gather(self.runtime.snapshot())
        if states!=inventory["rank_state_sha256"]:raise ValueError("native model/Adam/scheduler/per-rank RNG differs after restore")
        self.runtime.backup_actor()
        return inventory

    def _save_review(self,directory,completed_updates,policy_sha256,cursor,kind):
        directory=self.runtime.leader(lambda:str(_directory(directory)))
        directory=Path(directory)
        native=directory/"checkpoint"
        self.runtime.leader(lambda:_directory(native))
        states=self.runtime.gather(self.runtime.snapshot())
        self.runtime.save(native,completed_updates)
        self.runtime.restore(native,completed_updates)
        restored=self.runtime.gather(self.runtime.snapshot())
        if states!=restored:raise ValueError("new native checkpoint fails complete state/RNG roundtrip")
        components=_components(states,cursor)
        policy=policy_sha256 or digest(dict(native_model_state_sha256=components["model"],tp=self.runtime.tp,dp=self.runtime.dp))
        def inventory():
            cursor_ref=_write(directory/"cursor.json",cursor)
            files=[]
            for path in sorted(native.rglob("*")):
                if path.is_symlink():raise ValueError("native checkpoint contains a symlink")
                if path.is_file():files.append(dict(path=str(_safe_path(path)),sha256=_hash(path),
                    components=["model","optimizer","scheduler","rng"]))
            if not files:raise ValueError("native save emitted no checkpoint files")
            files.append(dict(cursor_ref,components=["cursor"]))
            value=dict(schema_version=1,verification_kind=self.runtime.verification_kind,diagnostic_only=self.diagnostic_only,
                manifest_sha256=self.store.manifest.sha256,validator_source_sha256=validator_source_sha256(),
                policy_sha256=policy,completed_updates=completed_updates,checkpoint_kind=kind,
                native_checkpoint_root=str(native),native_iteration=completed_updates,
                component_inventory_semantics="joint_native_checkpoint_bundle",
                files=files,rank_state_sha256=states,component_state_sha256=components,
                fresh_optimizer=completed_updates==0,fresh_scheduler=completed_updates==0,fresh_cursor=completed_updates==0,
                native_acceptance=self.native_acceptance_ref)
            return _write(directory/"inventory.json",value)
        reference=self.runtime.leader(inventory)
        self.runtime.backup_actor()
        return reference,self.runtime.leader(lambda:_json(reference))

    def initialize(self,contract):
        contract.validate()
        if contract.update_id!=0 or contract.policy_sha256!=self.manifest["initialization"]["original_actor_sha256"]:
            raise ValueError("new experiment must start at agreed original actor/update zero")
        original=self.checkpoint["original_actor_inventory"]
        if original["sha256"]!=contract.policy_sha256:raise ValueError("original policy identity must bind HF inventory")
        def validate_original():
            saved=_json(original)
            if saved["model_root"]!=self.checkpoint["original_actor_path"] or not saved.get("files"):
                raise ValueError("original model root or complete immutable source inventory absent")
            for entry in saved["files"]:
                path=_artifact_file(entry)
                if Path(saved["model_root"]) not in path.parents:raise ValueError("original model inventory escaped root")
        self.runtime.leader(validate_original)
        root=_safe_path(self.checkpoint["native_initial_state_root"])
        existing=root/"inventory.json"
        if self.runtime.leader(existing.exists):
            ref=self.runtime.leader(lambda:dict(path=str(existing),sha256=_hash(existing)))
            inventory=self._restore_inventory(ref)
            if (inventory["policy_sha256"]!=contract.policy_sha256 or inventory["completed_updates"]!=0
                    or inventory["checkpoint_kind"]!="native_initialization"
                    or any(inventory.get(k) is not True for k in ("fresh_optimizer","fresh_scheduler","fresh_cursor"))):
                raise ValueError("initial namespace contains a noninitial checkpoint")
            return ref
        self.runtime.leader(lambda:(_ for _ in ()).throw(ValueError("partial initial checkpoint requires reconciliation"))
                            if root.exists() and any(root.iterdir()) else None)
        self.runtime.initialize_original(self.checkpoint["original_actor_path"])
        cursor=dict(completed_updates=0,next_update_id=0,task_order_sha256=self.manifest["science"]["dataset"]["task_order_sha256"])
        return self._save_review(root,0,contract.policy_sha256,cursor,"native_initialization")[0]

    def restore_and_capture_rng(self,contract,inventory_ref,*,capture_id):
        inventory=self._restore_inventory(inventory_ref)
        if inventory["policy_sha256"]!=contract.policy_sha256 or inventory["completed_updates"]!=contract.update_id:
            raise ValueError("restore policy/cursor does not match next collection")
        directory=_safe_path(self.store.root/"native_rng"/str(contract.update_id)/digest(capture_id))
        self.runtime.leader(lambda:_directory(directory))
        rank_ref=_write(directory/f"rank-{self.runtime.rank}.pt",dict(rank=self.runtime.rank,rng=self.runtime.rng()),torch_payload=True)
        ranks=self.runtime.gather(dict(rank_ref,rank=self.runtime.rank))
        def sidecar():
            return _write(directory/"sidecar.json",dict(schema_version=1,manifest_sha256=self.store.manifest.sha256,
                policy_sha256=contract.policy_sha256,graph_contract_sha256=contract.sha256,update_id=contract.update_id,
                world_size=self.runtime.world_size,rank_rng_files=ranks,previous_checkpoint_inventory=inventory_ref,
                validator_source_sha256=validator_source_sha256()))
        return self.runtime.leader(sidecar)

    def execute_pending(self,update_id,execution_id,collection_acceptance_ref,*,deadline_unix):
        # Kernel lock prevents a second actor group from becoming a concurrent
        # writer. Process death releases it; persistent attempts remain intact.
        def acquire():
            self._execution_lock=_safe_path(self.store.root/"native-update.lock").open("a+b")
            try:fcntl.flock(self._execution_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:
                self._execution_lock.close()
                raise
        def release():
            fcntl.flock(self._execution_lock.fileno(),fcntl.LOCK_UN)
            self._execution_lock.close()
        self.runtime.leader(acquire)
        try:return self._execute_pending(update_id,execution_id,collection_acceptance_ref,deadline_unix=deadline_unix)
        finally:self.runtime.leader(release)

    def _execute_pending(self,update_id,execution_id,collection_acceptance_ref,*,deadline_unix):
        if (not isinstance(execution_id,str) or not execution_id or type(deadline_unix) not in (int,float)
                or not math.isfinite(deadline_unix)):
            raise ValueError("explicit execution identity and admitted allocation deadline required")
        started=time.time()
        def check_deadline():
            def check():
                if time.time()>=deadline_unix:raise TimeoutError("admitted allocation deadline reached")
            self.runtime.leader(check)
        check_deadline()
        pointer,payload,prepared=load_scored_pending(self.store,update_id,leader=self.runtime.leader)
        contract=prepared.plan.graph.contract
        acceptance=self.runtime.leader(lambda:_json(collection_acceptance_ref))
        expected=dict(schema_version=1,verification_kind=self.runtime.verification_kind,passed=True,
            manifest_sha256=self.store.manifest.sha256,policy_sha256=contract.policy_sha256,
            update_id=update_id,pending_file_sha256=pointer["file_sha256"],
            binding_sha256=pointer["binding_sha256"],ppo_schedule_sha256=pointer["ppo_schedule_sha256"],
            validator_source_sha256=validator_source_sha256())
        required=("actor_weight_transfer_verified","saved_episode_artifacts_verified","browser_replay_verified",
                  "sample_masks_verified","terminal_judge_artifacts_verified","all_attempt_costs_reconciled")
        if self.diagnostic_only and acceptance.get("synthetic_collection_provenance") is True:
            required=("saved_fixture_artifact_verified",)
        if any(acceptance.get(k)!=v for k,v in expected.items()) or any(acceptance.get(k) is not True for k in required):
            raise ValueError("online physical acceptance does not cover this exact complete pending batch")
        transaction_id="native-update-"+pointer["input_sha256"]
        completed=self.store.snapshot().get("native_completed_updates",{}).get(str(update_id))
        if completed:
            review_ref=completed["review"]
            review=self.runtime.leader(lambda:_json(review_ref))
            self._restore_inventory(review["checkpoint_inventory"])
            return self._commit(update_id,transaction_id,review_ref,review)
        self.runtime.leader(lambda:self.store.begin_update(update_id,attempt_id=transaction_id))
        directory=_safe_path(self.store.root/"native_updates"/str(update_id)/digest(execution_id))
        review_path=directory/"review.json"
        if self.runtime.leader(review_path.exists):
            review_ref=self.runtime.leader(lambda:dict(path=str(review_path),sha256=_hash(review_path)))
            review=self.runtime.leader(lambda:_json(review_ref))
            if review["pending_file_sha256"]!=pointer["file_sha256"]:raise ValueError("reconciliation batch changed")
            self._restore_inventory(review["checkpoint_inventory"])
            return self._commit(update_id,transaction_id,review_ref,review)
        self.runtime.leader(lambda:_write(directory/"execution.json",dict(execution_id=execution_id,
            started_unix=started,deadline_unix=deadline_unix,pending_file_sha256=pointer["file_sha256"],
            policy_sha256=contract.policy_sha256,ppo_schedule_sha256=pointer["ppo_schedule_sha256"],
            resume_from_window=0,all_attempts_chargeable=True)))
        rng_ref=payload["optimizer_rng_artifact"]
        sidecar=self.runtime.leader(lambda:_json(rng_ref))
        self._restore_inventory(sidecar["previous_checkpoint_inventory"])
        rank=next(r for r in sidecar["rank_rng_files"] if r["rank"]==self.runtime.rank)
        saved_rng=_trusted_torch(rank)
        if saved_rng["rank"]!=self.runtime.rank:raise ValueError("saved pre-optimizer RNG belongs to another rank")
        self.runtime.restore_rng(saved_rng["rng"])
        report=self.runtime.train(prepared,payload["ppo_schedule"],payload["rows"],payload["tensor_payloads"],check_deadline)
        check_deadline()
        cursor=dict(completed_updates=update_id+1,next_update_id=update_id+1,
            task_order_sha256=self.manifest["science"]["dataset"]["task_order_sha256"],
            pending_input_sha256=pointer["input_sha256"],ppo_schedule_sha256=pointer["ppo_schedule_sha256"],
            completed_windows=len(payload["ppo_schedule"].windows))
        inventory_ref,inventory=self._save_review(directory/"completed",update_id+1,None,cursor,"native_completed_update")
        def review():
            value=dict(schema_version=1,verification_kind=self.runtime.verification_kind,
                diagnostic_only=self.diagnostic_only,
                pending_file_sha256=pointer["file_sha256"],checkpoint_inventory=inventory_ref,
                policy_sha256=contract.policy_sha256,new_policy_sha256=inventory["policy_sha256"],
                component_state_sha256=inventory["component_state_sha256"],
                optimizer_updates=report["optimizer_updates_in_memory"],ppo_schedule_sha256=pointer["ppo_schedule_sha256"],
                native_save_restore_verified=True,per_rank_rng_verified=True,
                collection_acceptance=collection_acceptance_ref,elapsed_seconds=time.time()-started,
                charged_gpu_seconds_upper_bound=(time.time()-started)*self.runtime.world_size,
                external_allocation_ledger_owns_all_attempts=True)
            return _write(review_path,value)
        review_ref=self.runtime.leader(review)
        return self._commit(update_id,transaction_id,review_ref,self.runtime.leader(lambda:_json(review_ref)))

    def _commit(self,update_id,transaction_id,review_ref,review):
        update=self.store.snapshot()["updates"][str(update_id)]
        pointer=self.store.snapshot()["pending_batches"][str(update_id)]
        inventory=self.runtime.leader(lambda:_json(review["checkpoint_inventory"]))
        if (review.get("verification_kind")!=self.runtime.verification_kind
                or review.get("diagnostic_only") is not self.diagnostic_only
                or review.get("native_save_restore_verified") is not True or review.get("per_rank_rng_verified") is not True
                or review.get("pending_file_sha256")!=pointer["file_sha256"]
                or review.get("ppo_schedule_sha256")!=pointer["ppo_schedule_sha256"]
                or review.get("optimizer_updates")!=pointer["window_count"]
                or inventory["completed_updates"]!=update_id+1
                or inventory["policy_sha256"]!=review.get("new_policy_sha256")
                or inventory["component_state_sha256"]!=review.get("component_state_sha256")):
            raise ValueError("physical update review differs from complete saved native update")
        components=review["component_state_sha256"]
        receipt=dict(manifest_sha256=self.store.manifest.sha256,input_export_sha256=update["export_sha256"],
            policy_sha256=update["policy_sha256"],new_policy_sha256=review["new_policy_sha256"],
            new_checkpoint_sha256=review["checkpoint_inventory"]["sha256"],
            optimizer_state_sha256=components["optimizer"],scheduler_state_sha256=components["scheduler"],
            cursor_state_sha256=components["cursor"],validator_evidence_sha256=review_ref["sha256"],
            validator_code_revision_sha256=self.manifest["code_revision_sha256"],verification_kind=self.runtime.verification_kind)
        if self.diagnostic_only:return dict(committed=False,diagnostic_only=True,receipt=receipt,review=review_ref)
        def retain_commit_pointer():
            with self.store._transaction() as state:
                entries=state.setdefault("native_completed_updates",{})
                value=dict(review=review_ref,checkpoint_inventory=review["checkpoint_inventory"])
                if str(update_id) in entries and entries[str(update_id)]!=value:
                    raise ValueError("completed native update pointer cannot change")
                entries[str(update_id)]=value
        self.runtime.leader(retain_commit_pointer)
        result=self.runtime.leader(lambda:self.store.complete_update(update_id,attempt_id=transaction_id,receipt=receipt))
        return dict(committed=True,result=result,receipt=receipt,review=review_ref)


def operation(actor,command):
    """Thin Ray RPC target. Call all eight handlers; never pass a batch via Ray."""
    manifest=CampaignManifest.from_dict(_json(command["manifest"]))
    store=CampaignStore(command["store_root"],manifest)
    runtime=MegatronRuntime(actor)
    diagnostic=command.get("diagnostic_only",False)
    if type(diagnostic) is not bool:raise ValueError("diagnostic mode must be explicit boolean")
    controller=NativeUpdateController(store,runtime,command["native_acceptance"],diagnostic_only=diagnostic)
    name=command["operation"]
    if name in ("initialize","restore_capture"):
        data=dict(command["contract"])
        for key in ("train_task_ids","heldout_task_ids"):data[key]=tuple(data[key])
        contract=CollectionContract(**data)
        if name=="initialize":return controller.initialize(contract)
        return controller.restore_and_capture_rng(contract,command["checkpoint_inventory"],capture_id=command["capture_id"])
    if name=="score_save":
        from openwebrl.action_credit_online import save_scored_draft
        payload=_trusted_torch(command["draft_artifact"],
            artifact_verifier=lambda ref:runtime.leader(lambda:str(_artifact_file(ref))))
        draft,config=payload["draft"],payload["config"]
        bind_tensor_inventory(store, draft['tensor_payloads'].values(), payload['tensor_owners'], leader=runtime.leader)
        runtime.leader(lambda:validate_optimizer_rng_artifact(store,draft["graph"].contract,command["optimizer_rng_artifact"]))
        rng=_json(command["optimizer_rng_artifact"])
        controller._restore_inventory(rng["previous_checkpoint_inventory"])
        scores=native_frozen_scores(actor,draft["graph"],draft["rows"],draft["tensor_payloads"],
            scorer_source_sha256=hashlib.sha256(Path(__file__).with_name("action_credit_megatron.py").read_bytes()).hexdigest())
        def save_pending():
            pointer,_=save_scored_draft(store,config,draft,scores,
                scorer_source_sha256=next(iter(scores.values())).scorer_source_sha256,
                window_config=WindowConfig(**manifest.to_dict()["science"]["algorithm"]["ppo_window_config"]),
                optimizer_rng_artifact=command["optimizer_rng_artifact"])
            return pointer
        return runtime.leader(save_pending)
    if name=="update":return controller.execute_pending(command["update_id"],command["execution_id"],
        command["collection_acceptance"],deadline_unix=command["deadline_unix"])
    raise ValueError("unsupported action-credit native operation")
