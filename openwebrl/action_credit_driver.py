"""Opt-in trainer owner for the five-arm pilot; importing starts no work.

The caller creates the actual eight Megatron Ray actors and updatable SGLang
engines using the pinned native launcher. This driver serializes restore,
transfer, collection, scoring and complete PPO updates. Every remote child and
browser worker is awaited, including siblings after an error. Allocation/API
approval, the isolated native Sample factory, independent physical acceptance,
and milestone evaluation remain explicit caller inputs, never fabricated here.

Recovery reuses committed episodes/drafts/scored batches. An ambiguous browser
or judge attempt is not resent. Native partial updates restore the previous
full checkpoint and replay the same pending windows. The final return certifies
training ledger state only; evaluation completion is reported by its owner.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, replace
import fcntl
import gc
import inspect
import json
import math
import os
from pathlib import Path
import time
import uuid
from urllib.parse import urlsplit

from .action_credit import CollectionContract, digest
from .action_credit_campaign import _artifact_file
from .action_credit_collect import FrozenActor, physical_verifier_sha256
from .action_credit_native_controller import _hash, _json, _write
from .action_credit_objective import ObjectiveConfig
from .action_credit_online import collect_online_update, _persist
from .action_credit_selection import TreeSelectionPolicy, TreeUpdateSelectionPolicy, TreeSchedulePin
from .action_credit_state import CampaignManifest, CampaignStore, _safe_path
from .action_credit_worker import served_weight_version


@dataclass(frozen=True)
class DriverLaunch:
    schema_version: int
    manifest: dict
    store_root: str
    task_order: dict
    initial_contract: dict
    objective_config: dict
    native_acceptance: dict
    execution_id: str
    deadline_unix: float
    tree_policy: dict | None
    target_updates: int = 10
    worker_concurrency: int = 64

    def validate(self):
        if (self.schema_version != 1 or self.target_updates != 10 or self.worker_concurrency != 64
                or not self.execution_id or not isinstance(self.execution_id, str)
                or type(self.deadline_unix) not in (int, float) or not math.isfinite(self.deadline_unix)):
            raise ValueError("explicit ten-update/64-worker pilot identity and allocation deadline required")
        _safe_path(self.store_root)
        for reference in (self.manifest, self.task_order, self.native_acceptance):
            _artifact_file(reference)
        contract = _contract(self.initial_contract)
        if contract.update_id != 0:
            raise ValueError("pilot contract must start at initialization zero")
        ObjectiveConfig(**self.objective_config).validate()


def _contract(value):
    value = dict(value)
    for key in ("train_task_ids", "heldout_task_ids"):
        value[key] = tuple(value[key])
    contract = CollectionContract(**value)
    contract.validate()
    return contract


def _tree_policy(value):
    if value is None:
        return None
    return TreeSelectionPolicy(value["task_order_sha256"], tuple(TreeUpdateSelectionPolicy(
        u["update_id"], tuple(TreeSchedulePin(**s) for s in u["schedules"])) for u in value["updates"]))


async def _owned(refs):
    """Drain every started Ray ref/coroutine before propagating any failure."""
    group = asyncio.gather(*refs, return_exceptions=True)
    try:
        results = await asyncio.shield(group)
    except asyncio.CancelledError:
        # Cancelling a Python await does not stop an already-running Ray
        # collective. Keep ownership until its actual remote result arrives.
        await asyncio.shield(group)
        raise
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


def _real_handlers(actor_model):
    from ray.actor import ActorHandle
    handles = tuple(actor_model._actor_handlers)
    if len(handles) != 8 or len({h._actor_id for h in handles}) != 8 or any(not isinstance(h, ActorHandle) for h in handles):
        raise ValueError("exactly eight distinct actual native Ray actors required")
    return handles


async def _call_all(handles, method, *args):
    refs = []
    try:
        for handle in handles:
            refs.append(getattr(handle, method).remote(*args))
    except BaseException:
        # Already-started collectives cannot be abandoned. The allocation's
        # process supervisor remains the hard timeout owner for a dead rank.
        await _owned(refs)
        raise
    return await _owned(refs)


async def storage_boundary(store, provider, stage, phase, position, execution_id, *, reused=False):
    """Recheck the existing endpoint reservation; this is not a write quota.

    Evidence is fresh at each boundary, including retries/reused artifacts. No
    storage reservation is added, released, or resized by this adapter. Its
    small immutable receipts remain inside the already-owned campaign root.
    """
    if (not callable(provider) or stage not in ('native-startup', 'initial',
            *(f'optimizer-update-{i}' for i in range(10))) or position not in ('before', 'after', 'after-error')
            or not isinstance(phase, str) or not phase or not isinstance(execution_id, str) or not execution_id
            or type(reused) is not bool):
        raise ValueError('required storage-admission provider and exact phase/stage identity')
    started = time.time()
    identity = dict(schema_version=1, manifest_sha256=store.manifest.sha256, store_root=str(store.root),
        execution_id=execution_id, stage=stage, phase=phase, position=position, reused=reused,
        started_unix=started, hard_byte_cap_enforced=False, additional_storage_reserved_bytes=0)
    path = _safe_path(store.root/'storage_boundaries'/digest(execution_id)/(uuid.uuid4().hex+'.json'))
    reference = None
    try:
        value = provider(store, stage)
        reference = (await _owned([value]))[0] if inspect.isawaitable(value) else value
        proof = _json(reference)
        reservation = proof.get('reservation', {})
        if (proof.get('passed') is not True or proof.get('manifest_sha256') != store.manifest.sha256
                or proof.get('stage') != stage or reservation.get('manifest_sha256') != store.manifest.sha256
                or reservation.get('store_root') != str(store.root)
                or type(reservation.get('epoch')) not in (int, float)
                or not started <= reservation['epoch'] <= time.time()):
            raise ValueError('storage admission is stale, failed, or bound to another manifest/stage')
    except BaseException as error:
        # Refusal also remains attributable to the exact boundary. If even this
        # small evidence write fails, dispatch remains blocked and the original
        # exception carries that failure; no successful audit is fabricated.
        try:
            failed = _write(path, dict(identity, passed=False, finished_unix=time.time(),
                admission=reference, error_type=type(error).__name__, error=str(error)[:1024]))
            error.add_note('Storage boundary '+phase+'/'+position+' refused; evidence='+failed['path'])
        except BaseException as evidence_error:
            error.add_note('Storage boundary '+phase+'/'+position+' refusal evidence failed: '+type(evidence_error).__name__)
        raise
    return _write(path, dict(identity, passed=True, finished_unix=time.time(), admission=reference))


async def storage_guarded(store, provider, stage, phase, execution_id, operation, *, reused=False):
    """Own a stage and reconcile after it, without hiding its primary failure."""
    await storage_boundary(store, provider, stage, phase, 'before', execution_id, reused=reused)
    try:
        result = operation()
        result = await result if inspect.isawaitable(result) else result
    except BaseException as error:
        try:
            await storage_boundary(store, provider, stage, phase, 'after-error', execution_id, reused=reused)
        except BaseException as storage_error:
            error.add_note('Post-stage storage reconciliation also failed: '+repr(storage_error))
        raise
    await storage_boundary(store, provider, stage, phase, 'after', execution_id, reused=reused)
    return result


def _unanimous(results):
    if len(results) != 8 or any(result != results[0] for result in results):
        raise ValueError("native ranks disagree on the durable operation result")
    return results[0]


def _draft_value(value):
    """Canonical metadata only; never materialize a referenced tensor owner."""
    from dataclasses import fields, is_dataclass
    from .action_credit_tensor_refs import TensorBlobMapping
    if isinstance(value, TensorBlobMapping):
        return dict(tensor_owner=value.ref.to_dict())
    if is_dataclass(value):
        return {field.name: _draft_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError('draft metadata requires string keys')
        return {key: _draft_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_draft_value(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError('draft contains an unbound value or inline tensor')


def _draft_identity(store, contract, draft, config):
    """Bind an exact rebuilt batch to its committed collection and cursor."""
    from .action_credit_binding import _check_contract
    manifest, state = store.manifest.to_dict(), store.snapshot()
    _check_contract(manifest, contract)
    config.validate()
    algorithm = manifest['science']['algorithm']
    if (draft['graph'].contract != contract or manifest['arm_id'] != config.arm_id
            or digest(asdict(config)) != algorithm['objective_config_sha256']):
        raise ValueError('draft policy/update or objective differs from frozen campaign')
    key = str(contract.update_id)
    current = state['updates'].get(key)
    plan = state.get('online_collection_plans', {}).get(key)
    if (current is None or plan is None or current['policy_sha256'] != contract.policy_sha256
            or current['plan_sha256'] != plan['plan_sha256']
            or plan['graph_contract_sha256'] != contract.sha256
            or plan['task_order_sha256'] != manifest['science']['dataset']['task_order_sha256']
            or digest(current['expected_draws']) != current['plan_sha256']):
        raise ValueError('draft requires the exact committed collection plan')
    mapping = {item['leaf_id']: digest(store._draw_identity(contract.update_id,
        contract.policy_sha256, item['spec'])) for item in plan['draws']}
    if (draft['leaf_to_draw'] != mapping or set(mapping.values()) != set(current['expected_draws'])
            or tuple(draft['graph'].planned_leaf_ids) != tuple(mapping)):
        raise ValueError('draft differs from the fixed draw slots')
    prior = None
    if contract.update_id:
        previous = state['updates'].get(str(contract.update_id - 1), {})
        prior = previous.get('training_receipt')
        if (previous.get('phase') != 'consumed' or not prior
                or prior['new_policy_sha256'] != contract.policy_sha256):
            raise ValueError('draft has no exact previous consumed cursor')
    elif contract.policy_sha256 != manifest['initialization']['original_actor_sha256']:
        raise ValueError('draft update zero differs from original initialization')
    terminal = {}
    for leaf_id, draw_id in mapping.items():
        draw = state['draws'].get(draw_id)
        if not draw or draw['status'] not in ('valid', 'invalid', 'missing'):
            raise ValueError('draft has uncommitted episode slots')
        selected = draw['selected_attempt']
        episode = state.get('pending_episodes', {}).get(selected) if selected else None
        completion = state['attempts'][selected]['completion'] if selected else None
        if draw['status'] == 'valid' and episode is None:
            raise ValueError('draft valid slot lacks committed episode artifact')
        if episode is not None:
            _artifact_file(episode)
        terminal[leaf_id] = dict(draw=draw, completion=completion, episode=episode)
    return dict(schema_version=1, manifest_sha256=store.manifest.sha256,
        code_revision_sha256=manifest['code_revision_sha256'], graph_contract_sha256=contract.sha256,
        update_id=contract.update_id, policy_sha256=contract.policy_sha256,
        objective_config_sha256=digest(asdict(config)),
        ppo_window_config_sha256=digest(algorithm['ppo_window_config']),
        collection_plan_sha256=digest(plan), previous_consumed_receipt_sha256=digest(prior),
        terminal_episode_inventory_sha256=digest(terminal), batch_metadata_sha256=digest(_draft_value(draft)))


def _checked_draft_reference(store, contract, config, reference, *, expected_identity=None):
    from .action_credit_native_controller import _trusted_torch
    path = _artifact_file(reference)
    directory = _safe_path(store.root / 'driver_drafts' / str(contract.update_id))
    if path.parent != directory:
        raise ValueError('committed draft file is outside its exact update directory')
    payload = _trusted_torch(reference)
    identity = _draft_identity(store, contract, payload['draft'], payload['config'])
    if (payload['config'] != config or payload.get('identity') != identity
            or reference.get('batch_identity_sha256') != digest(identity)
            or reference.get('graph_contract_sha256') != contract.sha256
            or (expected_identity is not None and identity != expected_identity)):
        raise ValueError('committed draft conflicts with the exact source/batch/cursor identity')
    from .action_credit_tensor_refs import bind_tensor_inventory
    bind_tensor_inventory(store, payload['draft']['tensor_payloads'].values(), payload['tensor_owners'])
    return reference


def _stream_draft(store, contract, draft, config):
    """Commit a fresh exclusive file; uncommitted attempts are never adopted.

    The caller rebuilds from the immutable settled episodes. An interruption at
    any file-write boundary leaves evidence; a retry uses a fresh attempt path.
    This operation never consumes an episode, advances a cursor, or dispatches.
    """
    import torch
    from .action_credit_tensor_refs import tensor_inventory
    identity = _draft_identity(store, contract, draft, config)
    old = store.snapshot().get('driver_drafts', {}).get(str(contract.update_id))
    if old is not None:
        return _checked_draft_reference(store, contract, config, old, expected_identity=identity)
    owners = tensor_inventory(store, draft['tensor_payloads'].values())
    directory = _safe_path(store.root / "driver_drafts" / str(contract.update_id))
    directory.mkdir(parents=True, exist_ok=True)
    path = _safe_path(directory / ("draft-" + uuid.uuid4().hex + ".pt"))
    with path.open("xb") as handle:
        torch.save(dict(draft=draft, config=config, tensor_owners=owners, identity=identity), handle)
        handle.flush(); os.fsync(handle.fileno())
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    reference = dict(path=str(path), sha256=_hash(path), graph_contract_sha256=contract.sha256,
                     batch_identity_sha256=digest(identity))
    return _persist(store, "driver_drafts", str(contract.update_id), reference)


def native_driver_operation(actor, command):
    """Actual actor RPC helper: verified restore, then prepare transfer version.

    This label assignment does NOT establish that inference weights changed.
    The driver must next await all eight actual update_weights calls and query
    every served engine. Restarts can thus preserve update-specific versions.
    """
    from .action_credit_native_controller import MegatronRuntime, NativeUpdateController
    if command["operation"] != "restore_for_transfer":
        raise ValueError("unknown native driver RPC")
    manifest = CampaignManifest.from_dict(_json(command["manifest"]))
    store = CampaignStore(command["store_root"], manifest)
    runtime = MegatronRuntime(actor)
    controller = NativeUpdateController(store, runtime, command["native_acceptance"])
    contract = _contract(command["contract"])
    saved = controller._restore_inventory(command["checkpoint_inventory"])
    if (saved["policy_sha256"] != contract.policy_sha256 or saved["completed_updates"] != contract.update_id
            or actor.args.debug_train_only or actor.args.debug_rollout_only
            or getattr(actor.args, "keep_old_actor", False)
            or type(actor.weight_updater.weight_version) is not int):
        raise ValueError("verified active policy/cursor and ordinary native transfer required")
    actor.weight_updater.weight_version = contract.update_id
    return dict(policy_sha256=contract.policy_sha256, update_id=contract.update_id,
                expected_served_version=str(contract.update_id+1), checkpoint_inventory=command["checkpoint_inventory"])


class PilotDriver:
    def __init__(self, args, actor_model, rollout_manager, launch, *, worker_pool_factory,
                 request_pins, collection_acceptance, storage_admission=None, completion_hook=None):
        if not callable(storage_admission):
            raise ValueError('live storage-admission provider is required by the pilot driver')
        if type(launch) is not DriverLaunch:
            raise ValueError("typed explicit pilot launch required")
        launch.validate()
        if (args.offload_train or args.offload_rollout or args.debug_train_only or args.debug_rollout_only
                or getattr(args, "use_critic", False)):
            raise ValueError("pilot driver requires the validated nonoffloaded actor/rollout profile")
        self.launch, self.args, self.rollout_manager = launch, args, rollout_manager
        self.handles = _real_handlers(actor_model)
        self.store = CampaignStore(launch.store_root, CampaignManifest.from_dict(_json(launch.manifest)))
        self.initial = _contract(launch.initial_contract)
        self.config = ObjectiveConfig(**launch.objective_config)
        manifest = self.store.manifest.to_dict()
        if (manifest["arm_id"] != self.config.arm_id
                or manifest["initialization"]["original_actor_sha256"] != self.initial.policy_sha256
                or manifest["science"]["dataset"]["task_order_sha256"] != launch.task_order["sha256"]
                or manifest["science"]["algorithm"]["objective_config_sha256"] != digest(asdict(self.config))):
            raise ValueError("driver differs from the frozen scientific campaign")
        self.worker_pool_factory, self.request_pins = worker_pool_factory, request_pins
        self.collection_acceptance, self.completion_hook = collection_acceptance, completion_hook
        self.storage_admission = storage_admission
        self.common = dict(manifest=launch.manifest, store_root=launch.store_root, native_acceptance=launch.native_acceptance)

    def deadline(self):
        if time.time() >= self.launch.deadline_unix:
            raise TimeoutError("admitted allocation deadline reached; no new dispatch")

    async def guarded(self, update_id, phase, operation, *, reused=False):
        return await storage_guarded(self.store, self.storage_admission, 'optimizer-update-'+str(update_id),
            phase, self.launch.execution_id, operation, reused=reused)

    async def native(self, operation, **kwargs):
        self.deadline()
        return _unanimous(await _call_all(self.handles, "action_credit_operation", dict(self.common, operation=operation, **kwargs)))

    async def freeze(self, contract, inventory, rng_reference):
        self.deadline()
        command = dict(self.common, operation="restore_for_transfer", contract=asdict(contract), checkpoint_inventory=inventory)
        _unanimous(await _call_all(self.handles, "action_credit_driver_operation", command))
        await _call_all(self.handles, "update_weights")
        engines, _, _, counts, offsets = await self.rollout_manager.get_updatable_engines_and_lock.remote()
        if (len(engines) != 4 or any(engine is None for engine in engines)
                or counts != [2, 2, 2, 2] or offsets != [0, 2, 4, 6]):
            raise ValueError("exactly four owned TP2 inference engines on all eight GPUs required")
        urls = await _owned([engine.get_url.remote() for engine in engines])
        if len(urls) != 4 or any(not isinstance(url, str) or not url for url in urls):
            raise ValueError("all four inference engines must return nonempty URLs")
        urls = tuple(sorted(url.rstrip("/")+"/generate" for url in urls))
        if len(set(urls)) != 4:
            raise ValueError("direct actor engine endpoints must be nonempty and unique")
        for url in urls:
            parsed = urlsplit(url)
            if parsed.scheme != "http" or not parsed.hostname or parsed.port is None or parsed.path != "/generate":
                raise ValueError("unexpected direct engine URL")
        versions = await _owned([served_weight_version(url) for url in urls])
        expected = str(contract.update_id+1)
        if any(version != expected for version in versions):
            raise ValueError("actual weight transfer did not expose the exact restored policy version")
        value = dict(schema_version=1, policy_sha256=contract.policy_sha256, graph_contract_sha256=contract.sha256,
                     checkpoint_inventory=inventory, rng_reference=rng_reference, endpoints=urls,
                     weight_version=expected, all_eight_transfer_rpcs_awaited=True)
        directory = self.store.root / "driver_freezes" / str(contract.update_id)
        path = directory / (digest(value)+".json")
        reference = dict(path=str(path), sha256=_hash(path)) if path.exists() else _write(path, value)
        if _json(reference) != json.loads(json.dumps(value)):
            raise ValueError("persisted inference freeze changed")
        _persist(self.store, "driver_freezes", reference["sha256"], dict(reference, update_id=contract.update_id))
        by_url = {url: FrozenActor(contract.policy_sha256, expected, rng_reference["sha256"], reference["sha256"], digest(url)) for url in urls}
        def resolve(request):
            envelope = self.store.snapshot().get("online_requests", {}).get(request.request_id)
            if envelope is not None:
                # Loading/judge-only recovery retains the original physical
                # envelope. A fresh request chooses a current direct engine.
                frozen = FrozenActor(**envelope["actor"])
                frozen.validate(contract)
                if frozen.weight_version != expected:
                    raise ValueError("saved request belongs to a different weight version")
                return frozen
            return by_url[urls[int(digest(request.request_id), 16) % len(urls)]]
        def endpoint(actor):
            matches = [url for url in urls if digest(url) == actor.endpoint_sha256]
            if matches:
                return matches[0]
            prior = self.store.snapshot().get("driver_freezes", {}).get(actor.freeze_receipt_sha256)
            if prior:
                old_urls = _json(prior)["endpoints"]
                matches = [url for url in old_urls if digest(url) == actor.endpoint_sha256]
            if len(matches) != 1:
                raise ValueError("actor endpoint is absent from its retained physical freeze")
            return matches[0]
        return resolve, endpoint

    async def collect(self, contract, inventory, rng):
        draft = await self.guarded(contract.update_id, 'collection',
                                   lambda: self._collect(contract, inventory, rng))
        reference = await self.guarded(contract.update_id, 'draft',
            lambda: _stream_draft(self.store, contract, draft, self.config))
        del draft
        gc.collect()
        return reference

    async def _collect(self, contract, inventory, rng):
        resolve, endpoint = await self.freeze(contract, inventory, rng)
        pool = self.worker_pool_factory(self.store, contract, resolve, endpoint)
        if inspect.isawaitable(pool):
            pool = await pool
        if getattr(pool, "concurrency", None) != self.launch.worker_concurrency:
            error = ValueError("isolated worker pool differs from the approved64-worker profile")
            try:
                await pool.close()
            except BaseException as cleanup_error:
                error.add_note('Owned collection worker cleanup also failed: '+repr(cleanup_error))
            raise error
        slots = asyncio.Semaphore(self.launch.worker_concurrency)
        async def dispatch(request, prefix):
            # Check AFTER waiting for capacity: queued coroutines must not
            # launch a browser on an expired allocation just because they were
            # created before its deadline. The pool still owns each child.
            async with slots:
                self.deadline()
                return await pool.collect(request, prefix, resolve(request))
        async def resume(episode):
            async with slots:
                self.deadline()
                return await pool.resume_pending(episode, resolve(episode.request))
        failure = None
        try:
            draft = await collect_online_update(self.store, contract, resolve, self.launch.task_order["path"],
                dispatch=dispatch, request_pins=self.request_pins, attempt_namespace=self.launch.execution_id,
                verifier_sha256=physical_verifier_sha256(), tree_policy=_tree_policy(self.launch.tree_policy), resume_pending=resume)
        except BaseException as error:
            failure = error
            raise
        finally:
            try:
                await pool.close()
            except BaseException as cleanup_error:
                if failure is None:
                    raise
                failure.add_note('Owned collection worker cleanup also failed: '+repr(cleanup_error))
        return draft

    async def run(self):
        # Own all phases, not only native optimizer writes. A second driver may
        # not transfer actor weights while another driver collects browsers.
        with _safe_path(self.store.root / "driver-owner.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return await self._run()

    async def _run(self):
        evaluations = []
        for update_id in range(self.launch.target_updates):
            self.deadline()
            state = self.store.snapshot()
            current = state.get("updates", {}).get(str(update_id), {})
            if current.get("phase") == "consumed":
                await self.guarded(update_id, 'native-update-checkpoint', lambda: None, reused=True)
                if self.completion_hook is not None:
                    review = state["native_completed_updates"][str(update_id)]["review"]
                    evaluations.append(await self._completed_hook(update_id, dict(committed=True,
                        receipt=current["training_receipt"], review=review)))
                continue
            if any(int(k) > update_id and v.get("phase") == "consumed" for k, v in state.get("updates", {}).items()):
                raise ValueError("training cursor contains a hole")
            if update_id == 0:
                inventory = await self.guarded(update_id, 'native-initialize',
                    lambda: self.native("initialize", contract=asdict(self.initial)))
                contract = self.initial
            else:
                prior = state["updates"].get(str(update_id-1), {})
                if prior.get("phase") != "consumed":
                    raise ValueError("previous full update was not committed")
                inventory = state["native_completed_updates"][str(update_id-1)]["checkpoint_inventory"]
                if inventory["sha256"] != prior["training_receipt"]["new_checkpoint_sha256"]:
                    raise ValueError("next checkpoint differs from previous consumed receipt")
                contract = replace(self.initial, update_id=update_id, policy_sha256=prior["training_receipt"]["new_policy_sha256"])
            pointer = state.get("pending_batches", {}).get(str(update_id))
            if pointer is None:
                rng = state.get("driver_rng", {}).get(str(update_id))
                if rng is None:
                    rng = await self.guarded(update_id, 'restore-rng', lambda: self.native(
                        "restore_capture", contract=asdict(contract), checkpoint_inventory=inventory,
                        capture_id=f"{self.launch.execution_id}:{update_id}"))
                    _persist(self.store, "driver_rng", str(update_id), rng)
                else:
                    await self.guarded(update_id, 'restore-rng', lambda: None, reused=True)
                draft = state.get("driver_drafts", {}).get(str(update_id))
                if draft is None:
                    draft = await self.collect(contract, inventory, rng)
                else:
                    await self.guarded(update_id, 'draft', lambda: _checked_draft_reference(self.store, contract, self.config, draft), reused=True)
                if draft["graph_contract_sha256"] != contract.sha256:
                    raise ValueError("saved draft belongs to a different policy/update")
                _artifact_file(draft)
                pointer = await self.guarded(update_id, 'score-save',
                    lambda: self.native("score_save", draft_artifact=draft, optimizer_rng_artifact=rng))
            else:
                await self.guarded(update_id, 'score-save', lambda: None, reused=True)
            acceptance = await self.collection_acceptance(self.store, contract, pointer)
            _artifact_file(acceptance)
            execution = f"{self.launch.execution_id}:{update_id}"
            # Reconcile a saved review from any earlier driver attempt before
            # considering a new replay. Incomplete attempts retain their files.
            attempts = self.store.snapshot().get("driver_update_attempts", {})
            for value in attempts.values():
                candidate = self.store.root / "native_updates" / str(update_id) / digest(value["execution_id"]) / "review.json"
                if value["update_id"] == update_id and candidate.exists():
                    execution = value["execution_id"]
                    break
            _persist(self.store, "driver_update_attempts", execution, dict(update_id=update_id, execution_id=execution))
            result = await self.guarded(update_id, 'native-update-checkpoint', lambda: self.native(
                "update", update_id=update_id, execution_id=execution,
                collection_acceptance=acceptance, deadline_unix=self.launch.deadline_unix))
            if result.get("committed") is not True or self.store.snapshot()["updates"][str(update_id)]["phase"] != "consumed":
                raise ValueError("native update returned without a complete production commit")
            if self.completion_hook is not None:
                evaluations.append(await self._completed_hook(update_id, result))
        complete = self.store.snapshot()
        if any(complete["updates"].get(str(i), {}).get("phase") != "consumed" for i in range(10)):
            raise ValueError("training endpoint is incomplete")
        return dict(training_complete=True, completed_updates=10, evaluation_reviews=evaluations,
                    evaluation_complete_verified_by_driver=False)

    async def _completed_hook(self, update_id, result):
        recorded = self.store.snapshot().get("driver_completion_hooks", {})
        async def complete():
            if str(update_id) in recorded:
                return recorded[str(update_id)]["result"]
            value = await self.completion_hook(self.store, update_id+1, result)
            # A hook that crashed after saving an evaluation must reconcile its
            # own durable artifacts when called again; training is never repeated.
            _persist(self.store, "driver_completion_hooks", str(update_id), dict(result=value))
            return value
        if update_id == 9:
            return await self.guarded(update_id, 'iteration10-evaluation', complete, reused=str(update_id) in recorded)
        return await complete()


async def run_pilot(args, actor_model, rollout_manager, launch, **dependencies):
    return await PilotDriver(args, actor_model, rollout_manager, launch, **dependencies).run()
