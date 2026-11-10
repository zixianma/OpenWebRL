"""Actual-model, validation-only native acceptance entry.

Invoke run_native_gate(actor) in an ISOLATED actor.train early-return branch.
OPENWEBRL_ACTION_CREDIT_NATIVE_GATE names a pinned JSON manifest. No allocation,
browser, judge, download, training publication or production enablement occurs.
Historical real browser inputs are off-policy mechanics fixtures. Diagnostic
ancestry/reward labels are synthetic; this gate cannot certify a browser sampler.

The gate executes the actual initialized Qwen3-VL SFT model: frozen scoring,
five objective optimizer updates, full local optimizer-shard gradient comparison
between replicated and partitioned DP batches, TP log-probability/PPO oracle on
actual model logits, 32K real-image forward/backward, native model/Adam/scheduler
save and destructive-IN-MEMORY perturb/reload. Every file remains in the private
approved output directory. Parent controller supplies wall-time enforcement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

from openwebrl.action_credit import (ARMS, CollectionContract, ResponseEdge, ResponseGraph,
    Leaf, LossPlan, digest, leaf_loss_plan, append_local_terms)
from openwebrl.action_credit_advantages import Outcome, Episode, Step, outcome_group_advantages, tree_grpo_advantages, gigpo_advantages
from openwebrl.action_credit_objective import ObjectiveConfig, PreparedObjective
from openwebrl.action_credit_megatron import (BackendRow, EdgeIterator, tensor_payload_sha256,
    native_frozen_scores, finalize_frozen_scores, native_train_iterator, check_topology)
from openwebrl.action_credit_megatron import configure_native_training


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def writable_output(path):
    lexical=Path(path).absolute()
    forbidden=Path("/gpfs/home/zixianma")
    if lexical==forbidden or forbidden in lexical.parents:
        raise ValueError("gate output cannot traverse the forbidden home boundary")
    path = lexical.resolve()
    roots = (Path("/gpfs/scrubbed/zixianma/openwebrl-runtime"), Path("/tmp"))
    if not any(path == root or root in path.parents for root in roots):
        raise ValueError("gate output must resolve beneath scrubbed runtime or /tmp")
    return path


def validate_fixture(fixture):
    import torch
    if (fixture.get("schema_version") != 1 or fixture.get("origin") != "saved_browser_rows_off_policy_mechanics_only"
            or not isinstance(fixture.get("source", fixture.get("source_provenance")), dict) or not isinstance(fixture.get("rows"), list)
            or len(fixture["rows"]) < 2):
        raise ValueError("bounded real-browser mechanics fixture with source provenance required")
    normalized = []
    for row in fixture["rows"]:
        tokens = row["tokens"].tolist() if isinstance(row["tokens"], torch.Tensor) else row["tokens"]
        mask = row["loss_mask"].tolist() if isinstance(row["loss_mask"], torch.Tensor) else row["loss_mask"]
        response_length = row["response_length"]
        if (not isinstance(tokens, list) or any(type(x) is not int or x < 0 for x in tokens)
                or type(response_length) is not int or not 0 < response_length < len(tokens) <= 32768
                or len(mask) != response_length or any(x not in (0, 1, False, True) for x in mask) or not any(mask)):
            raise ValueError("invalid exact saved tokens/response/mask")
        tensors = row["multimodal_train_inputs"]
        if not isinstance(tensors, dict) or not tensors or "pixel_values" not in tensors:
            raise ValueError("actual multimodal image tensors required for every fixture row")
        payload_sha = tensor_payload_sha256(tensors)
        normalized.append(dict(input_ids=tuple(tokens[:-response_length]), response_ids=tuple(tokens[-response_length:]),
            sampled_mask=tuple(bool(x) for x in mask), tensors=tensors, tensor_sha256=payload_sha,
            sample_metadata=row.get("sample_metadata", {})))
    if len({digest((r["input_ids"], r["response_ids"], r["tensor_sha256"])) for r in normalized}) < 2:
        raise ValueError("at least two distinct actual saved actor rows required")
    if len({r["response_ids"] for r in normalized}) < 2:
        raise ValueError("mechanics gate requires two distinct sampled responses")
    return normalized


def mechanics_graph(contract, saved_rows, arm):
    """Synthetic topology over real model inputs; never production collection."""
    if arm not in ARMS:
        raise ValueError("unknown mechanics arm")
    edges, rows, tensors = [], {}, {}
    independent = arm in ("chain_outcome_grpo", "gigpo")
    keys = ("good", "bad") if independent else ("prefix", "good", "bad")
    for i, key in enumerate(keys):
        saved = saved_rows[i % len(saved_rows)]
        if arm in ("continuation_q", "gigpo") and key in ("good", "bad"):
            # Deliberate mechanics-only counterfactual composition: same real
            # image/context, two DIFFERENT archived response token sequences.
            # This is not asserted to be an archived executed browser branch.
            response = saved_rows[0 if key == "good" else 1]
            saved = {**saved_rows[0], "response_ids": response["response_ids"], "sampled_mask": response["sampled_mask"]}
        tree = key if independent else "diagnostic-tree"
        parent = None if independent or key == "prefix" else "prefix"
        # Tensor digest substitutes a diagnostic source-media identity ONLY here;
        # production collector must use hashes of actual source image bytes.
        edge = ResponseEdge(key, "diagnostic-task", tree, parent, contract.sha256,
            saved["input_ids"], (saved["tensor_sha256"],), saved["response_ids"], saved["sampled_mask"],
            (0.,)*len(saved["response_ids"]), contract.policy_sha256,
            digest(("diagnostic-state", parent)), digest(("not-browser-proof", key)),
            30 if parent is None else 29, "pending_frozen_actor_score")
        edges.append(edge)
        tensors[key] = saved["tensors"]
        rows[key] = BackendRow(key, edge.input_ids, edge.response_ids, edge.sampled_mask,
            edge.media_sha256, contract.processor_sha256, saved["tensor_sha256"])
    leaves = tuple(Leaf(key, "diagnostic-task", key if independent else "diagnostic-tree",
        (key,) if independent else ("prefix", key), "valid", reward, group_id="diagnostic-group")
        for key, reward in (("good", 1.), ("bad", 0.)))
    return ResponseGraph(contract, tuple(edges), leaves, tuple(l.leaf_id for l in leaves)), rows, tensors


def mechanics_objective(graph, arm):
    """Mechanics-only arm weights; production uses build_objective + seals."""
    outcomes = tuple(Outcome(l.leaf_id, l.task_id, l.group_id, l.reward, l.status, l.tree_id) for l in graph.leaves)
    fn = tree_grpo_advantages if arm == "tree_grpo" else outcome_group_advantages
    result = fn(outcomes, expected_ids=tuple(l.leaf_id for l in graph.leaves), singleton="zero")
    plan = leaf_loss_plan(graph, {v.key[0]: v.advantage for v in result.values}, arm_id=arm, reduction="response_mean")
    if arm == "continuation_q":
        plan = append_local_terms(plan, ({"good": .5, "bad": -.5},), coefficient=1.)
    if arm == "gigpo":
        edges = graph.validate(independent_chains=True)
        episodes = tuple(Episode(l.leaf_id, l.task_id, l.group_id, l.reward,
            (Step(l.edge_ids[0], digest((edges[l.edge_ids[0]].state_sha256,
                edges[l.edge_ids[0]].context_sha256, edges[l.edge_ids[0]].remaining_turns)), l.reward, 0),), "valid")
            for l in graph.leaves)
        result = gigpo_advantages(episodes, expected_ids=tuple(l.leaf_id for l in graph.leaves),
            normalization="mean_std", singleton="zero", step_weight=1., episode_weighting="trajectory")
        lookup = {v.key:v.advantage for v in result.values}
        plan = replace(plan, terms=tuple(replace(t, advantage=lookup[(t.source_id,t.edge_id)]) for t in plan.terms))
        plan.validate()
    config = ObjectiveConfig(arm, "response_mean", "mean_std", "zero", .2, .28, None,
        1. if arm == "continuation_q" else 0., 1. if arm == "gigpo" else 0., 1., "trajectory", "state_history_horizon")
    return PreparedObjective(config, plan, (result,))


def repeat_window(graph, rows, tensors, *, count=256):
    """Synthetic occurrence multiplicity over immutable real input tensors."""
    if count != 256 or len(graph.leaves) != 2 or any(len(l.edge_ids) != 1 for l in graph.leaves):
        raise ValueError("full accumulation probe requires two single-response source leaves and GBS256")
    edges, leaves, repeated_rows, repeated_tensors = [], [], {}, {}
    by_id = graph.validate(independent_chains=True)
    for index in range(count):
        leaf = graph.leaves[index % 2]
        original = by_id[leaf.edge_ids[0]]
        key = f"accumulation-{index:03d}"
        edges.append(replace(original, edge_id=key, tree_id=key))
        leaves.append(replace(leaf, leaf_id=key, tree_id=key, edge_ids=(key,)))
        repeated_rows[key] = replace(rows[original.edge_id], edge_id=key)
        repeated_tensors[key] = tensors[original.edge_id]
    return ResponseGraph(graph.contract, tuple(edges), tuple(leaves), tuple(l.leaf_id for l in leaves)), repeated_rows, repeated_tensors


def window_probe_inputs(graph,rows,tensors):
    """260 synthetic base occurrences plus a SEPARATE local candidate panel.

    This diagnostic deliberately combines shared-prefix leaf credit and a local
    candidate population; it is NOT a production Q campaign or independent-chain
    sampling fixture. Both extra scientific provenance and collection-role
    validation remain with the sealed production collector.
    """
    if len(graph.leaves)!=2 or any(len(l.edge_ids)!=2 for l in graph.leaves):
        raise ValueError("window probe requires the two-leaf shared-prefix diagnostic")
    leaves=tuple(replace(leaf,leaf_id=f"window-{leaf.leaf_id}-{i:03d}")
                 for i in range(65) for leaf in graph.leaves)
    all_edges=list(graph.edges)
    extra=[]
    rows,tensors=dict(rows),dict(tensors)
    by_id={e.edge_id:e for e in graph.edges}
    for key,reward in (("good",1.),("bad",0.)):
        new_key="q-"+key
        edge=replace(by_id[key],edge_id=new_key,tree_id="diagnostic-local-tree",parent_edge_id=None,
                     remaining_turns=30,state_sha256=digest("diagnostic-local-state"))
        all_edges.append(edge)
        extra.append(Leaf(new_key,edge.task_id,edge.tree_id,(new_key,),"valid",reward,group_id="diagnostic-local-panel"))
        rows[new_key]=replace(rows[key],edge_id=new_key)
        tensors[new_key]=tensors[key]
    expanded=replace(graph,edges=tuple(all_edges),leaves=(*leaves,*extra),
                     planned_leaf_ids=tuple(l.leaf_id for l in (*leaves,*extra)))
    return expanded,rows,tensors


def window_probe_objective(graph):
    base=tuple(l for l in graph.leaves if l.leaf_id.startswith("window-"))
    if len(base)!=130: raise ValueError("window diagnostic requires 130 two-response base leaves")
    result=outcome_group_advantages(tuple(Outcome(l.leaf_id,l.task_id,l.group_id,l.reward,l.status,l.tree_id) for l in base),
        expected_ids=tuple(l.leaf_id for l in base),singleton="zero")
    plan=leaf_loss_plan(graph,{v.key[0]:v.advantage for v in result.values},arm_id="continuation_q",
        reduction="response_mean",selected_leaf_ids=tuple(l.leaf_id for l in base))
    plan=append_local_terms(plan,({"q-good":.5,"q-bad":-.5},),coefficient=1.)
    config=ObjectiveConfig("continuation_q","response_mean","mean_std","zero",.2,.28,None,
        1.,0.,1.,"trajectory","state_history_horizon")
    return PreparedObjective(config,plan,(result,))


def run_window_probe(actor,contract,saved,manifest,backend_sha,check_time):
    """Exercise actual-model PPO2 driver, full gradients, tails and frozen scores."""
    from slime.backends.megatron_utils.model import train_one_step
    from openwebrl.action_credit_windows import WindowConfig,plan_ppo_windows,native_train_collection
    draft,rows,tensors=mechanics_graph(contract,saved,"continuation_q")
    draft,rows,tensors=window_probe_inputs(draft,rows,tensors)
    scores=native_frozen_scores(actor,draft,rows,tensors,scorer_source_sha256=backend_sha)
    graph=finalize_frozen_scores(draft,scores,rows=rows,scorer_source_sha256=backend_sha)
    prepared=window_probe_objective(graph)
    schedule=plan_ppo_windows(prepared,WindowConfig(2,256,17))
    score_hash=digest({key:receipt.sha256 for key,receipt in scores.items()})
    comparisons={}
    def before(window):
        check_time()
        configure_native_training(actor)
        replica=ReplicatedObjective(window.config,window.plan,())
        iterator=EdgeIterator(graph,rows,tensors,tuple(sorted({t.edge_id for t in window.plan.terms})),
            device="cuda",prepared=replica,global_batch_size=256)
        probe=GradientProbe(actor.optimizer)
        train_one_step(actor.args,0,window.plan.ordinal,[iterator],actor.model,probe,NoSchedulerUpdate(),len(iterator.slots))
        partitioned=native_train_iterator(actor,window,rows,tensors)
        comparison=GradientProbe(actor.optimizer,expected=probe.captured,
            atol=manifest["gradient_atol"],rtol=manifest["gradient_rtol"])
        train_one_step(actor.args,0,window.plan.ordinal,[partitioned],actor.model,comparison,
                       NoSchedulerUpdate(),len(partitioned.slots))
        comparisons[window.plan.ordinal]=comparison.report
        del probe.captured
    prior_epochs=actor.args.ppo_epochs
    actor.args.ppo_epochs=2
    try:
        report=native_train_collection(actor,prepared,schedule,rows,tensors,before_window=before)
    finally:
        actor.args.ppo_epochs=prior_epochs
    if score_hash!=digest({key:receipt.sha256 for key,receipt in scores.items()}):
        raise ValueError("native PPO2 recomputed frozen old scores")
    for item in report["windows"]:
        item["full_dp_gradient_parity"]=comparisons[item["ordinal"]]
        item["native_metrics"]={k:float(v) for k,v in item["native_metrics"].items()}
    if (report["optimizer_updates_in_memory"]!=4 or report["base_occurrence_exposures"]!=520
            or report["virtual_padding_slots"]!=504):
        raise ValueError("native PPO2 schedule lost occurrences or changed final tail")
    return dict(report,synthetic_topology_and_multiplicities=True,production_collection_provenance=False,
        full_collection_occurrences=260,shared_prefix_separate_leaf_terms=True,
        separate_base_and_local_populations=True,
        local_panels_repeated_every_window=True,frozen_score_receipts_sha256=score_hash,
        window_backend_source_sha256=file_sha256(Path(__file__).with_name("action_credit_windows.py")))


class ReplicatedObjective(PreparedObjective):
    """Gate-only oracle: every DP replica sees all globally weighted edges."""
    def loss(self, log_probs, *, edge_subset=None):
        from megatron.core import mpu
        return super().loss(log_probs, edge_subset=edge_subset) / mpu.get_data_parallel_world_size(False)


def optimizer_parameters(optimizer):
    if hasattr(optimizer, "chained_optimizers"):
        return [p for opt in optimizer.chained_optimizers for p in optimizer_parameters(opt)]
    return optimizer.get_parameters()


def optimizer_runtime_state(optimizer):
    """Include actual local Adam buffers; distributed wrapper state omits them."""
    if hasattr(optimizer, "chained_optimizers"):
        return [optimizer_runtime_state(opt) for opt in optimizer.chained_optimizers]
    return {"wrapper": optimizer.state_dict(), "inner": optimizer.optimizer.state_dict(),
            "main_parameters": [p.detach() for p in optimizer.get_parameters()]}


def first_adam_moment(optimizer):
    if hasattr(optimizer, "chained_optimizers"):
        for opt in optimizer.chained_optimizers:
            value = first_adam_moment(opt)
            if value is not None:
                return value
        return None
    for state in optimizer.optimizer.state.values():
        value = state.get("exp_avg")
        if value is not None and value.numel():
            return value
    return None


def compare_full_gradients(parameters, expected, *, atol, rtol):
    """Compare every local optimizer-shard gradient element before clipping."""
    import torch
    current = [p.grad for p in parameters]
    if len(current) != len(expected):
        raise ValueError("optimizer parameter count changed")
    bad, max_abs, max_scaled, count = 0, 0., 0., 0
    for actual, reference in zip(current, expected):
        if actual is None or reference is None:
            bad += int(actual is not None or reference is not None)
            continue
        if actual.shape != reference.shape:
            raise ValueError("optimizer shard shape changed")
        value = actual.detach().float().cpu()
        error = (value - reference).abs()
        finite = torch.isfinite(value) & torch.isfinite(reference)
        tolerance = atol + rtol * reference.abs()
        bad += int(((error > tolerance) | ~finite).sum())
        max_abs = max(max_abs, float(error.max()) if error.numel() else 0.)
        max_scaled = max(max_scaled, float((error / tolerance).max()) if error.numel() else 0.)
        count += error.numel()
    return dict(elements=count, outside_tolerance=bad, max_abs_error=max_abs, max_tolerance_ratio=max_scaled)


class GradientProbe:
    """Native optimizer interface, preserving parameters/Adam/scheduler."""
    def __init__(self, optimizer, *, expected=None, atol=1e-5, rtol=.03):
        self.optimizer, self.expected, self.atol, self.rtol = optimizer, expected, atol, rtol
        self.captured, self.report = None, None

    def __getattr__(self, key):
        return getattr(self.optimizer, key)

    def step(self):
        import torch
        import torch.distributed as dist
        found_inf = self.optimizer.prepare_grads()
        params = optimizer_parameters(self.optimizer)
        self.report = compare_full_gradients(params, [None if p.grad is None else p.grad.detach().float().cpu()
            for p in params] if self.expected is None else self.expected, atol=self.atol, rtol=self.rtol)
        if self.expected is None:
            self.captured = [None if p.grad is None else p.grad.detach().float().cpu().clone() for p in params]
        bad = torch.tensor(int(bool(found_inf)) + self.report["outside_tolerance"], dtype=torch.long, device="cuda")
        dist.all_reduce(bad)
        if bad.item():
            raise ValueError("actual-model DP full-gradient parity failed: " + str(self.report))
        return True, 0., 0


class NoSchedulerUpdate:
    def step(self, **kwargs):
        pass


def state_digest(value):
    """Streaming exact state fingerprint; no checkpoint payload is rewritten."""
    import torch
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            h.update(tensor_payload_sha256({"tensor": item}).encode())
        elif isinstance(item, dict):
            h.update(b"dict")
            for key in sorted(item, key=str):
                h.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (tuple, list)):
            h.update(type(item).__name__.encode())
            for part in item: visit(part)
        elif item is None or isinstance(item, (str, bool, int, float)):
            h.update(digest(item).encode())
        else:
            raise ValueError("unsupported native state fingerprint type: " + type(item).__name__)
    visit(value)
    return h.hexdigest()


def rng_component_sha256(cuda_state,tracker_states):
    """Fingerprint current Python/NumPy/CPUtorch plus supplied native GPU RNG."""
    import random
    import numpy as np
    import torch
    numpy_state=np.random.get_state()
    return dict(python=state_digest(random.getstate()),
        numpy=state_digest((numpy_state[0],numpy_state[1].tolist(),*numpy_state[2:])),
        torch_cpu=state_digest(torch.get_rng_state()),cuda=state_digest(cuda_state),
        megatron_tracker=state_digest(tracker_states))


def native_rng_sha256():
    import torch
    from megatron.core import tensor_parallel
    states=tensor_parallel.get_cuda_rng_tracker().get_states()
    if not states:raise ValueError("native checkpoint RNG tracker is empty")
    return rng_component_sha256(torch.cuda.get_rng_state(),
        {k:tensor_parallel.convert_cuda_rng_state(v,to_graphable=False) for k,v in states.items()})


def seed_native_rng_probe(rank,phase):
    """Set deliberately rank-distinct diagnostic states in every RNG component."""
    import random
    import numpy as np
    import torch
    from megatron.core import tensor_parallel
    seed=900000+1000*rank+100*phase
    random.seed(seed)
    np.random.seed(seed+1)
    torch.random.default_generator.manual_seed(seed+2)
    torch.cuda.manual_seed(seed+3)
    tracker=tensor_parallel.get_cuda_rng_tracker()
    for index,name in enumerate(sorted(tracker.get_states())):
        with tracker.fork(name):torch.cuda.manual_seed(seed+4+index)


def tp_oracle(args, batch, logits, native_logp):
    """Independent float64 full-vocabulary oracle on actual-model output rows.

    At most 8 response positions are gathered, bounding memory at ~10 MiB/TP
    rank for this vocabulary. Value/gradient checks cover asymmetric clipping
    with BOTH signs on a shared response, at ratios .7, 1, 1.5 and 4.
    """
    import torch
    import torch.distributed as dist
    from megatron.core import mpu
    from slime.utils.ppo_utils import calculate_log_probs_and_entropy
    edge = next(e for e in batch["action_credit_objective"].plan.graph.edges if e.edge_id == batch["action_credit_edge_id"])
    indices = {0, len(edge.response_ids)-1, *range(min(4, len(edge.response_ids)))}
    indices.add(next(i for i,value in enumerate(edge.sampled_mask) if value))
    if not all(edge.sampled_mask):
        indices.add(next(i for i,value in enumerate(edge.sampled_mask) if not value))
    indices = sorted(indices)
    selected = logits[0, [len(edge.input_ids)-1+i for i in indices]].detach().float().clone().requires_grad_(True)
    targets = torch.tensor([edge.response_ids[i] for i in indices], device=selected.device)
    group = mpu.get_tensor_model_parallel_group()
    tp, rank = dist.get_world_size(group), dist.get_rank(group)
    gathered = [torch.empty_like(selected) for _ in range(tp)]
    dist.all_gather(gathered, selected.detach(), group=group)
    full = torch.cat(gathered, -1).double().cpu().requires_grad_(True)
    full_scaled = full/args.rollout_temperature
    reference = full_scaled.log_softmax(-1).gather(-1, targets.cpu()[:,None]).flatten()
    observed, _ = calculate_log_probs_and_entropy(selected/args.rollout_temperature, targets, group)
    observed = observed.flatten()
    reference_values = reference.detach().float().to(observed.device)
    torch.testing.assert_close(observed, reference_values, atol=2e-4, rtol=2e-5)
    torch.testing.assert_close(native_logp[indices], reference_values, atol=2e-4, rtol=2e-5)
    errors, unquantized_errors = [], []
    selected_mask = torch.tensor([edge.sampled_mask[i] for i in indices], dtype=torch.bool)
    masks = [selected_mask]
    synthetic_mask_probe = bool(selected_mask.all()) and len(indices) > 1
    if synthetic_mask_probe:
        probe_mask = selected_mask.clone(); probe_mask[-1] = False
        masks.append(probe_mask)
    for mask, ratio in ((mask,ratio) for mask in masks for ratio in (.7, 1., 1.5, 4.)):
        # The installed fused CE backward consumes its saved softmax in-place.
        # Each independent derivative probe needs a fresh forward graph.
        probe_selected = selected.detach().clone().requires_grad_(True)
        probe_logp, _ = calculate_log_probs_and_entropy(
            probe_selected/args.rollout_temperature, targets, group)
        probe_logp = probe_logp.flatten()
        old = reference.detach() - math.log(ratio)
        cpu_ratio = (reference-old).exp()
        gpu_ratio = (probe_logp-old.float().to(probe_logp.device)).exp()
        def objective(r):
            r = r[mask.to(r.device)]
            return sum(-torch.minimum(r*a, r.clamp(.8, 1.28)*a).mean()*w for a,w in ((1.,.3),(-1.,.7)))
        cpu_loss, gpu_loss = objective(cpu_ratio), objective(gpu_ratio)
        cpu_grad = torch.autograd.grad(cpu_loss, full_scaled, retain_graph=True)[0]
        gpu_grad = torch.autograd.grad(gpu_loss, probe_selected)[0]
        shard = cpu_grad[:,rank*selected.shape[-1]:(rank+1)*selected.shape[-1]].float()
        # Installed fused_cross_entropy.calculate_gradients returns BF16 even
        # for FP32 input. Autograd restores the FP32 input dtype, then the
        # upstream temperature division applies. Quantize at that boundary,
        # not after division; retain the original numerical tolerances.
        expected = (shard.to(torch.bfloat16).float()/args.rollout_temperature).to(selected.device)
        unquantized_errors.append(float((gpu_grad-shard.to(selected.device)/args.rollout_temperature).abs().max()))
        torch.testing.assert_close(gpu_loss, cpu_loss.detach().float().to(selected.device), atol=2e-4, rtol=2e-5)
        torch.testing.assert_close(gpu_grad, expected, atol=2e-6, rtol=2e-4)
        if not bool((gpu_grad[~mask.to(gpu_grad.device)] == 0).all()):
            raise ValueError("transport-only response tokens contributed an oracle gradient")
        errors.append(float((gpu_grad-expected).abs().max()))
    return dict(positions=len(indices), logprob_max_abs_error=float((observed-reference_values).abs().max()),
                masked_positions=int((~selected_mask).sum()), masked_gradients_exact_zero=True,
                synthetic_mask_perturbation_tested=synthetic_mask_probe,
                ppo_gradient_max_abs_errors=errors,
                ppo_unquantized_gradient_max_abs_errors=unquantized_errors,
                native_ce_backward_dtype="bfloat16", fresh_backward_graph_per_probe=True)


def run_native_gate(actor):
    import torch
    import torch.distributed as dist
    from megatron.core import mpu
    from slime.backends.megatron_utils.model import train_one_step, save
    from slime.backends.megatron_utils.checkpoint import load_checkpoint
    manifest_path = Path(os.environ["OPENWEBRL_ACTION_CREDIT_NATIVE_GATE"])
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("diagnostic_only") is not True:
        raise ValueError("explicit native mechanics diagnostic manifest required")
    args, start = actor.args, time.monotonic()
    if Path(args.hf_checkpoint).resolve() != Path(manifest["hf_checkpoint"]).resolve():
        raise ValueError("actual model path differs from approved original SFT")
    if file_sha256(Path(args.hf_checkpoint)/"config.json") != manifest["hf_config_sha256"]:
        raise ValueError("actual model config changed")
    if (not torch.cuda.is_available() or len(actor.model) != 1 or args.async_save
            or getattr(args, "offload_train", False) or getattr(args, "overlap_param_gather", False)
            or not getattr(args, "check_for_nan_in_loss_and_grad", True)
            or getattr(args, "attention_dropout", 0.) != 0 or getattr(args, "hidden_dropout", 0.) != 0
            or getattr(args, "clip_grad", 0.) <= 0 or not getattr(args,"data_parallel_random_init",False)
            or getattr(args,"no_save_rng",False)):
        raise ValueError("native gate requires actual CUDA model and synchronous nonoverlapped optimizer")
    if manifest.get("stress_length") != 32768 or not 0 < manifest["max_seconds"] <= 21600:
        raise ValueError("explicit 32K gate and bounded approved validation time required")
    output = writable_output(manifest["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    result = dict(schema_version=1, production_enabled=False, browser_sampler_acceptance=False,
        input_origin="saved_browser_rows_off_policy_mechanics_only", manifest_sha256=file_sha256(manifest_path),
        native_gate_source_sha256=file_sha256(__file__), backend_source_sha256=file_sha256(Path(__file__).with_name("action_credit_megatron.py")),
        rank=rank, tp=mpu.get_tensor_model_parallel_world_size(), dp=mpu.get_data_parallel_world_size(False), cases=[])
    def write():
        (output/f"native-gate-rank{rank}.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    def check_time():
        if time.monotonic()-start >= manifest["max_seconds"]:
            raise TimeoutError("native acceptance consumed its bounded gate time")
    try:
        if Path(manifest["fixture_path"]).stat().st_size > 2*1024**3:
            raise ValueError("native gate fixture exceeds bounded 2GiB limit")
        if file_sha256(manifest["fixture_path"]) != manifest["fixture_sha256"]:
            raise ValueError("saved browser fixture changed")
        fixture = torch.load(manifest["fixture_path"], map_location="cpu", weights_only=True)
        saved = validate_fixture(fixture)
        # The entire source batch remains immutable; this copy is explicitly bounded.
        result["fixture_rows"] = len(saved)
        result["longest_real_context"] = max(len(r["input_ids"])+len(r["response_ids"]) for r in saved)
        result["actual_model_local_parameters"] = sum(p.numel() for model in actor.model for p in model.parameters())
        if result["actual_model_local_parameters"] * result["tp"] < 3_000_000_000:
            raise ValueError("tiny model cannot satisfy actual 4B actor acceptance")
        args.loss_type, args.custom_loss_function_path = "custom_loss", "openwebrl.action_credit_megatron.loss"
        args.entropy_coef, args.use_kl_loss, args.calculate_per_token_loss = 0., False, False
        check_topology(args, cp=mpu.get_context_parallel_world_size(), pp=mpu.get_pipeline_model_parallel_world_size())
        for case_index, arm in enumerate(sorted(ARMS)):
            check_time()
            contract = CollectionContract(case_index, *[digest((manifest["hf_config_sha256"], key, case_index)) for key in
                ("diagnostic-model-state", "tokenizer", "processor", "tasks", "judge", "generation")],
                args.rollout_temperature, ("diagnostic-task",), ())
            draft, rows, tensors = mechanics_graph(contract, saved, arm)
            scores = native_frozen_scores(actor, draft, rows, tensors, scorer_source_sha256=result["backend_source_sha256"])
            graph = finalize_frozen_scores(draft, scores, rows=rows, scorer_source_sha256=result["backend_source_sha256"])
            prepared = mechanics_objective(graph, arm)
            # Native training mode enables activation recomputation; dropout is
            # pinned to zero so scoring/training likelihoods remain comparable.
            configure_native_training(actor)
            replica = ReplicatedObjective(prepared.config, prepared.plan, prepared.diagnostics)
            reference_iterator = EdgeIterator(graph, rows, tensors, tuple(sorted({t.edge_id for t in prepared.plan.terms})),
                device="cuda", prepared=replica, global_batch_size=args.global_batch_size)
            oracle_reports = []
            reference_iterator.oracle = lambda *items: oracle_reports.append(tp_oracle(*items))
            probe = GradientProbe(actor.optimizer)
            train_one_step(args, 0, case_index, [reference_iterator], actor.model, probe,
                NoSchedulerUpdate(), len(reference_iterator.slots))
            sharded_iterator = native_train_iterator(actor, prepared, rows, tensors)
            comparison = GradientProbe(actor.optimizer, expected=probe.captured,
                atol=manifest["gradient_atol"], rtol=manifest["gradient_rtol"])
            train_one_step(args, 0, case_index, [sharded_iterator], actor.model, comparison,
                NoSchedulerUpdate(), len(sharded_iterator.slots))
            del probe.captured
            before = state_digest(actor.opt_param_scheduler.state_dict())
            metrics, norm = train_one_step(args, 0, case_index, [sharded_iterator.reset()], actor.model,
                actor.optimizer, actor.opt_param_scheduler, len(sharded_iterator.slots))
            if not math.isfinite(float(norm)) or float(norm) <= 0 or before == state_digest(actor.opt_param_scheduler.state_dict()):
                raise ValueError("actual native optimizer/scheduler failed to advance")
            result["cases"].append(dict(arm_id=arm, native_optimizer_update=True, grad_norm=float(norm),
                full_dp_gradient_parity=comparison.report, tp_oracles=oracle_reports,
                unique_edges=len({t.edge_id for t in prepared.plan.terms}), leaf_terms=len(prepared.plan.terms),
                frozen_score_receipts={key: value.sha256 for key,value in scores.items()},
                metrics={key:float(value) for key,value in metrics.items()}))
            write()
        check_time()
        if args.global_batch_size != 256:
            raise ValueError("representative native accumulation gate requires GBS256")
        draft, rows, tensors = mechanics_graph(contract, saved, "chain_outcome_grpo")
        scores = native_frozen_scores(actor, draft, rows, tensors, scorer_source_sha256=result["backend_source_sha256"])
        graph = finalize_frozen_scores(draft, scores, rows=rows, scorer_source_sha256=result["backend_source_sha256"])
        graph, rows, tensors = repeat_window(graph, rows, tensors)
        prepared = mechanics_objective(graph, "chain_outcome_grpo")
        iterator = native_train_iterator(actor, prepared, rows, tensors)
        configure_native_training(actor)
        torch.cuda.reset_peak_memory_stats()
        metrics, norm = train_one_step(args, 0, 5, [iterator], actor.model, actor.optimizer,
            actor.opt_param_scheduler, len(iterator.slots))
        if not math.isfinite(float(norm)) or float(norm) <= 0:
            raise ValueError("full native accumulation did not produce a valid optimizer update")
        result["full_accumulation"] = dict(global_responses=256, synthetic_multiplicities=True,
            local_microbatches=len(iterator.slots), grad_norm=float(norm), optimizer_updated=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write()
        check_time()
        result["ppo_windows"]=run_window_probe(actor,contract,saved,manifest,result["backend_source_sha256"],check_time)
        write()
        check_time()
        # 32K multimodal forward/backward, zero objective; records memory rather
        # than pretending this padded context is a real browser trajectory.
        longest = max(saved, key=lambda r: len(r["input_ids"])+len(r["response_ids"]))
        stress = {**longest, "input_ids": longest["input_ids"] + (0,)*(32768-len(longest["input_ids"])-len(longest["response_ids"]))}
        draft, rows, tensors = mechanics_graph(contract, [stress, saved[0]], "chain_outcome_grpo")
        scores = native_frozen_scores(actor, draft, rows, tensors, scorer_source_sha256=result["backend_source_sha256"])
        graph = finalize_frozen_scores(draft, scores, rows=rows, scorer_source_sha256=result["backend_source_sha256"])
        prepared = mechanics_objective(graph, "chain_outcome_grpo")
        prepared = replace(prepared, plan=replace(prepared.plan, terms=tuple(replace(t, advantage=0.) for t in prepared.plan.terms)))
        iterator = native_train_iterator(actor, prepared, rows, tensors)
        configure_native_training(actor)
        torch.cuda.reset_peak_memory_stats()
        train_one_step(args, 0, 6, [iterator], actor.model, GradientProbe(actor.optimizer), NoSchedulerUpdate(), len(iterator.slots))
        result["stress"] = dict(context=32768, actual_image_tensors=True, synthetic_text_padding=True,
            forward_backward=True, optimizer_updated=False, peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write(); check_time()
        checkpoint = output/"native-checkpoint"
        args.save = str(checkpoint)
        seed_native_rng_probe(rank,0)
        rng_before=native_rng_sha256()
        save(101, actor.model, actor.optimizer, actor.opt_param_scheduler)
        dist.barrier()
        def snapshot():
            return dict(model=state_digest([model.state_dict() for model in actor.model]),
                optimizer=state_digest(optimizer_runtime_state(actor.optimizer)), scheduler=state_digest(actor.opt_param_scheduler.state_dict()))
        before = snapshot()
        with torch.no_grad():
            next(actor.model[0].parameters()).view(-1)[0].add_(1.)
        actor.opt_param_scheduler.step(increment=args.global_batch_size)
        # Perturb actual local Adam storage, not metadata-only wrapper state.
        moment = first_adam_moment(actor.optimizer)
        mutated_adam = moment is not None
        if mutated_adam:
            moment.view(-1)[0].add_(1.)
        seed_native_rng_probe(rank,1)
        rng_perturbed=native_rng_sha256()
        if any(rng_perturbed[k]==rng_before[k] for k in rng_before):
            raise ValueError("native RNG perturbation did not exercise every component")
        args.load, args.ckpt_step = str(checkpoint), 101
        args.no_load_optim, args.no_load_rng, args.finetune = False, False, False
        loaded, _ = load_checkpoint(actor.model, actor.optimizer, actor.opt_param_scheduler, {}, False)
        rng_after=native_rng_sha256()
        after = snapshot()
        if loaded != 101 or before != after or not mutated_adam or rng_before!=rng_after:
            raise ValueError("native model/Adam/scheduler/per-rank RNG reload verification failed")
        result["reload"] = dict(loaded_iteration=loaded, exact_state_sha256=after, model_and_adam_perturbed=True,
            scheduler_perturbed=True, checkpoint=str(checkpoint),rng_restored=True,
            per_rank_rng_saved=True,rng_components_sha256_before=rng_before,
            rng_components_sha256_perturbed=rng_perturbed,rng_components_sha256_after=rng_after)
        result.update(passed=True, elapsed_seconds=time.monotonic()-start,
            pending_acceptance=["real online collector/sampler/replay provenance", "pilot initialization/cursor/budgets",
                                "production source and topology identical to validated backend"])
        write()
        return result
    except Exception as error:
        result.update(passed=False, error=type(error).__name__+": "+str(error), elapsed_seconds=time.monotonic()-start)
        write()
        raise
