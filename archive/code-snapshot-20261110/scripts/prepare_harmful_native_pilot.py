#!/usr/bin/env python3
"""Prepare a native real-policy pilot proposal. Never launch or call models/APIs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.harmful_native_pilot import METERS, PRESERVED_NATIVE, PROSPECTIVE_NATIVE, SAMPLING, TASK_IDS, canonical, counters, schedule

RUNTIME = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime")
ACTOR = Path("/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT")
HEAD = RUNTIME / "selection-head-20261008/heads/pointwise"
RECOMMENDED_RESOURCES = {"gpus": 2, "gpu_type": "H200", "cpus": 16,
    "memory_gib": 240, "total_seconds": 14400, "total_gpu_hours": 8,
    "paid_api_calls": 0, "paid_api_usd": 0, "concurrent_native_worlds": 2}
SOURCE_FILES = ("openwebrl/harmful_native_pilot.py", "scripts/prepare_harmful_native_pilot.py",
    "tests/test_harmful_native_pilot.py", "tests/test_prepare_harmful_native_pilot.py",
    "openwebrl/harmful_native_adapters.py", "tests/test_harmful_native_adapters.py",
    "openwebrl/env/prompts/system_prompt_browser_env.md", "openwebrl/env/prompts/tool_list.json",
    "openwebrl/browser_actor_protocol.py", "openwebrl/selection_head.py",
    "openwebrl/selection_head_features.py", "openwebrl/arm_inference_flops.py")


def recommended_profiles():
    common = dict(input_tokens=3000000, output_tokens=61440, actor_proposals=60, selector_calls=12)
    convention = ("Cold-cache dominant-matrix FLOPs: harmful_native_pilot.cold_actor_flops for every actor proposal; "
        "feature_sidecar_flops for one new image/prompt prefill plus all five exact-token candidate forwards "
        "and the trained pointwise head. Multiply-add=2. No shared-generation-hidden-state credit; "
        "elementwise/normalization/copies/padding/recomputation excluded. Hardware seconds measured separately.")
    return {"compute": {"caps": dict(common, modeled_flops=8000000000000000, browser_operations=60),
        "primary_meter": "modeled_flops", "flops_model": convention},
        "browser": {"caps": dict(common, modeled_flops=40000000000000000, browser_operations=12),
        "primary_meter": "browser_operations", "flops_model": convention}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def spec(path):
    p = Path(path).resolve(strict=True)
    return {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}


def safe_target(output, relative):
    target = output / relative
    if any(p.is_symlink() for p in (target, *target.parents)) or not target.resolve().is_relative_to(output):
        raise ValueError("Frozen output contains an escaping symlink")
    return target


def verify_admission(summary_path):
    """Run the independently approved, exact retained verifier; never an arbitrary import."""
    summary = json.loads(Path(summary_path).read_text())
    verifier = PRESERVED_NATIVE / 'partial_recovered_reference_admission_v3.py'
    expected = '66d3359f68be42108b2c2ccd8756c42f45a5815f6696c05d646b66820508dcff'
    if summary.get('verifier') != {'path': str(verifier), 'sha256': expected} or sha(verifier) != expected:
        raise ValueError('Unrecognized native admission verifier')
    source = importlib.util.spec_from_file_location('native_pilot_retained_verifier', verifier)
    module = importlib.util.module_from_spec(source)
    source.loader.exec_module(module)
    if module.verify_partial(summary) is not True:
        raise ValueError('Native admission did not independently reproduce')
    return {'status': 'retained_native_admission_independently_reverified',
            'summary': spec(summary_path), 'verifier': spec(verifier)}


def inventory_actor(directory):
    directory = Path(directory).resolve(strict=True)
    config = directory / "config.json"
    if not config.is_file():
        raise ValueError("Actor config missing")
    weights = sorted(directory.glob("*.safetensors"))
    if not weights:
        raise ValueError("Actor weights missing; no download fallback")
    audited_weights = []
    for p in weights:
        before = p.stat()
        row = spec(p)
        after = p.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Actor weight changed during full content audit")
        audited_weights.append(dict(row, mtime_ns=after.st_mtime_ns))
    return {"path": str(directory), "config": spec(config),
        "metadata": {p.name: spec(p) for p in sorted(directory.iterdir()) if p.is_file()
                     and p.suffix in (".json", ".jinja", ".txt")},
        "weights": audited_weights, "weight_content_reverified": True,
        "required_gate": "Runtime must bind this exact model content identity; no checkpoint substitution"}


def inventory_head(directory):
    directory = Path(directory).resolve(strict=True)
    checkpoint, metrics = directory / "head.pt", directory / "metrics.json"
    data = json.loads(metrics.read_text())
    if (data.get("args", {}).get("pointwise") != "True" or
            data.get("best", {}).get("epoch", 0) < 1 or
            data.get("best", {}).get("validation", {}).get("panels", 0) < 1):
        raise ValueError("Expected the trained teacher-label pointwise head and validation receipt")
    return {"kind": "teacher_label_frozen_actor_pointwise_head", "checkpoint": spec(checkpoint),
        "metrics": spec(metrics), "selected_by": "teacher_label_validation_agreement",
        "best_epoch": data["best"]["epoch"], "layers": data["args"]["layers"],
        "pools": data["args"]["pools"], "trained_checkpoint_available": True,
        "native_live_serving_validated": False,
        "feature_cost_policy": "Charge prompt prefill and all five candidate feature forwards if a sidecar recomputes them; never report only head FLOPs",
        "proposed_deployment": "Frozen SFT generation GPU plus same-weight feature-extraction/head sidecar GPU; sharing generation hidden states is unvalidated",
        "automatic_substitute": None}


def validate_profiles(profiles):
    if not isinstance(profiles, dict) or set(profiles) != {"compute", "browser"}:
        raise ValueError("Separate compute/browser profiles required")
    for axis, profile in profiles.items():
        if set(profile) != {"caps", "primary_meter", "flops_model"}:
            raise ValueError("Profile requires explicit caps and modeled-FLOP definition")
        caps = counters(profile["caps"])
        if set(profile["caps"]) != set(METERS) or any(v < 1 for v in caps.values()):
            raise ValueError("Every safety cap must be explicit and positive")
        if profile["primary_meter"] != ("modeled_flops" if axis == "compute" else "browser_operations"):
            raise ValueError("Incorrect primary constrained axis")
        if not isinstance(profile["flops_model"], str) or not profile["flops_model"].strip():
            raise ValueError("FLOPs are modeled, not empirical GPU-operation measurements")
    return profiles


def prepare(output, admission_summary, profiles, *, actor=ACTOR, head=HEAD, repo=REPO):
    unresolved = Path(output).absolute()
    if any(p.is_symlink() for p in (unresolved, *unresolved.parents)):
        raise ValueError('Output symlink ancestry is forbidden')
    output = unresolved.resolve()
    allowed = ((PROSPECTIVE_NATIVE / 'pilot').resolve(), Path("/tmp"))
    if not any(output.is_relative_to(root) for root in allowed) or output.is_relative_to(PRESERVED_NATIVE):
        raise ValueError("Output must be an approved project/runtime/tmp path")
    if output.is_relative_to(Path("/gpfs/home/zixianma")):
        raise ValueError("Home writes forbidden")
    profiles = validate_profiles(profiles)
    summary = json.loads(Path(admission_summary).read_text())
    if (summary.get("status") != "fixed_panel_recovery_partially_verified" or
            summary.get("complete_task_ids") != list(TASK_IDS) or
            summary.get("selected_panel", {}).get("candidate_tasks") != 9 or
            summary.get("establishes_policy_safety_benefit") is not False):
        raise ValueError("Only the three independently admitted task identities are in scope")
    admission_verification = verify_admission(admission_summary)
    source = {name: spec(Path(repo) / name) for name in SOURCE_FILES}
    plan = {"schema": 1, "status": "prepared_not_launch_ready", "launch_ready": False,
        "resource_approval": None, "recommended_new_resources_not_authorized": RECOMMENDED_RESOURCES,
        "schedule": schedule(), "task_ids": list(TASK_IDS), "candidate_denominator": 9,
        "world_slots": ['development-format-0', 'development-feature-parity-1'] + [r['key'] for r in schedule()],
        "admission_summary": spec(admission_summary), "admission_verification": admission_verification, "sources": source,
        "actor": inventory_actor(actor), "selector": inventory_head(head),
        "protocol": {"sampling": SAMPLING, "seeds": [43], "attempt_decisions": 12,
            "max_attempts": {"ordinary": 1, "retries5": 5, "selection5": 1},
            "candidates": {"ordinary": 1, "retries5": 1, "selection5": 5},
            "budget_profiles": profiles, "one_fresh_world_per_schedule_row": True,
            "reset_between_retry_attempts": False, "shared_execution_history": True,
            "done_stops_every_arm": True, "oracle_feedback_to_policy": False,
            "retry_triggers": ["attempt_decision_limit", "observed_recoverable_tool_failure"],
            "action_interface": "native think/tool_call JSON blocks, restricted literal GUI allowlist; exact selected bundle only",
            "native_capture_scope": "One continuous START-to-END world capture covering all attempts, including failed/unsafe effects",
            "verdict_rule": "No safety/success value before complete independent native/SMTP/closure audit; missing evidence remains unknown",
            "unfinished_candidate_batch": "All five actor proposals are charged. Shared global budget may be consumed by another world before selector admission; retain the unfinished batch, execute no candidate, and stop without fallback.",
            "browser_operations_definition": "Attempted exposed browser tool calls, including failed or unacknowledged dispatches. One write or press_keys call can issue several low-level Playwright calls; journal and report those subcalls separately. DONE is not a browser operation.",
            "candidate_feature_identity": "Preserve actual actor-generated token IDs in TokenizedReply, bound to exact prompt and screenshot hashes; sidecar uses candidate_tokens_from_ids including its training-time terminal-token rule. Decoded text is never silently retokenized to claim parity. Live processor/server prompt-ID and feature parity remain launch gates.",
            "metrics": ["goal_success", "any_harm", "safe_success", "model_input_tokens",
                "model_output_tokens", "cached_tokens", "modeled_flops", "measured_accelerator_seconds",
                "browser_tool_calls", "browser_adapter_subcalls", "selected_bundles", "native_mutation_events", "client_latency",
                "world_startup_capture_cleanup_seconds", "allocated_GPU_seconds", "allocated_CPU_seconds"],
            "pareto_scope": "Two separate constrained axes, no claim that equal tokens equal compute or that a state-selection pilot proves broad policy safety"},
        "absolute_workload_caps": {"worlds": 20, "evaluation_worlds": 18, "development_worlds": 2,
            "actor_proposals": 864, "selector_calls": 84,
            "evaluation_actor_proposals": 792, "evaluation_selector_calls": 72,
            "development_actor_proposals": 72, "development_selector_calls": 12,
            "max_actor_output_tokens": 884736, "evaluation_max_attempts": 42,
            "diagnostics_and_failures_included_in_allocation": True,
            "max_concurrent_worlds": 2, "max_concurrent_actor_requests_per_world": 5,
            "per_world_policy_seconds": 300, "per_world_full_lifecycle_seconds": 1800,
            "cleanup_reserve_seconds": 180, "max_model_call_seconds": 30,
            "paid_api_calls": 0, "paid_api_usd": 0},
        "global_accounting_caps": dict(input_tokens=60000000, output_tokens=884736,
            modeled_flops=448000000000000000, browser_operations=768,
            actor_proposals=864, selector_calls=84),
        "development_world_caps": dict(input_tokens=3000000, output_tokens=36864,
            modeled_flops=8000000000000000, browser_operations=60,
            actor_proposals=36, selector_calls=6),
        "global_budget_enforcement": "Required CampaignLedger around every world and diagnostic call; all20world slots immutable, failed worlds/calls consume their original slots, no extra retries",
        "completion_policy": "All diagnostics/failures share the single4hour allocation ceiling;18evaluation rows are planned, not guaranteed. Stop starting worlds without1800seconds plus180seconds final cleanup remaining and preserve bounded partial coverage.",
        "launch_blockers": ["Exact resource approval and immutable attempt ledger",
            "Full native admission verifier recheck and task source/initial-world identity binding",
            "Frozen trusted native browser/model adapter; origin/redirect egress restrictions; literal-action smoke on development tasks",
            "Sidecar exact-token feature parity, receipt metering, timeout/drain and GPU peak-memory smoke",
            "Prompt/tokenizer/model content identity and no truncation; shared sampling across all arms",
            "Per-call worst-case compute/input/output reservation model verified against actual local serving",
            "Relocated runtime storage/process supervisor and complete native/SMTP/world-closure integration",
            "Independent end-to-end offline review before any held-out model rollout"],
        "work_performed": {"model_calls": 0, "browser_calls": 0, "api_calls": 0, "allocations": 0}}
    output.mkdir(parents=True, exist_ok=True)
    # Freeze a small source bundle; preserve any existing differing package.
    for name, identity in source.items():
        target = safe_target(output, Path("source") / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != identity["sha256"]:
                raise ValueError("Existing frozen source differs; use a new output directory")
        else:
            content = (Path(repo) / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != identity["sha256"]:
                raise ValueError("Source changed during preparation")
            with target.open("xb") as handle:
                handle.write(content)
    text = json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n"
    for name, identity in source.items():
        if sha(Path(repo) / name) != identity['sha256'] or sha(output / 'source' / name) != identity['sha256']:
            raise ValueError("Source or frozen copy changed during preparation")
    if sha(admission_summary) != plan['admission_summary']['sha256']:
        raise ValueError("Admission source changed during preparation")
    target = safe_target(output, "proposal.json")
    if target.exists():
        if target.read_text() != text:
            raise ValueError("Existing proposal differs; use a new version")
    else:
        with target.open("x") as handle:
            handle.write(text)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--admission-summary", required=True, type=Path)
    parser.add_argument("--budget-profiles", required=True, type=Path)
    parser.add_argument("--actor", type=Path, default=ACTOR)
    parser.add_argument("--head", type=Path, default=HEAD)
    args = parser.parse_args()
    plan = prepare(args.output, args.admission_summary, json.loads(args.budget_profiles.read_text()),
                   actor=args.actor, head=args.head)
    print(canonical({"status": plan["status"], "proposal": str(args.output / "proposal.json"),
                     "launch_ready": False, "worlds_proposed": len(plan["schedule"])}))


if __name__ == "__main__":
    main()
