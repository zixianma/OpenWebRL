"""Prospective recovery of missing native controls, without rewriting old evidence.

This module prepares a bounded request. It neither allocates resources nor starts
an application. Native integration and exact resource approval are separate gates.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

CASES = ("correct", "wrong_target", "duplicate", "wrong_then_correct")
TASKS = tuple(range(470, 475))
EXPECTED_MISSING = {
    (472, "correct"), (472, "duplicate"), (472, "wrong_then_correct"),
    (474, "duplicate"), (474, "wrong_then_correct"),
}
READ_ROOT = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime")
FORBIDDEN_ROOT = Path("/gpfs/home/zixianma")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def checked_path(path, root):
    """Reject lexical escapes and symlink ancestors before any write/read binding."""
    path, root = Path(path), Path(root)
    need(path.is_absolute() and root.is_absolute(), "Absolute paths required")
    need(".." not in path.parts and (path == root or root in path.parents), "Path outside designated root")
    need(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink path refused")
    resolved = path.resolve()
    need(resolved != FORBIDDEN_ROOT and FORBIDDEN_ROOT not in resolved.parents, "Home boundary refused")
    return path


def binding(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "Expected regular bound file")
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def bound_json(spec, root):
    need(isinstance(spec, dict) and isinstance(spec.get("sha256"), str), "Missing source binding")
    path = checked_path(spec["path"], root)
    need(digest(path) == spec["sha256"], "Bound evidence changed")
    return json.loads(path.read_text())


def resource_request():
    return {
        "schema_version": 1, "status": "proposed_not_authorized",
        "nodes": 1, "cpus": 4, "memory_gib": 16, "gpus": 0,
        "total_allocation_seconds_including_all_attempts": 7200,
        "maximum_allocated_cpu_hours": 8,
        "actor_calls": 0, "selector_calls": 0, "judge_calls": 0,
        "api_budget_usd": 0, "internet_download_bytes": 0,
        "maximum_fresh_application_worlds_including_diagnostics_and_failures": 12,
        "maximum_concurrent_application_worlds": 2,
        "node_local_scratch_cap_bytes": 80 * 1024**3,
        "new_retained_artifacts_cap_bytes": 10 * 1024**3,
        "case_total_seconds": 1800, "case_closure_reserve_seconds": 120,
        "allocation_closure_reserve_seconds": 180,
        "browser_body_seconds_per_reference": 120,
        "browser_operations_per_reference": 60,
        "old_unused_seconds_carried_forward": 0,
        "separate_pilot_budget_added": 0,
        "model_comparison_in_scope": False,
        "gitlab_or_notification_deployment_in_scope": False,
        "automatic_budget_extension": False,
        "live_agent_supervision_required": True,
        "partial_endpoint_required_if_target_does_not_fit_remaining_budget": True,
    }


def validate_request(request):
    need(json.dumps(request, sort_keys=True, allow_nan=False) ==
         json.dumps(resource_request(), sort_keys=True, allow_nan=False),
         "Recovery resource request differs from frozen proposal")
    need(request["case_closure_reserve_seconds"] < request["case_total_seconds"], "No productive case interval")


def derive_recovery_cells(summary, selected_cases, original_cases):
    """Use the Cartesian task/case denominator; never repeat an already valid cell."""
    need(summary.get("status") == "fixed_panel_recovery_partially_verified", "Expected verified partial endpoint")
    need(summary.get("fixed_panel_complete") is False, "Completed panel cannot be recovered")
    panel = summary["selected_panel"]
    need(panel == {"admitted_tasks": 3, "candidate_tasks": 9,
                   "development_references": 4, "source_references": 15}, "Historical denominator changed")
    expected = {(task, case) for task in TASKS for case in CASES}
    complete = set()
    for value in selected_cases:
        reference, result = value["reference"], value["result"]
        key = (reference["task_id"], reference["case"])
        need(type(key[0]) is int and key in expected and key not in complete, "Unexpected or repeated valid control")
        need(value.get("status") == "reference_passed" and result.get("valid") is True,
             "Unverified reference cannot be preserved as valid")
        complete.add(key)
    need(len(complete) == 15, "Missing verified source control")
    missing = expected - complete
    need(missing == EXPECTED_MISSING, "Recovery would change the frozen five missing cells")
    complete_tasks = sorted(task for task in TASKS if all((task, case) in complete for case in CASES))
    need(complete_tasks == summary["complete_task_ids"] == [470, 471, 473], "Complete-task admission differs")
    originals = {}
    for row in original_cases:
        if row.get("phase") != "held_out" or row.get("task_id") not in TASKS:
            continue
        key = (row["task_id"], row["case"])
        need(key in expected and key not in originals, "Original control manifest is not unique")
        originals[key] = row
    need(set(originals) == expected, "Original fixed target is incomplete")
    return [
        {"task_id": task, "case": case,
         "failed_original_instance": originals[(task, case)]["planned_instance"],
         "new_instance_namespace": f"recovery-task{task}-{case}",
         "old_attempt_retained": True, "repeat_valid_capture": False}
        for task, case in sorted(missing)
    ]


def build_plan(summary, selected_cases, original_cases):
    cells = derive_recovery_cells(summary, selected_cases, original_cases)
    assignments = {
        f"instance-{index:04d}": {"phase": "development", "task_id": "dev_cancel_explicit", "case": case}
        for index, case in enumerate(CASES, 2)
    }
    assignments.update({
        f"instance-{index:04d}": dict(cell, phase="held_out")
        for index, cell in enumerate(cells, 6)
    })
    return {
        "schema_version": 1,
        "status": "prospective_recovery_prepared_pending_integration_and_exact_approval",
        "launch_authorized": False, "native_integration_verified": False,
        "historical_valid_source_controls_preserved": 15,
        "historical_invalid_source_attempts_preserved": 5,
        "source_candidates": 9, "cancellation_tasks": list(TASKS),
        "fixed_cancellation_target_controls": 20,
        "recovery_cells": cells,
        "reference_assignments": assignments,
        "development_control_cases": list(CASES),
        "development_and_source_use_same_prospective_runtime": True,
        "native_restore_diagnostic_worlds": 1,
        "planned_application_worlds_before_any_retry": 10,
        "maximum_additional_failed_attempt_worlds": 2,
        "new_worlds_never_reuse_original_instance_directories": True,
        "case_timing_change": {
            "historical_total_seconds": 700, "prospective_total_seconds": 1800,
            "prospective_emergency_closure_reserve_seconds": 120,
            "reason": "Admit the full bounded serial native capture and normal closure chain before starting; never borrow emergency time for successful-case closure.",
            "historical_failed_cases_reclassified": False,
            "browser_body_seconds_unchanged": 120, "browser_operations_unchanged": 60,
            "full_target_completion_guaranteed_within_global_budget": False,
        },
        "retry_may_only_repeat_a_failed_or_uncollected_cell": True,
        "conditions_before_source_controls": [
            "Exact prospective sources reviewed and frozen; old sources untouched",
            "Exact allocation request separately approved and scheduler identity verified",
            "Remaining time includes every prior attempt and guaranteed cleanup reserve",
            "Manifest-aware archive restore and full filesystem inventory verified",
            "Fresh diagnostic clone passes complete native DB/Redis/search reset gates",
            "Prospective process/storage supervisor passes native integration and closure",
            "All four development controls pass unchanged harm/goal/mail/capture predicates",
        ],
        "final_admission_rule": "Keep historical and recovery protocol versions separate; combine only fully verified cells with matching task/contracts/reset identity; require all four cells per admitted task.",
        "partial_results": "Preserve each started attempt and cost; missing, deadline-failed or unsupported evidence is unknown, never safe.",
        "notification_and_gitlab_tasks_in_scope": False,
        "model_comparison_in_scope": False,
        "budget": resource_request(),
    }


def validate_exact_approval(approval, proposal_sha256, request, *, job_id, elapsed_seconds):
    """Pure admission check; this does not manufacture an approval or submit a job."""
    validate_request(request)
    need(approval.get("status") == "approved", "Exact new allocation approval missing")
    need(approval.get("proposal_sha256") == proposal_sha256, "Approval binds a different proposal")
    need(json.dumps(approval.get("resource_request"), sort_keys=True, allow_nan=False) ==
         json.dumps(request, sort_keys=True, allow_nan=False), "Exact resource request not approved")
    need(isinstance(job_id, str) and job_id.isdigit() and job_id != "355424", "A new approved allocation is required")
    need(approval.get("slurm_job_id") == job_id, "Wrong allocation identity")
    need(isinstance(elapsed_seconds, (int, float)) and not isinstance(elapsed_seconds, bool)
         and math.isfinite(elapsed_seconds) and elapsed_seconds >= 0, "Invalid elapsed time")
    remaining = request["total_allocation_seconds_including_all_attempts"] - elapsed_seconds
    need(remaining > request["allocation_closure_reserve_seconds"], "Budget exhausted or reserved for closure")
    return remaining
