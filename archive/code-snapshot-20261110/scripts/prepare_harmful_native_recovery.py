#!/usr/bin/env python3
"""Freeze exact missing controls and a bounded request; never allocate or launch."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.harmful_native_recovery import (
    READ_ROOT, binding, bound_json, build_plan, checked_path, digest, need,
)

HISTORICAL = READ_ROOT / "evaluations/harmful-apps-20261010/shopping-native-validation-20261011"
PROSPECTIVE = READ_ROOT / "evaluations/harmful-apps-20261010/harmful-native-recovery-20261011-v1"
VERIFIER_SHA256 = "66d3359f68be42108b2c2ccd8756c42f45a5815f6696c05d646b66820508dcff"


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    sync_directory(Path(path).parent)


def compact_binding(path):
    value = binding(path)
    return {key: value[key] for key in ("path", "sha256")}


def native_inputs(output, plan):
    """Freeze private relocation inputs; no source import starts native services."""
    adapter = PROSPECTIVE / "native-adapter"
    source = adapter / "candidate-02"
    need(source.is_dir(), "Prospective native source directory is absent")
    spec = importlib.util.spec_from_file_location("native_adapter_source_inventory", adapter / "preflight.py")
    inventory_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inventory_module)
    inventory_module.SOURCE = source
    bundle = inventory_module.inventory()
    # The frozen launcher independently rehashes this known read-only addon
    # through its public symlink immediately before native execution. Bind the
    # canonical regular source here without relaxing writable-path checks.
    addon = READ_ROOT / "webprm-webarena-20261011/native-fakeroot-addon/usr/lib/libfakeroot.so"
    for field in ("files", "additional_sources"):
        for index, item in enumerate(bundle[field]):
            if item["path"] == str(addon):
                target = addon.resolve(strict=True)
                need(addon.parent == target.parent and target.name == "libfakeroot-0.so" and
                     digest(target) == item["sha256"], "Native addon link identity changed")
                bundle[field][index] = compact_binding(target)
    bundle["status"] = "prospectively_frozen_native_adapter"
    # Pure historical contract/assembler inputs are read-only lineage, never
    # live entrypoints from the released allocation.
    old = json.loads((HISTORICAL / "reference-controller-freeze-01.json").read_text())
    readonly = {item["path"]: item for item in bundle["read_only_files"]}
    for item in old["artifacts"].values():
        readonly[item["path"]] = compact_binding(item["path"])
    files = {item["path"]: item for item in bundle["files"]}
    files.update(readonly)
    bundle["files"] = [files[key] for key in sorted(files)]
    bundle["read_only_files"] = [readonly[key] for key in sorted(readonly)]
    for item in bundle["files"]:
        need(digest(item["path"]) == item["sha256"], "Native source changed during inventory")
        if item["path"].endswith(".py"):
            compile(Path(item["path"]).read_bytes(), item["path"], "exec")
    request_binding = compact_binding(output / "resource-request.json")
    review_path = adapter / "storage-reservation-review-03.json"
    review = json.loads(review_path.read_text())
    need(review["status"] == "conservative_storage_reservations_reviewed" and
         review["request_sha256"] == request_binding["sha256"] and
         review["actual_concurrency_limit"] == 1 and
         review["hard_storage_quota_claimed"] is False,
         "Storage review differs from exact request or guard semantics")
    evidence = [review["source_assumptions"], *review["source_evidence_reverified"],
                *review["browser_inputs"]]
    for item in evidence:
        need(digest(item["path"]) == item["sha256"], "Reviewed storage source changed")
    independent_path = adapter / "independent-storage-review-03.json"
    independent = json.loads(independent_path.read_text())
    need(independent["status"] == "reviewed_operational_reservations_accepted" and
         independent["review"] == compact_binding(review_path),
         "Independent reservation review is absent or differs")
    review = dict(review, source_review=compact_binding(review_path),
                  independent_review=compact_binding(independent_path))
    # The growth values remain conservative operational reservations, not a
    # claim that arbitrary native writes are enforced by a filesystem quota.
    values = {"storage-reservations.json": review,
              "native-adapter-bundle.json": bundle}
    for name, value in values.items():
        write_json(output / name, value)
    for name, field in (("diagnostic-plan.json", "native_diagnostic_plan"),
                        ("reference-plan.json", "native_reference_plan")):
        value = json.loads((adapter / name).read_text())
        value["storage_bounds_review"] = compact_binding(output / "storage-reservations.json")
        need(sum(stage["seconds"] for stage in value["stages"]) <=
             plan["budget"]["case_total_seconds"] - plan["budget"]["case_closure_reserve_seconds"],
             "Normal native stages do not fit before reserved emergency cleanup")
        write_json(output / name, value)
        plan[field] = compact_binding(output / name)
    campaign = json.loads((adapter / "campaign-plan.json").read_text())
    write_json(output / "campaign-plan.json", campaign)
    plan["native_campaign_plan"] = compact_binding(output / "campaign-plan.json")
    plan["native_preparation_evidence"] = [compact_binding(path) for path in (
        review_path, independent_path, adapter / "diagnostic-plan.json",
        adapter / "reference-plan.json", adapter / "campaign-plan.json")]
    plan["native_adapter_bundle"] = compact_binding(output / "native-adapter-bundle.json")
    plan["native_additional_sources"] = bundle["additional_sources"]
    plan["native_runtime_scopes"] = {
        "runtime_root": str(PROSPECTIVE / "run-01"),
        "historical_root": str(HISTORICAL), "proposal_root": str(inventory_module.P),
        "source_root": str(source), "scratch_root_template": "/tmp/harm-native-recovery-{job_id}"}
    cached = READ_ROOT / "webprm-baselines-20261009"
    assets = {"python": cached / "browser-env/bin/python",
              "pyvenv": cached / "browser-env/pyvenv.cfg",
              "chromium": READ_ROOT / "browsers/chromium-1208/chrome-linux64/chrome",
              "credentials": HISTORICAL.parent / "shopping-admin-probe-20261011/credentials-private.json"}
    plan["native_browser_assets"] = {
        key: {"path": str(path), "resolved_path": str(path.resolve(strict=True)), "sha256": digest(path)}
        for key, path in assets.items()}
    plan["launch_readiness"].update(relocated_native_worker_bound=True,
        scripted_reference_browser_bridge_bound=True, native_actor_bridge_bound=False,
        live_baseline_restore_verified=False, exact_new_allocation_approved=False)
    return bundle


def prepare(output):
    output = checked_path(output, PROSPECTIVE)
    need(output != PROSPECTIVE, "Use a fresh freeze directory inside the prospective campaign")
    need(not output.exists(), "Never overwrite a prospective freeze")
    summary_path = HISTORICAL / "source-admission-partial-03/partial-summary.json"
    summary = json.loads(summary_path.read_text())
    verifier = HISTORICAL / "partial_recovered_reference_admission_v3.py"
    need(summary["verifier"] == {"path": str(verifier), "sha256": VERIFIER_SHA256}, "Wrong historical verifier")
    need(digest(verifier) == VERIFIER_SHA256, "Historical verifier changed")
    sys.path.insert(0, str(HISTORICAL))
    spec = importlib.util.spec_from_file_location("historical_native_partial_admission", verifier)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = bound_json(summary["selected_case_paths"], HISTORICAL)
    case_paths = [checked_path(path, HISTORICAL) for path in paths]
    cases = [json.loads(path.read_text()) for path in case_paths]
    original_path = HISTORICAL / "reference-campaign-plan.json"
    original = json.loads(original_path.read_text())
    historical_snapshot = {str(path): digest(path) for path in [summary_path, original_path, *case_paths]}
    need(module.verify_partial(summary), "Historical partial evidence no longer verifies")
    plan = build_plan(summary, cases, original["cases"])
    contract_path = HISTORICAL / "durable-initialized-baseline-01/restore-contract.json"
    contract = json.loads(contract_path.read_text())
    need(contract["status"] == "durable_baseline_restore_contract_sealed", "Archive restore contract absent")
    for name in ("helper", "manifest", "evidence", "owner_closure"):
        item = contract[name]
        need(digest(checked_path(item["path"], HISTORICAL)) == item["sha256"], "Restore source changed: " + name)
    archive = checked_path(contract["archive"]["path"], HISTORICAL)
    need(archive.stat().st_size == contract["archive"]["bytes"], "Archive length differs")
    plan["historical_evidence"] = {
        "partial_summary": binding(summary_path),
        "selected_case_paths": summary["selected_case_paths"],
        "selected_case_artifacts": [binding(path) for path in case_paths],
        "original_campaign": binding(original_path),
        "original_controller_freeze": binding(HISTORICAL / "reference-controller-freeze-01.json"),
        "restore_contract": binding(contract_path),
        "original_initialized_baseline": binding(HISTORICAL / "initialized-baseline/receipt.json"),
        "archive": contract["archive"],
        "archive_bytes_rehashed_in_this_preparation": False,
        "archive_rehash_required_before_restore": True,
        "original_contract_freeze": binding(HISTORICAL / "reference-contract-freeze-01/freeze.json")
        if (HISTORICAL / "reference-contract-freeze-01/freeze.json").exists()
        else binding(HISTORICAL / "reference-controller-freeze-01.json"),
    }
    # Source inventory intentionally lists only prospective files, never task payloads.
    names = ["openwebrl/harmful_native_recovery.py", "scripts/prepare_harmful_native_recovery.py",
             "openwebrl/harmful_native_runtime.py", "tests/test_harmful_native_runtime.py",
             "openwebrl/harmful_native_restore.py", "tests/test_harmful_native_restore.py",
             "tests/test_harmful_native_recovery.py"]
    sources = {name: binding(REPO / name) for name in names}
    plan["prospective_sources"] = sources
    plan["launch_readiness"] = {
        "offline_components_prepared": True,
        "relocated_native_worker_bound": False,
        "native_actor_bridge_bound": False,
        "live_baseline_restore_verified": False,
        "exact_new_allocation_approved": False,
        "must_not_use_historical_job_bound_entrypoints_directly": True,
    }
    output.mkdir(parents=True, mode=0o700)
    for name, value in (("resource-request.json", plan["budget"]),
                        ("source-inventory.json", sources)):
        write_json(output / name, value)
    native_bundle = native_inputs(output, plan)
    for name, value in sources.items():
        source = Path(value["path"])
        data = source.read_bytes()
        need(hashlib.sha256(data).hexdigest() == value["sha256"], "Prospective source changed during freeze")
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        sync_directory(target.parent)
        need(digest(target) == value["sha256"], "Frozen source copy differs")
    need(all(digest(path) == expected for path, expected in historical_snapshot.items()),
         "Historical evidence changed while preparing recovery")
    need(module.verify_partial(summary), "Historical custody changed before prospective freeze")
    need(all(digest(Path(value["path"])) == value["sha256"] for value in sources.values()),
         "Prospective source changed before proposal seal")
    need(all(digest(item["path"]) == item["sha256"] for item in native_bundle["files"]),
         "Native bundle changed before proposal seal")
    need(all(digest(item["path"]) == item["sha256"] for item in plan["native_preparation_evidence"]),
         "Native preparation evidence changed before proposal seal")
    need(all(str(Path(item["path"]).resolve(strict=True)) == item["resolved_path"] and
             digest(item["path"]) == item["sha256"] for item in plan["native_browser_assets"].values()),
         "Browser assets changed before proposal seal")
    # The proposal is the last artifact: a failed or interrupted freeze cannot
    # leave an apparently complete request whose source copies are absent.
    for directory in sorted((p for p in output.rglob("*") if p.is_dir()),
                            key=lambda p: len(p.parts), reverse=True):
        sync_directory(directory)
    sync_directory(output)
    sync_directory(output.parent)
    write_json(output / "proposal.json", plan)
    return {"status": plan["status"], "proposal": binding(output / "proposal.json"),
            "recovery_cells": plan["recovery_cells"], "launch_authorized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))


if __name__ == "__main__":
    main()
