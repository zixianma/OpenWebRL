import copy
import json
from pathlib import Path

import tempfile
import unittest

from openwebrl.harmful_native_recovery import (
    CASES, EXPECTED_MISSING, TASKS, binding, bound_json, build_plan,
    checked_path, derive_recovery_cells, resource_request, validate_exact_approval,
)


def inputs():
    summary = {"status": "fixed_panel_recovery_partially_verified", "fixed_panel_complete": False,
               "selected_panel": {"admitted_tasks": 3, "candidate_tasks": 9,
                                  "development_references": 4, "source_references": 15},
               "complete_task_ids": [470, 471, 473]}
    original = [{"phase": "held_out", "task_id": t, "case": c,
                 "planned_instance": f"instance-{i:04d}"}
                for i, (t, c) in enumerate(((t, c) for t in TASKS for c in CASES), 12)]
    selected = [{"status": "reference_passed", "reference": {"task_id": t, "case": c},
                 "result": {"valid": True}}
                for t in TASKS for c in CASES if (t, c) not in EXPECTED_MISSING]
    return summary, selected, original


def approval():
    request = resource_request()
    return request, {"status": "approved", "proposal_sha256": "a" * 64,
                     "resource_request": copy.deepcopy(request), "slurm_job_id": "400000"}



class RecoveryTests(unittest.TestCase):
    def test_recovery_preserves_existing_valid_cells_and_failure_lineage(self):
        summary, selected, original = inputs()
        before = copy.deepcopy((summary, selected, original))
        plan = build_plan(summary, selected, original)
        assert (summary, selected, original) == before
        assert {r["failed_original_instance"] for r in plan["recovery_cells"]} == {
            "instance-0020", "instance-0022", "instance-0023", "instance-0030", "instance-0031"}
        assert not plan["launch_authorized"]
        assert not plan["native_integration_verified"]
        assert plan["planned_application_worlds_before_any_retry"] == 10


    def test_cannot_silently_change_recovery_denominator(self):
        for change in ['duplicate', 'invalid', 'wrong_cell', 'missing', 'wrong_tasks', 'denominator', 'original_duplicate']:
            with self.subTest(change=change):
                summary, selected, original = inputs()
                if change == "duplicate": selected[-1] = copy.deepcopy(selected[0])
                elif change == "invalid": selected[0]["result"]["valid"] = False
                elif change == "wrong_cell": selected[0]["reference"] = {"task_id": 472, "case": "correct"}
                elif change == "missing": selected.pop()
                elif change == "wrong_tasks": summary["complete_task_ids"] = [470, 471, 474]
                elif change == "denominator": summary["selected_panel"]["candidate_tasks"] = 5
                elif change == "original_duplicate": original.append(copy.deepcopy(original[0]))
                with self.assertRaises(ValueError): derive_recovery_cells(summary, selected, original)


    def test_budget_subtracts_all_attempts_without_carryover(self):
        request, value = approval()
        assert validate_exact_approval(value, "a" * 64, request, job_id="400000", elapsed_seconds=1000) == 6200


    def test_approval_and_budget_fail_closed(self):
        for change in ['not_approved', 'wrong_proposal', 'more_cpu', 'more_time', 'old_job', 'wrong_job', 'negative', 'nan', 'boolean', 'reserve', 'boolean_request', 'float_approval']:
            with self.subTest(change=change):
                request, value = approval()
                job, elapsed = "400000", 0
                if change == "not_approved": value["status"] = "proposed_not_authorized"
                elif change == "wrong_proposal": value["proposal_sha256"] = "b" * 64
                elif change == "more_cpu": value["resource_request"]["cpus"] = 8
                elif change == "more_time": request["total_allocation_seconds_including_all_attempts"] = 14400
                elif change == "old_job": job = value["slurm_job_id"] = "355424"
                elif change == "wrong_job": job = "400001"
                elif change == "negative": elapsed = -1
                elif change == "nan": elapsed = float("nan")
                elif change == "boolean": elapsed = True
                elif change == "reserve": elapsed = 7020
                elif change == "boolean_request": request["nodes"] = True
                elif change == "float_approval": value["resource_request"]["nodes"] = 1.0
                with self.assertRaises(ValueError): validate_exact_approval(value, "a" * 64, request, job_id=job, elapsed_seconds=elapsed)


    def test_path_boundaries_and_binding_tamper(self):
        with tempfile.TemporaryDirectory(prefix="harmful-recovery-test-", dir="/tmp") as directory:
            self._check_paths(Path(directory))

    def _check_paths(self, tmp_path):
        source = tmp_path / "source.json"
        source.write_text(json.dumps({"valid": True}))
        spec = binding(source)
        assert bound_json(spec, tmp_path) == {"valid": True}
        source.write_text(json.dumps({"valid": False}))
        with self.assertRaises(ValueError): bound_json(spec, tmp_path)
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        with self.assertRaises(ValueError): checked_path(alias / "new", tmp_path)
        with self.assertRaises(ValueError): checked_path(tmp_path / ".." / "escape", tmp_path)
        with self.assertRaises(ValueError): checked_path(Path("/gpfs/home/zixianma/forbidden"), Path("/"))


if __name__ == "__main__":
    unittest.main()
