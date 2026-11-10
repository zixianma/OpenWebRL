#!/usr/bin/env python3
"""Run the offline action-credit acceptance suite and emit a scoped audit.

No allocations, browser sessions, teacher calls, training-controller changes,
or network requests. Upstream files must already exist and match test hashes.
Run with the project's torch-enabled Python and -B.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path)
    parser.add_argument("--require-upstream", action="store_true",
                        help="Fail if any upstream test was skipped")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.require_upstream and args.upstream_dir is None:
        parser.error("--require-upstream requires --upstream-dir")
    if args.upstream_dir is not None:
        if not args.upstream_dir.is_dir():
            parser.error("upstream fixture directory does not exist")
        os.environ["ACTION_CREDIT_UPSTREAM_FIXTURE_DIR"] = str(args.upstream_dir.resolve())
    else:
        os.environ.pop("ACTION_CREDIT_UPSTREAM_FIXTURE_DIR", None)
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_action_credit*.py")
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / "tests"),
                                                     pattern="test_prepare_action_credit*.py"))
    def test_ids(group):
        for item in group:
            if isinstance(item, unittest.TestSuite):
                yield from test_ids(item)
            else:
                yield item.id()
    discovered = tuple(test_ids(suite))
    oracle_tests = [name for name in discovered if name.startswith("test_action_credit_upstream.PinnedUpstreamParity.")]
    if args.require_upstream and len(oracle_tests) < 5:
        parser.error("the required upstream oracle tests were not discovered")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    upstream_passed = args.upstream_dir is not None and len(oracle_tests) >= 5 and not result.skipped and result.wasSuccessful()
    passed = result.wasSuccessful() and not (args.require_upstream and not upstream_passed)
    files = sorted([*ROOT.glob("openwebrl/action_credit*.py"),
                    *ROOT.glob("tests/test_action_credit*.py"),
                    *ROOT.glob("tests/test_prepare_action_credit*.py"),
                    *ROOT.glob("scripts/prepare_action_credit*.py"),
                    ROOT / "scripts/run_action_credit_pilot.py",
                    ROOT / "scripts/check_storage_quota.py",
                    ROOT / "openwebrl/eval_monitor.py", Path(__file__).resolve()])
    oracle = sys.modules.get("test_action_credit_upstream")
    sources = {}
    for name, (repository, commit, source_path, expected_hash) in getattr(oracle, "SOURCES", {}).items():
        fixture = args.upstream_dir / (name + ".py") if args.upstream_dir else None
        sources[name] = dict(repository=repository, commit=commit, path=source_path,
            expected_sha256=expected_hash,
            observed_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest()
                            if fixture is not None and fixture.is_file() else None)
    report = {
        "schema_version": 1,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "status": "cpu_verified_not_launch_ready" if passed else "validation_failed",
        "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "upstream_oracle_passed": upstream_passed,
        "upstream_sources": sources,
        "python_version": sys.version.split()[0],
        "torch_version": getattr(sys.modules.get("torch"), "__version__", None),
        "launch_ready": False,
        "new_compute_allocations": 0, "api_calls": 0, "browser_sessions": 0,
        "files_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "validated_scope": [
            "Five scalar objectives; official Tree/GiGPO source functions when upstream_oracle_passed",
            "Masked clipped-loss values/gradients, shared-prefix terms, explicit reductions",
            "CPU simulated uneven rank/microbatch gradient reduction (not real distributed GPU execution)",
            "Policy/context/horizon/receipt binding and fit/holdout separation",
            "Sealed inventory and full objective/scientific configuration binding",
            "Outcome-blind tree scheduling, complete round barriers and retained leaf selection",
            "Locked crash-safe ledger, immutable planned panels, all-attempt reservation accounting",
            "CPU trainer/collector/manifest integration fixtures and saved-batch recovery",
            "Real CPU subprocess lifecycle checks; these do not certify browser/GPU execution",
            "Held-out300 artifact/owner checks with CPU fixtures and shared all-stage retry accounting",
            "Shared storage reservation race/floor checks with mocked quota and measured-forecast fixtures",
            "Production preparation and purpose-scoped verification accounting checks remain CPU-only",
        ],
        "remaining_gates": [
            "Fresh production source/bootstrap and complete owning iteration0/10 evaluation queue",
            "Live actor weight transfer and actual chain/tree/Q collection with retained judge artifacts",
            "Physical browser replay and parent-response/observation/context transition verification",
            "Actual multimodal token alignment and frozen old-policy scores",
            "TP2/DP4 full-batch/long-context numerical and restore/save/resume acceptance",
            "Physical native-controller transaction recovery and representative full-batch memory/IO",
            "Exact paid-API approval and per-arm resource/storage admission before production dispatch",
        ],
    }
    if args.report is not None:
        path = args.report.absolute()
        resolved = path.resolve(strict=False)
        forbidden = Path("/gpfs/home/zixianma")
        allowed = (ROOT, Path("/gpfs/scrubbed/zixianma/openwebrl-runtime"), Path("/tmp"))
        if (any(p == forbidden or forbidden in p.parents for p in (path, resolved))
                or not any(resolved == root or root in resolved.parents for root in allowed)
                or any(p.is_symlink() for p in (path, *path.parents))):
            parser.error("report must be in an approved project/runtime/tmp path without symlinks")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
        # Existing reports are replaced only by an explicit --report invocation.
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        with temporary.open("x") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
