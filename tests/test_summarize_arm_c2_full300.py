import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summarize_arm_c2_full300 import mcnemar_exact, paired_report, rate_report, wilson


def test_rate_report_keeps_scheduled_tasks_in_overall_denominator():
    records = {
        "a": {"valid": True, "reward": 1},
        "b": {"valid": True, "reward": 0},
        "c": {"valid": False, "reward": 0},
    }
    report = rate_report(records, ["a", "b", "c", "missing"])
    assert report["attempted"] == 3
    assert report["valid"] == 2
    assert report["unavailable"] == 1
    assert report["missing"] == 1
    assert report["overall"] == 0.25
    assert report["valid_only"] == 0.5
    assert wilson(1, 4) == report["overall_wilson_95"]


def test_paired_report_uses_only_common_valid_tasks():
    left = {
        "a": {"valid": True, "reward": 0},
        "b": {"valid": True, "reward": 1},
        "c": {"valid": False, "reward": 0},
        "d": {"valid": True, "reward": 1},
    }
    right = {
        "a": {"valid": True, "reward": 1},
        "b": {"valid": True, "reward": 0},
        "c": {"valid": True, "reward": 1},
        "d": {"valid": True, "reward": 1},
    }
    report = paired_report(left, right, ["a", "b", "c", "d"])
    assert report == {
        "common_valid": 3,
        "right_wins": 1,
        "right_losses": 1,
        "net_wins": 0,
        "mcnemar_exact_two_sided_p": 1.0,
    }
    assert mcnemar_exact(0, 0) == 1.0
