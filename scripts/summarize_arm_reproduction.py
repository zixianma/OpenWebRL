#!/usr/bin/env python3
"""Report complete denominators and paired task differences for ARM reproduction."""
import argparse
import json
import math
import random
from pathlib import Path


def paired_report(baseline, treatment, seed=42, draws=5000):
    common = sorted(set(baseline) & set(treatment))
    common = [t for t in common if baseline[t].get("valid") and treatment[t].get("valid")]
    if not common:
        return {"common_valid_tasks": 0}
    differences = [int(treatment[t]["reward"] == 1) - int(baseline[t]["reward"] == 1) for t in common]
    wins, losses = differences.count(1), differences.count(-1)
    discordant = wins + losses
    pvalue = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
                 / 2**discordant) if discordant else 1.
    rng = random.Random(seed)
    estimates = sorted(sum(rng.choices(differences, k=len(common))) / len(common) for _ in range(draws))
    return {
        "common_valid_tasks": len(common),
        "baseline_success_rate": sum(baseline[t]["reward"] == 1 for t in common) / len(common),
        "treatment_success_rate": sum(treatment[t]["reward"] == 1 for t in common) / len(common),
        "difference_percentage_points": 100 * sum(differences) / len(common),
        "paired_task_bootstrap_95_percent_interval_pp": [100 * estimates[int(draws * .025)],
                                                        100 * estimates[int(draws * .975)]],
        "treatment_only_wins": wins, "baseline_only_wins": losses,
        "mcnemar_exact_two_sided_p": pvalue,
        "limitations": "Conditional on common valid tasks; one run does not measure seed or date variability.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_root")
    args = ap.parse_args()
    root = Path(args.output_root)
    manifests, summaries, records = {}, {}, {}
    for mode in ("baseline", "scalar", "selection"):
        manifests[mode] = json.loads((root / mode / "manifest.json").read_text())
        summaries[mode] = json.loads((root / mode / "summary.json").read_text())
        rows = [json.loads(p.read_text()) for p in (root / mode / "results").glob("*.json")]
        records[mode] = {r["task_id"]: r for r in rows}
        if len(records[mode]) != len(rows):
            raise ValueError("Duplicate task result")
    for mode in ("scalar", "selection"):
        for key in ("actor", "task_file_sha256", "task_ids", "seed", "sampling", "max_steps", "judge", "judge_protocol"):
            if manifests[mode][key] != manifests["baseline"][key]:
                raise ValueError(f"Unmatched comparison field: {key}")
    report = {"summaries": summaries,
              "paired": {mode: paired_report(records["baseline"], records[mode])
                         for mode in ("scalar", "selection")},
              "reported_reference_rates": {"baseline": .338, "scalar": .463, "selection": .511},
              "reference_limitation": "README task denominators and all historical settings are not fully recovered."}
    text = json.dumps(report, indent=2)
    (root / "comparison.json").write_text(text + "\n")
    print(text, flush=True)
    # The allocation launcher calls this while its actor remains resident, after
    # all three evaluators and ARM servers have finished. An explicit queued
    # follow-up can therefore reuse the actor without touching original results.
    c2_queue = root / "queued-c2.json"
    if c2_queue.exists() and json.loads(c2_queue.read_text()).get("authorized") is True:
        import subprocess
        import sys
        config = json.loads(c2_queue.read_text())
        if Path(config["source_full"]).resolve() != root.resolve():
            raise ValueError("C2 queue belongs to a different source report")
        retry_queue = root / "queued-retry.json"
        if retry_queue.exists() and json.loads(retry_queue.read_text()).get("authorized") is True:
            raise ValueError("C2 and retry continuations cannot both be enabled")
        target = Path(config["output"]).resolve()
        if target == root.resolve() or root.resolve().parent in target.parents or target in root.resolve().parents:
            raise ValueError("C2 output must be separate from the original run")
        target.mkdir(parents=True, exist_ok=True)
        with (target / "controller.log").open("a") as log:
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("run_arm_c2.py")),
                                     "--queue", str(c2_queue.resolve())], stdout=log, stderr=subprocess.STDOUT)
        print(f"C2 continuation exited {result.returncode}; see {target / 'status.json'}", flush=True)
        return
    queue = root / "queued-retry.json"
    if queue.exists():
        import subprocess
        import sys
        config = json.loads(queue.read_text())
        if config.get("authorized") is not True:
            print("Retry held pending a cohort decision; original report is complete.", flush=True)
            return
        target = Path(config["output"]).resolve()
        from scripts.run_arm_retry import validate_layout
        validate_layout(config)
        if Path(config["source_full"]).resolve() != root.resolve():
            raise ValueError("Retry queue belongs to a different source report")
        target.mkdir(parents=True, exist_ok=True)
        with (target / "retry-controller.log").open("a") as log:
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("run_arm_retry.py")),
                                     "--queue", str(queue.resolve())], stdout=log, stderr=subprocess.STDOUT)
        # A retry failure is recorded separately; it does not invalidate the
        # completed original benchmark or erase its comparison.
        print(f"Queued retry exited {result.returncode}; see {target / 'retry-status.json'}", flush=True)


if __name__ == "__main__":
    main()
