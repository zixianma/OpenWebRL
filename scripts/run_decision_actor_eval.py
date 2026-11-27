#!/usr/bin/env python3
"""Crossbench CLI with the ``decision:<config.json>`` condition registered (decision actor, closed action set).

Same subcommands, plans, judges and summaries as ``scripts/run_crossbench_selector_eval.py``; the pinned
harness files are untouched. Use this script for every subcommand of a decision condition:

  python scripts/run_decision_actor_eval.py prepare --root R --condition decision:configs/decision_actor/projected_sft.json \
      --benchmarks online_mind2web --n 300 --seed 20261018 --temperature 1.0 --top-p 0.9
  python scripts/run_decision_actor_eval.py launch-script --root R --condition decision:... --benchmarks online_mind2web --time 04:00:00
  python scripts/run_decision_actor_eval.py summarize --dir R
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from openwebrl.decision_actor import crossbench as decision  # noqa: E402

decision.install()
_spec = importlib.util.spec_from_file_location("crossbench_cli", REPO / "scripts/run_crossbench_selector_eval.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)
cli.HOURS_PER_BENCHMARK.setdefault("decision", 2.5)

_default_time, _sbatch_text, _source_command = cli.default_time, cli.sbatch_text, cli.source_command


def default_time(condition_name, benchmarks):
    return _default_time("decision" if condition_name.startswith("decision-") else condition_name, benchmarks)


def sbatch_text(*args, **kwargs):
    # The job's controller must be this wrapper so the decision condition stays registered.
    return _sbatch_text(*args, **kwargs).replace(str(REPO / "scripts/run_crossbench_selector_eval.py"), str(Path(__file__).resolve()))


def source_command(argv):
    # Workers are spawned as `python -m openwebrl.crossbench_selector_eval`; route them through the patched module.
    argv = ["openwebrl.decision_actor.crossbench" if a == "openwebrl.crossbench_selector_eval" else a for a in argv]
    return _source_command(argv)


cli.default_time, cli.sbatch_text, cli.source_command = default_time, sbatch_text, source_command

if __name__ == "__main__":
    cli.main()
