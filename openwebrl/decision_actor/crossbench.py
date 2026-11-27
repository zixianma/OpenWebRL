"""Register ``decision:<config.json>`` as a crossbench condition without editing the digest-pinned harness files.

The canonical harness (``openwebrl/crossbench_selector_eval.py`` + ``scripts/run_crossbench_selector_eval.py``)
pins sha256 digests of its sources in every plan, so this module patches it at import time instead:

* ``parse_condition`` accepts ``decision:<path>``: kind ``decision_actor``, one actor sample's worth of serving,
  slug ``decision-<config sha12>``; the plan freezes the config and this package's source digests.
* ``make_selector`` builds :class:`DecisionActorPolicy` for that kind (other kinds unchanged).
* The CLI's worker launch and sbatch text point at this module / ``scripts/run_decision_actor_eval.py``.

Run as ``python -m openwebrl.decision_actor.crossbench --config worker.json`` (what the job spawns).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from openwebrl import crossbench_selector_eval as core

PACKAGE = Path(__file__).resolve().parent
SOURCES = ("options.py", "projection.py", "deciders.py", "typing_model.py", "policy.py", "crossbench.py")
_original_parse = core.parse_condition
_original_make_selector = core.make_selector


def package_digests() -> dict:
    return {name: hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() for name in SOURCES}


def parse_condition(condition):
    m = re.fullmatch(r"decision:(.+)", condition)
    if not m:
        return _original_parse(condition)
    path = Path(m[1]).resolve()
    if not path.is_file():
        raise ValueError(f"Decision config not found: {path}")
    config = json.loads(path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = "decision-" + config.get("name", "actor")
    return dict(name=name, kind="decision_actor", candidates=1, slug=f"{name}-{digest[:12]}", head_path=None,
                head_sha256=None, selection_rule="closed_action_set_argmax", decision_config=str(path),
                decision_config_sha256=digest, decision_sources=package_digests())


def make_selector(plan, config, directory):
    cond = plan["condition"]
    if cond["kind"] != "decision_actor":
        return _original_make_selector(plan, config, directory)
    from openwebrl.decision_actor.policy import DecisionActorPolicy
    path = Path(cond["decision_config"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != cond["decision_config_sha256"]:
        raise ValueError("Decision config changed since the plan was frozen")
    drift = {k: v for k, v in package_digests().items() if cond["decision_sources"].get(k) != v}
    if drift and not config.get("allow_source_drift"):
        raise ValueError(f"decision_actor sources changed since the plan was frozen: {sorted(drift)}")
    spec = json.loads(path.read_text())
    spec.setdefault("actor_tokenizer", plan["actor"])
    client = None
    if spec["decider"]["type"] == "systemone":
        import httpx
        client = httpx.AsyncClient(timeout=180, trust_env=False)
    return DecisionActorPolicy(spec, Path(directory) / "selections", client=client)


_original_protocol_for = core.protocol_for


def protocol_for(benchmark, condition, *args, **kwargs):
    cond = parse_condition(condition) if isinstance(condition, str) else condition
    if cond["kind"] != "decision_actor":
        return _original_protocol_for(benchmark, cond, *args, **kwargs)
    # Build the shared protocol as a one-sample condition, then record what actually differs.
    protocol = _original_protocol_for(benchmark, dict(cond, kind="baseline"), *args, **kwargs)
    protocol.update(selector_kind="decision_actor", selection_rule=cond["selection_rule"],
                    selector_serving="decision backend from the frozen config; the served actor is used only "
                                     "as the typing model (and as the proposer for the projected_actor diagnostic)",
                    decision_config_sha256=cond["decision_config_sha256"], decision_sources=cond["decision_sources"])
    return protocol


def install():
    core.parse_condition = parse_condition
    core.make_selector = make_selector
    core.protocol_for = protocol_for
    # run_collection binds make_selector as a keyword-only default at definition time; rebind it.
    core.run_collection.__kwdefaults__["selector_factory"] = make_selector


def main():
    install()
    core.main()


if __name__ == "__main__":
    main()
