#!/usr/bin/env python3
"""Freeze a paired, outcome-blind original/shared-reasoning replay pilot.

Preparation performs no browser, GPU, API, or scheduler work. New paid work is
blocked by an unapproved allocation ledger until an exact budget is approved.
"""

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


RUNTIME = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime")
CAMPAIGN = (RUNTIME / "arm-turn-bonus-preparation/branch-selector-scaleup-20261008/"
            "parallel-fresh64-preparation-20261010/piotr-matched-parallel10-v1/shards")
PRIOR = RUNTIME / "selection-action-chunk-outcome-20261017-v2/panels.jsonl"
SOURCE = (RUNTIME / "arm-turn-bonus-preparation/branch-selector-scaleup-20261008/"
          "pilot32-v1/source")
PROVEN_CONTROLLER = (RUNTIME / "arm-turn-bonus-preparation/branch-selector-scaleup-20261008/"
                     "parallel-fresh64-preparation-20261010/piotr-matched-parallel10-v1/"
                     "shards/shard0/controller")
ROOT = RUNTIME / "selection-shared-prefix-replay-20261018"
REPO = Path(__file__).resolve().parents[1]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare(root):
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError("A frozen replay preparation is never overwritten")
    root.mkdir(parents=True)
    source_dir = root / "source"
    source_dir.mkdir()
    shutil.copyfile(REPO / "scripts/score_shared_prefix_features.py",
                    source_dir / "score_shared_prefix_features.py")
    subprocess.run([sys.executable, str(source_dir / "score_shared_prefix_features.py"),
                    "--root", str(root), "--fit"], check=True, stdout=subprocess.DEVNULL)
    prep = load_module(REPO / "scripts/prepare_action_chunk_outcome_pilot.py", "chunk_prep")
    sys.path.insert(0, str(SOURCE))
    from openwebrl.arm_branch_protocol import action_key

    prior_ids, prior_groups = set(), set()
    with PRIOR.open() as stream:
        for line in stream:
            row = json.loads(line)
            prior_ids.add(row["state_id"])
            prior_groups.add(row["task_group"])
    entries = []
    for shard in sorted(CAMPAIGN.glob("shard*")):
        for state in read(shard / "accepted-states.json")["state_ids"]:
            entries.append(dict(root=str(shard), state=state))
    cells = prep.outcomes(entries)
    counts = Counter()
    admitted = []
    tasks = {}
    source_configs = {}
    for entry in entries:
        counts["accepted_source_states"] += 1
        state = entry["state"]
        if state in prior_ids:
            counts["prior_pilot_state"] += 1
            continue
        if any((candidate, repeat) not in cells[state]
               for candidate in range(5) for repeat in range(8)):
            counts["incomplete_original_panel"] += 1
            continue
        shard = Path(entry["root"])
        reference = read(shard / "states" / state / "anchor-reference.json")
        if prep.digest(reference["path"]) != reference["sha256"]:
            raise ValueError("Source anchor changed: " + state)
        original = read(reference["path"])
        try:
            shared, actions = prep.candidates(original)
        except ValueError:
            counts["unparseable_action"] += 1
            continue
        if len({action["name"] for action in actions}) < 2:
            counts["one_tool_name"] += 1
            continue
        group = prep.task_group(original["task"])
        if group in prior_groups:
            counts["prior_pilot_task_group"] += 1
            continue
        try:
            prompt, images = prep.repeat_zero_prefixes(entry, original)
        except ValueError:
            counts["exact_prompt_unavailable"] += 1
            continue
        common = deepcopy(original)
        for candidate in common["candidates"]:
            output = list(candidate["output"])
            if output[0].count("</think>") != 1:
                raise ValueError("Reasoning boundary is not unique")
            output[0] = shared + output[0].split("</think>", 1)[1]
            if action_key(output) != candidate["action_key"]:
                raise ValueError("Common-prefix substitution changed executable action")
            candidate["output"] = output
        task_id = original["task_id"]
        plan = read(shard / "plan.json")
        source_configs[str(shard)] = dict(plan_sha256=prep.digest(shard / "plan.json"),
                                          actor_revision=plan["actor_revision"])
        task_rows = [json.loads(line) for line in Path(plan["task_file"]).read_text().splitlines()
                     if line.strip()]
        matching = [row for row in task_rows if row["metadata"]["task_id"] == task_id]
        if len(matching) != 1:
            raise ValueError("Source task is not unique")
        if task_id in tasks and tasks[task_id] != matching[0]:
            raise ValueError("A task differs across source shards")
        tasks[task_id] = matching[0]
        admitted.append(dict(state=state, task_id=task_id, task_group=group,
                             shard=str(shard), original=original, common=common,
                             prompt=prompt, images=images, shared=shared,
                             actions=actions,
                             outcomes=[[cells[state][i, r] for r in range(8)] for i in range(5)]))
        counts["eligible_exact_states"] += 1
    # All eligible states are retained. Order and paired condition order are
    # fixed without looking at outcomes or pilot picks.
    admitted.sort(key=lambda item: item["state"])
    if len(admitted) != 8 or len({row["task_group"] for row in admitted}) != 5:
        raise ValueError("The prespecified 8-state/5-task eligibility snapshot changed")
    data = root / "data"
    data.mkdir()
    with (data / "tasks.jsonl").open("x") as stream:
        for task_id in sorted(tasks):
            stream.write(json.dumps(tasks[task_id], ensure_ascii=False) + "\n")
    panels = []
    states, schedule, frozen = [], [], []
    for item in admitted:
        state = item["state"]
        original_path = data / "anchors" / state / "original.json"
        shared_path = data / "anchors" / state / "shared.json"
        write(original_path, item["original"])
        write(shared_path, item["common"])
        anchor_hashes = dict(original=prep.digest(original_path), shared=prep.digest(shared_path))
        first = ("original", "shared") if int(state[:2], 16) % 2 == 0 else ("shared", "original")
        for condition in first:
            panel_id = state + "-" + ("o" if condition == "original" else "s")
            anchor_path = original_path if condition == "original" else shared_path
            panel_dir = root / "states" / panel_id
            write(panel_dir / "anchor-reference.json",
                  dict(path=str(anchor_path), sha256=anchor_hashes[condition]))
            states.append(dict(id=panel_id, source_state=state, condition=condition,
                               task_id=item["task_id"], task_group=item["task_group"],
                               depth=item["original"]["depth"], anchor_sha256=anchor_hashes[condition]))
            for candidate in range(5):
                for repeat in range(8):
                    schedule.append(dict(item_id=f"{panel_id}-c{candidate}-r{repeat}",
                                         state=panel_id, mode="branch", condition="L01",
                                         category="primary", task_id=item["task_id"],
                                         depth=item["original"]["depth"],
                                         candidate=candidate, repeat=repeat))
        panels.append(dict(state_id=state, task_id=item["task_id"],
                           task_group=item["task_group"], fold=0,
                           prompt_text=item["prompt"], images=item["images"],
                           shared_reasoning=item["shared"], actions=item["actions"],
                           outcomes=item["outcomes"],
                           prompt_sha256=__import__("hashlib").sha256(item["prompt"].encode()).hexdigest()))
        frozen.append(dict(state=state, task_group=item["task_group"],
                           source_shard=item["shard"], source_anchor_sha256=prep.digest(
                               Path(item["shard"]) / "states" / state / "anchor-reference.json"),
                           original_anchor_sha256=anchor_hashes["original"],
                           shared_anchor_sha256=anchor_hashes["shared"],
                           first_condition=first[0], distinct_tool_names=len({a["name"] for a in item["actions"]})))
    with (data / "panels.jsonl").open("x") as stream:
        for row in panels:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    exemplar = read(Path(admitted[0]["shard"]) / "plan.json")
    worker = deepcopy(exemplar["worker_config"])
    worker["tasks_path"] = str(data / "tasks.jsonl")
    worker["browser_env_config"]["path_to_task_file"] = str(data / "tasks.jsonl")
    for item in admitted[1:]:
        other = read(Path(item["shard"]) / "plan.json")
        for key in ("actor_revision", "actor_files"):
            if other[key] != exemplar[key]:
                raise ValueError("Selected shards disagree on frozen actor")
        for key in ("prompt_sha256", "tool_list_sha256", "canonical_reward_sha256"):
            if other["worker_config"][key] != exemplar["worker_config"][key]:
                raise ValueError("Selected shards disagree on worker recipe")
    manifest = dict(status="prepared_unapproved", protocol="paired-original-shared-reasoning-v1",
                    selection="All exact-replayable Piotr-matched source states with >=2 first tool names, a complete original 5x8 panel, unseen state and task group relative to the 280-state head pilot; no outcome-dependent screening",
                    cohort_states=8, task_groups=5, condition_panels=16,
                    branch_records_planned=640, new_unique_collection_states=0,
                    counts=dict(counts), source_configs=source_configs, states=frozen,
                    original_pilot_panels_sha256=prep.digest(PRIOR),
                    private_panels_sha256=prep.digest(data / "panels.jsonl"),
                    exact_action_identity=True, exact_pre_action_fingerprint=True,
                    interpretation="Original and shared reasoning arms replay the same executable action on adjacent, independently reconstructed live browser contexts; outcome comparison is a small mechanism diagnostic, not a population estimate")
    write(data / "cohort.json", manifest)

    controller = root / "source" / "run_shared_prefix_replay.py"
    shutil.copyfile(REPO / "scripts/run_shared_prefix_replay.py", controller)
    shutil.copyfile(REPO / "scripts/extract_shared_prefix_features.py",
                    controller.parent / "extract_shared_prefix_features.py")
    shutil.copyfile(REPO / "scripts/analyze_shared_prefix_replay.py",
                    controller.parent / "analyze_shared_prefix_replay.py")
    for name in ("owned_cleanup.py", "actor_entrypoint.py", "verify_cuda_threadname.py",
                 "adapter-manifest.json", "cuda_threadname_adapter.c",
                 "libcuda_threadname_adapter.so"):
        shutil.copyfile(PROVEN_CONTROLLER / name, controller.parent / name)
    resources = dict(gpus=1, gpu_type="H200", cpus=8, memory_gib=240,
                     total_seconds=28800, sft_replicas=1, collectors=40)
    limits = dict(browser_sessions=900, browser_sessions_per_condition=900,
                  concurrent_browsers=40, local_sft_generations=30000,
                  canonical_judge_http_attempts=1200, canonical_judge_usd=12.0,
                  luna_selector_requests=0, luna_usd=0.0,
                  jev_selector_requests=0, local_kev_requests=0)
    artifacts = {str(path): prep.digest(path) for path in
                 [data / "tasks.jsonl", data / "panels.jsonl", data / "cohort.json",
                  root / "model/value-head.npz", root / "model/head-manifest.json",
                  SOURCE / "reference_manifest.json",
                  RUNTIME / "selection-action-chunk-outcome-20261017-v2/source/run_action_chunk_outcome_pilot.py"] +
                 sorted(controller.parent.iterdir()) +
                 sorted((data / "anchors").glob("*/*.json")) +
                 sorted((root / "states").glob("*/anchor-reference.json"))}
    plan = dict(root=str(root), source=str(SOURCE), protocol_name=manifest["protocol"],
                states_target=16, states=states, schedule=schedule,
                conditions={"L01": {"actor": "official OpenWebRL-4B-SFT"}},
                candidate_count=5, candidate_draws=5, continuation_repetitions=8,
                retain_duplicate_candidates=True, resources=resources, limits=limits,
                task_file=str(data / "tasks.jsonl"), worker_config=worker,
                actor_files=exemplar["actor_files"], actor_revision=exemplar["actor_revision"],
                controller_sha256=prep.digest(controller), artifacts=artifacts,
                cohort_manifest=str(data / "cohort.json"),
                feature_panels=str(data / "panels.jsonl"),
                original_pilot_root=str(PRIOR.parent),
                comparison="Paired original vs shared reasoning on fixed five actions and eight new draws per arm; task clusters and invalid outcomes retained separately")
    write(root / "plan.json", plan)
    from openwebrl.controlled_sft_state import digest
    write(root / "approval.json", dict(approved=False, plan_sha256=digest(plan),
                                      resources=resources, limits=limits, attempts=[],
                                      reason="Exact new H200/browser/judge budget awaits user approval"))
    batch = f'''#!/usr/bin/env bash
#SBATCH --job-name=arm-shared-prefix
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output={root}/logs/slurm-%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 WANDB_PROJECT=openwebrl-evals
export HF_HOME={root}/cache/huggingface HF_HUB_CACHE={RUNTIME}/hf-cache/hub
export XDG_CACHE_HOME={root}/cache/xdg XDG_CONFIG_HOME={root}/cache/config
export XDG_STATE_HOME={root}/cache/state XDG_DATA_HOME={root}/cache/data
export TORCH_HOME={root}/cache/torch TRITON_CACHE_DIR={root}/cache/triton
export TORCHINDUCTOR_CACHE_DIR={root}/cache/torchinductor
export SGLANG_DG_CACHE_DIR={root}/cache/deep_gemm FLASHINFER_WORKSPACE_BASE={root}/cache/flashinfer
export CUDA_CACHE_PATH={root}/cache/cuda WANDB_DIR={root}/wandb-local
export WANDB_CACHE_DIR={root}/cache/wandb WANDB_CONFIG_DIR={root}/cache/config/wandb
export WANDB_DATA_DIR={root}/cache/wandb-data TMPDIR=/tmp/arm-shared-prefix-$SLURM_JOB_ID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" {root}/logs {root}/cache/huggingface {root}/cache/xdg {root}/cache/config
source {SOURCE}/scripts/h200_env.sh
exec {RUNTIME}/venv/bin/python -B {controller.parent}/verify_cuda_threadname.py \
  --manifest {controller.parent}/adapter-manifest.json --exec \
  {RUNTIME}/venv/bin/python -B {controller} --root {root} --execute "$SLURM_JOB_ID"
'''
    (root / "launch.sbatch").write_text(batch)
    return dict(root=str(root), states=8, task_groups=5, branch_records=640,
                resources=resources, limits=limits, approved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root), indent=2))


if __name__ == "__main__":
    main()
