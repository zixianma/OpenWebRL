"""Reconstruct campaign task inputs from the benchmark files already in this repo."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = API_ROOT.parents[1]
FILES = {"online-mind2web": "online-mind2web.jsonl", "webvoyager": "webvoyager_fara.jsonl", "deepshop": "deepshop.jsonl"}
DATASET_PINS = {'online-mind2web': {'sha256': '8343c23be98d6d63856e9b53ff3884222be099cc0f55ad5edb475176f54317ed', 'rows': 300}, 'webvoyager': {'sha256': 'ac9253317b76884abbd8d89e9b02570f3f4fbf18e428c335bc8761d778c6fe68', 'rows': 595}, 'deepshop': {'sha256': '479d35f1d2345b66512b9d398a5472f28bcc5e3e08da487073662ab188f8fefa', 'rows': 150}}
ALIASES = {"om2w": "online-mind2web", "online_mind2web": "online-mind2web", "onlinemind2web": "online-mind2web"}


def canonical_benchmark(value):
    value = str(value).lower()
    return ALIASES.get(value, value)


def dataset_rows(benchmark, repository_root=None):
    benchmark = canonical_benchmark(benchmark)
    if benchmark not in FILES:
        raise ValueError("Unknown public benchmark: " + benchmark)
    root = REPO_ROOT if repository_root is None else Path(repository_root)
    path = root / "openwebrl/data/eval" / FILES[benchmark]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATASET_PINS[benchmark]["sha256"]:
        raise ValueError("Public benchmark source changed: " + str(path))
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows) != DATASET_PINS[benchmark]["rows"]:
        raise ValueError("Public benchmark task count changed: " + str(path))
    key = "id" if benchmark == "webvoyager" else "task_id"
    ids = [str(row[key]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate benchmark task IDs: " + str(path))
    return dict(zip(ids, rows))


def normalize(row, benchmark, identity=None):
    benchmark = canonical_benchmark(benchmark)
    native_id = str(row["id"] if benchmark == "webvoyager" else row["task_id"])
    ident = dict(identity or {})
    task_id = str(ident.get("task_id", native_id))
    intent = row["ques"] if benchmark == "webvoyager" else row["task_name"]
    start_url = row["web"] if benchmark == "webvoyager" else row["website"]
    if not isinstance(intent, str) or not isinstance(start_url, str) or not intent or not start_url:
        raise ValueError("Benchmark task is missing its instruction or start URL")
    if benchmark == "online-mind2web":
        return {"task_id": task_id, "intent": intent, "start_url": start_url,
                "benchmark": ident.get("benchmark", benchmark), "difficulty": row.get("difficulty"),
                "host": urlsplit(start_url).hostname, "source_task_id": ident.get("source_task_id", native_id)}
    return {"campaign_task_id": ident.get("campaign_task_id", benchmark + "/" + native_id),
            "benchmark": benchmark, "task_id": task_id, "intent": intent, "start_url": start_url,
            "source_row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
            "native_metadata": row, "native_task_id": native_id}


def load_tasks(campaign_or_benchmark, task_ids=None, repository_root=None):
    """Load the exact exported campaign order, or public benchmark source order.

    Only task identifiers are duplicated in the campaign roster. Task text,
    start URLs and benchmark grading references come from the public source.
    """
    roster_path = API_ROOT / "portable_assets/campaign_specs" / str(campaign_or_benchmark) / "task-roster.json"
    if roster_path.is_file():
        roster = json.loads(roster_path.read_text())
    else:
        benchmark = canonical_benchmark(campaign_or_benchmark)
        roster = [{"task_id": key, "benchmark": benchmark, "native_task_id": key}
                  for key in dataset_rows(benchmark, repository_root)]
    datasets = {}
    tasks = []
    for identity in roster:
        benchmark = canonical_benchmark(identity.get("benchmark", "online-mind2web"))
        if benchmark not in datasets:
            datasets[benchmark] = dataset_rows(benchmark, repository_root)
        source_id = str(identity.get("native_task_id", identity.get("source_task_id", identity["task_id"])))
        if source_id not in datasets[benchmark]:
            raise ValueError("Campaign source task not found in public benchmark: " + benchmark + "/" + source_id)
        tasks.append(normalize(datasets[benchmark][source_id], benchmark, identity))
    ids = [task["task_id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Campaign roster contains duplicate task identities")
    if task_ids is not None:
        lookup = dict(zip(ids, tasks))
        chosen = [str(value) for value in task_ids]
        if len(chosen) != len(set(chosen)) or any(value not in lookup for value in chosen):
            raise ValueError("Requested task IDs must be unique members of the campaign roster")
        tasks = [lookup[value] for value in chosen]
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_or_benchmark")
    parser.add_argument("--output", type=Path, required=True, help="New local task manifest")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(Path("/gpfs/home/zixianma")):
        parser.error("Choose a writable project or temporary output directory")
    rows = load_tasks(args.campaign_or_benchmark)
    if output.exists() or output.is_symlink():
        parser.error("Output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"tasks": len(rows), "output": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
