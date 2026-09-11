#!/usr/bin/env python3
"""Run an explicitly queued, separate matched retry using the resident eval actor."""
import argparse
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
try:
    from project_docs import rewrite_links, write_document_section
except ModuleNotFoundError:  # Imported as scripts.run_arm_retry in tests/tools.
    from scripts.project_docs import rewrite_links, write_document_section

REPO = Path(__file__).resolve().parents[1]
MODES = ("baseline", "scalar")
MATCH_FIELDS = ("actor", "task_file", "task_file_sha256", "task_ids", "seed", "sampling",
                "max_steps", "history", "context_num_screenshots", "browser_format", "judge",
                "judge_protocol", "task_timeout")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_rows(directory):
    files = sorted((Path(directory) / "results").glob("*.json"))
    rows = [json.loads(p.read_text()) for p in files]
    result = {r["task_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate task outcomes")
    return result


def validate_layout(queue):
    source, output = Path(queue["source_full"]).resolve(), Path(queue["output"]).resolve()
    if output == source or output in source.parents or source.parent in output.parents:
        raise ValueError("Retry output must be separate from the original run")
    if queue.get("authorized") is not True or queue.get("modes") != list(MODES):
        raise ValueError("Require explicit baseline/scalar retry authorization")
    if queue.get("attempts_per_arm_per_task") != 1:
        raise ValueError("Only one fixed attempt per arm and task is supported")
    return source, output


def validate_source(queue, require_selection=True):
    source, output = validate_layout(queue)
    manifests, records = {}, {}
    modes = (*MODES, "selection") if require_selection else MODES
    for mode in modes:
        directory = source / mode
        manifest = json.loads((directory / "manifest.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        rows = load_rows(directory)
        expected = queue["expected_source_tasks"]
        if (len(manifest["task_ids"]) != expected or len(set(manifest["task_ids"])) != expected
                or set(rows) != set(manifest["task_ids"])
                or summary["scheduled"] != expected or summary["attempted"] != expected):
            raise ValueError(f"{mode}: source evaluation is incomplete")
        manifests[mode], records[mode] = manifest, rows
    for mode in modes[1:]:
        for field in MATCH_FIELDS:
            if manifests[mode][field] != manifests["baseline"][field]:
                raise ValueError(f"Original arms have different {field}")
    for mode, hashes in queue["original_result_sha256"].items():
        actual = {p.name: digest(p) for p in (source / mode / "results").glob("*.json")}
        if actual != hashes:
            raise ValueError(f"Original {mode} results changed since retry authorization")
    invalid = {task for mode in MODES for task, row in records[mode].items() if not row.get("valid")}
    task_ids = queue["task_ids"]
    if len(task_ids) != queue["expected_retry_tasks"] or len(set(task_ids)) != len(task_ids) or set(task_ids) != invalid:
        raise ValueError("Retry cohort must equal the frozen baseline/scalar unavailable-task union")
    source_ids = manifests["baseline"]["task_ids"]
    indices = [i for i, task in enumerate(source_ids) if task in invalid]
    if indices != queue["task_indices"] or [source_ids[i] for i in indices] != task_ids:
        raise ValueError("Frozen retry task IDs and source indices do not agree")
    if digest(manifests["baseline"]["task_file"]) != manifests["baseline"]["task_file_sha256"]:
        raise ValueError("Task dataset changed")
    return source, output, manifests, records


def command(argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=20).strip()


def process_identity(pid):
    root = Path("/proc") / str(pid)
    return {"pid": int(pid), "start_ticks": (root / "stat").read_text().rsplit(")", 1)[1].split()[19],
            "cgroup": (root / "cgroup").read_text().strip()}


def validate_allocation(queue):
    job = str(queue["allocation"])
    if os.environ.get("SLURM_JOB_ID") != job or os.environ.get("SLURM_STEP_ID") != queue["slurm_step"]:
        raise ValueError("Retry must inherit the authorized evaluation Slurm step")
    if f"/job_{job}/" not in Path("/proc/self/cgroup").read_text():
        raise ValueError("Retry process belongs to another allocation")
    info = command(["scontrol", "show", "job", job, "-o"])
    if f"UserId={getpass.getuser()}(" not in info or "JobState=RUNNING" not in info:
        raise ValueError("Authorized allocation is not running")
    devices = command(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]).splitlines()
    if devices != [queue["gpu_uuid"]]:
        raise ValueError("Assigned GPU does not match the queued GPU")
    pids = command(["nvidia-smi", "--id=" + queue["gpu_uuid"], "--query-compute-apps=pid",
                    "--format=csv,noheader"]).splitlines()
    expected = {int(p["pid"]): p for p in queue["resident_actor_processes"]}
    if {int(pid) for pid in pids} != set(expected):
        raise ValueError("GPU processes differ from the resident actor; leave other workloads untouched")
    for pid, identity in expected.items():
        if process_identity(pid) != identity:
            raise ValueError("Resident actor process identity changed")


def get_health(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as response:
        content = response.read()
    return json.loads(content) if content else {}


def stop_owned(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_eval(argv, logfile):
    with logfile.open("x") as stream:
        child = subprocess.Popen(argv, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            code = child.wait()
            if code:
                raise RuntimeError(f"Retry evaluator exited {code}; see {logfile}")
        finally:
            stop_owned(child)


def build_eval_command(queue, mode, manifest):
    sampling = manifest["sampling"]
    return [sys.executable, "-m", "openwebrl.arm_eval", "--mode", mode,
            "--output", str(Path(queue["output"]) / mode), "--actor", manifest["actor"],
            "--task-file", manifest["task_file"], "--task-indices", ",".join(map(str, queue["task_indices"])),
            "--parallel", str(queue["parallel_by_mode"][mode]), "--seed", str(manifest["seed"]),
            "--temperature", str(sampling["temperature"]), "--top-p", str(sampling["top_p"]),
            "--max-new-tokens", str(sampling["max_new_tokens"]), "--max-steps", str(manifest["max_steps"]),
            "--task-timeout", str(manifest["task_timeout"]), "--judge-model", manifest["judge"], "--env-file", ".env"]


def summarize_retry(queue, original):
    from openwebrl.arm_eval import summarize
    from scripts.summarize_arm_reproduction import paired_report
    output = Path(queue["output"])
    rows = {mode: load_rows(output / mode) for mode in MODES}
    for mode in MODES:
        if set(rows[mode]) != set(queue["task_ids"]):
            raise ValueError("Retry result cohort differs from the authorized cohort")
    summaries = {mode: summarize(list(rows[mode].values()), len(queue["task_ids"])) for mode in MODES}
    recovery = {}
    for mode in MODES:
        unavailable = [task for task in queue["task_ids"] if not original[mode][task].get("valid")]
        recovery[mode] = {"initially_unavailable": len(unavailable),
                          "now_valid": sum(bool(rows[mode][task].get("valid")) for task in unavailable),
                          "now_successful": sum(rows[mode][task].get("valid") and rows[mode][task].get("reward") == 1 for task in unavailable),
                          "previously_valid_now_unavailable": sum(original[mode][task].get("valid") and not rows[mode][task].get("valid") for task in queue["task_ids"])}
    report = {"completed_utc": datetime.now(timezone.utc).isoformat(), "scope": "Separate fixed matched retry cohort; original results unchanged",
              "summaries": summaries, "paired": paired_report(rows["baseline"], rows["scalar"]), "recovery": recovery,
              "source_full": queue["source_full"], "task_ids": queue["task_ids"], "attempts_per_arm_per_task": 1}
    atomic_json(output / "comparison.json", report)
    lines = ["# ARM matched retry results", "", "Completed: " + report["completed_utc"] + ".", "",
             f"One fresh attempt per model on the same {len(queue['task_ids'])}-task cohort selected from initial unavailable outcomes. "
             "These are follow-up rates on a selected cohort, not a replacement for the original 300-task benchmark.", "",
             f"| Arm | Successes | Valid | Unavailable | Overall (all {len(queue['task_ids'])}) | Valid-only |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for mode, s in summaries.items():
        valid_rate = f"{100*s['success_rate_valid']:.1f}% ({s['successes']}/{s['valid']})" if s["valid"] else "Unavailable (0 valid)"
        lines.append(f"| {mode} | {s['successes']} | {s['valid']} | {s['unavailable']} | "
                     f"{100*s['success_rate_all_scheduled']:.1f}% ({s['successes']}/{s['scheduled']}) | {valid_rate} |")
    lines += ["", "Original benchmark results remain unchanged:", ""]
    for mode in MODES:
        initial = summarize(list(original[mode].values()), len(original[mode]))
        valid_rate = f"{100*initial['success_rate_valid']:.1f}%" if initial["valid"] else "unavailable"
        lines.append(f"- {mode}: {initial['successes']}/{initial['scheduled']} "
                     f"({100*initial['success_rate_all_scheduled']:.1f}%) overall; "
                     f"{initial['successes']}/{initial['valid']} ({valid_rate}) valid-only.")
    lines += ["", "## Recovery of initially unavailable evaluations", ""]
    for mode, r in recovery.items():
        lines.append(f"- {mode}: {r['now_valid']}/{r['initially_unavailable']} now valid; "
                     f"{r['now_successful']} now successful; {r['previously_valid_now_unavailable']} originally valid counterparts now unavailable.")
    lines += ["", "## Paired statistics", "", "```json", json.dumps(report["paired"], indent=2), "```", "",
              "Models, sampling seeds, context/horizon, and timeouts match the original protocol. "
              "The browser sessions are fresh. No best-of-retries result selection or full-benchmark replacement was performed.", "",
              f"Artifacts: [{output.name}]({output}).", "",
              "[Original benchmark results](" + str(REPO / "openwebrl/docs/ARM_INFERENCE_RESULTS.md") + ").", ""]
    markdown = rewrite_links("\n".join(lines), "ARM_INFERENCE_RETRY_RESULTS.md")
    (output / "RETRY_RESULTS.md").write_text(markdown)
    report["markdown_report"] = str(output / "RETRY_RESULTS.md")
    document = REPO / "openwebrl/docs/ARM_INFERENCE_RETRY_RESULTS.md"
    try:
        write_document_section(document, markdown)
    except OSError as exc:
        # Scrubbed outputs remain authoritative if project quota prevents the
        # convenience copy. Do not truncate the existing project document.
        report["project_markdown_write_error"] = str(exc)
    atomic_json(output / "comparison.json", report)
    return report


def execute(queue_path):
    queue = json.loads(Path(queue_path).read_text())
    source, output = validate_layout(queue)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output / "retry-status.json"
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        if previous.get("phase") == "complete":
            return
        if previous.get("phase", "queued") != "queued":
            raise ValueError("This fixed retry pass has already started; refusing another attempt")
        def record(phase, **extra):
            value = dict(previous, phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(),
                         allocation=queue["allocation"], output=str(output), **extra)
            atomic_json(state_path, value)
            print(json.dumps(value), flush=True)
        scalar = None
        scalar_log = None
        try:
            source, output, manifests, original = validate_source(queue)
            validate_allocation(queue)
            actor_info = get_health("http://127.0.0.1:19100/get_model_info")
            if Path(actor_info["model_path"]).resolve() != Path(manifests["baseline"]["actor"]).resolve():
                raise ValueError("Resident actor differs from the pinned source actor")
            # Refuse to replace any outcome or append another trial to an existing arm.
            if any((output / mode).exists() for mode in MODES):
                raise ValueError("Retry arm directories already exist")
            snapshot = {str(p.relative_to(source)): digest(p) for mode in (*MODES, "selection")
                        for p in (source / mode / "results").glob("*.json")}
            atomic_json(output / "original-result-hashes.json", snapshot)
            atomic_json(output / "execution-settings.json", {"parallel_by_mode": queue["parallel_by_mode"]})
            record("running_baseline", controller_pid=os.getpid(), actor_reused=True)
            run_eval(build_eval_command(queue, "baseline", manifests["baseline"]), output / "baseline-eval.log")
            scalar_manifest = manifests["scalar"]["selector_health"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
            scalar_log = (output / "scalar-server.log").open("x")
            scalar = subprocess.Popen([sys.executable, str(REPO / "scripts/serve_arm.py"), "--mode", "scalar",
                                       "--model", scalar_manifest["model"], "--base", scalar_manifest["base"]],
                                      cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=scalar_log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 600
            while True:
                if scalar.poll() is not None:
                    raise RuntimeError("Scalar server exited during startup")
                try:
                    health = get_health("http://127.0.0.1:19101/health")
                except Exception:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Scalar startup timed out")
                    time.sleep(1)
                    continue
                if health != scalar_manifest:
                    raise ValueError("Retry scalar health/config differs from the original run")
                break
            record("running_scalar", controller_pid=os.getpid(), actor_reused=True)
            run_eval(build_eval_command(queue, "scalar", manifests["scalar"]), output / "scalar-eval.log")
            after = {str(p.relative_to(source)): digest(p) for mode in (*MODES, "selection")
                     for p in (source / mode / "results").glob("*.json")}
            if after != snapshot:
                raise ValueError("Original task results changed during the retry pass")
            report = summarize_retry(queue, original)
            record("complete", original_result_files_verified=len(snapshot), comparison=str(output / "comparison.json"),
                   markdown_report=report["markdown_report"],
                   project_markdown_write_error=report.get("project_markdown_write_error"))
        except BaseException as exc:
            record("failed", error_type=type(exc).__name__, reason=str(exc))
            raise
        finally:
            stop_owned(scalar)
            if scalar_log:
                scalar_log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    execute(args.queue)


if __name__ == "__main__":
    main()
