# Resuming the reference RL baseline

Use the existing allocation that the user has explicitly authorized. From the repository:

```bash
python3 scripts/resume_baseline.py --job-id JOB_ID --dry-run
python3 scripts/resume_baseline.py --job-id JOB_ID --launch
```

`--dry-run` is the default. It reads Slurm state, recipe hashes, checkpoint metadata and saved-batch metadata; it does not run models or allocate GPUs. The launch command stays in the foreground. An agent can run it with a persistent exec session; from a terminal, use `nohup` with output redirected to scrubbed storage if it must survive disconnects.

The script **never submits or extends an allocation**. It requires a running, user-owned single-node allocation with at least two H200s, eight CPUs and 240 GiB RAM, and ten usable minutes after the shutdown margin. It uses exactly two GPUs through `srun --jobid=... --overlap --exact`. It refuses launch if any non-interactive/non-extern Slurm steps are already present, and uses a per-job lock to prevent concurrent invocations. Inspect existing steps; do not stop unrelated work to bypass this check.

## State and source

The default persistent pointer is:

`/gpfs/scrubbed/zixianma/openwebrl-runtime/current_baseline.json`

It records the current run directory, checkpoint ancestry, preserved reference source, W&B identity and pending recovery batch. `--state PATH` selects a different recorded lineage. `--source PATH` selects another explicitly prepared reference snapshot; recipe hashes and required resume support are checked. The working repository's experimental recipe files are not copied into the run.

For the current baseline, W&B is [`qcq7i4ug`](https://wandb.ai/zixianma/openwebrl/runs/qcq7i4ug), and the preserved source is `reference-stage1-283214` under the runtime root. The credentials come from the repository `.env` and are never included in the preflight plan. `--wandb-run-id qcq7i4ug` can assert the expected identity; it cannot silently change this lineage's W&B ID.

## Checkpoint and batch selection

The script follows `launch_manifest.json` ancestry, selects the highest completed checkpoint marker, and verifies saved iteration, Adam counters, known scheduler offset, shard extents and matching dataset cursor. A partial checkpoint directory without a completion marker is ignored. If the newest marked checkpoint fails verification, resume stops for investigation; it does not silently roll training back. This CPU validation does not substitute for the actual model/optimizer restore on GPUs.

If a complete saved rollout immediately follows that checkpoint, it is replayed before collecting fresh browser trajectories. Replay requires either matching `.provenance.json` or a unique completed `[GenerateProgress]` line. The restored dataset cursor advances by **completed + pending submitted prompt groups**, not just the 48 accepted groups. Already-trained batches are skipped. An incomplete or ambiguous recovery file stops preflight for inspection. Zip-directory checks detect interrupted saves but do not read every tensor byte; the trainer's load is the full payload check.

Example: checkpoint 5 contains 90 Adam updates; saved batch `6.pt` has 1,946 turns and 144 submitted groups. Replay performs `floor(1946 / 256) × 2 = 14` updates, reaching 104 at checkpoint 6. This remains reward observation 7, not a new collection. W&B's historical `train/step` labels are not reliable cumulative Adam counts when batch lengths change.

## Monitoring and limits

The launcher retains the reference recipe (48 accepted groups × five trajectories, global batch 256, two PPO epochs), the tested runtime fixes, lossless file-backed image transport and per-iteration checkpointing. Runtime files stay in scrubbed storage or node-local `/tmp`. It stops before the existing allocation ends, with a three-minute margin and a maximum of eight training hours. A collection interrupted by the time boundary may need recollection unless its recovery file finished saving.

The supervisor prints progress and starts a read-only health recorder. Logs:

- `openwebrl-runtime/logs/resume-JOB_ID-TIMESTAMP.log`: launcher output.
- `openwebrl-runtime/logs/resume-JOB_ID-TIMESTAMP.json`: preflight/resume plan.
- Run directory `training.log`, `progress.log`, `health.jsonl`: detailed training, phase progress and GPU/cgroup memory samples.
- Run directory `resume_plan.json` and `resume_checkpoint_validation.json`: startup provenance.

The pointer is updated when the new launch manifest appears. This means **launched**, not verified healthy. After startup, confirm the actual checkpoint-loaded message, optimizer progress and W&B sync. After each save, validate the new checkpoint before reporting durable progress and refresh the pointer's checkpoint/replay fields. A background recorder cannot diagnose or fix failures: the active agent must check status every 30–60 seconds and investigate errors promptly.

CPU checks: `python3 -m unittest discover -s tests -p test_resume_baseline.py -v`. The real preflight was exercised against job 283214 and correctly selected checkpoint 5, replay batch 6 with 144 submitted groups, and the existing W&B ID. It identified the already-running trainer and monitor; no duplicate training run was launched for this test.
