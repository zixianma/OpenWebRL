# Resuming the reference RL baseline

Use the existing allocation that the user has explicitly authorized. From the repository:

```bash
python3 scripts/resume_baseline.py --job-id JOB_ID --dry-run
python3 scripts/resume_baseline.py --job-id JOB_ID --launch
```

`--dry-run` is the default. It reads Slurm state, recipe hashes, checkpoint metadata and saved-batch metadata; it does not run models or allocate GPUs. The launch command stays in the foreground. An agent can run it with a persistent exec session; from a terminal, use `nohup` with output redirected to scrubbed storage if it must survive disconnects.

The script **never submits or extends an allocation**. It requires a running, user-owned single-node allocation with at least two H200s, eight CPUs and 240 GiB RAM, and ten usable minutes after the shutdown margin. By default it uses two GPUs through `srun --jobid=... --overlap --exact`; the four-GPU profile is described below. It refuses launch if any non-interactive/non-extern Slurm steps are already present, and uses a per-job lock to prevent concurrent invocations. Inspect existing steps; do not stop unrelated work to bypass this check.

## Four-GPU continuation

The model/optimizer checkpoint format supports a different tensor-parallel size.
The prepared four-H200 profile uses TP4/DP1 and 32 browser slots, with conservative
resource guards of 16 CPUs and 480 GiB RAM on one node. This is a tested CPU
configuration and workflow. Checkpoint 14 received a successful full TP4 model
and optimizer reload in job 284885. Checkpoint 17 and the newer cache-release
source still require their own restore verification. Two separate two-GPU
allocations do not satisfy this single-node profile.

A prepared source snapshot is available at:
`/gpfs/scrubbed/zixianma/openwebrl-runtime/reference-stage1-tp4-cache-release-20260909`.
It preserves all five baseline recipe-file hashes. To recreate such a snapshot
from a newer preserved baseline (never the experimental working tree):

```bash
python3 scripts/prepare_resume_topology.py --source PRESERVED_SOURCE --output NEW_SOURCE
```

For an existing four-H200 allocation explicitly authorized by the user:

```bash
python3 scripts/resume_baseline.py --job-id JOB_ID --gpus 4 --source PREPARED_SOURCE --verify-resume-only --dry-run
python3 scripts/resume_baseline.py --job-id JOB_ID --gpus 4 --source PREPARED_SOURCE --verify-resume-only --launch
python3 scripts/resume_baseline.py --job-id JOB_ID --gpus 4 --source PREPARED_SOURCE --dry-run
python3 scripts/resume_baseline.py --job-id JOB_ID --gpus 4 --source PREPARED_SOURCE --launch
```

The restore-only pass runs offline, requests zero browser collections and zero
optimizer updates, and does not replace the current training pointer. Its
launcher is limited to 15 minutes within the existing allocation. Only an actual
successful `resume_verification.json` creates a verification receipt. Four-GPU
training requires a matching receipt for the checkpoint, source launcher and
allocation. If a newer checkpoint becomes available, repeat the restore check
for that checkpoint. Inspect the report and released Slurm steps before training.

The old trainer must be stopped at a safe boundary before the new trainer starts;
the wrapper refuses to fork the same lineage while its recorded old allocation
still has active steps. Offline verification may run separately while the old
trainer remains active, inside another explicitly authorized allocation. A
complete untrained rollout can be replayed after migration, preserving collection
work. The same W&B ID, dataset cursor, optimizer and scheduler state are retained.

At an evaluation boundary, the checkpoint is saved **before** evaluation. Finish
the scheduled evaluation before moving on, or migrate the preceding checkpoint
plus its complete untrained rollout so that replay triggers the due evaluation.
Do not treat a saved checkpoint alone as proof that its evaluation has completed.

Fourteen CPU resume tests cover resource/ownership checks, topology arguments,
verification receipts, preservation of the training pointer, replay selection,
and prevention of concurrent trainers. The prepared inner launcher dry run
selected checkpoint 8 / 130 Adam updates with four GPUs, TP4, 32 browsers,
48 groups × five attempts, global batch 256 and two PPO epochs. The live two-GPU
allocation correctly rejects a four-GPU request. No four-GPU allocation was
requested or consumed by these tests.

## Batch driver

`scripts/resume_baseline_4gpu.sbatch` requests one node, four H200s, sixteen CPUs,
480 GiB RAM, and eight hours in the normal QoS. Its two sequential stages are
restore-only verification followed by training/replay. The wrapper recognizes
its own batch driver while continuing to reject other compute steps and live
trainers in another allocation. Both stages share the one allocation's time
boundary; verification does not buy or extend time.

The user requested this resource configuration on 2026-09-08. Slurm's
`--test-only` accepted it and estimated $28.80 for 32 H200 GPU-hours. Automatic
approval review subsequently rejected submission pending explicit approval of
the dollar estimate; no batch job was submitted by that rejected action.
The script does not grant standing approval to submit paid jobs.

The first approved submission, job `283947`, exited after 46 seconds before any
model restore because its child `srun` mixed a typed H200 allocation with an
untyped GPU GRES request. Four-GPU child steps now request `gpu:h200:4`; the
regression suite checks the typed request. No checkpoint or W&B state changed.

## State and source

The default persistent pointer is:

`/gpfs/scrubbed/zixianma/openwebrl-runtime/current_baseline.json`

It records the current run directory, checkpoint ancestry, preserved reference source, W&B identity and pending recovery batch. `--state PATH` selects a different recorded lineage. `--source PATH` selects another explicitly prepared reference snapshot; recipe hashes and required resume support are checked. The working repository's experimental recipe files are not copied into the run.

For the current baseline, W&B is [`qcq7i4ug`](https://wandb.ai/zixianma/openwebrl/runs/qcq7i4ug), and the preserved source is `reference-stage1-tp4-cache-release-20260909` under the runtime root. The credentials come from the repository `.env` and are never included in the preflight plan. `--wandb-run-id qcq7i4ug` can assert the expected identity; it cannot silently change this lineage's W&B ID.

## Checkpoint and batch selection

The script follows `launch_manifest.json` ancestry, selects the highest completed checkpoint marker, and verifies saved iteration, Adam counters, known scheduler offset, shard extents and matching dataset cursor. A partial checkpoint directory without a completion marker is ignored. If the newest marked checkpoint fails verification, resume stops for investigation; it does not silently roll training back. This CPU validation does not substitute for the actual model/optimizer restore on GPUs.

If a complete saved rollout immediately follows that checkpoint, it is replayed before collecting fresh browser trajectories. Replay requires either matching `.provenance.json` or a unique completed `[GenerateProgress]` line. The restored dataset cursor advances by **completed + pending submitted prompt groups**, not just the 48 accepted groups. Already-trained batches are skipped. An incomplete or ambiguous recovery file stops preflight for inspection. Zip-directory checks detect interrupted saves but do not read every tensor byte; the trainer's load is the full payload check.

Example: checkpoint 5 contains 90 Adam updates; saved batch `6.pt` has 1,946 turns and 144 submitted groups. Replay performs `floor(1946 / 256) × 2 = 14` updates, reaching 104 at checkpoint 6. This remains reward observation 7, not a new collection. W&B's historical `train/step` labels are not reliable cumulative Adam counts when batch lengths change.

## Monitoring and limits

The launcher retains the reference recipe (48 accepted groups × five trajectories, global batch 256, two PPO epochs), the tested runtime fixes, lossless file-backed images during collection and training transfer, allocator trimming, and per-iteration checkpointing. Runtime files stay in scrubbed storage or node-local `/tmp`. It stops before the existing allocation ends, with a three-minute margin and a maximum of eight training hours. GNU `timeout` reports this planned online stop as raw exit code 124; the resume supervisor keeps that raw code in its exit receipt, records the pointer state as `TIME_LIMIT`, and returns success to Slurm. Verification timeouts and other launcher failures remain failures. A collection interrupted by the time boundary may need recollection unless its recovery file finished saving.

The supervisor prints progress and starts a read-only health recorder. Logs:

- `openwebrl-runtime/logs/resume-JOB_ID-TIMESTAMP.log`: launcher output.
- `openwebrl-runtime/logs/resume-JOB_ID-TIMESTAMP.json`: preflight/resume plan.
- Run directory `training.log`, `progress.log`, `health.jsonl`: detailed training, phase progress and GPU/cgroup memory samples.
- Run directory `resume_plan.json` and `resume_checkpoint_validation.json`: startup provenance.

The pointer is updated when the new launch manifest appears. This means **launched**, not verified healthy. After startup, confirm the actual checkpoint-loaded message, optimizer progress and W&B sync. After each save, validate the new checkpoint before reporting durable progress and refresh the pointer's checkpoint/replay fields. A background recorder cannot diagnose or fix failures. While an OpenWebRL run is active, the active agent checks health about every 30 minutes and investigates any observed error immediately.

CPU checks: `python3 -m unittest discover -s tests -p test_resume_baseline.py -v`. The real preflight was exercised against job 283214 and correctly selected checkpoint 5, replay batch 6 with 144 submitted groups, and the existing W&B ID. It identified the already-running trainer and monitor; no duplicate training run was launched for this test.

The command also launched the real continuation `openwebrl-4b-reference-283214-20260909T022832` from checkpoint 6, selecting no replay and starting the health recorder automatically. Source validation rejects older snapshots lacking collection-time image mapping and allocator trimming, which are required for this 240-GiB workflow.
