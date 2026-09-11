# Intermediate baseline checkpoint evaluation

Training lineage: W&B `zixianma/openwebrl/qcq7i4ug`. Inventory checked on
2026-09-11 while allocation 287371 continued training.

## Saved checkpoints and reward timing

Checkpoints after every completed training iteration **1–40** remain on disk.
Directory indices are zero-based: after iteration N is `iter_{N-1:07d}`.
The complete absolute-path inventory and CPU validation evidence are in
`/gpfs/scrubbed/zixianma/openwebrl-runtime/qcq7i4ug_checkpoint_inventory.json`.
The inventory follows the current run's resume ancestry and selects the newest
copy when a replay produced another checkpoint with the same index. It checks
saved iteration numbers, shard byte extents, and dataset cursors, not a full
tensor reload of every historical checkpoint.

All directories below are under
`/gpfs/scrubbed/zixianma/openwebrl-runtime/runs/`:

| Completed training iterations | Run directory suffix (prefix `openwebrl-4b-reference-`) |
| --- | --- |
| 1 | `281697-20260907T225751` |
| 2 | `281697-20260908T011805` |
| 3–4 | `282346-20260908T024410` |
| 5–6 | `282346-20260908T061434` |
| 7 | `283214-20260909T012629` |
| 8–9 | `283214-20260909T022832` |
| 10–15 | `284036-20260909T070508` |
| 16–18 | `284885-20260909T160238` |
| 19–23 | `285546-20260909T235114` |
| 24–29 | `286094-20260910T071441` |
| 30–34 | `286382-20260910T164338` |
| 35–40 | `287371-20260911T025909` |

`train/reward` at collection N is measured **before** its PPO updates. The policy
that generated that reward is therefore the checkpoint after N−1 training
iterations (`iter_{N-2:07d}`), or SFT for collection 1. Do not attribute a reward
spike to the checkpoint saved after that same collection.

| Reward collection | Reward | Increase from preceding point | Generating checkpoint |
| --- | ---: | ---: | --- |
| 23 | 54.69% | +10.77 percentage points | after 22, `iter_0000021` |
| 15 | 46.20% | +9.55 points | after 14, `iter_0000013` |
| 25 | 48.46% | +7.54 points | after 24, `iter_0000023` |
| 31 | 50.47% | +7.47 points | after 30, `iter_0000029` |

These are filtered, turn-weighted rewards on changing training tasks. They do
not establish that held-out performance improved by the same amount.

## First comparison

Evaluate **after 21 and after 22** on the same 300 Online-Mind2Web tasks to bracket
the largest jump. Both are in run `285546-20260909T235114`, directories
`iter_0000020` and `iter_0000021`. Then consider after 13/14 and after 23/24.
Keep the deterministic GPT-4.1 monitoring protocol, task file, screenshot/history
settings and timeouts identical. This is comparable with our existing monitor;
it is not the paper's official o4-mini evaluation protocol.

Existing full monitoring evaluations:

| Checkpoint after training iteration | Task successes / 300 | Invalid attempts | Valid-only success |
| --- | ---: | ---: | ---: |
| 10 | 70 / 300 (23.33%) | 66 | 70 / 234 (29.91%) |
| 20 | 95 / 300 (31.67%) | 68 | 95 / 232 (40.95%) |
| 21 | 89 / 300 (29.67%) | 67 | 89 / 233 (38.20%) |
| 22 | 86 / 300 (28.67%) | 64 | 86 / 236 (36.44%) |
| 30 | 96 / 300 (32.00%) | 52 | 96 / 248 (38.71%) |
| 38 | 107 / 300 (35.67%) | 72 | 107 / 228 (46.93%) |

Valid-only success is successes divided by `(300 - invalid attempts)`. After-38
has the highest observed valid-only and all-task success rates.
The valid subset differs between evaluations, so valid-only rates are not scores
on an identical task cohort.

Compare task outcomes on the common cohort and report invalidity alongside
success. Live websites and judge calls can still introduce variation between
evaluation dates. The one-success difference from 20 to 30 is not compelling
evidence of improvement.

## Prepared execution

`scripts/evaluate_baseline_checkpoint.py` defaults to a read-only plan and
requires `--execute` inside a dedicated, already authorized Slurm GPU step.
It builds a checkpoint view without modifying the original checkpoint, checks
its metadata/cursor, restores it through the preserved baseline runtime, and
uses the existing `num_rollout=0` evaluation-only path. It checks the GPU restore
log, requires all 300 outcomes, and rejects any optimizer-update records.

Each evaluation uses its own W&B ID, `qcq7i4ug-eval-afterN-JOB`, and retains the
parent run and checkpoint identity in `evaluation_manifest.json`. Native
`eval/iteration` is 1 in this evaluation-only path; compare by checkpoint identity.
Results, logs, and the full evaluation recovery file are retained under runtime
`evaluations/`. The training pointer and training W&B history are not modified.

The prepared `scripts/evaluate_baseline_pair_4gpu.sbatch` requests **4 H200 GPUs,
16 CPUs, 480 GiB RAM, 3 hours**: **12 GPU-hours**, estimated cluster charge
**$10.80**, plus judge API usage. It runs the two workers sequentially and owns
both until completion. Three hours allows margin beyond the observed 51-minute
evaluation per checkpoint and restoration/startup overhead.

The user explicitly approved this request on 2026-09-10. **Job 287046** started
at 15:49 PDT on g002, with an allocation deadline of 18:49 PDT. Slurm assigned
GPUs 4–7 and CPUs 32–47, separate from training job 286382's GPUs 0–3 and CPUs
0–15. The first evaluation created its own local Ray instance. The wrapper now
explicitly sets `RAY_ADDRESS=local` for subsequent workers so another local
training cluster cannot be selected automatically. Four CPU tests and the
preserved launcher's dry run passed; actual GPU restoration and completed
evaluation results require live verification.

The current evaluation pointer is runtime `evaluations/current_baseline_eval.json`.
The Slurm log is `logs/slurm-baseline-eval-287046.out`; per-checkpoint outputs
are `evaluations/qcq7i4ug-287046-after21` and `...-after22`. Each includes an
`evaluation.log`, checkpoint validation, and separate W&B identity. The
submission receipt is `logs/submission-baseline-eval-287046.json`. This approval
does not authorize another submission or an extension.

Job 287046 failed during startup after **2:13**, before any task evaluation:
the native `num_rollout=0` path initialized an optimizer scheduler with zero
decay steps. The wrapper now supplies a positive initialization horizon and
`--use-checkpoint-opt-param-scheduler`; the actual saved scheduler then replaces
that initialization state. A CPU check using both real checkpoint states
verified exact scheduler restoration, unchanged Adam counters (280 and 292),
and zero requested training rollouts. Evidence:
`evaluations/eval_only_scheduler_cpu_validation.json`. No training source or
checkpoint was changed.

The batch controller now waits for a per-attempt `retry_after_repair` marker
after a worker failure, bounded by the original allocation deadline. This lets
the supervising agent diagnose and repair within paid time; it does not retry
blindly or extend the allocation. Retry outputs and W&B runs have distinct
attempt suffixes. Cancel the allocation if a failure requires human input.

The failed attempt used approximately $0.13 of GPU time. The user subsequently
explicitly approved **4 H200s for 3 hours**, 16 CPUs / 480 GiB, estimated **$10.80**
plus judge API usage. Replacement **287370** started on **g004** at September 10
**19:57 PDT**, ending **22:57 PDT**. It is isolated from training 287371 on g005.

Its outputs are `evaluations/qcq7i4ug-287370-after21` and `...-after22`, with
controller log `logs/slurm-baseline-eval-287370.out` and submission receipt
`logs/submission-baseline-eval-287370.json`. The first evaluation restored the
selected checkpoint and began the 300-task monitor successfully.

Ray omitted the successful checkpoint-restore stdout line from the forwarded
evaluation log, although the actor's own stdout contained it. The supervisor now
captures this evidence directly, checking actor membership in the authorized
Slurm job and the exact checkpoint path/index, and saves
`checkpoint_restore_evidence.json`. Final validation accepts that receipt or the
original forwarded line. Nine CPU tests cover restore identity, incomplete
outcomes, child failure, and unexpected optimization records.

The already-running first wrapper predates this change. If it fails only during
final bookkeeping after completing evaluation, the supervising agent can validate
its results against the captured receipt and W&B before releasing the controller's
repair marker. A retry recognizes a verified complete result from the same
checkpoint/source/protocol/job and advances without rerunning its tasks. Future
workers capture the receipt automatically and save their launcher exit status.
No evaluation recipe or training source was changed.


The after-21 evaluation completed all 300 tasks in **53:10**: **89 successes
(29.67%)**, 67 invalid attempts (22.33%), and valid-only success 89/233 (38.20%).
All **41 logged evaluation scalars** matched W&B history row 0; evidence is
`evaluations/qcq7i4ug-287370-after21/wandb_audit.json`. Ray eventually forwarded
the restore line during shutdown, so the original worker validated successfully
without a retry. The batch controller immediately advanced to after-22.


Job **287370 completed successfully in 1:48:47**, releasing its allocation.
The after-22 checkpoint scored **86/300 (28.67%)**, with 64 invalid attempts
(21.33%) and valid-only success 86/236 (36.44%). All 41 scalars matched its separate
W&B run; receipt: `evaluations/qcq7i4ug-287370-after22/wandb_audit.json`.
Thus the training-reward jump from collection 22 to 23 did not correspond to a
higher task-success score in this paired checkpoint evaluation. At that time,
after-30 was the best evaluated checkpoint at 96/300 (32%), ahead of after-20 at
95/300. Stealth evaluation is now paused by user instruction; job 287521 was
canceled and its partial attempt is not a valid comparison result.

## First top-five-triggered result: after 38

Collection 39 produced verified `train/reward=0.4877300613`, entering fifth place
and triggering evaluation of its generating checkpoint **after 38**, directory
`287371-20260911T025909/iter_0000037`. Job **287588** used the approved two-H200,
two-hour profile. Its first step failed CPU binding before Python started; an
explicit `--cpu-bind=none` worker recovered inside the same allocation. The
submitter and future batch workers now prevent inherited observer CPU masks.

Full TP2 model/optimizer restoration from the selected checkpoint was verified.
The unchanged deterministic 300-task GPT-4.1 monitor completed in **52:21**,
with **107 successes, 72 invalid attempts, and 228 valid attempts**. All **41
scalar metrics** exactly matched W&B history row 0 (tolerance 1e-7). This is
11 more successes than after-30, alongside 20 more invalid attempts. The
valid-only comparison uses different subsets; this single live-web evaluation
does not by itself establish statistical significance.

Results and full recovery data are under runtime
`evaluations/qcq7i4ug-record-287588-after38/`, including `metrics.json`,
`evaluation.log`, `wandb_audit.json`, `checkpoint_restore_evidence.json`, and
`runtime/rollout_recovery/eval_0.pt`. W&B:
[after-38 evaluation](https://wandb.ai/zixianma/openwebrl/runs/qcq7i4ug-eval-after38-287588).
Worker **287588.1 completed with exit 0**. After verification, the waiting
original batch controller was canceled to release unused allocation time;
the resulting Slurm cancellation does not indicate failed evaluation results.
One of the four approved reward-triggered evaluation jobs has been used.
