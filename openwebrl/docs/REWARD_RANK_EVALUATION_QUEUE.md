# Evaluating checkpoints whose training reward enters the top five

The user requested this rule on 2026-09-11, replacing the initial all-time-record
trigger. Apply it to **new, verified `train/reward` points**, ranked among distinct
generating checkpoints. It is a training-reward selection heuristic, not proof of
held-out improvement. Existing completed evaluations remain available for review.

Collection N uses the checkpoint **after training N−1**, stored in directory
`iter_(N−2)`. Evaluate that generating checkpoint, not the one saved after update N.
Duplicates for the same checkpoint are skipped, including checkpoints already
fully evaluated under the same standard local-browser protocol. Equal scores at
the fifth-place boundary keep the earlier checkpoint. Historical points seed the
ranking; they do not automatically submit retrospective jobs.

At activation, the top five reward observations were:

| Reward iteration | Generating checkpoint after training | Reward |
| --- | ---: | ---: |
| 23 | 22 | 0.5469168901 |
| 35 | 34 | 0.5062240664 |
| 31 | 30 | 0.5046979866 |
| 36 | 35 | 0.4938101788 |
| 27 | 26 | 0.4868340478 |

## Explicitly approved compute budget

The user approved: “I approve this one training resume job, and all eval jobs's
required GPUs (max 4 jobs, each with 2gpusx1-2 hours)”. The selected evaluation
profile is **2 H200 GPUs × 2 hours**, 16 CPUs and 480 GiB RAM: **4 GPU-hours**,
estimated **$3.60 per job**, at most **four jobs / 16 GPU-hours / $14.40**.
GPT-4.1 judge API usage is additional. These jobs use standard local browsers;
**stealth evaluations are paused by user instruction**.

The approval receipt is runtime `reward_eval_approval_20260911.json`. Do not
request approval again for jobs within this exact standing budget. Stop automatic
submissions at four jobs; failed or uncertain submissions conservatively reserve
a slot until their outcome is resolved. A new budget requires explicit approval.
The separately approved training continuation is job **287530**, 4 H200 × 8 hours,
held with `afterok:287371`; it is outside this four-evaluation-job cap.

## Persistent queue and workflow

`current_baseline.json` links to `reward_eval_queue.json` and the approval record.
`scripts/reward_eval_queue.py` maintains the ranking and checkpoint queue. It
only submits with **both** `--submit-approved` and the approval receipt. It checks
checkpoint/source identity and the exact batch resource declarations, locks the
queue during updates, and writes a submission-intent record before invoking
Slurm. An ambiguous submission never automatically retries.

After each completed collection, first validate the entire rollout archive,
recovery batch and W&B reward. Then run, substituting the actual reward audit:

```bash
python3 scripts/reward_eval_queue.py \
  --state /gpfs/scrubbed/zixianma/openwebrl-runtime/reward_eval_queue.json \
  --reward-audit /path/to/run/iteration_N_reward_wandb_audit.json \
  --inventory /gpfs/scrubbed/zixianma/openwebrl-runtime/qcq7i4ug_checkpoint_inventory.json \
  --submit-approved \
  --approval /gpfs/scrubbed/zixianma/openwebrl-runtime/reward_eval_approval_20260911.json
```

The current allocation's collection auditor invokes this after its checks pass.
Carry this invocation into the next allocation's auditor. The queued worker is
`scripts/evaluate_record_checkpoint_2gpu.sbatch`; its checkpoint/source/iteration
come from the validated queue entry. It executes the unchanged 300-task GPT-4.1
monitor and saves a separate W&B run `qcq7i4ug-eval-afterN-JOB` and output under
runtime `evaluations/qcq7i4ug-record-JOB-afterN`.

The active supervising agent must monitor every submitted evaluation alongside
training, check GPU restoration and W&B output, investigate failures, and cancel
when a major error requires human input. This queue is not an independent
background monitoring service. On completion, retain the job ID in the queue so
it continues to count against the four-job budget, and add the checkpoint to
`completed_evaluations` only after results and W&B have been verified.

Eleven CPU tests cover top-five qualification below the all-time high, boundary
ties, replay deduplication, completed-checkpoint skipping, invalid rewards,
checkpoint identity, and submission budget/duplicate guards. No paid job is
submitted by those tests. The submitter removes inherited `SLURM_CPU_BIND*`
variables, and the worker explicitly uses `--cpu-bind=none`: a CPU mask inherited
from an observer running on another node caused the first step of job 287588 to
fail before Python started. That evaluation was restarted inside the same paid
allocation with an explicit binding override. Its original batch controller
remains waiting; release the allocation after the supervised worker and W&B
verification finish, before the controller's deadline. This recovery does not
consume another job from the approved cap.

Job 287596 exposed another cross-node inheritance issue: the observer's
`WANDB_SERVICE` points to a node-local service socket. Both submission and the
worker's `clean_environment()` now discard it while retaining credentials.
The first attempt failed before task evaluation; the supervised repair marker
starts a fresh attempt in the same allocation, with a distinct output/W&B suffix.
Do not copy a W&B service socket between nodes or reuse a failed attempt's score.
