# Intermediate baseline checkpoint evaluation

Training lineage: W&B `zixianma/openwebrl/qcq7i4ug`. Inventory checked on
2026-09-10 while allocation 286382 continued training.

## Saved checkpoints and reward timing

Checkpoints after every completed training iteration **1–33** remain on disk.
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
| 30–33 | `286382-20260910T164338` |

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

| Checkpoint after training iteration | Task successes / 300 | Invalid attempts |
| --- | ---: | ---: |
| 10 | 70 / 300 (23.33%) | 66 |
| 20 | 95 / 300 (31.67%) | 68 |
| 30 | 96 / 300 (32.00%) | 52 |

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

**This evaluation allocation has not been submitted or approved.** CPU planning
and safety checks do not substitute for GPU verification of this new wrapper.
Explicit approval of this resource request and budget is required before sbatch.
