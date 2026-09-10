# C2 update-500/update-700 full evaluation

Status: **deprioritized by the user on 2026-09-09; no Slurm job was submitted
and no GPU allocation was created**. This remains a prepared optional
checkpoint study. It evaluates both checkpoints on all 300 Online-Mind2Web
tasks and starts two isolated SGLang
servers on one H200 and gives each checkpoint six browser workers. This is the
same concurrent layout that completed the 100-task update-500/update-700 pair
in about one hour.

## Why all 300 need two reported strata

The fixed 100-task sample was already used to identify updates 500 and 700 as
the leading checkpoints. Reusing those tasks alone would overstate the
strength of the comparison. The full evaluation therefore predeclares:

- **Primary checkpoint comparison:** the other 200 tasks, which did not affect
  checkpoint selection.
- **Stability check:** the original 100 checkpoint-selection tasks.
- **Aggregate:** all 300 tasks.

The exact IDs and indices are frozen in
[arm_c2_full300_cohorts.json](arm_c2_full300_cohorts.json). The task-file
SHA-256 is `8343c23be98d6d63856e9b53ff3884222be099cc0f55ad5edb475176f54317ed`.
Results will report overall and valid-only success, Wilson intervals,
common-valid paired wins/losses with exact McNemar tests, unavailable counts,
and the existing trajectory/loop diagnostics. Unavailable outcomes remain in
the overall denominator. Any later retry will be labeled as a separate
sensitivity analysis.

## Frozen protocol

| Setting | Value |
| --- | --- |
| Policies | merged C2 update 500 and update 700 |
| Tasks | all 300 Online-Mind2Web tasks |
| Actor sampling | seed 42, temperature 0.7, top-p 0.9, maximum 1024 new tokens |
| Browser horizon | 30 turns, full history, current screenshot |
| Judge | `o4-mini`, Online-Mind2Web/AgentTrek terminal-success protocol |
| Parallelism | 6 browser workers per checkpoint; both checkpoints concurrent |
| Serving | two SGLang servers, tensor parallel 1, memory fraction 0.30 each |

The historical original actor is included in the generated analysis, but its
300 rollouts were collected earlier. Update 500 versus update 700 is the clean
fresh matched comparison; any comparison to the historical actor remains
subject to live-site drift.

## Execution and outputs

The prepared entry point is
[`scripts/run_arm_c2_full300_pair.sbatch`](../../scripts/run_arm_c2_full300_pair.sbatch).
It requests exactly **one H200 for four hours, four CPUs, and 120 GB host
memory** on account `zixianma`, partition `gpu-h200`, QoS `normal`: a maximum
budget of **4 H200-hours**. Prior throughput predicts roughly three hours for
both 300-task runs, leaving startup and shutdown margin.

Runtime outputs will be written below:

```text
/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/
  c2-full-282782-20260908T075414Z/evaluation/checkpoint-full-300/
```

`pair-status.json` tracks both result counts. Each checkpoint has independent
server/evaluation logs and resumable task results. When both finish, the
controller automatically writes `comparison.json` and `comparison.md` using
the frozen cohort manifest. The launcher refuses to run outside its assigned
Slurm cgroup, with another GPU count, or with mismatched checkpoint/dataset
provenance.

Submitting this script requires explicit approval for the resource request;
the script itself records no authorization.
