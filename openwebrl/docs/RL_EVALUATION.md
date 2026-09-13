# RL checkpoint evaluation: baseline, Browser Use, and scheduling

Reference-policy checkpoint evaluations, the separate Browser Use protocol, and reward-ranked evaluation scheduling. Preserve each protocol and cohort when comparing results. Allocation and cancellation entries remain dated experiment history.

## Contents

- [Intermediate baseline checkpoint evaluation](#baseline-checkpoint-evaluation)
- [Browser Use checkpoint evaluations](#browser-use-checkpoint-evaluation)
- [Evaluating checkpoints whose training reward enters the top five](#reward-rank-evaluation-queue)

---

<!-- document:BASELINE_CHECKPOINT_EVALUATION.md:start -->
<a id="debug-evaluation-wandb-migration-20260913"></a>
## Debug and remaining evaluation W&B migration, September 13

Moved another 19 inactive runs to `zixianma/openwebrl-evals`: six early pipeline
tests, five browser benchmarks/diagnostics, three ARM calibration attempts,
four synthetic GPU diagnostics, and the now-finished checkpoint-80 temperature-0
evaluation. The training project retains only baseline `qcq7i4ug` and the real
ARM training pilot `arm-turn-bonus-beta0.5-after70-293194`. New debug, smoke-test,
calibration and standalone evaluation launches belong in `openwebrl-evals`.

Verified all moved histories, summaries, configurations, statuses, run/file
metadata and artifact references against pre-move snapshots. Five linked
artifacts remain in their original project; the run references are preserved.
Audit snapshots and the URL mapping are in
`/gpfs/scrubbed/zixianma/openwebrl-runtime/wandb-debug-migration-20260913/`.
Checkpoint-80 temperature-0.6 job 294094 is logging directly to `openwebrl-evals`.

| Run | Preserved state |
| --- | --- |
| [l2puwtbj](https://wandb.ai/zixianma/openwebrl-evals/runs/l2puwtbj) | killed |
| [iamowkmv](https://wandb.ai/zixianma/openwebrl-evals/runs/iamowkmv) | killed |
| [jf5mo2ze](https://wandb.ai/zixianma/openwebrl-evals/runs/jf5mo2ze) | killed |
| [92182ncf](https://wandb.ai/zixianma/openwebrl-evals/runs/92182ncf) | failed |
| [6yineik5](https://wandb.ai/zixianma/openwebrl-evals/runs/6yineik5) | failed |
| [v2d9bk11](https://wandb.ai/zixianma/openwebrl-evals/runs/v2d9bk11) | crashed |
| [browser-scale-291224-tp4-b256](https://wandb.ai/zixianma/openwebrl-evals/runs/browser-scale-291224-tp4-b256) | failed |
| [browser-scale-291224-tp4-b192](https://wandb.ai/zixianma/openwebrl-evals/runs/browser-scale-291224-tp4-b192) | failed |
| [arm-turn-bonus-calibration-after70-291983](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-turn-bonus-calibration-after70-291983) | finished |
| [arm-turn-bonus-calibration-after70-291983-r2](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-turn-bonus-calibration-after70-291983-r2) | crashed |
| [browser-diagnostic-291905](https://wandb.ai/zixianma/openwebrl-evals/runs/browser-diagnostic-291905) | finished |
| [arm-turn-bonus-calibration-after70-292551](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-turn-bonus-calibration-after70-292551) | failed |
| [browser48-290926-20260913](https://wandb.ai/zixianma/openwebrl-evals/runs/browser48-290926-20260913) | finished |
| [browser48-hold-290926-20260913](https://wandb.ai/zixianma/openwebrl-evals/runs/browser48-hold-290926-20260913) | finished |
| [arm-sol-gpu-diagnostic-294080-training](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-sol-gpu-diagnostic-294080-training) | failed |
| [qcq7i4ug-stealth-o4-after80-t0-294093](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-stealth-o4-after80-t0-294093) | finished |
| [arm-sol-gpu-diagnostic-294103-training](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-sol-gpu-diagnostic-294103-training) | finished |
| [arm-sol-gpu-diagnostic-294103-training-r1](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-sol-gpu-diagnostic-294103-training-r1) | crashed |
| [arm-sol-gpu-diagnostic-294103-training-r2](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-sol-gpu-diagnostic-294103-training-r2) | finished |

<a id="historical-evaluation-wandb-migration-20260913"></a>
## Historical evaluation W&B migration, September 13

Moved 11 inactive standalone evaluation runs from `zixianma/openwebrl` to
[`zixianma/openwebrl-evals`](https://wandb.ai/zixianma/openwebrl-evals): eight
finished evaluations and three failed or partial attempts. Verified every run's
ID, history rows, summary, configuration, state, name, group, file checksums and
artifact references against its pre-move snapshot. W&B's internal storage ID
changed only in its project component. The two artifacts logged by the after-21
and after-22 runs remain in their original project with their references intact.

Training runs and their scheduled evaluation history remain in `openwebrl`.
At the first migration, checkpoint-80 temperature-0 run 294093 remained there
while active; it has now finished and moved in the second migration above.
Temperature-0.6 run 294094 is running in `openwebrl-evals`.
Operational snapshots and the URL mapping are stored in
`/gpfs/scrubbed/zixianma/openwebrl-runtime/wandb-eval-migration-20260913/`.

| Run | Preserved state |
| --- | --- |
| [qcq7i4ug-eval-after21-287046](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after21-287046) | failed |
| [qcq7i4ug-eval-after21-287370](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after21-287370) | finished |
| [qcq7i4ug-eval-after22-287370](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after22-287370) | finished |
| [qcq7i4ug-eval-browseruse-after20-287521](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-browseruse-after20-287521) | crashed |
| [qcq7i4ug-eval-after38-287588](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after38-287588) | finished |
| [qcq7i4ug-eval-after39-287596-r1](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after39-287596-r1) | finished |
| [qcq7i4ug-eval-browseruse-after38-287879](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-browseruse-after38-287879) | finished |
| [qcq7i4ug-eval-after52-288791](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after52-288791) | finished |
| [qcq7i4ug-eval-after58-290361](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after58-290361) | finished |
| [qcq7i4ug-benchmark-after58-291005](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-benchmark-after58-291005) | killed |
| [qcq7i4ug-eval-after69-293585](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after69-293585) | finished |

<a id="separate-evaluation-wandb-project-20260913"></a>
## Separate W&B project for new standalone evaluations

New standalone evaluation launches use **`zixianma/openwebrl-evals`**; training
continues in `zixianma/openwebrl`. Temperature-0 evaluation **294093** stayed
in its original project while active and moved after finishing. Temperature-0.6
job **294094** started directly in the new project.
Scheduled evaluations emitted by the trainer remain attached to its training
history. The historical migrations above move inactive standalone
evaluations; active runs are not restarted.

Both standalone launchers set the CLI project, environment and saved manifest
consistently; an explicit `--wandb-project` override remains available. The
shared planner used by ARM training pilots retains its training default.
Fourteen evaluation tests and two temperature-identity tests pass. The queued
job's real launch plan resolves to
[its new evaluation W&B location](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-stealth-o4-after80-t0.6-294094).
The project and run are now visible in W&B.

<a id="stealth80-temperature-pair-20260913"></a>
## Checkpoint 80: approved stealth/o4-mini temperature comparison

The user superseded the earlier three-checkpoint plan and explicitly approved
**checkpoint 80 only**, at actor temperatures **0 and 0.6**. No after-38 or
after-58 rerun is authorized. Both jobs evaluate all **300 Online-Mind2Web
tasks**, with Browser Use stealth, the same o4-mini/AgentTrek judge, top-p 0.95,
top-k 20, 4096 response tokens, 32768 context tokens, 30 turns and eight browser
sessions. Judge API defaults are unchanged; temperature varies the actor only.

| Actor temperature | Job | Dependency | Resources | Maximum GPU-hours |
| ---: | ---: | --- | --- | ---: |
| 0 | 294093 | none; started on g020 | 2 H200, 16 CPUs, 480 GiB, 3 hours | 6 |
| 0.6 | 294094 | afterok:294093 | 2 H200, 16 CPUs, 480 GiB, 3 hours | 6 |

Slurm confirmed **$5.40 estimated per job**, **$10.80 / 12 GPU-hours total**,
plus Browser Use and o4-mini usage. The batch controller awaits its evaluation;
failure of the first job holds the second for inspection. Jobs finish early
when evaluation completes. Training remains separate and checkpoints 38/58
are untouched. The 299/300 after-58 partial result remains labeled partial.

Temperature 0 uses isolated source `reference-stealth-o4-temp0-20260913`;
temperature 0.6 retains `reference-paper-om2w-20260912`. Only the generation
module's actor temperature and matching YAML setting change. Protected-source
hash checks and real CPU configuration checks passed for both temperatures;
13 evaluation regressions and two temperature-identity tests pass. The wrapper
checks that requested, manifest and executable temperatures agree, and uses
separate temperature-labeled W&B IDs. Temperature 0 is a controlled ablation,
not the paper's temperature-0.6 sampling setting.

Checkpoint: `iter_0000079` in runtime run
`openwebrl-4b-reference-293510-20260913T104752` (after 80).
Approval, exact plans, submission receipts and checks:
`/gpfs/scrubbed/zixianma/openwebrl-runtime/stealth80_temperature_pair_20260913.json`.
Logs: `logs/slurm-stealth-o4-294093.out` and `logs/slurm-stealth-o4-294094.out`
under runtime storage. Results: `evaluations/qcq7i4ug-stealth-o4-JOB-after80-tTEMP/`.
[Temperature 0 W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-stealth-o4-after80-t0-294093),
[temperature 0.6 W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-stealth-o4-after80-t0.6-294094).

<a id="stealth-o4-checkpoints38-58-80-20260913"></a>
## Superseded proposal: stealth/o4-mini after 38, 58 and 80

The user requested matching full-300 evaluations of checkpoints **38, 58 and
80**: Browser Use stealth, o4-mini/AgentTrek judge, **actor temperature 0.6**,
top-p 0.95, top-k 20, 4096 output tokens, 32768 context tokens, 30 turns, and
eight browser sessions. The judge uses its native API defaults. Use frozen
source `reference-paper-om2w-20260912`, separate output/W&B identities, and
zero optimizer updates. All three checkpoint component/source checks pass.

After-58's existing result is **partial**: its two-hour allocation expired
with **299/300** saved task records (175 successes, 290 valid, nine invalid,
one missing). **175/300 = 58.33%** is a lower bound, **175/299 = 58.53%** is
completed-only and **175/290 = 60.34%** is valid-only. Retain that attempt;
prepare a fresh full-300 after-58 evaluation rather than silently merging it
with a new attempt. After-38's earlier stealth result used GPT-4.1/greedy;
after-80's earlier result used local browsers/GPT-4.1. Neither is the requested
stealth/o4-mini comparison.

**Budget pending; no jobs submitted:** three sequential jobs, each **2 H200,
16 CPUs, 480 GiB, three hours**; maximum **18 GPU-hours total**, estimated
**$16.20**, plus Browser Use hosting and o4-mini calls. Use `afterok` dependencies
so only one eight-session evaluation runs at once and a failed predecessor
holds later jobs for inspection. Three hours adds headroom over the previous
two-hour partial attempt; unused time is released upon completion.

Template: `scripts/evaluate_stealth_checkpoint_2gpu_3hour.sbatch`.
Exact checkpoint paths, worker plans, protocol and template hash:
`/gpfs/scrubbed/zixianma/openwebrl-runtime/stealth_eval_38_58_80_plan_20260913.json`.
Shell syntax and 13 evaluation regression tests pass. The native worker will
inspect checkpoint metadata and verify GPU restoration during execution.
Update [RL_RESULTS.md](RL_RESULTS.md) with both success denominators and invalid
counts after each evaluation; preserve protocol and partial-result labels.

<a id="baseline-checkpoint-evaluation"></a>
## Intermediate baseline checkpoint evaluation

_Source record: `BASELINE_CHECKPOINT_EVALUATION.md`. Dated entries retain their historical context._


Training lineage: W&B `zixianma/openwebrl/qcq7i4ug`. Inventory checked on
2026-09-11 through saved iteration 53 at the end of allocation 287949.

<a id="baseline-checkpoint-evaluation--saved-checkpoints-and-reward-timing"></a>
### Saved checkpoints and reward timing

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
| 41–45 | `287530-20260911T105645` |
| 46–53 | `287949-20260911T183514` |

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

<a id="baseline-checkpoint-evaluation--first-comparison"></a>
### First comparison

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
| 39 | 98 / 300 (32.67%) | 72 | 98 / 228 (42.98%) |
| 40 | 100 / 300 (33.33%) | 69 | 100 / 231 (43.29%) |
| 50 | 105 / 300 (35.00%) | 66 | 105 / 234 (44.87%) |
| 52 | 92 / 300 (30.67%) | 73 | 92 / 227 (40.53%) |
| 58 | 109 / 300 (36.33%) | 70 | 109 / 230 (47.39%) |
| 60 | 105 / 300 (35.00%) | 70 | 105 / 230 (45.65%) |
| 69 | 103 / 300 (34.33%) | 79 | 103 / 221 (46.61%) |
| 70 | 103 / 300 (34.33%) | 71 | 103 / 229 (44.98%) |
| 80 | 114 / 300 (38.00%) | 71 | 114 / 229 (49.78%) |

Valid-only success is successes divided by `(300 - invalid attempts)`. After-80
has the highest observed local-browser valid-only and all-task success rates.
The tables-only overview is [RL_RESULTS.md](RL_RESULTS.md).
The valid subset differs between evaluations, so valid-only rates are not scores
on an identical task cohort.

Compare task outcomes on the common cohort and report invalidity alongside
success. Live websites and judge calls can still introduce variation between
evaluation dates. The one-success difference from 20 to 30 is not compelling
evidence of improvement.

<a id="baseline-checkpoint-evaluation--prepared-execution"></a>
### Prepared execution

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

<a id="baseline-checkpoint-evaluation--first-top-five-triggered-result-after-38"></a>
### First top-five-triggered result: after 38

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
[after-38 evaluation](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after38-287588).
Worker **287588.1 completed with exit 0**. After verification, the waiting
original batch controller was canceled to release unused allocation time;
the resulting Slurm cancellation does not indicate failed evaluation results.
One of the four approved reward-triggered evaluation jobs has been used.

<a id="baseline-checkpoint-evaluation--second-top-five-triggered-result-after-39"></a>
### Second top-five-triggered result: after 39

Collection 40 reward **0.5035112360** entered fourth place and triggered the
checkpoint after 39 (`287371-20260911T025909/iter_0000038`). Job **287596** used
two H200s with a two-hour limit. Its first attempt failed before evaluation
because an inherited `WANDB_SERVICE` referenced another node's socket. The
submitter and worker now clear that setting; a supervised retry reused the same
allocation with a separate output/W&B suffix.

The retry restored the selected checkpoint on both GPUs, completed all 300 tasks,
and scored **98/300 (32.67%)**, with **72 invalid attempts** and valid-only success
**98/228 (42.98%)**. All 41 scalars matched W&B history row 0. The recovery file's
ZIP directory is complete. Job and controller both report **COMPLETED / exit 0**,
elapsed **1:01:53** including startup/recovery; the successful worker took 54:41.

Artifacts: runtime `evaluations/qcq7i4ug-record-287596-after39-retry1/`.
W&B: [after-39 evaluation](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after39-287596-r1).
After-38 remains the highest observed all-task and valid-only result. **Two of
four** approved reward-triggered jobs have now been used. Stealth stays paused.

<a id="baseline-checkpoint-evaluation--scheduled-after-40-evaluation-recovered-in-training-job-287530"></a>
### Scheduled after-40 evaluation recovered in training job 287530

The first after-40 evaluation was interrupted at 201/300 tasks by allocation
287371's planned timeout and has no valid final score. Job **287530** restored
the exact after-40 model and optimizer on four GPUs, then ran the full 300-task
monitor before collecting iteration 41. This completed in **49:31** with
**100 successes, 69 invalid attempts, and 231 valid attempts**: **33.33% all-task
success**, **43.29% valid-only**.

All 41 scalar metrics matched main run `qcq7i4ug`, **history row 660,
`eval/iteration=40`**. The pending-evaluation flag was cleared only after that
verification and a complete recovery ZIP check. Audit and data:
runtime `runs/openwebrl-4b-reference-287530-20260911T105645/iteration_40_scheduled_eval_audit.json`
and `rollout_recovery/eval_39.pt`. This scheduled evaluation used the training
allocation and does not consume a reward-triggered evaluation job. After-38
remains the highest observed result; two approved triggered jobs remain.

<a id="baseline-checkpoint-evaluation--scheduled-after-50-evaluation"></a>
### Scheduled after-50 evaluation, September 11

Training job **287949** completed the 300-task local-browser evaluation after
iteration 50 at approximately **16:23 PDT**, using the new **32-task browser
gate**. It took **28:22**, scoring **105/300 = 35.00%** overall, with **66 invalid
attempts** and valid-only success **105/234 = 44.87%**. The after-38 local result
(35.67% overall, 46.93% valid-only) remains the highest observed. Separate live
runs and different valid subsets do not establish a statistically reliable
checkpoint ranking. The after-38 stealth result uses a different browser backend
and is recorded separately below.

All **41 evaluation scalars match W&B history row 805**, `eval/iteration=50`,
in the main run `qcq7i4ug`. The 82,586,413,765-byte saved evaluation file has a
complete ZIP central directory and metadata entry. Audit:
runtime `runs/openwebrl-4b-reference-287949-20260911T183514/iteration_50_scheduled_eval_audit.json`;
data: `rollout_recovery/eval_49.pt` in that directory. This evaluates the policy
immediately after training, without a separate checkpoint reload. The saved
checkpoint `iter_0000049` has 604 Adam updates and passed CPU validation.

The pending-evaluation flag was cleared after verification, and collection 51
started normally. This scheduled evaluation uses the existing training budget;
the two remaining approved top-five-triggered evaluation jobs are unchanged.

<a id="baseline-checkpoint-evaluation--third-top-five-result-after-52"></a>
### Third top-five-triggered result: after 52

Collection 53's verified reward **0.539293** ranked second and triggered the
policy that generated it, **checkpoint after 52 / `iter_0000051` / 626 Adam
updates**. Under the existing four-job approval, job **288791** requested two
H200s, 16 CPUs, 480 GiB and up to two hours (estimated maximum GPU cost $3.60,
plus judge usage). It ran on g001 and completed with exit 0 in **52:40**;
all 300 evaluation tasks took **48:55**. The TP4 checkpoint restored on TP2.

Result: **92/300 = 30.67%** overall; **73 invalid attempts**; valid-only
**92/227 = 40.53%**. All 41 metrics match W&B history row 0 in
[the after-52 evaluation run](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after52-288791).
The recovery ZIP central directory and metadata entry are complete. Artifacts
and audit reports: runtime `evaluations/qcq7i4ug-record-288791-after52/`.
This uses the established **16-task local-browser evaluation** source, not
Browser Use stealth. The 32-browser default applies to the four-GPU training
profile and its scheduled evaluations.

This held-out result did not improve despite the triggering training reward;
filtered rewards on changing training prompts do not establish held-out gains.
After-38 remains the highest observed checkpoint, with uncertainty from single
live-web runs. **Three of four approved triggered jobs are used; one remains.**

<a id="baseline-checkpoint-evaluation--fourth-top-five-result-after-58"></a>
### Fourth top-five-triggered result: after 58, September 12

Collection 59's verified reward **0.505842** ranked fourth and triggered
**after-58 / `iter_0000057` / 690 Adam updates**. Job **290361** used the final
slot of the four-job approval: two H200s, 16 CPUs, 480 GiB and a two-hour ceiling
(estimated maximum GPU cost $3.60, plus judge usage). It ran on **g022**, restored
the TP4 training checkpoint on TP2, and completed successfully at **02:18:43 PDT**
in **54:24**. The 300 evaluation tasks took **50:33**.

The result is **109/300 = 36.33%** overall, with **70 invalid attempts** and
**109/230 = 47.39%** valid-only success. This is the highest observed local-browser
result, but only **two additional successes** over after-38 (107/300). Single
live-web runs and changing valid subsets do not establish a reliable improvement.
The GPT-4.1 monitoring evaluator and 16-task local-browser source are unchanged;
this is not a stealth or official paper-protocol evaluation.

All **41 scalars** match W&B history row 0 in
[the after-58 evaluation run](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after58-290361).
Artifacts: runtime `evaluations/qcq7i4ug-record-290361-after58/`, including
`wandb_audit.json`, `checkpoint_restore_evidence.json` and
`evaluation_artifact_audit.json`. The saved **82,884,798,301-byte** evaluation ZIP
has a complete central directory and metadata entry; this check does not reload
all tensor payloads. Slurm and the evaluation launcher both report exit 0.
**All four authorized triggered evaluation jobs have completed; no slots remain.**
Scheduled evaluations within the active training allocation use that allocation's
existing budget.

<a id="baseline-checkpoint-evaluation--scheduled-after-60-evaluation"></a>
### Scheduled after-60 evaluation, September 12

Training allocation **288861** completed the full 300-task local-browser monitor
after checkpoint 60 in **29:21**, at approximately **03:01 PDT**. It used the
training profile's **32 concurrent browser tasks** and scored **105/300 = 35.00%**
overall, with **70 invalid attempts** and **105/230 = 45.65%** valid-only success.
After-58 (109/300) remains the highest observed local result; four successes
between these individual live-web runs do not establish a reliable ranking.

All **41 scalars** match main W&B run `qcq7i4ug`, **history row 951**,
`eval/iteration=60`. The **82,930,649,761-byte** recovery ZIP is complete.
Audit: runtime `runs/openwebrl-4b-reference-288861-20260912T040027/iteration_60_scheduled_eval_audit.json`;
data: `rollout_recovery/eval_59.pt`. This evaluates the policy immediately after
training rather than performing a separate reload. The checkpoint has **710
Adam updates** and passed metadata/extents/cursor and finite CPU sample checks.
The pending-evaluation flag was cleared after audit, and collection 61 began.
This used the active training allocation and did not create an additional job.

<!-- document:BASELINE_CHECKPOINT_EVALUATION.md:end -->

---

<!-- document:BROWSER_USE_CHECKPOINT_EVALUATION.md:start -->
<a id="browser-use-checkpoint-evaluation"></a>
## Browser Use checkpoint evaluations

_Source record: `BROWSER_USE_CHECKPOINT_EVALUATION.md`. Dated entries retain their historical context._


<a id="browser-use-checkpoint-evaluation--checkpoint-38-stealth-job-287879-running--2026-09-11"></a>
### Checkpoint 38 stealth job 287879 completed — 2026-09-11

The user selected **after training iteration 38** for a new stealth evaluation
and explicitly approved its prepared budget. **Job 287879** started on **g009
at 10:54:58 PDT**, with a deadline of **13:55:02 PDT**. Slurm confirmed the
requested two H200s, 16 CPUs, 480 GiB and three-hour limit. Receipt:
`stealth_after38_request_20260911.json` in the runtime.
Its checkpoint is runtime
`runs/openwebrl-4b-reference-287371-20260911T025909/iter_0000037`, with 468 Adam
updates. Its completed local-browser evaluation scored **107/300 (35.67%)**,
valid-only **107/228 (46.93%)**, the best observed result among evaluated
checkpoints through 40. The previous 20/30 stealth pair remains canceled.

The new template is `scripts/evaluate_browser_use_after38_2gpu.sbatch`: **2 H200,
16 CPUs, 480 GiB, up to 3 hours**, **6 GPU-hours / estimated $5.40**. It owns and
awaits its worker, uses explicit CPU binding, and permits only supervised retries
within the same deadline. This three-hour request exceeds the existing top-five
queue's two-hour-per-job approval; the user supplied the required separate
explicit approval before submission. It does not consume or extend the remaining two local-browser queue
slots unless the user explicitly changes that budget.

The isolated source is `reference-browseruse-eight-decimal-v2-20260911`. Its task gate is
**eight concurrent browsers**, below the previously observed ten-session account
limit. The 300 tasks, checkpoint, GPT-4.1 judge, prompts, decoding and maximum
steps remain the same. Source hashes record the browser backend, timeout and
concurrency changes. Fifteen CPU preparation/wrapper tests and the shell syntax
check passed; the exact checkpoint dry-run plan is saved under
`evaluations/browser-use-preflight-after38-20260911/evaluation_plan.json`.
W&B startup confirmed [qcq7i4ug-eval-browseruse-after38-287879](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-browseruse-after38-287879).
The output is runtime `evaluations/qcq7i4ug-browseruse-287879-after38`; its
`evaluation.log` contains detailed progress, and the controller log is
`logs/slurm-stealth-after38-287879.out`. Full TP2 model/optimizer restoration
of the TP4 checkpoint passed. The job and worker completed with exit 0 after
**1:52:14** allocation time; all 300 tasks took **1:48:24**.

| Backend, checkpoint after 38 | Success / all tasks | Success / valid tasks | Invalid tasks |
| --- | ---: | ---: | ---: |
| Local browser, job 287588 | 107/300 = **35.67%** | 107/228 = **46.93%** | 72 (24.0%) |
| Browser Use stealth, job 287879 | 137/300 = **45.67%** | 137/291 = **47.08%** | 9 (3.0%) |

All **41 metrics** match W&B history row 0 (`wandb_audit.json`). The saved
`runtime/rollout_recovery/eval_0.pt` is 93,803,489,805 bytes; its ZIP central
directory and metadata entry are complete (`evaluation_artifact_audit.json`).
This is structural validation, not a full reread of all tensor data.

All-task success increased **10 percentage points**, while valid-only success
changed by about **0.15 points**. Much lower invalid frequency is the main
observed difference. These are separate live-web runs with different valid
subsets and browser concurrency/timeouts; they do not isolate a causal effect
of stealth or establish a new policy improvement. The unsuffixed scalar
`eval/online-mind2web-monitor = 0.304037` is a turn-weighted reward, **not** the
45.67% task success rate. Use `/task/success_rate_all_completed` for that score.

The run recorded **300 confirmed stopped session receipts**, with no unconfirmed
receipts or runtime provider-limit/SDK-schema/stop errors. The independent
`browser_shutdown_cost_audit.json` also confirms **300/300 stopped** through
the provider API. Provider-reported browser cost is **$0.319000**,
and proxy cost is **$1.425584**, excluding GPU and judge usage.
These are API-reported amounts, not an official invoice. Proxy charges remain
nonzero despite the requested null proxy setting. The audit uses paced reads
after an initial eight-request burst hit the API rate limit; this read-only
audit limit did not affect evaluation tasks.

A fresh one-browser CDP connectivity probe reached Example Domain and saved
its screenshot. The session was independently confirmed stopped. The service
still reported a tiny proxy charge (**$0.00000491**) even with explicit
`proxyCountryCode: null`; the SDK preserves null through its HTTP serializer.
The provider documents null as disabling proxies, so this remains a service or
accounting discrepancy rather than a verified no-proxy run. Do not claim proxy
cost is zero. Browser hosting for 300 sessions capped at 12 minutes is at most
**$1.20 before refunds**, plus metered proxy and GPT-4.1 judge usage. These service
charges are additional to GPU cost. Preflight receipts stay in the same directory.


The sustained eight-browser probe found a second issue: the provider can return
billing values such as `2.8740614652633667968750E-7`, which SDK 3.11.3 rejects
under its decimal-only schema. The prepared source includes an isolated SDK copy
whose six browser-session billing-field patterns accept scientific notation.
Fifty-four actual model-validation checks pass, including rejection of nonfinite
and malformed values; three session-cleanup CPU tests also pass. The original SDK
and training sources remain preserved.

After this repair, two waves of eight concurrent browsers each performed repeated
navigation and screenshots for at least 45 seconds. All **16/16 succeeded and
were independently confirmed stopped**. Their recorded hosting total was
**$0.005333**, with **$0.00004568** reported proxy cost. The final SDK schema also
rechecked all 16 stopped sessions. Evidence:
`evaluations/browser-use-preflight-after38-20260911-fixed/concurrency_report.json`
and `final_sdk_shutdown_audit.json`. The earlier failed probe is preserved, and
its eight sessions were separately confirmed stopped through the raw API.

<a id="browser-use-checkpoint-evaluation--earlier-preparation-and-canceled-2030-attempt"></a>
### Earlier preparation and canceled 20/30 attempt

The user requested this follow-up on 2026-09-10 after the intermediate checkpoint
evaluations. The initial selection was **after training iteration 30**:
`/gpfs/scrubbed/zixianma/openwebrl-runtime/runs/openwebrl-4b-reference-286382-20260910T164338/iter_0000029`.
It scored **96/300 task successes (32%)**, with 52 invalid attempts, on the existing
GPT-4.1 monitor. After-20 scored 95/300, after-21 89/300, and after-22 86/300.
This is the best observed score among evaluated checkpoints; its one-task lead
over after-20 does not establish a statistically reliable ranking.

The user subsequently requested **both after-20 and after-30** on a new
allocation. After-20 has the best valid-only rate (95/232 = 40.95%); after-30 has
the best all-task rate (96/300 = 32%). The additional checkpoint is
`/gpfs/scrubbed/zixianma/openwebrl-runtime/runs/openwebrl-4b-reference-285546-20260909T235114/iter_0000019`.
The two checkpoints will use the identical 300-task cohort, cloud backend,
proxy policy, judge, and decoding settings. This replaces the earlier single
checkpoint job proposal.

<a id="browser-use-checkpoint-evaluation--prepared-execution-and-budget"></a>
### Prepared execution and budget

The recommended lower-cost template is `scripts/evaluate_browser_use_2gpu.sbatch`:
two sequential 300-task monitors using **2 H200 GPUs, 16 CPUs, 480 GiB RAM, for
3 hours**: **6 GPU-hours**, estimated cluster charge **$5.40**. The earlier
four-GPU template remains available but has not been submitted. Four GPUs were
chosen initially to reuse the verified TP4 loader and four inference workers;
they are not a minimum model-inference requirement. The wrapper now supports TP2
as well as TP4. Fifteen CPU tests and both TP2 launch dry runs pass; actual TP4-to-TP2
checkpoint restoration and paired evaluation duration still require live checks.
Host RAM and CPUs remain conservative because the task/data workload is unchanged. It requires explicit approval before submission. The
previous paired-evaluation job 287370 completed in 1:48:47 and released its GPUs.
Training job 287371 remains independent on g005.

Browser Use hosting and GPT-4.1 judge usage are additional. The provider currently
lists **$0.02 per browser-hour**, minute-rounded with unused time refunded. For
one attempt per checkpoint with 600 sessions total, each capped at 12 minutes,
hosting is at most approximately **$2.40 before refunds**. Actual judge cost depends on tokens.
Residential proxies are explicitly disabled (`proxy_country_code=None`). See
[Browser Use browser API](https://docs.browser-use.com/cloud/api-v2/browsers/create-browser-session).
The cloud browser itself has [stealth enabled by default](https://docs.browser-use.com/cloud/browser/stealth).

The launcher owns its GPU worker and waits after a failure for supervised repair
within the same allocation deadline. It does not submit another allocation or
retry blindly. A new attempt uses a new output directory and W&B identity.

<a id="browser-use-checkpoint-evaluation--protocol-and-source-isolation"></a>
### Protocol and source isolation

The frozen source is runtime `reference-browseruse-eval-20260910`, prepared by
`scripts/prepare_browser_use_evaluation.py` from
`reference-stage1-tp4-eval-cache-20260909`. Only the browser-mode launcher assignment,
Browser Use session lifecycle code, and browser session timeout differ. File
hashes and the parent source are recorded in `reference_manifest.json`. Baseline
training continues using its original source.

The task cohort, model checkpoint, prompts, GPT-4.1 judge, 30-step limit,
deterministic decoding, token limits, and 16 concurrent task limit are preserved.
The evaluation requests zero optimizer updates and validates full checkpoint
restoration and all 300 outcomes. W&B uses
`qcq7i4ug-eval-browseruse-after20-JOB` and `qcq7i4ug-eval-browseruse-after30-JOB`, separate from the training history and the
local-browser evaluation. The manifest identifies the changed browser backend.

The SDK is pinned to **3.11.3** in an isolated runtime import overlay. Its existing
runtime dependencies are retained: httpx 0.28.1, pydantic 2.13.5, idna 3.19. These
last two are newer patch versions than the SDK's exact metadata pins; import,
create/get/stop, CDP, and full adapter tests passed. The shared training
installation was not modified. An unused 2.0.15 overlay was inspected but is not
used because its browser-method names differ.

The requested browser screen is 1280×1000. The live cloud browser reported a
1280×788 CSS viewport at DPR 2 (2560×1576 physical pixels). The existing adapter
realigns coordinate transforms to the actual viewport. This rendering difference
must accompany any score comparison; the experiment changes the browser service
and its rendering environment, not solely a stealth flag.

<a id="browser-use-checkpoint-evaluation--validation-and-lifecycle"></a>
### Validation and lifecycle

Preflight evidence is runtime `evaluations/browser-use-preflight-20260910/`:

- `smoke_report.json`: authenticated CDP connection, Example Domain HTTP 200,
  screenshot capture, and independently confirmed remote stop.
- `adapter_smoke_report.json`: the actual OpenWebRL Browser Use adapter passed
  setup, screenshot capture, viewport alignment, and confirmed-stop receipt.
- `concurrency_report.json`: 16 simultaneously created cloud sessions, all 16
  subsequently confirmed stopped, without API errors.
- `evaluation_plan_after20.json`, `evaluation_plan.json`, and launcher dry runs:
  exact checkpoint indices 19 and 29,
  zero training rollouts, and selected cloud-browser environment.

Fifteen CPU tests cover checkpoint and browser identity, missing/incomplete
results, unexpected optimization, delayed remote shutdown, receipt preservation
on a stop failure, and concurrent initialization cleanup. Shell syntax passed.
Full GPU evaluation remains pending an explicitly approved allocation.

Each evaluation tracks its own remote session IDs under its output's
`browser_sessions/`; confirmed stopped IDs are retained under `stopped/`.
Cleanup runs once per process and cannot sweep another newly created session.
Stop failures retain the active receipt for supervised recovery. Live/CDP URLs
are excluded from shared logs. Creation POSTs disable automatic retries to avoid
creating an untracked duplicate browser after an ambiguous network failure.
The browser timeout is 12 minutes, exceeding the 600-second task deadline.

At evaluation completion or failure, inspect active receipts and confirm those
specific sessions stopped via the SDK before declaring cleanup complete. Never
stop account-wide or unrelated Browser Use sessions. A batch timeout does not
itself prove that the remote browsers have stopped.


<a id="browser-use-checkpoint-evaluation--canceled-attempt-and-current-hold-2026-09-11"></a>
### Canceled attempt and current hold, 2026-09-11

The user approved the two-H200 three-hour pair. **Job 287521** started on g004 at
00:06:50 PDT. The after-20 checkpoint successfully restored from TP4 storage into
TP2, with durable receipt `evaluations/qcq7i4ug-browseruse-287521-after20/checkpoint_restore_evidence.json`.

The provider then returned **HTTP 429: free-plan limit of 10 concurrent sessions**.
The burst preflight had accepted 16 session creations, but sustained evaluation
hit this limit and many tasks aborted before generating a turn. This attempt is
**invalid for score comparison**. Its GPU worker was stopped for diagnosis, then
the user explicitly canceled the entire stealth job to choose a checkpoint after
reviewing evaluations. Slurm reports **CANCELLED after 6:37** (about $0.20 GPU
cost). The second checkpoint never started. **Do not restart stealth evaluation
without a new user request and the necessary compute approval.**

All **16 recorded sessions were independently confirmed stopped**. Evidence:
`evaluations/qcq7i4ug-browseruse-287521-after20/cancellation_session_audit.json`.
The provider reported **$0.01833 browser hosting and $0.07839 proxy charges**,
despite the explicit null proxy configuration. SDK inspection shows null is
preserved in the request body; the unexpected proxy accounting remains
unresolved and must be investigated before a future paid stealth run.
A future attempt also needs a sustained concurrency cap at or below the actual
account limit, with meaningful browser tasks in its preflight. The earlier claim
of a validated 16-session evaluation capacity was too strong.

<!-- document:BROWSER_USE_CHECKPOINT_EVALUATION.md:end -->

---

<!-- document:REWARD_RANK_EVALUATION_QUEUE.md:start -->
<a id="scheduled-eval70-80-results-20260913"></a>
### Scheduled evaluations after 70 and 80

The tables-only overview is [RL_RESULTS.md](RL_RESULTS.md). Completed native
training logs provide two additional full-300 local-browser/GPT-4.1 results:
after-70 has **103 successes, 229 valid, 71 invalid: 34.33% overall and 44.98%
valid-only**; after-80 has **114 successes, 229 valid, 71 invalid: 38.00%
overall and 49.78% valid-only**. After-80 has the highest observed values in
this completed local-browser series. This is a single-run point estimate;
valid cohorts and live website conditions differ between evaluations.

Source records were parsed directly from complete native `eval 69` and
`eval 79` log payloads. They are saved as `iteration_70_scheduled_eval_metrics.json`
in runtime run `openwebrl-4b-reference-290926-20260912T185301` and
`iteration_80_scheduled_eval_metrics.json` in
`openwebrl-4b-reference-293510-20260913T104752`. Both task denominators and
reported rates were checked. Raw turn-weighted reward is not task success.

<a id="reward-rank-after69-result-293585"></a>
### After-69 evaluation result, September 13

Job **293585** completed successfully at **02:22:30 PDT**, after **55m20s**
(approximately **1.84 GPU-hours**). All **300 tasks** completed: **103 successes,
221 valid attempts, 79 invalid attempts**. Overall task success is
**103/300 = 34.33%**; valid-only success is **103/221 = 46.61%**; invalid rate
is **26.33%**. These are the local-browser/GPT-4.1 monitoring results.
The turn-weighted raw reward **0.4253456** is a different metric and must not
be reported as task success.

After-58 scored **109/300 = 36.33% overall** and **109/230 = 47.39% valid-only**.
After-69 therefore has six fewer successes and nine additional invalid
attempts; its higher observed training reward did not produce a higher held-out
score in this evaluation. The valid subsets differ, and one live-web evaluation
does not establish a statistically reliable checkpoint ranking.

The [W&B run](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after69-293585)
is finished. All **41 local metrics match remote history**; the run summary
contains only runtime metadata, so verification used the actual history row.
Evidence: `metrics.json`, `status.json`, `checkpoint_restore_evidence.json` and
`wandb_audit.json` under
`/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/qcq7i4ug-record-293585-after69/`.
The single additional approved evaluation slot is consumed; after-66 and
after-68 remain deferred and no further evaluation job is authorized.

<a id="reward-rank-final90-20260913"></a>
### Prepared best nearby checkpoint evaluation: after-69

Training continuation **293510** is queued. The user selected only the best
of the nearby after-66, after-68 and after-69 candidates: **after-69**, whose
collection 70 reward was **0.5561837455830388**. After-68 scored 0.5443213
and after-66 scored 0.5212766; both are deferred by the user's selection.
The selected checkpoint is `iter_0000068` in
`/gpfs/scrubbed/zixianma/openwebrl-runtime/runs/openwebrl-4b-reference-290926-20260912T185301/`.
Observation N is generated by the policy after N−1, before that collection's
optimizer updates. This is why the highest observation 70 selects after-69.

The selected reward matches its accepted-turn archive exactly (1415 rows).
Source hashes and checkpoint component presence pass CPU plan construction;
the worker will inspect checkpoint metadata and verify real GPU restoration.
Existing verified history and new W&B rows were merged, retaining all 75
observations. New API rows alone omitted older observations and were not used
as a replacement history.

**Approved and submitted as job 293585**, September 13 at 01:26 PDT:
**2 H200, 16 CPUs, 480 GiB, two hours**, maximum **4 GPU-hours**. Slurm confirmed
an estimated **$3.60** plus judge usage. Initial state is `PENDING (Resources)`;
no start estimate is available. This consumes the single additional job
explicitly approved for after-69; it does not renew the earlier four-job cap. Use the existing 300-task Online-Mind2Web local-browser/GPT-4.1
monitoring protocol and report all-task and valid-only success in a separate
W&B run. No stealth browser or official o4-mini benchmark is implied.
The earlier four-job evaluation budget is exhausted. Further reward-ranked
evaluations through iteration 90 require additional bounded approval.

Prepared state and exact worker plan:
`/gpfs/scrubbed/zixianma/openwebrl-runtime/reward_eval_queue_final90_20260913.json`.
This file holds only after-69 as the active candidate; after-66 and after-68
are explicitly deferred. Submission used a checkpoint-specific, single-job
approval receipt rather than the legacy helper's four-job approval contract.
Receipt: `/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/submission-after69-eval-293585.json`.
Batch log (created on startup):
`/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-record-eval-293585.out`.
Results directory:
`/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/qcq7i4ug-record-293585-after69/`.
Expected [W&B evaluation run](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after69-293585)
will appear after worker initialization. Training continuation 293510 remains
pending on `afterany:290926`; this evaluation has no dependency on training.

<a id="reward-rank-evaluation-queue"></a>
## Evaluating checkpoints whose training reward enters the top five

_Source record: `REWARD_RANK_EVALUATION_QUEUE.md`. Dated entries retain their historical context._


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

<a id="reward-rank-evaluation-queue--explicitly-approved-compute-budget"></a>
### Explicitly approved compute budget

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

<a id="reward-rank-evaluation-queue--persistent-queue-and-workflow"></a>
### Persistent queue and workflow

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
waited while the supervised worker completed; the allocation was then released
after the full result and W&B verification. This recovery did not consume another
job from the approved cap. Both 287588 and 287596 have now completed valid
300-task evaluations, leaving two authorized evaluation submissions available.

Job 287596 exposed another cross-node inheritance issue: the observer's
`WANDB_SERVICE` points to a node-local service socket. Both submission and the
worker's `clean_environment()` now discard it while retaining credentials.
The first attempt failed before task evaluation; the supervised repair marker
starts a fresh attempt in the same allocation, with a distinct output/W&B suffix.
Do not copy a W&B service socket between nodes or reuse a failed attempt's score.

<!-- document:REWARD_RANK_EVALUATION_QUEUE.md:end -->

---
