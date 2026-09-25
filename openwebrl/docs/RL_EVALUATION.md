# RL checkpoint evaluation: baseline, Browser Use, and scheduling

Reference-policy checkpoint evaluations, the separate Browser Use protocol, and reward-ranked evaluation scheduling. Preserve each protocol and cohort when comparing results. Allocation and cancellation entries remain dated experiment history.

## Contents

- [Paper protocol and best-checkpoint rerun](#paper-om2w-protocol-20260912)
- [Intermediate baseline checkpoint evaluation](#baseline-checkpoint-evaluation)
- [Browser Use checkpoint evaluations](#browser-use-checkpoint-evaluation)
- [Evaluating checkpoints whose training reward enters the top five](#reward-rank-evaluation-queue)
- [Canonical ARM comparison at rollout iteration 20](#arm-iteration-19-evaluations-20260915)

---

<a id="paper-om2w-protocol-20260912"></a>
## Paper protocol and best-checkpoint rerun (2026-09-12)

The user requested OpenWebRL's default judge/config for RL checkpoint evaluation.
The [paper, Appendix A.6 and B](https://arxiv.org/html/2606.02031v1#A6) explicitly
distinguishes two protocols:

| Setting | Existing training-curve monitor | Paper Online-Mind2Web benchmark |
| --- | --- | --- |
| Judge | GPT-4.1, native action-history prompt | o4-mini, OSU AgentTrek prompt |
| Actor decoding | temperature 0; top-p 1; top-k 1 | temperature 0.6; top-p 0.95; top-k 20 |
| Response/context cap | 4,096 / 32,768 tokens | 4,096 / 32,768 tokens |
| Max turns | 30 | 30 |
| Actor context | one screenshot, full reasoning history and tool feedback | same |
| Judge images | up to three recent screenshots | native AgentTrek implementation uses the final screenshot |
| Browser in our runs | local process; separately labeled Browser Use runs | Browser Use stealth |
| Reporting | task-level overall and valid-only | task-level overall and valid-only, invalid counts |

Training itself uses GPT-4.1 by default. Consequently, our existing deterministic
RL monitor was already consistent with the paper's **training-curve** protocol;
it was not the paper's official OM2W score. The new standalone benchmark uses
the right-hand column. Keep the old monitor series and live baseline unchanged.

Source check: upstream [run_evaluation.sh](https://github.com/OpenWebRL/OpenWebRL/blob/main/scripts/run_evaluation.sh)
chooses the canonical OM2W judge when no environment override is supplied.
However, [run_evaluate.py](https://github.com/OpenWebRL/OpenWebRL/blob/main/openwebrl/run_evaluate.py)
still has greedy CLI defaults and a hard-coded 2,048-token request. Thus simply
running the shell script or changing `JUDGE_MODEL` does not reproduce all paper
settings. The prepared benchmark module pins the paper's actor parameters and
loads the native AgentTrek reward function explicitly.

Native AgentTrek sends full interleaved reasoning/action history plus the final
image, seed 42, and no explicit judge temperature or token cap; its per-call
timeout is 120 seconds with at most four attempts. Actor repetition penalty is
1.0. Browser concurrency is eight, reflecting the verified account limit; the
paper's generic table uses 16. Proxy is disabled, matching our validated Browser
Use source. These operational differences and live-site variability prevent a
claim of bit-for-bit reproduction of the published score.

**Selected checkpoint:** after **58 training iterations**, index **57**,
`/gpfs/scrubbed/zixianma/openwebrl-runtime/runs/openwebrl-4b-reference-288861-20260912T040027/iter_0000057`.
It has the best observed local-browser task scores among the completed same-
protocol checkpoint evaluations: **109/300 = 36.33% overall; 109/230 = 47.39%
valid-only; 70 invalid**. Selection is based on these task scores, not the
turn-weighted legacy metric or peak training reward. Its edge over after-38 is
only two successes. The after-38 Browser Use result (137/300, 137/291 valid-only)
is a separate browser condition and was not pooled into this ranking.

Prepared [rerun manifest](arm_results/rl_integration/after58-benchmark-plan.json),
[benchmark generator](../eval_benchmark.py), and
[batch launcher](../../scripts/evaluate_paper_after58_2gpu.sbatch).
Runtime source is `.../openwebrl-runtime/reference-paper-om2w-20260912`.
It restores the selected distributed checkpoint directly, with zero optimizer
updates, and writes a separate W&B run/group `qcq7i4ug-paper-benchmark` and metric
prefix `eval/online-mind2web-benchmark`. Original checkpoints/results are preserved.
The source manifest records file hashes and the exact configuration.

An integration check prevents a missing AgentTrek verdict from being re-judged
by slime's generic training reward handler: the missing verdict stays in
metadata, all affected turns are marked invalid, and a numeric transport sentinel
prevents automatic second judging. It remains invalid in task statistics.

**Approved and submitted:** job **291005**, **two H200s × two hours**, 16 CPUs,
480 GiB (four GPU-hours, Slurm estimate **$3.60**), plus o4-mini and Browser Use
services. Submitted 2026-09-12 20:02 UTC and started on **g008**. The user reduced
the earlier three-hour proposal to two hours. A previous full-300 Browser
Use/GPT-4.1 run took 1:52:14 on two GPUs, so this budget is plausible but tight;
it is not an o4-mini runtime guarantee. Active baseline job 290926 is separate.
[Submission receipt](arm_results/rl_integration/after58-benchmark-submission.json).

Output: `/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/qcq7i4ug-paper-291005-after58`.
`evaluation.log` records execution; `completed_tasks/*.json` atomically preserves
each finished trajectory's task ID, verdict, invalid flag and response history
before the all-task completion barrier. A timeout therefore retains partial
task results, which must be labeled partial rather than a completed 300-task
score. Full completion also produces the existing rollout recovery archive and
`metrics.json`. W&B run: `qcq7i4ug-benchmark-after58-291005`.

**Startup verified:** TP2 GPU checkpoint restoration at index 57, both inference
workers generating browser actions, and the first native AgentTrek verdict
persisted with `judge_model=o4-mini`. This verifies startup, not completion or a
success-rate estimate. [Startup receipt](arm_results/rl_integration/after58-benchmark-startup.json).

**Allocation ended, partial result:** job 291005 exited after 1:58:44 when its
evaluation subprocess reached the internal timeout (return code 124, Slurm
FAILED). It preserved **299/300** task records: **175 successes, 115 valid
failures, nine unavailable**, and **one task without a saved result**. Seven
unavailable records have `remove_sample=true`; two additional browser-navigation
aborts have no judge verdict despite `remove_sample=false` and are excluded
from the valid denominator. Completed-only success is **175/299 = 58.53%**;
completed-valid-only success is **175/290 = 60.34%**.
Across the scheduled 300 tasks, **175/300 = 58.33% is a lower bound**, not a
completed full-300 score. The all-task finalization barrier did not finish.
The unresolved task is listed in the
[partial-result audit](arm_results/rl_integration/after58-benchmark-partial.json).
The two-hour budget was insufficient for full completion. No additional
allocation or retry was submitted. These benchmark rates must not be pooled
with historical GPT-4.1/greedy monitor scores.

The ARM track objectives and separate mechanics-pilot requests are recorded in
[the integration plan](ARM_INTEGRATION_PLAN.md#arm-rl-exact-losses-20260912).

<!-- document:BASELINE_CHECKPOINT_EVALUATION.md:start -->
<a id="debug-evaluation-wandb-migration-20260913"></a>
## Debug and remaining evaluation W&B migration, September 13

Moved another 19 inactive runs to `zixianma/openwebrl-evals`: six early pipeline
tests, five browser benchmarks/diagnostics, three ARM calibration attempts,
four synthetic GPU diagnostics, and the now-finished checkpoint-80 temperature-0
evaluation. The training project retains baseline `qcq7i4ug`, the real ARM
training pilot `arm-turn-bonus-beta0.5-after70-293194`, and the newly started
from-zero training run `arm-turn-bonus-fresh-294197`. New debug, smoke-test,
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

<a id="stealth80-temperature-completed-20260913"></a>
## Checkpoint 80: completed stealth/o4-mini temperature comparison

Both approved jobs finished all 300 tasks with exit 0. Each restored checkpoint
`iter_0000079`, after training iteration 80. The same Browser Use stealth,
o4-mini/AgentTrek judge, top-p 0.95, top-k 20, 4096 response tokens, 32768 context,
30-turn limit and eight browser sessions were used; temperature varied the actor.

| Actor temperature | Successes | Valid | Invalid | Overall % | Valid-only % | Job | Elapsed | H200-hours |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0 | 166 | 294 | 6 | 55.33 | 56.46 | 294093 | 1:42:32 | 3.4178 |
| 0.6 | 169 | 296 | 4 | 56.33 | 57.09 | 294094 | 1:37:49 | 3.2606 |

Temperature 0.6 added three successes (+1.00 percentage point overall); this
single comparison does not establish an advantage. The earlier local-browser,
GPT-4.1 checkpoint-80 monitor scored 114/300 (38.00%) and 114/229 valid (49.78%);
browser and judge changes prevent interpreting the difference as training gain.
Checkpoint-58's stealth/o4-mini temperature-0.6 result remains partial at
175/299 completed (58.53%) and 175/290 valid (60.34%), with one missing task.
It is numerically higher but is not a completed, same-date checkpoint ranking.

Verified all 39 logged scalars against each W&B history; both runs are finished.
Each result directory holds 300 completed task files, `metrics.json`,
`status.json`, restore evidence and `final_wandb_audit.json`:
`/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/qcq7i4ug-stealth-o4-294093-after80-t0/`
and `qcq7i4ug-stealth-o4-294094-after80-t0.6/` under the same parent.

[Temperature 0 W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-stealth-o4-after80-t0-294093) ·
[Temperature 0.6 W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-stealth-o4-after80-t0.6-294094).

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
<a id="after58-invalid-retry-plan-20260914"></a>
### Completed after-58 invalid/missing retry

Job **295069 completed normally at 22:58:01 PDT, September 13**, exit 0,
after **13m18s / 0.4433 H200-hours** within the approved two-H200, one-hour
allocation (16 CPUs / 480 GiB). It restored `iter_0000057` and retried exactly
nine invalid attempts plus the one missing task, once each, preserving all
290 originally valid attempts, including their failures.

| Result | Completed / planned | Successes | Valid | Invalid | Missing | Success / completed % | Valid-only % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original job 291005 | 299 / 300 | 175 | 290 | 9 | 1 | 58.53 | 60.34 |
| Retry subset job 295069 | 10 / 10 | 3 | 7 | 3 | 0 | 30.00 | 42.86 |
| Original valid + retries | 300 / 300 | 178 | 297 | 3 | 0 | 59.33 | 59.93 |

The retry produced **three successes, four valid failures and three remaining
invalid attempts**. The formerly missing task completed as a valid failure.
Remaining invalids are CVS initial navigation timeout, dblp connection closure,
and task `a48e2f1ee8d87eaeea56fe5e730427e6` timing out at 600 seconds.
There is no repeat-until-success loop or further authorized allocation.

The isolated `reference-after58-invalid-retry-20260914` source changed only the
task subset; Browser Use stealth, o4-mini/AgentTrek judging, actor temperature
0.6 and checkpoint identity are unchanged. The merged result is explicitly
labeled as including invalid/missing retries in [RL_RESULTS.md](RL_RESULTS.md).
It is not a fresh single-pass evaluation and has a different retry policy from
the original after-80/90 rows. The original partial results remain intact.

[W&B retry run](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-after58-invalid-retry-295069)
is finished. All 39 native subset metrics match remote history; the exact ten
retry identities and retained 290 valid original attempts were checked. Merged
metrics are recorded separately under `eval/merged/` in the W&B summary, leaving
the ten-task history unchanged. Evidence under runtime
`evaluations/qcq7i4ug-after58-invalid-retry-295069/`:
`metrics.json`, `merged_metrics.json`, `completed_tasks/`,
`checkpoint_restore_evidence.json`, `final_wandb_audit.json`, and `status.json`.
Controller log: runtime `logs/slurm-after58-retry-295069.out`.

Launcher: `scripts/retry_stealth_after58_2gpu.sbatch`; worker:
`scripts/retry_stealth_invalid.py`. Four selection/merge tests and 15 evaluation
loader tests passed before submission. Runtime provenance remains in
`after58_invalid_retry_plan.json` and `after58_invalid_retry_submission_plan.json`.

<a id="stealth80-gpt41-rejudge-feasibility-20260913"></a>
### Completed after-80 temperature-0 GPT-4.1 rejudging

| Saved trajectories | Successes | Valid | Invalid | Overall % | Valid-only % |
| --- | ---: | ---: | ---: | ---: | ---: |
| After-80, stealth browser, actor temperature 0 | 169 | 294 | 6 | 56.33 | 57.48 |

Rejudged all 300 task records from job 294093 with the frozen baseline
GPT-4.1 `action_history` reward implementation and its three recent screenshots.
The original six invalid attempts remain invalid. There were no additional
judge timeouts or exhausted API retries. The native protocol made 271 API calls;
23 other valid attempts received deterministic failure scores under its status,
format or missing-final-answer rules. **No new GPU inference or Browser Use
sessions** were used. The archive's tensor storage was skipped; screenshot and
response metadata were preserved. `scripts/rejudge_saved_gpt41.py` provides
CPU-only auditing by default, explicit execution and per-task resumable output.

The same trajectories originally scored **166/300 (55.33%)** with o4-mini/AgentTrek.
GPT-4.1 awarded 29 successes where o4-mini did not, and rejected 26 o4-mini
successes: **55/294 disagreements (18.71%)**, a net gain of three successes.
This comparison changes the judge model and judging protocol together, and is
not a second actor rollout or a pure judge-model ablation. The results sheet
labels it as rejudged saved trajectories.

[W&B rejudging run](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-after80-t0-gpt41-rejudge-20260913)
is finished; all 13 final metrics match remote history and all 300 unique task
identities match the original evaluation. The served model was
`gpt-4.1-2025-04-14`; recorded API usage totals **1,298,429 input tokens** and
**96,847 output tokens**. Evidence under runtime
`evaluations/qcq7i4ug-after80-t0-gpt41-rejudge-20260913/`:
`manifest.json`, `metrics.json`, `completed_tasks/`, `api_usage.jsonl`,
`final_wandb_audit.json` and `status.json`. Local log:
`logs/rejudge-after80-gpt41-20260913.log`. Three CPU tests cover denominator
handling, metadata preservation and rejection of unexpected pickle globals.

<a id="stealth90-temperature-plan-20260913"></a>
### Completed after-90 stealth evaluation, temperature 0.6

**Job 294983 completed normally on g013 at 22:01:55 PDT, September 13**,
exit 0, after **2h02m43s / 4.091 H200-hours**, within the approved two-H200,
three-hour allocation (16 CPUs / 480 GiB). It restored checkpoint
`iter_0000089` from job 294421 and executed zero optimizer updates.

| Checkpoint after iteration | Actor temperature | Completed | Successes | Valid | Invalid | Overall % | Valid-only % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 80 | 0.6 | 300 | 169 | 296 | 4 | 56.33 | 57.09 |
| 90 | 0.6 | 300 | 171 | 296 | 4 | 57.00 | 57.77 |

The frozen `reference-paper-om2w-20260912` source uses Browser Use stealth
and o4-mini/AgentTrek judging on all 300 tasks. After-90 has two more successes
than after-80: **+0.67 percentage points overall / +0.68 valid-only**. This small
observed difference does not establish a reliable checkpoint advantage; live-web
conditions and valid task identities can differ.

[W&B run](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-stealth-o4-after90-t0.6-294983)
is finished. All **39 local metrics** match remote history; all 300 unique
per-task records match the final counts. Browser navigation exceptions occurred
within handled task attempts; the overall evaluator and batch both exited 0.
Raw turn-weighted reward **0.46938186** is not the task success rate.

Evidence under runtime `evaluations/qcq7i4ug-stealth-o4-294983-after90-t0.6/`:
`metrics.json`, `completed_tasks/`, `checkpoint_restore_evidence.json`,
`final_wandb_audit.json`, `status.json`, and `launcher_exit.json`.
Controller log: runtime `logs/slurm-stealth-o4-294983.out`.
Submission plans remain `stealth90_t06_prepared_plan.json` and
`stealth90_t06_submission_plan.json`; no further evaluation submission is
covered by this consumed allocation approval.

<a id="scheduled-eval90-results-20260913"></a>
### Scheduled evaluation after 90, September 13

Job **294421 completed normally at 19:22:24 PDT** after **5h10m17s**
(**10.343 H200-hours**, within the approved two-H200/six-hour cap).
Training reached **90 completed iterations / 1,016 Adam updates** and saved
`iter_0000089` before the scheduled full-300 evaluation.

| Checkpoint | Browser | Judge | Actor temperature | Successes | Valid | Invalid | Overall % | Valid-only % |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 90 | Local | GPT-4.1 | 0 | 101 | 222 | 78 | 33.67 | 45.50 |

After-80 scored 38.00% overall and 49.78% valid-only under the same configuration;
after-90 is lower by 4.33 and 4.29 percentage points respectively. Valid cohorts
and live website conditions vary, so this single evaluation does not establish
a reliable checkpoint ranking. After-80 retains the highest observed overall
score in the completed local-browser series.

[W&B qcq7i4ug](https://wandb.ai/zixianma/openwebrl/runs/qcq7i4ug) is **finished**;
all **43** final evaluation metrics match local progress output within its printed
precision. The final `train/reward` observation is **0.5241433**; this is distinct
from evaluation task success. Evidence in runtime run
`openwebrl-4b-reference-294421-20260913T211532`:
`iteration_90_scheduled_eval_metrics.json`, `final90_wandb_audit.json`,
`checkpoint_after90_audit.json`, and `exit_status.json`. Checkpoint metadata,
optimizer/scheduler counters, dataset cursor presence and all shard byte extents
passed; small CPU tensor samples were finite. This audit is not a full GPU reload.

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


<a id="arm-early-295690"></a>
## Early ARM comparison: job 295690 (2026-09-14)

[Numeric results](RL_RESULTS.md) · [Per-task classifications and paired counts](arm_results/rl_integration/threeway-295690.json).
The fixed 100-task native GPT-4.1/action-history evaluation completed for baseline
Adam46 and all-failure Adam42. Baseline scored 24/100 overall and 24/65 valid;
all-failure scored 21/100 overall and 21/70 valid. On 56 common-valid tasks,
baseline scored 21/56 and all-failure 18/56. Only seven common-valid tasks have
discordant success outcomes (five baseline-only, two all-failure-only); exact
two-sided McNemar p=0.4531. The all-task comparison has p=0.5811.
This does not establish a reliable regression or equivalence. Availability is
poor (35 and 30 invalid), and the checkpoints differ by four Adam updates.
All-failure has no demonstrated benefit here, but the evidence is too weak to
claim it is worse. Hold additional all-failure training pending the missing
original-ARM control rather than declare a statistically established failure.

**Launcher incident:** the approved three-GPU allocation ran its one-GPU Slurm
steps sequentially. Baseline ran 16:57:48–18:04:04 PDT; all-failure ran
18:08:20–18:54:17. Original ARM only obtained a step at 18:54:17 and correctly
refused to start with less than 30 minutes remaining. It evaluated zero tasks;
this is missing data, not a zero success rate. Job ended FAILED after 1:57:02,
although the two completed evaluations passed checkpoint GPU restoration,
100-task identity verification and zero-training checks. This wasted reserved
GPU capacity and was not caught during startup monitoring.

The launcher is now changed to one three-task Slurm step with one GPU per task,
so all three workers acquire resources together. Python/shell syntax checks pass;
concurrent GPU execution remains unverified. No replacement allocation submitted.
A missing-control-only retry would preserve both completed evaluations.
Runtime artifacts: `/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/arm-threeway-295690/`.


<a id="arm-early-recovery-295759"></a>
## Early ARM control and invalid-only recovery — job295759

All planned task sets completed and passed per-case checkpoint-restoration,
identity and zero-training checks: original ARM46 100/100, baseline retry35/35,
all-failure retry30/30. Original ARM scored20/100 overall and20/72 valid-only.
The retries recovered seven baseline successes and five all-failure successes;
16/35 and15/30 retries respectively produced valid outcomes. Combining the
single retry only for initially invalid tasks gives baseline31/100 (31/81 valid)
and all-failure26/100 (26/85 valid). Repeated invalids remain invalid.
Original ARM has28 invalid tasks and **has not received a retry**; this recovery
view therefore has unequal retry opportunities and must not be treated as a
fair three-way ranking. Initial-attempt scores are24/20/21 for baseline/original
ARM/all-failure. On49 tasks valid for all three initial attempts, successes
are19/16/16. See [paired counts and per-task classifications](arm_results/rl_integration/threeway-295759-recovery.json).
Timing differences, local-browser invalids and the46/46/42 update near-match
limit causal conclusions. These results do not establish all-failure benefit.

Runtime root: `evaluations/arm-threeway-295759/` under the project runtime;
accepted directories are `arm46`, `baseline46-r3`, and `allfailure42-r3`.
Earlier baseline startup attempts produced zero task results and are excluded.
The repair controller returned0 and resumed the waiting batch controller after
its complete retry queue. Slurm retains FAILED/1:0 (42:52 elapsed) because the
initial rank failed before repair; per-case complete receipts distinguish the
successful scientific evaluations from that controller status.

<a id="arm-iteration-19-evaluations-20260915"></a>
## ARM iteration-19 evaluations — rollout iteration 20 — jobs 296794, 296795, 296816

The three ARM variants were evaluated on the same fixed 100-task cohort with
the local-process browser and GPT-4.1/action-history judge. The comparable
checkpoint is `iter_0000019` for each run: the training runtime stores the
checkpoint after rollout iteration 20 at that zero-based index. The evaluator
was corrected to accept the ARM scheduler offset of zero and to allocate a
job-specific distributed rendezvous port. All three evaluations completed all
100 scheduled tasks with return code zero.

| Run | Overall successes | Valid | Invalid | Overall rate | Valid-only rate | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Original ARM bonus | 26 | 73 | 27 | 26.00% | 35.62% | [296795](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after20-296795) |
| All-failure ARM | 28 | 70 | 30 | 28.00% | 40.00% | [296794](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after20-296794) |
| Additive ARM | 24 | 77 | 23 | 24.00% | 31.17% | [296816](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after20-296816) |

The evaluation artifacts are under
`/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/qcq7i4ug-record-{296794,296795,296816}-after19/`.
These are initial-attempt rates; no invalid-task retries were applied. Browser
navigation and step failures remain represented in each run's invalid/aborted
counts, so valid-only rates should be read alongside the invalid counts.

<a id="baseline-iteration-19-fixed-100-20260915"></a>
## Baseline iteration-19 fixed-100 evaluation (2026-09-15)

The baseline reference checkpoint was regenerated on the same fixed 100-task
cohort used by the three ARM variants. Job297011 completed all100 tasks with
return code0, then its idle allocation was canceled. It scored25/100 overall,
with71 valid and29 invalid tasks, giving35.21% valid-only success. The saved
evaluation artifacts and W&B run are under
`/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/qcq7i4ug-record-297011-after19/`.

### Aggregate comparison

The observed overall differences are small: original ARM is +1 percentage
point versus baseline, all-failure ARM is +3 points, and additive ARM is −1
point. Unpaired two-proportion checks are non-significant (approximate
two-sided p-values 0.87, 0.63, and 0.87 respectively). Valid-only rates are
35.21% baseline, 35.62% original ARM, 40.00% all-failure ARM, and 31.17%
additive ARM; these denominators differ because invalid rates differ. Since
the complete per-task records were not retained, a paired test is unavailable.
These 100-task results therefore show no significant ARM improvement or
regression; a larger common cohort is needed to resolve effects of this size.

<a id="arm-original-bonus-iteration-30-fixed-100-20260916"></a>
## Original ARM bonus iteration-30 evaluation (2026-09-16)

The original ARM-bonus checkpoint `iter_0000029` (the checkpoint after training
iteration 30) was evaluated on the fixed 100-task cohort with the local browser
and GPT-4.1/action-history judge. The corrected retry produced **27/100
successes**, **74 valid**, **26 invalid**, **27.00% overall**, and **36.49%
valid-only**. A first attempt on the same checkpoint produced 31/100; both
attempts are retained because live websites and browser availability vary even
at temperature 0. The corrected retry is the primary reported result.

Artifacts: `/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/qcq7i4ug-record-297412-after30-retry5/`.

<a id="all-failure-arm-full-300-task-evaluation-20260915"></a>
## All-failure ARM full-300-task evaluation (2026-09-15)

The all-failure ARM checkpoint `iter_0000019` was evaluated on the 200-task
complement of the fixed 100-task cohort. The complement produced 62 successes,
152 valid trajectories, and 48 invalid trajectories: **31.00% overall** and
**40.79% valid-only**. Combining those disjoint results with the earlier
all-failure 100-task result (28/100 successes, 70 valid, 30 invalid) gives
**90/300 = 30.00% overall**, **222 valid**, **78 invalid**, and
**90/222 = 40.54% valid-only**.

The completed metrics are in
`/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/qcq7i4ug-record-297227-after19-retry2/runtime/progress.log`;
the W&B run is [297227-r2](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after20-297227-r2).
The wrapper exited nonzero only because its legacy finalization guard expected
300 tasks in this complement-only invocation; the evaluator emitted complete
200-task metrics before that bookkeeping check. No task was duplicated between
the complement and fixed-100 cohorts.

<a id="arm-iter70-audit-20260919"></a>

## ARM iteration-70 results and iteration-80 availability — 2026-09-19

All three iteration-70 jobs completed 300 tasks under the deterministic local
browser / GPT-4.1 Online-Mind2Web monitor. Use the task-level success metrics,
not the turn-weighted `raw_reward_mean`. Original bonus (306478) has 103 successes,
231 valid, 69 invalid: **34.33% overall / 44.59% valid-only**. All-failure (306477)
has 106 successes, 233 valid, 67 invalid: **35.33% / 45.49%**. Additive (307120)
has 112 successes, 222 valid, 78 invalid: **37.33% / 50.45%**.
The historical outcome-only iteration-70 reference is **34.33% / 44.98%**
(103 successes, 229 valid). Additive leads this comparison, but different dates
and validity sets prevent interpreting these aggregates as a controlled or
statistically established gain. Numeric records are in [RL_RESULTS.md](RL_RESULTS.md)
and the [verified metric-source audit](arm_results/rl_integration/iteration70-audit.json).

No completed or queued ARM iteration-80 evaluation was found in the September 19
check. Original and additive have durable checkpoints past completed iteration
80. All-failure stopped at completed iteration 78 because the next collection's
96 labels missed the 100-label guard; [continuation preparation](ARM_INTEGRATION_PLAN.md#arm-allfailure-to100-prepared-20260919)
records the proposed recovery. The historical baseline iteration-80 full-300
result is **38.00% overall / 49.78% valid-only**.

Rollout preservation limitation: these three evaluators used older frozen
sources. Their status files report **zero task-addressable rollout files**, but
each retained a native aggregate `runtime/rollout_recovery/eval_0.pt` archive:
85.8 GB for all-failure, 88.0 GB for original, and 96.5 GB for additive. A bounded
ZIP-header inspection verified the archive members, not their full payloads.
The pickle metadata alone is 1.36–1.49 GB, above the 128 MiB login-node audit
limit, so per-task verdict extraction was not attempted there. These aggregate
archives must be preserved; per-task recovery/rejudging is not yet verified.
Before the next evaluation, verify that its **actual frozen source** writes
task-addressable rollouts; setting the destination environment variable alone
does not establish that the source implements the saver.

<a id="arm-iter80-launch-20260919"></a>

## ARM iteration-80 full-300 evaluations — approved and submitted 2026-09-19

The user approved the iteration-80 evaluations. Use the established per-evaluation
profile: **2 H200 × 2 hours, 16 CPUs, 480 GiB RAM** per job, 12 GPU-hours total
across three jobs. Each allocation exits when its evaluation finishes. Slurm
estimated $3.60 per job; GPT-4.1 judging is additional.

September 20 update: **all three jobs completed successfully and released their
allocations**. The checkpoint watcher released all-failure 309687 after verifying
iteration 80; its full-300 evaluation completed in 33m22s.

| Method | Evaluation job | Training root under runtime `evaluations/` | Iteration-80 Adam updates | Status |
| --- | --- | --- | ---: | --- |
| Original bonus | 309685 | `arm-turn-bonus-fresh-303573` | 950 | complete; all 300 tasks saved |
| Additive bonus | 309686 | `arm-failure-additive-303574` | 1,042 | complete; all 300 tasks saved |
| All-failure bonus | 309687 | `arm-turn-bonus-fresh-allfailure-309490` | 1,038 | complete; all 300 tasks saved |

All load **`runtime/iter_0000079`**, meaning 80 completed training iterations.
The protocol stays actor-only, local browser, GPT-4.1/action-history, temperature
0, one trajectory per task, all 300 tasks, 30 browser turns, 4,096 response-token
limit. W&B project is `openwebrl-evals`. The same frozen task-file hash is verified
across all three sources. Historical outcome-only iteration 80 remains the
**38.00% overall / 49.78% valid-only** reference; it is not a same-day control.

Original/additive checkpoints already have native completion and counter
validation. All-failure training job 309490 is now running from completed
iteration 78. Its evaluation job is submitted with a user hold, so it allocates
no GPUs while waiting. A persistent, lightweight login-host watcher checks every
five minutes for the iteration-80 completion/counter receipts and checkpoint
shard sizes, then releases **only job 309687**. It does not wait for training to
finish iteration 100 and cannot submit additional jobs. If training stops before
80, the evaluation stays held for supervision. Watcher:
`scripts/watch_arm_iteration80_checkpoint.py`.

The three isolated `reference-arm-eval80-{original,additive,allfailure}-20260919-v1`
sources preserve the corresponding iteration-70 evaluator code except for the
task persistence hook. Each attempt writes a lossless `rollouts/<task-hash>.pt`
and a small `.json` sidecar containing the task ID, native validity/success
metrics, terminal status, and judge verdict metadata. Aborted/empty attempts
are recorded as well. A judge exception preserves the completed browser turns
before propagating. Checkpoint-level completion requires all 300 distinct task
IDs and both artifact types, in addition to GPU checkpoint-restore evidence.
The aggregate native recovery archive is also retained.

**22 CPU tests passed against the actual frozen source**, including reload of
saved image tensors after temporary mappings were removed, invalid/empty-task
persistence, judge-exception recovery, and exact-cohort finalization. Resolved
launch commands were checked for TP2, zero optimizer rollouts, full-300 monitor
config, and `openwebrl-evals` tracking. These checks do not claim GPU execution
before Slurm starts the jobs.

Launcher: `scripts/evaluate_arm_iteration80_2gpu.sbatch`, with controller
`scripts/run_arm_iteration80_eval.py`. Outputs:
`evaluations/arm-{original,additive,allfailure}-iter80-JOB/`. Readiness hashes,
source/cohort audits, submitted commands and receipts, and watcher status are
under runtime `arm-turn-bonus-preparation/iteration80-evals-20260919/`.

Verified original-bonus result: **100/300 successes, 222 valid, 78 invalid:
33.33% overall / 45.05% valid-only**. Its historical outcome-only reference is
38.00% / 49.78%, so this endpoint does not show an improvement. This remains a
different-date comparison, not a paired significance result. The
[iteration-80 audit](arm_results/rl_integration/iteration80-audit.json) records
metric hashes, checkpoint identity and artifact checks. All 300 unique task IDs
match the planned cohort; per-task verdict sums exactly reproduce the native
aggregate metrics, with 300 corresponding rollout archives and no persistence
exceptions. One real trajectory was additionally reloaded on the compute node:
its image tensors, three judge screenshots, action history and judge metadata
were intact. This check made no judge API call and used the existing allocation.

Additive iteration 80 also completed: **111/300 successes, 212 valid, 88 invalid:
37.00% overall / 52.36% valid-only**. Its 300 unique task IDs and per-task verdict
sums match the native metrics, with all 300 rollout archives present and no
persistence exceptions; provenance is in the same iteration-80 audit. Relative
to historical outcome-only iteration 80, overall success is 1.00 percentage
point lower and valid-only is 2.58 points higher. Because validity sets differ,
the latter is not evidence of an unconditional performance gain.

All-failure iteration 80 completed with **92/300 successes, 218 valid, 82 invalid:
30.67% overall / 42.20% valid-only**. All 300 unique task IDs match the cohort,
all 300 rollout archives and verdict sidecars exist, and their sums reproduce
the native metrics. This is down from iteration 70's 35.33% / 45.49%, and below
the historical iteration-80 baseline by 7.33 / 7.58 percentage points. Different
dates and validity sets limit causal interpretation. The same
[iteration-80 audit](arm_results/rl_integration/iteration80-audit.json) now includes
all three variants.

<a id="arm-iter90-readiness-20260920"></a>

### Iteration-90 availability — September 20, 16:33 UTC

No ARM iteration-90 evaluation is completed, running, or queued. All-failure job
309490 has saved `iter_0000089` with 1,142 Adam updates; checkpoint receipts,
shard sizes, scheduler agreement, and cursor presence were verified. It is
collecting iteration 91 toward target 100. Original's latest durable checkpoint
is 85. Additive job 311202 saved 85, then failed collecting 86 because g008's
local temporary storage filled. Neither has an iteration-90 checkpoint yet.
The historical outcome-only iteration-90 reference is **33.67% / 45.50%**.

September 20, 22:35 UTC update: additive **311962** has now completed iteration
90 with **1,150 Adam updates**. All-failure's iteration-90 checkpoint has
**1,142 updates**. Both checkpoint identities, saved receipts, shard sizes and
cursor presence pass the preparation checks. Original bonus remains at 85.
No iteration-90 evaluation is running or queued.

The user requested these evaluations now. Two full-300 evaluations are prepared
using the same frozen evaluators as iteration 80; only the selected checkpoint
and identifying labels change. The launcher now accepts `--completed-iteration`
and `--training-root`, retaining iteration-80 defaults for its existing watcher.
Resolved commands verify TP2, zero optimizer rollouts, checkpoint index 89,
GPT-4.1/action-history and `openwebrl-evals`. The 300-task cohort hashes match,
and completion requires all task-level rollout archives and verdict sidecars.
Actual GPU restoration remains a startup check.

Proposed allocation: **two jobs, each 2 H200 × 1 hour, 16 CPUs, 480 GiB RAM**,
**4 GPU-hours total**, with the existing GPT-4.1 judging protocol. The three
iteration-80 evaluations took 33–37 minutes each, supporting the shorter
one-hour budget. Exact allocation approval is still required under the root
AGENTS.md; no new submission has been made. Ready-to-submit commands, CPU
checks and file hashes are under runtime
`arm-turn-bonus-preparation/iteration90-evals/`.

**Approved and submitted September 20, 22:40 UTC:** additive **313187** and
all-failure **313188**, with the exact profile above (4 GPU-hours maximum in
total; Slurm estimate $1.80 each, judging additional). Each batch controller
owns and awaits its evaluation worker and releases the allocation on exit.
Both are included in persistent supervision. Approval, submission receipts and
plans with their assigned job IDs are in the preparation directory above.
Outputs are `evaluations/arm-additive-iter90-313187/` and
`evaluations/arm-allfailure-iter90-313188/`, including `rollouts/` for the per-task
archives/verdicts. Initial state is queued; this is not a GPU-restoration claim.

<a id="arm-iter90-results-20260921"></a>

### Iteration-90 results — verified September 21, 00:34 UTC

All-failure **313188** completed in **33m50s**, exit 0, after verified GPU restore
of `iter_0000089` (1,142 Adam updates). Its 300 distinct saved verdict IDs match
the frozen cohort; all 300 nonempty rollout archives are present. Per-task
counts agree with the final metrics: **101 successes, 217 valid, 83 invalid**,
or **33.67% overall / 46.54% valid-only**. Protocol: local browser,
GPT-4.1/action-history, temperature 0, 30-turn evaluation horizon. Use these
task-level rates; the turn-weighted raw reward mean is not the success rate.

Historical outcome-only iteration 90 is **101 successes / 300**, **222 valid**:
**33.67% / 45.50%**. Overall success therefore matches, while the valid-only
denominator differs. The runs are from different dates, so this is descriptive,
not a paired or same-day estimate of an ARM improvement.

[Metric audit](arm_results/rl_integration/iteration90-audit.json) ·
[W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after90-313188) ·
[Metrics](/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/arm-allfailure-iter90-313188/metrics.json) ·
[Rollouts and verdicts](/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/arm-allfailure-iter90-313188/rollouts).
The audit reads small verdicts and archive metadata; it does not load all image
tensors on the login node.

Additive **313187** failed in 1m27s before collecting any tasks: concurrent
evaluators on g021 chose the same native model-server ports. Prepared v2 sources
and controllers now lease separate node-local port blocks; the fix passed CPU
tests and the rescue pilot's GPU startup. No replacement additive allocation
has been submitted. Its iteration-90 checkpoint remains available.

**September 21, 00:53 UTC:** the user requested the additive evaluation retry.
**313408** is submitted with the established **2 H200 × 1h, 16 CPUs / 480 GiB**
profile (Slurm estimate $1.80), initially pending priority. It loads the same
iteration-90 checkpoint / 1,150 Adam updates and evaluates all 300 tasks because
313187 saved none. Frozen source `reference-arm-eval80-additive-20260920-v2`
adds port isolation without changing the policy, cohort, judge or temperature.
Output is `evaluations/arm-additive-iter90-313408/`; per-task rollout archives and
verdicts remain required. Receipts and the resolved plan are under
`arm-turn-bonus-preparation/iteration90-evals/additive-retry-v2-*`.

**Retry completed; audited September 21, 02:21 UTC:** additive **313408**
finished in **37m53s**, exit 0, releasing its remaining allocation. It restored
iteration 90 / 1,150 Adam updates and completed the full cohort: **118 successes,
216 valid, 84 invalid**, **39.33% overall / 54.63% valid-only**. All 300 unique
task IDs match the declared cohort; all 300 nonempty trajectory archives and
verdict sidecars exist. Per-task totals match reported metrics, and all sidecars
identify GPT-4.1 / action-history. No tensor storage was reloaded on the login
node. These task-level rates differ from the log's turn-weighted reward mean.

Relative to historical outcome-only iteration 90 (**101/300; 222 valid**),
additive is +5.67 percentage points overall and +9.13 valid-only. Different
dates and valid subsets prevent a controlled improvement/significance claim.
All-failure iteration 90 remains 101/300. The comparison plot now includes the
additive iteration-90 point with sufficient vertical headroom.
[Updated audit](arm_results/rl_integration/iteration90-audit.json) ·
[W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after90-313408) ·
[Rollouts and verdicts](/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/arm-additive-iter90-313408/rollouts).

### B/C iteration-20 full-300 evaluations — preparation requested September 20

The user requested both evaluations. B is currently before iteration 10 and C
has completed 10; no iteration-20 checkpoint exists. The
[training/evaluation handoff plan](ARM_INTEGRATION_PLAN.md#arm-bc-iter20-prepared-20260920)
prepares two held 2-H200 × 1h evaluations, released only after the corresponding
iteration-20 checkpoint passes identity, counter and shard checks. No eval job
has been submitted pending approval of the required continuations and exact
allocation budgets. Evaluation protocol/cohort and task persistence match the
original ARM full-300 series. The historical additive/outcome-only iteration-20
rows remain comparison references; different-date evaluation is not a randomized
or same-day control.

September 20, 22:55 UTC: exact budgets are now approved and submitted. B
continuation **313208** follows job 311964; C continuation is **313210**. Held
full-300 evaluations **313209** (B) and **313211** (C), each 2 H200 × 1h, are
owned by checkpoint watchers that release them only after a validated
`iter_0000019` save. They consume no GPU allocation while held. All four jobs
are included in persistent supervision.

<a id="arm-additive-iter20-full300-20260920"></a>

## Additive iteration-20 full-300 result recovered into the docs — 2026-09-20

Job **307429** completed on September 19 in 41m17s with exit code zero, loading
`arm-failure-additive-295834/runtime/iter_0000019` on GPU. Its result was omitted
from the summary and plot: **85 successes, 236 valid, 64 invalid; 28.33% overall /
36.02% valid-only** across all 300 tasks. The older fixed-100 result remains
24.00% / 31.17%; these are separate evaluations, not a 100+200 merge.

Protocol: local browser, GPT-4.1/action-history, temperature 0, 30 turns,
4,096 response tokens, one trajectory per task. The frozen cohort has 300 unique
task IDs. Native final-log metrics exactly match `metrics.json`, and the GPU
restore receipt names checkpoint index 19. See the
[source audit](arm_results/rl_integration/additive-iteration20-audit.json) and
[W&B run](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after20-307429).

This older evaluator wrote **zero task-addressable rollout files** but retained
the 130.63 GB aggregate `runtime/rollout_recovery/eval_0.pt` archive. Its ZIP
directory is intact; large metadata/tensor payloads were not loaded on the login
node, so per-task rejudging recovery remains unverified. The iteration-80 saver
fix does not retroactively establish per-task artifacts for this evaluation.

<a id="baseline-iteration90-fixed100-recovery-20260921"></a>
### Outcome-only iteration 90: recovered fixed100 verdicts, September 21

For the ARM-only execution audit **314664**, extracted the original fixed100
IDs from the existing iteration90 `rollout_recovery/eval_89.pt` archive. This
uses saved GPT-4.1/action-history verdicts, not new browser executions or
rejudging. Overall **35.00%**, valid-only **51.47%**: 35 successes, 68 valid,
32 invalid. The full archived cohort reproduces the published 101/300 successes,
222 valid; this validates the task-level grouping and denominator rules.
Metadata extraction skipped tensor storage and discarded large image strings,
with a 90 CPU-second / 3 GiB cap. No GPU or API calls.

Source: runtime `runs/openwebrl-4b-reference-294421-20260913T211532/rollout_recovery/eval_89.pt`.
The exact task verdicts, extraction script, and all historical control references
are saved under runtime `arm-turn-bonus-preparation/task-success-audit-20260921/`.
Historical actor decoding was T=0, 4,096 response tokens, 30 turns. The new ARM
system uses K=5 and T=0.8; different dates/decoding preclude a selector-only causal
claim. See [audit protocol](ARM_INTEGRATION_PLAN.md#arm-task-success-audit-20260921).

<a id="arm-gate-c-iter20-results-20260921"></a>
### C iteration 20 completed; B released and queued — September 21

Both B and C saved iteration 20 / 284 Adam updates. C evaluation **313211**
completed in 34m53s: **110/300 successes, 247 valid, 53 invalid**, giving
**36.67% overall / 44.53% valid-only**. Its exact fixed100 slice is **38/100**,
79 valid and 21 invalid: **38.00% / 48.10%**. B evaluation **313209** is queued
for priority after its checkpoint watcher validated and released it.

C is the additive training recipe with at least two distinct valid candidate
actions and action-equivalence credit; evaluation uses the learned actor alone.
Local browser, GPT-4.1/action-history, T=0, 30 turns, 4,096 response tokens.
All 300 unique task IDs match the full cohort; all have nonempty rollout archives
and JSON verdicts with the expected judge. Checkpoint index19 restored from
`arm-failure-additive-313210/runtime/iter_0000019`.

Historical outcome-only iteration20 is 31.67% / 40.95%; historical additive20 is
28.33% / 36.02%. C's +5.00 pp overall versus baseline is descriptive: dates and
available task sets differ, so this is not a controlled improvement estimate.
Do not substitute the turn-weighted log mean (31.68%) for task success (36.67%).

[Result audit](arm_results/rl_integration/gate-c-iteration20-audit.json) ·
[W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-gate-c-iter20-313211) ·
[Saved rollouts and verdicts](/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/arm-gate-c-iter20-313211/rollouts).


<a id="arm-gate-b-iter20-results-20260921"></a>
### B iteration20 completed — September 21

Job **313209** finished in 34m26s. Full300: **101 successes, 228 valid,
72 invalid; 33.67% overall / 44.30% valid-only**. Its fixed100 slice is
**29 successes, 69 valid, 31 invalid; 29.00% / 42.03%**. All 300 unique
cohort task IDs, nonempty rollout archives, and GPT-4.1/action-history sidecars
are verified; evaluation uses T=0 and the actor alone. B relaxed the valid
candidate gate to at least two distinct actions, retaining response-index credit.

C has 110/300 successes and 247 valid, versus B's 101 and 228. The 199 tasks
valid in both runs contain **89 B successes and 97 C successes**. This is a
useful descriptive paired subset, not proof of training-method superiority:
there is one training seed per variant, differing exposure and rollout noise,
and common-valid selection excludes 101 tasks. B/C both have 284 Adam updates
at iteration20; the historical outcome-only control has 270.

[Result audit](arm_results/rl_integration/gate-b-iteration20-audit.json) ·
[W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-gate-b-iter20-313209) ·
[Rollouts](/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/arm-gate-b-iter20-313209/rollouts).

<a id="arm-additive-iter100-results-20260921"></a>
### Additive iteration 100 completed — September 21

Job **313669** completed additive training through **100 iterations / 1,262
Adam updates**, then evaluated the verified `iter_0000099` checkpoint on all
300 Online-Mind2Web tasks in the same allocation. Slurm completed at 15:34 PDT
with exit code zero after 8h41m17s, releasing the four H200s early.

| Cohort | Successes | Valid | Invalid | Overall | Valid-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full300 | 109 | 217 | 83 | 36.33% | 50.23% |
| Fixed100 IDs, sliced from full300 | 31 | 65 | 35 | 31.00% | 47.69% |

The evaluator used the local browser, GPT-4.1/action-history terminal judge,
T=0, 30 turns and 4,096 response tokens, with no inference-time ARM selector.
All 300 expected unique task IDs match the declared cohort; all nonempty `.pt`
rollouts and per-task JSON verdicts are present, with zero exception records.
Invalid attempts remain in the overall denominator. The fixed100 row is a
subset of this evaluation, not an independent rerun.

The generic turn-weighted reward metric is 40.85%; it is **not task success**.
Task-level rates above are computed from saved verdicts. The GPU restore receipt
confirms the additive checkpoint even though the inherited evaluation W&B ID
starts with `qcq7i4ug`.

Compared with additive90 (118 successes, 216 valid), overall drops **3.00 pp**
and valid-only drops **4.40 pp**. This is checkpoint/evaluation variability,
not evidence by itself of statistically established deterioration. The pending
baseline100 job **315098** is needed for the same-iteration control. All-failure
has trained to100 but has no iteration100 evaluation yet.

[Audited aggregate](arm_results/rl_integration/additive-iteration100-audit.json) ·
[W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after100-313669) ·
[Saved rollouts and verdicts](/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/arm-additive-iter100-313669/rollouts).

<a id="arm-allfailure-iter100-results-20260921"></a>
### All-failure iteration100 — completed September 21

Job **316392** completed in **30m58s** on two H200s, releasing the one-hour
allocation early. The checkpoint is `iter_0000099` from all-failure309490,
**100 completed collections / 1,242 Adam updates**. Full300: **107 successes,
222 valid, 78 invalid; 35.67% overall / 48.20% valid-only**. The frozen fixed100
slice has **26 successes / 66 valid / 34 invalid**, or **26.00% / 39.39%**.

The 300 unique task IDs match the frozen cohort; every task has a nonempty
rollout archive and verdict sidecar, with no persistence errors. Per-task sums
match native task metrics; GPU restoration evidence confirms checkpoint99.
Protocol: local browser, GPT-4.1/action_history, temperature0, same full300
monitor as other RL endpoints. [Machine-readable audit](arm_results/rl_integration/allfailure-iteration100-audit.json).

Compared with all-failure90, overall increases **2.00 pp** and valid-only
**1.65 pp**. Additive100 is **36.33% / 50.23%**, higher by **0.67 / 2.03 pp**.
These are descriptive comparisons across different-date rollouts and valid
sets, not evidence of significance. Baseline100 was pending at this September21
record; its [September24 completion](#baseline-iter100-results-20260924) now
provides the same-iteration outcome-only result. All archives are under
`evaluations/arm-allfailure-iter100-316392/rollouts/`; training was not updated.

<a id="baseline-iter100-results-20260924"></a>
### Outcome-only baseline iteration 100 completed — September 24

Job **318933** resumed the outcome-only lineage at iteration90 and completed
**100 iterations / 1,118 Adam updates**, preserving optimizer, scheduler and
task cursor. Its scheduled full-300 evaluation used the verified
`iter_0000099` checkpoint. The four-H200 allocation completed at **10:29 PDT**
with exit code zero after **9h07m49s**, releasing the remaining allocation.

| Cohort | Successes | Valid | Invalid | Overall | Valid-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full300 | 104 | 227 | 73 | 34.67% | 45.81% |
| Fixed100 IDs, sliced from full300 | 37 | 68 | 32 | 37.00% | 54.41% |

Protocol: local browser, GPT-4.1 `action_history` judge, temperature 0, maximum
30 turns and 4,096 response tokens; actor-only inference. All 300 unique task
IDs match the frozen cohort, and all 300 nonempty rollout archives and verdict
sidecars are saved. There are zero exception records. The fixed100 is the
original frozen subset sliced from this evaluation, not a separate rerun.

| Iteration100 method | Full300 overall / valid-only | Overall delta vs baseline | Fixed100 overall / valid-only | Adam updates |
| --- | ---: | ---: | ---: | ---: |
| Outcome-only baseline | 34.67% / 45.81% | — | 37.00% / 54.41% | 1,118 |
| All-failure ARM | 35.67% / 48.20% | +1.00 pp | 26.00% / 39.39% | 1,242 |
| Additive ARM | 36.33% / 50.23% | +1.67 pp | 31.00% / 47.69% | 1,262 |

These checkpoints align by training iteration, not optimizer-update count.
The ARM evaluations occurred on September21 and the baseline on September24;
valid-task sets differ. The small full300 gains and opposite fixed100 ordering
do not establish a consistent or controlled improvement. Per-task records allow
later paired analysis, subject to those live-web/date limitations. The combined
summary plot now includes iteration100 for all three methods.

[Machine-readable audit](arm_results/rl_integration/baseline-iteration100-audit.json) ·
[Training and scheduled-evaluation W&B](https://wandb.ai/zixianma/openwebrl/runs/qcq7i4ug) ·
[Saved rollouts and verdicts](/gpfs/scrubbed/zixianma/openwebrl-runtime/runs/openwebrl-4b-reference-318933-20260924T082345/evaluation/after100/rollouts).

<a id="arm-gate-b-iter30-40-results-20260924"></a>
### Gate B iteration 30 and 40 evaluations completed — September 24

Allocation **318934** completed B training to iteration 40, then evaluated the
durable iteration-30 and iteration-40 checkpoints in sequence before resuming
training toward 60. B uses the relaxed minimum-two-distinct-actions gate with
unchanged response-index credit. Evaluation is actor-only, local browser,
GPT-4.1 `action_history`, temperature 0, with the same frozen full-300 cohort.

| Iteration | Cohort | Successes | Valid | Invalid | Overall | Valid-only | Adam updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | Full300 | 102 | 234 | 66 | 34.00% | 43.59% | 412 |
| 30 | Fixed100 slice | 29 | 74 | 26 | 29.00% | 39.19% | 412 |
| 40 | Full300 | 108 | 228 | 72 | 36.00% | 47.37% | 542 |
| 40 | Fixed100 slice | 38 | 72 | 28 | 38.00% | 52.78% | 542 |

GPU restoration receipts identify native checkpoints `iter_0000029` and
`iter_0000039`. Each evaluation has all 300 expected unique task IDs, nonempty
rollout archives and per-task verdicts, with zero exception records. Fixed100
rows are slices of these evaluations, not additional independent trials.

Historical outcome-only overall rates are 32.00% at30 and 33.33% at40, so B is
**+2.00 / +2.67 pp** respectively. This is a descriptive comparison across
different evaluation dates and valid-task sets, not a controlled significance
claim. Iteration40 is also 2.00 pp above additive40's historical overall rate.

[Iteration30 audit](arm_results/rl_integration/gate-b-iteration30-audit.json) ·
[Iteration40 audit](arm_results/rl_integration/gate-b-iteration40-audit.json) ·
[Iteration30 W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-gate-b-iter30-318934) ·
[Iteration40 W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-gate-b-iter40-318934).

Saved trajectories and verdicts are under runtime
`evaluations/arm-gate-b-iter30-318934/rollouts/` and
`evaluations/arm-gate-b-iter40-318934/rollouts/`.

<a id="arm-gate-c-iter30-40-results-20260924"></a>
### Gate C iteration 30 and 40 evaluations completed — September 24

Allocation **318935** evaluated the durable iteration-30 and iteration-40
checkpoints, then resumed training toward60. C uses B's relaxed gate with
action-equivalence credit. Evaluation is actor-only, local browser, GPT-4.1
`action_history`, temperature0, on the same frozen full-300 cohort.

| Iteration | Cohort | Successes | Valid | Invalid | Overall | Valid-only | Adam updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | Full300 | 92 | 232 | 68 | 30.67% | 39.66% | 420 |
| 30 | Fixed100 slice | 28 | 72 | 28 | 28.00% | 38.89% | 420 |
| 40 | Full300 | 100 | 222 | 78 | 33.33% | 45.05% | 548 |
| 40 | Fixed100 slice | 31 | 72 | 28 | 31.00% | 43.06% | 548 |

GPU restoration receipts identify native checkpoints `iter_0000029` and
`iter_0000039`. Both evaluations have exactly the300 expected unique task IDs,
nonempty trajectory archives and per-task verdict records; there are zero
exception records. Invalid browser trajectories are included in the overall
denominator and excluded from valid-only. Fixed100 is a slice of each full300
evaluation, not a separate trial.

Against the historical outcome-only overall rates of32.00% and33.33%, C is
−1.33pp at30 and tied at40. Against B's same-day evaluations, C is −3.33pp and
−2.67pp respectively. Dates, browser availability and valid-task sets limit
causal interpretation; these are descriptive differences, not significance
claims. The iteration20 advantage has not persisted through30/40.

[Iteration30 audit](arm_results/rl_integration/gate-c-iteration30-audit.json) ·
[Iteration40 audit](arm_results/rl_integration/gate-c-iteration40-audit.json) ·
[Iteration30 W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-gate-c-iter30-318935) ·
[Iteration40 W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-gate-c-iter40-318935).

Saved trajectories and verdicts are under runtime
`evaluations/arm-gate-c-iter30-318935/rollouts/` and
`evaluations/arm-gate-c-iter40-318935/rollouts/`.
