# ARM matched retry status and results

Status: **Held for a cohort decision after SelectionARM completes. No retries have started.** On 2026-09-08 the user clarified that the retry queries should be reconsidered using all three completed runs. The automatic two-arm continuation is disabled (`authorized: false`); the original report can finish normally without launching retries.

## Decision after completed SelectionARM

**SelectionARM is now complete: 44 unavailable outcomes. The unavailable-task union across all three arms is 69 tasks**, including 13 SelectionARM cases outside the old 56-task two-arm union. A matched three-arm pass would require 207 task attempts. This is a prepared option, not an authorized launch; C2 currently has execution priority. Review originating failure categories before choosing a retry cohort. Decide which arms to compare, the common task cohort, and the one-attempt budget before any follow-up outcomes are observed. For a three-arm robustness comparison, use a matched cohort for all three arms. Distinguish transient browser/network failures from deterministic context-limit failures; changing context handling requires a separately labeled protocol.

The earlier baseline/ScalarRM proposal remains an inventory, **not the final scope**:

- 56 distinct tasks: union of 33 unavailable baseline and 49 unavailable ScalarRM evaluations, with 26 shared.
- One baseline and one ScalarRM attempt on every task would cost 112 task attempts, including previously valid counterparts.
- The completed SelectionARM set expands the three-arm union to 69 tasks. [Separate three-arm inventory](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/dedicated-282209-20260908T005547Z/retry-all-three-proposal.json).

## Reporting rules retained

Preserve the original results. Store retry outcomes, rollouts, action traces, logs, and summaries separately; verify original result hashes. Keep models, seeds, decoding, context, horizon, judge, and timeouts fixed for an ordinary retry. Freeze a single-attempt policy, with no best-of-retries selection.

Report successes divided by all tasks in the agreed retry cohort, successes divided by valid retry outcomes, recovery of initially unavailable tasks, regression of previously valid counterparts, and paired outcomes on common valid tasks. These selected-cohort rates do not replace the original 300-task benchmark rates.

| Item | Current state |
| --- | --- |
| Final retry arms and task count | Pending decision; all three original evaluations are now complete |
| Retry attempts started / completed | 0 / 0 |
| Overall / valid-only retry success | Not available; no retry outcomes |
| Earlier two-arm inventory | 56 tasks per arm; held |

Original completed results remain **30.0% overall / 33.7% valid-only** for baseline and **38.0% overall / 45.4% valid-only** for ScalarRM; see [ARM_INFERENCE_RESULTS.md](ARM_INFERENCE_RESULTS.md).

## Prepared implementation and artifacts

- [Reserved retry directory](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/retry-baseline-scalar-282782-20260908T070733Z).
- [Held manifest](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/retry-baseline-scalar-282782-20260908T070733Z/queue-manifest.json).
- [Retry status](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/retry-baseline-scalar-282782-20260908T070733Z/retry-status.json).
- [Disabled continuation marker](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/dedicated-282209-20260908T005547Z/full/queued-retry.json).
- Controller: [run_arm_retry.py](../../scripts/run_arm_retry.py); report/hold hook: [summarize_arm_reproduction.py](../../scripts/summarize_arm_reproduction.py).

The prepared controller supports the previous two-arm proposal only. A changed cohort or inclusion of SelectionARM requires updating and checking the controller/manifest before execution. It cannot simply be unheld with new task IDs. While held, the launcher writes the original report and follows its normal server cleanup; there is no promise that the actor will remain resident for a later decision.

Available existing allocation: **282782** on **g005**, assigned GPU `GPU-90ac2a02-abfa-147c-19ff-bbada4033da3`, expires **2026-09-08 02:37:55 PDT**. No new allocation or extension is requested or authorized by this document.

Validation: **13 retry tests passed**, including the held-queue case, original-result preservation, separate reporting, resource identity, and quota handling. The 9 existing ARM contract tests passed before this hold-only change. No retry GPU task has started.

The project filesystem previously rejected Git object creation with `Disk quota exceeded`; source is also backed up on scrubbed storage. If a future report copy to the project encounters the same quota, the complete Markdown report remains in its scrubbed run directory, with the copy error recorded in JSON.
