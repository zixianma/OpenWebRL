# RL checkpoint evaluation: baseline, Browser Use, and scheduling

Reference-policy checkpoint evaluations, the separate Browser Use protocol, and reward-ranked evaluation scheduling. Preserve each protocol and cohort when comparing results. Allocation and cancellation entries remain dated experiment history.

## Contents

- [Intermediate baseline checkpoint evaluation](#baseline-checkpoint-evaluation)
- [Browser Use checkpoint evaluations](#browser-use-checkpoint-evaluation)
- [Evaluating checkpoints whose training reward enters the top five](#reward-rank-evaluation-queue)

---

<!-- document:BASELINE_CHECKPOINT_EVALUATION.md:start -->
<a id="baseline-checkpoint-evaluation"></a>
## Intermediate baseline checkpoint evaluation

_Source record: `BASELINE_CHECKPOINT_EVALUATION.md`. Dated entries retain their historical context._


Training lineage: W&B `zixianma/openwebrl/qcq7i4ug`. Inventory checked on
2026-09-11 while allocation 287371 continued training.

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

Valid-only success is successes divided by `(300 - invalid attempts)`. After-38
has the highest observed valid-only and all-task success rates.
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
[after-38 evaluation](https://wandb.ai/zixianma/openwebrl/runs/qcq7i4ug-eval-after38-287588).
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
W&B: [after-39 evaluation](https://wandb.ai/zixianma/openwebrl/runs/qcq7i4ug-eval-after39-287596-r1).
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

<!-- document:BASELINE_CHECKPOINT_EVALUATION.md:end -->

---

<!-- document:BROWSER_USE_CHECKPOINT_EVALUATION.md:start -->
<a id="browser-use-checkpoint-evaluation"></a>
## Browser Use checkpoint evaluations

_Source record: `BROWSER_USE_CHECKPOINT_EVALUATION.md`. Dated entries retain their historical context._


<a id="browser-use-checkpoint-evaluation--checkpoint-38-stealth-job-287879-running--2026-09-11"></a>
### Checkpoint 38 stealth job 287879 running — 2026-09-11

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
W&B startup confirmed [qcq7i4ug-eval-browseruse-after38-287879](https://wandb.ai/zixianma/openwebrl/runs/qcq7i4ug-eval-browseruse-after38-287879).
The output is runtime `evaluations/qcq7i4ug-browseruse-287879-after38`; its
`evaluation.log` contains detailed progress, and the controller log is
`logs/slurm-stealth-after38-287879.out`. Full GPU restore and all-task evaluation
results are still pending.

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
