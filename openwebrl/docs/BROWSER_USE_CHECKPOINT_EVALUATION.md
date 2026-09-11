# Browser Use checkpoint evaluations

## Checkpoint 38 prepared, awaiting compute approval — 2026-09-11

The user selected **after training iteration 38** for a new stealth evaluation.
Its checkpoint is runtime
`runs/openwebrl-4b-reference-287371-20260911T025909/iter_0000037`, with 468 Adam
updates. Its completed local-browser evaluation scored **107/300 (35.67%)**,
valid-only **107/228 (46.93%)**, the best observed result among evaluated
checkpoints through 40. The previous 20/30 stealth pair remains canceled.

The new template is `scripts/evaluate_browser_use_after38_2gpu.sbatch`: **2 H200,
16 CPUs, 480 GiB, up to 3 hours**, **6 GPU-hours / estimated $5.40**. It owns and
awaits its worker, uses explicit CPU binding, and permits only supervised retries
within the same deadline. This three-hour request exceeds the existing top-five
queue's two-hour-per-job approval and must receive new explicit approval before
submission. It does not consume or extend the remaining two local-browser queue
slots unless the user explicitly changes that budget.

The isolated source is `reference-browseruse-eight-decimal-v2-20260911`. Its task gate is
**eight concurrent browsers**, below the previously observed ten-session account
limit. The 300 tasks, checkpoint, GPT-4.1 judge, prompts, decoding and maximum
steps remain the same. Source hashes record the browser backend, timeout and
concurrency changes. Fifteen CPU preparation/wrapper tests and the shell syntax
check passed; the exact checkpoint dry-run plan is saved under
`evaluations/browser-use-preflight-after38-20260911/evaluation_plan.json`.
W&B will use `qcq7i4ug-eval-browseruse-after38-JOB` (retry suffix if necessary).

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

## Earlier preparation and canceled 20/30 attempt

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

## Prepared execution and budget

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

## Protocol and source isolation

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

## Validation and lifecycle

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


## Canceled attempt and current hold, 2026-09-11

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
