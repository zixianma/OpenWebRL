# RL checkpoint evaluation: baseline, Browser Use, and scheduling

<a id="arm-mixed-bonus-iter90-results-20261002"></a>
## Mixed-only bonus iteration90 — October2

Job340425 completed training90 and full300 local-browser, GPT-4.1/action_history,
T0 evaluation. Overall success is122/300 (40.67%); valid-only is122/233 (52.36%).
Fixed100:38/100 (38.00%) overall and38/74 (51.35%) valid-only. The native89
checkpoint has1,106 Adam updates with matching scheduler and saved task cursor.
All300 task identities, nonempty rollout archives and verdicts, actual GPU
checkpoint restoration, and W&B final history passed independent checks.
Fourteen valid failures have native judge-not-run sentinels:9 truncated,5 failed.
The archive audit checked ZIP metadata and serialized entries, not every tensor byte.
This historical local-browser result is3.00pp above reweight90; collection dates
and valid-task sets differ, so this is not a controlled significance claim.
[Audit](arm_results/rl_integration/mixed-bonus-iteration90-audit.json).


Reference-policy checkpoint evaluations, the separate Browser Use protocol, and reward-ranked evaluation scheduling. Preserve each protocol and cohort when comparing results. Allocation and cancellation entries remain dated experiment history.

## Contents

- [Evaluation harness: code map, protocols, commands and tests](#evaluation-harness-guide)
- [Jev Ultrafast: completed10-task pilot](#jev-ultrafast-online-mind2web-20261004)
- [Kev0.8B/27B completed paired pilot](#kev-paired-online-mind2web-20261004)

- [Expanded 4,102-task outcome-only baseline: iterations10/20](#expanded4102-iter10-results-20261003)
- [WebVoyager iteration90: completed comparison](#arm-webvoyager90-results-20260930)
- [Matched iteration90 paired tests and95% CIs](ARM_RESULTS.md#arm-stealth90-paired-inference-20260930)
- [OM2W difficulty breakdown, three repeats](ARM_RESULTS.md#arm-stealth90-difficulty-20260930)
- [Matched iteration90 stealth rerun: o4-mini/T0.6](#arm-stealth90-o4-matched-20260929)
- [Additive/Gate B iteration90 first stealth evaluations](#arm-stealth90-threeway-repeats-20260928)
- [Mixed-only bonus/reweight iterations10–80 full300](#arm-mixed-pair-iter10-results-20260928)
- [Original-bonus20/30/40/50/60 full300 backfills](#arm-original-backfill-results-20260927)
- [Gate B iteration90 full300](#arm-gate-b-iter90-results-20260928)
- [Gate B iteration100 full300 and completed training](#arm-gate-b-iter100-results-20260929)
- [Gate B iteration80 full300](#arm-gate-b-iter80-results-20260927)
- [Gate B iteration70 full300](#arm-gate-b-iter70-results-20260927)
- [Failure sampling40% iteration20 full300](#arm-failure-coverage-iter20-results-20260926)
- [Failure sampling40% iteration10 full300](#arm-failure-coverage-iter10-results-20260926)
- [Gate B iteration50/60 completed evaluations](#arm-gate-b-iter50-60-results-20260926)
- [Failure β1 iteration20 full300](#arm-failure-beta1-iter20-results-20260925)
- [Failure β1 iteration10 recovered full300](#arm-failure-beta1-iter10-results-20260925)
- [Gate C iteration50/60 completed evaluations](#arm-gate-c-iter50-60-results-20260925)
- [Paper protocol and best-checkpoint rerun](#paper-om2w-protocol-20260912)
- [Intermediate baseline checkpoint evaluation](#baseline-checkpoint-evaluation)
- [Browser Use checkpoint evaluations](#browser-use-checkpoint-evaluation)
- [Evaluating checkpoints whose training reward enters the top five](#reward-rank-evaluation-queue)
- [Canonical ARM comparison at rollout iteration 20](#arm-iteration-19-evaluations-20260915)

---

<a id="jev-ultrafast-online-mind2web-20261004"></a>
## Jev Ultrafast browser evaluation — October 4

**Completed and independently audited: 1/10 successes (10%), 10 valid, 0 invalid.**
The cohort is the first 10 tasks in the unchanged 300-task Online-Mind2Web file,
in dataset order. This startup pilot is not a representative performance
estimate. Job 343691 completed in 148 seconds (2m28s), exit 0, within the approved
0-GPU/4-CPU/8 GiB/two-hour allocation. A full 300 cohort needs a separate approval.
[Aggregate audit and usage](rl_results/jev-ultrafast-pilot-20261004.json).

The adapter pins [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast/tree/1231850a0bf1a0c0341fe408ef1668dbbfdfac46)
at `1231850a0bf1a0c0341fe408ef1668dbbfdfac46`. Its DOM reader, operation/target
questions, choice validation, stale-page guards and executor are unchanged.
A Playwright CDP connection replaces the desktop Browser Harness connection.
[TypeSafe's model reference](https://docs.typesafe.ai/models) identifies
`jev-1.13.0` as the current version behind `jev-latest`; requests pin that
version and reject a different returned model identity.

| Setting | Prepared pilot |
| --- | --- |
| Policy | Jev 1.13.0; argmax operation and matching target, no actor screenshots |
| Text helper | GPT-4.1-mini-2025-04-14; T 0.6, top-p 0.95, 1,024 output tokens |
| Helper difference | Upstream demo uses Mercury 2.5; `--text-provider openrouter` prepares that alternative in a new cohort |
| Browser | Isolated Browser Use session per task; proxy disabled; 1120×780 upstream viewport |
| Episode | 30 executed actions, at most 60 decision cycles, 600-second actor timeout |
| Judge | o4-mini, canonical AgentTrek prompt and verdict parser, seed 42, actual final screenshot |
| Outcome | Independent verdict; `DONE` is only the actor's stopping signal |
| Approved allocation | 0 GPUs, 4 CPUs, 8 GiB RAM, 2 hours total including retries |
| Browser limit | 10 sessions, at most 2 concurrent, 12-minute expiry each |
| API attempt ceilings | 1,800 Jev, 1,800 helper, 40 judge HTTP attempts, including service retries |
| Execution status | Completed; 10/10 attempts and all 10 remote-session closures verified |

The helper is explicitly labeled because the configured credentials support
OpenAI, whereas no OpenRouter helper key was found. Jev's classification has no
temperature/top-p/top-k sampling controls. The helper's sampling is recorded
separately. This DOM agent also has no generated final-answer channel, unlike
the project's Qwen actor. These differences, the viewport, and instrumented
screenshot collection prevent treating its timing or success rate as an
otherwise identical Qwen experiment.

Entry points:
[`jev_eval.py`](../jev_eval.py),
[`evaluate_jev_ultrafast.py`](../../scripts/evaluate_jev_ultrafast.py),
[`evaluate_jev_ultrafast_cpu.sbatch`](../../scripts/evaluate_jev_ultrafast_cpu.sbatch).
The runner imports no GPU training stack. Dry-run preparation:

```bash
/gpfs/scrubbed/zixianma/openwebrl-runtime/jev-eval-venv/bin/python \
  scripts/evaluate_jev_ultrafast.py \
  --output /gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/jev-ultrafast-next-plan
```

The installed upstream package and dependency versions are checked and recorded
in the private plan. After exact approval, the prepared batch script runs the
same command with `--execute` and owns/awaits every worker. A new output
directory is required for a changed protocol. The controller preserves
completed attempts, stops dispatch after provider failure, charges its elapsed
time ledger on restart, and never silently reruns an interrupted browser task.
Remote session IDs have durable ownership markers and cleanup; capability URLs
and credentials are excluded from logs.

Private artifacts live under runtime
`evaluations/jev-ultrafast-om2w-pilot-20261004/`: plan, heartbeats, time ledger,
per-task compressed DOM/screenshot states, executed actions, API attempt/response
records with usage, final screenshot, judge request/response, verdict and
summary. Only task ID, instruction and start URL enter the actor; hidden
reference answers and rubrics are stripped. Summary denominators are
successes/planned tasks and successes/valid attempts, with missing, invalid and
provider-blocked states separate. Token usage and instrumented actor latency
are summarized. W&B project identity is fixed to `openwebrl-evals`; this runner
currently persists local metrics and does not create a W&B run.

Validation: **12 tests passed**, including a real local Chromium fixture with
mocked model calls that filled a field, clicked search, selected DONE, and saved
the independently checked final page and judge evidence. Tests also cover
reference stripping, unchanged canonical judge semantics, retry accounting,
provider/model failures, cleanup ownership and incomplete-cohort denominators.
Jev, both OpenAI models and Browser Use passed read-only credential checks.
The subsequent live pilot made 199 Jev, 4 helper and 10 judge calls, all HTTP 200.
Returned identities were `jev-1.13.0`, `gpt-4.1-mini-2025-04-14` and
`o4-mini-2025-04-16`. All 219 compressed snapshots, 10 fresh final screenshots,
judge messages/responses and exact task identities passed an independent audit.
The completed code is archived with the private run; its published source is
commit `7dc79cf488f95c33f29f57ccd23bd0a28ac63a6a`.

The actor executed 15 actions across 199 decisions. Five episodes stopped as
BLOCKED, two as DONE and three reached 60 decision cycles. The three capped
episodes executed 0, 2 and 5 actions: upstream freshness/target guards repeatedly
rejected selections. These failures and encountered anti-bot pages remain in
the overall denominator. Here **valid** means usable browser evidence and a
parseable judge verdict; it does not certify that the site was available.
DONE produced one success and one failure. Instrumented actor latency averaged
17.73 s (median 10.12 s); this includes browser setup and evidence collection.
Mean Jev HTTP latency was 95.4 ms (median 90.2 ms). The pilot measures the whole
DOM-agent stack, including executor limitations and the lack of a final-answer
channel, rather than isolated decision accuracy.

### What the public results establish

Checked October4. The1/10 above is an Online-Mind2Web startup pilot using
our adapter, fresh remote browser sessions and the unchanged Jev Ultrafast
executor. It is not a reproduced full300 score for Jev, and the public speed
demonstrations do not establish one.

| Public source | Reported result | Scope and comparison limit |
| --- | --- | --- |
| [Official Jev Ultrafast performance report](https://github.com/browser-use/jev-ultrafast/blob/main/docs/performance.md) | 3/3 for each runtime; optimized median7.092s | Three repeats of one Google Flights goal on an existing Chrome profile; explicitly not a broad reliability benchmark |
| [Independent jev-ra public benchmarks](https://github.com/brnyxx/jev-ra/blob/main/docs/BENCHMARKS.md#the-public-benchmarks) | Online-Mind2Web3/10 initially, then3/10 and2/10 | Different Jev-based implementation, difficulty-stratified sample, browser profile and official WebJudge; contextual evidence, not a matched reproduction |
| [Kev model evaluations](https://github.com/jaredpalmer/kev#models) | Kev27B0.851 development /0.889 test on new-source accuracy | Classification and decision datasets, not browser-task success |
| [Browser Use Cloud Online-Mind2Web](https://github.com/browser-use/online-mind2web) | bu-max291/300 (97%) | Different cloud agent; its published procedure uses Claude to judge final responses rather than our AgentTrek screenshot/history protocol |

The low pilot score warrants diagnosis. Three runs used all60 decision cycles
while executing0,2 and5 actions; two other tasks displayed Cloudflare security
pages. One task reached the requested comparison page but failed the judge
because this demo has no generated final-answer channel. Two zero-action
BLOCKED decisions occurred on ordinary initial pages, not blank navigation
results. These are distinct failure modes, not evidence that every failure
comes from the classifier or from site availability. Ten ordered tasks also do
not provide a representative ranking of the decision models. The adapter and
browser execution need further validation before interpreting a larger score
as general Jev/Kev capability.

<a id="kev27b-actor-full300-20261004"></a>
## Kev27B as the direct browser policy: full300 preparation — October 4

**Prepared and locally validated; not submitted or approved.** This scales the
standalone Kev27B pilot (3/10) to all300 Online-Mind2Web tasks, freshly collected
including the pilot10. It is separate from the ongoing SFT+Kev27B experiment,
where an image-conditioned SFT model proposes five actions for Kev to select.
Both selector runs and this prepared direct-policy run contain the same300 task IDs.

The direct policy retains the pinned `upstream-v1` Jev Ultrafast operation/target
questions, DOM reader and guarded executor. Kev27B receives DOM text, chooses
operation and target with calibrated argmax (checkpoint temperature1.319507910772894),
and delegates field text to GPT-4.1-mini-2025-04-14 at T0.6/p0.95/1,024 response
tokens. There is no SFT proposer or generative Kev sampling. The user's
T1.0/p0.95/4K SFT-proposal settings do not change this classifier's calibration.
The actor receives no screenshots. The o4-mini/AgentTrek judge receives the saved
fresh terminal screenshot and action history, with seed42 and4,096 response tokens.
Browser Use,1120×780 viewport,30 actions,60 decisions and600 seconds per task
match the direct pilot. The experimental change is cohort size; operational changes
add an isolated localhost port, frozen runtime and shared durable budget reservations.

| Proposed limit | Value |
| --- | ---: |
| Compute, including every failed/replacement attempt |1 H200,8 CPUs,120 GiB;14,400 scheduler seconds total |
| Fresh browser sessions, including recovery reserve |330 |
| Concurrent browsers / expiry |2 /12 minutes |
| Browser lifetime upper bound |3,960 browser-minutes |
| Local Kev requests, including warmups and retries |59,410 |
| GPT-4.1-mini HTTP attempts / output cap each |10,000 /1,024 tokens |
| o4-mini judge HTTP attempts / output cap each |1,320 /4,096 tokens |

The pilot averaged25.786 seconds per task including judging. Linear extrapolation
at two concurrent tasks is3,868 seconds (about64 minutes), before model startup
and variation across the remaining sites. The four-hour ceiling is a budget,
not an expected duration or a guarantee that every harder task will finish.
No unused pilot or selector allocation time transfers into this proposed budget.

[`evaluate_kev_actor_full300.py`](../../scripts/evaluate_kev_actor_full300.py)
prepares the source/model/dependency checks, exact approval gate, Slurm request,
aggregate W&B tracking in `openwebrl-evals`, GPU telemetry and owned worker
lifecycle. It charges scheduler elapsed for every previous attempt before a
replacement; browser and HTTP attempts are reserved durably before dispatch.
Valid failed tasks are never rerolled, and invalid/interrupted attempts require
preserved evidence and diagnosis before recovery. Collection completion remains
separate from `verified_complete`; all300 artifacts, verdicts, closed sessions,
W&B and scheduler accounting require independent audit. The prepared supervisor
queues the owning agent for repair; it becomes active only after approval/submission.

Private manifests: `runtime/evaluations/kev27b-actor-full300-20261004/`;
frozen runtime: `runtime/reference-kev27b-actor-full300-20261004-v1`.
Validation passed37 offline checks plus the opt-in local Chromium fill/click/done
and durable-judge-evidence test. Model calls were mocked in that browser test;
preparation made no paid requests and submitted no allocation. Task payloads,
screenshots and subsequent review HTML stay private.

<a id="kev-paired-online-mind2web-20261004"></a>
## Kev alternatives: smallest and largest — October 4

**Completed and independently audited: Kev 0.8B 0/10; Kev 27B 3/10.** All20 attempts have valid saved evidence and judge verdicts. The current
[Kev 1.0 family](https://github.com/jaredpalmer/kev) ranges from 0.8B to 27B.
Both used the exact same10 tasks as the completed Jev pilot, sequentially
within one shared approved budget. The unchanged Jev Ultrafast policy, DOM extraction, guarded
executor, GPT-4.1-mini field-text helper, Browser Use sessions and o4-mini judge
are shared. The experimental change is the System One decision endpoint and
model. Kev does not generate field text. Browser content is collected afresh for each model,
so this is a same-task comparison with live-site variability.

| Setting | Kev 0.8B | Kev 27B |
| --- | --- | --- |
| Hub repository | `jaredpalmer/kev-0.8b` | `jaredpalmer/kev-27b` |
| Pinned v1.0 revision | `bf75a6a8848ea6960ff2ed108d9ed44c2941174f` | `af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf` |
| Base | Qwen3.5-0.8B-Base | Qwen3.8-27B |
| Weights | LoRA adapter plus pinned base | Full BF16 weights, approximately 51.3 GB |
| Shipped probability calibration | T 2.3510958125672174 | T 1.319507910772894 |
| Selection | Argmax; no generative sampling | Argmax; no generative sampling |
| Server context limit | 65,536 tokens; no truncation | 65,536 tokens; no truncation |
| Upstream accuracy-validated context | 8,192 tokens | 65,536 tokens |

Kev serving source is pinned to
`fe64b1274ea7f80d4095866df90666abb03e9cf6`. An isolated environment uses
upstream's locked dependencies and its CUDA serving additions: torch 2.8.0,
transformers 5.17.0, flash-linear-attention 0.5.2, Triton 3.7.1 and
causal-conv1d 1.7.0. Triton intentionally overrides torch's older dependency pin,
as in upstream's serving image. Both models use BF16, fused kernels, CUDA
graphs and the checkpoint's shipped calibration; date preprocessing and state
truncation are disabled. Both weight caches passed SHA-256 verification against Hub metadata, including
all eleven 27B shards and the 0.8B adapter/base weights. Both tokenizers strictly
admitted all 199 saved requests; the prepared paired manifests passed dry-run
source, credential-presence and task-identity checks. Both models then passed actual
GPU loading, identity checks, warmup requests and live browser inference.

[`evaluate_kev_pair.py`](../../scripts/evaluate_kev_pair.py) verifies model-file
hash receipts, source/dependency identities and exact paired task identities.
Servers open the verified revision-specific cache directories directly; all
Hub/tokenizer loading is offline. This avoids an upstream Hub-ID resolution
path that still attempts a metadata request with the offline flag set.
At startup it checks the actual server checkpoint, base, CUDA device, dtype,
calibration and truncation policy, then validates short/long saved Jev requests
before opening paid browser sessions. The alias `kev-latest` alone is not
accepted as evidence of which checkpoint was loaded. The controller owns and
awaits both model servers and all browser workers, preserves separate variant
artifacts and charges a shared time ledger without resetting it on retry.

| Approved pilot budget | Limit |
| --- | ---: |
| Allocation, both models and all retries combined | 1 H200, 8 CPUs, 120 GiB, 2 hours |
| Model-task attempts | 20 (10 per model) |
| Remote browser sessions / concurrency / expiry | 20 / 2 / 12 min |
| Local Kev HTTP attempts, including four warmups | 3,604 |
| GPT-4.1-mini HTTP attempts, including retries | 3,600 |
| o4-mini HTTP attempts, including retries | 80 |

Prepared batch entry:
[`evaluate_kev_pair_1gpu.sbatch`](../../scripts/evaluate_kev_pair_1gpu.sbatch).
Private artifacts use runtime `evaluations/kev-pair-om2w-pilot-20261004/`.
The original29 adapter tests passed, including actual Chromium fill/click/DONE,
checkpoint/calibration mismatch rejection, secret-safe local routing and
inherited-setting isolation. All 199 saved Jev requests passed Kev's API schema and strict tokenizer
admission (up to 3,953 state tokens; 9,256 packed tokens across questions).
Local Kev requests allow 120 seconds for a new CUDA compilation; the 600-second
actor episode limit is unchanged. The new runner also enforces the project's
4,096-token judge response cap. The completed Jev pilot omitted that explicit
cap; its longest judge response used 681 completion tokens. Keep this recorded
configuration difference when interpreting the historical pilot comparison.

| Result | Jev 1.13.0 | Kev 0.8B | Kev 27B |
| --- | ---: | ---: | ---: |
| Successes / tasks |1/10 |0/10 |3/10 |
| Valid / invalid |10/0 |10/0 |10/0 |
| Executed actions |15 |75 |62 |
| Decision requests |199 |232 |231 |
| Mean decision HTTP latency, ms |95.4 |43.9 |224.7 |
| Median decision HTTP latency, ms |90.2 |21.5 |203.5 |
| Mean actor episode, seconds |17.73 |21.29 |22.57 |
| Text-helper requests |4 |0 |3 |
| Judge requests |10 |10 |10 |

[Kev aggregate audit](rl_results/kev-pair-pilot-20261004.json). Both Kev variants
finished within **600 seconds total allocated GPU time**, including every failed
attempt:34391424s,343916267s,343922309s. The first failure was a missing Python
header during Triton compilation; exporting the existing compatible headers
fixed it. The second was a server-port probe rejecting a closed listener's
TIME_WAIT connection; matching the server's SO_REUSEADDR bind policy fixed the
handoff. No completed task or billed browser session was repeated. The final
job exited0. All20 task identities, state archives, terminal screenshots, judge
requests/verdicts, provider responses and session-stop receipts passed independent
audits. Kev0.8B chose only CLICK/BLOCKED and never used the typing helper. Kev27B
had one valid failed trajectory ending in a dropdown-execution RuntimeError;
valid evidence does not imply that the actor ran without errors.

### Harness diagnosis and private review

All reported scores above use the unchanged pinned upstream DOM reader and
execution policy, labeled `upstream-v1`. The supplied Jev code offers controls
based on visibility and viewport geometry, but checks `elementFromPoint` only
at execution. A covered control can therefore be repeatedly offered and rejected.
A local Chromium fixture reproduces this mismatch for buttons and selects.
The opt-in `actionable-v2` snapshot filters offered targets with the same existing
center-point hit test. It keeps the execution guard and restores upstream globals
on exit. After dismissing the fixture overlay, the underlying controls reappear.
No live evaluation of this revised harness has run. Keep its results separate.

The original traces did not record each caught StalePage reason. Future attempts
now save code-owned rejection reasons in a private `execution-rejections.jsonl`.
The historical Speedo screenshots show an overlay while all three policies exhaust
60 decisions without executing an action. This supports investigating the mismatch,
but does not establish the exact rejection cause of every historical attempt.
The dropdown RuntimeError also needs a live replay with the improved telemetry;
blindly retrying a possibly partially executed dropdown is intentionally avoided.

[`render_jev_kev_review.py`](../../scripts/render_jev_kev_review.py) builds the
self-contained private review `runtime/visualizations/jev-kev-review-20261004.html`.
It contains10 matched tasks,30 trajectories and215 distinct original screenshots,
with provider probabilities, per-step execution status, typing outputs and judge
explanations. It is approximately30MiB and requires no server or external assets.
Chromium checks passed for all three columns, task filtering, timeline controls,
image enlargement and mobile layout with no JavaScript errors. Task payloads and
screenshots remain local; only the renderer and template are versioned publicly.

The review now resolves each selected target number from that decision's exact
request to its label and harness action ID. Click execution resolves the cached
DOM node, checks freshness/visibility/occlusion, then dispatches CDP mouse events
at the current bounding-box center. Model target numbers and harness IDs are
internal references, not HTML `id` attributes or model-generated coordinates.
Of568 element-target decisions,527 have saved geometry with an exact matching
page fingerprint;41 retain their request label without an invented location.
Highlights use a separate saved decision-state screenshot. They show observed
bounds, not recorded execution coordinates or proof that the click executed.
Three mapping regression tests and Chromium overlay/lightbox/mobile checks pass.

Before scaling, rerun the same small cohort under an explicitly recorded revised
harness, compare rejected-decision counts and terminal outcomes, then expand the
cohort if useful progress is verified. Distinguish website blocks, execution
limitations and model selection failures.

<a id="sft-jev-kev-selection-pilot-20261004"></a>
### Completed pilot: original SFT proposes five actions; Jev/Kev select

All40 terminal records are audited:35 valid episodes and five diagnosed invalid
records preserved from earlier attempts. The canonical o4-mini/AgentTrek scores
are **SFT4/10, SFT+Jev3/10, SFT+Kev0.8B4/10 and SFT+Kev27B9/10**. Valid-only
rates use9,6,10 and10 episodes respectively; see the
[results table](RL_RESULTS.md#sft-decision-selection-20261004).
The large-Kev tally includes three questionable positives: Speedo sizing/discount,
Carvana without a specific car/price, and an empty IGN boardgame search. The forum
maximum-replies claim also has a review limitation. These are preserved canonical
scores under a permissive judge, not independently relabeled strict task success.

Job344458 completed at21:59:57 Pacific on October4 after1419seconds. Together
with the three preserved earlier attempts, **4805seconds (80m05s)** were charged
against the effective6986-second approval, leaving2181seconds unused. All42
created browser sessions are stopped. Final usage is137 Jev requests,353 local
Kev requests including two warmups,35 judge requests and2625 saved actor proposals.
The ten superseded choices from interrupted small-Kev episodes remain charged.
All40 records have verified evidence or a documented invalid diagnosis, and no
terminal corrections remain staged. Five invalid records mean
`all40_evidence_verified=false`; completed supervision means
`all40_results_audited=true`, with limitations retained explicitly.

The following history preserves the preparation, repairs and intermediate audits.

The requested SelectionARM-style replacement is implemented separately from the
Jev Ultrafast direct agent. At each state, the original OpenWebRL-4B-SFT samples
five independently seeded full reasoning/action candidates. Jev, Kev0.8B or
Kev27B chooses one via a five-way probability distribution; the original actor
output tuple, tool arguments and text values are executed unchanged. There is
no GPT typing helper, action repair, training update or first-candidate fallback.
A malformed provider response stops the cohort and preserves the proposals.
A fresh SFT-only control executes candidate0 from the same seed schedule.

The proposer uses the standing stealth protocol: T0.6,p0.95,k20,4096 response
tokens,30 turns,32K context, full actor history and the current screenshot.
The text-only selectors receive task, URL, page text (explicit16000-character
limit), interactive-element geometry, last-five executed reasoning/actions and
all five full proposals. This modality differs from the visual SelectionARM;
frames/canvas content may be absent. No hidden benchmark rubric is supplied.
The shared judge is o4-mini/AgentTrek with fresh final screenshots and a4096-token
completion cap. All cohorts request Browser Use with `proxyCountryCode: null`
and a1280×1000 viewport; provider proxy accounting is discussed below. This is a separate protocol from the direct1120×780 Jev agent.

Prepared entry: [`evaluate_sft_decision_selection.py`](../../scripts/evaluate_sft_decision_selection.py),
[`batch template`](../../scripts/evaluate_sft_decision_selection_2gpu.sbatch).
The four conditions each use the same first10 tasks (40 fresh episodes total),
with2 concurrent browsers. Approved allocation: **2H200,16CPUs,240GiB,1hour total
including startup and retries**; one GPU for SFT and one for the sequential Kev
servers. It releases on completion and stops at its cap if incomplete. API caps:
300Jev calls,602local Kev calls including warmups,160judge attempts,4096judge
completion tokens per attempt;40browser sessions with12-minute expiry and no
additional helper calls. The user approved this exact allocation and API/browser budget.
Attempt lineage is344043 →344049 →344086; the recoveries are described below. The
completed direct-pilot budget is not reused. Combined offline tests passed39 cases, including literal candidate
preservation, probability failures, secret-safe routing and local overlay behavior.

A pre-start check found that `asyncio.wait_for` creates a child task: assigning a
ContextVar there did not return the final screenshot to the judge's task. Source
revision2 uses a shared per-episode image holder. A concurrent two-episode test
verifies correct propagation and isolation. The same queued job was briefly held
for this repair and released; no GPU or browser budget was consumed. Both source
revisions, prior approval/plan and repair lineage remain in private runtime storage.

Job344043 exposed a missing lazy `browser_use_sdk` import after loading SFT.
The first two cohorts aborted before browser creation; these are infrastructure
failures, not task scores. The attempt was canceled after74 scheduler seconds.
All failed rows, logs, manifests and source revisions remain preserved. No
browser sessions, Jev calls, judge calls or task proposals were consumed; one
successful Kev0.8B warmup is retained and reused. Source revision3 adds a browser
SDK preflight before GPU loading and before workers start, using an isolated
dependency directory with pinned package hashes. The training environment and
scientific settings are unchanged. Deployed-source CPU imports and nine selection
regressions pass. Replacement344049 requests3480seconds, so both attempts can
consume at most3554 of the approved3600seconds. The supervisor pointer follows
the replacement; results remain unverified until independent artifact audit.

During the SFT control, Discogs lost its active page/context after a `go_back`.
The environment returned an empty observation, and the next inference path
failed on the absent screenshot. Its15 recorded actions and stopped-session
receipt are preserved; no fresh final image or judge verdict exists. The task
remains invalid. The remote closure's underlying cause is not established.
An `ObservationGuard` now has a regression test proving that a missing image
stops before either actor or selector inference and that valid outputs pass
through unchanged. This diagnostic fix is deployed in source revision4 for
replacement344086. It cannot restore the closed session.
Repeating this task would require an additional browser session beyond the40
reserved for the four matched cohorts; none has been opened for a retry.

Job344049 then halted during the first two Jev episodes: one HTTP200 response
returned probabilities `[0.56, 0.15, 0.10, 0.04, 0.14]`, summing to0.99 after
hundredth rounding. Its selected candidate was the clear argmax. The harness's
0.001 sum tolerance incorrectly rejected it and stopped both concurrent episodes.
Source revision4 accepts only the rounding error implied by five hundredth-rounded
Jev values (at most0.025); other responses retain the original tolerance. Finite
values, range, keys, model identity and choice/argmax agreement remain required.
No probabilities are changed and there is no fallback. The saved failing response
passes offline replay, and19 selection/review regression tests pass. Both aborted
episodes, including their terminal images and the unexecuted request, remain
invalid and preserved; no judge was called for their aborted status.

Attempt344049 consumed1207seconds, bringing the charged total to1281seconds.
Replacement344086 requests2280seconds (38minutes), for a maximum combined3561
of the approved3600seconds. It reuses10 SFT results and2 interrupted Jev results,
then runs only the remaining28 unstarted episodes. The plan differs only in source,
manifest and controller hashes; actor, seeds, tasks, scientific settings and API
caps are unchanged. Source4 passes deployed CPU imports and browser preflight.
The persistent pointer and supervisor follow344086. Retrying the3 invalid episodes
would need browser-cap approval beyond40; no such retries are included.

Replacement344086 started on g023 and resumed Jev at request8, preserving the
seven earlier requests and all completed/invalid results. The actor GPU is
producing five-candidate batches, Jev responds in roughly0.1seconds, and W&B
resumed the existing evaluation identity. Discogs again lost its page/context
after `go_back` from its login page, this time after7 selected actions. The new
observation guard correctly stopped before another inference and saved a receipt.
The last saved state contained one tab; available evidence cannot distinguish a
closed page, context, CDP transport or remote browser. This fourth invalid episode
has no terminal image or verdict and is preserved without replay. Other tasks
continue under the same allocation; thirteen finished records have been audited.

At the17:49 Pacific handoff, both the SFT and Jev cohorts have all10 records.
The independent audit verifies9 SFT episodes (4 successes) and6 Jev episodes
(3 successes), with5 diagnosed invalid records across the two cohorts. The two
Jev step-limit terminal verdicts were recovered from their preserved final images
using the unchanged canonical judge and are both failures. W&B agrees with the
corrected summary; every SFT and Jev browser session is stopped.

The last Jev episode reached its600-second timeout after selecting `go_back`
as proposal30. Browser-level CDP commands still responded, but a read-only
`Runtime.evaluate` on the page timed out. The underlying cause is unresolved.
Its archived prompt, all30 input images and selected actor proposals are preserved
and cross-checked against the selection records. Completion of the last browser
action and the terminal state are unverified; there is no fresh final image or
judge verdict. The private review labels that distinction explicitly.

Kev0.8B started on the same allocation at17:47, with its pinned model card,
fresh successful selector requests and activity on both GPUs verified. Kev27B
remains pending. Roughly11minutes of controller time remained at this handoff,
so completing both Kev cohorts within the original budget is uncertain. These
partial results are not a completed four-way comparison or `verified_complete`.

The allocation subsequently stopped at its configured time guard at18:00 Pacific.
Final Slurm accounting charges344043=74s,344049=1207s and344086=2105s:
**3386/3600 approved scheduler seconds**, leaving214s. That balance cannot fit
another model startup, the configured180-second shutdown reserve and the remaining
episodes, so no replacement was submitted. All29 created browser sessions are
closed; two surviving remote sessions were explicitly stopped after the worker
exited. All attempts, original verdicts and corrections remain preserved.

The stopping-point audit contains **27 terminal records**: SFT4 successes/10
records (9 valid), SFT+Jev3/10 (6 valid), and SFT+Kev0.8B3/7 (7 valid). Small Kev
has two interrupted episodes with9 and1 saved selections respectively, plus one
unstarted episode; Kev27B has not started. Three small-Kev step-limit verdicts were
recovered from their original final images and are all failures. The completed
small-Kev successes are the AeroAPI comparison, CarMax search and MTA FEIS report.
These partial counts must not be used as a completed four-condition comparison.

The independent audit verifies all27 available terminal records or their diagnosed
invalid provenance, and separately inventories the13 missing results. There are
no remaining staged verdict corrections or unexplained audit errors. Saved totals
are29 browser sessions,137 Jev calls,136 local Kev calls including one warmup,
22 judge HTTP requests and1545 actor proposals (excluding startup warmup and
any interrupted generation before trace persistence). The seven small-Kev terminal
images were also visually reviewed. The private HTML now exposes the unfinished
proposal traces with explicit execution uncertainty, and distinguishes interrupted
episodes from tasks that never ran. `verified_complete` remains false.

W&B now records the corrected small-Kev3/7 summary with `complete=false`; its
stale running status was changed to failed after verifying Slurm termination.

The run pointer records `partial_budget_limited` and the supervisor stops at the
explicit approval blocker; no further automatic execution is authorized. A private,
unsubmitted continuation proposal requests one additional2-H200/16-CPU/240-GiB
hour and two additional browser sessions (42 total) to replay the two interrupted
episodes and run the11 untouched episodes. It preserves all27 terminal results,
including the five diagnosed invalid records, and does not increase the other API
caps. This proposal is not an approval or a guarantee that every remaining task
will finish within an hour.

The user subsequently approved that continuation: **one additional hour total on
2 H200 /16 CPUs /240 GiB**, including retries, and **two additional browser
sessions (42 total)**. Job**344458** was submitted for3600seconds and started
on g012 at21:36 Pacific on October4.
The enforced cumulative cap is6986seconds:3386 already charged plus3600 newly
approved; the original unused214seconds are excluded. All other API caps and the
scientific protocol remain unchanged. The active-agent supervisor now follows
344458, with a verified accepted continuation message.

The continuation preserves the27 terminal records and runs only the13 missing
results: three small-Kev tasks followed by ten27B tasks. Small-Kev continuation
state is isolated from its complete prior directory. The ten selection requests
from the two interrupted episodes remain intact and charged, with their hashes
checked independently; a lineage receipt excludes those old choices from the
new retry trajectories. All seven completed small-Kev results are unchanged.
The selector resumes after request135, preventing request-ID reuse. Frozen source
revision4 passes deployed CPU/browser imports, the19 existing regression tests
pass, and the prelaunch evidence audit retains27 records with no unexplained
issues. The earlier incomplete comparison remains provisional while this job runs.

Startup verification confirms fresh small-Kev selections after request135,
the pinned BF16 model, actor GPU activity and the resumed `openwebrl-evals` run.
The first new completed episode reaches the requested IGN Breath of the Wild
walkthrough; both its final screenshot and canonical success verdict were checked.
The current independent audit contains28 terminal records: small Kev has4
successes in8 valid completed episodes, while its last two tasks are running and
Kev27B awaits the handoff. The five earlier diagnosed invalid records remain
unchanged, with no new unexplained audit issues. This is still a partial comparison.
A regression-tested correction to the terminal-judge recovery helper permits
updating a finished cohort when a later cohort has no summary yet; it changes no
actor, selector, judge or sampling settings.

At21:45 Pacific, the small-Kev cohort is complete with **4 successes/10 valid
episodes**. Carvana and the IGN boardgame task both reached30 actions; their
preserved final screenshots received canonical failure verdicts, and the staged
corrections have been applied. All ten small-Kev rollouts, final images and
verdicts pass the independent audit. The three new terminal screenshots were
visually reviewed, and provider reads confirm all three new browser sessions
stopped. The two interrupted historical attempts remain preserved and charged.
W&B records the completed cohort; stale interrupted/not-started fields from the
previous attempt have been cleared while retaining historical attempt counts.

The controller has handed the same allocation to Kev27B. Its pinned model card,
BF16 CUDA backend, successful warmup and fresh selection requests are verified;
the actor and selector occupy roughly68GB and74GB of GPU memory respectively.
The fourth cohort uses the intended `openwebrl-evals` project and unchanged
five-candidate protocol. The audit now contains30 terminal records with the same
five diagnosed invalid records and no unexplained issues. No small-Kev judge
corrections remain pending; the ten large-Kev outcomes still need completion and
audit before the overall pilot can be marked verified.

At21:52 Pacific,33 terminal records have been audited. Kev27B has two successes
in three finished episodes: AeroAPI plan comparison and the Discogs submission
overview both finish in six actions with matching final screenshots and canonical
success verdicts. Trader Joe's reaches30 actions; the recovered judge verdict is
failure because the requested home-store setting was never completed. Its
correction is preserved and staged until the cohort exits. Seven large-Kev
results remain, with no new diagnosed invalid records or unexplained audit errors.
These are provisional counts, not a completed four-condition comparison.

The next audit contains36 terminal records, including six valid Kev27B episodes
with five canonical successes. Visual review flags the Speedo success for further
checking: the saved cart contains a black kneeskin in size`26 L`; both actor and
judge call that Large without a verified mapping, and the saved product page
states a regular$280 price without establishing the highest discount. The
[official product page](https://speedo.com/en-us/products/womens-lzr-racer-pro-recordbreaker-kneeskin-black-87190920001)
also lists numeric R/L variants. This leaves the size and discount requirements
unverified. The FlightAware forum screenshot confirms an opened64-post thread;
the maximum-replies comparison relies on the canonical judge's interpretation.
The private HTML displays these review notes alongside the preserved verdicts.
Canonical scores remain unchanged, and an evidence/provenance audit must not be
read as independent ground-truth relabeling.

The subsequent Carvana success also needs that distinction. Its judge explicitly
acknowledges that no individual listing or specific price was found, then accepts
a Bing/Copilot recommendation of model years and trims as an effective workaround.
That does not establish the cheapest available car meeting every constraint.
The frozen AgentTrek prompt permits partial-goal credit, including more than eight
correct actions or completing one of two subtasks. The saved canonical verdict is
retained and flagged in the private review. These pilot scores therefore measure
this permissive judge protocol, rather than independently verified strict goal
completion.

The final IGN boardgame verdict illustrates the same concern: the saved screen is
an empty search for `boardgame`, with no requested review opened. The judge awards
success for the conclusion that the review is unavailable and asserts that IGN
reviews only video games; the trajectory does not establish that premise. The
strict requested outcome is unsupported, and the review flags it alongside the
Speedo and Carvana concerns. All ten Kev27B final screenshots were visually
reviewed. The completed cohort retains nine canonical successes and one canonical
failure; the earlier staged Trader Joe's verdict has been applied.

A read-only provider audit confirms the first10 SFT sessions are stopped. The
provider reports approximately$0.01167 browser cost and$0.05273 proxy cost despite
the explicit null proxy request. Inspection of the pinned SDK's serialized body
confirms that it preserves `proxyCountryCode: null`. This is an unresolved provider
accounting discrepancy, not evidence that the requested setting was omitted or a
verified zero-proxy-charge run. The Discogs session stopped after roughly149seconds,
well before its12-minute expiry, so session expiry does not explain that failure.

[`audit_sft_decision_selection.py`](../../scripts/audit_sft_decision_selection.py)
independently checks source/task pins, deterministic seeds, selected actor text,
probability argmax, screenshot identity, canonical judge prompt/verdict, session
receipts and all scheduler attempts. It separates diagnosed invalid results
from complete evidence and never marks the run complete itself.
[`render_sft_decision_review.py`](../../scripts/render_sft_decision_review.py)
builds the private `sft-decision-selection-review-20261004.html`, linked from the
direct-agent review. It shows per-condition timelines, all proposed actions,
selected probabilities and terminal evidence. JPEG previews are capped at1600
pixels; original PNGs remain private. The partial review labels unfinished and
invalid episodes explicitly. Requested browser dimensions are1280×1000; the
provider returned differing actual viewport/DPR values, which the existing
environment reads for coordinate transforms and the saved images preserve.

W&B uses the same four evaluation run identities across recovery. Resuming a
run initially retained the failed bootstrap's `complete=true` summary; this was
corrected to incomplete and the original infrastructure counts retained under
job344043. Final metrics must be checked against the independent artifact audit.

The audit also found that the shared training reward returns zero without
calling the judge when status is not `COMPLETED`. Three SFT episodes
reached30steps and exposed this behavior. Their preserved terminal PNGs and
histories were sent to the same canonical o4-mini/AgentTrek judge, which returned
failure for all three. Corrections are applied after the SFT worker exited;
the audited control has4 successes,9 valid episodes and1 invalid episode out of10. This recovery uses the existing per-task four-call allowance
and160-call pilot cap, with no new browser or actor calls.
[`rejudge_sft_decision_terminal.py`](../../scripts/rejudge_sft_decision_terminal.py)
preserves original results, stages actual verdicts and applies corrections after
the cohort worker exits. Later step-limit episodes require this same check before
the final comparison. A browser abort without terminal evidence cannot use this
recovery. The supervisor now also wakes the agent for invalid results, skipped
terminal judges, halted selectors and stale heartbeats.

Persistent service `openwebrl-sft-selection-supervisor-20261004.service` polls the
current run pointer every60seconds and queues the owning agent thread on state
changes or every15minutes. The initial same-thread continuation was accepted and
the service heartbeat verified. The agent owns diagnosis, bounded recovery and
independent completion review; the watcher itself does not modify GPU work.
The requested endpoint is all40 task results with candidate traces, final images,
judge verdicts, closed browser sessions and actual scheduler-time accounting.

The [Jev model reference](https://docs.typesafe.ai/models) explicitly specifies
text-only input with no image/audio/video modality. Kev's pinned serving path
also tokenizes text. SFT and the terminal judge see screenshots; the Jev/Kev
selectors see textual observations and candidate reasoning. This is a material
modality difference from visual SelectionARM, not an equal-input model swap.


<a id="sft-selection-full300-20261004"></a>
### Full300 SFT + Jev and Kev27B comparison

The user requested all300 Online-Mind2Web tasks with the original SFT proposer
and each of Jev and Kev27B. Both runs preserve the pilot's five full
reasoning/action candidates, deterministic seed schedule, hosted browser,
two concurrent episodes, top-p0.95/top-k20,4096 response tokens,
30 turns and canonical o4-mini/AgentTrek judge. The user subsequently requested
**actor temperature1.0**, replacing the pilot's0.6, and authorized proceeding
with the proposed allocations and caps. A new frozen source revision preserves
the earlier pilot/preparation settings. Each run collects300 fresh
episodes, including the pilot's first10 tasks; the pilot remains separate.
Jobs **344661 (Kev27B)** and **344708 (Jev, replacing344662)** are submitted under the following
separate approvals; neither has a full300 result yet.

| Run | H200 | CPU | RAM GiB | Total hours including retries | Browser-session cap | Selector-call cap | Judge-call cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT + Kev27B ·344661 |2 |16 |240 |10 |330 |9910 local, including warmups |1320 |
| SFT + Jev ·344662→344708 |1 |8 |120 |10 |330 |9900 Jev |1320 |

These are separate caps totaling30 H200-hours,660 browser reservations and2640
judge calls. Each run reserves at most50000 actor proposals, including failed
generation reservations and a conservative startup allowance. Unused pilot
time is not transferred. The Kev27B pilot's1629.55 summed episode-seconds for10
tasks project to about6.8 hours for300 tasks at concurrency2, before startup and
load imbalance. This extrapolation used the earlier temperature0.6 pilot.
The10-hour ceiling adds headroom but cannot guarantee completion
on different live tasks. Valid failures are not automatically rerolled.

The recent [controlled SFT baseline](ARM_INFERENCE.md#arm-controlled-inference-results-20261004)
is reusable as a **reference**: actor0 is106/300 (35.33%), and pooled ordinary
pass@1 is528/1500 (35.20%). All300 task IDs and the original SFT actor agree.
That collection used local browsers, temperature0.7/top-p0.9/top-k−1,1024 output
tokens, different seeds and1800-second episode timeouts, versus the selector
pilot's hosted browsers and600-second timeout. Its judge uses the same
o4-mini/AgentTrek rubric, but the baseline automatically assigns failure at
the30-step limit without invoking it. The selector pilot recovered canonical
verdicts for those terminal states, and both full300 selector runs automate
that operation. Therefore the reused baseline does not isolate the
selector's causal gain under the pilot protocol. No new SFT-only run is included
in this request; any strict re-judging would need a shared protocol
applied to all compared methods.

[`evaluate_sft_selection_full300.py`](../../scripts/evaluate_sft_selection_full300.py)
prepares pinned per-mode plans and batch files, refuses submission without exact
matching approval, and charges every prior scheduler attempt before replacement.
The frozen worker adds durable shared request reservations, separate W&B identities
under `openwebrl-evals`, and automatic canonical judging at the step limit—the
same evidence-recovery operation applied during the pilot, with unchanged judge
prompt and termination status. CPU/browser import preflight and24 targeted tests
pass, including five-candidate requests at the exact actor sampling settings.
Dedicated continuation supervisors follow each run's current job pointer and
retain the original per-mode budget through recovery.
Independent completion still requires all300 records, preserved rollouts and
terminal evidence or explicit invalid diagnoses, browser shutdown, final budget
accounting, W&B verification, documentation and a private review.

Jev recovery on October4: job344662 stopped after1,003 scheduler seconds when
one HTTP200 reply named choice1 but assigned0.35 to it and0.36 to choice5.
The strict validator correctly rejected the contradiction; its global halt
interrupted a second browser episode. All eight valid completed results remain
unchanged; the two interrupted attempts and41 superseded decision traces are
preserved. The exact failed request returned a valid response in a charged replay.
Revision v3 adds up to two explicitly recorded retries of that identical request,
with no new actor samples and no replacement action. Every request consumes the
original shared9900-call cap. Existing model, probability, choice and argmax
validation remains strict. All147 earlier successful selections replay unchanged;
27 targeted tests pass, including exhaustion of retry and request budgets.
Replacement344708 requests1H200/8CPU/120GiB for34,980 seconds (9h43m), within
the34,997 seconds remaining. Kev344661 keeps its existing v2 worker. A private
full300 review now supports300-task navigation, external JPEG previews and saved
retry responses; its snapshots remain explicitly provisional until the final audit.

<a id="evaluation-harness-guide"></a>
## Evaluation harness: code map and entry points

The `arm` branch contains the evaluation code, its shared runtime dependencies,
and regression tests. Use the entry points below; dated runners preserve the
settings and recovery rules of earlier experiments. Paths remain stable so that
frozen jobs and old commands continue to resolve. Task-pool payloads, raw
trajectories, credentials, native checkpoints and live approval/budget ledgers
are private runtime artifacts, not part of this publication.

| Topic | Entry points | Purpose |
| --- | --- | --- |
| Native RL checkpoints | [`evaluate_baseline_checkpoint.py`](../../scripts/evaluate_baseline_checkpoint.py), [`eval_monitor.py`](../eval_monitor.py) | Evaluate a saved checkpoint with zero optimizer updates; validate restoration, cohort coverage and separate W&B routing |
| HF/student checkpoints | [`run_arm_full300_checkpoint_eval.py`](../../scripts/run_arm_full300_checkpoint_eval.py) | Actor-server evaluation of merged checkpoints; optional compact/serial-action variants |
| ARM inference | [`arm_eval.py`](../arm_eval.py), [`arm_inference.py`](../arm_inference.py), [`serve_arm.py`](../../scripts/serve_arm.py) | Actor-only, SelectionARM and ScalarRM; full versus action-only candidates; see [inference protocol](ARM_INFERENCE.md) |
| Stealth OM2W | [`eval_benchmark.py`](../eval_benchmark.py), [`prepare_paper_benchmark.py`](../../scripts/prepare_paper_benchmark.py) | Reusable o4-mini/AgentTrek template and isolated source preparation |
| Historical matched stealth runs | [`run_arm_stealth90_o4.py`](../../scripts/run_arm_stealth90_o4.py), [`run_arm_stealth90_o4_more.py`](../../scripts/run_arm_stealth90_o4_more.py) | Fixed baseline/Additive/Gate B checkpoints, independent repeat cohorts and provider-credit recovery |
| WebVoyager | [`arm_webvoyager_generator.py`](../../scripts/arm_webvoyager_generator.py), [`run_arm_webvoyager90.py`](../../scripts/run_arm_webvoyager90.py) | WebVoyager terminal judge, saved task records and the matched 595-task comparison |
| Judge-only replay and retries | [`rejudge_saved_gpt41.py`](../../scripts/rejudge_saved_gpt41.py), [`retry_stealth_invalid.py`](../../scripts/retry_stealth_invalid.py) | Rejudge existing evidence without actor/browser collection; separately record invalid-task retries |
| Milestones and supervision | [`run_arm_milestone_queue.py`](../../scripts/run_arm_milestone_queue.py), [`arm_job_supervisor.py`](../../scripts/arm_job_supervisor.py), [`run_expanded_baseline.py`](../../scripts/run_expanded_baseline.py) | Checkpoint handoffs, bounded recovery, artifact audits and active-agent review |
| Results and plots | [`analyze_arm_paired_evals.py`](../../scripts/analyze_arm_paired_evals.py), [`plot_arm_interactive.py`](../../scripts/plot_arm_interactive.py), [`plot_arm_reward_hacking.py`](../../scripts/plot_arm_reward_hacking.py) | Paired task comparisons, interactive learning curves and ARM/task-success diagnostics |

### Keep the protocols separate

| Protocol | Browser | Terminal judge | Actor T / top-p / top-k | Response / turns |
| --- | --- | --- | --- | --- |
| RL milestone monitor | Local | GPT-4.1 / `action_history` | 0 / 1 / 1 | 4,096 tokens / 30 |
| Matched stealth OM2W | Browser Use stealth | o4-mini / AgentTrek | 0.6 / 0.95 / 20 | 4,096 tokens / 30 |
| Matched WebVoyager | Browser Use stealth | GPT-4o / WebVoyager | 0.6 / 0.95 / 20 | 4,096 tokens / 30 |

ARM inference reproduction has its own candidate-sampling configuration; do not
substitute the RL monitor settings. The saved worker command, generation
function and judge identity define a cohort, not its filename. Separate eval
workers are routed to `openwebrl-evals` even when a legacy controller passes the
training project. Browser concurrency is an account-wide budget; parallel jobs
must share that limit.

### Running and checking an evaluation

Use the project's installed SGLang/Megatron/PyTorch environment and browser
setup from the [repository README](../../README.md). Packaging checks use
Python 3.12 and CPU execution. Launch scripts retain this cluster's GPFS paths,
Slurm resource profiles and reference manifests; a fresh clone supplies the
code but still needs models, benchmark data, environment configuration and a
validated frozen source. Historical `*90*` runners additionally require their
private run manifests. They are reproducibility profiles, not arbitrary-model
CLIs.

For a native checkpoint in an **existing authorized allocation**, first inspect
a dry-run plan (the command below does not submit a job):

```bash
python scripts/evaluate_baseline_checkpoint.py \
  --source "$EVAL_SOURCE" --checkpoint "$CHECKPOINT" \
  --output "$EVAL_OUTPUT" --job-id "$SLURM_JOB_ID" \
  --gpus 2 --browser-env local_process --protocol monitor
```

`EVAL_SOURCE` must pass `reference_manifest.json` validation. `CHECKPOINT` is a
native directory such as `iter_0000019` for completed iteration20, with its
saved data cursor; `EVAL_OUTPUT` is a new directory under runtime `evaluations/`.
After checking the plan, add `--execute`. Stealth requires an isolated Browser
Use source followed by `prepare_paper_benchmark.py`; select `--browser-env
browser-use --protocol benchmark`. One-GPU evaluation also requires the
validated single-GPU source profile. Frozen source hashes are checked before
execution; changing public code does not change an already frozen training run.

Every task attempt writes `rollouts/<sha256(task_id)>.pt` and a paired `.json`
verdict/metrics record, including aborted or judge-error attempts. The `.pt`
contains the trajectory and screenshot tensors so temporary tensor mappings
can expire without losing judge evidence. Legacy stealth subset tools also
retain `completed_tasks/*.json`. Do not treat a completed Slurm job as a
completed cohort: verify exact task IDs, rollout/verdict pairs, the actual GPU
checkpoint-restore receipt and aggregate metrics. Missing attempts stay partial.

Report **overall = successes / planned tasks** and **valid-only = successes /
valid attempts**, including the valid denominator. Do not use the per-turn
`eval/online-mind2web-monitor` scalar as task success. Preserve original and
retry cohorts independently; do not replace valid failures with new attempts.
Judge-only replay uses the saved artifacts:

```bash
python scripts/rejudge_saved_gpt41.py \
  --input "$EVAL_OUTPUT" --rollout-dir "$EVAL_OUTPUT/rollouts" \
  --source "$EVAL_SOURCE" --output "$REJUDGE_OUTPUT" \
  --wandb-id "$REJUDGE_ID"
```

This previews GPT-4.1/action-history rejudging; `--execute` makes judge API calls
but does not collect new browser trajectories. Keep its verdicts separate from
an o4-mini or WebVoyager cohort.

### Tests and publication checks

With the project dependencies and `pytest` installed, this focused suite uses
synthetic fixtures/mocked services, without a GPU allocation, live browser or
paid judge call:

```bash
python -m pytest -q \
  tests/test_eval_monitor.py tests/test_eval_task_persistence.py \
  tests/test_rejudge_saved_gpt41.py tests/test_runtime_ports.py \
  tests/test_evaluation_project_routing.py tests/test_retry_stealth_invalid.py \
  tests/test_arm_stealth_o4.py::GeneratorTests \
  tests/test_arm_stealth_o4.py::PublicTemplateTests \
  tests/test_arm_stealth_o4.py::PublicPreparationTests \
  tests/test_arm_webvoyager90.py::GeneratorTests
```

The remaining prepared-plan, scheduler, continuation and milestone tests include
cluster integration checks against frozen sources and saved manifests. These
require the corresponding private runtime artifacts and Megatron checkout.
October3 publication validation: **310 regression tests passed**, including
the **41-test portable suite** above; Python syntax, 30 shell launchers and
repository file references were checked. The larger suite uses private cluster
fixtures. These checks did not launch a new GPU evaluation.

CPU checks establish protocol, persistence, bookkeeping and regression behavior;
they do not establish fresh GPU restoration or live website availability.

The reusable stealth template now explicitly selects AgentTrek, saves invalid
attempts, and stops new session requests after provider-credit exhaustion. Its
source preparer includes the persistence dependencies when upgrading an older
frozen source. Historical Sol inference planning now reads one saved result at a time, keeping
only task ID, validity and reward instead of retaining screenshot payloads.
The topic commits also include shared ARM reward, resume and
transport modules imported by the evaluation controllers; they are necessary
code dependencies, not new experiment launches.

<a id="expanded4102-iter10-results-20261003"></a>
## Expanded task-pool baseline: iterations10/20 — October3–4

The first full300 evaluation of the outcome-only baseline trained on4,102 tasks
(original2,102 + selected2,000) is complete: **81/300 =27.00% overall** and
**81/236 =34.32% valid-only**, with64 invalid tasks retained in the overall
denominator. The model started from the original SFT checkpoint at iteration0;
no ARM reward or selection is enabled in this data-only experiment.

Evaluation uses local browsers, GPT-4.1/action_history, T0/p1/k1,4,096 response
tokens and30 turns. The evaluation generation function overrides the launcher's
15-turn training default to30, as in the original baseline. Checkpoint10 is
native9, with162 Adam updates and matching scheduler state. All300 exact task
identities, paired nonempty rollout archives and saved verdict records, actual
GPU actor restoration, and final W&B history were independently checked.
Native judge-not-run sentinels remain for53 failed and1 truncated valid
trajectories; these count as failures. Archive validation covers ZIP metadata
and serialized entries, with bounded screenshot/payload samples, rather than a
full scan of every tensor byte. Raw trajectories and task identities stay private.

The historical original-pool iteration10 result is70/300 (23.33%) overall and
70/234 (29.91%) valid-only: differences of **+3.67pp** and **+4.41pp** respectively.
Collection dates, valid-task sets and training randomness differ. This is an
early exploratory comparison, not evidence of a statistically established data
benefit.

Iteration20 completed on October4: **100/300 =33.33% overall** and
**100/237 =42.19% valid-only**, with63 invalid tasks. This is +6.33pp overall
and +7.87pp valid-only versus expanded-pool iteration10. The historical
original-pool iteration20 result was95/300 (31.67%) and95/232 (40.95%):
**+1.67pp overall / +1.25pp valid-only**, with the same historical-comparison
limitations. Native checkpoint19 has312 matching Adam/scheduler updates.
All300 task identities, rollout/verdict pairs, native GPU actor restoration and
final W&B history were independently verified. The raw per-turn reward metric
(30.78%) is not the task success rate.

Training continues from20 with the same optimizer, scheduler, cursor and W&B
lineage within the original allocation budget. Target60 and evaluations30–60
remain incomplete.

[Iteration20 aggregate audit](arm_results/rl_integration/expanded4102-iteration20-audit.json) ·
[Iteration20 evaluation W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/outcome-only-expanded-4102-20261003-iter20).

[Aggregate audit](arm_results/rl_integration/expanded4102-iteration10-audit.json) ·
[Evaluation W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/outcome-only-expanded-4102-20261003-iter10) ·
[Method and approved budget](ARM_INTEGRATION_PLAN.md#arm-expanded-outcome-baseline-20261002).

<a id="arm-webvoyager90-results-20260930"></a>
## WebVoyager: matched iteration90 results — September30

| Method | Iteration | Overall | Valid-only | Successes / valid / total | Overall Δ vs baseline |
| --- | ---: | ---: | ---: | --- | ---: |
| Outcome-only baseline |90 |66.89% |68.27% |398 /583 /595 |+0.00 pp |
| Additive ARM |90 |64.37% |65.03% |383 /589 /595 |-2.52 pp |
| Gate B |90 |67.06% |67.86% |399 /588 /595 |+0.17 pp |

Actor-only inference on all595 released OpenWebRL/FARA tasks; Browser Use stealth,
T0.6/p0.95/k20,4096 response tokens,30 turns, and GPT-4o/WebVoyager terminal judge.
This is the approved WebVoyager protocol; OM2W stealth uses o4-mini/AgentTrek.
All1,785 task records and paired rollout archives, embedded screenshot evidence,
native89 GPU restoration, checkpoint counters, final W&B metrics and browser
cleanup passed independent checks. Invalid attempts remain in the overall denominator.

Gate B has one more success than baseline (+0.17pp overall) and a slightly
lower valid-only rate; this pass shows no clear WebVoyager gain. Additive is
−2.52pp overall. These are single evaluations of fixed checkpoints, not
independent training seeds. The released dataset has53 date-updated instructions
relative to the paper-pinned version, so absolute scores are not exact paper reproductions.

Browser/proxy charges were$7.11 total: baseline
$2.33, Additive$2.47,
Gate B$2.31; balance after completion
$4.36, zero active sessions. GPU and judge API costs are separate.
All three GPU allocations were released. Including initial failed startups,
consumed GPU-hours were5.926,
6.282, and5.860,
within the separate12h caps; unspent budgets are released.

[Outcome-only baseline audit](arm_results/rl_integration/webvoyager-gpt4o-t06-baseline-iteration90-audit.json) · [Additive ARM audit](arm_results/rl_integration/webvoyager-gpt4o-t06-additive-iteration90-audit.json) · [Gate B audit](arm_results/rl_integration/webvoyager-gpt4o-t06-gate-b-iteration90-audit.json).

<a id="arm-stealth90-o4-matched-20260929"></a>
## Matched iteration90 stealth rerun: o4-mini/T0.6 — September29

<!-- arm-stealth-repeat3-progress-start -->
**All three repeats verified, September30.**

| Method | Iteration | Full300 overall mean ± SD | Valid-only mean ± SD | Overall Δ vs baseline | Valid denominators, repeats1/2/3 |
| --- | ---: | --- | --- | ---: | --- |
| Outcome-only baseline |90 |**55.22 ± 2.14%** |57.65 ± 1.77% |+0.00 pp |286/290/286 |
| Additive bonus |90 |**58.44 ± 1.35%** |61.45 ± 1.19% |+3.22 pp |286/286/284 |
| Gate B: relaxed gate |90 |**58.78 ± 3.89%** |61.36 ± 3.88% |+3.56 pp |286/288/288 |

Sample SD measures repeat-evaluation variability for fixed checkpoints, not
training-seed uncertainty. All nine sets of300 task IDs and paired archives,
saved judge records, frozen protocol, native89 restoration and final W&B metrics
passed audit. Repeat1 retains pre-outage outcomes plus only credit-blocked
retries; repeat2/3 are fresh collections with seeds1235/1236. Valid-only means
average per-repeat ratios. [Aggregate and source hashes](arm_results/rl_integration/stealth-o4-t06-iteration90-three-repeat-summary.json).

Third-pass jobs337132/337133/337134 completed with161/171/174 successes and
286/284/288 valid tasks. Each released its GPU after17,106/18,470/17,575 seconds,
respectively; all attempts remain charged within the original7h caps.
<!-- arm-stealth-repeat3-progress-end -->

The [difficulty breakdown](ARM_RESULTS.md#arm-stealth90-difficulty-20260930) is now reconstructed
from all 2,700 retained verdicts. Under human-step bins (80 easy / 141 medium /
79 hard), Gate B gains most on medium (+6.86 pp) and Additive on hard (+5.91 pp).
Two stale medium labels are corrected for this analysis only; raw records remain
unchanged. Full denominators, SD, valid-only rates and identity checks are linked.


**Fresh repeat2, September29:** Gate B90 is independently verified at
**189/300 =63.00% overall**, **189/288 =65.63% valid-only**; fixed100
**67/100 =67.00%**, **67/98 =68.37% valid-only**. All300 expected task IDs and
paired archives were checked; all288 valid tasks contain nonempty o4-mini/
AgentTrek verdicts, with no credit-exhaustion metadata. Frozen generation code
confirms T0.6/p0.95/k20/4096 tokens/30 turns; server RNG seed1235. Native89
model restoration,1136 Adam updates, scheduler offset0, cursor, shard extents
and sampled finite CPU tensors passed. W&B finished with matching metrics.
Job337131 exited0 after16,380 seconds and released8,820 unused seconds.
[Repeat2 audit](arm_results/rl_integration/stealth-o4-t06-gate-b-iteration90-repeat2-audit.json).
Baseline repeat2 is verified at **173/300 =57.67% overall**,
**173/290 =59.66% valid-only**; fixed100 **58.00% /59.18%** (58/98 valid).
The same independent checks passed for300 paired archives and290 nonempty
valid-task verdicts; native89 retained1016 Adam updates and its known scheduler
offset1. Job337129 exited0 after16,826 seconds and released8,374 unused seconds.
[Baseline repeat2 audit](arm_results/rl_integration/stealth-o4-t06-baseline-iteration90-repeat2-audit.json).
Additive repeat2 is also verified: **176/300 =58.67% overall**,
**176/286 =61.54% valid-only**; fixed100 **61.00% /61.62%** (61/99 valid).
All300 archives/verdict records and286 valid judge texts passed independent review;
native89 restoration,1150 Adam updates, scheduler offset0 and final W&B metrics
agree. Job337130 exited0 after18,585 seconds, releasing6,615 unused seconds.
[Additive repeat2 audit](arm_results/rl_integration/stealth-o4-t06-additive-iteration90-repeat2-audit.json).
Relative to baseline in this round, Additive is+1.00pp and Gate B+5.33pp overall.
All third passes337132–337134 started on g008 after round2 finished, preserving
the9-browser concurrency cap. These are repeated evaluations of the same models; the final three-repeat means are above; training-seed inference remains unavailable.

**Verified first-pass result, September29:** Gate B90 is complete at **166/300 =55.33%
overall**, **166/286 =58.04% valid-only**; fixed100 is **62/100 =62.00%**,
**62/98 =63.27% valid-only**. Baseline is also verified: **163/300 =54.33%
overall**, **163/286 =56.99% valid-only**, fixed100 **56.00% /57.14%**.
Additive is now verified at **179/300 =59.67% overall**, **179/286 =62.59%
valid-only**; fixed100 is **60/100 =60.00%**, **60/97 =61.86% valid-only**.
The full300 overall differences from baseline are Additive+5.33pp and Gate B+1.00pp.
The independent audit checked the merged300 IDs and archives, every valid
judge text, native89 model restoration and checkpoint counters, W&B's final
162-task retry history, and browser cleanup. Job336973 exited0 and released its
GPU after9,828 seconds; all three attempts used17,680/25,200 seconds, leaving
7,520 unused. Full300 statistics come from retained plus retried records, not
from the retry-only W&B counters.
[Gate B audit with completion-time windows](arm_results/rl_integration/stealth-o4-t06-gate-b-iteration90-audit.json).
Baseline336971 released its GPU after10,190 seconds,17,827 seconds across all
attempts and7,373 unused; the same independent artifact/restore/W&B checks passed.
[Baseline merged audit](arm_results/rl_integration/stealth-o4-t06-baseline-iteration90-audit.json).
Additive336972 completed and released its GPU after10,942 seconds;
all attempts used18,702 seconds, leaving6,498 unused. Its128 retained records
plus172 retries cover exactly300 task IDs. Native89 restoration, checkpoint
shards/counters, all archives and valid verdict texts, W&B's final172/164/99
retry counters and zero remaining owned browser sessions passed verification.
[Additive merged audit](arm_results/rl_integration/stealth-o4-t06-additive-iteration90-audit.json).
[Two additional evaluations per method are complete; approval record](ARM_INTEGRATION_PLAN.md#arm-stealth90-three-repeats-o4-20260929).


The user clarified that **all stealth evaluations should use actor T0.6 and
o4-mini/AgentTrek**. The previous Additive/Gate B GPT-4.1/T0 runs used the wrong
protocol for the intended comparison; retain them as separately labeled history.
This default is recorded in root `AGENTS.md`. Local-browser monitoring retains
its existing protocol.

**Approved and submitted September29:** one fresh
full300 pass each of outcome-only baseline, Additive and Gate B, all at
iteration90 (`iter_0000089`). All three use the same frozen task IDs, Browser Use
stealth, actor temperature0.6/top-p0.95/top-k20, 4096 response tokens, 30 turns,
and the historical o4-mini/AgentTrek judge implementation. No inference-time
action selection. Save every task's rollout and verdict, including invalid
attempts; use distinct outputs and W&B identities in `openwebrl-evals`.

The approved resources are **three parallel 1-H200 ×7h jobs**, each with
8 CPUs/240GiB and three browser sessions: **21 GPU-hours maximum**, including
startup/retries, plus browser/judge service usage. Nine simultaneous sessions
stay within the previously observed ten-session account limit. Estimated runtime
is about5–6h. The initial allocations started together on September29 and
required the startup repair below; record actual task-collection intervals when reporting results. Existing
training budgets remain separate.

| Method | Recovery job | Tasks retried | Retained valid tasks | Retained ordinary invalids | Maximum remaining GPU time |
| --- | ---: | ---: | ---: | ---: | --- |
| Outcome-only baseline90 |336971 |168 |125 |7 |4h52m |
| Additive90 |336972 |172 |122 |6 |4h50m |
| Gate B90 |336973 |162 |131 |7 |4h49m |

**September29 credit-outage recovery:** jobs336864/336865/336866 ended after
Browser Use exhausted its account balance. The evaluator recorded HTTP402 session
creation failures as invalid tasks, causing apparent300-record completion despite
168/172/162 tasks never receiving a browser session. These are **incomplete cohorts**,
not full300 performance results. Independent archive inspection distinguished them
from the seven/six/seven ordinary timeout/screenshot failures above. Original
archives and records are preserved; successful and unsuccessful valid outcomes
are never replaced.

After the user replenished credits, the account check passed and the three
recovery jobs above started together on g010. Each retries only its explicitly
identified HTTP402 task IDs with unchanged actor checkpoint, sampling and judge.
The recovery sources add a credit preflight and stop on HTTP402 instead of draining
the remaining task queue; completion audits reject credit-blocked cohorts.
Twelve CPU tests passed, including artifact preservation and rejecting replacement
of an originally valid failure. The completed task sources will be merged by ID
for full300/fixed100 reporting after independent review. Report both collection
intervals and the outage; do not describe this as an uninterrupted collection.

All attempts count against the original7h-per-method cap. Before these recoveries,
baseline consumed7,637 seconds, Additive7,760, and Gate B7,852 (including initial
startup failures). No GPU budget was added. Durable incident manifests and exact
retry IDs are in runtime `arm-turn-bonus-preparation/stealth90-o4-t06-20260929/*-provider-recovery.json`;
launch and budget receipts are in `provider-recovery-submission.json` in that
control directory. The supervisor follows336971/336972/336973 and requires the
merged300-task artifact audit before final completion.

All three jobs were registered with the active supervisor before release.
The7h cap per method includes retries; account for every attempt and preserve
partial archives. Final completion requires all300 task IDs, saved verdicts,
rollout archives, checkpoint restoration and final W&B metrics to be verified.

The initial336861/336862/336863 attempts failed after33/28/33 seconds because
Ray's expanded Unix-socket filenames exceeded107 bytes. No tasks or browser
sessions started. Short job-specific `/tmp/so4-JOBID` roots fix the cause; a
regression test checks the actual Ray validator. All15 CPU tests passed before
relaunch. Replacements each have6h59m, keeping prior consumption inside the
original7h-per-method approval. A fresh read-only account check still reports
`rate_limit=10`; planned browser concurrency is three per job, nine total.

Preflight verified all three iteration90 checkpoint identities and shard sizes,
the common cohort, actual sampling function and YAML, unchanged AgentTrek judge
source, worker routing and W&B project. Fifteen CPU tests passed, including
invalid-judge handling without generic rejudging, persistence on aborted/error
tasks, and rejection of wrong judge/config/environment. GPU restoration and
live browser/judge validation remain startup checks in the approved allocations.
Report overall and valid-only on full300/fixed100; use paired all-task tests with
Holm correction for the three method comparisons, plus common-valid sensitivity.
One pass per trained model cannot estimate training-seed or repeated-run variance.

Controller `scripts/run_arm_stealth90_o4.py`; template
`scripts/evaluate_arm_stealth90_o4_1gpu.sbatch`; frozen generator
`scripts/arm_stealth_o4_generator.py`. Runtime request and three preview plans:
`arm-turn-bonus-preparation/stealth90-o4-t06-20260929/`. The batch controller owns
and awaits each worker; startup registers the cohort with active supervision.

<a id="arm-stealth90-threeway-repeats-20260928"></a>
## Additive/Gate B iteration90: first stealth evaluations — September28–29

**Scope: one full300 evaluation each for Additive90 and Gate B90.** The user
accepted the proposed single-GPU/seven-hour profile with only one run per method.
No fresh baseline or additional repeat jobs are included. This supersedes the
previous six-cohort plan. Both jobs were submitted and registered with the active
supervisor before release. Both completed September29 and exited early.

| Job | Method | Status | GPU cap |
| --- | --- | --- | --- |
|336697 |Additive90 |Verified complete;4h03m48s used |1 H200 ×7h |
|336698 |Gate B90 |Verified complete;3h39m44s used |1 H200 ×7h |

| Method | Successes /300 | Valid | Invalid | Overall | Valid-only | Fixed100 successes / valid |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Additive90 |159 |283 |17 |53.00% |56.18% |49 /96 |
| Gate B90 |169 |288 |12 |56.33% |58.68% |57 /98 |

The original fixed100 slices are49.00% /51.04% for Additive and57.00% /58.16%
for Gate B. Both cohorts contain exactly the300 expected unique task IDs; all600
rollout ZIP archives and saved GPT-4.1/action_history verdicts passed independent
checks. Native checkpoint89 restores and final W&B metrics match; all owned
browser sessions stopped. Gate B exceeds Additive by3.33pp overall (ten successes)
and2.50pp valid-only. The paired all-300 comparison has 57 Gate-B-only successes
and 47 Additive-only successes: exact two-sided McNemar p=0.378, with a paired
task-bootstrap 95% interval of −3.33 to +10.00 pp. On 277 common-valid tasks,
wins/losses are 52/46 (p=0.614). These single evaluations do not establish a
difference, estimate training-seed/repeat variance, or account for website-level
task correlations. No matched test against the historical baseline is reported
because its judge and actor temperature differ.
[Paired analysis](ARM_RESULTS.md#arm-stealth90-results-20260929) ·
[Additive audit](arm_results/rl_integration/stealth-additive-iteration90-audit.json) ·
[Gate B audit](arm_results/rl_integration/stealth-gate-b-iteration90-audit.json) ·
[Additive W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/stealth90-additive-r1-336697) ·
[Gate B W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/stealth90-gate-b-r1-336698).

Compared with Gate B90's earlier local-browser result, overall is13.33pp higher
and valid-only3.55pp higher; valid tasks increase234→288. Browser backend and
collection date changed, so this is not a new training improvement or a controlled
estimate of the browser effect. The historical stealth baseline90 is57.00%
overall /57.77% valid-only under o4-mini/T0.6, a different judge/decoding protocol.
Additive's earlier local result was39.33% /54.63% (216 valid), compared with
53.00% /56.18% (283 valid) here; the same comparison limits apply.

Both actors ran alone, without inference-time ARM selection, using the prepared
GPT-4.1/action_history judge, actor T0,4,096 response tokens,30 turns, frozen300
tasks and screenshot history. Each job uses Browser Use stealth with four
sessions and12-minute session limits: eight concurrent browsers across two jobs,
below the previously observed ten-session ceiling. A read-only account check
reported `rate_limit=10` and zero active sessions before submission.

**Approved reduced budget:** two jobs of **1 H200 ×7h,8 CPUs/240GiB each**,
**14 GPU-hours maximum** including retries. Actual combined use was7h43m32s
(7.73 GPU-hours), releasing6h16m28s of the cap. Each job exited after its result
and browser-session cleanup were verified. Existing training budgets remain
separate. Browser and judge service usage is additional.

| Method | Native checkpoint | Adam updates | Scheduler offset | Cohort |
| --- | --- | ---: | ---: | --- |
| Additive | training311962, `iter_0000089` |1,150 |0 |Full300, one pass |
| Gate B | training335729, `iter_0000089` |1,136 |0 |Full300, one pass |

Each job owns its worker and has isolated W&B (`openwebrl-evals`), output,
browser-session and tensor directories. A two-slot lock prevents duplicate
attempts from exceeding browser capacity. TP1 disables sequence parallelism in
an isolated source. Checkpoint/argument CPU checks and native GPU restoration
passed for both models. All task rollouts and judge verdicts are preserved.
There is no repeat SD from this first pass; no additional repeats are authorized.

**Historical baseline markers retained in the plot at the user's request:**

| Iteration | Judge / actor temperature | Overall | Valid-only | Provenance |
| ---: | --- | ---: | ---: | --- |
| 58 | o4-mini /0.6 |59.33% |59.93% |[Original valid attempts plus invalid/missing retries](#after58-invalid-retry-plan-20260914) |
| 90 | o4-mini /0.6 |57.00% |57.77% |[Original full300](#stealth90-temperature-plan-20260913) |

The iteration80 stealth marker is removed from the plot; its archived result is
preserved in the results sheet. The58/90 markers use a different judge/decoding
protocol from the new GPT-4.1/T0 evaluations, and58 includes retries. They are
historical references, not matched controls for estimating ARM improvement.

Controller: `scripts/run_arm_stealth90_parallel.py`; template:
`scripts/evaluate_arm_stealth90_1gpu.sbatch`. Exact plans, approval, per-method
budget and submission receipts: runtime
`arm-turn-bonus-preparation/stealth90-repeats-20260928/first-pass/`.
Every submitted/replacement job is registered with the active ARM supervisor.

<a id="arm-mixed-pair-iter50-bonus-results-20260930"></a>
### Mixed-only bonus50 verified — September30

Full300 **116/300 =38.67% overall**, **116/234 =49.57% valid-only**;
fixed100 **35.00% /48.61%**, with72 valid tasks. Native49 restoration retains668
Adam updates; all300 archives and saved judge records, exact task IDs, executed
local GPT-4.1/action_history/T0/4096-token/30-turn protocol and final W&B history
were independently checked. Sixteen valid attempts contain the baseline's
explicit judge-not-run sentinel for truncated/failed status. Saved data supports
future rejudging. Bonus is+1.67pp overall versus reweight50; the planned40/50/60
aggregate awaits60. The controller resumed training in the same allocation.
[Audit](arm_results/rl_integration/mixed-bonus-iteration50-audit.json).

<a id="arm-mixed-pair-iter10-results-20260928"></a>
<a id="arm-mixed-pair-iter20-results-20260928"></a>
<a id="arm-mixed-pair-iter30-results-20260929"></a>
<a id="arm-mixed-pair-iter40-results-20260929"></a>
<a id="arm-mixed-pair-iter50-results-20260929"></a>
<a id="arm-mixed-pair-iter60-results-20260930"></a>
<a id="arm-mixed-bonus-iter70-results-20261001"></a>
<a id="arm-mixed-reweight-iter70-results-20261001"></a>
<a id="arm-mixed-bonus-iter80-results-20261001"></a>
<a id="arm-mixed-reweight-iter80-results-20261001"></a>
## Mixed-only bonus/reweight iterations10–80 — September28–October1

Evaluations inside335699 (bonus) and335697 (reweight) completed all300 tasks at10,
20 and30; both30 evaluations completed September29. Reweight40 completed in335698
on September29, restoring `iter_0000039` at564 Adam updates. Bonus40 completed in
335700 the same day, restoring native39 at548 Adam updates. Reweight50 also
completed September29 in335698, restoring native49 at694 Adam updates.
Their controllers own training
continuation after evaluation. Native checkpoints are `iter_0000009` for10, `iter_0000019` for20 and
`iter_0000029` for30. All use the
same frozen full300 cohort and local-browser GPT-4.1/action_history protocol at
temperature0,4096 response tokens and30 browser turns. Full300 membership,
per-task judge identity, native checkpoint identity, nonempty rollout ZIPs and
final metrics were independently checked.

| Method | Iteration | Successes /300 | Valid | Invalid | Overall | Valid-only | Fixed100 successes / valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Mixed-only bonus + relaxed B |10 |92 |234 |66 |30.67% |39.32% |31 /75 |
|  |20 |92 |225 |75 |30.67% |40.89% |30 /69 |
|  |30 |96 |228 |72 |32.00% |42.11% |29 /74 |
|  |40 |100 |228 |72 |33.33% |43.86% |32 /73 |
|  |50 |116 |234 |66 |38.67% |49.57% |35 /72 |
|  |60 |113 |232 |68 |37.67% |48.71% |41 /75 |
|  |70 |115 |235 |65 |38.33% |48.94% |44 /77 |
|  |80 |108 |221 |79 |36.00% |48.87% |35 /69 |
| Mixed-only reweight + relaxed B |10 |82 |240 |60 |27.33% |34.17% |25 /72 |
|  |20 |97 |227 |73 |32.33% |42.73% |31 /70 |
|  |30 |90 |219 |81 |30.00% |41.10% |28 /64 |
|  |40 |106 |225 |75 |35.33% |47.11% |33 /66 |
|  |50 |111 |237 |63 |37.00% |46.84% |33 /73 |
|  |60 |110 |233 |67 |36.67% |47.21% |34 /76 |
|  |70 |116 |234 |66 |38.67% |49.57% |38 /76 |
|  |80 |118 |231 |69 |39.33% |51.08% |37 /75 |

**Reweight80 evaluation verified, October1:** full300 **39.33% /51.08%**
(118 successes,231 valid), fixed100 **37.00% /49.33%** (37/75). Job338905
evaluated native79 at1,060 Adam updates under the unchanged local-browser
GPT-4.1/action_history/T0/4096-token/30-turn protocol. Exact300 task IDs,
nonempty archive payload entries, saved judge records, checkpoint counters/
cursor/shard extents/sampled finite tensors, actual GPU restoration and final
evaluation W&B history passed independent checks. Eighteen valid failures have
explicit judge-not-run sentinels (13 truncated,5 failed). Artifacts remain at
`evaluations/arm-mixed-reweight-iter80-338905/`; per-task records stay local.
Native79 model-and-optimizer restoration subsequently passed for training
toward90. This evaluation does not complete the to90 allocation.
[Evaluation audit](arm_results/rl_integration/mixed-reweight-iteration80-audit.json) ·
[Evaluation W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-mixed-reweight-iter80-338905).

**Bonus80 evaluation verified, October1:** full300 **36.00% /48.87%**
(108 successes,221 valid), fixed100 **35.00% /50.72%** (35/69). Job338904
evaluated native79 at1,004 Adam updates under the unchanged local-browser
GPT-4.1/action_history/T0/4096-token/30-turn protocol. Exact300 IDs, paired
nonempty archive payload entries, saved judge records, checkpoint counters/
cursor/shard extents/sampled finite tensors, native GPU restoration and final
evaluation W&B history passed independent checks. Fifteen valid failures have
explicit judge-not-run sentinels (8 truncated,7 failed). Artifacts remain at
`evaluations/arm-mixed-bonus-iter80-338904/`. The later training-stage quota
failure does not invalidate this completed evaluation; recovery will reuse it.
[Evaluation audit](arm_results/rl_integration/mixed-bonus-iteration80-audit.json) ·
[Evaluation W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-mixed-bonus-iter80-338904).

**Reweight70 evaluation verified, October1:** full300 **38.67% /49.57%**
(116 successes,234 valid), fixed100 **38.00% /50.00%** (38/76). Job338905
evaluated native69 at934 Adam updates under the same local-browser
GPT-4.1/action_history/T0/4096-token/30-turn protocol. All300 expected IDs,
paired nonempty archive payload entries and saved judge records passed checks;
checkpoint counters/cursor/shard extents/sampled finite tensors, actual GPU
restoration and final evaluation W&B history were also verified. Seventeen
valid failures have explicit judge-not-run sentinels (15 truncated,2 failed).
Artifacts remain at `evaluations/arm-mixed-reweight-iter70-338905/`.
The controller subsequently passed model-and-optimizer restoration for training
toward80; this evaluation does not complete the to90 allocation.
[Evaluation audit](arm_results/rl_integration/mixed-reweight-iteration70-audit.json) ·
[Evaluation W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-mixed-reweight-iter70-338905).

**Bonus70 evaluation verified, October1:** full300 **38.33% /48.94%**
(115 successes,235 valid), fixed100 **44.00% /57.14%** (44/77). Job338904
evaluated native69 at894 Adam updates using the unchanged local-browser
GPT-4.1/action_history/T0/4096-token/30-turn protocol. Independent checks covered
all300 expected IDs, nonempty archive payload entries, saved task verdicts,
checkpoint counters/cursor/shard extents/sampled finite tensors, actual GPU
restoration and final W&B history in `openwebrl-evals`. Nineteen valid failures
have explicit judge-not-run sentinels (14 truncated,5 failed), consistent with
the existing protocol. Artifacts remain at
`evaluations/arm-mixed-bonus-iter70-338904/`; per-task review records stay local.
The controller resumed training toward80 after this evaluation; the to90
endpoint remains incomplete.
[Evaluation audit](arm_results/rl_integration/mixed-bonus-iteration70-audit.json) ·
[Evaluation W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-mixed-bonus-iter70-338904).

**Bonus60 endpoint verified, September30:** full300 **37.67% /48.71%**
(113 successes,232 valid), fixed100 **41.00% /54.67%** (41/75). All300 exact
IDs and paired archives/records, GPT-4.1/action_history/T0/4096-token/30-turn
settings, native59 restoration at786 Adam updates and final W&B history passed.
Nine valid failures carry the protocol's explicit judge-not-run sentinel
(6 truncated,3 failed). All six milestone cohorts were independently rechecked
(1,800 archives/records); final checkpoint counters, cursor,16 shard extents and
sampled finite tensors passed. Three extension attempts exited0, consuming
167,384/172,800 seconds. The allocation released5,416 unused seconds.
[Evaluation audit](arm_results/rl_integration/mixed-bonus-iteration60-audit.json) ·
[Endpoint audit](arm_results/rl_integration/mixed-bonus-to60-completion.json).

**Reweight60 endpoint verified, September30:** full300 **36.67% /47.21%**
(110 successes,233 valid), fixed100 **34.00% /44.74%** (34/76). Independent
checks covered all300 task IDs, paired nonempty archive payloads, stored judge
records, local GPT-4.1/action_history/T0/4096-token/30-turn settings, native59
restoration at814 Adam updates and final W&B history. Nineteen valid attempts
carry the baseline's explicit judge-not-run sentinel (15 truncated,4 failed).
The final checkpoint's metadata,16 shard extents, cursor and sampled tensors
passed. All six milestone cohorts10–60 were checked together (1,800 records/
archives), both extension jobs exited0, and the allocation was released.
Consumed additional-budget time is140,602/172,800 seconds, including evals;
original8h attempts remain separate. Reweight's40/50/60 mean is36.33% /47.05%;
this averages checkpoints, not training seeds. Bonus40/50/60 averages
36.56% /47.38%, only0.22pp higher overall; no clear winner is established.
[Matched aggregate](arm_results/rl_integration/mixed-pair-iteration40-60-summary.json).
[Eval60 audit](arm_results/rl_integration/mixed-reweight-iteration60-audit.json) ·
[Endpoint audit](arm_results/rl_integration/mixed-reweight-to60-completion.json).

Reweight50's fixed100 slice is33.00% /45.21%. Full300 overall rises1.67pp from40,
with225→237 valid tasks; valid-only changes47.11%→46.84%. Historical baseline50
is35.00% /44.87%, a descriptive+2.00pp overall difference. Bonus50 subsequently
completed at38.67% /49.57%.
All300 expected unique IDs, nonempty archive payloads, GPT-4.1/action_history
identities, temperature0/30-turn settings and final W&B history were verified.
Native49 has16 complete shard extents, a matching dataset cursor,694 aligned
Adam/scheduler updates and finite sampled tensor payloads. The controller
passed eight-GPU model/optimizer restoration and started the stage targeting60;
the final60 checkpoint/evaluation are now independently verified above.
[Independent reweight50 audit](arm_results/rl_integration/mixed-reweight-iteration50-audit.json).

Bonus40's fixed100 slice is32.00% /43.84%. Full300 overall rises1.33pp from30,
with228 valid tasks in both cohorts. All300 expected task IDs, archive pairs,
judge identities, final W&B history and native39 restore were verified. The
checkpoint's16 shard extents, dataset cursor,548 Adam/scheduler counters and
sampled small CPU tensor payloads passed checks. This is not a full bytewise
CRC scan of every rollout archive. The controller owns training toward50 next.
[Independent bonus40 audit](arm_results/rl_integration/mixed-bonus-iteration40-audit.json).

Reweight40's fixed100 slice is33.00% /50.00%. Full300 overall rises5.33pp from30,
with219→225 valid tasks; valid-only rises6.02pp. Historical baseline40 has100
successes and231 valid tasks (33.33% /43.29%). These differing-date results are
descriptive. Reweight exceeds bonus40 by2.00pp overall; the predefined40/50/60
aggregate remains outstanding. Independent inspection verified exactly300 expected unique
task IDs, all nonempty ZIP archives, GPT-4.1/action_history verdict identities,
temperature0 config, native iteration39 restoration and final W&B history.
The durable checkpoint's16 shards, metadata and cursor passed checks, and its
optimizer/scheduler counters agree at564 updates. Job335698 owns the continuation
toward50 after this evaluation; the requested final target remains60.

Bonus30's fixed100 slice is29.00% /39.19%. Versus20, full300 overall increases
1.33pp and valid-only1.22pp, with225→228 valid tasks. It is2.00pp above reweight30
overall and1.01pp above valid-only. Historical baseline30 also has96 successes,
but248 valid tasks versus228 here; differing dates and validity sets limit the
comparison. All300 archives and verdicts, native29 restoration at424 Adam updates
and final W&B metrics passed independent checks. The training target remains60.

Reweight30's fixed100 slice is28.00% /43.75%. Versus20, full300 overall falls
2.33pp and valid-only1.64pp, with227→219 valid tasks. Historical baseline30 is
32.00% /38.71% with248 valid tasks; differing dates and validity sets prevent
attributing this difference solely to training. Native29 restoration, all300
archives/verdicts and final W&B history passed independent checks. The controller
passed TP2/DP4 model-and-optimizer restoration at436 Adam updates for the next
training stage; the target remains60 with milestone evaluations.

Saved reweight30 abort diagnostics contain27 browser-reset errors,48
`terminate_reason=env_step_error` records, five600-second task timeouts and one
generation error with an empty message.34 invalid attempts have only one turn.
These identify failure locations, not whether the policy, website or browser
caused each failure. No job-wide CUDA OOM or storage failure appeared in the logs;
the scientific protocol and all invalid attempts remain unchanged in the results.

Fixed100 is extracted from the original frozen task-ID manifest without new
rollouts. There are300 saved rollout/verdict pairs per cohort, including invalid
attempts; validity denominators differ. Reweight20 is5.00pp above its10 checkpoint
on overall success. Historical outcome-only20 is31.67% /40.95% (95 successes,
232 valid), giving a descriptive0.67pp overall difference. Baseline10 is23.33%
/29.91% (70 successes,234 valid). Historical evaluations have different dates;
these single-run comparisons do not establish significance. Bonus20 is30.67%
overall,1.00pp below historical baseline20 and1.67pp below reweight20. Its overall
rate is unchanged from10, with nine fewer valid tasks. One valid failed task has
combined reward−1; task success is computed from saved binary verdicts rather
than mean combined reward.

Runtime roots: `evaluations/arm-mixed-bonus-iter10-335699/`,
`evaluations/arm-mixed-bonus-iter20-335699/`,
`evaluations/arm-mixed-bonus-iter30-335699/`,
`evaluations/arm-mixed-reweight-iter10-335697/`,
`evaluations/arm-mixed-reweight-iter20-335697/`, and
`evaluations/arm-mixed-reweight-iter30-335697/`, and
`evaluations/arm-mixed-reweight-iter40-335698/`. Separate W&B evaluations are
in `openwebrl-evals`: [bonus10](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-mixed-bonus-iter10-335699),
[bonus20](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-mixed-bonus-iter20-335699),
[bonus30](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-mixed-bonus-iter30-335699),
[reweight10](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-mixed-reweight-iter10-335697),
[reweight20](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-mixed-reweight-iter20-335697),
[reweight30](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-mixed-reweight-iter30-335697),
[reweight40](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-mixed-reweight-iter40-335698).
[Bonus10 aggregate audit](arm_results/rl_integration/mixed-bonus-iteration10-audit.json) ·
[Bonus20 aggregate audit](arm_results/rl_integration/mixed-bonus-iteration20-audit.json) ·
[Bonus30 aggregate audit](arm_results/rl_integration/mixed-bonus-iteration30-audit.json) ·
[Reweight10 aggregate audit](arm_results/rl_integration/mixed-reweight-iteration10-audit.json) ·
[Reweight20 aggregate audit](arm_results/rl_integration/mixed-reweight-iteration20-audit.json) ·
[Reweight30 aggregate audit](arm_results/rl_integration/mixed-reweight-iteration30-audit.json) ·
[Reweight40 aggregate audit](arm_results/rl_integration/mixed-reweight-iteration40-audit.json).

<a id="arm-original-backfill-results-20260927"></a>
## Original-bonus full300 backfills — September27

Jobs332476–332480 all completed with exit0. Native GPU restoration, checkpoint
lineage, exact full300 membership and lossless ZIP archives were independently
verified against all saved per-task verdicts. Protocol: local browser,
GPT-4.1/action_history, temperature0,30-step horizon. Actor only at evaluation;
these checkpoints were trained with the original mixed-group ARM bonus.

| Iteration | Successes | Valid /300 | Overall | Valid-only | Fixed100 successes /valid | Fixed100 overall /valid-only | Job |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 20 | 84 | 227 | 28.00% | 37.00% | 25/70 | 25.00% /35.71% | [332476](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-original-iter20-backfill-332476-retry1) |
| 30 | 98 | 225 | 32.67% | 43.56% | 30/67 | 30.00% /44.78% | [332477](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-original-iter30-backfill-332477-retry1) |
| 40 | 96 | 224 | 32.00% | 42.86% | 31/69 | 31.00% /44.93% | [332478](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-original-iter40-backfill-332478-retry1) |
| 50 | 102 | 236 | 34.00% | 43.22% | 32/72 | 32.00% /44.44% | [332479](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-original-iter50-backfill-332479-retry3) |
| 60 | 107 | 234 | 35.67% | 45.73% | 34/73 | 34.00% /46.58% | [332480](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-original-iter60-backfill-332480-retry1) |

Iteration40 is the exact union of100 historical tasks from299156 and200 new
disjoint tasks from332478; it is not a same-day300 rerun. The other four rows
are fresh300 evaluations. Fixed100 slices above are computed from these full
cohorts; the summary retains the earlier standalone fixed100 results at20/30.
Exact50 (`iter_0000049`) remains distinct from historical51. Invalid outcomes
remain in the overall denominator and are excluded only from valid-only.

Against historical outcome-only results, overall deltas at20/30/40/50/60 are
−3.67,+0.67,−1.33,−1.00,+0.67 percentage points. This is a descriptive comparison
across dates and differing valid-task sets, with no consistent gain established.

All1,400 new task archives and verdicts plus100 reused pairs are retained.
Including every startup retry, recorded usage is7.466 H200 GPU-hours versus10
approved. Every batch exited after verification; no allocation remained idle.
The persistent backfill watcher independently verified all five completions.
[Aggregate audit](arm_results/rl_integration/original-backfill-20260927.json) ·
[startup recovery and accounting](RL_RUNTIME.md#arm-evaluation-backfill-20260927).
Private output directories below each have `rollouts/`, `status.json`,
`full300-audit.json`, and `checkpoint_restore_evidence.json`:

- runtime`evaluations/arm-original-iter20-332476-retry1/`
- runtime`evaluations/arm-original-iter30-332477-retry1/`
- runtime`evaluations/arm-original-iter40-332478-retry1/`
- runtime`evaluations/arm-original-iter50-332479-retry3/`
- runtime`evaluations/arm-original-iter60-332480-retry1/`

<a id="arm-gate-b-iter100-results-20260929"></a>
## Gate B iteration100 — September29

Job336893 completed training through100 and its full300 evaluation, then exited
successfully at12:57 PDT after1h58m14s, releasing its eight GPUs. The durable
checkpoint is native `iter_0000099`, with1,246 Adam updates; shard byte sizes,
metadata, optimizer/scheduler alignment and the task cursor passed inspection.
The evaluation restored that exact checkpoint on GPU. This completes the
authorized historical Gate B target100; no continuation beyond100 is scheduled.

| Cohort / method | Successes | Valid | Invalid | Overall | Valid-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate B100, full300 |110 |225 |75 |36.67% |48.89% |
| Gate B100, fixed100 slice |30 |68 |32 |30.00% |44.12% |
| Historical outcome-only100, full300 |104 |227 |73 |34.67% |45.81% |
| Historical additive100, full300 |109 |217 |83 |36.33% |50.23% |
| Gate B90, full300 |129 |234 |66 |43.00% |55.13% |

Gate B falls6.33 percentage points overall and6.24 points valid-only from90.
At100 it is2.00 points above historical outcome-only overall, and0.33 above
additive; these are descriptive comparisons across collection dates, not a
demonstrated treatment effect. Iteration90 remains its best evaluated checkpoint.

All300 task IDs match the frozen cohort, and all300 nonempty rollout archives
and task records are preserved. The fixed100 slice uses the original unchanged
manifest without new browser attempts. There are225 valid terminal verdicts and
75 invalid attempts; invalid attempts remain in the overall denominator. The
judge identities, YAML, executed command and source retain local browsers,
GPT-4.1/action_history, actor T0,4096 response tokens and30 turns. This is separate
from the ongoing o4-mini/T0.6 stealth evaluation at90.

W&B's finished evaluation history matches110/225/300 and uses `openwebrl-evals`;
the training lineage in `openwebrl` is finished. The earlier80/90 cohorts remain
preserved. Total time charged to the to100 budget is60,389 of64,095 seconds,
including both predecessor attempts;3,706 seconds were unused when the endpoint
completed. Saved runtime results: `evaluations/arm-gate-b-iter100-336893/`.
[W&B evaluation](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-gate-b-iter100-336893)
· [Independent aggregate audit](arm_results/rl_integration/gate-b-iteration100-audit.json).

<a id="arm-gate-b-iter90-results-20260928"></a>
## Gate B iteration 90 — September 28

The full300 evaluation inside job335729 completed with exit0, then its controller
resumed training toward100. Native `iter_0000089` corresponds to training
iteration90 and1,136 accumulated Adam updates. The checkpoint's shard sizes,
metadata, cursor and optimizer/scheduler alignment passed inspection, and the
evaluation log confirms native checkpoint89 restoration.

| Cohort | Successes | Valid | Invalid | Overall | Valid-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full300 | 129 | 234 | 66 | 43.00% | 55.13% |
| Fixed100 slice | 39 | 71 | 29 | 39.00% | 54.93% |
| Historical outcome-only90, full300 | 101 | 222 | 78 | 33.67% | 45.50% |

All300 unique task IDs match the frozen full300 cohort. Each has a nonempty
ZIP rollout archive and saved judge verdict, including invalid attempts;
there are no top-level error records. Fixed100 is extracted using the original
frozen task-ID manifest, with no additional rollouts. Protocol: actor-only
inference, local browsers, GPT-4.1/action_history, temperature0,4096 response
tokens,30 browser turns. The frozen evaluation source is identical to Gate B80;
protocol-related environment settings match apart from the output directory.

Gate B90 exceeds its80 overall result by7.33 percentage points and historical
baseline90 by9.33 points. Different dates, valid-task sets and a single training
seed limit interpretation; these are descriptive differences, not evidence of
statistical significance. Baseline90 remains the historical control.

Runtime root: `evaluations/arm-gate-b-iter90-335729/`, with per-task files under
`rollouts/`, plus `status.json`, `metrics.json` and
`checkpoint_restore_evidence.json`. Evaluation command, environment and W&B
manifest all target `openwebrl-evals`; training remains in `openwebrl`.
[W&B evaluation](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-gate-b-iter90-335729) ·
[Aggregate audit](arm_results/rl_integration/gate-b-iteration90-audit.json).

<a id="arm-gate-b-iter80-results-20260927"></a>
## Gate B iteration80 — September27 PDT / September28 UTC, 2026

Job334894 completed the full300 evaluation with exit0 and handed off to
checkpoint80 restoration for continuation toward90 in the same allocation.
The checkpoint contains1024 Adam updates; native GPU restoration loaded
zero-based79. Protocol: local browser, GPT-4.1/action_history, temperature0,
actor-only evaluation,30-step horizon.

| Cohort | Tasks | Successes | Valid | Invalid | Overall % | Valid-only % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full300 | 300 | 107 | 226 | 74 | 35.67 | 47.35 |
| Fixed100 slice | 100 | 34 | 71 | 29 | 34.00 | 47.89 |

Overall is1.67pp below B70 and2.33pp below the historical outcome-only baseline80
(38.00% overall /49.78% valid-only). These differing-date web evaluations do not
establish a causal difference. No significance claim is made.

All300 unique expected task IDs, nonempty rollout ZIP archives and saved
per-task verdicts passed verification; there are zero exception records.
ZIP central directories were checked without loading tensor/image payloads.
The fixed100 slice matches the original unchanged sample manifest. All74
invalid tasks remain in the overall denominator; valid-only uses226 tasks.

[Aggregate audit](arm_results/rl_integration/gate-b-iteration80-audit.json) ·
[W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-gate-b-iter80-334894).
Artifacts:`evaluations/arm-gate-b-iter80-334894/rollouts/`.
This evaluation is embedded in the training allocation and retains the
`openwebrl` training-project convention.

<a id="arm-gate-b-iter70-results-20260927"></a>
## Gate B iteration70 — September27, 2026

Job331778 completed its iteration70 evaluation and continued training toward80.
The checkpoint contains914 Adam updates; native restoration loaded zero-based69.
Protocol: local browser, GPT-4.1/action_history, temperature0, actor-only.

| Cohort | Tasks | Successes | Valid | Invalid | Overall % | Valid-only % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full300 | 300 | 112 | 229 | 71 | 37.33 | 48.91 |
| Fixed100 slice | 100 | 35 | 70 | 30 | 35.00 | 50.00 |

Overall performance is unchanged from B60. Historical outcome-only baseline70
scores34.33%/44.98%, a descriptive+3.00pp overall difference; collection dates
and valid-task sets differ, so this is not an established causal improvement.

Verified all300 unique expected task IDs,300 nonempty rollout ZIP archives and
300 saved per-task verdicts, with zero exception records. The fixed100 cohort
matches the original manifest; every ZIP central directory was checked without
loading tensor/image payloads. The71 invalid outcomes remain in the overall
denominator. Collection completed September26 PDT /September27 UTC.

[Aggregate audit](arm_results/rl_integration/gate-b-iteration70-audit.json) ·
[W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-gate-b-iter70-331778).
Artifacts:`evaluations/arm-gate-b-iter70-331778/rollouts/`.

<a id="arm-failure-coverage-iter20-results-20260926"></a>
## Failure sampling40% iteration20 — September26, 2026

Job329911 completed training through20/302 Adam updates and both scheduled
full300 evaluations, then exited. Native restoration loaded zero-based19.
Protocol remains local browser, GPT-4.1/action_history, temperature0, actor-only.

| Cohort | Tasks | Successes | Valid | Invalid | Overall % | Valid-only % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full300 | 300 | 83 | 222 | 78 | 27.67 | 37.39 |
| Fixed100 slice | 100 | 26 | 68 | 32 | 26.00 | 38.24 |

All300 unique expected tasks, saved ZIP trajectory archives and judge verdicts
were verified, with no exception records. The78 invalid browser outcomes remain
in the overall denominator. This does not show an improvement over historical
additive iteration20 (28.33%/36.02%) or baseline20 (31.67%/40.95%); evaluation
dates and valid-task sets differ, so these are descriptive comparisons.

[Aggregate audit](arm_results/rl_integration/failure-coverage-iteration20-audit.json)
· [W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-failure-coverage-iter20-329911).
Artifacts:`evaluations/arm-failure-coverage-iter20-329911/rollouts/`.

<a id="arm-failure-coverage-iter10-results-20260926"></a>
## Failure sampling40% iteration10 — September26, 2026

Job329911 finished the iteration10 full300 evaluation after completing training
through20. This checkpoint has156 Adam updates; the native restoration receipt
loaded zero-based iteration9. Protocol: local browsers, GPT-4.1/action_history,
temperature0, actor-only inference.

| Cohort | Tasks | Successes | Valid | Invalid | Overall % | Valid-only % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full300 | 300 | 87 | 233 | 67 | 29.00 | 37.34 |
| Fixed100 slice | 100 | 32 | 76 | 24 | 32.00 | 42.11 |

All300 expected unique task IDs,300 nonempty rollout ZIP archives and300 saved
verdicts were verified. No exception records were present;67 invalid browser
trajectories remain in the overall denominator. Fixed100 uses the original
frozen task IDs sliced from full300. Iteration20 evaluation follows in the
same allocation; this early result does not establish an improvement.

[Aggregate audit](arm_results/rl_integration/failure-coverage-iteration10-audit.json)
· [W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-failure-coverage-iter10-329911).
Artifacts: `evaluations/arm-failure-coverage-iter10-329911/rollouts/`.

<a id="arm-gate-b-iter50-60-results-20260926"></a>
## Gate B iteration50/60 results — September26, 2026

Job329908 completed training through iteration60 and both full300 evaluations
(September25 PDT / September26 UTC). Local browser, GPT-4.1/action_history,
temperature0; fixed100 is extracted from the same saved full300 verdicts.

| Checkpoint | Adam updates | Cohort | Successes | Valid | Invalid | Overall % | Valid-only % |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| B50 | 676 | full300 | 113 | 232 | 68 | 37.67 | 48.71 |
| B60 | 798 | full300 | 112 | 228 | 72 | 37.33 | 49.12 |
| B50 | 676 | fixed100 | 33 | 73 | 27 | 33.00 | 45.21 |
| B60 | 798 | fixed100 | 35 | 71 | 29 | 35.00 | 49.30 |

Historical outcome-only baseline50/60 both score35.00% overall; B is higher
by2.67pp /2.33pp. Different collection dates and valid-task sets prevent a
causal or statistically established improvement claim.

Verified300 expected unique task IDs,300 nonempty rollout ZIP archives and300
saved per-task verdicts per checkpoint, no exception records, original fixed100
membership, checkpoint identity, and native restoration receipts. ZIP central
directories were checked without loading model or trajectory tensors.

Sources: [B50 audit](arm_results/rl_integration/gate-b-iteration50-audit.json),
[B60 audit](arm_results/rl_integration/gate-b-iteration60-audit.json),
[B50 W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-gate-b-iter50-329908),
[B60 W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-gate-b-iter60-329908).
Artifacts: `evaluations/arm-gate-b-iter{50,60}-329908/rollouts/`.

<a id="arm-failure-beta1-iter20-results-20260925"></a>
## Failure β1 iteration20 evaluation — September25, 2026

Job329912 finished both evaluations and released its allocation. The iteration20
checkpoint has298 Adam updates; native restoration loaded zero-based iteration19.
Protocol: local browsers, GPT-4.1/action_history, T0, actor-only inference.

| Cohort | Tasks | Successes | Valid | Invalid | Overall % | Valid-only % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full300 | 300 | 99 | 237 | 63 | 33.00 | 41.77 |
| Fixed100 slice | 100 | 32 | 76 | 24 | 32.00 | 42.11 |

All300 expected unique task IDs, nonempty ZIP rollout archives and verdicts
were verified, with zero exception records. The fixed100 uses the original
frozen IDs sliced from this evaluation. The aggregate improved from iteration10's
27.33% overall to33.00%; this does not establish a gain over a matched baseline.
The approved continuation330278 remains capped at40, with evaluations30/40.

[Aggregate audit](arm_results/rl_integration/failure-beta1-iteration20-audit.json)
· [W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-failure-weight-iter20-329912).

<a id="arm-failure-beta1-iter10-results-20260925"></a>
## Failure β1 iteration10 recovered evaluation — September25, 2026

Job329912 completed only the44 missing tasks from the quota-interrupted
job318949 evaluation, then merged them with256 preserved results. No completed
task was rerun. Both attempts restored the same native iteration10 checkpoint
(154 Adam updates) and used local browsers, GPT-4.1/action_history and T=0.

| Cohort | Tasks | Successes | Valid | Invalid | Overall % | Valid-only % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original saved portion | 256 | 76 | 207 | 49 | 29.69 | 36.71 |
| Missing-task recovery | 44 | 6 | 36 | 8 | 13.64 | 16.67 |
| **Merged full300** | **300** | **82** | **243** | **57** | **27.33** | **33.74** |
| Fixed100 slice | 100 | 27 | 76 | 24 | 27.00 | 35.53 |

Verified all300 expected distinct task IDs,300 nonempty ZIP rollout archives
and300 verdicts, with no exception records. The merged directory links the
original artifacts; their payloads were not overwritten or duplicated. The two
collection windows were separated by several hours on September25. Treat this
as a completed checkpoint evaluation with recovery provenance, not a same-time
paired comparison or a comparison between the two disjoint task portions.

Sources: [aggregate audit](arm_results/rl_integration/failure-beta1-iteration10-audit.json)
and [recovery W&B run](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-failure-weight-iter10-329912).
The W&B run reports the44 recovered tasks; the full300 score is the audited
aggregate above. Runtime aggregate:
`evaluations/arm-failure-weight-iter10-329912-merged/`.
Job329912 also completed the [iteration20 full300 evaluation](#arm-failure-beta1-iter20-results-20260925) in the same allocation.

<a id="arm-gate-c-iter50-60-results-20260925"></a>
## Gate C iteration50/60 results — September25, 2026

Job318935 completed training through iteration60 /782 Adam updates and both
full-300 evaluations. Protocol: local browser, GPT-4.1 action-history judge,
temperature0. Fixed100 is extracted from the same saved full300 tasks.

| Checkpoint | Adam updates | Cohort | Successes | Valid | Invalid | Overall % | Valid-only % |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| C50 | 668 | Full300 | 101 | 232 | 68 | 33.67 | 43.53 |
| C60 | 782 | Full300 | 98 | 227 | 73 | 32.67 | 43.17 |
| C50 | 668 | Fixed100 | 31 | 75 | 25 | 31.00 | 41.33 |
| C60 | 782 | Fixed100 | 30 | 73 | 27 | 30.00 | 41.10 |

Historical baseline50 is35.00% /44.87% and baseline60 is35.00% /45.65%.
C is below these by1.33 /1.34 pp and2.33 /2.48 pp respectively. These are
descriptive comparisons across evaluation dates and different valid-task sets,
not evidence of a statistically established decline or gain.

Verified all300 expected unique task IDs,300 nonempty ZIP rollout archives and
300 per-task JSON verdicts for each checkpoint. ZIP central-directory checks
passed; no task exception records or malformed verdict JSON were present.
Native model restoration receipts and complete status files are saved. Invalid
browser trajectories are counted separately from missing/exception records.

Sources: [C50 audit](arm_results/rl_integration/gate-c-iteration50-audit.json),
[C60 audit](arm_results/rl_integration/gate-c-iteration60-audit.json),
[C50 W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-gate-c-iter50-318935),
[C60 W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-gate-c-iter60-318935).
Runtime artifacts are under `evaluations/arm-gate-c-iter{50,60}-318935/`.

Beta1's separate iteration10 evaluation stopped at256/300 tasks because of disk
quota; it has no completed full300 score. Its256 rollout archives and verdicts
are preserved, leaving44 tasks to recover. Its iteration20 evaluation has not
started. See [the current recovery record](RL_RUNTIME.md#arm-status-20260925-morning).

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
