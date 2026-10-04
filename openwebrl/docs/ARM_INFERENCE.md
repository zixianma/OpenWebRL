# ARM inference and judge protocol

Inference-time ARM selection, terminal-success judge alignment, and unavailable-task retry policy. Initial benchmark results and retries retain separate denominators. See [all experiment results](ARM_RESULTS.md) for comparison with standalone training.

## Contents

- [Inference cost versus episode pass@k](#arm-inference-cost-passk-20261004)
- [ARM inference results on Online-Mind2Web](#arm-inference-results)
- [ARM judge alignment audit](#arm-judge-alignment)
- [ARM matched retry status and results](#arm-inference-retry-results)

---

<a id="arm-inference-cost-passk-20261004"></a>
## Inference cost versus episode pass@k — October4

**The historical benchmark establishes a gain over one ordinary episode. It
does not establish a gain over five complete episodes at comparable compute.**
We reconstructed saved candidate-text costs and separately audited the existing
five-episode screening data. [Aggregate audit](arm_results/rl_integration/inference-cost-passk-20261004.json).
This analysis made no new model calls and launched no GPU work.

![Historical ARM output-cost proxy and separate empirical episode pass@k curve](arm_results/rl_integration/inference-cost-passk-20261004.png)

Standalone figure: [PNG](arm_results/rl_integration/inference-cost-passk-20261004.png)
or [SVG](arm_results/rl_integration/inference-cost-passk-20261004.svg).

### Historical matched task list:300 Online-Mind2Web tasks

Original SFT actor, temperature0.7/top-p0.9,30-turn horizon, local browser,
o4-mini/AgentTrek judge. Each method has one final result per task.

| Method | Successes /300 | Overall | Valid-only | Saved decisions | Actor responses | Saved actor-text token proxy | Ratio to baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One actor sample per turn |90|30.00%|90/267 =33.71%|4,698|4,698|1,521,052|1.00×|
| ScalarRM, five candidates |114|38.00%|114/251 =45.42%|4,638|23,190|7,869,942|5.17×|
| SelectionARM, five candidates |128|42.67%|128/256 =50.00%|4,662|23,310|7,687,807|5.05×|

The proxy retokenizes every saved candidate's reasoning plus action, including
unselected candidates, with the original actor tokenizer. It excludes stripped
delimiters and requests with no saved trace. It is **not** exact generated-token
usage, FLOPs, GPU-hours, dollars, or latency. Baseline traces contain19 repeated
task/turn keys from resumed work; their logged costs are retained. None occur
for scalar or selection. Initial and resumed execution had different parallel
settings, and actor/ARM shared the GPU, preventing a clean per-method wall-time
or GPU-hour reconstruction from these artifacts.

Selection adds4,662 saved selector decisions; scalar scores five candidates
per decision. Selector prefill, vision processing, actor prefill/cache behavior,
and unsaved failed requests are additional costs. Saved decision counts are
almost unchanged across methods, so these logs show no substantial reduction
in path length offsetting five-way candidate generation. The12.67-point ARM
gain therefore comes with roughly five times the recorded actor-text output
and additional selector work.

### Empirical episode scaling on the separate2,000-task training pool

This is a **different protocol and population**: original SFT actor,
temperature0.8/top-p1,15-turn horizon, GPT-4.1/action-history judge. All10,000
attempts are present;9,936 are valid. Treat an invalid attempt as no success in
the primary all-scheduled metric. For each task with c successes among n=5
episodes, average `1 − C(n−c,k)/C(n,k)` over tasks. This is the standard
finite-sample [pass@k estimator](https://arxiv.org/abs/2107.03374); k<5 means
uniformly sampling a subset of the observed five attempts.

| Complete-episode budget | Overall pass@k | Expected actor output tokens/task | Expected browser steps/task |
| --- | ---: | ---: | ---: |
|1|35.970%|3,660|11.39|
|2|49.285%|7,321|22.78|
|3|56.540%|10,981|34.17|
|4|61.280%|14,642|45.55|
|5|64.650%|18,302|56.94|

These are metered output-token costs, averaged over random k-subsets and paying
for all k episodes, without early stopping. Exactly1,293/2,000 tasks have at
least one success. Among the1,956 tasks with five valid attempts, pass@5 is
65.13%; this is a task-filtered diagnostic, not a per-trajectory valid-only rate.
Success-count histogram for c=0,1,2,3,4,5 is707,337,274,229,240,213 tasks.

Do not compare64.65% directly with historical ARM42.67%: task pool, temperature,
horizon, judge, and collection time differ. Likewise `1 − (1 − 0.30)^5 =83.19%`
is not an estimate of historical OM2W pass@5. It assumes a common independent
success probability and discards task difficulty. On the actual screening
data, the same shortcut using mean35.97% predicts89.24%, far above observed
64.65%. Repeated tasks, not an aggregate pass@1 number, are needed.

### What we already know about ARM versus retries

The [eight-task randomized retry pilot](ARM_RESULTS.md#arm-rescue-yield-313264)
used an outcome-only iteration90 actor and failure-selected training tasks.
It directly compares fresh attempts after the screen:

| Retry strategy | Tasks rescued | Metered actor output tokens |
| --- | ---: | ---: |
| One ordinary retry |1/8|19,137|
| Five ordinary retries, any success |3/8|119,630|
| One ARM-guided episode |0/8|200,148|

ARM cost1.67× the five retries in output tokens, plus selection. This is small,
conditional evidence against a rescue advantage in that panel; it is not a
powered original-SFT OM2W test or an exact compute match.

The newer682-task guided collection is larger:78 rescues (11.44%),679 valid,
15,636,144 actor output tokens,47,185 actor requests, and9,437 browser
steps/selector calls. Its *earlier* five failures used14,953,843 tokens and
47,003 browser steps. Thus guided collection had approximately the same actor
generation count and one fifth the browser steps, with extra selector compute.
But the earlier failures **selected the cohort**: zero-versus78 is not a valid
estimate of ARM's gain over a fresh actor retry budget. That control is missing.

As an observed adaptive collection policy, five actor episodes on all2,000
tasks followed by one guided episode on682 eligible failures covers1,371/2,000
tasks (68.55%), versus64.65% before the follow-up. It adds42.72% actor output
tokens (36.604M →52.240M total), plus selectors. Extra ordinary retries might
also add coverage; this does not identify the value of ARM selection itself.

### Fair next comparison and cost interpretation

| Quantity | Five candidates per action, one guided episode | Five independent complete episodes |
| --- | --- | --- |
| Actor decode |Approximately5L responses, with guided path length L |Sum of the five ordinary path lengths |
| Browser execution |Approximately L real steps |Approximately5L steps if lengths match |
| Critic cost |Selection at each real step |No action critic; final episode selection/verification may cost extra |
| Exploration |One committed prefix; local alternatives are not executed |Five distinct full trajectories |
| Headline outcome |Success of the single executed episode |Any successful episode: an oracle coverage metric |

Candidate batches may share actor prefixes; independent episodes may run in
parallel. GPU sharing, browser latency, selected path length and selector
prefill determine the actual crossover. Five unexecuted choices per turn are
not an evaluated search over5^T complete trajectories. Post-action candidate
ranking would require actual candidate branches and change this cost model.

Freeze one SFT actor and one task cohort, then interleave/randomize one guided
episode and five fresh ordinary episodes per task under the same browser,
sampling, horizon and judge protocol. Recover the ordinary pass@1…5 curve and
compare guided success on a joint cost/performance plot. Meter actor and
selector input/output tokens, GPU service time, browser steps/time, API costs,
failed attempts, and end-to-end median/p95 latency; report all-scheduled and
common-valid paired outcomes with task-clustered intervals. Log actual seeds,
candidate responses, selected indices and verdict artifacts. After a measured
pilot, prespecify resource-matched budgets; five retries alone do not guarantee
equal compute.

Also distinguish **oracle pass@k** from a deployable episode best-of-k policy:
if only one final artifact can be returned, a verifier/selector must pick it,
and its accuracy and cost count. On live tasks that change browser/account
state, retries also require an appropriate reset protocol. The prepared
[independent task-selection control](ARM_INTEGRATION_PLAN.md#arm-selection-control-20261003)
addresses the conditional rescue question; a fresh common-cohort OM2W control
addresses inference scaling. Neither new allocation is approved by this audit.

<!-- document:ARM_INFERENCE_RESULTS.md:start -->
<a id="arm-inference-results"></a>
## ARM inference results on Online-Mind2Web

_Source record: `ARM_INFERENCE_RESULTS.md`. Dated entries retain their historical context._


[ARM results dashboard](ARM_RESULTS.md#arm-results-dashboard)

Snapshot: **2026-09-08T08:17:46.840921+00:00**. All three original 300-task evaluations are complete. This is an inference comparison from the frozen `OpenWebRL/OpenWebRL-4B-SFT` actor, not an ARM-trained policy result.

<a id="arm-inference-results--completed-results"></a>
### Completed results

| Arm | Scheduled | Completed | Successes | Valid | Unavailable | Overall success | Valid-only success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, one candidate | 300 | 300 | 90 | 267 | 33 | **30.0% (90/300)** | **33.7% (90/267)** |
| ScalarRM-LoRA, best of five | 300 | 300 | 114 | 251 | 49 | **38.0% (114/300)** | **45.4% (114/251)** |
| SelectionARM, best of five | 300 | 300 | 128 | 256 | 44 | **42.7% (128/300)** | **50.0% (128/256)** |

Overall success = successes / all 300 scheduled tasks; unavailable outcomes contribute no success. Valid-only success = successes / valid judged outcomes. Report both; excluding failures changes the task population, and missingness can depend on the arm.

ScalarRM improves overall success by **8.0 percentage points**, or **24 additional successful tasks**. The difference between the separate valid-only rates is **11.7 points**, but those denominators contain different tasks.

<a id="arm-inference-results--paired-comparison-on-tasks-valid-in-both-arms"></a>
### Paired comparison on tasks valid in both arms

| Metric | Value |
| --- | ---: |
| Common valid tasks | 244 |
| Baseline successes | 88/244 = **36.1%** |
| ScalarRM successes | 113/244 = **46.3%** |
| Paired gain | **10.25 percentage points** |
| Paired task-bootstrap 95% interval | **[4.10, 16.39] points** |
| Scalar-only successes / baseline-only successes | 43 / 18 |
| Exact two-sided McNemar p-value | 0.00187 |

This supports an inference gain in this run. The interval conditions on the 244 commonly evaluable tasks and does not measure seed, judge, website, or date variability. It does not establish standalone-policy learning or exact reproduction of historical README rates.

<a id="arm-inference-results--completed-selectionarm-result-and-c2-teacher"></a>
### Completed SelectionARM result and C2 teacher

SelectionARM completed **300/300**, with **128 successes**, **256 valid**, and **44 unavailable** outcomes. Its success rates are **42.7% overall** and **50.0% valid-only**. Relative to baseline, this is **+12.7 percentage points overall**; relative to ScalarRM, it is **+4.7 points overall**.

On the **236 tasks valid for both ScalarRM and SelectionARM**, SelectionARM has 12 additional successes, a paired gain of **5.08 percentage points** (paired bootstrap 95% interval **−1.27 to +11.86 points**; exact McNemar p = **0.169**). Full paired statistics, including discordant counts, are recorded in `full/selection-vs-scalar-paired.json`. The difference is a single-run estimate, not proof of superiority across seeds or website states.

The C2 teacher rule selected **SelectionARM** because it has both more successes on all 300 tasks and a positive common-valid difference. [C2 run configuration and status](ARM_SFT.md#arm-c2-run). The student has not yet been evaluated.

<a id="arm-inference-results--unavailable-outcomes-and-retry-decision"></a>
### Unavailable outcomes and retry decision

| Recorded failure category | Baseline | ScalarRM |
| --- | ---: | ---: |
| No turn-level samples / turn-index wrapper error | 22 | 26 |
| Actor HTTP 400 | 5 | 12 |
| Environment step error | 5 | 6 |
| Actor request timeout (180 seconds) | 1 | 3 |
| Whole-task timeout (1800 seconds) | 0 | 2 |
| Total unavailable | **33** | **49** |

The no-turn wrapper hides the originating exception; earlier log inspection identified navigation failures in most audited baseline cases. It is not an independent diagnosis. Observed actor HTTP 400 logs include requests exceeding the 32768-token context budget. Timeouts and context overflow may reflect trajectory length or candidate-generation cost, so unavailable outcomes are not automatically unrelated to model behavior.

**Current decision: retries remain held. SelectionARM is now complete; the user requested C2 next, and the retry cohort/arms still need a separate decision.** The following two-arm counts describe the earlier proposal, not an active queue:

- The unavailable-task union contains **56 tasks**: **26 unavailable in both**, **7 baseline-only**, and **23 scalar-only**.
- Run both arms once on each of those 56 tasks: **112 additional task attempts**. This includes repeating the previously valid counterpart, so the follow-up compares the same tasks in the same time window. A smaller alternative is retrying only each arm's own unavailable cases (82 attempts), but that is a recovery sensitivity analysis with asymmetric extra opportunities.
- First classify the originating failures. Retry transient website/network/browser failures with unchanged model, sampling, horizon, and timeout settings. Record persistent failures explicitly.
- A larger context budget or a new truncation policy is a protocol change, not an ordinary retry. Define it for both arms and report a separate experiment; an unchanged retry will not reliably cure a context-limit failure.
- Freeze the task cohort and one-attempt budget before inspecting follow-up outcomes. Use fresh browsers; balance or interleave arm order where serving permits. No repeated attempts until success, no selecting the best result across attempts, and no overwriting original result files.
- Keep the original table above as the primary report. Report retry-cohort coverage and paired outcomes separately. If assembling a 300-task sensitivity table, replace **both** arms' entries for the entire selected cohort using the predeclared retry attempt and label the table as a mixed-session sensitivity analysis.
- For a three-arm robustness comparison, use the completed unavailable sets to freeze a common cohort across **all three** arms and give every arm the same follow-up cohort and budget. C2 currently has execution priority.

A proposed two-arm task/index inventory is prepared at [retry-baseline-scalar-union.json](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/dedicated-282209-20260908T005547Z/retry-baseline-scalar-union.json). **The user clarified on 2026-09-08 that we should redecide retry queries after SelectionARM. The automatic launch is disabled, and no retry has started.** [Separate retry status/results](ARM_INFERENCE.md#arm-inference-retry-results). Reconcile all three unavailable sets and originating failures before deciding the final cohort, arms, and budget; keep the 56-task two-arm inventory as a provisional option.

<a id="arm-inference-results--protocol-provenance-and-artifacts"></a>
### Protocol, provenance, and artifacts

- Actor: `OpenWebRL/OpenWebRL-4B-SFT`, revision `15e777db2ddba2e0e82080ebccd3ad8d215b7f0a`.
- Scalar: `PTeterwak/OpenWebRL-4B-ScalarRM-LoRA`, revision `71c58489cd7cbaebcd74656df7ef11ba818661b8`.
- Selection: `PTeterwak/OpenWebRL-4B-SelectionARM`, revision `81b452d800d9f859687074f82680dd5257e02d89`.
- ARM source revision: `02276b0ff3b9048d34e6a2afdcb9042dd9018c8c`.
- Same 300 task IDs; temperature 0.7, top-p 0.9, seed 42 with deterministic task/turn/candidate derivation; 1024 generated tokens, 30 browser steps, one current screenshot with full action history; actor context limit 32768.
- Online-Mind2Web/AgentTrek outcome protocol with o4-mini. Candidate counts: one baseline, five per ARM. Constrained canonical no-CoT JSON schema for SelectionARM; scalar uses the released OpenWebRL serialization and BF16 last-token head.
- Baseline used task concurrency 8; both ARMs used 16. The actor and one ARM shared one H200. This comparison does not isolate concurrency effects or match inference compute.
- Baseline and the first 277 saved scalar outcomes came from allocation 282209 on g001. The remaining 23 scalar tasks resumed in allocation 282782 on g005; earlier interrupted partial action traces were archived. All 577 previously saved result files were hash-verified unchanged after resumption. SelectionARM runs on g005. Live-site time differences remain a confound.
- Historical README targets are 33.8%, 46.3%, and 51.1%; exact historical denominators/settings are not fully recovered. Our paired ScalarRM rate rounding to 46.3% is not evidence that the historical protocol matches. The [judge alignment audit](ARM_INFERENCE.md#arm-judge-alignment) confirms the README uses o4-mini and the dashboard describes AgentTrek, seed 42, and a final high-detail screenshot. It also documents the author's separate GPT-4.1 re-judging and decoding differences; our run is not an exact end-to-end reconstruction.

Run root: `/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/dedicated-282209-20260908T005547Z`.

- [Baseline summary](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/dedicated-282209-20260908T005547Z/full/baseline/summary.json), [ScalarRM summary](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/dedicated-282209-20260908T005547Z/full/scalar/summary.json), [paired statistics](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/dedicated-282209-20260908T005547Z/full/scalar-paired-comparison.json).
- Per-arm `results/`: task outcomes; `selections/`: action candidates and chosen indices; `samples/turn/`: rollout text and embedded screenshots; `browser_logs/`: environment diagnostics.
- [Integration plan](ARM_INTEGRATION_PLAN.md). No Online-Mind2Web evaluation rollout is designated as SFT training data.

<!-- document:ARM_INFERENCE_RESULTS.md:end -->

---

<!-- document:ARM_JUDGE_ALIGNMENT.md:start -->
<a id="arm-judge-alignment"></a>
## ARM judge alignment audit

_Source record: `ARM_JUDGE_ALIGNMENT.md`. Dated entries retain their historical context._


Checked 2026-09-08 PDT following the user's question about GPT-4.1 versus o4-mini. **All three completed inference arms and the C2 collection use o4-mini with the Online-Mind2Web AgentTrek outcome rubric.** SelectionARM chooses candidate actions; o4-mini supplies the separate trajectory label used by C2's success filter.

<a id="arm-judge-alignment--evidence-and-comparison"></a>
### Evidence and comparison

| Item | Author's reported setup | Our setup / conclusion |
| --- | --- | --- |
| OpenWebRL README outcome judge | o4-mini; reported baseline/ScalarRM/SelectionARM rates 33.8% / 46.3% / 51.1% | All three saved manifests specify o4-mini. The model matches. |
| AgentTrek rubric and inputs | Author's dashboard describes AgentTrek, seed 42, final screenshot at high detail | Same rubric, seed, and image selection/detail in our code. |
| Prompt text | OSU AgentTrek reference | User template is byte-identical; system text matches after removing trailing spaces on each line. Do not call the system prompt byte-identical. |
| Judging eligibility | Dashboard describes judging completed trajectories; failures/truncations receive zero without judging | Our reward implementation also calls the judge only for `Sample.Status.COMPLETED`. Our validity accounting separately excludes aborted outcomes and unavailable calls. |
| Decoding | README summarizes temperature 0.7; dashboard's September comparison describes 0.6 / top-p 0.95 / top-k 20 | Our experiment freezes 0.7 / top-p 0.9 / max-new-tokens 1024. Top-k is not explicitly pinned in the request. This is not an exact reconstruction of the dashboard run. |
| Environment and denominator | Historical self-hosted browser runs and per-arm/non-aborted or intersection denominators | Our self-hosted Slurm browser runs report both all 300 scheduled tasks and valid-only outcomes, plus paired intersections. Historical cohorts are not assumed identical. |

Sources: [pinned ARM README](https://github.com/piotr-teterwak/action-reward-models/blob/02276b0ff3b9048d34e6a2afdcb9042dd9018c8c/README.md), [author's dashboard](https://weekly-dashboard-inky.vercel.app/) → **9.1.26 → GPT-4.1 judge — paper-metric comparison**, and [OSU AgentTrek evaluator](https://github.com/OSU-NLP-Group/Online-Mind2Web/blob/main/src/methods/agenttrek_eval.py). The public ARM checkout does not ship its original OpenWebRL evaluation harness; the dashboard documents the protocol but does not permit a full implementation comparison.

The dashboard reports that GPT-4.1 scores identical trajectories about 4–9 percentage points higher, with 93.6% verdict agreement and unchanged arm ordering in that comparison. Those are the author's observations, not a re-judging of our rollouts. Changing judges can change C2 retention as well as reported evaluation success. Switching to GPT-4.1 would not better match the README's OpenWebRL row.

<a id="arm-judge-alignment--consequences-for-the-current-run"></a>
### Consequences for the current run

- Resume the existing C2 cohort with its frozen o4-mini/AgentTrek labels and decoding. Do not mix judges or decoding settings within one resumed cohort.
- Treat the completed experiment as evidence of an inference gain under our documented protocol, with the judge aligned to the author's report. Do not describe it as an exact end-to-end reproduction.
- C2 retains usable executed ARM winners only from valid trajectories judged successful. AgentTrek is permissive: its rubric allows partial completion in some cases. A successful label is an evaluator decision, not verified completion of every task requirement.
- A GPT-4.1 sensitivity audit would re-judge the same saved trajectories for **all three arms**, preserving original labels and reporting paired flips/coverage separately. No GPT-4.1 re-judging or browser retries were launched by this audit.

<a id="arm-judge-alignment--local-evidence"></a>
### Local evidence

- [Judge implementation](../eval/reward_online_mind2web.py): model call, prompts, parser, and completed-status gate.
- Original manifests: `.../arm-reproduction/runs/dedicated-282209-20260908T005547Z/full/{baseline,scalar,selection}/manifest.json`.
- Audit snapshots: `/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/judge-alignment-20260909/`, including `alignment.json`, OSU source, dashboard JavaScript, and the dashboard's August score data. The August score data are a separate historical cohort and do not exactly reproduce the README rates.
- Local judge source SHA-256: `b1f2c30c852b36d8049e23bd8ef4efb44a0786742449b5e7ef4e5a7ea55af583`.
- Retrieved dashboard JavaScript SHA-256: `d2560b741ac2153d8670df13fa0c47e550213d55c10ad7ada22e98cf77f68592`.

See [inference results](ARM_INFERENCE.md#arm-inference-results) and [C2 run record](ARM_SFT.md#arm-c2-run).

<!-- document:ARM_JUDGE_ALIGNMENT.md:end -->

---

<!-- document:ARM_INFERENCE_RETRY_RESULTS.md:start -->
<a id="arm-inference-retry-results"></a>
## ARM matched retry status and results

_Source record: `ARM_INFERENCE_RETRY_RESULTS.md`. Dated entries retain their historical context._


Status: **Held for a cohort decision after SelectionARM completes. No retries have started.** On 2026-09-08 the user clarified that the retry queries should be reconsidered using all three completed runs. The automatic two-arm continuation is disabled (`authorized: false`); the original report can finish normally without launching retries.

<a id="arm-inference-retry-results--decision-after-completed-selectionarm"></a>
### Decision after completed SelectionARM

**SelectionARM is now complete: 44 unavailable outcomes. The unavailable-task union across all three arms is 69 tasks**, including 13 SelectionARM cases outside the old 56-task two-arm union. A matched three-arm pass would require 207 task attempts. This is a prepared option, not an authorized launch; C2 currently has execution priority. Review originating failure categories before choosing a retry cohort. Decide which arms to compare, the common task cohort, and the one-attempt budget before any follow-up outcomes are observed. For a three-arm robustness comparison, use a matched cohort for all three arms. Distinguish transient browser/network failures from deterministic context-limit failures; changing context handling requires a separately labeled protocol.

The earlier baseline/ScalarRM proposal remains an inventory, **not the final scope**:

- 56 distinct tasks: union of 33 unavailable baseline and 49 unavailable ScalarRM evaluations, with 26 shared.
- One baseline and one ScalarRM attempt on every task would cost 112 task attempts, including previously valid counterparts.
- The completed SelectionARM set expands the three-arm union to 69 tasks. [Separate three-arm inventory](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/dedicated-282209-20260908T005547Z/retry-all-three-proposal.json).

<a id="arm-inference-retry-results--reporting-rules-retained"></a>
### Reporting rules retained

Preserve the original results. Store retry outcomes, rollouts, action traces, logs, and summaries separately; verify original result hashes. Keep models, seeds, decoding, context, horizon, judge, and timeouts fixed for an ordinary retry. Freeze a single-attempt policy, with no best-of-retries selection.

Report successes divided by all tasks in the agreed retry cohort, successes divided by valid retry outcomes, recovery of initially unavailable tasks, regression of previously valid counterparts, and paired outcomes on common valid tasks. These selected-cohort rates do not replace the original 300-task benchmark rates.

| Item | Current state |
| --- | --- |
| Final retry arms and task count | Pending decision; all three original evaluations are now complete |
| Retry attempts started / completed | 0 / 0 |
| Overall / valid-only retry success | Not available; no retry outcomes |
| Earlier two-arm inventory | 56 tasks per arm; held |

Original completed results remain **30.0% overall / 33.7% valid-only** for baseline and **38.0% overall / 45.4% valid-only** for ScalarRM; see [ARM_INFERENCE_RESULTS.md](ARM_INFERENCE.md#arm-inference-results).

<a id="arm-inference-retry-results--prepared-implementation-and-artifacts"></a>
### Prepared implementation and artifacts

- [Reserved retry directory](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/retry-baseline-scalar-282782-20260908T070733Z).
- [Held manifest](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/retry-baseline-scalar-282782-20260908T070733Z/queue-manifest.json).
- [Retry status](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/retry-baseline-scalar-282782-20260908T070733Z/retry-status.json).
- [Disabled continuation marker](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/dedicated-282209-20260908T005547Z/full/queued-retry.json).
- Controller: [run_arm_retry.py](../../scripts/run_arm_retry.py); report/hold hook: [summarize_arm_reproduction.py](../../scripts/summarize_arm_reproduction.py).

The prepared controller supports the previous two-arm proposal only. A changed cohort or inclusion of SelectionARM requires updating and checking the controller/manifest before execution. It cannot simply be unheld with new task IDs. While held, the launcher writes the original report and follows its normal server cleanup; there is no promise that the actor will remain resident for a later decision.

Available existing allocation: **282782** on **g005**, assigned GPU `GPU-90ac2a02-abfa-147c-19ff-bbada4033da3`, expires **2026-09-08 02:37:55 PDT**. No new allocation or extension is requested or authorized by this document.

Validation: **13 retry tests passed**, including the held-queue case, original-result preservation, separate reporting, resource identity, and quota handling. The 9 existing ARM contract tests passed before this hold-only change. No retry GPU task has started.

The project filesystem previously rejected Git object creation with `Disk quota exceeded`; source is also backed up on scrubbed storage. If a future report copy to the project encounters the same quota, the complete Markdown report remains in its scrubbed run directory, with the copy error recorded in JSON.

<!-- document:ARM_INFERENCE_RETRY_RESULTS.md:end -->

---

<a id="sol-selection300-retry-294221"></a>

### 2026-09-13: full Sol inference retry, job 294221

Explicitly approved and submitted **2 H200 × 2 hours**, 16 CPUs / 240 GiB,
with a **$200 Sol selector cap** and additional o4-mini judge usage. Job 294221
started on g020. Source-matched readiness verification passed; both actors
loaded and the live two-task pilot began. The pilot is retained in the full
300-task total, followed by two disjoint 149-task shards. This is an evaluation
of the original SFT actor with Sol best-of-five selection, not an intermediate
RL checkpoint evaluation. The earlier diagnostic completed 2/2 selected tasks
successfully, with ten five-candidate selections and zero fallbacks; this does
not establish benchmark accuracy.

[W&B evaluation run](https://wandb.ai/zixianma/openwebrl-evals/runs/sol-selection300-294221).
Local controller log: `/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-sol-selection-294221.out`.
Results, actor/pilot/shard logs, API usage and status:
`/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/sol-selection300-294221/`.
The approval receipt is runtime `sol_inference_approval_294221.json`.

A separate `scripts/sync_sol_inference_wandb.py --job-id 294221 --watch` process
publishes durable task records and API usage every 30 seconds when values change;
it does not change the GPU worker or diagnose/restart failures. The normal
monitor remains `scripts/monitor_sol_inference.py --job-id 294221`.
W&B system sampling is disabled for the logger because it runs on the login
host. Two CPU tests check the metric denominators and reject duplicate tasks.

| W&B metric | Calculation / meaning |
| --- | --- |
| `progress/completed_tasks` | Unique saved task results; evaluation chart x-axis |
| `eval/successes` | Valid saved tasks with reward 1 |
| `eval/valid_tasks`, `eval/invalid_tasks` | Completed task validity counts |
| `eval/success_rate_all_scheduled` | Successes / 300; lower bound while incomplete |
| `eval/success_rate_completed` | Successes / completed tasks |
| `eval/success_rate_valid` | Successes / valid completed tasks |
| `eval/complete`, `eval/failed` | Controller completion / failure flags |
| `selector/requests`, `selector/accounted_cost_usd` | Selector usage receipts; judge usage excluded |
| `selector/consecutive_failures` | Selector API failure streak |
| `progress/stage`, `progress/slurm_state` | Controller stage and allocation status |

<a id="sol-selection300-completed-294221"></a>

### Sol best-of-five: completed 300-task evaluation, September 13

Job **294221 completed with exit 0 in 55m29s**, using **1.8494 H200-hours**
of the approved four-GPU-hour cap. All 300 released task IDs are present exactly
once. Sol achieved **132/300 = 44.00% overall**, or **132/256 = 51.56% valid-only**,
with 44 unavailable tasks. W&B finished successfully; its final summary and
history match the durable task results. All 4,040 API requests returned
`gpt-5.6-sol` and passed validation. The 4,039 action-selection turns each used
five candidates, with **zero fallbacks**. Selector accounting totals **$84.8546892**;
o4-mini terminal-judge usage is additional.

| Historical / current condition | Successes / 300 | Valid | Unavailable | Overall % | Valid-only % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original SFT, no selector | 90 | 267 | 33 | 30.00 | 33.71 |
| ScalarARM, best-of-five | 114 | 251 | 49 | 38.00 | 45.42 |
| SelectionARM, best-of-five | 128 | 256 | 44 | 42.67 | 50.00 |
| GPT-5.6 Sol, best-of-five | 132 | 256 | 44 | 44.00 | 51.56 |

The historical controls retain the same actor, task set, sampling and judging
protocol, but were collected on a different date. Sol is only **four successes**
ahead of SelectionARM across all 300 tasks. On the **235 tasks valid in both**,
SelectionARM succeeded on 118 and Sol on 125: **+2.98 percentage points**,
paired bootstrap 95% interval **−3.40 to +9.79 points**, exact McNemar **p=0.4426**.
This does not establish that Sol is better than SelectionARM. Live-site/date
variation and differing unavailable subsets remain comparison limitations.

[Machine-readable comparison](arm_results/sol-selection300.json) ·
[W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/sol-selection300-294221).
The full task-set/API/W&B verification is runtime
`evaluations/sol-selection300-294221/completion-audit.json`; raw per-task results,
selection traces, API receipts and comparison are preserved in that directory.
