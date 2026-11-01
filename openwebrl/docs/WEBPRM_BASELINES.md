# WebPRM baselines

<a id="webprmbench-20261018"></a>
## WebPRMBench: Piotr SelectionARM and our K=1 head (October 17–18)

**On WebArbiter's offline reward benchmark, Piotr averages 70.4% pairwise and 38.2% Best-of-N accuracy over the four environments, 18.8 and 36.4 pp below the published WebArbiter-7B (89.2% and 74.6%); our K=1 student/RL head is at chance (51.8% pairwise).** Both runs cover the full released benchmark ([`ZYao720/WEBPRMBENCH`](https://huggingface.co/datasets/ZYao720/WEBPRMBENCH): 1,150 states × 4 preference pairs) with the unchanged upstream parser and metric, and were independently re-audited.

**Caveats that apply to every number below.** The released benchmark is text only (intent, accessibility tree, history, URLs; no screenshots), so neither model is evaluated in its native setting: Piotr was trained on screenshots, and the K=1 head on screenshots with five candidates (scored here on two). Neither model's training data was checked for overlap with the benchmark. Unparseable outputs count as errors (Piotr 44; K=1 0).

**Pairwise accuracy (%)**

| Model | Source of number | Mind2Web | WebArena | AssistantBench | WorkArena | Average of 4 |
| −-- | −-- | −--: | −--: | −--: | −--: | −--: |
| WebArbiter-7B | WebArbiter paper | 97.1 | 88.4 | 89.2 | 82.1 | 89.2 |
| WebArbiter-3B | WebArbiter paper | 93.3 | 82.0 | 78.3 | 81.0 | 83.7 |
| WebShepherd-8B | WebArbiter paper | 86.7 | 68.3 | 55.9 | 54.6 | 64.3 |
| WebShepherd-3B | WebArbiter paper | 87.5 | 68.2 | 66.7 | 50.0 | 68.1 |
| GPT-5 | WebArbiter paper | 80.9 | 84.8 | 81.7 | 81.1 | 82.1 |
| GPT-4o | WebArbiter paper | 80.0 | 84.6 | 85.8 | 84.3 | 83.7 |
| Qwen2.5-7B-Instruct | WebArbiter paper | 77.8 | 74.9 | 84.2 | 77.6 | 77.6 |
| **Piotr SelectionARM** | our run | **67.3** | **73.4** | **80.8** | **60.0** | **70.4** |
| **K=1 student/RL head** | our run | **44.5** | **60.4** | **48.3** | **54.1** | **51.8** |
| Gap: Piotr − WebArbiter-7B (pp) | | −29.8 | −15.0 | −8.3 | −22.1 | −18.8 |

**Best-of-N accuracy (%): the better response must win all 4 of its pairs (our "state accuracy")**

| Model | Source of number | Mind2Web | WebArena | AssistantBench | WorkArena | Average of 4 |
| −-- | −-- | −--: | −--: | −--: | −--: | −--: |
| WebArbiter-7B | WebArbiter paper | 89.5 | 68.7 | 70.0 | 70.2 | 74.6 |
| WebArbiter-3B | WebArbiter paper | 78.4 | 56.2 | 46.7 | 54.8 | 59.1 |
| WebShepherd-8B | WebArbiter paper | 73.7 | 43.9 | 30.0 | 25.5 | 43.3 |
| WebShepherd-3B | WebArbiter paper | 65.2 | 41.3 | 46.7 | 21.2 | 43.6 |
| GPT-5 | WebArbiter paper | 62.4 | 71.6 | 63.3 | 64.6 | 65.5 |
| GPT-4o | WebArbiter paper | 52.6 | 66.7 | 66.7 | 55.2 | 60.3 |
| Qwen2.5-7B-Instruct | WebArbiter paper | 39.2 | 42.8 | 53.3 | 35.9 | 42.8 |
| **Piotr SelectionARM** | our run | **28.0** | **44.3** | **53.3** | **27.4** | **38.2** |
| **K=1 student/RL head** | our run | **14.1** | **26.4** | **6.7** | **21.7** | **17.2** |
| Gap: Piotr − WebArbiter-7B (pp) | | −61.5 | −24.4 | −16.7 | −42.8 | −36.4 |

Published rows are from the [WebArbiter paper](https://arxiv.org/abs/2601.21872), Table 2; we did not rerun those models. Mind2Web is the three Mind2Web splits pooled (707 states), and "Average of 4" is the unweighted mean of the four environments, as in the paper. (The upstream evaluator's "official" state score of 35.5% for Piotr averages six sources instead, counting the Mind2Web splits separately; it is not comparable to the paper's average.)

**The gap:** Piotr is 18.8 pp below WebArbiter-7B on average pairwise accuracy and 36.4 pp below on Best-of-N, and also below the paper's general-purpose LLM judges: below GPT-4o in every column, and below Qwen2.5-7B-Instruct on both averages (70.4 vs 77.6 pairwise; 38.2 vs 42.8 Best-of-N). The largest gaps are on Mind2Web and WorkArena. Part of this is setting: WebPRMBench is text-only (accessibility tree), the native input of WebArbiter but not of Piotr, which was trained on screenshots.

<details><summary>Our runs by source, including the three Mind2Web splits (counts and rates)</summary>

| Model | Count | Mind2Web cross-task | Mind2Web cross-website | Mind2Web cross-domain | WebArena | AssistantBench | WorkArena | All |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Size | States (pairs = 4 ×) | 142 | 148 | 417 | 201 | 30 | 212 | 1,150 |
| Piotr SelectionARM | Correct pairs | 395 | 395 | 1,113 | 590 | 97 | 509 | 3,099 |
| Piotr SelectionARM | Pair accuracy (%) | 69.5 | 66.7 | 66.7 | 73.4 | 80.8 | 60.0 | 67.4 |
| Piotr SelectionARM | Correct states | 50 | 40 | 108 | 89 | 16 | 58 | 361 |
| Piotr SelectionARM | State accuracy (%) | 35.2 | 27.0 | 25.9 | 44.3 | 53.3 | 27.4 | 31.4 |
| K=1 student/RL head | Correct pairs | 281 | 285 | 692 | 486 | 58 | 459 | 2,261 |
| K=1 student/RL head | Pair accuracy (%) | 49.5 | 48.1 | 41.5 | 60.4 | 48.3 | 54.1 | 49.2 |
| K=1 student/RL head | Correct states | 26 | 26 | 48 | 53 | 2 | 46 | 201 |
| K=1 student/RL head | State accuracy (%) | 18.3 | 17.6 | 11.5 | 26.4 | 6.7 | 21.7 | 17.5 |
</details>

- **Piotr has a strong first-position bias.** It answers "Response 1" on 3,257 of 4,600 pairs (71%) although the label is 1 on only 2,178 (47%): pair accuracy is 91.0% (1,983 / 2,178) when the better response is shown first and 46.1% (1,116 / 2,422) when it is shown second. The K=1 head scores each candidate separately, so it is order-invariant by construction (measured maximum order effect 1.9e-6).

<details><summary>Setup, cost and sources</summary>

| | Piotr SelectionARM | K=1 student/RL head |
| --- | --- | --- |
| Checkpoint | `PTeterwak/OpenWebRL-4B-SelectionARM` @ `81b452d8` | `teacher_k1` seed 0 head (SHA-256 `2a0fa078…`) on the OpenWebRL actor; trained on 1,908 states (1,474 SFT + 434 RL) from 454 tasks |
| How it chooses | Generates text with WebArbiter's pairwise prompt and answer tag; T 0, max 1,024 tokens, stop `</Answer>`; SGLang 0.5.6.post2 | Actor encodes the prompt and each response; the listwise head scores the two candidates; higher score wins |
| Job | 367841 (after 367696 failed at startup, 24 s) | 369117 |
| GPU time | 1,726 s (0.48 H200-hours) incl. failed attempt | 393 s (0.11 H200-hours) |
| API spend | none | none |
| Results | [JSON](webprm_results/piotr_webprmbench_20261018.json) | [JSON](webprm_results/k1_sftrl_webprmbench_20261018.json) |
| Raw outputs and audit | `RT/webprm-webarena-20261011/next-stage-preparation/piotr-webprmbench-v1/package-v3/attempt-367841/` | `RT/webprm-webarena-20261011/next-stage-preparation/k1-sftrl-webprmbench-v1/package-v3/run/attempt-369117/` |
| Runner | `scripts/run_piotr_webprmbench.py`, `scripts/submit_piotr_webprmbench.py` | `scripts/run_selection_head_webprmbench.py` |

`RT = /gpfs/scrubbed/zixianma/openwebrl-runtime/`. The K=1 cached-feature path matched the full actor forward pass within 1.1% (2% gate).
</details>

<a id="table5-plan-20261010"></a>
## WebArbiter Table 5 plus Piotr SelectionARM

**Table 5 remains incomplete: all 12 claimed primary attempts are independently audited—eight valid and four invalid.** The eight valid episodes contain **two guarded successes and seven native positive verdicts**. The two new positives are GPT-4o no-search and GPT-4o-mini with Piotr SelectionARM on task 14; independent saved-evaluator replays match both original audits. This is partial coverage of 190 approved Reddit cells within the fixed 1,550-cell reconstruction, and does not establish a selector ranking or cohort success rate. [Results and accounting](webprm_results/webarena_preparation_20261011.json).

| Primary disposition | Cells | Native positive verdicts | Guarded successes |
| --- | ---: | ---: | ---: |
| Saved, audited and valid: task 5 | 6 | 5 | 0 |
| Saved, audited and valid: task 14 | 2 | 2 | 2 |
| Saved and audited; wrapper-invalid | 2 | — | — |
| Saved and audited; interrupted, terminal verdict absent | 2 | — | — |

**The third and final approved Reddit allocation, 356127, closed after 305 seconds with 178 cells still unclaimed.** The queue permanently closed admission after a resource check returned false while another worker remained active, then drained its two admitted trials. Its scheduler `COMPLETED` state certifies neither cohort completion nor exhaustion of the time budget. The exact rejected resource was not logged. All three allocation slots are consumed; 83,434 of the original 86,400 seconds remain, but another allocation requires an explicit exception. The other 1,360 cells still need validated deployments and execution budgets. All claims, original verdicts and failure evidence remain preserved; no primary cell is resampled. No job or active supervisor remains: the watcher exited after all authorized work closed, with Table 5 still marked incomplete. Supervision must restart with any newly approved jobs.

Four reviewed requests await approval: Shopping and GitLab additional transfers, append-only Map transfer recovery, and CMS hosting validation. The transfer requests retain their original cumulative time and byte caps; only CMS requests new compute. All four have zero GPU/model/API budget. [Exact pending limits](#table5-pending-bundle-20261012).

<details>
<summary>Primary audit repairs, stop accounting and recovery limits</summary>

Four saved episodes exposed an auditor schema error: an empty native-dispatch object after a rejected action was mistaken for an executed action. Separate source-bound supplemental audits reconstruct the original errors and rerun the identity, reset, verdict, financial and closure checks. Independent reruns agree. Original rejections, trajectories, actions and native/guarded verdicts remain unchanged; no episode or model request was repeated. Explicit supplemental pointers and normalized audit records govern analysis; the older mechanical queue summaries do not independently certify audits. On task 5, the valid GPT-4o no-search episode has native score 0, while the other five valid episodes have native score 1; all six guarded scores are 0.

The operational stop preserves two wrapper-invalid results and two interrupted claims. The latter have saved `EpisodeWorkerExited` failure records with exit −15, five Arbiter turns and 13 Piotr turns, and closed watchdog process trees. Separate additive audits now bind their original claims, source/reset evidence, saved partial traces, financial journal and closure evidence; independent replays match byte for byte. They remain invalid attempts with absent native and effective verdicts represented as `null`, not invented failures from a judge. The auditor and strict collector passed 58 tests. Activation changed no primary evidence, created no model requests, refunds or state disposals, and preserves the unfinished local dispatch charge. The native fallback/click repair applies to future execution and does not replace any of these four primary attempts. The first allocation's eight paid judge controls and 11 local startup requests are charged to the same cap but excluded from primary results. Its financial chain and broker joins reconcile all 671 paid settlements, with no unresolved paid reservations. One unfinished local request remains included in the 472 local dispatches. All owned processes closed.

Job 356022 stopped after **2,614 seconds**, preserving eight audited results and two interrupted claims. Its native terminal-detector wrapper mishandled absent parsed blocks and nonterminal clicks; the prospective repair passed 78 focused tests. Successor 356094 failed during startup after **47 seconds**, before new primary admissions or model requests, when a storage scan encountered a disappearing database-build file. Its owned processes closed. The storage repair passed 35 focused tests plus real missing-file, recreated-path and directory-race checks. It accepts only complete rescans and stable final readings after owned process closure; sampled readings are not a peak quota.

Job 356127 added two new primary results; after the premature admission closure it awaited the admitted workers and exited. Its saved process trees and financial epoch are closed. The financial hash chain preserves every prior charge, with no unsettled paid reservations or overruns. The allocation added **36 paid requests, 17 local requests and USD 0.17124070**, including eight paid and 11 local startup controls. Campaign totals are **707 paid requests, 489 local requests and USD 4.90736970**; the unfinished historical local dispatch remains charged. All three allocations consumed **2,966 of 86,400 approved seconds**. Remaining caps are USD 245.09263030, 20,817 paid requests and 10,982 local requests, with zero allocation slots. Unused time and spending do not authorize another allocation.

Conservative cumulative apparent scratch is **384,763,249,194 bytes against 512 GiB**, including the final attempt's 48,092,985,316 bytes. Final durable campaign artifacts measured 173,209,452 bytes against 64 GiB. Six fresh-state disposal receipts are recorded; historical retained states remain preserved. All ten prior claim and queue-outcome hashes are unchanged, and the two new claims are durable. The collector now includes all 12 audited attempts, keeping eight valid results, two wrapper-invalid failures and two interrupted failures distinct. Independent closure/accounting and two exact saved-evaluator replays are bound in the [preparation artifact](webprm_results/webarena_preparation_20261011.json). The queue repair passed 52 focused tests and seven independent frozen-source checks. It waits for an owned worker to finish before rechecking a denied admission, records each decision's reason, and checks minimum startup resources before model loading or paid controls. Every existing cap and primary identity remains fixed; live recovery is unvalidated. A transient storage projection is a plausible original trigger, but memory or free-disk rejection cannot be excluded because the exact branch was not recorded. Campaign completion and sustained throughput remain unverified.

</details>

<a id="reddit-guarded-validation-20261011"></a>
<details>
<summary>Preserved guarded no-search diagnostic</summary>

| Check | Cases | Passed | Failed |
| --- | ---: | ---: | ---: |
| One parsed candidate and native dispatch per actor turn | 30 | 30 | 0 |
| Fixed judge controls under the explicit guard | 8 | 8 | 0 |
| Primary task success | 1 | 0 | 1 |

All 30 actor actions dispatched without a reported execution exception. That does not establish useful navigation: the actor made 28 repeated clicks at y=727 in a 720-pixel-high viewport. The released visibility filter admitted the partly visible target, and all dispatches matched the pinned native code; no adapter coordinate divergence was found. The episode reached its 30-turn limit and returned `N/A`; the saved native judge score is 1, while the separately recorded guarded score is 0. The eight judge controls produced six native positives and four guarded positives, correcting two non-answer false positives. Earlier cohorts and raw verdicts remain unchanged.

The separate CPU allocation used **371 seconds, 98 paid calls and USD 0.08072970**, with zero GPUs or local model calls. The scripted browser preflight and process-closure checks passed. All three Map frontend archives also downloaded and matched their published checksums; Map hosting is not yet validated. Original CMS acquisition hit the Drive quota. The separately approved third-party candidate has now been acquired and independently verified; hosting remains unvalidated.

</details>

<a id="reddit-pilot-20261011"></a>
<details>
<summary>Preserved five-selector mechanics pilot: all native positives were false positives</summary>

The prospectively fixed cohort contains one Reddit task and one primary attempt per selector, all using `gpt-4o-mini-2024-07-18`. A **mechanics-valid** cell has a saved complete native verdict, matching actor/auxiliary evidence and, for a guided row, actual scoring responses. This verifies execution of the evaluation path; it does not require error-free actions or a semantically correct judge. No primary attempt was resampled, relabeled or pooled with the earlier failed cohort.

| Selector | Attempted | Mechanics valid | Native positive verdicts | Supported successes | Saved turns | Primary selector calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No search | 1 | 1 | 1 | 0 | 30 | 0 |
| GPT-4o-mini | 1 | 1 | 1 | 0 | 30 | 120 |
| WebShepherd-8B | 1 | 1 | 1 | 0 | 30 | 151 |
| WebArbiter-7B | 1 | 1 | 1 | 0 | 30 | 120 |
| Piotr SelectionARM | 1 | 1 | 1 | 0 | 30 | 30 |

“Supported successes” counts outcomes established by the evidence; it is not a repaired benchmark success rate. All five native positives grade the same non-answer as correct, so neither their nominal 100% rate nor a zero-success replacement estimate is reported as task performance. One fixed task would not establish a selector ranking even with a sound judge.

An independent Piotr audit reconstructed all **30 actual request payloads** and all **150 original candidate slots**, checked each turn's own pre-action screenshot and last-five-action history, and verified that all 30 selected native actions were unchanged. All four guided selectors therefore reached scoring. Separately, Arbiter selected 14 IDs absent from both the contemporaneous accessibility tree and captured bounds; those were rejected before native dispatch, while 16 dispatches were generated. This supports stale or unavailable selected targets, not demonstrated coordinate-mapping corruption. No-search had 27 empty candidate turns because bare action objects lacked the required `BLOCKS` envelope; both the wrapper and released parser rejected them.


</details>

<details>
<summary>Corrected-pilot protocol, accounting and audit limits</summary>

The corrected protocol restores the released viewer's `send_msg_to_user [N/A]` fallback at ordinary turn exhaustion, records that origin separately from actor choices, and retains actual actor answers and infrastructure failures. Its approved judge substitution is `gpt-4o-2024-05-13`: the released `gpt-4-1106-preview` returned HTTP 404 in this account's metadata lookup. Native prompts and parsing remain pinned. The single synthetic startup positive passed, but could not detect the later false positives. All five complete raw verdicts and scores remain unchanged.

The allocation completed after **3,179 of 5,400 approved seconds** on one H200, eight CPUs and 120 GiB RAM. Its separate caps were USD 15, 575 paid requests, 400 local requests and 300 GiB temporary space, including retries. It used **571 paid requests and USD 0.39634990**, with no unresolved reservations, and **312 local requests: 301 primary plus 11 startup**. Paid calls comprise 445 actor/auxiliary calls, 120 GPT selection comparisons, five task judges and one synthetic startup judge. Final node-local storage was **288,561,127,428 bytes** against 322,122,547,200; this final reading is not a continuous peak measurement. All registered controllers closed; all **52 frozen source hashes** remained unchanged. The unused 2,221 seconds are not transferred to another approval.

The frozen corrected source had passed 137 offline tests before launch. Piotr's revised descriptor serialized all 73 previously saved pools in an offline check, compared with 61 under the old representation; the live audit now additionally covers its 30 scoring turns. Among the 150 live slots, 114 targets were in view, eight unavailable, two outside the viewport, and 26 were non-target operations. A stale trace-version label names the earlier descriptor version in all 30 records; actual payloads and execution used the approved version and were unchanged. The erratum is retained explicitly; a prospective label-only correction passed 21 focused tests without modifying the frozen cohort. Payload reconstruction does not attest numerical GPU internals or screenshot capture time independently of the frozen worker and artifact identities, and does not establish training-grammar compatibility or model quality.

</details>

<details>
<summary>Completed bounded prompt/judge diagnostic and CMS acquisition failure</summary>

A separate diagnostic made **16 paid calls for a conservative USD 0.01016620 charge** (cached-input discounts are not applied), with no browser or benchmark episodes. On four fixed saved inputs, the original no-search prompt produced **zero valid envelopes** and an explicit one-`BLOCKS` clarification produced **four valid envelopes**. This tests output formatting on those inputs, not browser-task success. The clarification now has an explicit opt-in worker implementation, with all eight diagnostic system prompts matching the generated original/clarified prompts byte for byte. Existing defaults and completed cohorts remain unchanged. The subsequent CPU-only v4 validation is now complete, as reported above. Its approval was eight CPUs, 16 GiB RAM, 45 minutes, zero GPUs, at most 98 paid calls/USD 2, 128 GiB scratch and one allocation. It ran the scripted browser final-answer check, eight fixed positive/negative/abstention controls and one fresh no-search episode on the same task. Its explicit amendment retains every native verdict and assigns zero to bare normalized `N/A` against ordinary fuzzy answer lists containing no explicit `N/A`; genuine unachievable-answer, URL and DOM routes retain native semantics. This is a narrow disclosed guard. The final apparent scratch reading was 96,184,042,107 bytes; it is not a continuous peak measurement. All 49 frozen source identities remained unchanged after execution. The package had passed 163 focused tests and 30 final controller checks.

The same diagnostic's eight synthetic native-judge controls agreed with their expected labels in **six cases**. Both `N/A` and normalized `n/a` were falsely accepted against numeric reference `0`; the other six controls agreed. Positive-only access checks are insufficient for this judge. Any semantic correction must be explicit, tested with negative controls and versioned before it supplies benchmark rates. The reporting audit now automatically flags a positive `N/A` verdict against a numeric fuzzy reference and withholds performance percentages while retaining the native counts and scores. It flagged all five saved rows; 18 analyzer tests passed. This narrow reporting check does not repair the judge or establish accuracy on other answers.

The diagnostic approval was one CPU, 2 GiB RAM, 15 minutes, zero GPUs, at most 16 paid calls/USD 1, 1 GiB storage and one attempt. It completed in **23 scheduler seconds**. Independent closure checks verified all 17 source files, 16 saved request/response pairs, native parser results and charges. The package occupied 1,357,803 bytes before the final audit was added; this is not a peak measurement.

Four separately authorized CMS attempts consumed **22 of the original 3,600 scheduler seconds**. The first two failed before network use; a corrected clock guard then reached a 2,477-byte Drive confirmation page. The separately approved confirmation-aware transfer got through that form but received a 2,040-byte download-quota error. Both network attempts closed with **334,674 charged bytes and zero archive bytes**. Their transfer allowances are spent; no automatic retry is authorized. Original caps remain two CPUs, 8 GiB RAM, zero GPUs/API calls, 10,000,000,000 charged HTTP bytes, 12 GiB retained storage and 512 MiB temporary space. The 3,578 unused seconds do not authorize another network attempt.

A bounded metadata investigation found a digest-pinned 31-layer CMS candidate in the third-party `webarenaimages` registry namespace. Its compressed layers total 2,156,742,179 bytes, but equivalence to the original Drive release is unverified. The [WebArena author's source clarification](https://github.com/web-arena-x/webarena/issues/231#issuecomment-3583249074) explicitly says those images were not uploaded by the authors. Archive.org marks its CMS item unavailable, and the CMU endpoints did not provide an accessible alternative. This investigation charged another 992,004 bytes, bringing cumulative charged network bytes to 1,326,678. Acquiring this candidate would require a separately disclosed third-party-source reconstruction request.

The approved CMS third-party acquisition then completed in **118 scheduler seconds**, bringing cumulative CMS time to **140 of 3,600 seconds**. All **31 compressed layers, totaling 2,156,742,179 bytes**, were independently rehashed; all 31 raw-layer receipts reconcile with the pinned configuration, covering 9,639,904,768 raw bytes. Cumulative charged HTTP bytes are **2,165,742,127**, with zero unsettled reservations, GPU/model/browser calls or extraction. The source, owner closure and terminal accounting checks passed. The launcher initially failed its immediate final-accounting check; the unchanged strict check accepted a fresh terminal record. The original failed check is preserved; its initial snapshot was not saved, so a transient scheduler state is an inference. The remaining 3,460 seconds do not authorize another transfer. Original-state equivalence and hosting readiness remain unverified. Final audit SHA-256: `c7c501fb4a620ea4799fea6178df023dfb9801e3bf84fbad8fd464ed32d9b327`.

The completed Map frontend acquisition used **378 of 1,800 approved seconds**, two CPUs, 4 GiB RAM, zero GPUs/API calls and one allocation. Its three files total **1,231,929,664 bytes**, all independently rehashed against the [published Zenodo checksums](https://zenodo.org/records/12636845). Charged downloads were 1,232,126,272 bytes against 1,300,000,000; retained logical bytes before the final audit were 1,232,004,900 against 2 GiB. No extraction, backend startup or browser validation occurred.

</details>

### Remaining comparison plan

1. Prepare any further Reddit attempt using the independently reviewed queue-admission repair and remaining original time, spending and storage caps; launch only after an explicit additional-allocation exception. Preserve all 12 claims and fixed identities for the 178 unclaimed cells, including invalid and interrupted attempts in final accounting.
2. Acquire and validate Shopping, CMS, GitLab and Map deployments independently, recording verified original identities separately from third-party candidates. Map frontend acquisition is complete; backend acquisition failed on a TLS handshake timeout and hosting remains unvalidated. The three native helper adapters passed 21 offline tests exercising all 13 released single-site helper task configurations, plus 30 existing evaluator tests. This proves evaluator contracts on fixtures, not live website/reset readiness. Each acquisition or hosting job needs its exact resource/attempt approval.
3. Obtain exact execution budgets for the remaining 1,360 cells. The full primary cohort has 155 released single-site tasks: 45 Shopping, 35 CMS, 19 Reddit, 30 GitLab and 26 Map. Five selectors and two actors give 1,550 planned episodes. Keep the ten multi-site tasks separate. Among the proposed 155 tasks, 39 use fuzzy grading and 13 use deployment-specific helpers; across all 165, those counts are 40 and 14.
4. Report per-domain success and the unweighted domain mean, planned/attempted/valid/success counts, invalid reasons, paired gains, uncertainty and inference cost. Overall success uses the fixed planned denominator; valid-only and common-valid comparisons are secondary. Incomplete or semantically unreliable cohorts do not supply performance claims.
5. Then run Online-Mind2Web with the same selector identities, its own frozen live-browser protocol and approved resources. Historical OM2W diagnostics below cannot populate the new matched comparison.

<details>
<summary>Execution readiness and remaining live gates</summary>

The fixed-grid generator, concurrent episode queue and full-table analyzer are implemented and tested offline. The controller snapshot fixes all 1,550 cells, both actor routes and the common seed, turn limit and replacement-judge settings. Its frozen implementation passed 267 tests and ten source/protocol/task identity audits spanning every actor-selector combination; the subsequent bounded Reddit package passed 278 tests. Durable claims prevent a completed or interrupted primary attempt from being silently replaced. The analyzer keeps native and effective scores separate, computes the equal-domain mean separately from pooled success, and withholds primary rates for incomplete or unaudited cohorts. Execution is approved for the 190 Reddit cells. The first attempt has ten audited dispositions: six valid, two wrapper-invalid and two interrupted; successor 356094 failed before primary admission; final allocation 356127 added two independently audited valid results, then stopped admitting work. All three allocation slots are consumed.

| Deployment | Released single-site tasks | Live readiness | Remaining gate |
| --- | ---: | --- | --- |
| Reddit | 19 | Original startup, authentication and reset validated; all 12 claimed attempts audited | Additional-allocation exception, live queue-repair validation and 178 unclaimed cells |
| Shopping | 45 | Third-party acquisition failed; file-accounting race repaired | Explicit additional-transfer exception, acquisition, startup and reset |
| CMS | 35 | Third-party candidate acquisition verified; hosting unvalidated | Identity assessment, owned service startup, authentication and reset |
| GitLab | 30 | Third-party acquisition failed; same repair prepared | Explicit additional-transfer exception, acquisition, startup and reset |
| Map | 26 | Three frontend archives verified; hosting not validated | Backend acquisition, database identity, owned services and native checks |

The prospective concurrent model-request broker now enforces shared cumulative dispatch and spending caps across allocation attempts, with unknown reservations retained and no automatic refunds. Its read-only financial auditor checks each episode against a fixed journal prefix while allowing unrelated concurrent requests. The independent episode auditor binds task, protocol, source, reset, action history and closure evidence before producing a table record. All six findings from its independent review were corrected; 45 combined episode-audit, financial-audit and temporary-state tests passed.

The evaluator transcript saves bounded page/helper inputs and replays the pinned scoring computation offline. Reddit and GitLab helpers are recomputed from saved observations; Shopping replay verifies the recorded helper return boundary, not its underlying REST query. A separate disposal helper permits only newly claimed temporary states to be discarded after complete durable evidence, an independent audit and process closure. These offline checks establish evidence consistency, not judge accuracy or independent historical process observations. All three allocations are closed. Durable evidence and independent audits cover all 12 claimed attempts, including both interrupted failures; campaign completion and sustained throughput remain unverified.

The approved CPU validation and Map frontend jobs completed; the original CMS transfers closed without archive data. The user approved all five next-stage requests below. CMS candidate acquisition is verified complete. Reddit successor 356094 failed during startup; final allocation 356127 closed with two new valid results and 178 cells unclaimed. Its three-allocation cap is exhausted. All three remaining acquisitions have stopped and need explicit transfer exceptions; their failure causes and preserved charges are recorded below. Shopping and GitLab also returned explicit original-Drive quota errors in bounded metadata checks. Their pinned third-party candidates, like CMS, have no verified original-state equivalence. Downloading them does not authorize hosting or establish benchmark readiness.

</details>

<a id="table5-next-resources-20261011"></a>
<details>
<summary>Approved and submitted: Reddit comparison and four independent acquisitions</summary>

The fixed Reddit request covers **19 tasks × five selectors × two actors = 190 primary cells**, retaining the same task and protocol identities within the 1,550-cell reconstruction. Its final frozen implementation passed **278 tests and ten identity audits**. Independent review verified cumulative recovery accounting, the three-allocation cap and cancellation if submission receipts cannot be written. Startup used 11 local model requests and eight paid judge controls; all passed under the disclosed guard, which changed two native positives to effective negatives. These synthetic controls are excluded from primary results. All three approved Reddit allocations are now closed; their accounting and preserved primary dispositions are reported above. Sustained throughput and completion within the requested cap remain unverified. The other 1,360 cells remain gated on their deployments and separate execution budgets.

| Request | GPUs | CPUs | RAM (GiB) | Cumulative scheduler time | Maximum allocations | API cap (USD) |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Reddit: 190 fixed cells | 1 H200 | 8 submitted | 240 | 24 h | 3 | 250 |
| Map backend acquisition | 0 | 2 | 8 | 4 h | 3 | 0 |
| CMS third-party acquisition | 0 | 2 | 8 | 3,578 s remaining from original 3,600 s | 2 additional | 0 |
| Shopping third-party acquisition | 0 | 2 | 8 | 2 h | 3 | 0 |
| GitLab third-party acquisition | 0 | 2 | 8 | 2 h | 3 | 0 |

Reddit has two concurrent episode lanes, caps of **21,524 paid and 11,471 local requests**, 512 GiB cumulative apparent scratch and 64 GiB durable campaign artifacts. Newly claimed temporary website states may be discarded only after evidence is saved durably, independently audited and all owned processes close. Immutable bases and historical states remain preserved. Every attempt consumes the same cumulative time and request budgets; a primary cell cannot be resampled after failure.

| Acquisition | Charged download cap (GB, decimal) | Retained cap (GiB) | Temporary envelope (MiB) | New transfer campaigns |
| --- | ---: | ---: | ---: | ---: |
| Map backend | 195 | 200 | 512 | 1 |
| CMS third-party | 10 | 12 | 512 | 1 additional |
| Shopping third-party | 70 | 72 | 512 | 1 |
| GitLab third-party | 80 | 80 | 512 | 1 |

All acquisitions have zero GPU/model/API calls, one network connection each and no automatic network retries or transfer resume. CMS includes its prior 22 scheduler seconds and 1,326,678 charged bytes; its temporary envelope reserves 4 MiB for prior artifacts. Shopping and GitLab include their respective 495,171 and 478,598 metadata charges. Each request preserves its own caps. CMS, Shopping and GitLab require explicit acceptance of third-party sources with unverified original-state equivalence. Map follows the original dataset recipe with pinned current image blobs; historical image equivalence and runtime compatibility remain unverified. These four jobs download and verify data only; hosting is a later gate.

The reviewed bundle is recorded in the [sanitized preparation artifact](webprm_results/webarena_preparation_20261011.json), including proposal digests and exact limits. Bundle file SHA-256: `e53d165ad01a9a69c777c9de9f28d4e9f76d0514df203aa8708fdcdd3004ebde`. The user approved the complete bundle. Submitted jobs are Reddit **356022**, Shopping **356007**, Map **356008**, CMS **356009** and GitLab **356010**; CMS subsequently completed; Reddit stopped for the wrapper repair, and Map/Shopping/GitLab failed as recorded below. Slurm rejected the original Reddit request before allocating a job because it permits at most eight CPUs per GPU. The submitted request reduces CPUs from 16 to **8**, preserving one H200, 240 GiB RAM, 24 cumulative hours and every API, storage, scientific and attempt cap. This resource reduction passed 22 focused checks, ten identity audits and a scheduler test-only check; all Python implementation files remain byte-identical. The rejection consumed no allocation time. Queue estimates are not guaranteed start times.

</details>

<a id="table5-pending-bundle-20261012"></a>
<details>
<summary>Four reviewed requests: pending approval, no submissions</summary>

| Request | CPUs | RAM (GiB) | Requested seconds | Additional allocations | Time accounting |
| --- | ---: | ---: | ---: | ---: | --- |
| Shopping transfer exception | 2 | 8 | 7,140 | 1 | 37 seconds already consumed from original 7,200 |
| GitLab transfer exception | 2 | 8 | 7,140 | 1 | 42 seconds already consumed from original 7,200 |
| Map append-only transfer exception | 2 | 8 | 13,020 | 1 | 1,372 seconds already consumed from original 14,400 |
| CMS hosting validation | 4 | 16 | 7,200 cumulative | At most 2 | New bounded compute request |

All four request zero GPUs, model calls and API spending. Shopping/GitLab/Map request one additional transfer campaign each with no added byte budget; Map alone permits three narrowly defined TLS handshake reconnects. CMS permits no downloads or benchmark episodes and retains its disclosed third-party identity limitation. Exact storage, transfer and recovery limits appear below and in the [sanitized preparation artifact](webprm_results/webarena_preparation_20261011.json). Bundle file SHA-256: `6dad3776d0364e1ae782b292430fe0476f1ae7880aaed7fc78842d4e3b0c93db`. This request supersedes the earlier pending three-request bundle; no new authority has been granted.

</details>

<a id="table5-transfer-recovery-20261012"></a>
<details>
<summary>Shopping/GitLab failure, tested repair and pending transfer exceptions</summary>

Shopping job 356007 failed after 37 seconds and GitLab job 356010 after 42 seconds. Each owner listed an atomic ledger temporary file just before its writer renamed it, then failed when the storage scan tried to inspect the vanished name. Both owned process trees closed. The new scanner recognizes only this bounded receipt-rename case, conservatively charges a 4 MiB reserve, and still rejects missing payloads or unsafe destinations. Deterministic frozen-code replays reproduce the error; the patch and cumulative-ledger controls pass. The frozen Map package remained unchanged; its separate TLS failure is recorded below.

| Acquisition | Consumed seconds | Requested successor seconds | Charged HTTP bytes preserved | Original charged-byte cap |
| --- | ---: | ---: | ---: | ---: |
| Shopping | 37 | 7,140 | 1,146,213,879 | 70,000,000,000 |
| GitLab | 42 | 7,140 | 1,192,051,429 | 80,000,000,000 |

Each reviewed successor requests **one additional fresh transfer and one allocation, two CPUs, 8 GiB RAM, zero GPU/API calls**, within its original 7,200 cumulative seconds and unchanged byte/storage caps. Failed partials and unknown reservations stay preserved and charged; the replacement starts from byte zero without resuming them. Each original 512 MiB temporary envelope reserves 4 MiB for prior state. Retained caps remain 72 GiB for Shopping and 80 GiB for GitLab. No further successor is authorized by these proposals, including after a pre-transfer startup failure.

These are pending requests, not submissions. The original approvals explicitly permitted one transfer campaign; unused time and bytes do not grant a second campaign. Shopping proposal SHA-256: `30ec937c4fcaa66389199ae48663ffccea260a4162d45f8b8785687784fa2b23`; GitLab: `1080ae609e745951db364853e62fd689c559fdf2a5fc8988e1449e728104deb2`. Both preserve their previously disclosed third-party source identities; original-state equivalence and hosting remain unverified.

</details>

<a id="table5-map-transfer-recovery-20261012"></a>
<details>
<summary>Map TLS failure: closed and audited; reviewed bounded resume awaits approval</summary>

Map job 356008 failed after **1,372 of 14,400 approved seconds**, with an SSL handshake timeout before the next response. Its owned process tree is closed. The ledger preserves 594 complete 64 MiB responses and one fully charged unknown reservation: **39,862,665,216 known body bytes and 39,968,768,000 charged bytes**. No complete dataset was verified. The first dataset's saved prefix has an independently computed local SHA-256, `75270ca31ce09e25c8316085c93a8b32f52afb7fae21ff462b6e79e3d096e191`, with unchanged inode, size and timestamps across that check. This binds saved bytes; it is not a published prefix checksum or proof of whole-archive identity.

A fresh full download would exceed the original 195,000,000,000-byte transfer cap. Copying the prefix and retaining a complete second copy would also exceed 200 GiB of apparent storage. The reviewed successor requests one additional allocation and transfer campaign, using two CPUs and 8 GiB RAM for 13,020 seconds. It rehashes the fixed prefix before network use, appends to the original inode, and verifies the prefix and complete first archive again after writing. Prior charges, the unresolved reservation and anticipated retained-file growth remain counted. At most three reconnects are allowed across the entire campaign, only for pre-response TLS handshake timeouts; each unknown reservation stays fully charged. Other network retries and further successors are prohibited. The 195,000,000,000-byte transfer, 200 GiB retained and 512 MiB temporary caps remain unchanged. Offline tests and independent review passed; approval and launch are still pending. The transport also handles bounded full-blob registry responses, a separate risk that did not cause the original failure.

</details>

<a id="table5-cms-hosting-request-20261012"></a>
<details>
<summary>Reviewed CMS hosting-validation request: pending approval</summary>

The prepared CMS stage requests **four CPUs, 16 GiB RAM, zero GPUs, two cumulative hours across at most two allocations**, with zero model/API calls or new downloads, 48 GiB cumulative apparent scratch and 2 GiB durable evidence. It uses the acquired third-party candidate, whose original-state equivalence remains unverified. The controller owns image materialization, services, browser checks, a single synthetic product mutation and an independent fresh-state reset; it preserves the imported database and bypasses the image initializer. This does not authorize benchmark episodes.

All 31 frozen tests pass under a private-file umask, including positive/negative controls through the pinned native evaluator and detached-process closure. A generated-config permission error caught during the prior freeze was corrected before any live allocation. The imported image has not yet been extracted or hosted; actual startup, authentication, native checks, database/media/session reset and independent closure audit remain live gates. Scratch is measured by apparent bytes with sampled enforcement, not a kernel peak quota.

Proposal SHA-256: `b0fe94c47a63466cb0e4536af902a6f324c54eff551f9372f4ada2f9571c18b6`. The consolidated four-request bundle includes the three transfer exceptions above. Any further Reddit allocation requires a separate explicit exception to its exhausted three-allocation cap. Its exact limits and readiness digests are in the [preparation artifact](webprm_results/webarena_preparation_20261011.json).

</details>

<details>
<summary>Published Table 5 references and the matched target comparison</summary>

The target remains the four published conditions plus **Piotr's released SelectionARM**, under both GPT-4o-mini and GPT-4o actors. WebArena-Lite comes first, then a separately frozen Online-Mind2Web comparison. Every primary row uses the same cohort, actor snapshot and benchmark evaluator; guided rows request five candidates, while no-search requests one action.

| Selector | Paper GPT-4o-mini mean success (%) | Paper GPT-4o mean success (%) | Full rerun status |
| --- | ---: | ---: | --- |
| No trajectory search | 23.48 | 31.90 | Partial Reddit coverage |
| GPT-4o-mini selection | 24.47 | 33.03 | Partial Reddit coverage |
| WebShepherd-8B | 34.34 | 40.65 | Partial Reddit coverage |
| WebArbiter-7B | 40.52 | 47.01 | Partial Reddit coverage |
| Piotr SelectionARM | — | — | Partial Reddit coverage |

Paper values are references from [WebArbiter v2, Table 5](https://arxiv.org/html/2601.21872v2#S5.T5), not reproduced measurements. They are unweighted means over Shopping, CMS, Reddit, GitLab and Map. The paper imports its no-search and Shepherd rows from earlier work; their implied denominators differ from its contemporaneous rows. Exact historical task membership and all inference settings remain unresolved, so this implementation is a disclosed reconstruction.

The authors' [trajectory dataset](https://huggingface.co/datasets/ZYao720/WebArbiter-Trajectories) still contained no released trajectories when checked on October 11. Its advertised 72 trajectories do not establish the complete Table 5 task manifest.

</details>

<details>
<summary>Preserved first Reddit pilot: five attempts, zero valid native verdicts</summary>

The first pilot used the same prospectively fixed task and GPT-4o-mini actor, with one primary attempt per selector. Its outcomes remain unchanged in the [original cohort artifact](webprm_results/webarena_pilot_20261011.json). With zero valid verdicts, valid-only success is undefined.

| Selector | Attempted | Valid | Invalid | Saved interaction turns | Completed selector calls | Failure |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| No search | 1 | 0 | 1 | 0 | 0 | Browser task accessed before reset |
| GPT-4o-mini | 1 | 0 | 1 | 30 | 120 | Missing turn-limit answer blocked native grading |
| WebShepherd-8B | 1 | 0 | 1 | 30 | 151 | Same grading block |
| WebArbiter-7B | 1 | 0 | 1 | 30 | 120 | Same grading block |
| Piotr SelectionARM | 1 | 0 | 1 | 0 | 0 | Local-call cap exhausted before scoring |

The first wrapper omitted the released turn-limit fallback and rejected the three nonterminating guided episodes before any native judge call. Piotr's allowance was exhausted before its first scoring dispatch. Those failures motivated the separately versioned correction; they were not resampled or relabeled as successful evaluations. Detailed first-pilot accounting remains below.

</details>

<details>
<summary>Frozen selector interfaces and limits of comparability</summary>

The first closed diagnostic used `webarena_table5_viewer_reconstruction_v2`; the separately frozen corrected cohort used `webarena_table5_viewer_reconstruction_v3`. Actor snapshots are `gpt-4o-mini-2024-07-18` and `gpt-4o-2024-05-13`. The released five-BLOCK actor prompt and native auxiliary state-difference/terminal stages are retained for guided rows. No-search minimally edits that prompt to request one action; it does not generate five and choose the first. GPT-4o-mini selection uses the same fixed pairwise prompt and four-comparison knockout as WebArbiter. First-valid-of-five and random selection remain optional diagnostics, outside the five primary rows.

Piotr’s checkpoint is [PTeterwak/OpenWebRL-4B-SelectionARM](https://huggingface.co/PTeterwak/OpenWebRL-4B-SelectionARM/tree/81b452d800d9f859687074f82680dd5257e02d89), revision `81b452d800d9f859687074f82680dd5257e02d89`. The pinned released prompt uses the current screenshot, five candidates, task/current URL and last five prior actions; DOM, vote counts and chain-of-thought output are disabled. Selection is greedy, with our adapted 128-token response cap (the released demo defaults to 2,048) and a strict one-based JSON index. A malformed response cannot fall back to the first action. Both weight files passed fresh content checks against the pinned release. A separate serving metadata copy enforces the intended 262,144-pixel ceiling and translates a newer tokenizer field into the installed library’s format; all 151,669 vocabulary entries, 14 special-token IDs and the chat template are unchanged. Original checkpoint files remain untouched. A controlled 1,280×720 image becomes 672×384 pixels and 252 image tokens. The earlier descriptor-v1 preflight used 956 input tokens. The corrected descriptor-v2 preflight used 1,021 input tokens: 65 additional tokens for the five candidates' target-ID/status fields, with the same 252 image tokens. It returned valid selection JSON in ten tokens. This verifies serving compatibility, not task performance.

WebArena element IDs are not meaningful on an unannotated screenshot. The added adapter supplies a deterministic selector-only descriptor containing the original operation and its pre-action pixel center/normalized point, including exact clear/Enter semantics for typing. Execution still uses the untouched GPT-authored native action. Prior actions retain their own earlier grounding. This is a disclosed action-representation adaptation and adds a visual modality: the comparison estimates complete selector systems, not solely model quality under identical inputs. The corrected pilot exercised all 30 Piotr scoring turns, and saved-input reconstruction matched every request. This establishes the disclosed input contract, not Piotr's selection quality or training-data independence.

The first pilot required all five original Piotr slots to parse and ground on the current screenshot. The corrected representation retains all five original syntactically valid slots, explicitly marking unavailable/offscreen targets from pre-action evidence and describing native tab/forward operations without inventing destination state. It adds no substitute candidates, target points or resampling; selected raw actions still execute unchanged and can fail. Malformed grammar, corrupt geometry and missing screenshots remain explicit errors. Other adapters may use the admissible remainder of their original five slots, so valid-only denominators can differ and must not be presented as matched cohorts. Evaluator references, future states and post-action outcomes are excluded from all selector inputs.

The earlier inventory of eight candidates and the local teacher-choice heads remains historical evidence. Pointwise epoch 13 and listwise epoch 6 were selected on 3,759 development panels with 65.4429% and 65.5227% teacher-choice agreement respectively; these are not the Piotr row and are not outcome-trained PRMs. Their task-text overlap checks do not establish Piotr’s training-data independence, which remains unverified.

</details>

<details>
<summary>Original Reddit hosting: completed controlled validation</summary>

| Check | Planned | Passed | Failed |
| --- | ---: | ---: | ---: |
| Original compressed/uncompressed image-layer pairs | 29 | 29 | 0 |
| Distinct released Reddit start URLs | 3 | 3 | 0 |
| Original-post positive/negative check pairs before mutation | 3 | 3 | 0 |
| Synthetic-post title/body positive/negative check pairs | 2 | 2 | 0 |
| Original-post positive/negative check pairs after reset | 3 | 3 | 0 |
| Synthetic post absent after fresh reset | 1 | 1 | 0 |

These checks verify original-image startup and a controlled mutation/reset, not benchmark task success or exhaustive database equality. The three URLs cover starting endpoints of the 19 single-site Reddit tasks. All service/browser/conversion processes are closed. Original Reddit deployment diagnosis completed in CPU-only job `354646`: four CPUs, 16 GiB RAM, zero GPUs and zero paid API calls, consuming 2,934 of 7,200 approved allocation seconds. The unused 4,266 seconds are not carried into the proposed pilot.

The original Drive archive hit its download-quota limit; its partial bytes remain preserved. A registry mirror has the exact original configuration digest and all 29 original uncompressed layer identities. Its manifest is `sha256:e5b9d29b26ca911048fb14a8d4806a871b796db85c2991d514a13df2af0e427f`. Every compressed/uncompressed layer pair passed verification. The services run with native Alpine fakeroot in an isolated network namespace, from fresh writable copies. Normal localhost mappings repaired HTTP 500 errors; the launcher preserves stale PID evidence and bypasses the original initialization entrypoint. Smaller modified benchmark images are not substituted. [Official environment reference](https://github.com/web-arena-x/webarena/blob/dce04686a56253aefba7b18a4fa0937cf1dc987b/environment_docker/README.md).

Charged downloads total 45,944,201,372 bytes against 54,000,000,000; retained artifacts occupy 89,434,872,832 bytes against 100 GiB. The conservative temporary-space bound is 303,743,941,293 bytes against 300 GiB. The converted immutable cache is 43,595,845,632 bytes, SHA-256 `a091e6433070df9dc450f37e9e35719c9ba43d4879d220898c0321d90053b7d6`; restoration from it remains untested. The next pilot uses the validated original OCI build path. All failed download/conversion attempts remain accounted.

</details>

<details>
<summary>Closed-pilot accounting, repairs and evidence limits</summary>

Seven allocation attempts consumed **3,834 of 10,800 approved seconds** on one H200, eight CPUs and 120 GiB RAM; all are terminal and all owned controller/site/browser/model processes are closed. The unused 6,966 seconds were not transferred to the separately approved corrected pilot. The final node-local reading was 240,454,772,707 bytes against 300 GiB; this is a final reading, not a continuous peak measurement.

All **389 paid calls** reconcile with saved complete responses and settled ledger entries: **USD 0.27544590**, with no unresolved reservations, against USD 15. Local accounting reached **320 calls: 315 broker dispatches plus five conservative historical warmup reservations**. Startup/recovery consumed 49 of those calls; Shepherd consumed 151 and Arbiter 120, leaving none for Piotr's first scoring call. Its two actor/auxiliary calls are preserved. No native judge call occurred. The separate original-hosting allocation is accounted below, not charged again here.

Startup repairs verified pinned checkpoint contents across filesystems, reconciled cluster-cleared scratch only on its original node, retained published nested rotary-position values, forced Piotr's byte-identical released chat template, and corrected the Reddit readiness cookie handshake. The no-search initialization failure was fixed by sharing an explicit 1,280×720 viewport before BrowserGym's single reset; its original invalid result remains. All 24 focused worker and 21 controller tests passed after those fixes. Three critic servers passed all eleven metered GPU startup checks. The 42-file live source manifest remained unchanged through final closure and has a preserved source snapshot.

Independent final analyses agree on all five outcomes, complete per-row call evidence and cumulative API accounting. GPT selection made 120 comparisons; Shepherd made one checklist and 150 pointwise calls; Arbiter made 120 comparisons. Each completed guided trajectory had 89 actor/auxiliary calls. Native action errors and stale target selections are saved, not silently removed. Offline tests and matching receipts establish the stated software/evidence contracts; they do not establish a successful task, a valid missing native verdict or exact paper reproduction.

</details>

<a id="repair-status-20261010"></a>
## Repair status — October 10, 2026

**The v4 repair run is closed and incomplete: 16 of 72 fixed cells were attempted, nine had valid terminal judgments, and none succeeded.** A proven browser-geometry mismatch hid visible Arbiter controls and stopped further task admission. The four active episodes finished, but the paused controller was not closed promptly; this supervision failure consumed substantial idle allocation time. No evaluation allocation remains live. V6 adds browser-geometry restoration, an external episode watchdog and durable result writing before cleanup. Its frozen source passed 102 focused tests and all six scripted real-browser conditions. V6 preparation is verified and ready for exact resource approval. The optional real-site diagnostic remains inconclusive, and v5 was frozen but never launched. No corrected live benchmark has run, and no new allocation or API budget has been approved or submitted. [Aggregate repair evidence](webprm_results/debug_20261010.json).

| Repair cohort | Planned cells | Attempted | Valid | Invalid | Missing | Successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3, closed after context failure | 72 | 6 | 5 | 1 | 66 | 0 |
| v4, closed after geometry diagnosis | 72 | 16 | 9 | 7 | 56 | 0 |

These separate partial cohorts do not establish critic effectiveness. Validity requires a completed episode and a complete, parseable terminal judgment; it does not establish correct browser geometry or error-free execution. Invalid attempts remain in the fixed 12-task denominator for each condition, while missing cells remain unobserved. All saved outcomes and prior cohorts are preserved.

<details>
<summary>Repairs and validation</summary>

| Problem | Evidence | Repair and validation |
| --- | --- | --- |
| One malformed Shepherd sample terminated the whole batch/episode | All 50 saved multi-sample batches had usable candidates; the old rule discarded 316 valid samples | Parse every fixed draw with the native parser, retain valid candidates, and record every rejection; an empty pool consumes a turn and returns error feedback without executing an action or resampling |
| Arbiter omitted the released state-difference and terminal-detection stages | Its sampler does not advertise terminal actions; all 25 valid Arbiter episodes reached the turn limit | Restore native auxiliary prompts/parsers in all three conditions, with the same pinned actor, T1/top-p0.9/2,048-token caps; preserve terminal refinement state |
| Arbiter failed actions were absent from subsequent prompt history | 112 of 153 execution errors repeated a previously failed action | Include exact execution errors in history and alongside the generated state difference |
| Synthetic browser activation raced navigation | At least 16 click-related context-destroyed errors came from the adapter callback | Restrict activation bridging to tab changes; 24 real-browser navigation, typing, save and tab actions passed |
| Startup and request preflight were brittle | Long shared temporary paths failed; a standalone worker reserved budget before discovering missing credentials | Force short local temporary/cache paths and validate/load credentials before browser startup or reservation |
| Initial navigation waited for every page subresource | A controlled interactive page timed out under the old 10-second load check | Wait for DOM readiness with a 30-second cap bounded by the episode deadline; retain HTTP/navigation failures explicitly |
| The released Shepherd serving example caps context at 10,000 tokens | The first repaired live guided episode failed on a 12,574-token prompt after ten executed turns | A distinct v4 retry uses the checkpoint's native 40,960-token limit, with no input truncation or RoPE scaling; GPU startup replay passed on the exact 12,574-token request with native score parsing |
| Native Arbiter filtering used an 800×600 screen inside a 1,280×720 viewport | After BrowserGym screenshot extraction, the native screen-based filter omitted visible right/bottom controls | Restore screen and viewport dimensions before every observation; three repeated screenshot/action cycles and right/bottom controls in a new tab passed a real-browser fixture, without changing the native observation processor |
| A hung browser lifecycle could stall a worker and its collector indefinitely | The optional real-site diagnostic hung at an unidentified phase; source inspection found result writing after browser cleanup and unbounded collector waiting | The v6 watchdog supervises each episode in an external process and persists its result before cleanup; 102 focused tests and all six frozen-source scripted browser conditions passed |

This is a **new, versioned evaluation protocol**, not a correction to old outcome labels. Individual Shepherd parsing agrees with the native single-action contract on all 1,136 saved samples; the fix changes candidate handling and recovery, not extraction heuristics. Original source code, rollouts, verdicts and shared API reservations remain preserved.

**All 102 focused tests passed against the final v6 source** in 123.49 seconds of test-runner time. Its 15-file source freeze and batch syntax were verified; the launcher also rejected pending authorization before starting any browser, model server or API request. All six scripted browser conditions passed serially, each in a fresh interpreter using the frozen source, and every fixture process exited successfully. These checks made zero provider or model-server calls and used no GPU allocation. They validate preparation; GPU startup and a corrected live benchmark remain unverified for v6.

The watchdog allows 180 seconds per browser phase and 20 seconds for cleanup plus interpreter shutdown, with a two-second termination grace and 0.25-second polling. It supervises only the episode's owned processes and writes the verdict or failure result before cleanup. These limits are recorded in the v6 scientific identity. The overall allocation deadline remains enforced, and timeout results are retained without automatic replacement attempts.

Historical v5 passed 85 focused tests and the separate geometry fixture; its 14-file freeze is preserved and was never launched. An earlier six-condition fixture exercised actual Chromium observations, native actions, candidate selection, malformed-action recovery, error feedback, state differences, terminal detection and saved terminal artifacts using scripted completions. The geometry fixture separately covered repeated screenshot/action cycles and right/bottom controls. Scripted fixture success does not establish benchmark success. The optional real-site diagnostic hung at an unidentified lifecycle phase and provides no evidence that the live failure is repaired.

During the initial v4 stage, both actual critics passed GPU preflight and completed a 30-turn live guided episode with a valid failure judgment. Shepherd retained 597 of 600 fixed actor draws and made 126 native scoring requests; its largest critic prompt contained 22,805 tokens. The exact 12,574-token context regression also passed without truncation at the native 40,960-token limit. These checks establish that collection and scoring executed on the supplied observations. They do not resolve the subsequently identified observation defect or establish critic effectiveness or deterministic serving.

The original judge was also audited: all 31 valid episodes supplied the correct task, pinned prompt, matching trajectory and byte-identical terminal screenshot; all complete model responses explicitly returned failure. The judge wiring did not cause universal zero success. Missing actor completion logic can harm behavior, but does not itself force this judge to return failure.

The two originally unavailable sites still time out under a 30-second DOM-readiness check. Their failures cannot be claimed repaired or relabeled as actor failures, and their tasks have not been silently replaced. The frozen browser version and live-site availability remain limitations.

</details>

<details>
<summary>Failed gate, geometry diagnosis, supervision failure and accounting</summary>

The original v4 rule required at least one valid episode per condition across the initial two tasks. It **failed**: Arbiter unguided's first-task policy response ended with `finish_reason=length` at the unchanged 2,048-token cap, and its second task failed at initial navigation with HTTP 403. All six conditions encountered that second-task website failure. The initial stage therefore had 12 attempts, five valid episodes and no successes; both guided conditions had a valid first-task episode.

An explicit operational waiver admitted missing cells after reviewing those failures; the original gate did not pass. The continuation preserved all initial results and the frozen model, worker, prompts, sampling and per-task arm shuffle, with four workers. Its driver hash, failure proof and revised deadline were recorded. Four additional episodes completed before the run closed, giving the final 16 attempts; the remaining 56 cells were never attempted. No failed cell was rerun or relabeled.

The subsequent geometry diagnostic proved that the released native screen-based observation filter used 800×600 bounds after BrowserGym screenshot extraction while the rendered viewport remained 1,280×720. This omitted visible controls on the right and bottom. Replaying the exact released metrics override independently reproduced the same screen reset in this Chromium build. The correction is therefore a disclosed browser-compatibility repair of an inherited assumption, not a change to the native observation processor. Before each observation, it retains the page's CDP session, clears the stale device-metrics override and explicitly restores screen/viewport dimensions. A real-browser fixture verified three repeated screenshot/observation/click cycles, an original-tab bottom control and new-tab right/bottom controls, with zero provider calls. The optional real-site diagnostic hung at an unidentified lifecycle phase; no corrected live benchmark has run. V4 remains a preserved diagnostic cohort, and the unlaunched v5 source freeze is preserved. V6 would start a fresh cohort with the same 12 tasks, six conditions, actor sampling and turn limit, with the geometry change explicitly versioned.

The admission controller was paused while its four active workers drained. Active supervision then failed to terminate the paused controller and release the allocation. The final allocation remained idle for **3,746.32 seconds** after the last durable result. The launcher eventually killed its remaining child processes, and the final allocation ended `FAILED`. All workers are now stopped. The saved aggregate was reconstructed from the 16 durable cells, and the earlier aggregate was archived. The implemented and tested cooperative stop-admission guard watches `stop-admission.json`, stops admitting new cells, drains active workers into durable results and returns without pausing the process. The launcher can then exit normally when those workers finish. Source review found that a worker could still hang during browser operations or cleanup, preventing the drain from completing; the implemented v6 process watchdog and result-before-cleanup ordering bound that path. The final 102-test suite includes lifecycle, cooperative-drain and retry-accounting checks; all six frozen-source browser fixtures also passed.

All allocation time, including the idle interval, is charged to the approved **one H200, eight CPUs, 120 GiB and 7,200-second repair budget**. The context-failure attempt, initial v4 stage and v4 continuation consumed 1,073, 1,371 and 4,576 seconds respectively: **7,020 seconds used, 180 remaining**. That remainder does not support another full evaluation. The validated v6 preparation requests **one H200, eight CPUs and 120 GiB for 10,800 additional allocation seconds (three hours), including all retries**, under the same USD 30 shared API cap with no additional API budget. It starts a fresh 72-cell cohort using the same 12 tasks and six conditions; the old 180-second remainder is not added. The proposed initial validity gate covers the first three fixed tasks, expanded from two after the earlier second-task website failures; all fixed tasks and failure outcomes remain included. This proposal is **not approved or submitted** and does not guarantee a successful outcome or complete coverage.

The shared API ledger contains **1,905 request reservations: 1,904 settled and one unresolved**. Total settled charges are **USD 1.91658900**, with **USD 0.00609330** retained conservatively, totaling **USD 1.92268230** against the unchanged **USD 30** cap. V4 alone contributed 523 settled calls and USD 0.60449560. Original v2, its two supplemental attempts, v3 and v4 outcomes remain separate; their request journals reconcile the shared ledger. The independent closure audit verified 108 preserved files, all 14 frozen v4 files and every archived ledger prefix, with no orphan reservations.

</details>

<a id="pilot-status-20261010"></a>
## Pilot results — October 10, 2026

**This feasibility pilot did not produce a usable critic comparison.** Across 12 tasks and six conditions, all 72 first attempts were recorded, but only 31 had valid terminal judgments; none of those 31 succeeded. The pilot primarily exposes actor-output and browser reliability limits. Shepherd guided produced no valid episode, so this pilot cannot establish its critic effectiveness or rank the two reward models. [Aggregate results and accounting](webprm_results/pilot_20261010.json).

“Attempted” includes browser startup and initial navigation failures. “Valid” requires a completed episode, saved rollout/screenshots and a complete, parseable terminal-judge response; it does not require error-free actions or a separate page-availability screen. Overall success is recorded successes divided by the 12 fixed tasks per condition, including invalid attempts in the denominator. Valid-only success uses the valid count; it is undefined when that count is zero.

| Condition | Attempted | Valid | Invalid | Successes | Overall success (%) | Valid-only success (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shepherd unguided | 12 | 5 | 7 | 0 | 0.0 | 0.0 |
| Shepherd random candidates | 12 | 1 | 11 | 0 | 0.0 | 0.0 |
| Shepherd 8B guided | 12 | 0 | 12 | 0 | 0.0 | — |
| Arbiter unguided | 12 | 9 | 3 | 0 | 0.0 | 0.0 |
| Arbiter random candidates | 12 | 8 | 4 | 0 | 0.0 | 0.0 |
| Arbiter 7B guided | 12 | 8 | 4 | 0 | 0.0 | 0.0 |

These are complete **attempt counts**, not 72 valid evaluations. All guided-minus-control success differences are zero percentage points; relative gains are undefined because control success is zero. The all-zero paired percentile bootstrap degenerates to [0, 0] and does **not** establish equivalence. Arbiter guided has eight common-valid tasks with unguided and seven with random selection, all unsuccessful; Shepherd guided has no common-valid comparison.

<details>
<summary>Why valid coverage is low</summary>

| First-attempt failure category | Episodes |
| --- | ---: |
| Shepherd native parser found zero or multiple actions | 24 |
| Initial website-load timeout, two sites across all six conditions | 12 |
| Collection deadline | 2 |
| Arbiter local critic timeout | 1 |
| Browser closed during an episode | 1 |
| Browser startup failure in the final CPU control attempt | 1 |
| Total invalid | 41 |

Shepherd guided encountered 33 malformed choices among 220 samples. Its frozen rule aborts a whole 20-sample response if any choice fails native parsing: ten of eleven responses aborted, discarding 167 otherwise one-action choices before selection. The guided condition's twelve failures comprise ten parser failures and two initial website-load failures. This diagnoses a brittle actor-output policy; it does not establish a critic-scoring defect. Changing that policy would define a new intervention and must preserve this cohort.

Fifteen of the 31 valid episodes contain action-execution errors. All 121 Arbiter coordinate-bound errors reference IDs absent from the contemporaneous saved accessibility tree, which supports stale or ungrounded actor targets rather than proving an observation-to-action mapping defect. These diagnostic frequencies come from different trajectories and are not matched estimates of critic effects.

</details>

<details>
<summary>Execution, recovery and budget accounting</summary>

The approved cap is one H200, eight CPUs and 120 GiB for 7,200 total allocation seconds, including retries, plus USD 30 shared actor/judge spending. Four terminal Slurm attempts consumed 20, 6,076, 4 and 849 seconds: **6,949 seconds total**. They include CUDA-toolkit discovery and scheduler-duration validation failures; neither made an API call. Scientific source, model, sampling and cohort identities were preserved.

The final random-control cell required no local critic. Its CPU attempt failed during Chromium startup before an API call. A zero-API diagnostic reproduced failure with the project temporary directory and passed with a short local temporary directory; this comparison does not isolate path length from filesystem behavior. Subsequent infrastructure recovery is stored separately, with the first-attempt table unchanged. All CPU attempt and diagnostic time is conservatively accounted against the unused original allowance. No additional GPU allocation was submitted.

Two separately retained recovery attempts added no valid outcome: the first stopped before transmission because its standalone launcher omitted credential-file loading; after that repair, the second stopped with a browser-closed error after four turns. Its parent receipt was interrupted, so actual parent duration is unavailable; the full configured 217-second allowance is conservatively accounted, and a host process check confirmed no remaining worker/browser. Verified GPU allocation time plus this conservative CPU accounting totals **7,194.93 seconds**. The final shared ledger contains **USD 0.98945495 settled** and **USD 0.00609330 retained as a conservative reservation**, totaling USD 0.99554825 against the USD 30 cap. The reservation belongs to the known pre-transmission credential-loading failure; it is preserved rather than silently released.

The primary cohort settled **USD 0.98745575 across 1,034 API calls**, with no unresolved reservations at its audit cutoff. Supplemental accounting uses the same USD 30 ledger and is reported separately in the aggregate artifact. Every primary result, saved verdict and earlier API entry is preserved. The original complete-cohort audit uses its archived budget snapshot; a separate audit reconciles the later shared ledger with supplemental journals.

</details>

<details>
<summary>Frozen protocol, serving validation and reproducibility</summary>

The actor is `gpt-4o-mini-2024-07-18`, temperature 1. Shepherd uses top-p 1, up to 512 output tokens and one sample for unguided versus 20 for random/guided selection. Arbiter uses the native five-BLOCK response, top-p 0.9 and up to 2,048 output tokens; its unguided control chooses the first candidate. Both critics use temperature 0. The terminal judge is `o4-mini-2025-04-16` with the Online-Mind2Web/AgentTrek success prompt, a 4,096-token response cap and a 30-turn episode limit. Truncated actor, critic and judge responses are rejected. The token caps and serving engine are disclosed adaptations of the released interfaces.

BrowserGym 0.14.3 and Playwright 1.44.0 use the shared Chromium-1208 executable, with a fresh browser per episode and native author observation/action interfaces. Live websites, differing episode durations and later continuations limit contemporaneous comparability; this is a 12-task feasibility pilot, not a full-300 benchmark or exact reproduction of a published paper result.

Both reward servers passed simultaneous H200 startup and local preflight. Explicit KV caches are 32,768 tokens for Shepherd (4.5 GiB) and 65,536 for Arbiter (3.5 GiB), with a 0.70 memory-profiling ceiling; observed combined GPU use was approximately 40 GiB. Repeated temperature-zero Shepherd preflight responses produced nonidentical text and scores (0.00004055 and 0.00005150), so deterministic serving is not established.

All **40 focused tests passed**, including complete/partial cohort handling, recovery-budget checks and immutable API snapshot validation. The independent audit checks cohort/identity hashes, saved result equality, readable screenshots, full terminal-judge evidence, request journals, API accounting and preservation across attempts. Task payloads, screenshots, raw trajectories, credentials and internal operational paths remain private; the linked artifact contains aggregate results only.

</details>

<a id="supported-harness-20261009"></a>
## Supported-harness reproduction — October 9, 2026

Both released baselines support GPT-4o-mini as the actor, but their candidate generation, observations and critic interfaces differ. Compare each critic's improvement over controls in its own supported harness; a difference between those improvements is a comparison of complete systems, not an isolated estimate of reward-model quality.

The diagnostic used the same 12 prespecified OM2W tasks for six conditions. All conditions use the frozen `gpt-4o-mini-2024-07-18` actor and the same terminal-success judge. Random-selection controls distinguish critic selection from the benefit of generating more candidates. Twelve tasks provide a mechanical pilot with wide uncertainty, not a definitive ranking.

| Family | Unguided control | Candidate-generation control | Reward-guided condition |
| --- | --- | --- | --- |
| Web-Shepherd | One sampled action | Random choice among up to five frequency-ranked actions from 20 samples | WebShepherd 8B checklist-probability selection from that pool |
| WebArbiter | First candidate from the released five-BLOCK policy response | Random choice among the five candidates | WebArbiter 7B pairwise knockout tournament |

Arbiter's unguided control preserves its released multi-candidate prompt; it is not a separate single-action prompt. Shepherd's online search loop is reconstructed from the paper because its repository does not release that loop. Its actor prompt comes from the authors' public BrowserGym demo. Published WebArena-Lite results and static Mind2Web preference accuracy must remain separate from live Online-Mind2Web success.

<details>
<summary>Source evidence and reproduction limits</summary>

| Component | Source | Immutable revision |
| --- | --- | --- |
| Shepherd critic prompts, label-probability parser and score | [Official repository](https://github.com/kyle8581/Web-Shepherd) | `b87898f614e43d7a5bdb738eaf6b01d416be24c2` |
| Shepherd actor prompt and preprocessing | [Author demo source](https://huggingface.co/spaces/LangAGI-Lab/Web-Shepherd-Demo/blob/main/app.py) | File digest recorded in the pilot manifest |
| Arbiter policy prompt/parser, observations, actions and tournament | [Official repository](https://github.com/YaoZ720/WebArbiterCode) | `764ec2670750c50388bac56efb1a3989fd5c1b54` |
| Shepherd 8B weights | [Author model](https://huggingface.co/LangAGI-Lab/WebShepherd_8B) | `37b57dbaa1e4c0c19b062172a7077b0b632a66d0` |
| Arbiter 7B weights | [Author model](https://huggingface.co/ZYao720/WebArbiter-7B) | `6518479bcf55043b92d5ffdc34e23ce7ed59bf65` |

The Shepherd serving script names `WebShepherd/WebShepherd_8B`. Public repository metadata shows the same 15 file objects, including all weight-shard LFS digests, as the staged LangAGI-Lab checkpoint. This is checked rather than assumed from model names.

Arbiter's released viewer critic prompt differs from the benchmark prompt used with its released model. The pilot uses the official benchmark's WebArena prompt template, recovered with byte-exact roundtrips over all 4,600 released rendered rows. This combination of viewer actor and benchmark critic is an explicit adaptation, not evidence that the paper used precisely this online loop. The static rows supply a template only; their labels and trajectories are not live-task inputs.

The completed manifest records hosted model snapshots, dependency versions, browser lifecycle, token limits, serving engine, malformed-output policy and thought/action splitting. Actor generation, full critic requests, response log probabilities, selection brackets and execution errors are saved privately. Highest available source fidelity does not establish exact reproduction of an unreleased paper harness.

</details>

<details>
<summary>Budget, analysis and evidence retention</summary>

The existing pilot authorization is one H200 for two hours, eight CPUs, 120 GiB RAM and a shared USD 30 API cap, including all retries. It does not authorize full-300 collection or a new allocation. Model staging and public local fixtures use no paid API calls. The allocation controller must own and await every server and evaluation worker, preserve consumed time across retries, and stop within both caps.

Report successes, valid attempts and planned tasks in separate count columns. Primary success uses all planned tasks in each arm as the denominator; report valid-only success separately. Compare within-family paired differences in percentage points with a paired task bootstrap and common-valid sensitivity. Relative percentage gain is undefined when the control has zero success. Incomplete cohorts cannot silently become completed comparisons, and invalid or malformed results cannot be replaced by whichever retry succeeds.

Retain task-file/cohort hashes, immutable model revisions, source/dependency manifest, shared API ledger, allocation attempt ledger, every task rollout and terminal verdict. Publish only aggregate results and methodology; task payloads and browser evidence remain local.

</details>
