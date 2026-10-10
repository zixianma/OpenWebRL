# ARM inference scaling

- **SFT actor + Piotr SelectionARM:** **+5.67 ± 1.15 percentage points** (mean ± sample SD) across three complete runs.
- **RL actor + Piotr:** **42.67% → 54.00%**, a **+11.33-point gain [+5.33, +17.33]** in one complete paired run.
- **Selection rule:** random-of-five reaches **33.33%** and highest-likelihood-of-five **26.67%**, versus **33.00%** for SFT and **40.00%** for Piotr. Controls were collected separately.
- **Other selectors:** Sol/GPT-5.5 add **10–12 points**; Luna/Kev add about **9 points** under a different protocol. These are not controlled cross-protocol rankings.
- **Still unresolved:** whether 50 steps or an RL-trained actor increases Piotr's gain relative to the SFT 30-step comparison.

Every main result uses the same **300 Online-Mind2Web tasks**. Invalid and unjudged episodes count as zero. “Success” is the canonical o4-mini/AgentTrek verdict, which permits partial progress. **N** means candidates generated per decision; one is executed. Gains are percentage points (**pp**); intervals quantify task sampling, not judge error or website drift.

## Piotr SelectionARM: three complete runs

<a id="sft-piotr-repeat-tracker-20261006"></a>

Official OpenWebRL-4B-SFT, **30 steps, T=0.7, top-p=0.9, 1,024 output tokens**. Compare one actor sample with greedy Piotr selection from five full candidates.

| Run | Tasks per arm | SFT successes | Piotr successes | SFT success | Piotr success | Gain, pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oct 6, seed 43 | 300 | 100 | 115 | 33.33% | 38.33% | +5.00 |
| Oct 6, seed 44 | 300 | 98 | 113 | 32.67% | 37.67% | +5.00 |
| Oct 7, seed 45 | 300 | 99 | 120 | 33.00% | 40.00% | +7.00 |
| **Mean ± sample SD** | 300 | — | — | **33.00 ± 0.33%** | **38.67 ± 1.20%** | **+5.67 ± 1.15** |

Mean gain: **+5.67 pp**, paired 95% interval **[+2.67, +8.67]**. The same 300 tasks repeat across runs: 900 episodes per arm, 300 task clusters. SD describes the three run rates; the interval resamples task clusters with all runs retained. [Aggregate](arm_results/selectionarm_piotr_sameday30_20261007/aggregate.json) · [Protocol and accounting](ARM_INFERENCE.md#selectionarm-piotr-sameday30-20261007).

## Controls and variants

<a id="random5-sameday-results-20261007"></a>

Same SFT actor and decoding as above; **300 tasks per arm in every row**.

**30 steps: selection rules, using the Oct 7 SFT control**

| Condition | N | Tasks | Successes | Success | Gain vs SFT, pp | Paired 95% interval, pp |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SFT alone — baseline | 1 | 300 | 99 | 33.00% | — | — |
| Random-of-five | 5 | 300 | 100 | 33.33% | +0.33 | [−4.67, +5.67] |
| Highest likelihood of five | 5 | 300 | 80 | 26.67% | −6.33 | [−11.33, −1.33] |
| **Piotr SelectionARM, Oct 7** | 5 | 300 | 120 | 40.00% | +7.00 | [+2.33, +11.67] |

**50 steps: Piotr, using a fresh 50-step SFT control**

| Condition | N | Tasks | Successes | Success | Gain vs SFT, pp | Paired 95% interval, pp |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SFT alone — baseline | 1 | 300 | 92 | 30.67% | — | — |
| Piotr | 5 | 300 | 120 | 40.00% | +9.33 | [+4.00, +14.67] |

**30 steps: RL-task-trained selector, using the earlier API-study SFT control**

| Condition | N | Tasks | Successes | Success | Gain vs SFT, pp | Paired 95% interval, pp |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SFT alone — baseline | 1 | 300 | 95 | 31.67% | — | — |
| RL-task-trained SelectionARM | 5 | 300 | 123 | 41.00% | +9.33 | [+4.00, +14.67] |

- **Random control:** pick uniformly among five candidate indices at **every step**, including duplicates and malformed candidates. It reuses the Oct 7 SFT/Piotr episodes above. Piotr beats random by **+6.67 pp [+1.00, +12.33]**; random shows no detectable gain over SFT.
- **Likelihood control:** choose the highest mean native base-policy log-probability of the full response, including reasoning. It trails random by **−6.67 pp [−12.00, −1.33]** and Piotr by **−13.33 pp [−18.67, −8.00]**. This is one run with reused controls, not a general test of confidence gating. [Action-only follow-up](ARM_INFERENCE.md#action-only-likelihood-20261008).
- **Horizon:** the 50-step pair has a fresh SFT control. Its gain exceeds the same-day 30-step gain by **+2.33 pp [−4.67, +9.33]**; no clear horizon effect.
- **Selector training:** RL-task-trained SelectionARM changes the **selector**, while the actor stays SFT. It reuses the earlier SFT control from the API-selector study below; this is not a matched ranking against Piotr.

The Oct 7 Piotr row is the third run above, not an extra repeat. Collections used separate browser episodes and times. [Random-control audit](ARM_INFERENCE.md#selectionarm-random5-sameday30-20261007) · [50-step audit](ARM_INFERENCE.md#selectionarm-piotr-steps50-results-20261007) · [RL-task-selector audit](ARM_INFERENCE.md#selectionarm-rltasks-repeat-results-20261007).

## API and other selectors

<a id="api-selector-results-20261007"></a>
<a id="local-sft-selector-results-20261006"></a>
<a id="matched-sft-control"></a>

Official SFT actor; one complete run per condition. **Compare each selector with its own protocol's SFT baseline.**

| Protocol | Selector | N | Tasks | Successes | Success | Gain, pp | Paired 95% interval, pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| September protocol, Oct reproduction | None / SFT | 1 | 300 | 95 | 31.67% | — | — |
| September protocol, Oct reproduction | GPT-5.6 Sol | 5 | 300 | 130 | 43.33% | +11.67 | [+6.33, +17.00] |
| September protocol, Oct reproduction | GPT-5.5 | 5 | 300 | 126 | 42.00% | +10.33 | [+5.00, +15.67] |
| Local v2 | None / SFT | 1 | 300 | 94 | 31.33% | — | — |
| Local v2 | GPT-6 Luna, medium | 5 | 300 | 121 | 40.33% | +9.00 | [+3.33, +14.67] |
| Local v2 | Jev 1.13.0 | 5 | 300 | 100 | 33.33% | +2.00 | [−3.67, +7.67] |
| Local v2 | Kev27B | 5 | 300 | 122 | 40.67% | +9.33 | [+3.67, +15.00] |

| Protocol | Actor T / top-p | Actor token cap | Selector input | Episode limit |
| --- | --- | ---: | --- | --- |
| September protocol | 0.7 / 0.9 | 1,024 | Current screenshot + full candidates | 30 turns |
| Local v2 | 1.0 / 0.95 | 4,096 | Text/DOM + full candidates + five recent actions; no images | 30 browser operations / 60 decisions |

Sol/GPT-5.5 use medium reasoning and 2,048 selector tokens; Luna uses medium reasoning and 4,096. Jev/Kev use choice classifiers.

**Sol reproduces a similar rate to September:** 43.33% now versus 44.00% then. Sol versus GPT-5.5 is inconclusive: **+1.33 pp [−3.67, +6.33]**. Luna/Kev differ by one success; Jev's gain interval includes zero. [API aggregate](arm_results/api_selector_september_reproduction_20261006/aggregate.json) · [Local v2 audit](arm_results/local_sft_selector_controlled_20261006.json) · [Local v2 protocol](arm_results/local_inference_rerun_plan_20261006.json).

<a id="sol-candidate-counts-20261007"></a>
## Sol: does increasing the candidate count help?

**N=10 improves over N=3 by +6.67 pp [95% paired interval: +2.00, +11.67]**, at **3.28× the mean actor output tokens**. Both arms used the corrected September protocol above and were collected together on the same 300 tasks.

| Condition | N | Collection | Tasks | Successes | Success | Gain vs SFT, pp | Valid tasks | Invalid tasks | Valid-only success |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT alone / no selector | 1 | Earlier control | 300 | 95 | 31.67% | — | 273 | 27 | 34.80% |
| SFT + Sol | 3 | Fresh paired study | 300 | 121 | 40.33% | +8.67 | 266 | 34 | 45.49% |
| SFT + Sol | 5 | Earlier reference | 300 | 130 | 43.33% | +11.67 | 268 | 32 | 48.51% |
| SFT + Sol | 10 | Fresh paired study | 300 | 141 | 47.00% | +15.33 | 264 | 36 | 53.41% |

N=10 wins alone on 38 tasks; N=3 wins alone on 18. Invalid episodes remain zero in the primary denominator. The SFT-alone control and N=5 reference come from the [earlier API-selector study](#api-selector-results-20261007) under the same corrected September protocol; comparisons with N=3/N=10 include collection-date differences. These are canonical o4-mini/AgentTrek scores, which permit partial progress; three targeted positive flags remain documented without relabeling.

<details>
<summary>Cost and paired sensitivity</summary>

| Candidates | Mean actor output tokens/task | Mean episode latency, s | All-attempt selector charge/reservation, USD |
| --- | ---: | ---: | ---: |
| 3 | 14,342 | 219.38 | 96.18 |
| 10 | 47,051 | 249.91 | 150.51 |

On the 255 tasks valid in both arms, the gain is +7.45 pp [95% paired interval: +1.96, +12.94]. Primary intervals use 10,000 task bootstrap draws, seed 42; they do not include judge error or website drift. Latency is time to result and includes overlapping work, not GPU compute. Shared campaign usage was 5.57 H200-hours and 618 browser attempts, including interrupted attempts. API charges/reservations are conservative ledger values, not invoices.

</details>

[Aggregate](arm_results/sol_candidate_scaling_september_20261007/aggregate.json) · [Accounting](arm_results/sol_candidate_scaling_september_20261007/accounting.json) · [Protocol and recovery audit](ARM_INFERENCE.md#sol-candidate-scaling-september-20261007).

<a id="sol-action-diversity-20261008"></a>
### How much action diversity do the extra candidates add?

**At the same saved N=10 states, candidates 4–10 add 3.61 distinct parsed action specifications**, on average, beyond the first three (95% task-bootstrap interval **[3.45, 3.76]**). Sol selects a specification absent from the first three in **38.1%** of decisions, averaging within each task and then equally across tasks.

| Candidates retained from each N=10 panel | Mean distinct specifications | 95% interval |
| --- | ---: | --- |
| First 3 | 2.37 | [2.33, 2.42] |
| First 5 | 3.52 | [3.43, 3.61] |
| All 10 | 5.98 | [5.78, 6.17] |

These are **4,205 decisions from 279 tasks**; 21 episodes without candidate panels are excluded from diversity means. Specifications distinguish tool-call sequences and exact arguments, including coordinates and final-answer text. They exclude reasoning and formatting. Merging terminal wording and grouping coordinates into 50-unit bins reduces the added diversity to **1.33 classes [1.24, 1.43]**. This sensitivity check is not semantic UI-target matching; the saved evidence lacks target identities. The analysis establishes additional proposals, not their usefulness or a causal explanation of the success gain.

<details>
<summary>Actual N=3/5/10 runs, diversity plot and method</summary>

| Actual Sol run | Audited episodes | Tasks with panels | Saved decisions | Mean distinct specifications | 95% interval | Unparseable candidates | Total candidates |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| N=3 | 300 | 279 | 4,323 | 2.27 | [2.22, 2.33] | 35 | 12,969 |
| N=5, earlier collection | 300 | 279 | 4,221 | 3.44 | [3.33, 3.53] | 53 | 21,105 |
| N=10 | 300 | 279 | 4,205 | 5.98 | [5.78, 6.17] | 120 | 42,050 |

These actual runs visit different states; N=5 also has a different collection date. The main table instead holds each N=10 panel fixed. It does not rerun the selector on smaller panels or estimate their counterfactual success. Averaging every possible 3-of-10 and 5-of-10 subset gives 2.37 and 3.53 distinct specifications, consistent with the ordered-prefix comparison.

![Additional candidate diversity at the same saved states](arm_results/sol_action_diversity_20261008/action-diversity.png)

The selected specification is absent from the first three in 1,378 of 4,204 decisions with a parseable selection: pooled rate 32.78%, versus the task-weighted 38.13% [35.77, 40.54] above. Beyond the first five, the task-weighted added count is 2.46 [2.34, 2.57], and selection novelty is 24.94% [22.91, 27.11]. Later indices that duplicate an earlier specification are not novel.

The native Qwen25 parser and ordered tool-call sequences match the frozen execution path. Parsing does not establish executability: missing required arguments and generation-length stops are audited separately. Normalizing executor defaults and ignored arguments leaves 3.59 additional specifications beyond three; merging terminal wording alone leaves 2.74. Coordinate bins remain a coarse proxy, not a bound on semantic diversity.

Means first average observed decisions within task, then weight tasks equally. Intervals use 10,000 task-bootstrap samples. Input hashes, all 900 episode identities, candidate counts, selector indices and parser behavior were independently checked; ten focused tests passed. All analysis used saved evidence, with no new browser or model calls. [Aggregate metrics and definitions](arm_results/sol_action_diversity_20261008/aggregate.json).

</details>

<a id="sol61-n5-launch-20261009"></a>
## GPT-6.1 Sol at N=5: actor controls and reasoning effort

**WebVoyager shows a measured benefit from the Sol61 medium N=5 pipeline over its matching SFT N=1 control:** **57.98% versus 46.72%**, a **+11.26 pp** difference with a **95% paired task-bootstrap interval of [+7.39, +15.13]**. DeepShop remains inconclusive: **46.00% versus 42.67%**, or **+3.33 pp [−5.33, +11.33]**. These controls match weights, decoding, tasks, browsers and native judges, but were collected sequentially; the intervals do not cover website/judge drift or run-to-run variability.

**Higher selector effort did not establish an improvement on Online-Mind2Web:** medium scored **45.67%** and high **43.67%**, with high minus medium **−2.00 pp [−7.33, +3.33]**. Those contemporaneous runs use the same 300 tasks, five SFT action proposals per decision, and a shared 2,048-token selector output cap.

| Benchmark | Condition | N | Collection | Tasks | Successes | Valid | Invalid | Overall | Valid-only |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Online-Mind2Web | **SFT alone: closest historical control** | **1** | **Oct 7, seed 45** | **300** | **99** | **271** | **29** | **33.00%** | **36.53%** |
| Online-Mind2Web | SFT alone: earlier API-study sensitivity | 1 | Oct 7, seed 42 | 300 | 95 | 273 | 27 | 31.67% | 34.80% |
| Online-Mind2Web | Sol61 medium | 5 | Oct 9, seed 42 | 300 | 137 | 269 | 31 | 45.67% | 50.93% |
| Online-Mind2Web | Sol61 high | 5 | Oct 9, seed 42 | 300 | 131 | 257 | 43 | 43.67% | 50.97% |
| DeepShop | SFT alone: matching control | 1 | Oct 9 | 150 | 64 | 145 | 5 | 42.67% | 44.14% |
| DeepShop | Sol61 medium | 5 | Oct 9 | 150 | 69 | 147 | 3 | 46.00% | 46.94% |
| WebVoyager | SFT alone: matching control | 1 | Oct 9 | 595 | 278 | 573 | 22 | 46.72% | 48.52% |
| WebVoyager | Sol61 medium | 5 | Oct 9 | 595 | 345 | 565 | 30 | 57.98% | 61.06% |

All displayed cohorts are verified complete. Against the closest historical SFT control, Sol61 medium and high are descriptively **+12.67 and +10.67 percentage points** overall. Against the earlier seed-42 control, the gaps are +14.00 and +12.00 points. These comparisons reuse October 7 outcomes; they do not isolate a causal selector gain from collection-date, seed or runtime differences. [Closest control aggregate](arm_results/selectionarm_piotr_sameday30_20261007/aggregate.json) · [Earlier API-study control](arm_results/api_selector_september_reproduction_20261006/aggregate.json).

Overall includes invalid records as zero; valid-only excludes each condition's invalids. On the **same 250 tasks valid in both Sol61 OM2W conditions**, medium succeeded on **134** and high on **128**: high minus medium **−2.40 pp [−8.40, +3.60]**. Neither comparison supports a reliable effort effect. This tests reasoning effort within N=5. The historical controls match the nominal SFT actor/judge protocol, but there is no contemporaneous N=1 control.

<a id="sol61-actor-controls-20261009"></a>

**Both matching SFT N=1 controls are verified.** They preserve the official-SFT weights, actor decoding, local browser and native benchmark judge used by Sol61, with zero selector calls. The positive WebVoyager result persists on the tasks valid in both conditions; DeepShop's intervals include zero in both analyses.

| Benchmark | Paired population | Tasks | N1 successes | N5 successes | N5 − N1, pp | Paired 95% interval, pp |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| WebVoyager | All tasks; invalids count as zero | 595 | 278 | 345 | +11.26 | [+7.39, +15.13] |
| WebVoyager | Valid in both conditions | 547 | 274 | 337 | +11.52 | [+7.31, +15.54] |
| DeepShop | All tasks; invalids count as zero | 150 | 64 | 69 | +3.33 | [−5.33, +11.33] |
| DeepShop | Valid in both conditions | 143 | 63 | 69 | +4.20 | [−4.20, +12.59] |

WebVoyager has 107 N5-only successes and 40 N1-only successes; DeepShop has 23 and 18. Pairing controls task identity. Common-valid estimates exclude 48 and seven tasks respectively, conditioning on availability in both arms. Intervals use 10,000 paired task-bootstrap resamples with seed 42. The WebVoyager result supports this N=5 selection pipeline under the saved native labels; it does not isolate selection from the additional proposal generation or establish a general gain across benchmarks. DeepShop's result establishes neither a reliable benefit nor equivalence. [Aggregate, paired analysis and audit hashes](arm_results/sol61_n5_launch_20261009.json).

<details>
<summary>Control audits, accounting and remaining comparability gaps</summary>

The control campaign audited all 745 records with zero selector or teacher calls. Each condition used one H200, eight CPUs, 120 GiB RAM and eight local browsers. Every attempt remains charged to its own approved cap; no budgets were transferred.

| SFT N1 control | Allocation seconds | Cap, seconds | Browser attempts | Browser cap | Judge HTTP calls | Call cap | Conservative judge charge, $ | Cap, $ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WebVoyager | 8,796 | 21,600 | 595 | 655 | 420 | 2,620 | 1.8753300 | 30 |
| DeepShop | 4,954 | 10,800 | 150 | 165 | 101 | 660 | 0.5487225 | 10 |

The WebVoyager audit checked all 595 records, 7,595 one-proposal decisions, 7,604 actor receipts, all available saved images and native judge bindings. Of 573 native-valid N1 trajectories, 417 received parsed GPT-4o verdicts and 156 retained native unjudged zeros (137 turn limits, 14 generation-length limits and 5 format failures). The 22 invalids comprise 5 missing-turn-sample exceptions, 4 environment-step errors, 9 failed actor requests, 3 native judge parse failures and 1 episode timeout. The audit preserves partial exception payloads and checks their available saved history; it does not invent missing terminal images or verdicts.

DeepShop's full review checked all 150 records, available saved images, 2,618 one-proposal decisions, 2,619 actor receipts and native verdicts. Of 145 native-valid N1 trajectories, 101 received a GPT-4o verdict and 44 retained unjudged native zeros (42 turn limits and two generation-length limits). Its five invalids comprise four environment-step errors and one failed actor request.

Both controls have finished W&B runs and reconciled lifetime accounting, with no selective outcome retries or rejudging. DeepShop has 150 explicit browser-exit messages. WebVoyager has 590 normal exit messages; five initial screenshot/reset failures take a separate frozen, awaited cleanup path, supported by later port reuse, no cleanup warnings and final scheduler/cgroup evidence. Pool-counter logs were not retained for those five. Controller-owned process groups, zero-exit Slurm batch/extern steps and released controller locks were verified. Postexit SSH was denied by PAM, so this is scoped resource teardown evidence rather than a physical whole-node scan.

Both OM2W controls use the same 300-task file, official SFT revision, pinned actor-policy hash, local browsers, T=0.7/top-p=0.9/native default top-k, 1,024 actor tokens, 30 turns, full text history/latest screenshot and canonical o4-mini/AgentTrek judge as Sol61. The shared actor, browser, reward and native evaluator source files are byte-identical. The October 7 seed-45 control is the latest completed matching control located; the earlier API-study control also matches Sol61's seed-42/server-4200 settings.

The seed-45 control uses server seeds 4500/4501, versus 4200/4201 for Sol61 and the API-study control. FlashInfer ignores per-request seeds, so matching seed labels do not imply identical trajectories. Campaign wrappers, interleaving and collection times differ; core-file identity does not prove complete runtime equivalence. Website/judge drift and differing invalid sets remain. No new paired interval or common-valid estimate is claimed for the historical OM2W controls. Existing native outcomes, including valid unjudged zeros, are unchanged.

</details>

<details>
<summary>Paired comparison, latency and outcome definitions</summary>

Across all 300 OM2W tasks, both conditions succeeded on 101, only medium on 36, only high on 30, and neither on 133. Intervals use the frozen paired task bootstrap: 10,000 resamples, seed 42. They quantify task sampling, not website drift, model-alias drift or judge error. The common-valid analysis conditions on measurement availability in both conditions.

| Per-task mean, all 300 committed episodes | Medium | High |
| --- | ---: | ---: |
| Actor output tokens across all five proposals | 23,843.10 | 25,273.64 |
| Selector output tokens, including reasoning | 631.78 | 1,465.80 |
| Selector service seconds | 41.42 | 61.51 |
| Episode elapsed seconds | 225.18 | 259.35 |

These are committed-episode measurements, including invalid episodes; service times are summed within each episode and are not allocated GPU time. Two task shards interleaved medium and high, so per-shard allocation time cannot be attributed to one reasoning effort. Failed attempts and all API reservations are accounted separately.

Native unjudged zero outcomes remain valid: OM2W has 63 in medium and 68 in high, including step limits, generation-length failures and format failures. Its other 395 valid outcomes received the native o4-mini verdict. DeepShop N5 has 50 step-limit zeros and 97 native GPT-4o-judged outcomes. Invalid records remain saved and excluded only from valid-only rates. OM2W's invalids include browser reset/navigation failures, missing observations, a native episode timeout and actor requests whose prompt plus reserved output exceeded the context limit. The frozen client guard checks prompt length alone; this boundary mismatch is retained as a harness limitation, with no selective retries or changed labels. The native judge action-history parser omits unsupported tool names, including observed `select_option` calls; saved canonical requests and verdicts are preserved. The audit verifies integrity and protocol, without independently rejudging semantic correctness.

</details>

<details>
<summary>Controlled protocol, endpoint verification and accounting</summary>

All four conditions use the corrected September proposal protocol: local browsers, actor T=0.7/top-p=0.9, native default top-k, 1,024 actor tokens and 30 turns. GPT-6.1 Sol selects an index from the full proposals and current screenshot, with the same 2,048-token output cap for medium and high. Incomplete responses retain their validity and cost consequences. The arms share the proposal protocol; stochastic trajectories and candidates differ. Recorded server seeds 4200/4201 do not establish deterministic replay because FlashInfer ignores request seeds.

Online-Mind2Web uses canonical o4-mini/AgentTrek; WebVoyager and DeepShop use their native GPT-4o prompts and parsers, up to 30 screenshots, and an explicit 4,096-token judge cap. The earlier GPT-5.6 Sol N=5 result is a historical reference. Earlier local RL actor-only WebVoyager/DeepShop cohorts use different weights and T=0.6/top-p=0.95/top-k=20/4,096-token actor decoding.

The OM2W audit covered all 600 records, 44,240 actor sampling receipts, 8,828 full N=5 selector input/trace bindings and 395 native judge bindings, preserving every available rollout/image and all invalid outcomes. Frozen source, finished W&B histories and owned-process teardown passed. Direct postexit node inspection was unavailable because SSH authentication was denied; teardown verification uses owned process-group receipts, scheduler/cgroup completion and released controller locks.

| OM2W accounting | Used or charged | Approved cap |
| --- | ---: | ---: |
| Shard 0 allocation seconds, all attempts | 10,558 | 21,600 |
| Shard 1 allocation seconds, all attempts | 9,602 | 21,600 |
| Medium physical browser attempts | 300 | 330 |
| High physical browser attempts | 300 | 330 |
| Medium selector HTTP requests | 4,296 | 9,900 |
| High selector HTTP requests | 4,532 | 9,900 |
| Medium judge HTTP requests | 206 | 1,320 |
| High judge HTTP requests | 189 | 1,320 |
| Medium selector conservative USD | 53.7529163 | 100 |
| High selector conservative USD | 58.9326232 | 200 |
| Medium judge USD | 1.7440951 | 5 |
| High judge USD | 1.6319413 | 5 |

Shared OM2W API ledgers are counted once across the two shards. All requests settled, with zero unresolved reservations. Usage-derived API cost is **$116.0483339**, including cache-write tokens; the authoritative conservative charge is **$116.0615759**. Nominal diagnostic cost fields omit cache-write premiums and are not used for accounting.

DeepShop's exhaustive audit covered all 150 records, 13,685 actor sampling receipts, 2,737 selector input/trace bindings, all 97 native judge bindings, and full pixel decoding of 2,364 unique committed images. The original 23 records and all interrupted-attempt evidence were preserved. Frozen source, finished W&B history, final scheduler accounting and owned-process teardown passed independent checks.

| DeepShop accounting | Used or reserved | Approved cap |
| --- | ---: | ---: |
| Allocation seconds, all attempts | 7,646 | 14,400 |
| Physical browser episode attempts | 159 | 165 |
| Selector HTTP requests, including one unresolved reservation | 2,870 | 4,950 |
| Judge HTTP requests | 97 | 660 |
| Selector conservative charged/reserved USD | 42.2210126 | 75 |
| Judge charged/reserved USD | 0.53135 | 10 |

Received DeepShop API usage costs **$42.5816541** at frozen prices, including cache-write tokens; an additional **$0.166405** remains reserved for the interrupted provider-error request. The authoritative conservative total is **$42.7523626**, including that reservation. These are receipt-based calculations, not an independently verified provider invoice. Every attempt remains charged; no budget was added or transferred.

The WebVoyager audit covers all 595 records, 35,580 actor sampling receipts, 7,103 full N=5 selector bindings and 463 native GPT-4o judge responses. Of the 565 valid outcomes, 461 received parsed native verdicts and 104 are native unjudged zeros: 100 step limits, three generation-length limits and one format failure. Two of the 463 judge responses lacked the required verdict tokens and remain invalid. The other 28 invalids comprise 13 actor HTTP400 generation failures, ten missing-turn-sample exceptions and five environment-step errors. All native outcomes are preserved without selective retries or rejudging. Frozen source, all 595 finished W&B task-history rows and owned process-group teardown passed. Physical postexit inspection was denied by the scheduler's PAM policy; teardown evidence is scoped to owned groups, scheduler/cgroup completion and the released controller lock.

| WebVoyager accounting | Used or charged | Approved cap |
| --- | ---: | ---: |
| Allocation seconds, all attempts | 14,024 | 36,000 |
| Physical browser episode attempts | 595 | 655 |
| Selector HTTP requests | 7,103 | 19,650 |
| Judge HTTP requests | 463 | 2,620 |
| Selector conservative USD | 85.5520876 | 200 |
| Judge USD | 2.0726950 | 30 |

All WebVoyager requests settled. Received API usage costs **$87.6141281** at frozen prices, including cache-write tokens; the authoritative conservative charge is **$87.6247826**.

The campaign ceilings remain **26 H200-hours**, **$625** for APIs and **1,480 physical browser attempts**, with eight concurrent episodes per job. Each job has one H200, eight CPUs and 120 GiB RAM; the two Online-Mind2Web shards have six hours each, WebVoyager ten and DeepShop four. Budgets are separate from branching. All **1,345 requested records** and the paired OM2W comparison are verified. The campaign used 41,830 allocation seconds (11.6194 H200-hours), 1,354 browser attempts and $246.4387211 in conservative API charges/reservations. The one unresolved DeepShop reservation remains charged; no budget was added or transferred.

</details>

[Results and approval aggregate](arm_results/sol61_n5_launch_20261009.json) · [Official model settings and pricing](https://developers.openai.com/api/docs/models/gpt-6.1-sol).

<a id="molmoweb-sol61-plan-20261010"></a>
## MolmoWeb 4B/8B: controls first; Sol selection awaits review

**The next result is the actor-only control.** Run official MolmoWeb 4B and 8B with **N=1 and zero selector calls** on all three benchmarks, then review their scores and protocol comparability before authorizing **N=5 + GPT-6.1 Sol medium**. No MolmoWeb performance result is available yet. The execution and spending gates currently allow N1 only; completion of the controls does not launch ARM automatically.

| Benchmark | Tasks per actor | Actors | N1 control episodes | Scoring |
| --- | ---: | --- | ---: | --- |
| WebVoyager | 595 | 4B, 8B | 1,190 | Native MolmoWeb GPT-4o |
| DeepShop | 150 | 4B, 8B | 300 | Native MolmoWeb GPT-4o |
| Online-Mind2Web | 300 | 4B, 8B | 600 | Native MolmoWeb and AgentTrek, reported separately |
| **Total** | **1,045 unique benchmark tasks** | **2** | **2,090** | **Six controls** |

The control report will show success counts, overall and valid-only success rates, validity and judged coverage, with protocol differences next to the scores. Invalid outcomes count as zero overall; missing infrastructure/judge outcomes remain unresolved. A partial run remains incomplete. Local browsers differ from the paper's Browserbase/advanced-stealth environment, so paper scores are contextual, not matched controls.

After user review and approval, the planned comparison is the **task-paired N=5 minus N=1 success difference**, with a paired task-bootstrap interval. Controls and ARM are now collected sequentially, so website changes over time remain a comparability caveat. This is not a repeated-seed estimate. The full potential comparison remains 4,180 episodes across 12 conditions; the six N5 conditions are blocked.

**Both Online-Mind2Web judges score the same saved actor episodes; they do not trigger extra continuations.** The native MolmoWeb judge decomposes the goal, assesses each supplied screenshot, then produces a final verdict: **S+2 logical calls for S screenshots**, before retries. The prior AgentTrek protocol instead uses the task, action history and last supplied actor observation in one terminal assessment, with its existing completed-episode eligibility rule. Truncated and format-ineligible episodes retain explicit unjudged zeros in that protocol. Report each judge's eligible/judged counts, overall and valid-only rates, and disagreement on the commonly judged subset. Never select the better judge outcome or pool the two scores.

<details>
<summary>Frozen scientific design, source and comparability</summary>

Use the official [MolmoWeb repository](https://github.com/allenai/molmoweb/tree/bab3fc3f1bff5b42c0624c073b68ddf9ad651d91) at `bab3fc3`, its native predictor, prompt, action parser and history handling. Checkpoints are [MolmoWeb-4B-Native](https://huggingface.co/allenai/MolmoWeb-4B-Native/tree/7623f7161ab00dd130e5a9f6e755d2a7cc32629c) and [MolmoWeb-8B-Native](https://huggingface.co/allenai/MolmoWeb-8B-Native/tree/b91f4f718120011a53643d3c1efc607f8b92e514). Both arms use temperature 0.7, top-p 0.8, one beam, at most 1,024 generated tokens, 30 decisions, ten prior actions and the current screenshot. N1 draws one action and makes no selector call. N5 draws exactly five from the same state/prompt, retains duplicate or malformed raw proposals, and lets Sol medium return a strict index into that unchanged panel; no candidate replacement, deduplication or actor fallback. The selected action enters native history once. Sol uses a 2,048-token output cap.

Local SimpleEnv browsers are used throughout. This is **not a reproduction of the paper's Browserbase/advanced-stealth environment**; historical paper success rates and the earlier OpenWebRL-SFT decoding settings are contextual comparisons, not matching controls. Native MolmoWeb judge inputs are unchanged. The AgentTrek adapter serializes MolmoWeb's native actions rather than applying the SFT-specific tool parser, preserving all executed actions. A separately saved immediate post-action screenshot is audit evidence and does not replace the last supplied observation in AgentTrek's input. Native OM uses o4-mini/2,048 completion tokens; AgentTrek retains o4-mini/4,096 completion tokens and its existing prompt/schema/seed/eligibility rules.

Source, checkpoint, task, candidate, screenshot, verdict and API identities are retained privately. Separate evaluation runs go to `openwebrl-evals`. Final completion requires exhaustive record/image/verdict checks, paired analysis, all-attempt accounting and owned-worker/browser teardown; count coverage alone is insufficient.

</details>

<details>
<summary>Approved resource ceilings and controls-only execution</summary>

Each row is a lifetime N1/N5 budget with **1 H200, 8 CPUs, 120 GiB and eight browser workers total**, including startup checks and every recovery attempt. CPUs were reduced from the approved ceiling of 16 to meet the cluster limit. At most four jobs run concurrently: **4 H200, 32 CPUs, 480 GiB and 32 browsers**. These are hard ceilings, not a throughput forecast or a guarantee of full coverage.

| Actor | Benchmark | All-attempt H200 hours | Sol ceiling, USD | Native judge ceiling, USD | AgentTrek ceiling, USD | Total API ceiling, USD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4B | WebVoyager | 16 | 200 | 70 | 0 | 270 |
| 4B | DeepShop | 8 | 75 | 22 | 0 | 97 |
| 4B | Online-Mind2Web | 12 | 100 | 300 | 10 | 410 |
| 8B | WebVoyager | 24 | 200 | 70 | 0 | 270 |
| 8B | DeepShop | 12 | 75 | 22 | 0 | 97 |
| 8B | Online-Mind2Web | 18 | 100 | 300 | 10 | 410 |
| **Total** | | **90** | **750** | **784** | **20** | **1,554** |

**The initial N1-only allocations reserve 45 of the approved 90 H200 hours:** 4B WV/DS/OM receive 8/4/6 hours; 8B receive 12/6/9 hours. The October 10 approved amendment adds $24 solely to N1 native judging: WebVoyager $30 → $40 and DeepShop $10 → $12 for each actor. Controls therefore have a $414 judge envelope ($404 native plus $10 AgentTrek), 2,300 browser starts and 69,000 local generations; selector allowance is zero. Compute and HTTP/browser counts are unchanged. [Amendment accounting](arm_results/molmoweb_n1_judge_amendment_20261010.json). Actual time from every attempt charges the corresponding lifetime row, and unused time is never reset or transferred. N5 remains blocked pending explicit review of the control results.

Native judge ceilings are N1/N5 $40/$30 for each WebVoyager actor, $12/$10 for each DeepShop actor, and $150/$150 for each OM actor; all Sol allowance belongs to N5. Each OM condition has a separate $5 AgentTrek allowance. Compute cannot transfer between jobs; API/browser caps cannot transfer between conditions or from other experiments. Aggregate limits are **4,600 browser starts, 414,000 local candidate generations and 256,360 actual HTTP attempts**. Every retry reserves its own physical HTTP attempt and worst-case cost before dispatch; unresolved charges retain their reservation. The first binding limit stops further paid work. Ordinary committed invalid outcomes are never selectively rerun, and a replacement receives only the unspent total allocation time.

</details>

## Actors alone

<a id="local-jev-actor-results-20261006"></a>

<!-- local-v2-actor-results:start -->
| Local v2 actor | Tasks | Successes | Success | Valid tasks | Valid-only success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official SFT | 300 | 94 | 31.33% | 254 | 37.01% |
| Jev + GPT-4.1-mini typing | 300 | 14 | 4.67% | 260 | 5.38% |
<!-- local-v2-actor-results:end -->

Jev is a separate actor system. Its judge-input adapter was repaired and all 23 valid DONE episodes rejudged from unchanged saved evidence. [Audit](arm_results/jev_actor_local_full300_20261006.json). Other actor runs and their limitations remain in the archive below.

## Piotr with an RL actor

<a id="piotr-rl-actor-20261007"></a>

| Actor | Selector | N | Tasks | Successes | Success |
| --- | --- | ---: | ---: | ---: | ---: |
| Released OpenWebRL-4B RL | None | 1 | 300 | 128 | 42.67% |
| Released OpenWebRL-4B RL | Piotr SelectionARM | 5 | 300 | 162 | 54.00% |

**Paired gain: +11.33 pp [+5.33, +17.33]**; 59 ARM-only wins and 25 actor-only wins. Both arms are fresh. Relative to the SFT setup, only actor weights change: **30 steps, T=0.7, top-p=0.9, 1,024 tokens, seed 45**, with the frozen SFT prompt/frontend and Piotr's original SFT base. This is one run, not the RL model's native prompting benchmark.

The gain exceeds the latest SFT pair's gain by **+4.33 pp [−3.00, +11.33]**: a stronger benefit on RL is not established. This comparison reuses SFT episodes collected at a different time. [Aggregate and plots](arm_results/selectionarm_piotr_rlactor30_20261007/aggregate.json) · [Four-outcome paired comparison](arm_results/selectionarm_piotr_rlactor30_20261007/sft_rl_gain_comparison.json) · [Protocol and final accounting](ARM_INFERENCE.md#selectionarm-piotr-rlactor30-20261007).

## Selective sampling: exploratory gate

Sample one candidate; below **−0.215576 nats/token** full-response mean base-policy log-probability, sample four more and let Piotr choose from all five. Otherwise execute the first.

| Method | Tasks | Successes | Success | Gain vs SFT, pp | Paired 95% interval, pp |
| --- | ---: | ---: | ---: | ---: | --- |
| Gated Piotr | 300 | 101 | 33.67% | +0.67 | [−4.33, +5.67] |

The gate triggered on **907 of 4,844 decisions (18.72%)**, averaging **1.749 candidates**. Its gain over SFT is inconclusive, and it trails always-Piotr by **−6.33 pp [−12.00, −0.67]**. Controls were collected separately; this does not establish noninferiority or matched wall-clock savings.

**Calibration limitation:** the exploratory cutoff is the 25th percentile of 278 saved SFT final-decision responses from these same evaluation tasks. The separate [benefit-based threshold study](ARM_INFERENCE.md#confidence-benefit-20261008) is now verified under the approved 59-pair fitting amendment: the frozen rule keeps the first action. On **20 usable pairs from 150 fixed held-out tasks**, it succeeds on 10 versus 7 for Piotr: **+15 pp [−5, +35]**, inconclusive. Replay coverage is **13.33%**; 17 usable pairs are from decisions 1–3 and 19 are from two sites. This is a single-intervention study, not an every-step gating result. The [follow-up proposal](ARM_INFERENCE.md#confidence-robust-20261008) replaces the 240-task placeholder with effect-based sizing: initially 448 usable fit and 448 usable held-out states for the 10-pp planning scenario, subject to replay reliability, fit stability and an expanded task pool. [Exploratory protocol and accounting](ARM_INFERENCE.md#likelihood-scaling-20261007).

<a id="training-task-pass5-pass8-20261006"></a>
## Frozen SFT training-pool coverage: five attempts to eight

**Observed task coverage rises from 64.65% to 73.20% (+8.55 percentage points).** Three fresh ordinary attempts rescue **171 of 707** tasks missed by the historical five. This uses the separate **2,000-task training screen** and the SFT checkpoint at iteration 0; it is not a held-out Online-Mind2Web evaluation or a trained-policy result.

| Coverage endpoint | Successful tasks | Rate |
| --- | ---: | ---: |
| Historical five attempts | 1,293 / 2,000 | 64.65% |
| Historical five + three fresh attempts on misses | 1,464 / 2,000 | **73.20%** |
| Rescue among historical misses | 171 / 707 | 24.19% |
| Rescue among five-valid-failure tasks | 166 / 682 | 24.34% |
| Rescue among misses containing invalid attempts | 5 / 25 | 20.00% |

The extension has **249 / 2,121 successful episodes (11.74% overall; 249 / 2,108 = 11.81% valid-only)**, with 13 invalid attempts. These episode rates apply only to historical misses, not to a fresh all-task pass@1 or pass@3 cohort. The collection uses local browsers, official SFT, T=0.8 / p=1 / k off, 1,024 response tokens, 15 turns and GPT-4.1/action_history judging.

All 10,000 historical records and 2,121 new primary records were audited, including terminal images for every valid new trajectory. The 1,293 prior successes required no new attempts. All four W&B runs finished with matching final metrics. Usage, including failed/interrupted physical attempts, is **21.23 H200-hours**, 2,137 browser dispatches, 602 judge calls and **$6.29 charged or reserved** within four separate approved caps. [Final aggregate and accounting](arm_results/rl_integration/training-sft-pass8-20261006.json).

This is historical-five plus fresh-three coverage across dates, not an exchangeable eight-sample estimator. Website drift remains possible. Native judge labels are unchanged; purposive checks found permissive positives, so the figures are judge-based rather than manually certified completion. The earlier guided follow-up rescued 78 / 682 five-valid-failure tasks; its dates and attempt budget differ from the fresh ordinary retries, preventing a matched-compute ARM comparison.

<details>
<summary>Original pass@k curve and rubric-difficulty analysis</summary>

The dataset difficulty value is the **number of reference rubric facts**, verified on all 2,000 tasks. Pass@1–4 below average subsets of the same five historical outcomes; invalid attempts count as no success.

| Rubric facts | Tasks | pass@1 | pass@2 | pass@3 | pass@4 | pass@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 1,297 | 37.22% | 51.27% | 58.91% | 63.82% | 67.23% |
| 6 | 542 | 32.47% | 44.58% | 51.51% | 56.31% | 59.96% |
| 7 | 111 | 38.92% | 51.44% | 56.94% | 60.00% | 62.16% |
| 8+ | 50 | 34.80% | 44.00% | 48.80% | 52.00% | 54.00% |
| All | 2,000 | 35.97% | 49.29% | 56.54% | 61.28% | 64.65% |

Spearman correlations with pass@1 through pass@5 are −0.052, −0.054, −0.057, −0.069 and −0.073. The pass@5 correlation's 95% website-bootstrap interval is **[−0.171, +0.033]** from 2,000 resamples over 76 websites. This selected difficulty ≥5 pool provides no clear monotonic relationship; rubric length is not a direct measure of browser interaction difficulty. Only 50 tasks have eight or more facts. [Aggregate counts, correlations, validity checks and input hashes](arm_results/rl_integration/training-sft-difficulty-passk-20261006.json).

</details>

Training with eight rollouts is a separate experiment on the original **2,102-task** pool. Its [uniform-G8 pilot reached iteration 10](RL_EVALUATION.md#uniform8-iter10-results-20261007), scoring 88 / 300 (29.33%) in an actor-only pass@1 evaluation. The pilot does not establish a training benefit from the screening coverage increase. Its [approved continuation to 60](ARM_INTEGRATION_PLAN.md#outcome56-uniform8-to60-20261008) remains queued.

## Supporting evidence and history

<details>
<summary>Plots and detailed evidence</summary>

| Study | Aggregate / audit | Compute or API cost | Latency | Tokens |
| --- | --- | --- | --- | --- |
| Piotr 30 steps / three-run summary | [Data](arm_results/selectionarm_piotr_sameday30_20261007/aggregate.json) | [Plot](arm_results/selectionarm_piotr_sameday30_20261007/cost.png) | [Plot](arm_results/selectionarm_piotr_sameday30_20261007/latency.png) | [Plot](arm_results/selectionarm_piotr_sameday30_20261007/tokens.png) |
| RL actor + Piotr, 30 steps | [Data](arm_results/selectionarm_piotr_rlactor30_20261007/aggregate.json) | [Plot](arm_results/selectionarm_piotr_rlactor30_20261007/cost.png) | [Plot](arm_results/selectionarm_piotr_rlactor30_20261007/latency.png) | [Plot](arm_results/selectionarm_piotr_rlactor30_20261007/tokens.png) |
| Exploratory gated Piotr / five-way comparison | [Data](arm_results/selectionarm_adaptive5_sft30_20261007/publication-aggregate.json) | [Plot](arm_results/selectionarm_adaptive5_sft30_20261007/cost.png) | [Plot](arm_results/selectionarm_adaptive5_sft30_20261007/latency.png) | [Plot](arm_results/selectionarm_adaptive5_sft30_20261007/tokens.png) |
| Highest likelihood of five | [Data](arm_results/selectionarm_likelihood5_sft30_20261007/publication-aggregate.json) | [Plot](arm_results/selectionarm_likelihood5_sft30_20261007/cost.png) | [Plot](arm_results/selectionarm_likelihood5_sft30_20261007/latency.png) | [Plot](arm_results/selectionarm_likelihood5_sft30_20261007/tokens.png) |
| Random-of-five | [Data](arm_results/selectionarm_random5_sameday30_20261007/aggregate.json) | [Plot](arm_results/selectionarm_random5_sameday30_20261007/cost.png) | [Plot](arm_results/selectionarm_random5_sameday30_20261007/latency.png) | [Plot](arm_results/selectionarm_random5_sameday30_20261007/tokens.png) |
| Piotr 50 steps | [Data](arm_results/selectionarm_piotr_steps50_20261007/aggregate.json) | [Plot](arm_results/selectionarm_piotr_steps50_20261007/cost.png) | [Plot](arm_results/selectionarm_piotr_steps50_20261007/latency.png) | [Plot](arm_results/selectionarm_piotr_steps50_20261007/tokens.png) |
| RL-task-trained selector | [Data](arm_results/selectionarm_rltasks_repeat_20261007/aggregate.json) | [Plot](arm_results/selectionarm_rltasks_repeat_20261007/cost.png) | [Plot](arm_results/selectionarm_rltasks_repeat_20261007/latency.png) | [Plot](arm_results/selectionarm_rltasks_repeat_20261007/tokens.png) |
| Sol / GPT-5.5 | [Audit](ARM_INFERENCE.md#api-selector-september-reproduction-20261006) | [Plot](arm_results/api_selector_september_reproduction_20261006/cost.png) | [Plot](arm_results/api_selector_september_reproduction_20261006/latency.png) | [Plot](arm_results/api_selector_september_reproduction_20261006/tokens.png) |

Learned-selector compute is an estimated generated-token proxy, not dollars or measured total FLOPs. Piotr 30-step plots cover the Oct 7 pair; the linked aggregate also contains the three-run summary. [Before/after branching studies](ARM_FORMULATIONS.md) are a separate experiment.

</details>

<details>
<summary>Efficiency, validity and accounting</summary>

| API study method | Mean latency, s | Mean metered input + output tokens | Selector API upper USD/task |
| --- | ---: | ---: | ---: |
| SFT alone | 149.7 | 148,549 | 0.000 |
| SFT + Sol | 201.1 | 798,293 | 0.374 |
| SFT + GPT-5.5 | 238.4 | 837,913 | 0.552 |

Tokens/API cost include all attempts; 107 failed actor requests lack token usage. Latency averages committed episodes and is not hardware-matched across studies. GPU/judge costs are separate. API reproduction used 6.84 allocated H200-hours; GPT-5.5's 13 output-cap failures remain zero.

| Local v2 condition | Tasks | Successes | Valid tasks | Invalid tasks | Valid-only success |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT alone | 300 | 94 | 254 | 46 | 37.01% |
| SFT + Luna | 300 | 121 | 255 | 45 | 47.45% |
| SFT + Jev | 300 | 100 | 250 | 50 | 40.00% |
| SFT + Kev | 300 | 122 | 247 | 53 | 49.39% |

<a id="local-browser-rerun-20261006"></a>
<a id="evidence-and-accounting"></a>

<!-- local-suite-live-status:start -->
| Study | Primary episodes | Separate smokes | Allocated H200-hours | Accounting |
| --- | ---: | ---: | ---: | --- |
| First two full300 Piotr repeats | 1,200 | 0 | 16.65 | [Final](arm_results/rl_integration/sft-piotr-repeats-final-20261006/combined.json) |
| Local v2 SFT + selectors | 1,200 | 24 | 22.88 | [Final](arm_results/local_sft_selector_controlled_20261006.json) |
| Local v2 Jev actor | 300 | 3 | 0 | [CPU/API ledger](arm_results/jev_actor_local_full300_20261006.json) |
<!-- local-suite-live-status:end -->

Local v2 judges completed episodes from full thoughts/actions and a fresh terminal screenshot. All valid records have terminal evidence; 186 of 194 invalid SFT records lack it and remain zero. One shard's cleanup used scheduler-cgroup/child-closure evidence instead of a post-exit process scan. Details are in the linked audits.

</details>

<details>
<summary>Incomplete October 6 run: 158 / 160 matched tasks</summary>

<a id="sft-piotr-three-run-summary-20261006"></a>
<a id="historical-corrected-partial-tracker-20261006"></a>

The first corrected run stopped at **493 of 900 episodes**: 158 tasks have all three arms; 160 have SFT/Piotr pairs. Every run used the same 300-task file. These are incomplete overlaps, excluded from the full300 headline means. The October 7 run supplies the third complete Piotr comparison.

[Partial results and accounting](ARM_INFERENCE.md#arm-rltasks-corrected-partial-20261006) · [Preserved 160-task three-run summary](arm_results/rl_integration/sft-piotr-three-run-summary-20261006.json) · [Historical saved-verdict audit](ARM_INFERENCE.md#arm-same-task-verdict-audit-20261006).

</details>

<details>
<summary>September reference: separate collection, excluded from current averages</summary>

<a id="compact-candidate-selection"></a>

| Official SFT with… | N | Tasks | Successes | Success |
| --- | ---: | ---: | ---: | ---: |
| No selector | 1 | 300 | 90 | 30.00% |
| ScalarARM | 5 | 300 | 114 | 38.00% |
| SelectionARM | 5 | 300 | 128 | 42.67% |
| SelectionARM, action-only candidates | 5 | 300 | 105 | 35.00% |
| GPT-5.6 Sol | 5 | 300 | 132 | 44.00% |

Historical actor-policy receipts are intact; these are not missing-policy runs. Dates and harness details differ from the current comparisons. No September GPT-5.5 online-selector result was found. [Historical-gain audit](ARM_INFERENCE.md#arm-same-task-verdict-audit-20261006) · [Sol](arm_results/sol-selection300.json) · [Action-only ablation](arm_results/selection-actions-only.json).

</details>

<details>
<summary>Buggy, superseded or unmatched experiments</summary>

<a id="rerun-triage-20261006"></a>
<a id="actor-selector-experiment-tracker-20261004"></a>
<a id="luna-cpu-family-results-20261005"></a>
<a id="api-actor-high-results-20261005"></a>
<a id="scaling-comparisons"></a>
<a id="qwen-actor-ablation"></a>
<a id="luna-actor-full300-20261004"></a>
<a id="luna-full300-results-20261006"></a>
<a id="sft-selector-judge-leniency-20261006"></a>
<a id="api-actor-reasoning-20261005"></a>
<a id="api-actor-stopping-audit-20261005"></a>
<a id="learned-arm-and-retries"></a>
<a id="luna-qwen-inference-20261004"></a>
<a id="luna-qwen-pilot-partial-20261005"></a>
<a id="luna-pixel-coordinate-repair-20261005"></a>

<!-- actor-selector-results:start -->
| Experiment | Limitation | Replacement / gap |
| --- | --- | --- |
| Oct 4 learned ARM versus oracle episode pass@k | Missing actor browser policy | Piotr corrected above; controlled pass@k rerun missing |
| Qwen3-VL actor; Luna medium/high and Sol high actors | Missing policy; API actors also lost native conversation state/call IDs | Corrected matched actor runs missing |
| Historical SFT + Luna N=5/N=10; Qwen alone/+Luna | Missing actor policy | SFT N=5 replaced by local v2; N=10 and Qwen remain open |
| Historical hosted SFT + Jev/Kev | Missing policy; browser/judge differences | Replaced by local v2 |
| Hosted Kev actor and small pilots | Different system/protocol; pilot Luna also had coordinate corruption | Excluded from controlled comparisons |
<!-- actor-selector-results:end -->

All attempts and verdicts are preserved; these scores do not enter current means. [Original tables and plots](https://github.com/zixianma/OpenWebRL/blob/a0a6db4e677c7f1959a53f0498ffc9a18d9c9ca0/openwebrl/docs/ARM_INFERENCE_SCALING.md#actor-selector-experiment-tracker-20261004) · [Actor pipeline audit](arm_results/reasoning_actors_full300_20261005/pipeline-debug.json) · [Judge audit](arm_results/sft_selector_judge_audit_20261006.json).

</details>
