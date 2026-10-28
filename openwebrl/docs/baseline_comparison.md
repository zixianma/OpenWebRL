# Baseline comparison: WebArbiter, WebShepherd and Piotr SelectionARM

Results as of October 28, 2026. Actor: **gpt-4o-2024-08-06** in every cell (gpt-4o-2024-05-13 is retired). Methods select among actor-proposed actions at each step: *no search* executes the actor's action; *random-of-five* picks uniformly among five candidates; *WebArbiter-7B*, *WebShepherd-8B* and *Piotr SelectionARM* are the released selectors.

**Takeaways.**
- **Online-Mind2Web (284 tasks with all five methods valid):** no selector beats no search. Piotr SelectionARM +2.5 pp and WebArbiter −2.8 pp are within noise; WebShepherd (−14.8 pp) and random-of-five (−12.0 pp) are significantly worse.
- **WebArena-Lite, four sites (128 tasks, 1280×720, same harness):** gpt-4o + Piotr SelectionARM 15/101 valid (14.9%) vs no search 22/106 (20.8%); paired −4.0 pp (100 tasks, sign-test p = 0.39), negative but not significant under every scoring. [Details](#webarena-lite-4site-piotr-20261028).
- **WebArena-Lite Reddit (18 tasks):** after fixing harness mismatches (October 26 rerun), no search reaches 4/15 valid (26.7%), close to the published 28.6%; WebArbiter 5/16 (31.2%) stays below the published 52.6%; WebShepherd 1/12 (8.3%) is far below 47.6%. At these denominators no between-method difference is reliable.

## Online-Mind2Web, 300 tasks

Judge: gpt-5.6-terra (Online-Mind2Web terminal-success judge; o4-mini is retired). Each cell is one actor episode; every one of the 1,500 cells was attempted.

<a id="om2w-paired-20261025"></a>
### Paired comparison with no search

The paired set is the 284 tasks on which all five methods produced a valid episode. Difference = method minus no-search success rate on those tasks; 95% CI is a paired-proportion normal interval; p is the two-sided sign test on discordant tasks.

| Method | Paired tasks | Successes | Success (%) | Method-only wins | No-search-only wins | Difference (pp) | 95% CI (pp) | Sign-test p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| No search | 284 | 78 | 27.5 | – | – | – | – | – |
| Piotr SelectionARM | 284 | 85 | 29.9 | 37 | 30 | +2.5 | −3.2 to +8.1 | 0.46 |
| WebArbiter-7B | 284 | 70 | 24.6 | 29 | 37 | −2.8 | −8.4 to +2.8 | 0.39 |
| Random-of-five | 284 | 44 | 15.5 | 15 | 49 | −12.0 | −17.3 to −6.6 | 0.00002 |
| WebShepherd-8B | 284 | 36 | 12.7 | 16 | 58 | −14.8 | −20.5 to −9.1 | 0.000001 |

<a id="om2w-per-method-20261025"></a>
### Per-method rates on all 300 tasks

Valid-only rate = successes / valid cells (95% Wilson interval). The last column counts invalid cells as failures (denominator 300).

| Method | Valid cells | Invalid cells | Successes | Valid-only success (%) | 95% CI (%) | Invalid as failure (%) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| No search | 294 | 6 | 79 | 26.9 | 22.1–32.2 | 26.3 |
| Piotr SelectionARM | 288 | 12 | 85 | 29.5 | 24.5–35.0 | 28.3 |
| WebArbiter-7B | 294 | 6 | 71 | 24.1 | 19.6–29.4 | 23.7 |
| Random-of-five | 292 | 8 | 44 | 15.1 | 11.4–19.6 | 14.7 |
| WebShepherd-8B | 292 | 8 | 36 | 12.3 | 9.0–16.6 | 12.0 |

<details><summary>Retries, invalid cells and provenance</summary>

- **Retries.** 96 cells with no valid first attempt received labelled retries: a first round (`rerun300-j{1,2,3}-retry1-vlm`), ended early at 13:52 PDT on October 25 when OpenAI returned HTTP 429 `organization_spend_limit_exceeded`, and a second round (`retry2`) after the limit was lifted; 61 became valid. A retry replaces only an invalid earlier attempt; valid attempts are never rerun. The paired set is 236 tasks without retries and 284 with them. Retried / recovered by method: no search 14 / 8, random-of-five 20 / 12, WebArbiter 12 / 7, Piotr 27 / 19, WebShepherd 23 / 15. Five Piotr/WebArbiter retry2 cells were not reached before the job's time limit.
- **Invalid cells (40).** `Error` cells (5 per method) are browser/navigation failures, mostly theweathernetwork.com tasks whose start page redirect-loops for every method. Piotr's `EvaluationNotReady`/`ValueError` cells come from its released prompt requiring five admissible original candidates; other `ValueError` cells are action-parse failures.
- **Comparability.** Retries ran hours after first attempts on live websites, so a retried cell may see a different site state than the other methods on that task. Earlier cohorts with the gpt-4o-2024-05-13 actor or the o4-mini judge are not pooled.
- **Source.** `/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-om2w-table5-20261012/rerun300-0806-20261023/final_table.py` → `final-table.json` (pools the rerun300 stage, completion and retry packages; asserts one primary attempt per cell).
</details>

<a id="webarena-lite-4site-piotr-20261028"></a>
## WebArena-Lite four sites: gpt-4o with and without Piotr SelectionARM (October 28)

Controlled comparison on the 128 single-site WebArbiter Table 5 tasks (Shopping 45, CMS/shopping_admin 35, Reddit 18 without original task 27, GitLab 30; Map excluded as in WebArbiter Table 5). Both arms run in the same native harness (released WebArbiter coordinator, WebArena accessibility tree and released evaluator), at the same **1280×720** viewport, on the same tasks, with each task's two cells run back to back in the same job. *No search* executes gpt-4o's single action. *+ Piotr SelectionARM* asks gpt-4o for five candidates in one call (the OM2W protocol); Piotr picks one from the actor's own 1280×720 screenshot. One episode per task × method; all 256 cells have a final outcome.

<a id="webarena-lite-4site-piotr-paired-20261028"></a>
### Paired comparison

Difference = ARM minus no-search success on the paired tasks; 95% CI is a paired-proportion normal interval; p is the two-sided sign test on discordant tasks.

| Scoring | Paired tasks | No-search successes | ARM successes | ARM-only wins | No-search-only wins | Difference (pp) | 95% CI (pp) | Sign-test p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Valid cells only (primary) | 100 | 19 | 15 | 4 | 8 | −4.0 | −10.7 to +2.7 | 0.39 |
| + gpt-4.1 re-score of fuzzy-judge cells | 116 | 24 | 19 | 4 | 9 | −4.3 | −10.4 to +1.7 | 0.27 |
| + ARM CMS load timeouts counted as ARM failures | 124 | 27 | 19 | 4 | 12 | −6.5 | −12.7 to −0.2 | 0.077 |

**Takeaway.** On WebArena, adding Piotr SelectionARM to gpt-4o does not help; the point estimate is negative under every scoring, but no difference is significant by the sign test. This contrasts with Online-Mind2Web, where Piotr was +2.5 pp (also not significant).

<a id="webarena-lite-4site-piotr-sites-20261028"></a>
### Per-site rates (valid cells only)

| Site | No-search valid | No-search successes | No-search (%) | ARM valid | ARM successes | ARM (%) | v9 no search at 1024×2490 (successes / valid) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Shopping | 35 | 3 | 8.6 | 35 | 7 | 20.0 | 9 / 35 |
| CMS | 29 | 9 | 31.0 | 23 | 3 | 13.0 | 6 / 29 |
| GitLab | 26 | 7 | 26.9 | 27 | 3 | 11.1 | 6 / 26 |
| Reddit | 16 | 3 | 18.8 | 16 | 2 | 12.5 | 6 / 16 |
| **All** | **106** | **22** | **20.8** (95% CI 14.1–29.4) | **101** | **15** | **14.9** (9.2–23.1) | 27 / 106 (25.5%) |

The ARM gains on Shopping (+4 successes) and loses on CMS (−6, partly the load timeouts below), GitLab (−4) and Reddit (−1). The v9 column is the October 26 Table 5 rerun's no-search arm (same harness, released 1024×2490 viewport); the smaller viewport costs no search mostly on Shopping and Reddit.

**Invalid cells.**

| Reason | No search | + ARM |
| --- | ---: | ---: |
| Retired fuzzy judge (`gpt-4-1106-preview`); final, re-scored with gpt-4.1 | 20 | 18 |
| Repeated page-load timeout (two attempts, CMS only) | 0 | 8 |
| Agent-chosen invalid action or task-81 evaluator config | 2 | 1 |
| **Valid** | **106** | **101** |

gpt-4.1 re-score (sensitivity, not the paper protocol): no search 5 successes of 18 rescorable cells, ARM 4 of 16; 2 cells per method cannot be replayed offline. Strict + re-scored: no search 27/124 (21.8%), ARM 19/117 (16.2%).

<a id="webarena-lite-4site-piotr-diagnosis-20261028"></a>
### Why the ARM does not help here

- **Candidate pools and grounding are fine.** 1,465 of 1,688 ARM steps (86.8%) had five distinct candidates, 16 had one (no ARM call), and 5,608 of 5,610 candidates were grounded to an on-screen element.
- **Piotr overrides gpt-4o's first candidate on 54% of steps** (slot 0 / 1 / 2 / 3 / 4: 771 / 482 / 207 / 139 / 73 of 1,672 calls). Override rates by site: Shopping 54%, CMS 54%, GitLab 49%, Reddit 62%. Overrides are rarer in successful ARM episodes (41% of 112 steps) than in failed ones (53% of 1,254) and invalid ones (60% of 306).
- **No-search-only wins (8 tasks).** In 2 the ARM never overrode the actor, so the difference is actor sampling variance. In the others the first override comes at step 0–2; for GitLab task 49, no search clicked the same element as gpt-4o's first candidate (the commit list), while the ARM chose a search query and failed.
- **CMS load timeouts (8 ARM cells, each failed twice).** The last action before the 30 s page-load timeout was the Magento admin REPORTS menu link in 5, the SALES menu in 1, a Bestsellers tab in 1 and a report date field in 1; 3 of the 8 were gpt-4o's first candidate. No search clicked REPORTS in 8 episodes and the ARM in 5 completed episodes without a timeout, so the click alone does not explain it; the cause (harness load wait on this CMS instance after these navigations) is unresolved. These cells are excluded from the primary row and counted as ARM failures in the sensitivity row.

<a id="webarena-lite-4site-piotr-cost-20261028"></a>
### Cost and latency

| Method | Cells (incl. reruns) | Steps | Seconds per step | Candidate generation (s/step) | Selection (s/step) | Steps per cell | API USD per cell | API USD total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No search | 120 | 1,720 | 17.7 | 2.0 | 0 | 14.3 | 0.56 | 66.90 |
| + Piotr SelectionARM | 125 | 1,760 | 18.5 | 4.4 | 0.16 (local GPU) | 14.1 | 0.52 | 64.61 |

The five-candidate call costs about 2.4 s and 220 more completion tokens per step than no search's single action; Piotr itself adds 0.16 s per step and no API cost. Per-cell API cost includes the released terminal detector and state-difference calls, which both methods make.

<details><summary>Comparability, protocol and accounting</summary>

- **Viewport.** 1280×720 for actor and ARM (the ARM sees the actor's observation PNG unchanged). WebArena's own `browser_env` and the OM2W harness default to 1280×720; the released WebArbiter service uses 1024×2490 (`run_management.py:172`), which v8/v9 followed. A 1024×2490 alternative was built (v10) and not run, because the ARM would have seen only the top 576 px (66% of gpt-4o's element actions fell inside that region).
- **Differences from OM2W.** The ARM chooses among 1–5 deduplicated admissible candidates (no ARM call when one survives); OM2W required exactly five and marked the cell invalid otherwise. Actions that cannot be described for the ARM get a placeholder descriptor. The harness is the native WebArena one, not OM2W's browser viewer.
- **Reruns.** Only infrastructure invalids were rerun; fuzzy-judge, agent-chosen invalid actions and the task-81 evaluator config error are final. A cell that failed twice on infrastructure is final (`repeated_infra_invalid`). Jobs stopped at the harness's 1 TiB per-job scratch cap after about 21 cells, so remaining cells ran in replacement jobs (fresh site copy per cell).
- **Startup bug.** The first submission (v11, jobs 392750–392757) died at the ARM's 8-token startup probe (`finish_reason=length`); v11b uses the selector-shaped probe from the earlier Reddit runs. No cells ran in v11.
- **Budget.** Approved 8 × 1 H200 (29 H200-hours) and USD 280. Used 25.2 H200-hours (including the failed v11 attempt) and USD 148.91 API; gpt-4.1 re-score 36 judge calls.
- **Packages.** `webarena-subset-review-20261021/launch-preparation-v11b-{shop-a,shop-b,shop-c,cms-a,cms-b,reddit-a,gitlab-a,gitlab-b}` and replacements `launch-preparation-v11c-*`; tools `v11-tools/`, `v11b-tools/` (pooling rule `replacement_blocks_v11r2.py`, cost/latency `cost_latency_v11.py` → `cost-latency-v11.json`); auditor `native-audit-v8`; ledger `gpu-budget-v11-20261027/`; re-score `fuzzy-rescore-gpt41-v11/`.
</details>

<a id="webarena-lite-reddit-20261026"></a>
## WebArena-Lite Reddit, 18 single-site tasks

Released WebArbiter WebArena-Lite Reddit tasks with the native released evaluator; task 5 (original 27) is excluded. One episode per task × method; all 54 cells independently audited (54/54 verified).

<a id="webarena-lite-reddit-v8-20261026"></a>
### Rerun with harness fixes (October 26)

| Method | Cells | Valid | Successes | Valid-only success (%) | 95% CI (%) | Published Table 5 (%) | October 25 run (valid-only, %) |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| No search | 18 | 15 | 4 | 26.7 | 10.9–52.0 | 28.6 | 6.7 (1/15) |
| WebArbiter-7B | 18 | 16 | 5 | 31.2 | 14.2–55.6 | 52.6 | 15.4 (2/13) |
| WebShepherd-8B | 18 | 12 | 1 | 8.3 | 1.5–35.4 | 47.6 | 6.7 (1/15) |

Successful tasks: no search 14, 97, 132, 135; WebArbiter 14, 97, 131, 135, 141; WebShepherd 141. With the retired fuzzy judge replaced by gpt-4.1 for tasks 139/140 (sensitivity analysis, 6 cells), no search gains task 139 (5/17, 29.4%), WebArbiter 5/18 (27.8%) and WebShepherd 1/14 (7.1%).

**What changed.** A forensic review of the October 25 run found no site or evaluator bug (official Postmill image and data, fresh state and login per cell, byte-identical released evaluator that scores every gold answer 1.0) but four harness mismatches:

| Item | October 25 run | Rerun | Evidence |
| --- | --- | --- | --- |
| Viewport (the observation tree is viewport-only) | 1280×720; first observation clipped to an 800×600 screen | 1024×2490 at every step | Released service default `run_management.py:172`; paper demo screenshots; submit/forum buttons were missing from 26/26 and 20/20 observations |
| Stopping | Released gpt-4o-mini terminal detector overrides the chosen action | gpt-4o-2024-08-06 detector; for WebArbiter/WebShepherd its stop is one more candidate the selector must choose | Detector ended 33 of 50 failed episodes, ~22 of them one action before completion; paper demo shows stop competing in the tournament |
| WebArbiter decoding | Temperature 1.0, top-p 0.9 | Greedy | Released `reward_guided_agent.json`, README `do_sample=False` |
| Terminal messages without brackets | Episode invalid or answer extracted as empty | Normalised to `send_msg_to_user [text]` | 10 affected cells |

<details><summary>Comparability, invalid cells and provenance</summary>

- **Comparability.** Table 5's no-search and WebShepherd rows are reported from the WebShepherd paper, which used a BrowserGym harness (screenshot + accessibility tree; 20 policy samples at temperature 1.0, the five most frequent actions kept as candidates, highest reward executed; 21 Reddit tasks including multi-site ones). Refinement is a separate WebShepherd experiment, not part of these rows. Only the WebArbiter row (10/19) was produced with the released WebArbiter code reproduced here. Our candidate generation asks gpt-4o for five diverse actions in one call, as the released explorer does; the WebArbiter paper instead says it follows WebShepherd's Best-of-N with N = 5 (Sec. 5.2.1).
- **Invalid (11).** Tasks 139/140 need the retired `gpt-4-1106-preview` fuzzy judge (6; re-scored above). The released browser environment crashes when an action opens a new tab, because only pages created at reset get a CDP session (`'Page' object has no attribute 'client'`, `browser_env/envs.py:156–179`): no search 136, WebShepherd 14, 151, 153 (4). WebShepherd 136 chose an unsupported `press Ctrl+c` action (1).
- **Cost.** 4 jobs × 1 H200, 20,990 GPU-seconds in total; API USD 34.34.
- **Packages.** `webarena-subset-review-20261021/launch-preparation-v8-{j1-ns-shep,j2-arb-a,j3-arb-b,j4-shep}` (jobs 388903, 388905–388907), builder `v8-tools/build_v8.py`, auditor `native-audit-v4`, fuzzy re-score `fuzzy-rescore-gpt41-v8/`.
</details>

<a id="webarena-lite-reddit-20261025"></a>
<details><summary>October 25 run (before the harness fixes)</summary>

| Method | Cells | Valid | Successes | Valid-only success (%) | 95% CI (%) | Invalid as failure (%) | Published Table 5 (%) |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| No search | 18 | 15 | 1 | 6.7 | 1.2–29.8 | 5.6 | 28.6 |
| WebArbiter-7B | 18 | 13 | 2 | 15.4 | 4.3–42.2 | 11.1 | 52.6 |
| WebShepherd-8B | 18 | 15 | 1 | 6.7 | 1.2–29.8 | 5.6 | 47.6 |

Successful tasks: no search 14; WebArbiter 14 and 134; WebShepherd 132. An earlier gpt-4o-2024-05-13 cohort was also far below the paper (15.8 / 5.3 / 10.5 %). Invalid (10): tasks 139/140 × 3 (retired fuzzy judge; gpt-4.1 re-score: all failures), unbracketed `send_msg_to_user` (3), new-tab crash (2). Audit 53/54 verified. Details: [WebPRM handoff, October 25 status](WEBPRM_HANDOFF.md#webprm-status-20261025); results `webarena-subset-review-20261021/results-native54/table.json`.
</details>
