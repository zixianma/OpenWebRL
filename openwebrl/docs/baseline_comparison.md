# Baseline comparison: WebArbiter, WebShepherd and Piotr SelectionARM

Results as of October 26, 2026. Actor: **gpt-4o-2024-08-06** in every cell (gpt-4o-2024-05-13 is retired). Methods select among actor-proposed actions at each step: *no search* executes the actor's action; *random-of-five* picks uniformly among five candidates; *WebArbiter-7B*, *WebShepherd-8B* and *Piotr SelectionARM* are the released selectors.

**Takeaways.**
- **Online-Mind2Web (284 tasks with all five methods valid):** no selector beats no search. Piotr SelectionARM +2.5 pp and WebArbiter −2.8 pp are within noise; WebShepherd (−14.8 pp) and random-of-five (−12.0 pp) are significantly worse.
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

- **Comparability.** Table 5's no-search and WebShepherd rows are reported from the WebShepherd paper, which used a BrowserGym harness (screenshot + accessibility tree, 20 samples → top-5 by frequency, reward-guided refinement, 21 tasks). Only the WebArbiter row (10/19) was produced with the released WebArbiter code reproduced here. Our candidate generation asks gpt-4o for five diverse actions in one call, as the released explorer does.
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
