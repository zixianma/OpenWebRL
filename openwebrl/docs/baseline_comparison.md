# Baseline comparison: WebArbiter, WebShepherd and Piotr SelectionARM

Results as of October 25, 2026. Actor: **gpt-4o-2024-08-06** in every cell (gpt-4o-2024-05-13 is retired). Methods select among actor-proposed actions at each step: *no search* executes the actor's action; *random-of-five* picks uniformly among five candidates; *WebArbiter-7B*, *WebShepherd-8B* and *Piotr SelectionARM* are the released selectors.

**Takeaways.**
- **Online-Mind2Web (253 tasks with all five methods valid):** no selector beats no search. Piotr SelectionARM +3.2 pp and WebArbiter −1.2 pp are within noise; WebShepherd (−13.8 pp) and random-of-five (−11.9 pp) are significantly worse.
- **WebArena-Lite Reddit (18 tasks):** all three methods succeed on 1–2 tasks, far below the published WebArbiter Table 5 (28.6 / 52.6 / 47.6 %). Differences between methods are within noise.

## Online-Mind2Web, 300 tasks

Judge: gpt-5.6-terra (Online-Mind2Web terminal-success judge; o4-mini is retired). Each cell is one actor episode; every one of the 1,500 cells was attempted.

<a id="om2w-paired-20261025"></a>
### Paired comparison with no search

The paired set is the 253 tasks on which all five methods produced a valid episode. Difference = method minus no-search success rate on those tasks; 95% CI is a paired-proportion normal interval; p is the two-sided sign test on discordant tasks.

| Method | Paired tasks | Successes | Success (%) | Method-only wins | No-search-only wins | Difference (pp) | 95% CI (pp) | Sign-test p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| No search | 253 | 70 | 27.7 | – | – | – | – | – |
| Piotr SelectionARM | 253 | 78 | 30.8 | 34 | 26 | +3.2 | −2.8 to +9.2 | 0.37 |
| WebArbiter-7B | 253 | 67 | 26.5 | 28 | 31 | −1.2 | −7.1 to +4.8 | 0.79 |
| Random-of-five | 253 | 40 | 15.8 | 12 | 42 | −11.9 | −17.4 to −6.4 | 0.00005 |
| WebShepherd-8B | 253 | 35 | 13.8 | 15 | 50 | −13.8 | −19.8 to −7.8 | 0.00002 |

<a id="om2w-per-method-20261025"></a>
### Per-method rates on all 300 tasks

Valid-only rate = successes / valid cells (95% Wilson interval). The last column counts invalid cells as failures (denominator 300).

| Method | Valid cells | Invalid cells | Successes | Valid-only success (%) | 95% CI (%) | Invalid as failure (%) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| No search | 287 | 13 | 77 | 26.8 | 22.0–32.2 | 25.7 |
| Piotr SelectionARM | 273 | 27 | 81 | 29.7 | 24.6–35.3 | 27.0 |
| WebArbiter-7B | 287 | 13 | 71 | 24.7 | 20.1–30.0 | 23.7 |
| Random-of-five | 283 | 17 | 43 | 15.2 | 11.5–19.8 | 14.3 |
| WebShepherd-8B | 288 | 12 | 36 | 12.5 | 9.2–16.8 | 12.0 |

<details><summary>Retries, invalid cells and provenance</summary>

- **Retries.** 94 cells with no valid first attempt received one labelled second attempt (`rerun300-j{1,2,3}-retry1-vlm`); 19 became valid. A retry replaces only an invalid first attempt; valid first attempts are never rerun. The paired set is 236 tasks without retries and 253 with them. Retried / recovered by method: no search 14 / 1, random-of-five 20 / 3, WebArbiter 12 / 0, Piotr 25 / 4, WebShepherd 23 / 11.
- **Invalid cells (82).** All 41 `BrokerError` cells are retry attempts rejected with OpenAI HTTP 429 `organization_spend_limit_exceeded` rejections that began at 13:52 PDT on October 25 and ended all three retry jobs; these cells could be retried after the organization spend limit is raised. `Error` cells are browser/navigation failures, including four theweathernetwork.com tasks whose start page redirect-loops for every method. Piotr's `EvaluationNotReady`/`ValueError` cells come from its released prompt requiring five admissible original candidates.
- **Comparability.** Retries ran hours after first attempts on live websites, so a retried cell may see a different site state than the other methods on that task. Earlier cohorts with the gpt-4o-2024-05-13 actor or the o4-mini judge are not pooled.
- **Source.** `/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-om2w-table5-20261012/rerun300-0806-20261023/final_table.py` → `final-table.json` (pools the rerun300 stage, completion and retry packages; asserts one primary attempt per cell).
</details>

<a id="webarena-lite-reddit-20261025"></a>
## WebArena-Lite Reddit, 18 single-site tasks

Released WebArbiter WebArena-Lite Reddit tasks with the native released evaluator; task 5 (original 27) is excluded. One episode per task × method.

| Method | Cells | Valid | Successes | Valid-only success (%) | 95% CI (%) | Invalid as failure (%) | Published Table 5 (%) |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| No search | 18 | 15 | 1 | 6.7 | 1.2–29.8 | 5.6 | 28.6 |
| WebArbiter-7B | 18 | 13 | 2 | 15.4 | 4.3–42.2 | 11.1 | 52.6 |
| WebShepherd-8B | 18 | 15 | 1 | 6.7 | 1.2–29.8 | 5.6 | 47.6 |

Successful tasks: no search 14; WebArbiter 14 and 134; WebShepherd 132. The published denominators appear to be 19–21 tasks, so this is a disclosed reconstruction, not an exact reproduction. An earlier gpt-4o-2024-05-13 cohort was also far below the paper (15.8 / 5.3 / 10.5 %), so the gap is not explained by the actor version.

<details><summary>Invalid cells, re-scoring and audit</summary>

- **Invalid (10).** Tasks 139/140 × 3 methods need the retired `gpt-4-1106-preview` fuzzy judge (6). Replaying each saved evaluator transcript with gpt-4.1-2025-04-14 inside the released `llm_fuzzy_match` scores all six as failures (no post was published and `url_match` fails). The model emitted `send_msg_to_user` without call syntax (3). The released browser environment crashed when an action opened a new tab without a CDP client (2).
- **Audit.** 53/54 cells independently verified; no-search task 132 (valid, failed) fails only exact replay because the released coordinator runs post-action processing in a background thread.
- **Details.** [WebPRM handoff, October 25 status](WEBPRM_HANDOFF.md#webprm-status-20261025); results file `/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/results-native54/table.json`.
</details>
