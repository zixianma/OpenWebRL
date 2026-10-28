# WebPRM Table 5 handoff

<a id="webprm-baselines-handover-20261028"></a>
## Collaborator handover — October 28, 2026

**Goal.** Reproduce WebArbiter's Table 5 (gpt-4o actor; no search vs WebShepherd-8B vs WebArbiter-7B on single-site WebArena-Lite tasks) and add our Piotr SelectionARM as an extra selector row. The same selector comparison was also run on live Online-Mind2Web (OM2W). This section supersedes the dated status sections below, which are kept as history.

**Code:** [`webprm_baselines/`](https://github.com/zixianma/OpenWebRL/tree/arm/webprm_baselines) on the `arm` branch (code map and run workflow in its [README](https://github.com/zixianma/OpenWebRL/blob/arm/webprm_baselines/README.md)), plus the `webprm_*` modules in `openwebrl/`, `scripts/` and `tests/`. **Runtime root:** `/gpfs/scrubbed/zixianma/openwebrl-runtime/` (`RT`); WebArena packages, ledgers and results under `D = RT/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/`; OM2W under `RT/webprm-om2w-table5-20261012/`.

### Results at a glance

All rows use actor gpt-4o-2024-08-06 unless stated; valid-only rates; one episode per task × method.

| Benchmark (tasks) | Method | Valid cells | Successes | Valid-only success (%) | Published (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| OM2W (300; paired set 284) | No search | 294 | 79 | 26.9 | – |
| | Piotr SelectionARM | 288 | 85 | 29.5 | – |
| | WebArbiter-7B | 294 | 71 | 24.1 | – |
| | WebShepherd-8B | 292 | 36 | 12.3 | – |
| WebArena-Lite 4 sites, 1024×2490 (v9; 128) | No search | 106 | 27 | 25.5 | 35.0 |
| | WebArbiter-7B | 104 | 31 | 29.8 | 49.2 |
| | WebShepherd-8B | 87 | 23 | 26.4 | 42.0 |
| WebArena-Lite 4 sites, 1280×720 (v11; 128) | No search | 106 | 22 | 20.8 | – |
| | Piotr SelectionARM | 101 | 15 | 14.9 | – |
| WebArena-Lite 4 sites, 1280×1000, OpenWebRL-4B-SFT actor (v12, option C) | No search / + Piotr | – | – | – | smoke test failed at startup; see below |

**Takeaways.**
- **No selector reliably beats no search with a gpt-4o actor.** OM2W paired (284 tasks): Piotr +2.5 pp (−3.2 to +8.1, p 0.46), WebArbiter −2.8 pp, WebShepherd −14.8 pp (significant). WebArena v11 paired (100 tasks): Piotr −4.0 pp (−10.7 to +2.7, p 0.39); −6.5 pp (p 0.077) when the ARM's repeated CMS load timeouts count as failures. Details: [baseline comparison](baseline_comparison.md#webarena-lite-4site-piotr-20261028).
- **Table 5 is not reproduced in absolute terms.** v9 is 9.5–19.4 pp below every published gpt-4o row, and the WebArbiter − no search gap is +4.3 pp vs the published +14.2. Reddit alone (v8, after harness fixes) matched the published no-search rate (26.7% vs 28.6%). Published no-search and WebShepherd rows come from the WebShepherd paper's BrowserGym harness, so only the WebArbiter row is a like-for-like reproduction target.
- **Why Piotr does not help on WebArena (v11):** it overrides gpt-4o's first candidate on 54% of steps, and overrides are more frequent in failed (53%) and invalid (60%) episodes than successful ones (41%). The ARM was trained on OpenWebRL-4B-SFT candidates in OpenWebRL's coordinate action space at 1280×1000, not on gpt-4o accessibility-tree actions — the motivation for option C.

<details><summary>WebArena v9 per site, gpt-4.1 re-score and cost (four-site Table 5 rerun, October 26–27)</summary>

Protocol: released WebArbiter coordinator, WebArena accessibility tree, released evaluator, 1024×2490 viewport (released service default), paper protocol of 20 samples at T 1.0 → five most frequent as candidates for WebArbiter/WebShepherd. Shopping 45, CMS 35, Reddit 18 (original task 27 dropped: its only judge, `gpt-4-1106-preview`, is retired), GitLab 30; Map excluded as in Table 5.

| Site | No search valid | No search successes | No search (%) | WebArbiter valid | WebArbiter successes | WebArbiter (%) | WebShepherd valid | WebShepherd successes | WebShepherd (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Shopping | 35 | 9 | 25.7 | 33 | 10 | 30.3 | 30 | 7 | 23.3 |
| CMS | 29 | 6 | 20.7 | 28 | 7 | 25.0 | 23 | 5 | 21.7 |
| GitLab | 26 | 6 | 23.1 | 27 | 7 | 25.9 | 21 | 7 | 33.3 |
| Reddit | 16 | 6 | 37.5 | 16 | 7 | 43.8 | 13 | 4 | 30.8 |
| **All** | **106** | **27** | **25.5** | **104** | **31** | **29.8** | **87** | **23** | **26.4** |

- **Invalid cells.** Retired fuzzy judge: 20 / 19 / 19 (no search / WebArbiter / WebShepherd); other (agent-chosen invalid actions such as `none`, new-tab crash, one broker failure): 1 / 5 / 18. One GitLab no-search cell never produced an outcome (127 cells).
- **gpt-4.1 re-score of fuzzy-judge cells** (sensitivity, not the paper protocol): no search 8 of 18 rescorable, WebArbiter 6 of 17, WebShepherd 7 of 17 → strict + re-scored 35/124 (28.2%), 37/121 (30.6%), 30/104 (28.8%).
- **Cost and latency** (median seconds per step; mean API USD per cell): no search 13.0 s, generation 1.8 s, USD 0.66; WebArbiter 29.6 s, generation 7.5 s, selection 6.5 s, USD 0.97; WebShepherd 36.1 s, generation 7.6 s, selection 9.7 s, USD 1.10. Total API USD 369.94; 38.6 H200-hours (ledger `D/gpu-budget-v9-20261026/`).
- **Sources.** `D/v9b-tools/summarize_v9.py` (pooling), `D/fuzzy-rescore-gpt41-v9/rescore-results.json`, `D/v9b-tools/cost-latency-v9.json`; packages `D/launch-preparation-v9{b,c,d,e,f}-*`; decision `D/v9b-tools/v9-decision-20261026.json`.
</details>

<a id="webprm-option-c-v12-20261028"></a>
### Option C (v12): OpenWebRL-4B-SFT ± Piotr SelectionARM — in progress

**Question.** Does Piotr help when both the actor and the ARM see their training distribution? The design reuses the crossbench full-set selector protocol (the one where Piotr gave +5.9 to +7.0 pp over OpenWebRL-4B-SFT on OM2W/WebVoyager/DeepShop) on the same 128 WebArena tasks and released evaluator.

- **Arms.** No search: OpenWebRL-4B-SFT (revision 15e777db), 1 sample per step. + Piotr: 5 samples per step, Piotr SelectionARM picks one. T 1.0 / top-p 0.9, 1,024 new tokens, full history, one screenshot in context, 30 steps; both arms of a task back to back in one job; actor and ARM sglang servers on one H200.
- **Viewport 1280×1000.** OpenWebRL's default (`openwebrl/env/config.yaml`, since the initial release) and the size of all 3,085 of Piotr's ARM training states. v12 is therefore not directly comparable with the v11 gpt-4o rows at 1280×720.
- **Answers.** `done(response)` becomes WebArena's final answer; step-limit, format and context endings get "N/A" (as in v11). Fuzzy-judge tasks remain invalid (gpt-4.1 re-score offline later).
- **CPU validation passed:** all 65 valid saved v11 string-match cells re-score identically through the v12 path; an end-to-end dry run with stand-in model servers (real browser, agent loop, selector, relays, evaluator) checked 1280×1000 screenshots at every `/generate` and `/select` call and a click that lands only at the 1000-px height.
- **Status.** The GPU smoke test (jobs 395121 Reddit tasks 14/97/132 and 395122 CMS tasks 0/40/101, both methods; 1 H200, 8 CPUs, 240 GiB each; caps 3.5 H200-hours and USD 16) **failed at startup after about 5 minutes: the launcher's actor `/server_info` check timed out.** A fix and resubmission are pending under the same approval; no cell ran.
- **Pending full-run request (not yet approved):** 8 jobs on the v11 site split (Shopping 3 × 30 cells × 4 h, CMS 2 × 35 × 3.5 h, Reddit 1 × 36 × 3 h, GitLab 2 × 30 × 4 h), each 1 H200 / 8 CPUs / 240 GiB, account vlm; cap 32 H200-hours (29 planned + 3 for replacements) and USD 64 API (expected under USD 1; the actor and ARM are local).
- **Open risks.** Final answers for string-match tasks depend on the actor calling `done` with the exact answer; Playwright 1.58 / bundled Chromium (crossbench) instead of v11's staged Chromium for site logins; GPU memory split (selector .30 + actor .45) and the namespace relays are unproven on hardware; heavy CMS report pages may hit the 30 s evaluation timeout.
- **Files.** Tools `D/v12-tools/` (committed as `webprm_baselines/webarena/tools/v12/`), packages `D/launch-preparation-v12-{reddit,cms}-smoke`, ledger and approval `D/gpu-budget-v12-20261028/`, superseded 1280×720 build `D/v12-superseded-720-20261028/`.

<details><summary>Harness pitfalls (read before launching anything)</summary>

- **Retired OpenAI models.** `gpt-4-1106-preview` (WebArena fuzzy judge) returns 404: fuzzy-judge cells are final invalids, re-scored with gpt-4.1 only as sensitivity. `gpt-4o-2024-05-13` and o4-mini were shut down on October 23: actor is gpt-4o-2024-08-06, OM2W judge is gpt-5.6-terra; older cohorts are not pooled.
- **Scratch cap.** Each job has a cumulative 1 TiB site-state cap (about 50 GB per cell), so a job admits about 21 cells unless the auditor verifies and disposes cells. Plan ≤ 20 cells per job or rely on audited disposal.
- **`min_cell_budget_usd` = 6.** A job whose API cap is below USD 6 starts no cells (v11c gitlab-b-r1 ran nothing at USD 5).
- **`min_start_seconds` = 1800.** No cell starts in a job's last 30 minutes; budget time accordingly.
- **Consecutive-invalid halt.** A job stops after 4 consecutive invalid cells (v11c cms-a-r1 on deterministic CMS page-load timeouts). Cells with two infrastructure-invalid attempts are final (`replacement_blocks_v11r2.py`).
- **ARM preflight.** An 8-token "Reply OK." probe ends with `finish_reason=length` and the worker rejects it; use the selector-shaped preflight (`webprm_piotr_preflight_v11b.json` / v12 `/select` probe). This killed all v11 jobs (392750–392757).
- **CMS page-load timeouts.** Eight ARM cells on Magento admin REPORTS/SALES navigation timed out twice; no search clicked the same menu without timeouts. Cause unresolved.
- **Frozen tooling.** Packages pin tool hashes; editing a referenced builder killed live jobs (v7). Copy tools to a new version directory instead.
- **Slurm.** Never change a job's nice value (it drops the submit priority boost); use `--account=vlm` until 2026-11-16; GPU jobs need explicit user approval of exact resources.
- **Spend accounting.** Actual API spend is the sum of `actual` on `settle` events in `campaign/shared-budget.json.events.jsonl`, not a `charged` field.
</details>

<details><summary>Where everything lives</summary>

| Item | Location |
| --- | --- |
| Code (this handover) | `arm` branch: `webprm_baselines/`, `openwebrl/webprm_*.py`, `scripts/*webprm*`, `tests/test_*webprm*` |
| Results write-up | [baseline_comparison.md](baseline_comparison.md) (OM2W, v11, v8 Reddit); [WebPRM baselines history](WEBPRM_BASELINES.md) |
| OM2W final table | `RT/webprm-om2w-table5-20261012/rerun300-0806-20261023/final-table.json` (`final_table.py`); judge agreement `…/judge-agreement-20261023/` |
| WebArena v9 | packages `D/launch-preparation-v9{b..f}-*`, tools `D/v9b-tools/`, ledger `D/gpu-budget-v9-20261026/`, approval `D/user-approval-v9-20261026.json`, re-score `D/fuzzy-rescore-gpt41-v9/` |
| WebArena v11 | packages `D/launch-preparation-v11b-*`, `D/launch-preparation-v11c-*`, tools `D/v11-tools/`, `D/v11b-tools/`, ledger `D/gpu-budget-v11-20261027/`, cost `D/v11b-tools/cost-latency-v11.json`, re-score `D/fuzzy-rescore-gpt41-v11/` |
| WebArena v12 | see Option C above |
| Auditors | `D/native-audit-v8/` (v9/v11), `D/v12-tools/native-audit-v12/` |

**What a collaborator needs beyond git:** access to the Tillicum `vlm` account and the runtime tree above (hosted WebArena site images and per-cell state, the released WebArbiter/WebArena evaluator sources pinned in each package, Piotr SelectionARM and OpenWebRL-4B-SFT weights at the paths recorded in each package's launcher configuration; Piotr's is under `/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/`), and their own OpenAI key in the repo `.env` (never committed).
</details>

<details><summary>Open questions and next steps</summary>

1. Fix the v12 launcher's actor `/server_info` startup check, rerun the 12-cell smoke test, then request the full v12 run.
2. If Piotr helps under v12, the WebArena result says the ARM is actor/action-space specific; if not, WebArena admin/developer sites are out of its training distribution.
3. Offline: a confidence threshold for overriding the actor's first candidate, tested on the saved v11 ARM decisions.
4. Unresolved: the gap between v9 and the published Table 5 (largest on WebArbiter), and the CMS page-load timeouts.
</details>

<a id="webprm-status-20261025"></a>
## Status — October 25, 2026 (Pacific)

**WebArena-Lite Reddit native 18×3 is complete and far below the published Table 5; OM2W rerun300 is finishing completion and invalid-retry jobs.** This section supersedes the October 24 supervision section below. Results file: `webarena-subset-review-20261021/results-native54/table.json` (paths below are under `/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/`).

### WebArena-Lite Reddit, GPT-4o 2024-08-06 actor, 18 released single-site tasks

| Method | Cells | Valid | Successes | Valid-only success (%) | Invalid as failure (%) | Published Table 5 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No search | 18 | 15 | 1 | 6.7 | 5.6 | 28.6 |
| WebArbiter-7B | 18 | 13 | 2 | 15.4 | 11.1 | 52.6 |
| WebShepherd-8B | 18 | 15 | 1 | 6.7 | 5.6 | 47.6 |

Successes: no search task 14; WebArbiter tasks 14 and 134; WebShepherd task 132. Differences between methods are within noise at these denominators. The earlier gpt-4o-2024-05-13 cohort was also far below the paper (15.8 / 5.3 / 10.5 %), so the gap is not caused by the actor substitution. Task 5 (original 27) is excluded; the paper denominators appear to be 19–21 tasks.

<details><summary>Invalid cells, audits and packages</summary>

- Invalid (10): tasks 139/140 × 3 methods call the retired `gpt-4-1106-preview` fuzzy judge (6). An offline replay of each saved evaluator transcript with gpt-4.1-2025-04-14 substituted inside the released `llm_fuzzy_match` scores all six as failures: no post was published and `url_match` fails (`fuzzy-rescore-gpt41/`). gpt-4.1 called an empty answer "correct" in 3 of 6 calls, so it is not a reliable substitute in general; the outcome here does not depend on it. Model emitted `send_msg_to_user …` without call syntax (3). Released `ScriptBrowserEnv` crashed when an action opened a new tab without a CDP client, `envs.py:156–179` (2).
- Independent audit: 53/54 verified; no-search task 132 (valid, failed) fails only exact replay because the released coordinator runs post-action state diff/terminal detection in a background thread (`step_management.py:647`).
- Packages: no search = `v7e` (tasks 14, 97, 98, 131, 132) + `v7k-ns-b`; WebArbiter = `v7k-arb-a/b`; WebShepherd = `v7j-shep-a/b`, all under `webarena-subset-review-20261021/launch-preparation-*`. Fixes on the way: browser v2 (geometry/nav), JSON mode + fence strip, auditor v2/v3 (54-cell cohort; v3 adds `start_url` to the Shepherd context), Reddit storage scan 300 s/1800 s and 1 TiB scratch, block packages ($50 cap bound in the worker), Shepherd 2048-token health preflight, phase-write lock for concurrent tournament comparisons. Builders `v7-tools/build_v7.py` / `build_v7k.py` are hash-frozen by their packages: copy, never edit.
</details>

### OM2W rerun300 (gpt-4o-2024-08-06 actor, gpt-5.6-terra judge)

Interim (before completions/retries): no search 28.3 %, Piotr 31.3 %, WebArbiter 25.7 %, random-of-five 15.2 %, WebShepherd 12.8 % of valid cells; on 223 tasks with all five valid, random5 (p<0.001) and Shepherd (p=0.0001) are below no search, Arbiter and Piotr are not distinguishable. Remaining cells run as completion packages (`rerun300-j*-completion{2,3}-vlm`, J1/J2 under the user cap raise) and one labelled retry of every invalid cell (`rerun300-j*-retry1-vlm`, user approval `rerun300-0806-20261023/user-approval-retry-20261025.json`). Final table: `rerun300-0806-20261023/final_table.py`. The launcher's 15 s `scontrol` startup timeout has failed three single attempts; packages now allow two attempts.


**Table 5 is incomplete.** As of 2026-10-12 (late), Reddit successor **357126 is queued**, and **CMS hosting/reset validation passed** (job 357831). Supervision was an interactive Claude Code session monitor, not an unattended supervisor. This handoff records the state as of 2026-10-12. Read the current execution pointer before acting; this document is not new resource authorization.

The goal is to reconstruct **WebArbiter Table 5 and add Piotr’s released SelectionARM**, then evaluate Online-Mind2Web separately. The fixed comparison contains **155 single-site tasks × 2 actors × 5 selectors = 1,550 episodes**. Exact historical Table 5 membership/settings remain unresolved, so describe this as a **disclosed reconstruction**, not an exact reproduction.

Start with [results and methodology](WEBPRM_BASELINES.md#table5-plan-20261010), the repository [working agreement](../../AGENTS.md), and these runtime records:

```text
R = /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011
N = R/next-stage-preparation
C = N/reddit-table5-190-v3/campaign

R/current_table5_preparation.json
N/reddit-terminal-independent-root-356127-v1/final-review.json
```

`R`, `N`, and `C` above are path abbreviations used throughout this document.

## Verified results

Only the **190-cell Reddit subset** has execution approval. **26 cells have recorded outcomes; none were resampled.** The first 12 (jobs 356022–356127) are independently audited. The 14 from job 356527 have per-cell worker audits but **no independent audit yet**.

| Task | Cells recorded | Valid | Successes (after scoring guard) | Invalid | No verdict (interrupted) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 | 10 | 6 | 0 | 2 | 2 |
| 14 | 10 | 10 | 8 | 0 | 0 |
| 97 | 6 | 5 | 1 | 1 | 0 |
| Total | 26 | 21 | 9 | 3 | 2 |

Task 5 shows zero after the explicit `N/A` scoring guard (five raw positives on task 5 incorrectly accepted `N/A`; original verdicts are preserved). Task 14 succeeded for 8 of its 10 actor × selector cells (failures: GPT-4o + Arbiter-7B, GPT-4o + Shepherd-8B). The task 97 invalid cell is **GPT-4o + Piotr SelectionARM**: a terminal-detector fallback click executed without a Piotr call and had no grounded history display, so the next Piotr prompt could not be built (`EvaluationNotReady`). This is fixed in v9 (below); the invalid cell stays invalid and is not resampled.

**164 Reddit cells remain unclaimed.** The other **1,360 cells** still need hosting-validated environments and execution budgets. Do not publish selector rankings or success percentages from this partial, mostly unaudited coverage.

## Current jobs and immediate next actions

| Job | What | Resources | Status |
| --- | --- | --- | --- |
| 357126 | Reddit successor `N/reddit-table5-190-v9` (5th and last allocation slot) | gpu-h200, 1 H200, 8 CPUs, 240 GiB, 1,344 min | Queued (priority) |
| 357831 | CMS hosting validation `N/cms-hosting-validation-v9` (7th CMS slot) | 0 GPU, 4 CPUs, 16 GiB | **Passed** in 231 s (see CMS section) |

1. Supervise 357126 from start through closure; independently audit its new cells and the 14 cells from 356527.
2. CMS hosting is validated. Next: a CMS Table 5 execution package (its cells still need a GPU budget approved by the user).
3. Prepare hosting/reset validation packages for Shopping, GitLab and Map by generalizing the CMS validator (downloads are complete and independently verified; see below).
4. The WebPRM code is **not tracked in git on any branch**; it exists in the main checkout and in frozen packages under `N/`. Committing it publishes the whole WebPRM code base to the public remote; ask the user first.

**User approvals on 2026-10-12:** "yes I approve" covered the v8 4th Reddit slot; a later "yes I approve" covered (a) per-node live scratch accounting and (b) the 5th Reddit slot, plus the 3rd CMS attempt; "yes" covered the 4th CMS attempt and, earlier, the four CPU-only bundle-v4 environment jobs. Later, "sure do it" covered the 5th CMS attempt, and "yes" covered the 6th, with a standing rule: **jobs requesting 0 GPUs need no approval**. GPU jobs still need explicit approval for each submission. Leftover time or API budget never authorizes another GPU slot.

<details>
<summary>2026-10-12 Reddit attempt 356527 and the v9 successor</summary>

- **v8** (`N/reddit-table5-190-v8`) = v7 + the reviewed admission-drain queue repair + an exactly approved 4th-slot exception. Job **356527** ran 2,766 s and recorded 14 new cells. The queue repair worked live (admission waited for active workers).
- It stopped early because the task 97 invalid cell's state (48 GB) was preserved, as policy requires for an incomplete audit. That pushed **cumulative** retained scratch across all attempts (480,954,886,176 bytes) to where one more cell would exceed the 549,755,813,888-byte cap, so admission closed with no active worker.
- **v9** (`N/reddit-table5-190-v9`, builder `N/reddit-table5-190-v9-build/build_v9.py`) adds:
  - Piotr history grounding for actions executed without a Piotr selection (`PiotrSelectionARMSelector.historical_action`, called from `run_loop`). A frozen-suite test (`test_webprm_terminal_fallback`) caught that the fallback path needs a real adapter; fixed before freezing.
  - `scratch_accounting_exception` with rule `per_node_live`: the cap counts this node's live usage plus prior attempts' retained state **still present on this node** (node-local `/tmp`); retained states on other nodes are preserved but not charged. The cap value and free-disk check are unchanged. It is exception-gated (approval digest + user text) and outside the allocation contract.
  - `allocation_exception.additional_attempts=2` (5 slots total).
- 311 frozen tests pass; `validate_proposal` passes; allocation contract identical to v8. Approval receipt: `N/reddit-table5-190-v9/approved-request.json` (previous closed epoch `bcc1c20f…`).

</details>

<details>
<summary>Failures, repairs and validation evidence</summary>

| Job | Scheduler seconds used | Outcome |
| --- | ---: | --- |
| 356022 | 2,614 | Stopped after detecting native terminal-detector wrapper bugs; 10 immutable primary claims preserved |
| 356094 | 47 | Startup failed when a storage scan encountered a disappearing database-build file; no new model calls or primary claims |
| 356127 | 305 | Completed two successful trials, then prematurely stopped admitting work |

Repairs are implemented and tested:

- Native terminal handling follows the pinned parser’s fallback/nonterminal-action behavior: **78 focused tests**.
- Storage scanning retries narrowly identified file-disappearance races, accepts only complete scans, and verifies final measurements after closure: **35 focused tests**.
- The latest queue fix waits for an active worker to close before rechecking a denied resource admission. It also records admission reasons and checks minimum startup capacity: **52 tests plus 7 independent frozen-source checks**.
- Additive interruption audits and collector support passed **58 tests**, with independent byte-identical replays. No primary evidence or verdict was rewritten, and no model call or state disposal was performed by audit activation.

**The queue fix has not been validated live.** Frozen v7 still contains the premature-exit behavior. The precise resource check that rejected admission in 356127 was not logged; storage pressure is plausible, not proven.

Reviewed repair and closure evidence:

```text
N/reddit-admission-drain-repair-v1/readiness.json
N/reddit-admission-drain-repair-v1/root-review.json
N/reddit-stop-audit-356127-v1/audit.json
N/reddit-terminal-independent-root-356127-v1/two-primary-audits.json
N/reddit-terminal-independent-root-356127-v1/closure-audit-review.json
N/reddit-interruption-audit-v3/readiness.json
N/reddit-terminal-independent-root-356127-v1/interruption-activation.json
```

Queue-repair readiness SHA256: `ac61509ed9d52c02fccb74e9f794325b2efa2366bbf8453909baa83eb2a8df91`.

Independent queue-repair review SHA256: `b93620beace6db15c63187be32922dce1f55181b63d7b339f27f25f4b00b26ca`.

Latest terminal/accounting audit SHA256: `1dff895e4e319d6ad17851d3958bbc859ba89d865aa852c5ecd2401faa816587`.

The queue/controller implementation is in `openwebrl/webprm_campaign_queue.py` and `scripts/launch_webprm_table5_campaign.py`. Reporting uses `scripts/collect_webprm_table5_campaign.py`; interruption auditing uses `scripts/audit_webprm_interrupted_cell.py`. Prefer reviewed frozen source packages over assuming every working-tree change belongs to this task.

</details>

<details>
<summary>Reddit resource authority and cumulative accounting</summary>

| Resource | Cap | Consumed (after 356527) |
| --- | ---: | ---: |
| Allocation attempts | 3 + 2 approved exceptions = 5 | **5** (356022, 356094, 356127, 356527, 357126 queued) |
| Cumulative allocation time | 86,400 s | 5,732 s before 357126 |
| API spending | USD 250 | USD 7.800215 |
| Paid dispatches | 21,524 | 1,037 |
| Local dispatches | 11,471 | 833 |
| Total dispatches | 32,995 | 1,870 |
| Retained apparent scratch (all attempts) | 549,755,813,888 bytes (per node live under v9) | 480,954,886,176 bytes across g017/g009/g001 |
| Durable campaign artifacts | 68,719,476,736 bytes | 304,104,480 bytes |

The actual allocation profile was **1 H200, 8 CPUs, 240 GiB RAM**, with at most two workers. The original request allowed 16 CPUs; the approved operational reduction used 8 because of scheduler constraints. 357126 is the last approved slot; another slot requires another explicit exception.

All failed attempts, startup controls and the historical unfinished local dispatch remain charged. The financial epoch is closed, with no unresolved paid reservations or overruns. Preserve the shared campaign, journal, queue claims, source versions and all original verdicts. Do not reset a budget on retry or resample a claimed primary cell.

Original authority and financial evidence:

```text
N/table5-next-jobs-bundle-v2.json
N/table5-next-jobs-bundle-v2-approval.json
C/allocation-history/
C/shared-budget.json
C/shared-budget.json.events.jsonl
C/queue/
C/audit-replacements/
```

Original approved bundle SHA256: `e53d165ad01a9a69c777c9de9f28d4e9f76d0514df203aa8708fdcdd3004ebde`.

The financial epoch recovery for 356022 has already been performed. Do not repeat it. Its preserved receipt is `N/reddit-operational-repair-v1/epoch-recovery.json`.

</details>

<details>
<summary>Other sites: downloads and CMS hosting validation (2026-10-12)</summary>

The four CPU-only bundle-v4 jobs were approved ("yes") and run. Downloads are complete and **independently re-verified by separate jobs**; none is hosting-validated:

| Site | Job | Result | Charged bytes / cap | Independent check |
| --- | --- | --- | ---: | --- |
| Shopping (third-party image) | 356538 | Completed, 1,771 s | 46,675,568,065 / 70,000,000,000 | 356576: 28/28 blobs re-hashed |
| GitLab (third-party image) | 356548 | Completed, 1,906 s | 49,673,134,791 / 80,000,000,000 | 356581: 12/12 blobs re-hashed |
| Map (original WebArena data) | 356547 | Completed, 5,146 s (append-only resume) | 190,513,780,871 / 195,000,000,000 | 356631: 35/35 SHA256 plus 4 published S3 multipart ETags |

Shopping and GitLab images' equivalence to the original WebArena state is unverified. No hosting/reset validation packages exist yet for Shopping, GitLab or Map.

**CMS hosting validation** (campaign root `N/cms-hosting-validation-v3/campaign`, shared by every version; cap 7,200 s cumulative, 4 CPUs/16 GiB/0 GPU):

| Job | Package | Seconds | Failure | Fix |
| --- | --- | ---: | --- | --- |
| 356545 | v3 | 155 | Inherited 32 MiB `RLIMIT_FSIZE` (log cap) killed `cp` on a 255 MB file | v4: logs capped by a pipe drain instead |
| 356561 | v4 | 156 | Host-user `cp` could not read mode-0 `run/crond.reboot` | v5: copy runs as namespace-mapped root |
| 357113 | v5 | 4 | `history()` still hard-coded `len(rows)>2` after the cap was raised to 3; the 3rd job failed its own startup check | v6: bound by `RESOURCES['max_allocations']`; a terminal attempt that never created its folder is charged time only |
| 357128 | v6 | 129 | `apptainer --writable` cannot create bind destination `/opt/webprm-cms` | v7: create empty `/opt/native-fakeroot` and `/opt/webprm-cms` in the private state copy only; symlinks rejected |
| 357728 | v7 | 198 | Container, MariaDB, services, origin rewrite and admin login all worked; the check then visited `/admin/dashboard`, a Magento 404 | v8: see the probes below |
| 357822 | v8 | 0 | Cancelled while pending: cumulative cross-node retained scratch (35.08 GB) would have failed storage admission | v9: charge prior retained scratch only while it exists on this node (the Reddit `per_node_live` rule); the free-disk check is unchanged |
| **357831** | **v9** | **231** | **Passed** | — |

CPU-only probes on disposable state (not validation slots) found the v8 fixes:
- **357743:** found the canonical dashboard route `/admin/admin/dashboard/`.
- **357750:** found that the product name field is rendered by JavaScript after `domcontentloaded`, so the check now waits for it.
- **357766:** applying the hash-verified image config `Env` made `elasticsearch` resolvable again (`--cleanenv` had dropped `PATH`), and the UI save then persisted. Save failures now report Magento's message verbatim.

**Result of 357831** (`C_CMS/attempt-357831/validation-result.json`, with `C_CMS = N/cms-hosting-validation-v3/campaign`):

| Check | Count / value |
| --- | --- |
| Released shopping_admin tasks / distinct start URLs loaded (HTTP 200) | 35 / 2 |
| Native evaluator on baseline: positive control / negative control | 1 / 0 |
| UI product rename persisted in the database | yes |
| Fresh second copy restored the product (`Joust Duffle Bag`) and the evaluator controls | yes |
| Owned processes remaining at closure; model calls; API spend | 0; 0; $0 |
| CMS time used / cap | 873 s / 7,200 s |

This validates the **third-party** image's hosting and reset; its equivalence to the original WebArena CMS state is **not** verified (`original_state_equivalence_verified=false`). Retained scratch on g005: attempts 357728 and 357831 (24.6 GB); earlier attempts' scratch is on other nodes.

</details>

<details>
<summary>Supervision, documentation publication and workspace constraints</summary>

The old `table5-active-supervisor-v2` remains stopped. Jobs on 2026-10-12 were supervised by Monitor scripts in an interactive Claude Code session (each expires after 30 minutes and must be re-armed). They do not persist after the session ends. Session monitor script for 357126 polls `sacct`, new queue outcomes, unexpected admission denials (ignoring cumulative-scratch denials while a worker is active) and tracebacks in `N/reddit-table5-190-v9/campaign-357126.out`.

Runtime pointer `R/current_table5_preparation.json` lists every job, cause and fix. The main checkout contains unrelated dirty work; avoid broad staging.

Operational constraints:

- **Never write under `/gpfs/home/zixianma/`**, including subprocess caches or symlink targets. Use approved project/scrubbed paths or `/tmp`, and Python `-B`. Do not override `HOME` or `CODEX_HOME` to evade this boundary.
- Use auto-review for routine permissions. Any command invoking `rm` needs explicit approval of the exact command and targets.
- Do not use SSH, add allocations or API budget without exact authorization, reset cumulative accounting, rerun claimed primary cells, or delete retained historical evidence.
- Preserve frozen science, task identities, actor/selector protocols, raw/effective verdicts, invalid attempts, source lineage and every resource charge.
- Historical Online-Mind2Web diagnostics cannot populate this Table 5 comparison or a new matched Online-Mind2Web comparison.
- Honor the standing requirement to commit/push substantial project documentation to `arm`, excluding unrelated changes and private payloads; verify publication before claiming it succeeded.

</details>
