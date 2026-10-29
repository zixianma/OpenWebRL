# WebPRM baselines: handoff

<a id="webprm-baselines-handover-20261028"></a>
_Last updated October 28, 2026. Earlier status logs are in this file's git history (commit 9bd626a and before)._

**Goal.** Reproduce WebArbiter's Table 5 (gpt-4o actor; no search vs WebShepherd-8B vs WebArbiter-7B on single-site WebArena-Lite tasks) and add our Piotr SelectionARM as an extra selector row. The same comparison was also run on live Online-Mind2Web (OM2W).

**Code:** [`webprm_baselines/`](https://github.com/zixianma/OpenWebRL/tree/arm/webprm_baselines) on `arm` ([README](https://github.com/zixianma/OpenWebRL/blob/arm/webprm_baselines/README.md): code map and workflow), plus `openwebrl/webprm_*.py`, `scripts/*webprm*`, `tests/test_*webprm*`.
**Runtime:** `RT = /gpfs/scrubbed/zixianma/openwebrl-runtime/`; WebArena under `D = RT/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/`; OM2W under `RT/webprm-om2w-table5-20261012/`.

## 1. Summary

1. **With a gpt-4o actor, no selector reliably beats no search.** OM2W: Piotr +2.5 pp (n.s.), WebArbiter −2.8 pp (n.s.), WebShepherd −14.8 pp (sig.). WebArena (v11): Piotr −4.0 pp (n.s.).
2. **Table 5 is not reproduced in absolute terms.** Our v9 rerun is 9.5–19.4 pp below every published row.
3. **Option C (v12)** tests Piotr with the actor it was trained on (OpenWebRL-4B-SFT, screenshot + coordinate actions, 1280×1000). The smoke test passed; the full run is not yet submitted.

| Experiment | Section | State |
| --- | --- | --- |
| OM2W, 300 tasks × 5 methods | [§2.1](#webprm-om2w-20261028) | Done |
| WebArena Table 5 rerun (v9), 128 tasks × 3 methods | [§2.2](#webprm-v9-20261028) | Done |
| WebArena gpt-4o ± Piotr (v11), 128 tasks × 2 methods | [§2.3](#webprm-v11-20261028) | Done |
| WebArena OpenWebRL-4B-SFT ± Piotr (v12) | [§2.4](#webprm-option-c-v12-20261028) | Smoke test passed; full run awaiting approval |

One episode per task × method; success rates are over valid cells; paired comparisons use tasks valid for both methods, with a two-sided sign test.

## 2. Experiments

<a id="webprm-om2w-20261028"></a>
### 2.1 Online-Mind2Web: gpt-4o × five selectors

300 live tasks; actor gpt-4o-2024-08-06; judge gpt-5.6-terra; selectors choose among five gpt-4o candidates.

| Method | Valid cells | Successes | Success (%) | Δ vs no search, 284 paired tasks (pp) | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| No search | 294 | 79 | 26.9 | – | – |
| Piotr SelectionARM | 288 | 85 | 29.5 | +2.5 | 0.46 |
| WebArbiter-7B | 294 | 71 | 24.1 | −2.8 | 0.39 |
| Random of five | 292 | 44 | 15.1 | −12.0 | <0.001 |
| WebShepherd-8B | 292 | 36 | 12.3 | −14.8 | <0.001 |

Source: `RT/webprm-om2w-table5-20261012/rerun300-0806-20261023/final-table.json`.

<a id="webprm-v9-20261028"></a>
### 2.2 WebArena Table 5 rerun (v9): gpt-4o × {no search, WebArbiter, WebShepherd}

Released WebArbiter coordinator and evaluator, 1024×2490 viewport, paper protocol (20 samples at T 1.0; five most frequent become candidates). 128 tasks: Shopping 45, CMS 35, Reddit 18, GitLab 30 (Map excluded as in the paper; Reddit task 27 dropped because its judge `gpt-4-1106-preview` is retired).

| Method | Valid cells | Successes | Success (%) | Published (%) |
| --- | ---: | ---: | ---: | ---: |
| No search | 106 | 27 | 25.5 | 35.0 |
| WebArbiter-7B | 104 | 31 | 29.8 | 49.2 |
| WebShepherd-8B | 87 | 23 | 26.4 | 42.0 |

Published no-search and WebShepherd rows come from a different harness (BrowserGym), so only WebArbiter is a like-for-like target. Most invalid cells (20 / 19 / 19) are tasks graded by the retired fuzzy judge.

<details><summary>Per site and sources</summary>

| Site | No search valid | No search successes | WebArbiter valid | WebArbiter successes | WebShepherd valid | WebShepherd successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shopping | 35 | 9 | 33 | 10 | 30 | 7 |
| CMS | 29 | 6 | 28 | 7 | 23 | 5 |
| GitLab | 26 | 6 | 27 | 7 | 21 | 7 |
| Reddit | 16 | 6 | 16 | 7 | 13 | 4 |

Sources: `D/v9b-tools/summarize_v9.py`; packages `D/launch-preparation-v9{b,c,d,e,f}-*`.
</details>

<a id="webprm-v11-20261028"></a>
### 2.3 WebArena gpt-4o ± Piotr SelectionARM (v11)

Same 128 tasks and evaluator, 1280×720. gpt-4o proposes five candidates and Piotr picks one; no search was rerun in the same jobs.

| Site | No search valid | No search successes | Piotr valid | Piotr successes |
| --- | ---: | ---: | ---: | ---: |
| Shopping | 35 | 3 | 35 | 7 |
| CMS | 29 | 9 | 23 | 3 |
| GitLab | 26 | 7 | 27 | 3 |
| Reddit | 16 | 3 | 16 | 2 |
| **All** | **106** | **22 (20.8%)** | **101** | **15 (14.9%)** |

Paired (100 tasks): Piotr − no search −4.0 pp (95% CI −10.7 to +2.7), p 0.39. Piotr overrides gpt-4o's first candidate on 54% of steps, more often in failed episodes than successful ones. Details: [baseline_comparison.md](baseline_comparison.md#webarena-lite-4site-piotr-20261028).

<a id="webprm-option-c-v12-20261028"></a>
### 2.4 Option C (v12): OpenWebRL-4B-SFT ± Piotr SelectionARM — in progress

**Question.** Does Piotr help when actor and ARM see their training distribution? (On OM2W/WebVoyager/DeepShop it gave +5.9 to +7.0 pp over OpenWebRL-4B-SFT.)

**Design.** Same 128 tasks and evaluator; OpenWebRL's agent loop and screenshot + coordinate actions; viewport 1280×1000 (OpenWebRL's default and the size of Piotr's training states). No search: 1 sample per step. + Piotr: 5 samples per step, Piotr picks one. T 1.0 / top-p 0.9, 30 steps. `done(response)` is the final answer.

**Status.** Smoke test passed (Reddit 14/97/132 and CMS 0/40/101 × both methods; 0.63 H200-hours). Three harness bugs were fixed on the way: local servers inheriting the cluster proxy (`NO_PROXY`), the worker not writing the terminal artifact that audited disposal needs (worker `r2/`), and Chromium in the site namespace inheriting the proxy (stripped in `r2/`). Final smoke package: `launch-preparation-v12-cms-smoke3` (job 395971), with all 6 cells valid, audit-verified and their site state freed.

| Smoke cells | No search valid | No search successes | + Piotr valid | + Piotr successes |
| --- | ---: | ---: | ---: | ---: |
| Reddit (job 395908, pre-`r2`; re-audited offline) | 3 | 1 | 3 | 2 |
| CMS (job 395971) | 3 | 1 | 3 | 1 |

Smoke cells are a pipeline check, not results. Throughput: about 3.7 s/step without search and 5.3 s/step with Piotr (5 samples + selection); ~89 of 141 GB GPU memory.

**Next:** full run, 8 jobs × 1 H200 / 8 CPU / 240 GiB (Reddit 1 × 2.5 h, CMS 2 × 3 h, GitLab 2 × 3.5 h, Shopping 3 × 3.5 h), 30 H200-hour cap, USD 64 API cap. Build it from the `r2/` worker.

**Files:** tools `D/v12-tools/` (committed as `webprm_baselines/webarena/tools/v12/`), packages `D/launch-preparation-v12-*`, ledger `D/gpu-budget-v12-20261028/`.

## 3. Running an experiment (WebArena)

Details in the [code README](https://github.com/zixianma/OpenWebRL/blob/arm/webprm_baselines/README.md#workflow-webarena).

1. **Build** packages in a CPU-only job: `tools/<ver>/build_*.py` → `D/launch-preparation-<ver>-<name>/`, then run the package tests.
2. **Approve:** write `D/gpu-budget-<ver>-<date>/user-approval-*.json`; the ledger there sets GPU and API caps.
3. **Submit:** `cd /tmp && Q='<quote>' tools/<ver>/submit_<ver>.sh <names>`.
4. **Audit:** cells are audited automatically; verified cells release their site copy.
5. **Replace** stopped jobs with `replace_v11c_r2.sh`, which builds a package for the remaining cells.
6. **Analyse:** `summarize_v9.py`, `paired_v11.py`.

**Access needed:** the Tillicum `vlm` account; read access to `RT` (hosted site images, packages, Piotr and OpenWebRL-4B-SFT weights, results); your own OpenAI key in `.env`. Paths are hard-coded to `RT`.

## 4. Where everything lives

| Item | Location |
| --- | --- |
| Code | `arm`: `webprm_baselines/`, `openwebrl/webprm_*.py`, `scripts/*webprm*`, `tests/test_*webprm*` |
| OM2W | `RT/webprm-om2w-table5-20261012/rerun300-0806-20261023/` |
| WebArena v9 | packages `D/launch-preparation-v9{b..f}-*`, tools `D/v9b-tools/` |
| WebArena v11 | packages `D/launch-preparation-v11{b,c}-*`, tools `D/v11-tools/`, `D/v11b-tools/` |
| WebArena v12 | packages `D/launch-preparation-v12-*`, tools `D/v12-tools/` |
| Not in git | API keys, task configs with answers, site images, trajectories, weights: under `D/` and `RT/` |

## 5. Pitfalls

| Pitfall | What to do |
| --- | --- |
| Retired OpenAI models: `gpt-4-1106-preview` (WebArena fuzzy judge), `gpt-4o-2024-05-13`, o4-mini | Fuzzy-judge tasks are invalid; use actor gpt-4o-2024-08-06 and OM2W judge gpt-5.6-terra |
| A job stops after ~21 cells (1 TiB site-state cap) | Keep ≤ 20 cells per job, or rely on audited disposal |
| A job with an API cap under USD 6 runs no cells | Give every job ≥ USD 8 |
| No cell starts in a job's last 30 minutes | Budget time accordingly |
| A job stops after 4 consecutive invalid cells | Submit a replacement for the remaining cells |
| Packages pin tool hashes; editing a referenced tool kills live jobs | Copy tools to a new version directory |
| Local servers inherit the cluster proxy | Put `127.0.0.1` in `NO_PROXY` (fixed in v12) |

## 6. Next steps

1. Submit the full v12 run (needs approval), then compare against no search with the paired sign test.
2. Open questions: the v9 gap to published Table 5 (largest for WebArbiter), and 8 CMS page-load timeouts that hit only Piotr in v11.
