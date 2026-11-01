# WebPRM baselines: handoff

<a id="webprm-baselines-handover-20261028"></a>
_Last updated November 1, 2026. Earlier status logs are in this file's git history (commit 9bd626a and before)._

**Goal.** Reproduce WebArbiter's Table 5 (gpt-4o actor; no search vs WebShepherd-8B vs WebArbiter-7B on single-site WebArena-Lite tasks) and add our Piotr SelectionARM as an extra selector row. The same comparison was also run on live Online-Mind2Web (OM2W).

**Code:** [`webprm_baselines/`](https://github.com/zixianma/OpenWebRL/tree/arm/webprm_baselines) on `arm` ([README](https://github.com/zixianma/OpenWebRL/blob/arm/webprm_baselines/README.md): code map and workflow), plus `openwebrl/webprm_*.py`, `scripts/*webprm*`, `tests/test_*webprm*`.
**Runtime:** `RT = /gpfs/scrubbed/zixianma/openwebrl-runtime/`; WebArena under `D = RT/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/`; OM2W under `RT/webprm-om2w-table5-20261012/`.

## 1. Summary

1. **With a gpt-4o actor, no selector reliably beats no search.** OM2W: Piotr +2.5 pp (n.s.), WebArbiter −2.8 pp (n.s.), WebShepherd −14.8 pp (sig.). WebArena (v11): Piotr −4.0 pp (n.s.).
2. **Table 5 is not reproduced in absolute terms.** Our v9 rerun is 9.5–19.4 pp below every published row.
3. **Option C (v12)** tests Piotr with the actor it was trained on (OpenWebRL-4B-SFT, screenshot + coordinate actions, 1280×1000). The smoke test passed; the full run is not yet submitted.
4. **WebPRMBench (offline, text only):** none of our selectors comes close to WebArbiter-7B (89.2% pairwise / 74.6% Best-of-N). Our best, the actor-feature heads, reach 75.2% / 46.4%, above Piotr SelectionARM (70.4% / 38.2%).

| Experiment | Section | State |
| --- | --- | --- |
| OM2W, 300 tasks × 5 methods | [§2.1](#webprm-om2w-20261028) | Done |
| WebArena Table 5 rerun (v9), 128 tasks × 3 methods | [§2.2](#webprm-v9-20261028) | Done |
| WebArena gpt-4o ± Piotr (v11), 128 tasks × 2 methods | [§2.3](#webprm-v11-20261028) | Done |
| WebPRMBench, 8 of our selectors, 1,150 states | [§2.5](#webprm-webprmbench-20261028) | Done |
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

<a id="webprm-webprmbench-20261028"></a>
### 2.5 WebPRMBench (offline): our selectors against WebArbiter

**No selector of ours comes close to WebArbiter-7B on its own benchmark (average pairwise 89.2%, Best-of-N 74.6%). Our best are the heads on actor features: Recipe 2 2.6K reaches 75.2% pairwise and the Recipe 1 (Piotr-data) head 74.2% pairwise / 46.4% Best-of-N, 14.0 and 28.2 pp below WebArbiter-7B; they beat Piotr SelectionARM (70.4% / 38.2%) but not the paper's GPT-4o (83.7% / 60.3%). Our 4B LoRAs score below Piotr (65.5% and 63.0% pairwise), mostly because 12–21% of their answers do not parse.**

**Caveats that apply to every number below.** The released benchmark is text only (intent, accessibility tree, history, URLs; no screenshots) and gives two candidates per comparison. None of our selectors runs in its native setting: all were trained on screenshots, and the heads and LoRAs on panels of 2–5 candidates. Published rows are copied from the [WebArbiter paper](https://arxiv.org/abs/2601.21872), Table 2; we did not rerun those models. Training overlap with the benchmark was not checked for any of our selectors. Unparseable answers count as errors, as in the paper.

**Pairwise accuracy (%)** (a pair is correct when the selector prefers the labelled better response):

| Model | Training data | Mind2Web | WebArena | AssistantBench | WorkArena | Average of 4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| WebArbiter-7B | *published (WebArbiter paper)* | 97.1 | 88.4 | 89.2 | 82.1 | 89.2 |
| WebArbiter-3B | *published (WebArbiter paper)* | 93.3 | 82.0 | 78.3 | 81.0 | 83.7 |
| WebShepherd-8B | *published (WebArbiter paper)* | 86.7 | 68.3 | 55.9 | 54.6 | 64.3 |
| WebShepherd-3B | *published (WebArbiter paper)* | 87.5 | 68.2 | 66.7 | 50.0 | 68.1 |
| GPT-5 | *published (WebArbiter paper)* | 80.9 | 84.8 | 81.7 | 81.1 | 82.1 |
| GPT-4o | *published (WebArbiter paper)* | 80.0 | 84.6 | 85.8 | 84.3 | 83.7 |
| Qwen2.5-7B-Instruct | *published (WebArbiter paper)* | 77.8 | 74.9 | 84.2 | 77.6 | 77.6 |
| Piotr SelectionARM (4B LoRA) | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | 67.3 | 73.4 | 80.8 | 60.0 | **70.4** |
| Our 4B LoRA | Recipe 2: 2.6K states, Luna 3 steps after execution | 63.2 | 65.8 | 70.0 | 63.1 | **65.5** |
| Our 4B LoRA | Recipe 2: 1.3K states, Luna 3 steps after execution | 55.8 | 66.2 | 78.3 | 51.9 | **63.0** |
| Our head (22.6M) | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | 68.4 | 82.1 | 77.5 | 68.9 | **74.2** |
| Our head (22.6M) | Recipe 2: 2.6K states, Luna 3 steps after execution | 64.1 | 80.2 | 81.7 | 74.6 | **75.2** |
| Our head (22.6M) | Recipe 2: 1.3K states, Luna 3 steps after execution | 58.4 | 76.4 | 63.3 | 70.0 | **67.0** |
| Our head (22.6M) | Recipe 2: 2.5K states, 22K panels, Luna 3 steps after execution | 43.6 | 64.9 | 67.5 | 52.9 | **57.2** |
| Our head (22.6M) | SFT+RL 1,908 states, Luna 1 step after execution | 44.5 | 60.4 | 48.3 | 54.1 | **51.8** |

**Best-of-N accuracy (%)** (the better response must win all 4 of its pairs; same as our "state accuracy"):

| Model | Training data | Mind2Web | WebArena | AssistantBench | WorkArena | Average of 4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| WebArbiter-7B | *published (WebArbiter paper)* | 89.5 | 68.7 | 70.0 | 70.2 | 74.6 |
| WebArbiter-3B | *published (WebArbiter paper)* | 78.4 | 56.2 | 46.7 | 54.8 | 59.1 |
| WebShepherd-8B | *published (WebArbiter paper)* | 73.7 | 43.9 | 30.0 | 25.5 | 43.3 |
| WebShepherd-3B | *published (WebArbiter paper)* | 65.2 | 41.3 | 46.7 | 21.2 | 43.6 |
| GPT-5 | *published (WebArbiter paper)* | 62.4 | 71.6 | 63.3 | 64.6 | 65.5 |
| GPT-4o | *published (WebArbiter paper)* | 52.6 | 66.7 | 66.7 | 55.2 | 60.3 |
| Qwen2.5-7B-Instruct | *published (WebArbiter paper)* | 39.2 | 42.8 | 53.3 | 35.9 | 42.8 |
| Piotr SelectionARM (4B LoRA) | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | 28.0 | 44.3 | 53.3 | 27.4 | **38.2** |
| Our 4B LoRA | Recipe 2: 2.6K states, Luna 3 steps after execution | 25.3 | 31.8 | 40.0 | 29.7 | **31.7** |
| Our 4B LoRA | Recipe 2: 1.3K states, Luna 3 steps after execution | 19.7 | 33.8 | 56.7 | 23.1 | **33.3** |
| Our head (22.6M) | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | 35.1 | 60.2 | 53.3 | 36.8 | **46.4** |
| Our head (22.6M) | Recipe 2: 2.6K states, Luna 3 steps after execution | 25.3 | 51.7 | 53.3 | 43.9 | **43.6** |
| Our head (22.6M) | Recipe 2: 1.3K states, Luna 3 steps after execution | 21.6 | 48.8 | 36.7 | 42.9 | **37.5** |
| Our head (22.6M) | Recipe 2: 2.5K states, 22K panels, Luna 3 steps after execution | 13.0 | 37.8 | 36.7 | 21.7 | **27.3** |
| Our head (22.6M) | SFT+RL 1,908 states, Luna 1 step after execution | 14.1 | 26.4 | 6.7 | 21.7 | **17.2** |

Mind2Web pools the three Mind2Web splits (707 states); "Average of 4" is the unweighted mean of the four environments, as in the paper. Our rows use the benchmark's original response order, as the paper does. Generative judges (Piotr and our LoRAs) write an answer with WebArbiter's pairwise prompt; heads score the two responses from SFT-4B actor features and pick the higher. (The upstream evaluator's own "official" state score averages six sources, counting the Mind2Web splits separately, and is not comparable to the paper's average.)

*Pending:* the heads trained on the same 1,380 states with pre-execution, 1-step-after and observed-outcome labels (approval requested).

- **On the same data, the head beats the LoRA.** The Recipe 1 head (Piotr's data on a 22.6M head) beats Piotr SelectionARM trained on the same data by +3.8 pp pairwise and +8.1 pp Best-of-N. Among Recipe 2 heads, 2.6K states beat 1.3K (75.2 vs 67.0% pairwise), and the 22K-panel head is at chance on pooled pairs (49.7%; 57.2% on the four-environment average).
- **Heads ignore response order** (swapping the two responses changed no head prediction). **Piotr does not:** it answers "Response 1" on 71–72% of pairs in both orders, so its accuracy moves with presentation order (pooled pairwise 67.4% original, 69.8% swapped). Our LoRAs show little order bias (within 2 pp).
- **Our LoRAs often fail to answer.** They frequently copy the prompt's instructions until the 1,024-token cap without an answer tag: 542 (2.6K) and 967 (1.3K) of 4,600 answers do not parse, vs 44 for Piotr. On parseable answers alone they are right 72.3% and 72.7% of the time.

<details><summary>Response order: generative judges in both orders</summary>

The swapped copy exchanges responses 1 and 2 in every prompt and flips the label; everything else is identical.

| Model | Training data | Correct pairs, original order | Correct pairs, swapped | Pooled pairwise (%), original | Pooled pairwise (%), swapped | Pooled pairwise (%), both orders | Answers "Response 1", original | Unparseable, original | Unparseable, swapped |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Piotr SelectionARM (4B LoRA) | Recipe 1: Piotr's 2,692 states | 3,099 | 3,212 | 67.4 | 69.8 | 68.6 | 3,257 | 44 | 46 |
| Our 4B LoRA | Recipe 2: 2.6K states | 2,934 | 2,952 | 63.8 | 64.2 | 64.0 | 2,160 | 542 | 547 |
| Our 4B LoRA | Recipe 2: 1.3K states | 2,643 | 2,612 | 57.5 | 56.8 | 57.1 | 2,035 | 967 | 1,001 |
</details>

<details><summary>Our runs by source, including the three Mind2Web splits (counts)</summary>

| Model | Training data | Count | Mind2Web cross-task | Mind2Web cross-website | Mind2Web cross-domain | WebArena | AssistantBench | WorkArena | All |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Size | | States (pairs = 4 ×) | 142 | 148 | 417 | 201 | 30 | 212 | 1,150 |
| Piotr SelectionARM (4B LoRA) | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | Correct pairs | 395 | 395 | 1,113 | 590 | 97 | 509 | 3,099 |
| Piotr SelectionARM (4B LoRA) | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | Correct states | 50 | 40 | 108 | 89 | 16 | 58 | 361 |
| Our 4B LoRA | Recipe 2: 2.6K states, Luna 3 steps after execution | Correct pairs | 387 | 371 | 1,028 | 529 | 84 | 535 | 2,934 |
| Our 4B LoRA | Recipe 2: 2.6K states, Luna 3 steps after execution | Correct states | 46 | 38 | 95 | 64 | 12 | 63 | 318 |
| Our 4B LoRA | Recipe 2: 1.3K states, Luna 3 steps after execution | Correct pairs | 338 | 313 | 926 | 532 | 94 | 440 | 2,643 |
| Our 4B LoRA | Recipe 2: 1.3K states, Luna 3 steps after execution | Correct states | 33 | 28 | 78 | 68 | 17 | 49 | 273 |
| Our head (22.6M) | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | Correct pairs | 414 | 415 | 1,106 | 660 | 93 | 584 | 3,272 |
| Our head (22.6M) | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | Correct states | 62 | 48 | 138 | 121 | 16 | 78 | 463 |
| Our head (22.6M) | Recipe 2: 2.6K states, Luna 3 steps after execution | Correct pairs | 365 | 384 | 1,064 | 645 | 98 | 633 | 3,189 |
| Our head (22.6M) | Recipe 2: 2.6K states, Luna 3 steps after execution | Correct states | 41 | 38 | 100 | 104 | 16 | 93 | 392 |
| Our head (22.6M) | Recipe 2: 1.3K states, Luna 3 steps after execution | Correct pairs | 325 | 348 | 979 | 614 | 76 | 594 | 2,936 |
| Our head (22.6M) | Recipe 2: 1.3K states, Luna 3 steps after execution | Correct states | 36 | 32 | 85 | 98 | 11 | 91 | 353 |
| Our head (22.6M) | Recipe 2: 2.5K states, 22K panels, Luna 3 steps after execution | Correct pairs | 246 | 275 | 712 | 522 | 81 | 449 | 2,285 |
| Our head (22.6M) | Recipe 2: 2.5K states, 22K panels, Luna 3 steps after execution | Correct states | 15 | 20 | 57 | 76 | 11 | 46 | 225 |
| Our head (22.6M) | SFT+RL 1,908 states, Luna 1 step after execution | Correct pairs | 281 | 285 | 692 | 486 | 58 | 459 | 2,261 |
| Our head (22.6M) | SFT+RL 1,908 states, Luna 1 step after execution | Correct states | 26 | 26 | 48 | 53 | 2 | 46 | 201 |
</details>
Setup, checkpoints, jobs and raw outputs: [WEBPRM_BASELINES.md](WEBPRM_BASELINES.md#webprmbench-20261018).

## 3. Running an experiment (WebArena)

Details in the [code README](https://github.com/zixianma/OpenWebRL/blob/arm/webprm_baselines/README.md#workflow-webarena).

1. **Build** packages in a CPU-only job: `tools/<ver>/build_*.py` → `D/launch-preparation-<ver>-<name>/`, then run the package tests.
2. **Approve:** write `D/gpu-budget-<ver>-<date>/user-approval-*.json`; the ledger there sets GPU and API caps.
3. **Submit:** `cd /tmp && Q='<quote>' tools/<ver>/submit_<ver>.sh <names>`.
4. **Audit:** cells are audited automatically; verified cells release their site copy.
5. **Replace** stopped jobs with `replace_v11c_r2.sh`, which builds a package for the remaining cells.
6. **Analyse:** `summarize_v9.py`, `paired_v11.py`.

**Access needed:** the Tillicum `vlm` account; read access to `RT` (site images: see [§4.1](#webprm-site-images-20261101); packages, Piotr and OpenWebRL-4B-SFT weights, results); your own OpenAI key in `.env`. Paths are hard-coded to `RT`.

## 4. Where everything lives

| Item | Location |
| --- | --- |
| Code | `arm`: `webprm_baselines/`, `openwebrl/webprm_*.py`, `scripts/*webprm*`, `tests/test_*webprm*` |
| OM2W | `RT/webprm-om2w-table5-20261012/rerun300-0806-20261023/` |
| WebArena v9 | packages `D/launch-preparation-v9{b..f}-*`, tools `D/v9b-tools/` |
| WebArena v11 | packages `D/launch-preparation-v11{b,c}-*`, tools `D/v11-tools/`, `D/v11b-tools/` |
| WebPRMBench | `RT/webprm-webarena-20261011/next-stage-preparation/{piotr,k1-sftrl}-webprmbench-v1/`; results `openwebrl/docs/webprm_results/*webprmbench*.json` |
| WebArena v12 | packages `D/launch-preparation-v12-*`, tools `D/v12-tools/` |
| Not in git | API keys, task configs with answers, site images, trajectories, weights: under `D/` and `RT/` |
| Site images | `RT/webarena-images/` (symlink index; [§4.1](#webprm-site-images-20261101)) |

<a id="webprm-site-images-20261101"></a>
### 4.1 WebArena site images

All downloaded images are indexed under **`RT/webarena-images/`** as symlinks to the original downloads (the originals stay in place because pinned run packages refer to their paths). Each `*-oci/` entry is an OCI layout; every manifest layer is present at the expected size (checked November 1). A `README.md` there gives digests and build notes.

| Entry | Site | Size (GB) |
| --- | --- | ---: |
| `shopping-oci/` | Shopping (Magento) | 45.4 |
| `gitlab-oci/` | GitLab | 48.4 |
| `cms-oci/` | CMS (`shopping_admin_final_0719`) | 2.2 |
| `reddit-oci/`, `reddit.sif` | Reddit (Postmill); `.sif` is the ready-to-run Apptainer image | 45.5, 43.6 |
| `map-oci/`, `map-datasets/` | Map (tile server, Nominatim, OSRM images; 4 data volumes) | 1.1, 189 |
| `scripts/` | links to CMS startup includes, `v12-tools/`, the five v12 packages | – |

**Sharing.** POSIX ACLs are not supported on this filesystem, so access is granted through the Unix group `zixianma` (members include Piotr, `piotrt`): these paths are group-readable, write access is unchanged. Files added later are created in group `all` and need `chgrp zixianma` + `chmod g+r` to be shared. Revoke with `chmod -R g-rwx` on the same targets. This is scrubbed storage (purged after 60 days of inactivity); copy anything you need to keep.

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
