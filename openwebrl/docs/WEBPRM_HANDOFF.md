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
4. **WebPRMBench (offline, text only):** with every candidate's thought removed, Piotr SelectionARM scored through its selection prompt is best (77.3% pairwise / 55.3% Best-of-N), ahead of WebArbiter-7B (74.1% / 49.7%, our rerun) and our LoRAs (73.9–75.1%); heads are at most 67.9%, WebShepherd-3B/8B 59.4% / 53.5%. On the released benchmark WebArbiter-7B leads (89.5% / 73.3%, reproducing the paper), and models trained on WebArbiter's data score high (Recipe 1 + WebArbiter LoRA 89.2% / 74.8%, C1 LoRA 85.5% / 66.4%, C1 head 81.3% / 59.8%) mostly by recognising how correct responses' thoughts are written.

| Experiment | Section | State |
| --- | --- | --- |
| OM2W, 300 tasks × 5 methods | [§2.1](#webprm-om2w-20261028) | Done |
| WebArena Table 5 rerun (v9), 128 tasks × 3 methods | [§2.2](#webprm-v9-20261028) | Done |
| WebArena gpt-4o ± Piotr (v11), 128 tasks × 2 methods | [§2.3](#webprm-v11-20261028) | Done |
| WebPRMBench, 13 of our selectors + WebArbiter/WebShepherd 3B/7–8B reruns, 1,150 states, released and actions only | [§2.5](#webprm-webprmbench-20261028) | Done |
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

**With every candidate's thought removed, WebArbiter loses its lead: Piotr SelectionARM, scored through its own selection prompt, is best (77.3% pairwise / 55.3% Best-of-N), ahead of WebArbiter-7B (74.1% / 49.7%) and our LoRAs (73.9–76.9% / 51.0–54.4%); every head is at or below 67.9%, and WebShepherd-3B and -8B fall to 59.4% and 53.5% ([actions only](#webprmbench-actiononly-20261102)).** On the released benchmark the ranking is different, mostly because correct responses can be recognised by how their thoughts are written ([head](#webprmbench-sft-heads-20261101), [LoRA](#webprmbench-lora-c1-20261102)): our reruns of WebArbiter-7B and -3B reproduce the paper (89.5% / 73.3% and 83.9% / 59.5%; paper 89.2% / 74.6% and 83.7% / 59.1%), Piotr's LoRA recipe trained on WebArbiter's SFT pairs plus our Recipe 2 data lands between them (85.5% / 66.4%, one seed), and plus Piotr's Recipe 1 data matches WebArbiter-7B (89.2% / 74.8%; [details](#webprmbench-lora-r1wa-20261104)); the same LoRA on WebArbiter's pairs alone scores higher still (90.6% / 74.8% released, 76.9% / 54.0% actions only) and adding Recipe 3 GPT-5.5 K = 3 data does not help (89.9% / 74.4%, 74.8% / 51.3%; [details](#webprmbench-lora-wa-r3wa-20261104)), as does the published state-matching ranker of [Li et al. 2026](https://arxiv.org/abs/2609.02885) (Qwen2.5-7B ranker given next states predicted by a Qwen3-8B world model: 89.4% / 72.7%, released only, not rerun), and our 22.6M head on the same data reaches the 3B level (81.3% / 59.8%, 3 seeds). Without WebArbiter data, Piotr SelectionARM scored through its selection prompt is best (79.0% / 57.7%), ahead of our Recipe 2 LoRAs (77.4% and 76.9% pairwise) and best heads (75.2% pairwise, Recipe 2 2.6K; 46.4% Best-of-N, Recipe 1). Asked WebArbiter's judge prompt instead, these LoRAs score only 63–70% pairwise, partly because their answers often do not parse.

**Caveats that apply to every number below.** The released benchmark is text only (intent, accessibility tree, history, URLs; no screenshots) and gives two candidates per comparison. None of our selectors runs in its native setting: all were trained on screenshots, and the heads and LoRAs on panels of 2–5 candidates. Published rows are copied from the [WebArbiter paper](https://arxiv.org/abs/2601.21872), Table 2; of those models we reran WebArbiter-3B/-7B and WebShepherd-3B/-8B (rows marked *our rerun*). Training overlap with the benchmark was not checked for any of our selectors. Unparseable answers count as errors, as in the paper.

**Pairwise accuracy (%)** (a pair is correct when the selector prefers the labelled better response):

| Model | Training data | Mind2Web | WebArena | AssistantBench | WorkArena | Average of 4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| WebArbiter-7B | *published (WebArbiter paper)* | 97.1 | 88.4 | 89.2 | 82.1 | 89.2 |
| WebArbiter-3B | *published (WebArbiter paper)* | 93.3 | 82.0 | 78.3 | 81.0 | 83.7 |
| WebArbiter-7B | *our rerun of the public weights* | 96.9 | 87.9 | 86.7 | 86.4 | 89.5 |
| WebArbiter-3B | *our rerun of the public weights* | 92.8 | 82.2 | 79.2 | 81.5 | 83.9 |
| WebShepherd-8B | *published (WebArbiter paper)* | 86.7 | 68.3 | 55.9 | 54.6 | 64.3 |
| WebShepherd-3B | *published (WebArbiter paper)* | 87.5 | 68.2 | 66.7 | 50.0 | 68.1 |
| WebShepherd-8B | *our rerun of the public weights* | 83.2 | 57.7 | 46.7 | 53.3 | 60.2 |
| WebShepherd-3B | *our rerun of the public weights* | 86.4 | 60.8 | 58.3 | 59.9 | 66.4 |
| GPT-5 | *published (WebArbiter paper)* | 80.9 | 84.8 | 81.7 | 81.1 | 82.1 |
| GPT-4o | *published (WebArbiter paper)* | 80.0 | 84.6 | 85.8 | 84.3 | 83.7 |
| Qwen2.5-7B-Instruct | *published (WebArbiter paper)* | 77.8 | 74.9 | 84.2 | 77.6 | 77.6 |
| Piotr SelectionARM (4B LoRA) | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | 67.3 | 73.4 | 80.8 | 60.0 | 70.4 |
| Our 4B LoRA | Recipe 2: 2.6K states, Luna 3 steps after execution | 63.2 | 65.8 | 70.0 | 63.1 | 65.5 |
| Our 4B LoRA | Recipe 2: 1.3K states, Luna 3 steps after execution | 55.8 | 66.2 | 78.3 | 51.9 | 63.0 |
| Our head (22.6M) | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | 68.4 | 82.1 | 77.5 | 68.9 | 74.2 |
| Our head (22.6M) | Recipe 2: 2.6K states, Luna 3 steps after execution | 64.1 | 80.2 | 81.7 | 74.6 | 75.2 |
| Our head (22.6M) | Recipe 2: 1.3K states, Luna 3 steps after execution | 58.4 | 76.4 | 63.3 | 70.0 | 67.0 |
| Our head (22.6M) | Recipe 2: 1.3K states, Luna before execution | 65.6 | 75.1 | 80.0 | 62.5 | 70.8 |
| Our head (22.6M) | Recipe 2: 2.5K states, 22K panels, Luna 3 steps after execution | 43.6 | 64.9 | 67.5 | 52.9 | 57.2 |
| Our head (22.6M) | SFT+RL 1,908 states, Luna 1 step after execution | 44.5 | 60.4 | 48.3 | 54.1 | 51.8 |
| Our head (22.6M), mean of 3 seeds | WebArbiter SFT: 9,642 pairs, 6,393 states (text, benchmark format) | **99.4** | 70.9 | 81.1 | 62.3 | 78.4 |
| Our head (22.6M), mean of 3 seeds | WebArbiter SFT + Recipe 2 2.6K (pairs) | 99.2 | 70.5 | 71.9 | 70.2 | 78.0 |
| Our head (22.6M), mean of 3 seeds | WebArbiter SFT + Recipe 2 2.6K (whole sets) | 99.1 | 79.1 | 72.5 | 74.4 | 81.3 |
| Piotr SelectionARM (4B LoRA), selection-prompt scoring | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | 74.8 | 85.0 | 81.7 | 74.6 | 79.0 |
| Our 4B LoRA, selection-prompt scoring | Recipe 2: 2.6K states, Luna 3 steps after execution | 72.1 | 82.0 | 80.0 | 75.7 | 77.4 |
| Our 4B LoRA, selection-prompt scoring | Recipe 2: 1.3K states, Luna 3 steps after execution | 72.8 | 81.3 | 79.2 | 74.4 | 76.9 |
| Our 4B LoRA, selection-prompt scoring | WebArbiter SFT pairs + Recipe 2 2.6K panels | 97.3 | 86.4 | 80.0 | 78.3 | 85.5 |
| Our 4B LoRA, selection-prompt scoring | WebArbiter SFT pairs + Recipe 1 panels | 98.0 | **87.3** | **90.8** | **80.5** | **89.2** |

**Best-of-N accuracy (%)** (the better response must win all 4 of its pairs; same as our "state accuracy"):

| Model | Training data | Mind2Web | WebArena | AssistantBench | WorkArena | Average of 4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| WebArbiter-7B | *published (WebArbiter paper)* | 89.5 | 68.7 | 70.0 | 70.2 | 74.6 |
| WebArbiter-3B | *published (WebArbiter paper)* | 78.4 | 56.2 | 46.7 | 54.8 | 59.1 |
| WebArbiter-7B | *our rerun of the public weights* | 89.3 | 70.1 | 70.0 | 63.7 | 73.3 |
| WebArbiter-3B | *our rerun of the public weights* | 77.2 | 55.7 | 50.0 | 55.2 | 59.5 |
| WebShepherd-8B | *published (WebArbiter paper)* | 73.7 | 43.9 | 30.0 | 25.5 | 43.3 |
| WebShepherd-3B | *published (WebArbiter paper)* | 65.2 | 41.3 | 46.7 | 21.2 | 43.6 |
| WebShepherd-8B | *our rerun of the public weights* | 69.4 | 37.3 | 30.0 | 30.7 | 41.9 |
| WebShepherd-3B | *our rerun of the public weights* | 69.2 | 38.3 | 36.7 | 33.0 | 44.3 |
| GPT-5 | *published (WebArbiter paper)* | 62.4 | 71.6 | 63.3 | 64.6 | 65.5 |
| GPT-4o | *published (WebArbiter paper)* | 52.6 | 66.7 | 66.7 | 55.2 | 60.3 |
| Qwen2.5-7B-Instruct | *published (WebArbiter paper)* | 39.2 | 42.8 | 53.3 | 35.9 | 42.8 |
| Piotr SelectionARM (4B LoRA) | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | 28.0 | 44.3 | 53.3 | 27.4 | 38.2 |
| Our 4B LoRA | Recipe 2: 2.6K states, Luna 3 steps after execution | 25.3 | 31.8 | 40.0 | 29.7 | 31.7 |
| Our 4B LoRA | Recipe 2: 1.3K states, Luna 3 steps after execution | 19.7 | 33.8 | 56.7 | 23.1 | 33.3 |
| Our head (22.6M) | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | 35.1 | 60.2 | 53.3 | 36.8 | 46.4 |
| Our head (22.6M) | Recipe 2: 2.6K states, Luna 3 steps after execution | 25.3 | 51.7 | 53.3 | 43.9 | 43.6 |
| Our head (22.6M) | Recipe 2: 1.3K states, Luna 3 steps after execution | 21.6 | 48.8 | 36.7 | 42.9 | 37.5 |
| Our head (22.6M) | Recipe 2: 1.3K states, Luna before execution | 28.0 | 45.3 | 63.3 | 32.5 | 42.3 |
| Our head (22.6M) | Recipe 2: 2.5K states, 22K panels, Luna 3 steps after execution | 13.0 | 37.8 | 36.7 | 21.7 | 27.3 |
| Our head (22.6M) | SFT+RL 1,908 states, Luna 1 step after execution | 14.1 | 26.4 | 6.7 | 21.7 | 17.2 |
| Our head (22.6M), mean of 3 seeds | WebArbiter SFT: 9,642 pairs, 6,393 states (text, benchmark format) | **97.8** | 41.5 | 54.4 | 34.0 | 56.9 |
| Our head (22.6M), mean of 3 seeds | WebArbiter SFT + Recipe 2 2.6K (pairs) | 97.1 | 39.1 | 46.7 | 38.4 | 55.3 |
| Our head (22.6M), mean of 3 seeds | WebArbiter SFT + Recipe 2 2.6K (whole sets) | 96.7 | 53.2 | 43.3 | 45.8 | 59.8 |
| Piotr SelectionARM (4B LoRA), selection-prompt scoring | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | 43.3 | 67.2 | 70.0 | 50.5 | 57.7 |
| Our 4B LoRA, selection-prompt scoring | Recipe 2: 2.6K states, Luna 3 steps after execution | 38.9 | 60.7 | 63.3 | 49.1 | 53.0 |
| Our 4B LoRA, selection-prompt scoring | Recipe 2: 1.3K states, Luna 3 steps after execution | 40.6 | 61.7 | 63.3 | 43.9 | 52.4 |
| Our 4B LoRA, selection-prompt scoring | WebArbiter SFT pairs + Recipe 2 2.6K panels | 90.2 | 66.7 | 53.3 | 55.2 | 66.4 |
| Our 4B LoRA, selection-prompt scoring | WebArbiter SFT pairs + Recipe 1 panels | 92.8 | **73.6** | **73.3** | **59.4** | **74.8** |

**Bold** marks the best of our selectors (including Piotr) in each column. Mind2Web pools the three Mind2Web splits (707 states); "Average of 4" is the unweighted mean of the four environments, as in the paper. Our rows use the benchmark's original response order, as the paper does. Generative judges (Piotr and our LoRAs) write an answer with WebArbiter's pairwise prompt, except rows marked "selection-prompt scoring", which read the LoRA's probability of "1" vs "2" in its own selection prompt (text only; [details](#webprmbench-lora-c1-20261102)); heads score the two responses from SFT-4B actor features and pick the higher. *Our rerun* rows run the public WebArbiter weights (`ZYao720/WebArbiter-3B` @ `a54e6274`, `ZYao720/WebArbiter-7B` @ `6518479b`) with their own pairwise prompt through the same evaluator; WebShepherd (`LangAGI-Lab/WebShepherd_3B` @ `adeb60b9`, `WebShepherd_8B` @ `37b57dba`) cannot answer a pairwise prompt, so it follows its released pipeline (one checklist per state, then each response judged against it; score = mean over checklist items of P(yes) + 0.5 P(in progress)), and the higher-scoring response of each pair goes to the same evaluator. Like published rows, reruns are not bolded. (The upstream evaluator's own "official" state score averages six sources, counting the Mind2Web splits separately, and is not comparable to the paper's average.)

The **pre-execution head** (same 1,380 states and recipe as the Recipe 2 1.3K head, labels from Luna judging each candidate before it runs) beats the 3-steps-after head on this benchmark (70.8 vs 67.0% pairwise, 42.3 vs 37.5% Best-of-N; one run, no seeds). The gain is on Mind2Web and AssistantBench (30 states); the 3-steps-after head is ahead on WebArena and WorkArena. The benchmark judges actions before execution, so this label type matches it best.

- **On the same data, the head beats the LoRA.** The Recipe 1 head (Piotr's data on a 22.6M head) beats Piotr SelectionARM trained on the same data by +3.8 pp pairwise and +8.1 pp Best-of-N. Among Recipe 2 heads, 2.6K states beat 1.3K (75.2 vs 67.0% pairwise), and the 22K-panel head is at chance on pooled pairs (49.7%; 57.2% on the four-environment average).
- **Heads ignore response order** (swapping the two responses changed no head prediction). **Piotr does not:** it answers "Response 1" on 71–72% of pairs in both orders, so its accuracy moves with presentation order (pooled pairwise 67.4% original, 69.8% swapped). Our LoRAs show little order bias (within 2 pp). WebArbiter loses a little when swapped (pooled 93.1 → 90.1% for 7B, 88.5 → 86.8% for 3B).
- **Our LoRAs often fail to answer.** They frequently copy the prompt's instructions until the 1,024-token cap without an answer tag: 542 (2.6K) and 967 (1.3K) of 4,600 answers do not parse, vs 44 for Piotr. On parseable answers alone they are right 72.3% and 72.7% of the time.

<a id="webprmbench-sft-heads-20261101"></a>
**Heads trained on WebArbiter's SFT pairs (November 1).** Question: how far does in-format training data take our head? We trained the same 22.6M head on the frozen OpenWebRL-4B-SFT actor's features of WebArbiter's released SFT pairs (`ZYao720/WebArbiter-Data` @ `4205055c`, `sft` split: 9,642 pairs over 6,393 states, 845 Mind2Web-live tasks, text only, in the benchmark's own prompt format; only the verdict is used, not the o3 rationale; the RL split is not used). A uses those pairs alone; B adds our Recipe 2 2.6K panels split into pairs; C1 adds the same panels as whole candidate sets (soft cross-entropy over 2–5 candidates; SFT pairs stay pairs). No intent overlaps the benchmark (0 exact, 0 with word-overlap ≥ 0.6); the epoch is chosen on held-out WebArbiter tasks (84 tasks, 941 pairs), never on the benchmark.

| Head | Training data | Mind2Web | WebArena | AssistantBench | WorkArena | Average of 4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | WebArbiter SFT: 9,642 pairs, 6,393 states (text, benchmark format) | 99.4 ± 0.1 / 97.8 ± 0.6 | 70.9 ± 5.8 / 41.5 ± 8.2 | 81.1 ± 8.8 / 54.4 ± 10.7 | 62.3 ± 4.7 / 34.0 ± 6.0 | 78.4 ± 4.8 / 56.9 ± 5.6 |
| B | WebArbiter SFT + Recipe 2 2.6K (pairs) | 99.2 ± 0.1 / 97.1 ± 0.2 | 70.5 ± 5.8 / 39.1 ± 5.8 | 71.9 ± 5.4 / 46.7 ± 3.3 | 70.2 ± 4.3 / 38.4 ± 5.3 | 78.0 ± 3.5 / 55.3 ± 2.3 |
| C1 | WebArbiter SFT + Recipe 2 2.6K (whole sets) | 99.1 ± 0.4 / 96.7 ± 1.4 | 79.1 ± 1.0 / 53.2 ± 2.0 | 72.5 ± 2.9 / 43.3 ± 3.3 | 74.4 ± 1.1 / 45.8 ± 1.2 | 81.3 ± 0.2 / 59.8 ± 1.2 |

Cells: pairwise / Best-of-N (%), mean ± standard deviation over 3 seeds; scores are identical in both response orders.

- **The 99% on Mind2Web comes from the thought text.** With thoughts removed from every candidate (actions only), seed-0 Mind2Web falls from 99.5 to 65.2% pairwise (A), 99.2 to 58.5% (B) and 99.3 to 61.2% (C1), back to the earlier heads' level (Recipe 1 head 54.5%, Recipe 2 2.6K 58.8% actions only). WebArena moves little (80.0 → 74.8% for C1).
- **The benchmark is almost linearly separable.** A logistic regression on the benchmark's own actor features, cross-validated with whole intents held out, reaches 98.2% pairwise / 95.0% Best-of-N (Mind2Web 99.6, WebArena 96.0, AssistantBench 90.0, WorkArena 94.3% Best-of-N). This is a diagnostic, not a result: correct responses (from successful, revised trajectories) differ from distractors (sampled from about ten other models) in surface features that hold across unseen tasks. Training on data built the same way teaches that difference.
- **Outside Mind2Web the gain is small.** C1 is the only stable variant (average ± 0.2 pp pairwise across seeds); its WebArena (79.1%) is still below the Recipe 1 head (82.1%), and WorkArena (74.4%) matches Recipe 2 2.6K (74.6%). A and B vary by 4–9 pp per environment across seeds.
- **What it means for us:** these benchmark gains probably do not carry over to live selection, where every candidate's thought comes from the same actor. B and C1 can run in the live harness unchanged (they also saw our screenshot features); that is the test that matters. A has never seen screenshots.

<a id="webprmbench-lora-c1-20261102"></a>
**C1 with Piotr's 4B LoRA recipe (November 2).** Question: does the C1 data do better in the 4B LoRA than in the head? Piotr's SelectionARM LoRA recipe is unchanged (rank 32 on OpenWebRL-4B-SFT, 2 epochs, effective batch 16, seed 42; one seed). It is trained on 45,506 examples: the 28,150 Recipe 2 2.6K screenshot panels of `decoder-4b-lora-recipe2-2.6k`, plus the same 8,701 WebArbiter SFT training pairs as the heads, each in both orders (17,356 examples; 46 over the 12,288-token cutoff dropped). WebArbiter pairs use Piotr's selection prompt with the accessibility tree and start URL as text in place of the screenshot. Scoring uses the same prompt: the pick is whichever of "1" or "2" has the higher next-token probability after `{"selection": `, passed to the official evaluator; equal probabilities count as errors (4–30 of 4,600 pairs per pass). The Recipe 2 2.6K LoRA, rescored the same way, is the like-for-like control.

| LoRA | Scoring view | Mind2Web | WebArena | AssistantBench | WorkArena | Average of 4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C1: WebArbiter SFT pairs + Recipe 2 2.6K panels | Released order | 97.3 / 90.2 | 86.4 / 66.7 | 80.0 / 53.3 | 78.3 / 55.2 | 85.5 / 66.4 |
| C1: WebArbiter SFT pairs + Recipe 2 2.6K panels | Swapped order | 97.2 / 90.1 | 86.6 / 66.7 | 79.2 / 46.7 | 79.6 / 53.8 | 85.6 / 64.3 |
| C1: WebArbiter SFT pairs + Recipe 2 2.6K panels | Actions only | 68.6 / 40.7 | 81.3 / 63.7 | 79.2 / 60.0 | 70.8 / 50.0 | 75.0 / 53.6 |
| Recipe 2 2.6K | Released order | 72.1 / 38.9 | 82.0 / 60.7 | 80.0 / 63.3 | 75.7 / 49.1 | 77.4 / 53.0 |
| Recipe 2 2.6K | Swapped order | 72.5 / 39.3 | 81.1 / 59.7 | 81.7 / 60.0 | 76.5 / 50.5 | 77.9 / 52.4 |
| Recipe 2 2.6K | Actions only | 69.2 / 43.3 | 77.9 / 56.7 | 75.0 / 56.7 | 73.3 / 47.2 | 73.9 / 51.0 |

Cells: pairwise / Best-of-N (%), one seed. "Actions only" removes every candidate's thought.

- **The C1 LoRA lands between WebArbiter-3B and -7B** (85.5% / 66.4% vs 83.7 / 59.1 and 89.2 / 74.6), above the C1 head (81.3 / 59.8, 3 seeds). It is one seed; head seeds varied by up to 9 pp per environment.
- **The same style tell.** With actions only, Mind2Web falls from 97.3 to 68.6% pairwise, the Recipe 2 LoRA's level (69.2%). On held-out WebArbiter pairs (84 tasks, 1,876 examples) the C1 LoRA is right 99.6% of the time: the WebArbiter data is almost trivially separable.
- **A small gain survives without thoughts, outside Mind2Web.** Actions only, C1 minus Recipe 2: WebArena +3.5, AssistantBench +4.2 (30 states), WorkArena -2.6 pp pairwise; average +1.1 pairwise / +2.6 Best-of-N. One seed, so this is not established.
- **For LoRAs, the scoring method matters more than the data.** Read from its own selection prompt, the Recipe 2 2.6K LoRA scores 77.4% / 53.0%, against 65.5% / 31.7% with WebArbiter's judge prompt (where 12% of answers do not parse). That is above our best heads (75.2% / 46.4%). Piotr SelectionARM gains most from this scoring (70.4% / 38.2% → 79.0% / 57.7%; [actions only](#webprmbench-actiononly-20261102)). Neither LoRA shows a position bias (swapped order within 0.5 pp pairwise).
- **Live test:** like heads B and C1, the C1 LoRA runs in the live harness unchanged (`lora/merged`), and that remains the test that matters.

<a id="webprmbench-lora-r1wa-20261104"></a>
**WebArbiter SFT pairs + Recipe 1 with Piotr's 4B LoRA recipe (November 4).** Question: does WebArbiter's data help when added to Piotr's data instead of ours? Same recipe and seed as the C1 LoRA, trained on 53,272 examples: Piotr's 35,916 Recipe 1 training panels (five sampled actor actions, one GPT-5.5 pick each; his 3,759 validation panels unused) plus the same 17,356 WebArbiter examples as C1. We rendered Recipe 1 ourselves, through the same selection-prompt code as the Recipe 2 data (task, active-tab URL and past responses parsed from Piotr's stored actor prompts, his screenshot); none was over the 12,288-token cutoff. Piotr's released SelectionARM, trained by him on the same panels without the WebArbiter pairs, is the control. Its prompt rendering may differ slightly from ours, so the comparison is not perfectly controlled. Scored as above.

| LoRA | Scoring view | Mind2Web | WebArena | AssistantBench | WorkArena | Average of 4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| WebArbiter SFT pairs + Recipe 1 panels | Released order | 98.0 / 92.8 | 87.3 / 73.6 | 90.8 / 73.3 | 80.5 / 59.4 | 89.2 / 74.8 |
| WebArbiter SFT pairs + Recipe 1 panels | Swapped order | 97.9 / 93.2 | 87.4 / 71.1 | 89.2 / 73.3 | 80.8 / 58.5 | 88.8 / 74.0 |
| WebArbiter SFT pairs + Recipe 1 panels | Actions only | 63.8 / 33.8 | 79.7 / 61.7 | 89.2 / 76.7 | 67.7 / 45.3 | 75.1 / 54.4 |
| Piotr SelectionARM (Recipe 1 only) | Released order | 74.8 / 43.3 | 85.0 / 67.2 | 81.7 / 70.0 | 74.6 / 50.5 | 79.0 / 57.7 |
| Piotr SelectionARM (Recipe 1 only) | Actions only | 71.4 / 44.7 | 84.5 / 66.7 | 81.7 / 63.3 | 71.8 / 46.7 | 77.3 / 55.3 |

Cells: pairwise / Best-of-N (%), one seed. "Actions only" removes every candidate's thought.

- **On the released benchmark it matches WebArbiter-7B** (89.2% / 74.8% vs our rerun's 89.5% / 73.3% and the paper's 89.2% / 74.6%), 3.7 pp pairwise above C1. It shows no order bias (88.8% / 74.0% swapped). On held-out WebArbiter pairs it is right 99.8% of the time.
- **Without thoughts it does not beat Piotr.** Actions only, it scores 75.1% / 54.4% against Piotr's 77.3% / 55.3% (−2.2 / −0.9 pp): lower on Mind2Web (−7.6), WebArena (−4.7) and WorkArena (−4.1), higher only on AssistantBench (+7.5, 30 states). It is level with C1 (75.0% / 53.6%) and WebArbiter-7B (74.1% / 49.7%).
- **It loses 14.1 pp pairwise without thoughts** (C1 −10.5, Piotr −1.7), almost all of it on Mind2Web (98.0 → 63.8%). Adding WebArbiter's pairs buys the style tell, not better action judgment.
- **Live test:** the merged model (`webprm-lora-r1wa-20261103/lora/merged`) runs in the live harness unchanged.

<a id="webprmbench-actiononly-20261102"></a>
**Every selector with actions only (November 2).** Question: how do the selectors compare once the thought text, which gives away the correct response, is removed? Every candidate keeps its action and loses its thought; everything else (intent, accessibility tree, history, URLs, the evaluator) is unchanged. WebArbiter models answer with their own pairwise prompt; LoRAs are scored through their selection prompt ([details](#webprmbench-lora-c1-20261102)); heads as above.

| Model | Training data | Scoring | Mind2Web | WebArena | AssistantBench | WorkArena | Average of 4 | Average of 4, released (with thoughts) | Δ vs WebArbiter-7B, actions only (pp) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WebArbiter-7B | *our rerun of the public weights* | own judge prompt | 61.2 / 28.1 | 80.7 / 61.2 | 80.8 / 63.3 | 73.6 / 46.2 | 74.1 / 49.7 | 89.5 / 73.3 | — |
| WebArbiter-3B | *our rerun of the public weights* | own judge prompt | 56.9 / 20.1 | 67.4 / 38.3 | 65.8 / 30.0 | 67.6 / 31.6 | 64.4 / 30.0 | 83.9 / 59.5 | -9.7 / -19.7 |
| WebShepherd-8B | *our rerun of the public weights* | checklist + Yes/No probabilities | 57.2 / 32.1 | 57.3 / 36.3 | 42.5 / 20.0 | 56.8 / 32.5 | 53.5 / 30.2 | 60.2 / 41.9 | -20.6 / -19.5 |
| WebShepherd-3B | *our rerun of the public weights* | checklist + Yes/No probabilities | 57.7 / 31.8 | 62.3 / 36.8 | 57.5 / 26.7 | 59.9 / 31.1 | 59.4 / 31.6 | 66.4 / 44.3 | -14.7 / -18.1 |
| Piotr SelectionARM (4B LoRA) | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | selection prompt | **71.4** / **44.7** | **84.5** / **66.7** | 81.7 / 63.3 | 71.8 / 46.7 | **77.3** / **55.3** | 79.0 / 57.7 | +3.2 / +5.6 |
| Our 4B LoRA | Recipe 2: 2.6K states, Luna 3 steps after execution | selection prompt | 69.2 / 43.3 | 77.9 / 56.7 | 75.0 / 56.7 | **73.3** / 47.2 | 73.9 / 51.0 | 77.4 / 53.0 | -0.2 / +1.3 |
| Our 4B LoRA | Recipe 2: 1.3K states, Luna 3 steps after execution | selection prompt | 70.5 / 43.7 | 78.7 / 60.7 | 76.7 / 60.0 | 71.8 / 43.4 | 74.4 / 51.9 | 76.9 / 52.4 | +0.3 / +2.2 |
| Our 4B LoRA | WebArbiter SFT pairs + Recipe 2 2.6K panels | selection prompt | 68.6 / 40.7 | 81.3 / 63.7 | 79.2 / 60.0 | 70.8 / **50.0** | 75.0 / 53.6 | 85.5 / 66.4 | +0.9 / +3.9 |
| Our 4B LoRA | WebArbiter SFT pairs + Recipe 1 panels | selection prompt | 63.8 / 33.8 | 79.7 / 61.7 | **89.2** / **76.7** | 67.7 / 45.3 | 75.1 / 54.4 | **89.2** / **74.8** | +1.0 / +4.6 |
| Our head (22.6M) | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | head | 54.5 / 27.4 | 75.6 / 51.2 | 73.3 / 56.7 | 66.9 / 39.6 | 67.6 / 43.7 | 74.2 / 46.4 | -6.5 / -6.0 |
| Our head (22.6M) | Recipe 2: 2.6K states, Luna 3 steps after execution | head | 58.8 / 27.7 | 70.4 / 44.8 | 78.3 / 50.0 | 64.2 / 31.1 | 67.9 / 38.4 | 75.2 / 43.6 | -6.2 / -11.3 |
| Our head (22.6M) | Recipe 2: 1.3K states, Luna 3 steps after execution | head | 57.7 / 27.7 | 72.4 / 48.3 | 44.2 / 20.0 | 61.8 / 31.6 | 59.0 / 31.9 | 67.0 / 37.5 | -15.1 / -17.8 |
| Our head (22.6M) | Recipe 2: 1.3K states, Luna before execution | head | 47.9 / 20.5 | 71.0 / 47.8 | 76.7 / 63.3 | 60.4 / 31.6 | 64.0 / 40.8 | 70.8 / 42.3 | -10.1 / -8.9 |
| Our head (22.6M) | Recipe 2: 2.5K states, 22K panels, Luna 3 steps after execution | head | 46.3 / 19.7 | 59.5 / 29.4 | 52.5 / 26.7 | 47.6 / 18.4 | 51.5 / 23.5 | 57.2 / 27.3 | -22.6 / -26.2 |
| Our head (22.6M) | SFT+RL 1,908 states, Luna 1 step after execution | head | 52.2 / 24.0 | 74.0 / 51.2 | 46.7 / 20.0 | 67.9 / 39.2 | 60.2 / 33.6 | 51.8 / 17.2 | -13.9 / -16.1 |
| Our head (22.6M), mean of 3 seeds | WebArbiter SFT (A) | head | 60.3 / 29.3 | 65.8 / 37.3 | 65.3 / 40.0 | 55.9 / 27.2 | 61.8 / 33.4 | 78.4 / 56.9 | -12.3 / -16.3 |
| Our head (22.6M), mean of 3 seeds | WebArbiter SFT + Recipe 2 2.6K pairs (B) | head | 60.3 / 27.3 | 74.0 / 47.1 | 64.7 / 37.8 | 64.8 / 33.0 | 66.0 / 36.3 | 78.0 / 55.3 | -8.1 / -13.4 |
| Our head (22.6M), mean of 3 seeds | WebArbiter SFT + Recipe 2 2.6K whole sets (C1) | head | 59.9 / 29.7 | 73.4 / 47.8 | 64.7 / 38.9 | 66.1 / 34.4 | 66.0 / 37.7 | 81.3 / 59.8 | -8.1 / -12.0 |

Cells: pairwise / Best-of-N (%), one run unless marked; seed means for A/B/C1 heads. **Bold**: best of our selectors (including Piotr) per column, excluding the WebArbiter reruns. Original-order averages for the older heads are from the main tables. Chance is 50% pairwise and 20% Best-of-N for a scorer that ranks the five candidates at random.

- **The ranking changes.** Piotr SelectionARM (77.3% / 55.3%) leads; WebArbiter-7B (74.1% / 49.7%) drops to the level of our LoRAs (73.9–75.0% / 51.0–53.6%), and WebArbiter-3B (64.4% / 30.0%) falls below most of our heads. WebShepherd-3B (59.4% / 31.6%) and -8B (53.5% / 30.2%, near chance) score below every LoRA and most heads. With thoughts, WebArbiter-7B was 10.5 pp pairwise ahead of Piotr.
- **Against WebArbiter-7B, the gain is on Mind2Web.** Actions only, Piotr minus WebArbiter-7B: Mind2Web +10.2, WebArena +3.8, AssistantBench +0.9 (30 states), WorkArena -1.8 pp pairwise; C1 LoRA: +7.4, +0.6, -1.6, -2.8. Our best own-data LoRA (Recipe 2 1.3K) is level (+0.3 / +2.2 pp). One run each, so differences of a few points are not established.
- **Models trained on WebArbiter-style data lose the most.** Pairwise drops: WebArbiter-7B −15.4, -3B −19.5, WebShepherd-3B −7.0, -8B −6.7, heads A/B/C1 −12.0 to −16.6, C1 LoRA −10.5, Recipe 1 + WebArbiter LoRA −14.1 pp; models trained only on our screenshots data lose 1.7–8.0 pp (Piotr −1.7, Recipe 2 LoRAs −2.5 and −3.5). Mind2Web carries most of the drop (WebArbiter-7B 96.9 → 61.2%).
- **LoRAs beat heads without thoughts.** Every LoRA (73.9–77.3%) is above every head (51.5–67.9%); with thoughts the best heads matched the Recipe 2 LoRAs. The extra WebArbiter data adds little to the LoRA (C1 75.0% vs Recipe 2 2.6K 73.9%; Recipe 1 + WebArbiter 75.1% vs Piotr 77.3%; one seed each).
- **Caveats.** AssistantBench has 30 states, so per-environment differences there of under about 15 pp are noise. Removing thoughts also removes legitimate reasoning, so this view understates a judge that reads thoughts well; it is a check on the style tell, not a replacement benchmark. LoRAs give exactly equal probability to "1" and "2" on 27–85 of 4,600 actions-only pairs (Piotr 76, Recipe 1 + WebArbiter 80), counted as errors; at most +0.9 pp pairwise if split evenly. The SFT+RL head rises without thoughts (51.8 → 60.2%), from below chance with them.

<details><summary>Response order: generative judges in both orders</summary>

The swapped copy exchanges responses 1 and 2 in every prompt and flips the label; everything else is identical. For selection-prompt rows, "unparseable" counts pairs whose two candidates got equal probability.

| Model | Training data | Correct pairs, original order | Correct pairs, swapped | Pooled pairwise (%), original | Pooled pairwise (%), swapped | Pooled pairwise (%), both orders | Answers "Response 1", original | Unparseable, original | Unparseable, swapped |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Piotr SelectionARM (4B LoRA) | Recipe 1: Piotr's 2,692 states | 3,099 | 3,212 | 67.4 | 69.8 | 68.6 | 3,257 | 44 | 46 |
| Our 4B LoRA | Recipe 2: 2.6K states | 2,934 | 2,952 | 63.8 | 64.2 | 64.0 | 2,160 | 542 | 547 |
| Our 4B LoRA | Recipe 2: 1.3K states | 2,643 | 2,612 | 57.5 | 56.8 | 57.1 | 2,035 | 967 | 1,001 |
| WebArbiter-7B | *our rerun of the public weights* | 4,283 | 4,143 | 93.1 | 90.1 | 91.6 | 2,038 | 1 | 0 |
| WebArbiter-3B | *our rerun of the public weights* | 4,072 | 3,994 | 88.5 | 86.8 | 87.7 | 1,868 | 25 | 25 |
| Piotr SelectionARM (4B LoRA), selection-prompt scoring | Recipe 1: Piotr's 2,692 states | 3,528 | 3,548 | 76.7 | 77.1 | 76.9 | 2,318 | 42 | 31 |
| Our 4B LoRA, selection-prompt scoring | Recipe 2: 2.6K states | 3,436 | 3,449 | 74.7 | 75.0 | 74.8 | 2,201 | 13 | 16 |
| Our 4B LoRA, selection-prompt scoring | Recipe 2: 1.3K states | 3,438 | 3,408 | 74.7 | 74.1 | 74.4 | 2,383 | 70 | 70 |
| Our 4B LoRA, selection-prompt scoring | WebArbiter SFT pairs + Recipe 2 2.6K panels | 4,207 | 4,214 | 91.5 | 91.6 | 91.5 | 2,279 | 8 | 4 |
| Our 4B LoRA, selection-prompt scoring | WebArbiter SFT pairs + Recipe 1 panels | 4,265 | 4,265 | 92.7 | 92.7 | 92.7 | 2,244 | 11 | 16 |
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
| Our head (22.6M) | Recipe 2: 1.3K states, Luna before execution | Correct pairs | 374 | 386 | 1,096 | 604 | 96 | 530 | 3,086 |
| Our head (22.6M) | Recipe 2: 1.3K states, Luna before execution | Correct states | 41 | 42 | 115 | 91 | 19 | 69 | 377 |
| Our 4B LoRA, selection-prompt scoring | Recipe 2: 2.6K states, Luna 3 steps after execution | Correct pairs | 430 | 425 | 1,184 | 659 | 96 | 642 | 3,436 |
| Our 4B LoRA, selection-prompt scoring | Recipe 2: 2.6K states, Luna 3 steps after execution | Correct states | 63 | 51 | 161 | 122 | 19 | 104 | 520 |
| Our 4B LoRA, selection-prompt scoring | WebArbiter SFT pairs + Recipe 2 2.6K panels | Correct pairs | 551 | 575 | 1,626 | 695 | 96 | 664 | 4,207 |
| Our 4B LoRA, selection-prompt scoring | WebArbiter SFT pairs + Recipe 2 2.6K panels | Correct states | 128 | 131 | 379 | 134 | 16 | 117 | 905 |
| Our 4B LoRA, selection-prompt scoring | WebArbiter SFT pairs + Recipe 1 panels | Correct pairs | 557 | 577 | 1,637 | 702 | 109 | 683 | 4,265 |
| Our 4B LoRA, selection-prompt scoring | WebArbiter SFT pairs + Recipe 1 panels | Correct states | 133 | 135 | 388 | 148 | 22 | 126 | 952 |
| WebArbiter-7B | *our rerun of the public weights* | Correct pairs | 555 | 570 | 1,614 | 707 | 104 | 733 | 4,283 |
| WebArbiter-7B | *our rerun of the public weights* | Correct states | 130 | 130 | 371 | 141 | 21 | 135 | 928 |
| WebArbiter-3B | *our rerun of the public weights* | Correct pairs | 542 | 553 | 1,530 | 661 | 95 | 691 | 4,072 |
| WebArbiter-3B | *our rerun of the public weights* | Correct states | 118 | 118 | 310 | 112 | 15 | 117 | 790 |
| WebShepherd-8B | *our rerun of the public weights* | Correct pairs | 494 | 499 | 1,359 | 464 | 56 | 452 | 3,324 |
| WebShepherd-8B | *our rerun of the public weights* | Correct states | 107 | 106 | 278 | 75 | 9 | 65 | 640 |
| WebShepherd-3B | *our rerun of the public weights* | Correct pairs | 502 | 510 | 1,432 | 489 | 70 | 508 | 3,511 |
| WebShepherd-3B | *our rerun of the public weights* | Correct states | 105 | 99 | 285 | 77 | 11 | 70 | 647 |
| Piotr SelectionARM (4B LoRA), selection-prompt scoring | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | Correct pairs | 447 | 444 | 1,223 | 683 | 98 | 633 | 3,528 |
| Piotr SelectionARM (4B LoRA), selection-prompt scoring | Recipe 1: Piotr's 2,692 states, 35K panels, GPT-5.5 before execution | Correct states | 75 | 53 | 178 | 135 | 21 | 107 | 569 |
| Our 4B LoRA, selection-prompt scoring | Recipe 2: 1.3K states, Luna 3 steps after execution | Correct pairs | 436 | 418 | 1,204 | 654 | 95 | 631 | 3,438 |
| Our 4B LoRA, selection-prompt scoring | Recipe 2: 1.3K states, Luna 3 steps after execution | Correct states | 67 | 44 | 176 | 124 | 19 | 93 | 523 |
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
