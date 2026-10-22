# Final inference results

## Headline performance cost and efficiency

<!-- three-benchmark-headlines-v1 -->
**Four methods × three benchmarks · BrowserCode · Updated October 20, 2026.** Both single-call methods use high effort; N3/N5 use high proposals and a high selector. Limits are 12,288 output tokens per call and 30 decisions.

**WebVoyager and DeepShop are complete across all four methods.** DeepShop Sol includes the explicitly approved recovery of 31 budget-affected cells; every original disposition and attempt cost is preserved. OM2W Luna single remains **Not run**.

### Matched task success

Common-valid cohorts: **256 OM2W tasks across Sol/N3/N5, 489 WebVoyager tasks across all four methods, and 140 DeepShop tasks across all four methods**. These rates condition on validity; full fixed-task rates follow.

| Method | Online-Mind2Web · 256 | WebVoyager · 489 | DeepShop · 140 |
|---|---:|---:|---:|
| GPT-6 Sol single high | 64.5% | 72.2% | 45.0% |
| GPT-6 Luna single high | — | 69.1% | 50.0% |
| GPT-6 Luna N3 high | 66.8% | 69.9% | 55.7% |
| GPT-6 Luna N5 high | 69.5% | 72.0% | 52.9% |

### Overall task success

Rates use every fixed task, including invalid outcomes as unsuccessful. Counts and validity are listed separately below.

| Method | Online-Mind2Web · 300 | WebVoyager · 595 | DeepShop · 150 |
|---|---:|---:|---:|
| GPT-6 Sol single high | 57.0% | 64.7% | 44.0% |
| GPT-6 Luna single high | Not run | 65.5% | 48.7% |
| GPT-6 Luna N3 high | 60.0% | 66.1% | 54.0% |
| GPT-6 Luna N5 high | 62.7% | 65.4% | 51.3% |

OM2W uses the original 300-task secondary cohort; the primary comparison on 200 new tasks is retained below. DeepShop reports the latest budget-amended Sol outcomes, including genuine remaining invalids, without replacing the archived original evidence.

### Model API cost across all attempts

Includes every proposal/selector and invalid attempt, with conservative unknown-call charges; DeepShop Sol also includes all pre-amendment attempts. Judge and compute costs are excluded. Dollar amounts are frozen-rate receipt estimates. Separate OM2W retry cohorts and the Sol multi-proposal pilot remain excluded.

| Method | Online-Mind2Web | WebVoyager | DeepShop |
|---|---:|---:|---:|
| GPT-6 Sol single high | $57.21 | $115.60 | $59.64 |
| GPT-6 Luna single high | Not run | $4.61 | $2.10 |
| GPT-6 Luna N3 high | $15.75 | $25.81 | $13.20 |
| GPT-6 Luna N5 high | $19.68 | $27.61 | $16.22 |

### Matched efficiency

All metrics below use the same common-valid tasks within each benchmark: **OM2W 256, WebVoyager 489, DeepShop 140**. OM2W includes its three run methods; the other benchmarks include all four. Latency excludes judge-only work. Browser steps count selected BrowserCode batches.

#### Mean model API cost per matched task

| Method | Online-Mind2Web | WebVoyager | DeepShop |
|---|---:|---:|---:|
| GPT-6 Sol single high | $0.2046 | $0.1882 | $0.3861 |
| GPT-6 Luna single high | — | $0.0073 | $0.0138 |
| GPT-6 Luna N3 high | $0.0546 | $0.0407 | $0.0871 |
| GPT-6 Luna N5 high | $0.0699 | $0.0443 | $0.1088 |

#### Mean task latency

| Method | Online-Mind2Web | WebVoyager | DeepShop |
|---|---:|---:|---:|
| GPT-6 Sol single high | 91.5 s | 89.5 s | 151.4 s |
| GPT-6 Luna single high | — | 94.1 s | 152.8 s |
| GPT-6 Luna N3 high | 177.7 s | 156.9 s | 254.8 s |
| GPT-6 Luna N5 high | 208.0 s | 167.0 s | 279.7 s |

#### Mean browser steps

| Method | Online-Mind2Web | WebVoyager | DeepShop |
|---|---:|---:|---:|
| GPT-6 Sol single high | 14.9 | 12.7 | 21.1 |
| GPT-6 Luna single high | — | 12.9 | 20.4 |
| GPT-6 Luna N3 high | 14.8 | 11.3 | 17.8 |
| GPT-6 Luna N5 high | 14.7 | 10.5 | 17.4 |

[Audited aggregate metrics](arm_results/sol6_luna_three_benchmarks_20261019/aggregate.json) · [Completion audit and recovery scope](ARM_INFERENCE_SCALING.md#deepshop-sol-budget-recovery-20261019)

<details>
<summary>Coverage and success counts</summary>

| Benchmark | Method | Fixed tasks | Successes | Valid | Invalid | Status |
|---|---|---:|---:|---:|---:|---|
| Online-Mind2Web | GPT-6 Sol single high | 300 | 171 | 266 | 34 | Audited |
| Online-Mind2Web | GPT-6 Luna single high | 300 | — | — | — | Not run |
| Online-Mind2Web | GPT-6 Luna N3 high | 300 | 180 | 269 | 31 | Audited |
| Online-Mind2Web | GPT-6 Luna N5 high | 300 | 188 | 267 | 33 | Audited |
| WebVoyager | GPT-6 Sol single high | 595 | 385 | 553 | 42 | Audited |
| WebVoyager | GPT-6 Luna single high | 595 | 390 | 579 | 16 | Audited |
| WebVoyager | GPT-6 Luna N3 high | 595 | 393 | 568 | 27 | Audited |
| WebVoyager | GPT-6 Luna N5 high | 595 | 389 | 555 | 40 | Audited |
| DeepShop | GPT-6 Sol single high | 150 | 66 | 145 | 5 | Audited after budget recovery |
| DeepShop | GPT-6 Luna single high | 150 | 73 | 150 | 0 | Audited |
| DeepShop | GPT-6 Luna N3 high | 150 | 81 | 148 | 2 | Audited |
| DeepShop | GPT-6 Luna N5 high | 150 | 77 | 147 | 3 | Audited |

</details>

<details>
<summary>Original controlled Online-Mind2Web comparison and full-300 cohort</summary>

### GPT-6 Sol single high versus GPT-6 Luna N3 and N5

**Online-Mind2Web 300 · BrowserCode · Final audited results · October 18, 2026.** All proposals and selectors use high reasoning effort; the limit is 30 decisions per episode.

**Luna reduces model API cost per task by 73.2% (N3) and 66.6% (N5), while taking 1.99× and 2.33× as long on the primary shared-valid comparison.** Its observed success rates are higher, but neither primary paired gain is statistically established.

#### Primary comparison on new tasks

The new 200-task cohort is primary. Efficiency below uses exactly the same 165 tasks valid in all three arms. Model cost includes every proposal and selector call; model cost and task latency exclude the judge.

| Model and policy | Shared valid tasks | Successes | Success rate | Mean model cost / task | Mean task latency | Model cost / successful task | Cost reduction vs Sol | Latency vs Sol |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-6 Sol single high | 165 | 104 | 63.0% | $0.2084 | 94.1 s | $0.3306 | 0.0% | 1.00× |
| GPT-6 Luna N3 high + high selector | 165 | 108 | 65.5% | $0.0558 | 186.9 s | $0.0852 | 73.2% | 1.99× |
| GPT-6 Luna N5 high + high selector | 165 | 114 | 69.1% | $0.0696 | 219.7 s | $0.1008 | 66.6% | 2.33× |

Cost per successful task is total model cost across these 165 tasks divided by successes, including cost spent on valid failures. It excludes infrastructure-invalid attempts. N3’s paired success difference is +2.4 percentage points (95% CI −4.8 to +9.7); N5’s is +6.1 points (−0.6 to +12.7).

#### Full 300 task results

The combined cohort is secondary because the first 100 Luna tasks had already been observed. All 900 original dispositions are retained; retry outcomes are separate. Overall success divides by all 300 tasks; valid-only success divides by each arm’s valid count.

| Model and policy | Tasks | Valid | Invalid | Successes | Overall success | Valid-only success |
|---|---:|---:|---:|---:|---:|---:|
| GPT-6 Sol single high | 300 | 266 | 34 | 171 | 57.0% | 64.3% |
| GPT-6 Luna N3 high + high selector | 300 | 269 | 31 | 180 | 60.0% | 66.9% |
| GPT-6 Luna N5 high + high selector | 300 | 267 | 33 | 188 | 62.7% | 70.4% |

For a matched comparison across the full cohort, the following rows use only the same 256 tasks valid in all three arms.

| Model and policy | Shared valid tasks | Successes | Success rate | Mean model cost / task | Mean task latency | Model cost / successful task |
|---|---:|---:|---:|---:|---:|---:|
| GPT-6 Sol single high | 256 | 165 | 64.5% | $0.2046 | 91.5 s | $0.3175 |
| GPT-6 Luna N3 high + high selector | 256 | 171 | 66.8% | $0.0546 | 177.7 s | $0.0817 |
| GPT-6 Luna N5 high + high selector | 256 | 178 | 69.5% | $0.0699 | 208.0 s | $0.1005 |

[Detailed comparison and audit](ARM_INFERENCE_SCALING.md#sol6-luna-browsercode300-20261018) · [Audited aggregate metrics](arm_results/sol6_luna_browsercode300_20261018/aggregate.json)

</details>

<a id="final-selection-head-ablations-20261021"></a>

## Selection head: which training labels work (October 21, 2026)

**A 22.6 M-parameter head on the frozen SFT-4B actor, trained on observed-outcome or lookahead K = 3 labels for 1,380 of our SFT-actor states, is level with Piotr SelectionARM on all 1,045 tasks at 59–66% of its compute per episode. Lookahead K = 1 labels gain about half as much and pre-execution labels gain nothing. Trained on Piotr's data, the same head is the best method overall.** Online-Mind2Web 300, WebVoyager 595, DeepShop 150; T 1.0 / top-p 0.9, 30 steps, five candidates per decision, one episode per task.

| Method | Training data | Params (actor + selector) | OM2W overall | OM2W valid-only | WebVoyager overall | WebVoyager valid-only | DeepShop overall | DeepShop valid-only | All overall | All valid-only | PFLOP per step | PFLOP per episode | Model time per episode | Steps per episode |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 4B, 1 sample | — | 4.44 B | 36.3% | 39.6% | 44.4% | 45.0% | 34.0% | 34.2% | 40.6% | 41.9% | 0.019 | 0.25 | 38 s | 13.6 |
| Piotr SelectionARM | 2,692 states / 35,916 panels, GPT-5.5 picks | 4.44 B + 4.44 B | 41.0% | 46.8% | **52.4%** | **53.8%** | 41.3% | 41.9% | 47.6% | 50.2% | 0.096 | 1.37 | 124 s | 14.3 |
| Head, Piotr's data | 2,692 states / 35,916 panels, GPT-5.5 picks | 4.44 B + 22.6 M | **43.7%** | **49.4%** | 51.3% | 53.4% | **44.0%** | **44.9%** | **48.0%** | **51.1%** | 0.062 | 0.84 | 113 s | 13.6 |
| Head, pre-execution labels | 1,380 states / 1,380 panels, Luna votes | 4.44 B + 22.6 M | 33.0% | 36.9% | 45.7% | 47.1% | 34.0% | 34.2% | 40.4% | 42.5% | 0.061 | 0.95 | 134 s | 15.6 |
| Head, K1 labels | 1,380 states / 1,380 panels, Luna votes | 4.44 B + 22.6 M | 34.0% | 38.6% | 49.2% | 50.5% | 38.0% | 38.8% | 43.3% | 45.6% | 0.061 | 0.96 | 133 s | 15.7 |
| Head, K3 labels | 1,380 states / 1,380 panels, Luna votes | 4.44 B + 22.6 M | 38.3% | 43.1% | 51.1% | 52.0% | 42.7% | 42.7% | 46.2% | 48.2% | 0.061 | 0.91 | 125 s | 14.9 |
| Head, outcome labels | 1,380 states / 1,380 panels, outcomes on draws 0–2 | 4.44 B + 22.6 M | 38.0% | 42.2% | 52.3% | 53.0% | 42.0% | 42.6% | 46.7% | 48.6% | 0.062 | 0.81 | 100 s | 13.1 |

![Overall success on all 1,045 tasks versus parameters (left), PFLOP per episode (middle) and model time per episode (right), one point per method](arm_results/rl_integration/selection-head-labels-vs-cost-20261021.png)

- **Definitions.** "Overall" counts invalid episodes as failures; "valid-only" divides by valid episodes. Model time is actor + selector decision time summed over an episode, excluding browser execution. PFLOP per episode = PFLOP per step × steps per episode.
- **Paired differences (all 1,045 tasks, 95% bootstrap).** vs SFT: Piotr +7.0 [+4.1, +10.0], Piotr-data head +7.5 [+4.4, +10.5], outcome +6.1 [+3.2, +9.0], K3 +5.6 [+2.7, +8.7], K1 +2.7 [−0.4, +5.7], pre-execution −0.2 [−3.2, +2.7]. vs Piotr: outcome −0.9 [−4.0, +2.2], K3 −1.3 [−4.4, +1.6], K1 −4.3 [−7.4, −1.4], pre-execution −7.2 [−10.2, −4.1].
- **Online-Mind2Web is where our labels fall short.** On WebVoyager and DeepShop the outcome and K3 heads match Piotr; on Online-Mind2Web none clearly beats SFT, while the Piotr-data head does (+7.3 pp [+1.7, +13.0]).
- **Why time barely drops.** Generating five candidates takes about 8.2 s of each 8–9 s decision; Piotr's selector adds 0.50 s and the head 0.02 s. Selector FLOPs are a fast prefill, while decoding is bandwidth-bound, so the head's 36% compute saving per step becomes only about 0.5 s per decision, roughly offset by its hidden-state capture. Episode length matters as much: the outcome head is cheapest per episode because it takes the fewest steps. Times come from live runs on different nodes, not a matched timing study.
- **Training data.** Piotr's panels are five actor samples each (about 13 per state), labelled with one GPT-5.5 pick; our panels are one per state with 2–5 distinct actions, reshuffled each epoch. Our states come from the tasks of Piotr's training split.
- A 4B LoRA trained with Piotr's recipe on our 1,380 states does no better than the head offline ([details](ARM_SELECTION_HEAD.md#selection-head-sft-actions-20261021)).
- [Details, per-benchmark intervals and caveats](ARM_SELECTION_HEAD.md#selection-head-sft-actions-20261021) · [Evidence](arm_results/rl_integration/selection-head-sft-actions-full-20261021.json) · [Plot script](arm_results/rl_integration/selection-head-labels-vs-cost-20261021.py)
