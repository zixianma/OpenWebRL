# ARM inference scaling: controlled results

- **Piotr SelectionARM:** **+5.00 percentage points** across two full300 repeats, at **6.3× tokens** and **1.48× episode latency**.
- **Local v2 selectors:** Luna and Kev27B each add about **9 points**; they differ by one success. Jev's **+2-point** interval includes zero.
- **Compare within each table.** Decoding and selector inputs differ between studies; these results do not rank Piotr against Luna/Kev.

Both headline comparisons cover all 300 Online-Mind2Web tasks. Recorded invalid and unjudged outcomes count as zero. Success is the canonical o4-mini/AgentTrek verdict, which allows partial progress—not strict task completion. **N** is the number of proposed actions; one is executed.

<a id="sft-piotr-repeat-tracker-20261006"></a>
## Piotr SelectionARM: two full300 repeats

Official OpenWebRL-4B-SFT; actor **T=0.7, p=0.9, 1,024 output tokens, 30 turns**. Piotr **SelectionARM**, N=5, greedy selection; this is not ScalarARM.

| Run | Tasks per arm | SFT successes | SFT success | ARM successes | ARM success | ARM gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeat 1 | 300 | 100 | 33.33% | 115 | 38.33% | +5.00 pp |
| Repeat 2 | 300 | 98 | 32.67% | 113 | 37.67% | +5.00 pp |
| **Mean ± sample SD** | 300 | — | **33.00 ± 0.47%** | — | **38.00 ± 0.47%** | **+5.00 ± 0.00 pp** |

Mean gain: **+5.00 pp; paired 95% interval [+1.50, +8.50]**. Bootstrap resamples 300 task clusters with both repeats retained. SD describes the two observed run rates; identical gains do not imply zero uncertainty.

| Method | Mean latency, s | Mean input + output tokens | Mean browser steps |
| --- | ---: | ---: | ---: |
| SFT alone | 144.8 | 146,124 | 14.76 |
| SFT + Piotr SelectionARM | 214.6 | 926,945 | 16.20 |

Plots: repeat 1 — [compute](arm_results/rl_integration/sft-piotr-repeats-final-20261006/repeat-1/cost.png), [latency](arm_results/rl_integration/sft-piotr-repeats-final-20261006/repeat-1/latency.png), [tokens](arm_results/rl_integration/sft-piotr-repeats-final-20261006/repeat-1/tokens.png); repeat 2 — [compute](arm_results/rl_integration/sft-piotr-repeats-final-20261006/repeat-2/cost.png), [latency](arm_results/rl_integration/sft-piotr-repeats-final-20261006/repeat-2/latency.png), [tokens](arm_results/rl_integration/sft-piotr-repeats-final-20261006/repeat-2/tokens.png). Compute is estimated decoder work, not dollars or total measured FLOPs. [Final audit and accounting](ARM_INFERENCE.md#sft-piotr-repeats-20261006) · [Combined aggregate](arm_results/rl_integration/sft-piotr-repeats-final-20261006/combined.json).

<a id="local-sft-selector-results-20261006"></a>
<a id="matched-sft-control"></a>
## Luna / Jev / Kev: matched local v2 comparison

Same official SFT actor; **T=1.0, p=0.95, top-k off, 4,096 output tokens, 30 action attempts**. All selectors receive the same text/DOM view and full candidates, without images. Fresh baseline; all four conditions complete.

<!-- local-v2-selector-results:start -->
| Selector | N | Tasks | Successes | Success | Gain, pp | Paired 95% interval, pp |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| None | 1 | 300 | 94 | **31.33%** | — | — |
| GPT-6 Luna, medium | 5 | 300 | 121 | **40.33%** | +9.00 | [+3.33, +14.67] |
| Jev 1.13.0 | 5 | 300 | 100 | **33.33%** | +2.00 | [-3.67, +7.67] |
| Kev27B | 5 | 300 | 122 | **40.67%** | +9.33 | [+3.67, +15.00] |
<!-- local-v2-selector-results:end -->

[Audited counts and paired comparisons](arm_results/local_sft_selector_controlled_20261006.json) · [Pinned protocol](arm_results/local_inference_rerun_plan_20261006.json). Paired intervals quantify task sampling, not judge error or website drift.

<details>
<summary>Supplementary: three-run common-task summary and partial RL-task ARM</summary>

<a id="sft-piotr-three-run-summary-20261006"></a>

**Same 160 tasks in all three runs.** The initial corrected run stopped early, so this is a duration-selected subset, not a three-run full300 estimate. September is excluded.

| Run | Tasks per arm | SFT successes | SFT success | ARM successes | ARM success | ARM gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Corrected initial | 160 | 53 | 33.13% | 62 | 38.75% | +5.63 pp |
| Repeat 1 | 160 | 57 | 35.63% | 64 | 40.00% | +4.38 pp |
| Repeat 2 | 160 | 57 | 35.63% | 64 | 40.00% | +4.38 pp |
| **Mean ± sample SD** | 160 | — | **34.79 ± 1.44%** | — | **39.58 ± 0.72%** | **+4.79 ± 0.72 pp** |

[Three-run aggregate and unchanged task-file verification](arm_results/rl_integration/sft-piotr-three-run-summary-20261006.json).

<a id="historical-corrected-partial-tracker-20261006"></a>

The initial three-arm comparison has **158 tasks with all three results**; the SFT/Piotr pair alone has 160. All runs were scheduled on the identical 300-task file.

| Selector | N | Matched tasks | Successes | Success |
| --- | ---: | ---: | ---: | ---: |
| None | 1 | 158 | 53 | 33.54% |
| Piotr SelectionARM | 5 | 158 | 60 | 37.97% |
| RL-task SelectionARM | 5 | 158 | 61 | 38.61% |

Partial collection: 493 episodes saved; 900 planned. Both selector-versus-baseline intervals include zero. [Details](ARM_INFERENCE.md#arm-rltasks-corrected-partial-20261006).

</details>

<details>
<summary>Supplementary: direct Jev actor, validity and protocol/accounting</summary>

<a id="local-jev-actor-results-20261006"></a>

<!-- local-v2-actor-results:start -->
| Local v2 actor | Tasks | Successes | Success | Valid tasks | Invalid tasks | Valid-only success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Official SFT | 300 | 94 | 31.33% | 254 | 46 | 37.01% |
| Jev + GPT-4.1-mini typing | 300 | 14 | 4.67% | 260 | 40 | 5.38% |
<!-- local-v2-actor-results:end -->

Jev is a separate complete actor system, not a selector ablation. Its judge-input adapter was repaired and all 23 valid DONE episodes rejudged from unchanged saved evidence. [Audit](arm_results/jev_actor_local_full300_20261006.json).

| Local v2 condition | Tasks | Successes | Valid tasks | Invalid tasks | Valid-only success |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT alone | 300 | 94 | 254 | 46 | 37.01% |
| SFT + GPT-6 Luna, medium | 300 | 121 | 255 | 45 | 47.45% |
| SFT + Jev 1.13.0 | 300 | 100 | 250 | 50 | 40.00% |
| SFT + Kev27B | 300 | 122 | 247 | 53 | 49.39% |

<a id="local-browser-rerun-20261006"></a>
<a id="evidence-and-accounting"></a>

<!-- local-suite-live-status:start -->
| Study | Primary episodes | Separate smokes | Allocated H200-hours | Accounting |
| --- | ---: | ---: | ---: | --- |
| Piotr full300 repeats | 1,200 | 0; startup episodes included | 16.65 | [Final](arm_results/rl_integration/sft-piotr-repeats-final-20261006/combined.json) |
| Local v2 SFT + selectors | 1,200 | 24 | 22.88 | [Final](arm_results/local_sft_selector_controlled_20261006.json) |
| Local v2 Jev actor | 300 | 3 | 0 | [CPU/API ledger](arm_results/jev_actor_local_full300_20261006.json) |
<!-- local-suite-live-status:end -->

Local v2 judges only completed episodes using full actor thoughts/actions and a fresh terminal screenshot. Every valid record has terminal evidence; 186 of 194 invalid SFT records lack fresh terminal evidence. All invalids remain zero. One SFT shard's cleanup used scheduler-cgroup/child-closure evidence rather than a post-exit process scan. [Full protocol and audit limitations](arm_results/local_sft_selector_controlled_20261006.json).

</details>

<details>
<summary>Historical September results — separate reference, excluded from current averages</summary>

<a id="compact-candidate-selection"></a>

| Official SFT with… | N | Tasks | Successes | Success | Valid tasks | Valid-only success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No selector | 1 | 300 | 90 | 30.00% | 267 | 33.71% |
| ScalarARM | 5 | 300 | 114 | 38.00% | 251 | 45.42% |
| SelectionARM | 5 | 300 | 128 | 42.67% | 256 | 50.00% |
| SelectionARM, action-only candidates | 5 | 300 | 105 | 35.00% | 253 | 41.50% |
| GPT-5.6 Sol selector | 5 | 300 | 132 | 44.00% | 256 | 51.56% |

Historical actor-policy receipts are intact; these are not classified as missing-policy runs. Collection dates and harness details differ from the current comparisons. [Historical-gain audit](ARM_INFERENCE.md#arm-same-task-verdict-audit-20261006) · [Sol](arm_results/sol-selection300.json) · [Action-only ablation](arm_results/selection-actions-only.json).

</details>

<details>
<summary>Archived buggy/superseded runs and open comparison gaps</summary>

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
| Archived comparison | Limitation | Current replacement / remaining gap |
| --- | --- | --- |
| October 4 learned ARM versus oracle episode pass@k | Missing actor browser policy | Piotr pair corrected above; controlled pass@k rerun still missing |
| GPT-6 Luna medium/high and GPT-6.1 Sol high actors | Missing policy and lost native conversation state/call IDs | Corrected matched actor comparison still missing |
| Historical SFT + Luna N=5/N=10; Qwen alone/+Luna | Missing actor policy | SFT N=5 replaced by local v2; N=10 and Qwen comparisons remain open |
| Historical hosted SFT + Jev/Kev | Missing actor policy; browser and judge-wrapper differences | Replaced by matched local v2 above |
| Hosted Kev actor and small pilots | Different system/protocol; pilot Luna also had coordinate corruption | Excluded from the controlled headline tables |
<!-- actor-selector-results:end -->

No archived score is used in current means or rankings. Original results, attempts and verdicts remain preserved. [Pre-cleanup tables, plots and protocol history](https://github.com/zixianma/OpenWebRL/blob/a0a6db4e677c7f1959a53f0498ffc9a18d9c9ca0/openwebrl/docs/ARM_INFERENCE_SCALING.md) · [Actor pipeline audit](arm_results/reasoning_actors_full300_20261005/pipeline-debug.json) · [Historical judge audit](arm_results/sft_selector_judge_audit_20261006.json).

</details>
