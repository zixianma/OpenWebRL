# RL results

| Dataset | Training run | Updated | Detailed records |
| --- | --- | --- | --- |
| Online-Mind2Web, 300 tasks | [qcq7i4ug](https://wandb.ai/zixianma/openwebrl/runs/qcq7i4ug) | 2026-09-24 | [RL_EVALUATION.md](RL_EVALUATION.md) |

## Local browser · GPT-4.1 · temperature 0 · full 300

| Checkpoint after iteration | Successes | Valid | Invalid | Overall % | Valid-only % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 70 | 234 | 66 | 23.33 | 29.91 |
| 20 | 95 | 232 | 68 | 31.67 | 40.95 |
| 21 | 89 | 233 | 67 | 29.67 | 38.20 |
| 22 | 86 | 236 | 64 | 28.67 | 36.44 |
| 30 | 96 | 248 | 52 | 32.00 | 38.71 |
| 38 | 107 | 228 | 72 | 35.67 | 46.93 |
| 39 | 98 | 228 | 72 | 32.67 | 42.98 |
| 40 | 100 | 231 | 69 | 33.33 | 43.29 |
| 50 | 105 | 234 | 66 | 35.00 | 44.87 |
| 52 | 92 | 227 | 73 | 30.67 | 40.53 |
| 58 | 109 | 230 | 70 | 36.33 | 47.39 |
| 60 | 105 | 230 | 70 | 35.00 | 45.65 |
| 69 | 103 | 221 | 79 | 34.33 | 46.61 |
| 70 | 103 | 229 | 71 | 34.33 | 44.98 |
| 80 | 114 | 229 | 71 | 38.00 | 49.78 |
| 90 | 101 | 222 | 78 | 33.67 | 45.50 |
| 100 | 104 | 227 | 73 | 34.67 | 45.81 |

## Stealth browser · GPT-4.1 · temperature 0 · full 300

| Checkpoint after iteration | Successes | Valid | Invalid | Overall % | Valid-only % | Evaluation |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 38 | 137 | 291 | 9 | 45.67 | 47.08 | Original |
| 80 | 169 | 294 | 6 | 56.33 | 57.48 | Saved trajectories rejudged |

## Stealth browser · o4-mini

| Checkpoint after iteration | Actor temperature | Completed / planned | Successes | Valid | Invalid | Overall % | Valid-only % | Evaluation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 58 | 0.6 | 300 / 300 | 178 | 297 | 3 | 59.33 | 59.93 | Invalid/missing retried once |
| 80 | 0 | 300 / 300 | 166 | 294 | 6 | 55.33 | 56.46 | Original |
| 80 | 0.6 | 300 / 300 | 169 | 296 | 4 | 56.33 | 57.09 | Original |
| 90 | 0.6 | 300 / 300 | 171 | 296 | 4 | 57.00 | 57.77 | Original |

## Metric denominators

| Metric | Calculation |
| --- | --- |
| Overall % | 100 × successes / 300 |
| Valid-only % | 100 × successes / valid tasks |
| Completed-only % | 100 × successes / completed tasks |
| Checkpoint 58 · merged attempts | 290 original valid + 10 retries; 0 missing |
| Checkpoint after N | `iter_{N-1:07d}` |
| Reward collection N → generating checkpoint | after N−1 |

## Evidence

| Results | Record |
| --- | --- |
| Evaluations / debug · 30 migrated runs | [W&B project](https://wandb.ai/zixianma/openwebrl-evals) · [Migration audit](RL_EVALUATION.md#debug-evaluation-wandb-migration-20260913) |
| Local 10–60 | [Historical comparison](RL_EVALUATION.md#baseline-checkpoint-evaluation--first-comparison) |
| Local 69 | [W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after69-293585) |
| Local 70, 80 | [Scheduled evaluations](RL_EVALUATION.md#scheduled-eval70-80-results-20260913) |
| Local 90 | [Completed evaluation](RL_EVALUATION.md#scheduled-eval90-results-20260913) |
| Local 100 | [Completed evaluation](RL_EVALUATION.md#baseline-iter100-results-20260924) |
| Stealth 38, GPT-4.1 | [Browser comparison](RL_EVALUATION.md#browser-use-checkpoint-evaluation) |
| Stealth 80, GPT-4.1 | [Rejudging audit](RL_EVALUATION.md#stealth80-gpt41-rejudge-feasibility-20260913) |
| Stealth 80, o4-mini, temperatures 0 / 0.6 | [Completed comparison](RL_EVALUATION.md#stealth80-temperature-completed-20260913) |
| Stealth 90, o4-mini | [Completed evaluation](RL_EVALUATION.md#stealth90-temperature-plan-20260913) |
| Stealth 58, o4-mini, retry merged | [Retry and original results](RL_EVALUATION.md#after58-invalid-retry-plan-20260914) |

## Original SFT actor · local browser · o4-mini

| Selector | Candidates / turn | Successes / 300 | Valid | Invalid | Overall % | Valid-only % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| None · historical | 1 | 90 | 267 | 33 | 30.00 | 33.71 |
| ScalarARM · historical | 5 | 114 | 251 | 49 | 38.00 | 45.42 |
| SelectionARM · historical | 5 | 128 | 256 | 44 | 42.67 | 50.00 |
| GPT-5.6 Sol · job 294221 | 5 | 132 | 256 | 44 | 44.00 | 51.56 |

| Sol vs SelectionARM · common-valid tasks | N | SelectionARM successes | Sol successes | Difference pp | Paired 95% interval pp | McNemar p |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Historical control | 235 | 118 | 125 | +2.98 | −3.40 to +9.79 | 0.4426 |

| Runtime | H200-hours | Sol API $ | Selector fallbacks | Detailed record |
| --- | ---: | ---: | ---: | --- |
| 55m29s | 1.8494 | 84.8547 | 0 / 4039 | [Sol evaluation](ARM_INFERENCE.md#sol-selection300-completed-294221) |

## Historical ARM diagnostic · GPT-4.1 · fixed 100 · 2026-09-14

| Run | Adam updates | Evaluated | Successes | Valid | Invalid | Overall % | Valid-only % | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 46 | 100 | 24 | 65 | 35 | 24.00 | 36.92 | [295690](RL_EVALUATION.md#arm-early-295690) |
| Original ARM | 46 | 100 | 20 | 72 | 28 | 20.00 | 27.78 | [295759](RL_EVALUATION.md#arm-early-recovery-295759) |
| All-failure ARM | 42 | 100 | 21 | 70 | 30 | 21.00 | 30.00 | [295690](RL_EVALUATION.md#arm-early-295690) |

| Paired subset | N | Baseline successes | All-failure successes | Baseline-only successes | All-failure-only successes | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All tasks | 100 | 24 | 21 | 8 | 5 | 0.5811 |
| Common-valid | 56 | 21 | 18 | 5 | 2 | 0.4531 |

## ARM checkpoints · canonical comparison · GPT-4.1 · local browser · fixed 100

| Run | Recipe | Checkpoint | Evaluated | Successes | Valid | Invalid | Overall % | Valid-only % | Source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | outcome-only reference | `iter_0000019` | 100 | 25 | 71 | 29 | 25.00 | 35.21 | [297011](RL_EVALUATION.md#baseline-iteration-19-fixed-100-20260915) |
| Baseline | outcome-only reference | `iter_0000089` | 100 | 35 | 68 | 32 | 35.00 | 51.47 | [Archived fixed100 slice](RL_EVALUATION.md#baseline-iteration90-fixed100-recovery-20260921) |
| Original ARM bonus | 48 mixed groups | `iter_0000019` | 100 | 26 | 73 | 27 | 26.00 | 35.62 | [296795](RL_EVALUATION.md#arm-iteration-19-evaluations-20260915) |
| All-failure ARM | mixed batch replaced by failure groups | `iter_0000019` | 100 | 28 | 70 | 30 | 28.00 | 40.00 | [296794](RL_EVALUATION.md#arm-iteration-19-evaluations-20260915) |
| Additive ARM | 48 mixed + 8 auxiliary failure groups | `iter_0000019` | 100 | 24 | 77 | 23 | 24.00 | 31.17 | [296816](RL_EVALUATION.md#arm-iteration-19-evaluations-20260915) |
| Original ARM bonus | 48 mixed groups | `iter_0000029` (iteration 30) | 100 | 27 | 74 | 26 | 27.00 | 36.49 | [297412](RL_EVALUATION.md#arm-original-bonus-iteration-30-fixed-100-20260916) |
| All-failure ARM, disjoint complement | 48 mixed + failure-task auxiliary groups | `iter_0000019` | 200 | 62 | 152 | 48 | 31.00 | 40.79 | [297227](RL_EVALUATION.md#all-failure-arm-full-300-task-evaluation-20260915) |
| All-failure ARM, merged full cohort | 48 mixed + failure-task auxiliary groups | `iter_0000019` | 300 | 90 | 222 | 78 | 30.00 | 40.54 | [297227](RL_EVALUATION.md#all-failure-arm-full-300-task-evaluation-20260915) |

## ARM audited full-300 results · GPT-4.1 · local browser · temperature 0

| Method | Iteration | Successes | Valid | Invalid | Overall % | Valid-only % | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Additive bonus | 20 | 85 | 236 | 64 | 28.33 | 36.02 | [307429](RL_EVALUATION.md#arm-additive-iter20-full300-20260920) |
| B: relaxed gate | 20 | 101 | 228 | 72 | 33.67 | 44.30 | [313209](RL_EVALUATION.md#arm-gate-b-iter20-results-20260921) |
| C: relaxed gate + action credit | 20 | 110 | 247 | 53 | 36.67 | 44.53 | [313211](RL_EVALUATION.md#arm-gate-c-iter20-results-20260921) |
| Outcome-only · historical | 70 | 103 | 229 | 71 | 34.33 | 44.98 | [Baseline](RL_EVALUATION.md#scheduled-eval70-80-results-20260913) |
| Original bonus | 70 | 103 | 231 | 69 | 34.33 | 44.59 | [306478](RL_EVALUATION.md#arm-iter70-audit-20260919) |
| All-failure bonus | 70 | 106 | 233 | 67 | 35.33 | 45.49 | [306477](RL_EVALUATION.md#arm-iter70-audit-20260919) |
| Additive bonus | 70 | 112 | 222 | 78 | 37.33 | 50.45 | [307120](RL_EVALUATION.md#arm-iter70-audit-20260919) |
| Outcome-only · historical | 80 | 114 | 229 | 71 | 38.00 | 49.78 | [Baseline](RL_EVALUATION.md#scheduled-eval70-80-results-20260913) |
| Original bonus | 80 | 100 | 222 | 78 | 33.33 | 45.05 | [309685](RL_EVALUATION.md#arm-iter80-launch-20260919) |
| Additive bonus | 80 | 111 | 212 | 88 | 37.00 | 52.36 | [309686](RL_EVALUATION.md#arm-iter80-launch-20260919) |
| All-failure bonus | 80 | 92 | 218 | 82 | 30.67 | 42.20 | [309687](RL_EVALUATION.md#arm-iter80-launch-20260919) |
| Outcome-only · historical | 90 | 101 | 222 | 78 | 33.67 | 45.50 | [Baseline](RL_EVALUATION.md#arm-iter90-results-20260921) |
| All-failure bonus | 90 | 101 | 217 | 83 | 33.67 | 46.54 | [313188](RL_EVALUATION.md#arm-iter90-results-20260921) |
| Additive bonus | 90 | 118 | 216 | 84 | 39.33 | 54.63 | [313408](RL_EVALUATION.md#arm-iter90-results-20260921) |
| Outcome-only | 100 | 104 | 227 | 73 | 34.67 | 45.81 | [318933](RL_EVALUATION.md#baseline-iter100-results-20260924) |
| All-failure bonus | 100 | 107 | 222 | 78 | 35.67 | 48.20 | [316392](RL_EVALUATION.md#arm-allfailure-iter100-results-20260921) |
| Additive bonus | 100 | 109 | 217 | 83 | 36.33 | 50.23 | [313669](RL_EVALUATION.md#arm-additive-iter100-results-20260921) |

## SelectionARM inference · fixed100 · historical actor-only controls

| Actor | Judge | Actor-only overall % | Actor-only valid-only % (successes/valid) | ARM overall % | ARM valid-only % (successes/valid) | Δ overall pp | Δ valid-only pp | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Starting SFT | o4-mini | 26.00 | 30.59 (26/85) | 36.00 | 43.37 (36/83) | +10.00 | +12.79 | [Historical](ARM_RESULTS.md#arm-task-success-314664) |
| Outcome-only iteration 20 | GPT-4.1 | 25.00 | 35.21 (25/71) | 38.00 | 47.50 (38/80) | +13.00 | +12.29 | [314664](ARM_RESULTS.md#arm-task-success-314664) |
| Outcome-only iteration 90 | GPT-4.1 | 35.00 | 51.47 (35/68) | 43.00 | 53.09 (43/81) | +8.00 | +1.62 | [314664](ARM_RESULTS.md#arm-task-success-314664) |

| Trained actor-only variant | Iteration | Fixed100 successes | Valid | Invalid | Overall % | Valid-only % | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B: relaxed gate | 20 | 29 | 69 | 31 | 29.00 | 42.03 | [313209](RL_EVALUATION.md#arm-gate-b-iter20-results-20260921) |
| C: relaxed gate + action credit | 20 | 38 | 79 | 21 | 38.00 | 48.10 | [313211](RL_EVALUATION.md#arm-gate-c-iter20-results-20260921) |
| Outcome-only | 100 | 37 | 68 | 32 | 37.00 | 54.41 | [318933, slice of full300](RL_EVALUATION.md#baseline-iter100-results-20260924) |
| Additive bonus | 100 | 31 | 65 | 35 | 31.00 | 47.69 | [313669, slice of full300](RL_EVALUATION.md#arm-additive-iter100-results-20260921) |
| All-failure bonus | 100 | 26 | 66 | 34 | 26.00 | 39.39 | [316392, slice of full300](RL_EVALUATION.md#arm-allfailure-iter100-results-20260921) |
