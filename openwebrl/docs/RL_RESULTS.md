# RL results

| Dataset | Training run | Updated | Detailed records |
| --- | --- | --- | --- |
| Online-Mind2Web, 300 tasks | [qcq7i4ug](https://wandb.ai/zixianma/openwebrl/runs/qcq7i4ug) | 2026-09-13 | [RL_EVALUATION.md](RL_EVALUATION.md) |

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

## Stealth browser · GPT-4.1 · temperature 0 · full 300

| Checkpoint after iteration | Successes | Valid | Invalid | Overall % | Valid-only % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 38 | 137 | 291 | 9 | 45.67 | 47.08 |

## Stealth browser · o4-mini · temperature 0.6 · partial

| Checkpoint after iteration | Completed / planned | Successes | Valid | Completed invalid | Missing | Completed-only % | Valid-only % | Full-300 lower bound % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 58 | 299 / 300 | 175 | 290 | 9 | 1 | 58.53 | 60.34 | 58.33 |

## Stealth browser · o4-mini · full 300

| Checkpoint after iteration | Actor temperature | Successes | Valid | Invalid | Overall % | Valid-only % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 80 | 0 | 166 | 294 | 6 | 55.33 | 56.46 |
| 80 | 0.6 | 169 | 296 | 4 | 56.33 | 57.09 |

## Metric denominators

| Metric | Calculation |
| --- | --- |
| Overall % | 100 × successes / 300 |
| Valid-only % | 100 × successes / valid tasks |
| Completed-only % | 100 × successes / completed tasks |
| Full-300 lower bound % | 100 × observed successes / 300 |
| Checkpoint after N | `iter_{N-1:07d}` |
| Reward collection N → generating checkpoint | after N−1 |

## Evidence

| Results | Record |
| --- | --- |
| Evaluations / debug · 30 migrated runs | [W&B project](https://wandb.ai/zixianma/openwebrl-evals) · [Migration audit](RL_EVALUATION.md#debug-evaluation-wandb-migration-20260913) |
| Local 10–60 | [Historical comparison](RL_EVALUATION.md#baseline-checkpoint-evaluation--first-comparison) |
| Local 69 | [W&B](https://wandb.ai/zixianma/openwebrl-evals/runs/qcq7i4ug-eval-after69-293585) |
| Local 70, 80 | [Scheduled evaluations](RL_EVALUATION.md#scheduled-eval70-80-results-20260913) |
| Stealth 38, GPT-4.1 | [Browser comparison](RL_EVALUATION.md#browser-use-checkpoint-evaluation) |
| Stealth 80, o4-mini, temperatures 0 / 0.6 | [Completed comparison](RL_EVALUATION.md#stealth80-temperature-completed-20260913) |
| Stealth 58, o4-mini, partial | [Partial-result audit](arm_results/rl_integration/after58-benchmark-partial.json) |

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

## ARM audited full-300 results · GPT-4.1 · local browser · temperature 0

| Method | Iteration | Successes | Valid | Invalid | Overall % | Valid-only % | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Additive bonus | 20 | 85 | 236 | 64 | 28.33 | 36.02 | [307429](RL_EVALUATION.md#arm-additive-iter20-full300-20260920) |
| Outcome-only · historical | 70 | 103 | 229 | 71 | 34.33 | 44.98 | [Baseline](RL_EVALUATION.md#scheduled-eval70-80-results-20260913) |
| Original bonus | 70 | 103 | 231 | 69 | 34.33 | 44.59 | [306478](arm_results/rl_integration/iteration70-audit.json) |
| All-failure bonus | 70 | 106 | 233 | 67 | 35.33 | 45.49 | [306477](arm_results/rl_integration/iteration70-audit.json) |
| Additive bonus | 70 | 112 | 222 | 78 | 37.33 | 50.45 | [307120](arm_results/rl_integration/iteration70-audit.json) |
| Outcome-only · historical | 80 | 114 | 229 | 71 | 38.00 | 49.78 | [Baseline](RL_EVALUATION.md#scheduled-eval70-80-results-20260913) |
| Original bonus | 80 | 100 | 222 | 78 | 33.33 | 45.05 | [309685](RL_EVALUATION.md#arm-iter80-launch-20260919) |
| Additive bonus | 80 | 111 | 212 | 88 | 37.00 | 52.36 | [309686](RL_EVALUATION.md#arm-iter80-launch-20260919) |
