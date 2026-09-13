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

## Planned · stealth browser · o4-mini · temperature 0.6

| Checkpoint after iteration | Tasks | GPUs | Hour cap | Status |
| ---: | ---: | ---: | ---: | --- |
| 38 | 300 | 2 | 3 | Budget approval pending |
| 58 | 300 | 2 | 3 | Budget approval pending; fresh run |
| 80 | 300 | 2 | 3 | Budget approval pending |

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
| Local 10–60 | [Historical comparison](RL_EVALUATION.md#baseline-checkpoint-evaluation--first-comparison) |
| Local 69 | [W&B](https://wandb.ai/zixianma/openwebrl/runs/qcq7i4ug-eval-after69-293585) |
| Local 70, 80 | [Scheduled evaluations](RL_EVALUATION.md#scheduled-eval70-80-results-20260913) |
| Stealth 38, GPT-4.1 | [Browser comparison](RL_EVALUATION.md#browser-use-checkpoint-evaluation) |
| Stealth 58, o4-mini, partial | [Partial-result audit](arm_results/rl_integration/after58-benchmark-partial.json) |
