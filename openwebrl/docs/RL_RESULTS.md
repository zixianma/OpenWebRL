# RL results

| Dataset | Training run | Updated | Detailed records |
| --- | --- | --- | --- |
| Online-Mind2Web, 300 tasks | [qcq7i4ug](https://wandb.ai/zixianma/openwebrl/runs/qcq7i4ug) | 2026-09-24 | [RL_EVALUATION.md](RL_EVALUATION.md) |

## DOM decision models · first 10-task pilot · Browser Use · o4-mini

| Decision model | Text helper | Tasks | Successes | Valid | Invalid | Overall % | Valid-only % | Record |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Jev 1.13.0 | GPT-4.1-mini |10 |1 |10 |0 |10.00 |10.00 |[October4 pilot](rl_results/jev-ultrafast-pilot-20261004.json), [Protocol](RL_EVALUATION.md#jev-ultrafast-online-mind2web-20261004) |
| Kev 0.8B | GPT-4.1-mini |10 |0 |10 |0 |0.00 |0.00 |[Paired pilot](rl_results/kev-pair-pilot-20261004.json), [Protocol](RL_EVALUATION.md#kev-paired-online-mind2web-20261004) |
| Kev 27B | GPT-4.1-mini |10 |3 |10 |0 |30.00 |30.00 |[Paired pilot](rl_results/kev-pair-pilot-20261004.json), [Protocol](RL_EVALUATION.md#kev-paired-online-mind2web-20261004) |

## DOM decision model · full300 · Browser Use · o4-mini · October5

| Decision model | Text helper | Tasks | Audited | Canonical successes | Valid | Invalid | Overall % | Valid-only % | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Kev27B | GPT-4.1-mini | 300 | 300 | 26 | 294 | 6 | 8.67 | 8.84 | [Aggregate](rl_results/kev27b-actor-full300-20261005.json), [Audit and caveats](RL_EVALUATION.md#kev27b-actor-full300-20261004) |

## SFT decision selection · Online-Mind2Web · October4 completed pilot

<a id="sft-decision-selection-20261004"></a>

| Condition | Scheduled | Finished records | Judge successes | Valid | Invalid | All scheduled % | Valid-only % | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SFT alone |10 |10 |4 |9 |1 |40.00 |44.44 |[Protocol and audit](RL_EVALUATION.md#sft-jev-kev-selection-pilot-20261004) |
| SFT + Jev |10 |10 |3 |6 |4 |30.00 |50.00 |[Protocol and audit](RL_EVALUATION.md#sft-jev-kev-selection-pilot-20261004) |
| SFT + Kev0.8B |10 |10 |4 |10 |0 |40.00 |40.00 |[Pilot status](RL_EVALUATION.md#sft-jev-kev-selection-pilot-20261004) |
| SFT + Kev27B |10 |10 |9 |10 |0 |90.00 |90.00 |[Final audit and judge caveats](RL_EVALUATION.md#sft-jev-kev-selection-pilot-20261004) |

## SFT decision selection · full300 · T1/p0.95/k20/4096 · mixed typing harness

| Condition | Scheduled | Audited | Canonical successes | Valid | Invalid | Overall % | Valid-only % | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SFT + Jev | 300 | 300 | 176 | 284 | 16 | 58.67 | 61.97 | [Final audit and caveats](RL_EVALUATION.md#sft-selection-full300-final-20261005) |
| SFT + Kev27B | 300 | 300 | 184 | 291 | 9 | 61.33 | 63.23 | [Aggregate](rl_results/sft-selector-full300-20261005.json) |

| Paired subset | Tasks | Both succeed | Jev only | Kev only | Neither | Jev successes | Kev successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Common-valid | 279 | 136 | 39 | 40 | 64 | 175 | 176 |
| Both post-typing-fix, common-valid | 238 | 116 | 31 | 33 | 58 | 147 | 149 |

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

## Expanded task pool · outcome-only · local browser · GPT-4.1 · temperature 0 · full 300

| Training pool | Iteration | Successes | Valid | Invalid | Overall % | Valid-only % | Record |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Original 2,102 tasks, historical |10 |70 |234 |66 |23.33 |29.91 |[Historical evaluation](RL_EVALUATION.md#baseline-checkpoint-evaluation--first-comparison) |
| Expanded 4,102 tasks |10 |81 |236 |64 |27.00 |34.32 |[October3 audit](arm_results/rl_integration/expanded4102-iteration10-audit.json) |
| Original 2,102 tasks, historical |20 |95 |232 |68 |31.67 |40.95 |[Historical evaluation](RL_EVALUATION.md#baseline-checkpoint-evaluation--first-comparison) |
| Expanded 4,102 tasks |20 |100 |237 |63 |33.33 |42.19 |[October4 audit](arm_results/rl_integration/expanded4102-iteration20-audit.json) |
| Original 2,102 tasks, historical |30 |96 |248 |52 |32.00 |38.71 |[Historical evaluation](RL_EVALUATION.md#baseline-checkpoint-evaluation--first-comparison) |
| Expanded 4,102 tasks |30 |99 |243 |57 |33.00 |40.74 |[October5 audit](arm_results/rl_integration/expanded4102-iteration30-audit.json) |
| Original 2,102 tasks, historical |40 |100 |231 |69 |33.33 |43.29 |[Historical evaluation](RL_EVALUATION.md#baseline-checkpoint-evaluation) |
| Expanded 4,102 tasks |40 |117 |258 |42 |39.00 |45.35 |[October6 audit](arm_results/rl_integration/expanded4102-iteration40-audit.json) |
| Original 2,102 tasks, historical |50 |105 |234 |66 |35.00 |44.87 |[Historical evaluation](RL_EVALUATION.md#baseline-checkpoint-evaluation) |
| Expanded 4,102 tasks |50 |105 |240 |60 |35.00 |43.75 |[October6 audit](arm_results/rl_integration/expanded4102-iteration50-audit.json) |

| Comparison | Overall Δ | Valid-only Δ | Interpretation |
| --- | ---: | ---: | --- |
| Expanded − historical original, iteration10 |+3.67 pp |+4.41 pp |[Exploratory historical comparison](RL_EVALUATION.md#expanded4102-iter10-results-20261003) |
| Expanded − historical original, iteration20 |+1.67 pp |+1.25 pp |[Exploratory historical comparison](RL_EVALUATION.md#expanded4102-iter10-results-20261003) |
| Expanded − historical original, iteration30 |+1.00 pp |+2.03 pp |[Exploratory historical comparison](RL_EVALUATION.md#expanded4102-iter10-results-20261003) |
| Expanded iteration30 −20 |−0.33 pp |−1.45 pp |[Checkpoint comparison](RL_EVALUATION.md#expanded4102-iter10-results-20261003) |
| Expanded − historical original, iteration40 |+5.67 pp |+2.06 pp |[Exploratory historical comparison](RL_EVALUATION.md#expanded4102-iter10-results-20261003) |
| Expanded iteration40 −30 |+6.00 pp |+4.61 pp |[Checkpoint comparison](RL_EVALUATION.md#expanded4102-iter10-results-20261003) |
| Expanded − historical original, iteration50 |0.00 pp |−1.12 pp |[Exploratory historical comparison](RL_EVALUATION.md#expanded4102-iter10-results-20261003) |
| Expanded iteration50 −40 |−4.00 pp |−1.60 pp |[Checkpoint comparison](RL_EVALUATION.md#expanded4102-iter10-results-20261003) |

## Mixed-only relaxed-B comparison · GPT-4.1 · temperature 0 · full 300

| Method | Iteration | Successes | Valid | Invalid | Overall % | Valid-only % | Record |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Outcome-only baseline, historical |10 |70 |234 |66 |23.33 |29.91 |[Historical evaluation](RL_EVALUATION.md#baseline-checkpoint-evaluation--first-comparison) |
|  |20 |95 |232 |68 |31.67 |40.95 |[Historical evaluation](RL_EVALUATION.md#baseline-checkpoint-evaluation) |
|  |30 |96 |248 |52 |32.00 |38.71 |[Historical evaluation](RL_EVALUATION.md#baseline-checkpoint-evaluation) |
|  |40 |100 |231 |69 |33.33 |43.29 |[Historical evaluation](RL_EVALUATION.md#baseline-checkpoint-evaluation) |
|  |50 |105 |234 |66 |35.00 |44.87 |[Historical evaluation](RL_EVALUATION.md#baseline-checkpoint-evaluation) |
|  |60 |105 |230 |70 |35.00 |45.65 |[Historical evaluation](RL_EVALUATION.md#baseline-checkpoint-evaluation) |
|  |70 |103 |229 |71 |34.33 |44.98 |[Historical evaluation](RL_EVALUATION.md#baseline-checkpoint-evaluation) |
|  |80 |114 |229 |71 |38.00 |49.78 |[Historical evaluation](RL_EVALUATION.md#baseline-checkpoint-evaluation) |
|  |90 |101 |222 |78 |33.67 |45.50 |[Historical evaluation](RL_EVALUATION.md#baseline-checkpoint-evaluation) |
| Mixed-only bonus + relaxed B |10 |92 |234 |66 |30.67 |39.32 |[September28 audit](arm_results/rl_integration/mixed-bonus-iteration10-audit.json) |
|  |20 |92 |225 |75 |30.67 |40.89 |[September28 audit](arm_results/rl_integration/mixed-bonus-iteration20-audit.json) |
|  |30 |96 |228 |72 |32.00 |42.11 |[September29 audit](arm_results/rl_integration/mixed-bonus-iteration30-audit.json) |
|  |40 |100 |228 |72 |33.33 |43.86 |[September29 audit](arm_results/rl_integration/mixed-bonus-iteration40-audit.json) |
|  |50 |116 |234 |66 |38.67 |49.57 |[September30 audit](arm_results/rl_integration/mixed-bonus-iteration50-audit.json) |
|  |60 |113 |232 |68 |37.67 |48.71 |[September30 audit](arm_results/rl_integration/mixed-bonus-iteration60-audit.json) |
|  |70 |115 |235 |65 |38.33 |48.94 |[October1 audit](arm_results/rl_integration/mixed-bonus-iteration70-audit.json) |
|  |80 |108 |221 |79 |36.00 |48.87 |[October1 audit](arm_results/rl_integration/mixed-bonus-iteration80-audit.json) |
|  |90 |122 |233 |67 |40.67 |52.36 |[October2 audit](arm_results/rl_integration/mixed-bonus-iteration90-audit.json) |
| Mixed-only reweight + relaxed B |10 |82 |240 |60 |27.33 |34.17 |[September28 audit](arm_results/rl_integration/mixed-reweight-iteration10-audit.json) |
|  |20 |97 |227 |73 |32.33 |42.73 |[September28 audit](arm_results/rl_integration/mixed-reweight-iteration20-audit.json) |
|  |30 |90 |219 |81 |30.00 |41.10 |[September29 audit](arm_results/rl_integration/mixed-reweight-iteration30-audit.json) |
|  |40 |106 |225 |75 |35.33 |47.11 |[September29 audit](arm_results/rl_integration/mixed-reweight-iteration40-audit.json) |
|  |50 |111 |237 |63 |37.00 |46.84 |[September29 audit](arm_results/rl_integration/mixed-reweight-iteration50-audit.json) |
|  |60 |110 |233 |67 |36.67 |47.21 |[September30 audit](arm_results/rl_integration/mixed-reweight-iteration60-audit.json) |
|  |70 |116 |234 |66 |38.67 |49.57 |[October1 audit](arm_results/rl_integration/mixed-reweight-iteration70-audit.json) |
|  |80 |118 |231 |69 |39.33 |51.08 |[October1 audit](arm_results/rl_integration/mixed-reweight-iteration80-audit.json) |
|  |90 |113 |227 |73 |37.67 |49.78 |[October2 audit](arm_results/rl_integration/mixed-reweight-iteration90-audit.json) |

## Mixed-only40/50/60 checkpoint mean · full300

| Method | Overall mean % | Valid-only mean % | Record |
| --- | ---: | ---: | --- |
| Bonus + relaxed B |36.56 |47.38 |[Checkpoint aggregate](arm_results/rl_integration/mixed-pair-iteration40-60-summary.json) |
| Reweight + relaxed B |36.33 |47.05 |[Checkpoint aggregate](arm_results/rl_integration/mixed-pair-iteration40-60-summary.json) |

## Stealth browser · GPT-4.1 · temperature 0 · full 300

| Method | Checkpoint after iteration | Successes | Valid | Invalid | Overall % | Valid-only % | Evaluation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Outcome-only baseline |38 |137 |291 |9 |45.67 |47.08 |Original |
|  |80 |169 |294 |6 |56.33 |57.48 |Saved trajectories rejudged |
| Additive bonus |90 |159 |283 |17 |53.00 |56.18 |[September29 first pass](arm_results/rl_integration/stealth-additive-iteration90-audit.json) |
| Gate B: relaxed gate |90 |169 |288 |12 |56.33 |58.68 |[September29 first pass](arm_results/rl_integration/stealth-gate-b-iteration90-audit.json) |

## Matched stealth browser · o4-mini · temperature 0.6 · September29–30

| Method | Iteration | Repeat | Successes | Valid | Invalid | Overall % | Valid-only % | Fixed100 overall / valid-only % | Record |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Outcome-only baseline |90 |1 |163 |286 |14 |54.33 |56.99 |56.00 /57.14 |[Merged audit](arm_results/rl_integration/stealth-o4-t06-baseline-iteration90-audit.json) |
| |90 |2 |173 |290 |10 |57.67 |59.66 |58.00 /59.18 |[Fresh repeat2 audit](arm_results/rl_integration/stealth-o4-t06-baseline-iteration90-repeat2-audit.json) |
| |90 |3 |161 |286 |14 |53.67 |56.29 |52.00 /54.17 |[Fresh repeat3 audit](arm_results/rl_integration/stealth-o4-t06-baseline-iteration90-repeat3-audit.json) |
| Additive bonus |90 |1 |179 |286 |14 |59.67 |62.59 |60.00 /61.86 |[Merged audit](arm_results/rl_integration/stealth-o4-t06-additive-iteration90-audit.json) |
| |90 |2 |176 |286 |14 |58.67 |61.54 |61.00 /61.62 |[Fresh repeat2 audit](arm_results/rl_integration/stealth-o4-t06-additive-iteration90-repeat2-audit.json) |
| |90 |3 |171 |284 |16 |57.00 |60.21 |55.00 /57.29 |[Fresh repeat3 audit](arm_results/rl_integration/stealth-o4-t06-additive-iteration90-repeat3-audit.json) |
| Gate B: relaxed gate |90 |1 |166 |286 |14 |55.33 |58.04 |62.00 /63.27 |[Merged audit](arm_results/rl_integration/stealth-o4-t06-gate-b-iteration90-audit.json) |
| |90 |2 |189 |288 |12 |63.00 |65.63 |67.00 /68.37 |[Fresh repeat2 audit](arm_results/rl_integration/stealth-o4-t06-gate-b-iteration90-repeat2-audit.json) |
| |90 |3 |174 |288 |12 |58.00 |60.42 |61.00 /61.00 |[Fresh repeat3 audit](arm_results/rl_integration/stealth-o4-t06-gate-b-iteration90-repeat3-audit.json) |

## Mixed-only relaxed B · stealth · o4-mini · T0.6 · October6 · partial repeat set

| Method | Iteration | Repeat | Verified repeats | Successes | Valid | Invalid | Overall % | Valid-only % | Status / record |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Mixed-only bonus + relaxed B |90 |1 |1/3 |182 |299 |1 |60.67 |60.87 |[Verified cohort](arm_results/rl_integration/mixed-stealth90-three-repeats-20261006.json) |
| Mixed-only reweight + relaxed B |90 |1 |1/3 |170 |298 |2 |56.67 |57.05 |[Verified cohort](arm_results/rl_integration/mixed-stealth90-three-repeats-20261006.json) |

| Protocol / comparison scope | Value |
| --- | --- |
| Actor / judge |T0.6 /p0.95 /k20 /4096 tokens /30 turns; o4-mini /AgentTrek |
| Task cohort / inference |Same300 OM2W task IDs; actor-only |
| Three-repeat mean / SD |Pending |
| September29–30 comparison |Different collection dates; descriptive only |
| Details |[Protocol and verification](RL_EVALUATION.md#arm-mixed-stealth90-three-repeats-20261006) |

## Matched stealth iteration90 · three-repeat mean ± sample SD

| Method | Iteration | Full300 overall mean ± SD | Valid-only mean ± SD | Overall Δ vs baseline | Valid denominators, repeats1/2/3 |
| --- | ---: | --- | --- | ---: | --- |
| Outcome-only baseline |90 |**55.22 ± 2.14%** |57.65 ± 1.77% |+0.00 pp |286/290/286 |
| Additive bonus |90 |**58.44 ± 1.35%** |61.45 ± 1.19% |+3.22 pp |286/286/284 |
| Gate B: relaxed gate |90 |**58.78 ± 3.89%** |61.36 ± 3.88% |+3.56 pp |286/288/288 |

| Source |
| --- |
| [Nine-cohort aggregate and audit hashes](arm_results/rl_integration/stealth-o4-t06-iteration90-three-repeat-summary.json) |

## Matched iteration90 · overall paired95% CIs · three-pair Holm correction

| Benchmark | Comparison | Difference pp | Pointwise95% CI pp | Exact p | Holm p | Record |
| --- | --- | ---: | --- | ---: | ---: | --- |
| OM2W300 ×3 | Additive − Outcome-only |+3.22 |[-0.44, +6.78] |0.0959 |0.2198 |[Analysis](ARM_RESULTS.md#arm-stealth90-paired-inference-20260930) |
| OM2W300 ×3 | Gate B − Outcome-only |+3.56 |[-0.22, +7.22] |0.0733 |0.2198 |[Analysis](ARM_RESULTS.md#arm-stealth90-paired-inference-20260930) |
| OM2W300 ×3 | Gate B − Additive |+0.33 |[-3.44, +4.22] |0.9096 |0.9096 |[Analysis](ARM_RESULTS.md#arm-stealth90-paired-inference-20260930) |
| WebVoyager595 ×1 | Additive − Outcome-only |-2.52 |[-6.72, +1.68] |0.2786 |0.8141 |[Analysis](ARM_RESULTS.md#arm-stealth90-paired-inference-20260930) |
| WebVoyager595 ×1 | Gate B − Outcome-only |+0.17 |[-4.20, +4.54] |1.0000 |1.0000 |[Analysis](ARM_RESULTS.md#arm-stealth90-paired-inference-20260930) |
| WebVoyager595 ×1 | Gate B − Additive |+2.69 |[-1.68, +7.23] |0.2714 |0.8141 |[Analysis](ARM_RESULTS.md#arm-stealth90-paired-inference-20260930) |

## OM2W difficulty · iteration90 stealth · o4-mini · T0.6 · three-repeat mean ± sample SD

| Method | Difficulty | Tasks/repeat | Overall mean ± SD | Valid-only mean ± SD | Overall Δ vs baseline | Valid counts, repeats 1/2/3 |
| --- | --- | ---: | --- | --- | ---: | --- |
| Outcome-only baseline | Easy | 80 | 70.00 ± 1.25% | 72.11 ± 1.39% | +0.00 pp | 78/78/77 |
| Outcome-only baseline | Medium | 141 | 55.79 ± 6.96% | 58.49 ± 6.25% | +0.00 pp | 132/137/134 |
| Outcome-only baseline | Hard | 79 | 39.24 ± 3.35% | 41.14 ± 3.32% | +0.00 pp | 76/75/75 |
| Additive | Easy | 80 | 74.58 ± 2.60% | 75.86 ± 3.20% | +4.58 pp | 78/79/79 |
| Additive | Medium | 141 | 56.74 ± 1.88% | 60.92 ± 1.20% | +0.95 pp | 136/130/128 |
| Additive | Hard | 79 | 45.15 ± 0.73% | 47.37 ± 1.07% | +5.91 pp | 72/77/77 |
| Gate B | Easy | 80 | 67.92 ± 5.05% | 69.64 ± 4.71% | -2.08 pp | 77/78/79 |
| Gate B | Medium | 141 | 62.65 ± 4.09% | 65.73 ± 3.58% | +6.86 pp | 133/136/134 |
| Gate B | Hard | 79 | 42.62 ± 3.19% | 44.92 ± 3.95% | +3.38 pp | 76/74/75 |

| Source / definition |
| --- |
| [Difficulty audit: human steps 1–5 / 6–10 / 11+; two stale labels corrected for analysis](ARM_RESULTS.md#arm-stealth90-difficulty-20260930) |
| [All 2,700 per-task outcomes](http://localhost:8765/arm_om2w_difficulty_verdicts.csv) |

## Stealth browser · o4-mini

| Checkpoint after iteration | Actor temperature | Completed / planned | Successes | Valid | Invalid | Overall % | Valid-only % | Evaluation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 58 | 0.6 | 300 / 300 | 178 | 297 | 3 | 59.33 | 59.93 | Invalid/missing retried once |
| 80 | 0 | 300 / 300 | 166 | 294 | 6 | 55.33 | 56.46 | Original |
| 80 | 0.6 | 300 / 300 | 169 | 296 | 4 | 56.33 | 57.09 | Original |
| 90 | 0.6 | 300 / 300 | 171 | 296 | 4 | 57.00 | 57.77 | Original |

## WebVoyager · stealth · GPT-4o · temperature0.6 ·595 tasks

| Method | Iteration | Overall | Valid-only | Successes / valid / total | Overall Δ vs baseline |
| --- | ---: | ---: | ---: | --- | ---: |
| Outcome-only baseline |90 |66.89% |68.27% |398 /583 /595 |+0.00 pp |
| Additive ARM |90 |64.37% |65.03% |383 /589 /595 |-2.52 pp |
| Gate B |90 |67.06% |67.86% |399 /588 /595 |+0.17 pp |

| Sources |
| --- |
| [Outcome-only baseline audit](arm_results/rl_integration/webvoyager-gpt4o-t06-baseline-iteration90-audit.json) · [Additive ARM audit](arm_results/rl_integration/webvoyager-gpt4o-t06-additive-iteration90-audit.json) · [Gate B audit](arm_results/rl_integration/webvoyager-gpt4o-t06-gate-b-iteration90-audit.json) |

## Metric denominators

| Metric | Calculation |
| --- | --- |
| Overall % | 100 × successes / cohort size (OM2W300; WebVoyager595) |
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
| Original bonus | 20 | 84 | 227 | 73 | 28.00 | 37.00 | [332476](RL_EVALUATION.md#arm-original-backfill-results-20260927) |
| Original bonus | 30 | 98 | 225 | 75 | 32.67 | 43.56 | [332477](RL_EVALUATION.md#arm-original-backfill-results-20260927) |
| Original bonus | 40 | 96 | 224 | 76 | 32.00 | 42.86 | [332478](RL_EVALUATION.md#arm-original-backfill-results-20260927) |
| Original bonus | 50 | 102 | 236 | 64 | 34.00 | 43.22 | [332479](RL_EVALUATION.md#arm-original-backfill-results-20260927) |
| Original bonus | 60 | 107 | 234 | 66 | 35.67 | 45.73 | [332480](RL_EVALUATION.md#arm-original-backfill-results-20260927) |
| Additive bonus | 20 | 85 | 236 | 64 | 28.33 | 36.02 | [307429](RL_EVALUATION.md#arm-additive-iter20-full300-20260920) |
| B: relaxed gate | 20 | 101 | 228 | 72 | 33.67 | 44.30 | [313209](RL_EVALUATION.md#arm-gate-b-iter20-results-20260921) |
| B: relaxed gate | 30 | 102 | 234 | 66 | 34.00 | 43.59 | [318934](RL_EVALUATION.md#arm-gate-b-iter30-40-results-20260924) |
| B: relaxed gate | 40 | 108 | 228 | 72 | 36.00 | 47.37 | [318934](RL_EVALUATION.md#arm-gate-b-iter30-40-results-20260924) |
| B: relaxed gate | 50 | 113 | 232 | 68 | 37.67 | 48.71 | [329908](RL_EVALUATION.md#arm-gate-b-iter50-60-results-20260926) |
| B: relaxed gate | 60 | 112 | 228 | 72 | 37.33 | 49.12 | [329908](RL_EVALUATION.md#arm-gate-b-iter50-60-results-20260926) |
| B: relaxed gate | 70 | 112 | 229 | 71 | 37.33 | 48.91 | [331778](RL_EVALUATION.md#arm-gate-b-iter70-results-20260927) |
| B: relaxed gate | 80 | 107 | 226 | 74 | 35.67 | 47.35 | [334894](RL_EVALUATION.md#arm-gate-b-iter80-results-20260927) |
| B: relaxed gate | 90 | 129 | 234 | 66 | 43.00 | 55.13 | [335729](RL_EVALUATION.md#arm-gate-b-iter90-results-20260928) |
| B: relaxed gate | 100 | 110 | 225 | 75 | 36.67 | 48.89 | [336893](RL_EVALUATION.md#arm-gate-b-iter100-results-20260929) |
| C: relaxed gate + action credit | 20 | 110 | 247 | 53 | 36.67 | 44.53 | [313211](RL_EVALUATION.md#arm-gate-c-iter20-results-20260921) |
| C: relaxed gate + action credit | 30 | 92 | 232 | 68 | 30.67 | 39.66 | [318935](RL_EVALUATION.md#arm-gate-c-iter30-40-results-20260924) |
| C: relaxed gate + action credit | 40 | 100 | 222 | 78 | 33.33 | 45.05 | [318935](RL_EVALUATION.md#arm-gate-c-iter30-40-results-20260924) |
| C: relaxed gate + action credit | 50 | 101 | 232 | 68 | 33.67 | 43.53 | [318935](RL_EVALUATION.md#arm-gate-c-iter50-60-results-20260925) |
| C: relaxed gate + action credit | 60 | 98 | 227 | 73 | 32.67 | 43.17 | [318935](RL_EVALUATION.md#arm-gate-c-iter50-60-results-20260925) |
| Additive: failure β = 1 | 10 | 82 | 243 | 57 | 27.33 | 33.74 | [318949 +329912](RL_EVALUATION.md#arm-failure-beta1-iter10-results-20260925) |
| Additive: failure β = 1 | 20 | 99 | 237 | 63 | 33.00 | 41.77 | [329912](RL_EVALUATION.md#arm-failure-beta1-iter20-results-20260925) |
| Additive: failure sampling40% | 10 | 87 | 233 | 67 | 29.00 | 37.34 | [329911](RL_EVALUATION.md#arm-failure-coverage-iter10-results-20260926) |
| Additive: failure sampling40% | 20 | 83 | 222 | 78 | 27.67 | 37.39 | [329911](RL_EVALUATION.md#arm-failure-coverage-iter20-results-20260926) |
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
| B: relaxed gate | 30 | 29 | 74 | 26 | 29.00 | 39.19 | [318934, slice of full300](RL_EVALUATION.md#arm-gate-b-iter30-40-results-20260924) |
| B: relaxed gate | 40 | 38 | 72 | 28 | 38.00 | 52.78 | [318934, slice of full300](RL_EVALUATION.md#arm-gate-b-iter30-40-results-20260924) |
| B: relaxed gate | 50 | 33 | 73 | 27 | 33.00 | 45.21 | [329908](RL_EVALUATION.md#arm-gate-b-iter50-60-results-20260926) |
| B: relaxed gate | 60 | 35 | 71 | 29 | 35.00 | 49.30 | [329908](RL_EVALUATION.md#arm-gate-b-iter50-60-results-20260926) |
| B: relaxed gate | 70 | 35 | 70 | 30 | 35.00 | 50.00 | [331778, slice of full300](RL_EVALUATION.md#arm-gate-b-iter70-results-20260927) |
| B: relaxed gate | 80 | 34 | 71 | 29 | 34.00 | 47.89 | [334894, slice of full300](RL_EVALUATION.md#arm-gate-b-iter80-results-20260927) |
| B: relaxed gate | 90 | 39 | 71 | 29 | 39.00 | 54.93 | [335729, slice of full300](RL_EVALUATION.md#arm-gate-b-iter90-results-20260928) |
| B: relaxed gate | 100 | 30 | 68 | 32 | 30.00 | 44.12 | [336893, slice of full300](RL_EVALUATION.md#arm-gate-b-iter100-results-20260929) |
| C: relaxed gate + action credit | 20 | 38 | 79 | 21 | 38.00 | 48.10 | [313211](RL_EVALUATION.md#arm-gate-c-iter20-results-20260921) |
| C: relaxed gate + action credit | 30 | 28 | 72 | 28 | 28.00 | 38.89 | [318935, slice of full300](RL_EVALUATION.md#arm-gate-c-iter30-40-results-20260924) |
| C: relaxed gate + action credit | 40 | 31 | 72 | 28 | 31.00 | 43.06 | [318935, slice of full300](RL_EVALUATION.md#arm-gate-c-iter30-40-results-20260924) |
| C: relaxed gate + action credit | 50 | 31 | 75 | 25 | 31.00 | 41.33 | [318935, slice of full300](RL_EVALUATION.md#arm-gate-c-iter50-60-results-20260925) |
| C: relaxed gate + action credit | 60 | 30 | 73 | 27 | 30.00 | 41.10 | [318935, slice of full300](RL_EVALUATION.md#arm-gate-c-iter50-60-results-20260925) |
| Additive: failure β = 1 | 10 | 27 | 76 | 24 | 27.00 | 35.53 | [318949 +329912, slice of full300](RL_EVALUATION.md#arm-failure-beta1-iter10-results-20260925) |
| Additive: failure β = 1 | 20 | 32 | 76 | 24 | 32.00 | 42.11 | [329912, slice of full300](RL_EVALUATION.md#arm-failure-beta1-iter20-results-20260925) |
| Additive: failure sampling40% | 10 | 32 | 76 | 24 | 32.00 | 42.11 | [329911, slice of full300](RL_EVALUATION.md#arm-failure-coverage-iter10-results-20260926) |
| Additive: failure sampling40% | 20 | 26 | 68 | 32 | 26.00 | 38.24 | [329911, slice of full300](RL_EVALUATION.md#arm-failure-coverage-iter20-results-20260926) |
| Outcome-only | 100 | 37 | 68 | 32 | 37.00 | 54.41 | [318933, slice of full300](RL_EVALUATION.md#baseline-iter100-results-20260924) |
| Additive bonus | 100 | 31 | 65 | 35 | 31.00 | 47.69 | [313669, slice of full300](RL_EVALUATION.md#arm-additive-iter100-results-20260921) |
| All-failure bonus | 100 | 26 | 66 | 34 | 26.00 | 39.39 | [316392, slice of full300](RL_EVALUATION.md#arm-allfailure-iter100-results-20260921) |
