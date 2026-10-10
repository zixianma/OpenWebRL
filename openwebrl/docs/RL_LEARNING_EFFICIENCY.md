# Faster reward improvement: research review

Reviewed 2026-10-10. This is a literature review and proposed experiment order; the reported paper gains have not been reproduced in OpenWebRL.

**The highest-upside research direction is to teach corrections at consequential decisions, supported by reliable comparisons between alternative actions.** Verified-success replay is the simpler near-term RL intervention. Adaptive task selection has supporting GUI evidence, but its benefit over our existing outcome-variance filter remains unmeasured. For wall-clock improvement, repeated context processing during optimization deserves attention alongside browser throughput.

The target should be **held-out task success per total training cost**. Reward on an adaptively sampled training batch can rise because the task distribution became easier. Better final accuracy at a fixed number of steps or rollouts is also weaker evidence than reaching a fixed accuracy sooner at matched total cost.

## Which ideas have the strongest evidence?

Paper results below are within-paper comparisons, not a common benchmark or a prediction of our gains. Percentages are the paper's task-success/accuracy metrics; percentage-point deltas are absolute differences.

| Direction and source | Controlled result | What it establishes; important limit | Assessment for OpenWebRL |
| --- | --- | --- | --- |
| **Localized hindsight correction.** [HinT-SD, May 2026](https://arxiv.org/html/2605.17873v1), Table 1 | Qwen3-4B BFCL Avg@4: GRPO **31.56%**, HinT-SD **41.88%**; AppWorld **7.49%**, **18.46%** | Better learning from matched rollout budgets. Main method uses a self/EMA teacher. Tool/API environments; total compute is not matched. Its **2.26×** step-time improvement is against dense OpenClaw-RL feedback, not GRPO. | Highest potential for extracting useful corrections from failed browser episodes. Requires evidence that the feedback identifies an actionable mistake. |
| **Several candidate actions per visited state, scored by a critic.** [Android Coach, April 2026](https://arxiv.org/html/2604.07277v1), Table 2 and Figure 4 | UI-TARS-7B AndroidLab: GRPO **34.8%**, Coach **39.4%**; AndroidWorld **38.2%**, **41.1%**. Reports **1.4×** efficiency at matched success | Direct GUI evidence under matched online training time. Additional candidate actions are scored without execution. Substantial critic/PRM preparation is not established as included in that timing. | Strong motivation for our action-comparison study. A reliable critic could amortize expensive browser interactions, but adds scoring and optimizer work. |
| **Adaptive positive replay.** [MobileRL, September 2025](https://arxiv.org/html/2509.18119v1), Figure 3 | AndroidWorld: full method **71.1%**; without adaptive replay **63.6%**; without failure-curriculum filtering **64.8%** | On-policy reward curves exclude replayed successes. Component evidence at a fixed step budget, not a measured time-to-target speedup. Replay includes selection/pruning, so this does not isolate naive replay alone. | Most practical first RL ablation: reuse verified successes when current groups fail, with explicit off-policy handling. |
| **Task selection near the current learning frontier.** [ScaleCUA, July 2026](https://arxiv.org/html/2607.11185v1), Table 4 | OSWorld: full method **68.7%**; without frontier sampling **63.7%** | Held-out benchmark evidence beyond a changed training-reward curve. EMA task success guides selection, with uniform exploration. Initial profiling costs rollouts; no isolated sampling speedup in total cost. | Predict informative groups before collection. Retain exploration and refresh stale difficulty estimates. |

Our existing filter removes constant-reward groups after collection. Predictive selection can avoid some of that expenditure, but does not automatically increase learning per accepted update. Replay and corrective supervision address a different limitation: the absence of a useful positive contrast or localized correction in failed experience.

## Recommended experiment order

**Scientific priority: outcome-grounded action correction.** Use the [branching study](ARM_FORMULATIONS.md#branching-experiment-map) to identify decisions where alternatives have distinguishable valid continuation outcomes. Generate concise comparisons grounded in observed consequences, then train the student from the original pre-action context. Future observations and outcomes may inform supervision, but must not enter the student's deployment input.

The smallest informative comparison extends [action-comparison SFT](ARM_SFT.md#arm-comparison-sft-20261007):

| Arm | Added supervision | Question answered |
| --- | --- | --- |
| A | Existing action-filtered SFT | Reference |
| B | Verified corrected action at selected states | Does better action supervision help? |
| C | The same states and corrected actions as B, plus a grounded comparison explanation | Does comparison reasoning add value beyond the corrected action? |

Keep task/state splits fixed, preserve an independent continuation holdout, and abstain on uncertain action rankings. Compare B/C with matched examples and report both token exposure and total compute; longer explanations create extra training work. A later critic-guided RL experiment should first demonstrate accurate ranking on fresh continuation outcomes.

**Practical RL priority: test verified-success replay, then task selection as a separate factor.** Preserve the terminal success definition and track original behavior-policy probabilities for reused data; merely limiting replay age is not a correction for off-policy bias. Start new scientific variants from the agreed iteration-zero initialization with fresh optimizer/scheduler/cursor. Keep the ongoing outcome-only continuations unchanged.

**Systems priority: profile repeated visual/text context processing.** First validate packing or batching changes on identical saved batches, preserving loss masks, update membership, global batch and PPO epochs. Treat changes to the model's visible history as a separate intervention requiring quality validation.

<details>
<summary>Timing evidence and why rollout savings are not total speedups</summary>

[SDPO, January 2026](https://arxiv.org/html/2601.20802v1), Table 3, supplies actual wall-clock learning evidence: Olmo3-7B reaches a five-hour GRPO chemistry result in 30 minutes. The closer tool-use comparison is smaller: Qwen3-8B reaches **68.0%** versus GRPO **64.9%** after one hour. These are best achieved average accuracies within the time budget, with hyperparameters selected using five-hour accuracy. The tool task is single-call ToolAlpaca, not long-horizon browsing; the 10× chemistry result is not a general agent-training multiplier.

ScaleCUA's **2.83×** timing claim belongs to visual-context segmentation, not frontier sampling: reported training-step time falls **750→265 seconds**, including actor update **485→154 seconds** and reference scoring **241→88 seconds**. This changes how trajectory contexts are processed; it is not evidence that our full collection/evaluation pipeline becomes 2.83× faster. [Source, §4.3](https://arxiv.org/html/2607.11185v1#S4.SS3)

Our completed extension-cycle aggregates are below; [machine-readable timing evidence](rl_results/learning_efficiency_profile_20261010.json). These windows exclude milestone evaluations, startup/restarts, storage-review waits and failed partial collections. Native timer boundaries and iteration windows differ, so this is a bottleneck diagnostic, not a causal comparison between recipes.

| Run | Completed iteration window | Cycles | Mean collection, min | Mean training, min | Mean cycle, min | Training share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Batch56 outcome-only | 21–34 | 14 | 20.06 | 25.41 | 46.09 | 55.15% |
| Uniform G8 outcome-only | 11–21 | 11 | 24.79 | 45.38 | 70.85 | 64.05% |

Holding all other work fixed, halving only collection time implies approximately **1.28×** and **1.21×** more completed cycles per hour, respectively: `cycle / (cycle − 0.5 × collection)`. This illustrative calculation is not an observed speedup or a prediction of reward growth.

</details>

<details>
<summary>Preparation costs, related papers, and transfer risks</summary>

- **Android Coach:** uses 20,000 action labels, GPT-4o annotations/filtering, separate critic preparation and PRM training. Critic training without suitable pretraining is unstable. In its action-count ablation, four candidates cost 1.62× the one-action training time; eight cost 2.18× with little further improvement. Its demonstration-based labels can penalize valid alternative actions. Our independent continuation outcomes would be an important validation layer. [Method and ablations](https://arxiv.org/html/2604.07277v1)
- **PivotRL, March 2026:** training at informative decision points reaches comparable SWE-Bench end-to-end-RL accuracy with roughly four times fewer rollout turns and 5.5 times less cumulative rollout time. This excludes some preparation/training costs and depends on programmatic local action verification. It supports investigating selected decision states, but that verifier is not directly available on live websites. [Paper](https://arxiv.org/html/2603.21383v1)
- **Tree-GRPO, ICLR 2026:** at approximately four full-trajectory equivalents, Qwen2.5-3B multi-hop QA rises from 31.8% with chain GRPO to 36.8% with trees. Experiments use search tools; browser replay/reset costs and failures could erase shared-prefix savings. [Paper](https://arxiv.org/html/2509.21240v2)
- **DPS, March 2026:** predictive sampling achieves 63.13% average math accuracy with 287k rollouts/39 hours versus dynamic sampling's 62.42% with 1,147k/73 hours. Uniform sampling takes 30 hours for 59.31%. Thus its efficiency advantage depends on the comparator; it is not simply faster than uniform sampling. [Table 1](https://arxiv.org/html/2603.10887v1)

</details>

<details>
<summary>Evaluation and accounting needed to establish faster learning</summary>

Use the same initialization, fixed held-out task cohort and evaluation protocol. Plot task success against cumulative allocated GPU-hours and browser/judge/teacher cost, and report time to a predeclared success threshold. Charge critic preparation, rejected rollouts, replay failures, teacher generation/scoring and all training attempts. Keep overall and valid-only success separate, with counts and invalid rates next to percentages.

Report fresh on-policy training reward before filtering and separately from replayed data. Compare matched tasks when changing the sampling distribution. Use paired task-level uncertainty for held-out differences; repeated evaluation draws do not replace independent training seeds. Distinguish accuracy per rollout, accuracy per optimizer update and accuracy per wall-clock hour.

This review proposes tests, not an allocation request. Existing compute/API caps and the paused adaptive run remain unchanged.

</details>
