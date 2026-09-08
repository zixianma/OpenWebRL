# Comparing the reference run's reward plot

The paper's Figure 2(a) uses **iteration** on the x-axis and **training reward (%)** on the y-axis. Section 5.1 defines an iteration as collection followed by policy updates. Its displayed curve spans the 90-iteration first stage; faint observations and a thicker trend are visible. [Paper](https://arxiv.org/pdf/2606.02031), [official figure](https://openwebrl.github.io/static/images/raw_success_rate_curves_sft_base.png).

## The chart to use

In [the active W&B run](https://wandb.ai/zixianma/openwebrl/runs/qcq7i4ug), select:

| Setting | Value |
| --- | --- |
| Primary y metric | `train/reward` |
| X metric | `train/reward_iteration` |
| Source | `rollout/raw_reward_mean` |
| X conversion | One-based collection number; subtract 1 to compare the paper axis |
| Frequency | One distinct observation per completed collection |
| Smoothing | None in logged values; retain raw observations for auditing |
| Selected-subset diagnostic | Original `rollout/raw_reward` |

Use `train/reward` as the sole primary reward series: 0.3874 corresponds to 38.74%, and collection 1 corresponds to paper iteration 0. The duplicate `paper/*` series is retired; its existing observations remain historical. This avoids changing the units or axes of previously recorded `train/reward` values. The collector mean is the closest documented metric in the release. **The exact field and smoothing parameters used to export the paper figure remain unverified.** Do not describe this chart as an exact reconstruction of the authors' plot.

The pinned official repository revision inspected is `9a120949aca3e58a2628f4b4e6edd0474d984873`. Its [collector](https://github.com/OpenWebRL/OpenWebRL/blob/9a120949aca3e58a2628f4b4e6edd0474d984873/slime/ray/rollout.py) computes raw reward mean over accepted flattened samples. Its [trainer](https://github.com/OpenWebRL/OpenWebRL/blob/9a120949aca3e58a2628f4b4e6edd0474d984873/slime/backends/megatron_utils/data.py) also logs the selected batch's mean raw reward. Both are iteration-level summaries. No figure-export script or curve CSV was found in that revision's file tree. The separate selected-batch chart preserves the distinction instead of silently assuming the two means are identical.

## Why two observations, not seven

The first collection yielded 2,034 turn samples. Seven full 256-turn minibatches fit in each epoch; two epochs made 14 optimizer updates. Those updates reused rewards assigned during that same collection. A per-minibatch mean could be logged, but would describe which stored samples each update consumed. It would not add another collection to the x-axis.

Two completed collections therefore give x=0 and x=1 on the comparison chart. Seven minibatches are not seven online collection iterations. The second collection has its own sample count and can have a different number of optimizer batches.

| Zero-based iteration | Collected-turn reward (%) |
| --- | ---: |
| 0 | 38.7413962635 |
| 1 | 33.2581227437 |

These are raw observations, not smoothed trend values. A two-point change is insufficient to establish convergence, divergence, or successful reproduction. Browser access, task sampling, trajectory lengths, and filtering can change the reward independently of policy quality. Inspect completed-task success and invalidity alongside it, and compare held-out evaluation with matching judge and validity conventions before claiming benchmark reproduction.

## Live behavior and provenance

The duplicate paper writer and native `paper/*` emission are disabled. Existing `paper_reward_sync.jsonl` is retained as provenance. `scripts/sync_training_rewards.py` supplies the current run's `train/reward` alias; updated repository launches emit it directly. These logging changes do not change the training recipe.

See [METRICS.md](METRICS.md) for reward formulas, subset weighting, task denominators, update counts, and known axis limitations. In particular, the current optimizer `train/step` label can jump when minibatch counts change; the paper comparison uses collection identity and is unaffected by that label issue.
