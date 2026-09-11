# RL metrics: reward, optimization, and diagnostics

Definitions and interpretation of training/W&B metrics, comparison with the paper reward curve, and gradient-spike diagnostics. The metric reference comes first; the later sections provide measurement and debugging context.

## Contents

- [OpenWebRL training and W&B metric reference](#metrics)
- [Comparing the reference run's reward plot](#paper-reward-comparison)
- [Gradient spikes and training diagnostics](#gradient-diagnostics)

---

<!-- document:METRICS.md:start -->
<a id="metrics"></a>
## OpenWebRL training and W&B metric reference

_Source record: `METRICS.md`. Dated entries retain their historical context._


Verified on 2026-09-07 against the running 4B reference baseline (`qcq7i4ug`), its isolated source snapshot, and the repository logging code. This document covers the metrics emitted by that baseline, plus clearly marked optional metric families. It does not imply that every metric exists in every run.

<a id="metrics--completed-rollout-archive-metrics-enabled-from-collection-24-2026-09-10"></a>
#### Completed-rollout archive metrics (enabled from collection 24, 2026-09-10)

These scalars are emitted with collection metrics when the archive is enabled.
They describe preservation of completed groups, including RL rejections; they
do not change the reward or training population. See [ROLLOUT_ARCHIVE.md](RL_RUNTIME.md#rollout-archive).

| Metric | Calculation |
| --- | --- |
| `rollout/archive/groups` | Number of completed groups in the successfully finished archive. |
| `rollout/archive/trajectories` | Sum of attempt counts across those groups. |
| `rollout/archive/unique_images` | Distinct raw-image SHA-256 hashes referenced by this collection, including images already present in the shared archive. |
| `rollout/archive/seconds` | Wall time spent writing the archive and completion manifest. |
| `rollout/archive/errors` | Zero on successful archival, one on a caught archive failure; the other archive scalars are omitted on failure. |

The first archive logged 88 groups, 440 trajectories, 2,290 images, 31.23 seconds,
and zero errors, verified against W&B and local files. A collection interrupted
before archival has no completed archive and no archival success metric.

<a id="metrics--1-why-one-reward-measurement-can-accompany-14-optimizer-updates"></a>
### 1. Why one reward measurement can accompany 14 optimizer updates

A **prompt group** is one sampled task with up to five browser attempts. A **trajectory** is one attempt. A **turn sample** is one assistant action within an attempt. A **rollout iteration** collects a training batch, trains on that batch, and then updates the inference model. A **PPO epoch** is one shuffled training pass over the selected turn samples. An **optimizer update** processes a global batch of turn samples, using gradient accumulation where needed.

The first reference collection had:

| Quantity | Observed value |
| --- | ---: |
| Accepted prompt groups | 48 |
| Requested attempts per group | 5 |
| Accepted trajectories after filtering | 228 |
| Accepted turn samples | 2,034 |
| Global batch size in turn samples | 256 |
| Full optimizer batches per PPO epoch | floor(2,034 / 256) = 7 |
| PPO epochs per collection | 2 |
| Optimizer updates | 7 × 2 = 14 |

Each epoch deterministically shuffles the samples and keeps 1,792 turns (7 × 256), leaving 242 out of that epoch. The selection seed changes between epochs, so the omitted turns need not be the same. In this two-GPU, tensor-parallel run there is one data-parallel shard. With multiple data-parallel shards, the current selection logic first balances their usable sample counts.

The judge scores each trajectory at its end, and that reward is copied to its turn samples. The collection's mean reward is measured **before training on that collection**. The 14 updates reuse those rewards and the stored old-policy log probabilities. Loss, clipping, gradients, and entropy can change during those updates; freshly measured browser reward cannot change until the updated policy collects another batch.

Thus iteration 1 supplies one aggregate training-reward point, summarizing many individual trajectory outcomes. Iteration 2 supplies the next point. Repeating the first mean at every optimizer update would display reuse of the same observation, not 14 measurements of policy quality. The first point uses the SFT policy; the second uses the policy after the first collection's updates. Subsequent points also use different sampled tasks, so changes combine policy changes and task difficulty.

<a id="metrics--collection-mean-versus-minibatch-mean"></a>
#### Collection mean versus minibatch mean

There **are seven optimizer minibatches per epoch**, and each has a well-defined mean stored reward:

```text
minibatch_reward_b = sum(raw_reward_i for i in selected minibatch b) / 256
```

One could log seven such values per epoch, or 14 across both epochs. They can differ because the minibatches contain different turns. These would be useful diagnostics of the data each update consumes. They would still describe outcomes collected by the old policy, not outcomes produced by the weights after that update.

| Statistic | Number of entries for the first collection | Currently logged? |
| --- | ---: | --- |
| Whole-collection mean reward | 1 distinct measurement | Yes: `train/reward`; averages all 2,034 collected turns. |
| Per-minibatch mean stored reward | 7 per epoch, 14 over two epochs | No separate metric currently. Do not interpret `train/reward` as this quantity. |
| Fresh browser reward after each optimizer update | Would require extra browser attempts after each update | No; the baseline collects again after both PPO epochs. |

The current first-epoch summary `rollout/raw_reward` averages the selected 1,792 turns together; it also is not a seven-point minibatch curve. Averaging the seven equally sized minibatch means would recover that selected-epoch mean, which can differ from the full 2,034-turn mean because of trimming.

Source: [`actor.py`](../../slime/backends/megatron_utils/actor.py), epoch selection and `train_actor`; [`data.py`](../../slime/backends/megatron_utils/data.py), `get_data_iterator`.

<a id="metrics--2-recommended-charts-axes-and-cadence"></a>
### 2. Recommended charts, axes, and cadence

For the paper comparison, use `train/reward` versus `train/reward_iteration` as the sole primary reward series: multiply y by 100 and subtract 1 from x when comparing displayed paper coordinates. The duplicate `paper/*` aliases were retired on 2026-09-07; their old observations remain historical. See [PAPER_REWARD_COMPARISON.md](RL_METRICS.md#paper-reward-comparison). Exact paper figure aggregation/smoothing is unverified.


| Chart / family | X-axis | Emission cadence and interpretation |
| --- | --- | --- |
| `train/reward`, reward components, and task reward/success aliases below | `train/reward_iteration` | Once per completed browser collection, before PPO updates. One-based collection number. |
| Other `train/*` optimizer metrics | `train/step` | Once per optimizer batch, zero-based step label. |
| `rollout/*` collection diagnostics | `rollout/iteration` | Once per completed collection; trainer-side batch summaries also use this namespace. |
| `perf/*` | `rollout/iteration` | At the end of the measured phase; collection and trainer timing records arrive separately. |
| `eval/*` | `eval/iteration` | When evaluation runs; the reference launcher requests evaluation every 10 rollout iterations. Initial evaluation may be reused explicitly. |
| Optional `multi_turn/*`, `passrate/*` | `rollout/iteration` | Only when their logging flags are enabled. Both are disabled in the audited baseline. |
| W&B system metrics | Wall-clock/runtime or W&B system axis | Periodic SDK hardware/process sampling, independent of optimizer updates. |

Start with `train/reward`, `train/task_success_rate`, `train/task_invalid_rate`, `train/pg_loss`, `train/pg_clipfrac`, `train/ppo_kl`, `train/grad_norm`, and `perf/rollout_time`. For held-out performance use `eval/<dataset>/task/success_rate_all_completed` together with `eval/<dataset>/task/invalid_rate`.

<a id="metrics--axis-details-and-known-limitations"></a>
#### Axis details and known limitations

- `train/reward_iteration = rollout_id + 1`. Its separate axis prevents optimizer-step metadata from giving reward charts the wrong x-coordinate.
- `rollout/iteration` and `eval/iteration` are intended to equal `rollout_id + 1`. They identify the associated iteration, not the number of evaluation calls or necessarily the number of completed updates. Pretraining evaluation and iteration-1 collection can both appear at 1.
- `rollout/step` and `eval/step` are legacy zero-based rollout IDs in this run. If `wandb_always_use_train_step=True`, their formula instead is `rollout_id × rollout_batch_size × n_samples_per_prompt // global_batch_size`; that option is false here.
- Earlier source snapshots emitted a zero-based `rollout/iteration` through the generic trainer summary while the collector emitted a one-based value. The recovery launched at 18:18 PDT and the repository contain the explicit one-based correction. Historical rows may retain the old coordinates; `train/reward_iteration` is the canonical reward axis.
- The current optimizer label is `rollout_id × ppo_epochs × K + ppo_epoch_id × K + step_id`, where `K` is the current collection's number of optimizer batches per epoch. If `K` changes with trajectory lengths, labels can jump or overlap between collections. **It is not a robust cumulative update counter across variable-size collections.** Local `[TrainMetrics]` records retain rollout and epoch IDs for auditing. The first collection's labels 0–13 are complete.
- W&B `_step` is a shared event index. Events from collection, training, evaluation, and auxiliary writers interleave, so consecutive training records need not have consecutive `_step` values. `_timestamp` is event time; `_runtime` is SDK runtime metadata, not an optimizer count.
- W&B axis tick labels are display choices. A tick such as `200m` on a numeric axis means 0.2 (milli), not 200 million steps. Use the integer collection axis to avoid fractional-step labels being mistaken for update counts.

Source: [`wandb_utils.py`](../../slime/utils/wandb_utils.py), [`training_reward_metrics.py`](../../slime/utils/training_reward_metrics.py), [`model.py`](../../slime/backends/megatron_utils/model.py), `train`; [`metric_utils.py`](../../slime/utils/metric_utils.py), `compute_rollout_step`.

<a id="metrics--3-reward-construction-and-weighting"></a>
### 3. Reward construction and weighting

For a terminal trajectory reward, the running browser recipe computes:

```text
F = terminal response format check (0 or 1 in the active browser_env format)
J = judge success verdict (0 or 1)
R = -1                 if terminate_reason == "format_error_failed"
    0                  otherwise if F == 0
    1                  otherwise if J == 1
    0                  otherwise
```

This executed rule takes precedence over older comments describing a weighted sum of format and judge rewards. The active `browser_env` format requires the expected closing thinking tag and at least one successfully parsed tool call. A different, inactive `slime` format additionally checks thinking/action/tool-call structure and ordering; its implementation is binary despite an older docstring describing partial 0.2 credits. In turn-sample mode the terminal response is checked, then its reward metadata is copied to every turn; `format_reward_mean` is therefore not an independent format audit of every action.

The baseline judge parser checks `NOT SUCCESS` before `SUCCESS`; unexpected text maps to zero in the running reference snapshot. Judge timeouts or transport failures can mark an attempt invalid/removed. The repository's experimental judge options can differ; consult the run's archived source and configuration when comparing runs.

Let trajectory `t` have reward `R_t` and `L_t` retained turns:

```text
Turn-weighted mean reward = sum_t(L_t × R_t) / sum_t(L_t)
Task-weighted mean reward = sum_valid_t(R_t) / number_of_valid_trajectories
Task success rate         = count(R_t == 1) / specified_trajectory_denominator
```

Negative format-failure rewards make mean reward different from success rate. Longer trajectories have greater weight in the turn mean. Accepted groups are selected by the dynamic filter, so accepted reward is also selection-biased; the completed-attempt metrics below include rejected groups but still are training diagnostics, not held-out benchmark scores.

Source: [`reward_browser.py`](../reward_browser.py), `reward_func` and `_score_single_sample`; [`trajectory_metrics.py`](../../slime/utils/trajectory_metrics.py).

<a id="metrics--training-reward-aliases"></a>
#### Training reward aliases

Every alias is an exact copy of the source scalar; it does not rescore trajectories. Missing source fields are omitted, not replaced with zero. All use `train/reward_iteration`.

| W&B metric | Source metric | Calculation / population |
| --- | --- | --- |
| `train/reward` | `rollout/raw_reward_mean` | Mean raw reward over collected accepted turn samples with a reward. |
| `train/reward_std` | `rollout/raw_reward_std` | Population standard deviation over those turn rewards; denominator N. |
| `train/judge_reward` | `rollout/judge_reward_mean` | Mean copied judge component over turns with that metadata. |
| `train/format_reward` | `rollout/format_reward_mean` | Mean copied terminal format component over turns with that metadata. |
| `train/task_reward_accepted` | `rollout/task/accepted/reward_mean` | Mean terminal reward across valid accepted trajectories, each counted once. |
| `train/task_reward_completed` | `rollout/task/completed/reward_mean` | Mean terminal reward across valid completed trajectories before dynamic filtering. |
| `train/task_success_rate` | `rollout/task/completed/success_rate_all_completed` | Successful trajectories / all completed trajectories, including invalid ones in the denominator. |
| `train/task_success_rate_valid` | `rollout/task/completed/success_rate_valid` | Successful trajectories / valid completed trajectories. |
| `train/task_invalid_rate` | `rollout/task/completed/invalid_rate` | Invalid trajectories / all completed trajectories. |

For the first collection: `train/reward = 0.3874139626`; completed-attempt success is `202 / 525 = 0.3847619048`; valid-only success is `202 / 484 = 0.4173553719`; invalid rate is `41 / 525 = 0.0780952381`. These similar-looking rewards/rates have different meanings and denominators.

The raw reward mean does not exclude accepted turns marked `remove_sample`. The
preserved reference filter can admit such trajectories when their terminal raw
reward exists. Their training loss masks are zeroed, while their rewards still
enter group normalization and the collection reward mean. In collection 30,
four of 1,686 accepted turns had this flag and raw reward zero; the archive's
validity-filtered trajectory reward was null. See [ROLLOUT_ARCHIVE.md](RL_RUNTIME.md#rollout-archive)
for the audited example and the distinction between raw reward and valid outcome.

<a id="metrics--4-per-update-optimization-metrics-train"></a>
### 4. Per-update optimization metrics (`train/*`)

These are computed over the selected optimizer batch, not fresh browser attempts. In the audited run `calculate_per_token_loss=False`, so the reducer first averages valid response tokens **within each turn sample**, then averages those sample means across the global batch. Prompt, observation, padding, and other masked tokens do not contribute.

For token statistic `x_it`, mask `m_it`, and N turn samples, define:

```text
M(x) = (1/N) × sum_i [sum_t(m_it × x_it) / max(sum_t(m_it), 1)]
ell_old = stored rollout token log probability
ell_new = current actor token log probability
rho = exp(ell_new - ell_old)
A = fixed group-normalized trajectory reward, broadcast to response tokens
u = -rho × A
v = -clip(rho, 1-eps_low, 1+eps_high) × A
```

The reference settings are `eps_low=0.20`, `eps_high=0.28`, `entropy_coef=0`, `use_kl_loss=False`, `kl_coef=0`, `use_tis=False`, and `normalize_advantages=False`.

| Metric | Exact meaning / calculation |
| --- | --- |
| `train/pg_loss` | `M(max(u, v))`, the clipped policy-gradient loss. Lower values do not directly mean higher browser success. |
| `train/loss` | `pg_loss - entropy_coef × entropy_loss + kl_loss_coef × kl_loss` when the KL term is enabled. Equals `pg_loss` in this baseline. |
| `train/abs_advantage` | `M(abs(A))`. Measures magnitude of the fixed normalized learning signal; **not reward or success rate**. |
| `train/ppo_kl` | `M(ell_old - ell_new)`, a sampled signed log-probability difference. It is not an exact nonnegative distributional KL and is not KL to the SFT reference model. |
| `train/pg_clipfrac` | `M(1[v > u])`: fraction for which the clipped loss is strictly larger and selected. It is not simply every token with rho outside the clipping interval. |
| `train/entropy_loss` | `M(-sum_vocab p × log(p))` for the current actor; positive predictive entropy in nats. It is logged even with entropy regularization coefficient zero. |
| `train/grad_norm` | Global gradient L2 norm returned by Megatron before clipping. The configured clipping threshold is 1; a logged norm above 1 is compatible with clipping working. |
| `train/raw_loss_samples` | Number of turn samples contributing to the aggregated update statistics; 256 here. With per-token loss enabled, `train/raw_loss_tokens` replaces this count. |
| `train/train_rollout_logprob_abs_diff` | `M(abs(ell_old - ell_rollout))`. With `use_rollout_logprobs=True`, both operands are the stored rollout log probabilities, so zero is expected. It does **not** prove current actor and inference engine agreement. |
| `train/lr-pg_<j>` | Scheduler learning rate for optimizer parameter group j at logging time, after the scheduler step. Both observed groups use 1e-6 in this constant-LR run. |
| `train/ppo_epoch` | Zero-based epoch index: 0 or 1 here. Local epoch-start messages use human-readable 1 or 2. |
| `train/step` | Optimizer step label; see the variable-batch-count limitation in section 2. |

GRPO reward normalization is done **per prompt across trajectories**, before expanding back to turns:

```text
mu_g = mean of available trajectory rewards in prompt group g
s_g  = sample standard deviation (PyTorch correction=1)
A_t  = (R_t - mu_g) / (s_g + 1e-6)
```

Every retained turn of trajectory t receives the same normalized reward. With zero KL reward penalty, GRPO returns and advantages broadcast that value across response tokens. A mean of these values across turns or a shuffled subset need not be zero, because trajectory lengths differ and the subset need not preserve prompt-group balance. If normalization flags change, the formula changes accordingly.

Source: [`loss.py`](../../slime/backends/megatron_utils/loss.py), `policy_loss_function` and `compute_advantages_and_returns`; [`ppo_utils.py`](../../slime/utils/ppo_utils.py), `compute_policy_loss`; [`cp_utils.py`](../../slime/backends/megatron_utils/cp_utils.py), `get_sum_of_sample_mean`; [`model.py`](../../slime/backends/megatron_utils/model.py), `train_one_step`; [`rollout.py`](../../slime/ray/rollout.py), `_post_process_rewards`.

<a id="metrics--5-collected-sample-diagnostics-rollout"></a>
### 5. Collected-sample diagnostics (`rollout/*`)

Unless stated otherwise, the population is the accepted, flattened turn samples returned by collection, before epoch-specific trimming. A ratio is a fraction from 0 to 1, not a percentage.

| Metric or family | Calculation |
| --- | --- |
| `rollout/raw_reward_mean`, `rollout/raw_reward_std`, `rollout/raw_reward_min`, `rollout/raw_reward_max` | Mean, population standard deviation, minimum, maximum of available raw turn rewards. |
| `rollout/combined_reward_mean`, `rollout/combined_reward_std` | Mean and population standard deviation of `metadata.reward.combined`; normally the same as raw reward in this recipe. |
| `rollout/format_reward_mean`, `rollout/format_reward_std` | Mean and population standard deviation of copied terminal format components. |
| `rollout/judge_reward_mean`, `rollout/judge_reward_std` | Mean and population standard deviation of copied terminal judge components. |
| `rollout/response_len/mean`, `rollout/response_len/median`, `rollout/response_len/min`, `rollout/response_len/max` | Statistics of `Sample.effective_response_length`: sum of its loss mask if present, otherwise response length. |
| `rollout/effective_batch_size` | Count of samples with `remove_sample=False`; counts turns, not prompt groups, trajectories, or optimizer batches. |
| `rollout/effective_batch_ratio` | Effective sample count / total collected sample count. |
| `rollout/remove_sample_ratio` | Fraction with `remove_sample=True`. |
| `rollout/truncated_ratio`, `rollout/aborted_ratio`, `rollout/failed_ratio` | Fractions whose sample status equals TRUNCATED, ABORTED, or FAILED, respectively. Status ratios are not judge success/failure rates. |
| `rollout/timeout_sample_ratio` | Fraction whose lowercased termination reason contains `rollout_task_timeout`, `sglang /generate timed out`, `sandbox exec timed out`, or `not healthy within`. |
| `rollout/sandbox_error_ratio` | Fraction whose reason contains `env_step_error`, `env_server /reset error`, or `env_server /step error`. |
| `rollout/judge_timeout_ratio` | Fraction whose top-level sample metadata has truthy `judge_timeout`. This flag is not the same field as nested `metadata.reward.judge_timeout`; propagation and filtering can make this undercount trajectory-level invalidity. |
| `rollout/aborted_or_timeout_ratio` | Fraction that is ABORTED or matches a timeout reason, counting the union once. |
| `rollout/repetition_frac` | Fraction with response text longer than 10,000 characters whose final 10,000 characters compress by more than 10× using zlib level 9. A heuristic, not a semantic repetition judgment. |
| `rollout/zero_std/count_<reward>` | Count of accepted prompt groups whose flattened raw rewards are all identical, named by reward rounded to one decimal, or `None`. GRPO-family diagnostic; absent if no such group exists. |
| `rollout/error_cat/<category>` | Fraction in a configured reward category; only with `log_reward_category`. |
| `rollout/replayed_first_batch` | Recovery marker: 1 means the collection was loaded from the previously saved first batch. It is not a performance score. |

Some ratios overlap; they must not be summed as disjoint failure categories. Accepted-only error ratios can be zero even when many attempted tasks failed before filtering. Prefer completed-trajectory invalidity for infrastructure health.

<a id="metrics--trainer-side-summaries-in-the-same-namespace"></a>
#### Trainer-side summaries in the same namespace

The trainer logs these over its **selected first-epoch subset**. They can differ from the collector's full-batch metrics. The generic logger reduces means across data-parallel ranks and uses a masked mean within samples for token-valued fields.

| Metric | Calculation |
| --- | --- |
| `rollout/raw_reward` | Arithmetic mean of raw rewards in the selected trainer subset; no `_mean` suffix. First collection: 0.3928571429 versus full-batch 0.3874139626. |
| `rollout/rewards` | Arithmetic mean of group-normalized rewards in that subset; this can be negative. |
| `rollout/advantages`, `rollout/returns` | Masked response-token mean within each turn, then sample/rank mean. Equal to normalized rewards up to numerical effects under this GRPO configuration. |
| `rollout/rollout_log_probs` | Masked mean of stored rollout token log probabilities. |
| `rollout/response_lengths`, `rollout/total_lengths` | Mean response token count and mean total prompt-plus-response token count in the selected subset. |
| `rollout/truncated` | Mean numeric truncation indicator in that subset. |
| `rollout/log_probs`, `rollout/ref_log_probs`, `rollout/entropy`, `rollout/values` | Corresponding masked token/sample means when those tensors are computed. Not all are computed in this actor-only baseline. |

Source: [`rollout.py`](../../slime/ray/rollout.py), `compute_metrics_from_samples`; [`data.py`](../../slime/backends/megatron_utils/data.py), `log_rollout_data`; [`types.py`](../../slime/utils/types.py), `effective_response_length`; [`metric_utils.py`](../../slime/utils/metric_utils.py).

<a id="metrics--6-trajectory-sampling-and-filtering-metrics"></a>
### 6. Trajectory, sampling, and filtering metrics

The following table applies under both `rollout/task/completed/` and `rollout/task/accepted/`.

- **completed**: all attempts in groups processed before collection stopped, including groups rejected by dynamic sampling. Pending attempts cancelled after the accepted batch fills are excluded. This is neither the whole dataset nor all submitted tasks.
- **accepted**: attempts retained in the final accepted groups, after group/attempt filtering.
- A trajectory is **valid** for these statistics when it is nonempty, no turn has `remove_sample=True` or ABORTED status, and its terminal raw reward exists and is finite. The terminal sample is selected by greatest `turn_index`. Valid does not mean successful; a finite zero or -1 reward is valid.

| Suffix under either prefix | Calculation |
| --- | --- |
| `trajectories` | Total number of trajectory objects in that population, including invalid ones. |
| `valid_trajectories` | Number meeting the validity rule above. |
| `invalid_trajectories` | Total minus valid. |
| `successes` | Valid terminal rewards exactly equal to 1. |
| `invalid_rate` | Invalid / total; omitted if total is zero. |
| `success_rate_all_completed` | Successes / total; omitted if total is zero. The suffix is also used under `accepted`, where its denominator is accepted trajectories. |
| `success_rate_valid` | Successes / valid; omitted if no valid trajectories. |
| `reward_mean` | Mean terminal raw reward among valid trajectories; omitted if none. |
| `turns_mean`, `turns_max` | Mean and maximum number of turn samples per trajectory across the whole population, including invalid attempts. |

The collection metrics are:

| Metric | Calculation |
| --- | --- |
| `rollout/sampling/completed_groups` | Number of processed completed prompt groups before collection ends. |
| `rollout/sampling/accepted_groups` | Number retained; target is 48. |
| `rollout/sampling/acceptance_rate` | Accepted groups / completed groups. |
| `rollout/sampling/completed_groups_per_accepted_group` | Completed groups / accepted groups; a measure of sampling overhead. |
| `rollout/dynamic_filter/drop_<reason>` | Number of rejected groups for that reason, not number of turns or individual attempts. Observed reasons: `all_none_reward_in_group`, `insufficient_nonempty_rewards_1`, `zero_std_0.0`, `zero_std_1.0`. Other reasons can occur, such as `single_valid_reward_in_group` or `empty_group_after_filter`. |

The reference dynamic filter removes attempts with missing terminal reward, requires at least two remaining attempts, and rejects groups whose reward sample standard deviation is at most 1e-6. Therefore 48 × 5 is the requested accepted-group attempt capacity, not a guarantee of exactly 240 valid trajectories; 228 remained in the first batch. The worktree's experimental filtering changes also inspect removal flags; do not silently attribute those rules to the running isolated reference snapshot.

Source: [`trajectory_metrics.py`](../../slime/utils/trajectory_metrics.py); [`sglang_rollout.py`](../../slime/rollout/sglang_rollout.py); [`dynamic_sampling_filters.py`](../../slime/rollout/filter_hub/dynamic_sampling_filters.py).

<a id="metrics--7-evaluation-metrics-eval"></a>
### 7. Evaluation metrics (`eval/*`)

Let `D` be the configured dataset name, currently `online-mind2web-monitor`. Evaluation metrics use `eval/iteration`. Evaluation occurs independently of the training reward measurements.

| Metric family | Meaning |
| --- | --- |
| `eval/D` | Mean of non-None rewards in the evaluator's reward list, or 0 if empty. With flattened browser turns this is a turn-weighted reward, not a task success rate. |
| `eval/D/raw_reward_*`, `combined_reward_*`, `format_reward_*`, `judge_reward_*` | Same calculations as section 5, on evaluated turn samples; expand each component under the full `eval/D/` prefix. |
| `eval/D/response_len/{mean,median,min,max}` | Same effective-response-length statistics as collection. |
| `eval/D/{effective_batch_size,effective_batch_ratio,remove_sample_ratio,truncated_ratio,aborted_ratio,failed_ratio,timeout_sample_ratio,sandbox_error_ratio,judge_timeout_ratio,aborted_or_timeout_ratio,repetition_frac}` | Same status, validity-flag, and text heuristics as section 5, on evaluated samples. |
| `eval/D/task/<suffix>` | The complete trajectory table in section 6, grouping flat samples by `(group_index, trajectory_id)` with `sample.index` fallback. These are the task-level evaluation metrics. |
| `eval/D/num_success` | Number of non-ABORTED **turn samples** with reward exactly 1. Not number of solved tasks. |
| `eval/D/num_aborted`, `eval/D/num_non_aborted` | Counts of turn samples by ABORTED status, not full validity or task counts. |
| `eval/D/success_rate_excluding_aborted` | Above turn-level successes / number of non-ABORTED samples, or 0 when denominator is zero. May include removed samples in the denominator. |
| `eval/D-truncated_ratio` | Mean of the evaluator's returned truncation list. Legacy hyphenated alias; inspect alongside `eval/D/truncated_ratio`. |
| `eval/initial_reused` | 1 when an already-computed initial evaluation was copied into a recovery run. |
| `eval/source_run` | String W&B run ID identifying that reused result. |

The reused initial evaluation in `qcq7i4ug` came from `v2d9bk11`: 300 tasks, 54 successes, 146 valid trajectories, 154 invalid trajectories. Consequently task success is **54/300 = 18%**, valid-only success is **54/146 ≈ 36.99%**, and invalidity is **154/300 ≈ 51.33%**. The legacy turn count `num_success=500` is not 500 solved tasks. This monitor used GPT-4.1 rather than the canonical Online-Mind2Web o4-mini judge, and the large invalid fraction limits comparison to paper scores. Reuse is not a fresh evaluation of the trained checkpoint.

Source: [`rollout.py`](../../slime/ray/rollout.py), `_log_eval_rollout_data`; [`online_mind2web_monitor.yaml`](../online_mind2web_monitor.yaml); [`trajectory_metrics.py`](../../slime/utils/trajectory_metrics.py).

<a id="metrics--8-performance-and-resource-metrics"></a>
### 8. Performance and resource metrics

<a id="metrics--collection-throughput"></a>
#### Collection throughput

Let `T` be the measured collection call duration, `G` the configured number of rollout GPUs, `r_i` response lengths, and `e_i` effective response lengths.

| Metric | Calculation / units |
| --- | --- |
| `perf/rollout_time` | T in seconds, including collection's browser/judge/filter wait. In a replay it measures replay loading/handling, not the original online collection. |
| `perf/tokens_per_gpu_per_sec` | `sum(r_i) / T / G` over returned accepted samples. Rejected-attempt tokens are not in the numerator. |
| `perf/effective_tokens_per_gpu_per_sec` | `sum(e_i) / T / G`. |
| `perf/longest_sample_tokens_per_sec` | `max(r_i) / T`. This is a proxy using the whole collection duration, not a measured decoding speed for that sample. |
| `perf/longest_effective_sample_tokens_per_sec` | `max(e_i) / T`. |
| `perf/non_generation_time/{mean,median,min,max}` | Statistics of sample-reported non-generation time, only emitted when some sample reports a positive value. |
| `perf/longest_sample_non_generation_time`, `perf/longest_effective_sample_non_generation_time` | Mean reported non-generation time among samples tied for longest raw/effective response. |
| `perf/longest_sample_tokens_per_sec_without_non_generation`, `perf/longest_effective_sample_tokens_per_sec_without_non_generation` | Longest raw/effective length divided by T minus the corresponding mean non-generation time. |
| `perf/seconds_per_accepted_group` | Original online collector elapsed seconds / accepted groups. |
| `perf/completed_trajectories_per_second` | Attempts in processed completed groups / original online collector elapsed seconds. |

A recovery can mix fresh replay timing with original collection metrics copied from the saved run. In the first recovered iteration, roughly 81 seconds of replay must not be mistaken for the roughly 2,750 seconds needed to collect the browser batch originally. Exclude replay points when estimating sustainable online throughput.

<a id="metrics--training-phase-timers-and-compute-estimates"></a>
#### Training phase timers and compute estimates

Timers accumulate wall-clock seconds on the reporting trainer process between timer resets. Nested timers overlap; do not add every `perf/*_time` field together.

| Metric | Calculation / units |
| --- | --- |
| `perf/actor_train_time` | Time inside the actor optimizer-training phase, summed over the PPO epochs. |
| `perf/data_preprocess_time` | Time in the measured preprocessing blocks, summed over epochs. |
| `perf/train_time` | Overall timed training phase, including its measured preprocessing and update work. |
| `perf/train_wait_time` | Time outside the active training block while the inverse wait timer is running. This is not necessarily the original collection duration, especially during recovery. |
| `perf/step_time` | `train_wait_time + train_time`; a phase-level cycle time, not one optimizer update's latency. |
| `perf/wait_time_ratio` | `train_wait_time / step_time`. |
| `perf/training_time_fraction` | `train_time / step_time`; wall-clock phase share, not GPU utilization or MFU. |
| `perf/sleep_time`, `perf/wake_up_time` | Timed model/engine offload and wake phases in the reporting process. |
| `perf/update_weights_time` | Timed actor-to-inference weight update. |
| `perf/log_probs_time`, `perf/ref_log_probs_time` | Timed optional actor/reference log-probability passes, when executed. |
| `perf/actor_train_tok_per_s` | `sum(timer.seq_lens) / actor_train_time`, using recorded full sequence lengths, not just generated tokens. |
| `perf/actor_train_tflops` | `3 × F / actor_train_time`, where `F = calculate_fwd_flops(recorded_seq_lens) / distributed_world_size / 1e12`. A per-rank analytical estimate. |
| `perf/log_probs_tflops`, `perf/ref_log_probs_tflops` | F divided by the corresponding log-probability-pass time, when available. |

`calculate_fwd_flops` estimates language-model projections, attention, MLPs, and LM head. It does not explicitly model the vision encoder or activation recomputation. The logger also does not explicitly multiply the numerator by PPO epoch count. These are code-derived proxies, **not a validated H200 MFU measurement**. MFU would require a correctly counted useful FLOP numerator and the appropriate hardware peak denominator. There is no calibrated `train/mfu` metric in this baseline.

Source: [`train_metric_utils.py`](../../slime/utils/train_metric_utils.py), [`flops_utils.py`](../../slime/utils/flops_utils.py), [`timer.py`](../../slime/utils/timer.py), [`rollout.py`](../../slime/ray/rollout.py), `compute_perf_metrics_from_samples`.

<a id="metrics--wb-system-tab-and-local-health-log"></a>
#### W&B system tab and local health log

W&B's SDK samples GPU utilization, GPU memory, GPU power, CPU, RAM, disk, and related process/system counters where supported. Names and available fields depend on SDK version and platform; they are vendor/OS telemetry, not values computed by the RL reward or loss functions. GPU utilization reports recent device activity, not useful FLOPs divided by hardware peak. Hostname/process labels can identify multiple shared-run writers.

Observed system-history keys have a `system.` prefix and a shared-writer suffix such as `/l:g001-7cjxbjrh` or `/l:g001-nyexeghf`. The sampled event stream contained these families:

| Key between prefix and writer suffix | Calculation / unit |
| --- | --- |
| `cpu` | Process CPU utilization percentage. |
| `proc.cpu.threads` | Process thread count. |
| `proc.memory.rssMB` | Resident process memory in MB. |
| `proc.memory.percent` | Process memory as percentage of available system memory. |
| `memory_percent` | Overall system memory utilization percentage. |
| `proc.memory.availableMB` | Available system memory in MB. |
| `disk./.usageGB`, `disk./.usagePercent` | Used space and utilization of monitored mount `/`. |
| `disk.nvme1n1p1.in`, `disk.nvme2n1p1.in` | Disk reads in MB accumulated since the first sample. |
| `disk.nvme1n1p1.out`, `disk.nvme2n1p1.out` | Disk writes in MB accumulated since the first sample. |
| `network.sent`, `network.recv` | Network bytes sent/received since monitoring started. |

These SDK units and definitions follow the [W&B system metrics reference](https://docs.wandb.ai/models/ref/python/experiments/system-metrics). Accumulated I/O counters are not bytes-per-second rates. These process/system memory figures are not interchangeable with the Slurm job's enforced cgroup memory budget. GPU fields were not present in the small sampled W&B event response used for this audit; GPU measurements are independently confirmed in the local health log below. This does not assert that every system event lacks GPU fields.

Separately, [`monitor_baseline.py`](../../scripts/monitor_baseline.py) writes `health.jsonl` every 30 seconds:

| Local field | Meaning |
| --- | --- |
| `utc` | UTC sample time. |
| `progress` | Most recent line from `progress.log`. |
| `memory.current`, `memory.peak` | Slurm job cgroup's current and peak memory usage in bytes. |
| `memory.events` | Kernel cgroup counters including `max`, `oom`, and `oom_kill`. They persist across process restarts within the allocation; compare increments, not just nonzero totals. |
| `gpu` | One CSV string per GPU: index, utilization percent, memory used in MiB, power draw in watts, from `nvidia-smi`. |
| `exit_status` | The launch supervisor's exit record, once present. |

These local health records are not automatically W&B history metrics. The monitor records state; it does not restart training or request allocations.

<a id="metrics--9-optional-metrics-and-deliberate-absences"></a>
### 9. Optional metrics and deliberate absences

- `multi_turn/raw_response_length/response_length_{mean,max,min}` describes lengths of loss-mask arrays; `.../response_length_clip_ratio` is the fraction at least `rollout_max_response_len`. `multi_turn/wo_obs_response_length/response_length_{mean,max,min}` uses mask sums instead. `multi_turn/multi_turn_metric/round_number_{mean,max,min}` describes available round counters. Requires `log_multi_turn=True`, currently false.
- `passrate/pass@k` and `eval/D-pass@k` require `log_passrate=True`, currently false. The helper averages `1 - C(n-c,k)/C(n,k)` over eligible prompt groups, with c rewards equal to 1 and k in powers of two up to group size. Valid trajectory grouping matters in turn-level mode; do not assume flattened groups of five turns are five independent attempts.
- `train/kl_loss` requires the KL-loss path; critic/value metrics require a critic. Both are off here. Importance-sampling, mismatch, OPSM, distillation, speculative-decoding, and reward-category metrics depend on their respective features and are not evidence of a logging failure when absent.
- There is no independent post-update browser-reward evaluation after every optimizer batch, no measured MFU, and no full-dataset training success rate in each collection record.
- A metric exists only when its computation is enabled and its population is available. Missing valid-only reward/success rates should not be filled with zero.

<a id="metrics--10-live-logging-checkpoints-and-reproducible-audit"></a>
### 10. Live logging, checkpoints, and reproducible audit

Current run: [W&B qcq7i4ug](https://wandb.ai/zixianma/openwebrl/runs/qcq7i4ug).

Run directory:

```text
/gpfs/scrubbed/zixianma/openwebrl-runtime/runs/openwebrl-4b-reference-282346-20260908T061434
```

| Local artifact | Contents |
| --- | --- |
| `training.log` | Full training/collection output and detailed metric dictionaries. |
| `progress.log` | Concise `[GenerateProgress]`, `[RolloutReward]`, `[TrainMetrics]`, evaluation, and phase records. Some float values are rounded for readability. |
| `health.jsonl` | Periodic cgroup/GPU health records described above. |
| `training_reward_sync.jsonl` (earlier run directories) | Exact reward aliases sent by the earlier auxiliary logger. The current continuation logs them natively. |
| `wandb/` | W&B SDK files; system data and training history need not share the same storage stream. |
| `launch_manifest.json`, `run_config/` | Effective launch settings and archived source/configuration. Use these to distinguish a running snapshot from later worktree edits. |
| `iter_<index>/` | Distributed model/optimizer checkpoint after both PPO epochs. The current continuation first saved `iter_0000002/`, containing 46 Adam updates in total. |
| `latest_checkpointed_iteration.txt` | Latest saved zero-based rollout checkpoint index. |
| `rollout_recovery/<index>.pt` | Saved collected samples for replay; not an updated model checkpoint. |

Checkpoints are saved synchronously **after every completed rollout's training** (`save_interval=1`), not after every optimizer update or at a fixed wall-clock interval. Subsequent checkpoint directories are `iter_0000001/`, etc.

The earlier trainer was already loaded before the reward aliases were added. [`sync_training_rewards.py`](../../scripts/sync_training_rewards.py) therefore copies the existing full-precision W&B collection values into `train/*`, once per collection, merging duplicate replay rows and retaining task metrics when available. It keeps the original reward history, does not invent intermediate observations, and does not mark the primary run finished. It stops at run exit or its explicit deadline. The 18:18 PDT recovery and future launches from the updated repository emit the aliases directly in the rollout logger. The old auxiliary writer exited with its original run.

For an audit, compare local `[TrainMetrics]` records with W&B history using `train/loss` and `train/step`; also retain rollout/epoch identity because the current step formula can overlap across variable-size collections. Compare `train/reward` to `rollout/raw_reward_mean` by reward iteration, deduplicating replayed collection entries. Confirm one new reward point per genuinely new completed collection. W&B summaries are not a history audit: the shared-writer summary queried at 18:39 PDT omitted the reward fields even though history contained them. Use history to check missing points. The API audit before this document found all 14 first-collection optimizer records and the matching reward alias.

Documentation preference: keep repository documents in `openwebrl/docs/`.

<a id="metrics--recovery-audit-on-2026-09-07"></a>
#### Recovery audit on 2026-09-07

At 18:39 PDT, W&B history contained `train/reward=0.38741396263520156`
for reward iteration 1 and `0.3325812274368231` for iteration 2. Two recovery
attempts also emitted iteration 2 with the same measured value. These are two
distinct browser collections, not four observations. Repeated optimizer records
from replay likewise do not imply extra completed baseline iterations. Durable
progress must be checked in the checkpoint marker, dataset cursor, and Adam
optimizer counters, with the scheduler checked separately; see
`H200_TESTING.md` for recovery outcomes.

At approximately 21:08 PDT, the g005 continuation saved checkpoint 2 with
Adam parameter-group steps `[46, 46]`, matching 30 restored plus 16 newly
logged optimizer updates. Its scheduler reports 12032/256 = 47 batches.
The old initialization code advanced the scheduler again after Megatron had
restored it, adding one batch when loading checkpoint iteration 1. This is a
bookkeeping offset, not another optimizer update. Constant LR and weight decay
make it numerically harmless for this recipe; nonconstant schedules would be
affected. The repository fix removes the redundant advance for future launches.

The checkpoint inspector rejects scheduler/Adam mismatches by default. For the
diagnosed offset in this continuation, use
`--expected-scheduler-offset-updates 1`, alongside the true
`--expected-updates` count. It reports Adam updates, raw scheduler batches,
and the offset separately. Do not count durable updates from the scheduler
alone when Adam counters are available. This change does not rewrite existing
checkpoint files.

<!-- document:METRICS.md:end -->

---

<!-- document:PAPER_REWARD_COMPARISON.md:start -->
<a id="paper-reward-comparison"></a>
## Comparing the reference run's reward plot

_Source record: `PAPER_REWARD_COMPARISON.md`. Dated entries retain their historical context._


The paper's Figure 2(a) uses **iteration** on the x-axis and **training reward (%)** on the y-axis. Section 5.1 defines an iteration as collection followed by policy updates. Its displayed curve spans the 90-iteration first stage; faint observations and a thicker trend are visible. [Paper](https://arxiv.org/pdf/2606.02031), [official figure](https://openwebrl.github.io/static/images/raw_success_rate_curves_sft_base.png).

<a id="paper-reward-comparison--the-chart-to-use"></a>
### The chart to use

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

<a id="paper-reward-comparison--why-optimizer-updates-do-not-add-reward-observations"></a>
### Why optimizer updates do not add reward observations

The first collection yielded 2,034 turn samples. Seven full 256-turn minibatches fit in each epoch; two epochs made 14 optimizer updates. Those updates reused rewards assigned during that same collection. A per-minibatch mean could be logged, but would describe which stored samples each update consumed. It would not add another collection to the x-axis.

Two completed collections therefore give x=0 and x=1 on the comparison chart. Seven minibatches are not seven online collection iterations. The third collection yielded 2271 turn samples: eight full minibatches per epoch and two epochs produced 16 optimizer updates, but only one new reward observation. Thus the live run had three reward points while collection 3's optimizer updates were running. Collection 4 adds the next point only after fresh browser collection finishes.

Audited observations as of 2026-09-08 03:34 PDT:

| Zero-based iteration | Collected-turn reward (%) |
| --- | ---: |
| 0 | 38.7413962635 |
| 1 | 33.2581227437 |
| 2 | 38.8815499780 |
| 3 | 36.2483602973 |
| 4 | 35.1788339049 |
| 5 | 34.2671335668 |
| 6 | 39.8766700925 |

These are raw observations, not smoothed trend values. These first seven points are insufficient to establish convergence, divergence, or successful reproduction. Browser access, task sampling, trajectory lengths, and filtering can change the reward independently of policy quality. Inspect completed-task success and invalidity alongside it, and compare held-out evaluation with matching judge and validity conventions before claiming benchmark reproduction.

<a id="paper-reward-comparison--how-much-variability-is-visible-after-four-collections"></a>
### How much variability is visible after four collections?

A lightweight check on 2026-09-07 reproduced all four W&B means from the saved rollout batches, then resampled the 48 accepted prompt groups in each collection with replacement. Each resampled group retains all its observed turns and trajectories; the statistic is the sum of rewards divided by the number of turns. This preserves the collector's turn weighting and avoids treating correlated turns from the same prompt as independent observations. With 10,000 draws and NumPy seed 7, the percentile intervals were:

| One-based collection | Reward (%) | Conditional 95% bootstrap interval (%) |
| --- | ---: | ---: |
| 1 | 38.74 | 31.45–46.24 |
| 2 | 33.26 | 26.26–40.05 |
| 3 | 38.88 | 33.50–44.48 |
| 4 | 36.25 | 29.27–43.36 |

The intervals overlap substantially. Independently resampling adjacent collections also gives difference intervals that include zero for all three adjacent changes. These four observations therefore provide little evidence for interpreting the oscillations as a learning trend. This calculation is conditional on the observed, filtered prompt groups: it does not measure held-out policy performance or account for every source of task selection, browser, or judge variability. Interval overlap alone is not proof that the policy has remained unchanged.

The complete numeric results and source-batch paths are in `/gpfs/scrubbed/zixianma/openwebrl-runtime/runs/openwebrl-4b-reference-282346-20260908T024410/reward_cluster_bootstrap.json`. This analysis does not add another W&B reward series or change training.

<a id="paper-reward-comparison--live-behavior-and-provenance"></a>
### Live behavior and provenance

The duplicate paper writer and native `paper/*` emission are disabled. Existing `paper_reward_sync.jsonl` is retained as provenance. `scripts/sync_training_rewards.py` supplied the earlier run's `train/reward` alias; the current g005 continuation emits it directly. These logging changes do not change the training recipe.

See [METRICS.md](RL_METRICS.md#metrics) for reward formulas, subset weighting, task denominators, update counts, and known axis limitations. In particular, the current optimizer `train/step` label can jump when minibatch counts change; the paper comparison uses collection identity and is unaffected by that label issue.

<!-- document:PAPER_REWARD_COMPARISON.md:end -->

---

<!-- document:GRADIENT_DIAGNOSTICS.md:start -->
<a id="gradient-diagnostics"></a>
## Gradient spikes and training diagnostics

_Source record: `GRADIENT_DIAGNOSTICS.md`. Dated entries retain their historical context._


Audit: 2026-09-09, W&B `qcq7i4ug`; active job 285546.

<a id="gradient-diagnostics--what-the-gradient-norm-measures"></a>
### What the gradient norm measures

`train/grad_norm` is the global L2 norm of the aggregated minibatch gradient,
returned by Megatron **before clipping**. The active configuration has
`clip_grad=1.0`. The optimizer computes the norm, scales gradients by roughly
`min(1, 1 / (norm + epsilon))`, then performs the Adam update. A logged norm of
10 therefore implies a roughly 0.1 clipping multiplier; it does not imply a
tenfold parameter update. Adam's accumulated moments and parameter-wise
preconditioning further separate gradient norm from parameter displacement.

The installed implementation is in
`openwebrl-runtime/src/Megatron-LM/megatron/core/optimizer/optimizer.py`,
`MegatronOptimizer.clip_grad_norm` and `ChainedOptimizer.step`.
The general operation is documented by
[PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html).

A high gradient norm is a diagnostic, not reward, generalization, or a universal
error threshold. Compare against this run's usual range and its neighboring
updates. Sustained increases, nonfinite values, increasing PPO KL/clipping,
entropy collapse or held-out regression would be more concerning than an
isolated clipped spike. Removing an example solely for causing a spike can
remove useful learning signal and should not be the default response.

<a id="gradient-diagnostics--observed-spikes"></a>
### Observed spikes

The W&B scan returned 303 distinct optimizer history-row IDs after collapsing
duplicate API rows. These include historical retries and are not 303 durable
Adam updates. Median gradient norm was 1.8474, p95 was 3.2182, and the maximum
was 10.5391. The current allocation's maximum at the initial audit was 2.9194.

| Reward iteration | PPO epoch, minibatch (one-based) | Gradient norm | Approximate clipping multiplier | Sampled PPO KL | PPO clipping fraction |
|---|---|---:|---:|---:|---:|
| 4 | 2, 4 | 10.5391 | 0.0949 | 0.001761 | 1.070% |
| 16 | 1, 2 | 9.4544 | 0.1058 | 0.001774 | 0.707% |

The scalar values are real; both spikes match the local trainer logs. However,
`train/step` is not a globally monotonic counter in this implementation: it is
derived from rollout index and the current batch's number of PPO updates.
For example, the label falls from 239 at iteration 20 to 200 at iteration 21,
because the latter has fewer minibatches. Inspect W&B history `_step` or wall
time when assessing temporal spike/recovery relationships. A durable cumulative
optimizer-update coordinate should be added in a future telemetry change.

<a id="gradient-diagnostics--saved-data-inspection"></a>
### Saved data inspection

`scripts/audit_gradient_spike_batches.py` memory-maps the trusted recovery
archive and inspects Python metadata, without accessing image tensor payloads.
It requires DP1, fixed global batch 256, no sequence-length balancing, and
GRPO standard-deviation normalization. It reconstructs each epoch's logged
shuffle seed, trim and 256-turn batches. Reconstructed mean absolute advantages
are checked against logged trainer statistics to catch an ordering mismatch.

| Batch property | Iteration 4 spike | Iteration 16 spike |
|---|---:|---:|
| Turn samples | 256 | 256 |
| Distinct trajectories | 148 | 132 |
| Most turns from one trajectory | 5 | 5 |
| Mean absolute normalized advantage | 0.7870 | 0.7400 |
| Maximum absolute normalized advantage | 1.7889 | 1.7889 |
| Mean response tokens | 327.7 | 346.2 |
| Response token range | 112–866 | 91–832 |
| Nonfinite stored rollout log probabilities | 0 | 0 |
| Removed samples | 0 | 0 |

These properties are comparable to neighboring minibatches. The checks do not
show a short/empty-response pathology, an exploding normalized advantage, or
an unusual single-trajectory concentration. They do not identify the causal
sample or rule out model-internal numerical sensitivity: per-sample or
per-module gradient measurements would be needed for that. No examples were
deleted, and the training recipe was not changed.

Full reports, including the 256 sample/task/trajectory IDs for each spike,
are under the current run directory
`openwebrl-runtime/runs/openwebrl-4b-reference-285546-20260909T235114/`:
`gradient_spike_wandb_history.json`, `gradient_spike_iteration4.json`, and
`gradient_spike_iteration16.json`.

<a id="gradient-diagnostics--metrics-to-watch-alongside-reward"></a>
### Metrics to watch alongside reward

<a id="gradient-diagnostics--additional-spike-investigated-during-job-286094"></a>
#### Additional spike investigated during job 286094

At iteration 26, second PPO epoch, minibatch 1 (zero-based), the norm reached
12.3879 (W&B history row 459, legacy `train/step=307`). The next update returned
to 1.2274. PPO KL was 0.001587 and clipped-objective fraction 0.011807 at the
spike. The configured norm threshold remained 1.0; clipping scales this gradient
by approximately 0.0807 before Adam, which does not imply the same factor for
the eventual parameter update.

CPU metadata reconstruction used logged seed 34440 and independently matched
the spike batch's mean absolute advantage to the trainer. Its 256 turns came
from 142 trajectories, at most five turns from one trajectory. Mean absolute
advantage was 0.8091, maximum 1.7889; response lengths ranged from 147 to 644
tokens (mean 335.3). There were no removed samples or nonfinite stored log
probabilities. These checks again found no clear metadata pathology, but do
not identify the causal sample or measure per-sample gradient contributions.

Training continued, all twelve iteration-26 updates matched W&B, and checkpoint
25 validated at 338 Adam updates. The sample-ID report is
`openwebrl-runtime/runs/openwebrl-4b-reference-286094-20260910T071441/gradient_spike_iteration26.json`.
It was collected while the latter half of the epoch was still running, so later
batch gradient fields in that report are null; the completed optimizer audit is
`iteration_26_wandb_audit.json`. No samples or recipe settings were changed.

| Existing metric | What to look for |
|---|---|
| `eval/online-mind2web-monitor/task/success_rate_all_completed` | Held-out task performance. Compare with invalid rate, not the turn-weighted evaluation reward. |
| `train/task_success_rate` and `train/task_invalid_rate` | Completed-attempt success before acceptance filtering; failures/timeouts can alter apparent reward trends. |
| `train/ppo_kl` and `train/pg_clipfrac` | Policy movement and the fraction where PPO's clipped objective is selected. Abrupt sustained increases merit investigation. |
| `train/entropy_loss` | Predictive entropy: a rapid fall can indicate shrinking exploration; a rise is not automatically good. |
| `train/abs_advantage` | Magnitude of the normalized learning signal; use to interpret gradient changes, not as a success metric. |
| Response/trajectory lengths and truncation rates | Detect looping, budget exhaustion, changed behavior and altered turn weighting. |
| Accepted/completed prompt groups and collection duration | Dynamic-filter yield and useful data throughput. |
| GPU utilization, cgroup memory/events, durable checkpoint updates | Efficiency, resource pressure, and actual recoverable progress. |

Metric formulas and exact existing keys are in [METRICS.md](RL_METRICS.md#metrics).
In this code, `ppo_kl` is a signed sampled old-minus-current log probability,
not exact distributional KL or KL to the SFT reference. `pg_clipfrac` measures
clipped-objective selection, not gradient clipping. `train_rollout_logprob_abs_diff`
is expected to be zero when both operands reuse stored rollout probabilities;
it is not an independent actor-versus-inference consistency check.

Useful future additions: a monotonic Adam-update coordinate; rollout/epoch/
minibatch identity on every update; gradient clipping multiplier and fraction
of updates clipped; per-module gradient norms (vision, connector, language);
and a compact sample-ID trace plus batch length, advantage-tail and task-mix
statistics when a norm exceeds a rolling outlier threshold. These are proposed
telemetry additions, not metrics already installed in the active worker.

<!-- document:GRADIENT_DIAGNOSTICS.md:end -->

---
