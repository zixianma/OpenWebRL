# Gradient spikes and training diagnostics

Audit: 2026-09-09, W&B `qcq7i4ug`; active job 285546.

## What the gradient norm measures

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

## Observed spikes

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

## Saved data inspection

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

## Metrics to watch alongside reward

### Additional spike investigated during job 286094

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

Metric formulas and exact existing keys are in [METRICS.md](METRICS.md).
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
