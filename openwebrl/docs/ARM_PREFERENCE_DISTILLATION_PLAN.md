# ARM same-state preference distillation plan

The next experiment is specified in the
[v2 controlled viability plan](ARM_PREFERENCE_V2_VIABILITY_PLAN.md). The
configuration and budgets below are historical proposals, not the next launch.

Status: **the first calibrated half-epoch run and fixed-100 evaluation are
complete**; see its [run record](ARM_PREFERENCE_RUN_286384.md). It scored
31/100, between the historical base at 26/100 and C2/1A at 33/100. Its pair
audit and loss instrumentation were inadequate, so do not scale this recipe or
run it on all 300 tasks. This document freezes the purpose and
comparison structure before inspecting preference-training results. Any GPU
allocation still requires a separate exact resource approval.

[ARM results dashboard](ARM_RESULTS_DASHBOARD.md) ·
[filtered-SFT results](ARM_C2_ABLATION_1A_RESULTS.md) ·
[broader integration plan](ARM_INTEGRATION_PLAN.md)

## Why this is next

SelectionARM best-of-five improved the starting actor by 12.7 percentage
points overall at inference time. The best standalone filtered-SFT policy,
1A, improved by 3.3 points over the starting actor, but the primary paired
all-scheduled comparison was not significant (`p=0.245`). Against original C2
on the fresh holdout, 1A gained 2.5 points (`p=0.256`).

The current C2 objective teaches only the selected response. It discards the
same-state comparison that made ARM useful at inference time. The next
experiment therefore retains winner imitation and adds a pairwise loss between
the selected response and a distinct rejected candidate from the same browser
state.

## Stage 1: candidate-pair audit

Use the immutable 8,394-turn C2 dataset and its original candidate exports.
For every retained turn:

1. Verify the causal pre-action prompt, screenshot, candidate token IDs,
   SelectionARM result, executed winner, and successful trajectory join.
2. Normalize executable tool calls and group exact or action-equivalent
   duplicates.
3. Retain the SelectionARM winner and enumerate distinct, parseable rejected
   candidates. Never assign an environment outcome to an unexecuted response.
4. Report pair coverage, candidates per state, action-type transitions,
   winner/loser response lengths, fallback/parse failures, and site/episode
   concentration.
5. Freeze a deterministic one-loser rule before training. The fastest first
   iteration uses one seeded distinct executable loser and requires no new ARM
   scoring. A later hard-negative version may score rejected candidates with
   ScalarRM, reported as a separate data recipe.

This audit is CPU-only. If fewer than 70% of C2 turns yield a distinct valid
pair, inspect the missing categories before training rather than silently
shrinking the dataset.

## Objective

For pre-action context `h`, winner `y_w`, loser `y_l`, current policy `pi`, and
the frozen starting actor `pi_ref`, cache the reference log-ratio and optimize:

```text
delta_pi  = log pi(y_w | h)     - log pi(y_l | h)
delta_ref = log pi_ref(y_w | h) - log pi_ref(y_l | h)

L_pref = -log sigmoid(beta * (delta_pi - delta_ref))
L      = L_sft(y_w | h) + lambda * L_pref
```

Use response-token means for the initial log probabilities so reasoning length
does not dominate the comparison; record this as length-normalized DPO rather
than claiming an exact reproduction of standard sequence-sum DPO. Mask the
entire shared context and supervise only current winner/loser response tokens.
Cache reference values with a model, tokenizer, processor, prompt, image-grid,
and pair-manifest hash. Recompute on any mismatch.

Before the full run, use a small frozen pair sample to measure both loss and
gradient scales. Freeze two conservative `lambda` values whose preference
gradient norms are approximately 10% and 30% of the winner-SFT gradient norm.
Keep `beta=0.1` for both unless the audit shows numerical saturation. The
lambda calibration uses training pairs only and cannot inspect Online-Mind2Web
outcomes.

## Frozen training comparison

Both variants start independently from `OpenWebRL/OpenWebRL-4B-SFT` and use
the same pair manifest and row order.

| Setting | Value |
| --- | --- |
| Variants | low-lambda and medium-lambda SFT + preference loss |
| Control | completed 1A winner-only SFT; rerun only if training-seed attribution is required |
| LoRA | language tower, rank 16, alpha 32, dropout 0.05 |
| Optimizer | AdamW, peak LR `1e-5`, betas 0.9/0.95, epsilon `1e-8`, weight decay 0.01 |
| Batch | microbatch one state-pair, accumulation 32, effective batch 32 states |
| Exposure | one pass over every audited pair; endpoint predeclared |
| Schedule | 512-state warmup and the 1A example-indexed half-cosine schedule |
| Precision | BF16, gradient checkpointing, gradient clipping 1.0 |
| Checkpoints | 25%, 50%, 75%, and endpoint; endpoint is primary |
| Logging | total, SFT, and preference loss; chosen/rejected log-ratios; accuracy; margins; gradient norms; source/action mix |

Serial winner and loser forwards keep peak activation memory near the 1A
configuration at the cost of additional compute. The implementation must be
resumable and must preserve cached reference scores and optimizer/RNG state.

## Evaluation and promotion

Evaluate both fixed endpoints with one actor candidate per turn and no ARM.
Use the existing fixed 100 tasks as development evaluation because those tasks
have already informed prior choices. Report overall and valid-only success,
paired wins/losses, availability, trajectory length, termination, 30-step caps,
scrolling, and repeated actions.

Promote at most one preference variant. It should beat 1A and the other lambda
on overall success with positive paired net wins, without materially worse
availability or the upstream scrolling/termination-collapse signature. Small
fixed-100 differences remain screening evidence, not a final claim.

Only the promoted endpoint receives a full comparison. Run it concurrently
with the starting actor on all 300 Online-Mind2Web tasks. The 200-task stratum
is no longer pristine because it has already been inspected, so report this as
a confirmatory repeat rather than a first untouched test. A publishable method
claim ultimately requires multiple training seeds or a new task-disjoint test
set.

## Fastest resource plan

Measured 1A timing on one H200 was 2h53 for 263 training updates and 1h02 for
fixed-100 evaluation, plus a few minutes for merge and server startup. A
preference run adds a loser policy pass and frozen-reference scoring. Before a
paid full run, benchmark 256 audited pairs to replace the multiplier below with
a measured estimate.

| Scope | GPUs | Wall-time estimate | Maximum H200-hours | Purpose |
| --- | ---: | ---: | ---: | --- |
| CPU pair audit and implementation | 0 | 2–4 engineering hours | 0 | Freeze pair manifest and loss contracts |
| GPU smoke/256-pair benchmark | 1 H200 | 30–45 minutes | 0.75 | Validate memory, cached reference scores, and throughput |
| Minimum pilot: one lambda + fixed-100 eval | 1 H200 | 7–8 hours | 8 | Cheapest result, no preference-weight comparison |
| **Fastest informative iteration**: two lambdas in parallel + fixed-100 evals | **2 H200s** | **7–8 hours** | **16** | Select a preference weight in one wall-clock cycle |
| Final promoted policy vs fresh concurrent base | 2 H200s | 1.0–1.5 hours | 3 | All-300 confirmation |

The recommended planning envelope is therefore **two H200s for eight hours**
for the fastest informative training iteration, after the short one-GPU
benchmark confirms it fits. At the cluster's recent estimate of $0.90 per
H200-hour, the maximum is about **$14.40**; actual use can be lower if both
controllers exit after evaluation. Including the later all-300 confirmation
raises training and evaluation to at most 19 H200-hours, about $17.10. The
separate 256-pair benchmark brings the full conservative envelope to **19.75
H200-hours, about $17.78**.

Two GPUs are the useful concurrency point for the first iteration: each owns
one independent lambda run and its evaluation. More GPUs would add variants
rather than shorten the critical path without first implementing distributed
training.

## Follow-on data experiment

After preference distillation, audit
`PTeterwak/action-reward-models-data` into episode-disjoint, one-target-per-state
examples. Compare C2 plus selected targets against C2 plus an equal number of
random executable targets. Keep the 1A optimizer recipe. This is the next
highest-upside data experiment, while rank-32 LoRA and full language-tower
fine-tuning remain secondary capacity checks.
