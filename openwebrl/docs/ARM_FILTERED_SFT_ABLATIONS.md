# Next action-level filtered-SFT ablations

Status: **ablation 1A is running in Slurm job 285567**. On 2026-09-09 the user
deprioritized the update-500/update-700 full-300 checkpoint study in favor of
training ablations compared with the existing C2 baseline. The approved first
job uses one H200 for five hours and includes fixed-100 validation. Update 500
is the reference policy. Live configuration and paths are recorded in
[ARM_C2_ABLATION_1A_RUN.md](ARM_C2_ABLATION_1A_RUN.md).

## What the first C2 curve says

The current dataset has **8,394 retained turns from 1,151 successful
trajectories**. Training used language-only LoRA rank 16/alpha 32/dropout 0.05,
AdamW at `1e-5`, effective batch 16, 3% warmup, and cosine decay planned over
two epochs (1,050 updates). Update 500 is about 95% of one data pass; update
700 is 1.33 passes and update 923 is 1.76 passes.

On the fixed 100-task evaluation, updates 500 and 700 both scored 33/100 on the
first pass, while update 923 fell to 28/100. Update 500 also had the best retry
replacement estimate and healthy termination/trajectory diagnostics. Training
cross-entropy changed only modestly, from 0.1550 in epoch 1 to 0.1474 during
the observed part of epoch 2. This points first to **stopping around one pass**;
it does not yet show that LoRA capacity is the bottleneck. Update 500 is the
reference for the next training jobs; the full-300 update-500/update-700 study
is now optional.

The likelihoods below are subjective priors for beating the current update-500
standalone policy on held-out task success. They are planning aids, not measured
probabilities.

## 1. Optimizer/exposure and LoRA-rank ablation

**Recommendation: run 1A first. If two runs are available, run 1A and the
combined variant 1AB. A rank-only 1B run is the lowest priority.**

All variants train on the same immutable 8,394 C2 rows, starting again from
`OpenWebRL/OpenWebRL-4B-SFT`. Keep the same LoRA modules, loss masking,
optimizer, and peak learning rate. For 1A and 1AB, change effective batch from
16 to **32** by using microbatch 1 and accumulation 32, and stop after exactly
**one pass**: `ceil(8394 / 32) = 263` updates.

For those two variants, parameterize the schedule by **examples seen** so the
larger batch does not accidentally change it. Warm up over the first 512
examples and use the same first-epoch portion of the old two-epoch cosine
curve, so learning rate is about `5e-6` at the endpoint rather than decaying to
zero. Save updates 66, 132, 198, 250, and 263 for diagnostics. Predeclare
update 263 as the policy endpoint; do not choose it using Online-Mind2Web
checkpoint results.

The three possible training runs are:

| Variant | Rank / alpha | Batch and schedule | Comparison purpose | Estimated chance of beating update 500 |
| --- | ---: | --- | --- | ---: |
| **1A: optimizer/exposure** | 16 / 32 | effective batch 32; one-pass exposure-matched schedule | Tests the change most directly supported by the current curve | **40–55%** |
| **1B: rank only** | 32 / 64 | effective batch 16; original first-pass schedule; stop at update 525 | Isolates adapter capacity against update 500 | **20–35%** |
| **1AB: combined** | 32 / 64 | effective batch 32; one-pass exposure-matched schedule | Compares directly with 1A to measure added rank capacity | **30–45%** |

Use dropout 0.05 and peak learning rate `1e-5` in all variants. Rank 32/alpha 64 is
also the released OpenWebRL ARM recipe, although that precedent trains a
reward model and is not evidence that rank 32 is optimal for actor
distillation. The upstream actor-distillation code defaults to rank 16/alpha
32. Adapter and optimizer memory roughly double, but remain small relative to
the frozen 4B base and long-context activations.

Variant 1A tests whether noisy effective-batch-16 updates contributed to the
shallow loss curve and removes the second pass that coincides with policy
regression. The current low training loss and late regression do not look like
clear under-capacity. Additional rank may instead fit noisy local selections
more closely, so 1B is less promising. Variant 1AB is useful after 1A because
their direct comparison isolates rank while holding the improved optimizer
recipe fixed. Rank 32 remains a much cheaper capacity test than full
fine-tuning.

If budget allows a third optimizer control, add one-pass batch 16 under the
same example-indexed schedule. That isolates effective batch size; otherwise
the existing update-500 result remains an approximate, rather than exact,
batch-16 reference.

## 2. ARM-native counterfactual data mixed with C2

**Recommendation: highest-upside data ablation. Estimated likelihood of a real
improvement: 50–65%, conditional on the confidence filter retaining useful
coverage.**

Use the released
[`PTeterwak/action-reward-models-data`](https://huggingface.co/datasets/PTeterwak/action-reward-models-data)
at revision `0d83b48c1659cac47a1044ef88fb573d3c16e180`. The local audit found:

| Released OpenWebRL asset | Audited count |
| --- | ---: |
| Extracted states | 3,085 states, 412 episodes, 397 task texts |
| GPT-5.5 teacher labels | 39,997 labels over 2,999 base states |
| Built selection examples | 39,155 train + 782 validation = 39,937 |
| Repeated draws | median 16 and maximum 16 labels per base state |
| Exact normalized task-text overlap with our 300 OM2W tasks | 0 |

The release split is by label draw, so sibling draws from one base state can
cross train and validation. Rebuild an **episode-disjoint** audit split and do
not use the released validation loss as evidence of state generalization.

For actor training, recover the teacher-selected candidate response from each
five-candidate prompt. Group all draws by base state, normalize executable tool
actions, and retain at most one target per base state. The initial strict tier
should require action-level agreement among the GPT-5.5 label, SelectionARM,
and ScalarRM; record actor candidate support and scalar winner/runner-up margin
as diagnostics. If that tier is too small or badly skewed, relax one criterion
before training according to a frozen retention report. These are confidence
proxies: the OpenWebRL release has no GPT-5.5 PRM score or calibrated 0.7 quality
floor.

Concatenate one pass over every C2 row with one pass over the retained
ARM-native base states, shuffle by source/episode, and cap each source episode
at eight retained states to limit trajectory dominance. Use the stabilized
LoRA recipe above, effective batch 32, and stop after one combined pass. The
exact update count is `ceil((8394 + N_ARM) / 32)` and must be frozen only after
the filter audit; do not assume all 39,937 repeated draws are independent
examples.

Run a matched **random-candidate SFT control**: keep the same C2 rows and
ARM-native states but replace each ARM-native selected target with a random
distinct executable candidate. This is required to distinguish useful action
selection from generic extra self-SFT. The upstream positive actor-distillation
result included this control; its failed online run did not have a quality
floor and developed long scrolling loops.

This recipe adds a signal C2 lacks: direct comparisons among counterfactual
actions at the same state, including states from trajectories that need not
have terminal success. Its main risks are selector-label circularity (both
released ARMs were trained on these teacher labels), best-of-bad targets, and
distribution mismatch. Report source-specific loss, action mix, termination,
step caps, scrolling, and repeated actions.

## 3. Full language-tower fine-tuning on stabilized C2

**Recommendation: conditional capacity ablation after the one-pass LoRA run.
Estimated likelihood of a real improvement: 20–35%; highest compute and
forgetting risk.**

Freeze the vision encoder and visual merger, but train all language-tower
parameters on the original C2 data for one pass. Keep effective batch 16 to
match the existing run, use AdamW with peak learning rate **`1e-6`**, 512-example
warmup, the same exposure-matched half-cosine schedule, gradient clipping 1.0,
BF16, and gradient checkpointing. Save at 25%, 50%, 75%, 95%, and 100% of the
8,394 examples; predeclare the one-pass endpoint.

This isolates whether LoRA rank 16 prevents the actor from absorbing selected
actions. It should not be the first rerun: the current loss is already low and
the regression occurs with more exposure, which is stronger evidence for data
or regularization limits than for insufficient capacity. A full vision-model
update would add more memory cost and could disturb an already useful visual
representation, so it is a later ablation only if the language-tower run is
promising.

## Training decision order

1. Use update 500 as the existing reference and one pass as the default
   stopping rule. The full-300 update-500/update-700 study remains optional.
2. Run 1A first. If a second optimizer run is available, run 1AB with the same
   update-263 endpoint, row order, and schedule. Intermediate checkpoints are
   for loss and behavior diagnostics. Promote rank 32 only if 1AB improves task
   success over 1A without degrading termination or loop metrics.
3. Skip rank-only 1B unless clean factor attribution is worth a third training
   run. Build the ARM-native retention audit and run selected-target plus matched
   random-target variants. Promote it only if selected targets beat both the
   stabilized C2 run and random-target control.
4. Run full language-tower fine-tuning only if rank 32 improves over rank 16
   but still appears capacity-limited, or the ARM-native recipe improves while
   its selected-target loss remains materially above the C2 loss.

All final comparisons use one-action inference with no ARM and the same
`o4-mini` Online-Mind2Web/AgentTrek judge. Use overall success as the primary
metric, valid-only success as a sensitivity metric, and preserve the behavior
checks that caught upstream loop collapse.
