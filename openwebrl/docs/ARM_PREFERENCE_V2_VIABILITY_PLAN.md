# ARM preference distillation v2: controlled viability test

Latest discussion: [joint C2 and direct-teacher data proposal](ARM_JOINT_DATA_TRAINING_PLAN.md)
would replace the 2,048-state cap with the full cleaned pool plus audited
OpenWebRL teacher states. Its latest revision proposes combined-data SFT
first, then full-response DPO-only and action-masked preference-only on
identical pairs. It also specifies the shared filtering contract.
It is a proposal, not a launched run.

Status: direction approved for preparation and discussion. No new GPU budget
has been approved. This plan supersedes the proposed next run in
[the original preference plan](ARM_PREFERENCE_DISTILLATION_PLAN.md).

The [lightweight CPU audit](ARM_PREFERENCE_V2_CPU_AUDIT.md) is complete:
7,140/8,394 states have a structurally distinct alternative; a provisional
coordinate exclusion leaves 6,247. Excluding done-versus-done alternatives
as well would leave 5,490. Before freezing pairs, resolve semantic ambiguity
and replace the provisional sampler, which overrepresents terminal answers.
The current manifests are preparation artifacts, not approved training inputs.

## Question and interpretation

Can a corrected action-preference objective improve held-out ARM preference
agreement beyond winner-only SFT on exactly the same states and exposure?
The prior 31/100 endpoint is an engineering pilot, not a clean test of this
question: it had numerical issues, action-equivalent negatives, and a different
winner exposure from 1A. Its result does not establish that distillation fails.

## CPU data preparation

Start with the immutable 8,394 C2 winner turns and their five original
candidates. Verify source hashes, executed-winner joins, pre-action contexts,
screenshots, token IDs, and absence of selector fallback. Parse tool calls
using the browser's parser and validate argument schemas. Canonicalize JSON
key order and documented defaults while preserving case-sensitive text, URLs,
and call order. Exclude rejected actions identical to the selected action,
including responses that differ only in reasoning.

Treat small coordinate differences as uncertain equivalence, not proof of a
different action or of identity. Report several distance thresholds and inspect
screenshots to set a conservative exclusion rule before training. Nearby
distinct elements must not automatically be merged. Keep ambiguous examples
out of the first pilot. Report retention by task, website, and action type, plus
counts of states with one or more usable negatives. The old 99.63% coverage
number measures different response strings and is not the v2 retention rate.

Inspect 64 stratified examples with their screenshots, calls, and ARM choices
before freezing the manifest. Unexecuted candidates have no measured terminal
outcome; ARM-selected is a preference label, not verified action correctness.

Split by task/episode before sampling turns. Target 2,048 training states and
256 fixed validation pairs, stratified by website and action type; report actual
task counts. Training and validation cannot share a trajectory. This is
within-source validation; it does not establish generalization to unseen sites.
If retention is too low to support these counts without duplicate states,
revise the pilot size before allocation rather than resampling duplicates.

Choose one structurally distinct rejected candidate per state. Rank valid
negatives by frozen-base action log-probability and use the highest-scoring
eligible negative as a model-hard negative; this is not an ARM runner-up or a
verified bad action. Cache all reference scores and freeze candidate selection
before training. Do not add ScalarRM labels in this first comparison.

## Matched two-arm experiment

Both arms start from OpenWebRL/OpenWebRL-4B-SFT with identical seeds, chosen
states, order, optimizer schedule, and winner-token exposure:

* Control: full-response winner SFT.
* Treatment: the same winner SFT plus action-token preference loss.

The historical 1A endpoint remains context, not the causal control. Comparing
2,048 preference pairs against 8,394 SFT winners would confound the objective
with exposure. Extra loser compute is reported separately.

For response y=(reasoning r, action sequence a), s_pi(h,y) is the mean
teacher-forced log probability over validated action-token positions only.
The original reasoning remains in the prefix but receives no direct preference
loss. Full winner SFT continues supervising reasoning and action tokens.

```
z = beta * [(s_pi(h,y_w) - s_pi(h,y_l))
            - (s_ref(h,y_w) - s_ref(h,y_l))]
loss = winner_response_CE + lambda * softplus(-z)
```

This compares p(a_w|h,r_w) and p(a_l|h,r_l), not marginal p(a|h). Candidate
reasoning therefore remains a confound; gradients can also flow through its
hidden states. Call this an action-masked preference surrogate, not standard
DPO or a pure same-context action ranking model. Report both action-only and
whole-response scores on validation. A common-reasoning intervention would be
a separate experiment because it changes the collected conditioning context.

Initial settings: language-only LoRA rank 16/alpha 32/dropout 0.05, AdamW
LR 1e-5, betas 0.9/0.95, epsilon 1e-8, weight decay 0.01, batch 32 states,
one pass (64 updates for 2,048 states), 512-state warmup, half-cosine with
horizon 4,096 states, gradient clipping 1.0. Schedule exposure counts states,
not chosen-plus-rejected responses. BF16 model with FP32 log-probability,
margin, and loss arithmetic. Start beta at 0.1; inspect score scale before
freezing beta and lambda. Neither is inherited from the failed pilot.

## Calibration and engineering gates

Use 128 stratified training pairs, grouped into four batch-32 calibration
batches. Measure SFT and unit-weight preference gradient norms and their dot
product/cosine, then choose one lambda initially targeting a median weighted
preference/SFT gradient ratio of 0.3. This target is a conservative heuristic,
not an evidence-backed optimum. Check whether the combined gradient is a
descent direction for preference loss; a norm ratio alone cannot ensure this.
If it conflicts, investigate representative batches before selecting lambda.

Check the choice with a disposable ten-step rehearsal and reset weights,
optimizer, and RNG before the actual comparison. Calibration never reads OM2W
outcomes or uses the held-out panel to tune lambda. Require finite gradients,
correct loss signs, masked-token alignment, FP32 score equivalence against
full-logit reference calculations, a preference-only repeated-batch learning
check, and saved/resumed equivalence. Freeze exact code/config/model/processor
and dataset hashes before launching the pilot.

## Measurements and decision gates

At initialization and updates 16/32/48/64, score the same 256 validation pairs
with dropout disabled. Log preference loss, raw winner ranking accuracy (ties
count one half), relative margins and quantiles, chosen/rejected action NLL,
winner response CE, and whole-response rankings. Reference-relative signs
start at ties, so they are not a standalone classification-accuracy baseline.
Report task-clustered uncertainty for treatment-minus-control differences.

Keep changing-batch training losses separate from fixed-panel scores. Monitor
per-term gradient norms/alignment on a fixed training probe. Save LoRA,
optimizer, RNG, data cursor, reference cache identity, and worker status; the
controller must own all stages and persist failures immediately.

Predeclare the endpoint as primary. As a pragmatic screening gate, seek at
least +5 percentage points of held-out ARM ranking agreement over matched SFT,
lower held-out preference loss, and a positive paired signal across tasks,
without more than 5% relative degradation in winner response CE. These are
screening thresholds, not a task-success or statistical-significance claim.
Inspect whether gains are concentrated in a few sites/actions.

If training pairs improve but validation does not, stop scaling and investigate
label quality/generalization. If neither improves, treat it as an objective or
implementation failure. Only if the gate passes propose a fresh fixed-100
browser comparison of treatment and matched SFT, retaining historical base/1A
as context. Require browser evidence before considering all 300 tasks.

## Compute and sequencing

Finish the CPU audit, pair manifest, tests, and executable stage controller
before obtaining more compute. Provisionally, two H200s for 2–3 hours could
cover reference scoring, calibration/rehearsal, concurrent 2,048-state arms,
and fixed-panel diagnostics. This is 4–6 H200-hours and excludes browser
evaluation. It is an estimate, not a submitted job or an approved request;
replace it with a throughput estimate from actual retained sequence lengths.

Keep online RL as a separate proposal. Best-of-N selected actions do not have
the original actor's behavior distribution, and unexecuted alternatives have
no observed transition/terminal return. Any later RL integration must specify
how it handles selection bias and ARM reward error rather than assigning
terminal-success advantages to every candidate.
