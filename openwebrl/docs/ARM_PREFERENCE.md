# ARM preference learning: experiments and audits

Preference-learning objectives, the first calibrated experiment, pair-quality audits, and the corrected viability design. The subsequent matched joint-data SFT/DPO runs are documented in [joint data](ARM_JOINT_DATA.md), with their completed evaluations in [ARM results](ARM_RESULTS.md). Historical proposals here do not authorize new compute.

## Contents

- [Calibrated ARM preference run 286384](#arm-preference-run-286384)
- [ARM same-state preference distillation plan](#arm-preference-distillation-plan)
- [ARM preference v2: lightweight CPU audit](#arm-preference-v2-cpu-audit)
- [ARM preference distillation v2: controlled viability test](#arm-preference-v2-viability-plan)

---

<!-- document:ARM_PREFERENCE_RUN_286384.md:start -->
<a id="arm-preference-run-286384"></a>
## Calibrated ARM preference run 286384

_Source record: `ARM_PREFERENCE_RUN_286384.md`. Dated entries retain their historical context._


Status: **training and fixed-100 evaluation are complete**.
Two H200s in the
user-provided four-hour allocation `286384` on `g001` completed all 131
updates. The handoff merged the endpoint, but its SGLang server failed during
startup because the evaluation environment did not expose `CUDA_HOME`; the
allocation then expired. The evaluation launcher was repaired, and the user
approved a separate one-H200, one-hour rerun. Job `286831` completed all 100
tasks on `g005` in about 25 minutes at rollout concurrency 12.

<a id="arm-preference-run-286384--frozen-experiment"></a>
### Frozen experiment

| Setting | Value |
| --- | --- |
| Starting actor and frozen reference | `OpenWebRL/OpenWebRL-4B-SFT` |
| Source | immutable 8,394-turn C2 dataset |
| Pair coverage | 8,363/8,394 (99.63%) have a distinct valid loser |
| Training subset | deterministic seed-42 prefix of 4,192 shuffled pairs |
| Loser | seeded distinct, parseable, non-truncated candidate from the same pre-action state |
| Objective | winner CE + length-normalized reference-relative DPO |
| Beta | 0.1 |
| Preference weight | **3.02186**, calibrated on four frozen training pairs |
| Calibration target | preference gradient norm = 20% of winner-SFT gradient norm at initialization |
| LoRA | language-only rank 16, alpha 32, dropout 0.05 |
| Batch | 16 pairs/rank, two DDP ranks, effective batch 32 pairs |
| Updates | 131; checkpoints at 33, 66, 99, and endpoint |
| LR | peak `1e-5`; 512 response-exposure warmup, then the 1A half-cosine schedule |
| Evaluation | existing fixed 100 OM2W tasks, one actor action, `o4-mini` AgentTrek judge |

The calibration measured SFT gradient norm 0.32303 and unit-weight preference
gradient norm 0.02138. Their ratio selected lambda 3.02186 for the declared
20% contribution.

<a id="arm-preference-run-286384--runtime-artifacts"></a>
### Runtime artifacts

```text
/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/arm-preference-calibrated-286384-r4
```

Key files are `frozen-config.json`, `pair-audit.json`,
`student/calibration.json`, `student/metrics.jsonl`, checkpoints under
`student/`, and evaluation output under
`evaluation/checkpoint-scaling-100/endpoint-000131/`.

W&B: [arm-preference-calibrated-286384](https://wandb.ai/zixianma/openwebrl-arm/runs/fd55eff3)

<a id="arm-preference-run-286384--fixed-100-evaluation"></a>
### Fixed-100 evaluation

| Policy | Overall | Valid-only | Unavailable |
| --- | ---: | ---: | ---: |
| Historical starting actor | 26/100 = 26.0% | 26/85 = 30.6% | 15 |
| C2 update 500 | 33/100 = 33.0% | 33/79 = 41.8% | 21 |
| 1A endpoint | 33/100 = 33.0% | 33/79 = 41.8% | 21 |
| **Preference endpoint 131** | **31/100 = 31.0%** | **31/86 = 36.0%** | **14** |

On common-valid tasks, the preference endpoint had 12 wins and 6 losses
against the historical base (81 tasks, exact McNemar p=0.238), 3 wins and 8
losses against C2 (76 tasks, p=0.227), and 6 wins and 10 losses against 1A (75
tasks, p=0.454). None establishes a difference. The preference policy is
directionally above the base and below both filtered-SFT controls.

| Policy | Mean / median steps | Terminated | Hit 30 steps | Scroll calls | Repeated primary action |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 17.18 / 14 | 52.8% | 41.6% | 13.7% | 62.0% |
| C2 update 500 | 14.24 / 10 | 65.1% | 27.9% | 7.9% | 59.4% |
| 1A endpoint | 15.67 / 12 | 55.7% | 33.0% | 10.0% | 59.0% |
| Preference endpoint | 16.84 / 12 | 59.8% | 37.9% | 12.1% | 61.2% |

The preference endpoint's behavior sits closer to the starting actor than C2
on trajectory length, step caps, scrolling, and repeated actions. Combined
with the flat preference margin, this does not justify an all-300 evaluation.

<a id="arm-preference-run-286384--completed-training-curve-diagnosis"></a>
### Completed training-curve diagnosis

The displayed preference loss increased from a mean of 0.69182 over updates
1–33 to 0.69288 over updates 99–131. Most of this visible movement is BF16
quantization. Reconstructing the loss in float from the logged relative margins
gives 0.693152 and 0.693193, respectively: an increase of only 0.000041. The
mean policy-minus-reference preference margin over the last 32 updates was
-0.00091 nats, so the run did not demonstrate generalizing winner/loser
separation even though winner CE fell.

These are losses on a different, previously unseen pair batch at every update,
measured before that batch's optimizer step. They are not a fixed validation
curve and therefore need not decrease monotonically. The run omitted the fixed
pair panel needed to distinguish within-pair learning from generalization.

The post-run action audit found a material data bug: 1,099/4,192 training pairs
(26.22%) used a rejected response with the **same normalized executable action**
as the winner, differing only in reasoning text. Two of the four calibration
pairs had identical executable actions; a third differed only by one click
coordinate. Consequently lambda 3.02186 was numerically calibrated but not
well informed about distinct action preferences. Whole-response mean log
probabilities also dilute the approximately 100-character action in responses
with median length around 306 tokens.

The next preference run must canonicalize and exclude action-equivalent pairs,
calibrate on a stratified sample, compute/log margins in float32, and log a
fixed held-out preference panel before using task-level evaluation.

<a id="arm-preference-run-286384--engineering-record"></a>
### Engineering record

Two startup attempts found memory problems before update 3 and remain in
separate audit directories. The final implementation passes only response
positions to Qwen3-VL's language-model head. This preserves the exact causal
response log probabilities while avoiding prompt-length-by-vocabulary logits.
Peak observed memory fell from 142.6 GB to about 39.5 GB per H200. Failed
startup artifacts are not training results and are excluded from evaluation.

<!-- document:ARM_PREFERENCE_RUN_286384.md:end -->

---

<!-- document:ARM_PREFERENCE_DISTILLATION_PLAN.md:start -->
<a id="arm-preference-distillation-plan"></a>
## ARM same-state preference distillation plan

_Source record: `ARM_PREFERENCE_DISTILLATION_PLAN.md`. Dated entries retain their historical context._


The next experiment is specified in the
[v2 controlled viability plan](ARM_PREFERENCE.md#arm-preference-v2-viability-plan). The
configuration and budgets below are historical proposals, not the next launch.

Status: **the first calibrated half-epoch run and fixed-100 evaluation are
complete**; see its [run record](ARM_PREFERENCE.md#arm-preference-run-286384). It scored
31/100, between the historical base at 26/100 and C2/1A at 33/100. Its pair
audit and loss instrumentation were inadequate, so do not scale this recipe or
run it on all 300 tasks. This document freezes the purpose and
comparison structure before inspecting preference-training results. Any GPU
allocation still requires a separate exact resource approval.

[ARM results dashboard](ARM_RESULTS.md#arm-results-dashboard) ·
[filtered-SFT results](ARM_RESULTS.md#arm-c2-ablation-1a-results) ·
[broader integration plan](ARM_INTEGRATION_PLAN.md)

<a id="arm-preference-distillation-plan--why-this-is-next"></a>
### Why this is next

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

<a id="arm-preference-distillation-plan--stage-1-candidate-pair-audit"></a>
### Stage 1: candidate-pair audit

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

<a id="arm-preference-distillation-plan--objective"></a>
### Objective

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

<a id="arm-preference-distillation-plan--frozen-training-comparison"></a>
### Frozen training comparison

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

<a id="arm-preference-distillation-plan--evaluation-and-promotion"></a>
### Evaluation and promotion

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

<a id="arm-preference-distillation-plan--fastest-resource-plan"></a>
### Fastest resource plan

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

<a id="arm-preference-distillation-plan--follow-on-data-experiment"></a>
### Follow-on data experiment

After preference distillation, audit
`PTeterwak/action-reward-models-data` into episode-disjoint, one-target-per-state
examples. Compare C2 plus selected targets against C2 plus an equal number of
random executable targets. Keep the 1A optimizer recipe. This is the next
highest-upside data experiment, while rank-32 LoRA and full language-tower
fine-tuning remain secondary capacity checks.

<!-- document:ARM_PREFERENCE_DISTILLATION_PLAN.md:end -->

---

<!-- document:ARM_PREFERENCE_V2_CPU_AUDIT.md:start -->
<a id="arm-preference-v2-cpu-audit"></a>
## ARM preference v2: lightweight CPU audit

_Source record: `ARM_PREFERENCE_V2_CPU_AUDIT.md`. Dated entries retain their historical context._


Completed 2026-09-10. The structural audit supports preparing the matched
preference-versus-SFT pilot, but the provisional pairs are **not training-ready**.
No model inference, GPU work, or allocation submission was performed.

<a id="arm-preference-v2-cpu-audit--retention-and-integrity"></a>
### Retention and integrity

Audited all 8,394 C2 winner turns from
`c2-full-282782-20260908T075414Z/training.jsonl`. Dataset SHA256:
`cb7c75df6a4824e9e653f6d913b0ae83268610966cd13dd13fc7314e9c667fe0`.

| Check / filter | States retained | Fraction of source |
| --- | ---: | ---: |
| At least one distinct schema-valid alternative | 7,140 | 85.1% |
| Also exclude coordinate-only differences within 2 units | 6,705 | 79.9% |
| Within 5 units | 6,247 | 74.4% |
| Within 10 units | 6,104 | 72.7% |
| Radius 5, also exclude every done-versus-done pair | 5,490 | 65.4% |

Coordinate thresholds use the stored coordinate system, not screenshot pixels.
For otherwise identical multi-call actions, distance is the maximum Euclidean
distance over corresponding coordinate arguments. A threshold excludes nearby
alternatives; it does not establish semantic identity or correctness.

The audit excluded 8,950 exact action-equivalent alternatives (26.7% of 33,560
alternatives inspected), including differences only in reasoning, and 4,167
repeated distinct alternatives within a state. These are candidate counts,
not counts of selected training pairs. Another 1,250 states had no distinct
schema-valid alternative. Four selected winners failed required-argument
validation; their source line/task IDs are recorded in the machine report.

Dataset/source/prompt hashes, execution-source joins, saved prompt-token IDs
and image grids, successful valid terminal labels, and absence of selector
fallback were checked. No provenance failures were recorded; the four failures
were winner schema checks. All 7,471 referenced image paths exist, with no
conflicting declared hashes. Image bytes were not hashed or decoded. Stored
terminal success is a trajectory label, not proof that each action is correct.

<a id="arm-preference-v2-cpu-audit--issues-revealed-by-the-audit"></a>
### Issues revealed by the audit

1. **Exact JSON inequality is insufficient.** A text-only skim of the 64-example
   review manifest found final-answer paraphrases and coordinate differences
   that could still target the same element. Screenshot semantic review remains
   pending. At radius 5 there are 3,297 done-versus-done alternatives; removing
   them all loses 757 states, leaving 5,490 states across 1,105 tasks. Some such
   pairs contain genuinely different answers, so excluding all is a conservative
   first-pilot choice, not a claim that all are equivalent. Retain done-versus-
   continue candidates when otherwise eligible; winner SFT can still supervise
   the selected final answer.
2. **The provisional sampler changes the action distribution.** Equal cycling
   across website/action groups produces 521/2,048 (25.4%) terminal-only winners
   in training and 64/256 (25.0%) in validation, versus 1,120/6,247 (17.9%) in the
   radius-5 pool. Use proportional action sampling, with explicit rare-group
   coverage, before freezing the experiment. Current website strata use the
   captured current URL, which can differ from the task's initial website.
3. **ARM preference is still a noisy label.** Alternatives were not executed.
   Schema validity and different actions do not imply the selected action is
   better. The audit establishes available data and structural exclusions,
   not that preference distillation will improve browser success.

The provisional radius-5 split contains 2,048 training states from 710 tasks,
256 validation states from 138 disjoint tasks, and 128 calibration states drawn
from training. The held-out task pool was assigned before sampling states.
These manifests contain eligible alternatives, not frozen model-hard negatives.
Rebuild them after the semantic exclusion and sampling decisions; recheck
per-split capacity after exclusions rather than assuming it from total counts.

<a id="arm-preference-v2-cpu-audit--resource-footprint-and-reproducibility"></a>
### Resource footprint and reproducibility

The corrected full scan took **102.9 seconds wall time, 16.7 CPU-seconds, and
119.3 MiB peak RSS**. It ran as one process at nice 15 with a 10 ms per-state
sleep, a 768 MiB address-space cap, and a 180-second CPU soft limit. It read
approximately 1.68 GB of JSON across dataset, source, and execution files.
No numerical libraries or model weights were loaded.

An earlier scan selected an empty documentation example of the tool-schema
tag and therefore produced invalid schema results. It cost another 12.3
CPU-seconds over 111.1 seconds wall time, peaking at 64.3 MiB. That output is
explicitly marked [invalid](arm_results/preference_v2_cpu_audit/INVALID.md);
all findings here use the corrected `r2` output. Resource figures describe the
audit processes, not every preliminary inspection command in the session.

The [audit script](../../scripts/audit_arm_preference_pairs.py) uses a
stdlib mirror of browser tool parsing plus schema validation. Five targeted
unit tests passed, including documented defaults, call ordering, coordinate
comparison, malformed inputs, and the empty-schema-tag regression. Native
parser parity and tokenizer-based action-mask alignment remain pending.

```bash
nice -n 15 python3 scripts/audit_arm_preference_pairs.py \
  --source-root /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-full-282782-20260908T075414Z \
  --output /tmp/arm-preference-v2-audit-recheck \
  --sleep-ms 10
python3 -m unittest tests/test_arm_preference_audit.py
```

<a id="arm-preference-v2-cpu-audit--artifacts-and-next-steps"></a>
### Artifacts and next steps

- [Complete machine report and manifest hashes](arm_results/preference_v2_cpu_audit_r2/audit.json)
- [Eligible states](arm_results/preference_v2_cpu_audit_r2/eligible-states.jsonl)
- [Provisional training states](arm_results/preference_v2_cpu_audit_r2/provisional-train.jsonl)
- [Provisional validation states](arm_results/preference_v2_cpu_audit_r2/provisional-validation.jsonl)
- [Provisional calibration states](arm_results/preference_v2_cpu_audit_r2/provisional-calibration.jsonl)
- [64 examples awaiting screenshot review](arm_results/preference_v2_cpu_audit_r2/review-64.jsonl)
- [Controlled viability plan](ARM_PREFERENCE.md#arm-preference-v2-viability-plan)

Before training: conservatively handle final-answer paraphrases, review visual
equivalence, rebuild representative disjoint samples, and verify image hashes,
native parser parity, and token masks. Frozen-base scoring, hard-negative
selection, gradient calibration, and matched training belong on a compute
node. No new compute budget has been requested or approved by this audit.

<!-- document:ARM_PREFERENCE_V2_CPU_AUDIT.md:end -->

---

<!-- document:ARM_PREFERENCE_V2_VIABILITY_PLAN.md:start -->
<a id="arm-preference-v2-viability-plan"></a>
## ARM preference distillation v2: controlled viability test

_Source record: `ARM_PREFERENCE_V2_VIABILITY_PLAN.md`. Dated entries retain their historical context._


Latest discussion: [joint C2 and direct-teacher data proposal](ARM_JOINT_DATA.md#arm-joint-data-training-plan)
would replace the 2,048-state cap with the full cleaned pool plus audited
OpenWebRL teacher states. Its latest revision proposes combined-data SFT
first, then full-response DPO-only and action-masked preference-only on
identical pairs. It also specifies the shared filtering contract.
It is a proposal, not a launched run.

Status: direction approved for preparation and discussion. No new GPU budget
has been approved. This plan supersedes the proposed next run in
[the original preference plan](ARM_PREFERENCE.md#arm-preference-distillation-plan).

The [lightweight CPU audit](ARM_PREFERENCE.md#arm-preference-v2-cpu-audit) is complete:
7,140/8,394 states have a structurally distinct alternative; a provisional
coordinate exclusion leaves 6,247. Excluding done-versus-done alternatives
as well would leave 5,490. Before freezing pairs, resolve semantic ambiguity
and replace the provisional sampler, which overrepresents terminal answers.
The current manifests are preparation artifacts, not approved training inputs.

<a id="arm-preference-v2-viability-plan--question-and-interpretation"></a>
### Question and interpretation

Can a corrected action-preference objective improve held-out ARM preference
agreement beyond winner-only SFT on exactly the same states and exposure?
The prior 31/100 endpoint is an engineering pilot, not a clean test of this
question: it had numerical issues, action-equivalent negatives, and a different
winner exposure from 1A. Its result does not establish that distillation fails.

<a id="arm-preference-v2-viability-plan--cpu-data-preparation"></a>
### CPU data preparation

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

<a id="arm-preference-v2-viability-plan--matched-two-arm-experiment"></a>
### Matched two-arm experiment

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

<a id="arm-preference-v2-viability-plan--calibration-and-engineering-gates"></a>
### Calibration and engineering gates

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

<a id="arm-preference-v2-viability-plan--measurements-and-decision-gates"></a>
### Measurements and decision gates

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

<a id="arm-preference-v2-viability-plan--compute-and-sequencing"></a>
### Compute and sequencing

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

<!-- document:ARM_PREFERENCE_V2_VIABILITY_PLAN.md:end -->

---
