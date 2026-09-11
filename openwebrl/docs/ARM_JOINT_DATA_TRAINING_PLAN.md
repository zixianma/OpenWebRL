# Joint C2 and Piotr teacher-data training proposal

2026-09-10. User approved starting the combined-data SFT and full-response
DPO-only in parallel, retaining the default history format, with automatic
full-300 OM2W evaluation. **Approved budget: SFT 2 H200s x 3 hours; DPO
2 H200s x 5 hours, including evaluation (16 H200-hours maximum).** The
two-GPU launch specification below supersedes the earlier resource options.
Data preparation is complete: [review report and interactive gallery](ARM_JOINT_DATA_REVIEW.md).
The current review revision has 5,540 train / 658 validation pairs. It uses
stricter coordinate ambiguity quarantine and excludes conflicting draw IDs
found in the full Piotr inventory. The user reviewed the retained examples.
This proposes expanding the data in the
[controlled preference viability test](ARM_PREFERENCE_V2_VIABILITY_PLAN.md).
The numerical and engineering gates in that plan still apply.

## Approved two-GPU execution

**Completed (2026-09-11 08:43 UTC):** both runs finished 174 updates and
fresh full-300 OM2W evaluations, then released their allocations automatically.

| Run | Overall | Valid-only | Unavailable | Allocation elapsed / GPU-hours |
| --- | ---: | ---: | ---: | --- |
| SFT job 287477, g006 | 102/300 = 34.0% | 102/270 = 37.8% | 30 | 1:53:34 / 3.79 |
| DPO job 287447, g003 | 104/300 = 34.7% | 104/254 = 40.9% | 46 | 3:22:06 / 6.74 |

Both have zero missing task results. DPO versus SFT has 32 wins / 30 losses
on all 300 tasks (exact paired p=0.899), so this run does not establish an
advantage for either objective. [Final comparison](ARM_JOINT_SFT_VS_DPO_RESULTS.md).
[Training diagnostics and checkpoint progression](ARM_JOINT_TRAINING_MONITOR.md).
The paired report was generated from the explicit r2 roots because the original
controller's cross-run lookup still names the superseded r1 root; individual
endpoint aggregation and protocol validation completed automatically.

Final resumable adapters and optimizer/RNG state are under each active run's
`student/update-000174`; intermediate checkpoints 0/44/50/87/131 are retained.
Merged evaluation actors are under `evaluation/merged-endpoint`. Active roots:

- SFT: `/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/joint-v2-sft-2gpu-r2`
- DPO: `/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/joint-v2-dpo-2gpu-r2`

No retry, further training, or new allocation was submitted after completion.
The failure and recovery history below is retained for future diagnosis.

**Current status / recovery (2026-09-11 ~05:50 UTC):** the jobs are separate.
SFT job 287446 ended after 2m49s with zero optimizer updates: the deterministic
longest-sequence rehearsal called `actor.train(False)`, which also disabled
Transformers activation checkpointing and caused a roughly 139-GiB OOM. The
old controller incorrectly returned success after the child failed. Both bugs
are corrected: backward always keeps training/checkpointing enabled while
dropout modules are toggled separately; controller failure exits nonzero.
The regression test passes. The corrected DPO rehearsal has been observed at
roughly 31–33 GiB per GPU; its complete GPU gate report remains authoritative.

DPO remains in **its original job 287447 on g003**. Its original batch owner
is temporarily paused while a recovery guardian owns a replacement controller
in the same allocation. The guardian resumes the original owner only after
replacement work ends, preventing premature allocation cancellation. No new
DPO allocation was requested. 2,768 cached pairs per rank were migrated with
unchanged base/data checks and recorded hashes; remaining scoring completed.
The active root is now `arm-reproduction/runs/joint-v2-dpo-2gpu-r2`, with
[recovery config](arm_results/joint_data_v2/dpo-2gpu-r2-config.json).

A separate SFT replacement is prepared under `joint-v2-sft-2gpu-r2`, with
the previous 472 validation reference scores preserved. It has its own
[batch launcher](../../scripts/run_arm_joint_sft_2gpu_r2.sbatch),
[configuration](arm_results/joint_data_v2/sft-2gpu-r2-config.json), W&B identity,
logs, checkpoints, and evaluation. **Replacement explicitly approved and submitted:** job **287477 on g006**,
started 2026-09-11 06:17 UTC (Sep 10 23:17 PDT), with a three-hour limit
ending around 09:17 UTC. Resources: 2 H200s x 3 hours, 16 CPUs / 240 GiB,
full-300 evaluation included (maximum 6 GPU-hours). The user approved the
fixed SFT replacement and requested close monitoring. Its W&B run is
[958a08ac](https://wandb.ai/zixianma/openwebrl-arm/runs/958a08ac). A batch
supervisor owns all stages and retains a failed pipeline for a bounded
15-minute repair window. The SFT and DPO jobs are not merged.

SFT replacement startup verified at 2026-09-11 06:24 UTC: all GPU gates
passed, including longest-sequence backward and optimizer/RNG replay
(maximum adapter error 4.91e-6). Repeated-batch CE decreased from 0.196999
to 0.195785; rehearsal weights were discarded. Update 1 completed with CE
0.196999, finite gradient norm 0.17346, and 15.8 seconds/update. W&B is
syncing. Both runs have a persistent ten-minute metric watcher, with closer
manual checks around startup and stage transitions.

The corrected DPO GPU gate report subsequently passed: repeated-batch loss
fell from 0.69314718 to 0.68161750; four calibration gradient norms were
3.136–3.234; serialized optimizer/RNG replay adapter error was 9.70e-6
(within the predeclared 2e-5 tolerance). Longest-sequence backward and native
parsing passed. A separate startup failure exposed an unavailable
`wandb.util.generate_id` helper before update 1. Durable IDs were pre-created
with `secrets.token_hex(4)` for both replacement runs, using the existing
identity-file branch without changing the training config. Failed zero-update
workers blocked in distributed teardown were stopped; the recovery guardian
retained the allocation and retried with the cached references and passed
rehearsal gates. No new DPO allocation or reference recomputation was needed.

Submitted 2026-09-10 22:20 PDT (2026-09-11 05:20 UTC); both allocations
started immediately: **J-SFT job 287446 on g001**, ending by 2026-09-11
01:20 PDT (08:20 UTC); **J-DPO job 287447 on g003**, ending by 2026-09-11
03:20 PDT (10:20 UTC).
These are allocation limits; controllers release resources after their work.
Initial Slurm state was RUNNING for both. Model startup/gates are distinct
from completed training updates and will be recorded in each run's status.

Both models start independently from the original OpenWebRL SFT actor. Each
uses two data-parallel workers, microbatch one state per GPU, accumulation 16
per worker, **global batch 32 and 174 updates**. The final global batch of
four states is split two per worker and normalized without duplicate padding.
Reference scores and fixed offline panels are sharded across both GPUs.

| Run | GPUs | Time limit (training + evaluation) | CPU / host memory | Maximum GPU-hours |
| --- | ---: | ---: | --- | ---: |
| J-SFT | 2 H200 | 3 hours | 16 CPUs / 240 GiB | 6 |
| J-DPO | 2 H200 | 5 hours | 16 CPUs / 240 GiB | 10 |

User authorization: after the exact 2x3-hour / 2x5-hour proposal, the user said
“ok just do it after you're ready.” No additional allocation is implied.

Each controller reserves 65 minutes before its five-minute shutdown margin
for merge and automatic evaluation of update 174. It starts two independent
actor replicas on the merged endpoint, each evaluating 150 disjoint OM2W tasks
with 12 browser workers. Both results must complete before aggregation and
allocation exit. Training checkpoints remain 44/50/87/131/174; offline
diagnostics run at 0/44/87/131/174. Full-300 evaluation happens only at the
predeclared endpoint. Interrupted training or evaluation remains resumable.

- [Shared distributed trainer](../../scripts/train_arm_joint_ddp.py)
- [Training/evaluation controller](../../scripts/run_arm_joint_pipeline.py)
- [SFT batch script](../../scripts/run_arm_joint_sft_2gpu.sbatch)
- [DPO batch script](../../scripts/run_arm_joint_dpo_2gpu.sbatch)
- [Frozen SFT config](arm_results/joint_data_v2/sft-2gpu-config.json)
- [Frozen DPO config](arm_results/joint_data_v2/dpo-2gpu-config.json)
- [Frozen evaluation shards](arm_results/joint_data_v2/om2w-shards.json)

All 17 CPU tests passed, including real FP32 DPO gradient signs, sequence-sum
normalization, paired-data/token checks, distributed final-batch exposure, LR
schedule, and 300-task shard coverage (2.25 s wall, 1.50 CPU-seconds, 509 MiB
peak). Shell/Python syntax and frozen code hashes passed. Model-level GPU gates
are part of startup: direct native parsing, processor/image-grid agreement,
selected/full-logit agreement, longest-sequence backward, 128-state training
probe, repeated-batch learning, and serialized optimizer/RNG replay. Rehearsal
weights and optimizer state are discarded before actual training. Beta 0.1 is
the common initial DPO setting; failed gates stop rather than silently change
the objective or select a beta using OM2W outcomes.

Run roots are `arm-reproduction/runs/joint-v2-sft-2gpu` and
`arm-reproduction/runs/joint-v2-dpo-2gpu` under the runtime directory. Each
contains `status.json`, `training-status.json`, `training.log`, W&B identity,
reference caches, `student/metrics.jsonl`, resumable checkpoints, and separate
evaluation shard directories. The controller writes result markdown in
`openwebrl/docs/` and overall/valid-only JSON summaries after completion. The
later-finishing run also produces the paired SFT/DPO comparison.

## Current SFT launch preparation (2026-09-10)

- **5,540 training winners, effective batch 32, one pass = 174 updates.**
  The final batch contains four examples, normalized by four, with no padding
  or duplicated states. The 658 validation examples and 5,174 reserve examples
  are not used for training.
- Start from the original local `OpenWebRL-4B-SFT`, with language-only LoRA
  rank 16 / alpha 32 / dropout 0.05, frozen vision weights, LR `1e-5`, and the
  AdamW settings below. Warmup is 512 states / 16 updates; the half-cosine
  endpoint is exactly `5e-6`. The old 1A schedule ended slightly above half
  peak because its horizon counted the warmup differently; this run explicitly
  implements the joint plan's post-warmup half-cosine definition.
- Save updates **44, 50, 87, 131, 174**, including optimizer, RNG and cursor.
  Update 50 matches 1,600-example exposure from the earlier scaling comparison.
  Endpoint 174 is primary, fixed before looking at metrics.
- At updates **0, 44, 87, 131, 174**, evaluate the fixed 472-pair diagnostic
  panel (256 C2 + 216 Piotr), logging winner CE and full-response/action-token
  ranking separately by source. Per-pair scores are retained. These are offline
  teacher-preference diagnostics, not OM2W task-success evaluations.
- W&B project `openwebrl-arm`, group `arm-joint-v2`, run `joint-v2-sft`;
  local metrics remain under the runtime run directory.

Prepared files:

- [Frozen configuration](arm_results/joint_data_v2/sft-config.json)
- [Trainer](../../scripts/train_arm_joint_sft.py)
- [One-H200 batch launcher](../../scripts/run_arm_joint_sft.sbatch)
- [CPU exposure-contract checks](../../tests/test_arm_joint_sft.py)

Four CPU tests and shell/Python syntax checks passed. Startup on the approved
GPU must pass actual image-processor/token-grid checks, direct native parser
checks, selected-logit versus full-logit agreement, finite gradients, and a
longest-training-sequence memory check. No GPU tests have run yet. Adapter
serialization is checked when saving; numerical optimizer-resume equivalence
is not yet established by the CPU cursor test.

Proposed SFT request: **one H200, four CPUs, 120 GiB RAM, four hours**
(maximum **4 H200-hours**), covering startup checks, training, checkpointing
and offline validation. This excludes browser evaluation. Historical 1A
training took 2h53 for 8,394 examples; linear scaling suggests about 1h54 for
5,540 before the new validation overhead, with sequence-length/implementation
uncertainty. Four hours is a planning envelope, not measured joint-run timing.

Submit only after explicit approval of this exact budget:

```bash
sbatch scripts/run_arm_joint_sft.sbatch
```

To resume after a separately approved allocation, use the same launcher with
`--resume`; it restores the run's durable checkpoint and validates the frozen
configuration. Do not regenerate the config during an interrupted run.

Current recommendation for a concurrent second run: **full-response DPO-only**
on the exact same 5,540 frozen pairs, same order and one-pass exposure, initialized
from the original SFT actor with that actor as frozen reference. No auxiliary
SFT term and no initialization from the new J-SFT endpoint. Action-masked
DPO-only is the subsequent third arm. A separate shorter SFT allocation and
longer DPO allocation avoid holding an idle SFT GPU while DPO finishes. DPO
implementation/calibration and its exact budget remain pending; the SFT
launcher does not silently start preference training.

### Expanded option: both training jobs with automatic OM2W evaluation

The user asked about this larger scope after the SFT-only request; no expanded
budget is approved. Recommendation: two independent one-H200 allocations,
each controller owning training, offline diagnostics, merge, and endpoint
evaluation before exit. Both train for 174 updates on matched data. Evaluate
only the predeclared update-174 endpoints automatically; quarter checkpoints
receive the offline panel diagnostics, not repeated OM2W selection sweeps.

| Scope per policy | J-SFT allocation | J-DPO allocation | Total GPU-hour ceiling |
| --- | --- | --- | ---: |
| Fixed 100-task OM2W endpoint evaluation | 1 H200 x 4 hours | 1 H200 x 7 hours | 11 |
| **All 300 OM2W tasks per endpoint (recommended expanded option)** | **1 H200 x 5 hours** | **1 H200 x 8 hours** | **13** |

Each job would request eight CPUs and 120 GiB host RAM to support the browser
handoff. Separate allocations let the SFT GPU be released as soon as its
evaluation finishes, rather than holding it for DPO. If both start together,
the planning envelope is eight hours elapsed, plus queue/preparation time.

Timing evidence: 1A completed 262 update intervals in 2.881 hours (39.59 s per
update on one H200). The final previous preference attempt completed 130
intervals in 1.746 hours on two H200s (48.34 s/update, 32 pairs globally).
Scaling that latter throughput to 174 updates on one H200 gives about 4.67
hours before new diagnostics/calibration. It is only an estimate because the
new objective, sequence lengths and reference-cache strategy differ. Reserve
roughly another 1–1.5 hours for its new offline checks/validation. The latest
100-task evaluation took about 25 minutes at concurrency 12; the two-policy
200-task evaluation completed in 58:28 on two GPUs. Budget 1.25–1.75 hours
per 300-task evaluation including merge/startup, allowing for live-site tails.

Use fresh rollouts for both endpoints: these checkpoints have never been
evaluated, so previous models' 100-task results cannot be reused. Preserve the
ARM-comparison protocol (one candidate, no inference-time ARM, 30 turns,
full textual history, current screenshot, o4-mini/AgentTrek judge), report
overall and valid-only rates, availability and paired task outcomes, and keep
any retry pass separate. Judge API charges are additional to GPU-hour budgets.
These comparisons occur at different times if each endpoint evaluates as soon
as training completes; record timestamps and do not claim contemporaneous
baseline control. No fresh base-model run is included in this budget.

The DPO implementation and automatic evaluation handoff still require completion
and checks before submission. Neither expanded job has been submitted.

## Recommendation

Use the full cleaned C2 pool rather than a 2,048-state training cap, reserving
task-disjoint validation. Add cleaned OpenWebRL teacher-selection examples
from Piotr's release. Latest discussion sequence: run combined-data winner
SFT first, followed by full-response DPO-only and action-masked DPO-only.
All three should use one frozen paired dataset. Proposed default: each starts
from the original actor; SFT runs first in time as a baseline. Starting the two
DPO branches from the new SFT checkpoint instead would test a sequential
SFT-then-preference recipe with extra data exposure; resolve this distinction
before launch. No SFT+DPO hybrid is in the current proposed run set.

The 5,490 figure is an eligible-state count, not a frozen pair count. Each
state still has one or more possible negatives. It excludes coordinate-only
differences within five normalized units and all done-versus-done alternatives.
Use one chosen/rejected pair per retained state, after semantic checks and
validation exclusion. Do not train on held-out tasks or expand every candidate
alternative into an independently weighted example.

## What was verified in Piotr's release

Source: [PTeterwak/action-reward-models-data](https://huggingface.co/datasets/PTeterwak/action-reward-models-data).
The API currently reports revision `13905e69ef10169167eab388533997008694bd6d`.
The Git object IDs and sizes for the four raw OpenWebRL JSONL files are
identical to our cached revision `0d83b48c1659cac47a1044ef88fb573d3c16e180`.
Pin the latter for reproducibility; do not silently change revisions.

| OpenWebRL asset | Verified observation |
| --- | --- |
| `states_full.jsonl` | 3,085 states, 412 episodes, 397 normalized task texts |
| `labels.jsonl` | 39,997 labels across 2,999 base-state IDs; median/max 16 per state |
| `candidates_merged.jsonl` | Sampled actor responses; inspected record has five candidates, temperature 0.7, and a draw-qualified state ID |
| `labels_drawlevel.jsonl` | Teacher selection and reasoning keyed by draw-qualified ID |
| Exact normalized task-text overlap with OM2W 300 | Zero; semantic overlap is not ruled out |

The count recheck streamed the cached states and labels at nice 15, took less
than one second wall time, and peaked at 10.5 MiB RAM. Remote inspection read
metadata, README, and one record from each raw JSONL; it did not download the
whole candidate archive or screenshots. Full candidate/label joins and
cross-source C2 deduplication remain pending.

The release's [state extractor](https://huggingface.co/datasets/PTeterwak/action-reward-models-data/blob/13905e69ef10169167eab388533997008694bd6d/code/data_generation/openwebrl_actor/extract_states.py)
uses SFT trajectories and retains the original actor prompt. These contexts
may already be familiar to the starting SFT actor. The new supervision is the
GPT-5.5 comparison among sampled actions. It is neither a measured outcome
for the rejected action nor a calibrated action-quality score.

## Data conversion and split

1. **Use only `openwebrl_actor/` initially.** The MolmoWeb data use another
   actor/prompt/action format, and the published distillation example references
   screenshots outside the release. Adding it would require a separate audit.
2. **Recover actor targets, not selector outputs.** Join
   `states_full.jsonl` to `candidates_merged.jsonl` and
   `labels_drawlevel.jsonl`. The inspected stored selection is zero-based;
   the selection-model output contract is one-based. Assert bounds and verify
   the conversion against the builder. Do not join repeated draws through
   `labels.jsonl`, which omits the draw suffix. Never train the actor to output
   the selector's `{"selection": N}` or append the teacher's judging rationale.
3. **Apply the same action checks as C2.** Preserve original actor reasoning,
   tool calls, context, image order and coordinate conventions. Validate schema,
   token masks, truncation, screenshot mapping, near-coordinate ambiguity,
   and action-equivalent alternatives. Exclude done-versus-done preferences,
   while keeping eligible done-versus-continue comparisons and selected-answer
   SFT supervision on retained states.
4. **Use repeated draws to inspect coverage, not as a mandatory consensus
   filter.** Each draw has different candidates. Even a perfectly consistent
   teacher can choose different winners because its preferred action is absent
   from some draws. The earlier three-draw/60%-modal filter is withdrawn as a
   default. Report action diversity, winner frequencies and comparisons where
   the same alternatives co-occur; marginal winner frequency also reflects
   actor sampling frequency. It is not calibrated teacher confidence.
   Do not require SelectionARM/ScalarRM agreement: these models were trained on
   this release, so agreement is not independent evidence and adds scoring cost.
5. **Select one pair per base state for the first matched comparison.** Among
   structurally and semantically eligible draws, choose a draw by fixed hash.
   Keep winner and loser from that same comparison. Among eligible distinct
   loser action groups, choose one by a separate fixed hash. This supersedes
   model-hard negative selection: using action likelihood to choose the data
   would favor the action-masked objective and introduce another selection
   choice. No model scoring is needed to choose negatives for this comparison.
   Preserve all source/draw IDs and selection provenance. A model-hard loser
   is not a verified incorrect action.
6. **Split both sources together.** Group episodes sharing normalized task
   identity across sources, resolve flagged near-duplicate tasks, and remove
   exact duplicate states by context/image identity. Reserve about 10% of task
   groups using a fixed seed before sampling pairs. No episode or duplicate
   task can cross train/validation. Review likely OM2W overlaps before inclusion.
   Freeze up to 256 validation pairs per source; use all eligible held-out pairs
   if a source supplies fewer. This is a new-training holdout, not proof those
   tasks were unseen by the pretrained actor or ARM.

This could yield at most 5,490 + 2,999 = 8,489 states before quality filters,
deduplication and holdout. Actual training counts must come from the manifest.
Review 64 examples across both sources with screenshots before freezing it.
Do not interpret a repeated winner as satisfying the 0.7 PRM quality floor
used in the separate MolmoWeb distillation experiment; that score is absent here.

## Joint training and controls

| Run | Data | Objective | What it establishes |
| --- | --- | --- | --- |
| J-SFT | Combined retained C2 + teacher states | Full-response winner SFT | Joint-data baseline |
| J-DPO | Identical states, order, pairs, exposure | Full-response DPO only | Preference learning without auxiliary SFT |
| J-DPO-Action | Identical states, order, pairs, exposure | Action-masked preference only | Effect of restricting scored tokens |

Start all models from `OpenWebRL/OpenWebRL-4B-SFT`, not a previously trained C2/1A
adapter. Use each training state once, source-interleaved at its natural
retained proportion. Avoid forced 50:50 oversampling and the earlier
equal-website/action sampler, which inflated terminal-answer frequency.
Record source, task and action distributions and actual token exposure.

Proposed shared configuration:

- Language-only LoRA rank 16, alpha 32, dropout 0.05; vision tower frozen.
- Effective batch 32 states; AdamW LR 1e-5, betas 0.9/0.95, epsilon 1e-8,
  weight decay 0.01, gradient clipping 1.0.
- One pass: `ceil(N_train / 32)` updates, with correct last-batch normalization.
  Warm up over 512 states, then half-cosine decay to half peak LR at the
  one-pass endpoint (full-cosine horizon twice post-warmup exposure).
- BF16 forward with FP32 log-probabilities, preference margins and loss.
  Standard DPO uses summed log-probability of the complete sampled assistant
  turn (reasoning plus tool calls and completion boundary), with the prompt
  excluded and the original SFT model frozen as reference. Do not silently
  substitute mean-token scores. J-DPO-Action uses the same reference-relative
  formula, but sums over tool-call spans and the common completion boundary.
  This is a masked surrogate, not standard full-response DPO. The prompt and
  reasoning tokens receive no direct preference scoring in that arm.
- Beta 0.1 is only an initial candidate; check its gradient/margin scale on
  training data because summed full-response scores differ from summed masked
  scores and the earlier mean-action scores. Neither preference arm has an
  auxiliary SFT term or a lambda to calibrate. Use a 128-pair training probe to
  check finite gradients, per-source update norms, and reference drift. Propose
  one common beta/LR/schedule for the first comparison and report the scale
  difference; changing normalization and masking together would confound it.
  This is a fixed-recipe comparison, not a claim each objective is optimally tuned.

Freeze manifests, reference scores, loss/config definitions and seeds after
the smoke/rehearsal checks, then reset training state for all runs. Save
resumable checkpoints and score fixed panels at initialization and roughly
25%, 50%, 75%, and 100% of training-state exposure. Only J-SFT receives winner
SFT supervision. All see the
same states and candidate pairs where applicable. Extra rejected-response
compute is reported separately.

## Evidence and next decision

Report fixed-panel ranking, winner response CE, chosen/rejected action NLL,
relative margins, and task-clustered uncertainty separately for C2 and the
direct-teacher source. A pooled gain must not conceal a regression in either.
Use the v2 endpoint screening gate (+5 pp ranking versus matched SFT, lower
preference loss, no >5% relative winner-CE degradation) as a heuristic; it is
not an OM2W success claim. Online evaluation must also inspect scrolling loops,
termination and trajectory length, alongside overall and valid-only success.

Teacher-release validation measures reproduction of stored teacher choices.
It cannot independently validate the ARM trained on those choices. Improved
agreement must ultimately translate to fresh browser success.

This comparison isolates the objective **conditional on joint data**.
It does not isolate the data benefit relative to C2. A subsequent matched
C2-only SFT comparison should hold total updates/exposure fixed by sampling
C2 states with replacement and report unique-state coverage; that measures
the joint recipe against extra C2 exposure. A random-target control answers
a different question: whether teacher selection matters beyond extra self-SFT.
Neither additional data control is proposed for the initial objective comparison.

My expectation is that direct-teacher data is a more promising change than
more epochs or increased rank on the current C2 labels. Preference adds a
plausible but unproven benefit, making J-SFT essential. There is no defensible
numerical probability of OM2W improvement from the current evidence.

Finish the data audit and pair/token verification first. Run SFT first as
requested, then the two preference arms can run concurrently on two GPUs. Set
the exact GPU-hour request using retained token lengths and a measured
throughput estimate; the previous 2,048-state budget should not be reused for
this larger dataset. No new rollout collection is required for preparation.

## Why action masking is an explicit comparison

Action-level feedback identifies the preferred next turn. It does not require
the training loss to score only tool-call tokens. Our original motivation was
to avoid rewarding unverified reasoning prose and focus on executable decisions.
However, masking scores the action conditional on the stored candidate's own
reasoning, `log p(a | h, r)`. That can improve without making the model generate
the useful reasoning `r` at deployment. Masking also leaves gradients through
the reasoning-prefix hidden states; it does not freeze reasoning behavior.

Full-response DPO scores `log p(r, a | h)` and provides the conventional
preference-only baseline. It can also learn spurious reasoning or length
correlations, so log response lengths, reasoning/action score contributions,
and browser outcomes. Neither masking choice is established as superior here.
The full-versus-masked comparison holds auxiliary SFT absent in both arms and
keeps sum normalization. The smaller set of scored tokens still changes
gradient scale, which must be reported rather than interpreted as purely
semantic credit assignment.

Source: [DPO paper, objective and preferred-SFT control](https://arxiv.org/html/2305.18290v3).
Neither actor preference objective is Piotr's scalar value-head RM training.

## Proposed filtering contract for the three-run comparison

This section makes the discussion pipeline concrete; it has not yet been
implemented on the full Piotr candidate release. Preserve immutable raw data
and write exclusion reasons and source counts at every stage.

1. **Normalize provenance, preserve training text.** Map both sources to
   task/episode/state/draw IDs, original actor prompt, ordered screenshots,
   original candidate responses, selected index and label source. C2 must pass
   the executed-winner/source-hash joins and existing valid-success trajectory
   check. Piotr data must pass the draw-qualified teacher join. Mark its
   candidate outcomes unobserved; do not invent the C2 terminal-success filter
   for a source without those outcomes.
2. **Group and split globally.** Identify shared tasks across both sources,
   deduplicate identical prompt+image states, flag near duplicates, and exclude
   confirmed OM2W task matches. Identical screenshots alone do not establish
   identical tasks. Put connected task/episode groups wholly in train or
   validation, using fixed seed 42 and about 10% held-out groups. Conflicting
   preferred actions for an identical state go to review rather than silently
   choosing a source. Freeze split membership before model-based diagnostics.
3. **Validate each candidate.** Require reconstructable image context, a valid
   selection index, parseable schema-valid tool calls, and an untruncated actor
   completion within the frozen model context limit. Verify native parser and
   tokenizer mask alignment. Missing finish metadata in the HF release is an
   unknown, not evidence of a normal stop; detect malformed/truncated structure
   and inspect the generation limit/provenance before declaring a pair usable.
   Canonicalize actions for comparison only, preserving raw response text for
   training. Do not lowercase URLs, typed strings, or answers.
4. **Remove non-informative pairs.** Drop identical executable actions despite
   different reasoning, duplicate loser action groups, and every done-versus-
   done pair in this first recipe. Provisionally drop coordinate-only changes
   within five normalized units when all other arguments match. Distances
   above five do not prove different targets. Flag differences only in scroll
   amount, wait duration, or clicks on an apparently identical element for
   semantic review; unresolved flagged pairs are quarantined in all three arms.
   Keep other valid alternatives in the same state if one is excluded.
5. **Review label plausibility without fabricating a quality score.** Inspect
   64 examples (32/source), covering clicks, typing, scrolling, terminal actions,
   multiple calls and flagged categories, plus resolve flagged cases needed for
   inclusion. Check visible grounding and obvious conflicts between reasoning
   and action. A 64-example review diagnoses systematic problems; it cannot
   certify every remaining pair. Turn systematic findings into explicit rules
   and re-audit all rows. No mandatory modal-winner threshold, scalar-score
   cutoff, ARM agreement requirement, or paid rejudging in the initial recipe.
6. **Choose and freeze one pair per state.** Hash-select a usable draw and
   then a distinct loser action group from that same draw. Preserve the original
   teacher choice. This gives repeated Piotr draws no extra per-state training
   weight while retaining them for later coverage experiments. Keep the raw
   alternatives and exclusion ledger so this reduction is reversible.
7. **Export matched views.** `joint_pairs` is the master manifest. The SFT
   view contains exactly its prompts and chosen responses. Both preference
   views contain exactly its chosen/rejected responses; masks alone differ.
   Keep valid winners without a usable negative in a separate SFT-only reserve,
   excluded from the first matched SFT baseline. In particular, discarding
   done-versus-done preferences must not blanket-delete every terminal state:
   retain done-versus-continue pairs and report terminal-action coverage.

Use all final training states once, at natural source/action proportions.
Do not repeat Piotr data to force 50:50 mixing or balance websites by equal
sampling. The 5,490 C2 count is a starting pool: semantic review, duplicate
resolution, context validation and task holdout can reduce it further.

The prelaunch report must show source/state/task/draw counts, removal reasons,
pair types, winner action mix (especially scroll/wait/done), source mixture,
prompt/response/action-token length percentiles, image availability, and split
overlap checks. Record dataset revision, source hashes, pair-selection seed,
normalization rules and all mask definitions. If filtering removes useful
coverage, discuss that measured tradeoff before changing the recipe.

## Upstream data explanation and limits

The relevant upstream code is also archived in the pinned
[Hugging Face code tree](https://huggingface.co/datasets/PTeterwak/action-reward-models-data/tree/13905e69ef10169167eab388533997008694bd6d/code/data_generation/openwebrl_actor).

- `extract_states.py` takes pre-action prefixes from released SFT episodes,
  renders the actor chat template with tools, and defaults to retaining the
  last screenshot. The demonstrated action is reference metadata; its text is
  truncated to 2,000 characters by the extractor and must not be silently used
  as a full imitation target.
- `sample_candidates.py` samples five responses per draw from the SFT actor.
  The code uses top-p 0.9 and supports mostly temperature 0.7 plus an optional
  temperature-1.0 subset. The actual release mixture needs a full candidate
  inventory; a single sampled record cannot establish it.
- `build_teacher_batch.py` asks GPT-5.5 to select a next action from context,
  screenshot and candidates. It skips sets with fewer than two distinct action
  strings and deduplicates identical action sets within a state. It does not
  execute alternatives to measure their terminal returns.
- `build_selection_sft.py` trains a judge to emit the selected index;
  `build_scalar_rm_data.py` constructs up to two chosen-versus-distinct-loser
  pairs per draw for a value-head RM. Neither implements actor DPO. Their
  random draw/pair holdouts must be replaced with grouped task holdouts here.
- The canonical selection prompt can include candidate thoughts. `VISION_NO_COT`
  suppresses the selector's output reasoning, not necessarily candidate
  reasoning in its input; candidate-thought removal has a separate flag.
  The OpenWebRL teacher builder imports an external `frontier_arbiter` prompt
  implementation, so exact historical input visibility needs archived requests
  or that dependency, not inference from the flag alone.

Piotr's [actor-distillation results](https://huggingface.co/datasets/PTeterwak/action-reward-models-data/blob/13905e69ef10169167eab388533997008694bd6d/code/actor_distillation/RESULTS.md)
report a positive offline selected-target SFT experiment in the **MolmoWeb**
setting, using separate GPT-5.5 PRM scores and a 0.7 quality floor. They also
report a negative online selection-distillation result with looping behavior.
These support investigating selection quality and actor behavior; they do not
establish that OpenWebRL action-masked DPO, or this joint dataset, will work.
