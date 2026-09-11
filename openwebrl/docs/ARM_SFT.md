# ARM filtered SFT: collection, training, and ablations

C2 rollout collection, action-level filtering, original SFT, optimizer ablation 1A, and checkpoint-study procedures. These records preserve the progression from the superseded pilot to full-pool training. For completed policy comparisons, use [ARM results](ARM_RESULTS.md); for the later combined C2/Piotr recipe, use [joint data](ARM_JOINT_DATA.md).

## Contents

- [ARM-filtered action-level SFT pilot](#arm-filtered-sft-plan)
- [C2 full-pool collection and SFT run](#arm-c2-run)
- [Resuming C2 student training](#arm-c2-resume)
- [C2 ablation 1A run](#arm-c2-ablation-1a-run)
- [Next action-level filtered-SFT ablations](#arm-filtered-sft-ablations)
- [C2 checkpoint scaling evaluation](#arm-c2-scaling-eval)
- [C2 update-500/update-700 full evaluation](#arm-c2-full300-eval)

---

<!-- document:ARM_FILTERED_SFT_PLAN.md:start -->
<a id="arm-filtered-sft-plan"></a>
## ARM-filtered action-level SFT pilot

_Source record: `ARM_FILTERED_SFT_PLAN.md`. Dated entries retain their historical context._


Status: **2026-09-08: the user selected all ~2K tasks and requested C2 after inference evaluation.** The first run now collects from all **2091 deduplicated task prompts**, then trains on all eligible executed winners from valid successful trajectories. Allocation 282782 has ended. C2 saved 220/2091 completed task outcomes and 711 eligible turns from 123 successful trajectories, then paused cleanly. Student optimization has not started; collection must resume before full-pool training. No new allocation is authorized by this plan. [Live C2 configuration and artifacts](ARM_SFT.md#arm-c2-run).

[Inference results and retry recommendation](ARM_INFERENCE.md#arm-inference-results) · [Main integration plan](ARM_INTEGRATION_PLAN.md)

<a id="arm-filtered-sft-plan--objective-and-first-experiment"></a>
### Objective and first experiment

Distill useful ARM choices into the actor so that **one generated action per turn**, without an ARM, improves task completion. The current scalar inference gain is evidence that action selection helps; it is not evidence that an SFT student will retain the gain.

At each training state, sample five independent actor responses from the frozen SFT checkpoint, score/select with a frozen ARM, execute the winner, and record the exact pre-action context. Use one frozen teacher in the first experiment. The queued rule uses SelectionARM if it has more successes than ScalarRM on all 300 tasks and a positive success difference on their common-valid tasks; otherwise it uses ScalarRM. Both evaluations must be complete. This is a practical teacher choice, not a statistical-significance claim. Its decision and model manifest are saved before collection. Do not mix teachers in the first collection round.

Use the same candidate count, sampling, action syntax, normalized coordinates, history, screenshot processing, and response format as the inference reproduction. Freeze teacher and actor revisions for the entire collection round. Do not add state-memory or reward-rewriting changes to this experiment.

| Branch | Training targets | What it tests |
| --- | --- | --- |
| C0 | Unmodified SFT checkpoint | Standalone reference |
| C1 | All usable executed ARM winners from valid judged trajectories | Plain best-of-five action distillation |
| C2 | Usable executed ARM winners from successful trajectories only | Added terminal-outcome filtering |
| C3 | C2 plus a calibrated local ARM selection filter | Added action-level filtering |

**Decision from the user, 2026-09-08: start with C2.** Train one C2 student and compare its one-action policy with C0. C1 is a later diagnostic control if we need to measure the incremental effect of terminal filtering; it is not a prerequisite for this pilot. C3 is deferred. A C2-versus-C0 improvement would establish a gain from this combined recipe, but would not isolate ARM selection from terminal filtering or the effect of additional SFT.

C2 already has two selection stages: the ARM chooses one of five responses at each state, then the terminal success label decides which trajectories contribute training turns. **There is no additional score threshold inside a successful trajectory.** For a successful ten-turn trajectory with ten usable executed winners, C2 trains on all ten. It does not assert that every retained turn was necessary or locally optimal.

C3 adds only a further rejection step: for example, a calibrated scalar winner-versus-runner-up margin might retain six of those ten turns for loss. The other four remain in later turns' historical context but are not themselves training targets. C3 does not add another teacher, generate better candidates, or discover which action caused success. Its hoped-for benefit is removing weak local preferences; its risk is dropping necessary actions, recovery steps, or good actions with an equally good alternative. SelectionARM supplies no directly calibrated margin, making this extra criterion less straightforward.

**Most promising first recipe:** C2 with the strongest validated frozen ARM, five candidates per state, one fresh attempt per training task, successful trajectories only, every usable executed winner retained, and current-response SFT. This balances an observed selection benefit with an observed successful trajectory while avoiding an unvalidated confidence cutoff. That is a judgment about the first experiment, not a claim that C2 has already beaten the alternatives. If we later test C1 or C3, match token/update budgets, include a random retained-turn control for C3, and report task/host/turn-position coverage.

Likelihood assessment: a modest one-action gain is plausible but unestablished; retaining the full inference gain is less likely because policy errors change subsequent visited states. The incremental value of an extra confidence filter is uncertain and may be negative if it removes recovery, late-stage, or difficult actions. These are qualitative hypotheses, not measured probabilities.

<a id="arm-filtered-sft-plan--prepared-data-inventory"></a>
### Prepared data inventory

The local `webgym_filtered_popular_2102_cleaned.parquet` contains **2102 task prompts and metadata**, not a demonstrated per-action SFT dataset. The audit found **0 exact task-ID overlaps**, **0 exact normalized-instruction overlaps**, and **0 flagged same-host near matches** against the 300 Online-Mind2Web tasks. It removed **11 duplicate training host/instruction pairs**, leaving **2091** task candidates.

This is a heuristic screen, not proof of semantic separation or absence of original model-training contamination. **441 training tasks share a hostname with an evaluation task**; this pilot is not a domain-transfer test. All deduplicated prompts are included as requested; unavailable tasks are recorded, with no retrospective exclusion based on teacher success.

<a id="arm-filtered-sft-plan--historical-128-task-draft--superseded-by-the-full-pool-request"></a>
#### Historical 128-task draft — superseded by the full-pool request

**128 was an arbitrary small collection-pilot size chosen during preparation.** It was not taken from the ARM paper, a power calculation, a measured successful-trajectory yield, or a demonstrated SFT data requirement. It is not an approved final training budget. The associated 32/32 holdouts were also provisional. No live task-quality, reachability, difficulty, or action-type screen was performed before writing these files.

The historical draft files remain unchanged for inspection. Their unexecuted train/calibration/dev assignments are superseded by the user’s full-pool request; all their eligible task IDs now belong to C2 training:

- `train_pilot.jsonl`: 128 task prompts; no generated trajectories.
- `filter_calibration.jsonl`: 32 separate task prompts originally reserved for C3. C2 needs no confidence-filter calibration; these tasks are included in the full training pool.
- `policy_dev.jsonl`: 32 separate task prompts originally proposed for checkpoint selection. These tasks are included in the full training pool. The student uses the fixed epoch-2 checkpoint instead of choosing a checkpoint from this old dev split.
- `split-audit.json`: source hashes, normalization rules, overlap counts, split hashes, and caveats.

Directory: `/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/sft-preparation/`.

The actual sampling procedure was:

1. Start from the 2102 local task prompts. Normalize instructions with Unicode NFKC, case folding, and word-token whitespace normalization. Normalize hostnames by lowercasing and removing a leading `www.`.
2. Screen against Online-Mind2Web: exclude exact task IDs or normalized instructions, and flag same-host near matches with word-set Jaccard at least 0.6 or `SequenceMatcher` ratio at least 0.85. This screen found no such evaluation overlaps. Deduplicate training host/instruction pairs, removing 11 rows and leaving 2091 candidates.
3. Group by hostname. Sort tasks inside each group by SHA-256 of `arm-sft-v1:42:` plus task ID; order hostnames by the same prefix plus hostname. Take tasks round-robin across hosts until 192 are selected.
4. Independently sort those 192 by SHA-256 of `arm-sft-split-v1:42:` plus task ID. Assign the first 128 to training, the next 32 to calibration, and the last 32 to policy development.

There were 1397 distinct training hostnames, so all 192 selected tasks came from different exact hostnames. This is a breadth-oriented deterministic sample, not a representative random sample of the 2091 tasks and not a selection of the easiest or best tasks. Hostnames distinguish subdomains and language sites; they do not guarantee distinct companies or registrable domains. The zero-overlap heuristic also does not establish clean domain transfer from the broader training pool to Online-Mind2Web.

A spot check illustrates the missing quality screen: the training draft contains a Welsh Wikipedia task on `cy.wikipedia.org` and a Spanish AliExpress task on `es.aliexpress.com`; the dev draft includes a translation task on `lingvo.yandex.ru`. These are not automatically bad tasks, but the sampler did not validate their current availability, language mix, or suitability for the intended policy distribution.

<a id="arm-filtered-sft-plan--authorized-full-pool-collection"></a>
#### Authorized full-pool collection

Collect one fresh trajectory per task across all **2091** deduplicated prompts. The source has 1397 exact hostnames. Use deterministic hash ordering with seed 42 and host round-robin scheduling so an interrupted run covers many sites; task order is recorded in an immutable JSONL and audit. The only 11 exclusions are duplicate hostname/instruction pairs. The evaluation-overlap heuristic flagged none.

The old **128-task cap, 2000-turn cap, 250-update cap, and draft 32/32 holdouts are superseded**. Every eligible successful-trajectory turn enters the C2 dataset. Do not add a confidence threshold. Preserve completed failed/unavailable outcomes; resumption skips completed tasks. An interrupted, unjudged attempt is stored separately and may be restarted from a fresh browser when compute is assigned, without overwriting the interrupted trace.

Before collection, run a synthetic multimodal backward/save/reload smoke; it uses no benchmark data and produces no deployable student. Then collect the first eight tasks from the same full-pool order and check actual processor-expanded prompt tokens, image grids, and response masks on their exported turns. These tasks remain part of the collection, and their outcomes are not regenerated to seek success. Expand to concurrency 16 only after the export check passes.

Hold all 300 Online-Mind2Web tasks and their evaluation trajectories out of student training. Train for two fixed epochs and use the epoch-2 student; no checkpoint selection using Online-Mind2Web. Repeated design iteration on that benchmark still limits later confirmatory claims, so a separate final benchmark may be needed for a later study.

The existing allocation ends at 02:37:55 PDT; the C2 controller stops by **02:32:55 PDT** to leave cleanup time. Collection is resumable, and full-pool training cannot start from a partial collection. If collection completes with at least 15 minutes left, the controller unloads the verified evaluation servers and starts/resumes the student on the same GPU. Otherwise it records that training awaits assigned compute. Neither collection nor training requests or extends an allocation automatically.

<a id="arm-filtered-sft-plan--per-turn-export-contract"></a>
### Per-turn export contract

Export at collection time; do not infer training-ready turn records from concatenated rollout text. The current evaluation trace stores prompt/image hashes, split candidate reasoning/actions, scores, and selections. The sample dump stores the last turn's prompt/response plus historical screenshots. Those artifacts are useful for inspection but do not certify exact training inputs for every earlier turn.

Each training record needs:

- Task ID, split, attempt/trajectory ID, turn index, source revision and configuration hashes.
- Exact rendered **pre-action** actor prompt, processor-expanded prompt token IDs, screenshot bytes or content-addressed image path, image hash, and image-grid metadata.
- All five raw responses, token IDs, finish reasons, stable candidate IDs, teacher scores or selected index, fallback/parse status, and duplicate-action groups.
- The executed candidate's raw response and tokens, valid parsed tool call, execution status, current URL, and subsequent environment outcome.
- Terminal judge result and validity in a separate trajectory record joined only for filtering. Terminal labels, future screenshots, later actions, and ARM scores never enter the student prompt.

The SFT unit is **one selected turn**, while its complete prior context remains visible. Preserve necessary earlier actions even when they are not themselves retained for loss. Keep image resizing and normalized coordinates identical to collection. Reject incomplete prompt/image/response joins; do not invent missing observations or claim a saved prefix restores a live browser.

<a id="arm-filtered-sft-plan--filtering-rules"></a>
### Filtering rules

Apply basic data-integrity rules to all branches: correct split/provenance, exact prompt-image alignment, parseable supported action, finite scores where applicable, valid selected index, no selector fallback, and no truncated target. Distinguish an unsuccessful judged task from an unavailable run. C1 may retain usable turns from valid failed trajectories; C2 and C3 require a valid successful trajectory.

Use only **executed winners** in the initial dataset. Unexecuted candidates have a local preference label but no observed transition or terminal outcome. They can support a later preference-training experiment, not fabricated execution-success labels.

Do not apply a global threshold such as `scalar_score > 0`: the scalar head is a preference scorer, not a calibrated success probability. For a later C3 experiment, inspect the margin between the winner and the next **distinct executable action** on calibration tasks. Duplicate text or semantically equivalent actions can make a margin misleading. Audit selection accuracy, action types, task stage, and retained coverage before freezing any threshold. A retention sweep such as 50/75/100% is a proposed calibration experiment, not a validated setting; choose on calibration/dev data only.

SelectionARM provides a winner, not an absolute confidence score. Candidate-order consistency on calibration states can diagnose sensitivity, but consistency is not correctness. Do not use its index-token probability as a success probability. Additional outcome and confidence filters alter the plain distillation target and therefore require their own controls.

<a id="arm-filtered-sft-plan--loss-and-implementation-route"></a>
### Loss and implementation route

For a chosen response y at pre-action context h, the basic objective is negative log likelihood, `L = -log pi_student(y | h)`. Use mean token loss within each current response and average those per-turn losses across the accumulated batch (equal turn weighting). Apply loss only to the **current chosen assistant response**, with earlier assistant turns, user/tool text, and image tokens masked out. Preserve the actor's reasoning-plus-tool-call format in the first experiment. “Action-level” describes the retained turn; it does not require stripping reasoning tokens. An action-JSON-only loss is a separate ablation. Record whether reduction is per-token or per-turn; per-turn length normalization changes example weights.

The repository already has `sft_loss` in `slime/backends/megatron_utils/loss.py`, so a new RL objective is unnecessary. However, `slime/rollout/sft_rollout.py` uses a generic chat loss-mask generator and does not construct the browser multimodal tensors despite loading a processor. We need a dedicated offline browser-turn adapter that reuses the processor/image-grid path and supervises only the final selected turn.

Implemented units, with live validation status recorded in [ARM_C2_RUN.md](ARM_SFT.md#arm-c2-run):

1. Opt-in `TurnExporter` on `ActionSelector`, saving exact prompts/images and all candidate response tokens. The default inference path leaves this disabled.
2. Offline join/filter builder producing an immutable versioned dataset and a reason-coded retention report.
3. HF browser-turn loader rebuilding multimodal tensors with the original processor, verifying prompt tokens/image grids, and using `-100` labels on every prefix token. It uses the recorded current-response token IDs directly; no browser or ARM call during optimizer training.
4. Isolated HF/PEFT LoRA training recipe with current-turn likelihood loss; optionally adapt the same records to existing Slime SFT loss later. No outcome/RL dynamic filters or advantage normalization in the offline optimizer path.
5. Resumable collection/controller and LoRA trainer with dataset checksums, durable adapter/optimizer checkpoints, and allocation limits. Student evaluation will use one candidate, following the original baseline protocol.

<a id="arm-filtered-sft-plan--first-full-pool-c2-training-configuration"></a>
#### First full-pool C2 training configuration

Use a small **LoRA pilot with an isolated Hugging Face/PEFT trainer** for the first C2 student. The installed stack is Transformers 4.57.1, PEFT 0.18.1, and PyTorch 2.9.1+cu129. The existing Slime SFT loss remains a possible later full-parameter route; the current inspection did not establish Megatron LoRA support. The browser-specific export and masking contracts apply to either trainer. The trainer is implemented; its synthetic GPU backward/save/reload check gates collection and is reported separately from student optimization.

| Setting | Proposed value |
| --- | --- |
| Initial checkpoint | Pinned `OpenWebRL/OpenWebRL-4B-SFT`; start each student independently from it |
| Architecture | Qwen3-VL, 36 language layers; exact checkpoint tensor names checked on CPU |
| Trainable parameters | LoRA on language-model `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` only |
| Frozen parameters | All base weights, vision encoder, visual merger/projector, embeddings, and LM head |
| LoRA | rank 16, alpha 32, dropout 0.05; no trainable bias |
| Optimizer | AdamW, learning rate **1e-5**, betas **(0.9, 0.95)**, epsilon **1e-8**, weight decay **0.01** |
| Schedule | Cosine decay; warmup 3% of optimizer updates; clip gradient norm at 1.0 |
| Precision / memory | BF16, gradient checkpointing, microbatch 1, no sequence packing |
| Effective batch | **16 retained turns** via gradient accumulation 16 on one GPU |
| Maximum sequence | **32768 expanded multimodal tokens**, including current response; exclude and count oversize rows without silently changing context |
| Image processing | Reuse the actor's collection processor, actual screenshot, and image-grid geometry; do not substitute the ARM's 262144-pixel scoring cap |
| Supervision | Current chosen response, including its reasoning/tool-call formatting and stop token; previous turns and image/prompt tokens masked |
| First student | **C2 only**, outcome-filtered ARM-winner SFT; compare with C0, which requires evaluation but no training |
| Data budget | **All eligible C2 turns from all 2091 tasks**, with no retained-turn subsample or cap |
| Training duration | **2 fixed epochs** over N retained turns; `2 * ceil(N/16)` optimizer updates, with no 250-update cap |
| Small-data review | If fewer than **256** C2 turns survive, review collection coverage before training; this is an engineering guard, not evidence that 256 turns suffice |
| Checkpoints | Save every 100 updates, each epoch, and on a graceful pause; optimizer/RNG state supports resumption. Use fixed epoch 2 for the first student comparison |
| Student evaluation | One sampled action per turn, no ARM, otherwise the same decoding/horizon/judge settings as the inference baseline |

Record actual examples, successful trajectories, target tokens, optimizer updates, and task coverage for C2. There is no C1 matching subsample in this first run. Any later C1/C3 ablation should use a predeclared matched budget, since equal update counts alone do not guarantee equal supervised tokens or compute.

This configuration is a conservative starting proposal, not a tuned recipe. The one-GPU memory/throughput estimate remains to be measured. The user selected the full task pool and post-evaluation C2 launch. The existing compute budget remains the limit; additional allocations still require exact resource/budget approval. The implemented configuration above is the first-run default, with teacher choice frozen from complete inference results.

The **effective batch of 16 is an adaptation, not a reproduced actor-SFT hyperparameter**. Upstream's MolmoWeb actor-distillation trainer defaults to microbatch 1 and gradient accumulation 8, hence effective batch 8. Upstream's OpenWebRL SelectionARM LLaMA-Factory recipe uses per-device batch 2 and accumulation 8, hence effective batch 16, but it trains the selection reward model rather than an actor. This pilot uses microbatch 1 because full-history multimodal turns can approach the 32768-token limit, then accumulation 16 to recover the effective batch used by the OpenWebRL ARM recipe. No batch-size sweep establishes 16 as optimal. The trainer gives each retained turn equal weight: it averages target tokens within a turn and then averages the accumulated turns, whose lengths can differ. A later actor ablation should compare effective batch 8 and 16 at matched data exposure; do not describe the current choice as an upstream default.

Before a pilot train, verify image-token expansion, prompt-prefix equality, correct target/EOS boundaries, zero loss on previous turns, finite backward gradients, and checkpoint save/reload. CPU data checks are prepared; GPU training validation remains outstanding. Long contexts must be handled consistently across collection and training, with exclusions counted rather than silently truncating away the selected action or screenshot.

<a id="arm-filtered-sft-plan--literature-and-how-it-changes-this-proposal"></a>
### Literature and how it changes this proposal

<a id="arm-filtered-sft-plan--upstream-distillation-material-added-since-the-initial-plan"></a>
#### Upstream distillation material added since the initial plan

The ARM checkout now includes `actor_distillation/`. At upstream commit `4d6dfff869f198f282e4b0e8cf6d429c23dc9fce`, the [reported MolmoWeb experiments](https://github.com/piotr-teterwak/action-reward-models/blob/4d6dfff869f198f282e4b0e8cf6d429c23dc9fce/actor_distillation/RESULTS.md) include a negative result for streaming trained-SelectionARM distillation (−4.2 percentage points, with scrolling/termination collapse) and a positive result for offline GPT-5.5 PRM-score selection with a 0.7 floor (+8.7 points versus baseline, +7.2 versus random-target SFT). These use a different actor and recipe; they do not establish that our C2 will improve.

Our successful-trajectory filter addresses poor local winners indirectly, while preserving the user's chosen C2 scope. It does not prove every action in a successful trajectory is useful. In particular, the [AgentTrek judge](ARM_INFERENCE.md#arm-judge-alignment) allows partial completion. Keep termination rate, trajectory length, and scroll frequency alongside success when evaluating the student. The first saved C2 quality diagnostic covers 124 successful trajectories: all end in `done`, none reaches 30 turns, and mean length is 5.77 turns. This is a collection check, not evidence of student generalization.

The user requested training from our own ARM branch. The configured handoff uses `openwebrl/c2-filtered-sft`, commit `adec0ad`, with `actor_distillation/train_openwebrl_c2.py`: the validated Qwen3-VL trainer and exact captured-turn loader, retaining the approved learning rate, two epochs, and complete-pool gate. The upstream MolmoWeb trainer remains unchanged. Two actor/teacher replicas now collect on g007 with 32 browser workers; the unchanged single-GPU student recipe starts after the complete dataset audit. See [run configuration and paths](ARM_SFT.md#arm-c2-run).

[BOND](https://arxiv.org/html/2407.14622v1#S4) connects winner-only likelihood training to forward KL; its full method combines KL directions. Our first pilot combines winner SFT with successful-trajectory filtering; a later C1 control can isolate the terminal filter. It is not a reproduction of full BOND/J-BOND.

Our application-specific inference: ScalarRM supplies a fixed same-state reward ordering, which better matches the assumptions behind an explicit best-of-N reward distribution. A listwise SelectionARM can still supply imitation targets, but its choices need not correspond to a fixed scalar ordering across candidate sets. Full BOND theory therefore should not be transferred to that selector without checking this assumption.

[ReST](https://arxiv.org/abs/2308.08998) motivates collecting model outputs and reusing offline selected data. [ReST-EM](https://arxiv.org/abs/2312.06585) motivates a separately measured outcome-filtered self-training branch. Their results do not establish gains for this browser actor. If the first student improves, recollect from that student on training tasks for a distinct second round; do not treat a static one-round dataset as permanently on-policy.

<a id="arm-filtered-sft-plan--order-of-work"></a>
### Order of work

1. Finish all three original evaluations and write their comparison report; keep retries held for a separate cohort decision.
2. Freeze the C2 teacher from the completed comparison; reuse the verified actor in the assigned allocation.
3. Pass synthetic training and first-eight-task export checks, then collect the entire 2091-task inventory with resumable outputs.
4. Once every task has a completed outcome, build the immutable C2 dataset and train the two-epoch LoRA student within assigned compute. Do not substitute a partial dataset if the allocation expires.
5. Evaluate the fixed epoch-2 student with one action per turn. Compare task completion, unavailable outcomes, tokens, latency, and recovery behavior with C0. C1 and C3 remain deferred.

No additional GPU allocation or budget extension is included in the queued run.

<!-- document:ARM_FILTERED_SFT_PLAN.md:end -->

---

<!-- document:ARM_C2_RUN.md:start -->
<a id="arm-c2-run"></a>
## C2 full-pool collection and SFT run

_Source record: `ARM_C2_RUN.md`. Dated entries retain their historical context._


Prepared 2026-09-08 at the user's request: use all ~2K tasks and launch C2 after the ARM evaluation. The inference retries remain held for a separate cohort decision.

<a id="arm-c2-run--full-300-checkpoint-comparison-prepared-then-deprioritized--2026-09-09"></a>
### Full-300 checkpoint comparison prepared, then deprioritized — 2026-09-09

Updates 500 and 700 are prepared for fresh evaluation on all 300
Online-Mind2Web tasks. The 200 tasks outside the earlier checkpoint-selection
sample are the primary comparison; the earlier 100 are a labeled stability
stratum. The paired controller, frozen cohort IDs, automatic analysis, exact
four-H200-hour request, and output paths are documented in
[ARM_C2_FULL300_EVAL.md](ARM_SFT.md#arm-c2-full300-eval). No job has been submitted.
The optimizer/data/full-fine-tuning candidates to consider after that result
are in [ARM_FILTERED_SFT_ABLATIONS.md](ARM_SFT.md#arm-filtered-sft-ablations).
The user subsequently chose to focus compute on new training ablations against
update 500. No full-300 Slurm job was submitted and no allocation was created.

<a id="arm-c2-run--checkpoint-scaling-complete--2026-09-09-1528-pdt"></a>
### Checkpoint scaling complete — 2026-09-09 15:28 PDT

The fixed 100-task evaluation completed for updates 100, 500, 700, and 923.
First-pass overall rates were **28%, 33%, 33%, and 28%**; valid-only rates were
**33.3%, 41.8%, 39.3%, and 32.6%**. The separately stored unavailable-task
retries covered 65/67 task/checkpoint pairs and recovered three additional
successes, all for update 500. Its replacement estimate is **36/100 overall and
36/89 valid (40.4%)**. Behavior diagnostics do not show the upstream
actor-distillation loop-collapse signature. Update **500** is the selected C2
candidate; do not resume the existing recipe toward update 1050. Full results,
uncertainty, paired comparisons, and runtime paths are in
[ARM_C2_SCALING_RESULTS.md](ARM_RESULTS.md#arm-c2-scaling-results).

<a id="arm-c2-run--training-stopped-for-checkpoint-scaling--2026-09-09-1153-pdt"></a>
### Training stopped for checkpoint scaling — 2026-09-09 11:53 PDT

At the user's request, the resumed trainer received a graceful interrupt after the loss had largely plateaued. It saved `student/paused-000923-1788980015` at optimizer update **923**, epoch-2 position **6368/8394**. The last 50/100-update mean target-token cross-entropies were **0.1487/0.1490**, and epoch 2 through the stop averaged **0.1474**, versus **0.1550** for epoch 1. The endpoint contains the adapter, optimizer, RNG, processor, and exact dataset cursor. `student/complete.json` remains absent; this is an intentionally truncated run, not a completed two-epoch checkpoint.

The prepared full epoch-2 evaluation waiter was stopped because epoch 2 will not complete. Allocation 285131 now evaluates checkpoints **100, 500, 700, and 923** on the one-time fixed 100-task sample recorded in [the scaling plan](ARM_SFT.md#arm-c2-scaling-eval). Update 100 began first; 500, 700, and 923 are queued sequentially. Every checkpoint uses the same sample and matched one-action o4-mini/AgentTrek setup. The original base actor scored 26.0% overall and 30.6% valid-only on that sample, with the historical live-site timing caveat.

<a id="arm-c2-run--training-resumed-on-g022--2026-09-09-1033-pdt"></a>
### Training resumed on g022 — 2026-09-09 10:33 PDT

The user assigned existing allocation **285131** on g022: one H200 for five hours, ending at **15:29 PDT**. The assigned GPU was verified free, and the fixed C2 student resumed from `student/paused-000671-1788962743` at optimizer update **671**. The first resumed updates are finite and retain dataset SHA-256 `cb7c75df6a4824e9e653f6d913b0ae83268610966cd13dd13fc7314e9c667fe0`; no fresh adapter was initialized. The training deadline is 15:24 PDT, five minutes before allocation expiry.

Future continuations use the checked-in [C2 resume launcher](ARM_SFT.md#arm-c2-resume). It follows the durable latest-checkpoint pointer, verifies checkpoint/dataset/trainer lineage, derives the visible GPU and deadline from an already assigned allocation, and records a per-allocation config and receipt. It never submits a job. Resuming beyond update 923 now requires a new explicit training decision; the checkpoint-scaling study comes first.

The prepared handoff requires `student/complete.json` to identify the fixed `student/epoch-2` adapter and refuses any other checkpoint. It safe-merges that adapter into `student/epoch-2-merged`, records the base path, adapter checksum, dataset checksum, update count, dtype, and weight-file sizes, and serves that immutable path with the same baseline SGLang settings used in the original comparison. A three-task smoke on indices 0, 50, and 100 checks the actor/browser/o4-mini path; all three were valid in the original baseline. The full run then uses the same 300-task file, seed 42, temperature 0.7, top-p 0.9, 1024-token response limit, 30-turn horizon, full history, one current screenshot, concurrency 8, and `online_mind2web/AgentTrek` judge with `o4-mini`. Per-task results are durable and the full output directory can resume without changing its manifest.

Training metrics are mirrored without modifying the trainer by a durable-file sidecar to [W&B run `57f0384c`](https://wandb.ai/zixianma/openwebrl-arm/runs/57f0384c). It backfilled every update and follows `student/metrics.jsonl`, logging target-token cross-entropy, perplexity, gradient norm, learning rate, and epoch progress. The local resume pointer is `student/wandb-sync.json`; `scripts/sync_arm_c2_wandb.py` can resume the same remote run after an interruption.

The complete-target audit records a 20.9% turn-level scroll rate, 13.7% `done` rate, mean/median trajectory lengths of 7.29/5, 52.6% selection away from exact-action plurality, and zero selector fallbacks. These describe training targets, not the trained policy. Because C2 has no action-local confidence floor or KL anchor, the final report must compare the base and C2 actor on trajectory length, termination, 30-step-cap, scroll, and repeated-action rates as well as overall and valid-only success. This directly checks the loop-collapse failure documented in the upstream actor-distillation results. Evidence: `diagnostics/c2-target-behavior.json`; report tool: `scripts/summarize_arm_policy_behavior.py`.

<a id="arm-c2-run--graceful-training-pause--2026-09-09-0706-pdt"></a>
### Graceful training pause — 2026-09-09 07:06 PDT

The allocation-end guard paused SFT cleanly at optimizer update **671**. The run completed one full epoch and **2336/8394 examples (27.8%)** of epoch 2. The resume point is `student/paused-000671-1788962743`: epoch 1 in zero-based trainer state, next position 2336, learning rate `3.047449137624976e-06`, and dataset SHA-256 `cb7c75df6a4824e9e653f6d913b0ae83268610966cd13dd13fc7314e9c667fe0`. `student/complete.json` is intentionally absent because the fixed two-epoch recipe is incomplete.

The pause audit successfully opens the **504-tensor** LoRA safetensors file, loads the optimizer and RNG state, matches its progress record to the latest metric and immutable dataset, and confirms all **671 losses and gradient norms are finite**. Mean loss was **0.1550** over epoch 1 and **0.1473** over the completed part of epoch 2; the observed loss range was 0.1064–0.2002 and gradient-norm range 0.1910–0.5397. The training process and controller exited normally, and both GPUs were released. Slurm subsequently closed allocation 283899 at its time limit with exit code `0:0`. Evidence: `diagnostics/training-pause-audit.json`, `diagnostics/allocation-283899-final-sacct.txt`, `student/latest-checkpoint.json`, `training.log`, and `controller-parallel.log`.

Completing epoch 2 requires resuming this exact checkpoint for the remaining **6058 examples**, approximately **379 optimizer updates**. At the measured rate near 19 seconds/update, allow about **two hours plus model-load and checkpoint overhead** on one H200. No new allocation has been requested.

<a id="arm-c2-run--epoch-1-complete--2026-09-09-0620-pdt"></a>
### Epoch 1 complete — 2026-09-09 06:20 PDT

The first full pass over all **8394 action examples** completed at optimizer update **525**. The durable `student/epoch-1` checkpoint contains the LoRA adapter, optimizer state, processor/tokenizer files, and a resume cursor at epoch 2 position 0, all tied to dataset SHA-256 `cb7c75df6a4824e9e653f6d913b0ae83268610966cd13dd13fc7314e9c667fe0`. Training continued directly into epoch 2 and had reached update 531 at the 06:22 health check.

Training remains numerically and operationally stable: losses and gradient norms are finite, no runtime errors or memory events have occurred, and the active H200 used 92.2 GiB at the check. With about 46 minutes remaining before the controller safety deadline, the expected graceful-pause checkpoint is near update **670–675**, roughly 28% through epoch 2. Completing the fixed second epoch requires a later continuation in assigned compute.

<a id="arm-c2-run--full-collection-and-sft-handoff--2026-09-09-0331-pdt"></a>
### Full collection and SFT handoff — 2026-09-09 03:31 PDT

Collection completed all **2091/2091** deduplicated training tasks: **1151 successful trajectories**, **1956 valid outcomes**, and **135 unavailable outcomes**. Success was **55.0% over all scheduled tasks (1151/2091)** and **58.8% over valid tasks (1151/1956)**. These are C2 training-pool collection rates, not Online-Mind2Web evaluation results.

The complete C2 dataset contains **8394 usable executed actions** from the 1151 successful trajectories. Three otherwise eligible turn records were excluded because no action executed. The immutable dataset SHA-256 is `cb7c75df6a4824e9e653f6d913b0ae83268610966cd13dd13fc7314e9c667fe0`. The final audit verifies 2091 unique scheduled task IDs, exactly one durable outcome per task, all dataset rows, current source hashes, all prior C2 milestone inventories, and all 900 original inference-evaluation outcomes. Evidence: `dataset-audit.json`, `diagnostics/final-collection-audit.json`, and `diagnostics/final-collection-outcome-sha256.json`.

The controller unloaded both actor/SelectionARM pairs and began the approved single-GPU SFT at **03:31 PDT**, leaving the second H200 free. Host memory fell to 18.9 GiB. The first six real optimizer updates had finite losses (**0.155–0.193**) and gradient norms (**0.192–0.226**), and all cite the frozen dataset checksum above. With 8394 rows, effective batch 16, and two epochs, the run requires about **1050 optimizer updates**. Initial steps take roughly 18–24 seconds; the remaining allocation therefore likely reaches approximately **650–700 updates (about 1.25–1.35 epochs)** before the 07:08 PDT controller deadline. This is an early throughput projection; checkpoints every 100 updates and sample-length variation can lower it. The trainer saves periodic and graceful-pause checkpoints for continuation.

<a id="arm-c2-run--1750-task-audit--2026-09-09-0230-pdt"></a>
### 1,750-task audit — 2026-09-09 02:30 PDT

The consistent dataset snapshot covers **1772/2091 completed outcomes** and retains **6632 usable turns from 959 successful trajectories**. Three additional outcomes completed while the outcome checksum inventory was being written, so that inventory contains 1775 files. The dataset builder found one otherwise eligible turn without an executed action and excluded it. Captured turn/image hashes and dataset joins pass. All **827 g022 outcomes**, all **1505 outcomes in the 1,500-task checksum inventory**, all **1634 outcomes present before the memory-reclaim restart**, and all **900 original evaluation outcomes** remain checksum-identical. Current source pins also match. Preview SHA-256: `afedd2889eb6077495c74586d0d5fc3487d2a8772b22d7e6d27571506a3ef4cb`; evidence: `diagnostics/milestone-1750-audit.json` and `diagnostics/milestone-1750-outcome-sha256.json`.

The preceding full health check recorded **1762 completed, 956 successful, and 126 unavailable**. Both GPUs remained busy at 81% and 90% mean activity during the sample, using 70.6 and 71.5 GiB. Host memory was 86.6 GiB of 240 GiB with no pressure or OOM events. Recent throughput was about **321 tasks/hour**, giving a straight-line collection ETA near **03:30 PDT**. Student optimizer updates/checkpoints remain **0 / 0** pending full-pool completion.

<a id="arm-c2-run--proactive-inference-service-memory-refresh--2026-09-09-0200-pdt"></a>
### Proactive inference-service memory refresh — 2026-09-09 02:00 PDT

Host memory had risen to **197.8 GiB of 240 GiB**, primarily because the two long-lived SelectionARM processes had grown to about 54 and 56 GiB RSS. The collector stayed near 32 GiB, no orphaned browser processes were present, and the cgroup reported no memory pressure or OOM events. To preserve enough headroom for the final collection tail, the owned collector received a graceful stop and the controller shut down its four inference services. The restart stayed within allocation 283899 and used the unchanged task queue, models, seeds, judge, and C2 recipe.

The restart audit preserved all **1634 completed outcomes**, including **872 successful trajectories and 5915 eligible turns**, and all original evaluation artifacts. No completed task was rerun. Preview SHA-256: `2866142a2904a91c80e34f3186163b886a1a943ab1c672258649821708e91e47`; evidence: `diagnostics/memory-reclaim-restart-audit.json` and `diagnostics/before-memory-reclaim-results.json`. Host memory fell to about **57 GiB** immediately after teardown and was 85.5 GiB after both replicas had reloaded and resumed collection. The monitor now performs full resource scans every 10–20 minutes, with lightweight phase checks between them to catch the collection-to-SFT handoff.

<a id="arm-c2-run--1500-task-audit--2026-09-09-0136-pdt"></a>
### 1,500-task audit — 2026-09-09 01:36 PDT

The consistent dataset snapshot covers **1504/2091 completed outcomes** and retains **5319 usable turns from 797 successful trajectories**. One additional outcome completed while the outcome checksum inventory was being written, so that inventory contains 1505 files. Captured turn/image hashes and dataset joins pass. All **827 g022 outcomes**, all **1258 outcomes in the 1,250-task checksum inventory**, and all **900 original evaluation outcomes** remain checksum-identical. Current source pins also match. Preview SHA-256: `3b501c49c0764c712915546bd056a6fca14eb08608227636b85c3d82d356a647`; evidence: `diagnostics/milestone-1500-audit.json` and `diagnostics/milestone-1500-outcome-sha256.json`.

The accompanying full health check recorded **1504 completed, 797 successful, and 111 unavailable**. Host memory was 181 GiB of 240 GiB with no pressure/OOM events; no new task errors appeared in that monitoring interval. Recent 10–30 minute rates gave a collection ETA near **03:30–03:45 PDT**. Student optimizer updates/checkpoints remain **0 / 0** pending full-pool completion.

<a id="arm-c2-run--1250-task-audit--2026-09-09-0052-pdt"></a>
### 1,250-task audit — 2026-09-09 00:52 PDT

The consistent dataset snapshot covers **1254/2091 completed outcomes** and retains **4329 usable turns from 654 successful trajectories**. Four additional outcomes completed while the outcome checksum inventory was being written, so that inventory contains 1258 files. Captured turn/image hashes and dataset joins pass. All **827 g022 outcomes**, all **1008 outcomes in the 1,000-task checksum inventory**, and all **900 original evaluation outcomes** remain checksum-identical. Current source pins also match. Preview SHA-256: `2e95612cac70372acbe4d35ea954d2626f118ba97b784ea9c81ead0ee1b7f2ae`; evidence: `diagnostics/milestone-1250-audit.json` and `diagnostics/milestone-1250-outcome-sha256.json`.

The accompanying full health check recorded **1254 completed, 654 successful, and 105 unavailable**. Both GPUs were active, host memory was 150 GiB of the assigned 240 GiB, and no memory-pressure or OOM events had occurred. Recent measured throughput was about **361 tasks/hour**. Selector connection retries have recovered live requests without resampling candidates or creating a new unavailable outcome. Student optimizer updates/checkpoints remain **0 / 0** pending full-pool completion.

<a id="arm-c2-run--1000-task-audit--2026-09-09-0007-pdt"></a>
### 1,000-task audit — 2026-09-09 00:07 PDT

The consistent dataset snapshot covers **1007/2091 completed outcomes** and retains **3502 usable turns from 537 successful trajectories**. One additional outcome completed while the outcome checksum inventory was being written, so that inventory contains 1008 files. Captured turn/image hashes and dataset joins pass. All **827 g022 outcomes**, all **861 outcomes present before the selector connection fix**, and all **900 original evaluation outcomes** remain checksum-identical. Current source pins also match. Preview SHA-256: `89457d00e0540f1704597b000775ed2c50c66d02343f1cb5bb913dc490d5e737`; evidence: `diagnostics/milestone-1000-audit.json` and `diagnostics/milestone-1000-outcome-sha256.json`.

The live count immediately after the audit was **1008 completed, 537 successful, and 90 unavailable**. Recent 10-, 20-, and 30-minute rates were **276, 279, and 272 tasks/hour**, giving straight-line collection ETAs near **04:00–04:06 PDT**. This improves the earlier forecast but remains sensitive to long final tasks and website failures. Student optimizer updates/checkpoints remain **0 / 0** until the full-pool audit passes.

<a id="arm-c2-run--allocation-forecast--2026-09-08-2347-pdt"></a>
### Allocation forecast — 2026-09-08 23:47 PDT

**Forecast, not completed work:** 911/2091 tasks were complete (489 successes, 86 unavailable), leaving 1180. The last five and ten minutes produced **252 and 240 completed tasks/hour**, respectively. Straight-line collection ETAs were 04:28 and 04:42 PDT; allowing for slower final tasks gives a central estimate near **05:00 PDT on September 9**, with a planning range of **04:30–06:00 PDT**. Allocation expiry remains 07:13 PDT, with the controller stopping at 07:08 PDT. Sustaining about **166 tasks/hour** would finish collection before the 06:53 PDT minimum-budget gate for starting SFT.

The latest audited retention (3000 turns from 863 processed tasks) extrapolates to approximately **7300 usable turns**, or about **910 optimizer updates for two epochs at effective batch 16**. Task mix can change this estimate. The central collection ETA leaves roughly **two hours for the unchanged single-GPU SFT recipe**. The expected allocation-end state is complete collection plus SFT progress with a resumable checkpoint; completing both epochs is uncertain until real-data update throughput is measured. Actual student updates/checkpoints at this forecast are **0 / 0**. This is not a prediction of the trained student's benchmark success rate.

<a id="arm-c2-run--selector-connection-recovery--2026-09-08-23332335-pdt"></a>
### Selector connection recovery — 2026-09-08 23:33–23:35 PDT

The first two-GPU collection session exposed a connection stall in the **selector client**: task `webvoyager/14758` spent 180 seconds inside HTTPX/AnyIO `connect_tcp` to port 19103, while that SelectionARM server continued answering other requests. Its unavailable outcome remains preserved. The earlier actor transport fix did not cover this separate client. Evidence: `diagnostics/selector-connect-timeout-trace.txt`.

Commit **`299f03f`** gives C2 selector connections a **10-second timeout and up to two retries**, bounded by **180 seconds total**. Retries apply only to connection failures before request submission, using the same candidates and payload. Read failures and HTTP errors are not retried. The standalone inference-evaluation client's default behavior remains unchanged. All **46 ARM tests passed**, including request preservation, total-deadline cancellation, and rejection of read/permanent-error retries.

The collector paused cleanly at **863 completed outcomes: 780 valid, 459 successful, and 83 unavailable**. The refreshed preview contains **3000 usable turns**, SHA-256 `c61d5827049d00e5201969cfd34d315933c0e7e02d46b5bc0ad22853856e137c`. Every outcome in the pre-pause checksum snapshot and all 900 original evaluation outcomes remain unchanged. Both inference service pairs exited cleanly before restart. Audit: `diagnostics/selector-connect-restart-audit.json`; source/config archive: `execution-sessions/283899-selector-connect-source.tar.gz`.

Both replicas resumed collection at **23:35 PDT**, in step **11** of the same allocation, with 16 browser workers each. Model, task pool, judge, filtering, and training recipe remain unchanged.

Before this restart, a real exported turn from each replica passed processor-prefix, image-grid, and history-loss-mask checks: **4146/287** and **4160/323** prefix/target tokens, respectively. Evidence: `diagnostics/parallel-production-export-check.json`. This checks serialization and loss targets; it does not estimate policy improvement. Student optimizer updates/checkpoints remain **0 / 0**.

<a id="arm-c2-run--two-gpu-continuation-on-g007--2026-09-08-2325-pdt"></a>
### Two-GPU continuation on g007 — 2026-09-08 23:25 PDT

At the user's request, C2 resumed in existing allocation **283899**, step **1**, on **g007**, with **two H200s, 8 CPUs, and 240 GiB RAM**. Both GPUs were verified free before launch. Allocation expiry is **2026-09-09 07:13:27 PDT**; the controller deadline is **07:08:27 PDT**. GPU UUIDs are `GPU-f07dcbbc-c700-31ae-89c2-372e72ed164c` and `GPU-eb0f41fc-737a-bc34-59e1-d6c49527294f`. No new allocation was submitted by the agent.

One frozen actor plus one SelectionARM server runs on each GPU, using actor/teacher ports **19100/19101** and **19102/19103**. A single collector owns the full task queue and outcome inventory. Its **32 browser workers are split evenly, 16 per replica**; each worker keeps its actor/teacher pairing. The remaining queue is shared, so faster workers can take the next unfinished task. Completed outcomes are skipped. Models, five-candidate sampling, request seeds, full history, context/turn limits, o4-mini/AgentTrek judge, and C2 filtering are unchanged. Sampling is not guaranteed bit-identical across serving schedules.

Both actor and teacher pairs passed their health/model checks, and collection began at **23:25:20 PDT**. Logs confirm 16 initial task starts per actor port. The original 16 interrupted tasks restarted in separate attempt directories. Sustained throughput is still being measured; doubling concurrency alone does not establish a twofold speedup.

The controller automatically releases its four inference services once all **2091** outcomes are recorded and the dataset audit passes, then launches the approved **single-GPU, two-epoch** student recipe if at least 15 minutes remain. The local ARM branch is `openwebrl/c2-filtered-sft`, now pinned to **`adec0ad95f0fe6c6d129c8984308c7e420dd6baa`**. Its only trainer change allows explicit GPU-UUID binding inside the two-GPU Slurm step; optimization, effective batch 16, and checkpoint rules are unchanged. Actual student optimizer updates/checkpoints are still **0 / 0**.

Implementation: `scripts/run_arm_c2_parallel.py`, commit **`8d1328c`**. **15 C2 tests passed**, covering replica isolation, occupied/wrong GPU rejection, explicit training-GPU binding, provenance, filtering, and full-pool gating. Source/config snapshots are under `execution-sessions/283899-parallel-source.tar.gz` and `execution-sessions/283899/`. Current logs: `controller-parallel.log`, `collection.log`, `actor-replica-{0,1}.log`, and `teacher-replica-{0,1}.log`. Active health observations are recorded in `diagnostics/monitor-283899-history.jsonl`.

<a id="arm-c2-run--final-g022-interruption-audit--2026-09-08-2316-pdt"></a>
### Final g022 interruption audit — 2026-09-08 23:16 PDT

Slurm records allocation **283221** as **CANCELLED at 23:12:41 PDT**, before its eight-hour limit. The launcher exited **137** after Slurm terminated the step. The final preview was rebuilt from durable outcomes after cancellation; this was not a clean controller shutdown.

| Metric | Preserved result |
| --- | ---: |
| Completed outcomes / full pool | **827 / 2091** |
| Valid / unavailable outcomes | **745 / 82** |
| Successful trajectories | **434** |
| Success among processed tasks | **52.5% (434/827)** |
| Success among valid processed tasks | **58.3% (434/745)** |
| Eligible training turns | **2853 from 434 successful tasks** |
| Remaining tasks | **1264**, including 16 interrupted tasks |
| Actual student optimizer updates / checkpoints | **0 / 0** |

These are training-pool collection rates, not Online-Mind2Web benchmark results. All **529 pre-transport-fix C2 outcomes and all 900 original evaluation outcomes** remain checksum-identical. There were **zero 180-second generation timeouts among the 298 completed tasks after the connection fix**. The last healthy memory sample was about 89 GiB of 120 GiB, with no allocation OOM events.

Preview SHA-256: `feceebfc4694aae70d7163ad69e95125e6074dca5539db9b35db48647e3e7e0f`. Evidence: `diagnostics/allocation-283221-final-audit.json`, `allocation-283221-final-outcome-sha256.json`, `allocation-283221-unfinished-attempts.json`, and `allocation-283221-final-sacct.txt`. Previous and refreshed preview files remain archived by checksum in `preview-history/`.

<a id="arm-c2-run--750-task-audit--2026-09-08-2250-pdt"></a>
### 750-task audit — 2026-09-08 22:50 PDT

The saved audit covers **755/2091 completed outcomes** and retains **2530 usable turns from 388 successful trajectories**. Captured source/image hashes and dataset joins pass the builder checks. All **529 pre-transport-fix C2 outcome files and all 900 original evaluation outcome files** remain byte-identical to their preserved checksum inventories. Both the previous and refreshed preview/audit files are archived in `preview-history/`. Preview SHA-256: `ee22dc15577486930765e16c4b0aee4a332556449ee834937e8510632c817728`; detailed evidence: `diagnostics/milestone-750-audit.json`.

The **226 tasks completed since the transport restart had zero 180-second generation timeouts**. Retryable connection errors have occurred and recovered. Oversized context requests now fail after one response per candidate rather than repeatedly retrying; their outcomes remain preserved as unavailable. Website and environment failures still occur. No allocation OOM events have been observed. These are operational observations, not a controlled estimate of reward-model gains.

The live check at **22:55 PDT** recorded **772 completed tasks, 397 successes, and 80 unavailable outcomes**. GPU activity averaged 70% over that short sample; GPU memory was 76.5 GiB and host memory 82.7 GiB of the assigned 120 GiB. Collection continued during an agent-session interruption; active supervision resumed after the user's reset. Recent throughput was about 215 tasks/hour, so completing the remaining pool before the 02:28 PDT allocation expiry is unlikely. No additional allocation is requested automatically.

Student optimizer updates and durable student checkpoints remain **0 / 0**. The partial preview is not the final training dataset. SFT still requires all 2091 outcomes and the complete audit before its configured ARM-branch handoff.

<a id="arm-c2-run--connection-timeout-recovery--2026-09-08-21222126-pdt"></a>
### Connection-timeout recovery — 2026-09-08 21:22–21:26 PDT

Collection paused cleanly at **529/2091 outcomes: 462 valid, 277 successes, and 67 unavailable**. The shutdown audit retains **1740 usable turns**, preview SHA-256 `f8e28a4ad075d452464ea1323cfe24478c4bb1ed149f0454ba13260cfa747c3e`. All 529 completed outcome hashes were verified unchanged. Student optimizer updates/checkpoints remain **0 / 0**.

A generation-timeout traceback showed HTTPX/AnyIO still opening a TCP connection to `127.0.0.1:19100`, with its connection timeout set to `None`. The actor was otherwise serving: the listen backlog was empty, and 12 fresh model-info connections succeeded (usually 1–8 ms, one 476 ms). This identifies the phase of the observed stall; it does not establish the underlying network/library cause. The trace is retained at `diagnostics/connect-timeout-trace.txt`.

Commit `6d408aa` gives C2 connection attempts a **10-second timeout**, allowing the existing retry loop to recover within the unchanged **180-second overall generation deadline**. Explicit HTTP 400 context-overflow responses now fail immediately rather than retrying the identical oversized prompt 60 times. Transient HTTP and connection errors remain retryable. The client still leaves its read/write/pool timeouts unset; the enclosing generation deadline controls the total wait. Retry logs now include exception types. **Four focused transport tests and 11 C2 tests passed.** Model, prompts, candidate count, judge, filtering, and training recipe are unchanged.

The controller resumed collection at **21:25:48 PDT**, within allocation **283221**, step **148**. Old configs, source snapshots, completed outcomes, and interrupted attempt directories remain preserved. The 750-task audit above records the subsequent live transport observations; no success-rate improvement is assumed.

<a id="arm-c2-run--500-task-audit--2026-09-08-2116-pdt"></a>
### 500-task audit — 2026-09-08 21:16 PDT

The post-500-task audit covers **502/2091 outcomes: 440 valid, 261 successful, and 62 unavailable**. The preview retains **1643 usable turns from 261 successful trajectories**. All captured source/image hashes and joins checked by the dataset builder pass; all 413 outcomes from before the I/O fix remain unchanged. Preview SHA-256: `b174a83243ef1aad820f98fcc92bad7d4307c3dd02af9ded1f69aac8541ced5e`. Detailed evidence: `diagnostics/milestone-500-audit.json`; the previous preview and audit are preserved in `preview-history/`.

A real turn collected after the fix also passed the processor-prefix, image-grid, and history-loss-mask check (`diagnostics/async-io-production-export-check.json`: task `webvoyager/88792`, 4142 prefix tokens, 220 target tokens). Student optimizer updates and checkpoints remain **0 / 0**; full-pool collection is still required before SFT.

Subsequent failures include generation timeouts, website navigation failures, a whole-task timeout, and context overflow. There has been no repeat of the local browser startup-health failures by this audit. At this audit, context-overflow requests still passed through the generic retry loop; the subsequent transport recovery above removes those permanent-error retries. A resource check found 3.55/4 CPU cores busy, no orphaned Chrome processes, and about 19 GiB of filesystem cache within the reported host memory. No allocation OOM events have occurred.

<a id="arm-c2-run--filesystem-stall-recovery--2026-09-08-20302036-pdt"></a>
### Filesystem stall recovery — 2026-09-08 20:30–20:36 PDT

Collection paused cleanly with **413/2091 completed outcomes, 360 valid, 217 successes, and 53 unavailable**. The audited preview contains **1261 eligible turns**; SHA-256 `7f10b20e46a05b5d21d041c92231d752a28ac8b3f5f42460cde432c76a67158b`. Student optimizer updates and checkpoints remain **0 / 0**.

The collector repeatedly blocked in shared-filesystem calls on its event-loop thread. A bounded `strace` measured screenshot and JSON file creation at **4.20, 5.42, and 6.21 seconds**. These stalls delay unrelated browser requests and startup health checks. Several new tasks became unavailable after the local browser server missed its 30-second health deadline; its logs did not show a process crash. This is an infrastructure limitation, not evidence that the ARM chose a bad action.

Commit `c355749` moves C2 selection traces, exact-turn exports, outcome writes, and optional rollout dumps to awaited worker threads. Cancellation drains outstanding writes; identical screenshots use a per-content-hash lock. Summaries use a cached outcome inventory during collection and a full disk audit at shutdown. Writes remain on persistent scrubbed storage. The actor, teacher, sampling, judge, filtering, and SFT recipe are unchanged. All **413 C2 and 900 original evaluation outcomes** passed checksum checks before continuation. Completed unavailable outcomes remain preserved; only unfinished tasks resume in new attempt directories.

Validation: **2 concurrency/cancellation tests, 11 C2 tests, and 9 ARM inference tests passed**, plus syntax checks. Old execution configs and a source snapshot are retained in `execution-sessions/`; the syscall trace is `diagnostics/collector-filesystem-strace-283221.txt`. Collection resumed under the same allocation after source-pin verification. Real workload throughput and browser failures must still be monitored; the code change is not itself evidence of an end-to-end speedup.

<a id="arm-c2-run--resume-on-g022--2026-09-08-1841-pdt"></a>
### Resume on g022 — 2026-09-08 18:41 PDT

The user assigned existing allocation **283221** for continued C2 collection followed by SFT. The allocation provides **one H200, 4 CPUs, and 120 GiB RAM** on g022, with scheduled expiry **2026-09-09 02:28:26 PDT**. The assigned GPU UUID is `GPU-b3af7cdf-9d21-56b4-0ad8-de51a96bd329`; it was verified free before launch. Controller stop is **02:23:26 PDT**, preserving a five-minute margin. No new allocation was requested.

The actor and SelectionARM services passed their startup checks; the controller entered `collecting` at **18:41:31 PDT**. Resume starts from **220 preserved outcomes**, with **1871 unfinished tasks**. The dataset, teacher, sampling, judge, and two-epoch recipe are unchanged. Completed failures and unavailable outcomes are not retried. Prior interrupted attempts remain separate from restarted attempts. The results checksum snapshot and old/new configurations are retained in `execution-sessions/`.

The automatic SFT handoff now uses the user's own local ARM branch:

- Checkout: `/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/action-reward-models-c2`.
- Branch: **`openwebrl/c2-filtered-sft`**; commit **`9f2998964e28b88f436fd50ff8be0ba473645465`**, based on upstream `4d6dfff869f198f282e4b0e8cf6d429c23dc9fce`.
- Entrypoint: `actor_distillation/train_openwebrl_c2.py`. This is byte-identical to our already GPU-validated OpenWebRL C2 trainer. It uses the OpenWebRL export/processor loader through the pinned local runtime dependency. The upstream MolmoWeb trainer's chat serialization and model defaults do not directly fit Qwen3-VL captured browser turns.
- Controller checks the ARM commit, entrypoint checksum, and OpenWebRL dependency checksums before training. CPU tests verify changed entrypoints/recipes are rejected and resource-only resumes preserve old configurations. **11 C2 tests passed**; the branch entrypoint imports successfully.
- Training begins only after all 2091 outcomes are recorded and the full dataset passes its audit, with at least 15 minutes of allocation time left. Collection/training remain resumable if the existing allocation is insufficient. At resume, actual student optimizer updates and durable student checkpoints remain **0 / 0**.

The [judge alignment audit](ARM_INFERENCE.md#arm-judge-alignment) confirms the o4-mini model and reported AgentTrek protocol match the author's OpenWebRL setup, while documenting unresolved historical implementation/cohort details and known decoding differences. Original evaluation artifacts remain separate.

**19:26 PDT collection audit:** 337/2091 completed outcomes, 181 successful trajectories, and **1044 eligible turns** in the updated preview. The 711-turn preview and its audit were archived under `preview-history/` before replacement. All 13 previously interrupted tasks have completed outcomes from their separate resumed attempts. Actual student optimizer updates/checkpoints remain **0 / 0**. Preview SHA-256: `ecad5cf0376fc8224054a856229a8822ffdb947ef4e90476bd5073e284c2209f`.

Health samples and errors checked during the active agent session are recorded under `diagnostics/monitor-283221-history.jsonl`; the latest snapshot is `diagnostics/monitor-283221.json`. These records are observations, not an independent recovery agent. At this audit, generation timeouts and browser failures remained task-local; no allocation OOM events had occurred. Runtime dependencies and all 900 original evaluation outcome files passed a post-resume checksum check. A verified source/configuration archive is saved as `execution-sessions/283221-source-and-config.tar.gz`.

<a id="arm-c2-run--serving-throughput-check-and-restart--19421952-pdt"></a>
#### Serving throughput check and restart — 19:42–19:52 PDT

A ten-second CPU sample used 3.77 of four allocated cores, while the actor queued requests. Collection was intentionally paused at 19:42:46 PDT to compare serving configurations. It finished shutdown with **373 completed outcomes, 203 successes, and 40 unavailable outcomes**. All completed files were preserved; interrupted attempts remain separate. The actor's initial graceful shutdown stalled, so its recorded PID/start-time/cgroup identities were verified before force-stopping those owned processes.

The engineering benchmark used 16 saved training states (4144–13298 expanded prompt tokens), five candidate requests per state, and two rounds per configuration. Each request generated 256 tokens with EOS ignored for timing. These outputs never enter C2 training. The benchmark excludes browser interaction and SelectionARM scoring, so its speedup is not an end-to-end collection estimate.

| Actor serving configuration | First-round seconds | Warm-round seconds | Warm output tokens/second |
| --- | ---: | ---: | ---: |
| CUDA graphs disabled, max active requests 24 | 19.45 | 16.54 | 1239 |
| CUDA graphs enabled through batch 48, max active requests 48 | 13.33 | 10.50 | 1950 |

The combined serving change gives **1.57× warm throughput** on this workload. All 320 requests passed expanded-prefix-length and finite-log-probability checks. Each state produced five distinct candidate sequences. Sampled outputs were not bit-identical across repetitions even within either configuration; fixed request seeds are not a promise of bitwise replay in this serving stack.

Collection resumed with CUDA graphs and 48 active requests, keeping task concurrency 16, the frozen actor/teacher, five candidates, temperature 0.7/top-p 0.9, 1024-token generation limit, 32768 context, o4-mini/AgentTrek judge, and the SFT recipe unchanged. This serving change applies to continued C2 collection; the 900 completed inference outcomes remain untouched. The controller archives both execution configurations and retains the pinned ARM-branch SFT handoff. Launcher commit: `49d69bb`.

Benchmark inputs, script, logs, metrics, preserved-outcome hashes, and summary are under `engineering-serving-check/`. The rolling monitor history was reset at the graph-enabled restart so the intentional pause does not enter the subsequent throughput estimate. The earlier monitor history is archived there.

<a id="arm-c2-run--final-allocation-outcome--2026-09-08-0232-pdt"></a>
### Final allocation outcome — 2026-09-08 02:32 PDT

**C2 data collection paused cleanly; student SFT has not started.** The collector saved its summary at 02:32:35 PDT, and Slurm step `282782.1` completed successfully at 02:32:42 with exit code `0:0`. The parent four-hour allocation later reached its time limit at 02:38:18. The C2 step had already exited, preserving its outputs.

| Metric | Saved result |
| --- | ---: |
| Task pool | 2091 |
| Tasks with completed outcomes | **220 (10.5%)** |
| Valid outcomes | 194 |
| Successful trajectories | **123** |
| Valid unsuccessful trajectories | 71 |
| Unavailable outcomes | 26 |
| Success among processed tasks | **55.9% (123/220)** |
| Success among valid processed tasks | **63.4% (123/194)** |
| Eligible C2 training turns | **711 from 123 successful tasks** |
| Started but interrupted/unjudged attempts | 13 |
| Remaining tasks without a completed outcome | **1871** (13 interrupted + 1858 not started) |
| Actual student optimizer updates / checkpoints | **0 / 0** |

The saved `success_rate_all_scheduled` value is 123/2091 = 5.88%; it is only a lower bound while most tasks remain unprocessed, not the collection's final success rate. The collection rates above describe the training-task pool and are not comparable to the Online-Mind2Web benchmark rates.

Collection ran from about 01:17 to 02:32 PDT; full concurrency 16 began after the first-eight-task gate around 01:30. The complete-pool training gate correctly prevented training on this partial dataset. Next: resume the 1871 unfinished tasks on assigned compute, preserving all 220 completed outcomes, then build the complete dataset and run the configured two-epoch student training. Inference retries remain held for a separate decision.

At the subsequent status check, the 711-row preview's SHA-256 matched its audit, all retained source files existed, and its rows covered exactly 123 task IDs. Thirteen separate interrupted-attempt records were present. There is no `student/` directory. These checks establish saved collection progress, not student learning.

SelectionARM completed at **42.7% overall / 50.0% valid-only** and was selected as teacher. The initial synthetic smoke detected missing PEFT in the base Python environment and stopped before collection. The run was restarted in the same allocation using the isolated ARM environment (PEFT 0.18.1, Transformers 4.57.1, Torch 2.9.1+cu129).

The corrected synthetic GPU check passed: finite loss 4.5122, gradient norm 11.1297, and adapter save/reload equality across 252 language target modules. The eight-turn export gate and independent successful-turn mask check also passed. The synthetic adapter is an engineering artifact, never a student checkpoint.

<a id="arm-c2-run--scope-and-configuration"></a>
### Scope and configuration

- **2091 unique training tasks** from `webgym_filtered_popular_2102_cleaned.parquet` (2102 rows minus 11 duplicate hostname/instruction pairs). No heuristic evaluation overlaps were flagged. All earlier draft holdout assignments are superseded; the historical draft files remain available.
- One completed attempt per task; interrupted, unjudged attempts are archived separately. Resume skips every completed outcome, including valid failures and unavailable tasks.
- Frozen `OpenWebRL/OpenWebRL-4B-SFT` actor; five candidates per turn; temperature 0.7, top-p 0.9, 1024 generated tokens, 30 turns, seed 42, full history with one current screenshot, 32768-token context. Teacher and o4-mini terminal judging match the inference setup.
- Teacher rule: choose SelectionARM if it has more successes on the full 300-task benchmark and a positive difference against ScalarRM on their common-valid tasks; otherwise choose ScalarRM. Record the decision and pinned teacher manifest. This rule is not a significance claim.
- C2 retains all usable executed winners from valid successful trajectories. No C3 confidence filter, retained-turn cap, or random C1 comparison is included.
- Student: language-only LoRA, rank 16 / alpha 32 / dropout 0.05, AdamW 1e-5, effective batch 16, BF16, gradient checkpointing, two fixed epochs. Freeze vision, merger, embeddings, LM head, and base weights. Mean target-token loss within each turn; average turns across the accumulated batch.
- Save adapter plus optimizer/RNG state every 100 updates, at each epoch, and on a graceful pause. Use the fixed epoch-2 checkpoint for evaluation; Online-Mind2Web records never enter the training data.

[Full training plan](ARM_SFT.md#arm-filtered-sft-plan) · [Original inference results](ARM_INFERENCE.md#arm-inference-results)

<a id="arm-c2-run--handoff-and-compute-limit"></a>
### Handoff and compute limit

The original comparison report was written before C2 started. The first continuation used only the verified actor/GPU in allocation **282782**, g005, GPU `GPU-90ac2a02-abfa-147c-19ff-bbada4033da3`. Current resume resources are recorded above.

1. CPU preflight: full task inventory, terminal filtering, prompt/image/response joins, loss masking, teacher-completion gate, and mutually exclusive retry/C2 handoff.
2. Synthetic GPU backward and adapter save/reload check. This uses no benchmark records and produces no usable student checkpoint.
3. First eight tasks from the actual collection order; verify processor-expanded prompt IDs and image grids on exported executed turns. These outcomes count toward the full pool.
4. Full collection at concurrency 16, with durable task outcomes and separate attempt directories.
5. Build the immutable training dataset only when the full task pool is processed. If at least 15 minutes remain, unload the verified inference servers and start/resume the student. Otherwise record that training needs subsequently assigned compute.

Original g005 controller stop: **2026-09-08 02:32:55 PDT**, five minutes before scheduled allocation expiry. The step actually exited at 02:32:42. That allocation has ended; no new allocation or extension has been requested automatically.

<a id="arm-c2-run--files-and-status"></a>
### Files and status

Run directory: `/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-full-282782-20260908T075414Z`.

- [Controller status](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-full-282782-20260908T075414Z/status.json), [controller log](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-full-282782-20260908T075414Z/controller.log). Status is a stage record, not a continuous heartbeat.
- [Queue manifest](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-full-282782-20260908T075414Z/queue-manifest.json), [task inventory](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-full-282782-20260908T075414Z/tasks.jsonl), [data audit](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-full-282782-20260908T075414Z/data-audit.json).
- `teacher-decision.json` and `frozen-config.json`: chosen teacher, immutable data hash, source hashes, and resource/configuration provenance.
- `training-smoke.log`, `synthetic-training-smoke/smoke-passed.json`: synthetic gradient/save/reload evidence, separate from student training.
- `collection.log`, `collection-summary.json`, `results/`: full-pool scheduling progress and completed trajectory outcomes. During collection, successes divided by all 2091 scheduled tasks is only a partial lower bound, not a final collection success rate.
- `attempts/<task-hash>/attempt-NNNN/`: exact `turn-NNNN.json` prompts/candidates, `execution.json` with actual prompt tokens/image grids/execution status, and `outcome.json` or `interrupted.json`.
- `images/`: content-addressed actor input images; `selections/`: candidate/selector traces; `samples/` and `browser_logs/`: trajectory inspection and browser diagnostics.
- `export-smoke-passed.json`: real captured-prefix validation gate before full collection.
- `training.jsonl`, `dataset-audit.json`: complete frozen C2 data. Before full completion, only clearly labeled preview artifacts can be generated.
- `student/`: actual LoRA checkpoints, optimizer state, metrics, and final completion marker, once full-pool training starts.

Restart within an already assigned allocation using [run_arm_c2.sh](../../scripts/run_arm_c2.sh); update the prepared allocation/deadline manifest only after compute is assigned. It rejects occupied or mismatched GPUs.

Implementation: [collection/data builder](../../openwebrl/arm_c2.py), [continuation controller](../../scripts/run_arm_c2.py), [ARM-branch LoRA trainer](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/action-reward-models-c2/actor_distillation/train_openwebrl_c2.py).

CPU checks: **9 C2 tests, 9 ARM inference tests, and 13 retry tests passed**. The pinned model exposes 252 eligible language projection modules (36 layers × 7 projections), verified on CPU without loading weights onto the evaluation GPU. Live GPU/export validation status is recorded in the run artifacts above.

A checksum-verified source snapshot is saved on scrubbed storage as `ready-source.tar.gz`. The project Git object store previously failed with `Disk quota exceeded`; prepared artifacts and logs use scrubbed storage.

<!-- document:ARM_C2_RUN.md:end -->

---

<!-- document:ARM_C2_RESUME.md:start -->
<a id="arm-c2-resume"></a>
## Resuming C2 student training

_Source record: `ARM_C2_RESUME.md`. Dated entries retain their historical context._


The durable entrypoint is `scripts/resume_arm_c2_training.py`. It resumes only
inside an already assigned Slurm allocation; it never calls `sbatch` or requests
compute. The launcher validates the allocation owner and cgroup, requires exactly
one free visible GPU, verifies the complete dataset checksum, follows
`student/latest-checkpoint.json`, checks the checkpoint progress record, and pins
the ARM checkout commit and trainer checksum before starting.

For the current C2 run, use this command after the user assigns a one-GPU
allocation:

```bash
srun --jobid=JOB_ID --overlap --gres=gpu:1 --cpus-per-task=4 --mem=120G \
  /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/venv/bin/python \
  scripts/resume_arm_c2_training.py \
  --job-id JOB_ID \
  --run-root /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-full-282782-20260908T075414Z \
  --launch
```

The allocation's `EndTime` supplies the deadline. By default the launcher records
a stop time five minutes before allocation expiry; the trainer begins its graceful
checkpoint three minutes before that stop time. Each allocation gets immutable
provenance at
`RUN_ROOT/execution-sessions/JOB_ID/training-resume-{config,receipt}.json`, and the
training output goes to `RUN_ROOT/training-resume-JOB_ID.log`.

Omit `--launch` to run all preflight checks and write the proposed per-allocation
config without loading the model. A completed `student/complete.json`, a changed
dataset, an incomplete checkpoint, a mismatched trainer checkout, an occupied
GPU, or a second visible GPU stops the launcher. It does not fall back to a fresh
LoRA initialization.

As of allocation **285131** on g022, training resumed from
`student/paused-000671-1788962743`: update 671, epoch 2 position 2336/8394,
dataset SHA-256
`cb7c75df6a4824e9e653f6d913b0ae83268610966cd13dd13fc7314e9c667fe0`.
The active run may advance `student/latest-checkpoint.json`; future resumptions
must follow that pointer rather than copying this historical path.

On 2026-09-09 the user stopped training gracefully at update **923** to evaluate
the learning curve before choosing a revised recipe. The current durable pointer
is `student/paused-000923-1788980015`, epoch-2 position 6368/8394. Do not resume
it merely because compute is available; the checkpoint study now gates any
further optimization.

<!-- document:ARM_C2_RESUME.md:end -->

---

<!-- document:ARM_C2_ABLATION_1A_RUN.md:start -->
<a id="arm-c2-ablation-1a-run"></a>
## C2 ablation 1A run

_Source record: `ARM_C2_ABLATION_1A_RUN.md`. Dated entries retain their historical context._


Status: **training and primary validation complete**. The user explicitly
approved one H200 for five hours for 1A training followed by fixed 100-task
validation. Slurm job `285567` ran on `g012` from 2026-09-09 17:03 to 21:00
PDT, using 3:57:14 of its five-hour limit. See the
[result report](ARM_RESULTS.md#arm-c2-ablation-1a-results).

<a id="arm-c2-ablation-1a-run--frozen-training-configuration"></a>
### Frozen training configuration

| Setting | Value |
| --- | --- |
| Starting model | `OpenWebRL/OpenWebRL-4B-SFT` |
| Dataset | 8,394 C2 turns; SHA-256 `cb7c75df6a4824e9e653f6d913b0ae83268610966cd13dd13fc7314e9c667fe0` |
| LoRA | language-only rank 16, alpha 32, dropout 0.05; 33,030,144 trainable parameters |
| Optimizer | AdamW, peak LR `1e-5`, betas 0.9/0.95, epsilon `1e-8`, weight decay 0.01 |
| Batch | microbatch 1, accumulation 32, effective batch 32 |
| Duration | one pass, 263 optimizer updates |
| Schedule | example-indexed: 512-example warmup, cosine horizon 16,788 examples; endpoint LR about `5.25e-6` |
| Checkpoints | updates 66, 132, 198, 250, and the predeclared epoch-1/update-263 endpoint |
| Evaluation | fixed 100-task cohort, one-action actor, seed 42, temperature 0.7, top-p 0.9, 30-turn horizon, `o4-mini` AgentTrek judge |

The endpoint is fixed before evaluation. Intermediate checkpoints are saved for
training diagnostics and are not used to select the 1A result.

After training, the same allocation also evaluated update 250 on the frozen
100-task cohort. This is the exposure-matched comparison to the original C2
update 500: `250 * 32 = 500 * 16 = 8,000` training examples. The endpoint and
update-250 evaluations used isolated actor/browser ports and shared the H200 at
40% and 30% static memory, respectively, following the already validated
two-checkpoint execution used by the original C2 scaling sweep. The endpoint
remains the primary predeclared 1A result; update 250 is a learning-curve
diagnostic and is not a selection criterion.

The schedule did not save update 50, which would have exactly matched original
C2 update 100 at 1,600 examples (`50 * 32 = 100 * 16`). W&B has its loss but
not its weights, so it cannot be evaluated without replaying training. The
earliest available checkpoint is update 66 at 2,112 examples, 32% more exposure
than the old update-100 point. It was queued only for capacity left after
either primary evaluation, but did not start before the batch controller
exited. Future effective-batch-32 ablations should save updates 50 and 250.

<a id="arm-c2-ablation-1a-run--implementation-and-artifacts"></a>
### Implementation and artifacts

The trainer is on the local ARM branch `openwebrl/c2-filtered-sft`, commit
`f3496dec08d416348900614e0183f2fa9fca08e6`. It adds the example-indexed
schedule while leaving the original schedule available for prior runs. The
allocation controller and handoff are in OpenWebRL commit `f95daf8`.

Runtime root:

```text
/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-ablation-1a
```

Key files are `frozen-config.json`, `status.json`, `training.log`,
`student/metrics.jsonl`, periodic checkpoints under `student/`, and evaluation
outputs under `evaluation/checkpoint-scaling-100/endpoint-000263/`. The
controller reserves the final 70 minutes for merge and evaluation, starts that
handoff immediately after training completes, and preserves a resumable paused
checkpoint if training cannot finish before the reserve boundary.

The W&B sidecar uses run name `arm-c2-ablation-1a`, group
`arm-c2-ablation`, and tag `ablation-1a`: [W&B run
9ac0cb8a](https://wandb.ai/zixianma/openwebrl-arm/runs/9ac0cb8a). Its durable local
identity and resume cursor are in `student/wandb-sync.json`.

<a id="arm-c2-ablation-1a-run--wb-project-migration"></a>
#### W&B project migration

The sidecar remained in `zixianma/openwebrl` through completion so that its
live stream and resume identity were not interrupted. After completion, W&B
move task `VGFzazoyNzc1Nzg2NjYw` moved runs `9ac0cb8a` (1A) and `57f0384c`
(original C2) to the dedicated `zixianma/openwebrl-arm` project. Both new paths
were verified and both old paths are absent. Future ARM launchers set
`--project openwebrl-arm` at initialization.
This migration concerns W&B organization only; the durable local run roots and
evaluation artifacts remain at their recorded GPFS paths.

<a id="arm-c2-ablation-1a-run--initial-health"></a>
### Initial health

The first three updates completed with finite cross-entropies 0.1703, 0.1598,
and 0.1886 and finite gradient norms 0.178, 0.158, and 0.177. Learning rates
were `6.25e-7`, `1.25e-6`, and `1.875e-6`, matching the 512-example warmup.
Early update cadence projects about 3.2–3.5 hours for training, leaving roughly
1.5 hours for the evaluation handoff. These are startup measurements, not a
final throughput estimate.

<a id="arm-c2-ablation-1a-run--final-execution-note"></a>
### Final execution note

Training completed all 263 updates at 19:58 PDT and the primary endpoint
evaluation completed at 21:00 PDT. The external update-250 worker preserved
97/100 outcomes before Slurm canceled it. The update-66 worker did not start.
The cause was orchestration: the batch controller exited after its endpoint
evaluation and Slurm canceled external overlapping steps, so 1:02:46 of the
approved limit went unused. External `srun` workers must not be relied on to
keep a batch allocation alive; future controllers must own and await their full
evaluation queue before exiting.

<!-- document:ARM_C2_ABLATION_1A_RUN.md:end -->

---

<!-- document:ARM_FILTERED_SFT_ABLATIONS.md:start -->
<a id="arm-filtered-sft-ablations"></a>
## Next action-level filtered-SFT ablations

_Source record: `ARM_FILTERED_SFT_ABLATIONS.md`. Dated entries retain their historical context._


Status: **ablation 1A and its 200-task holdout comparison are complete**.
The next selected direction is same-state winner/loser preference distillation;
see the [frozen experiment plan](ARM_PREFERENCE.md#arm-preference-distillation-plan). Rank-32
and full-fine-tuning variants are secondary until the preference/data recipe is
tested.

<a id="arm-filtered-sft-ablations--what-the-first-c2-curve-says"></a>
### What the first C2 curve says

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

<a id="arm-filtered-sft-ablations--1-optimizerexposure-and-lora-rank-ablation"></a>
### 1. Optimizer/exposure and LoRA-rank ablation

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

<a id="arm-filtered-sft-ablations--2-arm-native-counterfactual-data-mixed-with-c2"></a>
### 2. ARM-native counterfactual data mixed with C2

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

<a id="arm-filtered-sft-ablations--3-full-language-tower-fine-tuning-on-stabilized-c2"></a>
### 3. Full language-tower fine-tuning on stabilized C2

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

<a id="arm-filtered-sft-ablations--training-decision-order"></a>
### Training decision order

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

<!-- document:ARM_FILTERED_SFT_ABLATIONS.md:end -->

---

<!-- document:ARM_C2_SCALING_EVAL.md:start -->
<a id="arm-c2-scaling-eval"></a>
## C2 checkpoint scaling evaluation

_Source record: `ARM_C2_SCALING_EVAL.md`. Dated entries retain their historical context._


Status: **complete**. All four fixed 100-task checkpoint evaluations finished on
allocation **285131** on g022. The unavailable-task retry pass covered 65/67
requested task/checkpoint pairs before the allocation's safety cutoff. See the
[complete results and interpretation](ARM_RESULTS.md#arm-c2-scaling-results).

The earlier user-assigned allocation **285189** on g002 provided two H200s, 8
CPUs, 240 GiB RAM, and a four-hour limit. It was canceled by the user's UID at
11:49:03 PDT on 2026-09-09 after 8m41s. The only completed step was a one-second
GPU availability check; both H200s were free. Slurm rejected the first worker
after the allocation ended.

The user subsequently directed the scaling evaluation to allocation **285131**
on g022 and stopped training at update 923 to prioritize evaluation. The
single-H200 queue evaluates updates 100, 500, 700, and 923 sequentially. Each
checkpoint evaluation is resumable. No primary epoch-2 evaluation is possible
because the fixed epoch-2 checkpoint was not produced.

The evaluated states are the C2 adapters at optimizer updates **100, 500, 700,
and 923**. Update 1050 was the last planned update because the immutable dataset
has 8394 turns and the recipe has effective batch 16 for two epochs:
`2 * ceil(8394 / 16) = 1050`. At the user's request, training stopped after the
loss plateaued. The endpoint is the durable `paused-000923-1788980015`
checkpoint at epoch-2 position 6368/8394; it is an intentionally truncated run,
not a completed epoch-2 result.

<a id="arm-c2-scaling-eval--frozen-task-sample"></a>
### Frozen task sample

Before reading any checkpoint result, Python `random.Random(20260909)` sampled
100 positions without replacement from the 300-task Online-Mind2Web file. The
indices were sorted only for storage and dispatch. The complete indices, task
IDs, task-file checksum, creation time, and contemporaneous baseline reference
are committed in [arm_c2_scaling_100.json](arm_c2_scaling_100.json). The same
bytes are stored with the runtime outputs as `checkpoint-scaling-100/sample.json`.
Its SHA-256 is
`2b55f26b0a02296d4488b801bffc0806ce4830cf47350ef7917de887f424390d`.

The already completed base-policy evaluation scored **26/100 overall (26.0%)**
and **26/85 valid (30.6%)** on this sample; 15 outcomes were unavailable. This
historical base reference is useful but remains time-confounded against the new
live-site checkpoint runs.

<a id="arm-c2-scaling-eval--execution-and-interpretation"></a>
### Execution and interpretation

One worker uses the single assigned H200. It evaluates updates 100, 500, 700,
and 923 in sequence. Each adapter is safe-merged independently, then evaluated as a one-sample
policy with the same task file, seed 42, temperature 0.7, top-p 0.9, 1024-token
response limit, 30-turn horizon, full history, one current screenshot,
concurrency 8, and o4-mini AgentTrek terminal judge used by the main comparison.
Each output has an immutable manifest and durable per-task results.

Report both overall and valid-only success. Also compare trajectory length,
termination rate, 30-step-cap rate, scroll-call share, and repeated-primary-action
rate. The 100-task curve is diagnostic: confidence intervals are wide, repeated
benchmark use limits confirmatory claims, and concurrent waves reduce but do not
remove live-site time drift. Checkpoint 923 is the stopped-run endpoint.

Runtime root:
`/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-full-282782-20260908T075414Z/evaluation/checkpoint-scaling-100`.

<a id="arm-c2-scaling-eval--first-pass-results"></a>
### First-pass results

| Checkpoint | Success / scheduled | Overall | Success / valid | Valid-only | Unavailable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original base actor (historical) | 26/100 | 26.0% | 26/85 | 30.6% | 15 |
| Update 100 | 28/100 | 28.0% | 28/84 | 33.3% | 16 |
| Update 500 | 33/100 | 33.0% | 33/79 | 41.8% | 21 |
| Update 700 | 33/100 | 33.0% | 33/84 | 39.3% | 16 |
| Update 923 | 28/100 | 28.0% | 28/86 | 32.6% | 14 |

Update 100 completed at 12:55 PDT. Its behavior is close to the base actor on
the fixed cohort: mean/median trajectory length 17.66/17 versus 17.18/14,
termination 50.0% versus 52.8%, 30-step caps 41.1% versus 41.6%, scroll-call
share 11.4% versus 13.7%, and repeated-primary-action rate 59.0% versus 62.0%.
This first checkpoint does not show the sharp trajectory-length/termination/
scroll collapse reported by the upstream failed actor-distillation experiment.
The small success difference is diagnostic only.

Checkpoints 500, 700, and 923 were safe-merged in advance while update 100 was
evaluating. Update 500 ran from 12:55 to 14:00 PDT. Its mean/median trajectory
length fell to 14.24/10, termination rose to 65.1%, 30-step caps fell to 27.9%,
and scroll-call share fell to 7.9%. Thus its higher success does not carry the
upstream failure signature; repeated-primary actions remained nearly unchanged
at 59.4%.

The long tail left insufficient time to run updates 700 and 923 sequentially.
They therefore use isolated servers on the same H200, separate actor/browser
ports, six browser workers each, and 30% static server memory each. Both began
by 14:03 PDT. The model, sampling, task, horizon, context, and judge settings are
unchanged, and `execution-history.jsonl` records the concurrency change. This is
an execution-only change to finish the fixed sweep within allocation 285131.
The machine-readable success and behavior comparison lives under the runtime
root as `behavior-comparison.json`.

<!-- document:ARM_C2_SCALING_EVAL.md:end -->

---

<!-- document:ARM_C2_FULL300_EVAL.md:start -->
<a id="arm-c2-full300-eval"></a>
## C2 update-500/update-700 full evaluation

_Source record: `ARM_C2_FULL300_EVAL.md`. Dated entries retain their historical context._


Status: **deprioritized by the user on 2026-09-09; no Slurm job was submitted
and no GPU allocation was created**. This remains a prepared optional
checkpoint study. It evaluates both checkpoints on all 300 Online-Mind2Web
tasks and starts two isolated SGLang
servers on one H200 and gives each checkpoint six browser workers. This is the
same concurrent layout that completed the 100-task update-500/update-700 pair
in about one hour.

<a id="arm-c2-full300-eval--why-all-300-need-two-reported-strata"></a>
### Why all 300 need two reported strata

The fixed 100-task sample was already used to identify updates 500 and 700 as
the leading checkpoints. Reusing those tasks alone would overstate the
strength of the comparison. The full evaluation therefore predeclares:

- **Primary checkpoint comparison:** the other 200 tasks, which did not affect
  checkpoint selection.
- **Stability check:** the original 100 checkpoint-selection tasks.
- **Aggregate:** all 300 tasks.

The exact IDs and indices are frozen in
[arm_c2_full300_cohorts.json](arm_c2_full300_cohorts.json). The task-file
SHA-256 is `8343c23be98d6d63856e9b53ff3884222be099cc0f55ad5edb475176f54317ed`.
Results will report overall and valid-only success, Wilson intervals,
common-valid paired wins/losses with exact McNemar tests, unavailable counts,
and the existing trajectory/loop diagnostics. Unavailable outcomes remain in
the overall denominator. Any later retry will be labeled as a separate
sensitivity analysis.

<a id="arm-c2-full300-eval--frozen-protocol"></a>
### Frozen protocol

| Setting | Value |
| --- | --- |
| Policies | merged C2 update 500 and update 700 |
| Tasks | all 300 Online-Mind2Web tasks |
| Actor sampling | seed 42, temperature 0.7, top-p 0.9, maximum 1024 new tokens |
| Browser horizon | 30 turns, full history, current screenshot |
| Judge | `o4-mini`, Online-Mind2Web/AgentTrek terminal-success protocol |
| Parallelism | 6 browser workers per checkpoint; both checkpoints concurrent |
| Serving | two SGLang servers, tensor parallel 1, memory fraction 0.30 each |

The historical original actor is included in the generated analysis, but its
300 rollouts were collected earlier. Update 500 versus update 700 is the clean
fresh matched comparison; any comparison to the historical actor remains
subject to live-site drift.

<a id="arm-c2-full300-eval--execution-and-outputs"></a>
### Execution and outputs

The prepared entry point is
[`scripts/run_arm_c2_full300_pair.sbatch`](../../scripts/run_arm_c2_full300_pair.sbatch).
It requests exactly **one H200 for four hours, four CPUs, and 120 GB host
memory** on account `zixianma`, partition `gpu-h200`, QoS `normal`: a maximum
budget of **4 H200-hours**. Prior throughput predicts roughly three hours for
both 300-task runs, leaving startup and shutdown margin.

Runtime outputs will be written below:

```text
/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/
  c2-full-282782-20260908T075414Z/evaluation/checkpoint-full-300/
```

`pair-status.json` tracks both result counts. Each checkpoint has independent
server/evaluation logs and resumable task results. When both finish, the
controller automatically writes `comparison.json` and `comparison.md` using
the frozen cohort manifest. The launcher refuses to run outside its assigned
Slurm cgroup, with another GPU count, or with mismatched checkpoint/dataset
provenance.

Submitting this script requires explicit approval for the resource request;
the script itself records no authorization.

<!-- document:ARM_C2_FULL300_EVAL.md:end -->

---
