# ARM-filtered action-level SFT pilot

Status: **2026-09-08: the user selected all ~2K tasks and requested C2 after inference evaluation.** The first run now collects from all **2091 deduplicated task prompts**, then trains on all eligible executed winners from valid successful trajectories. Allocation 282782 has ended. C2 saved 220/2091 completed task outcomes and 711 eligible turns from 123 successful trajectories, then paused cleanly. Student optimization has not started; collection must resume before full-pool training. No new allocation is authorized by this plan. [Live C2 configuration and artifacts](ARM_C2_RUN.md).

[Inference results and retry recommendation](ARM_INFERENCE_RESULTS.md) · [Main integration plan](ARM_INTEGRATION_PLAN.md)

## Objective and first experiment

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

## Prepared data inventory

The local `webgym_filtered_popular_2102_cleaned.parquet` contains **2102 task prompts and metadata**, not a demonstrated per-action SFT dataset. The audit found **0 exact task-ID overlaps**, **0 exact normalized-instruction overlaps**, and **0 flagged same-host near matches** against the 300 Online-Mind2Web tasks. It removed **11 duplicate training host/instruction pairs**, leaving **2091** task candidates.

This is a heuristic screen, not proof of semantic separation or absence of original model-training contamination. **441 training tasks share a hostname with an evaluation task**; this pilot is not a domain-transfer test. All deduplicated prompts are included as requested; unavailable tasks are recorded, with no retrospective exclusion based on teacher success.

### Historical 128-task draft — superseded by the full-pool request

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

### Authorized full-pool collection

Collect one fresh trajectory per task across all **2091** deduplicated prompts. The source has 1397 exact hostnames. Use deterministic hash ordering with seed 42 and host round-robin scheduling so an interrupted run covers many sites; task order is recorded in an immutable JSONL and audit. The only 11 exclusions are duplicate hostname/instruction pairs. The evaluation-overlap heuristic flagged none.

The old **128-task cap, 2000-turn cap, 250-update cap, and draft 32/32 holdouts are superseded**. Every eligible successful-trajectory turn enters the C2 dataset. Do not add a confidence threshold. Preserve completed failed/unavailable outcomes; resumption skips completed tasks. An interrupted, unjudged attempt is stored separately and may be restarted from a fresh browser when compute is assigned, without overwriting the interrupted trace.

Before collection, run a synthetic multimodal backward/save/reload smoke; it uses no benchmark data and produces no deployable student. Then collect the first eight tasks from the same full-pool order and check actual processor-expanded prompt tokens, image grids, and response masks on their exported turns. These tasks remain part of the collection, and their outcomes are not regenerated to seek success. Expand to concurrency 16 only after the export check passes.

Hold all 300 Online-Mind2Web tasks and their evaluation trajectories out of student training. Train for two fixed epochs and use the epoch-2 student; no checkpoint selection using Online-Mind2Web. Repeated design iteration on that benchmark still limits later confirmatory claims, so a separate final benchmark may be needed for a later study.

The existing allocation ends at 02:37:55 PDT; the C2 controller stops by **02:32:55 PDT** to leave cleanup time. Collection is resumable, and full-pool training cannot start from a partial collection. If collection completes with at least 15 minutes left, the controller unloads the verified evaluation servers and starts/resumes the student on the same GPU. Otherwise it records that training awaits assigned compute. Neither collection nor training requests or extends an allocation automatically.

## Per-turn export contract

Export at collection time; do not infer training-ready turn records from concatenated rollout text. The current evaluation trace stores prompt/image hashes, split candidate reasoning/actions, scores, and selections. The sample dump stores the last turn's prompt/response plus historical screenshots. Those artifacts are useful for inspection but do not certify exact training inputs for every earlier turn.

Each training record needs:

- Task ID, split, attempt/trajectory ID, turn index, source revision and configuration hashes.
- Exact rendered **pre-action** actor prompt, processor-expanded prompt token IDs, screenshot bytes or content-addressed image path, image hash, and image-grid metadata.
- All five raw responses, token IDs, finish reasons, stable candidate IDs, teacher scores or selected index, fallback/parse status, and duplicate-action groups.
- The executed candidate's raw response and tokens, valid parsed tool call, execution status, current URL, and subsequent environment outcome.
- Terminal judge result and validity in a separate trajectory record joined only for filtering. Terminal labels, future screenshots, later actions, and ARM scores never enter the student prompt.

The SFT unit is **one selected turn**, while its complete prior context remains visible. Preserve necessary earlier actions even when they are not themselves retained for loss. Keep image resizing and normalized coordinates identical to collection. Reject incomplete prompt/image/response joins; do not invent missing observations or claim a saved prefix restores a live browser.

## Filtering rules

Apply basic data-integrity rules to all branches: correct split/provenance, exact prompt-image alignment, parseable supported action, finite scores where applicable, valid selected index, no selector fallback, and no truncated target. Distinguish an unsuccessful judged task from an unavailable run. C1 may retain usable turns from valid failed trajectories; C2 and C3 require a valid successful trajectory.

Use only **executed winners** in the initial dataset. Unexecuted candidates have a local preference label but no observed transition or terminal outcome. They can support a later preference-training experiment, not fabricated execution-success labels.

Do not apply a global threshold such as `scalar_score > 0`: the scalar head is a preference scorer, not a calibrated success probability. For a later C3 experiment, inspect the margin between the winner and the next **distinct executable action** on calibration tasks. Duplicate text or semantically equivalent actions can make a margin misleading. Audit selection accuracy, action types, task stage, and retained coverage before freezing any threshold. A retention sweep such as 50/75/100% is a proposed calibration experiment, not a validated setting; choose on calibration/dev data only.

SelectionARM provides a winner, not an absolute confidence score. Candidate-order consistency on calibration states can diagnose sensitivity, but consistency is not correctness. Do not use its index-token probability as a success probability. Additional outcome and confidence filters alter the plain distillation target and therefore require their own controls.

## Loss and implementation route

For a chosen response y at pre-action context h, the basic objective is negative log likelihood, `L = -log pi_student(y | h)`. Use mean token loss within each current response and average those per-turn losses across the accumulated batch (equal turn weighting). Apply loss only to the **current chosen assistant response**, with earlier assistant turns, user/tool text, and image tokens masked out. Preserve the actor's reasoning-plus-tool-call format in the first experiment. “Action-level” describes the retained turn; it does not require stripping reasoning tokens. An action-JSON-only loss is a separate ablation. Record whether reduction is per-token or per-turn; per-turn length normalization changes example weights.

The repository already has `sft_loss` in `slime/backends/megatron_utils/loss.py`, so a new RL objective is unnecessary. However, `slime/rollout/sft_rollout.py` uses a generic chat loss-mask generator and does not construct the browser multimodal tensors despite loading a processor. We need a dedicated offline browser-turn adapter that reuses the processor/image-grid path and supervises only the final selected turn.

Implemented units, with live validation status recorded in [ARM_C2_RUN.md](ARM_C2_RUN.md):

1. Opt-in `TurnExporter` on `ActionSelector`, saving exact prompts/images and all candidate response tokens. The default inference path leaves this disabled.
2. Offline join/filter builder producing an immutable versioned dataset and a reason-coded retention report.
3. HF browser-turn loader rebuilding multimodal tensors with the original processor, verifying prompt tokens/image grids, and using `-100` labels on every prefix token. It uses the recorded current-response token IDs directly; no browser or ARM call during optimizer training.
4. Isolated HF/PEFT LoRA training recipe with current-turn likelihood loss; optionally adapt the same records to existing Slime SFT loss later. No outcome/RL dynamic filters or advantage normalization in the offline optimizer path.
5. Resumable collection/controller and LoRA trainer with dataset checksums, durable adapter/optimizer checkpoints, and allocation limits. Student evaluation will use one candidate, following the original baseline protocol.

### First full-pool C2 training configuration

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

Before a pilot train, verify image-token expansion, prompt-prefix equality, correct target/EOS boundaries, zero loss on previous turns, finite backward gradients, and checkpoint save/reload. CPU data checks are prepared; GPU training validation remains outstanding. Long contexts must be handled consistently across collection and training, with exclusions counted rather than silently truncating away the selected action or screenshot.

## Literature and how it changes this proposal

[BOND](https://arxiv.org/html/2407.14622v1#S4) connects winner-only likelihood training to forward KL; its full method combines KL directions. Our first pilot combines winner SFT with successful-trajectory filtering; a later C1 control can isolate the terminal filter. It is not a reproduction of full BOND/J-BOND.

Our application-specific inference: ScalarRM supplies a fixed same-state reward ordering, which better matches the assumptions behind an explicit best-of-N reward distribution. A listwise SelectionARM can still supply imitation targets, but its choices need not correspond to a fixed scalar ordering across candidate sets. Full BOND theory therefore should not be transferred to that selector without checking this assumption.

[ReST](https://arxiv.org/abs/2308.08998) motivates collecting model outputs and reusing offline selected data. [ReST-EM](https://arxiv.org/abs/2312.06585) motivates a separately measured outcome-filtered self-training branch. Their results do not establish gains for this browser actor. If the first student improves, recollect from that student on training tasks for a distinct second round; do not treat a static one-round dataset as permanently on-policy.

## Order of work

1. Finish all three original evaluations and write their comparison report; keep retries held for a separate cohort decision.
2. Freeze the C2 teacher from the completed comparison; reuse the verified actor in the assigned allocation.
3. Pass synthetic training and first-eight-task export checks, then collect the entire 2091-task inventory with resumable outputs.
4. Once every task has a completed outcome, build the immutable C2 dataset and train the two-epoch LoRA student within assigned compute. Do not substitute a partial dataset if the allocation expires.
5. Evaluate the fixed epoch-2 student with one action per turn. Compare task completion, unavailable outcomes, tokens, latency, and recovery behavior with C0. C1 and C3 remain deferred.

No additional GPU allocation or budget extension is included in the queued run.
