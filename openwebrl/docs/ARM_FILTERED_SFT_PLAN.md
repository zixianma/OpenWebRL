# ARM-filtered action-level SFT pilot

Status: **Discuss this proposal with the user before any SFT run. No training-data rollouts, ARM rescoring, or optimizer updates have been launched.** The current GPU continues the inference evaluation. Default interpretation: initialize the student from `OpenWebRL/OpenWebRL-4B-SFT` and collect new data on separate training tasks. If “openwebrl-sft” means an existing demonstration dataset, use the alternative below after locating its records.

[Inference results and retry recommendation](ARM_INFERENCE_RESULTS.md) · [Main integration plan](ARM_INTEGRATION_PLAN.md)

## Objective and first comparison

Distill useful ARM choices into the actor so that **one generated action per turn**, without an ARM, improves task completion. The current scalar inference gain is evidence that action selection helps; it is not evidence that an SFT student will retain the gain.

At each training state, sample five independent actor responses from the frozen SFT checkpoint, score/select with a frozen ARM, execute the winner, and record the exact pre-action context. Start with ScalarRM as the validated teacher. Reconsider teacher choice when SelectionARM finishes; do not mix both teachers in the first causal comparison.

Use the same candidate count, sampling, action syntax, normalized coordinates, history, screenshot processing, and response format as the inference reproduction. Freeze teacher and actor revisions for the entire collection round. Do not add state-memory or reward-rewriting changes to this experiment.

| Branch | Training targets | What it tests |
| --- | --- | --- |
| C0 | Unmodified SFT checkpoint | Standalone reference |
| C1 | All usable executed ARM winners from valid judged trajectories | Plain best-of-five action distillation |
| C2 | Usable executed ARM winners from successful trajectories only | Added terminal-outcome filtering |
| C3 | C2 plus a calibrated local ARM selection filter | Added action-level filtering |

C1 is a necessary control: “selected from five candidates” already provides ARM-based selection. C3 must improve on C2 to justify an additional turn filter. For the small first pilot, prioritize C0, C1, and C2; add C3 after the calibration audit identifies a useful criterion. Match supervised-token and optimizer-update budgets, and include a random retained-turn control when testing C3. Keep task/host/turn-position coverage visible. Do not silently train longer because a filtered set has fewer examples.

Likelihood assessment: a modest one-action gain is plausible but unestablished; retaining the full inference gain is less likely because policy errors change subsequent visited states. The incremental value of an extra confidence filter is uncertain and may be negative if it removes recovery, late-stage, or difficult actions. These are qualitative hypotheses, not measured probabilities.

## Prepared data inventory

The local `webgym_filtered_popular_2102_cleaned.parquet` contains **2102 task prompts and metadata**, not a demonstrated per-action SFT dataset. The audit found **0 exact task-ID overlaps**, **0 exact normalized-instruction overlaps**, and **0 flagged same-host near matches** against the 300 Online-Mind2Web tasks. It removed **11 duplicate training host/instruction pairs**, leaving **2091** task candidates.

This is a heuristic screen, not proof of semantic separation or absence of original model-training contamination. **441 training tasks share a hostname with an evaluation task**; this pilot is not a domain-transfer test. Inspect the pilot manually before collection.

A proposed deterministic, host-balanced pilot is already written on scrubbed storage:

- `train_pilot.jsonl`: **128 tasks**, initially one trajectory per task; expand to a second independent collection seed only if usable data are insufficient.
- `filter_calibration.jsonl`: **32 separate tasks** for filter thresholds and ARM-quality inspection; never train on their turns.
- `policy_dev.jsonl`: **32 separate tasks** for choosing the student checkpoint; never train on their turns.
- `split-audit.json`: source hashes, normalization rules, overlap counts, split hashes, and caveats.

Directory: `/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/sft-preparation/`.

All sibling attempts and turns of a task stay in its assigned split. Hold all 300 Online-Mind2Web tasks and their current evaluation trajectories out of the new SFT dataset. Keep a separate held-out set for future final claims if Online-Mind2Web results drive many iterative design choices.

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

Do not apply a global threshold such as `scalar_score > 0`: the scalar head is a preference scorer, not a calibrated success probability. For C3, inspect the margin between the winner and the next **distinct executable action** on calibration tasks. Duplicate text or semantically equivalent actions can make a margin misleading. Audit selection accuracy, action types, task stage, and retained coverage before freezing any threshold. A retention sweep such as 50/75/100% is a proposed calibration experiment, not a validated setting; choose on calibration/dev data only.

SelectionARM provides a winner, not an absolute confidence score. Candidate-order consistency on calibration states can diagnose sensitivity, but consistency is not correctness. Do not use its index-token probability as a success probability. Additional outcome and confidence filters alter the plain distillation target and therefore require their own controls.

## Loss and implementation route

For a chosen response y at pre-action context h, the basic objective is negative log likelihood, `L = -log pi_student(y | h)`. Apply loss only to the **current chosen assistant response**, with earlier assistant turns, user/tool text, and image tokens masked out. Preserve the actor's reasoning-plus-tool-call format in the first experiment. “Action-level” describes the retained turn; it does not require stripping reasoning tokens. An action-JSON-only loss is a separate ablation. Record whether reduction is per-token or per-turn; per-turn length normalization changes example weights.

The repository already has `sft_loss` in `slime/backends/megatron_utils/loss.py`, so a new RL objective is unnecessary. However, `slime/rollout/sft_rollout.py` uses a generic chat loss-mask generator and does not construct the browser multimodal tensors despite loading a processor. We need a dedicated offline browser-turn adapter that reuses the processor/image-grid path and supervises only the final selected turn.

Proposed implementation units, **not yet implemented**:

1. Optional training-turn exporter beside the existing ARM selection hook, saving exact actor input and all raw candidates without changing evaluation behavior.
2. Offline join/filter builder producing an immutable versioned dataset and a reason-coded retention report.
3. Browser multimodal SFT loader producing `Sample.tokens`, `response_length`, `loss_mask`, and `multimodal_train_inputs` from the exported record; no browser or ARM call during optimizer training.
4. Isolated HF/PEFT LoRA training recipe with current-turn likelihood loss; optionally adapt the same records to existing Slime SFT loss later. No outcome/RL dynamic filters or advantage normalization in the offline optimizer path.
5. Student checkpoint evaluation using one candidate, plus the unchanged teacher-assisted control.

### Concrete proposed training configuration — awaiting discussion

Use a small **LoRA pilot with an isolated Hugging Face/PEFT trainer** for the first comparison. The installed stack is Transformers 4.57.1, PEFT 0.18.1, and PyTorch 2.9.1+cu129. The existing Slime SFT loss remains a possible later full-parameter route; the current inspection did not establish Megatron LoRA support. The browser-specific export and masking contracts apply to either trainer. No trainer implementation or GPU backward pass is claimed ready yet.

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
| Primary student arms | **C1 plain winner SFT** and **C2 outcome-filtered winner SFT**; C0 requires evaluation but no training |
| Matched data budget | Let N be min(2000, number of usable C2 turns). Select N C1 turns using task/length strata; use N distinct turns in each arm |
| Training duration | **2 epochs**, hence `2 * ceil(N/16)` optimizer updates, capped at **250 updates** per arm |
| Small-data stop | If fewer than **256** C2 turns survive, review collection coverage before training; consider a second collection seed rather than many repetitions |
| Checkpoints | Save initialization provenance, epoch 1, and epoch 2; choose using the separate 32-task policy-dev split |
| Student evaluation | One sampled action per turn, no ARM, otherwise the same decoding/horizon/judge settings as the inference baseline |

The C1 subsample should match C2's target-token budget to within 5% where feasible; record actual tokens and do not claim exact compute matching from equal update counts alone. This paired pilot compares filtering at a fixed dataset/update budget; a later full-data experiment can measure whether discarding data was worthwhile. Keep a separate unfiltered full-data branch if the matched pilot motivates scaling.

This configuration is a conservative starting proposal, not a tuned recipe. The one-GPU memory/throughput estimate remains to be measured. We should agree on the data source, C1/C2 comparison, and LoRA route before collecting training data or starting an optimizer run.

Before a pilot train, verify image-token expansion, prompt-prefix equality, correct target/EOS boundaries, zero loss on previous turns, finite backward gradients, and checkpoint save/reload. CPU data checks are prepared; GPU training validation remains outstanding. Long contexts must be handled consistently across collection and training, with exclusions counted rather than silently truncating away the selected action or screenshot.

## Literature and how it changes this proposal

[BOND](https://arxiv.org/html/2407.14622v1#S4) connects winner-only likelihood training to forward KL; its full method combines KL directions. Our first pilot isolates simple winner SFT, then tests the proposed filters separately. It is not a reproduction of full BOND/J-BOND.

Our application-specific inference: ScalarRM supplies a fixed same-state reward ordering, which better matches the assumptions behind an explicit best-of-N reward distribution. A listwise SelectionARM can still supply imitation targets, but its choices need not correspond to a fixed scalar ordering across candidate sets. Full BOND theory therefore should not be transferred to that selector without checking this assumption.

[ReST](https://arxiv.org/abs/2308.08998) motivates collecting model outputs and reusing offline selected data. [ReST-EM](https://arxiv.org/abs/2312.06585) motivates a separately measured outcome-filtered self-training branch. Their results do not establish gains for this browser actor. If the first student improves, recollect from that student on training tasks for a distinct second round; do not treat a static one-round dataset as permanently on-policy.

## Order of work

1. Finish SelectionARM and the initial three-arm inference report.
2. Decide the matched retry protocol described in the results document; preserve initial results.
3. Confirm the fresh-training-rollout interpretation, inspect the prepared split, and implement the exact turn exporter/loader on CPU.
4. Run a tiny training-data collection and backward/save/reload smoke within explicitly assigned compute, then set the pilot's exact data and training budgets.
5. Collect the frozen pilot dataset, compare the student branches at a matched budget, and evaluate one-action task completion, failure rate, generated tokens, latency, and recovery behavior.
6. Iterate on the training plan based on independent outcome gains, not the ARM score attained by the student.

The active allocation remains devoted to finishing inference evaluation. Preparation does not queue an optimizer run or request another allocation.
