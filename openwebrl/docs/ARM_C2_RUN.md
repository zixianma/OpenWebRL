# C2 full-pool collection and SFT run

Prepared 2026-09-08 at the user's request: use all ~2K tasks and launch C2 after the ARM evaluation. The inference retries remain held for a separate cohort decision.

## Resume on g022 — 2026-09-08 18:41 PDT

The user assigned existing allocation **283221** for continued C2 collection followed by SFT. The allocation provides **one H200, 4 CPUs, and 120 GiB RAM** on g022, with scheduled expiry **2026-09-09 02:28:26 PDT**. The assigned GPU UUID is `GPU-b3af7cdf-9d21-56b4-0ad8-de51a96bd329`; it was verified free before launch. Controller stop is **02:23:26 PDT**, preserving a five-minute margin. No new allocation was requested.

The actor and SelectionARM services passed their startup checks; the controller entered `collecting` at **18:41:31 PDT**. Resume starts from **220 preserved outcomes**, with **1871 unfinished tasks**. The dataset, teacher, sampling, judge, and two-epoch recipe are unchanged. Completed failures and unavailable outcomes are not retried. Prior interrupted attempts remain separate from restarted attempts. The results checksum snapshot and old/new configurations are retained in `execution-sessions/`.

The automatic SFT handoff now uses the user's own local ARM branch:

- Checkout: `/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/action-reward-models-c2`.
- Branch: **`openwebrl/c2-filtered-sft`**; commit **`9f2998964e28b88f436fd50ff8be0ba473645465`**, based on upstream `4d6dfff869f198f282e4b0e8cf6d429c23dc9fce`.
- Entrypoint: `actor_distillation/train_openwebrl_c2.py`. This is byte-identical to our already GPU-validated OpenWebRL C2 trainer. It uses the OpenWebRL export/processor loader through the pinned local runtime dependency. The upstream MolmoWeb trainer's chat serialization and model defaults do not directly fit Qwen3-VL captured browser turns.
- Controller checks the ARM commit, entrypoint checksum, and OpenWebRL dependency checksums before training. CPU tests verify changed entrypoints/recipes are rejected and resource-only resumes preserve old configurations. **11 C2 tests passed**; the branch entrypoint imports successfully.
- Training begins only after all 2091 outcomes are recorded and the full dataset passes its audit, with at least 15 minutes of allocation time left. Collection/training remain resumable if the existing allocation is insufficient. At resume, actual student optimizer updates and durable student checkpoints remain **0 / 0**.

The [judge alignment audit](ARM_JUDGE_ALIGNMENT.md) confirms the o4-mini model and reported AgentTrek protocol match the author's OpenWebRL setup, while documenting unresolved historical implementation/cohort details and known decoding differences. Original evaluation artifacts remain separate.

**19:26 PDT collection audit:** 337/2091 completed outcomes, 181 successful trajectories, and **1044 eligible turns** in the updated preview. The 711-turn preview and its audit were archived under `preview-history/` before replacement. All 13 previously interrupted tasks have completed outcomes from their separate resumed attempts. Actual student optimizer updates/checkpoints remain **0 / 0**. Preview SHA-256: `ecad5cf0376fc8224054a856229a8822ffdb947ef4e90476bd5073e284c2209f`.

Health samples and errors checked during the active agent session are recorded under `diagnostics/monitor-283221-history.jsonl`; the latest snapshot is `diagnostics/monitor-283221.json`. These records are observations, not an independent recovery agent. At this audit, generation timeouts and browser failures remained task-local; no allocation OOM events had occurred. Runtime dependencies and all 900 original evaluation outcome files passed a post-resume checksum check. A verified source/configuration archive is saved as `execution-sessions/283221-source-and-config.tar.gz`.

## Final allocation outcome — 2026-09-08 02:32 PDT

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

## Scope and configuration

- **2091 unique training tasks** from `webgym_filtered_popular_2102_cleaned.parquet` (2102 rows minus 11 duplicate hostname/instruction pairs). No heuristic evaluation overlaps were flagged. All earlier draft holdout assignments are superseded; the historical draft files remain available.
- One completed attempt per task; interrupted, unjudged attempts are archived separately. Resume skips every completed outcome, including valid failures and unavailable tasks.
- Frozen `OpenWebRL/OpenWebRL-4B-SFT` actor; five candidates per turn; temperature 0.7, top-p 0.9, 1024 generated tokens, 30 turns, seed 42, full history with one current screenshot, 32768-token context. Teacher and o4-mini terminal judging match the inference setup.
- Teacher rule: choose SelectionARM if it has more successes on the full 300-task benchmark and a positive difference against ScalarRM on their common-valid tasks; otherwise choose ScalarRM. Record the decision and pinned teacher manifest. This rule is not a significance claim.
- C2 retains all usable executed winners from valid successful trajectories. No C3 confidence filter, retained-turn cap, or random C1 comparison is included.
- Student: language-only LoRA, rank 16 / alpha 32 / dropout 0.05, AdamW 1e-5, effective batch 16, BF16, gradient checkpointing, two fixed epochs. Freeze vision, merger, embeddings, LM head, and base weights. Mean target-token loss within each turn; average turns across the accumulated batch.
- Save adapter plus optimizer/RNG state every 100 updates, at each epoch, and on a graceful pause. Use the fixed epoch-2 checkpoint for evaluation; Online-Mind2Web records never enter the training data.

[Full training plan](ARM_FILTERED_SFT_PLAN.md) · [Original inference results](ARM_INFERENCE_RESULTS.md)

## Handoff and compute limit

The original comparison report was written before C2 started. The first continuation used only the verified actor/GPU in allocation **282782**, g005, GPU `GPU-90ac2a02-abfa-147c-19ff-bbada4033da3`. Current resume resources are recorded above.

1. CPU preflight: full task inventory, terminal filtering, prompt/image/response joins, loss masking, teacher-completion gate, and mutually exclusive retry/C2 handoff.
2. Synthetic GPU backward and adapter save/reload check. This uses no benchmark records and produces no usable student checkpoint.
3. First eight tasks from the actual collection order; verify processor-expanded prompt IDs and image grids on exported executed turns. These outcomes count toward the full pool.
4. Full collection at concurrency 16, with durable task outcomes and separate attempt directories.
5. Build the immutable training dataset only when the full task pool is processed. If at least 15 minutes remain, unload the verified inference servers and start/resume the student. Otherwise record that training needs subsequently assigned compute.

Configured controller stop: **2026-09-08 02:32:55 PDT**, five minutes before scheduled allocation expiry. The step actually exited at 02:32:42. That allocation has ended; no new allocation or extension has been requested automatically.

## Files and status

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

Implementation: [collection/data builder](../../openwebrl/arm_c2.py), [continuation controller](../../scripts/run_arm_c2.py), [LoRA trainer](../../scripts/train_arm_c2.py).

CPU checks: **9 C2 tests, 9 ARM inference tests, and 13 retry tests passed**. The pinned model exposes 252 eligible language projection modules (36 layers × 7 projections), verified on CPU without loading weights onto the evaluation GPU. Live GPU/export validation status is recorded in the run artifacts above.

A checksum-verified source snapshot is saved on scrubbed storage as `ready-source.tar.gz`. The project Git object store previously failed with `Disk quota exceeded`; prepared artifacts and logs use scrubbed storage.
