# C2 full-pool collection and SFT run

Prepared 2026-09-08 at the user's request: use all ~2K tasks and launch C2 after the ARM evaluation. The inference retries remain held for a separate cohort decision.

## Connection-timeout recovery — 2026-09-08 21:22–21:26 PDT

Collection paused cleanly at **529/2091 outcomes: 462 valid, 277 successes, and 67 unavailable**. The shutdown audit retains **1740 usable turns**, preview SHA-256 `f8e28a4ad075d452464ea1323cfe24478c4bb1ed149f0454ba13260cfa747c3e`. All 529 completed outcome hashes were verified unchanged. Student optimizer updates/checkpoints remain **0 / 0**.

A generation-timeout traceback showed HTTPX/AnyIO still opening a TCP connection to `127.0.0.1:19100`, with its connection timeout set to `None`. The actor was otherwise serving: the listen backlog was empty, and 12 fresh model-info connections succeeded (usually 1–8 ms, one 476 ms). This identifies the phase of the observed stall; it does not establish the underlying network/library cause. The trace is retained at `diagnostics/connect-timeout-trace.txt`.

Commit `6d408aa` gives C2 connection attempts a **10-second timeout**, allowing the existing retry loop to recover within the unchanged **180-second overall generation deadline**. Explicit HTTP 400 context-overflow responses now fail immediately rather than retrying the identical oversized prompt 60 times. Transient HTTP and connection errors remain retryable. The client still leaves its read/write/pool timeouts unset; the enclosing generation deadline controls the total wait. Retry logs now include exception types. **Four focused transport tests and 11 C2 tests passed.** Model, prompts, candidate count, judge, filtering, and training recipe are unchanged.

The controller resumed collection at **21:25:48 PDT**, within allocation **283221**, step **148**. Old configs, source snapshots, completed outcomes, and interrupted attempt directories remain preserved. This transport change still requires observation on the live workload; no success-rate improvement is assumed.

## 500-task audit — 2026-09-08 21:16 PDT

The post-500-task audit covers **502/2091 outcomes: 440 valid, 261 successful, and 62 unavailable**. The preview retains **1643 usable turns from 261 successful trajectories**. All captured source/image hashes and joins checked by the dataset builder pass; all 413 outcomes from before the I/O fix remain unchanged. Preview SHA-256: `b174a83243ef1aad820f98fcc92bad7d4307c3dd02af9ded1f69aac8541ced5e`. Detailed evidence: `diagnostics/milestone-500-audit.json`; the previous preview and audit are preserved in `preview-history/`.

A real turn collected after the fix also passed the processor-prefix, image-grid, and history-loss-mask check (`diagnostics/async-io-production-export-check.json`: task `webvoyager/88792`, 4142 prefix tokens, 220 target tokens). Student optimizer updates and checkpoints remain **0 / 0**; full-pool collection is still required before SFT.

Subsequent failures include generation timeouts, website navigation failures, a whole-task timeout, and context overflow. There has been no repeat of the local browser startup-health failures by this audit. At this audit, context-overflow requests still passed through the generic retry loop; the subsequent transport recovery above removes those permanent-error retries. A resource check found 3.55/4 CPU cores busy, no orphaned Chrome processes, and about 19 GiB of filesystem cache within the reported host memory. No allocation OOM events have occurred.

## Filesystem stall recovery — 2026-09-08 20:30–20:36 PDT

Collection paused cleanly with **413/2091 completed outcomes, 360 valid, 217 successes, and 53 unavailable**. The audited preview contains **1261 eligible turns**; SHA-256 `7f10b20e46a05b5d21d041c92231d752a28ac8b3f5f42460cde432c76a67158b`. Student optimizer updates and checkpoints remain **0 / 0**.

The collector repeatedly blocked in shared-filesystem calls on its event-loop thread. A bounded `strace` measured screenshot and JSON file creation at **4.20, 5.42, and 6.21 seconds**. These stalls delay unrelated browser requests and startup health checks. Several new tasks became unavailable after the local browser server missed its 30-second health deadline; its logs did not show a process crash. This is an infrastructure limitation, not evidence that the ARM chose a bad action.

Commit `c355749` moves C2 selection traces, exact-turn exports, outcome writes, and optional rollout dumps to awaited worker threads. Cancellation drains outstanding writes; identical screenshots use a per-content-hash lock. Summaries use a cached outcome inventory during collection and a full disk audit at shutdown. Writes remain on persistent scrubbed storage. The actor, teacher, sampling, judge, filtering, and SFT recipe are unchanged. All **413 C2 and 900 original evaluation outcomes** passed checksum checks before continuation. Completed unavailable outcomes remain preserved; only unfinished tasks resume in new attempt directories.

Validation: **2 concurrency/cancellation tests, 11 C2 tests, and 9 ARM inference tests passed**, plus syntax checks. Old execution configs and a source snapshot are retained in `execution-sessions/`; the syscall trace is `diagnostics/collector-filesystem-strace-283221.txt`. Collection resumed under the same allocation after source-pin verification. Real workload throughput and browser failures must still be monitored; the code change is not itself evidence of an end-to-end speedup.

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

### Serving throughput check and restart — 19:42–19:52 PDT

A ten-second CPU sample used 3.77 of four allocated cores, while the actor queued requests. Collection was intentionally paused at 19:42:46 PDT to compare serving configurations. It finished shutdown with **373 completed outcomes, 203 successes, and 40 unavailable outcomes**. All completed files were preserved; interrupted attempts remain separate. The actor's initial graceful shutdown stalled, so its recorded PID/start-time/cgroup identities were verified before force-stopping those owned processes.

The engineering benchmark used 16 saved training states (4144–13298 expanded prompt tokens), five candidate requests per state, and two rounds per configuration. Each request generated 256 tokens with EOS ignored for timing. These outputs never enter C2 training. The benchmark excludes browser interaction and SelectionARM scoring, so its speedup is not an end-to-end collection estimate.

| Actor serving configuration | First-round seconds | Warm-round seconds | Warm output tokens/second |
| --- | ---: | ---: | ---: |
| CUDA graphs disabled, max active requests 24 | 19.45 | 16.54 | 1239 |
| CUDA graphs enabled through batch 48, max active requests 48 | 13.33 | 10.50 | 1950 |

The combined serving change gives **1.57× warm throughput** on this workload. All 320 requests passed expanded-prefix-length and finite-log-probability checks. Each state produced five distinct candidate sequences. Sampled outputs were not bit-identical across repetitions even within either configuration; fixed request seeds are not a promise of bitwise replay in this serving stack.

Collection resumed with CUDA graphs and 48 active requests, keeping task concurrency 16, the frozen actor/teacher, five candidates, temperature 0.7/top-p 0.9, 1024-token generation limit, 32768 context, o4-mini/AgentTrek judge, and the SFT recipe unchanged. This serving change applies to continued C2 collection; the 900 completed inference outcomes remain untouched. The controller archives both execution configurations and retains the pinned ARM-branch SFT handoff. Launcher commit: `49d69bb`.

Benchmark inputs, script, logs, metrics, preserved-outcome hashes, and summary are under `engineering-serving-check/`. The rolling monitor history was reset at the graph-enabled restart so the intentional pause does not enter the subsequent throughput estimate. The earlier monitor history is archived there.

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

Original g005 controller stop: **2026-09-08 02:32:55 PDT**, five minutes before scheduled allocation expiry. The step actually exited at 02:32:42. That allocation has ended; no new allocation or extension has been requested automatically.

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

Implementation: [collection/data builder](../../openwebrl/arm_c2.py), [continuation controller](../../scripts/run_arm_c2.py), [ARM-branch LoRA trainer](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/action-reward-models-c2/actor_distillation/train_openwebrl_c2.py).

CPU checks: **9 C2 tests, 9 ARM inference tests, and 13 retry tests passed**. The pinned model exposes 252 eligible language projection modules (36 layers × 7 projections), verified on CPU without loading weights onto the evaluation GPU. Live GPU/export validation status is recorded in the run artifacts above.

A checksum-verified source snapshot is saved on scrubbed storage as `ready-source.tar.gz`. The project Git object store previously failed with `Disk quota exceeded`; prepared artifacts and logs use scrubbed storage.
