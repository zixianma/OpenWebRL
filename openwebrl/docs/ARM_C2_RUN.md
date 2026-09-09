# C2 full-pool collection and SFT run

Prepared 2026-09-08 at the user's request: use all ~2K tasks and launch C2 after the ARM evaluation. The inference retries remain held for a separate cohort decision.

## 1,000-task audit — 2026-09-09 00:07 PDT

The consistent dataset snapshot covers **1007/2091 completed outcomes** and retains **3502 usable turns from 537 successful trajectories**. One additional outcome completed while the outcome checksum inventory was being written, so that inventory contains 1008 files. Captured turn/image hashes and dataset joins pass. All **827 g022 outcomes**, all **861 outcomes present before the selector connection fix**, and all **900 original evaluation outcomes** remain checksum-identical. Current source pins also match. Preview SHA-256: `89457d00e0540f1704597b000775ed2c50c66d02343f1cb5bb913dc490d5e737`; evidence: `diagnostics/milestone-1000-audit.json` and `diagnostics/milestone-1000-outcome-sha256.json`.

The live count immediately after the audit was **1008 completed, 537 successful, and 90 unavailable**. Recent 10-, 20-, and 30-minute rates were **276, 279, and 272 tasks/hour**, giving straight-line collection ETAs near **04:00–04:06 PDT**. This improves the earlier forecast but remains sensitive to long final tasks and website failures. Student optimizer updates/checkpoints remain **0 / 0** until the full-pool audit passes.

## Allocation forecast — 2026-09-08 23:47 PDT

**Forecast, not completed work:** 911/2091 tasks were complete (489 successes, 86 unavailable), leaving 1180. The last five and ten minutes produced **252 and 240 completed tasks/hour**, respectively. Straight-line collection ETAs were 04:28 and 04:42 PDT; allowing for slower final tasks gives a central estimate near **05:00 PDT on September 9**, with a planning range of **04:30–06:00 PDT**. Allocation expiry remains 07:13 PDT, with the controller stopping at 07:08 PDT. Sustaining about **166 tasks/hour** would finish collection before the 06:53 PDT minimum-budget gate for starting SFT.

The latest audited retention (3000 turns from 863 processed tasks) extrapolates to approximately **7300 usable turns**, or about **910 optimizer updates for two epochs at effective batch 16**. Task mix can change this estimate. The central collection ETA leaves roughly **two hours for the unchanged single-GPU SFT recipe**. The expected allocation-end state is complete collection plus SFT progress with a resumable checkpoint; completing both epochs is uncertain until real-data update throughput is measured. Actual student updates/checkpoints at this forecast are **0 / 0**. This is not a prediction of the trained student's benchmark success rate.

## Selector connection recovery — 2026-09-08 23:33–23:35 PDT

The first two-GPU collection session exposed a connection stall in the **selector client**: task `webvoyager/14758` spent 180 seconds inside HTTPX/AnyIO `connect_tcp` to port 19103, while that SelectionARM server continued answering other requests. Its unavailable outcome remains preserved. The earlier actor transport fix did not cover this separate client. Evidence: `diagnostics/selector-connect-timeout-trace.txt`.

Commit **`299f03f`** gives C2 selector connections a **10-second timeout and up to two retries**, bounded by **180 seconds total**. Retries apply only to connection failures before request submission, using the same candidates and payload. Read failures and HTTP errors are not retried. The standalone inference-evaluation client's default behavior remains unchanged. All **46 ARM tests passed**, including request preservation, total-deadline cancellation, and rejection of read/permanent-error retries.

The collector paused cleanly at **863 completed outcomes: 780 valid, 459 successful, and 83 unavailable**. The refreshed preview contains **3000 usable turns**, SHA-256 `c61d5827049d00e5201969cfd34d315933c0e7e02d46b5bc0ad22853856e137c`. Every outcome in the pre-pause checksum snapshot and all 900 original evaluation outcomes remain unchanged. Both inference service pairs exited cleanly before restart. Audit: `diagnostics/selector-connect-restart-audit.json`; source/config archive: `execution-sessions/283899-selector-connect-source.tar.gz`.

Both replicas resumed collection at **23:35 PDT**, in step **11** of the same allocation, with 16 browser workers each. Model, task pool, judge, filtering, and training recipe remain unchanged.

Before this restart, a real exported turn from each replica passed processor-prefix, image-grid, and history-loss-mask checks: **4146/287** and **4160/323** prefix/target tokens, respectively. Evidence: `diagnostics/parallel-production-export-check.json`. This checks serialization and loss targets; it does not estimate policy improvement. Student optimizer updates/checkpoints remain **0 / 0**.

## Two-GPU continuation on g007 — 2026-09-08 23:25 PDT

At the user's request, C2 resumed in existing allocation **283899**, step **1**, on **g007**, with **two H200s, 8 CPUs, and 240 GiB RAM**. Both GPUs were verified free before launch. Allocation expiry is **2026-09-09 07:13:27 PDT**; the controller deadline is **07:08:27 PDT**. GPU UUIDs are `GPU-f07dcbbc-c700-31ae-89c2-372e72ed164c` and `GPU-eb0f41fc-737a-bc34-59e1-d6c49527294f`. No new allocation was submitted by the agent.

One frozen actor plus one SelectionARM server runs on each GPU, using actor/teacher ports **19100/19101** and **19102/19103**. A single collector owns the full task queue and outcome inventory. Its **32 browser workers are split evenly, 16 per replica**; each worker keeps its actor/teacher pairing. The remaining queue is shared, so faster workers can take the next unfinished task. Completed outcomes are skipped. Models, five-candidate sampling, request seeds, full history, context/turn limits, o4-mini/AgentTrek judge, and C2 filtering are unchanged. Sampling is not guaranteed bit-identical across serving schedules.

Both actor and teacher pairs passed their health/model checks, and collection began at **23:25:20 PDT**. Logs confirm 16 initial task starts per actor port. The original 16 interrupted tasks restarted in separate attempt directories. Sustained throughput is still being measured; doubling concurrency alone does not establish a twofold speedup.

The controller automatically releases its four inference services once all **2091** outcomes are recorded and the dataset audit passes, then launches the approved **single-GPU, two-epoch** student recipe if at least 15 minutes remain. The local ARM branch is `openwebrl/c2-filtered-sft`, now pinned to **`adec0ad95f0fe6c6d129c8984308c7e420dd6baa`**. Its only trainer change allows explicit GPU-UUID binding inside the two-GPU Slurm step; optimization, effective batch 16, and checkpoint rules are unchanged. Actual student optimizer updates/checkpoints are still **0 / 0**.

Implementation: `scripts/run_arm_c2_parallel.py`, commit **`8d1328c`**. **15 C2 tests passed**, covering replica isolation, occupied/wrong GPU rejection, explicit training-GPU binding, provenance, filtering, and full-pool gating. Source/config snapshots are under `execution-sessions/283899-parallel-source.tar.gz` and `execution-sessions/283899/`. Current logs: `controller-parallel.log`, `collection.log`, `actor-replica-{0,1}.log`, and `teacher-replica-{0,1}.log`. Active health observations are recorded in `diagnostics/monitor-283899-history.jsonl`.

## Final g022 interruption audit — 2026-09-08 23:16 PDT

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

## 750-task audit — 2026-09-08 22:50 PDT

The saved audit covers **755/2091 completed outcomes** and retains **2530 usable turns from 388 successful trajectories**. Captured source/image hashes and dataset joins pass the builder checks. All **529 pre-transport-fix C2 outcome files and all 900 original evaluation outcome files** remain byte-identical to their preserved checksum inventories. Both the previous and refreshed preview/audit files are archived in `preview-history/`. Preview SHA-256: `ee22dc15577486930765e16c4b0aee4a332556449ee834937e8510632c817728`; detailed evidence: `diagnostics/milestone-750-audit.json`.

The **226 tasks completed since the transport restart had zero 180-second generation timeouts**. Retryable connection errors have occurred and recovered. Oversized context requests now fail after one response per candidate rather than repeatedly retrying; their outcomes remain preserved as unavailable. Website and environment failures still occur. No allocation OOM events have been observed. These are operational observations, not a controlled estimate of reward-model gains.

The live check at **22:55 PDT** recorded **772 completed tasks, 397 successes, and 80 unavailable outcomes**. GPU activity averaged 70% over that short sample; GPU memory was 76.5 GiB and host memory 82.7 GiB of the assigned 120 GiB. Collection continued during an agent-session interruption; active supervision resumed after the user's reset. Recent throughput was about 215 tasks/hour, so completing the remaining pool before the 02:28 PDT allocation expiry is unlikely. No additional allocation is requested automatically.

Student optimizer updates and durable student checkpoints remain **0 / 0**. The partial preview is not the final training dataset. SFT still requires all 2091 outcomes and the complete audit before its configured ARM-branch handoff.

## Connection-timeout recovery — 2026-09-08 21:22–21:26 PDT

Collection paused cleanly at **529/2091 outcomes: 462 valid, 277 successes, and 67 unavailable**. The shutdown audit retains **1740 usable turns**, preview SHA-256 `f8e28a4ad075d452464ea1323cfe24478c4bb1ed149f0454ba13260cfa747c3e`. All 529 completed outcome hashes were verified unchanged. Student optimizer updates/checkpoints remain **0 / 0**.

A generation-timeout traceback showed HTTPX/AnyIO still opening a TCP connection to `127.0.0.1:19100`, with its connection timeout set to `None`. The actor was otherwise serving: the listen backlog was empty, and 12 fresh model-info connections succeeded (usually 1–8 ms, one 476 ms). This identifies the phase of the observed stall; it does not establish the underlying network/library cause. The trace is retained at `diagnostics/connect-timeout-trace.txt`.

Commit `6d408aa` gives C2 connection attempts a **10-second timeout**, allowing the existing retry loop to recover within the unchanged **180-second overall generation deadline**. Explicit HTTP 400 context-overflow responses now fail immediately rather than retrying the identical oversized prompt 60 times. Transient HTTP and connection errors remain retryable. The client still leaves its read/write/pool timeouts unset; the enclosing generation deadline controls the total wait. Retry logs now include exception types. **Four focused transport tests and 11 C2 tests passed.** Model, prompts, candidate count, judge, filtering, and training recipe are unchanged.

The controller resumed collection at **21:25:48 PDT**, within allocation **283221**, step **148**. Old configs, source snapshots, completed outcomes, and interrupted attempt directories remain preserved. The 750-task audit above records the subsequent live transport observations; no success-rate improvement is assumed.

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
