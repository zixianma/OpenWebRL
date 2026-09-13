# RL runtime: resume, scaling, and execution history

Operational procedures for resuming the reference RL baseline, GPU scaling, rollout archives, and dated runtime validation. Use the resume instructions first; older execution entries explain recovery decisions and do not replace the persistent run pointer or authorize new allocations.

## Contents

- [Resuming the reference RL baseline](#resuming-baseline)
- [Baseline timing and four-GPU continuation](#baseline-scaling)
- [Completed rollout archive for future SFT](#rollout-archive)
- [H200 runtime and validation](#h200-testing)

---

<!-- document:RESUMING_BASELINE.md:start -->
<a id="resuming-baseline"></a>
## Resuming the reference RL baseline

_Source record: `RESUMING_BASELINE.md`. Dated entries retain their historical context._


Use the existing allocation that the user has explicitly authorized. From the repository:

```bash
python3 scripts/resume_baseline.py --job-id JOB_ID --dry-run
python3 scripts/resume_baseline.py --job-id JOB_ID --launch
```

`--dry-run` is the default. It reads Slurm state, recipe hashes, checkpoint metadata and saved-batch metadata; it does not run models or allocate GPUs. The launch command stays in the foreground. An agent can run it with a persistent exec session; from a terminal, use `nohup` with output redirected to scrubbed storage if it must survive disconnects.

The script **never submits or extends an allocation**. It requires a running, user-owned single-node allocation with at least two H200s, eight CPUs and 240 GiB RAM, and ten usable minutes after the shutdown margin. By default it uses two GPUs through `srun --jobid=... --overlap --exact`; the four-GPU profile is described below. It refuses launch if any non-interactive/non-extern Slurm steps are already present, and uses a per-job lock to prevent concurrent invocations. Inspect existing steps; do not stop unrelated work to bypass this check.

<a id="resuming-baseline--32-cpu-browser-concurrency-20260911"></a>
### 32-CPU continuation, September 11

Job **287949** was submitted at **11:32 PDT** with explicit approval for
**4 H200 GPUs, 32 CPUs, 480 GiB, eight hours** (32 GPU-hours, estimated
**$28.80**, plus judge usage). The user requested an immediate switch; job
**287530** was canceled after iteration **45 / 550 cumulative Adam updates**.
Its unfinished collection 46 will be collected again. Stealth evaluation
287879 continues independently.

The user selected **32 browsers as the four-GPU / 32-CPU default**. Both
`scripts/resume_baseline_4gpu.sbatch` and the explicit
`scripts/resume_baseline_4gpu_32cpu.sbatch` now select this profile, obtaining
explicit approval for each new allocation. It owns and awaits a full GPU
restore check followed by training in W&B lineage `qcq7i4ug`. Its preserved
source is runtime `reference-stage1-browsers32-20260911`, prepared with
`scripts/prepare_rollout_concurrency.py`; the matching pending-evaluation
recovery source is `reference-stage1-browsers32-pending-eval-20260911`.

Earlier TP4 runs had **32 browser pool slots but a 16-task concurrency gate**.
This profile raises the task gate to **32** and passes all 32 allocated CPUs
to the worker (the old resume wrapper capped this at 16). It retains 48 accepted
prompt groups × five attempts, reward definitions, GRPO/optimizer settings,
batches, prompts, decoding, step/time limits and save/evaluation schedules.
The five existing protected recipe hashes are unchanged, and the changed YAML
is additionally hashed. Live-web completion order can change with concurrency;
this does not promise identical trajectories or a twofold speedup. Measure
collection throughput, browser failures, CPU and host-memory use in the new
allocation.

The job started on **g013 at 11:33:04 PDT**, ending **19:33:07 PDT**.
Full four-GPU model/optimizer restoration of after-45 passed; receipt:
runtime `logs/resume-287949-4gpu-verification.json`. Worker arguments show
32 CPUs and a 32-task gate, and the browser pool reached 32/32 active slots.
Startup evidence and all new logs are in
`runs/openwebrl-4b-reference-287949-20260911T183514`.

The first collection (46) completed in **1711.6 seconds / 28.5 minutes**, versus
2392.2 and 2939.8 seconds for collections 45 and 44. This is an observational
comparison across different live-web collections, not a controlled speedup.
It completed 100 groups / 500 trajectories, accepted 48 groups / 1671 turns,
and preserved all completed trajectories, including rejected groups. Reward
**0.431478** and task success **246/500 = 49.2%** match the archive and W&B
row 736. All 12 optimizer records match rows 738–749. After-46 checkpoint
`iter_0000045` is saved with **562 cumulative Adam updates**; metadata,
shard extents, cursor and finite CPU tensor samples passed. Its own full GPU
reload remains untested. Maximum gradient norm was 2.07854 and maximum PPO KL
0.00408692. Collection 47 started at approximately 12:30 PDT.

<a id="resuming-baseline--32-browser-results-through-50"></a>
### 32-browser results through iteration 50

At **15:55 PDT on September 11**, job 287949 has saved iterations **46–50**,
reaching **604 cumulative Adam updates** at `iter_0000049`. All five rewards
and **54 optimizer records** match W&B. Complete rollout archives, checkpoint
metadata, byte extents, cursors and finite CPU tensor samples passed; these
new checkpoints have not undergone their own full GPU reloads.

| Iteration | Collection minutes | Reward | Task success, all completed | Adam updates added | Cumulative Adam updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| 46 | 28.5 | 0.431478 | 246/500 = 49.20% | 12 | 562 |
| 47 | 33.1 | 0.489987 | 273/580 = 47.07% | 12 | 574 |
| 48 | 27.5 | 0.490605 | 188/460 = 40.87% | 10 | 584 |
| 49 | 25.9 | 0.431034 | 224/465 = 48.17% | 10 | 594 |
| 50 | 35.4 | 0.472446 | 275/630 = 43.65% | 10 | 604 |

Collections average **30.1 minutes**. The prior two collections at the
16-task gate took 40 and 49 minutes; live tasks and filtering differ, so this
is observational evidence, not a controlled speedup. No GPU/backend errors
or host OOM events were observed. Host memory peaked around **416 GiB / 480
GiB**, then dropped to approximately 223 GiB after one iteration handoff.
A five-second collection sample used 14 of the 32 allocated CPU cores with
no throttling; it is not a sustained CPU utilization estimate.

The scheduled 300-task after-50 evaluation completed at approximately
**16:23 PDT**, scoring **105/300 (35.00%)**, valid-only **105/234 (44.87%)**.
All 41 scalars match W&B row 805; the saved evaluation ZIP is complete.
The pending-evaluation flag is cleared and collection 51 has started. No new top-five-triggered jobs were eligible from rewards 46–50;
two of the four approved standard-evaluation slots remain available.

<a id="resuming-baseline--health-query-timeout-recovery-20260911"></a>
### Health recorder timeout recovery, September 11

During collection 53 in job 287949, one `nvidia-smi` query exceeded its 15-second
timeout around 18:10 PDT. The recorder exited on the uncaught exception;
training continued normally. The stale health timestamp was detected at the
next supervised check, and a fresh in-allocation GPU query succeeded.
`scripts/monitor_baseline.py` now records an empty GPU sample with an explicit
`gpu_query_error` for timeouts, missing executables or nonzero exits, and
continues collecting subsequent samples. Three CPU tests cover timeout recovery
and unavailable/failed queries. The corrected recorder was restarted in the
same approved allocation, appending to the existing health log. No model or
optimizer restart was needed. Historical missing samples are not backfilled.

<a id="resuming-baseline--allocation-287949-finished"></a>
### Allocation 287949 finished: resume from 53, replay collection 54

The approved 4-H200 / 32-CPU job ended at its planned boundary on
**September 11, approximately 19:30 PDT**, elapsed **7:57:38**. Slurm reports
**COMPLETED / exit 0**; the training worker's raw **124** is the expected
allocation-boundary timeout, normalized by the controller. Eight new training
iterations (**46–53**) are saved, adding **86 durable Adam updates** and reaching
**636 total** at `iter_0000052`. These checkpoints passed metadata, shard extent,
cursor and finite CPU sample checks; their own full GPU reloads are untested.

| Iteration | Collection minutes | Reward | All-task success | Added Adam updates | Durable total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 51 | 25.7 | 0.482222 | 206/445 = 46.29% | 10 | 614 |
| 52 | 28.5 | 0.460653 | 217/550 = 39.45% | 12 | 626 |
| 53 | 24.9 | 0.539293 | 220/460 = 47.83% | 10 | 636 |
| 54, interrupted optimization | 30.7 | 0.474493 | 258/540 = 47.78% | 7 logged, not saved | 636 |

Collection 54's **44,291,206,929-byte** batch is complete and verified at
`rollout_recovery/53.pt`; its provenance records **144 submitted prompt groups**.
Resume from checkpoint 53 and replay **all ten** batch-54 optimizer updates
before fresh collection, advancing the prompt cursor by 144 groups. The seven
updates executed before shutdown have no durable checkpoint and must not be
counted as saved progress. No evaluation is pending. The existing resume helper
successfully selected this checkpoint, batch and cursor in a CPU-only check.
The pointer retains the 32-browser source and matching pending-eval recovery
source. A new paid allocation still requires explicit resource/budget approval.

All **nine collection reward records**, **86 optimizer records from completed
iterations**, and **seven partial optimizer records** match W&B. The last partial
record (`train/step=536`) was missing after shutdown; its exact dictionary was
recovered from the local training log and appended once after all workers exited,
then verified at W&B row 855. Recovery is explicitly tagged
`monitor/metric_recovered_from_local_log=1`; it does not represent a new update.
Evidence: `shutdown_metric_recovery.json` and
`iteration_54_partial_wandb_audit.json` in this run. The general hard-shutdown
flush path has not been changed; keep auditing tail records on future resumes.

The nine collections averaged **28.9 minutes**, and archives preserve **4630
completed trajectories**, including discarded groups. Host memory peaked at
**415.6 GiB / 480 GiB**. No GPU/backend or host OOM error was observed before
planned shutdown. The health-recorder timeout and repair are documented above;
shutdown `KeyboardInterrupt` traces are expected. Final evidence is
`runs/openwebrl-4b-reference-287949-20260911T183514/allocation_end_audit.json`.

Scheduled after-50 evaluation and separate after-52 evaluation both completed,
with all 41 scalars verified. After-52 used the third of four approved top-five
jobs; one remains for a future eligible reward. There are no active OpenWebRL
jobs or queued training continuation at this handoff.

<a id="resuming-baseline--allocation-288861"></a>
### Allocation 288861: continuation from checkpoint 53

The user explicitly requested another **4 H200 × eight hours** on September 11.
Job **288861** started on **g006 at 20:58:30 PDT**, with **32 CPUs, 480 GiB RAM
and 32 concurrent local browser tasks**, and ends at **04:58:34 PDT September 12**.
The authorized ceiling is **32 GPU-hours**, approximately **$28.80** at the
previously used GPU rate, plus external judge calls. Submission evidence is
`logs/submission-qcq7i4ug-after53-20260912T035830Z.json` under the runtime root.

Full four-GPU model and optimizer restoration of **after-53 / iter_0000052 /
636 Adam updates** passed before online continuation, with no optimizer steps or
browser tasks in the verification stage. Evidence:
`logs/resume-288861-4gpu-verification.json`. The online run is
`runs/openwebrl-4b-reference-288861-20260912T040027`, using preserved source
`reference-stage1-browsers32-20260911` and the same W&B run `qcq7i4ug`.

The saved collection 54 was replayed with **144 submitted groups** of cursor
advance and **ten optimizer updates** recomputed. Its replay reward matches the
original **0.4744933613**, verified at W&B history row **856** and tagged
`rollout/replayed_batch=1`. This is a repeated observation of collection 54,
not a fresh reward point. The replay's throughput counters describe loading a
saved batch, so exclude them from browser-throughput comparisons. Optimizer
verification must use the new history rows, excluding the seven unsaved updates
from the previous allocation. At **21:23 PDT**, checkpoint **after-54 / iter_0000053** passed metadata,
shard extents, cursor and finite sampled-tensor checks, with **646 durable Adam
updates**. All ten replayed optimizer records match W&B history rows **858–867**;
maximum gradient norm was **1.896** and maximum PPO KL **0.002512**. Fresh
collection 55 started. The new checkpoint has not itself been fully GPU-reloaded.

At **23:57 PDT September 11**, iterations **54–57** are saved, adding **42**
durable optimizer updates in this allocation and reaching **678 total**.
Collections 55–57 and all their images/trajectory records passed archive checks;
their rewards and all 32 optimizer metrics records match W&B. The checkpoints
passed metadata/extents/cursor and finite CPU sample checks; their own full GPU
reloads remain untested. No new top-five evaluation was triggered.

| Iteration | Collection minutes | Reward | All-task success | Added Adam updates | Durable total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 55 | 30.5 | 0.418288 | 283/575 = 49.22% | 12 | 658 |
| 56 | 28.7 | 0.453770 | 193/465 = 41.51% | 10 | 668 |
| 57 | 30.3 | 0.483749 | 262/540 = 48.52% | 10 | 678 |

Iteration 56 briefly reached gradient norm **3.925** at legacy `train/step=553`,
then returned near 1; maximum PPO KL for the iteration was **0.002207**.
Iteration 57 maximum gradient norm was **1.485**, with maximum PPO KL
**0.003254**. No GPU/backend or host OOM errors were observed.

At **03:03 PDT September 12**, iterations **54–60** are durably saved, adding
**74 optimizer updates** in this allocation for **710 total**. The latest three
collections and their rewards, all 32 optimizer records, and the three new
checkpoints passed the same archive/W&B/CPU checks described above.

| Iteration | Collection minutes | Reward | All-task success | Added Adam updates | Durable total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 58 | 30.6 | 0.440700 | 284/585 = 48.55% | 12 | 690 |
| 59 | 31.8 | 0.505842 | 288/590 = 48.81% | 10 | 700 |
| 60 | 29.9 | 0.461760 | 265/570 = 46.49% | 10 | 710 |

Collection 59 triggered the fourth and final approved standalone evaluation,
job **290361**, for after-58: **109/300 overall,109/230 valid-only**. Scheduled
after-60 evaluation also completed: **105/300 overall,105/230 valid-only**, all
41 scalars at W&B row **951**. See [RL_EVALUATION.md](RL_EVALUATION.md) for details.
The pending-evaluation flag is cleared; fresh collection 61 is running. No
additional triggered evaluation allocation remains authorized.

The project filesystem briefly rejected a documentation commit with **Disk quota
exceeded** at 23:57 PDT; scrubbed training storage continued working with no
storage errors. The user freed space, and the commit succeeded at approximately
00:18 PDT. Incident evidence is `project_quota_incident.json` in the run.

The **0.539293** reward at collection 53 belongs to its generating policy,
**after-52**, whose completed local-browser evaluation is **92/300 = 30.67%**
overall and **92/227 = 40.53%** valid-only. See the checkpoint table in
[RL_EVALUATION.md](RL_EVALUATION.md). This reward spike did not establish a
held-out improvement. One of the four authorized future top-five evaluation
slots remains; no additional training allocation beyond job 288861 is approved.

<a id="resuming-baseline--allocation-288861-finished"></a>
### Allocation 288861 finished: resume from checkpoint 62

Job **288861** finished at **04:56:25 PDT September 12**, elapsed **7:57:55**,
with Slurm **COMPLETED / exit 0**. Worker exit **124** is the planned timeout,
normalized by the batch controller. The allocation saved **nine iterations,
54–62**, adding **96 durable Adam updates** and reaching **732 total** at
`runs/openwebrl-4b-reference-288861-20260912T040027/iter_0000061`.
Iteration 54 replayed the preceding allocation's saved on-policy batch; the eight
fresh collections 55–62 averaged **30.27 minutes** and preserve **4385 completed
trajectories**, including discarded groups. All fresh rewards and all 96 optimizer
records match W&B. The last 12 optimizer records remain present through history
row **978** after shutdown. W&B reports run state **killed** after the planned
interrupt; this status does not indicate missing optimizer records or a failed
checkpoint, and the same run ID remains the intended resume identity.

| Iteration | Collection minutes | Reward | All-task success | Added Adam updates | Durable total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 61 | 30.2 | 0.493802 | 272/530 = 51.32% | 10 | 720 |
| 62 | 30.1 | 0.504920 | 262/530 = 49.43% | 12 | 732 |

The new checkpoints passed metadata, shard extent, cursor and finite sampled
CPU-tensor checks. Checkpoint 62 has **not** undergone a full GPU reload. The
source recipe hashes and CPU resume-selection checks pass. Host memory peaked
at **403.74 GiB / 480 GiB**, with no GPU/backend errors, host OOM events or health
recorder GPU-query errors. Iteration 61 had an isolated gradient norm **4.628**
that returned below 1.7 for all subsequent updates in that iteration; iteration
62 maximum norm was **2.120**, maximum PPO KL **0.003601**.

Collection **63 was interrupted during browser collection**, last reporting
14/48 accepted, 24 completed and 72 pending groups. It has **no complete batch,
no complete archive and no optimizer updates**. Do not count it as an additional
reward or checkpoint. On the next authorized allocation, fully verify and load
checkpoint **after-62 / index 61**, then **recollect collection 63 from its saved
dataset cursor**. No batch replay or pending scheduled evaluation is needed.
The pointer retains the 32-browser source and matching pending-eval recovery
source. Do not interpret an old replay file for collection 54 as next-batch work.

The separate after-58 evaluation completed (**109/300 overall, 109/230 valid-only**),
as did scheduled after-60 (**105/300 overall, 105/230 valid-only**). Collection
62's reward entered the top five at rank 5, making **after-61** an evaluation
candidate. Its queue entry is **HELD_BUDGET_CAP**: all four authorized triggered
evaluation jobs are already used, and no fifth job was submitted. No further
training allocation is queued or authorized. Details are in
[RL_EVALUATION.md](RL_EVALUATION.md).

Final evidence in this run: `allocation_end_audit.json`,
`shutdown_wandb_audit.json`, per-iteration archive/reward/optimizer audits, and
`checkpoint_61_validation.json`. Persistent resume pointer:
`/gpfs/scrubbed/zixianma/openwebrl-runtime/current_baseline.json`.

<a id="resuming-baseline--allocation-290926-16hour"></a>
### Sixteen-hour continuation: allocation 290926, September 12

The user requested either **8 H200 × eight hours** or **4 H200 × sixteen hours**,
choosing the more efficient option. **Job 290926** was submitted at **11:47 PDT**
for **4 H200, 32 CPUs, 480 GiB, sixteen hours**, with **32 concurrent browsers**:
**64 GPU-hours**, Slurm's estimated maximum GPU cost **$57.60**, plus judge calls.
The normal QoS permits up to 24 hours. Submission evidence:
`logs/submission-qcq7i4ug-after62-4gpu16h-20260912.json` under runtime storage.

The choice favors training progress per GPU-hour. The last eight fresh collections
averaged **30.27 minutes**, with roughly **20–24 further minutes** for training,
transfer and saving. Doubling GPUs would need a twofold speedup across the entire
cycle to match four GPUs over twice the wall time. That has not been measured;
browser work and current 32-task concurrency do not automatically halve. An
illustrative scenario that halves only optimizer time saves about ten minutes
per cycle, less than the twofold improvement required. The user correctly
pointed out that **8 GPUs + 64 CPUs + 64 browsers** could accelerate both phases
and approach twofold throughput. At exactly twofold throughput, 8×8 matches
4×16's training progress and GPU-hours while halving wall time. Below twofold,
it trades GPU efficiency for faster results. Neither is a measured Pareto winner:
**four GPUs are the validated choice**, while the combined eight-GPU configuration
needs its own topology and browser-concurrency benchmark. Forecast: about **16–18 additional
iterations**, ending around **78–80**, including scheduled evaluations. This is
a throughput estimate, not a convergence guarantee or an eight-GPU benchmark.

The preserved launcher previously capped itself at eight hours even in a longer
allocation. `scripts/prepare_run_duration.py` created isolated source
`reference-stage1-browsers32-16hour-20260912`, changing only that launcher cap to
sixteen hours, plus matching source
`reference-stage1-browsers32-pending-eval-16hour-20260912`. All protected recipe
hashes remain identical. The verification cap stays **15 minutes**, and the
actual allocation deadline minus **180 seconds** still bounds training. The old
source and default eight-hour template remain available.

Three duration-boundary tests and sixteen resume regressions pass; shell syntax
and a real preserved-launcher CPU dry run pass, showing **57,600 seconds**, four
GPUs, 48 prompt groups × five attempts, and the unchanged shutdown margin.
Template: `scripts/resume_baseline_4gpu_16hour.sbatch`. Its controller owns and
awaits full GPU restore verification and online continuation. Resume target is
**after-62 / index 61 / 732 Adam updates**, then fresh collection 63, in W&B
`qcq7i4ug`. No replay batch or evaluation is pending at launch. All four prior
standalone evaluation slots remain exhausted; this training request does not
approve additional standalone evaluation jobs.

<a id="eight-gpu-scaling-benchmark-20260912"></a>
### Eight-GPU topology and browser benchmark, September 12

<a id="browser48-shared32cpu-20260913"></a>
**48-browser quick test on the existing 32-CPU allocation.** Job 290926
remained running; GPU topology, baseline source, checkpoint lineage and W&B
`qcq7i4ug` were unchanged. CPU-only overlapping steps tested direct preserved
WebEnv setup/reset during iteration 74's optimizer phase, without model or
judge calls. No new allocation was requested. The initial short 32/48/32
comparison never reached its limits (workers finished before 48 overlapped),
so it is not evidence of 48-browser capacity.

A second 32/48/32 comparison kept every browser open for 20 seconds between
setup and reset, with 250-ms launch spacing and the same 96-URL sequence per
case. All three cases captured **96/96** observations. Timestamp overlap
verified **32/48/32 simultaneously open sessions**, respectively. Durations
were **87.4 / 64.8 / 86.8 seconds**. The 48-browser case therefore ran this
specific controlled probe **1.34x** as fast as the mean of its two controls.
This is **not an RL speedup measurement**: the deliberate hold contributes to
throughput, the optimizer shares CPU capacity, the sample covers six URLs,
and challenge pages still count as successful screenshots (32 per case).
Sampled total job memory peaked at **375.5 GiB / 480 GiB**, with no OOM events.
48 staggered browsers passed this preflight; full rollout throughput and model
serving contention remain untested, so the live default stays at 32.

Results and screenshots:
`/gpfs/scrubbed/zixianma/openwebrl-runtime/benchmarks/browser48-hold-290926-20260913/`;
see `final_audit.json` and `summary.json`. Log:
`/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/browser48-hold-290926-20260913.log`.
[Separate W&B diagnostic](https://wandb.ai/zixianma/openwebrl/runs/browser48-hold-290926-20260913)
finished and synced all three cases. Diagnostic code supports an optional
bounded hold and records setup/reset timestamps; five CPU regression tests pass.

<a id="browser-diagnostic-followup-20260912"></a>
**Prepared follow-up diagnostic.** `scripts/diagnose_browser_scaling.py` uses
the preserved reference `WebEnv`, retaining both navigation passes in the
original `setup()` + `reset()` path. It measures Python import, browser launch,
every navigation attempt, screenshot, reset and cleanup time. Each worker has
its own process group; the controller reaps that group before releasing its
slot. GPU visibility is disabled for browser workers. Successful screenshots,
individual JSON results/logs, aggregate per-URL results and process/CPU/memory
telemetry stay under `benchmarks/browser-diagnostic-JOB/` in runtime storage.
Separate W&B run `browser-diagnostic-JOB` records browser metrics, including
suspected challenge pages separately from screenshot availability.

The six URLs are a local HTML control, `example.com`, `velux.com`, `gtmetrix.com`,
`lens.blogs.nytimes.com` and `discworld.fandom.com`. The serial control visits
each once. Each concurrent case uses the same 96-entry sequence of these URLs:
32 and 64 worker limits, each with immediate launches and 250-ms launch spacing.
96 is attempted only after a complete 64-worker case returns screenshots for
at least 90% of attempts. This is an initial-observation diagnostic, not a
task-success evaluation or steady-state browser-agent workload benchmark.

Preliminary single-browser probes on the **login host**, September 12, passed
the full frozen setup/reset path: local HTML screenshot 0.032 seconds, VELUX
0.189 seconds, GTmetrix 0.047 seconds. Total worker times were 4.97, 10.72 and
6.03 seconds. VELUX displayed its homepage/cookie prompt; GTmetrix displayed a
Cloudflare human-verification challenge. These observations separate screenshot
availability from website access, and cannot replace paired tests on the same
compute node. Evidence is in
`benchmarks/browser-diagnostic-preflight-20260912/` under runtime storage.

Template `scripts/diagnose_browsers_64cpu.sbatch` is prepared but **not submitted**.
It requests **8 H200 / 64 CPUs / 960 GiB / one hour**, at most **8 GPU-hours,
estimated $7.20**, plus judge calls only if the optional RL check starts.
Tillicum has no unbilled CPU-only scheduled allocation: its
[scheduling rules](https://hyak.uw.edu/docs/systems/tillicum/scheduling-jobs/)
limit full-H200 requests to eight CPUs per GPU. The GPUs remain unused during
browser diagnosis. The previous benchmark allocation has ended and cannot be
reused. This is a new budget and still needs explicit approval.

To use any remaining approved time productively, the template enables
`--confirm-rl`: only a complete immediate-launch 64/96-worker case with at least
95% screenshot availability can qualify, and at least 30 minutes must remain.
The controller then attempts one fresh-SFT TP4/DP2 RL iteration in a separate
W&B run and output directory, bounded by the same allocation deadline. This
checks whether browser-only results transfer to the full pipeline; the main
RL run and its pointer are never changed. If no case qualifies or time is short,
the controller exits without starting policy training. Five diagnostic unit
tests and eight benchmark regressions pass; the real local-page probe validates
the timing instrumentation and cleanup without a GPU or judge.

**Approval and submission:** the user approved the above one-hour budget;
job **291905** was submitted on September 12 at **20:47 PDT** (September 13
03:47 UTC). Slurm confirmed 8 H200 / 64 CPUs / 960 GiB / one hour and an
estimated maximum GPU charge of $7.20. Initially queued for resources; elapsed
allocation time begins only when the job starts. Receipt:
`logs/submission-browser-diagnostic-8gpu1h-20260912.json` under runtime storage.
Submitted code is preserved with hashes in
`benchmarks/browser-diagnostic-291905-prepared-code/`. Results and separate W&B
identity will use `browser-diagnostic-291905`. Existing RL job 290926 and W&B
`qcq7i4ug` are outside this diagnostic's write scope.

Scheduler update: g011 was stuck in cleanup for a job that timed out on
September 11, so it was excluded from this pending job without changing its
resource request or budget. At 20:56 PDT September 12, Slurm estimated a start
on **g003 at 03:12 PDT September 13**; this estimate may change. No diagnostic
allocation time has elapsed. Resume monitoring from
`/gpfs/scrubbed/zixianma/openwebrl-runtime/current_browser_diagnostic.json`.

**Revised first stage:** the user requested a shorter topology test before the
browser sweep. Prepared template `scripts/benchmark_topology_8gpu_2hour.sbatch`
requests **8 H200, 64 CPUs, 960 GiB, two hours**: **16 GPU-hours**, estimated
maximum GPU charge **$14.40**, awaiting explicit approval. Its
`--topology-only` controller replays saved batches, makes no new browser/judge
requests, and exits after the topology candidates. Eight concurrent GPUs are
necessary to measure an eight-GPU layout; the budget saving is in duration.
One hour should cover an initial comparison of one or two layouts; two hours
is the recommended budget to attempt all four, allowing roughly 15–30 minutes
per layout including startup, restore and saving. These estimates are unverified
on eight GPUs, and slow restores or failures may leave fewer completed cases.
This stage can select a provisional optimizer layout; browser concurrency and
end-to-end scaling still require the separate browser benchmark below. A CPU
controller test verifies that topology-only mode launches no browser cases.

**Prepared, CPU-checked, awaiting an additional explicit budget approval.**
Proposed allocation: **8 H200, 64 CPUs, 960 GiB RAM, four hours on one node**
(32 GPU-hours; estimated maximum GPU charge **$28.80**, plus judge usage).
The existing four-GPU training job 290926 continues separately. This benchmark
does not update its checkpoint pointer or write into W&B run `qcq7i4ug`.

The prepared controller is `scripts/scaling_benchmark.py`; submission template
is `scripts/benchmark_scaling_8gpu.sbatch`. It owns and awaits each worker and
respects the allocation deadline with a three-minute shutdown margin. Source
`benchmark-eightgpu-20260912` under runtime storage was prepared using
`scripts/prepare_scaling_benchmark.py` from the validated 32-browser reference
snapshot. Recipe hashes are unchanged; modified launcher scripts and candidate
browser YAML files have separate hashes checked before execution.

The comparison has two stages:

1. **Optimizer topology:** TP4/DP2, TP2/DP4, TP8/DP1 and TP1/DP8, in that
   order. Each loads the same after-61 checkpoint (720 Adam updates) and replays
   the exact saved collection-62 batch, with its 144 submitted-group cursor
   advance. A candidate must complete all 12 updates, save the resulting
   checkpoint, and pass model/optimizer restore and checkpoint-counter checks.
   Compare the slowest rank's training timer per update; the existing four-GPU
   run took **1,289.5 seconds for these 12 updates**. Sequence parallelism is
   disabled only for TP1. Inference uses eight separate one-GPU engines for
   every actor topology, so actor TP and rollout engine count are distinct.
2. **Browser concurrency:** use the fastest successful actor topology, starting
   each trial from the same after-62 checkpoint and task cursor. Test gates and
   pools of **64, 96, 128**, then **192 and 256** if throughput, memory headroom
   and remaining time permit. Each trial completes a fresh 48-group × five-attempt
   collection, optimizer work and checkpoint save. Score elapsed time from
   generation start through durable save, and **8 × seconds / 3,600 GPU-hours**
   per iteration. Startup and checkpoint-loading time are reported separately
   through total case wall time, rather than included in the steady-state score.

The largest useful browser count is an empirical result. 256 is this search's
upper bound, not a validated capacity claim. Stop escalation after a failed
128-or-higher case, no throughput improvement at 128 or higher, or sampled host
memory above 85% of the allocation. OOM, incomplete optimizer work, nonfinite
loss/gradient/KL, missing timing, and invalid checkpoint counters disqualify a
candidate. Case time limits are 30 minutes for replay and 40 minutes for full
browser trials; allocation time can prevent later candidates from running.

Artifacts go to `benchmarks/qcq7i4ug-scaling-JOB/` under runtime storage:
per-case plans, full logs, checkpoint validation, GPU/memory `health.jsonl`,
CPU usage/pressure `cpu.jsonl`, results and a provisional summary. W&B runs use
`qcq7i4ug-scale-JOB-CASE`, preserving the baseline's existing training metrics.
Before promoting a configuration, inspect W&B synchronization, browser timeout
and invalid-task rates, GPU/CPU use, memory headroom and saved checkpoints.
These are single live-web trials, so repeat the promising configuration across
several baseline iterations before claiming a stable optimum. The four-hour
budget is a bounded first comparison, not a guarantee to exhaust the search.

CPU evidence: six unit tests pass; eight preserved-launcher dry runs and eight
shell-argument dry runs cover all topologies and browser limits, including TP1's
sequence-parallel exception. Reports are
`logs/scaling-eightgpu-cpu-preflight-20260912.json` and
`logs/scaling-eightgpu-shell-preflight-20260912.json` under runtime storage.
No eight-GPU restoration, throughput or browser capacity has yet been measured.

<a id="resuming-baseline--four-gpu-continuation"></a>
### Four-GPU continuation

The model/optimizer checkpoint format supports a different tensor-parallel size.
The default four-H200 profile uses TP4/DP1, 32 browser slots, a 32-task gate,
and 32 CPUs / 480 GiB RAM on one node. The resume wrapper retains a 16-CPU
minimum for older profiles; that minimum is not the current default request. Checkpoint 14 received a successful full TP4 model
and optimizer reload in job 284885. Checkpoint 17 also passed full TP4 restore
in job 285546. The latest checkpoint and prepared source still require their
own restore verification in each new allocation. Two separate two-GPU
allocations do not satisfy this single-node profile.

The default prepared source snapshot is:
`/gpfs/scrubbed/zixianma/openwebrl-runtime/reference-stage1-browsers32-20260911`.
It preserves all five original protected recipe-file hashes and additionally
hashes the YAML containing the 32-task gate. To recreate such a snapshot
from a newer preserved baseline (never the experimental working tree):

```bash
python3 scripts/prepare_resume_topology.py --source PRESERVED_SOURCE --output NEW_SOURCE
```

For an existing four-H200 allocation explicitly authorized by the user:

```bash
python3 scripts/resume_baseline.py --job-id JOB_ID --gpus 4 --source PREPARED_SOURCE --verify-resume-only --dry-run
python3 scripts/resume_baseline.py --job-id JOB_ID --gpus 4 --source PREPARED_SOURCE --verify-resume-only --launch
python3 scripts/resume_baseline.py --job-id JOB_ID --gpus 4 --source PREPARED_SOURCE --dry-run
python3 scripts/resume_baseline.py --job-id JOB_ID --gpus 4 --source PREPARED_SOURCE --launch
```

The restore-only pass runs offline, requests zero browser collections and zero
optimizer updates, and does not replace the current training pointer. Its
launcher is limited to 15 minutes within the existing allocation. Only an actual
successful `resume_verification.json` creates a verification receipt. Four-GPU
training requires a matching receipt for the checkpoint, source launcher and
allocation. If a newer checkpoint becomes available, repeat the restore check
for that checkpoint. Inspect the report and released Slurm steps before training.

The old trainer must be stopped at a safe boundary before the new trainer starts;
the wrapper refuses to fork the same lineage while its recorded old allocation
still has active steps. Offline verification may run separately while the old
trainer remains active, inside another explicitly authorized allocation. A
complete untrained rollout can be replayed after migration, preserving collection
work. The same W&B ID, dataset cursor, optimizer and scheduler state are retained.

At an evaluation boundary, the checkpoint is saved **before** evaluation. Do not
treat a saved checkpoint alone as proof that its evaluation has completed. When
`pending_evaluation_iteration_one_based` is set, the resume wrapper requires the
checkpoint at that exact boundary and a verified pending-evaluation source. It
finishes that evaluation before the next collection, using the original W&B run
and evaluation iteration. No optimizer updates are replayed for this recovery.

The prepared source is runtime `reference-stage1-pending-eval-20260911`, recorded
as `pending_evaluation_resume_source` in the pointer. It was copied from the
preserved source with `scripts/prepare_resume_pending_eval.py`; the five recipe
hashes and launcher hash are unchanged. Only the training driver gains a guarded
pre-collection evaluation call and the usual post-evaluation cache release.
The four-GPU restore verification still runs first. Five CPU regression tests
cover checkpoint identity, evaluation failure, ordinary-resume behavior, source
integrity, and stale environment controls. The complete recovery path passed on
GPUs in job **287530**: exact after-40 restoration, 300-task evaluation completed
in 49:31, all 41 metrics matched main-run W&B history row 660 at evaluation
iteration 40, then collection 41 started. The audit is
`runs/openwebrl-4b-reference-287530-20260911T105645/iteration_40_scheduled_eval_audit.json`
under runtime storage.

After recovery, verify the complete 300-task metrics and W&B synchronization,
then clear the pending field. `[ResumePendingEvaluation] ... status=completed`
alone is not proof of remote W&B synchronization. This runs inside the already
approved training allocation and does not submit another job or consume one of
the four separately approved reward-ranked evaluation jobs.

Fourteen CPU resume tests cover resource/ownership checks, topology arguments,
verification receipts, preservation of the training pointer, replay selection,
and prevention of concurrent trainers. The prepared inner launcher dry run
selected checkpoint 8 / 130 Adam updates with four GPUs, TP4, 32 browsers,
48 groups × five attempts, global batch 256 and two PPO epochs. The live two-GPU
allocation correctly rejects a four-GPU request. No four-GPU allocation was
requested or consumed by these tests.

<a id="resuming-baseline--batch-driver"></a>
### Batch driver

`scripts/resume_baseline_4gpu.sbatch` requests one node, four H200s, sixteen CPUs,
480 GiB RAM, and eight hours in the normal QoS. Its two sequential stages are
restore-only verification followed by training/replay. The wrapper recognizes
its own batch driver while continuing to reject other compute steps and live
trainers in another allocation. Both stages share the one allocation's time
boundary; verification does not buy or extend time.

The user requested this resource configuration on 2026-09-08. Slurm's
`--test-only` accepted it and estimated $28.80 for 32 H200 GPU-hours. Automatic
approval review subsequently rejected submission pending explicit approval of
the dollar estimate; no batch job was submitted by that rejected action.
The script does not grant standing approval to submit paid jobs.

The first approved submission, job `283947`, exited after 46 seconds before any
model restore because its child `srun` mixed a typed H200 allocation with an
untyped GPU GRES request. Four-GPU child steps now request `gpu:h200:4`; the
regression suite checks the typed request. No checkpoint or W&B state changed.

<a id="resuming-baseline--state-and-source"></a>
### State and source

The default persistent pointer is:

`/gpfs/scrubbed/zixianma/openwebrl-runtime/current_baseline.json`

It records the current run directory, checkpoint ancestry, preserved reference source, W&B identity and pending recovery batch. `--state PATH` selects a different recorded lineage. `--source PATH` selects another explicitly prepared reference snapshot; recipe hashes and required resume support are checked. The working repository's experimental recipe files are not copied into the run.

For the current baseline, W&B is [`qcq7i4ug`](https://wandb.ai/zixianma/openwebrl/runs/qcq7i4ug), and the preserved source for the next resume is `reference-stage1-tp4-eval-cache-20260909` under the runtime root. This snapshot adds post-evaluation file-cache release while preserving the five baseline recipe hashes. The credentials come from the repository `.env` and are never included in the preflight plan. `--wandb-run-id qcq7i4ug` can assert the expected identity; it cannot silently change this lineage's W&B ID.

Job 285546 was canceled at the user's request on 2026-09-10 after 7:24:54.
Resume from checkpoint 22 at 302 Adam updates. Reward observation 23 and the
iteration-20 evaluation are complete; collection 24 was unfinished and has no
saved recovery batch, so it must be collected again. Checkpoint 22 passed CPU
metadata, shard-extent, cursor, and small finite-payload checks; the batch
driver must perform its full GPU restore before training. No replacement
allocation is authorized by this document.

<a id="resuming-baseline--checkpoint-and-batch-selection"></a>
### Checkpoint and batch selection

The script follows `launch_manifest.json` ancestry, selects the highest completed checkpoint marker, and verifies saved iteration, Adam counters, known scheduler offset, shard extents and matching dataset cursor. A partial checkpoint directory without a completion marker is ignored. If the newest marked checkpoint fails verification, resume stops for investigation; it does not silently roll training back. This CPU validation does not substitute for the actual model/optimizer restore on GPUs.

If a complete saved rollout immediately follows that checkpoint, it is replayed before collecting fresh browser trajectories. Replay requires either matching `.provenance.json` or a unique completed `[GenerateProgress]` line. The restored dataset cursor advances by **completed + pending submitted prompt groups**, not just the 48 accepted groups. Already-trained batches are skipped. An incomplete or ambiguous recovery file stops preflight for inspection. Zip-directory checks detect interrupted saves but do not read every tensor byte; the trainer's load is the full payload check.

Example: checkpoint 5 contains 90 Adam updates; saved batch `6.pt` has 1,946 turns and 144 submitted groups. Replay performs `floor(1946 / 256) × 2 = 14` updates, reaching 104 at checkpoint 6. This remains reward observation 7, not a new collection. W&B's historical `train/step` labels are not reliable cumulative Adam counts when batch lengths change.

<a id="resuming-baseline--monitoring-and-limits"></a>
### Monitoring and limits

The launcher retains the reference recipe (48 accepted groups × five trajectories, global batch 256, two PPO epochs), the tested runtime fixes, lossless file-backed images during collection and training transfer, allocator trimming, and per-iteration checkpointing. Runtime files stay in scrubbed storage or node-local `/tmp`. It stops before the existing allocation ends, with a three-minute margin and a maximum of eight training hours. GNU `timeout` reports this planned online stop as raw exit code 124; the resume supervisor keeps that raw code in its exit receipt, records the pointer state as `TIME_LIMIT`, and returns success to Slurm. Verification timeouts and other launcher failures remain failures. A collection interrupted by the time boundary may need recollection unless its recovery file finished saving.

The supervisor prints progress and starts a read-only health recorder. Logs:

- `openwebrl-runtime/logs/resume-JOB_ID-TIMESTAMP.log`: launcher output.
- `openwebrl-runtime/logs/resume-JOB_ID-TIMESTAMP.json`: preflight/resume plan.
- Run directory `training.log`, `progress.log`, `health.jsonl`: detailed training, phase progress and GPU/cgroup memory samples.
- Run directory `resume_plan.json` and `resume_checkpoint_validation.json`: startup provenance.

The pointer is updated when the new launch manifest appears. This means **launched**, not verified healthy. After startup, confirm the actual checkpoint-loaded message, optimizer progress and W&B sync. After each save, validate the new checkpoint before reporting durable progress and refresh the pointer's checkpoint/replay fields. A background recorder cannot diagnose or fix failures. While an OpenWebRL run is active, the active agent checks health and updates the user about every 15 minutes, checks more closely around startup and checkpoint saves, and investigates any observed error immediately.

CPU checks: `python3 -m unittest discover -s tests -p test_resume_baseline.py -v`. The real preflight was exercised against job 283214 and correctly selected checkpoint 5, replay batch 6 with 144 submitted groups, and the existing W&B ID. It identified the already-running trainer and monitor; no duplicate training run was launched for this test.

The command also launched the real continuation `openwebrl-4b-reference-283214-20260909T022832` from checkpoint 6, selecting no replay and starting the health recorder automatically. Source validation rejects older snapshots lacking collection-time image mapping and allocator trimming, which are required for this 240-GiB workflow.


<a id="resuming-baseline--reward-ranked-evaluation-queue-2026-09-11"></a>
### Reward-ranked evaluation queue (2026-09-11)

During supervision, apply the user's top-five `train/reward` evaluation trigger
after each verified collection. Runtime `reward_eval_queue.json` tracks ranking,
checkpoint identity and submitted jobs. The user explicitly approved at most
four standard local-browser evaluation jobs, each two H200s for up to two hours;
use `reward_eval_approval_20260911.json` and do not re-ask within that cap.
See `REWARD_RANK_EVALUATION_QUEUE.md` for the exact invocation and accounting.
Stealth evaluations are paused. Keep monitoring training and these evaluations.

<!-- document:RESUMING_BASELINE.md:end -->

---

<!-- document:BASELINE_SCALING.md:start -->
<a id="baseline-scaling"></a>
## Baseline timing and four-GPU continuation

_Source record: `BASELINE_SCALING.md`. Dated entries retain their historical context._


Snapshot: 2026-09-08 21:30 PDT, W&B `qcq7i4ug`, job `283214` on `g021`.
Eight collection-and-training iterations are complete (116 durable Adam
updates); collection nine is in progress. These iterations are the reward
observations, not individual PPO optimizer updates.

The latest fresh cycle took approximately 87 minutes: 63 minutes of browser
collection, 22 minutes of training, and the remaining time for transfer,
checkpointing and weight synchronization. Using that cycle, and accounting
for collection nine being partly complete, reaching iteration 30 needs roughly
31–35 additional wall hours on two H200s; iteration 40 needs roughly 46–50.
These are planning estimates, not measured convergence or a guaranteed runtime.
Full 300-task Online-Mind2Web evaluations every ten iterations add time; their
current-runtime duration has not yet been measured. Browser latency, trajectory
length and dynamic-filter acceptance also change cycle duration.

<a id="baseline-scaling--checkpoint-portability"></a>
### Checkpoint portability

The current two-GPU topology is tensor parallelism 2, data parallelism 1.
The checkpoint stores optimizer state in `fully_sharded_model_space` format.
The installed Megatron implementation lists this as fully reshardable and
permits loading optimizer state when tensor/pipeline parallel sizes change:

- `openwebrl-runtime/src/Megatron-LM/megatron/core/optimizer/distrib_optimizer.py`,
  `checkpoint_fully_reshardable_formats` and `sharded_state_dict`.
- `openwebrl-runtime/src/Megatron-LM/megatron/training/checkpointing.py`,
  optimizer topology-mismatch checks in `load_checkpoint`.

Thus the checkpoint format does not bind this lineage to two GPUs. Four saved
shard files likewise do not imply that four GPUs are required. Actual model
and Adam-state restoration on four GPUs remains untested.

Four-GPU options include TP4/DP1, matching the paper's recorded tensor
parallel size, or TP2/DP2. Keep 48 accepted prompt groups, five attempts per
group, global batch 256, two PPO epochs, learning rate and data cursor unchanged.
Retain W&B ID `qcq7i4ug`, Adam counters and scheduler state. A topology change
is not a bitwise-equivalent continuation; the loader may discard incompatible
RNG state when tensor/pipeline parallelism changes.

The quick-resume wrapper now accepts `--gpus 4` with a topology-capable preserved
source prepared by `scripts/prepare_resume_topology.py`. This profile uses TP4,
32 browsers, and guards for 16 CPUs / 480 GiB RAM on one node. Thirteen CPU tests
and the real launcher dry run pass; four-GPU restoration and performance remain
unverified. See [RESUMING_BASELINE.md](RL_RUNTIME.md#resuming-baseline) for the required
restore-only pass and migration procedure. Older launchers still force two GPUs;
setting environment variables alone does not change those launchers.

<a id="baseline-scaling--expected-speedup-and-migration-checks"></a>
### Expected speedup and migration checks

If four GPUs halve only the 22-minute training portion, an 87-minute cycle
becomes roughly 76 minutes: about 1.14x faster. That is an illustrative ceiling
for that assumption, not a measured four-GPU benchmark. More browser concurrency
and CPU capacity are needed to substantially reduce the dominant collection
phase. The current job has eight CPUs and sixteen browser slots; a bounded
probe during collection nine averaged approximately six occupied CPU cores.
Doubling browser concurrency without additional CPU headroom is unvalidated.

Switch after a completed checkpoint, preserving any complete next rollout for
replay. First verify four-GPU model/optimizer restoration and counters, then
measure a complete collection-and-training cycle before buying a long run.
Keep full recomputation and the other known-working runtime controls for the
initial migration; assess their performance separately once restoration works.

No new compute is approved by this analysis. A new allocation or batch job
requires the user's explicit approval of the exact GPU count/type, CPU and RAM
request, duration, and compute budget. Continue the existing authorized job
until a checkpoint boundary and an approved replacement allocation are ready.

<!-- document:BASELINE_SCALING.md:end -->

---

<!-- document:ROLLOUT_ARCHIVE.md:start -->
<a id="rollout-archive"></a>
## Completed rollout archive for future SFT

_Source record: `ROLLOUT_ARCHIVE.md`. Dated entries retain their historical context._


The user requested preservation of discarded all-success and all-failure
groups on 2026-09-10. Earlier recovery `.pt` files contain accepted training
turns only. Sparse debug traces (configured probability 0.005) do not provide
complete rejected-group coverage. Prior discarded groups generally cannot be
reconstructed from aggregate metrics.

The archive saves **all completed groups**, before their in-memory telemetry
data is released, while preserving RL filtering and rewards. It excludes
in-flight attempts canceled at the collection cutoff and unfinished collections
interrupted before the archive hook runs. It records actual acceptance by sample
identity; `not_accepted` can also include excess completed groups at cutoff and
is not an exact dynamic-filter reason.

<a id="rollout-archive--location-and-contents"></a>
### Location and contents

For each run, inspect `completed_rollout_archive/iteration_NNNN_ID/manifest.json`.
Only directories with a complete manifest represent complete archives. Each
`group_NNNN.json.gz` contains group/trajectory IDs, terminal reward and validity,
acceptance status, turn-level prompts/responses, tokens, loss masks, stored log
probabilities, conversation metadata, and raw multimodal inputs. Group labels
are `all_success`, `all_failure` (all valid rewards zero), `all_nonpositive`,
`mixed`, and `contains_invalid`. These labels do not replace the actual rewards.

Trajectory-level `reward` is null for invalid attempts, even when their turns
retain numeric raw rewards. Recompute `train/reward` from accepted **turn** rewards,
not the validity-filtered trajectory reward. The preserved reference filter can
accept a trajectory marked `remove_sample`; training subsequently zeros its loss
masks, but its raw reward still enters the collection reward and group reward
normalization. Collection 30 demonstrated this: four masked turns with raw reward
zero among 1,686 accepted turns. Its complete archive reproduced W&B reward
`0.43001186239620404` and task success `259/610`. This is existing baseline behavior,
not an archive-induced change. Evidence: `iteration_30_archive_audit.json` and
`iteration_30_reward_wandb_audit.json` in
`openwebrl-runtime/runs/openwebrl-4b-reference-286382-20260910T164338`.

Screenshots are stored once per SHA-256 in the archive's `images/` directory.
JSON image references contain a relative path, MIME type, byte count and hash;
resolve them relative to `image_reference_root` in the manifest. Original data
URL bytes are retained, and PIL images are encoded losslessly as PNG. Files use
`.bin` regardless of MIME type. The export avoids processed image tensors;
recreate those with the intended SFT processor. Preserve the group JSON, shared
image directory and source run manifest together when moving an archive.

Use valid successful trajectories as candidates for positive SFT. Keep failures
for analysis, correction, or explicitly designed negative training; copying
failed actions into a standard positive SFT target would teach those actions.
Review screenshot availability and success-judge quality when building a dataset.

<a id="rollout-archive--enablement-and-validation"></a>
### Enablement and validation

`rollout_archive.enabled.json` in the run directory enables the archive, as does
`OPENWEBRL_ARCHIVE_COMPLETED_GROUPS=1`. The baseline resume wrapper creates the
control file automatically when its preserved source supports archiving. The
telemetry hook reads the completed one-based collection number from `progress.log`.
Archive errors appear in training logs, `rollout_archive_errors.jsonl`, and
`rollout/archive/errors`; they do not change RL filtering or stop training.
W&B also receives group/trajectory/image counts and archive duration.

Seven CPU tests covered existing metric denominators, unchanged training
metrics/sample data, valid/invalid outcome classification, image deduplication
and byte recovery, opt-in behavior, and visible error reporting. A CPU-only
export inside allocation 286094 verified a real six-turn trajectory from saved
collection 23: one trajectory, six unique images, about 95 KB compressed group
JSON, and about 0.07 seconds exporting. That checks the schema on real data;
live full-collection archive completion is a separate verification.

The hook was installed before collection 24 completed in job 286094. The
telemetry module is imported at collection completion; no worker restart was
needed. `rollout_archive_installation.json` and `archive_source_before/` in the
run preserve installation provenance. All five baseline recipe hashes remain
unchanged.

Collection 24's live archive completed at 01:00 PDT: 88 groups, 440 trajectories,
3,203 turns, and 2,290 unique images. It includes 11 rejected all-success groups,
12 rejected all-failure groups, 17 rejected groups containing invalid attempts,
and all 48 accepted groups. Archiving took 31.2 seconds. Every group JSON and
image reference passed a subsequent audit, including byte sizes and SHA-256 for
all images. Storage was 831,557,303 image bytes plus 59,114,630 compressed JSON
bytes. Evidence: `iteration_24_archive_audit.json` in
`openwebrl-runtime/runs/openwebrl-4b-reference-286094-20260910T071441`.

The archive independently reproduced W&B history row 420:
`train/reward=0.40914285714285714` across 1,750 accepted turns, and
`train/task_success_rate=173/440=0.3931818181818182`. See
`iteration_24_reward_wandb_audit.json`. Earlier rejected collections remain
unavailable except any sparse debug traces that happened to be saved.

<!-- document:ROLLOUT_ARCHIVE.md:end -->

---

<!-- document:H200_TESTING.md:start -->
<a id="h200-testing"></a>
## H200 runtime and validation

_Source record: `H200_TESTING.md`. Dated entries retain their historical context._


<a id="h200-testing--continuation-287530-started-2026-09-11-0354-pdt"></a>
### Continuation 287530 started, 2026-09-11 03:54 PDT

The user requested and explicitly approved another **4 H200 × 8-hour** training
allocation, 16 CPUs / 480 GiB, **32 GPU-hours / estimated $28.80**, after job
287371. **287530** started on **g022 at 03:54:49 PDT**, after its
`afterok:287371` dependency completed. Its deadline is **11:54:52 PDT**.
It restores checkpoint after **40 / directory 39 / 490 Adam updates** in W&B
lineage `qcq7i4ug`. Submission receipt: runtime
`logs/submission-qcq7i4ug-287530.json`. Full four-GPU restoration verification
runs first, followed by the pending iteration-40 evaluation and online training.
The prepared source `reference-stage1-pending-eval-20260911` preserves all five
recipe hashes and the launcher hash; see `RESUMING_BASELINE.md` for the guarded
evaluation recovery. An incomplete evaluation starts its 300 tasks afresh.

The actual four-GPU full model/optimizer restore passed with zero updates and
zero browser collections in the verification stage. Receipt:
`logs/resume-287530-4gpu-verification.json`. The online continuation directory is
`runs/openwebrl-4b-reference-287530-20260911T105645`; its logs and persistent pointer
retain W&B `qcq7i4ug`.

At the user-requested cancellation at **11:32 PDT**, iterations **41–45**
were saved and verified, reaching **550 cumulative Adam updates** at
`iter_0000044`. Their rewards were 0.480380, 0.431971, 0.471503, 0.431949
and 0.472973; all five rewards and all 60 optimizer records match W&B.
Complete archives and checkpoint metadata, shard extents, cursors and finite
CPU tensor samples passed. The new allocation will perform a full GPU reload
of checkpoint 45. Collection 46 had 21/48 accepted groups when interrupted;
no completed recovery batch existed. No GPU/backend or host OOM was observed.
The allocation-end audit is in this run directory.

The pending after-40 evaluation completed all 300 tasks before collection 41:
**100/300 (33.33%)**, valid-only **100/231 (43.29%)**. All 41 evaluation scalars
match W&B history row 660. Its audit is `iteration_40_scheduled_eval_audit.json`
in the current run; the persistent pending-evaluation field is now cleared.

The user also approved up to four separate two-H200 evaluation jobs, each up to
two hours, triggered by future verified rewards entering the top five. See
`REWARD_RANK_EVALUATION_QUEUE.md`. Stealth evaluation 287521 was canceled on the
user's request; its 16 recorded cloud sessions were confirmed stopped.

<a id="h200-testing--job-287371-completed-iterations-3540"></a>
### Job 287371 completed iterations 35–40

Slurm reports **COMPLETED / 7:57:23 / exit 0**. Its worker reached the planned
allocation timeout (raw 124), which the controller normalized to success.
All six reward observations and **62 added optimizer updates** matched W&B.
The final durable checkpoint is `iter_0000039` with **490 cumulative Adam
updates**, under runtime `runs/openwebrl-4b-reference-287371-20260911T025909`.
All checkpoint extents/cursors and finite CPU samples passed validation.

| Iteration | Train reward | Added Adam updates | Cumulative Adam updates |
| --- | ---: | ---: | ---: |
| 35 | 0.506224 | 10 | 438 |
| 36 | 0.493810 | 10 | 448 |
| 37 | 0.483221 | 10 | 458 |
| 38 | 0.442043 | 10 | 468 |
| 39 | 0.487730 | 12 | 480 |
| 40 | 0.503511 | 10 | 490 |

The scheduled after-40 evaluation stopped at **201/300 tasks** without a final
metric row; it is not a valid completed score. The pending field was carried
into 287530, which completed the evaluation as recorded above. Final audit:
`allocation_final_audit.json` in that run.
Peak host memory was **372.34 GiB**, with no host OOM; sampled average GPU
utilization over the allocation was **41.94%** (utilization, not MFU).
Completed-group archives include RL rejections and passed image SHA256 checks.

Top-five rewards at collections 39 and 40 triggered evaluations of their
generating checkpoints after 38 and 39. Job **287588** completed after-38 with
**107/300 (35.67%)**, valid-only **107/228 (46.93%)**, and all 41 W&B scalars
verified. Job **287596** completed after-39 with **98/300 (32.67%)**, valid-only
**98/228 (42.98%)**, and all 41 W&B scalars verified, following a supervised retry
that cleared an inherited cross-node W&B service socket. Both startup repairs
stayed within their original allocations; **2/4 approved evaluation jobs** are used.

<a id="h200-testing--approved-continuation-287371-2026-09-10-1957-pdt"></a>
### Approved continuation 287371, 2026-09-10 19:57 PDT

The user approved **4 H200 GPUs for 8 hours** (32 GPU-hours, estimated $28.80),
16 CPUs and 480 GiB RAM. Job **287371** runs on **g005**, with an allocation
ending September 11 at **03:57 PDT** and planned shutdown about three minutes
earlier. It resumes W&B `qcq7i4ug` using the preserved reference source.

The full TP4 restoration check passed for checkpoint **33 / 428 Adam updates**,
representing completed training iteration **34**. Its receipt is runtime
`logs/resume-287371-4gpu-verification.json`. Online training restored the same
checkpoint and began collection **35**. The next scheduled evaluation is **40**.
Reaching iteration 40 requires six collections and updates, plus that evaluation;
the eight-hour budget is tight at recent throughput and is not a guarantee.

Current run: runtime `runs/openwebrl-4b-reference-287371-20260911T025909`.
Local logs are `training.log`, `progress.log`, and `health.jsonl` there; the
controller log is `logs/slurm-qcq7i4ug-287371.out`. The submission receipt is
`logs/submission-qcq7i4ug-287371.json`, and `current_baseline.json` points to this
run. The first startup inspection found no GPU/backend or host OOM events.

The separately approved intermediate evaluation job **287370** runs on **g004**,
4 H200s for three hours, comparing checkpoints after iterations 21 and 22.
Its details and validation fixes are in `BASELINE_CHECKPOINT_EVALUATION.md`.

<a id="h200-testing--job-286382-completed-five-iterations-and-evaluation-30-2026-09-10"></a>
### Job 286382 completed five iterations and evaluation 30, 2026-09-10

The approved four-H200 allocation ended at its planned shutdown boundary around
17:39 PDT after **7:57:24**, with Slurm state COMPLETED. Raw launcher timeout 124
was normalized to success. Iterations **30–34** completed, adding **54 Adam
updates**, and the durable checkpoint is **`iter_0000033` / 428 updates** in
`openwebrl-runtime/runs/openwebrl-4b-reference-286382-20260910T164338`.
All 54 optimizer records and all five reward observations were checked against
W&B history. The checkpoint passed metadata, all referenced shard-extent and
cursor checks plus finite CPU tensor sampling; a full GPU reload of this final
checkpoint remains a prerequisite for the next continuation.

| Iteration | Train reward | Completed-attempt success | Adam updates |
| --- | ---: | ---: | ---: |
| 30 | 0.430012 | 42.46% | 12 |
| 31 | 0.504698 | 43.84% | 10 |
| 32 | 0.477212 | 48.15% | 10 |
| 33 | 0.469777 | 46.48% | 10 |
| 34 | 0.419214 | 41.60% | 12 |

Evaluation 30 completed all 300 tasks in about 51 minutes: **96 successes
(32.0%)**, 52 invalid attempts (17.33%), and valid-only success 96/248 (38.71%).
All 41 evaluation scalars matched W&B history row 525. Evaluation 20 had 95
successes and 68 invalid attempts; the success difference is only one task.
The automatic post-evaluation file-cache advice ran without error. Host memory
peaked at **414.65 GiB**, with zero memory-limit/OOM events. No GPU/backend errors
were observed, and the maximum gradient norm across these updates was 3.47.

The archive preserved **537 completed groups / 2,685 trajectories**, including
96 rejected all-success groups and 74 rejected all-failure groups. All completed
archives' group JSON and image sizes/hashes were checked. Collection 30 exposed
an audit assumption about masked raw rewards; the auditor was corrected and
the existing baseline behavior documented in `ROLLOUT_ARCHIVE.md` and `METRICS.md`.
Training and reward computation were not changed.

Collection 35 was interrupted at **42/48 accepted groups**, with 85 completed
and 11 pending groups. It has no complete recovery batch or archive and must be
recollected after restoring checkpoint 33. Evaluation 30 is already complete;
the next periodic evaluation is 40. The authoritative handoff is
`current_baseline.json`, with details in this run's `allocation_end_audit.json`.
Slurm child step 2's exit 124 is the planned cutoff; child step 4 was the initial
CPU archive-audit failure, not a training failure.

The separately approved intermediate-checkpoint evaluation job 287046 failed
at scheduler startup before evaluating tasks. Its fix passed CPU restoration
checks on both selected checkpoint schedulers; a replacement allocation awaits
explicit approval. See `BASELINE_CHECKPOINT_EVALUATION.md` for the saved checkpoint
inventory, target selection, failure evidence, and prepared replacement.

<a id="h200-testing--approved-continuation-286382-2026-09-10-0947-pdt"></a>
### Approved continuation 286382, 2026-09-10 09:47 PDT

The user explicitly approved another four H200 GPUs for eight hours, 16 CPUs,
and 480 GiB RAM: 32 GPU-hours, estimated $28.80. Job 286382 is running on g002
from 09:41 to 17:41 PDT, with a three-minute shutdown margin. Full four-GPU
model/optimizer restoration of checkpoint 28 / 374 Adam updates passed before
the online stage started. W&B `qcq7i4ug` reports running, and collection 30 is
underway. Evaluation 30 is due after that iteration trains and saves; it must
not be skipped. The previous evaluation took about 52 minutes.

Completed-group archiving is enabled automatically. The resume wrapper now
refreshes its archive-directory pointer instead of carrying over the prior
run's path; 22 CPU checks passed, including that regression. The current run
directory is `openwebrl-runtime/runs/openwebrl-4b-reference-286382-20260910T164338`.
The four-GPU restore receipt is `logs/resume-286382-4gpu-verification.json` under
the runtime root. Preserve the baseline recipe and monitor about every
15 minutes, with closer checks around failures and stage transitions.

<a id="h200-testing--job-286094-completed-six-additional-iterations-2026-09-10"></a>
### Job 286094 completed six additional iterations, 2026-09-10

The allocation ended at its planned shutdown boundary around 08:10 PDT after
7:57:23, with Slurm state COMPLETED. The training launcher's raw timeout code
124 was normalized to success by the supervisor. Iterations 24–29 completed,
adding 72 Adam updates and reaching checkpoint 28 / **374 durable updates**.
All 72 optimizer records and all six reward observations match W&B. The final
checkpoint passed metadata, shard-extent, cursor and finite CPU-sample checks;
its full GPU reload is required before the next training continuation.

| Iteration | Train reward | Completed-attempt success |
| --- | ---: | ---: |
| 24 | 0.409143 | 39.32% |
| 25 | 0.484587 | 48.79% |
| 26 | 0.426119 | 42.32% |
| 27 | 0.486834 | 53.89% |
| 28 | 0.456150 | 41.18% |
| 29 | 0.456683 | 46.87% |

Host memory peaked at 401.03 GiB with zero memory-limit, OOM, or OOM-kill
events. One finite gradient-norm spike to 12.39 was investigated; the next
update returned to 1.23 and subsequent iterations saved normally. See
`GRADIENT_DIAGNOSTICS.md`. Collection 30 was interrupted at 17/48 accepted
groups and has no complete recovery batch or archive. It must be recollected;
evaluation 30 remains due after that iteration trains and saves.

The new SFT-oriented archive preserved **608 completed groups / 3,040
trajectories**, including 107 discarded all-success groups and 86 discarded
all-failure groups. Every completed archive's group records and image hashes
were checked, and its independently calculated rewards matched W&B.
Earlier rejected collections and the unfinished collection 30 are not fully
archived. See `ROLLOUT_ARCHIVE.md` for schema and coverage.

Evidence and the next-resume handoff are in `allocation_end_audit.json` under
`openwebrl-runtime/runs/openwebrl-4b-reference-286094-20260910T071441`, and
`openwebrl-runtime/current_baseline.json`. A new four-H200, eight-hour request
passed Slurm preflight at an estimated $28.80; preflight does not submit a job
or authorize further spending.

<a id="h200-testing--approved-continuation-286094-2026-09-10-0017-pdt"></a>
### Approved continuation 286094, 2026-09-10 00:17 PDT

The user explicitly approved four H200 GPUs for eight hours, 16 CPUs and
480 GiB RAM: 32 GPU-hours, estimated $28.80. Job 286094 is running on g003,
ending at 08:12 PDT, with a three-minute training shutdown margin. Its batch
controller completed the full four-GPU model/optimizer restore of checkpoint
22 at 302 Adam updates, then launched online collection 24 in the same W&B
lineage `qcq7i4ug`. W&B reports the run as running. The prepared
`reference-stage1-tp4-eval-cache-20260909` source preserves the recipe and adds
the post-evaluation cache release described below.

The preceding job 285546 was canceled at the user's request after 7:24:54.
It trained saved collection 19 and fresh collections 20–23, reaching checkpoint
22 / 302 durable updates, and completed evaluation 20. Collection 24 stopped
at 35/48 accepted groups with no saved recovery batch and must be recollected.
Fresh collection times were 54–64 minutes, with roughly 19–23 minutes for
training/save. The new allocation is expected to finish another five or six
iterations, reaching about 28–29; this is an estimate, not a guarantee. The
next scheduled evaluation is iteration 30.

Current files are under runtime `runs/openwebrl-4b-reference-286094-20260910T071441`;
submission and restore receipts are `logs/submission-qcq7i4ug-286094.json` and
`logs/resume-286094-4gpu-verification.json`. Supervise about every 15 minutes,
audit rewards and optimizer records in W&B at iteration boundaries, validate
each saved checkpoint, and update `current_baseline.json`. No additional
allocation or budget extension is authorized.

<a id="h200-testing--evaluation-cache-retention-corrected-live-2026-09-09-1957-pdt"></a>
### Evaluation cache retention corrected live, 2026-09-09 19:57 PDT

During fresh collection 21, host memory rose to 358 GiB despite zero cgroup
memory-limit/OOM events. Cgroup accounting showed about 294 GiB in the `file`
category (including 124 GiB shared memory) and 59 GiB anonymous memory.
The existing cache-release calls covered training batches but omitted the
evaluation recovery file and completed evaluation image mappings.

A CPU-only step inside authorized job 285546 applied `fsync` and
`POSIX_FADV_DONTNEED` to `rollout_recovery/eval_19.pt` and node-local mapping
files written before collection 21 began at 19:36:55 PDT. It retained all files,
preserved their sizes, excluded current-collection mappings, and reported no
errors across 400 files. Immediate cgroup use fell from **358.2 to 219.9 GiB**.
Advised file lengths total 421.3 GB; that is not the amount of resident memory
released. Evidence: `evaluation_20_cache_release.json` in the current run.

The working trainer now applies the same recovery-file and mapping advice
after successful evaluation, before the next collection. Python compilation
and the evaluation filename convention were checked; the shared cache helpers
had already passed transport tests and the live cleanup above. This change
cannot alter the already imported driver, so the one-time live cleanup handles
this allocation's evaluation 20. Another evaluation is not expected before
the allocation ends.

<a id="h200-testing--iteration-20-trained-and-evaluated-2026-09-09-1940-pdt"></a>
### Iteration 20 trained and evaluated, 2026-09-09 19:40 PDT

Job 285546 completed fresh collection 20 in 3,828.3 seconds: 48 accepted
groups from 124 completed, 20 pending at cutoff, 144 submitted. The 1,685 turn
samples gave reward 0.4326409496, verified at W&B history row 363. All 12 PPO
updates completed and match W&B loss/KL/gradient-norm records. Checkpoint 19
has **270 durable Adam updates**, scheduler offset +1, intact metadata and
all eight shard extents, and its matching dataset cursor. Small CPU payload
samples from two shards were finite. No full checkpoint-19 reload was done.

The scheduled Online-Mind2Web evaluation completed all 300 tasks in about
52 minutes. Task-level success was **95/300 = 31.67%**, versus 70/300 = 23.33%
at iteration 10. There were 68 invalid trajectories (22.67%), versus 66 (22%)
previously. Success among valid trajectories was 95/232 = 40.95%. These results
are synchronized in W&B row 378; the task-level rate includes failed/invalid
attempts in its denominator and should not be confused with turn-weighted
reward metrics. Browser-evaluation variability remains a limitation.

Training host memory stayed near 346 GiB and dropped to 237 GiB after save;
there were no memory-limit, OOM or OOM-kill events through evaluation. Fresh
collection 21 started automatically. Reports in the current run directory:
`checkpoint_19_validation.json`, `iteration_20_wandb_audit.json`, and
`iteration_20_eval_wandb_audit.json`. The persistent pointer marks evaluation
20 complete and has no pending replay or evaluation.

<a id="h200-testing--replay-saved-successfully-in-job-285546-2026-09-09-1720-pdt"></a>
### Replay saved successfully in job 285546, 2026-09-09 17:20 PDT

Saved reward iteration 19 completed all 12 PPO updates. Checkpoint 18 now
contains 258 durable Adam updates in both parameter groups, with scheduler
offset +1. The 1,521 metadata entries, 4,957 stored extents, all eight shard
files totaling 62,137,280,293 bytes, and matching dataset cursor passed CPU
validation. Small payload samples from two of eight files were finite; this
is not a full checkpoint-18 reload. Checkpoint 17 was fully restored on all
four H200s at startup.

All 12 replay optimizer records match W&B history rows 350–361 numerically
for loss, KL and gradient norm. Their legacy `train/step` labels 216–227 are
not cumulative Adam counts. Replay does not create a new reward observation.
Fresh collection 20 has started, with its scheduled evaluation due after
training/save. Reports are `checkpoint_18_validation.json` and
`replay_19_wandb_audit.json` in the current run directory.

Recent replay GPU utilization averaged 70.9%. Host usage stayed around
258 GiB during PPO and fell to 234 GiB after saving; peak through this boundary
was approximately 289.5 GiB. Memory-limit, OOM and OOM-kill events were zero.
Both recovery-file and consumed-image cache-advice calls succeeded. This is
one successful replay/save cycle; prevention of cumulative memory growth
still needs observation across fresh collections. The persistent pointer
selects checkpoint 18 / 258 updates, with no pending replay. The user updated
active supervision and progress reporting to every 15 minutes, with closer
checks at transitions and on errors.

<a id="h200-testing--approved-continuation-submitted-2026-09-09-1644-pdt"></a>
### Approved continuation submitted, 2026-09-09 16:44 PDT

The user explicitly requested one further four-GPU, eight-hour continuation.
Job `285546` is running on `g003` with four H200s, 16 CPUs and 480 GiB RAM;
Slurm estimated $28.80 for 32 H200 GPU-hours. Its deadline is
2026-09-10 00:44:04 PDT. No additional submission is authorized by this request.
The preserved cache-release source passed recipe validation, and preflight
selected checkpoint 17 / 246 Adam updates plus saved rollout 18 (reward
iteration 19), with a cursor advance of 144 submitted groups. The batch driver
verifies full TP4 restoration before launching replay and online continuation
on W&B `qcq7i4ug`. Submission alone does not establish successful restoration
or validate the cache fix under GPU training.

Batch log: `/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/slurm-qcq7i4ug-285546.out`.

At 16:51 PDT, full model-and-optimizer restoration of checkpoint 17 passed on
all four H200s and the verification process exited zero. The supervisor recorded
`logs/resume-285546-4gpu-verification.json` and started online continuation
`runs/openwebrl-4b-reference-285546-20260909T235114`. The persistent pointer now
selects that running continuation, retains the 246 durable updates and pending
replay, and records the successful restoration. The verification source copy
took about five minutes on the shared filesystem; restore itself passed in
about one minute. The online process is preparing its own source snapshot.

<a id="h200-testing--four-gpu-continuation-stopped-at-host-memory-ceiling-2026-09-09-1415-pdt"></a>
### Four-GPU continuation stopped at host-memory ceiling, 2026-09-09 14:15 PDT

Approved job `284885` ran for 4:49:06 on four H200s and extended the durable
baseline from checkpoint 14 / 210 Adam updates to checkpoint 17 / **246 Adam
updates**. Checkpoint 17 has four complete shards totaling 62,137,280,293 bytes,
1,521 state-dictionary entries, consistent optimizer counters `[246, 246]`, the
known scheduler offset of +1, and a matching dataset cursor. Metadata, shard
extents and cursor checks pass. The job fully reloaded checkpoint 14 at startup;
checkpoint 17 has not yet received a full GPU reload.

W&B run `qcq7i4ug` contains synchronized fresh reward observations 16 through
19: 0.3759063, 0.4331288, 0.4427632 and 0.4029758. Collections 16, 17 and 18
were trained, adding 14, 12 and 10 updates respectively. Collection 19 completed
with 48 accepted groups, 117 completed groups, 27 pending groups and 144 total
submitted groups, but the first PPO step failed before any optimizer update.
Its reward is therefore a valid collection observation but does not describe an
updated policy.

The failure was `ncclUnhandledCudaError` / CUDA error 999 in a tensor-parallel
all-reduce. GPU 3 still had about 100.6 GiB free, while the online Slurm step's
MaxRSS reached 503,089,316 KiB and cgroup memory reached its exact 480 GiB
ceiling. `memory.events:max` rose from 0 in collection 16 to 6,714 by collection
19, with no cgroup OOM or OOM-kill event. The growth came from clean page-cache
charges for retained node-local image mappings and 47–56 GiB durable recovery
files. This evidence identifies host page-cache pressure as the cause; it was
not a model CUDA-memory OOM.

Recovery batch `rollout_recovery/18.pt` is a complete 49,951,567,097-byte torch
archive with 3,232 members and 1,613 turn samples. Its provenance records
zero-based rollout ID 18, checkpoint 17 and 144 submitted groups. Replaying it
will perform `floor(1613 / 256) × 2 = 12` updates and should reach 258 durable
Adam updates at checkpoint 18. Resume preflight selects checkpoint 17 and this
batch unambiguously.

The next preserved source is
`/gpfs/scrubbed/zixianma/openwebrl-runtime/reference-stage1-tp4-cache-release-20260909`.
It calls `fsync` and `POSIX_FADV_DONTNEED` after each recovery save, then advises
the consumed node-local multimodal mappings after training. Files and contents
remain intact; only clean cache pages become immediately reclaimable. All nine
transport tests pass, an actual Linux file-advice probe retained and reproduced
the file exactly, both changed modules compile, and preserved-source recipe hash
validation passes. The numerical recipe files are unchanged. This fix still
needs verification in a new GPU allocation; no new job was submitted.

Evidence is in the run directory as `checkpoint_17_validation.json`,
`rollout_recovery/18.provenance.json`, `training.log`, `progress.log` and
`health.jsonl`. The persistent pointer now selects checkpoint 17, the pending
replay batch, and the cache-release source.

<a id="h200-testing--second-fresh-cycle-verified-2026-09-08-2250-pdt"></a>
### Second fresh cycle verified, 2026-09-08 22:50 PDT

Collection 9 completed in 3730.8 seconds, with 48 accepted groups from 111
completed groups and 33 surplus groups pending at cutoff (144 submitted).
All pending tasks were cleared and the browser pool reached 1199 acquired /
1199 released. The batch contains 1860 turn samples; two PPO epochs produced
14 optimizer updates. The complete collection/training/save cycle took about
89.7 minutes. No cgroup OOM or OOM kill occurred.

Checkpoint 8 contains **130 durable Adam updates**. Metadata and shard-extent
checks pass; finite tensor samples were read from all four shards. The dataset
cursor is 1248 groups / 6240 attempts, advancing by exactly 144 groups. This is
CPU validation; a full GPU reload of checkpoint 8 has not been performed.
W&B reward observation 9 is 0.3903225806, and all fourteen optimizer records
match the local log numerically. The persistent pointer now selects checkpoint
8, with no pending replay. Iteration 10 is collecting in the existing allocation.

Evidence in the current run directory: `checkpoint_verification_8.json`,
`checkpoint_8_all_shards_cpu_samples.json`, `collection_9_completion_audit.json`,
`collection_9_complete_wandb_audit.json`, and `rollout_recovery/8.provenance.json`.
The four-GPU feasibility and timing estimate are documented in
[BASELINE_SCALING.md](RL_RUNTIME.md#baseline-scaling); no additional allocation was requested.

<a id="h200-testing--evaluation-image-retention-guarded-2026-09-08-2135-pdt"></a>
### Evaluation image retention guarded, 2026-09-08 21:35 PDT

The next full Online-Mind2Web evaluation retains every completed trajectory
until metrics and debug data are written. Its custom generator previously
returned anonymous CPU image tensors, bypassing the training collector's
file-mapping guard. The evaluation wrapper now applies the same opt-in,
lossless completed-group mapping after reward calculation, including aborted
trajectories. It logs `[EvalStorage] mapped_completed_turns=...` so the live
execution can be verified. Evaluation horizons, sampling, rewards and returned
sample identity are unchanged.

Three CPU evaluation tests pass, including actual bfloat16 mappings, release
of the original tensor, completed/aborted reward behavior, preserved metadata
and isolated evaluation arguments. The full 300-task evaluation has not yet
run with this change. The preserved source's evaluation module is loaded only
when its first evaluation task starts; the fix can therefore be installed
before that import without interrupting collection or training. The runtime
patch audit records exact source hashes and confirms the training recipe
files remain unchanged.

<a id="h200-testing--complete-fresh-cycle-verified-2026-09-08-2100-pdt"></a>
### Complete fresh cycle verified, 2026-09-08 21:00 PDT

The quick-resume continuation `openwebrl-4b-reference-283214-20260909T022832`
loaded checkpoint 6 on both GPUs, then completed fresh collection 8 with the
collection-time mapping and allocator trim. Collection took 3784.6 seconds:
116 groups completed, 48 accepted, 28 pending at cutoff, 144 submitted. Abort
cleanup reached zero pending tasks and all 610 acquired browser slots were
released. No cgroup OOM or native worker crash occurred.

The batch has 1709 turn samples and reward 0.4640140433. Its portable recovery
file `rollout_recovery/7.pt` is 52,869,642,689 bytes. Training completed twelve
optimizer updates in about 22 minutes; the checkpoint save took 33 seconds.
Checkpoint 7 has **116 durable Adam updates**, the unchanged scheduler offset
of +1, four intact shards, and finite CPU samples from every shard. The saved
dataset cursor is 1104 groups / 5520 attempts, advancing by the expected 144
groups. A full GPU reload of checkpoint 7 has not been performed; checkpoint 6
was fully reloaded at the start of this continuation.

W&B `qcq7i4ug` contains reward observation 8 and all twelve optimizer records,
with their numerical values matching the local training log. Reports in the run
directory include `checkpoint_verification_7.json`,
`checkpoint_7_all_shards_cpu_samples.json`, `collection_8_transition_audit.json`,
and `collection_8_complete_wandb_audit.json`. The persistent pointer records
checkpoint 7 / 116 updates with no pending replay. Collection 9 is running in
the same allocation, which ends at 02:16:49 PDT; no new compute was requested.

<a id="h200-testing--collection-time-memory-fix-2026-09-08-1924-pdt"></a>
### Collection-time memory fix, 2026-09-08 19:24 PDT

Fresh collection 8 revealed that `generate_rollout_async` retains every
completed group in `all_data`, including groups rejected by the dynamic filter.
The post-collection transport fix does not cover these live image tensors.
At 40 completed groups / 18 accepted groups, job usage had reached about
225 GiB, with substantial growth still expected before the 48-group target.

With the same opt-in file directory, completed groups now have their image
buffers replaced by read-only tensor views of lossless file mappings **before**
filtering and telemetry retention. File I/O runs in a thread so the browser loop
can continue. Tokens, rewards, sample identity, filtering and numerical image
values are preserved. Both accepted and rejected groups retain their complete
data; the kernel can reclaim image pages under pressure. Durable torch recovery
files contain the values themselves, not dependencies on node-local paths.

Six CPU transport tests pass, including nested turn groups, aliased sample
identity, original buffer release, bfloat16, and recovery-file portability after
the backing file is renamed. A bounded 64-MiB allocation-local probe checks
actual anonymous-memory release and exact retained values. The first probe
showed that glibc retained freed buffers, so the helper now releases its original
references and calls `malloc_trim(0)` when available. The repeated probe released
65,844 KiB while preserving every value; a Linux memory regression test was added
(seven transport tests pass). Its report is
`openwebrl-runtime/collection-mapping-probe-283214.json`. The fix requires a new
worker process; checkpoint 6 preserves all 104 completed optimizer updates.

<a id="h200-testing--successful-continuation-on-240-gib-2026-09-08-1902-pdt"></a>
### Successful continuation on 240 GiB, 2026-09-08 19:02 PDT

Run `openwebrl-4b-reference-283214-20260909T012629` on g021 loaded checkpoint 5,
replayed collection 7, completed all fourteen optimizer updates, saved checkpoint
6, and began fresh collection 8 at about 18:58 PDT. The two PPO epochs took
about 27 minutes. Both H200s performed training; no cgroup OOM or OOM kill
occurred. Host usage reached the 240 GiB limit while the kernel reclaimed file
cache, then fell to roughly 175 GiB at the collection transition. File-backed
image transport therefore passed this actual training-and-save cycle in the
smaller allocation; later collections remain under active observation.

Checkpoint 6 has 104 Adam updates (both optimizer groups agree), the known
scheduler offset of +1, 1,519 state entries and four complete shard extents
totaling 62,134,965,899 bytes. Its matching dataset cursor advanced from 816 to
960 groups, exactly the 144 submitted groups in the recovered batch. CPU
checkpoint validation and sampled payload checks passed. A full GPU reload of
checkpoint 6 has not been performed; training continues from its in-memory
state. Runtime reports are `checkpoint_verification_6.json` and
`replay_6_completion_audit.json` in the run directory.

The [quick resume command](RL_RUNTIME.md#resuming-baseline) is committed as `d1dd5b8`, with
seven passing CPU tests. Real preflight first selected checkpoint 5 plus replay
batch 6, then correctly selected checkpoint 6 with no replay after the save.
The stable run pointer now records 104 durable updates and no pending replay.
W&B remains `qcq7i4ug`; recovered reward observation 7 is 0.3987667009 and retains
its original reward-iteration coordinate. New collection 8 will supply the next
reward observation.

<a id="h200-testing--resume-preparation-for-allocation-283214-2026-09-08-1826-pdt"></a>
### Resume preparation for allocation 283214, 2026-09-08 18:26 PDT

The user explicitly assigned job 283214 on g021 for continuing W&B run
`qcq7i4ug`. It provides two H200s, eight CPUs, and 240 GiB RAM until
2026-09-09 02:16:49 PDT. The previous job had 16 CPUs and approximately
390 GiB RAM, so its in-memory image transfer cannot be assumed to fit.

Added opt-in `OPENWEBRL_MULTIMODAL_STORAGE_DIR` transport: image tensors are
written losslessly to a unique binary file per shard, and Ray sends path,
offset, shape, and dtype descriptors. Each training rank uses read-only memory
mappings; only the current microbatch moves to GPU. These files persist for the
run lifetime and must be accessible to all consuming ranks. The single-node
continuation uses `/tmp/openwebrl-283214-multimodal` (3.5 TiB available), while
recovery batches and checkpoints remain on scrubbed storage. Four CPU tests
passed, covering bfloat16, non-contiguous/empty/scalar tensors, independent
process readers, small serialized descriptors, and prior-batch lifetime.
Two actual Ray readers on g021 also reproduced the data exactly; see
`transport-ray-check-283214.json` in the runtime directory.

The isolated source is `reference-stage1-283214`. All five numerical-recipe
files match the preceding reference snapshot. It uses an 8-GiB Ray object
store and two OpenMP threads per process; the 48 accepted groups, five requested
attempts, batch size 256, two PPO epochs, learning rate, rewards, filtering, and
browser concurrency remain unchanged. Launch metadata reads resume counters
from the actual checkpoint instead of earlier descriptive constants.

Resume checkpoint 5 (90 Adam updates), replay saved batch 6 with 144 consumed
prompt groups, then continue fresh collection 8. The replay should add fourteen
updates and save checkpoint 6 with 104 Adam updates. This is still the seventh
collected reward observation. GPU training memory and checkpoint completion
must be verified in this smaller allocation before declaring the resume healthy.

<a id="h200-testing--final-handoff-2026-09-08-0334-pdt"></a>
### Final handoff, 2026-09-08 03:34 PDT

The baseline was intentionally paused after collection 7 finished, because its
14 optimizer updates could not fit before allocation 282346 ended at 03:36 PDT.
Only baseline step `282346.1` was stopped; no new allocation was submitted.
Checkpoint 5 contains **90 durable Adam updates** from six fully trained
collections. Collection 7 completed **zero optimizer updates** before the stop.

Collection 7 took 4064.5 seconds (67.7 minutes): 48 accepted groups, 121 completed
groups, 23 pending at cutoff, and 144 submitted. Cleanup completed with zero
pending tasks and no native abort. Its `rollout_recovery/6.pt` is 60440403683 bytes
and contains 1946 turns across 48 groups. A memory-mapped CPU metadata load
reproduced reward **0.3987667009249743**, matching W&B history row 145 at
`train/reward_iteration=7`. See `recovery_batch_6_metadata_audit.json` and
`rollout_recovery/6.provenance.json` in the current recovery run.

The remaining authorized GPUs were used for a full model-and-optimizer reload
of checkpoint 5. The verification run
`openwebrl-4b-resume-check-282346-20260908T102901` passed in about 60 seconds and
exited with code 0. It loaded iteration 5, selected next rollout ID 6, and ran
zero optimizer updates or browser collections. It used offline W&B, so it added
no training observations to the baseline. Both GPUs were confirmed released
with zero memory use afterward. See `checkpoint_5_full_resume_audit.json` and
the verification run's `resume_verification.json`.

The final W&B audit confirms seven distinct fresh reward observations. Its raw
API scan returned 166 optimizer rows: 53 duplicate early rows reduce to 113
unique history-row IDs. Of those, 23 belong to earlier failed/retried attempts;
the successful checkpoint lineage accounts for 90. Duplicate training payloads
agree, all optimizer values are finite, and each reward iteration has a single
consistent reward value. Do not infer progress from raw API row count or the
legacy overlapping `train/step` labels. See `final_wandb_audit.json`.

For the next **explicitly authorized allocation**, use the isolated reference
source and resume checkpoint 5 from the current recovery run. Before any fresh
collection, replay its saved batch with:

```bash
export OPENWEBRL_REPLAY_FIRST_BATCH=/gpfs/scrubbed/zixianma/openwebrl-runtime/runs/openwebrl-4b-reference-282346-20260908T061434/rollout_recovery/6.pt
export OPENWEBRL_REPLAY_ROLLOUT_ID=6
export OPENWEBRL_REPLAY_CONSUMED_GROUPS=144
```

Pass that run directory to `run_small_baseline.py --profile reference
--resume-from ... --wandb-run-id qcq7i4ug`, with the main repository's `.env`.
Verify the checkpoint marker is still 5 before launching. Replaying this batch
performs 14 updates, so checkpoint 6 should contain 104 Adam updates with the
existing scheduler offset +1. It is the existing seventh observation, not a new
eighth reward point. Fresh collection 8 follows after that checkpoint. Always
use an explicit `srun --jobid=...` step in the allocated job; plain SSH can attach
to a different allocation through Slurm PAM adoption.

<a id="h200-testing--checkpoint-5-validated-2026-09-08-0221-pdt"></a>
### Checkpoint 5 validated, 2026-09-08 02:21 PDT

Collection 6 completed all 14 optimizer updates and saved `iter_0000005` before
fresh collection 7 began around 02:19 PDT. The checkpoint's Adam counters are
`[90, 90]`, with scheduler count 23296/256 = 91: the existing offset remains +1.
All 1519 state entries, 3507 stored extents, four shard files totaling
62134965899 bytes, and the matching dataset cursor passed inspection. Small CPU
tensor samples (32768 bytes, covering two of four shards) were finite; this is
not a full tensor reload. See `checkpoint_verification_5.json` in the current
recovery run. The latest actual full GPU resume audit still refers to checkpoint 3.

A subsequent bounded CPU check loaded one serialized tensor extent from every
shard, reading 20978340 bytes total. All four samples had the expected dtype
and finite values. See `checkpoint_5_all_shards_cpu_samples.json`. This covers
all shard files with payload samples, but still does not constitute a full
model-and-optimizer reload.

W&B history rows 130–143 contain all fourteen collection-6 optimizer records,
with finite training values. The sixth fresh reward observation is in row 128.
See `wandb_collection_6_complete_audit.json`. Cgroup OOM and OOM-kill counters
remained zero through collection, cleanup, training, and checkpoint saving.
Memory dropped from the cap during training to about 148 GiB in collection 7.

At 02:21 PDT approximately 75 minutes remained in allocation 282346. Whether
collection 7 can also finish both PPO epochs depends on its browser duration;
any fully collected recovery batch must be preserved separately from durable
optimizer progress. No new allocation or budget extension is authorized.

<a id="h200-testing--collection-6-reward-and-cleanup-verified-2026-09-08-0152-pdt"></a>
### Collection 6 reward and cleanup verified, 2026-09-08 01:52 PDT

Collection 6 finished in 4106.0 seconds (68.4 minutes), with 48 accepted groups,
124 completed groups, and 20 pending groups at cutoff: 144 submitted in total.
Both inference workers acknowledged cancellation and cleanup ended with zero
pending tasks. A later process scan found no browser environment servers left
in allocation 282346 during training. There were no native aborts or OOM events.

The saved `rollout_recovery/5.pt` is 61946877909 bytes. Its memory-mapped CPU
metadata load verified 1999 turn samples, 48 groups, and mean reward
0.3426713356678339, exactly matching W&B history row 128 at
`train/reward_iteration=6`. This checks metadata and rewards, not every image
payload. See `recovery_batch_5_metadata_audit.json`,
`rollout_recovery/5.provenance.json`, and `wandb_collection_6_audit.json`.

Training began at 01:49 PDT. Seven full minibatches per epoch and two epochs
produce 14 updates; after they finish, checkpoint 5 should contain 90 Adam
updates and the existing scheduler offset +1. At this observation, the latest
validated checkpoint remains checkpoint 4 with 76 updates. If replay is needed,
explicitly load checkpoint 4 and advance its restored data cursor by the 144
submitted groups; replay does not create another fresh reward observation.

<a id="h200-testing--checkpoint-4-validated-2026-09-08-0042-pdt"></a>
### Checkpoint 4 validated, 2026-09-08 00:42 PDT

Collection 5 completed all 14 optimizer updates and saved `iter_0000004` in
the current recovery run before collection 6 began around 00:40 PDT. Its
Adam parameter-group counters are `[76, 76]`; scheduler count is 19712/256 = 77,
so the diagnosed offset remains +1 and the resume fix added no further offset.
The checkpoint has 1519 state entries, 3507 extents, four shard files totaling
62134965899 bytes, and a matching dataset cursor. All file extents and 32768
bytes of sampled CPU tensors passed checks. Samples cover two of four shard
files; this is not a full tensor reload. See `checkpoint_verification_4.json`.
The actual full GPU reload already verified in this continuation refers to
checkpoint 3, as recorded in `resume_load_audit.json`.

W&B history rows 113–126 contain all 14 optimizer records, with finite losses
and gradients, plus reward observation 5 in row 111. See
`wandb_collection_5_complete_audit.json`. The successfully cancelled collection
left no current-run browser servers alive during training. Cgroup OOM counters
remain zero, and host memory fell to about 145 GiB as collection 6 began.

The shared `project-krishna` fileset reached its 1 TiB quota and blocked the
initial documentation commit. The edits were preserved in the working tree
and backed up in the current run as `collection_5_documentation.patch`.
Scrubbed checkpoint storage was unaffected, and the checkpoint save completed
successfully. Project-quota availability should be checked before later commits.

<a id="h200-testing--recollection-5-and-cleanup-passed-2026-09-08-0016-pdt"></a>
### Recollection 5 and cleanup passed, 2026-09-08 00:16 PDT

The resumed run completed collection 5 in 3261.5 seconds: 48 accepted groups,
104 completed groups, and 40 pending groups at cutoff (144 submitted). Both
inference workers acknowledged cancellation, and cleanup finished with zero
pending tasks and no native abort. The saved `rollout_recovery/4.pt` is
63152215689 bytes. Its memory-mapped CPU metadata check found 2041 turns,
48 prompt groups, and reward 0.35178833904948553, exactly matching W&B's fifth
reward observation (history row 111). See `recovery_batch_4_metadata_audit.json`,
`rollout_recovery/4.provenance.json`, and `wandb_collection_5_audit.json` in the
current recovery run. Training is underway: seven minibatches per epoch and
two epochs give 14 new updates, so checkpoint 4 should contain 76 Adam updates
and the existing scheduler offset +1. It is not yet a saved checkpoint.

The first 12 distinct task starts match the failed collection when original
worker stdout and stderr are merged by timestamp. The worker also records
loading `global_dataset_state_dict_3.pt`; see `data_cursor_resume_audit.json`.
Do not compare task order using only Ray's deduplicated combined driver log.

The shorter batch also exercises the known legacy training-label limitation:
collection 5 labels its optimizer records 56–69, overlapping collection 4's
48–63. Use collection identity, history-row identity, and Adam counters when
auditing updates. The primary reward axis remains one-based collection number
and is unaffected. No historical optimizer records were removed.

One orphaned router and 12 orphaned browser process groups from the failed
attempt were identified by exact source snapshot, process identity, and job
cgroup, then stopped. All 12 old browser servers were verified exited. The
separate ARM allocation and current recovery workers were excluded. Inventories
and cleanup records are preserved under the failed run's `failure-collection-5/`.

<a id="h200-testing--recovery-running-2026-09-07-2318-pdt"></a>
### Recovery running, 2026-09-07 23:18 PDT

The same W&B run resumed from checkpoint 3 at 23:14 PDT, using an explicit
`srun --jobid=282346` step in the existing allocation. The new run directory is
`/gpfs/scrubbed/zixianma/openwebrl-runtime/runs/openwebrl-4b-reference-282346-20260908T061434`;
its `training.log`, `progress.log`, and `health.jsonl` are the current logs.
The new source is `reference-stage1-282346-stdlib-recovery` under the runtime.
Actual two-GPU model and optimizer loading completed successfully, followed by
fresh collection 5 at 23:16:42. See `resume_load_audit.json` in the new run.
The starting durable count remains 62 Adam updates; the existing scheduler
counter offset +1 was preserved, and the resume fix prevents adding another.

This continuation includes the tested stdlib event-loop fix, restored-scheduler
fix, consumed-batch-reference release, worker W&B shutdown ordering, and router
CLI precedence fix. The actual router imbalance threshold is now 2 as requested
by the launcher. All five recorded baseline recipe files are byte-identical to
the preceding live snapshot. The copied manifest's stale telemetry hash was
refreshed against those actual files, with the preceding recorded hashes retained.
W&B history still has four reward observations; the failed collection emitted no
fifth point. The allocation deadline is unchanged, 03:36:15 PDT, with launcher
shutdown three minutes earlier.

**Use an explicit existing job ID for node commands and recovery launches.**
After a separate user allocation started on g005, plain SSH attached to that
newer allocation through Slurm's adoption behavior. Its GPU workload was left
alone. Allocation 282346's two free GPUs were verified in its own `srun` step
before restarting; setting SLURM_JOB_ID in an arbitrary SSH shell would not
change that shell's actual resource cgroup.

<a id="h200-testing--collection-5-crash-and-recovery-preparation-2026-09-07-2311-pdt"></a>
### Collection 5 crash and recovery preparation, 2026-09-07 23:11 PDT

The first g005 continuation exited with code 1 after its rollout actor received
SIGABRT at 23:02:57 during cancellation of surplus browser tasks. The native
crash stack points to the uvloop background thread; preceding browser errors
reported a file descriptor already owned by a TCP transport. Cgroup OOM and
OOM-kill counters remained zero. Ray RPC failures were a consequence of the
actor abort. Collection 5 had reached 48 accepted groups in 2461.4 seconds,
with 105 completed and 39 pending groups, but its recovery batch and reward
metric had not been saved. It added no optimizer updates. Checkpoint 3 with
62 Adam updates remains valid; recollection is necessary.

The background stdlib loop now also installs a compatible stdlib event-loop
policy when SGLang has installed uvloop's policy. This fixes the Python 3.12
child-watcher mismatch that previously forced the browser runtime back onto
uvloop. The owned H200 environment now selects the stdlib loop. A CPU probe
reproduced the old NotImplementedError; a regression covering 16 subprocesses,
32 concurrent HTTP cancellations, and a subsequent subprocess passed. An actual
browser startup, screenshot, and cleanup test also passed on g005 through the
fixed background helper after uvloop policy installation. These tests validate
the replacement setup; they do not reproduce or prove the elimination of every
native transport race from the long-running failed worker.

Crash diagnostics are preserved under the first continuation's
`failure-collection-5/` directory. Recovery will use the existing allocation
and checkpoint 3, with no new paid allocation or budget extension.

<a id="h200-testing--previous-continuation-milestone-2026-09-07-2224-pdt"></a>
### Previous continuation milestone, 2026-09-07 22:24 PDT

W&B run `qcq7i4ug` is continuing on `g005`, inside existing allocation 282346
(two H200s; allocation ends 2026-09-08 03:36:15 PDT, launcher stops three
minutes earlier). The live source snapshot is
`/gpfs/scrubbed/zixianma/openwebrl-runtime/reference-stage1-282346`.
The run directory is
`/gpfs/scrubbed/zixianma/openwebrl-runtime/runs/openwebrl-4b-reference-282346-20260908T024410`.

Fresh collection 3 accepted 48 of 120 completed prompt groups in 3006.2 seconds,
after restoring checkpoint 1 with 30 durable optimizer updates. It produced
2271 turn samples and a saved recovery batch, `rollout_recovery/2.pt` (65.51 GiB).
W&B history independently confirms `train/reward_iteration=3` and
`train/reward=0.38881549977983265`. Two PPO epochs use eight global minibatches
each, for 16 new updates. All 16 finished, and checkpoint `iter_0000002` was
saved at about 21:08 PDT. It contains **46 actual Adam updates**, 1519 state
entries, 3507 stored extents, and 62134965899 shard bytes. Its matching dataset
cursor exists. All metadata-referenced file extents and a 2048-byte CPU tensor
sample passed checks; the sample covers only one of the two shards. This is
not a full tensor reload. See `checkpoint_verification_2.json` in the run.
Collection 4 began around 21:09 PDT.

Collection 4 finished in 2428.7 seconds with 48 accepted groups out of 90
completed groups (96 submitted), 2287 turn samples, and
`train/reward=0.3624836029733275`. It completed another 16 optimizer updates
and saved `iter_0000003` at about 22:22 PDT. This is now the latest checked
checkpoint: Adam steps `[62, 62]`, scheduler batches 63 (the same diagnosed
offset +1), matching dataset cursor, all shard extents, and a finite 2048-byte
sample from one shard. See `checkpoint_verification_3.json`. W&B history
contains all 16 optimizer records (legacy labels 48 through 63) and the fourth
reward observation. Collection 5 is running. OOM and OOM-kill counts remain zero.

Recovery batches `rollout_recovery/2.pt` and `3.pt` have accompanying
`.provenance.json` files recording the preceding policy checkpoint and exact
submitted-group counts, 144 and 96 respectively. Do not assume every replay
should advance the dataset by 144 groups.

The host-memory cap is 419430400000 bytes (390.625 GiB). Cache reclamation
occurred during training and saving, with no OOM or OOM-kill events. Memory
fell to about 252 GiB as collection 4 began. The live CUDA cache guard released
reserved memory from 109.63 to 48.07 GiB during training.

Checkpoint validation exposed a scheduler bookkeeping bug: Megatron restores
the scheduler, then OpenWebRL used to advance it by `loaded_iteration * GBS`
again. Loading iteration 1 added 256 to `num_steps`, so checkpoint 2 records
12032/256 = 47 scheduler batches but both Adam parameter groups record step
46. The learning rate (1e-6) and weight decay (0.1 for the decay group, 0 for
the no-decay group) are constant, so that extra scheduler count does not
change this run's optimization settings or represent an extra Adam update.
The one-batch offset remains in subsequent saves from this live process.

The repository now removes that redundant advance. Seven focused CPU tests
passed, including real scheduler restoration with linear decay and mocked
model setup, unchanged post-load learning rates, and rejection of unexplained
counter mismatches. The fix prevents new offsets; it does not rewrite existing
checkpoints or alter the already-running process. The inspector now uses Adam
group counters when present and separately reports scheduler counts. This
checkpoint requires an explicit acknowledgment of the diagnosed discrepancy:

```bash
python scripts/inspect_training_checkpoint.py "$RUN_DIRECTORY" \
  --expected-updates 46 --expected-scheduler-offset-updates 1 --sample-payloads
```

The repository driver now releases `rollout_data_ref` after all applicable
actor/critic training calls complete. Previously it retained the old batch's
Ray references while waiting for the next collection. A CPU weak-reference
probe of the actual loop with fake workers reproduced retention before the
change and passed after it for actor-only, actor-plus-critic, and critic-only
paths. This checks reference lifetime and ordering, not distributed memory
reclamation. This change is **not in the running g005 snapshot**; monitor its
next collection's memory before deciding whether a restart is needed.

<a id="h200-testing--runtime"></a>
### Runtime

The dedicated environment is `/gpfs/scrubbed/zixianma/openwebrl-runtime/venv`.
The checkpoint is `/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT`.
No other user's Python environment or container is required.

Activate from the repository root:

```bash
source scripts/h200_env.sh
```

The tested stack is Python 3.12.13, PyTorch 2.9.1+cu129, SGLang
0.5.6.post2, Transformers 4.57.1, Transformer Engine 2.10.0, CUDA compiler
12.9.86, and cuDNN 9.16.0.29. The complete package freeze lives at
`/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/environment.freeze.txt`.
cuDNN deliberately supersedes PyTorch's older dependency pin because SGLang's
correctness check requires cuDNN >=9.15. Subsequent dependency resolution may
downgrade it; rerun the final cuDNN install in `finish_transformer_engine.sh`.

Source commits:

- Megatron-LM: `3714d81d418c9f1bca4594fc35f9e8289f652862`
- Megatron-Bridge: `35b4ebfc486fb15dcc0273ceea804c3606be948a`
- SFT model: `15e777db2ddba2e0e82080ebccd3ad8d215b7f0a`

To rebuild on a compatible H200 node, run `bootstrap_h200_env.sh`,
`install_megatron_backend.sh`, then `build_transformer_engine.sh`, all under
`scripts/`. The last script also prepares Python headers without building
Python. Install Chromium with `python -m playwright install chromium` after
sourcing `h200_env.sh`. These scripts contain the fixes discovered during the
initial build; a complete clean rebuild of the final scripts has not been rerun.

<a id="h200-testing--completed-checks-2026-09-07-utc"></a>
### Completed checks, 2026-09-07 UTC

- All 24 offline tests passed in the dedicated environment.
- Both H200s passed CUDA backward and NCCL/DDP synchronization checks.
- The real checkpoint generated from an image and completed a policy-loss
  backward/AdamW update with finite gradients and changed parameters. HF model
  and optimizer files were saved. Peak PyTorch allocation was 41.77 GiB.
- SGLang served the real checkpoint and correctly read `DONE` from a Chromium
  screenshot.
- The repository's `LocalProcessWebEnv` successfully initialized, reset,
  returned a screenshot, and cleaned up its subprocess.
- Actual Megatron TP=2 training completed two GRPO updates on a synthetic
  text replay batch, saving distributed model and optimizer state after each.
  Reported total device memory use was approximately 48 GiB per H200.
- Distributed resume restored checkpoint iteration 1, performed iteration 2,
  and saved again. Extending the total rollout count required
  `--override-opt-param-scheduler`; model and optimizer state remained loaded.
- `train.py --help` now works after removing an accidental tuple-valued help
  string.

The synthetic rewards establish numerical and infrastructure behavior, not
web-task success. Short text timings and memory are not baseline throughput
or long-context memory estimates.

<a id="h200-testing--reproduce-tests"></a>
### Reproduce tests

```bash
python scripts/check_cpu_preflight.py
python scripts/local_browser_smoke.py
torchrun --standalone --nproc-per-node=2 scripts/gpu_nccl_smoke.py
bash scripts/run_model_smoke.sh
bash scripts/run_h200_train_smoke.sh
```

Megatron replay checkpoints from the successful run are under
`$OPENWEBRL_RUNTIME_ROOT/runs/megatron-smoke-280366`; the resumed output is
under `runs/megatron-resume-280366`. Each distributed checkpoint is roughly
58 GiB, including optimizer state. Use scrubbed storage, not the nearly full
project filesystem.

`run_h200_pipeline_smoke.sh` passed two complete text iterations with real
SGLang sampling, two PPO epochs each, nonzero gradients, distributed saves,
memory offloading, and weight updates to both rollout engines.

`run_h200_browser_fixture_smoke.sh` subsequently generated four real browser
turns with screenshots (~4282 total tokens per sample) and completed two PPO
epochs. Its first synthetic reward implementation used turn indices, producing
identical rewards and zero advantages. The fixture now uses original trajectory
IDs; a nonzero-reward browser rerun is still required. The earlier HF image
test and text Megatron tests did establish nonzero policy updates separately.

The initial browser workaround used `SLIME_ASYNC_USE_STDLIB_LOOP=0`: SGLang installs a
uvloop policy, and Python 3.12's stdlib selector loop cannot spawn subprocesses
under that policy. A CPU subprocess reproducer passed with the active-policy
option. This workaround is superseded by the compatible stdlib loop and policy
fix described above after the later native uvloop cancellation crash.

On 2026-09-07, allocation 281697 reran the corrected browser fixture on two
H200s. Both PPO epochs completed with nonzero advantages (mean absolute value
0.7071) and finite nonzero gradient norm (5.31 on epoch 2). Logs are in
`$OPENWEBRL_RUNTIME_ROOT/logs/browser-nonzero-281697.log`. This closes the
nonzero browser-gradient gap above; synthetic rewards still do not validate
the real judge or learning on live websites.

`scripts/h200_smoke.sbatch` is prepared with two H200s, 16 CPUs, 400 GiB memory,
and a one-hour limit. Shell syntax was checked; no batch job was submitted.
Submission always requires explicit user approval of resources and budget.
The real-web judge endpoint, representative longer trajectories, and execution
under a batch allocation remain unvalidated.

`python scripts/run_small_baseline.py --dry-run` describes the prepared small
real-web baseline. Running without `--dry-run` requires an already-authorized
Slurm allocation, judge settings, and W&B authentication. It reads relevant
settings privately from `.env`, logs online to W&B, writes a credential-free
manifest and training log under runtime storage, and imposes a deadline inside
the existing allocation. It never submits a new job. Defaults: 30 rollouts,
4 groups x 5 trajectories, 15 browser steps, 16K context, global batch 16,
2 PPO epochs, checkpoint every 5 iterations, and 8 held-out evaluation tasks.

<a id="h200-testing--real-web-baseline-launch"></a>
### Real-web baseline launch

`scripts/run_h200_browser.sh` defaults to a small two-GPU run: TP=2,
two rollout iterations, two prompt groups, five trajectories per group,
three browser steps, 8192 context tokens, and two PPO epochs. It uses local
browser subprocesses and requires `JUDGE_API_BASE` for an OpenAI-compatible
judge endpoint (`JUDGE_API_MODE=served`). Other existing judge modes can be
selected explicitly. No judge credentials were available during these tests.

Review the command without starting GPUs:

```bash
DRY_RUN=1 bash scripts/run_h200_browser.sh
```

For a larger baseline, explicitly set GPU count, batch, horizon, context, and
evaluation settings. These small defaults do not reproduce the paper's scale.
The launcher accepts additional training CLI arguments at the end, including
evaluation dataset/options. It avoids global process-kill commands and shell
environment dumps. It saves a package freeze and launcher copy with the run.

The curriculum can select this launcher with
`--launcher scripts/run_h200_browser.sh`. Its second stage now explicitly
overrides scheduler duration when extending 90 rollouts to 140. This is
appropriate for the recipe's constant learning rate; changing the LR schedule
requires reviewing the intended continuation behavior.

<a id="h200-testing--2026-09-07-reference-baseline-checkpoint-recovery"></a>
### 2026-09-07: reference baseline checkpoint recovery

Allocation 281697 has two H200s and ends at 18:54:27 PDT. The launchers
initially stopped five minutes earlier. The final guarded recovery uses an
independent deadline supervisor through 18:53:27 PDT, within that same paid
allocation, to leave time to save the recomputed interval. Its exact process
targets and deadline are recorded in `deadline_supervision.json` in the final
run directory. No new allocation or spending was authorized.

The reference run `openwebrl-4b-reference-281697-20260907T225751` completed two
online collections and 30 optimizer updates (14 then 16). Its first checkpoint,
`iter_0000000`, is valid. At 17:39 PDT, staging the second checkpoint exhausted
host memory: the Slurm cgroup OOM-kill counter increased from 2 to 3. The run
exited with a CUDA invalid-argument error in Megatron's bulk D2H checkpoint
preloader and Ray SIGBUS errors. `iter_0000001` in that old run directory is
partial and must not be used for resume. Both collected batches are preserved
under that run's `rollout_recovery/` directory.

The replacement `StreamingTorchDistSaveStrategy` retains Megatron's state-dict
mapping and save planner, but uses PyTorch's synchronous `FileSystemWriter`
with one writer thread and 16 MiB copy-ahead. Individual tensors can exceed the
copy-ahead setting; this bounds prefetching rather than imposing a strict
16 MiB peak-memory limit. It avoids eagerly staging the entire model and
optimizer on the CPU. The persisted format remains `torch_dist`.

The default H200 launcher enables it with `OPENWEBRL_STREAMING_CHECKPOINT=1`.
It requires synchronous `torch_dist` saving; explicitly disable it when choosing
another format or asynchronous saving. A two-GPU test successfully saved FP32
model-like tensors, BF16 optimizer-like tensors, per-rank state, and a step
counter, then loaded them through the standard Megatron loader with exact value
equality. The focused test is `scripts/gpu_checkpoint_smoke.py`; set
`OWRL_CHECKPOINT_PROBE_DIR` to a new directory before launching it with two
already-allocated GPUs. Full baseline checkpoint validation is tracked below.

Recovery run:

```text
/gpfs/scrubbed/zixianma/openwebrl-runtime/runs/openwebrl-4b-reference-281697-20260908T005427
```

It restored checkpoint 0 and its dataset cursor, replays the previously saved
batch 1, advances the consumed prompt cursor by the 144 groups submitted for
that collection, and recomputes the lost checkpoint interval. The supervisor
requests a clean stop after checkpoint 1 so the remaining time can verify resume
rather than start an online collection that cannot finish. The training recipe,
48 prompt groups, five requested attempts, global batch 256, two PPO epochs,
and nominal 90-iteration schedule are unchanged. Replay does not create a new
independent reward observation.

The first recovery failed at 18:11 PDT after seven replayed updates. GPU
reserved memory reached about 131 GiB before the preload allocator exited from
`cuMemCreate` with out-of-memory; host OOM counters did not increase. No new
checkpoint was saved. The first replay loss matched the original attempt at
logged precision; subsequent values were not bitwise identical.

A second recovery started at 18:18 PDT:
`openwebrl-4b-reference-281697-20260908T011805`. The H200 launcher now sets
`OPENWEBRL_CUDA_CACHE_LIMIT_GIB=100`. Before each microbatch, when reserved CUDA
memory exceeds this threshold and at least 8 GiB is unused, the trainer
synchronizes and releases unused cached blocks. Live tensors, batches, epochs,
and optimizer settings are unchanged. This addresses the allocator's direct
exit, which bypasses PyTorch's normal OOM cache-release retry. A test using the
same preload freed a 12 GiB cached allocation and verified that live tensor
values were unchanged. Runtime logs record allocated/reserved memory at each
release. At 18:23 PDT, the guard activated during real training and reduced reserved
memory from 101.74 to 47.99 GiB with 41.19 GiB live. Full replay and full-size
streaming checkpoint validation are pending; do not infer either from the small
allocator/checkpoint tests alone.

At 18:37 PDT, the guarded retry completed the first eight updates and entered
PPO epoch two, passing the previous replay failure point. Final results remain
pending. The read-only `scripts/inspect_training_checkpoint.py` checker verifies
the saved iteration, scheduler counter, matching dataset cursor presence, and
all metadata-referenced shard byte extents. Its optional `--sample-payloads`
reads small tensor samples on CPU and reports exactly which shard files were
covered; it is not a full model reload. It passed on the existing checkpoint
with 14 updates, 1,519 state entries, 3,507 extents, and 62,134,965,899 shard bytes.

Measured online collection durations were 46.4 and 40.9 minutes; training took
25.5 and 30.0 minutes. A healthy iteration is therefore about 72 minutes plus
save/transition overhead. From two completed observations, eight more reward
points require roughly 9–10 wall-clock hours on two GPUs, before recovery or
other interruptions. Completing the tenth optimizer cycle takes about another
half hour after collecting its reward. The present allocation cannot reach ten
observations.

The canonical primary reward series is `train/reward` versus
`train/reward_iteration`. Duplicate `paper/*` emission is retired; original
history is retained. See `METRICS.md` and `PAPER_REWARD_COMPARISON.md` for axes,
units, and paper-comparison limitations.

<a id="h200-testing--final-recovery-result"></a>
#### Final recovery result

At 18:51:32 PDT all 16 replay updates completed. The streaming writer saved
checkpoint 1 in 20.6 seconds, completing at 18:51:53. No additional host OOM or
OOM-kill events occurred. The memory-limit encounter counter increased during
save, so host-memory headroom remains a concern for longer runs.

The durable checkpoint is:

```text
/gpfs/scrubbed/zixianma/openwebrl-runtime/runs/openwebrl-4b-reference-281697-20260908T011805/iter_0000001
```

Its scheduler counter is 7,680 = 30 optimizer updates x 256 turns. It contains
1,519 state entries, 3,507 stored extents, and 62,134,965,899 shard bytes across
two files, with the matching `global_dataset_state_dict_1.pt` cursor. Structural
and sampled CPU payload checks passed; see `checkpoint_verification.json`.

The adjusted GNU timeout wrapper returned 124 after the training driver had
completed: resuming that suspended wrapper also delivered its pending original
alarm. This supervisory exit code must not be interpreted as a missing save.
The completion marker, save log, scheduler counter, and file checks establish
the durable checkpoint independently. Future runs should choose the intended
shutdown margin at launch rather than adjust an active timeout process.

W&B history contains every final replay optimizer record, steps 16 through 31.
A secondary W&B teardown reported BrokenPipeError; no missing optimizer records
were found. Reward/checkpoint summary fields were refreshed from verified data.
The run still contains only two independent online reward observations.

The full GPU reload check used the same allocation, two GPUs, offline logging,
and zero requested optimizer updates/browser collections. Its directory is
`openwebrl-4b-resume-check-281697-20260908T015248`.

Full model/optimizer reload result: PASSED; see `resume_verification.json` in that directory.

Repository launchers now accept explicit `--resume-from`, `--wandb-run-id`, and
`--verify-resume-only`. Verification requires an existing allocation and exits
after model/optimizer and rollout-cursor restoration. Recovery changes are
backed up under `openwebrl-runtime/checkpoint-streaming-fix/20260907/`; committing
was initially blocked by the project filesystem quota. Git writes later recovered
and the verified fixes were committed separately (see below).

<a id="h200-testing--2026-09-07-continuation-on-g005"></a>
### 2026-09-07: continuation on g005

The user explicitly requested continuing W&B run `qcq7i4ug` on `zixianma@g005`.
Existing allocation 282346 supplies two H200 GPUs, 16 CPUs and 400,000 MiB host
memory, ending 2026-09-08 at 03:36:15 PDT. No new allocation was submitted.

The continuation launched at 19:44:10 PDT from verified checkpoint 1 (30
optimizer updates), using the isolated source `reference-stage1-282346`.
Its run directory is `openwebrl-4b-reference-282346-20260908T024410` under
`openwebrl-runtime/runs/`. The checkpoint loaded successfully and collection 3
started by 19:46:55. Both GPUs were active and host OOM counters were zero.

The source retains the tested reference recipe: 48 accepted prompt groups, five
requested attempts, global batch 256, two PPO epochs, TP2, LR 1e-6, stage-one
15-step browser limit, and nominal 90 collection iterations. Checkpoints are
saved after every completed iteration. It uses the streaming checkpoint writer,
100 GiB CUDA-cache pressure guard, and 24 GiB Ray object store. All temporary
replay, stop-after-recovery, and resume-verification controls were cleared.

The supervisor runs only inside the existing allocation, with a 180-second
shutdown margin selected at launch. The requested router balance threshold of
2 is overwritten to 10 by the installed SGLang argument merge; startup logs
confirm the actual value is 10. Both engines were receiving requests, so the
active collection was preserved. Any correction must be validated and included
in a subsequent source snapshot rather than silently changing this run's code.

Live files: `training.log` (full output), `progress.log` (compact milestones),
`health.jsonl` (30-second GPU/cgroup samples), and `launch_manifest.json` (recipe
and budget). `openwebrl-runtime/current_baseline.json` points to this run and
retains the last validated checkpoint until a newer save is verified.

<a id="h200-testing--recovery-changes-preserved-in-git"></a>
#### Recovery changes preserved in Git

Git writes succeeded again on the login host at about 19:56 PDT. Commits were
kept separate and each changes fewer than ten files:

- `67a65db`: preserve explicit router thresholds across the two argument parsers; two CPU regression tests passed. The active g005 snapshot retains the observed threshold 10.
- `314e6b2`: stream synchronous Megatron checkpoint writes; small two-GPU round trip and full 4B save/reload passed.
- `c4ca104`: release unused CUDA cache under pressure; allocator probe and complete 16-update replay passed.
- `5bb62eb`: explicit baseline resume, initialization-only verification, and read-only checkpoint inspection.
- `088dc45`: canonical `train/reward` per collected batch; three metric tests and W&B history audit passed.

The g005 continuation reached 15/48 accepted groups in collection 3 by about
20:01 PDT, with no OOM events or runtime errors. This is a progress snapshot,
not an additional saved checkpoint.

<a id="h200-testing--explicit-tracking-shutdown"></a>
#### Explicit tracking shutdown

The previous run and resume check emitted nonfatal W&B BrokenPipeError messages
during worker teardown; the online audit nevertheless found all 16 final
optimizer records. The repository now flushes training-worker tracking before
closing the primary tracker and calls the SDK's explicit teardown operation.
The running g005 source snapshot is unchanged.

A CPU-only, offline multiprocess test started a primary writer and two child
writers, closed them in that order of dependency, and read back every expected
history value (0, 1, 2) from three local W&B files. Both child exit codes were
zero and no teardown traceback occurred. The reusable check is
`scripts/check_tracking_shutdown.py --output <new-directory>`; test artifacts
are preserved and no cloud run or GPU is required. This checks SDK flushing;
the complete training shutdown path still needs validation on a future run.

<!-- document:H200_TESTING.md:end -->

---
