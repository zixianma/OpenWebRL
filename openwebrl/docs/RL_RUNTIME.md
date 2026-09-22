# RL runtime: resume, scaling, and execution history

Operational procedures for resuming the reference RL baseline, GPU scaling, rollout archives, and dated runtime validation. Use the resume instructions first; older execution entries explain recovery decisions and do not replace the persistent run pointer or authorize new allocations.

## Contents

- [Stage-2 recipe and prepared launch](#baseline-stage2-20260914)
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

<a id="baseline-stage2-20260914"></a>
### Stage 2: recipe audit and prepared continuation, September 14

**Prepared, not submitted. Exact compute-budget approval remains required.**
Continue W&B `openwebrl/qcq7i4ug` from 90 completed iterations to 140,
using `reference-stage2-browsers32-20260914`. The stage-1 pointer is unchanged
until successful GPU restoration inside the approved batch. All paths below
prefixed `R/` refer to `/gpfs/scrubbed/zixianma/openwebrl-runtime/`.

The paper specifies the two-stage schedule in [§5](https://arxiv.org/html/2606.02031v1#S5),
and shared hyperparameters in [Table 7](https://arxiv.org/html/2606.02031v1#A1.T7).
It does not explicitly specify an optimizer reset or data-cursor reset at the
transition. Our implementation continues both; this is a documented continuation
choice, not an additional paper claim.

| Setting | Prepared stage 2 |
| --- | --- |
| Initial checkpoint | `R/runs/openwebrl-4b-reference-294421-20260913T211532/iter_0000089` |
| Completed iterations / Adam updates at start | 90 / 1,016 |
| Additional collections / cumulative target | 50 / 140 |
| Maximum browser turns | 30, previously 15 |
| Prompt pool | Released 2,102-task dataset; same shuffled cursor; empty added blacklist |
| Effective prompt groups / attempted trajectories per group | 48 / 5 |
| Dynamic group filtering | Enabled: nonempty rewards and nonzero reward variance |
| Adaptive history-weighted prompt selection | Disabled |
| Objective / PPO epochs | Preserved MM-GRPO / 2 |
| Global batch / microbatch | 256 training-turn samples / 1 |
| PPO clipping lower / upper | 0.2 / 0.28 |
| KL / entropy coefficients | 0 / 0 |
| Optimizer / LR / schedule | Adam / `1e-6` / constant |
| Weight decay / Adam betas | 0.1 / (0.9, 0.98) |
| Generation temperature / top-p / top-k | 0.8 / 1.0 / -1 |
| Response / context token limits | 1,024 / 32,768 |
| Context screenshots / judge screenshots | 1 / 3 |
| Training judge / prompt | GPT-4.1 / preserved action-history prompt |
| Inference-step / whole-task timeout | 30 s / 600 s |
| Browser backend / concurrency | Local process / 32 pool slots and task gate |
| GPU topology | Four H200s; TP4 actor; four TP1 rollout engines |
| Checkpoint interval | Every completed iteration |
| Scheduled monitoring | 300-task Online-Mind2Web at 100, 110, 120, 130, 140 |
| Monitoring protocol | Local browser, GPT-4.1, T=0, 30 steps, 4,096 response tokens |
| W&B / reward axis | `openwebrl/qcq7i4ug` / existing `train/reward_iteration`, continuing at 91 |
| Completed rollout archive | All completed groups, including dynamic-filter rejections |

**Scheduler transition.** The frozen backend derives its scheduler horizon from
the cumulative rollout target. Its checkpoint loader normally asserts that this
matches the saved horizon. Moving from 90 to 140 therefore requires
`--override-opt-param-scheduler`. The prepared launcher sets this explicitly after
clearing inherited overrides. CPU tests reproduce the original assertion and
verify that the fix restores the saved scheduler counter and keeps LR `1e-6`
and weight decay `0.1` over 1,000 additional simulated optimizer steps. Adam
moments and group step counters still load from the checkpoint. The historical
one-update scheduler/Adam offset remains preserved and checked. No new warmup
or SFT restart is introduced. The explicit `--max-steps 30` takes precedence
over the frozen YAML's default `max_steps: 16`.

**Dynamic sampling evidence from our actual stage-1 run.** The startup argument
dump has the dynamic-filter path enabled and `enable_adaptive_query_sampling=False`.
Iteration 90 completed 99 groups, rejected 26 groups with all valid rewards equal
to one, 24 with all valid rewards zero, and one with no valid reward. It retained
48 groups with 210 valid trajectories; five are attempted per group, but invalid
members can be removed. The 99 completed groups / 495 trajectories were archived
with zero archive errors. This is reward-variance filtering followed by continued
collection, not adaptive selection of the next prompt from historical scores.
Source: [iteration-90 training log](/gpfs/scrubbed/zixianma/openwebrl-runtime/runs/openwebrl-4b-reference-294421-20260913T211532/training.log).

**Remaining comparison limits.** Our inherited runtime uses H200s, 32 local
browsers, GPU Adam, full activation recomputation, SDPA vision attention and
disabled SGLang CUDA graphs. The paper describes B200 training and 80–100
Kubernetes sandboxes; its launcher also differs in optimizer offloading. These
affect throughput and numerical reproducibility. Live sites and the unpinned
GPT-4.1 endpoint can change outcomes. Scheduled local GPT-4.1 monitoring is
separate from the paper's stealth-browser official-score protocol. No separate
stealth allocation is included in this request. Keep the 600-second task timeout
for recipe fidelity, and inspect timeout rates with the longer horizon. A reward
change at iteration 91 also reflects the larger action budget, so compare fixed
30-step evaluations alongside the training curve.

| Launch artifact / check | Location / result |
| --- | --- |
| Preparation command | `python3 scripts/prepare_baseline_stage2.py --source R/reference-stage1-browsers32-20260911 --output R/reference-stage2-browsers32-20260914` |
| First-allocation batch controller | `scripts/resume_baseline_stage2_4gpu.sbatch` |
| Proposed budget | 4 H200 × 8 h = maximum 32 GPU-hours; 32 CPUs; 480 GiB |
| Billing | GPU budget above; training-judge and scheduled-monitoring API use additional |
| Plan | `R/baseline_stage2_plan.json` |
| Prepared state / preserved stage 1 | `R/baseline_stage2_prepared_state.json` / `R/baseline_stage1_completed90_before_stage2.json` |
| Source comparison | 199 regular files compared; only launcher and reference manifest changed |
| Source audit | `R/baseline_stage2_source_audit.json` |
| GPU-free launch manifest | `R/baseline_stage2_cpu_launch_manifest.json` |
| CPU regression checks | `tests/test_baseline_stage2.py`; existing `tests/test_resume_baseline.py` |
| Stage-2 test log | `R/logs/stage2-cpu-tests-20260914.log` |
| Pending-evaluation recovery | `R/reference-stage2-browsers32-20260914-pending-eval`; source-selection check passed |
| Remaining GPU gate | Full TP2-to-TP4 model + optimizer restoration before any training |

The controller first verifies restoration with offline W&B, then checks the
verification receipt and unchanged baseline pointer, activates the prepared
state, and awaits training in the same allocation. A failed restoration prevents
activation and training. Subsequent resumes use the ordinary resume script and
the activated current pointer, retaining stage 2's 30-step / 140-iteration
defaults. Do not resubmit the first-transition template after activation.
Every subsequent allocation requires a separate exact budget approval.
Monitor startup closely, then every 15 minutes; verify reward logging and durable
checkpoints. Estimate the full 50-iteration runtime from several completed
stage-2 collections rather than assuming this eight-hour allocation finishes it.

<a id="cluster-queue-audit-294983-20260913"></a>
### Cluster and account audit, September 13 evening

Job **294983** waited **4m43s** (19:54:29–19:59:12 PDT), then started on
**g013 via backfill**. Its requested resources were two H200s, 16 CPUs,
480 GiB and three hours. The earlier estimated start of 20:22:44 was a
scheduler forecast, not a guarantee or reservation.

| Visible cluster snapshot near startup | Count |
| --- | ---: |
| Full-H200 partition nodes | 22 |
| Healthy full-H200 nodes / GPUs | 21 / 168 |
| Allocated healthy full H200s | 158 |
| Unallocated healthy full H200s | 10 |
| Healthy nodes fitting 2 GPUs + 16 CPUs + 480 GiB at snapshot | 0 |
| Down node g018, GPU error / administrator testing | 1 / 8 GPUs |
| Separate MIG nodes / 18-GiB slices | 2 / 112 |
| Allocated MIG slices | 6 |

The apparently spare full GPUs were fragmented across nodes. At the snapshot,
g002 had two free GPUs but only 12 free CPUs; the remaining spare GPUs were
singletons with four free CPUs each. Thus `MIXED` did not mean this request
could fit. Scheduler reason was `Priority`; several higher-priority eligible
jobs also competed for resources. Preemption is disabled. Snapshots of nodes
and queue are separate reads, so resource totals can change between them.

| Account zixianma | Reported value |
| --- | --- |
| Allowed QOS | normal, interactive, debug |
| Normal QOS per-user maximum | 48 H200s / 384 CPUs |
| Fair-share factor / job priority at submission | 0.356147 / 1508 |
| Account enforced total budget / reported used | $14,000.00 / $514.89 |
| Reported budget remaining | $13,485.11 |
| Current-cycle ended-job usage | 572.10 GPU-hours / 88 jobs |
| Active credits | $0.00 |

No account-limit or budget-hold reason was reported. Billing is recorded by job
end date; the displayed usage excludes unfinished jobs and external judge/browser
API charges. The other user job, **294976**, remained pending for its larger
four-GPU/32-CPU request; it is a separate ARM job and was not modified.

For future stealth evaluations, validate an **8–12 CPU / two-GPU** launcher:
remote Browser Use sessions do not need one local CPU each, and a smaller CPU
request could fit fragmented nodes such as g002. Every `srun` and the batch
request must agree; editing only Slurm's requested CPUs would break the current
16-CPU step. A shorter wall-time cap can improve backfill opportunities but
risks incomplete evaluation; the prior matching run needed 1h37m49s. Interactive
QOS has only a modest priority advantage here, and debug is capped at one hour;
neither makes unavailable resources appear. Urgent QOS is not authorized for
this account. MIG slices are not a drop-in replacement for the validated TP2
full-H200 evaluation profile. No queue or resource changes were made.

Evidence: runtime `cluster_queue_audit_294983.json`, from `scontrol`, `squeue`,
`sprio`, `sshare`, `sacctmgr`, and `hyakusage`.

<a id="resuming-baseline--finish90-tp2-20260913"></a>
### Completed minimal-GPU finish to iteration 90, September 13

**Job 294421 completed on g020 at 19:22:24 PDT**, exit 0, after 5h10m17s
(**10.343 H200-hours**). It saved after-90 checkpoint `iter_0000089` with
**1,016 Adam updates**, then completed all 300 scheduled evaluation tasks:
**101 successes / 222 valid / 78 invalid**, **33.67% overall / 45.50% valid-only**.
W&B is finished and all 43 final evaluation metrics match the local log.
The persistent pointer now records after-90 and no pending evaluation.
Evidence: [completed evaluation](RL_EVALUATION.md#scheduled-eval90-results-20260913).

The approved budget was
2 H200s × six hours, 16 CPUs / 480 GiB, at most 12 GPU-hours; Slurm estimates
$10.80 GPU charges, plus judge API usage. The batch owns and awaits the TP2
restore check, remaining training and scheduled evaluation.
Submission receipt: runtime `logs/submission-qcq7i4ug-finish90-294421.json`.
Controller log: `logs/slurm-qcq7i4ug-294421.out`.

The starting checkpoint was `293510-20260913T104752/iter_0000086`: **87 completed iterations,
984 Adam updates**. The interrupted collection 88 has no complete recovery
batch. The persistent pointer has been corrected from its stale after-78 entry.
Metadata, all shard byte extents, optimizer/scheduler counters and dataset-cursor
presence pass. Small CPU tensor samples are finite but cover only two of eight
shards; this is not a full reload. The allocation first verified the
TP4 checkpoint's model and optimizer restoration on TP2 successfully.

The prepared `scripts/resume_baseline_2gpu_finish90.sbatch` requests **2 H200s,
16 CPUs, 480 GiB RAM, six hours (12 GPU-hours maximum)**. Two GPUs are the smallest
validated training topology; this does not prove one GPU is physically
impossible. Extra host RAM gives offload and browser headroom. Expected total
runtime is roughly **4–5 hours**, using historical TP2 collection/training timing
and allowing for restore, saving, and final evaluation; live websites can vary.
The controller exits when all work completes rather than occupying the full cap.

The CPU dry run keeps the frozen reference source, W&B run `qcq7i4ug`,
48 prompt groups × five attempts, batch 256, two PPO epochs and the iteration-90
stop. TP2 uses 16 concurrent browser tasks. The scheduled after-90 evaluation
covers all 300 Online-Mind2Web tasks with local browsers, temperature 0,
GPT-4.1 judging and a 30-step limit, matching previous scheduled baseline evals.
It stays in the main training series. No separate stealth/o4-mini eval is in
this budget. Do not set `OPENWEBRL_STOP_AFTER_SAVED_ROLLOUT` because it would
stop before the scheduled evaluation.

Both GPU profiles now request typed `gpu:h200:N` Slurm steps; this prevents
TP2 from mixing untyped steps with a typed H200 allocation. Sixteen resume
CPU tests and the new batch template's shell syntax check pass. No GPU work
was used for this preparation.

Prepared plan: runtime `finish90_tp2_plan_20260913.json`; inner-launcher dry run:
`finish90_tp2_dry_run_20260913.json`; checkpoint audit:
`runs/openwebrl-4b-reference-293510-20260913T104752/resume_after87_checkpoint_audit.json`.

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

<a id="resuming-baseline--final90-293510"></a>
### Queued continuation toward iteration 90: job 293510, September 13

The user explicitly requested another **4 H200 × eight hours** after job
290926. Submitted job **293510** at 01:15 PDT with dependency
`afterany:290926`: **32 CPUs, 480 GiB, 32 GPU-hours maximum**, Slurm estimated
**$28.80** plus judge calls. The queue confirms `PENDING (Dependency)`.
The current allocation continues unchanged. The dependent job uses the tested
`resume_baseline_4gpu_32cpu.sbatch`, preserved source
`reference-stage1-browsers32-20260911`, TP4, 32 browsers, and W&B `qcq7i4ug`.
Its controller awaits full GPU restoration verification before continuation,
selects the latest completed checkpoint in the recorded lineage at startup,
and replays a complete next batch when present. The native schedule stops at
90 iterations; the allocation deadline still applies first. No further
allocation or budget extension is authorized. With 75 iterations saved and
about 2.5 hours left in 290926 at submission, recent timings suggest a finish
around 86–88 after this continuation; reaching 90 is not guaranteed.

Scheduled evaluations at 80 and 90 remain in the training recipe. Additional
reward-ranked evaluations need a new bounded approval because the earlier
four-job evaluation budget is exhausted. The user selected only after-69
from the nearby top-five candidates; see
[the prepared evaluation plan](RL_EVALUATION.md#reward-rank-final90-20260913).
The existing completed-group archive remains enabled for future SFT use.

Receipt: `/gpfs/scrubbed/zixianma/openwebrl-runtime/logs/submission-qcq7i4ug-final-4gpu8h-293510.json`.
The persistent pointer records this job under `queued_continuation`.
Shell syntax, source hashes, 16 resume tests and 11 reward-queue tests pass.

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

**Approved and submitted browser comparison: job 291224.** The user revised
the request to one hour at the largest prepared browser count, then one lower
level, and explicitly required a brand-new run. Template
`scripts/benchmark_browsers_8gpu_1hour.sbatch` requests **8 H200 / 64 CPUs /
960 GiB / one hour**, **8 GPU-hours**, Slurm estimated **$7.20** plus judge calls.
Receipt: `logs/submission-browser-scale-8gpu1h-20260912.json` under runtime.
The controller uses `--browser-pair`: **256 then 192** browser slots, fixed
**TP4/DP2**, up to 25 minutes per case. Both start independently from SFT with
zero Adam updates; they do not load an RL checkpoint or prior rollout batch.
There is no separate topology sweep in this one-hour request. This measures
the two browser limits on one candidate layout, not the optimum across layouts.

Outputs: `benchmarks/browser-scale-291224/tp4-b256/` and `tp4-b192/` under runtime.
W&B IDs: `browser-scale-291224-tp4-b256` and `browser-scale-291224-tp4-b192`.
No case writes `current_baseline.json` or the existing `qcq7i4ug` W&B identity;
job 290926 continues separately. A CPU controller test verifies independent SFT
starts, unique W&B IDs, and that the lower browser level is attempted even if
the higher level fails. Eight tests pass. The two-hour topology and four-hour
combined plans below were prepared alternatives; neither was submitted.

**Result: both high-concurrency settings rejected.** Job 291224 finished after
**15 minutes 21 seconds (2.047 GPU-hours used)**; the two failed workers were
stopped early rather than consuming the full hour. Both initialized eight-GPU
TP4/DP2 model serving from SFT, but neither collected an accepted prompt group,
performed an optimizer update, or saved a training checkpoint. This does not
validate optimizer scaling or establish a winning topology.

At the first stop decision, 256 slots had completed 111 groups with zero
accepted groups and zero judge log events; additional failures continued during
shutdown. At the lower-count stop decision, 192 slots had completed 176 groups
with zero accepted groups. Individual environment-log audits found 612 screenshot
timeout logs and 386 navigation timeout logs among 1,180 files for 256 slots;
the interim 192-slot audit found 491 and 307 respectively among 932 files.
These are environment-log counts, not independent task-success denominators.
Combined Ray logs undercount repeated failures. Raw logs are preserved as
`environment_logs.zip` in each case directory.

There was no host OOM; sampled host memory peaked around 244.5 GiB and 228.2 GiB.
A single-browser diagnostic on the same node captured a local HTML page and
`example.com` successfully in under one second each. High-concurrency browser
operation was unreliable; the precise bottleneck remains unconfirmed, so these
results do not establish a universal hardware browser limit. Test substantially
lower concurrency before promoting an eight-GPU training profile.

Both separate W&B runs have `benchmark/status=stopped_early_browser_failures`
and failure counters in their summaries; no training-reward points were fabricated.
Final evidence: `benchmarks/browser-scale-291224/allocation_end_audit.json`,
per-case failure/health/environment audits, and `wandb_failure_audit.json`.
The main run remains job 290926 / `qcq7i4ug`, on its preserved source.
Project quota prevented committing this final browser-pair feature; the tested
code is preserved with hashes in this benchmark's `prepared_code/` directory.

**Failure diagnosis and next test.** The concrete failing path is
`env_server.reset_env → WebEnv.reset → get_screenshot → page.screenshot`:
`Page.screenshot: Timeout 30000ms exceeded`. Example task `webvoyager/139444`
failed this way at both browser counts. The trace says `fonts loaded` before
the timeout, so it does not establish font loading as the cause. Navigation
also failed with `Page.goto: Timeout 20000ms exceeded`, retried up to three
times while waiting for `load`. These failures become HTTP 500 responses from
the local `/reset` endpoint before a usable initial observation is available.

Job CPU accounting sampled average usage equivalent to 15.42 CPU cores at 256
slots and 17.53 at 192, with sample peaks of 31.66 and 24.26 respectively out
of 64 allocated CPUs. CPU pressure telemetry was unavailable. Thus neither
CPU exhaustion nor a general maximum supported browser count is proven. The
single-browser control used easy pages, not the failing task URLs, so it cannot
exclude site-specific behavior or distinguish browser rendering, process
startup, network waits, and concurrency effects.

Next use a **browser-only diagnostic**, with no policy model or judge calls:
replay the same failing URLs at concurrency 1, 32, 64 and then 96 if reliable,
on the same CPU allocation. Record launch, navigation, screenshot and cleanup
latencies separately, along with success rates, actual live browser/renderer
counts, CPU use and network failures. Compare simultaneous versus staggered
browser launches to separate startup bursts from steady concurrency. Preserve
the current timeout/navigation semantics for the control; change one setting
at a time only after reproducing a specific cause. Only a browser limit that
passes this control should enter a fresh eight-GPU full-iteration test. None
of these follow-up allocations has been submitted or approved.

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
[Separate W&B diagnostic](https://wandb.ai/zixianma/openwebrl-evals/runs/browser48-hold-290926-20260913)
finished and synced all three cases. Diagnostic code supports an optional
bounded hold and records setup/reset timestamps; five CPU regression tests pass.

<a id="browser-diagnostic-followup-20260912"></a>

**Completed diagnostic (job 291905).** Ran September 12, 22:01:39–22:07:58
PDT; Slurm `COMPLETED`, exit 0, 6m19s, approximately 0.84 allocated GPU-hours.
Initial-observation success for immediate launches was 96/96 at 32 workers,
90/96 at 64, and 11/96 at 96. With 250-ms launch spacing it was 96/96,
96/96, and 95/96 respectively. The staggered 96-worker case peaked at only
77 active workers, so this does not demonstrate reliable 96-browser concurrency.
All six concurrent cases processed the same 96 requests in roughly 51–59 seconds;
there was no demonstrated throughput benefit above 32. Challenge pages count as
successful screenshots and are flagged separately; these are not task rewards.
The conditional RL confirmation was skipped because neither 64 nor 96 passed
the 95% success threshold with immediate launches. The allocation ended early;
main training was unchanged. W&B reported successful final sync to
[browser-diagnostic-291905](https://wandb.ai/zixianma/openwebrl-evals/runs/browser-diagnostic-291905).
Full results: `/gpfs/scrubbed/zixianma/openwebrl-runtime/benchmarks/browser-diagnostic-291905/summary.json`.

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

<a id="arm-overnight-supervision-20260920"></a>

## ARM overnight supervision — 2026-09-20

The user requested monitoring of all ARM training/evaluation jobs and repair of
observed failures. Persistent host supervisor: `scripts/supervise_arm_runs.py`,
PID **1107462** (restarted from 1080944 to include B/C continuations), launched
for 36 hours. It tracks training jobs **303573, 303574, 309053, 309054, 309490,
310981, 311202, 311203, 311962, 311964, 311965** and evaluation jobs **309685, 309686, 309687**. Full
snapshots are recorded every 15 minutes; cheap file/scheduler checks detect
failures and stage changes every minute. Unrelated cooking jobs are outside
this supervisor's scope.

Durable status is under runtime
`arm-turn-bonus-preparation/overnight-20260920/`: `latest.json`, `alerts.json`,
`history.jsonl`, and `supervisor.log`. Training snapshots include the latest
durable iteration and Adam counter, collection/optimization progress, finite
gradient/clipping checks, and the embedded monitor's GPU/memory/W&B telemetry.
Evaluation snapshots count per-task archives and verdicts, check GPU restore
evidence, and sample GPU/memory/W&B state. A lack of measured progress for 30
minutes raises an inspection alert; it does not automatically discard a slow
browser collection.

Automatic recovery is deliberately narrow: a terminal evaluation with native
exit code zero and all 300 archive/verdict pairs can have bookkeeping finalized
again without browsers or GPUs. Arbitrary source fixes and training restarts
require active-agent investigation. This supervisor neither submits allocations
nor changes reward gates/hyperparameters. Replacement allocation budgets still
require exact approval under root `AGENTS.md`; no new compute was authorized
merely by starting this CPU monitor.

The separately verified checkpoint watcher (PID **2079618**) releases held
evaluation **309687** once all-failure job **309490** has durably completed
iteration 80. It checks every five minutes and allocates no GPUs while waiting.
All batch controllers own/await their workers and exit when complete or failed.

Initial health check: no nonfinite training gradients or OOMs observed; original
and additive clipping was below 2%, and B was making collection progress. The
older additive allocation **303574** completed normally at iteration **82**,
**1,064 Adam updates**, with insufficient reserved time for another full cycle.
Its resource release is a budget stop, not a crashed training process.

Subsequent verified boundary: original **303573** completed iteration **85**,
**1,002 Adam updates**, and stopped normally with insufficient time for another
cycle. Both older training allocations have released resources. The all-failure
continuation **309490** has started on g020 and is collecting iteration 79.

B **309053** stopped at calibration before any optimizer update: all checks
passed except bonus/outcome RMS **10.1672%**, just above the old 10% ceiling.
Its batch/cursor and full auxiliary population are preserved. See the
[staged recovery](ARM_INTEGRATION_PLAN.md#arm-bc-calibration-recovery-20260920)
for the approved guard and replay checks. Replacement **310981** is submitted
and queued. C **309054** failed before updating because its auxiliary tensor
exceeded 2 GiB; its corrected fresh restart is approved and queued as **311203**
(4 H200 × 7h, 32 CPUs, 480 GiB), preserving beta=0.5 and using the approved
13.67% RMS guard. GPU validation will occur at startup.
Original/additive iteration-80 evaluations completed and released their GPUs;
both have 300 per-task rollout archives and verdict records.

<a id="arm-additive-to100-prepared-20260920"></a>

## Additive ARM continuation to iteration 100 — submitted 2026-09-20

Approved and submitted as **311202**, initially queued for priority. Resume
`evaluations/arm-failure-additive-303574/runtime/iter_0000081`: **82 completed
collections, 1,064 Adam updates**, with the matching consumed dataset cursor
and scheduler sample counter **272,384**. Preserve source
`reference-arm-failure-additive-20260914-v3`, the original five-distinct gate,
beta=0.5, q=0.20, 48 ordinary groups plus up to eight auxiliary failure groups,
and W&B `arm-failure-additive-295786` in project `openwebrl`. B/C's experimental
gate/credit/guard changes are not part of this continuation.

The last ten checkpoint-to-checkpoint intervals averaged **67.32 minutes**,
with median **54.18**, range **46.15–128.16**. Eighteen remaining collections
suggest roughly **16–21 hours**, plus startup/save margin. Approved request:
**one 4-H200 × 24-hour allocation, 32 CPUs, 480 GiB, 32 browsers** (96 GPU-hours;
Slurm estimate $86.40).
The controller exits after saving iteration 100 or a time/health gate; the
24-hour budget is a ceiling, not a guarantee that every collection will fit.

Prepared batch script: `scripts/resume_arm_additive_to100_4gpu.sbatch`.
Checkpoint metadata/counters/shard extents and exact cursor identity passed
validation. Native CLI and scheduler checks confirmed target 100, TP4,
32 browsers, restored LR/weight decay and sample counter; no GPU was started.
The batch controller verifies actual GPU restoration before collecting,
preserves the optimizer/W&B lineage, and owns/awaits the training and monitor
workers. Plans and readiness evidence are under runtime
`arm-turn-bonus-preparation/additive-to100-20260920/`, including approval,
submission and exact submitted-plan receipts.


### September 20, 16:33 UTC: post-launch status and failure inspection

All-failure 309490 has durably completed **90 iterations / 1,142 Adam updates**
and is collecting 91. Its iteration-80 evaluation 309687 completed all 300 tasks
with 300 addressable rollout/verdict pairs; no iteration-90 evaluation is queued.
Original's earlier allocation stopped normally at 85.

Additive 311202 saved **85 iterations / 1,096 Adam updates**, then failed while
collecting 86. The fatal stack is `rollout_transport._encode_file_backed` at
`numpy.tofile`: `OSError: Not enough free space to write 30474240 bytes`.
Ray reported g008's 446.85 GB local `/tmp` filesystem had only 0.0035 GB free.
This is node-local storage exhaustion; the saved GPFS checkpoint is preserved.
No replacement allocation has been submitted or artifacts deleted.

B recovery 310981 failed before updating on the saved-batch provenance check
`admitted_distinct_action_counts`; inspect report/schema compatibility before
retrying, without weakening the content check. C restart 311203 completed
normally after six durable iterations / 92 Adam updates and released its
allocation under the remaining-budget gate. Persistent supervision recorded
these failures; it cannot repair source or submit replacement allocations.


<a id="arm-additive-to90-storage-recovery-20260920"></a>

### Additive iteration-85 → 90 storage recovery — submitted September 20

The user requested continuation to iteration 90 after job 311202 failed. Its
fatal write targeted `/tmp/arm-variant-311202-multimodal` on g008: file-backed
CPU image tensors accumulate because the frozen transport retains them for the
process lifetime. The checkpoint/recovery archives are on scrubbed GPFS, which
was not the failing filesystem. Resume from verified `iter_0000084`, **85 completed
iterations / 1,096 Adam updates**, with its exact saved task cursor. The incomplete
iteration-86 collection has no complete replay batch and will be recollected.

The prepared launcher `scripts/resume_arm_additive_to90_shared_4gpu.sbatch`
selects `ARM_VARIANT_MULTIMODAL_STORAGE=shared`. The controller assigns a
job-specific directory under runtime `multimodal-scratch/arm-variant-JOB`, checks
at least **2 TiB filesystem free space**, then performs a write/fsync/read probe
before GPU restoration. It preserves the files; no storage cleanup is authorized
or performed. The free-space measurement is filesystem capacity, not a certified
per-user quota. Frozen model/reward/optimizer code remains unchanged; beta=0.5,
q=0.20, five-distinct candidates, 48 mixed groups plus up to eight auxiliary
failure groups, TP4/32 browsers, and W&B `arm-failure-additive-295786` are preserved.

Five iterations remain. The latest completed checkpoint intervals were **47.95
and 45.45 minutes**, suggesting roughly four hours of training, with slower
shared I/O and startup/save margin. Approved allocation: **4 H200 × 6 hours,
32 CPUs, 480 GiB**, 24 GPU-hours, stopping at iteration 90. The controller verifies
GPU optimizer restoration before collection and owns/awaits all training/monitor
workers. The user approved this exact replacement budget; submitted as **311962**
at 16:40 UTC, initially pending for priority. Slurm estimates $21.60.
The persistent supervisor and in-job monitor cover this continuation.

Preparation and CPU validation receipts are under runtime
`arm-turn-bonus-preparation/additive-to90-storage-20260920/`. GPU execution and
throughput on the new storage path remain to be checked when allocated.

CPU readiness passed: native argument parsing and checkpoint scheduler load
preserve the **280,576** sample counter (1,096 × 256), LR 1e-6, weight decay 0.1,
TP4, 32 browsers, two PPO epochs and target 90. The shared-directory write/read
probe and frozen transport float32/bfloat16 mmap roundtrip passed. Low-space and
mismatched-path checks reject before GPU restoration. A generic actor-entry
preflight hit its uninitialized distributed-group boundary; the final CPU check
validates native arguments/scheduler only and makes no GPU-restoration claim.


### B/C outcome audit after the additive recovery submission

Additive recovery **311962** is submitted for the approved 4 H200 × 6h profile,
using shared multimodal storage; it was pending for priority at 16:43 UTC.
Its persistent supervisor is PID **1080944**, with 15-minute full snapshots.

B recovery's provenance-check failure is repaired in a new frozen v3 source;
29 tests and native CLI checks pass. No replacement B allocation is submitted.
C's six checkpoint receipts establish 92 completed optimizer updates; its normal
stop was the 90-minute cycle reserve. See the consolidated
[B/C audit](ARM_INTEGRATION_PLAN.md#arm-bc-firstupdates-20260920) for training curves,
coverage, reward-scale checks and the lack of held-out evaluation.


### September 20, 16:48 UTC: B/C continuations submitted

B **311964** replays the preserved zero-update first batch using the fixed v3
source; C **311965** resumes checkpoint `iter_0000005` with 92 optimizer updates.
Both use the established 4 H200 × 7h / 32 CPU / 480 GiB profile, shared GPFS
multimodal scratch and an iteration-10 cap. Both were initially pending.
The continuation routing fix preserves optimizer/cursor restoration rather than
calling first-launch validation on C's inherited ablation flags. Two routing
regressions pass; C's native scheduler and actual GPU-restore command dry run
pass. Actual GPU restoration is checked at allocated startup.

Supervisor PID **1107462** covers both jobs and additive-to-90 **311962**, alongside
all-failure training. See the [consolidated launch record](ARM_INTEGRATION_PLAN.md#arm-bc-continuations-20260920).

<a id="arm-training-status-20260920-2224"></a>
### September 20, 22:24 UTC: additive reaches 90; B/C are training

| Run | Job | Durable completed iteration / Adam updates | Current state |
| --- | ---: | --- | --- |
| Original bonus | 303573 | 85 / 1,002 | Stopped; no active continuation |
| Additive | 311962 | 90 / 1,150 | Completed target, exit 0; 4h26m18s of 6h used |
| All-failure | 309490 | 98 / 1,222 | Collecting 99, target 100 |
| B: relaxed gate | 311964 | 4 / 62 | Collecting 5; about 2h52m allocation remaining |
| C: relaxed gate + action-equivalence credit | 311965 | 9 / 134 | Training 10, its target; about 3h14m remaining |

Latest checkpoint validation receipts, optimizer/scheduler agreement and shard
sizes were verified. B's first-batch auxiliary replay now passes the fixed
provenance check, followed by three fresh completed collections. C and additive
passed GPU restoration. Shared temporary storage has allowed additive to finish;
no new disk failure appears in the active runs.

Recent B collections have 15.95–18.27% ordinary-turn label coverage and
8.84–10.17% bonus/outcome RMS; C's latest four collections have 17.01–18.58%
coverage and 9.90–10.88% RMS. All pass the unchanged calibration guard. Latest
PPO KL/clip fraction: B 0.00263 / 1.29%, C 0.00254 / 1.25%, all-failure
0.00553 / 2.71%. These are training diagnostics, not held-out success estimates.

Resource samples at about 22:22 UTC showed active GPU use and zero OOM/OOM-kill
events. C uses about 473.6/480 GiB host RAM during training; B previously touched
its 480 GiB cgroup cap (499 max events, no OOM) and is now about 365.6 GiB.
All-failure uses about 304.4 GiB. Memory pressure, especially C's final update/save,
remains a monitoring concern. W&B histories are advancing for all three active
runs. The persistent supervisor retains 15-minute full checks.

Recent cycle times suggest all-failure can finish 100 in roughly 1–1.5 hours;
C is in its final PPO epoch and should finish substantially sooner. B takes
about 63–67 minutes per iteration, so its current allocation is more likely to
finish around iteration 6 than its cap of 10. These estimates include no new
compute allocation. Iteration-90 checkpoints now exist for additive and
all-failure, but no iteration-90 evaluation has been submitted.

### September 20: B/C W&B display names corrected

With explicit user approval, the two live W&B display names were changed and
read back through the API: `arm-gate-b-309053` is now
`ARM-B | relaxed gate | response-index credit`; `arm-gate-c-309054` is now
`ARM-C | relaxed gate | duplicate-aware credit`. The common old display name
`arm-min2-independent-credit` came from using the shared group as the run name.
Run IDs and metric histories remain intact. Receipt:
`arm-turn-bonus-preparation/wandb-bc-display-names-20260920.json`.
Future continuation launch plans should explicitly retain these variant names;
the active frozen training sources were not edited.

### September 20, 22:40 UTC: iteration-90 evaluations submitted; C finished

Approved full-300 evaluations: additive **313187** and all-failure **313188**,
each 2 H200 × 1h, 16 CPUs / 480 GiB, initially pending for priority. Both use
checkpoint index 89, the established native GPT-4.1 monitor and per-task
rollout/verdict persistence. The batch controllers await their workers. See
[evaluation launch details](RL_EVALUATION.md#arm-iter90-readiness-20260920).
The existing supervisor was restarted as PID **2486117** to include both jobs;
ownership and its first snapshot were verified, retaining 15-minute full checks
and one-minute lightweight failure checks. No extra allocation is authorized.

C **311965** completed its target **10 iterations / 148 Adam updates**, exit 0,
in 3h54m43s. The final `iter_0000009` validation, save receipt and shard sizes
pass. Its controller released the allocation with roughly three hours left;
there is no pending stage assigned to that completed allocation.

### September 20, 22:55 UTC: B/C to 20 and held evaluations submitted

| Stage | Job | Approved resources | Initial state |
| --- | ---: | --- | --- |
| B continuation to 20 | 313208 | 4 H200 × 18h, 32 CPUs, 480 GiB | Dependency: afterany 311964 |
| C continuation to 20 | 313210 | 4 H200 × 12h, 32 CPUs, 480 GiB | Pending priority |
| B full-300 at 20 | 313209 | 2 H200 × 1h, 16 CPUs, 480 GiB | Held for checkpoint |
| C full-300 at 20 | 313211 | 2 H200 × 1h, 16 CPUs, 480 GiB | Held for checkpoint |

Approval and submission receipts are in
`arm-turn-bonus-preparation/bc-iter20-20260920/`. Watchers **2553206/2553214**
own the B/C evaluation holds, verify checkpoint lineage/counters/shards and then
release their single recorded evaluation. Supervisor **2553222** includes all
four jobs. Each batch controller awaits its own workers; monitoring is bounded
CPU work and cannot submit new allocations. B's final resume checkpoint is
resolved after its current job finishes. C resumes iteration 10 / Adam 148.
Distinct B/C W&B display names are preserved in continuation plans.

The separately requested [rescue-yield pilot](ARM_INTEGRATION_PLAN.md#arm-rescue-yield-pilot-20260920)
was CPU prepared at this point; its later approval and submission are below.

### September 20, 23:28 UTC: rescue-yield pilot submitted

Job **313256** requests the separately approved **2 H200 × 2h, 16 CPUs,
480 GiB RAM** (four GPU-hours), initially pending priority. Slurm estimates
$3.60. Approval, submission and readiness receipts are in
`arm-turn-bonus-preparation/rescue-yield-20260920/`.
The controller `scripts/run_arm_rescue_yield.py` owns and awaits the selector
and actor/evaluation workers; W&B uses `openwebrl-evals/arm-rescue-yield-313256`.
Output is `evaluations/arm-rescue-yield-313256/`. The existing supervisor now
tracks smoke/screen/retry counts, checkpoint restoration, failures and final
yield without submitting further allocations. GPU/browser smoke validation
must pass before the full screening stage. Every attempt retains its rollout
and judge metadata. No training updates or model changes are requested.

**Startup failure and prepared repair:** 313256 exited on g015 after 18 seconds,
before any trajectories: the selector process could not import `openwebrl`.
The controller now supplies frozen-source `PYTHONPATH` and Python-header
`CPATH`; the real selector CLI/dependency checks pass. The replacement source
is `reference-arm-rescue-yield-20260920-v2`, preserving the failed v1 source and
artifacts. `proposed-retry-v2.json` requests 2 H200 × 1h59m, 16 CPUs / 480 GiB;
it is not submitted or newly approved. Combined runtime ceiling with the
failed attempt is 3.977 GPU-hours. Supervisor **2698912** includes the rescue
pilot and all previous jobs, with the same 15-minute full snapshots.

Additive iteration-90 evaluation **313187** failed after 1m27s before any tasks:
two concurrent evaluations on g021 selected the same SGLang distributed ports
(15004 and 15035). `MASTER_PORT` alone does not control those engine ports.
New evaluation sources `reference-arm-eval80-{variant}-20260920-v2` now honor
`OPENWEBRL_ROLLOUT_PORT_BASE`; controllers hold a node-local file lock over a
free 256-port block until their workers exit. Three tests verify concurrent
lease exclusion/release, occupied-port rejection and native port-cursor behavior;
16 evaluator regression tests pass. The pilot uses the same isolation.
All-failure **313188** continues using its already-running v1 source; active
workers are unchanged. No replacement additive allocation was submitted.

### September 20, 23:34 UTC: corrected rescue-yield pilot submitted

The user approved **2 H200 × 1h59m, 16 CPUs / 480 GiB**; replacement job
**313264** is submitted, initially pending priority (Slurm estimate $3.57).
Approval and submission receipts are `retry-v2-approval.json` and
`retry-v2-submission.json` under the rescue-yield preparation directory.
Source v2 preserves the same cohort, policy, judge and zero-update protocol;
its changes isolate native model-server ports and fix selector import/header
paths. Output is `evaluations/arm-rescue-yield-313264/`, with W&B identity
`openwebrl-evals/arm-rescue-yield-313264`. The supervisor includes the replacement;
checkpoint restore, selector smoke, screening and retries still require live
verification. Failed job 313256 and all its artifacts remain preserved.
Supervisor **2726562** owns the updated monitor; its first snapshot includes
313264. At 23:35 UTC the replacement remained pending priority. In the same
snapshot, all-failure training **309490** had completed **iteration 100** with
exit 0, and its iteration-90 evaluation **313188** had saved 191/300 task
rollouts and verdicts. These are progress counts, not final evaluation results.

### September 21, 00:34 UTC: training and evaluation status

| Track | Durable iteration / Adam updates | Current state |
| --- | --- | --- |
| Original bonus | 85 / 1,002 | Previous allocation completed; no continuation queued |
| All-failure bonus | 100 / 1,242 | 309490 completed successfully; final checkpoint verified |
| Additive bonus | 90 / 1,150 | 311962 completed successfully |
| B relaxed gate | 6 / 90 | 311964 completed; 313208 pending priority to continue to 20 |
| C gate + action credit | 10 / 148 | 313210 training iteration 11 on g013; calibration passed |

B's final saved checkpoint/counters/shard extents and the queued job's resolved
resume plan now pass CPU checks. Plan:
`arm-turn-bonus-preparation/bc-iter20-20260920/B-final-checkpoint-plan.json`.
B/C evaluation jobs 313209/313211 remain held for verified iteration 20.

All-failure iteration-90 evaluation **313188** completed in 33m50s, exit 0,
with all 300 rollout/verdict pairs saved: **33.67% overall / 46.54% valid-only**.
[Metric audit and historical comparison](RL_EVALUATION.md#arm-iter90-results-20260921).
Additive evaluation 313187 failed before collecting tasks; its port-isolation
fix is prepared but no replacement allocation is submitted.

Rescue pilot **313264** passed real GPU restore, selector preflight and all three
smoke pairs. All 320 screen attempts are saved; 18 of 64 tasks have five valid
actor failures, and eight were selected for 48 retry attempts. At this check,
24 retries were saved and collection continued. The live counts are in
`evaluations/arm-rescue-yield-313264/status.json` and phase-record directories.
No final rescue-yield claim yet. Latest resource samples show no OOM events;
C's W&B identity remains `arm-gate-c-309054`, and rescue uses
`openwebrl-evals/arm-rescue-yield-313264`.

<a id="arm-live-monitor-20260921"></a>

### September 21: additive retry and automatic live reports

Additive iteration-90 full-300 evaluation retry **313408** was submitted at
00:53 UTC, following the user's request, with the established **2 H200 × 1h,
16 CPUs / 480 GiB** profile. It uses the v2 port-isolated source and preserves
all 300 rollout/verdict pairs; the failed prior attempt collected no tasks.
Receipts and the actual job plan are `iteration90-evals/additive-retry-v2-*`
under the runtime preparation directory.

Rescue pilot **313264** completed successfully in **50m24s** with **374 saved
trajectories**. On the eight selected all-failure tasks, ARM rescued **0/8**,
one ordinary retry **1/8**, and five ordinary retries **3/8**. See
[results and limitations](ARM_RESULTS.md#arm-rescue-yield-313264).

The user requested monitoring all running jobs and immediate result reports.
Supervisor **3094709** runs for 48 hours, with **60-second lightweight checks**
and **15-minute detailed health checks**. It tracks existing ARM training,
held/queued evaluations, rescue pilots and the new additive retry. New ARM
output roots are discovered from the live Slurm queue and retained in
`discovered-jobs.json`; all other current user Slurm jobs appear with scheduler
status only. No new allocation, restart or external message is sent automatically.

The [live HTML report](arm_results/rl_integration/live-status.html) refreshes
every minute, showing verified completion rates and recent transitions;
[JSON](arm_results/rl_integration/live-status.json) provides the same data.
`scripts/report_arm_jobs.py` records deduplicated result-ready, failure and
checkpoint events in runtime `arm-turn-bonus-preparation/overnight-20260920/events.jsonl`.
Reporting survives supervisor restarts; failed/incomplete jobs do not receive
final success rates. Five reporting tests and two supervisor tests pass.
These are automatic **local report updates**, not push notifications to this
chat; no asynchronous chat-delivery tool is available in this session.

<a id="stage1-baseline-additive-to100-20260921"></a>

## Baseline and additive stage-1 continuation to iteration 100 — submitted September 21

The user requested both outcome-only and additive ARM at iteration 100 followed
by full-300 Online-Mind2Web evaluation. This preserves each stage-1 recipe;
it does not start the separately prepared stage-2 experiment.

| Run | Verified resume origin | Target | Approved single allocation, including evaluation |
| --- | --- | --- | --- |
| Outcome-only baseline `qcq7i4ug` | Iteration 90 / 1,016 Adam updates | 100 | 4 H200 × 12h; 32 CPUs, 480 GiB, 32 browsers |
| Additive ARM, existing W&B lineage | Iteration 90 / 1,150 Adam updates | 100 | 4 H200 × 12h; 32 CPUs, 480 GiB, 32 browsers |

**Submitted September 21 after exact budget approval:** outcome-only **313668** and additive **313669**. Both initially queued for priority; each is four H200s × 12 hours including evaluation. Slurm estimates $43.20 per allocation. The independent selection-quality audit was subsequently approved, including its API payload transfer, and submitted as job **313774** (two H200s × two hours).
The persistent supervisor tracks both job IDs through restore, training and evaluation, retaining 15-minute detailed checks and one-minute local failure checks. Both launchers also retain their existing per-run monitors.

The two jobs cap usage at **96 GPU-hours combined**, releasing allocations on
completion. Recent ordinary cycles were about 45–59 minutes; ten cycles plus
restoration and evaluation should fit roughly 10–12 hours, but browser timing
can vary. A budget stop preserves the latest checkpoint and cannot silently
substitute an earlier iteration for the iteration-100 evaluation.

`prepare_stage1_to100.py` freezes and hash-validates the original sources.
Baseline uses `reference-stage1-to100-20260921-v1` plus the matching pending-eval
recovery source; additive uses `reference-arm-additive-to100-20260921-v1`.
Changes are limited to target/runtime, shared image scratch, port isolation,
and lossless baseline evaluation persistence. Rewards, data recipe, Adam state,
learning rate (1e-6), effective batch (256), five attempts per group and two PPO
epochs are preserved. Baseline's scheduler offset remains one; additive's is zero.
The iterations align, while optimizer-update counts and total compute differ.

`run_stage1_to100_4gpu.sbatch` calls `run_stage1_to100.py` with
`STAGE1_VARIANT=baseline` or `additive`. The batch controller owns and awaits
GPU restore verification, training, and evaluation. Baseline retains its native
scheduled evaluation and W&B identity; additive reserves the final hour for its
standalone evaluation. Training logs to `openwebrl`, standalone evaluation to
`openwebrl-evals`. Both use local browsers, GPT-4.1/action-history judging,
temperature zero, 30 turns, and 4,096 response tokens on all 300 tasks.
Completion requires all 300 unique cohort IDs, per-task verdicts, and nonempty
rollout archives. Additive's evaluation cannot run until iteration 100 is verified.

CPU preparation inspected checkpoint metadata, not model tensors. Baseline dry
run confirms 100 iterations / four GPUs / 32 browsers / 12 hours and the inherited
optimizer recipe. Additive's native CLI preflight also confirms the 100-iteration
target, four-GPU topology, LR 1e-6, batch 256, 48 groups, five attempts, two PPO
epochs, and restoration of the checkpoint scheduler. Resume/routing and
evaluation-persistence tests passed. Native
GPU restoration remains mandatory at allocation startup. Preparation receipts
are in runtime `arm-turn-bonus-preparation/stage1-to100-20260921/`.

The independent [64-state selection-quality audit](ARM_INTEGRATION_PLAN.md#arm-selection-quality-audit-20260921)
requests another 2 H200 × 2h and a $20 / 231-call teacher cap. All three proposed
allocations together cap GPU use at **100 GPU-hours**; ordinary training/evaluation
terminal-judge API usage is additional to the audit's teacher cap.

<a id="baseline-to100-scheduler-fix-20260921"></a>
### September 21, 16:08 UTC: baseline resume failure fixed on CPU

Baseline **313668** failed after **1m47s** during GPU restore verification,
before any training or browser collection. Increasing the target from 90 to 100
changed the constructed scheduler horizon to 23,808 while the saved scheduler
contains 21,504. Megatron correctly rejected that mismatch. The last durable
baseline remains iteration 90 / 1,016 Adam updates; no checkpoint was altered.

Prepared **`reference-stage1-to100-20260921-v2`** and matching pending-evaluation
source add `--use-checkpoint-opt-param-scheduler` only when resuming. This keeps
the saved scheduler and learning rate instead of resetting or overriding them.
Three CPU scheduler tests pass, including reproducing the old assertion and
verifying exact scheduler-state/optimizer-LR restoration under an extended
horizon. Source hashes and Python syntax pass. Allocation-aware dry run cannot
inspect the expired Slurm job; GPU restore verification remains mandatory in a
new authorized allocation. No replacement paid job has been submitted.
[Readiness receipt](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/stage1-to100-20260921/baseline-v2-readiness.json).

Additive **313669** remains on its unchanged v1 source: iteration 92 saved /
1,174 Adam updates, collecting 93 toward 100 with no current monitor alert.
B/C completed iteration20; C's full300 result is 36.67% / 44.53%, and B evaluation
313209 is queued for priority. [C results](RL_EVALUATION.md#arm-gate-c-iter20-results-20260921).

<a id="mig-feasibility-20260921"></a>
### MIG feasibility for queued jobs — September 21

Read-only audit; no MIG jobs submitted and no existing jobs moved. Tillicum's
MIG partition exposes `h200_1g.18gb` devices on g023/g024. At inspection, Slurm
reported 11/112 slices allocated (45 unallocated on g023, 56 on g024); this is
an inventory snapshot, not a guaranteed start time. CPU counts are not GPU
availability counts. Partition defaults are one CPU / 30 GiB host RAM per slice.

| Pending work | MIG assessment |
| --- | --- |
| ARM fixed100 audit 314664 | Plausible after adaptation: independent actor and selector processes, one slice each; no tensor parallelism; reduced request/browser concurrency, unchanged evaluation protocol |
| B/C iteration20 evals 313209/313211 | Plausible independent actor-only workers, but held until checkpoints are ready |
| Baseline/additive to100 313668/313669; ScreenSim RL 313606 | Keep the current full-H200 distributed training launchers; no validated MIG training path |
| CookSim evaluation array 313573 | Current Vulkan rendering preflight requires graphics unavailable on H200 MIG |

Our installed runtime uses PyTorch 2.9.1+cu129, SGLang 0.5.6.post2 and NCCL
2.27.5. NVIDIA documents **experimental** multi-MIG support starting in NCCL
2.31, requiring CUDA driver 13.0 or newer. An environment upgrade alone does not
validate the training/evaluation stack. H200 MIG also lacks graphics APIs.
Do not treat multiple slices as pooled VRAM or drop the existing TP2/TP4 jobs
into this partition unchanged.

Memory feasibility is an estimate, not a GPU test: SFT weights occupy about
8.99 GiB on disk; SelectionARM's weight index reports 8.27 GiB. BF16 KV for
36 layers × 8 KV heads × 128 dimensions costs 4.5 GiB per independent 32,768-token
sequence, before vision activations/workspace/framework overhead. One model per
slice with a small batch is plausible; the current five parallel candidates and
16 browsers are not validated. Preserve BF16, context, response length, seeds,
judge, and fixed100 IDs; control memory by serializing candidate requests or
sharding tasks across independent workers. RL checkpoints require a validated
standalone export/load path that reproduces the restored model. A bounded
memory/throughput pilot should precede moving queued evaluations. Any new MIG
allocation still requires its exact resource/time budget approval.

Evidence: runtime preparation `task-success-audit-20260921/mig-feasibility.json`.
[Tillicum scheduling](https://hyak.uw.edu/docs/systems/tillicum/scheduling-jobs/),
[NCCL MIG support](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-mig-instances),
[NVIDIA graphics restriction](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/deployment-considerations.html#application-considerations).

<a id="mig-probe-ready-20260921"></a>
### Separate actor/ARM MIG probe — prepared, awaiting exact allocation approval

Prepared `scripts/run_arm_mig_probe.sbatch`: **2 × 18-GB MIG slices × 1 hour**,
2 CPUs / 60 GiB host RAM, estimated compute **$0.2574**, and native GPT-4.1
terminal judging capped at **$1 / 6 requests**. No submission has been made.
This is separate from full-H200 audit **314664**, which is already running.
The user requested keeping full-H200 execution as fallback if MIG fails.
The full-H200 jobs and B/C checkpoint watchers remain in place.

The batch controller owns and awaits two one-slice Slurm service steps. It
checks CUDA identities and memory so actor and SelectionARM occupy distinct
18-GB slices. The actor uses SGLang TP1, BF16, 32k context, one active request,
512-token prefill chunks, and no CUDA graphs. All five deterministic candidate
seeds and the selector permutation are preserved; requests are serialized.

Test sequence: full 32k actor KV-capacity request (synthetic, excluded from
benchmark results); shortest and longest archived audit prompts with five
real candidates and ARM selection; one actor+ARM and one actor-only browser
trajectory on the first two fixed100 IDs. Browser runs retain native GPT-4.1
judging, 30 turns, 4,096 response tokens, full history, and lossless per-task
rollout/verdict archives. Report this as a **diagnostic**, never as a success-rate
comparison. W&B project is `openwebrl-evals`. Stop on model/service errors,
budget exhaustion, missing archives, wrong device binding, or failed judging.

The first fit test uses the starting SFT weights as an **architecture proxy**.
It does not establish that the RL checkpoint conversion is correct. B/C follow-up
requires durable iteration20, correct lineage, a standalone checkpoint export
with weight/output parity verification, and a one-slice actor-only probe.
`bc-followup-readiness.json` records these gates; no idle GPU allocation is held
waiting for checkpoints, and no MIG production evaluation is released by this
probe. On any failed gate, retain the full-H200 evaluation route.

Frozen source: `reference-arm-mig-probe-20260921-v1`. Preparation manifest:
runtime `arm-turn-bonus-preparation/mig-probe-20260921/plan.json`. Three CPU tests
pass for distinct device binding, serialized candidates preserving seeds and
selection inversion, and requiring real saved rollout/verdict artifacts.
Python compilation and shell syntax checks pass. GPU fit, latency and live
browser behavior remain **untested** until the dedicated allocation runs.

<a id="overnight-monitor-20260921"></a>
### Overnight supervision — September 21

Persistent read-only supervisor PID **654085** is active for 48 hours, with
15-minute full progress/resource/W&B/checkpoint checks and lightweight 60-second
failure/status checks. B/C checkpoint watchers poll every five minutes and own
release of the already-approved iteration20 evaluations. Freshness verified at
07:26 UTC: B had durable iteration11 / 162 Adam updates; C had durable iteration17
/ 244 Adam updates. Audit314664 had 20/100 first-checkpoint tasks saved, no caught
exceptions and no selector fallbacks. Baseline/additive continuations remain
pending priority. This is a dated snapshot, not a live result claim.

The [live dashboard](arm_results/rl_integration/live-status.html) and runtime
`arm-turn-bonus-preparation/overnight-20260920/{latest,alerts}.json` record ongoing
updates. No new compute budgets or automatic paid resubmissions are authorized
by the monitoring request. The separate prepared MIG pilot still needs exact
resource/time approval under the root `AGENTS.md`.


<a id="baseline-mig-submitted-20260921"></a>
### Approved baseline replacement and MIG pilot — September 21, 16:11 UTC

Submitted **315098**, outcome-only continuation from iteration90 to100 with
full300 evaluation, **4 H200 × 12h, 32 CPUs, 480 GiB**. Source v2 preserves
the saved scheduler; the controller now runs an allocation-aware dry run before
GPU restore verification, training and evaluation. W&B lineage remains `qcq7i4ug`.
The job is queued for priority; no competing baseline training is active.

Submitted **315099**, **two h200_1g.18gb MIG slices × 1h, 2 CPUs, 60 GiB**, with
**$1 / six-call** terminal-judge cap. The initial six-second startup failed on a
controller-only helper import inside the frozen service launcher. Lazy imports
fix this without changing inference. Frozen v2 imports successfully in the actual
worker environment; three MIG CPU tests pass. Preserved the failed attempt at
`evaluations/arm-mig-probe-315099-startup1/` and requeued the same job with a
**59-minute** limit: total maximum is 59m06s, inside the approved one hour.
Actor/selector separation, full-context capacity and live-browser checks remain
GPU gates, not established results. B/C migration still requires checkpoint export
and parity verification; the existing full-GPU B evaluation remains running.

Approval and submission receipts are in runtime preparation directories
`stage1-to100-20260921/` and `mig-probe-20260921/`.

MIG retry is running on **g023**. Both services report exactly one
`NVIDIA H200 MIG 1g.18gb` device, with distinct UUIDs and 16 GiB exposed CUDA
memory each. This verifies device separation; model/context/browser tests are
still in progress. Both jobs are registered with the persistent supervisor.


### MIG pilot outcome and prepared context-probe fix — September 21

The second attempt of **315099** failed after 1m13s at the synthetic context
request, before browser trajectories or paid judge calls. Both actor and selector
loaded on distinct slices and served health checks. SGLang rejected input32,760
plus output8 because its API requires a strict total below32,768. This is a
probe boundary error, not evidence of MIG memory exhaustion. Frozen v3 requests
32,759+8 tokens; three CPU tests pass, but full-context and browser feasibility
remain unverified. The expired job is no longer requeueable; no replacement
allocation has been submitted. Total allocated startup/runtime across attempts
was 1m19s; evidence remains under `evaluations/arm-mig-probe-315099*`.

<a id="mig-corrected-probe-315402"></a>
### Corrected MIG probe 315402 — September 21

**Final result: passed; allocation completed and both slices released.** Across
all four attempts, Slurm recorded **18m05s** on two slices (0.603 slice-hours),
within the approved two-slice × one-hour budget. Two GPT-4.1 calls cost $0.01844.
The final source is `reference-arm-mig-probe-20260921-v6`.

| Check | Result |
| --- | --- |
| Device isolation | Actor and SelectionARM each see one distinct `1g.18gb` MIG instance; 16 GiB exposed to CUDA |
| Actor context footprint | 32,759 input + 7 generated tokens; 7.17s; zero retractions |
| Archived selection | Short and long histories both pass; five candidates generated serially |
| ARM browser trajectory | 7 turns, 7 selections, zero fallbacks; valid success verdict |
| Actor-only browser trajectory | 26 turns in 245.73s; valid failure verdict |
| Persistence | Both lossless `.pt` rollouts and per-task JSON verdicts verified; ARM result recovered without rerun |
| Tracking | [W&B diagnostic run](https://wandb.ai/zixianma/openwebrl-evals/runs/arm-mig-probe-315402), `probe/passed=1` |

These are **two different tasks**, used only to test the inference pipeline;
their outcomes are not an actor-versus-ARM performance comparison. Both models
used BF16; actor context was 32,768, TP1, one request at a time, CUDA graphs off.
This validates SFT-architecture inference on MIG, not additive-checkpoint parity,
training, or coverage-pilot throughput. Moving the actual coverage pilot still
requires a verified additive checkpoint export and collector validation. The
approved native four-H200 pilot remains queued as **315204**.
Machine-readable [diagnostic result](arm_results/rl_integration/mig-probe-315402.json).

The user approved the corrected **two 18-GB slices × one hour** feasibility test,
with 2 CPUs, 60 GiB host RAM and at most $1 for terminal judging. Submitted as
**315402**; it started immediately on **g023**. The first attempt passed the
near-32k actor request (32,759 input / 7 generated tokens in 7.16s), then generated
five candidates and obtained a valid ARM choice for the short archived state.
Both services loaded on separate MIG UUIDs. The long archived state exposed a
probe bug: history was already structured as `{thought, action}`, but the live
selector expected response strings. No browser trajectories or judge calls ran.

Frozen v4 converts that history without changing the selector payload and
validates the archived inputs before GPU startup; four CPU tests pass. The same
job was requeued with 57 minutes remaining after 2m33s used, so the combined maximum
is 59m33s within the approved one-hour budget. The first attempt is preserved at
`evaluations/arm-mig-probe-315402-attempt1/`. The retry passed near-32k generation
and both short/long-history offline ARM selections. The ARM browser trajectory
completed seven turns with seven valid selections and no fallback; its original
GPT-4.1/action-history verdict is success. Saving the verdict summary exposed a
second probe-only bug: `reward_key='judge'` indexed a native scalar reward as a
dictionary. The full atomic `.pt` rollout had already been saved. Its verdict
and record were recovered without repeating the browser or judge request.

Frozen v5 uses `reward_key=None`, verified through the real rollout/verdict writer.
The second attempt used 7m44s; the same job was released with **49 minutes** for
the remaining actor-only check, keeping cumulative runtime below one hour
(maximum 59m17s). It reuses the recovered ARM record, checks its actor/task/judge
identity and record hashes, and shares the original judge ledger ($0.009772,
one request before retry). The original ARM elapsed time was not saved and is
explicitly unavailable. Attempt two is preserved at
`evaluations/arm-mig-probe-315402-attempt2/`. The persistent supervisor tracks the
retry; at this point a full pass still required the actor-only rollout and saved
verdict, which subsequently completed in the final attempt above.

The 65-second third attempt stopped before inference because PyTorch reported
the same UUID for the two services on g024. Frozen v6 now obtains the compute
instance UUID through `cuDeviceGetUuid_v2`, retaining the PyTorch UUID and Slurm
device assignment for diagnosis, and verifies isolation **before loading either
model**. This is the [NVIDIA-documented MIG-aware UUID API](https://docs.nvidia.com/cuda/archive/12.2.2/cuda-driver-api/group__CUDA__DEVICE.html).
The final retry was allotted 48 minutes (maximum cumulative 59m22s); no new
allocation was submitted. It completed in 6m43s. Six CPU tests cover history conversion, serial
candidates, scalar-reward persistence, recovery provenance, and isolation checks.

This uses the starting SFT actor as an architecture proxy. It does not validate
an exported additive checkpoint or replace the native training collector. Keep
the full-H200 failure-coverage pilot **315204** queued while assessing the MIG
result. No additional MIG allocation or training is authorized by this test.
Artifacts: runtime `evaluations/arm-mig-probe-315402/`; submission/approval
receipts: `arm-turn-bonus-preparation/mig-probe-20260921/`.

<a id="arm-job-inventory-20260921"></a>
### ARM job inventory and remaining work — September 21, 16:00 PDT

| Experiment | Training endpoint / job | Latest full300 evaluation | Current state |
| --- | --- | --- | --- |
| Outcome-only | 90 / 1,016 Adam; continuation 315098 | 90: 33.67% / 45.50% | 4 H200 × 12h queued for priority; includes training to100 and full300 |
| Original bonus | 85 / 1,002 Adam; 303573 | 80: 33.33% / 45.05% | Stopped; no continuation queued |
| All-failure | 100 / 1,242 Adam; 309490 | 90: 33.67% / 46.54% | Training complete; iteration100 evaluation missing |
| Additive | 100 / 1,262 Adam; 313669 | 100: 36.33% / 50.23% | Training and all300 rollout/verdict archives complete; allocation released |
| B, relaxed gate | 20 / 284 Adam; 313208 | 20: 33.67% / 44.30%, 313209 | Training and full300 complete; no continuation queued |
| C, gate + action credit | 20 / 284 Adam; 313210 | 20: 36.67% / 44.53%, 313211 | Training and full300 complete; no continuation queued |

Rate pairs are overall / valid-only; different rows are at different iterations.
They are an operational inventory, not a matched comparison. The latest additive
result is audited in [RL_EVALUATION](RL_EVALUATION.md#arm-additive-iter100-results-20260921).

| Diagnostic / pilot | Job | State |
| --- | --- | --- |
| Failure-specific coverage, zero optimizer updates | 315204 | Running on g003, 4 H200 × 3h; restoration from additive90 passed, 26/48 mixed groups collected at 15:59 PDT; deferred labels and coverage/cost report pending |
| Failed-task rescue yield | 313264 | Complete; ARM 0/8, one ordinary retry 1/8, five retries 3/8; all 374 trajectories saved |
| Fixed-state selection quality | 313774 | Complete; 192 candidate sets scored |
| Actor+ARM fixed100 task success | 314664 | Complete; 200 primary evaluations across outcome-only20/90, all saved |
| Separate-slice MIG inference | 315402 | Passed with SFT actor; released after 18m05s total across attempts |

**Open items.** Baseline100 is already queued. All-failure100 is the clearest
missing full300 endpoint. Original bonus stopped at85, short of the earlier
90-iteration target; its full300 curve is also sparse (verified plotted points
51/70/80). Finishing original90 and any older missing evaluations needs a
separate resource decision, rather than silently comparing unequal endpoints.
B/C20 is complete; C40/60 is a proposed follow-up, not a launched continuation.
The failure-only beta0.5→1.0 training ablation and coverage-treatment training
are prepared directions but have no approved training allocation; finish the
coverage/cost pilot before choosing its training budget. MIG checkpoint export,
parity and native collector migration remain unvalidated; the running full-H200
pilot is independent of that optional migration.

The additive100 tables and comparison plot are now updated. No new compute was
submitted during this inventory. Older failed calibration/startup/evaluation
jobs in monitor history have successful replacement runs and are not additional
active jobs. Unrelated account jobs at this snapshot: cooking evaluations
313573_1 and 313573_2 running; ScreenSim training 315088 queued. They were not modified.

<a id="arm-bc-to60-prepared-20260921"></a>
### B/C continuation through iteration 60 — submitted, September 21

The user approved eight GPUs per training job. Submitted at 16:26 PDT:
**B 316247** and **C 316248**, each **8 H200 × 24 hours, 64 CPUs, 960 GiB RAM**.
Both are queued for priority as of 16:30 PDT. This replaces the earlier proposal
of two four-GPU stages per variant while retaining the **384-GPU-hour total cap**.
No four-GPU continuation from that proposal was submitted.

Both verified starting checkpoints are `iter_0000019`, with 284 completed Adam
updates: B from 313208, C from 313210. Preserve optimizer, scheduler, dataset
cursor and original W&B identities. B keeps the at-least-two-action gate with
response-index credit; C keeps that gate with action-equivalence credit. Beta
stays 0.5; the 48-plus-up-to8 group recipe, PPO2, global batch 256 and learning
rate 1e-6 are unchanged.

| Variant | Job | Owned stages, in order | Training profile |
| --- | --- | --- | --- |
| B: relaxed gate | 316247 | Train20→40 → full300 at40 → train40→60 → full300 at60 | TP8, 64 browser pool slots and 64-task gate |
| C: gate + action credit | 316248 | Train20→40 → full300 at40 → train40→60 → full300 at60 | TP8, 64 browser pool slots and 64-task gate |

The single controller per variant owns and waits for every worker. Evaluations
use eight GPUs and 32 browsers, temperature0, GPT-4.1/action_history, with all
300 trajectory archives and per-task verdicts validated before the next stage.
Training logs to `openwebrl`; these separate evaluation workers log to
`openwebrl-evals`. Distinct B/C W&B identities and names are preserved.

CPU native argument and saved-scheduler checks pass for both eight-GPU plans,
and 29 regression tests pass, including ordered stage handoffs and monitor
routing. The frozen scientific sources are unchanged; runtime snapshots add
leased model-server ports. Actual eight-GPU model/optimizer restoration is
verified inside each allocation before training; it has not run while queued.

The speedup is unmeasured. Previous four-GPU segments took roughly one hour per
collection, but eight GPUs do not guarantee twice the throughput. The controller
reserves one hour for each stage's evaluation against the allocation's actual
remaining time. If the requested checkpoint cannot be reached, it preserves
progress and stops rather than evaluating an earlier checkpoint or extending
the budget. Normal QoS limits each allocation to 24h.

Controller: `scripts/prepare_arm_gate_to60.py`; submitted template:
`scripts/resume_arm_gate_to60_8gpu.sbatch`. The old four-GPU proposal is superseded.
Plans, approval, preflight reports and submission receipts are under
`arm-turn-bonus-preparation/bc-to60-20260921/`, especially
`approval-8gpu.json`, `readiness-8gpu.json`, and `submissions-8gpu.json`.
The persistent CPU supervisor was refreshed at 16:26 PDT (PID703230), with
one-minute status checks and 15-minute full checks, and follows the active
training/evaluation stage. No further allocation is automatically authorized.

<a id="arm-failure-coverage-result-315204"></a>
### Failure-coverage pilot 315204 — zero eligible groups, September 21

The four-H200 pilot completed successfully after **33m29s**, releasing its
allocation early. GPU restoration and collection completed; it retained
**48 ordinary mixed groups**, with **zero optimizer updates**. Its report found
**zero eligible five-valid-failure groups**: 16 zero-outcome groups were rejected
first for invalid termination and six first for removed/excluded rows.
Consequently both the historical admission pool and raw-valid pool were empty;
the four-turn comparison selected zero states and issued zero new actor or
selector requests. This does not establish a coverage benefit or failure of the
idea, and it does not exercise deferred labeling on a nonempty reservoir.

A bounded CPU audit of the 106 JSON group journals found 48 accepted mixed groups,
31 all-success groups, 22 all-zero groups and five groups with insufficient reward
information. The all-zero groups contain 110 trajectories: 46 completed, eight
failed, 38 truncated and 18 aborted. Fourteen groups have removed or untrainable
rows. These counts can overlap with the first-rejection categories above; a
completed status alone also does not establish valid native judge provenance.
The immediate bottleneck in this collection was group validity before ARM
label coverage. Audit the termination causes before budgeting a larger pilot;
do not admit technical failures as policy failures just to increase yield.

Evidence: `evaluations/arm-failure-coverage-pilot-315204/iterations/0090/`:
`failure-coverage.json`, `failure_auxiliary.json`, and `groups/90/*.json`.
The original coverage and beta-weight training ablations still need concrete
allocation budgets; this zero-yield pilot does not justify launching them.

<a id="arm-allfailure100-eval-316392"></a>
### All-failure iteration100 full300 — submitted September 21, 17:06 PDT

Job **316392** requests the approved **2 H200 × 1 hour**, 16 CPUs and 480 GiB
RAM; queued for priority at submission. It evaluates the verified completed
checkpoint `arm-turn-bonus-fresh-allfailure-309490/runtime/iter_0000099`
(**100 collections / 1,242 Adam updates**) on all 300 OM2W tasks. Use the same
local browser, GPT-4.1/action_history judge and temperature0 as the other RL
evaluations. The established evaluation source and lossless saver passed
preflight; no training or recipe change is included.

The batch controller owns and awaits the evaluation worker. All task rollouts
and per-task verdicts are required for successful finalization. Output:
`evaluations/arm-allfailure-iter100-316392/`; W&B project `openwebrl-evals`.
Approval, checkpoint plan and submission receipts:
`arm-turn-bonus-preparation/iteration100-evals/allfailure-{approval,plan,submission}.json`.
The persistent CPU supervisor includes316392, with one-minute status checks
and 15-minute detailed checks. Original iteration60 remains a separate,
unsubmitted backfill; this approval covers only all-failure100.

September 21 completion update: **316392 finished in30m58s**, exit0, and released
its GPUs. All300 verdicts and rollout archives passed cohort/completeness checks;
**35.67% overall / 48.20% valid-only**. [Result audit](RL_EVALUATION.md#arm-allfailure-iter100-results-20260921).

<a id="arm-failure-ablations-ready-20260921"></a>
### Beta and coverage continuations — submitted September 22 UTC

User-approved jobs **317100 (failure beta1)** and **317101 (failure sampling40%)**
are queued. Each owns **8 H200 ×16h,64 CPUs,960GiB**,64 training browsers,
training additive100/Adam1262 →120, then full300 GPT-4.1/T0 evaluation;
combined cap **256 GPU-hours**. The old third unchanged-control proposal was
not submitted. B/C jobs316247/316248 remain separate, each8 H200 ×24h to60
with full300 at40/60; both queued at this check.

[Exact recipes, loss scaling and tests](ARM_INTEGRATION_PLAN.md#arm-failure-termination-audit-20260921).
Coverage now uses nested Bernoulli q40%, **not a fixed four-turn cap**, and
preserves historical q20% group admission. Mixed beta0.5/q20% are unchanged.
Source: `reference-arm-failure-ablations-20260922-v3`;31 working-tree and31
frozen-source tests plus both native TP8/scheduler checks pass. GPU restoration
runs before training. Empty eligible pools are valid zero-auxiliary collections;
nonempty pools must complete deferred labels before the optimizer proceeds.

Plans, fingerprints, native reports, exact user approval and submitted commands:
`arm-turn-bonus-preparation/failure-ablations-after100-20260921/`.
The controller awaits train→eval, requires exact checkpoint120, preserves
partial progress and reserves one hour for evaluation. Its batch template is
`scripts/run_arm_failure_ablations_8gpu.sbatch`. Controller roots:
`evaluations/arm-failure-ablation-weight-317100` and
`evaluations/arm-failure-ablation-coverage-317101`; training roots use
`evaluations/arm-failure-additive-JOB_ID`.

The CPU supervisor now tracks both jobs, with60-second status discovery and
900-second detailed health checks. Each training worker also owns a15-minute
monitor. W&B training identities are `arm-failure-weight-after100-317100` and
`arm-failure-coverage-after100-317101`, both in `openwebrl`; endpoint evaluations
use `openwebrl-evals`. W&B points will appear only after the queued jobs start.

[Failure-turn and sampling plots across training](ARM_INTEGRATION_PLAN.md#arm-failure-sampling-history-20260922).

<a id="storage-inventory-20260922"></a>
### Storage quota, retention correction and cleanup inventory — September 22

The five queued continuations started overnight and failed: baseline315098
(54s), B316247 (6m17s), C316248 (36s), failure-weight317100 (50s), and
failure-coverage317101 (69s). Baseline/B logs explicitly report `Disk quota
exceeded`; the remaining startup logs are truncated. No new optimizer update
was verified. B passed eight-GPU checkpoint restoration before collection
failed. At the September22 morning audit no user Slurm jobs were active.
The personal **scrubbed hard quota was110TiB**, despite substantial filesystem
free space. The previous filesystem-space/write-probe check did not establish
personal quota headroom. The supervisor recorded failures but did not recover
them automatically; monitoring is not evidence that a run is healthy.

The user-authorized non-tenth checkpoint pruning removed375 real checkpoint
directories and18 symlinks, reclaiming **21.14TiB**. One-based iterations
10,20,30,…,100 were retained where present (`iter_0000009` means iteration10).
The operation completed at16:07:48UTC before the correction to also retain each
lineage's latest checkpoint arrived. **Original-bonus iteration85 was deleted;
iteration80 remains.** No accessible replacement copy was found in the bounded
backup search. Historical training metrics were preserved, but they must not be
treated as the optimizer counters of that older retained checkpoint.

The corrected permanent rule is **every tenth iteration plus the latest durable
checkpoint of each lineage**, with active resume/evaluation dependencies checked
before any future deletion. Baseline90, B20, C20, additive100 and all-failure100
remain available. Saved trajectories and judge verdicts were not deleted.
The post-cleanup quota query at approximately09:14PDT reports **88.86TiB used**,
with **11.14TiB below the100TiB soft quota** and **21.14TiB below the110TiB hard
quota**. These are shared personal limits, not independently available per job.

The following inventory identifies candidates; **no additional deletion has
been performed or authorized**. Sizes use allocated bytes and TiB (1024^4 bytes).
The first four categories below are disjoint; aliases/hard links were counted
once within the OpenWebRL payload scan.

| Data | Size | Location relative to runtime unless stated | Recommendation |
|---|---:|---|---|
| Completed-job temporary image-tensor mappings | 7.424TiB;4,946 files | `multimodal-scratch/arm-variant-{311964,311965,311962,313208,313210,313669}/rollout-*.bin` | First deletion candidate. All six jobs completed; native recovery files serialize tensor values independently. Preserve the surrounding directories and all other files. |
| ARM native recovery batches | 16.256TiB;331 files | ARM training roots under `evaluations/*/runtime/rollout_recovery/*.pt` | Selective pruning only after checking replay dependencies and retained source archives. Total is inventory, not approved/reclaimable space. |
| Baseline native recovery batches | 5.102TiB;106 files | `runs/openwebrl-4b-reference-*/rollout_recovery/*.pt` | Same dependency review. Preserve pending next-batch recovery and data needed for restart. |
| ARM per-group training archives | 31.385TiB;32,689 files | ARM training roots under `evaluations/*/iterations/*/groups/**/*.pt` | Preserve by default. Contains accepted and rejected groups, including data not in native optimizer batches; useful for audits, relabeling and future training. |
| Auxiliary failure tensors | 0.142TiB | `evaluations/*/iterations/*/failure_auxiliary.pt` | Lower priority; inspect dependencies before pruning. |
| Cooking full-suite processed image tensors | 4.989TiB;71,241 files | Under **checkpoints/web**, `full-suite-eval-20260920/cooking/{update-0,update-3-best,update-12}/episodes/*/visual_tensors/*.pt` | Promising separate-project candidate. Sampled decision logs retain original image data URIs; complete image coverage and preprocessing/replay dependencies still need verification before deletion. |

The entire `checkpoints/web` tree is about19.0TiB, mostly other cooking/ScreenSim
experiments. Large parents include `full-suite-eval-20260920` (~5.1TiB),
`cooking-full-295856` (~3.9TiB), `cooking-gemini-novice-rl-306481` (~2.5TiB),
and `cooking-gemini-novice-rl-308401` (~2.1TiB). These parent totals overlap
their contents above: **do not sum them as additional deletion candidates**.
In particular, `cooking-full-295856` is mostly episode data (~3.5TiB), with only
~401GiB in its `checkpoints/` subdirectory. Directory names alone do not identify
disposable model checkpoints. OpenWebRL SFT artifacts total only~196GiB and are
not the main storage pressure.

Transport implementation evidence: `slime/utils/rollout_transport.py` retains
`rollout-*.bin` for a process lifetime; page-cache eviction does not unlink them.
`file_back_completed_group` and native `torch.save` preserve independent tensor
values in durable recovery files. The exact4,946-file proposed cleanup manifest
records each path, inode, size and modification time. Durable audit files are in
`arm-turn-bonus-preparation/overnight-20260922/`: `checkpoint-pruning-manifest.json`,
`checkpoint-pruning-deletions.jsonl`, `temporary-tensors-candidates.json`,
`storage-payload-inventory.json`, and `other-project-visual-tensors.json`.

Prepared launchers now check **personal quota**, conservatively requiring2TiB
below the soft limit before GPU restoration, and fail closed if quota cannot be
read. This is a startup guard, not a reservation or guarantee that several long
jobs fit concurrently. The approved24h baseline and failure-ablation limits are
prepared; replacement submissions and their remaining budgets must be recorded
separately from the failed job IDs. Scientific sources and optimizer settings
remain unchanged. Quota parsing tests and the refreshed ablation CPU/native
scheduler checks pass; new GPU execution has not yet been validated.

<a id="training-relaunch-20260922"></a>
### Five approved training continuations resubmitted — September22

**Superseded for the two new interventions:**318936/318937 were canceled while
pending after the user clarified that both experiments must begin at iteration0.
See [the corrected jobs318949/318950](#failure-ablations-fromzero-20260922).
Baseline/B/C318933–318935 retain their existing continuation plans.

Following the explicit request to launch the failed training jobs and the new
ablations, all five replacements were submitted and verified in Slurm. At the
morning check they are **pending for priority**, not training yet. They use the
approved normal QoS on `gpu-h200`; the failed jobs' elapsed runtimes were deducted
and remaining walltime rounded down to whole minutes, keeping each lineage
within its approved24h allocation budget including the failed attempt.

| Replacement job | Replaces | Experiment | Resume → target | H200 GPUs | Walltime | Full300 evaluations in the same allocation |
|---|---|---|---|---:|---|---|
| 318933 | 315098 | Outcome-only baseline | 90 →100 | 4 | 23h59m | 100 |
| 318934 | 316247 | B: relaxed candidate gate | 20 →60 | 8 | 23h53m | 40,60 |
| 318935 | 316248 | C: relaxed gate + duplicate-aware credit | 20 →60 | 8 | 23h59m | 40,60 |
| 318936 | 317100 | Additive failure weight beta1 | 100 →120 | 8 | 23h59m | 120 |
| 318937 | 317101 | Additive failure sampling q40% | 100 →120 | 8 | 23h58m | 120 |

Baseline requests32 CPUs/480GiB; the other jobs each request64 CPUs/960GiB.
All native resume metadata, source hashes, optimizer counters and prepared
ablation fingerprints passed before submission. GPU restoration is still
required inside each allocation. Baseline resumes Adam1016; B/C resume284;
the two new ablations branch from additive100/Adam1262. Existing baseline/B/C
W&B identities are preserved; the two new ablations use separate training IDs
`arm-failure-weight-after100-318936` and
`arm-failure-coverage-after100-318937` in `openwebrl`.

The persistent CPU supervisor was restarted as PID1083186 for72h and verified
to include all five jobs. It checks status every60s and detailed health plus
personal quota every900s. Quota alerts fire below4TiB of soft-limit headroom or
2TiB of hard-limit headroom; these observations do not reserve capacity and do
not authorize deletion. Newly registered jobs are now loaded without restarting
the supervisor. Eight supervisor tests pass. The morning quota check passes with
about21.1TiB below the hard limit; additional temporary-tensor cleanup remains
unapproved. No saved rollout or judge verdict was removed for these launches.

Exact approval, submitted commands, per-job scheduler/resource verification and
monitor receipt are under
`arm-turn-bonus-preparation/overnight-20260922/relaunch/`.
Each controller owns and awaits its workers and evaluations, preserves partial
progress if the target does not fit, and does not obtain another allocation.

<a id="failure-ablations-fromzero-20260922"></a>
### Beta and sampling experiments corrected to iteration0 — September22

The user clarified that **every new experiment or intervention starts at
iteration0** for clean comparisons. This is now a permanent root`AGENTS.md`
instruction. Existing unchanged experiments may continue from their own saved
state; a new reward/loss/sampling/data intervention may not inherit a later RL
checkpoint without explicit user direction.

Jobs318936/318937 were canceled while pending, with zero elapsed GPU time.
Their replacements use the same previously approved resources and remaining
walltime; there is no compute-budget extension. Both were verified **pending
for priority** at09:31PDT.

| Job | New experiment | Initial model / state | Target | H200 GPUs | Walltime | Included evaluation |
|---|---|---|---:|---:|---|---|
| 318949 | Failure beta1, q20% | Original OpenWebRL-4B-SFT; rollout0; Adam0; fresh scheduler/cursor | 20 | 8 | 23h59m | Full300 at20 |
| 318950 | Failure q40%, beta0.5 | Original OpenWebRL-4B-SFT; rollout0; Adam0; fresh scheduler/cursor | 20 | 8 | 23h58m | Full300 at20 |

Mixed-outcome beta0.5/q20%,48 ordinary groups, up to8 eligible auxiliary groups,
distinct5 gate and response-index credit remain unchanged. Each job has64 CPUs,
960GiB RAM and64 training browsers. Native launch checks confirm both `load` and
`hf_checkpoint` resolve to the original SFT model, start rollout0, target20 and
no restored optimizer scheduler. Initialization guards reject resume/replay
state or a nonzero optimizer offset.25 focused tests pass both locally and
against the frozen source. GPU initialization remains pending inside allocation.

Training W&B identities are `arm-failure-weight-fromzero-318949` and
`arm-failure-coverage-fromzero-318950`, in `openwebrl`. Full300 GPT-4.1/T0
evaluation retains rollout archives and per-task verdicts in `openwebrl-evals`.
The controllers reserve one hour for evaluation and require exact checkpoint20;
they preserve partial progress if20 does not fit and do not extend their budgets.
The persistent supervisor includes both replacements and continues to monitor
baseline318933 and B/C318934–318935.

Plans, native argument reports,25-test readiness fingerprints, approval,
submission and scheduler verification receipts:
`arm-turn-bonus-preparation/failure-ablations-fromzero-20260922/`.
The [integration plan](ARM_INTEGRATION_PLAN.md#arm-failure-termination-audit-20260921)
now describes this corrected design; the earlier after100 design is superseded.
