# H200 runtime and validation

## Collection 5 crash and recovery preparation, 2026-09-07 23:11 PDT

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

## Previous continuation milestone, 2026-09-07 22:24 PDT

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

## Runtime

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

## Completed checks, 2026-09-07 UTC

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

## Reproduce tests

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

## Real-web baseline launch

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

## 2026-09-07: reference baseline checkpoint recovery

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

### Final recovery result

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

## 2026-09-07: continuation on g005

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

### Recovery changes preserved in Git

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

### Explicit tracking shutdown

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
