# Baseline timing and four-GPU continuation

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

## Checkpoint portability

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

The inner `run_h200_browser.sh` already accepts `NUM_GPUS` and `TP_SIZE`, but
`run_small_baseline.py` and the quick-resume wrapper currently force two GPUs.
They must be parameterized and their allocation checks and recorded metadata
updated before a four-GPU launch. Setting only environment variables does not
currently switch the convenience workflow to four GPUs.

## Expected speedup and migration checks

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
