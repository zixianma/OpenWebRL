# Resuming C2 student training

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
