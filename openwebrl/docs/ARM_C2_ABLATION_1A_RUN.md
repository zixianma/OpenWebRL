# C2 ablation 1A run

Status: **training and primary validation complete**. The user explicitly
approved one H200 for five hours for 1A training followed by fixed 100-task
validation. Slurm job `285567` ran on `g012` from 2026-09-09 17:03 to 21:00
PDT, using 3:57:14 of its five-hour limit. See the
[result report](ARM_C2_ABLATION_1A_RESULTS.md).

## Frozen training configuration

| Setting | Value |
| --- | --- |
| Starting model | `OpenWebRL/OpenWebRL-4B-SFT` |
| Dataset | 8,394 C2 turns; SHA-256 `cb7c75df6a4824e9e653f6d913b0ae83268610966cd13dd13fc7314e9c667fe0` |
| LoRA | language-only rank 16, alpha 32, dropout 0.05; 33,030,144 trainable parameters |
| Optimizer | AdamW, peak LR `1e-5`, betas 0.9/0.95, epsilon `1e-8`, weight decay 0.01 |
| Batch | microbatch 1, accumulation 32, effective batch 32 |
| Duration | one pass, 263 optimizer updates |
| Schedule | example-indexed: 512-example warmup, cosine horizon 16,788 examples; endpoint LR about `5.25e-6` |
| Checkpoints | updates 66, 132, 198, 250, and the predeclared epoch-1/update-263 endpoint |
| Evaluation | fixed 100-task cohort, one-action actor, seed 42, temperature 0.7, top-p 0.9, 30-turn horizon, `o4-mini` AgentTrek judge |

The endpoint is fixed before evaluation. Intermediate checkpoints are saved for
training diagnostics and are not used to select the 1A result.

After training, the same allocation also evaluated update 250 on the frozen
100-task cohort. This is the exposure-matched comparison to the original C2
update 500: `250 * 32 = 500 * 16 = 8,000` training examples. The endpoint and
update-250 evaluations used isolated actor/browser ports and shared the H200 at
40% and 30% static memory, respectively, following the already validated
two-checkpoint execution used by the original C2 scaling sweep. The endpoint
remains the primary predeclared 1A result; update 250 is a learning-curve
diagnostic and is not a selection criterion.

The schedule did not save update 50, which would have exactly matched original
C2 update 100 at 1,600 examples (`50 * 32 = 100 * 16`). W&B has its loss but
not its weights, so it cannot be evaluated without replaying training. The
earliest available checkpoint is update 66 at 2,112 examples, 32% more exposure
than the old update-100 point. It was queued only for capacity left after
either primary evaluation, but did not start before the batch controller
exited. Future effective-batch-32 ablations should save updates 50 and 250.

## Implementation and artifacts

The trainer is on the local ARM branch `openwebrl/c2-filtered-sft`, commit
`f3496dec08d416348900614e0183f2fa9fca08e6`. It adds the example-indexed
schedule while leaving the original schedule available for prior runs. The
allocation controller and handoff are in OpenWebRL commit `f95daf8`.

Runtime root:

```text
/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-ablation-1a
```

Key files are `frozen-config.json`, `status.json`, `training.log`,
`student/metrics.jsonl`, periodic checkpoints under `student/`, and evaluation
outputs under `evaluation/checkpoint-scaling-100/endpoint-000263/`. The
controller reserves the final 70 minutes for merge and evaluation, starts that
handoff immediately after training completes, and preserves a resumable paused
checkpoint if training cannot finish before the reserve boundary.

The W&B sidecar uses run name `arm-c2-ablation-1a`, group
`arm-c2-ablation`, and tag `ablation-1a`: [W&B run
9ac0cb8a](https://wandb.ai/zixianma/openwebrl-arm/runs/9ac0cb8a). Its durable local
identity and resume cursor are in `student/wandb-sync.json`.

### W&B project migration

The sidecar remained in `zixianma/openwebrl` through completion so that its
live stream and resume identity were not interrupted. After completion, W&B
move task `VGFzazoyNzc1Nzg2NjYw` moved runs `9ac0cb8a` (1A) and `57f0384c`
(original C2) to the dedicated `zixianma/openwebrl-arm` project. Both new paths
were verified and both old paths are absent. Future ARM launchers set
`--project openwebrl-arm` at initialization.
This migration concerns W&B organization only; the durable local run roots and
evaluation artifacts remain at their recorded GPFS paths.

## Initial health

The first three updates completed with finite cross-entropies 0.1703, 0.1598,
and 0.1886 and finite gradient norms 0.178, 0.158, and 0.177. Learning rates
were `6.25e-7`, `1.25e-6`, and `1.875e-6`, matching the 512-example warmup.
Early update cadence projects about 3.2–3.5 hours for training, leaving roughly
1.5 hours for the evaluation handoff. These are startup measurements, not a
final throughput estimate.

## Final execution note

Training completed all 263 updates at 19:58 PDT and the primary endpoint
evaluation completed at 21:00 PDT. The external update-250 worker preserved
97/100 outcomes before Slurm canceled it. The update-66 worker did not start.
The cause was orchestration: the batch controller exited after its endpoint
evaluation and Slurm canceled external overlapping steps, so 1:02:46 of the
approved limit went unused. External `srun` workers must not be relied on to
keep a batch allocation alive; future controllers must own and await their full
evaluation queue before exiting.
