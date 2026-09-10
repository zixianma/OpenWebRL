# C2 ablation 1A run

Status: **running**. The user explicitly approved one H200 for five hours for
1A training followed by the fixed 100-task validation. Slurm job `285567`
started on `g012` at 2026-09-09 17:03 PDT and ends at 22:03 PDT. It has one
H200, four CPUs, and 120 GB host memory: a maximum budget of five H200-hours.

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
9ac0cb8a](https://wandb.ai/zixianma/openwebrl/runs/9ac0cb8a). Its durable local
identity and resume cursor are in `student/wandb-sync.json`.

## Initial health

The first three updates completed with finite cross-entropies 0.1703, 0.1598,
and 0.1886 and finite gradient norms 0.178, 0.158, and 0.177. Learning rates
were `6.25e-7`, `1.25e-6`, and `1.875e-6`, matching the 512-example warmup.
Early update cadence projects about 3.2–3.5 hours for training, leaving roughly
1.5 hours for the evaluation handoff. These are startup measurements, not a
final throughput estimate.
