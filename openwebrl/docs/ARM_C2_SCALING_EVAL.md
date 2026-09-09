# C2 checkpoint scaling evaluation

Status: update 100 is evaluating on allocation **285131** on g022; updates 500,
700, and 923 are queued to start sequentially as soon as that GPU becomes free.
The first durable task record was produced at 12:03 PDT on 2026-09-09. The first
task was unavailable because United.com returned an HTTP/2 navigation error; it
will count in the overall denominator and be excluded from the valid-only rate.

The earlier user-assigned allocation **285189** on g002 provided two H200s, 8
CPUs, 240 GiB RAM, and a four-hour limit. It was canceled by the user's UID at
11:49:03 PDT on 2026-09-09 after 8m41s. The only completed step was a one-second
GPU availability check; both H200s were free. Slurm rejected the first worker
after the allocation ended.

The user subsequently directed the scaling evaluation to allocation **285131**
on g022 and stopped training at update 923 to prioritize evaluation. The
single-H200 queue evaluates updates 100, 500, 700, and 923 sequentially. Each
checkpoint evaluation is resumable. No primary epoch-2 evaluation is possible
because the fixed epoch-2 checkpoint was not produced.

The evaluated states are the C2 adapters at optimizer updates **100, 500, 700,
and 923**. Update 1050 was the last planned update because the immutable dataset
has 8394 turns and the recipe has effective batch 16 for two epochs:
`2 * ceil(8394 / 16) = 1050`. At the user's request, training stopped after the
loss plateaued. The endpoint is the durable `paused-000923-1788980015`
checkpoint at epoch-2 position 6368/8394; it is an intentionally truncated run,
not a completed epoch-2 result.

## Frozen task sample

Before reading any checkpoint result, Python `random.Random(20260909)` sampled
100 positions without replacement from the 300-task Online-Mind2Web file. The
indices were sorted only for storage and dispatch. The complete indices, task
IDs, task-file checksum, creation time, and contemporaneous baseline reference
are committed in [arm_c2_scaling_100.json](arm_c2_scaling_100.json). The same
bytes are stored with the runtime outputs as `checkpoint-scaling-100/sample.json`.
Its SHA-256 is
`2b55f26b0a02296d4488b801bffc0806ce4830cf47350ef7917de887f424390d`.

The already completed base-policy evaluation scored **26/100 overall (26.0%)**
and **26/85 valid (30.6%)** on this sample; 15 outcomes were unavailable. This
historical base reference is useful but remains time-confounded against the new
live-site checkpoint runs.

## Execution and interpretation

One worker uses the single assigned H200. It evaluates updates 100, 500, 700,
and 923 in sequence. Each adapter is safe-merged independently, then evaluated as a one-sample
policy with the same task file, seed 42, temperature 0.7, top-p 0.9, 1024-token
response limit, 30-turn horizon, full history, one current screenshot,
concurrency 8, and o4-mini AgentTrek terminal judge used by the main comparison.
Each output has an immutable manifest and durable per-task results.

Report both overall and valid-only success. Also compare trajectory length,
termination rate, 30-step-cap rate, scroll-call share, and repeated-primary-action
rate. The 100-task curve is diagnostic: confidence intervals are wide, repeated
benchmark use limits confirmatory claims, and concurrent waves reduce but do not
remove live-site time drift. Checkpoint 923 is the stopped-run endpoint.

Runtime root:
`/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-full-282782-20260908T075414Z/evaluation/checkpoint-scaling-100`.

## Results in progress

| Checkpoint | Success / scheduled | Overall | Success / valid | Valid-only | Unavailable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original base actor (historical) | 26/100 | 26.0% | 26/85 | 30.6% | 15 |
| Update 100 | 28/100 | 28.0% | 28/84 | 33.3% | 16 |

Update 100 completed at 12:55 PDT. Its behavior is close to the base actor on
the fixed cohort: mean/median trajectory length 17.66/17 versus 17.18/14,
termination 50.0% versus 52.8%, 30-step caps 41.1% versus 41.6%, scroll-call
share 11.4% versus 13.7%, and repeated-primary-action rate 59.0% versus 62.0%.
This first checkpoint does not show the sharp trajectory-length/termination/
scroll collapse reported by the upstream failed actor-distillation experiment.
The small success difference is diagnostic only.

Checkpoints 500, 700, and 923 were safe-merged in advance while update 100 was
evaluating. Update 500 began at 12:55 PDT; the queue will reuse the validated
merged artifacts to avoid transition delay. The machine-readable success and
behavior comparison lives under the runtime root as `behavior-comparison.json`.
