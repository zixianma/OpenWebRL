# C2 checkpoint scaling evaluation

Status: prepared for the user-assigned allocation **285189** on g002, which has
two H200s, 8 CPUs, 240 GiB RAM, and a four-hour limit ending at 15:40 PDT on
2026-09-09. This controller uses only that existing allocation.

The evaluated states are the C2 adapters at optimizer updates **100, 500, 900,
and 1050**. Update 1050 is the last planned update because the immutable dataset
has 8394 turns and the recipe has effective batch 16 for two epochs:
`2 * ceil(8394 / 16) = 1050`. The fixed epoch-2 checkpoint represents update
1050.

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

Two workers use one H200 each, separate actor ports, and disjoint browser-port
ranges. The first wave is updates 100 and 500; the second is updates 900 and
1050. Each adapter is safe-merged independently, then evaluated as a one-sample
policy with the same task file, seed 42, temperature 0.7, top-p 0.9, 1024-token
response limit, 30-turn horizon, full history, one current screenshot,
concurrency 8, and o4-mini AgentTrek terminal judge used by the main comparison.
Each output has an immutable manifest and durable per-task results.

Report both overall and valid-only success. Also compare trajectory length,
termination rate, 30-step-cap rate, scroll-call share, and repeated-primary-action
rate. The 100-task curve is diagnostic: confidence intervals are wide, repeated
benchmark use limits confirmatory claims, and concurrent waves reduce but do not
remove live-site time drift. The fixed 300-task epoch-2 evaluation remains the
primary C2 result.

Runtime root:
`/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-full-282782-20260908T075414Z/evaluation/checkpoint-scaling-100`.
