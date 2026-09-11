# Calibrated ARM preference run 286384

Status: **training and fixed-100 evaluation are complete**.
Two H200s in the
user-provided four-hour allocation `286384` on `g001` completed all 131
updates. The handoff merged the endpoint, but its SGLang server failed during
startup because the evaluation environment did not expose `CUDA_HOME`; the
allocation then expired. The evaluation launcher was repaired, and the user
approved a separate one-H200, one-hour rerun. Job `286831` completed all 100
tasks on `g005` in about 25 minutes at rollout concurrency 12.

## Frozen experiment

| Setting | Value |
| --- | --- |
| Starting actor and frozen reference | `OpenWebRL/OpenWebRL-4B-SFT` |
| Source | immutable 8,394-turn C2 dataset |
| Pair coverage | 8,363/8,394 (99.63%) have a distinct valid loser |
| Training subset | deterministic seed-42 prefix of 4,192 shuffled pairs |
| Loser | seeded distinct, parseable, non-truncated candidate from the same pre-action state |
| Objective | winner CE + length-normalized reference-relative DPO |
| Beta | 0.1 |
| Preference weight | **3.02186**, calibrated on four frozen training pairs |
| Calibration target | preference gradient norm = 20% of winner-SFT gradient norm at initialization |
| LoRA | language-only rank 16, alpha 32, dropout 0.05 |
| Batch | 16 pairs/rank, two DDP ranks, effective batch 32 pairs |
| Updates | 131; checkpoints at 33, 66, 99, and endpoint |
| LR | peak `1e-5`; 512 response-exposure warmup, then the 1A half-cosine schedule |
| Evaluation | existing fixed 100 OM2W tasks, one actor action, `o4-mini` AgentTrek judge |

The calibration measured SFT gradient norm 0.32303 and unit-weight preference
gradient norm 0.02138. Their ratio selected lambda 3.02186 for the declared
20% contribution.

## Runtime artifacts

```text
/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/arm-preference-calibrated-286384-r4
```

Key files are `frozen-config.json`, `pair-audit.json`,
`student/calibration.json`, `student/metrics.jsonl`, checkpoints under
`student/`, and evaluation output under
`evaluation/checkpoint-scaling-100/endpoint-000131/`.

W&B: [arm-preference-calibrated-286384](https://wandb.ai/zixianma/openwebrl-arm/runs/fd55eff3)

## Fixed-100 evaluation

| Policy | Overall | Valid-only | Unavailable |
| --- | ---: | ---: | ---: |
| Historical starting actor | 26/100 = 26.0% | 26/85 = 30.6% | 15 |
| C2 update 500 | 33/100 = 33.0% | 33/79 = 41.8% | 21 |
| 1A endpoint | 33/100 = 33.0% | 33/79 = 41.8% | 21 |
| **Preference endpoint 131** | **31/100 = 31.0%** | **31/86 = 36.0%** | **14** |

On common-valid tasks, the preference endpoint had 12 wins and 6 losses
against the historical base (81 tasks, exact McNemar p=0.238), 3 wins and 8
losses against C2 (76 tasks, p=0.227), and 6 wins and 10 losses against 1A (75
tasks, p=0.454). None establishes a difference. The preference policy is
directionally above the base and below both filtered-SFT controls.

| Policy | Mean / median steps | Terminated | Hit 30 steps | Scroll calls | Repeated primary action |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 17.18 / 14 | 52.8% | 41.6% | 13.7% | 62.0% |
| C2 update 500 | 14.24 / 10 | 65.1% | 27.9% | 7.9% | 59.4% |
| 1A endpoint | 15.67 / 12 | 55.7% | 33.0% | 10.0% | 59.0% |
| Preference endpoint | 16.84 / 12 | 59.8% | 37.9% | 12.1% | 61.2% |

The preference endpoint's behavior sits closer to the starting actor than C2
on trajectory length, step caps, scrolling, and repeated actions. Combined
with the flat preference margin, this does not justify an all-300 evaluation.

## Completed training-curve diagnosis

The displayed preference loss increased from a mean of 0.69182 over updates
1–33 to 0.69288 over updates 99–131. Most of this visible movement is BF16
quantization. Reconstructing the loss in float from the logged relative margins
gives 0.693152 and 0.693193, respectively: an increase of only 0.000041. The
mean policy-minus-reference preference margin over the last 32 updates was
-0.00091 nats, so the run did not demonstrate generalizing winner/loser
separation even though winner CE fell.

These are losses on a different, previously unseen pair batch at every update,
measured before that batch's optimizer step. They are not a fixed validation
curve and therefore need not decrease monotonically. The run omitted the fixed
pair panel needed to distinguish within-pair learning from generalization.

The post-run action audit found a material data bug: 1,099/4,192 training pairs
(26.22%) used a rejected response with the **same normalized executable action**
as the winner, differing only in reasoning text. Two of the four calibration
pairs had identical executable actions; a third differed only by one click
coordinate. Consequently lambda 3.02186 was numerically calibrated but not
well informed about distinct action preferences. Whole-response mean log
probabilities also dilute the approximately 100-character action in responses
with median length around 306 tokens.

The next preference run must canonicalize and exclude action-equivalent pairs,
calibrate on a stratified sample, compute/log margins in float32, and log a
fixed held-out preference panel before using task-level evaluation.

## Engineering record

Two startup attempts found memory problems before update 3 and remain in
separate audit directories. The final implementation passes only response
positions to Qwen3-VL's language-model head. This preserves the exact causal
response log probabilities while avoiding prompt-length-by-vocabulary logits.
Peak observed memory fell from 142.6 GB to about 39.5 GB per H200. Failed
startup artifacts are not training results and are excluded from evaluation.
