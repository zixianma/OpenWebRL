# OpenWebRL project documentation

Documents are organized by topic. Start with the results or operational guide
for the work you are doing, then follow its contents to dated experiment records.

| Document | Contents |
| --- | --- |
| [ARM_SUMMARY.md](ARM_SUMMARY.md) | Concise methods and headline results across inference, offline training and RL |
| [ARM_RESULTS.md](ARM_RESULTS.md) | All ARM benchmark results, C2/1A/joint comparisons, checkpoint scaling, and curves |
| [ARM_INTEGRATION_PLAN.md](ARM_INTEGRATION_PLAN.md) | ARM model understanding, literature, integration options, and longer-term roadmap |
| [ARM_INFERENCE.md](ARM_INFERENCE.md) | Inference protocol, judge alignment, and retry policy |
| [ARM_SFT.md](ARM_SFT.md) | C2 collection, filtering, SFT, 1A configuration, and checkpoint procedures |
| [ARM_PREFERENCE.md](ARM_PREFERENCE.md) | Preference experiments, pair audits, and viability plans |
| [ARM_JOINT_DATA.md](ARM_JOINT_DATA.md) | Combined C2/Piotr data review, examples, training, and recovery |
| [RL_RESULTS.md](RL_RESULTS.md) | Tables-only RL checkpoint scores; overall, valid-only, invalid counts, and separate protocols |
| [RL_RUNTIME.md](RL_RUNTIME.md) | Reference baseline resume, GPU scaling, archives, and runtime history |
| [RL_EVALUATION.md](RL_EVALUATION.md) | Baseline and Browser Use evaluations and reward-ranked scheduling |
| [RL_METRICS.md](RL_METRICS.md) | Metric definitions, paper reward comparisons, and gradient diagnostics |
| [RL_EXPERIMENTS.md](RL_EXPERIMENTS.md) | Prepared RL recipe experiments and validation |

Current scaling plan: [eight-GPU topology and browser benchmark](RL_RUNTIME.md#eight-gpu-scaling-benchmark-20260912).

Live operations: [job status, completions and failures](arm_results/rl_integration/live-status.html)
(one-minute refresh; [monitor behavior](RL_RUNTIME.md#arm-live-monitor-20260921)).

Prepared baseline continuation: [stage-2 recipe, dynamic sampling audit, and launch](RL_RUNTIME.md#baseline-stage2-20260914).

Latest requested comparison: [outcome-only and additive stage-1 continuation to iteration 100](RL_RUNTIME.md#stage1-baseline-additive-to100-20260921), with a full-300 evaluation inside each allocation.
Additive100 is [complete at 36.33% overall / 50.23% valid-only](RL_EVALUATION.md#arm-additive-iter100-results-20260921);
baseline100's queued continuation subsequently failed at startup from personal storage quota.
[September 22 quota recovery, checkpoint retention and storage inventory](RL_RUNTIME.md#storage-inventory-20260922).
[Training/evaluation inventory and remaining work](RL_RUNTIME.md#arm-job-inventory-20260921).

Latest baseline: [iteration-90 completion](RL_EVALUATION.md#scheduled-eval90-results-20260913),
[iteration-80 GPT-4.1 rejudging](RL_EVALUATION.md#stealth80-gpt41-rejudge-feasibility-20260913),
and [cluster/account queue audit](RL_RUNTIME.md#cluster-queue-audit-294983-20260913).

ARM RL: [saved termination audit and beta/coverage launch proposal](ARM_INTEGRATION_PLAN.md#arm-failure-termination-audit-20260921),
[all-failure100 results and updated curves](RL_EVALUATION.md#arm-allfailure-iter100-results-20260921),
[B/C20→60 eight-GPU launches and evaluation budget](RL_RUNTIME.md#arm-bc-to60-prepared-20260921),
[failure-coverage pilot: zero eligible groups](RL_RUNTIME.md#arm-failure-coverage-result-315204),
[agreed failure-coverage and failure-weight experiments; CPU preparation](ARM_INTEGRATION_PLAN.md#arm-additive-next-experiments-20260921),
[completed turn-bonus pilot](ARM_INTEGRATION_PLAN.md#arm-turn-bonus-pilot-completed),
[actor-stage selection-quality audit](ARM_INTEGRATION_PLAN.md#arm-selection-quality-audit-20260921),
[completed fixed-state quality results](ARM_RESULTS.md#arm-selection-quality-313774),
[ARM-only fixed100 terminal-success audit (314664)](ARM_INTEGRATION_PLAN.md#arm-task-success-audit-20260921),
[next-round discussion after iteration 70](ARM_INTEGRATION_PLAN.md#arm-rl-next-round-20260919),
[label coverage and candidate-confidence audit](ARM_INTEGRATION_PLAN.md#arm-label-coverage-confidence-audit-20260919),
[one-call gate allowing two or more actions](ARM_INTEGRATION_PLAN.md#arm-min2-gate-test-20260919),
[B/C iteration-zero launches](ARM_INTEGRATION_PLAN.md#arm-bc-fromzero-launch-20260919),
[B calibration stop and staged recovery](ARM_INTEGRATION_PLAN.md#arm-bc-calibration-recovery-20260920),
[overnight monitoring](RL_RUNTIME.md#arm-overnight-supervision-20260920),
[additive continuation to 100](RL_RUNTIME.md#arm-additive-to100-prepared-20260920),
[all-failure continuation to 100](ARM_INTEGRATION_PLAN.md#arm-allfailure-to100-prepared-20260919),
[iteration-80 full-300 evaluations](RL_EVALUATION.md#arm-iter80-launch-20260919),
[additive iteration-20 full-300 result](RL_EVALUATION.md#arm-additive-iter20-full300-20260920),
[ARM/Sol interactive GPU debugging](ARM_INTEGRATION_PLAN.md#arm-sol-interactive-debug-20260913),
[real ARM interactive training](ARM_INTEGRATION_PLAN.md#arm-turn-bonus-interactive-294197),
[eight-hour ARM continuation](ARM_INTEGRATION_PLAN.md#arm-bonus-eight-hour-continuation-20260913),
[eight-GPU ARM continuation](ARM_INTEGRATION_PLAN.md#arm-bonus-eight-gpu-continuation-20260914),
[failed-task rescue and next RL directions](ARM_INTEGRATION_PLAN.md#arm-rl-next-directions-20260913),
[prepared all-failure ARM experiment](ARM_INTEGRATION_PLAN.md#arm-all-failure-preparation-20260913),
and [exact objectives](ARM_INTEGRATION_PLAN.md#arm-rl-exact-losses-20260912).
Checkpoint benchmarking: [OpenWebRL protocols and prepared after-58 rerun](RL_EVALUATION.md#paper-om2w-protocol-20260912).

## Maintaining these documents

- Add experiments and results to the matching topic instead of creating another run-specific Markdown file.
- Keep full-300, held-out, partial, historical, and retry results labeled with their original denominators and judge protocol.
- Retain JSON manifests, frozen configs, audit data, and HTML galleries under their existing artifact directories. Their recorded hashes and historical paths are provenance and are not rewritten during documentation moves.
- Report writers use `scripts/project_docs.py` to update one marked section atomically. Preserve the `document:...:start/end` markers and explicit anchors.
- Historical filenames are identifiers in [document_map.json](document_map.json); use that mapping when following an old reference.
- Reproduce frozen runs from their recorded source revision. Documentation-writer changes do not silently update historical code hashes.

## Consolidation map

The 37 former top-level documents become 10 topic documents plus this index.
The two Markdown notes nested inside audit-artifact directories remain with
their datasets. `AGENTS.md` remains at the repository root.

**Migration status:** Complete. The 35 superseded source files below were
retired after their content and links were verified in the topic documents.
Their full text remains in marked sections, and their pre-migration SHA-256
hashes remain in `document_map.json`. Retained standalone files are
`ARM_INTEGRATION_PLAN.md` and `RL_EXPERIMENTS.md`.

| Retired source (formerly under `openwebrl/docs/`) | Preserved section |
| --- | --- |
| `ARM_C2_ABLATION_1A_RESULTS.md` | [ARM_RESULTS.md](ARM_RESULTS.md#arm-c2-ablation-1a-results) |
| `ARM_C2_ABLATION_1A_RUN.md` | [ARM_SFT.md](ARM_SFT.md#arm-c2-ablation-1a-run) |
| `ARM_C2_FULL300_EVAL.md` | [ARM_SFT.md](ARM_SFT.md#arm-c2-full300-eval) |
| `ARM_C2_RESUME.md` | [ARM_SFT.md](ARM_SFT.md#arm-c2-resume) |
| `ARM_C2_RUN.md` | [ARM_SFT.md](ARM_SFT.md#arm-c2-run) |
| `ARM_C2_SCALING_EVAL.md` | [ARM_SFT.md](ARM_SFT.md#arm-c2-scaling-eval) |
| `ARM_C2_SCALING_RESULTS.md` | [ARM_RESULTS.md](ARM_RESULTS.md#arm-c2-scaling-results) |
| `ARM_C2_VS_1A_FULL300_EVAL.md` | [ARM_RESULTS.md](ARM_RESULTS.md#arm-c2-vs-1a-full300-eval) |
| `ARM_FILTERED_SFT_ABLATIONS.md` | [ARM_SFT.md](ARM_SFT.md#arm-filtered-sft-ablations) |
| `ARM_FILTERED_SFT_PLAN.md` | [ARM_SFT.md](ARM_SFT.md#arm-filtered-sft-plan) |
| `ARM_INFERENCE_RESULTS.md` | [ARM_INFERENCE.md](ARM_INFERENCE.md#arm-inference-results) |
| `ARM_INFERENCE_RETRY_RESULTS.md` | [ARM_INFERENCE.md](ARM_INFERENCE.md#arm-inference-retry-results) |
| `ARM_JOINT_DATA_REVIEW.md` | [ARM_JOINT_DATA.md](ARM_JOINT_DATA.md#arm-joint-data-review) |
| `ARM_JOINT_DATA_TRAINING_PLAN.md` | [ARM_JOINT_DATA.md](ARM_JOINT_DATA.md#arm-joint-data-training-plan) |
| `ARM_JOINT_DPO_RESULTS.md` | [ARM_RESULTS.md](ARM_RESULTS.md#arm-joint-dpo-results) |
| `ARM_JOINT_ACTION_DPO_RESULTS.md` | [ARM_RESULTS.md](ARM_RESULTS.md#arm-joint-action-dpo-results) |
| `ARM_JOINT_SFT_HISTORY_EXAMPLE.md` | [ARM_JOINT_DATA.md](ARM_JOINT_DATA.md#arm-joint-sft-history-example) |
| `ARM_JOINT_SFT_RESULTS.md` | [ARM_RESULTS.md](ARM_RESULTS.md#arm-joint-sft-results) |
| `ARM_JOINT_SFT_VS_DPO_RESULTS.md` | [ARM_RESULTS.md](ARM_RESULTS.md#arm-joint-sft-vs-dpo-results) |
| `ARM_JOINT_TRAINING_MONITOR.md` | [ARM_RESULTS.md](ARM_RESULTS.md#arm-joint-training-monitor) |
| `ARM_JUDGE_ALIGNMENT.md` | [ARM_INFERENCE.md](ARM_INFERENCE.md#arm-judge-alignment) |
| `ARM_PREFERENCE_DISTILLATION_PLAN.md` | [ARM_PREFERENCE.md](ARM_PREFERENCE.md#arm-preference-distillation-plan) |
| `ARM_PREFERENCE_RUN_286384.md` | [ARM_PREFERENCE.md](ARM_PREFERENCE.md#arm-preference-run-286384) |
| `ARM_PREFERENCE_V2_CPU_AUDIT.md` | [ARM_PREFERENCE.md](ARM_PREFERENCE.md#arm-preference-v2-cpu-audit) |
| `ARM_PREFERENCE_V2_VIABILITY_PLAN.md` | [ARM_PREFERENCE.md](ARM_PREFERENCE.md#arm-preference-v2-viability-plan) |
| `ARM_RESULTS_DASHBOARD.md` | [ARM_RESULTS.md](ARM_RESULTS.md#arm-results-dashboard) |
| `BASELINE_CHECKPOINT_EVALUATION.md` | [RL_EVALUATION.md](RL_EVALUATION.md#baseline-checkpoint-evaluation) |
| `BASELINE_SCALING.md` | [RL_RUNTIME.md](RL_RUNTIME.md#baseline-scaling) |
| `BROWSER_USE_CHECKPOINT_EVALUATION.md` | [RL_EVALUATION.md](RL_EVALUATION.md#browser-use-checkpoint-evaluation) |
| `GRADIENT_DIAGNOSTICS.md` | [RL_METRICS.md](RL_METRICS.md#gradient-diagnostics) |
| `H200_TESTING.md` | [RL_RUNTIME.md](RL_RUNTIME.md#h200-testing) |
| `METRICS.md` | [RL_METRICS.md](RL_METRICS.md#metrics) |
| `PAPER_REWARD_COMPARISON.md` | [RL_METRICS.md](RL_METRICS.md#paper-reward-comparison) |
| `RESUMING_BASELINE.md` | [RL_RUNTIME.md](RL_RUNTIME.md#resuming-baseline) |
| `REWARD_RANK_EVALUATION_QUEUE.md` | [RL_EVALUATION.md](RL_EVALUATION.md#reward-rank-evaluation-queue) |
| `ROLLOUT_ARCHIVE.md` | [RL_RUNTIME.md](RL_RUNTIME.md#rollout-archive) |

- [Additive storage recovery: iteration 85 → 90](RL_RUNTIME.md#arm-additive-to90-storage-recovery-20260920)

- [ARM failure-turn sampling history and dynamic filtering](ARM_INTEGRATION_PLAN.md#arm-failure-sampling-history-20260922) · [summary plot](ARM_SUMMARY.md#arm-failure-sampling-history)

- [OpenWebRL task selection and additional WebGym candidates](ARM_INTEGRATION_PLAN.md#arm-task-pool-expansion-20260922)
- [Jev task-quality screening pilot](ARM_INTEGRATION_PLAN.md#arm-task-quality-jev-20260922)
  · [interactive uncertainty dashboard](arm_results/rl_integration/jev-quality-review.html)
  · [static overview](arm_results/rl_integration/jev-quality-overview.png)
