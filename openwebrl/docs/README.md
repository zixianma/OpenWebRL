# OpenWebRL project documentation

Documents are organized by topic. Start with the results or operational guide
for the work you are doing, then follow its contents to dated experiment records.

| Document | Contents |
| --- | --- |
| [ARM_SUMMARY.md](ARM_SUMMARY.md) | Concise methods and headline results across inference, offline training and RL; comparison plot includes Gate B at iterations20–80 |
| [ARM_RESULTS.md](ARM_RESULTS.md) | Detailed ARM results and analysis, online-RL coverage/admission audits, C2/1A/joint comparisons, and curves |
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

[Evolving ARM: later-actor offline agreement57.68%→59.57% (+1.89pp)](ARM_RESULTS.md#arm-offline-forward-transfer-results-20260925).
[Browser comparison331770 complete: frozen42.00% /56.50% versus refreshed41.33% /52.99% overall/valid-only on full300](ARM_RESULTS.md#arm-refresh-browser-results-20260926); no demonstrated browser gain.

[This week's priorities: the mixed-only pair and historical Gate B through90](ARM_INTEGRATION_PLAN.md#arm-weekly-priority-20260927).
[Historical Gate B:335729 queued after335681, approved additional8 H200 ×16h through100/eval100; preserve and verify90/eval90 first](ARM_INTEGRATION_PLAN.md#arm-gate-b-to100-20260928).
[Mixed-only pair to60 submitted: reweight335697→335698; bonus335699→335700 after335682; full300 every10, approved additional8 H200 ×48h per variant](ARM_INTEGRATION_PLAN.md#arm-mixed-pair-to60-20260928).
[Exact methods: lambda0.5, matched relaxed gate B and no failure auxiliary](ARM_INTEGRATION_PLAN.md#arm-outcome-aware-reweighting-20260926).
[Review: proposed outcome-supervised reward versus ORM, PRIME and SelectionARM—input/label/loss table, diagram, credit-assignment example and open alternative](ARM_INTEGRATION_PLAN.md#arm-outcome-orm-prime-comparison).
[Outcome-reward data prepared:2,000 train /250 dev /500 later-actor test complete trajectories, task-disjoint](ARM_INTEGRATION_PLAN.md#arm-outcome-trained-reward-investigation-20260927).
[Outcome-reward readiness: image hashes/processor/gradient checks pass; length-only later-test pair accuracy65.48%; GPU fit pending](ARM_INTEGRATION_PLAN.md#arm-outcome-reward-readiness-20260928).
[Task-pool expansion: quality, deduplication, browser-yield and controlled training plan](ARM_INTEGRATION_PLAN.md#arm-task-pool-next-investigation-20260927).
[Task screening v2 prepared:75 candidates +25 existing-site controls,4,411 semantic-reference texts; no new API/browser run](ARM_INTEGRATION_PLAN.md#arm-task-pool-screening-v2-20260928).

[ARM timing audit:48GiB cycles now40–46min; training/save time per update down27–29% in adjacent batches](RL_RUNTIME.md#arm-iteration-throughput-20260927).
[Additive/Gate B iteration90: one full300 stealth evaluation each, two one-GPU jobs; plot retains baseline stealth58/90](RL_EVALUATION.md#arm-stealth90-threeway-repeats-20260928).
[Current inventory, September28 evening: Gate B saved90/collects91; mixed bonus saved19/collects20; reweight saved22/collects23; automatic continuation verified](RL_RUNTIME.md#arm-progress-20260928).
[Mixed-only iteration20 full300: bonus30.67% /40.89%; reweight32.33% /42.73% overall/valid-only; archives verified](ARM_RESULTS.md#arm-mixed-pair-iter10-results-20260928).

Current scaling plan: [eight-GPU topology and browser benchmark](RL_RUNTIME.md#eight-gpu-scaling-benchmark-20260912).

ARM prefix pilot331932 [completed: actor-only38.54%,two-turn42.71%,four-turn40.63% overall on the training panel](ARM_RESULTS.md#arm-prefix-curriculum-results-331932).
Small, inconclusive positive signal for two-turn guidance; all288 primary attempts saved. [Protocol](ARM_INTEGRATION_PLAN.md#arm-prefix-curriculum-pilot-20260926).

Live operations: [job status, completions and failures](arm_results/rl_integration/live-status.html)
(one-minute refresh; [monitor behavior](RL_RUNTIME.md#arm-live-monitor-20260921)).
[Active-agent supervision: hourly summaries, urgent failure review and bounded retries](RL_RUNTIME.md#arm-active-agent-supervision-20260927).
[Persistent supervisor repaired: systemd user service with fresh heartbeat and current replacement IDs; hourly routine reports and urgent failure review](RL_RUNTIME.md#arm-progress-20260928).
[Other lineages: C completed through60; sampling40% stopped at27;
beta333431 stopped after21/312 updates](RL_RUNTIME.md#arm-progress-20260927).
[Sampling40% coverage audit and stop decision:1.85× usable labels, inconclusive early task-success difference](ARM_RESULTS.md#arm-failure-sampling40-stop-20260927).
[Gate B90 full300:43.00% overall /55.13% valid-only; fixed10039.00% /54.93%; comparison plot updated](RL_EVALUATION.md#arm-gate-b-iter90-results-20260928).
[Beta1 iteration10 full300:27.33% overall /33.74% valid-only](RL_EVALUATION.md#arm-failure-beta1-iter10-results-20260925);
[iteration20:33.00% overall /41.77% valid-only](RL_EVALUATION.md#arm-failure-beta1-iter20-results-20260925).
The priority pair's scratch-path and Ray socket-length startup bugs are fixed and validated. [First-batch validation](RL_RUNTIME.md#arm-mixed-first-batch-20260927). Beta333431 completed its full TP4 replay and preserved checkpoint21.
Browser comparison331770 and prefix pilot331932 are complete.
[Original-bonus20/30/40/50/60 full300 backfills completed:28.00%,32.67%,32.00%,34.00%,35.67% overall;
all rollout/verdict pairs saved,7.466 GPU-hours used](RL_EVALUATION.md#arm-original-backfill-results-20260927).

[Gate B recovery outcome: startup degradation persisted; no new update, early-stop guard CPU-tested, live diagnosis still required](RL_RUNTIME.md#arm-browser-slot-recovery-20260927).

Default checkpoint evaluation cadence and current queue:
[full-300 every ten iterations; September24 temporary-storage cleanup](RL_RUNTIME.md#milestone-evaluations-20260924).
[Gate B iteration50/60:37.67% /37.33% overall](RL_EVALUATION.md#arm-gate-b-iter50-60-results-20260926),
[Gate B iteration30/40 full-300 results](RL_EVALUATION.md#arm-gate-b-iter30-40-results-20260924)
and [Gate C iteration30/40 results](RL_EVALUATION.md#arm-gate-c-iter30-40-results-20260924)
are complete, as are [C50/C60 full300 and fixed100 results](RL_EVALUATION.md#arm-gate-c-iter50-60-results-20260925).


Prepared baseline continuation: [stage-2 recipe, dynamic sampling audit, and launch](RL_RUNTIME.md#baseline-stage2-20260914).

Latest requested comparison: [outcome-only and additive stage-1 continuation to iteration 100](RL_RUNTIME.md#stage1-baseline-additive-to100-20260921), with a full-300 evaluation inside each allocation.
Additive100 is [complete at 36.33% overall / 50.23% valid-only](RL_EVALUATION.md#arm-additive-iter100-results-20260921);
baseline100 [completed on September24 at34.67% overall /45.81% valid-only](RL_EVALUATION.md#baseline-iter100-results-20260924).
[Training continuation and quota-recovery history](RL_RUNTIME.md#training-relaunch-20260922).
[September 22 quota recovery, checkpoint retention and storage inventory](RL_RUNTIME.md#storage-inventory-20260922).
[September 24 verified cleanup and restored quota headroom](RL_RUNTIME.md#storage-cleanup-20260924-afternoon).
[Beta/sampling correction: new interventions start at iteration0](RL_RUNTIME.md#failure-ablations-fromzero-20260922).
[Passed two-GPU diagnostic and remaining validation gaps](RL_RUNTIME.md#failure-ablation-gpu-smoke-proposal-20260922).
[B eight-GPU throughput regression and verified topology](RL_RUNTIME.md#arm-b-eight-gpu-topology-audit-20260922).
[Confirmed browser NVIDIA EGL stall and live correction](RL_RUNTIME.md#arm-browser-egl-fix-20260922).
[Queued TP2/DP4 replay and B/C reward/efficiency monitoring](RL_RUNTIME.md#arm-tpdp-replay-20260922).
[Beta and sampling-rate migration to TP2/DP4](RL_RUNTIME.md#arm-ablation-tp2-migration-20260922).
[Training/evaluation inventory and remaining work](RL_RUNTIME.md#arm-job-inventory-20260921).

Latest baseline: [iteration-100 completion](RL_EVALUATION.md#baseline-iter100-results-20260924),
[iteration-90 completion](RL_EVALUATION.md#scheduled-eval90-results-20260913),
[iteration-80 GPT-4.1 rejudging](RL_EVALUATION.md#stealth80-gpt41-rejudge-feasibility-20260913),
and [cluster/account queue audit](RL_RUNTIME.md#cluster-queue-audit-294983-20260913).

ARM RL: [baseline40→90 ARM refresh through Piotr’s repo; approved pipeline / job329708](ARM_INTEGRATION_PLAN.md#arm-offline-forward-transfer-20260924),
[historical September22 priority: gate C](ARM_INTEGRATION_PLAN.md#arm-gate-c-priority-20260922),
[detailed analysis moved from the collaborator summary](ARM_RESULTS.md#arm-online-rl-analysis-20260922),
[saved termination audit and beta/coverage launch proposal](ARM_INTEGRATION_PLAN.md#arm-failure-termination-audit-20260921),
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
- Keep `ARM_SUMMARY.md` limited to core methods, result tables and the main comparison plot. Put detailed analysis, audits and examples in `ARM_RESULTS.md`, and operational history in `RL_RUNTIME.md`.
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

- [ARM failure-turn sampling history and dynamic filtering](ARM_INTEGRATION_PLAN.md#arm-failure-sampling-history-20260922) · [analysis and plot](ARM_RESULTS.md#arm-failure-sampling-history)

- [OpenWebRL task selection and additional WebGym candidates](ARM_INTEGRATION_PLAN.md#arm-task-pool-expansion-20260922)
- [Jev task-quality screening pilot](ARM_INTEGRATION_PLAN.md#arm-task-quality-jev-20260922)
  · [interactive uncertainty dashboard](arm_results/rl_integration/jev-quality-review.html)
  · [static overview](arm_results/rl_integration/jev-quality-overview.png)

[September26 ARM status: B69 recovery331778, B50/B60 results, C60 complete](RL_RUNTIME.md#arm-progress-20260926).

[Failure sampling40% iteration10:29.00% overall /37.34% valid-only](RL_EVALUATION.md#arm-failure-coverage-iter10-results-20260926).

[Next ARM experiments after B/C: hybrid token credit, failure-only auxiliary supervision, and conditional reward-model refresh (discussion draft)](ARM_INTEGRATION_PLAN.md#arm-next-experiments-after-bc-20260926).

[Additive relaxed, failure-only ARM: prepared; resources approved but launch deferred for scientific audit](ARM_INTEGRATION_PLAN.md#arm-additive-relaxed-failureonly-20260926).

[Gate B mixed-group audit: all69 iterations; weak positive outcome association](ARM_RESULTS.md#arm-gate-b-mixed-alignment-20260926) · [Outcome-aware turn reweighting proposal](ARM_INTEGRATION_PLAN.md#arm-outcome-aware-reweighting-20260926) · [Sampling40% iteration20 full300:27.67% /37.39%](RL_EVALUATION.md#arm-failure-coverage-iter20-results-20260926).
