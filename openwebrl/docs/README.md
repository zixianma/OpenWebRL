# OpenWebRL project documentation

[GPT-6.1 Sol N=5: verified OM2W medium 45.67%, high 43.67%; WebVoyager 57.98%; DeepShop 46.00%](ARM_INFERENCE_SCALING.md#sol61-n5-launch-20261009).


[Original bonus iteration 90: local OM2W300 verified; counts, validity and historical comparison](RL_EVALUATION.md#arm-original-iter90-results-20261008).

[Local iteration90 benchmarks complete: six WebVoyager/DeepShop pairs, paired comparisons and audit](ARM_RESULTS.md#local-webvoyager-deepshop90-20261007).

[Sol candidate count complete: N=3 40.33%, N=10 47.00%; paired gain +6.67 pp, with cost and validity audit](ARM_INFERENCE_SCALING.md#sol-candidate-counts-20261007).

[Sol action diversity: ten candidates add 3.61 distinct parsed specifications beyond the first three at the same saved states; coordinate/wording sensitivity and actual N=3/5/10 coverage](ARM_INFERENCE_SCALING.md#sol-action-diversity-20261008).

[Teacher action-selection reasoning: what Piotr’s saved GPT-5.5 explanations favor, and what they do not establish](ARM_TEACHER_REASONING.md).

[Confidence replay pilot verified: 3/32 fresh states matched; expansion criteria failed](ARM_INFERENCE.md#confidence-robust-20261008).

[Benefit-based threshold study verified: first action 50% versus Piotr 35% on 20 usable held-out pairs; wide interval and low coverage](ARM_INFERENCE.md#confidence-benefit-20261008).

[Action-only likelihood: offline reranking changes parsed actions on 27.89% of comparable decisions; fresh rollout deferred](ARM_INFERENCE.md#action-only-likelihood-20261008).

[Likelihood-of-five 26.67%; exploratory gated Piotr 33.67%; both full300 complete, paired comparisons, calibration limits and plots](ARM_INFERENCE.md#likelihood-scaling-20261007).

[RL actor + Piotr complete: 42.67% to 54.00%, paired gain +11.33 pp; frozen SFT frontend, final accounting and three plots](ARM_INFERENCE.md#selectionarm-piotr-rlactor30-20261007).

[Random-of-five complete: SFT 33.00%, random 33.33%, Piotr SelectionARM 40.00%; paired comparisons and three plots](ARM_INFERENCE_SCALING.md#random5-sameday-results-20261007).

[Same-day SFT/Piotr30-step pair complete: gain +7.00 pp; three full300 runs summarized](ARM_INFERENCE.md#selectionarm-piotr-sameday30-20261007).

[Piotr SelectionARM at 50 steps: full300 paired results, comparison with 30 steps and three plots](ARM_INFERENCE.md#selectionarm-piotr-steps50-results-20261007).

[RL-task SelectionARM repeat complete: 41.00% versus 31.67% reused SFT control; full300 paired results and three plots](ARM_INFERENCE.md#selectionarm-rltasks-repeat-results-20261007).

[API-selector reproduction complete: SFT 31.67%, Sol 43.33%, GPT-5.5 42.00%; 900 episodes, paired results and three plots](ARM_INFERENCE.md#api-selector-september-reproduction-20261006).

[SFT versus Piotr SelectionARM: three complete full300 runs, mean ± sample SD](ARM_INFERENCE.md#selectionarm-piotr-sameday30-20261007).

[SFT versus Piotr ARM: both full300 repeats complete; SFT 33.00%, ARM 38.00%, mean gain +5.00 pp. Final accounting and compute/latency/token plots](ARM_INFERENCE.md#sft-piotr-repeats-20261006).

[Historical-vs-corrected verdict audit: same 158 tasks, all 23 both-judged changes reviewed, unjudged-zero contribution quantified](ARM_INFERENCE.md#arm-same-task-verdict-audit-20261006).

[Corrected ARM comparison: 493/900 records, 158 paired tasks; final accounting and cost/latency/token plots](ARM_INFERENCE.md#arm-rltasks-corrected-partial-20261006).

[Inference scaling: three-run Piotr results, random/horizon controls, selectors by protocol, actors alone and completed RL-actor comparison; supporting evidence and history collapsed](ARM_INFERENCE_SCALING.md).

Documents are organized by topic. Start with the results or operational guide
for the work you are doing, then follow its contents to dated experiment records.

[Branching studies: two data families, matched results and remaining tests for post-action evidence and selector training](ARM_FORMULATIONS.md#branching-experiment-map). [Latest continuation-count comparison](ARM_FORMULATIONS.md#branch-extra24-20261008) · [Separately teacher-trained actor heads](ARM_FORMULATIONS.md#selection-head-20261009).

[Luna-high independent 200-task teacher diagnostic completed: 32/200 labels change; 100-task repeat controls, evidence review and plots](ARM_RESULTS.md#arm-teacher-evidence-primary200-20261006).
[Original early-state readout: 70 paired states; after-execution choices are more repeatable, while the success difference remains unresolved](ARM_INTEGRATION_PLAN.md#arm-continuation-early-signal-20261006).
[October4 critic data inventory: 24 sources/derived views, actual supervision, conversion readiness and benchmark exclusions](ARM_JOINT_DATA.md#arm-critic-source-inventory-20261004).
[Piotr post-execution audit: 2,673 verified transitions, recovered tool receipts, missing labels and next-experiment priorities](ARM_JOINT_DATA.md#piotr-post-execution-audit-20261006).
[October4 controlled inference completed: ARM39.33%, pass@1 35.20%, pass@5 59.33%; paired FLOP/cache and browser-cost comparisons](ARM_INFERENCE.md#arm-controlled-inference-results-20261004).
[October4 historical comparison, corrected: missing actor policy prevents a matched replication claim](ARM_INFERENCE.md#arm-historical-reconciliation-20261004).
[Historical October4 critic preparation:14,825 mixed-checkpoint transitions and leakage guards; superseded as first-fit data by the fixed-policy decision](ARM_INTEGRATION_PLAN.md#arm-critic-comparison-20261004).
[October4 historical inference audit: ARM actor-text cost, training-pool pass@k and the limitation addressed by the fresh controlled cohort](ARM_INFERENCE.md#arm-inference-cost-passk-20261004).

| Document | Contents |
| --- | --- |
| [ARM_SUMMARY.md](ARM_SUMMARY.md) | Concise methods and headline results; presentation figures for the three stages and RL variants; static/interactive plots; collapsed WebVoyager results |
| [ARM_FORMULATIONS.md](ARM_FORMULATIONS.md) | Branching experiment map; matched pre/post evidence, held-out outcome selection, continuation scaling, teacher-trained heads and planned student controls |
| [ARM_RESULTS.md](ARM_RESULTS.md) | Detailed ARM results and analysis, online-RL coverage/admission audits, C2/1A/joint comparisons, and curves |
| [ARM_INTEGRATION_PLAN.md](ARM_INTEGRATION_PLAN.md) | ARM model understanding, literature, integration options, and longer-term roadmap |
| [ARM_INFERENCE.md](ARM_INFERENCE.md) | Inference protocol, judge alignment, and retry policy |
| [ARM_INFERENCE_SCALING.md](ARM_INFERENCE_SCALING.md) | Controlled inference results, repeats and ablations; actor/selector comparisons, pending studies and collapsible evidence/history |
| [ARM_SFT.md](ARM_SFT.md) | C2 collection, filtering, SFT, 1A configuration, checkpoint procedures, and [explicit action comparison](ARM_SFT.md#arm-comparison-sft-20261007) |
| [ARM_PREFERENCE.md](ARM_PREFERENCE.md) | Preference experiments, pair audits, and viability plans |
| [ARM_JOINT_DATA.md](ARM_JOINT_DATA.md) | Combined C2/Piotr data review, examples, training, and recovery |
| [RL_RESULTS.md](RL_RESULTS.md) | Tables-only RL checkpoint scores; overall, valid-only, invalid counts, and separate protocols |
| [RL_RUNTIME.md](RL_RUNTIME.md) | Reference baseline resume, GPU scaling, archives, and runtime history |
| [RL_EVALUATION.md](RL_EVALUATION.md) | Evaluation harness code map, protocols, commands/tests, checkpoint results and scheduling |
| [RL_METRICS.md](RL_METRICS.md) | Metric definitions, paper reward comparisons, and gradient diagnostics |
| [RL_EXPERIMENTS.md](RL_EXPERIMENTS.md) | Prepared RL recipe experiments and validation |

[Interactive RL comparison](rl_results/arm_rl_interactive.html): includes both mixed-only runs through90 and the expanded4,102-task outcome-only run through50; toggle runs, select bias-corrected EMA or centered Gaussian smoothing, switch overall/valid-only and export raw data; fixed0–60% y-axis. Download and open the standalone HTML in a browser; no server required.

[Reward-hacking diagnostic](ARM_RESULTS.md#arm-reward-hacking-curves-20261003):673 saved training collections across nine ARM variants; independent full300 task success versus ARM bonus/proxy, with separate failure buffers, label coverage and credited selection. [Interactive view](rl_results/arm_rl_interactive.html#reward-hacking) · [Figure](rl_results/arm_reward_hacking.png) · [PDF](rl_results/arm_reward_hacking.pdf).

[October3 next experiment: independently test ARM-assisted task selection](ARM_INTEGRATION_PLAN.md#arm-selection-control-20261003). CPU audit and private paired schedule prepared for682 tasks;78 historical rescues span11 hosts. Fresh ARM versus actor retries and host-matched selection controls; no new allocation approved or submitted. [Concise current decision](ARM_SUMMARY.md#arm-next-experiment-20261003).

[Iteration 90 stealth: five-method table with three-repeat means, sample SDs and overall 95% task-cluster BCa CIs](ARM_SUMMARY.md#arm-stealth90-five-method-summary-20261007). Baseline, Additive and Gate B were collected September 29–30; mixed-only bonus and reweight were collected October 6 (PDT). Cross-period comparisons remain descriptive. [CI method and limits](RL_EVALUATION.md#arm-stealth90-five-method-summary-20261007) · [Aggregate](arm_results/rl_integration/stealth-iteration90-five-method-overall-ci.json).

[Matched iteration90 uncertainty: paired 95% CIs and Holm-corrected tests; no significant pair among outcome-only, Additive and Gate B](ARM_RESULTS.md#arm-stealth90-paired-inference-20260930).

[ARM method presentation guide](ARM_SUMMARY.md#arm-methods-at-a-glance): three stages, group composition, Gate B/C credit and bonus versus reweighting, with downloadable PNG/SVG figures.

[Evolving ARM: later-actor offline agreement57.68%→59.57% (+1.89pp)](ARM_RESULTS.md#arm-offline-forward-transfer-results-20260925).
[Browser comparison331770 complete: frozen42.00% /56.50% versus refreshed41.33% /52.99% overall/valid-only on full300](ARM_RESULTS.md#arm-refresh-browser-results-20260926); no demonstrated browser gain.

[This week's priorities: the mixed-only pair and historical Gate B through90](ARM_INTEGRATION_PLAN.md#arm-weekly-priority-20260927).
[Historical Gate B training100 and full300 evaluation verified; allocation released](ARM_INTEGRATION_PLAN.md#arm-gate-b-to100-20260928).
[Mixed-only pair continuations through90: bonus338904→338906 and reweight338905; full300 at70/80/90, unchanged recipes and bounded budgets](ARM_INTEGRATION_PLAN.md#arm-mixed-pair-to90-20260930).
[Mixed-only pair to60 completed: twelve full300 milestones, final checkpoints and all3,600 saved attempts verified; original approvals preserved](ARM_INTEGRATION_PLAN.md#arm-mixed-pair-to60-20260928).
[Exact methods: lambda0.5, matched relaxed gate B and no failure auxiliary](ARM_INTEGRATION_PLAN.md#arm-outcome-aware-reweighting-20260926).
[Review: proposed outcome-supervised reward versus ORM, PRIME and SelectionARM—input/label/loss table, diagram, credit-assignment example and open alternative](ARM_INTEGRATION_PLAN.md#arm-outcome-orm-prime-comparison).
[Outcome-reward data prepared:2,000 train /250 dev /500 later-actor test complete trajectories, task-disjoint](ARM_INTEGRATION_PLAN.md#arm-outcome-trained-reward-investigation-20260927).
[Outcome-reward readiness: image hashes/processor/gradient checks pass; length-only later-test pair accuracy65.48%; GPU fit pending](ARM_INTEGRATION_PLAN.md#arm-outcome-reward-readiness-20260928).
[Task-pool expansion: quality, deduplication, browser-yield and controlled training plan](ARM_INTEGRATION_PLAN.md#arm-task-pool-next-investigation-20260927).
[Task screening v2 and semantic overlap complete:52 provisional candidates,14 excluded,9 held; job337135 verified in118 GPU-seconds; browser pilot pending](ARM_INTEGRATION_PLAN.md#arm-task-pool-quality-v2-20260929).
[Task-pool redesign: five-per-host was our review cap; existing-host restriction leaves25 hosts and removes nearly all additional InSTA tasks; broader medium/hard pool spans15,933 hosts](ARM_INTEGRATION_PLAN.md#arm-task-pool-redesign-20260929).
[Upstream-code rerun: semantic job337317 verified in11m42s;184,546 candidates retained,21,396 quarantined; website/task quality remains pending](ARM_INTEGRATION_PLAN.md#arm-task-pool-upstream-rerun-20260929).
[Next experiment: ARM-based task selection among five-valid-failure groups; proposed outcome-verified screening and matched data-selection controls](ARM_INTEGRATION_PLAN.md#arm-task-selection-all-failure-20260929).
[Quality-screening protocol and trust audit: Jev/GPT/code comparison, blind human review, and separate browser validation](ARM_INTEGRATION_PLAN.md#arm-task-quality-human-review-20260929).
[Interactive manual review:10 paired cached cases,65 additional Jev cases,12 unlabeled broad-pool examples; model reveal and JSON export](arm_results/rl_integration/jev-quality-review-v2.html).

[Training pipeline profile, October 6: recent37.87min cycles; PPO50.5%, collection36.9%, checkpoint save0.8%. Prioritized systems checks; no deployed performance change or demonstrated end-to-end speedup](RL_RUNTIME.md#training-pipeline-profile-20261006).
[ARM timing audit:48GiB cycles now40–46min; training/save time per update down27–29% in adjacent batches](RL_RUNTIME.md#arm-iteration-throughput-20260927).
[Full browser availability audit complete:96,779 available URLs covering155,512 candidate tasks;21,328 inconclusive and7,654 unavailable URLs; instruction quality/actor difficulty still unassessed](ARM_INTEGRATION_PLAN.md#arm-task-pool-live-browser-full-20260929).
[Benchmark-site grouping:59,115 available tasks on185 benchmark-associated sites;96,397 on82,021 sites with no known match; WebVoyager/OM2W/WebTailBench/DeepShop flags and WebTailBench coverage caveat](ARM_INTEGRATION_PLAN.md#arm-task-pool-benchmark-websites-20260930).
[Difficulty-first 2K vs. weighted coverage over score≥5: exact task mix, 61.25% overlap and local reviews](ARM_INTEGRATION_PLAN.md#arm-task-pool-difficulty-comparison-20260930).
[Selected ≥5 weighted2K pool: all10,000 actor attempts verified;682 five-valid-failure tasks,1,061 mixed,213 all-success,44 unresolved;43.86GPUh/$52.62 used](ARM_INTEGRATION_PLAN.md#arm-task-pool-actor-screen-20260930) · [Local selected-task review](http://localhost:8765/arm_min5_weighted_tasks.html).
[Actor + ARM rescue collection complete:78/682 rescued;11.44% overall,11.49% valid-only; all rollouts/verdicts/candidate traces verified;9.33GPUh/$1.68 used](ARM_INTEGRATION_PLAN.md#arm-task-pool-guided-682-20261002).
[Uniform G8 outcome-only pilot: iteration 10 verified, 88/300 overall and 88/241 valid-only; comparison limits and artifact audit](RL_EVALUATION.md#uniform8-iter10-results-20261007).

[Batch-56 and uniform-G8 continuation to 60: separately approved 48h/$300 and 72h/$400 caps; queued October 8, full300 evaluations every ten iterations](ARM_INTEGRATION_PLAN.md#outcome56-uniform8-to60-20261008).

[Training-pool SFT coverage: historical five attempts plus three on misses improves 64.65% to 73.20%; rubric-difficulty analysis and sampling limits](ARM_INFERENCE_SCALING.md#training-task-pass5-pass8-20261006).

[Original2,102-task outcome-only56-group control: iteration30 verified at32.33% overall /38.80% valid-only; target60 incomplete, with historical comparison limits](RL_EVALUATION.md#outcome56-iter30-results-20261009) · [Control method and approved caps](ARM_INTEGRATION_PLAN.md#outcome56-control-20261006).
[Expanded-pool throughput audit:68.45min/iteration; two g011 GPUs have confirmed thermal throttling. Phase timing and safe recovery follow-up](RL_RUNTIME.md#expanded4102-thermal-throughput-20261003).
[Expanded4,102-task outcome-only baseline: complete through60 and all six full300 evaluations verified. Final32.33% overall /39.43% valid-only (97 successes,246 valid); best observed40=39.00% overall. Interactive curve includes10–60; historical comparisons remain descriptive](RL_EVALUATION.md#expanded4102-iter10-results-20261003) · [Method and continuation](ARM_INTEGRATION_PLAN.md#arm-expanded-outcome-to90-20261005) · [Browser repair](RL_RUNTIME.md#expanded4102-egl-regression-20261003).
[Original extension approval:96h compute/$400 additional judge allowance; current stop60, unchanged recipe and full300 every10, with unused allocations released after final verification](ARM_INTEGRATION_PLAN.md#arm-expanded-outcome-to90-20261005).
[Review all 2,000 selected tasks and difficulty-aware sampling proposal](ARM_INTEGRATION_PLAN.md#weighted-screening-cohort-frozen--september30) · [Local task-review page](http://localhost:8765/arm_selected_tasks.html).
[OM2W easy/medium/hard: all five methods, including mixed-only bonus and reweight, at iteration 90 with three stealth repeats](ARM_RESULTS.md#arm-stealth90-five-method-difficulty-20261007). September 29–30 and October 6 (PDT) collections are labeled separately; comparisons across dates are descriptive. [Aggregate](arm_results/rl_integration/stealth-o4-t06-iteration90-five-method-difficulty.json).
[Diversity clustering and sampling:59,115 tasks, interactive cluster map, weighted coverage vs. three alternatives](ARM_INTEGRATION_PLAN.md#arm-task-pool-clustering-20260930) · [Dashboard HTML](arm_results/rl_integration/task-pool-clusters.html).
[Browser-first pilot337474 verified:160/200 URLs available,32 inconclusive,8 unavailable;58,892 candidate tasks share available pages;4m26s CPU-only](ARM_INTEGRATION_PLAN.md#arm-task-pool-live-browser-first-20260929).
[Final three-repeat stealth90 means: baseline55.22 ±2.14%, Additive58.44 ±1.35%, Gate B58.78 ±3.89% overall; all nine cohorts verified, valid-only rates and denominators included](ARM_RESULTS.md#arm-stealth90-o4-three-repeat-summary-20260930).
[WebVoyager595 iteration90 complete: baseline66.89% /68.27%, Additive64.37% /65.03%, Gate B67.06% /67.86% overall/valid-only; all archives verified; browser charges$7.11](ARM_RESULTS.md#arm-webvoyager90-results-20260930).
[Gate B training100 complete: full30036.67% /48.89%, fixed10030.00% /44.12%; all artifacts verified and comparison plot updated](RL_EVALUATION.md#arm-gate-b-iter100-results-20260929).
[Historical GPT-4.1/T0 stealth evals: Additive53.00% /56.18% (283 valid), Gate B56.33% /58.68% (288 valid); these used the wrong protocol for the intended comparison](RL_EVALUATION.md#arm-stealth90-threeway-repeats-20260928).
[Current inventory, September28 evening: Gate B saved90/collects91; mixed bonus saved19/collects20; reweight saved22/collects23; automatic continuation verified](RL_RUNTIME.md#arm-progress-20260928).
[Mixed-only iteration20 full300: bonus30.67% /40.89%; reweight32.33% /42.73% overall/valid-only; archives verified](ARM_RESULTS.md#arm-mixed-pair-iter10-results-20260928).
[Mixed-only iteration30 full300: bonus32.00% /42.11% (228 valid); reweight30.00% /41.10% (219 valid); both300-task cohorts and archives verified](ARM_RESULTS.md#arm-mixed-pair-iter30-results-20260929).
[Mixed-only iteration40 full300: bonus33.33% /43.86% (228 valid), reweight35.33% /47.11% (225 valid); all archives/verdicts verified, both continue toward60](ARM_RESULTS.md#arm-mixed-pair-iter40-results-20260929).
[Both mixed-only runs completed60: bonus37.67% /48.71% (fixed10041.00% /54.67%); reweight36.67% /47.21%; all12cohorts verified, GPUs released;40/50/60 overall means36.56% vs36.33%](ARM_RESULTS.md#arm-mixed-pair-iter60-results-20260930).
[Mixed-only bonus70 verified:38.33% /48.94% full300 (235 valid),44.00% /57.14% fixed100; all300 saved attempts checked; both continuations remain active](ARM_RESULTS.md#arm-mixed-bonus-iter70-results-20261001).
[Mixed-only reweight70 verified:38.67% /49.57% full300 (234 valid),38.00% /50.00% fixed100; one success above bonus70; training continues](ARM_RESULTS.md#arm-mixed-reweight-iter70-results-20261001).
[Mixed-only bonus80 verified:36.00% /48.87% full300 (221 valid),35.00% /50.72% fixed100; recovery340425 held for storage before allocation, durable88 preserved](ARM_RESULTS.md#arm-mixed-bonus-iter80-results-20261001).
[Mixed-only pair90 complete: bonus40.67% /52.36%, reweight37.67% /49.78% full300; all nine milestones each verified and GPUs released](ARM_RESULTS.md#arm-mixed-bonus-iter90-results-20261002).

[Evaluation harness on `arm`: topic index, entry points, protocol table, dry-run commands and regression tests](RL_EVALUATION.md#evaluation-harness-guide). Runtime task data, trajectories, credentials and checkpoints remain private.

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
