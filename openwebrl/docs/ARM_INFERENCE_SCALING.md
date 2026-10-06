# ARM inference-time scaling: actors, selectors, and retries

**Final results through October 6, 2026.** This is the consolidated comparison of Qwen, official OpenWebRL SFT, GPT-6 Luna, GPT-6.1 Sol, Jev, Kev, and learned ARM inference policies on Online-Mind2Web. The eleven recent full-set rows are complete: **3,000 fresh episodes plus 300 reused SFT reference episodes**. Each row covers the same 300 task IDs, but the protocol families differ.

The clearest matched result is **Qwen + Luna N=5: 28.33% versus Qwen alone 17.67%**, at 2.23× estimated serving cost. Increasing SFT proposals from five to ten did not improve observed success. SFT + Jev/Kev has the highest recorded scores, but uses a different browser and judging protocol; it is not a controlled comparison against Luna selection.

## Contents

- [Full300 results](#actor-selector-experiment-tracker-20261004)
- [What each experiment measures](#luna-actor-full300-20261004)
- [Selection and reasoning scaling](#scaling-comparisons)
- [Cost, latency, tokens, and plots](#luna-full300-results-20261006)
- [GPT-6 actor and judge audit](#api-actor-stopping-audit-20261005)
- [Learned ARM versus episode retries](#learned-arm-and-retries)
- [Pilot results and recovery caveats](#luna-qwen-inference-20261004)
- [Evidence and accounting](#evidence-and-accounting)

<a id="actor-selector-experiment-tracker-20261004"></a>
## Full300 results

All rows below are complete. **Overall = successes / 300; valid-only = successes / valid records.** Invalid records remain in the overall denominator. These are saved judge verdicts, not independently certified strict task completions. Valid-only does not remove every blocked page: a blocked trajectory can be a valid judged failure.

<!-- actor-selector-results:start -->
| ID | Actor | Selector | N | Successes | Valid | Overall | Valid-only |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| AS01 | Qwen3-VL-4B-Thinking | None | 1 | 53 | 277 | 17.67% | 19.13% |
| AS02 | Qwen3-VL-4B-Thinking | GPT-6 Luna medium | 5 | 85 | 276 | 28.33% | 30.80% |
| AS04 | Official OpenWebRL-4B-SFT, reused reference | None | 1 | 106 | 272 | 35.33% | 38.97% |
| AS05 | Official OpenWebRL-4B-SFT | GPT-6 Luna medium | 5 | 113 | 274 | 37.67% | 41.24% |
| AS06 | Official OpenWebRL-4B-SFT | GPT-6 Luna medium | 10 | 112 | 274 | 37.33% | 40.88% |
| AS07 | GPT-6 Luna medium | None | 1 | 110 | 279 | 36.67% | 39.43% |
| AS08 | Official OpenWebRL-4B-SFT | Jev | 5 | 176 | 284 | 58.67% | 61.97% |
| AS09 | Official OpenWebRL-4B-SFT | Kev27B | 5 | 184 | 291 | 61.33% | 63.23% |
| AS10 | GPT-6 Luna high | None | 1 | 97 | 277 | 32.33% | 35.02% |
| AS11 | GPT-6.1 Sol high | None | 1 | 61 | 273 | 20.33% | 22.34% |
| AS12 | Kev27B direct DOM actor | None | 1 | 26 | 294 | 8.67% | 8.84% |
<!-- actor-selector-results:end -->

The stable IDs match the [aggregate tracker](arm_results/luna_full300_20261004/experiment_tracker.json), finalized at **2026-10-06 03:52:48 UTC**. AS03 (Qwen + Luna N=10) was excluded from the full300 scope. Kev0.8B and direct Jev have pilot results only. No full300 result is inferred for an unrun arm.

<a id="luna-actor-full300-20261004"></a>
## What each experiment measures

**N is the number of proposed next actions per browser step.** A selector chooses one candidate, and only that candidate is executed. Direct actors propose their own next action. Episode pass@k instead runs k independent browser trajectories and counts whether any succeeds.

| Protocol family | Actor and selector input | Sampling / output budget | Browser and episode limit | Judging / stopping |
| --- | --- | --- | --- | --- |
| Qwen/SFT + Luna; Qwen alone | Actor sees screenshot and history; Luna sees goal, history, current screenshot and shuffled full candidate actions | Local actor T=1.0, p=0.95, top-k disabled; 4,096 tokens. Luna medium, API-controlled sampling | Local Chromium, 1280×720; 30 actions, 1,800s | Actions-only o4-mini/AgentTrek evidence; step-limit failure scores zero |
| Luna/Sol direct actors | Screenshot and history; native GUI function schemas | Medium/high reasoning as labeled; 4,096 output tokens; API-controlled sampling | Same local browser and limits | Same actions-only protocol; see harness audit below |
| SFT + Jev/Kev27B | SFT sees screenshot/history. Selector sees page text, element geometry, five full reasoning/action candidates and five recent selected actions; **no selector images** | SFT T=1.0, p=0.95, top-k=20; 4,096 tokens | Hosted BrowserUse, 1280×1000; 30 turns, 600s | Thoughts/actions and final screenshot; terminal/step-limit outcomes judged by o4-mini/AgentTrek |
| Direct Jev/Kev | DOM text and operation/target selection; **no actor images**; GPT-4.1-mini generates typing text | Upstream decision policy; text helper T=0.6, p=0.95, 1,024 tokens | Hosted BrowserUse, 1120×780; 30 actions, up to 60 decisions, 600s | o4-mini/AgentTrek terminal judge |
| Reused SFT reference | Original SFT actor0 from the controlled ARM study | T=0.7, p=0.9; 1,024 tokens | Historical shared actor serving | Thoughts-inclusive evidence; unmatched to fresh selector cohorts |

The SFT actor is the released **`OpenWebRL/OpenWebRL-4B-SFT`**, not Qwen Thinking. The separate Qwen actor is **`Qwen/Qwen3-VL-4B-Thinking`**. API reasoning actors do not receive the local actors' temperature/top-p settings. Direct Kev is a different policy from SFT + Kev: it selects DOM operations/targets itself, whereas selectionARM chooses among SFT proposals. Its full300 score therefore does not isolate the value of selecting versus acting.

Model revisions and complete settings are preserved in the [Qwen/Luna summary](arm_results/luna_full300_20261004/summary.json), [API actor summary](arm_results/reasoning_actors_full300_20261005/summary.json), [Jev/Kev selector summary](rl_results/sft-selector-full300-20261005.json), and [direct Kev summary](rl_results/kev27b-actor-full300-20261005.json).

<a id="scaling-comparisons"></a>
## Selection and reasoning scaling

| Comparison | Observed change | Interpretation |
| --- | --- | --- |
| Qwen + Luna N=5 vs Qwen alone | **+10.67pp**, paired 95% interval **[+5.67, +16.00]**; 2.23× serving cost | A measured selection benefit over one actor episode; not a cost-matched retry comparison |
| SFT + Luna N=10 vs N=5 | **−0.33pp [−5.33, +4.67]**; +19.9% serving cost; tie on 271 common-valid tasks | No observed benefit from doubling proposals; interval permits modest gains or losses |
| SFT + Luna N=5 vs Luna medium actor | +1.00pp [−4.67, +6.67]; 8.16× serving cost | No established success advantage; CPU/GPU cohorts also differ in collection time |
| SFT + Kev27B vs SFT + Jev | +2.67pp overall; **176 vs 175 successes on 279 common-valid tasks** | The eight-success overall gap is largely due to different invalid sets; no clear selector ranking |
| Luna high vs Luna medium | −4.33pp [−9.33, +0.67] | High reasoning did not improve observed success; medium was collected earlier |
| Sol high vs Luna high | −12.00pp [−17.33, −6.67] | A result under this harness and judge, with unresolved causal attribution |

Jev/Kev also include a mid-collection typing repair. On the 238 tasks collected after the repair and valid in both arms, Jev/Kev scores are **147/238 versus 149/238**. The original results and pre/post strata remain intact. Reused SFT is a historical reference, so subtracting 35.33% from any fresh selector score does not give a controlled selector effect. Task-paired intervals exclude website drift, judge error and harness bias.

<a id="luna-full300-results-20261006"></a>
## Cost, latency, tokens, and plots

The five-arm Qwen/Luna study has complete primary token receipts. Serving cost includes all proposals, selector calls and one reserved H200 throughout each local-model episode, including waits, at the frozen **$0.90/H200-hour** rate. It excludes judging, separately priced browser CPU, and research startup/idle overhead. API prices are frozen receipt-based estimates, not provider invoices. Tokens include cached/repeated prefixes and use different tokenizers.

| Policy | Overall | Mean serving $/task | Episode p50 / p95, s | Mean input / output tokens per task |
| --- | ---: | ---: | ---: | ---: |
| Qwen alone | 17.67% | 0.03991 | 120.7 / 389.0 | 167.0k / 9.0k |
| Qwen + Luna N=5 | 28.33% | 0.08885 | 224.1 / 677.4 | 941.2k / 50.2k |
| SFT + Luna N=5 | 37.67% | 0.06023 | 151.2 / 489.9 | 684.3k / 28.7k |
| SFT + Luna N=10 | 37.33% | 0.07219 | 178.3 / 525.4 | 1,419.1k / 57.5k |
| Luna medium actor | 36.67% | 0.00738 | 56.0 / 254.4 | 54.3k / 1.3k |

Luna medium is the cheapest measured policy in this table under the frozen prices. N=10 costs about twice the proposal tokens of N=5, despite little change in success. Hosted model FLOPs are unavailable; local analytic FLOP estimates cannot establish equal total compute across API and local policies.

![Qwen/SFT/Luna success versus estimated serving cost](arm_results/luna_full300_20261004/cost.png)

[Latency plot](arm_results/luna_full300_20261004/latency.png) · [Token plot](arm_results/luna_full300_20261004/tokens.png) · [Metrics CSV](arm_results/luna_full300_20261004/metrics.csv) · SVG: [cost](arm_results/luna_full300_20261004/cost.svg), [latency](arm_results/luna_full300_20261004/latency.svg), [tokens](arm_results/luna_full300_20261004/tokens.svg).

<a id="luna-cpu-family-results-20261005"></a>
<a id="api-actor-high-results-20261005"></a>
### API actor efficiency

| Actor | Overall | Actor API $/task | Mean episode seconds | Mean input / output tokens | Mean actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Luna medium | 36.67% | 0.00738 | 78.09 | 54,269 / 1,275 | 12.80 |
| Luna high | 32.33% | 0.01181 | 114.63 | 83,716 / 2,812 | 17.98 |
| Sol6.1 high | 20.33% | 0.15255–0.15475 | 90.27 | ≥59,211 / ≥535 | 13.25 |

Sol's range includes cache-price uncertainty and four HTTP 5xx reservations with unknown returned usage; its token means are lower bounds. Episode latency excludes the terminal judge. High versus medium changes reasoning effort, with possible collection-date effects. [API metrics and accounting](arm_results/reasoning_actors_full300_20261005/summary.json); plots: [cost](arm_results/reasoning_actors_full300_20261005/cost.png), [latency](arm_results/reasoning_actors_full300_20261005/latency.png), [tokens](arm_results/reasoning_actors_full300_20261005/tokens.png).

No harmonized serving-cost/latency point is asserted for Jev/Kev or the reused SFT baseline. Their total resource ledgers remain available, but campaign allocation cost and per-policy serving cost measure different quantities.

<a id="api-actor-reasoning-20261005"></a>
<a id="api-actor-stopping-audit-20261005"></a>
## GPT-6 actor and judge audit

Two adapter defects limit interpretation of the API-actor scores:

1. **Missing system prompt:** frozen source packaging omitted Markdown prompt assets, and the loader silently supplied an empty system message. Actors still received the task, screenshot and tool schemas, but no system-level browser-agent instruction. The same missing browser-policy text affected the local Qwen/SFT actors in the Luna study; their tokenizer still supplied a nonempty tool-schema wrapper. Luna's separately constructed selector prompt was not shown missing.
2. **Lost native conversation state:** the adapter rebuilt later turns as Qwen-style XML text, discarding native Responses reasoning items and tool-call IDs. Stateless native tool loops should preserve returned output items and pair execution feedback with the original call ID. See the [Responses reasoning guide](https://developers.openai.com/api/docs/guides/reasoning).

These are confirmed implementation problems, not measured explanations for a particular percentage-point loss. Working-tree fixes passed **97 offline tests** and three-turn replays of saved Luna-high and Sol-high responses: prompt assets are pinned and required, native state/call IDs persist, and only the current screenshot is replayed. Completed frozen sources and canonical verdicts stay unchanged. Live provider acceptance and any success-rate improvement remain unmeasured; a new, matched cohort is needed. First-action failures cannot be caused by loss of earlier-turn state. [Offline verification summary](arm_results/reasoning_actors_full300_20261005/pipeline-debug.json).

The saved-response audit found the expected model and high reasoning effort on all **5,393 Luna-high and 3,974 Sol-high returned responses**, with no incomplete output or output-cap hits. The corrected full300 cohorts do not show the old coordinate-remapping bug described below.

| Stopping behavior, all 300 tasks | Luna high | Sol high |
| --- | ---: | ---: |
| `done` on first action | 7 | 58 |
| Successes among first-action endings | 0 | 2 |
| First-action terminal text reports a site/verification block | 7 | 53 |
| `done` within first three actions | 29 | 100 |
| Any terminal `done` | 162 | 178 |

Sol often stops when blocked or asks for missing information through `done`, which ends an autonomous episode. Among ten Luna-only passes in Sol's first-action-ending subset, eight involved visible blocks in both initial screenshots and two involved location clarification. This selected review does not explain most of the overall gap or establish the block rate for every task.

**A shared judge does not guarantee equally strict outcomes.** The saved AgentTrek rubric permits credit for partial progress, including more than eight correct actions, one of two subtasks, or omitting a final save. One reviewed Luna pass still showed a final access error; the judge credited effective navigation. Such a rubric can favor continuing over stopping even when neither actor finishes the requested task. Its contribution to the score difference is unmeasured. Original scores are retained; no adjusted rate is substituted. [Aggregate stopping/judge audit](arm_results/reasoning_actors_full300_20261005/stopping-behavior.json).

Across families, hosted versus local browsers, DOM versus screenshot inputs, candidate presentation, typing behavior, episode limits and judge evidence all differ. The table is a record of implemented systems, not a clean ranking of underlying model capabilities.

<a id="learned-arm-and-retries"></a>
## Learned ARM versus episode retries

The separate October 4 controlled study collected **1,800 episodes**: one learned-ARM N=5 trajectory and five ordinary SFT trajectories on each of 300 tasks. Ordinary pass@k averages all k-subsets of the five recorded outcomes. It assumes an oracle success verifier; it is not a deployed selector that can identify the successful trajectory.

| Policy | Overall success | Mean browser-step calls/task |
| --- | ---: | ---: |
| Learned ARM, N=5 | 118/300 = 39.33% | 15.89 |
| Ordinary pass@1, pooled | 528/1,500 = 35.20% | 14.39 |
| Oracle episode pass@2 | 46.17% | 28.77 |
| Oracle episode pass@3 | 52.30% | 43.16 |
| Oracle episode pass@4 | 56.40% | 57.55 |
| Oracle episode pass@5 | 59.33% | 71.94 |

ARM minus pass@1 is **+4.13pp [−0.13, +8.40]**; ARM minus pass@5 is **−20.00pp [−26.00, −14.33]**. Under the observed KV-cache/vision bounds, pass@4 has higher oracle success and lower estimated model work than ARM, but uses 3.62× as many browser actions. Equal model compute and equal browser cost are different comparisons. [Controlled results and FLOP assumptions](ARM_INFERENCE.md#arm-controlled-inference-results-20261004) · [Aggregate data](arm_results/rl_integration/controlled-inference-20261004.json).

Earlier September inference results remain a separate historical cohort:

| Official SFT actor with… | Successes / 300 | Valid | Overall | Valid-only |
| --- | ---: | ---: | ---: | ---: |
| No selector | 90 | 267 | 30.00% | 33.71% |
| ScalarARM, N=5 | 114 | 251 | 38.00% | 45.42% |
| SelectionARM, N=5 | 128 | 256 | 42.67% | 50.00% |
| SelectionARM, N=5, action-only candidates | 105 | 253 | 35.00% | 41.50% |
| GPT-5.6 Sol selector, N=5 | 132 | 256 | 44.00% | 51.56% |

GPT-5.6 Sol here is a **selector**, not the GPT-6.1 Sol direct actor above. Its comparison with SelectionARM spans different collection dates; the common-valid paired difference is +2.98pp [−3.40, +9.79]. [Historical scores](arm_results/sol-selection300.json) · [Why historical and fresh ARM gains differ](ARM_INFERENCE.md#arm-historical-reconciliation-20261004).

<a id="compact-candidate-selection"></a>
The [action-only candidate ablation](arm_results/selection-actions-only.json) removes candidate reasoning from the selector input, reducing selector input tokens by 39.6%; the actor still generates all full proposals. It scored below full-reasoning SelectionARM, with historical/date and validity differences limiting attribution. This is a candidate-input ablation, distinct from actions-only **judge** evidence in the Luna study.

<a id="luna-qwen-inference-20261004"></a>
<a id="luna-qwen-pilot-partial-20261005"></a>
## Pilot results and recovery caveats

Small pilots motivated the full300 studies; they do not replace the full-set results.

| Pilot | Saved successes / episodes | Status / caveat |
| --- | --- | --- |
| Direct Jev / Kev0.8B / Kev27B | 1/10; 0/10; 3/10 | All valid; DOM operation/target policy with text helper |
| SFT alone / +Jev / +Kev0.8B / +Kev27B | 4/10; 3/10; 4/10; 9/10 | 9/6/10/10 valid; T=0.6; three questionable Kev27B positives annotated, not rescored |
| Qwen alone / +Luna N=5 / +Luna N=10 / SFT+Luna N=5 | 0/10; 1/10; 0/9; 2/10 | T=1.0, p=0.9; 49/50 total records including withdrawn Luna actor; pilot incomplete |
| Luna actor pilot | Comparison withdrawn | Seven of ten trajectories exposed to coordinate corruption; remaining three are not a matched control |

Pilot evidence: [Jev](rl_results/jev-ultrafast-pilot-20261004.json), [Kev pair](rl_results/kev-pair-pilot-20261004.json), [SFT selector pilot](RL_RESULTS.md#sft-decision-selection-20261004), [corrected Qwen/Luna common-cohort analysis](arm_results/luna_qwen_pilot_20261004/common-cohort-summary.json).

<a id="luna-pixel-coordinate-repair-20261005"></a>
**Repairs and retained evidence:** Luna's initial pixel coordinates were mistakenly interpreted as normalized coordinates. Its corrected full300 primary cohort contains 90 unaffected original episodes plus 210 corrected/fresh episodes; 177 compromised records remain archived outside it. Jev/Kev collection includes a platform-specific input-clearing repair with explicit pre/post strata. Jev context-limit recovery preserved all 189 earlier outcomes and continued the 111 untouched tasks; provider-halt invalids were not relabeled as model task failures. Direct Kev preserves every retry and its six diagnosed invalids. These are provenance notes, not extra primary episodes or rescored successes.

The ten-task pilots, historical September runs, fresh October controls and full300 selector studies have separate budgets and protocols. The incomplete pilot's proposed tail allocation remains unapproved; it is not needed to interpret the completed full300 cohorts.

<a id="evidence-and-accounting"></a>
## Evidence and accounting

| Study | Public aggregate evidence | All-attempt resource accounting |
| --- | --- | --- |
| Five-arm Qwen/SFT/Luna | [Summary](arm_results/luna_full300_20261004/summary.json), [CSV](arm_results/luna_full300_20261004/metrics.csv), [tracker](arm_results/luna_full300_20261004/experiment_tracker.json) | 76.61 H200-hours; CPU-only pool 12,257s; Luna $16.871675 charged/reserved; judge $4.784384 |
| Luna-high / Sol-high | [Summary](arm_results/reasoning_actors_full300_20261005/summary.json), [stopping audit](arm_results/reasoning_actors_full300_20261005/stopping-behavior.json) | Zero local GPUs; respective allocations 8,907s / 7,056s; actors $3.544138 / $46.425246; shared judge $1.412495 |
| SFT + Jev / Kev27B | [Summary and paired strata](rl_results/sft-selector-full300-20261005.json), [operational audit](RL_EVALUATION.md#sft-selection-full300-final-20261005) | 8.3789 / 16.1989 H200-hours, including failures |
| Direct Kev27B | [Summary and invalid diagnoses](rl_results/kev27b-actor-full300-20261005.json), [operational audit](RL_EVALUATION.md#kev27b-actor-full300-20261004) | 2.7519 H200-hours; 307 browsers; 8,567 Kev / 325 text / 295 judge requests |
| Controlled learned ARM / retries | [Results](arm_results/rl_integration/controlled-inference-20261004.json), [detailed cost model](ARM_INFERENCE.md#arm-controlled-inference-results-20261004) | Separate cohort and accounting; do not add its reused actor0 records to fresh episode totals |

Final artifact audits check saved rollouts, screenshots, verdicts, request/model identities, accounting and W&B closure. They establish collection integrity under each saved protocol; they do not certify every positive as strict semantic completion. All canonical verdicts and physical attempts remain preserved. Raw task payloads, screenshots and request logs stay private.

The long Qwen/Luna planning and recovery narrative was consolidated here from `ARM_INFERENCE.md`; its [pre-consolidation history](https://github.com/zixianma/OpenWebRL/blob/50c6ab6/openwebrl/docs/ARM_INFERENCE.md) remains available. [ARM_INFERENCE.md](ARM_INFERENCE.md) retains learned-ARM methods, detailed controlled cost analysis and judge/retry history. Future summaries belong in this document; aggregate report scripts continue to update the linked JSON/CSV/plots.
