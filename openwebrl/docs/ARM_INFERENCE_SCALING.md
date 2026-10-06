# ARM inference-time scaling: actor alone versus actor + selector

**Main actor: official `OpenWebRL/OpenWebRL-4B-SFT`.** We compare its standalone performance with selecting among its proposed actions. Released Qwen3-VL-4B-Thinking is an actor ablation; direct GPT-6 and Kev policies are secondary references.

All eleven recent full300 rows are complete: **3,000 fresh episodes plus 300 reused SFT reference episodes**, finalized October 6, 2026. **Overall includes all 300 tasks; valid-only excludes records marked invalid.** These are saved judge successes, including documented partial-progress allowances. The protocol labels matter: the existing SFT-alone reference is not matched to the fresh selector runs.

## Contents

- [OpenWebRL-SFT: alone versus +selector](#actor-selector-experiment-tracker-20261004)
- [Qwen3 actor ablation](#qwen-actor-ablation)
- [Other standalone actors](#api-actor-high-results-20261005)
- [Recommended matched SFT control](#matched-sft-control)
- [Protocol differences](#luna-actor-full300-20261004)
- [Efficiency details](#luna-full300-results-20261006)
- [Actor/judge audit](#api-actor-stopping-audit-20261005)
- [Learned ARM and episode retries](#learned-arm-and-retries)
- [Pilots and provenance](#luna-qwen-inference-20261004)

<a id="actor-selector-experiment-tracker-20261004"></a>
<a id="scaling-comparisons"></a>
## OpenWebRL-SFT: alone versus +selector

The actor checkpoint is the same in every row. **N** is the number of proposed next actions; one is executed per step.

<!-- actor-selector-results:start -->
<!-- cohorts: AS04, AS05, AS06, AS08, AS09 -->

| Selector | N | Overall success | Valid-only success | Protocol |
| --- | ---: | ---: | ---: | --- |
| None — historical reference | 1 | 35.33% (106/300) | 38.97% (106/272) | Historical; unmatched |
| GPT-6 Luna medium | 5 | 37.67% (113/300) | 41.24% (113/274) | Local browser |
| GPT-6 Luna medium | 10 | 37.33% (112/300) | 40.88% (112/274) | Local browser |
| Jev | 5 | 58.67% (176/300) | 61.97% (176/284) | Hosted browser |
| Kev27B | 5 | 61.33% (184/300) | 63.23% (184/291) | Hosted browser |

**What we can conclude:** Luna N=10 did not improve on N=5 (−0.33pp, paired 95% interval −5.33 to +4.67) and cost 19.9% more. Jev/Kev are nearly tied on the same 279 valid tasks: **175 versus 176 successes**; after the typing repair in both arms, **147 versus 149 on 238 tasks**.

**What is missing:** a matched SFT-only control. The 35.33% reference used T=0.7, p=0.9 and 1,024 output tokens; the selector runs use T=1.0, p=0.95 and 4,096. Browser, prompt, top-k and judging also differ across families. Neither the gain over the historical baseline nor the gap between Luna and Jev/Kev isolates selector quality.

<a id="qwen-actor-ablation"></a>
## Qwen3: actor ablation

Both rows use released `Qwen/Qwen3-VL-4B-Thinking` under the same local-browser protocol.

<!-- cohorts: AS01, AS02 -->

| Selector | N | Overall success | Valid-only success |
| --- | ---: | ---: | ---: |
| None | 1 | 17.67% (53/300) | 19.13% (53/277) |
| GPT-6 Luna medium | 5 | **28.33% (85/300)** | 30.80% (85/276) |

Selection adds **10.67pp** (paired 95% interval **+5.67 to +16.00**) at **2.23× serving cost**. This is the completed actor-alone versus actor+selector comparison; it supports a benefit for this Qwen actor, not a quantified gain for official SFT. Cost-matched episode retries were not tested in this pair.

<a id="luna-cpu-family-results-20261005"></a>
<a id="api-actor-high-results-20261005"></a>
## Other standalone actors: reference results

<details>
<summary>GPT-6 and Kev direct actors</summary>

<!-- cohorts: AS07, AS10, AS11, AS12 -->

| Actor, no selector | Overall success | Valid-only success | Interface |
| --- | ---: | ---: | --- |
| GPT-6 Luna medium | 36.67% (110/300) | 39.43% (110/279) | Screenshot / local browser |
| GPT-6 Luna high | 32.33% (97/300) | 35.02% (97/277) | Screenshot / local browser |
| GPT-6.1 Sol high | 20.33% (61/300) | 22.34% (61/273) | Screenshot / local browser |
| Kev27B | 8.67% (26/300) | 8.84% (26/294) | DOM / hosted browser; GPT-4.1-mini typing |

<!-- actor-selector-results:end -->

API actor results have confirmed missing-policy and native-history defects; see the [audit](#api-actor-stopping-audit-20261005). Luna high minus medium is −4.33pp [−9.33, +0.67]; Sol high minus Luna high is −12.00pp [−17.33, −6.67]. These are harness-specific results, not a model-capability ranking.

Luna medium costs $0.00738 per task versus $0.06023 for SFT + Luna N=5, whose success difference is only +1.00pp [−4.67, +6.67]; collection dates differ. Kev's 8.67% direct-policy result and 61.33% SFT-selector result describe different assembled systems and action interfaces.

</details>

Source: [eleven-row aggregate tracker](arm_results/luna_full300_20261004/experiment_tracker.json). All eleven cohorts appear once above; IDs remain in table comments. Qwen + Luna N=10 was excluded from full300 scope; Kev0.8B and direct Jev have pilots only. Paired intervals quantify task sampling, not website drift, harness bias or judge error.

<a id="matched-sft-control"></a>
## Recommended next control: official SFT alone

**Yes: a fresh N=1 SFT-only control is the missing comparison.** Use the identical released SFT checkpoint/revision, **T=1.0, top-p=0.95, 4,096 output tokens**, and the same 300 tasks. Match the browser, viewport, prompt, history, context/horizon limits, typing behavior, judge inputs and terminal rules as well as decoding. Generate one action and execute it without a selector.

**Recommended first comparison: SFT alone (N=1) and SFT + Kev27B (N=5), collected together in the hosted harness.** Use top-k=20, 1280×1000, 30 turns/600s, the repaired typing path, and the same thoughts/actions + final-screenshot o4-mini/AgentTrek judging, including step-limit dispatch. Jev can be an additional selector arm under that same protocol.

The hosted Jev/Kev sources also omitted the SFT browser-policy prompt. Restore and pin that prompt for **both** new arms. A corrected SFT-only run alone would still be unmatched to the old selector results; a fresh pair avoids changing the prompt only for the control and reduces collection-date drift. Preserve the earlier cohorts, including their pre/post typing strata, as historical results.

The Luna family needs a **separate local control**: top-k disabled, 1280×720, 30 actions/1,800s, actions-only judge evidence and step-limit zero. Its saved actors lacked the browser-policy prompt. A control using the corrected prompt alone would also change the harness; for a clean corrected comparison, collect SFT alone and SFT + Luna together under the same corrected protocol. One new baseline cannot match both families.

This is a recommended design, not a submitted run. Exact compute/API/browser budgets must be approved before a new allocation. Report paired task differences and both denominators; keep original scores and invalids intact.

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

<a id="luna-full300-results-20261006"></a>
## Cost, latency, tokens, and plots

The five-arm Qwen/Luna study has complete primary token receipts. Serving cost includes all proposals, selector calls and one reserved H200 throughout each local-model episode, including waits, at the frozen **$0.90/H200-hour** rate. It excludes judging, separately priced browser CPU, and research startup/idle overhead. API prices are frozen receipt-based estimates, not provider invoices. Tokens include cached/repeated prefixes and use different tokenizers.

<details>
<summary>Detailed efficiency tables and plots</summary>

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

### API actor efficiency

| Actor | Overall | Actor API $/task | Mean episode seconds | Mean input / output tokens | Mean actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Luna medium | 36.67% | 0.00738 | 78.09 | 54,269 / 1,275 | 12.80 |
| Luna high | 32.33% | 0.01181 | 114.63 | 83,716 / 2,812 | 17.98 |
| Sol6.1 high | 20.33% | 0.15255–0.15475 | 90.27 | ≥59,211 / ≥535 | 13.25 |

Sol's range includes cache-price uncertainty and four HTTP 5xx reservations with unknown returned usage; its token means are lower bounds. Episode latency excludes the terminal judge. High versus medium changes reasoning effort, with possible collection-date effects. [API metrics and accounting](arm_results/reasoning_actors_full300_20261005/summary.json); plots: [cost](arm_results/reasoning_actors_full300_20261005/cost.png), [latency](arm_results/reasoning_actors_full300_20261005/latency.png), [tokens](arm_results/reasoning_actors_full300_20261005/tokens.png).

No harmonized serving-cost/latency point is asserted for Jev/Kev or the reused SFT baseline. Their total resource ledgers remain available, but campaign allocation cost and per-policy serving cost measure different quantities.

</details>

<a id="api-actor-reasoning-20261005"></a>
<a id="api-actor-stopping-audit-20261005"></a>
## GPT-6 actor and judge audit

The prompt defect affects SFT/Qwen proposal actors as well as API actors; the native-history defect is specific to the API actor adapter:

1. **Missing system prompt:** frozen source packaging omitted Markdown prompt assets, and the loader silently supplied an empty system message. Actors still received the task, screenshot and tool schemas, but no system-level browser-agent instruction. The same missing browser-policy text affected local Qwen/SFT actors in the Luna study and hosted SFT proposers in the Jev/Kev study; local actors still received their tokenizer tool-schema wrapper. The hosted frozen sources also omit the required Markdown file, and their loader falls back to an empty policy. This finding concerns the actor policy, not the separately constructed selector prompts.
2. **Lost native conversation state:** the adapter rebuilt later turns as Qwen-style XML text, discarding native Responses reasoning items and tool-call IDs. Stateless native tool loops should preserve returned output items and pair execution feedback with the original call ID. See the [Responses reasoning guide](https://developers.openai.com/api/docs/guides/reasoning).

These are confirmed implementation problems, not measured explanations for a particular percentage-point loss. Working-tree fixes to the Luna/API pipeline passed **112 offline tests** and three-turn replays of saved Luna-high and Sol-high responses: prompt assets are pinned and required, native state/call IDs persist, and only the current screenshot is replayed. The corrected protocol has a separate version and prompt hashes, and workers reject mismatched identities. Completed frozen sources and canonical verdicts stay unchanged. Live provider acceptance and any success-rate improvement remain unmeasured; a new, matched cohort is needed. The newly identified hosted SFT path still needs equivalent prompt-packaging/preflight validation before a fresh launch. First-action failures cannot be caused by loss of earlier-turn state. [Offline verification summary](arm_results/reasoning_actors_full300_20261005/pipeline-debug.json).

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

Across families, hosted versus local browsers, DOM versus screenshot inputs, candidate presentation, typing behavior, episode limits and judge evidence all differ. The tables record implemented systems, not a clean ranking of underlying model capabilities.

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
