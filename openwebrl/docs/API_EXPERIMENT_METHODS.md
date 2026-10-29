# API browser evaluation methods

This document describes how the API actor and candidate-selection experiments are run, measured and audited. The central comparison is **one model proposal per decision versus several proposals plus a selector, with only the selected proposal executed**. It is an inference-policy comparison, not pass@N independent browser rollouts or a fixed-compute comparison. Reasoning effort and output caps are requested controls; realized tokens and cost are measured separately.

[Results and figures](ARM_FINAL_RESULTS.md) · [Detailed study history and uncertainty](ARM_INFERENCE_SCALING.md) · [Code and frozen artifacts](#code-and-frozen-artifacts)

## Study families and controlled settings

| Study | Arms | Task coverage | Actor/selector limits | Judge and reporting boundary |
|---|---|---|---|---|
| GPT high, BrowserCode | Sol single; Luna single, N3, N5; all high | OM2W 300, WebVoyager 595, DeepShop 150 per method | 12,288 output tokens; 30 decisions; selected-only execution | Historical o4-mini terminal-success cohort; keep its judge/date distinct from Terra studies |
| GPT high, native CUAWright | Sol single; Luna single, N3, N5; all high | Same OM2W 300-task roster per method | 20,000 output tokens; 100 decisions; summary setting 20; high selector for N3/N5 | Native protocol and helper work retained; report fresh200 separately from reused100 |
| Claude, BrowserCode | Opus 5.5 single; Sonnet 5.5 single, N3, N5; all high | Fresh OM2W 300 per arm, 1,200 final dispositions | 12,288 output tokens per candidate/selector; 30 decisions | GPT-5.6 Terra, frozen terminal-success protocol; completed and independently audited |
| GPT Luna effort, BrowserCode | Single xhigh/medium/low; N5 medium+medium and low+low | Fresh OM2W 300 per arm targeted, 1,500 total | 12,288 output tokens per candidate/selector; 30 decisions | Same Terra protocol as Claude; ongoing, pilot outcomes are preliminary |

The BrowserCode three-benchmark figure and the OM2W two-harness figure answer different questions and remain separate. Cross-harness comparisons are descriptive: the native harness has a larger decision/output budget, different command semantics and timing boundaries. Cross-model comparisons also have different live-site collection dates. A common task roster alone does not remove those differences.

## Task selection and episode execution

Each campaign freezes its task roster and arm/shard assignments before bulk collection. The same fixed task IDs are used across its arms; each arm has its own browser trajectory. Once actions diverge, later observations naturally differ. Pilot cases are part of the final fixed roster, not additional cherry-picked demonstrations. The native extension preserves the earlier 100 outcomes, including invalids, for Luna single/N3 and Sol single, adds 200 fresh outcomes to each, and collects all 300 N5 outcomes fresh. This produces 900 new outcomes and 1,200 final dispositions.

The recent BrowserCode workers start owned local Chromium/CDP browsers, open the task's actual start URL, pin a 1280×1800 viewport, prepare the workspace and run a mechanical startup check without model calls. BrowserCode executes generated code through its browser bridge; native CUAWright executes its native browser-session commands. The recent BrowserCode workspace command timeout is 120 seconds. These are campaign settings, not assumptions about every historical run; consult the pinned worker and runtime configuration for each cohort.

The agent receives the task, observable execution history and current browser evidence. Each decision produces either an executable command or a final answer. The final browser image is captured and validated before external terminal grading. Browser setup, final judge-only capture and cleanup are recorded separately from actor-loop latency.

## Candidate generation, selection and history

For N=1, the one parsed actor response supplies the next command or completion. For N=3 or N=5:

1. Generate N proposals concurrently from identical actor input at that decision.
2. Parse every proposal under the frozen action-or-final-answer schema. A malformed proposal invalidates the decision; it is not silently dropped and replaced.
3. Give the selector the task, actual observable history and proposed commands/completions. Do not provide unexecuted tool outcomes or candidate private reasoning.
4. Require an index-only selection in the fixed range 0…N−1.
5. Execute only the selected command. Continue the actor history with the selected native response and actual execution result; preserve its provider-native reasoning/signature representation where required.

Unselected proposals are saved for audit but never executed or inserted as fabricated successful actions. N5 therefore normally entails five candidate requests plus one selector request per decision. Its API cost must include all six, not just the winner. Terminal-answer decisions, executable batches and internal browser operations are distinct counts.

## Native API adapters

| Adapter | Native request controls | Response and transport checks |
|---|---|---|
| GPT Luna | Responses API; exact `gpt-6-luna`; per-role `reasoning.effort`; `max_output_tokens=12288`; `store=false`; default service tier; native action tool, index schema for selection | Validate returned model/status/effort and strict actor/selector structure; preserve selected native history; disable SDK retries |
| Claude | Messages API; exact `claude-opus-5-5` or `claude-sonnet-5-5`; adaptive thinking; `output_config.effort=high`; `max_tokens=12288`; one action tool or selector JSON schema | Validate exact returned model, content/stop semantics and action-vs-answer structure; preserve native selected content/signatures; disable SDK retries |

Every physical request gets a stable ID, role, decision-group ID, payload hash, reservation, dispatch/response timestamps, native response or error and usage accounting. Roles distinguish candidates, selectors, input counters, helpers and external judges. Model aliases are checked against the requested identity but are not immutable provider weight snapshots. The campaign manifest, source hashes and saved responses define the actual experiment.

Input bounds determine conservative admission reservations. The Luna adapter uses its native input-token counter only when the conservative local bound exceeds its frozen threshold; counter requests have their own saved identities and call accounting. The Claude campaign uses conservative text/image bounds and has zero native input-counter calls. Neither implementation assumes a missing response was free.

## External terminal-success grading

Recent Claude and Luna effort runs use `gpt56-terra-terminal-success-v1`: exact `gpt-5.6-terra`, medium effort, 4,096 completion-token cap and seed 42. The frozen request combines task, selected action history and final screenshot. The parser requires a completed, non-refusal terminal verdict under the frozen prompt/schema. The exact prompts are loaded by `common_judge.py`; prose in this document does not replace them.

External judging is excluded from model-policy cost and native task latency, but included in a separate all-attempt judge ledger. Internal online helper/model work remains part of policy accounting. Earlier o4-mini/action-history grading and later Terra grading are separate cohorts. Historical Claude judge-migration/recovery attempts and their fees remain in lineage; a current valid result requires the qualified final judge identity. The Luna effort campaign does not replay an already attempted actor episode to improve its result.

## Outcomes, metrics and fair denominators

| Quantity | Definition | Required qualification |
|---|---|---|
| Valid-only success | Successful valid outcomes / all valid outcomes in that arm | Report successes, valid count and invalid count separately; invalids are not failures in this denominator |
| Common-valid comparison | Restrict every compared arm to the same task intersection valid in all of them | Label the intersection size; it is a selected cohort and does not remove cap/site/format selection bias |
| Model cost per task | Sum candidate, selector and applicable helper/counter charges for the specified task cohort | Include valid failures; distinguish plotted cohort means from all-attempt spending; retain unknown reserves |
| Browser batches | Selected executable BrowserCode blocks | A block can contain several browser operations; this is not a standardized action count |
| Native command volume | Saved native browser-command log entries | Do not equate these one-for-one with BrowserCode blocks or CDP calls |
| BrowserCode native task latency | Actor-loop wall time, including model admission/HTTP waits and selected browser execution | Exclude initial setup, final judge-only capture, cleanup and external judging; retain coverage counts |
| Tokens | Provider-reported input/output/cache usage, by physical ID and role | Output can include reasoning; do not add reasoning tokens to output a second time |

The completed Claude figures use each arm's valid cohort for both success and x-axis means; the common-valid alternative uses the identical 246-task cohort for all four metrics. Older cross-harness plots can use all observed valid and invalid task measurements for x-axis means, with their explicit coverage columns. These definitions must not be silently mixed. Missing latency, usage or command counts are never filled with zero.

Cost repricing uses the campaign's pinned rates and native cache fields, including cache-read/write distinctions and applicable long-context tiers. Unknown usage is charged conservatively at the retained full reservation. API-only spend and allocated CPU time are different quantities and are reported separately.

Observed Pareto lines connect nondominated aggregate points for success versus cost, browser volume or latency. They do not imply tested intermediate configurations, statistical superiority or uncertainty bounds. Paired uncertainty and primary/follow-up cohort definitions remain in the study-specific results records.

<details>
<summary>Validity, failure handling and recovery</summary>

Keep task failure distinct from invalid execution or grading. A valid terminal judge failure remains a valid failure. Transport errors, malformed native output, missing/invalid grading, accounting failures, interrupted execution and unverified cleanup remain explicit invalid/incomplete dispositions under the frozen worker rules. A conservative request reservation can fail admission before the USD ledger is exhausted; its cap-related invalid and sibling shared-halt consequences remain visible.

On an error, stop further dispatch into the affected shared pool, retain raw requests/responses and all charges, inspect scheduler/all-step accounting and owned teardown, then test a targeted recovery. Do not retry an unchanged failed episode, replace the model, loosen the parser or reinterpret an invalid as a success. Current routine recovery resumes only untouched task IDs, skips every existing valid/invalid result byte-for-byte, and keeps the original logical shard's fee/call/browser/time ledger. Archive an exact diagnosed halt only after reconciliation and a reviewed repair receipt. Never reset a shard's full budget on replacement.

The Luna effort campaign's initial shared-file HTTP gate had a measured concurrency violation. Those original attempts remain separately disclosed; they are not certified compliant retroactively. The repaired gate uses eight atomic directory slots, immutable owner identities and verified release; it passed offline tests and a two-node zero-API mechanical probe before replacement collection. Do not infer a measured peak of eight merely from a configured capacity of eight.

</details>

<details>
<summary>Staging, resources and budget enforcement</summary>

Each recent campaign requires independent audit of two explicit fresh canaries per arm before pilot release. A 20-task pilot per arm and an unconditional cost projection precede that arm's bulk release. The Luna projection is predeclared: 100,000 paired bootstrap samples of future 10-task shards, seed 20261028, higher 95th percentile bounded below by the point forecast, retaining unassigned overhead and invalid/unknown spend. A favorable success-conditioned estimate cannot authorize bulk.

Claude uses 120 logical CPU shards; Luna effort uses 150. Each shard is 0 GPU / 4 CPU / 16 GiB with a 4-hour all-attempt wall-time cap, two episode workers and six released jobs globally/two per arm. The Luna eight-model-HTTP limit is campaign-wide; the Claude frozen manifest specifies eight per job. These scopes differ and must not be conflated. Historical native full300 uses its separately approved90-shard,8-hour allocation profile.

Claude's approved financial amendment raises the initial model cap from $1,800 to $1,996, leaving the $300 judge cap unchanged: final combined cap $2,296. Its completed all-attempt conservative spend is $901.048412 model + $14.7273831 judge = $915.7757951. The source scientific plan and effective financial approval are separate artifacts; inspect both. Luna effort remains capped at $1,050 model + $375 judge = $1,425; its $70 + $25 pilot is included, not additional. Shard pools are frozen and cannot transfer unused balance.

A batch controller owns and awaits its episode workers. Scheduler state alone is insufficient: verify saved outcomes, all-step wall time, ledger IDs/settlements, browser starts, W&B termination and recorded owned process-group teardown. A quiet watcher queues an active-agent repair callback on meaningful changes/stalls and periodically while work is active. It is a continuation mechanism, not a substitute for independent final audit.

</details>

## Code and frozen artifacts

The exact recent executables are frozen campaign snapshots in the local evaluation runtime, rather than the current repository worktree. Set `EVAL_ROOT` to the local `evaluations/serial-vs-parallel-reasoning-20261013` directory. The directories below are relative to it; paths are navigation pointers, not commands to launch a new paid allocation.

| Campaign directory | Authoritative sources and records |
|---|---|
| `claude-browsercode-opus-sonnet-single-n3-n5-om2w300-20261028-v1/` | `claude_model.py`, `worker.py`, `environments.py`, `common_judge.py`, `episode_metrics.py`, `claude_budget.py`, `controller.py`, `financial_plan.py`, `orchestration/launch_campaign.py` |
| `luna-browsercode-effort-single-xhigh-medium-low-n5-medium-low-om2w300-20261028-v1/` | `luna_model.py`, `worker.py`, `environments.py`, `common_judge.py`, `episode_metrics.py`, `request_budget.py`, `controller.py`, `financial_plan.py`, `orchestration/launch_campaign.py` |
| `cuawright-native-luna-single-n3-n5-sol-single-local300-20261024-preparation-v1/` | Native full300 runtime and frozen source references; use its approval/registry/audit artifacts and the [native methods record](ARM_INFERENCE_SCALING.md#cuawright-native-full300-four-methods-20261025) |

For each recent campaign start with `campaign-plan.json`, `runtime-config.json`, `source-manifest.json`, effective launch/financial approvals, `orchestration/registry.json` and `current.json`. Follow manifest hashes to imported source dependencies; copying only an adapter is not a complete reproduction. Worker imports bind the pinned CUAWright `DefaultAgent`, local workspace/browser implementation and shared parsing/selection helpers. Native full300 uses its own artifact layout; do not assume its plan is named `campaign-plan.json`.

Within a logical shard, `attempts/attempt-*/episodes/` holds saved decisions, requests, results and browser evidence; `budget/` holds cumulative ledgers. Controller registration/launch/teardown receipts bind physical attempts to the logical shard. `analysis/` holds independent audits; `repairs/` holds diagnosis, tests and narrowly scoped recovery qualification. `completion.json` is authoritative only after its bound independent audits pass. Source-manifest SHA256 identifiers for recent execution snapshots are:

- Claude: `0704226805abb2f0ee6e33647a5fd11ce1557e3a789d99bc5874694ea976da9f`; confirm the final audit binds the same execution manifest.
- Luna effort repaired source: `037cc573b2f2f6107c69c2353298971b299e41534a27c3eecfa16f33542b8d25`.

Earlier repository code is available locally at `openwebrl/sol_serial_parallel_policy.py` (selection and observable selector input), `openwebrl/sol_serial_parallel_transport.py` (native actor transport/history), `openwebrl/sol_serial_parallel_eval.py` (metered API evaluation), and `scripts/run_sol_serial_parallel_om2w.py` (entry point). These are historical/local development pointers; they are not a claim that the current remote branch contains the exact frozen adapters above.

<details>
<summary>Validation and reproducibility checklist</summary>

Before dispatch: verify roster/hash, exact actor/selector model identities and effort, endpoint/payload/response parsing, frozen browser/judge setup, selected-only history, W&B project `openwebrl-evals`, zero hidden SDK retries, cumulative caps and active-job limits. Offline tests cover parsers, pricing/cache bounds, strict native history, concurrency ownership, source admission and preservation of prior outcomes. Mechanical browser/transport tests do not establish model quality.

Before bulk: independently verify the two canaries, exact 20 pilot dispositions and unconditioned cost projection against every remaining shard pool. Before completion: join every stable physical ID to native receipts/ledgers, distinguish prepared from admitted/dispatched calls, verify all attempts terminal and within cumulative time/resources, retain unknown reservations, check observed metric coverage and owned teardown, then bind the aggregate and plots to audit hashes. Do not claim a fresh cluster-wide orphan scan from recorded private-process-group cleanup alone.

Private task-level payloads, screenshots, raw trajectories and credentials remain local. Public artifacts are aggregate results, plots, definitions and provenance; scientific completion and publication completion are separate statuses.

</details>

## Results and analysis pointers

| Question | Result/analysis location |
|---|---|
| GPT high across three benchmarks | [Headline results](ARM_FINAL_RESULTS.md#headline-performance-cost-and-efficiency); [aggregate](arm_results/sol6_luna_three_benchmarks_20261019/aggregate.json) |
| BrowserCode versus native on OM2W | [Harness comparison](ARM_FINAL_RESULTS.md#online-mind2web-only-browsercode-and-native-cuawright); [aggregate with coverage](arm_results/cuawright_browsercode_om2w_20261027/om2w_harness_comparison_valid_only.json) |
| Completed Claude 300 per arm | [Result table and plots](ARM_FINAL_RESULTS.md#claude-browsercode-om2w300); [aggregate](arm_results/claude_browsercode_om2w300_20261028/aggregate.json); [accounting](arm_results/claude_browsercode_om2w300_20261028/accounting.json); [plot source](arm_results/claude_browsercode_om2w300_20261028/plot_claude_tradeoffs.py) |
| Claude raw final qualification, local | `analysis/independent-full1200-final-audit-v1/audit.json`, `analysis/independent-full1200-final-audit-v1/independent-final-qualification-v1.json`, `analysis/native-metric-inputs-final-v4/aggregate-inputs.json`, `completion.json` inside its campaign |
| Ongoing GPT effort study, local | `current.json`, `orchestration/registry.json`, `analysis/pilot-cost-projection-predeclaration-v1.json`, independent pilot audits and repair receipts inside its campaign; no full300 result claimed |

The completed Claude study records 1,200 dispositions across 142 physical attempts. On the 246 common-valid tasks, Sonnet single and N5 each solved 147; the observed N5 setup did not establish a success advantage. For earlier GPT results and all effect/uncertainty claims, use the linked primary cohort tables rather than mixing denominators or judging protocols. Live pilot counts belong in campaign status records until independently qualified, not as a final methods result.
