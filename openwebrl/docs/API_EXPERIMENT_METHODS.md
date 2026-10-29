# API browser evaluation methods

We compare **one proposed action per decision** with **N proposed actions plus a selector**. Only the selected action executes. This tests whether spending more model work on each decision improves browser-task success, cost, browser activity or latency.

The main results are in [ARM_FINAL_RESULTS.md](ARM_FINAL_RESULTS.md); study-specific comparisons and uncertainty are in [ARM_INFERENCE_SCALING.md](ARM_INFERENCE_SCALING.md). This document explains the shared procedure and the differences that matter for interpreting those results.

[Experiments](#experiments) · [Episode procedure](#episode-procedure) · [Metrics](#metrics-and-comparability) · [Code](#code-and-frozen-artifacts) · [Results](#results-and-analysis-pointers) · [Audit details](#audit-and-operational-details)

## Experiments

| Study | Methods | Tasks per method | Output cap / decision limit | Terminal judge |
|---|---|---|---|---|
| GPT high, BrowserCode | Sol single; Luna single, N3, N5; all high | OM2W 300; WebVoyager 595; DeepShop 150 | 12,288 / 30 | Historical o4-mini cohort |
| GPT high, native CUAWright | Sol single; Luna single, N3, N5; all high | OM2W 300 | 20,000 / 100; summary setting 20 | Separate native protocol; see its methods record |
| Claude, BrowserCode | Opus 5.5 single; Sonnet 5.5 single, N3, N5; all high | OM2W 300; completed | 12,288 / 30 | GPT-5.6 Terra |
| GPT Luna effort, BrowserCode | Single xhigh/medium/low; N5 medium+medium and low+low | OM2W 300 targeted; ongoing | 12,288 / 30 | GPT-5.6 Terra |

The output cap applies to each candidate and selector request. In N5 medium+medium, both candidates and selector use medium effort. Effort and token caps do not enforce equal realized compute; we measure actual usage.

Each campaign fixes its task roster and shard assignments. Arms share task IDs but run independent browser trajectories. Pilot tasks count toward the final roster. The native extension reuses the earlier 100 outcomes, including invalids, for Sol single and Luna single/N3, adds 200 fresh tasks to each, and collects all 300 N5 tasks fresh. Its fresh-200 comparison is reported separately.

## Episode procedure

1. **Start the browser.** Recent BrowserCode runs use owned local Chromium/CDP, the task's start URL, a 1280×1800 viewport and a 120-second command timeout. A mechanical startup check uses no model calls.
2. **Generate proposals.** The actor receives the task, actual execution history and current browser evidence. N=1 generates one response; N3/N5 generate concurrent responses from identical input at that decision.
3. **Select one proposal.** For N3/N5, the selector sees observed history and proposed commands/completions. It returns one index. It does not see private candidate reasoning or fabricated outcomes of unexecuted proposals.
4. **Execute and continue.** Only the selected command executes. Only its native response and actual execution result enter subsequent actor history. Preserve provider-native reasoning/signatures where required. A final answer ends the actor loop.
5. **Grade the terminal state.** Capture and validate the final screenshot, then call the external terminal-success judge.

Every proposal must pass the frozen action-or-answer parser. A malformed proposal is not silently discarded or replaced. Save all proposals and selections, including unselected candidates. N5 normally costs five candidate calls plus one selector call per decision; it is not five independent browser rollouts or a fixed-cost comparison.

<details>
<summary>Exact API requests and judge settings</summary>

| Adapter | Request settings | Validation |
|---|---|---|
| GPT Luna | Responses API; `gpt-6-luna`; per-role `reasoning.effort`; `max_output_tokens=12288`; `store=false`; default service tier | Exact returned model/status/effort; strict action or selector schema; selected native history |
| Claude | Messages API; `claude-opus-5-5` or `claude-sonnet-5-5`; adaptive thinking; high effort; `max_tokens=12288` | Exact returned model and content/stop semantics; strict action or selector schema; selected native content/signatures |

SDK retries are disabled. Every physical call has a stable ID, role, payload hash, reservation, timestamps, response/error and usage record. Provider model aliases are checked but do not identify immutable weight snapshots.

Recent Claude and Luna effort runs use `gpt56-terra-terminal-success-v1`: `gpt-5.6-terra`, medium effort, 4,096 completion tokens and seed 42. The request contains the task, selected action history and final screenshot. `common_judge.py` loads the frozen prompts and validates the completed, non-refusal verdict. Earlier o4-mini and later Terra results remain separate judge cohorts. Historical Claude grading/recovery attempts and fees remain in the lineage.

</details>

## Metrics and comparability

| Metric | Definition and reporting rule |
|---|---|
| Valid-only success | Successes / valid outcomes. Report success, valid and invalid counts separately; invalids are excluded. |
| Common-valid comparison | Use the same task intersection valid in every compared arm; label its size. |
| Model cost | All candidate, selector and applicable helper/counter fees in the stated cohort, including valid failures. Exclude the external judge. |
| Browser batches | Selected executable BrowserCode blocks. One block can contain several browser operations. |
| Native command volume | Native browser-command log entries; not interchangeable with BrowserCode blocks or CDP calls. |
| BrowserCode task latency | Actor-loop wall time, including model admission/HTTP waits and browser execution. Exclude setup, final judge-only capture, cleanup and external judging. |
| Tokens | Native input/output/cache usage. Output may already include reasoning tokens; do not count those twice. |

The Claude plots use each arm's valid cohort for success and all efficiency means. Their common-valid alternative uses the same 246 tasks for every metric. Older cross-harness plots use observed measurements from valid and invalid tasks for some efficiency means; their coverage tables identify the denominators. Missing measurements are never zero-filled. Plotted cohort costs and all-attempt campaign spending are separate quantities.

**Interpretation limits:** different valid subsets can change apparent rankings. Common-valid filtering does not remove selection from site, format or budget failures. Harness budgets, command semantics, judge protocols and live-site dates also differ. Treat cross-harness/model comparisons as descriptive where those controls differ. Pareto lines connect observed nondominated points; they do not supply uncertainty or prove superiority.

## Code and frozen artifacts

[Published API source index](../api_experiments/README.md) · [GPT Luna adapter](../api_experiments/browsercode/models/luna_model.py) · [Claude adapter](../api_experiments/browsercode/models/claude_model.py) · [Native proposal adapter](../api_experiments/cuawright/models/native_proposal_adapter.py). The published files match the original source bytes; the local runtime still supplies configuration and pinned dependencies.

The exact recent executables are **frozen runtime snapshots**, not necessarily the current repository worktree. Under the local `evaluations/serial-vs-parallel-reasoning-20261013/` directory, use these campaign folders:

| Campaign | Runtime directory |
|---|---|
| Claude | `claude-browsercode-opus-sonnet-single-n3-n5-om2w300-20261028-v1/` |
| Luna effort | `luna-browsercode-effort-single-xhigh-medium-low-n5-medium-low-om2w300-20261028-v1/` |
| Native GPT full300 | `cuawright-native-luna-single-n3-n5-sol-single-local300-20261024-preparation-v1/` |

Within the Claude and Luna folders:

| Component | Code / records |
|---|---|
| API actor and selector | `claude_model.py` or `luna_model.py` |
| Episode, browser and metrics | `worker.py`, `environments.py`, `episode_metrics.py` |
| External judge | `common_judge.py` |
| Budget and HTTP limits | `claude_budget.py`; Luna also uses `request_budget.py` |
| Job ownership and admission | `controller.py`, `financial_plan.py`, `orchestration/launch_campaign.py` |
| Frozen configuration and lineage | `campaign-plan.json`, `runtime-config.json`, `source-manifest.json`, effective approvals and `orchestration/registry.json` |
| Status and qualification | `current.json`, `analysis/`, `repairs/`, qualified `completion.json` |

Follow `source-manifest.json` to imported dependencies, including the pinned CUAWright `DefaultAgent` and shared helpers. Copying an adapter alone is not a complete reproduction. Native full300 has its own artifact layout; follow its [methods record](ARM_INFERENCE_SCALING.md#cuawright-native-full300-four-methods-20261025).

Earlier local repository implementations are `openwebrl/sol_serial_parallel_policy.py` (selection), `sol_serial_parallel_transport.py` (native history), `sol_serial_parallel_eval.py` (evaluation), and `scripts/run_sol_serial_parallel_om2w.py` (entry point). These are development/history pointers, not substitutes for the frozen adapters.

<details>
<summary>Source hashes and saved evidence</summary>

Execution source-manifest SHA256:

- Claude: `0704226805abb2f0ee6e33647a5fd11ce1557e3a789d99bc5874694ea976da9f`.
- Luna effort repaired source: `037cc573b2f2f6107c69c2353298971b299e41534a27c3eecfa16f33542b8d25`.

A logical shard's `attempts/attempt-*/episodes/` contains decisions, physical requests, results and browser evidence; `budget/` contains cumulative ledgers. Registration and teardown receipts bind attempts to shards. Independent audits bind final aggregates to source/evidence hashes. A completion pointer is authoritative only after those audits pass. Private task payloads, screenshots, trajectories and credentials remain local; public results contain aggregates and provenance.

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

## Audit and operational details

<details>
<summary>Validity, failure handling and recovery</summary>

Keep task failure distinct from invalid execution or grading. A valid terminal judge failure remains a valid failure. Transport errors, malformed native output, missing/invalid grading, accounting failures, interrupted execution and unverified cleanup remain explicit invalid/incomplete dispositions under the frozen worker rules. A conservative request reservation can fail admission before the USD ledger is exhausted; its cap-related invalid and sibling shared-halt consequences remain visible.

On an error, stop further dispatch into the affected shared pool, retain raw requests/responses and all charges, inspect scheduler/all-step accounting and owned teardown, then test a targeted recovery. Do not retry an unchanged failed episode, replace the model, loosen the parser or reinterpret an invalid as a success. Current routine recovery resumes only untouched task IDs, skips every existing valid/invalid result byte-for-byte, and keeps the original logical shard's fee/call/browser/time ledger. Archive an exact diagnosed halt only after reconciliation and a reviewed repair receipt. Never reset a shard's full budget on replacement.

The Luna effort campaign's initial shared-file HTTP gate had a measured concurrency violation. Those original attempts remain separately disclosed; they are not certified compliant retroactively. The repaired gate uses eight atomic directory slots, immutable owner identities and verified release; it passed offline tests and a two-node zero-API mechanical probe before replacement collection. Do not infer a measured peak of eight merely from a configured capacity of eight.

</details>

<details>
<summary>Staging, resources and budget enforcement</summary>

Each recent campaign requires independent audit of two explicit fresh canaries per arm before pilot release. A 20-task pilot per arm and an unconditional cost projection precede that arm's bulk release. The Luna projection is predeclared: 100,000 paired bootstrap samples of future 10-task shards, seed 20261028, higher 95th percentile bounded below by the point forecast, retaining unassigned overhead and invalid/unknown spend. A favorable success-conditioned estimate cannot authorize bulk.

Claude uses 120 logical CPU shards; Luna effort uses 150. Each shard is 0 GPU / 4 CPU / 16 GiB with a 4-hour all-attempt wall-time cap, two episode workers and six released jobs globally/two per arm. The Luna eight-model-HTTP limit is campaign-wide; the Claude frozen manifest specifies eight per job. These scopes differ and must not be conflated. Historical native full300 uses its separately approved 90-shard, 8-hour allocation profile.

Claude's approved financial amendment raises the initial model cap from $1,800 to $1,996, leaving the $300 judge cap unchanged: final combined cap $2,296. Its completed all-attempt conservative spend is $901.048412 model + $14.7273831 judge = $915.7757951. The source scientific plan and effective financial approval are separate artifacts; inspect both. Luna effort remains capped at $1,050 model + $375 judge = $1,425; its $70 + $25 pilot is included, not additional. Shard pools are frozen and cannot transfer unused balance.

A batch controller owns and awaits its episode workers. Scheduler state alone is insufficient: verify saved outcomes, all-step wall time, ledger IDs/settlements, browser starts, W&B termination and recorded owned process-group teardown. A quiet watcher queues an active-agent repair callback on meaningful changes/stalls and periodically while work is active. It is a continuation mechanism, not a substitute for independent final audit.

</details>

<details>
<summary>Cost accounting and final checks</summary>

Reprice native usage with the pinned rates, cache-read/write fields and applicable long-context tiers. Retain full reservations when usage is unknown. Luna uses the native input-token counter only above its conservative local-bound threshold; counters have their own identities and call accounting. Claude uses conservative text/image bounds and has zero native input-counter calls. Internal helper work stays in policy cost; external judge fees have a separate ledger.

Before dispatch, verify model/effort, roster/source hashes, browser/judge setup, selected-only history, disabled SDK retries, remaining caps and W&B project `openwebrl-evals`. Before bulk, audit the two canaries, exact 20-task pilot and unconditional projection. Before completion, reconcile every physical request and all attempts, metric coverage, scheduler resources, W&B and owned teardown. Mechanical startup tests do not establish model quality; recorded private-process-group cleanup does not establish a fresh cluster-wide orphan scan.

</details>
