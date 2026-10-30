<a id="api-browser-evaluation-methods"></a>
# API experiments: guide, methods and code

We compare **one proposed action per decision** with **N proposed actions plus a selector**. Only the selected action executes. This tests whether spending more model work on each decision improves browser-task success, cost, browser activity or latency.

**External reproduction is incomplete.** The public archive preserves selected source snapshots, but a clean clone cannot currently run all reported API experiments. See the [reproducibility audit](#external-reproducibility-status) before using the quickstart.

This is the main guide for the published [API experiment code](../api_experiments/). It connects the study protocols, runnable entry points, source archive and result evidence. The main results are in [ARM_FINAL_RESULTS.md](ARM_FINAL_RESULTS.md); study-specific comparisons and uncertainty are in [ARM_INFERENCE_SCALING.md](ARM_INFERENCE_SCALING.md).

| To… | Start with… |
|---|---|
| Check what an external collaborator can reproduce | [Coverage, missing dependencies and validation limits](#external-reproducibility-status) |
| Understand the comparisons | [Experiments](#experiments), [episode procedure](#episode-procedure) and [metric definitions](#metrics-and-comparability) |
| Run Luna or Claude BrowserCode episodes | [Reproduction quickstart](#reproduction-quickstart); requires an existing configured runtime and defaults to a read-only dry run |
| Trace or reconstruct the implementation | [Code map](#code-and-frozen-artifacts), [source reconstruction](#restore-an-original-campaign-layout) and [offline validation](#offline-validation) |
| Interpret published outcomes | [Results and evidence](#results-and-analysis-pointers) |
| Audit an original campaign | [Frozen runtime records](#frozen-runtime-records) and [operational details](#audit-and-operational-details) |

## External reproducibility status

**Audit: October 29, 2026. No campaign family has verified end-to-end reproduction from the public clone.** The published wrappers call an already configured runtime; the materializer reconstructs source filenames. Neither builds the missing runtime. Source integrity, code coverage and executable reproduction are separate checks.

| Reported experiments | Published source coverage | Blocking gaps for a clean clone |
|---|---|---|
| Luna effort and Claude BrowserCode, OM2W | Actor/selector adapters, episode workers, budget/judge logic and existing-runtime wrappers | Frozen plans/configuration/prompts, manifests, exact support sources, browser bridge/build inputs and portable installation |
| GPT-high BrowserCode, OM2W/WebVoyager/DeepShop | One earlier Sol single / Luna N3/N5 OM2W source snapshot | Later Luna single-high and WebVoyager/DeepShop campaign snapshots/configuration; missing shared support and preparation code; no reproduction wrapper |
| Native CUAWright, OM2W | Proposal/transport, worker, gateway, broker, capture and judge adaptation source | Upstream/runtime assembly, override configuration, child bootstrap, grading checkout, task-phase/preparation code and campaign launch entry point |
| Other historical API studies linked from the scaling document | Four archive keys do not enumerate every historical study | Per-study executed-source inventory and a runnable configuration are still needed; no complete coverage claim |

Public benchmark files already exist for [OM2W](../data/eval/online-mind2web.jsonl), [WebVoyager](../data/eval/webvoyager_fara.jsonl) and [DeepShop](../data/eval/deepshop.jsonl). What is missing is the reproducible conversion into each frozen campaign's ordered `tasks.json`, arm/shard assignments and reused/fresh cohorts. Existing aggregate tables cannot reconstruct those identities. Private task-level results, trajectories and credentials remain outside this publication.

<details>
<summary>Concrete missing execution dependencies and validation evidence</summary>

- [Reproduction preparation](../api_experiments/reproduce/common.py#L71) reads `campaign-plan.json`, `runtime-config.json`, `tasks.json` and `source-manifest.json`. None is produced by materialization. It then requires configured CUAWright/support trees, browser/Python executables and `browser_bridge.ts`, `browsercode/`, `node_modules/` and `javascript-runtime/bun`.
- [BrowserCode bootstrap](../api_experiments/browsercode/runtime/bootstrap.py) imports the configured external source trees. Actor system templates come from the unpublished configuration; [Terra grading](../api_experiments/browsercode/judges/terra.py#L24) loads prompts from frozen support source. The earlier [GPT-high worker](../api_experiments/browsercode/workers/gpt_high_worker.py#L5) imports unpublished `preparation_gate`; the [shared controller](../api_experiments/browsercode/controllers/shared_controller.py#L27) imports unpublished `orchestration.launch_campaign`. These are code dependencies, beyond credentials and task data.
- The [GPT-high adapter's policy table](../api_experiments/browsercode/models/paired_model.py#L11) supports Sol single and Luna N3/N5, without the later Luna single-high arm. Its [historical judge](../api_experiments/browsercode/judges/historical_gpt.py) is the OM2W o4-mini path; it does not supply the WebVoyager/DeepShop campaign grading integration. General benchmark reward implementations elsewhere in the repository do not establish the exact executed campaign wiring.
- [Native execution](../api_experiments/cuawright/runtime/native_episode.py#L92) requires `upstream/src`, `local_override.yaml` and `confinement_support/`. The [native worker](../api_experiments/cuawright/workers/campaign_worker.py) requires the absent child `sitecustomize.py`, task-phase orchestration and release records; it defines `run_shard()` without a campaign CLI. [External grading](../api_experiments/cuawright/judges/external_judge.py#L21) additionally points to a private prior runtime containing official WebJudge source/dependencies and its plan.
- The root [requirements.txt](../../requirements.txt) is a training environment specification, with no complete API runtime lock/install recipe. The archive additionally assumes host-specific executable paths and Linux sandbox features. Some upstream revisions are [documented historically](ARM_INFERENCE_SCALING.md#sol61-browsercode-cuawright-paired20-20261016), but the guide does not yet supply a verified runtime builder using all required pins and local adaptations.

The audit hash-checked and materialized all four source layouts, and parsed every restored Python file successfully. Both Luna and Claude public-source-only dry runs exited with code 2 at the missing `campaign-plan.json`, before creating outputs or dispatching browser/API work. The twelve public offline tests pass, but their reproduction fixtures use synthetic tasks, placeholder dependency files and mocked SDK/worker execution. They establish source integrity and wrapper boundaries, not an installed browser runtime, live API compatibility or complete campaign reproduction.

</details>

To establish external reproduction, publish or reproducibly acquire the missing non-secret code and effective scientific configuration; provide pinned API/browser dependencies and a portable runtime builder; specify benchmark loading and exact cohort construction; and cover every reported campaign variant with a documented entry point. Validate that assembly in a clean environment, then perform a separately budgeted live protocol check. Reproducing fresh live-site evaluations cannot guarantee identical historical outcomes or latency.

## Experiments

| Study | Methods | Tasks per method | Output cap / decision limit | Terminal judge |
|---|---|---|---|---|
| GPT high, BrowserCode | Sol single; Luna single, N3, N5; all high | OM2W 300; WebVoyager 595; DeepShop 150 | 12,288 / 30 | OM2W: historical o4-mini; WebVoyager/DeepShop: benchmark-native GPT-4o |
| GPT high, native CUAWright | Sol single; Luna single, N3, N5; all high | OM2W 300 | 20,000 / 100; summary setting 20 | Official WebJudge, GPT-4o; separate native protocol |
| Claude, BrowserCode | Opus 5.5 single; Sonnet 5.5 single, N3, N5; all high | OM2W 300; completed | 12,288 / 30 | GPT-5.6 Terra |
| GPT Luna effort, BrowserCode | Single xhigh/medium/low; N5 medium+medium and low+low | OM2W 300 targeted; no qualified full300 result linked here | 12,288 / 30 | GPT-5.6 Terra |

The limits and judges are campaign-specific. BrowserCode executes selected JavaScript through `browser_execute`; native CUAWright uses its native shell/Playwright tool loop. Their command counts and latency measurements are not interchangeable. The output cap applies to each candidate and selector request. In N5 medium+medium, both candidates and selector use medium effort. Effort and token caps do not enforce equal realized compute; we measure actual usage.

Each campaign fixes its task roster and shard assignments. Arms share task IDs but run independent browser trajectories. Pilot tasks count toward the final roster. The native extension reuses the earlier 100 outcomes, including invalids, for Sol single and Luna single/N3, adds 200 fresh tasks to each, and collects all 300 N5 tasks fresh. Its fresh-200 comparison is reported separately.

## Episode procedure

The following loop describes recent Luna effort and Claude BrowserCode runs. Earlier benchmark campaigns and native CUAWright use their own evidence and grading paths; see the [campaign code map](#code-and-frozen-artifacts) and [result records](#results-and-analysis-pointers).

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

The GPT-high BrowserCode headline cohort retains historical o4-mini grading for OM2W and benchmark-native GPT-4o protocols for WebVoyager and DeepShop. The separate [official-WebJudge regrading study](ARM_INFERENCE_SCALING.md#om2w-grading-browser-audit-20261020) does not replace those headline outcomes. The archived native judge uses official WebJudge with `gpt-4o-2024-08-06`; follow its [methods record](ARM_INFERENCE_SCALING.md#cuawright-native-full300-four-methods-20261025) for evidence and qualification. Recent Claude and Luna effort runs use `gpt56-terra-terminal-success-v1`: `gpt-5.6-terra`, medium effort, 4,096 completion tokens and seed 42. The request contains the task, selected action history and final screenshot. `common_judge.py` loads the frozen prompts and validates the completed, non-refusal verdict. Earlier o4-mini and later Terra results remain separate judge cohorts. Historical Claude grading/recovery attempts and fees remain in the lineage.

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

## Reproduction quickstart

Use [eval_luna.py](../api_experiments/reproduce/eval_luna.py) for GPT-6 Luna and [eval_claude.py](../api_experiments/reproduce/eval_claude.py) for Claude. Both call the original frozen `worker.episode`: same prompts, native tools, candidate/selector policy, 12,288 output tokens, 30 decisions, screenshots and GPT-5.6 Terra judge. The default is a **read-only dry run of one task**. Add `--execute` to spend API budget and start browsers.

These entry points require an **existing configured campaign runtime** with frozen plan/configuration/task manifests, pinned support sources, provider SDKs and browser dependencies. The public archive preserves source code; it does not supply frozen campaign task manifests or a standalone runtime installation. Use the [runtime directory records](#frozen-runtime-records) to locate the original inputs. There are no equivalent reproduction wrappers for the older GPT-high or native CUAWright campaigns.

### Dry-run commands

Run from the repository root. Set `LUNA_RUNTIME` or `CLAUDE_RUNTIME` to the corresponding frozen campaign directory, and use its configured Python environment. Set `REPRO_OUTPUT_ROOT` to a writable directory accepted by that runtime's browser bridge; on the original host this is below `/gpfs/scrubbed/zixianma/openwebrl-runtime/`.

```bash
python3 -B openwebrl/api_experiments/reproduce/eval_luna.py \
  --runtime "$LUNA_RUNTIME" --arm luna_n5_medium \
  --output "$REPRO_OUTPUT_ROOT/luna-n5-medium-01" \
  --model-usd 5 --judge-usd 1

python3 -B openwebrl/api_experiments/reproduce/eval_claude.py \
  --runtime "$CLAUDE_RUNTIME" --arm sonnet_n5 \
  --output "$REPRO_OUTPUT_ROOT/claude-sonnet-n5-01" \
  --model-usd 5 --judge-usd 1
```

Review the printed plan, then repeat the chosen command with `--execute`. The example dollar amounts are **explicit stop caps**, not estimates or permission to exceed an existing allocation. Execution requires actor credentials (`OPENAI_API_KEY` for Luna, `ANTHROPIC_API_KEY` for Claude) plus `JUDGE_API_KEY` or `OPENAI_API_KEY` for Terra. Supply them in the environment or with `--env-file /path/to/private.env`; never commit the key file.

| Script | `--arm` choices |
|---|---|
| Luna | `luna_single_xhigh`, `luna_single_medium`, `luna_single_low`, `luna_n5_medium`, `luna_n5_low` |
| Claude | `opus_single`, `sonnet_single`, `sonnet_n3`, `sonnet_n5` |

Luna N5 medium/low uses the named effort for both candidates and selector. Claude arms use high effort. To evaluate the full frozen roster, add `--limit 300` and set explicitly approved run caps. `--limit` takes the first N tasks in frozen order. There is no sampling or replacement of invalid tasks.

### What execution saves

Each invocation requires a **new output directory** outside the source runtime. It owns separate model/judge ledgers, browser counter, HTTP gate, SDK caches, W&B identity, episode evidence and `summary.json`. It cannot resume or overwrite an earlier invocation. Provider retries are disabled; a shared halt stops the run for diagnosis. Unknown usage retains its full reservation, and invalid episodes remain in cost accounting.

`summary.json` reports recorded/valid/invalid counts, valid-only success, valid-task mean latency and selected browser batches with observation counts, and separate all-attempt model/judge charges or reservations. Judge time is excluded from actor task latency. A browser batch is one selected executable snippet; it can contain several browser operations. Missing measurements are not zero. Raw outputs include private task content and should stay local.

W&B defaults to offline under the output directory in `openwebrl-evals`. Use `--wandb-mode online` to sync or `--wandb-mode disabled` to omit tracking. `--model-calls`, `--judge-calls` and `--browser-starts` provide additional stop caps; browser starts default to N tasks plus one mechanical preflight. Model calls include candidates, selectors and token counters. Model/judge pools never transfer funds.

### Scope of reproduction

The scripts preserve the **episode protocol**. They run one episode at a time, retaining parallel candidates within each decision, and apply one pair of run-level USD/call caps. The original studies used separately capped shards and concurrent scheduler jobs, so these entry points do not recreate the original scheduling, contention, shard admission history or measured latency environment. They submit no Slurm jobs and reserve no compute. Full campaign replication requires the original [campaign controllers and frozen approvals](#frozen-runtime-records), including canary/pilot audits and all-attempt resource accounting.

Startup checks verify the frozen plan, source hashes, task manifest and local browser dependencies before execution. They do not prove live provider access or site availability. Use `--help` on either entry point to inspect options without a runtime, browser or API call.

## Code and frozen artifacts

The files under `api_experiments/` are **frozen source archives** grouped by harness and function. Original sibling imports and host-specific paths remain intact; they are not an installable Python package. Identical files are shared once. [source-provenance.json](../api_experiments/source-provenance.json) maps each campaign's original filename to its published path, SHA256 and byte count. Archived source bytes remain unchanged.

For BrowserCode, read the worker first, then its model, budget, browser environment and judge. The reproduction wrappers import `worker` from the configured frozen runtime, rather than importing the grouped archive directly.

| Campaign key | Episode worker | Actor and selector | Budget / admission | Browser and metrics | Judge |
|---|---|---|---|---|---|
| `browsercode_luna_effort` | [luna_effort_worker.py](../api_experiments/browsercode/workers/luna_effort_worker.py) | [luna_model.py](../api_experiments/browsercode/models/luna_model.py) | [request_budget.py](../api_experiments/browsercode/budgets/request_budget.py), [claude_budget.py](../api_experiments/browsercode/budgets/claude_budget.py) | [environments.py](../api_experiments/browsercode/runtime/environments.py), [metrics](../api_experiments/browsercode/runtime/luna_effort_episode_metrics.py) | [Terra](../api_experiments/browsercode/judges/terra.py) |
| `browsercode_claude` | [claude_worker.py](../api_experiments/browsercode/workers/claude_worker.py) | [claude_model.py](../api_experiments/browsercode/models/claude_model.py) | [claude_budget.py](../api_experiments/browsercode/budgets/claude_budget.py) | [environments.py](../api_experiments/browsercode/runtime/environments.py), [metrics](../api_experiments/browsercode/runtime/claude_episode_metrics.py) | [Terra](../api_experiments/browsercode/judges/terra.py) |
| `browsercode_gpt_high` | [gpt_high_worker.py](../api_experiments/browsercode/workers/gpt_high_worker.py) | [paired_model.py](../api_experiments/browsercode/models/paired_model.py) | [`ReservationBudget`](../api_experiments/browsercode/models/paired_model.py) in the model; browser `Budget` from frozen support | [environment](../api_experiments/browsercode/runtime/gpt_high_environments.py), [metrics](../api_experiments/browsercode/runtime/gpt_high_episode_metrics.py) | [Historical GPT](../api_experiments/browsercode/judges/historical_gpt.py) |
| `cuawright_native` | [campaign_worker.py](../api_experiments/cuawright/workers/campaign_worker.py), [native_episode.py](../api_experiments/cuawright/runtime/native_episode.py) | [proposal adapter](../api_experiments/cuawright/models/native_proposal_adapter.py), [transport](../api_experiments/cuawright/models/native_group_transport.py), [relay](../api_experiments/cuawright/models/native_group_relay.py) | [budget_gateway.py](../api_experiments/cuawright/budgets/budget_gateway.py) | [browser broker](../api_experiments/cuawright/runtime/browser_broker.py), [capture](../api_experiments/cuawright/runtime/native_capture.py) | [Official WebJudge wrapper](../api_experiments/cuawright/judges/external_judge.py) |

[BrowserCode controllers](../api_experiments/browsercode/controllers/) own batch workers; [protocols](../api_experiments/browsercode/protocols/) define financial enrollment and canary scope; [recovery](../api_experiments/browsercode/recovery/) preserves saved-judge recovery logic. Native CUAWright has its own broker, confinement and timeout helpers under [runtime/](../api_experiments/cuawright/runtime/). The table maps published snapshots, not every historical variant in the result cohorts; use the provenance index and each cohort's frozen manifest to resolve exact executed sources.

### Restore an original campaign layout

Use [materialize.py](../api_experiments/materialize.py) to reconstruct a campaign's original sibling filenames. Run from the repository root:

```bash
python3 -B openwebrl/api_experiments/materialize.py browsercode_luna_effort \
  --output /tmp/luna-effort-source
```

The other keys are `browsercode_claude`, `browsercode_gpt_high` and `cuawright_native`. The output directory must be new. All source hashes are checked before any output files are written. This copies code only; it starts no workers or API requests. It does not create the configured runtime required by the reproduction wrappers.

Credentials, `.env`, task data, screenshots, trajectories, request logs, ledgers and scheduler records are excluded. The private Claude terminal-interruption qualifiers are also excluded because they embed task identities; the original worker still references them for historical recovery. Execution additionally requires frozen configuration, effective approvals, pinned CUAWright/OpenWebRL support sources, SDKs and browser/bridge dependencies; native execution also needs its gateway configuration.

### Offline validation

Run the public materializer and reproduction boundary tests without browsers or API calls:

```bash
python3 -B -m unittest discover -s openwebrl/api_experiments/tests -v
```

[tests/](../api_experiments/tests/) checks source hashes/layout reconstruction and reproduction boundaries, including dry runs, caps, selected cohorts, output isolation and mocked execution. [validation/](../api_experiments/validation/) contains archived campaign tests that may require the original runtime; it is not the self-contained public test suite. Keep source snapshots byte-for-byte intact and update the provenance index when publishing an additional snapshot.

### Frozen runtime records

<details>
<summary>Original runtime directories, configuration and lineage</summary>

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

</details>

Earlier local development used `openwebrl/sol_serial_parallel_policy.py` (selection), `sol_serial_parallel_transport.py` (native history), `sol_serial_parallel_eval.py` (evaluation), and `scripts/run_sol_serial_parallel_om2w.py` (entry point). These files are absent from the public checkout: they are historical names, not available reproduction entry points. The archived GPT-high adapter still imports the first helper, which is an additional unresolved source dependency.

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
| GPT effort qualification/status, local | `current.json`, `orchestration/registry.json`, `analysis/pilot-cost-projection-predeclaration-v1.json`, independent pilot audits and repair receipts inside its campaign; no qualified full300 result linked here |

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
