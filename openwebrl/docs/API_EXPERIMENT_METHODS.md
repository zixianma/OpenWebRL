<a id="api-browser-evaluation-methods"></a>
# API experiments: guide, methods and code

We compare **one proposed action per decision** with **N proposed actions plus a selector**. Only the selected action executes. This tests whether spending more model work on each decision improves browser-task success, cost, browser activity or latency.

This is the main guide for the published [API experiment code](../api_experiments/). It connects the study protocols, runnable entry points, source archive and result evidence. The main results are in [ARM_FINAL_RESULTS.md](ARM_FINAL_RESULTS.md); study-specific comparisons and uncertainty are in [ARM_INFERENCE_SCALING.md](ARM_INFERENCE_SCALING.md).

| To… | Start with… |
|---|---|
| Understand the comparisons | [Experiments](#experiments), [episode procedure](#episode-procedure) and [metric definitions](#metrics-and-comparability) |
| Run Luna or Claude BrowserCode episodes | [Reproduction quickstart](#reproduction-quickstart); install pinned dependencies, prepare a runtime, then use a read-only dry run |
| Trace or reconstruct the implementation | [Code map](#code-and-frozen-artifacts), [source reconstruction](#restore-an-original-campaign-layout) and [offline validation](#offline-validation) |
| Interpret published outcomes | [Results and evidence](#results-and-analysis-pointers) |
| Inspect historical campaign accounting | [Frozen runtime records](#frozen-runtime-records) and [operational details](#audit-and-operational-details) |

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

The public package includes the frozen scientific configurations, prompts, support code, benchmark-ID rosters and runtime builders. No original campaign directory, Slurm registration or private `.env` is needed. Use a separate working directory for dependencies, generated runtimes and new results.

### Install the pinned dependencies

Use **Linux x86_64, Python 3.12 and Git**. The browser sandbox requires Landlock ABI ≥3; native CUAWright also requires unprivileged user/PID namespaces. Install the host libraries required by Playwright Chromium before the browser checks. All model inference uses hosted APIs; no GPU or model-weight download is needed.

Run from the repository root:

```bash
export API_WORK=/tmp/openwebrl-api-reproduction
python3.12 -B openwebrl/api_experiments/reproduce/install.py \
  --prefix "$API_WORK/deps" --install
export API_PY="$API_WORK/deps/venv/bin/python"
```

[install.py](../api_experiments/reproduce/install.py) installs the [CPU API dependency lock](../api_experiments/reproduce/requirements.txt), fetches pinned CUAWright/BrowserCode/official WebJudge source, verifies the Bun archive and binary hashes, installs the frozen [JavaScript lock](../api_experiments/reproduce/bun.lock), and downloads Playwright 1.58.0 Chromium. It writes `dependencies.json` with local paths. Without `--install`, it prints the installation plan. The optional historical coordinate-browser studies use `--legacy` as described below.

### Prepare the episode runtime

[prepare_browsercode.py](../api_experiments/reproduce/prepare_browsercode.py) reconstructs the source layout and imports public benchmark tasks in the frozen campaign order. It verifies source/browser pins and records every portability adaptation in `portable-build.json`. Use a new output directory:

```bash
"$API_PY" -B openwebrl/api_experiments/reproduce/prepare_browsercode.py \
  --campaign browsercode_luna_effort \
  --dependencies "$API_WORK/deps/dependencies.json" --output "$API_WORK/luna"

"$API_PY" -B openwebrl/api_experiments/reproduce/prepare_browsercode.py \
  --campaign browsercode_claude \
  --dependencies "$API_WORK/deps/dependencies.json" --output "$API_WORK/claude"
```

The published [OM2W](../data/eval/online-mind2web.jsonl), [WebVoyager](../data/eval/webvoyager_fara.jsonl) and [DeepShop](../data/eval/deepshop.jsonl) files supply instruction text, start URLs and grading references. [tasks.py](../api_experiments/reproduce/tasks.py) joins them to the exported task-ID rosters, preserving order and native benchmark identities. No saved private trajectories or teacher labels are transferred into fresh runs.

### Dry-run commands

[eval_luna.py](../api_experiments/reproduce/eval_luna.py), [eval_claude.py](../api_experiments/reproduce/eval_claude.py) and [eval_gpt.py](../api_experiments/reproduce/eval_gpt.py) use the original episode code. They default to a **read-only plan for one task**:

```bash
"$API_PY" -B openwebrl/api_experiments/reproduce/eval_luna.py \
  --runtime "$API_WORK/luna" --arm luna_n5_medium \
  --output "$API_WORK/luna-n5-medium-01" --model-usd 5 --judge-usd 1

"$API_PY" -B openwebrl/api_experiments/reproduce/eval_claude.py \
  --runtime "$API_WORK/claude" --arm sonnet_n5 \
  --output "$API_WORK/claude-sonnet-n5-01" --model-usd 5 --judge-usd 1
```

Add `--execute` to run the selected command. Supply `OPENAI_API_KEY` for GPT/Luna, `ANTHROPIC_API_KEY` for Claude, and `JUDGE_API_KEY` or `OPENAI_API_KEY` for the external judge. BrowserCode wrappers also accept `--env-file /path/to/private.env`. Keep keys outside published source. The example dollar values are stop caps, not cost estimates. `--model-calls`, `--judge-calls` and `--browser-starts` impose additional caps; the default browser allowance includes one mechanical preflight.

### Main BrowserCode experiments

Prepare the named campaign using the same builder, then run its entry point and arm. `--limit` takes the first N tasks after an optional benchmark filter. To collect every method, run each arm into its own new output directory using the same roster and explicitly chosen caps.

| Campaign key | Entry point | Arms | Full task count per arm |
|---|---|---|---:|
| `browsercode_luna_effort` | `eval_luna.py` | `luna_single_xhigh`, `luna_single_medium`, `luna_single_low`, `luna_n5_medium`, `luna_n5_low` | 300 |
| `browsercode_claude` | `eval_claude.py` | `opus_single`, `sonnet_single`, `sonnet_n3`, `sonnet_n5` | 300 |
| `browsercode_gpt_high` | `eval_gpt.py` | `serial_high` (Sol), `parallel3_select_high`, `parallel5_select_high` (Luna) | 300 |
| `browsercode_luna_single_high` | `eval_gpt.py` | `luna_single_high` | 300 |
| `browsercode_gpt_high_wv_ds` | `eval_gpt.py` | `serial_high`, `parallel3_select_high`, `parallel5_select_high`, `luna_single_high` | 745 |

For the last campaign, pass `--benchmark webvoyager --limit 595` or `--benchmark deepshop --limit 150`; the filter precedes `--limit`. Luna N5 medium/low uses the named effort for both candidates and selector. Claude uses high effort. GPT-high uses high effort and the benchmark-specific judge. Effective settings come from the exported configuration, rather than a shared universal protocol.

### Native CUAWright experiments

[native.py](../api_experiments/reproduce/native.py) builds its own runtime from the same dependency manifest. It supports `luna_high_single`, `luna_high_n3_select_high`, `luna_high_n5_select_high`, `sol_high_single` and `sol_high_n3_select_high`:

```bash
"$API_PY" -B openwebrl/api_experiments/reproduce/native.py \
  --dependencies "$API_WORK/deps/dependencies.json" \
  --arm luna_high_n5_select_high --limit 1 --output "$API_WORK/native-luna-n5-01" \
  --model-usd 5 --judge-usd 1 --model-calls 250 --judge-calls 100 \
  --browser-starts 2 --timeout-seconds 1800
```

This is a read-only plan. Add `--execute` with `OPENAI_API_KEY` set to run it; all six USD/call/browser/time caps are required for execution. The default task file is public OM2W; `--limit 300` collects the full roster, while `--task-id` filters named tasks before `--limit` (raise the limit when selecting several IDs). Its browser cap includes one startup probe, so the example permits one probe plus one task browser. Native runs retain high effort, 20,000 output tokens, 100 decisions, summaries every 20 decisions, selected-only execution and official GPT-4o WebJudge. Original native format-repair defaults remain intact; the metered gateway never automatically repeats a physical provider request after an unknown-usage failure.

<details>
<summary>Earlier Sol 6.1 and Luna studies</summary>

The same BrowserCode builder also accepts:

| Campaign key | Policies / executor |
|---|---|
| `browsercode_sol61_pair20`, `browsercode_sol61_pair80` | Sol 6.1 `serial_high` versus `parallel3_select_low`; use `--harness browsercode` or `--harness cuawright` |
| `browsercode_sol61_onecall20`, `browsercode_sol61_onecall80` | Sol 6.1 `onecall3_low`; choose either paired executor |
| `browsercode_sol61_low_medium30` | Sol 6.1 `single_low`, `single_medium`; choose either paired executor |
| `browsercode_luna_high100` | Luna `parallel3_select_high`, `parallel5_select_high`; later 12,288-token phase, BrowserCode |
| `browsercode_luna_high100_initial`, `browsercode_luna_high100_recovery4096` | Original and repaired 4,096-token Luna N3/N5 phases; BrowserCode |

Use `eval_gpt.py` with the prepared runtime; it validates that the arm/model/output limits match that campaign. The paired 20-task and 80-task extension cohorts have separate keys. Earlier cap-limited and later 12,288-token phases remain separate scientific records in the source inventory.

The original coordinate-browser studies use [legacy.py](../api_experiments/reproduce/legacy.py). Install their optional CPU-only dependencies and pinned Qwen tokenizer/vision metadata with `install.py --prefix "$API_WORK/deps" --install --legacy`; this downloads no model weights. Then:

```bash
"$API_PY" -B openwebrl/api_experiments/reproduce/legacy.py prepare \
  --campaign sol61_om2w100_medium --dependencies "$API_WORK/deps/dependencies.json" \
  --output "$API_WORK/coordinate-medium"
"$API_PY" -B openwebrl/api_experiments/reproduce/legacy.py run \
  --runtime "$API_WORK/coordinate-medium" --output "$API_WORK/coordinate-medium-01" \
  --limit 1 --model-usd 5 --judge-usd 1
```

Other keys are `sol61_om2w100_low`, `sol61_om2w200_medium`, `sol61_webvoyager100_low` and `sol61_deepshop100_low`. The default runs both original policies in their fixed within-task order; `--arm` selects one. Add `--execute` to run. These use their original coordinate observation/context and grading paths, distinct from BrowserCode and native CUAWright.

</details>

### Saved-state calibration and selector probes

[snapshots.py](../api_experiments/reproduce/snapshots.py) runs the original Sol 6.1 calibration, medium-actor follow-up and low-versus-medium selector probe code. Prepare inputs from your own fresh coordinate-browser records:

```bash
"$API_PY" -B openwebrl/api_experiments/reproduce/snapshots.py extract-calibration \
  --episodes "$API_WORK/coordinate-medium-01" --output "$API_WORK/calibration" --limit 30
"$API_PY" -B openwebrl/api_experiments/reproduce/snapshots.py run-calibration \
  --runtime "$API_WORK/calibration" --model-usd 5 --model-calls 150
"$API_PY" -B openwebrl/api_experiments/reproduce/snapshots.py prepare-selector \
  --episodes "$API_WORK/coordinate-medium-01" --output "$API_WORK/selector-probe" --limit 100
"$API_PY" -B openwebrl/api_experiments/reproduce/snapshots.py run-selector \
  --runtime "$API_WORK/selector-probe" --model-usd 5 --model-calls 200
```

Runs default to a read-only plan; add `--execute` to dispatch within the supplied caps. `prepare-medium` and `run-medium` provide the follow-up from calibration receipts. The [input schema](../api_experiments/portable_assets/snapshots/calibration/input-schema.json) also accepts externally supplied states with hash-bound screenshots. Fresh state sampling is deterministic and outcome-independent; it is a new cohort, distinct from the historical private balanced C2/Piotr states. The package preserves the historical proposal/selector protocols without publishing those private states or transferring their labels.

### Compare fresh runs

[report.py](../api_experiments/reproduce/report.py) reads the saved task dispositions, verifies their run identity and produces aggregate and paired comparisons:

```bash
"$API_PY" -B openwebrl/api_experiments/reproduce/report.py \
  --run single="$API_WORK/luna-single-01" --run n5="$API_WORK/luna-n5-medium-01" \
  --output "$API_WORK/luna-comparison.json"
```

It reports task counts separately from success rates, retains invalid/missing outcomes, distinguishes all-attempt model/judge charges from valid-task efficiency, and labels common-valid paired comparisons. Paired uncertainty uses saved task identities; differing harnesses, benchmarks and judge protocols are not silently treated as matched controls. Historical plotting scripts and their public aggregate inputs remain linked under [results](#results-and-analysis-pointers).

### What execution saves

Each invocation requires a new output directory. It owns separate ledgers, browser counters, request evidence, tracking identity and per-task outcomes. Existing valid or invalid results are never silently overwritten or replaced. Unknown usage retains its reservation; invalid attempts remain in all-attempt accounting.

BrowserCode `summary.json` includes requested/recorded/valid/invalid counts, recorded and valid-only success rates, valid-task latency/browser-batch means with observation counts, separate model/judge fees or reservations and cleanup status. Native writes task `disposition.json` files and a summary with usage. Historical coordinate runs preserve their original per-task records. W&B defaults to offline in `openwebrl-evals`; `--wandb-mode disabled` omits it in BrowserCode/coordinate runs. Raw run outputs contain task content and should stay local.

### Scope of reproduction

These commands perform **fresh collection** with the frozen episode protocols and public task cohorts. They preserve candidate/selector settings, selected-only history, action parsing, benchmark judges and explicit accounting, while adapting local paths, dependency assembly and process ownership. Generated manifests record those adaptations and source hashes. Archived source files remain unchanged.

The original campaigns used concurrent scheduler shards and sometimes reused earlier recorded cohorts. Fresh reproduction collects those cells anew; it does not recreate historical site state, provider alias snapshots, scheduler contention, reused outcomes or original latency conditions. Use identical fresh task panels across arms and keep harness/judge cohorts separate. A new live-site run can legitimately produce different outcomes. No command submits a Slurm job.

## Code and frozen artifacts

The files under `api_experiments/` are **frozen source archives** grouped by harness and function. Original sibling imports and host-specific paths remain intact in the archive. The public [runtime builders](#reproduction-quickstart) assemble runnable copies with explicit portability overlays. Identical files are shared once. [source-provenance.json](../api_experiments/source-provenance.json) maps each campaign's original filename to its published path, SHA256 and byte count. Archived source bytes remain unchanged.

For BrowserCode, read the worker first, then its model, budget, browser environment and judge. The reproduction wrappers import `worker` from the generated runtime. [portable_assets/](../api_experiments/portable_assets/) supplies exact frozen support sources, effective scientific configurations, browser bridge and task-ID rosters; [variants/](../api_experiments/browsercode/variants/) preserves the additional executed campaign source snapshots.

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

All campaign keys and file hashes are listed in [source-provenance.json](../api_experiments/source-provenance.json). The output directory must be new. All source hashes are checked before any output files are written. This copies code only; it starts no workers or API requests. For a runnable environment, use the installer and runtime builders above.

Credentials, `.env`, task data, screenshots, trajectories, request logs, ledgers and scheduler records are excluded. The private Claude terminal-interruption qualifiers are also excluded because they embed task identities; the original worker still references them for historical recovery. The public builders provide the scientific configuration, support code and gateway assembly; the installer supplies pinned external code and browser dependencies. Original private campaign approvals and scheduler records are neither required nor copied into collaborator runs.

### Offline validation

Run archive, scientific-input and accounting tests without API calls:

```bash
python3 -B -m unittest discover -s openwebrl/api_experiments/tests -v
```

[tests/](../api_experiments/tests/) checks source/layout hashes, scientific rosters, dry runs, caps, output isolation, cleanup and accounting. Set `OPENWEBRL_API_DEPENDENCIES` to the installer manifest to enable full real-source runtime construction and import checks. [validation/](../api_experiments/validation/) contains archived campaign tests that may require the original runtime; it is not the self-contained public test suite. Keep source snapshots byte-for-byte intact and update the provenance index when publishing an additional snapshot.

To verify the installed real sources and adapters, enable the integration checks:

```bash
OPENWEBRL_API_DEPENDENCIES="$API_WORK/deps/dependencies.json" \
  "$API_PY" -B -m unittest discover -s openwebrl/api_experiments/tests -v
"$API_PY" -B openwebrl/api_experiments/reproduce/verify_runtime.py \
  --runtime "$API_WORK/luna" --browser-preflight
"$API_PY" -B openwebrl/api_experiments/reproduce/native_check.py \
  --dependencies "$API_WORK/deps/dependencies.json" \
  --output "$API_WORK/native-install-check" --browser-preflight
```

The preflights start owned local browsers against diagnostic pages, exercise tools/screenshots and verify cleanup with zero provider calls. Coordinate runtimes also support `legacy.py verify --runtime PATH --browser-preflight`. [verify_legacy.py](../api_experiments/reproduce/verify_legacy.py) additionally runs the original episode loop on diagnostic tasks through a fake SDK boundary, with real Chromium and tokenization, without provider calls. These checks validate installation and protocol boundaries; fresh model/browser outcomes require the explicit episode commands above.

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

The earlier `sol_serial_parallel_policy.py`, `sol_serial_parallel_transport.py` and `sol_serial_parallel_eval.py` implementations are preserved in the exported [frozen support](../api_experiments/portable_assets/support/) and [historical source trees](../api_experiments/portable_assets/legacy/). The portable builders select the matching version; use the published reproduction entry points rather than the original Slurm launcher.

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

<a id="audit-and-operational-details"></a>
## Historical campaign accounting

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
