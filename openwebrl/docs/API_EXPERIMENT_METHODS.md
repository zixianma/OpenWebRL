<a id="api-browser-evaluation-methods"></a>
# BrowserCode API experiments

We compare **one proposed action per decision** with **N proposed actions plus a selector**. Only the selected action executes. BrowserCode runs that action as JavaScript through `browser_execute`. The studies measure task success, model cost, browser activity and latency.

The [Online-Mind2Web GPT figure](ARM_FINAL_RESULTS.md#luna-browsercode-effort-om2w300-20261030) now uses **GPT-5.6 Terra for every point**, including 1,892 regraded historical endpoints and the medium/low effort arms after their prescribed invalid-task retries. Historical endpoint-evidence exclusions and metric coverage remain explicit; the original effort campaign remains provisional. The completed Claude study has 300 dispositions per arm; on its 246 common-valid tasks, Sonnet single and N5 each solved 147. See its [qualified results](ARM_FINAL_RESULTS.md#claude-browsercode-om2w300) and the historical GPT [three-benchmark tables](ARM_FINAL_RESULTS.md#headline-performance-cost-and-efficiency).

Use **[browsercode.py](../api_experiments/reproduce/browsercode.py)** for setup, evaluation, checks and reporting. This guide is the main collaborator reference; the [source package](../api_experiments/README.md) preserves the implementations and frozen inputs.

## Reproduction quickstart

### Install the pinned dependencies

Use **Linux x86_64, Python 3.12 and Git**, with Landlock ABI ≥3 and the host libraries required by Playwright Chromium. No GPU or model weights are needed. From the repository root, prepare one experiment:

```bash
export API_WORK=/tmp/openwebrl-browsercode-luna
python3.12 -B openwebrl/api_experiments/reproduce/browsercode.py setup \
  --workspace "$API_WORK" --campaign luna
export API_PY="$API_WORK/dependencies/venv/bin/python"
```

This installs the [CPU dependency lock](../api_experiments/reproduce/requirements.txt), pinned BrowserCode and shared agent sources, verified Bun/JavaScript dependencies and Playwright 1.58.0 Chromium. It prepares the frozen campaign runtime at `$API_WORK/runtime`; it makes no model calls. The default installation omits the archived harness and judge extras. Setup requires a new workspace. To reuse installed dependencies without downloads, pass `--dependencies /path/to/dependencies.json` and use the Python interpreter recorded in that manifest.

### Dry-run commands

Print a **read-only plan for one task**:

```bash
"$API_PY" -B openwebrl/api_experiments/reproduce/browsercode.py run \
  --workspace "$API_WORK" --arm luna_n5_medium \
  --output "$API_WORK/n5-medium-01" --model-usd 5 --judge-usd 1
```

Add `--execute` to this command to collect an episode. Supply `OPENAI_API_KEY` for GPT/Luna, `ANTHROPIC_API_KEY` for Claude, and `JUDGE_API_KEY` or `OPENAI_API_KEY` for judging; alternatively pass `--env-file /path/to/private.env`. Keep keys outside published source. The example dollar amounts are stop caps, not cost estimates. `--model-calls`, `--judge-calls` and `--browser-starts` add limits; the default browser allowance includes one mechanical preflight. Every execution requires a new output directory.

Run a local browser check before spending on inference:

```bash
"$API_PY" -B openwebrl/api_experiments/reproduce/browsercode.py check \
  --workspace "$API_WORK" --browser-preflight
```

This exercises the real imports, tools, screenshots and cleanup with zero provider calls. It does not measure model quality.

### Main BrowserCode experiments

Choose `--campaign` during setup. The entry point selects the matching model adapter automatically. Each campaign needs a separate workspace; installed dependencies can be shared. Run every arm into its own output directory with the same task roster and explicitly chosen caps.

| Campaign | Arms | Full task count per arm |
|---|---|---:|
| `luna` | `luna_single_xhigh`, `luna_single_medium`, `luna_single_low`, `luna_n5_medium`, `luna_n5_low` | 300 |
| `claude` | `opus_single`, `sonnet_single`, `sonnet_n3`, `sonnet_n5` | 300 |
| `gpt` | `serial_high` (Sol), `parallel3_select_high`, `parallel5_select_high` (Luna) | 300 |
| `gpt-luna-single` | `luna_single_high` | 300 |
| `gpt-crossbench` | `serial_high`, `parallel3_select_high`, `parallel5_select_high`, `luna_single_high` | 745 |

`--limit` defaults to one task. For the full OM2W panel use `--limit 300`. For `gpt-crossbench`, use `--benchmark webvoyager --limit 595` or `--benchmark deepshop --limit 150`; filtering precedes the limit. GPT OM2W's four methods span `gpt` and `gpt-luna-single`. Luna N5 medium/low applies that effort to both candidates and selector. The full historical `browsercode_*` campaign keys remain accepted.

### Compare fresh runs

Run the desired single-action arm into a separate output directory, then compare saved outputs:

```bash
"$API_PY" -B openwebrl/api_experiments/reproduce/browsercode.py report \
  --run single="$API_WORK/single-01" --run n5="$API_WORK/n5-medium-01" \
  --output "$API_WORK/comparison.json"
```

The report retains invalid/missing outcomes, separates all-attempt model/judge charges from valid-task efficiency, and labels common-valid paired comparisons and uncertainty. Different benchmarks or judge protocols are not silently treated as matched controls.

### Rejudge saved endpoints

Use [`rejudge.py`](../api_experiments/reproduce/rejudge.py), also exposed as `browsercode.py rejudge`, to grade your own archived terminal evidence with **GPT-5.6-Terra, medium effort, 4,096 completion tokens and seed 42**. It reuses the bundled frozen terminal prompt, verdict parser and conservative request accounting. It runs no actors or browsers and requires no scheduler or original campaign directory. Use the installed Python above; no browser setup is needed for judging alone.

```bash
"$API_PY" -B openwebrl/api_experiments/reproduce/browsercode.py rejudge \
  --manifest /path/to/local/endpoints.json --output /tmp/terra-regrade-01 \
  --judge-usd 5 --judge-calls 20 --wall-seconds 600 --concurrency 2
```

This validates inputs and prints a read-only plan. The four caps are explicit limits, not cost or runtime estimates. Add `--execute` to create a new output cohort and spend within them; use `JUDGE_API_KEY` or `OPENAI_API_KEY`, or `--env-file /path/to/private.env`. The parent terminates and awaits its sole judge process at the wall deadline. Every admitted physical request retains its known charge or full unknown/in-flight reservation. Failed judgments remain invalid, and prior claims/results are never overwritten or automatically retried.

<details>
<summary>Local endpoint manifest and saved outputs</summary>

Create a private JSON manifest with this schema. The strings in angle brackets are placeholders for your exact archived evidence, not released task content:

```json
{
  "schema": "terra-endpoint-input-v1",
  "episodes": [{
    "id": "local-episode-1",
    "original_valid": true,
    "eligible": true,
    "cohort": {
      "benchmark": "online-mind2web",
      "harness": "browsercode",
      "arm": "<archived arm>",
      "actor_model": "<archived actor model>",
      "actor_protocol_sha256": "<protocol SHA256>"
    },
    "task": "<exact archived task>",
    "actions": ["<exact selected-action string>"],
    "screenshot": "relative/path/final.png",
    "screenshot_sha256": "<PNG SHA256>",
    "endpoint_scope": "dedicated_terminal_capture"
  }]
}
```

Paths resolve relative to the manifest; absolute local paths also work. Preserve the original PNG bytes and geometry: the supported archived sizes are 1,280 × 1,800 and 1,280 × 1,657. A `last_browser_action_capture` also requires `last_action_capture_verified: true`: independently bind the PNG to the final browser action, with no later browser action. An older intermediate capture does not qualify. Optional `evidence_bindings: [{"path": "...", "sha256": "..."}]` freezes supporting task/action/provenance files. The wrapper checks those hashes; it does not infer terminal identity from an arbitrary screenshot.

Include excluded episodes with `eligible: false` and an `exclusion_reason`; they need no task/action/PNG fields. Retain `original_valid: false` for original invalids. The summary separates those original invalids, valid episodes lacking eligible evidence, missing new judgments and failed new judgments. Each distinct benchmark/harness/arm/model/protocol cohort has its own counts and valid-judgment success rate; different cohorts are not silently pooled.

The new directory saves `rejudge-plan.json`, exclusive episode claims, every prepared request and raw response, physical-ID reservations/settlements, per-episode results and `summary.json`. New judge costs stay separate from original actor costs and latencies. Changing the judge does not control differences in harnesses, action representations, image geometry, actor limits or collection windows. Raw prompts, images, trajectories and keys stay local.

Reproduce the published [all-Terra GPT figure](ARM_FINAL_RESULTS.md#luna-browsercode-effort-om2w300-20261030) from its public aggregate without API calls (rendered with Matplotlib 3.9.4):

```bash
"$API_PY" -m pip install matplotlib==3.9.4
"$API_PY" -B openwebrl/docs/arm_results/cuawright_browsercode_om2w_20261027/plot_om2w_terra_tradeoffs.py --output-dir /tmp/openwebrl-terra-plots
```

Add `--valid-only` to condition cost, latency and browser commands on each method's valid-task set as well as success. Add `--harness browsercode` for the focused BrowserCode view. This reads `om2w_terra_valid_only_aggregate.json`; the original aggregate and figure remain available. Costs retain original and retry work on recovered valid tasks, while excluding tasks left invalid. The [valid-task figure and coverage](ARM_FINAL_RESULTS.md#luna-browsercode-effort-om2w300-20261030) report missing observations explicitly.

</details>

<details>
<summary>Archived native endpoint eligibility (optional)</summary>

[`archived_endpoint.py`](../api_experiments/reproduce/archived_endpoint.py) publishes the exact generic eligibility algorithm used for archived native evidence. Its SHA256 is `2928f64f9988ab69197974d8018f7a7b3127ac2917457b304853720297933101`. It contains no private campaign discovery paths, task payloads or scheduler controls.

The read-only `native_endpoint(directory, task, canonical)` interface takes your saved episode directory, exact original task string and an independently qualified canonical row with `valid: true`. That canonical validity is an input, not an execution/accounting audit performed by this helper. The directory supplies ordered `browser-steps.jsonl`, `trajectory.json`, `external-judge-evidence.json`, the raw native `result.json`, original PNGs and, for manual captures, the selected step's code and hash. Action descriptions and saved final answers must match the archived judge evidence exactly.

```python
import sys
from pathlib import Path
sys.path.insert(0, "openwebrl/api_experiments/reproduce")
from archived_endpoint import native_endpoint

endpoint, bindings = native_endpoint(
    Path("/path/to/local/episode"), exact_archived_task, {"valid": True}
)
```

Use `endpoint["task"]`, `endpoint["actions"]`, `endpoint["screenshot_path"]` and `endpoint["screenshot_sha256"]` for the local rejudge manifest above; declare `endpoint_scope: "last_browser_action_capture"` and `last_action_capture_verified: true` only after this check succeeds. Convert `bindings` into `evidence_bindings` with `[{"path": path, "sha256": digest} for path, digest in bindings.items()]` to freeze the verified inputs.

The helper selects only the final logged browser action's capture, preserving its original supported geometry. Manual captures must be the final awaited screenshot operation, with no later browser/function side effects, including on the same source line. It rejects earlier-only images, changed task/action/capture hashes, ambiguous manual capture flow, paths outside the episode and incomplete actor endpoints. A rejection stays an explicit evidence exclusion; it does not trigger an actor rerun or image synthesis.

</details>

### Prepare the episode runtime

Setup assembles runnable copies from the frozen [configurations, prompts and support code](../api_experiments/portable_assets/). The public [OM2W](../data/eval/online-mind2web.jsonl), [WebVoyager](../data/eval/webvoyager_fara.jsonl) and [DeepShop](../data/eval/deepshop.jsonl) files supply instructions, start URLs and grading references. [tasks.py](../api_experiments/reproduce/tasks.py) joins them to fixed task-ID rosters in campaign order. No original campaign directory, scheduler registration, private trajectories or teacher labels are needed.

### What execution saves

Each run owns its ledgers, browser counters, request evidence, tracking identity and per-task outcomes. Existing valid or invalid results are never silently overwritten. Unknown usage retains its reservation; invalid attempts remain in all-attempt accounting. `summary.json` reports requested/recorded/valid/invalid counts, recorded and valid-only success rates, valid-task latency/browser-batch means with observation counts, separate model/judge fees or reservations, and cleanup status. W&B defaults to offline in `openwebrl-evals`; `--wandb-mode disabled` omits it. Raw task content and browser evidence stay local.

### Scope of reproduction

These commands perform **fresh collection** with frozen episode protocols and public task cohorts. They preserve candidate/selector settings, selected-only history, action parsing, benchmark judges and accounting. Generated manifests record path, dependency and process-ownership adaptations, including BrowserCode-only mechanical preflights. Archived source bytes remain unchanged.

Fresh runs cannot recreate historical site state, provider alias snapshots, scheduler contention, reused outcomes or original latency conditions. Use identical fresh task panels across arms and keep judge cohorts separate. A new live-site run can legitimately produce different outcomes. No command submits a Slurm job.

## Experiments

| Study | Methods | Tasks per method | Output cap / decision limit | Terminal judge |
|---|---|---|---|---|
| GPT high | Sol single; Luna single, N3, N5; all high | OM2W 300; WebVoyager 595; DeepShop 150 | 12,288 / 30 | OM2W: historical o4-mini; WebVoyager/DeepShop: benchmark-native GPT-4o |
| Claude | Opus 5.5 single; Sonnet 5.5 single, N3, N5; all high | OM2W 300; completed | 12,288 / 30 | GPT-5.6 Terra |
| Luna effort | Single xhigh/medium/low; N5 medium+medium and low+low | OM2W 300 recorded; provisional | 12,288 / 30 | GPT-5.6 Terra |

The output cap applies to each candidate and selector request. Effort and token caps do not enforce equal realized compute; measure actual usage. Each campaign fixes task order and shard assignments. Arms share task IDs but run independent browser trajectories; pilot tasks count toward the final roster. Judge protocols differ across studies and must remain labeled.

## Episode procedure

The following loop describes recent Luna effort and Claude BrowserCode runs. Earlier benchmark campaigns retain their frozen evidence and grading paths; see the [code map](#code-and-frozen-artifacts) and [result records](#results-and-analysis-pointers).

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

The GPT-high BrowserCode headline cohort retains historical o4-mini grading for OM2W and benchmark-native GPT-4o protocols for WebVoyager and DeepShop. The separate [official-WebJudge regrading study](ARM_INFERENCE_SCALING.md#om2w-grading-browser-audit-20261020) does not replace those headline outcomes. Recent Claude and Luna effort runs use `gpt56-terra-terminal-success-v1`: `gpt-5.6-terra`, medium effort, 4,096 completion tokens and seed 42. The request contains the task, selected action history and final screenshot. `common_judge.py` loads the frozen prompts and validates the completed, non-refusal verdict. The combined Online-Mind2Web GPT figure regrades qualified historical endpoints with this same Terra protocol; missing final evidence remains ungraded. The historical three-benchmark tables retain their original judge labels. Historical Claude grading/recovery attempts and fees remain in the lineage.

</details>

## Metrics and comparability

| Metric | Definition and reporting rule |
|---|---|
| Valid-only success | Successes / valid outcomes. Report success, valid and invalid counts separately; invalids are excluded. |
| Common-valid comparison | Use the same task intersection valid in every compared arm; label its size. |
| Model cost | All candidate, selector and applicable helper/counter fees in the stated cohort, including valid failures. Exclude the external judge. |
| Browser batches | Selected executable BrowserCode blocks. One block can contain several browser operations. |
| BrowserCode task latency | Actor-loop wall time, including model admission/HTTP waits and browser execution. Exclude setup, final judge-only capture, cleanup and external judging. |
| Tokens | Native input/output/cache usage. Output may already include reasoning tokens; do not count those twice. |

The Claude plots use each arm's valid cohort for success and all efficiency means. Their common-valid alternative uses the same 246 tasks for every metric. Missing measurements are never zero-filled. Plotted cohort costs and all-attempt campaign spending are separate quantities.

**Interpretation limits:** different valid subsets can change apparent rankings. Common-valid filtering does not remove selection from site, format or budget failures. Model budgets, judge protocols and live-site dates also differ. Treat comparisons as descriptive where those controls differ. Pareto lines connect observed nondominated points; they do not supply uncertainty or prove superiority.

## Results and analysis pointers

| Question | Evidence |
|---|---|
| Luna effort, 300 dispositions per arm; provisional | [Terra figure, prescribed retry counts and metric coverage](ARM_FINAL_RESULTS.md#luna-browsercode-effort-om2w300-20261030); [original aggregate](arm_results/luna_browsercode_effort_om2w300_20261030/aggregate.json) · [after prescribed retries](arm_results/luna_browsercode_effort_om2w300_20261030/retry_aggregate.json) |
| GPT high across three benchmarks | [Headline results](ARM_FINAL_RESULTS.md#headline-performance-cost-and-efficiency); [aggregate](arm_results/sol6_luna_three_benchmarks_20261019/aggregate.json) |
| Claude 300 tasks per arm | [Result tables and plots](ARM_FINAL_RESULTS.md#claude-browsercode-om2w300); [aggregate](arm_results/claude_browsercode_om2w300_20261028/aggregate.json); [accounting](arm_results/claude_browsercode_om2w300_20261028/accounting.json); [plot source](arm_results/claude_browsercode_om2w300_20261028/plot_claude_tradeoffs.py) |

Use the primary cohort tables for effects and uncertainty. Keep each valid denominator, judge protocol and collection period attached to its result. Luna effort collection is complete at 300 recorded dispositions per arm; its provisional publication does not certify final independent qualification or establish a causal gain over earlier judge/date cohorts.

## Code and frozen artifacts

The files under `api_experiments/` are **frozen source archives** grouped by harness and function. Original sibling imports and host-specific paths remain intact in the archive. The public [runtime builders](#reproduction-quickstart) assemble runnable copies with explicit portability overlays. Identical files are shared once. [source-provenance.json](../api_experiments/source-provenance.json) maps each campaign's original filename to its published path, SHA256 and byte count. Archived source bytes remain unchanged.

For BrowserCode, read the worker first, then its model, budget, browser environment and judge. The reproduction wrappers import `worker` from the generated runtime. [portable_assets/](../api_experiments/portable_assets/) supplies exact frozen support sources, effective scientific configurations, browser bridge and task-ID rosters; [variants/](../api_experiments/browsercode/variants/) preserves the additional executed campaign source snapshots.

| Campaign key | Episode worker | Actor and selector | Budget / admission | Browser and metrics | Judge |
|---|---|---|---|---|---|
| `browsercode_luna_effort` | [luna_effort_worker.py](../api_experiments/browsercode/workers/luna_effort_worker.py) | [luna_model.py](../api_experiments/browsercode/models/luna_model.py) | [request_budget.py](../api_experiments/browsercode/budgets/request_budget.py), [claude_budget.py](../api_experiments/browsercode/budgets/claude_budget.py) | [environments.py](../api_experiments/browsercode/runtime/environments.py), [metrics](../api_experiments/browsercode/runtime/luna_effort_episode_metrics.py) | [Terra](../api_experiments/browsercode/judges/terra.py) |
| `browsercode_claude` | [claude_worker.py](../api_experiments/browsercode/workers/claude_worker.py) | [claude_model.py](../api_experiments/browsercode/models/claude_model.py) | [claude_budget.py](../api_experiments/browsercode/budgets/claude_budget.py) | [environments.py](../api_experiments/browsercode/runtime/environments.py), [metrics](../api_experiments/browsercode/runtime/claude_episode_metrics.py) | [Terra](../api_experiments/browsercode/judges/terra.py) |
| `browsercode_gpt_high` | [gpt_high_worker.py](../api_experiments/browsercode/workers/gpt_high_worker.py) | [paired_model.py](../api_experiments/browsercode/models/paired_model.py) | [`ReservationBudget`](../api_experiments/browsercode/models/paired_model.py) in the model; browser `Budget` from frozen support | [environment](../api_experiments/browsercode/runtime/gpt_high_environments.py), [metrics](../api_experiments/browsercode/runtime/gpt_high_episode_metrics.py) | [Historical GPT](../api_experiments/browsercode/judges/historical_gpt.py) |

[BrowserCode controllers](../api_experiments/browsercode/controllers/) own batch workers; [protocols](../api_experiments/browsercode/protocols/) define financial enrollment and canary scope; [recovery](../api_experiments/browsercode/recovery/) preserves saved-judge recovery logic. The table maps published snapshots, not every historical variant in the result cohorts; use the provenance index and each cohort's frozen manifest to resolve exact executed sources.

### Offline validation

Run the self-contained archive, scientific-input, dry-run and accounting tests without API calls:

```bash
python3 -B -m unittest discover -s openwebrl/api_experiments/tests -v
```

[tests/](../api_experiments/tests/) covers hashes, rosters, the BrowserCode entry point, installation profiles, caps, isolation, cleanup and accounting. Installation-dependent cases require a dependency manifest; the local browser `check` command above validates the selected BrowserCode runtime directly. [validation/](../api_experiments/validation/) contains historical tests that may require original campaign directories.

## Archived implementations and campaign records

<details>
<summary>Advanced: CUA harnesses, earlier studies, source reconstruction and historical accounting</summary>

These implementations remain available for historical reproduction. They are outside the BrowserCode setup/run path. BrowserCode still uses CUAWright's shared `DefaultAgent`, workspace and model interfaces internally; retaining that library preserves parsing and history semantics. Installing the native harness extras is explicit:

```bash
python3.12 -B openwebrl/api_experiments/reproduce/install.py \
  --prefix "$API_WORK/dependencies" --profile all --install
```

The all-profile installation also fetches official WebJudge. Native execution additionally requires unprivileged user/PID namespaces. Its 20,000-token/100-decision protocol and command metrics differ from BrowserCode's 12,288-token/30-decision protocol. The original extension reused earlier 100-task outcomes (including invalids) for Sol single and Luna single/N3, added 200 fresh per method, and collected all 300 N5 tasks fresh. Its [qualified comparison](ARM_INFERENCE_SCALING.md#cuawright-native-full300-four-methods-20261025) and [coverage aggregate](arm_results/cuawright_browsercode_om2w_20261027/om2w_harness_comparison_valid_only.json) retain those distinctions.

### Native CUAWright experiments

[native.py](../api_experiments/reproduce/native.py) builds its own runtime from the same dependency manifest. It supports `luna_high_single`, `luna_high_n3_select_high`, `luna_high_n5_select_high`, `sol_high_single` and `sol_high_n3_select_high`:

```bash
"$API_PY" -B openwebrl/api_experiments/reproduce/native.py \
  --dependencies "$API_WORK/dependencies/dependencies.json" \
  --arm luna_high_n5_select_high --limit 1 --output "$API_WORK/native-luna-n5-01" \
  --model-usd 5 --judge-usd 1 --model-calls 250 --judge-calls 100 \
  --browser-starts 2 --timeout-seconds 1800
```

This is a read-only plan. Add `--execute` with `OPENAI_API_KEY` set to run it; all six USD/call/browser/time caps are required for execution. The default task file is public OM2W; `--limit 300` collects the full roster, while `--task-id` filters named tasks before `--limit` (raise the limit when selecting several IDs). Its browser cap includes one startup probe, so the example permits one probe plus one task browser. Native runs retain high effort, 20,000 output tokens, 100 decisions, summaries every 20 decisions, selected-only execution and official GPT-4o WebJudge. Original native format-repair defaults remain intact; the metered gateway never automatically repeats a physical provider request after an unknown-usage failure.

<details>
<summary>Earlier Sol 6.1 and Luna studies</summary>

The lower-level `prepare_browsercode.py` builder also accepts:

| Campaign key | Policies / executor |
|---|---|
| `browsercode_sol61_pair20`, `browsercode_sol61_pair80` | Sol 6.1 `serial_high` versus `parallel3_select_low`; use `--harness browsercode` or `--harness cuawright` |
| `browsercode_sol61_onecall20`, `browsercode_sol61_onecall80` | Sol 6.1 `onecall3_low`; choose either paired executor |
| `browsercode_sol61_low_medium30` | Sol 6.1 `single_low`, `single_medium`; choose either paired executor |
| `browsercode_luna_high100` | Luna `parallel3_select_high`, `parallel5_select_high`; later 12,288-token phase, BrowserCode |
| `browsercode_luna_high100_initial`, `browsercode_luna_high100_recovery4096` | Original and repaired 4,096-token Luna N3/N5 phases; BrowserCode |

Use `eval_gpt.py` with the prepared runtime; it validates that the arm/model/output limits match that campaign. The paired 20-task and 80-task extension cohorts have separate keys. Earlier cap-limited and later 12,288-token phases remain separate scientific records in the source inventory.

The original coordinate-browser studies use [legacy.py](../api_experiments/reproduce/legacy.py). Install their optional CPU-only dependencies and pinned Qwen tokenizer/vision metadata with `install.py --prefix "$API_WORK/dependencies" --install --legacy`; this downloads no model weights. Then:

```bash
"$API_PY" -B openwebrl/api_experiments/reproduce/legacy.py prepare \
  --campaign sol61_om2w100_medium --dependencies "$API_WORK/dependencies/dependencies.json" \
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

### Archived native code map

| Campaign key | Episode worker | Actor and selector | Budget / admission | Browser and metrics | Judge |
|---|---|---|---|---|---|
| `cuawright_native` | [campaign_worker.py](../api_experiments/cuawright/workers/campaign_worker.py), [native_episode.py](../api_experiments/cuawright/runtime/native_episode.py) | [proposal adapter](../api_experiments/cuawright/models/native_proposal_adapter.py), [transport](../api_experiments/cuawright/models/native_group_transport.py), [relay](../api_experiments/cuawright/models/native_group_relay.py) | [budget_gateway.py](../api_experiments/cuawright/budgets/budget_gateway.py) | [browser broker](../api_experiments/cuawright/runtime/browser_broker.py), [capture](../api_experiments/cuawright/runtime/native_capture.py) | [Official WebJudge wrapper](../api_experiments/cuawright/judges/external_judge.py) |

### Restore an original campaign layout

Use [materialize.py](../api_experiments/materialize.py) to reconstruct a campaign's original sibling filenames. Run from the repository root:

```bash
python3 -B openwebrl/api_experiments/materialize.py browsercode_luna_effort \
  --output /tmp/luna-effort-source
```

All campaign keys and file hashes are listed in [source-provenance.json](../api_experiments/source-provenance.json). The output directory must be new. All source hashes are checked before any output files are written. This copies code only; it starts no workers or API requests. For a runnable environment, use the installer and runtime builders above.

Credentials, `.env`, private task payloads, screenshots, trajectories, request logs, ledgers and scheduler records are excluded. The private Claude terminal-interruption qualifiers are also excluded because they embed task identities; the original worker still references them for historical recovery. The public builders provide the scientific configuration, support code and gateway assembly; the installer supplies pinned external code and browser dependencies. Original private campaign approvals and scheduler records are neither required nor copied into collaborator runs.

### Archived runtime checks

With the all-profile dependencies, enable the full runtime integration suite:

```bash
OPENWEBRL_API_DEPENDENCIES="$API_WORK/dependencies/dependencies.json" \
  "$API_PY" -B -m unittest discover -s openwebrl/api_experiments/tests -v
"$API_PY" -B openwebrl/api_experiments/reproduce/native_check.py \
  --dependencies "$API_WORK/dependencies/dependencies.json" \
  --output "$API_WORK/native-install-check" --browser-preflight
```

Coordinate runtimes support `legacy.py verify --runtime PATH --browser-preflight`. [verify_legacy.py](../api_experiments/reproduce/verify_legacy.py) additionally runs the original episode loop on diagnostic tasks through a fake SDK boundary, with real Chromium and tokenization, without provider calls.

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

</details>
