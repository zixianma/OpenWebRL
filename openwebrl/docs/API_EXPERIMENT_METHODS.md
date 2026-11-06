<a id="api-browser-evaluation-methods"></a>
# BrowserCode API experiments

We compare **one proposed action per decision** with **N proposed actions plus a selector**. Only the selected action executes. BrowserCode runs that action as JavaScript through `browser_execute`. The studies measure task success, model cost, browser activity and latency.

The completed local-browser [BrowserCode inference-scaling comparison](ARM_FINAL_RESULTS.md#browsercode-inference-scaling-20261031) shows eleven methods on OM2W, WebVoyager and DeepShop, all judged by **GPT-5.6 Terra**. Higher decision limits give mixed results: single120 has the strongest observed single-call valid-only success on WebVoyager/DeepShop, while single60 is higher on OM2W. N5/60 has the highest observed WebVoyager rate among the plotted methods. These are descriptive comparisons across live windows, runtime safeguards and different valid-task subsets. Historical single60 enforced 60 decisions but advertised 30 in its prompt; single120 enforces and advertises 120. The xhigh/1×, high/2× and high/0.5× token/effort variants show no consistent gain over high/1×. Actual tokens, cost, selected browser-code batches and per-task latency accompany the scores. The earlier [two-harness figure](ARM_FINAL_RESULTS.md#shared-terra-high-three-benchmarks-20261031), [lower-effort supporting results](ARM_FINAL_RESULTS.md#om2w-terra-supporting-effort-views) and [Claude comparison](ARM_FINAL_RESULTS.md#claude-browsercode-om2w300) remain separate historical views.

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
| `sol-high60` | `sol_single_high60` | 1,045 (300 / 595 / 150) |
| `sol-high120` | `sol_single_high120` | 1,045 (300 / 595 / 150) |
| `sol-xhigh30` | `sol_single_xhigh30` | 1,045 (300 / 595 / 150) |

`--limit` defaults to one task. For the full OM2W panel use `--limit 300`. For `gpt-crossbench`, use `--benchmark webvoyager --limit 595` or `--benchmark deepshop --limit 150`; filtering precedes the limit. GPT OM2W's four methods span `gpt` and `gpt-luna-single`. Luna N5 medium/low applies that effort to both candidates and selector. The full historical `browsercode_*` campaign keys remain accepted.

<a id="sol-high60"></a>
### Sol high single at 60 decisions

The [completed Sol/60 results](ARM_FINAL_RESULTS.md#browsercode-sol-high60) use **GPT-6 Sol, high effort, 12,288 output tokens, and 60 enforced and advertised decisions**. The actor deadline remains **900 seconds**; the judge is **GPT-5.6 Terra, medium effort, 4,096 completion tokens, seed 42**. There is one candidate per decision and no selector call.

```bash
python3.12 -B openwebrl/api_experiments/reproduce/browsercode.py setup \
  --workspace /tmp/sol-high-60 --campaign sol-high60 \
  --dependencies "$API_WORK/dependencies/dependencies.json"

"$API_PY" -B openwebrl/api_experiments/reproduce/browsercode.py run \
  --workspace /tmp/sol-high-60 --arm sol_single_high60 \
  --benchmark online-mind2web --limit 1 --output /tmp/sol-high-60-om-01 \
  --model-usd 5 --judge-usd 1 --model-calls 120 --judge-calls 1 --browser-starts 2
```

The run command is read-only until `--execute` is added. Use separate output directories and explicit caps for each full benchmark: `--benchmark online-mind2web --limit 300`, `--benchmark webvoyager --limit 595`, or `--benchmark deepshop --limit 150`. This campaign requires `--benchmark`; filtering occurs before task selection and the limit. The new runtime reuses the public campaign's globally unique task IDs and retains native task/source IDs from the pinned benchmark rosters. Task instructions, start URLs and source order match the native Sol evaluation exactly. Setup preserves the original sources and binds generated model, worker, prompt, judge and task files to an explicit protocol receipt; it makes no provider or browser calls.

The completed cohort has **1,045 recorded outcomes, 978 valid and 67 invalid**, with no unattempted tasks. Invalid outcomes remain separate from valid failures. [Per-benchmark rates and efficiency](ARM_FINAL_RESULTS.md#browsercode-sol-high60) use Sol's own valid tasks and are separate from the historical common-valid scaling plots; matched comparisons require a new intersection.


<a id="sol-high120"></a>
### Prospective Sol high single at 120 decisions

`sol-high120` raises both the enforced and advertised decision limit to **120**. It retains GPT-6 Sol high, the 12,288-output-token ceiling, one candidate, no selector, the 900-second actor deadline and the same Terra judging protocol. Its canonical 1,045-task roster is identical to Sol/60; benchmark outputs remain separate. This is a fresh experiment recipe, with no completed results implied.

New Sol/60, Sol/120 and Sol xhigh/30 setups share one HTTP gate across actor calls, input counters and Terra judging. Terra keeps its native `judge-UUID` identifier in request and budget receipts; only the gate slot key is normalized to UUID form. After updating the package, prepare a fresh runtime rather than editing source hashes in an existing run.

```bash
python openwebrl/api_experiments/reproduce/browsercode.py setup \
  --workspace /tmp/sol-high-120 --campaign sol-high120 \
  --dependencies /path/to/dependencies.json
python openwebrl/api_experiments/reproduce/browsercode.py check \
  --workspace /tmp/sol-high-120
python openwebrl/api_experiments/reproduce/browsercode.py run \
  --workspace /tmp/sol-high-120 --arm sol_single_high120 \
  --benchmark online-mind2web --limit 1 --output /tmp/sol-high-120-om-01 \
  --model-usd 1 --judge-usd 0.10
```

The final command is a zero-call dry plan; add `--execute` only for a separately budgeted run. Raising the step ceiling does not extend the actor deadline. Existing Sol/60 outcomes, invalids and source inventories remain unchanged.

<a id="sol-xhigh30"></a>
### Sol xhigh single at 30 decisions

`sol-xhigh30` uses **GPT-6 Sol xhigh, 12,288 output tokens and 30 enforced and advertised decisions** on the same 300 / 595 / 150 task roster. It retains one candidate, no selector, the 900-second actor deadline, and Terra medium/4,096/seed 42. This prepares a fresh reproduction; it implies no completed results.

```bash
python3.12 -B openwebrl/api_experiments/reproduce/browsercode.py setup \
  --workspace /tmp/sol-xhigh-30 --campaign sol-xhigh30 \
  --dependencies "$API_WORK/dependencies/dependencies.json"

"$API_PY" -B openwebrl/api_experiments/reproduce/browsercode.py run \
  --workspace /tmp/sol-xhigh-30 --arm sol_single_xhigh30 \
  --benchmark online-mind2web --limit 1 --output /tmp/sol-xhigh-30-om-01 \
  --model-usd 5 --judge-usd 1 --model-calls 60 --judge-calls 1 --browser-starts 2
```

The evaluation command is a read-only plan until `--execute` is added. Use separate benchmark outputs and explicit caps as above.

Fresh Sol setups bind a task-local truncation policy to their source receipt. An exactly matching native `incomplete/max_output_tokens` response that consumed all 12,288 output tokens and whose usage is already settled stays **invalid**; it is never retried, and subsequent untouched tasks may run. Transport, malformed-usage, counter, substitution and accounting failures retain the shared halt and conservative known/full-held charges. Existing halts are never cleared. This operational safeguard changes no actor request, token/step limit, price, judge, or run cap; archived runtimes and their invalid outcomes remain unchanged.

<a id="luna-xhigh-output-limits"></a>
### Luna single reasoning and output limits

For new Luna single experiments, set `--luna-single-output-tokens 6144`, `12288`, `24576` or `49152` and `--luna-single-reasoning-effort high` or `xhigh` during `luna` setup. Each setting needs a separate workspace. The generated runtime contains one arm and the same public OM2W 300, WebVoyager 595 and DeepShop 150 tasks; the completed output-cap comparison keeps 30 decisions and the shared Terra judge. The reasoning-versus-output-headroom comparison uses:

| Reasoning effort | Max output tokens per call | Relative output cap | Arm |
|---|---:|---:|---|
| xhigh | 12,288 | 1× | `luna_single_xhigh` |
| high | 24,576 | 2× | `luna_single_high` |
| high | 6,144 | 0.5× | `luna_single_high` |

For example, prepare high effort with the 2× ceiling:

```bash
python3.12 -B openwebrl/api_experiments/reproduce/browsercode.py setup \
  --workspace /tmp/luna-high-2x --campaign luna \
  --dependencies "$API_WORK/dependencies/dependencies.json" \
  --luna-single-output-tokens 24576 --luna-single-reasoning-effort high

"$API_PY" -B openwebrl/api_experiments/reproduce/browsercode.py run \
  --workspace /tmp/luna-high-2x --arm luna_single_high \
  --benchmark webvoyager --limit 1 --output /tmp/luna-high-2x-wv-01 \
  --model-usd 5 --judge-usd 1
```

The 1× reference inherits the original high-effort ceiling of 12,288 tokens per generation. The [initial calibration protocol](../api_experiments/portable_assets/snapshots/calibration/protocol.json) used that ceiling for a high-effort single call and 4,096 for each of three low-effort candidate calls, plus a separate 2,048-token selector. This establishes the historical convention; it does not establish an optimal ceiling or equal realized compute.

The second command is a read-only plan; add `--execute` to spend within its caps. Select a full benchmark with `--benchmark online-mind2web --limit 300`, `--benchmark webvoyager --limit 595` or `--benchmark deepshop --limit 150`. For fixed shards, `--task-ids /path/to/ids.json` selects a unique ordered JSON list of IDs after benchmark filtering and before the limit; its hash and selected IDs are recorded. The generated runtime binds the exact effort and cap to requests, validation, reservations and metrics, and retains its original sources and protocol hashes. A higher ceiling permits longer outputs; inspect actual output usage and truncations alongside success, cost and latency. The earlier `--luna-xhigh-output-tokens CAP` option remains an xhigh-only alias; setup without either output-cap option retains the frozen historical campaign.

<a id="luna-single-decision-limits"></a>
### Prospective Luna single decision limit

To isolate a larger action horizon, add `--luna-single-max-decisions 60` with explicit `high` effort and the unchanged 12,288-token output ceiling. The default remains 30. The published 60-decision cohort is complete. Its enforced limit is 60, while the system prompt retains the historical 30-decision wording; the full BrowserCode system prompt is identical to the 30-step controls. Executing this recipe creates a fresh independent reproduction on the same 1,045-task source; it leaves archived runs unchanged.

```bash
python3.12 -B openwebrl/api_experiments/reproduce/browsercode.py setup \
  --workspace /tmp/luna-high-60steps --campaign luna \
  --dependencies "$API_WORK/dependencies/dependencies.json" \
  --luna-single-output-tokens 12288 --luna-single-reasoning-effort high \
  --luna-single-max-decisions 60

"$API_PY" -B openwebrl/api_experiments/reproduce/browsercode.py run \
  --workspace /tmp/luna-high-60steps --arm luna_single_high \
  --benchmark online-mind2web --limit 1 --output /tmp/luna-high-60steps-om-01 \
  --model-usd 4 --judge-usd 3 --model-calls 120 --judge-calls 1 --browser-starts 2
```

The run command prints a read-only plan. The actual agent limit, model decision guard, saved identity and tracking all use 60. The actor-loop wall deadline stays **900 seconds**: episodes may time out before reaching the larger limit. Report timeouts, reached decision limits, actual decisions, output/reasoning usage, valid/planned counts and paired common-valid success alongside cost and latency. The shared Terra protocol and disabled automatic provider retries remain unchanged. A full campaign needs separate explicit compute/API caps; the example does not allocate cluster resources.

<a id="luna-single-120-decisions"></a>
### Luna high single at 120 decisions

This setting keeps **high reasoning, 12,288 output tokens per generation, the 900-second actor-loop deadline and Terra medium/4,096/seed 42**. Both the actual model/agent ceiling and the advertised prompt limit become **120 decisions**. The completed 60-step cohort enforced 60 while retaining the historical prompt wording of 30; comparing it with this new setting therefore changes both the enforced and advertised limits. Preserve that distinction alongside live collection dates and validity cohorts.

```bash
python3.12 -B openwebrl/api_experiments/reproduce/browsercode.py setup \
  --workspace /tmp/luna-high-120steps --campaign luna \
  --dependencies "$API_WORK/dependencies/dependencies.json" \
  --luna-single-output-tokens 12288 --luna-single-reasoning-effort high \
  --luna-single-max-decisions 120

"$API_PY" -B openwebrl/api_experiments/reproduce/browsercode.py run \
  --workspace /tmp/luna-high-120steps --arm luna_single_high \
  --benchmark online-mind2web --limit 1 --output /tmp/luna-high-120steps-om-01 \
  --model-usd 4 --judge-usd 3 --model-calls 240 --judge-calls 1 --browser-starts 2
```

The run command is a read-only plan until `--execute` is added. The generated runtime contains the same pinned 1,045 tasks across OM2W, WebVoyager and DeepShop. Setup binds the 120-step instruction, model guard, agent ceiling, identity and tracking to a new source-bound receipt; archived 30/60-step settings are retained. A task may finish or hit the unchanged 900-second deadline before 120 decisions. The qualified cohort records 1,045 native outcomes (946 valid, 99 invalid); the [per-benchmark results](ARM_FINAL_RESULTS.md#browsercode-inference-scaling-20261031) keep native counts and valid-only metric populations explicit. This recipe prepares an independent reproduction of that setting.

<a id="luna-selection-decision-limits"></a>
### Luna N2/30 and N5/60 selection experiments

These completed settings use Luna high for both proposals and selection, with a **12,288-output-token ceiling per generation** and the same **900-second actor-loop deadline**. N2 retains the default 30-decision horizon; N5 permits 60. Candidate generations run concurrently from identical observed history, followed by one selector call. Only the chosen action executes and enters subsequent history. The cap applies independently to every candidate and selector; it is not a total token allowance per decision or task.

| Setting | Candidates / decision | Selector calls / decision | Max decisions | Arm |
|---|---:|---:|---:|---|
| Luna high N2 | 2 | 1 | 30 | `parallel2_select_high` |
| Luna high N5 | 5 | 1 | 60 | `parallel5_select_high` |

Use separate new workspaces and explicitly pair both selection flags. Each generated runtime contains one method on the same pinned OM2W 300, WebVoyager 595 and DeepShop 150 tasks as the completed single-call sweep. All three benchmarks use Terra medium, 4,096 completion tokens and seed 42. Setup without these flags retains its historical campaign defaults.

```bash
python3.12 -B openwebrl/api_experiments/reproduce/browsercode.py setup \
  --workspace /tmp/luna-n2-30steps --campaign gpt-crossbench \
  --dependencies "$API_WORK/dependencies/dependencies.json" \
  --luna-selection-candidates 2 --luna-selection-max-decisions 30
python3.12 -B openwebrl/api_experiments/reproduce/browsercode.py setup \
  --workspace /tmp/luna-n5-60steps --campaign gpt-crossbench \
  --dependencies "$API_WORK/dependencies/dependencies.json" \
  --luna-selection-candidates 5 --luna-selection-max-decisions 60

"$API_PY" -B openwebrl/api_experiments/reproduce/browsercode.py run \
  --workspace /tmp/luna-n2-30steps --arm parallel2_select_high \
  --benchmark online-mind2web --limit 1 --output /tmp/luna-n2-30steps-om-01 \
  --model-usd 5 --judge-usd 1 --model-calls 180 --judge-calls 1 --browser-starts 2
"$API_PY" -B openwebrl/api_experiments/reproduce/browsercode.py run \
  --workspace /tmp/luna-n5-60steps --arm parallel5_select_high \
  --benchmark online-mind2web --limit 1 --output /tmp/luna-n5-60steps-om-01 \
  --model-usd 5 --judge-usd 1 --model-calls 720 --judge-calls 1 --browser-starts 2
```

The run commands print plans; add `--execute` for paid collection within their explicit caps. Model-call caps include proposals, selectors and native input counters. For a full benchmark, select its 300/595/150-task limit and budget it separately. The source-bound selection receipt retains the original model, worker and judge files, and validates the generated task roster, prompt, actual model guard, agent limit, raw model payloads and Terra protocol before execution. Archived experiments remain unchanged. Report success over planned tasks and valid-only success with counts, common-valid paired comparisons, actual tokens, model cost, browser-code batches and **elapsed actor latency per task**. Parallel API durations are not summed into actor latency. The qualified N2/30 and N5/60 cohorts each retain 1,044 native results and one original missing outcome: administrative missing on DeepShop for N2, and claimed-missing on OM2W for N5/60. Missing outcomes are not imputed as failures or retried. Their overall confirmed-success rates use the original planned denominators and remain lower bounds in those cells; the [main figure](ARM_FINAL_RESULTS.md#browsercode-inference-scaling-20261031) uses each method's own valid tasks. These recipes prepare independent reproductions.

<a id="fixed-cached-luna-selection"></a>
### Fixed cached-context Luna selection

Use `--luna-cached-selection luna_n3_cached_high30` or `luna_n5_cached_high60` with `--campaign luna` to prepare the cached-context implementation on the pinned OM2W 300, WebVoyager 595 and DeepShop 150 tasks. Both use `gpt-6-luna` high, a 12,288-output-token ceiling for each proposal and selector, a 900-second actor deadline, and `gpt-5.6-terra` medium with 4,096 completion tokens and seed 42. The actual and advertised decision limits are 30 and 60, respectively.

All N proposals run concurrently on identical native actor input. The selector receives that full actor context, including earlier selected encrypted reasoning and executed tool history, followed by a developer instruction and indexed proposal actions. It retains the actor tool definition, uses `tool_choice=none`, and returns a strict `selected_index` schema. Only the chosen candidate executes; its native response and the actual observation enter the next actor turn. Unselected candidates' reasoning never enters later actor context.

This changes the historical selector's context projection. Cached N5/60 also advertises 60 decisions; historical N5/60 advertised 30 while enforcing 60. Comparison with historical selection therefore does **not isolate provider caching alone**. Prefix reuse depends on the provider; measure actual cache telemetry and retain unknown usage conservatively. Actor latency is elapsed time per task, including proposal generation, selection and browser execution, excluding browser initialization and external Terra judging. The [completed cached-context results](ARM_FINAL_RESULTS.md#browsercode-cached-selection-complete-20261105) include separate own-valid and freshly intersected fourteen-method figures. Historical snapshots retain their original populations; these recipes prepare independent new reproductions.

```bash
python -B openwebrl/api_experiments/reproduce/browsercode.py setup \
  --campaign luna --luna-cached-selection luna_n3_cached_high30 \
  --workspace /tmp/luna-cached-n3 --dependencies /path/to/dependencies.json
```

A timeout during an input-token-count request still leaves that task invalid. If the counter has a verified released HTTP lease and exactly zero reserved cost, the cached actor runner preserves its unknown-usage record, physical call count and zero hold, then permits the next task. Billable unknown usage, malformed evidence and other failures retain the original halt behavior. The run plan binds the [cleanup guard](../api_experiments/reproduce/zero_price_counter_cancel.py) by source hash and checks it before execution.

For N5/60, use `luna_n5_cached_high60` and a new workspace. The ordinary `check`, `run` and `report` commands apply. Setup performs no browser or model work. Run prints a plan until `--execute` is supplied; provide explicit actor, judge, physical-call and browser-start caps. Preparation grants no spending authorization. Generated source manifests bind the method, roster, prompts, worker limits and original archived runtime files before execution.

<a id="cached-selection-completed-figures"></a>
### Reproduce the completed fourteen-setting figures without API calls

The [completed comparison](ARM_FINAL_RESULTS.md#browsercode-cached-selection-complete-20261105) contains the eleven gap-repaired historical settings, Sol high/60, cached N3/30 and cached N5/60. Its new common-valid intersection contains 230 OM2W, 505 WebVoyager and 122 DeepShop tasks. Separate own-valid figures use each method's valid outcomes; both views apply the same fixed planned benchmark weights to success, cost, elapsed latency and selected batches. The public aggregate includes all per-benchmark numerators, denominators and metric coverage, so the figures can be regenerated without task payloads or private operational records.

Install Matplotlib, then run from the repository root into a fresh output directory:

```bash
MPLCONFIGDIR=/tmp/browsercode14-mpl python3 -B \
  openwebrl/docs/arm_results/browsercode_cached_selection_20261105/plot_browsercode_scaling14.py \
  --aggregate openwebrl/docs/arm_results/browsercode_cached_selection_20261105/aggregate.json \
  --output-directory /tmp/browsercode14-render
```

The renderer validates the exact fourteen-method matrix, shared Terra protocol, complete metric coverage and explicit population definitions before producing both three-panel PNG/PDF/SVG figures, coordinate CSVs and render provenance. It preserves existing outputs, so use another fresh directory to repeat the command. This reproduces published aggregate figures without model or browser calls; it does not replace a native experiment rerun. Use the [fixed cached-context recipe](#fixed-cached-luna-selection) for new bounded evaluations. Historical and cached selectors differ in context projection, cached N5/60 also changes the advertised limit, and collection windows differ; these figures do not isolate the causal effect of caching.

<a id="browser-hosting-and-traffic-costs"></a>
### Browser hosting and traffic costs

The [completed full-300 comparison](#us-stealth-full300) reports measured hosting and residential-proxy fees separately from actor/Terra pricing and task-performance populations. **The earlier 20-task US-residential pilot cost $5.74 across both methods**, including model calls, Terra, hosting, proxy traffic and failed attempts. On 15 contemporaneous common-valid pairs, N5 succeeded on 15 and Single on 12, with mean actor-loop latency of 191.6 versus 120.8 seconds. This small pilot does not establish full-benchmark superiority; [results and accounting are below](#measured-us-residential-pilot).

<details>
<summary>Historical hosting projections and zero-API reproduction — omitted from the plots</summary>

For the completed local-browser scaling cohorts, we can project hosting charges without rerunning the experiments; their cloud traffic remains unmeasured. All 7,909 valid task–method observations in the completed 24 points have a saved first-close browser lifetime. At the [current Browser Use V4 rate](https://browser-use.com/pricing), projected hosting is **$0.00070–$0.00176 per valid task**, excluding traffic. This is a projection from local Chromium durations, not measured stealth-browser performance or spending; legacy account terms may differ.

Use **$0.02 per browser-hour**, rounding each session up to whole minutes with a one-minute minimum. Add network usage separately: direct/custom-proxy traffic is **$0.20/GB** and managed residential traffic **$5/GB**. The provider's [billing examples](https://docs.browser-use.com/cloud/guides/billing) convert reported MB to GB by dividing by 1,024. For example, an assumed 10 MB/task adds $0.001953 direct or $0.048828 residential, before hosting. Traffic is unmeasured for these local runs; batch counts are not network-byte measurements. N3/N5 still use one browser session per task; candidate/selector computation can keep that session open but does not create three or five browsers. Our agents call their model providers directly, so a hosted-agent orchestration fee is not part of this projection.

USD per valid task; traffic, model calls and judging excluded. Every mean below covers all of its method's valid tasks.

| Method | OM2W | WebVoyager | DeepShop |
|---|---:|---:|---:|
| Sol high single | 0.000702 | 0.000725 | 0.001037 |
| Luna high · 1× cap | 0.000700 | 0.000739 | 0.001062 |
| Luna high N3 | 0.001172 | 0.001087 | 0.001633 |
| Luna high N5 | 0.001338 | 0.001148 | 0.001764 |
| Luna xhigh · 1× cap | 0.000956 | 0.000785 | 0.001113 |
| Luna high · 2× cap | 0.000881 | 0.000750 | 0.000971 |
| Luna high · 0.5× cap | 0.000850 | 0.000756 | 0.000978 |
| Luna high · 60 steps | 0.001027 | 0.000897 | 0.001082 |

For each recorded task, use the first successful browser-close receipt, preferring `browser_cleanup_before_judge.owned_browser_seconds`. A repeated final close can report time elapsed after the browser already stopped, including judging, and is not substituted for that first receipt. The local interval includes process startup, task setup, actor work, terminal capture and shutdown. It is not a guarantee of the same interval on a remote browser. The 3,000 token/effort-sweep valid tasks and 993 additional 60-step valid tasks have full recorded process-group closure; the 3,916 historical valid tasks retain their older terminal-root-only proof.

Across the earlier 21-point snapshot’s 7,315 recorded task dispositions, 7,310 lifetimes are observed; five interrupted invalid WebVoyager lifetimes remain missing. Their hosting projection is **$6.770333 for the known durations**, including recorded invalid tasks. Startup probes and abandoned historical attempts are outside this projection, so this is not a full historical all-attempt bill. Missing durations and traffic remain missing, never zero.

The [aggregate](arm_results/browsercode_inference_scaling_20261031/browser_cost_projection.json) contains per-session-rounded minute sums and separate valid/invalid coverage. Reproduce the [scenario CSV](arm_results/browsercode_inference_scaling_20261031/browser_cost_scenarios.csv) without model or browser calls:

```bash
python3 -B openwebrl/docs/arm_results/browsercode_inference_scaling_20261031/project_browser_cost.py \
  --traffic-mb 0 10 50 100 --output /tmp/browser_cost_scenarios.csv
```

</details>

<details>
<summary>Measured 20-task US-residential pilot — results, costs and reproduction</summary>

<a id="measured-us-residential-pilot"></a>

**N5 completed three additional tasks on the same 15 valid pairs, at 1.59× the mean actor-loop latency.** Both succeeded on 12; N5 alone succeeded on three. The other nine actor attempts remain invalid: eight setup/infrastructure failures and one final-capture timeout. A deferred Single task completed later and is outside this paired comparison. Attrition and the small sample prevent treating these results as full-300 performance estimates.

| Method | Planned tasks | Valid tasks | Invalid tasks | Successes | Overall success (%) | Valid-only success (%) |
|---|---:|---:|---:|---:|---:|---:|
| Luna high Single | 20 | 16 | 4 | 13 | 65.0 | 81.25 |
| Luna high N5 | 20 | 15 | 5 | 15 | 75.0 | 100.0 |

The primary paired rates are **12/15 (80.0%) for Single and 15/15 (100.0%) for N5**. The rows above use each method's own valid cohort; the Single row includes its later unpaired success.

**All-attempt charges, USD:** all 40 actor dispositions and both failed mechanical preflights are included, with 21 owned browser sessions per method. Every session was verified stopped and queried again at least 90 seconds after its first verified stop. All model usage and latest provider fee fields were known. Provider session-API amounts are usage charges, not final invoices or cash paid after account credits.

| Method | Actor | Terra | Hosting | Residential proxy | Total | Proxy traffic (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Luna high Single | 0.2116 | 0.2664 | 0.0180 | 1.9107 | 2.4067 | 391.30 |
| Luna high N5 | 1.2258 | 0.2430 | 0.0213 | 1.8409 | 3.3310 | 377.02 |

Combined usage was **$5.7377**: $1.9468 for actors/Terra, $0.0393 for hosting and $3.7516 for residential traffic. Proxy traffic dominated browser charges; browser rental alone would substantially understate this pilot's cost. All eight allocations, including failed starts, consumed 5,975 of the approved 14,400 cumulative seconds with two CPUs, 16 GiB and no GPU.

**Timing means on the same 15 common-valid pairs, seconds per task:**

| Interval | Single | N5 |
|---|---:|---:|
| Actor loop, excluding Terra | 120.8 | 191.6 |
| Actor API wall time, including input-token counters | 78.4 | 162.4 |
| Selected browser-code execution | 40.5 | 25.6 |
| Browser/task setup | 5.0 | 5.0 |
| Terminal capture | 0.88 | 0.79 |
| Terra judging | 2.47 | 2.05 |
| First-close owned browser lifetime | 127.3 | 198.0 |

Actor API wall time is the union of overlapping call intervals; N5's summed call durations were 391.6 seconds per task, which is not elapsed latency. Its candidate calls actually overlapped. Setup, browser execution, capture and model intervals can be nested; do not add these rows. The browser stopped before Terra. The at-least-90-second fee reconciliation gate is outside the actor loop and browser lifetime, and can overlap judging.

The [aggregate](arm_results/browsercode_us_stealth_pilot_20261101/aggregate.json) and [CSV](arm_results/browsercode_us_stealth_pilot_20261101/summary.csv) separate overall, valid-only and common-valid populations and retain unrounded charges. Raw task content, provider identifiers and private diagnostics are not published.

**Reproduce a new pilot.** Use **20 fixed OM2W tasks × two methods: Luna high single and Luna high N5**, yielding 40 actor sessions. This pilot measures actual hosting/traffic cost and checks the comparison before separately budgeting the full 300-task benchmark. It cannot establish full-benchmark performance. Select tasks without outcomes: rank hosts and tasks by SHA256 with seed 42 and cycle through hosts for website coverage. Both arms receive the same tasks, with fresh sessions; preserve invalids and every charge.

Keep BrowserCode, high effort, 12,288 tokens per candidate/selector call, 30 decisions, the 900-second actor-loop deadline, raw 1280 × 1800 terminal screenshots and Terra medium/4,096/seed 42. Connect our existing agent over CDP to a standalone Browser Use V4 stealth browser. **Both arms use US residential routing (`proxyCountryCode: "us"`)**, explicitly requested for this study, with no proxy fallback or shared browser profile. N5 samples its five candidates concurrently before selecting one; it uses one browser per task. Run the two arms of each task contemporaneously, alternate worker assignment, and await both before releasing the next pair. Mechanical preflights and failed sessions must be separately counted and charged. Both arms enable `allowResizing: true` so CDP can enforce the common viewport; the provider notes that this opt-in can reduce stealthiness. The adapter verifies the actual layout and uncropped 1280 × 1800 PNG without resampling. Reported screen and device-scale fingerprints are retained as diagnostics.

The zero-API [pilot planner](../api_experiments/reproduce/stealth_plan.py) verifies the full 300-task reference identity set, freezes the selected pairs and records proposed resource/call/budget limits. Its output contains task identities and stays local. Prepare a **new** generated runtime, then apply the prospective [cloud/Terra adapter](../api_experiments/reproduce/cloud_browsercode.py); original campaign snapshots remain preserved. These commands create no sessions or SDK clients:

```bash
export STEALTH_WORK=/path/on/shared/storage/openwebrl-om2w-stealth
python3.12 -B openwebrl/api_experiments/reproduce/browsercode.py setup \
  --workspace "$STEALTH_WORK" --campaign gpt-luna-single \
  --dependencies "$API_WORK/dependencies/dependencies.json"
export PYTHONPATH="$PWD/openwebrl/api_experiments${PYTHONPATH:+:$PYTHONPATH}"
"$API_PY" -B -c 'from pathlib import Path; import sys; from reproduce.cloud_browsercode import configure; configure(Path(sys.argv[1]))' \
  "$STEALTH_WORK/runtime"
python3 -B openwebrl/api_experiments/reproduce/stealth_plan.py \
  --tasks "$STEALTH_WORK/runtime/tasks.json" --output "$STEALTH_WORK/proposal.json"
"$API_PY" -B -m reproduce.stealth_pilot --runtime "$STEALTH_WORK/runtime" \
  --proposal "$STEALTH_WORK/proposal.json" --output "$STEALTH_WORK/results" \
  > "$STEALTH_WORK/execution-plan.json"
```

The [pilot owner](../api_experiments/reproduce/stealth_pilot.py) proposes $1 for single actors, $3 for N5 actors and $2 for Terra, with a separate $10 browser spending target. It allows at most 40 actor sessions plus two mechanical preflights, 8,400 actor/input-counter HTTP calls and 40 first Terra calls. It starts no new pair after $8 of conservative observed browser charges and requests shutdown at $10; this is not a hard provider-dollar cap. It rechecks each pair's stopped-session fees at least 90 seconds after the first close before admitting the next pair. Maximum-observed charges are conservative admission reserves; actual cost tables use the separate latest provider receipts, including downward corrections.

Paid execution requires **one node, two CPUs, 16 GiB, no GPUs and at most four cumulative hours across all attempts**, with no implicit local mode or requeue. A routine recovery must preserve already spent time, task dispositions, fees and call credits. Keep the source checkout, dependency paths, generated runtime, key file and output on shared storage visible from the compute node; login-node `/tmp` is not a shared deployment location. Source manifests also bind the support files imported from the checkout, so copying only the launch script is insufficient. The owner verifies the actual allocation, reserves time for teardown, and awaits both workers.

A separate reviewed approval JSON must bind `plan_sha256` to SHA256 of `json.dumps(plan, sort_keys=True, default=str).encode()` for the printed execution plan, with `approved`, `proxy_overshoot_authorized`, `openai_payload_egress_approved` and `browser_use_payload_egress_approved` explicitly true. Only after that approval, add `--execute --approval /path/to/approval.json --env-file /path/to/private.env` to the pilot command inside the approved allocation. The key file needs `BROWSER_USE_API_KEY`, `OPENAI_API_KEY` and optionally `JUDGE_API_KEY`. This owner is separate from the ordinary local `browsercode.py run` frontdoor, which refuses a cloud-marked runtime. A source/protocol change requires fresh preparation and approval; no automatic creation or episode retries are enabled.

Bind each task/attempt to its created provider browser ID and immutable create request. Disable automatic creation retries; reconcile an ambiguous creation before admitting another session. Stop the owned browser with the provider API, verify `status=stopped`, then retain raw post-stop [browser usage receipts](https://docs.browser-use.com/cloud/api-v4/browsers/get-browser-session): `startedAt`, `finishedAt`, `browserCost`, `proxyCost` and `proxyUsedMb`. Disconnection alone does not stop billing. Poll within a bounded settlement window and retain unresolved costs explicitly. Parse present, finite nonnegative decimal amounts without replacing absent/null values with zero.

Report provider-recorded hosting, network/proxy and combined usage per task and for **every attempt**, including invalids, startup probes and failures; model and Terra fees remain separate. Provider usage is not necessarily net cash paid after credits or promotions. Cross-check project credits/invoices when attributable; shared account balance changes alone cannot assign costs to this campaign.

Record provider lifetime alongside model-call intervals, browser-command execution and setup/capture/cleanup intervals. Use the **union of overlapping N5 model-call intervals** for wall time, rather than summing five concurrent durations. Preserve overlap and unclassified time explicitly. An allocated cloud browser can remain billable while inference is in flight; browser-command time is a diagnostic, not a substitute for provider charges. Report overall and valid-only success with separate counts, plus paired common-valid results. A later comparison with historical local-browser runs remains descriptive because collection windows differ.

Before paid collection, freeze and explicitly approve exact session/compute/model/judge allowances and the browser spending policy. Verify charge monitoring and teardown. The current standalone-browser schema exposes no per-session traffic-dollar ceiling; the provider describes API-key spending limits as [soft limits](https://docs.browser-use.com/cloud/guides/billing). Polling can stop admission and request shutdown after observed usage thresholds, but cannot guarantee that already-running traffic stays below an exact dollar amount. Require an explicit policy for that possible overshoot; do not silently turn a spending target into a claimed hard cap. Reproduction needs its own exact approval; the completed pilot does not authorize a full-300 follow-up.

</details>

<a id="us-stealth-full300"></a>
### Completed full-300 US-residential comparison

**Latest native hosting plus residential-proxy fees are $22.5053 for Single and $24.2542 for N5**, covering all 608 owned sessions, including eight probes and invalid/recovery attempts. All hosting/proxy fees are observed, all sessions are stopped, and the final GET snapshots were taken at least 90 seconds after verified stop. These are native provider observations, not invoices. The [separate result figure and counts](ARM_FINAL_RESULTS.md#browsercode-us-stealth-full300) compare the same 270 common-valid tasks: 247 Single successes versus 255 N5 successes. The paired difference is +2.96 percentage points (two-sided task-level exact McNemar p = 0.134). The discordant outcomes do not establish a difference at the 0.05 level. This single-model, Single-versus-N5 comparison does not establish an inference-scaling law.

<details>
<summary>Full-300 measured accounting, performance means and protocol</summary>

**All-attempt USD:** API columns are priced native token-usage upper bounds, including invalid work and input counters; complete cache telemetry can make upper and lower bounds coincide. They exclude browser fees and are not direct invoice charges. Browser columns are latest native reported fees, with hosting and traffic separate; they include 300 actor sessions and four probes per method. Do not substitute historical maximum observations for these fee fields.

| Method | Actor API upper $ | Terra API upper $ | Hosting observed $ | US proxy observed $ | Browser subtotal $ |
|---|---:|---:|---:|---:|---:|
| Luna high single | 3.1875 | 4.6844 | 0.2547 | 22.2506 | 22.5053 |
| Luna high N5 | 22.4488 | 4.5763 | 0.4003 | 23.8539 | 24.2542 |

All 37,231 physical actor/input-counter/Terra requests have known native usage. No active, usage-unknown or known-but-unsettled requests remain, and no API or provider-fee reserves remain. Native usage pricing still gives upper bounds rather than direct invoice charges.

**Own-valid actor means:** cost and elapsed latency exclude Terra. These rows include each method’s valid failures and have different task populations; the primary figure uses their common-valid intersection.

| Method | Valid tasks | Priced tasks | Actor upper $ / priced task | Actor latency (s/task) | Selected batches / task |
|---|---:|---:|---:|---:|---:|
| Luna high single | 281 | 281 | 0.01098 | 120.2 | 16.65 |
| Luna high N5 | 280 | 280 | 0.07710 | 205.4 | 14.29 |

The common-valid discordant cells are **7 Single-only successes and 15 N5-only successes**; both succeed on 240 and both fail on 8. The exact paired test conditions on these discordants and treats task pairs as independent; it does not correct for website clustering. Own-valid rates, overall rates and common-valid rates use distinct denominators and are not interchangeable.

Both arms enable `allowResizing: true` to enforce the common viewport, an opt-in the provider warns can reduce stealthiness. The final uncropped 1280 × 1800 PNG is verified without resampling; screen/DPR fingerprints remain diagnostics. Temporary browser connections and remote owned sessions are closed even after setup failures. The browser stops before Terra. Pairing is contemporaneous within each original task pair; failed claims remain invalid and only genuinely unclaimed tasks continue during bounded recovery. All failed launches, probes, API calls and provider fee histories remain in all-attempt accounting.

[Aggregate](arm_results/browsercode_us_stealth_full300_20261102/aggregate.json) · [Paired statistics](arm_results/browsercode_us_stealth_full300_20261102/paired_statistics.json) · [Full metric/accounting table](arm_results/browsercode_us_stealth_full300_20261102/us_stealth_observed_costs_tables.md) · [CSV](arm_results/browsercode_us_stealth_full300_20261102/us_stealth_observed_costs.csv) · [PDF](arm_results/browsercode_us_stealth_full300_20261102/us_stealth_observed_costs.pdf). Recreate the compact comparison from sanitized aggregates only, with no API, browser or scheduler calls:

```bash
STEALTH_RESULTS=openwebrl/docs/arm_results/browsercode_us_stealth_full300_20261102
"$API_PY" -B "$STEALTH_RESULTS/plot_stealth_costs.py" \
  --aggregate "$STEALTH_RESULTS/aggregate.json" --population common-valid \
  --output-dir "$STEALTH_RESULTS"
```

</details>

The [full-study owner](../api_experiments/reproduce/stealth_study.py) reproduces **300 fresh OM2W task pairs: Luna high single versus Luna high N5**. The completed 20-task pilot remains separate. Both methods retain BrowserCode, US residential routing, high reasoning, 12,288 tokens per generation, 30 decisions, a 900-second actor loop and Terra medium/4,096/seed 42. N5 proposals run concurrently; only the selected action executes.

<details>
<summary>Full-study reproduction and spending controls</summary>

Prepare a new cloud-marked runtime with the setup and cloud-adapter steps above, then print two read-only plans:

```bash
python3.12 -B -m openwebrl.api_experiments.reproduce.stealth_study \
  --runtime "$STEALTH_WORK/runtime" --output "$STEALTH_WORK/full300-results" --shard 0
python3.12 -B -m openwebrl.api_experiments.reproduce.stealth_study \
  --runtime "$STEALTH_WORK/runtime" --output "$STEALTH_WORK/full300-results" --shard 1
```

Each shard contains 150 disjoint tasks and runs both methods contemporaneously, alternating their worker assignment. The union is the pinned full 300-task roster. At most four browsers and twelve actor HTTP calls run concurrently. Every previous pair must have stopped browsers and known provider fees rechecked at least 90 seconds after stop before its shard admits another pair. The source-bound Playwright lifecycle adapter closes temporary connections even when page setup fails; the failed task remains invalid.

The fixed preparation requests **$60 in hard model/judge caps**: $8 single actors, $36 N5 actors and $16 Terra. The shared provider pool stops new pairs at $80 of conservative observed browser/proxy charges and requests shutdown at a **$90 operational target**; the inherited per-session proxy watchdog remains $10. These provider thresholds can be exceeded by in-flight or delayed charges and require explicit overshoot approval. Final reported costs use latest native provider receipts, with hosting and proxy traffic separate from model/judge costs.

The maximum is 600 actor sessions plus eight mechanical-probe credits: four initial probes and four reserved for bounded recovery. Physical limits are 126,000 actor/input-counter calls and 600 first Terra attempts. Each paired shard requests 2 CPUs, 16 GiB, zero GPUs and at most 16 all-attempt allocation hours: 64 CPU-hours total. The shared collection calendar is seven days from the earliest actual allocation start. Waiting in the scheduler does not consume allocation hours. Keep every attempt's time and charges; recovery covers only genuinely unclaimed tasks without resetting limits.

Each execution approval must bind the printed shard's `plan_sha256`, exact `global_limits` and `local_limits`, with `approved` and `proxy_overshoot_authorized` true. Only then add `--execute --approval /path/to/approval.json --env-file /path/to/private.env` inside the approved CPU-only Slurm allocation. Plans and provider/task payloads stay local. Preparation alone starts no browser, model call or job and asserts no completed full-300 results.

</details>

<a id="gap-repaired-browsercode-scaling"></a>

### Recreate the gap-repaired BrowserCode comparison

The [current 953-task comparison](ARM_FINAL_RESULTS.md#browsercode-gap-repaired-20261104) preserves all original valid successes and failures and admits only the prespecified first new outcome for each of 703 previously invalid/missing cells. Thirty-three use saved actor evidence plus a first Terra judgment; 670 use one fresh actor attempt. The 386 valid repairs include successes and failures; 317 new invalids remain excluded, and no outcome is imputed. Intersect all eleven methods within each benchmark: **252 OM2W, 556 WebVoyager and 145 DeepShop**. Apply fixed planned weights 300/1,045, 595/1,045 and 150/1,045 to success and every actor mean. The aggregate also includes each method's own-valid counts and means; it does not mix those populations into the plotted intersection.

The renderer validates the 33 complete matched points, 42 joint-outcome contingency rows and their exact success marginals, and recomputes 50,000 paired bootstrap draws within benchmark. Pointwise percentile intervals condition on common validity and omit clustering, live-window variation and multiplicity correction. Equal decision limits do not equalize browser-resource use. Historical single/60 advertised 30; single/120 advertised 120. Fresh actor pricing/latency and saved-actor metrics retain their source definitions; valid-trajectory actor means exclude invalid-attempt spending, Terra and hosting fees. Cached reruns and new Sol/60 are not part of this reconstruction.

Run from the repository root using only public aggregates, with no API key, browser or model calls. Keep the sibling historical results directory: the renderer reuses its styles and paired-bootstrap implementation. PNG, PDF, SVG, weighted statistics and source/artifact hashes are regenerated together.

```bash
"$API_PY" -m pip install matplotlib==3.9.4 numpy==2.0.2
GAP_RESULTS=openwebrl/docs/arm_results/browsercode_gap_repaired_20261104
"$API_PY" -B "$GAP_RESULTS/plot_gap_repaired_scaling.py" \
  --aggregate "$GAP_RESULTS/aggregate.json" \
  --output-dir /tmp/openwebrl-browsercode-gap-repaired
```

The earlier 744-task figures and their original source files remain separate historical snapshots in the next subsection. Task identities, trajectories and operational audit payloads are not published.

### Recreate the completed scaling figures

The historical 744-task [compact strategy figure](ARM_FINAL_RESULTS.md#browsercode-scaling-families) groups output-cap, step-limit and effort curves in its single-run column, and candidate-count curves at 30/60 steps in its selection column. It represents all eleven historical methods: ten Luna settings as points and Sol high/30 as a numeric reference. Each column has its own explicit x-axis ranges; success limits match. The renderer asserts exhaustive historical-method coverage and exports a coverage manifest. The earlier shared-axis six-panel family figure is retained as an alternative. The [supplementary selection analysis](ARM_FINAL_RESULTS.md#browsercode-selection-scaling) fixes Luna high, the 12,288-token ceiling and 30 steps while varying N1/N2/N3/N5. All views use the published all-eleven-method common-valid intersection and fixed planned benchmark weights. The public four-outcome contingency counts preserve pairing without releasing task identities. The renderer checks every contingency's sample size and success marginals against the common-valid aggregate, then draws 50,000 multinomial paired bootstrap replicates within benchmark. Replicates resample the four joint outcome cells, not independent arm marginals. Reported intervals are pointwise percentile intervals without multiplicity correction; conditioning, website dependence, collection windows and the historical single60 prompt caveat remain. Candidate generation and selection vary together, so selector quality is not separately identified.

Recreate the compact figure, earlier two variants, coverage manifest, statistics and provenance using only public aggregates, with no browser/model calls:

```bash
"$API_PY" -m pip install matplotlib==3.9.4 numpy==2.0.2
SCALING_RESULTS=openwebrl/docs/arm_results/browsercode_inference_scaling_20261031
"$API_PY" -B "$SCALING_RESULTS/plot_browsercode_scaling_families.py" \
  --output-dir /tmp/openwebrl-browsercode-scaling-families
```

The [equal-weight benchmark-average view](ARM_FINAL_RESULTS.md#browsercode-equal-benchmark-average) gives each benchmark weight 1/3 for success and each within-benchmark actor mean. It avoids weighting WebVoyager more heavily just because it has more tasks. It retains each method's own valid cohort, cross-window limitations and the single60 advertised-30 versus single120 advertised-120 prompt caveat; no pooled rate or new uncertainty interval is inferred. [Average CSV](arm_results/browsercode_inference_scaling_20261031/browsercode_scaling_average_coordinates.csv) · [Average PDF](arm_results/browsercode_inference_scaling_20261031/browsercode_scaling_average.pdf).

The [task-count-weighted view](ARM_FINAL_RESULTS.md#browsercode-task-weighted-average) instead uses fixed planned benchmark weights 300/1,045 (OM2W), 595/1,045 (WebVoyager) and 150/1,045 (DeepShop), identically for success and within-benchmark actor means. This represents the planned task mix rather than equal benchmark influence. It is not pooled native-valid success, whose method-dependent weights would change with exclusions. [Weighted CSV](arm_results/browsercode_inference_scaling_20261031/browsercode_scaling_weighted_average_coordinates.csv) · [Weighted PDF](arm_results/browsercode_inference_scaling_20261031/browsercode_scaling_weighted_average.pdf).

The [all-method common-valid alternative](ARM_FINAL_RESULTS.md#browsercode-all-method-common-valid) intersects the native Terra-valid task identities across all eleven plotted methods separately within each benchmark. It recomputes success and every actor metric on that identical intersection, including valid failures, and retains the original planned benchmark weights. Invalid and missing outcomes exclude an identity from all methods in its benchmark; missing measurements are never imputed. Matching tasks does not control collection dates or the earlier prompt-limit difference. Only restricted aggregate counts and means are published; task identities and trajectories remain private. The original own-valid plots remain available.

The [historical own-valid comparison](ARM_FINAL_RESULTS.md#browsercode-inference-scaling-20261031) includes the three token/effort settings above, single60, single120, N2/30, N5/60 and the four historical high-effort controls. Recreate its 7 × 4.35-inch three-column main figure and 7 × 4.35-inch token figure, with PNG/PDF/SVG exports and coordinates from public aggregates with zero model or browser calls:

```bash
"$API_PY" -m pip install matplotlib==3.9.4
SCALING_RESULTS=openwebrl/docs/arm_results/browsercode_inference_scaling_20261031
"$API_PY" -B "$SCALING_RESULTS/plot_browsercode_scaling.py" \
  --aggregate "$SCALING_RESULTS/aggregate.json" \
  --output-dir /tmp/openwebrl-browsercode-scaling-plots
"$API_PY" -B "$SCALING_RESULTS/plot_browsercode_scaling.py" \
  --aggregate "$SCALING_RESULTS/aggregate.json" \
  --output-dir /tmp/openwebrl-browsercode-scaling-plots \
  --stem browsercode_scaling_tokens --tokens \
  --token-method luna_single_high --token-method luna_single_xhigh_12288 \
  --token-method luna_single_high_24576 --token-method luna_single_high_6144 \
  --token-method luna_single_high_60steps --token-method luna_single_high_120steps
"$API_PY" -B "$SCALING_RESULTS/plot_browsercode_average.py" \
  --aggregate "$SCALING_RESULTS/aggregate.json" \
  --output-dir /tmp/openwebrl-browsercode-scaling-plots
"$API_PY" -B "$SCALING_RESULTS/plot_browsercode_average.py" \
  --weighting task-count --aggregate "$SCALING_RESULTS/aggregate.json" \
  --output-dir /tmp/openwebrl-browsercode-scaling-plots
"$API_PY" -B "$SCALING_RESULTS/plot_browsercode_common_valid.py" \
  --aggregate "$SCALING_RESULTS/aggregate_common_valid.json" \
  --output-dir /tmp/openwebrl-browsercode-scaling-plots
```

The first figure uses each method's valid-task success, actor cost, selected browser-code batches and latency. Latency is elapsed actor-loop time per task, averaged over valid tasks including failures. Projected hosting is omitted from the paper figures; measured browser usage is reported separately in the [US-residential pilot](#measured-us-residential-pilot). A batch may contain multiple browser/CDP calls. The second uses actual summed output and reasoning tokens for the six Luna single-call settings; a per-call ceiling is not realized token consumption. Counts, exclusions, common-valid paired intervals and collection-window limitations accompany the figures.

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

Reproduce the published [three-benchmark Terra paper figure](ARM_FINAL_RESULTS.md#shared-terra-high-three-benchmarks-20261031) from its public aggregate without API calls (Matplotlib 3.9.4):

```bash
"$API_PY" -m pip install matplotlib==3.9.4
TERRA_RESULTS=openwebrl/docs/arm_results/cuawright_browsercode_om2w_20261027
"$API_PY" -B "$TERRA_RESULTS/plot_terra_benchmark_tradeoffs.py" \
  --aggregate "$TERRA_RESULTS/shared_terra_high_four_row_aggregate.json" \
  --output-dir /tmp/openwebrl-terra-plots
```

This exports a fixed 7 × 5.5-inch PNG/PDF/SVG and the exact 14-point CSV. It requires no private trajectories or original campaign directories. The shared Terra judge replaces benchmark-native judging for this view; preserve the original labels separately when rejudging your own saved endpoints.

For the compact OM2W-only view, run `plot_om2w_terra_tradeoffs.py --valid-only --high-only`; add `--harness browsercode` to focus on BrowserCode. Omit `--high-only` to include supporting lower efforts, or omit `--valid-only` for full-attempt actor metrics. Those source aggregates stay unchanged; recovered valid lower-effort tasks retain original plus retry work, and missing measurements remain explicit.

The [matching Claude figures](ARM_FINAL_RESULTS.md#claude-browsercode-om2w300) use the published qualified valid-task and common-valid aggregates. Render both into a new output directory:

```bash
CLAUDE_RESULTS=openwebrl/docs/arm_results/claude_browsercode_om2w300_20261028
"$API_PY" -B "$CLAUDE_RESULTS/plot_claude_tradeoffs.py" \
  --aggregate "$CLAUDE_RESULTS/aggregate.json" \
  --qualification "$CLAUDE_RESULTS/render-qualification.json" \
  --output-dir /tmp/openwebrl-claude-plots
```

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
| Browser-code batches | One count per saved decision with a nonempty selected `command`; see the [exact counting rule](#browsercode-batch-counting). One batch can contain several browser operations. |
| BrowserCode task latency | Actor-loop wall time, including model admission/HTTP waits and browser execution. Exclude setup, final judge-only capture, cleanup and external judging. |
| Tokens | Native input/output/cache usage. Output may already include reasoning tokens; do not count those twice. |

<a id="browsercode-batch-counting"></a>
**Browser-code batch counting.** The [GPT worker](../api_experiments/browsercode/workers/gpt_high_worker.py) and [Luna worker](../api_experiments/browsercode/workers/luna_effort_worker.py) save `selected_code_batches = sum(bool(d['selected_action']['command']) for d in decisions)`. Each selected nonempty command contributes one, regardless of whether its execution succeeds. The [model adapter](../api_experiments/browsercode/models/luna_model.py) encodes one selected `browser_execute` request as that command: its JavaScript `code` may contain several clicks, navigations, evaluations or other browser operations. The [environment](../api_experiments/browsercode/runtime/gpt_high_environments.py) dispatches that request once to the browser bridge. Individual JavaScript statements and underlying CDP calls are not counted separately; `native_browsercode_cdp_attempts` records a separate lower-level counter.

A model decision selects at most one batch or returns a final answer. A final-answer-only decision has an empty command and contributes zero batches. N3/N5 candidate generation and selection add model calls, but unselected proposals and the selector itself add no browser batches. Thus batches are neither model-call counts nor necessarily total decision counts. The completed 24-point BrowserCode figure averages `selected_code_batches` over each method's valid tasks, including valid failures, with full measurement coverage; it does not average over all planned tasks or a shared valid intersection.

The [shared-Terra four-row aggregate](arm_results/cuawright_browsercode_om2w_20261027/shared_terra_high_four_row_aggregate.json), schema v3, keeps the 12 BrowserCode points unchanged and updates four native OM2W points after bounded recovery. Native qualification is 277 / 279 / 254 / 250 of 300 for Sol / Luna single / N3 / N5. The aggregate distinguishes recorded unqualified tasks from `recovery_unclaimed` (0 / 0 / 28 / 29), and preserves `source_stop_classification`: complete roster, planned admission cutoff, or operational partial. Every eligible endpoint has its first Terra verdict; the 379-cell recovery is partial.

Native valid-task efficiency means sum original plus recovery work for recovered logical tasks. Model cost includes all physical candidates, selectors and helpers, excludes external judging, and retains unknown reservations. Original native time spans parent-worker creation through owned cleanup; recovery time spans the native episode runner through cleanup, less only the measured final dedicated observer RPC. Preclose capture overhead stays included. Browser records use literal-LF native browser-step entries. These cumulative times and record units differ from BrowserCode actor-loop time and selected-code batches. The [results table and accounting](ARM_FINAL_RESULTS.md#shared-terra-high-three-benchmarks-20261031) place coverage, unclaimed counts and all-attempt recovery spending beside the comparison. This update adds fresh attempts and evidence; it does not measure a fixed-cohort improvement or a causal harness effect.

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

For the shared-Terra protocol, add `--judge terra` to the same command. This keeps the actor settings and replaces WebJudge with GPT-5.6 Terra at medium effort, 4,096 completion tokens and seed 42. `--judge-usd` and `--judge-calls` fund one shared Terra pool across the invocation; external WebJudge credits are zero. Each qualified episode gets at most one physical judge attempt, with SDK retries disabled and unknown charges reserved. A trusted browser observer captures the final frame, including immediately before an explicit browser close, and verifies its session, source, selected paid actions and owned teardown. Missing or unqualified evidence stays ungraded; saved WebJudge labels are never substituted. A zero-model-call browser-close probe must pass before actors start. [Observer sources](../api_experiments/reproduce/portable_native/terminal_observer/) and [portable Terra adapter](../api_experiments/reproduce/native_terra.py) ship in the repository. Native latency retains preclose capture overhead and subtracts only the measured final observer RPC; it differs from BrowserCode actor-loop latency. The copied Terra runtime also repairs the upstream explicit `--no-sandbox` flag so it matches the enabled default; the original checkout stays pinned, and the repair is hash-recorded before execution.

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
