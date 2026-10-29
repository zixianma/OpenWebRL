# API evaluation code

For the shortest runnable entry points, see [reproduce/README.md](reproduce/README.md): one script for Luna and one for Claude, with read-only dry runs and explicit execution caps.

Code is grouped by harness and function. Identical files are shared once; the [provenance index](source-provenance.json) maps every original campaign filename to its published source and SHA256. Experiment source bytes remain unchanged.

[Methods](../docs/API_EXPERIMENT_METHODS.md) · [Results](../docs/ARM_FINAL_RESULTS.md)

## Start here

| Component | Code |
|---|---|
| GPT Luna effort: API calls and N5 selection | [luna_model.py](browsercode/models/luna_model.py) |
| Claude: API calls and N3/N5 selection | [claude_model.py](browsercode/models/claude_model.py) |
| Earlier GPT high: API calls and selection | [paired_model.py](browsercode/models/paired_model.py) |
| BrowserCode episodes | [workers](browsercode/workers/) |
| API budgets and concurrency | [budgets](browsercode/budgets/) |
| Terra and historical judges | [judges](browsercode/judges/) |
| Local browser, sandbox and metrics | [runtime](browsercode/runtime/) |
| Native CUAWright proposals and transport | [models](cuawright/models/) |
| Native CUAWright execution | [workers](cuawright/workers/), [runtime](cuawright/runtime/) |

## Layout

```text
browsercode/
  models/        # Actor proposals and index-only selection
  workers/       # Campaign-specific episode loops
  budgets/       # Reservations and shared HTTP admission
  judges/        # Terra and historical terminal graders
  runtime/       # Browser ownership, confinement, metrics, bootstrap
  controllers/   # Batch worker ownership
  protocols/     # Financial enrollment and canary scope
  recovery/      # Generic saved-judge recovery
cuawright/
  models/        # Native proposal adapter, group transport and relay
  workers/       # Native campaign worker
  budgets/       # Metered gateway
  judges/        # External judge
  runtime/       # Browser broker, capture, confinement and timeouts
validation/      # Archived campaign test sources
materialize.py   # Restore original file layout without launching anything
tests/           # Offline tests for materialize.py
```

## Restore an original campaign layout

The grouped files are experiment source archives, not an installable evaluation package. Original imports and environment-specific paths are preserved. Use the materializer when you need the original sibling filenames:

```bash
python3 openwebrl/api_experiments/materialize.py browsercode_luna_effort \
  --output /tmp/luna-effort-source
```

Other campaign keys are `browsercode_claude`, `browsercode_gpt_high` and `cuawright_native`. The output directory must be new. All source hashes are checked before any output files are written. The command copies code only; it starts no workers or API requests.

To execute a campaign, its original local runtime must also supply frozen configuration, task roster, execution manifest, effective approvals, pinned CUAWright/OpenWebRL support sources, provider SDKs and local browser/bridge dependencies. The native campaign has its own gateway configuration. Copying these sources does not recreate those inputs or authorize a paid launch.

The private Claude terminal-interruption qualifiers are excluded because they embed task identities; the original worker references them for historical recovery. Credentials, `.env`, task data, screenshots, trajectories, request logs, ledgers and scheduler records are also excluded. Archived tests under `validation/` may require that original runtime; only the materializer tests are advertised as self-contained:

```bash
python3 -B -m unittest discover -s openwebrl/api_experiments/tests
```
