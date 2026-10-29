# API evaluation source snapshots

The published Python files below are byte-for-byte copies of the code used by the API browser evaluation campaigns. They include the actor/selector adapters, episode workers, external judges, cost accounting and browser ownership code. [Methods](../docs/API_EXPERIMENT_METHODS.md) defines the protocols and metrics; [results](../docs/ARM_FINAL_RESULTS.md) contains the audited comparisons.

| Study | API / selection code | Execution and accounting |
|---|---|---|
| GPT high, BrowserCode | [paired_model.py](snapshots/browsercode_gpt_high/paired_model.py) | [worker.py](snapshots/browsercode_gpt_high/worker.py), [episode_metrics.py](snapshots/browsercode_gpt_high/episode_metrics.py) |
| Claude high, BrowserCode | [claude_model.py](snapshots/browsercode_claude/claude_model.py) | [worker.py](snapshots/browsercode_claude/worker.py), [claude_budget.py](snapshots/browsercode_claude/claude_budget.py), [common_judge.py](snapshots/browsercode_claude/common_judge.py) |
| Luna effort, BrowserCode | [luna_model.py](snapshots/browsercode_luna_effort/luna_model.py) | [worker.py](snapshots/browsercode_luna_effort/worker.py), [request_budget.py](snapshots/browsercode_luna_effort/request_budget.py), [common_judge.py](snapshots/browsercode_luna_effort/common_judge.py) |
| GPT high, native CUAWright | [native_proposal_adapter.py](snapshots/cuawright_native/native_proposal_adapter.py), [native_group_transport.py](snapshots/cuawright_native/native_group_transport.py) | [native_episode.py](snapshots/cuawright_native/native_episode.py), [budget_gateway.py](snapshots/cuawright_native/budget_gateway.py), [external_judge.py](snapshots/cuawright_native/external_judge.py) |

## Using these snapshots

These are versioned experiment sources, not a self-contained launch package. The campaign-specific files deliberately retain their original import layout and environment-specific paths. Importing a worker can require local configuration. Review the code before executing it.

For the recent BrowserCode campaigns, `bootstrap.py` reads a local `runtime-config.json`, which identifies the pinned CUAWright source and frozen OpenWebRL support source. Workers and controllers additionally require the original campaign plan, task roster, source manifest and effective approvals. The native snapshots have their own gateway/worker configuration and manifest layout. Follow the methods document's runtime index to those local artifacts; the snapshots alone do not authorize or recreate a paid launch.

`source-provenance.json` records the campaign identifiers and SHA256 of every published source file. Where available it also records the original execution-manifest hash. Source equality does not imply that every campaign shared a protocol: the historical GPT BrowserCode, native CUAWright and later Terra-judged studies remain distinct.

Dependencies include the pinned CUAWright/OpenWebRL support sources, provider SDKs, local Chromium/CDP and the BrowserCode bridge where applicable. The precise environment and scientific configuration are defined by each campaign's frozen runtime records, not by installing whatever versions are current today.

The Claude worker also references private task-bound terminal-interruption qualification helpers; those helpers remain local and are excluded because they embed task identities. This is another reason the snapshots alone are not a standalone launch package.

Credentials, `.env`, task identities/payloads, screenshots, trajectories, request/response logs, ledgers, approval records and run-specific scheduler metadata are excluded. Tests published beside the sources are historical validation sources; some require the original local runtime and are not advertised as a standalone test suite.
