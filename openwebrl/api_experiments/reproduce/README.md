# Reproduce the BrowserCode API evaluations

Use [`eval_luna.py`](eval_luna.py) for GPT-6 Luna and [`eval_claude.py`](eval_claude.py) for Claude. Both call the original frozen `worker.episode`: same prompts, native tools, candidate/selector policy, 12,288 output tokens, 30 decisions, screenshots and GPT-5.6 Terra judge. The default is a **read-only dry run of one task**. Add `--execute` to spend API budget and start browsers.

These are entry points into an **existing configured campaign runtime**, not a self-contained install. The public [source archive](../README.md) preserves the implementation; private task manifests and machine-specific browser dependencies are not published. See [methods and results](../../docs/API_EXPERIMENT_METHODS.md) for the study protocol and result links.

## Two commands

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

## What execution saves

Each invocation requires a **new output directory** outside the source runtime. It owns separate model/judge ledgers, browser counter, HTTP gate, SDK caches, W&B identity, episode evidence and `summary.json`. It cannot resume or overwrite an earlier invocation. Provider retries are disabled; a shared halt stops the run for diagnosis. Unknown usage retains its full reservation, and invalid episodes remain in cost accounting.

`summary.json` reports recorded/valid/invalid counts, valid-only success, valid-task mean latency and selected browser batches with observation counts, and separate all-attempt model/judge charges or reservations. Judge time is excluded from actor task latency. A browser batch is one selected executable snippet; it can contain several browser operations. Missing measurements are not zero. Raw outputs include private task content and should stay local.

W&B defaults to offline under the output directory in `openwebrl-evals`. Use `--wandb-mode online` to sync or `--wandb-mode disabled` to omit tracking. `--model-calls`, `--judge-calls` and `--browser-starts` provide additional stop caps; browser starts default to N tasks plus one mechanical preflight. Model calls include candidates, selectors and token counters. Model/judge pools never transfer funds.

## Scope of reproduction

The scripts preserve the **episode protocol**. They run one episode at a time, retaining parallel candidates within each decision, and apply one pair of run-level USD/call caps. The original studies used separately capped shards and concurrent scheduler jobs, so these entry points do not recreate the original scheduling, contention, shard admission history or measured latency environment. They submit no Slurm jobs and reserve no compute. Full campaign replication requires the original [campaign controllers and frozen approvals](../README.md), including canary/pilot audits and all-attempt resource accounting.

Startup checks verify the frozen plan, source hashes, task manifest and local browser dependencies before execution. They do not prove live provider access or site availability. Offline validation:

```bash
python3 -B -m unittest discover -s openwebrl/api_experiments/tests -v
```
