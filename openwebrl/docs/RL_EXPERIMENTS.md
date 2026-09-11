# CPU preparation for RL experiments 0, 4–6

No GPUs, model downloads, browser launches, or API calls are needed for the tests,
exporter, or curriculum dry run. Policy/value training and live recovery evaluation
remain future GPU/browser work. These additions do not establish benchmark gains.

## 0. Corrected baseline

- Invalid trajectories are excluded before dynamic filtering and advantage statistics,
  and do not update adaptive success/failure counts. A partially invalid trajectory is
  excluded as a whole. Agent format errors and horizon exhaustion remain valid failures.
- Aborted browser episodes have unavailable outcomes; their loss is masked.
- Judge output uses a JSON `verdict` or a standalone terminal SUCCESS/NOT SUCCESS line.
  Rationale substrings cannot determine the verdict. Malformed output retries; exhausted
  API errors and timeouts are invalid labels rather than task failures.
- The sampler's default blacklist points to the supplied `openwebrl/data` file.

Print the 90-iteration, 15-step stage followed by 50 more iterations at 30 steps:

```bash
python3.12 scripts/run_browser_curriculum.py --output outputs/curriculum
```

Only add `--execute` on a configured GPU host. Stage 2 resumes stage 1 and uses
`NUM_ROLLOUT=140`, because the training loop restores its rollout index. The wrapper
requires fresh stage directories and writes a manifest. Its launcher retains all other
current defaults; this is an explicit horizon schedule, not a claim of exact paper
reproduction. The underlying launcher performs its existing process cleanup.

## 4. Recovery curriculum preparation

Copy `browser_training_config.yaml` and add:

```yaml
rollout_all_samples_process_path: openwebrl.recipe_data.export_rollouts
browser_recipe_export_dir: /absolute/path/to/recipe_data
```

Launch with `BROWSER_TRAIN_CONFIG=/path/to/copied.yaml`. Each finished rollout batch
exports all finished groups, including dynamically rejected all-failure groups:

- `recovery_candidates.jsonl`: infrastructure review, protocol recovery, horizon
  extension candidates, or teacher-review candidates.
- `value_targets.jsonl`: pre-action prompts/screenshots with binary terminal returns.
- `memory_targets.jsonl`: pre-action states paired with prior observed tool feedback.
- `manifest.json`: schema and record counts. Batches have unique directories.

Horizon exhaustion is only a candidate for extension, not evidence of progress.
Teacher candidates include the final pre-action context and failed response. They
are review records, not executable browser snapshots or automatically accepted SFT
examples. Reconstruct and verify live state before collecting a teacher continuation;
judge the recovered full task before adding it to recovery SFT. No teacher is invoked.
Pending groups canceled at the rollout barrier are not included in the existing
all-finished-samples hook. The exporter does not label them as failures.

Already serialized, post-reward groups can be processed offline:

```bash
python3.12 -m openwebrl.recipe_data --input groups.json --output outputs/recipe_data
```

Input topology is `[group[trajectory[Sample-dict]]]`; dictionaries use `prompt`,
`response`, `reward`, `remove_sample`, `index`, `group_index`, `metadata`, and
`multimodal_inputs`. Supply `--reward-key` for dictionary rewards. Existing unjudged
rollout debug files are not interchangeable with this format. No real rollout data is
bundled with these changes. These exports can be large because prefix images repeat.

## 5. State-dependent advantage hook

In the copied training YAML, set:

```yaml
custom_reward_post_process_path: openwebrl.rl_recipe.state_advantages
browser_state_advantage_mix: 0.25
browser_value_provider_path: your_package.values.predict
```

`predict(args, states)` returns one finite success probability in `[0,1]` per state.
Each state contains only `prompt`, `images`, and `turn_index`; the hook does not supply
current responses, terminal metadata, future judge screenshots, or labels. The provider
must use a frozen value model trained on earlier policy rollouts or task-disjoint folds.
Split by task **before** expanding trajectories into turns to prevent prefix leakage.
No value model is supplied or trained here.

For valid binary outcomes the hook computes
`(1-mix) * normalized_GRPO + mix * (terminal_reward - frozen_value)`.
Negative protocol penalties retain their original GRPO advantage. Invalid trajectories
have zero advantage. `mix=0` is the existing trajectory-normalized GRPO behavior.
This hook keeps the actor-only multi-epoch path and does not enable the incompatible
PPO critic. It is a Monte Carlo baseline ablation, not GAE or a calibrated critic.
The existing dynamic filter still removes uniform-reward groups: this is not an
unbiased full-distribution policy-gradient estimator, and the two advantage components
have different scales. Keep mix fixed and log/update-budget-match comparisons.

## 6. Conservative explicit memory baseline

Set `browser_observation_memory: true` in the copied YAML. The turn-level generator
adds a bounded JSON observation ledger to the current user context, containing only
prior step indices, tool names, and actual tool feedback. It retains full reasoning
history and the existing screenshot window; rollout and optimization use the same
augmented prompt. Entries are bounded by `browser_memory_max_entries` (default 8)
and `browser_memory_max_chars` (default 4000), dropping oldest entries first.

This is a deterministic observed-history baseline, not a learned memory compressor.
It does not infer verified facts, mark task constraints satisfied, or trust agent
claims as observations. `memory_targets.jsonl` prepares an auxiliary distillation
experiment; no learned memory policy is supplied. Train and evaluate that separately
before considering removal of historical reasoning. Pass `--browser-observation-memory` to `python -m openwebrl.run_evaluate`
(with matching limits) when evaluating this ablation.

## Verification

```bash
python3.12 -m unittest discover -s tests -v
```

Tests exercise the actual reward-normalization method using an arithmetic adapter
because PyTorch/Ray/Megatron are unavailable here. They also check judge verdicts,
invalid-group filtering, prefix-only value inputs, causal/bounded memory, exports,
and curriculum planning. Full runtime integration still requires the training stack.

### Lightweight preflight before allocating GPUs

```bash
python3.12 scripts/check_cpu_preflight.py
```

This runs the offline tests, parses repository Python/JSON/JSONL/TOML files, checks
shell syntax, and parses YAML with an available PyYAML interpreter. It writes
`outputs/preflight/cpu_report.json`, including dependency availability and untested
runtime components. It does not download packages/models or start services.

The expanded tests execute the real turn-loop control flow with in-memory model and
browser substitutes, check rollout/training prompt identity and causal memory, test
adaptive sampler accounting, round-trip the export CLI, and simulate curriculum
checkpoint handoff. They found and cover initial-context overflow returning an empty
trajectory, invalid sibling turns with a zero-mix advantage hook, missing-reward
accounting, and incomplete stage-one checkpoint acceptance. Curriculum stage two now
requires checkpoint iteration 89 and its directory after the 90-iteration first stage.

These tests cannot establish GPU memory fit, actual chat-template/processor compatibility,
NCCL behavior, live site availability, judge connectivity, or checkpoint tensor integrity.
The report distinguishes the lightweight checks from those runtime prerequisites.
