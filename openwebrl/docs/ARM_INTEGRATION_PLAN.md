# Action reward models for OpenWebRL training

Status: **All three inference evaluations are complete. Overall success: baseline 30.0%, ScalarRM 38.0%, SelectionARM 42.7%; valid-only: 33.7%, 45.4%, 50.0%. C2 paused cleanly when allocation 282782 ended: 220/2091 tasks processed, 123 successful trajectories, and 711 eligible turns saved. Student SFT has not started; see ARM_C2_RUN.md for details. Inference retries remain held.**

Created: 2026-09-07. This is the working document for iterating on the plan originally proposed in conversation.

[Inference results, denominators, and retry plan](ARM_INFERENCE_RESULTS.md) · [Action-level filtered SFT pilot](ARM_FILTERED_SFT_PLAN.md).

[Matched retry pass](ARM_INFERENCE_RETRY_RESULTS.md): **held** pending a cohort decision using the three completed runs; C2 currently has execution priority. The earlier 56-task two-arm inventory is provisional, and automatic launch is disabled. **C2 is the selected first SFT experiment**; C1 is an optional later control and C3 is deferred. The user superseded the 128/32/32 draft with **all 2091 deduplicated tasks** and requested C2 after eval. The [SFT plan](ARM_FILTERED_SFT_PLAN.md) records the revised configuration; [C2 run status](ARM_C2_RUN.md) records execution and artifacts.

## 1. Current priority: reproduce Online-Mind2Web inference gains

The user selected inference validation as the first milestone. Use the frozen actor `OpenWebRL/OpenWebRL-4B-SFT` with these two exact reward model releases:

| Arm | Candidate actions per turn | Selection | README success target |
| --- | --- | --- | --- |
| Baseline | 1 | Execute actor sample | 33.8% |
| Scalar | 5 | `PTeterwak/OpenWebRL-4B-ScalarRM-LoRA` argmax | 46.3% |
| Selection | 5 | `PTeterwak/OpenWebRL-4B-SelectionARM` | 51.1% |

Use temperature 0.7 for the initial matched comparison, normalized coordinates, the same frozen actor, 300 fixed local task IDs, and the Online-Mind2Web AgentTrek/o4-mini evaluation protocol. Any unknown original setting must be marked as a reconstruction rather than claimed to be exact. A fresh-browser run is required for each arm; candidate sets can be shared for offline same-state audits, but complete interactive trajectories diverge after selection.

Pinned releases resolved on 2026-09-07:

- Actor: `15e777db2ddba2e0e82080ebccd3ad8d215b7f0a` (matches the existing downloaded actor manifest).
- Selection: `81b452d800d9f859687074f82680dd5257e02d89`.
- Scalar adapter/head: `71c58489cd7cbaebcd74656df7ef11ba818661b8`.
- ARM source: `02276b0ff3b9048d34e6a2afdcb9042dd9018c8c`.

### Reproduction sequence

1. Save pinned source/model provenance, prepare model-specific serving in an isolated environment, and verify score/prompt/coordinate contracts.
2. Run offline scoring and a small live smoke across all three arms.
3. Run the full fixed 300-task comparison with resumed per-task output, serialized candidate/selection traces, and a shared judge configuration.
4. Report success over all scheduled tasks, valid-run success, task coverage, infrastructure/judge failures, and paired differences on common evaluable tasks. Do not silently discard different failed tasks in each arm. Repeated runs and paired uncertainty estimates should qualify the observed gains.
5. Return to the training plan only after assessing inference behavior and resource cost.

### Newly discovered limitations

- The published dashboard does not give exactly the README denominators: its August data include n=1 95/276 and trained ARM 140/282; September reports another aborted-task exclusion. Exact README run manifests and the scalar per-task labels are not present in the inspected release. Record the README values as reported targets, not an established matching protocol.
- Some exported dashboard ARM prompts ask for an explanation and contain system-prompt text in the action-history field, whereas the released canonical selection builder requires the no-CoT variant. Use the released builder as the reproducible default and retain this discrepancy in the report; do not invent equivalence to the historical runs.
- The user assigned a separate one-H200 allocation (`282209`) to ARM evaluation after the initial training-handoff correction below. GPU occupancy alone is no longer accepted as an allocation handoff.

### Prepared implementation and current validation

The inference path is opt-in and separate from policy training:

- [Pinned model downloader](../../scripts/prepare_arm_models.py).
- [Candidate sampling and selection](../arm_inference.py).
- [Frozen ARM server](../../scripts/serve_arm.py).
- [Resumable benchmark runner](../arm_eval.py).
- [Allocation-aware launcher](../../scripts/run_arm_reproduction.sh) and [automatic watcher](../../scripts/watch_arm_reproduction.py).
- [Paired task report](../../scripts/summarize_arm_reproduction.py).

Models are downloaded under `/gpfs/scrubbed/zixianma/checkpoints/web/arm`; reference code, manifests, and an isolated serving environment are under `/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction`. The actor remains the existing pinned SFT download.

CPU validation completed: nine ARM contract/structured-decoding/execution tests, six watcher allocation/safety/promotion tests, six existing browser-turn tests, o4-mini model access, matching actor/selection vocabularies and templates, and matching shapes for all 713 released selection tensors. The omitted `lm_head.weight` is tied to the embedding. The frozen actor has now loaded on GPU and passed its generation health check. All three baseline and scalar smoke tasks completed with valid judge results. Scalar scores are finite and discriminate candidates. Selection model loading and live inference also passed. The constrained selection smoke completed with 3/3 valid judged tasks and no fallback across 34 actions. Aggregate benchmark performance remains **unvalidated**.

Compatibility note: the release stores newer Transformers processor/config metadata. Installed Transformers 4.57.1 reads the selection rotary settings as `None` and cannot directly load its processor config. The server uses the original actor config/processor after vocabulary, template, and architecture checks, preserving the frozen ARM weights. Record this compatibility adaptation and the 262144 image-pixel cap in the result manifest. The scalar head uses the reference demo's BF16 projection and last-token pooling.

Once two assigned GPUs are free, first run a small paired smoke from inside the active Slurm allocation:

```bash
# On the allocated node; the launcher refuses occupied GPUs.
ARM_JOB_ID=<active-job-id> \
srun --jobid=<active-job-id> --overlap --gres=gpu:2 \
  env TASK_INDICES=0,50,100 N_PARALLEL=1 \
  bash scripts/run_arm_reproduction.sh
```

For a manual full run, omit `TASK_INDICES` for all 300 tasks. Set an explicit `OUTPUT_ROOT` to resume the same run. Every output directory has a configuration manifest; the runner rejects mismatched resumes. All tasks remain in the scheduled denominator; valid-only and common-valid paired statistics are secondary reports. Runs occur baseline → scalar → selection, so date/time ordering is a residual confound to address with repeated or interleaved runs if the first comparison suggests a gain.

### Dedicated evaluation allocation provided at 17:47 PDT

The user explicitly assigned the new GPU on `g001` to ARM evaluation. Verified allocation **282209** has **one H200, 8 CPUs, and 200000 MiB host memory**, ending **2026-09-07 21:47:39 America/Los_Angeles**. Its physical GRES index is **7**, GPU UUID **`GPU-74ad27d6-2c55-e7b8-c160-604cbb6f44ff`**; inside its Slurm step it is CUDA device **0**. This UUID differs from both training GPUs in allocation 281697.

- The frozen actor and one frozen ARM share the assigned H200. Actor static memory fraction is **0.4**; scalar and selection servers run sequentially. Model weights, sampling, candidate counts, and evaluation protocol remain as specified above.
- The controller belongs to `job_282209/step_extern`, and the evaluation worker belongs to Slurm step `282209.11`. Neither lifetime depends on allocation 281697. Inherited Slurm/CUDA settings from other allocations are cleared before launching a step.
- Smoke uses indices `0,50,100`, concurrency 3. All three arms passed with 3/3 valid judged tasks: baseline 27 action traces, scalar 46, selection 34; no selection fallbacks under constrained decoding. The full 300-task-per-arm comparison uses **concurrency 8** to improve GPU utilization within the remaining allocation. Task seeds, sampling, models, and horizons are unchanged.
- Current run: `/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/dedicated-282209-20260908T005547Z`.
- [Live status](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/dedicated-282209-20260908T005547Z/watcher-status.json); model/browser logs and results are under `smoke/`, then `full/`.
- One initial attempt was stopped to relocate the supervisor out of the older allocation; a clean-shell missing `rg` dependency was fixed before this launch. These earlier attempts are retained separately and will not be merged into the new result denominator.
- New CLI requires `--dedicated-allocation` and supports `--gpu-count 1`; it refuses a controller tied to a different allocation. No additional allocation or extension was requested.

### Automatic ARM handoff and utilization at 19:51 PDT

Baseline completed all **300** tasks: **90 successes**, **267 valid outcomes**, **33 unavailable**. Success is **30.0% of all scheduled tasks** and **33.7% among valid outcomes**. These denominators remain distinct from the author-reported reference rate.

The scalar ARM started automatically at **19:49:47 PDT**, **20.7 seconds** after baseline completion. The actor server remained resident. Scalar and selection model files were warmed in cache beforehand. The active queue is **scalar → selection**, without a new resource request or a manual handoff.

A 30-second baseline probe measured **52.6% mean GPU activity**, an empty inference queue, about **3.8 decoding requests**, and **3.2/8 CPU cores** in use. Upcoming ARM concurrency was increased from **8 to 16**, with an explicit settings file and per-arm `execution-history.jsonl`. Task/model/sampling/horizon settings are unchanged. The runner bounds these local concurrency overrides at 16.

The first 30-second scalar ARM probe at concurrency 16 measured **84.6% mean GPU activity (peak 100%)**, **71.5 GiB GPU memory**, **4.9/8 CPU cores**, approximately **1230 decode tokens/sec**, and **23.9 active decoding requests** against a server limit of 24. The maximum observed inference queue was 36. This shows substantially better occupancy; it does not establish optimal throughput or isolate the concurrency effect, because the ARM workload generates five candidates rather than one. Scalar execution at concurrency 16 is verified in its execution record.

[Handoff and utilization evidence](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/dedicated-282209-20260908T005547Z/arm-handoff-and-utilization.json). [Pending-stage concurrency settings](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/dedicated-282209-20260908T005547Z/full/execution-settings.json).

### Resume on g005 at 22:46 PDT

The original allocation ended with **277/300 scalar tasks saved**: **112 successes**, **232 valid evaluations**, and **45 unavailable**. The user explicitly assigned the new four-hour GPU session on `zixianma@g005` to continuing these evaluations.

Verified allocation **282782** has **one H200, 8 CPUs, and 200000 MiB host memory**, ending **2026-09-08 02:37:55 America/Los_Angeles**. The assigned GPU is **`GPU-90ac2a02-abfa-147c-19ff-bbada4033da3`**, physical GRES index **4**, CUDA device **0** inside the step. The other user allocation on g005 is excluded. The new supervisor belongs to `job_282782/step_extern`; the worker runs in **282782.0**.

- Reuse the same run directory and all completed task outcomes, including unavailable outcomes. Baseline does not execute any tasks again. Resume only the **23** unfinished scalar tasks, then run all **300** SelectionARM tasks.
- Preserve the pinned models, seeds, prompts, image processing, horizons, judge, and strict configuration-match resume check. Concurrency remains **16** for each ARM, with the actor kept resident across their handoff.
- Archive prior server/evaluation logs under `allocation-history/282209/`. Archive incomplete scalar action traces before restarting those tasks so separate attempts are not mixed in their final traces. Saved task-result hashes are retained for verification.
- Revalidate the existing successful smoke gate; the new actor and ARM processes must pass service health checks on g005. The same shared runtime is used.
- Use only the existing user-provided allocation; stop at its expiry. No new allocation or budget extension was requested.

[Resume inventory and result hashes](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/dedicated-282209-20260908T005547Z/resume-282782.json). [Current controller](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/active-watcher.json). [Progress snapshot](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/dedicated-282209-20260908T005547Z/progress-snapshot.json).

The run is incomplete. Website navigation failures, actor context-limit errors, and other environment/inference failures remain in the all-scheduled denominator. Resuming on a later node/session adds a time-of-evaluation confound; matched-task statistics do not remove website changes or judge variability.

### Selection decoding correction during the dedicated smoke

The initial selection smoke produced some bare-index replies such as `</think>\n\n3`. The minimal released `selection_infer.py` demo uses JSON-only parsing and falls back to candidate 1, matching our initial implementation. The fuller canonical `selection_prompt.py` inference function supplies a **strict JSON schema** under `VISION_NO_COT=1`; its response parser also has more permissive recovery. Omitting that schema was a reproduction discrepancy.

The server now uses installed **XGrammar with Hugging Face generation** to enforce the canonical schema (required integer `selection`, bounds 1 through candidate count). This preserves the strict parser and the original zero-fallback smoke gate. The grammar rejects bare numbers, indices 0/6 for five candidates, and extra fields in CPU tests. Generation remains greedy; the original chat-template thinking default is preserved. The serving implementation differs from the reference OpenAI-compatible server, so this is semantic schema parity, not a claim of identical inference-engine numerics.

The unconstrained smoke is archived under `smoke/selection-unconstrained/`, with its logs and comparison preserved separately. Baseline and scalar smoke records were reused unchanged; selection was rerun from fresh browsers and passed. The full comparison uses fresh tasks in all three arms. The completed smoke was revalidated before increasing full-run concurrency from 4 to 8; no smoke outcomes were used to tune the policy or ARM weights. The selection health/manifest records `canonical_no_cot_json_schema/xgrammar` so constrained and unconstrained runs cannot silently resume into one another.

### Automatic launch registered on 2026-09-07

The user authorized starting when GPUs become available. The watcher ran in Slurm step `281697.1` on `g001`, with supervisor PID `86202`. It polled every 30 seconds and required two consecutive checks with no GPU compute processes and less than 1 GiB used on each assigned GPU. It is now stopped.

- Smoke: task indices `0,50,100`, all three arms, concurrency 3.
- Promotion: each arm must finish with at least one valid judged task and executed-action traces; scalar scores must be finite and selection outputs must parse without fallback. Task success is not a smoke requirement. This gate checks functionality, not performance gains or full scientific validation.
- Full run after smoke passes: baseline → scalar → selection, 300 tasks per arm, concurrency 4. Per-task results support resumption.
- Allocation bound: job `281697` ends at **2026-09-07 18:54:27 America/Los_Angeles**. No new allocation or extension is requested. The full comparison may exceed the remaining time; outputs persist if interrupted.
- Run directory: `/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/queued-281697-20260908T003731Z`.
- Live state: [watcher-status.json](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/queued-281697-20260908T003731Z/watcher-status.json). Logs: `supervisor.log`, `allocation-step.log`, then `smoke-launcher.log` and `full-launcher.log`.

At registration both GPUs were occupied. They became free at approximately 17:40 PDT, and the watcher started the smoke at **17:41:02 PDT**. The actor loaded and all three baseline smoke tasks started; the first completed with a valid judge result at 17:42 PDT. No benchmark gain has been established. Source is backed up on scrubbed storage. Initial Git commit attempts were blocked by the project filesystem quota; the validated inference implementation is now committed as `cdbe52a`.

### Training handoff correction at 17:44 PDT

The user reported that RL training was still intended to be running. ARM evaluation and its watcher were stopped; Slurm step `281697.1` disappeared and both GPUs showed no compute processes. Existing unrelated processes were not signaled.

The latest training run, `openwebrl-4b-reference-281697-20260907T225751`, had recorded exit code **1** at **17:40:11 PDT**, before ARM smoke started at **17:41:02 PDT**. Its last phase was checkpoint saving after rollout 2/90; the log reports `CUDA error: invalid argument` while copying checkpoint tensors to CPU. The health log shows an additional host-memory OOM kill near the failure. These observations do not establish a complete root-cause diagnosis.

The original handoff check was insufficient: unused GPU memory can mean a failed or restarting training run. **Do not automatically restart ARM from GPU occupancy alone.** Obtain a deliberate training-to-evaluation handoff or verify successful training completion and that no recovery/restart is pending. The original inference objective remains pending, and partial smoke artifacts are preserved.

Evidence: [training-handoff-check.json](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/queued-281697-20260908T003731Z/training-handoff-check.json).

## 1a. Longer-term training objective and recommendation

Improve held-out browser task completion by the **standalone, one-action policy** using fine-grained action or turn supervision. Measure training efficiency as well as final performance. Evaluate actor-plus-ARM inference separately so that search gains are not mistaken for policy learning.

Initial recommendation: retain terminal task success as the main objective and compare:

1. **Same-state action preference training**, using the selection ARM to label winner–loser pairs for DPO or selected-response SFT, followed by or interleaved with outcome RL.
2. **Outcome GRPO plus a bounded turn-level preference bonus**, initially using a frozen ARM and ordinary policy-sampled executed actions.

Validate the released models on our states before committing to either route. The strongest existing evidence is for inference-time action selection, not RL training with these checkpoints.

## 2. Scope and evidence

The initial investigation inspected:

- The [action-reward-models repository](https://github.com/piotr-teterwak/action-reward-models/tree/02276b0ff3b9048d34e6a2afdcb9042dd9018c8c), pinned to commit `02276b0ff3b9048d34e6a2afdcb9042dd9018c8c`.
- Published model configuration files on Hugging Face, pinned to the revisions recorded above.
- This OpenWebRL working tree, including existing uncommitted experimental changes.
- The primary literature linked below.

All three inference smoke arms passed, baseline is complete, and the full ARM comparison is in progress. No completed three-arm benchmark reproduction or ARM training has been performed. The linked dataset returned HTTP 401 on anonymous access; its actual records have not been inspected. Source-code observations and author-reported results are distinguished from proposed experiments throughout this document.

## 3. What the released models predict

| Model | Input and output | Interpretation |
| --- | --- | --- |
| Selection ARM | Task, screenshot, recent history, candidate actions → winning index | Relative preference within the candidate set |
| Scalar Bradley–Terry RM | State context and one proposed response → scalar | Preference score trained to rank chosen actions above rejected actions |

The models use teacher selections among candidate actions as supervision. Neither is explicitly trained to predict an action's realized contribution to eventual task completion.

For OpenWebRL-4B-SFT on Online-Mind2Web, the repository reports 33.8% success with one action, 46.3% with scalar selection, and 51.1% with selection ARM. These are author-reported inference-time results, using an o4-mini outcome judge, not demonstrated RL training gains. Results using other actors and judges should not be compared directly. [Source: ARM README](https://github.com/piotr-teterwak/action-reward-models/blob/02276b0ff3b9048d34e6a2afdcb9042dd9018c8c/README.md)

### Reproduction findings

- **Backbone mismatch in documentation:** the README broadly describes Qwen3.5-4B, but the [OpenWebRL scalar adapter config](https://huggingface.co/PTeterwak/OpenWebRL-4B-ScalarRM-LoRA/blob/main/adapter_config.json) names `OpenWebRL/OpenWebRL-4B-SFT`; the [merged selection config](https://huggingface.co/PTeterwak/OpenWebRL-4B-SelectionARM/blob/main/config.json) declares `Qwen3VLForConditionalGeneration`. Use checkpoint metadata and verify weight compatibility.
- **Scalar serving mismatch:** the supplied [scalar server](https://github.com/piotr-teterwak/action-reward-models/blob/02276b0ff3b9048d34e6a2afdcb9042dd9018c8c/inference/scalar_server.py) implements the older MolmoWeb prompt path. Reconstruct the OpenWebRL serialization from its training builder and [scalar inference example](https://github.com/piotr-teterwak/action-reward-models/blob/02276b0ff3b9048d34e6a2afdcb9042dd9018c8c/inference/scalar_infer.py), including real history, image preprocessing, assistant branch, and value-head pooling.
- **Split leakage risk:** the [selection builder](https://github.com/piotr-teterwak/action-reward-models/blob/02276b0ff3b9048d34e6a2afdcb9042dd9018c8c/data_generation/openwebrl_actor/build_selection_sft.py) splits by label draw; the [scalar builder](https://github.com/piotr-teterwak/action-reward-models/blob/02276b0ff3b9048d34e6a2afdcb9042dd9018c8c/data_generation/openwebrl_actor/build_scalar_rm_data.py) splits by pair. Repeated states, and potentially sibling pairs, can cross splits. Use task-disjoint validation before expanding states into draws or pairs.
- **Missing extraction dependency:** some data-generation scripts import `openwebrl.frontier_arbiter` from the author's filesystem; it is not present in this checkout. The release needs adaptation rather than direct execution of the entire pipeline.
- **Coordinate contract:** ARM examples expect normalized coordinates in `[0, 1000]`. The current browser config already uses `resize_scale: 1000`. Preserve the policy-space action representation and verify the execution conversion to avoid double normalization.
- **Selection parsing:** invalid output must be recorded as unavailable supervision. The demo's fallback to candidate 1 should not become a training preference label.

## 4. Current OpenWebRL integration points

| Component | Current behavior | Proposed use or change |
| --- | --- | --- |
| [generate_browser.py](../generate_browser.py), `_generate_turn_sample_impl` | Separate samples carry pre-action context, response, log probabilities, trajectory ID, and turn index | Record ARM state context and candidate metadata; optionally sample alternatives at selected turns |
| [reward_browser.py](../reward_browser.py), `reward_func` | Scores the completed trajectory and broadcasts its reward to every turn | Preserve outcome scoring; keep ARM supervision in separate fields |
| [slime/ray/rollout.py](../../slime/ray/rollout.py), `_post_process_rewards` | Normalizes across trajectories and broadcasts one advantage; unequal turn rewards trigger a warning and the first reward is used | Use a custom postprocessor for distinct turn advantages |
| [rl_recipe.py](../rl_recipe.py), `state_advantages` | Existing experimental hook mixes GRPO with a frozen prefix-state baseline | Reuse the hook contract, not the assumption that an ARM is a success-probability provider |
| [dynamic_sampling_filters.py](../../slime/rollout/filter_hub/dynamic_sampling_filters.py) | Rejects groups with uniform terminal rewards | Allow useful local preference supervision from valid all-failure/all-success groups |
| [recipe_data.py](../recipe_data.py) | Exports finished rollout groups, including dynamically rejected groups when configured | Extend records for offline ARM auditing and preferences |
| [loss.py](../../slime/backends/megatron_utils/loss.py), `compute_advantages_and_returns` | GRPO expands a sample scalar over response tokens; PPO handles returns within a sample | Existing GRPO path supports a turn scalar; cross-turn reward-to-go/GAE needs explicit trajectory handling |
| [actor.py](../../slime/backends/megatron_utils/actor.py) | Multi-epoch global shuffle path is actor-only | True critic training requires additional backend work or a compatible training path |

The main implementation trap is that **changing only `reward_func` to emit different turn rewards will not yield the intended credit assignment** under the current default normalization.

Keep task outcome, protocol validity, ARM availability, ARM preferences, and training advantages distinct. Preserve invalid-trajectory handling, and do not feed shaped scores into adaptive task-success counts.

## 5. Integration options and likelihood

“Success” below means improving held-out standalone-policy task completion, preferably at matched total training compute. Likelihoods are qualitative engineering/research judgments, conditional on correct implementation and a useful ARM on our distribution. They are not measured probabilities.

| ID | Route | Mechanism | Likelihood and main limitation |
| --- | --- | --- | --- |
| A | Selection distillation / filtered SFT | Train on ARM-selected responses at visited states; optionally require successful full trajectories | **Medium–high.** Closest to demonstrated selector use; limited by teacher errors and candidate coverage |
| B | Turn-level preference optimization | Label same-state winner–loser pairs and train with DPO or a related preference objective, then continue outcome RL | **Medium–high; first preference-training choice.** Avoids absolute-score calibration, but inherits ranking errors |
| C | Outcome advantage plus local ARM advantage | Add a bounded, same-state-relative preference signal to each executed turn's outcome advantage | **Medium–high; first RL choice.** Good fit for current hooks; local preferences can still oppose eventual success |
| D | Auxiliary RL over candidate actions | Assign local rewards to all candidates; execute one; give actual outcome supervision only to the executed branch | **Medium–high.** More supervision per browser interaction, with extra sampling and training plumbing |
| E | Dense additive rewards with reward-to-go | Add calibrated action rewards along the trajectory and propagate future reward backward | **Medium.** Local preference may not represent incremental progress; loop and length incentives need checking |
| F | Aggregate ARM scores into trajectory reward | Mix terminal success with sum, mean, minimum, or another aggregation; retain trajectory GRPO | **Low–medium.** Simple control, but loses fine-grained credit and introduces aggregation biases |
| G | ARM-only RL | Replace terminal success with preference scores or selections | **Low.** Locally preferred action sequences can fail; diagnostic ablation only |
| H | Potential shaping / state baseline | Train a state potential or success predictor, possibly initialized from ARM representations; use potential differences or `R - V(h_t)` | **Medium after outcome-based training.** Raw ARM scores are not state success values |
| I | Turn-level actor–critic / GAE | Train a critic on actual returns and bootstrap across browser turns | **Medium eventually; lower near-term practicality.** More compute and backend changes |
| J | Outcome-trained implicit PRM | Train an implicit reward model online from successful/failed trajectories; combine turn and outcome advantages | **Medium, promising second stage.** Addresses policy drift but extends beyond the released ARM |
| K | Counterfactual progress / reward redistribution | Execute alternative continuations in reproducible environments; learn success-likelihood changes or redistribute terminal credit | **Medium in resettable environments; low near-term on live websites.** Expensive restoration and continuation sampling |
| L | ARM-guided collection / recovery / curriculum | Select demonstrations, recovery states, or rollout actions using ARM | **Medium for downstream training.** Useful support route; selected rollouts change the behavior distribution |

Inference-only best-of-five is a separate positive control with the strongest direct evidence for actor-plus-ARM gains. It does not establish standalone policy improvement.

## 6. First hybrid RL formulation

Let `h_t` be the pre-action history, `a_t` the executed response, and `A_outcome[i]` the current trajectory GRPO advantage. Sample `K` candidate responses at exactly the same history. Let `e` index the executed candidate and `q(h, a)` be the frozen scalar ARM score.

Construct a bounded relative score:

```text
u_t = mean over j != e of:
      2 * sigmoid((q(h_t, a_t) - q(h_t, a_tj)) / temperature) - 1

A[i, t] = A_outcome[i] + alpha * confidence_gate[t] * u_t
```

The sigmoid expresses a preference model, **not a probability of task success**. Validate its temperature. Start with a small coefficient sweep such as `alpha ∈ {0.05, 0.1, 0.25}` after measuring advantage scales. These are proposed values, not established defaults.

For the selection ARM, prefer winner–loser pairs in an auxiliary preference loss. A centered winner indicator is a simpler, coarse local-bonus ablation. Candidate-order permutations can test reliability; agreement is not itself calibrated confidence.

### Estimator and data constraints

1. Initially execute a designated ordinary policy sample independently of ARM ranking. ARM-selected execution changes the behavior policy; its original actor log probability does not capture the selection probability.
2. Compare candidates within the same state. Bradley–Terry comparisons are unchanged by an arbitrary state-dependent score offset, so raw scores lack a reliable common origin across states.
3. Treat the local bonus as an auxiliary preference objective. It is not an unbiased estimate of the terminal-outcome advantage. If optimizing cumulative dense reward instead, compute reward-to-go explicitly across turns.
4. Apply one turn advantage across the generated response initially, matching current training. Action-token-only weighting is a separate ablation. Multiple tool calls in a response require explicit boundaries before claiming individual-action credit.
5. Unexecuted candidates have local preference labels, not observed terminal outcomes. Never copy the executed trajectory's success label onto alternative branches.
6. ARM inference failures contribute no ARM term; valid outcome supervision can remain usable. Invalid browser trajectories retain existing exclusion semantics in the first experiment.
7. Freeze ARM weights and score transformations within an update window. Log model and prompt versions, candidate sampling settings, and score scales.
8. Handle duplicate actions and ties explicitly. A selector always choosing a winner does not prove that any candidate is good, or that distinct text represents distinct actions.
9. Changing the dynamic filter changes the training distribution. Compare against an outcome-only control with matched retained groups/update budgets where needed, rather than attributing all gains to ARM credit.

## 7. Other reward formulations worth testing later

### Dense reward-to-go

```text
r_t = terminal_outcome_if_final + beta * calibrated_process_reward_t
G_t = sum over u >= t of gamma^(u-t) * r_u
A_t = G_t - baseline(h_t)
```

Use only after validating that process scores can sensibly be accumulated. Bound the auxiliary contribution and test loop, delay, and premature-completion incentives. Summing preferred-action scores can reward long trajectories; averaging changes the objective too.

### Potential-based shaping

```text
r'_t = r_t + beta * (gamma * Phi(h_{t+1}) - Phi(h_t))
```

Use a frozen history-state potential with correct terminal treatment and consistent discounting. With complete Monte Carlo returns and zero terminal potential, the shaping terms telescope to a pre-action baseline. Broadcasting only the total shaped trajectory reward does not create useful new turn attribution. Policy-invariance theory does not automatically guarantee identical behavior under clipped objectives, filtering, or approximate training.

### State-value and critic routes

Train `V(h_t)` on actual terminal outcomes from task-disjoint or earlier-policy data. The current `state_advantages` hook already supports a frozen `R - V` experiment, but the ARM value head is a preference head, not this predictor. Cross-turn GAE also requires successor links, terminal/truncation treatment, and a compatible critic training path.

### Outcome-trained implicit rewards and redistribution

Consider an iStar/PRIME-style outcome-trained model if frozen ARM quality deteriorates as the actor changes. This requires an explicit actor-response likelihood training objective; likelihood of a selector's index output is not automatically an implicit reward for actor actions.

Counterfactual continuations can supervise actual progress in resettable environments. Return redistribution is another longer-term possibility; assigning terminal reward proportionally to arbitrary ARM scores does not inherit RUDDER's guarantees.

## 8. Experiment sequence

### Phase 0 — Model and data contract

- Pin model, adapter, base, processor, prompt, and code revisions.
- Reproduce selection prompts and OpenWebRL scalar serialization/pooling.
- Verify normalized coordinates, action syntax, history, screenshot timing, and image preprocessing.
- Build a few hundred task-disjoint evaluation states spanning navigation, typing, scrolling, recovery, and completion.
- Measure ranking agreement, duplicate/tie behavior, candidate-order sensitivity, reasoning sensitivity, parse failures, latency, and memory use.
- Verify dataset accessibility and provenance before relying on released records.

Deliverable: reproducible scoring fixtures and an ARM audit report. Decide whether either released model is reliable enough for a pilot. Set numerical acceptance thresholds after inspecting pilot variability, before the main comparison.

### Phase 1 — Observation-only scoring

- Collect scores alongside ordinary rollouts without changing executed actions.
- Record current URL, pre-action screenshot, prior actions, proposed responses, candidate IDs, outcome, and model/prompt versions.
- Test effective actions against loops, premature completion, and plausible but irrelevant alternatives.
- Use a small resettable subset to check actual action consequences.
- Keep storage and serving overhead visible in the report.

Deliverable: evidence that ARM preferences discriminate useful behavior on our policy distribution.

### Phase 2 — Controlled training comparison

Start from the same actor checkpoint:

| Branch | Training |
| --- | --- |
| Baseline | Current outcome-only GRPO |
| Hybrid | Outcome GRPO plus bounded local ARM advantage |
| Preference | Selection-based turn preference training plus outcome GRPO |

Use inference-only ARM selection as a positive control. Hold task distribution, horizon, memory settings, outcome judge, and evaluation protocol fixed. Determine whether preference training is a short preparatory stage or an interleaved auxiliary loss before implementation.

### Phase 3 — Evaluation and decision

- Primary metric: held-out standalone one-action task completion.
- Secondary metrics: browser interactions, generated tokens, GPU-hours, wall time to a target success rate, trajectory length, invalid-action rate, loops, premature completion, and ARM/outcome disagreement.
- Report both environment-interaction-matched and total-compute-matched comparisons where feasible. Five candidate generations have a real cost even with prefix caching.
- Use multiple training seeds and task-level uncertainty estimates; do not treat turns from one task as independent observations.
- Separate development tasks from final held-out tasks and assess site/domain transfer where possible.
- Keep ARM-assisted evaluation separate from standalone evaluation.

Promote a method based on outcome improvements and cost, not increasing ARM scores alone.

### Phase 4 — Expand according to the observed bottleneck

- Ranking works but training gains fade: refresh ARM on current-policy data.
- Local preferences fail to predict progress: collect execution-grounded progress labels.
- Candidate generation dominates cost: distill the selector into a cheaper scalar model or directly into the actor.
- Credit assignment remains weak: investigate counterfactual progress, a state baseline, or cross-turn critic training.

## 9. Proposed implementation units

These are proposed changes, not completed work:

1. **ARM scoring adapter:** separate frozen service/client, model-specific serialization, batched requests, provenance, and explicit unavailable-score handling.
2. **Turn/candidate records:** causal pre-action context, stable candidate identity, response tokens/log probabilities, executed flag, and ARM outputs.
3. **Hybrid advantage postprocessor:** preserve outcome normalization; compute distinct turn bonuses; retain separate raw outcome metrics.
4. **Preference data/training path:** same-state pairs or selected-response examples, with task-disjoint splits and no invented outcomes for unexecuted candidates.
5. **Filtering and evaluation controls:** preserve valid local supervision from uniform-outcome groups and isolate the effect of changed sampling.

Meaningful verification should cover prompt/pooling parity, coordinate conversion, turn-to-trajectory mapping, invalid-score fallback, duplicate handling, the `alpha=0` baseline, and the fact that distinct turn advantages survive transport into the actor loss. GPU/browser tests are needed for serving, multimodal processing, and end-to-end behavior; arithmetic tests alone cannot establish these.

## 10. Literature and implications

| Source | Relevant evidence | Implication and limitation |
| --- | --- | --- |
| [Web-Shepherd](https://arxiv.org/abs/2505.15277) | Web-specific step preference model and reward-guided search | Supports domain-specific ranking; does not establish RL gains from these ARM checkpoints |
| [WebArbiter](https://arxiv.org/abs/2601.21872) | Comparative web action judging and guided search | Supports testing selection supervision; search and policy training remain separate |
| [iStar: Agentic Reinforcement Learning with Implicit Step Rewards](https://arxiv.org/abs/2509.19199) | Combines episode and implicit step advantages in agentic RL, including WebShop | Closest motivation for the hybrid route; its PRM is outcome-trained, unlike the released explicit ARM |
| [PRIME: Process Reinforcement through Implicit Rewards](https://arxiv.org/abs/2502.01456) | Online process reward updates from outcome-labeled rollouts | Motivates adaptation under policy drift; reported evidence is in math/coding |
| [Rewarding Progress](https://arxiv.org/abs/2410.08146) | Process rewards based on changes in future success likelihood under a prover policy | Motivates execution-grounded progress supervision; preference scores need not equal progress |
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | Policy optimization from pairwise preferences | Natural objective for same-state winner–loser pairs; inherits preference errors |
| [Ng, Harada, and Russell: Policy Invariance Under Reward Transformations](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf) | Potential-based reward shaping | Requires proper state potential, discount, and terminal treatment |
| [RUDDER](https://papers.nips.cc/paper/2019/file/16105fb9cc614fc29e1bda00dab60d41-Paper.pdf) | Return decomposition and reward redistribution | Longer-term credit assignment route; arbitrary ARM-weighted redistribution is not equivalent |

## 11. Open decisions for iteration

- [x] **First milestone:** reproduce inference gains on Online-Mind2Web with the named selection and scalar ARMs; training choices remain deferred.
- [ ] **First comparison:** run baseline + hybrid + preference branches, or narrow to one ARM training route after the audit?
- [ ] **Candidate budget:** five candidates on every turn for initial validation, then how much turn subsampling is affordable in training?
- [ ] **Evaluation split:** which tasks/sites are development, held-out validation, and final test, and what overlap exists with released ARM training data?
- [ ] **Budget and promotion criteria:** available GPU-hours/browser interactions, minimum worthwhile improvement, and acceptable overhead?
- [ ] **Preference schedule:** preparatory DPO/SFT stage or interleaved auxiliary training?
- [ ] **ARM refresh:** frozen for the first experiment; when should outcome-grounded or current-policy updates become a separate experiment?

## 12. Decision log

| Date | Decision or request | Status |
| --- | --- | --- |
| 2026-09-07 | Investigate ARM and propose integration options before implementation | Source investigation and initial proposal completed; model validation remains pending |
| 2026-09-07 | Record the plan in a Markdown file and iterate on that document | This file is the working draft |
| 2026-09-07 | Validate inference first using SelectionARM and ScalarRM-LoRA on Online-Mind2Web | Models and evaluation implementation prepared; GPU validation pending |
| 2026-09-07 | Start eval runs when GPUs become available | Smoke started at 17:41 PDT; stopped at 17:44 PDT after the user raised a training-ownership concern |
| 2026-09-07 | Use the new GPU on g001 for ARM evaluation | Dedicated allocation 282209; baseline completed; scalar saved 277/300 tasks before expiry; ARM concurrency increased to 16 |
| 2026-09-07 | Continue evaluations on zixianma@g005 within its four-hour session | Dedicated allocation 282782; resume 23 unfinished scalar tasks, then all 300 SelectionARM tasks; existing outcomes preserved |
| 2026-09-07 | Prepare ARM-filtered SFT and record inference results with both denominators | Results and paired analysis recorded; 128/32/32-task SFT preparation split written on CPU; matched invalid-task retries proposed, not launched |
| 2026-09-08 | Launch retries after SelectionARM and record separately | Prepared and initially queued a 56-task baseline/ScalarRM pass; superseded by the cohort-review clarification below before any retry started |
| 2026-09-08 | Reconsider retry queries after SelectionARM; start SFT with C2; explain C3 and the 128 tasks | Automatic retries held; decide from all three completed runs. C2 first, C1 optional later, C3 deferred. Documented the arbitrary pilot size and deterministic hostname sampling; final data budget remains under discussion |
| 2026-09-08 | Use all ~2K tasks and prepare C2 after eval | All 2091 deduplicated prompts; all eligible successful-trajectory turns; C2-only LoRA for two fixed epochs. Resumable handoff prepared in existing allocation, with retries held and no new compute request |

## 13. Self-critique carried forward

- Initial “medium–high” training likelihoods are hypotheses, not established downstream success probabilities. Inference validation now precedes choosing a training route.
- Same-state preference is not necessarily incremental progress; all candidates can be bad.
- Uniform-outcome groups have zero GRPO outcome advantage, so a local bonus can drive their entire update. Retaining those groups must be a separate training ablation.
- Short ARM context and reasoning/style shortcuts can cause disagreement with a better-informed actor.
- Filtering, loss weighting, candidate generation, and training objectives should be changed separately for interpretable experiments.
- Candidate generation and scoring costs must be measured before choosing a training budget.
