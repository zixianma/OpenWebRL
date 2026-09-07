# Action reward models for OpenWebRL training

Status: **Draft for discussion; implementation has not been approved.**

Created: 2026-09-07. This is the working document for iterating on the plan originally proposed in conversation.

## 1. Objective and recommendation

Improve held-out browser task completion by the **standalone, one-action policy** using fine-grained action or turn supervision. Measure training efficiency as well as final performance. Evaluate actor-plus-ARM inference separately so that search gains are not mistaken for policy learning.

Initial recommendation: retain terminal task success as the main objective and compare:

1. **Same-state action preference training**, using the selection ARM to label winner–loser pairs for DPO or selected-response SFT, followed by or interleaved with outcome RL.
2. **Outcome GRPO plus a bounded turn-level preference bonus**, initially using a frozen ARM and ordinary policy-sampled executed actions.

Validate the released models on our states before committing to either route. The strongest existing evidence is for inference-time action selection, not RL training with these checkpoints.

## 2. Scope and evidence

The initial investigation inspected:

- The [action-reward-models repository](https://github.com/piotr-teterwak/action-reward-models/tree/02276b0ff3b9048d34e6a2afdcb9042dd9018c8c), pinned to commit `02276b0ff3b9048d34e6a2afdcb9042dd9018c8c`.
- Published model configuration files on Hugging Face. Their revisions still need pinning before a reproducible experiment.
- This OpenWebRL working tree, including existing uncommitted experimental changes.
- The primary literature linked below.

No model inference, benchmark reproduction, or ARM training has been performed. The linked dataset returned HTTP 401 on anonymous access; its actual records have not been inspected. Source-code observations and author-reported results are distinguished from proposed experiments throughout this document.

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
| [generate_browser.py](generate_browser.py), `_generate_turn_sample_impl` | Separate samples carry pre-action context, response, log probabilities, trajectory ID, and turn index | Record ARM state context and candidate metadata; optionally sample alternatives at selected turns |
| [reward_browser.py](reward_browser.py), `reward_func` | Scores the completed trajectory and broadcasts its reward to every turn | Preserve outcome scoring; keep ARM supervision in separate fields |
| [slime/ray/rollout.py](../slime/ray/rollout.py), `_post_process_rewards` | Normalizes across trajectories and broadcasts one advantage; unequal turn rewards trigger a warning and the first reward is used | Use a custom postprocessor for distinct turn advantages |
| [rl_recipe.py](rl_recipe.py), `state_advantages` | Existing experimental hook mixes GRPO with a frozen prefix-state baseline | Reuse the hook contract, not the assumption that an ARM is a success-probability provider |
| [dynamic_sampling_filters.py](../slime/rollout/filter_hub/dynamic_sampling_filters.py) | Rejects groups with uniform terminal rewards | Allow useful local preference supervision from valid all-failure/all-success groups |
| [recipe_data.py](recipe_data.py) | Exports finished rollout groups, including dynamically rejected groups when configured | Extend records for offline ARM auditing and preferences |
| [loss.py](../slime/backends/megatron_utils/loss.py), `compute_advantages_and_returns` | GRPO expands a sample scalar over response tokens; PPO handles returns within a sample | Existing GRPO path supports a turn scalar; cross-turn reward-to-go/GAE needs explicit trajectory handling |
| [actor.py](../slime/backends/megatron_utils/actor.py) | Multi-epoch global shuffle path is actor-only | True critic training requires additional backend work or a compatible training path |

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

- [ ] **Primary scope:** strongest standalone-policy improvement, allowing DPO/SFT, or specifically online-RL reward integration?
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
| Pending | Choose online-RL-only versus broader policy-improvement scope | Open; the request to save the plan does not resolve this choice |
