# ARM selection head: choosing actions from the actor's own hidden states

**A small head on the frozen SFT actor's hidden states is a weaker action selector than the released Piotr SelectionARM.** On 124 fixed branching states, the best preselected head matches an independent teacher (Luna before execution) on 61.7% of judgments, against 73.2% for Piotr ARM: −11.5 pp [−18.5, −4.6]. No frozen layer, pooling or head size closes the gap. Chosen-action continuation success is too noisy to separate the selectors: Piotr ARM, Luna and the heads all land at roughly 34–37%. A top-layer scoring adapter that fine-tunes the actor's last eight layers for scoring is training now (job `353150`).

The question is whether selection can be built into the actor ([integration option 3](ARM_INTEGRATION_PLAN.md)). Today's best-of-N pipeline samples five candidates from the actor, then runs a separate 4B selector over the screenshot and all candidates. A head that reads the hidden states the actor computes while generating would need no second model.

<a id="selection-head-evaluations"></a>

## Three evaluations, three different questions

| Evaluation | Count | What it measures | Main limitation |
| --- | --- | --- | --- |
| GPT-5.5 teacher agreement, held-out task groups | 3,759 panels from 291 states | Imitation of the training teacher on unseen tasks; used to choose checkpoints | Piotr ARM cannot be scored: its draw-level training split very likely covers these tasks |
| Luna agreement, branching states | 124 states × 4 judgments | Agreement with an independent strong teacher that neither selector was trained on | Agreement is not correctness |
| Chosen-action continuation success, branching states | 124 states × 5 actions × 3 continuations | How often SFT continuations succeed after the chosen action; invalid outcomes count as zero | Low power: paired intervals about ±3.5 pp |

A choice counts as agreeing when it is action-equivalent to the reference choice: identical tool calls, with clicks within five normalized units. Branching intervals use 20,000 state-bootstrap draws; each state is a distinct task.

<a id="selection-head-results"></a>

## Head versus Piotr SelectionARM on the 124 branching states

| Selector | States | Luna agreement [95%] | − Piotr, pp [95%] | Continuation success | − Piotr, pp [95%] |
| --- | ---: | --- | --- | ---: | --- |
| Luna judgment versus another Luna judgment | 124 | 80.2% | — | — | — |
| Piotr SelectionARM (separate 4B model) | 124 | 73.2% [66.3, 79.6] | — | 35.75% | — |
| Head, one candidate at a time | 124 | 61.7% [54.2, 69.2] | −11.5 [−18.5, −4.6] | 34.68% | −1.08 [−4.30, +2.15] |
| Head, joint comparison | 124 | 60.7% [53.0, 68.1] | −12.5 [−20.0, −5.4] | 33.60% | −2.15 [−5.65, +1.34] |
| Actor first sample | 124 | 56.2% [48.4, 63.9] | — | 29.84% | — |
| Uniform candidate | 124 | 46.9% | — | 32.42% | — |

Luna before execution itself scores 35.11% continuation success on 119 of these states ([branching results](ARM_FORMULATIONS.md#arm-continuation-branches-results-20261007)). The heads choose the same action as Piotr ARM on 78/124 (one at a time) and 72/124 (joint) states. Joint comparison did not help: joint minus one-at-a-time continuation success is −1.08 pp [−3.23, +0.81], with equal teacher agreement.

On held-out GPT-5.5 panels, both heads agree with the teacher on about 65.5%. The baselines score 50.1% for the most common action and 44.7% for a uniform choice. Both heads overfit: training-panel agreement reaches 79–90%.

<a id="selection-head-sweep"></a>

## Frozen-feature sweep: no layer or pooling closes the gap

All nine configurations use the one-at-a-time head, train for ten epochs and choose the checkpoint by GPT-5.5 validation agreement. Luna agreement and continuation success are reported only.

| Frozen features | Validation agreement | Luna agreement | Continuation success | − Uniform, pp [95%] |
| --- | ---: | ---: | ---: | --- |
| Layer 9 | 61.9% | 55.9% | 30.65% | −1.77 [−5.16, +1.77] |
| Layer 18 | 63.0% | 57.7% | 36.02% | +3.60 [+0.43, +6.83] |
| Layer 27 | 64.6% | 60.1% | 36.56% | +4.14 [+0.75, +7.63] |
| Layer 36 | 64.7% | 60.9% | 33.06% | +0.65 [−2.53, +3.87] |
| Layers 9, 18, 27, 36 | 64.5% | 58.3% | 34.14% | +1.72 [−1.51, +5.16] |
| Layers 18, 27, 36: action tokens only | 64.6% | 63.9% | 36.02% | +3.60 [+0.27, +6.99] |
| Layers 18, 27, 36: reasoning only | 55.7% | 57.7% | 32.53% | +0.11 [−3.06, +3.39] |
| Layers 18, 27, 36: end token only | 60.6% | 61.9% | 32.53% | +0.11 [−3.33, +3.60] |
| Layers 18, 27, 36: smaller regularized head | **65.9%** | 56.0% | 35.22% | +2.80 [−0.65, +6.34] |

- **Frozen ceiling.** Validation agreement saturates at 64.5–65.9%. The best Luna agreement, 63.9%, remains about 9 points below Piotr ARM.
- **Action tokens carry the signal.** Reasoning-only features are weakest and peak at epoch 1, while action-only features match all pools combined. This is the opposite of Piotr's text selector, which lost 6.4 points when candidate reasoning was removed ([compact-input ablation](ARM_INFERENCE_SCALING.md)).
- **High continuation-success rows are post hoc.** Layer 27 (36.56%), layer 18 and action-only (36.02%) are the best of nine noisy estimates. None differs from Piotr ARM: layer 27 minus Piotr is +0.81 pp [−2.96, +4.84]. The configuration chosen by the preset rule, the smaller head, scores 35.22%.

<a id="selection-head-adapter"></a>

## Top-layer scoring adapter (running)

**Hypothesis:** frozen features miss what Piotr's separately fine-tuned 4B selector learns. The adapter adds rank-16 LoRA on the last eight of 36 language layers and is used **only to score candidates**:

1. The frozen SFT actor generates all five candidates, so candidates match the SFT and SFT + Piotr ARM baselines.
2. The prompt keeps the KV cache the actor already computed, with the adapter off.
3. Only candidate tokens pass through the adapted top layers. Scoring adds roughly 8/36 of one forward pass over the candidate tokens, against a full second 4B model for Piotr ARM.

Training starts exactly at the one-at-a-time head, because LoRA starts as an identity. It uses GPT-5.5 choices from the same 35,916 training panels and chooses checkpoints on a 100-state validation probe. At step 0 the probe scores 63.8%; at step 100, 61.5%, within the probe's noise. The best checkpoint is then scored on full validation and on the branching states. If it approaches Piotr ARM's 73.2% Luna agreement, the next step is a matched full300 inference comparison: SFT alone, SFT + Piotr ARM and SFT + adapter head.

<a id="selection-head-next"></a>

## What would settle the comparison

- **Power.** With 124 states, continuation success cannot resolve differences of 1–3 points. The planned outcome-labeled scale-up ([branching proposal](ARM_FORMULATIONS.md#branch-selector-scaleup-20261008)) would also allow outcome-supervised heads instead of teacher imitation.
- **Supervision.** Every selector here imitates GPT-5.5. Luna agreement and continuation success are only loosely related, so better imitation need not mean better actions.
- **Serving.** A deployed head needs the serving engine to return hidden states, or a sidecar scorer. Neither has been benchmarked.

<details>
<summary>Method: features, head and data</summary>

- **Features.** Frozen OpenWebRL-4B-SFT, bf16. For each candidate: the mean of reasoning-token states, the mean of action-token states and the end-token state, at layers 9/18/27/36; plus the prompt's last-token state. At serving these are by-products of generation. Offline, they are recomputed from the actor's exact tokens after a cached prompt prefix.
- **Head.** Fixed per-dimension standardization fitted on training features, then a projection to 512 dimensions. The joint variant adds a two-layer transformer over the state and five candidates without position embeddings, so candidate order cannot matter. The one-at-a-time variant replaces it with an MLP per candidate.
  - A per-candidate LayerNorm input was rejected: unit tests showed it erases within-panel differences.
- **Training data.** Piotr's pinned release `0d83b48` with the joint-data v2 task split and quarantine: 39,675 panels over 2,982 states, 35,916 for training. Excluded draw records: 300 conflicting draw IDs, 80 quarantined, 61 missing teacher selections and 32 outside the split.
- **Targets.** Soft cross-entropy that spreads the teacher's choice over action-equivalent candidates.
- **Checks.** Cached, batched extraction matches full forwards exactly in float32 unit tests. On the real bf16 model the relative difference is 1.43–1.48%, under a 2% abort threshold. Adapter tests confirm identity at initialization, an unchanged prompt and lower layers, and gradients reaching only the adapter.

</details>

<details>
<summary>Branching-state inputs, Piotr ARM protocol and subsets</summary>

- **Prompts.** Branch anchors do not save the actor prompt. It is rebuilt from continuation-rollout text and accepted only when its SHA-256 equals the hash journaled at candidate sampling: 114/124 match, and all 114 also reproduce the server's prompt token count, including image tokens. In the other 10, replayed prefixes word tool feedback differently from discovery; they use the replay prompt.
- **Candidates.** The heads read the actor's exact generated token ids; re-tokenizing the saved text would change 47/620 candidates.
- **Piotr ARM.** Run offline through the serving code, with its canonical prompt builder and JSON-schema decoding. It returned a valid selection on all 124 states and chose candidate 0 on 62.
- **Subsets.** Results on the 114 prompt-verified states, and on the 105 whose tasks are absent from Piotr's release, lead to the same conclusions. Piotr ARM scores 35.96% and 36.83% on them.

</details>

<details>
<summary>Compute, code and artifacts</summary>

| Job | Resources | Purpose | Elapsed | Status |
| --- | --- | --- | --- | --- |
| `351647` | 1 H200, 8 CPUs, 120 GiB, 3 h approved | Features for 39,675 panels and 124 states; Piotr ARM offline; two heads | 1 h 58 m | Completed |
| `353150` | 1 H200, 2 CPUs, 120 GiB, 4 h approved | GPU smoke test, nine-configuration sweep, scoring adapter | — | Running |

CPU-only jobs `353141` and `353145`, and the 8-CPU GPU job `353148`, were cancelled while pending because no node had enough idle CPUs. They consumed no allocation. No API calls or browsers were used.

Code is on branch `arm-selection-head`: `openwebrl/selection_head.py`, `selection_head_features.py`, `selection_adapter.py` and the scripts they call. Runtime artifacts are under `openwebrl-runtime/selection-head-20261008/`. [Aggregate estimates and accounting](arm_results/rl_integration/selection-head-20261009.json).

</details>
