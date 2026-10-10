# ARM selection head: choosing actions from the actor's own hidden states

**A small head on the frozen SFT actor's hidden states is a weaker action selector than the released Piotr SelectionARM.** On 124 fixed branching states, the best preselected head matches an independent teacher (Luna before execution) on 61.7% of judgments, against 73.2% for Piotr ARM: −11.5 pp [−18.5, −4.6]. No frozen layer, pooling or head size closes the gap. Chosen-action continuation success is too noisy to separate the selectors: Piotr ARM, Luna and the heads all land at roughly 34–37%. Fine-tuning the actor's top eight layers did not help either, with candidates encoded one at a time or compared through a Kev-style `<decide>` token in those layers. That token's pointer learned to 56.8% agreement on its own, below the head's 65%. A full Kev-style pass through all 36 layers is built but not yet run ([designs](#selection-head-designs)).

The question is whether selection can be built into the actor ([integration option 3](ARM_INTEGRATION_PLAN.md)). Today's best-of-N pipeline samples five candidates from the actor, then runs a separate 4B selector over the screenshot and all candidates. A head that reads the hidden states the actor computes while generating would need no second model.

<a id="selection-head-architecture"></a>

![Selection head inside the actor: (A) at inference the frozen SFT actor samples five candidates and a small head scores them from the actor's hidden states, read directly or through a top-layer scoring adapter; (B) training stage 1 fits the head on cached frozen features, stage 2 trains LoRA on the top eight layers plus the head with the prompt encoded by the unadapted actor](arm_results/methods/selection_head_workflows.svg)

[PNG](arm_results/methods/selection_head_workflows.png) / [SVG](arm_results/methods/selection_head_workflows.svg); rendered by `scripts/render_selection_head_figure.py`.

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

## Top-layer scoring adapter: no gain

**Fine-tuning the actor's top eight layers for scoring, with candidates still encoded one at a time, does not improve on the frozen head.** The adapter (rank-16 LoRA on the last eight of 36 language layers) scores candidates only: the frozen SFT actor still generates them, and the prompt keeps the KV cache the actor computed with the adapter off. Training starts exactly at the one-at-a-time head, because LoRA starts as an identity, and uses the same GPT-5.5 teacher panels.

| Selector | Held-out GPT-5.5 agreement | Luna agreement | Continuation success |
| --- | ---: | ---: | ---: |
| Frozen one-at-a-time head | 65.4% | 61.7% | 34.68% |
| Top-8 adapter, best checkpoint (step 150 of 525) | 65.5% | 58.7% | 33.60% |
| Piotr SelectionARM | — | 73.2% | 35.75% |

The final run froze the head for its first 100 steps and used learning rates of 5e-5 (adapter) and 1e-5 (head). It trained for 525 optimizer steps over 4,205 states, about 0.47 of a pass over all panels. On a fixed 250-state validation probe it stayed within 64.4–65.2% of its 65.0% start, with training loss near 0.92. Its latest checkpoint is saved and can be extended. The Luna and continuation differences are within the 124-state noise.

<details>
<summary>Earlier adapter attempts</summary>

Two earlier runs used a 100-state probe, about ±2 points of chance movement, and saved only their best checkpoint, which stayed at step 0.

| Run | Adapter / head learning rate | Probe agreement by step | Training loss | Elapsed |
| --- | --- | --- | --- | --- |
| `353150` | 1e-4 / 1e-4 | 63.8% → 61.5% → 61.8% (steps 0, 100, 200) | 0.958 → 1.039 | 43 m 55 s, stopped |
| `353205` | 2e-5 / 1e-5 | 63.8% → 63.2% → 61.7% → 62.1% (steps 0–300) | 0.909 → 0.948 → 0.924 | 1 h 03 m 45 s, stopped |
| `353247` | 5e-5 / 1e-5, head frozen 100 steps | 65.0% → 65.2% → 64.4% → 64.5% (steps 0–450; 250-state probe) | 0.923 → 0.939 → 0.922 | 2 h 08 m 20 s, completed |

</details>

<a id="selection-head-designs"></a>

## Kev-inspired variants: comparing candidates inside the actor

![Five selector designs compared: Piotr SelectionARM, the one-at-a-time actor-feature head, the joint top-layer variant, a Kev-style decision pass on the actor and Kev; rows give weights, inputs, which candidates can attend to each other, order effects, readout, added cost and status](arm_results/methods/selection_designs_compared.svg)

[PNG](arm_results/methods/selection_designs_compared.png) / [SVG](arm_results/methods/selection_designs_compared.svg); rendered by `scripts/render_selection_designs_figure.py`.

Kev ([jaredpalmer/kev](https://github.com/jaredpalmer/kev), an open reproduction of TypeSafe's Jev) packs a state, a question and its options into one sequence. A `<decide>` token reads every option, and a pointer scores q(`<decide>`) · k(option end). Neither the one-at-a-time head nor the top-8 adapter lets candidates interact. Two variants borrow Kev's mechanisms; both keep candidate order irrelevant by giving every candidate the same position ids, which is possible because the actor is attention-only.

| Variant | What runs | Trained | Readout | Warm start | Status |
| --- | --- | --- | --- | --- | --- |
| Joint top layers | Layers 1–28 reused per candidate; layers 29–36 once over [candidates, `<decide>`], `<decide>` reading every candidate | LoRA on top 8 layers, pointer, head | One-at-a-time head + gated pointer | Exact | No gain; pointer alone reaches 56.8% |
| Kev-style pass | Prompt, screenshot, candidates and `<decide>` re-encoded through all 36 layers; panels packed like Kev's questions | LoRA on all 36 layers, pointer, `<decide>` embedding | Pointer only | None | Built and tested; GPU job prepared |

Differences from Kev itself: the input is the actor's own prompt with the screenshot rather than a text page description; there is no question instruction; candidates are the actor's generated responses; Kev-27B cannot isolate options (it is a hybrid DeltaNet model), so its options see earlier ones; and Kev is trained on general decision datasets with a fitted temperature, not on GPT-5.5 web choices. Unit tests on a small Qwen3-VL confirm exact warm start (joint), candidate isolation at generation positions, order and packing invariance, and gradients reaching only the trained parts.

**Joint top layers: the comparison pointer learns, but stays below the head.** In job `353389`, the pointer's own validation-probe agreement rose from chance to 56.8%; the head alone scores 65%. The gate weighting the pointer stayed near zero, so the combined score never beat the step-0 checkpoint. The selected checkpoint is therefore the frozen head: held-out agreement 65.4%, Luna agreement 60.9%, continuation success 34.95%.

| Step | Combined probe agreement | Pointer alone | Gate |
| ---: | ---: | ---: | ---: |
| 0 | 64.9% | 46.5% | 0 |
| 150 | 64.0% | 55.8% | 0.012 |
| 300 | 63.9% | 56.8% | −0.005 |

The probe has 250 states; the run trained 373 optimizer steps over 2,987 states. An earlier attempt (`353319`) initialized the gate at zero without a pointer loss, which gave the pointer no gradient; it was stopped after 150 steps with the gate at 0.0021. The trained pointer's weights are in the saved latest checkpoint but have not been scored on the branching states.

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
| `353150` | 1 H200, 2 CPUs, 120 GiB; 4 h approved for all adapter runs | GPU smoke test, nine-configuration sweep, adapter run 1 | 43 m 55 s | Stopped |
| `353205` | same budget | Adapter run 2 | 1 h 03 m 45 s | Stopped |
| `353247` | same budget, 2 h 10 m limit | Adapter run 3 | 2 h 08 m 20 s | Completed |
| `353260`, `353319`, `353389` | 1 H200, 2 CPUs, 120 GiB; 3 h approved for all joint runs | Joint top layers: bf16 crash at smoke; inert-gate run; final run | 46 s, 53 m 47 s, 1 h 56 m 09 s | Failed, stopped, completed |

CPU-only jobs `353141` and `353145`, and the 8-CPU GPU job `353148`, were cancelled while pending because no node had enough idle CPUs. They consumed no allocation. No API calls or browsers were used.

Code is on branch `arm-selection-head`: `openwebrl/selection_head.py`, `selection_head_features.py`, `selection_adapter.py` and the scripts they call. Runtime artifacts are under `openwebrl-runtime/selection-head-20261008/`. [Aggregate estimates and accounting](arm_results/rl_integration/selection-head-20261009.json).

</details>
