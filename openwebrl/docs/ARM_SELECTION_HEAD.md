# ARM selection head: choosing actions from the actor's own hidden states

**No selector trained on teacher labels reliably picks actions that succeed more often than a uniform pick. Choosing by observed continuation outcomes does.** On 138 branching states with eight SFT continuation draws per action (120 task groups), each selector's chosen action is scored on held-out draws. Choosing the action that succeeded most often on five draws reaches **36.99%** against **31.67%** for a uniform pick: **+5.32 pp [+3.09, +7.62]**. Piotr SelectionARM reaches 34.15% (+2.48 pp [−0.14, +5.21]). Every actor-based selector lies between 31.34% and 33.70%, from −0.33 to +2.03 pp relative to uniform, and none differs from uniform at the 95% level.

**Agreement with teachers does not track action quality.** The Kev-style decision pass on the actor's weights, trained to step 600, matches held-out GPT-5.5 choices best of any actor-based selector: 67.1%, versus 65.4% for the frozen-feature head. On the original 124 states it matches an independent teacher (Luna before execution) on 66.1% of judgments, versus 73.2% for Piotr ARM. Its chosen actions succeed on 32.79% of held-out draws, +1.12 pp [−1.38, +3.75] over uniform.

**Next step: supervise the selector with continuation outcomes instead of teacher picks.** 188 states currently have eight draws for all five actions: 9,720 outcomes in 151 task groups. That is 14× fewer states than the 2,692 teacher-labelled training states used so far. Collection is ongoing ([data comparison](#selection-head-outcome-data-20261010)).

The question is whether selection can be built into the actor ([integration option 3](ARM_INTEGRATION_PLAN.md)). Today's best-of-N pipeline samples five candidates from the actor, then runs a separate 4B selector over the screenshot and all candidates. A head that reads the hidden states the actor computes while generating would need no second model.

<a id="selection-head-architecture"></a>

![Selection head inside the actor: (A) at inference the frozen SFT actor samples five candidates and a small head scores them from the actor's hidden states, read directly or through a top-layer scoring adapter; (B) training stage 1 fits the head on cached frozen features, stage 2 trains LoRA on the top eight layers plus the head with the prompt encoded by the unadapted actor](arm_results/methods/selection_head_workflows.svg)

[PNG](arm_results/methods/selection_head_workflows.png) / [SVG](arm_results/methods/selection_head_workflows.svg); rendered by `scripts/render_selection_head_figure.py`.

<a id="selection-head-evaluations"></a>

## Evaluations and metrics

Every selector is trained on GPT-5.5 choices from Piotr's release; no branching outcome or Luna judgment is used for training or checkpoint selection.

| Evaluation | Data | Metric | Main limitation |
| --- | --- | --- | --- |
| **Held-out continuation success (primary)** | 138 branching states with draws 0–7 for all five actions; 120 task groups | Success rate of the chosen action's held-out SFT continuations | Branching states only; not full-episode benchmark success |
| Luna agreement | Original 124 states × 4 Luna-before judgments | Share of Luna's judgments matched by the chosen action | Agreement is not correctness; Luna returns the same candidate index on all four judgments in only 49/124 states |
| Held-out GPT-5.5 agreement | 3,759 panels from 290 states in 38 held-out task groups | Share of panels whose teacher choice is matched | Imitation of the training teacher; used to choose checkpoints; Piotr ARM cannot be scored because its draw-level split likely covers these tasks |

**Definitions.**

- **State, candidate, draw.**
  - A *state* is a task, screenshot and action history before one decision.
  - The five *candidates* are fixed actor samples; duplicates are kept.
  - A *draw* executes one candidate in a separately replayed browser, then lets the frozen SFT actor continue to a terminal outcome judged by o4-mini/AgentTrek.
  - Only the root action differs between selectors.
- **Held-out continuation success.**
  - Fixed selectors (Piotr, heads, Kev-style, actor first, uniform) are scored on the mean of draws 0–7 of their chosen action.
  - The outcome-based reference chooses the candidate with the most successes on *k* draws, with exact uniform weight over ties, and is scored on the other 8 − *k*.
  - Results average all C(8, *k*) splits, and fixed selectors are scored on the same held-out draws.
  - Invalid draws count as zero. States are weighted equally.
- **Hindsight best:** the maximum held-out mean over candidates. It is optimistic: it chooses and scores on the same draws.
- **Action equivalence for agreement:** identical tool calls, with clicks within five normalized units.
- **Uncertainty:**
  - 8-draw results resample whole task groups: 10,000 draws, seed 0.
  - 124-state results resample states: 20,000 draws, seed 0, since each state is its own task.
  - All intervals are pointwise 95% percentile intervals, unadjusted for multiple comparisons.
- **Outcome-based selection is a reference, not a deployable selector:** it needs extra continuations of every candidate.

<a id="selection-head-outcomes-20261010"></a>

## Which selector picks actions that succeed? 138 states, eight draws per action

| Selector | States | Held-out success | − Uniform, pp [95%] | − Piotr ARM, pp [95%] |
| --- | ---: | ---: | --- | --- |
| Hindsight best, same held-out draws (optimistic) | 138 | 50.03% | +18.36 [+15.48, +21.24] | +15.88 [+12.27, +19.79] |
| **Outcome-based: best on 5 draws, scored on 3** | 138 | **36.99%** | **+5.32 [+3.09, +7.62]** | +2.84 [−0.24, +6.14] |
| Piotr SelectionARM (separate 4B model) | 138 | 34.15% | +2.48 [−0.14, +5.21] | — |
| Frozen-feature head, one candidate at a time | 138 | 33.70% | +2.03 [−0.26, +4.36] | −0.45 [−3.30, +2.13] |
| Actor's first sample | 138 | 33.33% | +1.67 [−1.05, +4.30] | −0.82 [−4.07, +2.27] |
| Top-8 adapter | 138 | 33.24% | +1.58 [−0.73, +3.98] | −0.91 [−3.69, +1.62] |
| Kev-style pass, step 600 | 138 | 32.79% | +1.12 [−1.38, +3.75] | −1.36 [−3.43, +0.56] |
| Kev-style pass, step 150 | 138 | 32.16% | +0.49 [−2.01, +2.99] | −1.99 [−4.50, +0.27] |
| Frozen-feature head, joint comparison | 138 | 31.88% | +0.22 [−2.63, +3.08] | −2.26 [−5.54, +0.63] |
| Frozen head, layer 27 only | 138 | 31.79% | +0.13 [−2.72, +2.94] | −2.36 [−5.93, +0.96] |
| Uniform candidate | 138 | 31.67% | — | −2.48 [−5.21, +0.14] |
| Frozen head, action tokens only | 138 | 31.34% | −0.33 [−2.74, +2.19] | −2.81 [−5.97, +0.09] |

- **Only outcome-based selection clearly beats uniform.** Its gain grows with the number of selection draws: on the same 138 states, choosing on 1, 3, 5 and 7 draws gives 34.10%, 36.01%, 36.99% and 38.00% (+2.43, +4.35, +5.32 and +6.33 pp over uniform). With one judged draw it ties Piotr ARM (−0.05 pp [−2.87, +2.89]).
- **The gap between the realistic and hindsight references is large.** Hindsight best on three draws is 50.03%. Part of that is estimation noise in three-draw scores, so it is not an achievable target.
- **Earlier post hoc winners do not replicate.** Layer 27 led the [frozen sweep](#selection-head-sweep) on the original three draws (36.56%), but it is the third-worst selector on fresh states.

<details>
<summary>Results by collection: historical, pilot and fresh states</summary>

The states come from three collections that differ in task source, decision depth and date; compare selectors within a row, not success rates across rows.

| Selector | Historical: 51 states / 51 groups | Pilot: 32 states / 29 groups | Fresh: 55 states / 45 groups |
| --- | ---: | ---: | ---: |
| Outcome-based, best on 5 draws | 29.08% | 38.50% | 43.44% |
| Piotr SelectionARM | 25.74% | 33.98% | 42.05% |
| Frozen-feature head, one at a time | 26.23% | 35.94% | 39.32% |
| Top-8 adapter | 26.96% | 34.77% | 38.18% |
| Kev-style pass, step 600 | 24.75% | 33.59% | 39.77% |
| Kev-style pass, step 150 | 23.04% | 33.59% | 39.77% |
| Frozen head, layer 27 only | 28.68% | 31.64% | 34.77% |
| Actor's first sample | 21.32% | 37.11% | 42.27% |
| Uniform candidate | 23.38% | 33.67% | 38.18% |

- **Outcome-based selection gains about 5 pp over uniform in every collection:**
  - historical +5.70 [+1.93, +9.90];
  - pilot +4.83 [+0.47, +9.27];
  - fresh +5.26 [+2.14, +8.82].
- **The actor's first sample is weak only in the historical states.** It scores 21.32% there, versus 37.11% and 42.27% in the pilot and fresh states. The historical states were admitted under depth quotas; earlier analyses found the first sample weaker at later decisions. The reason has not been isolated.
- **Layer 27 reverses on fresh states:** −3.41 pp [−8.59, +1.53] versus uniform, and −7.27 pp [−14.39, −1.25] versus Piotr.

*Historical* states are the 51 of the original 124 whose top-up reached eight draws. *Pilot* states are the 32-state WebVoyager-source training-pool pilot, using draws 0–7 of its 32. *Fresh* states are the 51 scored states of the fresh 64-state batch plus 4 states of its successor batch. The state counts are the states with selector picks; collection continues ([data](#selection-head-outcome-data-20261010)).

</details>

<a id="selection-head-teacher-agreement"></a>

## Agreement with teachers: the original 124 states

| Selector | Luna agreement, 124 states | − Piotr, pp [95%] | Held-out GPT-5.5 agreement, 3,759 panels |
| --- | ---: | --- | ---: |
| Luna judgment versus another Luna judgment | 80.2% | — | — |
| Piotr SelectionARM | 73.2% | — | not measurable |
| Kev-style pass, step 150 | 69.6% | −3.6 [−10.3, +2.8] | 61.5% |
| Kev-style pass, step 600 | 66.1% | — | **67.1%** |
| Frozen-feature head, one candidate at a time | 61.7% | −11.5 [−18.6, −4.8] | 65.4% |
| Frozen-feature head, joint comparison | 60.7% | −12.5 [−19.6, −5.2] | 65.5% |
| Top-8 adapter, best checkpoint | 58.7% | −14.5 [−22.0, −7.5] | 65.5% |
| Joint top layers, pointer alone, latest checkpoint | 57.5% | −15.7 [−24.4, −7.5] | 53.6% |
| Actor's first sample | 56.2% | — | 44.5% |
| Majority action among the five samples | 50.0% | — | — |
| Uniform candidate | 46.9% | — | 44.7% |

- **The ranking is stable; the gaps are not significant.**
  - Piotr > Kev-style step 150 > frozen head > actor first > majority > uniform holds in every subset checked:
    - prompt-verified states (114);
    - states whose tasks are absent from Piotr's release (105);
    - first decisions (33) and later decisions (91);
    - each of Luna's four judgments taken alone.
  - Kev-style step 150 minus Piotr ranges from −1.6 to −5.6 pp across these checks; every interval includes zero.
- **Longer Kev-style training raised imitation and lowered transfer.** From step 150 to 600, held-out GPT-5.5 agreement rose by 5.6 pp, while Luna agreement fell by 3.5 pp.
- **Neither agreement metric predicts held-out continuation success** ([above](#selection-head-outcomes-20261010)). The majority action among the five samples is worse than uniform on outcomes too: −1.48 pp [−3.84, +0.91] on the 124 states with all draws.

<details>
<summary>Robustness of the 124-state continuation results to added draws</summary>

The original 124-state results used three draws per action. The extra-two pass and the historical top-up later added draws for 59 of these states, up to eight per action. The saved draws reproduce the original three-draw outcomes in 619 of 620 candidate cells. The one exception is an unreleased invalid draw that the original evaluation counted as zero, and this analysis keeps that rule.

| Selector | Original 3 draws: 124 states | All draws: 124 states, 5.2 per action | Fresh draws 3–7 only: 59 states |
| --- | ---: | ---: | ---: |
| Piotr SelectionARM | 35.75% | 36.98% | 28.64% |
| Frozen-feature head, one at a time | 34.68% | 35.95% | 30.00% |
| Kev-style pass, step 150 | 34.14% | 34.80% | 26.10% |
| Uniform | 32.42% | 33.11% | 26.07% |
| Actor's first sample | 29.84% | 32.21% | 25.59% |
| Hindsight best, same draws | 50.00% | 49.13% | 41.69% |

Piotr minus uniform is +3.87 pp [+0.81, +7.05] with all draws, and +2.58 pp [−2.17, +7.46] on fresh draws only. The fresh-only states are those whose extensions finished, not a random subset. Valid-only variants lead to the same conclusions.

</details>

<a id="selection-head-outcome-data-20261010"></a>

## Is there enough outcome-labelled data to train a selector?

**Not yet for training an actor-scale selector; possibly for adapting an existing one.** Outcome labels are richer per state but cover far fewer states and task groups than the teacher data.

| Data | States | Task groups | Labelled units | Label per state |
| --- | ---: | ---: | ---: | --- |
| GPT-5.5 teacher panels used for every selector here (training split) | 2,692 | 353 | 35,916 panels | One teacher pick per panel; about 13 panels per state |
| Branching states with eight draws for all five actions | 188 | 151 | 9,720 continuation outcomes | Five success rates estimated from eight draws each |
| — of which the five actions' success rates differ at all | 133 | — | — | — |
| — of which the best action beats the panel mean by ≥ 25 pp | 53 | — | — | — |

- **Label noise.** An eight-draw success rate has a standard error of about ±0.15 at the observed rates.
  - 40 of the 188 states fail on every draw of every action, and 15 succeed on every draw; these carry no ranking signal.
  - Averaged over states, the best action beats the panel mean by 15.0 pp.
- **Scale.** 188 states are about 7% of the teacher-labelled states and 43% of its task groups.
  - Training the Kev-style pass took about 1,200 states to reach its step-150 teacher agreement.
  - A held-out evaluation also needs its own states: at 138 states, 95% intervals on selector differences span about ±2–3 pp.
- **Feasible now:**
  - outcome-supervised adaptation of existing selectors, with five-fold task-group cross-validation on the 188 states;
  - pairwise or soft-target losses on the eight-draw success rates;
  - starting from the teacher-trained frozen head or Kev-style checkpoint.
- **Needs more data:** training an actor-based selector from outcomes alone, or a powered comparison with Piotr ARM.
  - Current collection admits about six to seven fresh states per hour per job.
  - The [collection priority](ARM_FORMULATIONS.md) now favors tasks from Piotr's ARM training split. Those states would also allow outcome and teacher labels to be compared on the same states.

<details>
<summary>Eight-draw coverage by collection</summary>

| Collection | States | Task groups | Outcomes | Actions differ | All actions fail | All actions succeed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical (original 124, topped up) | 51 | 51 | 2,040 | 30 | 17 | 4 |
| Pilot (32 states; draws 0–31 counted) | 32 | 29 | 3,480 | 25 | 6 | 1 |
| Fresh 64-state batch | 64 | 52 | 2,560 | 46 | 13 | 5 |
| Fresh successor batch, in progress | 41 | 32 | 1,640 | 32 | 4 | 5 |
| **Total** | **188** | **151** | **9,720** | **133** | **40** | **15** |

These are saved, released records read on 2026-10-10 evening; native-invalid records are included and count as failures. Pilot outcome counts include all 32 draws; the 8-draw criterion uses draws 0–7. 261 states have at least one draw for every action (10,895 outcomes).

</details>

<a id="selection-head-sweep"></a>

## Frozen-feature sweep: no layer or pooling closes the gap

All nine configurations use the one-at-a-time head, train for ten epochs and choose the checkpoint by GPT-5.5 validation agreement. Luna agreement and continuation success (original three draws, 124 states) are reported only.

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
- **High continuation-success rows are post hoc.** Layer 27 (36.56%), layer 18 and action-only (36.02%) are the best of nine noisy estimates. None differs from Piotr ARM: layer 27 minus Piotr is +0.81 pp [−2.96, +4.84]. The configuration chosen by the preset rule, the smaller head, scores 35.22%. On the [138-state held-out evaluation](#selection-head-outcomes-20261010), layer 27 falls to 31.79% and the action-token head to 31.34%, both at uniform.

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
| Joint top layers | Layers 1–28 reused per candidate; layers 29–36 once over [candidates, `<decide>`], `<decide>` reading every candidate | LoRA on top 8 layers, pointer, head | One-at-a-time head + gated pointer | Exact | No gain; pointer alone 57.5% Luna agreement |
| Kev-style pass | Prompt, screenshot, candidates and `<decide>` re-encoded through all 36 layers; panels packed like Kev's questions | LoRA on all 36 layers, pointer, `<decide>` embedding | Pointer only | None | Luna agreement 69.6% (step 150), 66.1% (step 600); held-out success at uniform |

Differences from Kev itself: the input is the actor's own prompt with the screenshot rather than a text page description; there is no question instruction; candidates are the actor's generated responses; Kev-27B cannot isolate options (it is a hybrid DeltaNet model), so its options see earlier ones; and Kev is trained on general decision datasets with a fitted temperature, not on GPT-5.5 web choices. Unit tests on a small Qwen3-VL confirm exact warm start (joint), candidate isolation at generation positions, order and packing invariance, and gradients reaching only the trained parts.

**Joint top layers: the comparison pointer learns, but stays below the head.** In job `353389`, the pointer's own validation-probe agreement rose from chance to 56.8%; the head alone scores 65%. The gate weighting the pointer stayed near zero, so the combined score never beat the step-0 checkpoint. The selected checkpoint is therefore the frozen head: held-out agreement 65.4%, Luna agreement 60.9%, continuation success 34.95%.

| Step | Combined probe agreement | Pointer alone | Gate |
| ---: | ---: | ---: | ---: |
| 0 | 64.9% | 46.5% | 0 |
| 150 | 64.0% | 55.8% | 0.012 |
| 300 | 63.9% | 56.8% | −0.005 |

The probe has 250 states; the run trained 373 optimizer steps over 2,987 states. Scored separately (job `353554`), its latest checkpoint is weaker than the frozen head. The pointer alone reaches 53.6% held-out agreement, 57.5% Luna agreement and 31.72% continuation success; the combined score reaches 64.9%, 56.9% and 32.26%. An earlier attempt (`353319`) initialized the gate at zero without a pointer loss, which gave the pointer no gradient; it was stopped after 150 steps with the gate at 0.0021. The trained pointer's weights are in the saved latest checkpoint but have not been scored on the branching states.

**Kev-style pass: learns from scratch and keeps improving on its teacher.** Job `353512` started the pointer at chance and reached 61.0% on the 250-state validation probe after 150 optimizer steps, about 1,200 training states. Job `353914` resumed from its step-244 checkpoint with the same optimizer state and trained to step 714. The best probe checkpoint, step 600, was scored on full validation and on the branching states. Each training state runs the prompt and four panels of five candidates with gradients through all 36 layers: about 4 s per state on one H200 with gradient checkpointing, using about 57 GB.

| Step | Probe agreement, 3,283 panels | Held-out validation, 3,759 panels | Luna agreement, 124 states | Held-out continuation success, 138 states |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 45.5% | — | — | — |
| 150 | 61.0% | 61.5% | 69.6% | 32.16% |
| 300 | 62.3% | — | — | — |
| 450 | 64.5% | — | — | — |
| 600 | 66.6% | **67.1%** | 66.1% | 32.79% |

The probe was still rising at step 600. Training loss stopped falling after step 300. More teacher imitation is unlikely to help action quality, given the flat continuation success; the step-714 latest checkpoint is saved and unscored.

<a id="selection-head-next"></a>

## What would settle the comparison

- **Outcome supervision.** Adapt the frozen head or the Kev-style checkpoint to eight-draw success rates. Evaluate by task-group cross-validation on held-out draws, with the outcome-based reference and uniform as anchors ([data available](#selection-head-outcome-data-20261010)).
- **More fresh states.** Each additional 100 states with eight draws narrows selector-difference intervals and adds training signal. The [collection plan](ARM_FORMULATIONS.md) now prioritizes tasks from Piotr's ARM training split.
- **Benchmark impact.** Only a matched full300 run (SFT alone, SFT + Piotr ARM, SFT + an actor-based selector) shows whether any selector changes task success. On the [inference benchmark](ARM_INFERENCE_SCALING.md), Piotr ARM adds +5.67 ± 1.15 pp over three runs despite its weak branching-state gain here; the two evaluations measure different things.
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
<summary>Scoring the saved selectors on new branching states, and the reproduction check</summary>

- **Panels.** `scripts/build_selection_head_branch_eval.py` builds a panel per accepted state from each collection's anchors, using the same prompt-hash reconstruction: 86 of the 87 new states verify; the remaining fresh state is kept as an unverified replay-feedback case, like 10 of the original 124. `--expect-trials 0` admits any complete panel; its defaults rebuild the original 124-state files byte-identically.
- **Selectors.**
  - Piotr ARM: `score_selection_arm_offline.py`.
  - Frozen heads: `train_selection_head.py --load-head`, which loads the saved `head.pt` instead of retraining.
  - Kev-style and adapter checkpoints: `train_selection_adapter.py --eval-only --eval-checkpoint best --final-val-states 0`.
- **Reproduction check.** Job `354319` first re-extracted features for the original 124 states and re-scored every selector there. All seven reproduced their saved picks on 124/124 states: Piotr, the four frozen heads, Kev-style step 150 and the top-8 adapter. The job would have stopped on more than two changed picks per selector.
- **Outcomes.** Draws come read-only from each collection's ledger of released records, including the historical extra-two and top-up passes. No duplicate outcome slots conflict. Analysis scripts and outputs are archived with the runtime artifacts (`outcome-analysis-20261010/`). [Aggregate estimates](arm_results/rl_integration/selection-outcomes-20261010.json).

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
| `353512` | 1 H200, 2 CPUs, 120 GiB, 3 h approved | Kev-style pass | 2 h 59 m 25 s | Completed |
| `353554` | 1 H200, 2 CPUs, 120 GiB, 45 m approved | Evaluation-only: joint latest checkpoint (Kev-style step skipped, already complete) | 15 m 24 s | Completed |
| `353914` | 1 H200, 2 CPUs, 120 GiB, 6 h approved | Kev-style continuation from step 244 to 714; best-checkpoint branch and full validation | 5 h 43 m 29 s | Completed |
| `354318`, `354319` | 1 H200, 2 CPUs, 120 GiB, 1.5 h approved for both | Reproduction check on 124 states; all saved selectors on 87 new states | 12 s (import failure, no GPU work), 10 m 00 s | Failed, completed |
| `354320` | 1 H200, 2 CPUs, 120 GiB, 30 m approved | Kev-style step 600 on the 87 new states | 1 m 29 s | Completed |

CPU-only jobs `353141` and `353145`, and the 8-CPU GPU job `353148`, were cancelled while pending because no node had enough idle CPUs. They consumed no allocation. No API calls or browsers were used.

Code is on branch `arm-selection-head`: `openwebrl/selection_head.py`, `selection_head_features.py`, `selection_adapter.py`, `selection_joint.py`, `selection_kev.py` and the scripts they call. Runtime artifacts are under `openwebrl-runtime/selection-head-20261008/`. [Aggregate estimates and accounting](arm_results/rl_integration/selection-head-20261009.json) · [Outcome evaluation aggregate](arm_results/rl_integration/selection-outcomes-20261010.json).

</details>
