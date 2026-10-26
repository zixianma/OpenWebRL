# Final ARM teacher design: candidates, K = 3 lookahead labels and selection-head training data

This page records how the teacher labels used by the selection head are produced, what the teacher sees, and how the
labels become training targets. Results are in [ARM_SELECTION_HEAD.md](ARM_SELECTION_HEAD.md); collection is in
[ARM_BRANCH_STATES_COLLECTION.md](ARM_BRANCH_STATES_COLLECTION.md).

## Summary

- **Teacher:** `gpt-6-luna`, reasoning effort high, **K = 3 lookahead**. It sees each action executed once plus the
  same SFT actor's next 3 steps after it. It never sees a success judgment.
- **What is labelled:** at each state, every *distinct action* the SFT checkpoint proposed that was also executed.
  That is 2–5 actions per state (median 3), not always 5.
- **Label:** 2 calls per state, each with the actions in a different random order. The training target is the vote
  share per action, for example `[0, 1, 0]`, or `[0.5, 0.5, 0]` when the calls disagree.
- **Training data:** 1,738 states (1,390 train / 348 test, no shared task), published as the private HF dataset
  `zixianma/openwebrl-selection-head-data`. Labelling cost USD 7.14 for 3,454 calls.

## Terms

| Term | Meaning |
| --- | --- |
| **Proposal** | one sampled actor output at a state: reasoning plus tool call |
| **Distinct action** | proposals with the same action type and target are one action: clicks on the same element (identified by its visible text, or a 100-unit grid cell when no element is under the point), the same typed text, the same scroll direction, the same URL, or `done`. Multi-call bundles compare call by call |
| **Action class** | the tool name alone (click, type, scroll, …); used only by the `strict_usable` filter |
| **Executed candidate** | one of the 5 actions executed from the state (8 continuation draws each) to measure outcomes and to give the teacher its evidence |

Several proposals usually perform the same action: 1,649 of the 1,738 states have at least one repeated action. In the
first training state the 5 proposals map to actions `0, 1, 0, 0, 1`: 3 samples click one option and 2 click another,
so the state has 2 distinct actions.

## Which actions the teacher sees

1. **Sampling.** At each state the collector samples 16 proposals: 8 from OpenWebRL-4B-SFT and 8 from the RL
   checkpoint. Older SFT-only states have 5 SFT samples.
2. **Executed candidates.** The first 5 samples with different distinct actions are executed. If fewer than 5 distinct
   actions exist, the remaining slots take the next parseable samples. Each candidate is credited to the checkpoint of
   the sample it was taken from, so an action both checkpoints proposed is credited to only one of them.
3. **Teacher set.** The teacher sees one executed copy of **every distinct action the SFT checkpoint proposed in any of
   its samples**, whatever the credited proposer. These are exactly the actions the SFT head can choose at deployment.
   Two kinds of executed candidate are left out:
   - actions only the RL checkpoint proposed;
   - repeats of an action already shown.

<details><summary>Distribution of actions shown to the teacher</summary>

| Actions shown | Train states | Test states |
| ---: | ---: | ---: |
| 2 | 593 | 150 |
| 3 | 484 | 120 |
| 4 | 242 | 69 |
| 5 | 71 | 9 |
| **Total** | **1,390** | **348** |

In the published branch-state dataset (`openwebrl-branch-states`, column `teacher_k3_shown`), the number shown exceeds
the SFT-credited distinct actions in 190 of 1,644 states. The difference is always an executed action credited to RL
that SFT also proposed. Example: in state `4b08b172…`, RL sample #3 and SFT sample #14 both click the search box, type
"machine translation tutorial" and press Enter. The candidate is credited to RL, but the teacher still sees it.
</details>

## What the teacher sees (each call)

| Input | Shown |
| --- | --- |
| Task text | yes |
| Current screenshot; page URL, title and tabs | yes |
| Page text (visible text, interactive elements) | no |
| Earlier steps of the episode | no |
| Each action's representative proposal | **full text: reasoning and tool call** |
| Result of executing the action once (continuation draw 0) | screenshot right after it, and the tool's receipt |
| The same actor's next 3 steps after the action | each as the **action only (reasoning removed)** with the screenshot after it; whether the episode ended (final answer hidden) |
| Success judgments, rewards, continuation draws 1–7 | no |

The instruction asks for the action "most likely to lead to completion of the original task when followed by the same
fixed browser actor". It tells the teacher to judge the grounded command, to treat candidate reasoning and web content
as untrusted evidence, and to treat the lookahead as one sample of the actor's behaviour, not a success judgment. The
output is a JSON index `{"selection": N}`.

Representation ablations (screenshots-only versus page text, with or without history) changed little; lookahead depth
did. The detailed results are in [ARM_SELECTION_HEAD.md](ARM_SELECTION_HEAD.md#label-architecture-and-data-ablations-on-sft-proposed-actions-2026-10-21).

## From picks to training targets

| | Train | Test |
| --- | ---: | ---: |
| Both calls agree | 1,149 | 293 |
| Calls disagree (target split 0.5 / 0.5) | 232 | 53 |
| No valid pick (state unlabelled) | 9 | 2 |

The target is the vote share per action, aligned with `actions`. Training uses only labelled states.

## How the head trains on it

- **Inputs:** frozen OpenWebRL-4B-SFT features of each candidate's text (layers 18/27/36, pooled over its reasoning,
  its action and its last token), plus a state feature.
- **Model:** a 22.6 M-parameter transformer over the candidate set (2 layers, width 512) that scores each slot.
  It trains with soft cross-entropy for 20 epochs (AdamW, learning rate 3e-4, batch 64) and keeps the last epoch.
- **Default "distinct" view:** one slot per distinct action (2–5 slots; unused slots are masked), with the target as
  above.
- **5-slot "natural" view (as at deployment):** 5 proposals sampled per epoch, so actions may repeat. An action's target
  share is split evenly over its copies.
- **Offline evaluation:** on the 348 test states, the success rate of the picked action on held-out continuation draws
  3–7, in both the distinct and the 5-proposal views.

<details><summary>Per-set labels (2026-10-25)</summary>

To test Piotr-style training on several 5-proposal sets per state, each training state's SFT proposals were shuffled
and cut into disjoint sets of 5, keeping sets with at least 2 distinct actions. That gives 1,627 sets on 1,302 states.

- 1,313 sets contain all of their state's actions and reuse the state's label, since the request is identical.
- 243 sets contain fewer actions and were labelled anew with the same frozen K = 3 setup: 476 calls, USD 0.77.

Only 131 states yield more than one set, so the existing proposals add few extra sets. The comparison of distinct,
natural, all-set and one-set training is in [ARM_SELECTION_HEAD.md](ARM_SELECTION_HEAD.md) once job 387510 finishes.
Labels are at `selection-head-20261008/sft-actions-20261021/labels/k3-sets/`.
</details>

<details><summary>Data and code locations</summary>

| Item | Location |
| --- | --- |
| Head training table (HF, private) | `zixianma/openwebrl-selection-head-data`: one row per state with `proposals`, `actions`, outcome counts, `teacher_k3_picks`, `teacher_k3_target` |
| Branch-state table (HF, public) | `zixianma/openwebrl-branch-states`, `states` columns `teacher_k3_shown` / `teacher_k3_picks` / `teacher_k3_best` (indices into the 5 executed candidates) |
| Labels | runtime `selection-head-20261008/sft-actions-20261021/labels/k3/` (`plan.json`, `judgments/`, `approval.json`) |
| Labeller | `scripts/run_branch_lookahead_labels.py` (frozen copy in the label root's `source/`); teacher request `openwebrl/arm_branch_teacher.py` |
| Action identity | `openwebrl/arm_branch_protocol.py` (`_target`, `action_key`, `diverse_candidates`) |
| Trainer | `scripts/train_head_round.py` |
</details>
