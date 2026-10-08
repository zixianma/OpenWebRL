# ARM formulations: before versus after execution

**Early decisions: execution evidence makes choices more consistent, without a demonstrated success gain. Later decisions remain under-sampled.** Neither study trained a critic.

| Study | Before teacher sees | After teacher additionally sees | Teacher output |
| --- | --- | --- | --- |
| Saved Piotr action | Task, history, screenshot, one recorded action | That action's execution feedback and screenshot | Progress probabilities + evidence/rationale |
| Five-action selection | Task, history, screenshot, five candidate actions | Immediate execution evidence for all five | Only `{"selection": 1..5}` |

## 1. Does execution change a progress judgment?

Same saved action, Luna-high teacher, different evidence.

| Comparison | Examples | Changed labels | Change rate |
| --- | ---: | ---: | ---: |
| Before → after, full set | 200 | 32 | 16% |
| Before → after, repeat subset | 100 | 18 | 18% |
| Before → before repeat | 100 | 13 | 13% |
| After → after repeat | 100 | 7 | 7% |

On the repeat subset, the change exceeds before-repeat noise by **5 pp**, interval **[−3, +14]**. Changed labels do not establish better accuracy. Only the executed action has evidence. [Details](ARM_RESULTS.md#arm-teacher-evidence-primary200-20261006).

## 2. Does execution improve action selection?

```mermaid
flowchart LR
    A["5 actions × 3 continuations"] --> Y["One shared pool: 15 outcomes"]
    Y --> B["Score saved before picks"]
    Y --> C["Score saved after picks"]
    Y --> R["Average all: random score"]
```

**Before, after, and random score the same continuation data; there is no third rollout set.** Each of five candidates has three continuations with the same SFT actor. Teachers choose without seeing continuation outcomes; after sees immediate execution only. Both teachers are Luna-high and output only an index.

<a id="arm-continuation-branches-results-20261007"></a>

| Candidate decisions | Paired states | Random choice success | Before success | After success | After − before, pp | 95% interval, pp |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Early: 1–3 | 70 | 41.43% | 44.29% | 44.76% | +0.48 | [−2.38, +3.65] |
| Later: 4–15 | 12 | 28.89% | 28.70% | 30.56% | +1.85 | [0.00, +5.56] |

| Candidate decisions | Before-repeat disagreement | After-repeat disagreement |
| --- | ---: | ---: |
| Early: 1–3 | 30.48% | 18.25% |
| Later: 4–15 | 33.33% | 42.59% |

**No clear success gain in either sample.** Greater repeatability appears in early states only. The later gain comes from one of 12 states; its interval touches zero. Depth groups differ in replayability and remaining actions; their success rates are not a controlled depth comparison.

Rates average repeated selections; invalid continuations count as zero. Intervals resample whole states. [Aggregate results](arm_results/rl_integration/continuation-depth-balanced-partial-20261007/aggregate.json).

<details>
<summary>Success and repeat-control plots</summary>

Partial coverage: 86/149 states. Plot n denotes paired states; error bars are 95% state-bootstrap intervals. The deepest group's zero-width success interval reflects three all-failure states, not population certainty.

![Continuation success by decision group](arm_results/rl_integration/continuation-depth-balanced-partial-20261007/paired-success.png)
![Teacher choice disagreement by decision group](arm_results/rl_integration/continuation-depth-balanced-partial-20261007/repeat-controls.png)

</details>

### Which states were tested?

| Candidate decision | Completed states | Share | Completed / discovery attempts |
| --- | ---: | ---: | ---: |
| 1 | 33 | 44.6% | 21.2% |
| 2 | 23 | 31.1% | 13.7% |
| 3 | 18 | 24.3% | 12.2% |

**Early states only:** 30 site families, 57 distinct screenshots, and 37 states with five distinct actions. Eighteen states show blank/access/error pages; their outcomes remain included. Completion rates include discovery and replay attrition.

**Closed partial: 86/149 states and 1,290/2,235 continuations.** The extension added 12 later states. Strict replay and the remaining-time guard limited coverage; every attempt is preserved. The 30-action budget includes the prefix.

| Later decisions | Completed states | Target states |
| --- | ---: | ---: |
| 4–7 | 5 | 25 |
| 8–11 | 4 | 25 |
| 12–15 | 3 | 25 |

[Extension and exact quotas](ARM_INTEGRATION_PLAN.md#arm-continuation-finish100-20261007) · [Current coverage](arm_results/rl_integration/continuation-depth-balanced-extension-20261007.json) · [Early coverage](arm_results/rl_integration/continuation-state-diversity-20261007/aggregate.json).

<details>
<summary>State coverage and diversity plots</summary>

Partial coverage: 86/149 states. Bubble size represents the number of states.

![Completed states and remaining browser operations](arm_results/rl_integration/continuation-depth-balanced-partial-20261007/depth-coverage.png)

![Early state depth, site families, and distinct candidates; bar labels are counts](arm_results/rl_integration/continuation-state-diversity-20261007/state-diversity.png)

</details>

### What does “random candidate” mean?

Choose one of the **five sampled action entries** with equal probability, ignoring teacher scores. Duplicate actions retain their separate entries.

We already executed all five, three times each. Averaging those **15 outcomes** estimates random selection without another browser run. This randomizes **one action at the root state**, then continues with ordinary SFT.

### A random control for the pre-action critic

For a full-episode control, repeat this at **every step**:

**Generate five actions → randomly pick one → execute → repeat.**

Compare it with SFT alone and SFT + the pre-action SelectionARM, using the same tasks, actor, decoding, browser, step cap and judge. No execution evidence goes to the selector.

If candidates are independent, identically distributed actor samples, a random pick has the same action distribution as one actor sample. The full-episode run tests that expectation in the actual five-candidate pipeline.

**Complete:** random selection at every step succeeds on **100/300 tasks (33.33%)**, versus SFT **99/300 (33.00%)** and Piotr SelectionARM **120/300 (40.00%)**. Random − SFT: **+0.33 pp [−4.67, +5.67]**; Piotr − random: **+6.67 pp [+1.00, +12.33]**. This supports useful learned selection in this run; random and SFT equivalence is not established. [Full-episode comparison](ARM_INFERENCE_SCALING.md#random5-sameday-results-20261007).

<details>
<summary>Protocol and limitations</summary>

Fresh SFT prefixes were replayed into isolated browsers; all 15 observable-state checks had to pass. These were not restored Piotr browser snapshots, and hidden browser/server state need not match.

Collection reached **86/149 states and 1,290/2,235 continuations**; all 86 have saved teacher panels, and 82 have complete valid ordinary judgments for pairing. The four exclusions are early states. All 111 invalid continuations remain failures. A previously censored state retains four finished and 11 interrupted continuations outside this endpoint. Replayable training states are not a full-episode benchmark.

Actor: official SFT, T=1, p=.95, top-k off, 4,096 tokens. Three before judgments; three after judgments per execution repetition; separate order controls. Canonical o4-mini/AgentTrek success allows partial progress.

Saved-action labels use three progress probabilities, `evidence_sufficient`, `effect`, and `rationale`; score = P(progress) − P(regression). Branching uses only the selected index.

All-attempt accounting is closed: 44.46 H200-hours, $4.32 teacher, $6.53 judge. The extension retains 3,135 approved seconds, insufficient for the unchanged cohort admission and shutdown requirements. No budget was added or transferred.

</details>

## Examples

Local cluster directories; each contains separate **before/** and **after/** inputs and outputs. Branch **outcomes/** were never teacher inputs.

| Study | Example directories |
| --- | --- |
| Saved action | [Changed label](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-formulation-examples-20261007/progress-changed/) · [Unchanged label](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-formulation-examples-20261007/progress-unchanged/) |
| Five-action selection | [After scores higher](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-formulation-examples-20261007/selection-after-higher/) · [After scores lower](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-formulation-examples-20261007/selection-after-lower/) |

Illustrative examples, not an accuracy sample. Filesystem access is required; raw data remain private.

[Aggregate results](arm_results/rl_integration/continuation-branches-final-20261006.json) · [Random-control comparison](arm_results/rl_integration/continuation-random-control-20261007.json) · [Full protocol and repair history](ARM_INTEGRATION_PLAN.md#arm-continuation-branches-20261006).
