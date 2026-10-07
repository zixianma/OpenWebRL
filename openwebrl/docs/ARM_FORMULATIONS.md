# ARM formulations: before versus after execution

**Execution evidence changes the teacher's choices and makes them more consistent. We have not demonstrated better downstream success.** Neither study trained a critic.

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

| Selection rule | Paired states | Continuation success | Repeat disagreement |
| --- | ---: | ---: | ---: |
| Random choice: 20% per candidate | 70 | 41.43% | — |
| Before teacher | 70 | 44.29% | 30.48% |
| After teacher | 70 | 44.76% | 18.25% |

| Paired comparison | Gain, pp | 95% interval, pp |
| --- | ---: | --- |
| Before teacher − random | +2.86 | [−1.08, +6.92] |
| After teacher − before teacher | +0.48 | [−2.38, +3.65] |

**More consistent choices; no demonstrated success improvement.** Rates average repeated selections, not integer success counts. Invalid continuations count as zero; intervals resample whole states.

### What does “random candidate” mean?

Choose one of the **five sampled action entries** with equal probability, ignoring teacher scores. Duplicate actions retain their separate entries.

We already executed all five, three times each. Averaging those **15 outcomes** estimates random selection without another browser run. This randomizes **one action at the root state**, then continues with ordinary SFT.

### A random control for the pre-action critic

For a full-episode control, repeat this at **every step**:

**Generate five actions → randomly pick one → execute → repeat.**

Compare it with SFT alone and SFT + the pre-action SelectionARM, using the same tasks, actor, decoding, browser, step cap and judge. No execution evidence goes to the selector.

If candidates are independent, identically distributed actor samples, a random pick has the same action distribution as one actor sample. The full-episode run tests that expectation in the actual five-candidate pipeline.

**Prepared:** 300 tasks, 30 steps, random choice among five candidates at every step; compare with today's fresh SFT/Piotr pair. Eight offline tests passed. Scientific scope approved; exact additional compute/judge caps awaiting approval. [Run plan](ARM_INFERENCE.md#selectionarm-random5-sameday30-20261007).

<details>
<summary>Protocol and limitations</summary>

Fresh SFT prefixes were replayed into isolated browsers; all 15 observable-state checks had to pass. These were not restored Piotr browser snapshots, and hidden browser/server state need not match.

Collection reached **74/100 states and 1,110/1,500 continuations**; 70 states had complete teacher panels. Only 37 complete panels had five distinct actions. There were 333 reconstruction rejections and 110 invalid continuations. These early, replayable states are not a full-episode benchmark.

Actor: official SFT, T=1, p=.95, top-k off, 4,096 tokens. Three before judgments; three after judgments per execution repetition; separate order controls. Canonical o4-mini/AgentTrek success allows partial progress.

Saved-action labels use three progress probabilities, `evidence_sufficient`, `effect`, and `rationale`; score = P(progress) − P(regression). Branching uses only the selected index.

All attempts accounted for: 31.94 H200-hours; $3.32 teacher; $5.82 judge. Dataset partial, accounting final.

</details>

## Examples

Local cluster directories; each contains separate **before/** and **after/** inputs and outputs. Branch **outcomes/** were never teacher inputs.

| Study | Example directories |
| --- | --- |
| Saved action | [Changed label](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-formulation-examples-20261007/progress-changed/) · [Unchanged label](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-formulation-examples-20261007/progress-unchanged/) |
| Five-action selection | [After scores higher](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-formulation-examples-20261007/selection-after-higher/) · [After scores lower](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-formulation-examples-20261007/selection-after-lower/) |

Illustrative examples, not an accuracy sample. Filesystem access is required; raw data remain private.

[Aggregate results](arm_results/rl_integration/continuation-branches-final-20261006.json) · [Random-control comparison](arm_results/rl_integration/continuation-random-control-20261007.json) · [Full protocol and repair history](ARM_INTEGRATION_PLAN.md#arm-continuation-branches-20261006).
