# ARM formulations: before versus after execution

**Continuation-informed selection reaches 41.52%, versus 35.74% for the actor’s first sample, on 97 accepted replayable states. Execution evidence has not demonstrated a gain over the before-execution teacher.** No critic was trained; these are conditional continuation results, not benchmark pass@1.

## Does execution improve action selection?

<a id="arm-continuation-branches-results-20261007"></a>

At the October 8 10:07 UTC cutoff, 97 states have all five actions × three SFT continuations: **1,455 outcomes**, including 120 invalid outcomes scored zero. Each branch comparison reuses its stated cohort’s outcomes. States were accepted through strict replay checks and are not a random sample of tasks or decision points.

### Can observed continuation outcomes improve the root-action choice?

| Root-action choice | States | Continuation success | 95% state-bootstrap interval |
| --- | ---: | ---: | --- |
| Actor’s first sample | 97 | 35.74% | [27.15%, 43.99%] |
| Uniform candidate | 97 | 37.25% | [29.69%, 44.68%] |
| Select on two continuations; score held-out third | 97 | 41.52% | [33.16%, 49.74%] |
| Same-data hindsight action oracle | 97 | 53.95% | [45.36%, 62.20%] |

Leave-one-out selection improves over the actor’s first sample by **5.78 percentage points** (95% interval **[+1.06, +10.77]**) and over uniform selection by **4.27 points [+0.95, +7.69]**. For each state, select using two continuations per action, score the held-out third, and average all three overlapping folds. Ties use their exact uniform expectation without consulting the held-out outcome. Candidate 0 is the first actor sample; uniform selection averages the five sampled entries, including duplicates.

The hindsight oracle chooses and scores the action with the best mean over the **same three outcomes**, so its 53.95% is optimistic and not held-out performance. It does not select a successful trajectory from all 15. The LOO result uses additional outcome information, so it does not measure a trained selector’s performance.

### Does immediate execution evidence help the teacher?

Four historical states lack valid ordinary teacher panels; exclude them only from this comparison. All rows below use the **same 93 states and 1,395 outcomes**. Both Luna-high teachers output only a candidate index and never see continuation outcomes; after-execution additionally sees each candidate’s immediate execution evidence.

| Root-action choice | Matched states | Continuation success | 95% state-bootstrap interval |
| --- | ---: | ---: | --- |
| Actor’s first sample | 93 | 36.56% | [27.96%, 45.52%] |
| Uniform candidate | 93 | 38.28% | [30.61%, 46.16%] |
| Luna before execution | 93 | 41.58% | [32.97%, 50.54%] |
| Luna after immediate execution | 93 | 42.05% | [33.33%, 51.02%] |
| Select on two; score held-out third | 93 | 42.47% | [34.01%, 51.09%] |
| Same-data hindsight action oracle | 93 | 55.20% | [46.59%, 63.80%] |

After minus before is **+0.48 points [−1.79, +2.87]**. LOO minus after is **+0.41 points [−3.60, +4.49]**. Neither contrast establishes a gain. The teachers average repeated selections; this does not create additional rollout data. [Aggregate estimates, denominators, uncertainty and audit hashes](arm_results/rl_integration/continuation-fixed97-20261008.json).

<details>
<summary>Repeat controls, oracle-action agreement and ties</summary>

| Choice disagreement | Matched states | Rate |
| --- | ---: | ---: |
| Before versus before repeat | 93 | 31.18% |
| After versus after repeat | 93 | 21.86% |
| Before versus after | 93 | 44.96% |

Cross-condition disagreement exceeds the mean within-condition disagreement by 18.44 points [12.39, 24.91]. More consistent choices do not by themselves establish better success.

An oracle-action match means selecting any candidate tied for the highest observed success across its three continuations. This uses the same outcome data and is a descriptive, noisy label.

| Oracle subset | Matched states | Before agreement | After agreement | Actor-first agreement | Uniform agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Any tied best action | 93 | 70.61% | 70.13% | 63.44% | 64.09% |
| Unique best action | 27 | 23.46% | 20.58% | 14.81% | 20.00% |

Across all 97 states, 69 have tied hindsight maxima and 55 distinguish candidate success counts. LOO training maxima tie in 230 of 291 folds. Always taking the lowest-index tied action instead gives 41.58% on all 97 states, compared with 41.52% under uniform ties. The fixed-first result is a sensitivity check, not chosen for its held-out score.

</details>

<details>
<summary>Coverage, depth strata and planned independent continuations</summary>

| Collection measure | Completed | Target |
| --- | ---: | ---: |
| States with all 15 outcomes | 97 | 149 |
| Continuation outcomes | 1,455 | 2,235 |
| Early states: decisions 1–3 | 74 | 74 |
| Later states: decisions 4–15 | 23 | 75 |

The four new states since the 93-state snapshot contributed 60 valid outcomes and 20 successes; all prior outcomes and teacher judgments are unchanged. Partial states remain outside these aggregates. The accepted sample’s depth comparisons are descriptive: replayability and remaining action budgets differ across groups.

| Candidate decisions | Matched states | Actor first | Before teacher | After teacher | LOO selection |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–3 | 70 | 40.48% | 44.29% | 44.76% | 44.98% |
| 4–9 | 12 | 27.78% | 40.74% | 39.81% | 41.85% |
| 10–15 | 11 | 21.21% | 25.25% | 27.27% | 27.17% |

The depth groups are disjoint: 1–3, 4–9 and 10–15. Earlier/later uncertainty and complete-versus-teacher-matched views are included in the aggregate.

The approved next stage adds two fresh continuations for each of the same five actions after all 149 primary states are audited. Its primary comparison will select using the original three and evaluate using the fresh two; five-fold LOO is secondary. Original outcomes, anchors and candidate identities remain fixed. [Collection protocol and history](ARM_INTEGRATION_PLAN.md#arm-continuation-finish100-20261007).

</details>

<details>
<summary>Method, uncertainty and limitations</summary>

Actor: official OpenWebRL SFT, temperature 1.0, top-p 0.95, top-k off, 4,096 response tokens. The 30-action budget includes the replayed prefix. Fresh prefixes are replayed into isolated browsers under strict observable-state checks; hidden server or browser state need not match.

Teacher: unchanged index-only Luna-high; three before judgments and three after judgments per execution repetition, plus separate order controls excluded from primary estimates. Judge: unchanged canonical o4-mini/AgentTrek terminal-success protocol, which can credit partial progress. These rates do not establish strict independent task completion.

Every state contributes equally. Invalid continuations remain zero; incomplete teacher panels are excluded only from teacher comparisons. Ten thousand bootstrap draws resample whole states, retaining all candidates, repetitions and overlapping LOO folds. Intervals are exploratory and not adjusted for multiple comparisons. Outcome artifacts and teacher receipts were hash-checked; independent exact LOO and fixed-first calculations agree. Task-level artifacts remain private.

</details>

<details>
<summary>Earlier 82/84 matched-state snapshots and historical plots</summary>

These overlapping snapshots are history, not independent replications. All rates in each row use that row’s teacher-matched cohort.

| Completed states | Matched states | Actor first | Before teacher | After teacher | LOO selection | Hindsight oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 86 | 82 | 38.21% | 42.01% | 42.68% | 43.39% | 55.69% |
| 88 | 84 | 37.70% | 41.40% | 42.06% | 42.69% | 55.16% |

Historical estimates and aggregate source hashes are preserved in the [current aggregate](arm_results/rl_integration/continuation-fixed97-20261008.json). The plots below describe the older 86-completed/82-matched snapshot. A zero-width interval in a tiny all-failure group is not population certainty.

![Historical continuation success by decision group](arm_results/rl_integration/continuation-depth-balanced-partial-20261007/paired-success.png)
![Historical teacher choice disagreement](arm_results/rl_integration/continuation-depth-balanced-partial-20261007/repeat-controls.png)

[Historical depth and coverage aggregate](arm_results/rl_integration/continuation-depth-balanced-partial-20261007/aggregate.json) · [Early-state diversity](arm_results/rl_integration/continuation-state-diversity-20261007/aggregate.json).

</details>

## Does execution change a saved-action progress judgment?

This separate diagnostic presents the same saved Piotr action to Luna-high before and after execution. The teacher returns progress probabilities, evidence and a rationale, rather than selecting among five actions.

| Comparison | Examples | Changed labels | Change rate |
| --- | ---: | ---: | ---: |
| Before → after, full set | 200 | 32 | 16% |
| Before → after, repeat subset | 100 | 18 | 18% |
| Before → before repeat | 100 | 13 | 13% |
| After → after repeat | 100 | 7 | 7% |

On the repeat subset, change exceeds before-repeat noise by 5 points [−3, +14]. Changed labels do not establish better accuracy, and only the executed action has evidence. [Diagnostic details](ARM_RESULTS.md#arm-teacher-evidence-primary200-20261006).

## Does random selection explain full-episode gains?

The root-state uniform control above randomizes one action and then continues with SFT. A separate full-episode control samples five candidates and picks uniformly at **every step**, under the same actor, decoding, browser, step cap and judge as its paired controls.

| Full-episode policy | Successes | Tasks | Task success |
| --- | ---: | ---: | ---: |
| SFT alone | 99 | 300 | 33.00% |
| Random of five | 100 | 300 | 33.33% |
| Piotr SelectionARM | 120 | 300 | 40.00% |

Random minus SFT is +0.33 points [−4.67, +5.67]; Piotr minus random is +6.67 points [+1.00, +12.33]. This supports useful learned selection in that run; random/SFT equivalence is not established. [Full-episode comparison](ARM_INFERENCE_SCALING.md#random5-sameday-results-20261007).
