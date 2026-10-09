# ARM formulations: outcome-based selection and execution evidence

**Immediate execution evidence has not yet established an advantage for the critic.** On the 119 matched historical states, Luna after execution improves continuation success over Luna before execution by +2.24 percentage points, with a 95% interval of [−0.28, +4.95]. These frozen operational estimates count invalid outcomes as zero. Separately, fresh continuations support outcome-based selection over uniform selection on 58 replayable states, but its advantage over actor-first or Luna remains uncertain. No selector has been trained; these are conditional continuation results, not benchmark pass@1.

<a id="branching-design"></a>

## How the branching experiment works

The original design uses the same five fixed candidate actions at each accepted state:

![Saved prefix replayed into isolated browsers, checked against the same observable state, then five candidate actions with three separate SFT continuations each](arm_results/rl_integration/branching-experiment-design.svg)

Each green box is a separate SFT continuation after executing its candidate action, with its own prefix replay. Both teachers choose without seeing continuation outcomes; after additionally sees immediate execution evidence. **Before, after and uniform selection are scored from the same 15 outcomes.** Uniform selection requires no separate rollout set. [PNG](arm_results/rl_integration/branching-experiment-design.png).

<a id="branch-extra2-heldout58-20261008"></a>

## Does selection from old outcomes improve fresh continuations?

**The fixed extra-two pass ended with an audited partial result:** 58 of 124 states passed strict replay, adding 580 outcomes. All 1,860 original records are unchanged, giving **2,440 saved outcome records**. The remaining 66 states failed reconstruction; their 660 missing continuations are not counted as zero. The approved 3,100-record target and original 149-state target remain incomplete.

Each row below uses the **same 58 states from 58 task groups, with 580 fresh records and two draws per candidate**. The outcome-based choice uses only the three original draws. Tied best actions receive their exact uniform expected score. Luna’s score averages the fresh outcomes of its three saved before-execution choices; no new teacher calls were made. Committed invalid outcomes count as zero.

| Root-action choice | States | Continuation success | 95% task-bootstrap interval |
| --- | ---: | ---: | --- |
| Actor’s first candidate | 58 | 26.72% | [17.24%, 37.07%] |
| Uniform over five candidates | 58 | 26.03% | [17.76%, 34.83%] |
| Saved Luna before execution | 58 | 31.03% | [20.11%, 42.24%] |
| Select on original three; score fresh two | 58 | 34.54% | [23.97%, 45.23%] |

The outcome-based choice gains **+8.51 percentage points over uniform [3.45, 14.37]**. Its gains over actor-first, **+7.82 points [−0.26, 16.84]**, and Luna, **+3.51 points [−2.84, 10.80]**, remain unresolved. These exploratory, pointwise intervals support a useful continuation-outcome signal; they do not establish a trained selector’s performance or superiority to teacher-label training. [Aggregate estimates and audit hashes](arm_results/rl_integration/continuation-extra2-heldout58-20261008.json).

**Replay selection materially changes the cohort.** Using only original draws, actor-first scored 17.82% on these 58 states versus 40.40% on the 66 that later failed reconstruction. Consequently, comparing fresh 34.54% directly with the earlier 124-state LOO result of 38.12% would confound cohort composition, draw count and collection time. The accepted states span candidate decisions 1–3/4–9/10–15 with 36/11/11 states respectively. Frozen observable-state checks do not guarantee identical hidden state or eliminate website drift.

<details>
<summary>Five-draw sensitivity, ties, uncertainty and judge limitations</summary>

These secondary rows pool old and fresh draws on the same 58 states, yielding 1,450 outcome records. Five-fold LOO selects with four draws and scores the fifth, averaging all overlapping folds. It mixes collection times; the temporally separated old-three/fresh-two comparison above remains primary.

| Root-action choice | States | Pooled continuation success | 95% task-bootstrap interval |
| --- | ---: | ---: | --- |
| Actor’s first candidate | 58 | 21.38% | [13.10%, 30.35%] |
| Uniform candidate | 58 | 24.97% | [16.76%, 33.66%] |
| Select on four; score held-out fifth | 58 | 33.95% | [23.94%, 44.16%] |
| Same-data five-draw hindsight oracle | 58 | 41.38% | [31.03%, 51.72%] |

The hindsight row chooses and scores on the same outcomes and is optimistic. Original-three selection ties at the top in 39/58 states; 29/58 give every candidate equal success counts, including 25 all-zero panels. Five-draw hindsight still ties in 35/58 states. Five-fold training maxima tie in 187/290 folds. Ties and duplicate candidates are retained.

Every state has equal weight. Ten thousand bootstrap draws, seed 20261007, resample whole task groups with all their states and normalize by sampled state count. The 58 groups each contain one state here. Intervals are pointwise, exploratory and unadjusted for multiple comparisons. Exact tie expectations avoid choosing winners from held-out outcomes. Valid-only sensitivity, with its own weighted denominators, is included in the aggregate.

The official SFT actor and decoding remain temperature 1.0, top-p 0.95, top-k off and 4,096 response tokens, with the original 30-action budget including replayed prefixes. Canonical o4-mini/AgentTrek labels remain unchanged. Of the 580 fresh records, 523 are valid and 57 invalid; 303 received a judge call and 220 valid records retain native unjudged zero outcomes. Qualitative checks found positives that conflate website statistics or accept a different language edition. These scores measure the frozen judge’s operational reward, not independently established factual or instruction-complete success. Source and identity checks support seed assignment; an independently echoed request seed was unavailable.

</details>

<details>
<summary>Partial collection, reconstruction failures and final accounting</summary>

| Coverage | States | Saved outcome records |
| --- | ---: | ---: |
| Full five draws per action | 58 | 1,450 |
| Original three only | 66 | 990 |
| Preserved combined data | 124 | 2,440 |
| Approved combined target | 124 | 3,100 |

All 124 states were processed once under the frozen reconstruction rule. The first failures on 66 rejected states were 58 visible-snapshot mismatches, five raster mismatches, one page mismatch, one prefix-capture timeout and one initial-navigation timeout. Their 147 unreleased attempt receipts and 513 untouched queue slots remain evidence of missing continuations. They are not added to the outcome dataset. One historical post-commit release-read invalid remains among the original records as previously audited; no candidate execution is invented.

The sole extra-two allocation consumed 14,238 of 28,800 approved seconds on four H200s / 32 CPUs / 480 GiB; 14,562 seconds remain unused and are not transferred. All 727 attempts are charged. Extra-two used 727 browser starts, 6,788 SFT calls and 303 judge calls / $2.406679, with no teacher calls. Combined with prior charges, the shared ledger contains 6,923/9,000 browser starts, 40,512/110,000 SFT calls, 1,438/6,600 judge calls / $11.5469541 of $25, and 1,698/2,200 Luna calls / $6.5299025 of $15.

Scheduler and W&B finished. Independent audits reconciled artifacts, all-attempt accounting, the original-record hashes and controller-owned actor/browser teardown. Post-allocation SSH was unavailable, so teardown does not claim an independent physical node scan. The pass is closed as audited partial, with full-target completion false. The frozen rule forbids replacement or outcome-based replay after reconstruction rejection; no automatic rerun or relaxed replay was used. The separately approved 32-state pilot remains a separate experiment and budget.

</details>

<details>
<summary>Original three-draw results: 124 states and the matched teacher comparison</summary>

## Does execution improve action selection?

<a id="arm-continuation-branches-results-20261007"></a>

**Collection stopped short of its target:** all 2,090 fixed candidate tasks were considered, yielding **124 of 149 states** under unchanged replay checks and depth quotas. At the October 8, 17:58 UTC audit cutoff, these states have all five actions × three continuations: **1,860 outcomes**, including 1,710 valid and 150 invalid outcomes. Invalids remain zero. The 149-state target remains incomplete. The later extra-two campaign is reported above; none of its outcomes enter these original-three-draw estimates.

### Can observed continuation outcomes improve the root-action choice?

| Root-action choice | States | Continuation success | 95% state-bootstrap interval |
| --- | ---: | ---: | --- |
| Actor’s first sample | 124 | 29.84% | [22.85%, 37.10%] |
| Uniform candidate | 124 | 32.42% | [26.13%, 38.98%] |
| Select on two continuations; score held-out third | 124 | 38.12% | [31.06%, 45.31%] |
| Same-data hindsight action oracle | 124 | 50.00% | [42.47%, 57.53%] |

Leave-one-out selection improves over the actor’s first sample by **+8.28 percentage points [+3.84, +12.99]** and over uniform selection by **+5.70 points [+2.60, +9.03]**. For each state, select using two continuations per action, score the held-out third, and average all three overlapping folds. Ties use their exact uniform expectation without consulting the held-out outcome. Candidate 0 is the first actor sample; uniform selection averages all five entries, including duplicates.

The hindsight oracle chooses and scores using the **same three outcomes**, making its 50.00% optimistic. It does not select a successful trajectory from all 15. LOO uses additional continuation outcomes and does not measure a trained selector’s performance.

### Does immediate execution evidence help the teacher?

All rows below use the **same 119 states and 1,785 outcomes**. Four historical invalid teacher panels and one unavailable-after panel are excluded only here; their states remain in the 124-state outcome comparison. Both index-only Luna-high teachers lack continuation outcomes; after-execution additionally sees each candidate’s immediate execution evidence.

| Root-action choice | Matched states | Continuation success | 95% state-bootstrap interval |
| --- | ---: | ---: | --- |
| Actor’s first sample | 119 | 30.53% | [23.25%, 37.82%] |
| Uniform candidate | 119 | 33.22% | [26.72%, 39.89%] |
| Luna before execution | 119 | 35.11% | [27.73%, 42.67%] |
| Luna after immediate execution | 119 | 37.35% | [29.88%, 45.10%] |
| Select on two continuations; score held-out third | 119 | 38.51% | [31.20%, 45.88%] |
| Same-data hindsight action oracle | 119 | 50.70% | [43.14%, 58.26%] |

After minus before is **+2.24 points [-0.28, +4.95]**. LOO minus after is **+1.16 points [-2.56, +4.92]**. Teachers average repeated selections, which do not add rollout data. [Aggregate estimates, intervals and audit hashes](arm_results/rl_integration/continuation-fixed124-20261008.json).

### Where are the states, and how does selection perform by depth?

The groups below are disjoint. These are descriptive strata: tasks, replayability and remaining action budgets differ, so these rows do not identify a causal depth effect.

| Candidate decisions | States | Share | Actor first | Uniform | LOO selection | Hindsight oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–3 | 74 | 59.7% | 39.19% | 39.91% | 43.60% | 56.76% |
| 4–9 | 27 | 21.8% | 16.05% | 22.96% | 32.55% | 43.21% |
| 10–15 | 23 | 18.5% | 15.94% | 19.42% | 27.05% | 36.23% |

The early 74-state cohort is unchanged; **50 of the targeted 75 later states** were collected. Adding later states changes cohort composition, not the actor or teacher. Per-stratum intervals are in the aggregate.

<details>
<summary>Exact decision counts and teacher-matched depth comparisons</summary>

| Candidate decision | Complete states | Teacher-matched states |
| --- | ---: | ---: |
| 1 | 33 | 31 |
| 2 | 23 | 22 |
| 3 | 18 | 17 |
| 4 | 7 | 7 |
| 5 | 6 | 6 |
| 6 | 6 | 5 |
| 7 | 4 | 4 |
| 8 | 2 | 2 |
| 9 | 2 | 2 |
| 10 | 4 | 4 |
| 11 | 6 | 6 |
| 12 | 3 | 3 |
| 13 | 1 | 1 |
| 14 | 5 | 5 |
| 15 | 4 | 4 |

| Candidate decisions | Matched states | Actor first | Before teacher | After teacher | LOO selection |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–3 | 70 | 40.48% | 44.29% | 44.76% | 44.98% |
| 4–9 | 26 | 16.67% | 24.36% | 25.64% | 31.24% |
| 10–15 | 23 | 15.94% | 19.32% | 28.02% | 27.05% |

Matched groups total 119 states and use different denominators from the all-state rows above.

</details>

<details>
<summary>Repeat controls, oracle-action agreement and ties</summary>

| Choice disagreement | Matched states | Rate |
| --- | ---: | ---: |
| Before versus before repeat | 119 | 32.77% |
| After versus after repeat | 119 | 24.28% |
| Before versus after | 119 | 46.62% |

Cross-condition disagreement exceeds mean within-condition disagreement by +18.10 points [+12.53, +23.93]. More consistent choices do not by themselves establish better success. Presentation-order controls are separate from primary estimates and mix order sensitivity with teacher stochasticity; they do not identify a causal order effect.

Oracle-action agreement credits any candidate tied for the highest observed success across three continuations. This is a noisy same-data label; all-equal candidates make agreement automatic.

| Oracle subset | Matched states | Before agreement | After agreement | Actor-first agreement | Uniform agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Any tied best action | 119 | 68.07% | 69.37% | 59.66% | 63.70% |
| Unique best action | 36 | 22.22% | 21.91% | 11.11% | 20.00% |

Across 124 states, **86 have tied hindsight maxima (69.35%)** and **70 distinguish candidate success counts (56.45%)**. LOO training maxima tie in **286 of 372 folds (76.88%)**. Always choosing the lowest-index tied action gives 36.83%, versus 38.12% for uniform ties; this is a sensitivity calculation, not an independently validated selection rule. Agreement uncertainty and independent uniform tie-breaking are retained in the aggregate.

</details>

<details>
<summary>Method, uncertainty, judge limitations and next continuations</summary>

Actor: official OpenWebRL SFT, temperature 1.0, top-p 0.95, top-k off, 4,096 response tokens. The 30-action budget includes the replayed prefix. Isolated browsers must pass strict observable-state replay checks; hidden state need not match. The accepted cohort is not a random sample of tasks or decision points.

Teacher: unchanged index-only Luna-high, three before judgments and three after judgments per execution repetition, plus separate order controls. Judge: unchanged canonical o4-mini/AgentTrek terminal-success protocol. Saved positive examples retain partial-progress, temporal-grounding, absence-claim and filter-application concerns. Additional examples accept goal reinterpretation and conflate page/article or registered-user/active-editor metrics. Native labels remain unchanged; these rates do not establish strict independently verified task completion.

Every state contributes equally. Ten thousand bootstrap draws resample whole states, retaining candidates, repetitions and overlapping folds. Main tables use the LOO aggregation seed; repeat controls use their original seed. Intervals are exploratory and not adjusted for multiple comparisons. Saved outcome/teacher fingerprints and independent exact LOO calculations agree. Task-level artifacts remain private.

The cohort amendment froze these 124 states and all 1,860 original outcomes. The extra-two pass subsequently preserved these records and added 580 outcomes on 58 states; its full 1,240-new-outcome target remains incomplete. No candidates or teachers were regenerated. The original 149-state target remains unmet. The candidate pool was exhausted without relaxing quotas, replay checks or scientific settings. [Collection protocol and history](ARM_INTEGRATION_PLAN.md#arm-continuation-finish100-20261007).

</details>

</details>

<a id="branch-selector-scaleup-20261008"></a>

## Can a trained selector generalize beyond the branching states?

**The approved 32-state collection pilot is running, with startup and initial collections verified; the full pilot endpoint remains incomplete.** A separately approved [extension to 32 draws per fixed action](#branch-extra24-20261008) will compare eight-draw and 24-draw selection on the same held-out outcomes. The larger goal is outcome-based selector adaptation with eight continuations per new candidate and separate seen-task/new-state and unseen-task validation. The held-out continuation results motivate this test; they do not establish that a learned selector will reproduce the gain. No selector has been trained in this study.

<a id="branch-post-to-pre-study-20261008"></a>

**Research priority, October 8: establish the value of the next screenshot, then test whether that signal transfers to a pre-action critic.** The current pilot supplies branches and continuation targets; its zero-teacher-call budget does not include new pre/post critic judgments.

| Question | Matched comparison | Evidence required |
| --- | --- | --- |
| Does immediate execution evidence help? | Same frozen critic and repeated index-only selection protocol, with versus without each candidate’s immediate next screenshot | Paired selected-action success on held-out continuations; task-bootstrap interval and valid-panel coverage |
| Can it train a better pre-action critic? | Identical pre-action students, training states and optimization budgets; targets from pre-action versus post-action teacher choices | Improvement on distinct seen-task states and unseen tasks, using only pre-action inputs at inference |
| How informative are the continuation outcomes themselves? | Separately labeled continuation-return-supervised selector/reference | Held-out outcomes that never enter its selection or training targets |

The post-action input must be the screenshot immediately after the fixed candidate action, bound to its execution receipt. A terminal screenshot or later suffix cannot substitute. Prespecify the evidence draw independently of outcomes; do not choose the successful or clearest-looking transition. Teachers receive no continuation returns or future screenshots. Repeated teacher choices can supply matched target distributions for distillation; both students use the same loss and see only the original state and candidates. Keep teacher labels and continuation-derived targets distinct. Missing evidence excludes a paired comparison, with coverage reported, while other valid targets remain usable.

Keep collection active while preparing these comparisons from saved artifacts. Size the next collection request from valid candidate-panel yield, depth coverage and all-attempt cost, rather than raw accepted-state count alone. Larger collection, new critic calls and student training still require their own exact budget approvals.

| Dataset role | Total target states | Retained existing states | New state target | New continuation target |
| --- | ---: | ---: | ---: | ---: |
| Training | 1,000 | 124 | 876 | 35,040 |
| Seen-task validation | 250 | 0 | 250 | 10,000 |
| Unseen-task validation | 250 | 0 | 250 | 10,000 |

New states use five candidates × eight draws. Existing 124 states remain training eligible with their actual observations: 2,440 audited records, comprising five draws per action on 58 states and three on 66 states. Thus the full aspirational design requires 55,040 new outcomes and 57,480 combined outcomes; it does not require topping up historical states to eight draws. Full collection and selector-fitting allocations remain unapproved.

Task and state assignments are fixed before new outcomes. All candidates, draws and retries from one state stay together. Seen-task validation requires independently seeded discovery trajectories, distinct pre-action task/screenshot/history inputs, and an accepted training state from the same task. A different candidate panel alone does not create a new state. Previously collected task groups cannot enter unseen-task validation or test, but genuinely new states from those tasks may enter seen-task validation. Additional task groups remain locked for final testing. “Unseen” means unseen by this selector fit, not proven absent from backbone pretraining.

Fit a fresh adapter on released SelectionARM to each candidate’s success fraction using its **valid draw count**, retaining ties and all-equal panels. The proposed loss is independent sigmoid/binomial BCE over five candidate-index logits; serving still returns one index. Inputs contain only pre-action information and the candidate panel. **User amendment, October 8: skip invalid continuation measurements in targets and future success-rate denominators.** Keep valid unsuccessful outcomes, including native unjudged action-limit zeros. A candidate with no valid draws has no target and is masked from the loss. Preserve raw invalid records, reasons and costs without rerunning them. Missing records are unresolved, not invented failures. This is outcome adaptation of a teacher-pretrained model.

Evaluate selected-action continuation success separately on both validation sets against actor-first, uniform and the frozen selector. A rollout-based reference selects on the valid observations within six new draws and scores valid observations from the other two. Paired five-candidate reference comparisons require at least one valid selection draw and one valid held-out draw per candidate, on the same states for every method; report excluded states and per-candidate coverage. Do not renormalize the candidate set using held-out validity. Valid-only rates condition on measurement availability and can have unequal draw counts; the historical operational results above retain their original invalid-as-zero definition. Bootstrap whole task groups. Keep final test tasks unopened during checkpoint selection and label selected-checkpoint validation scores accordingly. Judge limitations and replayability selection remain; this does not establish end-to-end benchmark improvement or superiority to matched teacher-label training.

<details>
<summary>Frozen pilot opportunities, source overlap and resource accounting</summary>

The full draft groups 2,090 task identities and defines three decision slots per task across decisions 1–3, 4–9 and 10–15. It has 2,978 training, 1,489 seen-task validation, 1,203 unseen-task validation and 600 locked-test opportunities. The pilot freezes 1,200 opportunities across 400 task groups: 600 train, 300 seen-task validation, 300 unseen-task validation, and 400 in each depth band. These are opportunities, not collected states. Eight CPU tests and 12 provenance/isolation/arithmetic checks passed. Assignment never consults new successes, ties or teacher preferences.

Exact task/normalized-goal comparison found no overlap with protected OM2W, WebVoyager, DeepShop or WebGym test cohorts. This does not establish semantic decontamination. Earlier selector dev/future/retention and joint-validation splits overlap 366 task groups, including 25 old states. Those historical splits are study-specific and remain untouched: retaining these examples is allowed for this new fit, but reused historical cohorts cannot support independent evaluation claims for it.

The recent later-state collection produced 38 accepted states from 1,173 discoveries in 39,868 seconds on four H200s, consuming 2,695 browser starts with three draws per action. Eight draws require all 40 isolated replay contexts to pass the unchanged readiness gate before candidate dispatch. Pilot startup and initial replay groups have been validated; the complete 32-state endpoint still requires audit. The existing pool cannot guarantee the full state target; source expansion or more prespecified slots will be budgeted from measured pilot yield.

**Original eight-draw pilot allocation:** four H200s, up to 40 CPUs, 480 GiB and 43,200 seconds total including startup, tests and retries. The scheduler requires at most eight CPUs per GPU, so the deployed request uses 32 CPUs. Caps: 5,000 browser-start attempts / 40 concurrent browsers, 100,000 local SFT call attempts, 2,000 canonical judge HTTP attempts / $25, no teacher calls. Stop at 32 newly accepted states / 1,280 new continuations, 1,200 opportunities, or the first binding cap. Diagnose zero acceptance among the first 24 resolved replay groups. Preserve every attempt and reserve time for cleanup. Scheduler admission, GPU startup and scientific completion are separate milestones.

<a id="branch-extra24-20261008"></a>

**Approved continuation-precision extension:** job `351591` has started on four H200s and is reconstructing frozen states; first-batch collection validation is pending. It adds draws 8–31 to the same five frozen candidate responses on up to 32 pilot states, preserving draws 0–7. The maximum is 3,840 additional outcomes and 5,120 combined outcomes. Nineteen individually audited parent states, with 760 original records, are now bound; later audited parent states enter in their original acceptance order. It does not wait for the whole pilot to finish, regenerate proposals or choose states by their outcomes.

Compare action selection using the first eight versus first 24 draws, scoring both on the **same untouched final eight draws (24–31)**. Use valid observations and exact uniform averaging over tied winners; require a common five-candidate panel with valid fitting and held-out observations and report missing coverage. Keep task splits intact and bootstrap whole tasks. This tests how much more sampling improves action-value estimates. It estimates held-out performance of outcome-based selection within the fixed proposal set and continuation policy; a maximum selected and scored on all 32 draws is an optimistic hindsight diagnostic, not a true upper bound or evidence that a trained critic achieves it.

The separately approved cap is **four H200s, 32 CPUs, 480 GiB and 57,600 seconds across all attempts**, with 5,000 browser starts / 40 concurrent, 240,000 SFT calls, 8,000 judge HTTP attempts / $75 and zero teacher calls. Three sequential 40-context batches per state retain strict replay checks. Failed attempts remain charged; rejected or missing measurements are not replaced by invented outcomes. The extension has its own verified 30-minute agent-callback supervisor and does not borrow unused time from the pilot or historical extra-two campaign. Independent runtime validation, the requested output count and final analysis remain incomplete at this startup snapshot.

The later fitting estimate remains one H200, eight CPUs, 120 GiB and two hours total, pending separate exact approval and validated data/loss code. The existing 124-state extra-two campaign retains its own budget and protocol. [Machine-readable approved pilot and larger proposal](arm_results/rl_integration/branch-selector-scaleup-proposal-20261008.json).

</details>

<details>
<summary>Earlier snapshots and historical plots</summary>

These overlapping cohorts are history, not independent replications. Rows use their own teacher-matched cohorts.

| Completed states | Matched states | Actor first | Before teacher | After teacher | LOO selection | Hindsight oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 86 | 82 | 38.21% | 42.01% | 42.68% | 43.39% | 55.69% |
| 88 | 84 | 37.70% | 41.40% | 42.06% | 42.69% | 55.16% |
| 97 | 93 | 36.56% | 41.58% | 42.05% | 42.47% | 55.20% |
| 120 | 115 | 31.01% | 35.56% | 37.78% | 38.21% | 50.72% |

The [97-state](arm_results/rl_integration/continuation-fixed97-20261008.json) and [120-state](arm_results/rl_integration/continuation-fixed120-20261008.json) aggregates remain unchanged. Historical values and hashes are also indexed in the [124-state aggregate](arm_results/rl_integration/continuation-fixed124-20261008.json). The plots below describe the older 86-completed/82-matched snapshot; a zero-width interval in a tiny all-failure group is not population certainty.

![Historical continuation success by decision group](arm_results/rl_integration/continuation-depth-balanced-partial-20261007/paired-success.png)
![Historical teacher choice disagreement](arm_results/rl_integration/continuation-depth-balanced-partial-20261007/repeat-controls.png)

[Historical depth and coverage aggregate](arm_results/rl_integration/continuation-depth-balanced-partial-20261007/aggregate.json) · [Early-state diversity](arm_results/rl_integration/continuation-state-diversity-20261007/aggregate.json).

</details>

<a id="selection-head-20261009"></a>

## Can the actor's own features select actions? Piotr SelectionARM reference — October 9

**The released Piotr SelectionARM, run offline on the 124 fixed panels, chooses actions with 35.75% continuation success: +5.91 points over the actor's first sample and +3.33 points over uniform.** This is the outcome reference for the shared actor–selector head now being trained (job `351647`). Scores average each chosen action's three SFT continuations, with invalid outcomes counted as zero. Paired intervals use 20,000 state-bootstrap draws; each state is a distinct task.

| States | Count | Piotr ARM | Actor first | Uniform | ARM − uniform, pp [95%] | ARM − first, pp [95%] |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| All | 124 | 35.75% | 29.84% | 32.42% | +3.33 [+0.11, +6.72] | +5.91 [+1.88, +10.22] |
| Prompt-verified | 114 | 35.96% | 30.70% | 33.04% | +2.92 [−0.53, +6.61] | +5.26 [+1.17, +9.94] |
| Verified, task absent from Piotr's release | 105 | 36.83% | 31.11% | 33.33% | +3.49 [+0.25, +6.92] | +5.71 [+1.27, +10.48] |

Luna before execution scores 35.11% on a 119-state subset ([above](#arm-continuation-branches-results-20261007)); the two selectors have not been compared on identical states. Piotr ARM returned a valid selection on all 124 panels and chose candidate 0 on 62. Nine tasks also occur in Piotr's training release, hence the last row. These are conditional continuation results, not benchmark pass@1.

**Head under evaluation.** A small permutation-equivariant transformer reads the frozen SFT actor's hidden states — the prompt's last token, plus each candidate's mean reasoning, mean action and end-token states from layers 9/18/27/36 — and scores the five candidates jointly. At serving, these states are a by-product of generating the candidates, so selection needs no second 4B model. The head trains only on GPT-5.5 teacher choices from Piotr's release; the branching outcomes are used for evaluation only. A pointwise variant tests whether joint comparison matters.

| Teacher-labeled panels | Panels | Teacher agreement: uniform | First candidate | Most common action |
| --- | ---: | ---: | ---: | ---: |
| Train | 35,916 | 42.4% | 41.9% | 47.7% |
| Validation, held-out task groups | 3,759 | 44.7% | 44.5% | 50.1% |
| Validation, non-identical panels | 3,265 | 36.3% | 36.1% | 42.6% |

Agreement counts a choice as correct when it is action-equivalent to the teacher's pick: identical calls, with clicks within five normalized units. Validation selects the checkpoint, so its head score will not be an untouched test score.

<details>
<summary>Prompt reconstruction, feature checks, data exclusions and budget</summary>

- **Prompts.** Branch anchors do not save the actor prompt. It is rebuilt from continuation-rollout conversations and accepted only when its SHA-256 equals the hash journaled at candidate sampling: 114/124 match. All 114 also reproduce the server's prompt token count, including image tokens. In the other 10, replayed prefixes word the tool feedback differently from discovery; they use the replay prompt and appear only in the "All" row (token differences −23 to +9).
- **Candidates.** The head reads the actor's exact generated token ids. Re-tokenizing the saved text would change 47/620 candidates.
- **Features.** Cached-prefix, batched extraction matches full forwards exactly in a float32 unit test. On the real bf16 4B model the relative L2 difference is 1.43–1.48% on two checked states, under the 2% abort threshold.
- **Teacher panels.** Pinned release `0d83b48`, joint-data v2 task split and quarantine: 39,675 panels over 2,982 states. Excluded draw records: 300 conflicting draw IDs, 80 quarantined, 61 missing teacher selections and 32 outside the split.
- **Budget.** Job `351647`: approved 1 H200, 8 CPUs, 120 GiB, 3 h (estimated $2.70), including retries; no API calls or browsers. Code: branch `arm-selection-head`, commit `87a204b`.

[Aggregate estimates and accounting](arm_results/rl_integration/selection-head-20261009.json).

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
