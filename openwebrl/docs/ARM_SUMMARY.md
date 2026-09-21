# Action Reward Model integration: methods and results

This is the concise collaborator summary. Detailed configs, provenance, and
run history are in [ARM_RESULTS.md](ARM_RESULTS.md) and
[ARM_INTEGRATION_PLAN.md](ARM_INTEGRATION_PLAN.md).

## 1. Inference-time ARM selection

**Method.** For state (s_t), sample five actor responses (a_{t,1:5}). ScalarRM
scores each response independently; SelectionARM compares the candidates using
the full reasoning, action, and browser state. Execute

$$
a_t^* = \arg\max_i \mathrm{ARM}(s_t,a_{t,i}),
$$

while the baseline executes one actor sample. Results reproduce the Action
Reward Models setup with the o4-mini/AgentTrek Online-Mind2Web judge.

| Policy | Successes / 300 | Valid | Invalid | Overall | Valid-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline, one sample | 90 | 267 | 33 | 30.0% | 33.7% |
| ScalarRM, five candidates | 114 | 251 | 49 | 38.0% | 45.4% |
| SelectionARM, five candidates | 128 | 256 | 44 | **42.7%** | **50.0%** |

- **Baseline:** execute one actor sample without an ARM call.
- **ScalarRM:** score each of five candidates independently and execute the candidate with the highest scalar score.
- **SelectionARM:** compare the five candidates jointly using full state, reasoning, and action context, then execute the selected candidate.

**Conclusion:** ARM provides a large inference-time gain, with SelectionARM
stronger than ScalarRM. This requires five-way sampling and selection at every
turn, so it increases inference cost.

## 2. Offline filtered SFT and preference learning

**Method.** Retain ARM-selected trajectories and action/turn examples, then
train the actor offline. SFT minimizes token cross-entropy,

$$
\mathcal L_{\mathrm{SFT}}=-\sum_t\log \pi_\theta(y_t\mid x,y_{<t}).
$$

C2 is the original filtered recipe; 1A increases the effective batch size. The
joint-data runs add Piotr's ARM data. DPO uses preferred/rejected responses:

$$
\mathcal L_{\mathrm{DPO}}=-\log\sigma\!\left(\beta\left[\log\frac{\pi_\theta(y^+|x)}{\pi_{\rm ref}(y^+|x)}-\log\frac{\pi_\theta(y^-|x)}{\pi_{\rm ref}(y^-|x)}\right]\right).
$$

| Training recipe | Evaluation | Successes / 300 | Valid-only | Relative headline |
| --- | --- | ---: | ---: | --- |
| Starting actor reference | Full 300 | 90 | 33.7% | Baseline |
| C2 filtered SFT | Combined 300 | 95 | 36.8% | Control for 1A |
| 1A filtered SFT | Combined 300 | 100 | 39.1% | +1.7 pp vs C2 |
| Joint C2 + Piotr SFT | Fresh 300 | 102 | 37.8% | +4.0 pp vs starting actor |
| Joint C2 + Piotr DPO | Fresh 300 | **104** | **40.9%** | +4.7 pp vs starting actor |

- **Starting actor:** unmodified OpenWebRL-SFT policy used as the training and evaluation reference.
- **C2 filtered SFT:** retain every usable executed ARM-selected turn from a valid successful trajectory; train the chosen reasoning-plus-action response with language-only LoRA (rank 16, alpha 32, dropout 0.05), AdamW at `1e-5`, effective batch 16, cosine decay, 3% warmup, and two epochs.
- **1A filtered SFT:** use the same 8,394-turn C2 dataset and LoRA/optimizer settings, but microbatch 1 with accumulation 32 (effective batch 32) and one pass (263 updates); this isolates optimizer/exposure from the original effective batch 16 recipe.
- **Joint C2 + Piotr SFT:** concatenate 3,464 C2 and 2,076 Piotr retained states and apply the same full-response SFT objective for one pass.
- **Joint C2 + Piotr DPO:** use the same chosen/rejected pairs and frozen starting actor as reference, with full-response DPO, `beta=0.1`, and no auxiliary SFT term.

**Conclusion:** Offline training transfers only modestly compared with the
inference-time selector. DPO is the best measured endpoint, but its advantage
over joint SFT is small and not statistically significant in the paired audit.

## 3. Online RL with ARM turn-level bonuses

**Method.** Keep the ordinary terminal outcome reward and add a bounded ARM
bonus on eligible actor turns:

For trajectory $i$, the judge produces a terminal outcome $R_i\in\{0,1\}$
(with `-1` reserved for a format failure), and that outcome is propagated to
every turn before group normalization. For turn $t$:

$$
A'_{i,t}=A_i+\beta\,q_{i,t}\left(\mathbf{1}[j_{i,t}=0]-0.2\right),
\qquad q_{i,t}\in\{0,1\},
$$

where $A_i$ is the normalized outcome advantage, $q_{i,t}$ indicates that the
turn received a usable ARM label, and $j_{i,t}=0$ means SelectionARM chose the
executed actor candidate. Thus `beta=0.5` is the ARM scale and `q=0.20` is the
target fraction of turns sent for labeling; $q$ is a sampling gate, not a
multiplicative 0.20 applied to every reward. The bonus is applied only to
retained, valid ARM-labelled turns. All variants
use the local browser, GPT-4.1 action-history judge, 48-group collection, and
PPO2; they differ in which rollout groups contribute labels.

So a successful rollout does receive raw outcome reward `1` on every turn
because the terminal judge result is propagated across its turn history. The
training value is then the group-normalized $A_i$, with the ARM term added only
on labeled turns; it is therefore not literally `1` after normalization.

- **Shared hyperparameters:** `K=5` candidates, SelectionARM with full
  reasoning-plus-action input, turn sampling fraction `q=0.20`, ARM scale
  `beta=0.5`, 48 accepted groups, global batch 256, two PPO epochs, constant
  learning rate `1e-6`, weight decay 0.1, 32 browsers, 15-turn horizon,
  1,024-token responses, and 32,768-token context.
- **Why these values:** `K=5` matches the validated inference setup;
  `beta=0.5` and `q=0.20` passed calibration with an ARM/outcome RMS ratio near
  6%, keeping the auxiliary signal bounded; batch 256, PPO2, and `1e-6`
  preserve the tested OpenWebRL optimization scale while limiting off-policy
  reuse.

`q=0.20` is the probability of attempting an ARM label, not a guarantee that
20% of training turns receive a bonus. The realized applied-label fraction is
usually 8--10%. A [43-collection audit](ARM_INTEGRATION_PLAN.md#arm-label-coverage-confidence-audit-20260919)
found duplicate-action rejection was the largest cause: 40--50% of sampled
turns, versus 3--7% truncation/empty outputs and 1--2% candidate parsing errors.
The current gate requires all five candidate actions to differ. A completed
90-task confidence replay found that high confidence can reflect a position
tie-break on identical responses; confidence alone is insufficient to replace
the gate. A [one-call, at-least-two-action gate test](ARM_INTEGRATION_PLAN.md#arm-min2-gate-test-20260919)
increased archived gate passes from 47.3% to 88.5% of sampled turns (1.87×).
The gate and optional duplicate-aware action credit are independently configurable.
The three original runs retain their original gate; B/C test the changes below.

[Iteration-zero gate ablations](ARM_INTEGRATION_PLAN.md#arm-bc-firstupdates-20260920):
[B](https://wandb.ai/zixianma/openwebrl/runs/arm-gate-b-309053) relaxes the gate to
at least two actions; [C](https://wandb.ai/zixianma/openwebrl/runs/arm-gate-c-309054)
also uses action-equivalence credit. Their W&B display names now identify both
the variant and credit rule.
As of September 21, 17:00 UTC, B **313208** and C **313210** both completed
**iteration 20 / 284 Adam updates**. C's full-300 evaluation **313211** finished
at **36.67% overall / 44.53% valid-only**; B's **313209** finished at
**33.67% / 44.30%**. All 300 rollouts and verdicts are saved for each. Calibration passes; recent B/C
collections label 16–20% of retained ordinary turns, with bonus/outcome RMS
about 9–11%, beta=0.5 unchanged.
[C evaluation audit](RL_EVALUATION.md#arm-gate-c-iter20-results-20260921).
[Initial audit](arm_results/rl_integration/bc-firstupdates-20260920.json);
[continuation status](RL_RUNTIME.md#arm-training-status-20260920-2224).

Both full-300 evaluations were released after their iteration-20 checkpoints passed validation.
The separate [rescue-yield pilot](ARM_INTEGRATION_PLAN.md#arm-rescue-yield-pilot-20260920)
**313264** completed: among eight screened all-failure tasks, ARM rescued
**0/8**, one ordinary retry **1/8**, and five ordinary retries **3/8**.
All 374 trajectories are saved. This small training-task pilot shows no rescue
benefit; [protocol and results](ARM_RESULTS.md#arm-rescue-yield-313264).
The [fixed-state selection audit](ARM_RESULTS.md#arm-selection-quality-313774)
also finished: ARM minus random next-response acceptability was **+1.6 / +3.2 /
−3.2 pp** for SFT / outcome-only iteration 20 / iteration 90. This small,
teacher-labeled panel does not establish drift or task-success gains. The
[terminal-success audit](ARM_RESULTS.md#arm-task-success-314664) **314664**
completed actor+ARM on the **same fixed100 tasks** at outcome-only iterations
20 and 90. All 200 primary rollouts and verdicts are saved; zero selector
fallbacks. The SFT cells reuse saved results on those exact IDs.
Rates below are overall / valid-only; Δ is ARM minus actor-only in percentage
points, calculated from unrounded rates.

| Actor checkpoint | Actor alone · fixed100 | Actor + SelectionARM · fixed100 | Δ overall / valid-only (pp) |
| --- | ---: | ---: | ---: |
| Starting SFT | 26.00% / 30.59% (26/85) | 36.00% / 43.37% (36/83) | +10.00 / +12.79 |
| Outcome-only iteration 20 | 25.00% / 35.21% (25/71) | 38.00% / 47.50% (38/80) | +13.00 / +12.29 |
| Outcome-only iteration 90 | 35.00% / 51.47% (35/68) | 43.00% / 53.09% (43/81) | +8.00 / +1.62 |

Parentheses show successes / valid tasks. Overall always uses all 100 tasks;
valid-only excludes invalid attempts and includes both valid successes and
valid failures. The valid task sets can differ between the two runs.

SFT uses o4-mini; RL uses GPT-4.1. The actor-only controls are historical;
dates, availability and decoding differ, so these are descriptive inference
comparisons. Additive **313669** completed **100 iterations / 1,262 Adam updates**
and its full-300 evaluation: **36.33% overall / 50.23% valid-only**; all 300
rollouts and verdicts are saved. Baseline replacement **315098** is queued with the scheduler-restore fix,
using the approved 4 H200 × 12h including full300 evaluation. MIG pilot
**315402 passed**: actor and SelectionARM ran on separate 18-GB slices, including
near-32k context, short/long-history selection, and two browser trajectories with
saved GPT-4.1 verdicts. Total allocated runtime across retries was **18m05s**;
the slices are released. This SFT-actor feasibility test does not yet validate
the additive checkpoint or native coverage collector on MIG.
[Results and probe fixes](RL_RUNTIME.md#mig-corrected-probe-315402).
[Agreed next experiments](ARM_INTEGRATION_PLAN.md#arm-additive-next-experiments-20260921):
first audit up to four labeled turns per failed trajectory without updating the
actor; separately test failure-only beta 0.5→1.0 while mixed-group beta stays 0.5.
Coverage pilot **315204** is running on g003: GPU restoration passed and the
native collection is progressing, with 4 H200 × 3h, 32 browsers and zero
optimizer updates. [Job inventory and remaining work](RL_RUNTIME.md#arm-job-inventory-20260921).
[Live jobs and completion reports](arm_results/rl_integration/live-status.html)
refresh every minute; checkpoint/health details are checked every 15 minutes.

Here `K=5` means **one executed actor response plus four counterfactual actor
responses** sampled from the same state with different deterministic seeds.
SelectionARM receives the five reasoning-plus-action candidates, chooses one,
and the permutation is inverted to identify whether it selected the executed
response or an alternative. SelectionARM itself does not generate these
candidates.

| Method | Iteration | Fixed-100 overall / valid-only | Full-300 overall / valid-only |
| --- | ---: | --- | --- |
| **Outcome-only baseline** | 20 (historical) | 25.0% / 35.21% | **31.67% / 40.95%** |
|  | 20 (same-day control) | — | **29.00% / 36.86%** |
|  | 30 | — | 32.00% / 38.71% |
|  | 40 | — | 33.33% / 43.29% |
|  | 50 | — | 35.00% / 44.87% |
|  | 60 | — | 35.00% / 45.65% |
|  | 70 | — | 34.33% / 44.98% |
|  | 80 | — | 38.00% / 49.78% |
|  | 90 | — | 33.67% / 45.50% |
| **Original bonus** | 20 | 26.0% / 35.62% | — |
|  | 30 | 27.0% / 36.49% | — |
|  | 40 | **31.00% / 44.93%** | — |
|  | 51 | — | 34.00% / 45.74% |
|  | 70 | — | 34.33% / 44.59% |
|  | 80 | — | 33.33% / 45.05% |
| **All-failure bonus** | 20 | 28.0% / 40.00% | 30.00% / 40.54% |
|  | 30 | 36.0% / 49.32% | 35.67% / 47.56% |
|  | 40 | 28.0% / 39.44% | 30.67% / 41.44% |
|  | 50 | — | 37.67% / 48.50% |
|  | 60 | — | 33.33% / 43.67% |
|  | 70 | — | 35.33% / 45.49% |
|  | 80 | — | 30.67% / 42.20% |
|  | 90 | — | **33.67% / 46.54%** |
| **Additive bonus** | 20 | 24.0% / 31.17% | 28.33% / 36.02% |
|  | 30 | 30.0% / 44.12% | 34.33% / 47.03% |
|  | 40 | 30.0% / 44.12% | 34.00% / 46.79% |
|  | 50 | — | 32.67% / 46.23% |
|  | 60 | — | 30.00% / 41.86% |
|  | 70 | — | 37.33% / 50.45% |
|  | 80 | — | 37.00% / 52.36% |
|  | 90 | — | **39.33% / 54.63%** |
|  | 100 | 31.00% / 47.69% | **36.33% / 50.23%** |
| **B: relaxed gate** | 20 | 29.00% / 42.03% | 33.67% / 44.30% |
| **C: relaxed gate + action credit** | 20 | 38.00% / 48.10% | 36.67% / 44.53% |

### All-failure ARM full-300 curve

All three [iteration-80 evaluations](RL_EVALUATION.md#arm-iter80-launch-20260919)
are complete, each with 300 per-task rollout archives and verdict records.
Additive is the strongest ARM endpoint by both rates, but none exceeds the
historical baseline's overall success. These are different-date evaluations
with different valid-task sets.

All-failure [iteration 90](RL_EVALUATION.md#arm-iter90-results-20260921) completed
at **33.67% overall / 46.54% valid-only**, matching the historical baseline's
overall rate. Different evaluation dates and valid-task sets limit comparison.
All 300 rollouts/verdicts are saved. Training finished at **100 / 1,242 Adam
updates**. Additive's corrected iteration-90 full-300
evaluation **313408** completed at **39.33% overall / 54.63% valid-only**,
with all 300 rollouts/verdicts saved. This is 5.67 percentage points above the
historical iteration-90 baseline overall, but is not a controlled same-day or
paired significance claim. Additive subsequently completed **100 / 1,262**;
its [iteration-100 evaluation](RL_EVALUATION.md#arm-additive-iter100-results-20260921)
is **36.33% / 50.23%**, down 3.00 / 4.40 percentage points from iteration90.
Baseline100 is still pending, so there is no matched iteration100 comparison yet.
Original remains at **85 / 1,002**; all-failure100 is trained but not evaluated.

![All-failure ARM full-300 evaluation curve](rl_results/arm_allfailure_full300.png)

The curve uses the completed full-300 evaluations at iterations 20, 30, 40,
50, 60, 70, 80, and 90. Iteration 20 is the disjoint fixed-100 plus 200-task merge; later
points are full-300 evaluations under the same local-browser/GPT-4.1 protocol.

### Baseline comparison

![Outcome-only baseline versus all-failure ARM](rl_results/baseline_vs_arm_allfailure_full300.png)

This comparison overlays the historical outcome-only baseline curve with the
original, all-failure, and additive ARM full-300 points, including additive100.
Additive's
[iteration-20 full-300 result](RL_EVALUATION.md#arm-additive-iter20-full300-20260920)
completed as job 307429 and is now included; it had been omitted from the docs.
It is a fresh 300-task evaluation, independent of the older fixed-100 result.
The first plotted original ARM point is iteration 51:
job 303459 loaded `iter_0000050`, previously mislabeled as iteration 50.

The earlier significance calculation was an exploratory unpaired proportion
test over aggregate counts. It is hidden from this summary because the archived
evaluations did not preserve aligned task-level verdict IDs, so it cannot support
a rigorous paired claim. Future comparisons should use paired McNemar or
bootstrap/permutation tests on shared task IDs, with a predeclared primary
comparison and multiple-comparison correction.

**Per-task records.** The corrected evaluator now saves one addressable file per
task under each evaluation's `rollouts/` directory. Each file contains the task
ID and all turns, including final reward, status, and termination reason; a
completed task with reward `1` or `0` supplies the success verdict, while an
aborted or unavailable task supplies the invalid outcome. This is sufficient to
construct aligned paired tests for new evaluations. Older runs only have their
aggregate metrics or lossless batch archives and may need re-evaluation before a
paired test.

- **Outcome-only baseline:** terminal outcome reward only; no ARM-labelled turns.
- **All-failure bonus:** admit eligible five-failure groups alongside ordinary mixed groups within the same 48-group budget; accepted failure groups displace mixed-group slots.
- **Original bonus:** apply the ARM term only to eligible turns inside ordinary mixed outcome groups.
- **Additive bonus:** retain the ordinary mixed batch and add an independently normalized buffer of up to eight eligible all-failure groups.

**What “usable ARM label” means.** An all-failure group has five trajectories for
the same task, all with valid zero terminal outcome. It is admitted when at least
one turn in those five trajectories completes candidate generation and selection
with valid provenance and a parseable selector result. This is label availability,
not proof that the selector chose the objectively correct action. Candidate 0 is
the actor's executed action, so that labeled turn receives `+0.8` when ARM picks
candidate 0 and `-0.2` when ARM prefers one of the four alternatives; unlabeled
turns receive no ARM term.

**Illustration of the three online runs.** For one task, imagine five actor
trajectories all ending in a valid failure (`0`):

```text
five failure trajectories:  A(0)  B(0)  C(0)  D(0)  E(0)
usable labels:             A:t3, D:t7
ARM choices:               A:t3 -> candidate 4 (-0.2)
                           D:t7 -> candidate 0 (+0.8)

Original bonus:  ordinary mixed group ──┐
                                        ├─ ARM on eligible turns only
All-failure:   mixed + five-failure groups ──┘ (48 groups total)
Additive:      ordinary mixed groups + up to eight such failure groups
               └─ separate normalized loss terms for the two sources
```

Thus the all-failure run uses the same terminal outcome reward as usual, but
admits otherwise discarded zero-outcome groups when they contain at least one
usable turn-level ARM signal. The additive run keeps those groups alongside the
ordinary mixed collection instead of replacing it.

The full-300 baseline is the matched outcome-only OpenWebRL checkpoint under
the same local-browser/GPT-4.1 RL evaluation protocol. The all-failure full-300
result is a disjoint merge of its fixed-100 cohort and 200-task complement.

The iteration-20 baseline rows show both controls: the historical run is the
default comparison, while job `299148` is the fresh same-day control (87/300
successes, 236 valid, 64 invalid). The same-day rerun is retained to expose
live-web variance; the historical value remains the canonical comparison used
by the earlier evaluation tables.
Unless explicitly marked “same-day control,” the rows above use the historical
evaluation runs.

The original-bonus iteration-40 fixed-100 result is from job `299156` and has
31 successes, 69 valid tasks, and 31 invalid tasks; all 100 task-addressable
rollouts were saved.

The separate inference-time baseline is 90/300 with 267 valid tasks, or 30.0%
overall and 33.7% valid-only, under the historical o4-mini protocol; it should
not be used for the RL comparison.

**Conclusion:** The RL integrations are operational and stable, but the current
measurements show baseline-level performance rather than a reliable gain. The
all-failure result is the most complete measurement; its 300-task result is a
disjoint merge of a fixed 100-task cohort and a 200-task complement.

## Overall takeaway

ARM's strongest validated use is **inference-time candidate selection**. Offline
distillation and the first online turn-bonus integrations have not yet matched
that gain. The next RL experiments should therefore test whether ARM can rescue
fresh actor failures or improve candidate coverage, while preserving a matched
outcome-only control.
