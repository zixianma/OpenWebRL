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
usually 8--10% because the per-trajectory pending-label cap, selector or
transport failures, timeouts, and later removal of invalid/non-trainable turns
all reduce the number of usable labels. For example, original ARM iteration 45
attempted 263 selector requests, had 278 unavailable labels, and applied 139
usable labels to 1,454 retained turns (9.56%).

Here `K=5` means **one executed actor response plus four counterfactual actor
responses** sampled from the same state with different deterministic seeds.
SelectionARM receives the five reasoning-plus-action candidates, chooses one,
and the permutation is inverted to identify whether it selected the executed
response or an alternative. SelectionARM itself does not generate these
candidates.

| Method | Iteration | Fixed-100 overall / valid-only | Full-300 overall / valid-only |
| --- | ---: | --- | --- |
| **Outcome-only baseline** | 20 | 25.0% / 35.21% | 31.67% / 40.95% |
|  | 30 | — | 32.00% / 38.71% |
|  | 40 | — | 33.33% / 43.29% |
| **Original bonus** | 20 | 26.0% / 35.62% | — |
|  | 30 | 27.0% / 36.49% | — |
|  | 40 | **Pending checkpoint** | — |
| **All-failure bonus** | 20 | 28.0% / 40.00% | 30.00% / 40.54% |
|  | 30 | 36.0% / 49.32% | 35.67% / 47.56% |
|  | 40 | 28.0% / 39.44% | 30.67% / 41.44% |
| **Additive bonus** | 20 | 24.0% / 31.17% | — |
|  | 30 | 30.0% / 44.12% | 34.33% / 47.03% |
|  | 40 | 30.0% / 44.12% | 34.00% / 46.79% |

### Same-iteration significance checks

The archived fixed-100 evaluations retain aggregate counts rather than aligned
per-task verdict IDs, so these are two-sided unpaired two-proportion tests. The
reported p-values are exploratory and do not correct for the multiple ARM and
iteration comparisons; paired McNemar tests would be preferable for future
evaluations with stable task-level receipts.

| ARM variant and iteration | Cohort | Δ overall vs baseline | p | Δ valid-only vs baseline | p |
| --- | --- | ---: | ---: | ---: | ---: |
| Original bonus · 20 | Fixed-100 | +1.00 pp | 0.871 | +0.41 pp | 0.960 |
| All-failure bonus · 20 | Fixed-100 | +3.00 pp | 0.631 | +4.79 pp | 0.557 |
| Additive bonus · 20 | Fixed-100 | −1.00 pp | 0.869 | −4.04 pp | 0.602 |
| All-failure bonus · 20 | Full-300 | −1.67 pp | 0.659 | −0.41 pp | 0.930 |
| All-failure bonus · 30 | Full-300 | +3.67 pp | 0.343 | +8.85 pp | 0.052 |
| Additive bonus · 30 | Full-300 | +2.33 pp | 0.544 | +8.32 pp | 0.070 |
| All-failure bonus · 40 | Full-300 | −2.67 pp | 0.484 | −1.85 pp | 0.691 |
| Additive bonus · 40 | Full-300 | +0.67 pp | 0.863 | +3.50 pp | 0.456 |

No overall comparison is conventionally significant at 0.05. The iteration-30
all-failure valid-only result is close to that threshold, but it is not a
paired test and should not be treated as evidence of a reliable improvement.

- **Outcome-only baseline:** terminal outcome reward only; no ARM-labelled turns.
- **All-failure bonus:** replace the ordinary mixed collection with eligible groups containing five valid actor failures and at least one usable ARM label.
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
All-failure:   five-failure group  ────┘ (replaces the ordinary group mix)
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
