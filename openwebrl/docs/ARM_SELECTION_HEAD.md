# ARM selection head: choosing actions from the actor's own hidden states

**Goal: pick among the actor's five candidate actions without a second model.** Piotr SelectionARM runs a separate 4B selector over the screenshot and all candidates. The selection head reads the hidden states the actor already computes while generating.

## Headline results

**On Online-Mind2Web, WebVoyager and DeepShop (1,045 tasks, each benchmark weighted equally), a 22.6 M-parameter head on the frozen SFT-4B actor matches Piotr SelectionARM's 4.44 B selector when trained on Piotr's data (+1.4 pp [−2.1, +4.9]). Trained on observed-outcome or lookahead K = 3 labels for only 1,380 branch states, it is level with Piotr (−0.8 and −0.9 pp, intervals spanning 0) at 56–65% of his compute per episode. The training labels matter most; selector size does not, and an 8B actor without selection gains +3.3 pp [−0.3, +6.9].**
- **Setup:** every task of each benchmark, one episode per task, the same tasks for every method. Actor OpenWebRL-4B-SFT unless noted, T 1.0 / top-p 0.9, 30 steps, five candidates per decision; one H200 per method and benchmark. Judges: o4-mini for Online-Mind2Web (retired 2026-10-23; [re-judged with gpt-5.6-terra](#selection-head-data-scale-20261023)); GPT-4o for WebVoyager and DeepShop. Paired 95% bootstrap over tasks.

| Actor | Selector | Training data: states / panels / labels | Online-Mind2Web<br>valid-only | WebVoyager<br>valid-only | DeepShop<br>valid-only | Mean<br>overall | Mean<br>valid-only | − SFT 4B, pp<br>[95% CI] |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 4B | — (1 sample) | — | 39.6% | 45.0% | 34.2% | 38.2% | 39.6% | — |
| SFT 8B | — (1 sample) | — | 39.9% | 49.7% | 39.9% | 41.5% | 43.2% | +3.3 [−0.3, +6.9] |
| SFT 4B | Piotr SelectionARM (4B LoRA) | 2,692 (Piotr's) / 35,916 of 5 sampled / GPT-5.5, before | 46.8% | **53.8%** | 41.9% | 44.9% | 47.5% | **+6.7 [+3.0, +10.4]** |
| SFT 4B | **Head (22.6 M)** | 2,692 (Piotr's) / 35,916 of 5 sampled / GPT-5.5, before | **49.4%** | 53.4% | **44.9%** | **46.3%** | **49.2%** | **+8.1 [+4.4, +11.8]** |
| SFT 4B | **Head (22.6 M)** | 1,380 (branch) / 1,380 of 2–5 distinct / observed outcome | 42.2% | 53.0% | 42.6% | 44.1% | 45.9% | **+5.9 [+2.5, +9.4]** |
| SFT 4B | **Head (22.6 M)** | 1,380 (branch) / 1,380 of 2–5 distinct / Luna, 3 steps after | 43.1% | 52.0% | 42.7% | 44.0% | 45.9% | **+5.8 [+2.1, +9.5]** |
| SFT 4B | Head (22.6 M) | 1,380 (branch) / 1,380 of 2–5 distinct / Luna, 1 step after | 38.6% | 50.5% | 38.8% | 40.4% | 42.6% | +2.2 [−1.5, +5.8] |
| SFT 4B | Head (22.6 M) | 1,380 (branch) / 1,380 of 2–5 distinct / Luna, before | 36.9% | 47.1% | 34.2% | 37.6% | 39.4% | −0.7 [−4.2, +2.8] |

| Actor | Selector | Labels | Params (actor + selector) | PFLOP per step | PFLOP per episode | Model time per episode | Steps per episode |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| SFT 4B | — | — | 4.44 B | 0.019 | 0.28 | 42 s | 14.5 |
| SFT 8B | — | — | 8.77 B | 0.037 | 0.56 | 51 s | 15.0 |
| SFT 4B | Piotr SelectionARM | GPT-5.5, before | 4.44 B + 4.44 B | 0.098 | 1.53 | 143 s | 15.4 |
| SFT 4B | Head | GPT-5.5, before | 4.44 B + 22.6 M | 0.063 | 0.92 | 131 s | 14.6 |
| SFT 4B | Head | Observed outcome | 4.44 B + 22.6 M | 0.062 | 0.86 | 109 s | 13.8 |
| SFT 4B | Head | Luna, K = 3 | 4.44 B + 22.6 M | 0.062 | 1.00 | 142 s | 16.0 |
| SFT 4B | Head | Luna, K = 1 | 4.44 B + 22.6 M | 0.062 | 1.05 | 149 s | 16.8 |
| SFT 4B | Head | Luna, before execution | 4.44 B + 22.6 M | 0.062 | 1.05 | 155 s | 16.8 |

- **Columns.** "Overall" counts invalid episodes as failures; "valid-only" divides successes by valid episodes. "Mean" weights each benchmark equally (one third), for success and cost alike. Differences are paired on the same tasks: 95% bootstrap, tasks resampled within each benchmark; valid-only differences (tasks valid under both) agree within 1.1 pp. Parameters are those loaded at inference; the head's 22.6 M are its only trained weights. Model time is decision time summed over an episode, excluding browser execution; times come from live runs on different nodes.
- **Against Piotr SelectionARM** (mean overall, pp [95% CI]): Piotr-data head +1.4 [−2.1, +4.9], outcome −0.8 [−4.5, +2.8], K = 3 −0.9 [−4.5, +2.7], K = 1 −4.5 [−8.3, −0.8], pre-execution −7.4 [−11.1, −3.8], SFT 8B −3.4 [−7.3, +0.4].
- **Methods.** Every selector picks one of five candidates the frozen actor samples per step; "branch" data are the SFT-actor branching states collected for this project ([collection and labels](ARM_FORMULATIONS.md#lookahead-labels-current)), "Piotr's" the GPT-5.5-labelled panels of Piotr's release. SFT rows take a single sample. Training data reads states / panels (with the actions in each panel) / labels. Differences are on the mean overall rate. Piotr's panels are five sampled actions each (about 13 per state), duplicates included, each with one GPT-5.5 pick made before execution (from the screenshot and candidates only). Branch panels are one per state holding its 2–5 distinct SFT-proposed actions, reshuffled each epoch.
- **Labels.** The four label heads share one architecture and the same 1,380 SFT-actor training states; only the label differs. Outcome = the action with the best success on continuation draws 0–2 (draws 3–7 are kept for offline scoring); K = 1 / K = 3 = Luna judging one or three steps after execution; pre-execution = Luna before execution. Online-Mind2Web is where the branch labels fall short: no label head clearly beats SFT there, while the Piotr-data head does (+7.3 pp [+1.7, +13.0]). Details: [ablations](#selection-head-sft-actions-20261021).
- **Why time barely drops.** Generating five candidates dominates each 8–9 s decision; the head saves 36% of compute per step but only about 0.5 s per decision. Fewer candidates help more: **with three candidates on Online-Mind2Web, Piotr and the Piotr-data head lose no measurable success and save 26–33% of model time** ([details](#selection-head-n3-8b-20261022)).
- **Scaling the actor is not a substitute for selection.** SFT 8B gains on WebVoyager and DeepShop but not on Online-Mind2Web (+0.0 pp), and the Piotr-data head beats it by +4.8 pp [+1.0, +8.5] (Piotr +3.4 [−0.4, +7.3]) at 1.6× its compute per episode (Piotr 2.7×) ([details](#selection-head-n3-8b-20261022)).
- **Earlier heads.** The K = 1 head trained on 1,908 SFT + RL states (October 18) scores 43.9% mean overall, +5.7 pp [+2.0, +9.4] over SFT 4B and −1.0 [−4.5, +2.5] against Piotr ([details](#selection-head-crossbench-full-20261018)). At 100 steps, SFT gains +5.8 pp and Piotr +4.8 pp, and Piotr's lead holds ([details](#selection-head-100-steps-20261019)); the heads have not been run at 100 steps.
- **Evidence:** [label heads](arm_results/rl_integration/selection-head-sft-actions-full-20261021.json), [SFT 8B and N = 3](arm_results/rl_integration/selection-n3-sft8b-full-20261021.json), [benchmark means and differences](arm_results/rl_integration/selection-mean-benchmark-diffs-20261022.json). Plot: [final results](ARM_FINAL_RESULTS.md#final-selection-head-ablations-20261021).

**Earlier, Online-Mind2Web on identical hardware (3 × 300 tasks, T 0.7; head trained on Piotr's teacher labels): the selection head matches Piotr SelectionARM's success with 36% less compute and 11% lower latency per decision.**
- **Setup:** Online-Mind2Web, 300 tasks, three independent repetitions per method.
- **Hardware:** every method ran on one H200, with CUDA graphs on and any separate selector on the same GPU.

| Method | Success, mean of 3 × 300 | − SFT 4B, pp [95% CI] | − Piotr ARM, pp [95% CI] | PFLOP per decision | Seconds per decision |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Selection head (ours)** | **38.4%** | **+4.8 [+1.4, +8.2]** | **−1.0 [−4.0, +2.1]** | **0.060** | **6.72** |
| Piotr SelectionARM | 39.4% | +5.8 [+2.6, +9.0] | — | 0.094 | 7.56 |
| Kev-style pass | 39.7% | +6.0 [+2.8, +9.2] | +0.2 [−2.8, +3.2] | 0.175 | 16.60 |
| Luna (API) | 40.9% | +7.2 [+4.1, +10.3] | +1.4 [−1.8, +4.6] | ≥0.058 + API | 9.78 |
| SFT 8B, 1 sample | 34.6% | +0.9 [−2.4, +4.2] | −4.9 [−8.6, −1.3] | 0.036 | 3.20 |
| SFT 4B, 1 sample | 33.7% | — | −5.8 [−9.0, −2.6] | 0.019 | 2.44 |
| Action-type prior | 28.9% | −4.8 [−8.1, −1.6] | −10.6 [−13.9, −7.3] | 0.057 | 7.15 |

- **Every learned selector beats a single sample by 5–7 pp, and none clearly beats another.** The head's −1.0 pp against Piotr ARM is well inside its interval.
- **Scaling the actor does not substitute for selection.** SFT 8B, with one sample, is level with SFT 4B at twice the compute.
- **Cost.**
  - The head's decisions cost 64% of Piotr's compute: only the actor's work, since its selector adds under 1%.
  - It is the fastest five-candidate method, 0.4 s behind five samples with no selector.
  - The slow selectors slow down further under live load: Kev-style's decisions take 16.6 s live against 8.7 s offline.
- **Across benchmarks, 100-task subsets at T 0.7** ([details](#selection-head-crossbench-20261017)): on the same 100 tasks per benchmark, the lookahead K = 1 head trained on 1,908 SFT + RL branch states beats the 840-state version by +7.0 pp [+0.7, +13.3] (300 tasks). It is level with SFT (+1.7 pp [−4.7, +7.7]) and with the head trained on Piotr's data; no selector beats SFT on DeepShop.
- **Branching states with held-out outcomes.**
  - No teacher-trained selector clearly beats a uniform pick, while choosing by observed outcomes does: +5.32 pp [+3.09, +7.62] ([details](#selection-head-outcomes-20261010)).
  - Among small students, only **lookahead-labelled** ones move above uniform (+2.4 to +3.0 pp) ([label types](#selection-head-label-types-20261011)).

<a id="selection-head-om2w-plot"></a>

![Online-Mind2Web on identical hardware: mean task success over three repetitions versus compute per decision (left) and versus live seconds per decision (right), one point per method with 95% task-bootstrap intervals](arm_results/rl_integration/perf_vs_cost_samehw.png)

- **Bars.** They show each method's own 95% task-bootstrap interval (about ±4.5 pp), which is wide because tasks vary so much in difficulty. The paired differences in the table compare methods on the same tasks and are tighter.
- **Luna (hollow).** Its point shows the actor's compute only.
- **Intervals.** They resample tasks and keep each task's three repetitions together.
- **Repetitions and protocol.** The run-to-run SD across repetitions is 0.5–2.4 pp. Invalid episodes count as failures.
- **Timing.** The methods of a repetition were collected over about six hours as GPUs freed up, not at the same moment.
- **Details and evidence:** [same-hardware runs](#selection-head-samehw-20261012), [aggregate](arm_results/rl_integration/selectors-samehw-20261012.json).

<a id="selection-head-architecture"></a>

## How the selection head works

![Piotr SelectionARM runs a separate 4B selector over the screenshot and all five candidates; the selection head is a small MLP on hidden states captured while the actor generates](arm_results/methods/selection_head_overview.svg)

![Selection head: (A) at inference the frozen actor samples five candidates, their hidden states at layers 18/27/36 are captured during generation and a small head scores them; (B) in training the frozen actor encodes teacher-labelled panels once and only the head is trained](arm_results/methods/selection_head_method.svg)

- **Inference.** The frozen SFT actor samples five candidates. While it generates, forward hooks inside its sglang server capture the residual stream at layers 18, 27 and 36. The head pools these per candidate (reasoning mean, action mean, end token) together with the prompt's last token and scores the candidates. The highest-scoring candidate is executed.
  - The *pointwise* head of the same-hardware runs scores each candidate alone with a small MLP.
  - The *listwise* heads (all later heads) add a 2-layer transformer across the five candidates. Details: [training recipe](#selection-head-training-recipe).
- **Training.** The frozen actor encodes the Piotr release's GPT-5.5-labelled panels once (2,692 training states), and only the head is trained, with soft cross-entropy on the teacher's choice. The actor's weights never change. A detailed figure, including the top-layer adapter variant, is in the method section.

<details>
<summary>Earlier summary: teacher agreement, outcome data and adaptation</summary>

**No selector trained on teacher labels reliably picks actions that succeed more often than a uniform pick. Choosing by observed continuation outcomes does.** On 138 branching states with eight SFT continuation draws per action (120 task groups), each selector's chosen action is scored on held-out draws. Choosing the action that succeeded most often on five draws reaches **36.99%** against **31.67%** for a uniform pick: **+5.32 pp [+3.09, +7.62]**. Piotr SelectionARM reaches 34.15% (+2.48 pp [−0.14, +5.21]). Every actor-based selector lies between 31.34% and 33.70%, from −0.33 to +2.03 pp relative to uniform, and none differs from uniform at the 95% level.

**Agreement with teachers does not track action quality.** The Kev-style decision pass on the actor's weights, trained to step 600, matches held-out GPT-5.5 choices best of any actor-based selector: 67.1%, versus 65.4% for the frozen-feature head. On the original 124 states it matches an independent teacher (Luna before execution) on 66.1% of judgments, versus 73.2% for Piotr ARM. Its chosen actions succeed on 32.79% of held-out draws, +1.12 pp [−1.38, +3.75] over uniform.

**Supervising the selector with outcomes needs more states than exist now.** 188 states have eight draws for all five actions: 9,720 outcomes in 151 task groups, 14× fewer states than the 2,692 teacher-labelled training states ([data comparison](#selection-head-outcome-data-20261010)). A pre-registered five-fold [adaptation pilot](#selection-head-outcome-adaptation-20261010) on 138 of these states overfits: it fits training folds but does not beat uniform on held-out tasks. Ridge probes, a combined teacher-plus-outcome loss and selector combinations also find [no held-out gain](#selection-head-early-signal-20261010). Confirming a 3 pp selector difference needs about 200 states with eight draws, and confirming Piotr's ~2.5 pp gain over uniform about 300.

</details>

<a id="selection-head-outcome-objectives-20261025"></a>

## Outcome labels as advantages rather than a single best action (2026-10-25)

**Training on how much better each action did, not only which did best, does not clearly help offline: centred advantage regression is +0.7 pp [−0.3, +1.9] over the current best-action target, the other two objectives are level.** Same 1,380 SFT-action train states, joint transformer head (22.6 M), 20 epochs, seeds 0–2; every objective reads the same draws 0–2 (valid draws only), and the 348 test states (86 tasks) are scored on held-out draws 3–7. Live Online-Mind2Web runs of all three new heads (seed 0) are in progress.

| Objective | Target per action (from draws 0–2) | Held-out success, distinct actions | Held-out success, fixed five | − best action, pp [95% CI] | States with a different pick |
| --- | --- | ---: | ---: | ---: | ---: |
| Best action, soft CE (current) | 1 on the best rate, ties split | 36.97% | 35.59% | — | — |
| Soft CE, softmax(rate / 0.25) | smoothed ranking | 37.14% | 35.51% | +0.17 [−0.75, +1.16] | 65 of 348 |
| **Centred advantage MSE** | score − panel mean regresses rate − panel mean | **37.67%** | **36.10%** | **+0.69 [−0.30, +1.85]** | 85 of 348 |
| Count-weighted binomial BCE | sigmoid(score) fits the rate, weight = valid draws | 36.98% | 35.98% | +0.01 [−1.32, +1.59] | 90 of 348 |
| *References* | uniform over actions 36.26%, one SFT sample of the fixed five 35.65%, oracle 47.96% | | | | |

- **Columns.** Success is the held-out rate of the picked action, averaged over states and over the three seeds (seed-to-seed range ±0.5 pp). Distinct actions = one representative per action; fixed five = the state's first five SFT proposals, as served live. Differences are paired per state on the seed-averaged pick, task-clustered bootstrap.
- **Why three draws limit every objective.** A rate from draws 0–2 can only be 0, ⅓, ⅔ or 1. In the training state `bff8e3b6…` (arXiv, find a quantum-computing paper) the draw-0–2 winner, opening the subject dropdown (2/3), never succeeds on draws 3–7 (0/5), while the "Quantum Physics" link (1/3) is the held-out best (2/5). The best-action target trains all mass toward the dropdown; the advantage target (+0.32, +0.05, −0.35, −0.02 for the four actions) keeps the other actions' relative order; the binomial target treats 2/3 versus 1/3 as weak evidence.
- **Binomial BCE changes the most picks but gains nothing**, plausibly because it also spends capacity predicting each state's difficulty, which never affects the pick; the advantage loss removes it by construction.
- **Offline gaps are small for every head here** (about +1 pp over uniform against a 12 pp oracle gap), and offline differences have not predicted live ones: the current outcome head was +0.7 pp over uniform offline but +5.9 pp over SFT live.
- [Evidence](arm_results/rl_integration/selection-outcome-objectives-20261025.json) · [analysis script](arm_results/rl_integration/selection-outcome-objectives-20261025-analysis.py) · trainer `scripts/train_head_round.py --source outcome_soft | outcome_rate --loss advmse | binom`, losses `advantage_mse_loss` and `binomial_rate_loss` in `openwebrl/selection_head.py`. Runtime `selection-head-20261008/sft-actions-20261021/advantage-20261025/` (heads, configs and logs saved per seed).

<a id="selection-head-data-scale-20261023"></a>

## Teacher-data scaling, and the Online-Mind2Web judge change (2026-10-23/24)

**More of Piotr's GPT-5.5 panels does not make a better head. On all 1,045 tasks (each benchmark weighted equally), heads trained on 254 or 501 of Piotr's states gain +5.0 and +5.6 pp over one SFT sample, as much as our 1,380-state K = 3 and outcome heads; at 2,587 states the gain is +2.8 pp (2,587 − 254: −2.2 [−5.9, +1.7] pp). Offline, the gain over one SFT sample stays at +2.3 to +3.0 pp at every size.** Validation agreement with GPT-5.5 rises with data (58% → 66%), so the head learns the teacher better without picking better actions. At equal size, one Piotr panel per state (1,393 states) is below our 1,380 branch states on K = 3 labels live (**−4.3 [−8.1, −0.5]** pp) and level offline (+2.0 vs +2.0 pp). The new heads are flat on Online-Mind2Web and WebVoyager and lower on DeepShop (36.0% at 254 states, 31.3% at 2,587) than the earlier head's 44.0% on 10-18, but **DeepShop itself drifted: same-day reruns on 10-24 dropped SFT 4B from 34.0% to 28.0% and the earlier head from 44.0% to 34.0%**, and against that same-day earlier head the new heads differ by −6.0 to +2.7 pp (all intervals spanning 0). The DeepShop columns, and therefore the three-benchmark means, are not comparable across run dates.
- **Judge change.** OpenAI retired o4-mini on 2026-10-23 (both `o4-mini` and `o4-mini-2025-04-16` return `model_not_found`). Online-Mind2Web is now judged by **gpt-5.6-terra** with the same AgentTrek prompt, final screenshot and parser. All earlier Online-Mind2Web references below are **re-judged** by replaying each episode's saved judge request with only the model swapped. Terra is stricter: it agrees with o4-mini on 88–95% of episodes and lowers success by up to 6.7 pp ([below](#selection-head-terra-rejudge-20261023)). Online-Mind2Web numbers elsewhere in this document are o4-mini; WebVoyager and DeepShop still use GPT-4o.
- **Setup.** Joint transformer head (soft CE) on SFT-4B features, trained only on Piotr's training panels, seed 0. Subsets add whole tasks in a fixed random order (seed 20261023) until the state count is reached, so they are nested; the 86 tasks of our offline test set are excluded throughout. Recipe as the Piotr-data head: batch 256, 20 epochs, epoch chosen on Piotr's validation split (280 states). The one-panel row keeps one random panel per state and uses our branch recipe (batch 64, last of 20 epochs). Live: Online-Mind2Web 300 tasks, the [headline](#headline-results) task order, T 1.0 / top-p 0.9, 30 steps, five candidates, one H200 per run.
- **Offline metric (now the primary one).** On our 348 test states (86 tasks), the head picks among the **first five SFT proposals** by draw index, duplicates kept, as in live decoding; the baseline is one SFT sample from the same five (35.65%; the best of the five is 46.74%). Success on held-out continuation draws 3–7, valid draws only; 95% bootstrap over tasks.

| Training data | States | Panels | Tasks | GPT-5.5 validation agreement | Offline: fixed five | Offline − SFT sample, pp [95% CI] | OM2W valid | OM2W successes | OM2W overall | OM2W valid-only | OM2W − SFT 4B, pp [95% CI] | OM2W − Piotr ARM, pp [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Piotr's panels | 254 | 3,302 | 31 | 58.1% | 37.91% | +2.3 [+0.3, +4.5] | 264 | 109 | 36.3% | 41.3% | +4.7 [−1.3, +10.7] | −2.7 [−8.0, +2.7] |
| Piotr's panels | 501 | 6,527 | 60 | 60.7% | 38.70% | +3.0 [+1.4, +4.9] | 262 | 106 | 35.3% | 40.5% | +3.7 [−2.3, +9.3] | −3.7 [−9.0, +2.0] |
| Piotr's panels | 1,011 | 13,263 | 123 | 63.1% | 38.05% | +2.4 [+0.2, +4.6] | 260 | 107 | 35.7% | 41.2% | +4.0 [−2.0, +10.0] | −3.3 [−8.3, +1.7] |
| Piotr's panels, all | 2,587 | 34,518 | 344 | 66.2% | 38.31% | +2.7 [+0.6, +4.8] | 258 | 100 | 33.3% | 38.8% | +1.7 [−4.3, +7.7] | **−5.7 [−10.7, −0.7]** |
| Piotr's panels, one per state | 1,393 | 1,393 | 184 | 59.2% | 37.65% | +2.0 [−0.0, +4.3] | 263 | 98 | 32.7% | 37.3% | +1.0 [−4.7, +6.3] | **−6.3 [−11.3, −1.3]** |
| *Re-judged references (run 2026-10-18 to 10-21)* | | | | | | | | | | | | |
| SFT 4B, one sample | — | — | — | — | 35.65% | — | 275 | 95 | 31.7% | 34.5% | — | **−7.3 [−13.0, −1.7]** |
| Piotr SelectionARM (4B LoRA) | 2,692 | 35,916 | — | — | — | — | 263 | 117 | 39.0% | 44.5% | **+7.3 [+1.7, +13.0]** | — |
| Earlier Piotr-data head | 2,692 | 35,916 | — | — | — | — | 265 | 113 | 37.7% | 42.6% | **+6.0 [+0.7, +11.3]** | −1.3 [−6.3, +3.7] |
| Ours, K = 3 labels | 1,380 | 1,380 | — | — | 37.68% | +2.0 [+0.3, +3.9] | 267 | 99 | 33.0% | 37.1% | +1.3 [−4.0, +6.3] | **−6.0 [−11.3, −1.0]** |
| Ours, observed outcome | 1,380 | 1,380 | — | — | 35.31% | −0.3 [−2.1, +1.4] | 270 | 94 | 31.3% | 34.8% | −0.3 [−5.3, +4.7] | **−7.7 [−13.0, −2.3]** |
| Ours, K = 1 labels | 1,380 | 1,380 | — | — | 37.74% | +2.1 [+0.6, +3.7] | 264 | 93 | 31.0% | 35.2% | −0.7 [−5.7, +4.3] | **−8.0 [−13.0, −3.0]** |
| Ours, pre-execution labels | 1,380 | 1,380 | — | — | 36.76% | +1.1 [−0.6, +2.9] | 268 | 90 | 30.0% | 33.6% | −1.7 [−6.7, +3.3] | **−9.0 [−14.0, −4.0]** |

**All three benchmarks** (1,045 tasks; Online-Mind2Web judged by gpt-5.6-terra, WebVoyager and DeepShop by GPT-4o; mean weights each benchmark 1/3, bootstrap resamples tasks within each benchmark):

| Training data | States | OM2W overall | WebVoyager overall | DeepShop overall | Mean overall | Mean valid-only | − SFT 4B, pp [95% CI] | − Piotr ARM, pp [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Piotr's panels | 254 | 36.3% | 52.6% | 36.0% | 41.6% | 44.1% | **+5.0 [+1.1, +8.9]** | −2.6 [−6.3, +1.1] |
| Piotr's panels | 501 | 35.3% | 54.8% | 36.7% | 42.3% | 44.5% | **+5.6 [+1.7, +9.4]** | −2.0 [−5.8, +1.8] |
| Piotr's panels | 1,011 | 35.7% | 53.5% | 31.3% | 40.1% | 42.6% | +3.5 [−0.2, +7.2] | **−4.1 [−7.6, −0.7]** |
| Piotr's panels, all | 2,587 | 33.3% | 53.8% | 31.3% | 39.5% | 41.7% | +2.8 [−1.0, +6.6] | **−4.8 [−8.2, −1.3]** |
| Piotr's panels, one per state | 1,393 | 32.7% | 53.3% | 28.0% | 38.0% | 40.5% | +1.3 [−2.5, +5.2] | **−6.3 [−9.9, −2.6]** |
| *References (run 2026-10-18 to 10-21)* | | | | | | | | |
| SFT 4B, one sample | — | 31.7% | 44.4% | 34.0% | 36.7% | 37.9% | — | **−7.6 [−11.3, −4.0]** |
| Piotr SelectionARM (4B LoRA) | 2,692 | 39.0% | 52.4% | 41.3% | 44.3% | 46.7% | **+7.6 [+4.0, +11.3]** | — |
| Earlier Piotr-data head | 2,692 | 37.7% | 51.3% | 44.0% | 44.3% | 47.0% | **+7.6 [+3.9, +11.3]** | +0.1 [−3.5, +3.7] |
| Ours, K = 3 labels | 1,380 | 33.0% | 51.1% | 42.7% | 42.2% | 43.9% | **+5.6 [+2.0, +9.2]** | −2.0 [−5.5, +1.6] |
| Ours, observed outcome | 1,380 | 31.3% | 52.3% | 42.0% | 41.9% | 43.5% | **+5.2 [+1.8, +8.6]** | −2.4 [−6.1, +1.3] |
| Ours, K = 1 labels | 1,380 | 31.0% | 49.2% | 38.0% | 39.4% | 41.5% | +2.7 [−0.9, +6.4] | **−4.8 [−8.6, −1.1]** |
| Ours, pre-execution labels | 1,380 | 30.0% | 45.7% | 34.0% | 36.6% | 38.3% | −0.1 [−3.7, +3.5] | **−7.7 [−11.4, −4.1]** |

- **Columns.** OM2W = Online-Mind2Web, 300 tasks, gpt-5.6-terra judge. "Overall" counts invalid episodes as failures; differences are paired by task on the overall rate (95% bootstrap). The earlier Piotr-data head trained on all 2,692 of Piotr's states (no test-task exclusion; [recipe](#selection-head-sft-actions-20261021)). "Ours" rows are the [label ablation](#selection-head-sft-actions-20261021) heads; their offline numbers are the same fixed-five metric.
- **Within the scaling runs (pp [95% CI]), Online-Mind2Web / three-benchmark mean:** 2,587 − 254 states −3.0 [−8.3, +2.0] / −2.2 [−5.9, +1.7]; 1,011 − 254 −0.7 [−5.3, +4.0] / −1.5 [−5.2, +2.2]; one-panel 1,393 − all 2,587 −0.7 [−5.0, +3.7] / −1.5 [−5.0, +2.1]. One-panel − ours outcome +1.3 [−4.0, +6.3] / **−3.9 [−7.4, −0.3]**.
- **Caveats.** One seed per size. The new runs and the references ran on different days, and sites change: the new all-data head scores −4.3 pp [−10.0, +1.3] against the earlier Piotr-data head trained on nearly the same data, about as large as any scaling effect here. On the three-benchmark mean the gap is **−4.8 [−8.5, −1.1]** pp, driven by DeepShop (47 vs 66 of 150 successes; WebVoyager 53.8% vs 51.3% favours the new head). DeepShop episodes were equally valid (148 vs 147 of 150) and ended normally. **Same-day DeepShop reruns (10-24, jobs 385776–385777, runtime `evaluations/deepshop-sameday-20261024/`) confirm a site/day effect:** SFT 4B 51 → 42 of 150 (−6.0 pp [−14.0, +2.0]), earlier Piotr-data head 66 → 51 (−10.0 [−19.3, +0.0]). Against the same-day earlier head (34.0%), the new heads on DeepShop: 254 states +2.0 [−8.0, +12.0], 2,587 states −2.7 [−12.0, +6.7], one panel per state −6.0 [−16.0, +4.0]; earlier head − same-day SFT +6.0 [−3.3, +15.3]. So the new and earlier Piotr-data heads are indistinguishable live as well as offline, and the label-head references (DeepShop run 10-21) likely benefit from the earlier, easier DeepShop. The two heads are effectively the same model: same architecture, loss and recipe, and on our 348 test states they pick the same action in 85% of fixed-five panels and score 38.31% vs 37.99% (+0.3 pp [−0.3, +1.1]; CPU job 384787, runtime `data-scale-20261023/compare-old/`). The earlier head itself ranged 36.3–39.7% over three o4-mini repetitions ([same-hardware runs](#same-hardware-runs-three-repetitions-of-seven-methods)). Read the scaling rows against each other; read differences against the references with that in mind.
- **Offline, distinct-action view** (head − uniform over distinct actions): +1.4, +2.7, +2.8, +3.5 pp for 254 → 2,587 states, +2.1 for one panel per state; ours outcome +0.7, K = 3 +0.4, K = 1 +1.0, pre-execution +0.2. Only this view, which removes duplicates, shows a weak trend with data.
- [Evidence](arm_results/rl_integration/selection-data-scale-20261023.json) · [analysis script](arm_results/rl_integration/selection-data-scale-20261023-analysis.py). Runtime `selection-head-20261008/data-scale-20261023/` (heads, `fixed-five.json` beside each) and `evaluations/data-scale-20261023/` (Online-Mind2Web jobs 383632–383636, 11.1 H200-hours; WebVoyager/DeepShop jobs 384818–384827, 25.9 H200-hours).

<a id="selection-head-terra-rejudge-20261023"></a>

<details>
<summary>Re-judging the earlier Online-Mind2Web runs with gpt-5.6-terra: agreement and success by run</summary>

- **Method.** For each committed episode, the last received judge request of its committed attempt is replayed with `model` set to `gpt-5.6-terra` (same messages, final screenshot and seed 42) and parsed with the native parser (`scripts/rejudge_crossbench_om2w.py`). Before any call, every saved o4-mini reply was re-parsed and matched the recorded reward. Invalid episodes stay invalid; episodes that never reached the judge keep reward 0. All 2,911 calls returned a verdict; cost $44.46 at $2.50 / $15 per M tokens (an upper bound on the current price). CPU job 383637.
- **Result.** Terra agrees with o4-mini on 88–95% of episodes valid under both. Disagreements are mostly o4-mini success → Terra failure (8–26 per run against 2–9 the other way), so success drops by up to 6.7 pp (one run rises 0.3 pp). Piotr SelectionARM and the Piotr-data heads stay ahead of SFT 4B and of every branch-label head.

| Run | Valid | o4-mini successes | o4-mini overall | Terra successes | Terra overall | Terra valid-only | Agree | o4-mini 1 → Terra 0 | o4-mini 0 → Terra 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 4B | 275 | 109 | 36.3% | 95 | 31.7% | 34.5% | 251 | 19 | 5 |
| SFT 8B | 273 | 109 | 36.3% | 100 | 33.3% | 36.6% | 256 | 13 | 4 |
| Piotr SelectionARM, N = 5 | 263 | 123 | 41.0% | 117 | 39.0% | 44.5% | 249 | 10 | 4 |
| Piotr SelectionARM, N = 3 | 259 | 134 | 44.7% | 119 | 39.7% | 45.9% | 236 | 19 | 4 |
| Piotr-data joint head, N = 5 | 265 | 131 | 43.7% | 113 | 37.7% | 42.6% | 235 | 24 | 6 |
| Piotr-data joint head, N = 3 | 269 | 127 | 42.3% | 110 | 36.7% | 40.9% | 240 | 23 | 6 |
| Piotr-data pointwise head, N = 3 | 264 | 124 | 41.3% | 115 | 38.3% | 43.6% | 241 | 16 | 7 |
| K = 1 head, SFT + RL 1,908 states (10-18) | 261 | 112 | 37.3% | 103 | 34.3% | 39.5% | 246 | 12 | 3 |
| K = 1 head, SFT-only states (10-18) | 259 | 105 | 35.0% | 97 | 32.3% | 37.5% | 237 | 15 | 7 |
| Outcome head | 270 | 114 | 38.0% | 94 | 31.3% | 34.8% | 238 | 26 | 6 |
| K = 3 head | 267 | 115 | 38.3% | 99 | 33.0% | 37.1% | 241 | 21 | 5 |
| K = 1 head | 264 | 102 | 34.0% | 93 | 31.0% | 35.2% | 251 | 11 | 2 |
| Pre-execution head | 268 | 99 | 33.0% | 90 | 30.0% | 33.6% | 251 | 13 | 4 |
| Independent transformer, soft CE | 265 | 99 | 33.0% | 91 | 30.3% | 34.3% | 245 | 14 | 6 |
| Independent transformer, BCE | 267 | 110 | 36.7% | 100 | 33.3% | 37.5% | 253 | 12 | 2 |
| Joint transformer, BCE | 256 | 92 | 30.7% | 93 | 31.0% | 36.3% | 239 | 8 | 9 |

- **Note.** Under Terra, the outcome head falls from 38.0% to 31.3% on Online-Mind2Web, the largest drop of any run, so its Online-Mind2Web standing in the [headline](#headline-results) partly reflects o4-mini leniency.
- Per-run summaries and verdicts: runtime `evaluations/rejudge-terra-20261023/out/`; summary copies in the [evidence JSON](arm_results/rl_integration/selection-data-scale-20261023.json).

</details>

<a id="selection-head-n3-8b-20261022"></a>

## Three candidates per decision, and SFT 8B (2026-10-22)

**On Online-Mind2Web, cutting from five to three candidates per decision costs no measurable success and saves 26–33% of model time per episode; all three N = 3 selectors still beat SFT 4B. SFT 8B with one sample gains +3.3 pp over SFT 4B on all 1,045 tasks, nothing on Online-Mind2Web, and stays below both five-candidate selectors on the 4B actor.**
- **Setup** as in the [headline](#headline-results): same task order, T 1.0 / top-p 0.9, 30 steps, one episode per task, one H200 per run. N = 3 was run on Online-Mind2Web only (300 tasks). The pointwise head is the per-candidate MLP (16.8 M parameters) trained on the same Piotr data; it has no N = 5 full-set run. SFT 8B is OpenWebRL-8B-SFT at revision `838c4a4`, run on all three benchmarks.

| Actor | Selector | Training labels | Candidates per step | Online-Mind2Web overall | Online-Mind2Web valid-only | − SFT 4B, overall, pp [95% CI] | PFLOP per episode | Seconds per decision | Model time per episode |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 4B | — | — | 1 | 36.3% | 39.6% | — | 0.26 | 2.81 | 40 s |
| SFT 8B | — | — | 1 | 36.3% | 39.9% | +0.0 [−5.3, +5.3] | 0.48 | 3.24 | 44 s |
| SFT 4B | Piotr SelectionARM (4B LoRA) | GPT-5.5 before, Piotr's 2,692 states | 5 | 41.0% | 46.8% | +4.7 [−1.0, +10.0] | 1.41 | 8.42 | 125 s |
| SFT 4B | Piotr SelectionARM (4B LoRA) | GPT-5.5 before, Piotr's 2,692 states | 3 | **44.7%** | **51.7%** | **+8.3 [+2.3, +14.0]** | 1.24 | 5.97 | 93 s |
| SFT 4B | Joint head (22.6 M) | GPT-5.5 before, Piotr's 2,692 states | 5 | 43.7% | 49.4% | **+7.3 [+1.7, +13.0]** | 0.88 | 7.79 | 114 s |
| SFT 4B | Joint head (22.6 M) | GPT-5.5 before, Piotr's 2,692 states | 3 | 42.3% | 47.2% | **+6.0 [+0.3, +11.7]** | 0.74 | 5.39 | 77 s |
| SFT 4B | Pointwise head (16.8 M) | GPT-5.5 before, Piotr's 2,692 states | 3 | 41.3% | 47.0% | +5.0 [−0.7, +10.7] | 0.72 | 5.40 | 77 s |

- **N = 3 vs N = 5 (overall, pp [95% CI]):** Piotr +3.7 [−1.7, +9.0], joint head −1.3 [−7.3, +4.3]. At N = 3, joint head − Piotr −2.3 [−8.0, +3.3]; pointwise − joint −1.0 [−6.7, +4.7]. Valid only, the same comparisons are +4.4, −2.3, −4.7 and +0.8 pp, none excluding 0.
- **Time.** A decision waits for the slowest of its parallel candidates, and eight episodes share each GPU, so three candidates cut decisions from about 8 s to 5.4–6.0 s.
- **SFT 8B per benchmark, − SFT 4B:** Online-Mind2Web +0.0 [−5.3, +5.3], WebVoyager **+4.5 [+0.7, +8.6]**, DeepShop +5.3 [−2.7, +13.3]; all **+3.3 [+0.4, +6.3]** (valid only +3.5 [+0.4, +6.6]). Against SFT 8B on all tasks, Piotr is **+3.6 [+0.5, +6.7]** and the Piotr-data head **+4.1 [+1.0, +7.3]**.
- **Why 8B is cheap.** It generates one sample per step, so its compute is twice SFT 4B's (0.037 vs 0.019 PFLOP per step) but well under five 4B samples plus a selector (0.06–0.10). Its decisions take 3.3 s against 2.8 s: decoding speed is limited mostly by reading the weights, which is fast at both sizes on an H200.
- **Caveat.** One run per method; Piotr's N = 3 gain over N = 5 is not significant. WebVoyager and DeepShop were not run at N = 3.
- [Evidence](arm_results/rl_integration/selection-n3-sft8b-full-20261021.json) · [analysis script](arm_results/rl_integration/selection-n3-sft8b-full-20261021-analysis.py). Runtime `evaluations/n3-8b-20261021/` (jobs 378681–378686).

<a id="selection-head-sft-actions-20261021"></a>

## Label, architecture and data ablations on SFT-proposed actions (2026-10-21)

**Labels matter more than anything else we varied. On the three benchmarks (1,045 tasks, each benchmark weighted equally), heads trained on observed-outcome or lookahead K = 3 labels for 1,380 SFT-actor states gain about +6 pp over single-sample SFT and are level with Piotr SelectionARM, which trained on about 25× more panels, at 56–65% of its compute per episode. Lookahead K = 1 labels gain about a third as much; pre-execution labels gain nothing.** Architecture barely matters: on all 1,045 tasks the four transformer heads crossing joint / independent attention with soft CE / BCE are within 1.3 pp of each other. The MLP, pointer and data variants fall within the live noise of a 150-task screen. Offline, the variants differ by at most 1.6 pp.

**Full benchmarks, per benchmark (Online-Mind2Web 300, WebVoyager 595, DeepShop 150 tasks; one episode per task; default head: joint transformer, soft CE).** Success rates and mean differences are in the [headline table](#headline-results); this table splits the overall-success difference from SFT 4B by benchmark.

| Selector (actor SFT 4B, five candidates) | Training labels | Online-Mind2Web − SFT 4B, pp [95% CI] | WebVoyager − SFT 4B | DeepShop − SFT 4B |
| --- | --- | ---: | ---: | ---: |
| Piotr SelectionARM | GPT-5.5 before, Piotr's 2,692 states | +4.7 [−0.7, +10.3] | **+8.1 [+4.0, +11.8]** | +7.3 [−1.3, +16.0] |
| Head (22.6 M) | GPT-5.5 before, Piotr's 2,692 states | **+7.3 [+1.7, +13.0]** | **+6.9 [+2.9, +10.9]** | **+10.0 [+1.3, +18.7]** |
| Head (22.6 M) | Observed outcome, the 1,380 branch states | +1.7 [−3.3, +6.7] | **+7.9 [+3.9, +12.1]** | +8.0 [0.0, +16.0] |
| Head (22.6 M) | Luna K = 3, the 1,380 branch states | +2.0 [−3.7, +8.0] | **+6.7 [+2.9, +10.6]** | +8.7 [0.0, +17.3] |
| Head (22.6 M) | Luna K = 1, the 1,380 branch states | −2.3 [−7.3, +2.7] | **+4.9 [+0.7, +8.9]** | +4.0 [−4.7, +12.7] |
| Head (22.6 M) | Luna before execution, the 1,380 branch states | −3.3 [−8.7, +1.7] | +1.3 [−2.5, +5.4] | 0.0 [−8.0, +8.7] |

- **Setup** as in the 150-task screen below; SFT, Piotr and the Piotr-data head are the [full-benchmark runs](#selection-head-crossbench-full-20261018). The four label heads share one architecture and the same 1,380 training states; only the label differs.
- **Ranking.** Outcome ≈ K = 3 > K = 1 > pre-execution. On the mean overall rate, outcome and K = 3 are −2.2 pp [−5.9, +1.4] and −2.3 pp [−5.8, +1.3] against the Piotr-data head.
- **Per benchmark.** The branch-label heads gain on WebVoyager and DeepShop; on Online-Mind2Web none clearly beats SFT, while the Piotr-data head does.
- **K = 1 regressed from the 150-task screen** (+8.0 pp there, +2.7 pp pooled over all 1,045 tasks here), a reminder that the screen resolves only about ±8 pp.
- **Nodes.** All twelve runs excluded the degraded node g013; two WebVoyager runs hit CUDA faults on g014 and resumed from their saved records on other nodes. Every task has exactly one recorded episode.
- Evidence: [per-benchmark runs](arm_results/rl_integration/selection-head-sft-actions-full-20261021.json), [benchmark means and differences](arm_results/rl_integration/selection-mean-benchmark-diffs-20261022.json)

**Architecture variants.** The screen below crosses interaction (joint / independent) with the scorer (transformer / MLP) and the loss (soft CE / per-candidate BCE); every variant reads the same frozen actor features and has about 22.6 M parameters.

![Selection-head variants: a joint transformer uses full bidirectional attention without position embeddings, so every candidate attends to every other and order does not matter, an independent one masks attention so each candidate sees only the state and itself; a joint MLP adds the candidate minus the mean of the others, an independent MLP sees only the candidate and the state; a worked example with a 3 : 1 target shows soft cross-entropy on the softmax (0.84) and per-candidate BCE on each sigmoid (0.77) for the same logits](arm_results/methods/selection-head-variants-20261022.png)

*[Figure script](arm_results/methods/selection-head-variants-20261022.py); code `openwebrl/selection_head.py` (`CandidateSetHead`, `selection_loss`, `pointwise_bce_loss`).*

<a id="selection-head-arch-full-20261022"></a>

**Full benchmarks, interaction × loss (all K = 1 Luna labels on the same 1,380 branch states; Online-Mind2Web 300, WebVoyager 595, DeepShop 150 tasks).** The 150-task screen could not separate the transformer variants, so all four were run on every task.

| Head (K = 1 labels) | Online-Mind2Web<br>valid-only | WebVoyager<br>valid-only | DeepShop<br>valid-only | Mean<br>overall | Mean<br>valid-only | − SFT 4B, pp<br>[95% CI] | PFLOP per<br>episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 4B, 1 sample | 39.6% | 45.0% | 34.2% | 38.2% | 39.6% | — | 0.28 |
| Piotr SelectionARM | 46.8% | 53.8% | 41.9% | 44.9% | 47.5% | **+6.7 [+3.0, +10.4]** | 1.53 |
| Joint transformer, soft CE (default) | 38.6% | 50.5% | 38.8% | 40.4% | 42.6% | +2.2 [−1.5, +5.8] | 1.05 |
| Joint transformer, BCE | 35.9% | 51.1% | 44.6% | 41.5% | 43.9% | +3.3 [−0.2, +6.9] | 1.01 |
| Independent transformer, soft CE | 37.4% | 51.6% | 37.8% | 40.2% | 42.3% | +2.0 [−1.5, +5.6] | 0.98 |
| Independent transformer, BCE | 41.2% | 48.5% | 36.9% | 40.3% | 42.2% | +2.1 [−1.6, +5.7] | 0.99 |

- **Main effects on the mean overall rate, pp [95% CI]:** joint − independent +0.7 [−2.1, +3.4] (soft CE +0.2, BCE +1.2); soft CE − BCE −0.6 [−3.1, +1.9] (joint −1.1, independent 0.0); interaction term −1.1 [−6.0, +3.8]. Valid-only differences agree within 0.5 pp.
- **Against Piotr,** every K = 1 head is 3.4–4.7 pp lower; three of the four intervals exclude zero. The gap to Piotr is a label effect: outcome and K = 3 labels on the default head close it (see the [headline table](#headline-results)).
- **Setup.** Same actor, decoding, judges and task order as the label runs above; one H200 per run, nodes g013, g014 and g019 excluded. One Online-Mind2Web run lost its first two tasks to a Chromium launch failure on g019; those records were set aside and the tasks rerun. Every task has exactly one recorded episode and no selector fell back.
- **Differences:** stratified bootstrap, tasks resampled within each benchmark (10,000 draws); each benchmark weighted one third. Valid-only uses tasks valid under all compared heads.
- Evidence: [runs and contrasts](arm_results/rl_integration/selection-head-arch-2x2-full-20261022.json) · [script](arm_results/rl_integration/selection-head-arch-2x2-full-20261022.py)

**150-task screen (first 50 tasks of each benchmark's full-run plan, 150 tasks; one episode per task).**

| Head (training labels / variant) | Online-Mind2Web<br>valid-only | WebVoyager<br>valid-only | DeepShop<br>valid-only | Mean<br>overall | Mean<br>valid-only | − SFT, pp<br>[95% CI] | − Piotr, pp<br>[95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 4B, 1 sample | 36.2% | 38.8% | 34.0% | 35.3% | 36.3% | — | — |
| Piotr SelectionARM | 37.8% | 51.0% | 51.0% | 44.7% | 46.6% | +9.3 | — |
| **Labels** (default head: joint transformer, soft CE) | | | | | | | |
| Pre-execution (Luna) | 26.8% | 40.8% | 38.0% | 33.3% | 35.2% | −2.0 [−10.0, +6.0] | **−11.3 [−19.3, −3.3]** |
| Lookahead K = 1 (Luna) | 37.8% | 53.1% | 45.8% | 43.3% | 45.6% | +8.0 [0.0, +16.0] | −1.3 [−9.3, +6.7] |
| Lookahead K = 3 (Luna) | 42.2% | 53.2% | 46.9% | 44.7% | 47.5% | **+9.3 [+2.0, +16.7]** | 0.0 [−8.7, +8.7] |
| Observed outcome (draws 0–2) | 42.6% | 48.0% | 48.0% | 45.3% | 46.2% | **+10.0 [+2.7, +17.3]** | +0.7 [−7.3, +8.7] |
| **Architecture** (K = 1 labels) | | | | | | | |
| Joint transformer, BCE | 39.1% | 51.1% | 44.9% | 42.7% | 45.0% | +7.3 [−0.7, +15.3] | −2.0 [−10.7, +6.7] |
| Independent transformer, soft CE | 32.6% | 48.9% | 53.1% | 42.7% | 44.9% | +7.3 [−0.7, +15.3] | −2.0 [−10.7, +6.7] |
| Independent transformer, BCE | 41.9% | 43.8% | 53.1% | 43.3% | 46.2% | +8.0 [0.0, +16.7] | −1.3 [−10.0, +7.3] |
| Joint MLP, soft CE | 37.0% | 53.1% | 38.8% | 41.3% | 42.9% | +6.0 [−1.3, +13.3] | −3.3 [−12.0, +5.3] |
| Joint MLP, BCE | 33.3% | 51.1% | 36.7% | 37.3% | 40.4% | +2.0 [−5.3, +9.3] | −7.3 [−15.3, +0.7] |
| Independent MLP, soft CE | 39.5% | 44.7% | 41.7% | 38.7% | 42.0% | +3.3 [−4.0, +10.7] | −6.0 [−14.7, +2.7] |
| Independent MLP, BCE | 41.3% | 45.8% | 42.9% | 41.3% | 43.3% | +6.0 [−2.0, +14.0] | −3.3 [−11.3, +4.7] |
| `<decide>` pointer readout, soft CE | 43.2% | 38.8% | 44.9% | 40.0% | 42.3% | +4.7 [−4.0, +13.3] | −4.7 [−13.3, +4.0] |
| **Data** (default head) | | | | | | | |
| K = 1, natural-frequency panels | 33.3% | 45.7% | 45.8% | 38.7% | 41.6% | +3.3 [−4.7, +11.3] | −6.0 [−14.0, +2.0] |
| Pre-execution + Piotr's 34,518 panels | 48.8% | 40.8% | 34.0% | 38.7% | 41.2% | +3.3 [−5.3, +12.0] | −6.0 [−14.0, +2.0] |
| K = 1, 1,908 SFT + RL states (earlier head) | 46.5% | 41.7% | 51.0% | 43.3% | 46.4% | +8.0 [0.0, +16.0] | −1.3 [−10.0, +6.7] |

- **Setup.**
  - Actor OpenWebRL-4B-SFT, T 1.0 / top-p 0.9, 30 steps, five candidates per decision. One H200 per head. Judges: o4-mini for Online-Mind2Web; GPT-4o for WebVoyager and DeepShop.
  - SFT, Piotr and the earlier head are the [full-benchmark runs](#selection-head-crossbench-full-20261018) restricted to the same 150 tasks.
  - "Overall" counts invalid episodes as failures; "valid-only" divides successes by valid episodes. "Mean" weights the three benchmarks equally; with 50 tasks each, mean overall equals the rate over all 150 tasks.
  - Differences are on overall success, paired over tasks: 95% bootstrap, 10,000 resamples.
- **Resolution.** With 150 tasks, a head-vs-head interval spans about ±8 pp. Only label differences of about 10 pp are resolved. The architecture and data rows cannot be ranked against each other or against the default.
- **Cost is the same for every head.** Each has 22.6–23.1 M parameters and costs 0.061–0.065 PFLOP per step, with 15–23 ms per selection and no fallbacks in any of 2,000–2,700 decisions. Piotr SelectionARM costs 0.100 PFLOP per step and 0.60 s per selection.
- **Node caveats.**
  - The pre-execution head's first job (376526) ran on node g013, which decoded about 4.7× slower than other nodes at equal load. It was cancelled after 48 Online-Mind2Web episodes and resumed on g006 (376616).
  - Its Online-Mind2Web row (11 of 50, 9 invalid) may be depressed by episode-time limits. On WebVoyager and DeepShop, both run on the healthy node, it scored 39 of 100, against 36 for SFT and 50 for Piotr.
  - The outcome head ran on a node about 2× slower (g021). Seconds per decision for these two heads (21–23 s) reflect the nodes, not the heads; the other heads took 7.8–8.5 s.

<a id="offline-selector-evaluation-schematic"></a>

**Offline evaluation.** Orange “Train labels” use rollout 0's evidence or outcome rates on rollouts 0–2. Blue “Held-out eval” scores all picks on rollouts 3–7. These names identify the within-state rollout split; the head's training/test state split is separate.

![Offline evaluation. (a) K=5 candidate actions, eight rollouts each. One pale-cyan group with a teal outline marks rollout 0 across all candidates. Immediate post-action states are salmon; later lookahead states are purple. Rollout 0's future states are expanded only for actions 1 and 5; dots abbreviate actions 2–4. All outcomes remain shown. Green means success and red means failure. Pre-execution is slate, post-execution salmon pink, lookahead purple, and the observed-rollout oracle orange. (b) One light-gray shared-input box with a gray outline and matching connectors feeds task, history, current state and K=5 candidate actions into all three teachers. Braces indexed i=1 to K group the complete candidate/evidence set supplied to each teacher. Pre-execution has no rollout evidence; post-execution includes each candidate's immediate result; lookahead includes the next three actor steps for each candidate. Teacher = GPT-6 Luna. The oracle takes an argmax over label-pool rates; scores use held-out rollouts](arm_results/methods/selection-offline-eval-20261022.png)

*Illustrative example; valid rollouts only. Subscripts index time and superscripts index candidates: $a_t^{i}$, $s_{t+k}^{i}$. Only actions 1 and 5 have expanded rollout-0 state labels; dots abbreviate the others. One pale-cyan group with a teal outline marks rollout 0 across all candidates; immediate post-action states are salmon and later lookahead states purple. Slate, salmon pink, purple and orange identify pre-execution, post-execution, lookahead H = 3 and the observed-rollout oracle. Here K = 5 counts candidates and H = 3 counts continuation steps; historical result tables use K for lookahead depth. The light-gray shared-input box and gray connectors explicitly feed task, history, current state and the K = 5 actions into all three teacher rows. Each call jointly receives the complete candidate/evidence set, represented by braces indexed i=1 to K; it returns one index. Pre-execution receives no future evidence. Post-execution and lookahead receive the corresponding rollout-0 evidence for **every** candidate, without judge verdicts. The figure abbreviates pre-/post-execution as “Pre-exec”/“Post-exec”; the oracle legend identifies its rollout pool (0–2). Orange marks train-label rates on rollouts 0–2 and the oracle; blue marks held-out evaluation on rollouts 3–7. Percentages are rounded to at most one decimal.*

“Oracle (observed rollouts)” matches the observed-outcome reference in the [rollout-split comparison](arm_results/rl_integration/lookahead-draw-split-20261013/lookahead_draw_split_piotr_absolute_lines_valid_only_main.png). It selects using rollouts 0–2 and is evaluated on 3–7; it is distinct from an oracle that maximizes held-out success.

<details>
<summary>Protocol, scoring and head training</summary>

Each rollout replays the shared state $s_t$ in a fresh browser, executes candidate $a_t^{i}$ and lets the frozen SFT actor finish; o4-mini judges success. State symbols represent screenshots/observations and tool feedback. Continuation actions $a_{t+1}^{i}, a_{t+2}^{i}, a_{t+3}^{i}$ extend lookahead H = 3 to $s_{t+4}^{i}$; H = 1 ends at $s_{t+2}^{i}$. The superscript identifies the branch's original candidate, not a newly sampled candidate index at every step. All expanded states/actions belong to rollout 0. The omitted middle state labels are a display simplification, not missing teacher inputs or collected data. Every call receives all five candidate/evidence pairs together, in addition to shared task/history/current-state context. It does not receive all eight rollouts: only rollout 0 supplies future teacher evidence. Prefixes are schematic and may end earlier. GPT-6 Luna returns a candidate index, not success rates; final answers are hidden from lookahead, and terminal verdicts are withheld from all teacher inputs.

All eight example rollouts per action are valid. Scores are pre/post 60%, lookahead/observed-rollout oracle 80%, and uniform/SFT sample 44%; the held-out-max oracle is also 80% in this example. Tied picks average the tied actions' rates. SFT sample weights actions by proposal frequency; the five example proposals differ, so it equals uniform. The observed-rollout oracle is the outcome-label reference previously called “Observed outcome.” Argmax uses unrounded rates; percentage formatting does not change its choice.

Pre-execution and lookahead vote shares form pᵢ = nᵢ/N, where nᵢ counts calls selecting action i and N counts valid calls. Outcome ties split target mass equally. Post-execution is a teacher comparison only; no head is trained on its labels. Heads use pre-action features, train on 1,380 states and are evaluated on 348 states from 86 unseen tasks.

</details>

[PDF](arm_results/methods/selection-offline-eval-20261022.pdf) · [SVG](arm_results/methods/selection-offline-eval-20261022.svg) · [Source](arm_results/methods/selection-offline-eval-20261022.py). Seven-inch canvas, ≥7 pt text, embedded PDF fonts, 300 dpi PNG. Labels/scoring: `scripts/train_head_round.py` (`targets`, `held_rate`).

**Offline (348 held-out test states, 86 tasks; picks scored by held-out success on rollouts 3–7, valid only).**

| Head | Distinct-action panels | Natural panels | Natural − SFT sample, pp [95% CI] | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Uniform pick | 36.26% | — | — | — |
| SFT sample (actor's own frequency) | 35.63% | 35.63% | — | — |
| Oracle | 47.96% | — | — | — |
| Pre-execution | 36.44% | 36.31% | +0.68 [−0.85, +2.34] | 22.56 M |
| Lookahead K = 1 | 37.29% | **38.06%** | **+2.43 [+1.19, +3.74]** | 22.56 M |
| Lookahead K = 3 | 36.69% | 37.68% | +2.05 [+0.65, +3.60] | 22.56 M |
| Observed outcome | 36.97% | 35.65% | +0.02 [−1.56, +1.59] | 22.56 M |
| K = 1, joint transformer, BCE | 37.70% | 37.26% | +1.63 [+0.44, +2.97] | 22.56 M |
| K = 1, independent transformer, soft CE | 38.07% | 37.85% | +2.22 [+0.81, +3.78] | 22.56 M |
| K = 1, independent transformer, BCE | 37.67% | 37.68% | +2.05 [+0.58, +3.76] | 22.56 M |
| K = 1, joint MLP, soft CE | **38.26%** | 37.98% | +2.35 [+0.97, +3.91] | 22.56 M |
| K = 1, joint MLP, BCE | 37.47% | 37.25% | +1.62 [+0.33, +2.95] | 22.56 M |
| K = 1, independent MLP, soft CE | 37.56% | 37.61% | +1.98 [+0.60, +3.38] | 22.56 M |
| K = 1, independent MLP, BCE | 37.09% | 37.15% | +1.52 [+0.28, +2.88] | 22.56 M |
| K = 1, pointer readout | 38.15% | 37.68% | +2.05 [+0.69, +3.46] | 23.09 M |
| K = 1, natural-frequency training | 38.00% | 38.04% | +2.41 [+0.83, +4.19] | 22.56 M |
| Pre-execution + Piotr's panels | 37.53% | 36.90% | +1.27 [−0.29, +3.01] | 22.56 M |

- **Panels.**
  - "Distinct-action panels" show one candidate per distinct action.
  - "Natural panels" are five SFT proposals drawn at the actor's own frequencies (20 seeded panels per state). They mirror live selection, where duplicates are common.
- **Offline results agree with the live labels except for outcome.**
  - Lookahead labels lead on natural panels, and pre-execution trails, as live.
  - The outcome head is null offline on natural panels (+0.02) but best live.
  - Soft CE beats BCE in all four architecture pairings offline, by 0.4–1.0 pp. Live, those pairs split 2–2.
- **Data changes.**
  - Natural-frequency training equals distinct-panel training on natural panels (38.04% vs 38.06%).
  - Piotr's panels raise the pre-execution head by about 0.6–1.1 pp offline. That still leaves it short of K = 1 trained on the 1,380 branch states alone.

<a id="selection-head-sft-actions-lora-20261021"></a>

**Same data, larger selector (offline).** Piotr's SelectionARM recipe (LoRA rank 32 on OpenWebRL-4B-SFT, `{"selection": N}`, 2 epochs) trained on exactly the heads' 1,380 training states and labels does no better than the 22.6 M head. Scored on 345 of the 348 test states (3 lack a label set or replay anchor that the LoRA panel builder requires), held-out draws 3–7, argmax pick:

| Selector (training data) | Distinct-action panels | Natural five-proposal panels | Natural − uniform, pp [95% CI] |
| --- | ---: | ---: | ---: |
| Uniform over actions | 36.0% | — | — |
| SFT sample (actor frequency) | 35.4% | — | — |
| Oracle | 47.8% | — | — |
| Head, K = 3 labels | 36.4% | 37.4% | — |
| LoRA 4B, K = 3 labels | 37.2% | 37.5% | +1.5 [−0.1, +3.1] |
| Head, outcome labels | 36.7% | 35.4% | — |
| LoRA 4B, outcome labels | 36.4% | 36.6% | +0.6 [−0.6, +2.0] |
| Piotr SelectionARM (his 34,518 panels) | 39.1% | 38.4% | **+2.4 [+1.1, +3.8]** |

- LoRA − head on the same states: K = 3 +0.7 pp [−1.7, +3.2] distinct, +0.1 [−1.8, +2.0] natural; outcome −0.4 [−2.4, +1.8] and +1.3 [−0.2, +3.1]. **The 4B selector does not beat the head on branch data, while Piotr's data still gives the best offline selector**, consistent with the live finding that data, not selector size, sets the gap.
- Distinct panels average 5 candidate orderings per state; natural panels are 5 seeded five-proposal panels. Reading the pick after the serving prefix instead of the training prefix changes success by under 0.3 pp. Head rows are the round-1 heads of the offline table above, restricted to these 345 states (one distinct ordering), so they differ slightly from that table.
- **Caveat for the Piotr row:** the branch states were collected on the tasks of Piotr's ARM training split, so his selector has trained on other states of the same tasks; the heads and LoRAs have not (task-grouped split). His offline lead may be partly task familiarity.
- Not yet run live. [Evidence](arm_results/rl_integration/selection-lora-sft-actions-offline-20261021.json)

<details>
<summary>States, actions, labels and training</summary>

- **States.** There are 1,905 SFT-actor branching states in the collector's self-reached, usable, ≥ 2-distinct export. 1,738 keep at least two **SFT-proposed** actions (437 tasks); 167 were dropped.
  - An action is a distinct effective (type, target) key under the collector's diverse rule v2 that the SFT actor proposed and that some executed panel candidate carries.
  - RL-proposed text never enters a panel. When an action's executed candidate came from the RL actor (294 actions), the SFT actor's first proposal of the same key represents it. Its outcomes stay those of the executed candidate, since continuations are always by the SFT actor.
  - Actions per state: two in 743 states, three in 604, four in 311, five in 80.
- **Split.** One seeded, task-grouped split (seed 20261021): 1,390 train / 348 test states, 86 test tasks.
  - Every label arm trains on the same 1,380 train states, those with all four labels available.
  - One training seed (0), no cross-validation.
- **Labels.**
  - Luna, screenshots only, no history, two random candidate orders, final answers hidden. Conditions: before execution, lookahead K = 1 and K = 3. Luna sees only the state's SFT actions.
  - Outcome labels: best success on draws 0–2, valid draws only, ties split. Draws 3–7 are used only for scoring.
  - Labelling cost USD 13.92 of the USD 30 cap.
- **Architectures** (`openwebrl/selection_head.py`).
  - Interaction: joint, or independent through an attention mask (MLP: no other-candidate term). Scorer: 2-layer transformer, or a depth-2 MLP width-matched to 22.56 M parameters (1,686 joint / 1,856 independent). Loss: soft CE, or per-candidate BCE on the vote share.
  - Pointer: a learned `<decide>` token in the joint transformer, scoring q·k / √d against each candidate (Kev-style readout).
- **Training.** Frozen OpenWebRL-4B-SFT features from layers 18/27/36, extracted with the cached prefix as at inference. AdamW 3e-4, weight decay 0.05, batch 64, 20 epochs, one-cycle schedule.
  - Pre + Piotr adds Piotr's 34,518 training panels (2,587 base states). The 1,398 panels whose tasks are in the branch test set were excluded.
  - Every head is saved with config, seed and state list.
- **Not run.** The approved priority list did not include the 1/4 and 1/2 data-scale runs, so they await confirmation. Also not run: heads on RL-actor states or RL-proposed candidates. Piotr's 4B LoRA trained on branch data was scored offline only ([above](#selection-head-sft-actions-lora-20261021)), not live.
- **Evidence:** [offline and online results](arm_results/rl_integration/selection-head-sft-actions-20261021.json).
  - Runtime: `selection-head-20261008/sft-actions-20261021/` (panels, labels, `round{1,2,3}/*/head.pt`) and `evaluations/sft-actions-150-20261021/` (jobs 376523–376554, 376616; `analysis/analyze.py`).
  - Code `arm-selection-head` at `4511da2`: `scripts/build_sft_action_panels.py`, `scripts/train_head_round.py`.

</details>

<a id="selection-head-other-actors-20261020"></a>

**Other actors: the head does not transfer, even to an RL fine-tune of its own actor** (2026-10-19/20). A head trained on one actor's candidates did not rank another actor's candidates better than chance, offline or live.

| Actor | Test | Head | Head − baseline, pp [95% CI] |
| --- | --- | --- | ---: |
| Qwen3-VL-2B OpenWebRL SFT (`davidan03/qwen3-vl-2b-thinking-openwebrl-sft` @ `95d5b13`) | Live, full Online-Mind2Web (300 tasks), T 1.0, 30 steps | Head retrained on 2B features (layers 14/21/28), K = 1 labels on the 1,631 SFT-actor states (4B-sampled candidates) | **−8.3 [−13.3, −3.3]** vs SFT-2B, 1 sample |
| OpenWebRL-4B (RL) @ `616cc8f` | Offline, 444 RL-actor states (270 tasks), held-out draws 3–7 | Frozen SFT-only 4B head, reading SFT-4B features | −0.26 [−1.96, +1.42] vs uniform |
| OpenWebRL-4B (RL) @ `616cc8f` | Same, as it would run live | Frozen SFT-only 4B head, reading the RL actor's own features | −1.12 [−2.95, +0.67] vs uniform |

| Full Online-Mind2Web, 2B actor | Successes (of 300) | Valid | Overall | − SFT-2B, pp [95% CI] | PFLOP per step | Seconds per decision | Browser steps per episode | Context overflows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT-2B, 1 sample | 80 | 268 | 26.7% | — | 0.009 | 1.60 | 16.0 | 7 |
| Piotr SelectionARM (4B selector) + 2B | **85** | 267 | **28.3%** | +1.7 [−3.7, +7.0] | 0.060 | 6.69 | 19.3 | 5 |
| 2B selection head | 55 | 252 | 18.3% | −8.3 [−13.3, −3.3] | 0.024 | 6.21 | 21.0 | 15 |

- **Offline looked fine; live did not.**
  - Offline, the 2B head matched the 4B head (−0.07 pp [−2.23, +2.94] on the shared test fold), but those states' candidates came from the 4B actor.
  - Live, the 2B writes the candidates. The head ranks them confidently (median score spread 6.2) but badly: episodes run longer (21.0 vs 16.0 steps) and overflow the context twice as often.
  - The live captures were complete: all five candidates at every one of 6,287 decisions, no fallbacks.
  - Not yet ruled out: a mismatch between live sglang captures and the offline extractor for this actor. A 100-decision replay check would settle it.
- **Selection helps less with the weaker actor.** Piotr SelectionARM gains +1.7 pp with the 2B, against +4.7 with the 4B on the same tasks.
- **RL-4B check.** RL-actor states are easier: uniform 52.7% vs about 33.5% on SFT-actor states. A small shift in the candidate distribution is enough to erase the head's advantage (about +2 to +3 pp over uniform on SFT-actor states).
- **Implication.** Train the head on states whose candidates come from the actor it serves. A head trained on RL-actor states, tested with 5-fold cross-validation, is planned once the current RL-state collection finishes.
- Evidence: [2B Online-Mind2Web](arm_results/rl_integration/crossbench-2b-om2w-20261020.json), [RL-4B offline transfer](arm_results/rl_integration/selection-head-rl4b-transfer-20261020.json). Runtime `evaluations/crossbench-2b-t10-r2-20261019/` (jobs 373113–373115), `selection-head-20261008/{actor2b-head-20261019,rl4b-head-check-20261020}/`. Code `3643a4c` (actor registry), `9302f64` (frozen-head scorer).

<a id="selection-head-factorial-plan-20261019"></a>

<details>
<summary>Earlier plan: head architecture, loss and data-scale factorial (2026-10-19; run as the <a href="#selection-head-sft-actions-20261021">2026-10-21 ablation</a>)</summary>

**Question.** Which head choices matter: interaction between candidates, the scorer, the loss, and data scale?

| Dimension | Levels | Isolation |
| --- | --- | --- |
| Cross-candidate interaction | independent / joint | Transformer: independent masks attention so each candidate sees only the state and itself. MLP: joint adds the candidate minus the mean of the other four. |
| Scorer | MLP / 2-layer transformer | Same interaction and loss |
| Loss | per-candidate binary cross-entropy / softmax cross-entropy over the five | Same architecture; the pick is always the top score |
| Data scale | 12.5, 25, 50, 100% of training tasks (then the newly collected states) | Nested by task (frac seed 20261017) |

- **Held fixed.**
  - Features: OpenWebRL-4B-SFT layers 18/27/36, three summaries per candidate.
  - Labels: Luna lookahead K = 1 on SFT-actor states.
  - Training: d_model 512, dropout 0.1; AdamW 3e-4, weight decay 0.05, batch 64, 20 epochs, one-cycle schedule.
- **Offline protocol.**
  - 5-fold task-grouped cross-validation × 3 seeds: 160 CPU array tasks, `scripts/run_head_factorial.py`, code `f02599b` and the following commit.
  - Each test state's pick is scored by held-out success (draws 3–7, valid only).
  - Reported per cell: success and gain over uniform. Each dimension's paired contrast is averaged over the other two dimensions' four cells, at every data scale, with task-clustered bootstrap intervals.
- **Live check** (needs GPU approval): the best and worst variants and two or three data scales on full Online-Mind2Web at T 1.0, 30 steps, about 3.5 GPU-hours each. Offline scores have not predicted live gains well.
- **New states, once collected.**
  1. K = 1 labels (Luna, about USD 4.40 per 1,000 states).
  2. Panels and targets (CPU).
  3. 4B features (about 35 min per 1,000 states on one H200).
- **Pilot (fold 0, seed 0, 1,631 SFT states; 285 test states, 65 tasks).**
  - All eight variants: 36.06–37.24% against 33.71% uniform.
  - Joint − independent −0.30 pp [−1.61, +0.89]; transformer − MLP +0.46 [−0.41, +1.91]; softmax − BCE −0.12 [−1.11, +0.79].
  - [Pilot results](arm_results/rl_integration/selection-head-factorial-pilot-20261019.json).

</details>

<a id="selection-head-training-recipe"></a>

<details>
<summary>Head training recipe: every evaluated head, data, architecture and hyperparameters</summary>

**Shared recipe.** The actor is frozen and only the head is trained.

- **Features.** The actor reads each state's prompt and current screenshot once, then teacher-forces each of the five candidates after it (`selection_head_features.extract`; `scripts/extract_selection_head_features.py`).
  - The head reads three layers. State: the prompt's last token at each layer. Each candidate: reasoning-token mean, action-token mean and final token at each layer, giving 9 vectors.
- **Architecture** (`openwebrl/selection_head.py`, `CandidateSetHead`).
  1. Per-dimension standardization fitted on the training features, then a projector to d_model 512 (Linear → GELU → dropout 0.1 → Linear), one for the state and one for the candidates.
  2. Listwise: a 2-layer pre-norm transformer encoder (8 heads, feed-forward 2,048, dropout 0.1) over the state token plus the five candidate tokens. Pointwise: an MLP on each candidate joined with the state, with no interaction between candidates.
  3. LayerNorm, then a linear score per candidate.
  - Width 512 and depth 2 are the original defaults (2026-10-08) and have not been tuned; the feature sweep varied layers and pooling only.
- **Objective and optimizer** (`scripts/train_supervision_ablation.py`).
  - Soft cross-entropy between the softmax over five scores and the target distribution. Candidate order is permuted every batch.
  - AdamW, learning rate 3e-4, weight decay 0.05, batch 64, 20 epochs, one-cycle schedule (5% warm-up), gradient clip 1.0.
  - No validation split or early stopping.
- **Training cost.** Feature extraction: about 6 min (2B) to 70 min (4B) per 1,600–2,100 states on one H200. Head training: 3–20 s per head. Every head is saved with its config, seed and state count.

| Head | Actor whose features it reads (layers) | Training data | Label | Train states (all folds) | ≥ 2 distinct effective actions | Parameters | Evaluated |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Pointwise, Piotr's data | OpenWebRL-4B-SFT (18/27/36) | Piotr's 2,692 states (35,916 panels) | GPT-5.5 pick | 2,692 states | — | 16.8 M | Online-Mind2Web same-hardware; WebVoyager + DeepShop 100-task |
| Listwise, Piotr's data | OpenWebRL-4B-SFT (18/27/36) | Piotr's 2,692 states | GPT-5.5 pick | 2,692 states | — | 22.6 M | full sets, 30 steps |
| Listwise, our SFT + RL states | OpenWebRL-4B-SFT (18/27/36) | 1,474 SFT-actor + 434 RL-actor branching states | Luna lookahead K = 1 votes | 1,908 | 1,410 | 22.6 M | full sets, 30 steps |
| Listwise, our SFT-actor states | OpenWebRL-4B-SFT (18/27/36) | SFT-actor branching states only | Luna lookahead K = 1 votes | 1,631 | 1,186 | 22.6 M | full sets, 30 steps |
| Listwise, our SFT-actor states, 2B actor | Qwen3-VL-2B OpenWebRL SFT `95d5b13` (14/21/28) | SFT-actor branching states only (candidates sampled by the 4B actor) | Luna lookahead K = 1 votes | 1,631 | 1,186 | 19.4 M | pending |
| Listwise students (outcomes, pre-execution, K = 1, K = 3) | OpenWebRL-4B-SFT (18/27/36) | supervision-ablation folds 1–4 | per source | 840 | — | 22.6 M | 100-task subsets, T 0.7 |

- **Distinct actions.**
  - Effective = action type plus target (element under a click, URL, typed text, scroll direction, keys), as in the [diversity analysis](ARM_FORMULATIONS.md).
  - 445 of the 1,631 SFT-actor states (27%) have a single effective action across all five candidates.
  - The exact tool-call text is distinct in 1,438 of them.
  - [Composition](arm_results/rl_integration/selection-head-training-state-diversity-20261019.json).
- **Held-out check, 2B head.** Trained on 1,325 states with test fold 0 held out (3 seeds), it matches the 4B-feature head on the same 210 original test states: −0.07 pp [−2.23, +2.94] (o4-mini) and +0.92 pp [−1.34, +4.08] (Luna).
- **Heads and runtime:** `selection-head-20261008/{heads,sftrl-heads-20261017,actor2b-head-20261019}/…/head.pt`; code `arm-selection-head` at `3643a4c` (trainer `--layers`, actor registry).

</details>

<a id="selection-head-crossbench-full-20261018"></a>

## Full benchmarks: Online-Mind2Web, WebVoyager and DeepShop at T 1.0

**On all 1,045 tasks, the lookahead K = 1 head trained on 1,908 of our states beats the single-sample SFT actor by +4.6 pp [+1.5, +7.6] at 64% of Piotr SelectionARM's compute. Piotr SelectionARM gains +7.0 pp [+3.9, +10.0]; the head is 2.4 pp behind it [−5.3, +0.4].**
- **Setup:** every task of each benchmark, one episode per task, the same tasks for every condition.
  - Actor decoding T 1.0, top-p 0.9 (chosen after a [30-task decoding test](#selection-head-decoding-test-20261018)); 1,024 tokens and 30 steps.
  - Judges: o4-mini for Online-Mind2Web; GPT-4o for WebVoyager and DeepShop.
- **Hardware:** each condition and benchmark ran alone on one H200, with Piotr's selector on the same GPU.

| Condition | Online-Mind2Web successes (of 300) | WebVoyager successes (of 595) | DeepShop successes (of 150) | All successes (of 1,045) | Online-Mind2Web overall | WebVoyager overall | DeepShop overall | All overall | All valid only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 4B, 1 sample | 109 | 264 | 51 | 424 | 36.3% | 44.4% | 34.0% | 40.6% | 41.9% |
| Piotr SelectionARM | **123** | **312** | 62 | **497** | **41.0%** | **52.4%** | 41.3% | **47.6%** | **50.2%** |
| **K = 1 head, 1,908 SFT + RL states** | 112 | 292 | **68** | 472 | 37.3% | 49.1% | **45.3%** | 45.2% | 47.9% |

| Paired difference, same tasks, pp [95% CI] | Online-Mind2Web | WebVoyager | DeepShop | All 1,045 tasks |
| --- | ---: | ---: | ---: | ---: |
| K = 1 head − SFT | +1.0 [−4.3, +6.3] | **+4.7 [+0.7, +8.6]** | **+11.3 [+2.7, +20.0]** | **+4.6 [+1.5, +7.6]** |
| Piotr SelectionARM − SFT | +4.7 [−1.0, +10.0] | **+8.1 [+4.2, +11.9]** | +7.3 [−1.3, +16.0] | **+7.0 [+3.9, +10.0]** |
| K = 1 head − Piotr SelectionARM | −3.7 [−8.7, +1.0] | −3.4 [−7.2, +0.5] | +4.0 [−4.7, +12.7] | −2.4 [−5.3, +0.4] |

Overall rates count invalid episodes as failures. Valid-only paired differences on all tasks are +5.2 pp [+2.0, +8.4] for the head and +7.5 pp [+4.3, +10.5] for Piotr against SFT (978 and 979 tasks valid under both).

| Cost and latency, all tasks | PFLOP per step | Selector PFLOP per step | Seconds per decision, mean | Steps per episode | Context overflows |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT 4B, 1 sample | 0.019 | — | 2.81 | 13.6 | 5 |
| Piotr SelectionARM | 0.096 | 0.034 | 8.68 | 14.3 | 16 |
| K = 1 head | **0.060** | **<0.001** | **8.21** | 15.3 | 22 |

- **Both selectors beat a single sample on the full sets; the head costs 36% less compute than Piotr.** The head's selector adds under 0.1% of a step's compute because it reads hidden states the actor already computed. Its decisions are 0.5 s faster than Piotr's.
- **The head is behind Piotr SelectionARM on Online-Mind2Web and WebVoyager and ahead on DeepShop**; no per-benchmark difference between the two excludes zero.
- **Selection lengthens episodes.** Both selectors take more steps and overflow the 32K context more often than SFT. Overflows count as failures.
- **Comparison with the 100-task runs at T 0.7** ([below](#selection-head-crossbench-20261017)): the same 1,908-state head was level with SFT there (+1.7 pp [−4.7, +7.7], 300 tasks). Two things changed together: 3.5× more tasks and higher-temperature candidates. With the same decoding, the 30-task test put the head's margin over SFT at +8.9 pp under T 1.0 and +5.6 pp under T 0.7 on the same 90 tasks.

<a id="selection-head-100-steps-20261019"></a>

**100 steps** (same tasks and decoding; history keeps the last 20 turns in full and reduces older turns to their action and tool-result line, `browser_history_full_turns` 20, commit `de7ccb9`):

| Method | Online-Mind2Web successes (of 300) | WebVoyager successes (of 595) | DeepShop successes (of 150) | All successes (of 1,045) | All overall | Gain vs 30 steps, pp [95% CI] | − SFT, pp [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 4B, 1 sample | 120 | 302 | 63 | 485 | 46.4% | +5.8 [+2.9, +8.8] | — |
| Piotr SelectionARM | **133** | **336** | **78** | **547** | **52.3%** | +4.8 [+1.8, +7.9] | **+5.9 [+3.0, +9.0]** |

- **Gain per benchmark, 100 − 30 steps.**
  - SFT: Online-Mind2Web +3.7 [−2.0, +9.3], WebVoyager +6.4 [+2.7, +10.1], DeepShop +8.0 [−0.7, +17.3].
  - Piotr: +3.3 [−2.0, +9.0], +4.0 [+0.2, +8.1] and +10.7 [+2.7, +18.7].
- **Piotr − SFT at 100 steps:** Online-Mind2Web +4.3 [−1.0, +9.7], WebVoyager +5.7 [+1.7, +9.6], DeepShop +10.0 [+0.7, +19.3]. At 30 steps the same comparison was +4.7, +8.1 and +7.3; selection keeps its value with longer episodes.
- **Why the gain is modest.** Most episodes end well before 30 steps. On 90 tasks that hit the 30-step limit (30 per benchmark), SFT went from 7 to 27 successes.
- **Cost.** Episodes run 24–26 steps on average instead of 14. Compute per step rises from 0.019 to 0.067 PFLOP (SFT) and from 0.096 to 0.163 (Piotr) because late steps carry longer prompts.
- **Context.** With the 20-turn window, prompts reach about 21–23K tokens at step 100. Overflows of the 32K context: SFT 12, Piotr 32 of 1,045. Overflows count as failures.
- [Evidence](arm_results/rl_integration/crossbench-full-h100-20261019.json). Runtime `evaluations/crossbench-full-h100-20261019/` (jobs 370304–370309); 90-task diagnostic in `evaluations/crossbench-horizon100-20261019/`.

**Model × data and state source** (same protocol, all 1,045 tasks; head = the same frozen-feature listwise head, 22.6 M parameters):

| Selector | Training data | Online-Mind2Web successes | WebVoyager successes | DeepShop successes | All successes | All overall | − SFT, pp [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Head | Piotr: 2,692 states, GPT-5.5 picks | **131** | 305 | 66 | **502** | **48.0%** | **+7.5 [+4.4, +10.5]** |
| Piotr SelectionARM (LoRA 4B) | Piotr: 2,692 states, GPT-5.5 picks | 123 | **312** | 62 | 497 | 47.6% | +7.0 [+3.9, +10.0] |
| Head | ours: 1,908 SFT + RL states, Luna K = 1 | 112 | 292 | **68** | 472 | 45.2% | +4.6 [+1.5, +7.6] |
| Head | ours: 1,631 SFT-actor states only, Luna K = 1 | 105 | 295 | 64 | 464 | 44.4% | +3.8 [+0.8, +6.9] |
| SFT 4B, 1 sample | — | 109 | 264 | 51 | 424 | 40.6% | — |

- **Same data, different selector size:** head − Piotr SelectionARM, both on Piotr's data, is +0.5 pp [−2.5, +3.4]. Per benchmark: +2.7, −1.2 and +2.7 pp.
- **Same head, different data:** Piotr's data − our SFT + RL states is +2.9 pp [−0.1, +5.9]. The gap is concentrated on Online-Mind2Web: +6.3 pp [+0.7, +12.0].
- **RL-actor states:** dropping them (SFT-actor states only, 157 more SFT states kept because outcome targets are not required) gives −0.8 pp [−3.6, +2.1], with the same pattern per benchmark. Offline on the shared test fold the two heads are also level (+0.15 pp [−1.40, +2.08]; [evidence](arm_results/rl_integration/sftonly-head-offline-20261019.json)).
- **Episode length:** the Piotr-data head takes 13.6 steps per episode, like SFT (13.6); our-data heads take 15.3–15.6 and overflow the context more often.
- [Evidence, all five conditions](arm_results/rl_integration/crossbench-full-t10-5cond-20261019.json).

<details>
<summary>Decoding test, protocol, accounting and evidence</summary>

<a id="selection-head-decoding-test-20261018"></a>

- **Decoding test (30 tasks per benchmark, the first 30 of each 100-task subset; 90 tasks).**
  - SFT successes: T 0.7 35, T 0.6 / top-p 0.95 27, T 1.0 / top-p 0.9 32.
  - K = 1 head successes: 40, 30 and 40.
  - Paired T 0.6 − T 0.7: SFT −8.9 pp [−20.0, +2.2], head −11.1 pp [−22.2, 0.0]. Paired T 1.0 − T 0.7: SFT −3.3 pp [−13.3, +6.7], head 0.0 pp [−10.0, +10.0].
  - The T 0.6 drop is concentrated on DeepShop, mostly in episodes that exhaust the 30-step limit.
  - The T 0.7 runs were on 2026-10-17 and the others on 2026-10-18, so decoding and date are confounded.
  - [T 0.6](arm_results/rl_integration/crossbench-decoding-t06-20261018.json), [T 1.0](arm_results/rl_integration/crossbench-decoding-t10-20261018.json).
- **Runner.** `scripts/run_crossbench_selector_eval.py` (commit `bd5a048` adds `--temperature/--top-p` and `--subset-of-root`).
  - Each condition and benchmark is its own frozen plan under `evaluations/crossbench-full-t10-20261018/<benchmark>/`.
  - Task order is seeded; the full set is the same for every condition.
- **FLOP accounting, as in the [same-hardware runs](#selection-head-samehw-20261012).**
  - Actor: 2 × 4.44 B parameters × (uncached prompt + generated tokens) over every actor request.
  - Piotr's selector: the same formula over its input and output tokens.
  - Head: its logged per-decision FLOPs.
  - Seconds per decision is the wall clock from the first candidate request to the pick, with 8 episodes running in parallel per GPU.
- **Invalid episodes, all tasks:** SFT 34, Piotr SelectionARM 54, head 59. Most are site resets and step errors common to every condition; the rest are context overflows (table above).
- **Compute.** 9 jobs, one H200 each: 19.7 GPU-hours in total. Judging was $12.57.
- **Evidence:** [per-condition rows and paired differences](arm_results/rl_integration/crossbench-full-t10-20261018.json). The runtime root holds every record, judge verdict, rollout, actor-request receipt and selection log.

</details>

<a id="selection-head-crossbench-20261017"></a>

## Across benchmarks: Online-Mind2Web, WebVoyager and DeepShop

**Training the lookahead K = 1 head on 1,908 of our states (SFT- and RL-actor) instead of 840 raises its live success by 7.0 pp [+0.7, +13.3] over 300 tasks. It is now level with the single-sample SFT actor (+1.7 pp [−4.7, +7.7]) and with the same head trained on Piotr's data. No selector beats SFT on DeepShop.**
- **Setup:** the same seeded 100 tasks per benchmark for every condition (Online-Mind2Web 100 of 300, WebVoyager 100 of 595, DeepShop 100 of 150), one episode per task.
- **Decoding:** T 0.7, top-p 0.9 and 1,024 tokens, as in the same-hardware runs. Heads see pre-execution inputs only.
- **Judges:** o4-mini for Online-Mind2Web; GPT-4o for WebVoyager and DeepShop.
- **Rates:** "overall" counts invalid episodes as failures; "valid only" divides by valid episodes.

| Condition | Selector training data | Online-Mind2Web successes | WebVoyager successes | DeepShop successes | Online-Mind2Web valid only | WebVoyager valid only | DeepShop valid only |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 4B, 1 sample | — | 34 | 43 | **47** | 37.8% | 43.0% | 47.0% |
| Piotr SelectionARM (LoRA 4B selector) | Piotr: 2,692 states, 35,916 panels | — | **55** | 45 | — | 57.3% | 45.9% |
| Head, Piotr's data, pointwise | Piotr: 2,692 states, 35,916 panels | — | 50 | **51** | — | 51.5% | 53.1% |
| Head, Piotr's data, listwise | Piotr: 2,692 states, 35,916 panels | **42** | 48 | 42 | 46.7% | 48.0% | 42.0% |
| **Head, lookahead K = 1, SFT + RL states** | **ours: 1,908 states** | **39** | **49** | **41** | **46.4%** | **50.5%** | **41.0%** |
| Head, lookahead K = 1 | ours: 840 states | 27 | 45 | 36 | 30.3% | 45.5% | 37.1% |
| Head, observed outcomes (draws 0–2) | ours: 840 states | 37 | 42 | 35 | 40.7% | 42.4% | 35.4% |
| Head, pre-execution teacher | ours: 840 states | 34 | 38 | 37 | 39.1% | 38.4% | 37.4% |
| Head, lookahead K = 3 | ours: 840 states | 25 | 40 | 40 | 29.4% | 40.8% | 42.1% |

Successes are out of 100 tasks, so they equal the overall rate in %. Piotr SelectionARM and the pointwise Piotr-data head were not rerun on Online-Mind2Web; their 3 × 300 same-hardware results are [above](#selection-head-samehw-20261012).

| Paired difference, same tasks | Benchmarks | Tasks | Overall, pp [95% CI] | Valid tasks | Valid only, pp [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: |
| K = 1 head: 1,908 states − 840 states | all three | 300 | **+7.0 [+0.7, +13.3]** | 272 | +7.4 [+0.7, +14.0] |
| K = 1 head, 1,908 states − SFT | all three | 300 | +1.7 [−4.7, +7.7] | 279 | +2.5 [−4.3, +9.3] |
| K = 1 head, 840 states − SFT | all three | 300 | −5.3 [−11.3, +0.7] | 281 | −5.0 [−11.0, +1.1] |
| Listwise head, Piotr's data − SFT | all three | 300 | +2.7 [−3.7, +9.0] | 286 | +2.4 [−4.2, +9.1] |
| Pointwise head, Piotr's data − SFT | WebVoyager, DeepShop | 200 | +5.5 [−2.5, +13.5] | 193 | +7.3 [−0.5, +15.0] |
| Piotr SelectionARM − SFT | WebVoyager, DeepShop | 200 | +5.0 [−2.5, +12.5] | 194 | +5.7 [−2.1, +13.4] |
| K = 1 head, 1,908 states − Piotr SelectionARM | WebVoyager, DeepShop | 200 | −5.0 [−12.5, +2.0] | 191 | −5.2 [−12.6, +2.1] |

Intervals resample tasks (10,000 resamples). Per benchmark, only two differences exclude zero: Piotr SelectionARM − SFT on WebVoyager, +12.0 pp [+2.0, +22.0], and the 1,908-state − 840-state head on Online-Mind2Web, +12.0 pp [+2.0, +22.0].

- **More of our states help live; the offline test fold does not show it.**
  - On the shared held-out test fold (210 states, held-out draws 3–7), the 1,908-state head scores the same as the 840-state head: −1.5 pp [−4.7, +0.7] under o4-mini.
  - The live gain likely comes from coverage: 138 new tasks and RL-reached pages that the old test fold does not sample.
  - So offline held-out success is not a sufficient check for deployment.
- **Data, not the selector's size, is the main gap to Piotr.**
  - The same frozen-feature head trained on Piotr's 2,692 teacher-labelled states lands within about 5 pp of his full 4B selector.
  - Ours is now level with that head on Online-Mind2Web and WebVoyager.
  - Piotr's states carry about 13 five-candidate panels each, against one per state for ours, so his data has about 19× more labelled choices.
- **DeepShop is the exception.** Only the pointwise Piotr-data head is above SFT there, at +4 pp; Piotr SelectionARM is −2 pp and our heads are −6 to −12 pp.
- **Caveat.** Each cell is one 100-task run, so a single benchmark's difference needs about 10 pp to clear noise; the pooled rows are the reliable comparisons. The [full sets at T 1.0](#selection-head-crossbench-full-20261018) supersede these numbers for SFT, Piotr SelectionARM and the 1,908-state head.

<details>
<summary>Protocol, invalid episodes, training data, cost and evidence</summary>

- **Runner.** `scripts/run_crossbench_selector_eval.py` with `openwebrl/crossbench_selector_eval.py` (Sol harness port; commits `83e1230`, `2fb1f28`, `bd5a048`).
  - Actor: OpenWebRL-4B-SFT on sglang, 8 tasks in parallel, local browser, 30 steps, full history, one screenshot, seed 42.
  - Selectors: five candidates per step. Heads score them from layers 18/27/36 captured while the actor generates, and Piotr SelectionARM runs on its own sglang server on the same GPU.
  - Plans are frozen per condition and benchmark (task subset seed 0, protocol hash, head sha256). A restart skips verified task records.
- **Invalid episodes.** Each Online-Mind2Web condition has 6 site-unreachable resets and 2 browser-step errors. Actor context overflows (a request over 32K tokens) vary by selector:
  - 8 for the 1,908-state head and 7 for the K = 3 head;
  - 2 for SFT, Piotr SelectionARM and the listwise Piotr-data head.
  - Overflows count as failures in the overall rate, because the selector's picks lead to them.
- **Selector training data.**
  - **Our heads:** branching states with 5 fixed SFT-sampled candidates and 8 continuation draws per candidate.
  - **Label sources:** Luna lookahead votes (screenshots only, no history, 2 randomized-order calls), the pre-execution teacher, or observed outcomes on draws 0–2. Draws 3–7 are never used.
  - **840-state heads:** the supervision-ablation split (folds 1–4).
  - **1,908-state heads:** 1,171 labelled SFT-actor states plus 925 new ones: 460 SFT-actor and 465 collected with the RL checkpoint OpenWebRL-4B `616cc8f`. All states have K = 1 labels ($4.10, 1,851 Luna calls), and training keeps states that also have an observed-outcome target.
  - **New tasks:** they go to training folds only, so the test fold stays task-disjoint.
  - **Piotr-data heads:** the selection-head training set above (GPT-5.5 picks on Piotr's 2,692 states). Pointwise scores each candidate alone; listwise attends across the five.
- **Offline check on the shared test fold** (task-disjoint; held-out draws 3–7, valid only; mean of 3 seeds). On the original 210 test states:
  - **K = 1 head:** 840 states 36.16%, 1,908-state training set 34.66% (o4-mini). The 1,908-state version trains on the 1,564 non-test states.
  - **Observed-outcomes head:** 34.73% and 35.05%.
  - **Uniform pick:** 31.91%.
  - **Under the Luna judge:** the same ordering. [Evidence](arm_results/rl_integration/sftrl-heads-offline-20261018.json).
- **Cost.**
  - Original 17 runs: 16.2 GPU-hours (1 H200 each, including restarts).
  - Piotr-data heads: 4.2 GPU-hours.
  - 1,908-state head: 1.1 GPU-hours for features and training, plus 2.9 for the live runs.
  - Judging: about $9 for all runs.
- **Evidence.** [Per-condition rows and paired differences](arm_results/rl_integration/crossbench-selectors-20261017.json). Runtime root: `evaluations/crossbench-selectors-20261017/` (frozen plans, every record, judge verdicts and rollouts). Heads: `selection-head-20261008/sftrl-heads-20261017/heads-all/` and `supervision-ablation-20261017/students-repro/heads/`.

</details>

<a id="selection-head-samehw-20261012"></a>

## Same-hardware runs: three repetitions of seven methods

**The headline table and figure come from these runs.**
- **Tasks:** each method ran the 300 Online-Mind2Web tasks three times, with seeds 45, 46 and 47 (server seeds 4500, 4600 and 4700).
- **Hardware:** every method used one H200, 8 CPUs and 120 GiB, and the actor served with CUDA graphs.
- **Selectors:** each separate selector shared the actor's GPU, so Piotr ARM was served through sglang and the Kev-style pass ran as a sidecar.
- **Unchanged:** the actor, prompts, decoding (T 0.7, top-p 0.9, 1,024 tokens, 30 steps) and the o4-mini/AgentTrek judge are the same as in the earlier runs.
- **Pilot first:** a 50-task pilot of all seven methods passed its startup gates before the repetitions launched.

| Method | Repetition 1 | Repetition 2 | Repetition 3 | SD | Steps per episode | Prefix-cache hit | Selection seconds per decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 4B, 1 sample | 34.3% | 33.7% | 33.0% | 0.7 | 15.2 | 82% | — |
| SFT 8B, 1 sample | 35.7% | 33.7% | 34.3% | 1.0 | 14.5 | 82% | — |
| Action-type prior | 29.3% | 28.3% | 29.0% | 0.5 | 18.1 | 91% | 0.00 |
| Piotr SelectionARM | 40.0% | 40.3% | 38.0% | 1.3 | 15.3 | 90% | 0.43 |
| Selection head (ours) | 39.7% | 36.3% | 39.3% | 1.8 | 15.4 | 90% | 0.02 |
| Kev-style pass | 39.7% | 38.3% | 41.0% | 1.3 | 15.3 | 88% | 4.19 |
| Luna (API) | 39.7% | 43.7% | 39.3% | 2.4 | 14.6 | 90% | 4.22 |


<details>
<summary>Verification, recoveries and accounting</summary>

- **Verification** (`analysis/analyze.py` in the study directory): every method and repetition has all 300 tasks with matching artifact hashes and no selector fallbacks.
- **Startup race (Piotr ARM).** One shard (repetition 2 shard 0) failed at start-up because Piotr's sglang selector and the actor started together and the selector's memory check lost the race.
  - The three Piotr shards that had not yet started re-ran on `source-r2`, which starts the selector and waits for it before the actor.
  - All earlier attempts are recorded.
- **Time-limit extension (Kev-style).** Repetition 1 Kev-style shard 1 hit its 3-hour limit after 102 of 150 tasks. It finished the remaining 48 tasks in a user-approved one-hour extension on `source-r3`, using 12,951 of 14,400 seconds in total.
- **Scheduling.** Five shards had their approval step killed on the login node and were resubmitted while still pending. A priority change on 28 pending shards removed the scheduler's submit-time boost, so they were cancelled and resubmitted before starting. No time was used by either.
- **Luna:** within its USD 15 / 15,000-call cap.
- **Compute proxy:** 2 × parameters × tokens actually computed, using 4.44B parameters for the 4B actor and 8.77B for the 8B. It ignores attention cost and the split between the vision encoder and the language model.

</details>

<details>
<summary>Earlier live runs (single repetition, CUDA graphs off, separate GPU per selector)</summary>

<a id="selection-head-om2w-20261011"></a>

## Full episodes on Online-Mind2Web: selectors in the live loop

**In the live loop on Online-Mind2Web, the actor-based selectors land between the SFT actor and Piotr ARM, and neither result is significant.** Each condition ran the 300 Online-Mind2Web tasks (one episode per task). Every condition except SFT alone samples five candidates from the SFT actor at each step and executes the selector's pick. These are the tasks of the October 7 controls under the identical protocol, and they are paired by task with those controls. The protocol is T=0.7, top-p 0.9, 1,024 tokens, 30 steps, seed 45 and the o4-mini/AgentTrek judge. Invalid episodes count as failures.

| Condition | Selector | Successes | Valid | Success | Δ vs SFT alone, pp [95% CI] | Δ vs Piotr ARM, pp [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SFT alone (Oct 7 control) | — | 99 | 271 | 33.0% | — | −7.0 [−11.7, −2.3] |
| Piotr ARM (Oct 7 control) | separate 4B SelectionARM | 120 | 261 | 40.0% | +7.0 [+2.3, +11.7] | — |
| Luna | `gpt-6-luna`, medium effort, API | 124 | 265 | 41.3% | **+8.3 [+3.3, +13.7]** | +1.3 [−3.7, +6.0] |
| Kev-style, step 600 | LoRA pass over the actor | 113 | 265 | 37.7% | +4.7 [−0.3, +9.7] | −2.3 [−7.3, +2.7] |
| Frozen-feature head, separate pass | actor hidden states + pointwise head (sidecar) | 111 | 264 | 37.0% | +4.0 [−1.0, +9.0] | −3.0 [−8.0, +2.0] |
| **Frozen-feature head, reuses actor KV cache** | same head; features captured inside the actor server | 113 | 259 | 37.7% | +4.7 [−0.7, +10.0] | −2.3 [−7.3, +2.7] |
| Action-type prior | fitted on branching outcomes; no model | 81 | 264 | 27.0% | **−6.0 [−10.7, −1.7]** | −13.0 [−18.0, −8.0] |

- **Luna** is the only selector besides Piotr ARM whose gain over SFT alone excludes zero. Its difference from Piotr ARM is not significant.
- **Kev-style and the frozen head** sit about 4–5 pp above SFT alone and 2–3 pp below Piotr ARM. Neither interval excludes zero. This matches the branching-state result: both were close to uniform there, slightly below Piotr.
- **The two frozen-head runs agree.** The cache-reusing run differs from the sidecar run by +0.7 pp [−4.7, +6.0].
  - The sidecar run, and all branch-state scoring of the actor-based selectors, built candidate features from ids ending `<|im_end|>\n`. A second `<|im_end|>` was appended to these, whereas training candidates end at one `<|im_end|>`.
  - The cache-reusing run uses the server's own ids, which match training. Fixed in `4a311a4`.
  - At this sample size the mismatch did not visibly cost success.
- **The action-type prior** is significantly worse than sampling one action. Its episodes are also longer: 18.3 steps versus 14.8.

**Comparability.** The new conditions were collected four days after the controls, so website drift is not controlled. Across three earlier full-300 runs, the SFT actor scored 98–100/300; the run-to-run spread is small next to these differences, but it does not cover drift. Intervals are task bootstraps: 10,000 samples, seed 42.

**Whole-system cost per episode** (actor plus selector):
- **Compute proxy:** 2 × 4.44B parameters × tokens actually computed. For the actor, that is the uncached prompt tokens plus generated tokens, using sglang's per-request `cached_tokens`.
- **Prefix cache:** the five candidate requests share the prompt prefill, and consecutive steps share the history, so 82–91% of actor prompt tokens hit the prefix cache.
- **Selectors:** they have no cached-token telemetry and are counted in full.

| Condition | Steps | Actor uncached prompt tokens (k) | Actor generated tokens (k) | Selector tokens (k) | Actor PFLOP | Selector PFLOP | System PFLOP | Selector share | System PFLOP per step | API $ | Episode seconds | Seconds per step | Selector seconds per step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT alone | 14.8 | 25.9 | 4.8 | 0 | 0.27 | 0 | 0.27 | 0% | 0.018 | 0 | 147 | 9.9 | 0 |
| Piotr ARM | 15.2 | 76.8 | 25.9 | 58 | 0.91 | 0.51 | 1.43 | 36% | 0.094 | 0 | 193 | 12.7 | 0.52 |
| Luna | 13.9 | 71.2 | 23.6 | 73 | 0.84 | external | ≥0.84 | — | ≥0.061 | 0.0084 | 224 | 16.1 | 3.48 |
| Kev-style (sidecar) | 15.1 | 78.4 | 26.1 | 184 | 0.93 | 1.63 | 2.56 | 64% | 0.169 | 0 | 235 | 15.5 | 0.97 |
| Frozen head (sidecar, as run) | 15.6 | 78.9 | 26.6 | 187 | 0.94 | 1.66 | 2.59 | 64% | 0.166 | 0 | 217 | 13.9 | 0.54 |
| **Frozen head, reuses actor KV cache** | 15.5 | 78.5 | 26.3 | 0.4 | 0.93 | 0.003 | 0.93 | 0.4% | 0.060 | 0 | 246 | 15.8 | 0.64 |
| Action prior | 18.3 | 88.5 | 29.9 | 0 | 1.05 | 0 | 1.05 | 0% | 0.058 | 0 | 230 | 12.6 | 0 |

- **Piotr ARM's separate selector is 36% of the system's compute.** With the prefix cache, five candidates cost the actor about 3.4× SFT alone (0.27 → 0.91 PFLOP), and Piotr's 4B selector then adds 0.51 PFLOP.
- **Reusing the actor's KV cache removes the selector's compute: 0.060 versus 0.094 PFLOP per step, 36% less than Piotr ARM.**
  - The frozen head reads the actor's hidden states at layers 18, 27 and 36 for the prompt's last token and each candidate's tokens. sglang forward hooks capture them while the actor generates.
  - A follow-up request per candidate computes only its final token, which generation never feeds back. That is 375 tokens per episode in total.
  - The sidecar version of the same head re-encoded everything and was the most expensive selector (2.59 PFLOP per episode).
  - It also runs on one GPU instead of two: 2.83 GPU-hours for 300 episodes, against 5.21 for the sidecar head.
- **Wall-clock latency did not improve.**
  - Seconds per step were 15.8, against 12.7 for Piotr ARM.
  - The actor's own requests were slower: 42.5 versus 31.0 summed request-seconds per step.
  - The parity test attributes about 6% of generation time to the hooks and 17% including follow-ups. The rest reflects different shard load and collection days.
  - The follow-ups add 0.64 s per step, mostly image preprocessing for the repeated prompt.
- **Kev-style cannot reuse the cache.** Its LoRA changes those layers, so it needs its own pass.
- **Luna** costs $0.0084 per episode in API fees, about $0.0006 per decision. It adds the most latency, 3.5 seconds per step, and its compute is unknown.

<details>
<summary>Method, verification, accounting and caveats</summary>

- **Conditions.**
  - Each condition used one 2026-10-11 frozen source copy that changes only the selector (`source-diff.json` in the study directory).
  - Luna ran from a second copy, `source-luna-r2`. It adds retries for transient provider errors and stops the controller if the shared selector halts.
  - The scalar selectors run as a sidecar on a second GPU, `scripts/serve_actor_selector.py`. Before serving, the sidecar re-scores six saved branching states and refuses to start unless every pick matches the offline pick.
  - The prior uses `action_type_prior.json`, with ties broken by candidate order.
  - The KV-cache head (`headkv5`) ran from `source-headkv`:
    - `openwebrl/actor_head_capture.py` hooks decoder layers 17, 26 and 35, giving hidden-state indices 18, 27 and 36.
    - It caches capture requests' generated tokens in the radix cache.
    - It exports each candidate's states when the request finishes. `openwebrl/actor_head_scoring.py` pools them and applies the head on CPU.
    - Parity against the Hugging Face reference on 40 saved states (`scripts/test_actor_head_capture.py`): picks agree on 37/40, all disagreements near-ties (score difference ≤0.58); feature cosine is at least 0.9993.
    - The first launch was cancelled after 58 s: the controller's in-place `<|im_end|>\n` normalization failed the id check. Its 4 records are kept under `excluded-defect-records`.
- **Pairing.**
  - The controls come from `selectionarm-piotr-sameday30-20261007`: actor0 is SFT alone and arm is Piotr ARM.
  - The task file is byte-identical, verified by SHA-256.
  - The protocol differs only in selector fields and cohort labels.
  - Candidate seeds follow the same rule, but trajectories diverge with the browser.
- **Verification** (`analysis/analyze.py`, read-only):
  - All 1,800 records are present, each with a matching artifact hash. Each condition has exactly 300 tasks under a single protocol hash.
  - Selector fallbacks are zero, and every logged decision has five candidates and no fallback.
  - Scalar picks equal the argmax of the logged scores.
  - The decision count equals the selector-call count: 4,691 head, 4,544 Kev-style, 4,179 Luna and 5,500 prior decisions.
- **Recoveries, with episodes moved aside and not deleted:**
  - Concurrent sidecar requests returned errors until scoring was serialized with a lock; the affected head episodes were rerun.
  - A one-GPU resource check blocked the Luna and prior startups; it was fixed.
  - A single provider 503 halted the shared Luna ledger, and the episode loop then continued without a selector. The 195 affected Luna episodes were rerun under `source-luna-r2`.
- **Accounting.**
  - All shards finished within their 10,800-second caps, counting every attempt.
  - GPU-hours over all attempts, per 300 episodes:

    | Condition | GPUs | GPU-hours |
    | --- | ---: | ---: |
    | Frozen head | 2 | 5.21 |
    | Kev-style | 2 | 5.75 |
    | Luna | 1 | 3.00 |
    | Prior | 1 | 2.60 |

    The controls ran both modes in shared 2-GPU shards, so their GPU-hours cannot be split by mode.
  - The Luna ledger shows 4,426 calls and $3.21 in total. That includes the discarded halted episodes; the 300 counted episodes cost $2.52.
- **Metric definitions.**
  - Tokens are the processed input plus output tokens recorded per episode, image tokens included.
  - The FLOP proxy is linear in tokens computed. It ignores attention cost (which grows with context, including cached context), memory-bound decoding and the split between the vision encoder and the language model.
  - Actor tokens come from all metered requests of each counted episode; 0.1–0.3% of requests were unmetered ([cache evidence](arm_results/rl_integration/selectors-om2w-20261011-cache.json)).
  - For the actor-based selectors, the sidecar's own estimate counts vision on image patches and agrees within 6%.
  - Selector tokens are counted without caching, since there is no cached-token telemetry; Piotr's selector may hit a prefix cache, which would lower its share.
  - Luna's FLOPs are unknown because it is an external model.
  - Seconds per step are wall-clock elapsed time divided by browser steps.
  - Actor request seconds are summed over the five parallel requests, so they can exceed wall time.
  - Latency depends on each shard's concurrent load, which differed between conditions and from the controls.
- **Evidence:** [aggregate](arm_results/rl_integration/selectors-om2w-20261011.json).

</details>

</details>

<a id="selection-inference-serving-20261024"></a>

## Actor serving options: replay test (2026-10-24)

**Of four serving changes, only n-gram speculative decoding speeds up decisions materially: −15% decision time and +23% decode throughput, with the same output distribution. Asking sglang for five samples in one request saves about 5%; a cache-filling request before the five candidates saves about 1%; a 40× larger image cache saves nothing.**
- **Setup.** Eight saved DeepShop episodes (2,587-state head run) replayed concurrently, as the eval's eight browsers per GPU, 12 steps each, five candidates per step: 96 decisions and 480 requests per configuration. Each step's prompt and screenshot are rebuilt from the saved episode; there is no browser time. One H200 (job 385920), the eval's actor-server flags, T 1.0 / top-p 0.9, 1,024 new tokens. The current setting was run twice (before and after the other single-server modes) to gauge noise.

| Serving option | Decision time, mean | Decision time, p90 | Decode throughput (tokens/s) | Prompt tokens from cache | − current, decision time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current: five seeded requests (two runs) | 9.66 s / 9.54 s | 13.4 s / 12.6 s | 1,391 / 1,419 | 85% | — |
| Image cache 4 GB instead of 100 MB | 9.66 s | 14.1 s | 1,417 | 85% | +0% |
| Cache-filling request (`max_new_tokens=0`) first, then five seeded requests | 9.48 s | 13.6 s | 1,426 | 96% | −1% |
| One request with `n=5` | 9.11 s | 13.7 s | 1,491 | ≈100% | −5% |
| N-gram speculative decoding (`--speculative-algorithm NGRAM`) | **8.18 s** | **11.8 s** | **1,741** | 94% | **−15%** |

- **Why the caches barely matter.** Decoding about 400 tokens per candidate is about 90% of a decision. Removing the repeated processing of each step's new 2k prompt tokens (the cache-filling request and `n=5` rows) saves at most a few percent; the cache-filling request also adds a 0.28 s round trip. The larger image cache changed neither uncached tokens nor time, so any screenshot re-encoding costs too little to measure here.
- **N-gram decoding.** sglang keeps one table of recent token sequences shared by all requests on the server: after each step it adds each request's last 18 tokens (generated text, plus the prompt's tail at a candidate's first step); whole prompts are not inserted. It matches a request's last ≤12 tokens in the table, proposes what followed, and the actor checks the guesses in one pass. Guesses therefore come mostly from sibling candidates and other episodes on the GPU, which share prompts and action syntax. At temperature > 0 sglang verifies by rejection sampling after temperature and top-p (thresholds 1.0), so the sampled distribution is unchanged; per-request seeds are not honoured, so runs are not bit-reproducible.
- **Images under n-gram.** Screenshot patches enter the prompt as hash-derived placeholder ids whose vision embeddings are written into the KV cache at prefill. Placeholders are never proposed (no generated context ends in one), and guessed text tokens see the image only through the KV cache, as in plain decoding. The one image-specific requirement, Qwen3-VL's M-RoPE offsets for guessed positions, is covered by the capture parity below.
- **Caveats before production.** Single runs; the two current-setting runs differ by about 1%. `n=5` uses one seed per request, also breaking per-candidate seeding.
- **Selection-head capture under n-gram: parity holds.** Each n-gram step is a verify pass over guessed tokens; the capture keeps only the rows of the accepted path at their real positions (`openwebrl/actor_head_capture.py`, arm-selection-head f5f5d67). Parity test, 40 states × 5 candidates, CUDA graphs, T 1.0, captured head features vs Hugging Face features recomputed from the same tokens:

| Server | States with complete capture | Picks agree with HF | Min / mean feature cosine |
| --- | ---: | ---: | ---: |
| No n-gram (job 386206) | 40 / 40 | 36 / 40 | 0.9993 / 0.99994 |
| N-gram (job 386218) | 40 / 40 | 39 / 40 | 0.9978 / 0.99994 |

  The first n-gram run lost the final `<|im_end|>` state in 16 / 40 states (sglang re-checks finish after a verify step without adding a token, consuming the capture's one-step end-token deferral); the fix holds the deferral until a token is appended. The test's 30 s latency phases are too short to measure the n-gram speedup, but show no added cost from the capture hooks under n-gram (30.3 s with capture vs 34.5 s plain, single runs). [Evidence](arm_results/rl_integration/selection-ngram-capture-parity-20261024.json).
- **Live check with n-gram: no sign of a performance change.** The old Piotr-data head with n-gram on (job 386408, 2026-10-24) on the first 100 Online-Mind2Web tasks of the 10-18 full-run order, same protocol, judge gpt-5.6-terra, against the same 100 tasks from the 10-18 runs re-judged by terra:

| Run | Successes | Valid | Valid-only | Hit 30-step limit | Seconds per browser step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old head, n-gram (2026-10-24) | 35 | 87 | 40.2% | 27 | 13.0 |
| Old head (2026-10-18) | 37 | 86 | 43.0% | 16 | 13.9 |
| Piotr ARM (2026-10-18) | 35 | 88 | 39.8% | 23 | 16.8 |
| SFT 4B (2026-10-18) | 30 | 93 | 32.3% | 26 | 7.9 |

  Successes are out of 100. On the 80 tasks valid in both old-head runs, n-gram alone succeeds on 10 and the 10-18 run alone on 11. One 100-task run has a standard error of about ±5 points, and the runs are six days apart on live sites. Zero selector fallbacks; five episodes ended on an actor 400 error (four in the 10-18 run). Seconds per step depend on shard load and site latency, so they do not isolate the n-gram speedup.
- **Multi-token prediction.** sglang 0.5.6 runs built-in prediction heads only for models trained with them (DeepSeek-V3/R1, Qwen3-Next, GLM-4.5/4.6, MiMo, Ling, LongCat-Flash). Qwen3-VL has none; an EAGLE-3 draft head would need training and Qwen3-VL support in sglang's model code.
- [Evidence](arm_results/rl_integration/selection-inference-efficiency-20261024.json) · [replay script](arm_results/rl_integration/selection-inference-efficiency-20261024-bench.py). Runtime `evaluations/inference-efficiency-20261024/`.

<a id="selection-head-latency-20261012"></a>

## Latency on identical hardware

**Each configuration runs on one H200, with any separate selector on the same GPU.**
- **Setup:** the same 96 saved states pass through the controllers' own selection code, with 8 decisions in flight. No browser is involved.
- **Measures:** decision time, which is candidate generation plus selection.

| Configuration (one H200 each) | Mean decision (s) | p90 (s) | Selector part (s) |
| --- | ---: | ---: | ---: |
| SFT 4B, 1 sample, CUDA graphs off (live-run protocol) | 4.33 | 5.88 | — |
| SFT 4B, 1 sample | 2.55 | 4.06 | — |
| SFT 8B, 1 sample | 3.18 | 4.62 | — |
| 5 samples + action prior (no model) | 5.89 | 8.53 | ≈0 |
| **5 samples + selection head (reuses actor KV cache)** | **6.07** | **9.16** | **0.01** |
| 5 samples + Piotr ARM via sglang | 6.76 | 10.06 | 0.44 |
| 5 samples + Piotr ARM, original HF server | 7.41 | 10.22 | 1.44 |
| 5 samples + Kev-style pass | 8.74 | 13.03 | 1.29 |

- **CUDA graphs:** they nearly halve the actor's single-sample latency (4.33 → 2.55 s). The live runs above had them off, so every method will be re-run with them on.
- **Five candidates:** they cost about 2.3× a single sample even with prefix caching (5.89 s with no selector model).
- **Piotr ARM served by sglang:** in two separate runs it picks the same candidate as the saved offline picks on 123 of 124 states. It cuts selection time from 1.44 s to 0.44 s, so this is the version used from now on.
- **The selection head:**
  - A decision takes 6.07 s, within 3% of five samples with no selector and 10% below Piotr ARM.
  - Capture requests now finish one decode step after `<|im_end|>`. That step feeds the end token through the actor, so its hidden state is captured too, and the extra sampled token is removed before the response. Only candidates cut off by the token limit still need a follow-up request: 4 of 200 in the parity test.
  - With hooks, generation ran at 3.58 ms per token against 3.76 ms without, so the overhead is within noise.
  - Parity with CUDA graphs on (40 states): feature cosine is at least 0.9992; 34 of 40 picks match, and every mismatch is a near-tie (score difference ≤ 0.51).
  - An earlier version sent a follow-up per candidate that resent the screenshot, and took 7.54 s per decision.
- **Kev-style:** its separate pass shares the GPU with the actor and is slowest (8.74 s).
- **Luna:** it is API-bound and not included.

Code: `scripts/bench_selector_latency.py`, `scripts/serve_arm_sglang.py`.

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

<a id="selection-head-outcome-adaptation-20261010"></a>

## Outcome adaptation pilot: five-fold, no gain

**Training the frozen-feature head on continuation outcomes overfits at this data size.** The adapted heads choose a best observed action on 83–100% of their training states. On held-out task groups they score at or below uniform.

| Selector, 138 states, out-of-fold | Success on draws 6–7 | − Uniform, pp [95%] | − Teacher-trained head, pp [95%] |
| --- | ---: | --- | --- |
| Hindsight best on draws 6–7 (optimistic) | 53.99% | +21.67 [+17.59, +25.86] | +19.93 [+14.75, +25.18] |
| **Outcome reference: best on draws 0–5** | **38.88%** | **+6.56 [+3.53, +9.87]** | +4.82 [+0.16, +9.45] |
| Kev-style pass, step 150 | 34.78% | +2.46 [−1.05, +6.29] | +0.72 [−3.28, +5.04] |
| Teacher-trained head (starting point) | 34.06% | +1.74 [−1.46, +5.12] | — |
| Piotr SelectionARM | 34.06% | +1.74 [−2.31, +6.00] | 0.00 [−4.14, +4.20] |
| Head trained on outcomes from scratch | 32.97% | +0.65 [−3.09, +4.64] | −1.09 [−5.67, +3.60] |
| Uniform candidate | 32.32% | — | −1.74 [−5.12, +1.46] |
| Teacher head adapted to outcomes, best-action target | 31.16% | −1.16 [−5.52, +3.08] | −2.90 [−7.52, +1.81] |
| **Teacher head adapted to outcomes (primary)** | **30.07%** | −2.25 [−5.47, +1.09] | **−3.99 [−7.66, −0.70]** |

- **Design, fixed before any result:**
  - The 138 states with features and draws 0–7 are split into five folds by task id (27–29 states each); each state is scored once by a head that never saw its task.
  - Labels are each candidate's success rate on draws 0–5, pooled over action-equivalent candidates.
  - Draws 6–7 are used only for scoring, by every row.
  - The primary model warm-starts from the teacher-trained pointwise head, with a soft target proportional to the draw 0–5 rates.
  - Training: AdamW, learning rate 1e-4, weight decay 0.05, 20 epochs; seeds 0 and 1, with their logits averaged.
- **Why it fails:** only 76 of the 138 states have candidates that differ on draws 0–5, so each fold trains on about 61 informative states.
  - The adapted head changes 80 of 138 picks relative to its starting point.
  - Its training loss halves (2.65 → 1.29), yet held-out success falls.
- **The prespecified collection roles agree, but are too small to conclude:** 13 validation states each.
  - On unseen tasks, the adapted head scores 30.77% and the teacher head 23.08%.
  - On seen tasks, both score 34.62%, against 40.00% for uniform.
- **What could work next:**
  - more states (the [prospective set](#selection-head-outcome-data-20261010) is untouched);
  - much stronger regularization chosen inside training folds only, for example a few steps, a lower learning rate, or adapting only the last layer;
  - combining outcome and teacher losses.

Scores use two draws per action, so per-state estimates are noisy; this protocol is not directly comparable with the five-on-three scores above. [Aggregate and training diagnostics](arm_results/rl_integration/selection-outcomes-20261010.json).

<details>
<summary>Breakdown: training folds versus held-out tasks, all-draw scoring, collections and roles</summary>

**The adapted head learns state-specific preferences that do not transfer to new tasks.** It improves on its training states, including on their unseen draws 6–7, but not on held-out tasks.

| Fold | Training states | Informative | Held-out states | Training states, draws 0–5 (trained on) | Training states, draws 6–7 (unseen draws) | Held-out states, draws 0–7 |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | 110 | 63 | 28 | 40.8 / 40.9 / 34.8 / 31.8 | 35.9 / 41.8 / 34.5 / 32.4 | 29.0 / 29.0 / 29.5 / 30.4 |
| 1 | 109 | 58 | 29 | 40.5 / 40.1 / 33.2 / 31.5 | 38.5 / 37.6 / 33.5 / 33.0 | 29.7 / 28.0 / 35.3 / 30.9 |
| 2 | 111 | 58 | 27 | 41.4 / 40.2 / 33.6 / 31.9 | 38.7 / 36.5 / 33.3 / 31.8 | 29.2 / 26.9 / 34.3 / 30.8 |
| 3 | 111 | 62 | 27 | 40.4 / 39.9 / 33.2 / 30.8 | 38.7 / 36.9 / 35.1 / 32.3 | 33.8 / 31.9 / 33.8 / 33.7 |
| 4 | 111 | 63 | 27 | 40.4 / 39.8 / 33.0 / 31.2 | 39.6 / 38.7 / 33.8 / 32.2 | 31.9 / 34.7 / 35.6 / 32.5 |

Each cell lists success rates as adapted / from scratch / teacher-trained / uniform, in %. All draws of a held-out state are unused in training, so held-out states can be scored on all eight draws. That gives tighter intervals than the two-draw protocol above, with the same conclusion:

| Selector, 138 states, out-of-fold | Success on draws 0–7 | − Uniform, pp [95%] | − Teacher-trained head, pp [95%] |
| --- | ---: | --- | --- |
| Piotr SelectionARM | 34.15% | +2.48 [-0.14, +5.11] | +0.45 [-2.19, +3.26] |
| Teacher-trained head | 33.70% | +2.03 [-0.24, +4.40] | — |
| Actor's first sample | 33.33% | +1.67 [-0.94, +4.32] | -0.36 [-4.01, +3.17] |
| Uniform candidate | 31.67% | — | -2.03 [-4.40, +0.25] |
| Adapted, best-action target | 31.43% | -0.24 [-2.88, +2.37] | -2.26 [-5.02, +0.28] |
| **Adapted (primary)** | 30.71% | -0.96 [-3.50, +1.51] | -2.99 [-5.67, -0.46] |
| Trained on outcomes from scratch | 30.07% | -1.59 [-3.97, +0.66] | -3.62 [-6.93, -0.63] |

| Group, held-out scoring on draws 0–7 | States | Adapted | Teacher-trained head | Piotr ARM | Uniform |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical | 51 | 24.5% | 26.2% | 25.7% | 23.4% |
| Pilot | 32 | 32.8% | 35.9% | 34.0% | 33.7% |
| Fresh 64-state batch | 51 | 34.3% | 39.2% | 41.9% | 37.5% |
| Fresh successor batch | 4 | 46.9% | 40.6% | 43.8% | 46.2% |
| Role: train | 61 | 35.7% | 39.8% | 40.4% | 37.6% |
| Role: validation, seen task | 13 | 36.5% | 41.3% | 43.3% | 43.1% |
| Role: validation, unseen task | 13 | 26.0% | 26.9% | 28.8% | 24.8% |

- **Training states.** The adapted head beats the teacher-trained head by about 7 pp on the draws it trained on. It still beats it by 3–7 pp on unseen draws of the same states, so outcome labels are consistent within a state.
- **Held-out tasks.** The adapted head is at or below the teacher-trained head in four of five folds and ties in the fifth. It trails the teacher-trained head in every group with more than four states.
- **The bottleneck is task coverage, not labels.** Each fold trains on about 60 informative states, mostly from distinct tasks. More distinct tasks should help more than more draws per state.

Jobs `354654` (primary) and `354696` (breakdown; identical folds, seeds and picks) are CPU-only.

</details>

<a id="selection-head-early-signal-20261010"></a>

## Early-signal experiments: what else could extract outcome signal now?

**Nothing tried beats the teacher-trained selectors at 138 states.** Every experiment below reuses the adaptation pilot's 138 states, five task-grouped folds, draw 0–5 labels and the rule that a held-out state's draws never enter training. Held-out states are scored on all eight draws. The outcome reference cannot be scored that way, because it selects on draws 0–5.

| Question | Result | Implication |
| --- | --- | --- |
| Does a low-capacity model find transferable outcome signal in the frozen features? | No. Ridge probes shrink to near-zero weights (penalty 10⁶–10⁹, chosen inside training folds) and score at uniform (−1.6 to +0.4 pp) | At this scale the features carry no linear, task-transferable outcome signal |
| Does keeping the teacher loss while adding an outcome loss help? | No. Nested CV chooses outcome weight 0 in all five folds; held-out success falls as the weight rises | Outcomes act as noise, not as a correction |
| Do combinations of existing selectors help? | Marginally at most: the best vote is +0.27 pp over Piotr | Piotr and the teacher head help only where they agree |
| Does an action-type prior explain the selectors' gain? | No. Alone it ties among candidates in 118 of 138 states and reduces to the actor's first sample (+1.85 pp vs +1.67). Its earlier +2.21 came from breaking ties with the teacher head. | Selectors, the actor's first sample and the prior all lie within noise of each other |
| How many states would a confirmatory comparison need? | About 70 states for 5 pp, 185–225 for 3 pp and 415–505 for 2 pp, at eight draws per action | Confirming Piotr's ~2.5 pp needs about 300 states; adding states beats adding draws |
| Is the historical states' weak first sample a depth effect? | No: depth reweighting moves it from 21.3% to 23.4% (new states: 40.4%) | The historical states are harder after the first decision; the cause is not isolated |

<details>
<summary>Ridge probes on frozen features</summary>

| Selector, 138 states, out-of-fold, draws 0–7 | Success | − Uniform, pp [95%] |
| --- | ---: | --- |
| Teacher-trained head (reference) | 33.70% | +2.03 [-0.24, +4.40] |
| Ridge, all 12 feature blocks | 32.07% | +0.40 [-1.88, +2.76] |
| Ridge, layers 18/27/36 (the head's inputs) | 31.79% | +0.13 [-2.23, +2.52] |
| Ridge, layer 27 action tokens | 31.61% | -0.05 [-2.34, +2.30] |
| Ridge, layer 36 end token | 30.07% | -1.59 [-4.02, +0.82] |

- **Model.** scikit-learn `RidgeCV` predicts each candidate's within-state-centred success (draws 0–5, pooled over equivalent actions) from within-state-centred, standardized features.
- **Penalty.** Chosen by four-fold task-grouped CV inside each training fold.
- **Earlier probe run discarded.** A first run used the default candidate-level leave-one-out. Duplicate candidates of the same state stayed in training, so it chose the smallest penalty and fit training states perfectly.

</details>

<details>
<summary>Teacher loss plus outcome loss</summary>

| Selector, 138 states, out-of-fold, draws 0–7 | Success | − Teacher-trained head, pp [95%] |
| --- | ---: | --- |
| Teacher loss only, 200 further steps (weight 0) | 32.88% | -0.82 [-2.39, +1.09] |
| Outcome weight 0.3 | 32.25% | -1.45 [-3.76, +0.94] |
| Outcome weight 1 | 32.25% | -1.45 [-3.64, +0.66] |
| Outcome weight 3 | 31.07% | -2.63 [-5.24, -0.28] |

- **Training.** Each run starts from the teacher-trained head and trains 200 steps: 16 GPT-5.5 panels (soft teacher target) plus 8 outcome states (target proportional to draw 0–5 success) per step, at learning rate 1e-4.
- **Weight selection.** Four inner task-grouped folds per outer fold, scored on draws 0–5 of inner held-out states only.
- **Changed picks.** Weight 0 changes 32 picks relative to the starting head; outcome weights change 68–74.
- **Input check.** The rebuilt inputs reproduce the teacher head's saved logits to within 3×10⁻⁶.

</details>

<details>
<summary>Selector combinations and the action-type prior</summary>

| Selector, 138 states, draws 0–7 | Success | − Uniform, pp [95%] |
| --- | ---: | --- |
| Vote of Piotr, teacher head and actor's first sample | 34.42% | +2.75 [+0.11, +5.37] |
| Piotr SelectionARM | 34.15% | +2.48 [-0.14, +5.11] |
| Action-type prior learned on training folds, ties to the teacher head | 33.88% | +2.21 [+0.02, +4.45] |
| Action-type prior learned on training folds, ties to the first candidate | 33.51% | +1.85 [−0.51, +4.29] |
| Teacher-trained head | 33.70% | +2.03 [-0.24, +4.40] |

- **Agreement split.** Piotr and the teacher head pick equivalent actions on 95 states. There Piotr scores 32.4% against 28.1% for uniform. On the other 43 states, Piotr scores 38.1%, the teacher head 40.1% and uniform 39.5%.
- **Action-type prior.** It ranks candidates by their action type's mean within-state relative success on the training folds. The per-type differences are small, within ±2 pp for every common type. In 118 of 138 states several candidates share the top type, so the tie-break decides. With the teacher head's logit as tie-break it scores like the teacher head; with candidate order it scores like the actor's first sample. *Corrected 2026-10-11: an earlier version credited the prior alone with the tie-broken result.*

</details>

<details>
<summary>Sample size and draw noise</summary>

| Comparison | Mean, pp | Draw-noise share at 8 draws | States for 5 pp at 8 draws | 3 pp at 4 draws | 3 pp at 8 draws | 3 pp at 16 draws | 2 pp at 8 draws |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Piotr ARM - Uniform | +2.48 | 0.44 | 69 | 274 | 191 | 149 | 428 |
| Teacher-trained head - Uniform | +2.03 | 0.45 | 67 | 267 | 184 | 143 | 414 |
| Piotr ARM - Teacher-trained head | +0.45 | 0.59 | 81 | 356 | 224 | 158 | 504 |
| Actor's first sample - Uniform | +1.67 | 0.42 | 79 | 310 | 218 | 173 | 491 |
| Kev-style step 600 - Uniform | +1.12 | 0.34 | 67 | 248 | 186 | 154 | 417 |

- **Variance split.** The between-state variance is the covariance of differences computed on interleaved half-draws; draw noise is the remainder.
- **Power settings.** Required states assume a two-sided 5% test at 80% power, with independent states; most task groups hold one state.
- **Total outcomes versus states.**
  - Fewest total outcomes: four draws per action.
  - Fewest states: sixteen draws.
  - The better design depends on whether admitting a state or running a draw costs more.

</details>

<details>
<summary>Decision depth and action types</summary>

| Action type | Candidates | Mean success, draws 0–7 | Piotr picks | Best-action picks (hindsight, ties split) |
| --- | ---: | ---: | ---: | ---: |
| `click` | 297 | 38.9% | 62 | 61.2 |
| `scroll` | 117 | 33.7% | 23 | 23.1 |
| `goto_url` | 80 | 9.8% | 20 | 16.4 |
| `done` | 58 | 62.1% | 11 | 11.2 |
| `go_back` | 54 | 18.1% | 5 | 8.5 |
| `new_tab` | 39 | 10.6% | 9 | 8.9 |
| `wait` | 20 | 10.6% | 3 | 4.5 |
| `press_keys` | 12 | 19.8% | 4 | 3.0 |
| `write` | 11 | 11.4% | 1 | 1.2 |

- **Depth bands.** Historical states hold 16, 20 and 15 states at decision depth 0, 1–3 and 4+; the newer collections hold 34, 39 and 14.
- **Historical states beyond the first decision are much harder.** Uniform success is about 16% there, against about 43% in comparable fresh states.

</details>

<a id="selection-head-label-types-20261011"></a>

## Label types at matched scale: teacher picks versus outcomes

**At about 110–124 training states, no label type trains a useful student.** The same pointwise head is trained from scratch, with AdamW at learning rate 3e-4 on a one-cycle schedule, 20 epochs, batch 32, and two seeds whose logits are averaged. Only the labels change.

| Comparison, 138 eight-draw states, 5-fold task CV, all draws | Success | − Uniform, pp [95%] |
| --- | ---: | --- |
| Outcome labels, draws 0–5 | 31.52% | −0.14 [−2.57, +2.16] |
| GPT-5.5 labels, same number of training states (other tasks) | 31.16% | −0.51 [−3.30, +2.30] |

**How much teacher data does the head need?** Heads were trained on GPT-5.5 panels from task-grouped subsets of the 2,692 teacher states and scored on all 138 states, which are disjoint from the teacher data. The full teacher-trained head reaches +2.03 pp.

| GPT-5.5 training data | Panels | Success | − Uniform, pp [95%] | Held-out GPT-5.5 agreement |
| --- | ---: | ---: | --- | ---: |
| 110 states | 1,458 | 30.98% | −0.69 [−3.04, +1.59] | 54.9% |
| 110 states, one panel each | 110 | 34.78% | +3.12 [+0.76, +5.53] | 52.7% |
| 300 states | 4,103 | 31.61% | −0.05 [−2.45, +2.40] | 58.5% |
| 300 states, one panel each | 300 | 32.70% | +1.03 [−1.19, +3.38] | 56.1% |
| 1000 states | 13,227 | 33.24% | +1.58 [−0.88, +4.21] | 61.9% |
| 2692 states | 35,916 | 34.24% | +2.57 [+0.21, +5.09] | 64.5% |

**Lookahead labels are the only ones that move a small student above uniform.**
- **Setup:** the same student and recipe, trained on the original 124 branching states, with Luna labels of each kind.
- **Scoring:** 5-fold task-grouped CV on all 124 states, and a transfer test that trains on all 124 states and scores 87 new ones.
- **"Labelled states":** the training states that have that label type. States without labels are skipped in training; scoring always uses every state.

| Training labels | Labelled states | 5-fold CV, − uniform, pp [95%] | Train 124 → test 87 new, − uniform, pp [95%] |
| --- | ---: | --- | --- |
| **Luna, 3-step lookahead, answers hidden** | **48** | **+2.37 [−0.08, +4.97]** | **+2.99 [−0.56, +6.54]** |
| Luna after execution, draw-0 evidence | 119 | −0.42 [−3.49, +2.65] | −0.75 [−3.95, +2.34] |
| Luna after execution, evidence from draws 0–2 | 121 | −0.52 [−3.60, +2.49] | +0.26 [−2.72, +3.15] |
| Luna before execution | 124 | −1.71 [−4.82, +1.31] | −0.46 [−3.44, +2.67] |
| Sol before execution | 124 | +0.17 [−2.96, +3.27] | +0.26 [−2.76, +3.31] |

- **Why only 48:** the lookahead study covered 211 states, of which 51 are among the original 124; 3 of those failed the draw-0 rollout alignment check.
- **Next:** train on all lookahead-labelled states, the 194 from that study plus 121 fresh Piotr-matched states.

Agreement and outcome success rise together with teacher data: the head needs about 1,000–2,700 teacher states to clear uniform, roughly 10–25× the outcome-labelled states available per fold. The one-panel-per-state 110-state result is a single comparison among several; the same states with all their panels do not reproduce it. Pre- versus post-action labels and teacher strength are compared in [ARM_FORMULATIONS.md](ARM_FORMULATIONS.md#teacher-labels-211-20261011).

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

![Five selector designs: layer stacks show which passes run, what is reused from the actor's generation pass and what is trained; attention masks show which candidates can see each other; rows give inputs, order effects, readout, trained parts, extra compute and results](arm_results/methods/selection_designs_layers.png)

[PNG](arm_results/methods/selection_designs_layers.png) / [SVG](arm_results/methods/selection_designs_layers.svg); rendered by `scripts/render_selection_designs_v2.py`.
- **Layer stacks.** Blue is the actor's own generation pass. Hatched orange is extra computation through LoRA-trained weights. Gray is a separate model.
- **Attention masks.** Rows are queries and columns are keys: c1–c3 are candidates and D is the `<decide>` token.
- **Why the Kev-style pass cannot reuse generation.**
  - It runs the prompt, screenshot, candidates and `<decide>` as one packed sequence, with LoRA active on every token in all 36 layers (`openwebrl/selection_kev.py`).
  - That changes the prompt's hidden states and keys/values relative to the actor's generation, which has no LoRA, so the generation cache cannot be substituted.
  - This follows Kev's training, not a hard limit. A variant trained with the prompt encoded with LoRA switched off, as the top-8 adapter does, could reuse the prompt's cache. Its candidates would still need a LoRA pass through all 36 layers.

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

- **Outcome supervision at scale.** The [138-state pilot](#selection-head-outcome-adaptation-20261010) overfits. Repeat it as states accumulate, with regularization chosen inside training folds, and test once on the untouched prospective states.
- **More fresh states.** Each additional 100 states with eight draws narrows selector-difference intervals and adds training signal. The [collection plan](ARM_FORMULATIONS.md) now prioritizes tasks from Piotr's ARM training split.
- **Benchmark impact.** Only a matched full300 run (SFT alone, SFT + Piotr ARM, SFT + an actor-based selector) shows whether any selector changes task success. On the [inference benchmark](ARM_INFERENCE_SCALING.md), Piotr ARM adds +5.67 ± 1.15 pp over three runs despite its weak branching-state gain here; the two evaluations measure different things.
- **Serving.** A deployed head needs the serving engine to return hidden states, or a sidecar scorer. Neither has been benchmarked.

<details>
<summary>Method: features, head and data</summary>

![Selection head inside the actor: (A) at inference the frozen SFT actor samples five candidates and a small head scores them from the actor's hidden states, read directly or through a top-layer scoring adapter; (B) training stage 1 fits the head on cached frozen features, stage 2 trains LoRA on the top eight layers plus the head with the prompt encoded by the unadapted actor](arm_results/methods/selection_head_workflows.svg)

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
