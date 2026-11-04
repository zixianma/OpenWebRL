# Pre-execution vs. K = 3 lookahead teacher labels (GPT-6 Luna)

<a id="luna-pre-vs-k3p-20261103"></a>

**Question.** Both labels come from one judge, GPT-6 Luna, using the same prompt. They differ only in what the judge sees. What does seeing each candidate's execution plus 3 actor steps change about the picked action and the stated reason?

**Takeaways** (20,671 p16 panels, 2026-11-03):

- **Live, K = 3 labels help heads but not LoRAs.** Heads: +2.9 pp (Luna) and +3.2 pp (GPT-5.5) mean overall success over pre labels, both significant; Luna LoRA: +0.3 [−1.8, +2.4]. No p16 selector beats same-day SelectionARM. [Table](#luna-pre-vs-k3p-live-20261103)

- **The two labels pick the same executed action on 63.3% of panels.** Agreement falls as panels hold more distinct actions: 69.9% with 2 distinct targets, 44.3% with 5.
- **Lookahead mostly replaces guesses with observed effects.** k3p cites a concrete observed result in 88.8% of coded disagreements. Its reasoning hedges ("may", "likely", "appears") on 20.8% of panels, against 47.2% for pre.
- **Half of the disagreements are evidence-driven corrections.** In the 240 coded panels, 25.4% are a pre pick that k3p saw fail: a block, CAPTCHA, verification wall, no-op click, zero results, or the wrong page. Another 25.8% are a k3p pick that k3p saw reach the needed page or information.
- **Pre's main weakness is predicting what a click will do from the screenshot.** In 14.2% of coded disagreements, pre misjudged a click: it called a click "blank space" or "would collapse", and execution showed the click opening the target.
- **Lookahead changes stopping behaviour.** k3p picks stop/answer more often at depth 0 (7.0% vs 4.2% of panels). On blocked sites, pre keeps retrying while k3p sees the retry hit the same block and reports it honestly.
- **About a quarter of disagreements are label noise.** 14.2% are near-ties among equivalent actions, where the difference is arbitrary. In another 12.5%, k3p credits or blames a candidate for what the actor happened to do afterwards in its single sampled continuation. All 7 coded cases where pre looks better in that 12.5% are of this kind.
- **There is no outcome ground truth here.** Each p16 candidate was executed once, so "which looks better" is a text-only judgement by an LLM coder. Of the 240 coded disagreements, 37.1% favour k3p, 4.2% favour pre and 58.8% are unclear. Treat these as qualitative evidence, not a success-rate estimate. The live LoRA and head evaluations are the outcome test ([below](#luna-pre-vs-k3p-live-20261103)).

## Live outcome: K = 3 vs pre labels, Luna vs GPT-5.5, LoRA vs head (3 runs, 2026-11-03)

<a id="luna-pre-vs-k3p-live-20261103"></a>

**Every selector trained on the p16 panels beats one SFT sample except the two pre-label heads, and every 11-03 selector is below same-day SelectionARM (45.7% mean overall).** K = 3 labels help heads (+2.9 Luna, +3.2 GPT-5.5, both significant) but not LoRAs (Luna +0.3, interval crossing zero). With K = 3 labels, the GPT-5.5 head is 1.6 pp above the Luna head ([−0.4, +3.6]), and the Luna LoRA and head are tied (−0.6 [−2.5, +1.5]).

Three runs per row, all 1,045 tasks (WebVoyager 595, Online-Mind2Web 300, DeepShop 150). SFT 4B actor, N = 5 candidates, T 1.0 / top-p 0.9, seed 20261018, 30 steps, local browsers. Judges: gpt-5.6-terra for Online-Mind2Web, GPT-4o for WebVoyager and DeepShop. Per-benchmark columns are valid-only success (successes / valid episodes), pooled over three runs. Mean = equal-weight average of the three benchmarks, overall (invalid = failure) and valid-only. Differences are on mean overall against the same-day SFT run, with per-task success averaged over the three runs and [paired 95% intervals](ARM_FINAL_RESULTS.md#final-paired-bootstrap) (10,000 task resamples within each benchmark, seed 20261103); **bold** intervals exclude zero. Paired intervals against SelectionARM are in the [JSON](arm_results/luna_pre_vs_k3p_20261103/live_3run_20261103.json).

| Selector (labels, model) | Eval date | WebVoyager<br>valid-only | Online-Mind2Web<br>valid-only | DeepShop<br>valid-only | Mean<br>overall | Mean<br>valid-only | − SFT, pp<br>[95% CI] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 4B, one sample | 11-03/04 | 45.6% | 34.6% | 43.1% | 39.6% | 41.1% | — |
| SelectionARM (Piotr) | 11-03/04 | 52.4% | 43.8% | 50.5% | 45.7% | 48.9% | **+6.2 [+4.0, +8.3]** |
| Luna pre, LoRA | 11-03 | 51.2% | 40.2% | 46.2% | 42.1% | 45.9% | **+2.5 [+0.3, +4.6]** |
| Luna K = 3, LoRA | 11-03 | 49.4% | 37.8% | 50.9% | 42.3% | 46.1% | **+2.8 [+0.7, +4.9]** |
| Luna pre, head | 11-03 | 47.3% | 38.8% | 46.1% | 40.0% | 44.1% | +0.5 [−1.7, +2.7] |
| Luna K = 3, head | 11-03 | 48.0% | 39.8% | 52.2% | 42.9% | 46.7% | **+3.3 [+1.1, +5.5]** |
| GPT-5.5 pre, head | 11-03 | 49.9% | 36.8% | 45.0% | 41.3% | 43.9% | +1.7 [−0.5, +4.0] |
| GPT-5.5 K = 3, head | 11-03 | 52.8% | 37.6% | 51.7% | 44.5% | 47.4% | **+5.0 [+2.9, +7.1]** |
| GPT-5.5 pre, LoRA ¹ | 11-03 | 57.3% | 41.9% | 52.8% | 48.0% | 50.7% | **+8.4 [+6.5, +10.5]** |
| GPT-5.5 K = 3, LoRA ¹ | 11-02/03 | 56.3% | 39.5% | 51.4% | 46.5% | 49.1% | **+7.0 [+4.7, +9.2]** |

¹ The two GPT-5.5 LoRAs ran overnight 11-02/03, about a day before the SFT and SelectionARM runs; their differences include day-to-day drift (SelectionARM ran 1.6 pp higher on 11-02 than on 11-03). A same-day 3-run rerun of both GPT-5.5 LoRAs and SelectionARM is running on 11-04 and will replace these two rows.

| K = 3 − pre, pp [95% CI] | WebVoyager | Online-Mind2Web | DeepShop | Mean overall |
| --- | ---: | ---: | ---: | ---: |
| Luna, LoRA | −0.2 [−2.9, +2.6] | −2.6 [−5.6, +0.4] | +3.6 [−1.3, +8.2] | +0.3 [−1.8, +2.4] |
| Luna, head | +1.5 [−0.9, +3.8] | +1.1 [−1.9, +4.1] | **+6.0 [+0.9, +11.3]** | **+2.9 [+0.7, +5.0]** |
| GPT-5.5, head | **+3.0 [+0.5, +5.5]** | +0.9 [−2.1, +4.0] | **+5.8 [+0.9, +10.7]** | **+3.2 [+1.1, +5.3]** |
| GPT-5.5, LoRA ¹ | −0.8 [−2.9, +1.5] | −1.6 [−4.8, +1.7] | −2.0 [−6.7, +2.7] | −1.4 [−3.4, +0.6] |

<details>
<summary>Training data, runs and files</summary>

- **Training data:** all p16 panels (20,525 labelled by both Luna conditions; 2,309 states, 896 tasks), the same panels and train/validation split for every row. LoRAs: Piotr's SelectionARM LoRA recipe on the 4B decoder, served merged as the selector. Heads: `train_head_round.py --extra-only`, batch 256, 20 epochs, seed 0, best validation epoch. GPT-5.5 labels are the earlier K = 3 lookahead and pre-execution labels on the same panels.
- **Runs:** every row has 1,785 / 900 / 450 completed episodes; there are no missing tasks. Two jobs were retried and resumed from saved records (Luna pre LoRA run 1 DeepShop: startup goto timeouts; Luna pre head run 3 Online-Mind2Web: model service died at 181/300).
- **Files:** [per-row counts and all paired intervals](arm_results/luna_pre_vs_k3p_20261103/live_3run_20261103.json) · [script](arm_results/luna_pre_vs_k3p_20261103/live_3run_20261103.py). Runtime evaluation roots: `evaluations/{p16-lora-lunapre-20261103, p16-lora-lunak3-20261103, p16-heads-luna-20261103, p16-heads-all-20261102, p16-lora-pre-20261102, p16-lora-20261101, controls-sameday-20261103}`.
- **Interval scope:** the intervals cover task sampling only, not training-seed variance (one seed per selector) or drift between eval windows.
</details>

## Setup

| | Pre-execution (`pre`) | K = 3 lookahead (`k3p`) |
| --- | --- | --- |
| Judge | `gpt-6-luna`, effort high, 16,384 output tokens | same |
| Prompt | Piotr's system prompt and instructions: task, action history, current screenshot, 5 candidates with actor thoughts. The judge explains its choice in 2–4 sentences, then outputs `{"selection": N}` | same, plus one "Execution Lookahead" block before the instructions |
| Execution evidence | none | For each candidate: the draw-0 execution result (URL, title, tool response, screenshot), then the actor's next 3 steps with a screenshot after each. Page text and the final answer are hidden. |
| Candidate order | panel order | same, not shuffled |
| Panels | 20,671 (p16a, d, e, f, g); 0 failures | 20,671; 0 failures, 0 with missing lookahead |
| Cost | $13.73 | $63.41 |

Agreement compares the **executed target** of the two picks. Two picks that differ in position but point to the same deduplicated action count as agreement.

Note: the earlier GPT-5.5 K=3 labels used a different prompt: our instructions, a JSON-schema answer with no explanation, and shuffled order. That prompt difference is a confound in the GPT-5.5 K=3 vs pre comparison and does not apply here.

## Picks

**Agreement by panel difficulty and depth**

| Panels | Panels (count) | Same executed target (count) | Same executed target (%) |
| --- | ---: | ---: | ---: |
| All | 20,671 | 13,077 | 63.3 |
| 2 distinct targets | 11,009 | 7,692 | 69.9 |
| 3 distinct targets | 6,540 | 3,806 | 58.2 |
| 4 distinct targets | 2,448 | 1,199 | 49.0 |
| 5 distinct targets | 528 | 234 | 44.3 |
| Depth 0 | 7,400 | 4,440 | 60.0 |
| Depth 1–5 | 5,515 | 3,560 | 64.6 |
| Depth 6–11 | 4,933 | 3,193 | 64.7 |
| Depth 12+ | 2,823 | 1,884 | 66.7 |

**Action type of the pick in the 7,594 disagreement panels.** Types come from the candidate's first parsed call. Smaller types are omitted.

| Action type | Pre picks (count) | k3p picks (count) | Change (count) |
| --- | ---: | ---: | ---: |
| click | 4,532 | 4,400 | −132 |
| scroll | 1,271 | 1,008 | −263 |
| back | 337 | 643 | +306 |
| stop / answer | 332 | 479 | +147 |
| navigate (direct URL) | 362 | 157 | −205 |
| new tab | 293 | 352 | +59 |
| type / write | 325 | 417 | +92 |
| wait | 118 | 88 | −30 |

These are the most common switches between pre's pick and k3p's pick:

| Switch | Panels (count) |
| --- | ---: |
| click → a different click | 3,173 |
| scroll → click | 756 |
| click → scroll | 659 |
| scroll → back | 274 |
| click → back | 195 |
| click → stop / answer | 166 |
| navigate → new tab | 134 |
| stop / answer → scroll | 130 |

On all panels, the share of picks in each position (1–5) is:

| Label | Pos 1 (%) | Pos 2 (%) | Pos 3 (%) | Pos 4 (%) | Pos 5 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| pre | 37.8 | 24.8 | 14.9 | 11.5 | 11.0 |
| k3p | 39.6 | 24.2 | 14.6 | 11.8 | 9.9 |

The two labels share almost the same position bias, so the shared prompt and candidate order drive it, not the lookahead.

**Stop/answer picks by depth** (share of each depth band's panels):

| Depth | Panels (count) | Pre stop picks (count) | Pre stop (%) | k3p stop picks (count) | k3p stop (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 7,400 | 308 | 4.2 | 515 | 7.0 |
| 1–5 | 5,515 | 377 | 6.8 | 358 | 6.5 |
| 6–11 | 4,933 | 241 | 4.9 | 194 | 3.9 |
| 12+ | 2,823 | 87 | 3.1 | 93 | 3.3 |

The depth-0 increase matches the qualitative coding: on blocked or anti-bot landing pages, k3p sees that every retry hits the same wall.

**Immediate draw-0 effect of the two picks in disagreements.** All 7,594 disagreement panels have both effects recorded.

| Immediate effect | Pre pick (count) | k3p pick (count) |
| --- | ---: | ---: |
| URL changed | 4,041 | 4,119 |
| Tool response reports an error | 134 | 99 |

Gross effects are similar for the two picks. k3p does not simply prefer actions that navigate somewhere; the qualitative coding below shows the difference lies in what the resulting page contains.

## Reasoning

**Cue frequencies in the explanations (regular-expression counts, share of panels).** "Execution cue" also matches pre wording such as "navigate to" or "result", so compare the two labels rather than reading either number on its own. All 20,671 panels count for "All"; the 7,594 disagreement panels count for "Disagreements".

| Cue | Pre, all (%) | k3p, all (%) | Pre, disagreements (%) | k3p, disagreements (%) |
| --- | ---: | ---: | ---: | ---: |
| Execution / result language | 28.7 | 46.5 | 29.6 | 50.9 |
| Hedging (may, might, likely, appears …) | 47.2 | 20.8 | 48.9 | 19.1 |
| Error / block / no effect | 9.7 | 15.9 | 11.1 | 20.7 |
| Repetition / loop / already | 35.5 | 24.4 | 35.9 | 27.4 |
| Task constraint / filter | 11.1 | 10.3 | 11.5 | 11.2 |
| Premature stop / unverified | 4.5 | 4.9 | 4.3 | 4.2 |

Explanations are about the same length: a mean of 47.1 words for pre and 50.6 for k3p.

**Qualitative coding of 240 disagreements.** The 240 panels are a seeded sample (seed 20261103) from the 7,594 disagreements. Four LLM coders each coded one 60-panel batch, reading the candidates and both explanations only, with no screenshots.

| Primary reason for the different pick | Panels (count) | Panels (%) | Coder: k3p looks better (count) | Coder: pre looks better (count) | Coder: unclear (count) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A. Pre's pick observed to fail (block, CAPTCHA, no-op, zero results, wrong page) | 61 | 25.4 | 34 | 0 | 27 |
| B. k3p's pick observed to reach the needed page or info | 62 | 25.8 | 37 | 0 | 25 |
| D. Stop vs continue | 26 | 10.8 | 16 | 2 | 8 |
| E. Near-tie among equivalent actions | 34 | 14.2 | 0 | 0 | 34 |
| G. k3p over-trusts one sampled continuation | 30 | 12.5 | 0 | 7 | 23 |
| C. Pre's pick avoided for the actor's later steps | 7 | 2.9 | 1 | 0 | 6 |
| F. No lookahead use (ordinary judgement variance) | 20 | 8.3 | 1 | 1 | 18 |
| **Total** | **240** | 100 | **89** | **10** | **141** |

In the same 240 panels:

| Field | Panels (count) | Panels (%) |
| --- | ---: | ---: |
| k3p cites a concrete observed result | 213 | 88.8 |
| Pre misjudged a click's effect from the screenshot | 34 | 14.2 |

Category counts vary by coder: near-ties ranged from 2 to 14 per 60-panel batch. So the category shares carry single-coder uncertainty.

**60 coded agreement panels** (same executed target):

| k3p's basis for the shared pick | Panels (count) |
| --- | ---: |
| Confirmed by the pick's observed result | 35 |
| Rules out the other candidates by their observed failures | 9 |
| Same reasoning as pre (no execution evidence) | 16 |

Pre hedged its explanation in 28 of the 60 agreement panels. Even when the labels agree, k3p usually moves the justification from a guess to an observation, for example: "the follow-up view shows its specific phone number".

### What the lookahead adds

- **Hidden failures.** These include Google or site CAPTCHAs, security-verification pages, 403 errors and maintenance pages, direct URLs that redirect to a homepage, searches that return nothing, and query wordings that a site misinterprets (for example, WolframAlpha returning an aggregate instead of a per-country table).
- **True click effects.** Pre often rejects a click as "missing the link" or "landing in blank space", and k3p sees it open the target. The reverse also happens: a click pre trusted toggles a compare box, opens a video, or lands in the search field instead of on the button.
- **Honest stopping on dead ends.** Pre tends to argue that "done gives up" and picks another retry. k3p sees the retry reach the same block and picks the candidate that reports the access problem without inventing an answer. The opposite also occurs: k3p sometimes continues because scrolling revealed more relevant content than a partial answer covered.

Representative pairs (verbatim, trimmed):

| Case | Pre explanation | k3p explanation |
| --- | --- | --- |
| A — 5-star pancake recipe | "…the click candidates all target a blank area rather than a visible recipe link." (picks scroll) | "Candidate 2 clicks the visible 'Blueberry Buttermilk Pancakes' result, which leads to a recipe page showing a five-star rating, a photo, step-by-step directions…" |
| A — article on a blocked site | "Google can search for the specific GolfWRX article despite the site being blocked." | "Candidate 3 uses Google, but the lookahead shows its search was blocked for unusual traffic…" |
| A — translate 'success' into German | "The Translate link is the most direct route to Cambridge's translation tool…" | "The other candidates all click the Translate link, which leads to a security-verification page and does not advance the task." |
| D — job count on Indeed | "Clicking the visible 'Return home' button is the only candidate that could still make progress…" | "Candidate 1 repeats a navigation action that already led to another blocked Indeed page… Candidates 2–5 accurately report the access barrier…" |

### Where the lookahead hurts

- **Credit for the continuation instead of the action.** k3p sometimes prefers a bare "click the search field" over "type the query and press Enter" because, after the bare click, the actor happened to sort results by rating. It also sometimes ranks two identical searches by which continuation drifted.
- **Transient outcomes treated as decisive.** In one example, an HTTP 429 after Enter led k3p to pick "type without submitting". In another, one failed reload against one successful wait decided the label.
- **Inconsistent conclusions across panels of the same state.** For example, in one state (the Bohol Island area task) k3p draws opposite conclusions about the "Bohol" and "Bohol Island" queries in two panels. This is the signature of single-sample variance.
- **Arbitrary tie-breaks.** For identical actions, pre breaks ties on click centring or key naming, and k3p on whichever continuation went further. Neither carries signal.

| Case | Pre explanation | k3p explanation |
| --- | --- | --- |
| G — recipe comparison | "Candidates 2, 4, and 5 all click the visible search field, enter 'chicken curry,' and submit the search…" | "Candidates 1 and 3 reach the rating-sorted results; candidate 3 clicks the visible search field and its lookahead then advances through the search and sort…" |
| G — statistical learning paper | "Candidate 3 enters the topic query and submits it…" | "Candidate 2 enters the relevant search phrase … without immediately repeating the Enter action that led to an HTTP 429 error." |

## Implications for training on these labels

- **Most of k3p's added signal is in the immediate result, not the 3-step tail.** Categories A and B (51.3% of coded disagreements) rely on what the action itself did. Categories G and C (15.4%) rely on the actor's later steps and are where k3p is most often wrong.
- **About a quarter of the disagreement rate is noise from single sampled continuations and ties.** Categories E and G together make up 26.7%. Several continuation draws per candidate, or collapsing equivalent actions before labelling, would reduce this noise. Head targets already merge equivalent calls (`equivalence_target`), which absorbs part of category E. LoRA targets are a single index.
- **Pre labels rarely look better (4.2%), but their errors are systematic.** Pre misjudges clicks, sticks with direct URLs and Google fallbacks that are blocked, and keeps retrying on dead-end pages. A pre-execution student inherits these errors, and only execution evidence can correct them.

The live outcome comparison is in [Live outcome](#luna-pre-vs-k3p-live-20261103): K = 3 labels beat pre labels for heads (+2.9 Luna, +3.2 GPT-5.5) but not for the Luna LoRA (+0.3).

<details>
<summary>Method, files and caveats</summary>

- **Panels:** `labels/<batch>/panels.jsonl` for p16a, p16d, p16e, p16f and p16g. A panel is 5 candidates drawn from the state's 80 actor samples; "distinct targets" counts deduplicated executed actions. Labels are in `labels-luna/<batch>/{pre,k3p}/<state>-pNN.json`. Labellers: `labels-luna/p16_labels_luna.py` (pre) and `labels-luna/p16_labels_luna_k3p.py` (k3p, which adds the evidence parts to the pre request). All paths are under runtime `rl-actor-recollect-20261016/p16-k3-20261031/`.
- **Analysis:** `analysis/luna_pre_vs_k3p.py`, run as CPU-only job 416334. Action types come from the first parsed call of each candidate. The immediate effect comes from the draw-0 `immediate.json`, comparing the URL before and after and checking the tool response for error terms. Outputs: [`stats.json`](arm_results/luna_pre_vs_k3p_20261103/stats.json), plus the sampled panels (`disagreements.jsonl`, `agreements.jsonl`) and per-panel codes (`coded/`), which stay in runtime only because they contain task payloads.
- **Qualitative coding:** 4 LLM coders, one per 60-panel batch, plus one coder for the agreements. Each coder saw the candidates' calls, the tails of the actor thoughts and both explanations, but no screenshots. The verdicts ("looks better") are plausibility judgements without outcome labels. Aggregate: [`qualitative_coding.json`](arm_results/luna_pre_vs_k3p_20261103/qualitative_coding.json).
- **No ground truth:** p16 has one execution per candidate (the lookahead draw) and no judged continuations, so neither label can be scored against success on these panels.
</details>
