# RL reward design: grounded answers and efficiency

Proposal and handover, October 24, 2026. Status: **design only**. Nothing here has been run in
training. Offline re-scoring (step 2 below) needs only CPU and a small API budget. A training run
needs separate GPU and judge-budget approval from the user.

## Problem in one paragraph

Outcome-only GRPO with the GPT-4.1 `action_history` judge fixed the agent running out of turns,
but replaced it with **early wrong answers**:

- **Running out of turns stopped.** In the 56-group run, held-out turn-limit failures went from
  59 to 0 between iterations 10 and 90.
- **Wrong answers rose.** Wrong-answer failures went from 16 to 56.
- **Failed episodes got shorter,** from about 19 to about 11 steps.

A second judge and an answer-grounding audit show that a growing share of GPT-4.1 "successes"
are answers that **look right but do not satisfy the task's constraints**. The policy is
learning what GPT-4.1 accepts, so the fix belongs in the success signal. A step penalty would
make premature stopping worse.

## Evidence

All numbers come from saved held-out episodes of the outcome-only 56-group run. Protocol: local
browser, GPT-4.1/action_history, temperature 0, 30 turns, full300.

### Second-judge audit

- **Second judge:** gpt-5.6-terra, given the exact training prompt and inputs.
- **Pipeline check:** re-judging 10 episodes with GPT-4.1 reproduced 9 of the original verdicts.
- **Sample:** 75 random episodes per checkpoint × verdict cell.

| Iteration | GPT-4.1 verdict | Episodes | Second judge agrees | Second judge flips |
|---|---|---|---|---|
| 10 | success | 75 | 56 | 19 |
| 10 | failure | 75 | 72 | 3 |
| 90 | success | 75 | 40 | 35 |
| 90 | failure | 75 | 74 | 1 |

GPT-4.1 failures are almost always confirmed. The disagreement is in GPT-4.1 **successes**,
and it nearly doubles over training (19 → 35 of 75).

### Answer-grounding audit

Coverage: all 150 GPT-4.1 successes from the audit above.

- **Inputs to the vision model (gpt-5.6-terra):**
  - the task and the final answer;
  - every step as numbered items: reasoning `R<k>`, action `A<k>` and browser text response `T<k>`;
  - the last 6 screenshots `S<k>`, each taken before step k.
- **What it does:** splits the answer into atomic claims and labels each one `supported`,
  `partial` or `not_found`, citing evidence ids.
- **Reasoning is not evidence:** reasoning ids may be cited, but a claim that appears only in
  the agent's reasoning counts as `not_found`.
- **Task completion:** it also reports whether the answer completes the task.

| Iteration | Second judge | Episodes | Claims | Supported | Partial | Not found | Episodes with ≥1 not-found claim | Answer completes task |
|---|---|---|---|---|---|---|---|---|
| 10 | success | 56 | 362 | 278 | 66 | 18 | 11 | 50 |
| 10 | failure | 19 | 114 | 64 | 33 | 17 | 10 | 3 |
| 90 | success | 40 | 284 | 199 | 58 | 27 | 10 | 34 |
| 90 | failure | 35 | 272 | 123 | 84 | 65 | 27 | 3 |

- **Episodes both judges accept are mostly grounded.** About 77% of their claims are
  supported, and roughly 1 in 5 episodes has a not-found claim.
- **At iteration 90, episodes the second judge rejects are poorly grounded:**
  - only 123 of 272 claims are supported;
  - 27 of 35 episodes have a not-found claim;
  - only 3 of 35 complete the task.
- **The typical failure is an unchecked constraint, not an invented value.** The reported values
  are usually on screen. What is missing is a task requirement:
  - a "most / best / top" claim with no ranking observed;
  - the wrong date range, radius or filter;
  - a required step never done (for example, an event never opened, nothing added to a cart);
  - "nothing exists" concluded from one search.
- **Some claims are invented.** 20 of the 127 not-found claims are cited only to the agent's
  own reasoning.
- **Evidence is almost all visual:** 1,350 screenshot citations, 104 browser-response citations
  and 42 reasoning citations.

**Caveats**

- The grounding model is the same model as the second judge, so their agreement is **not**
  independent confirmation.
- Human labels on the private spot-check page are the intended ground truth and are pending.
  The page shows 40 flipped and 10 agreed GPT-4.1 successes; labels are stored in the browser
  and exported as JSON.
- The audit covers one run at two checkpoints.

### Re-scoring the held-out evaluations with the saved audits

No new API calls. The gate `G` (no `not_found` claim and `answer_completes_task`, `partial`
allowed) is applied to the 75 audited GPT-4.1 successes per checkpoint. The kept fraction is
then scaled to the full valid-only cohort, because the audit sample is stratified by verdict.

**Under the stricter rewards, held-out success does not rise between iterations 10 and 90.**

| Reward | Iteration 10 valid-only success | Iteration 90 valid-only success | Change, 90 − 10 | 95% interval of change |
|---|---|---|---|---|
| (a) GPT-4.1 alone | 35.1% | 45.1% | +10.1 pp | — (full cohort, no audit sampling) |
| (b) `J · G` (no checklist) | 22.0% | 16.3% | −5.7 pp | −11.9 to +0.7 pp |
| gpt-5.6-terra as judge | 28.8% | 24.8% | −4.0 pp | −10.6 to +2.9 pp |

- **Denominators:** 251 valid episodes at iteration 10 (88 GPT-4.1 successes) and 237 at
  iteration 90 (107), from [RL_RESULTS.md](RL_RESULTS.md).
- **`J · G` scaling:** GPT-4.1 successes × the kept fraction among audited successes (47 of 75 at
  iteration 10; 27 of 75 at iteration 90). The gate cannot add successes, so GPT-4.1 failures
  stay at 0.
- **gpt-5.6-terra scaling:** both strata are reweighted, including the few GPT-4.1 failures it
  accepts (3 of 75 and 1 of 75).
- **Interval:** bootstrap over the audit samples only (4,000 resamples). It leaves out
  evaluation-to-evaluation noise, so the real uncertainty is wider.

**How the gate compares with the second judge** (audited GPT-4.1 successes, 75 per checkpoint):

| Iteration | Both accept | Only `G` accepts | Only second judge accepts | Both reject |
|---|---|---|---|---|
| 10 | 44 | 3 | 12 | 16 |
| 90 | 27 | 0 | 13 | 35 |

- **`G` rejects every second-judge rejection at iteration 90 (35 of 35).** It also rejects about
  a quarter of second-judge acceptances (12 of 56; 13 of 40). The human labels should settle
  these 25 cases. They are where the gate may be too strict.
- **Also rejecting `partial` claims is far too strict.** That version keeps 22 and 12 of 75,
  and it rejects 35 and 28 second-judge acceptances.
- **Training signal:** at iteration 10 the gate keeps about 63% of GPT-4.1 successes. Expect
  correspondingly more all-zero GRPO groups early on (see Risks).
- **Same caveat as above:** `G` and the second judge use the same model.

## Proposed reward

**Base success is gated, not augmented.** Let `J` be the current GPT-4.1 `action_history`
verdict (0/1) and `G` the grounding verdict:

```
G = 1  iff  no claim is not_found
         and answer_completes_task
         and every extracted task constraint is satisfied (see below)
success = J · G
reward  = success + bonus_eff          # bonus_eff = 0 unless success = 1
```

**Why a gate, not an additive bonus.** An additive "grounded claims" bonus can be farmed by
padding the answer with easy, true, visible claims. A gate only removes credit from answers
GPT-4.1 already passes, so it cannot reward anything GPT-4.1 rejects.

**Task-constraint checklist (WebJudge-style key points).**

- **Extraction:** once per task, offline, a model lists the task's explicit constraints (sort
  order, filters, date range, location/radius, quantity, required end state such as "added to
  cart"). This costs roughly $1 for the full task pool.
- **Storage:** cache the checklist with the task and freeze it before training.
- **Check:** the grounding judge marks each constraint satisfied / not satisfied / not
  verifiable, citing `S`/`T` ids. Any constraint that is not satisfied or not verifiable sets
  `G = 0`.
- **Fit to the data:** this targets the failure mode the audit found.

**Efficiency bonus, successes only.**

- **Formula:** among successful rollouts, `bonus_eff = λ · (1 − steps / max_steps)` with a
  small `λ` (start at 0.1).
- **Step cap:** `max_steps` is the **training** cap, `BROWSER_MAX_STEPS=15` in the uniform-G8
  recipe. It is not the 30-turn evaluation cap.
- **Why successes only:** failures get 0, so stopping early with a wrong answer is never paid.
  Within a GRPO group, this ranks shorter successes above longer ones without changing which
  rollouts count as successes.
- **Do not add a per-step penalty to failures.** The current failure mode is stopping too
  early, not running too long.

**Optional, ARM-compatible.** Add action-level credit for verification actions (applying a
filter or sort, opening a detail page, confirming an end state). The running action-only ARM
lineage (`arm-mixed-action-only-383265`, to iteration 60) tests a generic action-level signal.
Wait for its iteration-60 result before combining.

### Cost

- **Per episode:** grounding cost about **$0.037** with gpt-5.6-terra (high-detail images,
  6 screenshots).
- **Per iteration:** uniform G8 collects 48 prompts × 8 = **384 episodes**. Grounding only the
  GPT-4.1 successes (about a third) costs about **$5 per iteration**.
- **Per run:** about **$450 for 90 iterations**, plus the existing GPT-4.1 judge cost.
- **Cheaper variants** (must be validated offline first):
  - fewer screenshots or low-detail images;
  - text-only grounding over `T<k>` plus the final screenshot;
  - a smaller grounding model.

### Risks

- A VLM grounding judge can also be gamed. Answers that copy visible text while ignoring the
  task would pass the claim check, which is why the constraint checklist is part of `G`.
- Stricter rewards lower the success rate early in training and can increase the share of
  all-zero groups, which carry no GRPO signal. Check the fraction of groups with mixed outcomes
  at step 0 under the gated reward (compare with the pass@K run, job 385599) before training.
  If too many groups become all-zero, consider G16.
- Judge latency: grounding adds one VLM call per GPT-4.1 success. Run it concurrently with the
  rest of the reward batch; do not serialize it after collection.

## Plan for the next agent

1. **Collect human labels** (user action). Ask the user for the exported spot-check labels JSON.
   Without labels, the comparison in step 2 is judge versus judge only.
2. **Offline re-scoring** (CPU job; under $15 API; no approval needed beyond the API cap the
   user sets).
   - Episodes: saved iteration-10 and iteration-90 held-out episodes of the 56-group run.
   - Candidate rewards:
     - (a) GPT-4.1 alone;
     - (b) `J · G` without the checklist;
     - (c) `J · G` with the checklist.
   - Report:
     - agreement with human labels (precision/recall of "success");
     - the success rate each reward would assign (done for (a) and (b) from the saved audits;
       see [re-scoring](#re-scoring-the-held-out-evaluations-with-the-saved-audits));
     - how often each reward would have changed the sign of a GRPO advantage within a group,
       using saved training groups if available (outcome-only 48 archive, iterations 24–100).
   - Reuse the existing audit code rather than writing new judge plumbing.
3. **Training** (needs explicit GPU and judge budget approval). Proceed only if (b) or (c)
   clearly beats (a) on human labels.
   - Run a fresh uniform-G8 run from **iteration 0**: SFT checkpoint, fresh optimizer and
     scheduler, initial task cursor. This follows the project rule that new interventions start
     at iteration 0.
   - Change only the reward.
   - Full300 evaluations every 10 iterations.
   - Compare with uniform G8 at iterations 70/80/90 (running now, jobs 385579 → 385580).
   - Report held-out success under **both** GPT-4.1 and the grounded reward, so the gain is not
     measured only by the reward being optimized.

## Judge specifications as run (for reuse)

These are the exact settings behind the evidence tables. Reuse them as the starting point for the
grounding term `G`; change one thing at a time and re-validate offline.

<details><summary>Answer-grounding judge (<code>scripts/audit_answer_grounding.py</code>)</summary>

**Model and call.**

- Model: `gpt-5.6-terra`, OpenAI chat completions, `response_format={"type": "json_object"}`,
  `seed=42`, default temperature.
- Calls run with concurrency 4.
- One call per episode, no retries beyond the client's defaults.
- Unparseable replies are stored raw with `grounding=null`. None occurred in the 150 episodes.

**Inputs, built from the saved per-turn trajectory.**

- **Task:** `metadata.intent` of the first turn.
- **Final answer:** the `response` argument of the last `done` tool call. The tool call uses
  key `arguments`, not `args`.
- **Per step `k` (1-based):**
  - `R<k>`: the turn's response with the `<tool_call>…` part removed and think tags stripped,
    last 700 characters;
  - `A<k>`: the tool-call JSON;
  - `T<k>`: `metadata.step_tool_responses[*].tool_response`, each truncated to 400 characters
    and joined with `|`.
- **Screenshots:** the last image of each turn that has one. Keep the last 6, labeled `S<k>`
  (the page before acting at step k), sent as `detail: high` image parts after the text.
  The training judge shows only the last 3.

**System prompt (verbatim).**

```
You audit whether a web agent's final answer is grounded in what it actually observed.
You get the task, the agent's final answer, and the trajectory as numbered items:
  R<k> = the agent's own reasoning before step k (NOT evidence by itself),
  A<k> = the action it took at step k,
  T<k> = the browser's text response to step k,
  S<k> = a screenshot of the page the agent saw before acting at step k.
Split the final answer into its atomic factual claims (ignore politeness and restated task text).
For each claim decide whether the observations support it:
  "supported": a screenshot or browser response shows it; cite those ids.
  "partial": only part of it is visible, or it is only weakly implied; cite ids and say what is missing.
  "not_found": no screenshot or browser response shows it (it may appear only in reasoning; cite R ids if so).
Also report whether the answer actually completes the task. Reply with JSON only:
{"claims":[{"claim":"...","status":"supported|partial|not_found","evidence":["S3","T4"],"note":"..."}],
 "answer_completes_task": true|false, "unsupported_claims": <int>, "summary": "<one sentence>"}
```

User message: `TASK: …\n\nFINAL ANSWER: …\n\nTRAJECTORY:\n` followed by the `R/A/T` blocks
separated by blank lines, then `S<k>:` text labels each followed by its image.

**Aggregation used in the tables.**

- Statuses outside the three labels count as `not_found`.
- "Episodes with ≥1 not-found claim" counts episodes with any `status == not_found`.
- "Answer completes task" counts `answer_completes_task == true`.
- The proposed `G` would be: no `not_found` claim **and** `answer_completes_task`. `partial` is
  allowed, pending human-label calibration.

**Cost:** 150 episodes for about $5.5, about $0.037 per episode. Budget accounting reserves an
upper-bound cost before each request ($2.5 / $15 per M input/output tokens) and settles the
actual usage afterwards.

</details>

<details><summary>Second-judge audit (<code>scripts/audit_judge_agreement.py</code>)</summary>

- **Prompt and inputs:** exactly the training judge,
  `reward_browser.compute_judge_reward_actionhistory`: the same action-history text, final
  answer and last **3** trajectory screenshots (`_collect_recent_trajectory_images(samples, 3)`).
  Only the model changes.
- **Sample rebuild:** `Sample.from_dict` for every saved turn. The judge runs on the turn with
  `metadata.is_last_turn`.
- **Episodes:** valid held-out episodes of the 56-group run at iterations 10 and 90, stratified
  by the original GPT-4.1 verdict. 75 per cell, `random.Random(0)`.
- **Pipeline check first:** 10 episodes re-judged with GPT-4.1 itself; 9 matched.
- **Budget:** the shared budget file caps the audit and the grounding check together at $30
  (spent $11.20).

</details>

<details><summary>Where the code lives</summary>

- **Location:** both scripts are committed on the local feature branch `arm-selection-head`,
  commits `a613e86` (audit) and `61a1827` (grounding). They have not been pushed to GitHub yet.
- **Imports:**
  - `audit_answer_grounding.py` imports `TOOL_CALL` from `scripts/analyze_rl_eval_failures.py`
    and `Budget`, `capped_client` and `load_episode` from `audit_judge_agreement.py`.
  - Both import `openwebrl.reward_browser` for the served OpenAI client.
- **How to run:** run them as CPU Slurm jobs; rebuilding the trajectory samples does not fit the
  login node's memory. Credentials come from the repo `.env`.

</details>

## Code and data pointers

| What | Where |
|---|---|
| Training reward entry | `openwebrl/reward_browser.py` (`reward_func`, `_score_single_sample`, `compute_judge_reward_actionhistory`) |
| Second-judge audit (budget-capped client, sample rebuild) | `scripts/audit_judge_agreement.py` |
| Answer-grounding audit (prompt, claim schema, summary) | `scripts/audit_answer_grounding.py` |
| Spot-check page builder (private output) | `scripts/build_judge_spotcheck.py` |
| Failure taxonomy | `scripts/analyze_rl_eval_failures.py`; results in [RL_BEHAVIOR.md](RL_BEHAVIOR.md#why-held-out-episodes-fail) |
| Aggregate audit summaries | [second judge](arm_results/rl_integration/behavior/judge-audit-second-judge-20261024.json), [grounding](arm_results/rl_integration/behavior/judge-audit-grounding-20261024.json) |
| Audit outputs (private, per-episode) | runtime `analysis/judge-audit-20261024/` (`verdicts.jsonl`, `grounding.jsonl`, `summary.json`, `grounding-summary.json`, `budget.json`) |
| Uniform-G8 recipe | runtime `training-controls/uniform8-original2102-20261006/to90-20261024/campaign_recipe.py` |

Per-episode audit files and the spot-check page contain task payloads. Keep them private: do not
commit or publish them.
