# ARM inference-scaling optimization: beating static N5 on cost, latency and success

<a id="adaptive-compute-20261101"></a>

**Interim, November 1, 2026.** The study goal was to beat static N5 (five sampled actions plus a high-effort selector) on the performance/cost tradeoff for the GPT-6 Luna BrowserCode actor. Phases 1 and 2 are about 96% complete, and phase 3 has just started. Tables are refreshed when each phase finishes.

## Takeaways

1. **Prompt caching the selector keeps N5's success at 59% of its cost.** The original N5 selector re-sends a projected copy of the context and gets no prompt-cache hits, so it costs about as much as the five candidate calls combined. Reusing the actor's cached native input instead (the candidate list is appended as a developer message; `tool_choice: none`; JSON-schema answer) costs about one cached candidate call.
   - Cached N5: **82.2% vs 82.9%** mean valid-only success.
   - Paired difference: **−0.5 pp [−3.2, +2.3]**.
   - Cost: **$0.042 vs $0.070 per task**, run on the same days.
   - Per-decision latency is unchanged (12.2 s vs 12.3 s): caching cuts billed input, not processing time.
2. **Gating compute per step loses accuracy.** Three gates were tested:
   - **Agreement gate:** sample two actions and escalate to five plus the selector only when they disagree or one would finish the task. It loses **3.0–3.5 pp** against N5 in both phases (all CIs exclude 0), mostly on DeepShop and OM2W. Steps where two samples "agree" still benefit from wider sampling and selection.
   - **Self-requested deliberation:** the actor calls a `request_deliberation` tool on steps it judges hard. It asks on 63–76% of steps, so it costs as much as N5 without matching N5's success.
   - **The combined gate** matches N5 (+1.2 pp [−1.3, +3.7]) but costs more ($0.079 per task).
3. **Sampling more actions is where the gain comes from.** N5 beats a single sample by **+4.6 to +6.3 pp** (both CIs exclude 0). It does so mainly by cutting episodes that hit the 30-decision limit from 15% to 8% of valid episodes. More reasoning per call does not add success: earlier xhigh and N3 results matched high and N5 ([ARM_FINAL_RESULTS](ARM_FINAL_RESULTS.md#browsercode-inference-scaling-20261031)).
4. **Phase 3 targets latency (in progress).** Both latency methods use the cached selector:
   - **N5 + skip** drops the selector when all five candidates have the same non-final effect, which happens in about 40% of decisions.
   - **N3 + skip** uses three candidates with the same skip.
   - On the first 58 tasks with a valid result in every phase-3 method, N3 + skip used **5.8 s vs 8.0 s per decision and $0.019 vs $0.030 per task** against cached N5. Its success difference is still undetermined at this size.
   - A final-answer check was dropped after this interim look: success +3.5 pp on 58 tasks, which is not significant, at +34% latency and +36% cost.

Mean valid-only success is the equal-weight mean of the OM2W, WebVoyager and DeepShop valid-only rates. Paired differences use tasks valid in both methods, with a benchmark-stratified task bootstrap (2,000 draws). Every method uses the shared GPT-5.6 Terra terminal-success judge.

## Phase 2: cached selector and cached agreement gates

Interim snapshot from 17:20 PT, November 1, 2026. All five methods ran side by side on the same 1,045 tasks.

| Method | OM2W | WebVoyager | DeepShop | Valid tasks (OM2W / WV / DS) | Mean valid-only | Mean overall | $ / task | s / task | Calls / decision | vs N5, pp [95% CI] | vs single, pp [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single | 64.6% | 86.9% | 83.2% | 257 / 572 / 143 | 78.2% | 72.7% | $0.0102 | 133 | 1.00 | −4.6 [−7.2, −1.9] | – |
| **N5 (baseline)** | 70.7% | 88.0% | 90.0% | 256 / 566 / 140 | 82.9% | 76.0% | $0.0704 | 242 | 6.00 | – | +4.6 [+1.9, +7.2] |
| N5, cached selector | 70.8% | 86.8% | 89.1% | 253 / 561 / 138 | 82.2% | 74.5% | $0.0415 | 225 | 6.00 | −0.5 [−3.2, +2.3] | +4.0 [+1.2, +7.0] |
| Agreement gate → 5, cached | 68.4% | 86.5% | 84.5% | 256 / 569 / 142 | 79.8% | 73.7% | $0.0312 | 227 | 3.64 | −3.0 [−5.6, −0.6] | +1.5 [−1.2, +4.4] |
| Agreement gate → 3, cached | 67.3% | 86.9% | 85.1% | 257 / 573 / 141 | 79.8% | 73.8% | $0.0246 | 205 | 2.82 | −3.0 [−5.5, −0.5] | +1.8 [−1.2, +4.7] |

- Per-benchmark columns are valid-only success.
- Mean overall divides successes by all planned tasks (OM2W 300, WebVoyager 595, DeepShop 150) and so falls while the run is incomplete.
- $ / task is model cost per valid task, excluding the judge.
- Calls / decision counts candidate, probe and selector requests.

## Phase 1: agreement and self-requested deliberation

Interim snapshot from 17:20 PT, November 1, 2026. Every method except single uses the original projected (uncached) selector.

| Method | OM2W | WebVoyager | DeepShop | Valid tasks (OM2W / WV / DS) | Mean valid-only | Mean overall | $ / task | s / task | Calls / decision | vs N5, pp [95% CI] | vs single, pp [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single | 67.2% | 82.7% | 79.2% | 265 / 591 / 149 | 76.4% | 73.4% | $0.0100 | 127 | 1.00 | −6.3 [−9.2, −3.5] | – |
| **N5 (baseline)** | 72.1% | 87.2% | 88.5% | 262 / 592 / 148 | 82.6% | 79.0% | $0.0741 | 224 | 6.00 | – | +6.3 [+3.5, +9.2] |
| Agreement gate | 67.7% | 85.9% | 83.3% | 266 / 590 / 150 | 79.0% | 76.2% | $0.0454 | 219 | 3.66 | −3.5 [−6.2, −0.6] | +2.6 [−0.0, +5.5] |
| Self-requested deliberation | 71.4% | 85.9% | 85.1% | 266 / 587 / 148 | 80.8% | 77.3% | $0.0716 | 250 | 4.61 | −2.0 [−4.6, +0.7] | +4.2 [+1.3, +7.1] |
| Self-request + agreement | 73.5% | 88.2% | 90.5% | 264 / 585 / 147 | 84.1% | 80.0% | $0.0785 | 265 | 6.17 | +1.2 [−1.3, +3.7] | +7.4 [+4.6, +10.3] |

Escalated share of decisions: agreement gate 40%, self-requested deliberation 63%, combined 76%.

## Phase 3: latency (in progress)

Before the final-answer check was dropped, on the first 58 tasks with a valid result in all five original phase-3 methods:

| Method | Success | $ / task | s / task | s / decision | Answer checks (accept / revise) |
|---|---:|---:|---:|---:|---:|
| N5, cached selector (control) | 86.2% | $0.0302 | 125 | 8.0 | – |
| N5 + skip | 89.7% | $0.0303 | 120 | 7.0 | – |
| N3 + skip | 87.9% | $0.0186 | 97 | 5.8 | – |
| N5 + final-answer check (dropped) | 89.7% | $0.0412 | 167 | 9.2 | 47 / 9 |
| N5 + skip + final-answer check (dropped) | 89.7% | $0.0376 | 142 | 7.6 | 44 / 11 |

- Success differences on 58 tasks are not distinguishable from noise.
- These tasks are easier than average, so their absolute costs and latencies are below the phase-2 means.
- Phase 3 continues with the first three methods on all 1,045 tasks.

<details>
<summary>Design rationale: where N5 spends time and where it fails</summary>

All figures use valid episodes.
- **Decision time.** In about 3,550 N5 decisions per method (phase-2 shards s00–s03), waiting for the slowest of five parallel candidates takes 6.5 s on average; the third-fastest arrives at 4.7 s and the fastest at 3.6 s. The selector then adds 3.4 s, cached or not, and HTTP admission adds 0.4 s.
- **Unanimous decisions.** In 40.5% of cached-N5 decisions (1,440 of 3,556), all five candidates share one non-final effect signature, so the selector can only choose among equivalent actions.
- **How episodes end.** Of 966 valid phase-1 N5 episodes:

| Outcome | Successes | Failures |
|---|---:|---:|
| Submitted a final answer | 786 | 106 |
| Hit the 30-decision limit | 25 | 49 |

  Single high has 709 / 114 submitted successes / failures and 50 / 96 at the limit, out of 969 valid episodes. N5's gain therefore comes mostly from fewer limit-exceeded episodes.
- **Agreement before phase 1.** An offline replay of saved N5 decisions found 44–48% unanimous by effect signature. The selector picked a minority action in 24% of decisions. Stopping after two agreeing samples would have changed the executed action in about 8% of decisions. The live agreement gate's 3.0–3.5-pp loss is consistent with that.

</details>

<details>
<summary>Methods: actor, arms, gating, budget and validity</summary>

**Actor and protocol.**
- GPT-6 Luna through the Responses API, high reasoning effort, 12,288 output tokens per call.
- BrowserCode harness, at most 30 decisions, a 900-second episode deadline and a 1280×1800 viewport.
- Terminal-success judging by GPT-5.6 Terra (frozen `common_judge`).
- Each phase runs every method on all 1,045 public tasks: OM2W 300, WebVoyager 595 and DeepShop 150.
- Tasks are split into 16 mixed shards; methods run in a per-task seeded random order, side by side.
- An invalid or interrupted (task, method) pair gets one prescribed retry. Valid outcomes are final.

**Arms.** Every multi-candidate arm uses the same candidate request, which is identical to N5's.
- **Agreement gate:** two candidates. It executes directly when their effect signatures agree and neither finishes the task; otherwise it samples up to five (or three) and runs the selector.
  - The effect signature combines action categories, navigation URLs and click/type target literals; read-only probes count as agreeing.
  - A final answer never counts as agreement.
- **Self-requested deliberation:** one probe (two for the combined arm) sees an extra `request_deliberation(reason)` tool and instruction. A request triggers five fresh candidates and the selector; a probe that requested deliberation is never executed.
- **Cached selector:** the actor's native input plus a developer message listing the candidates, with the same instruction and index schema as the projected selector.
- **Skip arms (phase 3):** no selector when every candidate has the same non-final signature.
- **Final-answer check (dropped):** a cached verifier call on the first selected final answer of an episode. A "revise" verdict resamples with the note appended for the rest of the episode.

**Validity and accounting.**
- A provider error invalidates only its episode; usage that can't be confirmed stays charged at its full reservation.
- Start pages that never load from the cluster (for example bestbuy.com, fedex.com, ups.com and theweathernetwork.com) are invalid in every arm, so they drop out of paired comparisons.
- On November 1 the scrubbed filesystem's 110 TB hard quota was exceeded at about 05:02. Episodes running then became invalid and took their prescribed retry. Six budget pools halted; their ledgers were validated before the halts were cleared.

**Budget.**
- Approved API spend was $1,000 (model plus judge) across all phases. A global guard stops new episodes at $990.
- All-attempt spend at the 17:20 snapshot: phase 1 $332.45, phase 2 $235.36. Phase 3 is expected to add about $170.
- Jobs are CPU-only Slurm jobs (account `vlm`).

**Design history.** Phase 2 was designed after an interim look at phase 1, and phase 3 after interim looks at phases 1 and 2. Each phase reruns its own baseline at the same time as its new methods.

</details>

<a id="adaptive-compute-implementation"></a>

<details>
<summary>Implementation: where each method lives in the code</summary>

All code is in [`openwebrl/api_experiments/adaptive/`](../api_experiments/adaptive/) (on `arm` as commits `cebb85e`, `99f901b` and `c27273a`). Every method is one class, [`AdaptiveLunaModel`](../api_experiments/adaptive/adaptive_model.py#L138). It subclasses the frozen Luna BrowserCode model ([`luna_model.py`](../api_experiments/browsercode/models/luna_model.py)) and keeps that model's request path: budget reservation, raw request receipts, model, effort and output limit. Only `query()` changes: it decides how many candidates to sample per decision and whether to run a selector. The arm name selects the method.

| Method | Arm name | Implementation |
|---|---|---|
| Single | `luna_single_high` | [`query`](../api_experiments/adaptive/adaptive_model.py#L260): one candidate, executed |
| N5 (baseline) | `luna_n5_high` | `query`: five parallel candidates ([`_batch`](../api_experiments/adaptive/adaptive_model.py#L159)), then the projected selector [`_select`](../api_experiments/adaptive/adaptive_model.py#L173) |
| N5, cached selector | `luna_n5_cached` | `query`: five candidates, then [`_select_cached`](../api_experiments/adaptive/adaptive_model.py#L188) |
| Agreement gate (→ 5 / → 3) | `adaptive_agree`, `adaptive_agree_cached`, `adaptive_agree3_cached` | `query`: two candidates. If [`agree`](../api_experiments/adaptive/action_signature.py#L60) fails or either would finish, top up to [`ESCALATION_WIDTH`](../api_experiments/adaptive/adaptive_model.py#L59) and select. |
| Self-requested deliberation | `adaptive_self`, `adaptive_self_agree` | `query`: probes get [`DELIBERATION_TOOL`](../api_experiments/adaptive/adaptive_model.py#L64) and [`DELIBERATION_INSTRUCTION`](../api_experiments/adaptive/adaptive_model.py#L70); [`parse_probe`](../api_experiments/adaptive/adaptive_model.py#L129) detects a request |
| N5 / N3 + skip | `n5_skip`, `n3_skip` | [`_choose`](../api_experiments/adaptive/adaptive_model.py#L219): no selector when `agree` holds over all candidates |
| Final-answer check (dropped) | `n5_verify`, `n5_skip_verify` | [`_query_phase3`](../api_experiments/adaptive/adaptive_model.py#L229) and [`_verify`](../api_experiments/adaptive/adaptive_model.py#L205): [`VERIFIER_INSTRUCTION`](../api_experiments/adaptive/adaptive_model.py#L84) and the [`REVISION_PREFIX`](../api_experiments/adaptive/adaptive_model.py#L95) note |

**Shared pieces**

- **Effect signatures.** [`signature`](../api_experiments/adaptive/action_signature.py#L46) in [`action_signature.py`](../api_experiments/adaptive/action_signature.py) reduces a BrowserCode command to its effect: action categories plus navigation URLs and click/type target literals.
- **Cached selector.** `_select_cached` copies the candidates' native payload and appends one developer message holding [`SELECTOR_INSTRUCTION`](../api_experiments/adaptive/adaptive_model.py#L81) and a JSON list of the candidate actions. It sets `tool_choice: none` and a strict `selected_index` JSON schema, so the cached prefix is identical to the candidate calls.
- **Projected selector.** `_select` builds its input with `selector_input` from the frozen support source `openwebrl/sol_serial_parallel_policy.py`. That module isn't in git; its hashed copy is recorded in each runtime's `source-manifest.json`.
- **Episode-scoped failures.** [`EpisodeScopedHalt`](../api_experiments/adaptive/adaptive_model.py#L99) makes a provider or parse error invalidate only its episode. The frozen model would halt the whole shard's budget pool.

**Campaign tooling**

| File | Role |
|---|---|
| [`prepare_campaign.py`](../api_experiments/adaptive/prepare_campaign.py) | Builds a runtime for `--phase 1/2/3`: copies the frozen effort runtime, freezes the 1,045-task roster into 16 seeded shards, and writes per-shard, per-arm USD caps, the global $990 guard and the hashed source manifest |
| [`adaptive_worker.py`](../api_experiments/adaptive/adaptive_worker.py) | Runs one shard slot: per-task seeded arm order ([`arm_order`](../api_experiments/adaptive/adaptive_worker.py#L53)), one retry per pair ([`prior_rows`](../api_experiments/adaptive/adaptive_worker.py#L59)), the global spend stop ([`global_spend`](../api_experiments/adaptive/adaptive_worker.py#L84)) and a stop after 6 consecutive invalid episodes |
| [`run_shard.py`](../api_experiments/adaptive/run_shard.py) | The Slurm controller: browser preflight, two worker slots, heartbeat, deadline, teardown |
| [`analyze_campaign.py`](../api_experiments/adaptive/analyze_campaign.py) | Produces the tables above |
| [`test_adaptive_model.py`](../api_experiments/adaptive/test_adaptive_model.py) | 38 offline checks with a scripted provider |

**Runtimes** are under `/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/`:

| Phase | Directory |
|---|---|
| 1 | `adaptive-compute-20261031-v2/` |
| 2 | `adaptive-compute-phase2-20261031-v1/` |
| 3 | `adaptive-compute-phase3-20261101-v1/` |

Each runtime keeps hashed copies of the sources it ran. Phase 3's original five-arm plan is kept as `campaign-plan-original-5arm.json`. Interim analysis JSON is in `adaptive-analysis-20261101/`. Per-decision records are saved under `shards/<shard>/attempts/*/episodes/<task>-browsercode-<arm>/decisions/`; each holds the candidates, their signatures, the selector or verifier response, the selected index and the decision time.

</details>
