# ARM inference-scaling optimization: beating static N5 on cost, latency and success

<a id="adaptive-compute-20261101"></a>

**Interim, November 1, 2026.** The study goal was to beat static N5 (five sampled actions plus a high-effort selector) on the performance/cost tradeoff for the GPT-6 Luna BrowserCode actor. Phases 1 and 2 are about 95% complete, and phase 3 has just started. Tables are refreshed when each phase finishes.

## Takeaways

1. **Prompt caching the selector keeps N5's success at 59% of its cost.** The original N5 selector re-sends a projected copy of the context and gets no prompt-cache hits, so it costs about as much as the five candidate calls combined. Reusing the actor's cached native input instead (the candidate list is appended as a developer message; `tool_choice: none`; JSON-schema answer) costs about one cached candidate call.
   - Cached N5: **82.1% vs 82.5%** mean valid-only success.
   - Paired difference: **−0.2 pp [−3.0, +2.5]**.
   - Cost: **$0.042 vs $0.071 per task**, run on the same days.
   - Per-decision latency is unchanged (12.2 s vs 12.3 s): caching cuts billed input, not processing time.
2. **Gating compute per step loses accuracy.** Three gates were tested:
   - **Agreement gate:** sample two actions and escalate to five plus the selector only when they disagree or one would finish the task. It loses **3.3–3.6 pp** against N5 in both phases (all CIs exclude 0), mostly on DeepShop and OM2W. Steps where two samples "agree" still benefit from wider sampling and selection.
   - **Self-requested deliberation:** the actor calls a `request_deliberation` tool on steps it judges hard. It asks on 63–76% of steps, so it costs as much as N5 without matching N5's success.
   - **The combined gate** matches N5 (+0.9 pp [−1.7, +3.4]) but costs more ($0.077 per task).
3. **Sampling more actions is where the gain comes from.** N5 beats a single sample by **+4.3 to +6.3 pp** (both CIs exclude 0). It does so mainly by cutting episodes that hit the 30-decision limit from 15% to 8% of valid episodes. More reasoning per call does not add success: earlier xhigh and N3 results matched high and N5 ([ARM_FINAL_RESULTS](ARM_FINAL_RESULTS.md#browsercode-inference-scaling-20261031)).
4. **Phase 3 targets latency (in progress).** Both latency methods use the cached selector:
   - **N5 + skip** drops the selector when all five candidates have the same non-final effect, which happens in about 40% of decisions.
   - **N3 + skip** uses three candidates with the same skip.
   - On the first 58 tasks with a valid result in every phase-3 method, N3 + skip used **5.8 s vs 8.0 s per decision and $0.019 vs $0.030 per task** against cached N5. Its success difference is still undetermined at this size.
   - A final-answer check was dropped after this interim look: success +3.5 pp on 58 tasks, which is not significant, at +34% latency and +36% cost.

Mean valid-only success is the equal-weight mean of the OM2W, WebVoyager and DeepShop valid-only rates. Paired differences use tasks valid in both methods, with a benchmark-stratified task bootstrap (2,000 draws). Every method uses the shared GPT-5.6 Terra terminal-success judge.

## Phase 2: cached selector and cached agreement gates

Interim snapshot from 13:30 PT, November 1, 2026. All five methods ran side by side on the same 1,045 tasks.

| Method | OM2W | WebVoyager | DeepShop | Valid tasks (OM2W / WV / DS) | Mean valid-only | Mean overall | $ / task | s / task | Calls / decision | vs N5, pp [95% CI] | vs single, pp [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single | 63.7% | 86.8% | 82.7% | 215 / 500 / 133 | 77.7% | 64.0% | $0.0102 | 132 | 1.00 | −4.3 [−6.9, −1.4] | – |
| **N5 (baseline)** | 68.7% | 88.9% | 90.0% | 217 / 494 / 130 | 82.5% | 67.1% | $0.0710 | 236 | 6.00 | – | +4.3 [+1.4, +6.9] |
| N5, cached selector | 68.7% | 87.0% | 90.6% | 214 / 492 / 127 | 82.1% | 65.9% | $0.0420 | 222 | 6.00 | −0.2 [−3.0, +2.5] | +4.1 [+0.9, +7.0] |
| Agreement gate → 5, cached | 66.5% | 86.0% | 83.8% | 215 / 500 / 130 | 78.8% | 64.2% | $0.0313 | 223 | 3.63 | −3.6 [−6.5, −1.0] | +0.7 [−2.2, +3.5] |
| Agreement gate → 3, cached | 66.0% | 87.0% | 84.7% | 215 / 500 / 131 | 79.3% | 64.8% | $0.0246 | 201 | 2.83 | −3.3 [−6.0, −0.7] | +1.5 [−1.7, +4.6] |

- Per-benchmark columns are valid-only success.
- Mean overall divides successes by all planned tasks (OM2W 300, WebVoyager 595, DeepShop 150) and so falls while the run is incomplete.
- $ / task is model cost per valid task, excluding the judge.
- Calls / decision counts candidate, probe and selector requests.

## Phase 1: agreement and self-requested deliberation

Interim snapshot from 13:30 PT, November 1, 2026. Every method except single uses the original projected (uncached) selector.

| Method | OM2W | WebVoyager | DeepShop | Valid tasks (OM2W / WV / DS) | Mean valid-only | Mean overall | $ / task | s / task | Calls / decision | vs N5, pp [95% CI] | vs single, pp [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single | 66.8% | 83.2% | 79.4% | 244 / 546 / 141 | 76.5% | 68.4% | $0.0100 | 127 | 1.00 | −6.3 [−9.3, −3.4] | – |
| **N5 (baseline)** | 71.8% | 87.1% | 89.1% | 241 / 551 / 138 | 82.7% | 73.4% | $0.0729 | 221 | 6.00 | – | +6.3 [+3.4, +9.3] |
| Agreement gate | 67.5% | 86.9% | 83.0% | 243 / 542 / 141 | 79.1% | 70.6% | $0.0451 | 218 | 3.67 | −3.5 [−6.2, −0.8] | +2.6 [−0.2, +5.4] |
| Self-requested deliberation | 71.1% | 86.5% | 84.1% | 242 / 542 / 138 | 80.6% | 71.2% | $0.0712 | 247 | 4.62 | −2.2 [−4.9, +0.5] | +4.1 [+1.2, +7.0] |
| Self-request + agreement | 72.9% | 88.0% | 90.6% | 240 / 541 / 139 | 83.8% | 74.1% | $0.0772 | 261 | 6.17 | +0.9 [−1.7, +3.4] | +7.5 [+4.5, +10.4] |

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
- **Agreement before phase 1.** An offline replay of saved N5 decisions found 44–48% unanimous by effect signature. The selector picked a minority action in 24% of decisions. Stopping after two agreeing samples would have changed the executed action in about 8% of decisions. The live agreement gate's 3.5-pp loss is consistent with that.

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
- All-attempt spend at the 13:30 snapshot: phase 1 $307.00, phase 2 $208.74. Phase 3 is expected to add about $170.
- Jobs are CPU-only Slurm jobs (account `vlm`).

**Design history.** Phase 2 was designed after an interim look at phase 1, and phase 3 after interim looks at phases 1 and 2. Each phase reruns its own baseline at the same time as its new methods.

</details>

<details>
<summary>Code and runtime</summary>

**Code** (branch `arm-selection-head`, commits `2e735b9`, `12b1a77`, `67aa038`), all in `openwebrl/api_experiments/adaptive/`:

| File | Contents |
|---|---|
| `adaptive_model.py` | All arms |
| `action_signature.py` | Effect signatures |
| `adaptive_worker.py`, `run_shard.py` | Shard worker and Slurm controller |
| `prepare_campaign.py` | Campaign setup (`--phase 1/2/3`) |
| `analyze_campaign.py` | Analysis |
| `test_adaptive_model.py` | 38 offline checks |

**Runtimes** are under `/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations/`:

| Phase | Directory |
|---|---|
| 1 | `adaptive-compute-20261031-v2/` |
| 2 | `adaptive-compute-phase2-20261031-v1/` |
| 3 | `adaptive-compute-phase3-20261101-v1/` |

Phase 3's original five-arm plan is kept as `campaign-plan-original-5arm.json`. Interim analysis JSON is in `adaptive-analysis-20261101/`.

</details>
