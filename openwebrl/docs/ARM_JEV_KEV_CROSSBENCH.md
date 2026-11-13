# Jev and Kev-4B cross-benchmark evaluation (2026-11-11)

Jev 1.13.0 used as a top-5 selector over the OpenWebRL-4B-SFT actor improves on SFT alone on two of three benchmarks: **+20.7 points on DeepShop** and **+4.9 on WebVoyager**. On Online-Mind2Web it is neutral (+1.7). Kev-4B as a selector is mixed (+13.3, −1.0 and −5.7 points). Jev running as the actor, directly or through our ADM, scores far below the SFT actor.

## Selectors over the SFT-4B actor {#jev-kev-selectors-20261111}

Every condition uses the same SFT-4B actor and the same candidate proposals: `candidate_seed(42, i, t, j)`, with 5 candidates per step for the selector conditions. Overall success divides by all tasks and counts invalid episodes as failures. Valid-only success divides by valid episodes.

| Benchmark | Condition | Successes | Valid | Total | Overall | Valid-only |
|---|---|---:|---:|---:|---:|---:|
| DeepShop | SFT alone | 45 | 150 | 150 | 30.00% | 30.00% |
| DeepShop | SFT + Jev top-5 | 76 | 150 | 150 | 50.67% | 50.67% |
| DeepShop | SFT + Kev-4B | 65 | 148 | 150 | 43.33% | 43.92% |
| Online-Mind2Web | SFT alone | 111 | 268 | 300 | 37.00% | 41.42% |
| Online-Mind2Web | SFT + Jev top-5 | 116 | 264 | 300 | 38.67% | 43.94% |
| Online-Mind2Web | SFT + Kev-4B | 94 | 266 | 300 | 31.33% | 35.34% |
| WebVoyager | SFT alone | 259 | 543 | 595 | 43.53% | 47.70% |
| WebVoyager | SFT + Jev top-5 | 288 | 538 | 595 | 48.40% | 53.53% |
| WebVoyager | SFT + Kev-4B | 253 | 539 | 595 | 42.52% | 46.94% |

Paired differences in overall success, in percentage points, with 95% paired-bootstrap intervals over all tasks:

| Benchmark | Jev − SFT | Kev-4B − SFT | Jev − Kev-4B |
|---|---|---|---|
| DeepShop | +20.7 [+12.7, +28.7] | +13.3 [+4.0, +23.3] | +7.3 [−2.0, +16.0] |
| Online-Mind2Web | +1.7 [−3.7, +7.3] | −5.7 [−10.7, −0.3] | +7.3 [+2.3, +12.3] |
| WebVoyager | +4.9 [+1.0, +8.7] | −1.0 [−4.9, +3.0] | +5.9 [+2.0, +9.8] |

WebVoyager SFT + Jev was interrupted when TypeSafe credits ran out (HTTP 402). The 24 rollouts that aborted on 402 were set aside and rerun, so all 595 tasks have exactly one recorded episode.


## Kev-4B system cost

Released `jaredpalmer/kev-4b` selector plus the SFT 4B actor, from the same November 11 run linked above. Latency sums actor generation and selection time per episode, excluding browser execution. PFLOP uses the same 2 × parameters × tokens proxy as the Recipe 3 plot: actor uncached prompt plus generated tokens, and Kev's reported input tokens. Kev's response is a choice classification; its reported output tokens count serialized answers and are excluded from model compute.

| Benchmark | Recorded episodes | System parameters (B) | PFLOP / episode | Model latency (s / episode) |
|---|---:|---:|---:|---:|
| OM2W | 300 | 8.644 | 1.444 | 100.8 |
| WebVoyager | 595 | 8.644 | 1.470 | 96.3 |
| DeepShop | 150 | 8.644 | 2.569 | 148.4 |
| Equal-weight mean | — | 8.644 | 1.828 | 115.2 |

The Kev text backbone has 4,205,751,296 parameters, counted from the pinned base checkpoint tensor headers, excluding vision and MTP modules; actor parameters are 4,437,815,808. The small pointer head and merged LoRA overhead are excluded, consistent with backbone parameter accounting for other selectors. Selector cache savings are not exposed in saved responses, so its entire reported input is charged: this is a full-input compute proxy, not an exact hardware FLOP count. Costs average recorded episodes (300 / 595 / 150); invalid episodes are included. The plot retains the user's rounded overall rates (OM2W 31.3%, WebVoyager 42.5%, DeepShop 43.3%, reported mean 39.1%). [Aggregate audit](arm_results/rl_integration/selection-kev-costs-20261113.json) · [Reproduce from saved telemetry](arm_results/rl_integration/selection-kev-costs-20261113.py).

## Jev as the actor {#jev-actor-20261111}

- **Direct Jev:** Jev 1.13.0 chooses and runs its own actions. On DONE, GPT-4.1-mini extracts the answer, and the benchmark's own judge grades it.
- **Jev top-5 → ADM:** Jev ranks its actions by P(op)·P(target|op). The ADM `decoder-4b-lora-gpt55-k3-p16` (arm5-9fadc8dc4655) picks one of the top 5, and Jev executes it.

All rates divide by every task. Episodes that end without DONE count as failures. Tasks whose browser failed all 3 allowed attempts count as invalid failures; for direct Jev on WebVoyager, 44 of the 49 invalid tasks are Wolfram Alpha, whose start page timed out. The comparison columns are paired differences over all tasks, with 95% bootstrap intervals.

| Benchmark | Condition | Successes | Valid | Invalid | Total | Overall | Valid-only | − SFT alone | − SFT + Jev top-5 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| WebVoyager | Direct Jev | 61 | 546 | 49 | 595 | 10.25% | 11.17% | −33.3 [−37.3, −29.1] | −38.2 [−42.4, −33.9] |
| WebVoyager | Jev top-5 → ADM | 121 | 539 | 56 | 595 | 20.34% | 22.45% | −23.2 [−27.4, −19.0] | −28.1 [−32.4, −23.7] |
| DeepShop | Direct Jev | 16 | 150 | 0 | 150 | 10.67% | 10.67% | −19.3 [−27.3, −11.3] | −40.0 [−48.0, −32.0] |
| DeepShop | Jev top-5 → ADM | 30 | 149 | 1 | 150 | 20.00% | 20.13% | −10.0 [−18.7, −1.3] | −30.7 [−39.3, −22.0] |
| Online-Mind2Web | Direct Jev | 18 | 269 | 31 | 300 | 6.00% | 6.69% | −31.0 [−36.3, −25.7] | −32.7 [−38.0, −27.3] |
| Online-Mind2Web | Jev top-5 → ADM | 32 | 271 | 29 | 300 | 10.67% | 11.81% | −26.3 [−32.0, −20.7] | −28.0 [−33.7, −22.3] |

Early partial rates were optimistic: top-5 → ADM on DeepShop was 35.6% after 59 tasks and 20.0% at completion.

- **Direct Jev gives up early; answers are not lost.** 439 of 549 recorded WebVoyager episodes and 236 of 290 Online-Mind2Web episodes end BLOCKED, mostly at step 0. Typical causes are an Allrecipes access-issue page, an empty first load on Amazon, and Akamai "Access Denied" pages. The SFT actor sees the same pages but navigates around them. Every DONE episode is extracted and judged.
- **Top-5 → ADM loops instead of stopping.** In the first 78 Online-Mind2Web episodes, 47 hit the 30-action limit, mostly in WAIT loops, because the ADM rarely picks DONE or BLOCKED. It beats direct Jev on every benchmark but still trails SFT alone.

<details>
<summary>Protocol</summary>

All runs follow `EVAL.md` on the `arm` branch.

| Setting | Value |
|---|---|
| Actor | OpenWebRL-4B-SFT; T=1.0, top-p 0.9, top-k server default, 1,024 response tokens |
| Episode | 30 steps; seed 20261018; 8 local browsers |
| Kev-4B | `jaredpalmer/kev-4b`, calibration temperature 2.406 |
| Judge, Online-Mind2Web | gpt-5.6-terra (o4-mini retired) |
| Judge, WebVoyager and DeepShop | Each benchmark's own gpt-4o judge |
| Judging | Deferred: rollouts first, then judged in a separate job |
| Answer extraction (Jev actor) | GPT-4.1-mini, T=0, ≤512 tokens; inputs: task, final URL and title, 16k characters of page text, final screenshot |
| Uncertainty | Paired bootstrap, 10,000 resamples, seed 20261029 |

Invalid episodes are infrastructure or provider failures, such as browser errors or a Jev context-limit HTTP 400 (`max_tokens_exceeded`). Episodes that aborted on TypeSafe HTTP 402 are left out and rerun.

Code: `openwebrl/crossbench_selector_eval.py`, `openwebrl/jev_native_bench_eval.py`, `openwebrl/jev_top5_adm_eval.py`. Runtime artifacts are private, under `evaluations/jev-kev-crossbench-eval-md-20261111/` in scrubbed storage.

</details>

<details>
<summary>Interruptions and cap changes</summary>

The Jev-actor runs were interrupted twice on 2026-11-11, and every interrupted episode was set aside and rerun:

1. **TypeSafe credits ran out (HTTP 402).** The jobs were resubmitted after credits were added.
2. **Approved caps were reached.** Browser launches lost to the outage had counted toward each run's browser-session cap (1.1× tasks). With the user's approval, the caps were raised:
    - Direct Jev: WebVoyager 655 → 720 sessions; Online-Mind2Web 330 → 360.
    - Top-5 → ADM: Online-Mind2Web 330 → 370 sessions; WebVoyager 655 → 690.
    - Top-5 → ADM, DeepShop: GPT-4.1-mini cap $4 → $5. Its judge cap went $3 → $5, because each judge call reserves its worst-case cost (≈$1.50) up front.

Unrecorded tasks failed all 3 browser attempts and are counted as invalid:

| Run | Unrecorded tasks | Main sites |
|---|---:|---|
| Direct Jev, WebVoyager | 46 | Wolfram Alpha (44), ESPN |
| Direct Jev, Online-Mind2Web | 10 | Best Buy, FedEx, UPS, ESPN |
| Jev top-5 → ADM, Online-Mind2Web | 5 | recreation.gov, FedEx, Best Buy |
| Jev top-5 → ADM, WebVoyager | 1 | Wolfram Alpha |

</details>
