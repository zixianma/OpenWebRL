# Jev and Kev-4B cross-benchmark evaluation (2026-11-11)

Jev 1.13.0 used as a top-5 selector over the OpenWebRL-4B-SFT actor improves on SFT alone on two of three benchmarks: **+20.7 points on DeepShop** and **+4.9 on WebVoyager**. On Online-Mind2Web it is neutral (+1.7). Kev-4B as a selector is mixed (+13.3, −1.0 and −5.7 points). Jev running as the actor, directly or through our ADM, scores far below the SFT actor. The Jev-actor cohorts stopped a few tasks short of complete when they hit approved browser-session or spending caps.

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

## Jev as the actor {#jev-actor-20261111}

- **Direct Jev:** Jev 1.13.0 chooses and runs its own actions. On DONE, GPT-4.1-mini extracts the answer, and the benchmark's own judge grades it.
- **Jev top-5 → ADM:** Jev ranks its actions by P(op)·P(target|op). The ADM `decoder-4b-lora-gpt55-k3-p16` (arm5-9fadc8dc4655) picks one of the top 5, and Jev executes it.

These cohorts stopped short of all tasks when they hit approved browser-session or spending caps, so each is compared with SFT alone on the same tasks only. Episodes that end without DONE count as failures. Early partial rates were optimistic: top-5 → ADM on DeepShop was 35.6% after 59 tasks and 20.5% after 146.

| Benchmark | Condition | Done | Total | Valid | Successes | Overall | SFT alone, same tasks | Difference [95% interval] |
|---|---|---:|---:|---:|---:|---:|---:|---|
| WebVoyager | Direct Jev | 545 | 595 | 544 | 60 | 11.0% | 47.2% | −36.1 [−40.7, −31.7] |
| WebVoyager | Jev top-5 → ADM | 582 | 595 | 538 | 121 | 20.8% | 44.5% | −23.7 [−28.2, −19.2] |
| DeepShop | Direct Jev | 103 | 150 | 103 | 15 | 14.6% | 30.1% | −15.5 [−25.2, −5.8] |
| DeepShop | Jev top-5 → ADM | 146 | 150 | 145 | 30 | 20.5% | 30.1% | −9.6 [−18.5, −0.7] |
| Online-Mind2Web | Direct Jev | 280 | 300 | 263 | 18 | 6.4% | 38.2% | −31.8 [−37.5, −26.4] |
| Online-Mind2Web | Jev top-5 → ADM | 276 | 300 | 262 | 30 | 10.9% | 39.5% | −28.6 [−34.8, −22.5] |

Overall here divides by tasks done.

- **Direct Jev gives up early; answers are not lost.** 438 of 545 WebVoyager episodes and 230 of 280 Online-Mind2Web episodes end BLOCKED, mostly at step 0. Typical causes are an Allrecipes access-issue page, an empty first load on Amazon, and Akamai "Access Denied" pages. The SFT actor sees the same pages but navigates around them. Every DONE episode is extracted and judged.
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
<summary>Why the Jev-actor cohorts are incomplete</summary>

| Cohort | Done | Total | Stopping cap | Used / cap |
|---|---:|---:|---|---|
| Direct Jev, WebVoyager | 545 | 595 | Browser sessions | 655 / 655 |
| Direct Jev, DeepShop | 103 | 150 | Halted with the WebVoyager run (shared job) | 108 / 165 sessions |
| Direct Jev, Online-Mind2Web | 280 | 300 | Browser sessions | 330 / 330 |
| Jev top-5 → ADM, WebVoyager | 582 | 595 | Browser sessions | 655 / 655 |
| Jev top-5 → ADM, Online-Mind2Web | 276 | 300 | Browser sessions | 330 / 330 |
| Jev top-5 → ADM, DeepShop | 146 | 150 | Halt with no cap reached; likely the $4 GPT-4.1-mini reservation (unverified) | $3.12 / $4.00 |

Session caps were 1.1× tasks per cohort and count every browser launch, including those lost to the 402 outage.

</details>
