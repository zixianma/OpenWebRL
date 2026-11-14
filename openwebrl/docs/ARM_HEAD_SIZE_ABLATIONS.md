# Selection-head size ablations

**Head size barely matters. From a 23K-parameter linear probe to a 1.8B transformer, every head trained on the same Recipe 3 GPT-5.5 H = 4 panels scores within about 4 pp of the 22.6M default on the SFT 4B actor, and none is reliably better: the only clear leads came from single training seeds and windows (the 4.4M head's +4.3 pp in the first screen vanished over three seeds, −0.2 [−1.8, +1.4]). The two smallest heads (0.25M, 1.0M) score +3 to +4 pp over the default in one window. Selection cost grows with size (15 ms per step at 4.4M, 337 ms at 1.8B, against 474 ms for the 4B LoRA), but end-to-end episode time does not, because generating five candidates dominates.** On the RL 8B actor, with the frozen SFT-4B model re-encoding its candidates, the heads are again close to each other (the seed-0 4.4M head +3.3 [+0.8, +5.9] pp over the default, all on DeepShop).

All success rates in this document are **overall success** (successes / episodes; invalid episodes count as failures); "Mean overall" weights the three benchmarks equally. Differences are in mean overall success, pp [paired 95% bootstrap over tasks common to both arms, 10,000 resamples, seed 20261029, benchmarks resampled independently, per-task success averaged over runs]; bold intervals exclude 0. Compare rows only within a table: absolute rates drift between windows (the SFT 4B baseline moved from 32.2% to 38.7% mean overall across these windows). Valid-only rates are in [ARM_FINAL_RESULTS](ARM_FINAL_RESULTS.md#final-selection-recipe3-head-scale-20261111).

**Setup.** Every head is the joint (bidirectional) transformer selection head with soft cross-entropy on the Recipe 3 GPT-5.5 H = 4 panels (18,781 training panels, 2,110 states, 802 tasks; 1,727 validation panels), learning rate 3e-4, batch 256, 20 epochs, best validation epoch, training seed 0 unless stated; only width (d) and depth (L) change, and the linear probe replaces the transformer with a linear readout of the standardized features. Heads read the actor's hidden states at layers 18/27/36 (prompt end and, per candidate, thought mean, action mean and last token). Evaluation: OpenWebRL-4B-SFT actor (unless stated), N = 5, T 1.0 / top-p 0.9, 30 steps, all 1,045 tasks (Online-Mind2Web 300, WebVoyager 595, DeepShop 150), 1 H200 per run, rolled out judge-deferred and judged afterwards (Online-Mind2Web gpt-5.6-terra, WebVoyager and DeepShop GPT-4o); no record refused.

## Summary: every size against the 22.6M default

| Head | Params | SFT 4B actor: − 22.6M head, pp [95% CI] | Window, runs | RL 8B actor: − 22.6M head, pp [95% CI] |
| --- | ---: | ---: | --- | ---: |
| linear probe | 23K | +0.2 [−2.5, +2.8] | Nov 13, 2 runs | +1.2 [−1.5, +3.9] |
| d8-L1 | 0.25M | **+3.1 [+0.7, +5.6]** | Nov 12–13, 2 runs | — |
| d32-L1 | 1.0M | **+4.0 [+1.6, +6.4]** | Nov 12–13, 2 runs | — |
| d128-L2 | 4.4M | −0.2 [−1.8, +1.4] | Nov 11 evening, 3 seeds pooled | **+3.3 [+0.8, +5.9]** (seed 0) |
| d256-L2 | 9.6M | +2.1 [−1.4, +5.6] | Nov 12, 1 run | — |
| d512-L2 (default) | 22.6M | — | | — |
| d2048-L4 | 272.8M | +3.2 [−0.3, +6.6] | Nov 12, 1 run | +1.8 [−0.8, +4.5] |
| d4096-L8 | 1.8B | −1.1 [−4.3, +2.2] | Nov 11 evening, 2 runs | — |

*Each SFT 4B entry is from its own same-window comparison (tables below); the RL 8B column is from [RL 8B actor](#rl-8b-actor). The 1.8B head was evaluated against the 22.6M default re-run in the same evening window (1 run).*

## SFT 4B actor

<a id="head-size-screen-20261111"></a>
### First screen: one training seed, 2 runs (November 11, 00:59–09:30 PST)

| Head | Params | Online-Mind2Web | WebVoyager | DeepShop | Mean overall | − SFT alone, pp [95% CI] | − 22.6M default, pp [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 4B alone (1 sample) | — | 30.5% | 41.3% | 44.3% | 38.7% | — | — |
| d128-L2 | 4.4M | 34.7% | 52.0% | 54.0% | 46.9% | **+8.2 [+5.7, +10.7]** | **+4.3 [+1.9, +6.7]** |
| d256-L2 | 9.6M | 36.0% | 49.1% | 50.3% | 45.1% | **+6.4 [+4.0, +8.8]** | **+2.6 [+0.0, +5.1]** |
| d512-L2 (default) | 22.6M | 34.7% | 45.7% | 47.3% | 42.6% | **+3.8 [+1.4, +6.3]** | — |
| d2048-L4 | 272.8M | 35.3% | 45.5% | 52.0% | 44.3% | **+5.6 [+3.1, +8.0]** | +1.7 [−0.7, +4.2] |

| Head | Online-Mind2Web valid / episodes | Online-Mind2Web successes | WebVoyager valid / episodes | WebVoyager successes | DeepShop valid / episodes | DeepShop successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 4B alone (1 sample) | 535 / 600 | 183 | 1,079 / 1,190 | 492 | 298 / 300 | 133 |
| d128-L2 | 528 / 600 | 208 | 1,081 / 1,190 | 619 | 298 / 300 | 162 |
| d256-L2 | 524 / 600 | 216 | 1,071 / 1,190 | 584 | 297 / 300 | 151 |
| d512-L2 (default) | 522 / 600 | 208 | 1,068 / 1,190 | 544 | 299 / 300 | 142 |
| d2048-L4 | 526 / 600 | 212 | 1,071 / 1,190 | 542 | 299 / 300 | 156 |

*Superseded for the size comparison by the seed replication below: the 4.4M head's lead here did not replicate.* Evidence: [data](arm_results/rl_integration/selection-head-scale-20261111.json) · [script](arm_results/rl_integration/selection-head-scale-20261111.py). Runtime: `recipe3-ablations-20261104/eval-scale/`.

<a id="head-size-seeds-20261111"></a>
### Training-seed replication and 1.8B (November 11, 16:00–21:00 PST)

Seeds 1 and 2 of the 4.4M and 22.6M heads (2 runs each) with seed 0 of both and SFT 4B alone re-run once in the same window.

| Head | Params | Online-Mind2Web | WebVoyager | DeepShop | Mean overall | − SFT alone, pp [95% CI] | − 22.6M, pp [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 4B alone (1 sample) | — | 26.7% | 40.7% | 29.3% | 32.2% | — | — |
| d128-L2, seed 0 | 4.4M | 35.0% | 50.4% | 50.0% | 45.1% | **+12.9 [+9.2, +16.6]** | — |
| d128-L2, seed 1 | 4.4M | 33.3% | 43.9% | 47.7% | 41.6% | **+9.4 [+6.5, +12.4]** | — |
| d128-L2, seed 2 | 4.4M | 34.3% | 46.5% | 43.3% | 41.4% | **+9.2 [+5.9, +12.5]** | — |
| d512-L2, seed 0 (default) | 22.6M | 38.0% | 46.2% | 46.0% | 43.4% | **+11.2 [+7.8, +14.7]** | — |
| d512-L2, seed 1 | 22.6M | 32.7% | 46.8% | 49.0% | 42.8% | **+10.6 [+7.4, +13.9]** | — |
| d512-L2, seed 2 | 22.6M | 34.8% | 48.2% | 41.7% | 41.6% | **+9.3 [+6.2, +12.5]** | — |
| *4.4M, 3 seeds pooled* | 4.4M | 34.1% | 46.3% | 46.4% | 42.2% | **+10.0 [+7.2, +12.9]** | −0.2 [−1.8, +1.4] |
| *22.6M, 3 seeds pooled* | 22.6M | 34.6% | 47.2% | 45.5% | 42.4% | **+10.2 [+7.4, +13.1]** | — |

| Head | Online-Mind2Web valid / episodes | Online-Mind2Web successes | WebVoyager valid / episodes | WebVoyager successes | DeepShop valid / episodes | DeepShop successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 4B alone (1 sample) | 271 / 300 | 80 | 538 / 595 | 242 | 149 / 150 | 44 |
| d128-L2, seed 0 | 256 / 300 | 105 | 533 / 595 | 300 | 148 / 150 | 75 |
| d128-L2, seed 1 | 523 / 600 | 200 | 1,075 / 1,190 | 523 | 296 / 300 | 143 |
| d128-L2, seed 2 | 529 / 600 | 206 | 1,069 / 1,190 | 553 | 298 / 300 | 130 |
| d512-L2, seed 0 (default) | 263 / 300 | 114 | 538 / 595 | 275 | 149 / 150 | 69 |
| d512-L2, seed 1 | 527 / 600 | 196 | 1,078 / 1,190 | 557 | 298 / 300 | 147 |
| d512-L2, seed 2 | 524 / 600 | 209 | 1,073 / 1,190 | 573 | 298 / 300 | 125 |
| *4.4M, 3 seeds pooled* | 1,308 / 1,500 | 511 | 2,677 / 2,975 | 1,376 | 742 / 750 | 348 |
| *22.6M, 3 seeds pooled* | 1,314 / 1,500 | 519 | 2,689 / 2,975 | 1,405 | 745 / 750 | 341 |

The 1.8B head (d4096-L8, 1.77B parameters), 2 runs in the same evening window, against the 22.6M seed-0 head and SFT 4B alone above:

| Head | Params | Online-Mind2Web | WebVoyager | DeepShop | Mean overall | − SFT alone, pp [95% CI] | − 22.6M default, pp [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d4096-L8 (2 runs) | 1.8B | 33.5% | 45.5% | 48.0% | 42.3% | **+10.1 [+7.1, +13.2]** | −1.1 [−4.3, +2.2] |

| Head | Online-Mind2Web valid / episodes | Online-Mind2Web successes | WebVoyager valid / episodes | WebVoyager successes | DeepShop valid / episodes | DeepShop successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d4096-L8 (2 runs) | 524 / 600 | 201 | 1,074 / 1,190 | 541 | 296 / 300 | 144 |

*This window's single SFT run is low (32.2% mean overall against 38.7% in the morning screen), so the − SFT column is larger here than elsewhere; the size comparison does not use it. Per seed, 4.4M − 22.6M: seed 0 +1.7 [−2.2, +5.7], seed 1 −1.2 [−3.5, +1.2], seed 2 −0.2 [−2.6, +2.2]. The 1.8B head ran through a launcher that serializes the head load (`deferjudge/xbench_defer_headlock.py`; same scores), after the first attempts were OOM-killed.* Evidence: [seeds](arm_results/rl_integration/selection-head-seeds-20261111.json) ([script](arm_results/rl_integration/selection-head-seeds-20261111.py)) · [1.8B](arm_results/rl_integration/selection-evening-20261111.json) (`head_1p8b`; [script](arm_results/rl_integration/selection-evening-20261111.py)). Runtime: `eval-seed/`, `eval-scale/` (1.8B).

<a id="head-size-sameday-20261112"></a>
### All sizes in one window, 1 run (November 12, 00:00–03:20 PST)

| Head | Params | Online-Mind2Web | WebVoyager | DeepShop | Mean overall | − SFT alone, pp [95% CI] | − 22.6M default, pp [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 4B alone (1 sample) | — | 32.7% | 41.7% | 40.7% | 38.3% | — | — |
| d256-L2 | 9.6M | 37.0% | 48.9% | 44.0% | 43.3% | **+5.0 [+1.5, +8.5]** | +2.1 [−1.4, +5.6] |
| d512-L2 (default) | 22.6M | 35.3% | 43.5% | 44.7% | 41.2% | +2.8 [−1.0, +6.7] | — |
| d2048-L4 | 272.8M | 35.0% | 46.7% | 51.3% | 44.4% | **+6.0 [+2.5, +9.6]** | +3.2 [−0.3, +6.6] |
| d4096-L8 | 1.8B | 31.7% | 45.4% | 45.3% | 40.8% | +2.5 [−1.3, +6.1] | −0.4 [−4.1, +3.2] |

| Head | Online-Mind2Web valid / episodes | Online-Mind2Web successes | WebVoyager valid / episodes | WebVoyager successes | DeepShop valid / episodes | DeepShop successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT 4B alone (1 sample) | 266 / 300 | 98 | 536 / 595 | 248 | 150 / 150 | 61 |
| d256-L2 | 261 / 300 | 111 | 539 / 595 | 291 | 149 / 150 | 66 |
| d512-L2 (default) | 261 / 300 | 106 | 529 / 595 | 259 | 149 / 150 | 67 |
| d2048-L4 | 266 / 300 | 105 | 537 / 595 | 278 | 148 / 150 | 77 |
| d4096-L8 | 260 / 300 | 95 | 536 / 595 | 270 | 148 / 150 | 68 |

*With this window every size from 9.6M to 1.8B has three eval runs across windows.* Evidence: [data](arm_results/rl_integration/selection-head-sizes-sameday-20261112.json) · [script](arm_results/rl_integration/selection-head-sizes-sameday-20261112.py). Runtime: `eval-sizes3/`.

<a id="head-size-small-20261113"></a>
### Smaller heads: 0.25M and 1.0M, 2 runs (November 12–13)

| Head | Params | Online-Mind2Web | WebVoyager | DeepShop | Mean overall | − 22.6M default, pp [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d8-L1 | 0.25M | 35.2% | 48.9% | 46.0% | 43.4% | **+3.1 [+0.7, +5.6]** |
| d32-L1 | 1.0M | 37.0% | 49.5% | 46.0% | 44.2% | **+4.0 [+1.6, +6.4]** |
| d512-L2 (default) | 22.6M | 31.2% | 46.1% | 43.3% | 40.2% | — |

| Head | Online-Mind2Web valid / episodes | Online-Mind2Web successes | WebVoyager valid / episodes | WebVoyager successes | DeepShop valid / episodes | DeepShop successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d8-L1 | 518 / 600 | 211 | 1,077 / 1,190 | 582 | 300 / 300 | 138 |
| d32-L1 | 517 / 600 | 222 | 1,082 / 1,190 | 589 | 298 / 300 | 138 |
| d512-L2 (default) | 522 / 600 | 187 | 1,077 / 1,190 | 549 | 296 / 300 | 130 |

*No SFT run in this window.* Evidence: [data](arm_results/rl_integration/selection-head-small-20261113.json) · [script](arm_results/rl_integration/selection-head-small-20261113.py). Runtime: `eval-small/`.

<a id="head-size-probe-20261113"></a>
### Linear probe: 23K parameters, 2 runs (November 13)

| Head | Params | Online-Mind2Web | WebVoyager | DeepShop | Mean overall | − 22.6M default, pp [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear probe | 23K | 35.5% | 48.8% | 46.3% | 43.6% | +0.2 [−2.5, +2.8] |
| d512-L2 (default) | 22.6M | 36.2% | 47.0% | 47.0% | 43.4% | — |

| Head | Online-Mind2Web valid / episodes | Online-Mind2Web successes | WebVoyager valid / episodes | WebVoyager successes | DeepShop valid / episodes | DeepShop successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear probe | 518 / 600 | 213 | 1,073 / 1,190 | 581 | 297 / 300 | 139 |
| d512-L2 (default) | 517 / 600 | 217 | 1,081 / 1,190 | 559 | 297 / 300 | 141 |

*No SFT run in this window. The probe loads through `deferjudge/xbench_defer_probe.py` (LinearProbeHead from `source-probe/`).* Evidence: [data](arm_results/rl_integration/selection-head-probe-20261113.json) · [script](arm_results/rl_integration/selection-head-probe-20261113.py). Runtime: `eval-probe/`.

<a id="head-size-cost-20261112"></a>
## Compute and latency by size (SFT 4B actor)

| Selector | Params | Selector TFLOP / step | of which head, GFLOP | Selection ms, mean / p90 | Model time / episode | End-to-end / episode, mean | Steps / episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d128-L2 head | 4.4M | 0.05 | 0.04 | 15 / 21 | 100 s | 168 s | 13.0 |
| d256-L2 head | 9.6M | 0.05 | 0.10 | 15 / 21 | 98 s | 161 s | 12.7 |
| d512-L2 head (default) | 22.6M | 0.05 | 0.23 | 17 / 24 | 112 s | 182 s | 14.1 |
| d2048-L4 head | 272.8M | 0.05 | 2.73 | 67 / 109 | 112 s | 179 s | 14.0 |
| d4096-L8 head | 1.8B | 0.06 | 17.70 | 337 / 512 | 116 s | 186 s | 13.9 |
| 4B LoRA decoder (ADM-4B) | 4.44B | 30.73 | — | 474 / 758 | 113 s | 181 s | 13.3 |

*Per decision step; the actor's five candidates cost about 61 TFLOP per step in every row. Head selector FLOPs = the actor's one-token follow-up for candidates whose end token was not cached (about 44 GFLOP) plus the head (2 × head params × 5 candidates, "of which head"); decoder = 2 × 4.44B × (uncached selection-prompt + generated tokens). Selection and model time from each step's decision trace (browser excluded); end-to-end = each episode's wall clock including browser. Heads pool every run of the size sweep (November 11–12); decoder from November 11 (3 runs).* Evidence: [data](arm_results/rl_integration/selection-head-scale-cost-20261112.json) · [script](arm_results/rl_integration/selection-head-scale-cost-20261112.py).

<a id="rl-8b-actor"></a>
## RL 8B actor (November 13–14)

The heads read OpenWebRL-4B-SFT features, so with the RL 8B actor (`OpenWebRL/OpenWebRL-8B` 824a54f, 4,096 response tokens) each step the frozen SFT-4B model re-encodes the RL 8B prompt and its five candidates and the unchanged head picks (`head5x`). 2 runs per head, training seed 0. No RL 8B-alone run in this window; the November 4–5 references (3 runs) are cross-window.

| Head | Params | Online-Mind2Web | WebVoyager | DeepShop | Mean overall | − 22.6M head, pp [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear probe | 23K | 42.7% | 56.5% | 48.3% | 49.2% | +1.2 [−1.5, +3.9] |
| d128-L2 | 4.4M | 42.2% | 54.8% | 57.0% | 51.3% | **+3.3 [+0.8, +5.9]** |
| d512-L2 (default) | 22.6M | 43.7% | 53.9% | 46.3% | 48.0% | — |
| d2048-L4 | 272.8M | 38.7% | 55.8% | 55.0% | 49.8% | +1.8 [−0.8, +4.5] |
| *Nov 4–5:* RL 8B alone (1 sample) | — | 38.3% | 56.5% | 57.8% | 50.9% | — |
| *Nov 4–5:* + 4B LoRA, GPT-5.5 H = 4 | 4.4B | 43.6% | 58.8% | 58.4% | 53.6% | — |

| Head | Online-Mind2Web valid / episodes | Online-Mind2Web successes | WebVoyager valid / episodes | WebVoyager successes | DeepShop valid / episodes | DeepShop successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear probe | 530 / 600 | 256 | 1,083 / 1,190 | 672 | 285 / 300 | 145 |
| d128-L2 | 520 / 600 | 253 | 1,077 / 1,190 | 652 | 285 / 300 | 171 |
| d512-L2 (default) | 521 / 600 | 262 | 1,072 / 1,190 | 642 | 286 / 300 | 139 |
| d2048-L4 | 532 / 600 | 232 | 1,085 / 1,190 | 664 | 291 / 300 | 165 |
| *Nov 4–5:* RL 8B alone (1 sample) | 787 / 900 | 345 | 1,752 / 1,785 | 1,008 | 447 / 450 | 260 |
| *Nov 4–5:* + 4B LoRA, GPT-5.5 H = 4 | 790 / 900 | 392 | 1,749 / 1,785 | 1,049 | 442 / 450 | 263 |

*In this window 73 tasks (mostly WebVoyager sites) failed at page load in every run (572 episodes), against almost none on November 4–5, so overall success is lower than on November 4–5 for reasons unrelated to selection; the head-vs-head column is same-window and unaffected. On valid-only success the heads are −0.3 to +3.2 pp against RL 8B alone and at or below the 4B LoRA ([ARM_FINAL_RESULTS](ARM_FINAL_RESULTS.md#final-selection-rl8b-heads-20261114)). The 4.4M lead is all DeepShop and one seed. Re-encoding makes selection slow here (about 4–5 s per step), so this tests transfer, not cost.* Evidence: [data](arm_results/rl_integration/selection-rl8b-heads-20261114.json) · [script](arm_results/rl_integration/selection-rl8b-heads-20261114.py). Runtime: `eval-rl8b-heads/`.
