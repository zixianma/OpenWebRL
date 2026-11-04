# Tinker LoRA: Qwen3.5-9B SFT and outcome-only RL

As of 2026-11-03. This doc ports the OpenWebRL SFT warm start and outcome-only RL baseline to [Tinker](https://thinkingmachines.ai/tinker/), using LoRA on Qwen3.5-9B. Code is in the separate `tinker-ft` workspace (`openwebrl_sft.py`, `openwebrl_eval.py`, `openwebrl_rl.py`). It reuses this repo's episode, environment and judge code unchanged.

**Headline results:**
- **RL at lr 3e-6 lifts the 9B LoRA well above its SFT start.**
  - Best checkpoint: 125/300 (41.7%) on Online-Mind2Web and 66/150 (44.0%) on DeepShop, vs SFT's 102/300 (34.0%) and 34/150 (22.7%).
  - The clean run from SFT has improved at every eval so far.
- **RL at lr 1e-5 peaked at iteration 10 and then collapsed.** Episodes got shorter and outputs degenerated.
- **All scores use the gpt-5.6-terra judge.** It is 4–6 points stricter than the GPT-4o/GPT-4.1 judges behind the OpenWebRL reference numbers.

<a id="tinker-rl-results"></a>
## RL results

Online-Mind2Web (OM2W) uses the 300-task monitor set; DeepShop has 150 tasks. Sampling is T 0.6 / top-p 0.95, with 30 turns. Valid-only excludes tasks whose site failed to load. DeepShop and WebVoyager count CAPTCHA-blocked pages as valid failures, so their valid-only rates are within 1 point of overall unless shown.

| Checkpoint | LR | OM2W successes | OM2W overall | OM2W valid-only | DeepShop successes | DeepShop overall | OM2W + DeepShop successes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT r32 (start) | — | 102 | 34.0% | 37.6% | 34 | 22.7% | 136 |
| Run 1 it10 | 1e-5 | 110 | 36.7% | 41.2% | 54 | 36.0% | 164 |
| Run 1 it20 | 1e-5 | 92 | 30.7% | 34.2% | 55 | 36.7% | 147 |
| Run 1 it30 | 1e-5 | 86 | 28.7% | 32.1% | 51 | 34.0% | 137 |
| **Mixed it20** (run 1 it10, then 3e-6) | 1e-5 → 3e-6 | **125** | **41.7%** | **46.5%** | **66** | **44.0%** | **191** |
| Clean it10 | 3e-6 | 109 | 36.3% | 41.8% | 44 | 29.3% | 153 |
| Clean it20 | 3e-6 | 110 | 36.7% | 43.8% | 57 | 38.0% | 167 |
| Clean it30 | 3e-6 | 123 | 41.0% | 45.9% | 60 | 40.0% | 183 |

**Significance** (exact McNemar paired tests over shared tasks):

| Comparison | OM2W p | DeepShop p |
| --- | ---: | ---: |
| Clean it30 vs SFT | 0.014 | <0.001 |
| Mixed it20 vs SFT | 0.009 | <0.001 |
| Mixed it20 vs run 1 it10 | 0.10 | 0.10 |
| Clean it30 vs mixed it20 | 0.91 | 0.44 |

**Other results:**
- **WebVoyager** (595 tasks) has only been run on run 1 it10: 287 successes (48.2%), vs SFT's 274 (46.1%), p=0.29.
- **The DeepShop gain is mostly from finishing within the turn budget.** At run 1 it10, out-of-turns tasks fell from 54 to 3.

**Run status:**
- **Clean run** (`lora32lr3e6-sft-outcome`): passed 40 iterations and is chained to 90, with OM2W + DeepShop evals at 50, 70 and 90.
- **Mixed run:** parked at iteration 20 as the best checkpoint.

<details>
<summary>Run 1 collapse at lr 1e-5</summary>

| Iteration | 10 | 17 | 20 | 24 | 30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean turns per episode | 6.66 | 6.03 | 5.61 | 3.63 | 3.98 |
| Training success | 43.0% | 34.5% | 39.2% | 29.3% | 20.4% |

At iteration 30, 121 of 300 OM2W episodes hit the generation limit repeating one token ("ContentsContents…"). At lr 3e-6, mean turns stayed at 6–8 through iteration 43. The clean run's lowest so far is 5.96 turns, at iteration 43; we watch for the same early-warning drift.
</details>

<a id="tinker-sft-results"></a>
## SFT results

All LoRA variants land within noise of each other (about ±2.7 points on OM2W), so r32 is the RL starting point. All rows below are evaluated at native screenshot resolution.

| Model | OM2W successes | OM2W overall | OM2W valid-only | WebVoyager successes | WebVoyager overall | DeepShop successes | DeepShop overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base Qwen3.5-9B | 40 | 13.3% | 14.6% | — | — | — | — |
| **SFT r32, LR 4.73e-4** | 102 | 34.0% | 37.6% | 274 | 46.1% | 34 | 22.7% |
| SFT r128, LR 4.73e-4 | 107 | 35.7% | 39.2% | 266 | 44.7% | 31 | 20.7% |
| SFT r128, LR 1e-4 | 94 | 31.3% | — | — | — | — | — |

**Against the OpenWebRL-4B-SFT reference (same judge):**
- OM2W: 102 vs 106/300.
- DeepShop, re-judged with GPT-4o: 40/150 for ours vs 42/150 for the reference, on the same cluster.

**Screenshot resolution was the largest single lever.** Evaluating at native 1280×1000 instead of the 262,144-pixel training size took SFT from 55 to 100/300 on OM2W.

<a id="tinker-setup"></a>
## Setup and differences from the OpenWebRL recipe

| Part | Tinker port | OpenWebRL reference |
| --- | --- | --- |
| Actor | Qwen3.5-9B, LoRA r32 | Qwen3-VL-4B, full fine-tune |
| SFT data | 412 trajectories → 3,085 per-turn examples, native-resolution screenshots | Same trajectories |
| SFT optimization | 3 epochs, batch 128, cookbook LoRA LR 4.73e-4 | 3 epochs, LR 1e-5 |
| RL objective | PPO clip 0.2/0.28, 48 groups × 5 rollouts (nonzero-std filter), 256-turn batches, 2 epochs, Adam (0.9, 0.98), wd 0.1 | Same |
| RL learning rate | 3e-6 (1e-5 in run 1) | 5e-7 |
| RL tasks and rollouts | 2,102 webgym tasks, 15 turns, 1,024 tokens per turn, T 0.8 | Same |
| Reward | Outcome only, `reward_browser` action-history judge | Same |
| Judge | gpt-5.6-terra | GPT-4.1 (RL), GPT-4o (benchmarks) |

Tinker only supports LoRA, so a full fine-tune cannot be loaded back into it for RL. All work stays on Tinker LoRA (decided 2026-10-31).

<details>
<summary>Implementation notes and lessons</summary>

**Implementation:**
- **Prompt format.** The eval reproduces the SFT data's rendering byte for byte. The SFT policy is taken from the canonical data, with `enable_thinking=True` (the prompt ends in `assistant\n<think>\n`). There is no blank line before `<observation>`.
- **Action parsing.** Qwen3.5 XML tool calls are parsed by an added `ToolParser` branch.
- **Loss weighting** matches OpenWebRL: the episode advantage is replicated to each turn, and each sample is averaged over its own tokens. Per-episode 1/T weighting is the next fix to try if turn counts drift down again.
- **Task sampler.** It originally repeated tasks within an epoch. It now draws a fixed permutation per epoch; this was fixed during run 1.

**Infrastructure and guards:**
- **Failure guards.** A Tinker billing block (error 402) and a revoked judge key each wasted hours of rollouts. Jobs now cancel their chain on billing errors and abort when more than half of the rollouts are invalid.
- **Cluster queue.** RL rollouts take about 70 minutes per iteration at 32 concurrent browsers (4 CPUs, ~36 GB). Controllers are CPU-only jobs. The `urgent` QoS requires a GPU, so it can only be used by holding MIG slices.
</details>

<a id="tinker-next"></a>
## Next

- **Clean run:** evals at 50, 70 and 90. If iterations 44–45 keep showing falling turns, evaluate iteration 45 early, and stop below 183 combined successes.
- **Qwen3.8-27B LoRA SFT** on the same data, compared with the 9B's 102 and 34. Qwen3.8's template does not split the data's `…</think>` reasoning, so it is moved into `reasoning_content`. The rendering then matches Qwen3.5 except for one reasoning-effort system line.
- **WebVoyager** on the best RL checkpoint.
