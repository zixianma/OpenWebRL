# Live evaluation on Online-Mind2Web, WebVoyager and DeepShop

This is the canonical protocol behind every live selector and actor result since November 6, 2026. The latest results are in [ARM_FINAL_RESULTS](ARM_FINAL_RESULTS.md#final-selection-recipe3-head-arch-20261107).

The code on `arm` is the exact version those runs used. Each eval plan records sha256 digests of 16 source files, and every file on `arm` matches the digests in the November 7 plans.

## Protocol

| | Online-Mind2Web | WebVoyager | DeepShop |
| --- | --- | --- | --- |
| Tasks | all 300 | all 595 | all 150 |
| Task file (sha256 prefix) | `openwebrl/data/eval/online-mind2web.jsonl` (`8343c23b`) | `openwebrl/data/eval/webvoyager_fara.jsonl` (`ac925331`) | `openwebrl/data/eval/deepshop_with_metadata.jsonl` (`4694e61e`) |
| Judge model | gpt-5.6-terra | gpt-4o | gpt-4o |
| Judge prompt | AgentTrek, final 3 screenshots | WebVoyager, up to 30 screenshots | DeepShop, up to 30 screenshots |
| Slurm time limit | 3.5 h | 5 h | 2.5 h |

- **Judges.**
  - On Online-Mind2Web, gpt-5.6-terra replaced o4-mini after o4-mini was retired on October 23. The prompt, images and parser are unchanged.
  - Compare conditions only within one judge round. Check `judge_model` in the saved verdicts.
- **DeepShop task file.** It is the repository's `deepshop.jsonl` plus a `metadata` block (start URL, intent) that the browser loader needs. The harness pins it by sha256.
- **Actor.** OpenWebRL-4B-SFT (revision `15e777db`), served by sglang on the job's H200.
  - Sampling: temperature 1.0, top-p 0.9, top-k left at the server default, 1,024 tokens per step.
  - Episodes: at most 30 steps, with the full history and only the current screenshot.
- **Selection.**
  - SFT (`sft1`) takes one sample per step.
  - Every selector condition samples 5 candidates per step and executes the selector's pick unchanged.
  - Candidate *j* at turn *t* of task *i* uses sampling seed `sha256("42:i:t:j")` (`candidate_seed`), so all selectors see the same proposals.
- **Browser.** Local Chromium, 8 episodes at a time. One H200, 8 CPUs and 120 GiB per benchmark job.
- **Repeated runs.** Repeated runs reuse every seed and differ only in the output root (`run1`, `run2`, …). Their variation comes from live websites and nondeterministic serving.
- **Same-day controls.** Websites drift, so always rerun SFT, and any reference selector, in the same window. Comparisons across days have been misleading before.

## Metrics

- **Valid episode.** The judge returned a binary verdict, the episode was not aborted, and the judge did not time out (`is_valid_result`).
- **Overall success.** Successes divided by all tasks, so invalid episodes count as failures.
- **Valid-only success.** Successes divided by valid episodes.
- **Mean overall and mean valid-only.** Unweighted means over the three benchmarks.
- **Differences between conditions.** A paired bootstrap over tasks: 10,000 resamples, seed 20261029, each benchmark resampled independently. Results tables report each benchmark valid-only, both means, and differences in separate columns.

## Running it

```bash
# 1. Plans (CPU, seconds): one per benchmark under ROOT/<benchmark>/<condition slug>/<benchmark>/plan.json
scripts/eval/run_crossbench_eval.sh prepare /path/run1 sft1
scripts/eval/run_crossbench_eval.sh prepare /path/run1 head5:/path/head.pt

# 2. Rollouts (GPU, one job per benchmark), judge deferred: records are saved with judge_text "DEFERRED"
QOS=normal scripts/eval/run_crossbench_eval.sh submit /path/run1 sft1

# 3. Judging (CPU only, no GPU), under a per-day API cap in USD; pass ABSOLUTE plan directories
sbatch scripts/eval/rejudge.sbatch 100 /path/run1/deepshop/sft1/deepshop /path/run1/webvoyager/sft1/webvoyager ...
```

- **Conditions.**
  - `sft1`: the actor alone.
  - `head5:<head.pt>`: a selection head that reads the actor's hidden states, captured inside the actor server.
  - `head5x:<head.pt>`: the same head with a separate encoder pass.
  - `arm5:<merged dir>`: a SelectionARM-style 4B decoder, for example the default ADM.
  - `piotr5`: Piotr's released SelectionARM.
- **Resuming.** A job that crashes resumes from its saved records when you resubmit the same `launch-<benchmark>-defer.sbatch`. The usual cause is an intermittent CUDA illegal-address error, in about 1 of 15 jobs.
- **Analysis.** It must refuse records whose `judge_text` is still `DEFERRED`, or "All judge API retry attempts exhausted.".
- **Judge credentials.** The judge key comes from `JUDGE_API_KEY` or the repository's `.env`, and is never logged. The daily ledger is shared by every copy of the wrapper.

## Code

| File | Role |
| --- | --- |
| `openwebrl/crossbench_selector_eval.py` | Protocol, plans, task sources, judge routes, selectors, validity rule |
| `scripts/run_crossbench_selector_eval.py` | `prepare`, `launch-script` and `execute`: starts the sglang actor (and any selector server) and runs the episodes |
| `openwebrl/run_evaluate.py`, `openwebrl/generate_browser.py`, `openwebrl/reward_browser.py` | Episode loop, browser actions, reward dispatch |
| `openwebrl/eval/reward_{online_mind2web,webvoyager,deepshop}.py` | Native judges, one per benchmark |
| `openwebrl/actor_head_capture.py`, `openwebrl/actor_head_scoring.py`, `openwebrl/selection_head.py` | Hidden-state capture inside sglang, and the head selectors |
| `openwebrl/arm_inference.py`, `scripts/serve_arm_sglang.py` | SelectionARM-style decoder selectors |
| `scripts/eval/run_crossbench_eval.sh` | Prepare and submit one condition on all three benchmarks |
| `scripts/eval/crossbench_defer.py`, `defer_launch.sh`, `rejudge.sbatch` | Rollout now, judge later, with a daily cap |

- **Machine-specific paths.** The harness hard-codes checkpoint and runtime paths on the Tillicum cluster: the actor, the Python environment, and the DeepShop pin location. A collaborator elsewhere needs to repoint the constants at the top of `crossbench_selector_eval.py`, place `deepshop_with_metadata.jsonl` where `TASK_SOURCES` expects it (the sha256 check guards it), and set `PYTHON`.
- **Check for reproduction.** `run_crossbench_eval.sh prepare` reproduces the November 6 SFT control plans exactly: the same protocol hash, task-file hash, harness revision (`crossbench-selector-eval-v1`) and task set.
