# Online RL with ARM turn credit: collaborator handoff

<a id="rl-handoff-20261110"></a>
_Written November 10, 2026. Results are in [ARM_FINAL_RESULTS.md](ARM_FINAL_RESULTS.md#final-rl-arm-turn-credit-20261108); this document covers how to continue the runs._

`RT = /gpfs/scrubbed/zixianma/openwebrl-runtime/` (UW Tillicum, readable by all cluster users). Every `RT/...` path below is cluster-only unless the table in [§2](#rl-handoff-where) says otherwise.

## 1. State of the runs

**Headline.** In a same-day iteration-90 evaluation of all nine runs (three full-300 repeats each, paper actor sampling, offline gpt-5.6-terra judge, November 9–10), every run that keeps mixed groups beats outcome-only 48 by 3.6–6.2 pp (p ≤ 0.044). Gate B scores 37.33% vs 32.11% (outcome-only 48) and 30.89% (outcome-only 56); 40M+16F (38.33%), 48M+16F and Additive are level with Gate B. Failures in main batch is not distinguishable from outcome-only. One training run per method. Tables and paired tests: [ARM_FINAL_RESULTS.md](ARM_FINAL_RESULTS.md#final-rl-iter90-terra-sameday-20261111). Against the SFT start model under the same protocol (November 11), outcome-only 48 gains +4.1 pp and Gate B +8.6 pp ([table](ARM_FINAL_RESULTS.md#final-rl-iter0-terra-anchor-20261111)).

| Run | Recipe change vs Gate B | Iterations done | Latest durable checkpoint | Frozen source (`RT/`) | W&B (`zixianma/openwebrl`) |
| --- | --- | ---: | --- | --- | --- |
| Outcome-only, 48 groups | No ARM credit; 4 H200, TP4, 32 browsers through 90 | 100 (15 turns) | `runs/openwebrl-4b-reference-318933-20260924T082345/iter_0000099` (iteration 90: `runs/openwebrl-4b-reference-294421-20260913T211532/iter_0000089`) | `reference-stage1-browsers32-20260911` to 90; `reference-stage1-to100-20260922-24h-v3` to 100 | `qcq7i4ug` |
| Outcome-only, 56 groups | No ARM credit; 56 groups to match Gate B's 48 + 8 | 90 | `runs/outcome-only-56groups-20261006/iter_0000089` | `reference-outcome-only-56groups-20261006-v1` | `outcome-only-56groups-20261006` |
| **Gate B** | — | **98** | `evaluations/arm-failure-additive-335729-b100-iter100/runtime/iter_0000097` (iteration 90: `…-335729-b100-iter90/runtime/iter_0000089`) | `reference-arm-gate-b-browser-cleanup-20260927-v1` | `arm-gate-b-309053` |
| 40M + 16F | 40 mixed + up to 16 failure groups | 90 | `evaluations/arm-failure-additive-40m16f-413757-iter90/runtime/iter_0000089` | `reference-arm-gate-b-40m16f-20261026-v1` | `arm-gate-b-40m16f-392093` |
| 48M + 16F | 48 mixed + up to 16 failure groups | 90 | `evaluations/arm-failure-additive-48m16f-440838-iter90/runtime/iter_0000089` | `reference-arm-gate-b-48m16f-20261031-v1` | `arm-gate-b-48m16f-405305` |
| Strict gate (additive) | Candidate gate D = 5 | 90 | `evaluations/arm-failure-additive-311962/runtime/iter_0000089` | `reference-arm-failure-additive-20260914-v3` | `arm-failure-additive-295786` |
| No failure groups (mixed bonus) | No auxiliary failure groups | 90 | `evaluations/arm-mixed-bonus-340425-iter90/runtime/iter_0000089` | `reference-arm-mixed-async-label-20261002-v3` | `arm-mixed-bonus-334493` |
| No failure groups, reweight | Credit reweights the outcome advantage | 90 | `evaluations/arm-mixed-reweight-340423-iter90/runtime/iter_0000089` | `reference-arm-mixed-async-label-20261002-v3` | `arm-mixed-reweight-334494` |
| Failure groups replace mixed | All-failure groups only | 90 | `evaluations/arm-turn-bonus-fresh-allfailure-309490/runtime/iter_0000089` | `reference-arm-failure-bonus-20260919-countguard-v1` | `arm-allfailure-bonus-295353` |

Checkpoint directories are zero-based (`iter_0000089` = 90 completed iterations). Each holds Megatron `torch_dist` shards plus optimizer, scheduler and RNG state, and is about 58 GB. The data cursor sits beside it (`rollout/global_dataset_state_dict_<N>.pt`). Every tenth iteration and the latest one are retained; intermediate ones were pruned. Each run's source and settings are also recorded in the `launch_manifest.json` beside its `runtime/`.

**Most useful next step.** Run stage 2 (iterations 90 → 140, 30 browser turns) for Gate B and outcome-only 56 groups, then compare them at 140 with the same-day protocol in [§6](#rl-handoff-eval). Use Gate B's iteration-90 checkpoint, not 98, so both arms start stage 2 from the same point. This was planned but not started because of judge cost (about $100 per 12 h of training per run).

<a id="rl-handoff-where"></a>
## 2. What is in git and what is cluster-only

| Item | Where | Needed for |
| --- | --- | --- |
| Training data `openwebrl/data/webgym_filtered_popular_2102_cleaned.parquet`; eval sets `openwebrl/data/eval/*.jsonl` | Git (`arm`) | All runs |
| Starting model OpenWebRL-4B-SFT | HF `OpenWebRL/OpenWebRL-4B-SFT` @ `15e777db2ddba2e0e82080ebccd3ad8d215b7f0a` | All runs |
| SelectionARM (per-turn credit teacher) | HF `PTeterwak/OpenWebRL-4B-SelectionARM` @ `81b452d800d9f859687074f82680dd5257e02d89` | ARM runs |
| SelectionARM prompt builder | GitHub `piotr-teterwak/action-reward-models` @ `02276b0ff3b9048d34e6a2afdcb9042dd9018c8c` (`inference/selection_prompt.py`); local copy `RT/arm-reproduction/source` | ARM runs |
| **Exact code each run used** | Private HF repo [`zixianma/openwebrl-rl-checkpoints`](https://huggingface.co/zixianma/openwebrl-rl-checkpoints), `<run>/source.tar.gz`. On the cluster these are the frozen trees `RT/reference-*`, about 17 MB each. They are not the same as `arm`: relative to `origin/arm`, the Gate B 48M+16F tree has 36 changed files (`train.py`, `slime/`, `openwebrl/arm_*.py`, `scripts/run_h200_browser.sh`, `scripts/serve_arm.py`, …) and 18 files not in git. | Reproducing or continuing any run |
| Per-run settings (`launch_manifest.json`, `arm-config.json`, `browser-training-config.json`, controller `*-plan.json`) | HF `zixianma/openwebrl-rl-checkpoints`, `<run>/settings/`; on the cluster, beside each run (`RT/evaluations/<run>/`, `RT/arm-turn-bonus-preparation/<run>/`) | Recipes ([§4](#rl-handoff-recipes)) |
| Iteration-90 checkpoints with optimizer state and data cursor | HF `zixianma/openwebrl-rl-checkpoints`, `<run>/checkpoint/` (all nine runs). Later checkpoints (Gate B iteration 98, outcome-only 48 iteration 100) are cluster-only ([§1](#rl-handoff-20261110) table) | Continuing a lineage |
| Controllers (`scripts/run_arm_gate_b_*.py`, `scripts/resume_baseline.py`, `RT/arm-turn-bonus-preparation/*/training-controller.py`) | Partly in git; frozen copies cluster-only | Our Slurm workflow ([§5](#rl-handoff-launch)) |

**Off-cluster, download one run folder** from `zixianma/openwebrl-rl-checkpoints` (ask zixianma for access; about 58 GB per run):

```bash
huggingface-cli download zixianma/openwebrl-rl-checkpoints --include 'gate_b/*' --local-dir rl-ckpts
tar -xzf rl-ckpts/gate_b/source.tar.gz           # exact frozen code tree
export SLIME_LOAD_CHECKPOINT=$PWD/rl-ckpts/gate_b/checkpoint   # iter_0000089 + rollout/ cursor
```

Train and resume with the extracted tree, not with `arm`. The latest code on `arm` is useful for new work and tooling; every file from the cluster checkout that was not yet on `arm` was pushed on November 10. It is not the code any run used.

## 3. Environment

- **Hardware.** Every ARM run and outcome-only 56 used 1 node with 8 H200s, 64 CPUs and 960 GiB RAM, at TP2 / DP4, microbatch 1, global batch 256 and 64 local browsers. Outcome-only 48 used 4 H200s at TP4 with 32 browsers. Checkpoint restoration has only been validated on the same layout; treat any other layout as untested until a resume reproduces the saved loss and scheduler state.
- **Python.** Follow the [README installation](../../README.md) (`bash set_up.sh`, `pip install -e .`). Our training venv is `RT/venv` with sglang 0.5.6.post2, torch 2.9.1+cu129, transformers 4.57.1, megatron-core 0.16.0rc0 and playwright 1.58.0. The SelectionARM server (`scripts/serve_arm.py`) runs from a second venv, `RT/arm-reproduction/venv`, with the same torch/transformers/sglang versions.
- **Browsers.** `local_process` mode (Playwright Chromium on the training node). About 23% of Online-Mind2Web episodes are invalid from this cluster (blocked or unreachable sites). Success rates measured from another network are not comparable with ours, so rerun an outcome-only baseline there.
- **Keys** (your own `.env`, never committed): `OPENAI_API_KEY` for the GPT-4.1 training-reward judge, plus the gpt-5.6-terra evaluation judge if you use it; `WANDB_API_KEY`; `HF_TOKEN` if the HF repos are private to you.

<a id="rl-handoff-recipes"></a>
## 4. Recipes

**Shared (all runs).** MM-GRPO from OpenWebRL-4B-SFT; Adam, LR 1e-6 constant, weight decay 0.1, betas (0.9, 0.98); PPO 2 epochs, clip 0.2 / 0.28; KL and entropy 0. Each group has 5 trajectories; dynamic sampling keeps groups with nonzero reward variance. Actor sampling is T 0.8 / top-p 1.0 / top-k −1, with responses up to 1,024 tokens and context up to 32,768. Each turn sees 1 screenshot; the judge sees 3. Stage 1 allows 15 browser turns per episode. The outcome reward is GPT-4.1 with the action-history prompt. A checkpoint is saved every iteration.

**Gate B (`arm-config.json`).** `beta` 0.5, `scored_fraction` 0.2, `k` 5, `candidate_gate` `min2`, `credit_assignment` `response_index` and `seed` 42. Ordinary (mixed-outcome) groups: 48. All-failure auxiliary groups are added on top (`admit_all_failure_groups`, `additive_failure_groups`) with `failure_group_cap` 8 and loss weight N/48 for N retained groups. The 40M+16F and 48M+16F variants change only the ordinary-group count and raise the cap to 16. Launch flags on top of the shared recipe: `--custom-generate-function-path openwebrl.arm_turn_bonus.generate`, `--dynamic-sampling-filter-path openwebrl.arm_failure_additive.filter_groups` and `--sglang-mem-fraction-static .35`. Pass `--use-checkpoint-opt-param-scheduler` when resuming.

**Outcome-only 56 groups.** `ROLLOUT_BATCH_SIZE=56` with no ARM flags; see `RT/runs/outcome-only-56groups-20261006/launch_manifest.json`.

**Stage 2 (90 → 140).** `--max-steps 30` (browser turns 15 → 30) and `NUM_ROLLOUT=140`. Pass `--override-opt-param-scheduler`: the scheduler horizon is derived from the rollout target, and the loader otherwise asserts it matches the saved one. Adam moments, scheduler counter and data cursor all continue; there is no warmup. Full audit: [RL_RUNTIME.md](RL_RUNTIME.md#baseline-stage2-20260914). Outcome-only 48 (`qcq7i4ug`) carries a historical one-update scheduler offset (`OPENWEBRL_EXPECTED_SCHEDULER_OFFSET=1`); every other lineage has 0.

<a id="rl-handoff-launch"></a>
## 5. Launching

**Outcome-only runs** launch directly. Run the frozen tree's `scripts/run_h200_browser.sh` with the `command` and `environment` from the run's `launch_manifest.json` or latest controller plan. Rewrite the paths, set `SLIME_LOAD_CHECKPOINT` to the checkpoint's parent directory, and set `WANDB_RUN_ID` plus `WANDB_RESUME=must` to continue the same W&B run. Use a new run ID for a fork.

**ARM runs** need an orchestrator around `train.py`, because each iteration alternates between collection and training on the same GPUs:

1. The actor collects rollouts. For the sampled 20% of turns, the generate hook sends the turn's k = 5 candidate actions to the SelectionARM endpoint (`selector_endpoint` in `arm-config.json`).
2. Before collection, the cycle hook (`openwebrl/arm_turn_bonus_cycles.py`) writes `teacher-request.json` in the run directory. The orchestrator starts `scripts/serve_arm.py --mode selection --model <SelectionARM> --port <p>` on the last GPU, checks `/health`, runs a selector preflight and writes `teacher-ready.json`.
3. After collection, the orchestrator stops the selector so training gets the GPU back. Per-iteration artifacts (labels, `calibration.json`, `failure_auxiliary.json`, `checkpoint-validation.json`) land in `iterations/<NNNN>/`.

Our implementation is the frozen controller `RT/arm-turn-bonus-preparation/gate-b-48m16f-20261031/to90-20261104/training-controller.py` (about 300 lines), driven by `scripts/run_arm_gate_b_48m16f_to90.py`. These controllers are tied to this cluster: they use Slurm job IDs, `srun --jobid`, approval and budget ledgers under `RT/`, and absolute paths. **There is no portable launcher yet.** Off-cluster, adapt `training-controller.py`: keep its selector start, health, preflight and stop sequence, and replace the Slurm and ledger code.

**On Tillicum** you can read our trees, checkpoints and settings directly. Copy a source tree and write to your own run directory; never write into ours. One controller writes to a lineage at a time. A resume must start from a checkpoint whose `iterations/<NNNN>/checkpoint-validation.json` exists.

<a id="rl-handoff-eval"></a>
## 6. Evaluation

- **Training monitor** (every 10 iterations, in-allocation): full-300 Online-Mind2Web, local browser, GPT-4.1 judge with the action-history prompt, T 0, 30 turns, 4,096 response tokens. It is cheap and consistent within a run. A single evaluation of one checkpoint varies by up to about 4 pp between days, so one evaluation per checkpoint cannot separate the runs.
- **Claims:** evaluate every compared checkpoint at the same time, with ≥ 3 full-300 repeats each and one judge for all of them. Use the paper actor protocol (T 0.6 / top-p 0.95 / top-k 20) and offline gpt-5.6-terra judging (`RT/reference-om2w-eval-deferjudge-20261107`, `scripts/run_iter90_terra_reeval.py`, `scripts/rejudge_rl_evals_shared_terra.py`). Report overall and valid-only rates, with counts in separate columns, and paired task-level differences.

## 7. Operational notes

- **Time and cost per iteration** (8 H200): about 35–40 min, of which PPO is about 50% and browser collection about 37% ([profile](RL_RUNTIME.md#training-pipeline-profile-20261006)). A full-300 evaluation takes about 15–45 min. The GPT-4.1 training judge costs roughly $100 per 12 h of training.
- **Storage.** About 58 GB per checkpoint. Saved training batches (`rollout_recovery/*.pt`) take about 50 GB per iteration. Temporary image-transfer files (`OPENWEBRL_MULTIMODAL_STORAGE_DIR/rollout-*.bin`) are never cleaned up and reach about 1.4 TB per 12 h job; delete them after the job ends.
- **Known failure.** SGLang `Insufficient multimodal embedding length` (`mm_utils._adjust_embedding_length`) once killed all 8 rollout engines in turn, followed by `no_available_workers` 503s. Resume from the last durable checkpoint. Watch logs for engine death, not only the status file.
- **Budget stop.** Controllers stop cleanly when the next iteration plus a 1 h evaluation reserve no longer fits the allocation. A 4 h job yields about 3 iterations; a 12 h job about 17.
