# WebPRM baseline experiments: code map

This directory, together with the `webprm_*` modules in `openwebrl/`, `scripts/` and `tests/`, is the code behind the
WebArbiter Table 5 reproduction and the Piotr SelectionARM baseline comparisons (Online-Mind2Web and WebArena-Lite).
Results, protocol decisions and the handover narrative are in
[`openwebrl/docs/WEBPRM_HANDOFF.md#webprm-baselines-handover-20261028`](../openwebrl/docs/WEBPRM_HANDOFF.md#webprm-baselines-handover-20261028)
and [`openwebrl/docs/baseline_comparison.md`](../openwebrl/docs/baseline_comparison.md).

**Paths are hard-coded.** Almost every builder, submitter and analysis script assumes the Tillicum runtime root
`/gpfs/scrubbed/zixianma/openwebrl-runtime/` (abbreviated below as `RT`) and
`D = RT/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021`. Change the constants at the
top of each script when moving to another machine. Every Slurm script uses `--account=vlm`.

## Layout

| Path | What it is |
| --- | --- |
| `openwebrl/webprm_*.py` | Harness library: released WebArbiter / WebShepherd coordinator and explorer wrappers (`webprm_arbiter_harness.py`, `webprm_shepherd_harness.py`), OM2W Table 5 collector (`webprm_om2w_table5.py`), native WebArena worker/episode/evaluator bridge (`webprm_webarena_*.py`), site hosting and reset (`webprm_{cms,shopping,gitlab,reddit}_*`, `webprm_map_*`), broker and budget journal (`webprm_broker.py`, `webprm_concurrent_broker.py`, `webprm_budget_journal.py`), Piotr SelectionARM adapter (`webprm_webarena_piotr.py`, v11 route `webprm_native_piotr_v11.py`), OpenWebRL-action-space route (`webprm_owrl_v12.py`). |
| `scripts/*webprm*` | Campaign entry points: `prepare_/launch_/submit_/collect_webprm_table5_campaign.py` (WebArena), `prepare_/launch_/collect_webprm_om2w_table5.py` (OM2W), `run_native_webarena_cell.py` (v9/v11 cell worker), `run_owrl_webarena_cell.py` (v12 cell worker), site acquisition / hosting validation (`acquire_*`, `validate_webprm_*_hosting.py`), audits (`audit_webprm_*`). |
| `tests/test_*webprm*` | Unit tests for the above (run inside a package with `PYTHONPATH=<package>/source`). |
| `webprm_baselines/webarena/tools/v9/` | v9 four-site Table 5 rerun (no search / WebArbiter-7B / WebShepherd-8B, 1024×2490): builders, launcher/broker/cell patches, replacement-block pooling, `summarize_v9.py`, `cost_latency.py`, GitLab bring-up (`gitlab/`), decision record. |
| `webprm_baselines/webarena/tools/v11/`, `v11b/` | v11 gpt-4o ± Piotr SelectionARM at 1280×720: builders, launcher patch with selector-shaped ARM preflight (v11b), submit (`submit_v11b.sh`, `submit_v11c.sh`), replacement-job preparation (`replace_v11c*.sh`, `replacement_blocks_v11r2.py`), ledger close, `cost_latency_v11.py`. |
| `webprm_baselines/webarena/tools/v12/` | Option C: OpenWebRL-4B-SFT ± Piotr SelectionARM in OpenWebRL's screenshot + coordinate action space, 1280×1000: cell worker, launcher patch (actor + ARM sglang servers per job), UNIX-socket relay into the site network namespace, builder, submit, tests, decision record. |
| `webprm_baselines/webarena/v12-owrl-snapshot-manifest/` | Git HEAD and SHA-256 list of the frozen OpenWebRL snapshot v12 runs against (the snapshot itself, `D/v12-tools/owrl-snapshot/`, is not committed: it is the `arm-selection-head` working tree including uncommitted crossbench files). |
| `webprm_baselines/webarena/audit/` | Independent per-cell auditors (`native-audit-v8` for v9/v11, `native-audit-v12` adds OpenWebRL trace checks). A verified audit is what allows a cell's site copy to be disposed. |
| `webprm_baselines/webarena/analysis/` | `paired_v11.py` (paired comparison, sign test; reproduces the published v11 table), `diag_v11_overrides.py` / `diag_v11_cms_reports.py` (ARM override and CMS timeout diagnosis), gpt-4.1 fuzzy re-score (`fuzzy-rescore-v9/`, `fuzzy-rescore-v11/`). |
| `webprm_baselines/om2w/` | OM2W rerun300 builders (`build_rerun.py`, `build_retry2.py`), an example campaign `sbatch`, final pooled table (`final_table.py`), judge-agreement analysis. |

## Workflow (WebArena)

1. **Build packages** (CPU-only `sbatch`): `tools/<ver>/build_*.py` copies the harness into
   `D/launch-preparation-<ver>-<name>/source`, applies the `patch_*` scripts, writes `proposal.json` /
   `resource-request.json`, and runs the package unit tests (`run_tests_*.sbatch`). Packages are hash-frozen: never edit a
   built package or the tools a live package references; copy to a new version instead.
2. **Approval receipt**: GPU jobs need explicit user approval of the exact resources. Write
   `D/gpu-budget-<ver>-<date>/user-approval-<ver>-<date>.json` with the quote; the ledger in the same directory binds the
   GPU-second and API caps.
3. **Submit**: `cd /tmp && Q='<quote>' tools/<ver>/submit_<ver>.sh <names>` reserves in the ledger, writes
   `approved-request.json`, then `submit_webprm_table5_campaign.py --submit` (sbatch --hold, register, release).
4. **Audit**: the campaign audits each cell (`audit/native-audit-*`); audited cells release their scratch site state.
5. **Replacements**: jobs stop at the 1 TiB per-job scratch cap (about 21 cells), on 4 consecutive invalid cells, or when
   under 30 min remain. `replace_v11c_r2.sh NEW SITE SECONDS API_USD OLD_JOB OLD_PKG [EARLIER...]` closes the old ledger
   entry, computes the remaining cells (`replacement_blocks_v11r2.py`: valid outcomes, fuzzy-judge and agent-invalid
   cells, and cells with two infrastructure-invalid attempts are final), builds and tests the replacement, and prints the
   submit command. `API_USD` must be at least the harness's `min_cell_budget_usd` (6) or no cell starts.
6. **Analysis**: pool with `summarize_v9.py` / `paired_v11.py`, re-score fuzzy-judge cells with gpt-4.1
   (`analysis/fuzzy-rescore-*`), cost/latency with `cost_latency*.py`.

## Not in git (reference by path)

API keys (`.env`), WebArena task configs with reference answers and the released evaluator source (pinned inside each
package), hosted site images and per-cell site state, raw trajectories, screenshots, ARM preflight payloads
(`webprm_piotr_preflight_v11b.json`, `webprm_owrl_v12_preflight.json`, copied from saved task calls), model weights,
build/test logs and campaign outputs. All live under `D/` and `RT/webprm-om2w-table5-20261012/`.
