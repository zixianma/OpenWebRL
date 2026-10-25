# WebPRM Table 5 handoff

<a id="webprm-status-20261025"></a>
## Status — October 25, 2026 (Pacific)

**WebArena-Lite Reddit native 18×3 is complete and far below the published Table 5; OM2W rerun300 is finishing completion and invalid-retry jobs.** This section supersedes the October 24 supervision section below. Results file: `webarena-subset-review-20261021/results-native54/table.json` (paths below are under `/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/`).

### WebArena-Lite Reddit, GPT-4o 2024-08-06 actor, 18 released single-site tasks

| Method | Cells | Valid | Successes | Valid-only success (%) | Invalid as failure (%) | Published Table 5 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No search | 18 | 15 | 1 | 6.7 | 5.6 | 28.6 |
| WebArbiter-7B | 18 | 13 | 2 | 15.4 | 11.1 | 52.6 |
| WebShepherd-8B | 18 | 15 | 1 | 6.7 | 5.6 | 47.6 |

Successes: no search task 14; WebArbiter tasks 14 and 134; WebShepherd task 132. Differences between methods are within noise at these denominators. The earlier gpt-4o-2024-05-13 cohort was also far below the paper (15.8 / 5.3 / 10.5 %), so the gap is not caused by the actor substitution. Task 5 (original 27) is excluded; the paper denominators appear to be 19–21 tasks.

<details><summary>Invalid cells, audits and packages</summary>

- Invalid (10): tasks 139/140 × 3 methods call the retired `gpt-4-1106-preview` fuzzy judge (6). An offline replay of each saved evaluator transcript with gpt-4.1-2025-04-14 substituted inside the released `llm_fuzzy_match` scores all six as failures: no post was published and `url_match` fails (`fuzzy-rescore-gpt41/`). gpt-4.1 called an empty answer "correct" in 3 of 6 calls, so it is not a reliable substitute in general; the outcome here does not depend on it. Model emitted `send_msg_to_user …` without call syntax (3). Released `ScriptBrowserEnv` crashed when an action opened a new tab without a CDP client, `envs.py:156–179` (2).
- Independent audit: 53/54 verified; no-search task 132 (valid, failed) fails only exact replay because the released coordinator runs post-action state diff/terminal detection in a background thread (`step_management.py:647`).
- Packages: no search = `v7e` (tasks 14, 97, 98, 131, 132) + `v7k-ns-b`; WebArbiter = `v7k-arb-a/b`; WebShepherd = `v7j-shep-a/b`, all under `webarena-subset-review-20261021/launch-preparation-*`. Fixes on the way: browser v2 (geometry/nav), JSON mode + fence strip, auditor v2/v3 (54-cell cohort; v3 adds `start_url` to the Shepherd context), Reddit storage scan 300 s/1800 s and 1 TiB scratch, block packages ($50 cap bound in the worker), Shepherd 2048-token health preflight, phase-write lock for concurrent tournament comparisons. Builders `v7-tools/build_v7.py` / `build_v7k.py` are hash-frozen by their packages: copy, never edit.
</details>

### OM2W rerun300 (gpt-4o-2024-08-06 actor, gpt-5.6-terra judge)

Interim (before completions/retries): no search 28.3 %, Piotr 31.3 %, WebArbiter 25.7 %, random-of-five 15.2 %, WebShepherd 12.8 % of valid cells; on 223 tasks with all five valid, random5 (p<0.001) and Shepherd (p=0.0001) are below no search, Arbiter and Piotr are not distinguishable. Remaining cells run as completion packages (`rerun300-j*-completion{2,3}-vlm`, J1/J2 under the user cap raise) and one labelled retry of every invalid cell (`rerun300-j*-retry1-vlm`, user approval `rerun300-0806-20261023/user-approval-retry-20261025.json`). Final table: `rerun300-0806-20261023/final_table.py`. The launcher's 15 s `scontrol` startup timeout has failed three single attempts; packages now allow two attempts.


**Table 5 is incomplete.** As of 2026-10-12 (late), Reddit successor **357126 is queued**, and **CMS hosting/reset validation passed** (job 357831). Supervision was an interactive Claude Code session monitor, not an unattended supervisor. This handoff records the state as of 2026-10-12. Read the current execution pointer before acting; this document is not new resource authorization.

The goal is to reconstruct **WebArbiter Table 5 and add Piotr’s released SelectionARM**, then evaluate Online-Mind2Web separately. The fixed comparison contains **155 single-site tasks × 2 actors × 5 selectors = 1,550 episodes**. Exact historical Table 5 membership/settings remain unresolved, so describe this as a **disclosed reconstruction**, not an exact reproduction.

Start with [results and methodology](WEBPRM_BASELINES.md#table5-plan-20261010), the repository [working agreement](../../AGENTS.md), and these runtime records:

```text
R = /gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011
N = R/next-stage-preparation
C = N/reddit-table5-190-v3/campaign

R/current_table5_preparation.json
N/reddit-terminal-independent-root-356127-v1/final-review.json
```

`R`, `N`, and `C` above are path abbreviations used throughout this document.

## Verified results

Only the **190-cell Reddit subset** has execution approval. **26 cells have recorded outcomes; none were resampled.** The first 12 (jobs 356022–356127) are independently audited. The 14 from job 356527 have per-cell worker audits but **no independent audit yet**.

| Task | Cells recorded | Valid | Successes (after scoring guard) | Invalid | No verdict (interrupted) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 | 10 | 6 | 0 | 2 | 2 |
| 14 | 10 | 10 | 8 | 0 | 0 |
| 97 | 6 | 5 | 1 | 1 | 0 |
| Total | 26 | 21 | 9 | 3 | 2 |

Task 5 shows zero after the explicit `N/A` scoring guard (five raw positives on task 5 incorrectly accepted `N/A`; original verdicts are preserved). Task 14 succeeded for 8 of its 10 actor × selector cells (failures: GPT-4o + Arbiter-7B, GPT-4o + Shepherd-8B). The task 97 invalid cell is **GPT-4o + Piotr SelectionARM**: a terminal-detector fallback click executed without a Piotr call and had no grounded history display, so the next Piotr prompt could not be built (`EvaluationNotReady`). This is fixed in v9 (below); the invalid cell stays invalid and is not resampled.

**164 Reddit cells remain unclaimed.** The other **1,360 cells** still need hosting-validated environments and execution budgets. Do not publish selector rankings or success percentages from this partial, mostly unaudited coverage.

## Current jobs and immediate next actions

| Job | What | Resources | Status |
| --- | --- | --- | --- |
| 357126 | Reddit successor `N/reddit-table5-190-v9` (5th and last allocation slot) | gpu-h200, 1 H200, 8 CPUs, 240 GiB, 1,344 min | Queued (priority) |
| 357831 | CMS hosting validation `N/cms-hosting-validation-v9` (7th CMS slot) | 0 GPU, 4 CPUs, 16 GiB | **Passed** in 231 s (see CMS section) |

1. Supervise 357126 from start through closure; independently audit its new cells and the 14 cells from 356527.
2. CMS hosting is validated. Next: a CMS Table 5 execution package (its cells still need a GPU budget approved by the user).
3. Prepare hosting/reset validation packages for Shopping, GitLab and Map by generalizing the CMS validator (downloads are complete and independently verified; see below).
4. The WebPRM code is **not tracked in git on any branch**; it exists in the main checkout and in frozen packages under `N/`. Committing it publishes the whole WebPRM code base to the public remote; ask the user first.

**User approvals on 2026-10-12:** "yes I approve" covered the v8 4th Reddit slot; a later "yes I approve" covered (a) per-node live scratch accounting and (b) the 5th Reddit slot, plus the 3rd CMS attempt; "yes" covered the 4th CMS attempt and, earlier, the four CPU-only bundle-v4 environment jobs. Later, "sure do it" covered the 5th CMS attempt, and "yes" covered the 6th, with a standing rule: **jobs requesting 0 GPUs need no approval**. GPU jobs still need explicit approval for each submission. Leftover time or API budget never authorizes another GPU slot.

<details>
<summary>2026-10-12 Reddit attempt 356527 and the v9 successor</summary>

- **v8** (`N/reddit-table5-190-v8`) = v7 + the reviewed admission-drain queue repair + an exactly approved 4th-slot exception. Job **356527** ran 2,766 s and recorded 14 new cells. The queue repair worked live (admission waited for active workers).
- It stopped early because the task 97 invalid cell's state (48 GB) was preserved, as policy requires for an incomplete audit. That pushed **cumulative** retained scratch across all attempts (480,954,886,176 bytes) to where one more cell would exceed the 549,755,813,888-byte cap, so admission closed with no active worker.
- **v9** (`N/reddit-table5-190-v9`, builder `N/reddit-table5-190-v9-build/build_v9.py`) adds:
  - Piotr history grounding for actions executed without a Piotr selection (`PiotrSelectionARMSelector.historical_action`, called from `run_loop`). A frozen-suite test (`test_webprm_terminal_fallback`) caught that the fallback path needs a real adapter; fixed before freezing.
  - `scratch_accounting_exception` with rule `per_node_live`: the cap counts this node's live usage plus prior attempts' retained state **still present on this node** (node-local `/tmp`); retained states on other nodes are preserved but not charged. The cap value and free-disk check are unchanged. It is exception-gated (approval digest + user text) and outside the allocation contract.
  - `allocation_exception.additional_attempts=2` (5 slots total).
- 311 frozen tests pass; `validate_proposal` passes; allocation contract identical to v8. Approval receipt: `N/reddit-table5-190-v9/approved-request.json` (previous closed epoch `bcc1c20f…`).

</details>

<details>
<summary>Failures, repairs and validation evidence</summary>

| Job | Scheduler seconds used | Outcome |
| --- | ---: | --- |
| 356022 | 2,614 | Stopped after detecting native terminal-detector wrapper bugs; 10 immutable primary claims preserved |
| 356094 | 47 | Startup failed when a storage scan encountered a disappearing database-build file; no new model calls or primary claims |
| 356127 | 305 | Completed two successful trials, then prematurely stopped admitting work |

Repairs are implemented and tested:

- Native terminal handling follows the pinned parser’s fallback/nonterminal-action behavior: **78 focused tests**.
- Storage scanning retries narrowly identified file-disappearance races, accepts only complete scans, and verifies final measurements after closure: **35 focused tests**.
- The latest queue fix waits for an active worker to close before rechecking a denied resource admission. It also records admission reasons and checks minimum startup capacity: **52 tests plus 7 independent frozen-source checks**.
- Additive interruption audits and collector support passed **58 tests**, with independent byte-identical replays. No primary evidence or verdict was rewritten, and no model call or state disposal was performed by audit activation.

**The queue fix has not been validated live.** Frozen v7 still contains the premature-exit behavior. The precise resource check that rejected admission in 356127 was not logged; storage pressure is plausible, not proven.

Reviewed repair and closure evidence:

```text
N/reddit-admission-drain-repair-v1/readiness.json
N/reddit-admission-drain-repair-v1/root-review.json
N/reddit-stop-audit-356127-v1/audit.json
N/reddit-terminal-independent-root-356127-v1/two-primary-audits.json
N/reddit-terminal-independent-root-356127-v1/closure-audit-review.json
N/reddit-interruption-audit-v3/readiness.json
N/reddit-terminal-independent-root-356127-v1/interruption-activation.json
```

Queue-repair readiness SHA256: `ac61509ed9d52c02fccb74e9f794325b2efa2366bbf8453909baa83eb2a8df91`.

Independent queue-repair review SHA256: `b93620beace6db15c63187be32922dce1f55181b63d7b339f27f25f4b00b26ca`.

Latest terminal/accounting audit SHA256: `1dff895e4e319d6ad17851d3958bbc859ba89d865aa852c5ecd2401faa816587`.

The queue/controller implementation is in `openwebrl/webprm_campaign_queue.py` and `scripts/launch_webprm_table5_campaign.py`. Reporting uses `scripts/collect_webprm_table5_campaign.py`; interruption auditing uses `scripts/audit_webprm_interrupted_cell.py`. Prefer reviewed frozen source packages over assuming every working-tree change belongs to this task.

</details>

<details>
<summary>Reddit resource authority and cumulative accounting</summary>

| Resource | Cap | Consumed (after 356527) |
| --- | ---: | ---: |
| Allocation attempts | 3 + 2 approved exceptions = 5 | **5** (356022, 356094, 356127, 356527, 357126 queued) |
| Cumulative allocation time | 86,400 s | 5,732 s before 357126 |
| API spending | USD 250 | USD 7.800215 |
| Paid dispatches | 21,524 | 1,037 |
| Local dispatches | 11,471 | 833 |
| Total dispatches | 32,995 | 1,870 |
| Retained apparent scratch (all attempts) | 549,755,813,888 bytes (per node live under v9) | 480,954,886,176 bytes across g017/g009/g001 |
| Durable campaign artifacts | 68,719,476,736 bytes | 304,104,480 bytes |

The actual allocation profile was **1 H200, 8 CPUs, 240 GiB RAM**, with at most two workers. The original request allowed 16 CPUs; the approved operational reduction used 8 because of scheduler constraints. 357126 is the last approved slot; another slot requires another explicit exception.

All failed attempts, startup controls and the historical unfinished local dispatch remain charged. The financial epoch is closed, with no unresolved paid reservations or overruns. Preserve the shared campaign, journal, queue claims, source versions and all original verdicts. Do not reset a budget on retry or resample a claimed primary cell.

Original authority and financial evidence:

```text
N/table5-next-jobs-bundle-v2.json
N/table5-next-jobs-bundle-v2-approval.json
C/allocation-history/
C/shared-budget.json
C/shared-budget.json.events.jsonl
C/queue/
C/audit-replacements/
```

Original approved bundle SHA256: `e53d165ad01a9a69c777c9de9f28d4e9f76d0514df203aa8708fdcdd3004ebde`.

The financial epoch recovery for 356022 has already been performed. Do not repeat it. Its preserved receipt is `N/reddit-operational-repair-v1/epoch-recovery.json`.

</details>

<details>
<summary>Other sites: downloads and CMS hosting validation (2026-10-12)</summary>

The four CPU-only bundle-v4 jobs were approved ("yes") and run. Downloads are complete and **independently re-verified by separate jobs**; none is hosting-validated:

| Site | Job | Result | Charged bytes / cap | Independent check |
| --- | --- | --- | ---: | --- |
| Shopping (third-party image) | 356538 | Completed, 1,771 s | 46,675,568,065 / 70,000,000,000 | 356576: 28/28 blobs re-hashed |
| GitLab (third-party image) | 356548 | Completed, 1,906 s | 49,673,134,791 / 80,000,000,000 | 356581: 12/12 blobs re-hashed |
| Map (original WebArena data) | 356547 | Completed, 5,146 s (append-only resume) | 190,513,780,871 / 195,000,000,000 | 356631: 35/35 SHA256 plus 4 published S3 multipart ETags |

Shopping and GitLab images' equivalence to the original WebArena state is unverified. No hosting/reset validation packages exist yet for Shopping, GitLab or Map.

**CMS hosting validation** (campaign root `N/cms-hosting-validation-v3/campaign`, shared by every version; cap 7,200 s cumulative, 4 CPUs/16 GiB/0 GPU):

| Job | Package | Seconds | Failure | Fix |
| --- | --- | ---: | --- | --- |
| 356545 | v3 | 155 | Inherited 32 MiB `RLIMIT_FSIZE` (log cap) killed `cp` on a 255 MB file | v4: logs capped by a pipe drain instead |
| 356561 | v4 | 156 | Host-user `cp` could not read mode-0 `run/crond.reboot` | v5: copy runs as namespace-mapped root |
| 357113 | v5 | 4 | `history()` still hard-coded `len(rows)>2` after the cap was raised to 3; the 3rd job failed its own startup check | v6: bound by `RESOURCES['max_allocations']`; a terminal attempt that never created its folder is charged time only |
| 357128 | v6 | 129 | `apptainer --writable` cannot create bind destination `/opt/webprm-cms` | v7: create empty `/opt/native-fakeroot` and `/opt/webprm-cms` in the private state copy only; symlinks rejected |
| 357728 | v7 | 198 | Container, MariaDB, services, origin rewrite and admin login all worked; the check then visited `/admin/dashboard`, a Magento 404 | v8: see the probes below |
| 357822 | v8 | 0 | Cancelled while pending: cumulative cross-node retained scratch (35.08 GB) would have failed storage admission | v9: charge prior retained scratch only while it exists on this node (the Reddit `per_node_live` rule); the free-disk check is unchanged |
| **357831** | **v9** | **231** | **Passed** | — |

CPU-only probes on disposable state (not validation slots) found the v8 fixes:
- **357743:** found the canonical dashboard route `/admin/admin/dashboard/`.
- **357750:** found that the product name field is rendered by JavaScript after `domcontentloaded`, so the check now waits for it.
- **357766:** applying the hash-verified image config `Env` made `elasticsearch` resolvable again (`--cleanenv` had dropped `PATH`), and the UI save then persisted. Save failures now report Magento's message verbatim.

**Result of 357831** (`C_CMS/attempt-357831/validation-result.json`, with `C_CMS = N/cms-hosting-validation-v3/campaign`):

| Check | Count / value |
| --- | --- |
| Released shopping_admin tasks / distinct start URLs loaded (HTTP 200) | 35 / 2 |
| Native evaluator on baseline: positive control / negative control | 1 / 0 |
| UI product rename persisted in the database | yes |
| Fresh second copy restored the product (`Joust Duffle Bag`) and the evaluator controls | yes |
| Owned processes remaining at closure; model calls; API spend | 0; 0; $0 |
| CMS time used / cap | 873 s / 7,200 s |

This validates the **third-party** image's hosting and reset; its equivalence to the original WebArena CMS state is **not** verified (`original_state_equivalence_verified=false`). Retained scratch on g005: attempts 357728 and 357831 (24.6 GB); earlier attempts' scratch is on other nodes.

</details>

<details>
<summary>Supervision, documentation publication and workspace constraints</summary>

The old `table5-active-supervisor-v2` remains stopped. Jobs on 2026-10-12 were supervised by Monitor scripts in an interactive Claude Code session (each expires after 30 minutes and must be re-armed). They do not persist after the session ends. Session monitor script for 357126 polls `sacct`, new queue outcomes, unexpected admission denials (ignoring cumulative-scratch denials while a worker is active) and tracebacks in `N/reddit-table5-190-v9/campaign-357126.out`.

Runtime pointer `R/current_table5_preparation.json` lists every job, cause and fix. The main checkout contains unrelated dirty work; avoid broad staging.

Operational constraints:

- **Never write under `/gpfs/home/zixianma/`**, including subprocess caches or symlink targets. Use approved project/scrubbed paths or `/tmp`, and Python `-B`. Do not override `HOME` or `CODEX_HOME` to evade this boundary.
- Use auto-review for routine permissions. Any command invoking `rm` needs explicit approval of the exact command and targets.
- Do not use SSH, add allocations or API budget without exact authorization, reset cumulative accounting, rerun claimed primary cells, or delete retained historical evidence.
- Preserve frozen science, task identities, actor/selector protocols, raw/effective verdicts, invalid attempts, source lineage and every resource charge.
- Historical Online-Mind2Web diagnostics cannot populate this Table 5 comparison or a new matched Online-Mind2Web comparison.
- Honor the standing requirement to commit/push substantial project documentation to `arm`, excluding unrelated changes and private payloads; verify publication before claiming it succeeded.

</details>
