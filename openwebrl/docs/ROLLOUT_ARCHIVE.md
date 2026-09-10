# Completed rollout archive for future SFT

The user requested preservation of discarded all-success and all-failure
groups on 2026-09-10. Earlier recovery `.pt` files contain accepted training
turns only. Sparse debug traces (configured probability 0.005) do not provide
complete rejected-group coverage. Prior discarded groups generally cannot be
reconstructed from aggregate metrics.

The archive saves **all completed groups**, before their in-memory telemetry
data is released, while preserving RL filtering and rewards. It excludes
in-flight attempts canceled at the collection cutoff and unfinished collections
interrupted before the archive hook runs. It records actual acceptance by sample
identity; `not_accepted` can also include excess completed groups at cutoff and
is not an exact dynamic-filter reason.

## Location and contents

For each run, inspect `completed_rollout_archive/iteration_NNNN_ID/manifest.json`.
Only directories with a complete manifest represent complete archives. Each
`group_NNNN.json.gz` contains group/trajectory IDs, terminal reward and validity,
acceptance status, turn-level prompts/responses, tokens, loss masks, stored log
probabilities, conversation metadata, and raw multimodal inputs. Group labels
are `all_success`, `all_failure` (all valid rewards zero), `all_nonpositive`,
`mixed`, and `contains_invalid`. These labels do not replace the actual rewards.

Screenshots are stored once per SHA-256 in the archive's `images/` directory.
JSON image references contain a relative path, MIME type, byte count and hash;
resolve them relative to `image_reference_root` in the manifest. Original data
URL bytes are retained, and PIL images are encoded losslessly as PNG. Files use
`.bin` regardless of MIME type. The export avoids processed image tensors;
recreate those with the intended SFT processor. Preserve the group JSON, shared
image directory and source run manifest together when moving an archive.

Use valid successful trajectories as candidates for positive SFT. Keep failures
for analysis, correction, or explicitly designed negative training; copying
failed actions into a standard positive SFT target would teach those actions.
Review screenshot availability and success-judge quality when building a dataset.

## Enablement and validation

`rollout_archive.enabled.json` in the run directory enables the archive, as does
`OPENWEBRL_ARCHIVE_COMPLETED_GROUPS=1`. The baseline resume wrapper creates the
control file automatically when its preserved source supports archiving. The
telemetry hook reads the completed one-based collection number from `progress.log`.
Archive errors appear in training logs, `rollout_archive_errors.jsonl`, and
`rollout/archive/errors`; they do not change RL filtering or stop training.
W&B also receives group/trajectory/image counts and archive duration.

Seven CPU tests covered existing metric denominators, unchanged training
metrics/sample data, valid/invalid outcome classification, image deduplication
and byte recovery, opt-in behavior, and visible error reporting. A CPU-only
export inside allocation 286094 verified a real six-turn trajectory from saved
collection 23: one trajectory, six unique images, about 95 KB compressed group
JSON, and about 0.07 seconds exporting. That checks the schema on real data;
live full-collection archive completion is a separate verification.

The hook was installed before collection 24 completed in job 286094. The
telemetry module is imported at collection completion; no worker restart was
needed. `rollout_archive_installation.json` and `archive_source_before/` in the
run preserve installation provenance. All five baseline recipe hashes remain
unchanged.

Collection 24's live archive completed at 01:00 PDT: 88 groups, 440 trajectories,
3,203 turns, and 2,290 unique images. It includes 11 rejected all-success groups,
12 rejected all-failure groups, 17 rejected groups containing invalid attempts,
and all 48 accepted groups. Archiving took 31.2 seconds. Every group JSON and
image reference passed a subsequent audit, including byte sizes and SHA-256 for
all images. Storage was 831,557,303 image bytes plus 59,114,630 compressed JSON
bytes. Evidence: `iteration_24_archive_audit.json` in
`openwebrl-runtime/runs/openwebrl-4b-reference-286094-20260910T071441`.

The archive independently reproduced W&B history row 420:
`train/reward=0.40914285714285714` across 1,750 accepted turns, and
`train/task_success_rate=173/440=0.3931818181818182`. See
`iteration_24_reward_wandb_audit.json`. Earlier rejected collections remain
unavailable except any sparse debug traces that happened to be saved.
