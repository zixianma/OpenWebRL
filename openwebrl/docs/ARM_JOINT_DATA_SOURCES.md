# ARM joint data sources: original five-candidate records

Where the five original actor candidates behind every joint C2 + Piotr SFT/DPO
example are stored, and how to join a joint row back to them. The recipe and
filtering rules are in [ARM_JOINT_DATA.md](ARM_JOINT_DATA.md#arm-joint-data-training-plan);
results are in [ARM_SUMMARY.md](ARM_SUMMARY.md#2-offline-filtered-sft-and-preference-learning).
All paths are local cluster storage and are not published.

**Verified 2026-10-24:** all 6,198 joint rows (5,540 train / 658 validation, in both
the pair and SFT views) resolve to a source record with exactly five candidates.
The source selection equals the row's `chosen_index`, with zero mismatches.

## Joint data (derived)

`/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/joint-data-v2-20260910/`

| File | Contents |
| --- | --- |
| `joint_pairs.{train,validation}.jsonl` | One chosen/rejected pair per state; DPO input |
| `joint_sft.{train,validation}.jsonl` | The same states, chosen response only; SFT input |

| Source | Train states | Validation states |
| --- | ---: | ---: |
| C2 | 3,464 | 442 |
| Piotr | 2,076 | 216 |

Joint rows keep only the chosen and rejected candidates (`chosen_index`,
`rejected_index`, both zero-based). The other three candidates are only in the sources below.

## C2: our ARM-guided collection (run `c2-full-282782`)

Root: `/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-full-282782-20260908T075414Z/`

| Item | Value |
| --- | --- |
| Turn list | `training.jsonl`: 8,394 executed ARM-selected turns from 1,151 valid successful episodes (of 2,091 tasks); sha256 `cb7c75df…`, matches the joint builder's record |
| Candidate record | `attempts/<hash>/attempt-NNNN/turn-NNNN.json` |
| Five candidates | `raw_candidates[i]` = `response`, `response_token_ids` (SGLang IDs), `finish_type`; also `candidate_seeds`, `scores`, `selector_raw`, `prompt`, `images` |
| Selection | `selected_index` (zero-based), chosen by frozen SelectionARM; that candidate was executed |
| Candidate generator | `OpenWebRL-4B-SFT` (`/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT`), five independent samples per turn, temperature 0.7, top-p 0.9, 1,024 new tokens |
| Run config | `frozen-config.json`; actor model confirmed in `actor-replica-*.log` |
| Joint-row link | `source_path` is the turn file |

**State distribution caveat:** each candidate comes from the SFT checkpoint, but each
state follows earlier ARM-selected actions. Only turns from successful trajectories are
kept. The states are therefore ARM-guided, success-filtered SFT rollouts, not plain
SFT rollouts.

## Piotr: `PTeterwak/action-reward-models-data`, `openwebrl_actor/`

Root: `/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/joint-data-inputs-0d83b48/openwebrl_actor/`
(pinned HF revision `0d83b48c1659cac47a1044ef88fb573d3c16e180`)

| File | Contents |
| --- | --- |
| `candidates_merged.jsonl` | 49,536 draws; each line has draw-level `state_id` (e.g. `ep28__t0#d0`), `base_state_id`, five `candidates`, `temperature` 0.7 |
| `labels_drawlevel.jsonl` | 39,997 GPT-5.5 teacher labels: `custom_id`, draw-level `state_id`, `selection` (zero-based), `reasoning` |
| `states_full.jsonl` | 3,085 states / 412 SFT-demonstration episodes: prompt, task, demonstrated action |
| `state_images/` | Local screenshots; the `images` paths in the candidate file point at Piotr's cluster (`/projectnb/...`) |

Joint-row link: `source_path` points to `states_full.jsonl`, **not** the candidate file.
Seek to byte `candidate_file_offset` in `candidates_merged.jsonl` and read one
line; its `state_id` equals `draw_id` without the `Piotr:` prefix. Join the label by
`custom_id` in `labels_drawlevel.jsonl`. Do not join through `labels.jsonl`; it
drops the `#dN` draw suffix.

- **Repeated draws:** most states have several draws, each with a different set of
  five candidates. The builder kept one draw per state by fixed hash, so `draw_id`
  names the exact five candidates behind a joint row.
- **Excluded draws:** 176 draw IDs across 11 states have conflicting duplicate records and
  are excluded. See `../candidate-identity-audit.json`.
- **Generator:** the release labels these as OpenWebRL actor samples on SFT-demonstration
  states. We have not verified the exact actor checkpoint locally.

