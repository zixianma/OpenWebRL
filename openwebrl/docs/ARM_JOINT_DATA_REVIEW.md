# Combined ARM data: prepared for review

Prepared 2026-09-10. The user reviewed the retained examples and approved
starting combined-data SFT and parallel DPO-only. **Compute approved: two
H200s for three hours (SFT) and two H200s for five hours (DPO), including
full-300 OM2W endpoint evaluation.** See the [launch plan](ARM_JOINT_DATA_TRAINING_PLAN.md).

Open the [interactive HTML review](arm_results/joint_data_v2/review.html).
It is one self-contained file (57.5 MB), including original screenshots, and
works offline. Filter by source, decision, pair type or exclusion reason;
search tasks/actions; compare chosen and rejected tool calls; expand full
actor responses, pre-action context and GPT-5.5 rationale; zoom screenshots
and toggle normalized click markers. Review decisions and notes are local to
the browser. Export the notes as JSON to apply them in a subsequent revision;
the page does not modify data or authorize a training job.

## Prepared dataset

| Source | Train pairs / SFT winners | Validation pairs / SFT winners | Retained total | Separate SFT-only reserve |
| --- | ---: | ---: | ---: | ---: |
| C2 / SelectionARM | 3,464 | 442 | 3,906 | 4,484 |
| Piotr / GPT-5.5 | 2,076 | 216 | 2,292 | 690 |
| **Combined** | **5,540** | **658** | **6,198** | **5,174** |

Each retained state contributes exactly one pair. SFT uses precisely those
chosen responses, in the same order, with no reserve examples added. The two
preference objectives will use identical pairs. Source proportions in training
are 62.5% C2 / 37.5% Piotr, with no source oversampling or action balancing.

There are 1,169 retained training task groups and 132 retained validation task
groups, with zero overlap. Fixed diagnostic panels contain 256 C2 pairs
(85 task groups) and all 216 Piotr validation pairs (37 task groups).
The HTML contains 32 retained examples per source plus 65 exclusion/quarantine
examples: **129 examples, 122 unique screenshots**. Its stratified review mix
is intentionally different from the natural training distribution.

## Context and supervision format

A data point is one browser state and its selected next assistant response,
not an entire trajectory or an individual tool call. The matched SFT dataset
has **5,540 training examples and 658 held-out validation examples**; the
5,174-example SFT-only reserve is separate.

A streaming inspection of all 6,198 retained prompts on 2026-09-10 found:

| Property | C2 (3,906 examples) | Piotr (2,292 examples) |
| --- | ---: | ---: |
| Screenshots per example | Exactly 1 | Exactly 1 |
| Prior assistant turns, minimum / median / 90th percentile / maximum | 0 / 4 / 13 / 29 | 0 / 4 / 13 / 29 |
| Examples with no prior assistant turn | 640 | 326 |
| Chosen responses containing reasoning (`</think>` boundary) | 3,906 | 2,292 |

The number of prior assistant turns equals the saved turn index in every row.
The prompt retains previous assistant reasoning, tool calls and textual browser
observations; historical screenshots are removed, leaving the current screenshot.
Every prior assistant turn also retains its reasoning boundary. There is no
fixed last-four-turn window: four is the observed median. One assistant turn can
contain multiple tool calls.

For the proposed SFT, the prompt (including historical assistant turns and image
tokens) is masked from the loss. The target is the **complete chosen current
response: reasoning, tool calls, and end-of-turn boundary**. This follows the
existing C2 trainer's response-only supervision. The action-only masks saved in
the paired data are for the subsequent DPO ablation, not this SFT objective.
The actor's reasoning is the target; the selection judge's rationale is review
metadata, not a response to imitate.

This matches the repository's documented released single-screenshot SFT recipe:
[`PER_TURN=1` with `mask_history: true`](../../sft/README.md), implemented by
[`render_record` and `iter_prefixes`](../../sft/prepare_openai_for_llamafactory.py).
That alignment concerns context and supervision format, not identical training
data, optimizer settings, or full-finetuning versus LoRA. The compute-node
processor smoke checks listed below remain outstanding.

Paper cross-check (2026-09-10): [Section 4.1](https://arxiv.org/html/2606.02031v2#S4.SS1)
confirms full textual response/feedback history and current-screenshot context;
[Section 4.2](https://arxiv.org/html/2606.02031v2#S4.SS2) confirms target-turn-only
SFT supervision. The observed 29 prior turns reflect our 30-turn collection
budget, not a universal history-window cap. The paper's default evaluation
budget is also 30 turns ([Table 8](https://arxiv.org/html/2606.02031v2#A1.T8)).

## Findings and filtering decisions

- **120 exact normalized task texts are shared across reconstructed C2 and
  Piotr states.** They were grouped before the split. There were no exact
  prompt-plus-image duplicate states across sources. The added data provide
  additional teacher comparisons and states, not exclusively new tasks.
- The Piotr candidate file has 49,536 records and 49,360 unique draw IDs.
  **176 IDs have conflicting records across 11 base states.** An inspected
  duplicate differs in its actual candidate responses. These identities cannot
  be safely joined to a single stored selection label. All affected base states
  are withheld; ten had reconstructed winner records (seven pairs and three
  reserve examples) and the remaining state had no such usable record.
- Five additional reconstructed Piotr states have identical context/image
  identities but conflicting chosen actions. These are also quarantined.
- Raw candidate draws are 75% temperature 0.7 / 25% temperature 1.0. Repeated
  draws are not independent states. Draw selection is fixed by a seed/hash;
  a distinct eligible loser is selected from the same draw without model scores.
- No exact or screened same-host near matches to OM2W were flagged by this
  audit. This is a heuristic screen, not proof of semantic or pretraining
  disjointness. Eleven near-task links within the combined data were grouped
  conservatively before splitting.

The earlier 5,490 C2 pool is reduced to 3,906 pairs by the stricter ambiguity
rules. **All coordinate-only differences beyond the five-unit near-click
threshold are quarantined pending visual review**, because the saved data do
not contain element identities proving different targets. Nearby differences,
identical actions, duplicate loser actions, and done-versus-done preferences
are excluded. Differences only in scroll amount or wait duration are withheld
for review. This is conservative and loses some potentially useful pairs;
the gallery exposes those cases rather than treating them as verified mistakes.

Removing a pair does not necessarily remove its state: another eligible loser
can remain. A valid winner with no eligible loser goes into the separate reserve.
Done-versus-continue preferences are retained. Terminal-only winners make up
502/5,540 = 9.1% of training rows (321 C2, 181 Piotr); scroll-only winners are
1,059/5,540 = 19.1%. These distributions should be watched during training and
browser evaluation.

There is no imposed 60% modal-winner threshold, no ARM-agreement filter on the
ARM's own training labels, and no invented PRM quality floor. Unselected actions
have no observed terminal outcomes. The 64 retained review examples have not
been certified as correct; they are ready for the user's semantic review.

## Verification and remaining gates

All **6,198 retained pairs** passed independent verification:

- SFT rows equal the chosen branch of the paired rows exactly.
- State IDs are unique and task groups do not cross train/validation.
- Saved response tokens decode exactly to the retained text, including valid
  noncanonical generation tokenizations; they are not silently re-tokenized.
- Each decoded action mask is exactly its tool-call spans plus the end-of-turn
  token, with no reasoning tokens scored directly.
- Context plus response fits 32,768 tokens.
- Image bytes were hashed, including all 2,630 Piotr images matched against
  checksum-verified downloads. Image headers supplied dimensions; no bulk pixel
  tensor preprocessing was performed on the login node.

The repository's actual regex parser was checked against the strict schema
parser without importing model dependencies. The actor tokenizer and actual
image resize function reproduce saved C2 expanded-prefix hashes and grids.
Full processor pixel-tensor and native SGLang parser smoke checks remain for
a compute node before training. Twelve focused unit tests passed. Browser
checks passed for filtering, screenshot loading/zoom, note export, empty search,
mobile layout, and absence of JavaScript errors.

Piotr's released candidate strings omit finish metadata. The preparation
requires complete tool-call structure and fewer than 1,024 raw tokens, then
explicitly appends the training completion boundary. This is recorded per
candidate; it is not presented as an observed normal stop. A conservative
structural completion check cannot rule out every hidden truncation.

Median prompt lengths are 5,673 tokens for C2 and 5,310 for Piotr. Median chosen
response lengths are 320 and 362 tokens; median scored action/boundary lengths
are 30 tokens in both sources. The planned full-versus-masked comparison must
account for that large difference in scored-token count.

## Files and reproduction

Data directory:
`/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/joint-data-v2-20260910/`

- [Training pairs](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/joint-data-v2-20260910/joint_pairs.train.jsonl)
- [Training SFT view](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/joint-data-v2-20260910/joint_sft.train.jsonl)
- [Validation pairs](/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/joint-data-v2-20260910/joint_pairs.validation.jsonl)
- [Full audit and artifact hashes](arm_results/joint_data_v2/audit.json)
- [Independent data verification](arm_results/joint_data_v2/verification.json)
- [Browser verification](arm_results/joint_data_v2/browser-verification.json)

The same directory holds the SFT validation view, source-specific validation
panels, reserve, quarantine states, exclusion ledger, split manifest, image
manifest and gallery sample records. The exclusion ledger counts processed
draws/candidates, not independent states; higher-hash draws can be skipped once
a usable lower-hash pair exists. Resume events are deduplicated. Do not sum
candidate exclusion counts to infer state retention.

Scripts are [fetch_arm_joint_inputs.py](../../scripts/fetch_arm_joint_inputs.py),
[audit_arm_draw_identity.py](../../scripts/audit_arm_draw_identity.py),
[prepare_arm_joint_data.py](../../scripts/prepare_arm_joint_data.py),
[verify_arm_joint_data.py](../../scripts/verify_arm_joint_data.py), and
[render_arm_joint_review.py](../../scripts/render_arm_joint_review.py).
Run the candidate identity audit before preparation. Use the pinned input
directory `arm-reproduction/joint-data-inputs-0d83b48` and its download manifest;
the dataset revision is `0d83b48c1659cac47a1044ef88fb573d3c16e180`.

Preparation used low-priority, bounded CPU processes, no model weights, and no
GPU allocation. The download used four network workers (1.63 GB; 40.8 CPU-seconds,
78.7 MiB peak). The first builder hit its 240 CPU-second safety cap; its output
is [marked incomplete](arm_results/JOINT_DATA_V1_INTERRUPTED.md). The optimized
build then exposed duplicate IDs and resumed from saved state. Its final resumed
phase took 39.2 CPU-seconds / 136.3 wall-seconds, peaking at 390 MiB. These phase
figures are **not total preparation cost**. Independent final verification took
3.2 CPU-seconds and 179 MiB. Image decoding for the small browser smoke test was
separate from data preprocessing. No training job was submitted.
