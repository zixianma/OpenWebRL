# Teacher action-selection reasoning

**Piotr’s data contains GPT-5.5 teacher explanations, even though the released selector emits only an index.** The default training builder discards the rationale; its `--cot` option retains it.

Our review finds three things:

- **The teacher evaluates expected progress:** task constraints, current page, prior failures, useful command sequences, and whether to stop. It can reject a promising actor explanation when the actual commands do too little.
- **A winning response need not contain a better action.** Some explanations acknowledge equivalent alternatives; one prefers clearer actor reasoning among four identical scroll commands. Formatting also appears as a tie-break criterion.
- **These are comparative judgments, not measured returns.** The cleaned records contain no terminal-success labels or candidate-branch outcomes. They cannot replace the paired continuations in the [benefit-based threshold study](ARM_INFERENCE.md#confidence-benefit-20261008).

Scope: **2,292 cleaned states**, from **379 task groups / 398 episodes**. Every retained teacher label and rationale was checked against its original source. Qualitative findings come from **32 deterministically sampled states**, covering all seven stored pair types. This deliberately varied sample does **not** estimate population prevalence. [Aggregate and audit](arm_results/teacher_reasoning_20261008/aggregate.json).

| Criterion in the reviewed explanations | Generalized example | Sample cases | Reviewed states |
| --- | --- | ---: | ---: |
| Task constraints / missing information | Continue gathering evidence when a required criterion remains unmet | 10 | 32 |
| Executing useful commands | Prefer entering and submitting a query over merely focusing the field | 6 | 32 |
| History / failed paths | Avoid repeating a failed search or inaccessible route | 6 | 32 |
| Stop versus continue | Weigh answering now against further evidence gathering | 5 | 32 |
| Equivalent or grouped alternatives | Acknowledge similar choices while still naming one winner | 11 | 32 |

Categories overlap. These are counts from a manual reading of the sample, not an automated semantic classification.

<details>
<summary>Cases that limit how we interpret the labels</summary>

- **Reasoning versus execution:** one explanation rejects candidates whose thoughts promise navigation but whose commands only focus a search box. Another chooses clearer planning among four identical scroll commands. Both action content and response-level reasoning can influence the preference.
- **Weak tie-breaks:** four sampled explanations invoke URL/key formatting; examples include a trailing-slash URL and Enter-key capitalization. This audit did not execute those alternatives to establish a practical difference.
- **Approximate completion:** one explanation explicitly accepts an approximate final answer over further unsuccessful navigation. That does not independently verify the answer or establish task completion.
- **State-dependent reliability:** some explanations prefer Enter when focus is assumed correct; another prefers clicking to avoid focus uncertainty. These involve different states, so they do not establish a contradiction. Screenshot-grounding claims were not independently verified here.

The stored text is the teacher’s returned explanation. We do not assume it exposes the complete internal process or faithfully identifies every cause of the selection.

</details>

<details>
<summary>Data provenance, filtering and sampling</summary>

| Source | Rows | Nonempty teacher explanations |
| --- | ---: | ---: |
| Original draw-level labels | 39,997 | 39,937 |
| Cleaned Piotr training subset analyzed | 2,076 | 2,076 |
| Cleaned Piotr validation subset analyzed | 216 | 216 |

The 60 empty original rationales accompany null selections. The cleaned subset has one retained draw per unique state, with no overlap between training and validation at the state, task-group or episode level. The 176 conflicting candidate-draw identities remain quarantined; none enters this analysis. This is a curated subset, not a representative sample of all original teacher labels. See [joint-data preparation](ARM_JOINT_DATA.md) and the [upstream dataset](https://huggingface.co/datasets/PTeterwak/action-reward-models-data), revision `0d83b48c1659cac47a1044ef88fb573d3c16e180`.

Median rationale length is 403 characters / 64 whitespace-delimited words; character IQR is 365–442. All 2,292 cleaned `terminal_success` fields are null. Multiple states share a task or episode: counts are descriptive, not independent success trials.

The qualitative sample uses a fixed SHA-256 ranking, covers every pair type, includes validation examples and varied chosen indices, then fills to 32 states. It contains 20 training and 12 validation states from 31 task groups. The reviewer read every teacher explanation and all five action bundles, plus actor-thought excerpts for six cases. One qualitative review was performed; no inter-rater reliability is claimed. Selection hashes and the complete codebook are in the aggregate. Source records, candidate identities and unchanged input hashes were verified; raw examples remain private.

</details>

<details>
<summary>Whole-subset text markers: descriptive only</summary>

| Literal marker family | Matching rationales | Rationales scanned | Match rate |
| --- | ---: | ---: | ---: |
| Current / visible / page / screen | 2,042 | 2,292 | 89.09% |
| Task / goal / user / requirements | 1,308 | 2,292 | 57.07% |
| Direct / efficient / redundant / avoid | 1,138 | 2,292 | 49.65% |
| History / already / repeat / prior | 731 | 2,292 | 31.89% |
| Likely / should / may / could | 1,577 | 2,292 | 68.80% |

These overlapping, case-insensitive regex matches count word presence once per rationale. They are **not** estimates of correct grounding, causal influence or semantic-category prevalence. Exact patterns are published in the aggregate. No new teacher, actor or browser calls were made.

</details>
