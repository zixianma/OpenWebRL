# ARM preference v2: lightweight CPU audit

Completed 2026-09-10. The structural audit supports preparing the matched
preference-versus-SFT pilot, but the provisional pairs are **not training-ready**.
No model inference, GPU work, or allocation submission was performed.

## Retention and integrity

Audited all 8,394 C2 winner turns from
`c2-full-282782-20260908T075414Z/training.jsonl`. Dataset SHA256:
`cb7c75df6a4824e9e653f6d913b0ae83268610966cd13dd13fc7314e9c667fe0`.

| Check / filter | States retained | Fraction of source |
| --- | ---: | ---: |
| At least one distinct schema-valid alternative | 7,140 | 85.1% |
| Also exclude coordinate-only differences within 2 units | 6,705 | 79.9% |
| Within 5 units | 6,247 | 74.4% |
| Within 10 units | 6,104 | 72.7% |
| Radius 5, also exclude every done-versus-done pair | 5,490 | 65.4% |

Coordinate thresholds use the stored coordinate system, not screenshot pixels.
For otherwise identical multi-call actions, distance is the maximum Euclidean
distance over corresponding coordinate arguments. A threshold excludes nearby
alternatives; it does not establish semantic identity or correctness.

The audit excluded 8,950 exact action-equivalent alternatives (26.7% of 33,560
alternatives inspected), including differences only in reasoning, and 4,167
repeated distinct alternatives within a state. These are candidate counts,
not counts of selected training pairs. Another 1,250 states had no distinct
schema-valid alternative. Four selected winners failed required-argument
validation; their source line/task IDs are recorded in the machine report.

Dataset/source/prompt hashes, execution-source joins, saved prompt-token IDs
and image grids, successful valid terminal labels, and absence of selector
fallback were checked. No provenance failures were recorded; the four failures
were winner schema checks. All 7,471 referenced image paths exist, with no
conflicting declared hashes. Image bytes were not hashed or decoded. Stored
terminal success is a trajectory label, not proof that each action is correct.

## Issues revealed by the audit

1. **Exact JSON inequality is insufficient.** A text-only skim of the 64-example
   review manifest found final-answer paraphrases and coordinate differences
   that could still target the same element. Screenshot semantic review remains
   pending. At radius 5 there are 3,297 done-versus-done alternatives; removing
   them all loses 757 states, leaving 5,490 states across 1,105 tasks. Some such
   pairs contain genuinely different answers, so excluding all is a conservative
   first-pilot choice, not a claim that all are equivalent. Retain done-versus-
   continue candidates when otherwise eligible; winner SFT can still supervise
   the selected final answer.
2. **The provisional sampler changes the action distribution.** Equal cycling
   across website/action groups produces 521/2,048 (25.4%) terminal-only winners
   in training and 64/256 (25.0%) in validation, versus 1,120/6,247 (17.9%) in the
   radius-5 pool. Use proportional action sampling, with explicit rare-group
   coverage, before freezing the experiment. Current website strata use the
   captured current URL, which can differ from the task's initial website.
3. **ARM preference is still a noisy label.** Alternatives were not executed.
   Schema validity and different actions do not imply the selected action is
   better. The audit establishes available data and structural exclusions,
   not that preference distillation will improve browser success.

The provisional radius-5 split contains 2,048 training states from 710 tasks,
256 validation states from 138 disjoint tasks, and 128 calibration states drawn
from training. The held-out task pool was assigned before sampling states.
These manifests contain eligible alternatives, not frozen model-hard negatives.
Rebuild them after the semantic exclusion and sampling decisions; recheck
per-split capacity after exclusions rather than assuming it from total counts.

## Resource footprint and reproducibility

The corrected full scan took **102.9 seconds wall time, 16.7 CPU-seconds, and
119.3 MiB peak RSS**. It ran as one process at nice 15 with a 10 ms per-state
sleep, a 768 MiB address-space cap, and a 180-second CPU soft limit. It read
approximately 1.68 GB of JSON across dataset, source, and execution files.
No numerical libraries or model weights were loaded.

An earlier scan selected an empty documentation example of the tool-schema
tag and therefore produced invalid schema results. It cost another 12.3
CPU-seconds over 111.1 seconds wall time, peaking at 64.3 MiB. That output is
explicitly marked [invalid](arm_results/preference_v2_cpu_audit/INVALID.md);
all findings here use the corrected `r2` output. Resource figures describe the
audit processes, not every preliminary inspection command in the session.

The [audit script](../../scripts/audit_arm_preference_pairs.py) uses a
stdlib mirror of browser tool parsing plus schema validation. Five targeted
unit tests passed, including documented defaults, call ordering, coordinate
comparison, malformed inputs, and the empty-schema-tag regression. Native
parser parity and tokenizer-based action-mask alignment remain pending.

```bash
nice -n 15 python3 scripts/audit_arm_preference_pairs.py \
  --source-root /gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/c2-full-282782-20260908T075414Z \
  --output /tmp/arm-preference-v2-audit-recheck \
  --sleep-ms 10
python3 -m unittest tests/test_arm_preference_audit.py
```

## Artifacts and next steps

- [Complete machine report and manifest hashes](arm_results/preference_v2_cpu_audit_r2/audit.json)
- [Eligible states](arm_results/preference_v2_cpu_audit_r2/eligible-states.jsonl)
- [Provisional training states](arm_results/preference_v2_cpu_audit_r2/provisional-train.jsonl)
- [Provisional validation states](arm_results/preference_v2_cpu_audit_r2/provisional-validation.jsonl)
- [Provisional calibration states](arm_results/preference_v2_cpu_audit_r2/provisional-calibration.jsonl)
- [64 examples awaiting screenshot review](arm_results/preference_v2_cpu_audit_r2/review-64.jsonl)
- [Controlled viability plan](ARM_PREFERENCE_V2_VIABILITY_PLAN.md)

Before training: conservatively handle final-answer paraphrases, review visual
equivalence, rebuild representative disjoint samples, and verify image hashes,
native parser parity, and token masks. Frozen-base scoring, hard-negative
selection, gradient calibration, and matched training belong on a compute
node. No new compute budget has been requested or approved by this audit.
