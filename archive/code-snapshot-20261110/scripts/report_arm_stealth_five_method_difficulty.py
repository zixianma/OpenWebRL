#!/usr/bin/env python3
"""Private preparation of aggregate-only five-method human-step difficulty rows.

Reuses reviewed outcomes; no bootstrap, model calls, archive scans or publication.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

import analyze_arm_stealth_overall_ci as verified
import report_arm_om2w_difficulty as original

BANDS = original.BANDS
EXPECTED_COUNTS = dict(easy=80, medium=141, hard=79)


def difficulty_from_reference_length(value):
    length = int(value)
    verified.need(not isinstance(value, bool) and length >= 1 and float(value) == length,
                  'Human reference length must be a positive integer')
    return 'easy' if length <= 5 else 'medium' if length <= 10 else 'hard'


def summarize_repeats(groups):
    verified.need(len(groups) == 3 and all(row['valid'] > 0 for row in groups),
                  'Expected three nonempty valid denominators')
    verified.need(len({row['tasks'] for row in groups}) == 1, 'Task-bin size changed')
    return dict(tasks_per_repeat=groups[0]['tasks'],
                overall_mean=statistics.mean(row['overall'] for row in groups),
                overall_sample_sd=statistics.stdev(row['overall'] for row in groups),
                valid_only_mean=statistics.mean(row['valid_only'] for row in groups),
                valid_only_sample_sd=statistics.stdev(row['valid_only'] for row in groups),
                successes_by_repeat=[row['successes'] for row in groups],
                valid_by_repeat=[row['valid'] for row in groups],
                invalid_by_repeat=[row['invalid'] for row in groups])


def same_fields(actual, expected, context):
    for key, value in expected.items():
        if isinstance(value, float):
            verified.need(abs(actual[key]-value) < 1e-12, f'{context}: {key} differs')
        else:
            verified.need(actual[key] == value, f'{context}: {key} differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-output', type=Path, required=True)
    args = parser.parse_args()
    output = args.private_output.resolve()
    verified.need(output.is_relative_to(verified.RUNTIME.resolve()), 'Outputs must remain private')
    output.mkdir(parents=True, exist_ok=True)
    read, sha = original.read, original.sha
    old_path = original.RESULTS / 'stealth-o4-t06-iteration90-difficulty.json'
    old = read(old_path)
    verified.need(sha(Path(original.__file__)) == old['implementation_sha256'],
                  'Existing difficulty implementation changed')
    source = original.RUNTIME / 'reference-arm-stealth90-o4-t06-20260929/online_mind2web_monitor.jsonl'
    verified.need(sha(source) == old['difficulty_source_sha256'], 'Frozen metadata differs')
    tasks = [json.loads(line)['metadata'] for line in source.read_text().splitlines()]
    metadata = {str(row['task_id']): row for row in tasks}
    ids = sorted(metadata)
    verified.need(len(ids) == len(tasks) == 300, 'Unexpected task universe')
    labels = np.array([difficulty_from_reference_length(metadata[ident]['reference_length']) for ident in ids])
    counts = dict(Counter(labels))
    verified.need(counts == EXPECTED_COUNTS == old['difficulty_counts'], 'Existing bins changed')
    mismatches = [ident for ident, band in zip(ids, labels) if metadata[ident]['difficulty'] != band]
    verified.need(set(mismatches) == {str(row['task_id']) for row in old['reference_length_label_mismatches']},
                  'Reference-length label corrections changed')
    success, valid, _, flat, sources, cohort_sources, provenance, private_sources, loader = verified.load_verified_inputs()
    verified.need(ids == sorted({row['task_id'] for row in provenance}), 'Outcome/metadata task sets differ')
    verified.need(success.shape == valid.shape == (300, 5, 3), 'Wrong outcome dimensions')
    per_repeat, summary, reconciliation = [], [], []
    for mi, method in enumerate(verified.METHODS):
        for repeat in range(3):
            for band in BANDS:
                mask = labels == band
                n, wins, nvalid = int(mask.sum()), int(success[mask, mi, repeat].sum()), int(valid[mask, mi, repeat].sum())
                verified.need(nvalid > 0, 'Empty valid denominator')
                per_repeat.append(dict(method=method, repeat=repeat+1, difficulty=band,
                    tasks=n, successes=wins, valid=nvalid, invalid=n-nvalid,
                    overall=wins/n, valid_only=wins/nvalid))
            groups = [row for row in per_repeat if row['method'] == method and row['repeat'] == repeat+1]
            observed = dict(tasks=sum(row['tasks'] for row in groups),
                            successes=sum(row['successes'] for row in groups),
                            valid=sum(row['valid'] for row in groups),
                            invalid=sum(row['invalid'] for row in groups))
            published = next(row for row in cohort_sources if row['method'] == method and row['repeat'] == repeat+1)
            verified.need(observed['tasks'] == 300 and observed['successes'] == published['successes']
                          and observed['valid'] == published['valid'] and observed['invalid'] == 300-published['valid'],
                          'Bin totals differ from full300 audited cohort')
            reconciliation.append(dict(method=method, repeat=repeat+1, **observed))
        for band in BANDS:
            groups = [row for row in per_repeat if row['method'] == method and row['difficulty'] == band]
            summary.append(dict(method=method, difficulty=band, **summarize_repeats(groups)))
        weighted = sum(row['overall_mean']*row['tasks_per_repeat'] for row in summary if row['method'] == method)/300
        totals = [row for row in reconciliation if row['method'] == method]
        valid_only = statistics.mean(row['successes']/row['valid'] for row in totals)
        verified.need(abs(weighted-flat[method]['overall_mean']) < 1e-12, 'Weighted overall mean differs')
        verified.need(abs(valid_only-flat[method]['valid_only_mean']) < 1e-12, 'Full300 valid-only mean differs')
    for row in summary:
        baseline = next(item for item in summary if item['method'] == 'baseline' and item['difficulty'] == row['difficulty'])
        row['overall_delta_baseline_pp'] = 100*(row['overall_mean']-baseline['overall_mean'])
    for row in old['summary']:
        actual = next(item for item in summary if (item['method'], item['difficulty']) == (row['method'], row['difficulty']))
        same_fields(actual, row, 'Existing summary')
    for row in old['per_repeat']:
        actual = next(item for item in per_repeat if (item['method'], item['repeat'], item['difficulty']) ==
                      (row['method'], row['repeat'], row['difficulty']))
        same_fields(actual, row, 'Existing per-repeat counts')
    private_receipt = dict(checked_utc=datetime.now(timezone.utc).isoformat(),
        metadata_source=str(source), metadata_source_sha256=sha(source),
        existing_difficulty_source=str(old_path), existing_difficulty_sha256=sha(old_path),
        verified_record_provenance_sha256=hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
        bin_assignment_sha256=hashlib.sha256(json.dumps(list(zip(ids, labels.tolist()))).encode()).hexdigest(),
        record_count=len(provenance), cohort_count=len(cohort_sources),
        old_summary_rows_reconciled=len(old['summary']), old_per_repeat_rows_reconciled=len(old['per_repeat']),
        all_bin_totals_reconcile=True, all_weighted_overall_means_reconcile=True,
        all_full300_valid_only_means_reconcile=True, cohort_totals=reconciliation,
        implementation_sha256=sha(Path(__file__)), outcome_loader_sha256=sha(Path(verified.__file__)),
        original_difficulty_implementation_sha256=sha(Path(original.__file__)),
        no_ci_or_significance_computation=True, no_archive_crc_rescans=True, new_model_calls=0, new_gpu_hours=0)
    receipt_path = output / 'difficulty-verification.json'
    receipt_path.write_text(json.dumps(private_receipt, indent=2)+'\n')
    report = dict(schema='openwebrl.stealth-five-method-difficulty.v1', iteration=90, repeats=3,
        tasks=300, cohorts=15, saved_outcomes=4500, protocol=old['protocol'],
        difficulty_definition=old['difficulty_definition'], difficulty_counts=EXPECTED_COUNTS,
        difficulty_source_sha256=sha(source), difficulty_definition_url=old['difficulty_definition_url'],
        label_policy='Apply the existing human-reference-length bins to the same frozen metadata; no evaluation artifacts or saved labels modified.',
        reference_length_label_mismatch_count=len(mismatches),
        reference_length_label_correction_summary='Two saved medium labels with human reference lengths11 and12 remain classified as hard, matching the previous table.',
        saved_difficulty_counts=old['saved_difficulty_counts'], summary=summary, per_repeat=per_repeat,
        collection_dates_pacific={method: '2026-09-29 to 2026-09-30' if mi < 3 else '2026-10-06'
                                  for mi, method in enumerate(verified.METHODS)},
        sources=sources+[dict(file=old_path.name, sha256=sha(old_path))], cohort_sources=cohort_sources,
        implementation_sha256=sha(Path(__file__)), outcome_loader_sha256=sha(Path(verified.__file__)),
        original_difficulty_implementation_sha256=sha(Path(original.__file__)),
        verification_receipt_sha256=sha(receipt_path),
        checks=['All4500 retained outcomes match the expected300 tasks, reviewed cohort counts and native judge identities.',
                'All9 original summary rows and27 original per-repeat bin rows reproduce the existing difficulty aggregate.',
                'Every cohort bin sum recovers its audited300-task success, valid and invalid totals.',
                'Difficulty-weighted overall means and reconstructed full300 per-repeat valid-only ratios reproduce all five published summaries.'],
        caveats=['Mean and sample SD (ddof1) over three evaluations of one fixed checkpoint per method, not independent training seeds.',
                 'Valid-only means average the three per-repeat ratios rather than pooling their denominators.',
                 'Invalid outcomes remain failures in each fixed difficulty-bin overall denominator.',
                 'Historical September29–30 versus mixed October6 collection windows differ; cross-date baseline deltas are descriptive, not matched causal comparisons.',
                 'Subgroup summaries only: no new confidence intervals or significance tests.',
                 'Original repeat1 retains the previously audited credit-outage recovery; no additional task retries or relabeling.',
                 'Human-reference-step difficulty is distinct from WebGym rubric-fact difficulty.'],
        new_model_calls=0, new_browser_calls=0, new_gpu_hours=0, archive_crc_rescans=0)
    verified.assert_public_safe(report)
    target = output / 'stealth-o4-t06-iteration90-five-method-difficulty.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(sanitized_aggregate=str(target), sha256=sha(target),
                         private_verification=str(receipt_path), verification_sha256=sha(receipt_path),
                         mixed_rows=[row for row in summary if row['method'].startswith('mixed-')]), indent=2))


if __name__ == '__main__':
    main()
