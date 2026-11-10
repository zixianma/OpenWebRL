#!/usr/bin/env python3
"""Verify five fixed-checkpoint marginal CIs from saved task outcomes only.

Writes private analysis artifacts, never docs or paired-test results. Resampling
uses 300 task clusters, each containing the complete three-repeat response vector.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

import analyze_arm_paired_evals as historical

RUNTIME = historical.RUNTIME
RESULTS = historical.RESULTS
METHODS = historical.METHODS + ['mixed-bonus', 'mixed-reweight']
MIXED = RUNTIME / 'arm-turn-bonus-preparation/mixed-stealth90-three-20261006'
SEED = 20260930
RESAMPLES = 50000


def need(condition, message):
    if not condition:
        raise ValueError(message)


def overall_intervals(success, *, resamples=RESAMPLES, seed=SEED):
    """Pointwise BCa intervals of task-cluster means, preserving exact ties."""
    success = np.asarray(success)
    need(success.ndim == 3 and success.shape[0] > 1 and success.shape[2] > 0,
         'Expected task x method x repeat outcomes')
    need(np.all((success == 0) | (success == 1)), 'Outcomes must be binary')
    counts = success.sum(axis=2, dtype=np.int64)
    boot = stats.bootstrap((counts.T,), np.mean, axis=-1, vectorized=True,
                           method='BCa', confidence_level=.95, n_resamples=resamples,
                           batch=256, rng=np.random.default_rng(seed))
    intervals = np.array(boot.confidence_interval).T / success.shape[2]
    need(np.isfinite(intervals).all(), 'Degenerate bootstrap interval')
    return intervals


def load_verified_inputs():
    read, sha = historical.read, historical.sha
    hp = RESULTS / 'stealth-iteration90-paired-inference.json'
    hs = RESULTS / 'stealth-o4-t06-iteration90-three-repeat-summary.json'
    mp = RESULTS / 'mixed-stealth90-three-repeats-20261006.json'
    prior, historical_summary, mixed_public = map(read, [hp, hs, mp])
    need(prior['seed'] == SEED and prior['resamples'] == RESAMPLES,
         'Historical bootstrap configuration changed')
    need(sha(Path(historical.__file__)) == prior['implementation_sha256'],
         'Historical implementation hash differs')
    success, valid, _, source_audits, provenance = historical.load_om2w()
    prior_sources = {row['file']: row['sha256'] for row in prior['sources']}
    need(all(prior_sources[row['file']] == row['sha256'] for row in source_audits),
         'Historical CI source audit hashes differ')
    ids = sorted({row['task_id'] for row in provenance})
    need(len(ids) == 300, 'Wrong historical task universe')

    helper = MIXED / 'prepare_final_repeat_summary.py'
    need(sha(helper) == mixed_public['calculation']['implementation_sha256'],
         'Mixed analysis implementation changed')
    spec = importlib.util.spec_from_file_location('verified_mixed_summary', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ready, pending = module.reviewed_cohorts()
    need(not pending and len(ready) == 6, 'Mixed cohorts lack independent final reviews')
    mixed_success = np.zeros((300, 2, 3), dtype=np.int64)
    mixed_valid = np.zeros_like(mixed_success)
    public_cohorts = {(row['method'], row['repeat']): row for row in mixed_public['cohorts']}
    private_sources = []
    for audit, audit_path, review_path in ready:
        method, repeat = audit['method'], audit['repeat']
        published = public_cohorts[method, repeat]
        need(published['full_runtime_audit_sha256'] == sha(audit_path) and
             published['independent_review_sha256'] == sha(review_path),
             'Published mixed audit/review hashes differ')
        pairs, task_provenance = module.task_jsons(audit, ids)
        array = np.asarray(pairs, dtype=np.int64)
        mi = module.METHODS.index(method)
        mixed_success[:, mi, repeat-1] = array[:, 0]
        mixed_valid[:, mi, repeat-1] = array[:, 1]
        provenance.extend(task_provenance)
        private_sources.append(dict(audit=str(audit_path), audit_sha256=sha(audit_path),
                                    review=str(review_path), review_sha256=sha(review_path)))
    paired = mixed_public['paired_comparison']
    need(paired['bootstrap_seed'] == SEED and paired['bootstrap_resamples'] == RESAMPLES,
         'Mixed bootstrap configuration differs')
    existing = {method: prior['om2w']['methods'][method]['overall_ci95']
                for method in historical.METHODS}
    existing.update(paired['method_overall_ci95'])
    summaries = {method: historical_summary['methods'][method] for method in historical.METHODS}
    summaries.update(mixed_public['mean_and_sample_sd'])
    sources = [dict(file=p.name, sha256=sha(p)) for p in [hp, hs, mp]]
    cohort_sources = [dict(method=method, repeat=row['repeat'], file=row['source'],
                           sha256=row['sha256'], successes=row['successes'], valid=row['valid'])
                      for method in historical.METHODS
                      for row in historical_summary['methods'][method]['per_repeat']]
    cohort_sources += [dict(method=row['method'], repeat=row['repeat'], file=mp.name,
                            aggregate_sha256=sha(mp),
                            full_runtime_audit_sha256=row['full_runtime_audit_sha256'],
                            independent_review_sha256=row['independent_review_sha256'],
                            checked_task_records_sha256=row['checked_task_records_sha256'],
                            successes=row['successes'], valid=row['valid'])
                       for row in mixed_public['cohorts']]
    need(len(cohort_sources) == 15, 'Expected fifteen source cohorts')
    return (np.concatenate([success, mixed_success], axis=1),
            np.concatenate([valid, mixed_valid], axis=1), existing, summaries,
            sources, cohort_sources, provenance, private_sources, helper)


def assert_public_safe(value):
    text = json.dumps(value)
    need('/gpfs/' not in text and '/tmp/' not in text, 'Private path in sanitized payload')
    def visit(item):
        if isinstance(item, dict):
            need(not {'task_id', 'task_ids', 'start_url', 'instruction', 'record'} & item.keys(),
                 'Private task data in sanitized payload')
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-output', type=Path, required=True)
    parser.add_argument('--prepare-integer-ci-payload', action='store_true',
                        help='Prepare a sanitized local aggregate after parent selects integer-count CIs')
    args = parser.parse_args()
    output = args.private_output.resolve()
    need(output.is_relative_to(RUNTIME.resolve()), 'Analysis outputs must remain in private runtime')
    output.mkdir(parents=True, exist_ok=True)
    success, valid, existing, summaries, sources, cohort_sources, provenance, private_sources, helper = load_verified_inputs()
    need(success.shape == valid.shape == (300, 5, 3), 'Wrong cluster dimensions')
    intervals = overall_intervals(success)
    methods = {}
    for mi, method in enumerate(METHODS):
        successes, valids = success[:, mi].sum(axis=0), valid[:, mi].sum(axis=0)
        overall, valid_only = successes / 300, successes / valids
        row = dict(repeats=3, tasks_per_repeat=300,
                   overall_mean=float(overall.mean()), overall_sample_sd=float(overall.std(ddof=1)),
                   valid_only_mean=float(valid_only.mean()), valid_only_sample_sd=float(valid_only.std(ddof=1)),
                   per_repeat_successes=successes.tolist(), per_repeat_valid=valids.tolist(),
                   per_repeat_invalid=(300-valids).tolist(),
                   existing_overall_ci95=existing[method], integer_count_overall_ci95=intervals[mi].tolist(),
                   interval_change_pp=(100*(intervals[mi]-np.asarray(existing[method]))).tolist(),
                   rounded_percentage_endpoints_match=[f'{100*a:.2f}' for a in intervals[mi]] ==
                       [f'{100*a:.2f}' for a in existing[method]],
                   collection_dates_pacific='2026-09-29 to 2026-09-30' if mi < 3 else '2026-10-06')
        for key in ['overall_mean', 'overall_sample_sd', 'valid_only_mean', 'valid_only_sample_sd']:
            need(abs(row[key] - summaries[method][key]) < 1e-12, f'{method}: published {key} differs')
        if mi < 3:
            published = summaries[method]['per_repeat']
            need(successes.tolist() == [r['successes'] for r in published] and
                 valids.tolist() == [r['valid'] for r in published], 'Historical per-repeat counts differ')
        else:
            need(successes.tolist() == summaries[method]['per_repeat_successes'] and
                 valids.tolist() == summaries[method]['per_repeat_valid'], 'Mixed per-repeat counts differ')
        methods[method] = row
    report = dict(schema='openwebrl.stealth-overall-ci-consistency.v1',
        checked_utc=datetime.now(timezone.utc).isoformat(), iteration=90, methods=methods,
        inference_unit='300 task clusters; retain all three observed repeats within every resampled task.',
        estimand='Mean full300 overall success of each fixed checkpoint across its three observed evaluation repeats; invalid attempts remain failures.',
        interval_method='Pointwise 95% task-cluster BCa bootstrap', bootstrap_seed=SEED,
        bootstrap_resamples=RESAMPLES, bootstrap_batch=256,
        numerical_convention='Bootstrap integer per-task three-repeat success sums and divide confidence limits by three; equal counts retain exact BCa bias-correction ties.',
        existing_numerical_conventions=dict(historical='Floating per-task three-repeat means', mixed='Integer per-task three-repeat sums, confidence limits divided by three'),
        all_rounded_intervals_match_existing=all(r['rounded_percentage_endpoints_match'] for r in methods.values()),
        published_ci_selection_pending_parent=not args.prepare_integer_ci_payload,
        existing_paired_tests_unchanged=True,
        valid_only_aggregation='Unweighted mean/sample SD of three per-repeat valid-only ratios; no valid-only CI calculated.',
        sources=sources, cohort_sources=cohort_sources, implementation_sha256=historical.sha(Path(__file__)),
        verified_loader_sha256=dict(historical=historical.sha(Path(historical.__file__)), mixed=historical.sha(helper)),
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        caveats=['Conditional uncertainty across tasks for fixed trained checkpoints and observed evaluation windows; not training-seed variability.',
                 'September and October collections are not contemporaneous; cross-window comparisons remain descriptive.',
                 'Intervals are pointwise, not simultaneous five-method coverage, and CI overlap is not a paired significance test.',
                 'Task resampling does not capture arbitrary shared website/time shocks or systematic judge error.',
                 'Exploratory checkpoint analysis; no new comparisons, paired tests, judge requests, task retries or archive CRC scans.'],
        new_gpu_hours=0, new_model_calls=0, archive_crc_rescans=0)
    assert_public_safe(report)
    target = output / 'overall-ci-consistency.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    provenance_path = output / 'private-task-provenance.json'
    provenance_path.write_text(json.dumps(dict(task_records=provenance, mixed_reviews=private_sources), indent=2)+'\n')
    if args.prepare_integer_ci_payload:
        public_methods = {}
        for method, row in methods.items():
            public_methods[method] = {key: row[key] for key in [
                'repeats', 'tasks_per_repeat', 'overall_mean', 'overall_sample_sd',
                'valid_only_mean', 'valid_only_sample_sd', 'per_repeat_successes',
                'per_repeat_valid', 'per_repeat_invalid', 'collection_dates_pacific']}
            public_methods[method]['overall_ci95'] = row['integer_count_overall_ci95']
        aggregate = {key: value for key, value in report.items()
                     if key not in ['methods', 'existing_numerical_conventions',
                                    'all_rounded_intervals_match_existing', 'published_ci_selection_pending_parent']}
        aggregate.update(schema='openwebrl.stealth-five-method-overall-ci.v1', methods=public_methods,
            interval_selection='Uniform integer-count BCa across all five methods; parent-selected after numerical consistency check.',
            numerical_consistency=dict(previous_baseline_lower_percent=51.0,
                integer_count_baseline_lower_percent=100*methods['baseline']['integer_count_overall_ci95'][0],
                baseline_lower_change_percentage_points=methods['baseline']['interval_change_pp'][0],
                other_displayed_endpoints_unchanged=True,
                explanation='Integer task sums preserve mathematical ties in BCa bias correction. Earlier floating one-third task means split some equal-count draws by rounding, changing the baseline lower endpoint by0.11percentage points. The estimand is unchanged.',
                archived_paired_tests_and_plots_unchanged=True),
            verification=dict(all_15_cohort_counts_and_reported_means_reconciled=True,
                all_mixed_task_jsons_match_full_audit_digests=True,
                consistency_receipt_sha256=historical.sha(target),
                private_task_provenance_sha256=historical.sha(provenance_path)))
        tie_path = output / 'numeric-tie-evidence.json'
        if tie_path.exists():
            aggregate['verification']['private_numeric_tie_evidence_sha256'] = historical.sha(tie_path)
        assert_public_safe(aggregate)
        public_path = output / 'stealth-iteration90-five-method-overall-ci.json'
        public_path.write_text(json.dumps(aggregate, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(sanitized_payload=str(public_path), sha256=historical.sha(public_path))))
    print(json.dumps(dict(receipt=str(target), sha256=historical.sha(target), methods=methods,
                         all_rounded_intervals_match_existing=report['all_rounded_intervals_match_existing']), indent=2))


if __name__ == '__main__':
    main()
