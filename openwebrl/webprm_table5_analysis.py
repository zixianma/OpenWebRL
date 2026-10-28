"""Offline Table 5 accounting for the fixed 155-task, 1,550-cell grid.

Inputs must already have independent terminal/trajectory/verdict audits. This
module validates normalized identities and arithmetic; it does not perform
those audits or infer semantic correctness from an apparently valid verdict.
No input task text, trajectories, credentials, or individual outcomes are
copied into its aggregate output.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import random
import re
from statistics import NormalDist

from scripts.prepare_webprm_table5_campaign import ACTORS, SELECTORS, COUNTS, TASKS_SHA

DOMAINS = tuple(COUNTS)
SEMANTICS = {'valid', 'invalid', 'unreviewed', 'not_applicable'}
RECORD_CONTRACT = {
    'identity': ['cell_id', 'released_task_id', 'actor_model', 'selector', 'domain',
                 'task_config_sha256', 'tasks_sha256', 'cohort_sha256', 'protocol_sha256'],
    'independent_audit': ['audit_verified', 'audit_sha256', 'terminal'],
    'outcomes': ['attempted', 'valid', 'success', 'native_success',
                 'effective_semantic_status', 'native_semantic_status'],
    'success_encoding': 'Integral 0/1, or null when unavailable; booleans must be normalized explicitly.',
    'invalid_outcomes': 'Invalid/unattempted effective success is null. An invalid raw native verdict may be retained diagnostically.',
    'semantic_statuses': sorted(SEMANTICS),
    'semantic_review_scope': 'Conformance to the declared evaluated protocol and known-defect review; valid does not certify an infallible success judge.',
    'protocol_binding': 'SHA256 of the shared scientific protocol_contract; runtime-specific worker identity is separate.',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_cohort(cohort):
    """Require the exact five-domain/task-count grid prepared for this study."""
    if (not isinstance(cohort, dict) or 'protocol_sha256' not in cohort or cohort.get('task_count') != 155 or
            cohort.get('episode_count') != 1550 or cohort.get('domain_counts') != COUNTS or
            cohort.get('actors') != list(ACTORS) or cohort.get('selectors') != list(SELECTORS) or
            cohort.get('source_sha256') != TASKS_SHA or cohort.get('primary_attempts_per_cell') != 1 or
            cohort.get('replacement_task_admission') is not False or
            cohort.get('protocol_sha256') is not None and not _sha(cohort['protocol_sha256'])):
        raise ValueError('Expected the fixed 155-task/1,550-cell Table 5 cohort and one primary attempt')
    cells = cohort.get('cells')
    if not isinstance(cells, list) or len(cells) != 1550:
        raise ValueError('Missing or extra planned cells')
    indexed, tasks = {}, {}
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError('Planned cell must be a mapping')
        task_id, actor, selector = cell.get('task_id'), cell.get('actor'), cell.get('selector')
        if (type(task_id) is not int or task_id < 0 or actor not in ACTORS or selector not in SELECTORS or
                cell.get('domain') not in COUNTS or not _sha(cell.get('task_config_sha256')) or
                type(cell.get('original_task_id')) is not int):
            raise ValueError('Invalid planned cell identity')
        expected = f'webarena_lite_{task_id:03d}__{actor}__{selector}'
        if cell.get('cell_id') != expected or expected in indexed:
            raise ValueError('Duplicate or malformed planned cell ID')
        identity = (cell['domain'], cell['task_config_sha256'], cell['original_task_id'])
        if task_id in tasks and tasks[task_id] != identity:
            raise ValueError('Mixed task configuration identities within the planned grid')
        tasks[task_id] = identity
        indexed[expected] = cell
    if (len(tasks) != 155 or Counter(v[0] for v in tasks.values()) != Counter(COUNTS) or
            set((c['task_id'], c['actor'], c['selector']) for c in cells) !=
            {(t, a, s) for t in tasks for a in ACTORS for s in SELECTORS}):
        raise ValueError('Planned cohort must be the complete paired Cartesian grid')
    return indexed


def _validate_records(cohort, planned, records):
    indexed = {}
    cohort_sha = digest(cohort)
    for row in records:
        if not isinstance(row, dict) or row.get('cell_id') not in planned:
            raise ValueError('Foreign or malformed result cell')
        key = row['cell_id']
        if key in indexed:
            raise ValueError('Duplicate result cell; primary attempts must be independently resolved before aggregation')
        cell = planned[key]
        expected = dict(released_task_id=cell['task_id'], actor_model=cell['actor'], selector=cell['selector'],
                        domain=cell['domain'], task_config_sha256=cell['task_config_sha256'],
                        tasks_sha256=cohort['source_sha256'], cohort_sha256=cohort_sha,
                        protocol_sha256=cohort['protocol_sha256'])
        if (cohort['protocol_sha256'] is None or any(row.get(k) != v for k,v in expected.items()) or
                type(row.get('released_task_id')) is not int):
            raise ValueError('Mixed or unbound protocol/cohort/task identities')
        if (row.get('audit_verified') is not True or not _sha(row.get('audit_sha256')) or
                row.get('terminal') is not True):
            raise ValueError('Independently audited terminal records are required')
        if type(row.get('attempted')) is not bool or type(row.get('valid')) is not bool:
            raise ValueError('Attempted and valid must be explicit booleans')
        if row['valid'] and not row['attempted']:
            raise ValueError('Unattempted records cannot be valid')
        for score_name, semantic_name in [('success', 'effective_semantic_status'),
                                           ('native_success', 'native_semantic_status')]:
            if score_name not in row or semantic_name not in row:
                raise ValueError('Both native and effective outcomes and semantic statuses are required')
            score, semantic = row[score_name], row[semantic_name]
            if score is not None and (type(score) is not int or score not in (0, 1)):
                raise ValueError('Success must be an explicit integral binary verdict or null')
            if semantic not in SEMANTICS or (score is None) != (semantic == 'not_applicable'):
                raise ValueError('Outcome availability and semantic audit status disagree')
        if row['valid'] and (row['success'] is None or row['native_success'] is None):
            raise ValueError('Valid records require both independently audited effective and native outcomes')
        if not row['valid'] and row['success'] is not None:
            raise ValueError('Invalid records cannot carry a task-success verdict')
        if not row['attempted'] and row['native_success'] is not None:
            raise ValueError('Unattempted records cannot carry a native verdict')
        indexed[key] = row
    return indexed


def wilson_interval(successes, n):
    """Two-sided 95% Wilson interval in percent; null for zero denominator."""
    if type(successes) is not int or type(n) is not int or not 0 <= successes <= n:
        raise ValueError('Integral valid binomial counts required')
    if not n:
        return None
    z = NormalDist().inv_cdf(.975)
    p, z2 = successes/n, z*z
    center = (p+z2/(2*n))/(1+z2/n)
    half = z*math.sqrt(p*(1-p)/n+z2/(4*n*n))/(1+z2/n)
    return [100*max(0., center-half), 100*min(1., center+half)]


def stratified_paired_bootstrap(differences, *, draws=2000, seed=20261011):
    """Resample paired task differences separately inside all five domains.

    Each resampled domain contributes one fifth regardless of its task count.
    Pairing is preserved by resampling differences, never two arms separately.
    """
    if type(draws) is not int or not 100 <= draws <= 100000 or type(seed) is not int:
        raise ValueError('Require 100–100,000 bootstrap draws and an integral seed')
    if set(differences) != set(DOMAINS):
        raise ValueError('The five explicit domain strata are required')
    groups = [differences[d] for d in DOMAINS]
    if any(any(type(x) is not int or x not in (-1, 0, 1) for x in group) for group in groups):
        raise ValueError('Each paired task difference must be -1, 0, or 1')
    if any(not group for group in groups):
        return None
    rng = random.Random(seed)
    samples = sorted(100*sum(sum(group[rng.randrange(len(group))] for _ in group)/len(group)
                             for group in groups)/len(DOMAINS) for _ in range(draws))
    def quantile(q):
        position = (len(samples)-1)*q
        low = int(position)
        fraction = position-low
        return samples[low]*(1-fraction)+samples[min(low+1, len(samples)-1)]*fraction
    return [quantile(.025), quantile(.975)]


def _counts(cells, indexed):
    rows = [indexed[c['cell_id']] for c in cells if c['cell_id'] in indexed]
    attempted = sum(r['attempted'] for r in rows)
    valid = sum(r['valid'] for r in rows)
    effective = sum(r['success'] for r in rows if r['valid'])
    native = sum(r['native_success'] for r in rows if r['valid'])
    return dict(planned=len(cells), recorded=len(rows), attempted=attempted, valid=valid,
        invalid=attempted-valid, missing=len(cells)-attempted,
        not_recorded=len(cells)-len(rows), recorded_unattempted=len(rows)-attempted,
        effective_successes=effective, native_successes_on_valid_episodes=native,
        raw_native_positive_verdicts=sum(r['native_success']==1 for r in rows),
        raw_native_positive_verdicts_on_invalid_episodes=sum(not r['valid'] and r['native_success']==1 for r in rows),
        changed_valid_verdicts=sum(r['valid'] and r['success']!=r['native_success'] for r in rows),
        effective_semantic_invalid=sum(r['effective_semantic_status']=='invalid' for r in rows),
        effective_semantic_unreviewed=sum(r['effective_semantic_status']=='unreviewed' for r in rows),
        native_semantic_invalid=sum(r['native_semantic_status']=='invalid' for r in rows),
        native_semantic_unreviewed=sum(r['native_semantic_status']=='unreviewed' for r in rows))


def _rates(counts, publish):
    return dict(success_percent=100*counts['effective_successes']/counts['planned'] if publish else None,
        denominator=counts['planned'], wilson95_percent=wilson_interval(counts['effective_successes'],counts['planned']) if publish else None,
        valid_only_success_percent=100*counts['effective_successes']/counts['valid'] if publish and counts['valid'] else None,
        valid_only_denominator=counts['valid'])


def _comparison(actor, selector, cells, indexed, publish, draws, seed):
    domain_pairs = {d:[] for d in DOMAINS}
    index = {(c['task_id'], c['actor'], c['selector']): c for c in cells}
    for cell in cells:
        if cell['actor'] != actor or cell['selector'] != selector:
            continue
        left = indexed.get(cell['cell_id'])
        right = indexed.get(index[cell['task_id'], actor, 'no_search']['cell_id'])
        if left is not None and right is not None and left['attempted'] and right['attempted']:
            a = left['success'] if left['valid'] else 0
            b = right['success'] if right['valid'] else 0
            domain_pairs[cell['domain']].append((a,b,left['valid'] and right['valid']))
    def summarize(common_only):
        domains, differences = {}, {}
        for domain, pairs in domain_pairs.items():
            rows = [(a,b) for a,b,common in pairs if common or not common_only]
            n = len(rows)
            denominator = n if common_only else COUNTS[domain]
            a, b = sum(x for x,_ in rows), sum(y for _,y in rows)
            differences[domain] = [x-y for x,y in rows]
            domains[domain] = dict(planned=COUNTS[domain], pairs=n, denominator=denominator,
                missing_pairs=COUNTS[domain]-len(pairs), selector_successes=a, no_search_successes=b,
                wins=sum(x>y for x,y in rows), losses=sum(x<y for x,y in rows),
                both_success=sum(x==y==1 for x,y in rows), both_failure=sum(x==y==0 for x,y in rows),
                gain_percentage_points=100*(a-b)/denominator if publish and denominator else None)
        total = sum(x['pairs'] for x in domains.values())
        denominator = total if common_only else 155
        gain = sum(x['selector_successes']-x['no_search_successes'] for x in domains.values())
        can_macro = publish and all(x['denominator'] for x in domains.values())
        stream_seed = int(digest([seed,actor,selector,'common_valid' if common_only else 'primary'])[:16],16)
        return dict(domains=domains, observed_pairs=total, denominator=denominator,
            pooled_gain_percentage_points=100*gain/denominator if publish and denominator else None,
            macro_gain_percentage_points=sum(x['gain_percentage_points'] for x in domains.values())/5 if can_macro else None,
            macro_gain_bootstrap95_percentage_points=stratified_paired_bootstrap(differences,draws=draws,seed=stream_seed) if can_macro else None,
            no_common_valid_domains=[d for d,x in domains.items() if not x['pairs']] if common_only else [],
            interval='Paired task percentile bootstrap, stratified by domain; equal 1/5 domain weights.',
            bootstrap_draws=draws if can_macro else 0, bootstrap_seed=stream_seed if can_macro else None)
    return dict(actor_model=actor, selector=selector, baseline='no_search',
                primary_invalid_as_failure=summarize(False), common_valid_secondary=summarize(True))


def aggregate_table5(cohort, records, *, bootstrap_draws=2000, seed=20261011):
    """Count partial progress, but release effective rates only for a full audit.

    Missing includes absent records and explicitly unattempted dispositions.
    Invalid attempts remain failures in the fixed planned denominators. Known
    bad or unreviewed effective judgments suppress *every* performance estimate;
    corrected effective verdicts may coexist with diagnostically bad raw ones.
    """
    if type(bootstrap_draws) is not int or not 100 <= bootstrap_draws <= 100000 or type(seed) is not int:
        raise ValueError('Explicit bounded bootstrap settings required')
    planned = validate_cohort(cohort)
    indexed = _validate_records(cohort,planned,records)
    cells = cohort['cells']
    counts = _counts(cells,indexed)
    reasons = []
    if cohort['protocol_sha256'] is None:reasons.append('scientific_protocol_not_bound')
    if counts['missing']:reasons.append('incomplete_fixed_cohort')
    if counts['effective_semantic_invalid']:reasons.append('known_invalid_effective_judgments')
    if counts['effective_semantic_unreviewed']:reasons.append('unreviewed_effective_judgments')
    coverage_gaps = []
    for actor in ACTORS:
        for selector in SELECTORS:
            for domain in DOMAINS:
                group = [c for c in cells if c['actor']==actor and c['selector']==selector and c['domain']==domain]
                coverage = _counts(group,indexed)
                if not coverage['valid']:
                    coverage_gaps.append(dict(actor_model=actor,selector=selector,domain=domain,
                        planned=coverage['planned'],attempted=coverage['attempted'],valid=0))
    if coverage_gaps:reasons.append('domain_method_groups_without_valid_episodes')
    publish = not reasons
    methods = []
    for actor in ACTORS:
        for selector in SELECTORS:
            group = [c for c in cells if c['actor']==actor and c['selector']==selector]
            total = _counts(group,indexed)
            domains = {}
            for domain in DOMAINS:
                c = _counts([x for x in group if x['domain']==domain],indexed)
                domains[domain] = dict(counts=c,rates=_rates(c,publish))
            rates = _rates(total,publish)
            rates['pooled_success_percent'] = rates.pop('success_percent')
            rates.pop('wilson95_percent')  # Wilson intervals are per domain, not a macro interval.
            for field in ('denominator', 'valid_only_success_percent', 'valid_only_denominator'):
                rates['pooled_'+field] = rates.pop(field)
            rates['macro_success_percent'] = sum(x['rates']['success_percent'] for x in domains.values())/5 if publish else None
            rates['macro_domain_denominators'] = dict(COUNTS)
            rates['macro_valid_only_success_percent'] = (sum(x['rates']['valid_only_success_percent'] for x in domains.values())/5
                if publish and all(x['counts']['valid'] for x in domains.values()) else None)
            methods.append(dict(actor_model=actor,selector=selector,counts=total,domains=domains,rates=rates))
    return dict(schema_version=1,status='complete_audited_effective_results' if publish else 'counts_only',
        scope='Fixed WebArbiter Table 5 reconstruction plus Piotr SelectionARM; not proof of historical numerical replication.',
        cohort_sha256=digest(cohort),tasks_sha256=cohort['source_sha256'],protocol_sha256=cohort['protocol_sha256'],
        paper_exact_task_membership_verified=cohort.get('paper_exact_task_membership_verified',False),
        counts=counts,performance_rates_withheld=not publish,withheld_reasons=reasons,
        terminal_grid_complete=counts['missing']==0,all_episodes_valid=counts['valid']==1550,
        coverage_gaps=coverage_gaps,
        completion_limitation='Complete means the fixed grid has audited terminal dispositions, not that Table 5 is numerically reproduced or every episode is valid. Primary invalid-as-failure estimates include infrastructure failures and are not model-only success rates.',
        domain_weights={d:1/5 for d in DOMAINS},methods=methods,
        comparisons=[_comparison(a,s,cells,indexed,publish,bootstrap_draws,seed)
                     for a in ACTORS for s in SELECTORS if s!='no_search'],
        normalization_contract=RECORD_CONTRACT,
        audit_scope='Consumes independently audited terminal records; does not itself inspect raw trajectories, replay native verdicts, or verify audit-file contents. Cohort validation checks the declared grid/hash identities, not source-file contents; released task membership must be verified upstream.',
        uncertainty='Wilson intervals describe each domain binomial rate. Paired domain-stratified bootstrap describes task resampling conditional on fixed domains; it excludes actor-seed, website-drift, and judge uncertainty.',
        count_semantics='planned = valid + invalid + missing; missing includes unattempted records. Success counts are recorded verdict positives on valid episodes, not external ground truth. Raw native positives on invalid episodes remain diagnostic only.',
        raw_native_reporting='Native counts are preserved separately; they never replace effective rates or validate semantic judgments.')
