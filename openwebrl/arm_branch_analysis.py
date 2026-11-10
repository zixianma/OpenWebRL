"""State-clustered analysis of frozen choices; never best-of-continuations."""
from collections import Counter, defaultdict
from itertools import combinations, product
import random
from statistics import mean


def paired_summary(records, judgments, *, candidates=5, repetitions=3,
                   teacher_repetitions=3, bootstrap_samples=10000, seed=0):
    outcomes = {}
    for row in records:
        key = row['state'], row['candidate'], row['repeat']
        if key in outcomes:
            raise ValueError('Duplicate continuation identity')
        outcomes[key] = row
    choices = defaultdict(dict)
    for row in judgments:
        key = row['condition'], row['execution_repeat'], row['teacher_repeat']
        if key in choices[row['state']]:
            raise ValueError('Duplicate teacher judgment identity')
        if row.get('valid') and row['candidate'] not in range(candidates):
            raise ValueError('Teacher candidate outside crossed design')
        choices[row['state']][key] = row
    rows, exclusions = [], []
    states = sorted({x[0] for x in outcomes})
    for state in states:
        expected = {(state, c, r) for c in range(candidates) for r in range(repetitions)}
        actual = {x for x in outcomes if x[0] == state}
        if actual != expected:
            exclusions.append(dict(state=state, reason='incomplete_crossed_continuations'))
            continue
        required = [('before', None, t) for t in range(teacher_repetitions)]
        required += [('after', r, t) for r in range(repetitions) for t in range(teacher_repetitions)]
        if any(not choices[state].get(k, {}).get('valid') for k in required):
            exclusions.append(dict(state=state, reason='incomplete_valid_teacher_repeats'))
            continue
        before = [choices[state]['before', None, t]['candidate'] for t in range(teacher_repetitions)]
        after = [[choices[state]['after', r, t]['candidate'] for t in range(teacher_repetitions)]
                 for r in range(repetitions)]
        def outcome(c, r):
            row = outcomes[state, c, r]
            return bool(row['valid']), int(row['valid'] and row['reward'] == 1)
        pairs = [(outcome(before[t], r), outcome(after[r][t], r))
                 for r in range(repetitions) for t in range(teacher_repetitions)]
        common = [(b, a) for b, a in pairs if b[0] and a[0]]
        b = mean(x[0][1] for x in pairs); a = mean(x[1][1] for x in pairs)
        before_noise = mean(x != y for x, y in combinations(before, 2))
        after_noise = mean(x != y for panel in after for x, y in combinations(panel, 2))
        flip = mean(x != y for panel in after for x, y in product(before, panel))
        permutations = {}
        for condition, rep, regular in [('before', None, before), ('after', 0, after[0])]:
            perm = choices[state].get((condition, rep, 'permuted'))
            permutations[condition] = (mean(perm['candidate'] == c for c in regular)
                                       if perm and perm['valid'] else None)
        rows.append(dict(state=state, before=b, after=a, difference=a-b,
            uniform=mean(outcome(c, r)[1] for c in range(candidates) for r in range(repetitions)),
            before_repeat_disagreement=before_noise, after_repeat_disagreement=after_noise,
            before_after_disagreement=flip,
            excess_disagreement=flip-(before_noise+after_noise)/2,
            common_valid_pairs=len(common), common_valid_before_success=sum(x[0][1] for x in common),
            common_valid_after_success=sum(x[1][1] for x in common),
            invalid_continuations=sum(not outcomes[k]['valid'] for k in expected),
            permutation_agreement=permutations))
    metrics = ('before', 'after', 'difference', 'uniform', 'before_repeat_disagreement',
               'after_repeat_disagreement', 'before_after_disagreement', 'excess_disagreement')
    estimates = {k: mean(row[k] for row in rows) if rows else None for k in metrics}
    common_n = sum(row['common_valid_pairs'] for row in rows)
    common = {k: sum(row['common_valid_'+k+'_success'] for row in rows)/common_n if common_n else None
              for k in ('before', 'after')}
    common['pairs'] = common_n
    common['difference'] = common['after']-common['before'] if common_n else None
    intervals = {k: None for k in metrics}
    # Every draw resamples whole states, preserving all choices and continuations.
    if len(rows) >= 2 and bootstrap_samples:
        rng = random.Random(seed); draws = {k: [] for k in metrics}
        for _ in range(bootstrap_samples):
            sampled = rng.choices(rows, k=len(rows))
            for k in metrics: draws[k].append(mean(row[k] for row in sampled))
        for k, values in draws.items():
            values.sort()
            intervals[k] = [values[int(.025*(len(values)-1))], values[int(.975*(len(values)-1))]]
    return dict(states_with_continuation_records=len(states), analyzed_states=len(rows),
        excluded_states=exclusions, estimates=estimates, state_bootstrap_95_intervals=intervals,
        common_valid_selected_pairs=common, by_state=rows,
        estimator='Equal weight per state; average all three execution repetitions and all three teacher calls. '
                  'Invalid continuations count as zero in failure-inclusive estimates; missing teacher panels are excluded explicitly. '
                  'Permutation calls are controls only. No best-of-repetition selection.',
        uncertainty='Percentile bootstrap over states; fewer than two states has no interval.')


def paired_by_phase(records, judgments, state_phases, **kwargs):
    """Report repairs as separate descriptive views without changing primary data."""
    states={r['state'] for r in records}
    if states != set(state_phases) or any(not isinstance(v,str) or not v for v in state_phases.values()):
        raise ValueError('Every continuation state needs one explicit execution phase')
    result={}
    for phase in sorted(set(state_phases.values())):
        cohort={s for s in states if state_phases[s]==phase}
        rows=[r for r in records if r['state'] in cohort]
        choices=[r for r in judgments if r['state'] in cohort]
        analysis=paired_summary(rows,choices,**kwargs)
        result[phase]={k:v for k,v in analysis.items() if k not in ('by_state','excluded_states')}
        result[phase].update(continuations=len(rows),invalid_continuations=sum(not r['valid'] for r in rows),
            exclusion_counts=dict(Counter(r['reason'] for r in analysis['excluded_states'])))
    return result
