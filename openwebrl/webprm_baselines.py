"""Offline preparation and selection primitives for supported-interface WebPRMs.

No model calls or browser actions occur here. Upstream prompt/scoring parity and
browser execution must be verified before the prepared cohort can be launched.
"""
from collections import Counter
import hashlib
import math
import random


ARMS = ('shepherd_unguided', 'shepherd_random5', 'shepherd8b',
        'arbiter_unguided', 'arbiter_random5', 'arbiter7b')
COMPARISONS = (('shepherd8b', 'shepherd_unguided'), ('shepherd8b', 'shepherd_random5'),
               ('arbiter7b', 'arbiter_unguided'), ('arbiter7b', 'arbiter_random5'))


def paired_summary(records, task_ids, resamples=10000, seed=20261009):
    """Strict complete-cohort analysis; invalid attempts retain the denominator.

    Each record contains task_id, arm, success (0/1), and valid (boolean).
    Retries must be resolved by the audited collector, never selected here by
    whichever attempt succeeded. Missing records cannot silently become failures.
    """
    ids = list(task_ids)
    if not ids or len(set(ids)) != len(ids) or resamples < 100:
        raise ValueError('Unique nonempty cohort and at least 100 bootstrap draws required')
    indexed = {}
    for row in records:
        if row.get('attempted', True) is not True:
            raise ValueError('Unattempted cells cannot complete a paired cohort')
        key = row['task_id'], row['arm']
        if key in indexed or key[0] not in ids or key[1] not in ARMS:
            raise ValueError('Duplicate or out-of-cohort result')
        if type(row['valid']) is not bool or type(row['success']) is not int or row['success'] not in (0, 1):
            raise ValueError('Invalid result types')
        if row['success'] and not row['valid']:
            raise ValueError('An invalid trajectory cannot count as success')
        indexed[key] = row
    if set(indexed) != {(i, a) for i in ids for a in ARMS}:
        raise ValueError('Incomplete paired cohort')
    n = len(ids)
    methods = {}
    for arm in ARMS:
        rows = [indexed[i, arm] for i in ids]
        successes = sum(r['success'] for r in rows)
        valid = sum(r['valid'] for r in rows)
        methods[arm] = dict(tasks=n, successes=successes, valid=valid,
            invalid=n-valid, overall_percent=100*successes/n,
            valid_only_percent=100*successes/valid if valid else None)
    comparisons = []
    rng = random.Random(seed)
    for guided, control in COMPARISONS:
            differences = [indexed[i, guided]['success']-indexed[i, control]['success'] for i in ids]
            delta = 100*sum(differences)/n
            bootstrap = sorted(100*sum(differences[rng.randrange(n)] for _ in ids)/n
                               for _ in range(resamples))
            common = [i for i in ids if indexed[i, guided]['valid'] and indexed[i, control]['valid']]
            wins, losses = differences.count(1), differences.count(-1)
            discordant = wins + losses
            p = min(1., 2*sum(math.comb(discordant, k) for k in range(min(wins, losses)+1))/2**discordant)
            baseline = methods[control]['overall_percent']
            comparisons.append(dict(guided=guided, control=control, tasks=n,
                delta_pp=delta, ci95_pp=[bootstrap[int(.025*(resamples-1))], bootstrap[int(.975*(resamples-1))]],
                interval='paired task percentile bootstrap', wins=wins, losses=losses,
                exact_mcnemar_p=p, relative_gain_percent=100*delta/baseline if baseline else None,
                common_valid_tasks=len(common),
                common_valid_delta_pp=100*sum(indexed[i, guided]['success']-indexed[i, control]['success']
                                             for i in common)/len(common) if common else None))
    return dict(methods=methods, comparisons=comparisons, bootstrap_draws=resamples,
                seed=seed, claim='within-harness selection gain; not isolated cross-harness reward quality')


def candidate_pool(actions, limit=5):
    """Top-frequency exact parsed actions, with first occurrence breaking ties.

    The caller must parse upstream responses first. Do not normalize quoted text
    or element IDs: doing so can merge actions with different effects.
    """
    if not actions or type(limit) is not int or limit < 1:
        raise ValueError('Nonempty parsed action samples and positive limit required')
    if any(not isinstance(a, str) or not a.strip() for a in actions):
        raise ValueError('Invalid parsed action')
    counts = Counter(actions)
    return [dict(action=a, frequency=counts[a], first_sample=actions.index(a))
            for a in sorted(counts, key=lambda a: (-counts[a], actions.index(a)))[:limit]]


def scalar_select(candidates, score):
    """Use upstream scores; ties favor frequency then original sample order."""
    if not candidates:
        raise ValueError('No candidates')
    scored = []
    for c in candidates:
        value = score(c)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Invalid reward score; never silently execute a fallback')
        scored.append((value, c['frequency'], -c['first_sample'], c))
    return max(scored, key=lambda r: r[:3])[3], [r[0] for r in scored]


def tournament(candidates, compare, seed):
    """Seeded adjacent-pair rounds with byes, saving every verdict.

    This is our explicit tournament implementation, pending upstream parity.
    compare(left, right) must return 0 or 1; malformed verdicts fail closed.
    """
    if not candidates:
        raise ValueError('No candidates')
    current = list(candidates)
    random.Random(seed).shuffle(current)
    trace = []
    while len(current) > 1:
        following = []
        for i in range(0, len(current), 2):
            if i + 1 == len(current):
                following.append(current[i])
                continue
            pair = current[i:i+2]
            winner = compare(*pair)
            if type(winner) is not int or winner not in (0, 1):
                raise ValueError('Invalid pairwise verdict')
            trace.append(dict(left=pair[0], right=pair[1], winner=winner))
            following.append(pair[winner])
        current = following
    return current[0], trace


def prepare_plan(tasks, task_file_sha256, pilot_tasks=12, seed=20261009):
    if len(tasks) != 300 or len({t['task_id'] for t in tasks}) != 300:
        raise ValueError('Expected the complete unique 300-task OM2W cohort')
    if type(pilot_tasks) is not int or not 1 <= pilot_tasks <= 300:
        raise ValueError('Invalid pilot size')
    ids = sorted(t['task_id'] for t in tasks)
    random.Random(seed).shuffle(ids)
    pilot = ids[:pilot_tasks]
    return dict(schema_version=1, status='prepared_not_launchable',
        benchmark='Online-Mind2Web', task_file_sha256=task_file_sha256,
        cohort_sha256=hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),
        task_count=300, pilot_task_ids=pilot, pilot_selection_seed=seed,
        arms=list(ARMS), paired_unit='task', training_updates=0,
        actor_model='gpt-4o-mini-2024-07-18',
        harness='BrowserGym accessibility-tree / element-ID actions',
        sampling=dict(shepherd=dict(temperature=1.0, raw_samples_guided=20,
                                   candidate_limit=5, raw_samples_unguided=1,
                                   top_p='not specified by paper; record chosen value'),
                      arbiter=dict(temperature=1.0, top_p=.9,
                                   generation='native viewer one response containing five BLOCKS',
                                   unguided='first valid native BLOCK', candidate_limit=5)),
        models=dict(shepherd8b='verify official released repository and immutable revision',
                    arbiter7b='ZYao720/WebArbiter-7B; pin immutable revision'),
        judge=dict(model='o4-mini', prompt='agenttrek', max_completion_tokens=4096),
        max_turns=30, browser='local', wandb_project='openwebrl-evals',
        pilot_episodes=pilot_tasks*len(ARMS),
        proposal=dict(gpus=1, gpu_type='H200', cpus=8, memory_gib=120,
                      wall_hours=2, total_gpu_hours=2, api_usd_cap=30,
                      includes_retries=True, full300_authorized=False),
        prerequisites=[
            'Acquire and hash upstream source, prompt templates and model revisions',
            'Verify BrowserGym observation and action execution parity on public fixture',
            'Verify Shepherd checklist generation and label-token probability scoring',
            'Verify Arbiter prompt, answer parser and tournament scheduling',
            'Measure context truncation, serving memory and per-decision cost',
            'Implement durable shared API/GPU ledger and fail-closed reservations',
            'Verify saved terminal screenshot, rollout and independent judge verdict',
            'Obtain exact pilot compute/API approval; submit only after preflight passes'],
        analysis=dict(primary='guided minus unguided overall success, percentage points',
                      selector_control='guided minus random5 overall success',
                      invalid_policy='retain denominator; report validity separately',
                      uncertainty='paired task bootstrap; common-valid sensitivity',
                      cost='all raw candidate tokens, verifier tokens, API cost and GPU time',
                      historical_comparison='published WebArena results remain separate'))
