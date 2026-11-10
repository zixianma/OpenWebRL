"""Offline retry estimates from complete saved episodes; no model/browser calls.

Each row is {"success": bool, "costs": {metric: nonnegative finite number}}.
Costs must be additive, with the same named metrics across every row. Missing
costs are rejected, never replaced with zero. These are empirical expectations
over uniform draws without replacement, not per-task hard-budget guarantees.

Perfect-oracle stopping pays the entire first successful episode and stops
before the next episode. It assumes free, error-free success recognition;
neither oracle pass@k nor its stopping costs describe a deployed verifier.
"""
from collections.abc import Mapping
from itertools import permutations
import math


def _validate(rows, label):
    if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= 8:
        raise ValueError(f'{label} must contain 1 through 8 complete episodes')
    metrics = None
    for row in rows:
        if not isinstance(row, Mapping) or type(row.get('success')) is not bool:
            raise ValueError(f'{label} success must be a bool')
        costs = row.get('costs')
        if not isinstance(costs, Mapping) or not costs:
            raise ValueError(f'{label} requires nonempty costs')
        if any(not isinstance(name, str) or not name for name in costs):
            raise ValueError('Cost metric names must be nonempty strings')
        if metrics is None:
            metrics = set(costs)
        elif set(costs) != metrics:
            raise ValueError('Every episode must report the same cost metrics')
        for value in costs.values():
            try:
                valid = type(value) in (int, float) and value >= 0 and math.isfinite(value)
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError('Costs must be finite nonnegative numbers, excluding bool')
    return sorted(metrics)


def _weighted_costs(rows, metrics, weights):
    try:
        result = {name: math.fsum(row['costs'][name] * weight
                                 for row, weight in zip(rows, weights))
                  for name in metrics}
    except OverflowError as exc:
        raise ValueError('Expected cost exceeds finite floating-point range') from exc
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('Expected cost exceeds finite floating-point range')
    return result


def _summarize_retries(actor, k, metrics):
    n = len(actor)
    orders = math.perm(n, k)
    successes = 0
    paid_counts = [0] * n
    for order in permutations(range(n), k):
        for index in order:
            paid_counts[index] += 1
            if actor[index]['success']:
                successes += 1
                break
    # The run-all inclusion probability is k/n. Oracle-stop inclusion depends
    # on each episode's outcome, so retain its paired cost rather than using
    # mean episode cost times mean number of attempts.
    return dict(k=k, success=successes / orders,
                run_all_costs=_weighted_costs(actor, metrics, [k / n] * n),
                oracle_stop_costs=_weighted_costs(
                    actor, metrics, [count / orders for count in paid_counts]))


def summarize_retries(actor, k):
    """Average success and costs over all ordered k-subsets of 1–8 episodes."""
    metrics = _validate(actor, 'actor')
    if type(k) is not int or not 1 <= k <= len(actor):
        raise ValueError('k must be an integer from 1 through the actor count')
    return _summarize_retries(actor, k, metrics)


def summarize_task(actor, guided):
    """Return task-level rows suitable for paired task bootstrap upstream.

    ``actor`` contains a row for each k=1..n. ``guided`` is the expectation
    from uniformly choosing ONE saved guided repetition, never any-of-guided.
    Both input groups require 1–8 episodes and identical cost metric names.
    The caller owns cohort/protocol matching and failure-validity definitions.
    """
    metrics = _validate(actor, 'actor')
    if _validate(guided, 'guided') != metrics:
        raise ValueError('Actor and guided episodes must report the same cost metrics')
    m = len(guided)
    return dict(actor_attempts=len(actor), guided_repetitions=m,
                actor=[_summarize_retries(actor, k, metrics)
                       for k in range(1, len(actor) + 1)],
                guided=dict(success=sum(row['success'] for row in guided) / m,
                            costs=_weighted_costs(guided, metrics, [1 / m] * m)))
