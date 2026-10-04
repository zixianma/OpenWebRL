"""Independent failure-coverage and failure-strength knobs; legacy defaults persist."""
def failure_beta(config):
    value = config.get('failure_beta', .5)
    if value not in (.5, 1.):
        raise ValueError('Failure beta must be 0.5 or 1.0')
    if value != .5 and config.get('failure_ablation') != 'weight':
        raise ValueError('Failure beta=1 requires the separate weight ablation')
    return value


def coverage_budget(config):
    budget = config.get('failure_turn_budget', 0)
    if type(budget) is not int or budget not in (0, 4):
        raise ValueError('Failure coverage budget must be zero or four')
    if budget and config.get('failure_ablation') != 'coverage':
        raise ValueError('Four-turn coverage requires its explicit ablation')
    return budget


def failure_fraction(config):
    value = config.get('failure_scored_fraction', .2)
    if value not in (.2, .4) or (value == .4 and config.get('failure_ablation') != 'coverage'):
        raise ValueError('Failure sampling=0.4 requires the coverage ablation')
    return value


def deferred_enabled(config):
    return bool(coverage_budget(config) or failure_fraction(config) == .4)


def auxiliary_window_limit(config):
    # Serial microbatch-one accumulation; no change to the optimizer or loss scale.
    return 128 if failure_fraction(config) == .4 else 32


def validate_recipe(config):
    beta, budget = failure_beta(config), coverage_budget(config)
    fraction = failure_fraction(config)
    variant = config.get('failure_ablation')
    if variant is None:
        return
    expected = {'control': (.5, 0), 'coverage': (.5, 4), 'weight': (1., 0)}
    if variant == 'coverage' and fraction == .4:
        expected['coverage'] = (.5, 0)
    if variant not in expected or (beta, budget) != expected[variant]:
        raise ValueError('Coverage and weight must remain separate treatments')
    if (config.get('beta') != .5 or config.get('scored_fraction') != .2
            or not config.get('additive_failure_groups')
            or config.get('failure_group_cap') != 8
            or config.get('failure_loss_coefficient') != 1/6
            or config.get('candidate_gate', 'distinct5') != 'distinct5'
            or config.get('credit_assignment', 'response_index') != 'response_index'):
        raise ValueError('Preserve the original additive mixed-group recipe and failure normalization')
