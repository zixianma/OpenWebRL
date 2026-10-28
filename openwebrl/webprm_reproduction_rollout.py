"""Explicit prospective helper routing for the paired Reddit diagnostic.

This changes no historical campaign. The full contract is included in each new
worker identity, and the host broker sees and journals the transformed payload.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

MINI = 'gpt-4o-mini-2024-07-18'
CONDITIONS = ('released_helper_defaults', 'same_with_saved_submitted_write_rule')
WRITE_RULE = ('\n\nWRITE-TASK CHECK: Finding an edit link, opening a form, or filling fields '
              'does not complete a requested change. Require visible evidence that the change '
              'has actually been saved or submitted. Never hand the remaining UI work to the '
              'user. If only a pending form is visible, return action=null. '
              'For READ tasks, retain the original completion rules.\n')


def contract(condition):
    if condition not in CONDITIONS:
        raise ValueError('Unknown reproduction diagnostic condition')
    return dict(protocol='reddit_paired_helper_rollout_v1', condition=condition,
                auxiliary_model=MINI, auxiliary_temperature=1., auxiliary_top_p=.9,
                terminal_system_suffix=WRITE_RULE if condition == CONDITIONS[1] else '',
                paper_exact=False, historical_campaign_unchanged=True)


def validate(config):
    value = config.get('reproduction_diagnostic')
    if value is None:
        return None
    if (not isinstance(value, dict) or value != contract(value.get('condition'))
            or config.get('selectors', ['no_search']) != ['no_search']
            or config.get('selector', 'no_search') != 'no_search'):
        raise ValueError('Exact no-search diagnostic contract required')
    path = Path(__file__).resolve()
    if config.get('code_sha256', {}).get(str(path)) != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Diagnostic adapter must be included in the frozen worker code')
    return value


def adapt_request(config, request):
    value = validate(config)
    if value is None:
        return request

    def adapted(call_config, payload, journal, **kwargs):
        purpose = kwargs.get('purpose')
        if purpose in ('terminal', 'state_diff'):
            payload = deepcopy(payload)
            payload['model'] = value['auxiliary_model']
            # Enforce shared helper sampling, rather than silently repairing it.
            if payload.get('temperature') != 1. or payload.get('top_p') != .9:
                raise ValueError('Paired helper sampling changed')
            if purpose == 'terminal':
                messages = payload['messages']
                if len(messages) != 2 or messages[0]['role'] != 'system':
                    raise ValueError('Pinned terminal prompt shape changed')
                messages[0]['content'] += value['terminal_system_suffix']
        return request(call_config, payload, journal, **kwargs)

    adapted._webprm_broker_client = getattr(request, '_webprm_broker_client', False)
    adapted._webprm_budget_owner = getattr(request, '_webprm_budget_owner', None)
    return adapted
