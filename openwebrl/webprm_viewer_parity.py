"""Offline access to pinned viewer filtering and history formatting.

These helpers are prospective diagnostics. Importing this module changes no
worker, frozen protocol, saved action, native verdict, or provider request.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

from .webprm_source import ARBITER_REVISION, _extract, source_manifest

CONDITIONS = ('unchanged', 'filter_only', 'history_only', 'filter_and_history')


def contract(condition, root):
    if condition not in CONDITIONS:
        raise ValueError('Unknown viewer parity condition')
    return dict(protocol='viewer_filter_history_diagnostic_v1', condition=condition,
                filter_policy_candidates=condition in ('filter_only', 'filter_and_history'),
                enrich_history=condition in ('history_only', 'filter_and_history'),
                helpers=load_viewer_helpers(root).manifest,
                history_input='action_time_flat_AX_label_reconstruction',
                resampling=False, paper_exact=False, historical_outcomes_unchanged=True)


def validate(config):
    value = config.get('viewer_parity_diagnostic')
    if value is None:
        return None
    from .webprm_reproduction_rollout import CONDITIONS as STOPPING, contract as stopping_contract
    if (not isinstance(value, dict) or value != contract(value.get('condition'), config['arbiter_source'])
            or config.get('reproduction_diagnostic') != stopping_contract(STOPPING[1])
            or config.get('selector', 'no_search') != 'no_search'
            or config.get('selectors', ['no_search']) != ['no_search']):
        raise ValueError('Exact no-search parity diagnostic with fixed stopping rule required')
    path = Path(__file__).resolve()
    if config.get('code_sha256', {}).get(str(path)) != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Parity diagnostic module must be frozen')
    return value


def load_viewer_helpers(root):
    root = Path(root)
    files = ['viewer/src/server/core/tools/sampling.py',
             'viewer/src/server/service/utils/trajectory_utils.py']
    manifest = source_manifest(root, files, ARBITER_REVISION)
    namespace = dict(re=re, Any=Any, BlockInfo=SimpleNamespace,
                     logger=logging.getLogger(__name__))
    _extract(root / files[0], namespace,
             names={'extract_elements', 'validate_candidate', 'filter_candidates'})
    _extract(root / files[1], namespace, names={
        '_extract_action_from_step_data', '_build_node_map_from_dict',
        '_build_node_map_from_list', '_build_node_map',
        'enrich_action_with_nodes', 'get_trajectory_text'})
    return SimpleNamespace(manifest=manifest, **{
        name: namespace[name] for name in ('extract_elements', 'validate_candidate',
            'filter_candidates', 'enrich_action_with_nodes', 'get_trajectory_text')})


def history_from_saved_observations(helpers, history, observations):
    """Apply upstream formatting to action-time observations, never current IDs.

    The saved flattened AX text supplies node labels; upstream's original
    obs_nodes_info dictionary is not saved, so this is an explicitly disclosed
    reconstruction of that input boundary, not byte-identical viewer history.
    """
    if len(history) != len(observations):
        raise ValueError('One actual pre-action observation per history entry required')
    steps = []
    for index, (row, observation) in enumerate(zip(history, observations)):
        nodes = {}
        for line in observation['text'].splitlines():
            match = re.match(r'\s*\[(\d+)\]\s+(.+)', line)
            if match:
                nodes[match[1]] = dict(text=match[2])
        steps.append(dict(step=index, selected_action=dict(
            thought=row['thought'], action=row['action']), obs_nodes_info=nodes))
    class Repository:
        def get_steps(self, *, run_id, start, limit):
            return steps[start:start + limit]
    return helpers.get_trajectory_text(Repository(), 'offline', len(steps))
