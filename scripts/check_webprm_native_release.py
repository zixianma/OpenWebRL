#!/usr/bin/env python3
"""Zero-provider-call fixture exercising whole released sampler/reward/tournament."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
from pathlib import Path

from openwebrl.webprm_native_release import load_tools, reference


async def fixture(source):
    tools, refs = load_tools(source)
    engine = tools['engine'].LMEngine
    original = engine.generate_text_async
    calls = []
    blocks = [
        dict(thought='First valid action', action='click [1]', confidence=.4),
        dict(thought='Nonexistent element', action='click [999]', confidence=.99),
        dict(thought='Duplicate with higher confidence', action='click [1]', confidence=.9),
        dict(thought='Second valid action', action='click [2]', confidence=.8),
        dict(thought='Third valid action', action='type [3] example', confidence=.7),
    ]

    async def saved_transport(self, system, user, model=None):
        spec = model or self._default_model
        calls.append({'model': spec.model, 'system': system, 'user': user})
        if spec.model == 'offline-policy':
            return json.dumps({'BLOCKS': blocks}), {}
        if spec.model == 'offline-critic':
            return '<Answer>Response 2</Answer>', {}
        raise AssertionError('Unexpected fixture model; network access forbidden')

    engine.generate_text_async = saved_transport
    try:
        sampled = await tools['sampling'].sample_candidates(
            intent='Choose an action', observation_text='[1] button First\n[2] button Second\n[3] textbox Input',
            current_url='http://offline.invalid/', model_name='offline-policy')
        assert sampled['success'], sampled
        assert sampled['round_info'] == {
            'generated_count': 5, 'filtered_count': 4, 'final_count': 3}, sampled
        candidates = sampled['candidates']
        assert [x['action'] for x in candidates] == ['click [1]', 'click [2]', 'type [3] example']
        assert candidates[0]['frequency'] == 2 and candidates[0]['confidence'] == .9
        generator = tools['reward'].RewardGenerator(
            tools['engine'].ModelSpec(provider='offline', model='offline-critic'))
        runner = tools['tournament'].TournamentRunner(reward_generator=generator)
        block_type = tools['sampling'].BlockInfo
        result = await runner(tools['tournament'].TournamentCommand(
            intent='Choose an action', observation_text='[1] button First\n[2] button Second\n[3] textbox Input',
            candidates=[block_type(**x) for x in candidates], current_url='http://offline.invalid/'))
        assert result.final_winner.action == 'click [2]', result
        assert result.total_comparisons == 2 and len(calls) == 3
        return {
            'verified': True, 'scope': 'Native whole-module business-logic fixture, not a live task or paper reproduction',
            'source_refs': refs, 'generation': sampled['round_info'],
            'duplicate_frequency': candidates[0]['frequency'],
            'final_action': result.final_winner.action,
            'native_comparisons': result.total_comparisons,
            'synthetic_transport_calls': len(calls), 'provider_calls': 0,
            'source_changes': 0,
            'operational_adaptations': ['Namespace-only packaging shim', 'Injected synthetic model transport'],
        }
    finally:
        engine.generate_text_async = original


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    # A mistaken call through the original engine cannot reach a provider.
    def deny_connection(*args, **kwargs):
        raise RuntimeError('Network forbidden in native release fixture')
    socket.create_connection = deny_connection
    original_connect = socket.socket.connect
    def guarded_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            deny_connection()
        return original_connect(sock, address)
    socket.socket.connect = guarded_connect
    result = asyncio.run(fixture(args.source))
    result['fixture'] = reference(__file__)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ['verified', 'generation', 'native_comparisons', 'provider_calls']}))


if __name__ == '__main__':
    main()
