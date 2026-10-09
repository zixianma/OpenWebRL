"""Checks specific to the full-set recipe and added official-SFT N=10 arm."""
import asyncio
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openwebrl.luna_qwen_eval import PROTOCOL, SFT_PROTOCOL
from openwebrl.luna_qwen_full_eval import (
    ACTORS, FAMILY_MODES, FullPolicy, LOCAL_SAMPLING, SELECTION_PROMPT_SHA256,
    protocol_for_family, sampling_for_family, validate_plan, verify_actor_files,
    verify_existing_records,
)
from openwebrl.luna_qwen_metrics import PRICING, digest, write_json


def family_plan(tmp_path, family='sft'):
    tasks = [f'online_mind2web/{i}' for i in range(300)]
    experiment = 'luna-qwen-full300-test'
    return dict(family=family, protocol=protocol_for_family(family), task_ids=tasks,
        schedule=[dict(task_id=task, modes=list(FAMILY_MODES[family])) for task in tasks],
        experiment_id=experiment, wandb_group=experiment, wandb_prefix=experiment + '-' + family,
        output=str(tmp_path / family), budget_root=str(tmp_path / 'budget'), pricing=dict(PRICING),
        selector_prompt_sha256=SELECTION_PROMPT_SHA256,
        limits=dict(luna_usd=100., luna_calls=10000, judge_usd=50., judge_calls=2000))


def validate(plan):
    return validate_plan(plan, dict(plan_sha256=digest(plan), output=plan['output']))


def test_full_protocol_isolated_from_pilot_and_luna_sampling_is_not_fabricated(tmp_path):
    for family in FAMILY_MODES:
        plan = family_plan(tmp_path, family)
        assert validate(plan) == family
        assert plan['protocol']['max_steps'] == 30
        assert plan['protocol']['judge'] == 'o4-mini-2025-04-16'
        assert plan['protocol']['judge_max_output_tokens'] == 4096
    sft = protocol_for_family('sft')
    assert sft['actor'] == 'OpenWebRL/OpenWebRL-4B-SFT'
    assert sft['actor_revision'] == ACTORS['sft'][1]
    assert sft['candidates'] == [5, 10]
    assert (sft['temperature'], sft['top_p'], sft['top_k']) == (1., .95, -1)
    luna = protocol_for_family('luna')
    assert luna['temperature'] is None and luna['top_p'] is None
    assert luna['selector'] is None and luna['concurrency_per_gpu'] == 0
    assert PROTOCOL['top_p'] == .9 and SFT_PROTOCOL['candidates'] == [5]


@pytest.mark.parametrize('mutation', [
    lambda p: p['protocol'].update(top_p=.9),
    lambda p: p['protocol'].update(actor_revision='project-trained-checkpoint'),
    lambda p: p['protocol'].update(judge='gpt-4.1'),
    lambda p: p['schedule'][0].update(modes=['sft_luna5', 'luna']),
    lambda p: p['schedule'][0].update(modes=['sft_luna5', 'sft_luna5']),
    lambda p: p['schedule'].pop(),
    lambda p: p['task_ids'].pop(),
    lambda p: p.update(wandb_group='luna-qwen-inference-20261004'),
    lambda p: p.update(selector_prompt_sha256='changed'),
])
def test_frozen_plan_rejects_recipe_drift_and_partial_or_cross_family_cohorts(tmp_path, mutation):
    plan = family_plan(tmp_path)
    mutation(plan)
    with pytest.raises(ValueError):
        validate(plan)


def test_config_must_pin_exact_family_plan_and_output(tmp_path):
    plan = family_plan(tmp_path)
    with pytest.raises(ValueError, match='plan changed'):
        validate_plan(plan, dict(plan_sha256='old', output=plan['output']))
    with pytest.raises(ValueError, match='output differs'):
        validate_plan(plan, dict(plan_sha256=digest(plan), output=str(tmp_path / 'pilot')))


def test_only_official_sft_can_use_ten_candidates(tmp_path):
    plan = family_plan(tmp_path, 'qwen')
    for item in plan['schedule']:
        item['modes'].append('qwen_luna10')
    with pytest.raises(ValueError, match='combination'):
        validate(plan)
    assert protocol_for_family('qwen')['candidates'] == [1, 5]
    assert protocol_for_family('sft')['candidates'] == [5, 10]


def test_sft_n10_receives_diverse_decoding_and_executes_tenth_displayed_candidate(tmp_path):
    async def run():
        api = SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(output_text='{"selection":10}')))
        def builder(task, history, url, clusters, screenshot):
            assert len(clusters) == 10
            return [dict(role='user', content=task)]
        policy = FullPolicy('sft_luna10', tmp_path, api, builder, {}, {})
        policy.set_context([dict(role='user', content='goal')], [], None, 100)
        params = []
        async def propose(url, text, sampling, images, timeout):
            params.append(sampling)
            i = len(params) - 1
            return (f'thought{i}</think>ACTION{i}', [i], [-.1], 'stop')
        policy.qwen = propose
        kwargs = dict(infer=None, url='local', input_text='prompt',
            sampling_params=sampling_for_family('sft'), images=[],
            observation=dict(screenshot=b'image', active_tab_url='url'), history=[],
            task='goal', task_id='task', turn=0, timeout=10)
        result, metadata = await policy(**kwargs)
        saved = json.loads((tmp_path / 'decisions/000.json').read_text())
        selected = saved['display_order'][9]
        assert result[0] == f'thought{selected}</think>ACTION{selected}'
        assert metadata == dict(mode='sft_luna10', selected_index=selected)
        assert saved['mode'] == 'sft_luna10'
        assert len(params) == 10 and len({p['sampling_seed'] for p in params}) == 10
        assert all(all(p[k] == v for k, v in LOCAL_SAMPLING.items()) for p in params)
        assert 'temperature' not in api.create.call_args.kwargs
        assert 'top_p' not in api.create.call_args.kwargs
        # Both internal clamping and equivalent framework clamping preserve
        # the strict input+output <32K SGLang boundary.
        policy.set_context([dict(role='user', content='goal')], [], None, 32000)
        params.clear()
        kwargs['turn'] = 1
        kwargs['sampling_params']['max_new_tokens'] = 767
        await policy(**kwargs)
        assert len(params) == 10 and all(p['max_new_tokens'] == 767 for p in params)
        kwargs['sampling_params']['max_new_tokens'] = 100
        with pytest.raises(ValueError, match='decoding differs'):
            await policy(**kwargs)
        kwargs['sampling_params']['max_new_tokens'] = 4096
        kwargs['sampling_params']['top_p'] = .9
        with pytest.raises(ValueError, match='decoding differs'):
            await policy(**kwargs)
        assert len(params) == 10  # A rejected recipe cannot spend on proposals.
    asyncio.run(run())


def test_verified_weight_fingerprints_and_processor_cannot_drift(tmp_path):
    plan = family_plan(tmp_path)
    actor = tmp_path / 'actor'
    actor.mkdir()
    plan.update(actor_path=str(actor), actor_revision=ACTORS['sft'][1])
    for name, key in (('config.json', 'actor_config_sha256'),
                      ('preprocessor_config.json', 'actor_preprocessor_sha256')):
        path = actor / name
        path.write_text('{}')
        plan[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    weights = actor / 'model.safetensors'
    weights.write_bytes(b'verified')
    status = weights.stat()
    plan['actor_files'] = {weights.name: dict(size=status.st_size, mtime_ns=status.st_mtime_ns)}
    assert verify_actor_files(plan) == ({}, {})
    weights.write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='weights changed'):
        verify_actor_files(plan)
    # Luna uses only the pinned CPU tokenizer/processor; it never requires or
    # reads the local actor's weight files to produce an API action.
    plan['family'] = 'luna'
    assert verify_actor_files(plan) == ({}, {})
    (actor / 'preprocessor_config.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='configuration changed'):
        verify_actor_files(plan)


def test_completed_task_records_are_verified_even_when_claim_would_skip_them(tmp_path):
    plan = family_plan(tmp_path)
    plan_hash = digest(plan)
    task = plan['task_ids'][0]
    artifact = tmp_path / 'result.json'
    artifact.write_text('{"reward":1}')
    for mode in FAMILY_MODES['sft']:
        write_json(tmp_path / 'sft/records' / (digest([task, mode])[:24] + '.json'),
            dict(task_id=task, mode=mode, experiment_id=plan['experiment_id'], family='sft',
                 plan_sha256=plan_hash, protocol_sha256=digest(plan['protocol']),
                 artifact=str(artifact), artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()))
    verify_existing_records(plan, plan_hash)
    artifact.write_text('{"reward":0}')
    with pytest.raises(ValueError, match='Saved result changed'):
        verify_existing_records(plan, plan_hash)
