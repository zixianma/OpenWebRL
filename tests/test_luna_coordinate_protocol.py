"""Native Luna pixels must reach Playwright unchanged; local actors normalize."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openwebrl.luna_qwen_policy import Policy
from openwebrl.luna_qwen_full_eval import FullPolicy, sampling_for_family


def fake_browser_environment(config):
    from openwebrl.env.web_env import WebEnv
    env = WebEnv(start_url='https://example.test', tool_list=[], policy='',
        **{key: config[key] for key in ('width', 'height', 'dpr', 'max_retries',
            'wait_timeout', 'screenshot_timeout', 'resize_output_coords', 'resize_scale', 'image_patch_size')})
    env.page = SimpleNamespace(url='https://example.test',
        mouse=SimpleNamespace(click=AsyncMock(), move=AsyncMock(), down=AsyncMock(), up=AsyncMock()),
        wait_for_load_state=AsyncMock(), wait_for_timeout=AsyncMock(),
        evaluate=AsyncMock(return_value=dict(x=0, y=0)))
    env.context = SimpleNamespace(pages=[env.page])
    env._get_element_description = AsyncMock(return_value='target')
    return env


def mock_resources(monkeypatch):
    from openwebrl import generate_browser as generation
    from openwebrl.base.registry import EnvRegistry
    created = []
    async def create(task_id, config, metadata, **kwargs):
        created.append(deepcopy(config))
        return fake_browser_environment(config), dict(task_id=task_id)
    monkeypatch.setattr(generation, '_create_env', create)
    monkeypatch.setattr(generation, 'GenerateState', lambda args: None)
    monkeypatch.setattr(generation, 'ENV_REGISTRY', EnvRegistry())
    for name in ('_apply_browser_env_mode_override', '_apply_sandbox_env_overrides',
                 '_apply_local_process_env_overrides'):
        monkeypatch.setattr(generation, name, lambda config: config)
    # The production frozen source pins this viewport in its config.yaml;
    # the shared checkout's general browser defaults serve other experiments.
    monkeypatch.setattr(generation, '_apply_local_process_env_overrides',
                        lambda config: dict(config, width=1280, height=720, dpr=1))
    return generation, created


async def initialize(generation, policy):
    args = SimpleNamespace(browser_action_selector=policy, sglang_router_ip='unused', sglang_router_port=0)
    env, adapter, _, _, _ = await generation._initialize_resources(args, 'test-task')
    return env, adapter


def invoke(policy, turn=0):
    return policy(infer=None, url='unused', input_text='unused', sampling_params=sampling_for_family('qwen'),
        images=[], observation=dict(screenshot=b'image', screen_size=[1280, 720], active_tab_url='https://example.test'),
        history=[], task='Find the target', task_id='test-task', turn=turn, timeout=10)


@pytest.mark.parametrize('name,arguments', [
    ('click', dict(point_2d=[357, 414])),
    ('hover', dict(point_2d=[1279, 719])),
    ('scroll', dict(point_2d=[357, 414], direction='down', amount=.5)),
    ('drag', dict(start_point_2d=[50, 100], end_point_2d=[1250, 700], steps=7)),
])
def test_luna_native_pixels_reach_real_executor_and_history_unchanged(tmp_path, monkeypatch, name, arguments):
    generation, created = mock_resources(monkeypatch)
    async def run():
        native = dict(name=name, arguments=arguments)
        call = dict(type='function_call', call_id='call_one', name=name, arguments=json.dumps(arguments))
        api = SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(output=[call])))
        policy = Policy('luna', tmp_path, api, None, {}, {})
        tools = [dict(type='function', function=dict(name=name, description='Use viewport pixel coordinates',
                                                   parameters=dict(type='object')))]
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()))
        messages = [dict(role='system', content='Browser policy'), dict(role='user', content=[
            dict(type='text', text='Goal: Find the target. Current URL: https://example.test'),
            dict(type='image_url', image_url='data:image/png;base64,CURRENT')])]
        policy.set_context(messages, tools, tokenizer, 100)
        env, adapter = await initialize(generation, policy)
        assert created[-1]['resize_output_coords'] is False
        assert adapter.config.resize_output_coords is False
        assert env.smart_resize_height is None and env.smart_resize_width is None
        output, metadata = await invoke(policy)
        parsed = adapter.tool_parser.parse(output[0])
        actions = adapter.actions_from_parsed(parsed.calls)
        assert actions == [dict(name=name, args=arguments)]
        ok, feedback = await env.execute_single_action(actions[0])
        assert ok, feedback
        if name == 'click':
            assert env.page.mouse.click.call_args.args == (357, 414)
        elif name in ('hover', 'scroll'):
            assert env.page.mouse.move.call_args.args == tuple(arguments['point_2d'])
        else:
            moves = env.page.mouse.move.call_args_list
            assert moves[0].kwargs == dict(x=50., y=100.)
            assert moves[1].kwargs == dict(x=1250., y=700., steps=7)
        assert metadata['browser_coordinate_space'] == 'viewport_pixels'
        # The following API turn sees the native pixel history together with
        # unchanged goal, URL, and current screenshot. No normalized rewrite.
        history = dict(role='assistant', content='<think>' + output[0])
        next_observation = deepcopy(messages[-1]['content'])
        next_observation[0]['text'] = '<tool_response>\n' + feedback + '\nCurrent URL: https://example.test\n</tool_response>'
        next_messages = [*messages, history, dict(role='user', content=next_observation)]
        policy.set_context(next_messages, tools, tokenizer, 200)
        api.create.return_value.output = [dict(call, call_id='call_two')]
        await invoke(policy, turn=1)
        request = api.create.call_args.kwargs
        assert request['input'][2] == call
        assert request['input'][3]['call_id'] == call['call_id']
        assert request['input'][3]['type'] == 'function_call_output'
        assert feedback in request['input'][3]['output'][0]['text']
        assert 'Find the target' in request['input'][1]['content'][0]['text']
        assert 'https://example.test' in request['input'][1]['content'][0]['text']
        assert request['input'][-1]['output'][1] == dict(type='input_image',
            image_url='data:image/png;base64,CURRENT', detail='high')
        assert request['tools'][0]['description'] == 'Use viewport pixel coordinates'
        assert messages[1]['content'][1]['type'] == 'image_url'
        # A later local-actor episode in the same process must restore defaults.
        local_policy = Policy('qwen', tmp_path / 'local', None, None, {}, {})
        local_env, local_adapter = await initialize(generation, local_policy)
        assert created[-1]['resize_output_coords'] is True
        assert local_adapter.config.resize_output_coords is True
        assert local_env.smart_resize_width == 1000
    asyncio.run(run())


@pytest.mark.parametrize('mode', ['qwen', 'qwen_luna5', 'qwen_luna10', 'sft_luna5', 'sft_luna10'])
def test_qwen_and_sft_candidates_remain_normalized_and_selected_action_unchanged(tmp_path, monkeypatch, mode):
    generation, created = mock_resources(monkeypatch)
    async def run():
        api = SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(output_text='{"selection":1}')))
        native = dict(name='click', arguments=dict(point_2d=[250, 500]))
        action = '<tool_call>' + json.dumps(native) + '</tool_call>'
        def builder(task, history, url, clusters, screenshot):
            assert all(c['rep'].molmo_action == action for c in clusters)
            return [dict(role='user', content='normalized [0,1000] ' + action)]
        policy_class = FullPolicy if mode == 'sft_luna10' else Policy
        policy = policy_class(mode, tmp_path, api, builder, {}, {})
        policy.set_context([dict(role='user', content='goal')], [], None, 100)
        policy.qwen = AsyncMock(return_value=('thought</think>' + action, [1], [-.1], 'stop'))
        env, adapter = await initialize(generation, policy)
        assert created[-1]['resize_output_coords'] is True
        output, metadata = await invoke(policy)
        assert output[0] == 'thought</think>' + action
        assert metadata['mode'] == mode
        parsed = adapter.tool_parser.parse(output[0])
        actions = adapter.actions_from_parsed(parsed.calls)
        ok, feedback = await env.execute_single_action(actions[0])
        assert ok, feedback
        assert env.page.mouse.click.call_args.args == (320, 360)
        if mode == 'qwen':
            api.create.assert_not_called()
        else:
            assert api.create.call_args.args == ('selector',)
            assert json.dumps(native) in api.create.call_args.kwargs['input'][0]['content']
    asyncio.run(run())
