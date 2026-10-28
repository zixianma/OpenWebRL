"""Owned synthetic BrowserGym check; no benchmark, provider, or website calls.

The campaign invokes this in a fresh user/network namespace before launching
models or fixed tasks. Offline tests inject the environment; importing this
module never starts a browser. The existing episode watchdog owns all children.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.parse import quote

from .webprm_arbiter_harness import (load_arbiter_harness,
    load_native_observation_processor, native_execution_code)
from .webprm_lifecycle import (EPISODE_WATCHDOG, mark_episode_phase,
    run_supervised_episode, save_outcome_and_close)
from .webprm_worker import CHROMIUM, prepare_browser

VIEWPORT = dict(width=1280, height=720)
LABEL = 'Synthetic native click'
HTML = '''<!doctype html><meta charset="utf-8"><title>Synthetic browser preflight</title>
<p id="state">Synthetic fixture ready</p>
<button style="position:absolute;left:1050px;top:610px;width:190px;height:60px"
onclick="document.body.dataset.clicked='yes';document.getElementById('state').textContent='Synthetic click observed'">
Synthetic native click</button>'''


def safe_path(value):
    path = Path(value)
    roots = (Path('/tmp'), Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'))
    if (not path.is_absolute() or path != path.resolve() or path.is_symlink()
            or not any(root in path.parents for root in roots)):
        raise ValueError('Preflight evidence must use a canonical approved path')
    return path


def write_new(path, value):
    path = safe_path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def _deadline(value):
    if type(value) not in (float, int) or not math.isfinite(value) or value <= time.time():
        raise ValueError('A future finite preflight deadline is required')


def _check_observation(observation):
    if not isinstance(observation, dict) or observation.get('last_action_error'):
        raise RuntimeError('BrowserGym observation/action contains an error')
    # BrowserGym's screenshot is a NumPy array, not directly JSON serializable.
    screenshot = observation.get('screenshot')
    if tuple(getattr(screenshot, 'shape', ())) != (720, 1280, 3):
        raise RuntimeError('BrowserGym screenshot does not match the native viewport')
    if str(getattr(screenshot, 'dtype', None)) != 'uint8':
        raise RuntimeError('Unexpected BrowserGym screenshot dtype')


def _exercise(env, processor, native, source, folder, *, deadline, terminal_answer_check=False):
    """Run one fixed action with injected dependencies; no completion clients."""
    _deadline(deadline)
    observation, _ = env.reset(seed=42)
    _check_observation(observation)
    if env.page.viewport_size != VIEWPORT or env.task.viewport != VIEWPORT:
        raise RuntimeError('Browser viewport changed')
    text, bounds = processor.observe(env.page)
    matches = re.findall(r"\[([^\]]+)\]\s+button\s+'Synthetic native click'", text)
    if len(matches) != 1 or matches[0] not in bounds:
        raise RuntimeError('Native observation omitted the unique synthetic target')
    target = matches[0]
    box = bounds[target]
    if (len(box) != 4 or any(type(value) not in (int, float) or not math.isfinite(value) for value in box)
            or box[0] < 1000 or box[1] < 600 or box[2] <= 0 or box[3] <= 0
            or box[0] + box[2] > 1280 or box[1] + box[3] > 720):
        raise RuntimeError('Native target geometry differs from the fixed bottom-right fixture')
    if env.page.evaluate('document.body.dataset.clicked') is not None:
        raise RuntimeError('Synthetic fixture is not in its initial state')
    png = env.page.screenshot(type='png')
    if not isinstance(png, bytes) or not png.startswith(b'\x89PNG\r\n\x1a\n'):
        raise RuntimeError('Missing preaction PNG')
    (folder / 'preaction.png').write_bytes(png)
    action = f'click [{target}]'
    code = native_execution_code(source, native.parse_action(action), bounds,
                                 key_mapping=native.key_mapping)
    def mapping(requested):
        if requested != action:
            raise ValueError('Only the prespecified synthetic click is permitted')
        return code
    env.action_mapping = mapping
    write_new(folder / 'preaction.json', dict(text=text, element_bounds=deepcopy(bounds),
        viewport=VIEWPORT, geometry=deepcopy(processor.last_geometry), action=action,
        execution_code=code, execution_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        screenshot_sha256=hashlib.sha256(png).hexdigest()))
    _deadline(deadline)
    observation, _, terminated, truncated, _ = env.step(action)
    _check_observation(observation)
    if terminated or truncated:
        raise RuntimeError('Synthetic task terminated unexpectedly')
    after_text, after_bounds = processor.observe(env.page)
    clicked = env.page.evaluate('document.body.dataset.clicked')
    if clicked != 'yes' or 'Synthetic click observed' not in after_text:
        raise RuntimeError('Native click did not produce the expected observable DOM change')
    write_new(folder / 'postaction.json', dict(text=after_text, element_bounds=deepcopy(after_bounds),
        geometry=deepcopy(processor.last_geometry), clicked=clicked,
        screenshot_shape=list(observation['screenshot'].shape), screenshot_dtype=str(observation['screenshot'].dtype)))
    result = dict(native_target_id=target, native_click_observed=True, viewport=VIEWPORT,
        browsergym_screenshot_shape=list(observation['screenshot'].shape),
        observation_manifest=processor.manifest, action_manifest=native.manifest)
    if terminal_answer_check:
        from .webprm_webarena import load_native_evaluator, ABSTENTION_EVALUATOR_PROTOCOL
        answer_action='send_msg_to_user [0]'
        answer_code=native_execution_code(source,native.parse_action(answer_action),after_bounds,
            key_mapping=native.key_mapping)
        def answer_mapping(requested):
            if requested!=answer_action:
                raise ValueError('Only the fixed synthetic final answer is permitted')
            return answer_code
        env.action_mapping=answer_mapping
        previous=deepcopy(observation.get('chat_messages',[]))
        final,_,_,_,_=env.step(answer_action)
        _check_observation(final)
        actual=final.get('chat_messages',[])
        appended=actual[len(previous):]
        if (actual[:len(previous)]!=previous or len(appended)!=1
                or appended[0].get('role')!='assistant' or appended[0].get('message')!='0'):
            raise RuntimeError('Synthetic normal final answer did not append its exact literal value')
        trajectory=[dict(observation=dict(text=after_text),info={}),
            dict(action_type='send_msg_to_user',answer='0',raw_action=answer_action)]
        evaluator=load_native_evaluator(source,scoring_protocol=ABSTENTION_EVALUATOR_PROTOCOL)
        controls=[]
        for reference,expected in (('0',1),('1',0)):
            task=dict(intent='Return the displayed synthetic count.',eval=dict(eval_types=['string_match'],
                reference_answers=dict(exact_match=reference)))
            verdict=evaluator.evaluate(task,trajectory,env.page,termination='stop')
            if verdict['score']!=expected or verdict['model_provider_calls']!=0:
                raise RuntimeError('Synthetic normal final-answer positive/negative control failed')
            controls.append(dict(reference=reference,expected_score=expected,verdict=verdict))
        write_new(folder/'normal-final-answer.json',dict(action=answer_action,execution_code=answer_code,
            chat_messages_before=previous,chat_messages_after=actual,trajectory=trajectory,controls=controls,
            purpose='fixed_synthetic_execution_and_exact_scoring_not_actor_task_performance'))
        result.update(normal_final_answer_verified=True,normal_final_answer_control_count=2)
    return result


def _child(task, arm, config):
    folder = Path(config['output']) / task['task_id'] / arm
    result = dict(task_id=task['task_id'], arm=arm, identity_sha256=config['identity_sha256'],
        attempted=False, valid=False, success=0, scope='synthetic_browser_startup_only',
        benchmark_episodes=0, model_calls=0, provider_calls=0)
    env = processor = None
    try:
        _deadline(config['deadline'])
        prepare_browser()
        result.update(browser_runtime=os.environ['TMPDIR'],
            playwright_browsers_path=os.environ['PLAYWRIGHT_BROWSERS_PATH'],
            cache_environment={name:os.environ[name] for name in ('XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME')})
        from browsergym.core.env import BrowserEnv
        from browsergym.core.task import OpenEndedTask
        from playwright.sync_api import Browser
        from unittest.mock import patch
        native = load_arbiter_harness(config['arbiter_source'])
        blocked_requests = []
        original = Browser.new_context
        def offline_context(browser, *args, **kwargs):
            kwargs['offline'] = True
            context = original(browser, *args, **kwargs)
            def abort(route):
                blocked_requests.append(route.request.resource_type)
                route.abort()
            context.route('**/*', abort)
            return context
        result['attempted'] = True
        mark_episode_phase(config, 'synthetic_browser_startup')
        with patch.object(Browser, 'new_context', offline_context):
            env = BrowserEnv(task_entrypoint=OpenEndedTask,
                task_kwargs=dict(start_url='data:text/html,'+quote(HTML),goal='Synthetic startup check'),
                action_mapping=lambda _: (_ for _ in ()).throw(ValueError('Action not armed')),
                slow_mo=0, headless=True, pw_chromium_kwargs={'executable_path':CHROMIUM})
            processor = load_native_observation_processor(config['arbiter_source'], viewport_size=VIEWPORT)
            result.update(_exercise(env, processor, native, config['arbiter_source'], folder,
                                     deadline=config['deadline'],terminal_answer_check=config.get('terminal_answer_check',False)))
        result['blocked_request_count'] = len(blocked_requests)
        if blocked_requests:
            raise RuntimeError('Unexpected browser resource requests were blocked')
        result['valid'] = True
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error)[:2000])
    finally:
        # Save the outcome before either native-CDP release or browser cleanup
        # can stall; the watchdog retains it and kills only this owned tree.
        class Close:
            def close(self):
                if processor is not None:
                    processor.close()
                if env is not None:
                    env.close()
        save_outcome_and_close(result, folder, Close(), config)


def run_preflight(source, output, *, deadline, host_netns, terminal_answer_check=False):
    _deadline(deadline)
    current_netns = os.readlink('/proc/self/ns/net')
    if (not isinstance(host_netns, str) or not re.fullmatch(r'net:\[\d+\]', host_netns)
            or current_netns == host_netns):
        raise ValueError('Synthetic browser must run in a separate network namespace')
    folder = safe_path(output)
    if folder.exists():
        raise ValueError('Synthetic preflight output must be fresh')
    configured = os.environ.get('WEBPRM_BROWSER_TMP_ROOT')
    if configured is None or not safe_path(configured).is_dir() or not str(configured).startswith('/tmp/'):
        raise ValueError('Explicit existing campaign browser temporary root is required')
    manifest = load_arbiter_harness(source).manifest
    identity_data = dict(scope='synthetic_browser_startup_v1', source=manifest,
        browser_tmp_root=configured, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        host_netns=host_netns, isolated_netns=current_netns)
    if terminal_answer_check:
        identity_data.update(scope='synthetic_browser_startup_with_literal_answer_v2',terminal_answer_check=True)
    identity = hashlib.sha256(json.dumps(identity_data, sort_keys=True).encode()).hexdigest()
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder/'configuration.json',identity_data)
    config = dict(output=str(folder), deadline=min(deadline, time.time()+175),
        arbiter_source=str(source), identity_sha256=identity,
        episode_watchdog=dict(EPISODE_WATCHDOG, browser_phase_timeout_seconds=150,
                              cleanup_timeout_seconds=10, termination_grace_seconds=1))
    if terminal_answer_check:config['terminal_answer_check']=True
    result = run_supervised_episode(dict(task_id='synthetic_browser'), 'startup', config, _child)
    passed = (result.get('valid') is True and result.get('cleanup_status') == 'completed'
        and result.get('native_click_observed') is True
        and result.get('watchdog',{}).get('remaining_owned_processes') == {}
        and result.get('watchdog',{}).get('child_exitcode') == 0
        and result.get('watchdog',{}).get('reason') is None)
    if terminal_answer_check:passed &= result.get('normal_final_answer_verified') is True
    receipt = dict(status='passed' if passed else 'failed', scope='synthetic_browser_startup_only',
        benchmark_episodes=0, model_calls=0, provider_calls=0, identity_sha256=identity,
        result_path=str(folder/'synthetic_browser/startup/result.json'), result=result)
    write_new(folder/'receipt.json', receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arbiter-source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--deadline', required=True, type=float)
    parser.add_argument('--host-netns', required=True)
    parser.add_argument('--terminal-answer-check', action='store_true')
    args=parser.parse_args()
    result=run_preflight(args.arbiter_source,args.output,deadline=args.deadline,host_netns=args.host_netns,
        terminal_answer_check=args.terminal_answer_check)
    print(json.dumps(dict(status=result['status'],receipt=str(Path(args.output)/'receipt.json'))))
    raise SystemExit(0 if result['status']=='passed' else 1)


if __name__=='__main__':
    main()
