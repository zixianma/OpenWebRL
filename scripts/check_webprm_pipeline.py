#!/usr/bin/env python3
"""Synthetic six-arm browser integration check; never a benchmark evaluation.

Runs real BrowserGym observations/actions and the worker's complete control
flow on a local data page. Actor, reward-model and judge responses are scripted.
Provider HTTP is disabled, credentials are explicitly fake, and no model server
or GPU is used. A successful scripted verdict requires observed DOM completion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch
from urllib.parse import quote

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

HTML = '''<!doctype html><html><head><title>WebPRM synthetic fixture</title>
<style>body{font:20px sans-serif;margin:25px}button{display:block;width:230px;
height:50px;margin:8px}</style></head><body data-solved="no">
<h1>Synthetic pipeline check</h1><p id="result">FIXTURE_PENDING</p>
''' + ''.join(
    f'''<button onclick="document.body.dataset.solved='yes';
document.getElementById('result').textContent='FIXTURE_SOLVED';
document.body.dataset.route='{i}'">Complete route {i}</button>'''
    for i in range(1, 6)) + '</body></html>'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_text(payload):
    pieces = []
    for message in payload['messages']:
        content = message['content']
        if isinstance(content, str):
            pieces.append(content)
        else:
            pieces.extend(block['text'] for block in content if block.get('type') == 'text')
    return '\n'.join(pieces)


def response(model, texts):
    return dict(model=model, choices=[dict(index=i, finish_reason='stop',
        message=dict(role='assistant', content=text)) for i, text in enumerate(texts)],
        usage=dict(prompt_tokens=0, completion_tokens=0), synthetic_fixture=True)


def synthetic_shepherd_score(model, probability):
    """Small explicit token distribution exercises the unmodified native parser."""
    result = response(model, ['<answer>Yes</answer>'])
    alternatives = [dict(token='Yes', logprob=math.log(probability)),
                    dict(token='No', logprob=math.log(1 - probability - .05)),
                    dict(token='In', logprob=math.log(.05))]
    result['choices'][0]['logprobs'] = dict(content=[
        dict(token='<answer>', logprob=0., top_logprobs=[]),
        dict(token='Yes', logprob=math.log(probability), top_logprobs=alternatives),
        dict(token='</answer>', logprob=0., top_logprobs=[]),
    ])
    return result


class ScriptedCompletions:
    def __init__(self, arm, environments, config, worker):
        self.arm, self.environments, self.config, self.worker = arm, environments, config, worker
        self.counts = Counter()
        self.evidence = dict(error_feedback_seen=False, judge_dom_checked=False)

    @property
    def env(self):
        return self.environments[-1]

    def dom(self):
        return self.env.page.evaluate('''() => ({solved:document.body.dataset.solved,
            route:document.body.dataset.route || null,
            result:document.getElementById('result').textContent})''')

    def __call__(self, config, payload, journal, *, local=False, endpoint=None, purpose=None):
        from openwebrl.jev_eval import append_json

        text = prompt_text(payload)
        model = payload['model']
        if local:
            assert endpoint in (config['shepherd_endpoint'], config['arbiter_endpoint'])
            assert endpoint.startswith('fixture://'), 'Fixture endpoints cannot contact model servers'
        else:
            assert endpoint is None
        append_json(journal, dict(event='request', synthetic_fixture=True, local=local,
                    endpoint=endpoint, purpose=purpose, payload=payload, reservation=None))

        if model == self.worker.JUDGE:
            assert not local
            dom = self.dom()
            assert dom['solved'] == 'yes' and dom['result'] == 'FIXTURE_SOLVED', dom
            assert 'FIXTURE_SOLVED' in text and 'Agent stopped: agent_stopped' in text
            assert 'send_msg_to_user' in text, 'Judge must see terminal action evidence'
            self.evidence.update(judge_dom_checked=True, final_dom=dom)
            self.counts['judge'] += 1
            result = response(model, ['SYNTHETIC FIXTURE: DOM and action history verified.\nStatus: success'])
        elif model == config['shepherd_model']:
            assert local and endpoint == config['shepherd_endpoint']
            if payload.get('logprobs'):
                self.counts['shepherd_score'] += 1
                # Fifth candidate wins against the more frequent first one.
                probability = .1 + .15 * ((self.counts['shepherd_score'] - 1) % 5)
                result = synthetic_shepherd_score(model, probability)
            else:
                self.counts['shepherd_checklist'] += 1
                result = response(model, ['<answer>1. Make the fixture show FIXTURE_SOLVED.</answer>'])
        elif model == config['arbiter_model']:
            assert local and endpoint == config['arbiter_endpoint']
            self.counts['arbiter_comparison'] += 1
            result = response(model, ['<Answer>Response 2</Answer>'])
        elif model == self.worker.ACTOR:
            assert not local
            if purpose == 'state_diff':
                self.counts['state_diff'] += 1
                result = response(model, ['The page now shows FIXTURE_SOLVED.' if
                    self.dom()['solved'] == 'yes' else 'The page is unchanged after the failed action.'])
            elif purpose == 'terminal':
                self.counts['terminal'] += 1
                if self.env.last_action_error:
                    assert self.env.last_action_error in text, 'Terminal detector lost execution feedback'
                    self.evidence['error_feedback_seen'] = True
                solved = self.dom()['solved'] == 'yes'
                result = response(model, [json.dumps(dict(thought='Observed fixture DOM state.',
                    action='send_msg_to_user [FIXTURE_SOLVED]' if solved else None,
                    confidence=.99 if solved else .1))])
            elif purpose == 'policy':
                self.counts['policy'] += 1
                if self.env.last_action_error:
                    assert self.env.last_action_error in text, 'Actor lost execution feedback'
                    self.evidence['error_feedback_seen'] = True
                ids = {int(route): bid for bid, route in
                    re.findall(r'\[(\d+)\][^\n]*button[^\n]*Complete route (\d)', text)}
                assert set(ids) == set(range(1, 6)), f'Current observation lacks fixture buttons: {ids}'
                if self.arm.startswith('arbiter'):
                    assert payload['n'] == 1 and payload['response_format'] == {'type': 'json_object'}
                    actions = (['click [999999]'] * 5 if self.counts['policy'] == 1 else
                               [f'click [{ids[i]}]' for i in range(1, 6)])
                    blocks = [dict(thought=f'Synthetic candidate {i}.', action=action, confidence=.5)
                              for i, action in enumerate(actions)]
                    result = response(model, [json.dumps(dict(BLOCKS=blocks))])
                else:
                    solved = self.dom()['solved'] == 'yes'
                    if self.arm.endswith('unguided'):
                        assert payload['n'] == 1
                        raw = ('This first response intentionally omits its action.'
                               if self.counts['policy'] == 1 else
                               'send_msg_to_user("FIXTURE_SOLVED")' if solved else
                               f'click("{ids[1]}")')
                        result = response(model, [raw])
                    else:
                        assert payload['n'] == 20
                        routes = [1] * 6 + [2] * 4 + [3] * 3 + [4] * 3 + [5] * 2
                        valid = ['send_msg_to_user("FIXTURE_SOLVED")' if solved else
                                 f'click("{ids[route]}")' for route in routes]
                        result = response(model, [valid[0], 'Intentionally missing action.',
                                                  *valid[1:], 'Another omitted action.'])
            else:
                raise AssertionError(f'Unexpected actor purpose: {purpose}')
        else:
            raise AssertionError(f'Unexpected fixture model: {model}')
        append_json(journal, dict(event='response', synthetic_fixture=True,
                                 reservation=None, response=result))
        return result


def verify_arm(arm, result, fixture, output, worker):
    assert result['attempted'] and result['valid'] and result['success'] == 1, result
    folder = output / 'synthetic-local-page' / arm
    rollout = json.loads((folder / 'rollout.json').read_text())
    verdict = json.loads((folder / 'verdict.json').read_text())
    assert verdict['parsed'] == 1 and verdict['model'] == worker.JUDGE
    assert verdict['raw'].startswith('SYNTHETIC FIXTURE:')
    assert (folder / 'terminal.jpg').stat().st_size > 0
    assert fixture.evidence['judge_dom_checked'] and fixture.counts['judge'] == 1
    assert rollout[-1]['selected']['action'].startswith('send_msg_to_user')
    if arm.startswith('shepherd'):
        if arm.endswith('unguided'):
            assert len(rollout) == 3
            assert rollout[0]['selected']['action'] == worker.NO_ADMISSIBLE_ACTION
            assert len(rollout[0]['rejected_samples']) == 1
            assert rollout[0]['action_error'] and fixture.evidence['error_feedback_seen']
            assert not rollout[1]['action_error']
        else:
            assert len(rollout) == 2
            for turn in rollout:
                assert len(turn['raw_samples']) == 18 and len(turn['rejected_samples']) == 2
                assert {s['sample_index'] for s in turn['raw_samples']} == set(range(20)) - {1, 19}
            assert len(rollout[0]['candidates']) == 5
            if arm == 'shepherd8b':
                assert fixture.counts['shepherd_checklist'] == 1
                assert fixture.counts['shepherd_score'] == 6
                first = rollout[0]
                assert len(first['selection_trace']) == 5
                assert first['selected']['score'] == max(c['score'] for c in first['candidates'])
                assert first['selected']['frequency'] == 2
                assert fixture.evidence['final_dom']['route'] == '5'
            else:
                assert not fixture.counts['shepherd_score']
    else:
        assert len(rollout) == 3 and fixture.counts['policy'] == 2
        assert fixture.counts['terminal'] == 3 and fixture.counts['state_diff'] == 2
        assert rollout[0]['selected']['action'] == 'click [999999]'
        assert 'lacks captured viewport bounds' in rollout[0]['action_error']
        assert fixture.evidence['error_feedback_seen']
        assert 'Last action execution error:' in rollout[1]['state_diff']
        assert not rollout[1]['action_error']
        assert rollout[-1]['native_terminal']['action'].startswith('send_msg_to_user')
        if arm == 'arbiter7b':
            assert fixture.counts['arbiter_comparison'] == 8
            assert [len(turn['selection_trace']) for turn in rollout] == [4, 4, 0]
        else:
            assert not fixture.counts['arbiter_comparison']
    return dict(arm=arm, verified=True, turns=len(rollout), scripted_calls=dict(fixture.counts),
                evidence=fixture.evidence, result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New output directory for synthetic artifacts')
    parser.add_argument('--runtime', default='/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009')
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--arm', help='Run one condition in an isolated interpreter')
    args = parser.parse_args()
    output, runtime = Path(args.output).resolve(), Path(args.runtime).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['OPENAI_API_KEY'] = 'fixture-only-not-a-provider-credential'
    credential = output / 'fixture.env'
    credential.write_text('OPENAI_API_KEY=fixture-only-not-a-provider-credential\n')
    from openwebrl import webprm_worker as worker
    if args.arm is not None and args.arm not in worker.PILOT_ARMS:
        parser.error('Unknown WebPRM condition')
    arms = [args.arm] if args.arm else list(worker.PILOT_ARMS)
    from openwebrl.jev_eval import JUDGE_SOURCE, write_json
    from openwebrl.webprm_source import load_shepherd, load_arbiter, load_arbiter_causal_renderer
    from openwebrl.webprm_arbiter_harness import load_arbiter_harness, load_native_observation_processor
    worker.prepare_browser()
    import browsergym.core.env as browser_env

    sources = dict(shepherd=load_shepherd(runtime / 'sources/shepherd').manifest,
        arbiter=load_arbiter(runtime / 'sources/arbiter').manifest,
        arbiter_harness=load_arbiter_harness(runtime / 'sources/arbiter').manifest,
        arbiter_observation=load_native_observation_processor(runtime / 'sources/arbiter',
            viewport_size={'width': 1280, 'height': 720}).manifest,
        arbiter_prompt=load_arbiter_causal_renderer(runtime / 'WebPRMBench.jsonl').manifest,
        shepherd_actor_sha256=digest(runtime / 'sources/shepherd-demo.py'))
    files = ['scripts/check_webprm_pipeline.py', 'openwebrl/webprm_worker.py',
        'openwebrl/webprm_runtime.py', 'openwebrl/webprm_lifecycle.py',
        'openwebrl/webprm_source.py', 'openwebrl/webprm_baselines.py',
        'openwebrl/webprm_arbiter_harness.py', 'openwebrl/webprm_shepherd_harness.py',
        'openwebrl/jev_eval.py', str(JUDGE_SOURCE)]
    hashes = {name: digest(REPO / name) for name in files}
    identity = dict(kind='SYNTHETIC_PIPELINE_CHECK_NOT_BENCHMARK', sources=sources,
                    conditions=arms,
                    code_sha256=hashes, html_sha256=hashlib.sha256(HTML.encode()).hexdigest(),
                    provider_calls=0, gpu_calls=0)
    config = dict(output=str(output), env_file=str(credential), seed=20261009,
        deadline=time.time() + args.seconds, code_sha256=hashes, source_manifests=sources,
        identity_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        shepherd_source=str(runtime / 'sources/shepherd'), arbiter_source=str(runtime / 'sources/arbiter'),
        shepherd_actor_source=str(runtime / 'sources/shepherd-demo.py'),
        arbiter_benchmark=str(runtime / 'WebPRMBench.jsonl'),
        shepherd_model='WebShepherd8B', arbiter_model='WebArbiter7B',
        shepherd_endpoint='fixture://shepherd-no-server', arbiter_endpoint='fixture://arbiter-no-server',
        api_base='fixture://provider-disabled', budget_path=str(output / 'unused-budget.json'))
    write_json(output / 'manifest.json', dict(identity=identity, config=config))
    task = dict(task_id='synthetic-local-page', start_url='data:text/html,' + quote(HTML),
                intent='Click a completion button so the page shows FIXTURE_SOLVED, then report completion.')
    environments = []
    original_env = browser_env.BrowserEnv

    class RecordingBrowserEnv(original_env):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            environments.append(self)

    def no_http(*args, **kwargs):
        raise AssertionError('Provider/model HTTP is forbidden in the synthetic fixture')

    reports = []
    for arm in arms:
        fixture = ScriptedCompletions(arm, environments, config, worker)
        with patch.object(browser_env, 'BrowserEnv', RecordingBrowserEnv), \
             patch.object(worker, 'completion', fixture), patch.object(worker, 'urlopen', no_http):
            # Exercise browser/protocol logic in this process so the explicit
            # no-HTTP mocks remain installed. Spawned worker isolation and
            # forced hangs are covered separately by lifecycle tests.
            result = worker._run_episode(task, arm, config)
        report = verify_arm(arm, result, fixture, output, worker)
        reports.append(report)
        write_json(output / 'progress.json', dict(kind=identity['kind'], verified=reports))
        print(json.dumps(dict(arm=arm, verified=True, turns=report['turns'],
                              scripted_calls=report['scripted_calls'])), flush=True)
    assert not Path(config['budget_path']).exists(), 'Synthetic responses must not reserve API funds'
    write_json(output / 'complete.json', dict(kind=identity['kind'], verified_arms=reports,
        provider_calls=0, model_server_calls=0, allocated_gpu_seconds=0,
        original_evaluation_artifacts_modified=False))
    print(json.dumps(dict(verified=len(reports), synthetic_only=True, output=str(output))), flush=True)


if __name__ == '__main__':
    main()
