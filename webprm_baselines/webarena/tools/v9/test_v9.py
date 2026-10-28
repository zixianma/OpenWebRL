"""v9 science changes on the real released coordinator + Chromium + native broker bridge, and native-audit-v5 replay.

No provider calls (fake broker). Run with V9_PACKAGE = a built v9 package (its source on PYTHONPATH and
WEBPRM_NATIVE_SOURCE_ROOT). Fixture: the v8 profile form (Save button 2,000 px down; 1024x2490 viewport).
Covers: 20-sample top-5-frequency aggregation (invalid / filtered samples dropped, ties to the first sampled),
candidate order reaching the released tournament and the WebShepherd-style pool, whole offline replay by
native-audit-v5 (re-aggregating the saved 20-sample replies) with a tamper control, and the cell / launcher /
broker static contract.
"""
import asyncio, hashlib, importlib.util, json, logging, os, re, sys, unittest, uuid
from copy import deepcopy
from pathlib import Path
from urllib.parse import quote

D = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021')
PKG = Path(os.environ['V9_PACKAGE'])
sys.path.insert(0, str(PKG/'source')); sys.path.insert(0, str(D/'native-browser-v3'))
from openwebrl.webprm_broker import payload_sha256  # noqa: E402
from openwebrl.webprm_webarena import digest  # noqa: E402
from openwebrl.webprm_webarena_worker import single_action_system_prompt, ENVELOPE_NO_SEARCH_PROMPT  # noqa: E402
from native_episode import NativeBrowser, load_native, run_episode, stage_chromium, write_json, plain, SOURCE  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


cell = load('v9cell', PKG/'source/scripts/run_native_webarena_cell.py')
bridge_module = load('v9_native_broker_bridge', D/'native-broker-v2/native_broker_bridge.py')
replay = load('v9_audit_replay', D/'native-audit-v5/native_lowlevel_replay.py')
NATIVE = load_native(); SPEC = NATIVE.tools['engine'].ModelSpec
DEFAULT = SPEC('openai', 'gpt-4o-mini', 'chat', dict(max_tokens=2048, context_length=8192, temperature=1., top_p=.9))
EVIDENCE = D/'v9-tools/test-evidence'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ref = lambda p: dict(path=str(p), sha256=sha(p))

HTML = '''<!doctype html><html><body><h1>Profile fixture</h1><div style="height:2000px">spacer</div>
<label>Biography<input id="bio" value="Original profile"></label>
<button id="save" onclick="document.getElementById('state').textContent='Saved '+document.getElementById('bio').value">Save</button>
<button>Cancel</button><p id="state">Original profile</p></body></html>'''
SAVED_STOP = 'send_msg_to_user [Saved profile]'


class FakeBroker:
    """Policy (n=20 for selectors): desired x9, Cancel x5, scroll down x3, scroll up x1, a nonexistent element x1,
    one unparsable reply. Detector: premature bracketless stop until saved. Critic prefers desired / the saved stop."""
    _webprm_broker_client = True
    _webprm_budget_owner = 'webprm_worker.completion'

    def __init__(self, cell_id):
        self.calls = []; self.policy_calls = 0; self.desired = None; self.cell_id = cell_id; self.pairs = []

    def __call__(self, config, payload, journal, *, purpose, local):
        self.calls.append(dict(payload=deepcopy(payload), purpose=purpose, local=local))
        text = payload['messages'][1]['content']
        if purpose == 'native_policy':
            def element(role, label):
                match = re.search(r'^\s*\[(\d+)\]\s+'+role+r"\s+'"+re.escape(label)+"'", text, re.M)
                assert match, 'Native AX target absent: '+role+' '+label
                return match[1]
            bio, save, cancel = element('textbox', 'Biography'), element('button', 'Save'), element('button', 'Cancel')
            self.desired = f'type [{bio}] Changed profile' if self.policy_calls == 0 else f'click [{save}]'
            self.policy_calls += 1
            assert payload.get('n') == 20 and payload['temperature'] == 1.0 and payload['top_p'] == 1.0, payload.get('n')
            assert 'Generate exactly ONE action' in payload['messages'][0]['content']
            order = (['scroll [down]'] + [self.desired]*9 + [f'click [{cancel}]']*5 + ['scroll [down]']*2
                     + ['scroll [up]', 'click [999999]', None])
            contents = [json.dumps(dict(BLOCKS=[dict(thought='t'+str(i), action=a, confidence=.5)])) if a else 'not json'
                        for i, a in enumerate(order)]
            choices = [dict(index=i, finish_reason='stop', message=dict(role='assistant', content=c)) for i, c in enumerate(contents)]
            usage = dict(prompt_tokens=100, completion_tokens=400, total_tokens=500)
        else:
            if purpose == 'native_reward':
                a = re.search(r'\[The Begin of Response 1\](.*?)\[The End of Response 1\]', text, re.S).group(1)
                b = re.search(r'\[The Begin of Response 2\](.*?)\[The End of Response 2\]', text, re.S).group(1)
                good = SAVED_STOP if ('Saved Changed profile' in text and (SAVED_STOP in a or SAVED_STOP in b)) else self.desired
                raw = '<Answer>Response 1</Answer>' if good in a else '<Answer>Response 2</Answer>'
            else:
                system = payload['messages'][0]['content']
                if 'terminal' in system.lower() or 'completion' in system.lower() or 'task has been' in system.lower() or '"action"' in system:
                    saved = 'Saved Changed profile' in text
                    raw = json.dumps(dict(thought='saved' if saved else 'premature', confidence=.9,
                                          action='send_msg_to_user Saved profile' if saved else 'send_msg_to_user Profile looks fine'))
                else:
                    raw = 'Synthetic state difference.'
            choices = [dict(finish_reason='stop', message=dict(role='assistant', content=raw))]
            usage = dict(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        receipt = dict(request_id=uuid.uuid4().hex, cell_id=self.cell_id, identity_sha256=config['identity_sha256'],
                       model=payload['model'], purpose=purpose, local=local, payload_sha256=payload_sha256(payload),
                       budget_owner='webprm_worker.completion', usage=usage, reservation_state='local' if local else 'settled',
                       reservation_id=None if local else len(self.calls), settlement_event_sha256='b'*64)
        response = dict(model=payload['model'], choices=choices, usage=usage, _webprm_broker=receipt)
        self.pairs.append(dict(request=dict(payload=deepcopy(payload), purpose=purpose, local=local, reservation=receipt['reservation_id']),
                               response=dict(response=deepcopy(response)), broker=dict(request_id=receipt['request_id'])))
        return response


class V9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL); cls.staged = stage_chromium()

    def run_episode(self, method):
        root = EVIDENCE/(method+'-'+uuid.uuid4().hex); folder = root/'episode-folder'; folder.mkdir(parents=True)
        plan = dict(selector=method, actor_model='gpt-4o-2024-08-06', task_ids=[14], max_turns=6, arbiter_source=str(SOURCE),
                    native_components=dict(episode=ref(D/'native-browser-v3/native_episode.py'), broker=ref(D/'native-broker-v2/native_broker_bridge.py')))
        identity = digest(plan); cell_id = 'webarena_lite_014/gpt-4o-2024-08-06__'+method
        broker = FakeBroker(cell_id)
        routes = {('openai', 'fixture-policy', 'chat'): dict(purpose='native_policy', allowed_gen_configs=[None]),
                  ('openai', 'gpt-4o-mini', 'chat'): dict(purpose='native_auxiliary', allowed_gen_configs=[DEFAULT.gen_config])}
        local = []
        if method == 'arbiter7b':
            routes[('openai', 'fixture-critic', 'chat')] = dict(purpose='native_reward', allowed_gen_configs=[None]); local = ['fixture-critic']
        observed, samples = {}, []
        class CellLikeBridge(bridge_module.NativeBrokerBridge):  # the cell's CellBridge (v7 JSON mode, v8 reward, v9 policy)
            def request(self, engine, system, user, model=None):
                value = cell.json_mode(super().request(engine, system, user, model))
                if value['purpose'] == 'native_reward':
                    value['payload'].update(cell.V8_REWARD_DECODING); value['reward_decoding'] = 'v8_greedy_released_reward_lm_config'
                if value['purpose'] == 'native_policy':
                    value['upstream_unadapted_payload'] = deepcopy(value['payload'])
                    value['payload']['messages'][0]['content'] = single_action_system_prompt(value['payload']['messages'][0]['content'], version=ENVELOPE_NO_SEARCH_PROMPT)
                    value['policy_adaptation'] = 'disclosed_single_action_envelope_v2'
                    if method != 'no_search':
                        value['payload'].update(cell.V9_SAMPLING); value['policy_adaptation'] = 'v9_best_of_20_top5_frequency'
                return value
            def response_text(self, response):
                purpose = response.get('_webprm_broker', {}).get('purpose')
                if purpose == 'native_policy' and method != 'no_search':
                    single = super().response_text
                    single(dict(choices=[dict(finish_reason='stop', message=dict(role='assistant', content='{}'))], usage=response['usage']))
                    text, record = cell.v9_aggregate(response, observed['text'], NATIVE.tools['sampling'], lambda v: single(v)[0])
                    samples.append(record); return text, deepcopy(response['usage'])
                text, usage = super().response_text(response)
                if purpose in ('native_policy', 'native_auxiliary'):
                    text = cell.strip_json_fence(text)
                return text, usage
        bridge = CellLikeBridge(NATIVE.tools['engine'].LMEngine, routes=routes, client=broker, config=dict(identity_sha256=identity),
            cell_id=cell_id, local_models=local, declared_default_model=DEFAULT, evidence_root=folder/'native-models',
            max_attempts=300, wire_model_aliases={'gpt-4o-mini': cell.MINI})
        context_record = dict(run_id='v9-'+method, intent='Change biography to Changed profile and save it', domain='reddit',
            domain_knowledge='', policy_model=dict(provider='openai', model='fixture-policy', mode='chat', gen_config=None),
            reward_model=dict(provider='openai', model='fixture-critic', mode='chat', gen_config=None) if method == 'arbiter7b' else None)
        context = NATIVE.coordinator.RunContext.create(**dict(context_record, policy_model=SPEC(**context_record['policy_model']),
            reward_model=SPEC(**context_record['reward_model']) if context_record['reward_model'] else None))
        async def select_control(pool, inputs, ctx):  # WebShepherd-style: score = desired / saved stop, ties to lower index
            actions = [c['action'] for c in pool]
            return (actions.index(SAVED_STOP) if SAVED_STOP in actions else actions.index(broker.desired)), []
        v8_adapter = cell.v8_decision_adapter(method, select_control, lambda step, value: write_json(folder/f'selector-{step:03d}.json', value))
        async def adapter(*, native, agent, context, inputs):  # the cell's v9 observation hand-off
            observed.update(text=inputs['observation_text'], step=inputs['current_step'])
            return await v8_adapter(native=native, agent=agent, context=context, inputs=inputs)
        browser = NativeBrowser(NATIVE, folder/'native-browser', viewport=dict(cell.V8_VIEWPORT), local_fixture=True,
                                compatibility_geometry=True, chromium=self.staged['executable'])
        task = dict(task_id='v9-profile', intent=context.intent, start_url='data:text/html,'+quote(HTML),
                    storage_state=None, geolocation=None, eval=dict(eval_types=['program_html']))
        seen = {}
        def evaluate(configured, raw, page, terminal):
            seen['state'] = page.locator('#state').inner_text()
            projection = []
            for index, entry in enumerate(raw):
                if index % 2 == 0:
                    projection.append(dict(observation={k: v for k, v in entry['observation'].items() if k != 'image'},
                        image=browser.observations[index//2]['image'], info=dict(fail_error=entry['info'].get('fail_error', ''),
                            observation_metadata=deepcopy(entry['info']['observation_metadata']))))
                else:
                    projection.append(plain(entry))
            write_json(folder/'native-raw-trajectory.json', projection)
            return dict(fixture=True, valid=True, success=0)
        result = asyncio.run(run_episode(task=task, context=context, browser=browser, completion_scope=bridge.installed(),
            evaluate=evaluate, evidence_root=folder/'native-episode', max_steps=6, native=NATIVE, decision_adapter=adapter))
        write_json(folder/'native-context.json', context_record)
        cfg = dict(routes=[dict(provider=k[0], model=k[1], mode=k[2], **v) for k, v in routes.items()], local_models=local,
                   declared_default_model=dict(provider='openai', model='gpt-4o-mini', mode='chat', gen_config=DEFAULT.gen_config),
                   wire_model_aliases={'gpt-4o-mini': cell.MINI}, max_attempts=300, selector=method)
        write_json(folder/'native-bridge-config.json', cfg)
        write_json(folder/'result.json', dict(fixture=True))
        paths = dict(native_result=folder/'native-episode/result.json', native_context=folder/'native-context.json',
                     native_bridge_config=folder/'native-bridge-config.json', native_raw_trajectory=folder/'native-raw-trajectory.json',
                     native_bridge_events=folder/'native-models/events.jsonl', result=folder/'result.json')
        write_json(root/'summary.json', dict(method=method, valid=result.get('valid'), status=result.get('status'), error=result.get('error'),
            seen=seen, samples=samples, actions=[r['selected']['action'] if isinstance(r['selected'], dict) else str(r['selected'])
                                                 for r in result.get('selected_actions', [])]))
        return result, samples, seen, broker, plan, paths, root

    def check_aggregation(self, samples, broker):
        first = samples[0]
        self.assertEqual(first['valid_samples'], 18)  # 20 - nonexistent element - unparsable
        self.assertEqual(first['distinct_valid_actions'], 4)
        self.assertEqual([c['frequency'] for c in first['candidates']], [9, 5, 3, 1])
        self.assertEqual(first['candidates'][0]['action'], 'type [' + re.search(r'type \[(\d+)\]', first['candidates'][0]['action'])[1] + '] Changed profile')
        self.assertEqual(first['candidates'][2]['action'], 'scroll [down]')
        self.assertEqual(first['candidates'][2]['first_sample'], 0)  # 'scroll [down]' was sampled first
        statuses = [s['status'] for s in first['samples']]
        self.assertEqual(statuses.count('valid'), 18); self.assertEqual(statuses.count('unparsed'), 1)
        self.assertEqual(statuses.count('rejected_by_released_filter'), 1)
        policy = [c for c in broker.calls if c['purpose'] == 'native_policy']
        self.assertTrue(policy and all(c['payload']['n'] == 20 and c['payload']['top_p'] == 1.0 for c in policy))

    def test_arbiter_tournament_gets_top5_and_whole_replay(self):
        result, samples, seen, broker, plan, paths, root = self.run_episode('arbiter7b')
        self.assertTrue(result['valid'], (result.get('error_type'), result.get('error')))
        self.assertEqual(seen['state'], 'Saved Changed profile'); self.assertEqual(result['terminal']['answer'], 'Saved profile')
        self.check_aggregation(samples, broker)
        step0 = json.loads(sorted(paths['result'].parent.glob('selector-*.json'))[0].read_text())
        self.assertEqual([c['action'] for c in step0['candidates']][:4], [c['action'] for c in samples[0]['candidates']])
        self.assertEqual(step0['candidates'][step0['v8_stop_index']]['action'], 'send_msg_to_user [Profile looks fine]')
        self.assertEqual(step0['final_winner']['action'], samples[0]['candidates'][0]['action'])
        proof = replay.replay_bundle(plan=plan, paths=paths, calls=dict(pairs=broker.pairs), replay_root=root/'replay')
        self.assertTrue(proof['verified'])
        print('arbiter v9 replay', {k: proof[k] for k in ('decisions', 'events', 'native_requests', 'actual_actions')}, flush=True)
        # tamper control: a saved policy reply whose sample counts change the top-5 must fail the replay
        tampered = deepcopy(broker.pairs)
        first_policy = next(p for p in tampered if p['request']['purpose'] == 'native_policy')
        for choice in first_policy['response']['response']['choices'][1:10]:
            choice['message']['content'] = json.dumps(dict(BLOCKS=[dict(thought='x', action='scroll [up]', confidence=.5)]))
        with self.assertRaises(Exception):
            replay.replay_bundle(plan=plan, paths=paths, calls=dict(pairs=tampered), replay_root=root/'replay-tampered')
        print('tampered 20-sample reply rejected', flush=True)

    def test_shepherd_pool_frequency_order_and_whole_replay(self):
        result, samples, seen, broker, plan, paths, root = self.run_episode('shepherd8b')
        self.assertTrue(result['valid'], (result.get('error_type'), result.get('error')))
        self.check_aggregation(samples, broker)
        step0 = json.loads(sorted(paths['result'].parent.glob('selector-*.json'))[0].read_text())
        self.assertEqual([c['action'] for c in step0['candidates']][:4], [c['action'] for c in samples[0]['candidates']])
        self.assertEqual(step0['selected_index'], 0)
        # replay needs the real ViewerSelector for shepherd8b; the selector trace here is the fixture's (not replayed)

    def test_aggregation_edge_cases(self):
        sampling = NATIVE.tools['sampling']
        obs = "[5] button 'Go'\n[6] link 'Home'"
        def reply(actions):
            return dict(choices=[dict(finish_reason='stop', message=dict(role='assistant', content=json.dumps(dict(BLOCKS=[dict(thought='t', action=a)]))))
                                 for a in actions], usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2))
        one = lambda v: bridge_module.base.NativeBridge.response_text(v)[0]
        # ties: first sampled wins; at most 5 candidates
        acts = ['click [6]', 'click [5]', 'scroll [down]', 'scroll [up]', 'go_back', 'go_forward'] + ['click [5]', 'click [6]'] + ['go_back']*12
        text, rec = cell.v9_aggregate(reply(acts), obs, sampling, one)
        self.assertEqual([b['action'] for b in json.loads(text)['BLOCKS']], ['go_back', 'click [6]', 'click [5]', 'scroll [down]', 'scroll [up]'])
        self.assertEqual(text, replay.v9_candidates(reply(acts), obs, sampling, one))
        # a length-truncated sample is dropped, not fatal
        r = reply(['click [5]']*20); r['choices'][3]['finish_reason'] = 'length'
        text, rec = cell.v9_aggregate(r, obs, sampling, one)
        self.assertEqual(rec['valid_samples'], 19); self.assertEqual(text, replay.v9_candidates(r, obs, sampling, one))
        # no valid sample -> empty BLOCKS (the released sampler then has no candidates)
        text, rec = cell.v9_aggregate(reply(['click [77]']*20), obs, sampling, one)
        self.assertEqual(json.loads(text)['BLOCKS'], [])
        with self.assertRaises(Exception):
            cell.v9_aggregate(reply(['click [5]']*19), obs, sampling, one)

    def test_static_contract(self):
        src = (PKG/'source/scripts/run_native_webarena_cell.py').read_text()
        self.assertEqual(cell.V9_SAMPLING, dict(n=20, temperature=1.0, top_p=1.0)); self.assertEqual(cell.V9_TOP_N, 5)
        self.assertEqual(cell.PROTOCOL, 'released_coordinator_original_browser_table5_v9'); self.assertEqual(len(cell.TASK_IDS), 128)
        self.assertIn("if purpose=='native_policy' and method!='no_search':", src)
        self.assertIn("value['payload'].update(V9_SAMPLING)", src)
        self.assertIn("v9_observation.update(text=inputs['observation_text'],step=inputs['current_step'])", src)
        self.assertIn("domain=v9_domain,domain_knowledge=native.history.load_domain_knowledge(v9_domain)", src)
        self.assertIn("urlsplit(instance['endpoint'])", src)
        audit = load('v9_audit_cell', D/'native-audit-v5/native_cell_audit.py')
        self.assertEqual(json.loads(json.dumps(cell.V9_RECEIPT)), json.loads(json.dumps(audit.V9_RECEIPT)))
        broker = (PKG/'source/openwebrl/webprm_concurrent_broker.py').read_text()
        self.assertIn("not 1<=choices<=route.get('max_choices',1)", broker); self.assertIn("usage['completion_tokens']>tokens*choices", broker)
        from scripts import launch_webprm_table5_campaign as L
        routes = L.native_routes(json.loads((PKG/'proposal.json').read_text())['models'])
        self.assertEqual({(r['model'], r['purpose']) for r in routes if r.get('max_choices', 1) != 1}, {('gpt-4o-2024-08-06', 'native_policy')})
        self.assertEqual(L.V9_SITE_TASKS, dict(shopping=45, shopping_admin=35, reddit=18, gitlab=30))
        self.assertTrue(L.v9_fuzzy_judge_invalid(dict(valid=False, error_type='EvaluationNotReady', error=L.V9_FUZZY_ERROR)))
        self.assertFalse(L.v9_fuzzy_judge_invalid(dict(valid=False, error_type='AttributeError', error='x')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
