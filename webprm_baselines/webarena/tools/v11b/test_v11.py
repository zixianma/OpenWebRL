"""v11 (1280x720 actor viewport; no_search + Piotr SelectionARM) on the real released coordinator + Chromium + native broker
bridge, and native-audit-v8 replay. Copy of v10-tools/test_v10.py, adapted.

No provider or GPU calls (fake broker). Run with V11_PACKAGE = a built v11 package (its source on PYTHONPATH and
WEBPRM_NATIVE_SOURCE_ROOT). Fixture: a profile form whose Biography field and Save button sit about 600-660 px down:
below v10's 576 px ARM fold but inside the 1280x720 viewport. Covers: the actor and the ARM see the same 1280x720
capture (the Piotr request image is byte-identical to the actor's observation PNG, 1280x720); targets below the old
576 px fold get a visible normalized point (target_status in_viewport, y > 800/1000); released 5-BLOCK prompt unchanged;
filtered/duplicate slots never reach Piotr; terminal detector short-circuits selection; whole offline replay by
native-audit-v8 for piotr_selectionarm (tamper control) and for no_search at 1280x720; the audit-v8 bridge-config
contract; the launcher / cell static contract (two interleaved methods).
"""
import asyncio, base64, hashlib, importlib.util, io, json, logging, os, re, sys, unittest, uuid
from copy import deepcopy
from pathlib import Path
from urllib.parse import quote

D = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021')
PKG = Path(os.environ['V11_PACKAGE'])
os.environ['WEBPRM_NATIVE_SOURCE_ROOT'] = str(PKG/'source')
sys.path.insert(0, str(PKG/'source')); sys.path.insert(0, str(D/'native-browser-v3'))
from openwebrl.webprm_broker import payload_sha256  # noqa: E402
from openwebrl.webprm_webarena import digest  # noqa: E402
from openwebrl.webprm_webarena_piotr import MODEL as PIOTR_MODEL  # noqa: E402
from native_episode import NativeBrowser, load_native, run_episode, stage_chromium, write_json, plain, SOURCE  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


cell = load('v11cell', PKG/'source/scripts/run_native_webarena_cell.py')
bridge_module = load('v11_native_broker_bridge', D/'native-broker-v2/native_broker_bridge.py')
replay = load('v11_audit_replay', D/'native-audit-v8/native_lowlevel_replay.py')
audit = load('v11_audit_cell', D/'native-audit-v8/native_cell_audit.py')
NATIVE = load_native(); SPEC = NATIVE.tools['engine'].ModelSpec
DEFAULT = SPEC('openai', 'gpt-4o-mini', 'chat', dict(max_tokens=2048, context_length=8192, temperature=1., top_p=.9))
EVIDENCE = Path(__file__).resolve().parent/'test-evidence'
TEMPLATE = json.loads((PKG/'worker-template.json').read_text())
if 'piotr_manifest' not in TEMPLATE:  # stage-1 package: the builder adds the CMS Piotr campaign's fields at stage 2
    from openwebrl.webprm_webarena_piotr import validate_piotr_selectionarm  # noqa: E402
    _ref = json.loads((D.parent/'cms-table5-350-bounded-reap-recovery-v10/worker-template.json').read_text())
    TEMPLATE = {k: _ref[k] for k in ('piotr_checkpoint', 'piotr_model', 'piotr_representation', 'piotr_revision', 'piotr_source')}
    TEMPLATE['piotr_manifest'] = validate_piotr_selectionarm(TEMPLATE)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ref = lambda p: dict(path=str(p), sha256=sha(p))

HTML = '''<!doctype html><html><body style="margin:8px"><h1>Profile fixture</h1><div style="height:540px">spacer</div>
<label>Biography<input id="bio" value="Original profile"></label>
<button id="save" onclick="document.getElementById('state').textContent='Saved '+document.getElementById('bio').value">Save</button>
<button>Cancel</button><p id="state">Original profile</p></body></html>'''


class FakeBroker:
    """Policy: one released 5-BLOCK reply (desired, duplicate desired, Cancel, scroll, nonexistent element).
    Detector: empty action until saved, then the stop. Piotr: picks the candidate line naming the desired action."""
    _webprm_broker_client = True
    _webprm_budget_owner = 'webprm_worker.completion'

    def __init__(self, cell_id, method='piotr_selectionarm'):
        self.calls = []; self.policy_calls = 0; self.desired = None; self.cell_id = cell_id; self.pairs = []; self.piotr = []
        self.method = method; self.images = []

    def __call__(self, config, payload, journal, *, purpose, local, **kwargs):
        self.calls.append(dict(payload=deepcopy(payload), purpose=purpose, local=local))
        if purpose == 'selector_piotr':
            assert local and payload['model'] == PIOTR_MODEL and payload['temperature'] == 0 and payload['max_tokens'] == 128
            user = payload['messages'][1]['content']
            texts = [p['text'] for p in user if p.get('type') == 'text']
            images = [p['image_url']['url'] for p in user if p.get('type') == 'image_url']
            assert len(images) == 1 and images[0].startswith('data:image/png;base64,')
            from PIL import Image
            self.images.append(base64.b64decode(images[0].split(',', 1)[1]))
            with Image.open(io.BytesIO(self.images[-1])) as im:
                assert im.size == (1280, 720), im.size
            lines = [m for t in texts for m in re.findall(r'^  (\d+)\. (\{.*)$', t, re.M)]  # candidate lines (builder: '  {i}. {action}')
            k = payload['response_format']['json_schema']['schema']['properties']['selection']['maximum']
            assert len(lines) == k, (lines, k)
            pick = next(int(i) for i, line in lines if self.desired in line)
            self.piotr.append(dict(k=k, pick=pick, lines=lines, text='\n'.join(texts)))
            choices = [dict(finish_reason='stop', message=dict(role='assistant', content=json.dumps(dict(selection=pick))))]
            usage = dict(prompt_tokens=900, completion_tokens=5, total_tokens=905)
        else:
            text = payload['messages'][1]['content']
            if purpose == 'native_policy':
                def element(role, label):
                    match = re.search(r'^\s*\[(\d+)\]\s+'+role+r"\s+'"+re.escape(label)+"'", text, re.M)
                    assert match, 'Native AX target absent: '+role+' '+label
                    return match[1]
                bio, save, cancel = element('textbox', 'Biography'), element('button', 'Save'), element('button', 'Cancel')
                self.desired = f'type [{bio}] Changed profile' if self.policy_calls == 0 else f'click [{save}]'
                self.policy_calls += 1
                assert payload.get('n', 1) == 1 and payload['temperature'] == 1.0 and payload['top_p'] == 0.9, payload
                if self.method == 'no_search':
                    assert 'Generate exactly ONE action' in payload['messages'][0]['content']
                    raw = json.dumps(dict(BLOCKS=[dict(thought='do it', action=self.desired, confidence=.9)]))
                else:
                    assert 'Generate exactly ONE action' not in payload['messages'][0]['content']
                    blocks = [dict(thought='do it', action=self.desired, confidence=.9),
                              dict(thought='same again', action=self.desired, confidence=.8),
                              dict(thought='cancel', action=f'click [{cancel}]', confidence=.6),
                              dict(thought='look further', action='scroll [down]', confidence=.4),
                              dict(thought='missing', action='click [999999]', confidence=.3)]
                    raw = json.dumps(dict(BLOCKS=blocks))
            else:
                system = payload['messages'][0]['content']
                if 'terminal' in system.lower() or 'completion' in system.lower() or 'task has been' in system.lower() or '"action"' in system:
                    saved = 'Saved Changed profile' in text
                    raw = json.dumps(dict(thought='saved' if saved else 'not yet', confidence=.9,
                                          action='send_msg_to_user Saved profile' if saved else ''))
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


class V11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL); cls.staged = stage_chromium()

    def run_episode(self, method='piotr_selectionarm'):
        root = EVIDENCE/(method+'-'+uuid.uuid4().hex); folder = root/'episode-folder'; folder.mkdir(parents=True)
        plan = dict(selector=method, actor_model='gpt-4o-2024-08-06', task_ids=[14], max_turns=6, arbiter_source=str(SOURCE),
                    native_components=dict(episode=ref(D/'native-browser-v3/native_episode.py'), broker=ref(D/'native-broker-v2/native_broker_bridge.py')),
                    **{k: TEMPLATE[k] for k in ('piotr_checkpoint', 'piotr_model', 'piotr_representation', 'piotr_revision',
                                                'piotr_source', 'piotr_manifest')}, piotr_endpoint='http://127.0.0.1:9/v1')
        identity = digest(plan); cell_id = 'webarena_lite_014/gpt-4o-2024-08-06__'+method
        broker = FakeBroker(cell_id, method)
        routes = {('openai', 'fixture-policy', 'chat'): dict(purpose='native_policy', allowed_gen_configs=[None]),
                  ('openai', 'gpt-4o-mini', 'chat'): dict(purpose='native_auxiliary', allowed_gen_configs=[DEFAULT.gen_config])}
        callconfig = dict(plan, identity_sha256=identity)
        piotr = cell.NativePiotrSelector(callconfig, folder/'requests-piotr.jsonl', request=broker) if method == 'piotr_selectionarm' else None
        class CellLikeBridge(bridge_module.NativeBrokerBridge):  # the cell's CellBridge for piotr_selectionarm
            def request(self, engine, system, user, model=None):
                value = cell.json_mode(super().request(engine, system, user, model))
                if value['purpose'] == 'native_policy' and method == 'piotr_selectionarm':
                    value['upstream_unadapted_payload'] = deepcopy(value['payload'])
                    value['policy_adaptation'] = 'v10_released_five_block_prompt_unchanged'
                elif value['purpose'] == 'native_policy':  # the cell's no_search single-action envelope
                    value['upstream_unadapted_payload'] = deepcopy(value['payload'])
                    value['payload']['messages'][0]['content'] = cell.single_action_system_prompt(
                        value['payload']['messages'][0]['content'], version=cell.ENVELOPE_NO_SEARCH_PROMPT)
                    value['policy_adaptation'] = 'disclosed_single_action_envelope_v2'
                return value
            def response_text(self, response):
                text, usage = super().response_text(response)
                if response.get('_webprm_broker', {}).get('purpose') in ('native_policy', 'native_auxiliary'):
                    text = cell.strip_json_fence(text)
                return text, usage
        bridge = CellLikeBridge(NATIVE.tools['engine'].LMEngine, routes=routes, client=broker, config=dict(identity_sha256=identity),
            cell_id=cell_id, local_models=[], declared_default_model=DEFAULT, evidence_root=folder/'native-models',
            max_attempts=300, wire_model_aliases={'gpt-4o-mini': cell.MINI})
        context_record = dict(run_id='v11-'+method, intent='Change biography to Changed profile and save it', domain='reddit',
            domain_knowledge='', policy_model=dict(provider='openai', model='fixture-policy', mode='chat', gen_config=None), reward_model=None)
        context = NATIVE.coordinator.RunContext.create(**dict(context_record, policy_model=SPEC(**context_record['policy_model']), reward_model=None))
        holder = {}
        async def select_control(pool, inputs, ctx):  # the cell's no_search / piotr branches
            if method == 'no_search':
                assert len(pool) == 1; return 0, []
            return await asyncio.to_thread(piotr.select, pool, dict(intent=ctx.intent, current_url=inputs['current_url'],
                observation_text=inputs['observation_text']), deepcopy(holder['browser'].observations[-1]), inputs['current_step'])
        v8_adapter = cell.v8_decision_adapter(method, select_control, lambda step, value: write_json(folder/f'selector-{step:03d}.json', value))
        async def adapter(*, native, agent, context, inputs):
            decision = await v8_adapter(native=native, agent=agent, context=context, inputs=inputs)
            if piotr: piotr.note_decision(decision)
            return decision
        class HookedBrowser(NativeBrowser):  # the cell's PhasedNativeBrowser hook
            async def execute(self, action_string):
                if piotr: piotr.note_execution(action_string, deepcopy(self.observations[-1]))
                return await super().execute(cell.v8_bracket(action_string))
        browser = HookedBrowser(NATIVE, folder/'native-browser', viewport=dict(cell.V8_VIEWPORT), local_fixture=True,
                                compatibility_geometry=True, chromium=self.staged['executable'])
        holder['browser'] = browser
        task = dict(task_id='v11-profile', intent=context.intent, start_url='data:text/html,'+quote(HTML),
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
        cfg = dict(routes=[dict(provider=k[0], model=k[1], mode=k[2], **v) for k, v in routes.items()], local_models=[],
                   declared_default_model=dict(provider='openai', model='gpt-4o-mini', mode='chat', gen_config=DEFAULT.gen_config),
                   wire_model_aliases={'gpt-4o-mini': cell.MINI}, max_attempts=300, selector=method)
        write_json(folder/'native-bridge-config.json', cfg)
        write_json(folder/'result.json', dict(fixture=True))
        paths = dict(native_result=folder/'native-episode/result.json', native_context=folder/'native-context.json',
                     native_bridge_config=folder/'native-bridge-config.json', native_raw_trajectory=folder/'native-raw-trajectory.json',
                     native_bridge_events=folder/'native-models/events.jsonl', result=folder/'result.json')
        write_json(root/'summary.json', dict(method=method, valid=result.get('valid'), status=result.get('status'), error=result.get('error'),
            seen=seen, piotr=broker.piotr, history=piotr.history if piotr else None,
            actions=[r['selected']['action'] if isinstance(r['selected'], dict) else str(r['selected']) for r in result.get('selected_actions', [])]))
        return result, seen, broker, plan, paths, root, piotr

    def test_piotr_selects_and_whole_replay(self):
        result, seen, broker, plan, paths, root, piotr = self.run_episode()
        self.assertTrue(result['valid'], (result.get('error_type'), result.get('error')))
        self.assertEqual(seen['state'], 'Saved Changed profile'); self.assertEqual(result['terminal']['answer'], 'Saved profile')
        self.assertEqual(len(broker.piotr), 2)  # one ARM call per non-terminal step
        self.assertTrue(all(p['k'] == 3 for p in broker.piotr))  # desired, Cancel, scroll: duplicate + nonexistent never reach Piotr
        step0 = json.loads(sorted(paths['result'].parent.glob('selector-*.json'))[0].read_text())
        step1 = json.loads(sorted(paths['result'].parent.glob('selector-*.json'))[1].read_text())
        self.assertEqual(step0['adaptation'], 'v10_piotr_selectionarm_released_five_block_pool')
        self.assertEqual(len(step0['candidates']), 3); self.assertIsNone(step0['v8_stop_index'])
        self.assertTrue(step0['trace']['arm_called']); self.assertEqual(step0['trace']['considered'], 3)
        self.assertEqual(step0['candidates'][step0['selected_index']]['action'], json.loads(step0['trace']['grounded_candidates'][step0['selected_index']])['native_action'])
        # v11: the actor's own 1280x720 capture reaches Piotr byte for byte
        obs = sorted((paths['result'].parent/'native-browser').glob('observation-*.png'))
        self.assertEqual(broker.images[0], obs[0].read_bytes()); self.assertEqual(broker.images[1], obs[1].read_bytes())
        from PIL import Image
        with Image.open(obs[0]) as im: self.assertEqual(im.size, (1280, 720))
        # targets below v10's 576 px fold are now visible with a normalized point
        save = json.loads(step1['trace']['grounded_candidates'][step1['selected_index']])
        typed = json.loads(step0['trace']['grounded_candidates'][step0['selected_index']])
        for g in (typed, save):
            self.assertEqual(g['target_status'], 'in_viewport'); self.assertIsNotNone(g['point_2d'])
            self.assertGreater(g['point_2d'][1], 800)  # y > 576 px of 720 (v10 frame fold)
        self.assertEqual(save['operation'], 'click'); self.assertEqual(typed['operation'], 'type')
        bounds = json.loads(obs[1].with_suffix('.json').read_text())['obs_nodes_info'][save['target_element_id']]['union_bound']
        self.assertGreater(bounds[1] + bounds[3]/2, 576); self.assertLess(bounds[1] + bounds[3]/2, 720)
        # step-1 prompt carries the step-0 executed action in its history; executed actions are all grounded
        self.assertIn('Changed profile', broker.piotr[1]['text'].split('Action History')[-1] if 'Action History' in broker.piotr[1]['text'] else '')
        self.assertEqual([h['action'] for h in piotr.history][:2], [step0['candidates'][step0['selected_index']]['action'], save['native_action']])
        self.assertEqual(piotr.ungroundable, 0)
        # the released policy prompt was sent unchanged (no envelope, no n)
        policy = [c for c in broker.calls if c['purpose'] == 'native_policy']
        self.assertTrue(policy and all('n' not in c['payload'] for c in policy))
        proof = replay.replay_bundle(plan=plan, paths=paths, calls=dict(pairs=broker.pairs), replay_root=root/'replay')
        self.assertTrue(proof['verified'])
        print('piotr v11 replay', {k: proof[k] for k in ('decisions', 'events', 'native_requests', 'actual_actions', 'all_coordinator_selector_requests')}, flush=True)
        tampered = deepcopy(broker.pairs)
        first = next(p for p in tampered if p['request']['purpose'] == 'selector_piotr')
        sel = json.loads(first['response']['response']['choices'][0]['message']['content'])['selection']
        first['response']['response']['choices'][0]['message']['content'] = json.dumps(dict(selection=1 + sel % 3))
        with self.assertRaises(Exception):
            replay.replay_bundle(plan=plan, paths=paths, calls=dict(pairs=tampered), replay_root=root/'replay-tampered')
        print('tampered Piotr selection rejected', flush=True)

    def test_no_search_1280_whole_replay(self):
        result, seen, broker, plan, paths, root, piotr = self.run_episode('no_search')
        self.assertTrue(result['valid'], (result.get('error_type'), result.get('error')))
        self.assertEqual(seen['state'], 'Saved Changed profile'); self.assertEqual(broker.piotr, [])
        proof = replay.replay_bundle(plan=plan, paths=paths, calls=dict(pairs=broker.pairs), replay_root=root/'replay')
        self.assertTrue(proof['verified'])
        print('no_search v11 replay', {k: proof[k] for k in ('decisions', 'events', 'native_requests', 'actual_actions')}, flush=True)

    def test_audit_v8_bridge_contract(self):
        """The audit-v8 expectation equals exactly what the v11 cell writes for each method."""
        site, task, method = 'shopping_admin', 27, 'piotr_selectionarm'
        plan = dict(selector=method, actor_model='gpt-4o-2024-08-06', task_ids=[task], endpoints={site: 'http://127.0.0.1:7780/admin'})
        helper = dict(provider='openai', model='gpt-4o-mini', mode='chat', gen_config=dict(max_tokens=2048, context_length=8192, temperature=1., top_p=.9))
        context = dict(run_id=digest(plan)[:16]+f'-webarena_lite_{task:03d}-'+method, domain=site,
                       domain_knowledge=NATIVE.history.load_domain_knowledge(site) or '',
                       policy_model=dict(provider='openai', model=plan['actor_model'], mode='chat', gen_config=None), reward_model=None)
        routes = [dict(provider='openai', model=plan['actor_model'], mode='chat', purpose='native_policy', allowed_gen_configs=[None]),
                  dict(provider='openai', model='gpt-4o-mini', mode='chat', purpose='native_auxiliary', allowed_gen_configs=[helper['gen_config']])]
        cfg = json.loads(json.dumps(dict(routes=routes, local_models=[], declared_default_model=helper, wire_model_aliases={'gpt-4o-mini': cell.MINI},
                   max_attempts=512, selector=method, no_search_prompt_version=None, v8=cell.V8_RECEIPT, v9=None, v11=cell.V11_RECEIPT)))
        self.assertTrue(audit.audit_context_contract(plan, {}, context, cfg, NATIVE)['verified'])
        self.assertEqual(cfg['v8']['viewport'], dict(width=1280, height=720))
        ns = dict(cfg, selector='no_search', no_search_prompt_version=cell.ENVELOPE_NO_SEARCH_PROMPT, v11=None)
        ns_plan = dict(plan, selector='no_search'); ns_ctx = dict(context, run_id=digest(ns_plan)[:16]+f'-webarena_lite_{task:03d}-no_search')
        self.assertTrue(audit.audit_context_contract(ns_plan, {}, ns_ctx, ns, NATIVE)['verified'])
        for bad in (dict(cfg, v11=None), dict(cfg, v9=json.loads(json.dumps(cell.V9_RECEIPT))),
                    dict(cfg, v8=dict(cfg['v8'], viewport=dict(width=1024, height=2490))),
                    dict(cfg, v11=dict(cfg['v11'], frame=dict(cfg['v11']['frame'], crop=[0, 0, 1024, 576])))):
            with self.assertRaises(ValueError):
                audit.audit_context_contract(plan, {}, context, bad, NATIVE)

    def test_static_contract(self):
        src = (PKG/'source/scripts/run_native_webarena_cell.py').read_text()
        self.assertEqual(cell.METHODS, ('no_search', 'arbiter7b', 'shepherd8b', 'piotr_selectionarm'))
        self.assertIn("if value['purpose']=='native_policy' and method=='piotr_selectionarm':", src)
        self.assertIn("if purpose=='native_policy' and method in ('arbiter7b','shepherd8b'):", src)
        self.assertIn("if piotr:piotr.note_execution(action_string,deepcopy(self.observations[-1]))", src)
        self.assertEqual(cell.PROTOCOL, 'released_coordinator_original_browser_table5_v9'); self.assertEqual(len(cell.TASK_IDS), 128)
        from scripts import launch_webprm_table5_campaign as L
        proposal = json.loads((PKG/'proposal.json').read_text())  # stage 1: still the v8 proposal
        models = proposal['models'] if 'piotr_selectionarm' in proposal.get('models', {}) else {
            'piotr_selectionarm': json.loads((D.parent/'cms-table5-350-bounded-reap-recovery-v10/proposal.json').read_text())['models']['piotr_selectionarm']}
        routes = L.native_routes(models)
        self.assertEqual([r for r in routes if r.get('max_choices', 1) != 1], [])
        self.assertIn(dict(model=PIOTR_MODEL, purpose='selector_piotr', critic='piotr_selectionarm', local=True,
                           max_input_tokens=65536, max_output_tokens=128), routes)
        self.assertEqual(L.native_cell_routes('piotr_selectionarm', models)[-1], [PIOTR_MODEL, 'selector_piotr'])
        self.assertNotIn([PIOTR_MODEL, 'selector_piotr'], L.native_cell_routes('no_search', models))
        self.assertEqual(cell.V8_VIEWPORT, dict(width=1280, height=720)); self.assertEqual(cell.V8_RECEIPT['viewport'], dict(width=1280, height=720))
        self.assertEqual(L.CRITICS, ('piotr_selectionarm',)); self.assertEqual(L.PREFIX, {'piotr_selectionarm': 'piotr'})
        self.assertEqual(L.V9_SITE_TASKS, dict(shopping=45, shopping_admin=35, reddit=18, gitlab=30))
        launcher = (PKG/'source/scripts/launch_webprm_table5_campaign.py').read_text()
        self.assertIn("argv=model_command(name,model,self.ports[name],compatible_config=compatible)", launcher)
        self.assertIn("for selector in ('piotr_selectionarm',):", launcher)
        self.assertIn("can_start=lambda cell,active:self.can_start(cell,active),", launcher)
        self.assertIn("block=[c for c in cells if store.state(c)=='pending']", launcher)
        self.assertIn("expected=[(task,method) for task in ids for method in cohort['selectors']]", launcher)


if __name__ == '__main__':
    unittest.main(verbosity=2)
