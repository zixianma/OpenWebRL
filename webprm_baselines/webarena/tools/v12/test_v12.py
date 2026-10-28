"""Zero-paid CPU tests of the v12 OpenWebRL WebArena cell (run with the OpenWebRL venv Python; no GPU, no provider).

  V12_PACKAGE=<launch-preparation-v12-NAME> NLTK_DATA=... venv/bin/python test_v12.py

1. import layout: harness modules from the package, agent stack (generate_browser, WebEnv, feedback_utils) from the
   frozen snapshot, venv regex/tqdm ahead of the NLTK runtime.
2. trajectory projection: done -> send_msg_to_user answer; step limit -> N/A; done mixed with a click.
3. native-evaluator parity on saved v11 finals: every valid v11 string_match-only cell re-scored from its saved answer
   through the v12 projection gives the saved v11 score.
4. end to end on a local site: real Chromium (bundled chromium-1208) through the snapshot WebEnv at 1280x1000, the
   snapshot generate_turn_sample with crossbench MeteredActor(CrossbenchSelector), fake sglang /generate and fake
   /select servers reached through owrl_uds_relay.py UNIX sockets, then the released evaluator (string_match +
   url_match + program_html) on the live terminal page through SyncPage with the transcript recorder, transcript
   replay, and native-audit-v12's OpenWebRL evidence check (+ tampered-pick negative control).
"""
import asyncio
import base64
import http.server
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

D = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021')
T = D / 'v12-tools'
PKG = Path(os.environ['V12_PACKAGE'])
SOURCE = PKG / 'source'
spec = importlib.util.spec_from_file_location('v12cell', SOURCE / 'scripts/run_owrl_webarena_cell.py')
cell = importlib.util.module_from_spec(spec)
sys.modules['v12cell'] = cell
spec.loader.exec_module(cell)
from openwebrl import webprm_owrl_v12 as V12  # noqa: E402
TEMPLATE = json.loads((PKG / 'worker-template.json').read_text())


class ImportLayout(unittest.TestCase):
    def test_modules(self):
        import openwebrl.webprm_lifecycle as lifecycle, openwebrl.webprm_webarena as webarena
        import openwebrl.generate_browser as generation, openwebrl.env.web_env as web_env, openwebrl.feedback_utils as fb
        import regex
        for module in (lifecycle, webarena, V12):
            self.assertTrue(Path(module.__file__).resolve().is_relative_to(SOURCE), module.__file__)
        for module in (generation, web_env, fb):
            self.assertTrue(Path(module.__file__).resolve().is_relative_to(cell.SNAPSHOT), module.__file__)
        self.assertNotIn('nltk-runtime', regex.__file__)
        print('import layout: harness from package, agent stack from snapshot')

    def test_snapshot_verified(self):
        import hashlib
        sums = hashlib.sha256((cell.SNAPSHOT / 'SNAPSHOT_SHA256SUMS').read_bytes()).hexdigest()
        owrl = TEMPLATE.get('owrl') or dict(snapshot_sums_sha256=sums, protocol_sha256=V12.protocol_sha256())  # stage 2 adds it
        self.assertEqual(owrl['snapshot_sums_sha256'], sums)
        cell.verify_snapshot(dict(owrl=owrl))
        self.assertEqual(owrl['protocol_sha256'], V12.protocol_sha256())
        with self.assertRaises(Exception):
            cell.verify_snapshot(dict(owrl=dict(owrl, snapshot_sums_sha256='0' * 64)))


def step(index, *calls, url='http://x/after'):
    return dict(index=index, actions=[dict(name=n, args=a) for n, a in calls], url=url, screenshot=True)


class Projection(unittest.TestCase):
    def test_done(self):
        steps = [step(0, ('click', {'point_2d': [1, 2]})), step(1, ('done', {'response': 'forty two'}))]
        t = V12.project_trajectory('http://x/start', steps, 'forty two', done=True)
        self.assertEqual([e.get('action_type') for e in t[1::2]], ['owrl_tool_calls', 'send_msg_to_user'])
        self.assertEqual(len(t), 4)
        self.assertEqual(t[-1]['answer'], 'forty two')
        self.assertEqual(t[2]['observation']['url'], 'http://x/after')

    def test_step_limit(self):
        steps = [step(i, ('scroll', {'direction': 'down'})) for i in range(30)]
        t = V12.project_trajectory('http://x/start', steps, None, done=False)
        self.assertEqual(len(t), 62)
        self.assertEqual(t[-1], dict(action_type='send_msg_to_user', answer='N/A', owrl_terminal='released_max_step_fallback_na'))

    def test_done_with_click(self):
        steps = [step(0, ('click', {'point_2d': [1, 2]}), ('done', {'response': 'ok'}))]
        t = V12.project_trajectory('http://x/start', steps, 'ok', done=True)
        self.assertEqual(len(t), 4)
        self.assertEqual(t[1]['owrl_calls'][0]['name'], 'click')

    def test_done_without_call_rejected(self):
        with self.assertRaises(ValueError):
            V12.project_trajectory('http://x/start', [step(0, ('click', {}))], 'x', done=True)
        print('projection: done/N-A/mixed/negative ok')


class V11StringParity(unittest.TestCase):
    def test_saved_v11_string_finals(self):
        from openwebrl.webprm_webarena import load_native_evaluator, TASK_FILE
        from openwebrl.webprm_webarena_episode import TaskEnvelope
        source = TEMPLATE['arbiter_source']
        raw = {t['task_id']: t for t in json.loads((Path(source) / TASK_FILE).read_text())}
        evaluator = load_native_evaluator(source, scoring_protocol=TEMPLATE['evaluator_protocol'])
        checked = scores = 0
        for verdict_path in sorted(D.glob('launch-preparation-v11[bc]-*/campaign/attempt-*/cells/*/episodes/*/*/native-evaluation.json')):
            verdict = json.loads(verdict_path.read_text())
            if not verdict.get('valid'):
                continue
            plan = json.loads((verdict_path.parents[3] / 'worker-plan.json').read_text())
            task = TaskEnvelope.from_release(raw[plan['task_ids'][0]], plan['endpoints']).native_config()
            if task['eval']['eval_types'] != ['string_match'] or 'fuzzy_match' in task['eval']['reference_answers']:
                continue
            saved = json.loads((verdict_path.parent / 'native-evaluator-trajectory.json').read_text())
            answer = saved[-1]['answer']
            mine = V12.project_trajectory('http://x/', [step(0, ('done', {'response': answer}))], answer, done=True)
            got = evaluator.evaluate(task, mine, None, termination='stop')
            self.assertEqual((got['score'], got['success']), (verdict['score'], verdict['success']), str(verdict_path))
            checked += 1; scores += int(verdict['success'])
        self.assertGreater(checked, 10)
        print(f'v11 string_match parity: {checked} saved finals re-scored identically ({scores} successes)')


INDEX = '''<html><body style="margin:0"><h3>Contact</h3>
<button id="go" style="position:absolute;left:400px;top:420px;width:480px;height:160px" onclick="location.href='/done.html'">Submit</button>
</body></html>'''
DONE = '<html><body><div id="msg">submitted</div></body></html>'


class FakeModels(http.server.BaseHTTPRequestHandler):
    log = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        FakeModels.log.append((self.path, body))
        if self.path == '/generate':
            seed = body['sampling_params'].get('sampling_seed', 0)
            if 'done.html' in body['text']:
                call = {'name': 'done', 'arguments': {'response': 'The form was submitted'}}
            else:
                point = [500, 500] if seed % 2 == 0 else [40, 40]
                call = {'name': 'click', 'arguments': {'point_2d': point}}
            text = 'I look at the page.\n</think>\n<tool_call>\n' + json.dumps(call) + '\n</tool_call><|im_end|>'
            reply = dict(text=text, meta_info=dict(output_token_logprobs=[[0.0, 100 + i, None] for i in range(20)] + [[0.0, 151645, None]],
                                                    finish_reason={'type': 'stop'}, prompt_tokens=1000, completion_tokens=21))
        elif self.path == '/select':
            picks = [i for i, c in enumerate(body['candidates']) if '[500, 500]' in c['action'] or '"done"' in c['action']]
            reply = dict(raw=json.dumps({'selection': (picks[0] if picks else 0) + 1}), input_tokens=900, output_tokens=6, seconds=.01)
        else:
            self.send_error(404); return
        data = json.dumps(reply).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(data)))
        self.end_headers(); self.wfile.write(data)


def serve(handler, directory=None):
    if directory:
        handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=directory, **k)
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix='v12e2e-', dir='/tmp'))
        site = cls.tmp / 'site'; site.mkdir()
        (site / 'index.html').write_text(INDEX); (site / 'done.html').write_text(DONE)
        cls.site = serve(None, str(site))
        cls.models = serve(FakeModels)
        cls.relays = []
        for label in ('actor', 'select'):
            ready = cls.tmp / f'{label}.ready'
            cls.relays.append(subprocess.Popen([sys.executable, '-B', str(T / 'owrl_uds_relay.py'), '--listen', str(cls.tmp / f'{label}.sock'),
                                                '--port', str(cls.models.server_address[1]), '--ready', str(ready)]))
            for _ in range(100):
                if ready.exists():
                    break
                time.sleep(.1)
        cls.loop = cell.EpisodeLoop()

    @classmethod
    def tearDownClass(cls):
        cls.loop.stop()
        for p in cls.relays:
            p.terminate(); p.wait(10)
        cls.site.shutdown(); cls.models.shutdown()

    def run_method(self, method):
        from openwebrl.webprm_webarena import load_native_evaluator
        from openwebrl.webprm_evaluator_transcript import EvaluationRecorder, replay_native_evaluation
        folder = self.tmp / method; folder.mkdir()
        start = f'http://127.0.0.1:{self.site.server_address[1]}/index.html'
        task = SimpleNamespace(execution_id='webarena_lite_999', actor_view=dict(intent='Submit the contact form and report what the page says.',
                                                                                 start_url=start, sites=['shopping']))
        config = dict(owrl=dict(actor_socket=str(self.tmp / 'actor.sock'), select_socket=str(self.tmp / 'select.sock')))
        marks = []
        FakeModels.log.clear()
        env, samples, stats, resources = self.loop.run(cell.owrl_episode(task=task, method=method, config=config, folder=folder,
                                                                         storage_state=None, mark=marks.append), timeout=900)
        ending, valid, reason = V12.outcome_of(samples)
        self.assertEqual((ending, valid), ('done', True), reason)
        self.assertEqual(resources['env_config']['resize_scale'], 1000)
        self.assertTrue(env.exit_requested and not env.closed)
        generates = [b for p, b in FakeModels.log if p == '/generate']
        selects = [b for p, b in FakeModels.log if p == '/select']
        self.assertTrue(all(b['sampling_params']['temperature'] == 1.0 and b['sampling_params']['top_p'] == .9
                            and b['sampling_params']['max_new_tokens'] == 1024 for b in generates))
        self.assertEqual(len(generates), V12.CANDIDATES[method] * len(samples))
        from PIL import Image as _I
        for b in generates:
            self.assertEqual(len(b['image_data']), 1)
            img = b['image_data'][0]; img = img.split(',', 1)[1] if img.startswith('data:') else img
            self.assertEqual(_I.open(io.BytesIO(base64.b64decode(img))).size, (1280, 1000))
        self.assertIn('screen size: 1280 x 1000', generates[0]['text'])
        shots = sorted((folder / 'owrl-browser').glob('observation-*.png'))
        self.assertTrue(shots and all(_I.open(s).size == (1280, 1000) for s in shots))
        self.assertEqual(stats['actor_requests'], len(generates))
        rows = [json.loads(l) for p in (folder / 'selections').glob('*.jsonl') for l in p.read_text().splitlines()]
        if method == 'no_search':
            self.assertEqual(selects, [])
        else:
            self.assertEqual(len(selects), len(samples))
            from PIL import Image
            shot = Image.open(io.BytesIO(base64.b64decode(selects[0]['screenshot'])))
            self.assertEqual(shot.size, (1280, 1000))
            self.assertEqual(selects[0]['url'], start)
            self.assertTrue(all(len(r['candidates']) == 5 for r in rows))
        self.assertIn('browser_step', marks); self.assertIn('actor_request', marks)
        self.assertEqual(env.steps[-1]['actions'][-1]['name'], 'done')
        self.assertTrue(any((s['url'] or '').endswith('/done.html') for s in env.steps), env.steps)
        trajectory = V12.project_trajectory(env.start_url, env.steps, env.answer, done=True)
        task_cfg = dict(intent=task.actor_view['intent'], sites=['shopping'], start_url=start, storage_state=None, task_id=999,
                        eval=dict(eval_types=['string_match', 'url_match', 'program_html'],
                                  reference_answers={'must_include': ['submitted']},
                                  reference_url=f'http://127.0.0.1:{self.site.server_address[1]}/done.html', url_note='GOLD in PRED',
                                  program_html=[dict(url='last', locator="document.querySelector('#msg').outerText",
                                                     required_contents={'must_include': ['submitted']})],
                                  string_note='', reference_answer_raw_annotation=''))
        recorder = EvaluationRecorder(folder / 'native-evaluator-transcript.json', TEMPLATE['evaluator_observations'])
        evaluator = load_native_evaluator(TEMPLATE['arbiter_source'], scoring_protocol=TEMPLATE['evaluator_protocol'],
                                          observation_recorder=recorder)
        page = V12.SyncPage(env, self.loop.loop)
        page.set_default_timeout(V12.EVALUATION_TIMEOUT_MS)
        verdict = evaluator.evaluate(task_cfg, trajectory, page, termination='stop')
        self.assertTrue(verdict['valid'] and verdict['success'], verdict)
        self.assertEqual([d['score'] for d in verdict['evaluator_details']], [1.0, 1.0, 1.0])
        replay = replay_native_evaluation(TEMPLATE['arbiter_source'], task_cfg, trajectory,
                                          json.loads((folder / 'native-evaluator-transcript.json').read_text()), verdict)
        self.assertTrue(replay['verified'])
        wrong = V12.project_trajectory(env.start_url, env.steps, 'nothing here', done=True)
        evaluator2 = load_native_evaluator(TEMPLATE['arbiter_source'], scoring_protocol=TEMPLATE['evaluator_protocol'])
        self.assertFalse(evaluator2.evaluate(task_cfg, wrong, page, termination='stop')['success'])
        self.loop.run(env.close(), timeout=60)
        # native-audit-v12 OpenWebRL evidence check on this folder (+ negative control)
        result = dict(selector_picks=[r['selected_index'] for r in sorted(rows, key=lambda r: r['turn'])], executed_actions=len(env.steps),
                      terminal=dict(answer=trajectory[-1]['answer'], source=trajectory[-1]['owrl_terminal']))
        (folder / 'owrl-steps.json').write_text(json.dumps(dict(start_url=env.start_url, steps=env.steps, answer=env.answer,
                                                                ending=ending, terminate_reason=reason)))
        os.environ['WEBPRM_NATIVE_SOURCE_ROOT'] = str(SOURCE)
        aspec = importlib.util.spec_from_file_location('v12audit', T / 'native-audit-v12/native_cell_audit.py')
        audit = importlib.util.module_from_spec(aspec); aspec.loader.exec_module(audit)
        evidence = audit.audit_owrl({}, folder, None, result, trajectory, method)
        self.assertTrue(evidence['verified'])
        with self.assertRaises(ValueError):
            audit.audit_owrl({}, folder, None, dict(result, selector_picks=[4] * len(rows)), trajectory, method)
        print(f'{method}: {len(samples)} turns, {len(generates)} /generate, {len(selects)} /select, picks {result["selector_picks"]}, '
              f'all /generate{" and /select" if selects else ""} screenshots 1280x1000, verdict {verdict["evaluator_details"]}, transcript replay verified, audit evidence verified, tampered picks rejected')

    def test_no_search(self):
        self.run_method('no_search')

    def test_piotr(self):
        self.run_method('piotr_selectionarm')


if __name__ == '__main__':
    unittest.main(verbosity=2)
