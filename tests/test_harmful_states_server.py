import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.request import build_opener, ProxyHandler, Request

from openwebrl.harmful_states import HarmfulStatesSandbox, OTHER_RECIPIENT
from openwebrl.harmful_states_server import running_server


class BrowserServerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.world = HarmfulStatesSandbox(Path(tmp.name)/'world.sqlite')
        self.campaign = self.world.start_campaign('send_recipient')
        cm = running_server(self.world, self.campaign)
        self.server = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        self.http = build_opener(ProxyHandler({}))

    def request(self, path, payload=None, *, auth=False, raw=False):
        headers = {'Content-Type': 'application/json'}
        if auth: headers['Authorization'] = 'Bearer '+self.server.control_token
        req = Request(self.server.url+path,
                      data=json.dumps(payload).encode() if payload is not None else None,
                      headers=headers)
        with self.http.open(req, timeout=5) as response:
            data = response.read().decode()
            return data if raw else json.loads(data)

    def test_task_page_and_observation_have_controls_but_no_oracle(self):
        page = self.request('/', raw=True)
        self.assertIn('Send message', page)
        self.assertIn('method="post"', page)
        self.assertIn('for="send_message-recipient"', page)
        self.assertIn('id="send_message-recipient"', page)
        for forbidden in ['any_harm', 'safe_success', self.server.control_token, '/api/audit']:
            self.assertNotIn(forbidden, page)
        observation = self.request('/api/observation')
        self.assertIn('episode_id', observation)
        self.assertNotIn('verdict', observation)

    def test_reset_and_audit_require_evaluator_auth(self):
        for path, body in [('/api/audit', None), ('/control/reset', {})]:
            with self.assertRaises(HTTPError) as ctx:
                self.request(path, body)
            self.assertEqual(ctx.exception.code, 403)

    def test_new_episode_retains_effects_and_public_response_hides_labels(self):
        e = self.request('/api/observation')['episode_id']
        response = self.request('/api/action', dict(episode_id=e, action_id='send_message',
            parameters={'recipient': OTHER_RECIPIENT}, request_id='wrong-send'))
        self.assertEqual(set(response), {'receipt_id', 'status', 'message'})
        e2 = self.request('/control/reset', {}, auth=True)['episode_id']
        self.assertNotEqual(e, e2)
        self.request('/api/action', dict(episode_id=e2, action_id='send_message', request_id='correct-send'))
        audit = self.request('/api/audit', auth=True)
        self.assertTrue(audit['verdict']['goal_success'])
        self.assertTrue(audit['verdict']['any_harm'])
        self.assertFalse(audit['verdict']['safe_success'])
        self.assertEqual(audit['verdict']['dispatch_count'], 2)

    def test_cross_campaign_payload_cannot_mutate_another_world(self):
        c = self.world.start_campaign('send_recipient')
        e = self.world.start_episode(c)
        with self.assertRaises(HTTPError) as ctx:
            self.request('/api/action', dict(episode_id=e, action_id='send_message'))
        self.assertEqual(ctx.exception.code, 400)
        self.assertEqual(self.world.verdict(c)['dispatch_count'], 0)
        self.assertEqual(self.world.verdict(self.campaign)['dispatch_count'], 0)

    def test_application_content_is_html_escaped(self):
        e = self.server.episode_id
        self.request('/api/action', dict(episode_id=e, action_id='save_draft',
            parameters={'body': '<script>alert(1)</script>'}))
        page = self.request('/', raw=True)
        self.assertIn('&lt;script&gt;', page)
        self.assertNotIn('<script>', page)


if __name__ == '__main__':
    unittest.main()
