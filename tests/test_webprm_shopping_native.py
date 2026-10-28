"""Offline Shopping native-control contracts; no service, browser, credential or network use."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl import webprm_shopping_native as native
from openwebrl.webprm_cms_hosting import Closed

SOURCE = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')


def row(instance, name, *, session='s', helper=True):
    value = dict(status='passed', instance_id=instance, base_sha256='b'*64, customer=dict(id=27, name=name),
                 session_sha256=session, native_positive=1, native_negative=0)
    if helper:
        value['helper'] = dict(status='passed', latest_order_id=191)
    return value


class ShoppingNativeTests(unittest.TestCase):
    def setUp(self):
        block = patch('socket.socket.connect', side_effect=AssertionError('Network forbidden'))
        block.start(); self.addCleanup(block.stop)

    def test_multi_tab_start_urls_are_split_like_the_native_env(self):
        tasks = [dict(start_url='__SHOPPING__'), dict(start_url='__SHOPPING__/a.html |AND| __SHOPPING__/b.html'),
                 dict(start_url='__SHOPPING__/a.html')]
        self.assertEqual(native.start_urls(tasks), [native.ORIGIN, native.ORIGIN+'/a.html', native.ORIGIN+'/b.html'])
        for bad in ('http://example.com/x', '__SHOPPING__/a.html |AND|__SHOPPING__/b.html'):
            with self.subTest(bad=bad), self.assertRaises(Closed):native.start_urls([dict(start_url=bad)])

    def test_released_tasks_have_one_multi_tab_start_url(self):
        import json
        from openwebrl.webprm_webarena import TASK_FILE
        tasks = [t for t in json.loads((SOURCE/TASK_FILE).read_text()) if t['sites'] == ['shopping']]
        self.assertEqual(len(tasks), 45)
        self.assertEqual(len(native.start_urls(tasks)), 7)

    def test_customer_sentinel_sql_rejects_injection(self):
        self.assertIn(b"WHERE email='a.b@example.com';", native.customer_sql('a.b@example.com'))
        for bad in ("x' OR '1'='1@x.com", 'no-at-sign', 'a@b.com;DROP', None):
            with self.subTest(bad=bad), self.assertRaises(Closed):native.customer_sql(bad)

    def test_marker_is_letters_only_and_bounded(self):
        self.assertTrue(native.marker_ok('OpenWebRL Reset abcDEF'))
        for bad in ('OpenWebRL Reset 123', 'Other abc', 'OpenWebRL Reset '+'a'*60):
            self.assertFalse(native.marker_ok(bad))

    def test_reset_gate_requires_fresh_restored_state_and_helper(self):
        base, mutation, reset = row('1-a', 'Emma', session='s1'), row('1-a', 'OpenWebRL Reset x', helper=False), row('1-b', 'Emma', session='s2')
        gate = native.reset_gate(base, mutation, reset, media_absent=True)
        self.assertTrue(gate['fresh_reset_verified']); self.assertEqual(gate['customer_id'], 27)
        bad = dict(not_restored=(base, mutation, row('1-b', 'OpenWebRL Reset x', session='s2'), True),
                   same_instance=(base, mutation, row('1-a', 'Emma', session='s2'), True),
                   media_kept=(base, mutation, reset, False),
                   reused_session=(base, mutation, row('1-b', 'Emma', session='s1'), True),
                   no_helper=(base, mutation, row('1-b', 'Emma', session='s2', helper=False), True),
                   no_change=(base, row('1-a', 'Emma', helper=False), reset, True))
        for name, (b, m, r, absent) in bad.items():
            with self.subTest(name=name), self.assertRaises(Closed):native.reset_gate(b, m, r, media_absent=absent)

    def test_helper_control_uses_released_function_and_two_owned_calls(self):
        calls = []
        def request(self, method, url, *, headers, data=None, params=None):
            calls.append((method, url))
            if method == 'POST':
                return SimpleNamespace(status_code=200, json=lambda: 'token')
            return SimpleNamespace(status_code=200, json=lambda: dict(items=[dict(increment_id='000000191')]))
        accounts = dict(shopping_site_admin=dict(username='admin-fixture', password='secret-fixture'))
        with patch('openwebrl.webprm_webarena_helpers.OwnedShoppingRequests.request', request):
            result = native.helper_control(SOURCE, accounts)
        self.assertEqual(result, dict(status='passed', latest_order_id=191))
        self.assertEqual(calls, [('POST', native.ORIGIN+'/rest/default/V1/integration/admin/token'),
                                 ('GET', native.ORIGIN+'/rest/V1/orders')])

    def test_mutation_evidence_decodes_flash_cookie_and_compares_form_key(self):
        from urllib.parse import quote
        flash = quote('[{"type":"error","text":"Invalid Form Key. Please refresh the page."}]')
        class Context:
            def __init__(self, cookies): self.rows = cookies
            def cookies(self): return [dict(name=k, value=v) for k, v in self.rows.items()]
        self.assertEqual(native.mage_messages(Context({'mage-messages': flash})),
                         [dict(type='error', text='Invalid Form Key. Please refresh the page.')])
        self.assertEqual(native.mage_messages(Context({})), [])
        self.assertEqual(native.mage_messages(Context({'mage-messages': 'not json'}))[0]['type'], 'unparsed')
        class Inputs:
            def __init__(self, value): self.value = value
            def count(self): return 1
            def input_value(self): return self.value
        page = SimpleNamespace(locator=lambda selector: Inputs('k1'))
        self.assertEqual(native.form_key_evidence(Context({'form_key': 'k1'}), page),
                         dict(inputs=1, cookie_present=True, input_equals_cookie=True))
        self.assertEqual(native.form_key_evidence(Context({}), page),
                         dict(inputs=1, cookie_present=False, input_equals_cookie=False))


if __name__ == '__main__':
    unittest.main()
