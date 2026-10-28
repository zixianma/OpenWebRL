"""Pinned single-site evaluator helpers against offline pages/transports only."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
import urllib.request

from openwebrl.webprm_webarena import (
    EvaluationNotReady, evaluation_requirements, load_native_evaluator,
    prepare_manifest, substitute_endpoints, validate_evaluator_input,
)
from openwebrl.webprm_webarena_helpers import (
    GITLAB, HELPER_FILE, HELPER_SHA256, REDDIT, SHOPPING,
    HelperNotReady, NativeHelpers, OwnedShoppingRequests, _NoRedirect,
)

SOURCE=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')
ORIGIN='http://127.0.0.1:8123'
ACCOUNTS={'shopping_site_admin':{'username':'offline-admin','password':'offline-password'}}
TRAJECTORY=[{'observation':{'text':'Offline helper contract fixture'},'info':{}},
            {'action_type':'send_msg_to_user','answer':'done'}]


def task(url='last', locator='', expected='fixture'):
    return {'intent':'Offline helper contract fixture','eval':{'eval_types':['program_html'],
        'program_html':[{'url':url,'locator':locator,'required_contents':{'exact_match':expected}}]}}


class Page:
    def __init__(self,url=ORIGIN+'/f/example/42/title',value='fixture'):
        self.url,self.value=url,value
        self.navigations,self.expressions=[],[]
        self.context=SimpleNamespace(browser=SimpleNamespace(is_connected=lambda:True))

    def is_closed(self):return False

    def goto(self,url):
        self.navigations.append(url);self.url=url

    def content(self):return self.value

    def evaluate(self,expression):
        self.expressions.append(expression)
        return self.value


class Response(io.BytesIO):
    def __init__(self,value,status=200,raw=False):
        super().__init__(value if raw else json.dumps(value).encode())
        self.status=status


class Opener:
    def __init__(self,responses):self.responses=list(responses);self.calls=[]
    def open(self,request,**kwargs):
        self.calls.append((request,kwargs))
        item=self.responses.pop(0)
        if isinstance(item,Exception):raise item
        return item


@unittest.skipUnless(SOURCE.exists(),'Pinned source checkout not installed')
class NativeHelperTests(unittest.TestCase):
    def setUp(self):
        network=patch('socket.socket.connect',side_effect=AssertionError('Network forbidden'))
        network.start();self.addCleanup(network.stop)
        sleep=patch('time.sleep',return_value=None)
        sleep.start();self.addCleanup(sleep.stop)
        self.helpers=NativeHelpers(SOURCE)

    def test_all_thirteen_released_single_site_contracts_validate_only_explicitly(self):
        _,groups,_=prepare_manifest(SOURCE)
        tasks=[t for t in groups['released_single_site_155'] if evaluation_requirements(t)['helper_functions_required']]
        self.assertEqual([t['task_id'] for t in tasks],[104,112,117,118,130,133,134,135,136,137,138,139,140])
        configured=NativeHelpers(SOURCE,shopping_origin=ORIGIN,shopping_accounts=ACCOUNTS)
        expressions=[]
        for original in tasks:
            current=substitute_endpoints(original,{site:ORIGIN for site in original['sites']})
            before=deepcopy(current)
            with self.subTest(task_id=current['task_id']):
                with self.assertRaisesRegex(EvaluationNotReady,'helper-function'):
                    validate_evaluator_input(current,TRAJECTORY,Page(),allow_fuzzy=True)
                validate_evaluator_input(current,TRAJECTORY,Page(),allow_fuzzy=True,deployment_helpers=configured)
                self.assertEqual(current,before)
            expressions.extend(t[field] for t in current['eval']['program_html']
                               for field in ('url','locator') if t[field].startswith('func:'))
        self.assertEqual(len(expressions),15)
        self.assertEqual(sum(SHOPPING in e for e in expressions),3)
        self.assertEqual(sum(GITLAB in e for e in expressions),3)
        self.assertEqual(sum(REDDIT in e for e in expressions),9)

    def test_native_source_pin_and_no_implicit_deployment_import(self):
        with patch.object(Path,'read_bytes',return_value=b'changed helper source'):
            with self.assertRaisesRegex(HelperNotReady,'source changed'):NativeHelpers(SOURCE)
        real_import=__import__
        def guarded(name,*args,**kwargs):
            if 'env_config' in name:raise AssertionError('Implicit deployment credentials forbidden')
            return real_import(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded):
            explicit=NativeHelpers(SOURCE,shopping_origin=ORIGIN,shopping_accounts=ACCOUNTS)
        self.assertEqual(explicit.manifest['source_sha256'],HELPER_SHA256)
        self.assertFalse(explicit.manifest['website_readiness_verified'])
        self.assertNotIn('offline-password',json.dumps(explicit.manifest))
        self.assertNotIn(SHOPPING,self.helpers.functions)
        with self.assertRaisesRegex(HelperNotReady,'configured'):
            self.helpers.validate_task(task('func:shopping_get_latest_order_url()'))

    def test_all_thirteen_actual_task_helper_routes_execute_with_offline_fixtures(self):
        _,groups,_=prepare_manifest(SOURCE)
        tasks=[t for t in groups['released_single_site_155'] if evaluation_requirements(t)['helper_functions_required']]
        helpers=NativeHelpers(SOURCE,shopping_origin=ORIGIN,shopping_accounts=ACCOUNTS)
        # This deterministic stub only checks that fuzzy-required HTML reaches
        # its declared reference; it makes no claim about actual judge quality.
        judge=SimpleNamespace(enabled=True,provider_calls=0,
            fuzzy_match=lambda pred,reference,question:float(reference in pred),
            ua_match=lambda *args:0.)
        native=load_native_evaluator(SOURCE,deployment_helpers=helpers,fuzzy_judge=judge)
        for original in tasks:
            cfg=substitute_endpoints(original,{site:ORIGIN for site in original['sites']})
            values=[value for target in cfg['eval']['program_html']
                    for ref in target['required_contents'].values()
                    for value in (ref if isinstance(ref,list) else [ref])]
            page=Page(value=', '.join(values))
            if 'url_match' in cfg['eval']['eval_types']:
                page.url=cfg['eval']['reference_url']+'/42/offline-fixture'
            if original['sites']==['gitlab']:
                def evaluate(expression,page=page):
                    page.expressions.append(expression)
                    return 0 if 'const elements =' in expression else page.value
                page.evaluate=evaluate
            opener=Opener([Response('offline-token'),Response({'items':[{'increment_id':'0000042'}]})])
            with self.subTest(task_id=original['task_id']),patch('urllib.request.build_opener',return_value=opener):
                result=native.evaluate(cfg,TRAJECTORY,page)
                self.assertTrue(result['valid']);self.assertTrue(result['success'])
                self.assertEqual(len(page.navigations),len(cfg['eval']['program_html']))
                self.assertEqual(len(opener.calls),2 if original['sites']==['shopping'] else 0)

    def test_shopping_needs_both_explicit_owned_origin_and_credentials(self):
        for kwargs in ({'shopping_origin':ORIGIN},{'shopping_accounts':ACCOUNTS},
                       {'shopping_origin':ORIGIN,'shopping_accounts':{}},
                       {'shopping_origin':ORIGIN,'shopping_accounts':{'shopping_site_admin':{'username':'u','password':''}}}):
            with self.subTest(kwargs=kwargs),self.assertRaises(HelperNotReady):NativeHelpers(SOURCE,**kwargs)

    def test_loader_rejects_arbitrary_helper_object(self):
        with self.assertRaisesRegex(EvaluationNotReady,'pinned native helper'):
            load_native_evaluator(SOURCE,deployment_helpers=SimpleNamespace(validate_task=lambda _:set()))

    def test_reddit_native_canonicalization_preserves_nonpost_urls(self):
        native=load_native_evaluator(SOURCE,deployment_helpers=self.helpers)
        cfg=task("func:reddit_get_post_url('__last_url__')")
        cases=[(ORIGIN+'/f/example/42/title/comment/91?x=1#fragment',ORIGIN+'/f/example/42/'),
               (ORIGIN+'/f/example/42',ORIGIN+'/f/example/42/'),
               (ORIGIN+'/user/example?x=1',ORIGIN+'/user/example?x=1')]
        for before,after in cases:
            page=Page(before)
            with self.subTest(before=before):
                result=native.evaluate(cfg,TRAJECTORY,page)
                self.assertTrue(result['success'])
                self.assertEqual(page.navigations,[after])
                self.assertEqual(result['model_provider_calls'],0)

    def test_reddit_multiple_targets_use_actual_current_page_after_navigation(self):
        native=load_native_evaluator(SOURCE,deployment_helpers=self.helpers)
        cfg=task(ORIGIN+'/f/second/71/title')
        cfg['eval']['program_html'].append(task("func:reddit_get_post_url('__last_url__')")['eval']['program_html'][0])
        page=Page(ORIGIN+'/f/first/42/title')
        self.assertTrue(native.evaluate(cfg,TRAJECTORY,page)['success'])
        self.assertEqual(page.navigations,[ORIGIN+'/f/second/71/title',ORIGIN+'/f/second/71/'])

    def test_quoted_current_url_is_dispatched_as_data_never_python(self):
        cfg=task("func:reddit_get_post_url('__last_url__')")
        page=Page(ORIGIN+"/f/example/42/title?x=');__import__('os').system('unsafe');#")
        native=load_native_evaluator(SOURCE,deployment_helpers=self.helpers)
        with patch('os.system',side_effect=AssertionError('URL became Python')) as shell:
            self.assertTrue(native.evaluate(cfg,TRAJECTORY,page)['success'])
            shell.assert_not_called()
        self.assertEqual(page.navigations,[ORIGIN+'/f/example/42/'])

    def test_unbound_changed_or_injected_expression_never_executes(self):
        cfg=task("func:reddit_get_post_url('__last_url__')")
        with self.assertRaisesRegex(HelperNotReady,'Unbound'):self.helpers.evaluate_expression('reddit_get_post_url("x")')
        with self.helpers.bind(cfg,Page()):
            for expression in ['__import__("os").system("unsafe")','reddit_get_post_url("changed")',
                               'shopping_get_latest_order_url()']:
                with self.subTest(expression=expression),self.assertRaises(HelperNotReady):
                    self.helpers.evaluate_expression(expression)
        with self.assertRaises(HelperNotReady):self.helpers.evaluate_expression('reddit_get_post_url("x")')

    def test_outside_released_call_shapes_rejected_before_navigation(self):
        expressions=[("func:reddit_get_post_url('fixed')",''),
                     ('func:shopping_get_latest_order_url(1)',''),
                     ("func:reddit_get_post_url('__last_url__', x=1)",''),
                     ('last',"func:gitlab_get_project_memeber_role(__page__, \"x');alert(1)//\")"),
                     ('last',"func:gitlab_get_project_memeber_role(other_page, 'user')"),
                     ('last','func:__import__("os").system("unsafe")')]
        native=load_native_evaluator(SOURCE,deployment_helpers=self.helpers)
        for url,locator in expressions:
            page=Page()
            with self.subTest(url=url,locator=locator),self.assertRaises(HelperNotReady):
                native.evaluate(task(url,locator),TRAJECTORY,page)
            self.assertEqual(page.navigations,[])

    def test_gitlab_native_exact_account_then_role_query_and_multiple_accounts(self):
        cfg=task('last',"func:gitlab_get_project_memeber_role(__page__, 'first.user')",'Guest')
        cfg['eval']['program_html'].append(task('last',"func:gitlab_get_project_memeber_role(__page__, 'second-user')",'Maintainer')['eval']['program_html'][0])
        page=Page();responses=iter([2,'Guest',7,'Maintainer'])
        def evaluate(expression):page.expressions.append(expression);return next(responses)
        page.evaluate=evaluate
        result=load_native_evaluator(SOURCE,deployment_helpers=self.helpers).evaluate(cfg,TRAJECTORY,page)
        self.assertTrue(result['success'])
        self.assertEqual(len(page.expressions),4)
        self.assertIn("elements[i].outerText === '@first.user'",page.expressions[0])
        self.assertIn('td.col-max-role span")[2].outerText',page.expressions[1])
        self.assertIn("elements[i].outerText === '@second-user'",page.expressions[2])
        self.assertIn('td.col-max-role span")[7].outerText',page.expressions[3])

    def test_gitlab_transport_swallowed_by_native_helper_still_invalidates_result(self):
        cfg=task('last',"func:gitlab_get_project_memeber_role(__page__, 'user')",'')
        page=Page()
        def fail(_):raise RuntimeError('Browser transport disappeared')
        page.evaluate=fail
        with self.assertRaisesRegex(EvaluationNotReady,'native result discarded'):
            load_native_evaluator(SOURCE,deployment_helpers=self.helpers).evaluate(cfg,TRAJECTORY,page)

    def test_gitlab_native_missing_dom_semantics_remain_empty_on_live_browser(self):
        from playwright.sync_api import Error
        cfg=task('last',"func:gitlab_get_project_memeber_role(__page__, 'user')",'Guest')
        page=Page();calls=[]
        def evaluate(expression):
            calls.append(expression)
            if len(calls)==1:return -1
            raise Error("TypeError: Cannot read properties of undefined (reading 'outerText')")
        page.evaluate=evaluate
        result=load_native_evaluator(SOURCE,deployment_helpers=self.helpers).evaluate(cfg,TRAJECTORY,page)
        self.assertTrue(result['valid']);self.assertFalse(result['success'])
        self.assertEqual(len(result['native_javascript_errors']),1)
        self.assertIn('[-1].outerText',calls[1])

    def test_gitlab_requires_browser_evaluate_capability(self):
        cfg=task('last',"func:gitlab_get_project_memeber_role(__page__, 'user')",'Guest')
        with self.assertRaisesRegex(EvaluationNotReady,'page.evaluate'):
            load_native_evaluator(SOURCE,deployment_helpers=self.helpers).evaluate(cfg,TRAJECTORY,SimpleNamespace(url=ORIGIN))

    def test_shopping_native_token_and_latest_order_query_are_exact_and_bounded(self):
        accounts=deepcopy(ACCOUNTS)
        helpers=NativeHelpers(SOURCE,shopping_origin=ORIGIN,shopping_accounts=accounts)
        accounts['shopping_site_admin']['password']='mutated-after-construction'
        opener=Opener([Response('offline-token'),Response({'items':[{'increment_id':'0000042'}]})])
        cfg=task('func:shopping_get_latest_order_url()')
        page=Page()
        with patch('urllib.request.build_opener',return_value=opener) as build:
            result=load_native_evaluator(SOURCE,deployment_helpers=helpers).evaluate(cfg,TRAJECTORY,page)
        self.assertTrue(result['success'])
        self.assertEqual(page.navigations,[ORIGIN+'/sales/order/view/order_id/42/'])
        self.assertEqual(len(opener.calls),2)
        token,order=[c[0] for c in opener.calls]
        self.assertEqual(token.get_method(),'POST')
        self.assertEqual(token.full_url,ORIGIN+'/rest/default/V1/integration/admin/token')
        self.assertEqual(json.loads(token.data),ACCOUNTS['shopping_site_admin'])
        self.assertEqual(order.get_method(),'GET')
        self.assertEqual(urlsplit(order.full_url).path,'/rest/V1/orders')
        self.assertEqual(parse_qs(urlsplit(order.full_url).query),{
            'searchCriteria[sortOrders][0][field]':['created_at'],
            'searchCriteria[sortOrders][0][direction]':['DESC'],'searchCriteria[pageSize]':['1']})
        self.assertEqual(order.get_header('Authorization'),'Bearer offline-token')
        self.assertTrue(all(kwargs=={'timeout':10} for _,kwargs in opener.calls))
        for call in build.call_args_list:
            self.assertIsInstance(call.args[0],urllib.request.ProxyHandler)
            self.assertEqual(call.args[0].proxies,{})
            self.assertIsInstance(call.args[1],_NoRedirect)

    def test_shopping_sessions_reset_only_between_evaluations(self):
        helpers=NativeHelpers(SOURCE,shopping_origin=ORIGIN,shopping_accounts=ACCOUNTS)
        cfg=task('func:shopping_get_latest_order_url()')
        opener=Opener([Response('a'),Response({'items':[{'increment_id':'1'}]}),
                       Response('b'),Response({'items':[{'increment_id':'2'}]})])
        native=load_native_evaluator(SOURCE,deployment_helpers=helpers)
        with patch('urllib.request.build_opener',return_value=opener):
            for order in (1,2):
                page=Page();self.assertTrue(native.evaluate(cfg,TRAJECTORY,page)['success'])
                self.assertEqual(page.navigations,[ORIGIN+f'/sales/order/view/order_id/{order}/'])
        self.assertIsNone(helpers.requests.calls.get())


class OwnedShoppingRequestsTests(unittest.TestCase):
    def setUp(self):
        network=patch('socket.socket.connect',side_effect=AssertionError('Network forbidden'))
        network.start();self.addCleanup(network.stop)
        self.requests=OwnedShoppingRequests(ORIGIN)

    def test_origins_must_be_explicit_loopback_http_with_port(self):
        for origin in ['http://localhost:8123','https://127.0.0.1:8123','http://8.8.8.8:8123',
                       'http://127.0.0.1','http://u:p@127.0.0.1:8123','http://127.0.0.1:8123/path',
                       'http://127.0.0.1:8123?token=x','file:///tmp/site','http://127.0.0.1:bad']:
            with self.subTest(origin=origin),self.assertRaises(HelperNotReady):OwnedShoppingRequests(origin)
        self.assertEqual(OwnedShoppingRequests('http://[::1]:8123/').origin,'http://[::1]:8123')

    def test_no_session_wrong_endpoint_or_method_cannot_open_transport(self):
        with patch('urllib.request.build_opener') as build:
            with self.assertRaises(HelperNotReady):self.requests.get(ORIGIN+'/rest/V1/orders',headers={})
            with self.requests.session():
                for method,url in [('GET',ORIGIN+'/other'),('GET','http://8.8.8.8:8123/rest/V1/orders'),
                                   ('DELETE',ORIGIN+'/rest/V1/orders'),('POST',ORIGIN+'/rest/V1/orders')]:
                    with self.subTest(method=method,url=url),self.assertRaises(HelperNotReady):
                        self.requests.request(method,url,headers={})
            build.assert_not_called()

    def test_two_call_cap_counts_failed_requests_without_retry(self):
        opener=Opener([TimeoutError('offline timeout'),Response({'items':[]})])
        with patch('urllib.request.build_opener',return_value=opener),self.requests.session():
            with self.assertRaises(TimeoutError):self.requests.get(ORIGIN+'/rest/V1/orders',headers={})
            self.requests.get(ORIGIN+'/rest/V1/orders',headers={})
            with self.assertRaisesRegex(HelperNotReady,'cap'):self.requests.get(ORIGIN+'/rest/V1/orders',headers={})
        self.assertEqual(len(opener.calls),2)

    def test_response_status_size_and_json_fail_without_retry(self):
        for response,error in [(Response({},status=503),HelperNotReady),
                               (Response(b'x'*(2**20+1),raw=True),HelperNotReady),
                               (Response(b'not-json',raw=True),json.JSONDecodeError)]:
            opener=Opener([response])
            with self.subTest(error=error),patch('urllib.request.build_opener',return_value=opener),self.requests.session():
                with self.assertRaises(error):self.requests.get(ORIGIN+'/rest/V1/orders',headers={})
                self.assertEqual(self.requests.calls.get(),1)
                self.assertEqual(len(opener.calls),1)

    def test_redirect_handler_fails_instead_of_following_any_location(self):
        handler=_NoRedirect();request=urllib.request.Request(ORIGIN+'/rest/V1/orders')
        for destination in [ORIGIN+'/other','https://external.invalid/']:
            with self.subTest(destination=destination),self.assertRaisesRegex(HelperNotReady,'redirects'):
                handler.redirect_request(request,None,302,'Found',{},destination)


if __name__=='__main__':unittest.main()
