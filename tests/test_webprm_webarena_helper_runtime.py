"""Generic helper injection with synthetic owned receipts; no network/browser."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl.webprm_webarena import TASK_FILE, EvaluationNotReady, evaluation_requirements
from openwebrl.webprm_webarena_episode import TaskEnvelope
from openwebrl.webprm_webarena_helper_runtime import (HELPER_CONFIGURATION,
    validate_helper_plan, owned_native_helpers)
from openwebrl.webprm_webarena_worker import (require_supported, episode_evaluator,
    protocol_definition, validate_protocol_contract)

SOURCE=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-baselines-20261009/sources/arbiter')
ENDPOINTS={site:'http://127.0.0.1:'+str(8000+i) for i,site in enumerate(
    ('reddit','shopping','shopping_admin','gitlab','map'))}


def wrapped(site='reddit'):
    targets={
        'reddit':dict(url="func:reddit_get_post_url('__last_url__')",locator='',required_contents={'exact_match':'expected'}),
        'gitlab':dict(url='last',locator="func:gitlab_get_project_memeber_role(__page__, 'fixture_user')",
            required_contents={'exact_match':'Owner'}),
        'shopping':dict(url='func:shopping_get_latest_order_url()',locator='',required_contents={'exact_match':'expected'})}
    raw=dict(task_id=900,intent='Synthetic visible task',start_url=ENDPOINTS[site],sites=[site],
        eval=dict(eval_types=['program_html'],program_html=[targets[site]]))
    return TaskEnvelope.from_release(raw,ENDPOINTS)


class HelperRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='webprm-helper-runtime-',dir='/tmp'))
        self.config=dict(arbiter_source=str(SOURCE),native_helpers=deepcopy(HELPER_CONFIGURATION),
            endpoints=ENDPOINTS,reset_id='a'*64)
        self.network=patch('socket.socket.connect',side_effect=AssertionError('Network forbidden'))
        self.network.start();self.addCleanup(self.network.stop)

    def instance(self,site='reddit'):
        return dict(scope='original',endpoint=ENDPOINTS[site],state_provenance_sha256='a'*64)

    def account_receipt(self):
        path=self.root/'accounts.json'
        path.write_text(json.dumps({'shopping_site_admin':{'username':'fixture-user','password':'fixture-private-value'}}))
        path.chmod(0o600)
        return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def test_all_released_single_site_helpers_have_explicit_supported_plans(self):
        tasks=json.loads((SOURCE/TASK_FILE).read_text())
        selected=[raw for raw in tasks if len(raw['sites'])==1 and evaluation_requirements(raw)['helper_functions_required']]
        self.assertEqual(len(selected),13)
        counts={}
        for raw in selected:
            task=TaskEnvelope.from_release(raw,ENDPOINTS)
            names=validate_helper_plan(task,self.config)
            self.assertTrue(names)
            counts[raw['sites'][0]]=counts.get(raw['sites'][0],0)+1
        self.assertEqual(counts,dict(reddit=8,shopping=3,gitlab=2))

    def test_missing_or_changed_explicit_configuration_remains_blocked(self):
        task=wrapped()
        for value in (None,{},dict(HELPER_CONFIGURATION,source_sha256='0'*64),
                dict(HELPER_CONFIGURATION,protocol='implicit')):
            with self.assertRaises(EvaluationNotReady):
                require_supported(task,'no_search',dict(self.config,native_helpers=value))
        require_supported(task,'no_search',self.config)

    def test_helper_adaptation_is_part_of_common_frozen_protocol_identity(self):
        legacy=protocol_definition(SOURCE)
        configured=protocol_definition(SOURCE,helper_configuration=HELPER_CONFIGURATION)
        self.assertNotEqual(legacy,configured)
        plan=dict(self.config,protocol=configured['identity'],protocol_contract=configured)
        self.assertEqual(validate_protocol_contract(plan),configured)
        with self.assertRaises(EvaluationNotReady):
            validate_protocol_contract(dict(plan,protocol_contract=legacy))

    def test_helper_declaration_cannot_cross_task_domain_or_execute_arbitrary_expression(self):
        for expression in ("func:shopping_get_latest_order_url()", "func:__import__('os').getcwd()"):
            raw=wrapped().native_config();raw['eval']['program_html'][0]['url']=expression
            task=TaskEnvelope.from_release(raw,ENDPOINTS)
            with self.assertRaises(EvaluationNotReady):validate_helper_plan(task,self.config)

    def test_only_matching_claimed_original_reset_can_bind(self):
        task=wrapped()
        for changes in (dict(scope='modified'),dict(endpoint='http://127.0.0.1:9999'),
                dict(state_provenance_sha256='b'*64),dict(state_provenance_sha256=None)):
            with self.assertRaises(EvaluationNotReady):
                owned_native_helpers(task,self.config,dict(self.instance(),**changes))
        with self.assertRaises(EvaluationNotReady):
            owned_native_helpers(task,dict(self.config,reset_id='unverified'),self.instance())

    def test_actual_episode_evaluator_routes_reddit_helper_and_retains_provenance(self):
        task=wrapped();helpers=owned_native_helpers(task,self.config,self.instance())
        native=episode_evaluator(task,self.config,lambda *a,**kw:self.fail('Provider forbidden'),self.root,
            deployment_helpers=helpers)
        navigations=[]
        page=SimpleNamespace(url=ENDPOINTS['reddit']+'/f/books/55/comments/66',
            goto=lambda value:navigations.append(value),content=lambda:'expected')
        trajectory=[dict(observation=dict(text='Synthetic state'),info={}),dict(action_type='send_msg_to_user',answer='N/A')]
        with patch('openwebrl.webprm_webarena.time.sleep'):
            result=native.evaluate(task.native_config(),trajectory,page,termination='stop')
        self.assertEqual(result['score'],1)
        self.assertEqual(navigations,[ENDPOINTS['reddit']+'/f/books/55/'])
        self.assertEqual(result['model_provider_calls'],0)
        self.assertEqual(result['native_helper_manifest']['configured_task_sha256'],task.configured_config_sha256)
        self.assertNotIn('eval',task.actor_view)

    def test_actual_episode_evaluator_routes_gitlab_helper_without_shopping_secrets(self):
        task=wrapped('gitlab');helpers=owned_native_helpers(task,self.config,self.instance('gitlab'))
        expressions=[]
        def evaluate(code):
            expressions.append(code)
            return 0 if 'let index = -1' in code else 'Owner'
        page=SimpleNamespace(url=ENDPOINTS['gitlab']+'/group/project/-/project_members',evaluate=evaluate)
        native=episode_evaluator(task,self.config,None,self.root,deployment_helpers=helpers)
        result=native.evaluate(task.native_config(),[dict(observation=dict(text='state'),info={}),
            dict(action_type='send_msg_to_user',answer='N/A')],page,termination='stop')
        self.assertEqual(result['score'],1)
        self.assertEqual(len(expressions),2)
        self.assertIsNone(helpers.requests)

    def test_helper_evaluator_rejects_missing_or_different_task_binding(self):
        task=wrapped()
        with self.assertRaises(EvaluationNotReady):episode_evaluator(task,self.config,None,self.root)
        helpers=owned_native_helpers(task,self.config,self.instance())
        changed=task.native_config();changed['intent']='Different frozen task'
        alternate=TaskEnvelope.from_release(changed,ENDPOINTS)
        with self.assertRaises(EvaluationNotReady):
            episode_evaluator(alternate,self.config,None,self.root,deployment_helpers=helpers)

    def test_shopping_requires_hash_bound_private_credentials_never_serialized(self):
        task=wrapped('shopping');instance=self.instance('shopping')
        with self.assertRaises(EvaluationNotReady):owned_native_helpers(task,self.config,instance)
        receipt=self.account_receipt();instance['native_helper_accounts']=receipt
        helper=owned_native_helpers(task,self.config,instance)
        self.assertEqual(helper.requests.origin,ENDPOINTS['shopping'])
        serialized=json.dumps(helper.manifest)
        for private in ('fixture-user','fixture-private-value',receipt['path']):self.assertNotIn(private,serialized)
        self.assertNotIn('native_helper_accounts',task.actor_view)
        Path(receipt['path']).write_text('{}')
        with self.assertRaisesRegex(EvaluationNotReady,'receipt changed'):
            owned_native_helpers(task,self.config,instance)

    def test_public_or_redirected_accounts_fail_before_reading_private_contents(self):
        receipt=self.account_receipt();task=wrapped('shopping')
        instance=dict(self.instance('shopping'),native_helper_accounts=receipt)
        Path(receipt['path']).chmod(0o644)
        with self.assertRaisesRegex(EvaluationNotReady,'private regular'):
            owned_native_helpers(task,self.config,instance)
        Path(receipt['path']).chmod(0o600)
        link=self.root/'alias.json';link.symlink_to(Path(receipt['path']))
        instance['native_helper_accounts']=dict(receipt,path=str(link))
        with self.assertRaisesRegex(EvaluationNotReady,'canonical non-symlink'):
            owned_native_helpers(task,self.config,instance)


if __name__=='__main__':unittest.main()
