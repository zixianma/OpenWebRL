"""CPU configuration/orchestration fixtures; never actor, browser, GPU or API calls."""
import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1]/'scripts/prepare_action_credit_evaluations.py'
SPEC = importlib.util.spec_from_file_location('evaluation_preparation', PATH)
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)

from openwebrl.action_credit_eval_executor import _validate_config
from openwebrl.action_credit_evaluation import Full300EvaluationOwner
from openwebrl.action_credit_verification import VerificationManifest, approved_authority


class PreparationTests(unittest.TestCase):
    def setUp(self):
        # All fixtures retained: this preparation task explicitly forbids deletion.
        self.root = Path(tempfile.mkdtemp(prefix='action-evaluation-draft-', dir='/tmp'))
        self.audit, self.runtime, self.pilots = [self.root/x for x in ('audit','runtime','pilots')]
        self.output = self.root/'output'
        self.source = self.runtime/'source'
        text = "class GenerateState:\n def __init__(self,args):\n  self.model=args.hf_checkpoint\n  self.stop=args.rollout_stop\n  self.limit=args.sglang_server_concurrency\n  self.flag=getattr(args,'sglang_enable_deterministic_inference',False)\n"
        source_files = {}
        for name in (prep.SERVER,prep.PROMPT,prep.TOOLS,prep.PARSER,prep.GENERATOR,
                     'openwebrl/action_credit_eval_executor.py','openwebrl/action_credit_evaluation.py','openwebrl/reward_browser.py'):
            content = text if name == prep.GENERATOR else 'CPU source fixture\n'
            if name.endswith('reward_browser.py'):
                content = "JUDGE_SYSTEM_PROMPT_ACTION_HISTORY='system'\nJUDGE_USER_PROMPT_ACTION_HISTORY='user'\n"
            ref = self.raw(self.source/name, content.encode())
            source_files[name] = ref['sha256']
        self.inventory = dict(source_root=str(self.source), files=source_files)
        self.model_root = self.root/'model'
        model_files = [self.raw(self.model_root/name, b'CPU fixture '+name.encode())
                       for name in sorted(set.union(*prep.RESOURCE_NAMES.values()))]
        self.model_ref = self.json(self.audit/'original-actor-inventory.json',dict(model_root=str(self.model_root),files=model_files))
        self.order_ref = self.json(self.audit/'frozen-task-order.json',dict(original_manifest_task_order=[f'train-{i}' for i in range(2102)]))
        verification = dict(judge_http_attempts_cap=1000,judge_usd_cap=25,browser_dispatches_cap=1500,
            local_actor_requests_cap=20000,includes_shared_initial_full300=True,no_execution_until_api_authority_bound=True,
            shared_initial_full300_reserve=dict(tasks=300,judge_http_attempts=600,usd=12,browser_dispatches=600))
        self.proposal_ref = self.json(self.audit/'five-arm-iter10-resource-science-proposal.json',
            dict(validation=dict(separately_proposed_online_verification=verification)))
        rows = [dict(prompt='CPU task',metadata=dict(task_id=f'eval-{i}')) for i in range(300)]
        self.tasks = self.raw(self.root/'tasks.jsonl', '\n'.join(json.dumps(r) for r in rows).encode())
        self.json(self.audit/'full300-task-reference.json',dict(evaluation_manifest=self.tasks,training_order=self.order_ref))
        self.pins = dict(native_args=dict(hf_checkpoint=str(self.model_root)), browser_config=dict(mode='local_process',
            width=1280,height=720,dpr=1,use_screenshot=True,use_a11ytree=False,resize_output_coords=True,resize_scale=1000,
            local_process=dict(server_module='openwebrl.arm_branch_browser',max_processes=64,python_bin='/cpu/python')))
        for kind,names in prep.RESOURCE_NAMES.items():
            self.pins[kind+'_files'] = [r for r in model_files if Path(r['path']).name in names]
        for arm in prep.ARM_IDS:
            self.json(self.pilots/arm/'source-inventory.json', self.inventory)
            self.json(self.pilots/arm/'runtime-pins.json', self.pins)
            self.json(self.pilots/arm/'worker.common.json',dict(approval_reference=None,judge_protocol=None))
            self.json(self.pilots/arm/'manifest.json',dict(arm_id=arm,initialization=dict(iteration=0),science=dict(
                checkpoint=dict(original_actor_inventory=self.model_ref),
                judge=dict(prompt_sha256=prep._judge_prompt(self.inventory),canonical_reward_source_sha256=source_files['openwebrl/reward_browser.py']))))
        self.source_ref = prep.reference(self.pilots/prep.ARM_IDS[0]/'source-inventory.json')
        self.argv = self.json(self.runtime/'native-argv.json', ['--hf-checkpoint',str(self.model_root)])
        self.config = self.json(self.runtime/'native-config.json',dict(max_steps=15))
        self.parsed = dict(schema_version=1,verification_kind='partial_cpu_argument_validation',cuda_initialized=False,
            browser_calls=0,judge_calls=0,physical_native_acceptance=False,source_inventory=self.source_ref,
            native_argv=self.argv,native_config=self.config,native_args=dict(hf_checkpoint=str(self.model_root),
                rollout_stop=None,sglang_server_concurrency=16,sglang_enable_deterministic_inference=False))
        self.parsed_ref = self.json(self.root/'parsed.json',self.parsed)
        self.expected = {Path(r['path']).name:r['sha256'] for r in (self.proposal_ref,self.model_ref,self.order_ref)}

    def raw(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value)
        return prep.reference(path)

    def json(self,path,value):
        return self.raw(path,prep.encoded(value))

    def build(self,**kwargs):
        with patch.object(prep,'PINNED_FULL300_SHA256',self.tasks['sha256']):
            return prep.build_drafts(audit=self.audit,runtime=self.runtime,pilots=self.pilots,output_root=self.output,
                native_arguments_reference=self.parsed_ref,expected_inputs=self.expected,**kwargs)

    def enrich(self, mutate=None):
        """Use the real binder and exclusive bundle writer, not hand-made paths."""
        from openwebrl.action_credit_execution import bind_bundles
        from prepare_action_credit_pilots import write_bundles
        target=self.root/'enriched-pilots'
        self.argv=self.json(self.runtime/'native-argv.json', ['--hf-checkpoint',str(self.model_root),
            '--save',str(self.root/'TEMPLATE'), '--custom-config-path',self.config['path'],
            '--wandb-project','openwebrl'])
        bundles={}
        for arm in prep.ARM_IDS:
            files={p.name:json.loads(p.read_text()) for p in (self.pilots/arm).glob('*.json')}
            files['manifest.json']['campaign_id']='CPU-'+arm
            value=files['manifest.json']
            manifest=SimpleNamespace(sha256=prep.digest(value),to_dict=lambda value=value:deepcopy(value))
            files['worker.common.json']['browser_config']=deepcopy(files['runtime-pins.json']['browser_config'])
            files['launch.template.json']=dict(store_root=str(target/arm/'ledger'))
            files['environment.json']=dict(WANDB_DIR=str(target/arm/'cache/wandb'))
            files['readiness.json']=dict(bundle_status='draft_pending_api_approval')
            bundles[arm]=dict(manifest=manifest,files=files)
        bundles=bind_bundles(bundles,native_argv_ref=self.argv,native_config_ref=self.config)
        if mutate:mutate(bundles)
        with patch('prepare_action_credit_pilots.CampaignStore.create',side_effect=AssertionError('No ledger')):
            write_bundles(bundles,target)
        self.pilots=target
        self.source_ref=prep.reference(target/prep.ARM_IDS[0]/'source-inventory.json')
        self.parsed.update(native_argv=self.argv,source_inventory=self.source_ref)
        self.parsed_ref=self.json(self.root/'parsed-enriched.json',self.parsed)

    def test_six_exact_configs_are_isolated_and_unapproved(self):
        plan = self.build(); index=plan['index']
        self.assertFalse(self.output.exists())
        self.assertIsNone(index['verification_manifest'])
        self.assertFalse(index['authorizes_execution']);self.assertEqual(index['status'],prep.STATUS)
        refs=[index['shared0_worker_configuration']]+[v['iter10_worker_configuration'] for v in index['arms'].values()]
        cache_roots=[]
        for ref in refs:
            name=str(Path(ref['path']).relative_to(self.output));config=json.loads(plan['files'][name])
            self.assertIsNone(config['accounting_manifest_sha256']);self.assertIsNone(config['judge_protocol'])
            self.assertIsNone(config['approval_reference'])
            self.assertEqual(config['protocol'],prep.evaluation_protocol())
            for k,v in prep.native_evaluation_controls().items():self.assertEqual(config['native_args'][k],v)
            self.assertEqual(config['native_args']['max_steps'],30)
            self.assertEqual(config['native_args']['rollout_max_response_len'],4096)
            self.assertEqual(config['native_args']['sglang_server_concurrency'],16)
            self.assertIn('rollout_stop',config['native_args'])
            self.assertEqual(config['browser_config']['local_process']['server_module'],'openwebrl.docker.env_server')
            self.assertEqual(config['environment']['WANDB_PROJECT'],'openwebrl-evals')
            self.assertNotIn('WANDB_RUN_ID',config['environment']);self.assertNotIn('HOME',config['environment'])
            self.assertNotIn('OPENAI_API_KEY',config['environment'])
            self.assertTrue(config['environment']['TMPDIR'].startswith('/tmp/acev-'))
            self.assertLess(len(config['environment']['TMPDIR']),40)
            self.assertTrue(config['environment']['PLAYWRIGHT_BROWSERS_PATH'].endswith('/browsers'))
            cache_roots.append(config['environment']['TMPDIR'])
        self.assertEqual(len(set(cache_roots)),6)
        self.assertEqual(index['source_inventory'],self.source_ref)
        self.assertTrue(all(v['accounting_manifest_sha256'] is None for v in index['arms'].values()))

    def test_draft_config_rejected_before_authority_or_dispatch(self):
        plan=self.build();config=json.loads(plan['files']['shared-initial/worker.configuration.json'])
        store=SimpleNamespace(manifest=SimpleNamespace(sha256='a'*64))
        with patch('openwebrl.action_credit_eval_executor._approval',side_effect=AssertionError('must reject before authority')):
            with self.assertRaisesRegex(ValueError,'billing store'):
                _validate_config(store,{},config)

    def test_missing_verification_selection_remains_null_not_guessed(self):
        plan=self.build();body=json.loads(plan['files']['shared-verification/manifest.template.json'])
        args=body['builder_arguments']
        self.assertEqual(args['excluded_training_task_ids'],[f'train-{i}' for i in range(2102)])
        for key in ('verification_task_ids','verification_protocol','shared_initial_max_attempts_per_task','verification_max_attempts_per_task'):
            self.assertIsNone(args[key])
        self.assertIsNone(body['selection_reference'])
        with self.assertRaises(Exception):VerificationManifest.from_dict(body)

    def test_explicit_source_bound_selection_uses_real_builder_without_authority(self):
        fields=dict(verification_task_ids=['train-3'],verification_protocol=dict(optimizer_updates=0,
            policy_sha256=self.model_ref['sha256'],browser='local_process',judge='gpt-4.1/action_history',max_turns=15),
            shared_initial_max_attempts_per_task=2,verification_max_attempts_per_task=1)
        receipt=self.json(self.root/'selection-authority.json',dict(selection=fields,
            user_authorization_text='CPU fixture explicit selection',user_authorization_reference='CPU fixture'))
        selection=self.json(self.root/'selection.json',dict(schema_version=1,task_order_reference=self.order_ref,
            selection_authority_reference=receipt,**fields))
        plan=self.build(verification_selection_reference=selection)
        manifest=VerificationManifest.from_dict(json.loads(plan['files']['shared-verification/manifest.json']))
        self.assertEqual(manifest.to_dict()['budget_caps']['api_microdollars'],25_000_000)
        self.assertEqual(manifest.to_dict()['shared_initial_caps']['api_requests'],600)
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(Exception,'approval is pending'):approved_authority(manifest)

    def test_missing_native_constructor_field_rejected(self):
        del self.parsed['native_args']['rollout_stop']
        self.parsed_ref=self.json(self.root/'parsed-missing.json',self.parsed)
        with self.assertRaisesRegex(ValueError,'GenerateState'):self.build()

    def test_changed_cpu_snapshot_source_or_gpu_claim_rejected(self):
        for key,value in [('cuda_initialized',True),('physical_native_acceptance',True),('judge_calls',1),
                          ('source_inventory',dict(self.source_ref,path='/tmp/another-source.json'))]:
            modified=dict(self.parsed,**{key:value});self.parsed_ref=self.json(self.root/(key+'.json'),modified)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'CPU provenance'):self.build()

    def test_source_tamper_and_partial_original_tokenizer_are_rejected(self):
        path=self.source/prep.SERVER;path.write_text('changed')
        with self.assertRaisesRegex(ValueError,'Source bytes changed'):self.build()
        path.write_text('CPU source fixture\n')
        for arm in prep.ARM_IDS:
            pins=deepcopy(self.pins);pins['tokenizer_files'].pop()
            self.json(self.pilots/arm/'runtime-pins.json',pins)
        with self.assertRaisesRegex(ValueError,'Every original tokenizer'):self.build()

    def test_no_overwrite_no_ledger_and_changed_input_before_write(self):
        plan=self.build()
        ref=prep.write_drafts(plan)
        self.assertEqual(ref,prep.reference(self.output/'index.json'))
        self.assertFalse(Path(plan['index']['verification_store_root']).exists())
        with self.assertRaisesRegex(ValueError,'overwrite'):prep.write_drafts(plan)
        self.output=self.root/'second'
        plan=self.build();Path(self.argv['path']).write_text('[]')
        # Snapshot's own immutable references remain retained; writer must recheck
        # their actual native argv/config bytes too before publishing templates.
        with self.assertRaises(ValueError):prep.write_drafts(plan)
        self.assertFalse(self.output.exists())

    def test_home_or_symlink_output_rejected(self):
        self.output=Path('/gpfs/home/zixianma/cannot-write')
        with self.assertRaisesRegex(ValueError,'Home'):self.build()
        alias=self.root/'alias';alias.symlink_to('/gpfs/home/zixianma',target_is_directory=True)
        self.output=alias/'cannot-write'
        with self.assertRaisesRegex(ValueError,'Home'):self.build()

    def test_real_enriched_five_arm_bundles_allow_only_bound_private_paths(self):
        self.enrich();plan=self.build()
        for arm, row in plan['index']['arms'].items():
            self.assertEqual(row['execution_inputs']['prepared'],prep.reference(self.pilots/arm/'PREPARED.json'))
            binding=prep.read(row['execution_inputs']['execution_binding'])
            self.assertEqual(binding['paths']['browser_logs'],str(self.pilots/arm/'browser-logs'))
        prep.write_drafts(plan)
        self.assertFalse(any((self.pilots/arm/'ledger').exists() for arm in prep.ARM_IDS))

    def test_enriched_path_change_without_prepared_receipt_is_rejected(self):
        self.enrich()
        path=self.pilots/'gigpo/runtime-pins.json';value=json.loads(path.read_text())
        value['browser_config']['local_process']['log_dir']=str(self.root/'elsewhere')
        self.json(path,value)
        with self.assertRaisesRegex(ValueError,'Prepared execution input changed'):self.build()

    def test_enriched_resealed_browser_science_change_is_rejected(self):
        def mutate(bundles):
            for name in ('runtime-pins.json','worker.common.json'):
                bundles['gigpo']['files'][name]['browser_config']['width']=1200
        self.enrich(mutate)
        with self.assertRaisesRegex(ValueError,'beyond verified private browser paths'):self.build()

    def test_enriched_resealed_cross_arm_paths_are_rejected(self):
        def mutate(bundles):
            source=bundles['tree_grpo']['files']['execution-binding.json']['paths']['browser_logs']
            for name in ('runtime-pins.json','worker.common.json'):
                bundles['gigpo']['files'][name]['browser_config']['local_process']['log_dir']=source
            bundles['gigpo']['files']['execution-binding.json']['paths']['browser_logs']=source
        self.enrich(mutate)
        with self.assertRaisesRegex(ValueError,'not bound to its own arm'):self.build()

    def test_enriched_mutation_between_plan_and_write_is_rejected(self):
        self.enrich();plan=self.build()
        path=self.pilots/'gigpo/native-config.json';value=json.loads(path.read_text())
        value['max_steps']=99;self.json(path,value)
        with self.assertRaisesRegex(ValueError,'changed before write'):prep.write_drafts(plan)
        self.assertFalse(self.output.exists())

    def test_unbound_legacy_drafts_do_not_get_path_exemption(self):
        path=self.pilots/'gigpo/runtime-pins.json';value=json.loads(path.read_text())
        value['browser_config']['local_process']['log_dir']=str(self.root/'unbound')
        self.json(path,value)
        with self.assertRaisesRegex(ValueError,'beyond verified private browser paths'):self.build()


class SharedSourceOwnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_arm_paths_reuse_one_canonical_shared_cohort(self):
        root=Path(tempfile.mkdtemp(prefix='action-eval-shared-source-',dir='/tmp'))
        def saved(name,value):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(prep.encoded(value));return prep.reference(p)
        canonical=saved('canonical-source.json',dict(source_root='/tmp/CPU-source',files={'CPU': '0'*64}))
        a=saved('arm-a/source.json',prep.read(canonical));b=saved('arm-b/source.json',prep.read(canonical))
        tasks=saved('tasks.json',dict(task_ids=[f'eval-{i}' for i in range(300)]))
        original=saved('original.json',dict(files=['CPU only']))
        manifest=SimpleNamespace(sha256='a'*64,to_dict=lambda:dict(source_inventory=canonical))
        store=SimpleNamespace(manifest=manifest,register_evaluation_cohort=lambda *a:None)
        calls=[]
        class Executor:
            async def run_or_resume(self,store,ref,path,context):
                calls.append(prep.read(ref));return saved('cpu-result.json',dict(cpu_only=True))
        executor=Executor()
        def owner(source,out):
            return Full300EvaluationOwner(task_manifest_ref=tasks,original_actor_inventory_ref=original,
                source_inventory_ref=source,shared_initial_root=root/'shared',arm_output_root=root/out,
                verification_store=store,executor=executor,diagnostic_only=True)
        def review(store,ref,result,**kwargs):
            return dict(request=ref,result=result,verified_complete=False,verification_kind='cpu_mock')
        contract=SimpleNamespace(update_id=0,policy_sha256=original['sha256'],train_task_ids=['train-only'])
        with patch('openwebrl.action_credit_evaluation.verify_full300',side_effect=review):
            first=await owner(a,'out-a').initial(store,contract,{})
            second=await owner(b,'out-b').initial(store,contract,{})
            self.assertEqual(first,second);self.assertEqual(len(calls),1)
            self.assertEqual(calls[0]['source_inventory'],canonical)
            changed=saved('changed-source.json',dict(source_root='/tmp/other',files={'CPU':'0'*64}))
            with self.assertRaisesRegex(ValueError,'source inventory differs'):
                await owner(changed,'out-c').initial(store,contract,{})
            self.assertEqual(len(calls),1)
            # Iteration10 remains bound to its own arm's source reference.
            await owner(b,'out-b')._cohort(store,10,original,original['sha256'],{})
            self.assertEqual(calls[-1]['source_inventory'],b)


if __name__ == '__main__':
    unittest.main()
