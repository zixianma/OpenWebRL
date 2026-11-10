"""Preparation-only fixtures; no scheduler/API/browser/GPU acceptance."""
import importlib.util
import asyncio
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

_PATH = Path(__file__).resolve().parents[1]/'scripts/prepare_action_credit_pilots.py'
_SPEC = importlib.util.spec_from_file_location('prepare_action_credit_pilots', _PATH)
prep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep)


class PilotPreparationTests(unittest.TestCase):
    def setUp(self):
        # Entire study fixture is local. No private runtime data, model weights,
        # credentials or production acceptance are required by these tests.
        self.temp = tempfile.TemporaryDirectory(prefix='action-credit-prepare-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        (self.source/'openwebrl').mkdir(parents=True)
        (self.source/'scripts').mkdir()
        sources = list((prep.REPO/'openwebrl').glob('action_credit*.py'))
        sources += [prep.REPO/'openwebrl'/name for name in
                    ('generate_browser.py', 'reward_browser.py', 'arm_branch_browser.py', 'arm_branch_protocol.py',
                     'rl_recipe.py', 'controlled_sft_worker.py', 'controlled_browser_server.py', 'response_format.py',
                     'eval_monitor.py')]
        for path in sources:
            shutil.copyfile(path, self.source/'openwebrl'/path.name)
        shutil.copyfile(prep.REPO/'scripts/check_storage_quota.py',self.source/'scripts/check_storage_quota.py')
        (self.source/'bootstrap.py').write_text('# CPU fixture; never executable production acceptance\n')
        self.audit=self.root/'audit';self.audit.mkdir()
        token_names = {'tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','chat_template.json','chat_template.jinja'}
        processor_names = {'config.json','processor_config.json','preprocessor_config.json','video_preprocessor_config.json'}
        model_root=self.root/'model';model_root.mkdir()
        model_files=[]
        for name in sorted(token_names|processor_names|{'model.safetensors'}):
            path=model_root/name;path.write_text('CPU fixture: '+name)
            model_files.append(dict(prep.reference(path),size_bytes=path.stat().st_size))
        model=dict(model_root=str(model_root),files=model_files,weights_rehashed=True,
                   fresh_native_optimizer_scheduler_required=True)
        dataset=self.root/'fixture.parquet';dataset.write_text('CPU dataset fixture; not a Parquet claim')
        tasks=['task-'+str(i) for i in range(2102)]
        order=dict(original_manifest_task_order=tasks,source_sha256=prep.hash_file(dataset),
            ordered_source_indices=list(range(2102)),groups_per_iteration=48,outcome_dependent_replacement=False,
            iterations=[dict(iteration=u+1,groups=[dict(group_id=f'update{u}-group{g}',
                task_id=tasks[48*u+g],source_index=48*u+g) for g in range(48)]) for u in range(10)])
        caps=dict(gpus=8,gpu_type='H200',cpus=64,memory_gib=960,total_allocation_seconds_all_attempts=86400,
            gpu_hours_cap=192,judge_http_attempts_cap_proposed=10000,judge_usd_cap_proposed=200,
            browser_dispatches_cap=20000,local_actor_requests_cap=300000,browser_concurrency=64,
            protective_storage_bytes_cap=5*1024**4,includes_full300_iteration10_and_all_retries=True,
            stop_at_first_cap=True,residuals_not_transferable=True)
        common=dict(seed=42,groups_per_iteration=48,independent_chains_per_group=8,
            fresh_optimizer_scheduler_rng_cursor=True,train_data=str(dataset),train_data_sha256=prep.hash_file(dataset),
            initial_model_path=str(model_root),model_sha256_from_download_manifest=prep.hash_file(model_root/'model.safetensors'),
            task_manifest_label='CPU fixture only',optimizer=dict(name='Adam',lr=1e-6))
        proposal=dict(pilot_caps={arm:dict(caps) for arm in prep.ARM_IDS},target_completed_collection_iterations=10,
            common_science_recommendation=common,authorization=dict(gpu_text='CPU fixture only, no allocation'),
            storage=dict(global_floor_bytes=2*1024**4),evaluation=dict(tasks=300,iterations=[0,10]))
        configs={arm:dict(arm_id=arm,clip_high=.28,clip_low=.2,dual_clip=None,
            gigpo_anchor_mode='state_history_horizon',gigpo_episode_weighting='trajectory',gigpo_gamma=1.,
            gigpo_step_weight=1. if arm=='gigpo' else 0.,local_coefficient=1. if arm=='continuation_q' else 0.,
            normalization='mean_std',reduction='response_mean',singleton='zero') for arm in prep.ARM_IDS}
        for name,value in zip(prep.INPUT_NAMES[:3],(proposal,order,configs)):
            (self.audit/name).write_bytes(prep.encoded(value))
        update=dict(schema='action_credit_update_protocol_v2',checkpoint=dict(cpu_fixture=True),
            pins={name:prep.hash_file(self.audit/name) for name in prep.INPUT_NAMES[:3]})
        failure=dict(schema_version=2,tree_pair=dict(version='prescheduled_slot_k_v2',m=2,n=2,l=2,k=4))
        for name,value in zip(prep.INPUT_NAMES[3:],(update,failure,model)):
            (self.audit/name).write_bytes(prep.encoded(value))
        self.expected={name:prep.hash_file(self.audit/name) for name in prep.INPUT_NAMES}
        def files(names):
            return [{k:f[k] for k in ('path','sha256')} for f in model['files'] if Path(f['path']).name in names]
        binary=self.root/'cpu-chromium-fixture';binary.write_bytes(b'CPU fixture, never execute')
        browser=dict(mode='local_process',width=1280,height=720,dpr=1,use_screenshot=True,
            use_a11ytree=False,resize_output_coords=True,resize_scale=1000,
            local_process=dict(server_module='openwebrl.arm_branch_browser',
                server_module_sha256=prep.hash_file(self.source/'openwebrl/arm_branch_browser.py'),
                log_dir=str(self.root/'browser-logs'),port_lock_dir=str(self.root/'browser-locks')))
        browser_manifest=dict(viewport=[1280,720],device_scale_factor=1,locale='en-US',timezone='UTC',
            headless=True,proxy=False,stealth=False,launch_extra_args=[],binary_path=str(binary),
            binary_sha256=prep.hash_file(binary),branch_state_version='observable-replay-v1')
        self.pins = dict(schema_version=1, production_source_root=str(self.source),
            entrypoint_files=['bootstrap.py'], tokenizer_files=files(token_names), processor_files=files(processor_names),
            native_args=dict(max_steps=15,context_num_screenshots=1,turn_history_reasoning_mode='full',
                browser_history_reasoning_max_turns=0,browser_include_tool_response=1,
                browser_history_env_feedback_max_chars=4000,rollout_max_context_len=32768,
                max_consecutive_parse_failures=3,serial_action_variant=None,browser_response_format_mode='browser_env'),
            browser_config=browser,browser_manifest=browser_manifest,
            sampling=prep.SAMPLING, max_attempts_per_draw=2)
        self.output = self.root/'pilots'

    def build(self, approvals=None):
        return prep.build_bundles(self.audit, self.output, self.pins, approvals,expected_inputs=self.expected)

    def approval(self, arm):
        receipt=dict(status='approved',campaign_id='action-credit-iter10-20261011-'+arm,
            model='gpt-4.1',api_microdollars_cap=200000000,api_requests_cap=10000,
            user_authorization_text='CPU test fixture only',user_authorization_reference='CPU test fixture only')
        path=self.root/('test-approval-'+arm+'.json');path.write_bytes(prep.encoded(receipt))
        return prep.reference(path)

    def test_dry_build_exact_five_arms_no_writes_or_authority(self):
        from openwebrl.action_credit_driver import DriverLaunch
        bundles = self.build()
        self.assertFalse(self.output.exists())
        self.assertEqual(set(bundles), set(prep.ARM_IDS))
        identities, ledgers, wandb = set(), set(), set()
        for arm,bundle in bundles.items():
            manifest,files = bundle['manifest'],bundle['files']
            identities.add(manifest.sha256)
            ledgers.add(files['launch.template.json']['store_root'])
            wandb.add(files['environment.json']['WANDB_RUN_ID'])
            self.assertEqual(manifest.to_dict()['budget_caps'], dict(gpu_seconds=691200,
                api_microdollars=200000000,api_requests=10000,browser_dispatches=20000))
            self.assertEqual(manifest.to_dict()['initialization']['iteration'],0)
            self.assertEqual(len(manifest.to_dict()['task_splits']),2102)
            self.assertFalse(files['readiness.json']['launch_ready'])
            self.assertIsNone(files['worker.common.json']['approval_reference'])
            self.assertIsNone(files['worker.common.json']['judge_protocol'])
            self.assertEqual(files['readiness.json']['bundle_status'],'draft_pending_api_approval')
            self.assertFalse(files['readiness.json']['ledger_creation_permitted'])
            self.assertNotIn('http_protocol_sha256',manifest.to_dict()['science']['judge'])
            self.assertEqual(files['resource-accounting.json']['wandb']['evaluation_project'],'openwebrl-evals')
            self.assertFalse(files['resource-accounting.json']['cleanup_authorized'])
            with self.assertRaises(ValueError):
                DriverLaunch(**files['launch.template.json']).validate()
        self.assertEqual(tuple(map(len,(identities,ledgers,wandb))), (5,5,5))
        tree = bundles['tree_grpo']['manifest'].to_dict()['science']['algorithm']['leaf_selection']
        self.assertEqual(tree,bundles['tree_outcome_sampling']['manifest'].to_dict()['science']['algorithm']['leaf_selection'])
        self.assertIn('prescheduled', json.dumps(tree))
        q=bundles['continuation_q']['manifest'].to_dict()['science']['algorithm']['q_inventory_protocol']
        self.assertFalse(q['q_continuations_in_outcome_objective'])

    def test_prepare_retry_preserves_ledger_revision_and_attempt_costs(self):
        arm=prep.ARM_IDS[0]
        bundles=self.build({arm:self.approval(arm)})
        receipts=prep.write_bundles(bundles,self.output)
        store=prep.CampaignStore(self.output/arm/'ledger',bundles[arm]['manifest'])
        with store._transaction() as state:
            state['preparation_regression']={'all_attempt_gpu_seconds':17,'must_not_reset':True}
        before=(store.root/'state.json').read_bytes()
        again=prep.write_bundles(bundles,self.output)
        self.assertEqual(receipts,again)
        self.assertEqual(before,(store.root/'state.json').read_bytes())
        self.assertEqual(store.snapshot()['preparation_regression']['all_attempt_gpu_seconds'],17)

    def test_changed_manifest_or_partial_output_is_not_overwritten(self):
        bundles=self.build()
        prep.write_bundles(bundles,self.output)
        self.pins['max_attempts_per_draw']=3
        with self.assertRaisesRegex(ValueError,'differs'):
            prep.write_bundles(self.build(),self.output)
        partial=self.root/'partial'
        (partial/prep.ARM_IDS[0]).mkdir(parents=True)
        self.output=partial
        with self.assertRaisesRegex(ValueError,'Partial preparation'):
            prep.write_bundles(self.build(),partial)

    def test_stale_source_rejected_even_with_matching_runtime_fields(self):
        path=self.source/'openwebrl/action_credit_worker.py'
        path.write_text(path.read_text()+'\n# stale\n')
        with self.assertRaisesRegex(ValueError,'Stale production source'):
            self.build()

    def test_storage_parser_and_evaluation_monitor_are_required_pinned_sources(self):
        bundles=self.build()
        for bundle in bundles.values():
            files=bundle['files']['source-inventory.json']['files']
            for name in ('scripts/check_storage_quota.py','openwebrl/eval_monitor.py'):
                self.assertEqual(files[name],prep.hash_file(prep.REPO/name))
        for name in ('scripts/check_storage_quota.py','openwebrl/eval_monitor.py'):
            path=self.source/name; original=path.read_bytes()
            path.write_bytes(original+b'\n# changed CPU fixture\n')
            with self.assertRaisesRegex(ValueError,'Stale production source'):
                self.build()
            path.write_bytes(original)

    def test_input_pins_not_aliases_or_selected_subsets(self):
        self.pins['tokenizer_files'].pop()
        with self.assertRaisesRegex(ValueError,'every original-model'):
            self.build()

    def test_exact_approval_is_separate_per_arm_and_never_enables_launch(self):
        arm=prep.ARM_IDS[0]
        ref=self.approval(arm)
        path=Path(ref['path']);receipt=json.loads(path.read_text())
        bundles=self.build({arm:ref})
        worker=bundles[arm]['files']['worker.common.json']
        self.assertEqual(worker['approval_reference'],ref)
        self.assertEqual(worker['judge_protocol']['reserved_microdollars_per_http'],2103344)
        protocol=prep.JudgeHTTPProtocol(**worker['judge_protocol'])
        self.assertEqual(bundles[arm]['manifest'].to_dict()['science']['judge']['http_protocol_sha256'],protocol.sha256)
        prep.write_bundles(bundles,self.output)
        store=prep.CampaignStore(self.output/arm/'ledger',bundles[arm]['manifest'])
        from openwebrl.action_credit_worker import _approval
        self.assertEqual(_approval(store,ref,protocol),receipt)
        self.assertFalse(bundles[arm]['files']['readiness.json']['launch_ready'])
        self.assertIsNone(bundles['gigpo']['files']['worker.common.json']['approval_reference'])
        with self.assertRaisesRegex(ValueError,'exact campaign'):
            self.build({'gigpo':ref})
        receipt['api_requests_cap']+=1;path.write_bytes(prep.encoded(receipt))
        with self.assertRaisesRegex(ValueError,'exact campaign'):
            self.build({arm:prep.reference(path)})

    def test_draft_has_no_ledger_and_approval_requires_fresh_root(self):
        draft=self.build()
        prep.write_bundles(draft,self.output)
        for arm in prep.ARM_IDS:
            self.assertFalse((self.output/arm/'ledger').exists())
            receipt=json.loads((self.output/arm/'PREPARED.json').read_text())
            self.assertFalse(receipt['ledger_created'])
        arm=prep.ARM_IDS[0]
        approvals={arm:self.approval(arm)}
        approved=self.build(approvals)
        self.assertNotEqual(draft[arm]['manifest'].sha256,approved[arm]['manifest'].sha256)
        self.assertEqual(draft[arm]['files']['launch.template.json']['initial_contract'],
                         approved[arm]['files']['launch.template.json']['initial_contract'])
        with self.assertRaisesRegex(ValueError,'fresh output root'):
            prep.write_bundles(approved,self.output)
        old_root=self.output
        self.output=self.root/'approved-pilots'
        prep.write_bundles(self.build(approvals),self.output)
        self.assertTrue((self.output/arm/'ledger/state.json').is_file())
        self.assertFalse((old_root/arm/'ledger').exists())
        self.assertEqual(json.loads((old_root/arm/'manifest.json').read_text()),draft[arm]['manifest'].to_dict())

    def test_home_or_symlink_output_rejected_and_no_secret_config(self):
        with self.assertRaises(ValueError):
            prep.build_bundles(self.audit,'/gpfs/home/zixianma/forbidden',self.pins,expected_inputs=self.expected)
        link=self.root/'link';link.symlink_to('/gpfs/home/zixianma')
        with self.assertRaises(ValueError):
            prep.build_bundles(self.audit,link/'forbidden',self.pins,expected_inputs=self.expected)
        self.pins['native_args']['api_key']='synthetic-secret'
        with self.assertRaisesRegex(ValueError,'Credentials'):
            self.build()

    def test_changed_scientific_inputs_or_entrypoint_escape_rejected(self):
        self.pins['sampling']=dict(prep.SAMPLING,temperature=.6)
        with self.assertRaisesRegex(ValueError,'sampling'):
            self.build()
        self.pins['sampling']=prep.SAMPLING
        self.pins['entrypoint_files']=['../outside.py']
        with self.assertRaisesRegex(ValueError,'relative'):
            self.build()

    def test_changed_frozen_input_bytes_rejected_without_private_environment(self):
        path=self.audit/prep.INPUT_NAMES[3]
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'frozen study'):
            self.build()

    def test_native_protocol_and_all_five_manifests_pass_real_offline_admission(self):
        from openwebrl.action_credit_collect import (_admit,EpisodeRequest,FrozenActor,
            prefix_sha256,run_native_episode,physical_verifier_sha256)
        from openwebrl.action_credit_online import plan_online_update
        from openwebrl.action_credit_driver import _contract,_tree_policy
        approvals={arm:self.approval(arm) for arm in prep.ARM_IDS}
        bundles=self.build(approvals);prep.write_bundles(bundles,self.output)
        native=prep.generator_protocol(SimpleNamespace(**self.pins['native_args']))
        cost=dict(gpu_seconds=0,api_requests=0,api_microdollars=0,browser_dispatches=1)
        for arm,b in bundles.items():
            launch=b['files']['launch.template.json'];contract=_contract(launch['initial_contract'])
            store=prep.CampaignStore(launch['store_root'],b['manifest'])
            self.assertEqual(store.manifest.to_dict()['science']['generation']['native_protocol_sha256'],prep.digest(native))
            plan,_=plan_online_update(store,contract,launch['task_order']['path'],tree_policy=_tree_policy(launch['tree_policy']))
            item=plan['draws'][0]
            request=EpisodeRequest(request_id='cpu-request',attempt_id='cpu-attempt',leaf_id=item['leaf_id'],
                **item['spec'],task_data_sha256=prep.digest('cpu-task'),tools_sha256=prep.digest('cpu-tools'),
                environment_policy_sha256=prep.digest('cpu-environment'),
                browser_config_sha256=prep.digest(self.pins['browser_config']),
                browser_manifest_sha256=prep.digest(self.pins['browser_manifest']),
                generator_protocol_sha256=prep.digest(native),prefix_sha256=prefix_sha256(()),generation_seed=42)
            actor=FrozenActor(contract.policy_sha256,'1',prep.digest('CPU restore, no actual proof'),
                              prep.digest('CPU frozen, no actual proof'),prep.digest('fixture://actor'))
            draw,_=_admit(store,contract,request,actor,self.pins['sampling'],15,cost)
            self.assertEqual(store.snapshot()['attempts']['cpu-attempt']['draw_id'],draw)
            changed=dict(self.pins['native_args'],context_num_screenshots=2)
            with self.assertRaisesRegex(ValueError,'native history/context protocol'):
                asyncio.run(run_native_episode(store=store,contract=contract,request=request,actor=actor,
                    args=SimpleNamespace(**changed),sample=None,sampling=self.pins['sampling'],
                    browser_config=self.pins['browser_config'],browser_manifest=self.pins['browser_manifest'],
                    reserved_cost=cost,actor_call=None,weight_version=None,
                    verifier_sha256=physical_verifier_sha256(),generation=object(),capture=object()))

    def test_incomplete_native_history_or_wrong_browser_rejected_before_preparation(self):
        original=dict(self.pins['native_args'])
        self.pins['native_args'].pop('context_num_screenshots')
        with self.assertRaisesRegex(ValueError,'all native'):
            self.build()
        self.pins['native_args']=dict(original,max_steps=14)
        with self.assertRaisesRegex(ValueError,'max_steps=15'):
            self.build()
        self.pins['native_args']=original
        self.pins['browser_config']['width']=1920
        with self.assertRaisesRegex(ValueError,'browser observation'):
            self.build()


if __name__=='__main__':
    unittest.main()
