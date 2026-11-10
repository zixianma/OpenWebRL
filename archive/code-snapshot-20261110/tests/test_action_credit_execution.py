"""CPU contract tests only; no scheduler/GPU/browser/API acceptance."""
import asyncio
import importlib.util
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch

from openwebrl import action_credit_execution as execution


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        # Retain all temporary fixtures; no production artifact cleanup.
        self.root = Path(tempfile.mkdtemp(prefix='action-execution-test-', dir='/tmp'))

    def file(self, name, value):
        path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(execution.encoded(value))
        return execution.boot.artifact(path)

    def bundles(self):
        result = {}
        for arm, sha in [('chain_outcome_grpo','a'*64),('gigpo','b'*64)]:
            root = self.root/'pilots'/arm
            manifest = SimpleNamespace(sha256=sha, to_dict=lambda arm=arm:dict(arm_id=arm,campaign_id='cpu-'+arm))
            browser = dict(local_process=dict(log_dir='/tmp/old',port_lock_dir='/tmp/oldlocks'))
            result[arm] = dict(manifest=manifest, files={
                'launch.template.json':dict(store_root=str(root/'ledger')),
                'environment.json':dict(WANDB_DIR=str(root/'cache/wandb'),TMPDIR=str(root/'cache/tmpdir')),
                'worker.common.json':dict(browser_config=browser),
                'runtime-pins.json':dict(browser_config=json.loads(json.dumps(browser))),
                'source-inventory.json':dict(source_root='/tmp/cpu-source',files={})})
        return result

    def bind(self, bundles=None, argv=None):
        argv = argv or ['--save','/tmp/TEMPLATE','--custom-config-path','/tmp/config','--wandb-project','openwebrl',
                        '--colocate','--no-offload-train','--no-offload-rollout']
        return execution.bind_bundles(bundles or self.bundles(),
            native_argv_ref=self.file('base-argv.json',argv),
            native_config_ref=self.file('base-config.json',dict(max_steps=15,path_to_save_generated_samples='')))

    def test_two_arms_bind_distinct_outputs_without_changing_science_or_writing(self):
        initial = self.bundles(); bound = self.bind(initial)
        self.assertFalse((self.root/'pilots').exists())
        outputs = []
        for arm, bundle in bound.items():
            files = bundle['files']; binding = files['execution-binding.json']
            self.assertEqual(bundle['manifest'].sha256,initial[arm]['manifest'].sha256)
            self.assertEqual(files['native-config.json']['max_steps'],15)
            self.assertEqual(binding['original_iteration'],0)
            self.assertFalse(binding['cpu_parse_is_acceptance'])
            self.assertEqual(files['execution.template.json']['status'],'draft_not_executable')
            for key in ('launch','job_id','attempt_ledger','storage_forecast','verification_manifest'):
                self.assertIsNone(files['execution.template.json'][key])
            argv = files['native-argv.json']
            self.assertIn('--no-offload-train',argv);self.assertIn('--no-offload-rollout',argv)
            self.assertEqual(argv[argv.index('--save')+1],binding['paths']['save'])
            self.assertEqual(files['worker.common.json']['browser_config']['local_process']['log_dir'],binding['paths']['browser_logs'])
            self.assertEqual(files['environment.json']['TMPDIR'],binding['paths']['scratch'])
            outputs.append(binding['paths']['save'])
        self.assertEqual(len(set(outputs)),2)
        self.assertEqual(initial['gigpo']['files']['worker.common.json']['browser_config']['local_process']['log_dir'],'/tmp/old')

    def test_binding_ref_hash_is_exact_preparation_encoding(self):
        bundle = self.bind()['gigpo']['files']
        self.assertEqual(bundle['execution.template.json']['binding'],
            execution.planned_ref(bundle['execution.template.json']['bundle']+'/execution-binding.json',bundle['execution-binding.json']))
        self.assertEqual(bundle['execution-binding.json']['native_argv'],
            execution.planned_ref(bundle['execution.template.json']['bundle']+'/native-argv.json',bundle['native-argv.json']))

    def test_real_five_arm_preparer_aliases_are_separated_before_binding(self):
        path=Path(__file__).with_name('test_prepare_action_credit_pilots.py')
        spec=importlib.util.spec_from_file_location('execution_preparer_fixture',path)
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        owner=fixture.PilotPreparationTests()
        retained=SimpleNamespace(name=tempfile.mkdtemp(prefix='action-execution-five-',dir='/tmp'),cleanup=lambda:None)
        with patch.object(fixture.tempfile,'TemporaryDirectory',return_value=retained):
            owner.setUp()
        actual=owner.build()
        # This reproduces the real producer's aliases, unlike two independent mocks.
        arms=list(actual)
        self.assertIs(actual[arms[0]]['files']['runtime-pins.json'],actual[arms[1]]['files']['runtime-pins.json'])
        bound=self.bind(actual)
        for arm,bundle in bound.items():
            expected=str(owner.output/arm/'browser-logs')
            for name in ('worker.common.json','runtime-pins.json'):
                self.assertEqual(bundle['files'][name]['browser_config']['local_process']['log_dir'],expected)
            self.assertNotEqual(actual[arm]['files']['worker.common.json']['browser_config']['local_process']['log_dir'],expected)

    def test_existing_bundle_cannot_be_rebound(self):
        bundles = self.bundles(); (self.root/'pilots/gigpo').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'fresh preparation'):
            self.bind(bundles)

    def test_ambiguous_argv_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Exactly one'):
            self.bind(argv=['--save','/a','--save','/b','--custom-config-path','/c'])

    def test_short_socket_paths_and_escaping_scratch(self):
        execution.validate_socket_paths(dict(scratch='/tmp/ace-'+'a'*16,ray='/tmp/ace-'+'a'*16+'/r'))
        for scratch,ray in [('/gpfs/home/zixianma/cache','/gpfs/home/zixianma/cache/r'),
                            ('/tmp/ace-'+'a'*16,'/tmp/other'),('/tmp/ace-'+'a'*120,'/tmp/ace-'+'a'*120+'/r')]:
            with self.assertRaises(ValueError): execution.validate_socket_paths(dict(scratch=scratch,ray=ray))

    def test_draft_rejected_before_source_or_owners_or_native(self):
        ref=self.file('draft.json',dict(status='draft_not_executable'))
        with (patch.object(execution.boot,'assemble') as assemble,
              patch.object(execution,'build_owners') as owners,
              patch.object(execution,'full_native_parse') as native):
            with self.assertRaisesRegex(ValueError,'Draft execution'):
                execution.execute(ref)
            assemble.assert_not_called();owners.assert_not_called();native.assert_not_called()

    def test_bound_label_does_not_authorize_null_input(self):
        template = self.bind()['gigpo']['files']['execution.template.json']
        template.update(status='bound_for_execution',job_id='123')
        ref=self.file('nulls.json',template)
        with (patch.object(execution.boot,'assemble') as assemble,
              patch.object(execution,'_json',wraps=execution._json) as read):
            with self.assertRaises(ValueError): execution.load_execution(ref)
            assemble.assert_not_called()
            read.assert_called_once_with(ref)

    def test_scheduler_deadline_requires_actual_running_exact_resources(self):
        row='JobId=123 JobState=RUNNING NumNodes=1 NumCPUs=64 AllocTRES=cpu=64,mem=960G,gres/gpu=8,gres/gpu:h200=8 EndTime=2030-01-01T00:00:00'
        self.assertEqual(execution.scheduler_deadline(row,'123'),1893456000.)
        for old,new in [('RUNNING','PENDING'),('960G','480G'),('NumNodes=1','NumNodes=2'),
                        ('gres/gpu=8','gres/gpu=16'),('h200','h100'),('2030-01-01T00:00:00','Unknown')]:
            with self.assertRaises(ValueError): execution.scheduler_deadline(row.replace(old,new),'123')

    def boundary(self, *, deadline=1050, used=80, cap=880, running=('123',)):
        ref=self.file('accounting.json',dict(attempts=[dict(job_id=x,state='RUNNING') for x in running]))
        provider=Mock(return_value=ref)
        owner=execution.AllocationBoundary(provider,manifest=SimpleNamespace(),job_id='123',deadline_unix=deadline,
            output_root=self.root/'accounting-evidence')
        audit=dict(observed_unix=1000,gpu_seconds_used=used,approved_gpu_seconds_cap=cap)
        return owner,provider,audit

    def test_dynamic_cap_includes_prior_attempts_and_query_latency(self):
        owner,provider,audit=self.boundary()
        process=SimpleNamespace(returncode=0,stdout='CPU scheduler fixture',stderr='')
        with (patch.object(execution,'audit_allocation',return_value=audit),patch.object(execution.time,'time',return_value=1010),
              patch.object(execution.subprocess,'run',return_value=process) as query,
              patch.object(execution,'scheduler_deadline',return_value=1100)):
            owner();provider.assert_called_once()
            self.assertEqual(query.call_args.args[0],['scontrol','show','job','-o','123'])
            self.assertEqual(query.call_args.kwargs['env']['TZ'],'UTC')
            owner.deadline=1100
            with self.assertRaisesRegex(ValueError,'remaining all-attempt cap'): owner()
            owner.deadline=1050
            with patch.object(execution,'scheduler_deadline',return_value=1049):
                with self.assertRaises(ValueError): owner()

    def test_other_active_writer_allocation_is_rejected(self):
        owner,_,audit=self.boundary(running=('123','124'))
        with patch.object(execution,'audit_allocation',return_value=audit),patch.object(execution.subprocess,'run') as query:
            with self.assertRaisesRegex(ValueError,'one active attempt'):owner()
            query.assert_not_called()

    def test_full_parser_is_real_entry_and_restores_argv_on_failure(self):
        called=[]; module=ModuleType('slime.utils.arguments')
        def parse():
            called.extend(sys.argv)
            raise RuntimeError('CPU modeled hardware validation failure')
        module.parse_args=parse; previous=sys.argv
        with patch.dict(sys.modules,{'slime.utils.arguments':module}):
            with self.assertRaises(RuntimeError):execution.full_native_parse(['--real-native-input'])
        self.assertIs(sys.argv,previous)
        self.assertEqual(called,['action-credit-production','--real-native-input'])

    def test_storage_requires_shared_and_scratch_coverage(self):
        arm=self.root/'arm';arm.mkdir()
        accounting=self.file('arm/resource-accounting.json',{})
        assembly=SimpleNamespace(root=arm,store=object())
        verification=SimpleNamespace(root=self.root/'verification')
        request=dict(shared_initial_root=str(self.root/'shared'))
        binding=dict(paths=dict(scratch='/tmp/ace-'+'a'*16))
        storage=SimpleNamespace(_forecast=lambda store:(dict(resource_accounting_ref=accounting),[arm],None))
        with self.assertRaisesRegex(ValueError,'omits arm/shared'):
            execution.validate_storage_coverage(request,assembly,binding,verification,{},storage)
        roots=[arm,verification.root,self.root/'shared',Path(binding['paths']['scratch'])]
        storage._forecast=lambda store:(dict(resource_accounting_ref=accounting),roots,None)
        execution.validate_storage_coverage(request,assembly,binding,verification,{},storage)

    def test_wrong_job_fails_before_owner_construction(self):
        request=dict(job_id='123');assembly=SimpleNamespace()
        with (patch.object(execution,'load_execution',return_value=(request,assembly,None,None,None)),
              patch.dict(os.environ,{'SLURM_JOB_ID':'124'}),patch.object(execution,'build_owners') as owners):
            with self.assertRaisesRegex(ValueError,'registered allocation'):execution.execute({})
            owners.assert_not_called()

    def owned_fixture(self):
        paths=dict(scratch=str(self.root/'scratch'),ray=str(self.root/'scratch/r'),
                   save=str(self.root/'native'),generated=str(self.root/'generated'),execution=str(self.root/'results'))
        argv=self.file('native-argv.json',['--CPU-modeled-only'])
        assembly=SimpleNamespace(store=object(),source={},environment={},launch=SimpleNamespace(deadline_unix=4102444800,execution_id='cpu'))
        return assembly,dict(paths=paths,native_argv=argv)

    def test_native_parse_failure_never_initializes_ray(self):
        assembly,binding=self.owned_fixture();events=[]
        owners=(lambda:events.append('accounting'),lambda *x:events.append('storage'),object())
        with (patch.object(execution,'build_owners',return_value=owners),
              patch.object(execution,'validate_storage_coverage',side_effect=lambda *x:events.append('coverage')),
              patch.object(execution,'apply_environment',side_effect=lambda x:events.append('env')),
              patch.object(execution,'deadline_scope',return_value=nullcontext()),
              patch.object(execution.boot,'verify_native_runtime_sources',side_effect=lambda x:events.append('sources')),
              patch.object(execution,'full_native_parse',side_effect=RuntimeError('CPU hardware check failure')),
              patch.dict(sys.modules,{'ray':Mock()})):
            with self.assertRaisesRegex(RuntimeError,'hardware check'):
                execution._execute_owned({}, {}, assembly,binding,None,{})
            self.assertEqual(events,['accounting','coverage','storage','env','sources'])
            sys.modules['ray'].init.assert_not_called()

    def test_native_train_failure_still_shuts_down_owned_ray(self):
        assembly,binding=self.owned_fixture();ray=Mock();ray.is_initialized.return_value=False
        args=SimpleNamespace(save=binding['paths']['save'],path_to_save_generated_samples=binding['paths']['generated'])
        with (patch.object(execution,'build_owners',return_value=(Mock(),Mock(),object())),
              patch.object(execution,'validate_storage_coverage'),patch.object(execution,'apply_environment'),
              patch.object(execution,'deadline_scope',return_value=nullcontext()),
              patch.object(execution.boot,'verify_native_runtime_sources'),patch.object(execution.boot,'validate_native_profile'),
              patch.object(execution,'full_native_parse',return_value=args),
              patch.object(execution.boot,'train',side_effect=RuntimeError('CPU modeled train failure')),
              patch.dict(sys.modules,{'ray':ray})):
            with self.assertRaisesRegex(RuntimeError,'train failure'):
                execution._execute_owned({}, {}, assembly,binding,None,{})
        ray.shutdown.assert_called_once()
        self.assertEqual(ray.init.call_args.kwargs['address'],'local')
        self.assertEqual(ray.init.call_args.kwargs['num_gpus'],8)

    def test_environment_rejects_home_and_legacy_bypass(self):
        with self.assertRaises(ValueError): execution.apply_environment(dict(HOME='/tmp/override'))
        with patch.dict(os.environ,{'OPENWEBRL_ACTION_CREDIT_NATIVE_GATE':'1'}):
            with self.assertRaisesRegex(ValueError,'legacy'):execution.apply_environment({})


if __name__=='__main__':unittest.main()
