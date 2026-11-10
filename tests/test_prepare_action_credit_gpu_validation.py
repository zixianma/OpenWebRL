"""Admission/topology checks only; these tests do not certify physical GPUs."""
import copy
import importlib.util
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('credit_gpu_prepare',
    Path(__file__).resolve().parents[1]/'scripts/prepare_action_credit_gpu_validation.py')
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def campaign():
    ledger = dict(total_allocation_seconds_cap=21600, total_gpu_seconds_cap=172800,
                  max_concurrent_gpus=8, attempts=[])
    live = {}
    for job, gpus, seconds, elapsed, state in (
            ('355629',8,21600,0,'CANCELLED'), ('355645',8,21600,0,'CANCELLED'),
            ('400001',2,5400,20,'RUNNING'), ('400002',2,5400,0,'PENDING')):
        ledger['attempts'].append(dict(job_id=job,gpus=gpus,allocated_seconds=seconds,
                                       elapsed_seconds=elapsed,state=state))
        live[job] = dict(gpus=gpus,allocated_seconds=seconds,elapsed_seconds=elapsed,state=state)
    return ledger, live


def rank_report(gpus, tp):
    parity = dict(outside_tolerance=0,elements=100)
    return dict(passed=True,rank=0,tp=tp,dp=gpus//tp,production_enabled=False,
        browser_sampler_acceptance=False,manifest_sha256='manifest',native_gate_source_sha256='gate',
        backend_source_sha256='backend',cases=[dict(arm_id=arm,native_optimizer_update=True,
            tp_oracles=[dict(logprob_max_abs_error=0.)],full_dp_gradient_parity=parity.copy())
            for arm in ('chain_outcome_grpo','tree_outcome_sampling','tree_grpo','continuation_q','gigpo')],
        full_accumulation=dict(global_responses=256,local_microbatches=256//(gpus//tp),optimizer_updated=True),
        ppo_windows=dict(optimizer_updates_in_memory=4,epochs=2,base_occurrence_exposures=520,
            virtual_padding_slots=504,additional_local_forwards=8,durable_checkpoint_committed=False,
            windows=[dict(full_dp_gradient_parity=parity.copy()) for _ in range(4)]),
        stress=dict(context=32768,forward_backward=True),reload=dict(loaded_iteration=101,rng_restored=True))


class SplitNativeValidationTests(unittest.TestCase):
    def test_legacy_profile_and_two_distinct_two_gpu_topologies(self):
        self.assertEqual(gate.resource_profile(),dict(gpus=8,cpus=64,memory_gib=960,
            allocated_seconds=21600,tp_sizes=[2,8]))
        for tp in (1,2):
            self.assertEqual(gate.resource_profile(2,[tp],5400),dict(gpus=2,cpus=16,memory_gib=240,
                allocated_seconds=5400,tp_sizes=[tp]))
        for args in ((1,[1],5400),(2,[8],5400),(2,[2,2],5400),(2,[],5400),(2,[2],21660),(2,[2],601)):
            with self.subTest(args=args), self.assertRaises(ValueError):gate.resource_profile(*args)

    def test_tp1_disables_sequence_parallel_without_mutating_launch_template(self):
        template=['python','train.py','--colocate','--use-rollout-logprobs','--sequence-parallel','--ppo-epochs','2']
        one=gate.configure_native_command(template,1);two=gate.configure_native_command(template,2)
        self.assertNotIn('--sequence-parallel',one);self.assertIn('--sequence-parallel',two)
        self.assertIn('--colocate',template);self.assertNotIn('--colocate',one)
        self.assertEqual(one[one.index('--ppo-epochs')+1],'1')
        with self.assertRaises(ValueError):gate.configure_native_command(template+['--use-wandb'],1)

    def test_both_small_jobs_reserve_full_time_and_charge_cancelled_attempts(self):
        ledger,live=campaign()
        first=gate.campaign_admission(ledger,'400001',live)
        self.assertEqual(first['allocation_seconds_reserved'],10800)
        self.assertEqual(first['gpu_seconds_reserved'],21600)
        self.assertEqual(first['outstanding_gpus'],4)
        ledger['attempts'][0]['elapsed_seconds']=90
        live['355629']['elapsed_seconds']=120
        live['355645']['elapsed_seconds']=30
        live['400002']['state']='RUNNING'
        second=gate.campaign_admission(ledger,'400002',live)
        self.assertEqual(second['allocation_seconds_reserved'],10950)
        self.assertEqual(second['gpu_seconds_reserved'],22800)
        live['355629']['elapsed_seconds']=1
        third=gate.campaign_admission(ledger,'400001',live)
        self.assertEqual(third['attempts'][0]['elapsed_seconds'],90)

    def test_still_pending_old_allocation_cannot_double_reserve_the_budget(self):
        ledger,live=campaign();live['355645']['state']='PENDING'
        with self.assertRaisesRegex(ValueError,'cap exceeded'):gate.campaign_admission(ledger,'400001',live)

    def test_overspent_failed_attempt_is_counted_and_rejected(self):
        ledger,live=campaign();live['355629'].update(state='TIMEOUT',elapsed_seconds=11000)
        with self.assertRaisesRegex(ValueError,'cap exceeded'):gate.campaign_admission(ledger,'400001',live)

    def test_missing_unregistered_duplicate_unknown_and_resource_mismatch_fail_closed(self):
        for mutate in (
            lambda ledger,live:live.pop('355629'),
            lambda ledger,live:ledger['attempts'].append(ledger['attempts'][0]),
            lambda ledger,live:live['400001'].update(state='UNKNOWN'),
            lambda ledger,live:live['400001'].update(gpus=8),
            lambda ledger,live:live['400001'].update(allocated_seconds=5460),
            lambda ledger,live:ledger.update(total_gpu_seconds_cap=200000)):
            ledger,live=campaign();mutate(ledger,live)
            with self.assertRaises(ValueError):gate.campaign_admission(ledger,'400001',live)
        ledger,live=campaign()
        with self.assertRaises(ValueError):gate.campaign_admission(ledger,'999999',live)

    def test_scheduler_parser_uses_requested_tres_for_pending_jobs(self):
        output=('1|RUNNING|30|cpu=16,mem=240G,gres/gpu=2,gres/gpu:h200=2|cpu=16,gres/gpu=2|90|\n'
                '2|PENDING|0||cpu=16,mem=240G,gres/gpu:h200=2|90|\n')
        with patch.object(gate.subprocess,'check_output',return_value=output) as query:
            rows=gate.scheduler_attempts(['1','2'])
        self.assertEqual(rows['2']['gpus'],2);self.assertEqual(rows['1']['allocated_seconds'],5400)
        self.assertIn('-X',query.call_args.args[0])
        with patch.object(gate.subprocess,'check_output',return_value=output+output):
            with self.assertRaises(ValueError):gate.scheduler_attempts(['1','2'])

    def test_all_topologies_require_exact_arm_ids_and_physical_evidence(self):
        for gpus,tp in ((2,1),(2,2),(8,2),(8,8)):
            kwargs=dict(rank=0,gpus=gpus,tp=tp,manifest_sha256='manifest',gate_sha256='gate',backend_sha256='backend')
            report=rank_report(gpus,tp);gate.validate_native_report(report,**kwargs)
            changes=(lambda x:x['cases'][0].update(arm_id='gigpo'),
                lambda x:x.update(dp=99),lambda x:x.update(rank=1),lambda x:x.update(manifest_sha256='old'),
                lambda x:x.update(native_gate_source_sha256='old'),lambda x:x['stress'].update(context=8192),
                lambda x:x['full_accumulation'].update(local_microbatches=1),
                lambda x:x['ppo_windows'].update(epochs=1),lambda x:x['reload'].update(rng_restored=False),
                lambda x:x['cases'][0]['full_dp_gradient_parity'].update(outside_tolerance=1),
                lambda x:x['cases'][0].update(tp_oracles=[]))
            for change in changes:
                changed=copy.deepcopy(report);change(changed)
                with self.subTest(gpus=gpus,tp=tp),self.assertRaises(ValueError):gate.validate_native_report(changed,**kwargs)

    def test_request_freeze_binds_resources_topology_and_campaign_path_without_hash_cycle(self):
        request=dict(selected_operational_cap=gate.resource_profile(2,[1],5400),campaign_ledger='/tmp/ledger')
        original=gate.request_core_sha(request)
        request['prepared_execution_sha256']='execution-proof'
        self.assertEqual(original,gate.request_core_sha(request))
        request['selected_operational_cap']['gpus']=8
        self.assertNotEqual(original,gate.request_core_sha(request))


class MechanicsSnapshotTests(unittest.TestCase):
    def snapshot(self):
        # Retain bounded test fixtures: this workflow forbids artifact deletion.
        root = Path(tempfile.mkdtemp(prefix='action-credit-source-test-', dir='/tmp'))
        package = root/'openwebrl';package.mkdir()
        names = ('action_credit', 'action_credit_advantages', 'action_credit_binding',
            'action_credit_objective', 'action_credit_receipts', 'action_credit_sampling',
            'action_credit_selection', 'action_credit_state', 'action_credit_megatron',
            'action_credit_windows', 'action_credit_native_gate')
        for name in names:
            (package/(name+'.py')).write_text('VALUE = 1\n')
        return root

    def test_copies_and_pins_complete_snapshot_and_local_resources(self):
        root=self.snapshot();package=root/'openwebrl'
        (package/'action_credit_selection.py').write_text(
            'def select():\n    from .action_credit_slot_selection import select\n')
        (package/'action_credit_slot_selection.py').write_text(
            'from openwebrl import action_credit_state\n')
        (package/'action_credit_state.py').write_text(
            'def account():\n    from .action_credit_eval_budget import reserve\n')
        (package/'action_credit_eval_budget.py').write_text('def reserve(): pass\n')
        (package/'controlled_sft_worker.py').write_text('class AttemptJournal: pass\n')
        (package/'env/prompts').mkdir(parents=True)
        (package/'env/prompts/system_prompt.md').write_text('Pinned prompt\n')
        (root/'action-credit-runtime-dependencies.json').write_text('{}\n')
        (package/'.env').write_text('NEVER_COPY=private\n')
        payloads,pin=gate.mechanics_payloads(root)
        for name in ('action_credit_slot_selection.py','action_credit_eval_budget.py',
                     'controlled_sft_worker.py','env/prompts/system_prompt.md'):
            key='openwebrl/'+name
            self.assertEqual(payloads[key],(package/name).read_bytes())
            self.assertEqual(pin['files_sha256'][key],hashlib.sha256(payloads[key]).hexdigest())
        self.assertIn('action-credit-runtime-dependencies.json',payloads)
        self.assertNotIn('openwebrl/.env',payloads)
        self.assertIsNone(pin['native_gate_override'])

    def test_rejects_missing_lazy_dependency_in_both_import_forms(self):
        for statement in ('from .action_credit_slot_selection import select',
                          'from openwebrl import action_credit_slot_selection'):
            root=self.snapshot()
            (root/'openwebrl/action_credit_selection.py').write_text('def select():\n    '+statement+'\n')
            with self.subTest(statement=statement),self.assertRaisesRegex(ValueError,'Missing local mechanics dependency'):
                gate.mechanics_payloads(root)

    def test_explicit_oracle_override_is_pinned_without_changing_snapshot(self):
        root=self.snapshot();original=root/'openwebrl/action_credit_native_gate.py'
        override_root=Path(tempfile.mkdtemp(prefix='action-credit-oracle-test-',dir='/tmp'))
        override=override_root/'action_credit_native_gate.py';override.write_text('VALUE = 2\n')
        payloads,pin=gate.mechanics_payloads(root,native_gate_source=override)
        self.assertEqual(original.read_text(),'VALUE = 1\n')
        self.assertEqual(payloads['openwebrl/action_credit_native_gate.py'],override.read_bytes())
        self.assertEqual(pin['native_gate_override']['sha256'],gate.sha(override))
        self.assertEqual(pin['native_gate_override']['original_sha256'],gate.sha(original))

    def test_nested_relative_imports_are_checked_and_external_symlinks_rejected(self):
        root=self.snapshot();nested=root/'openwebrl/nested';nested.mkdir()
        (nested/'worker.py').write_text('from ..action_credit_state import VALUE\n')
        gate.mechanics_payloads(root)
        outside=Path(tempfile.mkdtemp(prefix='action-credit-external-test-',dir='/tmp'))/'helper.py'
        outside.write_text('VALUE = 3\n')
        (nested/'external.py').symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'escapes the selected source'):
            gate.mechanics_payloads(root)


if __name__=='__main__':unittest.main()
