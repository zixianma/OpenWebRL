import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

SPEC=importlib.util.spec_from_file_location('recover_pass8_test',Path(__file__).resolve().parents[1]/'scripts/recover_sft_pass8_extension.py')
module=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(module)


class RecoveryOverlayTest(unittest.TestCase):
    def test_parser_success_without_orphan_deep_verification_is_not_ready(self):
        recovery={'preserved':{'saved_orphans':[{'path':'saved.json'}]}}
        ready=dict(recovery_manifest_sha256=module.digest(recovery),native_parse_passed=True,
            collector_import_verified=True,saved_orphan_deep_validation_passed=True,
            saved_orphans_deep_verified=1)
        module.validate_readiness(recovery,ready)
        for key,value in [('saved_orphan_deep_validation_passed',False),
                          ('saved_orphans_deep_verified',0),('collector_import_verified',False)]:
            with self.assertRaises(ValueError):module.validate_readiness(recovery,dict(ready,**{key:value}))
        del ready['saved_orphan_deep_validation_passed']
        with self.assertRaises(ValueError):module.validate_readiness(recovery,ready)

    def test_verifier_chain_rejects_changed_science_and_unpinned_previous_collector(self):
        with tempfile.TemporaryDirectory() as directory:
            roots=[Path(directory)/name for name in ('v1','v2','v3')]
            for version,root in enumerate(roots):
                contents={module.COLLECTOR:f'collector-{version}','scientific.py':'unchanged'}
                if version:contents[module.ENTRY]=f'controller-{version}'
                hashes={}
                for name,content in contents.items():
                    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
                    hashes[name]=module.sha(path)
                (root/'reference_manifest.json').write_text(json.dumps({'recipe_files_sha256':hashes}))
            with patch.object(module,'PARENT',roots[0]),patch.object(module,'SUPERSEDED',roots[1]),\
                    patch.object(module,'SOURCE',roots[2]),\
                    patch.object(module,'SUPERSEDED_COLLECTOR_SHA',module.sha(roots[1]/module.COLLECTOR)):
                chain=module.verifier_chain()
                self.assertEqual(set(chain['changed_from_previous']),{module.COLLECTOR,module.ENTRY})
                self.assertTrue(chain['scientific_protocol_unchanged'])
                with patch.object(module,'SUPERSEDED_COLLECTOR_SHA','wrong'):
                    with self.assertRaises(ValueError):module.verifier_chain()
                (roots[2]/'scientific.py').write_text('changed')
                with self.assertRaises(ValueError):module.verifier_chain()

    def test_only_collector_changes_and_new_entrypoint_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            before={module.COLLECTOR:'old','scientific.py':'preserved'}
            contents={module.COLLECTOR:'patched','scientific.py':'preserved',module.ENTRY:'entry'}
            after={}
            for name,content in contents.items():
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
                after[name]=module.sha(path)
            parent=dict(recipe_files_sha256={name:hashlib.sha256(content.encode()).hexdigest() for name,content in before.items()})
            overlay=dict(recipe_files_sha256=after)
            self.assertEqual(module.validate_overlay(parent,overlay,root),{module.COLLECTOR})
            changed=copy.deepcopy(overlay);changed['recipe_files_sha256']['scientific.py']='bad'
            with self.assertRaises(ValueError):module.validate_overlay(parent,changed,root)
            changed=copy.deepcopy(overlay);changed['recipe_files_sha256']['extra']='anything'
            with self.assertRaises(ValueError):module.validate_overlay(parent,changed,root)
            (root/module.COLLECTOR).write_text('later edit')
            with self.assertRaises(ValueError):module.validate_overlay(parent,overlay,root)

    def test_overlay_changes_native_import_source_but_preserves_bound_config(self):
        config=dict(job_id='new',attempt_output='/original/output/attempts/new',
            tasks=str(module.PARENT/'pass8-shard-0.jsonl'),partition_manifest=str(module.PARENT/'pass8-partition.json'),
            output='/original/output',seed=20260930,judge_max_usd=12.5)
        stage=dict(command=['bash',str(module.PARENT/'scripts/run_h200_browser.sh'),'--eval-config',
            str(module.PARENT/'pass8-shard-0-eval.json')],environment={'BROWSER_TRAIN_CONFIG':str(module.PARENT/'screen-sharded-browser.json')})
        launch=dict(collector_config=config,stage=stage)
        base=Mock();base.make_launch.side_effect=lambda *a:copy.deepcopy(launch)
        recovery=dict(shard=0,manifest_path='/receipt',collector_after_sha256='hash',source_manifest_sha256='sourcehash')
        with patch.object(module,'read',return_value=module.stable_config(config)):
            result=module.overlay_launch(base,{},recovery,'new')
        self.assertEqual(result['collector_config'],config)
        self.assertEqual(result['stage']['command'][1],str(module.SOURCE/'scripts/run_h200_browser.sh'))
        self.assertEqual(result['stage']['command'][3],str(module.PARENT/'pass8-shard-0-eval.json'))
        self.assertEqual(result['stage']['environment'],stage['environment'])
        with patch.object(module,'read',return_value=dict(module.stable_config(config),seed=1)):
            with self.assertRaises(ValueError):module.overlay_launch(base,{},recovery,'new')

    def test_slurm_rounding_never_exceeds_remaining_seconds(self):
        self.assertEqual(module.recommended_seconds(36000-2078),33900)
        self.assertLessEqual(2078+module.recommended_seconds(33922),36000)
        with self.assertRaises(ValueError):module.recommended_seconds(599)

    def test_registration_binds_source_receipt_and_existing_campaign_budget(self):
        base=module.load_base();plan=dict(resources=base.RESOURCES,limits=base.LIMITS)
        recovery=dict(shard=0,manifest_path='/receipt',prior_charged_seconds=2078,remaining_scheduler_seconds=33922)
        row=dict(job_id='2',shard=0,time_limit_seconds=33900,recovery_manifest='/receipt',
            recovery_manifest_sha256=module.digest(recovery),source=str(module.SOURCE))
        approval=dict(approved=True,plan_sha256=base.digest(plan),resources=base.RESOURCES,limits=base.LIMITS,
            attempts=[dict(job_id='1',shard=0,time_limit_seconds=36000),row])
        self.assertEqual(module.check_registration(base,plan,recovery,approval,'2',2078),33900)
        for key,value in [('source','wrong'),('recovery_manifest_sha256','wrong'),('time_limit_seconds',33960)]:
            changed=copy.deepcopy(approval);changed['attempts'][-1][key]=value
            with self.assertRaises(ValueError):module.check_registration(base,plan,recovery,changed,'2',2078)
        with self.assertRaises(ValueError):module.check_registration(base,plan,recovery,approval,'2',2077)

    def test_inventory_keeps_orphan_and_every_physical_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shard=root/'shard-0'
            for path,value in [
                (shard/'records/a.json',{'task_id':'a','attempt':5}),
                (shard/'attempts/1/first/screen_records/a.json',{'task_id':'a','attempt':5}),
                (shard/'attempts/1/orphan/screen_records/b.json',{'task_id':'b','attempt':5}),
                (shard/'physical-attempts/first/dispatch.json',{}),
                (shard/'physical-attempts/orphan/dispatch.json',{}),
                (shard/'physical-attempts/interrupted/dispatch.json',{}),
                (shard/'judge-budget/ledger.json',{'calls':2})]:
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            with patch.object(module,'ROOT',root):inventory=module.preserve_inventory(0,'1')
            self.assertEqual(inventory['canonical_records_before'],1)
            self.assertEqual(inventory['physical_dispatches_before'],3)
            self.assertEqual(inventory['saved_records_in_failed_job'],2)
            self.assertEqual([(r['task_id'],r['attempt']) for r in inventory['saved_orphans']],[('b',5)])
            self.assertEqual(len(inventory['immutable_files_sha256']),6)


if __name__=='__main__':unittest.main()
