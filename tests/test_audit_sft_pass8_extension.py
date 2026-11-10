import hashlib
import json
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_sft_pass8_extension as auditor
from openwebrl import arm_task_pass8 as collector_helpers


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class RecoverySourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.parent=self.base/'source-v1';self.recovery=self.base/'source-v2'
        self.before={}
        for name,text in {
            'openwebrl/arm_task_pass8.py':'# original verifier\n',
            'openwebrl/generate_browser.py':'# frozen actor and browser\n',
            'openwebrl/reward_browser.py':'# frozen native judge\n',
        }.items():
            for root in (self.parent,self.recovery):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
            self.before[name]=auditor.sha(self.parent/name)
        write(self.parent/'reference_manifest.json',{'recipe_files_sha256':self.before})
        collector=self.recovery/'openwebrl/arm_task_pass8.py'
        collector.write_text('# repaired verifier accepting native invalid timeout sentinel\n')
        controller=self.recovery/'scripts/recover_sft_pass8_extension.py'
        controller.parent.mkdir();controller.write_text('# isolated recovery controller\n')
        self.after={name:auditor.sha(self.recovery/name) for name in self.before}
        self.after['scripts/recover_sft_pass8_extension.py']=auditor.sha(controller)
        write(self.recovery/'reference_manifest.json',{'recipe_files_sha256':self.after})
        self.evidence=self.base/'physical-attempt/record.json';write(self.evidence,{'valid':False,'reward':None})
        self.plan={'source':str(self.parent),'source_manifest_sha256':auditor.sha(self.parent/'reference_manifest.json')}
        self.receipt_path=self.base/'recovery.json'
        self.receipt=dict(source=str(self.recovery),campaign_plan_sha256=auditor.digest(self.plan),shard=0,
            added_budget=False,scientific_protocol_unchanged=True,
            parent_source_manifest_sha256=self.plan['source_manifest_sha256'],
            source_manifest_sha256=auditor.sha(self.recovery/'reference_manifest.json'),
            collector_before_sha256=self.before['openwebrl/arm_task_pass8.py'],
            collector_after_sha256=self.after['openwebrl/arm_task_pass8.py'],
            recovery_controller_sha256=self.after['scripts/recover_sft_pass8_extension.py'],
            preserved={'immutable_files_sha256':{str(self.evidence):auditor.sha(self.evidence)}})
        self.attempt=dict(job_id='replacement',shard=0,source=str(self.recovery),
            recovery_manifest=str(self.receipt_path),recovery_manifest_sha256=auditor.digest(self.receipt))
        self.approval={'attempts':[{'job_id':'original','shard':0},self.attempt]}
        self.bind_receipt()

    def tearDown(self):
        self.tmp.cleanup()

    def bind_receipt(self):
        write(self.receipt_path,self.receipt)
        self.attempt['recovery_manifest_sha256']=auditor.digest(self.receipt)

    def bind_source_manifest(self):
        write(self.recovery/'reference_manifest.json',{'recipe_files_sha256':self.after})
        self.receipt['source_manifest_sha256']=auditor.sha(self.recovery/'reference_manifest.json')
        self.bind_receipt()

    def test_bound_recovery_selects_patched_verifier_and_preserves_every_input(self):
        paths=[path for path in self.base.rglob('*') if path.is_file()]
        before={str(path):path.read_bytes() for path in paths}
        plan_before=deepcopy(self.plan);approval_before=deepcopy(self.approval)
        source,receipts=auditor.audit_collector_source(self.plan,self.approval)
        self.assertEqual(source,str(self.recovery))
        self.assertEqual(receipts,[dict(job_id='replacement',manifest=str(self.receipt_path),
            manifest_sha256=auditor.digest(self.receipt),source=str(self.recovery))])
        self.assertEqual(self.plan,plan_before);self.assertEqual(self.approval,approval_before)
        self.assertEqual({str(path):path.read_bytes() for path in paths},before)
        self.assertEqual(auditor.audit_collector_source(self.plan,{'attempts':[{'job_id':'original'}]}),
                         (str(self.parent),[]))

    def test_modified_receipt_requires_approved_binding(self):
        changed=dict(self.receipt,collector_after_sha256='tampered')
        write(self.receipt_path,changed)
        with self.assertRaisesRegex(ValueError,'not bound'):
            auditor.audit_collector_source(self.plan,self.approval)

    def test_rebound_receipt_cannot_change_campaign_shard_budget_or_protocol(self):
        original=deepcopy(self.receipt)
        changes={'campaign_plan_sha256':'other-campaign','shard':1,'added_budget':True,
                 'scientific_protocol_unchanged':False,'parent_source_manifest_sha256':'other-source',
                 'source':str(self.parent)}
        for field,value in changes.items():
            self.receipt=dict(original,**{field:value});self.bind_receipt()
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'not bound'):
                auditor.audit_collector_source(self.plan,self.approval)

    def test_rebound_manifest_cannot_change_actor_or_judge(self):
        original_after=dict(self.after)
        for name in ('openwebrl/generate_browser.py','openwebrl/reward_browser.py'):
            self.after=dict(original_after)
            (self.recovery/name).write_text('# scientific settings changed\n')
            self.after[name]=auditor.sha(self.recovery/name)
            self.bind_source_manifest()
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'outside its declared'):
                auditor.audit_collector_source(self.plan,self.approval)
            (self.recovery/name).write_bytes((self.parent/name).read_bytes())

    def test_undeclared_missing_or_added_source_files_rejected(self):
        original=dict(self.after)
        for changed in ({name:value for name,value in original.items() if name!='openwebrl/reward_browser.py'},
                        dict(original,undeclared_script='not-approved')):
            self.after=changed;self.bind_source_manifest()
            with self.assertRaisesRegex(ValueError,'outside its declared'):
                auditor.audit_collector_source(self.plan,self.approval)

    def test_post_receipt_source_and_preserved_evidence_tampering_rejected(self):
        source_file=self.recovery/'openwebrl/arm_task_pass8.py'
        original=source_file.read_bytes();source_file.write_bytes(original+b'# changed after approval\n')
        with self.assertRaisesRegex(ValueError,'source hash mismatch'):
            auditor.audit_collector_source(self.plan,self.approval)
        source_file.write_bytes(original)
        self.evidence.write_bytes(self.evidence.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'modified preserved evidence'):
            auditor.audit_collector_source(self.plan,self.approval)

    def add_next_verifier(self):
        source=self.base/'source-v3'
        for name in self.after:
            target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((self.recovery/name).read_bytes())
        (source/'openwebrl/arm_task_pass8.py').write_text('# verifier for both native invalid stubs\n')
        (source/'scripts/recover_sft_pass8_extension.py').write_text('# V3 frozen entry\n')
        hashes={name:auditor.sha(source/name) for name in self.after}
        write(source/'reference_manifest.json',{'recipe_files_sha256':hashes})
        receipt=dict(self.receipt,source=str(source),shard=2,
            source_manifest_sha256=auditor.sha(source/'reference_manifest.json'),
            collector_after_sha256=hashes['openwebrl/arm_task_pass8.py'],
            recovery_controller_sha256=hashes['scripts/recover_sft_pass8_extension.py'],
            supersedes_verifier_source=str(self.recovery),
            supersedes_source_manifest_sha256=self.receipt['source_manifest_sha256'],
            supersedes_collector_sha256=self.receipt['collector_after_sha256'])
        path=self.base/'recovery-v3.json';write(path,receipt)
        row=dict(job_id='second-repair',shard=2,source=str(source),
            recovery_manifest=str(path),recovery_manifest_sha256=auditor.digest(receipt))
        self.approval['attempts'].append(row)
        return source,path,receipt,row

    def test_superseding_verifier_preserves_both_recovery_receipts(self):
        source,path,receipt,row=self.add_next_verifier()
        selected,receipts=auditor.audit_collector_source(self.plan,self.approval)
        self.assertEqual(selected,str(source))
        self.assertEqual([r['job_id'] for r in receipts],['replacement','second-repair'])
        self.assertEqual(receipts[0]['manifest_sha256'],auditor.digest(self.receipt))

    def test_missing_or_wrong_supersession_binding_rejected(self):
        source,path,receipt,row=self.add_next_verifier()
        for field in ('supersedes_verifier_source','supersedes_source_manifest_sha256',
                      'supersedes_collector_sha256'):
            for value in (None,'unbound'):
                changed=dict(receipt)
                if value is None:changed.pop(field)
                else:changed[field]=value
                write(path,changed);row['recovery_manifest_sha256']=auditor.digest(changed)
                with self.subTest(field=field,value=value),self.assertRaisesRegex(ValueError,'supersession chain'):
                    auditor.audit_collector_source(self.plan,self.approval)


class FinalAuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.root=self.base/'campaign';self.root.mkdir()
        self.source=self.base/'source';self.source.mkdir()
        self.pointer=self.base/'pointer.json'
        self.ids=[f'webvoyager/{i}' for i in range(2000)]
        self.targets=self.ids[:707]
        self.shards=[self.targets[i::4] for i in range(4)]
        source_file=self.source/'openwebrl/arm_task_pass8.py'
        source_file.parent.mkdir();source_file.write_text('# isolated audit fixture\n')
        write(self.source/'reference_manifest.json',dict(recipe_files_sha256={
            'openwebrl/arm_task_pass8.py':auditor.sha(source_file)}))
        (self.source/'screen-tasks.jsonl').write_text(''.join(json.dumps(dict(metadata=dict(
            task_id=t,start_url='https://example.com/',task='private instruction '+t))) +'\n' for t in self.ids))
        self.native=self.base/'native.jsonl'
        self.native.write_text(''.join(json.dumps(dict(task_id=str(i),benchmark_name='pae-webvoyager' if i<1914 else 'insta-v3',
            task_name='private instruction '+t,website='https://example.com/',difficulty=5+i%4,
            evaluator_reference=[dict(facts=['fact']*(5+i%4))]))+'\n' for i,t in enumerate(self.ids)))
        self.covariate=self.base/'covariate-reference.json'
        write(self.covariate,dict(input_hashes=dict(native_tasks_sha256=auditor.sha(self.native),
            frozen_tasks_sha256=auditor.sha(self.source/'screen-tasks.jsonl'))))
        self.plan=dict(root=str(self.root),source=str(self.source),
            source_manifest_sha256=auditor.sha(self.source/'reference_manifest.json'),
            resources=auditor.RESOURCES,limits=auditor.LIMITS,
            historical_task_order=self.ids,extension_task_order=self.targets,shards=self.shards,
            new_primary_episodes=2121,new_episodes_per_shard=[531,531,531,528],
            checkpoint='/official-sft',old_record_roots=['/immutable-old'],protocol={'fixture':True})
        for name in ('historical_inputs','historical_audit','actor_hash_proof','scope_authorization',
                     'original_plan','original_aggregate'):
            path=self.root/(name+'.json');write(path,{'fixture':name})
            self.plan[name]=str(path);self.plan[name+'_sha256']=auditor.sha(path)
        self.approval=dict(approved=True,plan_sha256=auditor.digest(self.plan),resources=auditor.RESOURCES,
            limits=auditor.LIMITS,attempts=[dict(job_id=str(101+i),shard=i,time_limit_seconds=36000) for i in range(4)])
        write(self.root/'plan.json',self.plan);write(self.root/'approval.json',self.approval)
        write(self.pointer,dict(root=str(self.root),plan_sha256=auditor.digest(self.plan),verified_complete=False,
            shards=[dict(shard=i,root=str(self.root/f'shard-{i}'),job_id=str(101+i)) for i in range(4)]))
        self.configs=[]
        for i in range(4):
            config=dict(output=str(self.root/f'shard-{i}'),shard=i,job_id=str(101+i),task_order=self.shards[i],
                historical_task_order=self.ids,extension_task_order=self.targets,checkpoint=self.plan['checkpoint'],
                old_record_roots=self.plan['old_record_roots'],historical_inputs=self.plan['historical_inputs'],
                historical_inputs_sha256=self.plan['historical_inputs_sha256'])
            write(self.root/f'shard-{i}/workers/{101+i}/collector-config.json',config)
            n=len(self.shards[i])*3
            successes=int(i in (0,2))
            write(self.root/f'shard-{i}/collection-complete.json',dict(complete=True,tasks=n//3,
                primary_attempts=n,expected_attempts=n,valid_attempts=n,invalid_attempts=0,successes=successes))
            self.configs.append(config)
        self.old=[]
        for i,t in enumerate(self.ids):
            for attempt in range(5):
                valid=not(attempt==4 and 682<=i<726)
                self.old.append(dict(task_id=t,attempt=attempt,valid=valid,
                    reward=float(i>=707 and attempt==0) if valid else None))
        self.calls=[]
        self.png='iVBORw0KGgoAAAANSUhEUgAAAAMAAAACCAIAAAASFvFNAAAAEElEQVR4nGP8zwAFTDAGAwATKQED8NgHhAAAAABJRU5ErkJggg=='
        self.fake_torch=ModuleType('torch')
        self.fake_torch.load=lambda path,*args,**kwargs:auditor.read(path)

    def tearDown(self):
        self.tmp.cleanup()

    def runner(self,state='COMPLETED',exit_code='0:0',elapsed=100,missing=(),prior_seconds=None):
        def run(command,**kwargs):
            self.calls.append(command)
            self.assertEqual(command[0],'/usr/bin/sacct')
            ids=command[command.index('-j')+1].split(',')
            return SimpleNamespace(returncode=0,stderr='',stdout=''.join(
                f'{job}|{("FAILED" if job=="100" else state)}|{(prior_seconds if job=="100" else elapsed)}|'
                f'{("1:0" if job=="100" else exit_code)}|cpu=8,mem=120G,gres/gpu=1,gres/gpu:h200=1|\n'
                for job in ids if job not in missing))
        return run

    def build_primary(self):
        for i,config in enumerate(self.configs):
            root=Path(config['output']);calls=0
            for task in config['task_order']:
                for attempt in (5,6,7):
                    identity=collector_helpers.key(task,attempt)
                    artifact=root/'archives'/identity
                    record=dict(task_id=task,attempt=attempt,valid=True,
                        reward=float(task in (self.ids[0],self.ids[690]) and attempt==6),
                        artifact=str(artifact),actor_requests=1,actor_output_tokens=10)
                    record_path=root/'records'/(identity+'.json');write(record_path,record)
                    archive=artifact/(hashlib.sha256(task.encode()).hexdigest()+'.pt')
                    write(archive,dict(turns=[dict(metadata=dict(turn_index=0,full_image_list=[self.png]),
                        multimodal_inputs=dict(images=['data:image/png;base64,'+self.png]))]))
                    write(archive.with_suffix('.json'),{'fixture':True})
                    physical=root/'physical-attempts'/identity
                    write(physical/'dispatch.json',dict(task_id=task,attempt=attempt,output=str(root/'physical-output')))
                    write(physical/'completed.json',{'fixture':True})
                    write(root/'physical-output/screen_records'/(identity+'.json'),record)
                    if record['reward']==1.:
                        calls+=1
                        write(root/'judge-budget'/f'{calls:05d}.json',dict(status='received',charged_usd=.01))
            write(root/'judge-budget/ledger.json',dict(calls=calls,charged_or_reserved_usd=.01*calls))

    def fake_collector(self):
        self.history_loads=0
        self.deep_shards=[]
        def load(config,ids):
            self.history_loads+=1
            self.assertEqual(ids,self.ids)
            return self.old
        module=SimpleNamespace(load_old_records=load,key=collector_helpers.key,
            summarize_joined=collector_helpers.summarize_joined,load_records=collector_helpers.load_records)
        def verify_record(record,task,attempt,*,deep=False):
            self.assertTrue(deep)
            import torch
            torch.load(Path(record['artifact'])/(hashlib.sha256(task.encode()).hexdigest()+'.pt'))
            return record
        module.verify_record=verify_record
        def deep_audit(config,*,require_complete,deep):
            self.assertTrue(require_complete and deep)
            self.deep_shards.append(config['shard'])
            module.load_old_records(config,self.ids)
            records=module.load_records(Path(config['output'])/'records')
            for record in records:
                module.verify_record(record,record['task_id'],record['attempt'],deep=True)
            summary=collector_helpers.summarize(records,config['task_order'],require_complete=True,verify_artifacts=False)
            summary['physical_dispatches_charged']=len(list((Path(config['output'])/'physical-attempts').iterdir()))
            return summary,{}
        module.audit=deep_audit
        return module

    def test_full_global_aggregation_history_verified_once_and_no_completion_mutation(self):
        self.build_primary();module=self.fake_collector()
        pointer_before=self.pointer.read_bytes();approval_before=(self.root/'approval.json').read_bytes()
        with patch.dict(sys.modules,{'torch':self.fake_torch}):
            report=auditor.audit(self.root,self.pointer,run=self.runner(),collector_loader=lambda source:module,
                                 native_tasks=self.native,covariate_reference=self.covariate)
        aggregate=report['aggregate']
        self.assertTrue(report['audit_complete']);self.assertFalse(report['verified_complete'])
        self.assertEqual(aggregate['combined_pass8_rate'],1295/2000)
        self.assertEqual(aggregate['rescued_tasks'],2)
        self.assertEqual(aggregate['rescues_five_valid_failures']['rate'],1/682)
        self.assertEqual(aggregate['rescues_invalid_misses']['rate'],1/25)
        self.assertEqual(aggregate['cost']['physical_dispatches_charged'],2121)
        self.assertEqual(aggregate['cost']['judge_http_calls_charged'],2)
        self.assertAlmostEqual(aggregate['cost']['judge_charged_or_reserved_usd'],.02)
        self.assertEqual(self.history_loads,1);self.assertEqual(self.deep_shards,[0,1,2,3])
        self.assertEqual(self.pointer.read_bytes(),pointer_before)
        self.assertEqual((self.root/'approval.json').read_bytes(),approval_before)
        self.assertEqual(len(report['task_rows']),2000)
        self.assertEqual(len(report['new_artifact_proofs']),2121)
        self.assertEqual(report['valid_new_terminal_images_verified'],2121)
        self.assertEqual(report['new_artifact_proofs'][0]['terminal_images']['images'][0]['width'],3)
        self.assertFalse(aggregate['fresh_pass3_on_all_tasks_measured'])
        self.assertEqual(aggregate['by_source']['pae-webvoyager']['historical_tasks'],1914)
        self.assertEqual(aggregate['by_source']['insta-v3']['historical_tasks'],86)
        self.assertEqual(aggregate['by_invalid_history_stratum']['five_valid_failures']['extension_tasks'],682)
        self.assertEqual(aggregate['by_rubric_fact_count_group']['8+']['historical_tasks'],500)
        saved=auditor.save_private(self.root,report)
        self.assertFalse(saved['verified_complete'])
        for proof in saved['artifacts'].values():
            self.assertEqual(auditor.sha(proof['path']),proof['sha256'])
        self.assertEqual(self.pointer.read_bytes(),pointer_before)
        self.assertEqual({command[0] for command in self.calls},{'/usr/bin/sacct'})

    def test_missing_or_overstated_collection_receipt_stops_before_scheduler(self):
        receipt=self.root/'shard-3/collection-complete.json'
        value=auditor.read(receipt);value['primary_attempts']=527;write(receipt,value)
        with self.assertRaisesRegex(ValueError,'complete collection receipt'):
            auditor.audit(self.root,self.pointer,run=lambda *a,**k:self.fail('No scheduler call'))

    def test_final_current_job_must_succeed_and_all_prior_attempts_are_charged(self):
        jobs=['101','102','103','104']
        for state,exit_code in [('FAILED','1:0'),('RUNNING','0:0'),('COMPLETED','1:0')]:
            with self.subTest(state=state,exit_code=exit_code),self.assertRaises(ValueError):
                auditor.scheduler_audit(self.approval,jobs,run=self.runner(state,exit_code))
        approval=dict(self.approval,attempts=[dict(job_id='100',shard=0,time_limit_seconds=36000),*self.approval['attempts']])
        report=auditor.scheduler_audit(approval,jobs,run=self.runner(prior_seconds=200))
        self.assertEqual(report['shards'][0]['all_attempt_seconds'],300)
        self.assertEqual(report['total_scheduler_seconds'],600)
        with self.assertRaisesRegex(ValueError,'exceeded'):
            auditor.scheduler_audit(approval,jobs,run=self.runner(prior_seconds=36000))
        with self.assertRaisesRegex(ValueError,'Missing scheduler'):
            auditor.scheduler_audit(approval,jobs,run=self.runner(missing=('100',)))

    def test_changed_frozen_source_or_historical_proof_rejected(self):
        auditor.verify_source(self.plan)
        path=self.source/'openwebrl/arm_task_pass8.py';path.write_text('# changed\n')
        with self.assertRaisesRegex(ValueError,'scientific source'):
            auditor.verify_source(self.plan)

    def test_changed_worker_config_or_nonfinal_pointer_rejected(self):
        path=self.root/'shard-0/workers/101/collector-config.json'
        value=auditor.read(path);value['task_order']=value['task_order'][:-1];write(path,value)
        with self.assertRaisesRegex(ValueError,'scientific configuration'):
            auditor.campaign_inputs(self.root,self.pointer)

    def test_unknown_interrupted_physical_work_and_judge_reservations_remain_charged(self):
        root=self.root/'shard-0'
        (root/'physical-attempts/lost-dispatch').mkdir(parents=True)
        write(root/'physical-attempts/interrupted/dispatch.json',dict(task_id='task',attempt=5,output=str(root/'missing')))
        write(root/'physical-attempts/interrupted/interrupted.json',{'error':'killed'})
        physical=auditor.physical_audit(root)
        self.assertEqual(physical['physical_dispatches_charged'],2)
        self.assertEqual(physical['incomplete_physical_attempts'],2)
        self.assertEqual(physical['actor_work_unknown_attempts'],2)
        write(root/'judge-budget/ledger.json',dict(calls=3,charged_or_reserved_usd=.6))
        write(root/'judge-budget/00001.json',dict(status='received',charged_usd=.1))
        write(root/'judge-budget/00002.json',dict(status='reserved',charged_usd=.2))
        ledger=auditor.ledger_audit(root)
        self.assertEqual(ledger['calls_charged'],3)
        self.assertEqual(ledger['reserved_calls'],1)
        self.assertEqual(ledger['unitemized_reserved_calls'],1)
        self.assertAlmostEqual(ledger['unitemized_reserved_usd'],.3)
        write(root/'judge-budget/ledger.json',dict(calls=2,charged_or_reserved_usd=.01))
        with self.assertRaisesRegex(ValueError,'exceed'):
            auditor.ledger_audit(root)

    def test_missing_or_extra_primary_cannot_get_global_final_rate(self):
        new=[dict(task_id=t,attempt=a,valid=True,reward=0.) for t in self.targets for a in (5,6,7)]
        for changed in [new[:-1],new+[dict(task_id=self.ids[707],attempt=5,valid=True,reward=1.)],new+[new[0]]]:
            with self.assertRaises(ValueError):
                collector_helpers.summarize_joined(self.old,changed,self.ids,require_complete=True,
                    extension_ids=self.targets,expected_old=auditor.BASELINE,verify_artifacts=False)

    def test_native_sources_and_rubric_covariates_require_exact_identity_and_definition(self):
        frozen=self.source/'screen-tasks.jsonl'
        rows=[json.loads(line) for line in frozen.read_text().splitlines()]
        native,proof=auditor.native_metadata(rows,self.native,self.covariate,frozen)
        self.assertEqual(native[self.ids[1999]]['benchmark_name'],'insta-v3')
        self.assertIn('rubric facts',proof['definition'])
        data=[json.loads(line) for line in self.native.read_text().splitlines()]
        data[0]['difficulty']=9
        self.native.write_text(''.join(json.dumps(r)+'\n' for r in data))
        with self.assertRaisesRegex(ValueError,'audited inputs'):
            auditor.native_metadata(rows,self.native,self.covariate,frozen)
        reference=auditor.read(self.covariate)
        reference['input_hashes']['native_tasks_sha256']=auditor.sha(self.native)
        write(self.covariate,reference)
        with self.assertRaisesRegex(ValueError,'definition changed'):
            auditor.native_metadata(rows,self.native,self.covariate,frozen)

    def test_terminal_images_require_real_decodable_bytes_and_check_latest_full_image(self):
        payload=dict(turns=[dict(metadata=dict(turn_index=2,full_image_list=['older-uninspected',self.png]),
            multimodal_inputs=dict(images=['data:image/png;base64,'+self.png]))])
        proof=auditor.verify_terminal_images(payload)
        self.assertEqual(proof['unique_images'],1)
        self.assertTrue(proof['full_image_list_last_checked'])
        self.assertEqual(proof['images'][0]['height'],2)
        self.assertEqual(len(proof['images'][0]['origins']),2)
        for image in ['not-base64', 'https://example.com/screenshot.png','aGVsbG8=']:
            bad=dict(turns=[dict(metadata=dict(turn_index=0),multimodal_inputs=dict(images=[image]))])
            with self.subTest(image=image),self.assertRaises((ValueError,OSError)):
                auditor.verify_terminal_images(bad)
        with self.assertRaisesRegex(ValueError,'lacks terminal image'):
            auditor.verify_terminal_images(dict(turns=[dict(metadata=dict(turn_index=0))]))


if __name__=='__main__':unittest.main()
