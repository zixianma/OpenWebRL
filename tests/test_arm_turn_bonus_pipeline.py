"""CPU integration checks using native normalization, transport, PPO and failures.

No model loading, browser sessions, paid API calls or GPU allocation.
"""
import ast
from collections import defaultdict
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import random
import shlex
import sys
import tempfile
import time
from types import MethodType, SimpleNamespace
import unittest
import io
from unittest.mock import patch

import torch
from slime.utils.types import Sample
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from openwebrl import arm_turn_bonus as bonus
from openwebrl import arm_turn_bonus_runtime as runtime
import test_arm_turn_bonus as unit_tests

SOURCE = unit_tests.PreparedSource.source

sys.path.insert(0,str(Path('scripts').resolve()))
import run_arm_turn_bonus_calibration as launcher


def native(path,names,namespace,source=None):
    source=source or SOURCE
    tree=ast.parse((source/path).read_text())
    definitions=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in definitions}==set(names)
    for n in definitions:
        n.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*definitions],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(source/path),'exec'),namespace)


def group_by(rows,key):
    groups=defaultdict(list)
    for row in rows: groups[key(row)].append(row)
    return groups


def fixture(folder,training=False,beta=.5):
    rows=[]
    for group in range(24):
        for trajectory in range(5):
            for turn in range(10):
                parent=group*5+trajectory
                label={} if turn else dict(eligible=True,policy_id='p',executed_index=0,
                    selected_index=0 if trajectory==group%5 else 1,
                    unit_bonus=.8 if trajectory==group%5 else -.2,reason='admitted')
                rows.append(Sample(index=(parent<<16)|turn,group_index=group,tokens=[1,2,3,4],
                    response='reason action',response_length=2,loss_mask=[1,1],rollout_log_probs=[-.3,-.3],
                    status=Sample.Status.COMPLETED,reward=float(trajectory<2),
                    metadata=dict(task_id=f'task-{group}',trajectory_id=parent,turn_index=turn,
                        arm_turn_bonus=label,num_turns_in_trajectory=10)))
    config=dict(output=str(folder),policy_id='p',checkpoint='frozen',shadow_only=not training,
        train_after_calibration=training,beta=beta,scored_fraction=.2,k=5,seed=42,
        label_timeout_seconds=120,max_pending_per_trajectory=2,deadline_epoch_seconds=time.time()+10000)
    state=dict(config=config,records=[],started=time.monotonic(),excluded_ids=set(),excluded_intents=set())
    args=SimpleNamespace(advantage_estimator='grpo',rewards_normalization=True,grpo_std_normalization=True,
        reward_key=None,global_batch_size=256,ppo_epochs=2,rollout_seed=42,balance_data=False,
        normalize_advantages=False,use_opd=False,use_rollout_logprobs=True,kl_coef=0.,
        save=str(Path(folder)/'runtime'),save_debug_rollout_data=str(Path(folder)/'runtime/rollout_recovery/{rollout_id}.pt'))
    bonus.write_json(Path(folder)/'arm-config.json',config)
    ns=dict(torch=torch,os=os,Sample=Sample,Path=Path,random=random,group_by=group_by,
        logger=logging.getLogger(__name__),logging_utils=SimpleNamespace(append_progress_log=lambda *a:None),
        Box=lambda x:x,ray=SimpleNamespace(put=lambda x:x))
    native('slime/ray/rollout.py',['_post_process_rewards','_convert_samples_to_train_data',
        '_split_train_data_by_dp','_save_debug_rollout_data'],ns)
    manager=SimpleNamespace(args=args,custom_reward_post_process_func=None,custom_convert_samples_to_train_data_func=None)
    manager._post_process_rewards=MethodType(ns['_post_process_rewards'],manager)
    return rows,state,args,ns,manager


class Pipeline(unittest.TestCase):
    def test_launch_profiles_resolve_native_topology_and_browser_limits(self):
        for gpus in (2,4):
            with self.subTest(gpus=gpus):
                p=launcher.plan('PROFILE_CHECK',phase='calibrate-train',gpus=gpus)
                # Exercise the preserved shell launcher without starting Python,
                # GPUs, browsers or a judge. Inspect the actual native arguments.
                env=dict(os.environ,**p['environment'],DRY_RUN='1')
                command=shlex.split(launcher.subprocess.check_output(p['command'],env=env,text=True))
                self.assertEqual(command[command.index('--tensor-model-parallel-size')+1],str(gpus))
                self.assertEqual(command[command.index('--actor-num-gpus-per-node')+1],str(gpus))
                self.assertEqual(command[command.index('--custom-config-path')+1],p['environment']['BROWSER_TRAIN_CONFIG'])
                native_config=SimpleNamespace(**p['browser_config'])
                ns=dict(os=os)
                native('openwebrl/generate_browser.py',['_get_browser_rollout_concurrency'],ns)
                self.assertEqual(ns['_get_browser_rollout_concurrency'](native_config),gpus*8)
                self.assertEqual(int(env['BROWSER_CONCURRENCY']),gpus*8)
                # Native launcher ties the pool environment to this variable.
                self.assertIn('SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES="${BROWSER_CONCURRENCY:-4}"',
                    (SOURCE/'scripts/run_h200_browser.sh').read_text())
                self.assertEqual(p['requested_resources']['cpus'],gpus*8)
                self.assertEqual(p['requested_resources']['gpu_hours'],gpus*2)
                self.assertEqual(p['environment']['OPENWEBRL_STOP_AFTER_SAVED_ROLLOUT'],'70')
                self.assertEqual(p['arm_config']['beta'],.5)
                self.assertEqual(p['arm_config']['scored_fraction'],.2)
                original=__import__('yaml').safe_load((SOURCE/'openwebrl/browser_training_config.yaml').read_text())
                original['browser_rollout_concurrency']=gpus*8
                self.assertEqual(p['browser_config'],original)

    def test_replay_admission_requires_complete_untrained_matching_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            rows,state,args,ns,manager=fixture(folder)
            policy='qcq7i4ug:after70:arm-shadow-v1'
            state['config'].update(policy_id=policy,checkpoint=str(launcher.CHECKPOINT),selector_checkpoint=str(launcher.SELECTOR))
            for row in rows:
                if row.metadata['arm_turn_bonus']:
                    row.metadata['arm_turn_bonus']['policy_id']=policy
            config_path=Path(folder)/'arm-config.json'
            bonus.write_json(config_path,state['config'])
            with patch.dict(os.environ,OPENWEBRL_ARM_TURN_BONUS_CONFIG=str(config_path)),patch.object(bonus,'state',return_value=state),patch.object(bonus,'log_progress'):
                ns['_save_debug_rollout_data'](manager,rows,70,False)
                ns['_convert_samples_to_train_data'](manager,rows)
                cursor=Path(args.save)/'rollout/global_dataset_state_dict_70.pt'
                cursor.parent.mkdir(parents=True)
                torch.save({'sample_index':600},cursor)
                self.assertTrue(runtime.before_optimizer(args,70))
            status_path=Path(folder)/'status.json'
            good=dict(collection_complete=True,no_optimizer_updates=True,exit_code=0)
            bonus.write_json(status_path,good)
            receipt=launcher.validate_replay(folder)
            self.assertEqual(receipt['cursor_sha256'],runtime.file_sha256(cursor))
            for change in ({'failed':True},{'exit_code':1},{'no_optimizer_updates':False},{'collection_complete':False}):
                bonus.write_json(status_path,dict(good,**change))
                with self.assertRaises(ValueError): launcher.validate_replay(folder)
            bonus.write_json(status_path,good)
            state['config']['selector_checkpoint']='/wrong/selector'
            bonus.write_json(config_path,state['config'])
            with self.assertRaisesRegex(ValueError,'teacher'):
                launcher.validate_replay(folder)

    def test_selector_preflight_rejects_invalid_decoder_and_wrong_model(self):
        for mode in ('ok','bad_label','wrong_model'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                health=dict(mode='selection',model=str(launcher.SELECTOR) if mode!='wrong_model' else '/wrong/model')
                responses=[health,dict(raw='garbage' if mode=='bad_label' else '{"selection": 3}')]
                def request(*args,**kw):
                    return io.BytesIO(json.dumps(responses.pop(0)).encode())
                with patch.object(launcher.urllib.request,'urlopen',side_effect=request):
                    if mode=='ok':
                        launcher.selector_preflight('http://127.0.0.1:1',folder)
                        self.assertTrue(json.loads((Path(folder)/'selector_preflight.json').read_text())['passed'])
                    else:
                        with self.assertRaises(ValueError): launcher.selector_preflight('http://127.0.0.1:1',folder)
                        self.assertFalse((Path(folder)/'selector_preflight.json').exists())

    def test_controller_records_failure_and_awaits_all_owned_children(self):
        for gpus,mode in ((g,m) for g in (2,4) for m in ('selector_failure','worker_failure','success')):
            with self.subTest(mode=mode,gpus=gpus),tempfile.TemporaryDirectory() as folder:
                p=launcher.plan('1234',gpus=gpus)
                output=Path(folder)/'run'
                p['output']=str(output)
                p['arm_config']['output']=str(output)
                children=[]
                class Process:
                    def __init__(self,is_worker):
                        self.pid=1000+len(children)
                        self.returncode=(1 if mode=='worker_failure' else 0) if is_worker else None
                        self.waited=False
                    def poll(self): return self.returncode
                    def wait(self,timeout=None):
                        self.waited=True
                        return self.returncode
                def popen(*a,**kw):
                    worker=bool(children)
                    expected_devices=','.join(str(i) for i in range(gpus)) if worker else str(gpus-1)
                    self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],expected_devices)
                    proc=Process(worker)
                    children.append(proc)
                    if worker:
                        (output/'collection.log').write_text('browser worker stopped\n')
                        if mode=='success': bonus.write_json(output/'collection_complete.json',{})
                    return proc
                def kill(pid,sig):
                    next(c for c in children if c.pid==pid).returncode=-int(sig)
                def restore(*a,**kw):
                    bonus.write_json(output/'checkpoint_restore_evidence.json',dict(restore_line=
                        f'successfully loaded checkpoint from {output}/checkpoint-view [ t 1/2, p 1/1 ] at iteration 69'))
                original_read=Path.read_text
                def read(path,*a,**kw):
                    return '/slurm/job_1234/step_0' if str(path)=='/proc/self/cgroup' else original_read(path,*a,**kw)
                class Health(io.BytesIO): status=200
                with patch.dict(os.environ,SLURM_JOB_ID='1234',SLURM_STEP_ID='0',CUDA_VISIBLE_DEVICES=','.join(str(i) for i in range(gpus)),WANDB_API_KEY='test',OPENAI_API_KEY='test'), \
                     patch.object(Path,'read_text',read), \
                     patch.object(launcher,'allocation',return_value=dict(cpus=gpus*8,allocated_memory_gib=480,maximum_seconds=7000)) as allocated, \
                     patch.object(launcher.subprocess,'check_output',return_value='1234.batch\n1234.0\n'), \
                     patch.object(launcher.subprocess,'run'),patch.object(launcher.subprocess,'Popen',side_effect=popen), \
                     patch.object(launcher.os,'killpg',side_effect=kill),patch.object(launcher,'record_restore_evidence',side_effect=restore), \
                     patch.object(launcher.urllib.request,'urlopen',return_value=Health(b'{}')), \
                     patch.object(launcher,'selector_preflight',side_effect=RuntimeError('decoder failed') if mode=='selector_failure' else None), \
                     patch('dotenv.dotenv_values',return_value={}):
                    if mode=='success': launcher.execute(p)
                    else:
                        with self.assertRaises(RuntimeError): launcher.execute(p)
                status=json.loads((output/'status.json').read_text())
                self.assertEqual(allocated.call_args.kwargs['requested_gpus'],gpus)
                self.assertEqual(json.loads((output/'browser-training-config.json').read_text())['browser_rollout_concurrency'],gpus*8)
                self.assertTrue(all(c.waited for c in children))
                self.assertTrue(all(c.poll() is not None for c in children))
                self.assertEqual(len(children),1 if mode=='selector_failure' else 2)
                if mode=='success': self.assertTrue(status['collection_complete'])
                else: self.assertTrue(status['failed'])

    def test_native_batch_to_two_epoch_ppo_preserves_turn_identity(self):
        for beta in (0.,.5):
            with self.subTest(beta=beta),tempfile.TemporaryDirectory() as folder:
                rows,state,args,ns,manager=fixture(folder,training=True,beta=beta)
                before=deepcopy([s.to_dict() for s in rows])
                with patch.dict(os.environ,OPENWEBRL_ARM_TURN_BONUS_CONFIG=str(Path(folder)/'arm-config.json')),patch.object(bonus,'state',return_value=state),patch.object(bonus,'log_progress'):
                    ns['_save_debug_rollout_data'](manager,rows,70,False)
                    data=ns['_convert_samples_to_train_data'](manager,rows)
                    cursor=Path(args.save)/'rollout/global_dataset_state_dict_70.pt'
                    cursor.parent.mkdir(parents=True)
                    torch.save({'sample_group_index':120,'sample_index':600},cursor)
                    self.assertFalse(runtime.before_optimizer(args,70))
                self.assertEqual([s.to_dict() for s in rows],before)
                report=json.loads((Path(folder)/'calibration.json').read_text())
                self.assertTrue(report['decision']['passed'])
                self.assertEqual(report['admitted'],120)
                self.assertEqual(report['admitted_distinct_tasks'],24)
                self.assertAlmostEqual(report['variants']['0.5']['bonus_to_outcome_rms'],.0707107,places=5)
                self.assertGreater(report['outcome_breakdown']['failure']['admitted'],0)
                shards=ns['_split_train_data_by_dp'](manager,data,1)
                base_by_id={s.index:((.6 if s.reward==1 else -.4)/(torch.tensor([1.,1.,0.,0.,0.]).std().item()+1e-6)) for s in rows}
                unit_by_id={s.index:bonus.unit_bonus(s,'p') for s in rows}
                old_by_id={s.index:s.rollout_log_probs for s in rows}
                ns.update(mpu=SimpleNamespace(get_data_parallel_group=lambda **kw:None,is_pipeline_last_stage=lambda:True),
                    dist=SimpleNamespace(all_gather_object=lambda out,value,**kw:out.__setitem__(0,value)))
                native('slime/backends/megatron_utils/actor.py',['_build_local_rollout_data_for_epoch_from_local_shard'],ns)
                native('slime/backends/megatron_utils/loss.py',['compute_advantages_and_returns'],ns)
                native('slime/utils/ppo_utils.py',['get_grpo_returns','compute_policy_loss'],ns)
                ids_by_epoch=[]
                for epoch in (0,1):
                    shard=ns['_build_local_rollout_data_for_epoch_from_local_shard'](args,shards[0],70,epoch,0,1)
                    ids=shard['sample_indices']
                    ids_by_epoch.append(ids)
                    self.assertEqual(len(ids),1024)
                    self.assertEqual(shard['effective_global_batch_size'],256)
                    for i,idx in enumerate(ids):
                        self.assertAlmostEqual(shard['rewards'][i],base_by_id[idx]+beta*unit_by_id[idx],places=5)
                        self.assertEqual(shard['rollout_log_probs'][i],old_by_id[idx])
                    shard['rollout_log_probs']=[torch.tensor(x) for x in shard['rollout_log_probs']]
                    ns['compute_advantages_and_returns'](args,shard)
                    advantages=torch.cat(shard['advantages'])
                    expected=torch.tensor([base_by_id[idx]+beta*unit_by_id[idx] for idx in ids]).repeat_interleave(2)
                    torch.testing.assert_close(advantages,expected)
                    log_ratio=torch.zeros_like(advantages,requires_grad=True)
                    losses,_=ns['compute_policy_loss'](-log_ratio,advantages,.2,.28)
                    losses.mean().backward()
                    torch.testing.assert_close(log_ratio.grad,-expected/expected.numel())
                self.assertNotEqual(ids_by_epoch[0],ids_by_epoch[1])

    def test_training_gate_stops_for_low_coverage_or_insufficient_time(self):
        for mode in ('low_labels','no_time','shadow'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                rows,state,args,ns,manager=fixture(folder,training=mode!='shadow')
                if mode=='low_labels':
                    for s in rows: s.metadata['arm_turn_bonus']={}
                if mode=='no_time':
                    state['config']['deadline_epoch_seconds']=time.time()+10
                    bonus.write_json(Path(folder)/'arm-config.json',state['config'])
                with patch.dict(os.environ,OPENWEBRL_ARM_TURN_BONUS_CONFIG=str(Path(folder)/'arm-config.json')),patch.object(bonus,'state',return_value=state),patch.object(bonus,'log_progress'):
                    ns['_save_debug_rollout_data'](manager,rows,70,False)
                    ns['_convert_samples_to_train_data'](manager,rows)
                    cursor=Path(args.save)/'rollout/global_dataset_state_dict_70.pt'
                    cursor.parent.mkdir(parents=True)
                    torch.save({},cursor)
                    self.assertTrue(runtime.before_optimizer(args,70))
                self.assertTrue((Path(folder)/'collection_complete.json').exists())

    def test_corrupt_archive_cannot_pass_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.pt'
            path.write_bytes(b'interrupted torch save')
            with self.assertRaises(Exception): runtime.verify_torch_archive(path)

    def test_group_journal_preserves_outcomes_and_filter_membership(self):
        from slime.utils.rollout_transport import file_back_completed_group
        with tempfile.TemporaryDirectory() as folder:
            rows,state,args,ns,manager=fixture(folder)
            group=[rows[i*10:(i+1)*10] for i in range(5)]
            group[4][-1].reward=None
            pixels=torch.arange(12,dtype=torch.bfloat16).reshape(3,4)
            group[0][0].multimodal_train_inputs={'pixel_values':pixels.clone()}
            result=DynamicFilterOutput(keep=True,reason=None,samples=group[:4])
            with patch.dict(os.environ,OPENWEBRL_MULTIMODAL_STORAGE_DIR=str(Path(folder)/'images')),patch.object(bonus,'state',return_value=state):
                self.assertEqual(file_back_completed_group(group),1)
                runtime.record_completed_group(args,70,group,result,0)
            path=Path(folder)/'groups/70/0.pt'
            saved=torch.load(path,weights_only=False)
            self.assertEqual(len(saved['samples']),50)
            self.assertEqual(len(saved['accepted_sample_indices']),40)
            self.assertIsNone(saved['samples'][-1]['reward'])
            torch.testing.assert_close(saved['samples'][0]['multimodal_train_inputs']['pixel_values'],pixels)
            self.assertTrue(saved['accepted'])
            with patch.object(bonus,'state',return_value=state),self.assertRaisesRegex(ValueError,'overwrite'):
                runtime.record_completed_group(args,70,group,result,0)

    def test_replay_restores_exact_native_cursor_without_advancing_again(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'source/rollout/global_dataset_state_dict_70.pt'
            path.parent.mkdir(parents=True)
            expected=dict(sample_offset=139,epoch_id=3,sample_group_index=9571,sample_index=47855,metadata={'x':1})
            torch.save(expected,path)
            config=dict(replay_origin=dict(dataset_cursor=str(path),cursor_sha256=runtime.file_sha256(path)))
            config_path=Path(folder)/'config.json'
            bonus.write_json(config_path,config)
            args=SimpleNamespace(rollout_global_dataset=True,load='original',rollout_shuffle=False)
            source=SimpleNamespace(args=args,metadata={},_use_adaptive_query_sampling=lambda:False)
            ns=dict(os=os,torch=torch,logger=logging.getLogger(__name__))
            native('slime/rollout/data_source.py',['load'],ns)
            source.load=MethodType(ns['load'],source)
            with patch.dict(os.environ,OPENWEBRL_ARM_TURN_BONUS_CONFIG=str(config_path),OPENWEBRL_ARM_REPLAY_CURSOR=str(path)):
                runtime.load_replay_cursor(source,70)
            self.assertIs(source.args,args)
            self.assertEqual(args.load,'original')
            for k,v in expected.items(): self.assertEqual(getattr(source,k),v)

    def test_invalid_configs_and_critical_health_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            _,state,_,_,_=fixture(folder,training=True)
            runtime.validate_config(state['config'])
            for change in ({'beta':float('nan')},{'scored_fraction':.1},{'train_after_calibration':False},{'max_pending_per_trajectory':0}):
                with self.subTest(change=change),self.assertRaises(ValueError):
                    runtime.validate_config(dict(state['config'],**change))
        self.assertIsNotNone(launcher.health_issue({},"train/grad_norm': nan"))
        self.assertIsNotNone(launcher.health_issue({},'grad_norm=inf'))
        self.assertIsNotNone(launcher.health_issue({'arm_collection/recorded_turns':100},''))
        self.assertIsNone(launcher.health_issue({},'grad_norm=1.2'))
        self.assertIsNotNone(launcher.health_issue({},"ModuleNotFoundError: No module named 'client'"))
        self.assertIsNotNone(launcher.health_issue({'arm_collection/native_failure_sentinels':100},''))
        self.assertIsNone(launcher.health_issue({'arm_collection/native_failure_sentinels':3},''))

    def test_journal_hook_is_the_only_native_collector_change(self):
        source=SOURCE
        parent=Path(json.loads((source/'reference_manifest.json').read_text())['arm_turn_bonus_calibration']['parent_source'])
        class RemoveJournal(ast.NodeTransformer):
            def visit_If(self,node):
                if 'OPENWEBRL_ARM_TURN_BONUS_CONFIG' in ast.unparse(node.test): return None
                return self.generic_visit(node)
        modified=RemoveJournal().visit(ast.parse((source/'slime/rollout/sglang_rollout.py').read_text()))
        self.assertEqual(ast.dump(modified),ast.dump(ast.parse((parent/'slime/rollout/sglang_rollout.py').read_text())))


if __name__=='__main__': unittest.main()
