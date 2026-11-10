"""Execute the preserved native train loop for two CPU-fixture ARM cycles.

GPU actors/browser calls are fixtures. Native loop, normalization, bonus/gates,
durable batch/cursor files and checkpoint inspection are exercised unchanged.
"""
import ast
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.distributed.checkpoint import save as save_dcp
from openwebrl import arm_turn_bonus as bonus
from openwebrl import arm_turn_bonus_cycles as cycles
from test_arm_turn_bonus_pipeline import fixture
sys.path.insert(0,str(Path('scripts').resolve()))
from run_arm_turn_bonus_cycles import plan,SOURCE
from inspect_training_checkpoint import inspect_checkpoint


class NativeCycles(unittest.TestCase):
    def test_two_native_iterations_with_real_files_and_teacher_acknowledgements(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); active=root/'arm-config.json'; events=[]; stop=threading.Event(); errors=[]
            arm=dict(plan('CPU_NATIVE')['arm_config'],run_output=str(root),output=str(root),
                deadline_epoch_seconds=time.time()+14400)
            bonus.write_json(active,arm)
            args=SimpleNamespace(save=str(root/'runtime'),start_rollout_id=0,num_rollout=2,
                save_debug_rollout_data=str(root/'runtime/rollout_recovery/{rollout_id}.pt'),
                offload_train=True,offload_rollout=True,use_critic=False,critic_train_only=False,
                check_weight_update_equal=False,eval_interval=None,skip_eval_before_train=True,
                rollout_global_dataset=True,save_interval=1,global_batch_size=256,ppo_epochs=2,
                judge_api_model='gpt-4.1',judge_prompt_variant='action_history',advantage_estimator='grpo',
                grpo_std_normalization=True)
            teacher={'active':False}; counters={'updates':0,'weights':-1}; checkpoints=[]
            def controller():
                previous=None
                try:
                    while not stop.wait(.005):
                        path=root/'teacher-request.json'
                        if not path.exists(): continue
                        req=json.loads(path.read_text())
                        if req==previous: continue
                        teacher['active']=req['phase']=='collection'
                        events.append(('teacher',req['rollout_id'],req['phase']))
                        bonus.write_json(root/'teacher-ready.json',req); previous=req
                except BaseException as exc: errors.append(exc)
            def save_cursor(rid):
                path=Path(args.save)/'rollout'/f'global_dataset_state_dict_{rid}.pt'
                path.parent.mkdir(parents=True,exist_ok=True)
                torch.save(dict(sample_index=(rid+1)*120),path)
            def generate(rid):
                self.assertTrue(teacher['active']); self.assertEqual(counters['weights'],rid)
                current=cycles.reset_collection(args,rid)
                self.assertEqual(current['records'],[])
                rows,unused,fargs,ns,manager=fixture(root/'fixture'/str(rid),training=True)
                fargs.save=args.save; fargs.save_debug_rollout_data=args.save_debug_rollout_data
                for row in rows:
                    label=row.metadata['arm_turn_bonus']
                    if label: label['policy_id']=current['config']['policy_id']
                ns['_save_debug_rollout_data'](manager,rows,rid,False)
                data=ns['_convert_samples_to_train_data'](manager,rows)
                save_cursor(rid)
                events.append(('collect',rid,current['config']['policy_id']))
                return data
            def train(rid,data):
                self.assertFalse(teacher['active'],'PPO started before teacher shutdown acknowledgement')
                gate=json.loads((root/'iterations'/f'{rid:04d}'/'training_gate.json').read_text())
                self.assertTrue(gate['ready']); self.assertEqual(gate['expected_optimizer_updates'],8)
                counters['updates']+=gate['expected_optimizer_updates']; events.append(('train',rid))
            def save(rid,**kw):
                checkpoint=Path(args.save)/f'iter_{rid:07d}'; checkpoint.mkdir(parents=True)
                save_dcp({'fixture_weight':torch.ones(4)},checkpoint_id=checkpoint,no_dist=True)
                torch.save(dict(iteration=rid,args=SimpleNamespace(global_batch_size=256),
                    opt_param_scheduler=dict(num_steps=counters['updates']*256),
                    optimizer=dict(optimizer=dict(param_groups=[dict(step=counters['updates'])]))),checkpoint/'common.pt')
                (Path(args.save)/'latest_checkpointed_iteration.txt').write_text(str(rid))
                checkpoints.append(inspect_checkpoint(Path(args.save),expected_updates=counters['updates']))
                events.append(('save',rid))
            def sync():
                self.assertFalse(teacher['active']); counters['weights']+=1
                events.append(('sync',counters['weights']))
            remote=lambda f:SimpleNamespace(remote=f)
            actor=SimpleNamespace(update_weights=sync,async_train=train,save_model=save,
                offload=lambda:events.append(('offload_actor',)),finish_tracking=lambda:None)
            manager=SimpleNamespace(generate=remote(generate),save=remote(save_cursor),dispose=remote(lambda:None),
                onload_weights=remote(lambda:None),onload_kv=remote(lambda:None),offload=remote(lambda:None))
            ns=dict(os=os,configure_logger=lambda:None,init_tracking=lambda *a:None,finish_tracking=lambda *a:None,
                create_placement_groups=lambda a:{'rollout':None},create_rollout_manager=lambda *a:(manager,1),
                create_training_models=lambda *a:(actor,None),_build_train_progress_bar=lambda *a:None,
                _log_train_progress=lambda *a:None,_update_train_progress_bar=lambda *a,**kw:None,
                _advance_train_progress_bar=lambda *a:None,_ray_get_with_actor_retry=lambda result,**kw:result,
                evict_file_cache=lambda *a,**kw:0,evict_file_backed_cache=lambda:dict(files=0,bytes=0),
                logger=logging.getLogger(__name__),should_run_periodic_action=lambda rid,interval,*a:bool(interval))
            tree=ast.parse((SOURCE/'train.py').read_text())
            func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
            exec(compile(ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[])),str(SOURCE/'train.py'),'exec'),ns)
            thread=threading.Thread(target=controller,daemon=True)
            try:
                with patch.dict(os.environ,OPENWEBRL_ARM_TURN_BONUS_CONFIG=str(active)),patch.object(bonus,'log_progress'):
                    thread.start(); ns['train'](args)
            finally:
                stop.set(); thread.join(timeout=5); bonus._STATE=None
            self.assertFalse(thread.is_alive()); self.assertEqual(errors,[])
            self.assertEqual([c['completed_optimizer_updates'] for c in checkpoints],[8,16])
            self.assertEqual([e for e in events if e[0]=='sync'],[('sync',0),('sync',1)])
            for rid in range(2):
                self.assertLess(events.index(('teacher',rid,'training')),events.index(('train',rid)))
                self.assertLess(events.index(('train',rid)),events.index(('save',rid)))
                saved=json.loads((root/'iterations'/f'{rid:04d}'/'checkpoint-saved.json').read_text())
                self.assertEqual(saved['rollout_id'],rid)
            first=json.loads((root/'iterations/0000/calibration.json').read_text())
            second=json.loads((root/'iterations/0001/calibration.json').read_text())
            self.assertNotEqual(first['policy_id'],second['policy_id'])
            self.assertEqual(first['admitted'],120); self.assertEqual(second['admitted'],120)


if __name__=='__main__': unittest.main()
