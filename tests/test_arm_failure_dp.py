import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from openwebrl.arm_failure_dp import shard_windows
from openwebrl.arm_failure_additive import window_schedule

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


class DPTransport(unittest.TestCase):
    def test_every_auxiliary_row_once_and_equal_collective_counts(self):
        for n in (0,1,3,7,32,65,95):
            windows=window_schedule([{}]*n,max(n,365),[256]*3,256,1/6)
            for dp in (1,2,4):
                shards=[shard_windows(windows,dp,r) for r in range(dp)]
                real=[]
                for w in range(3):
                    self.assertEqual(len({len(s[w]) for s in shards}),1)
                    for shard in shards:
                        for i,scale,pad in shard[w]:
                            if pad:self.assertEqual(scale,0)
                            else:real.append(i)
                self.assertEqual(sorted(real),list(range(n)))

    def test_native_loss_normalization_preserves_global_gradient(self):
        for n in (0,1,7,65):
            windows=window_schedule([{}]*n,365,[256]*3,256,1/6)
            reference=None
            for dp in (1,2,4):
                total=0
                for rank in range(dp):
                    for w,rows in enumerate(shard_windows(windows,dp,rank)):
                        outcome=sum(math.sin(i) for i in range(w*256+rank,(w+1)*256,dp))
                        aux=sum(scale*math.cos(i) for i,scale,pad in rows if not pad)
                        microbatches=256//dp+len(rows)
                        total+=(outcome+aux)*microbatches/256*dp/microbatches/dp
                if reference is None:reference=total
                else:self.assertAlmostEqual(total,reference,places=12)

    def test_actual_frozen_iterator_shards_and_pads(self):
        import prepare_arm_gate_dp as stage
        class Iterator:
            def __init__(self,data,micro_batch_size=1,micro_batch_indices=None):
                self.rollout_data=data;self.micro_batch_indices=micro_batch_indices
        spec=importlib.util.spec_from_file_location('dp_aux_test',stage.SOURCES['B']/'openwebrl/arm_failure_aux.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        data_module=types.ModuleType('slime.backends.megatron_utils.data');data_module.DataIterator=Iterator
        prepared=dict(manifest=dict(expected_windows=1,records=[{'advantage':.4}]*7,
            total_failure_rows=365,coefficient=1/6),data={'tokens':list(range(7))})
        seen=[];counts=[]
        for rank in range(4):
            mpu=types.SimpleNamespace(get_data_parallel_world_size=lambda *a:4,
                get_data_parallel_rank=lambda *a:rank)
            core=types.ModuleType('megatron.core');core.mpu=mpu
            base=Iterator({'tokens':list(range(rank,256,4))})
            with patch.dict(sys.modules,{'slime.backends.megatron_utils.data':data_module,'megatron.core':core}):
                result,c=module.attach_auxiliary(types.SimpleNamespace(global_batch_size=256),[base],[64],prepared)
            mixed=result[0].rollout_data;counts.append(c)
            self.assertEqual(mixed['arm_source'].count('outcome'),64)
            for i,source in enumerate(mixed['arm_source']):
                if source=='arm_failure':seen.append(mixed['tokens'][i])
                elif source=='arm_padding':self.assertEqual(mixed['arm_scale'][i],0)
        self.assertEqual(counts,[[66]]*4);self.assertEqual(sorted(seen),list(range(7)))

    def test_frozen_sources_only_change_auxiliary_transport(self):
        import prepare_arm_gate_dp as stage
        self.assertEqual(set(stage.expected_files('B')),{'openwebrl/arm_failure_aux.py','openwebrl/arm_failure_dp.py'})
        stage.prepare_sources()

    def test_requeue_budget_never_resets_full_time_limit(self):
        from requeue_arm_gate_dp import unused_budget
        self.assertEqual(unused_budget({'TimeLimit':'23:53:00','RunTime':'08:45:17'}),54300)
        self.assertLess(unused_budget({'TimeLimit':'23:59:00','RunTime':'01:10:00'}),23*3600)
        with self.assertRaises(ValueError):unused_budget({'TimeLimit':'23:59:00','RunTime':'23:10:00'})

    def test_padding_loss_has_zero_gradient_even_for_extreme_old_logps(self):
        import torch
        import prepare_arm_gate_dp as stage
        spec=importlib.util.spec_from_file_location('dp_padding_test',stage.SOURCES['B']/'openwebrl/arm_failure_aux.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        logp=torch.tensor([1.,2.],requires_grad=True)
        native=types.ModuleType('slime.backends.megatron_utils.loss')
        native.get_log_probs_and_entropy=lambda *a,**k:(None,{'log_probs':[logp]})
        native.policy_loss_function=lambda *a:None
        batch=dict(arm_source=['arm_padding'],arm_old_log_probs=[torch.tensor([-10000.,-10000.])],
            unconcat_tokens=[],total_lengths=[],response_lengths=[])
        with patch.dict(sys.modules,{'slime.backends.megatron_utils.loss':native}):
            loss,metrics=module.loss(types.SimpleNamespace(),batch,None,None)
        loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(loss.item(),0)
        self.assertTrue(torch.equal(logp.grad,torch.zeros_like(logp)))
        self.assertTrue(all(x.item()==0 for x in metrics.values()))

    def test_context_probe_preserves_response_and_real_rows(self):
        import torch
        from openwebrl.arm_dp_validation import add_stress_padding
        aux=dict(tokens=[torch.tensor([4,5,6,7])],response_lengths=[2],loss_masks=[torch.tensor([1,1])],
            arm_old_log_probs=[torch.tensor([-.2,-.3])],total_lengths=[4],multimodal_train_inputs=[{'image':'same'}])
        mixed={k:list(v) for k,v in aux.items()}
        mixed.update(arm_source=['arm_failure'],arm_scale=[2.],arm_advantage=[.4],effective_global_batch_size=256)
        with patch.dict('os.environ',OPENWEBRL_ARM_TPDP_DIAGNOSTIC='1'):
            result,counts=add_stress_padding(mixed,[1],aux)
        self.assertEqual(counts,[2]);self.assertEqual(len(result['tokens'][1]),32768)
        self.assertTrue(torch.equal(result['tokens'][1][-2:],aux['tokens'][0][-2:]))
        self.assertEqual(result['arm_source'],['arm_failure','arm_padding'])
        self.assertEqual(result['arm_scale'],[2.,0.])
        self.assertEqual(result['multimodal_train_inputs'][1],aux['multimodal_train_inputs'][0])

    def test_requeued_controller_awaits_validation_train_and_evals(self):
        import json,tempfile,os
        import prepare_arm_gate_dp as stage
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            runtime=Path(directory);(runtime/'evaluations/arm-gate-b-to60-318934').mkdir(parents=True)
            args=SimpleNamespace(job_id='318934',variant='B')
            req=dict(stage='released',remaining_seconds=40000,resume_from=str(runtime/'origin'))
            calls=[]
            def worker(command,**kwargs):
                calls.append((command[command.index('--worker')+1],
                    command[command.index('--target')+1] if '--target' in command else None))
            with patch.object(stage,'RUNTIME',runtime),patch.object(stage,'request',return_value=req),\
                    patch.object(stage,'prepare_sources'),patch.object(stage,'allocation'),\
                    patch.dict(os.environ,SLURM_JOB_ID='318934',SLURM_RESTART_COUNT='1'),\
                    patch.object(stage.subprocess,'check_output',return_value='allocation'),\
                    patch.object(stage.subprocess,'run',side_effect=worker),\
                    patch('run_arm_gate_checkpoint_eval.checkpoint_ready',return_value=Path('checkpoint')),\
                    patch('prepare_arm_gate_to60.evaluation_plan',return_value={'output':directory,'source':directory}),\
                    patch('run_stage1_to100.audit_rollouts',return_value={'tasks':300}):
                stage.execute(args)
            self.assertEqual(calls,[('validate',None),('train','40'),('eval','40'),('train','60'),('eval','60')])
            result=json.loads((runtime/'evaluations/arm-gate-b-to60-318934-tp2/status.json').read_text())
            self.assertTrue(result['complete'])


if __name__=='__main__':unittest.main()
