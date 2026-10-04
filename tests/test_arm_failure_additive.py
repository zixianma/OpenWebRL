"""CPU behavioral gates: native admission, additive normalization and packing."""
from copy import deepcopy
import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace,ModuleType
from unittest.mock import patch
import torch
torch.set_num_threads(1)
from test_arm_failure_bonus import failed_group
from test_arm_turn_bonus_pipeline import fixture
from openwebrl import arm_turn_bonus as bonus
from openwebrl import arm_failure_additive as add
from openwebrl import arm_failure_aux as aux

class Additive(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.rows,self.current,self.args,_,_=fixture(self.temp.name,training=True)
        self.current['config'].update(admit_all_failure_groups=True,additive_failure_groups=True,
            failure_group_cap=8,failure_loss_coefficient=1/6,rollout_id=0)
        self.p=patch.object(bonus,'_STATE',self.current);self.p.start();self.addCleanup(self.p.stop)

    def test_failures_never_count_toward_native_quota(self):
        g=failed_group(self.rows)
        result=add.filter_groups(self.args,g)
        self.assertFalse(result.keep);self.assertEqual(result.reason,'zero_std_0.0')
        self.assertEqual(len(self.current['failure_additive_pool']),1)
        mixed=[self.rows[i*10:(i+1)*10] for i in range(5)]
        self.assertTrue(add.filter_groups(self.args,mixed).keep)
        for t in mixed:
            for s in t:s.reward=1.
        self.assertFalse(add.filter_groups(self.args,mixed).keep)

    def test_size_failure_preserves_auxiliary_manifest_and_denominator(self):
        add.filter_groups(self.args,failed_group(self.rows))
        mixed=[]
        for i in range(48):
            sample=deepcopy(self.rows[0]);sample.group_index=i;mixed.extend([sample]*16)
        report={}
        with patch('openwebrl.arm_gate_recovery.auxiliary_payload_limit',return_value=0):
            with self.assertRaisesRegex(ValueError,'Auxiliary payload exceeds'):
                add.write_auxiliary(self.args,mixed,self.current,report)
        root=Path(self.temp.name)
        self.assertTrue((root/'failure_auxiliary.pt').is_file())
        manifest=json.loads((root/'failure_auxiliary.json').read_text())
        self.assertEqual(manifest['total_failure_rows'],50)
        self.assertEqual(manifest['failure_groups'],1)
        self.assertEqual(report['outcome_groups_preserved'],48)

    def test_invalid_and_negative_only(self):
        g=failed_group(self.rows);g[0][-1].remove_sample=True
        add.filter_groups(self.args,g);self.assertFalse(self.current.get('failure_additive_pool'))
        g=failed_group(self.rows)
        for t in g:t[0].metadata['arm_turn_bonus'].update(selected_index=1,unit_bonus=-.2)
        add.filter_groups(self.args,g);self.assertEqual(len(self.current['failure_additive_pool']),1)

    def test_reservoir_order_invariant_bounded(self):
        groups=[]
        for n in range(24):
            g=failed_group(self.rows)
            for t in g:
                for s in t:s.group_index=n;s.metadata['task_id']=f'task-{n}'
            groups.append(g)
        a=deepcopy(self.current);b=deepcopy(self.current)
        for g in groups:add.retain(a,g)
        for g in reversed(groups):add.retain(b,g)
        self.assertEqual(sorted(a['failure_additive_pool']),sorted(b['failure_additive_pool']))
        self.assertEqual(len(a['failure_additive_pool']),8)

    def test_sparse_denominator_and_window_count(self):
        records=[{'advantage':-.1} for _ in range(7)]
        windows=add.window_schedule(records,100,[256]*4,256,1/6)
        self.assertEqual(sorted(i for w in windows for i,_ in w),list(range(7)))
        # Native divides accumulated losses by256; average four optimizer windows.
        per_row=windows[0][0][1]/256/4
        self.assertAlmostEqual(per_row,(1/6)/100)
        self.assertEqual(add.window_schedule([],0,[256]*4,256,1/6),[[],[],[],[]])
        with self.assertRaises(ValueError):add.window_schedule(records,100,[128],256,1/6)
        with self.assertRaises(ValueError):add.window_schedule([{}]*33,100,[256],256,1/6)

    def test_packing_preserves_outcome_order_and_updates(self):
        class Iterator:
            def __init__(self,data,micro_batch_size=None):self.rollout_data=data;self.micro_batch_indices=None
        module=ModuleType('slime.backends.megatron_utils.data');module.DataIterator=Iterator
        base=SimpleNamespace(rollout_data={'tokens':['a','b','c','d'],'effective_global_batch_size':2},micro_batch_indices=[[3],[0],[2],[1]])
        prepared={'manifest':{'records':[{'advantage':-.1},{'advantage':.4}], 'total_failure_rows':10,'coefficient':1/6,'expected_windows':2},'data':{'tokens':['x','y']}}
        with patch.dict(sys.modules,{'slime.backends.megatron_utils.data':module}):
            result,counts=aux.attach_auxiliary(SimpleNamespace(global_batch_size=2),[base],[2,2],prepared)
        data=result[0].rollout_data
        self.assertEqual(counts,[3,3]);self.assertEqual(data['effective_global_batch_size'],2)
        self.assertEqual([t for t,s in zip(data['tokens'],data['arm_source']) if s=='outcome'],['d','a','c','b'])
        self.assertTrue(all(s==1 for s,t in zip(data['arm_scale'],data['arm_source']) if t=='outcome'))
        self.assertEqual(aux.attach_auxiliary(None,[base],[2,2],None),([base],[2,2]))

    def test_actual_auxiliary_loss_gradient_and_clipping(self):
        module=ModuleType('slime.backends.megatron_utils.loss')
        module.get_log_probs_and_entropy=lambda logits,**kw:(None,{'log_probs':[logits]})
        module.policy_loss_function=lambda *a:None
        args=SimpleNamespace(eps_clip=.2,eps_clip_high=.28)
        batch={'arm_source':['arm_failure'],'unconcat_tokens':[],'total_lengths':[], 'response_lengths':[],
               'arm_old_log_probs':[torch.zeros(2)],'arm_advantage':[.4],'arm_scale':[2.]}
        with patch.dict(sys.modules,{'slime.backends.megatron_utils.loss':module}):
            logits=torch.zeros(2,requires_grad=True);loss,_=aux.loss(args,batch,logits,torch.mean);loss.backward()
            self.assertTrue(torch.all(logits.grad<0))
            batch['arm_advantage']=[-.1];logits=torch.zeros(2,requires_grad=True);loss,_=aux.loss(args,batch,logits,torch.mean);loss.backward()
            self.assertTrue(torch.all(logits.grad>0))
            batch['arm_advantage']=[.4];logits=torch.full((2,),1.,requires_grad=True);loss,_=aux.loss(args,batch,logits,torch.mean);loss.backward()
            self.assertTrue(torch.all(logits.grad==0))

    def test_independent_credit_values_survive_auxiliary_manifest(self):
        for rule in ('response_index','action_class'):
            self.current['failure_additive_pool']={}
            self.current['config'].update(candidate_gate='min2',credit_assignment=rule)
            group=failed_group(self.rows)
            for trajectory in group:
                label=trajectory[0].metadata['arm_turn_bonus']
                label.update(candidate_gate='min2',selected_index=1,
                    **bonus.candidate_bonus([0,0,1,2,3],1,rule))
            add.filter_groups(self.args,group)
            mixed=[]
            for i in range(48):
                sample=deepcopy(self.rows[0]);sample.group_index=i;mixed.extend([sample]*16)
            add.write_auxiliary(self.args,mixed,self.current,{})
            manifest=json.loads((Path(self.temp.name)/'failure_auxiliary.json').read_text())
            self.assertEqual(manifest['credit_assignment'],rule)
            self.assertEqual(len(manifest['records']),5)
            for row in manifest['records']:
                self.assertAlmostEqual(row['advantage'],.3 if rule=='action_class' else -.1)
                aux.validate_auxiliary_advantage(row,self.current['config'])
                bad=deepcopy(row);bad['advantage']=.4
                with self.assertRaises(ValueError):aux.validate_auxiliary_advantage(bad,self.current['config'])
                bad=deepcopy(row);bad.pop('arm_turn_bonus')
                with self.assertRaises(ValueError):aux.validate_auxiliary_advantage(bad,self.current['config'])

    def test_durable_manifest_keeps_zero_rows_in_denominator(self):
        g=failed_group(self.rows)
        for trajectory in g:
            trajectory[0].multimodal_train_inputs={'pixel_values':torch.arange(8).reshape(2,4)}
        add.filter_groups(self.args,g)
        mixed=[]
        for i in range(48):
            s=deepcopy(self.rows[0]);s.group_index=i;mixed.extend([s]*16)
        report={};add.write_auxiliary(self.args,mixed,self.current,report)
        m=json.loads((Path(self.temp.name)/'failure_auxiliary.json').read_text())
        self.assertEqual(m['total_failure_rows'],50);self.assertEqual(len(m['records']),5)
        self.assertAlmostEqual(m['coefficient'],1/48)
        self.assertEqual(m['expected_windows'],3)
        saved=torch.load(Path(self.temp.name)/'failure_auxiliary.pt',weights_only=True)
        self.assertEqual(len(saved),5);self.assertNotIn('terminal_reward',saved[0])
        self.assertTrue(torch.equal(saved[0]['multimodal_train_inputs']['pixel_values'],torch.arange(8).reshape(2,4)))

if __name__=='__main__':unittest.main()
