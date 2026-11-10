import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from openwebrl.arm_mig_probe import SerialAuditSelector,validate_devices,checked_records,archived_history_responses,probe_args,recovered_records
from openwebrl.arm_inference import candidate_seed,split_response


class Devices(unittest.TestCase):
    def test_requires_separate_single_slice_devices(self):
        one=dict(count=1,memory_bytes=18*1024**3,uuid='mig-a')
        two=dict(one,uuid='mig-b');validate_devices(one,two)
        for bad in [dict(one),dict(two,count=2),dict(two,memory_bytes=141*1024**3)]:
            with self.assertRaises(ValueError):validate_devices(one,bad)


class Candidates(unittest.IsolatedAsyncioTestCase):
    async def test_archived_history_survives_actual_selector_payload(self):
        history=[dict(thought='Reason',action='<tool_call>{"x":123}</tool_call>'),
                 '<think>Older reason</think>click(2,3)']
        expected=[history[0],split_response(history[1])]
        async def infer(*args,**kwargs):return ('action',[1],[],'stop')
        payloads=[]
        async def select(endpoint,payload,*args):
            payloads.append(payload);return {'raw':'{"selection":1}'}
        with tempfile.TemporaryDirectory() as tmp:
            selector=SerialAuditSelector('selection','unused',tmp)
            with patch('openwebrl.arm_task_success.request_selection_result',select):
                await selector(infer=infer,url='unused',input_text='prompt',sampling_params={},images=[],
                    observation={'screenshot':b'png'},history=archived_history_responses(history),
                    task='task',task_id='x',turn=2,timeout=1)
        self.assertEqual(payloads[0]['history'],expected)
        for invalid in [[{'thought':'text','action':{}}],[None],[{'action':'x'}]]:
            with self.assertRaises(ValueError):archived_history_responses(invalid)

    async def test_serialization_preserves_five_seeds_and_selection_inverse(self):
        running=0;maximum=0;calls=[]
        async def infer(url,text,sampling,images,**kw):
            nonlocal running,maximum
            running+=1;maximum=max(maximum,running);calls.append(sampling['sampling_seed'])
            index=len(calls)-1
            await asyncio.sleep(.001);running-=1
            return (f'response-{index}',[index],[],'stop')
        async def select(*a,**kw):return {'raw':'{"selection":1}'}
        with tempfile.TemporaryDirectory() as tmp:
            selector=SerialAuditSelector('selection','unused',tmp,seed=23)
            with patch('openwebrl.arm_task_success.permutation',return_value=[4,3,2,1,0]),patch('openwebrl.arm_task_success.request_selection_result',select):
                result,meta=await selector(infer=infer,url='unused',input_text='prompt',sampling_params={},
                    images=[],observation={'screenshot':b'png'},history=[],task='task',task_id='x',turn=7,timeout=1)
            self.assertEqual(maximum,1)
            self.assertEqual(calls,[candidate_seed(23,'x',7,i) for i in range(5)])
            self.assertEqual(result[0],'response-4');self.assertEqual(meta['selected_index'],4)
            record=json.loads(next(Path(tmp).glob('*.json')).read_text())
            self.assertEqual(len(record['responses']),5)


class Archives(unittest.TestCase):
    def test_recovery_rejects_changed_record_and_actor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'probe_records').mkdir();art=root/'artifact';art.mkdir()
            archive=art/(hashlib.sha256(b'task').hexdigest()+'.pt')
            archive.write_bytes(b'rollout');archive.with_suffix('.json').write_text('{}')
            row=dict(task_id='task',mode='arm',artifact=str(art),error_type=None,valid=True,
                selector_calls=7,selector_fallback_turns=0,policy_id='sft-mig-architecture-probe',
                judge_id='gpt-4.1/action_history')
            record=root/'probe_records/a.json';record.write_text(json.dumps(row))
            config=dict(actor='actor',tasks=[{'metadata':{'task_id':'task'}}],
                recovery=dict(root=str(root),actor='actor',task_id='task',
                    files_sha256={str(record):hashlib.sha256(record.read_bytes()).hexdigest()}))
            self.assertEqual(recovered_records(config)[('task','arm')],row)
            with self.assertRaises(ValueError):recovered_records(dict(config,actor='different'))
            row['reward']=0.;record.write_text(json.dumps(row))
            with self.assertRaises(ValueError):recovered_records(config)

    def test_native_scalar_reward_survives_rollout_and_verdict_save(self):
        import torch
        from openwebrl.eval_monitor import save_task
        from slime.utils.types import Sample
        args=probe_args({'actor_port':1234,'actor':'unused'})
        with tempfile.TemporaryDirectory() as tmp:
            sample=Sample(index=0,reward=1.,status=Sample.Status.COMPLETED,
                metadata={'turn_index':0,'reward':{'judge':1.,'judge_prompt_variant':'action_history'}})
            save_task(args,Path(tmp),'task',[sample],None)
            archive=next(Path(tmp).glob('*.pt'))
            saved=torch.load(archive,map_location='cpu',weights_only=False)
            verdict=json.loads(archive.with_suffix('.json').read_text())
            self.assertEqual(saved['turns'][0]['reward'],1.)
            self.assertEqual(verdict['metrics']['valid_trajectories'],1)
            self.assertEqual(verdict['metrics']['successes'],1)
            self.assertEqual(verdict['judge_model'],'gpt-4.1')

    def test_requires_rollout_verdict_and_valid_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'probe_records').mkdir();art=root/'artifact';art.mkdir()
            archive=art/(hashlib.sha256(b'task').hexdigest()+'.pt')
            row=dict(task_id='task',mode='arm',artifact=str(art),error_type=None,valid=True,
                selector_calls=1,selector_fallback_turns=0)
            (root/'probe_records/a.json').write_text(json.dumps(row))
            with self.assertRaises(ValueError):checked_records(root,[('task','arm')])
            archive.write_bytes(b'rollout');archive.with_suffix('.json').write_text('{}')
            self.assertEqual(len(checked_records(root,[('task','arm')])),1)
            row['selector_fallback_turns']=1
            (root/'probe_records/a.json').write_text(json.dumps(row))
            with self.assertRaises(ValueError):checked_records(root,[('task','arm')])

if __name__=='__main__':unittest.main()
