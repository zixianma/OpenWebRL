import asyncio
import json
from pathlib import Path
import tempfile
import sys
import hashlib
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch

from openwebrl.arm_task_success import summarize,AuditSelector,collect
from openwebrl.arm_terminal_budget import CappedJudge


def row(task,mode,valid=True,reward=0.):
    return dict(task_id=task,mode=mode,valid=valid,reward=reward,policy_id='p',
        judge_id='gpt-4.1/action_history',actor_output_tokens=10,actor_requests=1,
        selector_calls=int(mode=='arm'),selector_fallback_turns=0,elapsed_seconds=1.,artifact='/saved')


class Outcomes(unittest.TestCase):
    def test_invalids_stay_in_primary_denominator(self):
        rows=[row('a','arm',valid=False),row('b','arm',reward=1.)]
        x=summarize(rows,['a','b'],'p')
        self.assertEqual(x['by_mode']['arm']['overall'],.5)
        self.assertEqual(x['by_mode']['arm']['valid_only'],1.)
        self.assertNotIn('paired',x)
        self.assertEqual(set(x['by_mode']),{'arm'})

    def test_partial_duplicate_or_actor_only_panels_are_rejected(self):
        with self.assertRaises(ValueError):summarize([row('a','actor0')],['a'],'p')
        with self.assertRaises(ValueError):summarize([row('a','arm'),row('a','arm')],['a'],'p')
        with self.assertRaises(ValueError):summarize([row('a','arm')],['a','b'],'p')


class Selector(unittest.IsolatedAsyncioTestCase):
    async def test_permutation_is_inverted_before_execution(self):
        calls=[]
        async def infer(*args,**kwargs):
            i=len(calls);calls.append(args);return (f'response-{i}',[i],[], 'stop')
        async def select(endpoint,payload,*args):return {'raw':'{"selection":2}'}
        with tempfile.TemporaryDirectory() as directory:
            selector=AuditSelector('selection','http://unused',directory,seed=42)
            with patch('openwebrl.arm_task_success.permutation',return_value=[4,3,2,1,0]),patch('openwebrl.arm_task_success.request_selection_result',select):
                output,metadata=await selector(infer=infer,url='unused',input_text='prompt',
                    sampling_params={},images=[],observation={'screenshot':b'png'},
                    history=[],task='task',task_id='a',turn=0,timeout=1)
            self.assertEqual(output[0],'response-3');self.assertEqual(metadata['selected_index'],3)
            saved=json.loads(next(Path(directory).glob('*.json')).read_text())
            self.assertEqual(len(saved['candidates']),5)
            self.assertEqual(saved['selected_index'],3)


class Budget(unittest.IsolatedAsyncioTestCase):
    async def test_native_prompt_preserved_and_usage_charged(self):
        seen=[]
        async def create(**kwargs):
            seen.append(kwargs)
            return SimpleNamespace(usage=SimpleNamespace(model_dump=lambda:{'prompt_tokens':100,'completion_tokens':10}))
        async def close():pass
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),close=close)
        with tempfile.TemporaryDirectory() as directory:
            judge=CappedJudge(directory,client=client)
            messages=[{'role':'user','content':'native prompt'}]
            await judge.chat.completions.create(model='gpt-4.1',messages=messages,seed=42)
            self.assertEqual(seen[0]['messages'],messages)
            self.assertEqual(seen[0]['seed'],42)
            self.assertFalse(seen[0]['store'])
            self.assertEqual(seen[0]['max_completion_tokens'],1024)
            ledger=json.loads((Path(directory)/'ledger.json').read_text())
            self.assertAlmostEqual(ledger['charged_or_reserved_usd'],.00028)
            self.assertEqual(ledger['calls'],1)

    async def test_cap_blocks_before_network_and_survives_restart(self):
        calls=[]
        async def create(**kwargs):calls.append(kwargs);raise RuntimeError('must not call')
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with tempfile.TemporaryDirectory() as directory:
            judge=CappedJudge(directory,client=client);judge.reserve(29.999)
            second=CappedJudge(directory,client=client)
            with self.assertRaisesRegex(ValueError,'budget cap'):
                await second.create(model='gpt-4.1',messages=[{'role':'user','content':'task'}])
            self.assertFalse(calls)
            self.assertTrue((Path(directory)/'halt.json').exists())


class Collection(unittest.IsolatedAsyncioTestCase):
    async def test_smoke_is_separate_and_all_primary_arm_trajectories_are_preserved(self):
        from slime.utils.eval_config import EvalDatasetConfig
        fake_native=ModuleType('slime.rollout.sglang_rollout')
        fake_reward=ModuleType('openwebrl.reward_browser');fake_reward._get_openai_client=lambda:None
        fake_output=ModuleType('slime.rollout.base_types');fake_output.RolloutFnEvalOutput=lambda **kwargs:kwargs
        async def close():pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tasks=root/'tasks.jsonl';ids=[str(i) for i in range(100)]
            tasks.write_text(''.join(json.dumps({'prompt':'task','metadata':{'task_id':i}})+'\n' for i in ids))
            calls=[]
            async def evaluate(args,rollout_id,dataset):
                phase=dataset.metadata_overrides['success_audit_phase'];calls.append(phase)
                for item in [json.loads(x) for x in Path(dataset.path).read_text().splitlines()]:
                    task=item['metadata']['task_id']
                    for mode in ['arm']:
                        artifact=root/f'{phase}-{task}-{mode}';artifact.mkdir()
                        archive=artifact/(hashlib.sha256(task.encode()).hexdigest()+'.pt')
                        archive.write_bytes(b'saved');archive.with_suffix('.json').write_text('{}')
                        r=row(task,mode);r.update(artifact=str(artifact),error_type=None)
                        folder=root/f'{phase}_records';folder.mkdir(exist_ok=True)
                        (folder/f'{task}-{mode}.json').write_text(json.dumps(r))
                return {}
            fake_native.eval_rollout_single_dataset=evaluate
            config=dict(output=str(root),tasks=str(tasks),task_order=ids,policy_id='p',budget=str(root/'budget'))
            args=SimpleNamespace(judge_api_model='gpt-4.1',judge_prompt_variant='action_history',
                eval_datasets=[EvalDatasetConfig(name='audit',path=str(tasks),n_samples_per_eval_prompt=1)])
            with patch.dict(sys.modules,{'slime.rollout.sglang_rollout':fake_native,
                    'openwebrl.reward_browser':fake_reward,'slime.rollout.base_types':fake_output}), \
                    patch('openwebrl.arm_terminal_budget.CappedJudge',return_value=SimpleNamespace(close=close)):
                result=await collect(args,0,config)
            self.assertEqual(calls,['smoke','compare'])
            receipt=json.loads((root/'collection-complete.json').read_text())
            self.assertEqual(receipt['trajectories'],100);self.assertEqual(receipt['smoke_trajectories'],3)
            self.assertEqual(result['metrics']['task_success/arm/tasks'],100)
            self.assertNotIn('task_success/actor0/tasks',result['metrics'])


if __name__=='__main__':unittest.main()
