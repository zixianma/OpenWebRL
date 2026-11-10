import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from openwebrl import arm_quality_audit as audit
import label_arm_quality as teacher


class TeacherContract(unittest.TestCase):
    def test_labels_invert_order_and_allow_no_acceptable_action(self):
        labels=dict(assessments=[dict(index=i,label='unacceptable',reason='evidence') for i in range(1,6)],best_indices=[])
        order=[4,2,0,1,3]
        normalized=teacher.unpermute_label(labels,order)
        self.assertEqual(normalized['best_indices'],[])
        labels['assessments'][0]['label']='acceptable';labels['best_indices']=[1]
        normalized=teacher.unpermute_label(labels,order)
        self.assertEqual(normalized['best_indices'],[4])
        self.assertEqual(normalized['assessments'][4]['label'],'acceptable')
        labels['assessments'][-1]['index']=1
        with self.assertRaises(ValueError):teacher.unpermute_label(labels,order)

    def test_call_panel_is_exact_and_stable(self):
        plan={'stages':[{'actor':x} for x in ['sft','baseline20','baseline90']]}
        states=[{'id':str(i)} for i in range(64)]
        items=teacher.work_items(plan,states)
        self.assertEqual(len(items),231);self.assertEqual(len(set(items)),231)
        self.assertEqual(sum(r[2] for r in items),39)
        self.assertEqual(set(items),set(teacher.work_items(plan,list(reversed(states)))))

    def test_blinded_request_does_not_include_checkpoint_or_selector(self):
        root=Path(tempfile.mkdtemp(prefix='arm-quality-test-'));image=root/'image.png';image.write_bytes(b'image')
        row=dict(id='a',task='task',url='url',actor_prompt='context',screenshot=str(image))
        candidates=dict(actor='SECRET_ACTOR',checkpoint='SECRET_CKPT',selected_index=3,
                        candidates=[dict(thought='r',action='a') for _ in range(5)])
        msgs,order,bound=teacher.messages(row,candidates,0)
        self.assertNotIn('SECRET',json.dumps(msgs));self.assertNotIn('selected_index',json.dumps(msgs))
        self.assertGreater(bound,0);self.assertEqual(sorted(order),list(range(5)))
        self.assertNotEqual(order,teacher.messages(row,candidates,1)[1])


class CollectorContract(unittest.IsolatedAsyncioTestCase):
    async def test_full_panel_uses_shared_seeds_and_saves_permuted_selection(self):
        root=Path(tempfile.mkdtemp(prefix='arm-quality-collect-'));image=root/'image.png';image.write_bytes(b'image')
        rows=[dict(id=str(i),task_id=str(i),task='task',url='url',history=[],turn=i%4,
                   actor_prompt='prompt',actor_images=[dict(path=str(image))],screenshot=str(image),
                   prompt_sha256='p',screenshot_sha256='s') for i in range(64)]
        calls=[]
        async def infer(url,prompt,params,images,timeout_secs):
            calls.append(params)
            return ('<tool_call>{"name":"done","arguments":{}}</tool_call>',[1,2],[], 'stop')
        async def select(endpoint,payload,timeout):return {'raw':'{"selection":1}'}
        browser=ModuleType('openwebrl.generate_browser');browser._run_inference_step=infer
        native=ModuleType('slime.rollout.sglang_rollout');native.GenerateState=lambda args:SimpleNamespace(sampling_params={})
        types=ModuleType('slime.rollout.base_types');types.RolloutFnEvalOutput=lambda **kwargs:kwargs
        config=dict(actor='baseline20',checkpoint='checkpoint',output=str(root),selector_endpoint='offline')
        args=SimpleNamespace(sglang_router_ip='localhost',sglang_router_port=1)
        with patch.dict(sys.modules,{'openwebrl.generate_browser':browser,'slime.rollout.sglang_rollout':native,
                                    'slime.rollout.base_types':types}),patch.object(audit,'validate_inputs',return_value=rows), \
             patch.object(audit,'request_selection_result',select),patch('builtins.print'):
            result=await audit.collect(args,config)
        self.assertEqual(len(calls),320);self.assertEqual(len(list((root/'records').glob('*.json'))),64)
        self.assertEqual(result['metrics']['arm_quality/states'],64)
        saved=json.loads((root/'records/0.json').read_text())
        self.assertEqual(saved['selected_index'],audit.permutation('0','selector')[0])
        self.assertEqual(saved['candidate_seeds'],[audit.candidate_seed(20260921,'0',0,i) for i in range(5)])
        self.assertEqual(json.loads((root/'collection-complete.json').read_text())['optimizer_updates'],0)


if __name__=='__main__':unittest.main()
