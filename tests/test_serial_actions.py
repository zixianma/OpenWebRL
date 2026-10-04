import copy
import unittest
from openwebrl.serial_actions import (action_distance, parse_serial, retained_indices,
                                     serialize, augment_prompt)


class SerialActionTests(unittest.TestCase):
    def setUp(self):
        self.candidates=[dict(reasoning=f'Option {i}: compare <tool_call> quoted markup.',
             actions=[dict(name='click',arguments={'point_2d':[i*100,50]})]) for i in range(5)]

    def test_round_trip_and_only_final_executes(self):
        for variant in ['all5','diverse3']:
            keep=retained_indices(self.candidates,2,variant,'test')
            text,_=serialize(self.candidates,keep,2)
            final,history,info=parse_serial(text,variant)
            self.assertEqual(text.count('<tool_call>'),1)
            self.assertEqual(final.count('<tool_call>'),1)
            self.assertNotIn('<alternative',history)
            self.assertEqual(info['candidates'][info['selected_index']]['actions'],self.candidates[2]['actions'])
            self.assertEqual(parse_serial('<think>\n'+text,variant)[0],final)

    def test_mismatch_missing_and_hypothetical_calls_fail(self):
        text,_=serialize(self.candidates,list(range(5)),2)
        malformed=[text.replace('Selected alternative: 3','Selected alternative: 9'),
                   text.replace('Selected alternative: 3','Selected alternative: 1'),
                   text.split('</think>')[0],text+'trailing text',
                   text.replace('&lt;tool_call&gt;','<tool_call>',1),
                   text.replace('<alternative id="2">','<alternative id="1">'),
                   text.replace('</think>','</think></think>')]
        for value in malformed:
            with self.assertRaises(ValueError):parse_serial(value,'all5')

    def test_duplicate_winner_and_multi_call(self):
        cs=copy.deepcopy(self.candidates)
        cs[1]['actions']=copy.deepcopy(cs[2]['actions'])
        keep=retained_indices(cs,2,'diverse3','repeat')
        self.assertIn(2,keep);self.assertNotIn(1,keep)
        cs=[copy.deepcopy(cs[2]) for _ in range(5)]
        keep=retained_indices(cs,3,'diverse3','all-identical')
        self.assertEqual(keep,[3])
        cs[3]['actions'].append(dict(name='press_keys',arguments={'keys':['ENTER']}))
        text,_=serialize(cs,[3],3)
        final,_,_=parse_serial(text,'diverse3')
        self.assertEqual(final.count('<tool_call>'),2)

    def test_coordinate_jitter_ranked_but_not_merged(self):
        a=self.candidates[0]['actions'];b=copy.deepcopy(a);b[0]['arguments']['point_2d'][0]+=1
        self.assertGreater(action_distance(a,b),0)
        self.assertLess(action_distance(a,b),action_distance(a,self.candidates[4]['actions']))
        prompt='<|im_start|>system\npolicy<|im_end|>history<think>\n'
        result=augment_prompt(prompt,'all5')
        self.assertTrue(result.endswith('<|im_end|>history<think>\n'))

    def test_protocol_follows_tools_and_live_path_uses_same_augmentation(self):
        import ast
        from pathlib import Path
        from openwebrl.serial_actions import instructions
        prompt='<|im_start|>system\npolicy\n<tools>schema</tools>\nDefault tool instructions<|im_end|>\n<|im_start|>assistant\n<think>\n'
        value=augment_prompt(prompt,'all5')
        self.assertGreater(value.index('Serial alternatives'),value.index('Default tool instructions'))
        self.assertEqual(value.replace(instructions('all5'),''),prompt)
        tree=ast.parse((Path(__file__).resolve().parents[1]/'openwebrl/generate_browser.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='_generate_turn_sample_impl')
        names=[n.func.id for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertIn('augment_prompt',names)
        self.assertNotIn('instructions',names)


if __name__=='__main__':unittest.main()
