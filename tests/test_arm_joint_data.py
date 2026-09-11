import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_arm_joint_data import Builder, pair_reason, pair_type


def call(name, **args): return [{'name':name,'arguments':args}]


class PairRules(unittest.TestCase):
    def test_coordinate_pairs_are_not_assumed_semantically_different(self):
        a=call('click',point_2d=[10,10])
        self.assertEqual(pair_reason(a,a),'identical_action')
        self.assertEqual(pair_reason(a,call('click',point_2d=[13,14])),'near_coordinates')
        self.assertEqual(pair_reason(a,call('click',point_2d=[900,900])),'coordinate_only_review')

    def test_terminal_answer_pairs_and_termination_decisions(self):
        self.assertEqual(pair_reason(call('done',message='A'),call('done',message='B')),'done_vs_done')
        self.assertEqual(pair_reason(call('done',message='A'),call('scroll',direction='down',amount=.5)),'eligible')
        self.assertEqual(pair_type(call('done',message='A'),call('scroll',direction='down',amount=.5)),'termination_vs_continue')

    def test_scroll_direction_is_distinct_but_amount_is_ambiguous(self):
        a=call('scroll',direction='down',amount=.5)
        self.assertEqual(pair_reason(a,call('scroll',direction='down',amount=.8)),'scroll_amount_review')
        self.assertEqual(pair_reason(a,call('scroll',direction='up',amount=.5)),'eligible')


@unittest.skipUnless(importlib.util.find_spec('tokenizers'), 'actor tokenizer environment required')
class TokenAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.b=Builder(SimpleNamespace(output=Path(cls.temp.name)/'data',repo=Path(__file__).resolve().parents[1],
          model=Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')))
        cls.schema={'click':{'type':'object','required':['point_2d'],'properties':{'point_2d':{'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':2}}}}
        cls.text='I see a button.\n</think>\n\n<tool_call>\n{"name":"click","arguments":{"point_2d":[10,20]}}\n</tool_call>'

    @classmethod
    def tearDownClass(cls):
        cls.b.db.close();cls.b.ledger.close();cls.temp.cleanup()

    def test_hf_completion_is_explicitly_appended_and_reasoning_not_scored(self):
        x=self.b.candidate(self.text,self.schema,'Piotr')
        self.assertTrue(x['completion_boundary_appended'])
        selected=[x['token_ids'][i] for i in x['action_token_indices']]
        masked=self.b.tok.decode(selected,skip_special_tokens=False)
        self.assertTrue(masked.startswith('<tool_call>'))
        self.assertTrue(masked.endswith('<|im_end|>'))
        self.assertNotIn('button',masked)

    def test_generated_noncanonical_tokens_preserved(self):
        # Split the reasoning prefix into individually encoded characters;
        # generation may legally use a different segmentation than encode(text).
        prefix='I see a button.'
        text=self.text+'<|im_end|>\n'
        ids=sum([self.b.tok.encode(c,add_special_tokens=False).ids for c in prefix],[])
        ids+=self.b.tok.encode(text[len(prefix):],add_special_tokens=False).ids
        self.assertNotEqual(ids,self.b.tok.encode(text,add_special_tokens=False).ids)
        x=self.b.candidate(text,self.schema,'C2','stop',ids)
        self.assertEqual(x['token_ids'],ids)

    def test_corrupted_generated_token_text_rejected(self):
        with self.assertRaisesRegex(ValueError,'response_token_text_mismatch'):
            self.b.candidate(self.text,self.schema,'C2','stop',[1,2,3])

    def test_finish_and_structural_checks(self):
        with self.assertRaisesRegex(ValueError,'truncated_response'):
            self.b.candidate(self.text,self.schema,'C2','length')
        with self.assertRaisesRegex(ValueError,'text_after_final_action'):
            self.b.candidate(self.text+' unfinished continuation',self.schema,'Piotr')


if __name__=='__main__':unittest.main()
