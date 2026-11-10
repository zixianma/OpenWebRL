"""Identity, leakage and teacher failure checks for the offline ARM pilot."""
import json
import unittest

from prepare_arm_forward_transfer import group_tasks, task_partition
from prepare_arm_refresh_teacher import cost_reservation
from label_arm_refresh import parsed_label
from run_arm_forward_transfer import target_text, paired_report


class ForwardTransferTests(unittest.TestCase):
    def test_alias_union_prevents_cross_split_leakage(self):
        data=[dict(task_key='1',task='Find blue shirts'),
              dict(task_key='2',task='  Find blue shirts  '),
              dict(task_key='2',task='Find a blue shirt')]
        result=group_tasks(data)
        self.assertEqual(len({r['task_group'] for r in result}),1)
        self.assertEqual(len({task_partition(r['task_group']) for r in result}),1)

    def test_teacher_permutation_is_inverted_for_training(self):
        request=dict(custom_id='test',state_id='state',split='train',treatment='main',
                     permutation=[3,0,4,1,2],pilot=True)
        body=dict(model='gpt-5.5-2026-04-23',choices=[dict(finish_reason='stop',
                  message=dict(content='{"selection": 2}'))],usage=dict(prompt_tokens=100,completion_tokens=10))
        label=parsed_label(request,body)
        self.assertEqual(label['selected_original'],0)
        self.assertEqual(target_text(dict(permutation=request['permutation']),label['selected_original']),
                         '{"selection": 2}<|im_end|>')
        body['choices'][0]['finish_reason']='length'
        with self.assertRaises(ValueError):parsed_label(request,body)
        body['choices'][0]['finish_reason']='stop'
        for output in ['{"selection": 0}','{"selection": true}','{"selection": 6}',
                       '{"selection":2,"uncertain":true}','']:
            body['choices'][0]['message']['content']=output
            with self.assertRaises(ValueError):parsed_label(request,body)

    def test_budget_counts_images_and_output_cap(self):
        messages=[dict(role='user',content=[dict(type='text',text='short'),
                  dict(type='image_url',image_url={'url':'data:image/png;base64,NOT_TOKENIZED_AS_TEXT'})])]
        standard=cost_reservation(messages,[1280,720],batch=False)
        batch=cost_reservation(messages,[1280,720],batch=True)
        self.assertAlmostEqual(standard['usd_reserved'],2*batch['usd_reserved'])
        self.assertGreater(standard['input_tokens_reserved'],1100)
        self.assertEqual(standard['max_completion_tokens'],2048)

    def test_paired_task_bootstrap_preserves_identical_predictions(self):
        records=[]
        for model in ['frozen','adapted']:
            for i in range(3):
                records.append(dict(pass_id=str(i),model=model,task_group=str(i//2),split='future',
                    treatment='main',teacher_original=0,selected_original=i%2,
                    action_correct=i%2==0,index_correct=i%2==0))
        result=paired_report(records)['future/main']
        self.assertEqual(result['tasks'],2)
        self.assertEqual(result['delta'],0)
        self.assertEqual(result['delta_task_bootstrap_95ci'],[0,0])
        with self.assertRaises(ValueError):paired_report(records[:-1])


if __name__=='__main__':unittest.main()
