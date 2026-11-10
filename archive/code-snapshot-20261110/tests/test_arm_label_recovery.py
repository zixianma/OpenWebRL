"""Known truncations may retry; unknown outcomes and budget overruns may not."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from arm_refresh_label_recovery import recover,paths,recovered_label
from label_arm_refresh import parsed_label
from prepare_arm_forward_transfer import sha,write


def response(reason='stop',content='{"selection": 2}'):
    return dict(model='gpt-5.5-2026-04-23',choices=[dict(finish_reason=reason,message=dict(content=content,refusal=None))],
                usage=dict(prompt_tokens=100,completion_tokens=2048 if reason=='length' else 30))


class LabelRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.request=dict(custom_id='x',body=dict(model='gpt-5.5',messages=[],max_completion_tokens=2048))
        self.entry=dict(custom_id='x',state_id='state',split='train',treatment='main',permutation=[4,3,2,1,0],pilot=False,
                        input_tokens_reserved=1000,body_sha256=sha(json.dumps(self.request['body'],sort_keys=True).encode()))
        self.original=response('length','');self.manifest=dict(cost_ceiling_usd=191.04);self.approval=dict(api_budget_usd=225)
        self.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=SimpleNamespace(model_dump=lambda **_:response())))))
    def tearDown(self):self.temp.cleanup()
    def run_recovery(self):return recover(self.root,self.entry,self.request,self.original,self.manifest,self.approval,self.client,parsed_label)

    def test_retry_preserves_original_and_bills_standard(self):
        before=copy.deepcopy(self.original);label=self.run_recovery()
        self.assertEqual(self.original,before);self.assertTrue(label['recovered_from_length'])
        self.assertEqual(label['recovery_completion_cap'],4096)
        self.assertEqual(label['usd'],.0014)
        self.assertEqual(self.client.chat.completions.create.call_args.kwargs['max_completion_tokens'],4096)
        self.run_recovery();self.client.chat.completions.create.assert_called_once()

    def test_budget_counts_reserved_unknown_outcomes(self):
        _,ticket=paths(self.root,'another',4096);write(ticket,dict(usd_reserved=34))
        with self.assertRaisesRegex(ValueError,'exceed approved'):self.run_recovery()
        self.client.chat.completions.create.assert_not_called()

    def test_unknown_outcome_never_retried(self):
        _,ticket=paths(self.root,'x',4096);write(ticket,dict(usd_reserved=.1))
        with self.assertRaisesRegex(ValueError,'Unknown recovery outcome'):self.run_recovery()
        self.client.chat.completions.create.assert_not_called()

    def test_non_length_failure_not_retried(self):
        self.original=response('content_filter','')
        with self.assertRaisesRegex(ValueError,'Only known'):self.run_recovery()
        self.client.chat.completions.create.assert_not_called()

    def test_unchanged_request_required(self):
        self.request['body']['model']='other'
        with self.assertRaisesRegex(ValueError,'frozen manifest'):self.run_recovery()
        self.client.chat.completions.create.assert_not_called()

    def test_failed_first_retry_can_use_bounded_second(self):
        self.client.chat.completions.create.side_effect=[SimpleNamespace(model_dump=lambda **_:response('length','')),SimpleNamespace(model_dump=lambda **_:response())]
        label=self.run_recovery();self.assertEqual(label['recovery_completion_cap'],8192)
        self.assertEqual(self.client.chat.completions.create.call_count,2)

    def test_valid_original_never_replaced(self):
        self.run_recovery();body=response(content='{"selection": 1}')
        self.assertEqual(recovered_label(self.root,self.entry,body,parsed_label)['selected_slot'],0)

if __name__=='__main__':unittest.main()
