"""Synthetic terminal-judge binding controls; no provider or browser is used."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.audit_webprm_om2w_saved_calls import judge_binding, audit_local_preflight

MODEL='o4-mini-2025-04-16'


def messages(intent,trajectory,image):
    return [dict(role='user',content=dict(intent=intent,trajectory=trajectory,image=image.hex()))]


def trajectory(rollout,fallback,termination):return json.dumps([rollout,fallback,termination],sort_keys=True)


class JudgeBindingTests(unittest.TestCase):
    def setUp(self):
        guard=patch('socket.socket.connect',side_effect=AssertionError('Network forbidden'))
        guard.start();self.addCleanup(guard.stop)
        self.temp=tempfile.TemporaryDirectory(dir='/tmp');self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name);self.task=SimpleNamespace(actor_view=dict(intent='Synthetic task'))
        rollout=[dict(action='Synthetic executed action')]
        (self.folder/'rollout.json').write_text(json.dumps(rollout))
        (self.folder/'judge-final.jpg').write_bytes(b'synthetic-frame')
        self.record=dict(termination='stop',raw='SUCCESS',usage=dict(prompt_tokens=10,completion_tokens=2))
        self.save_record()
        payload=dict(model=MODEL,messages=messages(self.task.actor_view['intent'],trajectory(rollout,None,'stop'),b'synthetic-frame'),max_completion_tokens=4096)
        self.pairs=[dict(request=dict(purpose='om2w_judge',payload=payload,reservation=7),
            response=dict(response=dict(model=MODEL,usage=deepcopy(self.record['usage']),choices=[dict(finish_reason='stop',message=dict(content='SUCCESS'))])),
            broker=dict(reservation_id=7))]

    def save_record(self): (self.folder/'om2w-judge.json').write_text(json.dumps(self.record))

    def run_check(self):
        return judge_binding(self.task,self.folder,self.pairs,judge_messages=messages,judge_trajectory=trajectory,model=MODEL,tokens=4096)

    def test_exact_saved_task_actions_frame_and_reply_bind_to_paid_judge(self):
        self.assertEqual(self.run_check(),7)

    def test_changed_screenshot_is_rejected(self):
        (self.folder/'judge-final.jpg').write_bytes(b'different-frame')
        with self.assertRaises(ValueError):self.run_check()

    def test_changed_intent_or_actual_payload_is_rejected(self):
        self.task.actor_view['intent']='Different task'
        with self.assertRaises(ValueError):self.run_check()

    def test_changed_executed_history_is_rejected(self):
        (self.folder/'rollout.json').write_text(json.dumps([dict(action='Different action')]))
        with self.assertRaises(ValueError):self.run_check()

    def test_changed_saved_verdict_cannot_replace_provider_reply(self):
        self.record['raw']='FAILURE';self.save_record()
        with self.assertRaises(ValueError):self.run_check()

    def test_changed_reply_usage_model_or_finish_is_rejected(self):
        original=deepcopy(self.pairs)
        for key,value in [('model','wrong-model'),('usage',dict(prompt_tokens=99,completion_tokens=2))]:
            self.pairs=deepcopy(original);self.pairs[0]['response']['response'][key]=value
            with self.assertRaises(ValueError):self.run_check()
        self.pairs=deepcopy(original);self.pairs[0]['response']['response']['choices'][0]['finish_reason']='length'
        with self.assertRaises(ValueError):self.run_check()

    def test_duplicate_or_absent_judge_request_is_rejected(self):
        original=deepcopy(self.pairs)
        self.pairs=[]
        with self.assertRaises(ValueError):self.run_check()
        self.pairs=original+deepcopy(original)
        with self.assertRaises(ValueError):self.run_check()


class StartupEpochTests(unittest.TestCase):
    def setUp(self):
        self.plan={'frozen':'plan'};payload={'model':'local-model'}
        self.meta=dict(request_id='r1',cell_id='local_preflight',local=True,model='local-model',
                       payload_sha256='payload-hash',purpose='check',policy_sha256='policy-1')
        start=dict(self.meta,identity_sha256='plan-hash')
        done=dict(start,journal_sha256='api-prefix',reservation_state='local',reservation_id=None,
                  budget_owner='webprm_worker.completion',usage={'total_tokens':3})
        self.groups={'r1':dict(dispatch_started=start,dispatch_completed=done)}
        self.chunk=dict(request=dict(payload=payload,local=True,reservation=None,purpose='check'),
                        response=dict(response=dict(model='local-model',usage={'total_tokens':3})),
                        error=None,terminal_prefix_sha256='api-prefix')
        self.replay=dict(epochs={'epoch-1':dict(broker_policy_sha256='policy-1'),
                                 'epoch-2':dict(broker_policy_sha256='policy-2')},
            dispatches={'r1':dict(self.meta,epoch_id='epoch-1',reservation_id=None),
                        'r2':dict(self.meta,request_id='r2',epoch_id='epoch-2',reservation_id=None)},
            charged=5,cap=10,overruns=0)
        self.helper=SimpleNamespace(_digest=lambda x:'plan-hash' if x==self.plan else 'payload-hash',
                                   _api_chunks=lambda p:([self.chunk],'api-file-hash',123))

    def check(self):return audit_local_preflight(self.plan,self.groups,None,self.replay,self.helper)

    def test_reused_diagnostic_cell_is_joined_by_epoch_without_changing_global_ledger(self):
        before=deepcopy(self.replay);result=self.check()
        self.assertEqual(result['local_calls'],1);self.assertEqual(result['epoch_id'],'epoch-1')
        self.assertEqual(self.replay,before)

    def test_missing_or_wrong_epoch_dispatch_is_rejected(self):
        self.replay['dispatches']['r1']['epoch_id']='epoch-2'
        with self.assertRaises(ValueError):self.check()

    def test_paid_or_mismatched_startup_receipt_is_rejected(self):
        for key,value in [('reservation_state','settled'),('journal_sha256','wrong')]:
            original=deepcopy(self.groups)
            self.groups['r1']['dispatch_completed'][key]=value
            with self.assertRaises(ValueError):self.check()
            self.groups=original
        self.replay['dispatches']['r1']['reservation_id']=4
        with self.assertRaises(ValueError):self.check()


if __name__=='__main__':unittest.main()
