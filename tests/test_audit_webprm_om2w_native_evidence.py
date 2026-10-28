"""Offline negative controls for declared OM2W turn-limit evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from scripts.audit_webprm_om2w_native_evidence import fallback_evidence, cleanup_result, audit_control_selection


class ControlSelectionTests(unittest.TestCase):
    def test_replays_surviving_pool_indices_and_terminal_override(self):
        import random
        for selector in ('first5','random5'):
            rows=[]
            for turn in range(6):
                pool=[dict(raw_slot=1),dict(raw_slot=4)]
                chosen=0 if selector=='first5' else random.Random(42+turn).randrange(2)
                rows.append(dict(proposal=dict(candidates=pool,terminal=False,requested_raw_slots=5),
                                 selected_index=chosen,selection_trace=[]))
            rows.append(dict(proposal=dict(candidates=[{}],terminal=True),selected_index=0,selection_trace=[]))
            plan=dict(selector=selector,seed=42)
            audit_control_selection(plan,rows)
            bad=deepcopy(rows);bad[0]['selected_index']=1-bad[0]['selected_index']
            with self.assertRaises(ValueError):audit_control_selection(plan,bad)
            bad=deepcopy(rows);bad[0]['proposal']['requested_raw_slots']=1
            with self.assertRaises(ValueError):audit_control_selection(plan,bad)
            bad=deepcopy(rows);bad[0]['selection_trace']=[dict(critic='unapproved')]
            with self.assertRaises(ValueError):audit_control_selection(plan,bad)


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(dir='/tmp');self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name)
        self.protocol='released_viewer_na_fallback_v1';self.origin='released_viewer_normal_turn_limit'
        self.action='send_msg_to_user [N/A]';self.code='saved synthetic native code'
        self.trajectory=[{'observation':{}},{'action_type':'send_msg_to_user','answer':'N/A','raw_action':self.action,
                         'origin':self.origin,'protocol':self.protocol}]
        self.rollout=[dict(selected_index=None,proposal=dict(candidates=[])) for _ in range(30)]
        self.fallback=dict(schema_version=1,protocol=self.protocol,origin=self.origin,termination_reason='turn_limit',
            action=self.action,answer='N/A',actor_selected=False,actor_turns=30,source_revision='revision',
            source_file='viewer/src/server/service/executors/step_loop_executor.py',status='executed',
            pre_fallback_trajectory_sha256='trajectory-hash',chat_messages_before=[],
            chat_messages_after=[dict(role='assistant',message='N/A')],appended_messages=[dict(role='assistant',message='N/A')],
            native_dispatch=dict(action=self.action,code=self.code,code_sha256=hashlib.sha256(self.code.encode()).hexdigest(),element_bounds={}))
        self.result=dict(protocol_fallback=self.fallback,turn_limit_protocol=self.protocol,turns=30)
        self.plan=dict(max_turns=30,arbiter_source='synthetic')
        self.harness=SimpleNamespace(parse_action=lambda x:dict(kind='send_message_to_user' if x==self.action else 'click'),key_mapping={})

    def check(self):
        (self.folder/'protocol-fallback.json').write_text(json.dumps(self.fallback))
        return fallback_evidence(self.folder,self.result,self.plan,self.trajectory,self.rollout,self.harness,
            lambda *a,**k:self.code,lambda x:'trajectory-hash','revision')

    def test_declared_executed_thirty_turn_fallback(self):self.assertEqual(self.check(),'protocol_turn_limit')

    def test_fallback_cannot_overwrite_actor_terminal(self):
        self.rollout[0]=dict(selected_index=0,proposal=dict(candidates=[dict(action=self.action)]))
        with self.assertRaises(ValueError):self.check()

    def test_wrong_chat_history_native_code_or_exhaustion_rejected(self):
        original=deepcopy(self.fallback)
        self.fallback['appended_messages']=[dict(role='assistant',message='Other')]
        with self.assertRaises(ValueError):self.check()
        self.fallback.clear();self.fallback.update(deepcopy(original));self.fallback['native_dispatch']['code']='changed'
        with self.assertRaises(ValueError):self.check()
        self.fallback.clear();self.fallback.update(deepcopy(original));self.rollout.pop()
        with self.assertRaises(ValueError):self.check()

    def test_absent_declaration_cannot_hide_fallback_file(self):
        self.result['protocol_fallback']=None
        with self.assertRaises(ValueError):self.check()

    def test_absent_post_close_file_requires_clean_watchdog_after_settled_verdict(self):
        watchdog=dict(reason='cleanup_timeout',last_phase=dict(phase='cleanup',identity_sha256='identity'),
                      remaining_owned_processes={},child_exitcode=-15)
        result=dict(cleanup_status='interrupted',browser_close_error='cleanup_timeout',watchdog=deepcopy(watchdog),identity_sha256='identity')
        path=self.folder/'watchdog.json';path.write_text(json.dumps(watchdog))
        self.assertEqual(cleanup_result(self.folder,result,'identity')[1],
                         'watchdog_reaped_after_settled_verdict')
        watchdog['remaining_owned_processes']={'42':{}}
        result['watchdog']=deepcopy(watchdog);path.write_text(json.dumps(watchdog))
        with self.assertRaises(ValueError):cleanup_result(self.folder,result,'identity')


if __name__=='__main__':unittest.main()
