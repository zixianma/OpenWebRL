"""Dataset identity and direct engine checks; no actors, browser or judge."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from openwebrl.action_credit import digest
from openwebrl.action_credit_collect import FrozenActor
from openwebrl.action_credit_task_inputs import NativeTaskInputs, direct_engine_args


class TaskInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        p = self.root/'tasks.parquet'
        pq.write_table(pa.Table.from_pylist([
            dict(prompt=[dict(role='user', content='a')], metadata=dict(task_id='task-a', task='a', start_url='https://example.org')),
            dict(prompt=[dict(role='user', content='b')], metadata=dict(task_id='task-b', task='b', start_url='https://example.net'))]),p)
        self.dataset = dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        self.order = dict(source_sha256=self.dataset['sha256'], original_manifest_task_order=['task-a','task-b'],
            iterations=[dict(groups=[dict(source_index=1,task_id='task-b')])])
        self.args = dict(max_steps=15, context_num_screenshots=1, turn_history_reasoning_mode='full',
            browser_history_reasoning_max_turns=-1, browser_include_tool_response=True,
            browser_history_env_feedback_max_chars=4000, rollout_max_context_len=32768,
            max_consecutive_parse_failures=3, serial_action_variant=None, browser_response_format_mode='slime')

    def load(self):
        p=self.root/'order.json';p.write_text(json.dumps(self.order))
        return NativeTaskInputs(dataset_reference=self.dataset,
            task_order_reference=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()),
            browser_config={},browser_manifest={},native_args=self.args)

    def test_source_membership_and_order_cannot_be_substituted(self):
        inputs=self.load()
        sample=inputs.sample(SimpleNamespace(task_id='task-b',request_id='specific-request'))
        self.assertEqual(sample['metadata']['task'],'b')
        self.assertEqual(sample['index'],'specific-request')
        sample['metadata']['task']='mutated'
        self.assertEqual(inputs.rows['task-b']['metadata']['task'],'b')
        self.order['original_manifest_task_order'].reverse()
        with self.assertRaisesRegex(ValueError,'membership/order'):
            self.load()

    def test_source_file_and_fixed_group_index_are_verified(self):
        self.order['iterations'][0]['groups'][0]['source_index']=0
        with self.assertRaisesRegex(ValueError,'source index'):
            self.load()
        self.dataset['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'input file changed'):
            self.load()

    def test_only_exact_assigned_endpoint_is_allowed(self):
        url='http://127.0.0.1:32123/generate'
        actor=FrozenActor(*([digest('fixture')]*4),endpoint_sha256=digest(url))
        out=direct_engine_args(self.args,actor,url)
        self.assertEqual(out['sglang_router_port'],32123)
        self.assertEqual(out['max_steps'],15)
        with self.assertRaisesRegex(ValueError,'Direct engine'):
            direct_engine_args(self.args,actor,url.replace('32123','32124'))
        with self.assertRaisesRegex(ValueError,'Direct engine'):
            direct_engine_args(self.args,replace(actor,endpoint_sha256=digest(url+'?fallback=1')),url+'?fallback=1')
