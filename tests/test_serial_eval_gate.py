import copy
import json
from pathlib import Path
import tempfile
import unittest

from openwebrl.serial_eval_gate import (validate_gate,load_gate,digest,
                                      generation_pass,browser_pass)


class SerialGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.artifact=Path(self.tmp.name)/'outputs.json';self.artifact.write_text('{}')
        self.identity={'variant':'all5','runtime_hashes':{'generator':'fixed'},'model':'fixed-model'}
        self.tasks=[f't{i}' for i in range(5)]
        self.report=dict(schema_version=1,stage='free_generation',passed=True,identity=self.identity,
            checks=dict(adapter_merge_parity=True,free_generation=True),pilot_task_ids=self.tasks,
            artifacts={str(self.artifact):digest(self.artifact)})

    def test_missing_failed_and_teacher_forced_only_gate_cannot_launch(self):
        with self.assertRaises(ValueError):load_gate(None,self.identity)
        for edit in [{'passed':False},{'passed':'true'},{'checks':{'adapter_merge_parity':True}}]:
            with self.assertRaises(ValueError):validate_gate(self.report|edit,self.identity,pilot=True,task_ids=self.tasks)

    def test_generation_proof_only_allows_exact_small_pilot(self):
        validate_gate(self.report,self.identity,pilot=True,task_ids=self.tasks)
        with self.assertRaises(ValueError):validate_gate(self.report,self.identity,task_ids=list(range(100)))
        with self.assertRaises(ValueError):validate_gate(self.report,self.identity,pilot=True,task_ids=self.tasks+['t5'])
        with self.assertRaises(ValueError):validate_gate(self.report,self.identity,pilot=True,task_ids=list(reversed(self.tasks)))

    def test_browser_proof_allows_broad_eval_but_rejects_stale_evidence(self):
        r=copy.deepcopy(self.report);r.update(stage='browser_pilot');r['checks']['browser_pilot']=True
        validate_gate(r,self.identity,task_ids=list(range(100)))
        with self.assertRaises(ValueError):validate_gate(r,self.identity|{'model':'other-model'})
        self.artifact.write_text('{"changed":true}')
        with self.assertRaises(ValueError):validate_gate(r,self.identity)

    def test_generation_requires_all_states_both_temperatures_and_actual_format(self):
        ids=['s0','s1'];rows=[dict(id=i,temperature=t,valid=True,finish='stop',prompt_tokens_match=True) for i in ids for t in [0.,.7]]
        self.assertTrue(generation_pass(rows,ids))
        self.assertFalse(generation_pass(rows[:-1],ids))
        self.assertFalse(generation_pass(rows+[rows[0]],ids))
        for key,value in [('valid',False),('finish','length'),('prompt_tokens_match',False)]:
            bad=copy.deepcopy(rows);bad[0][key]=value;self.assertFalse(generation_pass(bad,ids))

    def test_browser_requires_actual_execution_and_multiturn_projected_history(self):
        rows=[dict(task_id=i,valid=True,terminate_reason='max_steps_exhausted',metadata=dict(
            serial_action_executed=True,serial_history_projected=True,
            messages=[dict(role='assistant',content='<think>chosen</think><tool_call>{}</tool_call>')]*2)) for i in self.tasks]
        self.assertTrue(browser_pass(rows,self.tasks)) # zero task success can still exercise mechanics
        bad=copy.deepcopy(rows);bad[0]['terminate_reason']='serial_protocol_error: wrong count'
        self.assertFalse(browser_pass(bad,self.tasks))
        bad=copy.deepcopy(rows);bad[0]['metadata']['messages'][0]['content']='<alternative id="1">hypothetical</alternative>'
        self.assertFalse(browser_pass(bad,self.tasks))
        bad=copy.deepcopy(rows)
        for r in bad:r['metadata']['serial_action_executed']=False
        self.assertFalse(browser_pass(bad,self.tasks))


if __name__=='__main__':unittest.main()
