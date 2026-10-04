"""CPU checks for the one-call, two-distinct-action ARM experiment."""
import json
import tempfile
import time
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

from openwebrl import arm_turn_bonus as arm
from openwebrl.arm_turn_bonus_runtime import validate_config

PROMPT='<tools>\n'+json.dumps({'function':{'name':'click','parameters':{
    'type':'object','properties':{'x':{'type':'integer'}},'required':['x']}}})+'\n</tools>'


def output(action, response):
    return ('<think>response-'+str(response)+'</think><tool_call>'+json.dumps(
        {'name':'click','arguments':{'x':action}})+'</tool_call>',[100+response],[-.3],'stop')


def labeled(ids, selected):
    label=dict(eligible=True, policy_id='p', executed_index=0, selected_index=selected,
               candidate_gate='min2', **arm.candidate_bonus(ids,selected,'action_class'))
    return SimpleNamespace(metadata={'task_id':'train','arm_turn_bonus':label},
                           remove_sample=False,loss_mask=[1])


class Arithmetic(unittest.TestCase):
    def test_gate_is_opt_in_and_retains_all_five_responses(self):
        self.assertEqual(arm.candidate_minimum({}),5)
        self.assertEqual(arm.candidate_minimum({'candidate_gate':'min2'}),2)
        self.assertEqual(arm.credit_rule({'candidate_gate':'min2'}),'response_index')
        for n in range(2,6):
            outputs=[output(i%n,i) for i in range(5)]
            self.assertEqual(len(arm.eligible_actions(PROMPT,outputs,2)),5)
            if n<5:
                with self.assertRaisesRegex(ValueError,'duplicate_action'):
                    arm.eligible_actions(PROMPT,outputs)
        with self.assertRaisesRegex(ValueError,'single_action_class'):
            arm.eligible_actions(PROMPT,[output(0,i) for i in range(5)],2)

    def test_format_truncation_and_nonempty_checks_remain(self):
        clean=[output(i%2,i) for i in range(5)]
        for bad in [('missing tool',[1],[-.3],'stop'),(clean[0][0],[],[],'stop'),
                    (*clean[0][:3],'length'),
                    ('<tool_call>{"name":"unknown","arguments":{}}</tool_call>',[1],[-.3],'stop')]:
            rows=list(clean);rows[3]=bad
            with self.assertRaises(ValueError):arm.eligible_actions(PROMPT,rows,2)

    def test_bonus_centered_for_every_informative_partition(self):
        def partitions(prefix):
            if len(prefix)==5:
                yield prefix;return
            for k in range(max(prefix,default=-1)+2):yield from partitions(prefix+[k])
        count=0
        for ids in partitions([]):
            if len(set(ids))<2:continue
            count+=1
            values=[arm.action_class_bonus(ids,j)['unit_bonus'] for j in range(5)]
            self.assertAlmostEqual(sum(values),0)
            for j,value in enumerate(values):
                self.assertEqual(value>0,ids[j]==ids[0])
                self.assertLessEqual(abs(.5*value),.4)
        self.assertEqual(count,51)

    def test_equivalent_alternative_is_positive_and_reader_checks_provenance(self):
        row=labeled([0,0,1,1,1],1)
        self.assertAlmostEqual(arm.unit_bonus(row,'p'),.6)
        self.assertAlmostEqual(arm.apply_bonus([row],[0.],'p',.5)[0],.3)
        report=arm.panel([row],[1.],'p')
        self.assertEqual(report['selected_original_rate'],0)
        self.assertEqual(report['selected_executed_action_rate'],1)
        self.assertEqual(report['mean_chance_baseline'],.4)
        with self.assertRaises(ValueError):arm.unit_bonus(row,'stale-policy')
        for key,value in [('chance_baseline',.2),('executed_action_multiplicity',1),
                          ('unit_bonus',-.2),('bonus_kind','candidate_index_v1'),
                          ('candidate_gate','distinct5'),('selected_executed_action',False),
                          ('credit_assignment','response_index')]:
            bad=deepcopy(row);bad.metadata['arm_turn_bonus'][key]=value
            with self.assertRaises(ValueError):arm.unit_bonus(bad,'p')

    def test_legacy_values_and_zero_scale_unchanged(self):
        for j in range(5):
            expected=float(j==0)-.2
            self.assertEqual(arm.action_class_bonus(list(range(5)),j)['unit_bonus'],expected)
            row=SimpleNamespace(metadata={'arm_turn_bonus':dict(eligible=True,policy_id='p',executed_index=0,
                selected_index=j,unit_bonus=expected)},remove_sample=False,loss_mask=[1])
            self.assertEqual(arm.unit_bonus(row,'p'),expected)
            rewards=[.7];self.assertIs(arm.apply_bonus([row],rewards,'p',0),rewards)

    def test_invalid_semantics_fail_before_work(self):
        with self.assertRaises(ValueError):arm.ShadowSelector({'candidate_gate':'typo'},'t')
        for ids in [[0]*5,[0,1], [True,1,2,3,4], [0,1,2,3,5]]:
            with self.assertRaises(ValueError):arm.action_class_bonus(ids,0)
        config=dict(beta=.5,scored_fraction=.2,label_timeout_seconds=120,k=5,
                    max_pending_per_trajectory=2,shadow_only=True,candidate_gate='min2')
        validate_config(config)
        for gate in ('distinct5','min2'):
            for rule in ('response_index','action_class'):
                validate_config(dict(config,candidate_gate=gate,credit_assignment=rule))
        with self.assertRaises(ValueError):validate_config(dict(config,credit_assignment='typo'))
        config['candidate_gate']='typo'
        with self.assertRaises(ValueError):validate_config(config)

    def test_legacy_combined_audit_metadata_still_readable_but_not_launchable(self):
        row=labeled([0,0,1,2,3],1)
        label=row.metadata['arm_turn_bonus']
        label.pop('credit_assignment');label['candidate_gate']='min2_action_class'
        self.assertAlmostEqual(arm.unit_bonus(row,'p'),.6)
        with self.assertRaises(ValueError):arm.candidate_minimum(label)


class Pipeline(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, ids, chosen=1, malformed=False, gate='min2', rule='action_class'):
        with tempfile.TemporaryDirectory() as folder:
            config=dict(output=folder,policy_id='p',seed=42,scored_fraction=1.,
                        max_pending_per_trajectory=2,label_timeout_seconds=2,
                        selector_endpoint='unused',candidate_gate=gate)
            if rule is not None:config['credit_assignment']=rule
            state=dict(config=config,records=[],started=time.monotonic(),excluded_ids=set(),excluded_intents=set())
            outputs=[output(c,i) for i,c in enumerate(ids)];calls=[];requests=[]
            async def infer(url,prompt,params,images,timeout_secs=None):
                calls.append(params);return outputs[len(calls)-1]
            async def select(endpoint,payload,timeout,client):
                requests.append(payload)
                self.assertEqual(len(payload['candidates']),5)
                index=next(i for i,c in enumerate(payload['candidates']) if c['thought']=='response-'+str(chosen))
                return {'raw':'bad' if malformed else json.dumps({'selection':index+1})}
            async def browser(args,incoming,params):
                selector=args.browser_action_selector;selector.request=select
                selector.set_training_context({'parent_sample_index':10,'group_index':2})
                original,_=await selector(infer=infer,url='actor',input_text=PROMPT,sampling_params={},
                    images=['image'],observation={'screenshot':b'image','active_tab_url':'before'},
                    history=[],task='train',task_id='train',turn=0,timeout=2)
                self.assertIs(original,outputs[0])
                return [SimpleNamespace(index=10,metadata={'turn_index':0,'task_id':'train'},
                                        remove_sample=False,loss_mask=[1])]
            incoming=SimpleNamespace(index=77,metadata={'task_id':'train','intent':'train'})
            with patch.dict(sys.modules,{'openwebrl.generate_browser':SimpleNamespace(generate_turn_sample=browser)}), \
                    patch.object(arm,'state',return_value=state),patch.object(arm,'log_progress'):
                rows=await arm.generate(SimpleNamespace(),incoming,{})
            archived=json.loads((Path(folder)/'labels/77.json').read_text())['records'][0]
            # Persist/reload the exact metadata consumed by the reward reader.
            rows[0].metadata=json.loads(json.dumps(rows[0].metadata))
            return rows[0],archived,calls,requests

    async def test_duplicate_selection_one_call_and_durable_class_metadata(self):
        row,record,calls,requests=await self.run_case([0,0,1,1,1])
        self.assertEqual(len(calls),5);self.assertEqual(len(requests),1)
        self.assertEqual(record['candidate_requests'],4);self.assertEqual(record['selector_requests'],1)
        self.assertEqual(record['selected_index'],1)
        self.assertEqual(record['action_class_ids'],[0,0,1,1,1])
        self.assertAlmostEqual(arm.unit_bonus(row,'p'),.6)

    async def test_negative_labels_survive(self):
        row,_,_,requests=await self.run_case([0,0,1,1,1],chosen=2)
        self.assertEqual(len(requests),1)
        self.assertAlmostEqual(arm.unit_bonus(row,'p'),-.4)

    async def test_relaxed_gate_alone_preserves_response_credit(self):
        row,record,_,requests=await self.run_case([0,0,1,1,1],rule=None)
        self.assertEqual(len(requests),1)
        self.assertEqual(record['credit_assignment'],'response_index')
        self.assertEqual(arm.unit_bonus(row,'p'),-.2)
        report=arm.panel([row],[1.],'p')
        self.assertEqual(report['positive_bonus_rate'],0)
        self.assertEqual(report['selected_executed_action_rate'],1)
        self.assertEqual(report['mean_chance_baseline'],.2)

    async def test_strict_gate_is_independent_and_credit_matches_without_duplicates(self):
        for rule in ('response_index','action_class'):
            row,record,_,requests=await self.run_case([0,0,1,2,3],gate='distinct5',rule=rule)
            self.assertFalse(requests);self.assertFalse(record['eligible'])
            self.assertEqual(arm.unit_bonus(row,'p'),0)
            for chosen in (0,1):
                row,_,_,requests=await self.run_case([0,1,2,3,4],chosen=chosen,gate='distinct5',rule=rule)
                self.assertEqual(len(requests),1)
                self.assertEqual(arm.unit_bonus(row,'p'),float(chosen==0)-.2)

    async def test_single_action_has_no_selector_call_or_bonus(self):
        row,record,_,requests=await self.run_case([0]*5)
        self.assertFalse(requests);self.assertEqual(record['error'],'single_action_class')
        self.assertEqual(arm.unit_bonus(row,'p'),0)

    async def test_selector_parse_failure_keeps_outcome_only(self):
        row,record,_,requests=await self.run_case([0,0,1,1,1],malformed=True)
        self.assertEqual(len(requests),1);self.assertFalse(record['eligible'])
        self.assertEqual(arm.apply_bonus([row],[.7],'p',.5),[.7])


if __name__=='__main__':unittest.main()
