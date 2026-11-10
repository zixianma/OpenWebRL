import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openwebrl.arm_prefix_pilot import PREFIXES,PrefixSelector,condition_order,summarize,verify_archives


class HandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_prefix_common_seeds_and_no_selector_after_handoff(self):
        seeds,selector_calls = [],[]
        async def infer(url,prompt,params,images,**kwargs):
            seeds.append(params['sampling_seed'])
            return ('reasoning\n<tool_call>{"name":"click"}</tool_call>',[1,2],[], 'stop')
        async def select(*args):
            selector_calls.append(args)
            return {'raw':'{"selection":2}'}
        with tempfile.TemporaryDirectory() as directory:
            baseline = PrefixSelector(0,'http://unused',Path(directory)/'base',42)
            guided = PrefixSelector(2,'http://unused',Path(directory)/'guide',42)
            kwargs = dict(infer=infer,url='unused',input_text='prompt',sampling_params={},images=[],
                observation={'screenshot':b'png','active_tab_url':'https://example.test/'},
                history=[],task='task',task_id='task',turn=0,timeout=1)
            await baseline(**kwargs)
            first_seed = seeds[0]
            with patch('openwebrl.arm_task_success.request_selection_result',select):
                for turn in range(5):
                    _,metadata = await guided(**dict(kwargs,turn=turn,history=['past']*turn))
                    self.assertEqual(metadata['guided'],turn<2)
                    self.assertEqual(metadata['eligible_for_future_suffix_loss'],turn>=2)
            self.assertEqual(first_seed,seeds[1])
            self.assertEqual(len(selector_calls),2)
            self.assertEqual(guided.actor_requests,13)  # 2*5 + 3*1
            self.assertEqual(guided.suffix_requests,3)
            self.assertEqual(guided.handoff['turn'],2)
            self.assertEqual(guided.handoff['history_turns'],2)
            self.assertEqual(guided.actor_output_tokens,26)
            self.assertTrue((Path(directory)/'guide/handoff.json').is_file())

    async def test_bad_selector_halts_without_executing_fallback(self):
        async def infer(*args,**kwargs):return ('response',[1],[], 'stop')
        async def select(*args):return {'raw':'unparseable'}
        with tempfile.TemporaryDirectory() as directory:
            selector = PrefixSelector(2,'http://unused',directory,42)
            with patch('openwebrl.arm_task_success.request_selection_result',select):
                with self.assertRaises(ValueError):
                    await selector(infer=infer,url='unused',input_text='prompt',sampling_params={},images=[],
                        observation={'screenshot':b'png'},history=[],task='task',task_id='a',turn=0,timeout=1)
            self.assertEqual(selector.turns,[])
            self.assertIsNone(selector.handoff)

    async def test_noncontiguous_turns_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            selector = PrefixSelector(2,'http://unused',directory,42)
            with self.assertRaisesRegex(ValueError,'contiguous'):
                await selector(infer=None,turn=1)


def records():
    return [dict(task_id=t,prefix=k,repeat=r,policy_id='p',judge_id='gpt-4.1/action_history',
        valid=True,reward=0.,actor_suffix_turns=2,actor_requests=2,actor_output_tokens=20,
        selector_calls=k,elapsed_seconds=1.) for t in ('a','b') for k in PREFIXES for r in (0,1)]


class MetricTests(unittest.TestCase):
    def test_invalid_denominators_and_prefix_only_successes(self):
        rows=records()
        arm=[r for r in rows if r['prefix']==2]
        arm[0].update(valid=False,reward=None,actor_suffix_turns=0)
        arm[1].update(reward=1.,actor_suffix_turns=0)
        arm[2].update(reward=1.)
        x=summarize(rows,['a','b'],'p')
        m=x['by_prefix']['2']
        self.assertEqual(m['overall'],.5)
        self.assertAlmostEqual(m['valid_only'],2/3)
        self.assertEqual(m['prefix_only_successes'],1)
        self.assertEqual(m['suffix_success_per_attempt'],.25)
        self.assertEqual(x['paired']['2']['jointly_valid_attempts'],3)
        self.assertEqual(x['paired']['2']['overall_delta'],.5)

    def test_duplicate_partial_and_wrong_judge_rejected(self):
        rows=records()
        for bad in (rows[:-1],rows+[rows[0]], [dict(r,judge_id='other') for r in rows]):
            with self.assertRaises(ValueError):summarize(bad,['a','b'],'p')

    def test_all_conditions_randomized_reproducibly(self):
        a=condition_order(42,'a','compare')
        self.assertEqual(a,condition_order(42,'a','compare'))
        self.assertEqual(set(a),{(k,r) for k in PREFIXES for r in (0,1)})

    def test_missing_rollout_is_not_a_completed_result(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'Missing saved'):
                verify_archives([dict(task_id='a',artifact=directory)])


if __name__=='__main__':unittest.main()
