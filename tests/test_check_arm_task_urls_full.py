from pathlib import Path
import asyncio
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import check_arm_task_urls_full as full


class FullURLTests(unittest.TestCase):
    def test_balanced_disjoint_shards_reuse_every_pilot_id(self):
        probes=[dict(probe_id=f'{i:024d}') for i in range(31)]
        reuse={probes[i]['probe_id'] for i in [0,4,13]}
        parts=full.split_probes(probes,reuse)
        flat=[p['probe_id'] for shard in parts for p in shard]
        self.assertEqual(len(flat),len(set(flat)))
        self.assertEqual(set(flat),{p['probe_id'] for p in probes}-reuse)
        self.assertLessEqual(max(map(len,parts))-min(map(len,parts)),1)
        self.assertEqual(parts,full.split_probes(list(reversed(probes)),reuse))

    def test_retry_budget_counts_failed_and_current_attempts(self):
        rows=full.accounting_rows('11|FAILED|650|1:0|\n12|RUNNING|300|0:0|\n',['11','12'])
        self.assertEqual(full.BUDGET_SECONDS-sum(x['elapsed_seconds'] for x in rows),45850)

    def test_missing_accounting_fails_closed(self):
        with self.assertRaises(ValueError):full.accounting_rows('11|FAILED|650|1:0|\n',['11','12'])

    def test_infrastructure_error_is_not_a_dead_website_observation(self):
        h=[dict(classification='inconclusive',reason='browser_infrastructure'),
           dict(classification='unavailable',reason='navigation_error')]
        self.assertIsNone(full.final_decision(full.normal_observations(h)))


class ResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_available_decision_never_browses_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);full.write(p/'decision.json',dict(url='https://example.org',probe_id='abc',decision='page_available'))
            result=await full.check_probe(None,dict(url='https://example.org',probe_id='abc'),p)
            self.assertEqual(result['decision'],'page_available')

    async def test_resume_preserves_failure_evidence_and_adds_fresh_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            infra=dict(classification='inconclusive',reason='browser_infrastructure')
            bad=dict(classification='unavailable',reason='navigation_error')
            full.write(p/'attempt0001/result.json',infra)
            full.write(p/'attempt0002/result.json',bad)
            class Worker:
                async def attempt(self,manifest,probe,directory):
                    self.directory=directory
                    full.write(directory/'result.json',bad)
                    return bad
            worker=Worker();probe=dict(url='https://example.org',probe_id='abc',chunk='unused',task_ids=['x'],sources={'test':1})
            with patch.object(full.asyncio,'sleep',return_value=None):
                result=await full.check_probe(worker,probe,p)
            self.assertEqual(worker.directory.name,'attempt0003')
            self.assertEqual(result['decision'],'unavailable_after_retry')
            self.assertEqual(result['attempts'],3)
            self.assertEqual(json.loads((p/'attempt0001/result.json').read_text()),infra)

    async def test_infrastructure_failure_cannot_create_final_site_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            class Worker:
                async def attempt(self,manifest,probe,directory):
                    row=dict(classification='inconclusive',reason='browser_infrastructure')
                    full.write(directory/'result.json',row);return row
            probe=dict(url='https://example.org',probe_id='abc',chunk='unused',task_ids=['x'],sources={'test':1})
            with patch.object(full.asyncio,'sleep',return_value=None):
                with self.assertRaisesRegex(RuntimeError,'infrastructure'):
                    await full.check_probe(Worker(),probe,p)
            self.assertFalse((p/'decision.json').exists())
            self.assertEqual(len(list(p.glob('attempt*/result.json'))),2)


if __name__=='__main__':unittest.main()
