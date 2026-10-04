import json
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path('scripts').resolve()))
import run_sol_inference as runner


class Inference(unittest.TestCase):
    def test_result_index_omits_images_and_rejects_duplicate_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            record=dict(task_id='one',valid=True,reward=1,screenshots=['large-payload'])
            (root/'a.json').write_text(json.dumps(record))
            self.assertEqual(runner.read_result_summaries(root),
                             {'one':dict(task_id='one',valid=True,reward=1)})
            (root/'b.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                runner.read_result_summaries(root)

    def test_failed_http_preflight_stops_before_gpu_actors_and_reaps_selector(self):
        from unittest.mock import patch
        import arm_launch_preflight
        with tempfile.TemporaryDirectory() as temp:
            p=runner.plan('12345'); p['output']=str(Path(temp)/'run')
            spawned=[]
            class Process:
                pid=424242
                returncode=None
                waited=False
                def poll(self): return self.returncode
                def wait(self,timeout=None): self.waited=True; return self.returncode
            child=Process()
            def spawn(command,**kw): spawned.append(command); return child
            def kill(*args): child.returncode=-15
            original=Path.read_text
            def read(path,*args,**kw):
                return '/slurm/job_12345/step_0' if str(path)=='/proc/self/cgroup' else original(path,*args,**kw)
            class Health(io.BytesIO): status=200
            with patch.dict(os.environ,SLURM_JOB_ID='12345',SLURM_STEP_ID='0',CUDA_VISIBLE_DEVICES='0,1',OPENAI_API_KEY='test'), \
                patch.object(Path,'read_text',read),patch.object(arm_launch_preflight,'require_receipt',return_value={'passed':True}), \
                patch.object(runner,'allocation',return_value=dict(cpus=16,allocated_memory_gib=240,maximum_seconds=7000)), \
                patch.object(runner.subprocess,'check_output',return_value='12345.batch\n12345.0\n'), \
                patch.object(runner.subprocess,'Popen',side_effect=spawn),patch.object(runner.os,'killpg',side_effect=kill), \
                patch.object(runner.urllib.request,'urlopen',return_value=Health(b'{}')), \
                patch.object(runner,'selector_preflight',side_effect=ValueError('bad HTTP binding')), \
                patch.object(runner,'report',return_value={}),patch('dotenv.dotenv_values',return_value={}):
                with self.assertRaisesRegex(ValueError,'bad HTTP'): runner.execute(p)
            self.assertEqual(len(spawned),1)
            self.assertIn('serve_sol_selector.py',' '.join(spawned[0]))
            self.assertTrue(child.waited)
            self.assertTrue(json.loads((Path(p['output'])/'status.json').read_text())['failed'])

    def test_pilots_and_shards_partition_original_300_with_same_protocol(self):
        p=runner.plan('CPU_CHECK')
        indices=[i for s in p['segments'] for i in s['indices']]
        self.assertEqual(sorted(indices),list(range(300)))
        self.assertEqual(len(set(indices)),300)
        self.assertEqual([len(s['indices']) for s in p['segments']],[1,1,149,149])
        self.assertEqual(p['protocol']['judge'],'o4-mini')
        for s in p['segments']:
            cmd=s['command']
            self.assertEqual(cmd[cmd.index('--max-steps')+1],'30')
            self.assertEqual(cmd[cmd.index('--temperature')+1],'0.7')
            self.assertEqual(cmd[cmd.index('--judge-model')+1],'o4-mini')

    def test_partial_rate_keeps_all_scheduled_denominator_and_rejects_overlap(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            p=runner.plan('CPU_CHECK'); p['output']=temp
            root=Path(temp)/p['segments'][0]['name']/'results'; root.mkdir(parents=True)
            (root/'one.json').write_text(json.dumps(dict(task_id=p['task_ids'][0],valid=True,reward=1)))
            with patch.object(runner,'write_json'):
                r=runner.report(p,False)
                self.assertFalse(r['complete']); self.assertEqual(r['summary']['success_rate_all_scheduled'],1/300)
                self.assertEqual(r['summary']['success_rate_valid'],1)
                with self.assertRaisesRegex(ValueError,'not complete'): runner.report(p)
                other=Path(temp)/p['segments'][1]['name']/'results'; other.mkdir(parents=True)
                (other/'one.json').write_text((root/'one.json').read_text())
                with self.assertRaisesRegex(ValueError,'Duplicate'): runner.report(p,False)


if __name__=='__main__': unittest.main()
