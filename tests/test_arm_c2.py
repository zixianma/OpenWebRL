"""C2 provenance, terminal filtering, masking, and post-evaluation handoff tests."""
import base64
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openwebrl.arm_c2 import TurnExporter, build_dataset, eligible_turn, load_training_example, sha, write_json


def sample_export(root, task_id='task', reward=1, valid=True, executed=True, finish='stop', fallback=None):
    from PIL import Image
    buffer=io.BytesIO(); Image.new('RGB',(32,32),'gray').save(buffer,format='PNG')
    exporter=TurnExporter(root); exporter.begin(task_id)
    prompt='history containing an earlier assistant response; current prompt'
    record=dict(mode='selection',task_id=task_id,turn=0,selected_index=1,scores=None,fallback=fallback,
                prompt_sha256=sha(prompt.encode()),screenshot_sha256=sha(buffer.getvalue()))
    outputs=[('unchosen',[90,91],[-.1,-.2],'stop'),('chosen',[20,21],[-.3,-.4],finish)]
    exporter(record=record,prompt=prompt,images=['data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()],outputs=outputs)
    meta=dict(turn_index=0,image_grid_thw=[[1,2,2]])
    if executed: meta['step_tool_responses']=[]
    sample=SimpleNamespace(metadata=meta,tokens=[10,11,12,20,21],response_length=2,response='chosen',prompt=prompt,status='COMPLETED')
    outcome=exporter.finish(task_id,[sample],dict(task_id=task_id,reward=reward,valid=valid))
    return exporter,outcome,sample


def config(root,ids):
    tasks=root/'tasks.jsonl'
    tasks.write_text(''.join(json.dumps(dict(task_id=i))+'\n' for i in ids))
    write_json(root/'frozen-config.json',dict(task_file=str(tasks),training=dict(max_tokens=32768)))


class C2Tests(unittest.TestCase):
    def test_only_successful_trajectories_and_executed_winners_are_retained(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); config(root,['success','failure','unavailable','unexecuted','truncated','fallback'])
            sample_export(root,'success')
            sample_export(root,'failure',reward=0)
            sample_export(root,'unavailable',valid=False)
            sample_export(root,'unexecuted',executed=False)
            sample_export(root,'truncated',finish='length')
            sample_export(root,'fallback',fallback='malformed selector')
            audit=build_dataset(root)
            self.assertEqual(audit['retained_turns'],1)
            row=json.loads(Path(audit['dataset']).read_text())
            captured=json.loads(Path(row['source']).read_text())
            self.assertEqual(row['task_id'],'success')
            self.assertEqual(captured['raw_candidates'][row['selected_index']]['response_token_ids'],[20,21])
            self.assertEqual(audit['exclusions']['trajectory_unsuccessful_or_unavailable'],2)

    def test_partial_collection_cannot_become_full_training_data(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); config(root,['a','b']); sample_export(root,'a')
            with self.assertRaisesRegex(ValueError,'incomplete'): build_dataset(root)
            preview=build_dataset(root,False)
            self.assertFalse(preview['complete_collection'])
            self.assertFalse((root/'training.jsonl').exists())

    def test_turn_export_is_immutable_and_detects_executed_token_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); exporter,outcome,sample=sample_export(root)
            sample.tokens[-1]=999
            with self.assertRaisesRegex(ValueError,'tokens differ'): exporter.finish('task',[sample],outcome)
            saved=json.loads((Path(outcome['attempt'])/'outcome.json').read_text())
            self.assertEqual(saved['reward'],1)
            second=exporter.begin('task')
            self.assertNotEqual(str(second),outcome['attempt'])
            self.assertTrue(Path(outcome['attempt']).exists())

    def test_changed_images_and_cross_task_joins_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); config(root,['task']); _,outcome,_=sample_export(root)
            execution_path=Path(outcome['attempt'])/'execution.json'
            execution=json.loads(execution_path.read_text())
            execution['task_id']='other'; write_json(execution_path,execution)
            with self.assertRaisesRegex(ValueError,'Cross-task'): build_dataset(root)
            execution['task_id']='task'; write_json(execution_path,execution)
            captured=json.loads(Path(execution['turns'][0]['source']).read_text())
            Path(captured['images'][0]['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Image hash'): build_dataset(root)

    def test_oversize_and_nonfinite_scalar_scores_are_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); _,outcome,_=sample_export(root)
            execution=json.loads((Path(outcome['attempt'])/'execution.json').read_text())['turns'][0]
            captured=json.loads(Path(execution['source']).read_text())
            self.assertEqual(eligible_turn(captured,execution,max_tokens=4),'oversize')
            captured.update(mode='scalar',scores=[0,float('nan')])
            self.assertEqual(eligible_turn(captured,execution),'invalid_scores')

    def test_processor_prefix_and_grid_must_match_and_history_has_zero_loss(self):
        import torch
        class Processor:
            def __call__(self, **kw):
                self.kw=kw
                return dict(input_ids=[[10,11,12]],image_grid_thw=torch.tensor([[1,2,2]]),pixel_values=torch.zeros(4,3))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); config(root,['task']); sample_export(root)
            audit=build_dataset(root); row=json.loads(Path(audit['dataset']).read_text())
            processor=Processor(); example=load_training_example(row,processor)
            self.assertEqual(example['input_ids'].tolist(),[[10,11,12,20,21]])
            self.assertEqual(example['labels'].tolist(),[[-100,-100,-100,20,21]])
            self.assertIn('earlier assistant',processor.kw['text'][0])
            with self.assertRaisesRegex(ValueError,'prompt token mismatch'):
                load_training_example(dict(row,prompt_token_ids=[10,99,12]),processor)
            with self.assertRaisesRegex(ValueError,'grid mismatch'):
                load_training_example(dict(row,image_grid_thw=[[1,4,4]]),processor)

    def test_arm_training_checkout_rejects_drift_and_path_escape(self):
        from scripts.run_arm_c2 import training_command
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            entry = root / 'train.py'; entry.write_text('pass\n')
            checkout = dict(root=str(root), entrypoint='train.py', commit='revision', sha256=sha(entry.read_bytes()))
            queue = dict(training_python='/python', training_checkout=checkout)
            with patch('subprocess.check_output', return_value='revision\n'):
                argv, cwd = training_command(queue)
                self.assertEqual(argv, ['/python', str(entry)])
                self.assertEqual(cwd, root)
                entry.write_text('changed\n')
                with self.assertRaisesRegex(ValueError, 'changed after preflight'):
                    training_command(queue)
                checkout['entrypoint'] = '../outside.py'
                with self.assertRaisesRegex(ValueError, 'inside the pinned'):
                    training_command(queue)

    def test_resume_archives_resource_changes_but_rejects_recipe_changes(self):
        from scripts.run_arm_c2 import freeze_execution_config
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            original = dict(source_full='source', output=str(root), task_file_sha256='hash',
                            task_count=2091, seed=42, teacher='selection',
                            teacher_manifest={'judge': 'o4-mini'}, training={'epochs': 2}, allocation='old')
            freeze_execution_config(root, original)
            resumed = dict(original, allocation='new')
            freeze_execution_config(root, resumed)
            self.assertEqual(len(list((root / 'execution-sessions').glob('*.json'))), 2)
            for changed in (dict(resumed, training={'epochs': 1}),
                            dict(resumed, teacher_manifest={'judge': 'gpt-4.1'})):
                with self.assertRaisesRegex(ValueError, 'resume protocol changed'):
                    freeze_execution_config(root, changed)
            self.assertEqual(json.loads((root / 'frozen-config.json').read_text()), resumed)

    def test_teacher_choice_requires_completed_original_eval(self):
        from scripts.run_arm_c2 import choose_teacher
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); ids=[str(i) for i in range(300)]
            for mode,wins in [('baseline',90),('scalar',114),('selection',130)]:
                directory=root/mode
                manifest=dict(task_ids=ids,actor='actor',task_file_sha256='hash',seed=42,sampling={},max_steps=30,judge='judge')
                write_json(directory/'manifest.json',manifest)
                write_json(directory/'summary.json',dict(attempted=300))
                for i in range(300): write_json(directory/'results'/f'{i}.json',dict(task_id=str(i),reward=int(i<wins),valid=True))
            teacher,_,_=choose_teacher(root); self.assertEqual(teacher,'selection')
            write_json(root/'selection/summary.json',dict(attempted=299))
            with self.assertRaisesRegex(ValueError,'not completed'): choose_teacher(root)

    def test_c2_handoff_follows_report_and_held_retry_does_not_block_it(self):
        import subprocess
        from test_arm_retry import fixture
        spec=importlib.util.spec_from_file_location('report',Path(__file__).parents[1]/'scripts/summarize_arm_reproduction.py')
        report=importlib.util.module_from_spec(spec); spec.loader.exec_module(report)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); q=fixture(root); source=Path(q['source_full'])
            write_json(source/'queued-retry.json',dict(q,authorized=False))
            write_json(source/'queued-c2.json',dict(authorized=True,source_full=str(source),output=str(root/'c2')))
            def child(argv,**kw):
                self.assertTrue((source/'comparison.json').exists())
                self.assertTrue(argv[1].endswith('run_arm_c2.py'))
                return SimpleNamespace(returncode=0)
            with patch('sys.argv',['report',str(source)]),patch.object(subprocess,'run',side_effect=child) as run,patch('builtins.print'):
                report.main()
            run.assert_called_once()
            self.assertFalse(Path(q['output']).exists())

    def test_simultaneously_authorized_retry_and_c2_are_rejected(self):
        import subprocess
        from test_arm_retry import fixture
        spec=importlib.util.spec_from_file_location('report',Path(__file__).parents[1]/'scripts/summarize_arm_reproduction.py')
        report=importlib.util.module_from_spec(spec); spec.loader.exec_module(report)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); q=fixture(root); source=Path(q['source_full'])
            write_json(source/'queued-retry.json',q)
            write_json(source/'queued-c2.json',dict(authorized=True,source_full=str(source),output=str(root/'c2')))
            with patch('sys.argv',['report',str(source)]),patch.object(subprocess,'run') as run,patch('builtins.print'):
                with self.assertRaisesRegex(ValueError,'cannot both'): report.main()
            run.assert_not_called()


if __name__=='__main__': unittest.main()
