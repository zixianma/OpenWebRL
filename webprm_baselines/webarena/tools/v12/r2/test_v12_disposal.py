"""CPU regression for smoke job 395908 (all 6 cells audited, none disposed): the launcher's audit_input must produce every
REQUIRED_ARTIFACTS key of webprm_ephemeral_state (it lacked 'terminal'), and durable_artifacts must accept them. Uses a
read-only copy of a real 395908 cell; adds the r2 terminal capture with the real worker function. OpenWebRL venv Python.
  V12_PACKAGE=<pkg> V12_EVIDENCE_CELL=<attempt-395908 cell dir> venv/bin/python test_v12_disposal.py"""
import importlib.util, json, os, shutil, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
PKG = Path(os.environ['V12_PACKAGE']); CELL = Path(os.environ['V12_EVIDENCE_CELL'])
spec = importlib.util.spec_from_file_location('v12cell', PKG / 'source/scripts/run_owrl_webarena_cell.py')
cell = importlib.util.module_from_spec(spec); sys.modules['v12cell'] = cell; spec.loader.exec_module(cell)
from scripts import launch_webprm_table5_campaign as L  # noqa: E402  (package source is on sys.path)
from openwebrl import webprm_ephemeral_state as E  # noqa: E402

class Disposal(unittest.TestCase):
    def test_required_artifacts(self):
        campaign = Path(tempfile.mkdtemp(prefix='v12d-', dir='/tmp'))
        output = campaign / 'attempt'; output.mkdir()          # production: the attempt dir lives inside the campaign
        for name in ('base-evidence.json', 'broker.jsonl', 'financial-contract.json'):
            shutil.copy2(CELL.parents[1] / name, output / name)
        folder = output / 'cells' / CELL.name
        shutil.copytree(CELL, folder, symlinks=True, ignore=shutil.ignore_patterns('storage-state-private.json'))
        episode = next(folder.glob('episodes/webarena_lite_*/*'))
        for p in (episode / 'native-evaluation').glob('*'): p.unlink()
        if (episode / 'native-evaluation').exists(): (episode / 'native-evaluation').rmdir()
        plan = json.loads((folder / 'worker-plan.json').read_text())
        cellrow = dict(task_id=plan['task_ids'][0], actor=plan['actor_model'], selector=plan['selector'], cell_id='x')
        status = dict(budget=json.loads((folder / 'broker-status.json').read_text())['budget'])
        proposal = dict(campaign_root=str(campaign), cohort_path=str(PKG / 'cohort.json'), source_sha256={})
        before = L.audit_input(proposal, output, folder, cellrow, plan, status)
        missing = E.REQUIRED_ARTIFACTS - before['artifacts'].keys() - {'worker_plan', 'audit'}
        self.assertEqual(missing, {'terminal'}, missing)          # reproduces 395908
        steps = json.loads((episode / 'owrl-steps.json').read_text())
        ref = cell.write_terminal(episode, SimpleNamespace(steps=steps['steps'], start_url=steps['start_url']))
        after = L.audit_input(proposal, output, folder, cellrow, plan, status)
        files = {k: v['path'] for k, v in after['artifacts'].items()}
        files.update(worker_plan=str(folder / 'worker-plan.json'), audit=str(folder / 'audit.json'))
        self.assertTrue(E.REQUIRED_ARTIFACTS <= files.keys(), E.REQUIRED_ARTIFACTS - files.keys())
        manifest = E.durable_artifacts(campaign, files)
        terminal = json.loads((episode / 'native-evaluation/terminal.json').read_text())
        from PIL import Image
        self.assertEqual(Image.open(terminal['image']['path']).size, (1280, 1000))
        print(f'disposal regression: r0 lacked {sorted(missing)}; with r2 terminal capture all {len(E.REQUIRED_ARTIFACTS)} required artifacts present, '
              f'durable_artifacts hashed {len(manifest)} files; terminal index {terminal["index"]} url {terminal["url"][:60]}')

if __name__ == '__main__':
    unittest.main(verbosity=2)
