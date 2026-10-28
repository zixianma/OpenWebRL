"""v11c regression for job 392816: the independent auditor's cohort-membership check on a REAL v11b audit input
(reddit-a, first cell). native-audit-v8 rejects it with 'Fixed v9 cell membership differs' (it required 3 methods x site
tasks); native-audit-v9 passes that identity stage. Also: a synthetic cohort missing one cell is still rejected by v9.
Usage: V11_PACKAGE=<v11c pkg> python test_v11c_audit_membership.py"""
import hashlib, importlib.util, json, os, sys, unittest
from copy import deepcopy
from pathlib import Path
D = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021')
PKG = Path(os.environ['V11_PACKAGE']); REAL = D/'launch-preparation-v11b-reddit-a'
os.environ['WEBPRM_NATIVE_SOURCE_ROOT'] = str(REAL/'source'); sys.path.insert(0, str(REAL/'source'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


v8 = load('membership_v8', D/'native-audit-v8/native_cell_audit.py')
v9 = load('membership_v9', D/'native-audit-v9/native_cell_audit.py')


def inputs_for(aud):
    inp = json.loads(sorted(REAL.glob('campaign/attempt-392816/cells/*/audit-input.json'))[0].read_text())
    inp['source_sha256'] = dict(inp['source_sha256'], **{str(D/aud/n): sha(D/aud/n) for n in ('native_cell_audit.py', 'native_lowlevel_replay.py')})
    plan = json.loads(Path(inp['worker_plan']['path']).read_text())
    cohort = json.loads(Path(inp['cohort']['path']).read_text()); cohort = cohort.get('cohort', cohort)
    return inp, plan, cohort


class Membership(unittest.TestCase):
    def test_v8_rejects_real_v11_cell_v9_accepts(self):
        inp, plan, cohort = inputs_for('native-audit-v8')
        self.assertEqual(len(cohort['cells']), 36); self.assertEqual(cohort['selectors'], ['no_search', 'piotr_selectionarm'])
        with self.assertRaisesRegex(ValueError, 'Fixed v9 cell membership differs'):
            v8.audit_identity(inp, plan, cohort)
        inp, plan, cohort = inputs_for('native-audit-v9')
        task = v9.audit_identity(inp, plan, cohort)
        self.assertEqual(task.execution_id, f"webarena_lite_{inp['cell']['task_id']:03d}")

    def test_v9_still_rejects_incomplete_cohort(self):
        inp, plan, cohort = inputs_for('native-audit-v9')
        short = deepcopy(cohort); short['cells'] = [c for c in short['cells'] if c['cell_id'] != inp['cell']['cell_id']][:35]
        with self.assertRaises(ValueError):
            v9.audit_identity(inp, plan, short)

    def test_builder_binds_v9(self):  # the stage-2 proposal's native_auditor comes from this constant
        self.assertIn("AUD = D/'native-audit-v9'", (Path(__file__).resolve().parent/'build_v11c.py').read_text())
        prop = json.loads((PKG/'proposal.json').read_text())
        if prop.get('status') == 'prepared_v11c_pending_approval':
            self.assertEqual(prop['native_auditor']['path'], str(D/'native-audit-v9/native_cell_audit.py'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
