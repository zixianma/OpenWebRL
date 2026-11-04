"""Unpaid fixed cached runtime preparation and fail-closed frontdoor tests."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

API = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


f = module('_cached_existing_fixtures',API/'tests/test_browsercode_output_sweep.py')
p = module('_cached_portable_prepare',API/'reproduce/cached_selection.py')
c = module('_cached_portable_common',API/'reproduce/common.py')
b = module('_cached_portable_frontdoor',API/'reproduce/browsercode.py')


class CachedPortableTests(unittest.TestCase):
    def setUp(self):
        self.fixture = f.SweepTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def runtime(self, name):
        runtime = self.fixture.runtime(name)
        path = runtime/'worker.py'
        text = path.read_text().replace("'WEBWRIGHT_BROWSER_BACKEND':'cdp'",
            "'WEBWRIGHT_BROWSER_BACKEND':'cdp','PYTHONPATH':os.pathsep.join([str(ROOT),CONFIG['cuawright_source'],CONFIG['frozen_support_source']]),'PYTHONDONTWRITEBYTECODE':'1'")
        path.write_text(text)
        manifest = p.read(runtime/'source-manifest.json')
        manifest['source_hashes'][str(path)] = p.digest(path)
        p.write(runtime/'source-manifest.json',manifest)
        p.write(runtime/'execution-source-manifest.json',manifest)
        return runtime

    def args(self,runtime,arm):
        return c.parser('luna').parse_args(['--runtime',str(runtime),'--arm',arm,
            '--output',str(self.root/'output'),'--model-usd','1','--judge-usd','1','--limit','1045'])

    def test_both_fixed_protocols_plan_source_binding_and_no_execution(self):
        for arm, (n,decisions) in p.OPTIONS.items():
            runtime = self.runtime(arm)
            receipt = p.configure(runtime,arm)
            self.assertEqual(p.validate(runtime),receipt)
            with patch.object(c,'execute',side_effect=AssertionError('No paid execution')):
                plan,tasks = c.prepare(self.args(runtime,arm),'luna')
            self.assertEqual((plan['candidate_count'],plan['max_decisions'],plan['max_output_tokens']),(n,decisions,12288))
            self.assertEqual(plan['candidate_effort'],'high')
            self.assertEqual(plan['selector_effort'],'high')
            self.assertEqual(len(tasks),1045)
            self.assertFalse((self.root/'output').exists())
            config = p.read(runtime/'runtime-config.json')
            self.assertIn('Only'+str(decisions)+' model decisions',config['system_templates']['browsercode'])
            self.assertEqual(config['judge_protocol']['model'],'gpt-5.6-terra')
            self.assertIn('cached_model.py',(runtime/'worker.py').read_text().replace('from cached_model import','cached_model.py'))
            with self.assertRaises(ValueError):
                p.configure(runtime,arm)

    def test_each_rehashed_scientific_mutation_still_rejected(self):
        for name in ('cached_model.py','worker.py','runtime-config.json','campaign-plan.json','tasks.json'):
            runtime = self.runtime(name)
            arm = 'luna_n3_cached_high30'
            p.configure(runtime,arm)
            path = runtime/name
            path.write_text(path.read_text()+'\n')
            manifest = p.read(runtime/'source-manifest.json')
            manifest['source_hashes'][str(path)] = p.digest(path)
            p.write(runtime/'source-manifest.json',manifest)
            p.write(runtime/'execution-source-manifest.json',manifest)
            with self.subTest(name=name), self.assertRaises(ValueError):
                c.prepare(self.args(runtime,arm),'luna')

    def test_frontdoor_rejects_wrong_campaign_and_conflicting_overrides_before_dependencies(self):
        for extra in [['--campaign','gpt'],['--campaign','luna','--luna-single-output-tokens','12288']]:
            args = b.parser().parse_args(['setup','--workspace',str(self.root/'new'),
                '--luna-cached-selection','luna_n3_cached_high30',*extra])
            with patch.object(b,'call',side_effect=AssertionError('No dependency setup')), self.assertRaises(ValueError):
                b.setup(args)
            self.assertFalse((self.root/'new').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
