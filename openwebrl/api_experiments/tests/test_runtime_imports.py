"""Module entry points must import the generated worker and its frozen support."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("runtime_import_tested", ROOT / "reproduce/runtime_imports.py")
I = importlib.util.module_from_spec(spec)
spec.loader.exec_module(I)


class ImportGuards(unittest.TestCase):
    def test_cached_foreign_worker_or_generated_helper_is_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            runtime = Path(tmp) / "requested-runtime"
            runtime.mkdir()
            for name in ("worker", "environments", "bootstrap", "portable_browser_lifecycle", "portable_browser_startup", "request_budget", "claude_budget"):
                with self.subTest(name=name):
                    (runtime / (name + ".py")).write_text("# Frozen generated helper\n")
                    cached = ModuleType(name)
                    cached.__file__ = str(Path(tmp) / "different-runtime" / (name + ".py"))
                    with mock.patch.dict(sys.modules, {name: cached}), mock.patch.object(I.importlib, "import_module") as imported:
                        with self.assertRaisesRegex(ValueError, "Different runtime module already loaded"):
                            I.load(runtime)
                        imported.assert_not_called()

    def test_already_loaded_public_package_uses_runtime_frozen_support(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            support = root / "support/openwebrl"
            public = root / "public/openwebrl"
            runtime = root / "runtime"
            for folder in (support / "eval", public / "eval", runtime):
                folder.mkdir(parents=True)
            for package in (support, public, public / "eval"):
                (package / "__init__.py").write_text("")
            (support / "policy.py").write_text("VALUE='frozen'\n")
            (public / "policy.py").write_text("VALUE='public'\n")
            (support / "eval/reward.py").write_text("VALUE='frozen-judge'\n")
            (public / "eval/reward.py").write_text("VALUE='public-judge'\n")
            (runtime / "runtime-config.json").write_text(json.dumps(dict(frozen_support_source=str(support.parent))))
            (runtime / "worker.py").write_text("from openwebrl.policy import VALUE\nfrom openwebrl.eval.reward import VALUE as JUDGE\n")
            (runtime / "environments.py").write_text("")
            script = """import importlib.util,sys\nfrom pathlib import Path\nroot,source=map(Path,sys.argv[1:]);sys.path.insert(0,str(root/'public'))\nimport openwebrl\nassert list(openwebrl.__path__)==[str(root/'public/openwebrl')]\nspec=importlib.util.spec_from_file_location('tested_imports',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\nworker,_=module.load(root/'runtime')\nassert worker.VALUE=='frozen'and worker.JUDGE=='frozen-judge'\nassert str(root/'public/openwebrl')in openwebrl.__path__\n"""
            result = subprocess.run([sys.executable, "-B", "-c", script, str(root), str(ROOT / "reproduce/runtime_imports.py")], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_support_fails_before_worker_import(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            (root / "runtime-config.json").write_text(json.dumps(dict(frozen_support_source=str(root / "missing"))))
            with mock.patch.object(I.importlib, "import_module") as imported:
                with self.assertRaisesRegex(ValueError, "support package is missing"):
                    I.load(root)
                imported.assert_not_called()

    def test_cached_foreign_support_is_rejected_before_worker_import(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            package = root / "support/openwebrl"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            (package / "policy.py").write_text("VALUE='frozen'\n")
            foreign = root / "foreign.py"
            foreign.write_text("VALUE='different'\n")
            cached = ModuleType("openwebrl.policy")
            cached.__file__ = str(foreign)
            (root / "runtime-config.json").write_text(json.dumps(dict(frozen_support_source=str(package.parent))))
            with mock.patch.dict(sys.modules, {"openwebrl.policy": cached}), mock.patch.object(I.importlib, "import_module") as imported:
                with self.assertRaisesRegex(ValueError, "use a fresh process"):
                    I.load(root)
                imported.assert_not_called()

    def test_identical_cached_code_cannot_read_a_foreign_relative_prompt(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            package = root / "support/openwebrl"
            foreign = root / "foreign/openwebrl"
            for directory, prompt in ((package, "frozen prompt"), (foreign, "different prompt")):
                (directory / "env/prompts").mkdir(parents=True)
                (directory / "env/prompts/system_prompt_api_browser.md").write_text(prompt)
                (directory / "__init__.py").write_text("")
                (directory / "responses_actor_history.py").write_text("from pathlib import Path\nAPI_SYSTEM_PROMPT=Path(__file__).parent/'env/prompts/system_prompt_api_browser.md'\n")
            source = foreign / "responses_actor_history.py"
            spec = importlib.util.spec_from_file_location("openwebrl.responses_actor_history", source)
            cached = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cached)
            self.assertEqual(cached.API_SYSTEM_PROMPT.read_text(), "different prompt")
            self.assertEqual(source.read_bytes(), (package / source.name).read_bytes())
            (root / "runtime-config.json").write_text(json.dumps(dict(frozen_support_source=str(package.parent))))
            with mock.patch.dict(sys.modules, {"openwebrl.responses_actor_history": cached}), mock.patch.object(I.importlib, "import_module") as imported:
                with self.assertRaisesRegex(ValueError, "use a fresh process"):
                    I.load(root)
                imported.assert_not_called()


ENTRY_SCRIPT = r'''
import json,os,runpy,socket,subprocess,sys
from pathlib import Path
from unittest.mock import patch
from openai import OpenAI
from openwebrl.api_experiments.reproduce import common
runtime,output,provider,arm=map(str,sys.argv[1:])
runtime=Path(runtime);config=json.loads((runtime/'runtime-config.json').read_text())
seen=[]
class BeforeSDK(Exception):pass
def inspect(args,plan,tasks,root,keys,actor_key,cloud,worker,environments,worker_setup):
 assert Path(worker.__file__).resolve()==runtime/'worker.py'
 assert Path(environments.__file__).resolve()==runtime/'environments.py'
 assert environments.live_page.__module__=='portable_browser_lifecycle'
 assert worker.live_page.__module__=='_openwebrl_browser_lifecycle'
 support=Path(config['frozen_support_source'])/'openwebrl'
 for name,module in tuple(sys.modules.items()):
  candidate=support.joinpath(*name.split('.')[1:]).with_suffix('.py')
  if name.startswith('openwebrl.')and candidate.is_file():
   assert Path(module.__file__).resolve()==candidate.resolve(),name
 assert not(root/'budget').exists()
 seen.append(True);raise BeforeSDK()
def forbidden(*a,**k):raise AssertionError('No SDK, browser, subprocess or network is allowed')
common._execute_worker=inspect
sys.argv=['offline-module-entry','--runtime',str(runtime),'--arm',arm,'--output',output,'--limit','1','--model-usd','1','--judge-usd','1','--execute']
with patch.object(OpenAI,'__init__',forbidden),patch.object(socket.socket,'connect',forbidden),patch.object(socket.socket,'connect_ex',forbidden),patch.object(subprocess,'Popen',forbidden),patch.object(subprocess,'run',forbidden):
 try:runpy.run_module('openwebrl.api_experiments.reproduce.eval_'+provider,run_name='__main__')
 except BeforeSDK:pass
 else:raise AssertionError('Real worker import was not reached')
assert seen==[True]
print('actual-module-entry-worker-import-before-SDK: PASS')
'''


@unittest.skipUnless(os.environ.get("OPENWEBRL_API_DEPENDENCIES"), "Set OPENWEBRL_API_DEPENDENCIES for real frozen worker imports")
class ActualModuleEntry(unittest.TestCase):
    def test_all_provider_module_entries_import_real_workers_without_clients(self):
        dependencies = json.loads(Path(os.environ["OPENWEBRL_API_DEPENDENCIES"]).read_text())
        python = dependencies.get("python", sys.executable)
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="module-worker-import-") as temporary:
            root = Path(temporary)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR="/tmp", OPENAI_API_KEY="offline-unused", ANTHROPIC_API_KEY="offline-unused")
            for name in ("XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "HF_HOME", "WANDB_DIR", "WANDB_CONFIG_DIR", "WANDB_CACHE_DIR", "MPLCONFIGDIR", "PYTHONPYCACHEPREFIX"):
                env[name] = str(root / name.lower())
            for campaign, provider, arm in (("browsercode_gpt_high", "gpt", "parallel5_select_high"), ("browsercode_luna_effort", "luna", "luna_single_xhigh"), ("browsercode_claude", "claude", "sonnet_n5")):
                with self.subTest(provider=provider):
                    runtime = root / campaign
                    built = subprocess.run([python, "-B", str(ROOT / "reproduce/prepare_browsercode.py"), "--campaign", campaign, "--output", str(runtime), "--dependencies", os.environ["OPENWEBRL_API_DEPENDENCIES"]], env=env, text=True, capture_output=True, timeout=120)
                    self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                    result = subprocess.run([python, "-B", "-c", ENTRY_SCRIPT, str(runtime), str(root / (provider + "-output")), provider, arm], cwd=ROOT.parents[1], env=env, text=True, capture_output=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("actual-module-entry-worker-import-before-SDK: PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
