"""Real startup diagnostic confinement with a zero-browser creation callback."""
import json,os,shutil,subprocess,sys,unittest,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_diagnostic_and_lazy_cli_proxy_imports_no_browser(self):
  r=ROOT.parent/'integration-tests'/('diagnostic-'+uuid.uuid4().hex);r.mkdir(parents=True)
  for p in ROOT.glob('*.py'):shutil.copy2(p,r/p.name)
  shutil.copytree(ROOT/'confinement_support',r/'confinement_support');(r/'upstream').symlink_to(ROOT/'upstream',target_is_directory=True)
  output=r/'shard0'/'startup-check';output.mkdir(parents=True)
  script="""import json,subprocess,sys,runpy
from pathlib import Path
import diagnostic_browser
output=Path(sys.argv[1])
def mock_create():
 subprocess.run([sys.executable,'-m','cuawright.webwright.tools.browser_session','--help'],check=True,capture_output=True)
 cfg=json.loads((output/'guard-config.json').read_text())
 runpy.run_path(cfg['browser_launcher'],run_name='offline_proxy_import_only')
 (output/'callback-ok').write_text('yes')
record=diagnostic_browser.run(output,'http://127.0.0.1:1/browser/probe/reserve',1,mock_create)
assert record['actor_release_permitted'] and all(record['kernel_guard'].values()),record
assert record['offline_mock'] and record['model_dispatches']==0,record
print(json.dumps(record))
"""
  env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(r))
  proc=subprocess.run([sys.executable,'-B','-c',script,str(output)],env=env,cwd=r,capture_output=True,text=True,timeout=60)
  (r/'diagnostic.log').write_text(proc.stdout+'\n'+proc.stderr);self.assertEqual(proc.returncode,0,proc.stdout+'\n'+proc.stderr)
  record=json.loads((output/'startup-diagnostic.json').read_text());self.assertTrue(record['owned_teardown']['process_verification']['verified']);self.assertEqual((output/'callback-ok').read_text(),'yes')
  broker=next((output.parent/'.broker-evidence').glob('*/browser-broker-events.jsonl'));self.assertNotIn('"event": "reserved"',broker.read_text())
if __name__=='__main__':unittest.main()
