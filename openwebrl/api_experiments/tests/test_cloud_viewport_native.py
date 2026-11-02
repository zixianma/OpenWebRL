from pathlib import Path
import unittest,subprocess,sys,os,tempfile,json
SCRIPT=r"""
from pathlib import Path
import sys,json,types,urllib.parse
runtime,source,root=map(Path,sys.argv[1:]);sys.path.insert(0,str(runtime));sys.path.insert(1,str(source))
import worker,environments
from reproduce import cloud_browsercode as integration
LocalBrowser=worker.OwnedBrowser
root.mkdir(parents=True,exist_ok=True)
browser=LocalBrowser(root);registry=None;receipts=[]
def no_provider(*a,**k):raise AssertionError('Provider calls forbidden in real local regression')
try:
 ws=browser.start()
 registry=integration.install(worker,environments,http_factory=no_provider,admission=no_provider,metadata_factory=no_provider,on_halt=no_provider,max_proxy_usd='10')
 pages=[('exact-quirks-diagnostic','<html><body><h1>native-startup-ready</h1></body></html>','BackCompat')]
 for suffix,style in [('vertical','height:4000px;width:100px'),('horizontal','width:4000px;height:100px'),('both','width:4000px;height:4000px')]:
  pages.append(('standards-'+suffix,'<!doctype html><html><body style="margin:0"><div style="'+style+'">scrollbar regression</div></body></html>','CSS1Compat'))
 for name,html,mode in pages:
  directory=root/name;local=worker.environment(directory/'workspace',ws,'browsercode')
  try:
   worker.setup_native(local,'browsercode',ws,dict(task_id=name,intent='Local zero-provider geometry regression',start_url='data:text/html,'+urllib.parse.quote(html)))
   image=directory/'raw-native.png';local.capture(image);worker.verify_png(image)
   proof=json.loads(sorted(directory.glob('cloud-viewport-proof-*.json'))[-1].read_text());value=proof['measured'];assert value['dom']['compatMode']==mode
   assert proof['selected_client_source']==('bodyClient'if mode=='BackCompat'else'rootClient')
   assert value['dom']['width']==1280 and value['dom']['height']==1800
   receipts.append(dict(case=name,compat_mode=mode,selected_source=proof['selected_client_source'],measured=value,raw_png_qualified=True))
  finally:local.close()
 assert len(receipts)==4
except BaseException:
 for log in root.rglob('browsercode-stderr.log'):print(log.read_text()[-3000:],file=sys.stderr)
 raise
finally:
 if registry is not None:registry.restore()
 closed=browser.close();assert closed['returncode']is not None
(root/'real-local-proof.json').write_text(json.dumps(dict(cases=receipts,local_browser_closed=True,provider_calls=0,model_calls=0),indent=2)+'\n')
print(json.dumps(dict(cases=len(receipts),local_browser_closed=True,provider_calls=0,model_calls=0)))
"""
class Tests(unittest.TestCase):
 def test_actual_native_bridge_quirks_and_standards_scrollbars_raw_png(self):
  runtime_env=os.environ.get('OPENWEBRL_BROWSER_TEST_RUNTIME')
  if not runtime_env:self.skipTest('Set OPENWEBRL_BROWSER_TEST_RUNTIME to a generated BrowserCode runtime')
  runtime=Path(runtime_env).resolve();source=Path(__file__).resolve().parents[1]
  with tempfile.TemporaryDirectory(dir='/tmp',prefix='stealth-real-viewport-')as temporary:
   root=Path(temporary)/'owned-local'
   env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',BASH_ENV='/dev/null',ENV='/dev/null',TMPDIR='/tmp',OPENWEBRL_REPRO_OUTPUT=str(root))
   for key in ('OPENAI_API_KEY','BROWSER_USE_API_KEY','ANTHROPIC_API_KEY'):env.pop(key,None)
   for key in ('XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','XDG_STATE_HOME','WANDB_DIR','WANDB_CONFIG_DIR','WANDB_CACHE_DIR','WANDB_DATA_DIR','HF_HOME','TORCH_HOME','MPLCONFIGDIR','PYTHONPYCACHEPREFIX'):env[key]=str(Path(temporary)/key.lower())
   result=subprocess.run([sys.executable,'-B','-c',SCRIPT,str(runtime),str(source),str(root)],capture_output=True,text=True,timeout=120,env=env,cwd=temporary)
   self.assertEqual(result.returncode,0,result.stderr[-4000:]);proof=json.loads((root/'real-local-proof.json').read_text());self.assertEqual(len(proof['cases']),4);self.assertTrue(proof['local_browser_closed']);self.assertEqual(proof['provider_calls'],0)
if __name__=='__main__':unittest.main()
