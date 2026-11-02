from pathlib import Path
from unittest import TestCase,main,mock
import ast,sys,types,os,json,tempfile,subprocess,contextlib,io
import importlib.util
SOURCE=Path(__file__).resolve().parents[1]/'reproduce'
spec=importlib.util.spec_from_file_location('browser_lifecycle_tested',SOURCE/'browser_lifecycle.py');L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)

class Guards(TestCase):
 def fixture(self, failure=None):
  calls=[];state={'running':False};page=types.SimpleNamespace()
  def operation(name,value=None):
   def f(*args,**kwargs):
    calls.append((name,args,kwargs))
    if failure==name:raise RuntimeError(name)
    return value
   return f
  page.set_viewport_size=operation('viewport');page.goto=operation('navigate');session=types.SimpleNamespace(send=operation('target',{'targetInfo':{'targetId':'same-target'}}));page.context=types.SimpleNamespace(new_cdp_session=operation('session',session));browser=types.SimpleNamespace(contexts=[types.SimpleNamespace(pages=[] if failure=='empty'else[page])],close=operation('close'))
  def stop():state['running']=False;calls.append(('stop',(),{}))
  manager=types.SimpleNamespace(chromium=types.SimpleNamespace(connect_over_cdp=operation('connect',browser)),stop=stop)
  def start():
   if state['running']:raise RuntimeError('poisoned loop')
   state['running']=True;calls.append(('start',(),{}));return manager
  factory=lambda:types.SimpleNamespace(start=start)
  return types.SimpleNamespace(sync_playwright=factory),calls,state,browser
 def test_positive_native_call_order_and_arguments(self):
  module,calls,state,_=self.fixture()
  with mock.patch.dict(sys.modules,{'playwright.sync_api':module}):self.assertEqual(L.live_page('private-test-ws','https://example.invalid/'),'same-target')
  self.assertFalse(state['running']);self.assertEqual([x[0]for x in calls],['start','connect','viewport','navigate','session','target','close','stop']);self.assertEqual(calls[1][2],{'timeout':30000});self.assertEqual(calls[2][1],({'width':1280,'height':1800},));self.assertEqual(calls[3][2],{'wait_until':'domcontentloaded','timeout':30000})
 def test_connect_failure_still_stops_manager(self):self.check_failure('connect')
 def test_empty_page_failure_still_stops_manager(self):self.check_failure('empty',IndexError)
 def test_viewport_failure_still_stops_manager(self):self.check_failure('viewport')
 def test_navigation_failure_still_stops_manager(self):self.check_failure('navigate')
 def test_target_failure_still_stops_manager(self):self.check_failure('target')
 def test_browser_close_failure_still_stops_manager(self):self.check_failure('close')
 def check_failure(self,name,error=RuntimeError):
  module,calls,state,_=self.fixture(name)
  with mock.patch.dict(sys.modules,{'playwright.sync_api':module}):
   with self.assertRaises(error):L.live_page('private-test-ws','https://example.invalid/')
  self.assertFalse(state['running']);self.assertEqual(sum(x[0]=='stop'for x in calls),1)
 def test_scoped_binding_restores_existing_and_missing_globals(self):
  for existing in (True,False):
   original=object();worker=types.SimpleNamespace(**({'live_page':original}if existing else{}))
   with self.assertRaisesRegex(RuntimeError,'setup failure'):
    with L.scoped(worker):
     self.assertIs(worker.live_page,L.live_page);raise RuntimeError('setup failure')
   if existing:self.assertIs(worker.live_page,original)
   else:self.assertFalse(hasattr(worker,'live_page'))
 def test_empty_pages_does_not_poison_next_startup(self):
  module,calls,state,browser=self.fixture('empty')
  with mock.patch.dict(sys.modules,{'playwright.sync_api':module}):
   with self.assertRaises(IndexError):L.live_page('private-test-ws')
   self.assertFalse(state['running']);page=types.SimpleNamespace(set_viewport_size=lambda value:None,context=types.SimpleNamespace(new_cdp_session=lambda p:types.SimpleNamespace(send=lambda _: {'targetInfo':{'targetId':'same-target'}})));browser.contexts=[types.SimpleNamespace(pages=[page])];self.assertEqual(L.live_page('private-test-ws'),'same-target')
class ExecutionGuards(TestCase):
 def common(self):
  spec=importlib.util.spec_from_file_location('lifecycle_common_tested',SOURCE/'common.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 def fixture(self,path):
  runtime=path/'runtime';runtime.mkdir();(runtime/'campaign-plan.json').write_text('{"protocol":{}}');(runtime/'runtime-config.json').write_text('{}')
  return types.SimpleNamespace(env_file=None,model_usd='1',judge_usd='1',model_calls=1,judge_calls=1,wandb_mode='disabled',arm='luna_single_high'),{'runtime':str(runtime),'output':str(path/'out'),'provider':'gpt','browser_start_cap':1}
 def test_binding_change_rejects_before_keys_output_or_sdk(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as tmp:
   args,plan=self.fixture(Path(tmp));m=self.common();plan['browser_lifecycle']={'version':'foreign','source_sha256':'foreign'}
   with mock.patch.dict(os.environ,{},clear=True):
    with self.assertRaisesRegex(ValueError,'Browser lifecycle source changed'):m.execute(args,plan,[])
   self.assertFalse(Path(plan['output']).exists())
 def test_scope_installed_before_setup_and_restored_on_setup_failure(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as tmp:
   args,plan=self.fixture(Path(tmp));m=self.common();old=object();worker=types.SimpleNamespace(live_page=old);modules={'worker':worker,'environments':types.SimpleNamespace()}
   def setup(*unused):
    self.assertIsNot(worker.live_page,old);raise RuntimeError('setup failure')
   with mock.patch.dict(os.environ,{'OPENAI_API_KEY':'offline-never-used'},clear=True),mock.patch.object(m.importlib,'import_module',side_effect=modules.__getitem__),mock.patch.object(sys,'path',list(sys.path)):
    with self.assertRaisesRegex(RuntimeError,'setup failure'):m.execute(args,plan,[],worker_setup=setup)
   self.assertIs(worker.live_page,old)
 def test_scope_restored_and_integration_closed_if_sdk_initialization_fails(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as tmp:
   args,plan=self.fixture(Path(tmp));m=self.common();old=object();events=[]
   class Pool:
    def __init__(self,root,*unused):self.root=root
   worker=types.SimpleNamespace(live_page=old,Budget=Pool);modules={'worker':worker,'environments':types.SimpleNamespace()}
   def sdk(**unused):
    self.assertIsNot(worker.live_page,old);raise RuntimeError('sdk failure')
   def setup(*unused):return types.SimpleNamespace(close_all=lambda:events.append('close'),restore=lambda:events.append('restore'))
   with mock.patch.dict(os.environ,{'OPENAI_API_KEY':'offline-never-used'},clear=True),mock.patch.object(m.importlib,'import_module',side_effect=modules.__getitem__),mock.patch.object(sys,'path',list(sys.path)),mock.patch.dict(sys.modules,{'openai':types.SimpleNamespace(OpenAI=sdk)}):
    with self.assertRaisesRegex(RuntimeError,'sdk failure'):m.execute(args,plan,[],worker_setup=setup)
   self.assertIs(worker.live_page,old);self.assertEqual(events,['close','restore'])

LOCAL_SCRIPT=r'''
from pathlib import Path
import sys,json,types,urllib.parse,asyncio
from unittest.mock import patch
runtime,source,root=map(Path,sys.argv[1:]);sys.path.insert(0,str(runtime));sys.path.insert(1,str(source))
import worker,environments
from browser_lifecycle import scoped
from playwright.sync_api import sync_playwright as real_factory
root.mkdir(parents=True);browser=worker.OwnedBrowser(root);local=None;original=worker.live_page;managers=[]
def empty_contexts_factory():
 def start():
  manager=real_factory().start();managers.append(manager)
  def connect(*a,**k):
   connection=manager.chromium.connect_over_cdp(*a,**k)
   return types.SimpleNamespace(contexts=[],close=connection.close)
  return types.SimpleNamespace(chromium=types.SimpleNamespace(connect_over_cdp=connect),stop=manager.stop)
 return types.SimpleNamespace(start=start)
try:
 ws=browser.start()
 with scoped(worker):
  with patch('playwright.sync_api.sync_playwright',side_effect=empty_contexts_factory):
   try:worker.live_page(ws)
   except IndexError:pass
   else:raise AssertionError('Initial empty-pages failure must remain invalid')
  try:asyncio.get_running_loop()
  except RuntimeError:pass
  else:raise AssertionError('Temporary Playwright loop survived failed setup')
  url='data:text/html,'+urllib.parse.quote('<!doctype html><h1>lifecycle-ready</h1>')
  target=worker.live_page(ws,navigate=url);assert isinstance(target,str)and target
  local=worker.environment(root/'workspace',ws,'browsercode')
  worker.setup_native(local,'browsercode',ws,dict(task_id='lifecycle-ready',intent='Local mechanical lifecycle check',start_url=url))
  image=root/'raw-native.png';local.capture(image);worker.verify_png(image)
 assert worker.live_page is original
finally:
 if local is not None:local.close()
 cleanup=browser.close();assert cleanup['verified']is True
 environments.cleanup_registered_groups(root)
(root/'local-proof.json').write_text(json.dumps(dict(empty_page_failure_preserved=True,next_native_startup_qualified=True,raw_png_1280x1800=True,worker_binding_restored=True,owned_browser_cleanup_verified=True,provider_calls=0,model_calls=0))+'\n')
'''

class ActualLocalRegression(TestCase):
 def test_real_playwright_empty_page_cleanup_then_native_browsercode_startup(self):
  runtime=os.environ.get('OPENWEBRL_BROWSER_TEST_RUNTIME')
  if not runtime:self.skipTest('Set OPENWEBRL_BROWSER_TEST_RUNTIME to a generated BrowserCode runtime')
  with tempfile.TemporaryDirectory(dir='/tmp',prefix='browser-lifecycle-local-')as temporary:
   root=Path(temporary)/'owned-local';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',BASH_ENV='/dev/null',ENV='/dev/null',TMPDIR='/tmp',OPENWEBRL_REPRO_OUTPUT=str(root))
   for key in('OPENAI_API_KEY','BROWSER_USE_API_KEY','ANTHROPIC_API_KEY','JUDGE_API_KEY'):env.pop(key,None)
   for key in('XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','XDG_STATE_HOME','WANDB_DIR','WANDB_CONFIG_DIR','WANDB_CACHE_DIR','WANDB_DATA_DIR','HF_HOME','TORCH_HOME','MPLCONFIGDIR','PYTHONPYCACHEPREFIX'):env[key]=str(Path(temporary)/key.lower())
   result=subprocess.run([sys.executable,'-B','-c',LOCAL_SCRIPT,str(Path(runtime).resolve()),str(SOURCE),str(root)],capture_output=True,text=True,timeout=120,env=env,cwd=temporary)
   self.assertEqual(result.returncode,0,result.stderr[-4000:]);proof=json.loads((root/'local-proof.json').read_text());self.assertTrue(proof['next_native_startup_qualified']);self.assertTrue(proof['owned_browser_cleanup_verified']);self.assertEqual(proof['provider_calls'],0)
if __name__=='__main__':main()
