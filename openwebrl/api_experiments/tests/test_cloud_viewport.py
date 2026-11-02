from pathlib import Path
import sys,types,tempfile,unittest,json,subprocess,os,struct
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reproduce import cloud_browsercode as integration,cloud_browser as cloud
import shutil
# Controller-only fixture: no task content, provider IDs or capability URLs.
FRACTIONAL_REPLY={'id': 2, 'ok': True, 'result': {'output': 'owned-viewport-proof:{"dom":{"width":1280,"height":1800,"compatMode":"CSS1Compat","rootClient":{"width":1265,"height":1800},"bodyClient":{"width":1265,"height":1800},"dpr":1.0000000149011612,"screenWidth":1280,"screenHeight":1800},"visual":{"offsetX":0,"offsetY":0,"pageX":0,"pageY":0,"clientWidth":1264.800048828125,"clientHeight":1800,"scale":1,"zoom":1},"layout":{"pageX":0,"pageY":0,"clientWidth":1264,"clientHeight":1800}}\n\nReturn value: null', 'image_attachments': []}}
def resolve_test_bun():
 runtime=os.environ.get('OPENWEBRL_BROWSER_TEST_RUNTIME')
 binary=Path(runtime)/'javascript-runtime/bun'if runtime else None
 binary=binary if binary and binary.is_file()else shutil.which('bun')
 if not binary:raise unittest.SkipTest('Install Bun or set OPENWEBRL_BROWSER_TEST_RUNTIME to a generated runtime')
 return str(binary)
class Tests(unittest.TestCase):
 def test_actual_create_payload_opt_in_and_fixed_screen(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as d:
   requests=[];grants=[];identity='11111111-1111-4111-8111-111111111111'
   class HTTP:
    active=True
    def request(self,method,path,payload=None):
     requests.append((method,path,payload))
     if method=='PATCH':self.active=False
     return 200,json.dumps(dict(id=identity,metadata={'arm':'offline'},status='active'if self.active else'stopped',cdpUrl='wss://private.test/socket',startedAt='2026-11-01T00:00:00Z',finishedAt=None if self.active else'2026-11-01T00:00:01Z',browserCost='0',proxyCost='0',proxyUsedMb='0'))
   def admit(payload):grants.append(payload);return dict(approved=True,proxy_overshoot_authorized=True)
   owner=cloud.CloudBrowser(Path(d),HTTP(),admit,metadata={'arm':'offline'},on_halt=lambda reason:None,max_proxy_usd='10',minimum_settlement_seconds=0,sleep=lambda seconds:None)
   owner.start();owner.close();self.assertEqual(len(grants),1);self.assertIs(grants[0]['allowResizing'],True);self.assertEqual((grants[0]['browserScreenWidth'],grants[0]['browserScreenHeight']),(1280,1800));self.assertEqual(sum(method=='POST'for method,_,_ in requests),1)
 def workspace(self,root,*,wrong=False,fingerprint=False,pending=False,raw_first_wrong=False,raw_always_wrong=False,transport_error=False,scrollbar=(0,0),mismatched_css=False,actual_reply=None):
  events=[];raws=[];counts={'call':0,'capture':0}
  class Base:
   def __init__(self,output_dir):self.path=Path(output_dir);self.path.mkdir(parents=True,exist_ok=True)
   def _workspace_dir(self):return self.path
   def start(self,*a,**k):events.append('native-start')
   def capture(self,path):
    counts['capture']+=1;bad=raw_always_wrong or(raw_first_wrong and counts['capture']==1);size=(1600,1974)if bad else(1280,1800)
    raw=b'\x89PNG\r\n\x1a\n'+b'\x00\x00\x00\rIHDR'+struct.pack('>II',*size)+b'raw-native-fixture';Path(path).write_bytes(raw);raws.append(raw);events.append(('native-raw-capture',str(path)))
   def call(self,request,timeout):
    counts['call']+=1;self_test=request['args']['code'];events.append('viewport-controller')
    if actual_reply is not None:return actual_reply
    if transport_error:raise (transport_error if isinstance(transport_error,type) else RuntimeError)('bridge stream failure')
    bun=resolve_test_bun()
    shape=dict(clientWidth=1280-scrollbar[0],clientHeight=1800-scrollbar[1]);css=dict(shape);css.update(clientWidth=1200)if mismatched_css else None
    js='const document={compatMode:"CSS1Compat",documentElement:'+json.dumps(shape)+',body:'+json.dumps(shape)+'};const innerWidth=1280,innerHeight=1800,devicePixelRatio=1,screen={width:1280,height:1800};const session={Emulation:{setDeviceMetricsOverride:async(p)=>{if(p.width!==1280||p.height!==1800||p.screenWidth!==1280||p.screenHeight!==1800||p.deviceScaleFactor!==1)throw Error("wrong pin");}},Runtime:{evaluate:async(p)=>({result:{value:p.awaitPromise?await eval(p.expression):eval(p.expression)}})},Page:{getLayoutMetrics:async()=>('+json.dumps(dict(cssVisualViewport=css,cssLayoutViewport=css))+')}};'+self_test
    env=dict(os.environ,BUN_RUNTIME_TRANSPILER_CACHE_PATH=str(root/'bun-cache'),BUN_INSTALL_CACHE_DIR=str(root/'bun-install'),TMPDIR='/tmp')
    result=subprocess.run([bun,'-e',js],capture_output=True,text=True,env=env,cwd=root,check=True)
    text=result.stdout
    if wrong or(pending and counts['call']==1):text=text.replace('"width":1280','"width":1600')
    if fingerprint:text=text.replace('"dpr":1','"dpr":1.25').replace('"screenWidth":1280','"screenWidth":1600')
    return dict(ok=True,result={'output':text})
  worker=types.SimpleNamespace(OwnedBrowser=object,PairedModel=object,BrowserCodeWorkspace=Base,judge=lambda*a:None)
  registry=integration.install(worker,types.SimpleNamespace(),http_factory=lambda:None,admission=lambda*a:None,metadata_factory=lambda*a:{},on_halt=lambda*a:None,max_proxy_usd='10')
  result=worker.BrowserCodeWorkspace(output_dir=root/'episode/workspace');result.test_raws=raws
  return result,events,registry
 def test_actual_pin_javascript_raw_capture_order(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as d:
   root=Path(d);workspace,events,registry=self.workspace(root);workspace.start('private','target');workspace.capture(root/'unaltered-final.png')
   self.assertEqual(events,['native-start','viewport-controller','viewport-controller',('native-raw-capture',str(root/'unaltered-final.png'))])
   proofs=list((root/'episode').glob('cloud-viewport-proof-*.json'));self.assertEqual(len(proofs),2)
   self.assertEqual(json.loads(proofs[0].read_text())['measured']['dom']['height'],1800);registry.restore()
 def test_wrong_actual_viewport_blocks_before_capture(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as d:
   root=Path(d);workspace,events,registry=self.workspace(root,wrong=True)
   with self.assertRaisesRegex(ValueError,'did not honor'):workspace.capture(root/'unwritten.png')
   self.assertEqual(events,['viewport-controller']*3);self.assertFalse((root/'unwritten.png').exists());self.assertEqual(len(list((root/'episode').glob('cloud-viewport-attempt-*.json'))),3);registry.restore()
 def test_random_fingerprint_diagnostics_do_not_override_actual_css_raw(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as d:
   root=Path(d);workspace,events,registry=self.workspace(root,fingerprint=True);workspace.start('private','target');workspace.capture(root/'final.png')
   self.assertEqual((root/'final.png').read_bytes(),workspace.test_raws[-1]);proof=json.loads(next((root/'episode').glob('cloud-viewport-proof-*.json')).read_text());self.assertEqual(proof['measured']['dom']['dpr'],1.25);self.assertTrue(proof['reported_fingerprint_fields_are_diagnostic']);registry.restore()
 def test_pending_render_is_retried_same_workspace_with_all_replies_preserved(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as d:
   root=Path(d);workspace,events,registry=self.workspace(root,pending=True);workspace.start('private','target')
   attempts=sorted((root/'episode').glob('cloud-viewport-attempt-*.json'));self.assertEqual(len(attempts),2);self.assertEqual(events,['native-start','viewport-controller','viewport-controller']);self.assertTrue(all('raw_controller_reply'in json.loads(x.read_text())for x in attempts));registry.restore()
 def test_bad_first_raw_is_preserved_and_only_fresh_passing_raw_used(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as d:
   root=Path(d);workspace,events,registry=self.workspace(root,raw_first_wrong=True);target=root/'final.png';workspace.capture(target)
   self.assertEqual(target.read_bytes(),workspace.test_raws[1]);self.assertEqual((root/'final.png.rejected-1.png').read_bytes(),workspace.test_raws[0]);self.assertNotEqual(workspace.test_raws[0],workspace.test_raws[1]);registry.restore()
 def test_bad_raw_twice_never_populates_final_evidence(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as d:
   root=Path(d);workspace,events,registry=self.workspace(root,raw_always_wrong=True);target=root/'final.png'
   with self.assertRaisesRegex(ValueError,'screenshot'):workspace.capture(target)
   self.assertFalse(target.exists());self.assertEqual(len(workspace.test_raws),2);self.assertEqual(len(list(root.glob('final.png.rejected-*.png'))),2);registry.restore()
 def test_transport_failure_preserved_once_without_stream_retry(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as d:
   root=Path(d);workspace,events,registry=self.workspace(root,transport_error=True)
   with self.assertRaises(RuntimeError):workspace.capture(root/'never.png')
   self.assertEqual(events,['viewport-controller']);proofs=list((root/'episode').glob('cloud-viewport-attempt-*.json'));self.assertEqual(len(proofs),1);self.assertEqual(json.loads(proofs[0].read_text())['error_type'],'RuntimeError');self.assertFalse((root/'never.png').exists());registry.restore()
 def test_oserror_transport_preserved_once_without_stream_retry(self):
  for error_type in (OSError,ConnectionError):
   with self.subTest(error_type=error_type.__name__),tempfile.TemporaryDirectory(dir='/tmp')as d:
    root=Path(d);workspace,events,registry=self.workspace(root,transport_error=error_type)
    with self.assertRaises(error_type):workspace.capture(root/'never.png')
    self.assertEqual(events,['viewport-controller']);proofs=list((root/'episode').glob('cloud-viewport-attempt-*.json'));self.assertEqual(len(proofs),1);self.assertEqual(json.loads(proofs[0].read_text())['error_type'],error_type.__name__);self.assertFalse((root/'never.png').exists());registry.restore()
 def test_real_bun_scrollbar_uses_measured_document_client_but_raw_stays_full(self):
  for gutters in ((15,0),(0,15),(15,15)):
   with self.subTest(gutters=gutters),tempfile.TemporaryDirectory(dir='/tmp')as d:
    root=Path(d);workspace,events,registry=self.workspace(root,scrollbar=gutters);workspace.start('private','target');target=root/'raw.png';workspace.capture(target)
    self.assertEqual(struct.unpack('>II',target.read_bytes()[16:24]),(1280,1800));proof=json.loads(next((root/'episode').glob('cloud-viewport-proof-*.json')).read_text());self.assertEqual(proof['measured']['dom']['rootClient']['width'],1280-gutters[0]);self.assertTrue(proof['document_scrollbar_gutters_are_measured']);registry.restore()
 def test_mismatched_css_or_excessive_gutter_still_blocks_before_native_capture(self):
  for kwargs in (dict(scrollbar=(15,0),mismatched_css=True),dict(scrollbar=(65,0))):
   with self.subTest(kwargs=kwargs),tempfile.TemporaryDirectory(dir='/tmp')as d:
    root=Path(d);workspace,events,registry=self.workspace(root,**kwargs)
    with self.assertRaises(ValueError):workspace.capture(root/'never.png')
    self.assertFalse((root/'never.png').exists());self.assertEqual(len(workspace.test_raws),0);self.assertEqual(len(list((root/'episode').glob('cloud-viewport-attempt-*.json'))),3);registry.restore()
 def test_actual_saved_provider_fractional_css_reply_qualifies_without_altering_values(self):
  import copy
  reply=copy.deepcopy(FRACTIONAL_REPLY)
  with tempfile.TemporaryDirectory(dir='/tmp')as d:
   root=Path(d);workspace,events,registry=self.workspace(root,actual_reply=reply);workspace.capture(root/'raw.png')
   proof=json.loads(next((root/'episode').glob('cloud-viewport-proof-*.json')).read_text());self.assertEqual(proof['measured']['visual']['clientWidth'],1264.800048828125);self.assertEqual(proof['measured']['layout']['clientWidth'],1264);self.assertEqual(proof['css_delta_pixels']['layout']['width'],-1);self.assertEqual(struct.unpack('>II',(root/'raw.png').read_bytes()[16:24]),(1280,1800));registry.restore()
 def test_fractional_css_over_one_pixel_or_nonfinite_or_bool_is_rejected(self):
  import copy
  reply=copy.deepcopy(FRACTIONAL_REPLY);text=reply['result']['output'];row=text.split('owned-viewport-proof:',1)[1].split('\n',1)[0];value=json.loads(row)
  for bad in (1263.99,float('nan'),float('inf'),True,-1):
   with self.subTest(bad=bad),tempfile.TemporaryDirectory(dir='/tmp')as d:
    changed=copy.deepcopy(value);changed['layout']['clientWidth']=bad;r=copy.deepcopy(reply);r['result']['output']='owned-viewport-proof:'+json.dumps(changed)+'\n';root=Path(d);workspace,events,registry=self.workspace(root,actual_reply=r)
    with self.assertRaises(ValueError):workspace.capture(root/'never.png')
    self.assertFalse((root/'never.png').exists());self.assertEqual(len(workspace.test_raws),0);registry.restore()
if __name__=='__main__':unittest.main()
