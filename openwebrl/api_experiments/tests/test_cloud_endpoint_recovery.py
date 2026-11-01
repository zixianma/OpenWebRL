from pathlib import Path
import sys,tempfile,unittest,json
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reproduce import cloud_browser as c
ID='11111111-1111-4111-8111-111111111111'
BASE='https://'+ID+'.cdp.browser-use.com'
WS='wss://'+ID+'.cdp.browser-use.com/devtools/browser/private'
class Response:
 status=200
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def read(self):return json.dumps(dict(webSocketDebuggerUrl=WS)).encode()
class Opener:
 def open(self,request,timeout):
  assert request.full_url==BASE+'/json/version'
  assert not request.headers
  assert timeout==20
  return Response()
class Tests(unittest.TestCase):
 def test_same_owned_https_discovery(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as d,patch.object(c.urllib.request,'build_opener',return_value=Opener()):
   self.assertEqual(c.websocket_endpoint(BASE,ID,d),WS)
   self.assertTrue((Path(d)/'cloud-private/cdp-discovery.json').exists())
 def test_foreign_plaintext_credentials_queries_rejected_before_network(self):
  for url in ['http://'+ID+'.cdp.browser-use.com','https://foreign.example','https://user@'+ID+'.cdp.browser-use.com',BASE+'/unexpected']:
   with self.subTest(url=url),patch.object(c.urllib.request,'build_opener',side_effect=AssertionError('network')):
    with self.assertRaises(ValueError):c.websocket_endpoint(url,ID,'/tmp/never-written')
 def test_discovery_preserves_provider_query(self):
  class QueryOpener(Opener):
   def open(self,request,timeout):
    self_seen=request.full_url
    assert self_seen==BASE+'/json/version?opaque=private'
    return Response()
  with tempfile.TemporaryDirectory(dir='/tmp')as d,patch.object(c.urllib.request,'build_opener',return_value=QueryOpener()):
   self.assertEqual(c.websocket_endpoint(BASE+'?opaque=private',ID,d),WS+'?opaque=private')
 def test_chrome_local_ws_path_normalized_onto_owned_tls(self):
  for advertised in ('ws://127.0.0.1:9222/devtools/browser/private','ws://localhost:9222/devtools/browser/private','ws://'+ID+'.cdp.browser-use.com:9222/devtools/browser/private'):
   class Local(Response):
    def read(self):return json.dumps(dict(webSocketDebuggerUrl=advertised)).encode()
   class LocalOpener(Opener):
    def open(self,*a,**k):return Local()
   with tempfile.TemporaryDirectory(dir='/tmp')as d,patch.object(c.urllib.request,'build_opener',return_value=LocalOpener()):
    self.assertEqual(c.websocket_endpoint(BASE,ID,d),WS)
 def test_foreign_discovered_websocket_rejected(self):
  class Foreign(Response):
   def read(self):return b'{"webSocketDebuggerUrl":"wss://foreign.example/private"}'
  class ForeignOpener(Opener):
   def open(self,*a,**k):return Foreign()
  with tempfile.TemporaryDirectory(dir='/tmp')as d,patch.object(c.urllib.request,'build_opener',return_value=ForeignOpener()):
   with self.assertRaises(ValueError):c.websocket_endpoint(BASE,ID,d)
 def test_exception_does_not_echo_private_endpoint(self):
  with patch.object(c.urllib.request,'build_opener',side_effect=RuntimeError(BASE)):
   with self.assertRaisesRegex(RuntimeError,'Owned CDP discovery failed: RuntimeError')as x:c.websocket_endpoint(BASE,ID,'/tmp/never-written')
   self.assertNotIn(BASE,str(x.exception))
if __name__=='__main__':unittest.main()
