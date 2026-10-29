import asyncio,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from native_group_transport import BrokerTransport,Group
from native_group_relay import GroupRelay
class Owner:
 def __init__(self,path):self.evidence=path;self.events=[]
 def authorize(self,pid,uid,token,operation):
  if(pid,uid)!=(12,34):raise PermissionError('wrong owner')
 def record(self,event):self.events.append(event)
class Tests(unittest.TestCase):
 def test_oversized_frame_denied_before_socket_or_provider(self):
  from native_group_transport import rpc
  with patch('native_group_transport.MAX_FRAME',64),patch('native_group_transport.socket.socket')as connect:
   with self.assertRaises(ValueError):rpc('/tmp/never-connect',{'payload':'x'*100},time.monotonic()+1)
   connect.assert_not_called()
 def test_atomic_permit_rejects_partial_or_forged_admission(self):
  valid={'group_id':'g','held_model_calls':8,'admitted':True,'reserved_generation_calls':4,'shared_model_inflight_limit':8,'held_usd':1}
  for key,value in [('held_model_calls',4),('admitted',False),('reserved_generation_calls',3),('shared_model_inflight_limit',9),('held_usd',0)]:
   with self.subTest(key=key),self.assertRaises(RuntimeError):Group(None,dict(valid,**{key:value}),time.monotonic()+1)
  self.assertTrue(Group(None,valid,time.monotonic()+1).permit['admitted'])
 def test_upstream_error_wrapper_never_reaches_native_parser(self):
  class T:
   async def call(self,*args):return {'status':429,'body':{'output':[]}}
  g=Group(T(),{'group_id':'g','held_model_calls':8,'admitted':True,'reserved_generation_calls':4,'shared_model_inflight_limit':8,'held_usd':1},time.monotonic()+10)
  with self.assertRaises(RuntimeError):asyncio.run(g.create(role='candidate_0',payload={},deadline_monotonic=g.deadline))
 def test_wrong_peer_cannot_forward_or_record_group(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as td:
   p=Path(td);cfg=p/'cfg.json';cfg.write_text(json.dumps({'group_base_url':'http://127.0.0.1:1','group_secret':'s'}));relay=GroupRelay(cfg,Owner(p))
   for operation in ('group_open','group_create','group_finish','group_record'):
    with self.subTest(operation=operation),patch('native_group_relay.urllib.request.urlopen')as network:
     with self.assertRaises(PermissionError):relay.handle(99,34,{'op':operation,'data':{},'deadline_monotonic':time.monotonic()+2})
     network.assert_not_called()
 def test_pre_admission_failure_private_record_without_provider_or_group(self):
  with tempfile.TemporaryDirectory(dir='/tmp')as td:
   p=Path(td);cfg=p/'cfg.json';cfg.write_text(json.dumps({'group_base_url':'http://127.0.0.1:1','group_secret':'s'}));owner=Owner(p);relay=GroupRelay(cfg,owner);record={'committed':False,'error_type':'BudgetDenied'}
   with patch('native_group_relay.urllib.request.urlopen')as network:
    response=relay.handle(12,34,{'op':'group_record','data':{'kind':'decisions/000','record':record},'deadline_monotonic':time.monotonic()-1})
    self.assertTrue(response['stored']);network.assert_not_called()
   self.assertEqual(json.loads(next((p/'private-unadmitted-decisions').iterdir()).read_text()),record)
   with self.assertRaises(ValueError):relay.handle(12,34,{'op':'group_record','data':{'kind':'physical/000-candidate_0','record':record},'deadline_monotonic':time.monotonic()+1})
 def test_generation_deadline_cannot_be_extended_by_transport(self):
  tr=BrokerTransport('/tmp/no-such-native-fixture.sock','case')
  with self.assertRaises(TimeoutError):asyncio.run(tr.call('create',{},time.monotonic()-1))
 def test_finish_bookkeeping_after_deadline_has_no_generation_dispatch(self):
  class T:
   calls=[]
   async def call(self,op,data,deadline):self.calls.append((op,data,deadline));return {'verified':True}
  t=T();g=Group(t,{'group_id':'g','held_model_calls':8,'admitted':True,'reserved_generation_calls':4,'shared_model_inflight_limit':8,'held_usd':1},time.monotonic()-1)
  asyncio.run(g.finish(selected_index=None,error_type='TimeoutError'));self.assertEqual([x[0]for x in t.calls],['finish']);self.assertGreater(t.calls[0][2],time.monotonic())
if __name__=='__main__':unittest.main()
