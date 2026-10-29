"""Zero-paid-call/zero-browser actual native subprocess and Landlock integration."""
import hashlib,json,os,shutil,subprocess,sys,threading,time,unittest,uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def response(model,turn,index,command):
 output=[{'type':'reasoning','id':f'r-{turn}-{index}','encrypted_content':f'PRIVATE-{turn}-{index}','summary':[]}]
 if turn==0:output.append({'type':'function_call','id':f'f-{index}','call_id':f'call-{index}','name':'run_command','arguments':json.dumps({'command':command})})
 else:output.append({'type':'message','role':'assistant','content':[{'type':'output_text','text':'Fixture completion'}]})
 return {'model':model,'id':f'resp-{turn}-{index}','status':'completed','output':output,'usage':{'input_tokens':10,'output_tokens':2,'total_tokens':12}}

class Integration(unittest.TestCase):
 def test_luna_n3_confined(self):self.run_arm('luna_high_n3_select_high')
 def test_sol_n3_confined(self):self.run_arm('sol_high_n3_select_high')
 def test_sol_single_native_query_unmodified(self):self.run_arm('sol_high_single')
 def run_arm(self,arm):
  n3=arm!='sol_high_single';model='gpt-6-luna' if arm.startswith('luna')else'gpt-6-sol'
  fixture=ROOT.parent/'integration-tests'/('run-'+uuid.uuid4().hex);runtime=fixture/'runtime';runtime.mkdir(parents=True)
  for p in ROOT.glob('*.py'):shutil.copy2(p,runtime/p.name)
  shutil.copytree(ROOT/'confinement_support',runtime/'confinement_support')
  (runtime/'upstream').symlink_to(ROOT/'upstream',target_is_directory=True)
  import yaml
  override=yaml.safe_load((ROOT/'local_override.yaml').read_text());override['agent']['step_limit']=2
  (runtime/'local_override.yaml').write_text(yaml.safe_dump(override));(runtime/'plan.json').write_text('{}')
  shard=runtime/'shard0';host=shard/'.host';host.mkdir(parents=True)
  secret='fixture-secret-'+uuid.uuid4().hex;private=host/'private';private.mkdir();secretfile=host/'secret.txt';secretfile.write_text(secret)
  output=shard/'episodes'/'fixture-case';output.mkdir(parents=True)
  config=host/'group.json'
  # Actual shell helper must not read private config/evidence nor use native group RPC.
  command="""python - <<'PYCODE'
import json,os,socket,runpy,subprocess,sys
from pathlib import Path
cfg=json.loads(Path(os.environ['CUAWRIGHT_NATIVE_GUARD_CONFIG']).read_text())
for p in [SECRETFILE,PRIVATEFILE]:
 try:Path(p).read_text()
 except PermissionError:pass
 else:raise AssertionError('Private host data exposed')
for p in [PRIVATEDIR,HOSTDIR]:
 try:list(Path(p).iterdir())
 except PermissionError:pass
 else:raise AssertionError('Private directory names exposed')
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.connect(cfg['browser_broker_socket']);s.sendall((json.dumps({'op':'group_open','data':{},'deadline_monotonic':DEADLINE})+chr(10)).encode());r=json.loads(s.makefile('rb').readline());assert r['verified']is False,r
subprocess.run([sys.executable,'-m','cuawright.webwright.tools.browser_session','--help'],check=True,capture_output=True)
runpy.run_path(cfg['browser_launcher'],run_name='offline_browser_proxy_import_only')
Path('selected-marker').write_text('selected-only')
PYCODE""".replace('SECRETFILE',repr(str(secretfile))).replace('PRIVATEFILE',repr(str(private/'sentinel.json'))).replace('DEADLINE',repr(time.monotonic()+100)).replace('PRIVATEDIR',repr(str(private))).replace('HOSTDIR',repr(str(host)))
  (private/'sentinel.json').write_text('private raw proposal')
  state={'groups':{},'records':[],'calls':[],'active':0,'peak':0,'lock':threading.Lock()}
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def do_POST(self):
    data=json.loads(self.rfile.read(int(self.headers['Content-Length'])));kind=self.path.rsplit('/',1)[-1]
    try:
     if self.path.startswith('/group/'):
      assert self.headers.get('X-Campaign-Group-Secret')==secret
      if kind=='open':
       gid=str(uuid.uuid4());state['groups'][gid]=data;answer={'group_id':gid,'held_model_calls':8,'admitted':True,'reserved_generation_calls':4,'shared_model_inflight_limit':8,'held_usd':1,'deadline_epoch':time.time()+100}
      elif kind=='create':
       g=state['groups'][data['group_id']];turn=int(g['decision']);role=data['role'];index=data.get('candidate_index',9)
       with state['lock']:state['active']+=1;state['peak']=max(state['peak'],state['active']);state['calls'].append(data)
       try:
        time.sleep(.025)
        if role=='selector':
         raw=response(g['candidate_payload']['model'],turn,9,'');raw['output'][-1]={'type':'message','role':'assistant','content':[{'type':'output_text','text':'{"selected_index":1}'}]}
        else:
         assert data['payload']==g['candidate_payload']
         raw=response(g['candidate_payload']['model'],turn,index,command if index==1 else 'touch UNSELECTED-'+str(index))
        answer={'status':200,'body':raw,'physical_request_id':str(uuid.uuid4())}
       finally:
        with state['lock']:state['active']-=1
      elif kind=='finish':answer={'settled':True,'selected_index':data['selected_candidate_index']}
      elif kind=='record':
       record=data['record'];target=private/(str(len(state['records']))+'.json');target.write_text(json.dumps(record));state['records'].append(data);answer={'stored':True,'record_sha256':hashlib.sha256(canonical(record).encode()).hexdigest()}
      else:raise AssertionError(kind)
     else:
      assert not n3 and self.path.endswith('/responses') and data['model']==model
      turn=len(state['calls']);state['calls'].append({'payload':data});answer=response(model,turn,1,command)
     raw=json.dumps(answer).encode();self.send_response(200);self.end_headers();self.wfile.write(raw)
    except BaseException as exc:
     raw=json.dumps({'error':repr(exc)}).encode();self.send_response(500);self.end_headers();self.wfile.write(raw)
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  base=f'http://127.0.0.1:{server.server_port}';config.write_text(json.dumps({'group_base_url':base,'group_secret':secret}))
  task=host/'task.json';task.write_text(json.dumps({'task_id':'fixture-case','intent':'Offline selected-only fixture','start_url':'https://invalid.example'}))
  source_hashes={p.name:digest(p)for p in runtime.glob('*.py')};inventory={'passed':True,'frozen_plan_sha256':digest(runtime/'plan.json'),'source_hashes':source_hashes};ip=runtime/'fixture-inventory.json';ip.write_text(json.dumps(inventory));(shard/'registration.json').write_text(json.dumps({'shard':0,'execution_preflight_path':str(ip),'execution_preflight_sha256':digest(ip)}))
  cmd=[sys.executable,'-B',str(runtime/'native_episode.py'),'--task-json',str(task),'--output',str(output),'--endpoint',base+'/model/fixture-case/responses','--arm',arm,'--group-config',str(config),'--absolute-deadline',str(time.time()+100)]
  env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(runtime))
  try:proc=subprocess.run(cmd,env=env,cwd=runtime,capture_output=True,text=True,timeout=110)
  finally:server.shutdown();server.server_close();thread.join()
  (fixture/'native.log').write_text(proc.stdout+'\n'+proc.stderr)
  self.assertEqual(proc.returncode,0,proc.stdout+'\n'+proc.stderr)
  result=json.loads((output/'result.json').read_text());self.assertEqual(result['decisions'],2,result)
  self.assertEqual(result['exit_status'],'LimitsExceeded',result) # native completion gate rejected missing reflection evidence
  self.assertEqual((output/'selected-marker').read_text(),'selected-only');self.assertFalse(list(output.glob('UNSELECTED-*')))
  self.assertEqual(len(state['calls']),8 if n3 else 2);self.assertEqual(state['peak'],3 if n3 else 0);self.assertEqual(len(state['records']),18 if n3 else 0)
  self.assertTrue(result['owned_teardown']['process_verification']['verified']);self.assertEqual(result['native_protocol']['native_query_adapter_enabled'],n3)
  if not n3:self.assertTrue(result['native_protocol']['native_agent_query_unmodified'])
  self.assertNotIn(secret,(output/'guard-config.json').read_text());self.assertNotIn('group.json',(output/'guard-config.json').read_text())
  events=list((output.parent/'.broker-evidence').glob('*/browser-broker-events.jsonl'));self.assertEqual(len(events),1);self.assertNotIn('"event": "reserved"',events[0].read_text())
  trajectory=json.loads((output/'trajectory.json').read_text());txt=json.dumps(trajectory)
  self.assertIn('call-1',txt);self.assertNotIn('call-0',txt);self.assertNotIn('call-2',txt)
  next_payload=canonical([g['candidate_payload']for g in state['groups'].values()if g['decision']=='001'][0]if n3 else state['calls'][1]['payload']);self.assertIn('PRIVATE-0-1',next_payload);self.assertNotIn('PRIVATE-0-0',next_payload);self.assertNotIn('PRIVATE-0-2',next_payload)
  self.assertIn('Completion blocked',txt)
  summary={'verified':True,'fixture':str(fixture),'actual_native_landlock':True,'arm':arm,'native_decisions':2,'physical_fake_generations':8 if n3 else 2,'peak_candidate_inflight':state['peak'],'selected_execution_count':1,'unselected_execution_count':0,'private_records':len(state['records']),'native_completion_gate_rejected_missing_evidence':True,'paid_provider_calls':0,'physical_browser_starts':0,'source_hashes':source_hashes}
  (ROOT.parent/('confined-native-integration-'+arm+'.json')).write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':unittest.main()
