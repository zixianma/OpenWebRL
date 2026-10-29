"""Zero-paid-call/zero-browser actual native subprocess and Landlock integration."""
import hashlib,json,os,shutil,subprocess,sys,threading,time,unittest,uuid,importlib.util
from unittest.mock import patch
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

class RealGatewayIntegration(unittest.TestCase):
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
import json,os,socket
from pathlib import Path
cfg=json.loads(Path(os.environ['CUAWRIGHT_NATIVE_GUARD_CONFIG']).read_text())
for p in [SECRETFILE,PRIVATEFILE,METEREVENTS,METERLIMITS,OTHERCASEFILE,RAWREQUESTPATH,RAWRESPONSEPATH,GROUPCANDIDATEPATH]:
 try:Path(p).read_text()
 except PermissionError:pass
 else:raise AssertionError('Private host data exposed')
for folder in [Path(METEREVENTS).parent,Path(GROUPCANDIDATEPATH).parent,Path(PRIVATEFILE).parent,Path(OTHERCASEFILE).parent]:
 try:list(folder.iterdir())
 except PermissionError:pass
 else:raise AssertionError('Private host/cross-case directory names exposed')
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.connect(cfg['browser_broker_socket']);s.sendall((json.dumps({'op':'group_open','data':{},'deadline_monotonic':DEADLINE})+chr(10)).encode());r=json.loads(s.makefile('rb').readline());assert r['verified']is False,r
Path('selected-marker').write_text('selected-only')
PYCODE""".replace('SECRETFILE',repr(str(secretfile))).replace('PRIVATEFILE',repr(str(private/'sentinel.json'))).replace('DEADLINE',repr(time.monotonic()+100)).replace('METERDIR',repr(str(runtime/'shard0/.host/meter/private-groups'))).replace('METEREVENTS',repr(str(runtime/'shard0/.host/meter/events.jsonl'))).replace('METERLIMITS',repr(str(runtime/'shard0/.host/meter/limits.json'))).replace('OTHERCASEFILE',repr(str(shard/'episodes/other-case/private.json')))
  (private/'sentinel.json').write_text('private raw proposal')
  (shard/'episodes/other-case').mkdir(parents=True);(shard/'episodes/other-case/private.json').write_text('cross-case private evidence')
  state={'groups':{},'records':[],'calls':[],'active':0,'peak':0,'lock':threading.Lock()}
  spec=importlib.util.spec_from_file_location('actual_fixture_gateway_'+uuid.uuid4().hex,runtime/'budget_gateway.py');G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
  cap={'shard':0,'task_ids':['fixture-case'],'model_usd_all_attempts':100,'model_calls_all_attempts':100,'external_judge_usd_all_attempts':5,'external_judge_calls_all_attempts':1100,'browser_starts_all_attempts':60,'model_max_inflight':8}
  (runtime/'plan.json').write_text(json.dumps({'model':model,'arm_id':arm,'candidate_count':3 if n3 else 1,'shards':[cap]}))
  rates={'input':.1,'cached_input':.01,'cache_write_input':.125,'output':.5};rates={k:v*(20 if model=='gpt-6-sol' else 1)for k,v in rates.items()};(runtime/'pricing-snapshot.json').write_text(json.dumps({'model':model,'usd_per_million_tokens':rates}))
  def forward(endpoint,raw,key,timeout):
   data=json.loads(raw)
   assert endpoint=='responses',endpoint
   with state['lock']:
    position=len(state['calls']);state['calls'].append({'payload':data});state['active']+=1;state['peak']=max(state['peak'],state['active'])
   try:
    time.sleep(.025)
    turn=position//4 if n3 else position;index=position%4 if n3 else 1
    if n3 and index==3:
     assert data.get('tool_choice')=='none';body=response(model,turn,9,'');body['output'][-1]={'type':'message','role':'assistant','content':[{'type':'output_text','text':'{"selected_index":1}'}]}
    else:
     ledger=G._events(G._folder(runtime,0));first=next(e for e in ledger if e['event']=='reserve');groups=[e for e in ledger if e['event']=='group_open'];folder=G._folder(runtime,0)
     guarded=command.replace('RAWREQUESTPATH',repr(str(folder/(first['id']+'.request.json')))).replace('RAWRESPONSEPATH',repr(str(folder/(first['id']+'.response.json')))).replace('GROUPCANDIDATEPATH',repr(str(folder/'private-groups'/(groups[0]['id']+'.candidate.json')))if groups else repr(str(secretfile)))
     body=response(model,turn,index,guarded if index==1 else 'touch UNSELECTED-'+str(index))
    return 200,G._bytes(body)
   finally:
    with state['lock']:state['active']-=1
  mocked=patch.object(G,'_forward',side_effect=forward);mocked.start()
  server,base=G.start_gateway(runtime,0,time.time()+100,'fake-no-provider-key',group_secret=secret)
  config.write_text(json.dumps({'group_base_url':base,'group_secret':secret}))
  task=host/'task.json';task.write_text(json.dumps({'task_id':'fixture-case','intent':'Offline selected-only fixture','start_url':'https://invalid.example'}))
  source_hashes={p.name:digest(p)for p in runtime.glob('*.py')};inventory={'passed':True,'frozen_plan_sha256':digest(runtime/'plan.json'),'source_hashes':source_hashes};ip=runtime/'fixture-inventory.json';ip.write_text(json.dumps(inventory));(shard/'registration.json').write_text(json.dumps({'shard':0,'execution_preflight_path':str(ip),'execution_preflight_sha256':digest(ip)}))
  cmd=[sys.executable,'-B',str(runtime/'native_episode.py'),'--task-json',str(task),'--output',str(output),'--endpoint',base+'/model/fixture-case/responses','--arm',arm,'--group-config',str(config),'--absolute-deadline',str(time.time()+100)]
  env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(runtime))
  try:proc=subprocess.run(cmd,env=env,cwd=runtime,capture_output=True,text=True,timeout=110)
  finally:server.shutdown();server.server_close();mocked.stop()
  durable=G._events(G._folder(runtime,0))
  state['groups']={x['id']:{'decision':x['decision'],'candidate_payload':json.loads((runtime/'shard0/.host/meter/private-groups'/(x['id']+'.candidate.json')).read_text())}for x in durable if x['event']=='group_open'}
  state['records']=[x for x in durable if x['event']=='group_private_record']
  self.assertEqual(G._group_holds(durable),{})
  self.assertEqual(G.totals(runtime,0)['model']['calls'],8 if n3 else 2)
  self.assertEqual(G.totals(runtime,0)['model']['unknown'],0)
  (fixture/'native.log').write_text(proc.stdout+'\n'+proc.stderr)
  self.assertEqual(proc.returncode,0,proc.stdout+'\n'+proc.stderr)
  result=json.loads((output/'result.json').read_text());self.assertEqual(result['decisions'],2,result)
  self.assertEqual(result['exit_status'],'LimitsExceeded',result) # native completion gate rejected missing reflection evidence
  self.assertEqual((output/'selected-marker').read_text(),'selected-only');self.assertFalse(list(output.glob('UNSELECTED-*')))
  self.assertEqual(len(state['calls']),8 if n3 else 2);self.assertEqual(state['peak'],3 if n3 else 1);self.assertEqual(len(state['records']),18 if n3 else 0)
  self.assertTrue(result['owned_teardown']['process_verification']['verified']);self.assertEqual(result['native_protocol']['native_query_adapter_enabled'],n3)
  if not n3:self.assertTrue(result['native_protocol']['native_agent_query_unmodified'])
  self.assertNotIn(secret,(output/'guard-config.json').read_text());self.assertNotIn('group.json',(output/'guard-config.json').read_text())
  events=list((output.parent/'.broker-evidence').glob('*/browser-broker-events.jsonl'));self.assertEqual(len(events),1);self.assertNotIn('"event": "reserved"',events[0].read_text())
  trajectory=json.loads((output/'trajectory.json').read_text());txt=json.dumps(trajectory)
  self.assertIn('call-1',txt);self.assertNotIn('call-0',txt);self.assertNotIn('call-2',txt)
  next_payload=canonical([g['candidate_payload']for g in state['groups'].values()if g['decision']=='001'][0]if n3 else state['calls'][1]['payload']);self.assertIn('PRIVATE-0-1',next_payload);self.assertNotIn('PRIVATE-0-0',next_payload);self.assertNotIn('PRIVATE-0-2',next_payload)
  self.assertIn('Completion blocked',txt)
  summary={'verified':True,'fixture':str(fixture),'actual_native_landlock':True,'arm':arm,'native_decisions':2,'physical_fake_generations':8 if n3 else 2,'peak_candidate_inflight':state['peak'],'actual_gateway_source_sha256':digest(runtime/'budget_gateway.py'),'real_gateway_ledger_group_settlement_verified':True,'selected_execution_count':1,'unselected_execution_count':0,'private_records':len(state['records']),'native_completion_gate_rejected_missing_evidence':True,'paid_provider_calls':0,'physical_browser_starts':0,'source_hashes':source_hashes}
  (ROOT.parent/('confined-native-REAL-gateway-integration-'+arm+'.json')).write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':unittest.main()
