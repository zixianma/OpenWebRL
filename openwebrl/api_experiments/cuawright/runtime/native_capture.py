"""Trusted Chrome-origin CDP screenshot tee; no screenshot RPCs are added.
CDP text/binary payloads and screenshot params are forwarded unchanged. Only
Chrome-origin screenshot response bytes can authorize host capture evidence.
"""
from pathlib import Path
import asyncio,base64,contextlib,hashlib,json,math,os,re,shlex,threading,time,uuid
from urllib.parse import urlparse
VERSION='host-chrome-cdp-raw-capture-v1'
MAX_PENDING=4096;MAX_CAPTURE_BASE64=150000000
LOCK=threading.RLock()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def file_sha(p):return sha(Path(p).read_bytes())
def read(p):return json.loads(Path(p).read_text())
def need(ok,msg):
 if not ok:raise ValueError(msg)
def viewport_capture(params,metrics):
 try:
  v=metrics['visualViewport'];c=params['clip'];scale=v['scale']
  return params.get('format','png')in('png','jpeg')and params.get('captureBeyondViewport')is False and scale>0 and all(type(c[k])in(int,float)for k in('x','y','width','height','scale'))and all(abs(c[k]-want)<1e-6 for k,want in [('x',v['pageX']),('y',v['pageY']),('width',math.ceil(v['clientWidth']/scale)),('height',math.ceil(v['clientHeight']/scale)),('scale',scale)])
 except(KeyError,TypeError,ValueError,ZeroDivisionError):return False

class CaptureTracker:
 """Metadata observer only; malformed/ambiguous metadata poisons qualification."""
 def __init__(self,emit,connection):self.emit=emit;self.connection=connection;self.pending={};self.metrics={};self.poisoned=False
 def poison(self,reason):
  self.poisoned=True;self.emit({'event':'metadata_unverified','connection':self.connection,'reason':reason,'epoch':time.time()})
 def client(self,raw):
  try:
   v=json.loads(raw)
   if not isinstance(v,dict)or'id'not in v:return
   key=(v.get('sessionId'),v['id'])
   if key in self.pending:self.poison('concurrent_cdp_id_reuse');return
   if len(self.pending)>=MAX_PENDING:self.poison('pending_metadata_bound');return
   self.pending[key]={'method':v.get('method'),'params':v.get('params',{}),'request_sha256':sha(raw.encode()if isinstance(raw,str)else raw),'requested_epoch':time.time(),'metrics':self.metrics.get(v.get('sessionId'))}
  except(Exception):self.poison('malformed_client_cdp_metadata')
 def chrome(self,raw):
  try:
   v=json.loads(raw)
   if not isinstance(v,dict)or'id'not in v:return
   key=(v.get('sessionId'),v['id']);p=self.pending.pop(key,None)
   if p is None:return
   if p['method']=='Page.getLayoutMetrics'and'result'in v:self.metrics[key[0]]=v['result']
   if p['method']!='Page.captureScreenshot' or 'error'in v:return
   data=v['result']['data']
   if not isinstance(data,str)or len(data)>MAX_CAPTURE_BASE64:self.poison('capture_metadata_bound');return
   raw_image=base64.b64decode(data,validate=True)
   self.emit({'event':'chrome_raw_capture','schema':VERSION,'connection':self.connection,'cdp_id':v['id'],'session_id':key[0],'params':p['params'],'request_sha256':p['request_sha256'],'response_sha256':sha(raw.encode()if isinstance(raw,str)else raw),'raw_capture_sha256':sha(raw_image),'raw_capture_bytes':len(raw_image),'requested_epoch':p['requested_epoch'],'response_epoch':time.time(),'layout_metrics':p['metrics'],'raw_viewport_verified':viewport_capture(p['params'],p['metrics']),'metadata_unambiguous':not self.poisoned})
  except(Exception):self.poison('malformed_chrome_capture_metadata')

class CDPCaptureProxy:
 def __init__(self,target,evidence,browser_identity):
  parsed=urlparse(target);need(parsed.scheme=='ws'and parsed.hostname in('127.0.0.1','localhost','::1')and parsed.port and parsed.path.startswith('/devtools/browser/'),'CDP target must be owned local browser websocket')
  self.close_lock=threading.RLock();self.target=target;self.path=parsed.path;self.evidence=Path(evidence);self.identity=browser_identity;self.ready=threading.Event();self.loop=None;self.runner=None;self.clients=set();self.connection_tasks=set();self.error=None;self.log=self.evidence/('cdp-captures-'+str(browser_identity['pid'])+'.jsonl');self.thread=threading.Thread(target=self._thread,name='owned-cdp-capture-tee',daemon=False);self.thread.start();need(self.ready.wait(10),'Host capture tee startup timeout');need(self.error is None,str(self.error))
 def emit(self,row):
  row={'broker_pid':os.getpid(),'browser_identity':self.identity,'proxy_source_sha256':file_sha(__file__),'target_endpoint_sha256':sha(self.target.encode()),**row}
  with LOCK,self.log.open('a')as f:f.write(json.dumps(row,sort_keys=True)+'\n')
 def _thread(self):
  self.loop=asyncio.new_event_loop();asyncio.set_event_loop(self.loop)
  try:self.loop.run_until_complete(self._start());self.ready.set();self.loop.run_forever()
  except BaseException as e:self.error=repr(e);self.ready.set()
  finally:
   self.loop.run_until_complete(self._cleanup());self.loop.close()
 async def _start(self):
  from aiohttp import web
  app=web.Application();app.router.add_get('/{tail:.*}',self._connect);self.runner=web.AppRunner(app,shutdown_timeout=2);await self.runner.setup();site=web.TCPSite(self.runner,'127.0.0.1',0);await site.start();port=site._server.sockets[0].getsockname()[1];self.endpoint='ws://127.0.0.1:'+str(port)+self.path;self.emit({'event':'proxy_started','epoch':time.time(),'advertised_endpoint':self.endpoint})
 async def _connect(self,request):
  from aiohttp import web,ClientSession,ClientWSTimeout,WSMsgType
  if request.path in ('/json/version','/json/version/'):
   # Preserve native HTTP discovery if explicitly requested; no extra CDP RPC.
   parsed=urlparse(self.target);url='http://'+parsed.netloc+request.path
   async with ClientSession(trust_env=False)as client:
    async with client.get(url)as response:
     body=await response.read()
     if response.status==200:
      info=json.loads(body);need(info.get('webSocketDebuggerUrl')==self.target,'HTTP discovery endpoint is not the owned Chrome')
      info['webSocketDebuggerUrl']=self.endpoint;body=json.dumps(info).encode()
     return web.Response(status=response.status,body=body,content_type='application/json')
  if request.path!=self.path:return web.Response(status=404)
  self.connection_tasks.add(asyncio.current_task())
  downstream=web.WebSocketResponse(autoping=False,autoclose=False,compress=False,max_msg_size=0,timeout=1);await downstream.prepare(request);self.clients.add(downstream);tracker=CaptureTracker(self.emit,uuid.uuid4().hex)
  try:
   async with ClientSession(trust_env=False)as client:
    async with client.ws_connect(self.target,autoping=False,autoclose=False,compress=0,max_msg_size=0,timeout=ClientWSTimeout(ws_close=1))as upstream:
     self.clients.add(upstream)
     async def pipe(src,dst,observe):
      while True:
       msg=await src.receive()
       if msg.type==WSMsgType.TEXT:observe(msg.data);await dst.send_str(msg.data)
       elif msg.type==WSMsgType.BINARY:observe(msg.data);await dst.send_bytes(msg.data)
       elif msg.type==WSMsgType.PING:await dst.ping(msg.data)
       elif msg.type==WSMsgType.PONG:await dst.pong(msg.data)
       elif msg.type==WSMsgType.CLOSE:await dst.close(code=msg.data or 1000,message=(msg.extra or'').encode());break
       elif msg.type in(WSMsgType.ERROR,WSMsgType.CLOSED):break
     tasks=[asyncio.create_task(pipe(downstream,upstream,tracker.client)),asyncio.create_task(pipe(upstream,downstream,tracker.chrome))];done,pending=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
     for t in pending:t.cancel()
     await asyncio.gather(*tasks,return_exceptions=True);self.clients.discard(upstream)
  finally:self.clients.discard(downstream);await downstream.close();self.connection_tasks.discard(asyncio.current_task())
  return downstream
 async def _cleanup(self):
  for c in list(self.clients):
   with contextlib.suppress(Exception):await c.close()
  for t in list(self.connection_tasks):t.cancel()
  if self.connection_tasks:await asyncio.gather(*self.connection_tasks,return_exceptions=True)
  if self.runner is not None:await self.runner.cleanup()
 def close(self):
  with self.close_lock:
   if getattr(self,'closed',False):return self.close_receipt
   if self.loop is not None and self.loop.is_running():self.loop.call_soon_threadsafe(self.loop.stop)
   self.thread.join(10);ok=not self.thread.is_alive();self.emit({'event':'proxy_shutdown','epoch':time.time(),'verified':ok,'listener_and_threads_closed':ok});need(ok,'Host capture tee thread/listener not closed');self.closed=True;self.close_receipt={'verified':ok,'remaining':[] if ok else ['owned-cdp-capture-tee'],'log':str(self.log),'log_sha256':file_sha(self.log)};return self.close_receipt

class DevtoolsAdvertiser:
 def __init__(self,create):self.create=create;self.buffer=b'';self.proxy=None
 def feed(self,data):
  if self.proxy is not None:return data
  self.buffer+=data;need(len(self.buffer)<=65536,'DevTools advertisement buffer exceeded');out=b''
  while b'\n'in self.buffer:
   line,self.buffer=self.buffer.split(b'\n',1);line+=b'\n';match=re.search(rb'DevTools listening on (ws://[^\s]+)',line)
   if match:
    self.proxy=self.create(match.group(1).decode());line=line[:match.start(1)]+self.proxy.endpoint.encode()+line[match.end(1):];out+=line+self.buffer;self.buffer=b'';break
   out+=line
  return out

 def flush(self):
  out=self.buffer;self.buffer=b'';return out

def paid_command_binding(root,shard,episode,row,code,events,parser,owner_events):
 # Bind ANY released selected native command, not a new heredoc-only protocol.
 # Chrome bytes and outside-actor command intervals provide the trust boundary.
 intervals=[]
 for a in owner_events:
  if a['event']!='command_registered':continue
  for z in owner_events:
   if z['event']in('command_finished','command_cleanup')and z['token']==a['token']and z.get('verified',True)is True and a['time']<=row['started_at']<=row['finished_at']<=z['time']:intervals.append((a,z))
 need(len(intervals)==1,'Capture lacks unique closed host-owned selected command interval');admission,terminal=intervals[0];matches=[]
 for debug in sorted((episode/'debug/steps').glob('step_*.json')):
  d=read(debug);command=d.get('bash_command','')
  if sha(command.encode())!=admission['command_sha256']:continue
  need(d['command_text']==d['raw_response']['bash_command']==command,'Native selected command/debug differs')
  decision=int(debug.stem.split('_')[-1])-1
  if read(root/'plan.json')['candidate_count']>1:
   groups=[e for e in events if e['event']=='group_open'and e['case']==episode.name and str(e['decision'])==f'{decision:03d}'];need(len(groups)==1,'Native command lacks unique decision group');gid=groups[0]['id'];done=[e for e in events if e['event']=='group_finish'and e['group_id']==gid]
   if len(done)!=1 or done[0]['error_type']is not None or done[0]['time']>admission['time']:continue
   roles=[e for e in events if e['event']=='group_create_finished'and e['group_id']==gid and e['role']=='candidate_'+str(done[0]['selected_candidate_index'])];need(len(roles)==1 and roles[0]['known_success']is True,'Native command not selected paid proposal');ids=[roles[0]['physical_request_id']]
  else:ids=[e['id']for e in events if e['event']=='reserve'and e.get('case')==episode.name and e.get('pool')=='model'and e.get('stage')!='input_token_count']
  for ident in ids:
   response=root/f'shard{shard}/.host/meter'/(ident+'.response.json');res=[e for e in events if e['event']=='reserve'and e['id']==ident];fin=[e for e in events if e['event']=='finish'and e['id']==ident]
   if len(res)!=1 or len(fin)!=1 or fin[0]['time']>admission['time']or fin[0]['status']!=200 or fin[0]['usage_known']is not True or not response.exists():continue
   need(file_sha(response)==fin[0]['response_sha256'],'Paid native response digest changed')
   try:parsed=parser._parse_response(read(response))
   except (ValueError,KeyError,AttributeError):continue
   if parsed==d['raw_response']:matches.append((fin[0]['time'],debug,command,ident,response))
 need(matches,'Native command lacks known200 selected paid response')
 latest=max(m[0]for m in matches);latest_rows=[m for m in matches if m[0]==latest];identities={(m[2],m[3])for m in latest_rows};need(len(identities)==1,'Native command paid response ambiguous');_,debug,command,ident,response=latest_rows[-1]
 return {'selected_debug_path':str(debug.relative_to(episode)),'selected_debug_sha256':file_sha(debug),'selected_command_sha256':sha(command.encode()),'selected_physical_id':ident,'selected_paid_response_sha256':file_sha(response),'host_command_tokens':[admission['token']]}

def jsonl(path):
 p=Path(path);return [json.loads(line)for line in p.read_text().splitlines()if line.strip()]if p.exists()else []
def inventory_path(root,shard,episode):return Path(root)/f'shard{shard}/.host/capture-inventories'/(Path(episode).name+'.json')
def qualify_image(path,rows,chrome_rows,episode):
 digest=file_sha(path);size=path.stat().st_size;mtime=path.stat().st_mtime;matches=[]
 for c in chrome_rows:
  if c.get('event')!='chrome_raw_capture'or not c.get('metadata_unambiguous')or c['raw_capture_sha256']!=digest or c['raw_capture_bytes']!=size:continue
  for row in rows:
   # Preserve any genuine Chrome screenshot options and bytes. Viewport shape
   # is diagnostic only; no scientific screenshot/default/first50 change.
   if row['started_at']<=c['requested_epoch']<=c['response_epoch']<=row['finished_at'] and c['response_epoch']<=mtime<=row['finished_at']:
    matches.append((row,c))
 need(matches,'Current-folder image lacks trusted Chrome-origin raw byte/time provenance')
 steps={m[0]['browser_step']for m in matches};need(len(steps)==1,'Current-folder capture is ambiguous across native steps')
 return matches[0][0],[c for _,c in matches]
def collect_host_capture_inventory(root,shard,episode):
 """Unconfined worker only, after native owned teardown, before any grading.
 All current native images are inventoried, including unlisted default frames.
 No client observer file or actor-supplied provenance claim is trusted.
 """
 import sys
 from types import SimpleNamespace
 root=Path(root).resolve();episode=Path(episode).resolve();need(episode.is_relative_to(root/f'shard{shard}'),'Capture episode outside bound shard')
 manifest=episode/'browser-steps.jsonl';rows=jsonl(manifest);result=read(episode/'result.json');bv=result['owned_teardown']['broker_verification'];need(bv.get('verified')is True and bv.get('exit_code')==0,'Capture collector requires verified native broker teardown')
 evidence=Path(bv['evidence']).resolve();need(evidence.parent==episode.parent/'.broker-evidence' and evidence.name.startswith(episode.name+'-'),'Capture broker evidence outside private host root')
 owned=jsonl(evidence/'owned-browser-processes.jsonl');broker_events=jsonl(evidence/'browser-broker-events.jsonl');owner_events=jsonl(evidence/'command-timeout-events.jsonl');chrome=[];files={}
 starts=[e for e in broker_events if e['event']=='broker_started'];need(len(starts)==1 and Path(starts[0]['output']).resolve()==episode,'Broker source/episode start receipt missing')
 binding=starts[0]['runtime_child_source_binding'];need(binding==result['runtime_child_source_binding'] and binding['source_hashes'].get('native_capture.py')==file_sha(root/'native_capture.py') and binding['source_hashes'].get('browser_broker.py')==file_sha(root/'browser_broker.py'),'Host capture source not bound to native child receipt')
 inv=read(binding['registered_inventory_path']);need(file_sha(binding['registered_inventory_path'])==binding['registered_inventory_sha256'] and inv['passed']is True and all(inv['source_hashes'].get(n)==h and file_sha(root/n)==h for n,h in binding['source_hashes'].items()),'Capture source inventory differs/unapproved')
 for p in sorted(evidence.glob('cdp-captures-*.jsonl')):
  entries=jsonl(p);starts=[e for e in entries if e['event']=='proxy_started'];ends=[e for e in entries if e['event']=='proxy_shutdown'];need(len(starts)==len(ends)==1 and ends[0]['verified']is True and ends[0]['listener_and_threads_closed']is True,'Capture proxy listener/thread teardown unverified')
  need(not any(e['event']=='metadata_unverified'for e in entries),'Capture proxy observed ambiguous/overflow metadata')
  for e in entries:
   stderr=evidence/('chromium-stderr-'+str(e['browser_identity']['pid'])+'.log');endpoints=re.findall(rb'DevTools listening on (ws://[^\s]+)',stderr.read_bytes());need(len(endpoints)==1 and e['target_endpoint_sha256']==sha(endpoints[0]),'Capture endpoint differs from owned physical Chrome stderr');files[str(stderr)]=file_sha(stderr)
   need(e['proxy_source_sha256']==file_sha(root/'native_capture.py') and e['browser_identity']in owned and e['broker_pid']==e['browser_identity']['broker_pid'],'Capture Chrome/broker/source identity mismatch')
   need(any(b['event']=='spawned'and b['pid']==e['browser_identity']['pid']and b['start_ticks']==e['browser_identity']['start_ticks']and b['reservation']==e['browser_identity']['reservation']for b in broker_events),'Capture not from owned spawned Chrome')
  chrome+=entries;files[str(p)]=file_sha(p)
 for p in (evidence/'owned-browser-processes.jsonl',evidence/'browser-broker-events.jsonl',evidence/'command-timeout-events.jsonl'):
  if p.exists():files[str(p)]=file_sha(p)
 sys.path.insert(0,str(root/'upstream/src'))
 from cuawright.webwright.models.openai_model import parse_tool_call_output
 parser=SimpleNamespace(_parse_response=parse_tool_call_output)
 events=jsonl(root/f'shard{shard}/.host/meter/events.jsonl');images=[];folder=episode/'screenshots'
 for p in sorted(folder.iterdir()if folder.exists()else[],key=lambda p:(p.name.casefold(),p.name)):
  if not p.is_file()or p.suffix.lower()not in {'.png','.jpg','.jpeg','.webp'}:continue
  need(p.resolve().parent==folder.resolve()and not p.is_symlink(),'Capture escapes current native screenshot folder')
  row,proofs=qualify_image(p,rows,chrome,episode);binding=paid_command_binding(root,shard,episode,row,None,events,parser,owner_events)
  tokens=set(binding['host_command_tokens']);admitted=[e for e in owner_events if e['event']=='command_registered'and e['token']in tokens];terminal=[e for e in owner_events if e['event']in('command_finished','command_cleanup')and e['token']in tokens and e.get('verified',True)is True]
  need(any(a['token']==b['token']and a['time']<=row['started_at']<=row['finished_at']<=b['time']for a in admitted for b in terminal),'Capture lacks closed host-owned selected command interval')
  images.append({'path':str(p.relative_to(episode)),'sha256':file_sha(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'browser_step':row['browser_step'],'row_sha256':sha(json.dumps(row,sort_keys=True).encode()),'chrome_proofs':proofs,'selected_command':binding})
 out={'schema':VERSION,'root':str(root),'shard':int(shard),'episode':str(episode),'case':episode.name,'collector_source_sha256':file_sha(root/'native_capture.py'),'broker_source_sha256':file_sha(root/'browser_broker.py'),'upstream_browser_session_sha256':file_sha(root/'upstream/src/cuawright/webwright/tools/browser_session.py'),'browser_steps_sha256':file_sha(manifest)if manifest.exists()else None,'broker_evidence':str(evidence),'host_evidence_files':files,'images':images,'all_current_folder_images_qualified':True}
 target=inventory_path(root,shard,episode);target.parent.mkdir(parents=True,exist_ok=True);need(not target.exists(),'Immutable host capture inventory already exists');target.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');return out

def host_capture_provenance(root,episode,target):
 root=Path(root).resolve();episode=Path(episode).resolve();target=Path(target).resolve();parts=episode.relative_to(root).parts;need(parts[0].startswith('shard'),'Capture episode lacks shard binding');shard=int(parts[0][5:]);ip=inventory_path(root,shard,episode);inv=read(ip)
 need(inv['schema']==VERSION and inv['root']==str(root)and inv['episode']==str(episode)and inv['case']==episode.name and inv['shard']==shard and inv['all_current_folder_images_qualified']is True,'Capture inventory scope mismatch')
 need(inv['collector_source_sha256']==file_sha(root/'native_capture.py')and inv['broker_source_sha256']==file_sha(root/'browser_broker.py')and inv['upstream_browser_session_sha256']==file_sha(root/'upstream/src/cuawright/webwright/tools/browser_session.py'),'Capture inventory source changed')
 need(inv['browser_steps_sha256']==file_sha(episode/'browser-steps.jsonl'),'Capture rows changed after host inventory')
 for name,digest in inv['host_evidence_files'].items():need(file_sha(name)==digest,'Private Chrome/command evidence changed')
 folder=episode/'screenshots';current={str(p.relative_to(episode))for p in folder.iterdir()if p.is_file()and p.suffix.lower()in {'.png','.jpg','.jpeg','.webp'}}if folder.exists()else set();need(current=={i['path']for i in inv['images']},'Current native image inventory changed')
 items=[i for i in inv['images']if (episode/i['path']).resolve()==target];need(len(items)==1,'Current native image absent from host capture inventory');i=items[0]
 need(file_sha(target)==i['sha256']and target.stat().st_size==i['bytes']and target.stat().st_mtime_ns==i['mtime_ns'],'Native raw image changed after host inventory')
 d=episode/i['selected_command']['selected_debug_path'];need(file_sha(d)==i['selected_command']['selected_debug_sha256'],'Selected paid command debug changed')
 response=root/f'shard{shard}/.host/meter'/(i['selected_command']['selected_physical_id']+'.response.json');need(file_sha(response)==i['selected_command']['selected_paid_response_sha256'],'Selected private paid response changed')
 return {'browser_step':i['browser_step'],'capture_api':'Chrome.Page.captureScreenshot/raw-unmodified','host_inventory_path':str(ip),'host_inventory_sha256':file_sha(ip),'raw_capture_sha256':i['sha256'],'selected_physical_id':i['selected_command']['selected_physical_id'],'chrome_response_sha256':[c['response_sha256']for c in i['chrome_proofs']]}
