"""Offline whole-coordinator replay from immutable observations and paid replies (v7: plus v10 Piotr SelectionARM cells).

No live browser/model/provider/website is used. Upstream candidate parsing,
filtering, deduplication, confidence ordering, bracket comparisons, terminal
helpers and released history execute again. Only recorded response transport,
recorded browser observations and persistence are supplied.
"""
from pathlib import Path
from copy import deepcopy
from collections import defaultdict,deque
import asyncio,hashlib,importlib.util,json,sys,threading,re

# v2 (2026-10-24): independent re-implementation of the v7 cell adaptations the user directed
# ("force json format (and strip json if needed)"): native_policy requests carry
# response_format=json_object; a whole-reply ```json fence is stripped from policy/auxiliary text.
V7_JSON_FENCE=re.compile(r'\A\s*```(?:json)?[ \t]*\n?(.*?)\n?\s*```\s*\Z',re.S)
def v7_strip(text):
 m=V7_JSON_FENCE.match(text) if isinstance(text,str) else None
 return m.group(1) if m else text

# v4 (2026-10-25): independent re-implementation of the v8 cell science changes ("ok just use gpt-4o. and go for it"):
# 1024x2490 screen==viewport, greedy WebArbiter-7B reward payloads, bracketless terminal messages executed as
# `send_msg_to_user [text]`, and for arbiter7b/shepherd8b the detector stop action joins the selector pool.
V8_W,V8_H=1280,720  # v8 audit (2026-10-27): v11 cohorts use the user-chosen 1280x720 viewport
V8_TERMINAL=re.compile(r'\A\s*send_msg_to_user\b(.*)\Z',re.S)
V8_BRACKETED=re.compile(r'\A\s*send_msg_to_user\s*\[.*\]\s*\Z',re.S)
def v8_bracket(a):
 if not isinstance(a,str):return a
 m=V8_TERMINAL.match(a)
 if m is None or V8_BRACKETED.match(a):return a
 x=m.group(1).strip();return f'send_msg_to_user [{x}]' if x else a
def v8_terminal(t):
 return dict(t,action=v8_bracket(t['action'])) if isinstance(t,dict) and isinstance(t.get('action'),str) else t
def v8_plain(v):
 from dataclasses import is_dataclass,asdict
 if is_dataclass(v):return v8_plain(asdict(v))
 if isinstance(v,dict):return {str(k):v8_plain(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [v8_plain(x) for x in v]
 return v
def v8_action(s):return s.get('action') if isinstance(s,dict) else getattr(s,'action',None)
def v8_pool(candidates,terminal):
 pool=deepcopy(list(candidates))
 if not terminal or not isinstance(terminal.get('action'),str) or not terminal['action'].strip():return pool,None
 for i,c in enumerate(pool):
  if c.get('action')==terminal['action']:return pool,i
 pool.append(dict(thought=terminal.get('thought',''),action=terminal['action'],confidence=terminal.get('confidence',0.0),status='v8_terminal_detector_candidate',reasoning=None,meaning=None))
 return pool,len(pool)-1

# v5 (2026-10-26): independent re-implementation of the v9 Best-of-20 top-5-frequency candidate protocol
# (WebShepherd App. F.2 / WebArbiter Sec. 5.2.1) for arbiter7b/shepherd8b policy replies.
V9_N,V9_TOP=20,5
def v9_candidates(response,observation,sampling,one):
 from collections import Counter
 choices=response.get('choices');require(isinstance(choices,list) and len(choices)==V9_N,'v9 reply must carry 20 samples')
 valid=[]
 for i,c in enumerate(choices):
  try:
   p=json.loads(v7_strip(one(dict(choices=[c],usage=response.get('usage')))));b=p.get('BLOCKS') if isinstance(p,dict) else None
   if not isinstance(b,list) or len(b)!=1 or not isinstance(b[0],dict):continue
   th=b[0].get('thought').strip() if isinstance(b[0].get('thought'),str) else '';ac=b[0].get('action').strip() if isinstance(b[0].get('action'),str) else ''
   if th and ac and sampling.filter_candidates([sampling.BlockInfo(thought=th,action=ac,confidence=None)],observation):valid.append((i,th,ac))
  except Exception:continue
 n=Counter(a for _,_,a in valid);first={}
 for i,th,a in valid:first.setdefault(a,(i,th))
 top=sorted(n,key=lambda a:(-n[a],first[a][0]))[:V9_TOP]
 return json.dumps(dict(BLOCKS=[dict(thought=first[a][1],action=a,confidence=n[a]/len(valid),frequency=n[a]) for a in top]))

def require(v,m):
 if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def checked(v):
 p=Path(v['path']);require(p.is_absolute() and p.resolve()==p and p.is_file() and sha(p)==v['sha256'],'Replay reference changed');return p

def module(name,v):
 p=checked(v)
 if name in sys.modules:
  require(Path(sys.modules[name].__file__).resolve()==p,'Conflicting replay module');return sys.modules[name]
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def stripped_event(v):
 if isinstance(v,dict):return {k:stripped_event(x) for k,x in v.items() if k not in ('timestamp','execution_time')}
 if isinstance(v,list):return [stripped_event(x) for x in v]
 return v

class SavedClient:
 _webprm_broker_client=True
 _webprm_budget_owner='webprm_worker.completion'
 def __init__(self,pairs,native_responses):
  self._lock=threading.RLock();self.queues=defaultdict(deque);self.count=0
  for p in pairs:
   request=p['request']
   if request['purpose']=='native_fuzzy':continue
   require(p['response'] is not None and p['broker'] is not None,'Successful coordinator lacks settled response')
   key=digest([request['payload'],request['purpose'],bool(request.get('local'))])
   request_id=p['broker']['request_id'];response=deepcopy(native_responses.get(request_id,p['response']['response']))
   self.queues[key].append((request_id,response))
 def __call__(self,config,payload,journal,*,purpose,local,**kwargs):
  key=digest([payload,purpose,bool(local)])
  with self._lock:
   require(self.queues[key],'Offline coordinator attempted an unrecorded/different request')
   ident,response=self.queues[key].popleft();self.count+=1;return deepcopy(response)
 def finish(self):require(not any(self.queues.values()),'Saved coordinator/selector requests unused by offline replay')

class SavedBrowser:
 def __init__(self,native,raw,observations,rawresult):
  self.native=native;self.raw=raw;self.observations=observations;self.rawresult=rawresult;self.cursor=0;self.closed=False
  self.trajectory=deepcopy(raw)
 async def start(self,config):return deepcopy(self.observations[0])
 async def execute(self,action):
  action=v8_bracket(action)
  parsed=self.native.actions.create_id_based_action(action);plain=self.native_module.plain(parsed)
  expected=self.raw[2*self.cursor+1]
  require(plain==expected,'Original parser/action dispatch differs from saved native action')
  self.cursor+=1
  terminal=int(parsed['action_type'])==int(self.native.actions.ActionTypes.SEND_MESSAGE_TO_USER)
  if terminal:
   require(len(self.raw)%2==0 and self.cursor==len(self.observations),'Terminal action must not create a new browser state')
   observation=self.observations[-1]
  else:
   require(self.cursor<len(self.observations),'Actual browser action lacks next saved observation');observation=self.observations[self.cursor]
  result=dict(action=plain,observation=deepcopy(observation),terminal=terminal,answer=parsed.get('answer') if terminal else None)
  if terminal:
   dispatch=self.rawresult['selected_actions'][self.cursor-1]['execution']['terminal_dispatch'];document=read(checked(dispatch))
   require(document==dict(source=ref(self.native.post.__file__),method='StepManagementMixin.execute_action_and_append_state',rule='Faithful terminal boundary adapter: parsed SEND_MESSAGE_TO_USER skips browser and new state',whole_dispatch_method_invoked=False,no_browser_step=True,no_new_state=True,action_string=action,raw_action=plain,observation=observation['receipt']),'Terminal meta-action dispatch boundary/source/actual observation changed')
   result['terminal_dispatch']=dispatch
  return result
 async def evaluate(self,callback,task,terminal):
  require(self.cursor==len(self.raw)//2,'Recorded actions remain unexecuted at evaluation')
  return deepcopy(self.rawresult['verdict'])
 async def close(self):self.closed=True;return deepcopy(self.rawresult['closure'])


def replay_bundle(*,plan,paths,calls,replay_root):
 rawresult=read(paths['native_result']);context_record=read(paths['native_context']);cfg=read(paths['native_bridge_config']);raw=read(paths['native_raw_trajectory'])
 require(rawresult['valid'] is True and rawresult.get('closure',{}).get('closed') is True and not rawresult['closure'].get('errors'),'Closed native result required')
 require(raw and ((rawresult.get('terminal') is None and len(raw)%2==1) or (rawresult.get('terminal') is not None and len(raw)%2==0)),'Raw native terminal must end with actual action and no extra state')
 components=plan['native_components'];N=module('_webprm_independent_native_episode',components['episode']);B=module('_webprm_independent_native_broker',components['broker']);native=N.load_native(plan['arbiter_source'])
 # Exact observed-image/node-ID state projection; every raw state has its actual capture.
 observations=[]
 for i,entry in enumerate(raw[::2]):
  image=checked(entry['image']);obspath=image.with_suffix('.json');o=read(obspath);o['receipt']=ref(obspath)
  order=o.get('obs_node_order')
  require(isinstance(order,list),'Production replay requires explicit captured dictionary order')
  require(len(order)==len(set(order))==len(o['obs_nodes_info']) and set(order)==set(o['obs_nodes_info']),'Captured native AX node insertion order is unavailable')
  o['obs_nodes_info']={key:o['obs_nodes_info'][key] for key in order}
  require(o['index']==i and o['text']==entry['observation']['text'] and o['obs_nodes_info']==entry['info']['observation_metadata']['text']['obs_nodes_info'],'Raw original AX/node map differs from capture')
  require(checked(o['image'])==image,'Raw screenshot differs from observation')
  geometry=o['geometry'];require(geometry==dict(screenWidth=V8_W,screenHeight=V8_H,innerWidth=V8_W,innerHeight=V8_H,devicePixelRatio=1),'Native screen/viewport/DPR differs from v11 1280x720')
  observations.append(o)
 # Journal receipts join the native bridge to actual financial broker request IDs.
 events=[json.loads(line) for line in Path(paths['native_bridge_events']).read_text().splitlines()]
 pending={};responses={};recorded_requests=[]
 for e in events:
  if e['event']=='broker_request':
   require(e['call_id'] not in pending,'Repeated native call ID');pending[e['call_id']]=read(checked(e['request']));recorded_requests.append(pending[e['call_id']])
  elif e['event']=='broker_completed':
   require(e['call_id'] in pending,'Completed native call lacks request');response=read(checked(e['response']));request=pending.pop(e['call_id']);receipt=response['_webprm_broker']
   require(e['receipt']==receipt and receipt['payload_sha256']==__import__('openwebrl.webprm_broker',fromlist=['payload_sha256']).payload_sha256(request['payload']),'Native response/request/receipt mismatch')
   require(receipt['request_id'] not in responses,'Repeated financial request ID');responses[receipt['request_id']]=response
  else:raise ValueError('Successful native episode contains failed/rejected/incomplete bridge event')
 require(not pending,'Successful native episode retains unfinished model call')
 financial={p['broker']['request_id']:p for p in calls['pairs'] if p.get('broker')}
 for ident,response in responses.items():
  require(ident in financial,'Native response has no metered financial counterpart');pair=financial[ident];receipt=response['_webprm_broker']
  require(pair['request']['reservation']==receipt['reservation_id'] and all(pair['response']['response'].get(k)==response.get(k) for k in ('model','choices','usage')),'Native response differs from saved actual paid reply')
 client=SavedClient(calls['pairs'],responses)
 Spec=native.tools['engine'].ModelSpec
 def spec(value):return Spec(**value) if value is not None else None
 context=native.coordinator.RunContext.create(**dict(context_record,policy_model=spec(context_record['policy_model']),reward_model=spec(context_record['reward_model'])))
 routes={(r['provider'],r['model'],r['mode']):{k:r[k] for k in ('purpose','allowed_gen_configs')} for r in cfg['routes']}
 root=Path(replay_root);root.mkdir(parents=True,exist_ok=False)
 method=plan['selector'];callconfig=dict(plan,identity_sha256=digest(plan))
 from openwebrl.webprm_webarena_worker import single_action_system_prompt,ENVELOPE_NO_SEARCH_PROMPT,ViewerSelector
 class ReplayBridge(B.NativeBrokerBridge):
  def request(self,engine,system,user,model=None):
   value=super().request(engine,system,user,model)
   if value['purpose']=='native_policy':
    value['payload']['response_format']={'type':'json_object'};value['policy_json_mode']='v7_user_directed_json_object'
   if value['purpose']=='native_reward':
    value['payload'].update(temperature=0.,top_p=1.);value['reward_decoding']='v8_greedy_released_reward_lm_config'
   if value['purpose']=='native_policy' and method=='piotr_selectionarm':
    value['upstream_unadapted_payload']=deepcopy(value['payload']);value['policy_adaptation']='v10_released_five_block_prompt_unchanged'
   elif value['purpose']=='native_policy':
    value['upstream_unadapted_payload']=deepcopy(value['payload']);value['payload']['messages'][0]['content']=single_action_system_prompt(value['payload']['messages'][0]['content'],version=ENVELOPE_NO_SEARCH_PROMPT);value['policy_adaptation']='disclosed_single_action_envelope_v2'
    if method!='no_search':value['payload'].update(n=20,temperature=1.0,top_p=1.0);value['policy_adaptation']='v9_best_of_20_top5_frequency'
   return value
  def response_text(self,response):
   if response.get('_webprm_broker',{}).get('purpose')=='native_policy' and method in ('arbiter7b','shepherd8b'):
    require('text' in observed,'v9 policy reply replayed without its observation')
    one=lambda value:super(ReplayBridge,self).response_text(value)[0]
    super().response_text(dict(choices=[dict(finish_reason='stop',message=dict(role='assistant',content='{}'))],usage=response.get('usage')))
    return v9_candidates(response,observed['text'],native.tools['sampling'],one),deepcopy(response.get('usage'))
   text,usage=super().response_text(response)
   if response.get('_webprm_broker',{}).get('purpose') in ('native_policy','native_auxiliary'):text=v7_strip(text)
   if method=='no_search' and response.get('_webprm_broker',{}).get('purpose')=='native_policy':
    parsed=json.loads(text)
    require(isinstance(parsed,dict) and isinstance(parsed.get('BLOCKS'),list) and len(parsed['BLOCKS'])==1,'Single-action control must preserve exactly one raw BLOCK before filtering/deduplication')
   return text,usage
 bridge=ReplayBridge(native.tools['engine'].LMEngine,routes=routes,client=client,config=callconfig,cell_id=f"webarena_lite_{plan['task_ids'][0]:03d}/"+plan['actor_model']+'__'+method,local_models=cfg['local_models'],declared_default_model=spec(cfg['declared_default_model']),wire_model_aliases=cfg['wire_model_aliases'],evidence_root=root/'bridge',max_attempts=cfg['max_attempts'])
 shepherd=ViewerSelector(method,callconfig,root/'unused.jsonl',request=client) if method=='shepherd8b' else None
 from openwebrl.webprm_native_piotr_v11 import NativePiotrSelector
 piotr=NativePiotrSelector(callconfig,root/'unused-piotr.jsonl',request=client) if method=='piotr_selectionarm' else None
 selectors=[]
 observed={}
 async def adapter(*,native,agent,context,inputs):
  observed['text']=inputs['observation_text'];stash={};released_detect=type(agent)._detect_terminal_if_needed;old=native.coordinator.run_tournament
  async def detect(*a,**k):
   t=v8_terminal(await released_detect(agent,*a,**k))
   if method in ('no_search','piotr_selectionarm') or not t:return t
   stash['terminal']=deepcopy(t);return None
  async def select(**kwargs):
   candidates,stop=v8_pool(kwargs['candidates'],stash.get('terminal'))
   if method=='arbiter7b':
    out=await old(**dict(kwargs,candidates=candidates))
    selectors.append(dict(method=method,candidates=candidates,v8_stop_index=stop,v8_terminal=stash.get('terminal'),tournament_success=bool(out.get('success')),final_winner=v8_plain(out.get('final_winner')) if out.get('success') else None,adaptation='v8_stop_competes_in_released_tournament'))
    return out
   if method=='no_search':require(len(candidates)==1,'Single-action control no longer has exactly its one admissible response');index,trace=0,[]
   elif method=='piotr_selectionarm':
    index,trace=await asyncio.to_thread(piotr.select,candidates,dict(intent=context.intent,current_url=inputs['current_url'],observation_text=inputs['observation_text']),deepcopy(browser.observations[browser.cursor]),inputs['current_step'])
   else:
    # v3 (2026-10-25): WebShepherd's checklist prompt reads the task start URL (v7i KeyError 'start_url').
    proposal=dict(candidates=deepcopy(candidates),terminal=False,context=dict(intent=context.intent,start_url=resolved['start_url'],text_observation=inputs['observation_text'],current_url=inputs['current_url'],trajectory=inputs['trajectory_text'],state_diff=inputs['state_diff']))
    index,trace=await asyncio.to_thread(shepherd.select,proposal,inputs['current_step'])
   selectors.append(dict(method=method,candidates=candidates,selected_index=index,trace=trace,v8_stop_index=stop,v8_terminal=stash.get('terminal'),adaptation='shared_released_policy_matched_control' if method=='no_search' else 'v10_piotr_selectionarm_released_five_block_pool' if method=='piotr_selectionarm' else 'v8_stop_competes_with_shepherd_pool'))
   return dict(success=True,final_winner=deepcopy(candidates[index]))
  agent._detect_terminal_if_needed=detect;native.coordinator.run_tournament=select
  try:decision=await agent.execute_step(**inputs)
  finally:
   native.coordinator.run_tournament=old;del agent._detect_terminal_if_needed
  if method!='no_search' and not v8_action(decision.get('selected_action')) and stash.get('terminal'):
   decision=dict(decision,selected_action=deepcopy(stash['terminal']),v8_only_stop_candidate=True)
  if piotr:piotr.note_decision(decision)
  return decision
 class PiotrSavedBrowser(SavedBrowser):
  async def execute(self,action):
   piotr.note_execution(action,deepcopy(self.observations[self.cursor]))  # pre-action frame, as the cell's browser hook
   return await super().execute(action)
 browser=(PiotrSavedBrowser if piotr else SavedBrowser)(native,raw,observations,rawresult);browser.native_module=N
 resolved=read(Path(paths['native_result']).parent/'resolved-task.json')
 # Callback is deliberately never invoked by SavedBrowser: evaluator is replayed separately.
 result=asyncio.run(N.run_episode(task=resolved,context=context,browser=browser,completion_scope=bridge.installed(),evaluate=lambda *a:None,evidence_root=root/'episode',max_steps=plan['max_turns'],native=native,decision_adapter=adapter))
 require(result['valid'] is True,'Whole offline coordinator failed: '+str(result.get('error')))
 client.finish();require(result['selected_actions']==rawresult['selected_actions'] and result.get('terminal')==rawresult.get('terminal') and result['status']==rawresult['status'],'Whole upstream selected actions/terminal/status differ on offline replay')
 original_episode=Path(paths['native_result']).parent
 original_decisions=sorted(original_episode.glob('decision-*.json'));actual_decisions=sorted((root/'episode').glob('decision-*.json'))
 require(len(original_decisions)==len(actual_decisions),'Missing/extra saved decision turn')
 for a,b in zip(original_decisions,actual_decisions):require(read(a)==read(b),'Native candidate/selection/raw observation/history/state-difference decision changed')
 original_fallbacks=sorted(original_episode.glob('*fallback*.json'));actual_fallbacks=sorted((root/'episode').glob('*fallback*.json'))
 require([p.name for p in original_fallbacks]==[p.name for p in actual_fallbacks],'Native fallback receipt set differs')
 for a,b in zip(original_fallbacks,actual_fallbacks):require(stripped_event(read(a))==stripped_event(read(b)),'Whole original terminal or no-selection fallback receipt differs')
 original_events=sorted(original_episode.glob('event-*.json'));actual_events=sorted((root/'episode').glob('event-*.json'))
 require(len(original_events)==len(actual_events),'Missing/extra coordinator or post-action event')
 for a,b in zip(original_events,actual_events):require(stripped_event(read(a))==stripped_event(read(b)),'Original coordinator/post-action candidate or history event differs')
 replay_events=[json.loads(x) for x in (root/'bridge/events.jsonl').read_text().splitlines()]
 replay_requests=[read(checked(e['request'])) for e in replay_events if e['event']=='broker_request']
 require(CounterDigest(recorded_requests)==CounterDigest(replay_requests),'Complete original native model request contracts differ')
 original_selectors=sorted(Path(paths['result']).parent.glob('selector-*.json'))
 require(len(original_selectors)==len(selectors),'Control selector trace count differs')
 for path,selection in zip(original_selectors,selectors):require(read(path)==selection,'Control critic/candidate selection trace differs')
 return dict(verified=True,mode='whole_original_coordinator_and_saved_browser_no_network',decisions=len(original_decisions),events=len(original_events),native_requests=len(responses),all_coordinator_selector_requests=client.count,actual_actions=browser.cursor,raw_slots_and_survivors_preserved=True,source_module=ref(Path(__file__)),replay_result=ref(root/'episode/result.json'),ignored_event_fields=['timestamp','execution_time'],model_calls=0,browser_calls=0,provider_calls=0)

def CounterDigest(values):
 from collections import Counter
 return Counter(digest(v) for v in values)
