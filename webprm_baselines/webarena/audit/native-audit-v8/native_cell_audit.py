#!/usr/bin/env python3
"""Independent offline audit of a prospective original-coordinator v9 Table 5 cell (Shopping/CMS/Reddit/GitLab).

No browser, provider, allocation, database restoration or state disposal is opened.
Evidence consistency is independent of semantic judge truth. Invalid/guarded/native
verdicts remain distinct; missing evidence cannot become a valid result.
"""
from __future__ import annotations
import argparse,asyncio,hashlib,importlib.util,json,os,re,socket,sys,time
from unittest.mock import patch
from pathlib import Path
from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
ROOT=Path(os.environ.get('WEBPRM_NATIVE_SOURCE_ROOT','/gpfs/projects/krishna/zixianma/OpenWebRL'))
sys.path.insert(0,str(ROOT))
from openwebrl.webprm_webarena import digest
from openwebrl.webprm_webarena_sites import canonical_runtime_path
SCHEMA='native_table5_v11_independent_cell_audit_v8'
# v8 (2026-10-27): v11 cohorts (user chose the 1280x720 setup): no_search (rerun) + piotr_selectionarm per task,
# 1280x720 actor viewport; the ARM sees the actor's own screenshot.
# v7 (2026-10-27): v10 cohorts (user: "yes do A. so then we can use gpt4o + our ARM right?"): one method,
# piotr_selectionarm, on the released coordinator's 5-BLOCK candidates; replay re-requests the saved Piotr replies.
# v6 (2026-10-26): v5 plus GitLab cells: their reset chain is the third-party OCI build/copy/claim chain audited by the
# running package's audit_webprm_table5_cms_cell.audit_cms_reset with the gitlab site module (job 389698 recipe).
# v5 (2026-10-26): v9 cohorts. Independently restated: one released single-site Table 5 site per cohort, Map excluded,
# Reddit task 5 dropped; arbiter7b/shepherd8b candidates are the 5 most frequent valid actions of 20 samples
# (n=20, temperature 1, top_p 1, single-action envelope); no_search unchanged.
V9_PROTOCOL='released_coordinator_original_browser_table5_v9'
V9_SITE_TASKS={'shopping':45,'shopping_admin':35,'reddit':18,'gitlab':30}
V9_RECEIPT=dict(samples=20,top_n=5,sampling=dict(n=20,temperature=1.0,top_p=1.0),prompt='disclosed_single_action_envelope_v2',
 ranking='frequency_desc_then_first_sampled',validity='released_filter_candidates',confidence='sample_share_of_valid',
 methods=['arbiter7b','shepherd8b'],no_search='unchanged_v8')
V9_FUZZY_ERROR='Native fuzzy transport/accounting failed; no retry'
MAGENTO={'shopping','shopping_admin','gitlab'}
# v2 (2026-10-24): fresh v7 cohort -- released Lite task 5 (original task 27) dropped (its gpt-4-1106-preview judge is retired),
# actor gpt-4o-2024-08-06; replay mirrors the v7 JSON-mode request and fence-strip reply adaptations.
DROPPED_TASKS={5}  # released WebArena-Lite id 5 == original WebArena task 27 (old_task_id)
V7_ACTOR='gpt-4o-2024-08-06'
# v4 (2026-10-25): v8 cohort ("ok just use gpt-4o. and go for it"). Independently restated expectations:
# auxiliary helper spec bound to gpt-4o on the wire, 1024x2490 screen==viewport, greedy WebArbiter, stop-competes
# terminal policy and bracket-tolerant terminal execution; replay mirrors them in native_lowlevel_replay.py.
V8_AUX_WIRE='gpt-4o-2024-08-06'
V8_RECEIPT=dict(viewport={'width':1280,'height':720},  # v8 audit: v11 1280x720
 compatibility_geometry=True,auxiliary_wire_model=V8_AUX_WIRE,
 reward_decoding=dict(temperature=0.,top_p=1.),terminal_policy='stop_competes_with_selector_candidates_except_no_search',
 bracketless_terminal='send_msg_to_user [text]')
BASE_ARTIFACTS={'result','watchdog','api','site_closure','closure'}
VALID_ARTIFACTS={'rollout','provenance','authentication','trajectory','terminal','verdict','evaluator_inputs','terminal_screenshot','lease','instance','copy_receipt','state_claim','native_result','native_context','native_bridge_config','native_bridge_events','native_raw_trajectory'}
def require(value,message):
 if not value:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def read(path,cap=128*2**20):
 path=canonical_runtime_path(str(path));require(path.is_file() and path.stat().st_size<=cap,'Missing/oversized offline evidence')
 return json.loads(path.read_text())
def ref(path):return dict(path=str(path),sha256=sha(path))
def reference(value,root=None):
 require(isinstance(value,dict) and set(value)=={'path','sha256'} and re.fullmatch('[a-f0-9]{64}',str(value.get('sha256',''))),'Exact immutable file reference required')
 p=canonical_runtime_path(value['path']);require(root is None or root in p.parents,'Artifact escaped durable campaign')
 require(p.is_file() and sha(p)==value['sha256'],'Immutable evidence changed');return p
def read_lines(path):
 p=canonical_runtime_path(str(path));require(p.stat().st_size<=512*2**20,'Oversized journal')
 rows=[]
 for line in p.open('rb'):
  require(line.endswith(b'\n'),'Incomplete journal tail');rows.append(json.loads(line))
 return rows
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def legacy_auditor(inputs):
 # v2: the v7 freeze holds several package source trees; bind the helper of the running package (ROOT).
 matches=[Path(p) for p in inputs['source_sha256'] if p.endswith('/scripts/audit_webprm_table5_cell.py') and Path(p).resolve().is_relative_to(ROOT.resolve())]
 require(len(matches)==1,'One source-bound baseline closure/reset/replay helper required')
 reference(ref(matches[0]));require(sha(matches[0])==inputs['source_sha256'][str(matches[0])],'Legacy helper source changed')
 return load('native57_frozen_closure_reset_helpers',matches[0])

def v9_site_reset(inputs,inner,paths,plan,result,legacy):
 site=inner['cell']['domain']
 if site not in MAGENTO:return legacy.audit_reddit_reset(inner,paths,plan,result)
 matches=[Path(p) for p in inputs['source_sha256'] if p.endswith('/scripts/audit_webprm_table5_cms_cell.py') and Path(p).resolve().is_relative_to(ROOT.resolve())]
 require(len(matches)==1 and sha(matches[0])==inputs['source_sha256'][str(matches[0])],'One source-bound Magento reset auditor required')
 magento=load('native_v9_frozen_magento_reset_helpers',matches[0])
 require(magento.site_module(site) is not None,'Magento reset auditor lacks this site')
 return magento.audit_cms_reset(inner,paths,plan,result)

def audit_identity(inputs,plan,cohort):
 cells=cohort['cells'];cell=inputs['cell'];indexed={c['cell_id']:c for c in cells}
 domains={c['domain'] for c in cells};require(len(domains)==1 and next(iter(domains)) in V9_SITE_TASKS,'One v9 Table 5 site per cohort');site=next(iter(domains))
 require(len(cells)==len(indexed)==3*V9_SITE_TASKS[site] and indexed.get(cell.get('cell_id'))==cell,'Fixed v9 cell membership differs')
 from openwebrl.webprm_webarena import TASK_FILE,verify_source
 require(verify_source(plan['arbiter_source'])==plan['source_manifest'],'Native task/evaluator source changed')
 pinned_taskref=dict(path=str(Path(plan['arbiter_source'])/TASK_FILE),sha256=plan['source_manifest']['files'][TASK_FILE])
 taskref=cohort.get('native_task_file') or plan.get('native_task_file') or pinned_taskref
 require(taskref==pinned_taskref,'Task file must be exact pinned original source')
 released=read(reference(taskref));single=[t for t in released if t.get('sites')==[site]]
 require(len(released)==165,'Original released165 membership differs')
 rawmap={t['task_id']:t for t in single if t['task_id'] not in DROPPED_TASKS};require(len(rawmap)==V9_SITE_TASKS[site],'v9 site task membership differs');selectors={'no_search','piotr_selectionarm'}
 require({c['task_id'] for c in cells}==set(rawmap) and {c.get('selector',c.get('method')) for c in cells}==selectors,'Prospective method/task grid differs')
 require(all(sum(c['task_id']==t and c.get('selector',c.get('method'))==m for c in cells)==1 for t in rawmap for m in selectors),'Repeated/missing prospective cells')
 actors={c['actor'] for c in cells};require(actors=={V7_ACTOR},'Prospective actor identity changed')
 for c in cells:
  raw=rawmap[c['task_id']]
  expected=hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  require(c['domain']==site and c['task_config_sha256']==expected,'Released task hash/domain differs')
  require('native_task_config' not in c or c['native_task_config']==raw,'Embedded task/config changed')
 require(plan.get('task_ids')==[cell['task_id']] and plan.get('actor_model')==cell['actor'] and plan.get('selector')==cell.get('selector',cell.get('method')),'Worker fixed identity differs')
 require(plan.get('tasks_sha256')==digest([rawmap[cell['task_id']]]) and plan.get('actors')==[cell['actor']] and plan.get('selectors')==[cell.get('selector',cell.get('method'))],'Actual worker task/config actor/method freeze differs')
 require(plan.get('protocol_contract')==plan.get('native_contract'),'Stable financial protocol contract must equal native scientific contract')
 require(plan.get('protocol')==cohort.get('protocol')==V9_PROTOCOL,'Prospective native protocol differs')
 require(plan.get('max_turns')==30 and plan.get('paper_exact_protocol') is False,'Prospective30turn disclosed protocol required')
 sources=inputs['source_sha256'];require(sources and str(Path(__file__).resolve()) in sources,'Independent auditor must be in frozen source map')
 for p,h in sources.items():require(sha(reference(dict(path=p,sha256=h)))==h,'Frozen source mismatch')
 for p,h in plan.get('code_sha256',{}).items():require(sources.get(p)==h,'Worker code/source freeze differs')
 for component in plan.get('native_components',{}).values():
  reference(component);require(sources.get(component['path'])==component['sha256'],'Native adapter component outside frozen source map')
 from openwebrl.webprm_webarena_episode import TaskEnvelope
 return TaskEnvelope.from_release(rawmap[cell['task_id']],plan['endpoints'])

def audit_context_contract(plan,result,context,cfg,native):
 from openwebrl.webprm_webarena_worker import ENVELOPE_NO_SEARCH_PROMPT
 from openwebrl.webprm_native_piotr_v11 import RECEIPT as V11_RECEIPT
 method=plan['selector'];actor=plan['actor_model']
 critic=plan.get('arbiter_model') if method=='arbiter7b' else plan.get('shepherd_model') if method=='shepherd8b' else None
 helper=dict(provider='openai',model='gpt-4o-mini',mode='chat',gen_config=dict(max_tokens=2048,context_length=8192,temperature=1.,top_p=.9))
 policy=dict(provider='openai',model=actor,mode='chat',gen_config=None)
 reward=dict(provider='openai',model=critic,mode='chat',gen_config=None) if critic else None
 expected_run=digest(plan)[:16]+f"-webarena_lite_{plan['task_ids'][0]:03d}-"+method
 site=next(iter(plan['endpoints']))
 require(context['run_id']==expected_run and context['domain']==site and context['domain_knowledge']==(native.history.load_domain_knowledge(site) or '') and context['policy_model']==policy and context['reward_model']==reward,'Original coordinator context or model identity/configuration changed')
 routes=[dict(provider='openai',model=actor,mode='chat',purpose='native_policy',allowed_gen_configs=[None]),dict(provider='openai',model='gpt-4o-mini',mode='chat',purpose='native_auxiliary',allowed_gen_configs=[helper['gen_config']])]
 if method=='arbiter7b':routes.append(dict(provider='openai',model=critic,mode='chat',purpose='native_reward',allowed_gen_configs=[None]))
 expected=dict(routes=routes,local_models=[critic] if method=='arbiter7b' else [],declared_default_model=helper,wire_model_aliases={'gpt-4o-mini':V8_AUX_WIRE},max_attempts=512,selector=method,no_search_prompt_version=ENVELOPE_NO_SEARCH_PROMPT if method=='no_search' else None,v8=V8_RECEIPT,v9=V9_RECEIPT if method in ('arbiter7b','shepherd8b') else None,v11=json.loads(json.dumps(V11_RECEIPT)) if method=='piotr_selectionarm' else None)
 require(cfg==expected,'Actual model routes/purposes/local-paid/default alias or single-action adaptation changed')
 return dict(verified=True,matched_method=method,original_domain_knowledge=True,explicit_auxiliary_alias=expected['wire_model_aliases'])

def audit_normalization(raw,normalized,native,terminal):
 require(raw and ((terminal is None and len(raw)%2==1) or (terminal is not None and len(raw)%2==0)),'Original terminal action must end the native trajectory without invented afterSTOP state')
 expected=[]
 for index,entry in enumerate(raw):
  if index%2==0:
   expected.append(dict(observation={k:v for k,v in entry['observation'].items() if k!='image'},info=dict(fail_error=entry['info'].get('fail_error',''),observation_metadata=deepcopy(entry['info']['observation_metadata']))))
  else:
   action=deepcopy(entry);kind=native.actions.ActionTypes(int(action['action_type']))
   action['action_type']='send_msg_to_user' if kind==native.actions.ActionTypes.SEND_MESSAGE_TO_USER else kind.name.lower();expected.append(action)
 require(expected==normalized,'Evaluator projection changed/omitted raw actions, state order, AX metadata or answer')
 if terminal is not None:
  require(expected[-1].get('action_type')=='send_msg_to_user' and expected[-1]['answer']==terminal['answer'],'Terminal answer was not actually executed')
 return dict(verified=True,actual_actions=len(raw)//2,actual_terminal=terminal is not None)

def audit_terminal_boundary(rawresult,actual_raw):
 if rawresult.get('terminal') is None:return
 require(len(actual_raw)>=2 and len(actual_raw)%2==0,'Terminal must end raw actions without a new browser state')
 execution=rawresult['selected_actions'][-1]['execution'];previous=actual_raw[-2]
 require(execution.get('terminal') is True and execution['action']==actual_raw[-1],'Actual terminal action binding differs')
 require(execution['observation']['receipt']==previous['state_observation_receipt'],'Terminal did not reuse the existing last browser observation')
 # SEND_MESSAGE_TO_USER performs no browser work. A preceding action failure
 # remains part of the reused observation; it is neither cleared nor relabelled
 # as a failure of the terminal meta-action. Full capture/source/dispatch proofs
 # are checked independently below and by whole original-coordinator replay.
 require(execution['observation'].get('fail_error','')==previous['info'].get('fail_error',''),'Terminal changed the inherited browser failure evidence')

def audit_native_bundle(inputs,paths,plan,result,replay_root,calls):
 rawresult=read(paths['native_result']);context=read(paths['native_context']);cfg=read(paths['native_bridge_config'])
 require(result.get('native_episode')==ref(paths['native_result']) and result.get('native_context')==ref(paths['native_context']) and result.get('native_bridge_config')==ref(paths['native_bridge_config']),'Wrapper/native component evidence refs differ')
 require(rawresult['valid'] is True and rawresult['verdict']==result['native_evaluation']==read(paths['verdict']),'Native/effective verdict changed across browser close')
 require(rawresult.get('terminal')==result.get('terminal') and rawresult['run_id']==context['run_id'],'Executed native terminal/context differs')
 require(context['intent']==read(Path(paths['native_result']).parent/'resolved-task.json')['intent'],'Actual coordinator task intent changed')
 for r in rawresult['native_sources']:
  require(set(r) in ({'path','sha256'},{'path','sha256','module'}),'Unknown native imported source reference schema')
  reference({k:r[k] for k in ('path','sha256')});require(inputs['source_sha256'].get(r['path'])==r['sha256'],'Native actual imported source missing from freeze')
 component=plan['native_components']['episode'];reference(component)
 native_module=load('_native57_original_episode_audited',Path(component['path']));native=native_module.load_native(plan['arbiter_source'])
 context_proof=audit_context_contract(plan,result,context,cfg,native)
 normalized=read(paths['trajectory']);raw=read(paths['native_raw_trajectory'])
 normalization=audit_normalization(raw,normalized,native,rawresult.get('terminal'))
 # Independently bind complete native adapter projection and lossless pixel evidence.
 actual_raw=read(reference(rawresult['raw_native_trajectory']))
 audit_terminal_boundary(rawresult,actual_raw)
 # Every transitive evidence reference must remain in retained campaign files
 # or the frozen source set, never solely in a disposable site/profile copy.
 allowed=set(paths.values())|{Path(p) for p in inputs['source_sha256']}
 def check_nested(value):
  if isinstance(value,dict):
   if set(value)=={'path','sha256'}:require(reference(value) in allowed,'Nested native evidence is outside retained artifact/source manifest')
   else:
    for item in value.values():check_nested(item)
  elif isinstance(value,list):
   for item in value:check_nested(item)
 native_roots=[paths['result'].parent/name for name in ('native-browser','native-episode','native-models')]
 native_named={paths[k] for k in ('native_result','native_context','native_bridge_config','native_raw_trajectory','trajectory','verdict')}
 for path in set(paths.values()):
  if path not in native_named and not any(root in path.parents for root in native_roots) and not path.name.startswith('selector-'):continue
  if path.suffix=='.json':check_nested(read(path))
  elif path.suffix=='.jsonl':
   for row in read_lines(path):check_nested(row)
 require(len(actual_raw)==len(raw),'Worker/raw native action-state count differs')
 from PIL import Image
 import numpy as np
 for index,(a,b) in enumerate(zip(actual_raw,raw)):
  if index%2:
   require(a==b,'Actual raw action dictionary changed in worker projection');continue
  require(a['observation']['text']==b['observation']['text'] and a['info']['fail_error']==b['info']['fail_error'] and a['info']['observation_metadata']==b['info']['observation_metadata'],'Actual native AX/info changed in worker projection')
  image=reference(a['observation']['image']);require(ref(image)==b['image'],'Worker/native screenshot ref changed')
  observed=read(reference(a['state_observation_receipt']));order=observed.get('obs_node_order')
  require(isinstance(order,list) and len(order)==len(set(order))==len(observed['obs_nodes_info']) and set(order)==set(observed['obs_nodes_info']),'Explicit captured native AX dictionary order required for production replay')
  pixels=a['observation']['image_pixels']
  with Image.open(image) as im:array=np.array(im)
  require(hashlib.sha256(array.tobytes()).hexdigest()==pixels['sha256'] and list(array.shape)==pixels['shape'] and str(array.dtype)==pixels['dtype'],'Lossless native image pixel evidence differs')
 helper=Path(__file__).with_name('native_lowlevel_replay.py');require(inputs['source_sha256'].get(str(helper))==sha(helper),'Independent lowlevel replay helper outside freeze')
 replay=load('_native_table5_v11_independent_lowlevel_replay_v8',helper)
 proof=replay.replay_bundle(plan=plan,paths=paths,calls=calls,replay_root=replay_root)
 return dict(proof,normalization=normalization,context_contract=context_proof)

def verify(inputs,replay_root):
 require(inputs.get('schema_version')==1,'Explicit audit input schema required')
 campaign=canonical_runtime_path(inputs['campaign_root']);cohort=read(reference(inputs['cohort']));cohort=cohort.get('cohort',cohort)
 plan=read(reference(inputs['worker_plan'],campaign));task=audit_identity(inputs,plan,cohort)
 refs=inputs['artifacts'];require(BASE_ARTIFACTS<=refs.keys(),'Incomplete closed attempt evidence')
 paths={k:reference(v,campaign) for k,v in refs.items()};fingerprints={str(p):(sha(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths.values()}
 result=read(paths['result']);cell=inputs['cell'];method=cell.get('selector',cell.get('method'))
 require(result.get('identity_sha256')==digest(plan) and result.get('actor_model')==cell['actor'] and result.get('selector')==method and result.get('task_id')==task.execution_id,'Result identity differs')
 require(type(result.get('valid')) is bool and type(result.get('attempted')) is bool,'Explicit raw/effective validity required')
 legacy=legacy_auditor(inputs);legacy.audit_closure(dict(inputs,cell=dict(cell,selector=method)),paths,result)
 broker,budget=inputs['broker'],inputs['budget'];require(broker['cell_id']==task.execution_id+'/'+cell['actor']+'__'+method,'Broker cell differs')
 require(budget['financial_contract']['cohort_sha256']==digest(cohort) and budget['financial_contract']['protocol_sha256']==cohort['protocol_sha256']==digest(plan['native_contract']),'Fixed cohort/native protocol financial binding differs')
 from openwebrl.webprm_journal_audit import audit_cell_calls
 calls=audit_cell_calls(plan,broker['cell_id'],paths['api'],broker['audit_path'],budget['event_path'],budget['prefix_bytes'],budget['prefix_sha256'],budget['financial_contract'],broker_prefix_bytes=broker['prefix_bytes'],broker_prefix_sha256=broker['prefix_sha256'])
 require(calls['verified'] and calls['policy_within_caps'],'Cumulative request/financial proof failed')
 status=read(reference(read(paths['closure'])['broker_status_reference']))
 require(calls.get('broker_policy_sha256s')==[status.get('policy_sha256')],'Broker closure and financial epoch policy differ')
 if not result['attempted']:
  require(not calls.get('api_request_count') and 'native_result' not in paths,'Unattempted worker contradicts model/native activity')
 native_replay=None;bundle=None;fuzzy_disposable=False;outcome=dict(success=None,native_success=None,guarded_success=None,native_valid=False,guarded_valid=False,infrastructure_invalid=True)
 if result['valid']:
  require(result['attempted'] and VALID_ARTIFACTS<=paths.keys(),'Valid native result lacks complete evidence')
  instance=v9_site_reset(inputs,dict(inputs,cell=dict(cell,selector=method)),paths,plan,result,legacy)
  bundle=audit_native_bundle(inputs,paths,plan,result,replay_root,calls)
  trajectory,verdict,terminal,provenance,auth=[read(paths[k]) for k in ('trajectory','verdict','terminal','provenance','authentication')]
  require(reference(terminal['image'])==paths['terminal_screenshot'] and terminal['index']==(len(read(paths['native_raw_trajectory']))-1)//2,'Actual terminal capture/image changed')
  require(auth.get('authenticated') is True and auth.get('state_provenance_sha256')==result.get('reset_id'),'Live auth/reset evidence differs')
  require(provenance.get('identity_sha256')==digest(plan) and provenance.get('original_instance')==instance and provenance.get('authentication')==auth,'Source/site/auth provenance differs')
  require(result['native_evaluation']==verdict and bool(result['success'])==bool(verdict['success']) and bool(result['native_success'])==bool(verdict.get('native_success',verdict['success'])),'Wrapper raw/effective native verdict differs')
  from PIL import Image
  with Image.open(paths['terminal_screenshot']) as im:im.verify()
  judge=None;expected=0
  from openwebrl.webprm_webarena import evaluation_requirements
  if evaluation_requirements(task.native_config())['paid_fuzzy_judge_may_be_required']:
   require('native_fuzzy' in paths,'Actual paid fuzzy judge journal missing');judge,expected=legacy.saved_fuzzy_proxy(plan,calls['pairs'],read_lines(paths['native_fuzzy']))
  from openwebrl.webprm_evaluator_transcript import replay_native_evaluation
  native_replay=replay_native_evaluation(plan['arbiter_source'],task.native_config(),trajectory,read(paths['evaluator_inputs']),verdict,fuzzy_judge=judge)
  require(native_replay['verified'] and (judge is None or judge.provider_calls==expected),'Native evaluator replay or paid judge coverage failed')
  native_success=int(verdict.get('native_success',verdict['success']));guarded_success=int(verdict['success'])
  outcome=dict(success=guarded_success,native_success=native_success,guarded_success=guarded_success,native_valid=True,guarded_valid=True,infrastructure_invalid=False,native_semantic_status='amended_by_explicit_guard' if verdict.get('scoring_amendments') else 'valid',guarded_semantic_status='valid')
 else:
  require(result.get('error_type') or result.get('error') or result.get('watchdog',{}).get('reason'),'Invalid outcome lacks explicit saved error')
  if result.get('attempted') and result.get('error_type')=='EvaluationNotReady' and result.get('error')==V9_FUZZY_ERROR:
   # v5: the retired gpt-4-1106-preview judge alone made this cell invalid. Its site state may be disposed once its
   # reset/instance chain verifies and the evaluator transcript (offline re-score input) is retained; still invalid.
   require({'evaluator_inputs','instance','lease','copy_receipt','state_claim','native_result'}<=paths.keys(),'Fuzzy-unavailable cell lacks its evaluator transcript or reset chain')
   v9_site_reset(inputs,dict(inputs,cell=dict(cell,selector=method)),paths,plan,result,legacy)
   from openwebrl.webprm_webarena import evaluation_requirements
   require(evaluation_requirements(task.native_config())['paid_fuzzy_judge_may_be_required'],'Fuzzy-unavailable error on a task without a fuzzy judge')
   fuzzy_disposable=True
 for p,h in inputs['source_sha256'].items():require(sha(p)==h,'Frozen implementation changed during offline audit')
 for p,(h,size,mtime) in fingerprints.items():
  s=Path(p);require((sha(s),s.stat().st_size,s.stat().st_mtime_ns)==(h,size,mtime),'Evidence changed during offline audit')
 episode=paths['result'].parent;covered=set(paths.values())
 require(all(p in covered for p in episode.rglob('*') if p.is_file() and p.name!='storage-state-private.json'),'Episode evidence omitted from manifest')
 record=dict(cell_id=cell['cell_id'],released_task_id=cell['task_id'],actor_model=cell['actor'],selector=method,domain=cell['domain'],task_config_sha256=cell['task_config_sha256'],tasks_sha256=cohort['native_task_file']['sha256'],cohort_sha256=digest(cohort),protocol_sha256=cohort['protocol_sha256'],audit_verified=True,terminal=True,attempted=result['attempted'],valid=result['valid'],**outcome)
 return dict(record=record,native_replay=native_replay,native_bundle_replay=bundle,call_audit={k:v for k,v in calls.items() if k not in ('pairs','broker_only_failures')},complete_for_disposal=bool((result['valid'] and native_replay and native_replay['verified'] and bundle and bundle['verified']) or fuzzy_disposable),
  fuzzy_judge_unavailable_state_disposable=fuzzy_disposable)

def write_once(path,value):
 p=canonical_runtime_path(str(path));fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'w') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def audit_cell(input_path,output_path):
 inputs=read(input_path);output=canonical_runtime_path(str(output_path));campaign=canonical_runtime_path(inputs['campaign_root'])
 require(campaign in output.parents,'Audit output must be durable campaign owned')
 episode=Path(inputs.get('artifacts',{}).get('result',{}).get('path','/')) .parent
 require(output.parent!=episode and episode not in output.parents,'Audit output must remain outside immutable episode')
 report=dict(schema_version=1,protocol=SCHEMA,input_sha256=sha(input_path),artifact_refs=inputs.get('artifacts',{}),evidence_manifest_sha256=digest(inputs.get('artifacts',{})),audit_verified=False,complete_for_disposal=False,errors=[],scope='Offline saved source/action/request/native evaluator consistency; no model calls, browser, historical process observation or judge semantic truth certification')
 try:
  with patch.object(socket.socket,'connect',side_effect=RuntimeError('Independent offline auditor forbids all network/socket connections')):
   report.update(verify(inputs,output.with_suffix('.replay')));report['audit_verified']=True
 except Exception as error:report['errors']=[dict(type=type(error).__name__,message=str(error)[:700])]
 write_once(output,report);normalized=None
 if report['audit_verified']:
  normalized=dict(report['record'],audit_sha256=sha(output));write_once(output.with_suffix('.record.json'),normalized)
 return dict(report,normalized_record=normalized,audit_path=str(output),audit_sha256=sha(output))
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit_cell(a.input,a.output);print(json.dumps({k:r[k] for k in ('audit_verified','complete_for_disposal','errors','audit_path','audit_sha256')}));return 0 if r['audit_verified'] else 2
if __name__=='__main__':raise SystemExit(main())
