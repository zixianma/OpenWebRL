#!/usr/bin/env python3
"""Independent offline audit of one v12 cell (OpenWebRL-4B-SFT, no_search or + Piotr SelectionARM, hosted WebArena).

Copy of native-audit-v9 (which stays unchanged) with the v12 cohort identity and agent evidence. Kept: closure,
broker/financial journal, site reset chain and native-evaluator transcript replay (the released evaluator re-run on the
saved trajectory against the recorded page/helper observations). Replaced: the gpt-4o coordinator bundle replay by
OpenWebRL evidence checks (selection traces, executed steps, terminal answer projection, screenshots). No browser,
provider, allocation or network call; evidence consistency only, never judge semantic truth.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,re,socket,sys
from unittest.mock import patch
from pathlib import Path
ROOT=Path(os.environ.get('WEBPRM_NATIVE_SOURCE_ROOT','/nonexistent-v12-package'))
sys.path.insert(0,str(ROOT))
from openwebrl.webprm_webarena import digest
from openwebrl.webprm_webarena_sites import canonical_runtime_path
SCHEMA='native_table5_v12_owrl_independent_cell_audit_v1'
V12_PROTOCOL='native_table5_v12_owrl'
V12_ACTOR='openwebrl-sft-4b'
SITE_TASKS={'shopping':45,'shopping_admin':35,'reddit':18,'gitlab':30}
FUZZY_ERROR='Native fuzzy transport/accounting failed; no retry'
MAGENTO={'shopping','shopping_admin','gitlab'}
DROPPED_TASKS={5}
CANDIDATES={'no_search':1,'piotr_selectionarm':5}
KIND={'no_search':'baseline','piotr_selectionarm':'selection'}
BASE_ARTIFACTS={'result','watchdog','api','site_closure','closure'}
VALID_ARTIFACTS={'trajectory','verdict','evaluator_inputs','lease','instance','copy_receipt','state_claim','provenance'}
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
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def source_helper(inputs,suffix):
 matches=[Path(p) for p in inputs['source_sha256'] if p.endswith(suffix) and Path(p).resolve().is_relative_to(ROOT.resolve())]
 require(len(matches)==1 and sha(matches[0])==inputs['source_sha256'][str(matches[0])],'One source-bound helper required: '+suffix)
 return matches[0]
def legacy_auditor(inputs):return load('v12_frozen_closure_reset_helpers',source_helper(inputs,'/scripts/audit_webprm_table5_cell.py'))
def site_reset(inputs,paths,plan,result,legacy):
 site=inputs['cell']['domain']
 if site not in MAGENTO:return legacy.audit_reddit_reset(inputs,paths,plan,result)
 magento=load('v12_frozen_magento_reset_helpers',source_helper(inputs,'/scripts/audit_webprm_table5_cms_cell.py'))
 require(magento.site_module(site) is not None,'Magento reset auditor lacks this site')
 return magento.audit_cms_reset(inputs,paths,plan,result)

def audit_identity(inputs,plan,cohort):
 cells=cohort['cells'];cell=inputs['cell'];indexed={c['cell_id']:c for c in cells}
 domains={c['domain'] for c in cells};require(len(domains)==1 and next(iter(domains)) in SITE_TASKS,'One Table 5 site per cohort');site=next(iter(domains))
 require(len(cells)==len(indexed)==len(cohort['selectors'])*SITE_TASKS[site] and indexed.get(cell.get('cell_id'))==cell,'Fixed v12 cell membership differs')
 from openwebrl.webprm_webarena import TASK_FILE,verify_source
 require(verify_source(plan['arbiter_source'])==plan['source_manifest'],'Native task/evaluator source changed')
 taskref=dict(path=str(Path(plan['arbiter_source'])/TASK_FILE),sha256=plan['source_manifest']['files'][TASK_FILE])
 released=read(reference(taskref));single=[t for t in released if t.get('sites')==[site]]
 require(len(released)==165,'Original released165 membership differs')
 rawmap={t['task_id']:t for t in single if t['task_id'] not in DROPPED_TASKS};require(len(rawmap)==SITE_TASKS[site],'Site task membership differs')
 selectors={'no_search','piotr_selectionarm'}
 require({c['task_id'] for c in cells}==set(rawmap) and {c['selector'] for c in cells}==selectors,'Method/task grid differs')
 require(all(sum(c['task_id']==t and c['selector']==m for c in cells)==1 for t in rawmap for m in selectors),'Repeated/missing cells')
 require({c['actor'] for c in cells}=={V12_ACTOR},'Actor identity changed')
 for c in cells:
  expected=hashlib.sha256(json.dumps(rawmap[c['task_id']],sort_keys=True,separators=(',',':')).encode()).hexdigest()
  require(c['domain']==site and c['task_config_sha256']==expected,'Released task hash/domain differs')
 require(plan.get('task_ids')==[cell['task_id']] and plan.get('actor_model')==cell['actor'] and plan.get('selector')==cell['selector'],'Worker fixed identity differs')
 require(plan.get('tasks_sha256')==digest([rawmap[cell['task_id']]]) and plan.get('actors')==[cell['actor']] and plan.get('selectors')==[cell['selector']],'Worker task/actor/method freeze differs')
 require(plan.get('protocol_contract')==plan.get('native_contract'),'Financial protocol contract must equal the scientific contract')
 require(plan.get('protocol')==cohort.get('protocol')==V12_PROTOCOL,'v12 protocol differs')
 require(plan.get('max_turns')==30 and plan.get('paper_exact_protocol') is False,'30-step disclosed protocol required')
 sources=inputs['source_sha256'];require(sources and str(Path(__file__).resolve()) in sources,'Independent auditor must be in frozen source map')
 for p,h in sources.items():require(sha(reference(dict(path=p,sha256=h)))==h,'Frozen source mismatch')
 for p,h in plan.get('code_sha256',{}).items():require(sources.get(p)==h,'Worker code/source freeze differs')
 from openwebrl.webprm_webarena_episode import TaskEnvelope
 return TaskEnvelope.from_release(rawmap[cell['task_id']],plan['endpoints'])

def audit_owrl(paths,episode,plan,result,trajectory,method):
 """OpenWebRL evidence: selection traces, executed steps, terminal projection and screenshots agree with result."""
 steps=read(episode/'owrl-steps.json')
 traces=sorted((episode/'selections').glob('*.jsonl'))
 rows=[json.loads(line) for p in traces for line in p.read_text().splitlines() if line]
 require(rows,'No OpenWebRL decision trace')
 require(all(r['kind']==KIND[method] and len(r['candidates'])==CANDIDATES[method] and 0<=r['selected_index']<len(r['candidates']) for r in rows),'Decision trace kind/candidate count/pick differs from the method')
 require([r['turn'] for r in rows]==list(range(len(rows))),'Decision turns are not consecutive from 0')
 require(result.get('selector_picks')==[r['selected_index'] for r in rows],'Result picks differ from decision traces')
 if method=='no_search':require(all(r['selector_raw'] is None for r in rows),'no_search trace carries a selector reply')
 require(len(steps['steps'])==result.get('executed_actions')<=len(rows),'Executed browser steps exceed decisions or differ from result')
 done=steps['ending']=='done'
 terminal=trajectory[-1]
 require(terminal.get('action_type')=='send_msg_to_user' and terminal.get('answer')==(steps['answer'] if done else 'N/A'),'Terminal answer projection differs from the executed done call / N/A fallback')
 require(result.get('terminal')==dict(answer=terminal['answer'],source=terminal['owrl_terminal']),'Result terminal differs')
 require((len(trajectory)-2)//2<=len(steps['steps']),'Trajectory has more browser actions than executed steps')
 shots=sorted((episode/'owrl-browser').glob('observation-*.png'))
 require(shots and len(shots)==1+sum(bool(s.get('screenshot')) for s in steps['steps']),'Screenshot evidence count differs')
 from PIL import Image
 with Image.open(shots[-1]) as im:
  im.verify()
 with Image.open(shots[-1]) as im:
  require(im.size==(1280,1000),'Terminal screenshot is not the 1280x1000 viewport')
 return dict(verified=True,decisions=len(rows),executed_steps=len(steps['steps']),ending=steps['ending'],
             fallbacks=sum(bool(r.get('fallback')) for r in rows),screenshots=len(shots))

def verify(inputs):
 require(inputs.get('schema_version')==1,'Explicit audit input schema required')
 campaign=canonical_runtime_path(inputs['campaign_root']);cohort=read(reference(inputs['cohort']));cohort=cohort.get('cohort',cohort)
 plan=read(reference(inputs['worker_plan'],campaign));task=audit_identity(inputs,plan,cohort)
 refs=inputs['artifacts'];require(BASE_ARTIFACTS<=refs.keys(),'Incomplete closed attempt evidence')
 paths={k:reference(v,campaign) for k,v in refs.items()};fingerprints={str(p):(sha(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths.values()}
 result=read(paths['result']);cell=inputs['cell'];method=cell['selector']
 require(result.get('identity_sha256')==digest(plan) and result.get('actor_model')==cell['actor'] and result.get('selector')==method and result.get('task_id')==task.execution_id,'Result identity differs')
 require(type(result.get('valid')) is bool and type(result.get('attempted')) is bool,'Explicit validity required')
 legacy=legacy_auditor(inputs);legacy.audit_closure(inputs,paths,result)
 broker,budget=inputs['broker'],inputs['budget'];require(broker['cell_id']==task.execution_id+'/'+cell['actor']+'__'+method,'Broker cell differs')
 require(budget['financial_contract']['cohort_sha256']==digest(cohort) and budget['financial_contract']['protocol_sha256']==cohort['protocol_sha256']==digest(plan['native_contract']),'Cohort/protocol financial binding differs')
 from openwebrl.webprm_journal_audit import audit_cell_calls
 calls=audit_cell_calls(plan,broker['cell_id'],paths['api'],broker['audit_path'],budget['event_path'],budget['prefix_bytes'],budget['prefix_sha256'],budget['financial_contract'],broker_prefix_bytes=broker['prefix_bytes'],broker_prefix_sha256=broker['prefix_sha256'])
 require(calls['verified'] and calls['policy_within_caps'],'Cumulative request/financial proof failed')
 episode=paths['result'].parent
 owrl=None;native_replay=None;fuzzy_disposable=False
 outcome=dict(success=None,native_success=None,native_valid=False,infrastructure_invalid=True)
 if result['valid']:
  require(result['attempted'] and VALID_ARTIFACTS<=paths.keys(),'Valid result lacks complete evidence')
  site_reset(inputs,paths,plan,result,legacy)
  trajectory,verdict=read(paths['trajectory']),read(paths['verdict'])
  require(result['native_evaluation']==verdict and bool(result['success'])==bool(verdict['success']),'Wrapper verdict differs')
  require(result.get('closure',{}).get('closed') is True and not result['closure'].get('errors'),'Browser closure incomplete')
  owrl=audit_owrl(paths,episode,plan,result,trajectory,method)
  from openwebrl.webprm_webarena import evaluation_requirements
  require(not evaluation_requirements(task.native_config())['paid_fuzzy_judge_may_be_required'],'A valid fuzzy-judge verdict is impossible with the retired judge')
  from openwebrl.webprm_evaluator_transcript import replay_native_evaluation
  native_replay=replay_native_evaluation(plan['arbiter_source'],task.native_config(),trajectory,read(paths['evaluator_inputs']),verdict)
  require(native_replay['verified'],'Native evaluator replay failed')
  outcome=dict(success=int(verdict['success']),native_success=int(verdict.get('native_success',verdict['success'])),native_valid=True,infrastructure_invalid=False)
 else:
  require(result.get('error_type') or result.get('error') or result.get('watchdog',{}).get('reason'),'Invalid outcome lacks explicit saved error')
  if result.get('attempted') and result.get('error_type')=='EvaluationNotReady' and result.get('error')==FUZZY_ERROR:
   require({'evaluator_inputs','instance','lease','copy_receipt','state_claim'}<=paths.keys(),'Fuzzy-unavailable cell lacks its evaluator transcript or reset chain')
   site_reset(inputs,paths,plan,result,legacy)
   from openwebrl.webprm_webarena import evaluation_requirements
   require(evaluation_requirements(task.native_config())['paid_fuzzy_judge_may_be_required'],'Fuzzy-unavailable error on a task without a fuzzy judge')
   owrl=audit_owrl(paths,episode,plan,result,read(paths['trajectory']),method)
   fuzzy_disposable=True
 for p,h in inputs['source_sha256'].items():require(sha(p)==h,'Frozen implementation changed during offline audit')
 for p,(h,size,mtime) in fingerprints.items():
  s=Path(p);require((sha(s),s.stat().st_size,s.stat().st_mtime_ns)==(h,size,mtime),'Evidence changed during offline audit')
 covered=set(paths.values())
 require(all(p in covered for p in episode.rglob('*') if p.is_file() and p.name!='storage-state-private.json'),'Episode evidence omitted from manifest')
 record=dict(cell_id=cell['cell_id'],released_task_id=cell['task_id'],actor_model=cell['actor'],selector=method,domain=cell['domain'],task_config_sha256=cell['task_config_sha256'],cohort_sha256=digest(cohort),protocol_sha256=cohort['protocol_sha256'],audit_verified=True,attempted=result['attempted'],valid=result['valid'],owrl_ending=result.get('owrl_ending'),**outcome)
 return dict(record=record,native_replay=native_replay,owrl_evidence=owrl,call_audit={k:v for k,v in calls.items() if k not in ('pairs','broker_only_failures')},
  complete_for_disposal=bool((result['valid'] and native_replay and native_replay['verified'] and owrl and owrl['verified']) or fuzzy_disposable),
  fuzzy_judge_unavailable_state_disposable=fuzzy_disposable)

def write_once(path,value):
 p=canonical_runtime_path(str(path));fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'w') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def audit_cell(input_path,output_path):
 inputs=read(input_path);output=canonical_runtime_path(str(output_path));campaign=canonical_runtime_path(inputs['campaign_root'])
 require(campaign in output.parents,'Audit output must be durable campaign owned')
 episode=Path(inputs.get('artifacts',{}).get('result',{}).get('path','/')).parent
 require(output.parent!=episode and episode not in output.parents,'Audit output must remain outside the immutable episode')
 report=dict(schema_version=1,protocol=SCHEMA,input_sha256=sha(input_path),artifact_refs=inputs.get('artifacts',{}),evidence_manifest_sha256=digest(inputs.get('artifacts',{})),audit_verified=False,complete_for_disposal=False,errors=[],scope='Offline saved source/request/OpenWebRL trace/native evaluator consistency; no model calls, browser or judge semantic truth certification')
 try:
  with patch.object(socket.socket,'connect',side_effect=RuntimeError('Independent offline auditor forbids all network/socket connections')):
   report.update(verify(inputs));report['audit_verified']=True
 except Exception as error:report['errors']=[dict(type=type(error).__name__,message=str(error)[:700])]
 write_once(output,report);normalized=None
 if report['audit_verified']:
  normalized=dict(report['record'],audit_sha256=sha(output));write_once(output.with_suffix('.record.json'),normalized)
 return dict(report,normalized_record=normalized,audit_path=str(output),audit_sha256=sha(output))
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit_cell(a.input,a.output);print(json.dumps({k:r[k] for k in ('audit_verified','complete_for_disposal','errors','audit_path','audit_sha256')}));return 0 if r['audit_verified'] else 2
if __name__=='__main__':raise SystemExit(main())
