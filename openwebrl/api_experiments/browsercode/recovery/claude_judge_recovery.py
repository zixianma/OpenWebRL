"""Judge-only recovery from immutable, complete terminal actor evidence."""
from bootstrap import ROOT,CONFIG
from pathlib import Path
import json,hashlib,time,fcntl
from common_judge import judge
from claude_budget import exclusive_json

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def admitted_approval():
 approval=read(ROOT/'orchestration/judge-amendment-approval.json')
 if not(approval.get('approved')is True and approval.get('model')==CONFIG['judge_protocol']['model']and approval.get('plan_sha256')==sha(ROOT/'campaign-plan.json')):raise ValueError('Exact judge amendment not admitted')
 review_path=ROOT/approval['original_canary_review']['path']
 if sha(review_path)!=approval['original_canary_review']['sha256']:raise ValueError('Original evidence review changed')
 return approval,read(review_path)

def recovery_evidence(episode,task):
 approval,review=admitted_approval();bound={x['path']:x['sha256']for x in review['evidence_refs']}
 episode=Path(episode).resolve()
 def checked(path):
  rel=str(path.relative_to(ROOT))
  if rel not in bound or sha(path)!=bound[rel]:raise ValueError('Original sealed actor evidence changed: '+rel)
 checked(episode/'result.json');row=read(episode/'result.json');image=episode/'final.png'
 if row.get('task_id')!=task['task_id']:raise ValueError('Saved task identity mismatch')
 if row.get('agent_result',{}).get('exit_status')not in('Submitted','LimitsExceeded'):return None
 if not image.is_file()or row.get('final_png_sha256')!=sha(image):return None
 if row.get('cleanup_errors')or not row.get('browser_cleanup',{}).get('verified'):return None
 usage=row.get('usage_accounting',{})
 if usage.get('metric_errors')or usage.get('model_excluding_judge',{}).get('unknown_usage_calls',0):return None
 if row.get('error_type')not in('NotFoundError','ValueError')and not row.get('valid'):return None
 if row.get('error_type')=='ValueError'and row.get('error')!='Budget halted; explicit diagnosis required':return None
 decisions=sorted((episode/'decisions').glob('*.json'))
 if len(decisions)!=row.get('decisions'):raise ValueError('Saved decision count changed')
 refs=[episode/'result.json',image,episode/'trajectory.json']+decisions
 for path in refs:checked(path)
 from environments import verify_png
 verify_png(image)
 return {'original_row':row,'image':image,'actions':[json.dumps(read(p)['selected_action'])for p in decisions],
         'evidence_refs':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)}for p in refs]}

def recover_prior_judge(episode,task,client,budget):
 episode=Path(episode).resolve();version=CONFIG['judge_protocol']['version'];approval,_=admitted_approval()
 evidence=recovery_evidence(episode,task)
 if evidence is None:return {'eligible':False,'recovered':False,'reason':'No complete verified terminal actor evidence; original invalid preserved'}
 out=episode/'judge-recovery'/version
 if out.exists():
  verdict=out/'verdict.json'
  if not verdict.is_file():raise RuntimeError('Interrupted judge recovery preserved; no unchanged retry')
  data=read(verdict)
  if data.get('valid')is not True:raise RuntimeError('Prior judge recovery invalid; no unchanged retry')
  if not(data.get('task_id')==task['task_id']and data.get('judge_protocol')==CONFIG['judge_protocol']and data.get('amendment_approval_sha256')==sha(ROOT/'orchestration/judge-amendment-approval.json')and type(data.get('reward'))is int and data['reward']in(0,1)and data.get('actor_rerun')is False):raise ValueError('Cached judge recovery identity/protocol/approval mismatch')
  for ref in data['evidence_refs']:
   if sha(ROOT/ref['path'])!=ref['sha256']:raise RuntimeError('Recovered actor evidence changed')
  return data
 out.mkdir(parents=True,exist_ok=False);started=time.monotonic();row={'eligible':True,'recovered':False,'valid':False,'reward':None,'task_id':task['task_id'],
  'judge_protocol':CONFIG['judge_protocol'],'evidence_refs':evidence['evidence_refs'],'source_manifest_sha256':sha(ROOT/'source-manifest.json'),
  'plan_sha256':sha(ROOT/'campaign-plan.json'),'amendment_approval_sha256':sha(ROOT/'orchestration/judge-amendment-approval.json'),'actor_rerun':False,'native_latency_seconds_preserved':evidence['original_row']['task_latency_seconds'],'original_result_unchanged':True}
 try:
  row['reward']=judge(client,budget,task['intent'],evidence['actions'],evidence['image'],out);row.update(valid=True,recovered=True)
 except BaseException as exc:
  row.update(error_type=type(exc).__name__,error=str(exc));raise
 finally:
  row.update(epoch=time.time(),judge_seconds=time.monotonic()-started);exclusive_json(out/'verdict.json',row)
 return row

def effective_outcome(result_path):
 p=Path(result_path);row=read(p);v=p.parent/'judge-recovery'/CONFIG['judge_protocol']['version']/'verdict.json'
 if row.get('judge_protocol_version')==CONFIG['judge_protocol']['version']:return row
 if not v.exists():return {**row,'valid':False,'reward':None,'effective_protocol_missing':True}
 d=read(v)
 for ref in d['evidence_refs']:
  if sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Effective verdict source changed')
 if not(d.get('task_id')==row['task_id']and d.get('judge_protocol')==CONFIG['judge_protocol']and d.get('amendment_approval_sha256')==sha(ROOT/'orchestration/judge-amendment-approval.json')):raise ValueError('Effective verdict identity/protocol/approval mismatch')
 return {**row,'original_error_type':row.get('error_type'),'error_type':None if d['valid']else d.get('error_type'),'error':None if d['valid']else d.get('error'),'valid':d['valid'],'reward':d['reward'],'effective_judge_verdict':str(v),'judge_protocol_version':CONFIG['judge_protocol']['version']}

def qualify_saved_judge(logical,shard_config,tasks,client,budget):
 """Both workers wait until saved terminal evidence proves live Terra judging."""
 if shard_config['shard']!=0:return
 logical=Path(logical);version=CONFIG['judge_protocol']['version']
 with(logical/('judge-qualification-'+version+'.lock')).open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  recovered=[]
  for tid in shard_config['original_canary_task_ids']:
   paths=sorted(logical.glob('attempts/attempt-*/episodes/'+tid+'-browsercode-'+shard_config['arm']))
   if not paths:raise ValueError('Expected archived canary evidence missing')
   row=recover_prior_judge(paths[-1],tasks[tid],client,budget)
   if row.get('valid')is True:recovered.append(tid)
  if not recovered:raise ValueError('Live Terra saved-evidence qualification failed; no fresh actor requests')
  marker=logical/('judge-qualification-'+version+'.json')
  if not marker.exists():exclusive_json(marker,{'epoch':time.time(),'verified_live_judge':True,'model':CONFIG['judge_protocol']['model'],'plan_sha256':sha(ROOT/'campaign-plan.json'),'recovered_task_ids':recovered,'actor_calls':0})
