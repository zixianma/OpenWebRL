#!/usr/bin/env python3
"""Independent read-only audit and private review of the local four-condition suite.

Use the existing runtime venv for the actual SGLang tool parser. No browser,
model client, GPU inference, ledger write, or completion-pointer update occurs.
Baseline refreshes require ijson for bounded-memory parsing of private reports.
"""
from __future__ import annotations
import argparse
import ast
import base64
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import sys
from types import SimpleNamespace
from typing import Any

from audit_controlled_jev_actor import png_dimensions,scheduler_records

ARMS={'L01':None,'L08':'luna','L10':'jev','L11':'kev'}
LABELS={'L01':'SFT alone','L08':'SFT + Luna','L10':'SFT + Jev','L11':'SFT + Kev27B'}
MODELS={'luna':'gpt-6-luna','jev':'jev-1.13.0','kev':'kev-latest'}
KIND={'luna':'luna_selector_requests','jev':'jev_selector_requests','kev':'local_kev_requests'}
JUDGE_SHA='b1f2c30c852b36d8049e23bd8ef4efb44a0786742449b5e7ef4e5a7ea55af583'
JUDGE='o4-mini-2025-04-16'
SAMPLING=dict(temperature=1.,top_p=.95,top_k=-1,max_new_tokens=4096,repetition_penalty=1.)
RATES={'luna_selector_requests':('luna_usd',.125,.5),'om2w_judge_http_attempts':('canonical_judge_usd',1.1,4.4)}
CAP_KEY={'om2w_judge_http_attempts':'canonical_judge_http_attempts'}
ACCEPTANCE_PURPOSE='serving_cap_saved_context_acceptance'


def read(p):return json.loads(Path(p).read_text())
def read_baseline(p):
    # Only the immutable proofs participate in a refresh. Do not retain a second
    # complete copy of every historical proposal/history alongside the new audit.
    import ijson
    with Path(p).open('rb') as stream:
        return {'immutable_artifacts':dict(ijson.kvitems(stream,'immutable_artifacts'))}
def raw_sha(raw):return hashlib.sha256(raw).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def digest(v):return raw_sha(json.dumps(v,sort_keys=True,separators=(',',':')).encode())
def state_digest(v):return raw_sha(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode())
def seed(task,turn,index):return int.from_bytes(hashlib.sha256(f'20261006:{task}:{turn}:{index}'.encode()).digest()[:4],'big')%(2**31-1)
def stamp():return datetime.now(timezone.utc).isoformat()


def snapshot(root):
    with sqlite3.connect((root/'state.sqlite3').as_uri()+'?mode=ro',uri=True,timeout=30) as db:
        db.row_factory=sqlite3.Row;db.execute('BEGIN')
        tables={t:[dict(row) for row in db.execute('SELECT * FROM '+t)] for t in ('queue','attempts','reservations','counters','metadata','events')}
        db.rollback()
    return tables


class Checks:
    def __init__(self):self.issues=[];self.pending=[];self.warnings=[];self.artifacts={};self.controls={}
    def need(self,ok,code,**context):
        if not ok:self.issues.append(dict(code=code,**context))
    def preserve(self,path,expected=None,*,immutable=True):
        p=Path(path).resolve();key=str(p)
        actual=self.artifacts.get(key,self.controls.get(key))
        if actual is None:
            if not p.is_file():self.need(False,'missing_artifact',path=str(p));return None
            actual=sha(p)
        if expected:self.need(actual==expected,'artifact_hash_mismatch',path=str(p))
        (self.artifacts if immutable else self.controls)[key]=actual
        return actual
    def events(self,directory):
        result=[]
        for p in sorted(Path(directory).glob('[0-9]*.json')):
            raw=p.read_bytes();self.artifacts[str(p.resolve())]=raw_sha(raw)
            result.append((p,json.loads(raw)))
        return result


def formatter(source,tools):
    """Run only the frozen parser and canonical pure formatting definitions.

    Avoid importing the canonical reward module's model clients and rollout
    stack. Require its real SGLang parser, never silently use a substitute.
    """
    namespace=dict(json=json,re=re,Any=Any,ToolCall=SimpleNamespace,ParsedToolResult=SimpleNamespace)
    parser_ast=ast.parse((source/'openwebrl/base/utils.py').read_text())
    cls=next(n for n in parser_ast.body if isinstance(n,ast.ClassDef) and n.name=='ToolParser')
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source/'openwebrl/base/utils.py'),'exec'),namespace)
    parser=namespace['ToolParser'](tools_info=tools,parser_type='qwen25')
    if parser._get_sglang_parser() is None:raise RuntimeError('Actual SGLang parser unavailable; use the runtime venv')
    path=source/'openwebrl/eval/reward_online_mind2web.py'
    if sha(path)!=JUDGE_SHA:raise ValueError('Canonical judge source changed')
    names={'SYSTEM_PROMPT','USER_PROMPT','_THINK_PATTERN','_extract_thought','_format_action','_extract_thoughts_and_actions','_format_trajectory'}
    selected=[]
    for node in ast.parse(path.read_text()).body:
        name=node.name if isinstance(node,ast.FunctionDef) else node.targets[0].id if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) else None
        if name in names:selected.append(node)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    def expected(intent,messages,image):
        thoughts,actions=namespace['_extract_thoughts_and_actions'](messages,parser)
        text=namespace['USER_PROMPT'].format(task=intent,thoughts_and_actions=namespace['_format_trajectory'](thoughts,actions))
        return [dict(role='system',content=namespace['SYSTEM_PROMPT']),dict(role='user',content=[dict(type='text',text=text),
            dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(image).decode(),detail='high'))])]
    return expected,parser


def normalized_history(text):
    text=text.replace('<|im_end|>','').strip()
    if text.startswith('<think>'):text=text[len('<think>'):].lstrip()
    return text


def expected_selector_state(capture,task,outputs,order,operations):
    page=capture['page'];count=capture['action_attempts']
    return dict(task=task,page=dict(url=page['url'],title=page['title'],text=page['text'][:16000],
        text_characters=page.get('text_characters',len(page['text'])),text_truncated=page.get('text_characters',len(page['text']))>16000,
        interactive_elements=page['interactive_elements'],tabs=page['tabs'],screen_size=page['screen_size']),
        recent_executed_actions=[op for op in operations if op['attempt']<=count][-5:],
        candidates={str(i+1):dict(text=outputs[j]) for i,j in enumerate(order)})


def selector_index(response,provider):
    if provider=='luna':
        texts=[part['text'] for item in response.get('output',[]) if item.get('type')=='message' for part in item.get('content',[]) if part.get('type')=='output_text']
        if response.get('status')!='completed' or len(texts)!=1:raise ValueError('Luna selection missing')
        value=json.loads(texts[0]);index=value.get('selected_index')
        if set(value)!={'selected_index'} or type(index) is not int:raise ValueError('Luna schema changed')
    else:
        if response.get('truncated'):raise ValueError('Truncated classifier input')
        answer=response['answers']['selection'];probs=answer['probabilities'];index=int(answer['choice'])
        if set(probs)!={str(i) for i in range(1,6)}:raise ValueError('Classifier labels changed')
        if probs[str(index)]+1e-9<max(probs.values()):raise ValueError('Native choice rejected')
    if not 1<=index<=5:raise ValueError('Selection outside five candidates')
    return index


def check_coverage(plans,queues,checks):
    if len(plans)!=2:checks.need(False,'two_shards_required');return
    task_sets=[set(p['shard']['task_ids']) for p in plans]
    checks.need(all(len(s)==150 for s in task_sets) and not task_sets[0]&task_sets[1] and len(task_sets[0]|task_sets[1])==300,'disjoint_complete300')
    for i,(p,queue) in enumerate(zip(plans,queues)):
        expected=Counter((x['item_id'],x['task_id'],x['condition'],x['category']) for x in p['schedule'])
        actual=Counter((x['item_id'],x['task_id'],x['condition'],x['category']) for x in queue)
        checks.need(expected==actual and len(expected)==612,'queue_identity',shard=i)
        checks.need(Counter((x['task_id'],x['condition']) for x in p['schedule'] if x['category']=='primary')==Counter({(t,a):1 for t in task_sets[i] for a in ARMS}),'all_conditions_per_task',shard=i)
        smokes=[x for x in p['schedule'] if x['category']=='smoke'];smoke_tasks={x['task_id'] for x in smokes}
        checks.need(len(smoke_tasks)==3 and smoke_tasks<=task_sets[i] and Counter((x['task_id'],x['condition']) for x in smokes)==Counter({(t,a):1 for t in smoke_tasks for a in ARMS}),'matched_smoke_coverage',shard=i)


def check_budget(db,plan,checks):
    counts=Counter();charges=Counter();per_arm=Counter();rows={r['reservation_id']:r for r in db['reservations']};open_ids=set();peak=0
    for event in sorted(db['events'],key=lambda e:e['id']):
        d=json.loads(event['details_json'])
        if event['event']=='reserve' and d['kind']=='browser_sessions':open_ids.add(d['reservation_id'])
        elif event['event']=='settle':open_ids.discard(d['reservation_id'])
        elif event['event']=='owner_teardown':open_ids={rid for rid in open_ids if rows[rid]['attempt_id'] not in d['attempt_ids']}
        peak=max(peak,len(open_ids));checks.need(len(open_ids)<=plan['limits']['concurrent_browsers'],'historical_browser_concurrency')
    for r in rows.values():
        kind=r['kind'];counts[kind]+=r['count']
        if kind=='browser_sessions':per_arm[r['condition']]+=r['count']
        if kind in RATES:
            cap,ip,op=RATES[kind];charges[cap]+=r['charged_usd'];context=json.loads(r['context_json'])
            if kind=='luna_selector_requests' and context['input_token_bound']>272000:ip*=2;op*=1.5
            if r['status']=='settled':
                usage=json.loads(r['usage_json']);it=usage.get('prompt_tokens',usage.get('input_tokens'));ot=usage.get('completion_tokens',usage.get('output_tokens'))
                ok=type(it) is int and type(ot) is int and 0<=it<=context['input_token_bound'] and 0<=ot<=context['output_token_bound']
                checks.need(ok,'priced_usage_bounds',reservation=r['reservation_id'])
                if ok:checks.need(math.isclose(r['charged_usd'],(ip*it+op*ot)/1e6,abs_tol=1e-10),'priced_usage_charge',reservation=r['reservation_id'])
            elif r['status']=='charged_unknown':checks.need(r['charged_usd']==r['reserved_usd'],'unknown_charge_not_preserved',reservation=r['reservation_id'])
    counters={r['key']:r['value'] for r in db['counters']}
    for key in ('local_sft_generations','browser_sessions',*KIND.values(),'om2w_judge_http_attempts',*(v[0] for v in RATES.values())):
        amount=charges.get(key,0) if key.endswith('_usd') else counts.get(key,0)
        checks.need(math.isclose(counters.get(key,0),amount,abs_tol=1e-8),'counter_drift',counter=key)
        checks.need(amount<=plan['limits'][CAP_KEY.get(key,key)]+1e-8,'budget_exceeded',counter=key)
    for arm in ARMS:
        checks.need(per_arm[arm]==counters.get('browsers:'+arm,0) and per_arm[arm]<=plan['limits']['browser_sessions_per_condition'],'per_condition_browser_budget',condition=arm)
    active=sum(r['kind']=='browser_sessions' and r['status']=='reserved' for r in rows.values())
    checks.need(len(open_ids)==active==counters.get('active_browsers',0),'active_browser_drift')
    unsettled=sum(r['status']=='reserved' for r in rows.values())
    if unsettled:checks.pending.append(f"shard{plan['shard']['index']}:unsettled_reservations={unsettled}")
    return dict(counters=counters,reservation_counts=dict(counts),charges=dict(charges),peak_browsers=peak,limits=plan['limits'],unknown_charges=sum(r['status']=='charged_unknown' for r in rows.values()))


def check_amendment(descriptor,checks):
    """One bounded native-choice reclassification; unchanged original evidence remains."""
    aid=descriptor['attempt_id'];amend=descriptor['administrative_amendment']
    for name in ('original_result','response','final_screenshot'):
        proof=amend[name];checks.preserve(proof['path'],proof['sha256'])
    original=read(amend['original_result']['path']);response=read(amend['response']['path'])
    answer=response['record']['response']['answers']['selection'];prob=answer['probabilities']
    checks.need(prob[answer['choice']]<max(prob.values()),'amendment_native_guard_evidence',attempt_id=aid)
    checks.need(amend['model_and_judge_calls_added']==amend['browser_actions_added']==0 and amend['episode_replayed'] is False,'amendment_no_replay',attempt_id=aid)
    checks.need(original['status']==descriptor['status']=='Status.ABORTED' and original['reward']==descriptor['reward']==0 and original['final_screenshot']==descriptor['final_screenshot'],'amendment_original_outcome',attempt_id=aid)
    for key in ('actor_decisions','browser_operations','elapsed_seconds','task_id','condition','browser_closed'):
        checks.need(original[key]==descriptor[key],'amendment_actor_unchanged',attempt_id=aid,field=key)
    return response['record']['turn']


def check_recovery_lineage(approval,plan,db,checks):
    """Retain archived source/ledger proofs for registered implementation repairs."""
    for attempt in approval['attempts']:
        receipt_path=attempt.get('administrative_recovery_receipt')
        if not receipt_path:continue
        checks.preserve(receipt_path);receipt=read(receipt_path)
        lineage=next((x for x in receipt.get('shards',[]) if x['shard']==plan['shard']['index']),receipt)
        archive=Path(lineage['archive']);before=read(archive/'plan.json')
        checks.preserve(archive/'plan.json');checks.preserve(archive/'approval.json');checks.preserve(archive/'state.sqlite3')
        checks.need(digest(before)==lineage['parent_plan_sha256'],'recovery_parent_plan_binding',job_id=attempt['job_id'])
        for key in ('schedule','shard','task_file_sha256','conditions','limits','resources','actor_revision','actor_files'):
            checks.need(before[key]==plan[key],'recovery_science_or_cap_change',job_id=attempt['job_id'],field=key)
        manifest=archive/'source/reference_manifest.json';checks.preserve(manifest,before['source_manifest_sha256'])
        for relative,h in read(manifest)['recipe_files_sha256'].items():checks.preserve(archive/'source'/relative,h)
        prior=snapshot(archive);current_attempts={x['attempt_id'] for x in db['attempts']};current_reservations={x['reservation_id']:x for x in db['reservations']}
        checks.need({x['attempt_id'] for x in prior['attempts']}<=current_attempts,'recovery_attempt_dropped',job_id=attempt['job_id'])
        for old in prior['reservations']:
            new=current_reservations.get(old['reservation_id'])
            checks.need(new is not None and all(new[k]==old[k] for k in ('attempt_id','kind','condition','count','reserved_usd','context_json')),'recovery_reservation_changed',job_id=attempt['job_id'],reservation=old['reservation_id'])


def check_startup_work(shard,plan,db,checks):
    """Account for paid diagnostics once and separate saved-only serving tests."""
    def proof(value):
        checks.preserve(value['path'],value['sha256']);return read(value['path'])
    no_attempt={r['reservation_id']:r for r in db['reservations'] if r['attempt_id'] is None}
    accounted=[];adoptions=[];acceptances=[];recovery=plan.get('serving_recovery');manifest=None
    if recovery:
        prior=proof(recovery['prior_plan']);manifest=proof(recovery['acceptance_manifest'])
        checks.need(digest(prior)==recovery['prior_plan_sha256'] and recovery['sampling']==SAMPLING and recovery['previous_max_running_requests']==5 and recovery['max_running_requests']==10,'serving_recovery_identity')
        for key in ('root','resources','limits','conditions','schedule','task_file','task_file_sha256','scientific_config_sha256','worker_config','protocol_name','campaign_id','shard','campaign_task_shards','primary_episodes','smoke_episodes'):
            checks.need(prior.get(key)==plan.get(key),'serving_recovery_scientific_drift',field=key)
        entries=manifest['requests']
        checks.need(manifest['source_plan_sha256']==recovery['prior_plan_sha256'] and manifest['root']==plan['root'] and manifest['actor_checkpoint']==plan['worker_config']['actor_checkpoint'] and manifest['sampling']==SAMPLING and len(entries)==10 and len({e['attempt_id'] for e in entries})==10,'acceptance_manifest_identity')
        for e in entries:
            request=proof(e['request']);batch=proof(e['source_batch'])['record'];saved=proof(e['source_rollout']);result=proof(e['source_result'])
            image=request.get('image_data',[]);image_hash=raw_sha(base64.b64decode(image[0].split(',',1)[-1])) if len(image)==1 else None
            checks.need(request.get('sampling_params')==dict(SAMPLING,sampling_seed=batch['candidate_seeds'][0]) and request.get('return_logprob') is True and raw_sha(request['text'].encode())==batch['prompt_sha256'] and image_hash==batch['screenshot_sha256'] and e['prompt_tokens']==batch['prompt_tokens'] and 16384<=e['prompt_tokens']<32768-4096,'acceptance_request_science',attempt_id=e['attempt_id'])
            checks.need(saved.get('llm_input_texts')==request['text'] and result.get('attempt_id')==e['attempt_id'],'acceptance_original_saved_evidence',attempt_id=e['attempt_id'])
    for jobdir in sorted((Path(shard)/'scheduler-attempts').iterdir()):
        job=jobdir.name;diag_path=jobdir/'jev-context-preflight.json';adopted=False
        allocation_path=jobdir/'allocation-validated.json'
        allocation=read(allocation_path) if allocation_path.exists() else {}
        if diag_path.exists():
            d=read(diag_path);checks.preserve(diag_path)
            checks.preserve(d['request_evidence_path']);event=read(d['request_evidence_path'])['record']
            checks.need(d['request_sha256']==digest(event['request']) and d['browser_episodes']==d['sft_proposals']==0,'diagnostic_binding',job_id=job)
            rid=d['reservation']['reservation_id'];ledger=no_attempt.get(rid)
            checks.need(ledger is not None and ledger['kind']=='jev_selector_requests' and ledger['count']==1 and json.loads(ledger['context_json']).get('request_sha256')==d['request_sha256'],'diagnostic_ledger_binding',job_id=job)
            if d.get('reused_from'):
                adopted=True;old=proof(d['reused_from'])
                expected=dict(old,plan_sha256=d['plan_sha256'],job_id=job,external_calls_made=0,reused_from=d['reused_from'])
                valid_old=(old.get('completed') is True and old.get('expected_model')=='jev-1.13.0' and old.get('state_bytes')==262144 and ((old.get('outcome')=='accepted' and old.get('model_identity')=='jev-1.13.0') or (old.get('outcome')=='known_capacity_rejection' and old.get('http_status')==400 and old.get('error_type')=='max_tokens_exceeded')))
                checks.need(d==expected and valid_old and d['plan_sha256']==allocation.get('plan_sha256') and recovery is not None and d['reused_from']==recovery['prior_jev_diagnostic'] and old['plan_sha256']==recovery['prior_plan_sha256'],'adopted_diagnostic_identity',job_id=job)
                adoptions.append(dict(job_id=job,reservation_id=rid,new_external_calls=0))
            else:accounted.append(rid)
        directory=jobdir/'serving-acceptance';reservation_path=directory/'reservation.json';receipt_path=jobdir/'serving-acceptance.json'
        if not reservation_path.exists():
            checks.need(not adopted or not (jobdir/'worker-runtime.json').exists(),'serving_workers_before_acceptance',job_id=job)
            continue
        checks.preserve(reservation_path);reserved=read(reservation_path);rid=reserved['reservation_id'];ledger=no_attempt.get(rid);accounted.append(rid)
        context=json.loads(ledger['context_json']) if ledger else {}
        checks.need(manifest is not None and ledger is not None and ledger['kind']=='local_sft_generations' and ledger['count']==reserved['count']==10 and ledger['condition'] is None and ledger['status']=='consumed' and ledger['reserved_usd']==ledger['charged_usd']==0 and context.get('purpose')==ACCEPTANCE_PURPOSE and context.get('request_sha256')==digest({'acceptance_manifest_sha256':recovery['acceptance_manifest']['sha256']}),'acceptance_reservation_binding',job_id=job)
        checks.need(reserved.get('purpose')==ACCEPTANCE_PURPOSE and reserved.get('request_sha256')==context.get('request_sha256'),'acceptance_reservation_receipt',job_id=job)
        server_path=jobdir/'serving-acceptance-server-info.json';checks.preserve(server_path);server=read(server_path)
        expected_server=dict(model_path=plan['worker_config']['actor_checkpoint'],max_running_requests=10,context_length=32768,mem_fraction_static=.60,disable_cuda_graph=True,dtype='bfloat16',tp_size=1)
        checks.need(all(server.get(k)==v for k,v in expected_server.items()),'acceptance_server_settings',job_id=job)
        if not receipt_path.exists():
            checks.need(not (jobdir/'worker-runtime.json').exists(),'serving_workers_before_acceptance',job_id=job)
            checks.pending.append('serving_acceptance_unfinished:'+job)
            continue
        checks.preserve(receipt_path);receipt=read(receipt_path);completed=receipt.get('completed') is True
        checks.need(receipt.get('purpose')==ACCEPTANCE_PURPOSE and receipt.get('plan_sha256')==allocation.get('plan_sha256') and receipt.get('reservation')==reserved and receipt.get('requests')==receipt.get('charged_local_sft_generations')==10 and receipt.get('browser_sessions')==receipt.get('external_api_calls')==receipt.get('executed_actions')==0 and receipt.get('acceptance_manifest')==recovery['acceptance_manifest'] and receipt.get('actor_checkpoint')==manifest['actor_checkpoint'] and receipt.get('max_running_requests')==10 and receipt.get('prompt_tokens')==[e['prompt_tokens'] for e in manifest['requests']],'acceptance_receipt_binding',job_id=job)
        expected_files={f'{kind}-{i:02d}.json' for kind in ('response','error') for i in range(10)}
        outcome_files=list(directory.glob('response-*.json'))+list(directory.glob('error-*.json'))
        for path in outcome_files:checks.preserve(path)
        checks.need(all(path.name in expected_files for path in outcome_files),'acceptance_unaccounted_outcome',job_id=job)
        good=[];errors=[]
        for index,e in enumerate(manifest['requests']):
            response_path=directory/f'response-{index:02d}.json';error_path=directory/f'error-{index:02d}.json'
            if error_path.exists():checks.preserve(error_path);errors.append(index)
            if response_path.exists():
                checks.preserve(response_path);response=read(response_path);meta=response.get('response',{}).get('meta_info',{})
                checks.need(response.get('index')==index and response.get('request_sha256')==e['request']['sha256'],'acceptance_response_request_binding',job_id=job,index=index)
                ok=(response.get('http_status')==200 and meta.get('prompt_tokens')==e['prompt_tokens'] and type(meta.get('completion_tokens')) is int and 0<meta['completion_tokens']<=4096 and meta.get('finish_reason',{}).get('type') in ('stop','length') and meta.get('total_retractions')==0 and 0<=response.get('elapsed_seconds',180)<180)
                if ok:good.append(index)
                if completed:checks.need(ok,'acceptance_response_validation',job_id=job,index=index)
            checks.need(response_path.exists() or error_path.exists(),'acceptance_outcome_missing',job_id=job,index=index)
        checks.need((not receipt.get('failures') and len(good)==10 and not errors) if completed else bool(receipt.get('failures')) and bool(errors),'acceptance_completion_consistency',job_id=job)
        checks.need(completed or not (jobdir/'worker-runtime.json').exists(),'serving_workers_after_failed_acceptance',job_id=job)
        acceptances.append(dict(job_id=job,completed=completed,generations_charged=10,successful_responses=len(good),errors=len(errors),prompt_tokens=receipt['prompt_tokens']))
    checks.need(Counter(accounted)==Counter(no_attempt.keys()),'startup_reservation_coverage',shard=plan['shard']['index'])
    return dict(diagnostic_adoptions=adoptions,saved_context_acceptances=acceptances)


def audit_attempt(a,q,plan,reservations,checks,format_judge,parser):
    aid=a['attempt_id'];directory=Path(a['artifact_directory']);descriptor=json.loads(a['result_json']) if a.get('result_json') else None
    amended_rejected_turn=None
    if descriptor and descriptor.get('administrative_amendment'):
        amended_rejected_turn=check_amendment(descriptor,checks)
    events=checks.events(directory/'events');browser=checks.events(directory/'browser')
    grouped=defaultdict(list)
    for path,e in events:grouped[e['event']].append(e['record'])
    for path,e in browser:
        for key in ('image',):
            if isinstance(e['record'].get(key),dict):checks.preserve(e['record'][key]['path'],e['record'][key]['sha256'])
    # Preserve every saved rollout and browser manifest, including interrupted attempts.
    rollout_files=list((directory/'rollouts').rglob('*.json'))
    for p in rollout_files+list(directory.glob('browser-manifest.json')):checks.preserve(p)
    batches=grouped['proposal_batch_reserved'];reserved_actor=sum(r['count'] for r in reservations if r['kind']=='local_sft_generations')
    expected_actor=sum(len(x['candidate_seeds']) for x in batches)
    for p in rollout_files:
        saved=read(p);prompt=saved.get('llm_input_texts')
        if isinstance(prompt,str) and prompt:
            checks.need(raw_sha(prompt.encode()) in {b['prompt_sha256'] for b in batches},'saved_actor_prompt_hash',attempt_id=aid)
            checks.need(Path(plan['worker_config']['prompt_path']).read_text() in prompt,'saved_actor_policy_missing',attempt_id=aid)
    if a['status']=='finished':checks.need(reserved_actor==expected_actor,'actor_batch_reservations',attempt_id=aid)
    else:checks.need(reserved_actor>=expected_actor,'interrupted_actor_reservations',attempt_id=aid)
    for batch in batches:
        n=1 if q['condition']=='L01' else 5
        checks.need(batch['condition']==q['condition'] and batch['sampling']==SAMPLING and batch['candidate_seeds']==[seed(q['task_id'],batch['turn'],i) for i in range(n)],'proposal_sampling_seed',attempt_id=aid)
        checks.need(batch['prompt_tokens']+4096<32768,'actor_context_bound',attempt_id=aid)
    provider=ARMS[q['condition']];requests=grouped['selector_reserved'];responses={r['turn']:r['response'] for r in grouped['selector_received']}
    for kind,records in [('om2w_judge_http_attempts',grouped['judge_reserved'])]+([(KIND[provider],requests)] if provider else []):
        ledger=[r for r in reservations if r['kind']==kind]
        known=Counter(digest(r['request']) for r in records);charged=Counter(json.loads(r['context_json']).get('request_sha256') for r in ledger)
        checks.need(known==charged if a['status']=='finished' else not (known-charged),'request_reservation_binding',attempt_id=aid,kind=kind)
    if not provider:checks.need(not requests and not responses,'baseline_has_selector',attempt_id=aid)
    proposals=defaultdict(dict)
    for row in grouped['proposal_received']:proposals[row['turn']][row['candidate']]=row['output_text']
    selected={};choice_rejected={r['turn'] for r in grouped['selector_choice_rejected']}
    if amended_rejected_turn is not None:choice_rejected.add(amended_rejected_turn)
    captures=[(p,e['record']) for p,e in browser if e['event']=='capture_verified'];capture_by_identity=defaultdict(list)
    for _,capture in captures:capture_by_identity[(capture['image']['sha256'],state_digest(capture['page']))].append(capture)
    for batch in batches:
        checks.need((batch['screenshot_sha256'],batch['page_sha256']) in capture_by_identity,'proposal_shared_capture_binding',attempt_id=aid,turn=batch['turn'])
        if not provider and set(proposals[batch['turn']])=={0}:selected[batch['turn']]=0
    saved_operations=[e['record'] for _,e in browser if e['event']=='operation_finished']
    for req in requests:
        turn=req['turn'];request=req['request'];state=json.loads(request['input'][1]['content']) if provider=='luna' else request['state']
        order=list(range(5));random.Random(seed(q['task_id'],turn,999)).shuffle(order)
        checks.need(req['provider']==provider and request['model']==MODELS[provider] and req['displayed_order']==order and req['state_sha256']==state_digest(state),'selector_state_order_identity',attempt_id=aid,turn=turn)
        batch=next((b for b in batches if b['turn']==turn),None)
        matching_captures=capture_by_identity.get((batch['screenshot_sha256'],batch['page_sha256']),[]) if batch else []
        if matching_captures and set(proposals[turn])==set(range(5)):
            intent=json.loads(q['task_json']).get('metadata',{}).get('intent')
            checks.need(any(state==expected_selector_state(capture,intent,proposals[turn],order,saved_operations) for capture in matching_captures),'selector_shared_dom_and_history',attempt_id=aid,turn=turn)
        checks.need(set(proposals[turn])==set(range(5)) and state['candidates']=={str(i+1):dict(text=proposals[turn].get(j)) for i,j in enumerate(order)},'candidate_integrity',attempt_id=aid,turn=turn)
        checks.need(len(json.dumps(state,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode())<=262144,'selector_state_bound',attempt_id=aid)
        if provider=='luna':checks.need(request.get('reasoning')=={'effort':'medium'} and request.get('max_output_tokens')==4096 and request.get('store') is False,'luna_settings',attempt_id=aid)
        if turn in responses:
            response=responses[turn];checks.need(response.get('model')==MODELS[provider],'selector_response_model',attempt_id=aid)
            try:selected[turn]=order[selector_index(response,provider)-1]
            except (KeyError,TypeError,ValueError):checks.need(turn in choice_rejected,'selector_choice_validation',attempt_id=aid,turn=turn)
    descriptor=json.loads(a['result_json']) if a.get('result_json') else None
    if descriptor is None:
        checks.need(a['status']!='finished','finished_result_missing',attempt_id=aid)
        return None
    checks.preserve(descriptor['result_path'],descriptor['result_sha256']);result=read(descriptor['result_path'])
    checks.need(all(result.get(k)==v for k,v in descriptor.items() if k not in ('result_path','result_sha256')),'descriptor_contents',attempt_id=aid)
    for key in ('task_id','condition'):checks.need(result.get(key)==q[key],'result_identity',attempt_id=aid,field=key)
    checks.need(result['attempt_id']==aid,'result_attempt_identity',attempt_id=aid)
    starts=grouped['episode_started'];checks.need(len(starts)==1 and starts[0]['sampling']==SAMPLING and starts[0]['reward_source_sha256']==JUDGE_SHA,'episode_contract',attempt_id=aid)
    metadata=result.get('metadata',{});messages=metadata.get('messages',[]);assistant=[x['content'] for x in messages if x['role']=='assistant']
    chosen=[proposals[t][selected[t]] for t in sorted(selected)]
    checks.need([normalized_history(x) for x in assistant]==[normalized_history(x) for x in chosen],'selected_proposal_history_binding',attempt_id=aid)
    dispatched=[e['record'] for _,e in browser if e['event']=='operation_dispatched'];finished=[e['record'] for _,e in browser if e['event']=='operation_finished']
    checks.need(result['actor_decisions']==len(batches)<=60 and len(dispatched)==len(finished)<=30,'episode_operation_caps',attempt_id=aid)
    if result.get('final_screenshot') or result['valid']:
        checks.need(result['browser_operations']==len(dispatched),'terminal_operation_count',attempt_id=aid)
    elif result['browser_operations']!=len(dispatched):
        checks.warnings.append(dict(code='missing_terminal_operation_counter',attempt_id=aid,reported=result['browser_operations'],preserved_dispatched=len(dispatched)))
    checks.need([x['attempt'] for x in dispatched]==list(range(1,len(dispatched)+1)),'operation_numbering',attempt_id=aid)
    image=result.get('final_screenshot');raw=None;dims=None
    if image:
        checks.preserve(image['path'],image['sha256']);raw=Path(image['path']).read_bytes()
        try:dims=png_dimensions(raw)
        except Exception as exc:checks.need(False,'png_decode',attempt_id=aid,error=type(exc).__name__)
        checks.need(dims==[1280,720],'png_dimensions',attempt_id=aid)
        ctl=metadata.get('controlled_terminal',{});checks.need(ctl.get('screenshot_sha256')==image['sha256'] and ctl.get('original_status')==result['status'] and ctl.get('reward_source_sha256')==JUDGE_SHA,'terminal_evidence_binding',attempt_id=aid)
        matches=[x for _,x in captures if x['image']['sha256']==image['sha256'] and x['action_attempts']==result['browser_operations']]
        checks.need(bool(matches) and captures[-1][1]['image']['sha256']==image['sha256'],'fresh_final_capture',attempt_id=aid)
        if metadata.get('full_image_list'):
            b64=metadata['full_image_list'][-1].split(',',1)[-1]
            checks.need(raw_sha(base64.b64decode(b64))==image['sha256'],'terminal_metadata_image',attempt_id=aid)
    checks.need(result['browser_closed'] and all(r['status'] in ('settled','closed_by_owner') for r in reservations if r['kind']=='browser_sessions'),'browser_closure',attempt_id=aid)
    valid=result['valid'];status=result['status']
    if valid and status=='Status.ABORTED':
        ctl=metadata.get('controlled_terminal',{});allowed={'action_limit','decision_limit','episode_deadline','requires_user_input','selector_choice_rejected'}
        checks.need(ctl.get('terminal_reason') in allowed or ctl.get('policy_boundary_reason') in allowed,'valid_aborted_boundary_missing',attempt_id=aid)
    if assistant:
        checks.need(messages[0].get('role')=='system' and messages[0].get('content')==Path(plan['worker_config']['prompt_path']).read_text(),'restored_policy_history',attempt_id=aid)
    judge_calls=grouped['judge_reserved'];verdicts=grouped['judge_received']
    if valid:checks.need(bool(image) and result['browser_closed'] and result['invalid_reason'] is None,'valid_episode_evidence',attempt_id=aid)
    checks.need(result['judge_http_attempts']==len(judge_calls)<=4,'judge_count',attempt_id=aid)
    if status!='Status.COMPLETED':checks.need(not judge_calls and result['reward']==0,'noncompleted_canonical_zero',attempt_id=aid)
    elif valid:
        checks.need(bool(judge_calls) and bool(verdicts),'completed_judge_missing',attempt_id=aid)
        if format_judge and raw:
            expected=format_judge(metadata['intent'],messages,raw)
            for call in judge_calls:checks.need(call['request']['messages']==expected,'canonical_judge_input_history',attempt_id=aid)
        elif not format_judge:checks.pending.append('canonical_parser_runtime_required')
    for call in judge_calls:
        request=call['request'];checks.need(status=='Status.COMPLETED' and request.get('model')==JUDGE and request.get('seed')==42 and request.get('max_completion_tokens')==4096 and call['source_sha256']==JUDGE_SHA,'canonical_judge_settings',attempt_id=aid)
    reward=metadata.get('reward',{})
    if verdicts:
        response=verdicts[-1]['response'];text=response['choices'][0]['message']['content'];value=None if 'status:' not in text.lower() else float('success' in text.lower().split('status:',1)[1])
        checks.need(response['model']==JUDGE and result['reward']==value and reward.get('judge_text')==text,'verdict_result_binding',attempt_id=aid)
    elif valid:checks.need(reward.get('protocol')=='online_mind2web','noncompleted_reward_protocol',attempt_id=aid)
    # Bind each returned priced usage to the ledger independently of the charge arithmetic.
    for kind,receipt_rows in [('om2w_judge_http_attempts',[r['response'] for r in verdicts])]+([('luna_selector_requests',list(responses.values()))] if provider=='luna' else []):
        def pair(u):return (u.get('prompt_tokens',u.get('input_tokens')),u.get('completion_tokens',u.get('output_tokens')))
        checks.need(Counter(pair(r.get('usage',{})) for r in receipt_rows)==Counter(pair(json.loads(r['usage_json'])) for r in reservations if r['kind']==kind and r['status']=='settled'),'response_usage_ledger',attempt_id=aid,kind=kind)
    task=json.loads(q['task_json']);terminal=metadata.get('controlled_terminal',{})
    return dict(task_id=q['task_id'],condition=q['condition'],category=q['category'],attempt_id=aid,shard=plan['shard']['index'],
        valid=valid,canonical_reward=result['reward'],success=valid and result['reward']==1,status=status,invalid_reason=result['invalid_reason'],
        terminal_reason=terminal.get('terminal_reason'),policy_boundary_reason=terminal.get('policy_boundary_reason'),
        actions=len(dispatched),reported_terminal_operations=result['browser_operations'],decisions=result['actor_decisions'],task=task.get('metadata',{}).get('intent',task.get('task_name')),
        result_path=descriptor['result_path'],result_sha256=descriptor['result_sha256'],final_screenshot=image,png_dimensions=dims,
        judge_text=reward.get('judge_text'),executed_operations=finished,assistant_history=assistant,
        candidate_batches=[dict(turn=t,selected_candidate=selected.get(t),proposals=proposals[t]) for t in sorted(proposals)])


def audit(root,*,schedulers=None,previous=None,formatter_factory=formatter):
    root=Path(root).resolve();checks=Checks();plans=[read(root/f'shard-{i}/plan.json') for i in range(2)];dbs=[snapshot(root/f'shard-{i}') for i in range(2)]
    campaign=read(root/'campaign-manifest.json');top=read(root/'approval.json')
    checks.need(top.get('approved') is True and top['campaign_manifest_sha256']==sha(root/'campaign-manifest.json'),'campaign_approval_binding')
    check_coverage(plans,[d['queue'] for d in dbs],checks);rows=[];shards=[]
    for i,(plan,db) in enumerate(zip(plans,dbs)):
        shard=root/f'shard-{i}';source=Path(plan['source']);approval=read(shard/'approval.json');config=plan['worker_config']
        checks.need(approval.get('approved') is True and approval['plan_sha256']==digest(plan) and approval['limits']==plan['limits'] and approval['resources']==plan['resources'],'shard_approval_binding',shard=i)
        checks.need(dict((x['key'],x['value']) for x in db['metadata'])['plan_sha256']==digest(plan),'database_plan_binding',shard=i)
        checks.need(plan['resources']==top['per_shard_resources'],'shard_resource_binding',shard=i)
        checks.need(plan['protocol_name']=='local-openwebrl-om2w-v2' and set(plan['conditions'])==set(ARMS),'suite_protocol',shard=i)
        check_recovery_lineage(approval,plan,db,checks)
        checks.preserve(source/'reference_manifest.json',plan['source_manifest_sha256'],immutable=False)
        for relative,h in read(source/'reference_manifest.json')['recipe_files_sha256'].items():checks.preserve(source/relative,h,immutable=False)
        checks.preserve(plan['task_file'],plan['task_file_sha256'],immutable=False)
        for relative,h in plan['offline_receipts'].items():checks.preserve(shard/relative,h,immutable=False)
        for key in ('prompt','tool_list'):checks.preserve(config[key+'_path'],config[key+'_sha256'],immutable=False)
        checks.preserve(source/'openwebrl/eval/reward_online_mind2web.py',JUDGE_SHA,immutable=False)
        checks.preserve(shard/'kev-source-proof.json',plan['kev_source_proof_sha256'],immutable=False)
        checks.preserve(shard/'kev-file-proof.json',config['kev_proof']['file_proof_sha256'],immutable=False)
        checks.preserve(config['browser_manifest']['binary_path'],config['browser_manifest']['binary_sha256'],immutable=False)
        for relative,stat in plan['actor_files'].items():
            p=Path(config['actor_checkpoint'])/relative;s=p.stat();checks.need(s.st_size==stat['size'] and s.st_mtime_ns==stat['mtime_ns'],'actor_checkpoint_stat_changed',file=str(p))
        checks.need(plan['actor_revision']=='15e777db2ddba2e0e82080ebccd3ad8d215b7f0a','actor_revision')
        model=read(Path(config['actor_checkpoint'])/'config.json');checks.need(model.get('model_type')=='qwen3_vl','actor_parser_type')
        try:format_judge,parser=formatter_factory(source,read(config['tool_list_path']))
        except (ImportError,RuntimeError) as exc:format_judge=parser=None;checks.pending.append('canonical_parser_runtime_required');checks.warnings.append(str(exc))
        queue={q['item_id']:q for q in db['queue']};reservations=defaultdict(list);finished=defaultdict(list)
        for r in db['reservations']:reservations[r['attempt_id']].append(r)
        for a in db['attempts']:
            if a['status']=='active':continue
            row=audit_attempt(a,queue[a['item_id']],plan,reservations[a['attempt_id']],checks,format_judge,parser)
            if a['status']=='finished':
                finished[a['item_id']].append(a['attempt_id'])
                if row:rows.append(row)
        for q in queue.values():
            if q['status']=='finished':checks.need(len(finished[q['item_id']])==1,'current_result_cardinality',item_id=q['item_id'])
        count=Counter(q['status'] for q in queue.values())
        if count!={'finished':612}:checks.pending.append(f'shard{i}:unfinished_queue')
        startup_work=check_startup_work(shard,plan,db,checks)
        budget=check_budget(db,plan,checks);ids=[str(x['job_id']) for x in approval['attempts']]
        records=schedulers[i] if schedulers else next(s['scheduler_attempts'] for s in read(root/'supervisor-latest.json')['shards'] if s['shard']==i)
        checks.need(Counter(r['job_id'] for r in records)==Counter(ids),'scheduler_coverage',shard=i)
        elapsed=sum(r['elapsed_seconds'] for r in records);checks.need(elapsed<=plan['resources']['total_seconds'],'scheduler_time_cap',shard=i)
        for r in records:
            job=r['job_id'];directory=shard/'scheduler-attempts'/job;tres=dict(x.split('=',1) for x in r.get('alloc_tres','').split(',') if '=' in x)
            if tres:checks.need(tres.get('cpu')=='16' and tres.get('mem') in ('240G','245760M') and tres.get('gres/gpu')=='2','scheduler_resource_profile',job_id=job)
            else:checks.pending.append('scheduler_resource_record_required:'+job)
            if r['state'] in ('RUNNING','PENDING','CONFIGURING','COMPLETING','UNKNOWN'):checks.pending.append('scheduler_not_final:'+job)
            else:
                proof=directory/'owned-process-teardown.json'
                if proof.exists():checks.need(read(proof).get('verified') is True,'teardown_failed',job_id=job);checks.preserve(proof)
                else:checks.pending.append('teardown_receipt_required:'+job)
            runtime=directory/'worker-runtime.json'
            if runtime.exists():
                value=read(runtime);checks.preserve(runtime);proof=value['kev_proof'];card=proof.get('server_card',{})
                checks.need(all(proof.get(k)==v for k,v in config['kev_proof'].items()),'kev_source_identity',job_id=job)
                checks.need(card.get('truncate_states') is False and card.get('max_state_tokens')==65536 and card.get('run')==proof['verified_model_path'],'kev_serving_identity',job_id=job)
                identity=directory/'sft-0-identity.json';checks.preserve(identity);ident=read(identity)
                checks.need(ident.get('model_path')==config['actor_checkpoint'] and ident.get('has_image_understanding') is True,'actor_server_identity',job_id=job)
        complete=shard/'collection-complete.json'
        if not complete.exists() or read(complete).get('collection_complete') is not True:checks.pending.append(f'shard{i}:collection_not_complete')
        wandb_path=shard/'wandb-current-readonly-verification.json';wandb=read(wandb_path) if wandb_path.exists() else {}
        if not(wandb.get('project')=='openwebrl-evals' and wandb.get('id')==plan['experiment_id'] and wandb.get('state')=='finished' and complete.exists() and wandb.get('checked_unix',0)>=complete.stat().st_mtime):checks.pending.append(f'shard{i}:final_wandb_verification_required')
        shards.append(dict(shard=i,queue=dict(count),attempts=dict(Counter(a['status'] for a in db['attempts'])),budget=budget,
            scheduler_attempts=records,scheduler_seconds=elapsed,scheduler_cap=plan['resources']['total_seconds'],wandb=wandb,startup_work=startup_work))
    for key,cap in top['aggregate_limits'].items():checks.need(sum(p['limits'][key] for p in plans)==cap,'aggregate_cap_binding',counter=key)
    if previous:
        for p,h in previous.get('immutable_artifacts',{}).items():checks.preserve(p,h)
    ready=not checks.issues and not checks.pending;summary={}
    for arm in ARMS:
        rr=[r for r in rows if r['condition']==arm and r['category']=='primary'];valid=sum(r['valid'] for r in rr);wins=sum(r['success'] for r in rr)
        summary[arm]=dict(label=LABELS[arm],finished=len(rr),expected=300,valid=valid,invalid=len(rr)-valid,successes=wins,
            partial_success_rate=wins/len(rr) if rr else None,final_success_rate=wins/300 if ready else None,
            final_valid_only_rate=wins/valid if ready and valid else None,terminals=dict(Counter(r['terminal_reason'] or r['policy_boundary_reason'] or r['invalid_reason'] or r['status'] for r in rr)))
    return dict(created_utc=stamp(),root=str(root),auditor_script_sha256=sha(__file__),verified_complete=False,audit_passed=ready,status='ready_for_owner_verification' if ready else 'partial_unverified',
        issues=checks.issues,pending=sorted(set(checks.pending)),warnings=checks.warnings,summary=summary,
        primary_finished=sum(r['category']=='primary' for r in rows),smoke_finished=sum(r['category']=='smoke' for r in rows),
        shards=shards,immutable_artifacts=checks.artifacts,source_and_control_proofs=checks.controls,rows=rows,
        limitations=['Actor checkpoint identity uses the approved revision and exact frozen file size/mtime proof; this audit does not rehash multi-GB weights.',
            'Live snapshots exclude active attempts from per-episode checks; their reservations remain charged in budget accounting.',
            'Invalid ABORTED episodes may retain canonical zero; validity remains separate and these are not counted as valid task failures.'])


def write_review(report,out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);esc=html.escape;cards=[]
    for row in report['rows']:
        sub=out/'episodes'/row['attempt_id'];sub.mkdir(parents=True,exist_ok=True)
        value=json.loads(json.dumps(row));image=value.get('final_screenshot')
        if image:
            src=Path(image['path']);dest=sub/'final.png'
            if not dest.exists():shutil.copyfile(src,dest)
            if sha(dest)!=image['sha256']:raise ValueError('Private screenshot copy differs')
            image['path']='final.png'
        # Curated evidence excludes repeated base64 images and opaque model state.
        (sub/'evidence.json').write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
        status='invalid' if not row['valid'] else 'success' if row['success'] else 'failure'
        history=''.join('<details><summary>Turn '+str(i+1)+'</summary><pre>'+esc(text)+'</pre></details>' for i,text in enumerate(row['assistant_history']))
        candidates=''.join('<details><summary>Decision '+str(batch['turn']+1)+' · selected original candidate '+str(batch['selected_candidate'])+'</summary><pre>'+esc(json.dumps(batch['proposals'],indent=2,ensure_ascii=False))+'</pre></details>' for batch in row['candidate_batches'])
        page=STYLE+'<a href="../../index.html">All tasks</a><h1>'+esc(LABELS[row['condition']]+' · '+status)+'</h1><p>'+esc(row['task'] or row['task_id'])+'</p><p>'+esc(str(row['terminal_reason'] or row['policy_boundary_reason'] or row['invalid_reason'] or row['status']))+'</p><p><a href="evidence.json">Curated evidence and immutable source hash</a></p>'+('<img loading="lazy" src="final.png" alt="Fresh final screenshot">' if image else '<p>No final screenshot; invalid observation.</p>')+'<h2>Canonical judge</h2><pre>'+esc(row['judge_text'] or 'No judge request.')+'</pre><h2>Executed browser operations</h2><pre>'+esc(json.dumps(row['executed_operations'],indent=2,ensure_ascii=False))+'</pre><h2>Original assistant history</h2>'+history+'<h2>Original actor proposals</h2>'+candidates
        (sub/'index.html').write_text(page)
        cards.append('<tr data-condition="'+row['condition']+'" data-status="'+status+'" data-category="'+row['category']+'"><td>'+esc(LABELS[row['condition']])+'</td><td>'+status+'</td><td>'+esc(row['category'])+'</td><td><a href="episodes/'+row['attempt_id']+'/">'+esc(row['task'] or row['task_id'])+'</a></td><td>'+str(row['actions'])+'</td><td>'+str(row['decisions'])+'</td></tr>')
    summary=''.join('<tr><td>'+esc(v['label'])+'</td><td>'+str(v['finished'])+'/300</td><td>'+str(v['successes'])+'</td><td>'+str(v['valid'])+'</td><td>'+str(v['invalid'])+'</td><td>'+('pending' if v['final_success_rate'] is None else f"{100*v['final_success_rate']:.2f}%")+'</td></tr>' for v in report['summary'].values())
    page=STYLE+'<h1>Local SFT actor and selectors</h1><p>'+esc(report['created_utc'])+'</p><p class="note">'+esc(report['status'])+'. Four matched conditions, 300 tasks each plus 24 separate smoke episodes. Partial counts are not final rates. Invalid observations remain separate from canonical task failures.</p><table><thead><tr><th>Condition</th><th>Finished</th><th>Success</th><th>Valid</th><th>Invalid</th><th>Final overall rate</th></tr></thead><tbody>'+summary+'</tbody></table><details><summary>Audit gates and budgets</summary><pre>'+esc(json.dumps({k:report[k] for k in ('issues','pending','shards','limitations')},indent=2))+'</pre></details><p><input id="query" placeholder="Search task"><select id="condition"><option value="">All conditions</option>'+''.join('<option value="'+k+'">'+v+'</option>' for k,v in LABELS.items())+'</select><select id="status"><option value="">All outcomes</option><option>success</option><option>failure</option><option>invalid</option></select><select id="category"><option value="primary">Primary</option><option value="smoke">Smokes</option><option value="">Both</option></select></p><table><thead><tr><th>Condition</th><th>Outcome</th><th>Cohort</th><th>Task</th><th>Operations</th><th>Decisions</th></tr></thead><tbody id="tasks">'+''.join(cards)+'</tbody></table><script>const ids=["query","condition","status","category"],x=Object.fromEntries(ids.map(k=>[k,document.getElementById(k)]));function f(){for(const r of document.querySelectorAll("#tasks tr"))r.hidden=!(r.textContent.toLowerCase().includes(x.query.value.toLowerCase())&&ids.slice(1).every(k=>!x[k].value||r.dataset[k]===x[k].value));}ids.forEach(k=>x[k].addEventListener("input",f));f();</script>'
    (out/'index.html').write_text(page)


STYLE='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Private local SFT review</title><style>body{font:15px system-ui;max-width:1250px;margin:2rem auto;padding:0 1rem;color:#182330;background:#f7f9fc}table{border-collapse:collapse;width:100%;background:white}td,th{padding:.7rem;text-align:left;border-bottom:1px solid #dce2eb}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#eef1f5;padding:1rem}img{max-width:100%}details{padding:.5rem;background:white;margin:.4rem 0}summary{cursor:pointer}input,select{padding:.6rem;margin:.3rem}.note{border-left:4px solid #d7a51c;padding:1rem;background:#fff3d3}</style>'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline',type=Path);p.add_argument('--refresh-scheduler',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args();root=args.root.resolve();out=args.output or root/'private-review'
    os.environ.setdefault('FLASHINFER_WORKSPACE_BASE','/tmp/sft-audit-cpu-cache')
    os.environ.setdefault('XDG_CACHE_HOME','/tmp/sft-audit-cpu-cache')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/sft-audit-cpu-cache/torchinductor')
    os.environ.setdefault('TRITON_CACHE_DIR','/tmp/sft-audit-cpu-cache/triton')
    if out.resolve()==root or any(out.resolve().is_relative_to(root/f'shard-{i}') for i in range(2)):raise ValueError('Write only a separate review directory')
    schedules=[scheduler_records([str(a['job_id']) for a in read(root/f'shard-{i}/approval.json')['attempts']]) for i in range(2)] if args.refresh_scheduler else None
    report=audit(root,schedulers=schedules,previous=read_baseline(args.baseline) if args.baseline else None);write_review(report,out)
    path=out/('audit-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    with path.open('x') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps(dict(report=str(path),html=str(out/'index.html'),status=report['status'],summary=report['summary'],issues=len(report['issues']),pending=len(report['pending'])),indent=2))


if __name__=='__main__':main()
