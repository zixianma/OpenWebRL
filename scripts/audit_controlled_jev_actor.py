#!/usr/bin/env python3
"""Read-only scientific/accounting audit; writes only separate private review artifacts.

Never opens a browser, calls a model, changes the queue, or marks a run complete.
Original DONE verdicts are superseded until a saved-only repaired judgment exists.
"""
from __future__ import annotations
import argparse
import ast
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import subprocess
import zlib

JUDGE_SHA = 'b1f2c30c852b36d8049e23bd8ef4efb44a0786742449b5e7ef4e5a7ea55af583'
MODELS = {'jev':'jev-1.13.0', 'text':'gpt-4.1-mini-2025-04-14', 'judge':'o4-mini-2025-04-16'}
KINDS = {'jev':'jev_requests', 'text':'text_http_attempts', 'judge':'om2w_judge_http_attempts'}
RATES = {'text_http_attempts':('text_usd', .4, 1.6), 'om2w_judge_http_attempts':('judge_usd', 1.1, 4.4)}


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()


def png_dimensions(raw):
    """Verify all chunk CRCs, the terminal chunk, and dimensions without decoding pixels."""
    if raw[:8] != b'\x89PNG\r\n\x1a\n': raise ValueError('Not a PNG')
    offset, dimensions, done = 8, None, False
    while offset < len(raw):
        size = struct.unpack('>I',raw[offset:offset+4])[0]
        kind, data = raw[offset+4:offset+8], raw[offset+8:offset+8+size]
        crc = struct.unpack('>I',raw[offset+8+size:offset+12+size])[0]
        if zlib.crc32(kind+data)&0xffffffff != crc: raise ValueError('PNG CRC mismatch')
        if dimensions is None:
            if kind != b'IHDR' or size != 13: raise ValueError('Missing PNG IHDR')
            dimensions = list(struct.unpack('>II',data[:8]))
        offset += size+12
        if kind == b'IEND': done=True; break
    if not done or offset != len(raw): raise ValueError('Incomplete PNG or trailing bytes')
    return dimensions


def canonical_text(task, operations, system_source):
    """Independent construction of the documented canonical input, no model/parser imports."""
    constants = {}
    for node in ast.parse(Path(system_source).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in ('SYSTEM_PROMPT','USER_PROMPT'):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    turns=[]
    for i, op in enumerate(operations,1):
        action=op['action']['kind']
        params=', '.join(f'{k}={v!r}' for k,v in op.items())
        turns.append(f'Thought {i}: \nAction {i}: {action}({params})'.replace('\n\n',' '))
    turns.append(f"Thought {len(operations)+1}: \nAction {len(operations)+1}: done(response='')")
    return constants['SYSTEM_PROMPT'], constants['USER_PROMPT'].format(task=task,thoughts_and_actions='\n\n'.join(turns))


def check_repaired_request(request, task, operations, png, source):
    system,user=canonical_text(task,operations,source)
    if (request.get('model')!=MODELS['judge'] or request.get('seed')!=42
            or request.get('max_completion_tokens')!=4096): return 'repaired_judge_settings'
    messages=request.get('messages',[])
    expected=[{'role':'system','content':system}, {'role':'user','content':[
        {'type':'text','text':user}, {'type':'image_url','image_url':{
            'url':'data:image/png;base64,'+base64.b64encode(png).decode(),'detail':'high'}}]}]
    if messages != expected: return 'repaired_judge_evidence_differs'
    return None


def database(root):
    # A short consistent transaction: never initialize/migrate the live ledger.
    with sqlite3.connect((root/'state.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row; db.execute('BEGIN')
        tables={name:[dict(r) for r in db.execute('SELECT * FROM '+name)]
                for name in ('queue','attempts','reservations','counters','metadata')}
        db.rollback()
    return tables


def scheduler_records(ids):
    result=subprocess.run(['sacct','-X','-n','-P','-j',','.join(ids),
        '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'],capture_output=True,text=True,timeout=30,check=True)
    records=[]
    for line in result.stdout.splitlines():
        fields=line.split('|')
        if fields[0] in ids: records.append(dict(job_id=fields[0],state=fields[1],elapsed_seconds=int(fields[2]),alloc_tres=fields[3]))
    return records


def audit(root, *, scheduler=None, previous=None):
    root=Path(root).resolve(); plan=read(root/'plan.json'); approval=read(root/'approval.json')
    db=database(root); issues=[]; pending=[]; artifacts={}; rows=[]
    def require(ok,code,**context):
        if not ok: issues.append(dict(code=code,**context))
    def preserve(path, expected=None):
        p=Path(path).resolve()
        require(p.is_relative_to(root),'artifact_outside_root',path=str(p))
        if not p.is_file(): require(False,'missing_artifact',path=str(p)); return None
        actual=sha(p); artifacts[str(p)]=actual
        if expected: require(actual==expected,'artifact_hash_mismatch',path=str(p))
        return actual
    require(approval.get('approved') is True and approval.get('plan_sha256')==digest(plan),'approval_plan_binding')
    require(approval.get('limits')==plan['limits'] and approval.get('resources')==plan['resources'],'approval_caps_binding')
    require(dict((r['key'],r['value']) for r in db['metadata']).get('plan_sha256')==digest(plan),'database_plan_binding')
    source=Path(plan['source']); manifest=read(source/'reference_manifest.json')
    preserve(source/'reference_manifest.json',plan['source_manifest_sha256'])
    for relative,expected in manifest['recipe_files_sha256'].items(): preserve(source/relative,expected)
    preserve(plan['task_file'],plan['task_file_sha256'])
    preserve(root/'pricing-verification.json',plan['pricing_sha256'])
    reward_source=source/'openwebrl/eval/reward_online_mind2web.py'
    preserve(reward_source,JUDGE_SHA)
    expected=Counter((r['item_id'],r['task_id'],r['category'],r['condition']) for r in plan['schedule'])
    actual=Counter((r['item_id'],r['task_id'],r['category'],r['condition']) for r in db['queue'])
    require(actual==expected,'queue_identity_mismatch')
    require(len(expected)==303 and Counter(x[2] for x in expected)=={'primary':300,'smoke':3},'schedule_cardinality')
    require(len({x[1] for x in expected if x[2]=='primary'})==300,'primary_task_duplicates')
    attempts={r['attempt_id']:r for r in db['attempts']}; current=defaultdict(list)
    for a in db['attempts']:
        if a['status']=='finished': current[a['item_id']].append(a)
    reservation_groups=defaultdict(list)
    for r in db['reservations']: reservation_groups[r['attempt_id']].append(r)
    event_cache={}; total_received=Counter(); token_usage=defaultdict(Counter)
    for aid,a in attempts.items():
        entries=[]; directory=Path(a['artifact_directory'])
        for file in sorted((directory/'events').glob('[0-9]*.json')):
            event=read(file); entries.append((file,event))
            if a['status']!='active': preserve(file)
        event_cache[aid]=entries
        if a['status']=='active': pending.append('active_attempt:'+aid); continue
        if a.get('result_json'):
            descriptor=json.loads(a['result_json'])
            preserve(descriptor['result_path'],descriptor['result_sha256'])
        # Original immutable result survives an administrative amendment too.
        if (directory/'result.json').is_file(): preserve(directory/'result.json')
        for file in directory.glob('state-*.json.gz'): preserve(file)
        if (directory/'events/final.png').exists(): preserve(directory/'events/final.png')
        for provider,kind in KINDS.items():
            reserved=[e['record'] for _,e in entries if e['event']==provider+'_reserved']
            receipts=[e['record']['response'] for _,e in entries if e['event']==provider+'_received']
            ledger=[r for r in reservation_groups[aid] if r['kind']==kind]
            require(len(reserved)==len(ledger),'request_reservation_count',attempt_id=aid,provider=provider)
            require(Counter(digest(e['request']) for e in reserved)==Counter(json.loads(r['context_json']).get('request_sha256') for r in ledger),
                    'request_reservation_hashes',attempt_id=aid,provider=provider)
            total_received[provider]+=len(receipts)
            for response in receipts:
                require(response.get('model')==MODELS[provider],'response_model',attempt_id=aid,provider=provider)
                usage=response.get('usage',{})
                for target,keys in [('input_tokens',('prompt_tokens','input_tokens')),('output_tokens',('completion_tokens','output_tokens'))]:
                    value=next((usage[k] for k in keys if k in usage),None)
                    require(type(value) is int and value>=0,'missing_receipt_usage',attempt_id=aid,provider=provider)
                    if type(value) is int: token_usage[provider][target]+=value
            if provider in ('text','judge'):
                def usage_pair(value):return (value.get('prompt_tokens',value.get('input_tokens')),value.get('completion_tokens',value.get('output_tokens')))
                require(Counter(usage_pair(r.get('usage',{})) for r in receipts)==Counter(usage_pair(json.loads(r['usage_json'])) for r in ledger if r['status']=='settled'),
                    'response_ledger_usage',attempt_id=aid,provider=provider)
            for record in reserved:
                request=record['request']
                require(request.get('model')==MODELS[provider],'request_model',attempt_id=aid,provider=provider)
                if provider=='text': require(all(request.get(k)==v for k,v in {'temperature':.6,'top_p':.95,'max_tokens':1024}.items()),'typing_sampling',attempt_id=aid)
                if provider=='judge': require(request.get('seed')==42 and request.get('max_completion_tokens')==4096 and record.get('source_sha256')==JUDGE_SHA,'judge_settings',attempt_id=aid)
        final=[e['record'] for _,e in entries if e['event']=='native_final_state']
        judges=[e for _,e in entries if e['event']=='judge_reserved']
        if judges: require(bool(final) and final[-1]['terminal']=='done','judge_on_non_done',attempt_id=aid)
    rejudge_manifest=None
    if plan.get('saved_rejudge'):
        binding=plan['saved_rejudge'];preserve(binding['input_manifest'],binding['input_manifest_sha256'])
        rejudge_manifest=read(binding['input_manifest']);entries=rejudge_manifest['entries']
        wanted=set()
        for a in attempts.values():
            if a.get('result_json'):
                r=read(json.loads(a['result_json'])['result_path'])
                if r.get('valid') and r.get('terminal')=='done' and not r.get('saved_rejudge'):wanted.add(a['attempt_id'])
        require(Counter(e['source_attempt_id'] for e in entries)==Counter(wanted),'rejudge_selection_coverage')
        require(rejudge_manifest.get('canonical_reward_sha256')==JUDGE_SHA and rejudge_manifest.get('new_browser_episodes')==0
            and rejudge_manifest.get('external_calls')=='canonical_judge_only','rejudge_manifest_contract')
    corrected=0; original_done=0
    for q in sorted(db['queue'],key=lambda x:x['position']):
        candidates=current[q['item_id']]
        if q['status']!='finished': pending.append('unfinished_item:'+q['item_id']); continue
        require(len(candidates)==1,'canonical_attempt_cardinality',item_id=q['item_id'])
        if len(candidates)!=1: continue
        a=candidates[0]; aid=a['attempt_id']; descriptor=json.loads(a['result_json']); result=read(descriptor['result_path'])
        for key in ('attempt_id','task_id','condition','category'): require(result.get(key)==(aid if key=='attempt_id' else q[key]),'result_identity',attempt_id=aid,field=key)
        require(all(result.get(k)==v for k,v in descriptor.items() if k not in ('result_path','result_sha256')),'result_descriptor_mismatch',attempt_id=aid)
        require(result.get('actor_model')==MODELS['jev'] and result.get('typing_model')==MODELS['text'] and result.get('judge_source_sha256')==JUDGE_SHA,'result_models',attempt_id=aid)
        entries=event_cache[aid]; final=[e['record'] for _,e in entries if e['event']=='native_final_state']
        require(len(final)==1,'native_final_state_count',attempt_id=aid)
        native=final[-1] if final else {}; ops=native.get('operations',[])
        repaired=result.get('saved_rejudge')
        if repaired and not result['valid']:pending.append('repaired_judge_invalid:'+aid)
        origin=attempts.get(repaired.get('source_attempt_id')) if repaired else a
        origin_entries=event_cache.get(origin['attempt_id'],[]) if origin else []
        starts=[e['record'] for _,e in origin_entries if e['event']=='episode_started']
        require(len(starts)==1 and starts[0].get('actor')==MODELS['jev'] and starts[0].get('helper')==MODELS['text'] and starts[0].get('harness_revision')=='actionable-v2' and starts[0].get('actor_images') is False and starts[0].get('canonical_thoughts')=='empty: native Jev does not generate reasoning text','native_actor_input_contract',attempt_id=aid)
        dispatched=[e['record'] for _,e in origin_entries if e['event']=='operation_dispatched']
        finished=[e['record'] for _,e in origin_entries if e['event']=='operation_finished']
        require(len(ops)==len(dispatched)==len(finished),'native_operation_count',attempt_id=aid)
        require(ops==finished,'native_operation_receipts',attempt_id=aid)
        if result['valid']:
            require(origin is not None and sum(r['kind']=='browser_sessions' for r in reservation_groups[origin['attempt_id']])==1,'valid_origin_browser_reservation',attempt_id=aid)
        require(all(op.get('action',{}).get('kind') in ('click','fill','select','scroll','wait') for op in ops),'unknown_native_operation',attempt_id=aid)
        require(result['actions']==len(ops)+(result['terminal']=='done') and result['actions']<=30 and result['decisions']<=60,'operation_decision_caps',attempt_id=aid)
        image=result.get('final_screenshot'); image_ok=False; dimensions=None
        if image:
            preserve(image['path'],image['sha256'])
            try: dimensions=png_dimensions(Path(image['path']).read_bytes()); image_ok=dimensions==[1280,720]
            except (ValueError,struct.error,OSError): pass
            require(image_ok,'invalid_png',attempt_id=aid)
        amended_native_error=False
        amendment=result.get('offline_reconciliation')
        if amendment:
            preserve(amendment['original_result_path'],amendment['original_result_sha256'])
            preserve(amendment['response_path'],amendment['response_sha256'])
            old=read(amendment['original_result_path'])
            response=read(amendment['response_path'])['record']['response']
            answer=response['answers']['operation'];probabilities=answer['probabilities']
            amended_native_error=(result['terminal']=='native_choice_invalid' and result['actor_error']=='ValueError'
                and probabilities[answer['choice']]<max(probabilities.values())
                and old['final_screenshot']==image and old['api_attempts']==result['api_attempts']
                and old['actions']==result['actions'] and old['decisions']==result['decisions']
                and amendment['canonical_judge_calls']==amendment['additional_model_calls']==amendment['additional_browser_sessions']==0)
            require(amended_native_error,'native_choice_amendment_evidence',attempt_id=aid)
        if result['valid']: require(image_ok and result['browser_closed'] and not result['cleanup_errors'] and (result['actor_error'] is None or amended_native_error),'valid_evidence',attempt_id=aid)
        else: require(result['score'] is None,'invalid_scored_as_failure',attempt_id=aid)
        require(native.get('final_screenshot')==image,'terminal_image_binding',attempt_id=aid)
        judged=[e['record'] for _,e in entries if e['event']=='judge_reserved']
        if judged:
            names=[e['event'] for _,e in entries]
            require(names.index('native_final_state')<names.index('judge_reserved'),'judge_before_terminal_capture',attempt_id=aid)
        received=[e['record'] for _,e in entries if e['event']=='judge_received']
        api=result['api_attempts']
        require(all(api.get(p)==sum(e['event']==p+'_reserved' for _,e in entries) for p in KINDS),'result_api_counts',attempt_id=aid)
        score=result.get('score'); judge_status='canonical zero, no judge' if result['valid'] else 'invalid observation; unscored'
        old_score=None
        if result['terminal']=='done' and result['valid']:
            original_done+=1
            if not repaired:
                pending.append('saved_rejudge_required:'+aid); old_score=score; score=None; judge_status='original judge input superseded; repair pending'
            else:
                corrected+=1; judge_status='repaired saved-only canonical judge'
                require(rejudge_manifest is not None,'rejudge_manifest_missing',attempt_id=aid)
                binding=plan.get('saved_rejudge',{})
                require(repaired.get('input_manifest_sha256')==binding.get('input_manifest_sha256'),'rejudge_manifest_binding',attempt_id=aid)
                require(repaired.get('corrected_adapter_sha256')==sha(source/'openwebrl/controlled_jev_eval.py'),'rejudge_adapter_binding',attempt_id=aid)
                entry=next((e for e in (rejudge_manifest or {}).get('entries',[]) if e['source_attempt_id']==repaired['source_attempt_id']),{})
                require(all(entry.get(k)==q[k] for k in ('item_id','task_id','category')),'rejudge_entry_identity',attempt_id=aid)
                require(entry.get('task_sha256')==digest(next(t for t in plan['tasks'] if t['task_id']==q['task_id'])),'rejudge_task_binding',attempt_id=aid)
                require(origin is not None and origin['status']=='superseded_judge_input','rejudge_source_attempt',attempt_id=aid)
                require(repaired.get('actor_replayed') is False and repaired.get('browser_replayed') is False,'rejudge_no_replay',attempt_id=aid)
                require(not any(r['kind']!='om2w_judge_http_attempts' for r in reservation_groups[aid]),'rejudge_actor_browser_reservation',attempt_id=aid)
                require(api['jev']==api['text']==0,'rejudge_actor_api',attempt_id=aid)
                for field in ('result','native_final_state','final_screenshot'):
                    proof=repaired.get('source_'+field)
                    if proof:
                        preserve(proof['path'],proof['sha256'])
                        require(proof==entry.get(field),'rejudge_entry_artifact_binding',attempt_id=aid,field=field)
                    else: require(False,'rejudge_source_proof_missing',attempt_id=aid,field=field)
                if origin and origin.get('result_json'):
                    old=read(json.loads(origin['result_json'])['result_path']); old_score=old['score']
                    require(repaired.get('source_actor_api_attempts')=={k:v for k,v in old['api_attempts'].items() if k!='judge'} and repaired.get('source_episode_elapsed_seconds')==old['elapsed_seconds'],'rejudge_source_accounting',attempt_id=aid)
                    for key in ('task_id','category','actor_model','typing_model','actions','decisions','terminal','browser_closed','valid'):
                        require(result.get(key)==old.get(key),'rejudge_changed_actor_outcome',attempt_id=aid,field=key)
                    require(image['sha256']==old['final_screenshot']['sha256'],'rejudge_changed_image',attempt_id=aid)
                task=json.loads(q['task_json']); intent=task.get('metadata',{}).get('intent',task.get('task_name'))
                for request in judged:
                    issue=check_repaired_request(request['request'],intent,ops,Path(image['path']).read_bytes(),reward_source)
                    if issue: require(False,issue,attempt_id=aid)
                require(1<=len(judged)<=4 and bool(received),'repaired_judge_receipts',attempt_id=aid)
                if received:
                    response=received[-1]['response']; text=response['choices'][0]['message']['content']
                    expected_score=None if 'status:' not in text.lower() else float('success' in text.lower().split('status:',1)[1])
                    require(score==expected_score and result['judge']['judge_text']==text,'repaired_verdict_binding',attempt_id=aid)
        elif result['valid']:
            require(score==0 and not judged and not received,'non_done_canonical_zero',attempt_id=aid)
        original_result=None
        if repaired and origin:
            od=json.loads(origin['result_json']);ov=read(od['result_path'])
            original_result=dict(path=od['result_path'],sha256=od['result_sha256'],judge_text=(ov.get('judge') or {}).get('judge_text'))
        task=json.loads(q['task_json']); decisions=[]
        for _,e in origin_entries:
            if e['event']=='jev_received':
                answer=e['record']['response'].get('answers',{})
                decisions.append({key:{k:v for k,v in value.items() if k in ('choice','confidence','probabilities')} for key,value in answer.items()})
        rows.append(dict(task_id=q['task_id'],category=q['category'],attempt_id=aid,valid=result['valid'],terminal=result['terminal'],
            actor_error=result['actor_error'],score=score,original_superseded_score=old_score,judge_status=judge_status,
            actions=result['actions'],decisions=result['decisions'],operations=ops,model_decisions=decisions,
            task=task.get('task_name',task.get('metadata',{}).get('intent')),website=task.get('website'),
            final_screenshot=image,png_dimensions=dimensions,result_path=descriptor['result_path'],result_sha256=descriptor['result_sha256'],
            judge_text=(result.get('judge') or {}).get('judge_text'),original_superseded_result=original_result,original_attempt_id=origin['attempt_id'] if origin else None))
    counters={r['key']:r['value'] for r in db['counters']}; counts=Counter(); charges=Counter(); unsettled=[]
    for r in db['reservations']:
        require(r['count']==1,'reservation_unit_count',reservation_id=r['reservation_id'])
        counts[r['kind']]+=r['count']
        if r['status']=='reserved': unsettled.append(r['reservation_id'])
        if r['kind'] in RATES:
            key,ip,op=RATES[r['kind']];charges[key]+=r['charged_usd']
            if r['status']=='settled':
                usage=json.loads(r['usage_json']);it=usage.get('prompt_tokens',usage.get('input_tokens'));ot=usage.get('completion_tokens',usage.get('output_tokens'))
                require(type(it) is int and type(ot) is int,'ledger_usage_missing',reservation_id=r['reservation_id'])
                if type(it) is int and type(ot) is int:
                    bound=json.loads(r['context_json'])
                    require(0<=it<=bound['input_token_bound'] and 0<=ot<=bound['output_token_bound'],'ledger_usage_bound',reservation_id=r['reservation_id'])
                    require(math.isclose(r['charged_usd'],(it*ip+ot*op)/1e6,abs_tol=1e-10),'ledger_usage_cost',reservation_id=r['reservation_id'])
            elif r['status']=='charged_unknown': require(r['charged_usd']==r['reserved_usd'],'unknown_not_fully_charged',reservation_id=r['reservation_id'])
    for key in ('browser_sessions',*KINDS.values(),*(x[0] for x in RATES.values())):
        value=charges.get(key,0) if key.endswith('_usd') else counts.get(key,0)
        require(math.isclose(counters.get(key,0),value,abs_tol=1e-8),'counter_reconciliation',counter=key)
        require(value<=plan['limits'][key]+1e-8,'budget_exceeded',counter=key)
    open_browsers=sum(r['kind']=='browser_sessions' and r['status']=='reserved' for r in db['reservations'])
    require(counters.get('active_browsers',0)==open_browsers and open_browsers<=4,'active_browser_reconciliation')
    if unsettled:pending.append('unsettled_reservations:'+str(len(unsettled)))
    jobs=[str(x['job_id']) for x in approval['attempts']]
    if scheduler is None: scheduler=read(root/'supervisor-latest.json')['shards'][0]['scheduler_attempts']
    require(Counter(x['job_id'] for x in scheduler)==Counter(jobs),'scheduler_attempt_coverage')
    elapsed=sum(x.get('elapsed_seconds',0) for x in scheduler)
    require(elapsed<=plan['resources']['total_seconds'],'scheduler_budget_exceeded')
    for record in scheduler:
        job=record['job_id']
        if 'alloc_tres' in record:
            tres=dict(x.split('=',1) for x in record['alloc_tres'].split(',') if '=' in x)
            require(tres.get('cpu')=='8' and tres.get('mem') in ('32G','32768M') and not any(k.startswith('gres/gpu') and v!='0' for k,v in tres.items()),'scheduler_allocated_resources',job_id=job)
        else:pending.append('scheduler_resource_record_required:'+job)
        teardown=root/'scheduler-attempts'/job/'owned-process-teardown.json'
        if record['state'] in ('RUNNING','PENDING','CONFIGURING','COMPLETING','UNKNOWN'):pending.append('scheduler_not_final:'+job)
        elif teardown.exists():require(read(teardown).get('verified') is True,'teardown_unverified',job_id=job);preserve(teardown)
        else:pending.append('teardown_missing:'+job)
    complete=root/'collection-complete.json'
    if not complete.exists() or read(complete).get('collection_complete') is not True:pending.append('collection_not_complete')
    wandb_path=root/'wandb-current-readonly-verification.json';wandb=read(wandb_path) if wandb_path.exists() else {}
    require(plan['wandb_project']=='openwebrl-evals','wandb_project')
    if (wandb.get('state')!='finished' or wandb.get('project')!='openwebrl-evals'
            or wandb.get('id')!=plan['experiment_id'] or not complete.exists()
            or wandb.get('checked_unix',0)<complete.stat().st_mtime):pending.append('final_wandb_verification_required')
    if previous:
        for path,expected_hash in previous.get('immutable_artifacts',{}).items():
            require(Path(path).is_file() and sha(path)==expected_hash,'prior_audit_artifact_changed',path=path)
    primary=[x for x in rows if x['category']=='primary'];smokes=[x for x in rows if x['category']=='smoke']
    require(len(primary)<=300 and len(smokes)<=3,'result_cardinality_overflow')
    if len(primary)!=300 or len(smokes)!=3:pending.append('full_303_results_required')
    successes=sum(x['score']==1 for x in primary);valid=sum(x['valid'] for x in primary)
    ready=not issues and not pending
    return dict(created_utc=utc(),root=str(root),verified_complete=False,audit_passed=ready,
        status='ready_for_owner_verification' if ready else 'interim_unverified',issues=issues,pending=sorted(set(pending)),
        counts=dict(primary=len(primary),smoke=len(smokes),valid_primary=valid,invalid_primary=len(primary)-valid,
            completed_done=original_done,repaired_done=corrected,repaired_or_canonical_successes=successes,
            original_superseded_successes=sum(x['original_superseded_score']==1 for x in primary),
            final_screenshots=sum(bool(x['final_screenshot']) for x in primary),
            overall_success_rate=successes/300 if ready else None,valid_only_success_rate=successes/valid if ready and valid else None,
            terminals=dict(Counter(x['terminal'] for x in primary))),
        budget=dict(counters=counters,reconciled_counts=dict(counts),reconciled_charges=dict(charges),limits=plan['limits'],
            scheduler_attempts=scheduler,scheduler_seconds=elapsed,scheduler_cap=plan['resources']['total_seconds'],
            responses=dict(total_received),usage={k:dict(v) for k,v in token_usage.items()}),
        attempts=dict(Counter(a['status'] for a in attempts.values())),wandb=wandb,
        immutable_artifacts={p:h for p,h in artifacts.items() if Path(p).is_relative_to(root/'attempts')},
        frozen_source_and_control_proofs={p:h for p,h in artifacts.items() if not Path(p).is_relative_to(root/'attempts')},rows=rows)


def review_assets(report, directory):
    """Copy only intended review artifacts; relative HTML never escapes its served directory."""
    copied=json.loads(json.dumps(report));directory=Path(directory).resolve()
    for row in copied['rows']:
        dest=directory/'assets'/row['attempt_id'];dest.mkdir(parents=True,exist_ok=True)
        for field in ('result_path','final_screenshot','original_superseded_result'):
            value=row[field]
            if not value:continue
            original=Path(value if field=='result_path' else value['path'])
            expected=row['result_sha256'] if field=='result_path' else value['sha256']
            if sha(original)!=expected:raise ValueError('Review asset changed since audit')
            target=dest/('result.json' if field=='result_path' else 'original-superseded-result.json' if field=='original_superseded_result' else 'final.png')
            if target.exists():
                if sha(target)!=expected:raise ValueError('Existing review asset differs')
            else:shutil.copyfile(original,target)
            if sha(target)!=expected:raise ValueError('Review copy differs')
            if field=='result_path':row[field]=str(target)
            else:row[field]['path']=str(target)
    return copied


def render(report, target):
    target=Path(target); esc=html.escape
    def link(path,label):return '<a href="'+esc(os.path.relpath(path,target.parent),quote=True)+'">'+esc(label)+'</a>'
    cards=[]
    for row in report['rows']:
        operations='\n'.join(json.dumps(x,ensure_ascii=False) for x in row['operations']) or '(No browser operations dispatched.)'
        image=row['final_screenshot']; original=row.get('original_superseded_result'); status='invalid' if not row['valid'] else 'pending' if row['score'] is None else 'success' if row['score']==1 else 'failure'
        cards.append('<details class="task" data-status="'+status+'" data-category="'+row['category']+'"><summary><b>'+esc(row['category']+' · '+status+' · '+row['terminal'])+'</b> '+esc(row['task'] or row['task_id'])+'</summary><p>'+esc(row['judge_status'])+' · '+str(row['actions'])+' actions · '+str(row['decisions'])+' decisions'+(' · observed error: '+esc(row['actor_error']) if row['actor_error'] else '')+'</p><p>'+link(row['result_path'],'Immutable result')+' · Task '+esc(row['task_id'])+'</p>'+(('<a href="'+esc(os.path.relpath(image['path'],target.parent),quote=True)+'"><img loading="lazy" src="'+esc(os.path.relpath(image['path'],target.parent),quote=True)+'" alt="Final saved screenshot"></a>') if image else '<p>No final screenshot: preserved invalid observation.</p>')+'<h3>Executed native operations</h3><pre>'+esc(operations)+'</pre><details><summary>Jev choices and probabilities</summary><pre>'+esc(json.dumps(row['model_decisions'],indent=2,ensure_ascii=False))+'</pre></details><h3>'+('Original superseded verdict' if row['score'] is None and row['valid'] else 'Canonical verdict')+'</h3><pre>'+esc(row['judge_text'] or 'No judge request.')+'</pre>'+('<details><summary>Original superseded verdict (excluded from final score)</summary><p>'+link(original['path'],'Preserved original result')+'</p><pre>'+esc(original['judge_text'] or '')+'</pre></details>' if original else '')+'</details>')
    counts=report['counts'];summary=esc(json.dumps(counts,indent=2))
    return '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Local Jev actor · private review</title><style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;background:#f7f8fa;color:#18212b}h1{font-size:1.8rem}p{line-height:1.5}.note{padding:1rem;background:#fff2cd;border-left:4px solid #b7830a}details.task{background:white;border:1px solid #d6dce2;border-radius:8px;margin:.6rem 0;padding:1rem}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:1rem;font-size:12px}img{max-width:100%;border:1px solid #bbb}input,select{padding:.6rem;margin:.5rem}h3{font-size:1rem}</style><h1>Jev alone · local Online-Mind2Web</h1><p>Private evidence review · '''+esc(report['created_utc'])+'''</p><p class="note">'''+esc(report['status'])+'''. Original DONE judgments used malformed action input and are superseded. Saved-only repaired judgments use the same native operations and final screenshot; the actor and browser are not replayed. Observation failures remain invalid. Native Jev has no generated thoughts.</p><details open><summary>Audit counts</summary><pre>'''+summary+'''</pre></details><details><summary>Audit blockers and all-attempt budgets</summary><pre>'''+esc(json.dumps({k:report[k] for k in ('issues','pending','budget','attempts')},indent=2))+'''</pre></details><input id="query" placeholder="Search tasks or operation text"><select id="status"><option value="">All outcomes</option><option>success</option><option>failure</option><option>invalid</option><option>pending</option></select><select id="category"><option value="primary">Primary300</option><option value="smoke">Smokes</option><option value="">Both</option></select><span id="shown"></span>'''+''.join(cards)+'''<script>const q=document.querySelector('#query'),s=document.querySelector('#status'),c=document.querySelector('#category');function filter(){let n=0;for(const e of document.querySelectorAll('.task')){const yes=(!s.value||e.dataset.status===s.value)&&(!c.value||e.dataset.category===c.value)&&e.textContent.toLowerCase().includes(q.value.toLowerCase());e.hidden=!yes;if(yes)n++;}document.querySelector('#shown').textContent=n+' shown';}for(const x of [q,s,c])x.addEventListener('input',filter);filter();</script>'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--baseline',type=Path);p.add_argument('--refresh-scheduler',action='store_true')
    args=p.parse_args();root=args.root.resolve();out=args.output or root/'private-review'
    if out.resolve()==root or out.resolve().is_relative_to(root/'attempts'):raise ValueError('Use a separate review directory')
    scheduler=scheduler_records([str(x['job_id']) for x in read(root/'approval.json')['attempts']]) if args.refresh_scheduler else None
    report=audit(root,scheduler=scheduler,previous=read(args.baseline) if args.baseline else None)
    out.mkdir(parents=True,exist_ok=True);name='audit-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'
    with (out/name).open('x') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
    (out/'index.html').write_text(render(review_assets(report,out),out/'index.html'))
    print(json.dumps(dict(report=str(out/name),html=str(out/'index.html'),status=report['status'],counts=report['counts'],issue_count=len(report['issues']),pending_count=len(report['pending'])),indent=2))


if __name__=='__main__':main()
