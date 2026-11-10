#!/usr/bin/env python3
"""Run explicitly budget-approved GPT-5.5 labeling; persist every API response.

Stages: pilot, submit (Batch), collect (read-only poll), recover (known truncations).
Unknown request outcomes are never retried automatically.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

from prepare_arm_forward_transfer import CONTROL, REPO, rows, sha, write
from openwebrl.arm_inference import parse_selection
from arm_refresh_label_recovery import recovered_label,recover,known_length,original_responses


def file_sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def parsed_label(request,body):
    if not body.get('model','').startswith('gpt-5.5') or body['model'].startswith('gpt-5.5-pro'):
        raise ValueError('Teacher model differs from approved GPT-5.5')
    choice=body['choices'][0]
    if choice['finish_reason']!='stop' or choice['message'].get('refusal'):
        raise ValueError('Incomplete or refused teacher output')
    text=choice['message'].get('content') or ''
    # The public no-CoT prompt requests one JSON object, never a fallback index.
    value=json.loads(text)
    if set(value)!={'selection'} or type(value['selection']) is not int:
        raise ValueError('Teacher did not return the expected selection object')
    selected=parse_selection(text,5)
    usage=body['usage']
    price=(usage['prompt_tokens']*5+usage['completion_tokens']*30)/1e6
    return dict(custom_id=request['custom_id'],state_id=request['state_id'],split=request['split'],
                treatment=request['treatment'],selected_slot=selected,
                selected_original=request['permutation'][selected],permutation=request['permutation'],
                model=body['model'],usage=usage,usd=price*(1 if request['pilot'] else .5),raw=text)


def run(stage,root,approval_path):
    root=Path(root)
    pause=root/'external-transfer-approval.json'
    if pause.exists() and json.loads(pause.read_text()).get('approved') is not True:
        raise ValueError('Explicit payload and OpenAI destination approval is pending')
    manifest=json.loads((root/'manifest.json').read_text())
    if (CONTROL/'active-proposal.json').exists():
        proposal=json.loads((CONTROL/'active-proposal.json').read_text())
        if proposal.get('superseded_teacher_requests')==root.name:
            raise ValueError('Superseded additive requests; build baseline40 requests before labeling')
    approval=json.loads(approval_path.read_text())
    if (approval.get('teacher_approved') is not True or approval.get('model')!='gpt-5.5'
        or approval.get('requests_manifest_sha256')!=sha((root/'manifest.json').read_bytes())
        or float(approval.get('api_budget_usd',0))<manifest['cost_ceiling_usd']):
        raise ValueError('Exact model, request manifest and labeling budget approval required')
    for name,meta in manifest['files'].items():
        if file_sha(root/name)!=meta['sha256']:raise ValueError('Requests changed')
    index={r['custom_id']:r for r in rows(root/'index.jsonl')}
    with (root/'api.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from dotenv import dotenv_values
        from openai import OpenAI
        api_key=dotenv_values(REPO/'.env').get('OPENAI_API_KEY')
        if not api_key:raise ValueError('Project .env is missing OPENAI_API_KEY')
        client=OpenAI(api_key=api_key,base_url='https://api.openai.com/v1',max_retries=0,timeout=180)
        rawdir=root/'responses';rawdir.mkdir(exist_ok=True)
        if stage=='pilot':
            for request in rows(root/'pilot.jsonl'):
                cid=request['custom_id'];dest=rawdir/f'{cid}.json';ticket=rawdir/f'{cid}.intent.json'
                if dest.exists():
                    body=json.loads(dest.read_text())
                    if known_length(body):recover(root,index[cid],request,body,manifest,approval,client,parsed_label)
                    else:parsed_label(index[cid],body)
                    continue
                if ticket.exists():raise ValueError('Unresolved pilot request; reconcile before retry')
                write(ticket,dict(custom_id=cid,usd_reserved=index[cid]['usd_reserved']))
                body=client.chat.completions.create(**request['body']).model_dump(mode='json')
                write(dest,body)
                if known_length(body):recover(root,index[cid],request,body,manifest,approval,client,parsed_label)
                else:parsed_label(index[cid],body)
            write(root/'pilot-passed.json',dict(passed=True,requests=20))
        elif stage=='submit':
            if not (root/'pilot-passed.json').exists():raise ValueError('Pilot gate has not passed')
            for path in sorted(root.glob('batch-*.jsonl')):
                receipt=root/f'{path.stem}.submitted.json';intent=root/f'{path.stem}.intent.json'
                if receipt.exists():continue
                if intent.exists():raise ValueError('Unresolved batch creation; reconcile before retry')
                uploaded_receipt=root/f'{path.stem}.uploaded.json'
                if uploaded_receipt.exists():
                    uploaded=json.loads(uploaded_receipt.read_text())
                    if uploaded['request_file_sha256']!=manifest['files'][path.name]['sha256']:
                        raise ValueError('Uploaded request file changed')
                    input_id=uploaded['input_file_id']
                else:
                    with path.open('rb') as f:uploaded=client.files.create(file=f,purpose='batch')
                    input_id=uploaded.id
                    write(uploaded_receipt,dict(input_file_id=input_id,request_file_sha256=manifest['files'][path.name]['sha256'],
                        api_file=uploaded.model_dump(mode='json')))
                write(intent,dict(input_file_id=input_id,request_file_sha256=manifest['files'][path.name]['sha256']))
                batch=client.batches.create(input_file_id=input_id,endpoint='/v1/chat/completions',
                                            completion_window='24h',metadata={'experiment':'arm-refresh-20260924','chunk':path.stem})
                write(receipt,batch.model_dump(mode='json'))
        elif stage=='collect':
            for receipt in sorted(root.glob('batch-*.submitted.json')):
                batch=client.batches.retrieve(json.loads(receipt.read_text())['id'])
                write(root/f'{receipt.stem}.status.json',batch.model_dump(mode='json'))
                for kind in ('output','error'):
                    ident=getattr(batch,f'{kind}_file_id',None)
                    if not ident:continue
                    dest=rawdir/f'{batch.id}.{kind}.jsonl'
                    if not dest.exists():dest.write_text(client.files.content(ident).text)
        elif stage=='recover':
            originals=original_responses(root,index);needed={}
            for cid,body in originals.items():
                if not known_length(body):continue
                try:recovered_label(root,index[cid],body,parsed_label)
                except ValueError as exc:
                    if str(exc)!='Known token-limit failure awaits bounded recovery':raise
                    needed[cid]=body
            # Scan each large image-bearing request file only once.
            for name in sorted({index[cid]['file'] for cid in needed}):
                for request in rows(root/name):
                    cid=request['custom_id']
                    if cid in needed:recover(root,index[cid],request,needed[cid],manifest,approval,client,parsed_label)
        labels={};errors=[]
        for cid,r in index.items():
            path=rawdir/f'{cid}.json'
            if path.exists():
                try:labels[cid]=recovered_label(root,r,json.loads(path.read_text()),parsed_label)
                except (ValueError,KeyError,IndexError) as exc:errors.append(dict(custom_id=cid,error=str(exc)))
        for path in rawdir.glob('batch_*.output.jsonl'):
            for response in rows(path):
                cid=response['custom_id']
                if cid not in index or cid in labels:raise ValueError('Unknown or duplicate teacher response')
                try:
                    if response.get('error') or response['response']['status_code']!=200:raise ValueError('API request failure')
                    labels[cid]=recovered_label(root,index[cid],response['response']['body'],parsed_label)
                except (ValueError,KeyError,IndexError) as exc:errors.append(dict(custom_id=cid,error=str(exc)))
        for path in rawdir.glob('batch_*.error.jsonl'):
            for response in rows(path):
                cid=response.get('custom_id')
                if cid not in index:raise ValueError('Unknown teacher error response')
                errors.append(dict(custom_id=cid,error='Batch request error',detail=response.get('error')))
        (root/'labels.jsonl').write_text(''.join(json.dumps(labels[k])+'\n' for k in sorted(labels)))
        originals=original_responses(root,index)
        billed=sum((b['usage']['prompt_tokens']*5+b['usage']['completion_tokens']*30)/1e6*(1 if index[cid]['pilot'] else .5) for cid,b in originals.items())
        for path in (rawdir/'length-recovery').glob('*.json'):
            if path.name.endswith('.intent.json'):continue
            b=json.loads(path.read_text());billed+=(b['usage']['prompt_tokens']*5+b['usage']['completion_tokens']*30)/1e6
        extra_reserved=sum(json.loads(p.read_text())['usd_reserved'] for p in (rawdir/'length-recovery').glob('*.intent.json'))
        status=dict(stage=stage,valid=len(labels),expected=len(index),complete=len(labels)==len(index),
                    known_successful_request_cost_usd=sum(x['usd'] for x in labels.values()),errors=errors,
                    known_billed_upper_bound_usd=billed,total_reserved_usd=manifest['cost_ceiling_usd']+extra_reserved,
                    recovered_length_labels=sum(bool(x.get('recovered_from_length')) for x in labels.values()),
                    labels_sha256=sha((root/'labels.jsonl').read_bytes()))
        write(root/'label-status.json',status);print(json.dumps(status,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['pilot','submit','collect','recover'])
    p.add_argument('--root',type=Path,default=CONTROL/'teacher-v1')
    p.add_argument('--approval',type=Path,default=CONTROL/'approval.json')
    a=p.parse_args();run(a.stage,a.root,a.approval)
