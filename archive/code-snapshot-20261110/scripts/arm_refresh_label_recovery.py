"""Recover only recorded token-limit failures, within the original API cap."""
import copy
import json
from pathlib import Path
from prepare_arm_forward_transfer import rows,sha,write

CAPS=(4096,8192)


def known_length(body):
    choices=body.get('choices',[])
    return (body.get('model','').startswith('gpt-5.5') and not body['model'].startswith('gpt-5.5-pro')
        and len(choices)==1 and choices[0].get('finish_reason')=='length'
        and not choices[0].get('message',{}).get('refusal') and 'usage' in body)


def paths(root,cid,cap):
    directory=Path(root)/'responses/length-recovery'
    return directory/f'{cid}-{cap}.json',directory/f'{cid}-{cap}.intent.json'


def recovered_label(root,entry,original,parse):
    try:return parse(entry,original)
    except (ValueError,KeyError,IndexError):
        if not known_length(original):raise
    for cap in CAPS:
        path,ticket=paths(root,entry['custom_id'],cap)
        if not path.exists():continue
        receipt=json.loads(ticket.read_text())
        if receipt['original_response_sha256']!=sha(json.dumps(original,sort_keys=True).encode()):
            raise ValueError('Recovery original response changed')
        body=json.loads(path.read_text())
        try:label=parse(dict(entry,pilot=True),body)
        except (ValueError,KeyError,IndexError):
            if known_length(body):continue
            raise
        label.update(recovered_from_length=True,recovery_completion_cap=cap,
                     original_response_sha256=receipt['original_response_sha256'])
        return label
    raise ValueError('Known token-limit failure awaits bounded recovery')


def recover(root,entry,request,original,manifest,approval,client,parse):
    """Caller owns api.lock; durable intents reserve even unknown-outcome costs."""
    if not known_length(original):raise ValueError('Only known token-limit failures may be retried')
    if sha(json.dumps(request['body'],sort_keys=True).encode())!=entry['body_sha256']:
        raise ValueError('Original request differs from frozen manifest')
    for cap in CAPS:
        try:return recovered_label(root,entry,original,parse)
        except ValueError as exc:
            if str(exc)!='Known token-limit failure awaits bounded recovery':raise
        path,ticket=paths(root,entry['custom_id'],cap)
        if path.exists():continue
        if ticket.exists():raise ValueError('Unknown recovery outcome; manual reconciliation required')
        reserved=sum(json.loads(p.read_text())['usd_reserved'] for p in
                     (Path(root)/'responses/length-recovery').glob('*.intent.json'))
        cost=(entry['input_tokens_reserved']*5+cap*30)/1e6
        total=manifest['cost_ceiling_usd']+reserved+cost
        if total>float(approval['api_budget_usd']):raise ValueError('Recovery would exceed approved total API budget')
        body=copy.deepcopy(request['body']);body['max_completion_tokens']=cap
        write(ticket,dict(custom_id=entry['custom_id'],max_completion_tokens=cap,usd_reserved=cost,
            total_reserved_usd=total,original_response_sha256=sha(json.dumps(original,sort_keys=True).encode()),
            request_sha256=sha(json.dumps(body,sort_keys=True).encode()),reason='Known finish_reason=length; no label was inferred'))
        response=client.chat.completions.create(**body).model_dump(mode='json')
        write(path,response)
    return recovered_label(root,entry,original,parse)


def original_responses(root,index):
    """Return known HTTP200 bodies; retain separate API error records unchanged."""
    found={}
    for cid in index:
        path=Path(root)/'responses'/f'{cid}.json'
        if path.exists():found[cid]=json.loads(path.read_text())
    for path in (Path(root)/'responses').glob('batch_*.output.jsonl'):
        for row in rows(path):
            cid=row['custom_id']
            if cid not in index:raise ValueError('Unknown teacher response')
            if row.get('error') or row['response']['status_code']!=200:continue
            if cid in found:raise ValueError('Duplicate original teacher response')
            found[cid]=row['response']['body']
    return found
