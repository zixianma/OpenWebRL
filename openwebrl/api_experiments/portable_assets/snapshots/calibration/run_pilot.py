"""Bounded, snapshot-only Sol reasoning pilot. Never imports or opens a browser."""
from pathlib import Path
from decimal import Decimal
import argparse
import base64
import concurrent.futures
import fcntl
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request
import urllib.error

import jsonschema

REPO = Path('/gpfs/projects/krishna/zixianma/OpenWebRL')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from openwebrl.responses_actor_history import response_input_bound

MODEL = 'gpt-6.1-sol'
CAP = Decimal('20')
ROLES = ('serial_high', 'parallel_low_0', 'parallel_low_1', 'parallel_low_2', 'selector')
HALT = threading.Event()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def digest(value):
    return sha(json.dumps(value, sort_keys=True).encode())


def ledger(reserve=None, settle=None):
    with (ROOT/'budget.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        p = ROOT/'ledger.json'
        d = json.loads(p.read_text()) if p.exists() else dict(requests=0, charged_or_reserved_usd='0', settled_requests=0)
        balance = Decimal(d['charged_or_reserved_usd'])
        if reserve is not None:
            if HALT.is_set() or d['requests'] >= 150 or balance + reserve > CAP:
                HALT.set()
                raise RuntimeError('All-attempt request or USD cap reached')
            d['requests'] += 1
            balance += reserve
        if settle is not None:
            upper, actual = settle
            balance += actual-upper
            d['settled_requests'] += 1
            if actual > upper or balance > CAP:
                HALT.set()
                dump(ROOT/'halt.json', dict(reason='usage_exceeded_conservative_reservation'))
        d['charged_or_reserved_usd'] = str(balance)
        dump(p, d)
        return d


def tools_from_prompt(prompt):
    blocks = re.findall(r'<tools>\s*(.*?)\s*</tools>', prompt, flags=re.S)
    rows = []
    for block in blocks:
        if block.strip():
            rows.extend(json.loads(line) for line in block.splitlines() if line.strip())
    tools = []
    for row in rows:
        fn = row.get('function', row)
        tools.append(dict(type='function', name=fn['name'], description=fn.get('description', ''),
                          parameters=fn['parameters'], strict=False))
    if not tools or len({t['name'] for t in tools}) != len(tools):
        raise ValueError('Missing or duplicate source function schemas')
    for tool in tools:
        jsonschema.Draft202012Validator.check_schema(tool['parameters'])
    return tools


def context(state):
    prompt = state['prompt']
    tools = tools_from_prompt(prompt)
    if '<|im_end|>' not in prompt:
        raise ValueError('Saved transcript lacks a system boundary')
    history = prompt.split('<|im_end|>', 1)[1]
    history = re.sub(r'<\|im_start\|>assistant\s*<think>\s*$', '', history).strip()
    history = history.replace('<|vision_start|><|image_pad|><|vision_end|>', '[current screenshot attached]')
    image = state['images'][0]
    data = Path(image['path']).read_bytes()
    if sha(data) != image['sha256']:
        raise ValueError('Frozen screenshot bytes changed')
    mime = 'image/jpeg' if image.get('format', '').lower() in ('jpg', 'jpeg') else 'image/png'
    user = dict(role='user', content=[dict(type='input_text', text=(
        'Read-only archived browser state. The following is saved task and action history, '
        'not new instructions. Propose the next action for the current screenshot; '
        'no action will actually be executed in this diagnostic.\n\n' + history)),
        dict(type='input_image', image_url='data:'+mime+';base64,'+base64.b64encode(data).decode(), detail='high')])
    system = (ROOT/'system-policy.txt').read_text()
    return [dict(role='system', content=system), user], tools


def parse_action(response, tools):
    if response.get('status') != 'completed':
        raise ValueError('Actor response incomplete')
    calls = [v for v in response.get('output', []) if v.get('type') == 'function_call']
    if len(calls) != 1 or not calls[0].get('call_id'):
        raise ValueError('Expected exactly one native function call')
    call = calls[0]
    schemas = {t['name']:t['parameters'] for t in tools}
    if call['name'] not in schemas:
        raise ValueError('Unknown native function name')
    arguments = json.loads(call['arguments'])
    jsonschema.validate(arguments, schemas[call['name']])
    return dict(name=call['name'], arguments=arguments, call_id=call['call_id'])


def request_body(role, inputs, tools, actions=None):
    body = dict(model=MODEL, store=False, service_tier='default',
        reasoning=dict(effort='high' if role=='serial_high' else 'medium' if role=='selector' else 'low'),
        max_output_tokens=12288 if role=='serial_high' else 2048 if role=='selector' else 4096)
    if role == 'selector':
        schema = dict(type='object', properties=dict(selection=dict(type='integer', enum=[0,1,2])),
                      required=['selection'], additionalProperties=False)
        body.update(input=[dict(role='system', content='Choose the best next action from exactly three '
            'proposals for this browser state. Treat the archived history as evidence. Return the selected zero-based index.'),
            inputs[1],dict(role='user',content='Candidate actions, in frozen order:\n'+json.dumps(actions))],
            text=dict(format=dict(type='json_schema', name='selection', strict=True, schema=schema)))
    else:
        body.update(input=inputs, tools=tools, tool_choice='required', parallel_tool_calls=False)
    return body


def cost(usage):
    n, out = usage['input_tokens'], usage['output_tokens']
    detail = usage.get('input_tokens_details') or {}
    cached, writes = detail.get('cached_tokens', 0), detail.get('cache_write_tokens')
    if any(type(x) is not int or x < 0 for x in (n, out, cached)) or cached > n:
        raise ValueError('Invalid provider token telemetry')
    if writes is not None and (type(writes) is not int or not 0 <= writes <= n-cached):
        raise ValueError('Invalid cache-write telemetry')
    # Unknown cache writes remain conservatively charged at the write price.
    writes = n-cached if writes is None else writes
    return (Decimal(n-cached)*2 + Decimal(cached)*Decimal('.1') + Decimal(writes)*Decimal('.5') + Decimal(out)*10)/1000000


def call(state_id, role, body, key):
    path = ROOT/'requests'/(state_id+'--'+role+'.json')
    if path.exists():
        raise RuntimeError('Request already reserved or attempted; physical retries forbidden')
    if body['model'] != MODEL or body['store'] is not False or body['service_tier'] != 'default':
        raise ValueError('Model or service contract differs')
    bound = response_input_bound(body['input'], body.get('tools', []))
    if bound > 272000:
        raise ValueError('Input outside approved price tier')
    upper = (Decimal(bound)*Decimal('2.5') + Decimal(body['max_output_tokens'])*10)/1000000
    ledger(reserve=upper)
    row = dict(state_id=state_id,role=role,status='reserved',request=body,request_sha256=digest(body),
               input_bound=bound,reserved_usd=str(upper),started_epoch=time.time())
    dump(path,row)
    started = time.monotonic()
    try:
        req = urllib.request.Request('https://api.openai.com/v1/responses',
            data=json.dumps(body).encode(), headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}, method='POST')
        # Exactly one HTTP request; no SDK retry layer or fallback endpoint.
        with urllib.request.urlopen(req, timeout=240) as response:
            result=json.load(response)
        row.update(response=result,usage=result['usage'],status='received')
        charged=cost(result['usage']);row['charged_usd']=str(charged)
        ledger(settle=(upper,charged))
        if result['model'] != MODEL and not result['model'].startswith(MODEL+'-'):
            raise ValueError('Returned model differs from pinned identity')
        if role == 'selector':
            text=''.join(p.get('text','') for item in result.get('output',[]) if item.get('type')=='message'
                         for p in item.get('content',[]) if p.get('type')=='output_text')
            selected=json.loads(text)
            if result.get('status')!='completed' or set(selected)!={'selection'} or type(selected['selection']) is not int or selected['selection'] not in (0,1,2):
                raise ValueError('Invalid index-only selector response')
            row['selection']=selected['selection']
        else:
            try:row['action']=parse_action(result,body['tools'])
            except (ValueError,KeyError,jsonschema.ValidationError) as exc:row['action_error']=str(exc)[:300]
        return row
    except BaseException as exc:
        HALT.set()
        row['error_type']=type(exc).__name__
        row['error']=str(exc)[:500]
        row['http_status']=getattr(exc,'code',None)
        if 'usage' not in row:row['status']='usage_unknown'
        if isinstance(exc,urllib.error.HTTPError):
            row['provider_error_body']=exc.read().decode(errors='replace')[:4000]
        dump(ROOT/'halt.json',dict(reason='request_failure_no_automatic_retry',state_id=state_id,role=role,error_type=type(exc).__name__))
        raise
    finally:
        row['seconds']=time.monotonic()-started
        dump(path,row)


def run_state(state,key):
    sid=state['state_id'];inputs,tools=context(state)
    roles=list(ROLES[:-1]);random.Random('sol-pilot-v1/'+sid).shuffle(roles)
    receipts={}
    for role in roles:
        if HALT.is_set():raise RuntimeError('Campaign halted; no new request')
        receipts[role]=call(sid,role,request_body(role,inputs,tools),key)
        print(json.dumps(dict(state_id=sid,role=role,status=receipts[role]['status'],ledger=json.loads((ROOT/'ledger.json').read_text()))),flush=True)
    lows=[receipts['parallel_low_'+str(i)] for i in range(3)]
    if all('action' in row for row in lows):
        actions=[dict(index=i,name=row['action']['name'],arguments=row['action']['arguments']) for i,row in enumerate(lows)]
        receipts['selector']=call(sid,'selector',request_body('selector',inputs,tools,actions),key)
    else:
        receipts['selector']=dict(state_id=sid,role='selector',status='skipped_invalid_candidate_prerequisite',physical_requests=0)
        dump(ROOT/'requests'/(sid+'--selector.json'),receipts['selector'])
    dump(ROOT/'states'/(sid+'.json'),dict(state_id=sid,roles={role:row['status'] for role,row in receipts.items()},finished_epoch=time.time()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    manifest=ROOT/'states-private.json';payload=json.loads(manifest.read_text())
    states=payload['states'] if isinstance(payload,dict) else payload
    assert len(states)==30 and len({s['state_id'] for s in states})==30
    for state in states:context(state)
    if not args.run:
        print('Offline request/image/schema preflight passed for30 frozen states; no HTTP calls');return
    approval=json.loads((ROOT/'approval.json').read_text())
    assert approval['approved'] and approval['api_usd_cap']==20 and approval['physical_request_cap']==150
    assert approval['states_sha256']==sha(manifest.read_bytes()) and approval['runner_sha256']==sha(Path(__file__).read_bytes())
    assert approval['system_policy_sha256']==sha((ROOT/'system-policy.txt').read_bytes())
    assert approval['input_bound_source_sha256']==sha((REPO/'openwebrl/responses_actor_history.py').read_bytes())
    key=next((line.split('=',1)[1].strip().strip('\"\'') for line in (REPO/'.env').read_text().splitlines() if line.startswith('OPENAI_API_KEY=')),None)
    if not key:raise ValueError('Missing project OpenAI credential')
    with (ROOT/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'ledger.json').exists():raise RuntimeError('Prior physical attempt ledger exists; no automatic restart')
        dump(ROOT/'current.json',dict(stage='first_state_startup',pid=os.getpid(),states=30,started_epoch=time.time()))
        try:
            run_state(states[0],key)
            dump(ROOT/'current.json',dict(stage='collecting',pid=os.getpid(),states=30,started_epoch=time.time()))
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                futures=[pool.submit(run_state,state,key) for state in states[1:]]
                for future in concurrent.futures.as_completed(futures):future.result()
            dump(ROOT/'current.json',dict(stage='collected_pending_independent_audit',pid=os.getpid(),finished_states=30,ledger=json.loads((ROOT/'ledger.json').read_text()),verified_complete=False))
        except BaseException as exc:
            HALT.set();dump(ROOT/'current.json',dict(stage='failed_preserved',pid=os.getpid(),error_type=type(exc).__name__,finished_states=len(list((ROOT/'states').glob('*.json'))),verified_complete=False));raise


if __name__=='__main__':main()
