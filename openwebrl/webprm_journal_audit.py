"""Read-only financial and per-cell call audit for prospective broker v2.

Only explicit, hash-bound file prefixes are consumed. No budget writer, lock,
provider, scheduler, browser or export is opened. Verification concerns saved
accounting/evidence consistency, not provider semantic truth or process closure.
"""
from __future__ import annotations

from decimal import Decimal,localcontext
import hashlib
import json
import math
from pathlib import Path


def _encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _digest(value):return hashlib.sha256(_encoded(value)).hexdigest()


def _sha(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _money(value):
    if isinstance(value,bool) or not isinstance(value,(str,int,float,Decimal)):raise ValueError('Invalid money value')
    amount=Decimal(str(value))
    if not amount.is_finite() or amount<0:raise ValueError('Finite nonnegative money required')
    return amount


def _sum(*values):
    with localcontext() as context:
        context.prec=max(1,max(v.adjusted() for v in values)-min(v.as_tuple().exponent for v in values)+3)
        return sum(values,Decimal(0))


def _product(*values):
    with localcontext() as context:
        context.prec=max(1,sum(len(v.as_tuple().digits) for v in values))
        result=Decimal(1)
        for value in values:result*=value
        return result


def _prefix(path,size,expected):
    if type(size) is not int or size<0 or not _sha(expected):raise ValueError('Explicit byte count/SHA256 required')
    remaining=size;position=0;hasher=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while remaining:
            raw=stream.readline(remaining+1)
            if not raw or len(raw)>remaining or not raw.endswith(b'\n'):
                raise ValueError('Bound prefix is truncated or ends inside a record')
            hasher.update(raw);position+=len(raw);remaining-=len(raw)
            row=json.loads(raw)
            if not isinstance(row,dict):raise ValueError('Journal records must be objects')
            yield row,position,hasher.hexdigest()
    if hasher.hexdigest()!=expected:raise ValueError('Bound prefix SHA256 mismatch')


def _api_chunks(path):
    chunks=[];hasher=hashlib.sha256();position=0
    if not Path(path).exists():return chunks,hasher.hexdigest(),0
    with Path(path).open('rb') as stream:
        for raw in stream:
            if not raw.endswith(b'\n'):raise ValueError('Incomplete API journal line')
            hasher.update(raw);position+=len(raw);row=json.loads(raw)
            if not isinstance(row,dict):raise ValueError('Malformed API event')
            if row.get('event')=='request':
                chunks.append(dict(request=row,response=None,error=None))
            elif row.get('event') in ('response','response_error'):
                if not chunks or chunks[-1]['response'] is not None or chunks[-1]['error'] is not None:
                    raise ValueError('Unmatched or duplicate API response')
                chunk=chunks[-1]
                if row.get('reservation')!=chunk['request'].get('reservation'):raise ValueError('API reservation identity changed')
                chunk['response' if row['event']=='response' else 'error']=row
                chunk['terminal_prefix_sha256']=hasher.hexdigest()
                chunk['terminal_prefix_bytes']=position
            else:raise ValueError('Unexpected API event; no request may be silently discarded')
    return chunks,hasher.hexdigest(),position


def _financial(path,size,expected,contract,receipt_offsets):
    cap=_money(contract['api_usd_cap']);charged=Decimal(0);halt=None
    calls={};dispatches={};epochs={};active=None;last_closed=None;counts={False:0,True:0}
    sequence=0;previous='0'*64;prefixes={};header=False
    for event,position,prefix_sha in _prefix(path,size,expected):
        core={k:v for k,v in event.items() if k!='event_sha256'}
        if (event.get('sequence')!=sequence or event.get('previous_sha256')!=previous or event.get('event_sha256')!=_digest(core)):
            raise ValueError('Financial event hash chain/sequence mismatch')
        sequence+=1;previous=event['event_sha256']
        if position in receipt_offsets:prefixes[position]=prefix_sha
        kind=event.get('event')
        if not header:
            if (kind!='header' or event.get('version')!='single-owner-journal-v2' or event.get('financial_contract')!=contract or
                    event.get('policy_sha256')!=_digest(contract) or _money(event.get('cap'))!=cap):
                raise ValueError('Stable financial header/contract mismatch')
            header=True
        elif kind=='epoch_started':
            ident=event.get('epoch_id');sources=event.get('implementation_sha256');deadline=event.get('deadline')
            if (active is not None or not isinstance(ident,str) or not ident or ident in epochs or
                    event.get('previous_closed_epoch_sha256')!=last_closed or not _sha(event.get('broker_policy_sha256')) or
                    not _sha(event.get('registered_cells_sha256')) or type(deadline) not in (int,float) or
                    not math.isfinite(deadline) or deadline<=0 or not isinstance(sources,dict) or not sources or
                    any(not isinstance(k,str) or not k or not _sha(v) for k,v in sources.items())):
                raise ValueError('Invalid broker epoch lineage/policy')
            epochs[ident]=event;active=ident
        elif kind=='dispatch':
            meta=event.get('dispatch',{});ident=meta.get('request_id');local=meta.get('local')
            if (active is None or event.get('epoch_id')!=active or not isinstance(ident,str) or len(ident)!=32 or
                    any(c not in '0123456789abcdef' for c in ident) or ident in dispatches or type(local) is not bool or
                    meta.get('policy_sha256')!=epochs[active]['broker_policy_sha256'] or
                    not isinstance(meta.get('cell_id'),str) or not meta['cell_id'] or not isinstance(meta.get('model'),str) or not meta['model'] or
                    not local and meta['model'] not in contract['prices_usd_per_token']):
                raise ValueError('Invalid durable dispatch identity')
            if len(dispatches)>=contract['max_dispatches'] or counts[local]>=contract['max_local_dispatches' if local else 'max_paid_dispatches']:
                raise ValueError('Cumulative dispatch cap exceeded')
            dispatches[ident]=dict(meta,epoch_id=active,reservation_id=None);counts[local]+=1
        elif kind=='reserve':
            ident=event.get('id');request=dispatches.get(event.get('request_id'));amount=_money(event.get('amount'))
            if (type(ident) is not int or ident!=len(calls) or request is None or active is None or request['epoch_id']!=active or
                    request['local'] or request['reservation_id'] is not None or request['cell_id']!=event.get('cell_id') or
                    request['model']!=event.get('model') or halt or _sum(charged,amount)>cap):
                raise ValueError('Unadmitted, duplicate or over-cap reservation')
            calls[ident]=dict(id=ident,request_id=event['request_id'],model=event['model'],cell_id=event['cell_id'],
                reserved=amount,state='reserved',charged=amount)
            request['reservation_id']=ident;charged=_sum(charged,amount)
        elif kind=='settle':
            ident=event.get('id');row=calls.get(ident);actual=_money(event.get('actual'));usage=event.get('usage')
            if (type(ident) is not int or row is None or row['state']!='reserved' or active is None or
                    dispatches[row['request_id']]['epoch_id']!=active or not isinstance(usage,dict) or
                    any(type(usage.get(k)) is not int or usage[k]<0 for k in ('prompt_tokens','completion_tokens'))):
                raise ValueError('Invalid settlement or usage identity')
            charged=_sum(charged,actual,row['reserved'].copy_negate())
            overrun=actual>row['reserved']
            row.update(charged=actual,usage=usage,state='overrun' if overrun else 'settled',settlement_event_sha256=event['event_sha256'],
                       settlement_prefix_bytes=position)
            if overrun:halt='observed_usage_exceeded_reservation'
        elif kind=='halt':
            if not isinstance(event.get('reason'),str) or not event['reason']:raise ValueError('Invalid halt reason')
            halt=event['reason']
        elif kind=='epoch_closed':
            if (active is None or event.get('epoch_id')!=active or not isinstance(event.get('audit_path'),str) or
                    type(event.get('audit_prefix_bytes')) is not int or event['audit_prefix_bytes']<0 or not _sha(event.get('audit_sha256'))):
                raise ValueError('Invalid epoch closure')
            active=None;last_closed=event['event_sha256']
        else:raise ValueError('Unknown or repeated financial event')
    if not header:raise ValueError('Financial header missing')
    if set(receipt_offsets)-set(prefixes):raise ValueError('Completion receipt is outside the bound financial prefix')
    return dict(calls=calls,dispatches=dispatches,epochs=epochs,prefixes=prefixes,charged=charged,cap=cap,halt=halt,
        counts=counts,active_epoch=active,events=sequence,overruns=sum(r['state']=='overrun' for r in calls.values()),
        unknown=sum(r['state']=='reserved' for r in calls.values()))


def audit_cell_calls(plan,cell_id,api_path,broker_audit_path,budget_event_path,budget_prefix_bytes,
        budget_prefix_sha256,financial_contract,*,broker_prefix_bytes,broker_prefix_sha256,
        interrupted_local_request_ids=(), interrupted_paid_request_ids=()):
    """Return internal request/response pairs only after exact-prefix accounting.

    The caller must verify task membership, source/plan freeze, runtime closure
    and provider semantics separately. Returned pairs contain private payloads
    for native replay and must never be copied into a public results document.

    A dedicated interruption auditor may explicitly identify at most one local
    or paid dispatch lost at verified owner shutdown. A paid dispatch must retain
    its exact unsettled reservation in full. That caller must bind the recovered
    epoch and closure proofs. This opt-in never supplies a missing response or
    terminal broker event, and cannot authorize native success/disposal.
    """
    if (not isinstance(interrupted_local_request_ids,(tuple,list)) or len(interrupted_local_request_ids)>1 or
            any(not isinstance(x,str) or len(x)!=32 or any(c not in '0123456789abcdef' for c in x)
                for x in interrupted_local_request_ids)):
        raise ValueError('At most one exact interrupted local request identity is allowed')
    if (not isinstance(interrupted_paid_request_ids,(tuple,list)) or len(interrupted_paid_request_ids)>1 or
            any(not isinstance(x,str) or len(x)!=32 or any(c not in '0123456789abcdef' for c in x)
                for x in interrupted_paid_request_ids) or
            len(interrupted_local_request_ids)+len(interrupted_paid_request_ids)>1):
        raise ValueError('At most one exact interrupted request per serial cell is allowed')
    interrupted_local=set(interrupted_local_request_ids)
    interrupted_paid=set(interrupted_paid_request_ids)
    interrupted=interrupted_local|interrupted_paid
    required={'schema_version','budget_backend','cohort_sha256','protocol_sha256','prices_usd_per_token',
              'api_usd_cap','max_dispatches','max_paid_dispatches','max_local_dispatches'}
    if (not isinstance(financial_contract,dict) or set(financial_contract)!=required or
            financial_contract.get('schema_version')!=1 or financial_contract.get('budget_backend')!='single-owner-journal-v2' or
            not _sha(financial_contract.get('cohort_sha256')) or not _sha(financial_contract.get('protocol_sha256')) or
            _digest(plan.get('protocol_contract'))!=financial_contract['protocol_sha256']):
        raise ValueError('Exact frozen protocol and financial contract required')
    for name in ('max_dispatches','max_paid_dispatches','max_local_dispatches'):
        if type(financial_contract[name]) is not int or not 0<=financial_contract[name]<=financial_contract['max_dispatches']:
            raise ValueError('Invalid cumulative dispatch caps')
    prices=financial_contract['prices_usd_per_token']
    if not isinstance(prices,dict) or not prices or any(not isinstance(k,str) or not k or not isinstance(v,list) or len(v)!=2 or
            any(_money(x)<=0 for x in v) for k,v in prices.items()):raise ValueError('Fixed positive token prices required')
    if (_money(plan.get('api_usd_cap'))!=_money(financial_contract['api_usd_cap']) or
            plan.get('prices_usd_per_token')!=prices):
        raise ValueError('Actual worker cap/prices differ from stable financial contract')
    identity=_digest(plan);broker={};offsets=set();broker_policies=set()
    for row,_,_ in _prefix(broker_audit_path,broker_prefix_bytes,broker_prefix_sha256):
        if not _sha(row.get('policy_sha256')):raise ValueError('Broker record lacks an exact epoch policy')
        broker_policies.add(row['policy_sha256'])
        if row.get('event') in ('epoch_started','epoch_closed'):
            continue
        if row.get('cell_id')!=cell_id:continue
        kind=row.get('event');ident=row.get('request_id')
        if kind not in ('dispatch_started','dispatch_completed','dispatch_failed') or not isinstance(ident,str):
            raise ValueError('Unknown target-cell broker event')
        if row.get('identity_sha256')!=identity:raise ValueError('Mixed target-cell worker identities')
        group=broker.setdefault(ident,{})
        if kind in group:raise ValueError('Duplicate target-cell broker event')
        group[kind]=row
        if kind=='dispatch_completed' and not row.get('local'):
            offset=row.get('budget_prefix_bytes')
            if type(offset) is not int or offset<=0:raise ValueError('Missing financial receipt prefix')
            offsets.add(offset)
    state=_financial(budget_event_path,budget_prefix_bytes,budget_prefix_sha256,financial_contract,offsets)
    known_policies={e['broker_policy_sha256'] for e in state['epochs'].values()}
    if len(broker_policies)!=1 or not broker_policies<=known_policies:
        raise ValueError('Exactly one broker epoch policy bound by the financial prefix is required')
    target={k:v for k,v in state['dispatches'].items() if v['cell_id']==cell_id}
    if set(target)!=set(broker):raise ValueError('Financial and broker target-cell dispatch sets differ')
    if not interrupted<=set(target):raise ValueError('Interrupted local request is not in the exact target cell')
    for ident,group in broker.items():
        start=group.get('dispatch_started');done=group.get('dispatch_completed');failed=group.get('dispatch_failed')
        if start is None:raise ValueError('Target cell has unfinished broker dispatch')
        if ident in interrupted:
            is_local=ident in interrupted_local
            reservation=target[ident]['reservation_id']
            if (done is not None or failed is not None or target[ident]['local'] is not is_local or
                    is_local and reservation is not None or not is_local and
                    (reservation is None or state['calls'][reservation]['state']!='reserved')):
                raise ValueError('Interrupted request must be unfinished with its exact local or retained paid reservation')
        elif done is None and failed is None:raise ValueError('Target cell has unfinished broker dispatch')
        meta={k:v for k,v in target[ident].items() if k not in ('epoch_id','reservation_id')}
        if any(start.get(k)!=v for k,v in meta.items()):raise ValueError('Durable admission and broker start differ')
        for terminal in (done,failed):
            if terminal and any(terminal.get(k)!=v for k,v in meta.items()):raise ValueError('Broker terminal identity differs')
    chunks,api_sha,api_bytes=_api_chunks(api_path)
    starts=list(broker.items());cursor=0;pairs=[];broker_only=[]
    for ident,group in starts:
        dispatch=target[ident];done=group.get('dispatch_completed');failed=group.get('dispatch_failed')
        reservation=dispatch['reservation_id'];row=state['calls'].get(reservation)
        chunk=chunks[cursor] if cursor<len(chunks) else None
        matches=False
        if chunk:
            request=chunk['request'];payload=request.get('payload')
            matches=(isinstance(payload,dict) and _digest(payload)==dispatch.get('payload_sha256') and
                request.get('purpose')==dispatch.get('purpose') and request.get('local') is dispatch['local'] and
                request.get('reservation')==reservation and payload.get('model')==dispatch['model'])
        if not matches:
            if done is not None or failed is None or reservation is not None:
                raise ValueError('Admitted/reserved call has no exact API request')
            broker_only.append(dict(request_id=ident,broker=failed));continue
        cursor+=1
        request,response=chunk['request'],chunk['response'];payload=request['payload']
        if ident in interrupted and (response is not None or chunk['error'] is not None):
            raise ValueError('Interrupted dispatch has a contradictory saved terminal API event')
        if not dispatch['local']:
            if row is None:raise ValueError('Paid API request lacks durable reservation')
            from .webprm_worker import input_token_upper_bound
            rate_in,rate_out=map(_money,prices[payload['model']])
            output=payload.get('max_completion_tokens',payload.get('max_tokens',1024))*payload.get('n',1)
            expected_reservation=_product(Decimal(2),_sum(_product(Decimal(input_token_upper_bound(payload['messages'])),rate_in),
                                                         _product(Decimal(output),rate_out)))
            if row['reserved']!=expected_reservation:raise ValueError('Paid reservation differs from fixed price/token bound')
            if row['state'] in ('settled','overrun'):
                returned=response.get('response') if response else None
                if not isinstance(returned,dict) or returned.get('model')!=payload['model'] or returned.get('usage')!=row['usage']:
                    raise ValueError('Settled usage/model lacks exact saved provider response')
                actual=_sum(_product(Decimal(row['usage']['prompt_tokens']),rate_in),
                            _product(Decimal(row['usage']['completion_tokens']),rate_out))
                if row['charged']!=actual:raise ValueError('Settled charge differs from fixed prices and saved usage')
        if done is not None:
            if response is None or done.get('reservation_id')!=reservation or done.get('journal_sha256')!=chunk.get('terminal_prefix_sha256'):
                raise ValueError('Completed broker receipt lacks exact API prefix/response')
            if not isinstance(response.get('response'),dict) or response['response'].get('model')!=payload['model']:
                raise ValueError('Completed response model differs from the fixed requested model')
            if done.get('usage')!=response['response'].get('usage') or done.get('budget_owner')!='webprm_worker.completion':
                raise ValueError('Completed broker usage/owner mismatch')
            if dispatch['local']:
                if done.get('reservation_state')!='local':raise ValueError('Local receipt incorrectly charged')
            elif (row['state']!='settled' or done.get('reservation_state')!='settled' or
                    done.get('settlement_event_sha256')!=row['settlement_event_sha256'] or
                    done['budget_prefix_bytes']<row['settlement_prefix_bytes'] or
                    done.get('budget_ledger_sha256')!=state['prefixes'][done['budget_prefix_bytes']]):
                raise ValueError('Settled financial event/prefix receipt mismatch')
        pair=dict(request=request,response=response,error=chunk['error'],broker=done or failed,
                  delivery_failure=failed if done is not None else None,request_id=ident)
        if ident in interrupted:
            pair.update(interrupted_local=ident in interrupted_local,interrupted_paid=ident in interrupted_paid,
                        broker_start=group['dispatch_started'])
        pairs.append(pair)
    if cursor!=len(chunks):raise ValueError('Unmatched API requests remain; no calls may be dropped')
    result=dict(verified=True,scope='Read-only saved accounting/evidence consistency; provider semantics, runtime closure and real cohort membership require separate audit.',
        broker_policy_sha256s=sorted(broker_policies),
        paid_calls=sum(not d['local'] for d in target.values()),local_calls=sum(d['local'] for d in target.values()),
        api_request_count=len(chunks),pairs=pairs,broker_only_failures=broker_only,
        failed_calls=sum('dispatch_failed' in g for g in broker.values()),
        cell_unknown_reservations=sum(r['state']=='reserved' for r in state['calls'].values() if r['cell_id']==cell_id),
        global_dispatches=len(state['dispatches']),global_paid_dispatches=state['counts'][False],global_local_dispatches=state['counts'][True],
        global_reservations=len(state['calls']),global_unknown_reservations=state['unknown'],global_overruns=state['overruns'],
        count_semantics='Paid/local counts are admitted dispatches, including preparation failures. API requests record provider intent. Global reserved/unknown counts include other-cell in-flight calls; only target-cell terminal failures establish unknown outcomes.',
        global_charged_usd=str(state['charged']),global_cap_usd=str(state['cap']),global_halt_reason=state['halt'],
        policy_within_caps=state['charged']<=state['cap'] and state['overruns']==0,
        api_sha256=api_sha,api_bytes=api_bytes,broker_prefix_sha256=broker_prefix_sha256,broker_prefix_bytes=broker_prefix_bytes,
        budget_prefix_sha256=budget_prefix_sha256,budget_prefix_bytes=budget_prefix_bytes)
    if interrupted:
        result['interrupted_local_dispatches']=sorted(interrupted_local)
        result['interrupted_paid_dispatches']=sorted(interrupted_paid)
        result['scope']+=' One explicitly identified unfinished call is counted without inventing a terminal event or response; paid reservations remain charged.'
    return result
