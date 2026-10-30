"""Approved 30-request, $5 maximum medium probe; zero automatic retries."""
from pathlib import Path
import sys,json,fcntl,time
from decimal import Decimal
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import run_pilot as base
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parents[1]

def ledger(reserve=None,settle=None):
    with (ROOT/'budget.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=ROOT/'ledger.json'
        d=json.loads(path.read_text()) if path.exists() else dict(requests=0,settled_requests=0,charged_or_reserved_usd='0')
        balance=Decimal(d['charged_or_reserved_usd'])
        if reserve is not None:
            if base.HALT.is_set() or d['requests']>=30 or balance+reserve>Decimal('5'):
                raise RuntimeError('Medium probe all-attempt cap reached')
            d['requests']+=1;balance+=reserve
        if settle is not None:
            upper,actual=settle;balance+=actual-upper;d['settled_requests']+=1
            if actual>upper or balance>Decimal('5'):
                base.HALT.set();base.dump(ROOT/'halt.json',dict(reason='reservation_overrun'))
                raise RuntimeError('Reservation overrun')
        d['charged_or_reserved_usd']=str(balance);base.dump(path,d);return d

def preflight():
    approval=json.loads((ROOT/'approval.json').read_text())
    assert approval['approved'] and approval['physical_requests_cap']==30 and approval['api_usd_cap']==5
    assert approval['runner_sha256']==base.sha(Path(__file__).read_bytes())
    assert approval['base_runner_sha256']==base.sha((PARENT/'run_pilot.py').read_bytes())
    assert approval['input_bound_source_sha256']==base.sha((base.REPO/'openwebrl/responses_actor_history.py').read_bytes())
    manifest=ROOT/'request-manifest-private.json'
    assert base.sha(manifest.read_bytes())==approval['request_manifest_sha256']
    rows=json.loads(manifest.read_text());assert len(rows)==30 and len({r['state_id'] for r in rows})==30
    for r in rows:
        path=Path(r['request']);high=Path(r['high_receipt'])
        assert base.sha(path.read_bytes())==r['request_file_sha256']
        assert base.sha(high.read_bytes())==r['high_receipt_sha256']
        body=json.loads(path.read_text());prior=json.loads(high.read_text())['request']
        expected=dict(prior,reasoning=dict(effort='medium'),max_output_tokens=4096)
        assert body==expected
        assert body['model']=='gpt-6.1-sol' and body['store'] is False and body['tool_choice']=='required'
        bound=base.response_input_bound(body['input'],body['tools']);assert bound<=272000
        assert (Decimal(bound)*Decimal('2.5')+Decimal(4096)*10)/1000000<=Decimal('5')
    return rows

def main():
    rows=preflight()
    if '--run' not in sys.argv:
        print('Preflight passed: 30 frozen requests, no HTTP calls');return
    with (ROOT/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'ledger.json').exists():raise RuntimeError('Existing attempt ledger; no automatic restart')
        base.ROOT=ROOT;base.CAP=Decimal('5');base.ledger=ledger
        key=next(line.split('=',1)[1].strip().strip(chr(34)+chr(39)) for line in (base.REPO/'.env').read_text().splitlines() if line.startswith('OPENAI_API_KEY='))
        base.dump(ROOT/'current.json',dict(status='running',started_epoch=time.time(),request_cap=30,usd_cap=5))
        for i,r in enumerate(rows):
            receipt=base.call(r['state_id'],'actor_medium_probe',json.loads(Path(r['request']).read_text()),key)
            if 'action' not in receipt:raise RuntimeError('Invalid or incomplete action; stop without retry')
            print(json.dumps(dict(completed=i+1,ledger=ledger())),flush=True)
        base.dump(ROOT/'current.json',dict(status='collected_pending_independent_audit',finished_epoch=time.time(),ledger=ledger()))
if __name__=='__main__':main()
