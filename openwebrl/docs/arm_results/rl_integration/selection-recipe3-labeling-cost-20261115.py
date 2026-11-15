"""Teacher API labeling cost for the exact matched Recipe 3 head manifests.
Reads saved label usage and frozen run pricing; makes no API calls. Charges one
accepted label per panel, deduplicated by manifest state_id. Separates training
and validation. Excludes candidate generation/rollouts, feature extraction,
student training, evaluation, retries and unused labels. These are usage-priced
attributions, not invoices or total collection budgets. Input is charged at the
frozen labeller's full-input rate (including cached tokens), as its ledger did.
Only aggregate counts, costs and source fingerprints are exported.
"""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/rl-actor-recollect-20261016/p16-k3-20261031')
CONFIGS=[('luna_pre','GPT-6 Luna Pre-exec.',40.0,'data-luna','pre','labels-luna','pre',125,500),
         ('luna_lookahead','GPT-6 Luna Lookahead H=4',42.9,'data-luna','k3','labels-luna','k3p',125,500),
         ('gpt55_pre','GPT-5.5 Pre-exec.',41.3,'data-all2','pre','labels','pre',5000,30000),
         ('gpt55_lookahead','GPT-5.5 Lookahead H=4',44.5,'data-all2','k3','labels','k3/judgments',5000,30000)]
FIELDS=['panels','input_tokens','output_tokens','cached_input_tokens','long_context_labels','usd_nano']

def receipt(args):
    row,family,sub,ip,op=args
    path=ROOT/family/row['part']/sub/(row['state_id']+'.json')
    raw=path.read_bytes(); data=json.loads(raw)
    if sub=='k3/judgments':
        data=next(x for x in data if x.get('valid'))
        assert data['shown_position']==row['picked']
    else:assert data['position']==row['picked']
    assert data.get('model','').startswith('gpt-6-luna' if family=='labels-luna' else 'gpt-5.5')
    u=data['usage']; it,ot=u['input_tokens'],u['output_tokens']
    assert isinstance(it,int) and isinstance(ot,int) and it>0 and ot>0
    long=sub=='k3/judgments' and it>272000
    # Frozen GPT-5.5 lookahead labeller applies 2x input / 1.5x output above 272K.
    cost=it*ip*(2 if long else 1)+ot*op*(3 if long else 2)//2
    return dict(split=row['split'],part=row['part'],panels=1,input_tokens=it,output_tokens=ot,
                cached_input_tokens=(u.get('input_tokens_details') or {}).get('cached_tokens',0),
                long_context_labels=int(long),usd_nano=cost,sha256=hashlib.sha256(raw).hexdigest())

def finalize(counter):
    out=dict(counter);out['estimated_usd']=out['usd_nano']/1e9
    out['usd_per_1000_panels']=out['estimated_usd']*1000/out['panels']
    return out

if __name__=='__main__':
    result={'accounting':__doc__,'horizon':'H=4 includes the candidate action plus 3 continuation steps.',
            'conditions':{},'frozen_price_source':'labels/p16*/k3/plan.json; labels/p16_labels.py; labels-luna/p16_labels_luna.py; labels-luna/p16_labels_luna_k3p.py'}
    identities=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for key,label,success,folder,arm,family,sub,ip,op in CONFIGS:
            manifest=ROOT/'train'/folder/arm/'panels.jsonl'
            raw=manifest.read_bytes(); rows=[json.loads(line) for line in raw.splitlines()]
            ids={(r['state_id'],r['part'],r['split']) for r in rows}
            assert len(rows)==len(ids)==20508
            identities.append(ids)
            totals=Counter();splits=defaultdict(Counter);batches=defaultdict(Counter)
            digest=hashlib.sha256()
            for r in pool.map(receipt,((row,family,sub,ip,op) for row in rows)):
                digest.update(r['sha256'].encode())
                for field in FIELDS:
                    totals[field]+=r[field];splits[r['split']][field]+=r[field];batches[r['part']][field]+=r[field]
            assert splits['train']['panels']==18781 and splits['validation']['panels']==1727
            item=dict(label=label,mean_overall_success_pct=success,
                price_usd_per_million=dict(input=ip/1000,output=op/1000),
                total=finalize(totals),by_split={k:finalize(v) for k,v in splits.items()},
                by_batch={k:finalize(v) for k,v in batches.items()},
                states=len({r['base_state_id'] for r in rows}),tasks=len({r['task_group'] for r in rows}),
                manifest_sha256=hashlib.sha256(raw).hexdigest(),label_receipts_sha256=digest.hexdigest())
            result['conditions'][key]=item
            print(label,'total',item['total']['estimated_usd'],'train',item['by_split']['train']['estimated_usd'],
                  'per1k',item['total']['usd_per_1000_panels'],flush=True)
    assert all(ids==identities[0] for ids in identities)
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
