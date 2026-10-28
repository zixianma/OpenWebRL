import json, glob, collections, re, sys
D='/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021'
res={}; fails=collections.Counter(); epdir={}
for f in glob.glob(D+'/launch-preparation-v11[bc]-*/campaign/attempt-*/cells/*/episodes/*/*/result.json'):
    d=json.load(open(f)); site=f.split('launch-preparation-')[1].split('-')[1]
    k=(site,d['selector'],d['task_id']); e=str(d.get('error',''))
    if not d.get('valid') and 'fuzzy' not in e.lower() and not e.startswith(('Invalid action','List requirements')): fails[k]+=1
    if d.get('valid') or k not in res or (not res[k].get('valid') and 'fuzzy' in e.lower()):
        res[k]=d; epdir[k]=f.rsplit('/',1)[0]
def sels(ep):
    out=[]
    for f in sorted(glob.glob(ep+'/selector-*.json')):
        d=json.load(open(f)); out.append(d)
    return out
pool=collections.Counter(); idx=collections.Counter(); site_ov=collections.defaultdict(lambda:[0,0]); outcome_ov=collections.defaultdict(lambda:[0,0])
for k,d in res.items():
    if k[1]!='piotr_selectionarm': continue
    tag='success' if d.get('valid') and d.get('success') else 'fail' if d.get('valid') else 'invalid'
    for s in sels(epdir[k]):
        t=s.get('trace') or {}; pool[t.get('pool_size')]+=1
        if not t.get('arm_called'): continue
        i=s['selected_index']; idx[i]+=1; ov=int(i!=0)
        site_ov[k[0]][0]+=ov; site_ov[k[0]][1]+=1; outcome_ov[tag][0]+=ov; outcome_ov[tag][1]+=1
print('pool',sorted(pool.items(),key=str)); print('idx',sorted(idx.items()))
print('override by site',{s:(a,b,round(a/b,3)) for s,(a,b) in site_ov.items()})
print('override by outcome',{s:(a,b,round(a/b,3)) for s,(a,b) in outcome_ov.items()})
# ns-only tasks
def act(c): return (c.get('action') if isinstance(c,dict) else str(c)) or ''
def norm(a): return re.sub(r'\s+',' ',a.strip().lower())
def ns_actions(ep):
    p=ep+'/native-raw-trajectory.json'
    try: tr=json.load(open(p))
    except Exception: return None
    acts=[]
    def walk(o):
        if isinstance(o,dict):
            for key in ('action','raw_action','parsed_action'):
                if isinstance(o.get(key),str) and '[' in o.get(key,'') or (isinstance(o.get(key),str) and o.get(key,'').split(' ')[0] in ('click','type','scroll','goto','stop','press','hover','go_back','select')):
                    acts.append(o[key]); return
            for v in o.values(): walk(v)
        elif isinstance(o,list):
            for v in o: walk(v)
    walk(tr); return acts
nsonly=[]
for k,d in res.items():
    if k[1]!='no_search' or not (d.get('valid') and d.get('success')): continue
    ka=(k[0],'piotr_selectionarm',k[2]); a=res.get(ka)
    if a and a.get('valid') and not a.get('success'): nsonly.append(k)
print('ns-only tasks',len(nsonly))
for k in sorted(nsonly):
    ka=(k[0],'piotr_selectionarm',k[2]); ss=sels(epdir[ka]); nsa=ns_actions(epdir[k]) or []
    first=None
    for j,s in enumerate(ss):
        if (s.get('trace') or {}).get('arm_called') and s['selected_index']!=0: first=j; break
    if first is None: print(k, 'no override; arm steps',len(ss)); continue
    s=ss[first]; c=s['candidates']; slot0=norm(act(c[0])); chosen=norm(act(c[s['selected_index']]))
    same_prefix=[norm(x) for x in nsa[:first]]==[norm(act(x['candidates'][x['selected_index']])) for x in ss[:first]]
    nsat=norm(nsa[first]) if first<len(nsa) else None
    print(k,'first_override_step',first,'of',len(ss),'| slot0',slot0[:40],'| chosen',chosen[:40],'| ns_at_step',(nsat or '')[:40],'| identical_prefix',same_prefix,'| ns==slot0',nsat==slot0,'| ns==chosen',nsat==chosen)
# CMS repeated timeouts
print('CMS repeated load timeouts')
for k,d in res.items():
    if k[1]=='piotr_selectionarm' and not d.get('valid') and fails[k]>=2:
        ss=sels(epdir[k])
        if not ss: print(k,'no selector steps'); continue
        s=ss[-1]; c=s['candidates'][s['selected_index']]
        th=(c.get('thought') or '')[:140] if isinstance(c,dict) else ''
        print(k,'steps',len(ss),'last',act(c),'idx',s['selected_index'],'|',th.replace('\n',' '))
