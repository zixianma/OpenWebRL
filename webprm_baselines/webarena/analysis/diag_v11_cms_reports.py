import json, glob, collections, re
D='/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021'
res={}; epdir={}
for f in glob.glob(D+'/launch-preparation-v11[bc]-*/campaign/attempt-*/cells/*/episodes/*/*/result.json'):
    d=json.load(open(f)); site=f.split('launch-preparation-')[1].split('-')[1]; k=(site,d['selector'],d['task_id']); e=str(d.get('error',''))
    if d.get('valid') or k not in res or (not res[k].get('valid') and 'fuzzy' in e.lower()): res[k]=d; epdir[k]=f.rsplit('/',1)[0]
def steps(ep):
    try: t=json.load(open(ep+'/native-raw-trajectory.json'))
    except Exception: return []
    out=[]
    for i in range(0,len(t)-1,2):
        obs=t[i]['observation']['text'] if isinstance(t[i].get('observation'),dict) else ''
        a=t[i+1]; eid=a.get('element_id','')
        m=re.search(r'\['+re.escape(str(eid))+r'\] ([^\n]*)',obs) if eid else None
        out.append((a.get('action_type'),eid,(m.group(1)[:50] if m else '')))
    return out
# REPORTS clicks: all methods on cms, count how episodes with REPORTS click ended
c=collections.Counter()
for k,d in res.items():
    if k[0]!='cms': continue
    st=steps(epdir[k]); hit=[s for s in st if 'REPORTS' in s[2]]
    if hit: c[(k[1], 'valid' if d.get('valid') else 'invalid:'+str(d.get('error',''))[:25])]+=1
print('CMS episodes with a menubar REPORTS click, by method/outcome:',dict(c))
# step-0 override tasks: compare ns step0 element vs arm slot0/chosen
for t in ['webarena_lite_049','webarena_lite_143']:
    ns=steps(epdir[('gitlab','no_search',t)]); s=json.load(open(epdir[('gitlab','piotr_selectionarm',t)]+'/selector-000.json'))
    c0=s['candidates'][0]['action']; ch=s['candidates'][s['selected_index']]['action']
    print(t,'ns step0',ns[:1],'| arm slot0',c0,'| chosen',ch)
