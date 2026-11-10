#!/usr/bin/env python3
"""Prepare a task-disjoint ARM refresh experiment from saved additive rollouts.

CPU only: metadata reads, image copies, tokenization, no API/model/GPU calls.
"""
import argparse, base64, hashlib, io, json, math, os, random, re, resource, time
from collections import Counter, defaultdict
from pathlib import Path

from audit_arm_preference_pairs import canonical, parse_action, schemas
from prepare_arm_joint_data import norm, task_text
from replay_arm_confidence import archive_metadata, image_bytes, current_url, sha
from openwebrl.arm_inference import load_selection_builder, selection_messages, split_response

REPO=Path(__file__).resolve().parents[1]
RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
CONTROL=RUNTIME/'arm-turn-bonus-preparation/arm-refresh-20260924'
BASE=Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
MODEL=Path('/gpfs/scrubbed/zixianma/checkpoints/web/arm/selection/81b452d800d9f859687074f82680dd5257e02d89')
PIOTR=RUNTIME/'arm-reproduction/joint-data-inputs-0d83b48/openwebrl_actor'
JOINT=RUNTIME/'arm-reproduction/joint-data-v2-20260910'
SOURCE=RUNTIME/'arm-reproduction/source'
SEED='arm-forward-transfer-v1:42:'

def stable(x):return sha((SEED+str(x)).encode())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')
def rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():yield json.loads(line)
def order(ident):
    p=list(range(5));random.Random(stable(ident)).shuffle(p);return p
def task_partition(task):
    n=int(stable(task)[:8],16)%100
    return 'train' if n<70 else 'dev' if n<85 else 'future'
def group_tasks(items):
    """Union ID aliases and identical normalized intents before splitting."""
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while x!=parent[x]:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for row in items:
        a,b=find('id:'+row['task_key']),find('intent:'+norm(row['task']))
        parent[max(a,b)]=min(a,b)
    for row in items:row['task_group']=find('id:'+row['task_key'])
    return items

def join(row):
    raw=Path(row['label_file']).read_bytes()
    if sha(raw)!=row['label_sha256']:raise ValueError('Candidate log changed')
    matches=[r for r in json.loads(raw)['records'] if r['turn']==row['turn'] and str(r['trajectory_id'])==str(row['trajectory_id'])]
    if len(matches)!=1:raise ValueError('Ambiguous record identity')
    r=matches[0];d=archive_metadata(row['archive']);samples=d['samples']
    trajectory=sorted([s for s in samples if str(s['metadata'].get('trajectory_id'))==str(row['trajectory_id'])],key=lambda s:s['metadata']['turn_index'])
    target=[s for s in trajectory if s['metadata']['turn_index']==row['turn']]
    if len(target)!=1:raise ValueError('Missing or ambiguous executed turn')
    s=target[0];meta=s['metadata']
    if (d['policy_id']!=row['policy_id'] or d['rollout_id']+1!=row['iteration']
        or row['task_id'] not in {str(meta['task_id']),str(meta.get('old_task_id'))}
        or s['prompt']!=r['prompt'] or sha(s['prompt'].encode())!=row['prompt_sha256']
        or s['response']!=r['candidates'][0]['response']):raise ValueError('Candidate/state/policy mismatch')
    history=[x for x in trajectory if x['metadata']['turn_index']<row['turn']]
    if [x['metadata']['turn_index'] for x in history]!=list(range(row['turn'])):raise ValueError('Incomplete history')
    history=[x['response'] for x in history];url=current_url(s)
    if ('history' in r and r['history']!=history) or ('url' in r and r['url']!=url):raise ValueError('Saved selector context mismatch')
    image=image_bytes(s,row['screenshot_sha256'])
    candidates=[split_response(x['response']) for x in r['candidates']]
    classes=[canonical(parse_action(x['action'],schemas(s['prompt']))[0]) for x in candidates]
    if classes!=row['classes']:raise ValueError('Action classes changed')
    return dict(row,task=meta['intent'],url=url,history=[split_response(x) for x in history],candidates=candidates),image

def prepare(output,train=2000,dev=250,future=500,replay=858,retention=200,reuse_current=None):
    from PIL import Image
    from tokenizers import Tokenizer
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    started=time.monotonic();output.mkdir(parents=True,exist_ok=False);(output/'images').mkdir()
    tokenizer=Tokenizer.from_file(str(BASE/'tokenizer.json'))
    env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
    def fail(msg):raise ValueError(msg)
    env.globals['raise_exception']=fail
    template=env.from_string((BASE/'chat_template.jinja').read_text())
    builder=load_selection_builder(SOURCE);image_token=tokenizer.token_to_id('<|image_pad|>')
    benchmark=list(rows(REPO/'openwebrl/data/eval/online-mind2web.jsonl'))
    excluded_intents={norm(x.get('task_name') or x['metadata']['intent']) for x in benchmark}
    excluded_ids={str(x.get('task_id',x.get('id',''))).split('/')[-1] for x in benchmark}
    items=list(rows(CONTROL/'inventory.jsonl'));cache={};filtered=[];exclusions=Counter()
    for row in sorted(items,key=lambda r:r['label_file']):
        if row['label_file'] not in cache:
            cache={row['label_file']:json.loads(Path(row['label_file']).read_text())['records']}
        rec=next(r for r in cache[row['label_file']] if r['turn']==row['turn'] and str(r['trajectory_id'])==str(row['trajectory_id']))
        task=task_text(rec['prompt']);row['task']=task
        if norm(task) in excluded_intents or row['task_key'] in excluded_ids:
            exclusions['benchmark_overlap']+=1;continue
        if any(c.get('finish_type')!='stop' for c in rec['candidates']):
            exclusions['candidate_not_stopped']+=1;continue
        filtered.append(row)
    items=group_tasks(filtered)
    for row in items:row['partition']=task_partition(row['task_group'])
    heldout_intents={norm(x['task']) for x in items if x['partition']!='train'}
    all_intents={norm(x['task']) for x in items}
    prepared=defaultdict(list);stats=defaultdict(Counter);lengths=defaultdict(list)
    seen_states=set();task_counts=Counter();trajectory_counts=Counter()
    def materialize(row,image,split):
        if norm(row['task']) in excluded_intents:raise ValueError('Benchmark overlap')
        ident=sha((row['prompt_sha256']+row['screenshot_sha256']).encode())
        if ident in seen_states:raise ValueError('Duplicate pre-action state')
        p=output/'images'/f'{row["screenshot_sha256"]}.png'
        if sha(image)!=row['screenshot_sha256']:raise ValueError('Image hash mismatch')
        permutation=order(row['id'])
        messages=selection_messages(builder,row['task'],row['url'],row['history'],[row['candidates'][i] for i in permutation],image)
        # Render the exact production prefix, including the open thinking marker.
        messages[1]['content'][1]={'type':'image'}
        prompt=template.render(messages=messages,add_generation_prompt=True,tools=None)
        with Image.open(io.BytesIO(image)) as im:w,h=im.size
        factor=32;hh=max(factor,round(h/factor)*factor);ww=max(factor,round(w/factor)*factor)
        if hh*ww>262144:
            scale=math.sqrt(h*w/262144);hh=math.floor(h/scale/factor)*factor;ww=math.floor(w/scale/factor)*factor
        elif hh*ww<65536:
            scale=math.sqrt(65536/(h*w));hh=math.ceil(h*scale/factor)*factor;ww=math.ceil(w*scale/factor)*factor
        ids=tokenizer.encode(prompt,add_special_tokens=False).ids
        if ids.count(image_token)!=1:raise ValueError('Expected one current screenshot')
        grid=[1,hh//16,ww//16];expanded=math.prod(grid)//4
        prefix=[x for token in ids for x in ([token]*expanded if token==image_token else [token])]
        if len(prefix)+32>8192:raise ValueError('Selection context exceeds8192 without truncation')
        p.write_bytes(image);seen_states.add(ident)
        row.update(split=split,permutation=permutation,screenshot=str(p),image_size=[w,h],state_identity=ident,
                   prompt=prompt,prompt_tokens=len(prefix),prompt_token_sha256=sha(json.dumps(prefix,separators=(',',':')).encode()),image_grid_thw=[grid])
        prepared[split].append(row);stats[split]['states']+=1;stats[split][f'distinct{len(set(row["classes"]))}']+=1
        stats[split]['success' if row.get('reward')==1 else 'failure_or_unknown']+=1;lengths[split].append(len(prefix))
    wanted={'train':train,'dev':dev,'future':future}
    if reuse_current:
        previous=json.loads((reuse_current/'manifest.json').read_text())
        for split,n in wanted.items():
            path=reuse_current/f'{split}.jsonl'
            if previous['counts'][split]!=n or sha(path.read_bytes())!=previous['data_hashes'][path.name]:
                raise ValueError('Prepared current-policy data changed')
            for row in rows(path):
                if sha(Path(row['screenshot']).read_bytes())!=row['screenshot_sha256']:
                    raise ValueError('Prepared screenshot changed')
                prepared[split].append(row);seen_states.add(row['state_identity'])
                lengths[split].append(row['prompt_tokens'])
            stats[split].update(previous['statistics'][split])
    for row in sorted(items,key=lambda r:stable(r['id'])):
        split=row['partition']
        if row['kind']!=('late' if split=='future' else 'early') or len(prepared[split])>=wanted[split]:continue
        if task_counts[(split,row['task_group'])]>=5 or trajectory_counts[(split,row['policy_id'],str(row['trajectory_id']))]>=1:continue
        try:ready,image=join(row);materialize(ready,image,split)
        except (ValueError,KeyError,OSError) as exc:exclusions[str(exc)]+=1;continue
        task_counts[(split,row['task_group'])]+=1;trajectory_counts[(split,row['policy_id'],str(row['trajectory_id']))]+=1
    # Original teacher labels, recovered from the previously verified draw offsets.
    old=[]
    for split in ('train','validation'):
        for row in rows(JOINT/f'joint_sft.{split}.jsonl'):
            if row['source']=='Piotr' and norm(row['task']) not in all_intents|excluded_intents:old.append(row)
    with (PIOTR/'candidates_merged.jsonl').open('rb') as f:
        for row in sorted(old,key=lambda r:stable(r['id'])):
            split='retention' if int(stable(norm(row['task']))[:8],16)%5==0 else 'replay'
            limit=retention if split=='retention' else replay
            if len(prepared[split])>=limit:continue
            try:
                f.seek(row['candidate_file_offset']);raw=f.readline()
                if sha(raw)!=row['candidate_line_sha256']:raise ValueError('Original candidate line changed')
                draw=json.loads(raw)
                if 'Piotr:'+draw['state_id']!=row['draw_id']:raise ValueError('Original draw mismatch')
                candidates=[split_response(x) for x in draw['candidates']]
                classes=[canonical(parse_action(c['action'],schemas(row['prompt']))[0]) for c in candidates]
                if len(classes)!=5 or len(set(classes))<2:raise ValueError('Invalid original candidate count/diversity')
                img=row['images'][-1];image=Path(img['path']).read_bytes()
                history=[split_response(x) for x in re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',row['prompt'],re.S)]
                item=dict(id=row['id'],task=row['task'],task_group='old:'+norm(row['task']),task_key='old:'+norm(row['task']),
                    url=row['url'],history=history,candidates=candidates,classes=classes,prompt_sha256=row['prompt_sha256'],
                    screenshot_sha256=img['sha256'],source='Piotr',teacher_selection_original=row['chosen_index'],
                    source_draw_id=row['draw_id'],source_label_sha256=row['label_sha256'])
                materialize(item,image,split)
            except (ValueError,KeyError,OSError) as exc:exclusions['original:'+str(exc)]+=1
    expected=dict(wanted,replay=replay,retention=retention)
    tasks={s:{norm(r['task']) for r in rs} for s,rs in prepared.items()}
    for a in tasks:
        for b in tasks:
            if a!=b and tasks[a]&tasks[b]:raise ValueError('Fine-tuning split task overlap')
    for split,rs in prepared.items():
        (output/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
    summary=dict(expected=expected,counts={s:len(prepared[s]) for s in expected},tasks={s:len(v) for s,v in tasks.items()},
        passed=all(len(prepared[s])==n for s,n in expected.items()),exclusions=dict(exclusions),statistics=dict(stats),
        lengths={s:dict(min=min(v),median=sorted(v)[len(v)//2],p95=sorted(v)[int(.95*(len(v)-1))],max=max(v)) for s,v in lengths.items()},
        data_hashes={p.name:sha(p.read_bytes()) for p in output.glob('*.jsonl')},split_seed=SEED,
        early_iterations=[31,40],future_iterations=[51,60],lineage='arm-failure-additive-295786',
        notes=['Task IDs and normalized identical intents grouped before splitting.','Maximum five states per task and one per trajectory.','All five candidates structurally valid and stopped; at least two distinct actions.','No outcome or old ARM preference selects the teacher winner.','Original retention is a forgetting probe, not unseen pretraining data.'],
        elapsed_seconds=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write(output/'manifest.json',summary);print(json.dumps(summary,indent=2))
    if not summary['passed']:raise ValueError('Insufficient prepared states; inspect counts before labeling')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=CONTROL/'data-v1')
    p.add_argument('--reuse-current',type=Path);a=p.parse_args()
    os.environ['TOKENIZERS_PARALLELISM']='false'
    resource.setrlimit(resource.RLIMIT_AS,(1536*1024**2,1536*1024**2));resource.setrlimit(resource.RLIMIT_CPU,(180,185))
    prepare(a.output,reuse_current=a.reuse_current)
