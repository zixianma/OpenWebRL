#!/usr/bin/env python3
"""Build task-disjoint baseline40/replay and baseline90 transfer datasets."""
import argparse,json,re,resource,time
from collections import Counter
from pathlib import Path
from prepare_arm_forward_transfer import PIOTR,JOINT,rows,write,sha,stable,norm,order
from audit_arm_preference_pairs import canonical,parse_action,schemas
from replay_arm_confidence import archive_metadata,image_bytes,current_url
from openwebrl.arm_inference import split_response
from openwebrl.arm_refresh_format import SelectionFormatter

R=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
C=R/'arm-turn-bonus-preparation/arm-refresh-20260924'
OUT=C/'baseline40';DATA=OUT/'data'
REPO=Path(__file__).resolve().parents[1]

def dump(name,rs):
    (DATA/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))

def prepare_static():
    if (DATA/'static-manifest.json').exists():raise ValueError('Static manifest is already frozen')
    DATA.mkdir(exist_ok=True);(DATA/'images').mkdir(exist_ok=True)
    formatter=SelectionFormatter();prepared={s:[] for s in ('future','replay','retention')};exclusions=Counter()
    outcomes={r['task_id']:r['arm'] for r in json.loads((R/'evaluations/arm-task-success-314664/baseline90/success-summary.json').read_text())['task_outcomes']}
    archive_path=None;turns=None
    for ref in sorted(rows(C/'baseline90-reusable-panel.jsonl'),key=lambda r:(r['archive'],r['turn'])):
        raw=Path(ref['record']).read_bytes()
        if sha(raw)!=ref['record_sha256']:raise ValueError('Cached late selector record changed')
        record=json.loads(raw)
        if archive_path!=ref['archive']:
            turns=archive_metadata(ref['archive'])['turns'];archive_path=ref['archive']
        sample=next(s for s in turns if s['metadata']['turn_index']==ref['turn'] and sha(s['prompt'].encode())==ref['prompt_sha256'])
        try:
            image=image_bytes(sample,ref['screenshot_sha256']);path=DATA/'images'/(ref['screenshot_sha256']+'.png');path.write_bytes(image)
            history=sorted([s for s in turns if s['metadata']['turn_index']<ref['turn']],key=lambda s:s['metadata']['turn_index'])
            if [s['metadata']['turn_index'] for s in history]!=list(range(ref['turn'])):raise ValueError('Incomplete history')
            item=dict(ref,split='future',task_group='benchmark:'+norm(ref['task']),task_key=ref['task_id'],
                history=[split_response(s['response']) for s in history],candidates=record['candidates'],screenshot=str(path),
                classes=[canonical(parse_action(c['action'],schemas(sample['prompt']))[0]) for c in record['candidates']],
                source='outcome-only-baseline90+frozenARM',rollout_valid=outcomes[ref['task_id']]['valid'],
                rollout_reward=int(outcomes[ref['task_id']]['success']) if outcomes[ref['task_id']]['valid'] else None)
            item.update(formatter.prepare(item,ref['frozen_permutation']));prepared['future'].append(item)
        except (ValueError,KeyError) as exc:exclusions['future:'+str(exc)]+=1
    early=json.loads((OUT/'states-manifest.json').read_text())
    excluded=set(early['all_source_intents'])|{norm(r['metadata']['intent']) for r in rows(REPO/'openwebrl/data/eval/online-mind2web.jsonl')}
    old=[r for split in ('train','validation') for r in rows(JOINT/f'joint_sft.{split}.jsonl') if r['source']=='Piotr' and norm(r['task']) not in excluded]
    seen=set()
    with (PIOTR/'candidates_merged.jsonl').open('rb') as f:
        for row in sorted(old,key=lambda r:stable(r['id'])):
            split='retention' if int(stable(norm(row['task']))[:8],16)%5==0 else 'replay'
            limit=200 if split=='retention' else 858
            if len(prepared[split])>=limit:continue
            try:
                f.seek(row['candidate_file_offset']);raw=f.readline()
                if sha(raw)!=row['candidate_line_sha256']:raise ValueError('Original candidate line changed')
                draw=json.loads(raw)
                if 'Piotr:'+draw['state_id']!=row['draw_id']:raise ValueError('Original draw mismatch')
                candidates=[split_response(x) for x in draw['candidates']]
                classes=[canonical(parse_action(c['action'],schemas(row['prompt']))[0]) for c in candidates]
                if len(classes)!=5 or len(set(classes))<2:raise ValueError('Invalid original candidates')
                img=row['images'][-1];identity=(row['prompt_sha256'],img['sha256'])
                if identity in seen:continue
                history=[split_response(x) for x in re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',row['prompt'],re.S)]
                item=dict(id=row['id'],task=row['task'],task_group='original:'+norm(row['task']),task_key='original:'+norm(row['task']),
                    url=row['url'],history=history,candidates=candidates,classes=classes,prompt_sha256=row['prompt_sha256'],
                    screenshot=img['path'],screenshot_sha256=img['sha256'],source='Piotr',split=split,
                    teacher_selection_original=row['chosen_index'],source_draw_id=row['draw_id'],source_label_sha256=row['label_sha256'])
                item.update(formatter.prepare(item,order(item['id'])));prepared[split].append(item);seen.add(identity)
            except (ValueError,KeyError,OSError) as exc:exclusions['original:'+str(exc)]+=1
    if len(prepared['replay'])!=858 or len(prepared['retention'])!=200:raise ValueError('Insufficient original teacher-label replay after exclusions')
    if len(prepared['future'])<300:raise ValueError('Too few late states after context check')
    if sum(v for k,v in exclusions.items() if k.startswith('future:'))!=377-len(prepared['future']):
        raise ValueError('Late exclusion accounting mismatch')
    for split,rs in prepared.items():dump(split,rs)
    summary=dict(passed=True,counts={s:len(rs) for s,rs in prepared.items()},exclusions=exclusions,
        hashes={s:sha((DATA/(s+'.jsonl')).read_bytes()) for s in prepared})
    write(DATA/'static-manifest.json',summary);print(json.dumps(summary,indent=2))

def assemble(job):
    stage=R/f'evaluations/arm-refresh-candidates-{job}/baseline40'
    result=json.loads((stage/'collection-complete.json').read_text())
    if not result['complete'] or not (stage/'checkpoint_restore_evidence.json').exists():raise ValueError('Native collection/restore not verified')
    static=json.loads((DATA/'static-manifest.json').read_text())
    for s,d in static['hashes'].items():
        if sha((DATA/(s+'.jsonl')).read_bytes())!=d:raise ValueError('Static dataset changed')
    states={r['id']:r for r in rows(OUT/'states.jsonl')};by={s:[] for s in ('train','dev')}
    for f in sorted((stage/'records').glob('*.json')):
        record=json.loads(f.read_text())
        if not record['eligible']:continue
        state=states[record['id']]
        for key in ('prompt_sha256','screenshot_sha256','split'):
            if state[key]!=record[key]:raise ValueError('Candidate provenance mismatch')
        item=dict(state,**{k:v for k,v in record.items() if k not in state})
        item.update(source='outcome-only-baseline40',candidate_file=str(f),candidate_file_sha256=sha(f.read_bytes()))
        # Actor prompt/images are retained in immutable states.jsonl, not duplicated in fit files.
        item.pop('actor_prompt');item.pop('actor_images')
        by[state['split']].append(item)
    for split,count in [('train',2000),('dev',250)]:
        rs=sorted(by[split],key=lambda r:stable(r['id']))[:count]
        if len(rs)!=count:raise ValueError('Insufficient eligible states')
        dump(split,rs)
    splits={s:list(rows(DATA/(s+'.jsonl'))) for s in ('train','dev','future','replay','retention')}
    tasks={s:{norm(r['task']) for r in rs} for s,rs in splits.items()}
    for a in tasks:
        for b in tasks:
            if a!=b and tasks[a]&tasks[b]:raise ValueError('Task leakage across dataset splits')
    write(DATA/'manifest.json',dict(passed=True,counts={s:len(rs) for s,rs in splits.items()},tasks={s:len(v) for s,v in tasks.items()},
        data_hashes={s+'.jsonl':sha((DATA/(s+'.jsonl')).read_bytes()) for s in splits},candidate_job=str(job),
        lineage='outcome-only baseline; early actor40 / future actor90+frozenARM',early_iterations=[31,40],future_iteration=90))
    print('Assembled',json.dumps({s:len(rs) for s,rs in splits.items()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--static',action='store_true');p.add_argument('--job-id');a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(1536*1024**2,1536*1024**2));resource.setrlimit(resource.RLIMIT_CPU,(180,185))
    if a.static:prepare_static()
    elif a.job_id:assemble(a.job_id)
    else:p.error('Specify --static or --job-id')
