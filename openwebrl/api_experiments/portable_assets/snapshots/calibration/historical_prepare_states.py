#!/usr/bin/env python3
"""Prepare private fixed pilot inputs without network/model/browser calls."""
from pathlib import Path
import json,hashlib,unicodedata,re,collections,struct
REPO=Path('/gpfs/projects/krishna/zixianma/OpenWebRL')
SRC=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/joint-data-v2-20260910/joint_pairs.train.jsonl')
OUT=Path(__file__).parent
sha=lambda b:hashlib.sha256(b).hexdigest()
norm=lambda s:' '.join(unicodedata.normalize('NFKC',s).casefold().split())
bench={};excluded=set()
for name,count,key in [('online-mind2web',300,'task_name'),('webvoyager_fara',595,'ques'),('deepshop',150,'task_name')]:
 p=REPO/'openwebrl/data/eval'/f'{name}.jsonl';raw=p.read_bytes();rows=[json.loads(s) for s in raw.splitlines()];assert len(rows)==count
 texts={norm(r[key]) for r in rows};excluded|=texts;bench[name]={'path':str(p),'sha256':sha(raw),'records':count,'normalized_unique_instructions':len(texts)}
raw=SRC.read_bytes();rows=[json.loads(s) for s in raw.splitlines()];pool={};counts=collections.Counter();dups=0;overlap=0
for line,r in enumerate(rows,1):
 assert r['split']=='train' and r['source'] in ('C2','Piotr')
 instruction=norm(r['task'])
 if instruction in excluded:overlap+=1;continue
 prompt=r['prompt'];assert sha(prompt.encode())==r['prompt_sha256']
 assert len(r['images'])==1
 turn=int(r['turn']);depth=turn+1;stratum='1-3' if depth<=3 else '4-9' if depth<=9 else '10+'
 key=sha(json.dumps([instruction,depth,r['prompt_sha256'],r['images'][0]['sha256']],separators=(',',':')).encode())
 if key in pool:dups+=1;continue
 order=sha(('sol61-reasoning-pilot-v1:'+key).encode())
 pool[key]=(order,stratum,line,r);counts[stratum]+=1
selected=[];audit=[]
for stratum in ['1-3','4-9','10+']:
 entries=sorted((v for v in pool.values() if v[1]==stratum),key=lambda x:x[0]);assert len(entries)>=10
 for order,_,line,r in entries[:10]:
  img=r['images'][0];p=Path(img['path']);b=p.read_bytes();assert sha(b)==img['sha256'];assert b[:8]==b'\x89PNG\r\n\x1a\n';w,h=struct.unpack('>II',b[16:24]);assert [w,h]==[img['width'],img['height']]
  prompt=r['prompt'];sys=prompt.split('<|im_start|>system\n',1)[1].split('<|im_end|>',1)[0];matches=re.findall(r'<tools>\s*(.*?)\s*</tools>',sys,re.S); blocks=[b for b in matches if b.strip().startswith('{')]; assert len(blocks)==1
  tools=[json.loads(s) for s in blocks[0].splitlines() if s.strip()];assert len(tools)>0 and all(t.get('type')=='function' for t in tools)
  assert Path(r['source_path']).exists()
  if r.get('source_sha256'):assert sha(Path(r['source_path']).read_bytes())==r['source_sha256']
  state={'state_id':r['id'],'source':r['source'],'source_identity':r['identity'],'source_state_id':r.get('source_state_id',r['base_id']),'source_path':r['source_path'],'episode_id':r['episode_id'],'task_group':r['task_group'],'task':r['task'],'task_instruction_normalized_sha256':sha(norm(r['task']).encode()),'url':r['url'],'host':r['host'],'source_turn_zero_based':int(r['turn']),'decision_depth':int(r['turn'])+1,'stratum':stratum,'prompt':prompt,'prompt_sha256':r['prompt_sha256'],'prompt_tokens':r['prompt_tokens'],'images':[dict(img)],'tools':tools,'source_line_1based':line,'deterministic_order_sha256':order}
  selected.append(state);audit.append({'state_id':r['id'],'source':r['source'],'decision_depth':state['decision_depth'],'source_line_1based':line,'prompt_sha256':r['prompt_sha256'],'image_sha256':img['sha256'],'image_bytes':len(b),'image_dimensions':[w,h],'function_names':[t['function']['name'] for t in tools]})
assert len(selected)==30 and len({r['state_id'] for r in selected})==30
report={'schema':'sol61-reasoning-pilot-source-audit-v1','source_file':str(SRC),'source_sha256':sha(raw),'source_rows':len(rows),'source_row_counts':dict(collections.Counter(r['source'] for r in rows)),'normalization':'Unicode NFKC, casefold, whitespace collapse; exact equality only','benchmark_files':bench,'benchmark_instruction_union':len(excluded),'excluded_rows_benchmark_exact_overlap':overlap,'duplicate_input_image_task_decision_rows':dups,'unique_eligible_pool':len(pool),'eligible_strata_counts':dict(counts),'selection_rule':'10 smallest SHA256(sol61-reasoning-pilot-v1:+dedupkey) per depth stratum; dedupkey=SHA256(JSON([normalized instruction,one-based depth,prompt SHA,image SHA]))','depth_definition':'one-based decision depth = archived zero-based turn + 1','selected_strata_counts':dict(collections.Counter(r['stratum'] for r in selected)),'selected_source_counts':dict(collections.Counter(r['source'] for r in selected)),'selected_unique_tasks':len({norm(r['task']) for r in selected}),'selected':audit,'teacher_fields_removed':['chosen','rejected','teacher_reasoning','chosen_index','rejected_index','label_source','terminal_success','pair_type','label_sha256'],'tools_provenance':'JSON-lines inside <tools>...</tools> in original archived system message; original prompt preserved byte-for-byte','network_model_browser_calls':0}
(OUT/'states-private.json').write_text(json.dumps({'schema':'sol61-reasoning-pilot-states-v1','source_file_sha256':sha(raw),'states':selected},indent=2)+'\n')
(OUT/'source-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['source_rows','excluded_rows_benchmark_exact_overlap','duplicate_input_image_task_decision_rows','unique_eligible_pool','eligible_strata_counts','selected_strata_counts','selected_source_counts','selected_unique_tasks']},indent=2))
