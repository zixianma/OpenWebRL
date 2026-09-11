#!/usr/bin/env python3
"""Bounded offline preparation; no torch/models, no training, no image raster decode.

Uses disk-backed state/pair storage, the actor's Rust tokenizer, image headers,
and the repository's actual regex parser extracted without heavyweight imports.
"""
import argparse
import ast
from collections import Counter, defaultdict
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import sqlite3
import time
import unicodedata
from urllib.parse import urlsplit

from audit_arm_preference_pairs import canonical, coordinate_distance, parse_action, schemas, validate

SEED = 'arm-joint-v1:42:'
REV = '0d83b48c1659cac47a1044ef88fb573d3c16e180'


def sha(value): return hashlib.sha256(value).hexdigest()
def order(value): return sha((SEED + value).encode())
def norm(value): return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', value).casefold()))
def encode(value): return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def task_text(prompt):
    matches = re.findall(r'<user_request>(.*?)</user_request>', prompt, re.S)
    if not matches: raise ValueError('missing_task_text')
    return matches[-1].strip()


def pair_reason(winner, loser):
    if canonical(winner) == canonical(loser): return 'identical_action'
    wn = [x['name'] for x in winner]; ln = [x['name'] for x in loser]
    if wn == ln == ['done']: return 'done_vs_done'
    distance = coordinate_distance(winner, loser)
    if distance is not None:
        return 'near_coordinates' if distance <= 5 else 'coordinate_only_review'
    if wn == ln and set(wn) <= {'scroll', 'wait'}:
        if wn == ['scroll']:
            wa, la = winner[0]['arguments'], loser[0]['arguments']
            # Only an amount/duration change is ambiguous; opposite directions
            # are distinct teacher preferences, not automatically wrong actions.
            diffs = {k for k in set(wa)|set(la) if wa.get(k) != la.get(k)}
            if diffs <= {'amount', 'pixels', 'duration', 'seconds'}: return 'scroll_amount_review'
        if set(wn) == {'wait'}: return 'wait_duration_review'
    return 'eligible'


def pair_type(winner, loser):
    wn = [x['name'] for x in winner]; ln = [x['name'] for x in loser]
    if ('done' in wn) != ('done' in ln): return 'termination_vs_continue'
    if len(winner) > 1 or len(loser) > 1: return 'multi_action'
    if any(x in wn+ln for x in ['write', 'press_keys']): return 'typing_or_keys'
    if any(x in wn+ln for x in ['goto_url', 'go_back', 'switch_tab', 'new_tab']): return 'navigation'
    if 'scroll' in wn+ln: return 'scroll_choice'
    if 'wait' in wn+ln: return 'wait_choice'
    if 'click' in wn+ln: return 'click_choice'
    return 'other'


def native_parser(root):
    namespace = {}
    types_path = root / 'openwebrl/base/types.py'
    exec(compile(types_path.read_text(), str(types_path), 'exec'), namespace)
    path = root / 'openwebrl/base/utils.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ToolParser')
    namespace.update(json=json, re=re)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['ToolParser'](parser_type='regex')


class Builder:
    def __init__(self, args):
        from tokenizers import Tokenizer
        self.a = args; self.start = time.monotonic()
        resume=getattr(args,'resume',False)
        args.output.mkdir(parents=True, exist_ok=resume)
        self.db = sqlite3.connect(args.output / 'working.sqlite')
        self.db.execute('PRAGMA cache_size=-8192')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS states(id TEXT PRIMARY KEY, source TEXT, episode TEXT,
          task TEXT, host TEXT, identity TEXT, winner TEXT, pick TEXT, payload TEXT);
          CREATE TABLE IF NOT EXISTS samples(bucket TEXT PRIMARY KEY, pick TEXT, payload TEXT);''')
        self.stats = defaultdict(Counter); self.images = {}; self.contexts = {}
        self.tok = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
        self.parser = native_parser(args.repo)
        self.event_keys=set()
        if resume:
            with (args.output/'exclusion-ledger.jsonl').open() as f:
                for line in f:
                    event=json.loads(line);key=sha(canonical(event).encode())
                    if key not in self.event_keys:
                        self.event_keys.add(key);self.stats[event['source']][event['reason']]+=1
        self.ledger = (args.output / 'exclusion-ledger.jsonl').open('a' if resume else 'w', buffering=1)
        self.processor = json.loads((args.model / 'preprocessor_config.json').read_text())
        self.eos = self.tok.token_to_id('<|im_end|>')
        self.image_token = self.tok.token_to_id('<|image_pad|>')
        bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
        for byte in range(256):
            if byte not in bs:bs.append(byte);cs.append(256+n);n+=1
        self.byte_decoder={chr(c):b for b,c in zip(bs,cs)}
        self.added_ids=set(self.tok.get_added_tokens_decoder())
        self.resize = self.load_resize()
        cache_path=args.output/'working.sqlite' if resume else getattr(args,'context_cache',None)
        if cache_path:
            cache=sqlite3.connect(str(cache_path))
            # Recover the interrupted SQLite transaction before taking a snapshot.
            for (text,) in cache.execute('SELECT payload FROM states'):
                row=json.loads(text)
                self.contexts[row['identity']]={k:row[k] for k in ['identity','prompt_tokens','prompt_token_sha256','image_grid_thw','prompt_sha256']}
                if resume:
                    for img in row['images']:self.image_info(img['path'],img['sha256'])
            cache.close()
            print(f'Reusing {len(self.contexts)} verified context tokenizations; image bytes will be rehashed.',flush=True)
        if resume:
            ds_audit=json.loads((args.c2/'dataset-audit.json').read_text())
            self.c2_hash=ds_audit['dataset_sha256']
            self.stats['C2']['source_states']=8394
            self.stats['C2']['draws_seen']=8394-self.stats['C2']['state_integrity_failure']
        identity_file=getattr(args,'hf',Path('/nonexistent'))/'candidate-identity-audit.json'
        identity_audit=json.loads(identity_file.read_text()) if identity_file.exists() else {}
        self.conflicting_draws={x['id'] for x in identity_audit.get('duplicates',[]) if not x['identical']}
        self.conflicting_states={'Piotr:'+x.split('#')[0] for x in self.conflicting_draws}

    def load_resize(self):
        import importlib.util
        path = Path(importlib.util.find_spec('transformers').origin).parent / 'models/qwen2_vl/image_processing_qwen2_vl.py'
        tree = ast.parse(path.read_text())
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'smart_resize')
        ns = {'math': math}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
        self.resize_source_sha = sha(path.read_bytes())
        return ns['smart_resize']

    def image_info(self, path, expected=None):
        from PIL import Image
        path = str(path)
        if path not in self.images:
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
            with Image.open(path) as im: width, height = im.size; fmt = im.format
            self.images[path] = {'path': path, 'sha256': h.hexdigest(), 'width': width, 'height': height, 'format': fmt}
        info = self.images[path]
        if expected and info['sha256'] != expected: raise ValueError('image_hash_mismatch')
        return info

    def context(self, prompt, images, saved_ids=None, saved_grid=None):
        identity = sha((prompt + encode([i['sha256'] for i in images])).encode())
        if identity not in self.contexts:
            ids = self.tok.encode(prompt, add_special_tokens=False).ids
            grid, counts = [], []
            cfg = self.processor
            for img in images:
                h,w = self.resize(img['height'], img['width'], factor=cfg['patch_size']*cfg['merge_size'],
                                  min_pixels=cfg['size']['shortest_edge'], max_pixels=cfg['size']['longest_edge'])
                g = [1,h//cfg['patch_size'],w//cfg['patch_size']]; grid.append(g)
                counts.append(math.prod(g)//cfg['merge_size']**2)
            if ids.count(self.image_token) != len(images): raise ValueError('image_placeholder_count')
            expanded = []; index=0
            for token in ids:
                if token == self.image_token:
                    expanded.extend([token]*counts[index]); index+=1
                else: expanded.append(token)
            # Do not cache full prefixes: prompt text is already in disk storage.
            value = {'identity': identity, 'prompt_tokens':len(expanded), 'prompt_token_sha256':sha(encode(expanded).encode()),
                     'image_grid_thw':grid, 'prompt_sha256':sha(prompt.encode())}
            self.contexts[identity] = value
        value = self.contexts[identity]
        if saved_ids is not None and sha(encode(saved_ids).encode()) != value['prompt_token_sha256']:
            raise ValueError('processor_prefix_mismatch')
        if saved_grid is not None and saved_grid != value['image_grid_thw']: raise ValueError('image_grid_mismatch')
        return value

    def candidate(self, text, schema, source, finish=None, saved_ids=None):
        actions, spans = parse_action(text, schema)
        native = self.parser.parse(text)
        actual = [{'name':c.name, 'arguments':validate(c.parameters, schema[c.name])} for c in native.calls]
        if not native.success or actual != actions: raise ValueError('native_regex_parser_mismatch')
        raw = self.tok.encode(text, add_special_tokens=False)
        noncanonical = saved_ids is not None and raw.ids != saved_ids
        if noncanonical and self.tok.decode(saved_ids,skip_special_tokens=False)!=text:
            raise ValueError('response_token_text_mismatch')
        if source == 'C2' and finish != 'stop': raise ValueError('truncated_response')
        if source == 'Piotr' and len(raw.ids) >= 1024: raise ValueError('generation_cap_suspected')
        if '</think>' not in text[:spans[0][0]]: raise ValueError('missing_reasoning_boundary')
        tail = text[spans[-1][1]:]
        if tail.replace('<|im_end|>', '').strip(): raise ValueError('text_after_final_action')
        added = '<|im_end|>' not in text
        target = text + '<|im_end|>\n' if added else text
        enc = self.tok.encode(target, add_special_tokens=False)
        ids=enc.ids; offsets=enc.offsets; compare_spans=spans
        if noncanonical:
            if added:raise ValueError('saved_response_missing_eos')
            ids=saved_ids; offsets=[];raw_bytes=bytearray()
            for token in ids:
                string=self.tok.id_to_token(token)
                chunk=string.encode() if token in self.added_ids else bytes(self.byte_decoder[c] for c in string)
                start=len(raw_bytes);raw_bytes.extend(chunk);offsets.append((start,len(raw_bytes)))
            if bytes(raw_bytes)!=text.encode():raise ValueError('generated_token_byte_alignment')
            compare_spans=[(len(text[:a].encode()),len(text[:b].encode())) for a,b in spans]
            self.stats[source]['noncanonical_generated_tokenization_preserved']+=1
        mask=[]
        for j,(begin,end) in enumerate(offsets):
            overlaps = [(a,b) for a,b in compare_spans if begin < b and end > a]
            if overlaps:
                if len(overlaps)!=1 or begin < overlaps[0][0] or end > overlaps[0][1]:
                    raise ValueError('token_crosses_action_boundary')
                mask.append(j)
            elif ids[j] == self.eos: mask.append(j)
        if not mask or ids.count(self.eos)!=1: raise ValueError('invalid_completion_boundary')
        return {'text':target, 'raw_text_sha256':sha(text.encode()), 'token_ids':ids,
                'action_token_indices':mask, 'action_char_spans':spans, 'actions':actions,
                'tokens':len(ids), 'action_tokens':len(mask), 'completion_boundary_appended':added,
                'finish_provenance':'observed_stop' if source=='C2' else 'metadata_absent_structurally_complete_below_1024'}

    def event(self, state, reason, **details):
        event={'id':state['id'],'source':state['source'],'reason':reason,**details}
        key=sha(canonical(event).encode())
        if key in self.event_keys:return
        self.event_keys.add(key)
        self.stats[state['source']][reason] += 1
        self.ledger.write(encode(event)+'\n')

    def sample(self, state, winner, loser, status, reason, index):
        bucket = state['source']+':'+reason+':'+pair_type(winner.get('actions',[]),loser.get('actions',[]))
        pick = order(state['id']+':'+str(index))
        old = self.db.execute('SELECT pick FROM samples WHERE bucket=?',(bucket,)).fetchone()
        if old and old[0] <= pick: return
        payload={**state,'chosen':winner,'rejected':loser,'status':status,'reason':reason,
                 'pair_type':pair_type(winner.get('actions',[]),loser.get('actions',[]))}
        self.db.execute('INSERT OR REPLACE INTO samples VALUES(?,?,?)',(bucket,pick,encode(payload)))

    def accept_draw(self, state, raw_candidates, selected, schema):
        source = state['source']; self.stats[source]['draws_seen']+=1
        existing=self.db.execute('SELECT pick FROM states WHERE id=?',(state['base_id'],)).fetchone()
        if existing and existing[0].startswith('0') and '0'+order(state['id']) >= existing[0]:
            self.event(state,'draw_not_selected_by_seed')
            return
        try:
            if isinstance(selected,bool) or not isinstance(selected,int) or not 0 <= selected < len(raw_candidates):
                raise ValueError('invalid_selection_index')
            r=raw_candidates[selected]
            winner=self.candidate(r['text'],schema,source,r.get('finish'),r.get('ids'))
            if state['prompt_tokens']+winner['tokens']>32768: raise ValueError('winner_context_limit')
        except (ValueError,TypeError,KeyError) as err:
            self.event(state,'invalid_winner',detail=str(err));return
        self.stats[source]['structurally_valid_winner_draws']+=1
        eligible=[]; seen=set()
        for i,r in enumerate(raw_candidates):
            if i==selected:continue
            try:
                actions,spans=parse_action(r['text'],schema)
                if source=='C2' and r.get('finish')!='stop':raise ValueError('truncated_response')
                loser={'text':r['text'],'actions':actions,'action_char_spans':spans}
            except (ValueError,TypeError,KeyError) as err:
                self.event(state,'invalid_loser',candidate_index=i,detail=str(err))
                self.sample(state,winner,{'text':r['text'],'actions':[],'validation_error':str(err)},'excluded','invalid_loser',i);continue
            reason=pair_reason(winner['actions'],loser['actions'])
            group=canonical(loser['actions'])
            if group in seen:reason='duplicate_loser_action'
            seen.add(group)
            if reason!='eligible':
                self.event(state,reason,candidate_index=i)
                self.sample(state,winner,loser,'quarantined' if reason.endswith('_review') else 'excluded',reason,i)
                continue
            eligible.append((order(state['id']+':loser:'+group),i,r))
        base_id=state['base_id']
        drawpick=order(state['id'])
        valid=None
        for _,li,r in sorted(eligible):
            try:
                loser=self.candidate(r['text'],schema,source,r.get('finish'),r.get('ids'))
                if state['prompt_tokens']+loser['tokens']>32768:raise ValueError('loser_context_limit')
                valid=(li,loser);break
            except (ValueError,TypeError,KeyError) as err:
                self.event(state,'invalid_loser',candidate_index=li,detail=str(err))
                self.sample(state,winner,{'text':r['text'],'actions':[],'validation_error':str(err)},'excluded','invalid_loser',li)
        if valid:
            li,loser=valid
            payload={**state,'id':base_id,'draw_id':state['id'],'chosen_index':selected,'rejected_index':li,
                     'chosen':winner,'rejected':loser,'status':'retained','reason':'eligible',
                     'pair_type':pair_type(winner['actions'],loser['actions'])}
            kind='pair'
        else:
            self.event(state,'draw_without_eligible_negative')
            payload={**state,'id':base_id,'draw_id':state['id'],'chosen_index':selected,
                     'chosen':winner,'status':'sft_reserve','reason':'no_eligible_negative'}
            kind='reserve'
        # A valid pair always takes priority over an SFT-only reserve from another draw.
        pick=('0' if kind=='pair' else '1')+drawpick
        old=self.db.execute('SELECT pick FROM states WHERE id=?',(base_id,)).fetchone()
        if not old or pick < old[0]:
            self.db.execute('INSERT OR REPLACE INTO states VALUES(?,?,?,?,?,?,?,?,?)',
               (base_id,source,state['episode_id'],norm(state['task']),state['host'],state['identity'],
                canonical(winner['actions']),pick,encode(payload)))

    def c2(self):
        root=self.a.c2
        audit=json.loads((root/'dataset-audit.json').read_text())
        ds=Path(audit['dataset']); h=hashlib.sha256(); prev=None
        with ds.open('rb') as stream:
            for line_no,line in enumerate(stream,1):
                h.update(line);r=json.loads(line)
                stub={'id':'C2:'+r['task_id']+':'+str(r['turn']),'source':'C2'}
                self.stats['C2']['source_states']+=1
                try:
                    raw=Path(r['source']).read_bytes()
                    if sha(raw)!=r['source_sha256']:raise ValueError('source_hash')
                    c=json.loads(raw)
                    if (c['task_id'],c['turn'],c['selected_index'])!=(r['task_id'],r['turn'],r['selected_index']):raise ValueError('source_join')
                    if sha(c['prompt'].encode())!=c['prompt_sha256']:raise ValueError('prompt_hash')
                    ep=Path(c['attempt'])
                    if ep!=prev:
                        execution=json.loads((ep/'execution.json').read_text())
                        turns={x['turn']:x for x in execution['turns']}
                        outcome=json.loads((ep/'outcome.json').read_text());prev=ep
                    t=turns[r['turn']]
                    if not outcome.get('valid') or outcome.get('reward')!=1 or outcome['task_id']!=r['task_id']:raise ValueError('terminal_outcome')
                    if execution['task_id']!=r['task_id'] or not t['executed'] or t['source_sha256']!=r['source_sha256'] or t['source']!=r['source']:raise ValueError('execution_join')
                    if t['prompt_token_ids']!=r['prompt_token_ids'] or t['image_grid_thw']!=r['image_grid_thw']:raise ValueError('execution_token_join')
                    if c.get('fallback') or c['mode']!='selection':raise ValueError('selector_fallback')
                    images=[self.image_info(x['path'],x['sha256']) for x in c['images']]
                    context=self.context(c['prompt'],images,r['prompt_token_ids'],r['image_grid_thw'])
                    state={**stub,**context,'base_id':stub['id'],'episode_id':'C2:'+r['task_id'],
                           'task':task_text(c['prompt']),'host':urlsplit(c['url']).hostname or '',
                           'url':c['url'],'turn':r['turn'],'prompt':c['prompt'],'images':images,
                           'source_path':r['source'],'source_sha256':r['source_sha256'],
                           'label_source':'SelectionARM; executed in valid successful trajectory','terminal_success':True}
                    candidates=[{'text':x['response'],'finish':x['finish_type'],'ids':x['response_token_ids']} for x in c['raw_candidates']]
                    self.accept_draw(state,candidates,r['selected_index'],schemas(c['prompt']))
                except (ValueError,TypeError,KeyError,OSError) as error:self.event(stub,'state_integrity_failure',detail=str(error))
                if line_no%500==0:
                    self.db.commit();print(f'C2 {line_no}/8394',flush=True)
                time.sleep(.005)
        if h.hexdigest()!=audit['dataset_sha256']:raise ValueError('C2 dataset hash mismatch')
        self.c2_hash=h.hexdigest();self.db.commit()

    def piotr(self):
        root=self.a.hf/'openwebrl_actor'
        states={}; labels={}
        for line in (root/'states_full.jsonl').open():
            r=json.loads(line); self.stats['Piotr']['source_states']+=1
            stub={'id':'Piotr:'+r['state_id'],'source':'Piotr'}
            try:
                images=[self.image_info(root/'state_images'/Path(p).name) for p in r['images']]
                prompt=r['prompt_text']; context=self.context(prompt,images)
                urls=re.findall(r'https?://[^\s<>]+',prompt)
                obs=prompt.rsplit('<observation>',1)[-1]
                current=re.search(r'\(active\):\s*(https?://\S+)',obs)
                url=current[1] if current else (urls[-1] if urls else '')
                state={**stub,**context,'base_id':stub['id'],'episode_id':'Piotr:'+r['episode_id'],
                       'task':task_text(prompt),'host':urlsplit(url).hostname or '',
                       'url':url,'turn':r['turn'],'prompt':prompt,'images':images,
                       'source_path':str(root/'states_full.jsonl'),'source_state_id':r['state_id'],
                       'label_source':'GPT-5.5 selection; candidate outcomes unobserved','terminal_success':None}
                states[r['state_id']]=(state,schemas(prompt))
            except (ValueError,KeyError,OSError) as error:self.event(stub,'state_integrity_failure',detail=str(error))
            time.sleep(.002)
        for line in (root/'labels_drawlevel.jsonl').open():
            lab=json.loads(line)
            if lab['state_id'] in labels:raise ValueError('duplicate draw label')
            labels[lab['state_id']]=lab
        consumed=set()
        with (root/'candidates_merged.jsonl').open('rb') as f:
            line_no=0
            while True:
                offset=f.tell();line=f.readline()
                if not line:break
                line_no+=1;c=json.loads(line);sid=c['state_id'];base=c.get('base_state_id',sid.split('#')[0])
                self.stats['Piotr']['candidate_draws']+=1
                self.stats['Piotr']['temperature_'+str(c.get('temperature'))]+=1
                if sid not in labels:continue
                if sid in consumed:
                    if sid in self.conflicting_draws:continue
                    raise ValueError('duplicate candidate draw not inventoried')
                consumed.add(sid)
                if sid in self.conflicting_draws:
                    self.event({'id':'Piotr:'+sid,'source':'Piotr'},'conflicting_candidate_draw_review')
                    continue
                if base not in states:continue
                base_state,schema=states[base]
                if [Path(x).name for x in c['images']] != [Path(x['path']).name for x in base_state['images']]:raise ValueError('candidate image join')
                lab=labels[sid]
                state={**base_state,'id':'Piotr:'+sid,'teacher_reasoning':lab.get('reasoning',''),
                       'temperature':c.get('temperature'),'candidate_file_offset':offset,
                       'candidate_line_sha256':sha(line),'label_sha256':sha(encode(lab).encode()),'custom_id':lab.get('custom_id')}
                self.accept_draw(state,[{'text':x} for x in c['candidates']],lab.get('selection'),schema)
                if line_no%1000==0:
                    self.db.commit();print(f'Piotr candidate draws {line_no}',flush=True)
                time.sleep(.002)
        self.stats['Piotr']['unmatched_labels']=len(set(labels)-consumed)
        self.db.commit()

    def finish(self):
        records=list(self.db.execute('SELECT id,source,episode,task,host,identity,winner,pick FROM states'))
        parent={r[2]:r[2] for r in records}
        def find(x):
            while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
            return x
        def union(a,b):
            a,b=find(a),find(b)
            if a!=b:parent[max(a,b)]=min(a,b)
        bytask=defaultdict(list); byidentity=defaultdict(list); byhost=defaultdict(dict)
        for r in records:
            bytask[r[3]].append(r);byidentity[r[5]].append(r);byhost[r[4]][r[3]]=r[2]
        for group in list(bytask.values())+list(byidentity.values()):
            for r in group[1:]:union(group[0][2],r[2])
        near=[]
        for host,tasks in byhost.items():
            vals=sorted(tasks)
            for i,a in enumerate(vals):
                aa=set(a.split())
                for b in vals[i+1:]:
                    bb=set(b.split());jac=len(aa&bb)/max(1,len(aa|bb))
                    if jac>=.75 and difflib.SequenceMatcher(None,a,b,autojunk=False).ratio()>=.9:
                        union(tasks[a],tasks[b]);near.append({'host':host,'task_a':a,'task_b':b,'reason':'conservative_near_task_group'})
        benchmark=[json.loads(l) for l in self.a.eval.open()]
        exact={norm(r.get('task_name') or r['metadata']['intent']) for r in benchmark}
        evalhost=defaultdict(list)
        for r in benchmark:
            host=urlsplit(r.get('website') or r.get('metadata',{}).get('start_url','')).hostname or ''
            evalhost[host].append(norm(r.get('task_name') or r['metadata']['intent']))
        flags={}; eval_flags=[]
        for task,rs in bytask.items():
            reason='exact_benchmark_task' if task in exact else None; matches=[]
            if not reason:
                a=set(task.split())
                for host in {r[4] for r in rs}:
                    for target in evalhost[host]:
                        b=set(target.split());jac=len(a&b)/max(1,len(a|b))
                        if jac>=.6 or (jac>=.3 and difflib.SequenceMatcher(None,task,target,autojunk=False).ratio()>=.85):matches.append(target)
                if matches:reason='possible_benchmark_overlap_review'
            if reason:
                for r in rs:flags[find(r[2])]=reason
                eval_flags.append({'task':task,'reason':reason,'matches':matches})
        groups=sorted({find(r[2]) for r in records},key=order)
        val=set(groups[:math.ceil(.1*len(groups))])
        duplicate={}
        for identity,rows in byidentity.items():
            if len(rows)>1:
                if len({r[6] for r in rows})>1:
                    for r in rows:duplicate[r[0]]='conflicting_duplicate_state_review'
                else:
                    keep=min(rows,key=lambda r:(r[7][0],order(r[0])))
                    for r in rows:
                        if r[0]!=keep[0]:duplicate[r[0]]='duplicate_state'
        paths={n:(self.a.output/(n+'.jsonl')).open('w') for n in ['joint_pairs.train','joint_pairs.validation','joint_sft.train','joint_sft.validation','sft-only-reserve','quarantine-states','split-manifest']}
        distributions=defaultdict(Counter); lengths=defaultdict(list); review=defaultdict(list)
        train_tasks=set();val_tasks=set()
        for r in sorted(records,key=lambda r:order(r[0])):
            x=json.loads(self.db.execute('SELECT payload FROM states WHERE id=?',(r[0],)).fetchone()[0])
            group=find(r[2]);split='validation' if group in val else 'train'
            x.update(task_group=group,split=split)
            why=('conflicting_candidate_draw_review' if r[0] in self.conflicting_states else None) or flags.get(group) or duplicate.get(r[0])
            paths['split-manifest'].write(encode({'id':x['id'],'task_group':group,'split':split,'source':x['source'],'excluded':why})+'\n')
            if why:
                x.update(status='quarantined' if why.endswith('_review') else 'excluded',reason=why)
                paths['quarantine-states'].write(encode(x)+'\n');self.event(x,why)
                if x.get('rejected'):self.sample(x,x['chosen'],x['rejected'],x['status'],why,0)
                continue
            if x['status']=='sft_reserve':
                paths['sft-only-reserve'].write(encode(x)+'\n');self.stats[x['source']]['sft_reserve_states']+=1;continue
            paths['joint_pairs.'+split].write(encode(x)+'\n')
            sft={k:v for k,v in x.items() if k not in ['rejected','rejected_index']}
            paths['joint_sft.'+split].write(encode(sft)+'\n')
            source=x['source'];self.stats[source]['retained_'+split]+=1
            distributions[source+':'+split]['winner:'+'+'.join(a['name'] for a in x['chosen']['actions'])]+=1
            distributions[source+':'+split]['pair:'+x['pair_type']]+=1
            (train_tasks if split=='train' else val_tasks).add(group)
            for key,value in [('prompt',x['prompt_tokens']),('chosen',x['chosen']['tokens']),('rejected',x['rejected']['tokens']),('action',x['chosen']['action_tokens'])]:lengths[source+':'+key].append(value)
            review[source+':'+x['pair_type']].append((order(x['id']),x['id']))
        for f in paths.values():f.close()
        assert not train_tasks&val_tasks
        selected=[]
        for source in ['C2','Piotr']:
            buckets=[sorted(v) for k,v in sorted(review.items()) if k.startswith(source+':')]
            count=0;offset=0
            while count<32 and any(offset<len(v) for v in buckets):
                for bucket in buckets:
                    if offset<len(bucket) and count<32:
                        sid=bucket[offset][1]
                        x=json.loads(self.db.execute('SELECT payload FROM states WHERE id=?',(sid,)).fetchone()[0])
                        group=find(x['episode_id']);x.update(task_group=group,split='validation' if group in val else 'train')
                        selected.append(x);count+=1
                offset+=1
        selected.extend(json.loads(r[0]) for r in self.db.execute('SELECT payload FROM samples ORDER BY bucket'))
        with (self.a.output/'review-examples.jsonl').open('w') as f:
            for x in selected:f.write(encode(x)+'\n')
        self.ledger.close(); self.db.commit()
        def percentiles(v):
            v=sorted(v)
            return {str(p):v[min(len(v)-1,round((len(v)-1)*p/100))] for p in [0,50,90,95,99,100]} if v else {}
        usage=resource.getrusage(resource.RUSAGE_SELF)
        report={'status':'prepared_for_human_review_not_training_approved','revision':REV,'seed':SEED,'counts':dict(self.stats),
                'distributions':dict(distributions),'token_length_percentiles':{k:percentiles(v) for k,v in lengths.items()},
                'task_groups':{'all':len(groups),'train_retained':len(train_tasks),'validation_retained':len(val_tasks),'overlap':0},
                'conflicting_candidate_draw_ids':len(self.conflicting_draws),'conflicting_candidate_base_states':len(self.conflicting_states),
                'near_task_groups':near,'benchmark_overlap_flags':eval_flags,'unique_images_hashed':len(self.images),
                'review_examples':len(selected),'c2_dataset_sha256':self.c2_hash,'tokenizer_sha256':sha((self.a.model/'tokenizer.json').read_bytes()),
                'resize_source_sha256':self.resize_source_sha,'parser_source_sha256':sha((self.a.repo/'openwebrl/base/utils.py').read_bytes()),
                'conservative_changes':['All coordinate-only different clicks beyond radius 5 quarantined pending visual review; no DOM element IDs available.',
                 'HF missing finish metadata: require structurally complete action and <1024 raw tokens; append explicit completion boundary, do not assert observed stop.'],
              'remaining_gates':['User visual review and SFT approval','Full processor pixel-tensor/native SGLang parity smoke on compute node; prefix IDs checked from actual resize function and tokenizer here.'],
                'count_scope':'All C2 states and HF draw identities inventoried. Detailed candidate exclusions concern processed draws; higher-hash draws skipped after a usable lower-hash pair is found. Token validation applies to retained candidates and attempted replacements, not all excluded alternatives.',
                'resources':{'wall_seconds':time.monotonic()-self.start,'cpu_seconds':usage.ru_utime+usage.ru_stime,'peak_rss_mib':usage.ru_maxrss/1024}}
        hashes={}
        for p in self.a.output.glob('*.jsonl'):
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            hashes[p.name]=h.hexdigest()
        report['artifact_sha256']=hashes
        (self.a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
        (self.a.output/'image-manifest.json').write_text(json.dumps(list(self.images.values()),indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ['counts','task_groups','review_examples','resources']},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--hf',type=Path,required=True);p.add_argument('--c2',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--context-cache',type=Path)
    p.add_argument('--resume',action='store_true',help='Resume after C2 phase; retain SQLite pairs and deduplicate event ledger.')
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--model',type=Path,default=Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT'))
    p.add_argument('--eval',type=Path,default=Path('openwebrl/data/eval/online-mind2web.jsonl'))
    a=p.parse_args()
    os.environ['TOKENIZERS_PARALLELISM']='false'
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(240,245))
    b=Builder(a)
    if not a.resume:b.c2()
    b.piotr();b.finish()


if __name__=='__main__':main()
