#!/usr/bin/env python3
"""Exploratory website/workflow clustering of cached task-instruction embeddings.

No tasks are filtered or selected for training. One CPU thread; memory maps the
cache and processes one stratum at a time. sklearn==1.6.1, numpy==1.26.4.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
import mmap
from pathlib import Path
import re
import resource
import time

import numpy as np
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

SEED = 20260930
# These detect words, not verified browser behavior. Keep all matched tags and
# the matched phrases. Priority only chooses the hard stratum for this first pass.
RULES = [
    ('cart_purchase', r'\b(?:add\b.{0,60}\b(?:cart|basket)|checkout|check out|purchase|buy|order\b.{0,40}\b(?:delivery|pickup))\b'),
    ('booking_application', r'\b(?:book a|book an|reserve|reservation|apply for|submit|register|sign up)\b'),
    ('comparison', r'\b(?:compare|comparison|versus|vs\.?|difference between|differences between)\b'),
    ('search_filter_sort', r'\b(?:filter|sort|cheapest|highest.rated|lowest.pric\w*|top.rated|under \$|between \$|at least|at most|no more than|no fewer than|rated.{0,12}stars|search for)\b'),
    ('calculation', r'\b(?:calculate|compute|solve|equation|derivative|integral|roots of|convert)\b'),
    ('list_aggregate', r'\b(?:list|summarize|summary|how many|identify (?:three|four|five|all)|find (?:three|four|five|all))\b'),
]
COMPILED = [(name, re.compile(pattern, re.I)) for name, pattern in RULES]


def workflow(text):
    text = ' '.join(text.split())
    matches = {name: sorted({m.group(0) for m in regex.finditer(text)}) for name, regex in COMPILED}
    matches = {k:v for k,v in matches.items() if v}
    tags = list(matches)
    return (tags[0] if tags else 'lookup_or_unclassified'), tags, matches


def rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def load_metadata(root):
    plan = json.loads((root/'semantic-qwen8b/plan.json').read_text())
    audit = json.loads((root/'audit.json').read_text())
    selected_path = root/'benchmark-sites-20260930/verified/available-benchmark-tasks.jsonl'
    selected = {(r['source'], str(r['task_id'])):r for r in rows(selected_path)}
    result = []
    h = hashlib.sha256(); h.update(b'[')
    n = 0
    for filename, field in [('semantic-references.jsonl','text'), ('candidates.jsonl','task_name')]:
        path = root/filename
        assert sha(path) == audit['artifacts'][filename]['sha256'], filename
        for i, row in enumerate(rows(path)):
            if n:
                h.update(b', ')
            h.update(json.dumps(row[field], ensure_ascii=False).encode()); n += 1
            if filename == 'candidates.jsonl':
                key = (row['benchmark_name'], str(row['task_id']))
                if key in selected:
                    ref = selected[key]
                    primary, tags, evidence = workflow(row['task_name'])
                    result.append(dict(source=key[0], task_id=key[1], site=ref['site'], host=ref['host'],
                        start_url=ref['start_url'], instruction=row['task_name'], embedding_row=plan['references']+i,
                        primary_workflow=primary, workflow_tags=tags, workflow_evidence=evidence,
                        rubric_difficulty=row.get('difficulty'), domain=row.get('domain'), subdomain=row.get('subdomain')))
    h.update(b']')
    assert n == plan['total_texts'] and h.hexdigest() == plan['text_sha256']
    assert len(result) == len(selected) == 59115
    assert len({(r['source'],r['task_id']) for r in result}) == len(result)
    assert {(r['source'],r['task_id']) for r in result} == set(selected)
    kept = {(r['benchmark_name'],str(r['task_id'])) for r in rows(root/'semantic-qwen8b/kept-337317.jsonl')}
    assert set(selected) <= kept
    return plan, result, selected_path


def fit(x, k, seed):
    if k == 1:
        return np.zeros(len(x), dtype=np.int32)
    model = BisectingKMeans(n_clusters=k, random_state=seed, n_init=2, max_iter=40,
                            tol=1e-4, bisecting_strategy='biggest_inertia')
    return model.fit_predict(x)


def partition(x, labels, records, group_id, resolution):
    members, clusters = [], []
    # Order cluster IDs by earliest source embedding row, not arbitrary sklearn labels.
    unique = sorted(np.unique(labels), key=lambda label:min(records[i]['embedding_row'] for i in np.flatnonzero(labels==label)))
    for ordinal, label in enumerate(unique):
        indices = np.flatnonzero(labels == label)
        sub = x[indices]
        center = sub.mean(axis=0); center /= max(float(np.linalg.norm(center)), 1e-12)
        cosine = sub @ center
        order = np.argsort(-cosine, kind='stable')
        cid = f'{group_id}-r{resolution}-c{ordinal:04d}'
        examples = []
        pick = [('representative',int(order[0])), ('second_nearest',int(order[min(1,len(order)-1)])), ('boundary',int(order[-1]))]
        seen = set()
        for role, local in pick:
            if local in seen:
                continue
            seen.add(local); rec=records[int(indices[local])]
            examples.append(dict(role=role,task_id=rec['task_id'],source=rec['source'],
                instruction=rec['instruction'],cosine_to_centroid=float(cosine[local])))
        first = records[int(indices[0])]
        clusters.append(dict(cluster_id=cid,site=first['site'],workflow=first['primary_workflow'],
            size=len(indices),median_cosine=float(np.median(cosine)),min_cosine=float(cosine.min()),
            representative_task_id=examples[0]['task_id'],representative_source=examples[0]['source'],examples=examples))
        for local, index in enumerate(indices):
            r=records[int(index)]
            members.append(dict(task_id=r['task_id'],source=r['source'],site=r['site'],
                primary_workflow=r['primary_workflow'],workflow_tags=r['workflow_tags'],
                cluster_id=cid,embedding_row=r['embedding_row'],cosine_to_centroid=float(cosine[local])))
    return members, clusters


def make_html(path, clusters):
    # Raw public task examples stay in runtime; summary/metrics are the published artifacts.
    data = json.dumps(clusters, ensure_ascii=False).replace('<','\\u003c')
    page = '''<!doctype html><meta charset="utf-8"><title>ARM task clusters</title>
<style>body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:0 20px;color:#182230}select,input{font:inherit;padding:6px}article{border:1px solid #cbd5e1;border-radius:8px;padding:14px;margin:12px 0}small{color:#536375}.boundary{background:#fff7ed;padding:8px}button{font:inherit;margin:8px;padding:5px}code{font-size:13px}</style>
<h1>Website × heuristic interaction type × instruction cluster</h1>
<p>Exploratory groups; all tasks retained. Neither interaction tags nor embeddings establish behavior, difficulty, feasibility or ARM value. A boundary example is distant in text space, not necessarily bad.</p>
<label>Website <select id="site"><option value="">All</option></select></label>
<label>Type <select id="type"><option value="">All</option></select></label>
<label>Resolution <select id="res"><option value="50">~50 tasks/cluster</option><option value="100">~100 tasks/cluster</option></select></label>
<input id="search" placeholder="Search example text"><p id="count"></p><div id="cards"></div><button id="more">Show 30 more</button>
<script>const DATA=__DATA__;const $=id=>document.getElementById(id);let limit=30;
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
for(const [id,key] of [['site','site'],['type','workflow']])for(const v of [...new Set(DATA.map(x=>x[key]))].sort())$(id).add(new Option(v,v));
function draw(){const q=$('search').value.toLowerCase();let rows=DATA.filter(x=>(!$('site').value||x.site===$('site').value)&&(!$('type').value||x.workflow===$('type').value)&&x.cluster_id.includes('-r'+$('res').value+'-')&&(!q||JSON.stringify(x.examples).toLowerCase().includes(q)));rows.sort((a,b)=>b.size-a.size||a.cluster_id.localeCompare(b.cluster_id));$('count').textContent=rows.length+' clusters; showing '+Math.min(limit,rows.length);$('cards').innerHTML=rows.slice(0,limit).map(x=>`<article><b>${esc(x.site)} · ${esc(x.workflow)}</b><p>${x.size} tasks · median cosine to centroid ${x.median_cosine.toFixed(3)} · <code>${esc(x.cluster_id)}</code></p>${x.examples.map(e=>`<div class="${e.role==='boundary'?'boundary':''}"><small>${esc(e.role)} · ${esc(e.source)}:${esc(e.task_id)}</small><p>${esc(e.instruction)}</p></div>`).join('')}</article>`).join('');$('more').hidden=limit>=rows.length;}
for(const id of ['site','type','res','search'])$(id).addEventListener('input',()=>{limit=30;draw()});$('more').onclick=()=>{limit+=30;draw()};draw();</script>'''
    path.write_text(page.replace('__DATA__',data))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--cpu-seconds',type=int,default=240)
    args=p.parse_args();resource.setrlimit(resource.RLIMIT_CPU,(args.cpu_seconds,args.cpu_seconds+5))
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    with threadpool_limits(limits=1):
        plan, records, selected_path=load_metadata(args.root)
        cache=args.root/'semantic-qwen8b/embeddings.npy';cache_stat=cache.stat()
        embeddings=np.load(cache,mmap_mode='r')
        assert embeddings.shape==(plan['total_texts'],4096) and embeddings.dtype==np.float32
        strata=defaultdict(list)
        for rec in records:strata[(rec['site'],rec['primary_workflow'])].append(rec)
        ordered=sorted(strata.items(),key=lambda kv:(-len(kv[1]),kv[0]))
        write_json(args.output/'metadata-audit.json',dict(tasks=len(records),strata=len(strata),
            workflow_counts=dict(Counter(r['primary_workflow'] for r in records)),
            multiple_tag_tasks=sum(len(r['workflow_tags'])>1 for r in records),text_sha256_verified=True,
            cache=str(cache),cache_shape=list(embeddings.shape),cache_size=cache_stat.st_size))
        if args.smoke:
            subset=ordered[0][1][:1200];x=np.array(embeddings[[r['embedding_row'] for r in subset]],copy=True)
            t=time.monotonic(); labels=fit(x,24,SEED)
            write_json(args.output/'smoke.json',dict(tasks=len(x),clusters=len(set(labels)),fit_seconds=time.monotonic()-t,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
            print((args.output/'smoke.json').read_text(),flush=True);return
        all_clusters=[];group_rows=[];label_counts=Counter();norm_max=0.0
        handles={r:(args.output/f'assignments-r{r}.jsonl').open('w') for r in [50,100]}
        try:
            for ordinal, ((site,wf), group) in enumerate(ordered):
                idx=[r['embedding_row'] for r in group]
                x=np.array(embeddings[idx],dtype=np.float32,copy=True)
                embeddings._mmap.madvise(mmap.MADV_DONTNEED)
                assert np.isfinite(x).all()
                norm_error=float(np.max(np.abs(np.linalg.norm(x,axis=1)-1)));norm_max=max(norm_max,norm_error)
                assert norm_error<1e-4
                gid=hashlib.sha256((site+'\0'+wf).encode()).hexdigest()[:12]
                entry=dict(site=site,workflow=wf,tasks=len(group)); labels_by_res={}
                for resolution in [50,100]:
                    k=max(1,math.ceil(len(x)/resolution));labels=fit(x,k,SEED)
                    labels_by_res[resolution]=labels
                    members,clusters=partition(x,labels,group,gid,resolution)
                    assert len(members)==len(group)
                    for member in members:handles[resolution].write(json.dumps(member)+'\n')
                    handles[resolution].flush();all_clusters.extend(clusters);label_counts[resolution]+=len(members)
                    entry[f'clusters_r{resolution}']=len(clusters)
                # Same-resolution second-seed diagnostic in the three largest strata.
                if ordinal<3:
                    alternate=fit(x,max(1,math.ceil(len(x)/50)),SEED+1)
                    entry['seed_ari_r50']=float(adjusted_rand_score(labels_by_res[50],alternate))
                group_rows.append(entry)
                if ordinal<8 or ordinal%50==0:print(f'{ordinal+1}/{len(ordered)} {site} {wf} n={len(group)} elapsed={time.monotonic()-start:.1f}s',flush=True)
                del x
        finally:
            for f in handles.values():f.close()
        assert all(label_counts[r]==len(records) for r in [50,100])
        assert cache.stat().st_mtime_ns==cache_stat.st_mtime_ns and cache.stat().st_size==cache_stat.st_size
        with (args.output/'clusters.jsonl').open('w') as f:
            for c in all_clusters:f.write(json.dumps(c,ensure_ascii=False)+'\n')
        with (args.output/'strata.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['site','workflow','tasks','clusters_r50','clusters_r100','seed_ari_r50']);w.writeheader();w.writerows(group_rows)
        make_html(args.output/'review.html',all_clusters)
        result=dict(complete=True,tasks=len(records),sites=len({r['site'] for r in records}),strata=len(strata),
            seed=SEED,implementation_sha256=sha(Path(__file__)),algorithm='BisectingKMeans on unchanged normalized 4096D embeddings; Euclidean objective',
            k_rule='ceil(stratum_tasks / target_size); target_size 50 or 100; not a maximum cluster size',
            embedding_model=plan['model'],embedding_revision=plan['revision'],embedding_instruction=None,
            source_sha256={str(selected_path):sha(selected_path),str(args.root/'semantic-qwen8b/plan.json'):sha(args.root/'semantic-qwen8b/plan.json')},
            cache_identity=dict(path=str(cache),shape=list(embeddings.shape),size=cache_stat.st_size,mtime_ns=cache_stat.st_mtime_ns,
                note='All selected rows checked for finiteness/norm; ordered text hash verified; full 3.45GB cache not rehashed'),
            workflow_counts=dict(Counter(r['primary_workflow'] for r in records)),
            workflow_rules=RULES,workflow_tag_quality='Heuristic/unvalidated; first matching rule sets stratum; fallback includes uncategorized tasks',
            multiple_tag_tasks=sum(len(r['workflow_tags'])>1 for r in records),max_embedding_norm_error=norm_max,
            seed_checks=group_rows[:3],training_data_changed=False,api_calls=0,gpu_hours=0,
            elapsed_seconds=time.monotonic()-start,cpu_seconds=time.process_time(),
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            resolutions={r:dict(clusters=len([c for c in all_clusters if f'-r{r}-' in c['cluster_id']]),
                size_quantiles={str(q):float(np.quantile([c['size'] for c in all_clusters if f'-r{r}-' in c['cluster_id']],q)) for q in [0,.1,.5,.9,.99,1]},
                singletons=sum(c['size']==1 for c in all_clusters if f'-r{r}-' in c['cluster_id']),
                median_cluster_cosine=float(np.median([c['median_cosine'] for c in all_clusters if f'-r{r}-' in c['cluster_id']]))) for r in [50,100]},
            artifacts={f.name:sha(f) for f in sorted(args.output.iterdir()) if f.is_file()})
        write_json(args.output/'summary.json',result)
        print(json.dumps({k:result[k] for k in ['tasks','sites','strata','elapsed_seconds','cpu_seconds','peak_rss_mib','resolutions','seed_checks']},indent=2),flush=True)


if __name__=='__main__':main()
