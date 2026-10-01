#!/usr/bin/env python3
"""Build a local interactive cluster map and bounded semantic sampling diagnostic.

Uses cached embeddings only. Samples are diagnostic, not a training manifest.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from collections import Counter,defaultdict
import hashlib
import heapq
import json
import mmap
from pathlib import Path
import resource
import time
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

REPO=Path(__file__).resolve().parents[1]
SEED=20260930
BUDGETS=[500,1000,2000]
METHODS=['weighted_coverage','cluster_round_robin','farthest_point','site_balanced_random']


def read(path):
    return [json.loads(l) for l in Path(path).open() if l.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quotas(budget, capacities, weights):
    if budget<len(capacities) or budget>sum(capacities.values()):
        raise ValueError('Budget cannot satisfy site coverage/capacities')
    result={s:1 for s in capacities}
    # Deterministic sequential highest-averages apportionment with a site floor.
    heap=[(-weights[s]/(result[s]+1),s) for s in capacities if result[s]<capacities[s]]
    heapq.heapify(heap)
    for _ in range(budget-len(result)):
        _,s=heapq.heappop(heap);result[s]+=1
        if result[s]<capacities[s]:heapq.heappush(heap,(-weights[s]/(result[s]+1),s))
    return result


def facility_order(sim, weights, budget, initial_coverage=None):
    """Exact lazy greedy for this finite, nonnegative proxy similarity matrix."""
    covered=(np.zeros(len(sim),dtype=np.float32) if initial_coverage is None
             else np.asarray(initial_coverage,dtype=np.float32).copy())
    if covered.shape!=(len(sim),) or not np.isfinite(covered).all() or (covered<0).any():
        raise ValueError('Invalid initial coverage')
    if budget<0 or budget>sim.shape[1]:raise ValueError('Invalid selection budget')
    gains=weights@np.maximum(sim-covered[:,None],0)
    heap=[(-float(g),j) for j,g in enumerate(gains)];heapq.heapify(heap)
    selected=[]
    while len(selected)<budget:
        _,j=heapq.heappop(heap)
        gain=float(weights@np.maximum(sim[:,j]-covered,0))
        if not heap or gain>=-heap[0][0]-1e-7:
            selected.append(j);np.maximum(covered,sim[:,j],out=covered)
        else:heapq.heappush(heap,(-gain,j))
    return selected


def run(root,output):
    start=time.monotonic();output.mkdir(parents=True,exist_ok=False)
    cluster_root=root/'clusters-20260930/run'
    clusters=read(cluster_root/'clusters.jsonl');cluster_ix={c['cluster_id']:i for i,c in enumerate(clusters)}
    members=defaultdict(list);record={};assignments={}
    for r in [50,100]:
        for x in read(cluster_root/f'assignments-r{r}.jsonl'):
            members[x['cluster_id']].append(x)
            if r==50:record[x['embedding_row']]=x
            assignments[(x['embedding_row'],r)]=x['cluster_id']
    embeddings=np.load(root/'semantic-qwen8b/embeddings.npy',mmap_mode='r')
    centers=np.empty((len(clusters),4096),dtype=np.float32)
    for i,c in enumerate(clusters):
        x=np.asarray(embeddings[[r['embedding_row'] for r in members[c['cluster_id']]]])
        centers[i]=x.mean(axis=0);centers[i]/=np.linalg.norm(centers[i])
        if i%100==0:embeddings._mmap.madvise(mmap.MADV_DONTNEED)
    embeddings._mmap.madvise(mmap.MADV_DONTNEED)
    sites=sorted({c['site'] for c in clusters});site_meta={};candidates={};audits={};fine_indices={}
    rng=np.random.default_rng(SEED)
    for site in sites:
        idx=[i for i,c in enumerate(clusters) if c['site']==site]
        fine=[i for i in idx if '-r50-' in clusters[i]['cluster_id']];fine_indices[site]=fine
        if len(fine)>2:
            model=PCA(n_components=2,svd_solver='randomized',iterated_power=3,random_state=SEED)
            model.fit(centers[fine]);xy=model.transform(centers[idx]);variance=model.explained_variance_ratio_.tolist()
        elif len(fine)==2:
            direction=centers[fine[1]]-centers[fine[0]];direction/=max(np.linalg.norm(direction),1e-8)
            xy=np.column_stack(((centers[idx]-centers[fine].mean(axis=0))@direction,np.zeros(len(idx))));variance=[1.,0.]
        else:xy=np.zeros((len(idx),2));variance=[0.,0.]
        for i,coord in zip(idx,xy):clusters[i]['xy']=[round(float(x),6) for x in coord]
        all_rows=sorted(r['embedding_row'] for i in fine for r in members[clusters[i]['cluster_id']])
        audit=set(rng.choice(all_rows,size=64,replace=False).tolist()) if len(all_rows)>=200 else set()
        audits[site]=sorted(audit)
        eligible=[]
        for i in fine:
            values=[r for r in members[clusters[i]['cluster_id']] if r['embedding_row'] not in audit]
            values.sort(key=lambda r:(-r['cosine_to_centroid'],r['embedding_row']))
            if len(all_rows)<=1000 or len(values)<=6:chosen=values
            else:
                middle=rng.choice(np.arange(2,len(values)-1),size=3,replace=False)
                chosen=[values[0],values[1],values[-1],*[values[int(j)] for j in middle]]
            eligible.extend(r['embedding_row'] for r in chosen)
        candidates[site]=sorted(set(eligible))
        site_meta[site]=dict(tasks=len(all_rows),fine_clusters=len(fine),coarse_clusters=len(idx)-len(fine),
            candidate_tasks=len(eligible),audit_tasks=len(audit),pca_variance=variance)
    budgets={b:quotas(b,{s:len(v) for s,v in candidates.items()},
                          {s:len(fine_indices[s])**.5 for s in sites}) for b in BUDGETS}
    selections={b:{m:[] for m in METHODS} for b in BUDGETS};diagnostics=[]
    for count,site in enumerate(sorted(sites,key=lambda s:-site_meta[s]['tasks'])):
        ids=candidates[site];x=np.asarray(embeddings[ids]);fine=fine_indices[site]
        sim=np.maximum(centers[fine]@x.T,0)
        weights=np.sqrt(np.asarray([clusters[i]['size'] for i in fine],dtype=np.float64));weights/=weights.sum()
        maximum=budgets[max(BUDGETS)][site]
        facility=facility_order(sim,weights,maximum)
        pair=np.clip(x@x.T,-1,1)
        far=[facility[0]];closest=pair[:,far[0]].copy();closest[far[0]]=np.inf
        while len(far)<maximum:
            j=int(np.argmin(closest));far.append(j);np.maximum(closest,pair[:,j],out=closest);closest[j]=np.inf
        lookup={row:j for j,row in enumerate(ids)};queues=[]
        for i in sorted(fine,key=lambda i:(-clusters[i]['size'],clusters[i]['cluster_id'])):
            values=sorted((r for r in members[clusters[i]['cluster_id']] if r['embedding_row'] in lookup),
                          key=lambda r:(-r['cosine_to_centroid'],r['embedding_row']))
            queues.append([lookup[r['embedding_row']] for r in values])
        rr=[q[k] for k in range(max(map(len,queues))) for q in queues if k<len(q)]
        random=rng.permutation(len(ids)).tolist()
        orders=dict(weighted_coverage=facility,cluster_round_robin=rr,farthest_point=far,site_balanced_random=random)
        probe=np.asarray(embeddings[audits[site]]) if audits[site] else None
        probe_sim=np.clip(probe@x.T,-1,1) if probe is not None else None
        for b in BUDGETS:
            k=budgets[b][site]
            for method,order in orders.items():
                chosen=order[:k];assert len(chosen)==len(set(chosen))==k
                selected=[ids[j] for j in chosen]
                assert not(set(selected)&set(audits[site]))
                selections[b][method].extend(selected)
                if probe_sim is not None:
                    nearest=probe_sim[:,chosen].max(axis=1)
                    diagnostics.append(dict(site=site,budget=b,method=method,audit_tasks=len(nearest),
                        nearest_cosines=nearest.tolist(),mean=float(nearest.mean())))
        del x,sim,pair
        embeddings._mmap.madvise(mmap.MADV_DONTNEED)
        if count<10:print(site,site_meta[site]['tasks'],'candidates',len(ids),'quota2000',maximum,
                           'elapsed',round(time.monotonic()-start,1),flush=True)
    counts={}
    for b in BUDGETS:
        counts[b]={}
        for method,rows in selections[b].items():
            assert len(rows)==len(set(rows))==b
            ct=Counter(assignments[(row,r)] for row in rows for r in [50,100]);counts[b][method]=ct
    for c in clusters:
        c['selected']={b:{m:counts[b][m].get(c['cluster_id'],0) for m in METHODS} for b in BUDGETS}
    scores=[]
    for b in BUDGETS:
        for method in METHODS:
            ds=[d for d in diagnostics if d['budget']==b and d['method']==method]
            values=np.concatenate([d['nearest_cosines'] for d in ds])
            scores.append(dict(budget=b,method=method,mean_cosine=float(values.mean()),p10_cosine=float(np.quantile(values,.1)),
                audit_tasks=len(values),sites=len(ds),per_site_means={d['site']:d['mean'] for d in ds}))
    (output/'diagnostic-selections.json').write_text(json.dumps(selections)+'\n')
    (output/'audit-task-rows.json').write_text(json.dumps(audits)+'\n')
    payload=dict(clusters=clusters,sites=site_meta,budgets=budgets,scores=scores)
    (output/'visual-data.json').write_text(json.dumps(payload,ensure_ascii=False)+'\n')
    template=(REPO/'scripts/templates/arm_task_clusters.html').read_text()
    (output/'index.html').write_text(template.replace('__CLUSTER_DATA__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')))
    summary=dict(tasks=len(record),sites=len(sites),fine_clusters=len(fine_indices and [c for c in clusters if '-r50-' in c['cluster_id']]),
        algorithm='Site-balanced weighted facility-location on cluster-centroid proxies and a common candidate shortlist',
        quota='One per site; additional slots via highest-averages apportionment weighted by sqrt(fine_cluster_count), capped by candidate capacity',
        objective='Within site: sum_c sqrt(n_c) max_j max(0, cosine(normalized_cluster_centroid_c, task_embedding_j))',
        candidate_rule='All eligible tasks on sites<=1000; otherwise per cluster two central, three seeded random, one boundary; 64 audit tasks withheld on sites>=200',
        candidate_tasks=sum(map(len,candidates.values())),audit_tasks=sum(map(len,audits.values())),
        scores=scores,seed=SEED,selection_is_diagnostic_only=True,training_data_changed=False,
        caveats=['No actor/ARM/task-success measurement; scores are embedding coverage proxies.',
            'Audit tasks are excluded from candidate selection but contributed to the existing unsupervised clustering.',
            'All methods share website quotas and the same candidate shortlist; this does not measure quality of the quotas.',
            'Centroid-proxy objective is not exact all-task facility location. Workflow tags do not become additional hard sampling quotas.',
            'PCA uses fine-cluster centers separately within each website; distances across websites are not comparable; 2D omits most variance.'],
        input_sha256={name:sha(cluster_root/name) for name in ['clusters.jsonl','assignments-r50.jsonl','assignments-r100.jsonl']},
        implementation_sha256=sha(Path(__file__)),template_sha256=sha(REPO/'scripts/templates/arm_task_clusters.html'),
        elapsed_seconds=time.monotonic()-start,cpu_seconds=time.process_time(),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:summary[k] for k in ['candidate_tasks','audit_tasks','elapsed_seconds','cpu_seconds','peak_rss_mib']},indent=2),flush=True)
    print(json.dumps([r for r in scores if r['budget']==2000],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();resource.setrlimit(resource.RLIMIT_CPU,(180,185))
    with threadpool_limits(limits=1):run(a.root,a.output)
