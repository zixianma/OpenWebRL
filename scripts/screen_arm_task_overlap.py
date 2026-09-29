#!/usr/bin/env python3
"""Prepare or run pinned Qwen task-overlap screening; never changes training data.

GPU execution requires a separately approved allocation. Cosines route pairs to
review, not automatic acceptance/rejection or a guarantee of benchmark separation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/task-pool-expansion-20260922')
MODEL = 'Qwen/Qwen3-Embedding-8B'
REVISION = '1d8ad4ca9b3dd8059ad90a75d4983776a23d44af'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def prepare(source, output):
    frozen = json.loads((source/'plan.json').read_text())
    names = ['candidate-review-75.jsonl', 'semantic-reference.jsonl']
    for name in names:
        actual = hashlib.sha256((source/name).read_bytes()).hexdigest()
        if actual != frozen['manifest_sha256'][name]:
            raise ValueError('Frozen input hash mismatch: '+name)
    candidates, refs = [read_rows(source/name) for name in names]
    if len({str(r['task_id']) for r in candidates}) != len(candidates):
        raise ValueError('Duplicate candidate IDs')
    texts = [r['task_name'] for r in candidates] + [r['text'] for r in refs]
    if not all(isinstance(t, str) and t.strip() for t in texts):
        raise ValueError('Empty input text')
    manifest = dict(model=MODEL, revision=REVISION, candidates=len(candidates), references=len(refs),
        total_texts=len(texts), source_hashes={k:frozen['manifest_sha256'][k] for k in names},
        text_sha256=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest(),
        embedding=dict(pooling='last nonpadding token', normalize=True, dimensions=4096,
            instruction=None, symmetric=True, dtype='bfloat16', batch_size=8,
            max_tokens=8192, truncation=False, attention='sdpa'),
        review=dict(top_neighbors_per_source=10, high_similarity_threshold=.95,
            threshold_is_not_a_guarantee=True, automatic_rejection=False,
            reference_strata=['released_openwebrl','heldout'], candidate_to_candidate=True),
        resources=dict(gpus=1, gpu_type='h200', cpus=8, memory_gib=120, seconds=1800),
        approved=False, submitted=False, training_data_changed=False)
    output.mkdir(parents=True, exist_ok=True)
    path=output/'plan.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Refusing to replace an existing different overlap plan')
    write_json(path, manifest)
    return candidates, refs, texts, manifest


def nearest_report(candidates, refs, embeddings):
    import numpy as np
    n=len(candidates)
    e=np.asarray(embeddings, dtype=np.float32)
    if e.ndim!=2 or len(e)!=n+len(refs) or not np.isfinite(e).all():
        raise ValueError('Invalid embedding array')
    norm=np.linalg.norm(e,axis=1)
    if not np.allclose(norm,1,atol=1e-4):
        raise ValueError('Embeddings must be unit normalized')
    scores=e[:n]@e.T
    output=[]
    for i,row in enumerate(candidates):
        def ref_matches(kind):
            indices=[k for k,ref in enumerate(refs) if any(
                (p['source']=='released_openwebrl') == (kind=='released_openwebrl')
                for p in ref['provenance'])]
            ranked=sorted(indices,key=lambda k:(-float(scores[i,n+k]),k))[:10]
            return [dict(cosine=float(scores[i,n+k]),**refs[k]) for k in ranked]
        other=sorted((k for k in range(n) if k!=i),key=lambda k:(-float(scores[i,k]),k))[:10]
        output.append(dict(task_id=str(row['task_id']), instruction=row['task_name'],
            existing_training=ref_matches('released_openwebrl'),heldout=ref_matches('heldout'),
            candidates=[dict(task_id=str(candidates[k]['task_id']),instruction=candidates[k]['task_name'],
                             cosine=float(scores[i,k])) for k in other],decision='review'))
    return output


def execute(source, output):
    import fcntl
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run only inside an approved GPU allocation; no login-node model inference')
    candidates,refs,texts,manifest=prepare(source,output)
    with (output/'owner.lock').open('a+') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.time()
        status=dict(complete=False,stage='loading',job_id=os.environ['SLURM_JOB_ID'],started_epoch=started)
        write_json(output/'status.json',status)
        try:
            tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,padding_side='left',local_files_only=True)
            lengths=[len(tok(t)['input_ids']) for t in texts]
            if max(lengths)>8192:
                raise ValueError('Input exceeds max tokens; refusing silent truncation')
            model=AutoModel.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.bfloat16,
                attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
            blocks=[]
            with torch.inference_mode():
                for first in range(0,len(texts),8):
                    batch=tok(texts[first:first+8],padding=True,truncation=False,return_tensors='pt').to('cuda')
                    # Left padding: the final position is always the last real token.
                    out=model(**batch,use_cache=False).last_hidden_state[:,-1].float()
                    blocks.append(torch.nn.functional.normalize(out,p=2,dim=1).cpu().numpy())
                    status.update(stage='embedding',completed_texts=min(first+8,len(texts)),updated_epoch=time.time())
                    write_json(output/'status.json',status)
            embeddings=np.concatenate(blocks)
            if embeddings.shape!=(len(texts),4096):
                raise ValueError('Unexpected embedding shape')
            np.save(output/'embeddings.npy',embeddings,allow_pickle=False)
            report=nearest_report(candidates,refs,embeddings)
            write_json(output/'neighbors.json',report)
            status.update(complete=True,stage='awaiting_pair_review',total_texts=len(texts),
                total_tokens=sum(lengths),max_tokens=max(lengths),elapsed_seconds=time.time()-started,
                high_similarity_candidates=sum(any(m['cosine']>=.95 for key in
                    ['existing_training','heldout','candidates'] for m in row[key]) for row in report),
                model_revision=REVISION,training_data_changed=False)
            write_json(output/'status.json',status)
            print(json.dumps(status))
        except Exception as exc:
            status.update(complete=False,stage='failed',error_type=type(exc).__name__,updated_epoch=time.time())
            write_json(output/'status.json',status)
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'screening-v2')
    p.add_argument('--output',type=Path,default=ROOT/'screening-v2/overlap-qwen8b')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.execute: execute(args.source,args.output)
    else: print(json.dumps(prepare(args.source,args.output)[-1],indent=2))


if __name__=='__main__': main()
