#!/usr/bin/env python3
"""Paper-based Qwen cosine greedy dedup; original upstream program is unreleased.

--prepare is CPU metadata/tokenization only. --execute requires a matching exact
resource approval receipt and a GPU allocation. No browser/API calls. A prepared
output remains a metadata/semantic candidate pool, not a quality-approved dataset.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from rebuild_arm_task_pool import ROOT, read_rows, sha, frozen_rows, frozen_json

MODEL = 'Qwen/Qwen3-Embedding-8B'
REVISION = '1d8ad4ca9b3dd8059ad90a75d4983776a23d44af'


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def inputs(source):
    audit = json.loads((source / 'audit.json').read_text())
    for name in ['candidates.jsonl', 'semantic-references.jsonl']:
        if sha(source / name) != audit['artifacts'][name]['sha256']:
            raise ValueError('Frozen metadata mismatch: ' + name)
    refs = list(read_rows(source / 'semantic-references.jsonl'))
    candidates = list(read_rows(source / 'candidates.jsonl'))
    return refs, candidates


def prepare(source, output):
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (90, 95))
    from transformers import AutoTokenizer
    refs, candidates = inputs(source)
    texts = [r['text'] for r in refs] + [r['task_name'] for r in candidates]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    lengths = []
    for start in range(0, len(texts), 1024):
        lengths.extend(map(len, tok(texts[start:start + 1024], truncation=False)['input_ids']))
    if max(lengths) > 8192:
        raise ValueError('Long inputs require review; no silent truncation')
    old = json.loads((ROOT / 'screening-v2/overlap-qwen8b/status.json').read_text())
    token_extrapolation = sum(lengths) / old['total_tokens'] * old['elapsed_seconds']
    plan = dict(schema=1, model=MODEL, revision=REVISION, references=len(refs), candidates=len(candidates),
        total_texts=len(texts), text_sha256=hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
        total_tokens=sum(lengths), max_tokens=max(lengths), threshold=0.95,
        source_audit_sha256=sha(source / 'audit.json'),
        embedding=dict(pooling='last_nonpadding_token', dimensions=4096, normalized=True, dtype='bfloat16',
            instruction=None, batch=8, truncation=False, attention='sdpa'),
        dedup=dict(cosine_dtype='float32', tf32=False, threshold_rule='greater_than_or_equal',
            order='source_train_order', block_size=256, references='all_heldout_and_released_training_first',
            candidate_rule='compare_only_to_previously_retained_candidates',
            cross_site_near_duplicates='quarantine_with_neighbor_ID_and_cosine_for_review'),
        exact_original_implementation=False, website_quality_still_pending=True,
        resources=dict(gpus=1, gpu_type='h200', cpus=8, memory_gib=120, seconds=10800),
        estimate=dict(prior_texts=old['total_texts'], prior_tokens=old['total_tokens'],
            prior_measured_seconds=old['elapsed_seconds'], token_scaled_seconds=round(token_extrapolation),
            caveat='Extrapolation includes old startup; new length/padding/dedup costs may differ.'),
        approval_required=True, submitted=False, api_budget=0, browser_budget=0, training_data_changed=False)
    output.mkdir(parents=True, exist_ok=True)
    frozen_json(output / 'plan.json', plan)
    print(json.dumps(plan, indent=2))


def choose_block(prior_values, prior_indices, within, threshold, base_index):
    """Exact greedy semantics: a rejected row must never suppress later rows."""
    import numpy as np
    kept, decisions = [], []
    for i in range(len(prior_values)):
        score, neighbor = float(prior_values[i]), int(prior_indices[i])
        if kept:
            local = int(np.argmax(within[i, kept]))
            k = kept[local]
            if float(within[i, k]) > score:
                score, neighbor = float(within[i, k]), base_index + k
        rejected = score >= threshold
        decisions.append(dict(rejected=rejected, nearest_index=neighbor, cosine=score))
        if not rejected:
            kept.append(i)
    return kept, decisions


def execute(source, output):
    import fcntl
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    plan_path = output / 'plan.json'
    plan = json.loads(plan_path.read_text())
    approval = json.loads((output / 'approval.json').read_text())
    if not approval.get('approved') or approval.get('plan_sha256') != sha(plan_path) or approval.get('resources') != plan['resources']:
        raise ValueError('Missing or mismatched exact compute approval')
    job = os.environ.get('SLURM_JOB_ID')
    if not job or not torch.cuda.is_available():
        raise RuntimeError('Embedding inference must run inside an approved GPU allocation')
    # Submitter records every attempt and remaining total, including failed work.
    budgets = approval.get('attempt_budgets_seconds', {})
    seconds = budgets.get(job)
    if not isinstance(seconds, int) or not 0 < seconds <= plan['resources']['seconds']:
        raise ValueError('Current job is absent from the remaining-budget ledger')
    deadline = time.monotonic() + seconds - 180
    if sha(source / 'audit.json') != plan['source_audit_sha256']:
        raise ValueError('Source audit changed')
    refs, candidates = inputs(source)
    texts = [r['text'] for r in refs] + [r['task_name'] for r in candidates]
    if hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest() != plan['text_sha256']:
        raise ValueError('Text order changed')
    with (output / 'owner.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads((output / 'status.json').read_text()) if (output / 'status.json').exists() else {}
        if previous.get('complete'):
            for artifact in previous['artifacts'].values():
                if sha(artifact['path']) != artifact['sha256']:
                    raise ValueError('Completed semantic artifact hash mismatch')
            print('Verified output already exists; no GPU work needed')
            return
        status = dict(previous, complete=False, job_id=job, updated_epoch=time.time(), stage='loading')
        write_json(output / 'status.json', status)
        try:
            path = output / 'embeddings.npy'
            done = previous.get('completed_texts', 0)
            if done and not path.exists():
                raise ValueError('Resume cursor exists without embeddings')
            array = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+', dtype=np.float32, shape=(len(texts), 4096))
            if array.shape != (len(texts), 4096):
                raise ValueError('Unexpected cached embedding shape')
            if done < len(texts):
                tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side='left', local_files_only=True)
                model = AutoModel.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
                with torch.inference_mode():
                    for first in range(done, len(texts), 8):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Approved attempt nearing its reserve; embeddings are resumable')
                        batch = tok(texts[first:first+8], padding=True, truncation=False, return_tensors='pt').to('cuda')
                        if batch['input_ids'].shape[1] > 8192:
                            raise ValueError('No silent truncation')
                        hidden = model(**batch, use_cache=False).last_hidden_state[:, -1].float()
                        values = torch.nn.functional.normalize(hidden, p=2, dim=1).cpu().numpy()
                        if not np.isfinite(values).all():
                            raise ValueError('Nonfinite embeddings')
                        array[first:first+len(values)] = values
                        # Durably commit before advancing cursor. Redo at most128 texts.
                        if (first + 8) % 128 == 0 or first + 8 >= len(texts):
                            array.flush()
                            status.update(stage='embedding', completed_texts=min(first + 8, len(texts)), updated_epoch=time.time())
                            write_json(output / 'status.json', status)
                del model
                torch.cuda.empty_cache()
            array.flush()
            status.update(stage='deduplication', completed_texts=len(texts), updated_epoch=time.time())
            write_json(output / 'status.json', status)
            torch.backends.cuda.matmul.allow_tf32 = False
            embedding = torch.from_numpy(np.asarray(array)).to('cuda')
            if not torch.isfinite(embedding).all() or not torch.allclose(embedding.norm(dim=1), torch.ones(len(texts), device='cuda'), atol=1e-4):
                raise ValueError('Embedding normalization failed')
            retained = list(range(len(refs)))
            decisions = []
            for first in range(len(refs), len(texts), 256):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Approved attempt budget reserve reached during deduplication')
                block = embedding[first:first+256]
                scores = block @ embedding[retained].T
                values, indices = scores.max(dim=1)
                prior_indices = np.asarray(retained)[indices.cpu().numpy()]
                within = (block @ block.T).cpu().numpy()
                kept, result = choose_block(values.cpu().numpy(), prior_indices, within, plan['threshold'], first)
                decisions.extend(result)
                retained.extend(first+i for i in kept)
                status.update(candidate_decisions=len(decisions), kept_candidates=len(retained)-len(refs), updated_epoch=time.time())
                write_json(output / 'status.json', status)
            kept_rows, rejected_rows, evidence = [], [], []
            for row, decision in zip(candidates, decisions, strict=True):
                idx = decision['nearest_index']
                neighbor = refs[idx] if idx < len(refs) else dict(source='retained_candidate', task_id=str(candidates[idx-len(refs)]['task_id']))
                evidence.append(dict(task_id=str(row['task_id']), **decision, nearest_source=neighbor['source'], nearest_task_id=neighbor['task_id']))
                (rejected_rows if decision['rejected'] else kept_rows).append(row)
            # Incomplete artifact sets from interrupted output writes are preserved.
            # Use attempt-specific files; status publishes only a complete set.
            names = {'kept': f'kept-{job}.jsonl', 'quarantine': f'semantic-quarantine-{job}.jsonl', 'decisions': f'decisions-{job}.jsonl'}
            for key, rows in [('kept', kept_rows), ('quarantine', rejected_rows), ('decisions', evidence)]:
                frozen_rows(output / names[key], rows)
            if len(kept_rows) + len(rejected_rows) != len(candidates):
                raise ValueError('Unaccounted candidates')
            status.update(complete=True, stage='semantic_complete_site_and_task_quality_pending',
                kept=len(kept_rows), quarantined=len(rejected_rows),
                artifacts={k: dict(path=str(output/n), sha256=sha(output/n)) for k,n in names.items()},
                training_data_changed=False, updated_epoch=time.time())
            write_json(output / 'status.json', status)
        except Exception as exc:
            status.update(complete=False, stage='failed_or_budget_stop', error_type=type(exc).__name__, error=str(exc), updated_epoch=time.time())
            write_json(output / 'status.json', status)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'curation-v3-20260929')
    parser.add_argument('--output', type=Path, default=ROOT / 'curation-v3-20260929/semantic-qwen8b')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    execute(args.source, args.output) if args.execute else prepare(args.source, args.output)
