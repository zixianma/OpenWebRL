#!/usr/bin/env python3
"""Freeze a proposed browser-screening cohort and semantic-overlap inputs.

CPU metadata only: no embeddings, APIs, browser runs, or training-pool changes.
"""
import collections
import hashlib
import json
from pathlib import Path
import resource
import time
from urllib.parse import urlsplit

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/task-pool-expansion-20260922')
DEST=ROOT/'screening-v2'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def host(url):
    return (urlsplit(url if '://' in url else 'https://'+url).hostname or '').removeprefix('www.').lower()


def intent(row):
    m=row.get('metadata') or {}
    return row.get('task_name') or row.get('confirmed_task') or row.get('question') or row.get('ques') or row.get('task') or m.get('task') or m.get('intent') or ''


def dump(path, rows):
    raw=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows)
    if path.exists() and path.read_text()!=raw:
        raise ValueError('Refusing to replace frozen cohort: '+str(path))
    path.write_text(raw)
    return hashlib.sha256(raw.encode()).hexdigest()


def main():
    import pyarrow.parquet as pq
    resource.setrlimit(resource.RLIMIT_CPU,(60,65))
    started=time.monotonic(); DEST.mkdir(exist_ok=True)
    candidates=read_rows(ROOT/'review-shortlist-75.jsonl')
    released=read_rows(ROOT/'openwebrl-rl-tasks.jsonl')
    active=pq.read_table(REPO/'openwebrl/data/webgym_filtered_popular_2102_cleaned.parquet').to_pylist()
    active_ids={str(r['metadata']['task_id']).split('/')[-1] for r in active}
    weights=collections.Counter(r['site_host'] for r in candidates)
    bins=collections.defaultdict(list)
    for row in released:
        site=host(row['website'])
        if str(row['task_id']) in active_ids and site in weights:
            bins[site].append(row)
    # One existing task per represented host, then a second on the six most
    # represented hosts if available. Hash-based choice, independent of outcomes.
    for rows in bins.values():
        rows.sort(key=lambda r:hashlib.sha256(('screen-v2-seed42:'+str(r['task_id'])).encode()).hexdigest())
    order=sorted(weights,key=lambda h:(-weights[h],h)); control=[]
    for offset in [0,1]:
        for site in order:
            if len(control)>=25: break
            if len(bins[site])>offset:
                control.append(dict(bins[site][offset],site_host=site,cohort='existing_pool_control'))
    if len(control)!=25: raise ValueError('Insufficient matched-site control tasks')
    if {str(r['task_id']) for r in control}&{str(r['task_id']) for r in candidates}: raise ValueError('Cohort overlap')
    refs=[dict(source='released_openwebrl',task_id=str(r['task_id']),text=r['task_name']) for r in released]
    refs += [dict(source='webgym_test',task_id=str(r['task_id']),text=r['task_name']) for r in read_rows(ROOT/'webgym-test.jsonl')]
    for name in ['online-mind2web.jsonl','deepshop.jsonl','webvoyager_fara.jsonl']:
        for i,row in enumerate(read_rows(REPO/'openwebrl/data/eval'/name)):
            text=intent(row)
            if not text: raise ValueError('Missing heldout intent')
            refs.append(dict(source=name,task_id=str(row.get('task_id',i)),text=text))
    for i,row in enumerate(pq.read_table(REPO/'openwebrl/data/webvoyager_val.parquet').to_pylist()):
        text=intent(row) or row.get('prompt',[{}])[-1].get('content','')
        if not text: raise ValueError('Missing WebVoyager validation intent')
        refs.append(dict(source='webvoyager_val.parquet',task_id=str(row.get('task_id',i)),text=text))
    for i,row in enumerate(read_rows(REPO/'openwebrl/data/online-mind2web.jsonl')):
        text=intent(row)
        if not text: raise ValueError('Missing OM2W intent')
        refs.append(dict(source='online-mind2web.jsonl_legacy',task_id=str(row.get('task_id',i)),text=text))
    # Keep source provenance while eliminating identical texts for embedding.
    unique={}
    for row in refs:
        key=' '.join(row['text'].lower().split())
        entry=unique.setdefault(key,dict(text=row['text'],provenance=[]))
        entry['provenance'].append(dict(source=row['source'],task_id=row['task_id']))
    hashes={}
    for name,rows in [('candidate-review-75.jsonl',candidates),('existing-control-25.jsonl',control),
                     ('semantic-reference.jsonl',list(unique.values()))]:
        hashes[name]=dump(DEST/name,rows)
    summary=dict(status='metadata_prepared_not_browser_ready',candidate_tasks=len(candidates),
        candidate_hard=sum(r['difficulty']>=7 for r in candidates),candidate_medium=sum(4<=r['difficulty']<7 for r in candidates),
        candidate_hosts=len(weights),existing_control_tasks=len(control),control_hosts=len({r['site_host'] for r in control}),
        control_difficulty_histogram=dict(collections.Counter(r['difficulty'] for r in control)),
        semantic_reference_texts=len(unique),semantic_reference_source_rows=dict(collections.Counter(r['source'] for r in refs)),
        manifest_sha256=hashes,
        proposed_execution=dict(actor='one frozen outcome-only actor for both cohorts; checkpoint to finalize',
            attempts_per_task=5,max_turns=15,primary_trajectory_cap=500,
            candidate_budget_before_filtering=375,control_budget=125,
            interleave_cohorts=True,judge='gpt-4.1/action_history',arm='released frozen SelectionARM, scoring only; no action selection',
            metric='mixed and valid all-failure group yield, usable ARM labels, and useful groups per browser-hour',
            rescue='bounded separate stronger-policy check only for valid all-failure tasks; exact budget not yet chosen'),
        prerequisites=['Review revised Jev/GPT quality questions and disagreements; no automatic rejection.',
            'Semantic candidate-to-reference and candidate-to-candidate overlap check; review borderline matches.',
            'Finalize fixed actor/temperatures and resource/API limits before browser execution.'],
        training_pool_changed=False,resources=dict(seconds=time.monotonic()-started,gpus=0,api_calls=0),
        limitations=['Existing controls match websites, not empirical difficulty; report site-stratified results.',
            'Five valid failures do not establish learnability; ARM label yield is not verified progress.',
            'Candidate count is an upper bound before quality and semantic filtering.'])
    (DEST/'plan.json').write_text(json.dumps(summary,indent=2)+'\n')
    (REPO/'openwebrl/docs/arm_results/rl_integration/task-pool-screening-v2.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
