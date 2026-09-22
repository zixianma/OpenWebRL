#!/usr/bin/env python3
"""CPU-only WebGym task inventory and a review shortlist; never alter training inputs."""
import argparse
from collections import Counter,defaultdict
import hashlib,html,json,re,statistics
from pathlib import Path
from urllib.parse import urlsplit
import pyarrow.parquet as pq

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/task-pool-expansion-20260922')
ALLOWED={'insta-v3','pae-webvoyager'}

def norm(s):return ' '.join(re.findall(r'\w+',str(s).casefold()))
def tokens(s):return frozenset(norm(s).split())
def host(url):return (urlsplit(url if '://' in url else 'https://'+url).hostname or '').removeprefix('www.').casefold()
def bucket(r):return 'hard_7plus' if r['difficulty']>=7 else 'medium_4to6' if r['difficulty']>=4 else 'easy_1to3'
def lines(p):
 with p.open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def intent(r):
 m=r.get('metadata') or {};return r.get('task_name') or r.get('confirmed_task') or r.get('question') or r.get('ques') or r.get('task') or m.get('task') or m.get('intent') or ''
def similarity(a,b):return len(a&b)/len(a|b) if a or b else 1.
def signatures(r):
 words=tokens(r['task_name']);return words

def main():
 current=pq.read_table(REPO/'openwebrl/data/webgym_filtered_popular_2102_cleaned.parquet').to_pylist()
 current_ids={str(r['metadata']['task_id']).split('/')[-1] for r in current}
 current_names={norm(r['metadata']['task']) for r in current};hosts={host(r['metadata']['start_url']) for r in current}
 released=list(lines(ROOT/'openwebrl-rl-tasks.jsonl'));released_ids={str(r['task_id']) for r in released}
 assert current_ids<=released_ids,'Current tasks not covered by released pool; review IDs'
 test=list(lines(ROOT/'webgym-test.jsonl'));test_ids={str(r['task_id']) for r in test}
 heldout_names={norm(r['task_name']) for r in test}
 eval_files=[]
 for relative in ['openwebrl/data/eval/online-mind2web.jsonl','openwebrl/data/eval/deepshop.jsonl','openwebrl/data/eval/webvoyager_fara.jsonl','openwebrl/data/online-mind2web.jsonl']:
  p=REPO/relative;rr=list(lines(p));nn=[intent(r) for r in rr]
  assert all(nn),'Missing heldout text field: '+relative
  heldout_names.update(norm(t) for t in nn);eval_files.append(dict(path=relative,rows=len(rr)))
 rr=pq.read_table(REPO/'openwebrl/data/webvoyager_val.parquet').to_pylist()
 for r in rr:
  text=intent(r) or r.get('prompt',[{}])[-1].get('content','')
  if not text:raise ValueError('Missing WebVoyager validation instruction')
  heldout_names.add(norm(text))
 hist=Counter();bench=Counter();all_diff=Counter();candidates=[];matched={};duplicates=set();reasons=Counter()
 for r in lines(ROOT/'webgym-train.jsonl'):
  hist['raw_train']+=1;bench[r['benchmark_name']]+=1;all_diff[bucket(r)]+=1
  tid=str(r['task_id'])
  if tid in current_ids:matched[tid]=r
  if r['benchmark_name'] not in ALLOWED:reasons['other_benchmark_family']+=1;continue
  hist['allowed_families']+=1
  if r.get('task_id_decomposed_from') is not None:reasons['decomposed']+=1;continue
  hist['original_tasks']+=1
  if tid in released_ids:reasons['in_released_openwebrl_pool']+=1;continue
  if tid in test_ids:reasons['webgym_test_id']+=1;continue
  text=norm(r['task_name'])
  if text in current_names or text in heldout_names:reasons['exact_existing_or_eval_intent']+=1;continue
  if text in duplicates:reasons['exact_duplicate_intent']+=1;continue
  if not host(r['website']) or not r.get('evaluator_reference'):reasons['missing_url_or_rubric']+=1;continue
  duplicates.add(text);hist['remaining_original_train_tasks']+=1
  r['site_host']=host(r['website']);r['same_host_as_current_rl']=r['site_host'] in hosts
  candidates.append(r)
 assert len(matched)==len(current_ids),(len(matched),len(current_ids))
 # Cheap lexical screen, explicitly NOT Qwen3-Embedding-8B semantic deduplication.
 # Existing websites first avoids silently expanding into arbitrary long-tail domains.
 refs=[tokens(r['task_name']) for r in released]+[tokens(x) for x in heldout_names]
 inverted=defaultdict(set)
 for i,words in enumerate(refs):
  for word in words:inverted[word].add(i)
 same=[r for r in candidates if r['same_host_as_current_rl'] and r['difficulty']>=4]
 filtered=[];lexical_rejected=Counter()
 for r in same:
  words=signatures(r);overlaps=Counter(i for word in words for i in inverted.get(word,()))
  best=max((n/(len(words)+len(refs[i])-n) for i,n in overlaps.items()),default=0.)
  r['max_token_jaccard_existing_or_eval']=best
  if best>=.65:lexical_rejected[bucket(r)]+=1;continue
  filtered.append(r)
 # Two candidate lists and a host-capped human/browser-review shortlist of up to100 tasks.
 for name,rows in [('same-host-medium-hard-candidates.jsonl',filtered),('same-host-hard-candidates.jsonl',[r for r in filtered if r['difficulty']>=7])]:
  with (ROOT/name).open('w') as f:
   for r in sorted(rows,key=lambda r:(r['site_host'],-r['difficulty'],str(r['task_id']))):f.write(json.dumps(r,ensure_ascii=False)+'\n')
 def risk(r):
  t=norm(r['task_name']);flags=[]
  if re.search(r'\b(?:202[0-5]|201\d)\b',t):flags.append('dated_reference')
  if any(s in t for s in ['specific ', 'given ', 'your ', 'my ', 'log in','sign in','account','checkout','purchase','book a','reservation']):flags.append('context_or_account_review')
  if re.search(r'next \d+ days|over the next week',t):flags.append('requires_future_observation')
  if r['difficulty']>=12:flags.append('high_constraint_count')
  return flags
 def ordering(r):return (len(risk(r)),r['max_token_jaccard_existing_or_eval'],hashlib.sha256(str(r['task_id']).encode()).hexdigest())
 shortlist=[];site_counts=Counter();selected_words=[];skips=Counter()
 # 50 hard and50 medium: this is a screening allocation, not a proposed train mixture.
 for band in ['hard_7plus','medium_4to6']:
  count=0
  for r in sorted([x for x in filtered if bucket(x)==band],key=ordering):
   if site_counts[r['site_host']]>=5:skips['host_cap']+=1;continue
   words=signatures(r)
   if any(similarity(words,w)>=.65 for w in selected_words):skips['shortlist_lexical_duplicate']+=1;continue
   r=dict(r,review_flags=risk(r));shortlist.append(r);selected_words.append(words);site_counts[r['site_host']]+=1;count+=1
   if count==50:break
 shortlist_name=f'review-shortlist-{len(shortlist)}.jsonl'
 with (ROOT/shortlist_name).open('w') as f:
  for r in shortlist:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 def stats(rows):return dict(tasks=len(rows),hosts=len({host(r['website']) for r in rows}),difficulty=dict(Counter(bucket(r) for r in rows)),sources=dict(Counter(r['benchmark_name'] for r in rows)),domains=dict(Counter(r['domain'] for r in rows)))
 audit=dict(schema=1,sources=json.loads((ROOT/'sources.json').read_text()),raw_counts=dict(hist),rejections=dict(reasons),
  raw_benchmarks=dict(bench),raw_difficulty=dict(all_diff),current_pool=stats(list(matched.values())),
  released_pool=stats(released),remaining=stats(candidates),same_hosts=stats([r for r in candidates if r['same_host_as_current_rl']]),
  same_hosts_medium_hard_before_lexical=stats(same),same_hosts_medium_hard_after_lexical=stats(filtered),
  lexical_rejections=dict(lexical_rejected),shortlist=stats(shortlist),shortlist_host_counts=dict(site_counts),
  eval_files=eval_files,webgym_test_rows=len(test),
  limitations=['Metadata candidates only; current-actor hardness and browser executability unmeasured.',
   'Difficulty counts rubric facts, not measured steps or actor failure probability.',
   'Only original insta-v3/pae-webvoyager train tasks; all released2198 OpenWebRL tasks excluded, including96 absent from current2102.',
   'Exact text/test-ID exclusions plus token-Jaccard0.65 screening; Qwen3-Embedding-8B semantic0.95 dedup and semantic benchmark overlap checks remain.',
   'Existing training host membership is a proxy for initial site choice, not a live availability guarantee.',
   'Target up to50 medium+50 hard for screening, with a shared5-per-host cap; actual count may be lower. Not an approved training mixture. No new allocation or browser execution.'],
  artifacts={name:str(ROOT/name) for name in ['same-host-medium-hard-candidates.jsonl','same-host-hard-candidates.jsonl',shortlist_name,'review.html']},
  manual_review_findings=dict(future_observation_task_ids=[r['task_id'] for r in shortlist if 'requires_future_observation' in r['review_flags']],
   interpretation='Rubric fact-count can label vague/simple prompts hard; candidate quantity is not usable-task yield.'))
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 public=REPO/'openwebrl/docs/arm_results/rl_integration/task-pool-expansion-audit.json';public.write_text(json.dumps(audit,indent=2)+'\n')
 cards=[]
 for r in shortlist:
  esc=html.escape
  cards.append('<article><h3>'+esc(r['site_host'])+' · rubric difficulty '+str(r['difficulty'])+'</h3><p>'+esc(r['task_name'])+'</p><details><summary>Rubric and provenance</summary><pre>'+esc(json.dumps(r,indent=2))+'</pre></details></article>')
 page='''<!doctype html><meta charset="utf-8"><title>WebGym candidate review</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;background:#f4f6fa;color:#142334}article{background:white;padding:20px;margin:16px 0;border:1px solid #ccd4df;border-radius:8px}pre{white-space:pre-wrap;font-size:13px}h3{color:#20558c}</style><h1>Additional WebGym tasks: review shortlist</h1><p>REVIEW_COUNTS metadata candidates, maximum5 per existing training host. Medium=4–6 rubric facts; hard=7+. No browser or actor validation yet. Lexical deduplication only; semantic overlap review remains. Source:microsoft/webgym_tasks (CDLA-Permissive-2.0).</p>'''+''.join(cards)
 page=page.replace('REVIEW_COUNTS',f"{len(shortlist)} total ({sum(r['difficulty']>=7 for r in shortlist)} hard, {sum(r['difficulty']<7 for r in shortlist)} medium)")
 (ROOT/'review.html').write_text(page)
 print(json.dumps({k:audit[k] for k in ['raw_counts','current_pool','same_hosts','same_hosts_medium_hard_after_lexical','shortlist','shortlist_host_counts']},indent=2))

if __name__=='__main__':main()
