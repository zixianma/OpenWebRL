import json, collections, sys
from sklearn.metrics import cohen_kappa_score
rows = [json.loads(l) for l in open(sys.argv[1])]
o4 = {}
for r in rows: o4[r['path']] = (r['o4_verdict'], r['o4_cost_usd'], r['o4_usage'])
print('cells', len(o4), 'o4-mini success', sum(v[0] for v in o4.values()), 'cost $%.2f' % sum(v[1] for v in o4.values()),
      'mean in/out/reason tokens', [round(sum((v[2].get(k) or 0) for v in o4.values())/len(o4)) for k in ('prompt_tokens','completion_tokens')],
      round(sum((v[2].get('completion_tokens_details') or {}).get('reasoning_tokens',0) for v in o4.values())/len(o4)))
for m in sorted({r['model'] for r in rows}):
    rs = [r for r in rows if r['model'] == m]; ok = [r for r in rs if r.get('verdict') is not None]
    a = [r['o4_verdict'] for r in ok]; b = [r['verdict'] for r in ok]
    cm = collections.Counter((x, y) for x, y in zip(a, b))
    u = [r.get('usage') or {} for r in rs]
    print(f"\n{m}: calls {len(rs)} parsed {len(ok)} errors {collections.Counter(r.get('http_status') or r.get('finish_reason') for r in rs if r.get('verdict') is None)}")
    print(f"  agree {sum(x==y for x,y in zip(a,b))}/{len(ok)} = {sum(x==y for x,y in zip(a,b))/len(ok):.3f}  kappa {cohen_kappa_score(a,b):.3f}")
    print(f"  success o4 {sum(a)} vs new {sum(b)}; o4 yes/new no {cm[(True,False)]}  o4 no/new yes {cm[(False,True)]}")
    print(f"  cost ${sum(r.get('cost_usd') or 0 for r in rs):.2f}  per call ${sum(r.get('cost_usd') or 0 for r in rs)/len(rs):.4f}  mean in/out/reason "
          f"{round(sum(x.get('prompt_tokens',0) for x in u)/len(u))}/{round(sum(x.get('completion_tokens',0) for x in u)/len(u))}/"
          f"{round(sum((x.get('completion_tokens_details') or {}).get('reasoning_tokens',0) for x in u)/len(u))}")
    by = collections.defaultdict(lambda: [0, 0])
    for r in ok:
        sel = r['cell'].split('__')[-1]; by[sel][0] += 1; by[sel][1] += r['o4_verdict'] == r['verdict']
    print('  agreement by method', {k: f'{v[1]}/{v[0]}' for k, v in sorted(by.items())})
both = collections.defaultdict(dict)
for r in rows:
    if r.get('verdict') is not None: both[r['path']][r['model']] = r['verdict']
pairs = [(v['o3-2025-04-16'], v['gpt-5.6-terra']) for v in both.values() if len(v) == 2]
print(f"\no3 vs terra: agree {sum(x==y for x,y in pairs)}/{len(pairs)} kappa {cohen_kappa_score([p[0] for p in pairs],[p[1] for p in pairs]):.3f}")
