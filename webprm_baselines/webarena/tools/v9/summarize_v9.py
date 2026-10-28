"""Pool v9 cell outcomes across all v9b/c/d/e packages (a valid outcome wins over invalid ones; fuzzy-judge
invalids are labelled F). Prints per-site x method valid/success counts, invalid breakdown and API spend."""
import json, glob, re, sys
from collections import defaultdict, Counter
D = '/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021'
SITE = {'reddit': 'Reddit', 'shop': 'Shopping', 'cms': 'CMS', 'gitlab': 'GitLab'}
best = {}
for f in glob.glob(D+'/launch-preparation-v9[b-f]-*/campaign/attempt-*/cells/*/episodes/*/*/result.json'):
    pkg = f.split('launch-preparation-')[1].split('/')[0]
    if 'probe' in pkg: continue
    site = SITE[re.match(r'v9[b-f]-([a-z]+)', pkg).group(1)]
    d = json.load(open(f)); key = (site, d['selector'], d['task_id'])
    err = str(d.get('error', ''))
    o = 'S' if d['valid'] and d.get('success') else 'V' if d['valid'] else 'F' if 'fuzzy' in err.lower() else 'X:' + re.sub(r'\d+', 'N', err[:60])
    if key not in best or best[key][0] not in 'SV':
        best[key] = (o, pkg)
tab = defaultdict(Counter)
for (site, sel, _), (o, _) in best.items():
    tab[(site, sel)][o[0] if o[0] in 'SVF' else 'X'] += 1
sels = ['no_search', 'arbiter7b', 'shepherd8b']; tot = defaultdict(Counter)
print(f"{'site':9}{'method':12}{'cells':>6}{'valid':>6}{'succ':>5}{'rate':>7}{'fuzzy':>6}{'other_inv':>10}")
for site in ['Reddit', 'Shopping', 'CMS', 'GitLab']:
    for sel in sels:
        c = tab[(site, sel)]; v = c['S'] + c['V']
        for k in c: tot[sel][k] += c[k]
        print(f"{site:9}{sel:12}{sum(c.values()):>6}{v:>6}{c['S']:>5}{(100*c['S']/v if v else 0):>6.1f}%{c['F']:>6}{c['X']:>10}")
for sel in sels:
    c = tot[sel]; v = c['S'] + c['V']
    print(f"{'ALL':9}{sel:12}{sum(c.values()):>6}{v:>6}{c['S']:>5}{(100*c['S']/v if v else 0):>6.1f}%{c['F']:>6}{c['X']:>10}")
print('other invalid:', Counter(o[2:] for o, _ in best.values() if o.startswith('X')))
spent = sum(json.load(open(p)).get('charged', 0) for p in glob.glob(D+'/launch-preparation-v9[b-f]-*/campaign/shared-budget.json') if 'probe' not in p)
print(f'API spent (v9b-e packages): ${spent:.2f}')
