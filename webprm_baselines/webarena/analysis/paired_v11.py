"""Pool v11b/v11c WebArena cells (a valid outcome wins over invalid ones) and print the paired
no_search vs piotr_selectionarm comparison: strict valid-only, + gpt-4.1 fuzzy re-score, and
+ repeated-infra ARM cells counted as ARM failures. 95% CI: paired-proportion normal interval;
p: two-sided exact sign test on discordant tasks (scipy.stats.binomtest).
Run with PYTHONPATH=/gpfs/scrubbed/zixianma/openwebrl-runtime/selection-head-20261008/pylib for scipy."""
import glob, json, math
from scipy.stats import binomtest

D = '/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021'
res, infra = {}, {}
for f in glob.glob(D + '/launch-preparation-v11[bc]-*/campaign/attempt-*/cells/*/episodes/*/*/result.json'):
    d = json.load(open(f)); site = f.split('launch-preparation-')[1].split('-')[1]
    k = (site, d['selector'], d['task_id']); e = str(d.get('error', ''))
    if not d.get('valid') and 'fuzzy' not in e.lower() and not e.startswith(('Invalid action', 'List requirements')):
        infra[k] = infra.get(k, 0) + 1
    if d.get('valid') or k not in res or (not res[k].get('valid') and 'fuzzy' in e.lower()):
        res[k] = d
rescore = {}
for r in json.load(open(D + '/fuzzy-rescore-gpt41-v11/rescore-results.json')):
    if r.get('rescorable'):
        site = r['package'].split('launch-preparation-')[1].split('-')[1]
        rescore[(site, r['selector'], r['task_id'])] = int(r.get('success') or 0)

def outcome(k, use_rescore, infra_fail):
    d = res.get(k)
    if d is None: return None
    if d.get('valid'): return int(bool(d.get('success')))
    if use_rescore and k in rescore: return rescore[k]
    if infra_fail and k[1] == 'piotr_selectionarm' and infra.get(k, 0) >= 2: return 0
    return None

def paired(use_rescore, infra_fail):
    a = b = n = ns = arm = 0
    for (site, sel, task) in res:
        if sel != 'no_search': continue
        x = outcome((site, 'no_search', task), use_rescore, infra_fail)
        y = outcome((site, 'piotr_selectionarm', task), use_rescore, infra_fail)
        if x is None or y is None: continue
        n += 1; ns += x; arm += y; a += y and not x; b += x and not y
    diff = (a - b) / n; se = math.sqrt(a + b - (a - b) ** 2 / n) / n
    p = binomtest(a, a + b).pvalue if a + b else 1.0
    return dict(paired=n, ns=ns, arm=arm, arm_only=a, ns_only=b, diff_pp=round(100 * diff, 1),
                ci_pp=(round(100 * (diff - 1.96 * se), 1), round(100 * (diff + 1.96 * se), 1)), p=round(p, 3))

for name, args in [('strict', (False, False)), ('+rescore', (True, False)), ('+rescore+infra_as_fail', (True, True))]:
    print(name, paired(*args))
