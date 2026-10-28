"""Final OM2W rerun300 table (actor gpt-4o-2024-08-06, judge gpt-5.6-terra): all packages pooled, one primary attempt per cell."""
import collections, glob, json, sys
from pathlib import Path
from scipy.stats import binomtest

O = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-om2w-table5-20261012')
M = ['no_search', 'random5', 'arbiter7b', 'piotr_selectionarm', 'shepherd8b']
PACKAGES = ['rerun300-j1-vlm', 'rerun300-j2-vlm', 'rerun300-j3-vlm', 'rerun300-j3-completion-vlm',
            'rerun300-j1-completion2-vlm', 'rerun300-j2-completion2-vlm', 'rerun300-j3-completion2-vlm',
            'rerun300-j1-completion3-vlm', 'rerun300-j3-completion3-vlm']
RETRY_PACKAGES = ['rerun300-j1-retry1-vlm', 'rerun300-j2-retry1-vlm', 'rerun300-j3-retry1-vlm',  # labelled retries of invalid cells, in order
                  'rerun300-j1-retry2-vlm', 'rerun300-j2-retry2-vlm', 'rerun300-j3-retry2-vlm']

cells = {}
for package in PACKAGES:
    for f in glob.glob(str(O/package/'campaign/queue/outcomes/*.json')):
        o = json.load(open(f)); r = o['result']; cid = o.get('cell_id') or r['cell_id']
        assert cid not in cells, ('duplicate primary cell', cid)
        cells[cid] = dict(package=package, task=cid.split('/')[0], method=cid.split('__')[-1], valid=bool(r.get('valid')),
                          success=int(bool(r.get('valid')) and bool(r.get('success'))), error=r.get('error_type') or r.get('reason'))
primary = dict(cells)
retried = {}
for package in RETRY_PACKAGES:
    for f in glob.glob(str(O/package/'campaign/queue/outcomes/*.json')):
        o = json.load(open(f)); r = o['result']; cid = o.get('cell_id') or r['cell_id']
        prior = retried.get(cid) or primary.get(cid)
        assert prior is None or not prior['valid'], ('retry of a cell that already has a valid attempt', cid)
        retried[cid] = dict(package=package, rounds=(retried[cid]['rounds']+1 if cid in retried else 1), task=cid.split('/')[0], method=cid.split('__')[-1], valid=bool(r.get('valid')), retried=True,
                            success=int(bool(r.get('valid')) and bool(r.get('success'))), error=r.get('error_type') or r.get('reason'))
cells.update(retried)


def index(cs):
    by = collections.defaultdict(dict)
    for c in cs.values():
        by[c['task']][c['method']] = c
    return by


by = index(cells); by_primary = index(primary)
tasks = sorted(by)
rows = []
for m in M:
    cs = [c for c in cells.values() if c['method'] == m]
    v = [c for c in cs if c['valid']]
    rows.append(dict(method=m, cells=len(cs), retried=sum(1 for c in cs if c.get('retried')), retried_now_valid=sum(1 for c in cs if c.get('retried') and c['valid']), missing=300-len(cs), valid=len(v), invalid=len(cs)-len(v), successes=sum(c['success'] for c in v),
                     valid_rate=round(100*sum(c['success'] for c in v)/max(len(v), 1), 1),
                     invalid_reasons=dict(collections.Counter(c['error'] for c in cs if not c['valid']))))
paired = [t for t in tasks if all(m in by[t] and by[t][m]['valid'] for m in M)]
comparisons = []
for m in M[1:]:
    a = sum(1 for t in paired if by[t][m]['success'] and not by[t]['no_search']['success'])
    b = sum(1 for t in paired if by[t]['no_search']['success'] and not by[t][m]['success'])
    comparisons.append(dict(method=m, paired_tasks=len(paired), successes=sum(by[t][m]['success'] for t in paired),
                            no_search_successes=sum(by[t]['no_search']['success'] for t in paired), method_only=a, no_search_only=b,
                            sign_test_p=round(binomtest(a, a+b).pvalue, 4) if a+b else 1.0))
paired_primary = [t for t in sorted(by_primary) if all(m in by_primary[t] and by_primary[t][m]['valid'] for m in M)]
out = dict(cohort='om2w_rerun300_0806_terra', tasks_touched=len(tasks), cells=len(cells), retried_cells=len(retried),
           paired_tasks_before_retries=len(paired_primary), paired_tasks_with_retries=len(paired), per_method=rows, paired_vs_no_search=comparisons)
(O/'rerun300-0806-20261023/final-table.json').write_text(json.dumps(out, indent=1)+'\n')
print(json.dumps(out, indent=1))
