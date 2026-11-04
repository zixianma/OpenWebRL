"""Same-day (11-03) 3-run table: Luna/GPT-5.5 selectors vs same-day SFT and SelectionARM.
Overall = success / scheduled (invalid = failure); valid-only = success / valid.
Paired diffs: per-task success averaged over runs, task bootstrap (10,000, seed 20261103) within benchmark; Avg = equal-weight mean of 3 benchmarks."""
import glob, json, os, numpy as np
E = '/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations'
B = ('webvoyager', 'online_mind2web', 'deepshop'); N = {'webvoyager': 595, 'online_mind2web': 300, 'deepshop': 150}
C = {
 'SFT (1 sample)': ('controls-sameday-20261103', 'sft1'),
 'SelectionARM': ('controls-sameday-20261103', 'piotr5'),
 'Luna pre LoRA': ('p16-lora-lunapre-20261103', 'arm5-3b148a583f85'),
 'Luna K=3 LoRA': ('p16-lora-lunak3-20261103', 'arm5-2ed62dd7c4dc'),
 'Luna pre head': ('p16-heads-luna-20261103', 'head5-77ab61f94124'),
 'Luna K=3 head': ('p16-heads-luna-20261103', 'head5-8d58b41c8726'),
 'GPT-5.5 pre head': ('p16-heads-all-20261102', 'head5-cecccef33bba'),
 'GPT-5.5 K=3 head': ('p16-heads-all-20261102', 'head5-91e7a5a67417'),
 'GPT-5.5 pre LoRA (11-02)': ('p16-lora-pre-20261102', 'arm5-80ee47643be2'),
 'GPT-5.5 K=3 LoRA (11-01/02)': ('p16-lora-20261101', 'arm5-9fadc8dc4655'),
}
def load(root, slug, b):
    runs = []
    for r in ('run1', 'run2', 'run3'):
        d = {}
        for f in glob.glob(f'{E}/{root}/{r}/{b}/{slug}/{b}/collection/records/*.json'):
            x = json.load(open(f)); d[x['task_id']] = (bool(x['valid']), bool(x['valid']) and (x['reward'] or 0) >= 1)
        runs.append(d)
    return runs
data = {c: {b: load(*v, b) for b in B} for c, v in C.items()}
out = {'rows': {}, 'paired': {}}
for c in C:
    row = {}
    for b in B:
        rs = data[c][b]; s = sum(v[1] for d in rs for v in d.values()); va = sum(v[0] for d in rs for v in d.values())
        row[b] = dict(succ=s, valid=va, done=sum(len(d) for d in rs), sched=3 * N[b], overall=100 * s / (3 * N[b]), valid_only=100 * s / va if va else None)
    row['avg_overall'] = np.mean([row[b]['overall'] for b in B]); row['avg_valid'] = np.mean([row[b]['valid_only'] for b in B])
    out['rows'][c] = row
rng = np.random.default_rng(20261103)
def pertask(c, b):
    rs = data[c][b]; tasks = sorted(set().union(*[set(d) for d in rs]))
    return {t: np.mean([d[t][1] if t in d else 0 for d in rs]) for t in tasks}
PAIRS = [(c, ref) for ref in ('SFT (1 sample)', 'SelectionARM') for c in C if c != ref] + [
    ('Luna K=3 LoRA', 'Luna pre LoRA'), ('Luna K=3 head', 'Luna pre head'), ('GPT-5.5 K=3 head', 'GPT-5.5 pre head'),
    ('GPT-5.5 K=3 LoRA (11-01/02)', 'GPT-5.5 pre LoRA (11-02)'), ('Luna K=3 LoRA', 'Luna K=3 head'), ('GPT-5.5 K=3 head', 'Luna K=3 head')]
for c, ref in PAIRS:
    if True:
        per, boots = {}, []
        for b in B:
            a, r = pertask(c, b), pertask(ref, b); ts = sorted(set(a) | set(r))
            dd = 100 * np.array([a.get(t, 0) - r.get(t, 0) for t in ts])
            idx = rng.integers(0, len(dd), (10000, len(dd))); bs = dd[idx].mean(1)
            per[b] = [dd.mean(), *np.percentile(bs, [2.5, 97.5])]; boots.append(bs)
        avg = np.mean(boots, 0); per['avg'] = [np.mean([per[b][0] for b in B]), *np.percentile(avg, [2.5, 97.5])]
        out['paired'][f'{c} - {ref}'] = per
json.dump(out, open(os.path.join(os.path.dirname(__file__), 'table_1103.json'), 'w'), indent=1, default=float)
f = lambda x: f'{x:.1f}'
print('| Condition | WV ov/vo | OM2W ov/vo | DS ov/vo | Avg ov/vo | − SFT avg [CI] | − SelARM avg [CI] |')
for c, row in out['rows'].items():
    p = lambda ref: (lambda v: f'{v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]')(out['paired'][f'{c} - {ref}']['avg']) if f'{c} - {ref}' in out['paired'] else '—'
    print(f"| {c} | " + ' | '.join(f"{f(row[b]['overall'])} / {f(row[b]['valid_only'])}" for b in B) + f" | {f(row['avg_overall'])} / {f(row['avg_valid'])} | {p('SFT (1 sample)')} | {p('SelectionARM')} |", [row[b]['done'] for b in B])
