"""RL 4B actor (OpenWebRL-4B RL @ 616cc8f, 4,096 tokens) with the Recipe 3 GPT-5.5 K=3 selectors: ADM-4B (4B LoRA, arm5) and
ADM-22M (22.6M head on the RL actor's own layer 18/27/36 features, head5; trained on SFT-actor features, no retrain), N=5, 3 runs
each, all 1,045 tasks, November 4-5 (sameday-rl4b-adm-20261104); RL 4B alone from the adjacent November 4 window
(sameday-actors-20261104) as the reference, plus SFT 4B + the same selectors and RL 8B for context. Overall = success / scheduled
(invalid = failure); valid-only = success / valid. Paired diffs: per-task success averaged over the three runs, 10,000 task
resamples within benchmark (seed 20261105); mean = equal-weight average of 3 benchmarks."""
import glob, json, os, numpy as np
E = '/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations'
B = ('online_mind2web', 'webvoyager', 'deepshop'); N = {'webvoyager': 595, 'online_mind2web': 300, 'deepshop': 150}
C = {'RL 4B (11-04)': ('sameday-actors-20261104/rl', 'sft1'),
     'RL 4B + ADM-4B (GPT-5.5)': ('sameday-rl4b-adm-20261104', 'arm5-9fadc8dc4655'),
     'RL 4B + ADM-22M (GPT-5.5)': ('sameday-rl4b-adm-20261104', 'head5-91e7a5a67417'),
     'SFT 4B + ADM-4B (GPT-5.5) (11-04)': ('sameday-lora-20261104', 'arm5-9fadc8dc4655'),
     'SFT 4B + ADM-22M (GPT-5.5) (11-02)': ('p16-heads-all-20261102', 'head5-91e7a5a67417'),
     'RL 8B (11-04/05)': ('sameday-rl8b-20261104/rl8b', 'sft1'),
     'RL 8B + ADM-4B (GPT-5.5) (11-04/05)': ('sameday-rl8b-20261104/rl8b-lora', 'arm5-9fadc8dc4655')}
def load(root, slug, b):
    out = []
    for r in ('run1', 'run2', 'run3'):
        d = {}
        for f in glob.glob(f'{E}/{root}/{r}/{b}/{slug}/{b}/collection/records/*.json'):
            x = json.load(open(f)); d[x['task_id']] = (bool(x['valid']), bool(x['valid']) and (x['reward'] or 0) >= 1)
        out.append(d)
    return out
D = {(c, b): load(*v, b) for c, v in C.items() for b in B}
rows = {}
for c in C:
    r = {}
    for b in B:
        s = sum(v[1] for d in D[c, b] for v in d.values()); va = sum(v[0] for d in D[c, b] for v in d.values())
        r[b] = dict(succ=s, valid=va, done=sum(len(d) for d in D[c, b]), overall=100 * s / (3 * N[b]), valid_only=100 * s / va)
    r['avg_overall'] = float(np.mean([r[b]['overall'] for b in B])); r['avg_valid'] = float(np.mean([r[b]['valid_only'] for b in B]))
    rows[c] = r
rng = np.random.default_rng(20261105)
def pt(c, b):
    ts = sorted(set().union(*[set(d) for d in D[c, b]])); return {t: np.mean([d[t][1] if t in d else 0 for d in D[c, b]]) for t in ts}
paired = {}
for a, ref in [('RL 4B + ADM-4B (GPT-5.5)', 'RL 4B (11-04)'), ('RL 4B + ADM-22M (GPT-5.5)', 'RL 4B (11-04)'),
               ('RL 4B + ADM-4B (GPT-5.5)', 'RL 4B + ADM-22M (GPT-5.5)'), ('RL 4B + ADM-4B (GPT-5.5)', 'SFT 4B + ADM-4B (GPT-5.5) (11-04)'),
               ('RL 4B + ADM-22M (GPT-5.5)', 'SFT 4B + ADM-22M (GPT-5.5) (11-02)'), ('RL 8B (11-04/05)', 'RL 4B + ADM-4B (GPT-5.5)'),
               ('RL 8B + ADM-4B (GPT-5.5) (11-04/05)', 'RL 4B + ADM-4B (GPT-5.5)')]:
    per, boots = {}, []
    for b in B:
        x, y = pt(a, b), pt(ref, b); ts = sorted(set(x) | set(y))
        dd = 100 * np.array([x.get(t, 0) - y.get(t, 0) for t in ts]); bs = dd[rng.integers(0, len(dd), (10000, len(dd)))].mean(1)
        per[b] = [float(dd.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]; boots.append(bs)
    m = np.mean(boots, 0); per['avg'] = [float(np.mean([per[b][0] for b in B])), *map(float, np.percentile(m, [2.5, 97.5]))]
    paired[f'{a} - {ref}'] = per
json.dump(dict(note=__doc__, rows=rows, paired=paired), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'table_rl4b_adm.json'), 'w'), indent=1)
for c, r in rows.items():
    print(f"{c:40s} " + '  '.join(f"{b[:4]} {r[b]['valid_only']:.1f}" for b in B) + f"  mean ov {r['avg_overall']:.1f} vo {r['avg_valid']:.1f}  done {[r[b]['done'] for b in B]}")
for k, v in paired.items(): print(k, ' | '.join(f"{b[:4]} {x[0]:+.1f} [{x[1]:+.1f}, {x[2]:+.1f}]" for b, x in v.items()))
