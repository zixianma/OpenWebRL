"""RL 8B actor (OpenWebRL/OpenWebRL-8B @ 824a54f, 4,096 tokens) alone and with the GPT-5.5 K=3 / pre LoRA selectors (Recipe 3,
N=5), 3 runs each, all 1,045 tasks, November 4-5 (sameday-rl8b-20261104); with the November 4 RL 4B / SelectionARM / GPT-5.5 LoRA
rows (sameday-actors / sameday-lora-20261104; adjacent window) for reference. Overall = success / scheduled (invalid = failure);
valid-only = success / valid. Paired diffs: per-task success averaged over the three runs, 10,000 task resamples within
benchmark (seed 20261105); mean = equal-weight average of 3 benchmarks."""
import glob, json, os, numpy as np
E = '/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations'
B = ('online_mind2web', 'webvoyager', 'deepshop'); N = {'webvoyager': 595, 'online_mind2web': 300, 'deepshop': 150}
C = {'RL 8B': ('sameday-rl8b-20261104/rl8b', 'sft1'),
     'RL 8B + GPT-5.5 K=3 LoRA': ('sameday-rl8b-20261104/rl8b-lora', 'arm5-9fadc8dc4655'),
     'RL 8B + GPT-5.5 pre LoRA': ('sameday-rl8b-20261104/rl8b-lora', 'arm5-80ee47643be2'),
     'RL 4B (11-04)': ('sameday-actors-20261104/rl', 'sft1'), 'SFT 8B (11-04)': ('sameday-actors-20261104/sft8b', 'sft1'),
     'SFT 4B + GPT-5.5 K=3 LoRA (11-04)': ('sameday-lora-20261104', 'arm5-9fadc8dc4655'),
     'SFT 4B + GPT-5.5 pre LoRA (11-04)': ('sameday-lora-20261104', 'arm5-80ee47643be2'),
     'SFT 4B + SelectionARM (11-04)': ('sameday-lora-20261104', 'piotr5')}
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
for a, ref in [('RL 8B + GPT-5.5 K=3 LoRA', 'RL 8B'), ('RL 8B + GPT-5.5 pre LoRA', 'RL 8B'), ('RL 8B + GPT-5.5 pre LoRA', 'RL 8B + GPT-5.5 K=3 LoRA'),
               ('RL 8B', 'RL 4B (11-04)'), ('RL 8B', 'SFT 8B (11-04)'), ('RL 8B + GPT-5.5 K=3 LoRA', 'SFT 4B + GPT-5.5 K=3 LoRA (11-04)'),
               ('RL 8B + GPT-5.5 pre LoRA', 'SFT 4B + GPT-5.5 pre LoRA (11-04)'), ('RL 8B', 'SFT 4B + SelectionARM (11-04)')]:
    per, boots = {}, []
    for b in B:
        x, y = pt(a, b), pt(ref, b); ts = sorted(set(x) | set(y))
        dd = 100 * np.array([x.get(t, 0) - y.get(t, 0) for t in ts]); bs = dd[rng.integers(0, len(dd), (10000, len(dd)))].mean(1)
        per[b] = [float(dd.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]; boots.append(bs)
    m = np.mean(boots, 0); per['avg'] = [float(np.mean([per[b][0] for b in B])), *map(float, np.percentile(m, [2.5, 97.5]))]
    paired[f'{a} - {ref}'] = per
json.dump(dict(note=__doc__, rows=rows, paired=paired), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(__file__)[:-3] + '.json'), 'w'), indent=1)
for c, r in rows.items():
    print(f"{c:36s} " + '  '.join(f"{b[:4]} {r[b]['valid_only']:.1f}" for b in B) + f"  mean ov {r['avg_overall']:.1f} vo {r['avg_valid']:.1f}  done {[r[b]['done'] for b in B]}")
for k, v in paired.items(): print(k, ' | '.join(f"{b[:4]} {x[0]:+.1f} [{x[1]:+.1f}, {x[2]:+.1f}]" for b, x in v.items()))
