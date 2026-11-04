"""Same-day 2026-11-04 rerun: GPT-5.5 K=3 LoRA, GPT-5.5 pre LoRA and SelectionARM, plus SFT 8B alone and the RL 4B checkpoint
alone (616cc8f, 4,096 response tokens), 3 runs each, all 1,045 tasks.
Also against the 11-03/04 night SFT 4B run (controls-sameday-20261103; adjacent window, not same-day).
Overall = success / scheduled (invalid = failure); valid-only = success / valid. Paired diffs: per-task success averaged
over the three runs, 10,000 task resamples within benchmark (seed 20261104); mean = equal-weight average of 3 benchmarks;
weighted = task-weighted (pooled over all 1,045 tasks)."""
import glob, json, os, numpy as np
E = '/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations'
B = ('online_mind2web', 'webvoyager', 'deepshop'); N = {'webvoyager': 595, 'online_mind2web': 300, 'deepshop': 150}
C = {'GPT-5.5 K=3 LoRA': ('sameday-lora-20261104', 'arm5-9fadc8dc4655'), 'GPT-5.5 pre LoRA': ('sameday-lora-20261104', 'arm5-80ee47643be2'),
     'SelectionARM': ('sameday-lora-20261104', 'piotr5'), 'SFT 4B (11-03/04)': ('controls-sameday-20261103', 'sft1'),
     'SelectionARM (11-03/04)': ('controls-sameday-20261103', 'piotr5'),
     'SFT 8B': ('sameday-actors-20261104/sft8b', 'sft1'), 'RL 4B (4,096 tokens)': ('sameday-actors-20261104/rl', 'sft1')}
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
    r['w_overall'] = 100 * sum(r[b]['succ'] for b in B) / (3 * sum(N.values())); r['w_valid'] = 100 * sum(r[b]['succ'] for b in B) / sum(r[b]['valid'] for b in B)
    rows[c] = r
rng = np.random.default_rng(20261104)
def pt(c, b):
    ts = sorted(set().union(*[set(d) for d in D[c, b]])); return {t: np.mean([d[t][1] if t in d else 0 for d in D[c, b]]) for t in ts}
paired = {}
for a, ref in [('GPT-5.5 pre LoRA', 'GPT-5.5 K=3 LoRA'), ('GPT-5.5 K=3 LoRA', 'SelectionARM'), ('GPT-5.5 pre LoRA', 'SelectionARM'),
               ('GPT-5.5 K=3 LoRA', 'SFT 4B (11-03/04)'), ('GPT-5.5 pre LoRA', 'SFT 4B (11-03/04)'), ('SelectionARM', 'SFT 4B (11-03/04)'),
               ('SelectionARM', 'SelectionARM (11-03/04)'), ('SFT 8B', 'SFT 4B (11-03/04)'), ('RL 4B (4,096 tokens)', 'SFT 4B (11-03/04)'),
               ('SelectionARM', 'SFT 8B'), ('GPT-5.5 pre LoRA', 'SFT 8B'), ('GPT-5.5 K=3 LoRA', 'SFT 8B'), ('SelectionARM', 'RL 4B (4,096 tokens)'), ('GPT-5.5 pre LoRA', 'RL 4B (4,096 tokens)')]:
    per, boots = {}, []
    for b in B:
        x, y = pt(a, b), pt(ref, b); ts = sorted(set(x) | set(y))
        dd = 100 * np.array([x.get(t, 0) - y.get(t, 0) for t in ts]); bs = dd[rng.integers(0, len(dd), (10000, len(dd)))].mean(1)
        per[b] = [float(dd.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]; boots.append(bs)
    m = np.mean(boots, 0); per['avg'] = [float(np.mean([per[b][0] for b in B])), *map(float, np.percentile(m, [2.5, 97.5]))]
    W = np.array([N[b] for b in B]) / sum(N.values()); wb = np.tensordot(W, np.array(boots), 1)
    per['weighted'] = [float(sum(W[i] * per[b][0] for i, b in enumerate(B))), *map(float, np.percentile(wb, [2.5, 97.5]))]
    paired[f'{a} - {ref}'] = per
json.dump(dict(note=__doc__, rows=rows, paired=paired), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'table_1104.json'), 'w'), indent=1)
for c, r in rows.items():
    print(f"{c:24s} " + '  '.join(f"{b[:4]} {r[b]['valid_only']:.1f}" for b in B) + f"  mean ov {r['avg_overall']:.1f} vo {r['avg_valid']:.1f}  done {[r[b]['done'] for b in B]}")
for k, v in paired.items(): print(k, ' | '.join(f"{b[:4]} {x[0]:+.1f} [{x[1]:+.1f}, {x[2]:+.1f}]" for b, x in v.items()))
