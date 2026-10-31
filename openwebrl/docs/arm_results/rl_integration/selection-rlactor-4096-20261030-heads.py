#!/usr/bin/env python3
"""RL-checkpoint actor at 4,096 response tokens (10-30): SFT-trained 1,380 head on SFT-4B features (head5x) and RL-state head
(rl4b seed 0) on the actor's own states (head5); same-window 4,096-token baselines RL alone, RL + SelectionARM, SFT 4B alone. Rates (valid-only per benchmark, mean overall / valid-only), length-limit
terminations and how often the selector picked a length-truncated candidate. The 10-29 RL baselines used a 1,024 cap and
are shown for reference only (not the same protocol)."""
import json, glob, os, sys, collections, numpy as np
R = '/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations'; B = ('online_mind2web', 'webvoyager', 'deepshop')
RUNS = {'rl4096_alone': ('rlactor-4096-20261030/rl', 'sft1'), 'rl4096_arm': ('rlactor-4096-20261030/rl', 'piotr5'),
        'sft4096_alone': ('rlactor-4096-20261030/sft', 'sft1'), 'rl4096_hpiotrx': ('rlactor-4096-20261030/rl', 'head5x-755d4cd49137'), 'rl4096_h1380x': ('rlactor-4096-20261030/rl', 'head5x-11d5eff8ffe1'), 'rl4096_hrl': ('rlactor-4096-20261030/rl', 'head5-d954817f5c0e'),
        'rl1024_alone': ('rlactor-20261029', 'sft1'), 'rl1024_arm': ('rlactor-20261029', 'piotr5')}
out = {}
for name, (root, slug) in RUNS.items():
    o = {}
    for b in B:
        d = f'{R}/{root}/{b}/{slug}/{b}/collection'
        recs = [json.load(open(f)) for f in glob.glob(d + '/records/*.json')]
        valid = [r for r in recs if str(r['valid']) == 'True']; succ = [r for r in valid if float(r.get('reward') or 0) >= 1]
        lenlim = sum(r.get('terminate_reason') == 'generation_length_limit' for r in recs)
        picks = trunc_avail = trunc_picked = steps = 0
        for f in glob.glob(d + '/attempts/*/*/selections/*.jsonl'):
            for l in open(f):
                x = json.loads(l); ft = x.get('finish_types') or []; steps += 1
                if 'length' in ft:
                    trunc_avail += 1; trunc_picked += ft[int(x['selected_index'])] == 'length'
        o[b] = dict(n=len(recs), valid=len(valid), succ=len(succ), overall=round(100 * len(succ) / len(recs), 1),
                    valid_only=round(100 * len(succ) / max(len(valid), 1), 1), length_limit_episodes=lenlim,
                    selection_steps=steps, steps_with_truncated_candidate=trunc_avail, truncated_picked=trunc_picked)
    o['mean_overall'] = round(np.mean([o[b]['overall'] for b in B]), 1); o['mean_valid_only'] = round(np.mean([o[b]['valid_only'] for b in B]), 1)
    out[name] = o
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'heads.json'), 'w'), indent=1)
for k, v in out.items():
    print(k, [v[b]['valid_only'] for b in B], v['mean_overall'], v['mean_valid_only'],
          'lenlim', [v[b]['length_limit_episodes'] for b in B], 'trunc picked/avail', [(v[b]['truncated_picked'], v[b]['steps_with_truncated_candidate']) for b in B])
