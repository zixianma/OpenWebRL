"""Recipe 3 selectors vs same-day SFT 4B and SelectionARM (SFT 4B actor, evaluated 2026-11-02/04): success and cost per
episode. GPT-5.5 LoRAs: November 4 same-day rerun (sameday-lora-20261104).
RL 8B actor (OpenWebRL/OpenWebRL-8B @ 824a54f, 4,096 tokens) alone and with the GPT-5.5 LoRAs: sameday-rl8b-20261104 (November 4–5). Success: 3 runs pooled per condition (overall = successes / scheduled, invalid = failure). Cost from run 1 (all
1,045 tasks), exactly as selection-n3-sft8b-full-20261021-analysis.py: actor PFLOP = 2 x params x (uncached prompt +
generated tokens); 4B LoRA selectors (SelectionARM and Recipe 3 LoRAs) 2 x 4.44e9 x their tokens; head FLOPs as logged.
Model time per episode = summed decision seconds / tasks. Benchmark means weight the three benchmarks equally."""
import json, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
E = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations')
HERE = Path(__file__).resolve().parent
BENCH = ('online_mind2web', 'webvoyager', 'deepshop')
P4B = 4022468096 + 415347712
P8B = 8767123696
CONDS = {  # key: (label, root, slug, kind)
    'sft1': ('SFT 4B, 1 sample', 'controls-sameday-20261103', 'sft1', 'none'),
    'sft8b': ('SFT 8B, 1 sample', 'sameday-actors-20261104/sft8b', 'sft1', 'none'),
    'rl4b': ('RL 4B, 1 sample (4,096 tokens)', 'sameday-actors-20261104/rl', 'sft1', 'none'),
    'rl8b': ('RL 8B, 1 sample (4,096 tokens)', 'sameday-rl8b-20261104/rl8b', 'sft1', 'none'),
    'rl8b-lora-gpt55-k3': ('RL 8B + 4B LoRA, Recipe 3 GPT-5.5 K=3', 'sameday-rl8b-20261104/rl8b-lora', 'arm5-9fadc8dc4655', 'lora'),
    'rl8b-lora-gpt55-pre': ('RL 8B + 4B LoRA, Recipe 3 panels GPT-5.5 pre', 'sameday-rl8b-20261104/rl8b-lora', 'arm5-80ee47643be2', 'lora'),
    'piotr5': ('Piotr SelectionARM', 'controls-sameday-20261103', 'piotr5', 'lora'),
    'lora-gpt55-k3': ('4B LoRA, Recipe 3 GPT-5.5 K=3', 'sameday-lora-20261104', 'arm5-9fadc8dc4655', 'lora'),
    'lora-luna-k3': ('4B LoRA, Recipe 3 Luna K=3', 'p16-lora-lunak3-20261103', 'arm5-2ed62dd7c4dc', 'lora'),
    'lora-gpt55-pre': ('4B LoRA, Recipe 3 panels GPT-5.5 pre', 'sameday-lora-20261104', 'arm5-80ee47643be2', 'lora'),
    'lora-luna-pre': ('4B LoRA, Recipe 3 panels Luna pre', 'p16-lora-lunapre-20261103', 'arm5-3b148a583f85', 'lora'),
    'head-gpt55-k3': ('Head, Recipe 3 GPT-5.5 K=3', 'p16-heads-all-20261102', 'head5-91e7a5a67417', 'head'),
    'head-luna-k3': ('Head, Recipe 3 Luna K=3', 'p16-heads-luna-20261103', 'head5-8d58b41c8726', 'head'),
    'head-gpt55-pre': ('Head, Recipe 3 panels GPT-5.5 pre', 'p16-heads-all-20261102', 'head5-cecccef33bba', 'head'),
    'head-luna-pre': ('Head, Recipe 3 panels Luna pre', 'p16-heads-luna-20261103', 'head5-77ab61f94124', 'head'),
}
def episode(p):
    r = json.loads(Path(p).read_text())
    out = dict(steps=int(r.get('browser_steps') or 0), sel_tokens=int(r.get('selector_input_tokens') or 0) + int(r.get('selector_output_tokens') or 0),
               prompt=0, cached=0, completion=0, head_flops=0.0, dec=0.0)
    for tf in (r.get('trace_files') or []):
        q = Path(tf['path'] if isinstance(tf, dict) else tf)
        if not q.exists(): continue
        for line in open(q):
            x = json.loads(line); out['dec'] += x['timing']['decision_seconds']
            out['head_flops'] += float((x.get('selector_telemetry') or {}).get('flops') or 0)
        for f in (q.parent.parent / 'actor-requests').glob('*.json'):
            m = json.loads(f.read_text()).get('meta') or {}
            out['prompt'] += m.get('prompt_tokens') or 0; out['cached'] += m.get('cached_tokens') or 0; out['completion'] += m.get('completion_tokens') or 0
    return out
def success(root, slug, b):
    s = n = 0
    for run in ('run1', 'run2', 'run3'):
        for f in (E / root / run / b / slug / b / 'collection' / 'records').glob('*.json'):
            r = json.loads(f.read_text()); n += 1; s += str(r.get('valid')) == 'True' and float(r.get('reward') or 0) >= 1
    return s, n
rows = []
with ThreadPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))) as pool:
    for k, (label, root, slug, kind) in CONDS.items():
        for b in BENCH:
            eps = list(pool.map(episode, sorted((E / root / 'run1' / b / slug / b / 'collection' / 'records').glob('*.json'))))
            n = len(eps); steps = sum(e['steps'] for e in eps)
            actor_pf = 2 * (P8B if k == 'sft8b' or k.startswith('rl8b') else P4B) * sum(e['prompt'] - e['cached'] + e['completion'] for e in eps) / 1e15
            sel_pf = (2 * P4B * sum(e['sel_tokens'] for e in eps) if kind == 'lora' else sum(e['head_flops'] for e in eps)) / 1e15
            s, ns = success(root, slug, b)
            rows.append(dict(key=k, condition=label, kind=kind, benchmark=b, cost_tasks=n, success_episodes=ns, successes=s,
                             overall=round(100 * s / ns, 2), steps_per_episode=round(steps / n, 2), pflop_per_step=round((actor_pf + sel_pf) / steps, 4),
                             pflop_per_episode=round((actor_pf + sel_pf) / n, 3), model_seconds_per_episode=round(sum(e['dec'] for e in eps) / n, 1)))
            print(rows[-1], flush=True)
json.dump(dict(note=__doc__, rows=rows), open(HERE / 'selection-recipe3-vs-cost-20261104.json', 'w'), indent=1)
