#!/usr/bin/env python3
"""Does SelectionARM avoid repeated actions during RL training? (CPU only, aggregate output.)

For every admitted training label (five candidate responses at one pre-action state), compare each
candidate's first tool call with the first tool calls already executed earlier in the same trajectory
(from the label's saved history). A candidate "repeats" if that call (name and arguments) was already
issued at an earlier step; wait and done are ignored, as in the evaluation loop rule.

Reports per lineage and training phase:
  p_exec_repeat / p_exec_new: P(ARM selects the executed response | executed action repeats / is new).
  mixed-state choice: among states whose five candidates contain both repeated and new actions, the share
    of ARM selections that are repeats, against the share of repeat candidates (the chance rate).
Limitation: history holds the turns visible to the actor (the prompt's history window), so very old
repeats can be missed; this only undercounts repeats.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import glob
import json
import os
from pathlib import Path
import re

EVALS = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations')
TOOL_CALL = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)
SKIP = ('wait', 'done')


def norm(call):
    if not isinstance(call, dict) or call.get('name') in SKIP or not call.get('name'):
        return None
    return json.dumps(dict(name=call['name'], arguments=call.get('arguments') or {}), sort_keys=True)


def first_history_call(text):
    m = TOOL_CALL.search(str(text))
    if not m:
        return None
    try:
        return norm(json.loads(m.group(1)))
    except json.JSONDecodeError:
        return None


def first_action(action):
    try:
        calls = json.loads(action) if isinstance(action, str) else action
    except json.JSONDecodeError:
        return None
    return norm(calls[0]) if isinstance(calls, list) and calls else None


def lineage(run_id, root):
    for key, lin in (('40m16f', 'g40m16f'), ('48m16f', 'g48m16f'), ('gate-b', 'gate_b'), ('gate-c', 'gate_c'),
                     ('mixed-bonus', 'mixed_bonus'), ('mixed-reweight', 'mixed_reweight'), ('action-only', 'action_only'),
                     ('failure-weight', 'beta1'), ('failure-coverage', 'coverage'), ('allfailure', 'allfailure'),
                     ('failure-additive', 'additive')):
        if key in run_id:
            return lin
    if 'allfailure' in root:
        return 'allfailure'
    return 'original' if Path(root).name.startswith('arm-turn-bonus-fresh-') else None


def scan_iteration(it_dir):
    it_dir = Path(it_dir)
    try:
        cfg = json.loads((it_dir/'arm-config.json').read_text())
    except Exception:
        return None
    lin = lineage(cfg.get('run_id') or str(cfg.get('policy_id', '')).split(':')[0], str(it_dir.parent.parent))
    if not lin:
        return None
    c = Counter()
    for f in it_dir.glob('labels/*.json'):
        try:
            recs = json.loads(f.read_text()).get('records') or []
        except Exception:
            continue
        for r in recs:
            if r.get('reason') != 'admitted' or not r.get('actions'):
                continue
            prior = {x for x in (first_history_call(h) for h in r.get('history') or []) if x}
            acts = [first_action(a) for a in r['actions']]
            rep = [a is not None and a in prior for a in acts]
            ex, sel = r.get('executed_index', 0), r.get('selected_index')
            if sel is None or not (0 <= ex < len(acts)) or not (0 <= sel < len(acts)):
                continue
            k = 'exec_repeat' if rep[ex] else 'exec_new'
            c[k+'_n'] += 1
            c[k+'_sel'] += sel == ex
            if any(rep) and not all(rep):
                c['mixed_states'] += 1
                c['mixed_sel_repeat'] += rep[sel]
                c['mixed_repeat_share'] += sum(rep)/len(rep)
            c['states'] += 1
            c['states_any_prior'] += bool(prior)
    return dict(lineage=lin, iteration=int(it_dir.name), root=it_dir.parent.parent.name, counts=dict(c))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    its = sorted(glob.glob(str(EVALS/'*/iterations/[0-9][0-9][0-9][0-9]')))
    its = [i for i in its if os.path.isdir(i+'/labels') and 'diagnostic' not in i and 'pilot' not in i]
    rows = []
    with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK', 1))) as ex:
        for r in ex.map(scan_iteration, its, chunksize=1):
            if r and r['counts'].get('states'):
                rows.append(r)
    # one root per (lineage, iteration): keep the one with most labels
    best = {}
    for r in rows:
        k = (r['lineage'], r['iteration'])
        if k not in best or r['counts']['states'] > best[k]['counts']['states']:
            best[k] = r
    agg = defaultdict(Counter)
    for (lin, it), r in best.items():
        phase = 'iterations 0-29' if it < 30 else 'iterations 30-59' if it < 60 else 'iterations 60-99'
        agg[(lin, phase)].update(r['counts']); agg[(lin, 'all')].update(r['counts'])
    out = {}
    for (lin, phase), c in sorted(agg.items()):
        o = dict(states=c['states'], exec_repeat_n=c['exec_repeat_n'], exec_new_n=c['exec_new_n'],
                 p_exec_repeat=c['exec_repeat_sel']/c['exec_repeat_n'] if c['exec_repeat_n'] else None,
                 p_exec_new=c['exec_new_sel']/c['exec_new_n'] if c['exec_new_n'] else None,
                 mixed_states=c['mixed_states'],
                 mixed_selected_repeat=c['mixed_sel_repeat']/c['mixed_states'] if c['mixed_states'] else None,
                 mixed_chance_repeat=c['mixed_repeat_share']/c['mixed_states'] if c['mixed_states'] else None,
                 counts=dict(c))
        out.setdefault(lin, {})[phase] = o
    a.output.write_text(json.dumps(dict(note=__doc__, iterations_scanned=len(best), lineages=out), indent=1))
    for lin, v in out.items():
        x = v['all']
        print(lin, x['states'], x['p_exec_repeat'], x['p_exec_new'], x['mixed_states'], x['mixed_selected_repeat'], x['mixed_chance_repeat'])


if __name__ == '__main__':
    main()
