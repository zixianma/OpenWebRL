#!/usr/bin/env python3
"""Deep comparison of ARM RL lineages with outcome-only RL (CPU only, 2026-11-03).

Inputs (from the scan in analysis/scripts/deep_scan.py and analyze_rl_eval_failures.py --extract):
  --evals    evals.jsonl: per-cohort per-task outcomes of every saved evaluation.
  --train    train.jsonl: per-iteration ARM label statistics from saved training groups.
  --compact  directory of compact trajectory JSONL files (behavior categories).
Protocol for every eval comparison: local browser, GPT-4.1/action_history judge, actor T0, 30 turns,
full 300 Online-Mind2Web tasks. Overall counts invalid episodes as failures.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_rl_eval_failures import classify  # noqa: E402

RT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
TASKS = RT/'reference-arm-stealth90-o4-t06-20260929/online_mind2web_monitor.jsonl'
OUTCOME = ('outcome56', 'uniform8')
ARM = ('gate_b', 'mixed_bonus', 'mixed_reweight', 'g40m16f', 'g48m16f', 'gate_c', 'original', 'additive',
       'allfailure', 'action_only', 'beta1', 'coverage')
EXCLUDE = {('g48m16f', 40): 'actor-server stall; 127 aborted episodes; re-run pending'}
WINDOWS = {'early 10-30': (10, 20, 30), 'mid 40-60': (40, 50, 60), 'late 70-90': (70, 80, 90)}


def lineage_of(name):
    """Map an evaluation directory name to (lineage, iteration, repeat) or None."""
    m = re.search(r'iter(\d+)', name)
    it = int(m.group(1)) if m else None
    rules = [('repeat-iter90-gate-b', 'rep_gate_b'), ('repeat-iter90-outcome56', 'rep_outcome56'),
             ('repeat-iter90-outcome48', 'rep_outcome48'), ('outcome-only-56groups', 'outcome56'),
             ('outcome-uniform8', 'uniform8'), ('outcome-only-expanded', 'expanded'),
             ('gate-b-iter10-full300', 'gate_b'), ('arm-gate-b-40m16f', 'g40m16f'), ('arm-gate-b-48m16f', 'g48m16f'),
             ('arm-gate-b-iter', 'gate_b'), ('arm-gate-c-iter', 'gate_c'), ('arm-mixed-bonus-iter', 'mixed_bonus'),
             ('arm-mixed-reweight-iter', 'mixed_reweight'), ('arm-mixed-action-only-iter', 'action_only'),
             ('arm-original-iter', 'original'), ('arm-allfailure-iter', 'allfailure'),
             ('arm-failure-weight-iter', 'beta1'), ('arm-failure-coverage-iter', 'coverage'),
             ('arm-additive-iter', 'additive')]
    if name.startswith('arm-additive-iter10-318949'):
        return None  # partial 256-task cohort
    for key, lin in rules:
        if name.startswith(key):
            rep = re.search(r'-r(\d)-', name)
            return lin, it, int(rep.group(1)) if rep else 0
    if 'reference-318933' in name and name.endswith('after100'):
        return 'outcome48', 100, 0
    return None


def task_meta():
    out = {}
    for line in TASKS.read_text().splitlines():
        m = json.loads(line)['metadata']
        n = int(m['reference_length'])
        host = re.sub(r'^www\d?\.', '', re.sub(r'^https?://', '', m['start_url']).split('/')[0])
        meta = dict(band='easy' if n <= 5 else 'medium' if n <= 10 else 'hard', host=host)
        out[str(m['task_id'])] = meta
        out[str(m['task_id']).split('_')[0]] = meta
    return out


def boot_ci(diff, rng, draws=20000):
    idx = rng.integers(0, len(diff), size=(draws, len(diff)))
    means = diff[idx].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2*min((means <= 0).mean(), (means >= 0).mean())
    return float(lo), float(hi), float(min(1.0, p))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evals', type=Path, required=True)
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--compact', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    meta = task_meta()

    # ---- load local GPT-4.1 full-300 cohorts
    cohorts = defaultdict(dict)   # lineage -> iteration -> {task: (s, v, turns)}
    repeats = defaultdict(list)
    info = []
    for line in a.evals.read_text().splitlines():
        r = json.loads(line)
        name = r['dir'].rstrip('/').split('/')[-1] if '/evaluations/' in r['dir'] else r['dir']
        lab = lineage_of(name)
        if not lab or r['n'] != 300 or set(r['judge']) != {'gpt-4.1'}:
            continue
        lin, it, rep = lab
        if (lin, it) in EXCLUDE:
            continue
        rows = {t: (x['s'], x['v'], x['t'] or 0) for t, x in r['tasks'].items()}
        info.append(dict(lineage=lin, iteration=it, repeat=rep, dir=name, successes=r['succ'], valid=r['valid'],
                         date=r['mtime']))
        if lin.startswith('rep_'):
            repeats[lin[4:]].append(rows)
        else:
            if it in cohorts[lin]:
                raise ValueError(f'duplicate {lin} {it}: {name}')
            cohorts[lin][it] = rows
    tasks = sorted(next(iter(cohorts['outcome56'].values())))
    T = len(tasks)
    missing_meta = [t for t in tasks if t not in meta]

    def mat(lin, its, field=0):
        return np.array([[cohorts[lin][i][t][field] for t in tasks] for i in its], float)

    res = dict(protocol=__doc__.split('\n')[6:9], cohorts=sorted(info, key=lambda x: (x['lineage'], x['iteration'], x['repeat'])),
               excluded=[dict(lineage=k[0], iteration=k[1], reason=v) for k, v in EXCLUDE.items()],
               tasks=T, tasks_without_metadata=len(missing_meta))

    # ---- 1. window means and paired differences vs each outcome-only run and their mean
    win = {}
    for lin in OUTCOME+ARM+('expanded',):
        for w, its in WINDOWS.items():
            if not all(i in cohorts.get(lin, {}) for i in its):
                continue
            s, v = mat(lin, its, 0), mat(lin, its, 1)
            row = dict(overall=float(s.mean()), valid_only=float(s.sum()/v.sum()), invalid_per_300=float(300-v.sum(1).mean()))
            per_task = s.mean(0)
            for ref in OUTCOME:
                if ref == lin or not all(i in cohorts[ref] for i in its):
                    continue
                d = per_task-mat(ref, its, 0).mean(0)
                lo, hi, p = boot_ci(d, rng)
                row[f'minus_{ref}'] = dict(delta_pp=100*float(d.mean()), ci95_pp=[100*lo, 100*hi], p=p)
                # common-valid: only tasks valid in both runs at the same checkpoint
                vv = mat(lin, its, 1)*mat(ref, its, 1)
                cs, cr = (s*vv).sum(), (mat(ref, its, 0)*vv).sum()
                row[f'common_valid_vs_{ref}'] = dict(pairs=int(vv.sum()), this=int(cs), ref=int(cr),
                                                     delta_pp=100*float((cs-cr)/max(1, vv.sum())))
            if lin not in OUTCOME and all(i in cohorts[r] for r in OUTCOME for i in its):
                ref = np.mean([mat(r, its, 0).mean(0) for r in OUTCOME], 0)
                lo, hi, p = boot_ci(per_task-ref, rng)
                row['minus_outcome_mean'] = dict(delta_pp=100*float((per_task-ref).mean()), ci95_pp=[100*lo, 100*hi], p=p)
            win.setdefault(lin, {})[w] = row
    res['windows'] = win

    # ---- 2. curve area: mean over every shared checkpoint 10..90 with outcome56
    auc = {}
    for lin in OUTCOME+ARM+('expanded',):
        its = sorted(i for i in cohorts.get(lin, {}) if i in cohorts['outcome56'])
        if len(its) < 3:
            continue
        s = mat(lin, its, 0)
        d = s.mean(0)-mat('outcome56', its, 0).mean(0)
        lo, hi, p = boot_ci(d, rng)
        auc[lin] = dict(iterations=its, overall=float(s.mean()), valid_only=float(s.sum()/mat(lin, its, 1).sum()),
                        minus_outcome56_pp=100*float(d.mean()), ci95_pp=[100*lo, 100*hi], p=p)
    res['curve_mean'] = auc

    # ---- 3. iteration-90 repeats (same harness, same dates): per-task consistency
    rep = {}
    for k, runs in repeats.items():
        s = np.array([[r[t][0] for t in tasks] for r in runs], float)
        v = np.array([[r[t][1] for t in tasks] for r in runs], float)
        rep[k] = dict(repeats=len(runs), successes=s.sum(1).astype(int).tolist(), valid=v.sum(1).astype(int).tolist(),
                      always=int((s.min(0) == 1).sum()), never=int((s.max(0) == 0).sum()),
                      sometimes=int(((s.max(0) == 1) & (s.min(0) == 0)).sum()))
        rep[k]['per_task'] = s.mean(0)
    for x, y in (('gate_b', 'outcome56'), ('gate_b', 'outcome48'), ('outcome56', 'outcome48')):
        if x in rep and y in rep:
            d = rep[x]['per_task']-rep[y]['per_task']
            lo, hi, p = boot_ci(d, rng)
            band = {b: 100*float(d[[meta.get(t, {}).get('band') == b for t in tasks]].mean()) for b in ('easy', 'medium', 'hard')}
            rep[f'{x}-{y}'] = dict(delta_pp=100*float(d.mean()), ci95_pp=[100*lo, 100*hi], p=p, by_band_pp=band,
                                   tasks_gate_better=int((d > 0).sum()), tasks_ref_better=int((d < 0).sum()))
    for k in list(rep):
        rep[k].pop('per_task', None) if isinstance(rep[k], dict) else None
    res['iter90_repeats'] = rep

    # ---- 4. difficulty bands per window
    bands = {}
    for lin, ws in win.items():
        for w in ws:
            s = mat(lin, WINDOWS[w], 0).mean(0)
            bands.setdefault(lin, {})[w] = {b: 100*float(s[[meta.get(t, {}).get('band') == b for t in tasks]].mean())
                                           for b in ('easy', 'medium', 'hard')}
    res['bands'] = bands

    # ---- 5. task-level structure: ARM family vs outcome family, mid+late windows (40..90)
    its = (40, 50, 60, 70, 80, 90)
    fam_arm = [l for l in ('gate_b', 'mixed_bonus', 'mixed_reweight', 'g40m16f') if all(i in cohorts[l] for i in its)]
    fam_out = [l for l in OUTCOME if all(i in cohorts[l] for i in its)]
    A = np.array([mat(l, its, 0).mean(0) for l in fam_arm]); O = np.array([mat(l, its, 0).mean(0) for l in fam_out])
    d = A.mean(0)-O.mean(0)
    # null: differences between the two outcome-only runs show how much per-task gap is noise
    null = np.abs(O[0]-O[1]) if len(O) == 2 else None
    hosts = Counter(meta.get(t, {}).get('host') for t in tasks)
    host_d = defaultdict(list)
    for t, x in zip(tasks, d):
        host_d[meta.get(t, {}).get('host')].append(x)
    res['task_structure'] = dict(
        arm_family=fam_arm, outcome_family=fam_out, iterations=list(its),
        tasks_arm_ge_25pp=int((d >= .25).sum()), tasks_outcome_ge_25pp=int((d <= -.25).sum()),
        outcome_pair_abs_gap_ge_25pp=int((null >= .25).sum()) if null is not None else None,
        mean_delta_pp=100*float(d.mean()),
        by_band_pp={b: 100*float(d[[meta.get(t, {}).get('band') == b for t in tasks]].mean()) for b in ('easy', 'medium', 'hard')},
        hosts_with_ge3_tasks_top=sorted(((h, len(v), round(100*float(np.mean(v)), 1)) for h, v in host_d.items() if len(v) >= 3),
                                        key=lambda x: -x[2])[:12],
        hosts_with_ge3_tasks_bottom=sorted(((h, len(v), round(100*float(np.mean(v)), 1)) for h, v in host_d.items() if len(v) >= 3),
                                           key=lambda x: x[2])[:12],
        distinct_hosts=len(hosts))

    # ---- 6. invalid decomposition: start page (<=1 turn) vs mid-episode
    inv = {}
    for lin, ws in win.items():
        for w in ws:
            its_w = WINDOWS[w]
            v, t = mat(lin, its_w, 1), mat(lin, its_w, 2)
            inv.setdefault(lin, {})[w] = dict(start_page=float(((v == 0) & (t <= 1)).sum(1).mean()),
                                              mid_episode=float(((v == 0) & (t > 1)).sum(1).mean()))
    res['invalid'] = inv

    # ---- 7. behavior categories from compact trajectories (where step-level data exist)
    beh = {}
    for f in sorted(a.compact.glob('*.jsonl')):
        lab = lineage_of(f.stem) or (('outcome48', 100, 0) if f.stem == 'outcome48-iter100' else None)
        if not lab:
            continue
        rows = [json.loads(x) for x in f.read_text().splitlines()]
        if not any(r.get('steps') for r in rows):
            continue
        cats = Counter(classify(r, 30) for r in rows)
        coarse = Counter()
        for c, n in cats.items():
            coarse['invalid' if c.startswith('invalid') else c] += n
        vf = [r for r in rows if r['valid'] and not r['success']]
        beh.setdefault(lab[0], {})[f'{lab[1]}' + (f'r{lab[2]}' if lab[2] else '')] = dict(
            categories=dict(coarse), mean_steps_valid_failure=float(np.mean([r.get('total_steps') or 0 for r in vf])) if vf else None,
            mean_steps_success=float(np.mean([r.get('total_steps') or 0 for r in rows if r['success']])) if any(r['success'] for r in rows) else None)
    res['behavior'] = beh

    # ---- 8. training: does the ARM prefer executed actions more in successful trajectories?
    runid = {}
    for line in a.train.read_text().splitlines():
        r = json.loads(line)
        cfg = Path(r['root'])/'iterations'/f"{r['iteration']:04d}"/'arm-config.json'
        try:
            c = json.loads(cfg.read_text())
            rid = c.get('run_id') or str(c.get('policy_id', '')).split(':')[0]
            r['credit'] = c.get('credit_assignment', 'response_index'); r['gate'] = c.get('candidate_gate', 'strict')
        except Exception:
            rid = ''
        root = Path(r['root']).name
        lin = ('g40m16f' if '40m16f' in rid else 'g48m16f' if '48m16f' in rid else 'gate_b' if 'gate-b' in rid else
               'gate_c' if 'gate-c' in rid else 'beta1' if 'failure-weight' in rid else 'coverage' if 'failure-coverage' in rid else
               'mixed_bonus' if 'mixed-bonus' in rid else 'mixed_reweight' if 'mixed-reweight' in rid else
               'action_only' if 'action-only' in rid else 'allfailure' if 'allfailure' in root or 'allfailure' in rid else
               'additive' if 'failure-additive' in rid else 'original' if root.startswith('arm-turn-bonus-fresh-') else None)
        if 'diagnostic' in root or 'pilot' in root or lin is None:
            continue
        key = (lin, r['iteration'])
        if key not in runid or r['stats'].get('admitted', 0) > runid[key]['stats'].get('admitted', 0):
            runid[key] = r
    tr = {}
    for (lin, it), r in runid.items():
        w = 'early 0-29' if it < 30 else 'mid 30-59' if it < 60 else 'late 60-99'
        x = tr.setdefault(lin, {}).setdefault(w, dict(iterations=0, sel=Counter(), n=Counter(), groups=Counter(), aux_sel=0, aux_n=0))
        x['iterations'] += 1
        for k, (s, n) in r['selected'].items():
            x['sel'][k] += s; x['n'][k] += n
        for k in ('groups_M', 'groups_F', 'groups_S', 'accepted'):
            x['groups'][k] += r['stats'].get(k, 0)
        x['aux_sel'] += r['aux'].get('selected', 0); x['aux_n'] += r['aux'].get('admitted', 0)
    train = {}
    for lin, ws in tr.items():
        for w, x in ws.items():
            out = dict(iterations=x['iterations'], labels=dict(x['n']),
                       p_select_executed={k: x['sel'][k]/x['n'][k] for k in x['n'] if x['n'][k]},
                       groups_per_iteration={k: v/x['iterations'] for k, v in x['groups'].items()},
                       failure_aux_p_select=(x['aux_sel']/x['aux_n']) if x['aux_n'] else None, failure_aux_labels=x['aux_n'])
            ms, mf = x['n']['M:succ'], x['n']['M:fail']
            if ms and mf:
                ps, pf = x['sel']['M:succ']/ms, x['sel']['M:fail']/mf
                se = (ps*(1-ps)/ms+pf*(1-pf)/mf)**.5
                out['mixed_succ_minus_fail_pp'] = dict(delta=100*(ps-pf), se_turn_level=100*se)
            train.setdefault(lin, {})[w] = out
    res['training_labels'] = train
    a.output.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float)[:200])


if __name__ == '__main__':
    main()
