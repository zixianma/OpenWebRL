#!/usr/bin/env python3
"""Reward/success diagnostic from durable JSON summaries; never load trajectories/tensors.

Public outputs contain numeric aggregates only. Source paths and input hashes stay
in a private runtime receipt. --render-only reproduces figures from public JSON.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from plot_arm_interactive import build_data

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
OUT = REPO / 'openwebrl/docs/rl_results'
PRIVATE = RUNTIME / 'arm-turn-bonus-preparation/reward-hacking-curves-20261003'
RUNS = {
    'original': 'arm-turn-bonus-fresh-294197',
    'all_failure': 'arm-allfailure-bonus-295353',
    'additive': 'arm-failure-additive-295786',
    'gate_b': 'arm-gate-b-309053',
    'gate_c': 'arm-gate-c-309054',
    'failure_beta1': 'arm-failure-weight-fromzero-318949',
    'failure_coverage': 'arm-failure-coverage-fromzero-318950',
    'mixed_bonus': 'arm-mixed-bonus-334493',
    'mixed_reweight': 'arm-mixed-reweight-334494',
}


def read(path):
    return json.loads(path.read_text())


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError(f'Numeric inconsistency: {a} != {b}')


def panel(c, beta):
    """Mean includes all trainable turns; credited selection rate conditions on labels."""
    n, labels = c['rows'], c['admitted']
    assert isinstance(n, int) and isinstance(labels, int) and 0 <= labels <= n and n > 0
    v = c['variants'][str(float(beta))]
    win = c.get('positive_bonus_rate', c['selected_original_rate']) if labels else None
    chance = c.get('mean_chance_baseline', .2) if labels else None
    close(c['effective_scored_fraction'], labels / n)
    close(v['bonus_mean'], beta * labels / n * (win - chance) if labels else 0)
    return dict(turns=n, labels=labels, coverage=labels/n, credited_rate=win,
                chance=chance, mean=v['bonus_mean'], rms=v['bonus_rms'], beta=beta,
                rms_to_outcome=v['bonus_to_outcome_rms'])


def auxiliary(a):
    n, records, beta = a['total_failure_rows'], a['records'], a['beta']
    assert 0 <= len(records) <= n
    values, chances, wins = [], [], []
    for r in records:
        label = r.get('arm_turn_bonus')
        # Older manifests saved the actual advantage but no label metadata.
        # Binary centered credit uniquely recovers chance from advantage / beta.
        unit = r['advantage'] / beta
        win = int(unit > 0)
        chance = win - unit
        assert 0 < chance < 1
        if label:
            assert label['eligible'] and label['policy_id'] == a['policy_id']
            close(unit, label['unit_bonus'])
            close(chance, label.get('chance_baseline', .2))
        close(unit, win - chance)
        close(r['advantage'], beta * unit)
        values.append(r['advantage']); chances.append(chance); wins.append(win)
    # Empty auxiliary buffers are absent observations, never fabricated zero means.
    return dict(turns=n, labels=len(records), coverage=len(records)/n if n else None,
                credited_rate=sum(wins)/len(wins) if wins else None,
                chance=sum(chances)/len(chances) if chances else None,
                mean=sum(values)/n if n else None,
                rms=math.sqrt(sum(v*v for v in values)/n) if n else None,
                rms_to_outcome=None, beta=beta, groups=a['failure_groups'],
                loss_coefficient=a['coefficient'])


def align(c, saved, validation, run_id, rid):
    assert c['policy_id'] == f'{run_id}:rollout{rid}', c['policy_id']
    assert saved['rollout_id'] == rid
    assert int(Path(saved['checkpoint']).name.removeprefix('iter_')) == rid
    if validation:
        assert validation['iteration'] == rid
    actor = Path(c['checkpoint']).name
    if actor.startswith('iter_'):
        assert int(actor.removeprefix('iter_')) + 1 == rid, (rid, actor)
    else:
        assert rid == 0 and actor == 'OpenWebRL-4B-SFT', (rid, actor)
    return rid  # Data were collected BEFORE the PPO update that saved native iter_rid.


def weighted(rows, field, weight):
    available = [r for r in rows if r.get(field) is not None and r.get(weight, 0) > 0]
    total = sum(r[weight] for r in available)
    return sum(r[field]*r[weight] for r in available)/total if total else None


def smooth(rows, field, weight, window):
    """Trailing iteration window; weight by actual denominator, never fill missing data."""
    return [weighted([p for p in rows if r['iteration']-window < p['iteration'] <= r['iteration']],
                     field, weight) if r.get(field) is not None else None for r in rows]


def aggregate():
    evaluations = build_data()
    series = {s['id']: dict(id=s['id'], label=s['label'], evaluations=s['points'], rows=[])
              for s in evaluations['series'] if s['id'] in RUNS}
    baseline = next(s['points'] for s in evaluations['series'] if s['id'] == 'outcome_only')
    reverse = {v: k for k, v in RUNS.items()}
    receipts, omissions, seen = [], [], set()
    for mp in sorted((RUNTIME/'evaluations').glob('arm-*/launch_manifest.json')):
        manifest = read(mp)
        run_id = manifest.get('arm_config', {}).get('run_id')
        if run_id not in reverse:
            continue
        key = reverse[run_id]
        for marker in sorted((mp.parent/'iterations').glob('*/checkpoint-saved.json')):
            folder = marker.parent; rid = int(folder.name)
            cp = folder/'calibration.json'; vp = folder/'checkpoint-validation.json'
            if not cp.exists():
                omissions.append(dict(method=key, iteration=rid, reason='missing calibration', path=str(cp)))
                continue
            assert (key, rid) not in seen, ('Duplicate durable iteration', key, rid)
            seen.add((key, rid))
            c, saved = read(cp), read(marker)
            validation = read(vp) if vp.exists() else None
            x = align(c, saved, validation, run_id, rid)
            beta = c['applied_beta']
            row = dict(iteration=x, saved_iteration=rid+1,
                       optimizer_updates_after=validation.get('completed_optimizer_updates') if validation else None,
                       q=c['config']['scored_fraction'], **panel(c, beta))
            inputs = [mp, marker, cp] + ([vp] if validation else [])
            scopes = dict(main=row)
            if (folder/'failure_auxiliary.json').exists():
                ap = folder/'failure_auxiliary.json'; a = read(ap)
                assert a['policy_id'] == c['policy_id'] and a['rollout_id'] == rid
                extra = auxiliary(a)
                assert extra['turns'] == c['additive_failure_rows']
                assert extra['labels'] == c['additive_failure_labels']
                scopes['failure'] = dict(iteration=x, **extra)
                inputs.append(ap)
            elif key == 'all_failure':
                n, labels = c['all_failure_rows'], c['all_failure_usable_labels']
                pos = c['all_failure_positive_turns']
                scopes['failure'] = dict(iteration=x, turns=n, labels=labels,
                    coverage=labels/n if n else None, credited_rate=pos/labels if labels else None,
                    chance=.2 if labels else None, mean=beta*(pos-.2*labels)/n if n else None,
                    rms=c['all_failure_bonus_rms'] if n else None, rms_to_outcome=None,
                    beta=beta, groups=c['all_failure_groups'], loss_coefficient=None)
                scopes['mixed'] = dict(iteration=x, **panel(c['mixed_outcome_panel'], beta))
                f = scopes['failure']; m = scopes['mixed']
                assert f['turns'] + m['turns'] == row['turns']
                close((f['mean'] or 0)*f['turns'] + m['mean']*m['turns'], row['mean']*row['turns'])
            if key == 'mixed_reweight':
                rp = folder/'reweighting.json'; rw = read(rp)
                assert rw['mode'] == 'outcome_reweight' and not rw['all_failure_auxiliary']
                row['reweight_lambda'] = rw['lambda_value']
                row['actual_perturbation_to_outcome_rms'] = rw['perturbation_to_outcome_rms']
                inputs.append(rp)
            series[key]['rows'].append(scopes)
            receipts.append(dict(method=key, actor_iteration=x, source=str(folder),
                files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}))
    summaries = {}
    for key, s in series.items():
        s['rows'].sort(key=lambda r:r['main']['iteration'])
        assert s['rows'], key
        iterations = [r['main']['iteration'] for r in s['rows']]
        s['missing_iterations'] = sorted(set(range(max(iterations)+1))-set(iterations))
        s['reweight'] = key == 'mixed_reweight'
        s['failure_scope'] = 'Failure groups inside the main batch' if key == 'all_failure' else 'Extra all-failure buffer'
        s['windows'] = []
        for scope in ['main', 'failure', 'mixed']:
            rows = [r[scope] for r in s['rows'] if scope in r]
            if not rows: continue
            for lo, hi in [(0,19), (max(iterations)-19,max(iterations))]:
                rr = [r for r in rows if lo <= r['iteration'] <= hi]
                s['windows'].append(dict(scope=scope, first=lo, last=hi, iterations=len(rr),
                    turns=sum(r['turns'] for r in rr), labels=sum(r['labels'] for r in rr),
                    mean=weighted(rr,'mean','turns'),coverage=weighted(rr,'coverage','turns'),
                    credited_rate=weighted(rr,'credited_rate','labels'),chance=weighted(rr,'chance','labels')))
        summaries[key] = dict(durable_iterations=len(iterations), first=min(iterations), last=max(iterations),
                              missing=s['missing_iterations'], windows=s['windows'])
    data = dict(schema=1, title='ARM reward versus held-out task success',
        protocol=evaluations['protocol'], benchmark=evaluations['benchmark'],
        definitions={
            'success':'Independent GPT-4.1 terminal judge, not oracle ground truth. Overall: successes/300; valid-only: successes/valid tasks.',
            'arm':'Mean centered turn bonus b=beta*m*(credit-chance) over trainable turns, with unlabeled/gated-out turns set to zero. Gate C credits equivalent actions and subtracts their multiplicity/5; others credit executed response index and subtract 1/5.',
            'reweight':'Reweighting does not add b to the outcome advantage. Its plotted b is the common beta=0.5 ARM diagnostic; actual training uses lambda=0.5 outcome reweighting.',
            'alignment':'ARM at x=t was collected by actor checkpoint t before the update saving checkpoint t+1. Success at x=t evaluates checkpoint t. x counts collection/PPO cycles, not Adam updates.',
            'population':'Main = retained training batch before native epoch trimming, not a token/loss-weighted mean. Extra failure buffers are separate and shown before their Nf/48 loss coefficient. Historical All-failure includes failure groups inside the main batch.',
            'smoothing':'Trailing five-iteration pooled mean by default, weighted by turns (mean/coverage) or labels (selection/chance). Raw observations remain available. Missing iterations are not filled; evaluation points are unsmoothed.',
            'interpretation':'Divergence is a warning, not proof of hacking. ARM compares five same-policy responses: under exchangeable sampling and position-neutral selection, each response wins about 1/5 even as absolute quality changes. Filtering and decoding can break exchangeability. A flat centered mean cannot rule out hacking or measure absolute quality. Held-out task-success uses different tasks and varying collection dates/valid sets.',
            'provenance':'Only calibration summaries with a saved-checkpoint marker are included. Marker/validation identities and bonus formulas are cross-checked. Older auxiliary manifests store centered advantages without labels: binary credit and chance are recovered from advantage/beta. Failed collection attempts are excluded. Pruned historical checkpoint tensors are not reloaded.',
        }, baseline=baseline, series=list(series.values()),
        omitted=[{k:v for k,v in o.items() if k != 'path'} for o in omissions])
    PRIVATE.mkdir(parents=True,exist_ok=True)
    (PRIVATE/'source-receipt.json').write_text(json.dumps(dict(sources=receipts,omissions=omissions,summary=summaries),indent=2)+'\n')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'arm_reward_hacking.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    return data


def render(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    methods = ['all_failure','additive','gate_b','gate_c','mixed_bonus','mixed_reweight']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.titleweight':'bold'})
    fig, axs = plt.subplots(3,2,figsize=(13,11),sharex=True)
    blue, orange = '#2563eb','#c65b08'
    right_limit=.016
    for ax, key in zip(axs.flat, methods):
        s = next(s for s in data['series'] if s['id']==key)
        rows=[r['main'] for r in s['rows']]; xp=[r['iteration'] for r in rows]
        e=s['evaluations']; ax.plot([r['iteration'] for r in e], [100*r['successes']/r['tasks'] for r in e],
                                  color=blue,lw=2,marker='o',ms=4,zorder=4)
        ax.set_ylim(0,60);ax.set_xlim(0,101);ax.set_yticks(range(0,61,10));ax.tick_params(axis='y',colors=blue)
        ax.set_ylabel('Held-out success (%)',color=blue);ax.grid(alpha=.14)
        twin=ax.twinx();raw=[r['mean'] for r in rows];trend=smooth(rows,'mean','turns',5)
        assert all(abs(v)<=right_limit for v in raw), (key, min(raw), max(raw))
        twin.plot(xp,raw,color=orange,alpha=.2,lw=.9)
        twin.plot(xp,trend,color=orange,lw=1.8)
        twin.axhline(0,color=orange,alpha=.35,lw=.8,ls=':')
        twin.set_ylim(-right_limit,right_limit);twin.set_yticks([-.015,-.01,-.005,0,.005,.01,.015])
        twin.tick_params(axis='y',colors=orange,labelsize=8)
        twin.set_ylabel('Mean ARM proxy' if s['reweight'] else 'Mean ARM bonus',color=orange)
        title=s['label'].replace(' · relaxed B',' (relaxed B)').replace(' · duplicate-aware credit',' (action credit)')
        ax.set_title(title,loc='left',fontsize=11)
        if s['reweight']:ax.text(.02,.95,'Proxy only; training reweights outcome advantage',transform=ax.transAxes,va='top',fontsize=8)
    for ax in axs[-1]:ax.set_xlabel('Actor checkpoint (completed training iterations)')
    fig.suptitle('Does increasing ARM reward track task success?',fontsize=17)
    fig.legend(handles=[Line2D([0],[0],color=blue,marker='o',label='Held-out full-300 success (independent judge)'),
                        Line2D([0],[0],color=orange,label='Mean ARM signal (trailing 5 iterations; faint = raw)')],
               loc='upper center',bbox_to_anchor=(.5,.957),ncol=2,frameon=False,fontsize=10)
    fig.subplots_adjust(left=.065,right=.91,bottom=.135,top=.89,hspace=.29,wspace=.43)
    fig.text(.025,.024,'Success: local browser, GPT-4.1/action_history, actor T=0. ARM: retained main-batch turns; unlabeled turns contribute zero.\n'
             'ARM is aligned to the actor BEFORE its next update. Extra failure buffers and label coverage are available in the interactive view.\n'
             'Separate axes; all panels share their scales. Changing training states and live-web conditions prevent a causal reward-hacking claim.',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(OUT/f'arm_reward_hacking.{ext}',dpi=180)
    plt.close(fig)
    template=(REPO/'scripts/templates/arm_reward_hacking.html').read_text()
    assert template.count('__REWARD_DATA__')==1
    (OUT/'arm_reward_hacking.html').write_text(template.replace('__REWARD_DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')))
    fields=['method','scope','iteration','turns','labels','coverage','credited_rate','chance','mean','rms','beta','loss_coefficient']
    with (OUT/'arm_reward_hacking.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for s in data['series']:
            for r in s['rows']:
                for scope,p in r.items():writer.writerow(dict(method=s['id'],scope=scope,**{k:p.get(k) for k in fields[2:]}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    data=read(OUT/'arm_reward_hacking.json') if args.render_only else aggregate()
    render(data)
    print(json.dumps({s['id']:dict(iterations=len(s['rows']),missing=s['missing_iterations'],windows=s['windows']) for s in data['series']},indent=2))
