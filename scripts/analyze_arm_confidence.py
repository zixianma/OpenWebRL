#!/usr/bin/env python3
"""CPU analysis of a completed ARM confidence replay; no additional inference."""
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import statistics

from replay_arm_confidence import cases, confidence_metrics, spearman


def avg(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def bootstrap(rows, fn, seed=20260919, samples=1500):
    rng=random.Random(seed)
    values=[]
    for _ in range(samples):
        sample=[rows[rng.randrange(len(rows))] for _ in rows]
        value=fn(sample)
        if value is not None:values.append(value)
    values.sort()
    return [values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]] if values else None


def analyze(source, inputs, output):
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text())
    if hashlib.sha256(inputs.read_bytes()).hexdigest()!=manifest['inputs_sha256']:
        raise ValueError('Replay inputs changed')
    if not (source/'complete.json').exists():raise ValueError('Incomplete replay')
    panel=list(map(json.loads,inputs.read_text().splitlines()))
    records=list(map(json.loads,(source/'scores.jsonl').read_text().splitlines()))
    expected={f'{r["id"]}:{t}:{p}' for r,t,_ in cases(panel) for p in range(5)}
    if len(records)!=len(expected) or {r['pass_id'] for r in records}!=expected:
        raise ValueError('Replay pass coverage or uniqueness mismatch')
    groups=defaultdict(list)
    for r in records:
        if abs(sum(r['probabilities'])-1)>1e-6 or any(p<0 for p in r['probabilities']):
            raise ValueError('Invalid probabilities')
        groups[(r['state_id'],r['treatment'])].append(r)
    states=[]
    for (state_id,treatment),rows in groups.items():
        rows.sort(key=lambda r:int(r['pass_id'].rsplit(':',1)[1]));first=rows[0]
        if len(rows)!=5:raise ValueError('Incomplete ordering block')
        p=[avg(r['probabilities'][j] for r in rows) for j in range(5)]
        u=len(set(first['action_class_ids']))
        cls=first['selected_action_class']
        selected_mass=first['action_mass'][str(cls)]
        selected_count=Counter(first['action_class_ids'])[cls]
        agreement=avg(x['selected_action_class']==cls for x in rows[1:])
        chance=selected_count/5
        r={k:first[k] for k in ['state_id','task_id','run','iteration_index','stratum','split',
                    'reasoning_jaccard_distance','coordinate_clusters_10','coordinate_clusters_25','response_tokens_std']}
        r.update(treatment=treatment, unique_actions=u, first_confidence=first['candidate_max'],
                 first_margin=first['candidate_margin'], first_selected_class_mass=selected_mass,
                 first_selected_class_excess=selected_mass-selected_count/5,
                 first_selected_slot=first['selected_slot'], first_selected_class=cls,
                 first_entropy=first['candidate_entropy'], mean_raw_confidence=avg(x['candidate_max'] for x in rows),
                 averaged_confidence=max(p), averaged_entropy=confidence_metrics(p,first['action_class_ids'])['candidate_entropy'],
                 heldout_candidate_agreement=avg(x['selected_slot']==first['selected_slot'] for x in rows[1:]),
                 heldout_action_agreement=agreement, selected_class_chance_baseline=chance,
                 chance_corrected_agreement=(agreement-chance)/(1-chance) if chance<1 else None,
                 input_tokens=first['input_tokens'], legal_index_mass=first['legal_index_mass'],
                 logged_greedy_match=(first['selected_source']==first['logged_selected_index']) if first['logged_selected_index'] is not None else None)
        states.append(r)
    natural=[r for r in states if r['treatment']=='natural']
    informative=[r for r in natural if r['unique_actions']>1]
    names=['all_distinct_actions','partially_duplicated_actions','all_same_action']
    summary=dict(source=str(source),passes=len(records),natural_tasks=len(natural),
        probability_definition='Pre-grammar selection-index scores normalized across five legal indices; conditional on generated JSON prefix.',
        agreement_definition='First ordering chooses a winner; agreement is measured on the other four orderings, after mapping back to original candidates/actions.',
        ordering_definition='The first ordering is the archived permutation when available, otherwise candidate identity order. All five cyclic rotations are scored; balanced-rotation sensitivity treats each in turn as the reference.',
        caveats=['Neither confidence nor order agreement measures correctness.',
                 'Same-action states have trivial action agreement and are excluded from threshold analysis.',
                 'Balanced 30/30/30 panel; coverage is panel coverage, not a population estimate.',
                 'Bootstrap intervals resample states, not the five dependent orderings; they do not correct checkpoint/site confounding or multiple exploratory comparisons.',
                 'Natural candidate count is observational. Synthetic duplication is a within-state control with omitted alternatives, so it changes choice difficulty too.'],
        strata={},correlations={},thresholds=[],synthetic_controls={})
    fields=['first_confidence','mean_raw_confidence','averaged_confidence','first_entropy','first_margin',
            'heldout_candidate_agreement','heldout_action_agreement','selected_class_chance_baseline',
            'chance_corrected_agreement','reasoning_jaccard_distance','response_tokens_std','input_tokens']
    for name in names:
        rows=[r for r in natural if r['stratum']==name]
        summary['strata'][name]={'n':len(rows),**{k:avg(r[k] for r in rows) for k in fields},
            'confidence_ci95':bootstrap(rows,lambda xs:avg(r['first_confidence'] for r in xs)),
            'unique_action_counts':dict(Counter(r['unique_actions'] for r in rows))}
    for name,rows in [('all_natural',natural),('informative_only',informative)]+[(s,[r for r in natural if r['stratum']==s]) for s in names]:
        summary['correlations'][name]={}
        for left,right in [('unique_actions','first_confidence'),('coordinate_clusters_10','first_confidence'),
                           ('reasoning_jaccard_distance','first_confidence'),('response_tokens_std','first_confidence'),
                           ('input_tokens','first_confidence'),('first_confidence','heldout_candidate_agreement'),
                           ('first_confidence','heldout_action_agreement'),('first_confidence','chance_corrected_agreement')]:
            valid=[r for r in rows if r.get(left) is not None and r.get(right) is not None]
            value=spearman(valid,left,right)
            summary['correlations'][name][f'{left} vs {right}']={'rho':value,'n':len(valid),
                'ci95':bootstrap(valid,lambda xs:spearman(xs,left,right)) if value is not None else None}
    for split in ['development','validation']:
        rows=[r for r in informative if r['split']==split]
        for threshold in [0.,.5,.6,.7,.8,.9]:
            keep=[r for r in rows if r['first_confidence']>=threshold]
            summary['thresholds'].append(dict(split=split,threshold=threshold,kept=len(keep),total=len(rows),
                 coverage=len(keep)/len(rows),heldout_action_agreement=avg(r['heldout_action_agreement'] for r in keep),
                 chance_corrected_agreement=avg(r['chance_corrected_agreement'] for r in keep),
                 heldout_candidate_agreement=avg(r['heldout_candidate_agreement'] for r in keep),
                 distinct_kept=sum(r['unique_actions']==5 for r in keep),partial_kept=sum(r['unique_actions']<5 for r in keep)))
    originals={r['state_id']:r for r in natural}
    for treatment in ['synthetic_three_copies','synthetic_five_copies']:
        rows=[r for r in states if r['treatment']==treatment]
        summary['synthetic_controls'][treatment]={'n':len(rows),
             'natural_confidence':avg(originals[r['state_id']]['first_confidence'] for r in rows),
             'synthetic_confidence':avg(r['first_confidence'] for r in rows),
             'confidence_delta':avg(r['first_confidence']-originals[r['state_id']]['first_confidence'] for r in rows),
             'confidence_delta_ci95':bootstrap(rows,lambda xs:avg(r['first_confidence']-originals[r['state_id']]['first_confidence'] for r in xs)),
             'mean_raw_confidence':avg(r['mean_raw_confidence'] for r in rows),
             'averaged_confidence':avg(r['averaged_confidence'] for r in rows),
             'maximum_first_confidence':max(r['first_confidence'] for r in rows),
             'first_confidence_at_least_07':sum(r['first_confidence']>=.7 for r in rows)}
    matches=[r for r in natural if r['logged_greedy_match'] is not None]
    summary['archived_greedy_reproduction']={'states':len(matches),'matches':sum(r['logged_greedy_match'] for r in matches),
                                           'mismatches':[r['state_id'] for r in matches if not r['logged_greedy_match']]}
    summary['raw_score_checks']={'unique_passes':len(expected),'minimum_legal_index_mass':min(r['legal_index_mass'] for r in records),
                                'median_pass_seconds':statistics.median(r['seconds'] for r in records),
                                'json_prefix_counts':dict(Counter(r['selection_prefix'] for r in records))}
    summary['identical_copy_display_position_counts']=dict(Counter(
        r['permutation'].index(r['selected_slot'])+1 for r in records if r['treatment']=='synthetic_five_copies'))
    summary['natural_display_position_counts']={s:dict(Counter(r['permutation'].index(r['selected_slot'])+1
           for r in records if r['treatment']=='natural' and r['stratum']==s)) for s in names}
    summary['balanced_rotation_validation_thresholds']=[]
    for threshold in [0.,.5,.6,.7,.8,.9]:
        agreements=[];total=0
        for (_,treatment),rows in groups.items():
            if treatment!='natural' or rows[0]['split']!='validation' or len(set(rows[0]['action_class_ids']))==1:continue
            for r in rows:
                total+=1
                if r['candidate_max']>=threshold:
                    agreements.append(avg(x['selected_action_class']==r['selected_action_class']
                                          for x in rows if x['pass_id']!=r['pass_id']))
        summary['balanced_rotation_validation_thresholds'].append(dict(threshold=threshold,
             retained_orderings=len(agreements),total_orderings=total,independent_tasks=30,
             coverage=len(agreements)/total,heldout_action_agreement=avg(agreements)))
    (output/'confidence-analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (output/'confidence-states.jsonl').open('w') as f:
        for r in states:f.write(json.dumps(r)+'\n')
    plot(natural,states,summary,output)
    page=['<!doctype html><meta charset="utf-8"><title>ARM confidence audit</title>',
          '<style>body{font:16px system-ui;max-width:1200px;margin:35px auto;color:#172535}img{width:100%}td,th{padding:9px;border-bottom:1px solid #ccc;text-align:right}td:first-child,th:first-child{text-align:left}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>',
          '<h1>ARM confidence, candidate diversity, and order sensitivity</h1>',
          '<p>90 distinct tasks, 650 selector passes. Confidence and repeated-order agreement are diagnostics, not accuracy.</p>',
          '<img src="confidence-diversity.png" alt="ARM confidence distributions, diversity correlations, and duplication controls">',
          '<h2>Thresholds on the held-out 30 informative tasks</h2>',
          '<p>One ordering supplies confidence; four other orderings measure agreement. All-identical-action states are excluded.</p>',
          '<table><tr><th>Confidence threshold</th><th>Retained</th><th>Coverage</th><th>Action agreement</th><th>Partial-duplicate states retained</th></tr>']
    for r in summary['thresholds']:
        if r['split']!='validation':continue
        agreement=f'{100*r["heldout_action_agreement"]:.1f}%' if r['heldout_action_agreement'] is not None else '—'
        page.append(f'<tr><td>{r["threshold"]:.1f}</td><td>{r["kept"]}/{r["total"]}</td><td>{100*r["coverage"]:.1f}%</td><td>{agreement}</td><td>{r["partial_kept"]}</td></tr>')
    page+=['</table><h2>Definitions and full measurements</h2><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>']
    (output/'confidence-report.html').write_text('\n'.join(page))
    print(json.dumps({k:summary[k] for k in ['strata','synthetic_controls','archived_greedy_reproduction','thresholds']},indent=2))


def plot(natural,states,summary,out):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng=random.Random(42)
    colors={'all_distinct_actions':'#2879b9','partially_duplicated_actions':'#d98a24','all_same_action':'#8b72af'}
    names=list(colors); labels=['5 distinct actions','2–4 distinct actions','1 distinct action']
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.5),constrained_layout=True)
    ax=axes[0,0]
    data=[[r['first_confidence'] for r in natural if r['stratum']==name] for name in names]
    ax.boxplot(data,tick_labels=labels,showfliers=False,medianprops={'color':'black'})
    for i,(values,name) in enumerate(zip(data,names),1):
        ax.scatter([i+rng.uniform(-.13,.13) for _ in values],values,s=21,alpha=.6,color=colors[name])
    ax.set(ylabel='P(selected response), first ordering',ylim=(0,1.04),title='A. Confidence across natural candidate sets')
    ax=axes[0,1]
    for name,label in zip(names,labels):
        rows=[r for r in natural if r['stratum']==name]
        ax.scatter([r['reasoning_jaccard_distance'] for r in rows],[r['first_confidence'] for r in rows],label=label,color=colors[name],alpha=.75,s=27)
    ax.set(xlabel='Reasoning lexical distance (Jaccard)',ylabel='First-ordering confidence',ylim=(0,1.04),title='B. Reasoning diversity vs confidence')
    ax.legend(frameon=False,fontsize=8)
    ax=axes[1,0]
    for name,label in zip(names[:2],labels[:2]):
        rows=[r for r in natural if r['stratum']==name]
        ax.scatter([r['first_confidence'] for r in rows],[r['heldout_action_agreement']+rng.uniform(-.012,.012) for r in rows],label=label,color=colors[name],alpha=.7,s=27)
    ax.set(xlabel='First-ordering confidence',ylabel='Agreement over four other orderings',ylim=(-.05,1.05),xlim=(.2,1),title='C. Action agreement (60 informative states)')
    ax=axes[1,1]
    baseline={r['state_id']:r for r in natural}
    three={r['state_id']:r for r in states if r['treatment']=='synthetic_three_copies'}
    five={r['state_id']:r for r in states if r['treatment']=='synthetic_five_copies'}
    for state_id,r in three.items():
        ax.plot([0,1,2],[baseline[state_id]['first_confidence'],r['first_confidence'],five[state_id]['first_confidence']],color='#999999',alpha=.32,lw=.8)
    means=[avg(baseline[k]['first_confidence'] for k in three),avg(r['first_confidence'] for r in three.values()),avg(r['first_confidence'] for r in five.values())]
    ax.plot([0,1,2],means,'o-',color='#a63147',lw=2,label='Mean of 20 matched states')
    ax.axhline(.2,color='#444444',ls=':',lw=1,label='Uniform over five indices')
    ax.set(xticks=[0,1,2],xticklabels=['Original five','3 copies + 2 others','5 identical copies'],ylabel='First-ordering confidence',ylim=(0,1.04),title='D. Controlled response duplication')
    ax.legend(frameon=False,fontsize=8)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16)
    fig.suptitle('SelectionARM confidence audit · 90 tasks · 650 passes\nConfidence and order agreement do not establish correctness',fontsize=13)
    fig.savefig(out/'confidence-diversity.png',dpi=190)
    fig.savefig(out/'confidence-diversity.svg')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores',required=True,type=Path)
    p.add_argument('--inputs',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    analyze(a.scores,a.inputs,a.output)
