#!/usr/bin/env python3
"""Aggregate saved JSON journals only; no tensor loading, GPUs, APIs or trajectory publishing."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

REPO=Path(__file__).resolve().parents[1]
RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/evaluations')
RUNS={'arm-turn-bonus-fresh-294197':'Original bonus','arm-allfailure-bonus-295353':'All-failure',
      'arm-failure-additive-295786':'Additive','arm-gate-b-309053':'B: relaxed gate',
      'arm-gate-c-309054':'C: action credit'}
OUTPUT=REPO/'openwebrl/docs/arm_results/rl_integration/failure-sampling-history.json'

def read(p):return json.loads(p.read_text())
def counts(rows):
    return dict(turns=len(rows),sampled=sum(bool(r.get('arm_turn_bonus',{}).get('sampled')) for r in rows),
        usable=sum(bool(r.get('arm_turn_bonus',{}).get('eligible')) for r in rows))
def correlation(rows,x,y):
    pairs=[(r.get(x),r.get(y)) for r in rows if r.get(x) is not None and r.get(y) is not None]
    if len(pairs)<3:return None
    try:return statistics.correlation([p[0] for p in pairs],[p[1] for p in pairs])
    except statistics.StatisticsError:return None

def aggregate():
    seen={}; omissions=[]
    for mp in sorted(RUNTIME.glob('arm-*/launch_manifest.json')):
        m=read(mp); name=RUNS.get(m.get('arm_config',{}).get('run_id'))
        if not name:continue
        for marker in sorted((mp.parent/'iterations').glob('*/checkpoint-saved.json')):
            folder=marker.parent; cp=folder/'calibration.json';vp=folder/'checkpoint-validation.json'
            if not cp.exists():omissions.append(str(folder));continue
            c=read(cp); rid=int(folder.name); key=(name,rid)
            if key in seen:raise ValueError('Duplicate durable collection: '+str(key))
            v=read(vp) if vp.exists() else {}
            row=dict(method=name,iteration=rid+1,optimizer_updates=v.get('completed_optimizer_updates'),
                source=str(folder),q=c['config']['scored_fraction'],beta=c['config']['beta'],
                native_rows=c.get('batch_rows'),native_zero_reward_turns=c.get('outcome_breakdown',{}).get('failure',{}).get('rows'),
                auxiliary_groups=c.get('additive_failure_groups',c.get('all_failure_groups',0)),
                auxiliary_turns=c.get('additive_failure_rows',c.get('all_failure_rows',0)),
                auxiliary_usable=c.get('additive_failure_labels',c.get('all_failure_usable_labels',0)),
                ordinary_groups=c.get('outcome_groups_preserved',c.get('mixed_outcome_groups',48)),
                collection_seconds=c.get('elapsed_collection_seconds'))
            paths=list((folder/'groups').rglob('*.json'))
            if not paths:
                row['journals_available']=False;seen[key]=row;continue
            groups=[read(p) for p in paths]
            if any(g['policy_id']!=c['policy_id'] for g in groups):raise ValueError('Policy mismatch')
            allrows=[r for g in groups for r in g['rows']]
            zero=[r for r in allrows if r.get('reward')==0]
            auxfile=folder/'failure_auxiliary.json'
            if auxfile.exists():
                a=read(auxfile);ids={str(r['group_index']) for r in a['records']}
                aux=[r for g in groups if str(g['group_id']) in ids for r in g['rows']]
                # Every retained historical group has >=1 usable label; no guessing missing IDs.
                if len(ids)!=a['failure_groups'] or len(aux)!=a['total_failure_rows']:
                    raise ValueError('Auxiliary/journal denominator mismatch: '+str(folder))
                if counts(aux)['usable']!=len(a['records']):
                    raise ValueError('Usable label mismatch: '+str(folder))
            elif name=='All-failure':
                aux=[r for g in groups if g['accepted'] and all(s.get('reward')==0 for s in g['rows'])
                     for r in g['rows'] if r['index'] in set(g['accepted_sample_indices'])]
            else:aux=[]
            hist=Counter(g.get('filter_reason') or 'accepted' for g in groups)
            row.update(journals_available=True,completed_candidate_groups=len(groups),
                accepted_journal_groups=sum(bool(g['accepted']) for g in groups),
                all_zero_candidate_groups=sum(all(r.get('reward')==0 for r in g['rows']) for g in groups),
                filter_reasons=dict(hist),collected_turns=len(allrows),
                collected_zero_reward_turns=len(zero),collected_zero_reward_sampled=counts(zero)['sampled'],
                collected_zero_reward_usable=counts(zero)['usable'],
                auxiliary_journal_turns=len(aux),auxiliary_sampled=counts(aux)['sampled'],
                auxiliary_journal_usable=counts(aux)['usable'],
                collected_sample_fraction=counts(zero)['sampled']/len(zero) if zero else None,
                auxiliary_sample_fraction=counts(aux)['sampled']/len(aux) if aux else None)
            seen[key]=row
    rows=sorted(seen.values(),key=lambda r:(r['method'],r['iteration']))
    summary={}
    for name in RUNS.values():
        rr=[r for r in rows if r['method']==name]; jj=[r for r in rr if r['journals_available']]
        def span(k):
            vv=[r[k] for r in jj if r.get(k) is not None]
            return dict(min=min(vv),median=statistics.median(vv),max=max(vv)) if vv else None
        summary[name]=dict(completed_iterations=len(rr),journal_iterations=len(jj),
            zero_auxiliary_iterations=sum(r['auxiliary_turns']==0 for r in rr),
            candidate_groups=span('completed_candidate_groups'),auxiliary_groups=span('auxiliary_groups'),
            auxiliary_turns=span('auxiliary_turns'),auxiliary_sampled=span('auxiliary_sampled'),
            auxiliary_usable=span('auxiliary_usable'),collected_zero_reward_turns=span('collected_zero_reward_turns'),
            collected_sample_fraction=sum(r['collected_zero_reward_sampled'] for r in jj)/sum(r['collected_zero_reward_turns'] for r in jj) if jj else None,
            auxiliary_sample_fraction=sum(r['auxiliary_sampled'] for r in jj)/sum(r['auxiliary_journal_turns'] for r in jj) if sum(r['auxiliary_journal_turns'] for r in jj) else None,
            corr_collection_groups_failure_turns=correlation(jj,'completed_candidate_groups','collected_zero_reward_turns'),
            corr_auxiliary_turns_sampled=correlation(jj,'auxiliary_turns','auxiliary_sampled'))
    data=dict(schema=1,definitions={
        'iteration':'Completed rollout collection + PPO/checkpoint cycle; not Adam update. Adam counters are separate.',
        'collected_zero_reward':'All journaled zero-reward turns, INCLUDING truncated/aborted trajectories; not a valid-failure count.',
        'auxiliary':'Strictly admitted all-failure groups. Additive: side buffer outside the48 mixed groups. All-failure: inside48.',
        'sampled':'Bernoulli sampling flags, including candidate/gate failures. Usable labels are separate.',
        'denominators':'Group journals are before final native pruning/shuffle/PPO epoch trimming. Auxiliary additive denominators are validated against exact manifests.',
        'collection_groups':'Completed journaled groups, not all submitted/in-flight/canceled groups.',
        'causality':'Descriptive counts/correlations; on-policy task/state distributions change and no causal attribution is claimed.'},
        summary=summary,omitted_without_calibration=omissions,rows=rows)
    # Verify the task-sampling mode against actual native argument dumps.
    evidence=['295786','299277','303574','313669']
    for job in evidence:
        with (RUNTIME/f'arm-failure-additive-{job}/collection.log').open(errors='replace') as f:
            setting=next(line for line in f if 'enable_adaptive_query_sampling ' in line)
        if not setting.strip().endswith('False'):raise ValueError('Adaptive sampler changed')
    data['sampler_audit']=dict(enable_adaptive_query_sampling=False,rollout_shuffle=True,
        evidence_jobs=evidence,evidence='Actual native argument dumps in collection.log',
        mechanism='Native outcome-variance filtering fills48 mixed groups; up to8 strict valid all-failure groups with a usable q=.2 label form the additive buffer.',
        implication='Constant q does not imply constant supervision. Fewer admitted groups lower both sampled turns and the Nf/48 auxiliary coefficient.')
    additive=[r for r in rows if r['method']=='Additive'];windows=[]
    for lo,hi in [(1,20),(21,40),(41,60),(61,80),(81,100)]:
        window=[r for r in additive if lo<=r['iteration']<=hi and r['journals_available']]
        windows.append(dict(first=lo,last=hi,**{k:sum(r[k] for r in window)/len(window) for k in
            ['completed_candidate_groups','collected_zero_reward_turns','collected_zero_reward_sampled',
             'auxiliary_groups','auxiliary_turns','auxiliary_sampled','auxiliary_usable']}))
    data['additive_windows']=windows
    OUTPUT.parent.mkdir(parents=True,exist_ok=True);OUTPUT.write_text(json.dumps(data,indent=2)+'\n')
    return data

def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors=['#2459a6','#ec8b23','#229679','#9855af','#d1445c']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    rows=[r for r in data['rows'] if r['method']=='Additive' and r['journals_available']]
    x=[r['iteration'] for r in rows]
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,layout='constrained')
    panels=[
      [('collected_zero_reward_turns','Collected zero-reward turns',colors[0]),('collected_zero_reward_sampled','Sampled for ARM',colors[1]),('collected_zero_reward_usable','Usable ARM labels',colors[2])],
      [('auxiliary_turns','Admitted failure turns',colors[0]),('auxiliary_sampled','Sampled for ARM',colors[1]),('auxiliary_usable','Usable ARM labels',colors[2])],
      [('completed_candidate_groups','Completed candidate groups',colors[0]),('all_zero_candidate_groups','All-zero candidate groups',colors[1]),('ordinary_groups','Retained mixed groups',colors[2])],
      [('collected_sample_fraction','Collected zero-reward turns',colors[0]),('auxiliary_sample_fraction','Admitted failure turns',colors[1])]]
    titles=['Collected failures (includes invalid trajectories)','Admitted all-failure buffer (strict validity)',
            'Dynamic sampling: effort to fill 48 mixed groups','Realized sampling rate (q stays 20%)']
    for j,(ax,series,title) in enumerate(zip(axes.flat,panels,titles)):
        for key,label,col in series:
            yy=[r.get(key) if r.get(key) is not None else np.nan for r in rows]
            if j==3:yy=[v*100 for v in yy]
            ax.plot(x,yy,label=label,color=col,lw=1.2,alpha=.85)
        ax.set_title(title,loc='left',fontsize=11);ax.set_ylabel('Percent' if j==3 else ('Groups' if j==2 else 'Turns'))
        ax.set_xlim(0,101);ax.set_ylim(bottom=0);ax.grid(alpha=.18);ax.legend(fontsize=8,loc='upper right')
        if j==3:ax.axhline(20,color='black',ls='--',lw=1)
        if j>=2:ax.set_xlabel('Completed training iteration (collection + PPO)')
    fig.suptitle('Additive ARM: failure population and sampling across 100 iterations',fontsize=15)
    fig.get_layout_engine().set(rect=(0,.065,1,.91))
    fig.text(.02,.012,'JSON journals only • sampled ≠ usable labels • buffer remains outside 48 mixed groups • no-data gaps are omitted\nCounts precede native epoch trimming; rate in admitted groups is conditional on at least one usable label.',fontsize=9)
    out=REPO/'openwebrl/docs/rl_results/arm_additive_failure_sampling.png';fig.savefig(out,dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,layout='constrained')
    for name,col in zip(RUNS.values(),colors):
        rr=[r for r in data['rows'] if r['method']==name and r['journals_available']]
        for ax,key in zip(axes,['collected_zero_reward_turns','collected_zero_reward_sampled','auxiliary_turns']):
            ax.plot([r['iteration'] for r in rr],[r[key] for r in rr],label=name,color=col,lw=1,alpha=.8)
    for ax,title in zip(axes,['Collected zero-reward turns (includes invalid trajectories)',
                            'Collected zero-reward turns sampled for ARM',
                            'Admitted all-failure turns (original bonus: no failure-only groups)']):
        ax.set_title(title,loc='left',fontsize=11);ax.set_ylabel('Turns');ax.set_ylim(bottom=0);ax.grid(alpha=.18)
    axes[0].legend(ncol=3,fontsize=9);axes[-1].set_xlabel('Completed training iteration (collection + PPO)');axes[-1].set_xlim(0,101)
    fig.suptitle('ARM variants: failure-turn and sampling counts',fontsize=15)
    fig.get_layout_engine().set(rect=(0,.055,1,.92))
    fig.text(.02,.012,'q = 20% throughout these historical runs • B/C use a different candidate gate • All-failure shares 48 group slots; Additive/B/C preserve 48 mixed slots\nOnly durable iterations; journal coverage can begin later than training. No smoothing or causal comparison.',fontsize=9)
    fig.savefig(REPO/'openwebrl/docs/rl_results/arm_variants_failure_sampling.png',dpi=170);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plot-only',action='store_true');a=p.parse_args()
    data=read(OUTPUT) if a.plot_only else aggregate();plot(data);print(json.dumps(data['summary'],indent=2))
