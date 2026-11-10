#!/usr/bin/env python3
"""Bounded JSON-only audit of Gate B: no tensors, models, browsers or API calls."""
import argparse
from collections import defaultdict,Counter
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import statistics
import time

RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
REPO=Path(__file__).resolve().parents[1]
RUN='arm-gate-b-309053'


def avg(xs):return statistics.mean(xs) if xs else None

def reconstruct_group(header,policy):
    ids=set(header['accepted_sample_indices'])
    rows=[r for r in header['rows'] if r['index'] in ids]
    trajectories=defaultdict(list)
    for r in rows:trajectories[str(r['trajectory_id'])].append(r)
    rewards=[]
    for tr in trajectories.values():
        outcome=tr[0]['reward']
        if outcome not in (-1,0,1) or any(r['reward']!=outcome for r in tr):raise ValueError('Inconsistent terminal outcome')
        rewards.append(outcome)
    if len(set(rewards))<2:raise ValueError('Accepted group must have nonzero outcome variance')
    mean=statistics.mean(rewards);std=statistics.stdev(rewards)
    result=[]
    for tid,tr in trajectories.items():
        outcome=tr[0]['reward'];adv=(outcome-mean)/(std+1e-6);turns=[]
        for r in tr:
            if not r['trainable']:continue
            label=r.get('arm_turn_bonus',{});eligible=bool(label.get('eligible'));unit=0.
            if eligible:
                classes=label.get('action_class_ids',[]);j=label.get('selected_index')
                if (label.get('policy_id')!=policy or label.get('candidate_gate')!='min2'
                    or label.get('credit_assignment')!='response_index' or len(classes)!=5
                    or len(set(classes))<2 or type(j) is not int or not 0<=j<5):
                    raise ValueError('Unverified B label')
                unit=float(j==0)-.2
                if label['unit_bonus']!=unit:raise ValueError('Stored bonus mismatch')
            turns.append(dict(outcome=outcome,A=adv,u=unit,b=.5*unit,labeled=eligible,
                              win=eligible and label['selected_index']==0,
                              sampled=bool(label.get('sampled'))))
        if turns:
            result.append(dict(task=str(tr[0]['task_id']),trajectory=tid,outcome=outcome,A=adv,turns=turns,
                               mean_bonus=avg([t['b'] for t in turns]),
                               win_rate=avg([int(t['win']) for t in turns if t['labeled']])))
    return result


def reweighted(trajectory,lam):
    A=trajectory['A'];sign=1 if A>0 else -1 if A<0 else 0
    raw=[math.exp(lam*sign*t['u']) for t in trajectory['turns']]
    normalizer=statistics.mean(raw)
    return [(w/normalizer,A*w/normalizer) for w in raw]


def cluster_ci(groups,key,n=2000):
    clusters=defaultdict(list)
    for g in groups:
        if g.get(key) is not None:clusters[g['task']].append(g[key])
    pairs=[(sum(v),len(v)) for v in clusters.values()]
    if len(pairs)<2:return None
    rng=random.Random(20260926);means=[]
    for _ in range(n):
        sampled=rng.choices(pairs,k=len(pairs))
        means.append(sum(x[0] for x in sampled)/sum(x[1] for x in sampled))
    means.sort()
    return dict(task_clusters=len(pairs),groups=sum(n for _,n in pairs),
                mean=sum(s for s,_ in pairs)/sum(n for _,n in pairs),
                lower=means[int(.025*n)],upper=means[int(.975*n)],bootstrap_replicates=n)


def summarize(trajectories,groups):
    turns=[t for tr in trajectories for t in tr['turns']]
    outcomes={}
    for outcome,name in [(1,'successful'),(0,'failed'),(-1,'native_error_sentinel')]:
        ts=[t for t in turns if t['outcome']==outcome];ls=[t for t in ts if t['labeled']]
        trs=[tr for tr in trajectories if tr['outcome']==outcome]
        outcomes[name]=dict(trajectories=len(trs),turns=len(ts),usable_labels=len(ls),
            coverage=len(ls)/len(ts) if ts else None,selected=sum(t['win'] for t in ls),
            win_rate=sum(t['win'] for t in ls)/len(ls) if ls else None,
            bonus_mean_all_turns=avg([t['b'] for t in ts]),
            trajectory_mean_bonus=avg([tr['mean_bonus'] for tr in trs]))
    a2=sum(t['A']**2 for t in turns);b2=sum(t['b']**2 for t in turns)
    between=sum(len(tr['turns'])*tr['mean_bonus']**2 for tr in trajectories)
    weights={}
    for lam in (.25,.5,1.):
        changed=[(t,w,new) for tr in trajectories for t,(w,new) in zip(tr['turns'],reweighted(tr,lam))]
        deltas=[new-t['A'] for t,w,new in changed]
        ws=sorted(w for t,w,new in changed)
        weights[str(lam)]=dict(delta_to_outcome_rms=math.sqrt(sum(d*d for d in deltas)/a2),
            minimum_weight=ws[0],maximum_weight=ws[-1],p05=ws[int(.05*(len(ws)-1))],p95=ws[int(.95*(len(ws)-1))],
            sign_flips=sum(t['A']*new<0 for t,w,new in changed),
            maximum_trajectory_mean_advantage_error=max(abs(avg([a for w,a in reweighted(tr,lam)])-tr['A']) for tr in trajectories))
    return dict(groups=len(groups),binary_only_groups=sum(g.get('binary_only',False) for g in groups),
        trajectories=len(trajectories),turns=len(turns),outcomes=outcomes,
        outcome_rms=math.sqrt(a2/len(turns)),bonus_rms=math.sqrt(b2/len(turns)),
        bonus_to_outcome_rms=math.sqrt(b2/a2),
        scalar_advantage_bonus_cosine=sum(t['A']*t['b'] for t in turns)/math.sqrt(a2*b2),
        opposing_sign_labeled_fraction=sum(t['A']*t['b']<0 for t in turns)/sum(t['labeled'] for t in turns),
        sign_flips=sum(t['A']*(t['A']+t['b'])<0 for t in turns),
        bonus_energy_in_trajectory_means=between/b2,
        within_group_mean_bonus_difference=cluster_ci(groups,'bonus_difference'),
        within_group_labeled_win_rate_difference=cluster_ci(groups,'win_rate_difference'),
        outcome_aware_reweighting=weights)


def audit(output):
    start=time.monotonic();read_bytes=0;files=0
    def read(p):
        nonlocal read_bytes,files
        n=p.stat().st_size
        if n>4*1024**2 or read_bytes+n>1024**3 or time.monotonic()-start>240:
            raise RuntimeError('Bounded CPU audit limit reached; no unreported partial result')
        read_bytes+=n;files+=1
        return json.loads(p.read_text())
    iterations={};excluded=[];sources={}
    for mp in sorted((RUNTIME/'evaluations').glob('arm-*/launch_manifest.json')):
        m=read(mp)
        if m.get('wandb_run_id')!=RUN:continue
        source=Path(m['source']);sources[str(source)]=hashlib.sha256((source/'slime/ray/rollout.py').read_bytes()).hexdigest()
        for marker in sorted((mp.parent/'iterations').glob('*/checkpoint-saved.json')):
            f=marker.parent;iteration=int(f.name)+1
            if iteration in iterations:raise ValueError('Duplicate durable iteration')
            c=read(f/'calibration.json')
            journal_folder=f
            paths=sorted((journal_folder/'groups').glob('*/*.json'))
            replay=c['config'].get('gate_replay_origin') or c['config'].get('replay_origin')
            if not paths and replay:
                candidate=Path(replay['source'])/'iterations'/f'{iteration-1:04d}'
                prior=candidate/'calibration.json'
                if (prior.is_file() and replay.get('calibration_sha256')
                    and hashlib.sha256(prior.read_bytes()).hexdigest()==replay['calibration_sha256']):
                    journal_folder=candidate
                    paths=sorted((candidate/'groups').glob('*/*.json'))
            if not paths:
                excluded.append(dict(iteration=iteration,reason='No JSON group journals',folder=str(f)));continue
            if c['config']['beta']!=.5 or c['applied_beta']!=.5:raise ValueError('Mixed beta changed')
            all_tr=[];groups=[]
            for p in paths:
                h=read(p)
                if not h['accepted']:continue
                if h['policy_id']!=c['policy_id']:raise ValueError('Wrong journal policy')
                tr=reconstruct_group(h,c['policy_id']);all_tr+=tr
                yes=[t for t in tr if t['outcome']==1];no=[t for t in tr if t['outcome']==0]
                sy=[t['win_rate'] for t in yes if t['win_rate'] is not None]
                sn=[t['win_rate'] for t in no if t['win_rate'] is not None]
                binary_only=all(t['outcome'] in (0,1) for t in tr)
                groups.append(dict(task=tr[0]['task'],iteration=iteration,binary_only=binary_only,
                    bonus_difference=avg([t['mean_bonus'] for t in yes])-avg([t['mean_bonus'] for t in no]) if binary_only and yes and no else None,
                    win_rate_difference=avg(sy)-avg(sn) if binary_only and sy and sn else None))
            turns=[t for tr in all_tr for t in tr['turns']]
            observed=dict(rows=len(turns),admitted=sum(t['labeled'] for t in turns),
                          outcome_rms=math.sqrt(avg([t['A']**2 for t in turns])),
                          unit_bonus_rms=math.sqrt(avg([t['u']**2 for t in turns])))
            if len(groups)!=48:raise ValueError('Ordinary quota mismatch')
            for k,v in observed.items():
                if not math.isclose(v,c[k],rel_tol=2e-6,abs_tol=1e-7):
                    raise ValueError(f'Journal/calibration mismatch iter{iteration} {k}: {v} vs {c[k]}')
            iterations[iteration]=dict(trajectories=all_tr,groups=groups,folder=str(f),journal_folder=str(journal_folder),
                calibration_sha256=hashlib.sha256((f/'calibration.json').read_bytes()).hexdigest(),
                checked=observed)
    if not iterations:raise ValueError('No verified completed B collections')
    # Private group-level table retains task identities for reproducibility.
    output.mkdir(parents=True,exist_ok=True)
    with (output/'groups.jsonl').open('w') as handle:
        for it in sorted(iterations):
            for g in iterations[it]['groups']:handle.write(json.dumps(g)+'\n')
    windows={}
    for lo,hi in [(1,20),(21,40),(41,60),(61,69)]:
        picked=[r for it,r in iterations.items() if lo<=it<=hi]
        if picked:windows[f'{lo}-{hi}']=summarize([t for r in picked for t in r['trajectories']],[g for r in picked for g in r['groups']])
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),run=RUN,iterations=sorted(iterations),
        excluded=excluded,source_normalizer_sha256=sources,
        checks={str(it):{k:v for k,v in d.items() if k not in ('groups','trajectories')} for it,d in iterations.items()},
        aggregate=summarize([t for r in iterations.values() for t in r['trajectories']],[g for r in iterations.values() for g in r['groups']]),
        windows=windows,resources=dict(gpus=0,api_calls=0,tensors_loaded=0,files=files,bytes_read=read_bytes,
            wall_seconds=time.monotonic()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        limitations=['Accepted trainable turns before native PPO epoch trimming; not exact optimizer gradients.',
            'Native -1 error sentinels are kept for exact scale reconstruction but excluded from binary-only paired-group outcome associations.',
            'All counts and RMS reconstructed from trajectory-normalized outcomes and checked against each saved calibration.',
            'Task-cluster bootstrap describes this single training lineage; it does not quantify training-seed uncertainty.',
            'Outcome association is not action correctness: good turns occur in failed trajectories and conversely.',
            'Reweighting is a saved-batch scalar diagnostic, not a trained-model performance estimate.',
            'Mean turn advantage conservation does not imply unbiased policy gradient or return-equivalent RL.'])
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_CPU,(180,180))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=RUNTIME/'arm-turn-bonus-preparation/mixed-alignment-20260926')
    a=parser.parse_args();result=audit(a.output)
    print(json.dumps({k:result[k] for k in ['iterations','excluded','aggregate','resources']},indent=2))
