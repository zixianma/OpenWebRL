"""Side-buffer all-failure supervision; never consumes native accepted-group slots."""
import hashlib
import json
import math
from pathlib import Path


def priority(policy_id, task_id, group_id):
    return hashlib.sha256(f'{policy_id}:{task_id}:{group_id}'.encode()).hexdigest()


def retain(current, samples):
    """Uniform hash-priority reservoir, independent of label sign and arrival order."""
    rows=[s for t in samples for s in (t if isinstance(t,list) else [t])]
    config=current['config'];cap=config['failure_group_cap']
    if not isinstance(cap,int) or not 0 <= cap <= 8:raise ValueError('Failure group cap must be <=8')
    key=priority(config['policy_id'],rows[0].metadata['task_id'],rows[0].group_index)
    pool=current.setdefault('failure_additive_pool',{})
    pool[key]=samples
    for remove in sorted(pool)[cap:]:pool.pop(remove)
    current['failure_additive_eligible']=current.get('failure_additive_eligible',0)+1


def filter_groups(args,samples,**kwargs):
    from openwebrl.arm_turn_bonus import state
    from openwebrl.arm_failure_bonus import filter_groups as failure_filter
    from slime.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonempty_nonzero_std
    current=state(args)
    if not current['config'].get('additive_failure_groups'):raise ValueError('Additive configuration required')
    native=check_reward_nonempty_nonzero_std(args,samples,**kwargs)
    if native.keep or native.reason!='zero_std_0.0':return native
    # Reuse strict validity and current-policy label validation. It marks only
    # rejected rows; return native rejection so the ordinary quota stays 48.
    eligible=failure_filter(args,samples,**kwargs)
    if eligible.keep:retain(current,samples)
    return native


def window_schedule(records,total_rows,counts,gbs,coefficient):
    """Each nonzero row once per epoch; preserve the zero-signal denominator.

    Averaged over U optimizer windows, the auxiliary objective is
    coefficient / total_rows * sum(nonzero-row clipped PPO losses).
    Native loss rescales by 1/gbs; compensate without changing outcome scales.
    """
    if not math.isfinite(coefficient) or not 0 <= coefficient <= 1/6:raise ValueError('Invalid failure coefficient')
    if total_rows < len(records) or total_rows > 600:raise ValueError('Invalid failure population')
    if not counts or any(c!=gbs for c in counts):raise ValueError('Require complete DP1 microbatch1 outcome windows')
    if not records:return [[] for _ in counts]
    if total_rows<=0:raise ValueError('Missing failure denominator')
    result=[[] for _ in counts]
    scale=coefficient*gbs*len(counts)/total_rows
    # Manifest order is randomized by identity, never sorted by bonus sign.
    for i in range(len(records)):result[i%len(counts)].append((i,scale))
    if any(len(x)>32 for x in result):raise ValueError('More than 32 auxiliary turns per outcome update')
    return result


def write_auxiliary(args,samples,current,report):
    import torch
    from openwebrl.arm_turn_bonus import unit_bonus,write_json
    config=current['config']
    if len({s.group_index for s in samples})!=48:raise ValueError('Original 48-group quota changed')
    pool=current.get('failure_additive_pool',{})
    rows=[s for key in sorted(pool) for t in pool[key] for s in (t if isinstance(t,list) else [t])]
    if len(rows)>600:raise ValueError('Failure turn population exceeds 8*5*15')
    items=[]
    for s in rows:
        unit=unit_bonus(s,config['policy_id'])
        if not unit:continue
        n=s.response_length;mask=s.loss_mask if s.loss_mask is not None else [1]*n
        if not 0<n<len(s.tokens) or len(mask)!=n or not any(mask):raise ValueError('Bad auxiliary response')
        items.append((priority(config['policy_id'],s.metadata['trajectory_id'],s.index),s,unit,mask))
    records=[];payload=[]
    for _,s,unit,mask in sorted(items,key=lambda x:x[0]):
        records.append(dict(sample_index=s.index,group_index=s.group_index,advantage=.5*unit))
        # File-backed image tensors stay lossless; torch.save serializes their
        # values into a durable, bounded artifact. No cross-node /tmp references.
        payload.append(dict(tokens=torch.tensor(s.tokens,dtype=torch.long),response_length=s.response_length,
            loss_mask=torch.tensor(mask,dtype=torch.int),multimodal_train_inputs=s.multimodal_train_inputs))
    root=Path(config['output']);path=root/'failure_auxiliary.pt';temporary=path.with_suffix('.partial')
    torch.save(payload,temporary);temporary.replace(path)
    if path.stat().st_size>2*1024**3:raise ValueError('Auxiliary payload exceeds 2GiB')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    manifest=dict(schema_version=1,rollout_id=config['rollout_id'],policy_id=config['policy_id'],
        checkpoint=config['checkpoint'],training_judge='gpt-4.1/action_history',records=records,
        tensor_file=path.name,tensor_sha256=h.hexdigest(),total_failure_rows=len(rows),
        failure_groups=len(pool),eligible_groups=current.get('failure_additive_eligible',0),
        coefficient=config['failure_loss_coefficient'],beta=.5,q=.2,mixed_groups=48,
        expected_windows=len(samples)//args.global_batch_size)
    window_schedule(records,len(rows),[args.global_batch_size]*manifest['expected_windows'],args.global_batch_size,manifest['coefficient'])
    write_json(root/'failure_auxiliary.json',manifest)
    report.update(additive_failure_groups=len(pool),additive_failure_rows=len(rows),
        additive_failure_labels=len(records),additive_failure_positive=sum(x['advantage']>0 for x in records),
        additive_failure_negative=sum(x['advantage']<0 for x in records),
        additive_failure_coefficient=manifest['coefficient'],outcome_groups_preserved=48)
    write_json(root/'calibration.json',report)
