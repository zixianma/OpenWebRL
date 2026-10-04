"""Isolated full-batch validation and zero-loss long-context memory probe."""
import hashlib
import json
import os
from pathlib import Path

from openwebrl.arm_tpdp_diagnostic import (
    before_collection, before_training, before_optimizer, after_checkpoint,
    reset_collection, post_process_rewards, measure_step, require_diagnostic,
)


def trim_train_data(data):
    """Keep every saved outcome row and its original normalization."""
    require_diagnostic()
    lengths=list(map(len,data['tokens']))
    value=dict(rows=len(lengths),tokens=sum(lengths),max_context=max(lengths),
        sha256=hashlib.sha256(json.dumps({k:data[k] for k in
            ('tokens','response_lengths','loss_masks','rewards','sample_indices')},sort_keys=True).encode()).hexdigest(),
        normalization='unchanged complete saved outcome groups',stress_context=32768)
    (Path(os.environ['OPENWEBRL_TPDP_OUTPUT'])/'batch.json').write_text(json.dumps(value)+'\n')
    return data


def add_stress_padding(mixed, counts, auxiliary):
    """One zero-loss 32K row per DP rank, only in the first optimizer window.

    Preserve the real row's screenshots, response and mask; add text before its
    response. Native microbatch scaling cancels the extra accumulation pass.
    This function is injected into validation sources only.
    """
    require_diagnostic()
    import torch
    if not counts or not auxiliary['tokens']:
        raise ValueError('Stress probe requires a populated auxiliary batch')
    i=max(range(len(auxiliary['tokens'])),key=lambda j:len(auxiliary['tokens'][j]))
    tokens=auxiliary['tokens'][i]; n=auxiliary['response_lengths'][i]
    padding=32768-len(tokens)
    if padding<0:raise ValueError('Saved row exceeds the context bound')
    extended=torch.cat((tokens[:-n],tokens.new_zeros(padding),tokens[-n:]))
    at=counts[0]
    for key,value in mixed.items():
        if not isinstance(value,list):continue
        source=auxiliary.get(key)
        item=source[i] if source is not None else None
        if key=='tokens':item=extended
        elif key=='total_lengths':item=32768
        elif key=='arm_source':item='arm_padding'
        elif key in ('arm_scale','arm_advantage'):item=0.
        value.insert(at,item)
    counts[0]+=1
    return mixed,counts
