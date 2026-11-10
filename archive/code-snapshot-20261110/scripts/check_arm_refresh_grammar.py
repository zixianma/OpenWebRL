#!/usr/bin/env python3
"""Exercise the actual XGrammar/Triton CUDA mask kernel before evaluation."""
import json,os
from pathlib import Path

def main():
    job=os.environ.get('SLURM_JOB_ID')
    if not job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Approved GPU allocation required')
    import torch,xgrammar
    torch.set_num_threads(1)
    for dtype in (torch.float32,torch.bfloat16):
        logits=torch.zeros((1,64),device='cuda',dtype=dtype)
        mask=torch.zeros((1,2),device='cuda',dtype=torch.int32);mask[0,0]=1<<7
        xgrammar.apply_token_bitmask_inplace(logits,mask)
        torch.cuda.synchronize()
        assert logits[0,7].item()==0 and torch.isneginf(logits).sum().item()==63
    value=dict(passed=True,job=job,backend='XGrammar Triton CUDA',dtypes=['float32','bfloat16'])
    root=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/arm-refresh-20260924/baseline40')
    (root/'grammar-smoke.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))

if __name__=='__main__':main()
