"""Two-GPU NCCL collectives and distributed backward on allocated devices."""
import json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl')
model=DistributedDataParallel(torch.nn.Linear(256,256).cuda(),device_ids=[rank])
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
for _ in range(3):
    x=torch.randn(32,256,device='cuda')
    loss=model(x).square().mean()
    loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
weight=model.module.weight.detach()
reference=weight.clone();dist.broadcast(reference,src=0)
assert torch.equal(weight,reference)
signal=torch.tensor(float(rank+1),device='cuda');dist.all_reduce(signal)
assert signal.item()==3
print(json.dumps({'rank':rank,'gpu':torch.cuda.get_device_name(),'all_reduce':signal.item(),'ddp_weights_equal':True,'loss':loss.item()}),flush=True)
dist.destroy_process_group()
