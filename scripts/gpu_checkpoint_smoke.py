"""Two-GPU streaming-checkpoint round trip; set OWRL_CHECKPOINT_PROBE_DIR to a new directory."""
import os
from pathlib import Path
import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedObject
from slime.backends.megatron_utils.streaming_checkpoint import StreamingTorchDistSaveStrategy

rank=int(os.environ['RANK']); torch.cuda.set_device(rank)
dist.init_process_group('nccl')
p=Path(os.environ['OWRL_CHECKPOINT_PROBE_DIR'])
if rank == 0:
    p.mkdir(parents=True, exist_ok=False)
dist.barrier()
x=torch.arange(128*16,device='cuda',dtype=torch.float32).reshape(128,16)+rank*10000
y=x.to(torch.bfloat16)
def state(a,b):
 return {'model':ShardedTensor.from_rank_offsets('model',a,(0,rank,2)),
         'optim':ShardedTensor.from_rank_offsets('optim',b,(0,rank,2)),
         'rank_state':ShardedObject('rank_state',{'rank':rank},(2,),(rank,)),
         'step':14}
dist_checkpointing.save(state(x,y),p,sharded_strategy=StreamingTorchDistSaveStrategy())
result=dist_checkpointing.load(state(torch.zeros_like(x),torch.zeros_like(y)),p)
assert torch.equal(result['model'],x)
assert torch.equal(result['optim'],y)
assert result['rank_state']=={'rank':rank} and result['step']==14
if rank==0: print('PASS: two-GPU streaming torch_dist save loaded by standard Megatron loader; model, optimizer-like BF16 shards, rank state, and step equal.',flush=True)
dist.destroy_process_group()
