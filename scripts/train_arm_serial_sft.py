#!/usr/bin/env python3
"""Two-GPU response-segment SFT with durable checkpoints and fixed diagnostics."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

from train_arm_joint_sft import (REPO, digest, encoded, read_rows, write_json,
                                lr_factor, training_order, make_example)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path)
    p.add_argument('--stop-unix',required=True,type=float);p.add_argument('--resume',action='store_true')
    a=p.parse_args();config=json.loads(a.config.read_text());root=Path(config['output']);student=root/'student'
    job=os.getenv('SLURM_JOB_ID')
    if not job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():raise ValueError('Requires authorized Slurm compute')
    for path,checksum in config['code_hashes'].items():
        if digest(REPO/path)!=checksum:raise ValueError(f'Frozen code changed: {path}')
    data=Path(config['data'])
    for name,checksum in config['data_hashes'].items():
        if digest(data/name)!=checksum:raise ValueError('Data hash changed')
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model, PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict,set_peft_model_state_dict
    from train_arm_c2 import language_targets
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);torch.set_num_threads(4)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30));rank=dist.get_rank();world=dist.get_world_size()
    if world!=2:raise ValueError('Expected two workers')
    random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    rows=read_rows(data/f"{config['variant']}.train.jsonl")
    validation=read_rows(data/f"{config['variant']}.validation.jsonl")
    order=training_order(rows,42)
    if len(rows)!=config['train_examples']:raise ValueError('Unexpected training exposure')
    panel_ids=set(config['validation_panel_ids']);panel=[r for r in validation if r['id'] in panel_ids]
    if len(panel)!=len(panel_ids):raise ValueError('Validation panel mismatch')
    processor=AutoProcessor.from_pretrained(config['actor'],local_files_only=True)
    digits=[processor.tokenizer.encode(str(i),add_special_tokens=False) for i in range(1,6)]
    if any(len(v)!=1 for v in digits):raise ValueError('Selection IDs must each be a single token')
    digit_ids=torch.tensor([v[0] for v in digits],device='cuda')
    for row in rows+validation:
        position=row['chosen']['text'].rindex('Selected alternative: ')+len('Selected alternative: ')
        enc=processor.tokenizer(row['chosen']['text'],add_special_tokens=False,return_offsets_mapping=True)
        indices=[i for i,(start,end) in enumerate(enc['offset_mapping']) if start<=position<end]
        if len(indices)!=1 or enc['input_ids']!=row['chosen']['token_ids']:raise ValueError('Selection token alignment')
        row['choice_token']=indices[0]
        if enc['input_ids'][indices[0]]!=digits[row['selected_index']][0]:raise ValueError('Choice label mismatch')
    actor=AutoModelForImageTextToText.from_pretrained(config['actor'],local_files_only=True,
        torch_dtype=torch.bfloat16,device_map={'':f'cuda:{local}'},attn_implementation='sdpa')
    checkpoint=None
    if a.resume:
        checkpoint=Path(json.loads((student/'latest-checkpoint.json').read_text())['path'])
        actor=PeftModel.from_pretrained(actor,checkpoint,is_trainable=True)
    else:
        actor=get_peft_model(actor,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
            target_modules=language_targets(actor),bias='none',task_type='CAUSAL_LM'))
    params=[p for p in actor.parameters() if p.requires_grad]
    actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    actor.enable_input_require_grads();actor.config.use_cache=False

    def values(model,example,row,full_logits=False):
        inputs,prefix,count=example
        positions=torch.arange(prefix-1,prefix+count-1,device='cuda')
        logits=model(**inputs,logits_to_keep=0 if full_logits else positions,use_cache=False).logits
        if full_logits:logits=logits[:,positions]
        if logits.shape[1]!=count:raise ValueError('Response logit alignment')
        labels=inputs['input_ids'][:,prefix:];pieces=[]
        for start in range(0,count,32):
            chunk=logits[:,start:start+32].float();target=labels[:,start:start+32]
            pieces.append(chunk.gather(-1,target.unsqueeze(-1)).squeeze(-1)-chunk.logsumexp(-1))
        lp=torch.cat(pieces,1).squeeze(0)
        segments=torch.tensor(row['chosen']['segments'],device='cuda')
        losses=torch.stack([-lp[segments==k].mean() for k in range(3)])
        weight=torch.tensor([.5,.1,.4],device='cuda');loss=(losses*weight).sum()
        k=len(row['retained_original_indices']);scores=logits[0,row['choice_token'],digit_ids[:k]].float()
        correct=(scores.argmax()==row['selected_index']).float()
        return loss,torch.cat([losses,(-lp.mean()).unsqueeze(0),correct.unsqueeze(0)])

    class Forward(torch.nn.Module):
        def __init__(self,model):super().__init__();self.actor=model
        def forward(self,example,row):return values(self.actor,example,row)
    policy=DDP(Forward(actor),device_ids=[local],broadcast_buffers=False)
    optimizer=torch.optim.AdamW(params,lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    state=dict(updates=0,examples_seen=0,epoch=0,next_position=0,
        dataset_sha256=config['data_hashes'][f"{config['variant']}.train.jsonl"],config_sha256=digest(a.config))
    def rng():return dict(python=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
    def restore_rng(v):random.setstate(v['python']);torch.set_rng_state(v['torch']);torch.cuda.set_rng_state(v['cuda'])
    if checkpoint:
        saved=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
        if saved['state']['config_sha256']!=state['config_sha256']:raise ValueError('Resume config mismatch')
        state=saved['state'];optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng_by_rank'][rank])
    else:
        torch.manual_seed(42+rank);torch.cuda.manual_seed_all(42+rank)
    stopping=False;tracking=None
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def check_stop():
        value=torch.tensor(int(stopping or time.time()>=a.stop_unix),device='cuda');dist.all_reduce(value)
        if value.item():raise InterruptedError('Training deadline')
    def save():
        dest=student/f"update-{state['updates']:06d}";states=[None]*world;dist.all_gather_object(states,rng())
        if rank==0 and not dest.exists():
            tmp=dest.with_name(dest.name+f'.incomplete-{os.getpid()}');tmp.mkdir(parents=True)
            actor.save_pretrained(tmp,safe_serialization=True);processor.save_pretrained(tmp)
            torch.save(dict(optimizer=optimizer.state_dict(),state=dict(state),rng_by_rank=states),tmp/'optimizer.pt')
            write_json(tmp/'progress.json',state)
            from safetensors.torch import load_file
            disk=load_file(str(tmp/'adapter_model.safetensors'));live=get_peft_model_state_dict(actor)
            if set(disk)!=set(live) or any(not torch.equal(disk[k],live[k].detach().cpu()) for k in disk):raise ValueError('Checkpoint serialization differs')
            tmp.rename(dest)
        dist.barrier()
        if rank==0:write_json(student/'latest-checkpoint.json',dict(path=str(dest),**state))
        return dest
    def emit(metrics):
        if rank==0:
            event=dict(utc=datetime.now(timezone.utc).isoformat(),**state,**metrics)
            with (student/'metrics.jsonl').open('a') as f:f.write(encoded(event)+'\n')
            print(encoded(event),flush=True)
            if tracking:tracking.log(event)
    def evaluate(full=False):
        actor.eval();subset=validation if full else panel;results=[]
        with torch.no_grad():
            for i,row in enumerate(subset[rank::world]):
                loss,v=values(actor,make_example(row,processor),row)
                results.append(dict(id=row['id'],source=row['source'],loss=loss.item(),
                    alternatives_ce=v[0].item(),selection_ce=v[1].item(),final_ce=v[2].item(),
                    token_ce=v[3].item(),selection_accuracy=v[4].item()))
        all_results=[None]*world;dist.all_gather_object(all_results,results)
        if rank==0:
            results=sum(all_results,[]);write_json(student/f"validation-{state['updates']:06d}.json",dict(full=full,examples=results))
            emit({'val/'+k:sum(r[k] for r in results)/len(results) for k in ['loss','alternatives_ce','selection_ce','final_ce','token_ce','selection_accuracy']})

    try:
        student.mkdir(parents=True,exist_ok=True)
        if rank==0:
            write_json(student/'training-config.json',config)
            write_json(root/'dataset-audit.json',dict(dataset_sha256=state['dataset_sha256'],retained_turns=len(rows),validation_turns=len(validation)))
            write_json(root/'training-status.json',dict(phase='smoke',allocation=job,**state))
        if not checkpoint:
            # Check the actual longest example before committing to an epoch.
            before={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(actor).items()};initial_rng=rng()
            representative=sorted(rows,key=lambda r:r['prompt_tokens']+r['chosen']['tokens'])
            smoke=[]
            for row in [representative[len(rows)//2],representative[-1]]:
                check_stop();actor.train();optimizer.zero_grad(set_to_none=True)
                start=time.monotonic();loss,v=policy(make_example(row,processor),row);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(loss) or not torch.isfinite(norm) or norm<=0:raise ValueError('Smoke gradient failure')
                smoke.append(dict(id=row['id'],loss=loss.item(),norm=norm.item(),seconds=time.monotonic()-start,
                    peak_gib=torch.cuda.max_memory_allocated()/2**30))
            # Short example full-logits parity, with dropout off.
            actor.eval();example=make_example(representative[0],processor)
            with torch.no_grad():
                fast,_=values(actor,example,representative[0]);full,_=values(actor,example,representative[0],True)
            if not torch.allclose(fast,full,atol=2e-4,rtol=2e-4):raise ValueError('Full/response-only loss mismatch')
            optimizer.zero_grad(set_to_none=True);set_peft_model_state_dict(actor,before);restore_rng(initial_rng)
            del before,example
            torch.cuda.empty_cache()
            if rank==0:write_json(root/'smoke-passed.json',dict(longest_backward=True,checks=smoke,loss_parity_error=abs(fast.item()-full.item()),actual_updates=0))
        if rank==0:
            from dotenv import load_dotenv
            import wandb
            load_dotenv(REPO/'.env',override=False)
            tracking=wandb.init(project='openwebrl-arm',id=config['wandb_id'],resume='allow',
                name='serial-'+config['variant'],group='serial-alternatives-v1',job_type='sft',
                dir=str(root),config=config,settings=wandb.Settings(init_timeout=60))
            write_json(root/'wandb-identity.json',dict(id=config['wandb_id'],url=tracking.url))
        save()
        if state['updates']==0:evaluate()
        for begin in range(state['examples_seen'],len(order),32):
            check_stop();start=time.monotonic();batch=order[begin:begin+32];n=len(batch)
            padded=batch+[batch[-1]]*((-n)%world);local_ids=padded[rank::world]
            actor.train();optimizer.zero_grad(set_to_none=True);metrics=torch.zeros(6,device='cuda')
            for j,index in enumerate(local_ids):
                real=(rank+j*world)<n;row=rows[index]
                with policy.no_sync() if j+1<len(local_ids) else nullcontext():
                    loss,v=policy(make_example(row,processor),row)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    (loss*(world/n if real else 0)).backward()
                if real:metrics+=torch.cat([loss.detach().unsqueeze(0),v.detach()])/n
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite gradient norm')
            seen=begin+n;lr=1e-5*lr_factor(seen,len(rows))
            for g in optimizer.param_groups:g['lr']=lr
            optimizer.step();dist.all_reduce(metrics)
            state.update(updates=state['updates']+1,examples_seen=seen,next_position=seen)
            emit(dict(zip(['train/loss','train/alternatives_ce','train/selection_ce','train/final_ce','train/token_ce','train/selection_accuracy'],metrics.tolist()))|
                {'train/gradient_norm':norm.item(),'train/learning_rate':lr,'train/update_seconds':time.monotonic()-start,
                 'train/peak_gib':torch.cuda.max_memory_allocated()/2**30})
            if rank==0:write_json(root/'training-status.json',dict(phase='training',allocation=job,**state))
            if state['updates'] in config['checkpoint_updates']:
                save();evaluate(full=seen==len(rows))
        endpoint=save()
        if rank==0:
            write_json(student/'complete.json',dict(checkpoint=str(endpoint),**state))
            write_json(root/'training-status.json',dict(phase='complete',allocation=job,**state))
    except InterruptedError as exc:
        checkpoint=save()
        if rank==0:write_json(root/'training-status.json',dict(phase='paused',error=str(exc),checkpoint=str(checkpoint),**state))
        raise SystemExit(2)
    except Exception as exc:
        if rank==0:write_json(root/'training-status.json',dict(phase='failed',error=repr(exc),**state))
        raise
    finally:
        if tracking:tracking.finish()
        dist.destroy_process_group()


if __name__=='__main__':main()
