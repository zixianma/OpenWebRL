#!/usr/bin/env python3
"""One-pass winner SFT on the frozen joint manifest; GPU work requires Slurm.

No model imports occur during --prepare. The batch process owns this trainer
through completion, including fixed-panel validation and checkpoint writes.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction')
DATA = RUNTIME / 'joint-data-v2-20260910'
ACTOR = Path('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
DEFAULT_CONFIG = REPO / 'openwebrl/docs/arm_results/joint_data_v2/sft-config.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024): h.update(block)
    return h.hexdigest()


def encoded(value): return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def read_rows(path):
    with Path(path).open() as stream: return [json.loads(line) for line in stream]


def lr_factor(seen, total, warmup=512):
    if not 0 < warmup < total or not 0 < seen <= total:
        raise ValueError('Invalid exposure-based schedule')
    if seen <= warmup: return seen / warmup
    # Half of a full cosine, so the endpoint is exactly half the peak LR.
    return .5 * (1 + math.cos(math.pi * (seen - warmup) / (2 * (total - warmup))))


def training_order(rows, seed=42):
    """Shuffle within sources and interleave at natural retained proportions."""
    groups = defaultdict(list)
    for i, row in enumerate(rows): groups[row['source']].append(i)
    rng = random.Random(seed)
    for source in sorted(groups): rng.shuffle(groups[source])
    tickets = [( (k + .5) / len(items), source, index)
               for source, items in sorted(groups.items()) for k, index in enumerate(items)]
    return [index for _, _, index in sorted(tickets)]


def prepare(path):
    checks = json.loads((DATA / 'verification.json').read_text())
    if checks['passed'] is not True: raise ValueError('Data verification did not pass')
    rows = read_rows(DATA / 'joint_sft.train.jsonl')
    if len(rows) != 5540 or Counter(r['source'] for r in rows) != {'C2':3464, 'Piotr':2076}:
        raise ValueError('Unexpected reviewed data counts')
    names = ['joint_sft.train.jsonl', 'joint_sft.validation.jsonl',
             'joint_pairs.train.jsonl', 'joint_pairs.validation.jsonl',
             'validation-panel-c2.jsonl', 'validation-panel-piotr.jsonl', 'verification.json']
    model_names = ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                   'preprocessor_config.json', 'chat_template.jinja']
    config = dict(actor=str(ACTOR), data=str(DATA), seed=42, objective='full-response winner SFT',
                  train_examples=5540, validation_examples=658, batch_size=32, epochs=1,
                  total_updates=174, final_batch_size=4, learning_rate=1e-5,
                  warmup_examples=512, endpoint_lr_factor=.5, lora_rank=16, lora_alpha=32,
                  lora_dropout=.05, max_tokens=32768, checkpoint_updates=[44,50,87,131,174],
                  validation_updates=[0,44,87,131,174], wandb_project='openwebrl-arm',
                  wandb_group='arm-joint-v2',
                  data_hashes={name:digest(DATA/name) for name in names},
                  model_metadata_hashes={name:digest(ACTOR/name) for name in model_names},
                  trainer_sha256=digest(__file__),
                  order_sha256=hashlib.sha256(encoded([rows[i]['id'] for i in training_order(rows)]).encode()).hexdigest(),
                  output=str(RUNTIME/'runs/joint-v2-sft'))
    write_json(path, config)
    print(json.dumps(dict(config=str(path), updates=174, train_examples=5540,
                          final_batch_size=4, submitted=False), indent=2))


def validate_allocation():
    job = os.environ.get('SLURM_JOB_ID')
    if not job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('GPU work must run inside its approved Slurm allocation')
    from resume_arm_c2_training import parse_job_record, allocation_deadline
    raw = subprocess.check_output(['scontrol','show','job',job,'-o'], text=True, timeout=20)
    record = parse_job_record(raw)
    if record.get('JobState') != 'RUNNING': raise ValueError('Allocation is not running')
    devices = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'], text=True, timeout=20).strip().splitlines()
    if len(devices) != 1 or 'H200' not in devices[0]:
        raise ValueError('Run this worker in a dedicated one-H200 allocation/step')
    busy = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True, timeout=20).strip()
    if busy: raise ValueError('Assigned GPU is already occupied')
    return job, allocation_deadline(record, margin_minutes=5).timestamp()


def make_example(row, processor, branch='chosen'):
    import torch
    from PIL import Image
    from slime.utils.processing_utils import build_processor_kwargs
    images = []
    for item in row['images']:
        if digest(item['path']) != item['sha256']: raise ValueError('Screenshot hash changed')
        with Image.open(item['path']) as im: images.append(im.convert('RGB'))
    features = processor(text=[row['prompt']], **build_processor_kwargs({'images':images}))
    prefix = features['input_ids'][0]
    if hasattr(prefix, 'tolist'): prefix = prefix.tolist()
    if len(prefix) != row['prompt_tokens'] or hashlib.sha256(encoded(prefix).encode()).hexdigest() != row['prompt_token_sha256']:
        raise ValueError(f"Expanded prompt differs: {row['id']}")
    if features['image_grid_thw'].tolist() != row['image_grid_thw']: raise ValueError('Image grid mismatch')
    target = row[branch]['token_ids']
    if processor.tokenizer.decode(target, skip_special_tokens=False) != row[branch]['text']:
        raise ValueError('Response token/text mismatch')
    if len(prefix) + len(target) > 32768: raise ValueError('Context overflow')
    ids = torch.tensor([prefix + target], device='cuda', dtype=torch.long)
    tensors = {k:v.to('cuda') for k,v in features.items()
               if k not in ('input_ids','attention_mask') and isinstance(v,torch.Tensor)}
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), **tensors), len(prefix), len(target)


def token_logps(model, example, full_logits=False):
    """Causal response positions only; log-softmax arithmetic is FP32."""
    import torch
    inputs, prefix, count = example
    positions = torch.arange(prefix-1, prefix+count-1, device='cuda')
    output = model(**inputs, logits_to_keep=0 if full_logits else positions, use_cache=False)
    logits = output.logits[:, positions] if full_logits else output.logits
    if logits.shape[1] != count: raise ValueError('Prediction/target alignment failure')
    targets = inputs['input_ids'][:,prefix:]
    parts = []
    for start in range(0,count,32):
        chunk = logits[:,start:start+32].float()
        labels = targets[:,start:start+32]
        parts.append(chunk.gather(-1,labels.unsqueeze(-1)).squeeze(-1) - chunk.logsumexp(-1))
    return torch.cat(parts,dim=1).squeeze(0)


def run(args):
    job, deadline = validate_allocation()
    config = json.loads(args.config.read_text())
    if digest(__file__) != config['trainer_sha256']: raise ValueError('Trainer changed; regenerate reviewed config')
    for name, checksum in config['data_hashes'].items():
        if digest(Path(config['data'])/name) != checksum: raise ValueError(f'Data changed: {name}')
    for name, checksum in config['model_metadata_hashes'].items():
        if digest(Path(config['actor'])/name) != checksum: raise ValueError(f'Model metadata changed: {name}')
    root = Path(config['output']); root.mkdir(parents=True,exist_ok=True)
    lock = (root/'trainer.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'complete.json').exists(): raise ValueError('This run is complete')
    if (root/'latest-checkpoint.json').exists() and not args.resume:
        raise ValueError('Use --resume to continue the existing run')
    write_json(root/'status.json',dict(phase='loading',allocation=job,pid=os.getpid()))
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model, PeftModel
    from train_arm_c2 import language_targets
    torch.set_num_threads(4)
    random.seed(config['seed']); torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    rows = read_rows(Path(config['data'])/'joint_sft.train.jsonl')
    panels = sum([read_rows(Path(config['data'])/f'validation-panel-{source}.jsonl') for source in ['c2','piotr']],[])
    order = training_order(rows,config['seed'])
    if hashlib.sha256(encoded([rows[i]['id'] for i in order]).encode()).hexdigest() != config['order_sha256']:
        raise ValueError('Training order changed')
    processor = AutoProcessor.from_pretrained(config['actor'],local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(config['actor'],local_files_only=True,
                torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
    if args.resume:
        checkpoint = Path(json.loads((root/'latest-checkpoint.json').read_text())['path'])
        if checkpoint.parent != root: raise ValueError('Checkpoint outside this run')
        model = PeftModel.from_pretrained(model,checkpoint,is_trainable=True)
    else:
        model = get_peft_model(model,LoraConfig(r=config['lora_rank'],lora_alpha=config['lora_alpha'],
                  lora_dropout=config['lora_dropout'],target_modules=language_targets(model),bias='none',task_type='CAUSAL_LM'))
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not named or any('lora_' not in n or 'language_model.layers.' not in n for n,_ in named):
        raise ValueError('Unexpected trainable weights')
    params = [p for _,p in named]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads(); model.config.use_cache=False
    optimizer = torch.optim.AdamW(params,lr=config['learning_rate'],betas=(.9,.95),eps=1e-8,weight_decay=.01)
    config_hash = digest(args.config)
    state = dict(updates=0,examples_seen=0,config_sha256=config_hash)
    if args.resume:
        saved = torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
        state = saved['state']
        if state['config_sha256'] != config_hash: raise ValueError('Resume recipe differs')
        optimizer.load_state_dict(saved['optimizer'])
        random.setstate(saved['python_rng']); torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    write_json(root/'training-config.json',config)
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    def should_stop(): return stopping or time.time() >= deadline
    def save():
        dest = root/f"update-{state['updates']:06d}"
        if dest.exists(): return dest
        tmp = dest.with_name(dest.name+f'.incomplete-{os.getpid()}'); tmp.mkdir()
        model.save_pretrained(tmp,safe_serialization=True); processor.save_pretrained(tmp)
        torch.save(dict(optimizer=optimizer.state_dict(),state=dict(state),python_rng=random.getstate(),
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp/'optimizer.pt')
        write_json(tmp/'progress.json',state)
        # Verify serialized adapter bytes before publishing the checkpoint pointer.
        from safetensors.torch import load_file
        from peft.utils.save_and_load import get_peft_model_state_dict
        weights = load_file(str(tmp/'adapter_model.safetensors'))
        live = get_peft_model_state_dict(model)
        if set(weights)!=set(live) or any(not torch.equal(weights[k],live[k].detach().cpu()) for k in weights):
            raise ValueError('Checkpoint adapter save mismatch')
        tmp.rename(dest); write_json(root/'latest-checkpoint.json',dict(path=str(dest),**state))
        return dest
    # Real-image / causal alignment gate, before any parameter updates.
    if not args.resume:
        model.eval()
        smoke_rows = [min((r for r in rows if r['source']==source),key=lambda r:r['prompt_tokens']) for source in ['C2','Piotr']]
        for row in smoke_rows:
            # Invoke the native parser directly: do not accept silent regex fallback.
            from openwebrl.base.utils import ToolParser
            tool_block = re.search(r'<tools>\n(.*?)\n</tools>',row['prompt'],re.S).group(1)
            tool_parser = ToolParser([json.loads(line) for line in tool_block.splitlines() if line.strip()])
            native = tool_parser._get_sglang_parser()
            if native is None: raise ValueError('Native SGLang parser unavailable')
            _, calls = native.parse_non_stream(row['chosen']['text'])
            actions=[]
            for call in calls:
                value=call.model_dump(); parameters=value['parameters']
                if isinstance(parameters,str): parameters=json.loads(parameters)
                actions.append(dict(name=value['name'],arguments=parameters))
            if actions != row['chosen']['actions']: raise ValueError('Native parser/action mismatch')
            ex = make_example(row,processor)
            with torch.no_grad():
                small = token_logps(model,ex); full = token_logps(model,ex,full_logits=True)
                if not torch.allclose(small,full,atol=.02,rtol=.002): raise ValueError('Selected-logit/full-logit mismatch')
            loss = -token_logps(model,ex).mean(); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0: raise ValueError('Smoke loss/gradient failure')
            optimizer.zero_grad(set_to_none=True)
        longest=max(rows,key=lambda r:r['prompt_tokens']+r['chosen']['tokens'])
        long_loss=-token_logps(model,make_example(longest,processor)).mean()
        long_loss.backward()
        long_norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(long_loss) or not torch.isfinite(long_norm):
            raise ValueError('Longest-sequence memory/numerics gate failed')
        optimizer.zero_grad(set_to_none=True)
        # Rehearsal performed no optimizer updates. Restore the seeded RNG.
        random.seed(config['seed']); torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
        write_json(root/'smoke-passed.json',dict(sources=['C2','Piotr'],pixel_processor=True,
                   native_parser=True,longest_training_sequence_checked=True,
                   causal_logits_checked=True,finite_nonzero_gradients=True,optimizer_updates=0))
    from dotenv import load_dotenv
    import wandb
    load_dotenv(REPO/'.env',override=False)
    identity_path = root/'wandb-identity.json'
    if identity_path.exists(): identity = json.loads(identity_path.read_text())
    else:
        identity = dict(id=wandb.util.generate_id(),project=config['wandb_project'])
        write_json(identity_path,identity)
    tracking = wandb.init(project=identity['project'],id=identity['id'],resume='allow',
                         name='joint-v2-sft',group=config['wandb_group'],job_type='sft',
                         dir=str(root),config=config,settings=wandb.Settings(init_timeout=60))
    write_json(identity_path,dict(identity,url=tracking.url))
    def emit(event):
        event = dict(utc=datetime.now(timezone.utc).isoformat(),updates=state['updates'],examples_seen=state['examples_seen'],**event)
        with (root/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
        tracking.log(event)
    def evaluate():
        path = root/f"validation-{state['updates']:06d}.json"
        if path.exists(): return
        model.eval(); results=[]
        with torch.no_grad():
            for row in panels:
                if should_stop(): raise InterruptedError('Pause during fixed-panel validation')
                values={}
                for branch in ['chosen','rejected']:
                    lp=token_logps(model,make_example(row,processor,branch))
                    values[branch]=dict(logp_sum=lp.sum().item(),ce=-lp.mean().item(),
                         action_logp_sum=lp[row[branch]['action_token_indices']].sum().item())
                results.append(dict(id=row['id'],source=row['source'],task_group=row['task_group'],**values))
        write_json(path,dict(updates=state['updates'],examples=results))
        metrics={}
        for source in ['C2','Piotr']:
            subset=[r for r in results if r['source']==source]
            metrics[f'val/{source}/winner_ce']=sum(r['chosen']['ce'] for r in subset)/len(subset)
            for key in ['logp_sum','action_logp_sum']:
                metrics[f'val/{source}/{key}_ranking']=sum((r['chosen'][key]>r['rejected'][key])+.5*(r['chosen'][key]==r['rejected'][key]) for r in subset)/len(subset)
        emit(metrics)
        model.train()
    try:
        save()
        if state['updates'] in config['validation_updates']: evaluate()
        model.train()
        for begin in range(state['examples_seen'],len(order),config['batch_size']):
            if should_stop(): raise InterruptedError('Allocation deadline or stop request')
            batch=order[begin:begin+config['batch_size']]; optimizer.zero_grad(set_to_none=True)
            losses=defaultdict(list); started=time.monotonic()
            for index in batch:
                row=rows[index]; loss=-token_logps(model,make_example(row,processor)).mean()
                if not torch.isfinite(loss): raise ValueError('Nonfinite SFT loss')
                (loss/len(batch)).backward(); losses[row['source']].append(loss.item())
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm): raise ValueError('Nonfinite gradient')
            seen=begin+len(batch)
            lr=config['learning_rate']*lr_factor(seen,len(order),config['warmup_examples'])
            for group in optimizer.param_groups: group['lr']=lr
            optimizer.step(); state.update(updates=state['updates']+1,examples_seen=seen)
            event={'train/cross_entropy':sum(map(sum,losses.values()))/len(batch),
                   'train/gradient_norm':norm.item(),'train/learning_rate':lr,
                   'train/batch_size':len(batch),'train/update_seconds':time.monotonic()-started}
            event.update({f'train/{source}/cross_entropy':sum(v)/len(v) for source,v in losses.items()})
            emit(event); write_json(root/'status.json',dict(phase='training',allocation=job,**state))
            if state['updates'] in config['checkpoint_updates']: save()
            if state['updates'] in config['validation_updates']: evaluate()
        checkpoint=save()
        write_json(root/'complete.json',dict(checkpoint=str(checkpoint),**state))
        write_json(root/'status.json',dict(phase='complete',allocation=job,**state))
    except InterruptedError as exc:
        checkpoint=save()
        write_json(root/'status.json',dict(phase='paused',reason=str(exc),checkpoint=str(checkpoint),**state))
    except Exception as exc:
        write_json(root/'status.json',dict(phase='failed',error=repr(exc),**state))
        raise
    finally:
        tracking.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare(args.config)
    else: run(args)


if __name__=='__main__': main()
