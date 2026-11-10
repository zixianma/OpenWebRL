#!/usr/bin/env python3
"""Frozen-versus-refreshed SelectionARM comparison; fresh ARM LoRA, fixed data.

--prepare is CPU-only. --run requires the exact approved allocation and labels.
The process owns frozen evaluation, training, adapted evaluation and reporting.
"""
import argparse
from collections import defaultdict
import contextlib
import json
import math
import os
from pathlib import Path
import random
import time

from prepare_arm_forward_transfer import BASE, MODEL, CONTROL, REPO, SOURCE, rows, sha, stable, write
from openwebrl.arm_inference import load_selection_builder, selection_messages, selection_schema, parse_selection


def prepare():
    data=CONTROL/'data-v2';teacher=CONTROL/'teacher-v1'
    manifest=json.loads((data/'manifest.json').read_text())
    if not manifest['passed']:raise ValueError('Data gate failed')
    config=dict(data=str(data),teacher=str(teacher),model=str(MODEL),processor=str(BASE),
                output=str(CONTROL/'run-v1'),initialization='released merged SelectionARM; fresh ARM LoRA and optimizer',
                actor_training=False,seed=42,train_examples=2858,epochs=1,batch_size=32,microbatch_size=1,
                total_updates=90,final_batch_size=10,learning_rate=1e-5,warmup_updates=5,
                scheduler='cosine to 10% of peak',weight_decay=.01,adam_betas=[.9,.95],
                lora_rank=32,lora_alpha=64,lora_dropout=.05,gradient_clip=1.,max_tokens=8192,
                min_pixels=65536,max_pixels=262144,training_project='openwebrl',evaluation_project='openwebrl-evals',
                objective='mean completion-token cross entropy on teacher selection JSON and end-of-turn token',
                data_hashes=manifest['data_hashes'],teacher_manifest_sha256=sha((teacher/'manifest.json').read_bytes()),
                model_metadata_hashes={n:sha((MODEL/n).read_bytes()) for n in ['config.json','tokenizer.json','chat_template.jinja']},
                script_sha256=sha(Path(__file__).read_bytes()),
                resources=dict(gpus=1,gpu_type='H200',hours=4,cpus=8,memory_gib=120),submitted=False)
    write(CONTROL/'training-config.json',config);print(json.dumps(config,indent=2))


def label_map(data,teacher):
    status=json.loads((teacher/'label-status.json').read_text())
    if not status['complete'] or sha((teacher/'labels.jsonl').read_bytes())!=status['labels_sha256']:
        raise ValueError('Teacher labeling is incomplete or changed')
    labels={(r['state_id'],r['treatment']):r for r in rows(teacher/'labels.jsonl')}
    result={}
    for split in ['train','dev','future','replay','retention']:
        for row in rows(data/f'{split}.jsonl'):
            result[row['id']]=row.get('teacher_selection_original') if split in ['replay','retention'] else labels[(row['id'],'main')]['selected_original']
    return result,labels


def target_text(row,winner):
    if winner not in range(5) or sorted(row['permutation'])!=list(range(5)):
        raise ValueError('Invalid candidate identity or permutation')
    return json.dumps({'selection':row['permutation'].index(winner)+1})+'<|im_end|>'


def paired_report(records):
    """Task-cluster bootstrap of the paired, state-weighted accuracy difference."""
    by=defaultdict(dict)
    for r in records:by[r['pass_id']][r['model']]=r
    grouped=defaultdict(list)
    for versions in by.values():
        if set(versions)!={'frozen','adapted'}:raise ValueError('Missing paired prediction')
        a,b=versions['frozen'],versions['adapted']
        if (a['task_group'],a['teacher_original'])!=(b['task_group'],b['teacher_original']):raise ValueError('Pair mismatch')
        grouped[(a['split'],a['treatment'])].append((a,b))
    result={}
    for key,pairs in grouped.items():
        tasks=defaultdict(list)
        for a,b in pairs:tasks[a['task_group']].append(float(b['action_correct'])-float(a['action_correct']))
        clusters=list(tasks.values());rng=random.Random(42);samples=[]
        for _ in range(10000):
            chosen=[rng.choice(clusters) for _ in clusters]
            samples.append(sum(map(sum,chosen))/sum(map(len,chosen)))
        samples.sort();n=len(pairs)
        result['/'.join(key)]=dict(states=n,tasks=len(tasks),
            frozen_action_accuracy=sum(a['action_correct'] for a,b in pairs)/n,
            adapted_action_accuracy=sum(b['action_correct'] for a,b in pairs)/n,
            frozen_index_accuracy=sum(a['index_correct'] for a,b in pairs)/n,
            adapted_index_accuracy=sum(b['index_correct'] for a,b in pairs)/n,
            delta=sum(b['action_correct']-a['action_correct'] for a,b in pairs)/n,
            delta_task_bootstrap_95ci=[samples[249],samples[9749]],
            frozen_parse_failures=sum(a['selected_original'] is None for a,b in pairs),
            adapted_parse_failures=sum(b['selected_original'] is None for a,b in pairs))
    return result


def run():
    if (CONTROL/'active-proposal.json').exists():
        proposal=json.loads((CONTROL/'active-proposal.json').read_text())
        if proposal.get('superseded_gpu_trainer')=='scripts/run_arm_forward_transfer.py':
            raise ValueError('Superseded: use the baseline40 LLaMA-Factory recipe in action-reward-models-refresh')
    config=json.loads((CONTROL/'training-config.json').read_text())
    approval=json.loads((CONTROL/'approval.json').read_text())
    if approval.get('gpu_approved') is not True or approval.get('gpu_resources')!=config['resources']:
        raise ValueError('Exact GPU request is not approved')
    from train_arm_joint_sft import validate_allocation, token_logps
    job,deadline=validate_allocation()
    if str(approval.get('gpu_job_id'))!=job:raise ValueError('GPU allocation identity differs')
    data=Path(config['data']);teacher=Path(config['teacher']);out=Path(config['output']);out.mkdir(exist_ok=True)
    if sha(Path(__file__).read_bytes())!=config['script_sha256']:raise ValueError('Trainer changed')
    for n,digest in config['data_hashes'].items():
        if sha((data/n).read_bytes())!=digest:raise ValueError('Data changed')
    for n,digest in config['model_metadata_hashes'].items():
        if sha((MODEL/n).read_bytes())!=digest:raise ValueError('Model metadata changed')
    if sha((teacher/'manifest.json').read_bytes())!=config['teacher_manifest_sha256']:raise ValueError('Teacher requests changed')
    winners,teacher_labels=label_map(data,teacher)
    sets={s:list(rows(data/f'{s}.jsonl')) for s in ['train','dev','future','replay','retention']}
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    from tokenizers import Tokenizer
    from PIL import Image
    from peft import LoraConfig,get_peft_model,PeftModel
    from train_arm_c2 import language_targets
    import xgrammar as xgr
    from xgrammar.contrib.hf import LogitsProcessor
    import wandb
    from dotenv import load_dotenv
    load_dotenv(REPO/'.env',override=False)
    torch.set_num_threads(4);torch.manual_seed(42);random.seed(42)
    cfg=AutoConfig.from_pretrained(BASE,local_files_only=True)
    released=json.loads((MODEL/'config.json').read_text())
    for k in ['hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','vocab_size']:
        if released['text_config'][k]!=getattr(cfg.text_config,k):raise ValueError('Model architecture differs')
    processor=AutoProcessor.from_pretrained(BASE,local_files_only=True,min_pixels=65536,max_pixels=262144)
    if Tokenizer.from_file(str(MODEL/'tokenizer.json')).get_vocab()!=processor.tokenizer.get_vocab():raise ValueError('Vocabulary differs')
    if (MODEL/'chat_template.jinja').read_bytes()!=(BASE/'chat_template.jinja').read_bytes():raise ValueError('Template differs')
    model=AutoModelForImageTextToText.from_pretrained(MODEL,config=cfg,torch_dtype=torch.bfloat16,
            device_map='cuda:0',attn_implementation='sdpa',local_files_only=True)
    resume=json.loads((out/'latest.json').read_text()) if (out/'latest.json').exists() else None
    if resume:model=PeftModel.from_pretrained(model,resume['path'],is_trainable=True)
    else:model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,lora_dropout=.05,target_modules=language_targets(model),task_type='CAUSAL_LM',bias='none'))
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not named or any('lora_' not in n or 'language_model.layers.' not in n for n,_ in named):raise ValueError('Non-language weights trainable')
    params=[p for _,p in named]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.config.use_cache=False
    optimizer=torch.optim.AdamW(params,lr=1e-5,betas=(.9,.95),weight_decay=.01)
    completed=0
    if resume:
        saved=torch.load(Path(resume['path'])/'optimizer.pt',map_location='cpu',weights_only=False)
        if saved['config_sha256']!=sha((CONTROL/'training-config.json').read_bytes()):raise ValueError('Resume configuration changed')
        optimizer.load_state_dict(saved['optimizer']);completed=saved['updates'];torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    builder=load_selection_builder(SOURCE)
    grammar=xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(processor.tokenizer,
        vocab_size=cfg.text_config.vocab_size)).compile_json_schema(selection_schema(5))
    def features(row,permutation=None):
        raw=Path(row['screenshot']).read_bytes()
        if sha(raw)!=row['screenshot_sha256']:raise ValueError('Screenshot changed')
        text=row['prompt']
        if permutation is not None:
            m=selection_messages(builder,row['task'],row['url'],row['history'],[row['candidates'][i] for i in permutation],raw)
            m[1]['content'][1]={'type':'image'};text=processor.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
        with Image.open(row['screenshot']) as im:f=processor(text=[text],images=[im.convert('RGB')],return_tensors='pt')
        ids=f.input_ids[0].tolist()
        if permutation is None and (sha(json.dumps(ids,separators=(',',':')).encode())!=row['prompt_token_sha256'] or f.image_grid_thw.tolist()!=row['image_grid_thw']):raise ValueError('Expanded prompt changed')
        if len(ids)+32>8192:raise ValueError('Context overflow; truncation forbidden')
        return f.to('cuda')
    def example(row):
        f=features(row);prefix=f.input_ids.shape[-1]
        ids=processor.tokenizer.encode(target_text(row,winners[row['id']]),add_special_tokens=False)
        f['input_ids']=torch.cat([f.input_ids,torch.tensor([ids],device='cuda')],dim=1)
        f['attention_mask']=torch.ones_like(f.input_ids)
        return dict(f),prefix,len(ids)
    def tracker(project,name):
        return wandb.init(project=project,group='arm-forward-transfer',name=name,id='arm-refresh-'+name,
                          resume='allow',dir=str(out),config=config,settings=wandb.Settings(init_timeout=45))
    passes=[(r,'main',r['permutation']) for s in ['dev','future','retention'] for r in sets[s]]
    reverse=sorted(sets['future'],key=lambda r:stable('order:'+r['id']))[:100]
    passes.extend((r,'reverse',list(reversed(r['permutation']))) for r in reverse)
    def evaluate(name):
        path=out/f'{name}.jsonl';done={r['pass_id'] for r in rows(path)} if path.exists() else set()
        tracking=tracker('openwebrl-evals',name);model.eval()
        with model.disable_adapter() if name=='frozen' else contextlib.nullcontext():
            for row,treatment,permutation in passes:
                ident=row['id']+':'+treatment
                if ident in done:continue
                if time.time()>deadline:raise TimeoutError('Allocation budget reached; predictions preserved')
                f=features(row,None if treatment=='main' else permutation)
                with torch.inference_mode():g=model.generate(**f,do_sample=False,max_new_tokens=32,logits_processor=[LogitsProcessor(grammar)])
                text=processor.tokenizer.decode(g[0,f.input_ids.shape[-1]:],skip_special_tokens=True)
                try:selected=permutation[parse_selection(text,5)]
                except ValueError:selected=None
                winner=winners[row['id']]
                # Secondary permutation comparisons hold the main teacher label fixed.
                record=dict(pass_id=ident,state_id=row['id'],task_group=row['task_group'],split=row['split'],treatment=treatment,
                            model=name,teacher_original=winner,selected_original=selected,
                            action_correct=selected is not None and row['classes'][selected]==row['classes'][winner],
                            index_correct=selected==winner,raw=text,distinct_actions=len(set(row['classes'])),
                            rollout_reward=row.get('reward'))
                with path.open('a') as f:f.write(json.dumps(record)+'\n')
                done.add(ident);tracking.log({'eval/completed':len(done)})
                if len(done)%50==0:print(json.dumps(dict(phase=name,completed=len(done))),flush=True)
        if len(done)!=len(passes):raise ValueError('Evaluation incomplete')
        tracking.finish()
    train=sets['train']+sets['replay']
    # Interleave shuffled sources at 70/30; each example occurs exactly once.
    ordered=[]
    for source in [sets['train'],sets['replay']]:
        source=sorted(source,key=lambda r:stable('training:'+r['id']))
        ordered.extend(((i+.5)/len(source),r['id'],r) for i,r in enumerate(source))
    train=[r for _,_,r in sorted(ordered)]
    def save(update):
        dest=out/f'update-{update:04d}'
        if dest.exists():return
        tmp=out/f'update-{update:04d}.incomplete';tmp.mkdir(exist_ok=False)
        model.save_pretrained(tmp,safe_serialization=True);processor.save_pretrained(tmp)
        from safetensors.torch import load_file
        from peft.utils.save_and_load import get_peft_model_state_dict
        disk=load_file(str(tmp/'adapter_model.safetensors'));live=get_peft_model_state_dict(model)
        if set(disk)!=set(live) or any(not torch.equal(v,live[k].detach().cpu()) for k,v in disk.items()):raise ValueError('Adapter checkpoint mismatch')
        torch.save(dict(optimizer=optimizer.state_dict(),updates=update,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                        config_sha256=sha((CONTROL/'training-config.json').read_bytes())),tmp/'optimizer.pt')
        tmp.rename(dest);write(out/'latest.json',dict(path=str(dest),updates=update))
    model.eval();saved_cpu_rng=torch.get_rng_state();saved_cuda_rng=torch.cuda.get_rng_state_all()
    # Full versus reduced causal logits must agree on a short real example.
    short=example(min(train,key=lambda r:r['prompt_tokens']))
    with torch.no_grad():
        if not torch.allclose(token_logps(model,short),token_logps(model,short,full_logits=True),atol=1e-4,rtol=1e-4):raise ValueError('Causal target alignment failed')
    # Exercise the longest real context, backward and gradient cleanup before updates.
    model.train();loss=-token_logps(model,example(max(train,key=lambda r:r['prompt_tokens']))).mean()
    loss.backward();optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(saved_cpu_rng);torch.cuda.set_rng_state_all(saved_cuda_rng)
    write(out/'gpu-smoke.json',dict(passed=True,job=job,max_prompt_tokens=max(r['prompt_tokens'] for r in train)))
    evaluate('frozen')
    tracking=tracker('openwebrl','train');model.train()
    started=time.monotonic()
    for update in range(completed+1,91):
        if time.time()>deadline:save(update-1);raise TimeoutError('Allocation budget reached; optimizer preserved')
        batch=train[(update-1)*32:update*32];tick=time.monotonic();mean=0.
        factor=update/5 if update<=5 else .1+.9*.5*(1+math.cos(math.pi*(update-5)/85))
        for group in optimizer.param_groups:group['lr']=1e-5*factor
        for row in batch:
            loss=-token_logps(model,example(row)).mean()
            if not torch.isfinite(loss):raise ValueError('Non-finite loss')
            (loss/len(batch)).backward();mean+=float(loss.detach())/len(batch)
        grad=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(grad):raise ValueError('Non-finite gradient')
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        metric=dict(update=update,ce_loss=mean,gradient_norm=float(grad),lr=1e-5*factor,seconds=time.monotonic()-tick)
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
        tracking.log(metric);print(json.dumps(metric),flush=True)
        if update%5==0 or update==90:save(update)
    tracking.finish();evaluate('adapted')
    records=list(rows(out/'frozen.jsonl'))+list(rows(out/'adapted.jsonl'))
    report=paired_report(records)
    reverse_by_id={r['id']:r for r in reverse}
    teacher_stability=sum(reverse_by_id[ident]['classes'][teacher_labels[(ident,'main')]['selected_original']]
                          ==reverse_by_id[ident]['classes'][teacher_labels[(ident,'reverse')]['selected_original']]
                          for ident in reverse_by_id)/len(reverse_by_id)
    strata={}
    for count in range(2,6):
        subset=[r for r in records if r['split']=='future' and r['treatment']=='main' and r['distinct_actions']==count]
        strata[f'distinct_actions_{count}']=paired_report(subset)
    for reward in [0,1]:
        subset=[r for r in records if r['split']=='future' and r['treatment']=='main' and r['rollout_reward']==reward]
        strata[f'rollout_reward_{reward}']=paired_report(subset)
    write(out/'report.json',dict(paired=report,strata=strata,teacher_order_action_agreement=teacher_stability,
        complete=True,training_and_adapted_eval_seconds=time.monotonic()-started,
        limitation='Teacher preference agreement; not terminal task success. Retention is not unseen pretraining data.'))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true')
    a=p.parse_args();prepare() if a.prepare else run()
