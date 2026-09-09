#!/usr/bin/env python3
"""Train a C2 LoRA student from a complete, audited browser-turn dataset."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time

from openwebrl.arm_c2 import load_training_example, sha, write_json


def language_targets(model):
    suffixes = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    names = [name for name, _ in model.named_modules()
             if name.startswith('model.language_model.layers.') and name.rsplit('.', 1)[-1] in suffixes]
    if not names or any('visual' in name for name in names):
        raise ValueError('Language-only LoRA target discovery failed')
    return names


def validate_compute(config, allow_resident_eval=False):
    job = str(config['allocation'])
    if os.getenv('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Trainer is outside the authorized allocation')
    def command(argv): return subprocess.check_output(argv, text=True, timeout=20).strip()
    info = command(['scontrol', 'show', 'job', job, '-o'])
    if 'JobState=RUNNING' not in info: raise ValueError('Allocation is not running')
    devices = command(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']).splitlines()
    # A two-GPU collection step can hand off the unchanged single-GPU trainer.
    # Bind by UUID so CUDA cannot select the other allocated GPU accidentally.
    explicit_binding = os.getenv('CUDA_VISIBLE_DEVICES') == config['gpu_uuid']
    if devices != [config['gpu_uuid']] and not (config['gpu_uuid'] in devices and explicit_binding):
        raise ValueError('Wrong visible GPU')
    pids = command(['nvidia-smi', '--id=' + config['gpu_uuid'], '--query-compute-apps=pid', '--format=csv,noheader']).splitlines()
    if pids and not allow_resident_eval:
        raise ValueError('Training requires the assigned GPU to be free of inference servers')
    if allow_resident_eval:
        from scripts.run_arm_retry import process_identity
        expected = {str(p['pid']): p for p in config['resident_actor_processes']}
        if set(pids) != set(expected): raise ValueError('Smoke GPU has unexpected processes')
        for pid, identity in expected.items():
            if process_identity(pid) != identity: raise ValueError('Resident actor identity changed')


def run(config_path, resume=None, smoke=False):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model, PeftModel
    config = json.loads(Path(config_path).read_text())
    root = Path(config['output'])
    recipe = config['training']
    validate_compute(config, allow_resident_eval=smoke)
    actor = config.get('teacher_manifest', {}).get('actor', '/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT')
    processor = AutoProcessor.from_pretrained(actor, local_files_only=True)
    random.seed(recipe['seed']); torch.manual_seed(recipe['seed']); torch.cuda.manual_seed_all(recipe['seed'])
    if not smoke:
        audit = json.loads((root / 'dataset-audit.json').read_text())
        if not audit['complete_collection'] or audit['scheduled'] != config['task_count']:
            raise ValueError('Cannot train the full-pool student on partial collection')
        path = Path(audit['dataset'])
        if sha(path.read_bytes()) != audit['dataset_sha256']: raise ValueError('Dataset checksum mismatch')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) < 256: raise ValueError('Fewer than 256 usable turns; inspect retention before optimization')
        dataset_hash = audit['dataset_sha256']
    else:
        rows, dataset_hash = [], 'synthetic-engineering-smoke-only'
    model = AutoModelForImageTextToText.from_pretrained(actor, local_files_only=True,
                torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='sdpa')
    targets = language_targets(model)
    if resume:
        model = PeftModel.from_pretrained(model, resume, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=recipe['lora_rank'], lora_alpha=recipe['lora_alpha'],
                    lora_dropout=recipe['lora_dropout'], target_modules=targets, bias='none', task_type='CAUSAL_LM'))
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in name or 'language_model.layers.' not in name for name, _ in trainable):
        raise ValueError('Unexpected trainable parameters outside language LoRA')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    params = [p for _, p in trainable]
    optimizer = torch.optim.AdamW(params, lr=recipe['learning_rate'], betas=(.9, .95), eps=1e-8, weight_decay=.01)
    output = root / ('synthetic-training-smoke' if smoke else 'student')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'training-config.json', dict(recipe, dataset_sha256=dataset_hash, actor=actor,
               targets=targets, trainable_parameters=sum(p.numel() for p in params), supervision='current response only',
               selection='fixed epoch 2; no evaluation-driven checkpoint selection', synthetic_smoke=smoke))
    if smoke:
        from PIL import Image
        from slime.utils.processing_utils import build_processor_kwargs
        prompt = processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':'Describe this test image briefly.'}]}], tokenize=False, add_generation_prompt=True)
        features = processor(text=[prompt], **build_processor_kwargs({'images':[Image.new('RGB',(128,128),'gray')]}))
        prefix = features['input_ids'][0]
        if hasattr(prefix, 'tolist'): prefix = prefix.tolist()
        target = processor.tokenizer.encode('A gray test image.<|im_end|>\n', add_special_tokens=False)
        ids = torch.tensor([prefix + target], dtype=torch.long)
        labels = torch.full_like(ids, -100); labels[0,len(prefix):] = torch.tensor(target)
        tensors = {k:v for k,v in features.items() if k not in ('input_ids','attention_mask') and isinstance(v,torch.Tensor)}
        example = dict(input_ids=ids, labels=labels, attention_mask=torch.ones_like(ids), **tensors)
        batches = [example]
    else:
        batches = None
    accum = recipe['gradient_accumulation']
    total_updates = 1 if smoke else recipe['epochs'] * math.ceil(len(rows) / accum)
    warmup = max(1, math.ceil(total_updates * .03))
    state = dict(epoch=0, next_position=0, updates=0, dataset_sha256=dataset_hash)
    if resume:
        saved = torch.load(Path(resume) / 'optimizer.pt', map_location='cpu', weights_only=False)
        state = saved['state']
        if state['dataset_sha256'] != dataset_hash: raise ValueError('Resume dataset differs')
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    if not smoke and state['epoch'] >= recipe['epochs']:
        print('Training already completed the configured epochs.', flush=True)
        return
    stop = False
    def interrupted(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    deadline = datetime.fromisoformat(config['stop_utc']).timestamp()
    def save_checkpoint(label):
        checkpoint = output / label
        if checkpoint.exists(): raise ValueError('Refuse to overwrite a durable student checkpoint')
        temporary = output / (label + '.incomplete')
        temporary.mkdir()
        model.save_pretrained(temporary, safe_serialization=True)
        processor.save_pretrained(temporary)
        torch.save(dict(optimizer=optimizer.state_dict(), state=state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all()), temporary / 'optimizer.pt')
        write_json(temporary / 'progress.json', state)
        temporary.rename(checkpoint)
        write_json(output / 'latest-checkpoint.json', dict(path=str(checkpoint), **state))
        return checkpoint
    for epoch in range(state['epoch'], 1 if smoke else recipe['epochs']):
        order = list(range(1 if smoke else len(rows)))
        random.Random(recipe['seed'] + epoch).shuffle(order)
        start = state['next_position'] if epoch == state['epoch'] else 0
        for position in range(start, len(order), accum):
            if stop or (not smoke and time.time() > deadline - 180):
                checkpoint = save_checkpoint(f'paused-{state["updates"]:06d}-{int(time.time())}')
                print('PAUSED', checkpoint, flush=True); return
            selected = order[position:position+accum]
            optimizer.zero_grad(set_to_none=True)
            value = 0.
            for index in selected:
                example = batches[0] if smoke else load_training_example(rows[index], processor)
                if example['input_ids'].shape[1] > recipe['max_tokens']: raise ValueError('Oversize training sequence')
                example = {k:v.to('cuda:0') for k,v in example.items()}
                result = model(**example, use_cache=False)
                if not torch.isfinite(result.loss): raise ValueError('Non-finite training loss')
                (result.loss / len(selected)).backward()
                value += result.loss.detach().item() / len(selected)
                del result, example
            gradient = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(gradient) or (smoke and gradient.item() == 0): raise ValueError('Invalid gradient norm')
            step = state['updates'] + 1
            factor = min(1., step / warmup) if step <= warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1,total_updates - warmup)))
            for group in optimizer.param_groups: group['lr'] = recipe['learning_rate'] * factor
            optimizer.step()
            state.update(epoch=epoch, next_position=position+len(selected), updates=step)
            event = dict(utc=datetime.now(timezone.utc).isoformat(), **state, loss=value, grad_norm=gradient.item(), lr=optimizer.param_groups[0]['lr'])
            with (output / 'metrics.jsonl').open('a') as stream: stream.write(json.dumps(event)+'\n')
            print('UPDATE', json.dumps(event), flush=True)
            if step % 100 == 0: save_checkpoint(f'update-{step:06d}')
        state.update(epoch=epoch+1, next_position=0)
        checkpoint = save_checkpoint('smoke-adapter' if smoke else f'epoch-{epoch+1}')
    if smoke:
        from safetensors.torch import load_file
        from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
        expected = {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
        loaded = load_file(str(checkpoint / 'adapter_model.safetensors'))
        if set(expected) != set(loaded) or any(not torch.equal(expected[k],loaded[k]) for k in expected):
            raise ValueError('Adapter save does not match in-memory weights')
        set_peft_model_state_dict(model, loaded)
        actual = get_peft_model_state_dict(model)
        if any(not torch.equal(expected[k],actual[k].detach().cpu()) for k in expected): raise ValueError('Adapter reload mismatch')
        write_json(output / 'smoke-passed.json', dict(loss=value, gradient_norm=gradient.item(), adapter_save_reload=True,
                   language_target_modules=len(targets), trained_on_evaluation_data=False, usable_student_checkpoint=False))
    else:
        write_json(output / 'complete.json', dict(checkpoint=str(checkpoint), **state))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.config, args.resume, args.smoke)


if __name__ == '__main__': main()
