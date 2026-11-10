#!/usr/bin/env python3
"""Prepare/run matched-prompt, model-parity, generation and five-task browser gates.

Never submits compute or launches a broad evaluation. The Slurm controller owns
both one-GPU validation workers until they exit; failed gates stop promotion.
"""
import argparse
import asyncio
import base64
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from train_arm_joint_sft import REPO,RUNTIME,ACTOR,read_rows,write_json,make_example
from resume_arm_c2_training import allocation_deadline,parse_job_record
from openwebrl.serial_actions import augment_prompt,instructions,parse_serial
from openwebrl.serial_eval_gate import (identity,digest,generation_pass,browser_pass)

CONFIG=REPO/'openwebrl/docs/arm_results/serial_sft/recovery-v2'
OUTPUT=RUNTIME/'serial-validation-v2'
PYTHON=RUNTIME/'venv/bin/python'
SERVER_PYTHON=RUNTIME.parent/'venv/bin/python'
TASK_FILE=REPO/'openwebrl/data/eval/online-mind2web.jsonl'


def prepare():
    CONFIG.mkdir(parents=True,exist_ok=True)
    cohorts=json.loads((REPO/'openwebrl/docs/arm_c2_full300_cohorts.json').read_text())
    if digest(TASK_FILE)!=cohorts['task_file_sha256']:raise ValueError('Task cohort changed')
    cohort=cohorts['cohorts']['checkpoint_selection_100']
    # Outcome-independent fixed five tasks; same pilot for both models.
    pairs=sorted(zip(cohort['indices'],cohort['task_ids']),key=lambda x:hashlib.sha256(('serial-pilot:42:'+x[1]).encode()).hexdigest())[:5]
    pairs.sort();pilot=dict(count=5,indices=[x[0] for x in pairs],task_ids=[x[1] for x in pairs])
    write_json(CONFIG/'pilot.json',dict(task_file_sha256=digest(TASK_FILE),cohorts={'pilot':pilot}))
    rows=read_rows(RUNTIME/'serial-data-v1/all5.validation.jsonl');panel=[]
    for source in ['C2','Piotr']:
        for initial in [True,False]:
            subset=[r for r in rows if r['source']==source and ((r['turn']==0) if initial else (r['turn']>=5))]
            subset.sort(key=lambda r:hashlib.sha256(('serial-free:42:'+r['id']).encode()).hexdigest())
            panel.extend(r['id'] for r in subset[:2])
    if len(set(panel))!=8:raise ValueError('Need eight held-out states across source/history strata')
    for variant in ['all5','diverse3']:
        root=RUNTIME/f'runs/serial-{variant}-v1';model=root/'evaluation/merged-endpoint'
        rows=read_rows(RUNTIME/f'serial-data-v1/{variant}.validation.jsonl')
        rows={r['id']:r for r in rows};selected=[rows[i] for i in panel]
        for row in selected:
            original=row['prompt'].replace(instructions(variant),'',1)
            if augment_prompt(original,variant)!=row['prompt']:raise ValueError('Training prompt round-trip mismatch')
        path=CONFIG/f'{variant}-panel.jsonl'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
        config=dict(variant=variant,run_root=str(root),model=str(model),checkpoint=str(root/'student/update-000170'),
            output=str(OUTPUT/variant),panel=str(path),panel_sha256=digest(path),
            panel_ids=panel,pilot=str(CONFIG/'pilot.json'),pilot_sha256=digest(CONFIG/'pilot.json'),
            pilot_task_ids=pilot['task_ids'],identity=identity(model,variant,TASK_FILE,REPO),
            checker_sha256=digest(__file__),
            worker_sha256=digest(REPO/'scripts/run_arm_full300_checkpoint_eval.py'),
            gate_rules=dict(generations=16,require_all_valid=True,temperatures=[0,.7],
                merge_first_token_kl_max=.01,merge_label_logprob_mae_max=.1,
                browser_tasks=5,min_valid=3,min_multiturn=2,allow_protocol_failures=0),
            budget=dict(gpus=2,hours=1,cpus=16,memory_gib=240,scope='both variants; validation and five-task pilots only'))
        out=CONFIG/f'{variant}-config.json'
        if out.exists():raise ValueError('Recovery config already exists; preserve frozen attempt')
        write_json(out,config);print(str(out))


def parity(config):
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from peft import PeftModel
    torch.set_num_threads(4);torch.cuda.set_device(0)
    rows=read_rows(config['panel']);processor=AutoProcessor.from_pretrained(ACTOR,local_files_only=True)
    reference={};results=[]
    for mode in ['adapter','merged']:
        model=AutoModelForImageTextToText.from_pretrained(str(ACTOR) if mode=='adapter' else config['model'],
            local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
        if mode=='adapter':model=PeftModel.from_pretrained(model,config['checkpoint'],local_files_only=True)
        model.eval()
        with torch.no_grad():
            for row in rows:
                inputs,prefix,count=make_example(row,processor)
                n=min(64,count);positions=torch.arange(prefix-1,prefix+n-1,device='cuda')
                logits=model(**inputs,logits_to_keep=positions,use_cache=False).logits[0].float()
                lp=logits.log_softmax(-1);labels=inputs['input_ids'][0,prefix:prefix+n]
                selected=lp.gather(-1,labels[:,None]).squeeze(-1).cpu()
                first=lp[0].cpu()
                if mode=='adapter':reference[row['id']]=(first,selected)
                else:
                    old,old_selected=reference[row['id']]
                    kl=(old.exp()*(old-first)).sum().item();mae=(selected-old_selected).abs().mean().item()
                    results.append(dict(id=row['id'],first_token_kl=kl,label_logprob_mae=mae,
                        same_first_token=int(old.argmax())==int(first.argmax()),first_token_id=int(first.argmax()),
                        first_token_text=processor.tokenizer.decode([int(first.argmax())],skip_special_tokens=False)))
                del inputs,logits,lp
        del model;gc.collect();torch.cuda.empty_cache()
    rules=config['gate_rules']
    passed=all(r['same_first_token'] and r['first_token_kl']<=rules['merge_first_token_kl_max'] and
        r['label_logprob_mae']<=rules['merge_label_logprob_mae_max'] for r in results)
    write_json(Path(config['output'])/'parity.json',dict(passed=passed,examples=results,
        note='HF original base+adapter versus exported model, same eight image-expanded training prefixes and first 64 response positions.'))
    if not passed:raise RuntimeError('Adapter/merged model parity failed')


async def free_generation(config,port):
    import httpx
    from audit_arm_preference_pairs import parse_action,schemas
    rows=read_rows(config['panel']);sem=asyncio.Semaphore(4);results=[]
    async with httpx.AsyncClient(timeout=300,trust_env=False) as client:
        async def one(row,temperature):
            async with sem:
                record=dict(id=row['id'],temperature=temperature,valid=False)
                start=time.monotonic()
                try:
                    images=[]
                    for item in row['images']:
                        if digest(item['path'])!=item['sha256']:raise ValueError('Panel image changed')
                        images.append(base64.b64encode(Path(item['path']).read_bytes()).decode())
                    resp=await client.post(f'http://127.0.0.1:{port}/generate',json=dict(text=row['prompt'],image_data=images,
                        sampling_params=dict(temperature=temperature,top_p=.9,max_new_tokens=min(6144,32768-row['prompt_tokens']),sampling_seed=42),
                        return_logprob=True))
                    resp.raise_for_status();data=resp.json();meta=data.get('meta_info',{})
                    record.update(text=data['text'],finish=meta.get('finish_reason',{}).get('type'),
                        prompt_tokens=meta.get('prompt_tokens'),prompt_tokens_match=meta.get('prompt_tokens')==row['prompt_tokens'],
                        completion_tokens=meta.get('completion_tokens'))
                    final,history,parsed=parse_serial(data['text'],config['variant'])
                    tools=schemas(row['prompt']);parse_action(final,tools)
                    for candidate in parsed['candidates']:
                        action='\n'.join('<tool_call>'+json.dumps(c)+'</tool_call>' for c in candidate['actions'])
                        parse_action(action,tools)
                    first=meta.get('output_token_logprobs',[])
                    record.update(valid=True,candidate_count=parsed['candidate_count'],selected_index=parsed['selected_index'],
                        distinct_actions=parsed['distinct_actions'],history=history,
                        first_token_id=first[0][1] if first else None)
                except Exception as exc:record['error']=repr(exc)
                record['seconds']=time.monotonic()-start;return record
        results=await asyncio.gather(*(one(row,t) for row in rows for t in [0.,.7]))
    write_json(Path(config['output'])/'free-generation.json',dict(examples=results))
    return results


def worker(config,deadline):
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    processes=[];logs=[];stopping=False
    def status(phase,**extras):write_json(out/'status.json',dict(phase=phase,job=os.getenv('SLURM_JOB_ID'),
        variant=config['variant'],updated_utc=datetime.now(timezone.utc).isoformat(),**extras))
    def launch(cmd,name):
        log=(out/name).open('a');logs.append(log)
        child=subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(child);return child
    def stop(*_):
        nonlocal stopping
        stopping=True
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
    def wait(p):
        while p.poll() is None:
            if stopping or time.time()>=deadline:stop();raise TimeoutError('Validation allocation deadline')
            time.sleep(2)
        return p.returncode
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if identity(config['model'],config['variant'],TASK_FILE,REPO)!=config['identity']:raise ValueError('Runtime/model differs from prepared recovery config')
        for path,sha in [(config['panel'],config['panel_sha256']),(config['pilot'],config['pilot_sha256']),
            (__file__,config['checker_sha256']),(REPO/'scripts/run_arm_full300_checkpoint_eval.py',config['worker_sha256'])]:
            if digest(path)!=sha:raise ValueError('Prepared validation artifact changed')
        config_path=CONFIG/f"{config['variant']}-config.json"
        status('checking_adapter_merge_parity')
        if wait(launch([str(PYTHON),__file__,'--config',str(config_path),'--parity'],'parity.log')):
            raise RuntimeError('Adapter/merged model parity failed; no generation/browser evaluation launched')
        port=31110 if config['variant']=='all5' else 31510
        status('starting_generation_server')
        server=launch([str(SERVER_PYTHON),'-m','sglang.launch_server','--model-path',config['model'],
            '--host','127.0.0.1','--port',str(port),'--dtype','bfloat16','--tp','1','--mem-fraction-static','.45',
            '--context-length','32768','--max-running-requests','8','--chunked-prefill-size','4096','--disable-cuda-graph'],'generation-server.log')
        import urllib.request
        startup=min(deadline,time.time()+600)
        while time.time()<startup:
            if server.poll() is not None:raise RuntimeError('Generation server exited')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/get_model_info',timeout=5) as r:info=json.load(r)
                if Path(info['model_path']).resolve()!=Path(config['model']).resolve():raise ValueError('Wrong server model')
                break
            except OSError:time.sleep(2)
        else:raise TimeoutError('Generation server startup')
        status('checking_free_generation')
        records=asyncio.run(free_generation(config,port))
        parity_report=json.loads((out/'parity.json').read_text());expected={r['id']:r['first_token_id'] for r in parity_report['examples']}
        backend_match=all(r.get('first_token_id')==expected[r['id']] for r in records if r['temperature']==0)
        passed=generation_pass(records,config['panel_ids']) and backend_match
        gate=dict(schema_version=1,stage='free_generation',passed=passed,identity=config['identity'],
            checks=dict(adapter_merge_parity=parity_report['passed'],free_generation=passed,sglang_first_token_parity=backend_match),
            pilot_task_ids=config['pilot_task_ids'],artifacts={str(out/n):digest(out/n) for n in ['parity.json','free-generation.json']})
        write_json(out/'generation-gate.json',gate)
        # Release the generation server before the browser worker owns its server.
        os.killpg(server.pid,signal.SIGTERM)
        try:server.wait(timeout=60)
        except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
        if not passed:raise RuntimeError('Free-generation gate failed; no browser pilot or broad evaluation launched')
        status('running_five_task_browser_pilot')
        cmd=[str(PYTHON),str(REPO/'scripts/run_arm_full300_checkpoint_eval.py'),'--job-id',os.environ['SLURM_JOB_ID'],
            '--run-root',config['run_root'],'--label','pilot','--model',config['model'],'--expected-checkpoint',config['checkpoint'],
            '--dataset-audit',str(Path(config['run_root'])/'dataset-audit.json'),'--output-group',str(out/'browser'),
            '--task-cohort',config['pilot'],'--cohort-name','pilot','--actor-port',str(port),
            '--browser-port-start',str(port+90),'--browser-port-end',str(port+289),'--parallel','3',
            '--mem-fraction-static','.45','--serial-variant',config['variant'],
            '--serial-gate',str(out/'generation-gate.json'),'--serial-gate-pilot']
        if wait(launch(cmd,'browser-pilot.log')):raise RuntimeError('Browser pilot worker failed')
        paths=list((out/'browser/pilot/online-mind2web/results').glob('*.json'))
        rows=[json.loads(p.read_text()) for p in paths];passed=browser_pass(rows,config['pilot_task_ids'])
        gate.update(stage='browser_pilot',passed=passed,checks=dict(gate['checks'],browser_pilot=passed))
        gate['artifacts'].update({str(p):digest(p) for p in paths})
        gate['artifacts'][str(out/'generation-gate.json')]=digest(out/'generation-gate.json')
        write_json(out/'browser-gate.json',gate)
        if not passed:raise RuntimeError('Browser execution/history gate failed; broad evaluation remains blocked')
        status('passed',browser_gate=str(out/'browser-gate.json'),broad_evaluation_launched=False)
    except Exception as exc:status('failed',error=repr(exc),broad_evaluation_launched=False);raise
    finally:
        stop()
        for child in processes:
            if child.poll() is None:
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for log in logs:log.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--config',type=Path)
    p.add_argument('--parity',action='store_true');args=p.parse_args()
    if args.prepare:return prepare()
    job=os.getenv('SLURM_JOB_ID')
    if not job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():raise ValueError('GPU validation requires an approved allocation')
    record=parse_job_record(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True,timeout=20))
    deadline=allocation_deadline(record,margin_minutes=5).timestamp()
    if args.config:
        config=json.loads(args.config.read_text())
        return parity(config) if args.parity else worker(config,deadline)
    if int(record['NumNodes'])!=1 or int(record['NumCPUs'])<16:raise ValueError('Expected one node and 16 CPUs')
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True,timeout=20).strip().splitlines()
    if len(devices)!=2 or any('H200' not in d for d in devices):raise ValueError('Expected two H200s')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=20).strip():raise ValueError('GPUs occupied')
    OUTPUT.mkdir(parents=True,exist_ok=True);children=[];logs=[]
    def stop(*_):
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for index,variant in enumerate(['all5','diverse3']):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=devices[index].split(',',1)[0]
            log=(OUTPUT/f'{variant}-controller.log').open('a');logs.append(log)
            children.append(subprocess.Popen([str(PYTHON),__file__,'--config',str(CONFIG/f'{variant}-config.json')],
                cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(c.poll() is None for c in children):
            if time.time()>=deadline:stop()
            time.sleep(5)
        statuses={v:json.loads((OUTPUT/v/'status.json').read_text()) for v in ['all5','diverse3']}
        write_json(CONFIG/'status.json',statuses)
        if any(c.returncode for c in children):raise RuntimeError('At least one validation gate failed; inspect per-variant reports')
    finally:
        stop()
        for child in children:
            if child.poll() is None:
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for log in logs:log.close()


if __name__=='__main__':main()
