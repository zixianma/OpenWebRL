"""Diagnostic controller snapshot; production controller/source remain unchanged.

Lifecycle copied from run_arm_turn_bonus_cycles.execute on 2026-09-22.
Only the readiness entry is replaced with the isolated diagnostic validator.
"""
from run_arm_turn_bonus_cycles import (Path, os, subprocess, time, signal, json,
    urllib, re, RUNTIME, REPO, SELECTOR, allocation, clean_environment,
    source_command, validate_source, write_json, selector_preflight, health_issue, tail)
from run_arm_failure_gpu_diagnostic import validate_plan

def execute(p):
    readiness=validate_plan(p)
    SOURCE = Path(p['source'])
    validate_source(SOURCE)
    job=p['job_id']
    if os.getenv('SLURM_JOB_ID') != job or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Execute inside the dedicated authorized allocation')
    requested=p['requested_resources']
    resources=allocation(subprocess.check_output(['scontrol','show','job',job,'-o'],text=True),job,requested_gpus=requested['gpus'],maximum_hours=requested['hours'])
    if resources['cpus']<requested['cpus'] or resources['allocated_memory_gib']<requested['memory_gib']:
        raise ValueError('Allocation does not meet the prepared CPU/memory profile')
    steps=subprocess.check_output(['squeue','--steps',f'--jobs={job}','--noheader','--format=%i'],text=True).split()
    if set(steps)-{f'{job}.{x}' for x in ('batch','extern','interactive',os.getenv('SLURM_STEP_ID'))}:
        raise ValueError('Another worker is active in this allocation')
    reserve=p.get('evaluation_reserve_seconds',0)
    if reserve not in (0,3600):raise ValueError('Unsupported reserved evaluation time')
    deadline=time.time()+min(resources['maximum_seconds'],p['requested_resources']['hours']*3600-180)-reserve
    if deadline-time.time()<600:raise ValueError('Insufficient training time after evaluation reserve')
    p['arm_config']['deadline_epoch_seconds']=deadline
    from dotenv import dotenv_values
    env={k:v for k,v in clean_environment().items() if not k.startswith('OPENWEBRL_ARM_')}
    for k,v in dotenv_values(REPO/'.env').items():
        if v and k.startswith(('WANDB_','JUDGE_','OPENAI_','AZURE_')): env.setdefault(k,v)
    if env.get('OPENAI_API_KEY') and not env.get('JUDGE_API_BASE'):
        env.update(JUDGE_API_MODE='served',JUDGE_API_BASE='https://api.openai.com/v1')
    if not env.get('WANDB_API_KEY') or not any(env.get(k) for k in ('OPENAI_API_KEY','JUDGE_API_KEY','AZURE_OPENAI_API_KEY')):
        raise ValueError('Missing W&B or terminal-judge credentials')
    env.update(p['environment'])
    devices=env.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=requested['gpus'] or len(set(devices))!=requested['gpus'] or not all(devices):
        raise ValueError('GPU assignment does not match the prepared profile')
    if p.get('resume_from'):
        from resume_arm_turn_bonus import validate_resume_plan
        validate_resume_plan(p,require_gpu_restore=True)
    root=Path(p['output']); root.mkdir(parents=True,exist_ok=False)
    if p.get('resume_from'):
        from resume_arm_turn_bonus import stage_checkpoint_reference
        stage_checkpoint_reference(p)
    for name,value in [('launch_manifest',p),('arm-config',p['arm_config']),('browser-training-config',p['browser_config'])]:
        write_json(root/f'{name}.json',value)
    write_json(root/'launch-readiness.json',readiness)
    children=[]; handles=[]; selector=None; worker=None; validated={}; stage='actor-startup'; current=None
    result=dict(training_complete=False,no_optimizer_updates=True,completed_iterations=0)
    def status(**values):
        result.update(values,updated_at_epoch_seconds=time.time(),stage=stage,seconds_remaining=int(deadline-time.time()))
        write_json(root/'status.json',result)
    def interrupted(*unused): raise InterruptedError('Allocation shutdown requested')
    old=[signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)]
    def stop_child(child):
        if child is not None and child.poll() is None:
            try: os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                try: os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                child.wait()
    port_lease=None
    try:
        if json.loads((SOURCE/'reference_manifest.json').read_text()).get('runtime_port_leases'):
            from runtime_ports import PORT_ENV, lease_ports
            port_lease,port_base=lease_ports(job)
            env[PORT_ENV]=str(port_base)
            p['selector_port']=port_base+250
            p['arm_config']['selector_endpoint']=f"http://127.0.0.1:{p['selector_port']}"
            write_json(root/'arm-config.json',p['arm_config'])
            write_json(root/'launch_manifest.json',p)
            write_json(root/'port-lease.json',dict(base=port_base,size=256,job_id=job))
        status()
        handles.append((root/'collection.log').open('w'))
        worker=subprocess.Popen(source_command(SOURCE,['timeout','--signal=INT','--kill-after=60',str(int(deadline-time.time())),*p['command']]),
            cwd=SOURCE,env=env,stdout=handles[-1],stderr=subprocess.STDOUT,start_new_session=True)
        children.append(worker)
        last_request=None
        while True:
            if time.time()>=deadline: raise TimeoutError('Allocation deadline reached')
            request_path=root/'teacher-request.json'
            request=json.loads(request_path.read_text()) if request_path.exists() else None
            if request is not None and request!=last_request and worker.poll() is None:
                current=root/'iterations'/f"{request['rollout_id']:04d}"
                if request['phase']=='collection':
                    if selector is not None: raise ValueError('Teacher from previous collection was not released')
                    stage='selector-startup'; status(iteration=request['rollout_id']+1)
                    handles.append((current/'selector.log').open('w'))
                    selector_env=dict(env,CUDA_VISIBLE_DEVICES=devices[-1],PYTHONPATH=str(SOURCE),
                        CPATH=str(RUNTIME/'src/python-headers/Include')+':'+str(RUNTIME/'src/python-headers'))
                    selector_command=[str(RUNTIME/'arm-reproduction/venv/bin/python'),str(SOURCE/'scripts/serve_arm.py'),
                        '--mode','selection','--model',str(SELECTOR),'--port',str(p['selector_port'])]
                    if p['selector_provider']=='sol':
                        selector_command=[str(RUNTIME/'venv/bin/python'),str(REPO/'scripts/serve_sol_selector.py'),
                            '--output',str(root/'sol-api'),'--port',str(p['selector_port']),'--max-cost-usd','200']
                        selector_env['CUDA_VISIBLE_DEVICES']=''
                    selector=subprocess.Popen(selector_command,cwd=SOURCE,env=selector_env,
                        stdout=handles[-1],stderr=subprocess.STDOUT,start_new_session=True)
                    children.append(selector)
                    startup=min(deadline-120,time.time()+540)
                    while True:
                        if selector.poll() is not None: raise RuntimeError('Selector failed to start')
                        try:
                            with urllib.request.urlopen(p['arm_config']['selector_endpoint']+'/health',timeout=2) as response:
                                if response.status==200: break
                        except OSError: pass
                        if time.time()>startup: raise TimeoutError('Selector startup timeout')
                        time.sleep(3)
                    selector_preflight(p['arm_config']['selector_endpoint'],current,
                        expected_api_model='gpt-5.6-sol' if p['selector_provider']=='sol' else None)
                    stage='collection'
                elif request['phase']=='training':
                    stop_child(selector); selector=None; stage='training'
                else: raise ValueError('Unknown teacher lifecycle request')
                write_json(root/'teacher-ready.json',request); last_request=request
            if selector is not None and selector.poll() is not None: raise RuntimeError('Selector exited during collection')
            api_usage=root/'sol-api/usage.json'
            if (root/'sol-api/budget-exhausted.json').exists():
                raise RuntimeError('Sol API usage cap reached; preserve collection for inspection')
            if api_usage.exists() and json.loads(api_usage.read_text()).get('consecutive_failures',0)>=5:
                raise RuntimeError('Five consecutive Sol selector failures; preserve collection for inspection')
            recent=tail(root/'runtime/progress.log')+'\n'+tail(root/'collection.log',65536)
            live_path=current/'live_metrics.json' if current else root/'unused'
            live=json.loads(live_path.read_text()) if live_path.exists() else {}
            if issue:=health_issue(live,recent): raise RuntimeError(issue)
            if re.search(r'\[TrainMetrics\][^\n]*\bstep=\d+\b[^\n]*\bgrad_norm=',recent):
                result['no_optimizer_updates']=False
            # Validate each saved checkpoint against accumulated native PPO steps.
            for saved in sorted((root/'iterations').glob('*/checkpoint-saved.json')):
                row=json.loads(saved.read_text()); rid=row['rollout_id']
                if rid in validated: continue
                marker=root/'runtime/latest_checkpointed_iteration.txt'
                if not marker.exists() or int(marker.read_text())!=rid:
                    raise ValueError('Checkpoint marker disagrees with completed native save')
                expected=p.get('initial_optimizer_updates',0)+sum(json.loads(x.read_text())['expected_optimizer_updates']
                    for x in (root/'iterations').glob('*/training_gate.json')
                    if int(x.parent.name)<=rid)
                report=saved.parent/'checkpoint-validation.json'
                subprocess.run(source_command(SOURCE,['python',str(REPO/'scripts/inspect_training_checkpoint.py'),
                    str(root/'runtime'),'--expected-updates',str(expected),'--report',str(report)]),
                    env=env,check=True,stdout=subprocess.DEVNULL,timeout=120)
                validated[rid]=json.loads(report.read_text())
                write_json(root/'completed-checkpoints.json',list(validated.values()))
                status(completed_iterations=len(validated),completed_optimizer_updates=expected,
                    last_valid_checkpoint=row['checkpoint'])
            status(live_metrics=live)
            if worker.poll() is not None: break
            time.sleep(10)
        if worker.returncode: raise RuntimeError(f'Native training exited {worker.returncode}; inspect preserved artifacts')
        stage='complete'
        if p.get('diagnostic_live_shadow'):
            report=verify_live_shadow(p,root,result['no_optimizer_updates'])
            status(shadow_complete=True,training_complete=False,has_saved_checkpoints=False,exit_code=0,
                completed_optimizer_updates=0,live_admitted_labels=report['admitted'],
                stop_reason='Live executed-turn hook verified; no training or statistical calibration claim')
            return
        status(training_complete=len(validated)==p['requested_iterations'],has_saved_checkpoints=bool(validated),exit_code=0,
            stop_reason='requested iterations complete' if len(validated)==p['requested_iterations'] else 'budget or calibration gate; inspect iteration artifacts')
        if not validated: raise RuntimeError('No completed ARM training iteration')
    except BaseException as exc:
        status(failed=True,error_type=type(exc).__name__,error=str(exc)[:500])
        raise
    finally:
        for child in reversed(children): stop_child(child)
        for handle in handles: handle.close()
        if port_lease is not None:port_lease.close()
        for sig,handler in zip((signal.SIGTERM,signal.SIGINT),old): signal.signal(sig,handler)

