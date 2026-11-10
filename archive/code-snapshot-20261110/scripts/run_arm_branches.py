#!/usr/bin/env python3
"""Own an approved continuation-branch allocation; never submit new jobs."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

SOURCE = Path(__file__).resolve().parents[1]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')
sys.path.insert(0, str(SOURCE))
from openwebrl.arm_branch_state import BranchState
from openwebrl.arm_branch_worker import sha, write
from openwebrl.controlled_sft_state import digest
from resume_baseline import source_command, validate_source
from runtime_ports import lease_ports
from run_controlled_sft_suite import check_approval, scheduler_accounting, stop_owned, cleanup_browser_groups, initialize_wandb


def read(path): return json.loads(Path(path).read_text())


def pointer(path, value):
    temp = Path(str(path)+'.tmp-'+str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temp, path)


def expand_ready_preflights(groups, state, start_worker, urls):
    """Start the remaining 14 only after c0/r0 reaches the unchanged barrier.

    Inspect every independent state before the ordered release pass. The probe
    is an ordinary scientific branch, not an extra browser or an extra sample.
    """
    for group, sd, anchor, items, slot_base, workers in groups:
        if len(workers) != 1 or workers[0].poll() is not None:
            continue
        row, attempts = state.item(items[0]['item_id'])
        if (row['status'] != 'active' or not attempts
                or attempts[-1]['status'] != 'active'
                or not (Path(attempts[-1]['artifact_directory'])/'ready.json').exists()):
            continue
        for j, item in enumerate(items[1:], 1):
            workers.append(start_worker(item, slot_base+j, urls[(slot_base+j)%len(urls)]))


def discovery_wave(plan, offset):
    """Prefetch prefixes in fixed-order waves; scientific release stays paired."""
    width=plan.get('discovery_prefetch_states',plan['parallel_states'])
    if (type(width) is not int or width < plan['parallel_states']
            or width % plan['parallel_states'] or width > plan['limits']['concurrent_browsers']
            or width + plan['parallel_states'] > plan['resources']['collectors']):
        raise ValueError('Discovery prefetch must preserve browser and collector limits')
    return plan['states'][offset:offset+width] if offset % width == 0 else []


def validate(root):
    root = Path(root).resolve(); plan = read(root/'plan.json')
    if Path(plan['root']) != root or Path(plan['source']) != SOURCE or sha(__file__) != plan['controller_sha256']:
        raise ValueError('Use the exact prepared controller from its frozen source')
    validate_source(SOURCE)
    for path, expected in plan['artifacts'].items():
        if sha(path) != expected: raise ValueError('Frozen input changed: '+path)
    for name, expected in plan['actor_files'].items():
        info = (Path(plan['worker_config']['actor_checkpoint'])/name).stat()
        if dict(size=info.st_size, mtime_ns=info.st_mtime_ns) != expected:
            raise ValueError('Pinned actor file changed')
    candidate_limit=plan.get('candidate_state_limit',200)
    if (type(candidate_limit) is not int or not 200 <= candidate_limit <= 800
            or plan['states_target'] != 100 or len(plan['states']) != candidate_limit
            or plan['candidate_count'] != 5 or plan['candidate_draws'] not in (5, 15)
            or plan['continuation_repetitions'] != 3):
        raise ValueError('Unexpected branch cohort')
    if len({x['id'] for x in plan['states']}) != candidate_limit or len({x['task_id'] for x in plan['states']}) != candidate_limit:
        raise ValueError('Candidate states must remain unique, one per task')
    discovery_wave(plan,0)
    if plan['candidate_draws'] == 5 and not plan.get('retain_duplicate_candidates'):
        raise ValueError('Five draws must retain duplicate candidates')
    return plan


def allocation_deadline(info, job, seconds, resources):
    fields = dict(re.findall(r'(\w+)=(\S+)', info))
    tres = dict(x.split('=',1) for x in fields.get('AllocTRES','').split(',') if '=' in x)
    if (fields.get('JobId') != str(job) or fields.get('JobState') != 'RUNNING'
            or not fields.get('UserId','').endswith(f'({os.getuid()})') or fields.get('NumNodes') != '1'
            or fields.get('NumCPUs') != str(resources['cpus'])
            or tres.get('gres/gpu:h200') != str(resources['gpus'])
            or tres.get('mem') not in (f"{resources['memory_gib']}G", f"{resources['memory_gib']*1024}M")):
        raise ValueError('Allocation differs from exact approved profile')
    value = fields['TimeLimit']; days, clock = value.split('-',1) if '-' in value else ('0',value)
    parts = [int(x) for x in clock.split(':')]
    if len(parts) == 2: parts.insert(0,0)
    duration = int(days)*86400 + parts[0]*3600 + parts[1]*60 + parts[2]
    if duration > seconds: raise ValueError('Allocation time exceeds remaining approved cap')
    end = datetime.strptime(fields['EndTime'], '%Y-%m-%dT%H:%M:%S').timestamp()
    return min(time.time()+duration, end)-180


def execute(root, job):
    root = Path(root).resolve(); plan = validate(root)
    if str(job) != os.getenv('SLURM_JOB_ID') or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Controller must own the registered Slurm allocation')
    approval = read(root/'approval.json')
    used, previous = scheduler_accounting(approval, job)
    seconds = check_approval(plan, approval, job, used)
    info = subprocess.check_output(['scontrol','show','job','-o',str(job)], text=True)
    deadline = allocation_deadline(info, job, seconds, plan['resources'])
    devices = os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    replicas=plan['resources']['sft_replicas']; parallel=plan['parallel_states']
    if len(devices) != replicas or len(set(devices)) != replicas or not all(devices):
        raise ValueError('Distinct approved GPUs required')
    from dotenv import dotenv_values
    env = dict(os.environ)
    for key,value in dotenv_values('/gpfs/projects/krishna/zixianma/OpenWebRL/.env').items():
        if value and key.startswith(('OPENAI_', 'JUDGE_', 'WANDB_')): env.setdefault(key,value)
    if not env.get('OPENAI_API_KEY'): raise ValueError('Missing teacher credential')
    hub = str(RUNTIME/'hf-cache/hub')
    env.update(WANDB_PROJECT='openwebrl-evals', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HOME=str(RUNTIME/'hf-cache'), HF_HUB_CACHE=hub, HUGGINGFACE_HUB_CACHE=hub,
        TRANSFORMERS_CACHE=hub, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    journal = root/'scheduler-attempts'/str(job); journal.mkdir(parents=True, exist_ok=True)
    write(journal/'prior-accounting.json', dict(prior_seconds=used, attempts=previous))
    children, handles, servers = [], [], []
    lease = wandb_run = state = None
    completed = False
    def interrupted(signum, frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    def spawn(argv, name, extra=None):
        handle = (journal/(name+'.log')).open('a'); handles.append(handle)
        process = subprocess.Popen(source_command(SOURCE, argv), cwd=SOURCE, env=dict(env, **(extra or {})),
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        return process
    def health(stage):
        if time.time() >= deadline: raise TimeoutError('All-attempt allocation boundary reached')
        if any(p.poll() is not None for p in servers): raise RuntimeError('Actor server failed')
        snapshot = state.snapshot()
        if snapshot['halt']: raise RuntimeError('Durable budget/provider halt requires diagnosis')
        pointer(root/'current.json', dict(stage=stage, job_id=str(job), updated_unix=time.time(), **snapshot))
    def wait_processes(processes, stage):
        while any(p.poll() is None for p in processes):
            health(stage)
            if any(p.poll() not in (None,0) for p in processes): raise RuntimeError('Worker failed: '+stage)
            time.sleep(2)
        if any(p.returncode != 0 for p in processes): raise RuntimeError('Worker failed: '+stage)
    def result(item):
        row, attempts = state.item(item['item_id'])
        authorized = read(root/'recovery-authorizations.json').get('attempt_ids', []) if (root/'recovery-authorizations.json').exists() else []
        if (row['status'] != 'finished' or not attempts or attempts[-1]['status'] != 'finished'
                or any(a['status'] != 'interrupted' or a['attempt_id'] not in authorized for a in attempts[:-1])):
            raise ValueError('Unresolved item requires diagnosed recovery')
        answer = json.loads(attempts[-1]['result_json'])
        if sha(answer['result_path']) != answer['result_sha256']: raise ValueError('Result receipt changed')
        return answer
    def start_worker(item, slot, actor):
        return spawn([str(RUNTIME/'venv/bin/python'),'-m','openwebrl.arm_branch_worker',
            '--root',str(root),'--item',item['item_id'],'--actor-url',actor], item['item_id'],
            dict(CUDA_VISIBLE_DEVICES='', OPENWEBRL_ROLLOUT_PORT_BASE=str(base+64+slot*12)))

    with (root/'controller.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            write(journal/'allocation-validated.json', dict(job_id=str(job), plan_sha256=digest(plan),
                resources=plan['resources'], limits=plan['limits'], prior_actual_seconds=used,
                current_attempt_limit=seconds, work_deadline_unix=deadline))
            state = BranchState(root)
            snap = state.snapshot()
            if snap['halt'] or any(snap['queue'].get(k) for k in ('active','interrupted')):
                raise ValueError('Diagnose and preserve interrupted attempts before retrying')
            lease, base = lease_ports(job,bases=range(30000,54000,1024),size=640)
            urls = [f'http://127.0.0.1:{base+i*16}' for i in range(replicas)]
            for i in range(replicas):
                servers.append(spawn([str(RUNTIME/'venv/bin/python'),'-m','sglang.launch_server',
                    '--model-path',plan['worker_config']['actor_checkpoint'],'--host','127.0.0.1',
                    '--port',str(base+i*16),'--dtype','bfloat16','--tp','1','--mem-fraction-static','.65',
                    '--context-length','32768','--max-running-requests','5','--chunked-prefill-size','4096',
                    '--disable-cuda-graph'], f'actor-{i}', {'CUDA_VISIBLE_DEVICES':devices[i]}))
            spawn(['nvidia-smi','--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw',
                '--format=csv','-l','15'], 'gpu')
            pending = set(range(replicas)); startup = min(deadline, time.time()+900)
            while pending:
                health('model_startup')
                if time.time() > startup: raise TimeoutError('Actor startup failed')
                for i in list(pending):
                    try:
                        with urllib.request.urlopen(urls[i]+'/get_model_info', timeout=2) as response: card=json.load(response)
                        if card.get('model_path') != plan['worker_config']['actor_checkpoint']: raise ValueError('Wrong actor')
                        write(journal/f'actor-{i}-identity.json', card); pending.remove(i)
                    except OSError: pass
                if pending: time.sleep(2)
            wandb_run = initialize_wandb(plan,journal,env)
            accepted, teachers = [], []
            for offset in range(0,len(plan['states']),parallel):
                if len(accepted) >= plan['states_target']: break
                groups = plan['states'][offset:offset+parallel]
                discoveries = []
                for i, group in enumerate(discovery_wave(plan,offset)):
                    item = next(x for x in plan['schedule'] if x['state']==group['id'] and x['mode']=='discover')
                    status,_ = state.item(item['item_id'])
                    if status['status'] == 'pending': discoveries.append(start_worker(item,i,urls[i%replicas]))
                    elif status['status'] != 'finished': raise ValueError('Discovery requires recovery')
                wait_processes(discoveries, 'discovering_prefixes')
                branch_groups = []
                for i, group in enumerate(groups):
                    items = [x for x in plan['schedule'] if x['state']==group['id']]
                    d = result(next(x for x in items if x['mode']=='discover'))
                    sd = root/'states'/group['id']; sd.mkdir(parents=True, exist_ok=True)
                    if not d['captured']:
                        if not (sd/'reject.json').exists(): write(sd/'reject.json', dict(reason=d['reason'], stage='discovery'))
                        continue
                    if (sd/'reject.json').exists(): continue
                    anchor = Path(d['result_path']).parent/'anchor.json'
                    if not (sd/'anchor-reference.json').exists(): write(sd/'anchor-reference.json', dict(path=str(anchor),sha256=sha(anchor)))
                    branch_items = sorted((x for x in items if x['mode']=='branch'),
                                          key=lambda x:(x['candidate'],x['repeat']))
                    if (sd/'release.json').exists():
                        for item in branch_items: result(item)
                        accepted.append(group['id']); continue
                    slot_base = i*len(branch_items)
                    workers = [start_worker(branch_items[0],slot_base,urls[slot_base%replicas])]
                    branch_groups.append((group, sd, anchor, branch_items, slot_base, workers))
                unresolved = list(branch_groups)
                while unresolved:
                    health('reconstructing_and_continuing')
                    expand_ready_preflights(unresolved,state,start_worker,urls)
                    for row in list(unresolved):
                        group, sd, anchor, items, slot_base, workers = row
                        attempts = [state.item(x['item_id'])[1] for x in items]
                        dirs = [a[-1]['artifact_directory'] for a in attempts if a]
                        ready = len(dirs)==len(items) and all((Path(d)/'ready.json').exists() for d in dirs)
                        if ready:
                            if len(accepted)<plan['states_target']:
                                state.release(group['id'],sha(anchor),dirs); accepted.append(group['id'])
                            else: write(sd/'reject.json',dict(reason='Prespecified target already reached'))
                            unresolved.remove(row)
                        elif any(p.poll() is not None for p in workers):
                            if any(p.poll() not in (None,0) for p in workers): raise RuntimeError('Replay worker crashed')
                            write(sd/'reject.json',dict(reason='At least one reconstruction failed before release'))
                            unresolved.remove(row)
                        else:
                            # Keep prespecified task order when the target is
                            # reached; readiness speed must not choose the cohort.
                            break
                    if unresolved: time.sleep(1)
                wait_processes([p for row in branch_groups for p in row[-1]], 'continuations')
                # Prior teachers run concurrently with this batch's browser work.
                wait_processes(teachers, 'teacher_closeout'); teachers=[]
                for group in groups:
                    if group['id'] not in accepted: continue
                    item = next(x for x in plan['schedule'] if x['state']==group['id'] and x['mode']=='teacher')
                    status,_=state.item(item['item_id'])
                    if status['status']=='pending':
                        teachers.append(spawn([str(RUNTIME/'venv/bin/python'),'-m','openwebrl.arm_branch_teacher',
                            '--root',str(root),'--item',item['item_id']],item['item_id'],{'CUDA_VISIBLE_DEVICES':''}))
                pointer(root/'accepted-states.json', dict(state_ids=accepted, considered=offset+len(groups)))
                wandb_run.log(dict(accepted_states=len(accepted), considered_states=offset+len(groups)))
                handoff = root/'handoff-request.json'
                if handoff.exists() and read(handoff).get('job_id') == str(job):
                    wait_processes(teachers,'handoff_teachers'); teachers=[]
                    write(journal/'handoff-ready.json',dict(job_id=str(job),accepted_states=accepted,
                        considered=offset+len(groups),request=read(handoff),all_batch_workers_finished=True))
                    break
                if offset+len(groups)>=plan['reconstruction_gate_states'] and not accepted:
                    write(root/'reconstruction-gate-failed.json',dict(considered=offset+len(groups),accepted=0,
                        reason='Stop before scaling unreliable live reconstruction'))
                    break
            wait_processes(teachers,'final_teachers')
            completed = len(accepted)==plan['states_target']
            write(journal/'collection-closeout.json',dict(collection_target_met=completed,verified_complete=False,
                accepted_states=accepted,required_continuations=plan['states_target']*plan['candidate_count']*plan['continuation_repetitions'],independent_final_audit_required=True))
            pointer(root/'current.json',dict(stage='collected_pending_audit' if completed else 'partial_reconstruction_or_budget',
                job_id=str(job),accepted_states=len(accepted),updated_unix=time.time(),**state.snapshot()))
        finally:
            direct = stop_owned(children,job)
            browsers = cleanup_browser_groups(root,job)
            clean = all(x['process_gone'] for x in direct+browsers)
            write(journal/'owned-process-teardown.json',dict(verified=clean,direct=direct,browsers=browsers))
            if state is not None: state.mark_interrupted_after_teardown(teardown_verified=clean,reason='branch controller closeout')
            if wandb_run is not None: wandb_run.finish(exit_code=0 if completed else 1)
            if lease is not None: lease.close()
            for handle in handles: handle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--root',required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dry-run',action='store_true'); group.add_argument('--execute')
    args=parser.parse_args()
    if args.dry_run:
        plan=validate(args.root)
        print(json.dumps(dict(prepared=True,launch_authorized=read(Path(args.root)/'approval.json')['approved'],
            resources=plan['resources'],limits=plan['limits'],states=plan['states_target'],
            continuations=plan['states_target']*plan['candidate_count']*plan['continuation_repetitions']),indent=2))
    else: execute(args.root,args.execute)


if __name__=='__main__': main()
