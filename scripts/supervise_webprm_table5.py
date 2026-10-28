#!/usr/bin/env python3
"""Read-only Slurm/artifact poller that queues the owning Table5 agent.

It never launches experiments, changes scheduler state, edits campaign artifacts,
loads a model or evaluates a browser. The queued agent owns diagnosis and repair.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011')
OM2W=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-om2w-table5-20261012')
POINTER=R/'current_table5_preparation.json'
ROOT=R/'table5-supervisor'
THREAD='01a124a3-446f-7f22-9d8d-bbb11119270a'
CODEX='/gpfs/scrubbed/zixianma/codex/packages/standalone/releases/0.162.1-x86_64-unknown-linux-musl/bin/codex'
CODEX_HOME='/gpfs/scrubbed/zixianma/codex'
TERMINAL={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','PREEMPTED','DEADLINE'}
ACTIVE={'RUNNING','COMPLETING','CONFIGURING'}
KINDS={'reddit','shopping','cms','gitlab','map','om2w','webprmbench'}
# A full 450-cell campaign retains many small per-cell receipts. The previous
# 2,500-file ceiling could reject its normal shallow inventory before closure.
MAX_ARTIFACT_FILES=12000


def checked(path):
    path=Path(path)
    if (not path.is_absolute() or path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents))
            or not any(base in path.parents for base in (R,OM2W,Path('/tmp')))):
        raise ValueError('Supervisor path must be canonical under either approved WebPRM runtime or /tmp')
    return path


def read(path):return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def write(path,value):
    path=checked(path);raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    if len(raw)>4*2**20:raise ValueError('Supervisor receipt exceeded four MiB')
    temporary=checked(path.with_name(path.name+'.partial'))
    with temporary.open('wb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
    temporary.replace(path)


def safe_environment(root):
    env=dict(os.environ)
    if env.get('CODEX_HOME')!=CODEX_HOME:raise ValueError('Required inherited CODEX_HOME differs; do not override it')
    for variable,name in [('XDG_CACHE_HOME','cache'),('XDG_CONFIG_HOME','config'),('XDG_DATA_HOME','data'),
            ('XDG_STATE_HOME','state'),('XDG_RUNTIME_DIR','runtime')]:
        path=checked(root/name);path.mkdir(mode=0o700,exist_ok=True);env[variable]=str(path)
    env.update(TMPDIR='/tmp',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HISTFILE='/dev/null',
        BASH_ENV='/dev/null',ENV='/dev/null')
    return env


def prompt(root,pointer):
    message=(f'[WebPRM Table5 active supervision] Read {pointer} and {root}/latest.json. '
        f'Ack: python3 -B scripts/supervise_webprm_table5.py --root {root} --ack. '
        'If verified_complete, silently ack stale callbacks; no repeated report. Follow current next_stage_jobs: '
        'check scheduler, logs, useful progress, budgets and closure. Actively diagnose, test and repair approved work; '
        'charge every attempt against all time/API/transfer caps. Launch successors only within existing exact approvals '
        'and remaining caps; no added allocations or budget. Preserve frozen science, raw/effective verdicts and evidence; '
        'update replacement pointers. Independently audit before verified_complete. Report meaningful changes; routine summaries at most hourly.')
    if len(message.encode())>1000:raise ValueError('Continuation prompt exceeds 1000 bytes')
    return message


def scheduler(jobs,run=None):
    run=subprocess.run if run is None else run
    ids=','.join(jobs)
    def query(argv):
        return run(argv,capture_output=True,text=True,timeout=20,env=dict(os.environ,LC_ALL='C'))
    def failure(command,result):
        raise RuntimeError(f'{command} failed ({result.returncode}): {result.stderr.strip()[:300]}')
    # Accounting must be collected even when a finished ID has aged out of
    # squeue. Absence alone establishes neither terminal state nor closure.
    a=query(['sacct','-X','-n','-P','-j',ids,'--format=JobIDRaw,State,ElapsedRaw,ExitCode'])
    if a.returncode:failure('sacct',a)
    rows={}
    for line in a.stdout.splitlines():
        fields=line.split('|')
        if fields[0] not in jobs:continue
        if (len(fields)!=4 or fields[0] in rows or not fields[1].split() or not fields[2].isdigit()
                or len(fields[3].split(':'))!=2 or not all(x.isdigit() for x in fields[3].split(':'))):
            raise ValueError('Exact unique Slurm accounting row required')
        rows[fields[0]]=dict(state=fields[1].split()[0].rstrip('+'),elapsed_seconds=int(fields[2]),exit_code=fields[3])
    def queued(result,requested):
        found={}
        for line in result.stdout.splitlines():
            fields=line.split('|')
            if len(fields)!=2 or fields[0] not in requested or not fields[1] or fields[0] in found:
                raise ValueError('Exact unique Slurm queue row required')
            found[fields[0]]=fields[1]
        return found
    q=query(['squeue','-h','-j',ids,'-o','%i|%T'])
    requested=jobs
    if q.returncode:
        expected='slurm_load_jobs error: Invalid job id specified'
        if q.returncode!=1 or q.stderr.strip()!=expected:failure('squeue',q)
        terminal={job for job,row in rows.items() if row['state'] in TERMINAL}
        if not terminal:raise RuntimeError('Expired queue ID lacks exact terminal accounting')
        requested=[job for job in jobs if job not in terminal]
        if requested:
            # A mixed active/expired query may return no usable queue output.
            # Requery only unresolved IDs once, retaining terminal accounting.
            q=query(['squeue','-h','-j',','.join(requested),'-o','%i|%T'])
            if q.returncode:failure('squeue unresolved IDs',q)
            live=queued(q,requested)
            if set(live)!=set(requested):raise RuntimeError('Missing queue ID lacks exact terminal accounting')
        else:live={}
    else:live=queued(q,requested)
    for job,state in live.items():rows.setdefault(job,dict(elapsed_seconds=0,exit_code='')).update(state=state)
    return {job:rows.get(job,dict(state='UNKNOWN',elapsed_seconds=0,exit_code='')) for job in jobs}


def artifacts(row):
    job=row['job_id'];package=checked(row['package']);campaign=checked(row.get('campaign_root',row['package']))
    roots={campaign/('attempt-'+job),package/('attempt-'+job)}
    for startup in package.glob('startup-*'):
        if startup.is_dir() and read(startup/'run-claim.json').get('job_id')==job:
            roots.add(checked(startup));roots.add(startup/('attempt-'+job))
    files=set()
    for root in roots:
        if root.exists():
            # Shallow stats cover download partials/ledgers and controller output;
            # Reddit per-call progress is also recorded in the shared journal.
            for pattern in ('*','*/*','*/*/*','cells/*/cell-outcome.json'):
                for path in root.glob(pattern):
                    if path.is_file() and not path.is_symlink():files.add(path)
                    if len(files)>MAX_ARTIFACT_FILES:raise ValueError('Bounded supervisor artifact inventory exceeded')
    for name in ('shared-budget.json.events.jsonl','shared-budget.json'):
        path=campaign/name
        if path.is_file():files.add(path)
    complete=package/'completed.json'
    if read(complete).get('job_id')==job:files.add(complete)
    if read(package/'run-claim.json').get('job_id')==job:
        failure=package/'failure.json'
        if failure.is_file():files.add(failure)
    pre_actor_failures=0
    for path in (campaign/'queue/outcomes').glob('*.json'):
        if path.is_symlink():raise ValueError('Symlink queue outcome')
        if path.stat().st_size>4*2**20:raise ValueError('Oversized queue outcome')
        outcome=read(path);cell_id=outcome.get('cell_id')
        if not isinstance(cell_id,str):raise ValueError('Queue outcome lacks cell identity')
        key=hashlib.sha256(cell_id.encode()).hexdigest()
        if not any((root/'cells'/key/'state-claim.json').is_file() and
                   not (root/'cells'/key/'worker-launch.json').exists() for root in roots):continue
        if outcome.get('result',{}).get('reason')=='worker_exception':
            pre_actor_failures+=1;files.add(path)
        if len(files)>MAX_ARTIFACT_FILES:raise ValueError('Bounded supervisor artifact inventory exceeded')
    stamps=[];failed=bool(pre_actor_failures);done=False;startup_failures=0
    for path in sorted(files):
        stat=path.stat();stamps.append([str(path),stat.st_size,stat.st_mtime_ns])
        failed|=path.name=='failure.json';done|=path.name in ('completed.json','queue-summary.json','stage-result.json')
        if row.get('kind')=='webprmbench' and path.name=='final.json' and path.parent in roots:
            if stat.st_size>4*2**20:raise ValueError('Oversized static benchmark final receipt')
            final=read(path)
            if type(final.get('verified_complete')) is not bool:
                raise ValueError('Static benchmark final receipt lacks explicit completion status')
            done=True
            failed|=not final['verified_complete']
        if path.name=='controller-closure.json':failed|=read(path).get('child_exitcode') not in (None,0)
        if path.name=='cell-outcome.json':
            outcome=read(path)
            if outcome.get('valid') is False and isinstance(outcome.get('result_path'),str):
                result_path=checked(outcome['result_path'])
                if path.parent/'episodes' not in result_path.parents:
                    raise ValueError('Cell outcome result escaped its episode directory')
                if result_path.is_file():
                    if result_path.stat().st_size>4*2**20:raise ValueError('Oversized startup result')
                    result=read(result_path)
                    if (result.get('error_type')=='EvaluationNotReady'
                            and str(result.get('error','')).startswith('Native authentication failed:')):
                        startup_failures+=1;failed=True
    return dict(fingerprint=hashlib.sha256(json.dumps(stamps).encode()).hexdigest(),files=len(stamps),
        latest_mtime=max((x[2]/1e9 for x in stamps),default=0),failure_artifact=failed,completion_artifact=done,
        native_authentication_failures=startup_failures,pre_actor_failures=pre_actor_failures)


def snapshot(pointer,old,now):
    pointer=checked(pointer)
    if not pointer.is_file():raise ValueError('Current Table5 registry is missing')
    current=read(pointer)
    if current.get('verified_complete') is True or current.get('status')=='verified_complete':
        return dict(epoch=now,verified_complete=True,jobs=[])
    jobs=[];seen=set()
    for row in current.get('next_stage_jobs',[]):
        if row.get('verified_complete') is True or row.get('status')=='verified_complete' or row.get('supervision_required') is False:continue
        job=row.get('job_id')
        if not isinstance(job,str) or not job.isdigit() or job in seen or row.get('kind') not in KINDS:
            raise ValueError('Invalid/duplicate current Table5 registry job')
        seen.add(job);jobs.append(row)
    if not jobs:return dict(epoch=now,verified_complete=False,jobs=[],no_registered_work=True)
    states=scheduler([r['job_id'] for r in jobs]);progress=old.get('progress',{});result=[]
    for row in jobs:
        job=row['job_id'];status=states[job];files=artifacts(row);previous=progress.get(job,{})
        last=previous.get('last_progress_epoch',max(files['latest_mtime'],now-status['elapsed_seconds']))
        if previous and previous.get('fingerprint')!=files['fingerprint']:last=now
        if status['state'] not in ACTIVE:last=now
        progress[job]=dict(fingerprint=files['fingerprint'],last_progress_epoch=last)
        result.append(dict(job_id=job,kind=row['kind'],package=row['package'],**status,**files,
            stalled=status['state'] in ACTIVE and now-last>=1200))
    return dict(epoch=now,verified_complete=False,jobs=result,progress={j:progress[j] for j in seen})


def decision(old,ack,current,now):
    signature=[[r['job_id'],r['state'],r['failure_artifact'],r['completion_artifact'],r['stalled'],
                r.get('native_authentication_failures',0),r.get('pre_actor_failures',0)]
               for r in current.get('jobs',[])]
    previous=old.get('signature');pending=old.get('queued_epoch',0)>ack
    changes={tuple(r) for r in signature}-{tuple(r) for r in previous or []}
    urgent=bool(changes and any(r[1] in ACTIVE|TERMINAL|{'UNKNOWN'} or r[2] or r[3] or r[4] for r in changes))
    # Routine owning-agent wakeups are hourly; polling and urgent transitions
    # remain independent so reduced chatter cannot delay failure recovery.
    initial=previous is None;due=now-max(old.get('queued_epoch',0),ack)>=3600
    # A successful enqueue is not proof of delivery. Retry an unacknowledged
    # callback hourly so one lost delivery cannot suppress recovery forever.
    overdue_pending=pending and now-old.get('queued_epoch',0)>=3600
    send=bool(signature) and (urgent or overdue_pending or not pending and (initial or due))
    return send,signature,urgent,pending


def queue_review(root,pointer,current,old,ack,now,run=subprocess.run):
    send,signature,urgent,pending=decision(old,ack,current,now)
    if send:
        result=run([CODEX,'queue','--thread',THREAD,'--message',prompt(root,pointer)],capture_output=True,text=True,
            timeout=45,env=safe_environment(root))
        if result.returncode:raise RuntimeError('Agent continuation queue failed: '+result.stderr[:300])
        old.update(queued_epoch=now,signature=signature,queue_receipt=result.stdout.strip(),
            notification_kind='urgent' if urgent else 'routine',pending_callback_bypassed=bool(pending and urgent))
        with checked(root/'notifications.jsonl').open('a') as out:out.write(json.dumps(old)+'\n');out.flush();os.fsync(out.fileno())
    return old


def poll(root,pointer,now=None):
    now=time.time() if now is None else now;old=read(root/'state.json');current=snapshot(pointer,old,now)
    write(root/'latest.json',current);write(root/'heartbeat.json',dict(epoch=now,pid=os.getpid(),thread=THREAD))
    if current.get('verified_complete') or current.get('no_registered_work'):
        write(root/'finished.json',dict(epoch=now,verified_complete=current['verified_complete'],no_registered_work=current.get('no_registered_work',False)))
        return False
    ack=read(root/'ack.json').get('epoch',0)
    old=queue_review(root,pointer,current,old,ack,now);old['progress']=current.get('progress',{});write(root/'state.json',old)
    return True


def loop(root,pointer,once,keep_alive=False):
    with checked(root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                if not poll(root,pointer):
                    if not keep_alive or read(pointer).get('verified_complete') is True:return
            except Exception as error:
                now=time.time();problem=dict(epoch=now,error_type=type(error).__name__,error=str(error)[:500]);write(root/'error.json',problem)
                write(root/'heartbeat.json',dict(epoch=now,pid=os.getpid(),thread=THREAD,status='poll_error'))
                try:
                    old=read(root/'state.json');current=dict(jobs=[dict(job_id='supervisor',state='UNKNOWN',failure_artifact=True,
                        completion_artifact=False,stalled=False)])
                    old=queue_review(root,pointer,current,old,read(root/'ack.json').get('epoch',0),now);write(root/'state.json',old)
                except Exception as queue_error:write(root/'queue-error.json',dict(epoch=now,error_type=type(queue_error).__name__,error=str(queue_error)[:300]))
                if once:raise
            if once:return
            time.sleep(60)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--pointer',type=Path,default=POINTER);parser.add_argument('--ack',action='store_true')
    parser.add_argument('--once',action='store_true');parser.add_argument('--daemon',action='store_true')
    parser.add_argument('--keep-alive',action='store_true',help='Remain available while unverified work awaits registration')
    args=parser.parse_args()
    root=checked(args.root);root.mkdir(mode=0o700,parents=True,exist_ok=True)
    if args.ack:write(root/'ack.json',dict(epoch=time.time()));return
    if args.daemon:
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(root),'--pointer',str(checked(args.pointer))]
        if args.keep_alive:command.append('--keep-alive')
        with checked(root/'process.log').open('a') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env=safe_environment(root))
        ticks=int(Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
        write(root/'service.json',dict(pid=process.pid,proc_start_ticks=ticks,thread=THREAD,started_epoch=time.time(),
            command=command,queue_delivery_verified=False,mechanism='poller plus owning Codex agent continuation'))
        print(json.dumps(dict(pid=process.pid,root=str(root),queue_delivery_verified=False)));return
    loop(root,checked(args.pointer),args.once,args.keep_alive)


if __name__=='__main__':main()
