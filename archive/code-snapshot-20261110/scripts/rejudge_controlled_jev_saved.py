#!/usr/bin/env python3
"""Repair Jev judge inputs using only immutable saved operations and terminal PNGs.

--prepare-manifest is offline/read-only apart from its new output file. The owner
archives the source/DB, supersedes every selected attempt, and reopens only those
queue items. --execute requires that exact manifest bound into the approved plan
under saved_rejudge, plus the existing registered CPU allocation and total caps.
No browser, Jev, typing helper, actor generation, or episode replay is permitted.
"""
from __future__ import annotations
import argparse
import asyncio
from copy import deepcopy
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from openwebrl.controlled_jev_eval import JevBudget, Transport, canonical_score, private_exception_details
from openwebrl.controlled_jev_state import JevState
from openwebrl.controlled_sft_state import digest
from openwebrl.controlled_sft_worker import AttemptJournal
from run_controlled_jev_actor import (RUNTIME, read, sha, write, validate_frozen,
    scheduler_accounting, check_approval, validate_allocation, teardown_owned)


def immutable_json(path, value):
    path = Path(path)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()+b'\n'
    with path.open('xb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest())


def checked(root, proof):
    path = Path(proof['path']).resolve()
    if (root/'attempts') not in path.parents or not path.is_file() or sha(path) != proof['sha256']:
        raise ValueError('Saved evidence escaped its attempt root or changed: '+str(path))
    return path


def load_evidence(root, entry, plan):
    root = Path(root).resolve()
    result_path = checked(root, entry['result'])
    event_path = checked(root, entry['native_final_state'])
    image_path = checked(root, entry['final_screenshot'])
    directory = root/'attempts'/entry['source_attempt_id']
    if any(directory not in p.parents for p in (result_path,event_path,image_path)):
        raise ValueError('Saved evidence belongs to another source attempt')
    result, event = read(result_path), read(event_path)
    native = event.get('record', {})
    if (event.get('event') != 'native_final_state' or native.get('terminal') != 'done'
            or result.get('terminal') != 'done' or result.get('valid') is not True
            or result.get('attempt_id') != entry['source_attempt_id'] or result.get('task_id') != entry['task_id']
            or result.get('condition') != 'L06' or result.get('category') != entry['category']
            or result.get('browser_closed') is not True or result.get('actor_error') is not None
            or native.get('cleanup_errors') or result.get('cleanup_errors')
            or native.get('final_screenshot') != result.get('final_screenshot')
            or result.get('final_screenshot') != entry['final_screenshot']):
        raise ValueError('Only originally valid native DONE evidence can be rejudged')
    operations = native.get('operations')
    if not isinstance(operations,list) or result.get('actions') != len(operations)+1:
        raise ValueError('Saved operation count changed')
    task = next(t for t in plan['tasks'] if t['task_id'] == entry['task_id'])
    if digest(task) != entry['task_sha256']:
        raise ValueError('Frozen task identity changed')
    image = image_path.read_bytes()
    from PIL import Image
    with Image.open(io.BytesIO(image)) as png:
        if png.format != 'PNG' or png.size != (1280,720): raise ValueError('Expected saved1280x720 terminal PNG')
        png.verify()
    return result, native, image, task


def prepare_manifest(root, output):
    root = Path(root).resolve(); plan = read(root/'plan.json')
    db = sqlite3.connect('file:'+str(root/'state.sqlite3')+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        if db.execute("SELECT 1 FROM queue WHERE status!='finished'").fetchone():
            raise ValueError('Finish browser collection before preparing saved-only rejudge')
        if db.execute("SELECT 1 FROM attempts WHERE status='active'").fetchone():
            raise ValueError('A collection worker is still active')
        if db.execute("SELECT 1 FROM counters WHERE key='active_browsers' AND value!=0").fetchone():
            raise ValueError('Browser teardown is not complete')
        if db.execute("SELECT 1 FROM reservations WHERE status='reserved'").fetchone():
            raise ValueError('Unsettled prior requests need diagnosis before recovery')
        rows = db.execute("SELECT q.item_id,q.task_id,q.category,a.attempt_id,a.result_json FROM queue q JOIN attempts a USING(item_id) WHERE a.status='finished' ORDER BY q.position").fetchall()
        if len(rows) != len(plan['schedule']) or len({r['item_id'] for r in rows}) != len(rows):
            raise ValueError('Expected exactly one finished result per scheduled item')
        entries = []
        for row in rows:
            descriptor = json.loads(row['result_json'])
            if descriptor.get('valid') is not True or descriptor.get('terminal') != 'done': continue
            proof = dict(path=descriptor['result_path'],sha256=descriptor['result_sha256'])
            path = checked(root,proof)
            event_paths = [p for p in (path.parent/'events').glob('[0-9]*.json')
                           if read(p).get('event') == 'native_final_state']
            if len(event_paths) != 1: raise ValueError('Exactly one saved terminal event required')
            task = next(t for t in plan['tasks'] if t['task_id'] == row['task_id'])
            entry = dict(item_id=row['item_id'],task_id=row['task_id'],category=row['category'],
                source_attempt_id=row['attempt_id'],task_sha256=digest(task),result=proof,
                native_final_state=dict(path=str(event_paths[0]),sha256=sha(event_paths[0])),
                final_screenshot=descriptor['final_screenshot'])
            result,*_ = load_evidence(root,entry,plan)
            if any(result.get(k) != descriptor.get(k) for k in result):
                raise ValueError('Saved result differs from the committed descriptor')
            entries.append(entry)
        manifest = dict(schema_version=1,purpose='canonical_native_judge_input_repair',root=str(root),
            source_plan_sha256=digest(plan),canonical_reward_sha256=plan['worker_config']['canonical_reward_sha256'],
            complete_source_items=len(rows),selected_count=len(entries),selection='all originally valid native DONE, regardless of old score',
            external_calls='canonical_judge_only',new_browser_episodes=0,entries=entries)
        return immutable_json(output,manifest)
    finally: db.close()


def validate_manifest(root, plan):
    root = Path(root).resolve(); binding = plan['saved_rejudge']
    if binding['controller_sha256'] != sha(__file__): raise ValueError('Saved-only controller changed')
    path = Path(binding['input_manifest']).resolve()
    if root not in path.parents or sha(path) != binding['input_manifest_sha256']:
        raise ValueError('Rejudge manifest changed or escaped the cohort')
    manifest = read(path)
    entries = manifest['entries']
    if (manifest.get('root') != str(root) or manifest.get('external_calls') != 'canonical_judge_only'
            or manifest.get('new_browser_episodes') != 0 or manifest['selected_count'] != len(entries)
            or manifest['complete_source_items'] != len(plan['schedule'])
            or manifest['canonical_reward_sha256'] != plan['worker_config']['canonical_reward_sha256']
            or len({e['item_id'] for e in entries}) != len(entries)
            or len({e['source_attempt_id'] for e in entries}) != len(entries)):
        raise ValueError('Saved-only manifest coverage/identity changed')
    schedule = {r['item_id']:r for r in plan['schedule']}
    for entry in entries:
        item = schedule[entry['item_id']]
        if any(item[k] != entry[k] for k in ('task_id','category')) or item['condition'] != 'L06':
            raise ValueError('Manifest item differs from the frozen queue')
        load_evidence(root,entry,plan)
    return manifest


def validate_recovery_queue(state, manifest):
    entries = {e['item_id']:e for e in manifest['entries']}
    with state.connect() as db:
        for row in db.execute('SELECT * FROM queue'):
            if row['item_id'] not in entries:
                if row['status'] != 'finished': raise ValueError('Rejudge cannot collect a new browser task')
                latest = db.execute("SELECT result_json FROM attempts WHERE item_id=? AND status='finished'",(row['item_id'],)).fetchall()
                if len(latest) != 1: raise ValueError('Unselected item needs exactly one preserved result')
                preserved = json.loads(latest[0]['result_json'])
                if preserved.get('valid') is True and preserved.get('terminal') == 'done':
                    raise ValueError('Manifest omitted an originally valid DONE item')
                continue
            entry = entries[row['item_id']]
            source = db.execute('SELECT * FROM attempts WHERE attempt_id=?',(entry['source_attempt_id'],)).fetchone()
            if not source or source['item_id'] != row['item_id'] or source['status'] != 'superseded_judge_input':
                raise ValueError('Original attempt must be archived and explicitly superseded')
            original = json.loads(source['result_json'])
            if original['result_path'] != entry['result']['path'] or original['result_sha256'] != entry['result']['sha256']:
                raise ValueError('Superseded result identity differs from input manifest')
            if row['status'] == 'finished':
                latest = db.execute("SELECT result_json FROM attempts WHERE item_id=? AND status='finished'",(row['item_id'],)).fetchall()
                if len(latest) != 1: raise ValueError('A completed rejudge needs exactly one result')
                result = json.loads(latest[0]['result_json'])
                if result.get('saved_rejudge',{}).get('source_attempt_id') != entry['source_attempt_id']:
                    raise ValueError('Completed result is not this saved-only rejudge')
            elif row['status'] not in ('pending','active'):
                raise ValueError('Interrupted rejudges require explicit owner recovery')


class JudgeOnlyBudget(JevBudget):
    def reserve(self, kind, count=1, **context):
        if kind != 'om2w_judge_http_attempts': raise ValueError('Saved-only repair forbids browser/actor/helper reservations')
        return super().reserve(kind,count,**context)


class JudgeOnlyTransport(Transport):
    def __init__(self,budget,journal):
        self.budget,self.journal = budget,journal
        self.counts = dict(jev=0,text=0,judge=0)
    def request(self,*args,**kwargs): raise ValueError('Saved-only repair has no actor/helper transport')
    def post(self,*args,**kwargs): raise ValueError('Saved-only repair has no actor/helper transport')


def rejudge_claim(claim, plan, state, entry):
    if any(claim[k] != entry[k] for k in ('task_id','category')) or claim['condition'] != 'L06':
        raise ValueError('Claim differs from the bound saved record')
    result,native,image,task = load_evidence(state.root,entry,plan)
    journal = AttemptJournal(Path(claim['artifact_directory'])/'events')
    budget = JudgeOnlyBudget(state,claim); transport = JudgeOnlyTransport(budget,journal)
    started = time.monotonic()
    journal('saved_rejudge_started',dict(source_entry=entry,original_elapsed_seconds=result['elapsed_seconds'],
        original_api_attempts=result['api_attempts'],new_browser_episodes=0))
    image_artifact = journal.blob('final.png',image)
    copied_native = deepcopy(native); copied_native['final_screenshot'] = image_artifact
    journal('native_final_state',copied_native)
    score,verdict,status = asyncio.run(canonical_score(task,native['operations'],'done',image,
        transport,budget,journal,plan['worker_config']))
    repaired = dict(result,attempt_id=claim['attempt_id'],valid=score is not None,score=score,reward=score,
        judge=verdict,native_status=status,final_screenshot=image_artifact,
        api_attempts=transport.counts,elapsed_seconds=time.monotonic()-started,
        halt_required=bool(state.snapshot()['halt']),
        saved_rejudge=dict(source_attempt_id=entry['source_attempt_id'],source_result=entry['result'],
            source_native_final_state=entry['native_final_state'],source_final_screenshot=entry['final_screenshot'],
            source_actor_api_attempts={k:v for k,v in result['api_attempts'].items() if k!='judge'},
            source_episode_elapsed_seconds=result['elapsed_seconds'],actor_replayed=False,browser_replayed=False,
            corrected_adapter_sha256=sha(SOURCE/'openwebrl/controlled_jev_eval.py'),
            input_manifest_sha256=plan['saved_rejudge']['input_manifest_sha256']))
    artifact = immutable_json(Path(claim['artifact_directory'])/'result.json',repaired)
    return dict(repaired,result_path=artifact['path'],result_sha256=artifact['sha256'])


def worker(root, worker_id):
    root = Path(root).resolve(); plan = validate_frozen(root); manifest = validate_manifest(root,plan)
    state = JevState(root); state.require_approved_allocation()
    by_task = {(e['task_id'],e['category']):e for e in manifest['entries']}
    while claim := state.claim(worker_id):
        try:
            entry = by_task[(claim['task_id'],claim['category'])]
            result = rejudge_claim(claim,plan,state,entry)
            state.finish(claim['attempt_id'],result)
            if result['halt_required']: raise RuntimeError('Judge repair halted with preserved evidence')
        except BaseException as exc:
            proof = AttemptJournal(Path(claim['artifact_directory'])/'events')('saved_rejudge_failed',private_exception_details(exc))
            state.halt('Saved-only judge repair failed',attempt_id=claim['attempt_id'],evidence_path=proof)
            raise


def judge_environment():
    from dotenv import dotenv_values
    values = {**dotenv_values('/gpfs/projects/krishna/zixianma/OpenWebRL/.env'),**os.environ}
    key = values.get('JUDGE_API_KEY') or values.get('OPENAI_API_KEY')
    if not key: raise ValueError('Canonical judge credential required')
    env = {k:v for k,v in os.environ.items() if k not in ('TYPESAFE_API_KEY','JEV_API_KEY','TEXT_MODEL_API_KEY','OPENAI_API_KEY')}
    env.update({k:v for k,v in values.items() if k.startswith('WANDB_') and v})
    env.update(JUDGE_API_KEY=key,WANDB_PROJECT='openwebrl-evals',CUDA_VISIBLE_DEVICES='',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(SOURCE)+':'+str(RUNTIME/'src/Megatron-LM'))
    return env


def execute(root, job):
    root = Path(root).resolve(); plan = validate_frozen(root); manifest = validate_manifest(root,plan)
    approval = read(root/'approval.json')
    if os.getenv('SLURM_JOB_ID') != str(job) or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Run inside the registered original-budget allocation')
    used,old = scheduler_accounting(approval,job); seconds = check_approval(plan,approval,job,used)
    left = validate_allocation(subprocess.check_output(['scontrol','show','job',str(job),'-o'],text=True),job,seconds)
    deadline=time.monotonic()+left; env=judge_environment()
    journal=root/'scheduler-attempts'/str(job); journal.mkdir(parents=True,exist_ok=True)
    children,logs,state,wandb_run,completed=[],[],None,None,False
    def stop(signum,frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (root/'controller.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            write(journal/'prior-accounting.json',dict(prior_seconds=used,attempts=old))
            write(journal/'allocation-validated.json',dict(job_id=str(job),plan_sha256=digest(plan),
                resources=plan['resources'],limits=plan['limits'],prior_actual_seconds=used,
                current_attempt_limit=seconds,work_deadline_unix=time.time()+left,mode='saved_only_rejudge'))
            state=JevState(root);initial=state.snapshot();validate_recovery_queue(state,manifest)
            if initial['halt'] or initial['queue'].get('active') or initial['queue'].get('interrupted'):
                raise RuntimeError('Preserved interrupted/failed work requires owner diagnosis')
            before={k:initial['counters'].get(k,0) for k in ('browser_sessions','jev_requests','text_http_attempts','text_usd')}
            import wandb
            wandb_run=wandb.init(project='openwebrl-evals',id=plan['experiment_id'],resume='allow',
                config=dict(protocol=plan['protocol_name'],mode='saved_only_rejudge',plan_sha256=digest(plan)),
                dir=str(journal),settings=wandb.Settings(init_timeout=90,api_key=env.get('WANDB_API_KEY')))
            for phase in ('smoke','primary'):
                with state.connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    db.execute("UPDATE metadata SET value=? WHERE key='phase'",(phase,))
                    state._event(db,'saved_rejudge_phase',dict(phase=phase));db.commit()
                workers=[]
                for i in range(plan['resources']['collectors']):
                    log=(journal/f'rejudge-{phase}-worker-{i}.log').open('a');logs.append(log)
                    process=subprocess.Popen([str(RUNTIME/'venv/bin/python'),__file__,'--root',str(root),
                        '--worker-id',f'{job}-rejudge-{phase}-{i}'],cwd=SOURCE,env=env,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    children.append(process);workers.append(process)
                while any(p.poll() is None for p in workers):
                    snapshot=state.snapshot()
                    write(root/'current.json',dict(stage='saved_rejudge_'+phase,job_id=job,updated_unix=time.time(),**snapshot))
                    if snapshot['halt'] or any(p.poll() not in (None,0) for p in workers): raise RuntimeError('Saved rejudge worker failed')
                    if time.monotonic()>deadline: raise TimeoutError('Original approved wall-clock boundary')
                    wandb_run.log({'saved_rejudge/finished_total':snapshot['counts_by_condition'].get('L06',0)})
                    time.sleep(5)
                if any(p.returncode for p in workers): raise RuntimeError('Saved rejudge worker failed')
            snapshot=state.snapshot();validate_recovery_queue(state,manifest)
            if (snapshot['queue']!={'finished':len(plan['schedule'])} or snapshot['counters']['active_browsers']
                    or snapshot['unsettled_reservations']): raise ValueError('Unresolved saved-only evidence/accounting remains')
            if any(snapshot['counters'].get(k,0)!=v for k,v in before.items()): raise ValueError('Saved-only repair consumed forbidden actor/browser budget')
            with state.connect() as db:
                for row in db.execute("SELECT result_json FROM attempts WHERE status='finished'"):
                    result=json.loads(row['result_json'])
                    if sha(result['result_path'])!=result['result_sha256']: raise ValueError('Saved result changed')
            completed=True
            write(root/'collection-complete.json',dict(collection_complete=True,verified_complete=False,
                primary_episodes=300,smoke_episodes=3,job_id=job,saved_rejudge_count=len(manifest['entries']),
                input_manifest_sha256=plan['saved_rejudge']['input_manifest_sha256'],final_independent_audit_required=True))
        finally:
            try:
                teardown=teardown_owned(children,job,wandb_run,0 if completed else 1)
                write(journal/'owned-process-teardown.json',teardown)
                if state is not None: state.mark_interrupted_after_teardown(teardown_verified=teardown['verified'],reason='Saved-only judge repair closeout')
            finally:
                for log in logs:log.close()
                raw=subprocess.run(['sacct','-X','-n','-P','-j',str(job),'--format=JobIDRaw,State,ElapsedRaw,AllocTRES'],capture_output=True,text=True)
                write(journal/'scheduler-accounting.json',dict(sacct=raw.stdout,prior_actual_seconds=used,
                    current_attempt_limit=seconds,final_scheduler_accounting_required=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-manifest',type=Path);mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--execute');mode.add_argument('--worker-id')
    args=parser.parse_args()
    if args.prepare_manifest: print(json.dumps(prepare_manifest(args.root,args.prepare_manifest)))
    elif args.dry_run:
        plan=validate_frozen(args.root);manifest=validate_manifest(args.root,plan)
        print(json.dumps(dict(saved_only_preflight_passed=True,selected_count=len(manifest['entries']),new_browsers=0,
            resources=plan['resources'],limits=plan['limits'],final_independent_audit_required=True)))
    elif args.worker_id: worker(args.root,args.worker_id)
    else:execute(args.root,args.execute)


if __name__=='__main__':main()
