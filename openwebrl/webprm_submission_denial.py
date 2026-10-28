"""Validate independently reconciled non-job Slurm denials without erasing intents.

This helper never submits, queries Slurm, modifies an intent or invents a job ID.
It admits only a hash-bound denial attestation plus two independently captured
queue/accounting snapshot pairs at least 60 seconds apart, covering the exact
nonce and submission interval. Ambiguous transports remain blocked.
"""
from __future__ import annotations

from datetime import datetime,timezone
from pathlib import Path
import re
import time

from . import webprm_cms_hosting as h


def epoch(value):
    parsed=datetime.fromisoformat(value)
    if parsed.tzinfo is None:parsed=parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def verified_denial(root,intent_path,sidecar_path):
    root=h.private_path(root);intent_path=h.private_path(intent_path)
    sidecar=h.read(sidecar_path)
    if (intent_path.parent!=root or sidecar.get('schema')!='webprm_submission_denial_v1' or
            sidecar.get('status')!='scheduler_rejected_no_job_independently_reconciled' or
            sidecar.get('job_id','missing') is not None or
            h.reference(sidecar['intent'])!=intent_path):
        raise h.Closed('Explicit hash-bound non-job denial required')
    intent=h.read(intent_path);nonce=intent.get('nonce')
    if not isinstance(nonce,str) or not re.fullmatch('[a-f0-9]{24}',nonce):raise h.Closed('Bounded exact submission nonce required')
    if intent_path.name!='submission-intent-'+nonce+'.json' or Path(sidecar_path).name!='submission-denial-'+nonce+'.json':
        raise h.Closed('Exact intent and denial nonce filenames required')
    comment=next((arg.removeprefix('--comment=') for arg in intent['argv'] if arg.startswith('--comment=')),None)
    if comment is None or not comment.endswith(':'+nonce):raise h.Closed('Exact denied submission comment required')
    proof=h.read(h.reference(sidecar['reconciliation']))
    if (proof.get('schema')!='webprm_nonjob_submission_reconciliation_v1' or
            h.reference(proof['intent'])!=intent_path or proof.get('nonce')!=nonce or proof.get('comment')!=comment or
            proof.get('intent_mtime_ns')!=intent_path.stat().st_mtime_ns):
        raise h.Closed('Independent reconciliation does not bind the raw intent')
    attestation=h.read(h.reference(proof['root_denial_attestation']))
    if (attestation.get('status')!='approved_submissions_blocked_by_host_account_budget' or
            attestation.get('no_new_job_ids') is not True or root.parent.name not in attestation.get('rejected_packages',[])):
        raise h.Closed('Explicit root-observed host denial attestation required')
    if 'captured_root_submission' in proof:
        captured=h.read(h.reference(proof['captured_root_submission']))
        if (type(captured.get('returncode')) is not int or captured['returncode']==0 or captured.get('stdout')!='' or
                not isinstance(captured.get('stderr'),str) or comment not in captured['stderr'] or
                'over-budget' not in captured['stderr'] or 'Batch job submission failed' not in captured['stderr'] or
                intent['approval']['path'] not in captured.get('argv',[])):
            raise h.Closed('Captured non-job host denial does not match the exact submission')
    # This is an attestation of observed scheduler stderr, never mislabeled a
    # captured-byte stderr transcript. Absence evidence independently corroborates it.
    batch=Path(intent['batch_script']).read_text()
    names=re.findall(r'^#SBATCH --job-name=(\S+)$',batch,re.M)
    if len(names)!=1:raise h.Closed('One exact expected batch name required')
    start=intent_path.stat().st_mtime
    pairs=proof.get('observation_pairs')
    if not isinstance(pairs,list) or len(pairs)!=2:raise h.Closed('Two queue/accounting observation pairs required')
    windows=[]
    for pair in pairs:
        if set(pair)!={'squeue','sacct'}:raise h.Closed('Both scheduler views required')
        times=[]
        for kind in ('squeue','sacct'):
            observation=h.read(h.reference(pair[kind]))
            begin,end=epoch(observation['started_utc']),epoch(observation['completed_utc'])
            if (observation.get('returncode')!=0 or observation.get('stderr') or observation.get('tz')!='UTC' or
                    begin<start or end<begin or end>time.time()+1):raise h.Closed('Successful UTC scheduler observations after intent required')
            argv=observation['argv']
            if kind=='squeue':
                expected=['squeue','-u','zixianma','-h','-o','%i|%V|%j|%k']
                submit_index,name_index,fields=1,2,4
            else:
                expected=['sacct','-u','zixianma','-S',argv[4],'-E','now','-X','-P','-n','-o',
                          'JobIDRaw,Submit,Start,End,State,JobName,Comment%150']
                if epoch(argv[4])>start-60:raise h.Closed('Accounting interval must cover before intent')
                submit_index,name_index,fields=1,5,7
            if argv!=expected:raise h.Closed('Unfiltered exact user queue/accounting command required')
            stdout=observation['stdout']
            if nonce in stdout or comment in stdout:raise h.Closed('Scheduler has a matching submission nonce')
            for line in stdout.splitlines():
                if not line:continue
                columns=line.split('|')
                if len(columns)!=fields:raise h.Closed('Malformed scheduler absence evidence')
                if columns[name_index]==names[0] and epoch(columns[submit_index])>=start-60:
                    raise h.Closed('Scheduler has an unaccounted matching batch name/time')
            times.extend((begin,end))
        windows.append((min(times),max(times)))
    if windows[1][0]-windows[0][1]<60:raise h.Closed('Independent observations must be at least 60 seconds apart')
    if proof.get('matched_job_ids')!=[]:raise h.Closed('Non-job denial cannot contain scheduler IDs')
    return dict(intent=str(intent_path),nonce=nonce,job_id=None,allocations_created=0,
                allocation_seconds_consumed=0,raw_intent_preserved=True)
