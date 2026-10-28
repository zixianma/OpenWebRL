#!/usr/bin/env python3
"""Approval-bound entry point reusing the unchanged validated Reddit service code."""
import argparse
import math
import os
from pathlib import Path
import secrets
import sys
import time
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.launch_webprm_reddit_pilot import (BROWSER, RUNTIME, SCRATCH, ORIGINAL_MANIFEST,
    checked_path, digest, read, reference, sha, validate)
from openwebrl.webprm_lifecycle import _process


def context(path):
    value=read(checked_path(path,RUNTIME));proposal=validate(value['proposal'])
    if (value['proposal_sha256']!=digest(proposal) or value['approval'].get('status')!='approved' or
            value['approval'].get('proposal_sha256')!=digest(proposal) or
            str(value['job_id'])!=os.environ.get('SLURM_JOB_ID') or
            value['controller_sha256']!=sha(REPO/'scripts/launch_webprm_reddit_pilot.py')):
        raise ValueError('Service requires the current exact approved controller context')
    if not math.isfinite(value['deadline']) or value['deadline']<=time.time():raise ValueError('Service deadline expired')
    owner=_process(value['owner_pid'])
    if owner['start_ticks']!=value['owner_start_ticks'] or owner['state']=='Z':raise ValueError('Owning controller is stale')
    ancestor=os.getppid()
    for _ in range(12):
        if ancestor==value['owner_pid']:break
        ancestor=_process(ancestor)['ppid']
    else:raise ValueError('Service is not a descendant of its owning controller')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('context','sandbox','output','state-provenance'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--inside',action='store_true');parser.add_argument('--launch-token')
    args=parser.parse_args();owned=context(args.context)
    site=reference('run_reddit_instance.py');site.SCRATCH=checked_path(owned['scratch'],SCRATCH,directory=True)
    sandbox=checked_path(args.sandbox,site.SCRATCH,directory=True)
    output=checked_path(args.output,Path(owned['output']))
    a=SimpleNamespace(sandbox=str(sandbox),output=str(output),identity=ORIGINAL_MANIFEST,scope='original',
        deadline=owned['deadline'],state_provenance=str(checked_path(args.state_provenance,RUNTIME)),
        startup_review=owned['proposal']['startup_review'],launch_token=args.launch_token)
    if args.inside:return site.inside(a)
    fields=site.provenance(a);output.mkdir(exist_ok=False);a.launch_token=secrets.token_hex(24)
    claim=site.claim_state(a,a.launch_token)
    site.write(output/'launch.json',dict(sandbox=a.sandbox,output=a.output,scope=a.scope,image_sha256=a.identity,
        launch_token=a.launch_token,parent_pid=os.getpid(),parent_start_ticks=_process(os.getpid())['start_ticks'],
        outer_network_namespace=os.readlink('/proc/self/ns/net'),state_claim_path=claim,provenance=fields,
        service_adapter_sha256=sha(__file__),owner_context_path=args.context,owner_context_sha256=sha(args.context)))
    site.prepare(sandbox,output)
    def child(_):
        os.setsid()
        command=['unshare','--user','--map-root-user','--net',BROWSER,'-B',str(Path(__file__).resolve()),
            '--context',args.context,'--sandbox',a.sandbox,'--output',a.output,
            '--state-provenance',a.state_provenance,'--inside','--launch-token',a.launch_token]
        os.execvpe(command[0],command,site.safe_env())
    site.supervise(a,target=child)


if __name__=='__main__':main()
