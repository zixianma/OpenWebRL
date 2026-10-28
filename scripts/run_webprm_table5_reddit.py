#!/usr/bin/env python3
"""Exact-context original Reddit build/copy/service stages for the future queue.

Reuses unchanged validated original service and preparation primitives. It cannot
allocate resources and rejects missing/unapproved/stale parent context first.
"""
import argparse
import math
import os
from pathlib import Path
import secrets
import sys
import time
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.launch_webprm_table5_campaign import (BROWSER,RUNTIME,SCRATCH,ORIGINAL_MANIFEST,
    read,sha,digest,checked,reference,validate_proposal,cell_key,DISPOSAL_PROTOCOL)
from openwebrl.webprm_lifecycle import _process


def context(path):
    value=read(checked(path));proposal=value['proposal'];validate_proposal(proposal)
    if (value['proposal_sha256']!=digest(proposal) or value['approval'].get('status')!='approved'
            or value['approval'].get('proposal_sha256')!=digest(proposal)
            or str(value['job_id'])!=os.environ.get('SLURM_JOB_ID')
            or value['controller_sha256']!=sha(REPO/'scripts/launch_webprm_table5_campaign.py')
            or not math.isfinite(value['deadline']) or value['deadline']<=time.time()):
        raise ValueError('Stage requires exact current approved campaign context')
    owner=_process(value['owner_pid'])
    if owner['start_ticks']!=value['owner_start_ticks'] or owner['state']=='Z':
        raise ValueError('Owning campaign controller is stale')
    ancestor=os.getppid()
    for _ in range(32):
        if ancestor==value['owner_pid']:break
        if ancestor<=1:raise ValueError('Stage has no owning controller ancestor')
        ancestor=_process(ancestor)['ppid']
    else:raise ValueError('Stage is not owned by its campaign controller')
    scratch=checked(value['scratch']);output=checked(value['output'])
    if SCRATCH not in scratch.parents or Path(proposal['campaign_root']) not in output.parents:
        raise ValueError('Stage scratch/output differs from approved campaign')
    return value


def stage_paths(owned,mode,state,out):
    scratch=checked(owned['scratch']);output=checked(owned['output'])
    state,out=checked(state),checked(out)
    if scratch not in state.parents or output not in out.parents:
        raise ValueError('State/output must belong to this fresh owned attempt')
    if mode=='build':
        if state!=scratch/'baseline' or out!=output/'original-build':
            raise ValueError('Build must create the fixed private immutable baseline')
        return state,out
    if mode not in ('copy','serve'):
        raise ValueError('Unknown original-site stage')
    claim=read(checked(out.parent/'state-claim.json'))
    cells=read(owned['proposal']['cohort_path']);cells=cells.get('cohort',cells)['cells']
    cell=next((c for c in cells if c['cell_id']==claim.get('cell_id')),None)
    if (cell is None or cell['domain']!='reddit' or claim.get('protocol')!=DISPOSAL_PROTOCOL
            or claim.get('policy_sha256')!=digest(owned['proposal']['ephemeral_policy'])
            or claim.get('campaign_root')!=owned['proposal']['campaign_root']
            or claim.get('scratch_root')!=str(scratch) or claim.get('base_path')!=str(scratch/'baseline')
            or claim.get('state_path')!=str(state) or claim.get('destination_absent_at_claim') is not True
            or claim.get('preserve_historical_states') is not True
            or state!=scratch/'states'/cell_key(cell)
            or out!=output/'cells'/cell_key(cell)/('copy' if mode=='copy' else 'site')
            or mode=='copy' and state.exists() or mode=='serve' and not state.is_dir()):
        raise ValueError('Stage requires this fixed cell\'s fresh private state claim')
    return state,out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context',required=True);parser.add_argument('--mode',choices=('build','copy','serve'),required=True)
    parser.add_argument('--state',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--inside',action='store_true');parser.add_argument('--launch-token')
    args=parser.parse_args();owned=context(args.context)
    scratch=checked(owned['scratch']);state,out=stage_paths(owned,args.mode,args.state,args.output)
    reddit=owned['proposal']['domains']['reddit']
    if args.mode!='serve':
        if args.inside or args.launch_token:raise ValueError('Preparation has no namespace handoff')
        prepare=reference('prepare_reddit_original.py');prepare.SCRATCH=scratch
        kwargs=(dict(download_receipt=reddit['download_receipt']) if args.mode=='build' else
                dict(build_receipt=Path(owned['output'])/'original-build/receipt.json',verification=reddit['raw_verification']))
        return prepare.prepare(args.mode,state,out,owned['deadline'],**kwargs)
    site=reference('run_reddit_instance.py');site.SCRATCH=scratch
    provenance=out.parent/'copy/receipt.json'
    a=SimpleNamespace(sandbox=str(state),output=str(out),identity=ORIGINAL_MANIFEST,scope='original',
        deadline=owned['deadline'],state_provenance=str(provenance),startup_review=reddit['startup_review'],
        launch_token=args.launch_token)
    if args.inside:return site.inside(a)
    fields=site.provenance(a);out.mkdir(mode=0o700);a.launch_token=secrets.token_hex(24)
    claim=site.claim_state(a,a.launch_token)
    site.write(out/'launch.json',dict(sandbox=a.sandbox,output=a.output,scope=a.scope,image_sha256=a.identity,
        launch_token=a.launch_token,parent_pid=os.getpid(),parent_start_ticks=_process(os.getpid())['start_ticks'],
        outer_network_namespace=os.readlink('/proc/self/ns/net'),state_claim_path=claim,provenance=fields,
        service_adapter_sha256=sha(__file__),owner_context_path=args.context,owner_context_sha256=sha(args.context)))
    site.prepare(state,out)
    def child(_):
        os.setsid()
        command=['unshare','--user','--map-root-user','--net',BROWSER,'-B',str(Path(__file__).resolve()),
            '--context',args.context,'--mode','serve','--state',str(state),'--output',str(out),
            '--inside','--launch-token',a.launch_token]
        os.execvpe(command[0],command,site.safe_env())
    return site.supervise(a,target=child)


if __name__=='__main__':main()
