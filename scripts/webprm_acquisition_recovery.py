"""Offline proof for bounded, manually prepared acquisition startup successors.

No scheduler submission, automatic retry, test execution, or data transfer. The
caller obtains the scheduler snapshot directly and binds all proof into its
next immutable proposal. Actual transfer claims permanently disable successors.
"""
import hashlib
import json
import os
from pathlib import Path


class Closed(ValueError): pass


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def file_sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recovery_proof(root, proposal, approval, launch, snapshot, repair):
    """Accept only a closed zero-network predecessor and a bound tested fix."""
    root=Path(root)
    if (approval.get('status')!='approved' or approval.get('proposal_sha256')!=digest(proposal)
            or not approval.get('exact_user_approval') or launch.get('proposal_sha256')!=digest(proposal)):
        raise Closed('Predecessor exact approval/launch identity is not intact')
    # This claim is created exclusively before constructing any network opener.
    # Also reject legacy ledgers even if a claim is absent due to an old source.
    if any(root.rglob('network-claim.json')) or any(root.rglob('transfer-ledger.json')) or (root/'completed.json').exists():
        raise Closed('Any network attempt or completed acquisition forbids startup recovery')
    previous=proposal.get('startup_recovery',{})
    number=previous.get('attempt_number',1)
    consumed=previous.get('prior_allocation_seconds',0)
    resources=proposal['resources']
    if (type(number) is not int or type(consumed) is not int or consumed<0 or
            number>=resources['maximum_allocation_attempts']):
        raise Closed('Startup attempt cap exhausted')
    try:
        if snapshot.get('errors')!=[] or len(snapshot['jobs'])!=1: raise ValueError()
        row=snapshot['jobs'][0]
        def finite(key):
            v=row[key]
            if v.get('set') is not True or v.get('infinite') is not False or type(v.get('number')) is not int:raise ValueError(key)
            return v['number']
        start,end=finite('start_time'),finite('end_time')
        requested=finite('time_limit')*60
        elapsed=end-start
        tres=dict(x.split('=',1) for x in row['tres_alloc_str'].split(','))
        if (row['job_state'] not in (['FAILED'],['CANCELLED'],['TIMEOUT'],['NODE_FAIL']) or
            row['user_id']!=os.getuid() or row['restart_cnt']!=0 or
            row['comment']!='webprm-acquire:'+launch['nonce'] or
            row['account']!=resources['account'] or row['partition']!=resources['partition'] or row['qos']!=resources['qos'] or
            finite('cpus')!=resources['cpus'] or finite('node_count')!=1 or
            finite('memory_per_node')!=resources['memory_gib']*1024 or
            tres.get('cpu')!=str(resources['cpus']) or tres.get('node')!='1' or
            tres.get('mem') not in (str(resources['memory_gib'])+'G',str(resources['memory_gib']*1024)+'M') or
            any(k.startswith('gres/gpu') and v!='0' for k,v in tres.items()) or
            requested!=launch['requested_allocation_seconds'] or not 0<start<=end or not 0<=elapsed<=requested+60):raise ValueError()
    except (KeyError,ValueError,TypeError,AttributeError) as e:raise Closed('Invalid closed scheduler identity/resource/time proof') from e
    # Charge the actual terminal duration, including scheduler grace, not just
    # useful worker time. Round the NEXT request down to whole Slurm minutes.
    consumed+=elapsed
    remaining=(resources['total_allocation_seconds_including_retries']-consumed)//60*60
    if remaining<=120:raise Closed('No useful cumulative allocation budget remains')
    if (repair.get('status')!='passed' or type(repair.get('test_exit_code')) is not int or repair['test_exit_code']!=0 or
            not isinstance(repair.get('diagnosis'),str) or not repair['diagnosis'].strip() or
            not isinstance(repair.get('test_argv'),list) or not repair['test_argv'] or
            repair.get('predecessor_proposal_sha256')!=digest(proposal) or
            not isinstance(repair.get('replacement_source_sha256'),dict) or not repair['replacement_source_sha256']):
        raise Closed('A diagnosed source fix and successful bound test receipt are required')
    if repair['replacement_source_sha256']==proposal['source_sha256']:
        raise Closed('Do not retry an unchanged startup failure')
    return dict(attempt_number=number+1,prior_allocation_seconds=consumed,
        requested_allocation_seconds=remaining,predecessor_root=str(root),
        predecessor_proposal_sha256=digest(proposal),predecessor_job_id=str(row['job_id']),
        predecessor_elapsed_seconds=elapsed,terminal_snapshot=snapshot,source_fix=repair,
        network_attempts=0,manual_successor=True)
