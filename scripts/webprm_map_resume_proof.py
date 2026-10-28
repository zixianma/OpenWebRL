"""Read-only admission proof for the one disclosed Map append exception."""
import hashlib
import os
from pathlib import Path
import pwd
import shlex
from scripts.webprm_map_resume import check_prefix,PROTOCOL,SCRATCH_RESERVE


def require(value,message):
    if not value:raise ValueError(message)


def proof(backend,previous,metadata,replacement_sources):
    read,sha,digest=backend.read,backend.sha,backend.digest
    p,auth,launch,claim,network,start=(read(previous/name) for name in
        ('proposal.json','approval.json','launch.json','run-claim.json','network-claim.json','scheduler.json'))
    audit=read(metadata/'prior-failure-audit.json');terminal=read(metadata/'prior-terminal-scheduler.json')
    prefix=read(metadata/'prior-prefix-proof.json');scratch=read(metadata/'prior-scratch-observation.json')
    require(p.get('resources')==backend.RESOURCES and p.get('limits')==backend.LIMITS and p.get('profile')==backend.PROFILE
        and p.get('site')=='map_backend' and p.get('startup_recovery')==dict(attempt_number=1,prior_allocation_seconds=0,requested_allocation_seconds=14400),
        'Only the original first Map allocation supports this exception')
    require(auth.get('status')=='approved' and isinstance(auth.get('exact_user_approval'),str) and auth['exact_user_approval'].strip()
        and auth.get('proposal_sha256')==digest(p)==launch.get('proposal_sha256') and not (previous/'completed.json').exists(),
        'Original Map approval changed or acquisition completed')
    for name,wanted in p['source_sha256'].items():
        require(sha(previous/name)==wanted,'Original Map source changed')
        if name.startswith('metadata/'):
            require(replacement_sources.get(name)==wanted,'Original Map metadata must remain pinned')
    job=claim['job_id']
    require(job.isdigit() and network.get('job_id')==job and claim.get('nonce')==launch.get('nonce'), 'Prior job/network claim differs')
    require(terminal.get('errors')==[] and len(terminal.get('jobs',[]))==1 and
        start.get('errors')==[] and len(start.get('jobs',[]))==1,'One closed scheduler epoch required')
    row,started=terminal['jobs'][0],start['jobs'][0];t=row['time'];elapsed=t['elapsed']
    require(row['job_id']==int(job) and row['user']==pwd.getpwuid(os.getuid()).pw_name and row['restart_cnt']==started['restart_cnt']==0
        and started['job_id']==int(job) and started['user_id']==os.getuid() and row['state']['current']==['FAILED']
        and row['account']=='zixianma' and row['partition']=='gpu-h200' and row['qos']=='normal'
        and row['working_directory']==str(previous) and shlex.split(row['submit_line'])==launch['argv']
        and started['comment']=='webprm-acquire:'+launch['nonce'],'Prior Map scheduler identity/resource ownership differs')
    require(type(elapsed) is int and 0<elapsed<14400 and t['end']-t['start']==elapsed
        and started['start_time']==dict(set=True,infinite=False,number=t['start'])
        and t['limit']==dict(set=True,infinite=False,number=240) and launch.get('requested_allocation_seconds')==14400,
        'Prior Map elapsed or requested clock differs')
    allocated={x['type']+('/'+x['name'] if x['name'] else ''):x['count'] for x in row['tres']['allocated']}
    require(len(allocated)==len(row['tres']['allocated']) and allocated.get('cpu')==2 and allocated.get('mem')==8192
        and allocated.get('node')==1 and not any(k.startswith('gres') and v for k,v in allocated.items()),'Prior Map allocation shape differs')
    attempt=previous/('attempt-'+job);closure=read(attempt/'closure.json');ledger=read(attempt/'transfer-ledger.json')
    require(closure.get('remaining_owned_processes')=={} and closure.get('child_exitcode')==1,'Prior Map workers not verifiably closed')
    require(audit.get('status')=='failed_transfer_closed_independently_audited' and audit.get('job_id')==job
        and audit.get('predecessor_root')==str(previous) and audit.get('proposal_sha256')==digest(p)
        and audit['terminal']['snapshot_sha256']==sha(metadata/'prior-terminal-scheduler.json'), 'Map failure audit differs')
    for name,reference in audit['preserved_artifact_sha256'].items():
        require(sha(previous/name)==reference['sha256'],'Preserved Map failure evidence changed')
    require(prefix.get('failure_audit',{}).get('sha256')==sha(metadata/'prior-failure-audit.json')
        and prefix.get('preserved_ledger_sha256')==sha(attempt/'transfer-ledger.json')
        and prefix.get('source_transfer_sha256')==p['source_sha256']['acquisition.py']
        and prefix.get('predecessor_proposal_sha256')==digest(p) and prefix.get('job_id')==job,
        'Closed local prefix proof is not bound to this failed transfer')
    item=p['items'][0];path=backend.checked(attempt/item['filename'])
    require(item['kind']=='dataset' and item['filename']=='datasets/osm_tile_server.tar' and
        prefix.get('prefix_path')==str(path) and 0<prefix['fixed_prefix_bytes']<item['bytes'] and
        prefix.get('expected_archive_bytes')==item['bytes'],'Bound first-dataset prefix required')
    check_prefix(path,prefix)
    reservations=ledger['reservations'];charged=known=0
    url_sha=hashlib.sha256(item['url'].encode()).hexdigest()
    for index,reservation in enumerate(reservations):
        complete=index<len(reservations)-1
        require(reservation.get('id')==index and reservation.get('header_reserved_bytes')==65536
            and reservation.get('reserved_bytes')==64*2**20 and reservation.get('url_sha256')==url_sha
            and reservation.get('host')=='webarena-map-server-data.s3.amazonaws.com'
            and reservation.get('state')==('complete' if complete else 'unknown')
            and reservation.get('received_bytes')==(64*2**20 if complete else 0),
            'Prior sequential prefix/unknown reservation geometry changed')
        charged+=reservation['reserved_bytes']+65536;known+=reservation['received_bytes']
    require(reservations and charged==ledger['charged_bytes'] and known==ledger['known_received_bytes']==prefix['fixed_prefix_bytes']
        and ledger['metadata_received_bytes']==0 and ledger['cap_bytes']==backend.LIMITS['download_charged_bytes'],
        'Original Map cumulative network accounting changed')
    context=read(attempt/'context.json')
    require(scratch.get('path')==context['scratch'] and scratch.get('node')==row['nodes'] and scratch.get('exists') is True
        and type(scratch.get('apparent_bytes')) is int and 0<=scratch['apparent_bytes']<SCRATCH_RESERVE,
        'Prior Map node scratch must fit the explicit reservation')
    remaining=sum(x['bytes'] for x in p['items'])-prefix['fixed_prefix_bytes']
    require(charged+remaining<backend.LIMITS['download_charged_bytes'],'Original byte cap cannot fund the remaining payload')
    seconds=(14400-elapsed)//60*60
    require(seconds>backend.LIMITS['closure_reserve_seconds'],'Original Map time exhausted')
    return dict(protocol=PROTOCOL,status='pending_exact_append_transfer_tls_exception',predecessor_root=str(previous),
        predecessor_job_id=job,predecessor_proposal_sha256=digest(p),attempt_number=2,
        prior_allocation_seconds=elapsed,requested_allocation_seconds=seconds,original_total_seconds=14400,
        additional_allocation_attempts=1,additional_transfer_campaigns=1,maximum_tls_handshake_reconnects=3,
        prior_network_accounting=dict(prior_charged_bytes=charged,prior_metadata_bytes=0,prior_responses=len(reservations)),
        prior_temporary_reserve_bytes=SCRATCH_RESERVE,prefix_proof=prefix,
        preserved_partial_append_growth_bytes=item['bytes']-prefix['fixed_prefix_bytes'],remaining_body_bytes=remaining,
        prefix_sha_verified_in_closed_read=True,prefix_rehash_before_network_required=True,
        append_original_inode=True,overwrite_truncate_rename_delete_allowed=False,
        further_successors_allowed=False,refund_unknown_reservations=False,
        preserved_evidence_sha256={name:ref['sha256'] for name,ref in audit['preserved_artifact_sha256'].items()})
