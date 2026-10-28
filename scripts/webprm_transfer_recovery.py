"""Evidence for an explicitly approved extra OCI transfer within original caps.

This module neither grants approval nor launches, downloads, resumes or deletes.
The original startup-only recovery policy remains separate and unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import pwd
import shlex


class Closed(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checked(value):
    p = Path(value)
    roots = (Path('/tmp'), Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'))
    if (not p.is_absolute() or p.resolve() != p or not any(r in p.parents for r in roots)
            or any(x.is_symlink() for x in (p, *p.parents))):
        raise Closed('Canonical private runtime evidence required')
    return p


def read(path):
    p = checked(path)
    if p.stat().st_size > 4*2**20:
        raise Closed('Oversized recovery evidence')
    return json.loads(p.read_text())


def sha(path):
    return hashlib.sha256(checked(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise Closed(message)


def proof(root, terminal, replacement_sources, repair_path, *, prior_temporary_reserve_bytes):
    """Bind one closed first transfer; return a proof requiring a new exception.

    sacct's terminal epochs/resources are bound to the original saved scontrol
    start, exact submission argv/nonce, frozen approval and closed worker tree.
    Every unresolved response retains its entire reservation, without refunds.
    """
    root = checked(root)
    p, auth, launch, claim, network, start_snapshot = (
        read(root/name) for name in ('proposal.json', 'approval.json', 'launch.json',
            'run-claim.json', 'network-claim.json', 'scheduler.json'))
    resources = p['resources']
    require(auth.get('status') == 'approved' and auth.get('exact_user_approval')
        and auth.get('third_party_unverified_state_source_accepted') is True
        and auth.get('proposal_sha256') == digest(p) == launch.get('proposal_sha256'),
        'Original exact source/resource approval changed')
    require(p.get('site') in ('shopping', 'gitlab')
        and p.get('identity_status') == 'third_party_reconstruction_original_state_unverified'
        and p.get('original_equivalence_verified') is False
        and p.get('startup_recovery') == dict(attempt_number=1, prior_allocation_seconds=0,
                                             requested_allocation_seconds=7200)
        and resources.get('total_allocation_seconds_including_retries') == 7200
        and resources.get('maximum_allocation_attempts') == 3
        and resources.get('maximum_transfer_attempts') == 1
        and resources.get('cpus') == 2 and resources.get('memory_gib') == 8
        and resources.get('gpus') == resources.get('api_usd') == 0,
        'Only this closed first OCI transfer supports the proposed exception')
    require(not (root/'completed.json').exists(), 'Completed acquisition cannot be retried')
    for name, expected in p['source_sha256'].items():
        require(sha(root/name) == expected, 'Preserved source changed')
    job = claim['job_id']
    require(job.isdigit() and network.get('job_id') == job and claim.get('nonce') == launch.get('nonce'),
            'Attempt/network ownership changed')
    require(terminal.get('errors') == [] and len(terminal.get('jobs', [])) == 1
        and start_snapshot.get('errors') == [] and len(start_snapshot.get('jobs', [])) == 1,
        'One start and one terminal scheduler record required')
    row, start = terminal['jobs'][0], start_snapshot['jobs'][0]
    t = row['time']
    require(row['job_id'] == int(job) and row['user'] == pwd.getpwuid(os.getuid()).pw_name
        and row['restart_cnt'] == start['restart_cnt'] == 0
        and start['job_id'] == int(job) and start['user_id'] == os.getuid()
        and row['state']['current'] in (['FAILED'], ['CANCELLED'], ['TIMEOUT'], ['NODE_FAIL'])
        and row['account'] == resources['account'] == 'zixianma'
        and row['partition'] == resources['partition'] == 'gpu-h200'
        and row['qos'] == resources['qos'] == 'normal'
        and row['working_directory'] == str(root)
        and shlex.split(row['submit_line']) == launch['argv']
        and start['comment'] == 'webprm-acquire:' + launch['nonce'],
        'Terminal scheduler identity/nonce/submission mismatch')
    elapsed = t['elapsed']
    require(type(elapsed) is int and 0 <= elapsed <= 7260
        and t['end']-t['start'] == elapsed and t['start'] > 0
        and start['start_time'] == dict(set=True, infinite=False, number=t['start'])
        and t['limit'] == dict(set=True, infinite=False, number=120)
        and launch['requested_allocation_seconds'] == 7200,
        'Terminal cumulative clock mismatch')
    allocated = {v['type'] + ('/'+v['name'] if v['name'] else ''): v['count']
                 for v in row['tres']['allocated']}
    require(len(allocated) == len(row['tres']['allocated'])
        and allocated.get('cpu') == 2 and allocated.get('mem') == 8192
        and allocated.get('node') == 1
        and not any(k.startswith('gres') and v for k, v in allocated.items()),
        'Terminal resource allocation mismatch')
    attempt = root/('attempt-'+job)
    closure, ledger = read(attempt/'closure.json'), read(attempt/'transfer-ledger.json')
    require(closure.get('remaining_owned_processes') == {}
        and type(closure.get('child_exitcode')) is int, 'Prior process closure is absent')
    prior = p['prior_metadata_accounting']
    require(set(prior) == {'prior_charged_bytes', 'prior_metadata_bytes', 'prior_responses'}
        and all(type(v) is int and v >= 0 and ledger.get(k) == v for k, v in prior.items()),
        'Prior metadata accounting changed')
    reservations = ledger['reservations']
    charge = known = 0
    for i, r in enumerate(reservations):
        require(type(r['id']) is int and r['id'] == i and r['state'] in ('complete', 'unknown', 'uncertain')
            and type(r['reserved_bytes']) is int and type(r['received_bytes']) is int
            and 0 <= r['received_bytes'] <= r['reserved_bytes'] and r['reserved_bytes'] > 0
            and r['header_reserved_bytes'] == 65536, 'Invalid response reservation')
        charge += (r['received_bytes'] if r['state'] == 'complete' else r['reserved_bytes']) + 65536
        known += r['received_bytes']
    require(charge == ledger['charged_bytes'] and known == ledger['known_received_bytes'],
            'Prior transfer reservation arithmetic differs')
    require(type(ledger['metadata_received_bytes']) is int
        and 0 <= ledger['metadata_received_bytes'] <= known, 'Invalid prior metadata byte count')
    cumulative = dict(prior_charged_bytes=charge+prior['prior_charged_bytes'],
        prior_metadata_bytes=ledger['metadata_received_bytes']+prior['prior_metadata_bytes'],
        prior_responses=len(reservations)+prior['prior_responses'])
    require(ledger['cumulative_cap_bytes'] == p['limits']['download_charged_bytes']
        and ledger['cap_bytes'] + prior['prior_charged_bytes'] == ledger['cumulative_cap_bytes']
        and cumulative['prior_charged_bytes'] < p['limits']['download_charged_bytes']
        and cumulative['prior_metadata_bytes'] < p['limits']['metadata_received_bytes']
        and cumulative['prior_responses'] < p['limits']['max_http_responses'], 'Prior caps exhausted or changed')
    repair_path = checked(repair_path)
    repair = read(repair_path)
    scanner = 'dependencies/scripts/acquire_webprm_cms_original.py'
    metadata = {'metadata/'+name+'.json' for name in ('manifest', 'config', 'source-assessment',
                                                     'drive-preflight', 'oci-metadata-ledger')}
    require(metadata <= p['source_sha256'].keys()
        and p['source_sha256'].keys() <= replacement_sources.keys()
        and all(replacement_sources.get(k) == p['source_sha256'][k] for k in metadata),
        'Replacement must preserve the complete source set and all pinned metadata')
    require(repair.get('status') == 'passed' and repair.get('tests', {}).get('exit_code') == 0
        and repair['tests'].get('passed', 0) > 0
        and replacement_sources.get(scanner) == repair['source_sha256']['scripts/acquire_webprm_cms_original.py']
        and replacement_sources[scanner] != p['source_sha256'][scanner], 'Tested scanner repair is not bound')
    require(type(prior_temporary_reserve_bytes) is int and 0 < prior_temporary_reserve_bytes < p['limits']['temporary_bytes'],
            'Explicit preserved temporary-space reservation required')
    requested = (7200-elapsed)//60*60
    require(requested > p['limits']['closure_reserve_seconds'], 'No useful original time remains')
    return dict(schema_version=1, status='requires_explicit_additional_transfer_exception',
        attempt_number=2, predecessor_root=str(root), predecessor_job_id=job,
        predecessor_proposal_sha256=digest(p), prior_allocation_seconds=elapsed,
        requested_allocation_seconds=requested, original_total_seconds=7200,
        prior_transfer_attempts=1, proposed_additional_transfer_attempts=1,
        prior_network_accounting=cumulative, prior_temporary_reserve_bytes=prior_temporary_reserve_bytes,
        unresolved_prior_reservations=sum(r['state'] != 'complete' for r in reservations),
        prior_evidence_sha256={str(f.relative_to(root)): sha(f) for f in
            [root/n for n in ('proposal.json','approval.json','launch.json','run-claim.json','network-claim.json','scheduler.json')]
            + [attempt/'closure.json',attempt/'transfer-ledger.json']},
        terminal_snapshot=terminal, repair_receipt=dict(path=str(repair_path),sha256=sha(repair_path)),
        replacement_source_sha256=replacement_sources, pinned_metadata_hashes_unchanged=True,
        successor_profile_limits_items_validation_required=True,
        resume_existing_partials=False, refund_unknown_reservations=False,
        new_compute_budget_seconds=0, new_transfer_byte_budget=0, new_api_budget_usd=0)
