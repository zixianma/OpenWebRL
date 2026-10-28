#!/usr/bin/env python3
"""Collect durable OM2W Table 5-style cell results across stage packages (read-only).

Rates and paired comparisons are withheld until every planned cell has a verified
queue outcome or a separately root-reviewed forensic invalid disposition. Raw queue
states and absent native verdicts remain unchanged; invalid attempts contribute no
success to the overall denominator. Other claims without results remain unresolved.
This artifact-integrity collector does not
replace the independent financial and scheduler completion audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import time
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.webprm_campaign_queue import identity
from openwebrl.webprm_webarena import digest
from openwebrl.webprm_om2w_table5 import cell_outcome, JUDGE, judge_trajectory
from openwebrl.jev_eval import parse_verdict


def read(path): return json.loads(Path(path).read_text())


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_result(path, cell, plan):
    """Verify closed worker identity, native verdict and terminal image/trajectory bindings."""
    outcome = cell_outcome(path, cell, digest(plan))
    result = read(path)
    if result['valid']:
        folder = Path(path).parent
        judge = read(folder/'om2w-judge.json')
        native = read(folder/'native-evaluation/native-verdict.json')
        fallback = read(folder/'protocol-fallback.json') if (folder/'protocol-fallback.json').exists() else None
        trajectory = judge_trajectory(read(folder/'rollout.json'), fallback, judge['termination'])
        if (judge.get('model') != JUDGE or judge.get('finish_reason') != 'stop'
                or parse_verdict(judge.get('raw', '')) != result['success']
                or judge.get('parsed') != result['success']
                or native.get('judge') != judge or native.get('valid') is not True
                or native.get('score') != result['success'] or native.get('success') != bool(result['success'])
                or judge.get('final_screenshot_sha256') != sha(folder/'judge-final.jpg')
                or judge.get('trajectory_sha256') != hashlib.sha256(trajectory.encode()).hexdigest()
                or result.get('terminal', {}).get('screenshot_sha256') != sha(folder/'native-evaluation/terminal.png')):
            raise ValueError('OM2W judge/image/trajectory evidence differs: '+cell['cell_id'])
    return result, outcome


def verified_watchdog_exception(path, cell, plan):
    """Preserve the frozen watchdog's closed invalid result without adding fields.

    The v2 watchdog fallback omitted actor_model/selector; the controller then
    recorded its ValueError as a worker_exception. Only that exact invalid-only
    pattern is recognized here. Assigned-plan/source/claim binding is checked by
    stage_records before this verifier is called.
    """
    result=read(path)
    watchdog=read(Path(path).parent/'watchdog.json')
    if (result.get('identity_sha256')!=digest(plan) or result.get('task_id')!='om2w_'+cell['task_id']
            or result.get('arm')!=cell['actor']+'__'+cell['selector']
            or 'actor_model' in result or 'selector' in result
            or result.get('attempted') is not True or result.get('valid') is not False
            or type(result.get('success')) is not int or result['success']!=0
            or result.get('error_type')!='EpisodeWatchdogTimeout'
            or result.get('watchdog')!=watchdog or watchdog.get('reason')!='phase_timeout'
            or watchdog.get('remaining_owned_processes')!={}
            or watchdog.get('last_phase',{}).get('identity_sha256')!=digest(plan)):
        raise ValueError('Frozen watchdog exception does not prove this exact closed invalid attempt')
    return result


def verified_stopped_worker_exception(path, cell, plan, outcome):
    """Recognize only saved closed invalids from controller SIGTERM shutdown.

    The frozen controller can fail after its child saved a raw invalid result,
    yielding a worker_exception queue disposition rather than cell-outcome.json.
    Preserve both records verbatim; this supplies neither a native verdict nor
    permission to retry the consumed claim.
    """
    path = Path(path); result = read(path); watchdog = read(path.parent/'watchdog.json')
    if (outcome.get('status') != 'invalid' or outcome.get('reason') != 'worker_exception'
            or set(outcome) != {'status', 'reason', 'error_type'}
            or result.get('identity_sha256') != digest(plan)
            or result.get('task_id') != 'om2w_'+cell['task_id']
            or result.get('arm') != cell['actor']+'__'+cell['selector']
            or result.get('attempted') is not True or result.get('valid') is not False
            or type(result.get('success')) is not int or result['success'] != 0
            or result.get('watchdog') != watchdog
            or watchdog.get('remaining_owned_processes') != {}
            or watchdog.get('last_phase', {}).get('identity_sha256') != digest(plan)
            or type(watchdog.get('supervisor_signal')) is not int or watchdog['supervisor_signal'] != 15
            or watchdog.get('supervisor_error') is not None or watchdog.get('reason') is not None
            or (path.parent/'om2w-judge.json').exists()
            or (path.parent/'native-evaluation/native-verdict.json').exists()):
        raise ValueError('Stopped worker exception does not prove this exact closed invalid attempt')
    if result.get('error_type') == 'EpisodeWorkerExited':
        if (outcome['error_type'] not in ('ValueError', 'RuntimeError')
                or 'actor_model' in result or 'selector' in result
                or watchdog.get('child_exitcode') != -15
                or result.get('error') != 'Episode child exited without result (exit -15)'):
            raise ValueError('Stopped worker fallback identity or SIGTERM exit differs')
    elif result.get('error_type') == 'BrokerError':
        if (outcome['error_type'] != 'ConnectionRefusedError' or watchdog.get('child_exitcode') != 0
                or result.get('cleanup_status') != 'completed'):
            raise ValueError('Stopped broker worker lacks completed invalid cleanup')
        # The normal verifier checks actor/selector and all raw closure bindings.
        verified_result(path, cell, plan)
    else:
        raise ValueError('Unrecognized stopped worker exception')
    return result


def assigned_plan(template, proposal, cell, attempt, claim):
    """Reconstruct every field, allowing only the original controller's recorded runtime values."""
    from scripts.launch_webprm_om2w_table5 import cell_plan
    context = read(attempt/'execution-context.json')
    epochs = context['scheduler_epochs']
    job = context['job_id']
    if (not isinstance(job, str) or not job.isdigit() or attempt.name != 'attempt-'+job
            or epochs['job_id'] != job or context['proposal_sha256'] != digest(proposal)
            or context['output'] != str(attempt)
            or context['deadline'] != min(epochs['end_epoch'], epochs['start_epoch']+epochs['limit_seconds'])-proposal['limits']['cleanup_seconds']
            or context['prior_seconds']+epochs['limit_seconds'] > proposal['resources']['total_seconds']
            or not epochs['start_epoch'] <= claim['epoch'] < context['deadline']
            or not re.fullmatch(r'[a-f0-9]{32}', claim.get('owner', ''))):
        raise ValueError('Claim does not belong to its recorded allocation context')
    scratch = Path(context['scratch'])
    if not scratch.is_absolute() or scratch.resolve() != scratch or not str(scratch).startswith('/tmp/'):
        raise ValueError('Recorded local runtime root is not canonical /tmp scratch')
    ports = {}
    for name, model in proposal['models'].items():
        command = read(attempt/(name+'-command.json'))
        argv = command['argv']
        if argv.count('--port') != 1 or argv.count('--host') != 1 or argv[argv.index('--host')+1] != '127.0.0.1':
            raise ValueError('Local critic command is not an explicit loopback service')
        port = int(argv[argv.index('--port')+1])
        if not 1024 <= port <= 65535 or port in ports.values():
            raise ValueError('Invalid or overlapping recorded critic ports')
        ports[name] = port
    return cell_plan(template, proposal, context, cell, socket_path=scratch/'broker/socket', ports=ports)


def attempt_protocol(proposal, template, attempt, parent=None):
    """Bind an attempt to this package or its explicitly verified repair parent.

    Runtime proposal hashes are canonical JSON digests, whereas repair references
    bind the original proposal and approval *file bytes*. Never infer ancestry
    from matching cell IDs, source paths or an attempt directory name.
    ``parent`` must be the result of the launcher's strict ``repair_parent``
    verifier; it is deliberately limited to one immutable predecessor.
    """
    recorded = read(Path(attempt)/'execution-context.json')['proposal_sha256']
    options = [(proposal, template)] + ([parent] if parent is not None else [])
    matches = [(p, t) for p, t in options if digest(p) == recorded]
    if len(matches) != 1:
        raise ValueError('Attempt proposal is outside the explicitly verified repair lineage')
    return matches[0]



NODEFAIL_PROTOCOL='om2w_saved_nodefail_activation_v1'
NODEFAIL_REVIEW_PROTOCOL='om2w_saved_nodefail_root_review_v1'
NODEFAIL_INDEPENDENT_IMPLEMENTATION='2380e727541b9bc56f81708737f347b67c819c0b784c4bc38dc624d167f219cc'


def require_nodefail(condition,message):
    if not condition:raise ValueError(message)


class NodeFailEvidence:
    """Bounded immutable references and append-tolerant exact journal prefixes."""
    def __init__(self):
        self.cache={};self.prefixes={};self.deadline=time.monotonic()+180

    def reference(self,value):
        path=Path(value['path']);roots=(Path('/tmp'),Path('/gpfs/scrubbed/zixianma/openwebrl-runtime'))
        require_nodefail(path.is_absolute() and path.resolve()==path and any(r in path.parents for r in roots) and
            not any(p.is_symlink() for p in (path,*path.parents)),'Node-failure evidence must be canonical private runtime files')
        require_nodefail(time.monotonic()<self.deadline and path.is_file() and path.stat().st_size<=2**30,
            'Missing/oversized node-failure evidence or deadline exhausted')
        stat=path.stat();fingerprint=(stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino,stat.st_dev)
        key=(str(path),value['sha256'])
        if key not in self.cache:
            require_nodefail(sha(path)==value['sha256'],'Node-failure evidence hash differs')
            self.cache[key]=fingerprint
        require_nodefail(self.cache[key]==fingerprint and ('bytes' not in value or value['bytes']==stat.st_size),
            'Node-failure evidence changed or size differs')
        return path

    def read(self,value):
        raw=self.reference(value).read_bytes()
        require_nodefail(hashlib.sha256(raw).hexdigest()==value['sha256'],'Node-failure JSON changed during interpretation')
        return json.loads(raw)

    def prefix(self,path,size,checksum):
        path=Path(path)
        require_nodefail(path.resolve()==path and not any(p.is_symlink() for p in (path,*path.parents)) and
            type(size) is int and 0<size<=2**30 and path.stat().st_size>=size,'Invalid exact node-failure journal prefix')
        key=(str(path),size,checksum);stat=path.stat()
        fingerprint=(stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino,stat.st_dev)
        require_nodefail(time.monotonic()<self.deadline,'Node-failure prefix verification deadline exhausted')
        with path.open('rb') as stream:raw=stream.read(size)
        require_nodefail(hashlib.sha256(raw).hexdigest()==checksum and raw.endswith(b'\n'),
            'Node-failure journal prefix differs')
        last=json.loads(raw.splitlines()[-1]);self.prefixes[key]=(fingerprint,last)
        return last

    def unchanged(self):
        for (name,_),fingerprint in self.cache.items():
            stat=Path(name).stat()
            require_nodefail(fingerprint==(stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino,stat.st_dev),
                'Previously verified node-failure evidence changed')
        for name,size,checksum in list(self.prefixes):self.prefix(name,size,checksum)



def verified_nodefail_disposition(pointer,proposal,cohort,template,cell,claim,evidence):
    """Activate only a separately reviewed invalid interpretation of an unchanged raw claim."""
    value=evidence.read(dict(path=str(pointer),sha256=sha(pointer)));root=Path(proposal['campaign_root'])
    require_nodefail(value.get('protocol')==NODEFAIL_PROTOCOL and value.get('cell_id')==cell['cell_id'] and
        value.get('raw_artifacts_preserved') is True and value.get('primary_rerun') is False and
        value.get('native_success') is None and value.get('effective_success') is None,
        'Unreviewed node-failure pointer or fabricated verdict')
    candidate=evidence.read(value['original_audit']);independent=evidence.read(value['independent_review'])
    review=evidence.read(value['root_review'])
    require_nodefail(review.get('protocol')==NODEFAIL_REVIEW_PROTOCOL and review.get('verified') is True and
        review.get('errors')==[] and review.get('raw_artifacts_preserved') is True and
        review.get('primary_rerun') is False and review.get('native_verdict_created') is False and
        all(review.get(k)==value[k] for k in ('original_audit','independent_review','financial_helper_archive')),
        'Node-failure supplement lacks exact root activation review')
    implementation=independent['implementation'];evidence.reference(implementation)
    require_nodefail(implementation['sha256']==NODEFAIL_INDEPENDENT_IMPLEMENTATION and
        independent.get('status')=='independent_review_passed_pending_root_activation' and independent.get('verified') is True and
        independent.get('activated') is False and independent.get('candidate',{}).get('path')==value['original_audit']['path'] and
        independent['candidate']['sha256']==value['original_audit']['sha256'] and independent.get('full_plan_mismatches')==0 and
        independent.get('recorded_plans_verified')==230 and independent.get('interrupted_plans_verified')==6 and
        independent.get('normalized_records_created') is False and independent.get('raw_artifacts_modified') is False,
        'Node-failure independent assignment/financial review differs')
    evidence.reference(value['financial_helper_archive'])
    require_nodefail(value['financial_helper_archive']['sha256']==independent['financial_helper']['sha256'],
        'Archived independently reviewed financial helper differs')
    require_nodefail(candidate.get('schema_version')==1 and candidate.get('verified') is True and
        candidate.get('job_id')=='361144' and candidate.get('interrupted_primary_count')==6 and
        candidate.get('collector_activation')=='not_activated_pending_independent_review' and len(candidate.get('cells',[]))==6,
        'Node-failure forensic candidate differs')
    ids=[r['cell_id'] for r in candidate['cells']]
    require_nodefail(len(set(ids))==6 and sorted(review.get('cell_ids',[]))==sorted(ids),
        'Root activation review omits/repeats original interrupted claims')
    for name in ('closure','recovery','receipt_audit','proposal','script'):evidence.reference(candidate[name])
    require_nodefail(evidence.read(candidate['proposal'])==proposal,'Forensic node-failure proposal differs')
    require_nodefail(evidence.read(independent['template'])==template and
        independent['template']['path']==proposal['worker_template'],'Forensic node-failure template differs')
    attempt=root/'attempt-361144';folder=attempt/'cells'/hashlib.sha256(cell['cell_id'].encode()).hexdigest()
    matches=[r for r in candidate['cells'] if r['cell_id']==cell['cell_id']]
    require_nodefail(len(matches)==1,'Node-failure cell is absent/repeated in forensic audit')
    item=matches[0]
    require_nodefail(item.get('task_id')==cell['task_id'] and item.get('actor')==cell['actor'] and
        item.get('selector')==cell['selector'] and item.get('raw_queue_state')=='attempted_without_outcome' and
        item.get('raw_native_valid') is None and item.get('raw_native_success') is None and
        item.get('retry_primary_permitted') is False and item.get('valid_success_evidence') is False and
        item.get('raw_artifacts_modified') is False,'Node-failure claim identity or absent verdict changed')
    claim_path=root/'queue/claims'/(identity(cell['cell_id'])+'.json')
    require_nodefail(item['claim']['path']==str(claim_path) and evidence.read(item['claim'])==claim,
        'Node-failure original fixed claim differs')
    require_nodefail(item['worker_plan']['path']==str(folder/'worker-plan.json'),'Node-failure worker plan path differs')
    plan=evidence.read(item['worker_plan'])
    require_nodefail(plan==assigned_plan(template,proposal,cell,attempt,claim) and digest(plan)==item['worker_plan_identity_sha256'],
        'Node-failure full assigned worker plan differs')
    assignments=[r for r in independent['full_plan_assignments'] if r['cell_id']==cell['cell_id']]
    require_nodefail(len(assignments)==1 and assignments[0]['status']=='interrupted' and
        assignments[0]['plan']['path']==item['worker_plan']['path'] and
        assignments[0]['plan']['sha256']==item['worker_plan']['sha256'] and
        assignments[0]['reconstructed_plan_identity']==digest(plan),'Independently reconstructed node-failure plan differs')
    episode=folder/'episodes'/('om2w_'+cell['task_id'])/(cell['actor']+'__'+cell['selector'])
    require_nodefail(not any((episode/name).exists() for name in ('result.json','watchdog.json','om2w-judge.json',
        'native-evaluation/native-verdict.json')) and not (folder/'cell-outcome.json').exists() and
        not (root/'queue/outcomes'/(identity(cell['cell_id'])+'.json')).exists(),
        'Forensic invalid supplement cannot replace a raw result, verdict, watchdog or outcome')
    actual={str(p) for p in folder.rglob('*') if p.is_file()}
    require_nodefail(actual=={v['path'] for v in item['evidence']} and len(actual)==len(item['evidence']),
        'Node-failure artifact omitted or duplicated')
    for reference in item['evidence']:
        require_nodefail(folder in Path(reference['path']).parents,'Node-failure cell artifact escaped its original folder')
        evidence.reference(reference)
    phase=read(episode/'phase.json')
    require_nodefail(phase.get('identity_sha256')==digest(plan) and phase['phase']==item['last_saved_phase'] and
        len(read(episode/'rollout.json'))==item['recorded_rollout_turns'],'Interrupted phase/rollout identity differs')
    context=evidence.read(independent['context'])
    require_nodefail(independent['context']['path']==str(attempt/'execution-context.json') and
        context==read(attempt/'execution-context.json'),'Independently reviewed allocation context differs')
    require_nodefail(set(independent['model_commands'])==set(proposal['models']),'Independent model command panel differs')
    for name,reference in independent['model_commands'].items():
        require_nodefail(reference['path']==str(attempt/(name+'-command.json')),'Independent model command belongs to another owner')
        evidence.reference(reference)
    evidence.reference(independent['frozen_controller']);evidence.reference(independent['frozen_server_builder'])
    closure=evidence.read(candidate['closure']);recovery=evidence.read(candidate['recovery']);receipt=evidence.read(candidate['receipt_audit'])
    require_nodefail(closure.get('verified') is True and closure.get('errors')==[] and closure.get('job_id')=='361144' and
        closure.get('scheduler_state')=='NODE_FAIL' and closure.get('scheduler_terminal') is True and
        closure.get('all_owned_processes_reaped') is True,'Node-failure global closure proof differs')
    scheduler=evidence.reference(closure['scheduler_evidence']).read_text();node=evidence.reference(closure['process_closure_evidence']).read_text()
    rows=[r.split('|') for r in scheduler.splitlines()];rows=[r for r in rows if r[0]=='361144']
    require_nodefail(len(rows)==1 and rows[0][1]=='NODE_FAIL' and rows[0][5]=='g011' and re.search(r'\bNodeName=g011\b',node),
        'Node-failure scheduler/node identity differs')
    boot=re.search(r'\bBootTime=(\S+)',node)
    require_nodefail(boot is not None and datetime.fromisoformat(boot.group(1))>datetime.fromisoformat(rows[0][4]) and
        independent['original_scheduler_end']==rows[0][4] and independent['recorded_node_boot']==boot.group(1),
        'Recorded reboot does not establish post-allocation global closure')
    before,after=recovery['before'],recovery['after']
    require_nodefail(recovery.get('verified') is True and before['charged']==after['charged']==independent['original_charge_usd'] and
        before['unsettled_reservations']==after['unsettled_reservations']==independent['retained_unknown_paid_reservations']==5 and
        before['unsettled_reserved']==after['unsettled_reserved']==independent['retained_unknown_charge_usd']=='0.70415300' and
        after['active_epoch_id'] is None,'Node-failure recovery changed charges or unknown reservations')
    budget=root/'shared-budget.json.events.jsonl';broker=attempt/'broker.jsonl'
    require_nodefail(independent['recovered_financial_prefix_bytes']==after['budget_prefix_bytes'] and
        independent['recovered_financial_prefix_sha256']==after['budget_ledger_sha256'] and receipt['audit_path']==str(broker),
        'Independent financial/broker prefix binding differs')
    last=evidence.prefix(budget,after['budget_prefix_bytes'],after['budget_ledger_sha256'])
    evidence.prefix(broker,receipt['audit_prefix_bytes'],receipt['audit_sha256'])
    require_nodefail(last.get('event')=='epoch_closed' and last.get('epoch_id')=='job-361144' and
        last.get('closure')=='audited_recovery' and last.get('event_sha256')==recovery['closed_epoch_sha256'] and
        last.get('recovery_evidence')==[{k:v for k,v in candidate[n].items() if k!='bytes'} for n in ('closure','receipt_audit')],
        'Node-failure recovered epoch is not bound to exact closure/accounting proof')
    call=item['financial_call_audit'];audits=[r for r in independent['interrupted_cells'] if r['cell_id']==cell['cell_id']]
    require_nodefail(call is not None and len(audits)==1 and audits[0]['paid']==call['paid'] and
        audits[0]['local']==call['local'] and audits[0]['unknown_paid_reservations']==call['unknown_reservations'] and
        call['cell_id']==cell['cell_id'] and call['plan_sha256']==digest(plan) and
        audits[0].get('native_result_absent') is True and audits[0].get('graceful_worker_closure_claimed') is False,
        'Independent interrupted cell accounting differs')
    require_nodefail(call['api']['path']==str(episode/'api.jsonl'),'Interrupted API evidence belongs to another cell')
    evidence.reference(call['api'])
    return dict(task_id='om2w_'+cell['task_id'],actor_model=cell['actor'],selector=cell['selector'],attempted=True,
        valid=False,success=None,native_success=None,raw_native_valid=None,error_type='NODE_FAIL',
        forensic_protocol=NODEFAIL_PROTOCOL,raw_queue_state='attempted_without_outcome',
        global_closure_basis='saved post-allocation node reboot; no graceful worker closure inferred',
        forensic_activation=str(pointer),forensic_audit=value['original_audit'],independent_review=value['independent_review']),attempt

def stage_records(package):
    """Verify fixed package, claims and immutable outcomes without acquiring a writer lock."""
    proposal = read(Path(package)/'proposal.json')
    from scripts.launch_webprm_om2w_table5 import repair_parent
    parent = repair_parent(proposal)
    for path, expected in proposal['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen package source changed: '+path)
    cohort = read(proposal['cohort_path'])
    if digest(cohort) != proposal['cohort_sha256']:
        raise ValueError('Frozen cohort differs')
    root = Path(proposal['campaign_root'])
    from scripts.launch_webprm_om2w_table5 import stage_cells
    cells = stage_cells(proposal, cohort)
    manifest = root/'queue/manifest.json'
    if manifest.exists() and read(manifest) != dict(cell_count=len(cells), cells_sha256=identity(cells)):
        raise ValueError('Queue manifest differs from fixed stage')
    expected_keys = {identity(cell['cell_id']) for cell in cells}
    for group in ('claims', 'outcomes'):
        if any(path.stem not in expected_keys for path in (root/'queue'/group).glob('*.json')):
            raise ValueError('Unexpected primary claim/outcome outside fixed stage')
    template = read(proposal['worker_template'])
    forensic=root/'forensic-dispositions'
    if any(path.suffix!='.json' or path.stem not in expected_keys for path in forensic.glob('*')):
        raise ValueError('Unexpected forensic pointer outside fixed stage')
    evidence=NodeFailEvidence()
    rows = []
    attempt_owners, owner_attempts = {}, {}
    for cell in cells:
        key = identity(cell['cell_id'])
        claim_path, outcome_path = root/'queue/claims'/(key+'.json'), root/'queue/outcomes'/(key+'.json')
        claim = read(claim_path) if claim_path.exists() else None
        outcome = read(outcome_path) if outcome_path.exists() else None
        if claim is not None and (claim.get('cell_id') != cell['cell_id'] or claim.get('cell_sha256') != identity(cell)):
            raise ValueError('Durable claim identity differs')
        if outcome is not None and (not claim or outcome.get('cell_id') != cell['cell_id'] or
                        outcome.get('cell_sha256') != identity(cell) or outcome.get('owner') != claim.get('owner')):
            raise ValueError('Durable outcome identity/owner differs')
        if (claim is not None or outcome is not None) and not manifest.exists():
            raise ValueError('Primary evidence without fixed queue manifest')
        state = 'recorded' if outcome is not None else 'attempted_without_outcome' if claim is not None else 'pending'
        result, found = None, []
        for attempt in sorted(root.glob('attempt-*')):
            folder = attempt/'cells'/hashlib.sha256(cell['cell_id'].encode()).hexdigest()
            path = folder/'episodes'/('om2w_'+cell['task_id'])/(cell['actor']+'__'+cell['selector'])/'result.json'
            if path.exists(): found.append((folder, path))
        if len(found) > 1:
            raise ValueError('A primary cell has multiple results: '+cell['cell_id'])
        if found and not claim:
            raise ValueError('Result without durable primary claim')
        if outcome is not None:
            if len(found) != 1:
                raise ValueError('Recorded outcome has no unique saved result')
            folder, path = found[0]
            plan = read(folder/'worker-plan.json')
            attempt = folder.parents[1]
            attempt_proposal, attempt_template = attempt_protocol(proposal, template, attempt, parent)
            if plan != assigned_plan(attempt_template, attempt_proposal, cell, attempt, claim):
                raise ValueError('Assigned worker plan differs from complete frozen protocol/runtime reconstruction')
            owner = claim['owner']
            if attempt_owners.setdefault(str(attempt), owner) != owner or owner_attempts.setdefault(owner, str(attempt)) != str(attempt):
                raise ValueError('Claim owner is inconsistent with its allocation attempt')
            if outcome['result'].get('reason') == 'worker_exception':
                if (folder/'cell-outcome.json').exists():
                    raise ValueError('Worker exception contradicts a normal cell outcome')
                if (outcome['result']==dict(status='invalid',reason='worker_exception',error_type='ValueError')
                        and read(path).get('error_type')=='EpisodeWatchdogTimeout'):
                    result=verified_watchdog_exception(path,cell,plan)
                else:
                    result=verified_stopped_worker_exception(path,cell,plan,outcome['result'])
            else:
                result, expected_outcome = verified_result(path, cell, plan)
                if outcome['result'] != expected_outcome or read(folder/'cell-outcome.json') != expected_outcome:
                    raise ValueError('Saved result hash differs from durable queue outcome')
        pointer=forensic/(key+'.json')
        if pointer.exists():
            if claim is None or outcome is not None or found:
                raise ValueError('Forensic supplement conflicts with absent claim or existing raw outcome/result')
            result,attempt=verified_nodefail_disposition(pointer,proposal,cohort,template,cell,claim,evidence)
            owner=claim['owner']
            if attempt_owners.setdefault(str(attempt),owner)!=owner or owner_attempts.setdefault(owner,str(attempt))!=str(attempt):
                raise ValueError('Forensic claim owner is inconsistent with its allocation attempt')
            state='audited_interrupted'
        # Results without queue closure remain unresolved unless exact root-reviewed evidence resolves an invalid interruption.
        rows.append((cell, result, state))
    evidence.unchanged()
    return rows


def bootstrap(differences, *, draws=10000, seed=0):
    if not differences:
        return None
    rng = random.Random(seed); n = len(differences); means = []
    for _ in range(draws):
        means.append(sum(differences[rng.randrange(n)] for _ in range(n))/n)
    means.sort()
    return [100*means[int(.025*draws)], 100*means[int(.975*draws)-1]]


def summarize(rows):
    table = defaultdict(lambda: dict(planned=0, recorded=0, audited_interrupted=0, attempted=0, valid=0, successes=0, invalid_reasons=Counter()))
    outcome = {}
    for cell, result, state in rows:
        row = table[(cell['actor'], cell['selector'])]
        row['planned'] += 1
        row['attempted'] += state != 'pending'
        if result is None:
            continue
        row['recorded'] += state=='recorded'
        row['audited_interrupted'] += state=='audited_interrupted'
        row['valid'] += bool(result.get('valid'))
        row['successes'] += int(result.get('success') or 0)
        if not result.get('valid'):
            row['invalid_reasons'][result.get('error_type') or 'unknown'] += 1
        outcome[(cell['actor'], cell['selector'], cell['task_id'])] = result
    complete = all(state in ('recorded','audited_interrupted') and result is not None for _, result, state in rows)
    summary = []
    for (actor, selector), row in sorted(table.items()):
        summary.append(dict(actor=actor, selector=selector, planned=row['planned'], recorded=row['recorded'],
            attempted=row['attempted'], audited_interrupted=row['audited_interrupted'],
            resolved=row['recorded']+row['audited_interrupted'], valid=row['valid'], successes=row['successes'],
            overall_success_pct=100*row['successes']/row['planned'] if complete and row['planned'] else None,
            valid_only_success_pct=100*row['successes']/row['valid'] if complete and row['valid'] else None,
            invalid_reasons=dict(row['invalid_reasons'])))
    paired = []
    actors = sorted({a for a, _ in table}); selectors = sorted({s for _, s in table})
    for actor in actors:
        for selector in selectors:
            if not complete or selector == 'no_search':
                continue
            diffs = []; common_valid = []
            for (a, s, task), result in outcome.items():
                if a != actor or s != selector:
                    continue
                base = outcome.get((actor, 'no_search', task))
                if base is None:
                    continue
                diffs.append(int(result.get('success') or 0)-int(base.get('success') or 0))
                if result.get('valid') and base.get('valid'):
                    common_valid.append(diffs[-1])
            paired.append(dict(actor=actor, selector=selector, paired_tasks=len(diffs),
                difference_pp=100*sum(diffs)/len(diffs) if diffs else None, bootstrap95_pp=bootstrap(diffs),
                common_valid_tasks=len(common_valid),
                common_valid_difference_pp=100*sum(common_valid)/len(common_valid) if common_valid else None))
    return dict(coverage_complete=complete, rates_withheld=not complete, by_condition=summary, paired_vs_no_search=paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', action='append', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = [row for package in args.package for row in stage_records(package)]
    if len({cell['cell_id'] for cell, _, _ in rows}) != len(rows):
        raise ValueError('Overlapping stage packages would double-count primary cells')
    states = Counter(state for _, _, state in rows)
    raw_states=Counter('attempted_without_outcome' if state=='audited_interrupted' else state for _,_,state in rows)
    value = dict(schema_version=2, packages=[str(p) for p in args.package], cells=len(rows), queue_states=dict(raw_states),
                 resolution_states=dict(states), forensic_semantics='Root-reviewed interruptions remain raw claims without outcomes and absent native verdicts; their invalid interpretation contributes zero successes to the overall denominator.',
                 **summarize(rows))
    text = json.dumps(value, indent=1, sort_keys=True)
    if args.output:
        args.output.write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
