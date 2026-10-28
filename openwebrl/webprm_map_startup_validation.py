"""Map startup/version/reset gates for a future explicitly approved CPU owner.

No scheduler, process, service, socket, browser or provider call is made here.
The production Map site registry remains unavailable. Evidence validation is
separate from implementing or approving the future extractor/service launcher.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import time

from openwebrl.webprm_map_startup_inventory import Closed,checked,json_digest
from openwebrl.webprm_lifecycle import _process

RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011')
PLAN=RUNTIME/'next-stage-preparation/map-startup-validation-bounded-plan-v1.json'
PLAN_SHA='78961857ede8f624c179b42d542feae55f893ac68515ddf9d8e84a92f0ffc1e4'
PROTOCOL='map_owned_startup_validation_context_v1'
ROLES=('frontend_db','frontend_web','tile','nominatim','osrm-car','osrm-bike','osrm-foot')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED'}


def plan():
    raw=PLAN.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PLAN_SHA:raise Closed('Map bounded resource plan changed')
    return json.loads(raw)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def snapshot(value,job_id,state):
    if value.get('errors')!=[] or len(value.get('jobs',[]))!=1:raise Closed('One independently saved scheduler job required')
    row=value['jobs'][0]
    def number(key):
        v=row[key]
        if v.get('set') is not True or v.get('infinite') is not False or type(v.get('number')) is not int:raise Closed('Finite integral scheduler resource required')
        return v['number']
    actual=row.get('job_state')
    if (str(row.get('job_id'))!=str(job_id) or row.get('user_id')!=os.getuid() or row.get('restart_cnt')!=0 or
        row.get('account')!='zixianma' or row.get('partition')!='gpu-h200' or row.get('qos')!='normal' or
        number('cpus')!=8 or number('node_count')!=1 or number('memory_per_node')!=65536 or
        not isinstance(actual,list) or len(actual)!=1 or actual[0] not in state):raise Closed('Exact approved CPU allocation identity differs')
    tres=dict(item.split('=',1) for item in row.get('tres_alloc_str','').split(',') if '=' in item)
    if any(k.startswith('gres/gpu') and v!='0' for k,v in tres.items()):raise Closed('GPU allocation is outside zero-GPU validation')
    start,end,seconds=number('start_time'),number('end_time'),60*number('time_limit')
    if start<=0 or end<start or seconds<=0:raise Closed('Scheduler clock is unavailable')
    return dict(job_id=str(job_id),start_epoch=start,end_epoch=end,requested_seconds=seconds,state=actual[0],snapshot_sha256=json_digest(value))


def execution_context(proposal,approval,current_snapshot,prior_snapshots,registry,*,now=None,
                      job_id=None,controller_pid=None,controller_start_ticks=None,hostname=None):
    """Validate exact caps and registered immutable attempt prefix before work.

    A future submission owner must record the job before releasing it. This
    function does not create that registry, submit a job or activate any service.
    """
    fixed=plan();now=time.time() if now is None else now;job_id=job_id or os.environ.get('SLURM_JOB_ID','')
    if (proposal.get('protocol')!=PROTOCOL or proposal.get('resource_plan_sha256')!=PLAN_SHA or
        proposal.get('resources')!=fixed['resources'] or proposal.get('storage')!=fixed['storage'] or
        approval.get('status')!='approved' or approval.get('proposal_sha256')!=json_digest(proposal) or
        approval.get('job_id')!=job_id or not isinstance(approval.get('exact_user_approval'),str) or not approval['exact_user_approval'].strip()):
        raise Closed('Exact new Map startup resource/browser approval required')
    attempts=registry.get('job_ids')
    if (registry.get('proposal_sha256')!=json_digest(proposal) or not isinstance(attempts,list) or
        not 1<=len(attempts)<=3 or len(attempts)!=len(set(attempts)) or attempts[-1]!=job_id or
        approval.get('attempt_job_ids')!=attempts or len(prior_snapshots)!=len(attempts)-1):
        raise Closed('Complete durable allocation registration and three-start cap required')
    prior=[]
    for job,raw in zip(attempts[:-1],prior_snapshots):
        row=snapshot(raw,job,TERMINAL);prior.append(row)
    used=sum(row['end_epoch']-row['start_epoch'] for row in prior)
    current=snapshot(current_snapshot,job_id,{'RUNNING'})
    if used+current['requested_seconds']>14400:raise Closed('Prior failed seconds cannot reset the four-hour cap')
    if any(a['end_epoch']>b['start_epoch'] for a,b in zip(prior,prior[1:]+[current])):
        raise Closed('Validation allocation attempts cannot overlap')
    deadline=min(current['end_epoch'],current['start_epoch']+current['requested_seconds'])
    if type(now) not in (float,int) or not math.isfinite(now) or not current['start_epoch']<=now<deadline-900:
        raise Closed('No useful approved time before closure reserve')
    if controller_pid is None:controller_pid=os.getpid()
    if controller_start_ticks is None:controller_start_ticks=_process(controller_pid)['start_ticks']
    if type(controller_pid) is not int or controller_pid<=1 or type(controller_start_ticks) is not int or controller_start_ticks<=0:
        raise Closed('Live owned controller identity required')
    return dict(protocol=PROTOCOL,proposal_sha256=json_digest(proposal),job_id=job_id,
        hostname=hostname or socket.gethostname(),controller_pid=controller_pid,controller_start_ticks=controller_start_ticks,
        allocation_deadline=deadline,work_deadline=deadline-900,prior_elapsed_seconds=used,
        registry_sha256=json_digest(registry),scheduler=current,prior_scheduler=prior,
        max_browser_contexts=6,max_browser_actions=300,max_metadata_bytes=32*2**20,
        model_calls=0,download_bytes=0,production_adapter_enabled=False)


def stage_deadline(context,name,started_epoch):
    fixed=plan();stage=next((x for x in fixed['phases'] if x['name']==name),None)
    if stage is None or not math.isfinite(started_epoch) or started_epoch<context['scheduler']['start_epoch']:
        raise Closed('Known bounded stage and original allocation clock required')
    final=context['allocation_deadline'] if name=='closure_and_independent_artifact_audit' else context['work_deadline']
    deadline=min(final,started_epoch+stage['maximum_seconds'])
    if deadline<=started_epoch:raise Closed('Stage exceeds remaining approved time')
    return deadline


def owned_process(context,pid,role,*,process_reader=None,namespace_reader=None):
    if role not in ROLES or type(pid) is not int or pid<=1:raise Closed('Known service role and actual PID required')
    read=process_reader or _process;ns=namespace_reader or (lambda p:os.readlink(f'/proc/{p}/ns/net'))
    owner=read(context['controller_pid']);process=read(pid)
    if owner['start_ticks']!=context['controller_start_ticks'] or owner['state']=='Z' or process['state']=='Z':
        raise Closed('Recorded controller/service process is stale')
    ancestor=process['ppid']
    for _ in range(64):
        if ancestor==context['controller_pid']:break
        if ancestor<=1:raise Closed('Service is not an owned controller descendant')
        ancestor=read(ancestor)['ppid']
    else:raise Closed('Unbounded service ancestry')
    return dict(protocol='map_owned_process_v1',context_sha256=json_digest(context),job_id=context['job_id'],
        role=role,pid=pid,start_ticks=process['start_ticks'],controller_pid=context['controller_pid'],
        controller_start_ticks=context['controller_start_ticks'],network_namespace=ns(pid),hostname=context['hostname'])


def probe_record(context,ownership,argv,stdout,returncode,closure,*,image_config_sha256):
    """Bind a captured native --version/SQL probe to its owned closed process."""
    if (ownership.get('context_sha256')!=json_digest(context) or ownership.get('job_id')!=context['job_id'] or
        ownership.get('role') not in ROLES or closure.get('remaining_owned_processes')!={} or
        ownership.get('pid') not in closure.get('owned_pids',[]) or type(closure.get('child_exitcode')) is not int or
        closure['child_exitcode']!=0 or type(returncode) is not int or returncode!=0 or
        not isinstance(argv,list) or not argv or any(not isinstance(v,str) or not v for v in argv) or
        not isinstance(stdout,bytes) or len(stdout)>4*2**20 or re.fullmatch('[0-9a-f]{64}',image_config_sha256) is None):
        raise Closed('Native probe requires exact owned successful process closure')
    stdout.decode('utf-8')
    return dict(protocol='map_native_probe_v1',context_sha256=json_digest(context),ownership=ownership,
        command=argv,stdout=stdout.decode(),stdout_sha256=sha(stdout),returncode=returncode,
        closure=closure,closure_sha256=json_digest(closure),image_config_sha256=image_config_sha256)


def probe_text(value,context,role,config_sha,expected_argv):
    ownership=value.get('ownership',{});closure=value.get('closure',{})
    if (value.get('protocol')!='map_native_probe_v1' or value.get('context_sha256')!=json_digest(context) or
        ownership.get('role')!=role or ownership.get('context_sha256')!=json_digest(context) or ownership.get('job_id')!=context['job_id'] or
        value.get('image_config_sha256')!=config_sha or value.get('command')!=expected_argv or
        type(value.get('returncode')) is not int or value['returncode']!=0 or
        value.get('closure_sha256')!=json_digest(closure) or closure.get('remaining_owned_processes')!={} or
        type(closure.get('child_exitcode')) is not int or closure['child_exitcode']!=0 or
        ownership.get('pid') not in closure.get('owned_pids',[]) or
        not isinstance(value.get('stdout'),str) or len(value['stdout'].encode())>4*2**20 or
        sha(value['stdout'].encode())!=value.get('stdout_sha256')):raise Closed('Bound native probe evidence differs')
    return value['stdout']


def postgres_compatibility(inventory,member,probe,context,role,config_sha,binary):
    """Compare the observed archive PG_VERSION with actual postgres --version."""
    if role not in ('tile','nominatim','frontend_db') or inventory.get('schema')!='map_selected_inventory_v1':
        raise Closed('Observed PostgreSQL data inventory and role required')
    files=inventory.get('captured',{});row=files.get(member)
    if not row or member.split('/')[-1]!='PG_VERSION' or sha(row['raw'])!=row['sha256']:
        raise Closed('Actual captured PG_VERSION bytes required')
    cluster=row['raw'].decode().strip()
    if re.fullmatch(r'(?:[1-9][0-9]?|9[.][0-6])',cluster) is None:raise Closed('Invalid cluster major')
    if not isinstance(binary,str) or not binary.startswith('/') or '..' in Path(binary).parts or Path(binary).name!='postgres':
        raise Closed('Observed absolute postgres binary path required')
    text=probe_text(probe,context,role,config_sha,[binary,'--version'])
    match=re.fullmatch(r'postgres \(PostgreSQL\) ([0-9]+(?:\.[0-9]+){1,2})(?: [^\r\n]+)?\s*',text)
    if not match:raise Closed('Actual postgres version output is malformed')
    parts=match[1].split('.');major='.'.join(parts[:2]) if parts[0]=='9' else parts[0]
    if major!=cluster:raise Closed('PostgreSQL binary/data major mismatch; no migration or reimport permitted')
    return dict(status='postgres_major_compatible',role=role,data_major=cluster,binary_version=match[1],
        pg_version_sha256=row['sha256'],inventory_content_sha256=inventory['content_sha256'],
        native_probe_sha256=json_digest(probe),image_config_sha256=config_sha,
        postgis_catalog_compatibility_verified=False,native_database_startup_verified=False,ready_to_host=False)


def osrm_version(probe,context,role,config_sha,binary='/usr/local/bin/osrm-routed'):
    if role not in ('osrm-car','osrm-bike','osrm-foot'):raise Closed('Explicit routing profile required')
    text=probe_text(probe,context,role,config_sha,[binary,'--version']).strip()
    if text not in ('v5.27.1','5.27.1'):raise Closed('Pinned OSRM5.27.1 binary differs')
    return dict(status='osrm_binary_version_verified',role=role,version='5.27.1',native_probe_sha256=json_digest(probe),
        mld_data_load_verified=False,ready_to_host=False)


def reset_controls(context,proof,*,base_identity_sha256,query_contract_sha256):
    """Check saved native mutation/reset controls; does not enable the adapter."""
    if (proof.get('protocol')!='map_native_reset_controls_v1' or proof.get('context_sha256')!=json_digest(context) or
        proof.get('base_identity_sha256')!=base_identity_sha256 or proof.get('query_contract_sha256')!=query_contract_sha256):
        raise Closed('Exact native reset controls and immutable base identity required')
    a,b=proof['state_a'],proof['state_b'];base=checked(proof['base_path'])
    paths=[checked(row['path']) for row in (a,b)]
    scratch=checked(proof['scratch_root'])
    if (paths[0]==paths[1] or paths[0] in paths[1].parents or paths[1] in paths[0].parents or
        any(scratch not in path.parents or path==base or base in path.parents or path in base.parents for path in paths) or
        a.get('instance_id')==b.get('instance_id') or a.get('network_namespace')==b.get('network_namespace') or
        any(row.get('fresh_claim') is not True or row.get('controller_pid')!=context['controller_pid'] or
            not isinstance(row.get('instance_id'),str) or not row['instance_id'] or
            re.fullmatch(r'net:\[[0-9]+\]',str(row.get('network_namespace'))) is None or
            row.get('base_identity_sha256')!=base_identity_sha256 for row in (a,b))):raise Closed('Distinct fresh owned frontend states required')
    observed=proof['native_observations']
    keys=('a_before','a_after_mutation','b_fresh')
    if any(not isinstance(observed.get(key),dict) for key in keys):raise Closed('Native before/mutation/fresh observations required')
    before,changed,fresh=(observed[key] for key in keys)
    if (any(not isinstance(row.get('query_fingerprints'),dict) or not row['query_fingerprints'] or
            any(not isinstance(k,str) or not k or re.fullmatch('[0-9a-f]{64}',str(v)) is None for k,v in row['query_fingerprints'].items()) or
            type(row.get('mutation_marker_count')) is not int for row in (before,changed,fresh)) or
        set(changed['query_fingerprints'])!=set(before['query_fingerprints']) or
        fresh.get('query_fingerprints')!=before['query_fingerprints'] or
        changed.get('query_fingerprints')==before['query_fingerprints'] or
        [x.get('mutation_marker_count') for x in (before,changed,fresh)]!=[0,1,0]):
        raise Closed('Native mutation was not demonstrated or leaked into fresh state')
    closures=proof.get('closures',{})
    if any(closures.get(key,{}).get('remaining_owned_processes')!={} for key in ('state_a','state_b','shared_backends')):
        raise Closed('Every owned state/backend must be closed')
    return dict(status='native_reset_controls_consistent',proof_sha256=json_digest(proof),
        original_data_identity_verified=False,production_adapter_enabled=False,ready_to_host=False,
        remaining_gate='Independently review actual native SQL/browser probe implementation, original data identity and service/namespace evidence before enabling Map.')
