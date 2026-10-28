#!/usr/bin/env python3
"""Prospective fixed-cohort owner; preview by default, never requests resources.

Hosting runners exist for original Reddit and the validated third-party Magento
images, CMS (shopping_admin) and Shopping; one hosted domain per campaign. Other domains remain pending,
without task replacement. Execution needs an exact hash-bound resource,
API, protocol and ephemeral-state approval in an existing matching allocation.
Frozen Reddit pilot/CPU packages are not used as writable execution templates.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import fcntl
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.webprm_campaign_queue import ClaimStore, checked, identity, run_queue, write_once
from openwebrl.webprm_webarena import digest, TASK_FILE, verify_source
from openwebrl.webprm_webarena_sites import SITE_RUNTIME_CONFIGURATION, registry_status
from openwebrl.webprm_webarena_helper_runtime import HELPER_CONFIGURATION
from openwebrl.webprm_evaluator_transcript import TRANSCRIPT_CONFIGURATION
from openwebrl.webprm_ephemeral_state import PROTOCOL as DISPOSAL_PROTOCOL
from openwebrl.webprm_lifecycle import _process, _processes, _subreaper, OwnedEpisodeTree
from openwebrl.webprm_worker import loopback_urlopen
from scripts.prepare_webprm_table5_campaign import (TASKS, TASKS_SHA, cohort as fixed_cohort,
                                                  digest as cohort_task_digest)
from scripts.run_webprm_webarena_pilot import REQUIRED_CODE, validate_plan
from openwebrl.webprm_webarena_worker import CAMPAIGN_SCIENCE
from scripts.launch_webprm_reddit_pilot import (BROWSER, REPO, RUNTIME, SCRATCH, ORIGINAL_MANIFEST,
    environment, reference, model_command, piotr_compatible_config, cpu_scheduler_epochs,
    _command, _stop, website_ready, GUARDED_NO_SEARCH, GUARDED_EVALUATOR,
    CORRECTED_TURN_LIMIT, CORRECTED_PIOTR_REPRESENTATION)

CONTROLLER_PROTOCOL = 'fixed_table5_original_site_queue_v1'
CRITICS = ('shepherd8b', 'arbiter7b', 'piotr_selectionarm')
PREFIX = dict(zip(CRITICS, ('shepherd', 'arbiter', 'piotr')))
HOSTED_DOMAINS = {'reddit','shopping_admin','shopping'}
# Both third-party Magento sites share one stage script and auditor (site module per domain).
MAGENTO_DOMAINS = ('shopping_admin','shopping')
STAGE_SCRIPTS = {'reddit':'scripts/run_webprm_table5_reddit.py','shopping_admin':'scripts/run_webprm_table5_cms.py',
    'shopping':'scripts/run_webprm_table5_cms.py'}
# Startup includes the native DB/origin configuration for CMS (validated ~90 s);
# its graceful supervisord shutdown needs more than Reddit's 30 s stop window.
SITE_STARTUP_SECONDS = {'reddit':180,'shopping_admin':900,'shopping':1200}
SITE_STOP_SECONDS = {'reddit':30,'shopping_admin':150,'shopping':150}
SITE_LABELS = {'reddit':'Reddit','shopping_admin':'CMS','shopping':'Shopping'}
READINESS = {'reddit':'from scripts.launch_webprm_reddit_pilot import website_ready; website_ready()',
    'shopping_admin':'from openwebrl.webprm_cms_site import website_ready; website_ready()',
    'shopping':'from openwebrl.webprm_shopping_site import website_ready; website_ready()'}
AUDITORS = {'reddit':'scripts/audit_webprm_table5_cell.py','shopping_admin':'scripts/audit_webprm_table5_cms_cell.py',
    'shopping':'scripts/audit_webprm_table5_cms_cell.py'}
CMS_CODE = ('scripts/run_webprm_table5_cms.py','scripts/audit_webprm_table5_cms_cell.py','openwebrl/webprm_cms_site.py','scripts/validate_webprm_cms_hosting.py',
    'openwebrl/webprm_cms_hosting.py','openwebrl/webprm_cms_materializer.py','openwebrl/webprm_cms_native.py')
SHOPPING_CODE = CMS_CODE + ('openwebrl/webprm_shopping_site.py','openwebrl/webprm_shopping_native.py',
    'scripts/validate_webprm_shopping_hosting.py')
# Shared Magento stage and auditor import the Shopping adapter even for CMS.
# Freeze that transitive dependency so real entrypoint imports cannot fail.
CMS_CODE = SHOPPING_CODE
SITE_CODE = {'shopping_admin':CMS_CODE,'shopping':SHOPPING_CODE}
SITE_MODULES = {'shopping_admin':'openwebrl.webprm_cms_site','shopping':'openwebrl.webprm_shopping_site'}
# Storage/memory sampling per hosted domain. The Shopping image has ~4.5M files per tree
# (2.25 GB materialization manifest, probe 358020): a base plus two states is >10M files,
# so `du` needs minutes and is sampled every 5 minutes instead of every 15 s; and ~170 GB
# of materialize/copy writes fill the job cgroup's page cache, which memory.current counts
# although the kernel reclaims it, so Shopping admission excludes reclaimable file cache.
STORAGE_SAMPLE_SECONDS = {'shopping':300,'shopping_admin':300}
DU_TIMEOUT_SECONDS = {'shopping':1800,'shopping_admin':1800}
MEMORY_ACCOUNTING = {'shopping':'cgroup_minus_reclaimable_file_cache'}


def du_options(timeout):
    """Only non-default domains pass a timeout; Reddit/CMS scans keep their exact call."""
    return {} if timeout==15 else dict(timeout=timeout)
CRITIC_REFERENCE = RUNTIME/'next-stage-preparation/reddit-corrected-pilot-v3/combined-v2/proposal.json'
EXTRA_CODE = ('scripts/launch_webprm_table5_campaign.py', 'scripts/submit_webprm_table5_campaign.py', 'scripts/run_webprm_table5_reddit.py',
    'scripts/launch_webprm_reddit_pilot.py', 'scripts/prepare_webprm_table5_campaign.py',
    'openwebrl/webprm_campaign_queue.py', 'openwebrl/webprm_concurrent_broker.py',
    'openwebrl/webprm_budget_journal.py', 'openwebrl/webprm_ephemeral_state.py',
    'openwebrl/webprm_evaluator_transcript.py', 'openwebrl/webprm_webarena_helpers.py',
    'openwebrl/webprm_webarena_helper_runtime.py', 'openwebrl/webprm_reddit_pilot_preflight.py',
    'scripts/audit_webprm_table5_cell.py', 'openwebrl/webprm_journal_audit.py',
    'openwebrl/webprm_table5_analysis.py', 'scripts/analyze_webprm_webarena_pilot.py',
    'scripts/runtime_ports.py', 'scripts/prepare_webprm_piotr_processor.py')


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def put(path, value): write_once(checked(path), value)
def cell_key(cell): return hashlib.sha256(cell['cell_id'].encode()).hexdigest()
def broker_cell_id(cell): return f"webarena_lite_{cell['task_id']:03d}/"+cell['actor']+'__'+cell['selector']


def native_authentication_failure(result):
    return (result.get('valid') is False and result.get('error_type') == 'EvaluationNotReady'
            and str(result.get('error', '')).startswith('Native authentication failed:'))


def artifact(path):
    path=checked(path)
    return dict(path=str(path),sha256=sha(path))


def frozen_cohort(protocol_sha256):
    """Commit science without a proposal/approval/cohort hash cycle."""
    if re.fullmatch('[0-9a-f]{64}',str(protocol_sha256)) is None:
        raise ValueError('Explicit frozen scientific protocol SHA256 required')
    value=fixed_cohort(read(TASKS))
    value.update(protocol_sha256=protocol_sha256,status='protocol_frozen_resources_separately_approved',
        warning='Fixed reconstruction cohort; exact resources require a separate hash-bound proposal approval.')
    # The separate approval binds the proposal, which binds this cohort file.
    # Embedding that approval hash here would create a circular identity.
    return value


def route_contract(actors):
    """Extend the validated actor routes to each fixed actor, keeping critics fixed."""
    prior=read(CRITIC_REFERENCE);routes=[];actor_purposes={'policy','state_diff','terminal'}
    for source in prior['broker_routes']:
        route=deepcopy(source)
        if route['local']:
            endpoint=route.pop('endpoint_key')
            route['critic']=next(name for name,prefix in PREFIX.items() if endpoint==prefix+'_endpoint')
        if route['purpose'] in actor_purposes:
            routes.extend(dict(route,model=actor) for actor in actors)
        else:routes.append(route)
    cells={actor:{selector:[[actor if purpose in actor_purposes else model,purpose]
        for model,purpose in rows] for selector,rows in prior['cell_routes'].items()} for actor in actors}
    return routes,cells


def audit_input(proposal, output, folder, cell, plan, status):
    """Freeze independently replayable evidence; this function makes no verdict."""
    episode=folder/'episodes'/f"webarena_lite_{cell['task_id']:03d}"/(cell['actor']+'__'+cell['selector'])
    native=episode/'native-evaluation'
    names=dict(result=episode/'result.json',watchdog=episode/'watchdog.json',rollout=episode/'rollout.json',
        api=episode/'api.jsonl',provenance=episode/'provenance.json',authentication=episode/'authentication.json',
        lease=folder/'lease.json',instance=folder/'site/instance.json',site_closure=folder/'site/closure.json',
        copy_receipt=folder/'copy/receipt.json',state_claim=folder/'state-claim.json',closure=folder/'closure.json',
        trajectory=native/'trajectory.json',terminal=native/'terminal.json',terminal_screenshot=native/'terminal.png',
        verdict=native/'native-verdict.json',outcome=native/'outcome-before-close.json',
        evaluator_inputs=episode/'native-evaluator-transcript.json',native_fuzzy=episode/'native-fuzzy.jsonl')
    refs={name:artifact(path) for name,path in names.items() if path.is_file()}
    refs['shared_base_evidence']=artifact(output/'base-evidence.json')
    for index,path in enumerate(sorted(episode.rglob('*'))):
        if path.is_file() and path.name!='storage-state-private.json':refs['episode_file_'+str(index)]=artifact(path)
    broker=output/'broker.jsonl';raw=broker.read_bytes()
    # The target cell is closed. Other cells may append concurrently; only
    # complete broker records belong in this immutable read-only prefix.
    raw=raw[:raw.rfind(b'\n')+1]
    budget=status['budget']
    return dict(schema_version=1,campaign_root=proposal['campaign_root'],cell=cell,
        cohort=artifact(proposal['cohort_path']),worker_plan=artifact(folder/'worker-plan.json'),artifacts=refs,
        budget=dict(financial_contract=read(output/'financial-contract.json'),
            event_path=plan['budget_path']+'.events.jsonl',prefix_bytes=budget['budget_prefix_bytes'],
            prefix_sha256=budget['budget_ledger_sha256']),
        broker=dict(cell_id=broker_cell_id(cell),audit_path=str(broker),prefix_bytes=len(raw),
            prefix_sha256=hashlib.sha256(raw).hexdigest()),source_sha256=proposal['source_sha256'])


def validate_proposal(proposal):
    if proposal.get('protocol') != CONTROLLER_PROTOCOL:
        raise ValueError('Explicit prospective full-cohort controller protocol required')
    if sha(TASKS) != TASKS_SHA:
        raise ValueError('Pinned released 155-task source changed')
    expected = frozen_cohort(proposal['protocol_sha256'])
    supplied = read(proposal['cohort_path'])
    supplied = supplied.get('cohort', supplied)
    if sha(proposal['cohort_path']) != proposal['cohort_file_sha256'] or supplied != expected:
        raise ValueError('Fixed 1550-cell membership, order or source differs')
    if proposal['cohort_sha256'] != digest(expected):
        raise ValueError('Fixed cohort digest differs')
    routes,cell_routes=route_contract(expected['actors'])
    if (proposal.get('broker_routes')!=routes or proposal.get('cell_routes')!=cell_routes
            or proposal.get('env_file')!=read(CRITIC_REFERENCE)['env_file']):
        raise ValueError('Exact actor/critic routes and preserved provider environment required')
    checked(proposal['campaign_root'])
    source = proposal['source_sha256']
    if any(str(REPO / name) not in source for name in (*REQUIRED_CODE, *EXTRA_CODE)):
        raise ValueError('Complete controller/worker/broker/helper source freeze required')
    for path, value in source.items():
        if sha(path) != value:
            raise ValueError('Frozen prospective source changed: ' + path)
    if sha(proposal['worker_template']) != proposal['worker_template_sha256']:
        raise ValueError('Worker template changed')
    template = read(proposal['worker_template'])
    if any(source.get(path)!=value for path,value in template['code_sha256'].items()):
        raise ValueError('Worker/controller implementation freezes differ')
    for key,value in (('no_search_prompt_version',GUARDED_NO_SEARCH),('evaluator_protocol',GUARDED_EVALUATOR),
                      ('turn_limit_protocol',CORRECTED_TURN_LIMIT),('piotr_representation',CORRECTED_PIOTR_REPRESENTATION)):
        if template.get(key)!=value:raise ValueError('Prospective scientific protocol differs: '+key)
    for key, value in (('site_runtime', SITE_RUNTIME_CONFIGURATION), ('native_helpers', HELPER_CONFIGURATION),
                       ('evaluator_observations', TRANSCRIPT_CONFIGURATION),('campaign_science',CAMPAIGN_SCIENCE)):
        if template.get(key) != value:
            raise ValueError('Prospective common protocol lacks explicit ' + key)
    validate_plan(template, allow_missing_endpoints=True)
    if digest(template['protocol_contract']) != proposal['protocol_sha256']:
        raise ValueError('Common scientific protocol changed')
    resources, limits = proposal['resources'], proposal['limits']
    if (set(resources) != {'gpus','gpu_type','cpus','memory_gib','total_seconds'}
            or resources['gpu_type'] != 'H200'
            or any(type(resources[k]) is not int or resources[k] <= 0
                   for k in ('gpus','cpus','memory_gib','total_seconds'))):
        raise ValueError('Exact positive H200/CPU/RAM/cumulative-time envelope required')
    integers = ('paid_requests','local_requests','total_requests','scratch_bytes','runtime_reserve_bytes',
        'max_workers','per_cell_memory_bytes','fixed_memory_bytes','min_start_seconds','cleanup_seconds','retained_bytes')
    if any(type(limits.get(k)) is not int or limits[k] <= 0 for k in integers):
        raise ValueError('Exact positive dispatch/storage/concurrency/memory/time caps required')
    if (type(limits.get('api_usd')) not in (int,float) or not math.isfinite(limits['api_usd'])
            or limits['api_usd'] <= 0 or limits['max_workers'] > 8
            or limits['total_requests'] != limits['paid_requests'] + limits['local_requests']
            or limits['cleanup_seconds'] < 180
            or limits['fixed_memory_bytes'] + limits['max_workers'] * limits['per_cell_memory_bytes'] > resources['memory_gib'] * 2**30):
        raise ValueError('Inconsistent explicit cumulative resource caps')
    if proposal.get('ephemeral_policy', {}).get('protocol') != DISPOSAL_PROTOCOL:
        raise ValueError('Explicit future-only audited ephemeral state policy required')
    if set(proposal['models']) != set(CRITICS):
        raise ValueError('All three pinned critic servers are required')
    if str(CRITIC_REFERENCE) not in source:raise ValueError('Validated three-critic reference must be frozen')
    reference_models=read(CRITIC_REFERENCE)['models']
    for name, model in proposal['models'].items():
        if {key:value for key,value in model.items() if key!='gpu_index'} != reference_models[name]:
            raise ValueError('Critic identity/context/loading settings differ from validated pilot')
        if (sha(Path(model['path'])/'config.json') != model['config_sha256']
                or type(model.get('gpu_index')) is not int or not 0 <= model['gpu_index'] < resources['gpus']):
            raise ValueError('Pinned critic/config/GPU assignment differs')
        if any((Path(model['path'])/f).stat().st_size != size for f,size in model['weight_sizes'].items()):
            raise ValueError('Pinned previously verified critic weights changed size')
    magento=[d for d in MAGENTO_DOMAINS if d in proposal['domains']]
    if magento:
        # A Magento campaign hosts only one validated third-party image; Reddit checks below are unchanged.
        if len(proposal['domains'])!=1:
            raise ValueError('A Magento (CMS/Shopping) campaign hosts exactly one domain')
        validate_magento_domain(magento[0],proposal['domains'][magento[0]],source)
    else:
        reddit = proposal['domains'].get('reddit')
        if (not isinstance(reddit,dict) or type(reddit.get('state_apparent_bytes')) is not int
                or reddit['state_apparent_bytes'] <= 0 or reddit.get('endpoint') != 'http://127.0.0.1'):
            raise ValueError('Explicit original Reddit state bound and native endpoint required')
        for key in ('download_receipt','raw_verification','startup_review'):
            if str(reddit[key]) not in source:
                raise ValueError('Original Reddit source/readiness receipts must be hash-bound')
        if any(str(RUNTIME / name) not in source for name in
               ('prepare_reddit_original.py','run_reddit_instance.py','start_reddit_services.sh')):
            raise ValueError('Unchanged original Reddit hosting primitives must be frozen')
    nltk = proposal['nltk_environment']
    base = RUNTIME/'nltk-runtime'
    if nltk != dict(PYTHONPATH=str(base/'venv/lib/python3.12/site-packages')+':'+str(REPO),
                    NLTK_DATA=str(base/'nltk_data')):
        raise ValueError('Preserved NLTK package and tokenizer paths must be explicit')
    return expected, template


def validate_magento_domain(domain, entry, source):
    import importlib
    site=importlib.import_module(SITE_MODULES[domain]);label=SITE_LABELS[domain]
    if (not isinstance(entry,dict) or set(entry)!={'endpoint','state_apparent_bytes','hosting_proposal',
            'hosting_validation','base_content_sha256'} or entry['endpoint']!=site.ENDPOINT
            or type(entry['state_apparent_bytes']) is not int or entry['state_apparent_bytes']<=0
            or entry['hosting_proposal']!=site.HOSTING_PROPOSAL or entry['hosting_validation']!=site.HOSTING_VALIDATION
            or entry['base_content_sha256']!=site.BASE_CONTENT_SHA256):
        raise ValueError('Explicit validated '+label+' hosting recipe, state bound and endpoint required')
    if any(str(REPO/name) not in source for name in SITE_CODE[domain]):
        raise ValueError(label+' hosting/stage/site adapter source must be frozen')
    for key in ('hosting_proposal','hosting_validation'):
        if source.get(entry[key]['path'])!=entry[key]['sha256']:
            raise ValueError(label+' hosting proposal/validation receipts must be hash-bound')
    site.hosting()


def validate_cms_domain(entry, source):
    validate_magento_domain('shopping_admin',entry,source)


def preview(proposal):
    cohort, template = validate_proposal(proposal)
    ready = registry_status()
    return dict(status='prepared_not_executed', planned_cells=len(cohort['cells']),
        currently_implemented_domains=[d for d in proposal['domains'] if d in HOSTED_DOMAINS and ready[d]['live_adapter_ready']],
        blocked_domains=[d for d in cohort['domain_counts'] if d not in proposal['domains'] or d not in HOSTED_DOMAINS
                         or not ready[d]['live_adapter_ready']],
        resources=proposal['resources'], limits=proposal['limits'],
        launches_models=list(CRITICS), fixed_primary_attempts_per_cell=1,
        disposal_requires_independent_audit=True, scratch_accounting='apparent_bytes',
        allocations=0, provider_calls=0, browser_calls=0)


def allocation_contract(proposal):
    """Resource/science identity survives reviewed operational source repairs."""
    fields=('campaign_root','cohort_sha256','protocol_sha256','resources','limits',
        'allocation_policy','ephemeral_policy','models','broker_routes','cell_routes','domains')
    return digest({key:proposal[key] for key in fields})


def allocation_history(proposal):
    root=checked(proposal['campaign_root']);folder=root/'allocation-history'
    rows=[]
    if folder.exists():
        for path in sorted(folder.glob('*.json')):
            row=read(checked(path));index=len(rows)+1
            if (row.get('schema_version')!=1 or row.get('sequence')!=index
                    or not isinstance(row.get('job_id'),str) or not row['job_id'].isdigit()
                    or row.get('previous_job_ids')!=[r['job_id'] for r in rows]
                    or row.get('proposal_sha256')!=digest(row.get('proposal'))
                    or row.get('allocation_contract_sha256')!=allocation_contract(proposal)
                    or allocation_contract(row['proposal'])!=allocation_contract(proposal)
                    or path.name!=f"{index:04d}-{row['job_id']}.json"
                    or row['job_id'] in {r['job_id'] for r in rows}):
                raise ValueError('Durable allocation history or unchanged resource/science contract differs')
            rows.append(row)
    known={r['job_id'] for r in rows}
    if root.exists() and any(p.name.removeprefix('attempt-') not in known for p in root.glob('attempt-*')):
        raise ValueError('Unregistered historical allocation directory; independent accounting required')
    return rows


def attempt_cap(proposal,approval):
    """Allocation-slot cap: the original three, plus a separately approved exception.

    The exception lives outside allocation_policy, so the cumulative resource and
    science contract (time, API, storage, cohort) of earlier attempts is unchanged.
    """
    if proposal.get('allocation_policy',{}).get('maximum_attempts')!=3:
        raise ValueError('Original three-attempt cumulative allocation policy required')
    exception=proposal.get('allocation_exception');cap=3
    if exception is not None:
        extra=exception.get('additional_attempts')
        if (type(extra) is not int or not 1<=extra<=5
                or exception.get('cumulative_seconds')!=proposal['resources']['total_seconds']
                or approval.get('allocation_exception_sha256')!=digest(exception)
                or not isinstance(approval.get('allocation_exception_user_approval'),str)
                or not approval['allocation_exception_user_approval'].strip()):
            raise ValueError('Additional allocation slots require an exact approved exception')
        cap+=extra
    if approval.get('maximum_allocation_attempts')!=cap:
        raise ValueError('Approved allocation-attempt cap differs')
    return cap


def scratch_accounting(proposal,approval):
    """Which prior retained scratch counts against limits['scratch_bytes'].

    Default 'cumulative': every earlier attempt's retained node-local bytes.
    An exactly approved exception may switch to 'per_node_live': only earlier
    attempts' retained state still present on this node's scratch. The cap value
    itself and the free-disk check are unchanged; retained states stay preserved.
    """
    exception=proposal.get('scratch_accounting_exception')
    if exception is None:return 'cumulative'
    if (exception.get('rule')!='per_node_live' or exception.get('scratch_bytes')!=proposal['limits']['scratch_bytes']
            or approval.get('scratch_accounting_exception_sha256')!=digest(exception)
            or not isinstance(approval.get('scratch_accounting_exception_user_approval'),str)
            or not approval['scratch_accounting_exception_user_approval'].strip()):
        raise ValueError('Per-node scratch accounting requires an exact approved exception')
    return 'per_node_live'


def prior_scratch_charge(rule,finals):
    """finals: storage-final receipts of earlier attempts (scratch path, apparent bytes)."""
    if rule=='cumulative':return sum(f['apparent_bytes'] for f in finals)
    return sum(f['apparent_bytes'] for f in finals if Path(f['scratch']).exists())


def closed_attempt_storage(proposal, job):
    """Keep never-started held cancellations distinct from executed closures.

    An absent controller receipt is acceptable only for a source-bound review
    and independent Slurm proof of zero runtime, no start and no allocation.
    This does not refund or omit the consumed submission slot.
    """
    folder=checked(Path(proposal['campaign_root'])/('attempt-'+job))
    if folder.exists():
        if read(folder/'controller-closure.json').get('remaining_owned_processes')!={}:
            raise ValueError('Previous allocation process closure is incomplete')
        storage=read(folder/'storage-final.json')
        if type(storage.get('apparent_bytes')) is not int or storage['apparent_bytes']<0:
            raise ValueError('Previous retained state accounting is incomplete')
        return storage
    review=proposal.get('never_started_attempts',{}).get(job)
    if not review or sha(checked(review['path']))!=review['sha256']:
        raise ValueError('Missing executed closure requires pinned never-started review')
    proof=read(review['path'])
    if (proof.get('job_id')!=job or proof.get('status')!='cancelled_held_before_any_allocation'
            or proof.get('elapsed_seconds')!=0 or proof.get('start_time')!='None'
            or proof.get('allocated_tres')!='' or proof.get('controller_launched') is not False
            or proof.get('campaign_attempt_exists') is not False
            or proof.get('primary_claims')!=0 or proof.get('model_calls')!=0):
        raise ValueError('Incomplete never-started submission evidence')
    raw=subprocess.check_output(['sacct','-X','-j',job,'-n','-P',
        '--format=JobID,State,ElapsedRaw,Start,AllocTRES'],text=True,timeout=15)
    rows=[line.split('|') for line in raw.splitlines() if line.strip()]
    if (len(rows)!=1 or len(rows[0])!=5 or rows[0][0]!=job
            or rows[0][1].split()[0]!='CANCELLED' or rows[0][2:]!=['0','None','']):
        raise ValueError('Scheduler does not prove a never-allocated cancellation')
    return None


def register_allocation(proposal,approval,job):
    """Register an approved held submission before release; never submits a job.

    Prior proposal bodies preserve operational source lineage. A source-only
    repair may use a newly reviewed exact proposal under the identical cumulative
    allocation contract; it cannot reset caps, cohort, protocol or attempt history.
    """
    if (approval.get('status')!='approved' or approval.get('proposal_sha256')!=digest(proposal)
            or approval.get('resources')!=proposal['resources'] or approval.get('limits')!=proposal['limits']
            or approval.get('protocol_sha256')!=proposal['protocol_sha256']
            or approval.get('ephemeral_policy_sha256')!=digest(proposal['ephemeral_policy'])
            or approval.get('job_id') not in (None,job) or not isinstance(job,str) or not job.isdigit()):
        raise ValueError('Exact approved bounded allocation registration required')
    cap=attempt_cap(proposal,approval)
    root=checked(proposal['campaign_root']);root.mkdir(mode=0o700,exist_ok=True)
    if root.stat().st_mode & 0o077:raise ValueError('Allocation registry requires a private campaign root')
    folder=root/'allocation-history';folder.mkdir(mode=0o700,exist_ok=True)
    fd=os.open(checked(folder/'owner.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=allocation_history(proposal);known=[r['job_id'] for r in rows]
        prefix=approval.get('attempt_job_ids')
        if not isinstance(prefix,list) or len(prefix)>=cap or len(prefix)!=len(set(prefix)) or job in prefix:
            raise ValueError(f'At most {cap} distinct cumulative allocation attempts are permitted'
                +('' if cap>3 else ' (three distinct without an approved exception)'))
        expected=dict(schema_version=1,sequence=len(prefix)+1,job_id=job,previous_job_ids=prefix,
            proposal_sha256=digest(proposal),allocation_contract_sha256=allocation_contract(proposal),proposal=proposal)
        if known==prefix+[job]:
            if rows[-1]!=expected:raise ValueError('Registered allocation identity changed')
        elif known==prefix:
            put(folder/f"{len(prefix)+1:04d}-{job}.json",expected)
        else:raise ValueError('Supplied attempts omit or reorder durable allocation history')
        return expected
    finally:os.close(fd)


def validate_registered_allocation(proposal,approval):
    attempts=approval.get('attempt_job_ids');job=str(approval.get('job_id'))
    rows=allocation_history(proposal)
    cap=attempt_cap(proposal,approval)
    if (not isinstance(attempts,list) or not 1<=len(attempts)<=cap
            or attempts!=[r['job_id'] for r in rows] or attempts[-1]!=job
            or rows[-1]['proposal_sha256']!=digest(proposal)):
        raise ValueError('Complete registered allocation history and approved attempt cap required')
    return rows


def approved_context(proposal, approval, scheduler_text, scheduler_json, prior_seconds, *, now=None):
    if (approval.get('status') != 'approved' or approval.get('proposal_sha256') != digest(proposal)
            or approval.get('resources') != proposal['resources'] or approval.get('limits') != proposal['limits']
            or approval.get('protocol_sha256') != proposal['protocol_sha256']
            or approval.get('ephemeral_policy_sha256') != digest(proposal['ephemeral_policy'])):
        raise ValueError('Exact resource/API/protocol/ephemeral proposal is not approved')
    job = str(approval['job_id']); now = time.time() if now is None else now
    if os.environ.get('SLURM_JOB_ID') != job:
        raise ValueError('Controller must run inside the approved existing allocation')
    fields = dict(re.findall(r'(\w+)=(\S+)', scheduler_text))
    tres = dict(item.split('=',1) for item in fields.get('AllocTRES','').split(',') if '=' in item)
    resources = proposal['resources']
    if (fields.get('JobId') != job or fields.get('JobState') != 'RUNNING' or fields.get('NumNodes') != '1'
            or fields.get('NumCPUs') != str(resources['cpus'])
            or not fields.get('UserId','').endswith(f'({os.getuid()})')
            or tres.get('gres/gpu:h200') != str(resources['gpus'])
            or tres.get('mem') not in (str(resources['memory_gib'])+'G',str(resources['memory_gib']*1024)+'M')):
        raise ValueError('Actual scheduler allocation differs from approval')
    epochs = cpu_scheduler_epochs(scheduler_json, job)
    if (type(prior_seconds) is not int or prior_seconds < 0
            or prior_seconds + epochs['limit_seconds'] > resources['total_seconds']):
        raise ValueError('Every prior allocation attempt counts against cumulative approved time')
    deadline = min(epochs['end_epoch'], epochs['start_epoch']+epochs['limit_seconds']) - proposal['limits']['cleanup_seconds']
    if not epochs['start_epoch'] <= now < deadline:
        raise ValueError('No approved runtime remains before mandatory cleanup')
    return dict(job_id=job, deadline=deadline, scheduler_epochs=epochs, prior_seconds=prior_seconds)


def cell_plan(template, proposal, context, cell, *, socket_path, ports):
    """Actor-specific fixed task plan; only execution bindings vary per cell."""
    raw = next(t for t in read(Path(template['arbiter_source'])/TASK_FILE) if t['task_id'] == cell['task_id'])
    # The committed cohort hashes JSON with ASCII escaping. Worker identity
    # uses literal UTF-8; those serializations differ on real Unicode tasks.
    # Verify with the original cohort serializer without changing either hash.
    if cohort_task_digest(raw) != cell['task_config_sha256']:
        raise ValueError('Cell task configuration differs from fixed cohort')
    plan = deepcopy(template)
    plan.update(authorization_status='approved', job_id=context['job_id'], deadline=context['deadline'],
        actor_model=cell['actor'],selector=cell['selector'],actors=[cell['actor']], selectors=[cell['selector']],
        task_ids=[cell['task_id']], tasks_sha256=digest([raw]),
        api_usd_cap=proposal['limits']['api_usd'], budget_path=str(Path(proposal['campaign_root'])/'shared-budget.json'),
        broker_socket=str(socket_path), endpoints={cell['domain']:proposal['domains'][cell['domain']]['endpoint']})
    for name, model in proposal['models'].items():
        plan[PREFIX[name]+'_model'] = model['served_model']
        plan[PREFIX[name]+'_endpoint'] = f'http://127.0.0.1:{ports[name]}/v1'
    return plan


def cell_result(path, cell, plan):
    result = read(path); watchdog = read(Path(path).parent/'watchdog.json')
    if (result.get('identity_sha256') != digest(plan) or result.get('task_id') != f"webarena_lite_{cell['task_id']:03d}"
            or result.get('arm') != cell['actor']+'__'+cell['selector']
            or result.get('actor_model') != cell['actor'] or result.get('selector') != cell['selector']
            or type(result.get('attempted')) is not bool or type(result.get('valid')) is not bool
            or type(result.get('success')) is not int or result['success'] not in (0,1)
            or result['success'] and not result['valid'] or watchdog != result.get('watchdog')
            or watchdog.get('remaining_owned_processes') != {}):
        raise ValueError('Durable assigned result or worker closure is incomplete')
    return dict(cell_id=cell['cell_id'], result_path=str(path), result_sha256=sha(path),
        attempted=result['attempted'], valid=result['valid'], success=result['success'],
        native_success=result.get('native_success'), audit_verified=False)


async def wait_process(process, deadline, check, *, allowed_returncodes=(0,)):
    while process.poll() is None:
        if time.time() >= deadline:
            raise TimeoutError('Shared allocation or stage deadline reached')
        check(); await asyncio.sleep(.5)
    if process.returncode not in allowed_returncodes:
        raise RuntimeError('Owned command failed; attempt retained without automatic retry')


async def wait_site_readiness(command, prefix, domain, folder, deadline, check):
    """Retry only readiness probes within the existing site/allocation deadline.

    Magento's HTTP probe permits ten seconds per request. Give its subprocess
    fifteen seconds, and retain/reap a timed-out probe before the next probe.
    No actor episode or model request has started at this point.
    """
    for attempt in range(180):
        check()
        if time.time() >= deadline:
            raise TimeoutError('Original '+SITE_LABELS[domain]+' HTTP readiness failed')
        probe = command(prefix+[BROWSER, '-B', '-c', READINESS[domain]],
                        folder/f'readiness-{attempt}.log')
        try:
            await wait_process(probe, min(deadline, time.time()+15), check)
        except (RuntimeError, TimeoutError):
            # Resource/deadline guards must still propagate; a local probe
            # timeout alone does not invalidate a fixed benchmark cell.
            check()
            if time.time() >= deadline:
                raise TimeoutError('Original '+SITE_LABELS[domain]+' HTTP readiness failed')
            await asyncio.sleep(1)
        else:
            return
        finally:
            _stop(probe, seconds=1)
    raise RuntimeError('Original '+SITE_LABELS[domain]+' readiness exhausted')


def apparent(path, *, stable=False, timeout=15):
    """Complete apparent-byte samples; retry only verified vanished descendants.

    PostgreSQL initialization can unlink a file between du's enumeration/stat.
    Its partial stdout is never an estimate. Real I/O/permission failures and a
    changed/missing root remain fatal. Final closed-owner accounting requires
    two equal complete samples, with at most four bounded scans in all cases.
    """
    path=checked(path);initial=path.lstat()
    if not stat.S_ISDIR(initial.st_mode):raise ValueError('Storage root must be a real directory')
    root_identity=(initial.st_dev,initial.st_ino,initial.st_mode)
    previous=None;maximum=0
    for attempt in range(4):
        result=subprocess.run(['du','--apparent-size','-s','-B1',str(path)],text=True,
            capture_output=True,timeout=timeout,env=dict(os.environ,LC_ALL='C',LANG='C'))
        current=path.lstat()
        if (current.st_dev,current.st_ino,current.st_mode)!=root_identity:
            raise RuntimeError('Storage root identity changed while measuring')
        if result.returncode==0:
            match=re.fullmatch(r'([0-9]+)\t'+re.escape(str(path))+r'\n',result.stdout)
            if not match or result.stderr:raise RuntimeError('Malformed or warned apparent-byte result')
            value=int(match[1]);maximum=max(maximum,value)
            if not stable or previous==value:return maximum
            previous=value
        else:
            # Locale and exact full-line parsing prevent permission errors,
            # unrelated paths or arbitrary du failures from being retried.
            messages=result.stderr.splitlines()
            if result.returncode!=1 or not messages:raise subprocess.CalledProcessError(result.returncode,result.args,result.stdout,result.stderr)
            for message in messages:
                match=re.fullmatch(r"du: (?:cannot access|cannot read directory) '([^\n]+)': No such file or directory",message)
                vanished=Path(match[1]) if match else None
                if vanished is None or path not in vanished.parents or '..' in vanished.parts:
                    raise subprocess.CalledProcessError(result.returncode,result.args,result.stdout,result.stderr)
                # A legitimate atomic replacement may already have recreated
                # the name. In either case discard this entire failed sample
                # and measure again; never reuse its partial byte total.
                try:vanished.lstat()
                except FileNotFoundError:pass
            previous=None
        if attempt<3:time.sleep(.1*(attempt+1))
    raise RuntimeError('Bounded apparent-byte scan did not obtain a complete'+(' stable' if stable else '')+' sample')


def storage_after_closure(closure,scratch,campaign,prior_retained,limits,timeout=15):
    if closure.get('remaining_owned_processes')!={}:
        raise RuntimeError('Cannot certify final storage before every owned process is closed')
    return dict(scratch=str(scratch),apparent_bytes=apparent(scratch,stable=True,**du_options(timeout)),
        cumulative_prior_retained_bytes=prior_retained,cap_bytes=limits['scratch_bytes'],
        campaign_apparent_bytes=apparent(campaign,stable=True,**du_options(timeout)),campaign_cap_bytes=limits['retained_bytes'],
        accounting='maximum complete samples; final equal-sample agreement after owned process closure')


def allocation_memory_bytes(job_id, accounting='cgroup_current', *, proc=Path('/proc/self/cgroup'), cgroups=Path('/sys/fs/cgroup')):
    """Read the owned job cgroup, never substitute whole-host free memory.

    'cgroup_minus_reclaimable_file_cache' subtracts the cgroup's own reclaimable page
    cache (file minus shmem, from the same cgroup's memory.stat); anonymous, kernel and
    tmpfs/shm memory stay charged.
    """
    if accounting not in ('cgroup_current','cgroup_minus_reclaimable_file_cache'):
        raise ValueError('Unknown allocation memory accounting')
    for line in proc.read_text().splitlines():
        _,controllers,relative=line.split(':',2)
        if controllers and 'memory' not in controllers.split(','):continue
        path=cgroups/('memory' if controllers else '')/relative.lstrip('/')
        for candidate in (path,*path.parents):
            if candidate.name!='job_'+str(job_id):continue
            value=candidate/('memory.usage_in_bytes' if controllers else 'memory.current')
            text=value.read_text().strip()
            if not text.isdigit():raise ValueError('Allocation cgroup memory usage is not finite')
            if accounting=='cgroup_current':return int(text)
            stats={}
            for row in (candidate/'memory.stat').read_text().splitlines():
                key,_,number=row.partition(' ')
                if number.strip().isdigit():stats[key]=int(number)
            names=('total_cache','total_shmem') if controllers else ('file','shmem')
            if not all(name in stats for name in names):raise ValueError('Allocation cgroup page-cache statistics unavailable')
            return max(0,int(text)-max(0,stats[names[0]]-stats[names[1]]))
    raise ValueError('Owned allocation memory cgroup is unavailable')


class CampaignRunner:
    """Concrete original-Reddit execution; no user-selected executable callbacks."""
    def __init__(self, proposal, context, output, scratch):
        self.proposal, self.context = proposal, context
        self.output, self.scratch = Path(output), Path(scratch)
        self.env = environment(self.scratch)
        self.env.update(proposal['nltk_environment'])
        os.environ.update(self.env)
        self.servers, self.children, self.plans, self.folders = [], [], {}, {}
        self.broker = self.port_lease = None
        self.reserved = {}; self.last_storage = 0.; self.used = 0; self.retained = 0
        self.halt_reason = None
        self.scratch_reservation_floor = 0

    @property
    def site_domain(self):
        return next((d for d in MAGENTO_DOMAINS if d in self.proposal.get('domains',{})),'reddit')

    @property
    def du_timeout(self):
        return DU_TIMEOUT_SECONDS.get(self.site_domain,15)

    def measure_storage(self, path, role):
        """Keep the initiating scan failure even if later cleanup also fails."""
        try:
            return apparent(path,**du_options(self.du_timeout))
        except subprocess.TimeoutExpired as error:
            receipt=self.output/'storage-measurement-timeout.json'
            if not receipt.exists():
                put(receipt,dict(protocol='table5_storage_timeout_diagnostic_v1',
                    epoch=time.time(),role=role,path=str(path),timeout_seconds=error.timeout,
                    error_type=type(error).__name__,partial_sample_used=False,
                    admissions_may_continue=False))
            raise

    def check(self):
        if self.halt_reason is not None:
            raise RuntimeError(self.halt_reason)
        if any(p.poll() is not None for p in self.servers):
            raise RuntimeError('An owned critic server exited')
        if self.broker is not None and not self.broker.process.is_alive():
            raise RuntimeError('Shared budget broker exited')
        if time.time() >= self.context['deadline']:
            raise TimeoutError('Allocation deadline reached')
        if time.monotonic()-self.last_storage > STORAGE_SAMPLE_SECONDS.get(self.site_domain,15):
            self.used = self.measure_storage(self.scratch,'node_scratch')
            self.retained = self.measure_storage(Path(self.proposal['campaign_root']),'durable_campaign')
            self.last_storage = time.monotonic()
            if self.retained > self.proposal['limits']['retained_bytes']:
                raise RuntimeError('Durable campaign artifact cap reached')
            charged=max(self.used,getattr(self,'scratch_reservation_floor',0))
            if charged + self.context['prior_retained_apparent_bytes'] > self.proposal['limits']['scratch_bytes']:
                raise RuntimeError('Cumulative retained apparent scratch cap reached')

    def command(self, argv, log, env=None):
        process = _command(argv, log, self.env if env is None else env)
        self.children.append(process)
        return process

    def service_command(self, mode, *, state, folder):
        return [BROWSER,'-B',str(REPO/STAGE_SCRIPTS[self.site_domain]),
            '--context',str(self.output/'execution-context.json'),'--mode',mode,
            '--state',str(state),'--output',str(folder)]

    def startup_capacity(self):
        """Reject a base-only fit before starting critics or paid preflights."""
        limits=self.proposal['limits']
        state=self.proposal['domains'][self.site_domain]['state_apparent_bytes']
        initial=apparent(self.scratch,**du_options(self.du_timeout));free=shutil.disk_usage(self.scratch).free
        required=2*state+limits['runtime_reserve_bytes']
        projected=initial+self.context['prior_retained_apparent_bytes']+required
        memory=limits['fixed_memory_bytes']+limits['per_cell_memory_bytes']
        value=dict(protocol='table5_minimum_startup_capacity_v1',initial_apparent_bytes=initial,
            prior_retained_apparent_bytes=self.context['prior_retained_apparent_bytes'],
            base_bytes=state,one_cell_bytes=state,runtime_reserve_bytes=limits['runtime_reserve_bytes'],
            minimum_projected_scratch_bytes=projected,scratch_cap_bytes=limits['scratch_bytes'],
            node_free_bytes=free,minimum_new_node_bytes=required,
            minimum_memory_bytes=memory,memory_cap_bytes=self.proposal['resources']['memory_gib']*2**30,
            approved_max_workers=limits['max_workers'],minimum_lanes=1,
            admission_allowed=(projected<=limits['scratch_bytes'] and free>=required and
                memory<=self.proposal['resources']['memory_gib']*2**30))
        put(self.output/'startup-capacity.json',value)
        if not value['admission_allowed']:
            raise ValueError('No room for the immutable base and one cell within approved resource caps')
        return initial

    async def start(self, cells):
        from scripts.runtime_ports import lease_ports
        from openwebrl.webprm_concurrent_broker import ConcurrentBrokerProcess
        from openwebrl.webprm_budget_journal import VERSION as BUDGET_VERSION
        from openwebrl.webprm_broker import make_broker_client
        from openwebrl.webprm_reddit_pilot_preflight import (native_fuzzy_preflight_cell,
            validate_local_critics, validate_guarded_fuzzy)
        # Preserve current apparent accounting even if this filesystem supports
        # reflinks. No physical-allocation shortcut is inferred from the tiny probe.
        limits=self.proposal['limits'];base_bytes=self.proposal['domains'][self.site_domain]['state_apparent_bytes']
        initial=self.startup_capacity()
        self.scratch_reservation_floor=initial+base_bytes+limits['runtime_reserve_bytes']
        build=self.command(self.service_command('build',state=self.scratch/'baseline',folder=self.output/'original-build'),
            self.output/'original-build.log')
        ports_dir=self.scratch/'ports';ports_dir.mkdir(mode=0o700)
        self.port_lease,base=lease_ports(self.context['job_id'],size=3,directory=ports_dir)
        ports={name:base+i for i,name in enumerate(CRITICS)}
        self.socket=self.scratch/'broker/socket';self.socket.parent.mkdir(mode=0o700)
        if len(os.fsencode(self.socket))>=108:raise ValueError('Broker socket path too long')
        self.template=read(self.proposal['worker_template'])
        for name,model in self.proposal['models'].items():
            compatible=None
            if name=='piotr_selectionarm':
                compatible=self.output/'piotr-compatible-config.json';put(compatible,piotr_compatible_config(model))
            env=dict(self.env,CUDA_VISIBLE_DEVICES=str(model['gpu_index']))
            argv=model_command(name,model,ports[name],compatible_config=compatible)
            put(self.output/(name+'-command.json'),dict(argv=argv,gpu_index=model['gpu_index']))
            self.servers.append(self.command(argv,self.output/(name+'-server.log'),env))
        until=min(self.context['deadline'],time.time()+1200);ready=set()
        while len(ready)<3:
            self.check()
            if time.time()>=until:raise TimeoutError('Critic startup deadline reached')
            for name in CRITICS:
                try:
                    with loopback_urlopen(f'http://127.0.0.1:{ports[name]}/v1/models',timeout=2) as response:
                        if self.proposal['models'][name]['served_model'] not in [v['id'] for v in json.load(response)['data']]:
                            raise ValueError('Served critic model differs')
                    ready.add(name)
                except OSError:pass
            await asyncio.sleep(1)
        registrations={};routes={}
        for route in self.proposal['broker_routes']:
            endpoint=None if not route['local'] else f"http://127.0.0.1:{ports[route['critic']]}/v1"
            routes[(route['model'],route['purpose'])]=dict(local=route['local'],endpoint=endpoint,
                max_input_tokens=route['max_input_tokens'],max_output_tokens=route['max_output_tokens'])
        for cell in cells:
            if not self.domain_ready(cell['domain']):continue
            folder=self.output/'cells'/cell_key(cell);folder.mkdir(parents=True,mode=0o700)
            plan=cell_plan(self.template,self.proposal,self.context,cell,socket_path=self.socket,ports=ports)
            put(folder/'worker-plan.json',plan);self.plans[cell['cell_id']]=plan;self.folders[cell['cell_id']]=folder
            journal=folder/'episodes'/f"webarena_lite_{cell['task_id']:03d}"/(cell['actor']+'__'+cell['selector'])/'api.jsonl'
            journal.parent.mkdir(parents=True,mode=0o700)
            registered=f"webarena_lite_{cell['task_id']:03d}/"+cell['actor']+'__'+cell['selector']
            allowed=self.proposal['cell_routes'][cell['actor']][cell['selector']]
            registrations[registered]=dict(config=dict(plan,identity_sha256=digest(plan),env_file=self.proposal['env_file']),
                journal=str(journal),allowed_routes=allowed)
        if not registrations:raise ValueError('No currently implemented pending domain cells')
        diagnostic_plan=next(iter(self.plans.values()));diagnostic=dict(diagnostic_plan,identity_sha256=digest(diagnostic_plan))
        registrations['local_preflight']=dict(config=diagnostic,journal=str(self.output/'local-preflight-api.jsonl'),
            allowed_routes=[list(k) for k,v in routes.items() if v['local']])
        registrations['native_fuzzy_preflight']=native_fuzzy_preflight_cell(diagnostic,self.proposal['env_file'],self.output)
        limits=self.proposal['limits']
        financial=dict(schema_version=1,budget_backend=BUDGET_VERSION,cohort_sha256=self.proposal['cohort_sha256'],
            protocol_sha256=self.proposal['protocol_sha256'],prices_usd_per_token=diagnostic['prices_usd_per_token'],
            api_usd_cap=limits['api_usd'],max_dispatches=limits['total_requests'],
            max_paid_dispatches=limits['paid_requests'],max_local_dispatches=limits['local_requests'])
        put(self.output/'financial-contract.json',financial)
        self.broker=ConcurrentBrokerProcess(socket_path=str(self.socket),cells=registrations,routes=routes,
            deadline=self.context['deadline'],max_dispatches=limits['total_requests'],
            max_paid_dispatches=limits['paid_requests'],max_local_dispatches=limits['local_requests'],
            max_concurrency=limits['max_workers'],financial_contract=financial,epoch_id='job-'+self.context['job_id'],
            previous_closed_epoch_sha256=self.context['approval'].get('previous_closed_epoch_sha256'),
            audit_path=str(self.output/'broker.jsonl')).start()
        await wait_process(build,self.context['deadline'],self.check)
        # Preserve small transitive reset/source evidence once per allocation.
        # Original image blobs remain hash-bound and are never rehashed per cell.
        base_paths=set(Path(p) for p in self.proposal['source_sha256'])
        base_paths.update(path for path in (self.output/'original-build').rglob('*') if path.is_file())
        base_paths.add(self.output/'original-build.log')
        base_refs=[]
        for path in sorted(base_paths):
            ref=artifact(path)
            with path.open('rb') as stream:os.fsync(stream.fileno())
            base_refs.append(dict(ref,bytes=path.stat().st_size))
        put(self.output/'base-evidence.json',dict(schema_version=1,artifacts=base_refs,
            preserve_original_sources_and_shared_receipts=True))
        for name,call in [('local_preflight',validate_local_critics),('native_fuzzy_preflight',validate_guarded_fuzzy)]:
            client=make_broker_client(str(self.socket),cell_id=name,identity_sha256=diagnostic['identity_sha256'],
                deadline=min(self.context['deadline'],time.time()+600))
            proof=await asyncio.to_thread(call,diagnostic,client,self.output)
            put(self.output/(name+'-receipt.json'),proof)

    def domain_ready(self, domain):
        return domain in HOSTED_DOMAINS and domain in self.proposal['domains'] and registry_status()[domain]['live_adapter_ready']

    def can_start(self, cell, active):
        self.check();limits=self.proposal['limits']
        decision=dict(protocol='table5_admission_decision_v1',cell_id=cell['cell_id'],
            active_workers=active,epoch=time.time(),deadline=self.context['deadline'])
        def finish(allowed,reason):
            folder=self.output/'admission-decisions';folder.mkdir(mode=0o700,exist_ok=True)
            sequence=getattr(self,'admission_sequence',0)
            put(folder/f'{sequence:05d}.json',dict(decision,allowed=allowed,reason=reason))
            self.admission_sequence=sequence+1
            return allowed
        decision['min_start_seconds']=limits['min_start_seconds']
        if getattr(self,'admission_halt_reason',None):
            return finish(False,self.admission_halt_reason)
        if self.site_domain=='shopping_admin' and active and not getattr(self,'cms_startup_verified',False):
            return finish(False,'cms_startup_canary_inflight')
        if decision['epoch']+limits['min_start_seconds']>=self.context['deadline']:
            return finish(False,'insufficient_start_time')
        memory=allocation_memory_bytes(self.context['job_id'],MEMORY_ACCOUNTING.get(self.site_domain,'cgroup_current'))
        decision.update(memory_accounting=MEMORY_ACCOUNTING.get(self.site_domain,'cgroup_current'),current_memory_bytes=memory,projected_memory_bytes=memory+(active+1)*limits['per_cell_memory_bytes'],
            memory_cap_bytes=self.proposal['resources']['memory_gib']*2**30)
        if decision['projected_memory_bytes']>decision['memory_cap_bytes']:
            return finish(False,'allocation_memory_capacity')
        required=self.proposal['domains'][cell['domain']]['state_apparent_bytes']
        self.used=self.measure_storage(self.scratch,'admission_scratch');self.last_storage=time.monotonic()
        projected=self.used+self.context['prior_retained_apparent_bytes']+sum(self.reserved.values())+required+limits['runtime_reserve_bytes']
        free=shutil.disk_usage(self.scratch).free
        decision.update(current_scratch_bytes=self.used,prior_retained_apparent_bytes=self.context['prior_retained_apparent_bytes'],
            outstanding_reservation_bytes=sum(self.reserved.values()),new_cell_bytes=required,
            runtime_reserve_bytes=limits['runtime_reserve_bytes'],projected_scratch_bytes=projected,
            scratch_cap_bytes=limits['scratch_bytes'],node_free_bytes=free)
        if projected>limits['scratch_bytes']:return finish(False,'cumulative_scratch_capacity')
        if free<required+limits['runtime_reserve_bytes']:return finish(False,'node_free_disk_capacity')
        status=self.broker.status()
        budget=status['budget']
        decision.update(broker_admission_open=status.get('admission_open'),broker_halt_reason=budget.get('halt_reason'),
            broker_reservation_rejection=budget.get('reservation_rejection'),
            budget_remaining_usd=budget['remaining'],paid_dispatches=budget['paid_dispatches'],
            local_dispatches=budget['local_dispatches'],paid_cap=limits['paid_requests'],local_cap=limits['local_requests'])
        if budget.get('reservation_rejection') is not None:
            return finish(False,'paid_reservation_rejected')
        if (status.get('admission_open') is not True or budget.get('halt_reason') is not None
                or float(budget['remaining'])<=0 or budget['paid_dispatches']>=limits['paid_requests']
                or budget['local_dispatches']>=limits['local_requests']):return finish(False,'broker_admission_closed')
        finish(True,'all_gates_passed')
        self.reserved[cell['cell_id']]=required
        self.scratch_reservation_floor=max(getattr(self,'scratch_reservation_floor',0),
            projected-self.context['prior_retained_apparent_bytes'])
        return True

    def release(self, cell): self.reserved.pop(cell['cell_id'],None)

    async def run_one(self, cell):
        from openwebrl.webprm_ephemeral_state import claim_state
        folder=self.folders[cell['cell_id']];plan=self.plans[cell['cell_id']]
        state=self.scratch/'states'/cell_key(cell)
        claim=folder/'state-claim.json'
        claim_state(claim,state,scratch_root=self.scratch,campaign_root=Path(self.proposal['campaign_root']),
            cell_id=cell['cell_id'],base_path=self.scratch/'baseline',policy_sha256=digest(self.proposal['ephemeral_policy']))
        site=worker=None
        try:
            copy=self.command(self.service_command('copy',state=state,folder=folder/'copy'),folder/'copy.log')
            await wait_process(copy,self.context['deadline'],self.check)
            site=self.command(self.service_command('serve',state=state,folder=folder/'site'),folder/'site-controller.log')
            until=min(self.context['deadline'],time.time()+SITE_STARTUP_SECONDS[self.site_domain])
            while not (folder/'site/instance.json').exists():
                if site.poll() is not None or time.time()>=until:raise RuntimeError('Original '+SITE_LABELS[self.site_domain]+' startup failed')
                self.check();await asyncio.sleep(.5)
            instance=read(folder/'site/instance.json');prefix=['nsenter','--target',str(instance['service_pid']),
                '--user','--net','--preserve-credentials']
            await wait_site_readiness(self.command, prefix, self.site_domain, folder, until, self.check)
            lease=folder/'lease.json'
            put(lease,dict(task_id=f"webarena_lite_{cell['task_id']:03d}",actor_model=cell['actor'],selector=cell['selector'],
                instance_receipt=str(folder/'site/instance.json'),instance_receipt_sha256=sha(folder/'site/instance.json')))
            argv=prefix+[BROWSER,'-B',str(REPO/'scripts/run_webprm_webarena_pilot.py'),'--plan',str(folder/'worker-plan.json'),
                '--output',str(folder/'episodes'),'--execute','--task-id',str(cell['task_id']),
                '--actor',cell['actor'],'--selector',cell['selector'],'--lease',str(lease)]
            put(folder/'worker-launch.json',dict(argv=argv,cell_id=cell['cell_id'],resampling_forbidden=True))
            worker=self.command(argv,folder/'worker.log')
            await wait_process(worker,self.context['deadline'],self.check)
            result_path=folder/'episodes'/f"webarena_lite_{cell['task_id']:03d}"/(cell['actor']+'__'+cell['selector'])/'result.json'
            result=cell_result(result_path,cell,plan)
            put(folder/'site/stop.json',dict(reason='fixed cell closed'))
            await wait_process(site,min(self.context['deadline'],time.time()+SITE_STOP_SECONDS[self.site_domain]),self.check)
            closure=read(folder/'site/closure.json')
            if closure.get('remaining_owned_processes')!={}:raise RuntimeError('Website closure incomplete')
            status=self.broker.status()
            if broker_cell_id(cell) in status['active_cell_ids']:
                raise RuntimeError('Cell still owns an in-flight model request after worker closure')
            api=result_path.parent/'api.jsonl'
            if not api.exists():
                records=[json.loads(line) for line in (self.output/'broker.jsonl').read_text().splitlines() if line]
                if any(row.get('cell_id')==broker_cell_id(cell) for row in records):
                    raise RuntimeError('Dispatched cell has no API journal; do not fabricate missing evidence')
                fd=os.open(api,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                try:os.fsync(fd)
                finally:os.close(fd)
            put(folder/'broker-status.json',status)
            closed=dict(cell_id=cell['cell_id'],
                worker_remaining_processes=read(result_path.parent/'watchdog.json')['remaining_owned_processes'],
                service_remaining_processes=closure['remaining_owned_processes'],broker_cell_inflight=0,
                worker_watchdog_sha256=sha(result_path.parent/'watchdog.json'),
                site_closure_sha256=sha(folder/'site/closure.json'),
                broker_status_reference=artifact(folder/'broker-status.json'))
            put(folder/'closure.json',closed)
            evidence=audit_input(self.proposal,self.output,folder,cell,plan,status)
            put(folder/'audit-input.json',evidence)
            result.update(state_preserved=True,audit_status='independent_audit_pending')
            auditor=REPO/AUDITORS[self.site_domain]
            process=self.command([BROWSER,'-B',str(auditor),'--input',str(folder/'audit-input.json'),
                '--output',str(folder/'audit.json')],folder/'audit.log')
            await wait_process(process,self.context['deadline'],self.check,allowed_returncodes=(0,2))
            audit=read(folder/'audit.json')
            result['audit_verified']=audit.get('audit_verified') is True
            result['audit_status']='verified' if audit.get('audit_verified') is True else 'rejected'
            if audit.get('audit_verified') is True and audit.get('complete_for_disposal') is True:
                from openwebrl.webprm_ephemeral_state import durable_artifacts,dispose_state
                files={k:v['path'] for k,v in evidence['artifacts'].items()}
                files.update(worker_plan=str(folder/'worker-plan.json'),audit=str(folder/'audit.json'),
                    closure=str(folder/'closure.json'),audit_input=str(folder/'audit-input.json'))
                # Include service/copy/worker diagnostics and all image inputs,
                # while the private login storage state stays local to the cell.
                for index,path in enumerate(sorted(folder.rglob('*'))):
                    if path.is_file() and path.name!='storage-state-private.json':files['cell_file_'+str(index)]=str(path)
                manifest=durable_artifacts(Path(self.proposal['campaign_root']),files)
                audit_binding=dict(verified=True,cell_id=cell['cell_id'],artifact_manifest_sha256=digest(manifest),
                    audit_artifact_sha256=manifest['audit']['sha256'])
                dispose_state(claim,approved_policy_sha256=digest(self.proposal['ephemeral_policy']),
                    artifact_manifest=manifest,audit=audit_binding,closure=closed,
                    intent_path=folder/'disposal-intent.json',receipt_path=folder/'disposal-receipt.json')
                result['state_preserved']=False
                if self.site_domain=='shopping_admin' and not getattr(self,'cms_startup_verified',False):
                    authentication=result_path.parent/'authentication.json'
                    if read(authentication).get('authenticated') is not True:
                        raise RuntimeError('Audited CMS canary lacks native authentication evidence')
                    put(self.output/'cms-startup-canary.json',dict(cell_id=cell['cell_id'],
                        native_authentication=artifact(authentication),independent_audit=artifact(folder/'audit.json'),
                        owned_closure=artifact(folder/'closure.json'),extra_model_calls=0,
                        scope='First fixed primary episode; permit parallel admission only after verified native audit'))
                    self.cms_startup_verified=True
            put(folder/'cell-outcome.json',result)
            if native_authentication_failure(read(result_path)):
                # Stop only future admissions. Existing workers finish and
                # retain their own closure/verdicts before the owner exits.
                self.admission_halt_reason='native_authentication_failure'
            # Independent normalized audit records, not these mechanical outcomes,
            # are the only input admitted to the comparison table.
            return result
        finally:
            _stop(worker);_stop(site)
            # An exceptional path with missing closure cannot release a lane
            # for more fresh states. The outer owner still reaps the full tree.
            if worker is not None:
                episode=folder/'episodes'/f"webarena_lite_{cell['task_id']:03d}"/(cell['actor']+'__'+cell['selector'])
                try:
                    if read(episode/'watchdog.json').get('remaining_owned_processes')!={}:
                        self.halt_reason='Worker closure is incomplete; fresh admissions halted'
                except (OSError,ValueError):self.halt_reason='Worker closure is absent; fresh admissions halted'
            if site is not None:
                try:
                    if read(folder/'site/closure.json').get('remaining_owned_processes')!={}:
                        self.halt_reason='Site closure is incomplete; fresh admissions halted'
                except (OSError,ValueError):self.halt_reason='Site closure is absent; fresh admissions halted'

    async def run(self, cells, store):
        try:
            await self.start([cell for cell in cells if store.state(cell)=='pending'])
            return await run_queue(cells,store,domain_ready=self.domain_ready,can_start=self.can_start,
                run_one=self.run_one,release=self.release,max_workers=self.proposal['limits']['max_workers'],
                stop_admission=lambda:time.time()+self.proposal['limits']['min_start_seconds']>=self.context['deadline'])
        finally:
            for process in reversed(self.children):_stop(process)
            if self.broker is not None:put(self.output/'broker-closure.json',self.broker.close())
            if self.port_lease is not None:self.port_lease.close()


def _runner(proposal,context,output,scratch,cells):
    os.setsid()
    with ClaimStore(Path(proposal['campaign_root'])/'queue',cells) as store:
        runner=CampaignRunner(proposal,context,output,scratch)
        pending=[c for c in cells if store.state(c)=='pending']
        if not any(runner.domain_ready(c['domain']) for c in pending):
            put(Path(output)/'queue-summary.json',dict(status='no_ready_pending_domains',new_cells=0));return
        summary=asyncio.run(runner.run(cells,store));put(Path(output)/'queue-summary.json',summary)


def execute(proposal,approval):
    # Authorization is checked before scheduler queries, directories or processes.
    if approval.get('status')!='approved' or approval.get('proposal_sha256')!=digest(proposal):
        raise ValueError('Exact prospective campaign approval required')
    cohort,template=validate_proposal(proposal)
    job=str(approval['job_id']);attempts=approval['attempt_job_ids']
    validate_registered_allocation(proposal,approval)
    if not attempts or attempts[-1]!=job or len(attempts)!=len(set(attempts)):
        raise ValueError('Complete immutable allocation-attempt list required')
    if f'/job_{job}/' not in Path('/proc/self/cgroup').read_text():
        raise ValueError('Controller cgroup is not the approved allocation')
    root=checked(proposal['campaign_root'])
    if RUNTIME not in root.parents:
        raise ValueError('Original Reddit execution receipts must remain under its approved runtime root')
    root.mkdir(mode=0o700,exist_ok=True)
    if root.stat().st_mode & 0o077:raise ValueError('Campaign root must be private0700')
    prior=0;finals=[];rule=scratch_accounting(proposal,approval)
    for old in attempts[:-1]:
        storage=closed_attempt_storage(proposal,old)
        if storage is not None:finals.append(storage)
    retained=prior_scratch_charge(rule,finals)
    if attempts[:-1]:
        history=subprocess.check_output(['sacct','-j',','.join(attempts[:-1]),'-X','--noheader','--parsable2',
            '--format=JobID,State,ElapsedRaw'],text=True,timeout=15)
        rows=[line.split('|') for line in history.strip().splitlines()]
        terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE'}
        if ({r[0] for r in rows}!=set(attempts[:-1]) or len(rows)!=len(attempts)-1
                or any(r[1].split()[0] not in terminal or not r[2].isdigit() for r in rows)):
            raise ValueError('Prior allocation attempts are not terminal/accounted')
        prior=sum(int(r[2]) for r in rows)
    scheduler=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=15)
    epochs=subprocess.check_output(['scontrol','--json','show','job',job],text=True,timeout=15)
    context=approved_context(proposal,approval,scheduler,epochs,prior)
    output=root/('attempt-'+job);output.mkdir(mode=0o700)
    scratch=SCRATCH/('t5-'+job+'-'+secrets.token_hex(3));scratch.mkdir(mode=0o700,parents=True)
    (scratch/'states').mkdir(mode=0o700)
    context.update(proposal=proposal,approval=approval,proposal_sha256=digest(proposal),scratch=str(scratch),output=str(output),
        controller_sha256=sha(__file__),owner_pid=os.getpid(),owner_start_ticks=_process(os.getpid())['start_ticks'],
        prior_retained_apparent_bytes=retained,scratch_accounting=rule,
        cumulative_prior_retained_apparent_bytes=sum(f['apparent_bytes'] for f in finals))
    put(output/'execution-context.json',context)
    previous=_subreaper(True);baseline=set(_processes());tree=None;stopped=[False]
    handlers={sig:signal.signal(sig,lambda *_:stopped.__setitem__(0,True)) for sig in (signal.SIGTERM,signal.SIGINT)}
    process=multiprocessing.get_context('fork').Process(target=_runner,args=(proposal,context,str(output),str(scratch),cohort['cells']))
    try:
        process.start();tree=OwnedEpisodeTree(process,baseline)
        while process.is_alive() and time.time()<context['deadline'] and not stopped[0]:tree.refresh();time.sleep(.5)
    finally:
        try:
            closure=tree.stop(10) if tree else dict(remaining_owned_processes={},child_exitcode=None)
            put(output/'controller-closure.json',closure)
            domain=next((d for d in MAGENTO_DOMAINS if d in proposal['domains']),'reddit')
            put(output/'storage-final.json',storage_after_closure(closure,scratch,root,retained,proposal['limits'],
                DU_TIMEOUT_SECONDS.get(domain,15)))
        finally:
            _subreaper(previous)
            for sig,handler in handlers.items():signal.signal(sig,handler)
    if closure['remaining_owned_processes'] or closure['child_exitcode']!=0:
        raise RuntimeError('Campaign attempt failed; durable claims and budget remain charged')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal',required=True,type=Path);parser.add_argument('--approval',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--register-allocation',help='Record an approved held job before release; never allocate')
    args=parser.parse_args()
    proposal=read(args.proposal)
    if args.register_allocation:
        if args.execute or args.approval is None:parser.error('Registration requires approval and cannot execute')
        validate_proposal(proposal)
        row=register_allocation(proposal,read(args.approval),args.register_allocation)
        print(json.dumps(dict(status='registered_not_released',job_id=row['job_id'],sequence=row['sequence'])))
        return
    if not args.execute:print(json.dumps(preview(proposal),indent=2));return
    if args.approval is None:parser.error('--execute requires --approval')
    execute(proposal,read(args.approval))


if __name__=='__main__':main()
