"""Approval/resource ownership for the third-party Magento (CMS, Shopping) hosting validators.

Nothing executes at import. Exact user resource approval and a durably registered
matching live allocation are required before constructing a hosting context. Each
site has one hosting Profile (protocol, resources, limits, pinned evidence); the
module-level CMS constants and keyword defaults keep the CMS contract unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import stat
import subprocess
import time

from .webprm_cms_materializer import Closed, private_path, put
from .webprm_lifecycle import OwnedEpisodeTree, _process, _processes, _subreaper

RESOURCES = dict(account='zixianma', partition='gpu-h200', qos='normal', nodes=1,
                 gpus=0, cpus=4, memory_gib=16, total_seconds=7200, max_allocations=7)
LIMITS = dict(scratch_bytes=48*2**30, retained_bytes=2*2**30,
              runtime_reserve_bytes=8*2**30, closure_seconds=120,
              model_calls=0, api_usd=0, download_bytes=0)
PROTOCOL = 'cms_thirdparty_owned_hosting_validation_v1'
PLAN_SHA = '7446e8ca23ac19ffc032a92c25491ae75988f83375100899c6b5d151ea8144e9'
ACQUISITION_SHA = 'c7c501fb4a620ea4799fea6178df023dfb9801e3bf84fbad8fd464ed32d9b327'
LAYER_PROOF_SHA = '79a680cfb544431ba14cc01e27b8ef9803f5d0a072e3779e883e3fb931d9792e'
MANIFEST_SHA = '428cdedfc10996d7f5378299cff541c93c187d1243643e31d439a66c0156147d'
SHOPPING_PLAN_SHA = '0ef41781bf97b950801b1cbfb4369427f685c39cab4a1ff9350595ede9ae0be6'
TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE'}


@dataclass(frozen=True)
class Profile:
    name: str
    label: str
    protocol: str
    resources: dict
    limits: dict
    plan_sha: str
    # Proposal evidence fields that must carry exactly these hashes.
    pins: dict
    # CMS additionally requires its read-only real-image metadata preflight receipt.
    preflight: bool
    # The Shopping image has millions of files (2.25 GB manifest, probe 358020): the
    # apparent-size walk needs a larger entry cap and a slower sampling cadence.
    apparent_entries: int = 1000000
    sample_seconds: int = 10
    cumulative_scratch: bool = False


CMS = Profile('cms', 'CMS', PROTOCOL, RESOURCES, LIMITS, PLAN_SHA,
    dict(acquisition_audit=ACQUISITION_SHA, layer_proof=LAYER_PROOF_SHA, manifest=MANIFEST_SHA), True)
SHOPPING = Profile('shopping', 'Shopping', 'shopping_thirdparty_owned_hosting_validation_v1',
    dict(account='zixianma', partition='gpu-h200', qos='normal', nodes=1, gpus=0, cpus=8,
         memory_gib=128, total_seconds=14400, max_allocations=4),
    # 384 GiB: base + two states (~170 GB) + reserve beside one preserved failed attempt's scratch
    # (job 358132 left 112.7 GB on its node); 12 GiB retained: a 2.25 GB manifest per attempt.
    dict(scratch_bytes=384*2**30, retained_bytes=12*2**30, runtime_reserve_bytes=16*2**30,
         closure_seconds=300, model_calls=0, api_usd=0, download_bytes=0),
    SHOPPING_PLAN_SHA,
    dict(acquisition_audit='9588c1bacf7b98b5a8ba5d8ee2bb8675e06bd48c6e9369c52bf2984a3ac5ec9c',
         layer_proof='b3c5dd229c65c9bfcb1e1455751f73764b42660766db8c8e0ebe7e541f6d0ed8',
         manifest='8f9618d9194a46bc9f7bc96dc14deba0b968abce6c393d59cc7cd1dfec86ed6f',
         bringup_probe='2bd1880b76b9e388fed7e18c08586c432aed6bfee398cbb8a7c5c36e372cfe91'),
    False, apparent_entries=64000000, sample_seconds=300)
GITLAB = Profile('gitlab', 'GitLab', 'gitlab_thirdparty_owned_hosting_validation_v1',
    dict(account='zixianma', partition='gpu-h200', qos='normal', nodes=1, gpus=0, cpus=8,
         memory_gib=96, total_seconds=7200, max_allocations=2),
    dict(scratch_bytes=160*2**30, retained_bytes=4*2**30, runtime_reserve_bytes=16*2**30,
         closure_seconds=300, model_calls=0, api_usd=0, download_bytes=0),
    'ba0aa96a2be9e8537abf47992f9ff8eb7cdd3b2d74c854cb91becdbe646e334d',
    dict(acquisition_audit='daf80268fd9659c6293083205429f2e2d284adcd6b582850c546af50519a73e0',
         layer_proof='233266db780525b13040d0e69c5edf670914d5d450dafc4429f94989a43ad263',
         manifest='5e367035617c9419c2123a8319b77a26cdeb1525fbd82913ba8fcfc12d7e2d57',
         bringup_probe='d37630ce23c04b3770e2cf98e172fff69b1ffa2555e6a6070e40805e59fdcc17'),
    False, apparent_entries=30000000, sample_seconds=60, cumulative_scratch=True)
PROFILES = {CMS.protocol: CMS, SHOPPING.protocol: SHOPPING, GITLAB.protocol: GITLAB}


def profile_of(proposal):
    profile = PROFILES.get(proposal.get('protocol')) if isinstance(proposal, dict) else None
    if profile is None:
        raise Closed('Exact disclosed hosting/resource approval required for a known protocol')
    return profile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, maximum=32*2**20):
    path = private_path(path)
    if not path.is_file() or path.stat().st_size > maximum:
        raise Closed('Bounded regular receipt required')
    return json.loads(path.read_text())


def reference(value, *, maximum=32*2**20):
    if not isinstance(value,dict) or not {'path','sha256'} <= value.keys():
        raise Closed('Hash-bound reference required')
    path = private_path(value['path'])
    if not path.is_file() or path.stat().st_size > maximum or sha(path) != value['sha256']:
        raise Closed('Referenced evidence changed')
    return path


def validate(proposal, approval):
    p = profile_of(proposal)
    if (proposal.get('resources') != p.resources or proposal.get('limits') != p.limits or
            proposal.get('original_state_equivalence_verified') is not False or
            proposal.get('source') != 'third_party_digest_pinned_oci_candidate' or
            approval.get('status') != 'approved' or approval.get('proposal_sha256') != digest(proposal) or
            approval.get('resources') != p.resources or approval.get('limits') != p.limits or
            approval.get('third_party_source_accepted') is not True or
            not isinstance(approval.get('exact_user_approval'),str) or not approval['exact_user_approval'].strip()):
        raise Closed('Exact disclosed '+p.label+' hosting/resource approval required')
    root = private_path(proposal['campaign_root'])
    if root.exists() and (not root.is_dir() or root.stat().st_mode & 0o077):
        raise Closed('Campaign root must remain private')
    if proposal['service_plan']['sha256'] != p.plan_sha:
        raise Closed('Reviewed service plan differs')
    plan = read(reference(proposal['service_plan']))
    for ref in plan['planned_configs'].values():
        reference(ref)
    source_root = private_path(proposal['frozen_source'])
    if not proposal.get('source_sha256'):
        raise Closed('Frozen execution source required')
    for name, expected in proposal['source_sha256'].items():
        path = private_path(source_root / name)
        if source_root not in path.parents or sha(path) != expected:
            raise Closed('Frozen '+p.label+' execution source changed')
    if reference(proposal['batch_script']).parent != source_root:
        raise Closed('Frozen batch path differs')
    for name in ('account_binding','native_source_receipt')+(('materialization_preflight',) if p.preflight else ()):
        reference(proposal[name])
    for name,expected in p.pins.items():
        if proposal[name]['sha256']!=expected:
            raise Closed('Pinned acquired third-party source differs')
        reference(proposal[name])
    if p.preflight:
        preflight=read(proposal['materialization_preflight']['path'])
        if preflight.get('status')!='metadata_layout_supported' or preflight.get('rootfs_materialized') is not False:
            raise Closed('Successful read-only real-image metadata preflight required')
    return plan


def scheduler(raw, job, *, requested_seconds, running=True, now=None, profile=CMS):
    """One exact Slurm JSON record, including cumulative-clock-safe deadline."""
    now = time.time() if now is None else now
    resources, limits = profile.resources, profile.limits
    cpus_, gib = resources['cpus'], resources['memory_gib']
    try:
        data = json.loads(raw)
        if data.get('errors') != [] or len(data['jobs']) != 1:
            raise ValueError()
        row = data['jobs'][0]
        def finite(key):
            value = row[key]
            if value['set'] is not True or value['infinite'] is not False or type(value['number']) is not int:
                raise ValueError(key)
            return value['number']
        cpus,nodes,memory,seconds = finite('cpus'),finite('node_count'),finite('memory_per_node'),finite('time_limit')*60
        if (str(row['job_id']) != str(job) or row['user_id'] != os.getuid() or row['restart_cnt'] != 0 or
                any(row[k] != resources[k] for k in ('account','partition','qos')) or
                (cpus,nodes,memory) != (cpus_,1,gib*1024) or seconds != requested_seconds or
                not 60 <= seconds <= resources['total_seconds'] or seconds % 60):
            raise ValueError()
        tres = dict(x.split('=',1) for x in row.get('tres_alloc_str' if running else 'tres_req_str','').split(',') if '=' in x)
        if (tres.get('cpu')!=str(cpus_) or tres.get('node')!='1' or tres.get('mem') not in (f'{gib}G',f'{gib*1024}M') or
                any(k.startswith('gres/gpu') and v!='0' for k,v in tres.items())):
            raise ValueError()
        if not running:
            if row['job_state'] != ['PENDING'] or row.get('state_reason') != 'JobHeldUser':
                raise ValueError()
            return dict(job_id=str(job), requested_seconds=seconds, state='held')
        start,end = finite('start_time'),finite('end_time')
        deadline = min(end,start+seconds)
        if row['job_state']!=['RUNNING'] or not math.isfinite(now) or not 0 < start <= now < deadline-limits['closure_seconds']:
            raise ValueError()
    except (ValueError,TypeError,KeyError,AttributeError) as error:
        raise Closed(f'Exact {cpus_}CPU/{gib}GiB/0GPU scheduler proof/clock required') from error
    return dict(job_id=str(job),started_epoch=start,deadline=deadline,
                work_deadline=deadline-limits['closure_seconds'],requested_seconds=seconds)


def history(root, profile=CMS):
    root = private_path(root)
    rows = [read(p) for p in sorted((root/'allocations').glob('*.json'))] if (root/'allocations').exists() else []
    if len(rows)>profile.resources['max_allocations'] or any(r['sequence']!=i+1 or not str(r['job_id']).isdigit() for i,r in enumerate(rows)):
        raise Closed('Complete bounded allocation registry required')
    if len({r['job_id'] for r in rows})!=len(rows):
        raise Closed('Duplicate registered allocation')
    attempts = {p.name.removeprefix('attempt-') for p in root.glob('attempt-*')}
    if not attempts <= {r['job_id'] for r in rows}:
        raise Closed('Unregistered historical attempt')
    return rows


def prior_budget(root, rows, terminal_rows, profile=CMS):
    if set(terminal_rows) != {r['job_id'] for r in rows}:
        raise Closed('Every historical allocation must be accounted')
    seconds = scratch = 0
    for row in rows:
        terminal = terminal_rows[row['job_id']]
        if (terminal.get('state') not in TERMINAL or type(terminal.get('elapsed_seconds')) is not int or
                not 0 <= terminal['elapsed_seconds'] <= row['requested_seconds']+60):
            raise Closed('Prior allocation is not terminal/accounted')
        old = Path(root)/('attempt-'+row['job_id'])
        if not old.exists() and not os.path.lexists(old):
            # run() creates the attempt folder before any scratch, so a terminal job
            # that never created it (failed at startup checks, e.g. 357113) used no
            # scratch; its elapsed time is still charged.
            seconds += terminal['elapsed_seconds']; continue
        closure, storage = read(old/'closure.json'),read(old/'storage-final.json')
        if closure.get('remaining_owned_processes')!={} or type(storage.get('scratch_bytes')) is not int or storage['scratch_bytes']<0:
            raise Closed('Prior closure/retained scratch accounting missing')
        seconds += terminal['elapsed_seconds']
        # Per-node live accounting (same rule the user approved for Reddit): retained scratch is
        # node-local /tmp, so charge it only while it still exists on this node. Without a recorded
        # scratch path the charge stays conservative. shutil.disk_usage still guards real space.
        context = read(old/'context.json') if (old/'context.json').exists() else {}
        where = context.get('scratch')
        if profile.cumulative_scratch or not isinstance(where,str) or os.path.lexists(where):
            scratch += storage['scratch_bytes']
    remaining=(profile.resources['total_seconds']-seconds)//60*60
    if remaining<=profile.limits['closure_seconds']+60:
        raise Closed('Cumulative hosting time exhausted')
    return dict(prior_seconds=seconds,remaining_seconds=remaining,prior_scratch_bytes=scratch)


def register(proposal, approval, job, held_raw, terminal_rows):
    validate(proposal,approval);p=profile_of(proposal)
    root=private_path(proposal['campaign_root']);root.mkdir(mode=0o700,exist_ok=True)
    with os.fdopen(os.open(root/'register.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600),'r+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=history(root,p)
        if len(rows)>=p.resources['max_allocations'] or any(r['job_id']==str(job) for r in rows):
            raise Closed(p.label+' allocation slots exhausted/duplicate')
        budget=prior_budget(root,rows,terminal_rows,p)
        scheduler(held_raw,job,requested_seconds=budget['remaining_seconds'],running=False,profile=p)
        folder=root/'allocations';folder.mkdir(mode=0o700,exist_ok=True)
        record=dict(sequence=len(rows)+1,job_id=str(job),proposal_sha256=digest(proposal),
                    requested_seconds=budget['remaining_seconds'],prior_job_ids=[r['job_id'] for r in rows],
                    terminal_rows=terminal_rows,budget=budget,held_scheduler=held_raw,registered_epoch=time.time())
        put(folder/f'{len(rows)+1:04d}-{job}.json',record)
        return record


def apparent(root, *, maximum_entries=1000000):
    """lstat only: image absolute links are never followed on the host."""
    root=private_path(root);total=count=0
    if not root.exists():
        return 0
    for directory,dirs,files in os.walk(root,followlinks=False):
        for entry in dirs+files:
            path=Path(directory)/entry
            try:
                value=path.lstat()
            except FileNotFoundError:
                # Runtime services can rotate/remove files while this read-only
                # sample runs. Missing files contribute zero at this sample;
                # this is not advertised as a kernel-enforced peak quota.
                continue
            count+=1;total+=value.st_size
            if count>maximum_entries:
                raise Closed('Hosting file-count cap')
            if stat.S_ISDIR(value.st_mode) and path.is_mount():
                raise Closed('Unexpected nested host mount')
    return total


def safe_env(scratch):
    scratch=private_path(scratch)
    env={k:v for k,v in os.environ.items() if not k.startswith(('APPTAINER_','SINGULARITY_'))}
    for key,leaf in {'TMPDIR':'tmp','APPTAINER_TMPDIR':'apptainer-tmp','APPTAINER_CONFIGDIR':'apptainer-config',
        'APPTAINER_CACHEDIR':'apptainer-cache','XDG_CACHE_HOME':'cache','XDG_CONFIG_HOME':'config',
        'XDG_DATA_HOME':'data','XDG_STATE_HOME':'state','XDG_RUNTIME_DIR':'run'}.items():
        p=scratch/leaf;p.mkdir(mode=0o700,exist_ok=True);env[key]=str(p)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HISTFILE='/dev/null',BASH_ENV='/dev/null',ENV='/dev/null',
               APPTAINER_NO_HOME='true',APPTAINER_CONTAINALL='true',APPTAINER_NO_MOUNT='bind-paths,home,cwd,hostfs,tmp')
    return env


def own(target, args, *, attempt, scratch, campaign_root, deadline, prior_scratch_bytes=0, profile=CMS):
    """One subreaper owns ALL builders, helpers, services and browser descendants."""
    previous=_subreaper(True);baseline=set(_processes());tree=process=None;stopped=[False];reason=None
    handlers={s:signal.signal(s,lambda *_:stopped.__setitem__(0,True)) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        process=multiprocessing.get_context('fork').Process(target=target,args=args)
        try:
            process.start()
        finally:
            if process.pid is not None:tree=OwnedEpisodeTree(process,baseline)
        next_sample=0
        while process.is_alive() and time.time()<deadline and not stopped[0]:
            tree.refresh()
            if time.time()>=next_sample:
                scratch_used=apparent(scratch,maximum_entries=profile.apparent_entries);retained=apparent(campaign_root)
                if (scratch_used+prior_scratch_bytes>profile.limits['scratch_bytes'] or
                        retained>profile.limits['retained_bytes']-16*2**20):
                    reason='storage_cap';break
                next_sample=time.time()+profile.sample_seconds
            time.sleep(.25)
        if process.is_alive() and reason is None:reason='deadline_or_signal'
    finally:
        try:
            closure=tree.stop(10) if tree else dict(remaining_owned_processes={},child_exitcode=None,owned_pids=[])
            closure['stop_reason']=reason;put(Path(attempt)/'closure.json',closure)
            put(Path(attempt)/'storage-final.json',dict(scratch_bytes=apparent(scratch,maximum_entries=profile.apparent_entries),retained_bytes=apparent(campaign_root),
                prior_scratch_bytes=prior_scratch_bytes,measurement=f'Final apparent lstat sum; runtime guard samples every {profile.sample_seconds} seconds, not a kernel quota.'))
        finally:
            _subreaper(previous)
            for s,h in handlers.items():signal.signal(s,h)
    if closure['remaining_owned_processes'] or closure['child_exitcode']!=0 or reason:
        raise Closed(profile.label+' validation did not complete with verified process closure')
    return closure
