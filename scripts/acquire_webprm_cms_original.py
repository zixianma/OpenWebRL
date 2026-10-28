#!/usr/bin/env python3
"""Prepare or own one approved original-CMS acquisition/identity allocation.

No services, extraction, image build, browser, model or provider calls. There is
one allocation and one transfer attempt: redirects are bounded, but automatic
retry/resume is deliberately unsupported. A failed attempt remains preserved.
"""
from __future__ import annotations

import argparse
import email.message
import hashlib
import http.client
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import resource
import secrets
import shutil
import signal
import ssl
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.inspect_webprm_cms_archive import ARCHIVE_BYTES, ARCHIVE_NAME, GateClosed, checked_path, inspection
from openwebrl.webprm_lifecycle import OwnedEpisodeTree, _processes, _subreaper

SCRIPT = Path(__file__).resolve()
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011')
DRIVE_ID = '1See0ZhJRw0WTTL9y8hFlgaduwPZ_nGfd'
SOURCE_URL = 'https://drive.google.com/uc?export=download&id=' + DRIVE_ID + '&confirm=t'
CHUNK = 64 * 2**20
BLOCK = 2**20
METADATA_RESPONSE_CAP = 65536
HEADER_CAP = 65536
RECEIPT_BYTES = 4 * 2**20
RESOURCES = dict(account='zixianma', partition='gpu-h200', qos='normal', nodes=1,
    cpus=2, memory_gib=8, gpus=0, total_allocation_seconds_including_retries=3600,
    maximum_allocation_attempts=1, maximum_transfer_attempts=1, maximum_cpu_hours=2,
    api_budget_usd=0, model_provider_calls=0)
REPLACEMENT_PROFILE = 'cms_clock_corrected_59m_v2'
REPLACEMENT_RESOURCES = dict(RESOURCES, total_allocation_seconds_including_retries=3540)
PRIOR_ROOT = RUNTIME/'next-stage-preparation/cms-original-acquisition-v1'
PRIOR_JOB = '355534'
LIMITS = dict(download_charged_bytes=10000000000, metadata_received_bytes=16*2**20,
    retained_bytes=12*2**30, temporary_bytes=512*2**20, initial_persistent_free_bytes=12*2**30,
    initial_temporary_free_bytes=2**30, closure_reserve_seconds=90, max_redirects_per_range=3,
    max_http_responses=620, receipt_reserve_bytes=16*2**20, network_connections=1,
    response_header_bytes_per_request=HEADER_CAP)
SOURCES = ('scripts/acquire_webprm_cms_original.py', 'scripts/inspect_webprm_cms_archive.py',
           'openwebrl/webprm_lifecycle.py', 'openwebrl/jev_eval.py',
           'tests/test_acquire_webprm_cms_original.py', 'tests/test_inspect_webprm_cms_archive.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(checked_path(path).read_bytes())


def put(path, value, *, replace=False):
    path = checked_path(path)
    raw = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    if len(raw) > RECEIPT_BYTES:
        raise GateClosed('Bounded receipt exceeded four MiB')
    target = checked_path(path.with_suffix(path.suffix + '.tmp')) if replace else path
    # A leftover temporary receipt belongs to a failed attempt. Never truncate
    # it, a FIFO, or a hard link that might alias an unrelated protected file.
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_EXCL
    fd = os.open(target, flags, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    if replace:
        os.replace(target, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def safe_environment(base):
    base = checked_path(base); base.mkdir(mode=0o700, exist_ok=True)
    env = {k:v for k,v in os.environ.items() if not k.startswith(('SALLOC_', 'SBATCH_', 'SRUN_', 'APPTAINER_', 'SINGULARITY_'))}
    for key in ('TMPDIR', 'XDG_CONFIG_HOME', 'XDG_CACHE_HOME', 'XDG_DATA_HOME', 'XDG_RUNTIME_DIR', 'PIP_CACHE_DIR'):
        path = checked_path(base/key.lower()); path.mkdir(mode=0o700, exist_ok=True); env[key] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='',
               HISTFILE='/dev/null', BASH_ENV='/dev/null', ENV='/dev/null', OMP_NUM_THREADS='1')
    return env


def predecessor_evidence(snapshot=None):
    """Bind only the closed, zero-transfer four-second predecessor; no retry."""
    proposal, auth = read(PRIOR_ROOT/'proposal.json'), read(PRIOR_ROOT/'approval.json')
    claim, launch_claim = read(PRIOR_ROOT/'run-claim.json'), read(PRIOR_ROOT/'launch.json')
    if (proposal.get('resources') != RESOURCES or proposal.get('limits') != LIMITS or
            auth.get('status') != 'approved' or auth.get('proposal_sha256') != digest(proposal) or
            claim.get('job_id') != PRIOR_JOB or claim.get('nonce') != launch_claim.get('nonce') or
            launch_claim.get('proposal_sha256') != digest(proposal) or
            any(PRIOR_ROOT.glob('attempt-*')) or any(PRIOR_ROOT.rglob('transfer-ledger.json'))):
        raise GateClosed('Expected preserved zero-transfer predecessor is unavailable or changed')
    for name, expected in proposal['source_sha256'].items():
        frozen = PRIOR_ROOT/'frozen-source'/Path(name).relative_to(REPO)
        if sha(frozen) != expected:
            raise GateClosed('Original approved source evidence changed')
    if snapshot is None:
        snapshot = json.loads(subprocess.check_output(['scontrol','--json','show','job',PRIOR_JOB],text=True,timeout=15))
    try:
        if snapshot.get('errors') != [] or len(snapshot['jobs']) != 1:
            raise ValueError('Expected one completed predecessor')
        row = snapshot['jobs'][0]
        def epoch(name):
            value = row[name]
            if value.get('set') is not True or value.get('infinite') is not False or type(value.get('number')) is not int:
                raise ValueError('Finite predecessor timestamp required')
            return value['number']
        start, end = epoch('start_time'), epoch('end_time')
        if (type(row.get('job_id')) is not int or str(row['job_id']) != PRIOR_JOB or row.get('job_state') != ['FAILED'] or
                row.get('user_id') != os.getuid() or row.get('restart_cnt') != 0 or end-start != 4):
            raise ValueError('Predecessor must be the observed four-second failed allocation')
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise GateClosed('Invalid closed predecessor scheduler proof') from error
    return dict(job_id=PRIOR_JOB, root=str(PRIOR_ROOT), elapsed_seconds=4,
        transfer_body_bytes=0, transfer_attempts=0, prior_proposal_sha256=digest(proposal),
        files_sha256={name:sha(PRIOR_ROOT/name) for name in ('proposal.json','approval.json','launch.json','run-claim.json')},
        scheduler_snapshot=snapshot, absence_of_transfer_basis='Clock gate precedes attempt/scratch creation and worker/network; no attempt directory or transfer ledger exists.')


def prepare(root, *, replacement=False, predecessor_snapshot=None):
    if predecessor_snapshot is not None and not replacement:
        raise GateClosed('Predecessor scheduler proof applies only to the explicit replacement profile')
    root = checked_path(root); root.mkdir(mode=0o700, parents=True, exist_ok=False)
    recipe = RUNTIME/'recipe-sources/webarena-deployment-README.md'
    if sha(recipe) != '50aac7e58d83f3e61b1e90ca8f5ffa40e1e725c45109601aa8a7fd4ce0f4a0ff':
        raise GateClosed('Pinned official deployment recipe changed')
    proposal = dict(schema_version=1, status='prepared_not_user_approved', approval_received=False,
        scope='Original CMS acquisition and offline internal identity only; no hosting or benchmark execution',
        resources=REPLACEMENT_RESOURCES if replacement else RESOURCES, limits=LIMITS, campaign_root=str(root),
        original=dict(deployment_revision='dce04686a56253aefba7b18a4fa0937cf1dc987b',
            recipe_path=str(recipe), recipe_sha256=sha(recipe), drive_file_id=DRIVE_ID,
            official_reference='https://drive.google.com/file/d/'+DRIVE_ID+'/view',
            first_request_url=SOURCE_URL, archive_name=ARCHIVE_NAME, archive_bytes=ARCHIVE_BYTES,
            archive_sha256=None, content_identity='Pinned only after complete official-source acquisition and offline verification.'),
        source_sha256={str(REPO/name):sha(REPO/name) for name in SOURCES},
        behavior=dict(retries=0, resume=False, preserves_partial_attempts=True,
            no_fallback_mirrors=True, require_strong_etag=True, require_exact_content_disposition=True,
            require_exact_206_range=True, automatic_redirects=False, extraction=False,
            unknown_transfers_keep_full_reservation=True, bytes_stream_directly_to_retained_runtime=True,
            transfer_accounting='Application response bodies plus a full bounded-header reservation per request; TLS/TCP framing excluded.'),
        limitations=['One bounded attempt is not a completion guarantee; Drive quota/HTML/no strong entity/range metadata stops cleanly.',
            'Source location is official; new observed bytes are not an independently published historical checksum.',
            'The inspector verifies hashes/layout only and never declares the image safe to import or ready to host.',
            'No retry allocation or transfer is authorized by this proposal; a failed launch/attempt cannot reset its claim.',
            'All body reservations include errors/redirects/partials; completed finite responses reconcile to measured bytes, unknown responses retain their full bound.'])
    if replacement:
        proposal.update(resource_profile=REPLACEMENT_PROFILE,
            predecessor=predecessor_evidence(read(predecessor_snapshot) if predecessor_snapshot is not None else None),
            cumulative_allocation=dict(original_cap_seconds=3600, prior_seconds=4,
                additional_seconds=3540, maximum_total_seconds=3544, unused_seconds=56,
                prior_maximum_one_allocation_exception_required=True),
            predecessor_storage_counted=True)
    put(root/'proposal.json', proposal)
    approval = dict(status='pending', proposal_sha256=digest(proposal), exact_user_approval=None)
    if replacement:
        approval['prior_maximum_one_allocation_exception_approved'] = False
    put(root/'approval.json', approval)
    return proposal


def approved(root):
    root = checked_path(root)
    proposal, auth = read(root/'proposal.json'), read(root/'approval.json')
    profile = proposal.get('resource_profile')
    resources = REPLACEMENT_RESOURCES if profile == REPLACEMENT_PROFILE else RESOURCES
    if (profile not in (None, REPLACEMENT_PROFILE) or proposal.get('resources') != resources or proposal.get('limits') != LIMITS or
            proposal.get('campaign_root') != str(root) or
            auth.get('status') != 'approved' or auth.get('proposal_sha256') != digest(proposal) or
            not isinstance(auth.get('exact_user_approval'), str) or not auth['exact_user_approval'].strip()):
        raise GateClosed('Exact immutable acquisition resource proposal is not approved')
    if profile == REPLACEMENT_PROFILE:
        expected = dict(original_cap_seconds=3600, prior_seconds=4, additional_seconds=3540,
            maximum_total_seconds=3544, unused_seconds=56, prior_maximum_one_allocation_exception_required=True)
        if (auth.get('prior_maximum_one_allocation_exception_approved') is not True or
                proposal.get('cumulative_allocation') != expected or proposal.get('predecessor_storage_counted') is not True or
                proposal.get('predecessor') != predecessor_evidence(proposal['predecessor']['scheduler_snapshot'])):
            raise GateClosed('Explicit second-allocation exception and preserved cumulative predecessor proof required')
    if proposal['original'] != dict(deployment_revision='dce04686a56253aefba7b18a4fa0937cf1dc987b',
            recipe_path=str(RUNTIME/'recipe-sources/webarena-deployment-README.md'),
            recipe_sha256='50aac7e58d83f3e61b1e90ca8f5ffa40e1e725c45109601aa8a7fd4ce0f4a0ff',
            drive_file_id=DRIVE_ID, official_reference='https://drive.google.com/file/d/'+DRIVE_ID+'/view',
            first_request_url=SOURCE_URL, archive_name=ARCHIVE_NAME, archive_bytes=ARCHIVE_BYTES,
            archive_sha256=None, content_identity='Pinned only after complete official-source acquisition and offline verification.'):
        raise GateClosed('Original official source identity changed')
    expected = {str(REPO/name):sha(REPO/name) for name in SOURCES}
    if proposal['source_sha256'] != expected or sha(proposal['original']['recipe_path']) != proposal['original']['recipe_sha256']:
        raise GateClosed('Frozen acquisition/inspector/ownership source changed')
    return proposal


def launch(root):
    proposal = approved(root)  # before any scheduler/subprocess or launch claim
    if (root/'launch.json').exists():
        raise GateClosed('Single allocation already claimed; retries cannot reset time or transfer accounting')
    nonce = secrets.token_hex(16)
    allocation_time = '00:59:00' if proposal.get('resource_profile') == REPLACEMENT_PROFILE else '01:00:00'
    argv = ['salloc', '--account=zixianma', '--partition=gpu-h200', '--qos=normal',
        '--nodes=1', '--ntasks=1', '--cpus-per-task=2', '--mem=8G', '--time='+allocation_time,
        '--job-name=wa-cms-identity', '--chdir='+str(root),
        'srun', '--ntasks=1', '--cpus-per-task=2', sys.executable, '-I', '-B', str(SCRIPT),
        '--run', '--root', str(root), '--nonce', nonce]
    put(root/'launch.json', dict(status='submission_claimed_not_yet_allocated', nonce=nonce,
        at_epoch=time.time(), argv=argv, proposal_sha256=digest(proposal)))
    env = safe_environment(root/'launch-cache')
    env = {k:v for k,v in env.items() if not k.startswith('SLURM_')}
    os.chdir(root)
    os.execvpe(argv[0], argv, env)  # salloc awaits srun; controller awaits every child.


def allocation_context(text, job, nonce, claim, now=None, *, resources=None):
    """Validate one scheduler JSON snapshot without local-time interpretation.

    Slurm's finite-number wrappers are deliberately checked, rather than
    coercing unset/infinite timestamps or falling back to timezone-free text.
    """
    now = time.time() if now is None else now
    resources = RESOURCES if resources is None else resources
    # Recovery may request fewer whole minutes after closed startup failures.
    # Every other resource field remains the original exact CPU-only profile.
    seconds = resources.get('total_allocation_seconds_including_retries') if isinstance(resources, dict) else None
    if (type(seconds) is not int or not 60 <= seconds <= 3600 or seconds % 60 or
            resources != dict(RESOURCES, total_allocation_seconds_including_retries=seconds)):
        raise GateClosed('Unrecognized bounded acquisition resource profile')
    try:
        data = json.loads(text)
        rows = data['jobs']
        if data.get('errors') != [] or not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError('Expected one error-free scheduler job')
        fields = rows[0]
        parser = data['meta']['plugin']['data_parser']
        if not isinstance(parser, str) or not parser.startswith('data_parser/'):
            raise ValueError('Scheduler JSON parser identity missing')
        def finite(name):
            value = fields[name]
            if (not isinstance(value, dict) or value.get('set') is not True or
                    value.get('infinite') is not False or type(value.get('number')) is not int or value['number'] < 0):
                raise ValueError('Expected a finite set integer: ' + name)
            return value['number']
        start, end = finite('start_time'), finite('end_time')
        limit_seconds = finite('time_limit') * 60  # Slurm JSON time_limit is minutes.
        nodes, cpus, memory = finite('node_count'), finite('cpus'), finite('memory_per_node')
        tres = dict(x.split('=', 1) for x in fields['tres_alloc_str'].split(',') if '=' in x)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise GateClosed('Malformed or unsupported scheduler JSON allocation proof') from error
    if (claim.get('nonce') != nonce or not nonce or type(fields.get('job_id')) is not int or
            str(fields['job_id']) != job or fields.get('job_state') != ['RUNNING'] or
            type(fields.get('user_id')) is not int or fields['user_id'] != os.getuid() or
            fields.get('account') != RESOURCES['account'] or fields.get('partition') != RESOURCES['partition'] or
            fields.get('qos') != RESOURCES['qos'] or nodes != 1 or cpus != 2 or memory != 8192 or
            type(fields.get('restart_cnt')) is not int or fields['restart_cnt'] != 0 or
            tres.get('cpu') != '2' or tres.get('node') != '1' or
            tres.get('mem') not in ('8G', '8192M') or
            any(key.startswith('gres/gpu') and value != '0' for key,value in tres.items())):
        raise GateClosed('Actual allocation differs from approved one-node2CPU/8GiB/0GPU request')
    # Slurm can report a slightly later end_time than start_time + time_limit
    # (observed: +3 seconds in job 355691). Never spend that extra scheduler
    # allowance: enforce the earlier of its deadline and our approved clock.
    deadline = min(end, start + limit_seconds)
    if (type(now) not in (int, float) or not math.isfinite(now) or
            not 0 < limit_seconds <= resources['total_allocation_seconds_including_retries'] or
            not start <= now < deadline-LIMITS['closure_reserve_seconds']):
        raise GateClosed('Original allocation clock exhausted or exceeds cumulative cap')
    return dict(job_id=job, started_epoch=start, allocation_deadline=deadline,
                work_deadline=deadline-LIMITS['closure_reserve_seconds'], nonce=nonce,
                clock_source='scontrol_json_unix_epochs', scheduler_json_parser=parser,
                scheduler_reported_end_epoch=end,
                scheduler_time_limit_seconds=limit_seconds,
                scheduler_snapshot_sha256=hashlib.sha256(text.encode()).hexdigest())


def apparent_bytes(root):
    total, count = 0, 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = checked_path(Path(directory)/name)
            count += 1
            if count > 10000:
                raise GateClosed('Unexpected acquisition artifact count')
            try:
                size = path.lstat().st_size
            except FileNotFoundError:
                # put(replace=True) can rename this bounded temporary receipt
                # after walk lists it. Count its entire allowed size instead
                # of aborting the transfer or silently discounting storage.
                # Missing downloads, directories and other files still fail.
                if name not in files or not name.endswith('.json.tmp'):
                    raise
                destination = checked_path(path.with_suffix(''))
                info = destination.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_size > RECEIPT_BYTES:
                    raise GateClosed('Vanished temporary has no bounded regular receipt')
                size = RECEIPT_BYTES
            total += size
    return total


def saved_allocation_context(root, job, nonce, claim, resources):
    """Preserve the exact scheduler source before any admission can fail."""
    argv = ['scontrol', '--json', 'show', 'job', job]
    text = subprocess.check_output(argv, text=True, timeout=15)
    put(root/'scheduler-before-worker.json', dict(argv=argv, captured_epoch=time.time(),
        raw_json=text, sha256=hashlib.sha256(text.encode()).hexdigest()))
    return allocation_context(text, job, nonce, claim, resources=resources)


class TransferLedger:
    def __init__(self, root, *, cap=LIMITS['download_charged_bytes'], metadata_cap=LIMITS['metadata_received_bytes']):
        self.root, self.cap, self.metadata_cap = checked_path(root), cap, metadata_cap
        self.path = self.root/'transfer-ledger.json'
        if self.path.exists():
            raise GateClosed('Transfer attempt already exists; no resume/retry into preserved partials')
        self.data = dict(charged_bytes=0, known_received_bytes=0, metadata_received_bytes=0,
                         accounting='Known body bytes plus bounded header reservations; unknown responses retain full reservations.',
                         cap_bytes=cap, reservations=[])
        put(self.path, self.data)

    def reserve(self, maximum, url, start, end):
        if (len(self.data['reservations']) >= LIMITS['max_http_responses'] or
                not 0 < maximum <= CHUNK or self.data['charged_bytes']+maximum+HEADER_CAP > self.cap):
            raise GateClosed('Cumulative response/transfer reservation cap; no request sent')
        index = len(self.data['reservations'])
        self.data['charged_bytes'] += maximum+HEADER_CAP
        self.data['reservations'].append(dict(id=index, reserved_bytes=maximum, header_reserved_bytes=HEADER_CAP, received_bytes=0,
            state='unknown', url_sha256=hashlib.sha256(url.encode()).hexdigest(),
            host=urllib.parse.urlsplit(url).hostname, start=start, end=end))
        put(self.path, self.data, replace=True)
        return index

    def settle(self, index, received, *, complete, metadata=False):
        record = self.data['reservations'][index]
        if record['state'] != 'unknown' or not 0 <= received <= record['reserved_bytes']:
            raise GateClosed('Invalid/double transfer accounting')
        record.update(received_bytes=received, state='complete' if complete else 'uncertain')
        self.data['known_received_bytes'] += received
        if metadata:
            if self.data['metadata_received_bytes']+received > self.metadata_cap:
                raise GateClosed('Metadata response body accounting exceeded its explicit cap')
            self.data['metadata_received_bytes'] += received
        if complete:
            self.data['charged_bytes'] -= record['reserved_bytes']-received
        put(self.path, self.data, replace=True)


def allowed_url(value):
    parts = urllib.parse.urlsplit(value)
    host = parts.hostname or ''
    if (parts.scheme != 'https' or parts.username or parts.password or parts.port not in (None,443) or
            parts.fragment or not (host in ('drive.google.com','drive.usercontent.google.com') or
                                   host.endswith('.googleusercontent.com'))):
        raise GateClosed('Redirect is outside the reviewed official TLS delivery hosts')
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class HeaderReader:
    """Bound all response status/header lines, including interim HTTP 100 blocks."""
    def __init__(self, stream):
        self.stream, self.received = stream, 0

    def readline(self, limit=-1):
        remaining = HEADER_CAP-self.received
        if remaining <= 0:
            raise GateClosed('Response status/header byte cap exceeded')
        value = self.stream.readline(min(remaining, limit) if limit >= 0 else remaining)
        self.received += len(value)
        if not value.endswith(b'\n'):
            raise GateClosed('Truncated or oversized response status/header line')
        return value

    def __getattr__(self, name):
        return getattr(self.stream,name)


class BoundedHTTPResponse(http.client.HTTPResponse):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.fp = HeaderReader(self.fp)


class BoundedHTTPSConnection(http.client.HTTPSConnection):
    response_class = BoundedHTTPResponse


class BoundedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self,request):
        return self.do_open(BoundedHTTPSConnection,request,context=self._context)


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
        BoundedHTTPSHandler(context=ssl.create_default_context())).open


def download(root, ledger, deadline, *, open_response=None, size=ARCHIVE_BYTES, chunk=CHUNK):
    """One contiguous range transfer; no retry/resume or unbounded response read."""
    open_response = opener() if open_response is None else open_response
    target = checked_path(root/'archive-attempt-001.partial')
    url, etag, offset = SOURCE_URL, None, 0
    checksum = hashlib.sha256()
    with target.open('xb') as output:
        os.chmod(target, 0o600)
        while offset < size:
            end = min(offset+chunk,size)-1
            for redirect in range(LIMITS['max_redirects_per_range']+1):
                if time.time() >= deadline:
                    raise GateClosed('Transfer work deadline reached')
                allowed_url(url)
                maximum = max(end-offset+1, METADATA_RESPONSE_CAP)
                index = ledger.reserve(maximum, url, offset, end)
                received, complete, metadata = 0, False, False
                response = None
                try:
                    headers = {'Range':f'bytes={offset}-{end}','Accept-Encoding':'identity'}
                    if etag is not None:
                        headers['If-Match'] = etag
                    request = urllib.request.Request(url, headers=headers)
                    try:
                        response = open_response(request, timeout=max(.1,min(10,deadline-time.time())))
                    except urllib.error.HTTPError as error:
                        response = error
                    status = response.status
                    if hasattr(response.headers,'get_all') and any(
                            len(response.headers.get_all(key,[]))>1 for key in
                            ('Content-Length','Content-Range','ETag','Content-Disposition',
                             'Content-Encoding','Transfer-Encoding','Location')):
                        raise GateClosed('Ambiguous duplicate response identity or framing headers')
                    length = response.headers.get('Content-Length','')
                    length = int(length) if length.isdigit() else None
                    if status != 206:
                        metadata = True
                        budget = min(METADATA_RESPONSE_CAP, ledger.metadata_cap-ledger.data['metadata_received_bytes'])
                        if budget <= 0:
                            raise GateClosed('Metadata body cap exhausted')
                        count = min(budget,length) if length is not None else budget
                        raw = response.read(count)
                        if len(raw)>count:
                            raise GateClosed('Transport exceeded requested metadata read bound')
                        received = len(raw)
                        complete = length is not None and len(raw) == length
                        fd = os.open(root/f'response-{index:04d}-metadata.bin', os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
                        with os.fdopen(fd,'wb') as evidence:
                            evidence.write(raw); evidence.flush(); os.fsync(evidence.fileno())
                        if status in (301,302,303,307,308) and complete:
                            if redirect == LIMITS['max_redirects_per_range']:
                                raise GateClosed('Official redirect bound exhausted')
                            location = response.headers.get('Location','')
                            if not location:
                                raise GateClosed('Official redirect omitted its location')
                            url = allowed_url(urllib.parse.urljoin(url,location))
                            continue
                        raise GateClosed('Source quota/HTML/unavailable or unbounded metadata; no retry or mirror fallback')
                    if (length != end-offset+1 or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{size}' or
                            response.headers.get('Content-Encoding','identity') != 'identity' or
                            response.headers.get('Transfer-Encoding') is not None):
                        raise GateClosed('Exact original file range/length/encoding contract unavailable')
                    current_etag = response.headers.get('ETag','')
                    header = email.message.Message(); header['content-disposition'] = response.headers.get('Content-Disposition','')
                    if (not re.fullmatch(r'"[^"\r\n]+"',current_etag) or
                            etag is not None and current_etag != etag or header.get_filename() != ARCHIVE_NAME):
                        raise GateClosed('Strong consistent entity or exact original filename unavailable')
                    etag = current_etag
                    while received < length:
                        if time.time() >= deadline:
                            raise GateClosed('Transfer work deadline reached')
                        count = min(BLOCK,length-received)
                        block = response.read(count)
                        if len(block)>count:
                            raise GateClosed('Transport exceeded requested archive read bound')
                        if not block:
                            raise GateClosed('Partial response body; no automatic retry')
                        received += len(block); output.write(block); checksum.update(block)
                    output.flush(); os.fsync(output.fileno())
                    complete = True
                    offset = end+1
                    break
                finally:
                    if response is not None:
                        response.close()
                    ledger.settle(index, received, complete=complete, metadata=metadata)
            else:
                raise GateClosed('Redirect loop')
    final = checked_path(root/ARCHIVE_NAME)
    if final.exists() or final.is_symlink():
        raise GateClosed('Original completed archive path already exists')
    os.rename(target,final)
    result = dict(status='complete_official_location_transfer', archive_path=str(final), bytes=offset,
        archive_sha256=checksum.hexdigest(), strong_etag=etag, official_drive_id=DRIVE_ID,
        final_url_sha256=hashlib.sha256(url.encode()).hexdigest(), final_host=urllib.parse.urlsplit(url).hostname,
        historical_checksum_independently_published=False)
    put(root/'download-complete.json',result)
    return result


def worker(root, scratch, deadline):
    os.setsid(); os.chdir(root); os.environ.update(safe_environment(scratch))
    resource.setrlimit(resource.RLIMIT_AS,(6*2**30,6*2**30))
    resource.setrlimit(resource.RLIMIT_FSIZE,(LIMITS['retained_bytes'],LIMITS['retained_bytes']))
    try:
        transfer = download(root,TransferLedger(root),deadline)
        proof = inspection(transfer['archive_path'],deadline=deadline)
        if proof['archive_sha256'] != transfer['archive_sha256']:
            raise GateClosed('Archive changed between official transfer and offline identity inspection')
        put(root/'archive-identity.json',proof)
        put(root/'stage-result.json',dict(status='acquisition_and_internal_identity_verified',
            archive_sha256=proof['archive_sha256'],ready_to_import=False,ready_to_host=False,
            network_scope='official archive only',model_provider_calls=0))
    except BaseException as error:
        put(root/'failure.json',dict(status='stopped_preserving_partial_evidence',error_type=type(error).__name__,
            reason=str(error)[:1000] if isinstance(error,GateClosed) else 'Transport/IO/inspection error; private evidence retained.',
            at_epoch=time.time(),automatic_retry=False))
        raise SystemExit(1)


def own_worker(root,scratch,deadline,*,target=worker,prior_retained_bytes=0):
    previous = _subreaper(True); baseline = set(_processes()); process = tree = None
    stopped = [False]
    handlers = {sig:signal.signal(sig,lambda *_:stopped.__setitem__(0,True)) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        process = multiprocessing.get_context('fork').Process(target=target,args=(root,scratch,deadline))
        try:
            process.start()
        finally:
            if process.pid is not None:
                tree = OwnedEpisodeTree(process,baseline)
        while process.is_alive() and time.time()<deadline and not stopped[0]:
            tree.refresh()
            if (apparent_bytes(root.parent)+prior_retained_bytes>LIMITS['retained_bytes']-LIMITS['receipt_reserve_bytes'] or
                    apparent_bytes(scratch)>LIMITS['temporary_bytes']):
                stopped[0]=True
                break
            time.sleep(.25)
    finally:
        try:
            closure = tree.stop(3) if tree else dict(remaining_owned_processes={},child_exitcode=None)
            put(root/'closure.json',closure)
        finally:
            _subreaper(previous)
            for sig,handler in handlers.items():
                signal.signal(sig,handler)
    if closure['remaining_owned_processes'] or closure['child_exitcode'] != 0:
        raise GateClosed('Acquisition incomplete; owned children closed and partials retained')
    return closure


def run(root,nonce):
    proposal = approved(root)
    claim = read(root/'launch.json')
    job = os.environ.get('SLURM_JOB_ID','')
    if (not job.isdigit() or f'/job_{job}/' not in Path('/proc/self/cgroup').read_text() or
            claim.get('proposal_sha256') != digest(proposal)):
        raise GateClosed('Must execute inside this one approved owned allocation')
    # A durable exclusive claim forbids reentry even if the previous node-local
    # directory vanished. No second controller can reset the attempt clock.
    put(root/'run-claim.json',dict(job_id=job,nonce=nonce,at_epoch=time.time()))
    context = saved_allocation_context(root,job,nonce,claim,proposal['resources'])
    prior_retained = apparent_bytes(PRIOR_ROOT) if proposal.get('resource_profile') == REPLACEMENT_PROFILE else 0
    if (shutil.disk_usage(root).free<LIMITS['initial_persistent_free_bytes'] or
            shutil.disk_usage('/tmp').free<LIMITS['initial_temporary_free_bytes']):
        raise GateClosed('Approved persistent or node-local free-space admission failed')
    projected = (apparent_bytes(root)+prior_retained+ARCHIVE_BYTES+LIMITS['metadata_received_bytes']+
                 2*LIMITS['receipt_reserve_bytes'])
    if projected>LIMITS['retained_bytes']:
        raise GateClosed('Conservative archive/metadata/receipt retained-space projection exceeds cap')
    scratch = checked_path(Path('/tmp')/('webprm-cms-'+job+'-'+secrets.token_hex(4)))
    scratch.mkdir(mode=0o700)
    attempt = checked_path(root/('attempt-'+job)); attempt.mkdir(mode=0o700)
    put(attempt/'context.json',dict(context,scratch=str(scratch),proposal_sha256=digest(proposal),prior_retained_bytes=prior_retained))
    own_worker(attempt,scratch,context['work_deadline'],prior_retained_bytes=prior_retained)
    result = read(attempt/'stage-result.json')
    proof, transfer = read(attempt/'archive-identity.json'), read(attempt/'download-complete.json')
    if (result.get('status') != 'acquisition_and_internal_identity_verified' or
            result.get('archive_sha256') != proof.get('archive_sha256') or
            result.get('archive_sha256') != transfer.get('archive_sha256') or
            not re.fullmatch('[0-9a-f]{64}',result.get('archive_sha256','')) or
            time.time()>=context['allocation_deadline'] or apparent_bytes(root)+prior_retained>LIMITS['retained_bytes'] or
            apparent_bytes(scratch)>LIMITS['temporary_bytes']):
        raise GateClosed('Final identity, clock or storage proof failed; no completion promoted')
    put(root/'completed.json',dict(status='stage_completed',job_id=job,
        completed_epoch=time.time(),allocation_deadline=context['allocation_deadline'],
        retained_bytes=apparent_bytes(root)+prior_retained,prior_retained_bytes=prior_retained,temporary_bytes=apparent_bytes(scratch),
        result_path=str(attempt/'stage-result.json'),ready_to_host=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    for mode in ('prepare','launch','run'):
        modes.add_argument('--'+mode,action='store_true')
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--nonce')
    parser.add_argument('--replacement-59m',action='store_true',help='Prepare only: pending explicit exception for the closed four-second predecessor')
    parser.add_argument('--predecessor-scheduler-json',type=Path,help='Preserved terminal scheduler JSON when Slurm has aged out the predecessor')
    args=parser.parse_args(); root=checked_path(args.root)
    if args.replacement_59m and not args.prepare:
        parser.error('--replacement-59m applies only to a new pending proposal')
    if args.predecessor_scheduler_json is not None and not (args.prepare and args.replacement_59m):
        parser.error('--predecessor-scheduler-json requires --prepare --replacement-59m')
    if args.prepare:
        proposal=prepare(root,replacement=args.replacement_59m,predecessor_snapshot=args.predecessor_scheduler_json); print(json.dumps(dict(status='prepared_not_approved',proposal_sha256=digest(proposal))))
    elif args.launch:
        launch(root)
    else:
        run(root,args.nonce)


if __name__=='__main__':
    main()
