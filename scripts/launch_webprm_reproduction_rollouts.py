#!/usr/bin/env python3
"""Prepare or own the exact CPU-only paired diagnostic; never submit a job.

Preparation copies the already frozen original campaign source. Execution needs
a separate approval receipt and one matching Slurm allocation. Every condition
gets a new original site copy; all copies and outcomes are retained.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openwebrl.webprm_reproduction_rollout import CONDITIONS, MINI, contract
from openwebrl.webprm_lifecycle import OwnedEpisodeTree, _process, _processes, _subreaper
from scripts.launch_webprm_reddit_pilot import (BROWSER, RUNTIME, SCRATCH, ORIGINAL_MANIFEST,
    checked_path, cpu_scheduler_epochs, digest, environment, read, reference, sha, _command, _stop)

BASE = RUNTIME / 'next-stage-preparation/reddit-table5-190-v13'
DIAGNOSTIC = RUNTIME / 'reproduction-diagnostics-20261016-v3'
DEFAULT_ROOT = RUNTIME / 'next-stage-preparation/reproduction-rollouts-20-v5'
ACTORS = (MINI, 'gpt-4o-2024-05-13')
GPT4O = ACTORS[1]
STUDY = 'stopping_rule_20_v1'
PRIOR_PILOT = RUNTIME / 'next-stage-preparation/reproduction-rollouts-20-v5'
TASKS = (14, 97, 131, 133, 141)
RESOURCES = dict(account='vlm', partition='gpu-h200', gpus=0, cpus=8, memory_gib=32,
                 total_seconds=14340, maximum_allocations=1)
LIMITS = dict(api_usd=15., paid_requests=1800, per_episode_requests=90,
              scratch_bytes=1280 * 1024**3, durable_bytes=16 * 1024**3,
              cleanup_seconds=180, new_model_or_image_download_bytes=0, max_workers=1)
OWN_FILES = ('openwebrl/webprm_reproduction_rollout.py', 'openwebrl/webprm_webarena_worker.py',
             'openwebrl/webprm_browser_preflight.py',
             'scripts/launch_webprm_reproduction_rollouts.py', 'tests/test_webprm_reproduction_rollouts.py',
             'scripts/probe_webprm_reproduction.py', 'scripts/audit_webprm_reproduction_probe.py',
             'tests/test_webprm_reproduction_probe.py')
REQUIRED_ENTRYPOINTS = ('openwebrl/webprm_browser_preflight.py',
                        'openwebrl/webprm_webarena_worker.py',
                        'scripts/launch_webprm_reddit_pilot.py')


def configure_study(name):
    """Select only a hard-coded prospective study, before reading its approval."""
    global STUDY, CONDITIONS, ACTORS, TASKS, RESOURCES, LIMITS, OWN_FILES, REQUIRED_ENTRYPOINTS
    if name == STUDY:
        return
    if name not in ('viewer_parity_8_v1', 'viewer_parity_4_v1') or STUDY != 'stopping_rule_20_v1':
        raise ValueError('Unknown study or attempted study switch')
    from openwebrl.webprm_viewer_parity import CONDITIONS as parity_conditions
    STUDY, CONDITIONS, ACTORS, TASKS = name, parity_conditions, (MINI,), (97,)
    RESOURCES = dict(RESOURCES, total_seconds=7200 if name == 'viewer_parity_4_v1' else 10800)
    LIMITS = dict(LIMITS, api_usd=6.5 if name == 'viewer_parity_4_v1' else 13.,
                  paid_requests=360 if name == 'viewer_parity_4_v1' else 720,
                  scratch_bytes=(1280 if name == 'viewer_parity_4_v1' else 1536)*1024**3)
    OWN_FILES += ('openwebrl/webprm_viewer_parity.py', 'tests/test_webprm_viewer_parity.py',
                  'scripts/audit_webprm_viewer_parity.py', 'scripts/audit_webprm_reproduction_rollouts.py',
                  'tests/test_audit_webprm_reproduction_rollouts.py')
    REQUIRED_ENTRYPOINTS += ('openwebrl/webprm_viewer_parity.py',)


def condition_contract(condition):
    if STUDY == 'stopping_rule_20_v1':
        return contract(condition)
    from openwebrl.webprm_reproduction_rollout import CONDITIONS as stopping_conditions
    return contract(stopping_conditions[1])


def prior_limits():
    if STUDY == 'stopping_rule_20_v1':
        return dict(charged_usd=0, paid_requests=0, scheduler_seconds=0, scratch_bytes=0, durable_bytes=0)
    final = read(PRIOR_PILOT/'final-audit/final.json')
    analysis = read(PRIOR_PILOT/'final-audit/analysis.json')
    if (final.get('verified_complete') is not True or not final['all_owned_processes_closed']
            or final['unknown_reservations'] != 0):
        raise ValueError('Earlier pilot must be audited and closed')
    prior = dict(charged_usd=float(final['charged_usd']), paid_requests=final['paid_requests'],
                 scheduler_seconds=final['total_scheduler_seconds'],
                 scratch_bytes=analysis['current_scratch_apparent_bytes'],
                 durable_bytes=apparent(PRIOR_PILOT))
    if (prior['charged_usd']+LIMITS['api_usd'] > 15
            or prior['paid_requests']+LIMITS['paid_requests'] > 1800
            or prior['scheduler_seconds']+RESOURCES['total_seconds'] > 14400):
        raise ValueError('Successor exceeds the existing cumulative API/request/time caps')
    return prior


def validate_entrypoints(proposal):
    source=Path(proposal['source'])
    for relative in REQUIRED_ENTRYPOINTS:
        path=source/relative
        if not path.is_file() or proposal['sha256'].get(str(path))!=sha(path):
            raise ValueError('Required subprocess entrypoint missing or unfrozen: '+relative)


def put(path, value):
    path = checked_path(path, RUNTIME)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def cells():
    # Alternate order within task/actor pairs to balance chronological drift.
    result = []
    if STUDY in ('viewer_parity_8_v1', 'viewer_parity_4_v1'):
        # Two fixed draws per factor cell; reverse second order to balance time.
        orders = (CONDITIONS,) if STUDY == 'viewer_parity_4_v1' else (CONDITIONS, CONDITIONS[::-1])
        for replicate, order in enumerate(orders):
            for condition in order:
                result.append(dict(index=len(result), task_id=97, actor=MINI,
                                   condition=condition, replicate=replicate))
        return result
    for task in TASKS:
        for actor in ACTORS:
            order = CONDITIONS if (TASKS.index(task) + ACTORS.index(actor)) % 2 == 0 else CONDITIONS[::-1]
            for condition in order:
                result.append(dict(index=len(result), task_id=task, actor=actor, condition=condition))
    return result


def validate(proposal):
    if (proposal.get('study', 'stopping_rule_20_v1') != STUDY
            or proposal.get('resources') != RESOURCES or proposal.get('limits') != LIMITS
            or proposal.get('cells') != cells() or proposal.get('retain_all_site_copies') is not True
            or proposal.get('resampling') is not False or proposal.get('additional_api_budget_usd') != 0):
        raise ValueError('Exact fixed diagnostic/resource contract changed')
    root = checked_path(proposal['root'], RUNTIME, directory=True)
    if proposal.get('source') != str(root / 'source'):
        raise ValueError('Frozen source must belong to this preparation')
    validate_entrypoints(proposal)
    for path, expected in proposal['sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen diagnostic input changed: ' + path)
    if proposal['sha256'].get(str(Path(__file__).resolve())) != sha(__file__):
        raise ValueError('Run the frozen diagnostic controller')
    from scripts.run_webprm_webarena_pilot import validate_plan
    for condition in CONDITIONS:
        plan = read(root / ('plan-' + condition + '.json'))
        if (plan['reproduction_diagnostic'] != condition_contract(condition) or plan['task_ids'] != list(TASKS)
                or plan['actors'] != list(ACTORS) or plan['selectors'] != ['no_search']):
            raise ValueError('Fixed condition/task/actor settings changed')
        if STUDY in ('viewer_parity_8_v1', 'viewer_parity_4_v1'):
            from openwebrl.webprm_viewer_parity import contract as parity_contract
            if plan.get('viewer_parity_diagnostic') != parity_contract(condition, plan['arbiter_source']):
                raise ValueError('Exact parity condition differs')
        validate_plan(plan)
    if proposal.get('prior_limits', prior_limits()) != prior_limits():
        raise ValueError('Prior consumed resources changed')
    return proposal


def prepare(root):
    # This branch starts no browser, namespace, allocation, service, or API call.
    root = checked_path(root, RUNTIME)
    root.mkdir(mode=0o700)
    source = root / 'source'
    shutil.copytree(BASE / 'source', source)
    for name in OWN_FILES:
        target = source / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    previous = read(read(BASE / 'approved-request.json')['proposal'])
    hashes = {str(p): sha(p) for p in source.rglob('*') if p.is_file()}
    external = dict(previous['source_sha256'])
    # Old code is still immutable, but only its provenance inputs are needed.
    external = {p: h for p, h in external.items() if str(BASE / 'source') not in p}
    for name in ('next-rollout-design.json', 'audits/final.json', 'budget-reservation-overlay.json', 'budget.json'):
        path = DIAGNOSTIC / name; external[str(path)] = sha(path)
    if STUDY in ('viewer_parity_8_v1', 'viewer_parity_4_v1'):
        for name in ('final-audit/final.json', 'final-audit/analysis.json',
                     'final-audit/viewer-parity.json', 'campaign/shared-budget.json'):
            path = PRIOR_PILOT/name; external[str(path)] = sha(path)
    for condition in CONDITIONS:
        plan = read(BASE / 'worker-template.json')
        from openwebrl.webprm_webarena import TASK_FILE, digest as task_digest
        tasks = json.loads((Path(plan['arbiter_source']) / TASK_FILE).read_text())
        indexed = {row['task_id']: row for row in tasks}
        plan.update(task_ids=list(TASKS), actors=list(ACTORS), selectors=['no_search'],
                    tasks_sha256=task_digest([indexed[i] for i in TASKS]),
                    endpoints={'reddit': 'http://127.0.0.1'},
                    reproduction_diagnostic=condition_contract(condition), code_sha256=hashes,
                    budget_path=str(root / 'campaign/shared-budget.json'), api_usd_cap=0,
                    authorization_status='pending', deadline=0)
        if STUDY in ('viewer_parity_8_v1', 'viewer_parity_4_v1'):
            from openwebrl.webprm_viewer_parity import contract as parity_contract
            plan['viewer_parity_diagnostic'] = parity_contract(condition, plan['arbiter_source'])
        put(root / ('plan-' + condition + '.json'), plan)
    proposal = dict(schema_version=1, status='prepared_not_approved', root=str(root), source=str(source),
                    resources=RESOURCES, limits=LIMITS, cells=cells(), resampling=False,
                    retain_all_site_copies=True, additional_api_budget_usd=0,
                    api_funding='Suballocate $15 from the closed Reddit campaign remaining $250 cap',
                    provider='OpenAI: task intent, current page text, and action history; same broker transport',
                    original_site=previous['domains']['reddit'], env_file=previous['env_file'],
                    historical_reproduction=False, sha256=dict(hashes, **external))
    if STUDY in ('viewer_parity_8_v1', 'viewer_parity_4_v1'):
        proposal.update(study=STUDY, prior_limits=prior_limits(),
            api_funding=f'At most ${LIMITS["api_usd"]} and {LIMITS["paid_requests"]} requests from the closed $15/1800 pilot cap; no added API budget',
            scratch_scope='Global retained previous pilot bytes plus this study; all copies retained',
            required_new_approval=('One additional CPU-only allocation and four fixed episodes; preserve original1280GiB scratch/API/time caps'
                if STUDY == 'viewer_parity_4_v1' else 'One additional allocation, eight episodes, and 1536 GiB cumulative scratch cap'))
    for condition in CONDITIONS:
        p = root / ('plan-' + condition + '.json'); proposal['sha256'][str(p)] = sha(p)
    batch = ('#!/usr/bin/env bash\n#SBATCH --job-name=webprm-repro20\n#SBATCH --account=vlm\n'
             '#SBATCH --partition=gpu-h200\n#SBATCH --qos=normal\n#SBATCH --nodes=1\n'
             '#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=8\n#SBATCH --mem=32G\n'
             f'#SBATCH --time={RESOURCES["total_seconds"]//3600:02d}:{RESOURCES["total_seconds"]%3600//60:02d}:00\n#SBATCH --no-requeue\n#SBATCH --signal=B:TERM@180\n'
             '#SBATCH --export=ALL,TMPDIR=/tmp\n#SBATCH --output=' + str(root / 'campaign-%j.out') + '\n'
             'set -euo pipefail\ncd ' + shlex.quote(str(source)) + '\n'
             'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TMPDIR=/tmp\n'
             'export WANDB_PROJECT=openwebrl-evals WANDB_MODE=disabled\n'
             'export BASH_ENV=/dev/null ENV=/dev/null HISTFILE=/dev/null\n'
             ': "${1:?Provide exact approval receipt}"\nexec ' + shlex.quote(BROWSER) +
             ' -B scripts/launch_webprm_reproduction_rollouts.py --proposal ' + shlex.quote(str(root / 'proposal.json')) +
             ' --approval "$1" --execute\n')
    (root / 'campaign.sbatch').write_text(batch)
    proposal['sha256'][str(root / 'campaign.sbatch')] = sha(root / 'campaign.sbatch')
    put(root / 'proposal.json', proposal)
    print(json.dumps(dict(status='prepared_not_submitted', proposal=str(root / 'proposal.json'),
                          proposal_sha256=digest(proposal), resources=RESOURCES, limits=LIMITS)))


def approval_check(proposal, approval):
    if (approval.get('status') != 'approved' or approval.get('proposal_sha256') != digest(proposal)
            or approval.get('resources') != RESOURCES or approval.get('limits') != LIMITS
            or approval.get('provider_payload_approved') is not True
            or not approval.get('user_approval')):
        raise ValueError('Exact allocation, existing-cap suballocation, and provider payload approval required')
    if STUDY in ('viewer_parity_8_v1', 'viewer_parity_4_v1') and (
            approval.get('prior_scheduler_seconds') != prior_limits()['scheduler_seconds']
            or approval.get('original_total_cap_seconds') != 14400):
        raise ValueError('Exact cumulative prior time and original cap required')


def scheduler_contract(text, snapshot, job):
    fields = dict(re.findall(r'(\w+)=(\S+)', text))
    epochs = cpu_scheduler_epochs(snapshot, job)
    tres = dict(part.split('=', 1) for part in fields.get('AllocTRES', '').split(',') if '=' in part)
    if (fields.get('Account') != 'vlm' or fields.get('Partition') != 'gpu-h200'
            or fields.get('NumCPUs') != '8' or fields.get('NumNodes') != '1'
            or tres.get('cpu') != '8' or tres.get('mem') not in ('32G', '32768M')
            or any(k.startswith('gres/gpu') and v != '0' for k, v in tres.items())
            or epochs['limit_seconds'] > RESOURCES['total_seconds']):
        raise ValueError('Scheduler allocation differs from the CPU-only exact approval')
    return epochs


def stage(context_path, index, inside=False, token=None):
    owned = read(checked_path(context_path, RUNTIME)); proposal = validate(owned['proposal'])
    approval_check(proposal, owned['approval'])
    owner = _process(owned['owner_pid'])
    if (str(owned['job_id']) != os.environ.get('SLURM_JOB_ID') or owned['deadline'] <= time.time()
            or owner['state'] == 'Z' or owner['start_ticks'] != owned['owner_start_ticks']):
        raise ValueError('Stale diagnostic owner')
    ancestor = os.getppid()
    for _ in range(32):
        if ancestor == owned['owner_pid']: break
        if ancestor <= 1: raise ValueError('Service requires its live owner ancestor')
        ancestor = _process(ancestor)['ppid']
    else: raise ValueError('Service requires its live owner ancestor')
    cell = cells()[index]
    if cell['index'] != index: raise ValueError('Unknown fixed cell')
    folder = Path(owned['output']) / f'cell-{index:02d}'
    sandbox = checked_path(Path(owned['scratch']) / f'state-{index:02d}', SCRATCH, directory=True)
    site = reference('run_reddit_instance.py'); site.SCRATCH = Path(owned['scratch'])
    out = folder / 'site'
    a = SimpleNamespace(sandbox=str(sandbox), output=str(out), identity=ORIGINAL_MANIFEST, scope='original',
                        deadline=owned['deadline'], state_provenance=str(folder / 'copy/receipt.json'),
                        startup_review=proposal['original_site']['startup_review'], launch_token=token)
    if inside: return site.inside(a)
    fields = site.provenance(a); out.mkdir(mode=0o700); a.launch_token = secrets.token_hex(24)
    claim = site.claim_state(a, a.launch_token)
    site.write(out / 'launch.json', dict(sandbox=a.sandbox, output=a.output, scope=a.scope,
        image_sha256=a.identity, launch_token=a.launch_token, parent_pid=os.getpid(),
        parent_start_ticks=_process(os.getpid())['start_ticks'],
        outer_network_namespace=os.readlink('/proc/self/ns/net'), state_claim_path=claim, provenance=fields))
    site.prepare(sandbox, out)
    def child(_):
        os.setsid()
        command = ['unshare', '--user', '--map-root-user', '--net', BROWSER, '-B', str(Path(__file__).resolve()),
                   '--context', str(context_path), '--stage-index', str(index), '--inside', '--token', a.launch_token]
        os.execvpe(command[0], command, site.safe_env())
    return site.supervise(a, target=child)


def apparent(path):
    return int(subprocess.check_output(['du', '--apparent-size', '-s', '-B1', str(path)],
                                       text=True, timeout=30).split()[0])


def broker_directory(scratch, index):
    """Do not depend on the submitter's umask for the broker's private socket."""
    path = checked_path(Path(scratch) / f'b{index:02d}', SCRATCH)
    path.mkdir(mode=0o700)
    if path.stat().st_mode & 0o077:
        raise ValueError('Fresh broker directory is not private')
    return path


def _runner(context):
    os.setsid()
    proposal = context['proposal']; output = Path(context['output']); scratch = Path(context['scratch'])
    env = environment(scratch)
    env['PYTHONPATH'] = str(RUNTIME / 'nltk-runtime/venv/lib/python3.12/site-packages') + ':' + str(REPO)
    env['NLTK_DATA'] = str(RUNTIME / 'nltk-runtime/nltk_data')
    os.environ.update(env)
    site = worker = broker = None
    def check():
        if time.time() >= context['deadline']: raise TimeoutError('Allocation deadline')
        if site is not None and site.poll() is not None: raise RuntimeError('Original site exited')
    def wait(process, until):
        while process.poll() is None:
            check()
            if time.time() >= until: raise TimeoutError('Owned process deadline')
            time.sleep(.5)
        if process.returncode: raise RuntimeError('Owned command failed; inspect its log')
    preparation = reference('prepare_reddit_original.py'); preparation.SCRATCH = scratch
    try:
        preflight = _command(['unshare', '--user', '--map-root-user', '--net', BROWSER, '-B', '-m',
            'openwebrl.webprm_browser_preflight', '--arbiter-source',
            read(Path(proposal['root']) / ('plan-' + CONDITIONS[0] + '.json'))['arbiter_source'],
            '--output', str(output / 'browser-preflight'), '--deadline', str(min(context['deadline'], time.time()+170)),
            '--host-netns', os.readlink('/proc/self/ns/net'), '--terminal-answer-check'], output / 'browser-preflight.log', env)
        worker = preflight; wait(preflight, min(context['deadline'], time.time()+175)); worker = None
        if read(output / 'browser-preflight/receipt.json')['status'] != 'passed': raise RuntimeError('Browser gate failed')
        preparation.prepare('build', scratch / 'baseline', output / 'original-build', context['deadline'],
                            download_receipt=proposal['original_site']['download_receipt'])
        for cell in cells():
            check()
            if time.time()+600 >= context['deadline']: raise TimeoutError('Insufficient time to admit another episode')
            if (prior_limits()['scratch_bytes']+apparent(scratch)+66*1024**3 > LIMITS['scratch_bytes']
                    or prior_limits()['durable_bytes']+apparent(Path(proposal['root'])) >= LIMITS['durable_bytes']):
                raise RuntimeError('Retained storage cap reached')
            index = cell['index']; folder = output / f'cell-{index:02d}'; folder.mkdir()
            put(folder / 'claim.json', cell)
            state = scratch / f'state-{index:02d}'
            preparation.prepare('copy', state, folder / 'copy', context['deadline'],
                                build_receipt=output / 'original-build/receipt.json',
                                verification=proposal['original_site']['raw_verification'])
            site = _command([BROWSER, '-B', str(Path(__file__).resolve()), '--context',
                            str(output / 'execution-context.json'), '--stage-index', str(index)], folder / 'site.log', env)
            until = min(context['deadline'], time.time()+180)
            while not (folder / 'site/instance.json').exists():
                check()
                if time.time() >= until: raise TimeoutError('Site startup')
                time.sleep(.5)
            instance = read(folder / 'site/instance.json')
            enter = ['nsenter', '--target', str(instance['service_pid']), '--user', '--net', '--preserve-credentials']
            number = 0
            while True:
                check(); number += 1
                if time.time() >= until: raise TimeoutError('Database-backed readiness')
                worker = _command(enter+[BROWSER, '-B', '-c',
                    'from scripts.launch_webprm_reddit_pilot import website_ready; website_ready()'],
                    folder / f'readiness-{number:03d}.log', env)
                try: worker.wait(timeout=6)
                except subprocess.TimeoutExpired: _stop(worker, seconds=1)
                if worker.returncode == 0: worker = None; break
                worker = None; time.sleep(1)
            plan = read(Path(proposal['root']) / ('plan-' + cell['condition'] + '.json'))
            socket_path = broker_directory(scratch, index)
            plan.update(authorization_status='approved', job_id=context['job_id'], deadline=context['deadline'],
                        api_usd_cap=LIMITS['api_usd'], broker_socket=str(socket_path / 's'))
            put(folder / 'plan.json', plan)
            identity = digest(plan); task = f"webarena_lite_{cell['task_id']:03d}"
            name = cell['actor'] + '__no_search'; episodes = folder / 'episodes'
            journal = episodes / task / name / 'api.jsonl'; journal.parent.mkdir(parents=True)
            from openwebrl.webprm_broker import BrokerProcess
            routes = {(cell['actor'], 'policy'): dict(local=False, endpoint=None, max_input_tokens=120000, max_output_tokens=2048),
                      (MINI, 'terminal'): dict(local=False, endpoint=None, max_input_tokens=120000, max_output_tokens=2048),
                      (MINI, 'state_diff'): dict(local=False, endpoint=None, max_input_tokens=120000, max_output_tokens=2048),
                      (GPT4O, 'native_fuzzy'): dict(local=False, endpoint=None, max_input_tokens=8192, max_output_tokens=768)}
            registrations = {task+'/'+name: dict(config=dict(plan, identity_sha256=identity, env_file=proposal['env_file']),
                                              journal=str(journal), allowed_routes=[list(k) for k in routes])}
            broker = BrokerProcess(socket_path=str(socket_path / 's'), cells=registrations, routes=routes,
                deadline=context['deadline'], max_dispatches=90, max_paid_dispatches=90, max_local_dispatches=0,
                audit_path=str(folder / 'broker.jsonl')).start()
            lease = folder / 'lease.json'
            put(lease, dict(task_id=task, actor_model=cell['actor'], selector='no_search',
                           instance_receipt=str(folder / 'site/instance.json'),
                           instance_receipt_sha256=sha(folder / 'site/instance.json')))
            argv = enter + [BROWSER, '-B', str(REPO / 'scripts/run_webprm_webarena_pilot.py'),
                '--plan', str(folder / 'plan.json'), '--output', str(episodes), '--execute',
                '--task-id', str(cell['task_id']), '--actor', cell['actor'], '--selector', 'no_search', '--lease', str(lease)]
            put(folder / 'worker-launch.json', dict(argv=argv, cell=cell, identity_sha256=identity))
            worker = _command(argv, folder / 'worker.log', env); wait(worker, context['deadline']); worker = None
            result = read(journal.parent / 'result.json')
            if result['identity_sha256'] != identity: raise ValueError('Worker result identity differs')
            put(folder / 'broker-closure.json', broker.close()); broker = None
            put(folder / 'site/stop.json', dict(reason='fixed diagnostic episode finished', at_epoch=time.time()))
            closing = site; site = None; wait(closing, min(context['deadline'], time.time()+35))
            if read(folder / 'site/closure.json')['remaining_owned_processes']: raise RuntimeError('Site closure failed')
            put(folder / 'finished.json', dict(cell=cell, result_sha256=sha(journal.parent / 'result.json')))
        put(output / 'completed.json', dict(status='execution_finished_requires_independent_audit', episodes=len(cells())))
    finally:
        _stop(worker); _stop(site)
        if broker is not None: put(output / 'interrupted-broker-closure.json', broker.close())


def execute(proposal, approval):
    # Refuse before scheduler access, credentials, or child startup if unapproved.
    approval_check(proposal, approval); validate(proposal)
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit(): raise ValueError('Existing approved Slurm allocation required')
    root = Path(proposal['root']); campaign = root / 'campaign'; campaign.mkdir(exist_ok=True)
    with (campaign / 'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # One immutable slot, including failed startup; no successor submission.
        put(campaign / 'allocation.json', dict(job_id=job, approval=approval, proposal_sha256=digest(proposal)))
        text = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True, timeout=15)
        snapshot = subprocess.check_output(['scontrol', '--json', 'show', 'job', job], text=True, timeout=15)
        epochs = scheduler_contract(text, snapshot, job)
        deadline = min(epochs['end_epoch'], epochs['start_epoch']+RESOURCES['total_seconds'])-LIMITS['cleanup_seconds']
        if deadline <= time.time()+600: raise ValueError('Insufficient approved time remains')
        # Earlier diagnostics and the parent ledger are frozen in the proposal.
        overlay = read(DIAGNOSTIC / 'budget-reservation-overlay.json')
        parent = read(overlay['parent_budget']['path']); prior = read(DIAGNOSTIC / 'budget.json')
        if (sha(overlay['parent_budget']['path']) != overlay['parent_budget']['sha256']
                or parent['charged']+prior['charged']+prior_limits()['charged_usd']+LIMITS['api_usd'] > overlay['combined_cap_usd']):
            raise ValueError('Original combined API cap cannot fund this suballocation')
        put(campaign / 'budget-reservation-overlay.json', dict(parent=overlay, closed_diagnostic_charged=prior['charged'],
            closed_rollout_charged=prior_limits()['charged_usd'],
            reserved_existing_cap_usd=LIMITS['api_usd'], additional_budget_usd=0, child_budget=str(campaign/'shared-budget.json')))
        output = campaign / ('attempt-'+job); output.mkdir()
        scratch = SCRATCH / ('r'+job+'-'+secrets.token_hex(3)); scratch.mkdir(parents=True)
        required = (len(cells())+1)*proposal['original_site']['state_apparent_bytes']+32*1024**3
        if prior_limits()['scratch_bytes']+required > LIMITS['scratch_bytes'] or shutil.disk_usage(scratch).free < required:
            raise ValueError('Fresh retained copies do not fit disk/cap; no services started')
        owned = dict(proposal=proposal, approval=approval, job_id=job, deadline=deadline, scheduler_epochs=epochs,
                     owner_pid=os.getpid(), owner_start_ticks=_process(os.getpid())['start_ticks'],
                     scratch=str(scratch), output=str(output))
        put(output / 'execution-context.json', owned)
        previous = _subreaper(True); baseline = set(_processes()); process = tree = None; stop = [False]
        handlers = {s: signal.signal(s, lambda *_: stop.__setitem__(0, True)) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            process = multiprocessing.get_context('fork').Process(target=_runner, args=(owned,))
            try: process.start()
            finally:
                if process.pid is not None: tree = OwnedEpisodeTree(process, baseline)
            while process.is_alive() and time.time() < deadline and not stop[0]: tree.refresh(); time.sleep(.5)
        finally:
            try:
                closure = tree.stop(10) if tree else dict(remaining_owned_processes={}, child_exitcode=None)
                put(output / 'controller-closure.json', closure)
                put(output / 'storage-final.json', dict(scratch_path=str(scratch), apparent_bytes=apparent(scratch)))
            finally:
                _subreaper(previous)
                for sig, handler in handlers.items(): signal.signal(sig, handler)
        if closure['remaining_owned_processes'] or closure['child_exitcode'] != 0 or not (output/'completed.json').exists():
            raise RuntimeError('Diagnostic incomplete; retain all attempts and reservations, no resampling')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--proposal', type=Path); parser.add_argument('--approval', type=Path)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--context', type=Path)
    parser.add_argument('--stage-index', type=int); parser.add_argument('--inside', action='store_true'); parser.add_argument('--token')
    parser.add_argument('--study', choices=['stopping_rule_20_v1', 'viewer_parity_8_v1', 'viewer_parity_4_v1'], default='stopping_rule_20_v1')
    args = parser.parse_args()
    name = args.study
    if args.context is not None:
        name = read(args.context)['proposal'].get('study', name)
    elif args.proposal is not None:
        name = read(args.proposal).get('study', name)
    configure_study(name)
    if args.prepare: return prepare(args.root)
    if args.context is not None:
        if args.stage_index not in range(len(cells())): parser.error('Exact stage index required')
        return stage(args.context, args.stage_index, args.inside, args.token)
    if args.proposal is None: parser.error('--proposal required')
    proposal = validate(read(args.proposal))
    if not args.execute:
        print(json.dumps(dict(status='validated_not_executed', proposal_sha256=digest(proposal),
                             resources=RESOURCES, limits=LIMITS, episodes=len(cells())))); return
    if args.approval is None: parser.error('Exact --approval required')
    execute(proposal, read(args.approval))


if __name__ == '__main__': main()
