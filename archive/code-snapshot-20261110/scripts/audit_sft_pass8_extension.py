#!/usr/bin/env python3
"""Read-only final audit of the filtered SFT pass@8 extension.

No jobs, browsers, model/judge APIs, queue messages, or completion-pointer updates
are issued. Only a scheduler accounting query and private audit artifacts are
allowed. A successful report is still subject to the owner's W&B/image review.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit

DEFAULT_POINTER = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/current_sft_training_pass8.json')
NATIVE_TASKS = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-turn-bonus-preparation/'
    'task-pool-expansion-20260922/curation-v3-20260929/cohorts-20260930/'
    'comparison-min5-weighted-2000/weighted-tasks.jsonl')
COVARIATE_REFERENCE = Path(__file__).resolve().parents[1]/'openwebrl/docs/arm_results/rl_integration/training-sft-difficulty-passk-20261006.json'
RESOURCES = dict(jobs=4, gpus_per_job=1, gpu_type='H200', cpus_per_job=8,
    memory_gib_per_job=120, total_seconds_per_shard=36000, total_gpu_hours=40, max_concurrent_jobs=4)
LIMITS = dict(browsers_per_shard=1600, concurrent_browsers_per_shard=8,
    judge_http_calls_per_shard=4800, judge_usd_per_shard=12.5, browsers_total=6400,
    judge_http_calls_total=19200, judge_usd_total=50., judge_id='gpt-4.1/action_history')
FINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL',
         'BOOT_FAIL', 'PREEMPTED', 'DEADLINE', 'REVOKED'}
BASELINE = dict(tasks=2000, successes=1293, misses=707, five_valid_failure_tasks=682, any_invalid_tasks=44)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def frozen_collector(source):
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    module = importlib.import_module('openwebrl.arm_task_pass8')
    if Path(module.__file__).resolve() != source/'openwebrl/arm_task_pass8.py':
        raise ValueError('Audit must load the pinned frozen collector, not a working-tree import')
    return module


def verify_source(plan):
    source = Path(plan['source'])
    if sha(source/'reference_manifest.json') != plan['source_manifest_sha256']:
        raise ValueError('Frozen source manifest changed')
    manifest = read(source/'reference_manifest.json')
    for name, expected in manifest['recipe_files_sha256'].items():
        path = (source/name).resolve()
        if not path.is_relative_to(source.resolve()) or sha(path) != expected:
            raise ValueError('Frozen scientific source changed: '+name)
    for name in ('historical_inputs', 'historical_audit', 'actor_hash_proof', 'scope_authorization',
                 'original_plan', 'original_aggregate'):
        if sha(plan[name]) != plan[name+'_sha256']:
            raise ValueError('Pinned campaign input changed: '+name)


def audit_collector_source(plan, approval):
    """Validate additive recovery receipts without changing the original plan.

    A repaired archive verifier can inspect both generations of records. The
    actor/browser/judge files must remain byte-identical to the campaign source.
    """
    parent = Path(plan['source']).resolve()
    before = read(parent/'reference_manifest.json')['recipe_files_sha256']
    selected, receipts = parent, []
    for attempt in approval['attempts']:
        path = attempt.get('recovery_manifest')
        if path is None:
            continue
        receipt = read(path)
        source = Path(receipt['source']).resolve()
        if (digest(receipt) != attempt.get('recovery_manifest_sha256') or
                receipt['campaign_plan_sha256'] != digest(plan) or
                receipt['shard'] != attempt['shard'] or
                str(source) != attempt.get('source') or
                receipt.get('added_budget') is not False or
                receipt.get('scientific_protocol_unchanged') is not True or
                receipt['parent_source_manifest_sha256'] != plan['source_manifest_sha256'] or
                sha(source/'reference_manifest.json') != receipt['source_manifest_sha256']):
            raise ValueError('Recovery collector receipt is not bound to the approved campaign')
        after = read(source/'reference_manifest.json')['recipe_files_sha256']
        entry = 'scripts/recover_sft_pass8_extension.py'
        collector = 'openwebrl/arm_task_pass8.py'
        if (set(after) != set(before) | {entry} or
                {name for name in before if before[name] != after[name]} != {collector} or
                after[collector] != receipt['collector_after_sha256'] or
                before[collector] != receipt['collector_before_sha256'] or
                after[entry] != receipt['recovery_controller_sha256']):
            raise ValueError('Recovery changed files outside its declared verification repair')
        for name, expected in after.items():
            file = (source/name).resolve()
            if not file.is_relative_to(source) or sha(file) != expected:
                raise ValueError('Recovery source hash mismatch')
        for file, expected in receipt['preserved']['immutable_files_sha256'].items():
            if sha(file) != expected:
                raise ValueError('Recovery modified preserved evidence')
        if receipts and selected != source:
            # A later repair must explicitly extend the verifier already bound
            # to this campaign. All scientific files above still match V1.
            if (receipt.get('supersedes_verifier_source') != str(selected) or
                    receipt.get('supersedes_source_manifest_sha256') !=
                    sha(selected/'reference_manifest.json') or
                    receipt.get('supersedes_collector_sha256') != sha(selected/collector)):
                raise ValueError('Different recovery verifier versions require a bound supersession chain')
        selected = source
        receipts.append(dict(job_id=attempt['job_id'], manifest=path,
                             manifest_sha256=digest(receipt), source=str(source)))
    return str(selected), receipts


def campaign_inputs(root, pointer):
    root = Path(root).resolve()
    plan, approval, current = read(root/'plan.json'), read(root/'approval.json'), read(pointer)
    if (plan.get('root') != str(root) or current.get('root') != str(root) or
            plan.get('resources') != RESOURCES or plan.get('limits') != LIMITS or
            approval.get('approved') is not True or approval.get('plan_sha256') != digest(plan) or
            approval.get('resources') != RESOURCES or approval.get('limits') != LIMITS or
            current.get('plan_sha256') != digest(plan)):
        raise ValueError('Plan, approval or current pointer identity/caps mismatch')
    historical, targets = plan['historical_task_order'], plan['extension_task_order']
    if (len(historical) != 2000 or len(set(historical)) != 2000 or len(targets) != 707 or
            len(set(targets)) != 707 or [t for t in historical if t in set(targets)] != targets or
            plan['shards'] != [targets[i::4] for i in range(4)] or
            plan['new_primary_episodes'] != 2121 or plan['new_episodes_per_shard'] != [531,531,531,528]):
        raise ValueError('Historical denominator or filtered extension cohort changed')
    pointers = current.get('shards', [])
    if len(pointers) != 4 or {p['shard'] for p in pointers} != set(range(4)):
        raise ValueError('Exactly four current shard pointers are required')
    attempts = approval.get('attempts', [])
    if not attempts or len({str(a['job_id']) for a in attempts}) != len(attempts):
        raise ValueError('Missing or duplicate scheduler attempt registration')
    for row in attempts:
        if (not str(row['job_id']).isdecimal() or type(row['shard']) is not int or row['shard'] not in range(4) or
                type(row['time_limit_seconds']) is not int or not 0 < row['time_limit_seconds'] <= 36000):
            raise ValueError('Invalid scheduler registration')
    configs, jobs = [], []
    for index in range(4):
        registrations = [r for r in attempts if r['shard'] == index]
        if not registrations:
            raise ValueError('Missing shard scheduler registration')
        job = str(registrations[-1]['job_id'])
        shard_pointer = next(p for p in pointers if p['shard'] == index)
        directory = root/f'shard-{index}'
        if (Path(shard_pointer['root']).resolve() != directory or
                str(shard_pointer.get('job_id') or shard_pointer.get('current_job_id')) != job):
            raise ValueError('Current shard pointer does not identify the final registered attempt')
        receipt = read(directory/'collection-complete.json')
        if (receipt.get('complete') is not True or receipt.get('tasks') != len(plan['shards'][index]) or
                receipt.get('primary_attempts') != plan['new_episodes_per_shard'][index] or
                receipt.get('expected_attempts') != plan['new_episodes_per_shard'][index]):
            raise ValueError('Every shard needs its exact complete collection receipt')
        config = read(directory/'workers'/job/'collector-config.json')
        expected = dict(output=str(directory), shard=index, job_id=job, task_order=plan['shards'][index],
            historical_task_order=historical, extension_task_order=targets, checkpoint=plan['checkpoint'],
            old_record_roots=plan['old_record_roots'], historical_inputs=plan['historical_inputs'],
            historical_inputs_sha256=plan['historical_inputs_sha256'])
        if any(config.get(k) != v for k,v in expected.items()):
            raise ValueError('Final worker scientific configuration differs from the pinned plan')
        configs.append(config)
        jobs.append(job)
    return plan, approval, current, configs, jobs


def scheduler_audit(approval, jobs, *, run=subprocess.run):
    ids = [str(a['job_id']) for a in approval['attempts']]
    response = run(['/usr/bin/sacct', '-X', '-n', '-P', '-j', ','.join(ids),
        '--format=JobIDRaw,State,ElapsedRaw,ExitCode,AllocTRES'], capture_output=True, text=True, timeout=30)
    if response.returncode:
        raise RuntimeError('Final scheduler accounting query failed')
    by_id = {}
    for line in response.stdout.splitlines():
        cols = line.split('|')
        if cols[0] not in ids:
            continue
        if cols[0] in by_id or len(cols)<5 or not cols[2].isdigit():
            raise ValueError('Ambiguous scheduler accounting')
        state = cols[1].split()[0].rstrip('+')
        if state not in FINAL:
            raise ValueError('Every scheduler attempt must be terminal before final audit')
        tres = dict(part.split('=',1) for part in cols[4].split(',') if '=' in part)
        seconds = int(cols[2])
        if seconds and (tres.get('cpu') != '8' or tres.get('mem') not in ('120G','122880M') or
                        tres.get('gres/gpu') != '1' or tres.get('gres/gpu:h200') != '1'):
            raise ValueError('Scheduler allocation differs from approved resources')
        by_id[cols[0]] = dict(job_id=cols[0], state=state, elapsed_seconds=seconds,
                              exit_code=cols[3], alloc_tres=cols[4])
    if set(by_id) != set(ids):
        raise ValueError('Missing scheduler accounting; cannot infer remaining budget')
    for job in jobs:
        if by_id[job]['state'] != 'COMPLETED' or by_id[job]['exit_code'] != '0:0':
            raise ValueError('All four final jobs must have successful scheduler completion')
    shards = []
    for shard in range(4):
        rows = [by_id[str(a['job_id'])] for a in approval['attempts'] if a['shard'] == shard]
        seconds = sum(r['elapsed_seconds'] for r in rows)
        if seconds > 36000:
            raise ValueError('Cumulative scheduler attempts exceeded the original shard cap')
        shards.append(dict(shard=shard, all_attempt_seconds=seconds, cap_seconds=36000,
                           remaining_seconds=36000-seconds, attempts=rows))
    total = sum(s['all_attempt_seconds'] for s in shards)
    return dict(verified=True, shards=shards, total_scheduler_seconds=total,
                h200_gpu_hours=total/3600, h200_gpu_hour_cap=40)


@contextmanager
def cached_history(collector, historical, ids, configs):
    """Use one independently verified 10K history for four frozen deep audits."""
    original = collector.load_old_records
    shared = ('historical_inputs', 'historical_inputs_sha256', 'old_record_roots', 'historical_task_order')
    expected = {k: configs[0][k] for k in shared}
    def load(config, requested_ids):
        if list(requested_ids) != list(ids) or any(config[k] != expected[k] for k in shared):
            raise ValueError('A shard attempted to change the cached immutable history')
        return historical
    collector.load_old_records = load
    try:
        yield
    finally:
        collector.load_old_records = original


def verify_terminal_images(payload):
    """Decode the terminal image bytes and verify their actual image encoding."""
    from PIL import Image
    turns = payload.get('turns') or []
    if not turns:
        raise ValueError('Valid trajectory has no terminal turn/image evidence')
    terminal = max(turns,key=lambda t:(t.get('metadata') or {}).get('turn_index',-1))
    metadata = terminal.get('metadata') or {}
    multimodal = terminal.get('multimodal_inputs') or {}
    candidates = []
    for field in ('images','judge_images'):
        values = multimodal.get(field) or []
        if not isinstance(values,(list,tuple)):
            raise ValueError('Terminal multimodal image collection is malformed')
        candidates.extend((f'multimodal_inputs.{field}[{i}]',value) for i,value in enumerate(values))
    full = metadata.get('full_image_list') or []
    if full:
        if not isinstance(full,(list,tuple)):
            raise ValueError('Archived full_image_list is malformed')
        candidates.append(('metadata.full_image_list[-1]',full[-1]))
    if not candidates:
        raise ValueError('Valid trajectory lacks terminal image evidence')
    images = {}
    for origin,value in candidates:
        if isinstance(value,str):
            if value.startswith(('http://','https://')):
                raise ValueError('Image references alone are insufficient; no network fetch is permitted')
            encoded = value.split(',',1)[1] if value.startswith('data:') else value
            raw = base64.b64decode(''.join(encoded.split()),validate=True)
        elif isinstance(value,bytes):
            raw = value
        elif isinstance(value,Image.Image):
            stream = io.BytesIO();value.save(stream,format='PNG');raw = stream.getvalue()
        else:
            raise ValueError('Unsupported archived terminal image representation')
        identity = hashlib.sha256(raw).hexdigest()
        if identity not in images:
            with Image.open(io.BytesIO(raw)) as image:
                shape,format_name = image.size,image.format
                image.verify()
            if shape[0]<1 or shape[1]<1:
                raise ValueError('Terminal image has invalid dimensions')
            images[identity] = dict(sha256=identity,bytes=len(raw),width=shape[0],height=shape[1],
                                    format=format_name,origins=[])
        images[identity]['origins'].append(origin)
    return dict(verified=True,unique_images=len(images),images=list(images.values()),
        full_image_list_length=len(full),full_image_list_last_checked=bool(full),
        temporal_note='The last archived image is checked; post-action timing is not inferred from presence alone')


@contextmanager
def image_audit(collector, proofs):
    """Inspect images while the frozen verifier already holds each CPU archive.

    Only in-process call wrappers change; no source, archive or tensor is edited.
    This avoids a second full read of every potentially large trajectory archive.
    Tensor storage is mapped read-only for inspection: metadata/image checks do
    not need hundreds of MB of saved vision tensors resident in memory.
    """
    import torch
    original_verify = collector.verify_record
    def verify(record,task,attempt,*,deep=False):
        if not deep:
            return original_verify(record,task,attempt,deep=deep)
        expected = Path(record['artifact'])/(hashlib.sha256(task.encode()).hexdigest()+'.pt')
        original_load = torch.load
        def load(path,*args,**kwargs):
            kwargs.setdefault('mmap', True)
            payload = original_load(path,*args,**kwargs)
            if record['valid'] and Path(path).resolve()==expected.resolve():
                proofs[task,attempt] = verify_terminal_images(payload)
            return payload
        torch.load = load
        try:
            result = original_verify(record,task,attempt,deep=True)
            if record['valid'] and (task,attempt) not in proofs:
                raise ValueError('Frozen deep verifier did not expose terminal image evidence')
            return result
        finally:
            torch.load = original_load
    collector.verify_record = verify
    try:
        yield
    finally:
        collector.verify_record = original_verify


def ledger_audit(root):
    root = Path(root)
    ledger_path = root/'judge-budget/ledger.json'
    ledger = read(ledger_path) if ledger_path.exists() else dict(calls=0, charged_or_reserved_usd=0.)
    calls, charged = ledger['calls'], ledger['charged_or_reserved_usd']
    if type(calls) is not int or not 0 <= calls <= 4800 or not math.isfinite(charged) or not 0 <= charged <= 12.5:
        raise ValueError('Invalid or exceeded persistent judge budget')
    receipts = []
    for path in (root/'judge-budget').glob('*.json'):
        if path.stem.isdecimal():
            value = read(path)
            if value.get('status') not in ('reserved','received') or not math.isfinite(value['charged_usd']):
                raise ValueError('Malformed judge reservation receipt')
            receipts.append(value)
    itemized = sum(r['charged_usd'] for r in receipts)
    if len(receipts)>calls or itemized>charged+1e-7:
        raise ValueError('Judge receipts exceed the charged persistent ledger')
    return dict(calls_charged=calls, charged_or_reserved_usd=charged,
        received_calls=sum(r['status']=='received' for r in receipts),
        reserved_calls=sum(r['status']=='reserved' for r in receipts),
        unitemized_reserved_calls=calls-len(receipts), unitemized_reserved_usd=max(0.,charged-itemized),
        note='Reservations remain charged; this is not a provider invoice')


def physical_audit(root):
    root = Path(root)
    directory = root/'physical-attempts'
    paths = sorted(directory.iterdir()) if directory.exists() else []
    if len(paths)>1600:
        raise ValueError('Physical dispatch cap exceeded')
    report = dict(physical_dispatches_charged=len(paths), completed_receipts=0,
        interrupted_receipts=0, missing_dispatch_receipts=0, incomplete_physical_attempts=0,
        physical_attempts_with_saved_metering=0, actor_requests_known=0,
        actor_output_tokens_known=0, actor_work_unknown_attempts=0)
    for path in paths:
        report['completed_receipts'] += (path/'completed.json').is_file()
        report['interrupted_receipts'] += (path/'interrupted.json').is_file()
        report['incomplete_physical_attempts'] += not (path/'completed.json').is_file()
        if not (path/'dispatch.json').is_file():
            report['missing_dispatch_receipts'] += 1
            report['actor_work_unknown_attempts'] += 1
            continue
        dispatch = read(path/'dispatch.json')
        identity = hashlib.sha256(f'{dispatch["task_id"]}/screen/actor/{dispatch["attempt"]}'.encode()).hexdigest()
        record = Path(dispatch['output'])/'screen_records'/(identity+'.json')
        if record.exists():
            value = read(record)
            report['physical_attempts_with_saved_metering'] += 1
            report['actor_requests_known'] += value['actor_requests']
            report['actor_output_tokens_known'] += value['actor_output_tokens']
        else:
            report['actor_work_unknown_attempts'] += 1
    return report


def breakdown(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    result = {}
    for name, group in sorted(grouped.items()):
        targets = [r for r in group if r['extension_targeted']]
        rescues = sum(r['newly_rescued'] for r in targets)
        result[name] = dict(historical_tasks=len(group), historical_successes=sum(r['historical_pass5_success'] for r in group),
            extension_tasks=len(targets), rescued=rescues, rescue_rate=rescues/len(targets) if targets else None,
            combined_successes=sum(r['combined_pass8_success_observed'] for r in group),
            combined_rate=sum(r['combined_pass8_success_observed'] for r in group)/len(group))
    return result


def native_metadata(task_rows, native_path, reference_path, frozen_path):
    reference = read(reference_path)
    expected = reference['input_hashes']
    if (sha(native_path) != expected['native_tasks_sha256'] or
            sha(frozen_path) != expected['frozen_tasks_sha256']):
        raise ValueError('Recorded source/rubric covariates differ from the independently audited inputs')
    native = [json.loads(line) for line in Path(native_path).read_text().splitlines() if line.strip()]
    indexed = {'webvoyager/'+str(row['task_id']): row for row in native}
    ids = [str(row['metadata']['task_id']) for row in task_rows]
    if len(native)!=2000 or len(indexed)!=2000 or set(indexed)!=set(ids):
        raise ValueError('Native source/rubric covariates do not cover the exact historical cohort')
    for row in task_rows:
        frozen = row['metadata']; item = indexed[str(frozen['task_id'])]
        facts = sum(len(group['facts']) for group in item['evaluator_reference'])
        if (item['task_name']!=frozen['task'] or item['website']!=frozen['start_url'] or
                type(item['difficulty']) is not int or item['difficulty']!=facts or facts<5):
            raise ValueError('Native task identity or rubric-fact-count definition changed')
    return indexed, dict(native_tasks=str(native_path),native_tasks_sha256=sha(native_path),
        covariate_reference=str(reference_path),covariate_reference_sha256=sha(reference_path),
        definition='Recorded difficulty equals the number of evaluator-reference rubric facts on all 2,000 tasks')


def annotate_rows(joined, old, new, task_rows, native):
    prior, fresh = defaultdict(list), defaultdict(list)
    for record in old:
        prior[record['task_id']].append(record)
    for record in new:
        fresh[record['task_id']].append(record)
    metadata = {str(row['metadata']['task_id']): row['metadata'] for row in task_rows}
    output = []
    for row in joined['tasks']:
        task = row['task_id']
        before, after = prior[task], fresh[task]
        fact_count = native[task]['difficulty']
        output.append(dict(row, source=native[task]['benchmark_name'],
            rubric_fact_count=fact_count, rubric_fact_count_group=str(fact_count) if fact_count<8 else '8+',
            website=urlsplit(metadata[task]['start_url']).hostname or 'unknown',
            prior_valid_attempts=sum(r['valid'] for r in before),
            prior_format_failure_attempts=sum(r['valid'] and r['reward']==-1. for r in before),
            new_valid_attempts=sum(r['valid'] for r in after),
            new_invalid_attempts=sum(not r['valid'] for r in after),
            new_successful_attempts=sum(r['valid'] and r['reward']==1. for r in after),
            invalid_history_stratum='previously_successful' if row['historical_pass5_success'] else
                'five_valid_failures' if row['historical_five_valid_failures'] else 'zero_success_with_invalid_attempts',
            new_artifacts=[dict(attempt=r['attempt'], artifact=r['artifact']) for r in after]))
    return output


def audit(root, pointer=DEFAULT_POINTER, *, run=subprocess.run, collector_loader=frozen_collector,
          native_tasks=NATIVE_TASKS, covariate_reference=COVARIATE_REFERENCE):
    root = Path(root).resolve()
    plan, approval, current, configs, jobs = campaign_inputs(root, pointer)
    verify_source(plan)
    scheduler = scheduler_audit(approval, jobs, run=run)
    verifier_source, recovery_receipts = audit_collector_source(plan, approval)
    collector = collector_loader(verifier_source)
    old = collector.load_old_records(configs[0], plan['historical_task_order'])
    baseline = collector.summarize_joined(old, [], plan['historical_task_order'], verify_artifacts=False,
        expected_old=BASELINE, extension_ids=plan['extension_task_order'])
    if baseline['rescue_among_invalid_pass5_misses']['denominator'] != 25:
        raise ValueError('Historical invalid-miss stratum differs from the prespecified 25 tasks')
    shards, new, proofs, images = [], [], [], {}
    with cached_history(collector, old, plan['historical_task_order'], configs), image_audit(collector,images):
        for config in configs:
            summary, _ = collector.audit(config, require_complete=True, deep=True)
            directory = Path(config['output'])
            records = collector.load_records(directory/'records')
            receipt = read(directory/'collection-complete.json')
            for field in ('primary_attempts','expected_attempts','valid_attempts','invalid_attempts','successes'):
                if receipt.get(field) != summary[field]:
                    raise ValueError('Collection receipt disagrees with the independent deep audit')
            ledger, physical = ledger_audit(directory), physical_audit(directory)
            if summary['physical_dispatches_charged'] != physical['physical_dispatches_charged']:
                raise ValueError('Physical attempt accounting changed during audit')
            shards.append(dict(shard=config['shard'], summary=summary, judge=ledger, physical=physical,
                collection_receipt_sha256=sha(directory/'collection-complete.json')))
            for record in records:
                path = directory/'records'/(collector.key(record['task_id'],record['attempt'])+'.json')
                archive = Path(record['artifact'])/(hashlib.sha256(record['task_id'].encode()).hexdigest()+'.pt')
                proofs.append(dict(task_id=record['task_id'], attempt=record['attempt'], record=str(path),
                    record_sha256=sha(path), sidecar_sha256=sha(archive.with_suffix('.json')),
                    archive=str(archive), archive_size_bytes=archive.stat().st_size,
                    archive_deep_validation=True,
                    terminal_images=images.get((record['task_id'],record['attempt']),
                        dict(verified=False,reason='invalid trajectory; no complete terminal image required'))))
            new.extend(records)
    if set(images)!={(r['task_id'],r['attempt']) for r in new if r['valid']}:
        raise ValueError('Every valid new trajectory requires decoded and verified terminal images')
    joined = collector.summarize_joined(old, new, plan['historical_task_order'], require_complete=True,
        verify_artifacts=False, expected_old=BASELINE, extension_ids=plan['extension_task_order'])
    if len(new)!=2121 or joined['expected_new_primary_attempts']!=2121:
        raise ValueError('All 2,121 new primary attempts are required')
    task_path = Path(plan['source'])/'screen-tasks.jsonl'
    task_rows = [json.loads(line) for line in task_path.read_text().splitlines() if line.strip()]
    if [str(row['metadata']['task_id']) for row in task_rows] != plan['historical_task_order']:
        raise ValueError('Full historical task metadata/order changed')
    native, covariate_proof = native_metadata(task_rows,native_tasks,covariate_reference,task_path)
    rows = annotate_rows(joined, old, new, task_rows,native)
    judge_cost = sum(s['judge']['charged_or_reserved_usd'] for s in shards)
    judge_calls = sum(s['judge']['calls_charged'] for s in shards)
    browsers = sum(s['physical']['physical_dispatches_charged'] for s in shards)
    if judge_cost>50.+1e-7 or judge_calls>19200 or browsers>6400:
        raise ValueError('Campaign totals exceed the original independent-shard caps')
    aggregate = dict(audit_complete=True, verified_complete=False, tasks=2000,
        old_successes=1293, historical_pass5_rate=1293/2000,
        new_primary_attempts=2121, targeted_historical_misses=707,
        rescued_tasks=joined['rescue_among_pass5_misses']['successes_observed'],
        combined_successes=joined['combined_pass8_successes_observed'],
        combined_pass8_rate=joined['combined_pass8_rate'], pass8_minus_pass5=joined['pass8_minus_pass5'],
        rescues_all_misses=joined['rescue_among_pass5_misses'],
        rescues_five_valid_failures=joined['rescue_among_five_valid_failures'],
        rescues_invalid_misses=joined['rescue_among_invalid_pass5_misses'],
        historical_invalid_episodes=sum(not r['valid'] for r in old),
        new_invalid_episodes=sum(not r['valid'] for r in new),
        new_tasks_with_invalid_episodes=sum(r['new_invalid_attempts']>0 for r in rows),
        new_native_format_failure_episodes=sum(r['valid'] and r['reward']==-1. for r in new),
        valid_new_trajectories_with_verified_terminal_images=len(images),
        cost=dict(h200_gpu_hours=scheduler['h200_gpu_hours'], physical_dispatches_charged=browsers,
            judge_http_calls_charged=judge_calls, judge_charged_or_reserved_usd=judge_cost,
            actor_requests_known=sum(s['physical']['actor_requests_known'] for s in shards),
            actor_output_tokens_known=sum(s['physical']['actor_output_tokens_known'] for s in shards),
            actor_work_unknown_physical_attempts=sum(s['physical']['actor_work_unknown_attempts'] for s in shards)),
        by_source=breakdown(rows,'source'), by_website=breakdown(rows,'website'),
        by_invalid_history_stratum=breakdown(rows,'invalid_history_stratum'),
        by_rubric_fact_count=breakdown(rows,'rubric_fact_count'),
        by_rubric_fact_count_group=breakdown(rows,'rubric_fact_count_group'),
        by_prior_valid_attempt_count=breakdown(rows,'prior_valid_attempts'),
        by_prior_format_failure_count=breakdown(rows,'prior_format_failure_attempts'),
        difficulty_label_note='Recorded difficulty is rubric-fact count, not intrinsic interaction difficulty. '
            'This selected pool has at least five facts per task; invalid history is a separate reliability stratum.',
        fresh_pass3_on_all_tasks_measured=False, limitations=joined['limitations'],
        pending_owner_verification=['W&B final run reconciliation','independent image/trajectory review'],
        optimizer_updates=0, protocol=plan['protocol'])
    for key in ('rescues_all_misses','rescues_five_valid_failures','rescues_invalid_misses'):
        aggregate[key] = {k:v for k,v in aggregate[key].items() if k!='task_ids'}
    if read(root/'plan.json')!=plan or read(root/'approval.json')!=approval or read(pointer)!=current:
        raise ValueError('Campaign inputs changed during final audit')
    return dict(audit_complete=True, verified_complete=False, audited_unix=time.time(), root=str(root),
        plan_sha256=digest(plan), approval_sha256=digest(approval), source=verifier_source,
        campaign_source=plan['source'], registered_recovery_receipts=recovery_receipts,
        collector_sha256=sha(Path(verifier_source)/'openwebrl/arm_task_pass8.py'),
        historical_records_verified=10000, new_records_deep_verified=2121,
        valid_new_terminal_images_verified=len(images),
        scheduler=scheduler, shards=shards, covariate_proof=covariate_proof,
        aggregate=aggregate, task_rows=rows, new_artifact_proofs=proofs)


def save_private(root, report):
    directory = Path(root)/'private-review'/f'final-audit-{time.time_ns()}'
    directory.mkdir(parents=True, exist_ok=False)
    paths = {}
    for name, value in [('audit.json',{k:v for k,v in report.items() if k not in ('task_rows','new_artifact_proofs')}),
                        ('aggregate.json',report['aggregate']), ('task-rows.json',report['task_rows']),
                        ('new-artifact-proofs.json',report['new_artifact_proofs'])]:
        path = directory/name
        with path.open('x') as handle:
            json.dump(value,handle,indent=2,allow_nan=False)
            handle.write('\n');handle.flush();os.fsync(handle.fileno())
        paths[name] = dict(path=str(path),sha256=sha(path))
    return dict(audit_complete=True, verified_complete=False, artifacts=paths,
                note='Owner W&B/image proof is still required; completion pointer was not modified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pointer', type=Path, default=DEFAULT_POINTER)
    parser.add_argument('--native-tasks', type=Path, default=NATIVE_TASKS)
    parser.add_argument('--covariate-reference', type=Path, default=COVARIATE_REFERENCE)
    args = parser.parse_args()
    print(json.dumps(save_private(args.root,audit(args.root,args.pointer,
        native_tasks=args.native_tasks,covariate_reference=args.covariate_reference)),indent=2))


if __name__=='__main__':
    main()
