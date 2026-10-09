"""Bounded serving-only recovery; no browser or external-provider entry points.

Preparation reads saved evidence. Requesting a drain is an explicit administrative
operation: already claimed episodes finish, while the old worker's phase lookup
finds no new work. The existing controller then tears down its owned processes.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlparse

from openwebrl.controlled_sft_state import digest
from openwebrl.controlled_sft_worker import SAMPLING

DRAIN_PHASE = 'serving_drain'
PURPOSE = 'serving_cap_saved_context_acceptance'
MIN_PROMPT_TOKENS = 16384
REQUEST_COUNT = 10
INFERENCE_TIMEOUT = 180


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    return {'path': str(path.resolve()), 'sha256': sha(path)}


def checked_artifact(receipt):
    path = Path(receipt['path'])
    if sha(path) != receipt['sha256']:
        raise ValueError('Saved serving recovery artifact changed')
    return read(path)


def recovery_config(plan):
    config = plan.get('serving_recovery')
    if config is None:
        return None
    if (config.get('previous_max_running_requests') != 5
            or config.get('max_running_requests') != 10
            or config.get('sampling') != SAMPLING
            or config.get('scientific_config_sha256') != plan['scientific_config_sha256']
            or config.get('actor_checkpoint') != plan['worker_config']['actor_checkpoint']
            or not isinstance(config.get('prior_plan_sha256'), str)):
        raise ValueError('Only the pinned 5 to10 serving-cap recovery is supported')
    prior = checked_artifact(config['prior_plan'])
    if digest(prior) != config['prior_plan_sha256']:
        raise ValueError('Preserved prior plan identity changed')
    for key in ('root', 'resources', 'limits', 'conditions', 'schedule', 'task_file', 'task_file_sha256',
                'scientific_config_sha256', 'worker_config', 'protocol_name', 'campaign_id', 'shard',
                'campaign_task_shards', 'primary_episodes', 'smoke_episodes'):
        if plan.get(key) != prior.get(key):
            raise ValueError('Serving recovery changed a scientific, resource or task-queue field: '+key)
    manifest = checked_artifact(config['acceptance_manifest'])
    if (manifest.get('source_plan_sha256') != config['prior_plan_sha256']
            or manifest.get('root') != plan['root']
            or manifest.get('actor_checkpoint') != config['actor_checkpoint']
            or manifest.get('sampling') != SAMPLING
            or len(manifest.get('requests', [])) != REQUEST_COUNT):
        raise ValueError('Saved-context acceptance belongs to a different frozen run')
    return config


def server_max_running_requests(plan):
    return 10 if recovery_config(plan) is not None else 5


def request_drain(root, *, expected_job_id, receipt_path):
    """Explicit mutation, never invoked during preparation or dry-run."""
    root = Path(root).resolve()
    plan, approval = read(root/'plan.json'), read(root/'approval.json')
    current = read(root/'current.json')
    if (approval.get('approved') is not True or approval['plan_sha256'] != digest(plan)
            or str(current.get('job_id')) != str(expected_job_id)
            or current.get('stage') != 'primary'
            or not any(str(a['job_id']) == str(expected_job_id) for a in approval['attempts'])):
        raise ValueError('Drain requires the current approved primary allocation')
    db = sqlite3.connect(root/'state.sqlite3', timeout=60, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        db.execute('BEGIN IMMEDIATE')
        if db.execute("SELECT 1 FROM metadata WHERE key='halt'").fetchone():
            raise ValueError('A provider/budget halt requires separate diagnosis')
        phase = db.execute("SELECT value FROM metadata WHERE key='phase'").fetchone()['value']
        if phase != 'primary' or db.execute('SELECT 1 FROM queue WHERE category=?', (DRAIN_PHASE,)).fetchone():
            raise ValueError('Only an active primary phase can enter the empty drain phase')
        active = [dict(x) for x in db.execute("SELECT attempt_id,job_id FROM attempts WHERE status='active'")]
        if any(str(x['job_id']) != str(expected_job_id) for x in active):
            raise ValueError('Another allocation owns an active episode')
        pending = [x['item_id'] for x in db.execute("SELECT item_id FROM queue WHERE status='pending' ORDER BY position")]
        counters = dict(db.execute('SELECT key,value FROM counters'))
        receipt = dict(job_id=str(expected_job_id), plan_sha256=digest(plan), previous_phase=phase,
            new_phase=DRAIN_PHASE, active_attempts=active, pending_item_ids=pending,
            counters_before=counters, requested_unix=time.time(),
            expected_controller_exit='Incomplete queue causes non-success exit after orderly owned teardown; not collection completion')
        immutable_json(receipt_path, receipt)
        db.execute("UPDATE metadata SET value=? WHERE key='phase'", (DRAIN_PHASE,))
        db.execute('INSERT INTO events(event,details_json,created) VALUES(?,?,?)',
            ('serving_drain_requested', json.dumps(receipt), time.time()))
        db.commit()
        return receipt
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def assert_drained(root, receipt):
    """Read-only: no claim/result/counter is rewritten or retried."""
    root = Path(root)
    with sqlite3.connect(f'file:{root}/state.sqlite3?mode=ro', uri=True) as db:
        if db.execute("SELECT 1 FROM metadata WHERE key='halt'").fetchone():
            raise ValueError('An independent provider/budget halt occurred during drain')
        if db.execute("SELECT value FROM metadata WHERE key='phase'").fetchone()[0] != DRAIN_PHASE:
            raise ValueError('Drain phase changed')
        if db.execute("SELECT 1 FROM attempts WHERE status='active' OR (status='interrupted' AND job_id=?)", (receipt['job_id'],)).fetchone():
            raise ValueError('An episode has not finished normally')
        if db.execute("SELECT 1 FROM reservations WHERE status='reserved'").fetchone():
            raise ValueError('A browser or priced request is unsettled')
        for old in receipt['active_attempts']:
            result = db.execute('SELECT status,result_json FROM attempts WHERE attempt_id=?', (old['attempt_id'],)).fetchone()
            if result is None or result[0] != 'finished' or not result[1]:
                raise ValueError('An originally active episode did not preserve its normal finished result')
            descriptor = json.loads(result[1])
            if (descriptor.get('browser_closed') is not True
                    or sha(descriptor['result_path']) != descriptor['result_sha256']):
                raise ValueError('A drained result or its browser-closure proof changed')
        counters = dict(db.execute('SELECT key,value FROM counters'))
        if counters.get('active_browsers', 0):
            raise ValueError('A browser remains active')
        pending = [x[0] for x in db.execute("SELECT item_id FROM queue WHERE status='pending' ORDER BY position")]
        if pending != receipt['pending_item_ids']:
            raise ValueError('Pending task identities or ordering changed while draining')
        for key, before in receipt['counters_before'].items():
            if key != 'active_browsers' and key not in ('luna_usd', 'canonical_judge_usd') and counters.get(key, 0) < before:
                raise ValueError('An all-attempt counter was reset')
        return dict(drained=True, pending_count=len(pending), counters=counters)


def prepare_acceptance_manifest(root, output):
    """Copy ten distinct, hash-matched, already executed long-context requests."""
    root, output = Path(root).resolve(), Path(output).resolve()
    plan = read(root/'plan.json')
    with sqlite3.connect(f'file:{root}/state.sqlite3?mode=ro', uri=True) as db:
        rows = db.execute("SELECT attempt_id,artifact_directory,result_json FROM attempts WHERE status='finished'").fetchall()
    candidates = []
    for attempt_id, directory, result_text in rows:
        directory = Path(directory)
        batches = {}
        for path in (directory/'events').glob('[0-9]*.json'):
            with path.open() as handle:
                if '"proposal_batch_reserved"' not in handle.read(150):
                    continue
            event = read(path)['record']
            if MIN_PROMPT_TOKENS <= event['prompt_tokens'] < 32768-4096:
                batches[event['prompt_sha256']] = (event, path)
        if not batches:
            continue
        result = json.loads(result_text)
        if sha(result['result_path']) != result['result_sha256']:
            raise ValueError('Original committed result changed')
        for rollout in (directory/'rollouts').rglob('*.json'):
            saved = read(rollout)
            prompt = saved.get('llm_input_texts')
            if not isinstance(prompt, str):
                continue
            match = batches.get(hashlib.sha256(prompt.encode()).hexdigest())
            if match is None:
                continue
            batch, event_path = match
            images = []
            for group in saved.get('images', []):
                for image in group:
                    if isinstance(image, dict):
                        image = image.get('url')
                    if not isinstance(image, str):
                        continue
                    raw = base64.b64decode(image.split(',', 1)[-1], validate=True)
                    if hashlib.sha256(raw).hexdigest() == batch['screenshot_sha256']:
                        images.append(image)
            if not images or prompt.count('<|image_pad|>') != 1:
                continue
            request = dict(text=prompt, image_data=[images[0]], return_logprob=True,
                sampling_params=dict(SAMPLING, sampling_seed=batch['candidate_seeds'][0]))
            candidates.append(dict(attempt_id=attempt_id, prompt_tokens=batch['prompt_tokens'], request=request,
                source_rollout=dict(path=str(rollout), sha256=sha(rollout)),
                source_batch=dict(path=str(event_path), sha256=sha(event_path)),
                source_result=dict(path=result['result_path'], sha256=result['result_sha256'])))
    selected = []
    for candidate in sorted(candidates, key=lambda x: x['prompt_tokens'], reverse=True):
        if candidate['attempt_id'] in {x['attempt_id'] for x in selected}:
            continue
        selected.append(candidate)
        if len(selected) == REQUEST_COUNT:
            break
    if len(selected) != REQUEST_COUNT:
        raise ValueError('Ten distinct saved long-context prompts are required before recovery')
    requests = []
    for index, row in enumerate(selected):
        request = row.pop('request')
        requests.append(dict(row, request=immutable_json(output/f'request-{index:02d}.json', request)))
    manifest = dict(root=str(root), source_plan_sha256=digest(plan), actor_checkpoint=plan['worker_config']['actor_checkpoint'],
        scientific_config_sha256=plan['scientific_config_sha256'], sampling=SAMPLING,
        purpose=PURPOSE, request_count=REQUEST_COUNT, minimum_prompt_tokens=MIN_PROMPT_TOKENS, requests=requests)
    return immutable_json(output/'manifest.json', manifest)


def reuse_jev_diagnostic(plan, journal, job):
    config = recovery_config(plan)
    if config is None:
        raise ValueError('Only an explicit serving recovery may reuse diagnostic evidence')
    old = checked_artifact(config['prior_jev_diagnostic'])
    if (old.get('completed') is not True or old.get('plan_sha256') != config['prior_plan_sha256']
            or old.get('expected_model') != 'jev-1.13.0'
            or old.get('outcome') not in ('accepted', 'known_capacity_rejection')):
        raise ValueError('Prior Jev diagnostic does not belong to the preserved plan')
    if old['outcome'] == 'known_capacity_rejection' and (old.get('http_status') != 400 or old.get('error_type') != 'max_tokens_exceeded'):
        raise ValueError('Only the exact known Jev capacity rejection is reusable')
    if old['outcome'] == 'accepted' and old.get('model_identity') != 'jev-1.13.0':
        raise ValueError('Prior accepted Jev identity differs')
    receipt = dict(old, plan_sha256=digest(plan), job_id=str(job), external_calls_made=0,
                   reused_from=config['prior_jev_diagnostic'])
    return immutable_json(Path(journal)/'jev-context-preflight.json', receipt)


async def run_saved_context_acceptance(plan, state, actor_base_url, journal, client):
    """Ten concurrent local calls, charged before dispatch; execute no actions."""
    parsed = urlparse(actor_base_url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.path not in ('', '/'):
        raise ValueError('Serving acceptance can contact only its local actor')
    config = recovery_config(plan)
    if config is None:
        raise ValueError('Missing frozen serving recovery')
    response = await client.get(actor_base_url.rstrip('/')+'/get_server_info', timeout=10)
    response.raise_for_status()
    server = response.json()
    expected_server = dict(model_path=plan['worker_config']['actor_checkpoint'], max_running_requests=10,
        context_length=32768, mem_fraction_static=.60, disable_cuda_graph=True, dtype='bfloat16', tp_size=1)
    if any(server.get(key) != value for key, value in expected_server.items()):
        raise ValueError('Effective local actor serving settings differ from the frozen recovery')
    immutable_json(Path(journal)/'serving-acceptance-server-info.json', server)
    manifest = checked_artifact(config['acceptance_manifest'])
    requests = []
    for entry in manifest['requests']:
        request = checked_artifact(entry['request'])
        batch = checked_artifact(entry['source_batch'])['record']
        checked_artifact(entry['source_rollout'])
        checked_artifact(entry['source_result'])
        expected = dict(SAMPLING, sampling_seed=batch['candidate_seeds'][0])
        image = base64.b64decode(request['image_data'][0].split(',', 1)[-1], validate=True)
        if (request['sampling_params'] != expected or request.get('return_logprob') is not True
                or hashlib.sha256(request['text'].encode()).hexdigest() != batch['prompt_sha256']
                or len(request['image_data']) != 1 or hashlib.sha256(image).hexdigest() != batch['screenshot_sha256']
                or entry['prompt_tokens'] != batch['prompt_tokens']
                or not MIN_PROMPT_TOKENS <= entry['prompt_tokens'] < 32768-4096):
            raise ValueError('Acceptance request changed scientific settings, context, seed or image')
        requests.append(request)
    reservation = state.reserve('local_sft_generations', REQUEST_COUNT, purpose=PURPOSE,
        request={'acceptance_manifest_sha256': config['acceptance_manifest']['sha256']})
    directory = Path(journal)/'serving-acceptance'
    immutable_json(directory/'reservation.json', reservation)
    started = time.monotonic()
    async def one(index, request):
        begin = time.monotonic()
        try:
            async with asyncio.timeout(INFERENCE_TIMEOUT):
                response = await client.post(actor_base_url.rstrip('/')+'/generate', json=request, timeout=180)
                raw = response.json()
            receipt = dict(index=index, elapsed_seconds=time.monotonic()-begin,
                http_status=response.status_code, request_sha256=manifest['requests'][index]['request']['sha256'], response=raw)
            immutable_json(directory/f'response-{index:02d}.json', receipt)
            response.raise_for_status()
            meta = raw['meta_info']
            if (meta.get('prompt_tokens') != manifest['requests'][index]['prompt_tokens']
                    or type(meta.get('completion_tokens')) is not int or not 0 < meta['completion_tokens'] <= 4096
                    or meta.get('finish_reason', {}).get('type') not in ('stop', 'length')
                    or meta.get('total_retractions', 0) != 0 or receipt['elapsed_seconds'] >= INFERENCE_TIMEOUT):
                raise ValueError('Serving acceptance failed token, timeout, finish or retraction boundary')
            return receipt
        except BaseException as exc:
            immutable_json(directory/f'error-{index:02d}.json', dict(error_type=type(exc).__name__, error=str(exc)[:2048]))
            raise
    results = await asyncio.gather(*(one(i, request) for i, request in enumerate(requests)), return_exceptions=True)
    failures = [type(result).__name__ for result in results if isinstance(result, BaseException)]
    receipt = dict(completed=not failures, purpose=PURPOSE, plan_sha256=digest(plan),
        actor_checkpoint=manifest['actor_checkpoint'], max_running_requests=10,
        requests=REQUEST_COUNT, charged_local_sft_generations=REQUEST_COUNT,
        reservation=reservation, elapsed_seconds=time.monotonic()-started, failures=failures,
        prompt_tokens=[x['prompt_tokens'] for x in manifest['requests']],
        browser_sessions=0, external_api_calls=0, executed_actions=0,
        acceptance_manifest=config['acceptance_manifest'])
    immutable_json(Path(journal)/'serving-acceptance.json', receipt)
    if failures:
        raise RuntimeError('Saved long-context serving acceptance failed; primary remains stopped')
    return receipt
