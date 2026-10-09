"""Append three ordinary SFT trajectories to the 707 historical pass@5 misses.

This is collection only. The original records are read-only inputs, and the
ordered five-plus-three union is not an exchangeable, contemporaneous sample.
Run this module from the pinned original screening source with this isolated
extension added; it deliberately reuses that source's trajectory and judge.
"""
import argparse
import asyncio
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

from openwebrl.arm_rescue_yield import stable_seed, trajectory, write_json
from openwebrl.arm_task_screen import JUDGE, POLICY, key, verify_record as verify_screen_record

SHARDS = 4
EXTENSION_TASKS = 707
NEW_ATTEMPTS = (5, 6, 7)
SEED = 20260930
BROWSER_CAP_PER_SHARD = 1600
JUDGE_CALLS_PER_SHARD = 4800
JUDGE_USD_PER_SHARD = 12.5
DECODING = dict(temperature=.8, top_p=1., top_k=-1, max_new_tokens=1024)
LIMITATION = ("Ordered historical-five plus three new attempts only on its 707 zero-success tasks. "
              "The 1,293 prior successes are preserved without new attempts; overall coverage uses all 2,000 tasks. "
              "Collection dates differ; website and provider drift remain possible. "
              "This does not measure fresh pass@3 on all tasks, complete eight-attempt groups, "
              "or an exchangeable random-eight/subset estimator. Invalid attempts remain distinct from valid failures.")


def partition(ids, num_shards=SHARDS):
    if len(ids) != EXTENSION_TASKS or len(set(ids)) != EXTENSION_TASKS:
        raise ValueError('Expected the exact 707 unique historical pass@5 misses')
    if type(num_shards) is not int or num_shards != SHARDS:
        raise ValueError('The prepared extension has exactly four separately capped shards')
    return [ids[i::num_shards] for i in range(num_shards)]


def validate_partition(shards, ids):
    if shards != partition(ids):
        raise ValueError('Changed, missing or overlapping task shards')


def seed_for(task, attempt):
    return stable_seed(SEED, POLICY, task, 'screen', 'actor', attempt)


def _read(path):
    return json.loads(Path(path).read_text())


def _sync_dir(path):
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _new_json(path, value):
    """Create an immutable receipt. Existing evidence is never overwritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    _sync_dir(path.parent)


def _archive(record):
    return Path(record['artifact'])/(hashlib.sha256(record['task_id'].encode()).hexdigest()+'.pt')


def _native_empty_invalid_stub(record, turns):
    """Require one untouched native aborted input, never a generated turn."""
    if record['valid'] or record.get('reward') is not None or len(turns) != 1:
        return False
    turn = turns[0]
    meta = turn.get('metadata') or {}
    inputs = turn.get('multimodal_inputs')
    empty_inputs = inputs is None or (isinstance(inputs, dict) and
        set(inputs) <= {'images', 'videos'} and
        all(value is None or value == [] for value in inputs.values()))
    return ('turn_index' not in meta and meta.get('task_id') == record['task_id'] and
            meta.get('is_last_turn') is True and type(meta.get('total_steps')) is int and
            meta['total_steps'] == 0 and meta.get('num_turns_in_trajectory', 1) == 1 and
            not meta.get('reward') and turn.get('status') == 'aborted' and
            turn.get('remove_sample') is True and turn.get('reward') is None and
            turn.get('response') == '' and turn.get('tokens') == [] and
            type(turn.get('response_length')) is int and turn['response_length'] == 0 and
            turn.get('session_id') is None and turn.get('multimodal_train_inputs') is None and
            empty_inputs)


def _native_timeout_stub(record, turns):
    """Recognize the untouched input returned by native rollout_task_timeout.

    The outer timeout can discard turns after actor calls. Preserve the invalid
    attempt without inventing a turn index or changing its reward.
    """
    if not _native_empty_invalid_stub(record, turns):
        return False
    reason = turns[0]['metadata'].get('terminate_reason', '')
    prefix = 'generation_error: rollout_task_timeout after '
    if not isinstance(reason, str) or not reason.startswith(prefix) or not reason.endswith('s'):
        return False
    try:
        seconds = float(reason[len(prefix):-1])
    except ValueError:
        return False
    return math.isfinite(seconds) and seconds > 0


def _native_local_health_stub(record, turns):
    """Recognize native local-browser startup failure before any actor request.

    local_process_env raises this exact readiness-error format; the native
    no-turn exception handler returns the original aborted input and its removal
    classifier marks 'not healthy within' as infrastructure-invalid.
    """
    if not _native_empty_invalid_stub(record, turns):
        return False
    if any(type(record.get(k)) is not int or record[k] != 0
           for k in ('actor_requests', 'actor_output_tokens')):
        return False
    meta = turns[0]['metadata']
    if type(meta.get('num_turns_in_trajectory')) is not int or meta['num_turns_in_trajectory'] != 1:
        return False
    reason = meta.get('terminate_reason', '')
    if not isinstance(reason, str):
        return False
    match = re.fullmatch(
        r'generation_error: local_process env_server at http://(?:127\.0\.0\.1|localhost):([0-9]{1,5}) '
        r'was not healthy within ([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)s; '
        r'attempts=([0-9]+); last_error=(?:[A-Za-z_][A-Za-z0-9_]*: .+|none)', reason)
    if not match:
        return False
    port, seconds, attempts = int(match[1]), float(match[2]), int(match[3])
    return 0 < port <= 65535 and math.isfinite(seconds) and seconds > 0 and attempts > 0


def verify_record(record, task, attempt, *, deep=False):
    """Check identities and native terminal outcomes; optionally load the archive."""
    if type(attempt) is not int or attempt not in range(8):
        raise ValueError('Unsupported attempt index')
    if (record.get('seed') != seed_for(task, attempt) or
            type(record.get('valid')) is not bool or
            record.get('selector_fallback_turns') != 0):
        raise ValueError('Changed seed, validity or ordinary-actor protocol')
    verify_screen_record(record, task, attempt)
    archive = _archive(record)
    sidecar = _read(archive.with_suffix('.json'))
    if (sidecar.get('rollout_file') != archive.name or
            sidecar.get('turns') != record.get('steps') or
            sidecar.get('error_type') != record.get('error_type')):
        raise ValueError('Archive sidecar does not match the attempt')
    if record['valid']:
        if record.get('error_type') is not None or record['steps'] < 1:
            raise ValueError('Valid attempt lacks a terminal trajectory')
    elif record.get('reward') is not None:
        raise ValueError('Invalid attempt must not report a valid task reward')
    if record['valid'] and record['reward'] == 1.:
        if (sidecar.get('terminal_status') != 'completed' or
                sidecar['reward_metadata'].get('judge') != 1. or
                not sidecar['reward_metadata'].get('judge_text')):
            raise ValueError('Success lacks a completed, positive native verdict')
    if deep:
        import torch
        # Only trusted local archives produced by the frozen collector are read.
        payload = torch.load(archive, map_location='cpu', weights_only=False)
        turns = payload.get('turns')
        if (payload.get('task_id') != task or not isinstance(turns, list) or
                len(turns) != record['steps'] or
                payload.get('error_type') != record.get('error_type')):
            raise ValueError('Trajectory payload identity/count mismatch')
        native_stub = _native_timeout_stub(record, turns) or _native_local_health_stub(record, turns)
        indices = [(t.get('metadata') or {}).get('turn_index') for t in turns]
        if not native_stub and (any(type(i) is not int for i in indices) or
                                len(set(indices)) != len(indices)):
            raise ValueError('Invalid or repeated archived turn index')
        if turns:
            terminal = turns[0] if native_stub else max(turns, key=lambda t: t['metadata']['turn_index'])
            if (terminal.get('status') != sidecar.get('terminal_status') or
                    terminal['metadata'].get('reward', {}) != sidecar['reward_metadata']):
                raise ValueError('Terminal archive and sidecar disagree')
            if record['valid']:
                if (any(t.get('remove_sample') or t.get('status') == 'aborted' for t in turns) or
                        terminal.get('reward') != record['reward'] or
                        not terminal['metadata'].get('is_last_turn')):
                    raise ValueError('Valid record disagrees with archived terminal state')
                if record['reward'] == 1. and not (
                        (terminal.get('multimodal_inputs') or {}).get('images') or
                        (terminal.get('multimodal_inputs') or {}).get('judge_images') or
                        terminal['metadata'].get('full_image_list')):
                    raise ValueError('Successful trajectory lacks archived image evidence')
        elif record['valid']:
            raise ValueError('Empty trajectory cannot be valid')
    return record


def _index(records, ids, attempts, *, verify_artifacts=True, deep=False):
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate task IDs')
    allowed = {(task, attempt) for task in ids for attempt in attempts}
    result = {}
    for record in records:
        ident = (record['task_id'], record['attempt'])
        if ident not in allowed or ident in result:
            raise ValueError('Unexpected or duplicate task attempt')
        if verify_artifacts:
            verify_record(record, *ident, deep=deep)
        result[ident] = record
    return result, allowed


def summarize(records, ids, require_complete=False, *, verify_artifacts=True, deep=False):
    indexed, allowed = _index(records, ids, NEW_ATTEMPTS,
                              verify_artifacts=verify_artifacts, deep=deep)
    complete = set(indexed) == allowed
    if require_complete and not complete:
        raise ValueError('Every task needs exactly three new primary attempts')
    dispositions = {}
    for task in ids:
        group = [indexed.get((task, a)) for a in NEW_ATTEMPTS]
        if any(r is None for r in group):
            status = 'pending'
        elif any(_win(r) for r in group):
            status = 'success'
        elif not all(r['valid'] for r in group):
            status = 'unresolved_invalid'
        else:
            status = 'all_valid_failure'
        dispositions[task] = status
    valid = sum(r['valid'] for r in records)
    successes = sum(_win(r) for r in records)
    return dict(complete=complete, tasks=len(ids), primary_attempts=len(records),
                expected_attempts=len(allowed), valid_attempts=valid,
                invalid_attempts=len(records)-valid, successes=successes,
                native_format_failure_attempts=sum(r['valid'] and r['reward'] == -1. for r in records),
                overall=successes/len(records) if records else None,
                valid_only=successes/valid if valid else None,
                task_counts=dict(Counter(dispositions.values())), dispositions=dispositions,
                optimizer_updates=0, policy_id=POLICY, judge_id=JUDGE,
                attempt_indices=list(NEW_ATTEMPTS))


def _win(record):
    return bool(record['valid'] and record['reward'] == 1.)


def summarize_joined(old_records, new_records, ids, *, require_complete=False,
                     verify_artifacts=True, expected_old=None, extension_ids=None):
    old, old_allowed = _index(old_records, ids, range(5), verify_artifacts=verify_artifacts)
    if set(old) != old_allowed:
        raise ValueError('Historical five-attempt cohort is incomplete')
    misses = [task for task in ids if not any(_win(old[task, a]) for a in range(5))]
    if extension_ids is not None and extension_ids != misses:
        raise ValueError('Extension targets must be every historical pass@5 miss in frozen order')
    extension_ids = misses
    new, new_allowed = _index(new_records, extension_ids, NEW_ATTEMPTS, verify_artifacts=verify_artifacts)
    complete = set(new) == new_allowed
    if require_complete and not complete:
        raise ValueError('Fresh three-attempt cohort is incomplete')
    old_success, old_missing, old_valid_failure, old_invalid = [], [], [], []
    fresh_success, rescued_missing, rescued_valid, rescued_invalid, joined_success = [], [], [], [], []
    rows = []
    for task in ids:
        before = [old[task, a] for a in range(5)]
        after = [new[task, a] for a in NEW_ATTEMPTS if (task, a) in new]
        won5 = any(_win(r) for r in before)
        won3 = any(_win(r) for r in after)
        invalid5 = any(not r['valid'] for r in before)
        valid_failure5 = not won5 and not invalid5
        (old_success if won5 else old_missing).append(task)
        if invalid5:
            old_invalid.append(task)
        if valid_failure5:
            old_valid_failure.append(task)
        if won3:
            fresh_success.append(task)
        if won3 and not won5:
            rescued_missing.append(task)
        if won3 and valid_failure5:
            rescued_valid.append(task)
        if won3 and not won5 and invalid5:
            rescued_invalid.append(task)
        if won5 or won3:
            joined_success.append(task)
        rows.append(dict(task_id=task, historical_pass5_success=won5,
                         historical_any_invalid=invalid5,
                         historical_five_valid_failures=valid_failure5,
                         extension_targeted=task in extension_ids,
                         fresh_attempts=len(after), fresh_any_invalid=any(not r['valid'] for r in after),
                         fresh_pass3_success=won3, combined_pass8_success_observed=won5 or won3,
                         newly_rescued=won3 and not won5))
    baseline = dict(tasks=len(ids), successes=len(old_success), misses=len(old_missing),
                    five_valid_failure_tasks=len(old_valid_failure), any_invalid_tasks=len(old_invalid))
    if expected_old is not None and baseline != expected_old:
        raise ValueError('Historical baseline differs from the frozen audited counts')
    total = len(ids)
    return dict(complete=complete, historical=baseline,
                historical_pass5_rate=len(old_success)/total if total else None,
                combined_pass8_successes_observed=len(joined_success),
                combined_pass8_rate=len(joined_success)/total if complete and total else None,
                pass8_minus_pass5=(len(joined_success)-len(old_success))/total if complete and total else None,
                additional_three_successes_observed=len(fresh_success),
                additional_three_rate_among_historical_misses=(len(fresh_success)/len(extension_ids)
                    if complete and extension_ids else None),
                fresh_pass3_on_all_tasks_measured=False,
                rescue_among_pass5_misses=dict(denominator=len(old_missing), successes_observed=len(rescued_missing),
                    rate=len(rescued_missing)/len(old_missing) if complete and old_missing else None,
                    task_ids=rescued_missing),
                rescue_among_five_valid_failures=dict(denominator=len(old_valid_failure),
                    successes_observed=len(rescued_valid),
                    rate=len(rescued_valid)/len(old_valid_failure) if complete and old_valid_failure else None,
                    task_ids=rescued_valid),
                rescue_among_invalid_pass5_misses=dict(denominator=len(old_missing)-len(old_valid_failure),
                    successes_observed=len(rescued_invalid),
                    rate=len(rescued_invalid)/(len(old_missing)-len(old_valid_failure))
                        if complete and len(old_missing)>len(old_valid_failure) else None,
                    task_ids=rescued_invalid),
                extension_task_count=len(extension_ids), skipped_historical_success_tasks=len(old_success),
                new_primary_attempts=len(new), expected_new_primary_attempts=len(new_allowed),
                tasks=rows, limitations=LIMITATION, optimizer_updates=0)


def pending_rows(rows, record_root):
    pending = []
    for row in rows:
        task = str(row['metadata']['task_id'])
        missing = False
        for attempt in NEW_ATTEMPTS:
            path = Path(record_root)/(key(task, attempt)+'.json')
            if path.exists():
                verify_record(_read(path), task, attempt)
            else:
                missing = True
        if missing:
            pending.append(row)
    return pending


def validate_config(config):
    if (config.get('seed') != SEED or config.get('policy_id') != POLICY or
            config.get('max_steps', 15) != 15 or config.get('inference_timeout', 180) != 180):
        raise ValueError('Frozen policy, seed or trajectory protocol changed')
    root = Path(config['output']).resolve()
    attempts = Path(config['attempt_output']).resolve()
    if attempts == root or root not in attempts.parents:
        raise ValueError('Physical trajectory outputs must live inside this new shard')
    for old in config['old_record_roots']:
        old = Path(old).resolve()
        if old == root or old in root.parents or root in old.parents:
            raise ValueError('Historical inputs and new outputs must not overlap')
    historical = config['historical_task_order']
    extension = config['extension_task_order']
    if len(historical) != 2000 or len(set(historical)) != 2000:
        raise ValueError('Expected the preserved 2,000-task historical denominator')
    if any(task not in historical for task in extension):
        raise ValueError('Extension contains a task outside the historical cohort')
    shards = partition(extension)
    if type(config['shard']) is not int or config['shard'] not in range(4):
        raise ValueError('Invalid extension shard')
    if config['task_order'] != shards[config['shard']]:
        raise ValueError('Shard differs from the exact filtered extension partition')
    for field, cap in [('browser_cap', BROWSER_CAP_PER_SHARD),
                       ('judge_max_calls', JUDGE_CALLS_PER_SHARD),
                       ('judge_max_usd', JUDGE_USD_PER_SHARD)]:
        if config.get(field, cap) != cap:
            raise ValueError('Per-shard budget changed')


def validate_protocol(args, config, sampling_params=None, dataset=None):
    validate_config(config)
    if (args.judge_api_model != 'gpt-4.1' or args.judge_prompt_variant != 'action_history' or
            args.num_rollout != 0 or getattr(args, 'max_steps', 15) != 15):
        raise ValueError('Frozen actor/native outcome protocol changed')
    if config.get('checkpoint') and Path(args.hf_checkpoint).resolve() != Path(config['checkpoint']).resolve():
        raise ValueError('The actual actor checkpoint differs from the pinned official SFT checkpoint')
    if sampling_params is not None and any(sampling_params.get(k) != v for k, v in DECODING.items()):
        raise ValueError('Frozen actor decoding changed')
    if dataset is not None and (dataset.n_samples_per_eval_prompt, dataset.temperature,
            dataset.top_p, dataset.top_k, dataset.max_response_len) != (3, .8, 1., -1, 1024):
        raise ValueError('Extension requires three attempts with frozen decoding')


def _bind_config(config):
    # Scheduler retries use new job/output directories; scientific identity,
    # cohort, canonical outputs and all persistent caps must stay identical.
    immutable = {k: v for k, v in config.items() if k not in ('job_id', 'attempt_output')}
    path = Path(config['output'])/'collector-config.json'
    if path.exists():
        if _read(path) != immutable:
            raise ValueError('A resumed collection cannot change its pinned configuration')
    else:
        _new_json(path, immutable)
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    execution = path.parent/'execution-configs'/(identity+'.json')
    if not execution.exists():
        _new_json(execution, config)


@asynccontextmanager
async def task_lock(root, task):
    """Serialize this task across coroutines/processes without blocking the loop."""
    directory = Path(root)/'task-locks'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/(hashlib.sha256(task.encode()).hexdigest()+'.lock')).open('a+') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(.1)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def reserve_dispatch(config, task, attempt):
    """Durably charge a physical dispatch before any browser or model work."""
    root = Path(config['output'])
    directory = root/'physical-attempts'
    directory.mkdir(parents=True, exist_ok=True)
    with (root/'dispatch-account.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = list(directory.iterdir())
        # Even a process killed between mkdir and dispatch.json consumes its slot.
        if len(existing) >= BROWSER_CAP_PER_SHARD:
            raise RuntimeError('Physical browser dispatch budget exhausted')
        ident = f'{len(existing)+1:06d}-{uuid.uuid4().hex}'
        receipt = directory/ident
        receipt.mkdir()
        _sync_dir(directory)
        value = dict(physical_attempt_id=ident, task_id=task, attempt=attempt,
                     seed=seed_for(task, attempt), policy_id=POLICY, judge_id=JUDGE,
                     reserved_browser_dispatches=1, created_unix=time.time(), pid=os.getpid(),
                     output=str(Path(config['attempt_output'])/ident))
        _new_json(receipt/'dispatch.json', value)
        index = root/'physical-index'/key(task, attempt)
        index.mkdir(parents=True, exist_ok=True)
        os.link(receipt/'dispatch.json', index/(ident+'.json'))
        _sync_dir(index)
        return receipt, value


def _publish(record, source_path, canonical, receipt, *, recovered=False):
    canonical.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_path, canonical)
        _sync_dir(canonical.parent)
    except FileExistsError:
        if _read(canonical) != record:
            raise ValueError('Canonical attempt already has different evidence')
    completion = receipt/'completed.json'
    value = dict(canonical_record=str(canonical), task_id=record['task_id'], attempt=record['attempt'],
                 valid=record['valid'], reward=record['reward'], recovered_after_interruption=recovered,
                 canonical_sha256=hashlib.sha256(canonical.read_bytes()).hexdigest())
    if not completion.exists():
        _new_json(completion, value)


def _recover_saved(config, task, attempt, canonical):
    found = []
    root = Path(config['output'])
    for path in sorted((root/'physical-index'/key(task, attempt)).glob('*.json')):
        dispatch = _read(path)
        receipt = root/'physical-attempts'/dispatch['physical_attempt_id']
        if (dispatch['task_id'], dispatch['attempt']) != (task, attempt):
            continue
        saved = Path(dispatch['output'])/'screen_records'/(key(task, attempt)+'.json')
        if saved.exists():
            record = verify_record(_read(saved), task, attempt, deep=True)
            found.append((record, saved, receipt))
    if len(found) > 1:
        raise ValueError('Multiple durable physical results for one primary attempt; audit before choosing')
    if found:
        _publish(*found[0][:2], canonical, found[0][2], recovered=True)
        return True
    return False


async def generate(args, sample, sampling_params, evaluation=False):
    if not evaluation:
        raise ValueError('The extension never trains')
    config = _read(os.environ['OPENWEBRL_TASK_PASS8_CONFIG'])
    validate_protocol(args, config, sampling_params=sampling_params)
    task = str(sample.metadata['task_id'])
    if task not in config['task_order'] or type(sample.index) is not int or sample.index < 0:
        raise ValueError('Unregistered task or invalid sample index')
    attempt = sample.index % 3 + 5
    root = Path(config['output'])
    canonical = root/'records'/(key(task, attempt)+'.json')
    async with task_lock(root, task):
        if canonical.exists():
            verify_record(_read(canonical), task, attempt)
            if not _recover_saved(config, task, attempt, canonical):
                raise ValueError('Canonical attempt lacks preserved physical dispatch evidence')
            return []
        if _recover_saved(config, task, attempt, canonical):
            return []
        if (root/'halt.json').exists() or (root/'judge-budget'/'halt.json').exists():
            raise RuntimeError('Collection halted; preserve all physical attempts')
        receipt, dispatch = reserve_dispatch(config, task, attempt)
        local = dict(config, output=dispatch['output'], selector_endpoint='http://127.0.0.1:1')
        try:
            # The original helper resets session_id=None and samples each action
            # with the per-task/attempt seed; no previous trajectory is supplied.
            await trajectory(args, sample, sampling_params, 'actor', attempt, 'screen', local)
            saved = Path(local['output'])/'screen_records'/(key(task, attempt)+'.json')
            record = verify_record(_read(saved), task, attempt, deep=True)
            _publish(record, saved, canonical, receipt)
        except BaseException as exc:
            if not (receipt/'interrupted.json').exists():
                _new_json(receipt/'interrupted.json', dict(error_type=type(exc).__name__,
                    charged_browser_dispatches=1, created_unix=time.time(),
                    note='Preserved physical attempt; canonical publication may be incomplete'))
            raise
    return []  # Archives are durable; do not retain all trajectories in Ray.


def load_records(record_root, ids=None):
    records = []
    for path in sorted(Path(record_root).glob('*.json')):
        record = _read(path)
        if path.name != key(record['task_id'], record['attempt'])+'.json':
            raise ValueError('Record filename does not match primary identity')
        if ids is None or record['task_id'] in ids:
            records.append(record)
    return records


def load_old_records(config, ids):
    # The parent audit establishes original 10K provenance. Each shard checks
    # immutable record/sidecar hashes for the global join without Torch scans.
    proof_path = Path(config['historical_inputs'])
    if hashlib.sha256(proof_path.read_bytes()).hexdigest() != config['historical_inputs_sha256']:
        raise ValueError('Historical input manifest hash changed')
    proof = _read(proof_path)
    if not isinstance(proof, list) or len(proof) != 10000:
        raise ValueError('Historical manifest must preserve all 10,000 original attempts')
    proof_index = {(p['task_id'], p['attempt']): p for p in proof}
    proof_tasks = {p['task_id'] for p in proof}
    if (len(proof_tasks) != 2000 or
            set(proof_index) != {(task, attempt) for task in proof_tasks for attempt in range(5)}):
        raise ValueError('Historical manifest has duplicate, missing or changed attempt indices')
    wanted = set(ids)
    records = [record for root in config['old_record_roots'] for record in load_records(root, wanted)]
    indexed, allowed = _index(records, ids, range(5))
    if set(indexed) != allowed:
        raise ValueError('Missing preserved historical attempt')
    old_roots = {Path(r).resolve() for r in config['old_record_roots']}
    for ident, record in indexed.items():
        evidence = proof_index[ident]
        path = Path(evidence['record'])
        if (path.parent.resolve() not in old_roots or
                _read(path) != record or
                hashlib.sha256(path.read_bytes()).hexdigest() != evidence['record_sha256']):
            raise ValueError('Preserved historical canonical record changed')
        archive = _archive(record)
        if (archive.resolve() != Path(evidence['archive']).resolve() or
                archive.stat().st_size != evidence['archive_size_bytes'] or
                hashlib.sha256(archive.with_suffix('.json').read_bytes()).hexdigest() != evidence['sidecar_sha256']):
            raise ValueError('Preserved historical sidecar or archive changed')
    return records


def audit(config, *, require_complete=False, deep=True):
    validate_config(config)
    ids = config['task_order']
    records = load_records(Path(config['output'])/'records')
    summary = summarize(records, ids, require_complete, deep=deep)
    old = load_old_records(config, config['historical_task_order'])
    # A completed shard cannot establish campaign-wide pass@8; the owner joins
    # all four record directories against the same 707 required target tasks.
    joined = summarize_joined(old, records, config['historical_task_order'],
        verify_artifacts=False, extension_ids=config['extension_task_order'])
    dispatch_root = Path(config['output'])/'physical-attempts'
    physical = list(dispatch_root.iterdir()) if dispatch_root.exists() else []
    if len(physical) > BROWSER_CAP_PER_SHARD:
        raise ValueError('Physical dispatch cap exceeded')
    summary['physical_dispatches_charged'] = len(physical)
    summary['unsettled_physical_attempts'] = sum(not (p/'completed.json').exists() for p in physical)
    settled = set()
    for path in physical:
        if (path/'completed.json').exists():
            receipt = _read(path/'completed.json')
            canonical = Path(receipt['canonical_record'])
            if (not canonical.is_file() or
                    hashlib.sha256(canonical.read_bytes()).hexdigest() != receipt['canonical_sha256']):
                raise ValueError('Physical completion receipt does not match its canonical record')
            dispatch = _read(path/'dispatch.json')
            ident = (receipt['task_id'], receipt['attempt'])
            if (ident in settled or ident != (dispatch['task_id'], dispatch['attempt']) or
                    canonical.resolve() != (Path(config['output'])/'records'/
                        (key(*ident)+'.json')).resolve()):
                raise ValueError('Duplicate or mismatched physical completion identity')
            settled.add(ident)
    if settled != {(r['task_id'], r['attempt']) for r in records}:
        raise ValueError('Canonical records lack exactly one charged physical attempt receipt')
    budget_path = Path(config['output'])/'judge-budget'/'ledger.json'
    budget = _read(budget_path) if budget_path.exists() else dict(calls=0, charged_or_reserved_usd=0.)
    if (type(budget['calls']) is not int or not 0 <= budget['calls'] <= JUDGE_CALLS_PER_SHARD or
            not math.isfinite(budget['charged_or_reserved_usd']) or
            not 0 <= budget['charged_or_reserved_usd'] <= JUDGE_USD_PER_SHARD):
        raise ValueError('Persistent native judge budget is invalid or exceeded')
    summary['judge_calls_charged'] = budget['calls']
    summary['judge_charged_or_reserved_usd'] = budget['charged_or_reserved_usd']
    summary['actor_requests'] = sum(r['actor_requests'] for r in records)
    summary['actor_output_tokens'] = sum(r['actor_output_tokens'] for r in records)
    summary['trajectory_seconds_sum_not_gpu_seconds'] = sum(r['elapsed_seconds'] for r in records)
    return summary, joined


def batch_gate(records, *, startup=False):
    if startup and len(records) != 6:
        raise RuntimeError('Startup must contain six primary attempts')
    if records and sum(not r['valid'] for r in records)/len(records) > .5:
        raise RuntimeError('More than 50% invalid in the completed batch; diagnose before scaling')


async def collect(args, rollout_id, config):
    import openwebrl.reward_browser as reward
    from openwebrl.arm_terminal_budget import CappedJudge
    from slime.rollout.sglang_rollout import eval_rollout_single_dataset
    from slime.rollout.base_types import RolloutFnEvalOutput

    if len(args.eval_datasets) != 1:
        raise ValueError('Exactly one frozen shard dataset is required')
    dataset = args.eval_datasets[0]
    validate_protocol(args, config, dataset=dataset)
    rows = [json.loads(line) for line in Path(config['tasks']).read_text().splitlines() if line.strip()]
    ids = [str(row['metadata']['task_id']) for row in rows]
    manifest = _read(config['partition_manifest'])
    validate_partition(manifest['shards'], manifest['extension_task_order'])
    if (manifest['historical_task_order'] != config['historical_task_order'] or
            manifest['extension_task_order'] != config['extension_task_order'] or
            manifest['task_order'] != manifest['extension_task_order']):
        raise ValueError('Historical denominator or selected extension cohort changed')
    if ids != config['task_order'] or ids != manifest['shards'][config['shard']]:
        raise ValueError('Frozen shard order changed')
    root = Path(config['output'])
    _bind_config(config)
    old_records = load_old_records(config, config['historical_task_order'])
    summarize_joined(old_records, [], config['historical_task_order'], verify_artifacts=False,
                     extension_ids=config['extension_task_order'])
    records = load_records(root/'records')
    summarize(records, ids, deep=True)  # Verify archives before skipping resumed primary attempts.
    for record in records:
        # A process may stop after the atomic canonical link but just before
        # its completion receipt. Settle that charged attempt without replay.
        canonical = root/'records'/(key(record['task_id'], record['attempt'])+'.json')
        if not _recover_saved(config, record['task_id'], record['attempt'], canonical):
            raise ValueError('Resumed canonical attempt lacks physical dispatch evidence')
    judge = CappedJudge(root/'judge-budget', max_calls=JUDGE_CALLS_PER_SHARD,
                        max_usd=JUDGE_USD_PER_SHARD)
    previous = reward._get_openai_client
    reward._get_openai_client = lambda **unused: judge
    try:
        boundaries = [0, 2, *range(22, len(rows), 20), len(rows)]
        for a, b in zip(boundaries, boundaries[1:]):
            selected = rows[a:b]
            pending = pending_rows(selected, root/'records')
            if pending:
                path = root/f'work-{a}.jsonl'
                content = ''.join(json.dumps(row)+'\n' for row in pending)
                # Unique resume worklists preserve every earlier scheduled batch.
                path = path.with_name(f'{path.stem}-{uuid.uuid4().hex}{path.suffix}')
                with path.open('x') as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                await eval_rollout_single_dataset(args, rollout_id,
                    replace(dataset, name=f'sft-pass8-shard{config["shard"]}-{a}', path=str(path)))
            recent = []
            for row in selected:
                task = str(row['metadata']['task_id'])
                recent.extend(verify_record(_read(root/'records'/(key(task, n)+'.json')), task, n)
                              for n in NEW_ATTEMPTS)
            records = load_records(root/'records')
            summary = summarize(records, ids, verify_artifacts=False)
            write_json(root/'progress.json', {k: v for k, v in summary.items() if k != 'dispositions'})
            joined = summarize_joined(old_records, records, config['historical_task_order'],
                verify_artifacts=False, extension_ids=config['extension_task_order'])
            write_json(root/'joined-progress.json', {k: v for k, v in joined.items() if k != 'tasks'})
            try:
                batch_gate(recent, startup=a == 0)
            except RuntimeError:
                write_json(root/'halt.json', dict(reason='batch_invalid_rate', batch_start=a,
                    primary_attempts=len(recent), invalid=sum(not r['valid'] for r in recent)))
                raise
            if a == 0:
                write_json(root/'startup-check.json', dict(passed=True, primary_attempts=6,
                    invalid=sum(not r['valid'] for r in recent), included_in_primary=True))
        summary, joined = audit(config, require_complete=True, deep=True)
        write_json(root/'summary.json', summary)
        write_json(root/'joined-summary.json', joined)
        write_json(root/'collection-complete.json', {k: v for k, v in summary.items() if k != 'dispositions'})
        return RolloutFnEvalOutput(data={}, metrics={k: v for k, v in summary.items()
                                                   if type(v) in (int, float)})
    finally:
        reward._get_openai_client = previous
        await judge.close()


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if not evaluation or args.num_rollout != 0:
        raise ValueError('The extension must execute zero optimizer updates')
    from slime.utils.async_utils import run
    config = _read(os.environ['OPENWEBRL_TASK_PASS8_CONFIG'])
    return run(collect(args, rollout_id, config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--audit', action='store_true', required=True)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--shallow', action='store_true', help='Skip Torch archive decoding')
    args = parser.parse_args()
    summary, joined = audit(_read(args.config), require_complete=args.require_complete, deep=not args.shallow)
    print(json.dumps(dict(summary=summary, joined=joined), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
