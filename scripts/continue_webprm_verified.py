#!/usr/bin/env python3
"""Continue the fixed v4 cohort after one explicitly evidenced operational waiver.

This driver neither grants resources nor verifies scheduler closure. Its launcher
must close the former allocation and bound both deadlines by remaining approved
time. It never changes the frozen scientific identity or replays saved cells.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import sys
import time


ARMS = ('shepherd_unguided', 'shepherd_random5', 'shepherd8b',
        'arbiter_unguided', 'arbiter_random5', 'arbiter7b')
GATE_FAILURE = 'quality_gate_no_valid_episode:arbiter_unguided'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity_hash(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def verify_hashes(root, hashes):
    for name, expected in hashes.items():
        require(sha256(Path(root) / name) == expected, 'Preserved file changed: ' + name)


def saved_results(root, task_ids, identity):
    expected = {(task, arm) for task in task_ids for arm in ARMS}
    rows, hashes = [], {}
    for path in sorted(root.glob('*/*/result.json')):
        row = read_json(path)
        key = (path.parent.parent.name, path.parent.name)
        require(key in expected and key == (row.get('task_id'), row.get('arm')),
                'Unexpected or misfiled saved result')
        require(row.get('identity_sha256') == identity, 'Saved result identity changed')
        require(row.get('attempted') is True, 'Saved unattempted cell cannot be retried by this driver')
        rows.append(row)
        hashes[str(path.relative_to(root))] = sha256(path)
    return rows, hashes


def gate_proof(root, rows, task_ids, identity):
    """Require the precise observed failure, including durable truncation evidence."""
    gate_rows = [r for r in rows if r['task_id'] in task_ids[:2]]
    gate = {(r['task_id'], r['arm']): r for r in gate_rows}
    require(len(gate) == len(gate_rows) == 12, 'All twelve initial cells must be durable and unique')
    require(all(r.get('attempted') is True and r.get('identity_sha256') == identity
                for r in gate.values()), 'Initial cells must be attempted under the original identity')
    valid = {arm: sum(bool(gate[(task, arm)]['valid']) for task in task_ids[:2]) for arm in ARMS}
    require([arm for arm in ARMS if valid[arm] == 0] == ['arbiter_unguided'],
            'Only Arbiter unguided may have zero valid initial episodes')
    require(valid['shepherd8b'] >= 1 and valid['arbiter7b'] >= 1,
            'Both guided arms must have a valid initial episode')
    for arm in ARMS:
        row = gate[(task_ids[1], arm)]
        require(row.get('valid') is False and row.get('error_type') == 'RuntimeError'
                and row.get('error') == 'Initial website HTTP status 403'
                and row.get('turns') == 0, 'Every second-task cell must be the initial website 403 failure')
    row = gate[(task_ids[0], 'arbiter_unguided')]
    require(row.get('valid') is False and row.get('error_type') == 'ValueError'
            and row.get('error') == 'Incomplete Arbiter policy response',
            'First-task Arbiter unguided must have the observed policy truncation')
    journal = root / task_ids[0] / 'arbiter_unguided' / 'requests.jsonl'
    events = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    require(len(events) >= 2, 'Missing durable policy truncation evidence')
    request, response = events[-2:]
    payload, result = request.get('payload', {}), response.get('response', {})
    require(request.get('event') == 'request' and request.get('purpose') == 'policy'
            and request.get('local') is False and payload.get('model') == 'gpt-4o-mini-2024-07-18'
            and payload.get('max_tokens') == 2048 and payload.get('n') == 1
            and payload.get('temperature') == 1. and payload.get('top_p') == .9
            and response.get('event') == 'response'
            and response.get('reservation') == request.get('reservation')
            and request.get('reservation') is not None
            and len(result.get('choices', [])) == 1
            and result['choices'][0].get('finish_reason') == 'length'
            and result.get('usage', {}).get('completion_tokens') == 2048,
            'Policy response must prove truncation at the unchanged 2048-token cap')
    return dict(original_gate_status='failed', original_gate_failure=GATE_FAILURE,
                gate_task_ids=task_ids[:2], attempted=12, valid_by_arm=valid,
                override='manual_operational_gate_waiver',
                rationale='One fixed capped actor response and a shared website 403; retain all failures.',
                truncation_journal_sha256=sha256(journal),
                scientific_caveat='Diagnostic gate was overridden after inspecting outcomes; it did not pass.')


def remaining_cells(tasks, rows, seed):
    """Keep the frozen shuffle, including draws consumed by saved/gate tasks."""
    saved = {(r['task_id'], r['arm']) for r in rows}
    rng, cells = random.Random(seed), []
    for index, task in enumerate(tasks):
        order = list(ARMS)
        rng.shuffle(order)
        if index >= 2:
            cells.extend((task, arm) for arm in order if (task['task_id'], arm) not in saved)
    return cells


def validate_deadlines(deadline, allocation_deadline, now=None):
    now = time.time() if now is None else now
    require(all(math.isfinite(x) for x in (deadline, allocation_deadline, now)), 'Deadlines must be finite')
    require(now < deadline <= allocation_deadline and allocation_deadline - now <= 6000,
            'Continuation must fit the launcher-provided remaining allocation, at most 6000 seconds')


def verify_manifest(runtime, manifest):
    identity, config = manifest['identity'], manifest['config']
    digest = identity_hash(identity)
    require(manifest['identity_sha256'] == config['identity_sha256'] == digest, 'Identity hash mismatch')
    require(identity['protocol_version'] == 'native-recovery-v4', 'Only frozen v4 can continue')
    require(len(identity['task_ids']) == len(set(identity['task_ids'])) == 12
            and identity['arms'] == list(ARMS), 'Exactly twelve fixed tasks and six arms are required')
    require(config['workers'] == 4 and config['quality_gate_tasks'] == 2
            and identity['scheduling_policy']['max_outstanding_cells'] == 4
            and identity['scheduling_policy']['initial_stage_tasks'] == 2
            and identity['recovery_protocol']['quality_gate_tasks'] == 2,
            'The unchanged four-worker/two-task scheduling contract is required')
    require(identity['scheduling_policy']['name'] == 'bounded_cross_task_two_stage_v1'
            and identity['scheduling_policy']['expansion'] == 'only_after_all_initial_cells_attempted_and_quality_gate_passes',
            'Unexpected original scheduling policy')
    require(config['seed'] == identity['seed'] == 20261009, 'Fixed shuffle seed changed')
    require(config['code_sha256'] == identity['code_sha256']
            and config['source_manifests'] == identity['sources'], 'Code or source identity changed')
    require(Path(config['output']).resolve() == (runtime / 'episodes').resolve(), 'Output root changed')
    require(sha256(config['tasks']) == identity['task_file_sha256'], 'Task file changed')
    plan = read_json(config['plan'])
    require(plan['pilot_task_ids'] == identity['task_ids'] and plan['arms'] == list(ARMS)
            and plan['task_file_sha256'] == identity['task_file_sha256']
            and plan['sources'] == identity['sources']
            and plan['models'] == identity['model_revisions']
            and plan['model_source_equivalence'] == identity['model_source_equivalence'],
            'Prepared cohort/source/model identity changed')
    require(Path(config['budget_path']).resolve() == Path(plan['shared_budget_path']).resolve(),
            'Shared ledger path changed')
    require(Path(config['budget_path']).resolve().is_relative_to('/gpfs/scrubbed/zixianma/openwebrl-runtime'),
            'Shared ledger must remain in the approved writable runtime root')
    budget = read_json(config['budget_path'])
    require(budget['cap'] == identity['api_usd_cap'] == 30 and not budget.get('halt_reason')
            and 0 <= budget['charged'] < 30, 'Shared API cap, charge or halt blocks continuation')
    frozen = read_json(runtime / 'code-freeze-v4.json')
    code = Path(frozen['code_directory']).resolve()
    require(code == (runtime / 'code-v4').resolve(), 'Frozen code root changed')
    verify_hashes(code, frozen['code_sha256'])
    verify_hashes(code, identity['code_sha256'])
    for name, version in identity['dependencies'].items():
        require(importlib.metadata.version(name) == version, 'Dependency changed: ' + name)
    for name in ('shepherd', 'arbiter', 'arbiter_harness', 'arbiter_observation'):
        source_root = config['shepherd_source' if name == 'shepherd' else 'arbiter_source']
        verify_hashes(source_root, identity['sources'][name]['files'])
    require(sha256(config['shepherd_actor_source']) == identity['sources']['shepherd_actor_sha256']
            and sha256(config['arbiter_benchmark']) == identity['sources']['arbiter_prompt']['dataset_sha256'],
            'Actor source or fixed critic template changed')
    return code


def load_frozen_pilot(code):
    """Import frozen modules without replacing any already loaded module."""
    for name, module in list(sys.modules.items()):
        if name == 'openwebrl' or name.startswith('openwebrl.'):
            location = getattr(module, '__file__', None)
            require(location is not None and Path(location).resolve().is_relative_to(code),
                    'A non-frozen OpenWebRL module is already imported: ' + name)
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location('webprm_frozen_continuation_pilot', code / 'scripts/run_webprm_pilot.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(args):
    runtime = Path(args.runtime).resolve()
    require(runtime.is_relative_to('/gpfs/scrubbed/zixianma/openwebrl-runtime'), 'Runtime must use the approved scrubbed root')
    root = runtime / 'episodes'
    require(root.resolve().is_relative_to(runtime), 'Episode root cannot escape the approved runtime through a symlink')
    for name in ('manifest.json', 'results.json', 'status.json', 'partial.json', 'summary.json',
                 'complete.json', 'operational-continuations'):
        require((root / name).resolve().is_relative_to(root.resolve()), 'Writable aggregate path escapes episode root')
    manifest_path = root / 'manifest.json'
    manifest = read_json(manifest_path)
    code = verify_manifest(runtime, manifest)
    validate_deadlines(args.deadline, args.allocation_deadline)
    closure_path = Path(args.closure_evidence).resolve()
    require(read_json(closure_path).get('former_job_closed') is True,
            'Launcher must provide its explicit former-job closure attestation')
    identity = manifest['identity_sha256']
    task_ids = manifest['identity']['task_ids']
    rows, prior_hashes = saved_results(root, task_ids, identity)
    proof = gate_proof(root, rows, task_ids, identity)
    require(read_json(root / 'partial.json').get('stop_reason') == GATE_FAILURE
            or bool(manifest['config'].get('operational_continuations')),
            'Original controller must have durably recorded its failed gate')
    pilot = load_frozen_pilot(code)
    require(tuple(pilot.PILOT_ARMS) == ARMS and pilot.quality_gate_failure(rows, task_ids[:2]) == GATE_FAILURE,
            'Frozen controller must independently confirm the original gate failed')
    indexed = {task['task_id']: task for task in pilot.load_tasks(manifest['config']['tasks'])}
    tasks = [indexed[task_id] for task_id in task_ids]
    cells = remaining_cells(tasks, rows, manifest['config']['seed'])
    require(cells, 'No unattempted cells remain')
    for task, arm in cells:
        folder = root / task['task_id'] / arm
        require(folder.resolve().is_relative_to(root.resolve()), 'Cell path escapes approved episode root')
        require(not folder.exists(), 'Unfinished cell artifacts require separate review; no replay: ' + str(folder))
    # Verify imports and source adapters against the exact original manifests.
    from openwebrl.webprm_source import load_shepherd, load_arbiter, load_arbiter_causal_renderer
    from openwebrl.webprm_arbiter_harness import load_arbiter_harness, load_native_observation_processor
    config = dict(manifest['config'])
    sources = dict(shepherd=load_shepherd(config['shepherd_source']).manifest,
        arbiter=load_arbiter(config['arbiter_source']).manifest,
        arbiter_harness=load_arbiter_harness(config['arbiter_source']).manifest,
        arbiter_observation=load_native_observation_processor(config['arbiter_source'], viewport_size={'width':1280,'height':720}).manifest,
        arbiter_prompt=load_arbiter_causal_renderer(config['arbiter_benchmark']).manifest,
        shepherd_actor_sha256=sha256(config['shepherd_actor_source']))
    require(sources == manifest['identity']['sources'], 'Loaded frozen source manifests changed')
    pilot.load_api_credentials(config['env_file'])
    validate_deadlines(args.deadline, args.allocation_deadline)
    verify_hashes(root, prior_hashes)
    receipt_dir = root / 'operational-continuations' / str(time.time_ns())
    receipt_dir.mkdir(parents=True, exist_ok=False)
    for name in ('manifest.json', 'results.json', 'status.json', 'partial.json', 'summary.json', 'complete.json'):
        if (root / name).exists():
            shutil.copyfile(root / name, receipt_dir / ('prior-' + name))
    override = dict(proof, driver_sha256=sha256(__file__), driver_path=str(Path(__file__).resolve()),
                    original_identity_sha256=identity, old_deadline=config['deadline'],
                    new_deadline=args.deadline, allocation_deadline=args.allocation_deadline,
                    closure_evidence=str(closure_path), closure_evidence_sha256=sha256(closure_path),
                    closure_verification='Launcher-attested; this driver performs no scheduler or process queries',
                    workers=4, planned_denominator=72, prior_results_sha256=prior_hashes,
                    unattempted_cells=len(cells), receipt_directory=str(receipt_dir), recorded_at=time.time())
    config['deadline'] = args.deadline
    config['seconds'] = math.ceil(args.deadline - time.time())
    config['operational_continuations'] = list(config.get('operational_continuations', [])) + [override]
    manifest['config'] = config
    manifest['operational_gate_status'] = 'failed_then_manually_overridden'
    pilot.write_json(receipt_dir / 'proof.json', override)
    pilot.write_json(receipt_dir / 'config.json', config)
    receipt = dict(status='running', override=override, started_at=time.time())
    pilot.write_json(receipt_dir / 'receipt.json', receipt)
    pilot.write_json(manifest_path, manifest)
    def record_result(result):
        require((result['task_id'], result['arm']) not in {(r['task_id'], r['arm']) for r in rows},
                'Continuation returned an already saved cell')
        require(result.get('identity_sha256') == identity, 'New cell identity changed')
        verify_hashes(root, prior_hashes)
        rows.append(result)
        pilot.write_json(root / 'results.json', rows)
        status = pilot.cohort_status(rows, task_ids)
        status['operational_gate_status'] = 'failed_then_manually_overridden'
        pilot.write_json(root / 'status.json', status)
    try:
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
            reason = pilot.collect_cells(pool, cells, config, 4, record_result)
        verify_hashes(root, prior_hashes)
        require(read_json(manifest_path)['identity'] == manifest['identity'], 'Scientific identity changed during continuation')
        status = pilot.cohort_status(rows, task_ids)
        status.update(stop_reason=reason, time=time.time(), budget=read_json(config['budget_path']),
                      operational_gate_status='failed_then_manually_overridden', operational_receipt=str(receipt_dir / 'receipt.json'))
        pilot.write_json(root / 'results.json', rows)
        pilot.write_json(root / 'status.json', status)
        if status['status'] == 'completed':
            pilot.write_json(root / 'summary.json', pilot.paired_summary(rows, task_ids))
            pilot.write_json(root / 'complete.json', status)
        else:
            pilot.write_json(root / 'partial.json', status)
        receipt.update(status=status['status'], stop_reason=reason, prior_results_preserved=True,
                       finished_at=time.time(), attempted_episodes=status['attempted_episodes'], valid=status['valid'])
        pilot.write_json(receipt_dir / 'receipt.json', receipt)
    except BaseException as error:
        receipt.update(status='interrupted', error_type=type(error).__name__, error=str(error), finished_at=time.time())
        pilot.write_json(receipt_dir / 'receipt.json', receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--allocation-deadline', type=float, required=True)
    parser.add_argument('--closure-evidence', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
