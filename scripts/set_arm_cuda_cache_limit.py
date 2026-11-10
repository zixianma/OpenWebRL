#!/usr/bin/env python3
"""Change the cache threshold on existing ARM actors between serialized methods.

No new Ray actors, GPU allocations, model changes, or training restarts.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time

RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime')


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.partial')
    tmp.write_text(json.dumps(data, indent=2)+'\n'); tmp.replace(path)


def change_actor(actor, expected_root, limit, receipt_directory):
    # Ray executes this closure as an ordinary serialized actor method, after
    # the active train call has returned. It must not change any learned state.
    import os
    import json
    import time
    from pathlib import Path
    if limit != 48:
        raise ValueError('This operation only implements the requested48GiB setting')
    root = Path(expected_root)
    if (getattr(actor, 'role', None) != 'actor'
            or Path(actor.args.save).resolve() != (root/'runtime').resolve()
            or actor._world_size != 8 or actor._rank not in range(8)):
        raise ValueError('Actor identity does not match the approved mixed run')
    key = 'OPENWEBRL_CUDA_CACHE_LIMIT_GIB'
    old = os.environ.get(key)
    if old not in ('24', '48'):
        raise ValueError(f'Unexpected prior cache threshold: {old!r}')
    scheduler = getattr(actor, 'opt_param_scheduler', None)
    scheduler_steps = getattr(scheduler, 'num_steps', None)
    optimizer = getattr(actor, 'optimizer', None)
    optimizer_identity = id(optimizer)
    os.environ[key] = str(limit)
    assert id(actor.optimizer) == optimizer_identity
    assert getattr(scheduler, 'num_steps', None) == scheduler_steps
    result = dict(passed=True, rank=actor._rank, pid=os.getpid(), old_limit_gib=int(old),
        new_limit_gib=limit, applied_epoch=time.time(), scheduler_num_steps=scheduler_steps,
        learned_state_untouched=True, train_root=str(root), application='serialized Ray actor method boundary')
    path = Path(receipt_directory)/f'rank-{actor._rank}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.partial')
    tmp.write_text(json.dumps(result, indent=2)+'\n'); tmp.replace(path)
    return result


def cluster_address(job):
    expected = f'/tmp/am-{job}/ray/'
    addresses = set()
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode(errors='replace').split('\0')
        except (OSError, PermissionError):
            continue
        if args and args[0].endswith('/raylet') and any(expected in x for x in args):
            addresses.update(x.split('=',1)[1] for x in args if x.startswith('--gcs-address='))
    if len(addresses) != 1:
        raise ValueError('Expected exactly one matching job-local Ray cluster')
    return addresses.pop()


def existing_actor_handle(actor_id):
    """Register a weak handle for these driver-created, synchronous actors.

    CoreWorker.get_actor_handle only works for handles already registered in
    this driver. Reconstruct the ordinary serialized handle from GCS metadata;
    validate this adapter against Ray's own serialization in the CPU probe.
    """
    import struct
    import ray
    from ray.core.generated import core_worker_pb2, gcs_pb2
    aid = ray.ActorID.from_hex(actor_id)
    accessor = ray._private.state.state._connect_and_get_accessor()
    info = gcs_pb2.ActorTableData.FromString(accessor.get_actor_info(aid))
    if (info.parent_id not in (b'', ray.ActorID.nil().binary())
            or info.owner_address.worker_id != info.job_id+b'\xff'*24
            or info.is_detached):
        raise ValueError(f'Requires an ordinary live actor created by this job driver: '
                         f'parent={info.parent_id.hex()} nil={ray.ActorID.nil().hex()} '
                         f'owner_worker={info.owner_address.worker_id.hex()} '
                         f'job={info.job_id.hex()} detached={info.is_detached}')
    handle = core_worker_pb2.ActorHandle(
        actor_id=info.actor_id,
        owner_id=ray.TaskID.for_driver_task(ray.JobID.from_binary(info.job_id)).binary(),
        owner_address=info.owner_address, creation_job_id=info.job_id,
        actor_creation_task_function_descriptor=info.function_descriptor,
        actor_cursor=ray.TaskID.for_actor_creation_task(aid).binary()+struct.pack('<I', 1),
        extension_data=b'0', name=info.name, ray_namespace=info.ray_namespace,
        max_pending_calls=-1, enable_task_events=True,
        labels=dict(info.labels))
    serialized = handle.SerializeToString()
    return ray.actor.ActorHandle._deserialization_helper(serialized, True, None), serialized


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--job-id', required=True)
    a.add_argument('--variant', choices=('bonus','reweight'), required=True)
    a.add_argument('--apply', action='store_true')
    args = a.parse_args(); job = args.job_id
    if not job.isdecimal() or os.getenv('SLURM_JOB_ID') != job:
        raise ValueError('Use an overlapping CPU step inside the existing allocation')
    root = RUNTIME/f'evaluations/arm-mixed-{args.variant}-{job}'
    manifest = json.loads((root/'launch_manifest.json').read_text())
    if (manifest.get('wandb_run_id') != f'arm-mixed-{args.variant}-{job}'
            or manifest.get('mixed_reweight_variant') != args.variant
            or manifest['environment'].get('TP_SIZE') != '2'):
        raise ValueError('Wrong run lineage or topology')
    out = root/'cache-policy-48'; address = cluster_address(job)
    # Import the frozen actor class before handle deserialization. Otherwise
    # Ray silently creates a placeholder class when dependencies are missing.
    from slime.backends.megatron_utils.actor import MegatronTrainRayActor
    import inspect
    if not Path(inspect.getfile(MegatronTrainRayActor)).resolve().is_relative_to(Path(manifest['source'])):
        raise ValueError('Control driver must import the same frozen training source')
    import ray
    from ray._private.state import actors
    ray.init(address=address, log_to_driver=False)
    try:
        table = actors(actor_state_name='ALIVE')
        selected = {key:row for key,row in table.items()
                    if row.get('ActorClassName') == 'MegatronTrainRayActor'}
        if len(selected) != 8:
            write(out/'discovery.json', {'actors': [
                {k:v for k,v in row.items() if k in ('ActorID','ActorClassName','State','Pid','Name')}
                for row in table.values()]})
            raise ValueError(f'Expected8 training actors, found{len(selected)}')
        handles = [existing_actor_handle(key)[0] for key in selected]
        if any(not h._ray_method_signatures.get('__ray_call__') for h in handles):
            raise ValueError('Ray actor class failed to load; do not use a placeholder handle')
        write(out/'discovery.json', dict(job_id=job, address=address, actors=list(selected),
            handle_count=len(handles), inspected_epoch=time.time()))
        print(json.dumps(dict(job_id=job, actors=len(handles), apply=args.apply)), flush=True)
        if not args.apply:
            return
        requested = time.time()
        write(out/'status.json', dict(stage='queued_at_actor_boundary', applied=False,
            requested_limit_gib=48, requested_epoch=requested, job_id=job))
        # Existing actor classes can retain Ray's older fn-only signature.
        # Capture parameters in the closure rather than passing RPC varargs.
        def apply_at_boundary(actor):
            return change_actor(actor, str(root), 48, str(out))
        refs = [h.__ray_call__.remote(apply_at_boundary) for h in handles]
        pending = list(refs); results = []
        deadline = time.monotonic()+5400
        while pending and time.monotonic() < deadline:
            ready, pending = ray.wait(pending, num_returns=1, timeout=30)
            results.extend(ray.get(ready))
            write(out/'status.json', dict(stage='applied' if not pending else 'waiting_for_actor_boundary',
                applied=not pending, requested_limit_gib=48, completed_ranks=len(results),
                pending_ranks=len(pending), requested_epoch=requested,
                updated_epoch=time.time(), job_id=job, receipts=results))
        if pending:
            raise TimeoutError('Cache change still queued; preserve intent and inspect per-rank receipts')
        if sorted(x['rank'] for x in results) != list(range(8)) or any(x['new_limit_gib'] != 48 for x in results):
            raise ValueError('Per-rank verification failed')
        print(json.dumps(dict(job_id=job, applied=True, ranks=8, limit_gib=48)), flush=True)
    except Exception as exc:
        if args.apply:
            write(out/'status.json', dict(stage='failed', applied=False, job_id=job,
                updated_epoch=time.time(), error_type=type(exc).__name__, error=str(exc)[:2000],
                inspect_rank_receipts_before_retry=True))
        raise
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
