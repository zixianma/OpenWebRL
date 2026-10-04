"""Isolated saved-batch TP/DP diagnostic; never imported by production sources."""
import hashlib
import json
import math
import os
from pathlib import Path
import time


def shard_windows(additions, dp, rank):
    """Assign every real row once, padding each rank to equal collective counts."""
    if dp not in (1, 2, 4) or not 0 <= rank < dp:
        raise ValueError('Diagnostic supports DP1, DP2 and DP4 only')
    result = []
    for rows in additions:
        local = [(i, scale, False) for i, scale in rows[rank::dp]]
        target = (len(rows) + dp - 1) // dp
        if target > len(local):
            local += [(rows[0][0], 0., True)] * (target - len(local))
        result.append(local)
    return result


def require_diagnostic():
    if os.environ.get('OPENWEBRL_ARM_TPDP_DIAGNOSTIC') != '1':
        raise ValueError('This source is restricted to the isolated TP/DP replay')


def before_collection(*args):
    require_diagnostic()
    return False


before_optimizer = before_collection
after_checkpoint = before_collection
before_training = before_collection
reset_collection = before_collection


def post_process_rewards(args, samples, raw_rewards, normalized_rewards):
    """Reuse the saved calibration decision; never contact a teacher or judge."""
    require_diagnostic()
    from openwebrl.arm_turn_bonus import apply_bonus
    config = json.loads(Path(os.environ['OPENWEBRL_ARM_TURN_BONUS_CONFIG']).read_text())
    return raw_rewards, apply_bonus(samples, normalized_rewards, config['policy_id'],
                                   config['diagnostic_saved_applied_beta'])


def trim_train_data(data):
    """Normalize all original groups first, then fix exactly one global window."""
    require_diagnostic()
    count = len(data['tokens'])
    if count < 256:
        raise ValueError('Saved batch lacks one complete global window')
    result = {k: v[:256] if isinstance(v, list) and len(v) == count else v
              for k, v in data.items()}
    # Used for telemetry only; rewards were normalized before this diagnostic cut.
    result['raw_reward_group_sizes'] = [256]
    fingerprint = hashlib.sha256(json.dumps({k: result[k] for k in
        ('tokens', 'response_lengths', 'loss_masks', 'rewards', 'sample_indices')},
        sort_keys=True).encode()).hexdigest()
    root = Path(os.environ['OPENWEBRL_TPDP_OUTPUT'])
    (root/'batch.json').write_text(json.dumps(dict(rows=256,sha256=fingerprint,
        tokens=sum(map(len,result['tokens'])),normalization='full saved groups before fixed cut'))+'\n')
    return result


def record_assignment(additions, counts, dp, rank, manifest):
    require_diagnostic()
    import torch.distributed as dist
    root = Path(os.environ['OPENWEBRL_TPDP_OUTPUT'])
    value = dict(dp=dp,rank=rank,counts=counts,windows=additions,
                 total_failure_rows=manifest['total_failure_rows'],coefficient=manifest['coefficient'])
    with (root/f'assignments-rank{dist.get_rank()}.jsonl').open('a') as f:
        f.write(json.dumps(value)+'\n')


def measure_step(fn):
    """Time actual updates, record finite gradients and per-device peak memory."""
    def wrapped(*args, **kwargs):
        require_diagnostic()
        import torch
        dist = torch.distributed
        torch.cuda.synchronize(); dist.barrier(); torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        result = fn(*args, **kwargs)
        torch.cuda.synchronize(); dist.barrier()
        losses, norm = result
        value = dict(seconds=time.monotonic()-start,grad_norm=float(norm),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                     losses={k:float(v) for k,v in losses.items()})
        if not math.isfinite(value['grad_norm']) or any(not math.isfinite(x) for x in value['losses'].values()):
            raise ValueError('Nonfinite diagnostic gradient or loss')
        root = Path(os.environ['OPENWEBRL_TPDP_OUTPUT'])
        with (root/f'updates-rank{dist.get_rank()}.jsonl').open('a') as f:
            f.write(json.dumps(value)+'\n')
        return result
    return wrapped
