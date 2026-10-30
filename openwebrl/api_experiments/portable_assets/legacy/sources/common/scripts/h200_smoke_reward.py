"""Synthetic alternating rewards ONLY for pipeline tests; no task-quality meaning."""


async def reward(args, sample, **kwargs):
    if isinstance(sample, list):
        return [await reward(args, item, **kwargs) for item in sample]
    trajectory_id = (sample.metadata or {}).get('trajectory_id', sample.index)
    return float(trajectory_id % 2)
