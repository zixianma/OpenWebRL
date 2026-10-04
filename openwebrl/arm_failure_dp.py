"""Distribute auxiliary ARM rows without changing the global PPO objective."""


def shard_windows(additions, dp, rank):
    """Each real row appears once; zero-loss padding equalizes DP collectives."""
    if dp not in (1, 2, 4) or not 0 <= rank < dp:
        raise ValueError('ARM auxiliary transport supports DP1/2/4 only')
    result = []
    for rows in additions:
        local = [(i, scale, False) for i, scale in rows[rank::dp]]
        target = (len(rows) + dp - 1) // dp
        local.extend([(rows[0][0], 0., True)] * (target - len(local)) if rows else [])
        result.append(local)
    return result
