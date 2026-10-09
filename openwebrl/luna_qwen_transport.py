"""Transport isolation for one browser and up to ten concurrent candidates."""
import asyncio
import time


def init_candidate_http_client(args, initializer):
    if args.rollout_num_gpus != 1 or args.rollout_num_gpus_per_engine != 1:
        raise ValueError('Candidate transport requires one dedicated model GPU')
    args.sglang_server_concurrency = 10
    initializer(args)


async def flush_cache_when_idle(client, url, timeout=60., poll=.25):
    """A cancelled request may need a moment to leave the server scheduler."""
    start=time.monotonic(); attempts=0
    while True:
        remaining=timeout-(time.monotonic()-start)
        if remaining<=0:
            raise TimeoutError('Actor requests did not drain before the next episode')
        response=await client.post(url,timeout=min(10.,remaining));attempts+=1
        if response.status_code==200:
            return dict(attempts=attempts,seconds=time.monotonic()-start,cache_flushed=True)
        if response.status_code!=400 or 'running or waiting requests' not in response.text:
            response.raise_for_status()
            raise ValueError('Unexpected cache-flush response')
        await asyncio.sleep(min(poll,max(0.,timeout-(time.monotonic()-start))))
