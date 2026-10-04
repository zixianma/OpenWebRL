"""Await durable synchronous writes without blocking the rollout event loop."""
import asyncio


async def run_durable_io(function, *args):
    """Finish the write before propagating cancellation to its owner.

    A thread cannot be cancelled while inside a filesystem call. Shield its
    task and retain ownership even if the caller is cancelled repeatedly.
    Exceptions from the write itself are propagated normally.
    """
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    cancellation = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            cancellation = exc
    if cancellation is not None:
        raise cancellation
    return result


async def log_progress_async(current, log_progress):
    # One telemetry writer owns the shared .partial path. Different trajectory
    # label files are independent and need no common filesystem lock.
    lock = current.setdefault('_progress_io_lock', asyncio.Lock())
    async with lock:
        await run_durable_io(log_progress, current)
