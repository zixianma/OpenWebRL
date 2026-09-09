"""Keep slow artifact storage off the browser event loop."""
import asyncio


async def run_artifact_io(function, /, *args, **kwargs):
    """Await a blocking write, including its completion before cancellation exits.

    A cancelled task must not leave its writer racing with outcome validation or
    shutdown. Files still go directly to persistent storage.
    """
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
        worker.result()
        raise
