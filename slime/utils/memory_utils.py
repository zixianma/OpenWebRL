import gc
import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def clear_memory(clear_host_memory: bool = False):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info


def release_unused_cuda_cache_under_pressure():
    """Release cached blocks before variable-size microbatches exhaust HBM.

    The preload allocator exits on failed cuMemCreate, bypassing PyTorch's
    normal OOM/cache-release retry. Live tensors are never freed. Enable on
    affected runtimes with OPENWEBRL_CUDA_CACHE_LIMIT_GIB.
    """
    limit_gib = float(os.environ.get("OPENWEBRL_CUDA_CACHE_LIMIT_GIB", "0"))
    if limit_gib <= 0:
        return
    reserved = torch.cuda.memory_reserved()
    if reserved < limit_gib * 1024**3:
        return
    allocated = torch.cuda.memory_allocated()
    if reserved - allocated < 8 * 1024**3:
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    logger.info(
        "CUDA cache pressure release: allocated_GiB=%.2f reserved_before_GiB=%.2f reserved_after_GiB=%.2f",
        allocated / 1024**3, reserved / 1024**3, torch.cuda.memory_reserved() / 1024**3,
    )
