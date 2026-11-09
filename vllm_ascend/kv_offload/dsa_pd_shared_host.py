# SPDX-License-Identifier: Apache-2.0
"""TP-shared Host backing for the eager Mooncake DSA consumer.

The worker's existing KV registry owns this pool for its entire lifetime.
Only TP0 owns CPU allocations; all ranks borrow device views of their shared
GVAs. Selection/indexer caches remain rank-local. This opt-in implementation
requires a single-host TP group, eager V2, and no context/pipeline parallelism.
"""

from __future__ import annotations

import socket
from bisect import bisect_right
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

_GIB = 1 << 30
_SPARSE_COPY_MAX_BYTES = (1 << 31) - 1


def partition_shared_block_groups(
    local_groups: list[list[int]],
    remote_groups: list[list[int]],
    rank: int,
    size: int,
    *,
    coalesce: bool = True,
) -> tuple[list[list[int]], list[list[int]]]:
    """Assign balanced consecutive ranges of sorted destination blocks.

    Keep every source/destination pair together when sorting. Repeated
    destinations stay on one rank and preserve their original source order.
    The caller disables coalescing when adjacent block IDs do not describe
    adjacent byte ranges, for example with strided or partial-block copies.
    """
    if size < 1 or not 0 <= rank < size:
        raise ValueError("Invalid TP ownership group")
    if len(local_groups) != len(remote_groups):
        raise ValueError("Shared Host block group counts differ")
    sources_by_destination: dict[int, list[int]] = {}
    for local, remote in zip(local_groups, remote_groups):
        if len(local) != len(remote):
            raise ValueError("Shared Host block group lengths differ")
        for dst, src in zip(local, remote):
            if dst < 0 or src < 0:
                raise ValueError("Shared Host block IDs must be nonnegative")
            sources_by_destination.setdefault(dst, []).append(src)
    destinations = sorted(sources_by_destination)
    blocks_per_rank, remainder = divmod(len(destinations), size)
    start = rank * blocks_per_rank + min(rank, remainder)
    count = blocks_per_rank + int(rank < remainder)
    local_result: list[list[int]] = []
    remote_result: list[list[int]] = []
    for dst in destinations[start : start + count]:
        for src in sources_by_destination[dst]:
            if coalesce and local_result and dst == local_result[-1][-1] + 1 and src == remote_result[-1][-1] + 1:
                local_result[-1].append(dst)
                remote_result[-1].append(src)
            else:
                local_result.append([dst])
                remote_result.append([src])
    return local_result, remote_result


class DSAPDSharedHostPool:
    """Own shared allocations, borrowed views and one ordered write fence."""

    def __init__(self, vllm_config: Any, capacity_bytes: int, store_port: int):
        # The optional transport provisions native resources. Import it only
        # after the caller has explicitly opted into this allocation path.
        import torch
        import torch.distributed as dist
        import torch_npu
        from memfabric_hybrid import offload
        from vllm.distributed import get_tp_group
        from vllm.logger import logger

        parallel = vllm_config.parallel_config
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("DSA TP-shared Host KV currently requires enforce_eager")
        if any(
            getattr(parallel, name, 1) != 1
            for name in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")
        ):
            raise ValueError("DSA TP-shared Host KV requires PP=PCP=DCP=1")
        if parallel.data_parallel_size != 1:
            raise ValueError("DSA TP-shared Host KV currently requires DP=1")
        model = vllm_config.model_config
        hf = model.hf_text_config
        if (
            model.dtype != torch.bfloat16
            or not hasattr(hf, "index_topk")
            or hasattr(hf, "compress_ratios")
            or getattr(hf, "qk_rope_head_dim", 0) <= 0
        ):
            raise ValueError("DSA TP-shared Host KV currently requires BF16 split MLA main KV with a sparse indexer")
        if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise ValueError("DSA TP-shared Host KV does not support quantized main KV")
        self.group = get_tp_group()
        self.rank = self.group.rank_in_group
        self.size = self.group.world_size
        hosts: list[str | None] = [None] * self.size
        dist.all_gather_object(hosts, socket.gethostname(), group=self.group.cpu_group)
        if len(set(hosts)) != 1:
            raise ValueError("DSA TP-shared Host KV requires all TP ranks on one host")
        if not hasattr(torch_npu._C, "_construct_storage_from_data_pointer"):
            raise RuntimeError("Installed TorchNPU cannot borrow a shared Host device view")
        self.device = torch.device("npu", torch.npu.current_device())
        self._torch_npu = torch_npu
        self._offload = offload
        self.capacity = ((capacity_bytes + _GIB - 1) // _GIB) * _GIB
        config = offload.OffloadConfig()
        config.device_id = self.device.index
        config.reserve_size = self.capacity
        config.alloc_size = self.capacity if self.rank == 0 else 0
        config.world_size = self.size
        config.rank_id = self.rank
        config.scene = offload.Scene.SHARED
        config.store_url = f"tcp://127.0.0.1:{store_port}"
        if offload.initialize(config) != 0:
            raise RuntimeError("Could not initialize the DSA shared Host pool")
        self.group.barrier()
        self._owners: list[torch.Tensor] = []
        self._views: list[torch.Tensor] = []
        self._starts: list[int] = []
        self._ends: list[int] = []
        self._allocated = 0
        self._masked_slots: dict[tuple[Any, ...], torch.Tensor] = {}
        self._fence = torch.zeros(1, dtype=torch.int32, device=self.device)
        logger.info(
            "DSA TP-shared Host pool: rank=%s/%s reserved_bytes=%s owner_rank=0", self.rank, self.size, self.capacity
        )

    def allocate(self, size: int, alignment: int) -> torch.Tensor:
        import torch

        needed = size + alignment
        if size <= 0 or alignment <= 0 or self._allocated + needed > self.capacity:
            raise ValueError("DSA shared Host allocation exceeds its planned capacity")
        host = self._offload.empty([needed], dtype=torch.int8) if self.rank == 0 else None
        ptr = 0 if host is None else host.data_ptr()
        base = torch.tensor([ptr, needed], dtype=torch.int64, device=self.device)
        self.group.broadcast(base, src=0)
        shared_ptr, actual_size = map(int, base.cpu().tolist())
        if shared_ptr <= 0 or actual_size != needed:
            raise RuntimeError("TP ranks disagree about a DSA shared allocation")
        # This is the same non-owning storage constructor used by TorchNPU's
        # graph tensor reconstruction. The pool retains the true CPU owner.
        storage = self._torch_npu._C._construct_storage_from_data_pointer(shared_ptr, self.device, needed)
        raw = torch.empty(0, dtype=torch.int8, device=self.device).set_(storage, 0, (needed,), (1,))
        offset = (-shared_ptr) % alignment
        view = raw[offset : offset + size]
        self._register_range(view.data_ptr(), size)
        if host is not None:
            self._owners.append(host)
        self._views.append(view)
        self._allocated += needed
        return view

    def _register_range(self, ptr: int, length: int) -> None:
        index = bisect_right(self._starts, ptr)
        end = ptr + length
        if ptr <= 0 or length <= 0:
            raise ValueError("Invalid shared Host allocation range")
        if (index and self._ends[index - 1] > ptr) or (index < len(self._starts) and end > self._starts[index]):
            raise RuntimeError("Shared Host allocations overlap")
        self._starts.insert(index, ptr)
        self._ends.insert(index, end)

    def contains(self, ptr: int, length: int = 1) -> bool:
        index = bisect_right(self._starts, ptr) - 1
        return length > 0 and index >= 0 and ptr + length <= self._ends[index]

    def write_slots(self, slots: torch.Tensor) -> torch.Tensor:
        """Only TP0 writes newly computed main KV; other ranks still compute Q."""
        if self.rank == 0:
            return slots
        key = (tuple(slots.shape), slots.dtype, slots.device)
        masked = self._masked_slots.get(key)
        if masked is None:
            import torch

            masked = torch.full_like(slots, -1)
            self._masked_slots[key] = masked
        return masked

    def fence_writes(self) -> None:
        # The device collective orders each rank's current stream, including
        # the owner's cache write, before any rank's attention reads the pool.
        # This fence is deliberately kept in the correctness-first version.
        self._fence = self.group.all_reduce(self._fence)

    def land(self, copies: Any, staging: torch.Tensor, owners: list[Any]) -> None:
        """Use the shared-memory copy kernel, not raw ACL memcpy on an imported GVA."""
        if not copies:
            return
        import torch

        for copy in copies:
            if copy.length > _SPARSE_COPY_MAX_BYTES:
                raise ValueError("Shared Host landing length exceeds the sparse-copy int32 ABI")
            if not self.contains(copy.destination, copy.length):
                raise ValueError("Shared Host landing is outside the allocated KV pool")
        addresses = torch.tensor(
            [
                [staging.data_ptr() + copy.staging_offset for copy in copies],
                [copy.destination for copy in copies],
            ],
            dtype=torch.int64,
            device=self.device,
        )
        # The native ABI reads 64-bit addresses but 32-bit byte lengths.
        # An int64 length array silently interleaves zero high words between
        # lengths and skips every second copy in a multi-descriptor batch.
        lengths = torch.tensor([copy.length for copy in copies], dtype=torch.int32, device=self.device)
        count = torch.tensor([len(copies)], dtype=torch.int32, device=self.device)
        # Retain these before calling the native API as it may enqueue work
        # before returning an error. The caller drains the bank on failure.
        owners.append((addresses, lengths, count))
        result = self._offload.sparse_copy(*addresses.unbind(0), lengths, count, self.device)
        if result not in (None, 0):
            raise RuntimeError(f"DSA shared Host landing failed: {result}")
