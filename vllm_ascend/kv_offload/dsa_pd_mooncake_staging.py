# SPDX-License-Identifier: Apache-2.0
"""DSA Host KV landing shared by the Mooncake pull connectors."""

import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend import envs as ascend_envs
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.kv_offload.dsa_pd_mooncake import (
    DSAPDMooncakeTransferTarget,
    create_dsa_pd_mooncake_transfer_targets,
    get_dsa_pd_mooncake_kv_register_location,
    get_dsa_pd_mooncake_offloaded_kv_byte_tensor,
    get_dsa_pd_mooncake_offloaded_kv_tensor,
    prepare_dsa_pd_mooncake_transfer_targets,
)

_DSA_PD_MOONCAKE_LANDING_VIEW_CACHE_SIZE = 256


class _DSAPDMooncakeLandingViews:
    """Bound reusable byte views without owning their source Tensor objects.

    One bank owns each cache; the staging request lock serializes receives.
    Source slices repeat within a
    receive; destination slices are admitted only after a second access, so a
    one-off destination needs no detach. Cached slices are detached so
    their base Tensor can die and clear its entries, while writes still alias
    the original storage. Registry byte views and staging layouts are immutable
    for their Tensor lifetime; a replacement Tensor gets distinct entries.
    """

    def __init__(self, capacity: int = _DSA_PD_MOONCAKE_LANDING_VIEW_CACHE_SIZE) -> None:
        if capacity < 1:
            raise ValueError("Landing view cache capacity must be positive.")
        self.capacity = capacity
        self._views: OrderedDict[
            tuple[int, int, int, bool], tuple[weakref.ReferenceType[torch.Tensor], torch.Tensor]
        ] = OrderedDict()
        self._recent: OrderedDict[tuple[int, int, int, bool], weakref.ReferenceType[torch.Tensor]] = OrderedDict()

    def get(self, tensor: torch.Tensor, offset: int, length: int, *, source: bool = False) -> torch.Tensor:
        key = (id(tensor), offset, length, source)
        entry = self._views.get(key)
        if entry is not None and entry[0]() is tensor:
            self._views.move_to_end(key)
            return entry[1]

        view = tensor[offset : offset + length] if source else tensor.narrow(0, offset, length)
        if not source:
            previous = self._recent.pop(key, None)
            if previous is None or previous() is not tensor:
                self._recent[key] = weakref.ref(tensor)
                if len(self._recent) > self.capacity:
                    self._recent.popitem(last=False)
                return view
        view = view.detach()
        cache_ref = weakref.ref(self)

        def release(tensor_ref: weakref.ReferenceType[torch.Tensor]) -> None:
            cache = cache_ref()
            if cache is not None:
                cached = cache._views.get(key)
                if cached is not None and cached[0] is tensor_ref:
                    cache._views.pop(key, None)

        self._views[key] = (weakref.ref(tensor, release), view)
        if len(self._views) > self.capacity:
            self._views.popitem(last=False)
        return view


@dataclass
class _DSAPDMooncakeStagingBank:
    """Only this bank's executor may mutate its events and view cache.

    Contexts survive receives to reuse views. Pending copies retain their Tensor
    owners until draining succeeds, including when an event/stream wait fails.
    The request lock also protects context replacement and final draining.
    """

    slot_base: int
    slot_count: int
    stream: Any
    views: _DSAPDMooncakeLandingViews | None
    completion_event: Any = None
    completion_pending: bool = False
    inflight_owners: list[tuple[torch.Tensor, ...]] = field(default_factory=list)
    needs_stream_drain: bool = False


@dataclass(frozen=True)
class _DSAPDMooncakeStagingCopy:
    destination: int
    length: int
    staging_offset: int
    target_bytes: torch.Tensor
    target_offset: int


@dataclass(frozen=True)
class _DSAPDMooncakeBankBatch:
    """Prepared under the request lock; address lists are read-only to executors."""

    batch_start: int
    bank_index: int
    local_staging_ptrs: list[int]
    remote_src_ptrs: list[int]
    transfer_lengths: list[int]
    copies: tuple[_DSAPDMooncakeStagingCopy, ...]


@dataclass(frozen=True)
class _DSAPDMooncakeStagingPlan:
    batches: tuple[_DSAPDMooncakeBankBatch, ...]
    # Future schedulers must preserve serial order for overlapping bank writes.
    has_cross_bank_overlap: bool


@dataclass(frozen=True)
class _DSAPDMooncakeBankBatchResult:
    return_code: int
    read_secs: float
    read_settle_secs: float
    land_exposed_secs: float


class DSAPDMooncakeStagingMixin:
    """Own a bounded staging window until every Host KV copy has landed."""

    engine: Any
    kv_caches: dict[str, Any]
    block_len_per_addr: list[list[int]]

    def _init_dsa_pd_mooncake_staging(self) -> None:
        self._dsa_pd_mooncake_staging_storage: torch.Tensor | None = None
        self._dsa_pd_mooncake_staging_tensor: torch.Tensor | None = None
        self._dsa_pd_mooncake_staging_bytes = 0
        self._dsa_pd_mooncake_staging_lock = threading.Lock()
        self._dsa_pd_mooncake_landing_views = _DSAPDMooncakeLandingViews()
        self._dsa_pd_mooncake_staging_banks: tuple[_DSAPDMooncakeStagingBank, ...] = ()
        self._pipeline_banks = ascend_envs.VLLM_ASCEND_DSA_PD_MOONCAKE_BANKS
        self._landing_streams = [torch.npu.Stream() for _ in range(max(1, self._pipeline_banks))]
        self._landing_stream = self._landing_streams[0]
        self._pipeline_landing_enabled = ascend_envs.VLLM_ASCEND_DSA_PD_MOONCAKE_PIPELINE
        # Host-only KV still needs a registered HBM region before ADXL connects.
        self._ensure_dsa_pd_mooncake_staging_tensor()

    @staticmethod
    def _iter_kv_cache_tensors(kv_cache_tuple: Any) -> Iterator[torch.Tensor]:
        if isinstance(kv_cache_tuple, torch.Tensor):
            yield kv_cache_tuple
            return
        if isinstance(kv_cache_tuple, (list, tuple)):
            for item in kv_cache_tuple:
                if isinstance(item, torch.Tensor):
                    yield item

    def _get_dsa_pd_mooncake_staging_device(self) -> torch.device | str:
        for kv_cache_tuple in self.kv_caches.values():
            for tensor in self._iter_kv_cache_tensors(kv_cache_tuple):
                if tensor.device.type != "cpu":
                    return tensor.device
        return "npu"

    def _get_dsa_pd_mooncake_staging_chunk_bytes(self) -> int:
        block_bytes = [
            int(block_len)
            for per_layer_block_lens in self.block_len_per_addr
            for block_len in per_layer_block_lens
            if int(block_len) > 0
        ]
        alignment = 2 * 1024 * 1024
        chunk_bytes = max(max(block_bytes, default=alignment), alignment)
        return ((chunk_bytes + alignment - 1) // alignment) * alignment

    def _get_dsa_pd_mooncake_staging_window(self) -> int:
        return ascend_envs.VLLM_ASCEND_DSA_PD_MOONCAKE_STAGING_WINDOW

    def _ensure_dsa_pd_mooncake_staging_tensor(self) -> torch.Tensor:
        chunk_bytes = self._get_dsa_pd_mooncake_staging_chunk_bytes()
        staging_window = self._get_dsa_pd_mooncake_staging_window()
        required_bytes = chunk_bytes * staging_window
        if self._dsa_pd_mooncake_staging_tensor is not None and self._dsa_pd_mooncake_staging_bytes >= required_bytes:
            return self._dsa_pd_mooncake_staging_tensor

        # Preserve the old allocation if a previous request could not drain it.
        self._drain_dsa_pd_mooncake_staging_banks(getattr(self, "_dsa_pd_mooncake_staging_banks", ()))
        alignment = 2 * 1024 * 1024
        device = self._get_dsa_pd_mooncake_staging_device()
        storage = torch.empty((required_bytes + alignment,), dtype=torch.int8, device=device)
        offset = (-int(storage.data_ptr())) % alignment
        staging = storage[offset : offset + required_bytes]
        if staging.data_ptr() % alignment != 0:
            raise RuntimeError("DSA PD Mooncake NPU staging tensor is not 2MB aligned.")

        global_te.register_buffer([int(staging.data_ptr())], [required_bytes])
        self._dsa_pd_mooncake_staging_storage = storage
        self._dsa_pd_mooncake_staging_tensor = staging
        self._dsa_pd_mooncake_staging_bytes = required_bytes
        logger.info(
            "Allocated Mooncake NPU staging buffer: bytes=%s chunk_bytes=%s window=%s device=%s",
            required_bytes,
            chunk_bytes,
            staging_window,
            staging.device,
        )
        return staging

    def _copy_dsa_pd_mooncake_staging_to_offloaded_kv(
        self,
        dst: int,
        length: int,
        staging_offset: int = 0,
        *,
        staging: torch.Tensor | None = None,
        target_bytes: torch.Tensor | None = None,
        target_offset: int = 0,
        bank: _DSAPDMooncakeStagingBank | None = None,
    ) -> None:
        if target_bytes is None:
            resolved = get_dsa_pd_mooncake_offloaded_kv_byte_tensor(dst, length)
            if resolved is None:
                raise RuntimeError(f"Missing DSA PD Mooncake offloaded KV tensor for dst={dst} length={length}.")
            target_bytes, target_offset = resolved
        if staging is None:
            staging = self._ensure_dsa_pd_mooncake_staging_tensor()
        views = bank.views if bank is not None else getattr(self, "_dsa_pd_mooncake_landing_views", None)
        if views is None and bank is None:
            # Keep compatibility with readers constructed without _init in
            # connector integrations and CPU regression fixtures.
            views = self._dsa_pd_mooncake_landing_views = _DSAPDMooncakeLandingViews()
        if views is None:
            # Unusual configurations with more banks than cache entries keep
            # the same total budget by leaving some banks uncached.
            target_bytes.narrow(0, target_offset, length).copy_(
                staging[staging_offset : staging_offset + length], non_blocking=False
            )
            return
        views.get(target_bytes, target_offset, length).copy_(
            views.get(staging, staging_offset, length, source=True), non_blocking=False
        )

    @staticmethod
    def _coalesce_dsa_pd_mooncake_copy_plan(copy_plan: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        """Merge adjacent plan entries that are contiguous on both sides.

        Consecutive pieces of one layer run are contiguous in the offloaded
        KV pool; when their remote page offsets are zero they are contiguous
        in staging too (consecutive slots), so the whole run lands as one
        copy instead of one copy per 2 MiB piece. Non-contiguous neighbours
        stay separate entries.
        """
        merged: list[tuple[int, int, int]] = []
        for dst, length, staging_offset in copy_plan:
            if merged:
                prev_dst, prev_length, prev_offset = merged[-1]
                if prev_dst + prev_length == dst and prev_offset + prev_length == staging_offset:
                    merged[-1] = (prev_dst, prev_length + length, prev_offset)
                    continue
            merged.append((dst, length, staging_offset))
        return merged

    def _get_dsa_pd_mooncake_staging_banks(
        self, bank_count: int, bank_window: int, pipeline_active: bool
    ) -> tuple[_DSAPDMooncakeStagingBank, ...]:
        streams = self._landing_streams[:bank_count] if pipeline_active else [self._landing_stream]
        if len(streams) != bank_count:
            raise RuntimeError("DSA Mooncake staging bank count exceeds its allocated landing streams.")
        previous = getattr(self, "_dsa_pd_mooncake_staging_banks", ())
        # A failed event wait can leave an old context pending. Finish it before
        # starting another request or dropping the stream/cache ownership.
        self._drain_dsa_pd_mooncake_staging_banks(previous)
        if len(previous) == bank_count and all(
            bank.slot_count == bank_window and bank.stream is stream for bank, stream in zip(previous, streams)
        ):
            return previous
        capacity, extra = divmod(_DSA_PD_MOONCAKE_LANDING_VIEW_CACHE_SIZE, bank_count)
        banks = []
        for index, stream in enumerate(streams):
            budget = capacity + (index < extra)
            banks.append(
                _DSAPDMooncakeStagingBank(
                    slot_base=index * bank_window,
                    slot_count=bank_window,
                    stream=stream,
                    views=_DSAPDMooncakeLandingViews(budget) if budget else None,
                )
            )
        self._dsa_pd_mooncake_staging_banks = tuple(banks)
        # Standalone helper calls use bank zero rather than a second cache budget.
        self._dsa_pd_mooncake_landing_views = banks[0].views
        return self._dsa_pd_mooncake_staging_banks

    def _prepare_dsa_pd_mooncake_bank_plan(
        self,
        staged_transfers: list[tuple[int, int, int]],
        staged_targets: dict[int, tuple[torch.Tensor, int]],
        staging: torch.Tensor,
        staging_chunk_bytes: int,
        banks: tuple[_DSAPDMooncakeStagingBank, ...],
    ) -> _DSAPDMooncakeStagingPlan:
        """Freeze all READ arguments and validate landing bounds before dispatch."""
        staging_ptr = int(staging.data_ptr())
        bank_window = banks[0].slot_count
        batches = []
        intervals = []
        for batch_start in range(0, len(staged_transfers), bank_window):
            bank_index = (batch_start // bank_window) % len(banks)
            bank = banks[bank_index]
            local_staging_ptrs, remote_src_ptrs, transfer_lengths, copy_plan = [], [], [], []
            for position, (local_dst, remote_src, length) in enumerate(
                staged_transfers[batch_start : batch_start + bank_window]
            ):
                staging_offset = (bank.slot_base + position) * staging_chunk_bytes + remote_src % (2 * 1024 * 1024)
                local_staging_ptrs.append(staging_ptr + staging_offset)
                remote_src_ptrs.append(remote_src)
                transfer_lengths.append(length)
                copy_plan.append((local_dst, length, staging_offset))
            copies = []
            for local_dst, length, staging_offset in self._coalesce_dsa_pd_mooncake_copy_plan(copy_plan):
                target_bytes, target_offset = staged_targets[local_dst]
                if target_offset + length > target_bytes.numel():
                    raise RuntimeError("Coalesced DSA Mooncake copy crosses a Host KV tensor boundary.")
                copies.append(_DSAPDMooncakeStagingCopy(local_dst, length, staging_offset, target_bytes, target_offset))
                intervals.append((local_dst, local_dst + length, bank_index))
            batches.append(
                _DSAPDMooncakeBankBatch(
                    batch_start, bank_index, local_staging_ptrs, remote_src_ptrs, transfer_lengths, tuple(copies)
                )
            )
        furthest_end, furthest_bank = -1, -1
        overlap = False
        for start, end, bank_index in sorted(intervals):
            if start < furthest_end and bank_index != furthest_bank:
                overlap = True
                break
            if end > furthest_end:
                furthest_end, furthest_bank = end, bank_index
        return _DSAPDMooncakeStagingPlan(tuple(batches), overlap)

    def _execute_dsa_pd_mooncake_bank_batch(
        self,
        session_id: str,
        bank: _DSAPDMooncakeStagingBank,
        batch: _DSAPDMooncakeBankBatch,
        staging: torch.Tensor,
        pipeline_active: bool,
        remote_request_id: str,
    ) -> _DSAPDMooncakeBankBatchResult:
        """Reclaim, READ, then land one batch without reacquiring the request lock."""
        land_exposed_secs = 0.0
        if pipeline_active:
            reclaim_started = time.perf_counter()
            if bank.completion_pending:
                bank.completion_event.synchronize()
                bank.completion_pending = False
            bank.inflight_owners.clear()
            land_exposed_secs += time.perf_counter() - reclaim_started
        read_started = time.perf_counter()
        ret = self.engine.batch_transfer_sync_read(
            session_id, batch.local_staging_ptrs, batch.remote_src_ptrs, batch.transfer_lengths
        )
        read_secs = time.perf_counter() - read_started
        read_settle_secs = 0.0
        if ret >= 0 and ascend_envs.VLLM_ASCEND_DSA_PD_MOONCAKE_READ_PROBE:
            settle_started = time.perf_counter()
            torch.npu.synchronize()
            read_settle_secs = time.perf_counter() - settle_started
        if ret < 0:
            logger.error(
                "Staged Mooncake batch read failed: request=%s "
                "batch_start=%s batch_size=%s first_local=%s first_remote=%s first_len=%s ret=%s",
                remote_request_id,
                batch.batch_start,
                len(batch.transfer_lengths),
                batch.local_staging_ptrs[0],
                batch.remote_src_ptrs[0],
                batch.transfer_lengths[0],
                ret,
            )
            return _DSAPDMooncakeBankBatchResult(ret, read_secs, read_settle_secs, land_exposed_secs)
        if pipeline_active:
            bank.inflight_owners.append((staging, *(copy.target_bytes for copy in batch.copies)))
            bank.needs_stream_drain = True
            try:
                with torch.npu.stream(bank.stream):
                    for copy in batch.copies:
                        self._copy_dsa_pd_mooncake_staging_to_offloaded_kv(
                            copy.destination,
                            copy.length,
                            copy.staging_offset,
                            staging=staging,
                            target_bytes=copy.target_bytes,
                            target_offset=copy.target_offset,
                            bank=bank,
                        )
                    # All copies precede this marker on the same stream. Reuse
                    # the event only after the previous recording has drained.
                    if bank.completion_event is None:
                        bank.completion_event = torch.npu.Event()
                    bank.completion_event.record(bank.stream)
                    bank.completion_pending = True
                    bank.needs_stream_drain = False
            except Exception:
                bank.stream.synchronize()
                bank.completion_pending = False
                bank.completion_event = None
                bank.needs_stream_drain = False
                raise
        else:
            land_started = time.perf_counter()
            try:
                for copy in batch.copies:
                    self._copy_dsa_pd_mooncake_staging_to_offloaded_kv(
                        copy.destination,
                        copy.length,
                        copy.staging_offset,
                        staging=staging,
                        target_bytes=copy.target_bytes,
                        target_offset=copy.target_offset,
                        bank=bank,
                    )
            finally:
                torch.npu.synchronize()
            land_exposed_secs += time.perf_counter() - land_started
        return _DSAPDMooncakeBankBatchResult(ret, read_secs, read_settle_secs, land_exposed_secs)

    def _drain_dsa_pd_mooncake_staging_banks(self, banks: tuple[_DSAPDMooncakeStagingBank, ...]) -> None:
        first_error = None
        for bank in banks:
            bank_error = None
            try:
                if bank.completion_pending:
                    bank.completion_event.synchronize()
                    bank.completion_pending = False
            except Exception as error:
                bank_error = error
            if bank_error is not None or bank.needs_stream_drain:
                try:
                    bank.stream.synchronize()
                    bank.completion_pending = False
                    # A failed record/wait may leave the event unusable. Drop
                    # it only after the stream confirms its copies are done.
                    bank.completion_event = None
                    bank.needs_stream_drain = False
                except Exception as error:
                    if bank_error is None:
                        bank_error = error
            if not bank.needs_stream_drain and not bank.completion_pending:
                bank.inflight_owners.clear()
            # A failed wait on one bank must not skip the other banks.
            if first_error is None and bank_error is not None:
                first_error = bank_error
        if first_error is not None:
            raise first_error

    def _get_dsa_pd_mooncake_bank_workers(self, banks: tuple[_DSAPDMooncakeStagingBank, ...]) -> int:
        """Use one worker per effective bank; keep a single bank on the caller."""
        return len(banks) if len(banks) > 1 else 0

    def _execute_dsa_pd_mooncake_bank_workers(
        self,
        session_id: str,
        plan: _DSAPDMooncakeStagingPlan,
        banks: tuple[_DSAPDMooncakeStagingBank, ...],
        staging: torch.Tensor,
        remote_request_id: str,
        workers: int,
    ) -> list[tuple[_DSAPDMooncakeBankBatch, _DSAPDMooncakeBankBatchResult]]:
        """Run each bank's queue in one task under the caller's request lock.

        A task keeps reclaim -> READ -> copy order within its bank and advances
        without a coordinator round trip between batches.
        Device setup is thread-local; the staging allocation and all target
        validation are complete before the pool starts. The pool is joined on
        every exit, leaving final stream/event draining to the coordinator.
        """
        bank_batches: list[list[_DSAPDMooncakeBankBatch]] = [[] for _ in banks]
        for batch in plan.batches:
            bank_batches[batch.bank_index].append(batch)
        stopped = threading.Event()
        completed = {}
        failures: dict[int, BaseException] = {}
        pending: dict[Future, int] = {}

        def execute(
            batches: list[_DSAPDMooncakeBankBatch],
        ) -> tuple[
            list[tuple[_DSAPDMooncakeBankBatch, _DSAPDMooncakeBankBatchResult]],
            tuple[int, BaseException] | None,
        ]:
            results = []
            for batch in batches:
                if stopped.is_set():
                    break
                try:
                    result = self._execute_dsa_pd_mooncake_bank_batch(
                        session_id, banks[batch.bank_index], batch, staging, True, remote_request_id
                    )
                except BaseException as error:
                    stopped.set()
                    return results, (batch.batch_start, error)
                results.append((batch, result))
                if result.return_code < 0:
                    stopped.set()
                    break
            return results, None

        pool = ThreadPoolExecutor(
            max_workers=min(workers, len(banks)),
            thread_name_prefix="dsa-mooncake-bank",
            initializer=torch.npu.set_device,
            initargs=(staging.device,),
        )
        try:
            for batches in bank_batches:
                if stopped.is_set():
                    break
                if batches:
                    pending[pool.submit(execute, batches)] = batches[0].batch_start
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in sorted(finished, key=lambda value: pending[value]):
                    batch_start = pending.pop(future)
                    if future.cancelled():
                        continue
                    try:
                        results, failure = future.result()
                    except BaseException as error:
                        failures[batch_start] = error
                        stopped.set()
                    else:
                        for batch, result in results:
                            completed[batch.batch_start] = (batch, result)
                        if failure is not None:
                            failures[failure[0]] = failure[1]
                if stopped.is_set():
                    for future in pending:
                        future.cancel()
        except BaseException:
            stopped.set()
            for future in pending:
                future.cancel()
            raise
        finally:
            # Running READs are allowed to finish. No bank state is drained or
            # made available to another request while its host task can use it.
            pool.shutdown(wait=True, cancel_futures=True)
        if failures:
            raise failures[min(failures)]
        return [completed[index] for index in sorted(completed)]

    def _batch_transfer_sync_read_with_optional_staging(
        self,
        session_id: str,
        src_list: list[int],
        dst_list: list[int],
        length_list: list[int],
        remote_request_id: str,
        *,
        transfer_targets: list[DSAPDMooncakeTransferTarget] | None = None,
    ) -> int:
        if not (len(src_list) == len(dst_list) == len(length_list)):
            raise ValueError("Mooncake READ address and length lists must have equal sizes.")
        if transfer_targets is not None:
            if len(transfer_targets) != len(src_list):
                raise ValueError("Mooncake transfer target count must match address count.")
            requires_staging = any(target.offloaded for target in transfer_targets)
        else:
            requires_staging = any(
                get_dsa_pd_mooncake_offloaded_kv_tensor(local_dst, length) is not None
                for local_dst, length in zip(src_list, length_list)
            )
        target_kwargs = {} if transfer_targets is None else {"transfer_targets": transfer_targets}
        if not requires_staging:
            return self._batch_transfer_sync_read_with_optional_staging_unlocked(
                session_id, src_list, dst_list, length_list, remote_request_id, **target_kwargs
            )

        # The staging slots are shared by all peer handlers on this worker.
        # Keep ownership across both the Mooncake read and the NPU-to-offload
        # copy so another request cannot overwrite a slot still being consumed.
        with self._dsa_pd_mooncake_staging_lock:
            return self._batch_transfer_sync_read_with_optional_staging_unlocked(
                session_id, src_list, dst_list, length_list, remote_request_id, **target_kwargs
            )

    def _batch_transfer_sync_read_with_optional_staging_unlocked(
        self,
        session_id: str,
        src_list: list[int],
        dst_list: list[int],
        length_list: list[int],
        remote_request_id: str,
        *,
        transfer_targets: list[DSAPDMooncakeTransferTarget] | None = None,
    ) -> int:
        # The upstream address builder retains each local KV tensor's identity.
        # Validate the layout once under the staging lock, then use its targets
        # directly instead of rediscovering Host/HBM from every destination.
        resolved_targets = (
            prepare_dsa_pd_mooncake_transfer_targets(transfer_targets, src_list, length_list)
            if transfer_targets is not None
            else None
        )
        direct_src_list: list[int] = []
        direct_dst_list: list[int] = []
        direct_length_list: list[int] = []
        staged_transfers: list[tuple[int, int, int]] = []
        # Request-local plans reuse byte views owned by the KV registry.
        # The registry drops its views on KV Tensor destruction or reset.
        staged_targets: dict[int, tuple[torch.Tensor, int]] = {}
        staging_chunk_bytes = self._get_dsa_pd_mooncake_staging_chunk_bytes()
        staging_alignment = 2 * 1024 * 1024

        for position, (local_dst, remote_src, length) in enumerate(zip(src_list, dst_list, length_list)):
            resolved = (
                resolved_targets[position]
                if resolved_targets is not None
                else get_dsa_pd_mooncake_offloaded_kv_byte_tensor(local_dst, length)
            )
            if resolved is None:
                if resolved_targets is None and get_dsa_pd_mooncake_kv_register_location(local_dst, length) == "cpu":
                    raise RuntimeError("Missing DSA PD Mooncake offloaded KV tensor for a Host READ destination.")
                direct_src_list.append(local_dst)
                direct_dst_list.append(remote_src)
                direct_length_list.append(length)
                continue
            target_tensor, target_offset = resolved
            local_dst = int(local_dst)
            remote_src = int(remote_src)
            remaining = int(length)
            offset = 0
            remote_page_offset = remote_src % staging_alignment
            while remaining > 0:
                piece_capacity = staging_chunk_bytes - remote_page_offset
                piece_len = min(remaining, piece_capacity)
                piece_dst = local_dst + offset
                staged_transfers.append((piece_dst, remote_src + offset, piece_len))
                staged_targets[piece_dst] = (target_tensor, target_offset + offset)
                offset += piece_len
                remaining -= piece_len
                # Chunk sizes are alignment multiples. A nonfinal piece fills
                # its slot, leaving every subsequent source page aligned.
                remote_page_offset = 0

        if direct_src_list:
            ret = self.engine.batch_transfer_sync_read(
                session_id,
                direct_src_list,
                direct_dst_list,
                direct_length_list,
            )
            if ret < 0:
                return ret

        if not staged_transfers:
            return 0

        staging = self._ensure_dsa_pd_mooncake_staging_tensor()
        staging_window = self._get_dsa_pd_mooncake_staging_window()
        # Pipeline mode: WINDOW keeps meaning the total staging capacity and
        # the buffer is split into BANKS banks of window/BANKS slots that
        # consecutive reads round-robin between (the network batch shrinks
        # accordingly; the staging footprint is identical with the pipeline
        # on or off). Flipping to a fresh bank needs no wait at all, so one
        # read's landing hides under the next reads; coming back to a bank
        # BANKS reads later is guarded by a just-in-time bank-event wait. The
        # final drain keeps the external contract: returning 0 means every
        # window has landed and is visible.
        pipeline_active = self._pipeline_landing_enabled and staging_window >= 2
        if self._pipeline_landing_enabled and not pipeline_active:
            logger.warning(
                "VLLM_ASCEND_DSA_PD_MOONCAKE_PIPELINE needs a staging window >= 2 "
                "(got %s); falling back to the synchronous landing path.",
                staging_window,
            )
        banks = self._pipeline_banks if pipeline_active else 1
        if pipeline_active and banks > staging_window:
            logger.warning(
                "VLLM_ASCEND_DSA_PD_MOONCAKE_BANKS=%s exceeds the staging window (%s); clamping banks to %s.",
                banks,
                staging_window,
                staging_window,
            )
            banks = staging_window
        # Trailing slots stay idle when the window is not divisible by the
        # bank count; every bank keeps bank_window contiguous slots.
        bank_window = (staging_window // banks) if pipeline_active else staging_window
        bank_contexts = self._get_dsa_pd_mooncake_staging_banks(banks, bank_window, pipeline_active)
        # Per-receive timing (one INFO line per request): read_s is the
        # network stage, land_exposed_s the landing time spent on the
        # critical path (waits plus, in sync modes, the copies themselves).
        # copies/plan bracket the coalescing attribution: plan counts raw
        # staging-window entries, copies the actual copy_ calls after
        # adjacent runs were merged.
        # Device-side landing duration is deliberately NOT measured with
        # enable_timing events: elapsed_time() can block for a long time on
        # a busy device (observed ~75s stalls), which would poison the very
        # receive it instruments.
        timing_total_start = time.perf_counter()
        timing_read_secs = 0.0
        timing_read_settle_secs = 0.0
        timing_land_exposed_secs = 0.0
        timing_batches = 0
        timing_plan_entries = 0
        timing_copy_calls = 0
        workers = 0
        try:
            plan = self._prepare_dsa_pd_mooncake_bank_plan(
                staged_transfers, staged_targets, staging, staging_chunk_bytes, bank_contexts
            )
            if (
                pipeline_active
                and not plan.has_cross_bank_overlap
                and not ascend_envs.VLLM_ASCEND_DSA_PD_MOONCAKE_READ_PROBE
            ):
                workers = self._get_dsa_pd_mooncake_bank_workers(bank_contexts)
            if workers:
                batch_results = self._execute_dsa_pd_mooncake_bank_workers(
                    session_id, plan, bank_contexts, staging, remote_request_id, workers
                )
            else:
                # Preserve caller-thread READ order for the original control,
                # overlapping destinations, and device-wide READ probe runs.
                def serial_batches() -> Iterator[tuple[_DSAPDMooncakeBankBatch, _DSAPDMooncakeBankBatchResult]]:
                    for batch in plan.batches:
                        if plan.has_cross_bank_overlap:
                            # Serial Host submission alone does not order two
                            # streams writing overlapping destination bytes.
                            self._drain_dsa_pd_mooncake_staging_banks(bank_contexts)
                        yield (
                            batch,
                            self._execute_dsa_pd_mooncake_bank_batch(
                                session_id,
                                bank_contexts[batch.bank_index],
                                batch,
                                staging,
                                pipeline_active,
                                remote_request_id,
                            ),
                        )

                batch_results = serial_batches()
            for batch, result in batch_results:
                timing_read_secs += result.read_secs
                timing_read_settle_secs += result.read_settle_secs
                timing_land_exposed_secs += result.land_exposed_secs
                if result.return_code < 0:
                    return result.return_code
                timing_plan_entries += len(batch.transfer_lengths)
                timing_copy_calls += len(batch.copies)
                timing_batches += 1
        finally:
            if pipeline_active:
                drain_started = time.perf_counter()
                self._drain_dsa_pd_mooncake_staging_banks(bank_contexts)
                timing_land_exposed_secs += time.perf_counter() - drain_started
        logger.info(
            "Staged DSA PD mooncake receive timing: request=%s mode=%s batches=%s chunks=%s "
            "copies=%s plan=%s total_s=%.3f read_s=%.3f read_settle_s=%.3f land_exposed_s=%.3f bank_workers=%s",
            remote_request_id,
            "pipeline" if pipeline_active else "sync",
            timing_batches,
            len(staged_transfers),
            timing_copy_calls,
            timing_plan_entries,
            time.perf_counter() - timing_total_start,
            timing_read_secs,
            timing_read_settle_secs,
            timing_land_exposed_secs,
            workers,
        )
        return 0


class DSAPDMooncakeStagingReader(DSAPDMooncakeStagingMixin):
    """V2 adapter; local destinations are the first address list in READ."""

    def __init__(
        self,
        engine: Any,
        kv_caches: dict[str, Any],
        block_len_per_addr: list[list[int]],
    ) -> None:
        self.engine = engine
        self.kv_caches = kv_caches
        self.block_len_per_addr = block_len_per_addr
        self._init_dsa_pd_mooncake_staging()

    def get_transfer_targets(
        self, layer_names: list[str], base_addresses: list[list[int]]
    ) -> list[list[DSAPDMooncakeTransferTarget]] | None:
        """Bind local layer/cache metadata once; never serialize these handles."""
        if len(layer_names) != len(base_addresses):
            raise ValueError("Mooncake local KV layer names and address metadata differ.")
        layers = []
        for layer_name, addresses in zip(layer_names, base_addresses):
            tensors = list(self._iter_kv_cache_tensors(self.kv_caches[layer_name]))
            if [int(tensor.data_ptr()) for tensor in tensors] != addresses:
                raise ValueError("Mooncake local KV tensor identity differs from address metadata.")
            layers.append(tensors)
        return create_dsa_pd_mooncake_transfer_targets(layers)

    def batch_transfer_sync_read(
        self,
        session_id: str,
        local_destinations: list[int],
        remote_sources: list[int],
        lengths: list[int],
        request_id: str,
        *,
        transfer_targets: list[DSAPDMooncakeTransferTarget] | None = None,
    ) -> int:
        target_kwargs = {} if transfer_targets is None else {"transfer_targets": transfer_targets}
        return self._batch_transfer_sync_read_with_optional_staging(
            session_id, local_destinations, remote_sources, lengths, request_id, **target_kwargs
        )
