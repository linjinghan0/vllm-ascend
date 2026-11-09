# SPDX-License-Identifier: Apache-2.0
"""Adapt Mooncake V2 transport to the existing SFA offload cache owners.

The manager owns all Host and hot KV. This adapter owns only transfer views,
bounded landing descriptors and the PD-to-decode readiness bookkeeping.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import (
    copy_sfa_prefill_dest_geometry,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    OFFLOAD_KV_CACHE_TUPLE_LEN,
)
from vllm_ascend.kv_offload.dsa_pd_mooncake import _DSAPDMooncakeKVRegistry
from vllm_ascend.kv_offload.dsa_pd_mooncake_staging import DSAPDMooncakeStagingReader

_SPARSE_COPY_MAX_BYTES = (1 << 31) - 1


class MooncakeSFAHostKV:
    """Expose persistent manager-owned caches to the V2 address planner."""

    def __init__(self, manager: Any, kv_caches: dict[str, Any]) -> None:
        self.manager = manager
        self.rank = manager.tp_rank
        self.size = manager.tp_group.world_size
        main_groups = {
            group_id
            for group_id, group in enumerate(manager.kv_cache_config.kv_cache_groups)
            if set(group.layer_names) & set(manager.offload_layer_names)
        }
        if len(main_groups) != 1:
            raise ValueError("Mooncake V2 SFA requires one main KV cache group")
        self.main_group_id = main_groups.pop()
        # Reuse transfer identity/range validation without allocating a DSA pool
        # or publishing the SFA views into the legacy process-global registry.
        self.registry = _DSAPDMooncakeKVRegistry()
        self.transfer_caches = dict(kv_caches)
        for name in manager.offload_layer_names:
            if len(kv_caches[name]) != OFFLOAD_KV_CACHE_TUPLE_LEN:
                raise ValueError(f"Mooncake V2 SFA layer {name!r} must expose the SFA six-component cache")
            layer_id = manager._get_offload_layer_id(name)
            pair = (manager.k_caches_cpu[layer_id], manager.v_caches_cpu[layer_id])
            for tensor in pair:
                if tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                    raise ValueError("Mooncake V2 SFA Host KV must be contiguous BF16 CPU views")
                storage = tensor.untyped_storage()
                self.registry.register_cpu_buffer(int(storage.data_ptr()), int(storage.nbytes()))
                self.registry.register_offloaded_tensor(tensor)
            # Placeholders and hot rows are execution state, never PD payload.
            self.transfer_caches[name] = pair

    def contains(self, ptr: int, size: int = 1) -> bool:
        return self.registry.resolve_offloaded_tensor(ptr, size) is not None

    def land(self, copies: Any, staging: torch.Tensor, owners: list[Any]) -> None:
        """Batch a bank's landing through the manager, preserving its event lifetime."""
        if not copies:
            return
        for copy in copies:
            if not 0 < copy.length <= _SPARSE_COPY_MAX_BYTES or not self.contains(copy.destination, copy.length):
                raise ValueError("Mooncake V2 SFA landing exceeds a Host KV range or the sparse-copy int32 ABI")
        addresses = torch.tensor(
            [[staging.data_ptr() + copy.staging_offset for copy in copies], [copy.destination for copy in copies]],
            dtype=torch.int64,
            device=staging.device,
        )
        lengths = torch.tensor([copy.length for copy in copies], dtype=torch.int32, device=staging.device)
        count = torch.tensor([len(copies)], dtype=torch.int32, device=staging.device)
        # sparse_copy may enqueue before reporting failure. The existing bank
        # drain keeps these descriptors alive on both success and failure.
        owners.append((addresses, lengths, count))
        self.manager.copy_sfa_kv(*addresses.unbind(0), lengths, count)


class MooncakeSFAStagingReader(DSAPDMooncakeStagingReader):
    """Keep V2's batching/pipeline; substitute SFA-owned destination and copy APIs."""

    def __init__(self, engine: Any, host_kv: MooncakeSFAHostKV, block_lens: list[list[int]]) -> None:
        self.host_kv = host_kv
        super().__init__(engine, host_kv.transfer_caches, block_lens)

    def _resolve_host_tensor(self, ptr, size):
        return self.host_kv.registry.resolve_offloaded_tensor(ptr, size)

    def _resolve_host_bytes(self, ptr, size):
        return self.host_kv.registry.resolve_offloaded_byte_tensor(ptr, size)

    def _host_register_location(self, ptr, size):
        return self.host_kv.registry.get_register_location(ptr, size)

    def _prepare_host_targets(self, targets, ptrs, lengths):
        return self.host_kv.registry.prepare_transfer_targets(targets, ptrs, lengths)

    def _create_host_targets(self, layers):
        return self.host_kv.registry.create_transfer_targets(layers)

    def get_shared_host_pool(self):
        return self.host_kv


class MooncakeSFAPDState:
    """Track received prefixes until MRV2 initializes their existing SFA rows."""

    def __init__(self, main_group_id: int) -> None:
        self.main_group_id = main_group_id
        self.bindings: dict[str, int] = {}
        self.pending: set[str] = set()
        self.received: set[str] = set()

    def update(self, bindings: dict[str, int], new_requests: Sequence[str]) -> None:
        if len(set(bindings.values())) != len(bindings):
            raise ValueError("Mooncake V2 SFA requests cannot share a hot row")
        for req_id, slot in self.bindings.items():
            if req_id in bindings and bindings[req_id] != slot:
                raise ValueError("Mooncake V2 SFA cannot rebind a live request")
        if not set(new_requests) <= bindings.keys():
            raise ValueError("Mooncake V2 SFA transfer is missing its reserved hot row")
        newly_bound = set(bindings) - self.bindings.keys()
        self.bindings = dict(bindings)
        self.pending.intersection_update(bindings)
        self.received.intersection_update(bindings)
        self.pending.update(newly_bound | set(new_requests))
        # A complete local prefix hit needs no READ, but its new hot row still
        # needs initialization from the retained Host blocks before decoding.
        self.received.update(newly_bound - set(new_requests))
        self.received.difference_update(new_requests)

    def record_received(self, finished: set[str], failed: set[str]) -> None:
        self.received.update((finished - failed) & self.pending)
        self.received.difference_update(failed)

    def prepare(
        self,
        req_ids: Sequence[str],
        slots: Sequence[int],
        computed_tokens: Sequence[int],
        *,
        block_size: int,
        hot_tokens: int,
    ) -> tuple[bool, dict[int, tuple[int, int]], set[str]]:
        restore_tails = False
        dense_fills: dict[int, tuple[int, int]] = {}
        initialized: set[str] = set()
        for row, req_id in enumerate(req_ids):
            if req_id not in self.pending:
                continue
            if req_id not in self.received:
                raise RuntimeError("Mooncake V2 SFA decode started before Host KV receive completed")
            if int(slots[row]) != self.bindings[req_id]:
                raise RuntimeError("Mooncake V2 SFA decode row differs from its PD reservation")
            computed = int(computed_tokens[row])
            if computed < 0:
                raise ValueError("Mooncake V2 SFA computed-token count must be nonnegative")
            dense, tail_tokens, _ = copy_sfa_prefill_dest_geometry(computed, block_size, hot_tokens)
            if dense and computed:
                dense_fills[int(slots[row])] = (row, computed)
            restore_tails |= tail_tokens > 0 and not dense
            initialized.add(req_id)
        return restore_tails, dense_fills, initialized

    def commit(self, initialized: set[str]) -> None:
        self.pending.difference_update(initialized)
        self.received.difference_update(initialized)
