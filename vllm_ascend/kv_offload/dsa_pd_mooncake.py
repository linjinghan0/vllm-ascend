# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
import weakref
from bisect import bisect_right
from dataclasses import dataclass
from itertools import accumulate
from typing import Any

import torch

_HOST_CACHE_ALIGNMENT = 2 * 1024 * 1024
_MAX_HOST_COMPONENTS_PER_LAYER = 2
# SFA's allocator reserves a base-alignment chunk and rounds each component
# up to a whole chunk. DSA keeps its separate K/V allocation granularity.
_MAX_ALIGNMENT_OVERHEAD_PER_COMPONENT = 2 * _HOST_CACHE_ALIGNMENT


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.element_size() * tensor.numel())


@dataclass(frozen=True, eq=False)
class DSAPDMooncakeTransferTarget:
    """Local-only KV destination identity captured for one registry layout."""

    generation: int
    base_ptr: int
    nbytes: int
    tensor_ref: weakref.ReferenceType[torch.Tensor]
    offloaded: bool
    tensor_offset: int = 0


class _DSAPDMooncakeRangeIndex:
    """Address-sorted ranges with their original registration positions."""

    def __init__(self, ranges: list[tuple[int, int]]) -> None:
        self.order = sorted(range(len(ranges)), key=lambda position: ranges[position][0])
        self.starts = [ranges[position][0] for position in self.order]
        self.ends = [ranges[position][0] + ranges[position][1] for position in self.order]
        # A predecessor alone is insufficient when registered ranges overlap.
        self.prefix_max_ends = list(accumulate(self.ends, max))


class _DSAPDMooncakeKVRegistry:
    """Process-local bridge between KV allocation and Mooncake setup.

    Tensor references are weak so reinitializing a model runner cannot keep an
    obsolete KV cache alive. Registration happens during startup; connector
    threads only read the registry after initialization. The same worker-local
    object tracks allocation and publishes final views through register_kv_caches,
    matching the SFA initialization lifecycle without owning its decode state.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cpu_buffers: list[tuple[int, int]] = []
        self._offloaded_tensors: list[tuple[int, int, weakref.ReferenceType[torch.Tensor]]] = []
        self._offloaded_byte_views: dict[int, tuple[weakref.ReferenceType[torch.Tensor], torch.Tensor]] = {}
        self._cpu_buffer_index: _DSAPDMooncakeRangeIndex | None = None
        self._offloaded_tensor_index: _DSAPDMooncakeRangeIndex | None = None
        self._generation = 0
        # One cache allocation per worker process. Retain shared backing for
        # its full lifetime, including borrowed NPU views and connector copies.
        self.shared_host_pool = None
        self._pending_offloaded_ranges: list[tuple[int, int]] = []

    @property
    def has_pending_allocations(self) -> bool:
        return bool(self._pending_offloaded_ranges)

    def initialize(self, vllm_config: Any, kv_cache_config: Any, dsa_config: Any, *, num_layers: int) -> None:
        """Prepare the existing worker registry before any raw KV allocation."""
        if getattr(dsa_config, "mode", None) == "fused_overlap":
            self.clear()
        self._pending_offloaded_ranges.clear()
        if getattr(dsa_config, "tp_shared_host_kv", False):
            # Include device-only groups as before, plus the worst-case padding
            # of SFA's aligned allocator for each separately allocated component.
            capacity = sum(tensor.size for tensor in kv_cache_config.kv_cache_tensors)
            capacity += num_layers * _MAX_HOST_COMPONENTS_PER_LAYER * _MAX_ALIGNMENT_OVERHEAD_PER_COMPONENT
            self.initialize_shared_host_pool(vllm_config, capacity)

    def initialize_shared_host_pool(self, vllm_config: Any, capacity_bytes: int) -> None:
        from vllm_ascend.kv_offload.dsa_pd_shared_host import DSAPDSharedHostPool

        if self.shared_host_pool is not None:
            raise RuntimeError("DSA shared Host backing cannot be reinitialized in a live worker")
        config = _get_dsa_sparse_attention_config(vllm_config)
        port = int(_config_get(config, "shared_host_store_port", 8650))
        self.shared_host_pool = DSAPDSharedHostPool(vllm_config, capacity_bytes, port)

    def allocate(self, tensor_size: int, alignment: int) -> torch.Tensor:
        """Allocate Host backing and remember the range awaiting final reshape."""
        if tensor_size <= 0 or alignment <= 0:
            raise ValueError("DSA Host KV size and alignment must be positive")
        if self.shared_host_pool is not None:
            tensor = self.shared_host_pool.allocate(tensor_size, alignment)
        else:
            raw = empty_mooncake_cpu_kv_memory((tensor_size + alignment,), dtype=torch.int8)
            offset = (-raw.data_ptr()) % alignment
            tensor = raw[offset : offset + tensor_size]
        if tensor.data_ptr() % alignment or tensor.numel() != tensor_size:
            raise RuntimeError("DSA PD Mooncake offloaded KV allocation is not correctly aligned and sized")
        # Include all backing padding so the connector cannot classify it as HBM.
        storage = tensor.untyped_storage()
        self.register_cpu_buffer(int(storage.data_ptr()), int(storage.nbytes()))
        self._pending_offloaded_ranges.append((int(tensor.data_ptr()), int(tensor.numel())))
        return tensor

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Bind final compute views before the connector publishes its targets.

        Validate the complete layout before publishing any view. Only tensors
        inside this allocation's Host ranges are registered; indexer HBM and
        other runtime state keep their existing storage and target semantics.
        """
        pending = self._pending_offloaded_ranges
        if not pending:
            return
        matched: set[int] = set()
        tensors: list[torch.Tensor] = []

        def collect(value: Any) -> None:
            if isinstance(value, torch.Tensor):
                ptr, size = int(value.data_ptr()), _tensor_nbytes(value)
                for index, (start, length) in enumerate(pending):
                    if ptr >= start and ptr + size <= start + length:
                        if self.shared_host_pool is not None and value.dtype != torch.bfloat16:
                            raise ValueError("DSA TP-shared Host KV requires BF16 main cache views")
                        tensors.append(value)
                        matched.add(index)
                        return
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        collect(kv_caches)
        if len(matched) != len(pending):
            raise RuntimeError(
                "Failed to bind all DSA PD Mooncake offloaded KV allocations to final runtime tensor views."
            )
        for tensor in tensors:
            self.register_offloaded_tensor(tensor)
        pending.clear()

    def register_cpu_buffer(self, ptr: int, size: int) -> None:
        if ptr <= 0 or size <= 0:
            return
        entry = (int(ptr), int(size))
        with self._lock:
            if entry not in self._cpu_buffers:
                self._cpu_buffers.append(entry)
                self._cpu_buffer_index = None
                self._generation += 1

    def register_offloaded_tensor(self, tensor: torch.Tensor) -> None:
        ptr = int(tensor.data_ptr())
        size = _tensor_nbytes(tensor)
        if ptr <= 0 or size <= 0:
            return

        with self._lock:
            live_entries = []
            already_registered = False
            for start, length, tensor_ref in self._offloaded_tensors:
                existing = tensor_ref()
                if existing is None:
                    continue
                live_entries.append((start, length, tensor_ref))
                already_registered = already_registered or (start == ptr and length == size)
            if not already_registered:
                live_entries.append((ptr, size, weakref.ref(tensor)))
            if not already_registered or len(live_entries) != len(self._offloaded_tensors):
                self._generation += 1
            self._offloaded_tensors = live_entries
            self._offloaded_tensor_index = None

    def create_transfer_targets(
        self, tensor_layers: list[list[torch.Tensor]]
    ) -> list[list[DSAPDMooncakeTransferTarget]] | None:
        """Classify known KV tensors once, without inferring kinds from device.

        Ambiguous aliases and partial registrations retain the address-based
        path: resolving one whole tensor must not change subrange precedence.
        """
        with self._lock:
            live_entries = [
                (start, start + length, tensor)
                for start, length, tensor_ref in self._offloaded_tensors
                if (tensor := tensor_ref()) is not None
            ]
            result = []
            for tensors in tensor_layers:
                targets = []
                for tensor in tensors:
                    start = int(tensor.data_ptr())
                    length = _tensor_nbytes(tensor)
                    if start <= 0 or length <= 0:
                        return None
                    # Direct layouts may have gaps between blocks/heads.
                    span = int(tensor.element_size()) * (
                        1 + sum((dim - 1) * stride for dim, stride in zip(tensor.shape, tensor.stride()))
                    )
                    end = start + span
                    overlaps = [(a, b, t) for a, b, t in live_entries if a < end and start < b]
                    if overlaps:
                        if len(overlaps) != 1:
                            return None
                        base, limit, registered = overlaps[0]
                        if start < base or end > limit or span != length or not registered.is_contiguous():
                            return None
                        target = DSAPDMooncakeTransferTarget(
                            self._generation, start, length, weakref.ref(registered), True, start - base
                        )
                    else:
                        if any(a < end and start < a + size for a, size in self._cpu_buffers):
                            return None
                        target = DSAPDMooncakeTransferTarget(self._generation, start, span, weakref.ref(tensor), False)
                    targets.append(target)
                result.append(targets)
            return result

    def prepare_transfer_targets(
        self, targets: list[DSAPDMooncakeTransferTarget], ptrs: list[int], lengths: list[int]
    ) -> list[tuple[torch.Tensor, int] | None]:
        """Validate local descriptors and obtain views without address lookup."""
        if not len(targets) == len(ptrs) == len(lengths):
            raise ValueError("Mooncake transfer target count must match address and length counts.")
        with self._lock:
            prepared: dict[DSAPDMooncakeTransferTarget, tuple[torch.Tensor]] = {}
            result: list[tuple[torch.Tensor, int] | None] = []
            for target, ptr, length in zip(targets, ptrs, lengths):
                if target.generation != self._generation:
                    raise RuntimeError(
                        "Stale DSA Mooncake transfer targets; recreate the reader after KV layout changes."
                    )
                offset = int(ptr) - target.base_ptr
                if length < 0 or offset < 0 or offset + length > target.nbytes:
                    raise ValueError("Mooncake READ destination crosses its declared KV tensor boundary.")
                cached = prepared.get(target)
                if cached is None:
                    tensor = target.tensor_ref()
                    if tensor is None:
                        raise RuntimeError("DSA Mooncake transfer target tensor has been released.")
                    view = self._get_offloaded_byte_view(tensor) if target.offloaded else tensor
                    cached = (view,)
                    prepared[target] = cached
                result.append((cached[0], target.tensor_offset + offset) if target.offloaded else None)
            return result

    def clear(self) -> None:
        with self._lock:
            self._pending_offloaded_ranges.clear()
            self._cpu_buffers.clear()
            self._offloaded_tensors.clear()
            self._offloaded_byte_views.clear()
            self._cpu_buffer_index = None
            self._offloaded_tensor_index = None
            self._generation += 1

    def resolve_offloaded_byte_tensor(self, ptr: int, size: int) -> tuple[torch.Tensor, int] | None:
        with self._lock:
            resolved = self.resolve_offloaded_tensor(ptr, size)
            if resolved is None:
                return None
            tensor, offset = resolved
            return self._get_offloaded_byte_view(tensor), offset

    def _get_offloaded_byte_view(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reuse a detached view while the caller holds the registry lock."""
        key = id(tensor)
        cached = self._offloaded_byte_views.get(key)
        if cached is None or cached[0]() is not tensor:
            # detach keeps storage alive without owning the registered Tensor.
            byte_view = tensor.view(torch.int8).reshape(-1).detach()

            def release(tensor_ref: weakref.ReferenceType[torch.Tensor]) -> None:
                with self._lock:
                    entry = self._offloaded_byte_views.get(key)
                    if entry is not None and entry[0] is tensor_ref:
                        self._offloaded_byte_views.pop(key)

            cached = (weakref.ref(tensor, release), byte_view)
            self._offloaded_byte_views[key] = cached
        return cached[1]

    def get_register_location(self, ptr: int, size: int) -> str | None:
        ptr = int(ptr)
        size = int(size)
        end = ptr + size
        with self._lock:
            if self._cpu_buffer_index is None:
                self._cpu_buffer_index = _DSAPDMooncakeRangeIndex(self._cpu_buffers)
            index = self._cpu_buffer_index
            position = bisect_right(index.starts, ptr) - 1
            if position >= 0 and end <= index.prefix_max_ends[position]:
                return "cpu"
        return None

    def resolve_offloaded_tensor(
        self,
        ptr: int,
        size: int,
    ) -> tuple[torch.Tensor, int] | None:
        ptr = int(ptr)
        size = int(size)
        end = ptr + size
        resolved = None
        with self._lock:
            if self._offloaded_tensor_index is None:
                self._offloaded_tensor_index = _DSAPDMooncakeRangeIndex(
                    [(start, length) for start, length, _ in self._offloaded_tensors]
                )
            index = self._offloaded_tensor_index
            position = bisect_right(index.starts, ptr) - 1
            first_registration = len(self._offloaded_tensors)
            # Disjoint ranges need at most one candidate check. Overlapping
            # ranges retain the first-registered live Tensor's precedence.
            while position >= 0 and index.prefix_max_ends[position] >= end:
                registration = index.order[position]
                if registration < first_registration and index.ends[position] >= end:
                    start, _, tensor_ref = self._offloaded_tensors[registration]
                    tensor = tensor_ref()
                    if tensor is not None:
                        resolved = (tensor, ptr - start)
                        first_registration = registration
                        if registration == 0:
                            break
                position -= 1
            # Dead weak references are pruned by registration or clear(), not
            # by rebuilding the entire address book on every lookup. The index
            # holds only integer metadata and never owns a Tensor.
        return resolved


_DSA_PD_MOONCAKE_KV_REGISTRY = _DSAPDMooncakeKVRegistry()


def get_dsa_pd_mooncake_kv_registry() -> _DSAPDMooncakeKVRegistry:
    """Return the worker's existing allocation and final-view registration owner."""
    return _DSA_PD_MOONCAKE_KV_REGISTRY


def create_dsa_pd_mooncake_transfer_targets(
    tensor_layers: list[list[torch.Tensor]],
) -> list[list[DSAPDMooncakeTransferTarget]] | None:
    return _DSA_PD_MOONCAKE_KV_REGISTRY.create_transfer_targets(tensor_layers)


def prepare_dsa_pd_mooncake_transfer_targets(
    targets: list[DSAPDMooncakeTransferTarget], ptrs: list[int], lengths: list[int]
) -> list[tuple[torch.Tensor, int] | None]:
    return _DSA_PD_MOONCAKE_KV_REGISTRY.prepare_transfer_targets(targets, ptrs, lengths)


def register_dsa_pd_mooncake_cpu_kv_buffer(ptr: int, size: int) -> None:
    _DSA_PD_MOONCAKE_KV_REGISTRY.register_cpu_buffer(ptr, size)


def register_dsa_pd_mooncake_offloaded_kv_tensor(tensor: torch.Tensor) -> None:
    _DSA_PD_MOONCAKE_KV_REGISTRY.register_offloaded_tensor(tensor)


def reset_dsa_pd_mooncake_kv_registries() -> None:
    _DSA_PD_MOONCAKE_KV_REGISTRY.clear()


def clear_dsa_pd_mooncake_kv_buffer_registries_for_testing() -> None:
    reset_dsa_pd_mooncake_kv_registries()


def get_dsa_pd_mooncake_kv_register_location(ptr: int, size: int) -> str | None:
    return _DSA_PD_MOONCAKE_KV_REGISTRY.get_register_location(ptr, size)


def get_dsa_pd_mooncake_offloaded_kv_tensor(
    ptr: int,
    size: int,
) -> tuple[torch.Tensor, int] | None:
    return _DSA_PD_MOONCAKE_KV_REGISTRY.resolve_offloaded_tensor(ptr, size)


def get_dsa_pd_mooncake_offloaded_kv_byte_tensor(
    ptr: int,
    size: int,
) -> tuple[torch.Tensor, int] | None:
    return _DSA_PD_MOONCAKE_KV_REGISTRY.resolve_offloaded_byte_tensor(ptr, size)


def is_dsa_pd_mooncake_offloaded_kv_buffer(ptr: int, size: int) -> bool:
    return get_dsa_pd_mooncake_offloaded_kv_tensor(ptr, size) is not None


def _iter_connector_names(kv_transfer_config: Any) -> list[str]:
    connector_names: list[str] = []
    connector_name = getattr(kv_transfer_config, "kv_connector", "")
    if connector_name:
        connector_names.append(str(connector_name))

    extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    connectors = extra_config.get("connectors", [])
    if isinstance(connectors, dict):
        connectors = connectors.values()

    for connector in connectors:
        if isinstance(connector, dict):
            nested_name = connector.get("kv_connector", "")
        else:
            nested_name = getattr(connector, "kv_connector", "")
        if nested_name:
            connector_names.append(str(nested_name))

    return connector_names


def kv_transfer_uses_mooncake(kv_transfer_config: Any) -> bool:
    if kv_transfer_config is None:
        return False
    return any("mooncake" in name.lower() for name in _iter_connector_names(kv_transfer_config))


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _get_dsa_sparse_attention_config(vllm_config: Any) -> Any:
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    return _config_get(additional_config, "dsa_sparse_attention_config", {}) or {}


def _is_dsa_sparse_model(vllm_config: Any) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    return bool(
        hf_text_config is not None
        and hasattr(hf_text_config, "index_topk")
        and not hasattr(hf_text_config, "compress_ratios")
    )


def _is_mooncake_kv_consumer(vllm_config: Any) -> bool:
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    return bool(
        kv_transfer_config is not None
        and getattr(kv_transfer_config, "is_kv_consumer", False)
        and kv_transfer_uses_mooncake(kv_transfer_config)
    )


def should_use_dsa_pd_mooncake_long_context_kv(vllm_config: Any) -> bool:
    """Return whether sparse KV capacity is backed by Mooncake CPU offload."""
    dsa_sparse_attention_config = _get_dsa_sparse_attention_config(vllm_config)
    return should_use_dsa_pd_mooncake_cpu_kv(
        vllm_config,
        use_sparse=_is_dsa_sparse_model(vllm_config),
        is_kv_consumer=_is_mooncake_kv_consumer(vllm_config),
        enable_cpu_kv_store=bool(_config_get(dsa_sparse_attention_config, "enable_cpu_kv_store", False)),
        dsa_sparse_attention_mode=_config_get(dsa_sparse_attention_config, "mode", "baseline"),
    )


def should_use_dsa_pd_mooncake_cpu_kv(
    vllm_config: Any,
    *,
    use_sparse: bool,
    is_kv_consumer: bool,
    enable_cpu_kv_store: bool,
    dsa_sparse_attention_mode: str | None = None,
) -> bool:
    if dsa_sparse_attention_mode == "baseline":
        return False
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    use_cpu_kv = bool(
        use_sparse
        and is_kv_consumer
        and enable_cpu_kv_store
        and kv_transfer_config is not None
        and getattr(kv_transfer_config, "is_kv_consumer", False)
        and kv_transfer_uses_mooncake(kv_transfer_config)
    )
    if use_cpu_kv and "MooncakeLayerwiseConnector" in _iter_connector_names(kv_transfer_config):
        raise ValueError(
            "DSA CPU KV offload does not support MooncakeLayerwiseConnector. "
            "Use MooncakeConnectorV1 or MooncakeConnectorV2 for request-level transfers."
        )
    return use_cpu_kv


def empty_mooncake_cpu_kv_memory(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    try:
        import torch_npu
    except ImportError as e:
        raise RuntimeError("DSA PD Mooncake offloaded KV requires torch_npu.empty_with_swapped_memory.") from e

    empty_with_swapped_memory = getattr(torch_npu, "empty_with_swapped_memory", None)
    if empty_with_swapped_memory is None:
        raise RuntimeError("DSA PD Mooncake offloaded KV requires torch_npu.empty_with_swapped_memory.")
    return empty_with_swapped_memory(shape, dtype=dtype, device="npu")


def get_dsa_pd_shared_host_pool():
    return _DSA_PD_MOONCAKE_KV_REGISTRY.shared_host_pool


def initialize_dsa_pd_shared_host_pool(vllm_config: Any, capacity_bytes: int) -> None:
    _DSA_PD_MOONCAKE_KV_REGISTRY.initialize_shared_host_pool(vllm_config, capacity_bytes)
