# SPDX-License-Identifier: Apache-2.0
"""Original DSA allocation and final-view contracts used by the V2 runner."""

import ctypes
import gc
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.kv_offload import dsa_pd_mooncake as dsa
from vllm_ascend.kv_offload.dsa_pd_shared_host import DSAPDSharedHostPool


def make_shared_pool(rank=0, *, capacity=1 << 20, shared=None):
    """Use real CPU storage in place of memfabric's imported device storage."""
    shared = {} if shared is None else shared

    def empty(shape, *, dtype):
        tensor = torch.zeros(shape, dtype=dtype)
        shared[tensor.data_ptr()] = tensor.untyped_storage()
        return tensor

    def broadcast(metadata, src):
        assert src == 0
        if rank == 0:
            shared["metadata"] = metadata.clone()
        else:
            metadata.copy_(shared["metadata"])

    pool = object.__new__(DSAPDSharedHostPool)
    pool.rank, pool.size = rank, 2
    pool.device = torch.device("cpu")
    pool.capacity, pool._allocated = capacity, 0
    pool._owners, pool._views = [], []
    pool._starts, pool._ends = [], []
    pool.group = SimpleNamespace(broadcast=broadcast)
    pool._torch_npu = SimpleNamespace(
        _C=SimpleNamespace(_construct_storage_from_data_pointer=lambda ptr, _device, _size: shared[ptr])
    )
    pool._offload = SimpleNamespace(empty=MagicMock(side_effect=empty))
    return pool


@pytest.mark.parametrize("size", [1, 64, 65, 4095, 4096, 4097])
def test_tp_owner_keeps_original_allocation_and_all_ranks_borrow_the_same_view(size):
    shared = {}
    owner = make_shared_pool(shared=shared)
    borrower = make_shared_pool(rank=1, shared=shared)
    first = owner.allocate(size, 64)
    second = borrower.allocate(size, 64)
    needed = size + 64
    owner._offload.empty.assert_called_once_with([needed], dtype=torch.int8)
    borrower._offload.empty.assert_not_called()
    assert first.data_ptr() == second.data_ptr()
    assert owner._owners[0].data_ptr() <= first.data_ptr() < owner._owners[0].data_ptr() + 64
    assert first.numel() == second.numel() == size
    assert first.data_ptr() % 64 == 0
    assert owner._allocated == borrower._allocated == needed
    assert owner.contains(first.data_ptr(), size)
    assert borrower.contains(second.data_ptr(), size)
    assert not owner.contains(first.data_ptr() + size)
    first.fill_(17)
    assert second.tolist() == [17] * size
    assert len(owner._owners) == 1 and borrower._owners == []


def test_padding_capacity_is_checked_before_allocating_or_broadcasting():
    pool = make_shared_pool(capacity=128)
    pool.group = MagicMock()
    with pytest.raises(ValueError, match="planned capacity"):
        pool.allocate(65, 64)  # The original DSA allocation needs 129 bytes.
    pool._offload.empty.assert_not_called()
    pool.group.broadcast.assert_not_called()


def test_initialization_budget_keeps_original_alignment_reserve_per_layer():
    registry = dsa._DSAPDMooncakeKVRegistry()
    config = SimpleNamespace(additional_config={"dsa_sparse_attention_config": {"shared_host_store_port": 8651}})
    cache = SimpleNamespace(kv_cache_tensors=[SimpleNamespace(size=1), SimpleNamespace(size=3)])
    with patch("vllm_ascend.kv_offload.dsa_pd_shared_host.DSAPDSharedHostPool") as pool:
        registry.initialize(config, cache, SimpleNamespace(mode="fused_overlap", tp_shared_host_kv=True), num_layers=2)
    pool.assert_called_once_with(config, 4 + 2 * 2 * (2 << 20), 8651)
    with pytest.raises(RuntimeError, match="reinitialized"):
        registry.initialize_shared_host_pool(config, 1 << 30)


def test_staging_lands_in_the_registered_compute_view_and_keeps_indexer_local():
    pool = make_shared_pool()
    registry = dsa._DSAPDMooncakeKVRegistry()
    registry.shared_host_pool = pool
    raw = registry.allocate(32, 64)
    main = raw.view(torch.bfloat16).reshape(2, 8)
    indexer = torch.arange(8, dtype=torch.int8)
    before = indexer.clone()
    registry.register_kv_caches({"main": (main,), "alias": [main], "indexer": (indexer,)})
    assert not registry.has_pending_allocations
    targets = registry.create_transfer_targets([[main], [indexer]])
    assert targets is not None and targets[0][0].offloaded and not targets[1][0].offloaded
    prepared = registry.prepare_transfer_targets([targets[0][0]], [main.data_ptr()], [32])
    assert prepared[0][0].data_ptr() == main.data_ptr()
    assert pool._owners[0].data_ptr() <= main.data_ptr() < pool._owners[0].data_ptr() + 64

    def sparse_copy(sources, destinations, lengths, _count, _device):
        for source, destination, length in zip(sources.tolist(), destinations.tolist(), lengths.tolist()):
            ctypes.memmove(destination, source, length)
        return 0

    pool._offload.sparse_copy = sparse_copy
    staging = torch.arange(32, dtype=torch.int8)
    owners = []
    pool.land([SimpleNamespace(destination=main.data_ptr(), length=32, staging_offset=0)], staging, owners)
    assert torch.equal(main.view(torch.int8).flatten(), staging)
    assert torch.equal(indexer, before)
    assert len(owners) == 1


@pytest.mark.parametrize("invalid", ["missing", "dtype"])
def test_failed_registration_publishes_no_partial_layout_and_can_be_retried(invalid):
    registry = dsa._DSAPDMooncakeKVRegistry()
    with patch.object(dsa, "empty_mooncake_cpu_kv_memory", side_effect=lambda shape, **kw: torch.zeros(shape, **kw)):
        first, second = registry.allocate(32, 64), registry.allocate(32, 64)
    good = [first.view(torch.bfloat16), second.view(torch.bfloat16)]
    registry.shared_host_pool = object()
    bad = {"a": good[0]} if invalid == "missing" else {"a": good[0], "b": second}
    with pytest.raises((ValueError, RuntimeError)):
        registry.register_kv_caches(bad)
    assert registry.resolve_offloaded_tensor(first.data_ptr(), 32) is None
    assert registry.has_pending_allocations
    registry.register_kv_caches({"a": good[0], "b": good[1]})
    assert registry.resolve_offloaded_tensor(first.data_ptr(), 32)[0] is good[0]
    assert not registry.has_pending_allocations


def test_final_view_lifetime_and_registry_reset_invalidate_connector_targets():
    registry = dsa._DSAPDMooncakeKVRegistry()
    with patch.object(dsa, "empty_mooncake_cpu_kv_memory", side_effect=lambda shape, **kw: torch.zeros(shape, **kw)):
        raw = registry.allocate(32, 64)
    final = raw.reshape(2, 16)
    registry.register_kv_caches({"main": final})
    target = registry.create_transfer_targets([[final]])[0][0]
    ref = weakref.ref(final)
    registry.clear()
    with pytest.raises(RuntimeError, match="Stale"):
        registry.prepare_transfer_targets([target], [raw.data_ptr()], [32])
    del final
    gc.collect()
    assert ref() is None
    assert not registry.has_pending_allocations
