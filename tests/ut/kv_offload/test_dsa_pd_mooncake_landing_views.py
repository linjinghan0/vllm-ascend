# SPDX-License-Identifier: Apache-2.0
"""CPU byte aliasing, lifetime, and capacity checks for landing view reuse."""

import gc
import weakref
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.staging import _DSAPDMooncakeLandingViews

from .test_dsa_pd_mooncake_connector_v2 import make_reader


@pytest.mark.parametrize("source", [False, True])
def test_reused_views_alias_live_bytes(source):
    cache = _DSAPDMooncakeLandingViews()
    tensor = torch.arange(32, dtype=torch.int8)
    cache.get(tensor, 4, 12, source=source)
    first = cache.get(tensor, 4, 12, source=source)
    second = cache.get(tensor, 4, 12, source=source)
    assert first is second
    tensor[4:16].fill_(7)
    assert torch.all(second == 7)
    second.fill_(9)
    assert torch.all(tensor[4:16] == 9)
    assert torch.equal(tensor[:4], torch.arange(4, dtype=torch.int8))


@pytest.mark.parametrize("source", [False, True])
def test_cache_drops_views_when_owner_dies(source):
    cache = _DSAPDMooncakeLandingViews()
    tensor = torch.empty(32, dtype=torch.int8)
    cache.get(tensor, 4, 12, source=source)
    view = cache.get(tensor, 4, 12, source=source)
    tensor_ref, view_ref = weakref.ref(tensor), weakref.ref(view)
    del tensor, view
    gc.collect()
    assert tensor_ref() is None and view_ref() is None
    assert not cache._views


def test_capacity_bounds_live_storage_and_preserves_recent_view():
    cache = _DSAPDMooncakeLandingViews(capacity=2)
    tensor = torch.zeros(32, dtype=torch.int8)
    cache.get(tensor, 0, 4)
    first = cache.get(tensor, 0, 4)
    cache.get(tensor, 4, 4)
    second = cache.get(tensor, 4, 4)
    first_ref, second_ref = weakref.ref(first), weakref.ref(second)
    assert cache.get(tensor, 0, 4) is first
    del first, second
    cache.get(tensor, 8, 4)
    third = cache.get(tensor, 8, 4)
    assert len(cache._views) == 2
    assert first_ref() is not None and second_ref() is None
    assert cache.get(tensor, 8, 4) is third


@pytest.mark.parametrize("source", [False, True])
def test_offsets_lengths_and_replacement_tensors_are_distinct(source):
    cache = _DSAPDMooncakeLandingViews()
    first, replacement = torch.zeros(32, dtype=torch.int8), torch.ones(32, dtype=torch.int8)
    cache.get(first, 4, 8, source=source)
    a = cache.get(first, 4, 8, source=source)
    assert cache.get(first, 5, 8, source=source) is not a
    assert cache.get(first, 4, 7, source=source) is not a
    assert torch.all(cache.get(replacement, 4, 8, source=source) == 1)


def test_invalid_destination_range_is_not_cached():
    cache = _DSAPDMooncakeLandingViews()
    tensor = torch.zeros(16, dtype=torch.int8)
    with pytest.raises(RuntimeError):
        cache.get(tensor, 12, 8)
    assert not cache._views
    assert cache.get(tensor, 16, 0).numel() == 0


def test_nonpositive_cache_capacity_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        _DSAPDMooncakeLandingViews(capacity=0)


def test_prepared_landing_reuses_both_views_without_caching_payload(monkeypatch):
    reader = make_reader(MagicMock())
    target, staging = torch.zeros(64, dtype=torch.int8), torch.arange(32, dtype=torch.int8)
    original_narrow, original_getitem = torch.Tensor.narrow, torch.Tensor.__getitem__
    calls = {"narrow": 0, "slice": 0}

    def narrow(tensor, *args):
        if tensor is target:
            calls["narrow"] += 1
        return original_narrow(tensor, *args)

    def getitem(tensor, key):
        if tensor is staging:
            calls["slice"] += 1
        return original_getitem(tensor, key)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "narrow", narrow)
        patch.setattr(torch.Tensor, "__getitem__", getitem)
        for value in [3, 7, 11]:
            staging.fill_(value)
            reader._copy_dsa_pd_mooncake_staging_to_offloaded_kv(
                target.data_ptr() + 8, 12, 4, staging=staging, target_bytes=target, target_offset=8
            )
    assert calls == {"narrow": 2, "slice": 1}
    assert torch.all(target[8:20] == 11)
    assert torch.count_nonzero(target[:8]) == 0 and torch.count_nonzero(target[20:]) == 0


def test_one_off_destination_history_is_bounded_and_does_not_pin_tensors():
    cache = _DSAPDMooncakeLandingViews(capacity=2)
    tensor = torch.zeros(32, dtype=torch.int8)
    for offset in range(8):
        cache.get(tensor, offset, 4)
    assert not cache._views and len(cache._recent) == 2
    tensor_ref = weakref.ref(tensor)
    del tensor
    gc.collect()
    assert tensor_ref() is None
